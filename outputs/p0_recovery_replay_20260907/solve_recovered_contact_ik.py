"""Re-solve all frames from newly reconstructed visual TCPs. Offline only.

No waypoint displacement, frame dropping or relaxed software walls. Seven arm
joint angles may change while each requested pose stays fixed. The source and
robot contact offsets use the SAME measured width and supplied retreat formula.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/fr3'))
from recalculate_contact_tcp_20260907 import SEED, retreat, pose
from replay_ik_trajectory_guarded import LOWER, UPPER, WALL_BAND


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'contact_ik.right.csv'
    if output.exists():
        raise FileExistsError(output)
    source_path = HERE / 'tracking_sidecar/state_action.right.csv'
    width_path = ROOT / 'outputs/recalculation_20260907_193725/input/state_action.right.csv'
    source, widths = read(source_path), read(width_path)
    assert len(source) == len(widths) == 991
    model_path = HERE / 'model/fr3_v2_p0_recovered.urdf'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'corenetic_gripper_ee')
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'fr3_joint{i}') for i in range(1, 8)]
    idx = model.jnt_qposadr[joint_ids]
    bounds = (np.maximum(model.jnt_range[joint_ids, 0], LOWER + WALL_BAND) + 1e-5,
              np.minimum(model.jnt_range[joint_ids, 1], UPPER - WALL_BAND) - 1e-5)
    grip = [i for i in range(model.njnt) if 'gripper' in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)]
    gi = model.jnt_qposadr[grip]
    cl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link_gripper_contact_left')
    cr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link_gripper_contact_right')
    assert len(grip) == 7 and min(body, cl, cr) >= 0
    upper_drive = float(np.min(model.jnt_range[grip, 1]))
    def opening(a):
        data.qpos[gi] = a
        mujoco.mj_kinematics(model, data)
        return float(np.linalg.norm(data.xpos[cl] - data.xpos[cr]))
    results, contacts, fixed, episodes = [], [], [], []
    for ep in [0, 1]:
        prev = np.clip(SEED.copy(), *bounds)
        ep_rows = []
        for row, width in zip(source, widths):
            for key in ['episode_index', 'frame_index']:
                assert int(row[key]) == int(width[key]), (key, row[key], width[key])
            assert abs(float(row['timestamp_s']) - float(width['timestamp_s'])) < 1e-6
            if int(row['episode_index']) != ep:
                continue
            T = pose(row)
            d = float(width['gripper_width_m'])
            z = retreat(d)
            target_r = T[:3, :3]
            target_p = T[:3, 3] - target_r[:, 2] * z
            if not np.isfinite(T).all() or abs(np.linalg.det(target_r) - 1) > 1e-8:
                raise ValueError('Invalid reconstructed pose')
            def residual(q):
                data.qpos[idx] = q
                mujoco.mj_kinematics(model, data)
                r = data.xmat[body].reshape(3, 3)
                p = data.xpos[body] - r[:, 2] * z
                return np.r_[p - target_p, Rotation.from_matrix(target_r.T @ r).as_rotvec() * .2]
            solution = least_squares(residual, prev, bounds=bounds, method='dogbox',
                                     max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            err = residual(solution.x)
            pe = float(np.linalg.norm(err[:3]))
            re = float(np.degrees(np.linalg.norm(err[3:]) / .2))
            ok = pe <= 5e-5 and re <= .01
            angle = brentq(lambda a: opening(a) - d, 0, upper_drive, xtol=1e-12)
            opening(angle)
            actual_contact = (data.xpos[cl] + data.xpos[cr]) / 2
            contact_error = float(np.linalg.norm(actual_contact - target_p))
            item = dict(episode_index=ep, frame_index=int(row['frame_index']), timestamp_s=float(row['timestamp_s']),
                        gripper_width_m=d, tcp_retreat_m=z, ik_ok=ok, position_error_m=pe,
                        orientation_error_deg=re, max_joint_step_rad=float(np.max(np.abs(solution.x-prev))) if ep_rows else 0.,
                        actual_urdf_contact_error_m=contact_error, gripper_drive_rad=angle,
                        min_software_wall_margin_rad=float(np.min(np.minimum(solution.x - LOWER, UPPER - solution.x) - WALL_BAND)))
            item.update({f'fr3_joint{i+1}': float(q) for i, q in enumerate(solution.x)})
            item.update({f'contact_target_{a}_m': float(target_p[i]) for i, a in enumerate('xyz')})
            results.append(item)
            ep_rows.append(item)
            prev = solution.x.copy()
            for target_list, pos in [(contacts, target_p), (fixed, T[:3, 3])]:
                trow = {k: item[k] for k in ['episode_index', 'frame_index', 'timestamp_s', 'gripper_width_m']}
                trow.update({f'state_{a}_m': float(pos[i]) for i, a in enumerate('xyz')})
                trow.update({f'state_q{a}': float(x) for a, x in zip('xyzw', Rotation.from_matrix(target_r).as_quat())})
                target_list.append(trow)
        q = np.asarray([[r[f'fr3_joint{i}'] for i in range(1, 8)] for r in ep_rows])
        t = np.asarray([r['timestamp_s'] for r in ep_rows])
        assert len(ep_rows) == [571, 420][ep] and np.all(np.diff(t) > 0)
        dt = float(np.median(np.diff(t)))
        report = dict(episode=ep, frames=len(ep_rows), ik_pass=sum(r['ik_ok'] for r in ep_rows),
                      max_position_error_mm=max(r['position_error_m'] for r in ep_rows)*1000,
                      max_orientation_error_deg=max(r['orientation_error_deg'] for r in ep_rows),
                      max_actual_urdf_contact_error_mm=max(r['actual_urdf_contact_error_m'] for r in ep_rows)*1000,
                      max_joint_step_rad=max(r['max_joint_step_rad'] for r in ep_rows),
                      min_software_wall_margin_rad=min(r['min_software_wall_margin_rad'] for r in ep_rows),
                      recorded_sample_period_s=dt, first_joints_rad=q[0].tolist(),
                      max_joint_velocity_recorded_rad_s=np.max(np.abs(np.diff(q, axis=0)/np.diff(t)[:,None]),axis=0).tolist(),
                      max_joint_acceleration_uniform_dt_rad_s2=np.max(np.abs(np.diff(q,n=2,axis=0)/dt**2),axis=0).tolist(),
                      first_contact_world_m=[ep_rows[0][f'contact_target_{a}_m'] for a in 'xyz'])
        episodes.append(report)
        print(json.dumps(report), flush=True)
    save(output, results)
    save(HERE / 'contact_targets.right.csv', contacts)
    save(HERE / 'fixed_tcp_targets.right.csv', fixed)
    manifest = dict(hardware_ready=False, production_modified=False,
                    model=str(model_path), model_sha256=digest(model_path),
                    new_visual_source=str(source_path), new_visual_sha256=digest(source_path),
                    width_source=str(width_path), width_source_sha256=digest(width_path),
                    width_join='Strict episode/frame/timestamp identity across all 991 rows; reuse recorded right BOX widths only, not old poses.',
                    formula='z_mm=49.699345-sqrt(49.699345**2-5.474953*d_mm-d_mm**2/4)',
                    target_definition='world_T_closed_TCP @ Trans(0,0,-z); same width-dependent offset in robot FK',
                    solver='MuJoCo FK + scipy least_squares dogbox, continuous previous-frame seed, independent reset per episode',
                    bounds='Intersection of URDF bounds with FR3 5.8.1 verified software-wall-safe interior; no wall weakening.',
                    task_poses_displaced=False, collision_constrained_ik=False, retimed=False,
                    width_assumption='Existing width values treated as true fingertip gap as in previous V2 reconstruction; no V1 spacer conversion.',
                    episodes=episodes)
    with (HERE / 'ik_manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
