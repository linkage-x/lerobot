"""Offline, width-dependent contact IK. Does not import hardware interfaces."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/recalculation_20260907_193725'
SUB = ROOT / 'third_party/opencv_kalibr'
REV = '6b4f9e7aef458b71b318909b074646c5ea33545a'
CAL = 'hikon_cube_tracking_offline/config_thor/marker_to_tcp_calibration_20260825.json'
OLD = CAL.replace('20260825', '20260812')
MODEL = ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper_v2_p0.urdf'
SEED = np.array([-.2982022354854064, -.20546837567339093, .2008775163648066,
                 -2.707162497847623, -.09350475554363503, 2.9366955831629005, .8043834376561214])
RADIUS, LINEAR = 49.699345, 5.474953


def retreat(d_m):
    d = float(d_m) * 1000
    rad = RADIUS**2 - LINEAR*d - d*d/4
    if not np.isfinite(d) or d < 0 or rad < -1e-9:
        raise ValueError(f'Invalid true opening: {d} mm')
    return (RADIUS - np.sqrt(max(0., rad))) / 1000


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def pose(row):
    t = np.eye(4)
    t[:3, 3] = [float(row[f'state_{a}_m']) for a in 'xyz']
    t[:3, :3] = Rotation.from_quat([float(row[f'state_q{a}']) for a in 'xyzw']).as_matrix()
    return t


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    # Snapshot the exact Git object; no checkout or change to production config.
    raw = subprocess.check_output(['git', '-C', str(SUB), 'show', f'{REV}:{CAL}'])
    bundle = json.loads(raw)
    old = json.loads(subprocess.check_output(['git', '-C', str(SUB), 'show', f'{REV}:{OLD}']))
    (OUT / 'marker_to_tcp_calibration_20260825.json').write_bytes(raw)
    hop = np.diag([1., -1., -1., 1.])
    delta = np.linalg.inv(np.array(old['cubes']['right']['T_cube_tcp']) @ hop) @ (np.array(bundle['cubes']['right']['T_cube_tcp']) @ hop)
    source = read_rows(OUT / 'input/state_action.right.csv')
    reference = read_rows(OUT / 'input/candidate_0825.right.csv')
    assert len(source) == len(reference) == 991
    assert retreat(0) == 0
    assert abs(retreat(.08) - .028914046026229492) < 1e-12
    try:
        retreat(.09)
    except ValueError:
        pass
    else:
        raise AssertionError('Formula domain must be checked')
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'corenetic_gripper_ee')
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'fr3_joint{i}') for i in range(1,8)]
    idx = model.jnt_qposadr[ids]
    bounds = (model.jnt_range[ids,0]+1e-5, model.jnt_range[ids,1]-1e-5)
    results, contacts, fixed, summaries = [], [], [], []
    for ep in sorted({int(r['episode_index']) for r in source}):
        prev = SEED.copy()
        ep_results = []
        for row, check in zip(source, reference):
            if int(row['episode_index']) != ep:
                continue
            fixed_t = pose(row) @ delta
            assert np.max(np.abs(fixed_t - pose(check))) < 1e-10
            d = float(row['gripper_width_m'])
            z = retreat(d)
            # New FR3 TCP +Z is the approach axis; retreat is local -Z.
            target_r = fixed_t[:3,:3]
            target_p = fixed_t[:3,3] - target_r[:,2]*z
            def residual(q):
                data.qpos[idx] = q
                mujoco.mj_kinematics(model, data)
                rot = data.xmat[body].reshape(3,3)
                contact_p = data.xpos[body] - rot[:,2]*z
                return np.r_[contact_p-target_p, Rotation.from_matrix(target_r.T @ rot).as_rotvec()*.2]
            sol = least_squares(residual, np.clip(prev, *bounds), bounds=bounds, method='dogbox',
                                max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            err = residual(sol.x)
            pos_err, rot_err = np.linalg.norm(err[:3]), np.rad2deg(np.linalg.norm(err[3:])/.2)
            ok = pos_err <= 5e-5 and rot_err <= .01
            step = float(np.max(np.abs(sol.x-prev))) if ep_results else None
            prev = sol.x.copy()
            result = dict(episode_index=ep, frame_index=int(row['frame_index']), timestamp_s=float(row['timestamp_s']),
                          gripper_width_m=d, tcp_retreat_m=z, ik_ok=bool(ok), position_error_m=float(pos_err),
                          orientation_error_deg=float(rot_err), max_joint_step_rad=step)
            result.update({f'fr3_joint{i+1}':float(q) for i,q in enumerate(sol.x)})
            result.update({f'contact_target_{a}_m':float(target_p[i]) for i,a in enumerate('xyz')})
            results.append(result); ep_results.append(result)
            for collection, p in [(fixed,fixed_t[:3,3]), (contacts,target_p)]:
                item = dict(episode_index=ep, frame_index=int(row['frame_index']), timestamp_s=float(row['timestamp_s']), gripper_width_m=d)
                item.update({f'state_{a}_m':float(p[i]) for i,a in enumerate('xyz')})
                item.update({f'state_q{a}':float(v) for a,v in zip('xyzw',Rotation.from_matrix(target_r).as_quat())})
                collection.append(item)
        report = dict(episode_index=ep, frames=len(ep_results), ik_pass=sum(r['ik_ok'] for r in ep_results),
                      q6_above_previous_panda_py_wall_count=sum(r['fr3_joint6'] > 3.7525 for r in ep_results),
                      max_position_error_mm=max(r['position_error_m'] for r in ep_results)*1000,
                      max_orientation_error_deg=max(r['orientation_error_deg'] for r in ep_results),
                      max_joint_step_rad=max(r['max_joint_step_rad'] or 0 for r in ep_results),
                      opening_range_mm=[min(r['gripper_width_m'] for r in ep_results)*1000,max(r['gripper_width_m'] for r in ep_results)*1000],
                      retreat_range_mm=[min(r['tcp_retreat_m'] for r in ep_results)*1000,max(r['tcp_retreat_m'] for r in ep_results)*1000])
        summaries.append(report)
        print(json.dumps(report), flush=True)
    write_csv(OUT / 'contact_ik.right.csv', results)
    write_csv(OUT / 'contact_targets.right.csv', contacts)
    write_csv(OUT / 'fixed_tcp_targets.right.csv', fixed)
    manifest = dict(dataset='thor_gmsl2_10ch_v1_20260903_193725', cube='right', device_id='box1819152274',
                    calibration_commit=REV, calibration_path=CAL, calibration_sha256=hashlib.sha256(raw).hexdigest(),
                    model=str(MODEL), model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                    formula=dict(radius_mm=RADIUS, linear_mm=LINEAR, max_domain_mm=2*(np.hypot(RADIUS,LINEAR)-LINEAR)),
                    input_sha256=hashlib.sha256((OUT/'input/state_action.right.csv').read_bytes()).hexdigest(),
                    target_definition='world_T_fixed_TCP @ Translation(0,0,-z(d)); same contact offset applied in robot FK',
                    source_processing='Reused existing smoothed camera tracking. Applied inv(old_cube_TCP @ Rx(pi)) @ (new_cube_TCP @ Rx(pi)). Checked against prior 0825 candidate.',
                    opening_assumption='Recorded gripper_width_m treated as true distance d, same opening for source and replay; no old spacer calibration applied.',
                    calibration_assumption='Requested 0825 closed-TCP bundle reused as-is; no unsupported 19.01mm mount transfer inferred.',
                    solver='MuJoCo FK + scipy least_squares dogbox, previous-frame seed; reset seed per episode',
                    hardware_ready=False, collision_checked=False, retimed=False,
                    limitations=['Right calibration is copied, validated=false; current V2 mount equivalence unverified.',
                                 'P0 base remains existing coarse calibration.',
                                 'Formula-based contact center does not certify physical opening or mount accuracy.',
                                 'Same source/replay opening makes dynamic retreat cancel in exact IK; does not by itself fix a wrong fixed-TCP calibration.',
                                 'Joint bounds are URDF bounds, not panda-py software walls.'], episodes=summaries)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
