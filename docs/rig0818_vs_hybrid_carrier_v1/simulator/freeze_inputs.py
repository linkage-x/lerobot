#!/usr/bin/env python3
"""Freeze the real, local inputs of the rig0818 x Hybrid Carrier V1 A/B.

Nothing in the three "read" inputs is simulated: the production 0804 fisheye
calibration, the 132514 bimanual manipulation capture, and the gripper URDF are
read from this checkout and written into ``inputs/`` together with the SHA-256
of every source file. The image runner reads only ``inputs/``.

What is *assumed* rather than read -- the BOX housing, the hand envelope, and
where a target that was never built would sit on the gripper -- is written with
``status: "assumed"`` next to its numbers, so the page shows it as an assumption
and not as a measurement.

No network, no Thor, no production service.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'third_party/opencv_kalibr'))
from metrology.calibration_io import find_intrinsics_json, load_camera_poses_in_base  # noqa: E402
from metrology.trajectory_smoothing import smooth_pose_trajectory  # noqa: E402

CAL = ROOT / 'outputs/calibration'
EXTRINSICS = CAL / 'thor_gmsl2_selfcal_0804_fisheye_extrinsics/summary.json'
INTRINSICS = CAL / 'thor_gmsl2_selfcal_0804_fisheye_intrinsics'
WORLD_REFERENCE = ROOT / 'tools/thor/gmsl2/world/world_reference.json'
TASK = ROOT / 'outputs/metrology/phase5_contact_review_132514'
TASK_BUNDLE = TASK / 'src/marker_to_tcp_calibration_20260812.left_right_tmp.json'
GRIPPER = ROOT / 'third_party/opencv_kalibr/corenetic_gripper_V2_0_description'
URDF = GRIPPER / 'urdf/gripper_V2_0_description.urdf'
R0_CAD = ROOT / 'third_party/opencv_kalibr/metrology/fixtures/cad/marker_rig_20260818_cad.json'
HYBRID = ROOT / 'third_party/opencv_kalibr/metrology/fixtures/cad/hybrid_carrier_v1_20260907.json'
CAMERAS = ['cam_06', 'cam_07', 'cam_08', 'cam_09', 'cam_12', 'cam_13', 'cam_14']
OUT = HERE / 'inputs'

# Decimated triangle budget per URDF mesh. Occluders only need their silhouette;
# the renderer pays per triangle for every ray inside a target's image window.
MESH_BUDGET = {
    'link_lt_gripper_base': 180, 'link_lt_gripper_drive': 60,
    'link_lt_gripper_finger_left': 110, 'link_lt_gripper_finger_right': 110,
    'link_lt_gripper_knuckle_left_outer': 36, 'link_lt_gripper_knuckle_left_inner': 36,
    'link_lt_gripper_knuckle_right_outer': 36, 'link_lt_gripper_knuckle_right_inner': 36,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def rounded(array, digits=7):
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def freeze_cameras() -> dict:
    poses, source = load_camera_poses_in_base(EXTRINSICS)
    world = json.loads(WORLD_REFERENCE.read_text())
    files = {rel(EXTRINSICS): sha256(EXTRINSICS), rel(WORLD_REFERENCE): sha256(WORLD_REFERENCE)}
    cameras = []
    for name in CAMERAS:
        T = np.asarray(poses[name], dtype=float)
        if not np.allclose(T, np.asarray(world['cameras'][name]['T_world_camera']), atol=1e-12):
            raise SystemExit(f'{name}: 0804 extrinsics disagree with the frozen world reference')
        path = find_intrinsics_json(INTRINSICS, name)
        intrinsics = json.loads(path.read_text())
        if intrinsics.get('model') != 'opencv_fisheye':
            raise SystemExit(f'{name}: expected an opencv_fisheye intrinsics file, got {intrinsics.get("model")!r}')
        files[rel(path)] = sha256(path)
        cameras.append({
            'name': name, 'serial': intrinsics['camera_serial'], 'model': 'fisheye',
            'width': int(intrinsics['image_width']), 'height': int(intrinsics['image_height']),
            'K': intrinsics['camera_matrix'], 'D': np.ravel(intrinsics['dist_coeffs']).tolist(),
            'T_world_camera': T.tolist(), 'sigma_pixel': 0.2,
            'heldout_rmse_px': intrinsics['self_calibration']['heldout_time_block_rmse_px'],
        })
    return {
        'schema': 'rig_target_ab_cameras/v1', 'status': 'frozen-real',
        'world_frame_id': world['world_frame_id'], 'calibration_id': world['calibration_id'],
        'extrinsics_source': source,
        'note': ('Seven calibrated Thor GMSL2 cameras, 0804 fisheye self-calibration: the pairing the 132514 '
                 'trajectory was itself tracked with. Production extrinsics have since moved to '
                 'calib_20260902_103833 (Thor only). In simulation this snapshot is the truth camera AND the '
                 'nominal estimator camera, so staleness is not an error source; calibration error enters only '
                 'as the declared L3 perturbation.'),
        'sources': files, 'cameras': cameras,
    }


def _pose_matrix(position, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def _read_hand(hand: str):
    path = TASK / f'ba_markers/{hand}/marker_rig_ba.{hand}.csv'
    episode_path = TASK / f'src/{hand}_full/marker_rig_ba.{hand}.with_episode.csv'
    rows = list(csv.DictReader(path.open()))
    episodes = {int(r['frame_index']): int(r['episode_index']) for r in csv.DictReader(episode_path.open())}
    frame = np.array([int(r['frame_index']) for r in rows])
    if not np.array_equal(frame, np.arange(len(rows))) or set(episodes) != set(frame.tolist()):
        raise SystemExit(f'{hand}: frame index / episode mapping is not one row per frame')
    episode = np.array([episodes[i] for i in frame])
    position = np.array([[float(r[f'rig_base_{a}_m']) for a in 'xyz'] for r in rows])
    quat = np.array([[float(r[f'rig_base_q{a}']) for a in 'xyzw'] for r in rows])
    rmse = np.array([float(r['reprojection_rmse_px'] or 'nan') for r in rows])
    success = np.array([r['success'] == '1' for r in rows])
    return path, episode_path, episode, position, quat, rmse, success


def freeze_task() -> dict:
    bundle = json.loads(TASK_BUNDLE.read_text())
    files = {rel(TASK_BUNDLE): sha256(TASK_BUNDLE)}
    manifest_frames = []
    for e in range(3):
        manifest = TASK / f'dataset/episodes/episode_{e:06d}/online_sync_manifest.json'
        files[rel(manifest)] = sha256(manifest)
        doc = json.loads(manifest.read_text())
        if int(doc['fps']) != 60:
            raise SystemExit(f'{manifest}: expected 60 fps')
        manifest_frames.append(int(doc['actual_frames']))
    T_box_tcp = urdf_box_tcp()
    hands = {}
    for hand in ('left', 'right'):
        path, episode_path, episode, position, quat, rmse, success = _read_hand(hand)
        files[rel(path)] = sha256(path)
        files[rel(episode_path)] = sha256(episode_path)
        counts = [int(np.sum(episode == e)) for e in range(3)]
        if counts != manifest_frames:
            raise SystemExit(f'{hand}: per-episode row counts {counts} != manifest frames {manifest_frames}')
        frame_in_episode = np.concatenate([np.arange(c) for c in counts])
        t = frame_in_episode / 60.0
        finite = np.isfinite(position).all(1) & np.isfinite(quat).all(1)
        observed = success & finite & (np.nan_to_num(rmse, nan=np.inf) <= 6.0)
        # Robust pre-filter: a BA answer that disagrees with its own +/-5 frame
        # neighbourhood by >15 mm or >8 deg is a wrong basin (camera-set switch),
        # not motion a hand makes in 83 ms.
        rejected_jump = 0
        keep = observed.copy()
        for i in np.flatnonzero(observed):
            window = [j for j in range(max(0, i - 5), min(len(keep), i + 6))
                      if j != i and observed[j] and episode[j] == episode[i]]
            if len(window) < 3:
                continue
            if np.linalg.norm(position[i] - np.median(position[window], axis=0)) > 0.015:
                keep[i] = False
                rejected_jump += 1
                continue
            mean = Rotation.from_quat(quat[window]).mean()
            if (mean.inv() * Rotation.from_quat(quat[i])).magnitude() > np.deg2rad(8.0):
                keep[i] = False
                rejected_jump += 1
        idx = np.flatnonzero(keep)
        pose7 = np.column_stack([position[idx], quat[idx]])
        smoothed = smooth_pose_trajectory(
            pose7, t[idx], episode_indices=episode[idx], measurement_sigma_m=5e-4,
            measurement_sigma_rad=np.deg2rad(0.3), velocity_change_sigma_mps=0.03,
            velocity_change_sigma_radps=0.5, loss='soft_l1', f_scale=2.0, max_nfev=50)
        truth_p = np.full((len(t), 3), np.nan)
        truth_q = np.full((len(t), 4), np.nan)
        truth_p[idx], truth_q[idx] = smoothed.pose7[:, :3], smoothed.pose7[:, 3:7]
        interpolated = np.zeros(len(t), bool)
        requested = np.zeros(len(t), bool)
        for e in range(3):
            members = np.flatnonzero(episode == e)
            known = [i for i in members if keep[i]]
            if len(known) < 2:
                continue
            first, last = known[0], known[-1]
            slerp = Slerp(t[known], Rotation.from_quat(truth_q[known]))
            for i in members:
                if i < first or i > last:
                    continue  # no truth outside the observed span: not requested
                requested[i] = True
                if not keep[i]:
                    truth_p[i] = [np.interp(t[i], t[known], truth_p[known][:, k]) for k in range(3)]
                    truth_q[i] = slerp([t[i]]).as_quat()[0]
                    interpolated[i] = True
        T_cube_tcp = np.asarray(bundle['cubes'][hand]['T_cube_tcp'], dtype=float)
        T_cube_box = T_cube_tcp @ np.linalg.inv(T_box_tcp)
        R_mount = np.asarray(bundle['cubes'][hand]['mount']['R_cube_box'], dtype=float)
        if not np.allclose(T_cube_box[:3, :3], R_mount, atol=1e-6):
            raise SystemExit(f'{hand}: URDF TCP chain and the bundle mount rotation disagree')
        tcp = np.array([(_pose_matrix(p, q) @ T_cube_tcp)[:3, 3] if np.isfinite(p).all() else [np.nan] * 3
                        for p, q in zip(truth_p, truth_q)])
        speed = []
        for e in range(3):
            members = np.flatnonzero((episode == e) & requested)
            speed.extend((np.linalg.norm(np.diff(tcp[members], axis=0), axis=1) * 60).tolist())
        hands[hand] = {
            'frames': int(len(t)), 'episode': episode.tolist(), 'frame_in_episode': frame_in_episode.tolist(),
            't_s': rounded(t, 6), 'requested': requested.astype(int).tolist(),
            'observed': keep.astype(int).tolist(), 'interpolated': interpolated.astype(int).tolist(),
            'T_world_cube_position_m': [None if not np.isfinite(p).all() else rounded(p) for p in truth_p],
            'T_world_cube_quat_xyzw': [None if not np.isfinite(q).all() else rounded(q) for q in truth_q],
            'T_cube_tcp': rounded(T_cube_tcp, 9), 'T_cube_box': rounded(T_cube_box, 9),
            'stats': {
                'ba_success': int(success.sum()), 'kept_after_prefilter': int(keep.sum()),
                'rejected_as_jump': int(rejected_jump), 'requested': int(requested.sum()),
                'interpolated': int(interpolated.sum()),
                'tcp_speed_mps_p50_p95_p99_max': rounded(np.percentile(speed, [50, 95, 99, 100]), 4),
                'smoother_position_shift_mm_p95': float(np.percentile(smoothed.position_shift_m, 95) * 1e3),
            },
        }
    return {
        'schema': 'rig_target_ab_task/v1', 'status': 'frozen-real-task',
        'dataset': 'thor_gmsl2_9ch_v1_20260817_132514', 'fps': 60, 'world_frame_id': 'world_20260819_031843',
        'truth_definition': ('Task-distribution proxy, not a GT: the recorded 0812-cube corner-BA poses, '
                             'pre-filtered for wrong basins and passed through the production offline SE(3) '
                             'smoother. Interior gaps are SE(3)-interpolated and flagged; frames outside an '
                             "episode's observed span are not requested. Neither simulated target produced this "
                             'trajectory, so it favours neither arm; its gaps sit where the cube failed.'),
        'dev_episodes': [0], 'heldout_episodes': [1, 2], 'sources': files, 'hands': hands,
    }


def _urdf_origin(element):
    origin = element.find('origin')
    xyz = np.array([float(v) for v in origin.get('xyz', '0 0 0').split()])
    rpy = np.array([float(v) for v in origin.get('rpy', '0 0 0').split()])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()  # URDF fixed-axis roll/pitch/yaw
    T[:3, 3] = xyz
    return T


def urdf_link_poses(q: float) -> dict[str, np.ndarray]:
    root = ET.parse(URDF).getroot()
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    cache = {'link_box': np.eye(4)}

    def pose(link):
        if link not in cache:
            joint = joints[link]
            T = pose(joint.find('parent').get('link')) @ _urdf_origin(joint)
            if joint.get('type') == 'revolute':
                axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
                J = np.eye(4)
                J[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                T = T @ J
            cache[link] = T
        return cache[link]

    return {link.get('name'): pose(link.get('name')) for link in root.findall('link')}


def urdf_box_tcp() -> np.ndarray:
    return urdf_link_poses(0.0)['link_lt_gripper_tcp']


def _cuboid(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [[v[a], v[b], v[c]] for a, b, c, d in faces] + [[v[a], v[c], v[d]] for a, b, c, d in faces]


def freeze_gripper(opening_rad: float) -> dict:
    import open3d as o3d

    poses = urdf_link_poses(opening_rad)
    root = ET.parse(URDF).getroot()
    files = {rel(URDF): sha256(URDF)}
    triangles, links = [], []
    for link in root.findall('link'):
        name = link.get('name')
        mesh = link.find('visual/geometry/mesh')
        if mesh is None:
            continue
        path = URDF.parent / mesh.get('filename')
        if not path.exists():
            path = GRIPPER / mesh.get('filename')
        files[rel(path)] = sha256(path)
        raw = o3d.io.read_triangle_mesh(str(path))
        raw.remove_duplicated_vertices()
        reduced = raw.simplify_quadric_decimation(MESH_BUDGET[name])
        vertices = np.asarray(reduced.vertices)
        T = poses[name] @ _urdf_origin(link.find('visual'))
        world = vertices @ T[:3, :3].T + T[:3, 3]
        tris = world[np.asarray(reduced.triangles)]
        triangles.extend(tris.tolist())
        links.append({'link': name, 'source_triangles': len(raw.triangles), 'kept_triangles': len(tris)})
    # BOX housing: the URDF has no mesh for link_box. The cube bundle puts the
    # cube centre 40 mm above the box origin, the finger base 70 mm below it on
    # the front face, so the housing is drawn as the smallest box that holds
    # both. Its size is a declared assumption, not a measurement.
    box_lo, box_hi = [-0.040, -0.130, -0.100], [0.040, 0.030, 0.0]
    housing = _cuboid(box_lo, box_hi)
    # Handle under the housing and the operator's hand around it, as spheres.
    handle = [{'c': [0.0, -0.06, -0.10 - 0.02 * k], 'r': 0.021, 'part': 'handle'} for k in range(6)]
    hand = ([{'c': [0.0, -0.06 + 0.028 * np.cos(a), -0.14 + 0.028 * np.sin(a)], 'r': 0.024, 'part': 'hand'}
             for a in np.linspace(0.3, 2.8, 5)]
            + [{'c': [0.0, -0.105 - 0.05 * k, -0.19 - 0.035 * k], 'r': 0.036 + 0.004 * k, 'part': 'forearm'}
               for k in range(4)])
    T_box_tcp = poses['link_lt_gripper_tcp']
    return {
        'schema': 'rig_target_ab_gripper/v1', 'status': 'urdf-mechanism+assumed-housing-and-hand',
        'units': 'm', 'frame': 'link_box of gripper_V2_0_description', 'opening_rad': opening_rad,
        'sources': files, 'T_box_tcp': rounded(T_box_tcp, 9),
        'mechanism': {'status': 'urdf', 'triangles': rounded(triangles, 6), 'links': links},
        'housing': {'status': 'assumed', 'min_m': box_lo, 'max_m': box_hi, 'triangles': rounded(housing, 6)},
        'hand': {'status': 'assumed', 'spheres': [dict(s, c=rounded(s['c'], 5)) for s in handle + hand],
                 'note': 'pistol-grip handle under the housing, a palm wrapped round it and a forearm running back'},
        'mount': {
            'status': 'assumed',
            'gripper': {
                'socket_xy_box_m': [0.0053, -0.0493], 'housing_top_z_m': 0.0, 'clearance_m': 0.005,
                'z_axis': '+z_box before tilt (socket seated on a stud, opening toward the housing)',
                'yaw_candidates_deg': [0, 90, 180, 270],
                'pitch_candidates_deg': [0, -25],
                'rule': ('Both targets seat their CAD socket origin on the same stud, directly over the point the '
                         "cube's centre occupies, with the lowest point of the rotated target 5 mm above the housing. "
                         'The free installation choices are yaw about the box z axis and a boom tilt about the '
                         "target's own y axis; the same candidate set is searched for each arm on the dev episode "
                         'and the choice is frozen before the held-out episodes are rendered.'),
            },
            'socket_tcp': {'rule': 'Existing-structure control: the CAD socket origin is the TCP, no gripper.'},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opening-rad', type=float, default=0.35)
    parser.add_argument('--r0-layout', type=Path,
                        help='measured rig0818 layout in the CAD frame; copied byte for byte to inputs/r0_layout_measured.json')
    parser.add_argument('--r0-layout-origin', default='', help='where that file came from, recorded in the manifest')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    measured_target = OUT / 'r0_layout_measured.json'
    if args.r0_layout:
        from ab_scene import r0_measured_layout
        raw = args.r0_layout.read_bytes()
        r0_measured_layout(json.loads(raw), json.loads(R0_CAD.read_text()))  # refuses a wrong frame, id set or winding
        measured_target.write_bytes(raw)
        print(f'wrote inputs/{measured_target.name} from {args.r0_layout_origin or args.r0_layout}')
    outputs = {
        'cameras_0804_fisheye.json': freeze_cameras(),
        'task_132514.json': freeze_task(),
        'gripper_v2.json': freeze_gripper(args.opening_rad),
    }
    for name, doc in outputs.items():
        (OUT / name).write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        print(f'wrote inputs/{name} ({(OUT / name).stat().st_size / 1024:.0f} KiB)')
    # The manifest names the exact bytes the runner reads, next to the files they came from.
    manifest_path = HERE / 'inputs_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema'] = 'rig_target_ab_inputs/v2'
    manifest['repository_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest['metrology_revision'] = subprocess.check_output(
        ['git', '-C', str(ROOT / 'third_party/opencv_kalibr'), 'rev-parse', 'HEAD'], text=True).strip()
    manifest['coordinate_contract']['world_frame_id'] = outputs['cameras_0804_fisheye.json']['world_frame_id']
    manifest['real_inputs'] = {
        name: {'path': f'inputs/{name}', 'sha256': sha256(OUT / name), 'status': doc['status'], 'sources': doc['sources']}
        for name, doc in outputs.items()}
    if measured_target.exists():
        layout = json.loads(measured_target.read_text())
        previous = manifest['targets']['R0'].get('measured_layout') or {}
        origin = args.r0_layout_origin or previous.get('origin') or ''
        entry = {'path': f'inputs/{measured_target.name}', 'sha256': sha256(measured_target),
                 'layout_id': layout.get('layout_id'), 'origin': origin,
                 'frame_convention': layout.get('frame_convention'), 'generated_utc': layout.get('generated_utc')}
        manifest['targets']['R0'].update(geometry_status='measured', measured_layout=entry, geometry_note=(
            'The real-input runner uses the measured layout for both the rendered truth and the solver; the CAD '
            'descriptor is kept for plate assignment and normals checks. The browser sandbox still starts from the '
            'CAD proxy unless the layout is loaded.'))
        manifest['real_inputs'][measured_target.name] = {'path': entry['path'], 'sha256': entry['sha256'],
                                                          'status': 'measured', 'sources': {origin: entry['sha256']}}
    manifest['cameras']['scope'] = 'MC-lite sandbox only; the real-input runner reads real_inputs'
    manifest['trajectory']['scope'] = 'MC-lite sandbox only; the real-input runner reads real_inputs'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('updated inputs_manifest.json')
    for hand, doc in outputs['task_132514.json']['hands'].items():
        print(hand, doc['stats'])


if __name__ == '__main__':
    main()
