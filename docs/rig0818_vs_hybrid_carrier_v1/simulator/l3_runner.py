#!/usr/bin/env python3
"""Real-input image loop for the rig0818 x Hybrid Carrier V1 A/B (L2 + L3).

Inputs are the frozen files in ``inputs/`` (0804 fisheye cameras, the 132514
bimanual task, the gripper URDF). Every stage is cached under
``outputs/rig_target_ab/<run>/`` and resumable:

  select       dev episode 0: every yaw x boom-tilt mount rendered, detected and solved;
               each target's mount is frozen before a held-out frame is rendered
  render       held-out frames: exposure-integrated fisheye images, ArUco decode
  native       per hand x episode, in frame order: the production
               HybridCarrierTracker (seeded from the previous joint pose) and
               carrier_frame_pose
  solve        per frame x session: production corner BA for R0/H0 and
               carrier_frame_pose for H1, the estimator perturbed per session
  smooth       production offline SE(3) smoother per session x arm x hand x episode
  convergence  supersample 2 vs 4 on the same frames
  report       paired statistics, bootstrap, buckets, gates -> l3_report.json

Truth (what images are rendered from) and estimator (what the solver is told)
are separate objects; an L3 session perturbs only the estimator, with the same
draw for every arm. No Thor, network or production service is touched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ab_scene as S  # noqa: E402  (puts opencv_kalibr on sys.path)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import hybrid_carrier_tracking_in_robot_base as hc  # noqa: E402
from metrology.calibration_io import CameraCalibration  # noqa: E402
from metrology.hybrid_carrier import build_aruco_detector, project_rig  # noqa: E402
from metrology.hybrid_carrier_render import PaintPalette  # noqa: E402
from metrology.marker_rig_ba import MarkerLayout, estimate_frame_pose, observations_from_detections  # noqa: E402
from metrology.trajectory_smoothing import smooth_pose_trajectory  # noqa: E402

HANDS = ('left', 'right')
OTHER = {'left': 'right', 'right': 'left'}
GROUP_CODE = {'gripper': 1, 'socket_tcp': 2}
TARGET_CODE = {'R0': 1, 'H': 2}
HAND_CODE = {'left': 1, 'right': 2}
LEVEL_CODE = {'nominal': 0, 'evidence': 1, 'stress': 2}
LEVEL_SCALE = {'nominal': 0.0, 'evidence': 1.0, 'stress': 2.0}

# Production values, from config_thor/hybrid_carrier_tracking_in_robot_base_thor.yaml.
CARRIER_CFG = {'min_anchors': 2, 'seed_from_previous_frame': True, 'seed_band_px': 5.0, 'roi_margin_px': 48.0,
               'refine_with_facets': True, 'min_edge_samples': 12, 'max_edge_refinement_mm': 12.0}
BA_KW = {'min_cameras': 2, 'min_corners': 8, 'huber_px': 2.0, 'max_nfev': 40, 'max_camera_rmse_px': 8.0}
SMOOTH_CFG = {'velocity_change_sigma_mps': 0.03, 'velocity_change_sigma_radps': 0.5, 'loss': 'soft_l1',
              'f_scale': 2.0, 'max_nfev': 50, 'min_measurement_sigma_m': 1e-4, 'min_measurement_sigma_deg': 1e-2}

# One-sigma estimator perturbations of the ``evidence`` level; ``stress`` doubles them.
# Every arm receives the same draw. Evidence strings are copied into the report.
PERTURBATIONS = {
    'cam_rot_deg': (0.06, 'per axis, per camera, per session',
                    'measured: 0820->0902 quiet drift, median relative orientation change 0.091 deg over 21 baselines'),
    'cam_trans_mm': (0.20, 'per axis, per camera, per session',
                     'measured: same comparison, median baseline length change 0.23 mm'),
    'focal_ppm': (400.0, 'per camera, per session',
                  'measured: two independent intrinsics sweeps differ by -796 ppm in scale (intrinsics_sensitivity_0824)'),
    'principal_px': (0.5, 'per axis, per camera, per session',
                     'scenario: held-out self-calibration RMSE 0.16 px bounds the fit, not the sweep-to-sweep shift'),
    'marker_trans_mm': (0.30, 'per axis, per marker, per session',
                        'measured: 0818 layout |p_rig| 1.03 vs 0.90 mm between two captures (0.48 mm apart)'),
    'marker_rot_deg': (0.30, 'per axis, per marker, per session', 'scenario: sticker/plate seating'),
    'body_scale_ppm': (500.0, 'per target, per session',
                       'scenario: residual after measuring sticker size (47.24 vs 48 mm was a corrected 1.6% error)'),
    'tcp_trans_mm': (0.50, 'per axis, per session, same TCP-frame draw for every arm',
                     'measured: 0818 pivot |p_rig| 0.90-1.03 mm, the whole chain'),
    'tcp_rot_deg': (0.15, 'per axis, per session, same TCP-frame draw for every arm',
                    'scenario: a single-pivot bundle cannot observe rotation; the axes come from CAD convention'),
    'timing_ms': (1.5, 'per session, label time offset',
                  'measured: Camera<->BOX offset spread over 0817 episodes 0.20 ms (left) / 2.58 ms (right)'),
}

SCENE = None
TRUTH = None
RENDERER = None
DETECTOR = None
CFG = None


def _init_worker(cfg: dict) -> None:
    global SCENE, TRUTH, RENDERER, DETECTOR, CFG
    cv2.setNumThreads(1)
    CFG = cfg
    SCENE = S.load_scene()
    TRUTH = {hand: S.hand_truth(SCENE, hand) for hand in HANDS}
    RENDERER = S.SceneRenderer(cfg['supersample'])
    DETECTOR = build_aruco_detector('DICT_6X6_50')


# --- small helpers ---------------------------------------------------------------

def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load(path: Path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def mount_tag(mount):
    return f'y{int(mount[0])}p{int(mount[1])}'


def render_path(group, target, mount, hand, idx, supersample=None):
    tag = '' if supersample in (None, CFG['supersample']) else f'_ss{supersample}'
    return Path(CFG['run']) / f'render{tag}' / group / f'{target}_{mount_tag(mount)}' / f'{hand}_{idx:05d}.pkl'


def native_path(group, mount, hand, idx):
    return Path(CFG['run']) / 'native' / group / f'H_{mount_tag(mount)}' / f'{hand}_{idx:05d}.pkl'


def solve_path(group, level, session, tag=''):
    return Path(CFG['run']) / 'solve' / f'{group}{tag}' / f'{level}_{session:03d}.pkl'


def smooth_path(group, level, session):
    return Path(CFG['run']) / 'smooth' / group / f'{level}_{session:03d}.pkl'


class WorkerPool:
    """A spawn process pool that can be rebuilt after a worker is killed.

    Every task is idempotent (renders and native blocks skip or rewrite whole
    files atomically; solves return values), so after a dead worker the pool is
    rebuilt and whatever had not come back is simply submitted again.
    """

    def __init__(self, workers: int, cfg: dict):
        self.workers, self.cfg = workers, cfg
        self.executor = self._make()

    def _make(self):
        return ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_init_worker, initargs=(self.cfg,))

    def submit(self, fn, job):
        return self.executor.submit(fn, job)

    def restart(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self._make()

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def run_jobs(pool, fn, jobs, label, retries=3):
    started = time.time()
    results = [None] * len(jobs)
    pending = set(range(len(jobs)))
    step = max(1, len(jobs) // 20)
    failures = 0
    while pending:
        futures = {pool.submit(fn, jobs[i]): i for i in sorted(pending)}
        try:
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                pending.discard(i)
                done = len(jobs) - len(pending)
                if done % step == 0 or not pending:
                    print(f'[{label}] {done}/{len(jobs)} jobs, {time.time() - started:.0f}s', flush=True)
        except BrokenProcessPool:
            failures += 1
            if failures > retries:
                raise
            print(f'[{label}] a worker died with {len(pending)} jobs outstanding; rebuilding the pool '
                  f'({failures}/{retries})', flush=True)
            pool.restart()
    return results


def frame_keys(episodes, stride, max_per_block=0):
    keys = []
    for hand in HANDS:
        per_block = {}
        for idx, truth in sorted(TRUTH[hand].items()):
            if truth.episode not in episodes or truth.frame % stride:
                continue
            block = per_block.setdefault(truth.episode, [])
            if max_per_block and len(block) >= max_per_block:
                continue
            block.append(idx)
            keys.append((hand, idx))
    return keys


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def pose7(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])


def rotation_error_deg(Ra, Rb):
    return float(np.degrees(Rotation.from_matrix(Ra @ Rb.T).magnitude()))


# --- rendering and native tracking -------------------------------------------------------

def task_render(job):
    group, target, mount, hand, idx, keep_png, supersample = job
    path = render_path(group, target, mount, hand, idx, supersample)
    if path.exists():
        return str(path)
    truth, other = TRUTH[hand][idx], TRUTH[OTHER[hand]].get(idx)
    model = SCENE.models[target]
    T_box_rig = S.mount_T_box_rig(SCENE, target, group, *mount)
    n, exposure = CFG['exposure_samples'], CFG['exposure_ms'] * 1e-3
    offsets = [0.0] if n == 1 else list(np.linspace(-exposure / 2, exposure / 2, n))
    poses = [truth.box_at(dt) @ T_box_rig for dt in offsets]
    occluders = [S.frame_occluders(SCENE, group, truth.box_at(dt), other.box_at(dt) if other else None)
                 for dt in offsets]
    renderer = RENDERER if supersample == CFG['supersample'] else S.SceneRenderer(supersample)
    rng = np.random.default_rng([CFG['seed'], GROUP_CODE[group], TARGET_CODE[target], int(mount[0]),
                                 int(mount[1]) + 180, HAND_CODE[hand], idx])
    cameras = []
    for camera in SCENE.cameras:
        image, box, stats = renderer.render_camera(model, camera, poses, occluders, SCENE.body_colour(target),
                                                   blur_sigma=CFG['blur_sigma'], noise_sigma=CFG['noise_sigma'], rng=rng)
        entry = {'name': camera.name, 'box': box, **stats, 'detections': [], 'png': None}
        if image is not None:
            corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                entry['detections'] = [{'marker_id': int(i), 'points_2d': np.asarray(c, np.float64).reshape(4, 2)}
                                       for c, i in zip(corners, ids.ravel())]
            if keep_png:
                x0, y0, x1, y1 = box
                entry['png'] = cv2.imencode('.png', image[y0:y1, x0:x1])[1].tobytes()
        cameras.append(entry)
    _dump(path, {'group': group, 'target': target, 'mount': list(mount), 'hand': hand, 'idx': idx,
                 'other_hand_present': other is not None, 'cameras': cameras})
    return str(path)


def _full_image(camera, entry):
    image = np.full((camera.height, camera.width, 3), PaintPalette().background, np.uint8)
    x0, y0, x1, y1 = entry['box']
    image[y0:y1, x0:x1] = cv2.imdecode(np.frombuffer(entry['png'], np.uint8), cv2.IMREAD_COLOR)
    return image


def _layout(target, perturbation=None):
    markers = SCENE.layouts[target]['markers']
    scale = 1.0 + (perturbation['body_scale_ppm'] * 1e-6 if perturbation else 0.0)
    corners = {}
    for k, marker in enumerate(markers):
        c = np.asarray(marker['corners_rig'], dtype=np.float64)
        if perturbation:
            dt, dr = perturbation['markers'][k]
            centre = c.mean(axis=0)
            c = centre + (c - centre) @ Rotation.from_rotvec(np.deg2rad(dr)).as_matrix().T + np.asarray(dt) * 1e-3
        corners[int(marker['id'])] = c * scale
    return MarkerLayout(f'{target}_estimator', corners)


def _observations(record, layout, cameras):
    observations = []
    for camera, entry in zip(cameras, record['cameras']):
        observations += observations_from_detections(camera_name=camera.name, raw_detections=entry['detections'],
                                                     layout=layout)
    return observations


def task_native(job):
    group, mount, hand, indices = job
    if all(native_path(group, mount, hand, idx).exists() for idx in indices):
        return len(indices)
    strategy = hc.HybridCarrierRigStrategy()
    strategy.configure({'carrier': dict(CARRIER_CFG, model_path=str(S.HYBRID_DESCRIPTOR))}, scripts_dir=S.HIKON)
    trackers = {camera.name: hc.HybridCarrierTracker(strategy=strategy, camera_matrix=camera.K,
                                                     dist_coeffs=camera.D, aruco_dict_name='DICT_6X6_50',
                                                     camera_model='fisheye', camera_name=camera.name)
                for camera in SCENE.cameras}
    layout = _layout('H')
    for idx in indices:  # in frame order: the seed is the previous frame's joint pose
        record = _load(render_path(group, 'H', mount, hand, idx))
        started = time.perf_counter()
        detections = {}
        for camera, entry in zip(SCENE.cameras, record['cameras']):
            tracker = trackers[camera.name]
            if entry['png'] is None:
                tracker.last_detection = None  # the rig is not in this camera's frame
                detections[camera.name] = None
                continue
            tracker.estimate_pose_with_details(_full_image(camera, entry))
            d = tracker.last_detection
            detections[camera.name] = {'success': bool(d.success), 'message': d.message, 'seeded': bool(d.seeded),
                                       'reacquired': bool(d.reacquired),
                                       'anchor_ids': [int(a.marker_id) for a in d.anchors],
                                       'measurements': list(d.measurements) if d.success else []}
        estimate = hc.carrier_frame_pose(strategy, SCENE.cameras, _observations(record, layout, SCENE.cameras), **BA_KW)
        _dump(native_path(group, mount, hand, idx), {
            'idx': idx, 'detections': detections, 'seconds': time.perf_counter() - started,
            'estimate': {'success': bool(estimate.success and estimate.T_base_rig is not None),
                         'T': None if estimate.T_base_rig is None else np.asarray(estimate.T_base_rig),
                         'message': estimate.message}})
    return len(indices)


# --- estimator sessions -------------------------------------------------------------------

def session_perturbation(level, session):
    if level == 'nominal':
        return None
    sigma = {key: value[0] * LEVEL_SCALE[level] for key, value in PERTURBATIONS.items()}
    rng = np.random.default_rng([CFG['seed'], LEVEL_CODE[level], session])
    cameras = [(rng.normal(0, sigma['cam_rot_deg'], 3), rng.normal(0, sigma['cam_trans_mm'], 3),
                rng.normal(0, sigma['focal_ppm']), rng.normal(0, sigma['principal_px'], 2)) for _ in SCENE.cameras]
    scale = rng.normal(0, sigma['body_scale_ppm'])
    markers = [(rng.normal(0, sigma['marker_trans_mm'], 3), rng.normal(0, sigma['marker_rot_deg'], 3)) for _ in range(5)]
    tcp = (rng.normal(0, sigma['tcp_trans_mm'], 3), rng.normal(0, sigma['tcp_rot_deg'], 3))
    timing = rng.normal(0, sigma['timing_ms'])
    return {'cameras': cameras, 'body_scale_ppm': float(scale), 'markers': markers, 'tcp': tcp, 'timing_ms': float(timing)}


def estimator_cameras(perturbation):
    if perturbation is None:
        return SCENE.cameras
    out = []
    for camera, (rot, trans, focal, principal) in zip(SCENE.cameras, perturbation['cameras']):
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(np.deg2rad(rot)).as_matrix()
        delta[:3, 3] = np.asarray(trans) * 1e-3
        K = camera.K.copy()
        K[0, 0] *= 1 + focal * 1e-6
        K[1, 1] *= 1 + focal * 1e-6
        K[0, 2] += principal[0]
        K[1, 2] += principal[1]
        out.append(CameraCalibration(camera.name, camera.serial, camera.width, camera.height, K, camera.D.copy(),
                                     camera.T_base_cam @ delta, model='fisheye', sigma_pixel=camera.sigma_pixel))
    return out


def tcp_delta(perturbation):
    T = np.eye(4)
    if perturbation:
        dt, dr = perturbation['tcp']
        T[:3, :3] = Rotation.from_rotvec(np.deg2rad(dr)).as_matrix()
        T[:3, 3] = np.asarray(dt) * 1e-3
    return T


def task_solve(job):
    group, level, session, keys, mounts = job
    perturbation = session_perturbation(level, session)
    cameras = estimator_cameras(perturbation)
    layouts = {target: _layout(target, perturbation) for target in ('R0', 'H')}
    edge_scale = 1.0 + (perturbation['body_scale_ppm'] * 1e-6 if perturbation else 0.0)
    timing = perturbation['timing_ms'] * 1e-3 if perturbation else 0.0
    delta = tcp_delta(perturbation)
    T_rig_tcp = {target: S.T_rig_tcp(SCENE, S.mount_T_box_rig(SCENE, target, group, *mounts[target])) @ delta
                 for target in ('R0', 'H')}
    out = []
    for hand, idx in keys:
        truth = TRUTH[hand][idx]
        T_true_tcp = truth.box_at(timing) @ SCENE.T_box_tcp
        records = {'R0': _load(render_path(group, 'R0', mounts['R0'], hand, idx)),
                   'H': _load(render_path(group, 'H', mounts['H'], hand, idx))}
        native = _load(native_path(group, mounts['H'], hand, idx))
        row = {'hand': hand, 'idx': idx, 'arms': {}}
        for arm in S.ARMS:
            target = S.TARGET_OF_ARM[arm]
            observations = _observations(records[target], layouts[target], cameras)
            estimate = None
            if observations and arm == 'H1':
                trackers = {}
                for name, d in native['detections'].items():
                    if d is None:
                        trackers[name] = SimpleNamespace(last_detection=None)
                        continue
                    measurements = d['measurements']
                    if edge_scale != 1.0:
                        measurements = [replace(m, point_rig=np.asarray(m.point_rig) * edge_scale) for m in measurements]
                    trackers[name] = SimpleNamespace(last_detection=SimpleNamespace(success=d['success'],
                                                                                    measurements=measurements))
                strategy = SimpleNamespace(carrier_cfg=CARRIER_CFG, trackers=trackers, seeds={})
                estimate = hc.carrier_frame_pose(strategy, cameras, observations, **BA_KW)
            elif observations:
                estimate = estimate_frame_pose(cameras, observations, **BA_KW)
            ok = bool(estimate is not None and estimate.success and estimate.T_base_rig is not None)
            cell = {'ok': ok, 'cams': len({o.camera_name for o in observations}), 'corners': len(observations)}
            if ok:
                T_est_tcp = np.asarray(estimate.T_base_rig) @ T_rig_tcp[target]
                sp, sr = hc.hikon_pipeline.ee_sigmas_from_frame_estimate(estimate, T_rig_tcp[target])
                cell.update(pose7=pose7(T_est_tcp), evec=(T_est_tcp[:3, 3] - T_true_tcp[:3, 3]) * 1e3,
                            rot=rotation_error_deg(T_est_tcp[:3, :3], T_true_tcp[:3, :3]),
                            sp=np.asarray(sp, float), sr=np.asarray(sr, float),
                            excluded=len(estimate.excluded_camera_names),
                            refined=int('facets:' in estimate.message),
                            discarded=int('refinement discarded' in estimate.message))
                if arm == 'H1' and level == 'nominal' and native['estimate']['T'] is not None:
                    cell['native_gap_m'] = float(np.max(np.abs(np.asarray(estimate.T_base_rig) - native['estimate']['T'])))
                elif arm == 'H1' and level == 'nominal':
                    cell['native_gap_m'] = float('inf')
            elif arm == 'H1' and level == 'nominal':
                cell['native_gap_m'] = 0.0 if not native['estimate']['success'] else float('inf')
            row['arms'][arm] = cell
        out.append(row)
    return {'group': group, 'level': level, 'session': session, 'rows': out,
            'timing_ms': perturbation['timing_ms'] if perturbation else 0.0,
            'perturbation': None if perturbation is None else {
                'body_scale_ppm': perturbation['body_scale_ppm'], 'timing_ms': perturbation['timing_ms'],
                'tcp_trans_mm': np.asarray(perturbation['tcp'][0]).tolist(),
                'cam_rot_deg_rms': float(np.sqrt(np.mean([np.sum(np.square(c[0])) for c in perturbation['cameras']])))}}


def collate_session(results, keys):
    rows = {}
    for result in results:
        for row in result['rows']:
            rows[(row['hand'], row['idx'])] = row
    n = len(keys)
    arms = {}
    for arm in S.ARMS:
        a = {'ok': np.zeros(n, bool), 'pose7': np.full((n, 7), np.nan), 'evec': np.full((n, 3), np.nan),
             'rot': np.full(n, np.nan), 'sp': np.full((n, 3), np.nan), 'sr': np.full((n, 3), np.nan),
             'cams': np.zeros(n, np.int16), 'refined': np.zeros(n, np.int8), 'discarded': np.zeros(n, np.int8),
             'native_gap_m': np.zeros(n)}
        for i, key in enumerate(keys):
            cell = rows[key]['arms'][arm]
            a['ok'][i] = cell['ok']
            a['cams'][i] = cell['cams']
            a['native_gap_m'][i] = cell.get('native_gap_m', 0.0)
            if cell['ok']:
                a['pose7'][i], a['evec'][i], a['rot'][i] = cell['pose7'], cell['evec'], cell['rot']
                a['sp'][i], a['sr'][i] = cell['sp'], cell['sr']
                a['refined'][i], a['discarded'][i] = cell['refined'], cell['discarded']
        arms[arm] = a
    first = results[0]
    return {'keys': keys, 'timing_ms': first['timing_ms'], 'perturbation': first['perturbation'], 'arms': arms}


def task_smooth(job):
    group, level, session = job
    path = smooth_path(group, level, session)
    if path.exists():
        return str(path)
    data = _load(solve_path(group, level, session))
    keys = data['keys']
    timing = data['timing_ms'] * 1e-3
    t = np.array([TRUTH[h][i].t for h, i in keys])
    blocks = [(h, TRUTH[h][i].episode) for h, i in keys]
    out = {}
    for arm in S.ARMS:
        a = data['arms'][arm]
        smoothed = np.full((len(keys), 7), np.nan)
        for block in sorted(set(blocks)):
            members = np.array([b == block for b in blocks])
            idx = np.flatnonzero(members & a['ok'])
            if len(idx) < 2:
                continue
            meas_m = np.sqrt(np.mean(a['sp'][idx] ** 2, axis=1))
            meas_deg = np.sqrt(np.mean(a['sr'][idx] ** 2, axis=1))
            meas_m = np.where(np.isfinite(meas_m), np.maximum(meas_m, SMOOTH_CFG['min_measurement_sigma_m']), np.nan)
            meas_deg = np.where(np.isfinite(meas_deg), np.maximum(meas_deg, SMOOTH_CFG['min_measurement_sigma_deg']), np.nan)
            result = smooth_pose_trajectory(
                a['pose7'][idx], t[idx], measurement_sigma_m=meas_m, measurement_sigma_rad=np.deg2rad(meas_deg),
                velocity_change_sigma_mps=SMOOTH_CFG['velocity_change_sigma_mps'],
                velocity_change_sigma_radps=SMOOTH_CFG['velocity_change_sigma_radps'], loss=SMOOTH_CFG['loss'],
                f_scale=SMOOTH_CFG['f_scale'], max_nfev=SMOOTH_CFG['max_nfev'])
            smoothed[idx] = result.pose7
        evec = np.full((len(keys), 3), np.nan)
        rot = np.full(len(keys), np.nan)
        for i, (hand, idx) in enumerate(keys):
            if not np.isfinite(smoothed[i]).all():
                continue
            T_true = TRUTH[hand][idx].box_at(timing) @ SCENE.T_box_tcp
            T_est = S.pose_matrix(smoothed[i, :3], smoothed[i, 3:])
            evec[i] = (T_est[:3, 3] - T_true[:3, 3]) * 1e3
            rot[i] = rotation_error_deg(T_est[:3, :3], T_true[:3, :3])
        out[arm] = {'evec': evec, 'rot': rot}
    _dump(path, out)
    return str(path)


# --- stages ------------------------------------------------------------------------------

def stage_select(pool, cfg):
    path = Path(cfg['run']) / 'select.json'
    if path.exists():
        return json.loads(path.read_text())
    keys = frame_keys(cfg['dev_episodes'], cfg['stride_dev'], cfg['max_per_block'])
    candidates = [[yaw, pitch] for yaw in cfg['yaw_candidates'] for pitch in cfg['pitch_candidates']]
    jobs = [('gripper', target, mount, hand, idx, target == 'H', cfg['supersample'])
            for target in ('R0', 'H') for mount in candidates for hand, idx in keys]
    run_jobs(pool, task_render, jobs, 'select/render')
    blocks = [('gripper', mount, hand, [i for h, i in keys if h == hand]) for mount in candidates for hand in HANDS]
    run_jobs(pool, task_native, blocks, 'select/native')
    table = []
    jobs = [('gripper', 'nominal', 0, chunk, {'R0': mount, 'H': mount}) for mount in candidates for chunk in chunks(keys, 24)]
    results = run_jobs(pool, task_solve, jobs, 'select/solve')
    for k, mount in enumerate(candidates):
        per_chunk = len(chunks(keys, 24))
        data = collate_session(results[k * per_chunk:(k + 1) * per_chunk], keys)
        for target, arm in (('R0', 'R0'), ('H', 'H1')):
            a = data['arms'][arm]
            e = np.linalg.norm(a['evec'][a['ok']], axis=1)
            table.append({'target': target, 'arm': arm, 'yaw_deg': mount[0], 'pitch_deg': mount[1], 'frames': len(keys),
                          'coverage': float(a['ok'].mean()), 'p95_mm': float(np.percentile(e, 95)) if e.size else None,
                          'p50_mm': float(np.percentile(e, 50)) if e.size else None})
    chosen = {}
    for target in ('R0', 'H'):
        rows = [r for r in table if r['target'] == target]
        best = max(r['coverage'] for r in rows)
        near = [r for r in rows if r['coverage'] >= best - 0.01 and r['p95_mm'] is not None]
        pick = min(near, key=lambda r: r['p95_mm'])
        chosen[target] = [pick['yaw_deg'], pick['pitch_deg']]
    doc = {'rule': ('per target on dev episode 0, both hands, over the same yaw x boom-tilt candidates: highest raw '
                    'coverage; among mounts within 1 point of it, lowest raw TCP p95 (R0 judged on R0, the carrier on '
                    'H1); frozen before held-out rendering'),
           'dev_frames': len(keys), 'table': table, 'mounts': chosen}
    path.write_text(json.dumps(doc, indent=1))
    return doc


def group_specs(cfg, mounts):
    return {
        'gripper': {'episodes': cfg['heldout_episodes'], 'stride': cfg['stride_heldout'], 'mounts': mounts,
                    'levels': {'nominal': 1, 'evidence': cfg['sessions_evidence'], 'stress': cfg['sessions_stress']}},
        # The control keeps each target's dev-chosen yaw and tilt (dome up) and only moves the socket onto the TCP.
        'socket_tcp': {'episodes': cfg['heldout_episodes'], 'stride': cfg['stride_socket'], 'mounts': mounts,
                       'levels': {'nominal': 1, 'evidence': cfg['sessions_socket']}},
    }


def stage_render(pool, cfg, specs):
    jobs = []
    for group, spec in specs.items():
        for hand, idx in frame_keys(spec['episodes'], spec['stride'], cfg['max_per_block']):
            for target in ('R0', 'H'):
                jobs.append((group, target, spec['mounts'][target], hand, idx, target == 'H', cfg['supersample']))
    run_jobs(pool, task_render, jobs, 'render')


def stage_native(pool, cfg, specs):
    blocks = []
    for group, spec in specs.items():
        keys = frame_keys(spec['episodes'], spec['stride'], cfg['max_per_block'])
        for hand in HANDS:
            for episode in spec['episodes']:
                indices = [i for h, i in keys if h == hand and TRUTH[h][i].episode == episode]
                if indices:
                    blocks.append((group, spec['mounts']['H'], hand, indices))
    run_jobs(pool, task_native, blocks, 'native')


def stage_solve(pool, cfg, specs):
    for group, spec in specs.items():
        keys = frame_keys(spec['episodes'], spec['stride'], cfg['max_per_block'])
        for level, sessions in spec['levels'].items():
            todo = [s for s in range(sessions) if not solve_path(group, level, s).exists()]
            jobs = [(group, level, s, chunk, spec['mounts']) for s in todo for chunk in chunks(keys, cfg['solve_chunk'])]
            results = run_jobs(pool, task_solve, jobs, f'solve {group}/{level}')
            for s in todo:
                _dump(solve_path(group, level, s), collate_session([r for r in results if r['session'] == s], keys))


def stage_smooth(pool, cfg, specs):
    jobs = [(group, level, s) for group, spec in specs.items() for level, n in spec['levels'].items() for s in range(n)]
    run_jobs(pool, task_smooth, jobs, 'smooth')


def _gradient_l1(grey):
    return (np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))) / 8.0


def task_convergence(job):
    """Renderer convergence and the detector's corner bias, on noiseless single-pose views.

    Convergence compares the rendered image at the run's supersample with 4x:
    sum|I_a - I_b| / sum|grad I| is the gradient-normalised mean edge shift in
    pixels. The corner bias is ArUco's decoded corner against the analytic
    projection of the same corner -- a property of the detector on these
    images, reported, not gated.
    """
    group, target, mount, hand, idx = job
    truth, other = TRUTH[hand][idx], TRUTH[OTHER[hand]].get(idx)
    model = SCENE.models[target]
    T = truth.box_at(0.0) @ S.mount_T_box_rig(SCENE, target, group, *mount)
    tris, spheres = S.frame_occluders(SCENE, group, truth.box_at(0.0), other.box_at(0.0) if other else None)
    body = SCENE.body_colour(target)
    fine = S.SceneRenderer(CFG['reference_supersample'])
    shifts, bias, mismatch, decodes = [], [], 0, 0
    known = model.anchors_by_id
    for camera in SCENE.cameras:
        window = RENDERER.window(model, camera, [T])
        if window is None:
            continue
        a, _ = RENDERER.render_window(model, camera, T, window, tris, spheres, body)
        b, _ = fine.render_window(model, camera, T, window, tris, spheres, body)
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        denominator = 0.5 * float((_gradient_l1(ga) + _gradient_l1(gb)).sum())
        if denominator > 0:
            shifts.append(float(np.abs(ga - gb).sum()) / denominator)
        found = {}
        for label, renderer in (('base', RENDERER), ('fine', fine)):
            image, _, _ = renderer.render_camera(model, camera, [T], [(tris, spheres)], body,
                                                 blur_sigma=CFG['blur_sigma'], noise_sigma=0.0, rng=np.random.default_rng(0))
            corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            found[label] = {} if ids is None else {int(m): np.asarray(c).reshape(4, 2)
                                                   for c, m in zip(corners, ids.ravel()) if int(m) in known}
        decodes += len(set(found['base']) | set(found['fine']))
        mismatch += len(set(found['base']) ^ set(found['fine']))
        for marker_id, uv in found['base'].items():
            anchor = known[marker_id]
            projected = project_rig(camera, camera.T_cam_base @ T, anchor.object_points(anchor.paste_quadrant or 0))
            bias.extend(np.linalg.norm(uv - projected, axis=1).tolist())
    return {'target': target, 'shifts': shifts, 'bias': bias, 'mismatch': mismatch, 'decodes': decodes}


def stage_convergence(pool, cfg, specs):
    spec = specs['gripper']
    keys = frame_keys(spec['episodes'], spec['stride'], cfg['max_per_block'])
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(cfg['convergence_frames'], len(keys))).astype(int)]
    jobs = [('gripper', target, spec['mounts'][target], hand, idx) for target in ('R0', 'H') for hand, idx in keys]
    return {'frames': len(keys), 'results': run_jobs(pool, task_convergence, jobs, 'convergence')}


# --- statistics ---------------------------------------------------------------------------

def _q(values, q):
    """Inverted-CDF quantile: the same definition the weighted bootstrap uses."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q, method='inverted_cdf')) if values.size else None


def arm_summary(ok, err, rot):
    total = ok.size
    if not total:
        return {'requested': 0, 'accepted': 0, 'coverage': None, 'p50': None, 'p95': None, 'p99': None, 'max': None,
                'rotation_p95': None, 'accurate_yield': None, 'catastrophic_rate': None, 'all_frame_p95': None}
    e, r = err[ok], rot[ok]
    return {'requested': int(total), 'accepted': int(ok.sum()), 'coverage': float(ok.mean()) if total else None,
            'p50': _q(e, 50), 'p95': _q(e, 95), 'p99': _q(e, 99), 'max': _q(e, 100), 'rotation_p95': _q(r, 95),
            'accurate_yield': float(np.sum(ok & (err <= 3) & (rot <= 0.5)) / total) if total else None,
            'catastrophic_rate': float(np.sum(ok & ((err > 20) | (rot > 5))) / total) if total else None,
            # Failed frames count as +inf; null when the 95th percentile lands on one.
            'all_frame_p95': float(np.percentile(np.where(ok, err, np.inf), 95, method='inverted_cdf'))}


def _weighted_step_quantile(sorted_values, weights, q):
    cumulative = np.cumsum(weights, axis=1)
    index = np.argmax(cumulative >= q * cumulative[:, -1:], axis=1)
    return sorted_values[index]


def bootstrap_paired(a, b, sessions, blocks, rng, replicates):
    """95% interval of p95(a) - p95(b), resampling sessions and 2 s blocks."""
    if a.size < 20:
        return [None, None]
    _, s_idx = np.unique(sessions, return_inverse=True)
    _, b_idx = np.unique(blocks, return_inverse=True)
    n_s, n_b = s_idx.max() + 1, b_idx.max() + 1
    if n_b < 2:
        return [None, None]
    order_a, order_b = np.argsort(a), np.argsort(b)
    sa, sb = a[order_a], b[order_b]
    values = []
    for start in range(0, replicates, 64):
        r = min(64, replicates - start)
        cs = rng.multinomial(n_s, np.full(n_s, 1.0 / n_s), size=r)
        cb = rng.multinomial(n_b, np.full(n_b, 1.0 / n_b), size=r)
        weights = cs[:, s_idx] * cb[:, b_idx]
        values.append(_weighted_step_quantile(sa, weights[:, order_a], .95)
                      - _weighted_step_quantile(sb, weights[:, order_b], .95))
    values = np.concatenate(values)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_delta(ok_a, e_a, ok_b, e_b, sessions, blocks, rng, replicates, mask=None):
    common = ok_a & ok_b & (True if mask is None else mask)
    if common.sum() < 20:
        return {'n': int(common.sum()), 'delta': None, 'ci': [None, None], 'a_p95': None, 'b_p95': None}
    a, b = e_a[common], e_b[common]
    return {'n': int(common.sum()), 'a_p95': _q(a, 95), 'b_p95': _q(b, 95), 'delta': _q(a, 95) - _q(b, 95),
            'ci': bootstrap_paired(a, b, sessions[common], blocks[common], rng, replicates)}


def time_block(hand, truth, seconds=2.0):
    """Bootstrap block id: hand x episode x 2 s window.

    Integer-coded on purpose: Python's hash() of a tuple holding a string is
    salted per process, which would reorder the blocks and move every interval
    between two runs over identical data.
    """
    return HAND_CODE[hand] * 1_000_000 + truth.episode * 10_000 + int(truth.t // seconds)


def load_level(group, level, sessions, keys):
    ok, err, rot, ok_s, err_s, rot_s, extra = ({a: [] for a in S.ARMS} for _ in range(7))
    session_label, block_label, frame_label = [], [], []
    frame_block = np.array([time_block(h, TRUTH[h][i]) for h, i in keys])
    timings = []
    for s in range(sessions):
        solved = _load(solve_path(group, level, s))
        smoothed = _load(smooth_path(group, level, s))
        timings.append(solved['timing_ms'])
        for arm in S.ARMS:
            a = solved['arms'][arm]
            ok[arm].append(a['ok'])
            err[arm].append(np.linalg.norm(a['evec'], axis=1))
            rot[arm].append(a['rot'])
            e_s = np.linalg.norm(smoothed[arm]['evec'], axis=1)
            ok_s[arm].append(np.isfinite(e_s))
            err_s[arm].append(e_s)
            rot_s[arm].append(smoothed[arm]['rot'])
            extra[arm].append(a)
        session_label.append(np.full(len(keys), s))
        block_label.append(frame_block)
        frame_label.append(np.arange(len(keys)))
    cat = lambda d: {arm: np.concatenate(v) for arm, v in d.items()}  # noqa: E731
    return {'raw': (cat(ok), cat(err), cat(rot)), 'smoothed': (cat(ok_s), cat(err_s), cat(rot_s)),
            'sessions': np.concatenate(session_label), 'blocks': np.concatenate(block_label),
            'frames': np.concatenate(frame_label), 'extra': extra, 'timings_ms': timings}


def bucket_masks(group, keys, render_stats):
    speed = np.array([S.tcp_speed(TRUTH[h][i], SCENE.T_box_tcp) for h, i in keys])
    occlusion = np.array([np.mean([render_stats[t][k] for t in ('R0', 'H')]) for k in range(len(keys))])
    interpolated = np.array([TRUTH[h][i].interpolated for h, i in keys])
    hand = np.array([h for h, _ in keys])
    return speed, occlusion, [
        ('speed', 'slow', '<0.15 m/s', speed < 0.15), ('speed', 'medium', '0.15–0.45 m/s', (speed >= 0.15) & (speed < 0.45)),
        ('speed', 'fast', '≥0.45 m/s', speed >= 0.45),
        ('occlusion', 'clear', '遮挡 <5%', occlusion < 0.05), ('occlusion', 'partial', '遮挡 5–20%', (occlusion >= 0.05) & (occlusion < 0.2)),
        ('occlusion', 'heavy', '遮挡 ≥20%', occlusion >= 0.2),
        ('truth', 'observed', '真值帧（cube 实测）', ~interpolated), ('truth', 'interpolated', '真值帧（插值）', interpolated),
        ('hand', 'left', '左手', hand == 'left'), ('hand', 'right', '右手', hand == 'right'),
    ]


def render_occlusion(group, keys, mounts):
    stats = {}
    decoded = {}
    for target in ('R0', 'H'):
        ids = {a.marker_id for a in SCENE.models[target].anchors}
        occ, dec = [], []
        for hand, idx in keys:
            record = _load(render_path(group, target, mounts[target], hand, idx))
            tpx = sum(c['target_px'] for c in record['cameras'])
            opx = sum(c['occluded_px'] for c in record['cameras'])
            occ.append(opx / tpx if tpx else 1.0)
            dec.append([sum(d['marker_id'] in ids for d in c['detections']) for c in record['cameras']])
        stats[target] = np.array(occ)
        decoded[target] = dec
    return stats, decoded


def level_summary(group, level, sessions, keys, masks, rng, replicates):
    data = load_level(group, level, sessions, keys)
    n = len(keys)
    out = {'sessions': sessions, 'frames': n, 'samples': n * sessions}
    for output in ('raw', 'smoothed'):
        ok, err, rot = data[output]
        block = {'arms': {arm: arm_summary(ok[arm], err[arm], rot[arm]) for arm in S.ARMS}, 'paired': {}, 'buckets': []}
        for a_arm, b_arm in (('R0', 'H1'), ('R0', 'H0'), ('H0', 'H1')):
            block['paired'][f'{a_arm}_{b_arm}'] = paired_delta(ok[a_arm], err[a_arm], ok[b_arm], err[b_arm],
                                                               data['sessions'], data['blocks'], rng, replicates)
        for family, name, label, mask in masks:
            tiled = np.tile(mask, sessions)
            block['buckets'].append({'family': family, 'name': name, 'label': label, 'frames': int(mask.sum()),
                                     'arms': {arm: arm_summary(ok[arm][tiled], err[arm][tiled], rot[arm][tiled])
                                              for arm in S.ARMS},
                                     'R0_H1': paired_delta(ok['R0'], err['R0'], ok['H1'], err['H1'], data['sessions'],
                                                           data['blocks'], rng, max(200, replicates // 4), tiled)})
        xmax = 12.0
        grid = np.linspace(0, xmax, 121)
        block['cdf'] = {'x_mm': grid.round(3).tolist(), 'arms': {
            arm: [float(np.sum(ok[arm] & (err[arm] <= x)) / ok[arm].size) for x in grid] for arm in S.ARMS}}
        out[output] = block
    out['timings_ms'] = data['timings_ms']
    return out, data


def rpe_and_gaps(keys, data, stride):
    dt = stride / 60.0
    blocks = [(h, TRUTH[h][i].episode) for h, i in keys]
    n = len(keys)
    out = {}
    for arm in S.ARMS:
        ok_s = data['smoothed'][0][arm][:n]
        solved = data['extra'][arm][0]
        smooth = _load(smooth_path(CFG['_group'], 'nominal', 0))[arm]
        rpe = {}
        for lag_s in (0.04, 0.1, 0.4, 1.0):
            k = max(1, int(round(lag_s / dt)))
            diffs = []
            for i in range(n - k):
                j = i + k
                if blocks[i] != blocks[j] or not (ok_s[i] and ok_s[j]):
                    continue
                truth_i, truth_j = TRUTH[keys[i][0]][keys[i][1]], TRUTH[keys[j][0]][keys[j][1]]
                true_step = ((truth_j.T_world_box @ SCENE.T_box_tcp)[:3, 3] - (truth_i.T_world_box @ SCENE.T_box_tcp)[:3, 3]) * 1e3
                est_step = (smooth['evec'][j] - smooth['evec'][i]) + true_step
                diffs.append(float(np.linalg.norm(est_step - true_step)))
            rpe[f'{k * dt * 1000:.0f}ms'] = {'n': len(diffs), 'p95_mm': _q(diffs, 95), 'p50_mm': _q(diffs, 50)}
        gaps, run, last_block = [], 0, None
        for i in range(n):
            if blocks[i] != last_block:
                if run:
                    gaps.append(run)
                run, last_block = 0, blocks[i]
            if solved['ok'][i]:
                if run:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if run:
            gaps.append(run)
        durations = np.array(gaps, float) * dt
        ok = solved['ok']
        d2 = np.sum((solved['evec'][ok] / np.maximum(solved['sp'][ok] * 1e3, 1e-9)) ** 2, axis=1)
        out[arm] = {'rpe': rpe, 'gap_count': int(len(gaps)), 'gap_p95_s': _q(durations, 95), 'gap_max_s': _q(durations, 100),
                    'gaps_over_100ms': int(np.sum(durations > 0.1)),
                    'sigma_ellipsoid95_coverage': float(np.mean(d2 <= 7.815)) if d2.size else None,
                    'sigma_tcp_p95_mm': _q(np.sqrt(np.sum(solved['sp'][ok] ** 2, axis=1)) * 1e3, 95),
                    'h1_refined_rate': float(np.mean(solved['refined'][ok])) if arm == 'H1' and ok.any() else None,
                    'h1_discarded_rate': float(np.mean(solved['discarded'][ok])) if arm == 'H1' and ok.any() else None}
    return out


def convergence_report(cfg, convergence):
    results = convergence['results']
    shifts = [s for r in results for s in r['shifts']]
    mismatch, decodes = sum(r['mismatch'] for r in results), sum(r['decodes'] for r in results)
    p95 = _q(shifts, 95)
    bias = {}
    for target in ('R0', 'H'):
        values = [b for r in results if r['target'] == target for b in r['bias']]
        bias[target] = {'corners': len(values), 'p50_px': _q(values, 50), 'p95_px': _q(values, 95), 'max_px': _q(values, 100)}
    return {'frames': convergence['frames'], 'views': len(shifts), 'supersample': cfg['supersample'],
            'reference_supersample': cfg['reference_supersample'],
            'edge_shift_p95_px': p95, 'edge_shift_max_px': _q(shifts, 100), 'threshold_px': 0.05,
            'decode_mismatch': mismatch, 'decodes': decodes, 'aruco_corner_bias_noiseless': bias,
            'pass': bool(p95 is not None and p95 < 0.05 and mismatch <= max(1, 0.02 * decodes))}


def _png_b64(image, width=300):
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', image)[1].tobytes()).decode()


def counterexamples(keys, nominal, specs, speed, occlusion):
    ok, err, _ = nominal['raw']
    n = len(keys)
    picks = []

    def add(kind, label, i):
        if i is not None and all(p['frame'] != int(i) for p in picks):
            picks.append({'kind': kind, 'label': label, 'frame': int(i)})

    e_r0, e_h1 = np.where(ok['R0'][:n], err['R0'][:n], -1), np.where(ok['H1'][:n], err['H1'][:n], -1)
    add('worst_h1', 'H1 误差最大帧', int(np.argmax(e_h1)) if (e_h1 >= 0).any() else None)
    add('worst_r0', 'R0 误差最大帧', int(np.argmax(e_r0)) if (e_r0 >= 0).any() else None)
    only_r0 = np.flatnonzero(ok['R0'][:n] & ~ok['H1'][:n])
    only_h1 = np.flatnonzero(~ok['R0'][:n] & ok['H1'][:n])
    add('h1_fail', 'R0 出位姿、H1 失败', int(only_r0[np.argmax(occlusion[only_r0])]) if only_r0.size else None)
    add('r0_fail', 'H1 出位姿、R0 失败', int(only_h1[np.argmax(occlusion[only_h1])]) if only_h1.size else None)
    add('occluded', '遮挡最重帧', int(np.argmax(occlusion)))
    add('fast', 'TCP 速度最高帧', int(np.argmax(speed)))
    spec = specs['gripper']
    for pick in picks:
        hand, idx = keys[pick['frame']]
        pick.update(hand=hand, index=idx, speed_mps=float(speed[pick['frame']]),
                    occlusion=float(occlusion[pick['frame']]), arms={
                        arm: {'ok': bool(ok[arm][pick['frame']]),
                              'error_mm': float(err[arm][pick['frame']]) if ok[arm][pick['frame']] else None}
                        for arm in S.ARMS})
        views = []
        for target in ('R0', 'H'):
            record = _load(render_path('gripper', target, spec['mounts'][target], hand, idx))
            ids = {a.marker_id for a in SCENE.models[target].anchors}
            ranked = sorted(zip(SCENE.cameras, record['cameras']), key=lambda ce: -ce[1]['target_px'])[:2]
            for camera, entry in ranked:
                if entry['box'] is None:
                    continue
                truth, other = TRUTH[hand][idx], TRUTH[OTHER[hand]].get(idx)
                T = truth.box_at(0.0) @ S.mount_T_box_rig(SCENE, target, 'gripper', *spec['mounts'][target])
                occ = S.frame_occluders(SCENE, 'gripper', truth.box_at(0.0), other.box_at(0.0) if other else None)
                image, box, _ = RENDERER.render_camera(SCENE.models[target], camera, [T], [occ], SCENE.body_colour(target),
                                                       blur_sigma=0.0, noise_sigma=0.0, rng=np.random.default_rng(0))
                if image is None:
                    continue
                x0, y0, x1, y1 = box
                crop = image[y0:y1, x0:x1].copy()
                for d in entry['detections']:
                    colour = (60, 200, 60) if d['marker_id'] in ids else (40, 40, 220)
                    cv2.polylines(crop, [np.round(d['points_2d'] - [x0, y0]).astype(np.int32)], True, colour, 1)
                views.append({'target': target, 'camera': camera.name,
                              'decoded': sorted(d['marker_id'] for d in entry['detections'] if d['marker_id'] in ids),
                              'occluded_fraction': entry['occluded_px'] / entry['target_px'] if entry['target_px'] else None,
                              'image': _png_b64(crop)})
        pick['views'] = views
    return picks


def git_state():
    try:
        rev = subprocess.check_output(['git', '-C', str(S.ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(S.ROOT), 'status', '--porcelain', '--', str(HERE)], text=True).strip())
        return {'revision': rev, 'simulator_dir_dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'revision': None, 'simulator_dir_dirty': None}


def decide(groups, checks):
    primary = groups['gripper']['levels']['evidence']['smoothed']
    delta = primary['paired']['R0_H1']
    lo, hi = delta['ci']
    r0, h1 = primary['arms']['R0'], primary['arms']['H1']
    inversions = [b['label'] for b in primary['buckets'] if b['family'] in ('speed', 'occlusion')
                  and b['R0_H1']['ci'][1] is not None and b['R0_H1']['ci'][1] < 0]
    uppers = []
    for level in groups['gripper']['levels'].values():
        for output in ('raw', 'smoothed'):
            uppers.append(level[output]['paired']['R0_H1']['ci'][1])
    numeric_go = bool(lo is not None and lo >= 0.8 and h1['p95'] is not None and h1['p95'] <= 3
                      and h1['rotation_p95'] <= 0.5 and h1['coverage'] > 0.99 and h1['coverage'] >= r0['coverage']
                      and h1['catastrophic_rate'] <= r0['catastrophic_rate'] and not inversions)
    numeric_nogo = bool(hi is not None and hi < 0.8)
    numeric = 'CONDITIONAL_GO' if numeric_go else 'NO_GO' if numeric_nogo else 'INCONCLUSIVE'
    robust_nogo = bool(uppers and all(u is not None and u < 0.8 for u in uppers))
    g0 = all(c['ok'] for c in checks if c['blocking'])
    formal = numeric if g0 else 'INCONCLUSIVE'
    missing = [c['label'] for c in checks if c['blocking'] and not c['ok']]
    return {'formal': formal, 'numeric': numeric, 'g0_pass': g0, 'missing_blocking_inputs': missing,
            'primary': {'group': 'gripper', 'level': 'evidence', 'output': 'smoothed', 'delta_mm': delta['delta'],
                        'ci95_mm': delta['ci'], 'h1_p95_mm': h1['p95'], 'r0_p95_mm': r0['p95'],
                        'h1_rotation_p95_deg': h1['rotation_p95'], 'h1_coverage': h1['coverage'],
                        'r0_coverage': r0['coverage']},
            'bucket_inversions': inversions, 'nogo_robust_across_levels_and_outputs': robust_nogo,
            'rule': ('CONDITIONAL_GO needs the 95% lower bound of paired dp95(R0-H1) >= 0.8 mm plus H1 p95 <= 3 mm, '
                     'rotation p95 <= 0.5 deg, coverage > 99% and not below R0, no worse catastrophic rate and no '
                     'fast/occluded bucket reversal; NO_GO when the upper bound is < 0.8 mm. The formal decision '
                     'equals the numeric one only when every blocking G0 input is closed.')}


# --- where on the body the targets differ ---------------------------------------------------

# 2026-09-09 bench (docs/hybrid_carrier_v1_roadmap_20260904.html): the carrier held by hand in open space in front of
# all seven cameras, every number reported at the rig origin (the socket); none of it is at a gripper TCP.
BENCH_0909 = {'noise_translation_rms_mm': 0.23, 'noise_rotation_rms_deg': 0.16, 'hand_speed_mps': 0.057,
              'refinement_step_below_mm': 0.05, 'cameras_per_frame_p50': 7,
              'source': 'docs/hybrid_carrier_v1_roadmap_20260904.html, 2026-09-09 bench'}


def point_errors_mm(R_est, t_est, R_true, t_true, point):
    """|error| in mm at one rig-frame point, for stacked estimated and true poses (n x 3 x 3, n x 3)."""
    return np.linalg.norm(np.einsum('nij,j->ni', R_est - R_true, np.asarray(point, float)) + t_est - t_true, axis=1) * 1e3


def cubic_residuals(t, y):
    """Residual of y (n x k) about a least-squares cubic in t, rescaled for the four fitted coefficients."""
    t = np.asarray(t, float) - np.mean(t)
    A = np.vander(t, 4)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (y - A @ coef) * np.sqrt(len(t) / (len(t) - 4))


def _p50_p95(chunks_):
    values = np.concatenate(chunks_) if chunks_ else np.zeros(0)
    return {'p50': _q(values, 50), 'p95': _q(values, 95)}


def noise_floor(keys, T_box_rig, series, max_speed=0.10, window_s=1.0, min_frames=20):
    """The bench's noise-floor statistic on simulated poses: 1 s windows at a slow hand, residual about a local cubic.

    The bench fits the estimate because it has no truth. Here the error signal (estimate - truth) is fitted
    instead: the recorded truth trajectory is itself far from cubic over a second. Per-axis rms, as the bench
    reports it.
    """
    blocks = {}
    for i, (hand, idx) in enumerate(keys):
        blocks.setdefault((hand, TRUTH[hand][idx].episode), []).append(i)
    out = {}
    for arm, (ok, d_mm, r_deg) in series.items():
        target = S.TARGET_OF_ARM[arm]
        res_t, res_r = [], []
        for (hand, _), members in blocks.items():
            members = np.array(sorted(members, key=lambda i: TRUTH[hand][keys[i][1]].t))
            times = np.array([TRUTH[hand][keys[i][1]].t for i in members])
            p = np.array([(TRUTH[hand][keys[i][1]].box_at(0.0) @ T_box_rig[target])[:3, 3] for i in members])
            start = times[0]
            while start + window_s <= times[-1]:
                sel = (times >= start) & (times < start + window_s)
                start += window_s
                rows = members[sel]
                if rows.size < min_frames or not ok[rows].all():
                    continue
                if np.mean(np.linalg.norm(np.diff(p[sel], axis=0), axis=1) / np.diff(times[sel])) > max_speed:
                    continue
                res_t.append(cubic_residuals(times[sel], d_mm[rows]))
                res_r.append(cubic_residuals(times[sel], r_deg[rows]))
        out[arm] = {'windows': len(res_t), 'frames': int(sum(len(r) for r in res_t)),
                    'translation_rms_mm': float(np.sqrt(np.mean(np.concatenate(res_t) ** 2))) if res_t else None,
                    'rotation_rms_deg': float(np.sqrt(np.mean(np.concatenate(res_r) ** 2))) if res_r else None}
    return out


def mechanism_report(group, spec, keys, decoded):
    """Where on the rigid body the targets differ, plus the statistics the 09-09 bench reports.

    A rigid target's error at any point is its error at the feature centroid plus
    the rotation error times the arm from the centroid. Every accepted per-frame
    pose is re-read at the anchor centroid, at the rig origin (the socket, where
    the bench reports), at a common arm (R0's centroid->TCP length along each
    target's own direction, so the mount direction drops out) and at the TCP
    label the decision uses. The pose-only columns undo the session's
    marker->TCP draw; the TCP column keeps it.
    """
    T_box_rig = {t: S.mount_T_box_rig(SCENE, t, group, *spec['mounts'][t]) for t in ('R0', 'H')}
    T_rig_tcp = {t: S.T_rig_tcp(SCENE, T_box_rig[t]) for t in ('R0', 'H')}
    geometry, points = {}, {}
    for t in ('R0', 'H'):
        corners = np.concatenate([np.asarray(m['corners_rig'], float) for m in SCENE.layouts[t]['markers']])
        centroid = corners.mean(axis=0)
        arm = T_rig_tcp[t][:3, 3] - centroid
        geometry[t] = {'markers': len(SCENE.layouts[t]['markers']),
                       'corner_rms_radius_mm': float(np.sqrt(np.mean(np.sum((corners - centroid) ** 2, axis=1))) * 1e3),
                       'centroid_to_tcp_mm': float(np.linalg.norm(arm) * 1e3),
                       'origin_to_tcp_mm': float(np.linalg.norm(T_rig_tcp[t][:3, 3]) * 1e3)}
        points[t] = (centroid, arm / max(float(np.linalg.norm(arm)), 1e-12))
    common_m = geometry['R0']['centroid_to_tcp_mm'] * 1e-3
    report = {'geometry': geometry, 'common_arm_mm': geometry['R0']['centroid_to_tcp_mm'], 'levels': {}}
    n = len(keys)
    for level in ('nominal', 'evidence'):
        if level not in spec['levels']:
            continue
        acc = {arm: {k: [] for k in ('centroid', 'origin', 'common', 'tcp', 'rotation_deg')} for arm in S.ARMS}
        series, steps = {}, []
        for s in range(spec['levels'][level]):
            data = _load(solve_path(group, level, s))
            timing = data['timing_ms'] * 1e-3
            undo = S.invert(tcp_delta(session_perturbation(level, s)))
            truth = {t: np.array([TRUTH[h][i].box_at(timing) @ T_box_rig[t] for h, i in keys]) for t in ('R0', 'H')}
            rig = {}
            for arm in S.ARMS:
                t, a = S.TARGET_OF_ARM[arm], data['arms'][arm]
                ok = np.asarray(a['ok'], bool)
                T_est = np.repeat(np.eye(4)[None], n, axis=0)
                rig[arm] = (ok, T_est)
                if not ok.any():
                    continue
                T_est[ok, :3, :3] = Rotation.from_quat(a['pose7'][ok, 3:]).as_matrix()
                T_est[ok, :3, 3] = a['pose7'][ok, :3]
                T_est = T_est @ undo @ S.invert(T_rig_tcp[t])
                rig[arm] = (ok, T_est)
                R_est, t_est = T_est[ok, :3, :3], T_est[ok, :3, 3]
                R_true, t_true = truth[t][ok, :3, :3], truth[t][ok, :3, 3]
                centroid, direction = points[t]
                dR = np.transpose(R_true, (0, 2, 1)) @ R_est
                acc[arm]['centroid'].append(point_errors_mm(R_est, t_est, R_true, t_true, centroid))
                acc[arm]['origin'].append(np.linalg.norm(t_est - t_true, axis=1) * 1e3)
                acc[arm]['common'].append(point_errors_mm(R_est, t_est, R_true, t_true, centroid + direction * common_m))
                acc[arm]['tcp'].append(np.linalg.norm(a['evec'][ok], axis=1))
                acc[arm]['rotation_deg'].append(np.degrees(Rotation.from_matrix(dR).magnitude()))
                if level == 'nominal':
                    d, r = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
                    d[ok] = (t_est - t_true) * 1e3
                    r[ok] = np.degrees(Rotation.from_matrix(dR).as_rotvec())
                    series[arm] = (ok, d, r)
            if level == 'nominal':
                both = rig['H0'][0] & rig['H1'][0]
                steps.append(np.linalg.norm(rig['H1'][1][both, :3, 3] - rig['H0'][1][both, :3, 3], axis=1) * 1e3)
        report['levels'][level] = {'arms': {arm: {k: _p50_p95(v) for k, v in acc[arm].items()} for arm in S.ARMS}}
        if level == 'nominal' and group == 'gripper':
            per_frame = {t: [sum(c > 0 for c in frame) for frame in decoded[t]] for t in decoded}
            views = {t: [c for frame in decoded[t] for c in frame if c > 0] for t in decoded}
            report['bench_crosscheck'] = {
                'bench': BENCH_0909,
                'noise_floor': noise_floor(keys, T_box_rig, series),
                'refinement_step_mm': _p50_p95(steps),
                'cameras_decoding': {t: {'p50': _q(v, 50), 'min': int(min(v))} for t, v in per_frame.items()},
                'single_marker_view_fraction': {t: float(np.mean(np.asarray(v) == 1)) if v else None
                                                for t, v in views.items()}}
    return report


def build_report(cfg, specs, selection, conv_keys, out_path: Path):
    started = time.time()
    rng = np.random.default_rng([cfg['seed'], 99])
    groups, checks_extra = {}, {}
    replay = None
    for group, spec in specs.items():
        CFG['_group'] = group
        keys = frame_keys(spec['episodes'], spec['stride'], cfg['max_per_block'])
        occ, decoded = render_occlusion(group, keys, spec['mounts'])
        speed, occlusion, masks = bucket_masks(group, keys, occ)
        levels, nominal = {}, None
        for level, sessions in spec['levels'].items():
            summary, data = level_summary(group, level, sessions, keys, masks, rng, cfg['bootstrap'])
            if level == 'nominal':
                nominal = data
                summary['tracking'] = rpe_and_gaps(keys, data, spec['stride'])
                gaps = data['extra']['H1'][0]['native_gap_m']
                checks_extra[group] = {'native_mismatch': int(np.sum(~(gaps < 1e-6))), 'frames': len(keys)}
            levels[level] = summary
            print(f'[report] {group}/{level} {time.time() - started:.0f}s', flush=True)
        mounts = {}
        for target in ('R0', 'H'):
            T_box_rig = S.mount_T_box_rig(SCENE, target, group, *spec['mounts'][target])
            T_rig_tcp = S.T_rig_tcp(SCENE, T_box_rig)
            mounts[target] = {'yaw_deg': spec['mounts'][target][0], 'pitch_deg': spec['mounts'][target][1],
                              'T_box_rig': T_box_rig.round(6).tolist(),
                              'socket_to_tcp_mm': float(np.linalg.norm(T_rig_tcp[:3, 3]) * 1e3),
                              'feature_centroid_to_tcp_mm': float(np.linalg.norm(
                                  np.mean([a.centre_m for a in SCENE.models[target].anchors], axis=0) - T_rig_tcp[:3, 3]) * 1e3)}
        groups[group] = {'episodes': spec['episodes'], 'stride': spec['stride'], 'frames': len(keys),
                         'interpolated_fraction': float(np.mean([TRUTH[h][i].interpolated for h, i in keys])),
                         'mounts': mounts, 'levels': levels,
                         'mechanism': mechanism_report(group, spec, keys, decoded),
                         'occlusion_mean': {t: float(np.mean(v)) for t, v in occ.items()},
                         'speed_p95_mps': _q(speed, 95)}
        if group == 'gripper':
            n = len(keys)
            ok, err, rot = nominal['raw']
            ok_s, err_s, _ = nominal['smoothed']
            replay = {
                'hand': [h for h, _ in keys], 'index': [int(i) for _, i in keys],
                'episode': [TRUTH[h][i].episode for h, i in keys], 't_s': [round(TRUTH[h][i].t, 4) for h, i in keys],
                'interpolated': [int(TRUTH[h][i].interpolated) for h, i in keys],
                'speed_mps': np.round(speed, 3).tolist(), 'occlusion': {t: np.round(v, 3).tolist() for t, v in occ.items()},
                'box_pose7': [np.round(pose7(TRUTH[h][i].T_world_box), 5).tolist() for h, i in keys],
                'decoded': decoded,
                'arms': {arm: {'ok': ok[arm][:n].astype(int).tolist(),
                               'error_mm': [round(float(x), 3) if o else None for o, x in zip(ok[arm][:n], err[arm][:n])],
                               'rotation_deg': [round(float(x), 3) if o else None for o, x in zip(ok[arm][:n], rot[arm][:n])],
                               'smoothed_error_mm': [round(float(x), 3) if o else None
                                                     for o, x in zip(ok_s[arm][:n], err_s[arm][:n])]}
                         for arm in S.ARMS}}
            examples = counterexamples(keys, nominal, specs, speed, occlusion)
    convergence = convergence_report(cfg, conv_keys)
    h_layout = S.load_json(S.HYBRID_PRODUCTION_LAYOUT)
    got = {m['id']: np.asarray(m['corners_rig']) for m in SCENE.layouts['H']['markers']}
    h_gap = max(float(np.max(np.abs(got[m['id']] - np.asarray(m['corners_rig'])))) for m in h_layout['markers'])
    command = ['node', str(HERE / 'export-world.mjs'), '{"poses":1,"episodes":1}']
    if SCENE.r0_status == 'measured':
        command.append(str(S.INPUTS / S.R0_MEASURED_INPUT))
    world = json.loads(subprocess.check_output(command, text=True))
    measured_doc = S.load_json(HERE / 'inputs_manifest.json')['targets']['R0'].get('measured_layout') or {}
    r0_geometry = {'status': SCENE.r0_status, 'layout_id': SCENE.layouts['R0'].get('layout_id'),
                   'origin': measured_doc.get('origin'), 'sha256': measured_doc.get('sha256')}
    js = {a['id']: np.asarray(a['points']) for a in world['rig']['anchors']}
    r0_gap = max(float(np.max(np.abs(js[m['id']] - np.asarray(m['corners_rig'])))) for m in SCENE.layouts['R0']['markers'])
    quadrants = {a.marker_id: a.paste_quadrant for a in SCENE.models['H'].anchors}
    checks = [
        {'id': 'h-layout', 'ok': h_gap == 0.0, 'blocking': True, 'label': '0907 anchor 角点与生产 marker_layout 文件逐点一致'},
        {'id': 'paste-quadrant', 'ok': quadrants == {30: 3, 31: 3, 32: 2}, 'blocking': True, 'label': '0907 贴纸转角 30→3 / 31→3 / 32→2'},
        {'id': 'r0-parity', 'ok': r0_gap < 1e-12, 'blocking': True, 'label': 'R0 几何与浏览器内核逐点一致'},
        {'id': 'native-parity', 'ok': all(v['native_mismatch'] == 0 for v in checks_extra.values()), 'blocking': True,
         'label': '名义 session 的 H1 复现生产 tracker 的逐帧位姿（' + ', '.join(
             f'{g}: {v["native_mismatch"]}/{v["frames"]} 帧不一致' for g, v in checks_extra.items()) + '）'},
        {'id': 'convergence', 'ok': convergence['pass'], 'blocking': True,
         'label': (f'渲染收敛：{convergence["supersample"]}× 与 {convergence["reference_supersample"]}× 超采样的梯度归一化边缘位移 p95 '
                   + ('—' if convergence['edge_shift_p95_px'] is None else f'{convergence["edge_shift_p95_px"]:.3f}')
                   + f' px（阈值 {convergence["threshold_px"]} px），解码不一致 {convergence["decode_mismatch"]}/{convergence["decodes"]}')},
        {'id': 'denominator', 'ok': True, 'blocking': True, 'label': '覆盖率、达标率、CDF 分母均为全部请求帧'},
        {'id': 'heldout', 'ok': True, 'blocking': True, 'label': f'安装方向只在 dev episode {cfg["dev_episodes"]} 选定；数值全部来自 episode {cfg["heldout_episodes"]}'},
        {'id': 'cameras', 'ok': SCENE.camera_doc['status'] == 'frozen-real', 'blocking': True, 'label': '真实 7 相机 K/D/外参快照（0804 fisheye，已哈希）'},
        {'id': 'task', 'ok': SCENE.task['status'] == 'frozen-real-task', 'blocking': True, 'label': '真实双手 manipulation 轨迹（132514，已哈希）'},
        {'id': 'mechanism', 'ok': True, 'blocking': True, 'label': '夹爪机构遮挡来自 URDF 网格'},
        {'id': 'measured-r0', 'ok': SCENE.r0_status == 'measured', 'blocking': True,
         'label': (f'measured R0 layout：{r0_geometry["layout_id"]}（{r0_geometry["origin"] or "inputs/r0_layout_measured.json"}，已哈希），真值与解算共用'
                   if SCENE.r0_status == 'measured' else 'measured R0 layout（未提供，R0 为 CAD+实测边长代理）')},
        {'id': 'housing-hand', 'ok': False, 'blocking': False, 'label': 'BOX 外壳与手部包络为声明假设（非实测网格）'},
        {'id': 'mount', 'ok': False, 'blocking': False, 'label': '靶标上夹爪的安装位为候选（同一球窝座，搜索 yaw × 臂仰角），未做结构设计'},
        {'id': 'extrinsics-0902', 'ok': False, 'blocking': False, 'label': '0902 生产外参只在 Thor；标定误差以 L3 扰动计入'},
    ]
    decision = decide(groups, checks)
    report = {
        'schema': 'rig_target_ab_l3/v1', 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'layer': 'L2 image loop + L3 estimator perturbation on real inputs',
        'config': {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'run'},
        'arms': {'R0': 'rig0818 · 5 ArUco · production corner BA', 'H0': '0907 · anchors only · production corner BA',
                 'H1': '0907 · anchors + facet edges · production carrier_frame_pose (seeded tracker)'},
        'selection': selection, 'groups': groups, 'replay': replay, 'counterexamples': examples,
        'convergence': convergence, 'checks': checks, 'decision': decision, 'r0_geometry': r0_geometry,
        'perturbations': {k: {'sigma_evidence': v[0], 'sigma_stress': v[0] * 2, 'scope': v[1], 'evidence': v[2]}
                          for k, v in PERTURBATIONS.items()},
        'production_config': {'carrier': CARRIER_CFG, 'corner_ba': BA_KW, 'offline_smoothing': SMOOTH_CFG},
        'provenance': {'inputs': SCENE.sources, 'code': {S.rel(p): S.sha256(p) for p in (
            HERE / 'l3_runner.py', HERE / 'ab_scene.py', HERE / 'freeze_inputs.py',
            S.KALIBR / 'metrology/hybrid_carrier.py', S.KALIBR / 'metrology/marker_rig_ba.py',
            S.KALIBR / 'metrology/trajectory_smoothing.py', S.HIKON / 'hybrid_carrier_tracking_in_robot_base.py')},
            'git': git_state(), 'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                             'scipy': scipy.__version__, 'opencv': cv2.__version__}},
        'limitations': [
            ('R0 uses the measured layout for truth and solver alike; the bracket between the pads is still drawn as declared rods.'
             if SCENE.r0_status == 'measured' else 'R0 is the CAD + measured-edge proxy; no measured layout was frozen into inputs/.'),
            'BOX housing and hand are declared envelopes; only the finger mechanism is a real mesh.',
            'The mount of either target on the gripper is a candidate (same socket stud, yaw and boom tilt chosen on dev data), not a design.',
            'The socket-at-TCP control moves only the socket onto the TCP point: each target keeps its dev-chosen yaw and tilt, and there is no own gripper.',
            'Images are ideal diffuse renders with 0.4 px blur and 1 DN noise; lighting, specular highlights and real print quality are not modelled.',
            'L3 re-solves with the estimator perturbed but keeps the nominal image association (edge search band 5 px).',
            f'Held-out frames are subsampled every {specs["gripper"]["stride"]} frames (60 fps capture).',
            'Truth is a smoothed cube trajectory used as a task-distribution proxy, not ground truth of the 132514 demo.',
        ],
    }
    out_path.write_text(json.dumps(_clean(report), ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'[report] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) in {time.time() - started:.0f}s', flush=True)
    return report


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-name', default='main')
    parser.add_argument('--cache', type=Path, default=S.ROOT / 'outputs/rig_target_ab')
    parser.add_argument('--out', type=Path, default=HERE / 'l3_report.json')
    parser.add_argument('--stages', default='select,render,native,solve,smooth,convergence,report')
    parser.add_argument('--workers', type=int, default=max(1, min(28, (os.cpu_count() or 4) - 2)))
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--supersample', type=int, choices=[2, 3, 4], default=4)
    parser.add_argument('--exposure-ms', type=float, default=4.0)
    parser.add_argument('--exposure-samples', type=int, default=3)
    parser.add_argument('--blur-sigma', type=float, default=0.4)
    parser.add_argument('--noise-sigma', type=float, default=1.0)
    parser.add_argument('--stride-dev', type=int, default=6)
    parser.add_argument('--stride-heldout', type=int, default=2)
    parser.add_argument('--stride-socket', type=int, default=4)
    parser.add_argument('--sessions-evidence', type=int, default=24)
    parser.add_argument('--sessions-stress', type=int, default=12)
    parser.add_argument('--sessions-socket', type=int, default=12)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--convergence-frames', type=int, default=12)
    parser.add_argument('--max-per-block', type=int, default=0, help='debug: cap frames per hand x episode')
    args = parser.parse_args()
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    run = args.cache / args.run_name
    cfg = {'run': str(run), 'seed': args.seed, 'supersample': args.supersample, 'exposure_ms': args.exposure_ms,
           'exposure_samples': args.exposure_samples, 'blur_sigma': args.blur_sigma, 'noise_sigma': args.noise_sigma,
           'stride_dev': args.stride_dev, 'stride_heldout': args.stride_heldout, 'stride_socket': args.stride_socket,
           'sessions_evidence': args.sessions_evidence, 'sessions_stress': args.sessions_stress,
           'sessions_socket': args.sessions_socket, 'bootstrap': args.bootstrap,
           'convergence_frames': args.convergence_frames, 'max_per_block': args.max_per_block,
           'dev_episodes': [0], 'heldout_episodes': [1, 2], 'solve_chunk': 24,
           'reference_supersample': min(8, 2 * args.supersample)}
    mount_doc = S.load_json(S.INPUTS / 'gripper_v2.json')['mount']['gripper']
    cfg['yaw_candidates'], cfg['pitch_candidates'] = mount_doc['yaw_candidates_deg'], mount_doc['pitch_candidates_deg']
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = {k: cfg[k] for k in ('seed', 'supersample', 'exposure_ms', 'exposure_samples', 'blur_sigma',
                                       'noise_sigma', 'stride_dev', 'stride_heldout', 'stride_socket', 'max_per_block')}
    fingerprint['scene'] = S.sha256(HERE / 'ab_scene.py')
    fingerprint['inputs'] = {name: S.sha256(S.INPUTS / name) for name in sorted(os.listdir(S.INPUTS))}
    fingerprint['carrier'] = CARRIER_CFG
    stamp = run / 'fingerprint.json'
    if stamp.exists() and json.loads(stamp.read_text()) != fingerprint:
        raise SystemExit(f'{run} was rendered with a different scene/config; use a new --run-name or delete it')
    stamp.write_text(json.dumps(fingerprint, indent=1))
    # Solve and smooth results are cheap next to rendering: when the estimator
    # settings change they are recomputed rather than silently reused.
    estimator = {'perturbations': {k: v[0] for k, v in PERTURBATIONS.items()}, 'levels': LEVEL_SCALE,
                 'corner_ba': BA_KW, 'smoothing': SMOOTH_CFG}
    estimator_stamp = run / 'estimator_fingerprint.json'
    if estimator_stamp.exists() and json.loads(estimator_stamp.read_text()) != estimator:
        for sub in ('solve', 'smooth'):
            shutil.rmtree(run / sub, ignore_errors=True)
        print('estimator settings changed: cached solve/smooth results cleared', flush=True)
    estimator_stamp.write_text(json.dumps(estimator, indent=1))
    _init_worker(cfg)
    stages = set(args.stages.split(','))
    pool = WorkerPool(args.workers, cfg)
    try:
        selection = stage_select(pool, cfg) if 'select' in stages or (run / 'select.json').exists() else None
        if selection is None:
            raise SystemExit('run the select stage first: the held-out mounts are frozen by it')
        specs = group_specs(cfg, selection['mounts'])
        if 'render' in stages:
            stage_render(pool, cfg, specs)
        if 'native' in stages:
            stage_native(pool, cfg, specs)
        if 'solve' in stages:
            stage_solve(pool, cfg, specs)
        if 'smooth' in stages:
            stage_smooth(pool, cfg, specs)
        conv_keys = stage_convergence(pool, cfg, specs) if ('convergence' in stages or 'report' in stages) else None
    finally:
        pool.shutdown()
    if 'report' in stages:
        build_report(cfg, specs, selection, conv_keys, args.out)


if __name__ == '__main__':
    main()
