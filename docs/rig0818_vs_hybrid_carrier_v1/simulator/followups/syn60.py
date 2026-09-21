#!/usr/bin/env python3
"""Production-rate (60 Hz) checks of the offline smoother on the held-out truth TCP trajectories.

White position/rotation noise (the same draw for every variant of a noise level) is smoothed under
what each pipeline tells the smoother, with the production left residual or a body-frame (right)
residual. The `shift` rows repeat one case with the base origin moved by 1 m: a smoother whose
output depends on where the base origin sits is wrong by construction.
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import csv  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoother_ablation as A  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

NOISE = {'none': (0.0, 0.0), 'lo': (0.17, 0.09), 'hi': (0.60, 0.25)}  # per-axis sigma: mm, deg
HELDOUT = (1, 2)
SHIFT_M = np.array([1.0, 0.0, 0.0])


def phase5_sigma_m():
    vals = []
    for hand in ('left', 'right'):
        path = A.ROOT / f'outputs/metrology/phase5_contact_review_132514/src/{hand}_full/marker_rig_ba.{hand}.with_episode.csv'
        with path.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    v = float(row['sigma_tcp_max_mm'])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    vals.append(v)
    return max(5e-5, float(np.median(vals)) * 1e-3)


def configs():
    # name: (sigma_m, sigma_deg, velocity_change_sigma_mps, velocity_change_sigma_radps)
    return {
        'prod': (1e-4, 0.03, 0.03, 0.5),  # config_thor floors, typical BA sigma
        'prod_matched': ('noise', 'noise', 0.03, 0.5),  # sigma equal to the injected noise
        'phase5_v4': (phase5_sigma_m(), 0.5, 0.15, float(np.deg2rad(120.0))),  # CLI run behind the phase 5 review
    }


def run(spec):
    noise, cfg, sm, sd, vm, vr, frame, hand, ep, shift = spec
    rng = np.random.default_rng([list(NOISE).index(noise), 0 if hand == 'left' else 1, ep])
    frames = sorted(f for f, tr in A.TRUTH[hand].items() if tr.episode == ep)
    T_true = np.asarray([A.TRUTH[hand][f].box_at(0.0) @ A.SCENE.T_box_tcp for f in frames])
    t = np.asarray([A.TRUTH[hand][f].t for f in frames])
    n = len(frames)
    s_mm, s_deg = NOISE[noise]
    obs = T_true.copy()
    obs[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, np.deg2rad(s_deg), (n, 3))).as_matrix() @ T_true[:, :3, :3]
    obs[:, :3, 3] += rng.normal(0, s_mm * 1e-3, (n, 3))
    info = {}
    if frame == 'raw':
        est = obs
    else:
        off = SHIFT_M if shift else np.zeros(3)
        moved = obs.copy()
        moved[:, :3, 3] += off
        est, info = A.smooth_segment(moved, t, np.full(n, sm), np.full(n, np.deg2rad(sd)), vm, vr, 'soft_l1', 2.0, 50, frame)
        est = np.asarray(est)
        est[:, :3, 3] -= off
    evec = (est[:, :3, 3] - T_true[:, :3, 3]) * 1e3
    rot = np.asarray([A.rot_err_deg(est[i, :3, :3], T_true[i, :3, :3]) for i in range(n)])
    speed = np.asarray([A.S.tcp_speed(A.TRUTH[hand][f], A.SCENE.T_box_tcp) for f in frames])
    return dict(spec=spec, evec=evec, raw_evec=evec, rot=rot, speed=speed, t=t, stride=1, info=info)


def main():
    cfgs = configs()
    print('phase5_v4 translation sigma (median sigma_tcp_max_mm): %.3f mm' % (cfgs['phase5_v4'][0] * 1e3))
    specs = []
    blocks = [(hand, ep) for hand in ('left', 'right') for ep in HELDOUT]
    for noise, (s_mm, s_deg) in NOISE.items():
        specs += [(noise, 'raw', 0, 0, 0, 0, 'raw', hand, ep, False) for hand, ep in blocks]
        for name, (sm, sd, vm, vr) in cfgs.items():
            if name == 'prod_matched':
                if noise == 'none':
                    continue
                sm, sd = max(1e-5, s_mm * 1e-3), max(0.005, s_deg)
            specs += [(noise, name, sm, sd, vm, vr, frame, hand, ep, False) for frame in ('left', 'right') for hand, ep in blocks]
    sm, sd, vm, vr = cfgs['prod']
    specs += [('lo', 'prod', sm, sd, vm, vr, frame, hand, ep, True) for frame in ('left', 'right') for hand, ep in blocks]
    with ProcessPoolExecutor(16, mp_context=multiprocessing.get_context('spawn'), initializer=A._init) as ex:
        results = list(ex.map(run, specs))
    print('60 Hz, held-out episodes, TCP position p95 mm: all | slow / mid / fast ; rot p95 deg ; '
          'local p95 mm: 1-frame RPE, 200 ms RPE, 2nd-diff jitter')
    for noise in NOISE:
        for name in ['raw'] + list(cfgs):
            for frame in (('raw',) if name == 'raw' else ('left', 'right')):
                rows = [r for r in results if r['spec'][:2] == (noise, name) and r['spec'][6] == frame and not r['spec'][9]]
                if not rows:
                    continue
                err = np.concatenate([np.linalg.norm(r['evec'], axis=1) for r in rows])
                spd = np.concatenate([r['speed'] for r in rows])
                rot = np.concatenate([r['rot'] for r in rows])
                bands = {b: A.p95(err[(spd >= lo) & (spd < hi)]) for b, lo, hi in A.BANDS}
                rpe1, rpe200, jit = A.local_metrics(rows, raw=False)
                infos = [r['info'] for r in rows if r['info']]
                nfev = max((i.get('nfev', 0) for i in infos), default=0)
                cap = sum(1 for i in infos if 'maximum number' in i.get('message', ''))
                print(f'{noise:4s} {name:12s} {frame:5s} {A.p95(err):5.2f} | {bands["slow"]:5.2f} / {bands["mid"]:5.2f} / '
                      f'{bands["fast"]:5.2f} ; rot {A.p95(rot):.3f} ; {rpe1:.3f} {rpe200:.3f} {jit:.3f} | nfev<={nfev} cap {cap}/{len(infos)}')
    print('base origin moved by 1 m (noise lo, prod settings): output change after moving back, mm')
    for frame in ('left', 'right'):
        base = {(r['spec'][7], r['spec'][8]): r for r in results
                if r['spec'][:2] == ('lo', 'prod') and r['spec'][6] == frame and not r['spec'][9]}
        moved = {(r['spec'][7], r['spec'][8]): r for r in results
                 if r['spec'][:2] == ('lo', 'prod') and r['spec'][6] == frame and r['spec'][9]}
        d = np.concatenate([np.linalg.norm(moved[k]['evec'] - base[k]['evec'], axis=1) for k in base])
        print(f'  {frame:5s} p50 {np.median(d):.3f}  p95 {A.p95(d):.3f}  max {d.max():.3f}')


if __name__ == '__main__':
    main()
