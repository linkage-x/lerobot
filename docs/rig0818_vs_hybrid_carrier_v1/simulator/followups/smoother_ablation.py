#!/usr/bin/env python3
"""Which part of the production offline smoother adds error in the L3 A/B?

Re-smooths the cached per-frame solves of one L3 session (no rendering) under
controlled variants and reports, per arm, TCP position p95 by speed band plus
local metrics (1-step and 200 ms relative error, second-difference jitter).
Truth variants feed noise-free truth poses through the same smoother, so any
error they show is model bias of the smoother itself.

From the repository root:
  python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/followups/smoother_ablation.py
  python3 .../smoother_ablation.py --level evidence --session 3 --groups gripper --only prod_copy,right
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(SIM))
import ab_scene as S  # noqa: E402  (puts opencv_kalibr on sys.path)
from scipy.optimize import least_squares  # noqa: E402

from metrology import trajectory_smoothing as TS  # noqa: E402
from metrology.metrics import matrix_to_pose7, pose7_to_matrix  # noqa: E402

RUN = ROOT / 'outputs/rig_target_ab/measured_r0_socketfix'
OUT = ROOT / 'outputs/rig_target_ab/followups_20260911/smoother_ablation_v2'
STRIDE = {'gripper': 2, 'socket_tcp': 4}
ARMS = ('R0', 'H0', 'H1')
CLI_VR = float(np.deg2rad(30.0))
TRUTH_SIGMA = ('const', 1e-4, 0.03)

# Production = config_thor offline_smoothing of the hikon / hybrid-carrier trackers.
PROD = dict(vm=0.03, vr=0.5, loss='soft_l1', f_scale=2.0, max_nfev=50, frame='left', dt_scale=None,
            sigma=('ba', 1e-4, 1e-2))
SOLVE_VARIANTS = {
    'prod_lib': dict(lib=True),
    'prod_copy': dict(),
    'nfev400': dict(max_nfev=400),
    'linear': dict(loss='linear'),
    'right': dict(frame='right'),
    'dt_sqrt': dict(dt_scale='sqrt'),
    'dt_lin': dict(dt_scale='lin'),
    'right_dt_sqrt': dict(frame='right', dt_scale='sqrt'),
    'const_sigma': dict(sigma=('const', 1e-4, 0.03)),
    'right_const_sigma': dict(frame='right', sigma=('const', 1e-4, 0.03)),
    # metrology/cli/smooth_pose_trajectory.py defaults: per-row position sigma (floor 5e-5 m),
    # a constant 0.5 deg rotation sigma and a 30 deg/s velocity-change sigma.
    'cli_left': dict(sigma=('ba_rotconst', 5e-5, 0.5), vr=CLI_VR),
    'cli_right': dict(frame='right', sigma=('ba_rotconst', 5e-5, 0.5), vr=CLI_VR),
}
TRUTH_VARIANTS = {
    'truth_prod': dict(sigma=TRUTH_SIGMA),
    'truth_right': dict(frame='right', sigma=TRUTH_SIGMA),
    'truth_dt_sqrt': dict(dt_scale='sqrt', sigma=TRUTH_SIGMA),
    'truth_prod_s1': dict(stride=1, sigma=TRUTH_SIGMA),
    'truth_right_s1': dict(stride=1, frame='right', sigma=TRUTH_SIGMA),
    'truth_cli_left_s1': dict(stride=1, sigma=('const', 1e-4, 0.5), vr=CLI_VR),
    'truth_cli_right_s1': dict(stride=1, frame='right', sigma=('const', 1e-4, 0.5), vr=CLI_VR),
}
BANDS = (('slow', 0.0, 0.15), ('mid', 0.15, 0.45), ('fast', 0.45, np.inf))
HEAVY = ('nfev400', 'truth_prod_s1', 'truth_right_s1', 'truth_cli_left_s1', 'truth_cli_right_s1')

SCENE = TRUTH = None
_DATA = {}


def _init(out=None):
    global SCENE, TRUTH, OUT
    if out:
        OUT = Path(out)
    SCENE = S.load_scene()
    TRUTH = {h: S.hand_truth(SCENE, h) for h in ('left', 'right')}


def _solve(group, level, session):
    key = (group, level, session)
    if key not in _DATA:
        with (RUN / 'solve' / group / f'{level}_{session:03d}.pkl').open('rb') as fh:
            _DATA[key] = pickle.load(fh)
    return _DATA[key]


def _job_path(spec):
    kind, variant, group, arm, hand, episode, level, session = spec
    return OUT / level / f'{session:03d}' / f'{kind}__{variant}__{group}__{arm}__{hand}__{episode}.pkl'


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))


def smooth_segment(obs_T, ts, sig_t, sig_r, vm, vr, loss, f_scale, max_nfev, frame):
    """Line-for-line copy of TS._smooth_segment with the perturbation side switchable."""
    n = obs_T.shape[0]
    dt = np.diff(ts)
    good = np.isfinite(dt) & (dt > 1e-12)
    fill = float(np.median(dt[good])) if good.any() else 1.0
    dt = np.maximum(np.where(good, dt, fill), 1e-12)
    jac = TS._jacobian_sparsity(n)

    def transforms(x):
        d = x.reshape(n, 6)
        if frame == 'left':
            return [TS.se3_exp(d[i]) @ obs_T[i] for i in range(n)]
        return [obs_T[i] @ TS.se3_exp(d[i]) for i in range(n)]

    def residuals(x):
        d = x.reshape(n, 6)
        meas = d.copy()
        meas[:, :3] /= sig_t[:, None]
        meas[:, 3:6] /= sig_r[:, None]
        Ts = transforms(x)
        rel = np.asarray([TS.se3_log(TS._invert(Ts[i]) @ Ts[i + 1]) / dt[i] for i in range(n - 1)])
        acc = rel[1:] - rel[:-1]
        acc[:, :3] /= max(vm, 1e-12)
        acc[:, 3:6] /= max(vr, 1e-12)
        return np.concatenate([meas.reshape(-1), acc.reshape(-1)])

    opt = least_squares(residuals, np.zeros(6 * n), loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                        jac_sparsity=jac, tr_solver='lsmr', x_scale=1.0)
    info = dict(success=bool(opt.success), status=int(opt.status), nfev=int(opt.nfev),
                optimality=float(opt.optimality), cost=float(opt.cost), message=str(opt.message))
    return np.asarray(transforms(opt.x)), info


def job(spec):
    kind, variant, group, arm, hand, episode, level, session = spec
    path = _job_path(spec)
    if path.exists():
        return str(path)
    t0 = time.time()
    data = _solve(group, level, session)
    keys = data['keys']
    timing = data['timing_ms'] * 1e-3
    cfg = {**PROD, **(SOLVE_VARIANTS if kind == 'solve' else TRUTH_VARIANTS)[variant]}
    member = [i for i, (h, idx) in enumerate(keys) if h == hand and TRUTH[h][idx].episode == episode]
    T_box_tcp = SCENE.T_box_tcp
    sigma_kind, sigma_m, sigma_deg = cfg['sigma']

    def true_tcp(f):
        return TRUTH[hand][f].box_at(timing) @ T_box_tcp

    if kind == 'solve':
        a = data['arms'][arm]
        use = [i for i in member if a['ok'][i]]
        frames = [keys[i][1] for i in use]
        pose7 = np.asarray(a['pose7'][use], float)
        ba_m = np.sqrt(np.mean(a['sp'][use] ** 2, axis=1))
        ba_deg = np.sqrt(np.mean(a['sr'][use] ** 2, axis=1))
        if sigma_kind == 'ba':
            sm, sd = np.maximum(ba_m, sigma_m), np.maximum(ba_deg, sigma_deg)
        elif sigma_kind == 'ba_rotconst':
            sm, sd = np.maximum(ba_m, sigma_m), np.full(len(use), sigma_deg)
        else:
            sm, sd = np.full(len(use), sigma_m), np.full(len(use), sigma_deg)
        eval_frames = frames
        stride = STRIDE[group]
        key_index = use
    else:
        stride = cfg.get('stride', STRIDE[group])
        key_frames = sorted(keys[i][1] for i in member)
        if stride == STRIDE[group]:
            frames = key_frames
        else:
            lo, hi = key_frames[0], key_frames[-1]
            frames = [f for f in sorted(TRUTH[hand]) if lo <= f <= hi and TRUTH[hand][f].episode == episode
                      and (f - lo) % stride == 0]
        pose7 = np.asarray([matrix_to_pose7(true_tcp(f)) for f in frames], float)
        sm, sd = np.full(len(frames), sigma_m), np.full(len(frames), sigma_deg)
        eval_frames = key_frames
        key_index = None
    t = np.asarray([TRUTH[hand][f].t for f in frames], float)
    k = {'sqrt': np.sqrt(stride), 'lin': float(stride)}.get(cfg['dt_scale'], 1.0)
    if cfg.get('lib'):
        res = TS.smooth_pose_trajectory(pose7, t, measurement_sigma_m=sm, measurement_sigma_rad=np.deg2rad(sd),
                                        velocity_change_sigma_mps=cfg['vm'] * k,
                                        velocity_change_sigma_radps=cfg['vr'] * k, loss=cfg['loss'],
                                        f_scale=cfg['f_scale'], max_nfev=cfg['max_nfev'])
        out_T = pose7_to_matrix(res.pose7)
        info = res.segments[0].as_dict() if res.segments else {}
    else:
        obs_T = pose7_to_matrix(TS._stabilize_quat_signs(pose7))
        out_T, info = smooth_segment(obs_T, t, sm, np.deg2rad(sd), cfg['vm'] * k, cfg['vr'] * k, cfg['loss'],
                                     cfg['f_scale'], cfg['max_nfev'], cfg['frame'])
    pos = {f: i for i, f in enumerate(frames)}
    evec, rot, speed, dist, t_eval = [], [], [], [], []
    for f in eval_frames:
        Tt = true_tcp(f)
        Te = out_T[pos[f]]
        evec.append((Te[:3, 3] - Tt[:3, 3]) * 1e3)
        rot.append(rot_err_deg(Te[:3, :3], Tt[:3, :3]))
        speed.append(float(S.tcp_speed(TRUTH[hand][f], T_box_tcp)))
        dist.append(float(np.linalg.norm(Tt[:3, 3])))
        t_eval.append(float(TRUTH[hand][f].t))
    if kind == 'solve':
        raw_evec = np.asarray(data['arms'][arm]['evec'][use], float)
        raw_rot = np.asarray(data['arms'][arm]['rot'][use], float)
        sr_lever = np.deg2rad(sd) * np.asarray(dist) * 1e3
    else:
        raw_evec, raw_rot, sr_lever = np.zeros((len(eval_frames), 3)), np.zeros(len(eval_frames)), np.zeros(0)
    result = dict(spec=spec, info=info, frames=eval_frames, key_index=key_index, evec=np.asarray(evec),
                  rot=np.asarray(rot), raw_evec=raw_evec, raw_rot=raw_rot, speed=np.asarray(speed),
                  dist=np.asarray(dist), t=np.asarray(t_eval), stride=int(STRIDE[group]),
                  sigma_m=sm, sigma_deg=sd, sr_lever_mm=sr_lever, seconds=time.time() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as fh:
        pickle.dump(result, fh)
    tmp.replace(path)
    return str(path)


def p95(x):
    x = np.asarray(x, float)
    return float(np.percentile(x, 95)) if x.size else float('nan')


def _pairs(t, gap_s, tol=0.3):
    """(i, j) with t[j] - t[i] within tol * gap_s of gap_s; t ascending."""
    out = []
    for i in range(len(t)):
        target = t[i] + gap_s
        k = int(np.searchsorted(t, target))
        best = min((c for c in (k - 1, k) if i < c < len(t)), key=lambda c: abs(t[c] - target), default=None)
        if best is not None and abs(t[best] - target) <= tol * gap_s:
            out.append((i, best))
    return out


def local_metrics(rows, raw):
    rpe1, rpe200, jit = [], [], []
    for r in rows:
        order = np.argsort(r['t'])
        t = np.asarray(r['t'])[order]
        e = np.asarray(r['raw_evec'] if raw else r['evec'], float)[order]
        step = r['stride'] / 60.0
        nxt = dict(_pairs(t, step))
        rpe1 += [np.linalg.norm(e[j] - e[i]) for i, j in nxt.items()]
        rpe200 += [np.linalg.norm(e[j] - e[i]) for i, j in _pairs(t, 0.2)]
        jit += [np.linalg.norm(e[nxt[j]] - 2 * e[j] + e[i]) for i, j in nxt.items() if j in nxt]
    return p95(rpe1), p95(rpe200), p95(jit)


def summarize(level, session, groups):
    rows = {}
    for path in sorted((OUT / level / f'{session:03d}').glob('*.pkl')):
        with path.open('rb') as fh:
            r = pickle.load(fh)
        kind, variant, group, arm = r['spec'][:4]
        rows.setdefault((group, variant, arm), []).append(r)
    summary = {}
    for group in groups:
        cached = None
        cpath = RUN / 'smooth' / group / f'{level}_{session:03d}.pkl'
        if cpath.exists():
            with cpath.open('rb') as fh:
                cached = pickle.load(fh)
        print(f'\n=== {group} {level}_{session:03d} (stride {STRIDE[group]}) — TCP position p95 mm: all | slow / mid / fast'
              f' ; rot p95 deg ; local p95 mm: 1-step RPE, 200 ms RPE, 2nd-diff jitter')
        for variant in ['raw'] + list(SOLVE_VARIANTS) + list(TRUTH_VARIANTS):
            for arm in (('truth',) if variant.startswith('truth') else ARMS):
                src = rows.get((group, 'prod_copy' if variant == 'raw' else variant, arm))
                if not src:
                    continue
                raw = variant == 'raw'
                err = np.concatenate([np.linalg.norm(r['raw_evec'] if raw else r['evec'], axis=1) for r in src])
                spd = np.concatenate([r['speed'] for r in src])
                rot = np.concatenate([r['raw_rot'] if raw else r['rot'] for r in src])
                bands = {name: p95(err[(spd >= lo) & (spd < hi)]) for name, lo, hi in BANDS}
                rpe1, rpe200, jit = local_metrics(src, raw)
                extra = ''
                if not raw:
                    infos = [r['info'] for r in src if r['info']]
                    nfev = max((i.get('nfev', i.get('num_function_evals', 0)) for i in infos), default=0)
                    cap = sum(1 for i in infos if 'maximum number' in str(i.get('message', '')))
                    extra = f' | nfev<={nfev} cap {cap}/{len(infos)}'
                if variant == 'prod_lib' and cached is not None:
                    d = max(np.max(np.abs(np.linalg.norm(cached[arm]['evec'][r['key_index']], axis=1)
                                          - np.linalg.norm(r['evec'], axis=1))) for r in src)
                    extra += f' | vs cached {d:.1e}'
                if variant == 'prod_copy':
                    lever = np.concatenate([r['sr_lever_mm'] for r in src])
                    sig = np.concatenate([r['sigma_m'] for r in src]) * 1e3
                    dist = np.concatenate([r['dist'] for r in src])
                    extra += f' | sig_t p50 {np.median(sig):.3f}, sig_r*|t| p50 {np.median(lever):.3f} mm, |t| p50 {np.median(dist):.2f} m'
                print(f'{variant:>19s} {arm:>5s}  {p95(err):5.2f} | {bands["slow"]:5.2f} / {bands["mid"]:5.2f} / '
                      f'{bands["fast"]:5.2f} ; rot {p95(rot):.3f} ; {rpe1:.3f} {rpe200:.3f} {jit:.3f}{extra}')
                summary.setdefault(group, {}).setdefault(variant, {})[arm] = dict(
                    p95=p95(err), bands=bands, rot_p95=p95(rot), rpe1_p95=rpe1, rpe200_p95=rpe200, jitter_p95=jit,
                    frames=int(err.size))
    (OUT / level / f'{session:03d}' / 'summary.json').write_text(json.dumps(summary, indent=1))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='nominal')
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--groups', default='socket_tcp,gripper')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', default='', help='comma list of variants to run (default all)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    OUT = Path(args.out)
    groups = args.groups.split(',')
    if not args.summary_only:
        _init(str(OUT))
        only = set(filter(None, args.only.split(',')))
        specs = []
        for group in groups:
            data = _solve(group, args.level, args.session)
            blocks = sorted({(h, TRUTH[h][i].episode) for h, i in data['keys']})
            for variant in SOLVE_VARIANTS:
                if not only or variant in only:
                    specs += [('solve', variant, group, arm, hand, ep, args.level, args.session)
                              for arm in ARMS for hand, ep in blocks]
            for variant in TRUTH_VARIANTS:
                if not only or variant in only:
                    specs += [('truth', variant, group, 'truth', hand, ep, args.level, args.session)
                              for hand, ep in blocks]
        todo = sorted((s for s in specs if not _job_path(s).exists()), key=lambda s: (s[1] not in HEAVY, s))
        print(f'{len(specs)} jobs, {len(todo)} to run', flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init, initargs=(str(OUT),)) as ex:
            futs = [ex.submit(job, s) for s in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if n % 25 == 0 or n == len(futs):
                    print(f'[ablation] {n}/{len(futs)} {time.time() - t0:.0f}s', flush=True)
    summarize(args.level, args.session, groups)
    print('ablation exit=0', flush=True)


if __name__ == '__main__':
    main()
