#!/usr/bin/env python3
"""Phase 5 smoothing on the real 132514 poses: the CLI as it was run (left residual) vs a body-frame residual.

Runs `metrology.cli.smooth_pose_trajectory` with the phase 5 `smooth_v4_epaware` settings on the
recorded rig poses, once unmodified and once with the segment solver's perturbation moved to the
body frame, so gating and segmentation are the CLI's own. There is no ground truth here: this
reports how far each output moves from the raw poses, how far the two outputs differ, and how
rough each output is.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'third_party/opencv_kalibr'))
import metrology.trajectory_smoothing as TS  # noqa: E402
from metrology.cli import smooth_pose_trajectory as CLI  # noqa: E402

REVIEW = ROOT / 'outputs/metrology/phase5_contact_review_132514'
OUT = ROOT / 'outputs/rig_target_ab/followups_20260911/real_v4'
V4 = ['--input-prefix', 'rig_base', '--output-prefix', 'smooth_base', '--sigma-column', 'sigma_tcp_max_mm',
      '--max-reprojection-rmse-px', '8.0', '--measurement-sigma-deg', '0.5',
      '--velocity-change-sigma-mps', '0.15', '--velocity-change-sigma-degps', '120']
ORIGINAL = TS._smooth_segment
BANDS = (('slow', 0.0, 0.15), ('mid', 0.15, 0.45), ('fast', 0.45, np.inf))


def body_frame_segment(obs_T, timestamps_s, measurement_sigma_m, measurement_sigma_rad, velocity_change_sigma_mps,
                       velocity_change_sigma_radps, *, loss, f_scale, max_nfev):
    """TS._smooth_segment with T = T_obs * Exp(delta) instead of Exp(delta) * T_obs."""
    n = obs_T.shape[0]
    if n < 3:
        return obs_T.copy(), TS.SegmentOptimizationInfo(0, n, n, True, 0.0, 0.0, 0, 'segment shorter than 3 frames')
    dt = np.diff(timestamps_s)
    good = np.isfinite(dt) & (dt > TS._EPS)
    fill = float(np.median(dt[good])) if good.any() else 1.0
    dt = np.maximum(np.where(good, dt, fill), TS._EPS)

    def transforms(x):
        d = x.reshape(n, 6)
        return [obs_T[i] @ TS.se3_exp(d[i]) for i in range(n)]

    def residuals(x):
        d = x.reshape(n, 6)
        meas = d.copy()
        meas[:, :3] /= measurement_sigma_m[:, None]
        meas[:, 3:6] /= measurement_sigma_rad[:, None]
        Ts = transforms(x)
        rel = np.asarray([TS.se3_log(TS._invert(Ts[i]) @ Ts[i + 1]) / dt[i] for i in range(n - 1)])
        acc = rel[1:] - rel[:-1]
        acc[:, :3] /= max(float(velocity_change_sigma_mps), TS._EPS)
        acc[:, 3:6] /= max(float(velocity_change_sigma_radps), TS._EPS)
        return np.concatenate([meas.reshape(-1), acc.reshape(-1)])

    opt = least_squares(residuals, np.zeros(6 * n), loss=loss, f_scale=float(f_scale), max_nfev=int(max_nfev),
                        jac_sparsity=TS._jacobian_sparsity(n), tr_solver='lsmr', x_scale=1.0)
    info = TS.SegmentOptimizationInfo(0, n, n, bool(opt.success), float(opt.cost), float(opt.optimality),
                                      int(opt.nfev), str(opt.message))
    return np.asarray(transforms(opt.x)), info


def run(hand, frame):
    OUT.mkdir(parents=True, exist_ok=True)
    src = REVIEW / f'src/{hand}_full/marker_rig_ba.{hand}.with_episode.csv'
    out = OUT / f'{hand}.{frame}.csv'
    TS._smooth_segment = body_frame_segment if frame == 'body' else ORIGINAL
    sys.argv = ['smooth_pose_trajectory', '--input-csv', str(src), '--output-csv', str(out),
                '--summary-json', str(out.with_suffix('.summary.json'))] + V4
    try:
        CLI.main()
    finally:
        TS._smooth_segment = ORIGINAL
    return out


def load(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def num(rows, key):
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r.get(key, ''))
        except ValueError:
            pass
    return out


def xyz(rows, prefix):
    return np.stack([num(rows, f'{prefix}_{a}_m') for a in 'xyz'], axis=1)


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float('nan')


def contact_mask(hand, frames):
    rows = load(REVIEW / 'contact_windows_touch_strict_v2/contact_frames_60hz.csv')
    cols = list(rows[0])
    fcol = next((c for c in ('frame_index', 'global_frame', 'frame') if c in cols), None)
    ccol = next((c for c in cols if c.lower() in ('in_contact', 'contact', 'is_contact', f'{hand}_contact', f'contact_{hand}')), None)
    hcol = next((c for c in ('hand', 'side') if c in cols), None)
    if fcol is None or ccol is None:
        print(f'  contact file columns not recognised: {cols[:15]}')
        return None
    hit = set()
    for r in rows:
        if hcol and r[hcol] not in (hand, hand[0].upper(), hand.upper()):
            continue
        if str(r[ccol]).strip().lower() in ('1', 'true', 'yes'):
            hit.add(int(float(r[fcol])))
    return np.array([int(f) in hit for f in frames])


def main():
    for hand in ('left', 'right'):
        left_out = run(hand, 'left')
        body_out = run(hand, 'body')
        L, B = load(left_out), load(body_out)
        ref = load(REVIEW / f'src/{hand}_full/marker_rig_ba.{hand}.smooth_v4_epaware.csv')
        applied = (num(L, 'smoothing_applied') == 1) & (num(B, 'smoothing_applied') == 1)
        raw, pl, pb, pr = xyz(L, 'rig_base'), xyz(L, 'smooth_base'), xyz(B, 'smooth_base'), xyz(ref, 'smooth_base')
        repro = np.nanmax(np.abs(pl[applied] - pr[applied])) * 1e3
        frames, eps = num(L, 'frame_index'), num(L, 'episode_index')
        t = frames / 60.0
        idx = np.flatnonzero(applied)
        consecutive = np.zeros(len(L), bool)
        consecutive[1:-1] = ((frames[2:] - frames[1:-1]) == 1) & ((frames[1:-1] - frames[:-2]) == 1) & \
                            (eps[2:] == eps[1:-1]) & (eps[1:-1] == eps[:-2]) & applied[2:] & applied[1:-1] & applied[:-2]
        c = np.flatnonzero(consecutive)
        speed = np.full(len(L), np.nan)
        speed[c] = np.linalg.norm(pb[c + 1] - pb[c - 1], axis=1) / (t[c + 1] - t[c - 1])
        vel_l = (pl[c + 1] - pl[c - 1]) / (t[c + 1] - t[c - 1])[:, None]
        vel_b = (pb[c + 1] - pb[c - 1]) / (t[c + 1] - t[c - 1])[:, None]

        def rough(p):
            return np.linalg.norm(p[c + 1] - 2 * p[c] + p[c - 1], axis=1) * 1e3

        shift_l = np.linalg.norm(pl - raw, axis=1) * 1e3
        shift_b = np.linalg.norm(pb - raw, axis=1) * 1e3
        diff = np.linalg.norm(pl - pb, axis=1) * 1e3
        dist = np.linalg.norm(raw, axis=1)
        sl = json.loads(left_out.with_suffix('.summary.json').read_text())
        sb = json.loads(body_out.with_suffix('.summary.json').read_text())
        print(f'\n=== {hand}: {idx.size} smoothed frames, {sl.get("num_segments")} segments; '
              f'CLI(left) vs phase 5 smooth_v4_epaware max |d| {repro:.2e} mm')
        print(f'  rig origin |t| p50 {np.nanmedian(dist[idx]):.2f} m -> 0.5 deg * |t| = {np.deg2rad(0.5) * np.nanmedian(dist[idx]) * 1e3:.1f} mm'
              f'; sigma_tcp_max p50 {np.nanmedian(num(L, "sigma_tcp_max_mm")[idx]):.2f} mm')
        for name, s in (('left (as run)', shift_l), ('body frame', shift_b)):
            print(f'  shift from raw, {name:13s}: p50 {q(s[idx], 50):.2f}  p95 {q(s[idx], 95):.2f}  p99 {q(s[idx], 99):.2f}  max {q(s[idx], 100):.2f} mm')
        print(f'  |left - body|: p50 {q(diff[idx], 50):.2f}  p95 {q(diff[idx], 95):.2f}  p99 {q(diff[idx], 99):.2f}  max {q(diff[idx], 100):.2f} mm')
        for b, lo, hi in BANDS:
            m = np.isfinite(speed) & (speed >= lo) & (speed < hi)
            print(f'    {b:4s} ({m.sum():4d} frames): shift left p95 {q(shift_l[m], 95):.2f}, body p95 {q(shift_b[m], 95):.2f}, '
                  f'|left - body| p95 {q(diff[m], 95):.2f} mm')
        print(f'  2nd-difference roughness p95 mm: raw {q(rough(raw), 95):.3f}, left {q(rough(pl), 95):.3f}, body {q(rough(pb), 95):.3f}')
        dv = np.linalg.norm(vel_l - vel_b, axis=1) * 1e3
        print(f'  |velocity left - body| p50 {q(dv, 50):.1f}  p95 {q(dv, 95):.1f} mm/s')
        cm = contact_mask(hand, frames)
        if cm is not None:
            m = cm & applied
            print(f'  in contact ({m.sum()} frames): shift left p95 {q(shift_l[m], 95):.2f}, body p95 {q(shift_b[m], 95):.2f}, '
                  f'|left - body| p95 {q(diff[m], 95):.2f} max {q(diff[m], 100):.2f} mm')
        print(f'  nfev: left max {max(s["num_function_evals"] for s in sl["segments"])}, '
              f'body max {max(s["num_function_evals"] for s in sb["segments"])}')


if __name__ == '__main__':
    main()
