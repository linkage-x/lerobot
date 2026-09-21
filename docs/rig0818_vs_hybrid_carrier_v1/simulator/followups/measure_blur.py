#!/usr/bin/env python3
"""Estimate the real exposure time of the 132514 recording from motion blur.

For each candidate frame (select_blur_frames.py) the cube's image velocity v (px/s) comes from the
recorded trajectory projected through the 0804 cameras. A global-shutter exposure of duration E
smears the image with a box of length L = v * E along the motion direction, which widens only edges
whose gradient is parallel to the motion:  sigma^2 = s0^2 + (L^2 / 12) * cos^2(phi - theta).
Static frames of the same cameras are the control: L must come out near 0 there.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

BLUR = Path(__file__).resolve().parents[4] / 'outputs/rig_target_ab/followups_20260911/blur'
MKV = BLUR / 'mkv'
CROPS = BLUR / 'crops'
R, STEP = 7.0, 0.25
W1090 = 2.5631  # 10-90 % rise of a Gaussian edge, in sigma


def decode(ep, cam, frames):
    cap = cv2.VideoCapture(str(MKV / f'ep{ep}_{cam}.mkv'))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, k, last = {}, 0, max(frames)
    while k <= last:
        ok, img = cap.read()
        if not ok:
            break
        if k in frames:
            out[k] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k += 1
    cap.release()
    return out, count, k


def edge_widths(gray):
    """Per strong edge pixel: 10-90 % width (px) along the gradient and the gradient angle."""
    img = gray.astype(np.float64)
    g = gaussian_filter(img, 0.7)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > 0.3 * np.percentile(mag, 99.5))
    m = int(R) + 2
    keep = (xs > m) & (xs < img.shape[1] - m) & (ys > m) & (ys < img.shape[0] - m)
    xs, ys = xs[keep], ys[keep]
    nx, ny = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]
    ahead = map_coordinates(mag, [ys + ny, xs + nx], order=1)
    behind = map_coordinates(mag, [ys - ny, xs - nx], order=1)
    nms = (mag[ys, xs] >= ahead) & (mag[ys, xs] >= behind)
    xs, ys, nx, ny = xs[nms], ys[nms], nx[nms], ny[nms]
    offs = np.arange(-R, R + 1e-9, STEP)
    px = xs[:, None] + nx[:, None] * offs
    py = ys[:, None] + ny[:, None] * offs
    prof = map_coordinates(img, [py.ravel(), px.ravel()], order=1).reshape(px.shape)
    lo, hi = prof[:, :4].mean(1), prof[:, -4:].mean(1)
    con = hi - lo
    widths, angles = [], []
    c = len(offs) // 2
    for row, a, h, k in zip(prof, lo, hi, range(len(prof))):
        if h - a < 40:
            continue
        z = (row - a) / (h - a)
        if np.mean(np.diff(z) < -0.06) > 0.08:
            continue
        left = np.flatnonzero(z[:c + 1] < 0.1)
        right = np.flatnonzero(z[c:] > 0.9)
        if not left.size or not right.size:
            continue
        i = left[-1]
        j = c + right[0]
        if i + 1 >= len(z) or j - 1 < 0 or z[i + 1] == z[i] or z[j] == z[j - 1]:
            continue
        x10 = offs[i] + (0.1 - z[i]) / (z[i + 1] - z[i]) * STEP
        x90 = offs[j - 1] + (0.9 - z[j - 1]) / (z[j] - z[j - 1]) * STEP
        w = x90 - x10
        if 0.3 < w < 2 * R - 1.5:
            widths.append(w)
            angles.append(np.arctan2(ny[k], nx[k]))
    return np.asarray(widths), np.asarray(angles), int(np.count_nonzero(con > 40))


def blur_length(widths, angles, theta, rng, boot=300):
    c2 = np.cos(angles - theta) ** 2
    s2 = (widths / W1090) ** 2
    along, cross = s2[c2 > 0.75], s2[c2 < 0.25]
    if along.size < 15 or cross.size < 15:
        return None

    def est(a, b):
        return float(np.sqrt(max(0.0, 12.0 * (np.median(a) - np.median(b)))))

    L = est(along, cross)
    bs = [est(rng.choice(along, along.size), rng.choice(cross, cross.size)) for _ in range(boot)]
    return dict(L_px=L, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_along=int(along.size), n_cross=int(cross.size),
                w_along=float(np.median(widths[c2 > 0.75])), w_cross=float(np.median(widths[c2 < 0.25])))


def main():
    cand = json.loads((BLUR / 'candidates.json').read_text())
    CROPS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    need = {}
    for kind in ('fast', 'static'):
        for c in cand[kind]:
            need.setdefault((c['episode'], c['cam']), set()).add(c['frame'])
    frames = {}
    for (ep, cam), fs in sorted(need.items()):
        got, count, read = decode(ep, cam, fs)
        print(f'decoded ep{ep} {cam}: container frames {count}, read {read}, got {sorted(got)}')
        for f, img in got.items():
            frames[(ep, cam, f)] = img
    theta_of_cam = {c['cam']: np.radians(c['vdir_deg']) for c in cand['fast']}
    results = []
    for kind in ('fast', 'static'):
        for c in cand[kind]:
            img = frames.get((c['episode'], c['cam'], c['frame']))
            if img is None:
                print('missing frame', c)
                continue
            half = int(max(40, 0.8 * c['tag_px']))
            u, v = int(round(c['u'])), int(round(c['v']))
            crop = img[max(0, v - half):v + half, max(0, u - half):u + half]
            theta = np.radians(c['vdir_deg']) if kind == 'fast' else theta_of_cam[c['cam']]
            widths, angles, strong = edge_widths(crop)
            res = blur_length(widths, angles, theta, rng)
            row = dict(kind=kind, **{k: c[k] for k in ('hand', 'idx', 'episode', 'frame', 'cam', 'tag_px', 'vpx', 'speed_mps')},
                       edges=int(widths.size), strong=strong, **(res or {}))
            if res and kind == 'fast':
                row['E_ms'] = 1e3 * res['L_px'] / c['vpx']
                row['E_ci_ms'] = [1e3 * x / c['vpx'] for x in res['ci']]
            results.append(row)
            big = cv2.cvtColor(cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            ctr = np.array([big.shape[1] / 2, big.shape[0] / 2])
            if kind == 'fast':
                tip = ctr + 120 * np.array([np.cos(theta), np.sin(theta)])
                cv2.arrowedLine(big, tuple(int(x) for x in ctr), tuple(int(x) for x in tip), (0, 0, 255), 3, tipLength=0.2)
            label = f"{kind} {c['cam']} ep{c['episode']} f{c['frame']} v={c['vpx']:.0f}px/s"
            if res:
                label += f" L={res['L_px']:.1f}px"
            cv2.putText(big, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(str(CROPS / f"{c['cam']}_{kind}_ep{c['episode']}_f{c['frame']:04d}.png"), big)
    for r in results:
        if 'L_px' in r:
            extra = f" E {r['E_ms']:.1f} ms [{r['E_ci_ms'][0]:.1f}, {r['E_ci_ms'][1]:.1f}]" if 'E_ms' in r else ''
            print(f"{r['kind']:6s} {r['cam']} ep{r['episode']} f{r['frame']:4d} tag {r['tag_px']:4.0f}px v {r['vpx']:5.0f}px/s | "
                  f"edges {r['edges']:4d} (along {r['n_along']}, cross {r['n_cross']}) w10-90 along {r['w_along']:.2f} "
                  f"cross {r['w_cross']:.2f} px -> L {r['L_px']:.2f} px [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]{extra}")
        else:
            print(f"{r['kind']:6s} {r['cam']} ep{r['episode']} f{r['frame']:4d} too few edges ({r['edges']})")
    fast = [r for r in results if 'E_ms' in r]
    if fast:
        e = np.array([r['E_ms'] for r in fast])
        print(f'fast frames: E median {np.median(e):.1f} ms, IQR [{np.percentile(e, 25):.1f}, {np.percentile(e, 75):.1f}], n={e.size}')
    stat = [r['L_px'] for r in results if r['kind'] == 'static' and 'L_px' in r]
    if stat:
        print(f'static control: L median {np.median(stat):.2f} px, max {np.max(stat):.2f} px, n={len(stat)}')
    (BLUR / 'blur_results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
