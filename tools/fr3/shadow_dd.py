"""Data-driven cast-shadow estimate for the FR3 ee camera (11-06).

Why: the 11-05 relight finetune (relight_aug.py, MuJoCo shadow maps of two fitted lamps) removed the lift-retry under
moved light but kept failing on some frames; overlaying the maps on those frames showed the sharpest real shadows
(peg + finger silhouettes) are not where the simulated ones are (explained variance only ~0.3-0.4 even after the
constant shift), so the policy's actual cue survived the augmentation. Here the shadow is read off the image itself:

    table   = pixels that are neither the fingers (fixed sprite while open), nor warm-coloured (pink peg, red/yellow
              gear, tape), nor saturated, nor very dark relative to the lit table (peg-top hole, gaps)
    lit_c   = per-channel upper envelope of the table (dilate 71 px disc, blur 30 px; computed at half res) = the
              unshadowed table colour
    r_c     = I_c / lit_c on table pixels, 1 elsewhere, clipped to [R_MIN, 1]

so I / r is the frame with the cast shadows on the table removed (colour-correct: shadowed table is bluer), and
I / r * r'^g re-shadows it with any other ratio map r' (shifted own map, a sim map, another frame's map).
Shadows falling on the peg or fingers themselves are not modelled.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

R_MIN = 0.35           # ratio below this on "table" pixels is an object (peg top hole, finger gap), not a shadow
SPRITE = Path(__file__).resolve().parent / "assets/ee_finger_sprite.png"   # copy of outputs/real2sim_20261104/s1_twin_r05/finger_sprite.png
_spr_cache: dict[tuple[int, int], np.ndarray] = {}


def finger_mask(h: int, w: int) -> np.ndarray:
    if (h, w) not in _spr_cache:
        a = cv2.imread(str(SPRITE), cv2.IMREAD_UNCHANGED)[..., 3]
        m = cv2.resize(a, (w, h)) > 12
        _spr_cache[(h, w)] = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return _spr_cache[(h, w)]


def shadow_ratio(img: np.ndarray, fingers: bool = True, fill: bool = False) -> np.ndarray:
    """img (H, W, 3) RGB uint8 or float 0..255 -> r (H, W, 3) float32 in [R_MIN, 1].

    fill=True: inside the object mask r is interpolated from the surrounding table (normalized convolution) instead
    of 1, so a shifted copy does not carry bright peg / finger-shaped holes ("ghost pegs") into the shadow."""
    f = img.astype(np.float32) + 1.0
    H, W = f.shape[:2]
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    warm = (R > G + 12) | ((R > B + 25) & (G > B + 25))
    obj = cv2.dilate(warm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) | (f.max(-1) >= 250)
    if fingers:
        obj |= finger_mask(H, W)
    table = ~obj
    # lit envelope at half resolution (36 px disc, blur 15 px there = the full-res 71 / 30), then upsampled
    h2, w2 = (H + 1) // 2, (W + 1) // 2
    ft = cv2.resize(np.where(table[..., None], f, 0).astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    ok = cv2.resize(table.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA) > 0.99
    ft[~ok] = 0
    e = cv2.dilate(ft, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
    lit = cv2.resize(cv2.GaussianBlur(e, (0, 0), 15), (W, H), interpolation=cv2.INTER_LINEAR)
    r = f / np.maximum(lit, f)
    dark = r.mean(-1) < R_MIN
    obj = ~table | dark
    r[obj] = 1.0
    r = cv2.medianBlur(np.ascontiguousarray(r, dtype=np.float32), 5)
    if fill:
        w = (~obj).astype(np.float32)
        num = cv2.GaussianBlur(r * w[..., None], (0, 0), 12); den = cv2.GaussianBlur(w, (0, 0), 12)[..., None]
        r = np.where(obj[..., None], np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0), r)
    return np.clip(r, R_MIN, 1.0).astype(np.float32)


def shift(r: np.ndarray, du: float, dv: float) -> np.ndarray:
    M = np.float32([[1, 0, du], [0, 1, dv]])
    return cv2.warpAffine(r, M, (r.shape[1], r.shape[0]), borderMode=cv2.BORDER_REFLECT_101)


def relight(img: np.ndarray, r_old: np.ndarray, r_new: np.ndarray | None = None, g_new: float = 1.0) -> np.ndarray:
    """uint8 RGB -> uint8 RGB: remove r_old, apply r_new ** g_new."""
    out = img.astype(np.float32) / r_old
    if r_new is not None:
        out = out * r_new ** g_new
    return out.round().clip(0, 255).astype(np.uint8)
