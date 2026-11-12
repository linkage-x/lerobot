#!/usr/bin/env python
"""Has the side camera moved? Read off where the yellow tape square sits in its image (11-11).

The yellow tape square taped round the gear fixture is fixed to the table, so where it sits in
a side still is fixed too for as long as the camera does not move. The check finds the tape's
pixels in a reference still and in the current one and fits the rigid image motion (shift +
in-plane rotation) that carries one onto the other.

What makes it robust rather than a centroid comparison:
- The gear (orange, inside the square) is cut out by colour and as a filled blob, so a gear
  that crept or rotated does not read as a camera move; only the thin tape lines are used.
- The arm, the peg and their shadows cover parts of the tape in most stills. The fit scores
  each current tape pixel by its distance to the reference tape, *truncated*, so a covered
  side or a stray yellow pixel costs a constant instead of dragging the fit. A plain
  nearest-point ICP was tried first and slid 5-12 px along a side the peg covered.
- A coarse grid over +-40 px and +-3 degrees comes first, so a large move is not a local
  minimum away from the answer.

Validated (scratch, 11-11) on the 51 grasp-loop side snapshots 10-16..11-11 against
`grasp_20261027_104526` trial 0: every one within 2 px and 0.8 degrees (the camera has not
moved over that time), and synthetic shifts of 3/5/8/15 px with up to 2 degrees of rotation
come back within ~1 px / 0.2 degrees. The tape mask is binary, so the resolution is ~0.5-1 px.

The verdict thresholds are fixed here, not tuned per call: <= 3 px and <= 1 degree is "ok"
(history's spread plus the method's resolution), <= 8 px and <= 2 degrees is "shifted", beyond
that "moved". Too little tape visible, or a tape shape that does not match the reference after
the fit, is "unknown" -- not "ok".
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# HSV (OpenCV: H 0-180). Tape is H 17-20 under the lab light, the gear 5-11; S/V floors keep
# the teal table (H ~84, S ~60) and the white arm out.
TAPE_HSV_LOW = (13, 80, 120)
TAPE_HSV_HIGH = (36, 255, 255)
GEAR_HSV_RANGES = (((0, 90, 90), (11, 255, 255)), ((170, 90, 90), (180, 255, 255)))
# Anything warm that survives an opening this size is a solid part (the gear), not a tape line.
SOLID_OPEN_PX = 15
SOLID_MIN_AREA_PX = 800

SEARCH_SHIFT_PX = 40
SEARCH_ROT_DEG = 3.0
MAX_POINTS = 1500
TRUNCATE_PX = 6.0

OK_SHIFT_PX = 3.0
OK_ROT_DEG = 1.0
SHIFTED_SHIFT_PX = 8.0
SHIFTED_ROT_DEG = 2.0
MIN_TAPE_PIXELS = 300
MIN_COVERAGE = 0.4
MAX_SCORE_PX = 1.5

VERDICT_TEXT = {
    "ok": "未移动",
    "shifted": "轻微移动",
    "moved": "已移动",
    "unknown": "无法判断",
}


def _disc(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def tape_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = tape line) of the yellow tape, gear and other solid warm parts removed."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, TAPE_HSV_LOW, TAPE_HSV_HIGH)
    gear = np.zeros_like(yellow)
    for low, high in GEAR_HSV_RANGES:
        gear |= cv2.inRange(hsv, low, high)
    solid = cv2.morphologyEx(yellow | gear, cv2.MORPH_OPEN, _disc(SOLID_OPEN_PX))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    hull = np.zeros_like(yellow)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < SOLID_MIN_AREA_PX:
            continue
        contours = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contours))], 255)
    tape = yellow & ~cv2.dilate(hull, _disc(11)) & ~cv2.dilate(gear, _disc(7))
    return cv2.morphologyEx(tape, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _points(mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(mask)
    return np.stack([u, v], axis=1).astype(np.float64)


def _rotation(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _sample(field: np.ndarray, u: np.ndarray, v: np.ndarray, outside: float) -> np.ndarray:
    """Bilinear lookup of `field` at (u, v); `outside` beyond the image."""

    h, w = field.shape
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uc = np.clip(u, 0, w - 1.001)
    vc = np.clip(v, 0, h - 1.001)
    u0 = np.floor(uc).astype(int)
    v0 = np.floor(vc).astype(int)
    fu = uc - u0
    fv = vc - v0
    value = (
        field[v0, u0] * (1 - fu) * (1 - fv)
        + field[v0, u0 + 1] * fu * (1 - fv)
        + field[v0 + 1, u0] * (1 - fu) * fv
        + field[v0 + 1, u0 + 1] * fu * fv
    )
    return np.where(inside, value, outside)


@dataclass
class TapePose:
    """How the tape moved in the image, current relative to reference."""

    dxPx: float  # + = the tape appears further right than in the reference
    dyPx: float  # + = further down
    rotDeg: float  # + = turned counter-clockwise as displayed
    scorePx: float  # mean truncated distance of current tape pixels to the reference tape after the fit
    coverage: float  # share of the reference tape a current tape pixel lies on after the fit
    tapePixels: int
    referenceTapePixels: int
    atSearchEdge: bool

    @property
    def shiftPx(self) -> float:
        return math.hypot(self.dxPx, self.dyPx)


def fit_tape_pose(reference_mask: np.ndarray, current_mask: np.ndarray) -> TapePose | None:
    """Rigid motion of the tape from `reference_mask` to `current_mask`; None if either is empty."""

    ref = _points(reference_mask)
    cur = _points(current_mask)
    if len(ref) < 20 or len(cur) < 20:
        return None
    distance = cv2.distanceTransform((reference_mask == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float64)
    centre = ref.mean(axis=0)
    points = cur
    if len(points) > MAX_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)]
    offsets = points - centre

    # The fit maps current -> reference: q = R(a) (p - c) + c + t.
    def scores(angle: float, tx: np.ndarray, ty: np.ndarray, truncate: float) -> np.ndarray:
        q = offsets @ _rotation(angle).T + centre
        u = q[None, :, 0] + tx[:, None]
        v = q[None, :, 1] + ty[:, None]
        return np.minimum(_sample(distance, u, v, truncate), truncate).mean(axis=1)

    def search(angles: np.ndarray, shifts_x: np.ndarray, shifts_y: np.ndarray, truncate: float):
        gx, gy = np.meshgrid(shifts_x, shifts_y, indexing="ij")
        gx = gx.ravel()
        gy = gy.ravel()
        best = (math.inf, 0.0, 0.0, 0.0)
        for angle in angles:
            s = scores(float(angle), gx, gy, truncate)
            i = int(np.argmin(s))
            if s[i] < best[0]:
                best = (float(s[i]), float(angle), float(gx[i]), float(gy[i]))
        return best

    coarse = np.arange(-SEARCH_SHIFT_PX, SEARCH_SHIFT_PX + 1, 2.0)
    best = search(np.arange(-SEARCH_ROT_DEG, SEARCH_ROT_DEG + 1e-9, 0.5), coarse, coarse, TRUNCATE_PX)
    for step_px, step_deg, truncate in ((0.5, 0.2, 4.0), (0.125, 0.05, 3.0)):
        _, a0, x0, y0 = best
        ring = np.arange(-4, 5)
        best = search(a0 + step_deg * np.arange(-3, 4), x0 + step_px * ring, y0 + step_px * ring, truncate)
    score, angle, tx, ty = best

    # Coverage: how much of the reference tape the fitted current tape lies on.
    fitted = offsets @ _rotation(angle).T + centre + [tx, ty]
    h, w = reference_mask.shape
    landed = np.zeros_like(reference_mask)
    ui = np.rint(fitted[:, 0]).astype(int)
    vi = np.rint(fitted[:, 1]).astype(int)
    keep = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    landed[vi[keep], ui[keep]] = 255
    near = cv2.distanceTransform((cv2.dilate(landed, _disc(5)) == 0).astype(np.uint8), cv2.DIST_L2, 3)
    coverage = float((near[ref[:, 1].astype(int), ref[:, 0].astype(int)] < 3).mean())

    # Report the inverse (reference -> current): where the reference tape's centre shows up now.
    seen_at = _rotation(-angle) @ np.array([-tx, -ty]) + centre
    edge = abs(tx) >= SEARCH_SHIFT_PX - 1 or abs(ty) >= SEARCH_SHIFT_PX - 1 or abs(angle) >= SEARCH_ROT_DEG - 0.25
    return TapePose(
        dxPx=round(float(seen_at[0] - centre[0]), 2),
        dyPx=round(float(seen_at[1] - centre[1]), 2),
        # Image v points down, so the matrix's +angle is clockwise on screen; the reference->current
        # rotation is -angle, which is then counter-clockwise-positive as displayed.
        rotDeg=round(float(angle), 2),
        scorePx=round(float(score), 3),
        coverage=round(coverage, 3),
        tapePixels=int(len(cur)),
        referenceTapePixels=int(len(ref)),
        atSearchEdge=bool(edge),
    )


def verdict(pose: TapePose | None) -> tuple[str, str]:
    """(code, why) for a fitted pose."""

    if pose is None or pose.tapePixels < MIN_TAPE_PIXELS:
        return "unknown", "画面里能看到的黄色胶带太少（被遮挡，或不在画面里）"
    if pose.atSearchEdge:
        return "moved", f"胶带偏移超出搜索范围（±{SEARCH_SHIFT_PX} px / ±{SEARCH_ROT_DEG:g}°）"
    if pose.coverage < MIN_COVERAGE or pose.scorePx > MAX_SCORE_PX:
        return "unknown", "对齐后胶带形状与基准对不上（大面积遮挡，或胶带本身被改动）"
    if pose.shiftPx <= OK_SHIFT_PX and abs(pose.rotDeg) <= OK_ROT_DEG:
        return "ok", f"偏移 {pose.shiftPx:.1f} px、旋转 {pose.rotDeg:+.1f}°，在正常抖动范围内"
    if pose.shiftPx <= SHIFTED_SHIFT_PX and abs(pose.rotDeg) <= SHIFTED_ROT_DEG:
        return "shifted", f"偏移 {pose.shiftPx:.1f} px、旋转 {pose.rotDeg:+.1f}°，超出正常范围但不大"
    return "moved", f"偏移 {pose.shiftPx:.1f} px、旋转 {pose.rotDeg:+.1f}°"


def overlay(current_bgr: np.ndarray, reference_mask: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    """The current still, dimmed, with the reference tape in red and the current tape in green
    (where they coincide: yellow)."""

    out = (current_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    ref = cv2.dilate(reference_mask, _disc(3)) > 0
    cur = cv2.dilate(current_mask, _disc(3)) > 0
    out[ref & ~cur] = (40, 40, 255)
    out[cur & ~ref] = (60, 230, 60)
    out[ref & cur] = (0, 230, 255)
    return out


def check_images(reference_bgr: np.ndarray, current_bgr: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Fit, verdict and overlay for one pair of stills."""

    h, w = reference_bgr.shape[:2]
    ch, cw = current_bgr.shape[:2]
    if (ch, cw) != (h, w) and abs(cw / ch - w / h) < 0.01:
        # Same view at another size (a preview JPEG vs a full still): compare in reference pixels.
        current_bgr = cv2.resize(current_bgr, (w, h), interpolation=cv2.INTER_AREA if cw > w else cv2.INTER_LINEAR)
    if reference_bgr.shape[:2] != current_bgr.shape[:2]:
        return (
            {
                "verdict": "unknown",
                "verdictText": VERDICT_TEXT["unknown"],
                "reason": f"分辨率与基准不同（基准 {w}x{h}，当前 {current_bgr.shape[1]}x{current_bgr.shape[0]}）",
                "pose": None,
            },
            current_bgr,
        )
    reference_mask = tape_mask(reference_bgr)
    current_mask = tape_mask(current_bgr)
    pose = fit_tape_pose(reference_mask, current_mask)
    code, reason = verdict(pose)
    result = {
        "verdict": code,
        "verdictText": VERDICT_TEXT[code],
        "reason": reason,
        "pose": None if pose is None else {**asdict(pose), "shiftPx": round(pose.shiftPx, 2)},
        "thresholds": {
            "okShiftPx": OK_SHIFT_PX,
            "okRotDeg": OK_ROT_DEG,
            "shiftedShiftPx": SHIFTED_SHIFT_PX,
            "shiftedRotDeg": SHIFTED_ROT_DEG,
        },
    }
    return result, overlay(current_bgr, reference_mask, current_mask)


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the image")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("current", type=Path, nargs="+", help="stills to check against the reference")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    args = parser.parse_args()
    reference = cv2.imread(str(args.reference))
    if reference is None:
        raise SystemExit(f"cannot read {args.reference}")
    for path in args.current:
        image = cv2.imread(str(path))
        if image is None:
            print(json.dumps({"path": str(path), "error": "unreadable"}))
            continue
        result, picture = check_images(reference, image)
        print(json.dumps({"path": str(path), **result}, ensure_ascii=False))
        if args.overlay_dir is not None:
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            name = "_".join(path.parts[-2:]).replace(".png", "").replace(".jpg", "")
            cv2.imwrite(str(args.overlay_dir / f"{name}.jpg"), picture)


if __name__ == "__main__":
    main()
