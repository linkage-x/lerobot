#!/usr/bin/env python3
"""Statistics on BOX IMU quaternion hemisphere (sign flips) and yaw wrap.

Reads box_sensors.jsonl files and reports, per episode:
  - sample count, sign(w) split
  - true antipodal flips: dot(q[i], q[i+1]) < 0  (same rotation, opposite sign)
  - min |w| and how often |w| sits near 0 (where a w>=0 convention would itself
    introduce a discontinuity)
  - yaw wrap events (|yaw[i+1] - yaw[i]| > 180) for comparison

Usage: quat_hemisphere_stats.py <dataset_root_or_jsonl> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def episode_stats(path: Path, sid_filter: str) -> dict | None:
    """Stats for ONE sid stream. Multi-box episodes interleave several
    ``<box_id>/box_imu`` sids in the same file -- mixing them would fabricate
    per-sample sign flips, so callers pass an exact sid."""

    quats: list[list[float]] = []
    yaws: list[float] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("sid", "")) != sid_filter:
                continue
            data = rec.get("data") or {}
            q = data.get("quat_wxyz")
            if isinstance(q, list) and len(q) == 4:
                quats.append([float(v) for v in q])
            y = data.get("yaw_deg")
            if y is not None:
                yaws.append(float(y))
    if not quats:
        return None

    neg_w = sum(1 for q in quats if q[0] < 0.0)
    antipodal = 0
    max_step_raw = 0.0
    max_step_hemi = 0.0
    for a, b in zip(quats, quats[1:]):
        dot = sum(x * y for x, y in zip(a, b))
        if dot < 0.0:
            antipodal += 1
        # L2 step in raw stream vs after forcing w >= 0
        step_raw = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        ha = a if a[0] >= 0 else [-v for v in a]
        hb = b if b[0] >= 0 else [-v for v in b]
        step_hemi = sum((x - y) ** 2 for x, y in zip(ha, hb)) ** 0.5
        max_step_raw = max(max_step_raw, step_raw)
        max_step_hemi = max(max_step_hemi, step_hemi)

    abs_w = [abs(q[0]) for q in quats]
    norms = [sum(v * v for v in q) ** 0.5 for q in quats]

    yaw_wraps = sum(1 for a, b in zip(yaws, yaws[1:]) if abs(b - a) > 180.0)

    return {
        "path": str(path),
        "sid": sid_filter,
        "n": len(quats),
        "neg_w": neg_w,
        "neg_w_pct": 100.0 * neg_w / len(quats),
        "antipodal_flips": antipodal,
        "min_abs_w": min(abs_w),
        "n_abs_w_lt_0p05": sum(1 for v in abs_w if v < 0.05),
        "max_step_raw": max_step_raw,
        "max_step_hemi": max_step_hemi,
        "norm_min": min(norms),
        "norm_max": max(norms),
        "yaw_n": len(yaws),
        "yaw_wraps": yaw_wraps,
    }


def imu_sids(path: Path) -> list[str]:
    sids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(rec.get("sid", ""))
            if sid == "box_imu" or sid.endswith("/box_imu"):
                sids.add(sid)
    return sorted(sids)


def main(argv: list[str]) -> int:
    files: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_file():
            files.append(p)
        else:
            files.extend(sorted(p.rglob("box_sensors.jsonl")))
    if not files:
        print("no box_sensors.jsonl found", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        for sid in imu_sids(f):
            st = episode_stats(f, sid)
            if st:
                rows.append(st)

    hdr = f"{'episode[sid]':<86} {'n':>6} {'w<0':>7} {'flip':>5} {'min|w|':>7} {'|w|<.05':>8} {'dRaw':>6} {'dHemi':>6} {'yawWrap':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        parts = Path(r["path"]).parts
        short = f"{parts[-4]}/{parts[-2]}[{r['sid']}]"
        print(
            f"{short:<86} {r['n']:>6} {r['neg_w_pct']:>6.1f}% {r['antipodal_flips']:>5} "
            f"{r['min_abs_w']:>7.3f} {r['n_abs_w_lt_0p05']:>8} {r['max_step_raw']:>6.3f} "
            f"{r['max_step_hemi']:>6.3f} {r['yaw_wraps']:>8}"
        )

    tot_n = sum(r["n"] for r in rows)
    tot_flip = sum(r["antipodal_flips"] for r in rows)
    tot_neg = sum(r["neg_w"] for r in rows)
    tot_near0 = sum(r["n_abs_w_lt_0p05"] for r in rows)
    tot_yaw_wrap = sum(r["yaw_wraps"] for r in rows)
    print("-" * len(hdr))
    print(
        f"TOTAL episodes={len(rows)} samples={tot_n} w<0={tot_neg} ({100.0*tot_neg/tot_n:.1f}%) "
        f"antipodal_flips={tot_flip} |w|<0.05={tot_near0} yaw_wraps={tot_yaw_wrap}"
    )
    print(
        f"norm range over all: {min(r['norm_min'] for r in rows):.6f} .. "
        f"{max(r['norm_max'] for r in rows):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
