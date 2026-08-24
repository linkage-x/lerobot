#!/usr/bin/env python3
"""Offline integrity audit of the M2020 per-taxel touch array.

Reads an already-recorded dataset's ``episodes/*/box_sensors.jsonl`` and asks
one question: **can the 9-slot ``fx/fy/fz_0p1N`` array be believed?**

The 2026-08-24 dedicated press experiment
(``thor_gmsl2_10ch_v1_20260824_123727``, four 20 s episodes, one pad each,
pressed corner-by-corner then edge-midpoints then centre) says no. The array
reassigns itself wholesale to a disjoint set of slots inside a single 16.7 ms
frame while (a) the packet spacing stays nominal, (b) the device's own
``total_force_0p1N`` walks smoothly through the discontinuity, and (c) the
BOX 6D force sensor sees no load change. A finger cannot shed 13 N and put it
back in one frame, so those transitions are a readout artefact, not mechanics.

That matters beyond a diagnostic, because every touch channel currently in
``observation.state`` is derived from this array
(:func:`tools.thor.gmsl2.thor_lerobot_v3._touch_summary` -> ``mean_fx/fy/fz``,
``max_abs_fz``, ``active_points``), while ``total_force_0p1N`` -- the one
channel that survives this audit -- is not exported at all.

What the checks below print, in order:

  A. per-pad taxel census (live%, max, share of normal load);
  B. hard switches: adjacent frames whose supports are disjoint, cross-checked
     against packet spacing, ``total_force`` continuity and 6D force;
  C. mode occupancy, which is what makes the pooled "corner live% 0-2%" and
     "centre saturated 50-73%" figures from ``..._20260817_162847`` invalid --
     they average two mutually exclusive regimes;
  D. ``sum(per-taxel fz)`` vs ``total_force_0p1N[z]`` agreement;
  E. saturation, which is a uint8 25.5 N ceiling on the *pad total*, not a
     property of the centre taxel.

Usage (dataset dir, local or NFS-mounted; no BOX or SDK needed)::

    python tools/thor/gmsl2/audit_m2020_taxels.py <dataset_dir> [--csv out.csv]

Section B doubles as the minimal reproduction to hand the pad vendor: each
line is one frame index where the array jumps and the device's own total does
not.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Slots that a 3x3 row-major reading puts at the corners / edge midpoints. The
# audit never assumes this mapping is correct -- it only uses it to label
# output, and section B's conclusion holds under any permutation.
CORNER_SLOTS = (0, 2, 6, 8)
EDGE_SLOTS = (1, 3, 5, 7)
CENTRE_SLOT = 4

# uint8 ceiling shared by the per-taxel fz and by total_force_0p1N[z].
FZ_CEILING = 255
# 0.1 N units. Contact gate for the statistics; well above the hardware
# deadband, which emits exact zeros (verified: 14168 untouched frames, no
# non-zero sample).
CONTACT_SUM = 40.0
# A support flip only counts as a hard switch if the pad was solidly loaded
# beforehand, so a release transient cannot masquerade as one.
SWITCH_MIN_SUM = 60.0
# 0.1 N. Below this a slot is indistinguishable from quantisation noise.
LIVE_FZ = 5.0


def _load(episode_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[Any]]]:
    """Split one episode's sensor log into touch frames and 6D force samples."""
    touch: dict[str, list[dict[str, Any]]] = {}
    six_d: dict[str, list[Any]] = {}
    with (episode_dir / "box_sensors.jsonl").open() as fh:
        for line in fh:
            if "touch" not in line and "six_d_force" not in line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sid", ""))
            data = rec.get("data") or {}
            if "box_touch_" in sid:
                touch.setdefault(sid, []).append(
                    {"t": rec["t_rel_s"], "mcu": rec["mcu_ts"], **data}
                )
            elif sid.endswith("box_six_d_force"):
                fxyz = (data.get("fxyz_mxyz_no_gravity") or [0.0] * 6)[:3]
                six_d.setdefault(sid.split("/")[0], []).append(
                    (rec["t_rel_s"], math.sqrt(sum(v * v for v in fxyz)))
                )
    return touch, six_d


def _interp(series: list[Any], t: float) -> float:
    """Nearest-sample lookup; the 6D stream runs ~8x the touch rate."""
    if not series:
        return math.nan
    lo, hi = 0, len(series) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    return series[lo][1]


def _support(fz: list[float]) -> tuple[int, ...]:
    return tuple(i for i, v in enumerate(fz) if v >= LIVE_FZ)


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def audit(dataset_dir: Path, csv_out: Path | None = None) -> int:
    episodes = sorted(p for p in (dataset_dir / "episodes").iterdir() if p.is_dir())
    if not episodes:
        raise SystemExit(f"no episodes under {dataset_dir}")

    pads: list[dict[str, Any]] = []
    for ep_idx, ep_dir in enumerate(episodes):
        touch, six_d = _load(ep_dir)
        for sid, frames in sorted(touch.items()):
            frames.sort(key=lambda f: f["t"])
            fz = [[float(v) for v in f["fz_0p1N"]] for f in frames]
            if not any(any(row) for row in fz):
                continue  # pad never touched in this episode
            box = sid.split("/")[0]
            pads.append({
                "ep": ep_idx, "sid": sid, "box": box,
                "t": [f["t"] for f in frames],
                "mcu": [f["mcu"] for f in frames],
                "model": frames[0].get("model"), "points": frames[0].get("points"),
                "fz": fz,
                "tfz": [float((f.get("total_force_0p1N") or [0, 0, 0])[2]) for f in frames],
                "six_d": six_d.get(box, []),
            })
    if not pads:
        raise SystemExit("no touched pad found; nothing to audit")

    print("=== A. per-pad taxel census ===")
    for pad in pads:
        fz = pad["fz"]
        n = len(fz)
        span = pad["t"][-1] - pad["t"][0]
        cols = list(zip(*fz, strict=True))
        total = sum(sum(c) for c in cols) or 1.0
        print(f"\n  ep{pad['ep']} {pad['sid']}  n={n} {n / span:.2f} Hz  "
              f"model={pad['model']} points={pad['points']}")
        print("    slot        " + " ".join(f"{i:6d}" for i in range(len(cols))))
        print("    live%(>=5)  " + " ".join(
            f"{sum(1 for v in c if v >= LIVE_FZ) / n * 100:6.1f}" for c in cols))
        print("    max         " + " ".join(f"{max(c):6.0f}" for c in cols))
        print("    load share% " + " ".join(f"{sum(c) / total * 100:6.2f}" for c in cols))

    print("\n=== B. hard switches (disjoint support in one frame) ===")
    print("  ep pad                    t(s)  dmcu(ms)   sum before->after   support before -> after"
          "        tfz before|after     |F6D| delta")
    rows: list[dict[str, Any]] = []
    for pad in pads:
        fz, t, mcu, tfz = pad["fz"], pad["t"], pad["mcu"], pad["tfz"]
        sums = [sum(r) for r in fz]
        for i in range(1, len(fz) - 5):
            if sums[i - 1] < SWITCH_MIN_SUM:
                continue
            before, after = set(_support(fz[i - 1])), set(_support(fz[i]))
            settled = set(_support(fz[i + 4]))
            # Require the new support to persist, so a single dropped/garbled
            # frame is not reported as a reassignment.
            if not before or before & after or before & settled or len(settled) < 2:
                continue
            pre = [v for tt, v in pad["six_d"] if t[i] - 0.12 <= tt < t[i]]
            post = [v for tt, v in pad["six_d"] if t[i] < tt <= t[i] + 0.12]
            d6 = (sum(post) / len(post) - sum(pre) / len(pre)) if pre and post else math.nan
            print(f"  {pad['ep']}  {pad['sid']:22s} {t[i]:6.2f} {(mcu[i] - mcu[i - 1]) / 1000:8.2f}"
                  f"   {sums[i - 1]:5.0f}->{sums[i]:4.0f}   "
                  f"{str(sorted(before)):16s}->{str(sorted(settled)):16s}"
                  f" {tfz[i - 1]:5.0f}|{tfz[i]:5.0f}   {d6:+6.2f} N")
            rows.append({"ep": pad["ep"], "sid": pad["sid"], "frame": i, "t_rel_s": t[i],
                         "dmcu_ms": (mcu[i] - mcu[i - 1]) / 1000,
                         "sum_before": sums[i - 1], "sum_after": sums[i],
                         "support_before": " ".join(map(str, sorted(before))),
                         "support_after": " ".join(map(str, sorted(settled))),
                         "tfz_before": tfz[i - 1], "tfz_after": tfz[i], "d_six_d_n": d6})
    print(f"\n  {len(rows)} hard switches. A switch with nominal dmcu, continuous tfz and a flat"
          f" 6D force is a readout artefact, not a load change.")

    print("\n=== C. mode occupancy (contact frames) ===")
    for pad in pads:
        contact = [r for r in pad["fz"] if sum(r) >= CONTACT_SUM]
        if not contact:
            continue
        with_centre = sum(1 for r in contact if r[CENTRE_SLOT] > 0)
        corner_only = sum(1 for r in contact
                          if r[CENTRE_SLOT] == 0 and any(r[c] > 0 for c in CORNER_SLOTS))
        share = [sum(r[c] for c in CORNER_SLOTS) / sum(r)
                 for r in contact if r[CENTRE_SLOT] == 0 and sum(r) > 0]
        share.sort()
        med = share[len(share) // 2] if share else 0.0
        print(f"  ep{pad['ep']} {pad['sid']:22s} n={len(contact):4d}  "
              f"centre-present={_pct(with_centre / len(contact))}  "
              f"centre-absent-with-corner={_pct(corner_only / len(contact))}  "
              f"corner share of those frames (median)={_pct(med)}")
    print("  Pooling these two regimes is what produced the retracted"
          " 'corner live% 0-2%' / 'centre saturated 50-73%' figures.")

    print("\n=== D. per-taxel sum vs device total_force_0p1N[z] ===")
    for pad in pads:
        pairs = [(sum(r), tf)
                 for r, tf in zip(pad["fz"], pad["tfz"], strict=True)
                 if sum(r) >= CONTACT_SUM]
        if not pairs:
            continue
        ratios = sorted(tf / s for s, tf in pairs)

        def q(frac: float, ratios: list[float] = ratios) -> float:
            return ratios[min(len(ratios) - 1, int(frac * len(ratios)))]

        print(f"  ep{pad['ep']} {pad['sid']:22s} tfz/sum  p10={q(0.10):.2f} "
              f"median={q(0.50):.2f} p90={q(0.90):.2f}")
    print("  A stable ratio would mean the array decomposes the total. It does not.")

    print("\n=== E. saturation (uint8 ceiling = 25.5 N) ===")
    for pad in pads:
        n = len(pad["fz"])
        centre_sat = sum(1 for r in pad["fz"] if r[CENTRE_SLOT] >= FZ_CEILING)
        total_sat = sum(1 for v in pad["tfz"] if v >= FZ_CEILING)
        print(f"  ep{pad['ep']} {pad['sid']:22s} centre fz==255: {_pct(centre_sat / n)}  "
              f"total_force==255: {_pct(total_sat / n)}  max centre fz={max(r[CENTRE_SLOT] for r in pad['fz']):.0f}")
    print("  total_force is the trustworthy channel but shares the ceiling:"
          " flag ==255 as censored, never use it as a value.")

    if csv_out is not None and rows:
        import csv
        with csv_out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} hard-switch rows to {csv_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_dir", type=Path, help="LeRobot v3 dataset root (contains episodes/)")
    ap.add_argument("--csv", type=Path, default=None, help="write the hard-switch table here")
    args = ap.parse_args()
    return audit(args.dataset_dir, args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
