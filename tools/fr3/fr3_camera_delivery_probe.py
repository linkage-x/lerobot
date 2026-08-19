#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the cameras actually delivered during a recording, read back out of the dataset.

``fr3_sync_audit.py`` judges alignment; this explains it, from the same column and without the
rig. Every camera column in ``observation.device_capture_timestamp`` is an acquisition instant,
so two things fall out of it that the audit does not report:

**Frame reuse.** A repeated value means the loop wrote the *same image* to two frames because no
new one had arrived. The skew numbers do not show this at all -- a reused frame is perfectly
aligned with itself -- but it is the more damaging failure: the images stand still while state
and action keep moving, and a policy trained on it learns that they are unrelated.

**Real delivery rate.** The gaps between *distinct* values are the sensor's own intervals. A rate
at nominal says the cameras were healthy and the alignment failure is in selection or stamping; a
rate at half nominal says they were not, and no amount of budget tuning will fix the take.

What this cannot tell you is *why* the rate fell -- a frame the sensor never produced and one the
host dropped leave the same hole here. That needs the frame counter, which only exists live:
``fr3_camera_delivery_bench.py``.

Usage: ``python tools/fr3/fr3_camera_delivery_probe.py <dataset_root> [--camera-fps 60]``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.shared.capture_timestamp_audit import (  # noqa: E402
    capture_timestamp_names,
    load_dataset_info,
    read_dataset_rows,
)

ARM_COLUMN = "fr3.arm.capture_timestamp_s"


def summarize_camera_delivery(
    camera_capture_s: np.ndarray,
    arm_capture_s: np.ndarray,
    *,
    camera_fps: float,
) -> dict[str, Any]:
    """Reuse, delivery intervals and staleness for one camera over one episode.

    Delivery intervals are histogrammed in units of the nominal sensor period rather than in
    milliseconds, because that is the shape that names the fault: a sensor running at half rate
    puts every interval at 2 periods, while a host that stalls occasionally leaves a mostly-1
    histogram with a tail.
    """
    ticks = int(camera_capture_s.shape[0])
    delta_s = np.diff(camera_capture_s)
    fresh = delta_s > 1e-6
    gaps_s = delta_s[fresh]
    stale_s = arm_capture_s - camera_capture_s

    period_s = 1.0 / camera_fps if camera_fps > 0 else 0.0
    histogram: dict[str, int] = {}
    if period_s and gaps_s.size:
        multiples = np.clip(np.round(gaps_s / period_s).astype(int), 0, 99)
        for multiple in sorted(set(multiples.tolist())):
            histogram[f"{multiple}x"] = int(np.sum(multiples == multiple))

    return {
        "ticks": ticks,
        "delivered_frames": int(np.sum(fresh)),
        "reused_fraction": float(1.0 - fresh.mean()) if fresh.size else 0.0,
        "gap_ms": {
            "p50": float(np.median(gaps_s)) * 1e3 if gaps_s.size else 0.0,
            "p95": float(np.percentile(gaps_s, 95)) * 1e3 if gaps_s.size else 0.0,
            "max": float(gaps_s.max()) * 1e3 if gaps_s.size else 0.0,
        },
        "effective_fps": 1.0 / float(np.median(gaps_s)) if gaps_s.size else 0.0,
        "gap_histogram_periods": histogram,
        "stale_vs_arm_ms": {
            "p50": float(np.median(stale_s)) * 1e3,
            "p95": float(np.percentile(stale_s, 95)) * 1e3,
            "max": float(stale_s.max()) * 1e3,
        },
    }


def summarize_bursts(over_budget: np.ndarray) -> dict[str, Any]:
    """How the out-of-budget frames are distributed in time.

    A steady fault (a stamping regression, a camera stuck at half rate) spreads its bad frames
    evenly; a load fault arrives in bursts. The two call for different fixes, and the per-episode
    counts in the sync report cannot tell them apart.
    """
    if not over_budget.any():
        return {"bursts": 0, "longest_burst": 0, "worst_decile_share": 0.0}
    changes = np.diff(over_budget.astype(np.int8))
    starts = int(np.sum(changes == 1)) + int(over_budget[0])
    lengths: list[int] = []
    current = 0
    for flag in over_budget:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    deciles = np.array_split(over_budget, 10)
    per_decile = [int(np.sum(decile)) for decile in deciles]
    return {
        "bursts": starts,
        "longest_burst": max(lengths) if lengths else 0,
        "worst_decile_share": max(per_decile) / max(1, int(np.sum(over_budget))),
        "per_decile": per_decile,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="Dataset root (the one holding meta/info.json).")
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=60.0,
        help="Configured sensor rate; sets the period the delivery histogram is expressed in.",
    )
    parser.add_argument(
        "--skew-budget-ms",
        type=float,
        default=20.0,
        help="Cross-camera budget, for the burst analysis. Matches camera_max_skew_ms.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    info = load_dataset_info(args.dataset)
    names = capture_timestamp_names(info)
    rows = read_dataset_rows(args.dataset)
    capture, episodes = rows["capture_timestamps"], rows["episode_index"]

    if ARM_COLUMN not in names:
        raise SystemExit(f"{args.dataset} has no {ARM_COLUMN} column; this probe is FR3-specific.")
    arm_index = names.index(ARM_COLUMN)
    camera_indices = [index for index, name in enumerate(names) if name.startswith("camera.")]
    if len(camera_indices) < 1:
        raise SystemExit(f"{args.dataset} carries no camera capture-timestamp columns.")

    print(
        f"dataset fps={info.get('fps')}  sensor fps={args.camera_fps:g}  "
        f"cameras={[names[index] for index in camera_indices]}"
    )
    for episode in sorted(set(episodes.tolist())):
        selected = episodes == episode
        arm = capture[selected, arm_index]
        print(f"\nepisode {episode}  ({int(selected.sum())} frames)")
        for index in camera_indices:
            summary = summarize_camera_delivery(
                capture[selected, index], arm, camera_fps=args.camera_fps
            )
            gap, stale = summary["gap_ms"], summary["stale_vs_arm_ms"]
            print(f"  {names[index]}")
            print(
                f"    reused frame   {100 * summary['reused_fraction']:5.1f}% of ticks "
                f"({summary['delivered_frames']} distinct frames for {summary['ticks']} ticks)"
            )
            print(
                f"    delivery gap   p50 {gap['p50']:6.1f}  p95 {gap['p95']:6.1f}  "
                f"max {gap['max']:6.1f} ms   -> {summary['effective_fps']:5.1f} fps effective"
            )
            print(f"    in periods     {summary['gap_histogram_periods']}")
            print(
                f"    stale vs arm   p50 {stale['p50']:6.1f}  p95 {stale['p95']:6.1f}  "
                f"max {stale['max']:6.1f} ms"
            )

        if len(camera_indices) >= 2:
            members = capture[np.ix_(selected, camera_indices)]
            skew_ms = (members.max(axis=1) - members.min(axis=1)) * 1e3
            bursts = summarize_bursts(skew_ms > args.skew_budget_ms)
            print(
                f"  cross-camera skew > {args.skew_budget_ms:g} ms on "
                f"{int(np.sum(skew_ms > args.skew_budget_ms))} frame(s) in {bursts['bursts']} "
                f"burst(s), longest {bursts['longest_burst']}"
            )
            print(f"    per tenth of the episode: {bursts.get('per_decile')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
