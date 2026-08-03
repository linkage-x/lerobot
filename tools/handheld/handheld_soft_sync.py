#!/usr/bin/env python3

"""Offline soft-sync diagnostics for a raw handheld dataset.

The statistics live in :mod:`tools.shared.capture_timestamp_audit`, which is rig-independent.
What is handheld-specific stays here: the raw-capture sidecar, the CLI, and the report name.

The handheld rig has no device that is read on demand -- every sensor free-runs -- so it passes
no grid-lag reference and the frame grid is measured against the median across devices, which is
what this tool has always done.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.shared.capture_timestamp_audit import (  # noqa: E402
    DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
    DEFAULT_TOLERANCE_MS,
)
from tools.shared.capture_timestamp_audit import build_report as _build_capture_timestamp_report  # noqa: E402

# Written by the handheld recorder alongside the dataset; no other rig produces it.
RAW_CAPTURE_METADATA_FILENAME = "handheld_raw_capture.json"

__all__ = [
    "DEFAULT_GLOBAL_LAG_TOLERANCE_MS",
    "DEFAULT_TOLERANCE_MS",
    "build_report",
    "main",
    "parse_args",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline soft-sync diagnostics for a raw handheld LeRobot dataset. "
            "The input dataset is not modified."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Raw handheld dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to <dataset>/meta/handheld_soft_sync_report.json.",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help="Maximum allowed per-frame device timestamp skew.",
    )
    parser.add_argument(
        "--global-lag-tolerance-ms",
        type=float,
        default=DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
        help="Maximum allowed absolute median capture timestamp lag from the dataset timestamp.",
    )
    return parser.parse_args()


def build_report(
    *,
    dataset_root: Path,
    tolerance_ms: float,
    global_lag_tolerance_ms: float,
) -> dict[str, Any]:
    """Audit a handheld dataset. Same numbers, same report shape as before the split."""
    return _build_capture_timestamp_report(
        dataset_root=dataset_root,
        tolerance_ms=tolerance_ms,
        global_lag_tolerance_ms=global_lag_tolerance_ms,
        raw_capture_metadata_filename=RAW_CAPTURE_METADATA_FILENAME,
    )


def main() -> int:
    args = parse_args()
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms must be >= 0.")
    if args.global_lag_tolerance_ms < 0:
        raise ValueError("--global-lag-tolerance-ms must be >= 0.")

    dataset_root = args.dataset.resolve()
    output_path = args.output
    if output_path is None:
        output_path = dataset_root / "meta" / "handheld_soft_sync_report.json"
    report = build_report(
        dataset_root=dataset_root,
        tolerance_ms=args.tolerance_ms,
        global_lag_tolerance_ms=args.global_lag_tolerance_ms,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    summary = report["summary"]
    print(f"handheld_soft_sync_report={output_path}")
    print(f"total_frames={report['total_frames']}")
    print(f"devices={len(report['device_capture_timestamp_names'])}")
    print(f"nonfinite_capture_timestamp_frames={summary['nonfinite_capture_timestamp_frames']}")
    print(f"skew_over_tolerance_frames={summary['skew_over_tolerance_frames']}")
    print(f"global_lag_over_tolerance_frames={summary['global_lag_over_tolerance_frames']}")
    print(f"max_skew_s={summary['max_skew_s']['max']}")
    print(f"global_lag_p95_s={summary['global_lag_s']['p95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
