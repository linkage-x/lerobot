#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.calibration.fr3_teleop import compare_pose_traces, load_trace_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FR3 teleop traces and print scale suggestions.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--current-scale-x", type=float, default=None)
    parser.add_argument("--current-scale-y", type=float, default=None)
    parser.add_argument("--current-scale-z", type=float, default=None)
    parser.add_argument("--current-scale-wx", type=float, default=None)
    parser.add_argument("--current-scale-wy", type=float, default=None)
    parser.add_argument("--current-scale-wz", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = load_trace_bundle(args.reference)
    measured = load_trace_bundle(args.measured)
    result = compare_pose_traces(reference, measured)

    current_scales = {
        "x": args.current_scale_x,
        "y": args.current_scale_y,
        "z": args.current_scale_z,
        "wx": args.current_scale_wx,
        "wy": args.current_scale_wy,
        "wz": args.current_scale_wz,
    }

    print(f"reference_mode={result['reference_mode']}")
    print(f"measured_mode={result['measured_mode']}")
    for axis_name in ("x", "y", "z"):
        axis_summary = result["translation_axis_summaries"].get(axis_name)
        if axis_summary is None:
            continue
        print(
            f"{axis_name}: ref={axis_summary['reference_total_displacement_m']:.6f} "
            f"measured={axis_summary['measured_total_displacement_m']:.6f} "
            f"multiplier={axis_summary['suggested_scale_multiplier']}"
        )
        current_scale = current_scales[axis_name]
        multiplier = axis_summary["suggested_scale_multiplier"]
        if current_scale is not None and multiplier is not None:
            suggested_scale = current_scale * multiplier
            print(f"{axis_name}: current_scale={current_scale} suggested_scale={suggested_scale}")

    for axis_name in ("wx", "wy", "wz"):
        axis_summary = result["rotation_axis_summaries"].get(axis_name)
        if axis_summary is None:
            continue
        print(
            f"{axis_name}: ref={axis_summary['reference_total_displacement_rad']:.6f} "
            f"measured={axis_summary['measured_total_displacement_rad']:.6f} "
            f"multiplier={axis_summary['suggested_scale_multiplier']}"
        )
        current_scale = current_scales[axis_name]
        multiplier = axis_summary["suggested_scale_multiplier"]
        if current_scale is not None and multiplier is not None:
            suggested_scale = current_scale * multiplier
            print(f"{axis_name}: current_scale={current_scale} suggested_scale={suggested_scale}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"output={args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
