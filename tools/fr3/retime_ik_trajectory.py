#!/usr/bin/env python3
"""Uniformly slow solved FR3 joint trajectories without dropping source waypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
MAX_VELOCITY_RAD_S = np.asarray([2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208])
MAX_ACCELERATION_RAD_S2 = np.full(7, 8.0)
MAX_JERK_RAD_S3 = np.full(7, 4000.0)
MIN_POSITION_RAD = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
MAX_POSITION_RAD = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def _finite_float(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in {None, ""} else float("nan")
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _load_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    episodes: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode_index", *[f"{name}_rad" for name in JOINT_NAMES]}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            if str(row.get("reachable", "true")).strip().lower() not in {"true", "1", "yes", "y"}:
                raise ValueError("Input contains unreachable rows; retiming requires a complete reachable trajectory.")
            episode = int(row["episode_index"])
            joints = [_finite_float(row.get(f"{name}_rad")) for name in JOINT_NAMES]
            if any(value is None for value in joints):
                raise ValueError(f"Episode {episode} contains a non-finite joint value.")
            episodes[episode].append(row)
    for rows in episodes.values():
        rows.sort(key=lambda row: (int(row.get("frame_order") or 0), int(row.get("frame_index") or 0)))
    if not episodes:
        raise ValueError(f"No trajectory rows found in {path}")
    return dict(sorted(episodes.items()))


def _joint_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{name}_rad"]) for name in JOINT_NAMES] for row in rows],
        dtype=np.float64,
    )


def _dense_derivative_maxima(spline: CubicSpline, duration_s: float, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_time = np.linspace(0.0, duration_s, max(2, samples), dtype=np.float64)
    return tuple(np.max(np.abs(spline(dense_time, order)), axis=0) for order in (1, 2, 3))  # type: ignore[return-value]


def _required_scale(
    maxima: tuple[np.ndarray, np.ndarray, np.ndarray],
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    jerk_limit: np.ndarray,
) -> float:
    velocity, acceleration, jerk = maxima
    return float(
        max(
            1.0,
            np.max(velocity / velocity_limit),
            math.sqrt(float(np.max(acceleration / acceleration_limit))),
            math.cbrt(float(np.max(jerk / jerk_limit))),
        )
    )


def _jsonable(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    episodes = _load_rows(input_path)

    source_fps = float(args.source_fps)
    output_fps = float(args.output_fps)
    if abs(source_fps - output_fps) > 1e-9:
        raise ValueError("This waypoint-preserving retimer currently requires source_fps == output_fps.")
    safety_fraction = float(args.limit_fraction)
    velocity_limit = MAX_VELOCITY_RAD_S * safety_fraction
    acceleration_limit = MAX_ACCELERATION_RAD_S2 * safety_fraction
    jerk_limit = MAX_JERK_RAD_S3 * safety_fraction

    episode_models: dict[int, tuple[list[dict[str, str]], np.ndarray, np.ndarray, CubicSpline, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    raw_required_scale = 1.0
    for episode, rows in episodes.items():
        joints = _joint_matrix(rows)
        if len(joints) < 2:
            raise ValueError(f"Episode {episode} needs at least two frames.")
        source_time = np.arange(len(joints), dtype=np.float64) / source_fps
        # Zero endpoint velocity prevents a sudden start or stop on hardware.
        spline = CubicSpline(source_time, joints, axis=0, bc_type="clamped")
        maxima = _dense_derivative_maxima(spline, float(source_time[-1]), (len(joints) - 1) * 20 + 1)
        raw_required_scale = max(
            raw_required_scale,
            _required_scale(maxima, velocity_limit, acceleration_limit, jerk_limit),
        )
        episode_models[episode] = (rows, joints, source_time, spline, maxima)

    if args.slowdown is None:
        slowdown = int(math.ceil(raw_required_scale * float(args.scale_margin)))
    else:
        slowdown = int(args.slowdown)
    if slowdown < 1:
        raise ValueError("slowdown must be a positive integer.")
    if slowdown + 1e-12 < raw_required_scale and not args.allow_limit_violation:
        raise ValueError(
            f"slowdown={slowdown} is below the required {raw_required_scale:.4f}; "
            "choose a larger value or pass --allow-limit-violation."
        )

    output_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    global_index = 0
    for episode, (source_rows, source_joints, source_time, spline, original_maxima) in episode_models.items():
        output_count = (len(source_rows) - 1) * slowdown + 1
        output_time = np.arange(output_count, dtype=np.float64) / output_fps
        query_source_time = output_time / float(slowdown)
        retimed_joints = np.asarray(spline(query_source_time), dtype=np.float64)

        source_gripper = np.asarray(
            [float(value) if (value := _finite_float(row.get("gripper_pos"))) is not None else np.nan for row in source_rows],
            dtype=np.float64,
        )
        retimed_gripper = (
            np.interp(query_source_time, source_time, source_gripper)
            if np.all(np.isfinite(source_gripper))
            else np.full(output_count, np.nan)
        )

        waypoint_error = float(np.max(np.abs(retimed_joints[::slowdown] - source_joints)))
        position_ok = bool(
            np.all(retimed_joints >= MIN_POSITION_RAD.reshape(1, 7))
            and np.all(retimed_joints <= MAX_POSITION_RAD.reshape(1, 7))
        )
        retimed_maxima = tuple(values / (float(slowdown) ** order) for order, values in enumerate(original_maxima, start=1))
        velocity_ok = bool(np.all(retimed_maxima[0] <= velocity_limit + 1e-12))
        acceleration_ok = bool(np.all(retimed_maxima[1] <= acceleration_limit + 1e-12))
        jerk_ok = bool(np.all(retimed_maxima[2] <= jerk_limit + 1e-12))

        for output_order in range(output_count):
            source_position = output_order / float(slowdown)
            source_index = min(int(math.floor(source_position + 1e-12)), len(source_rows) - 1)
            source_row = source_rows[source_index]
            row: dict[str, Any] = {
                "cube": source_row.get("cube", "left"),
                "episode_index": episode,
                "frame_order": output_order,
                "frame_index": output_order,
                "global_index": global_index,
                "reachable": True,
                "trajectory_time_s": float(output_time[output_order]),
                "source_time_s": float(query_source_time[output_order]),
                "source_frame_position": source_position,
                "source_frame_index": int(source_row.get("frame_index") or source_index),
                "is_source_waypoint": output_order % slowdown == 0,
                "gripper_pos": "" if not np.isfinite(retimed_gripper[output_order]) else float(retimed_gripper[output_order]),
            }
            for joint_index, name in enumerate(JOINT_NAMES):
                value = float(retimed_joints[output_order, joint_index])
                row[f"{name}_rad"] = value
                row[f"{name}_deg"] = float(np.rad2deg(value))
            output_rows.append(row)
            global_index += 1

        episode_summaries.append(
            {
                "episode_index": episode,
                "source_frames": len(source_rows),
                "output_frames": output_count,
                "source_duration_s": float(source_time[-1]),
                "retimed_duration_s": float(output_time[-1]),
                "source_waypoint_max_error_rad": waypoint_error,
                "position_limits_ok": position_ok,
                "velocity_limits_ok": velocity_ok,
                "acceleration_limits_ok": acceleration_ok,
                "jerk_limits_ok": jerk_ok,
                "original_max_velocity_rad_s": _jsonable(original_maxima[0]),
                "original_max_acceleration_rad_s2": _jsonable(original_maxima[1]),
                "original_max_jerk_rad_s3": _jsonable(original_maxima[2]),
                "retimed_max_velocity_rad_s": _jsonable(retimed_maxima[0]),
                "retimed_max_acceleration_rad_s2": _jsonable(retimed_maxima[1]),
                "retimed_max_jerk_rad_s3": _jsonable(retimed_maxima[2]),
                "gripper_samples": int(np.count_nonzero(np.isfinite(retimed_gripper))),
                "gripper_min": float(np.nanmin(retimed_gripper)) if np.any(np.isfinite(retimed_gripper)) else None,
                "gripper_max": float(np.nanmax(retimed_gripper)) if np.any(np.isfinite(retimed_gripper)) else None,
            }
        )

    if not all(
        ep[key]
        for ep in episode_summaries
        for key in ("position_limits_ok", "velocity_limits_ok", "acceleration_limits_ok", "jerk_limits_ok")
    ):
        raise RuntimeError("Retimed trajectory failed one or more safety-limit checks; output was not written.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "slowdown": slowdown,
        "raw_required_scale": raw_required_scale,
        "scale_margin": float(args.scale_margin),
        "limit_fraction": safety_fraction,
        "effective_limits": {
            "velocity_rad_s": _jsonable(velocity_limit),
            "acceleration_rad_s2": _jsonable(acceleration_limit),
            "jerk_rad_s3": _jsonable(jerk_limit),
            "position_min_rad": _jsonable(MIN_POSITION_RAD),
            "position_max_rad": _jsonable(MAX_POSITION_RAD),
        },
        "source_frames": int(sum(len(rows) for rows in episodes.values())),
        "output_frames": len(output_rows),
        "episodes": episode_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--output-fps", type=float, default=60.0)
    parser.add_argument("--slowdown", type=int)
    parser.add_argument("--limit-fraction", type=float, default=0.8)
    parser.add_argument("--scale-margin", type=float, default=1.05)
    parser.add_argument("--allow-limit-violation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
