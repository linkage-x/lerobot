#!/usr/bin/env python3
"""Plot P0 replay XYZ and Corenetic gripper commands for every episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_9ch_v1_20260921_163918"
)
SIDECAR_RELATIVE_DIR = Path("derived/april_cube_tracking_in_robot_base")
DEFAULT_OUTPUT_RELATIVE_DIR = Path("derived/p0_replay_sync_diagnostics")
GRIPPER_FEATURE_NAME = "box_gripper.distance_m"


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_gripper_samples(dataset_root: Path) -> tuple[dict[tuple[int, int], dict[str, float]], str]:
    import pyarrow.parquet as pq

    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_key = None
    gripper_index = None
    for candidate in ("observation.state_raw", "observation.state"):
        names = (features.get(candidate) or {}).get("names")
        if isinstance(names, list) and GRIPPER_FEATURE_NAME in names:
            state_key = candidate
            gripper_index = names.index(GRIPPER_FEATURE_NAME)
            break
    if state_key is None or gripper_index is None:
        raise RuntimeError(f"dataset has no named {GRIPPER_FEATURE_NAME} feature")

    samples: dict[tuple[int, int], dict[str, float]] = {}
    parquet_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no Parquet files under {dataset_root / 'data'}")
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=[state_key, "timestamp", "episode_index", "frame_index"],
        )
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            state = row[state_key]
            width_m = float(state[gripper_index])
            timestamp_s = float(row["timestamp"])
            if not (math.isfinite(width_m) and math.isfinite(timestamp_s)):
                raise RuntimeError(f"non-finite gripper sample at episode={episode}, frame={frame}")
            key = (episode, frame)
            if key in samples:
                raise RuntimeError(f"duplicate gripper sample at episode={episode}, frame={frame}")
            samples[key] = {"timestamp_s": timestamp_s, "gripper_width_m": width_m}
    return samples, state_key


def _load_sidecar_rows(csv_path: Path) -> tuple[list[dict[str, float | int]], int]:
    rows: list[dict[str, float | int]] = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            values = {
                name: _finite_float(raw.get(name))
                for name in ("timestamp_s", "state_x_m", "state_y_m", "state_z_m")
            }
            if any(value is None for value in values.values()):
                skipped += 1
                continue
            rows.append(
                {
                    "episode_index": int(raw["episode_index"]),
                    "frame_index": int(raw["frame_index"]),
                    **{name: float(value) for name, value in values.items()},
                }
            )
    return rows, skipped


def _first_change_time(
    time_s: np.ndarray,
    width_m: np.ndarray,
    *,
    direction: str,
    threshold_m: float = 0.0005,
) -> float | None:
    delta = np.diff(width_m)
    indices = np.flatnonzero(delta < -threshold_m if direction == "close" else delta > threshold_m)
    return None if len(indices) == 0 else float(time_s[int(indices[0]) + 1])


def _write_episode_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "episode_index",
        "frame_index",
        "replay_time_s",
        "x_m",
        "y_m",
        "z_m",
        "gripper_command_m",
        "gripper_command_mm",
        "gripper_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_episode(
    path: Path,
    *,
    side: str,
    episode: int,
    time_s: np.ndarray,
    xyz_m: np.ndarray,
    gripper_width_m: np.ndarray,
    first_close_s: float | None,
    first_open_s: float | None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (axis_xyz, axis_gripper) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    for column, label, color in zip(range(3), ("x", "y", "z"), ("tab:red", "tab:green", "tab:blue")):
        axis_xyz.plot(time_s, xyz_m[:, column], label=label, color=color, linewidth=1.5)
    axis_xyz.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="shared replay t=0")
    axis_xyz.set_ylabel("Position in robot base [m]")
    axis_xyz.grid(True, alpha=0.25)
    axis_xyz.legend(loc="best", ncols=4)
    axis_xyz.set_title(f"P0 replay sync diagnostic — side={side}, episode={episode}")

    axis_gripper.plot(
        time_s,
        gripper_width_m * 1000.0,
        color="tab:purple",
        linewidth=1.8,
        label="replay command from measured box_gripper.distance_m",
    )
    axis_gripper.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    if first_close_s is not None:
        axis_gripper.axvline(
            first_close_s,
            color="tab:red",
            linestyle=":",
            linewidth=1.2,
            label=f"first >0.5 mm close @ {first_close_s:.3f}s",
        )
    if first_open_s is not None:
        axis_gripper.axvline(
            first_open_s,
            color="tab:green",
            linestyle=":",
            linewidth=1.2,
            label=f"first >0.5 mm open @ {first_open_s:.3f}s",
        )
    axis_gripper.set_xlabel("Original episode time / replay source time [s]")
    axis_gripper.set_ylabel("Gripper command [mm]")
    axis_gripper.grid(True, alpha=0.25)
    axis_gripper.legend(loc="best")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def generate_diagnostics(dataset_root: Path, output_dir: Path, sides: tuple[str, ...]) -> dict[str, object]:
    gripper_samples, gripper_state_key = _load_gripper_samples(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "p0_replay_sync_diagnostics/v1",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "gripper_source": gripper_state_key,
        "gripper_feature": GRIPPER_FEATURE_NAME,
        "important": (
            "The dataset contains measured gripper width, not a gripper action. "
            "Replay uses this measured width as the requested command."
        ),
        "sides": {},
    }

    for side in sides:
        csv_path = dataset_root / SIDECAR_RELATIVE_DIR / f"state_action.{side}.csv"
        side_rows, skipped = _load_sidecar_rows(csv_path)
        episode_ids = sorted({int(row["episode_index"]) for row in side_rows})
        side_summary: dict[str, object] = {
            "sidecar": str(csv_path),
            "skipped_nonfinite_pose_rows": skipped,
            "episodes": {},
        }
        for episode in episode_ids:
            pose_rows = sorted(
                (row for row in side_rows if int(row["episode_index"]) == episode),
                key=lambda row: int(row["frame_index"]),
            )
            aligned: list[dict[str, float | int]] = []
            for pose in pose_rows:
                key = (episode, int(pose["frame_index"]))
                gripper = gripper_samples.get(key)
                if gripper is None:
                    raise RuntimeError(f"missing gripper sample at episode={episode}, frame={key[1]}")
                aligned.append({**pose, **gripper})
            if not aligned:
                continue

            source_time = np.asarray([float(row["timestamp_s"]) for row in aligned])
            time_s = source_time - source_time[0]
            xyz_m = np.asarray(
                [[float(row[f"state_{axis}_m"]) for axis in "xyz"] for row in aligned],
                dtype=np.float64,
            )
            width_m = np.asarray([float(row["gripper_width_m"]) for row in aligned])
            first_close_s = _first_change_time(time_s, width_m, direction="close")
            first_open_s = _first_change_time(time_s, width_m, direction="open")
            stem = f"episode_{episode:03d}_{side}_time_xyz_gripper"
            png_path = output_dir / f"{stem}.png"
            csv_output_path = output_dir / f"{stem}.csv"
            _plot_episode(
                png_path,
                side=side,
                episode=episode,
                time_s=time_s,
                xyz_m=xyz_m,
                gripper_width_m=width_m,
                first_close_s=first_close_s,
                first_open_s=first_open_s,
            )
            _write_episode_csv(
                csv_output_path,
                [
                    {
                        "episode_index": episode,
                        "frame_index": int(row["frame_index"]),
                        "replay_time_s": float(t),
                        "x_m": float(xyz[0]),
                        "y_m": float(xyz[1]),
                        "z_m": float(xyz[2]),
                        "gripper_command_m": float(width),
                        "gripper_command_mm": float(width * 1000.0),
                        "gripper_source": f"{gripper_state_key}.{GRIPPER_FEATURE_NAME}",
                    }
                    for row, t, xyz, width in zip(aligned, time_s, xyz_m, width_m, strict=True)
                ],
            )
            side_summary["episodes"][str(episode)] = {
                "frames": len(aligned),
                "duration_s": float(time_s[-1]),
                "start_xyz_m": xyz_m[0].tolist(),
                "end_xyz_m": xyz_m[-1].tolist(),
                "gripper_start_mm": float(width_m[0] * 1000.0),
                "gripper_min_mm": float(width_m.min() * 1000.0),
                "gripper_max_mm": float(width_m.max() * 1000.0),
                "gripper_end_mm": float(width_m[-1] * 1000.0),
                "first_close_over_0p5mm_s": first_close_s,
                "first_open_over_0p5mm_s": first_open_s,
                "plot": str(png_path),
                "csv": str(csv_output_path),
            }
        manifest["sides"][side] = side_summary

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--side", choices=("all", "left", "right"), default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / DEFAULT_OUTPUT_RELATIVE_DIR
    )
    sides = ("left", "right") if args.side == "all" else (args.side,)
    manifest = generate_diagnostics(dataset_root, output_dir, sides)
    episode_count = sum(
        len(side_summary["episodes"])
        for side_summary in manifest["sides"].values()
    )
    print(f"Wrote {episode_count} episode plots and CSVs to {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
