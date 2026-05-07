#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_TOLERANCE_MS = 20.0
DEFAULT_GLOBAL_LAG_TOLERANCE_MS = 50.0


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


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_raw_capture_metadata(dataset_root: Path) -> dict[str, Any] | None:
    metadata_path = dataset_root / "meta" / "handheld_raw_capture.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _capture_timestamp_names(info: dict[str, Any]) -> list[str]:
    feature = info.get("features", {}).get("observation.device_capture_timestamp")
    if not isinstance(feature, dict):
        raise KeyError("Dataset is missing observation.device_capture_timestamp.")
    names = feature.get("names")
    if not isinstance(names, list):
        shape = feature.get("shape") or []
        width = int(shape[0]) if shape else 0
        return [f"device_{index}.capture_timestamp_s" for index in range(width)]
    return [str(name) for name in names]


def _read_dataset_rows(dataset_root: Path) -> dict[str, np.ndarray]:
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}.")

    timestamps: list[float] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    capture_timestamps: list[list[float]] = []
    for parquet_file in parquet_files:
        table = pq.read_table(
            str(parquet_file),
            columns=[
                "timestamp",
                "episode_index",
                "frame_index",
                "observation.device_capture_timestamp",
            ],
        ).to_pydict()
        timestamps.extend(float(value) for value in table["timestamp"])
        episode_indices.extend(int(value) for value in table["episode_index"])
        frame_indices.extend(int(value) for value in table["frame_index"])
        capture_timestamps.extend(table["observation.device_capture_timestamp"])

    return {
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "episode_index": np.asarray(episode_indices, dtype=np.int64),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "capture_timestamps": np.asarray(capture_timestamps, dtype=np.float64),
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _first_bad_rows(
    *,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    limit: float,
    kind: str,
    max_rows: int = 50,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in np.where(mask)[0][:max_rows]:
        rows.append(
            {
                "kind": kind,
                "episode": int(episode_index[row_index]),
                "frame": int(frame_index[row_index]),
                "value_s": float(values[row_index]),
                "limit_s": float(limit),
            }
        )
    return rows


def build_report(
    *,
    dataset_root: Path,
    tolerance_ms: float,
    global_lag_tolerance_ms: float,
) -> dict[str, Any]:
    info = _load_info(dataset_root)
    raw_metadata = _load_raw_capture_metadata(dataset_root)
    rows = _read_dataset_rows(dataset_root)
    capture_names = _capture_timestamp_names(info)

    capture_timestamps = rows["capture_timestamps"]
    if capture_timestamps.ndim != 2:
        raise ValueError(
            "observation.device_capture_timestamp must be a 2D array, "
            f"but got shape {capture_timestamps.shape}."
        )
    if len(capture_names) != capture_timestamps.shape[1]:
        raise ValueError(
            "capture timestamp feature name count does not match data width: "
            f"{len(capture_names)} names for width {capture_timestamps.shape[1]}."
        )

    finite_mask = np.all(np.isfinite(capture_timestamps), axis=1)
    finite_capture_timestamps = capture_timestamps[finite_mask]
    timestamps = rows["timestamp"]
    finite_timestamps = timestamps[finite_mask]

    max_skew_s = np.full(len(timestamps), np.nan, dtype=np.float64)
    global_lag_s = np.full(len(timestamps), np.nan, dtype=np.float64)
    device_lag_s = np.full(capture_timestamps.shape, np.nan, dtype=np.float64)
    if finite_capture_timestamps.size:
        max_skew_s[finite_mask] = np.max(finite_capture_timestamps, axis=1) - np.min(
            finite_capture_timestamps,
            axis=1,
        )
        global_lag_s[finite_mask] = np.median(finite_capture_timestamps, axis=1) - finite_timestamps
        device_lag_s[finite_mask] = finite_capture_timestamps - finite_timestamps[:, None]

    skew_limit_s = tolerance_ms / 1000.0
    global_lag_limit_s = global_lag_tolerance_ms / 1000.0
    skew_bad_mask = max_skew_s > skew_limit_s
    global_lag_bad_mask = np.abs(global_lag_s) > global_lag_limit_s
    nonfinite_mask = ~finite_mask

    per_device_lag = {
        name: _percentiles(device_lag_s[:, index])
        for index, name in enumerate(capture_names)
    }
    per_episode: list[dict[str, Any]] = []
    for episode in sorted(set(rows["episode_index"].tolist())):
        episode_mask = rows["episode_index"] == episode
        per_episode.append(
            {
                "episode": int(episode),
                "frames": int(np.sum(episode_mask)),
                "nonfinite_capture_timestamp_frames": int(np.sum(nonfinite_mask & episode_mask)),
                "skew_over_tolerance_frames": int(np.sum(skew_bad_mask & episode_mask)),
                "global_lag_over_tolerance_frames": int(np.sum(global_lag_bad_mask & episode_mask)),
                "max_skew_s": _percentiles(max_skew_s[episode_mask]),
                "global_lag_s": _percentiles(global_lag_s[episode_mask]),
            }
        )

    first_bad_rows = []
    first_bad_rows.extend(
        _first_bad_rows(
            episode_index=rows["episode_index"],
            frame_index=rows["frame_index"],
            values=max_skew_s,
            mask=skew_bad_mask,
            limit=skew_limit_s,
            kind="max_skew",
        )
    )
    first_bad_rows.extend(
        _first_bad_rows(
            episode_index=rows["episode_index"],
            frame_index=rows["frame_index"],
            values=np.abs(global_lag_s),
            mask=global_lag_bad_mask,
            limit=global_lag_limit_s,
            kind="abs_global_lag",
        )
    )

    return {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "fps": int(info.get("fps", 0)),
        "total_frames": int(len(timestamps)),
        "device_capture_timestamp_names": capture_names,
        "raw_capture_metadata_present": raw_metadata is not None,
        "raw_capture_soft_sync_applied": None
        if raw_metadata is None
        else bool(raw_metadata.get("soft_sync_applied", False)),
        "limits": {
            "max_skew_s": skew_limit_s,
            "abs_global_lag_s": global_lag_limit_s,
        },
        "summary": {
            "nonfinite_capture_timestamp_frames": int(np.sum(nonfinite_mask)),
            "skew_over_tolerance_frames": int(np.sum(skew_bad_mask)),
            "global_lag_over_tolerance_frames": int(np.sum(global_lag_bad_mask)),
            "max_skew_s": _percentiles(max_skew_s),
            "global_lag_s": _percentiles(global_lag_s),
        },
        "per_device_lag_s": per_device_lag,
        "per_episode": per_episode,
        "first_bad_rows": first_bad_rows,
    }


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
