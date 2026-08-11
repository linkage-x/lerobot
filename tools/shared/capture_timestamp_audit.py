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

"""Timestamp-alignment measurement for any LeRobot v3 dataset carrying per-device capture times.

Rig-independent by construction: it knows about ``observation.device_capture_timestamp`` and the
nominal ``frame_index / fps`` grid, and nothing about which robot produced them. Anything that
*is* rig-specific -- which device the frame grid should be measured against, which sidecar file
holds raw capture metadata, how to phrase the verdict -- is a parameter, so a new rig adds a
caller rather than a copy of these statistics.

That matters because the same quantities are needed twice per rig, from two different sources: a
v3 dataset keeps one parquet file open until ``finalize()``, so a just-saved episode has to be
audited from the in-memory frame buffer, while the persisted report is built from the files
afterwards. Those two paths previously each carried their own implementation and drifted -- one
reported the p95 of ``|grid lag|`` and the other the p95 of the signed value, so a single episode
was described as both ``13.49`` and ``-3.75``. :func:`compute_frame_metrics` is the one
implementation both now share.

Deliberately depends on numpy and pyarrow only. These audits run from the data-collection gateway
and from bare-python shells where the ``lerobot`` package (and torch behind it) is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

DEFAULT_TOLERANCE_MS = 20.0
DEFAULT_GLOBAL_LAG_TOLERANCE_MS = 50.0


@dataclass(frozen=True)
class FrameMetrics:
    """Per-frame alignment quantities. All arrays are frame-indexed; bad frames hold NaN."""

    # Spread across devices within one frame: how simultaneous the modalities actually were.
    max_skew_s: np.ndarray
    # Offset of the frame from the dataset's nominal frame_index/fps grid.
    grid_lag_s: np.ndarray
    # Per-device offset from the grid, kept so a constant per-device bias can be characterised.
    device_lag_s: np.ndarray
    # Frames where every device reported a finite timestamp.
    finite_mask: np.ndarray


def resolve_grid_lag_reference_index(
    device_names: Sequence[str], prefixes: Sequence[str]
) -> int | None:
    """Index of the device the frame grid should be measured against, or None for the median.

    Grid lag asks whether frames landed where the dataset says they did, which is a question
    about the *control loop's* cadence. Measuring it against the median across devices answers
    that only while the devices sit close together: a modality with a real, constant offset drags
    the median with it and the number stops describing the loop. On the FR3 rig, camera columns
    honestly sit 25 ms ahead of the arm, which moved the median to -12 ms and reported 13.5 ms of
    grid lag for a loop whose cadence was exact to 0.03%.

    So a rig with a device that carries no pipeline delay of its own -- an arm, read for the frame
    rather than delivered to it -- should name it here. A rig without one (a handheld capture,
    where every device free-runs) has no better reference than the median, and passing no prefixes
    keeps exactly that.

    The reference need not be the loop tick exactly, and on the FR3 it is not: that column is the
    instant its 200 Hz state reader sampled, so it trails the tick by up to one poll period. That
    puts a few ms of quantization noise on this metric, which is the right trade -- the column has
    to mean *when the value was sampled* for every other number here to mean anything, and 5 ms of
    noise against a 50 ms budget is cheaper than a reference carrying a 25 ms pipeline offset.
    """
    for prefix in prefixes:
        for index, name in enumerate(device_names):
            if name.startswith(prefix):
                return index
    return None


def compute_frame_metrics(
    capture_timestamps: np.ndarray,
    frame_timestamps: np.ndarray,
    *,
    grid_lag_reference_index: int | None = None,
) -> FrameMetrics:
    """Measure skew and grid lag for every frame.

    ``grid_lag_reference_index`` selects the device the grid is measured against; ``None`` uses
    the median across devices. See :func:`resolve_grid_lag_reference_index` for why that choice
    is a rig property rather than a constant.
    """
    capture_timestamps = np.asarray(capture_timestamps, dtype=np.float64)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    if capture_timestamps.ndim != 2:
        raise ValueError(
            "observation.device_capture_timestamp must be a 2D array, "
            f"but got shape {capture_timestamps.shape}."
        )
    if capture_timestamps.shape[0] != frame_timestamps.shape[0]:
        raise ValueError(
            "capture_timestamps and frame_timestamps must have the same frame count "
            f"({capture_timestamps.shape[0]} vs {frame_timestamps.shape[0]})."
        )

    frames = int(capture_timestamps.shape[0])
    finite_mask = np.all(np.isfinite(capture_timestamps), axis=1)
    max_skew_s = np.full(frames, np.nan, dtype=np.float64)
    grid_lag_s = np.full(frames, np.nan, dtype=np.float64)
    device_lag_s = np.full(capture_timestamps.shape, np.nan, dtype=np.float64)

    if finite_mask.any():
        finite_capture = capture_timestamps[finite_mask]
        finite_frame_timestamps = frame_timestamps[finite_mask]
        max_skew_s[finite_mask] = np.max(finite_capture, axis=1) - np.min(finite_capture, axis=1)
        if grid_lag_reference_index is None:
            frame_centre = np.median(finite_capture, axis=1)
        else:
            frame_centre = finite_capture[:, grid_lag_reference_index]
        grid_lag_s[finite_mask] = frame_centre - finite_frame_timestamps
        device_lag_s[finite_mask] = finite_capture - finite_frame_timestamps[:, None]

    return FrameMetrics(
        max_skew_s=max_skew_s,
        grid_lag_s=grid_lag_s,
        device_lag_s=device_lag_s,
        finite_mask=finite_mask,
    )


def measured_frame_interval_s(capture_timestamps: np.ndarray) -> float:
    """Cadence the loop actually delivered: elapsed time over frame count.

    Not a median of per-frame gaps. Jitter is asymmetric -- a frame that lands late is followed
    by one that lands early -- so the median gap sits above the true average and condemns a
    cadence that was in fact held. Measured on the FR3 rig: median gap 35.4 ms against a 33.34 ms
    average, for an episode whose total duration was correct to 0.03% of nominal.
    """
    centres = np.median(np.asarray(capture_timestamps, dtype=np.float64), axis=1)
    centres = centres[np.isfinite(centres)]
    if centres.size < 2:
        return 0.0
    return float(centres[-1] - centres[0]) / (centres.size - 1)


def percentiles(values: np.ndarray) -> dict[str, float | None]:
    # Key order is part of the serialised report; keep it stable.
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


def first_bad_rows(
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


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def capture_timestamp_names(info: dict[str, Any]) -> list[str]:
    feature = info.get("features", {}).get("observation.device_capture_timestamp")
    if not isinstance(feature, dict):
        raise KeyError("Dataset is missing observation.device_capture_timestamp.")
    names = feature.get("names")
    if not isinstance(names, list):
        shape = feature.get("shape") or []
        width = int(shape[0]) if shape else 0
        return [f"device_{index}.capture_timestamp_s" for index in range(width)]
    return [str(name) for name in names]


def read_dataset_rows(dataset_root: Path) -> dict[str, np.ndarray]:
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


def build_report(
    *,
    dataset_root: Path,
    tolerance_ms: float,
    global_lag_tolerance_ms: float,
    grid_lag_reference_prefixes: Sequence[str] = (),
    raw_capture_metadata_filename: str | None = None,
    report_absolute_grid_lag: bool = False,
) -> dict[str, Any]:
    """Audit a finalized dataset on disk.

    Rig-specific inputs are the three keyword arguments: which device anchors the frame grid,
    which sidecar (if any) carries raw capture metadata, and whether to also summarise
    ``|grid lag|`` -- the signed distribution alone cannot answer "how far off was it", because
    a symmetric offset averages away.
    """
    info = load_dataset_info(dataset_root)
    raw_metadata = None
    if raw_capture_metadata_filename is not None:
        metadata_path = dataset_root / "meta" / raw_capture_metadata_filename
        if metadata_path.exists():
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_dataset_rows(dataset_root)
    names = capture_timestamp_names(info)

    capture_timestamps = rows["capture_timestamps"]
    if len(names) != capture_timestamps.shape[1]:
        raise ValueError(
            "capture timestamp feature name count does not match data width: "
            f"{len(names)} names for width {capture_timestamps.shape[1]}."
        )

    timestamps = rows["timestamp"]
    metrics = compute_frame_metrics(
        capture_timestamps,
        timestamps,
        grid_lag_reference_index=resolve_grid_lag_reference_index(
            names, grid_lag_reference_prefixes
        ),
    )

    skew_limit_s = tolerance_ms / 1000.0
    global_lag_limit_s = global_lag_tolerance_ms / 1000.0
    skew_bad_mask = metrics.max_skew_s > skew_limit_s
    global_lag_bad_mask = np.abs(metrics.grid_lag_s) > global_lag_limit_s
    nonfinite_mask = ~metrics.finite_mask

    per_device_lag = {
        name: percentiles(metrics.device_lag_s[:, index]) for index, name in enumerate(names)
    }

    def _episode_entry(episode: int, episode_mask: np.ndarray) -> dict[str, Any]:
        entry = {
            "episode": int(episode),
            "frames": int(np.sum(episode_mask)),
            "nonfinite_capture_timestamp_frames": int(np.sum(nonfinite_mask & episode_mask)),
            "skew_over_tolerance_frames": int(np.sum(skew_bad_mask & episode_mask)),
            "global_lag_over_tolerance_frames": int(np.sum(global_lag_bad_mask & episode_mask)),
            "max_skew_s": percentiles(metrics.max_skew_s[episode_mask]),
            "global_lag_s": percentiles(metrics.grid_lag_s[episode_mask]),
        }
        if report_absolute_grid_lag:
            entry["abs_global_lag_s"] = percentiles(np.abs(metrics.grid_lag_s[episode_mask]))
        return entry

    per_episode = [
        _episode_entry(episode, rows["episode_index"] == episode)
        for episode in sorted(set(rows["episode_index"].tolist()))
    ]

    bad_rows = first_bad_rows(
        episode_index=rows["episode_index"],
        frame_index=rows["frame_index"],
        values=metrics.max_skew_s,
        mask=skew_bad_mask,
        limit=skew_limit_s,
        kind="max_skew",
    )
    bad_rows.extend(
        first_bad_rows(
            episode_index=rows["episode_index"],
            frame_index=rows["frame_index"],
            values=np.abs(metrics.grid_lag_s),
            mask=global_lag_bad_mask,
            limit=global_lag_limit_s,
            kind="abs_global_lag",
        )
    )

    summary = {
        "nonfinite_capture_timestamp_frames": int(np.sum(nonfinite_mask)),
        "skew_over_tolerance_frames": int(np.sum(skew_bad_mask)),
        "global_lag_over_tolerance_frames": int(np.sum(global_lag_bad_mask)),
        "max_skew_s": percentiles(metrics.max_skew_s),
        "global_lag_s": percentiles(metrics.grid_lag_s),
    }
    if report_absolute_grid_lag:
        summary["abs_global_lag_s"] = percentiles(np.abs(metrics.grid_lag_s))

    return {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "fps": int(info.get("fps", 0)),
        "total_frames": int(len(timestamps)),
        "device_capture_timestamp_names": names,
        "raw_capture_metadata_present": raw_metadata is not None,
        "raw_capture_soft_sync_applied": None
        if raw_metadata is None
        else bool(raw_metadata.get("soft_sync_applied", False)),
        "limits": {
            "max_skew_s": skew_limit_s,
            "abs_global_lag_s": global_lag_limit_s,
        },
        "summary": summary,
        "per_device_lag_s": per_device_lag,
        "per_episode": per_episode,
        "first_bad_rows": bad_rows,
    }
