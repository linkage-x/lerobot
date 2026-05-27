"""Minimal LeRobot v3 tabular writer for Thor BOX collection samples.

Thor records camera streams as hardware-encoded MKV files. Those files are
kept as-is, but the GUI replay path expects LeRobot v3 tabular data under
``meta/info.json`` and ``data/chunk-*/*.parquet`` when it displays
``observation.state`` / ``action``. This module writes that lightweight v3
side of the dataset from BOX snapshots without changing the camera pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_TOUCH_SUMMARY_NAMES = (
    "mean_fx_0p1N",
    "mean_fy_0p1N",
    "mean_fz_0p1N",
    "max_abs_fz_0p1N",
    "active_points",
)


BOX_STATE_NAMES: tuple[str, ...] = (
    "gripper.pos",
    "box_gripper.distance_m",
    "box_gripper.timestamp",
    "box_trigger.travel_pct",
    "box_trigger.timestamp",
    "box_imu.acc_x_g",
    "box_imu.acc_y_g",
    "box_imu.acc_z_g",
    "box_imu.gyr_x_deg_s",
    "box_imu.gyr_y_deg_s",
    "box_imu.gyr_z_deg_s",
    "box_imu.roll_deg",
    "box_imu.pitch_deg",
    "box_imu.yaw_deg",
    "box_imu.quat_w",
    "box_imu.quat_x",
    "box_imu.quat_y",
    "box_imu.quat_z",
    "box_imu.timestamp",
    "box_six_d_force.fx",
    "box_six_d_force.fy",
    "box_six_d_force.fz",
    "box_six_d_force.mx",
    "box_six_d_force.my",
    "box_six_d_force.mz",
    "box_six_d_force.timestamp",
    *tuple(f"box_touch_left.{name}" for name in _TOUCH_SUMMARY_NAMES),
    "box_touch_left.timestamp",
    *tuple(f"box_touch_right.{name}" for name in _TOUCH_SUMMARY_NAMES),
    "box_touch_right.timestamp",
    "box_status.valid",
    "box_status.liwp_index",
    "box_status.liwp_timestamp",
    "box_status.received_wall_time_s",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sensor(snapshot: dict[str, Any], sensor_id: str) -> dict[str, Any]:
    sensors = snapshot.get("sensors")
    if not isinstance(sensors, dict):
        return {}
    sensor = sensors.get(sensor_id)
    return sensor if isinstance(sensor, dict) else {}


def _timestamp(sensor: dict[str, Any]) -> float:
    return _finite_float(sensor.get("timestamp"))


def _list_values(sensor: dict[str, Any], key: str, count: int) -> list[float]:
    raw = sensor.get(key)
    if not isinstance(raw, (list, tuple)):
        return [0.0] * count
    values = [_finite_float(v) for v in raw[:count]]
    return values + [0.0] * max(0, count - len(values))


def _touch_summary(sensor: dict[str, Any]) -> list[float]:
    fx = _list_values(sensor, "fx_0p1N", 239)
    fy = _list_values(sensor, "fy_0p1N", 239)
    fz = _list_values(sensor, "fz_0p1N", 239)
    if not fx and not fy and not fz:
        return [0.0] * len(_TOUCH_SUMMARY_NAMES)
    count = max(len(fx), len(fy), len(fz), 1)
    active = sum(1 for x, y, z in zip(fx, fy, fz, strict=False) if x != 0.0 or y != 0.0 or z != 0.0)
    return [
        sum(fx) / count,
        sum(fy) / count,
        sum(fz) / count,
        max((abs(v) for v in fz), default=0.0),
        float(active),
    ]


def box_snapshot_to_state(snapshot: dict[str, Any]) -> list[float]:
    """Flatten one BOX snapshot into the named LeRobot state vector."""

    gripper = _sensor(snapshot, "box_gripper")
    trigger = _sensor(snapshot, "box_trigger")
    imu = _sensor(snapshot, "box_imu")
    six_d = _sensor(snapshot, "box_six_d_force")
    touch_left = _sensor(snapshot, "box_touch_left")
    touch_right = _sensor(snapshot, "box_touch_right")

    distance_m = _finite_float(gripper.get("distance_m"))
    state = [
        distance_m,
        distance_m,
        _timestamp(gripper),
        _finite_float(trigger.get("travel_pct")),
        _timestamp(trigger),
        *_list_values(imu, "acc_g", 3),
        *_list_values(imu, "gyr_deg_s", 3),
        _finite_float(imu.get("roll_deg")),
        _finite_float(imu.get("pitch_deg")),
        _finite_float(imu.get("yaw_deg")),
        *_list_values(imu, "quat_wxyz", 4),
        _timestamp(imu),
        *_list_values(six_d, "fxyz_mxyz", 6),
        _timestamp(six_d),
        *_touch_summary(touch_left),
        _timestamp(touch_left),
        *_touch_summary(touch_right),
        _timestamp(touch_right),
        1.0 if bool(snapshot.get("valid")) else 0.0,
        _finite_float(snapshot.get("liwp_index")),
        _finite_float(snapshot.get("liwp_timestamp")),
        _finite_float(snapshot.get("received_wall_time_s")),
    ]
    if len(state) != len(BOX_STATE_NAMES):
        raise RuntimeError(f"BOX state length mismatch: {len(state)} != {len(BOX_STATE_NAMES)}")
    return [float(v) for v in state]


def _stats(values: list[list[float]]) -> dict[str, list[float] | list[int]]:
    if not values:
        return {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "count": [0],
            "q01": [],
            "q10": [],
            "q50": [],
            "q90": [],
            "q99": [],
        }
    width = len(values[0])
    columns = [[row[i] for row in values] for i in range(width)]
    means = [sum(col) / len(col) for col in columns]
    stds = [
        (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5
        for col, mean in zip(columns, means, strict=True)
    ]

    def quantile(col: list[float], q: float) -> float:
        ordered = sorted(col)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    return {
        "min": [min(col) for col in columns],
        "max": [max(col) for col in columns],
        "mean": means,
        "std": stds,
        "count": [len(values)],
        "q01": [quantile(col, 0.01) for col in columns],
        "q10": [quantile(col, 0.10) for col in columns],
        "q50": [quantile(col, 0.50) for col in columns],
        "q90": [quantile(col, 0.90) for col in columns],
        "q99": [quantile(col, 0.99) for col in columns],
    }


def _feature(dtype: str, shape: list[int], names: list[str] | None = None) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names}


def _rows_to_table(pa, rows: list[dict[str, Any]]):
    state_width = len(BOX_STATE_NAMES)

    def vector_column(key: str):
        flat: list[float] = []
        for row in rows:
            values = list(row[key])
            if len(values) != state_width:
                raise ValueError(f"{key} width mismatch: {len(values)} != {state_width}")
            flat.extend(values)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), state_width)

    return pa.table(
        {
            "observation.state": vector_column("observation.state"),
            "action": vector_column("action"),
            "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float32()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "index": pa.array([row["index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        }
    )


def write_box_lerobot_v3_episode(
    dataset_root: Path,
    *,
    repo_id: str,
    task: str,
    fps: int,
    episode_index: int,
    snapshots: list[dict[str, Any]],
    duration_s: float | None = None,
) -> Path | None:
    """Append BOX snapshots for one episode to a minimal LeRobot v3 dataset.

    Returns the parquet path when rows were written. If ``pyarrow`` is missing
    or no snapshots were captured, returns ``None`` so the recorder can keep
    the hardware capture path alive.
    """

    if not snapshots:
        return None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return None

    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "file-000.parquet"
    existing_rows: list[dict[str, Any]] = []
    if data_path.exists():
        existing_rows = pq.read_table(data_path).to_pylist()

    frame_count = len(snapshots)
    if duration_s is not None and duration_s > 0:
        frame_count = max(frame_count, int(round(float(duration_s) * max(int(fps), 1))))

    ordered_snapshots = sorted(
        snapshots,
        key=lambda snap: _finite_float(snap.get("t_relative_s")),
    )

    def snapshot_for_timestamp(timestamp_s: float) -> dict[str, Any]:
        selected = ordered_snapshots[0]
        for candidate in ordered_snapshots:
            if _finite_float(candidate.get("t_relative_s")) <= timestamp_s + 1e-9:
                selected = candidate
            else:
                break
        return selected

    rows: list[dict[str, Any]] = []
    for local_frame in range(frame_count):
        timestamp_s = local_frame / max(int(fps), 1)
        snapshot = snapshot_for_timestamp(timestamp_s)
        state = box_snapshot_to_state(snapshot)
        rows.append(
            {
                "observation.state": state,
                # BOX collection has no separate command stream. Mirror the
                # latest measured state so replay exposes the same named values
                # through its action panel instead of leaving action blank.
                "action": list(state),
                "timestamp": timestamp_s,
                "frame_index": local_frame,
                "episode_index": episode_index,
                "index": 0,
                "task_index": 0,
            }
        )

    existing_rows = [
        row for row in existing_rows
        if int(row.get("episode_index") if row.get("episode_index") is not None else -1) != episode_index
    ]
    all_rows = sorted(
        [*existing_rows, *rows],
        key=lambda row: (int(row.get("episode_index") or 0), int(row.get("frame_index") or 0)),
    )
    for global_index, row in enumerate(all_rows):
        row["index"] = global_index
    pq.write_table(_rows_to_table(pa, all_rows), data_path)

    total_episodes = len({int(row["episode_index"]) for row in all_rows})
    total_frames = len(all_rows)
    features = {
        "observation.state": _feature("float32", [len(BOX_STATE_NAMES)], list(BOX_STATE_NAMES)),
        "action": _feature("float32", [len(BOX_STATE_NAMES)], list(BOX_STATE_NAMES)),
        "timestamp": _feature("float32", [1]),
        "frame_index": _feature("int64", [1]),
        "episode_index": _feature("int64", [1]),
        "index": _feature("int64", [1]),
        "task_index": _feature("int64", [1]),
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "thor_gmsl2_box",
        "repo_id": repo_id,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": task}]), meta_dir / "tasks.parquet")

    state_values = [list(row["observation.state"]) for row in all_rows]
    action_values = [list(row["action"]) for row in all_rows]
    stats = {
        "observation.state": _stats(state_values),
        "action": _stats(action_values),
        "timestamp": _stats([[float(row["timestamp"])] for row in all_rows]),
        "frame_index": _stats([[float(row["frame_index"])] for row in all_rows]),
        "episode_index": _stats([[float(row["episode_index"])] for row in all_rows]),
        "index": _stats([[float(row["index"])] for row in all_rows]),
        "task_index": _stats([[float(row["task_index"])] for row in all_rows]),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    episode_rows = []
    for ep in sorted({int(row["episode_index"]) for row in all_rows}):
        ep_rows = [row for row in all_rows if int(row["episode_index"]) == ep]
        indices = [int(row["index"]) for row in ep_rows]
        episode_rows.append(
            {
                "episode_index": ep,
                "tasks": [task],
                "length": len(ep_rows),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": min(indices),
                "dataset_to_index": max(indices) + 1,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_dir / "file-000.parquet")
    return data_path
