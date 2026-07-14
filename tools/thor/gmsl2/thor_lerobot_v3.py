"""Minimal LeRobot v3 tabular writer for Thor BOX collection samples.

Thor records camera streams as hardware-encoded MKV files. Those files are
kept as-is, but the GUI replay path expects LeRobot v3 tabular data under
``meta/info.json`` and ``data/chunk-*/*.parquet`` when it displays
``observation.state``. This module writes that lightweight v3 side of the
dataset from BOX snapshots without changing the camera pipeline. ``action`` is
left for the export step, where it can be derived from the next aligned state.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("thor_lerobot_v3")


_TOUCH_SUMMARY_NAMES = (
    "mean_fx_0p1N",
    "mean_fy_0p1N",
    "mean_fz_0p1N",
    "max_abs_fz_0p1N",
    "active_points",
)


BOX_STATE_NAMES: tuple[str, ...] = (
    "box_gripper.distance_m",
    "box_trigger.travel_pct",
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
    "box_six_d_force.fx",
    "box_six_d_force.fy",
    "box_six_d_force.fz",
    "box_six_d_force.mx",
    "box_six_d_force.my",
    "box_six_d_force.mz",
    *tuple(f"box_touch_left.{name}" for name in _TOUCH_SUMMARY_NAMES),
    *tuple(f"box_touch_right.{name}" for name in _TOUCH_SUMMARY_NAMES),
)

# Per-frame timestamp metadata, emitted as a SEPARATE non-observation parquet
# column ``box.timestamps`` (float64 to preserve the µs-resolution MCU counters
# that overflow float32's 2**24 exact-integer range). These are diagnostic
# alignment values, NOT trainable observations, so they are deliberately kept
# out of ``observation.state``.
BOX_TIMESTAMP_NAMES: tuple[str, ...] = (
    "box_gripper.timestamp",
    "box_trigger.timestamp",
    "box_imu.timestamp",
    "box_six_d_force.timestamp",
    "box_touch_left.timestamp",
    "box_touch_right.timestamp",
)

BOX_SENSOR_IDS: tuple[str, ...] = (
    "box_gripper",
    "box_trigger",
    "box_imu",
    "box_six_d_force",
    "box_touch_left",
    "box_touch_right",
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


def _split_box_sensor_id(sensor_id: str) -> tuple[str, str] | None:
    if sensor_id in BOX_SENSOR_IDS:
        return ("", sensor_id)
    if "/" not in sensor_id:
        return None
    box_id, bare = sensor_id.split("/", 1)
    if box_id and bare in BOX_SENSOR_IDS:
        return (box_id, bare)
    return None


def _box_ids_from_sensor_ids(sensor_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    box_ids = {
        split[0]
        for sid in sensor_ids
        if (split := _split_box_sensor_id(str(sid))) is not None
    }
    if not box_ids:
        return ("",)
    return tuple(sorted(box_ids, key=lambda item: (item != "", item)))


def box_ids_from_snapshot(snapshot: dict[str, Any]) -> tuple[str, ...]:
    sensors = snapshot.get("sensors")
    if not isinstance(sensors, dict):
        return ("",)
    return _box_ids_from_sensor_ids(tuple(str(sid) for sid in sensors))


def box_ids_from_snapshots(snapshots: list[dict[str, Any]]) -> tuple[str, ...]:
    sensor_ids: list[str] = []
    for snapshot in snapshots:
        sensors = snapshot.get("sensors")
        if isinstance(sensors, dict):
            sensor_ids.extend(str(sid) for sid in sensors)
    return _box_ids_from_sensor_ids(sensor_ids)


def box_ids_from_sensor_samples(sensor_samples: dict[str, list[dict[str, Any]]] | None) -> tuple[str, ...]:
    if not sensor_samples:
        return ("",)
    return _box_ids_from_sensor_ids(tuple(str(sid) for sid in sensor_samples))


def box_ids_from_inputs(
    snapshots: list[dict[str, Any]],
    sensor_samples: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, ...]:
    ids = set(box_ids_from_snapshots(snapshots))
    if sensor_samples:
        ids.update(box_ids_from_sensor_samples(sensor_samples))
    ids.discard("")
    if ids:
        return tuple(sorted(ids))
    return ("",)


def _prefix_box_name(box_id: str, name: str) -> str:
    return f"{box_id}.{name}" if box_id else name


def box_state_names(box_ids: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    ids = tuple(box_ids or ("",))
    if ids == ("",):
        return BOX_STATE_NAMES
    return tuple(_prefix_box_name(box_id, name) for box_id in ids for name in BOX_STATE_NAMES)


def box_timestamp_names(box_ids: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    ids = tuple(box_ids or ("",))
    if ids == ("",):
        return BOX_TIMESTAMP_NAMES
    return tuple(_prefix_box_name(box_id, name) for box_id in ids for name in BOX_TIMESTAMP_NAMES)


def _snapshot_for_box(snapshot: dict[str, Any], box_id: str) -> dict[str, Any]:
    sensors = snapshot.get("sensors")
    if not isinstance(sensors, dict):
        return {"valid": False, "sensors": {}}
    if not box_id:
        return snapshot
    grouped: dict[str, Any] = {}
    prefix = f"{box_id}/"
    for sid, payload in sensors.items():
        sid_str = str(sid)
        if not sid_str.startswith(prefix):
            continue
        bare = sid_str[len(prefix):]
        if bare in BOX_SENSOR_IDS and isinstance(payload, dict):
            grouped[bare] = payload
    return {"valid": bool(grouped), "sensors": grouped}


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
    """Flatten one BOX snapshot into the named LeRobot state vector.

    Per-sensor MCU timestamps are intentionally excluded -- they are diagnostic
    alignment metadata, not trainable observations, and are emitted separately
    via :func:`box_snapshot_to_timestamps` into the ``box.timestamps`` column.
    See :data:`BOX_STATE_NAMES` / :data:`BOX_TIMESTAMP_NAMES`.
    """

    gripper = _sensor(snapshot, "box_gripper")
    trigger = _sensor(snapshot, "box_trigger")
    imu = _sensor(snapshot, "box_imu")
    six_d = _sensor(snapshot, "box_six_d_force")
    touch_left = _sensor(snapshot, "box_touch_left")
    touch_right = _sensor(snapshot, "box_touch_right")

    state = [
        _finite_float(gripper.get("distance_m")),
        _finite_float(trigger.get("travel_pct")),
        *_list_values(imu, "acc_g", 3),
        *_list_values(imu, "gyr_deg_s", 3),
        _finite_float(imu.get("roll_deg")),
        _finite_float(imu.get("pitch_deg")),
        _finite_float(imu.get("yaw_deg")),
        *_list_values(imu, "quat_wxyz", 4),
        *_list_values(six_d, "fxyz_mxyz", 6),
        *_touch_summary(touch_left),
        *_touch_summary(touch_right),
    ]
    if len(state) != len(BOX_STATE_NAMES):
        raise RuntimeError(f"BOX state length mismatch: {len(state)} != {len(BOX_STATE_NAMES)}")
    return [float(v) for v in state]


def box_snapshot_to_timestamps(snapshot: dict[str, Any]) -> list[float]:
    """Per-frame timestamp metadata for the ``box.timestamps`` column.

    Returned as floats (stored as float64) so the µs-resolution MCU counters
    keep full integer precision -- which they would lose inside the float32
    ``observation.state`` vector. Order follows :data:`BOX_TIMESTAMP_NAMES`.
    """

    gripper = _sensor(snapshot, "box_gripper")
    trigger = _sensor(snapshot, "box_trigger")
    imu = _sensor(snapshot, "box_imu")
    six_d = _sensor(snapshot, "box_six_d_force")
    touch_left = _sensor(snapshot, "box_touch_left")
    touch_right = _sensor(snapshot, "box_touch_right")

    timestamps = [
        _timestamp(gripper),
        _timestamp(trigger),
        _timestamp(imu),
        _timestamp(six_d),
        _timestamp(touch_left),
        _timestamp(touch_right),
    ]
    if len(timestamps) != len(BOX_TIMESTAMP_NAMES):
        raise RuntimeError(
            f"BOX timestamp length mismatch: {len(timestamps)} != {len(BOX_TIMESTAMP_NAMES)}"
        )
    return [float(v) for v in timestamps]


def box_snapshot_to_state_for_boxes(
    snapshot: dict[str, Any],
    box_ids: tuple[str, ...] | list[str] | None = None,
) -> list[float]:
    ids = tuple(box_ids or box_ids_from_snapshot(snapshot))
    if ids == ("",):
        return box_snapshot_to_state(snapshot)
    state: list[float] = []
    for box_id in ids:
        state.extend(box_snapshot_to_state(_snapshot_for_box(snapshot, str(box_id))))
    return state


def box_snapshot_to_timestamps_for_boxes(
    snapshot: dict[str, Any],
    box_ids: tuple[str, ...] | list[str] | None = None,
) -> list[float]:
    ids = tuple(box_ids or box_ids_from_snapshot(snapshot))
    if ids == ("",):
        return box_snapshot_to_timestamps(snapshot)
    timestamps: list[float] = []
    for box_id in ids:
        timestamps.extend(box_snapshot_to_timestamps(_snapshot_for_box(snapshot, str(box_id))))
    return timestamps


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


def _parse_ffprobe_pts(stdout: str) -> list[float]:
    pts: list[float] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line:
            try:
                pts.append(float(line))
            except ValueError:
                pass
    return pts


def _extract_pts_ffprobe(mkv_path: Path, *, timeout_s: float) -> list[float] | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=p=0",
        str(mkv_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        logger.warning("ffprobe not found; falling back to GStreamer PTS extraction")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out on %s", mkv_path)
        return []
    if r.returncode != 0:
        logger.warning("ffprobe failed on %s: %s", mkv_path, r.stderr.strip()[:200])
        return []
    return _parse_ffprobe_pts(r.stdout)


def _extract_pts_gstreamer(mkv_path: Path, *, timeout_s: float) -> list[float]:
    try:
        import gi  # type: ignore

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore
    except Exception as exc:
        logger.warning("GStreamer PTS extraction unavailable: %s", exc)
        return []

    Gst.init(None)
    pipeline = Gst.Pipeline.new("thor-pts-probe")
    filesrc = Gst.ElementFactory.make("filesrc", None)
    demux = Gst.ElementFactory.make("matroskademux", None)
    sink = Gst.ElementFactory.make("appsink", None)
    if pipeline is None or filesrc is None or demux is None or sink is None:
        logger.warning("GStreamer PTS extraction missing required elements")
        return []

    filesrc.set_property("location", str(mkv_path))
    sink.set_property("emit-signals", False)
    sink.set_property("sync", False)
    sink.set_property("max-buffers", 4096)
    sink.set_property("drop", False)

    for element in (filesrc, demux, sink):
        pipeline.add(element)
    if not filesrc.link(demux):
        logger.warning(
            "GStreamer PTS extraction failed to link filesrc -> matroskademux"
        )
        pipeline.set_state(Gst.State.NULL)
        return []

    sink_pad = sink.get_static_pad("sink")
    linked = {"done": False}

    def _on_pad_added(_demux, pad) -> None:
        if linked["done"] or sink_pad is None or sink_pad.is_linked():
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_text = caps.to_string() if caps is not None else ""
        if "video/" not in caps_text:
            return
        if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
            linked["done"] = True

    demux.connect("pad-added", _on_pad_added)
    bus = pipeline.get_bus()
    pts: list[float] = []
    deadline = time.monotonic() + max(float(timeout_s), 0.1)

    try:
        state_ret = pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            logger.warning(
                "GStreamer PTS extraction failed to start pipeline for %s", mkv_path
            )
            return []
        while time.monotonic() < deadline:
            sample = sink.emit("try-pull-sample", int(0.2 * Gst.SECOND))
            if sample is not None:
                buf = sample.get_buffer()
                if buf is not None and buf.pts != Gst.CLOCK_TIME_NONE:
                    pts.append(float(buf.pts) / float(Gst.SECOND))
                continue
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.warning(
                    "GStreamer PTS extraction failed on %s: %s (%s)",
                    mkv_path, err, debug,
                )
                return []
            if msg.type == Gst.MessageType.EOS:
                break
        else:
            logger.warning("GStreamer PTS extraction timed out on %s", mkv_path)
    finally:
        pipeline.set_state(Gst.State.NULL)
    return pts


def extract_pts(mkv_path: Path, *, timeout_s: float = 30) -> list[float]:
    """Extract per-frame PTS (seconds) from an MKV.

    Prefer ffprobe when it is installed.  Thor images do not always include
    FFmpeg, so fall back to GStreamer, which is already required for capture.
    """

    pts = _extract_pts_ffprobe(mkv_path, timeout_s=timeout_s)
    if pts is not None:
        return pts
    return _extract_pts_gstreamer(mkv_path, timeout_s=timeout_s)


def calibrate_mcu_clock(
    mcu_timestamps: list[int],
    host_times: list[float],
) -> tuple[float, float, float]:
    """Fit mcu_timestamp → host_time via least-squares linear regression.

    Returns ``(slope, intercept, residual_std)`` where
    ``host_time ≈ slope * mcu_ts + intercept``.  *residual_std* is the
    standard deviation of the fit residuals (an estimate of calibration
    accuracy).  Returns ``(0, 0, inf)`` when there are fewer than 2 points.
    """
    n = len(mcu_timestamps)
    if n < 2:
        return (0.0, 0.0, float("inf"))
    sx = sum(mcu_timestamps)
    sy = sum(host_times)
    sxx = sum(x * x for x in mcu_timestamps)
    sxy = sum(x * y for x, y in zip(mcu_timestamps, host_times))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return (0.0, 0.0, float("inf"))
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    residuals = [y - (slope * x + intercept)
                 for x, y in zip(mcu_timestamps, host_times)]
    res_std = (sum(r * r for r in residuals) / n) ** 0.5
    return (slope, intercept, res_std)


def calibrate_sensor_samples(
    sensor_samples: dict[str, list[dict[str, Any]]],
    t0_wall_s: float,
) -> dict[str, list[dict[str, Any]]]:
    """Replace ``t_rel_s`` with MCU-clock-calibrated times when possible.

    For each sensor with enough samples, fits a linear MCU→host mapping and
    recomputes ``t_rel_s`` from MCU timestamps.  Sensors with <10 samples or
    poor fits (residual_std > 0.05s) keep their original poll-based times.

    Returns a *new* dict (original is not mutated).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, slist in sensor_samples.items():
        if len(slist) < 10:
            out[sid] = list(slist)
            continue
        mcu_ts = [s["data"].get("timestamp", 0) for s in slist]
        host_ts = [s.get("wall_s", t0_wall_s + s["t_rel_s"]) for s in slist]
        if not any(mcu_ts):
            out[sid] = list(slist)
            continue
        slope, intercept, res_std = calibrate_mcu_clock(mcu_ts, host_ts)
        if res_std > 0.05 or slope == 0.0:
            logger.warning("MCU clock calibration poor for %s (std=%.4fs); "
                           "keeping poll-based times", sid, res_std)
            out[sid] = list(slist)
            continue
        logger.info("MCU clock calibration for %s: slope=%.9f intercept=%.3f "
                    "residual_std=%.6fs (%d samples)",
                    sid, slope, intercept, res_std, len(slist))
        calibrated: list[dict[str, Any]] = []
        for s in slist:
            ts = s["data"].get("timestamp", 0)
            if ts:
                cal_wall = slope * ts + intercept
                cal_rel = cal_wall - t0_wall_s
            else:
                cal_rel = s["t_rel_s"]
            calibrated.append({**s, "t_rel_s": cal_rel})
        out[sid] = calibrated
    return out


def _nearest_sample_data(
    times: list[float],
    samples: list[dict[str, Any]],
    t: float,
) -> dict[str, Any]:
    """Find the sensor sample whose ``t_rel_s`` is nearest to *t*."""
    if not samples:
        return {}
    idx = bisect.bisect_left(times, t)
    if idx == 0:
        return samples[0].get("data", {})
    if idx >= len(samples):
        return samples[-1].get("data", {})
    if t - times[idx - 1] <= times[idx] - t:
        return samples[idx - 1].get("data", {})
    return samples[idx].get("data", {})


def _table_column_stats(table, col_name: str, *, width: int) -> dict[str, list]:
    """Per-channel min/max/mean/std/quantiles from a pyarrow Table column.

    Bypasses to_pylist()/sorted() which the old _stats(list[list[float]]) helper
    forced, and which produced O(N^2) Python heap growth as the dataset
    accumulated episodes (see development_status.md). Streams Arrow buffers
    into a single numpy view per column instead.

    ``width=1`` is for scalar columns (timestamp / frame_index / etc.) and
    returns length-1 lists matching the legacy schema. ``width>1`` is for
    fixed-size-list observation.state column and returns
    per-channel lists.
    """
    import numpy as np

    arr = table[col_name]
    n = arr.length()
    if n == 0:
        empty: list[float] = []
        return {
            "min": empty, "max": empty, "mean": empty, "std": empty,
            "count": [0],
            "q01": empty, "q10": empty, "q50": empty, "q90": empty, "q99": empty,
        }

    if width == 1:
        np_arr = arr.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        np_arr = np_arr.reshape(n, 1)
    else:
        flat = arr.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        np_arr = flat.reshape(n, width)

    quantiles = np.quantile(np_arr, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np_arr.min(axis=0).tolist(),
        "max": np_arr.max(axis=0).tolist(),
        "mean": np_arr.mean(axis=0).tolist(),
        "std": np_arr.std(axis=0).tolist(),
        "count": [int(n)],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _rows_to_table(
    pa,
    rows: list[dict[str, Any]],
    *,
    state_names: tuple[str, ...] | list[str] = BOX_STATE_NAMES,
    ts_names: tuple[str, ...] | list[str] = BOX_TIMESTAMP_NAMES,
):
    state_width = len(state_names)
    ts_width = len(ts_names)

    def vector_column(key: str, width: int, dtype):
        flat: list[float] = []
        for row in rows:
            values = list(row[key])
            if len(values) != width:
                raise ValueError(f"{key} width mismatch: {len(values)} != {width}")
            flat.extend(values)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype), width)

    return pa.table(
        [
            vector_column("observation.state", state_width, pa.float32()),
            vector_column("box.timestamps", ts_width, pa.float64()),
            pa.array([row["timestamp"] for row in rows], type=pa.float32()),
            pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            pa.array([row["index"] for row in rows], type=pa.int64()),
            pa.array([row["task_index"] for row in rows], type=pa.int64()),
        ],
        schema=_box_table_schema(pa, state_width=state_width, ts_width=ts_width),
    )


def _box_table_schema(
    pa,
    *,
    state_width: int = len(BOX_STATE_NAMES),
    ts_width: int = len(BOX_TIMESTAMP_NAMES),
):
    return pa.schema([
        ("observation.state", pa.list_(pa.float32(), state_width)),
        ("box.timestamps", pa.list_(pa.float64(), ts_width)),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])


def _box_features(
    state_names: tuple[str, ...] | list[str] = BOX_STATE_NAMES,
    ts_names: tuple[str, ...] | list[str] = BOX_TIMESTAMP_NAMES,
) -> dict[str, Any]:
    return {
        "observation.state": _feature("float32", [len(state_names)], list(state_names)),
        "box.timestamps": _feature("float64", [len(ts_names)], list(ts_names)),
        "timestamp": _feature("float32", [1]),
        "frame_index": _feature("int64", [1]),
        "episode_index": _feature("int64", [1]),
        "index": _feature("int64", [1]),
        "task_index": _feature("int64", [1]),
    }


def _build_episode_rows(
    *,
    fps: int,
    episode_index: int,
    snapshots: list[dict[str, Any]],
    duration_s: float | None = None,
    sensor_samples: dict[str, list[dict[str, Any]]] | None = None,
    t0_wall_s: float = 0.0,
    pts_offset_s: float | None = None,
    start_index: int = 0,
    box_ids: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    use_hf = bool(sensor_samples and any(sensor_samples.values()))
    resolved_box_ids = tuple(box_ids or box_ids_from_inputs(snapshots, sensor_samples))
    if not snapshots and not use_hf:
        return []

    frame_count = len(snapshots) if snapshots else 0
    if duration_s is not None and duration_s > 0:
        frame_count = max(frame_count, int(round(float(duration_s) * max(int(fps), 1))))

    rows: list[dict[str, Any]] = []
    if use_hf:
        calibrated = calibrate_sensor_samples(sensor_samples, t0_wall_s) if t0_wall_s else sensor_samples
        frame_origin_s = pts_offset_s if pts_offset_s is not None else 0.0

        per_sensor: dict[str, tuple[list[float], list[dict[str, Any]]]] = {}
        for sid, slist_raw in calibrated.items():
            slist = sorted(slist_raw, key=lambda s: s["t_rel_s"])
            per_sensor[sid] = ([s["t_rel_s"] for s in slist], slist)
        for local_frame in range(frame_count):
            timestamp_s = frame_origin_s + local_frame / max(int(fps), 1)
            sensors: dict[str, Any] = {}
            for sid, (times, slist) in per_sensor.items():
                data = _nearest_sample_data(times, slist, timestamp_s)
                if data:
                    sensors[sid] = data
            snap = {"valid": bool(sensors), "sensors": sensors}
            state = box_snapshot_to_state_for_boxes(snap, resolved_box_ids)
            rows.append({
                "observation.state": state,
                "box.timestamps": box_snapshot_to_timestamps_for_boxes(snap, resolved_box_ids),
                "timestamp": timestamp_s,
                "frame_index": local_frame,
                "episode_index": episode_index,
                "index": start_index + local_frame,
                "task_index": 0,
            })
    else:
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

        for local_frame in range(frame_count):
            timestamp_s = local_frame / max(int(fps), 1)
            snapshot = snapshot_for_timestamp(timestamp_s)
            state = box_snapshot_to_state_for_boxes(snapshot, resolved_box_ids)
            rows.append({
                "observation.state": state,
                "box.timestamps": box_snapshot_to_timestamps_for_boxes(snapshot, resolved_box_ids),
                "timestamp": timestamp_s,
                "frame_index": local_frame,
                "episode_index": episode_index,
                "index": start_index + local_frame,
                "task_index": 0,
            })

    return rows


class Lr3Writer:
    """Stateful BOX LeRobot v3 writer backed by one long-lived ParquetWriter."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        repo_id: str,
        task: str,
        fps: int,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.dataset_root = dataset_root
        self.repo_id = repo_id
        self.task = task
        self.fps = int(fps)
        self.pa = pa
        self.pq = pq
        self.data_dir = dataset_root / "data" / "chunk-000"
        self.meta_dir = dataset_root / "meta"
        self.episodes_dir = self.meta_dir / "episodes" / "chunk-000"
        self.data_path = self.data_dir / "file-000.parquet"
        self.episodes_path = self.episodes_dir / "file-000.parquet"
        self.total_frames = 0
        self._episode_rows: list[dict[str, Any]] = []
        self._episode_indices: set[int] = set()
        self._closed = False
        self._finalized = False
        self.state_names: tuple[str, ...] = BOX_STATE_NAMES
        self.ts_names: tuple[str, ...] = BOX_TIMESTAMP_NAMES
        self._schema = None
        self._writer = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        if self.data_path.exists():
            raise FileExistsError(
                f"{self.data_path} already exists; Lr3Writer cannot append to an existing parquet file"
            )
        self._write_tasks()
        self._write_episodes()
        self._write_info()

    def append_episode(
        self,
        *,
        episode_index: int,
        snapshots: list[dict[str, Any]],
        duration_s: float | None = None,
        sensor_samples: dict[str, list[dict[str, Any]]] | None = None,
        t0_wall_s: float = 0.0,
        pts_offset_s: float | None = None,
    ) -> Path | None:
        if self._closed:
            raise RuntimeError("cannot append to a closed Lr3Writer")
        if int(episode_index) in self._episode_indices:
            raise ValueError(f"episode_index {episode_index} was already appended")

        box_ids = box_ids_from_inputs(snapshots, sensor_samples)
        state_names = box_state_names(box_ids)
        ts_names = box_timestamp_names(box_ids)
        if self._writer is not None and (state_names != self.state_names or ts_names != self.ts_names):
            raise ValueError(
                "BOX schema changed across episodes: "
                f"state {len(state_names)} != {len(self.state_names)} or "
                f"timestamps {len(ts_names)} != {len(self.ts_names)}"
            )

        rows = _build_episode_rows(
            fps=self.fps,
            episode_index=episode_index,
            snapshots=snapshots,
            duration_s=duration_s,
            sensor_samples=sensor_samples,
            t0_wall_s=t0_wall_s,
            pts_offset_s=pts_offset_s,
            start_index=self.total_frames,
            box_ids=box_ids,
        )
        if not rows:
            return None

        if self._writer is None:
            self.state_names = state_names
            self.ts_names = ts_names
            self._schema = _box_table_schema(
                self.pa,
                state_width=len(self.state_names),
                ts_width=len(self.ts_names),
            )
            self._writer = self.pq.ParquetWriter(
                self.data_path,
                schema=self._schema,
                compression="snappy",
                use_dictionary=True,
            )

        table = _rows_to_table(self.pa, rows, state_names=self.state_names, ts_names=self.ts_names)
        self._writer.write_table(table)
        n_rows = table.num_rows
        start = self.total_frames
        stop = start + n_rows
        self.total_frames = stop
        self._episode_indices.add(int(episode_index))
        self._episode_rows.append({
            "episode_index": int(episode_index),
            "tasks": [self.task],
            "length": int(n_rows),
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(start),
            "dataset_to_index": int(stop),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        })
        self._write_episodes()
        self._write_info()
        return self.data_path

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._writer is None:
            self._write_episodes()
            self._write_info()
            self._write_tasks()
            self._finalized = True
            return
        self.close()
        table = self.pq.read_table(self.data_path)
        self._write_stats(table)
        self._write_episodes()
        self._write_info()
        self._write_tasks()
        self._finalized = True

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    def _write_stats(self, table) -> None:
        state_width = len(self.state_names)
        stats = {
            "observation.state": _table_column_stats(table, "observation.state", width=state_width),
            "box.timestamps": _table_column_stats(table, "box.timestamps", width=len(self.ts_names)),
            "timestamp": _table_column_stats(table, "timestamp", width=1),
            "frame_index": _table_column_stats(table, "frame_index", width=1),
            "episode_index": _table_column_stats(table, "episode_index", width=1),
            "index": _table_column_stats(table, "index", width=1),
            "task_index": _table_column_stats(table, "task_index", width=1),
        }
        (self.meta_dir / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    def _write_episodes(self) -> None:
        rows = sorted(self._episode_rows, key=lambda row: int(row["episode_index"]))
        if rows:
            table = self.pa.Table.from_pylist(rows)
        else:
            table = self.pa.table({
                "episode_index": self.pa.array([], type=self.pa.int64()),
                "tasks": self.pa.array([], type=self.pa.list_(self.pa.string())),
                "length": self.pa.array([], type=self.pa.int64()),
                "data/chunk_index": self.pa.array([], type=self.pa.int64()),
                "data/file_index": self.pa.array([], type=self.pa.int64()),
                "dataset_from_index": self.pa.array([], type=self.pa.int64()),
                "dataset_to_index": self.pa.array([], type=self.pa.int64()),
                "meta/episodes/chunk_index": self.pa.array([], type=self.pa.int64()),
                "meta/episodes/file_index": self.pa.array([], type=self.pa.int64()),
            })
        self.pq.write_table(table, self.episodes_path)

    def _write_info(self) -> None:
        n_eps = len(self._episode_rows)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "thor_gmsl2_box",
            "repo_id": self.repo_id,
            "total_episodes": int(n_eps),
            "total_frames": int(self.total_frames),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": int(self.fps),
            "splits": {"train": f"0:{n_eps}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": _box_features(self.state_names, self.ts_names),
        }
        (self.meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    def _write_tasks(self) -> None:
        self.pq.write_table(
            self.pa.Table.from_pylist([{"task_index": 0, "task": self.task}]),
            self.meta_dir / "tasks.parquet",
        )

    def __enter__(self) -> "Lr3Writer":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.finalize()


def open_box_lerobot_v3_writer(
    dataset_root: Path,
    *,
    repo_id: str,
    task: str,
    fps: int,
) -> Lr3Writer | None:
    try:
        return Lr3Writer(dataset_root, repo_id=repo_id, task=task, fps=fps)
    except ImportError:
        return None

def write_box_lerobot_v3_episode(
    dataset_root: Path,
    *,
    repo_id: str,
    task: str,
    fps: int,
    episode_index: int,
    snapshots: list[dict[str, Any]],
    duration_s: float | None = None,
    sensor_samples: dict[str, list[dict[str, Any]]] | None = None,
    t0_wall_s: float = 0.0,
    pts_offset_s: float | None = None,
) -> Path | None:
    """Append BOX snapshots for one episode to a minimal LeRobot v3 dataset.

    When *sensor_samples* is provided (high-frequency per-sensor data from
    ``BoxClient.stop_recording``), each camera frame is composed via
    per-sensor nearest-neighbor interpolation for maximum accuracy.
    Otherwise falls back to the legacy snapshot path.

    Enhanced alignment (L3b):
      - *t0_wall_s*: episode start wall time, used for MCU clock calibration.
      - *pts_offset_s*: first-frame PTS from the reference camera MKV.  When
        provided, the frame time grid shifts by this offset so that frame 0
        aligns with the actual first capture instant rather than the pipeline
        spawn time.

    Returns the parquet path when rows were written. If ``pyarrow`` is missing
    or no data was captured, returns ``None`` so the recorder can keep
    the hardware capture path alive.
    """

    use_hf = bool(sensor_samples and any(sensor_samples.values()))
    if not snapshots and not use_hf:
        return None
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception:
        return None

    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "file-000.parquet"

    box_ids = box_ids_from_inputs(snapshots, sensor_samples)
    state_names = box_state_names(box_ids)
    ts_names = box_timestamp_names(box_ids)
    rows = _build_episode_rows(
        fps=fps,
        episode_index=episode_index,
        snapshots=snapshots,
        duration_s=duration_s,
        sensor_samples=sensor_samples,
        t0_wall_s=t0_wall_s,
        pts_offset_s=pts_offset_s,
        box_ids=box_ids,
    )
    if not rows:
        return None

    # Merge current-episode rows into existing parquet using Arrow Table
    # operations end-to-end. The old path did pq.read_table().to_pylist()
    # then sorted([*existing, *rows]) then rebuilt every row dict, which
    # peaked at N * per_ep_rows * ~10 Python dict copies per episode and
    # caused permanent glibc malloc arena growth — ~7.7 MB/ep on Thor
    # at 50 ep with no GStreamer involved (see development_status.md).
    new_table = _rows_to_table(pa, rows, state_names=state_names, ts_names=ts_names)
    if data_path.exists():
        existing_table = pq.read_table(data_path)
        # Drop any prior rows for this same episode_index (idempotent overwrite).
        mask = pc.not_equal(
            existing_table["episode_index"],
            pa.scalar(int(episode_index), type=pa.int64()),
        )
        existing_table = existing_table.filter(mask)
        combined = pa.concat_tables([existing_table, new_table])
    else:
        combined = new_table
    combined = combined.sort_by(
        [("episode_index", "ascending"), ("frame_index", "ascending")]
    )
    n_total = combined.num_rows
    combined = combined.set_column(
        combined.schema.get_field_index("index"),
        "index",
        pa.array(range(n_total), type=pa.int64()),
    )
    pq.write_table(combined, data_path)

    # Per-column stats via numpy on Arrow buffers (no row-level Python loop).
    state_width = len(state_names)
    stats = {
        "observation.state": _table_column_stats(combined, "observation.state", width=state_width),
        "box.timestamps": _table_column_stats(combined, "box.timestamps", width=len(ts_names)),
        "timestamp": _table_column_stats(combined, "timestamp", width=1),
        "frame_index": _table_column_stats(combined, "frame_index", width=1),
        "episode_index": _table_column_stats(combined, "episode_index", width=1),
        "index": _table_column_stats(combined, "index", width=1),
        "task_index": _table_column_stats(combined, "task_index", width=1),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    # Episode-level rollup via group_by aggregation on Arrow. group_by
    # produces columns named "<col>_<agg>" (e.g. index_min, index_count).
    ep_groups = combined.select(["episode_index", "index"]).group_by("episode_index").aggregate([
        ("index", "min"),
        ("index", "max"),
        ("index", "count"),
    ]).sort_by([("episode_index", "ascending")])
    n_eps = ep_groups.num_rows
    ep_idx_list = ep_groups["episode_index"].to_pylist()
    ep_count_list = ep_groups["index_count"].to_pylist()
    ep_from_list = ep_groups["index_min"].to_pylist()
    ep_to_list = [int(v) + 1 for v in ep_groups["index_max"].to_pylist()]
    ep_table = pa.table({
        "episode_index": pa.array(ep_idx_list, type=pa.int64()),
        "tasks": pa.array([[task]] * n_eps, type=pa.list_(pa.string())),
        "length": pa.array(ep_count_list, type=pa.int64()),
        "data/chunk_index": pa.array([0] * n_eps, type=pa.int64()),
        "data/file_index": pa.array([0] * n_eps, type=pa.int64()),
        "dataset_from_index": pa.array(ep_from_list, type=pa.int64()),
        "dataset_to_index": pa.array(ep_to_list, type=pa.int64()),
        "meta/episodes/chunk_index": pa.array([0] * n_eps, type=pa.int64()),
        "meta/episodes/file_index": pa.array([0] * n_eps, type=pa.int64()),
    })
    pq.write_table(ep_table, episodes_dir / "file-000.parquet")

    features = {
        "observation.state": _feature("float32", [len(state_names)], list(state_names)),
        "box.timestamps": _feature("float64", [len(ts_names)], list(ts_names)),
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
        "total_episodes": int(n_eps),
        "total_frames": int(n_total),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": int(fps),
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": task}]), meta_dir / "tasks.parquet")
    return data_path
