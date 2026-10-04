"""Align coherent FR3 samples to measured Thor camera times using only stdlib.

All times belong to Thor's CLOCK_MONOTONIC domain. Missing hardware camera
times never fall back to frame_index/fps; missing or old samples remain invalid.
The native O_T_EE vector retains libfranka's column-major end-effector frame.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any

_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
_TCP_NAMES = ["x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"]
_WRENCH_NAMES = ["fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm"]

# name -> (Arrow primitive, width, feature names). Scalar flags deliberately
# use one-element vectors, matching their declared LeRobot feature shape.
FR3_COLUMNS: dict[str, tuple[str, int, list[str]]] = {
    "observation.fr3.q": ("float32", 7, _JOINT_NAMES),
    "observation.fr3.dq": ("float32", 7, _JOINT_NAMES),
    "observation.fr3.tau_J": ("float32", 7, _JOINT_NAMES),
    "observation.fr3.tau_ext_hat_filtered": ("float32", 7, _JOINT_NAMES),
    "observation.fr3.O_T_EE": ("float32", 16, [f"column_{c}_row_{r}" for c in range(4) for r in range(4)]),
    "observation.fr3.O_F_ext_hat_K": ("float32", 6, _WRENCH_NAMES),
    "observation.fr3.tcp": ("float32", 6, _TCP_NAMES),
    "action": ("float32", 7, [*_TCP_NAMES, "gripper_command_0_to_1"]),
    "fr3.timestamps": ("float64", 2, ["sample_monotonic_s", "receiver_monotonic_s"]),
    "fr3.valid": ("float32", 1, ["valid"]),
    "fr3.action_valid": ("float32", 1, ["valid"]),
    "fr3.control_command_success_rate": ("float32", 1, ["success_rate"]),
}
_OBSERVATION_KEYS = {
    "q": "observation.fr3.q",
    "dq": "observation.fr3.dq",
    "tau_J": "observation.fr3.tau_J",
    "tau_ext_hat_filtered": "observation.fr3.tau_ext_hat_filtered",
    "O_T_EE": "observation.fr3.O_T_EE",
    "O_F_ext_hat_K": "observation.fr3.O_F_ext_hat_K",
    "measured_tcp": "observation.fr3.tcp",
}


def features() -> dict[str, dict[str, Any]]:
    return {key: {"dtype": dtype, "shape": [width], "names": list(names)}
            for key, (dtype, width, names) in FR3_COLUMNS.items()}


def schema_fields(pa) -> list[tuple[str, Any]]:
    """Accept an already-imported pyarrow module; importing this helper is light."""
    return [(key, pa.list_(getattr(pa, dtype)(), width))
            for key, (dtype, width, _) in FR3_COLUMNS.items()]


def empty_fr3_row() -> dict[str, list[float]]:
    return {key: [0.0] * width for key, (_, width, _) in FR3_COLUMNS.items()}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _vector(value: Any, width: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    numbers = [_number(item) for item in value]
    return None if any(item is None for item in numbers) else numbers


def load_fr3_samples(ep_dir: Path) -> list[dict[str, Any]]:
    """Read the raw per-episode JSONL archive; malformed lines do not fabricate samples."""
    path = ep_dir / "fr3_state.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            out = []
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    out.append(item)
            return out
    except OSError:
        return []


def align_fr3_samples(
    fr3_samples: list[dict[str, Any]] | None,
    frame_times_s: list[float | None] | None,
    t0_mono_s: float | None,
    *,
    n_frames: int,
    max_skew_s: float = 0.025,
) -> list[dict[str, list[float]]]:
    """Nearest sample at each actual camera exposure time, with explicit validity.

    ``frame_times_s`` is the existing camera hardware timeline relative to
    ``t0_mono_s``. A missing entry, timestamp origin or sample outside the skew
    budget produces zeros and both validity flags zero. Observation and action
    completeness are checked separately; a missing gripper command does not
    erase otherwise coherent measured joints/torque.
    """
    if n_frames < 0 or not math.isfinite(max_skew_s) or max_skew_s < 0:
        raise ValueError("n_frames and max_skew_s must be non-negative")
    result = [empty_fr3_row() for _ in range(n_frames)]
    origin = _number(t0_mono_s)
    if origin is None or origin <= 0 or frame_times_s is None:
        return result
    ordered: list[tuple[float, dict[str, Any]]] = []
    for sample in fr3_samples or []:
        source = _number(sample.get("sample_monotonic_s"))
        receive = _number(sample.get("receiver_monotonic_s"))
        uncertainty = _number(sample.get("clock_uncertainty_s", 0))
        if sample.get("clock_sync_valid", True) is not True or uncertainty is None or not 0 <= uncertainty <= .01:
            continue
        if source is not None and source > 0 and receive is not None and receive + uncertainty >= source:
            ordered.append((source, sample))
    ordered.sort(key=lambda pair: pair[0])
    times = [time for time, _ in ordered]
    if not times:
        return result
    for frame in range(min(n_frames, len(frame_times_s))):
        relative = _number(frame_times_s[frame])
        if relative is None:
            continue
        target = origin + relative
        at = bisect.bisect_left(times, target)
        candidates = [i for i in (at - 1, at) if 0 <= i < len(times)]
        nearest = min(candidates, key=lambda i: abs(times[i] - target))
        sample = ordered[nearest][1]
        if abs(times[nearest] - target) + float(sample.get("clock_uncertainty_s", 0)) > max_skew_s + 1e-12:
            continue
        row = result[frame]
        row["fr3.timestamps"] = [times[nearest], float(sample["receiver_monotonic_s"])]
        observations = {column: _vector(sample.get(raw_key), FR3_COLUMNS[column][1])
                        for raw_key, column in _OBSERVATION_KEYS.items()}
        success = _number(sample.get("control_command_success_rate"))
        if all(values is not None for values in observations.values()) and success is not None and 0 <= success <= 1:
            row.update(observations)
            row["fr3.valid"] = [1.0]
            row["fr3.control_command_success_rate"] = [success]
        command = _vector(sample.get("commanded_ee"), 6)
        gripper = _number(sample.get("gripper_command"))
        if command is not None and gripper is not None and 0 <= gripper <= 1:
            row["action"] = [*command, gripper]
            row["fr3.action_valid"] = [1.0]
    return result


def align_fr3_rows_by_frame_index(
    rows: list[dict[str, Any]], n_frames: int,
) -> list[dict[str, list[float]]]:
    """Preserve recorder-aligned rows; a missing camera row never holds a stale action."""
    by_frame = {int(row["frame_index"]): row for row in rows}
    result = []
    for frame in range(n_frames):
        source = by_frame.get(frame)
        row = empty_fr3_row()
        if source is not None:
            values = {key: _vector(source.get(key), width) for key, (_, width, _) in FR3_COLUMNS.items()}
            if all(value is not None for value in values.values()):
                row.update(values)
        result.append(row)
    return result


def columns_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [row[key] for row in rows] for key in FR3_COLUMNS}
