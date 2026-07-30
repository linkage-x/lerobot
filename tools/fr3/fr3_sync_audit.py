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

"""Offline timestamp-synchronisation audit for FR3 ee2ee datasets.

Every FR3 recording writes an ``observation.device_capture_timestamp`` column holding one
wall-clock reading per device (arm, gripper, and each camera) sampled at the instant that
device's value entered the frame. This tool turns that column into a verdict: are the
modalities inside one frame actually simultaneous, and does the frame as a whole sit on the
dataset's nominal ``frame_index / fps`` grid?

Clock semantics -- read this before interpreting the numbers
------------------------------------------------------------
The two backends do not mean the same thing by "capture timestamp", and the report says which
one produced it in ``clock_semantics`` rather than silently mixing them:

``hardware_mixed``
    Arm and gripper columns are host ``perf_counter`` readings taken immediately after their
    driver read returns, inside ``get_observation()``. Camera columns are the **acquisition**
    instant: RealSense reports it on the device clock, global time maps that onto the host wall
    clock, and the camera moves it onto the ``perf_counter`` basis by subtracting the frame's
    age at handover. So the two are comparable, and a camera column is *not* the exposure
    midpoint -- it is what the sensor reports as acquisition.

    Cameras are older than the arm read by construction: the arm is read on demand while a
    frame already exists by the time anything asks for it. On the FR3 rig that gap measures
    42-45 ms at 30 fps, stable to ~4 ms -- a real image-vs-state offset, not a clock error, and
    one that halves if the cameras run at 60 fps. Between the two cameras the soft-sync brings
    skew down to ~3 ms. Constant biases are reported per device rather than corrected away.

    Before this was measured from acquisition, camera columns carried each camera's pipeline
    delay instead (a D405 hands frames over 4.8 ms after acquisition, a D435i 29.1 ms), which
    both put a fake 24 ms between the two cameras and understated how stale the images were.

``sim_extraction_wallclock``
    In MuJoCo every modality is extracted from the *same* physics instant, so there is no
    physical acquisition skew to measure at all. The timestamps record how long extraction
    took (state read, then one render per camera). They are useful for catching a straggling
    render that would stall the control loop, and are **not** comparable to hardware sensor
    timestamps.

The report never rewrites the dataset. It is written next to the data so any later training or
replay run can be audited against the alignment that was actually achieved at capture time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.handheld.handheld_soft_sync import build_report as build_soft_sync_report  # noqa: E402

DEFAULT_TOLERANCE_MS = 20.0
DEFAULT_GLOBAL_LAG_TOLERANCE_MS = 50.0
REPORT_RELATIVE_PATH = Path("meta") / "fr3_sync_report.json"
CLOCK_SEMANTICS = ("hardware_mixed", "sim_extraction_wallclock")


def _classify_devices(names: list[str]) -> dict[str, list[str]]:
    """Group capture-timestamp columns by modality so the summary stays readable."""
    groups: dict[str, list[str]] = {"arm": [], "gripper": [], "camera": [], "other": []}
    for name in names:
        if name.startswith("camera."):
            groups["camera"].append(name)
        elif "_gripper." in name or name.startswith("gripper."):
            groups["gripper"].append(name)
        elif name.startswith("fr3.arm."):
            groups["arm"].append(name)
        else:
            groups["other"].append(name)
    return groups


def _infer_clock_semantics(dataset_root: Path, names: list[str]) -> str:
    """Decide which clock produced these timestamps, from the dataset itself.

    The sim robot names its gripper column ``sim_gripper.capture_timestamp_s`` and records
    ``robot_type=franka_research3_mujoco``; either signal is enough. Guessing is not
    acceptable here -- a wrong label would make a sim report look like a hardware sync claim.
    """
    if any(name.startswith("sim_gripper.") for name in names):
        return "sim_extraction_wallclock"
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        try:
            robot_type = str(json.loads(info_path.read_text(encoding="utf-8")).get("robot_type") or "")
        except (OSError, json.JSONDecodeError):
            robot_type = ""
        if robot_type.endswith("_mujoco"):
            return "sim_extraction_wallclock"
    return "hardware_mixed"


def _cross_modality_bias_ms(report: dict[str, Any], groups: dict[str, list[str]]) -> dict[str, float]:
    """Median lag of each modality relative to the arm read, in milliseconds.

    A constant bias here is a fixed pipeline offset (exposure/readout, driver handover), not a
    per-episode alignment failure -- it is reported so it can be characterised, never
    subtracted silently.
    """
    per_device = report.get("per_device_lag_s", {})
    arm_names = groups["arm"]
    if not arm_names:
        return {}
    arm_reference = per_device.get(arm_names[0], {}).get("p50")
    if arm_reference is None:
        return {}
    bias: dict[str, float] = {}
    for name, stats in per_device.items():
        median = stats.get("p50")
        if median is None or not np.isfinite(median):
            continue
        bias[name] = float((median - arm_reference) * 1e3)
    return bias


def summarize_episode_capture_timestamps(
    *,
    capture_timestamps: np.ndarray,
    frame_timestamps: np.ndarray,
    device_names: list[str],
    clock_semantics: str,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    global_lag_tolerance_ms: float = DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
) -> dict[str, Any]:
    """Audit one episode straight from the in-memory frame buffer.

    A LeRobot v3 dataset keeps appending into one open parquet file until ``finalize()``, so an
    episode cannot be re-read from disk the moment it is saved. Computing the same statistics
    from the buffer is what lets the operator see a skew problem on the episode that caused it
    instead of at the end of a twenty-episode session. The file-based
    :func:`build_fr3_sync_report` still runs at finalize and remains the persisted record.
    """
    capture_timestamps = np.asarray(capture_timestamps, dtype=np.float64)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    if capture_timestamps.ndim != 2:
        raise ValueError(f"capture_timestamps must be 2D, got shape {capture_timestamps.shape}.")
    if capture_timestamps.shape[0] != frame_timestamps.shape[0]:
        raise ValueError(
            "capture_timestamps and frame_timestamps must have the same frame count "
            f"({capture_timestamps.shape[0]} vs {frame_timestamps.shape[0]})."
        )
    if capture_timestamps.shape[1] != len(device_names):
        raise ValueError(
            f"{len(device_names)} device names for width {capture_timestamps.shape[1]}."
        )

    frames = int(capture_timestamps.shape[0])
    finite_mask = np.all(np.isfinite(capture_timestamps), axis=1)
    max_skew_s = np.full(frames, np.nan, dtype=np.float64)
    grid_lag_s = np.full(frames, np.nan, dtype=np.float64)
    if finite_mask.any():
        finite_capture = capture_timestamps[finite_mask]
        max_skew_s[finite_mask] = np.max(finite_capture, axis=1) - np.min(finite_capture, axis=1)
        grid_lag_s[finite_mask] = np.median(finite_capture, axis=1) - frame_timestamps[finite_mask]

    skew_limit_s = tolerance_ms / 1000.0
    lag_limit_s = global_lag_tolerance_ms / 1000.0
    nonfinite_frames = int(np.sum(~finite_mask))
    skew_bad = int(np.sum(max_skew_s > skew_limit_s))
    lag_bad = int(np.sum(np.abs(grid_lag_s) > lag_limit_s))

    def _stat(values: np.ndarray, reducer) -> float:
        finite = values[np.isfinite(values)]
        return float(reducer(finite)) if finite.size else 0.0

    # The dataset's own `timestamp` column is the nominal frame_index/fps grid, not a
    # measurement. When the control loop cannot keep up, frames land at a wider real spacing
    # while still being labelled as evenly spaced -- a policy trained on that data would
    # assume a cadence the robot never achieved. Comparing measured spacing against the
    # nominal one names that failure directly instead of leaving it as cumulative drift.
    nominal_interval_s = float(np.median(np.diff(frame_timestamps))) if frames > 1 else 0.0
    frame_centres_s = np.median(capture_timestamps, axis=1)
    frame_centres_s = frame_centres_s[np.isfinite(frame_centres_s)]
    measured_intervals_s = np.diff(frame_centres_s) if frame_centres_s.size > 1 else np.array([])
    # Averaged across the episode rather than taken as a median of per-frame gaps. The question
    # this answers -- did the loop hold the nominal cadence -- is about elapsed time, and
    # per-frame gaps are not symmetric: a frame that lands late is followed by one that lands
    # early, so their median sits above the true average and reports drift the episode does not
    # have. Measured on the hardware rig: median gap 35.4 ms against a mean of 33.34 ms, for a
    # 30 fps episode whose total duration was correct to 0.03%. The median reading would have
    # condemned a cadence that was in fact exact.
    measured_interval_s = (
        float(frame_centres_s[-1] - frame_centres_s[0]) / (frame_centres_s.size - 1)
        if frame_centres_s.size > 1
        else 0.0
    )

    arm_index = next((i for i, name in enumerate(device_names) if name.startswith("fr3.arm.")), None)
    bias_ms: dict[str, float] = {}
    if arm_index is not None and finite_mask.any():
        device_lag = capture_timestamps - frame_timestamps[:, None]
        arm_median = _stat(device_lag[:, arm_index], np.median)
        for index, name in enumerate(device_names):
            bias_ms[name] = (_stat(device_lag[:, index], np.median) - arm_median) * 1e3

    return {
        "clock_semantics": clock_semantics,
        "frames": frames,
        "device_capture_timestamp_names": list(device_names),
        "nonfinite_capture_timestamp_frames": nonfinite_frames,
        "skew_over_tolerance_frames": skew_bad,
        "global_lag_over_tolerance_frames": lag_bad,
        "max_skew_ms": _stat(max_skew_s, np.max) * 1e3,
        "p95_skew_ms": _stat(max_skew_s, lambda v: np.percentile(v, 95)) * 1e3,
        "grid_lag_p95_ms": _stat(np.abs(grid_lag_s), lambda v: np.percentile(v, 95)) * 1e3,
        "nominal_frame_interval_ms": nominal_interval_s * 1e3,
        "measured_frame_interval_ms": measured_interval_s * 1e3,
        "measured_frame_interval_p95_ms": (
            float(np.percentile(measured_intervals_s, 95)) * 1e3 if measured_intervals_s.size else 0.0
        ),
        "cross_modality_bias_ms": bias_ms,
        "limits": {"max_skew_ms": tolerance_ms, "abs_global_lag_ms": global_lag_tolerance_ms},
        "status": "pass" if not (nonfinite_frames or skew_bad or lag_bad) else "fail",
    }


def format_episode_sync_line(summary: dict[str, Any], *, episode: int) -> str:
    return (
        f"episode={episode} status={summary['status']} clock={summary['clock_semantics']} "
        f"frames={summary['frames']} skew_p95_ms={summary['p95_skew_ms']:.2f} "
        f"skew_max_ms={summary['max_skew_ms']:.2f} grid_lag_p95_ms={summary['grid_lag_p95_ms']:.2f} "
        f"interval_ms={summary['measured_frame_interval_ms']:.1f}"
        f"/{summary['nominal_frame_interval_ms']:.1f}nominal "
        f"bad_skew_frames={summary['skew_over_tolerance_frames']} "
        f"bad_lag_frames={summary['global_lag_over_tolerance_frames']}"
    )


def build_fr3_sync_report(
    *,
    dataset_root: Path,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    global_lag_tolerance_ms: float = DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
) -> dict[str, Any]:
    report = build_soft_sync_report(
        dataset_root=dataset_root,
        tolerance_ms=tolerance_ms,
        global_lag_tolerance_ms=global_lag_tolerance_ms,
    )
    names = list(report["device_capture_timestamp_names"])
    groups = _classify_devices(names)
    clock_semantics = _infer_clock_semantics(dataset_root, names)

    summary = report["summary"]
    frames = int(report["total_frames"])
    skew_bad = int(summary["skew_over_tolerance_frames"])
    lag_bad = int(summary["global_lag_over_tolerance_frames"])
    nonfinite = int(summary["nonfinite_capture_timestamp_frames"])

    failures: list[str] = []
    if frames == 0:
        failures.append("dataset contains no frames")
    if nonfinite:
        failures.append(f"{nonfinite} frame(s) have a non-finite capture timestamp")
    if skew_bad:
        failures.append(
            f"{skew_bad}/{frames} frame(s) exceed the {tolerance_ms:.1f} ms intra-frame skew budget"
        )
    if lag_bad:
        failures.append(
            f"{lag_bad}/{frames} frame(s) drift more than {global_lag_tolerance_ms:.1f} ms "
            "from the nominal frame grid"
        )

    report["schema_version"] = 2
    report["report_kind"] = "fr3_sync_audit"
    report["clock_semantics"] = clock_semantics
    report["device_groups"] = groups
    report["cross_modality_bias_ms"] = _cross_modality_bias_ms(report, groups)
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    if clock_semantics == "sim_extraction_wallclock":
        report["interpretation"] = (
            "MuJoCo backend: all modalities come from one physics instant, so these timestamps "
            "measure extraction cost (state read, then per-camera render), not sensor acquisition "
            "skew. They are not comparable to hardware sensor timestamps."
        )
    else:
        report["interpretation"] = (
            "Hardware backend: arm and gripper columns are host reads taken when their driver "
            "read returns; camera columns are the acquisition instant reported by the sensor "
            "(device clock via global time, moved onto the host monotonic basis), so they are "
            "neither exposure midpoint nor driver handover. Cameras are older than the arm read "
            "by construction -- the arm is read on demand, a frame already exists when asked "
            "for -- which measures 42-45 ms at 30 fps on this rig and halves at 60 fps. A "
            "constant camera-vs-arm bias is a real image-vs-state offset and is reported, not "
            "corrected."
        )
    return report


def write_fr3_sync_report(
    dataset_root: Path,
    *,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    global_lag_tolerance_ms: float = DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    report = build_fr3_sync_report(
        dataset_root=dataset_root,
        tolerance_ms=tolerance_ms,
        global_lag_tolerance_ms=global_lag_tolerance_ms,
    )
    destination = output_path if output_path is not None else dataset_root / REPORT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, destination


def format_sync_summary_line(report: dict[str, Any]) -> str:
    """One-line digest for the recorder's stdout protocol / gateway event log."""
    summary = report["summary"]
    max_skew_ms = float(summary["max_skew_s"]["max"] or 0.0) * 1e3
    p95_skew_ms = float(summary["max_skew_s"]["p95"] or 0.0) * 1e3
    lag_p95_ms = float(summary["global_lag_s"]["p95"] or 0.0) * 1e3
    return (
        f"status={report['status']} clock={report['clock_semantics']} "
        f"frames={report['total_frames']} devices={len(report['device_capture_timestamp_names'])} "
        f"skew_p95_ms={p95_skew_ms:.2f} skew_max_ms={max_skew_ms:.2f} "
        f"grid_lag_p95_ms={lag_p95_ms:.2f} "
        f"bad_skew_frames={summary['skew_over_tolerance_frames']} "
        f"bad_lag_frames={summary['global_lag_over_tolerance_frames']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit per-device capture-timestamp alignment of an FR3 ee2ee dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root to audit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Report path. Defaults to <dataset>/{REPORT_RELATIVE_PATH.as_posix()}.",
    )
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--global-lag-tolerance-ms", type=float, default=DEFAULT_GLOBAL_LAG_TOLERANCE_MS)
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero when the audit status is 'fail'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance_ms < 0 or args.global_lag_tolerance_ms < 0:
        raise ValueError("tolerances must be >= 0.")
    report, destination = write_fr3_sync_report(
        args.dataset.resolve(),
        tolerance_ms=args.tolerance_ms,
        global_lag_tolerance_ms=args.global_lag_tolerance_ms,
        output_path=args.output,
    )
    print(f"fr3_sync_report={destination}")
    print(format_sync_summary_line(report))
    for name, bias_ms in sorted(report["cross_modality_bias_ms"].items()):
        print(f"bias_vs_arm_ms[{name}]={bias_ms:.2f}")
    for failure in report["failures"]:
        print(f"WARN: {failure}")
    if args.fail_on_violation and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
