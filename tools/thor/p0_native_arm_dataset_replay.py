#!/usr/bin/env python3
"""Replay a UI-generated left/right EE sidecar with the audited native P0 arm path.

This is the dataset companion to the sealed 2026-09-08 P0 native runner.  It
loads one episode from the trajectory sidecar written by Dataset Processing,
solves a continuous FR3 joint path offline against the frozen P0 URDF, and then
hands those joints to the native controller.  Dataset gripper mode reads the
recorded Corenetic opening width and follows joint-path progress during replay.

No robot object is created unless ``--execute`` is supplied and the operator
confirms from an interactive terminal.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import numpy as np


REPO_ROOT = Path("/home/nvidia/lerobot")
BUNDLE_ROOT = Path("/home/nvidia/box_api/replay_p0_native_arm_only_20260908")
ARM_ROOT = Path("/home/nvidia/box_api/replay_p0_arm_only_20260908")
SOURCE_ROOT = Path("/home/nvidia/box_api/replay_p0_once_20260908")
DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs/datasets/thor_gmsl2_9ch_v1_20260921_163918"
SIDECAR_RELATIVE_DIR = Path("derived/april_cube_tracking_in_robot_base")
SIDES = ("left", "right")
ROBOT_IP = "192.168.11.102"
WIDTH_MAX_MM = 89.05
POSITION_TOLERANCE_M = 0.012
ORIENTATION_TOLERANCE_DEG = 6.0
MAX_JOINT_STEP_ABS_RAD = 0.35
MAX_JOINT_STEP_L2_RAD = 0.70
NATIVE_SPEED_FACTOR = 0.01
START_SPEED_FACTOR = 0.03
CHUNK_SIZE = 10
CORENETIC_MAX_WIDTH_M = 0.09
CORENETIC_COMMAND_RATE_HZ = 15.0
IK_INITIAL_POSE_NAME = "current_start_pose_20260828_2048"
IK_INITIAL_JOINTS_RAD = np.asarray(
    [
        -0.2982022354854064,
        -0.20546837567339093,
        0.2008775163648066,
        -2.707162497847623,
        -0.09350475554363503,
        2.9366955831629005,
        0.8043834376561214,
    ],
    dtype=np.float64,
)
MULTISTART_SEED_COUNT = 256
HALTON_BASES = (2, 3, 5, 7, 11, 13, 17)
TRACKING_RUN_SUFFIX = "thor_april_tracking_in_robot_base"
NETWORK_GATE_ATTEMPTS = 3
MAX_DEVIATION_RAD = 0.02
PLANNING_TIMEOUT_S = 30.0
STIFFNESS = [300.0, 300.0, 300.0, 300.0, 120.0, 80.0, 30.0]
DAMPING = [25.0, 25.0, 25.0, 25.0, 10.0, 8.0, 5.0]
UNCHECKED_RISK_BANNER = """\
[HIGH RISK] box0909-compatible unchecked dataset replay selected.
- Project-side trajectory derivative, robot-state, TCP, network, and collision/scene checks are disabled.
- The current joints are read once for an unaudited slow move to the generated trajectory start.
- Replay chunks are planned before the FR3 connection and are not replanned from measured positions.
- Fixed robot-base/no-auxiliary-marker provenance and deterministic IK generation remain enforced.
- Dataset gripper mode commands the Corenetic gripper from recorded widths. Clear people, payloads,
  cables, and obstacles; keep the physical E-stop ready.
"""


@dataclass(frozen=True)
class PoseTarget:
    frame_index: int
    position_xyz: np.ndarray
    quaternion_xyzw: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def _load_episode_targets(csv_path: Path, episode: int) -> tuple[list[PoseTarget], dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"generated trajectory sidecar not found: {csv_path}")
    targets: list[PoseTarget] = []
    skipped_invalid = 0
    pose_sources: set[str] = set()
    smoothing_modes: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "episode_index",
            "frame_index",
            "state_x_m",
            "state_y_m",
            "state_z_m",
            "state_qx",
            "state_qy",
            "state_qz",
            "state_qw",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"trajectory sidecar is missing columns: {missing}")
        for row in reader:
            try:
                row_episode = int(float(row.get("episode_index", "")))
            except (TypeError, ValueError):
                skipped_invalid += 1
                continue
            if row_episode != episode:
                continue
            try:
                frame_index = int(float(row["frame_index"]))
                position = np.asarray(
                    [_finite_float(row, f"state_{axis}_m") for axis in "xyz"],
                    dtype=np.float64,
                )
                quaternion = np.asarray(
                    [_finite_float(row, f"state_q{axis}") for axis in "xyzw"],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(quaternion))
                if norm < 1e-12:
                    raise ValueError("zero quaternion")
                quaternion /= norm
            except (KeyError, TypeError, ValueError):
                skipped_invalid += 1
                continue
            targets.append(PoseTarget(frame_index, position, quaternion))
            if row.get("pose_source"):
                pose_sources.add(str(row["pose_source"]))
            if row.get("smoothing"):
                smoothing_modes.add(str(row["smoothing"]))
    targets.sort(key=lambda target: target.frame_index)
    if not targets:
        raise ValueError(f"episode {episode} has no finite state pose in {csv_path}")
    if len({target.frame_index for target in targets}) != len(targets):
        raise ValueError(f"episode {episode} contains duplicate frame_index values")
    return targets, {
        "frames": len(targets),
        "first_frame": targets[0].frame_index,
        "last_frame": targets[-1].frame_index,
        "skipped_invalid_rows": skipped_invalid,
        "pose_sources": sorted(pose_sources),
        "smoothing_modes": sorted(smoothing_modes),
    }


def _load_episode_gripper_widths(
    dataset_root: Path,
    targets: list[PoseTarget],
    episode: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load recorded Corenetic opening widths from the original dataset Parquet."""
    import pyarrow.parquet as pq

    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_key = None
    gripper_index = None
    for candidate in ("observation.state_raw", "observation.state"):
        names = (features.get(candidate) or {}).get("names")
        if isinstance(names, list) and "box_gripper.distance_m" in names:
            state_key = candidate
            gripper_index = names.index("box_gripper.distance_m")
            break
    if state_key is None or gripper_index is None:
        raise RuntimeError("dataset has no box_gripper.distance_m feature")

    widths_by_frame: dict[int, float] = {}
    parquet_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"dataset has no Parquet files under {dataset_root / 'data'}")
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=[state_key, "episode_index", "frame_index"],
        )
        for row in table.to_pylist():
            if int(row["episode_index"]) != episode:
                continue
            frame_index = int(row["frame_index"])
            if frame_index in widths_by_frame:
                raise RuntimeError(f"duplicate gripper frame {frame_index} in episode {episode}")
            state = row[state_key]
            if state is None or gripper_index >= len(state):
                raise RuntimeError(f"missing gripper state at episode {episode} frame {frame_index}")
            widths_by_frame[frame_index] = float(state[gripper_index])

    missing = [target.frame_index for target in targets if target.frame_index not in widths_by_frame]
    if missing:
        raise RuntimeError(
            f"dataset gripper is missing {len(missing)} replay frames; first missing={missing[0]}"
        )
    widths = np.asarray(
        [widths_by_frame[target.frame_index] for target in targets],
        dtype=np.float64,
    )
    if not np.isfinite(widths).all():
        raise RuntimeError("dataset gripper contains non-finite widths")
    if np.any(widths < 0.0) or np.any(widths > CORENETIC_MAX_WIDTH_M):
        raise RuntimeError(
            f"dataset gripper widths must be within [0, {CORENETIC_MAX_WIDTH_M}] m"
        )
    digest = hashlib.sha256(np.ascontiguousarray(widths).tobytes()).hexdigest()
    return widths, {
        "source": state_key,
        "feature_name": "box_gripper.distance_m",
        "units": "m",
        "frames": len(widths),
        "minimum_width_m": float(widths.min()),
        "maximum_width_m": float(widths.max()),
        "start_width_m": float(widths[0]),
        "end_width_m": float(widths[-1]),
        "widths_sha256": digest,
        "parquet_files": [str(path) for path in parquet_paths],
    }


def _drop_consecutive_duplicate_knots(q: np.ndarray) -> tuple[np.ndarray, int]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"expected Nx7 joint knots, got {q.shape}")
    keep = np.ones(len(q), dtype=bool)
    if len(q) > 1:
        keep[1:] = np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-10
    filtered = q[keep]
    return filtered, int(len(q) - len(filtered))


def _chunk_indices(length: int, chunk_size: int = CHUNK_SIZE):
    start = 0
    while start < length - 1:
        end = min(length, start + chunk_size)
        yield start, end
        start = end - 1


def _build_unchecked_trajectories(
    core,
    q: np.ndarray,
    chunk_size: int,
    speed_factor: float,
):
    """Build fixed replay chunks without invoking any project-side audit module."""
    trajectories = []
    for start, end in _chunk_indices(len(q), chunk_size):
        trajectory = core.JointTrajectory(
            q[start:end].tolist(),
            speed_factor,
            MAX_DEVIATION_RAD,
            PLANNING_TIMEOUT_S,
        )
        trajectories.append((start, end, trajectory))
    return trajectories


def _run_unchecked_trajectory(
    panda,
    core,
    trajectory,
    *,
    gripper=None,
    gripper_widths_m: np.ndarray | None = None,
    joint_knots: np.ndarray | None = None,
) -> None:
    """Run only the native controller and its libfranka/firmware protections."""
    controller = core.NativeJointTrajectoryController(
        trajectory,
        STIFFNESS,
        DAMPING,
        0.001,
    )
    started = False
    gripper_index = 0
    if (gripper_widths_m is None) != (joint_knots is None):
        raise ValueError("gripper widths and joint knots must be provided together")
    if gripper_widths_m is not None:
        gripper_widths_m = np.asarray(gripper_widths_m, dtype=np.float64).reshape(-1)
        joint_knots = np.asarray(joint_knots, dtype=np.float64)
        if joint_knots.shape != (len(gripper_widths_m), 7):
            raise ValueError("gripper widths must align one-to-one with joint knots")
    try:
        panda.start_controller_guarded(controller)
        started = True
        if gripper is not None and gripper_widths_m is not None:
            # This is the first gripper command.  It intentionally happens only
            # after the arm controller starts, so both streams share replay t=0.
            gripper.set_position(float(gripper_widths_m[0] / CORENETIC_MAX_WIDTH_M))
        deadline = time.monotonic() + float(trajectory.get_duration()) + 30.0
        while panda.control_thread_active():
            if gripper is not None and gripper_widths_m is not None:
                measured_q = np.asarray(panda.get_state().q, dtype=np.float64)
                remaining = joint_knots[gripper_index:]
                nearest = gripper_index + int(np.argmin(np.linalg.norm(remaining - measured_q, axis=1)))
                gripper_index = max(gripper_index, nearest)
                gripper.set_position(float(gripper_widths_m[gripper_index] / CORENETIC_MAX_WIDTH_M))
            if time.monotonic() > deadline:
                raise RuntimeError("Native controller timeout")
            time.sleep(0.02)
        if gripper is not None and gripper_widths_m is not None:
            gripper.set_position(float(gripper_widths_m[-1] / CORENETIC_MAX_WIDTH_M))
        panda.stop_controller()
        panda.raise_error()
    finally:
        if started:
            panda.stop_controller()


def _flush_gripper_position(gripper, width_m: float, command_rate_hz: float) -> None:
    """Ensure the final rate-limited Corenetic target is transmitted before disconnect."""
    normalized_position = float(width_m / CORENETIC_MAX_WIDTH_M)
    gripper.set_position(normalized_position)
    time.sleep(1.0 / command_rate_hz)
    gripper.set_position(normalized_position)


def _audit_native_partitions(
    native,
    core,
    q: np.ndarray,
    *,
    chunk_size: int,
    speed_factor: float,
    width_m: float,
    out_dir: Path,
) -> list[tuple[int, int]]:
    """Audit native chunks, splitting only planner timeouts down to two points."""
    accepted: list[tuple[int, int]] = []
    attempt = 0

    def audit(start: int, end: int) -> None:
        nonlocal attempt
        folder = out_dir / f"nominal_attempt_{attempt:03d}_{start}_{end}"
        attempt += 1
        folder.mkdir()
        try:
            native.prepare(core, q[start:end], speed_factor, width_m, folder)
        except RuntimeError as exc:
            # The native planner has its own 30 s complexity timeout. Splitting
            # preserves the same ordered knot path and every accepted child is
            # independently derivative/geometry audited. Never split around a
            # joint-wall, motion-envelope, or collision rejection.
            if "Trajectory generation faild" not in str(exc) or end - start <= 2:
                raise
            midpoint = start + (end - start) // 2
            _save_json(
                folder / "planner_split.json",
                {
                    "reason": str(exc),
                    "range": [start, end],
                    "children": [[start, midpoint + 1], [midpoint, end]],
                },
            )
            audit(start, midpoint + 1)
            audit(midpoint, end)
            return
        accepted.append((start, end))

    for chunk_start, chunk_end in _chunk_indices(len(q), chunk_size):
        audit(chunk_start, chunk_end)
    return accepted


def _load_native_module():
    sys.path.insert(0, str(BUNDLE_ROOT))
    return importlib.import_module("native_arm")


def _make_corenetic_gripper(command_rate_hz: float):
    from lerobot.robots.franka_research3.backends import CoreneticGripperHardwareDriver

    return CoreneticGripperHardwareDriver(
        max_width_m=CORENETIC_MAX_WIDTH_M,
        command_rate_limit_hz=command_rate_hz,
        command_deadband_m=0.0005,
        release_mode_on_disconnect=True,
    )


def _pose_matrix(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    from lerobot.utils.rotation import Rotation

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[:3, 3] = position_xyz
    return matrix


def _rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    cosine = np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _safe_multistart_seeds(
    lower: np.ndarray,
    upper: np.ndarray,
    preferred_seed: np.ndarray,
    count: int = MULTISTART_SEED_COUNT,
) -> list[tuple[str, np.ndarray]]:
    """Return a deterministic, bounded low-discrepancy set of IK seeds."""
    lower = np.asarray(lower, dtype=np.float64).reshape(7)
    upper = np.asarray(upper, dtype=np.float64).reshape(7)
    preferred_seed = np.asarray(preferred_seed, dtype=np.float64).reshape(7)
    if count < 1 or np.any(lower >= upper):
        raise ValueError("invalid safe joint bounds or multistart count")
    seeds = [("preferred", np.clip(preferred_seed, lower, upper))]
    span = upper - lower
    for index in range(1, count + 1):
        point = np.asarray(
            [_van_der_corput(index, base) for base in HALTON_BASES],
            dtype=np.float64,
        )
        seeds.append((f"halton_{index:03d}", lower + point * span))
    return seeds


def _legacy_wall_margin(q: np.ndarray, safe_lower: np.ndarray, safe_upper: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    return float(np.minimum(q - safe_lower, safe_upper - q).min())


def _verify_fixed_base_provenance(dataset_root: Path) -> dict[str, object]:
    """Fail closed unless the sidecar was generated directly in the robot base."""
    summary_path = (
        dataset_root.parent.parent
        / "tracking_analysis"
        / f"{dataset_root.name}_{TRACKING_RUN_SUFFIX}"
        / "summary.json"
    )
    if not summary_path.is_file():
        raise RuntimeError(f"fixed-base tracking provenance not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = summary.get("alignment")
    calibration = summary.get("calibration_inputs")
    problems: list[str] = []
    if summary.get("robot_base_mode") != "fixed":
        problems.append("robot_base_mode is not fixed")
    if not isinstance(alignment, dict):
        problems.append("alignment provenance is missing")
    elif (
        alignment.get("enabled") is not False
        or alignment.get("applied") is not False
        or alignment.get("method") != "none"
    ):
        problems.append("auxiliary alignment is enabled or was applied")
    if not isinstance(calibration, dict):
        problems.append("calibration input provenance is missing")
    elif (
        calibration.get("auxiliary_marker_required") is not False
        or bool(calibration.get("auxiliary_marker_summary"))
        or bool(calibration.get("auxiliary_marker_run_name"))
    ):
        problems.append("auxiliary marker localization is configured")
    fixed_camera_summary = calibration.get("fixed_camera_summary") if isinstance(calibration, dict) else None
    if not fixed_camera_summary or not Path(str(fixed_camera_summary)).is_file():
        problems.append("fixed-camera extrinsics summary is missing")
    if problems:
        raise RuntimeError("dataset is not approved for direct robot-base replay: " + "; ".join(problems))
    return {
        "tracking_summary": str(summary_path),
        "tracking_summary_sha256": _sha256(summary_path),
        "robot_base_mode": summary["robot_base_mode"],
        "alignment": {
            "enabled": alignment["enabled"],
            "applied": alignment["applied"],
            "method": alignment["method"],
        },
        "intrinsics_summary": calibration.get("intrinsics_summary"),
        "fixed_camera_summary": fixed_camera_summary,
        "auxiliary_marker_required": calibration["auxiliary_marker_required"],
        "auxiliary_marker_summary": calibration.get("auxiliary_marker_summary", ""),
    }


def _solve_joint_knots(
    targets: list[PoseTarget],
    legacy_lower: np.ndarray,
    legacy_upper: np.ndarray,
    wall_band: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    from lerobot.robots.franka_research3 import FrankaResearch3Config
    from third_party.opencv_kalibr.verification.verify_fr3_cube_pose_ik import (
        _fk_pose,
        _joint_limits,
        _make_kinematics_driver,
        _solve_target,
    )

    robot = FrankaResearch3Config(
        robot_ip=ROBOT_IP,
        gripper_backend="corenetic",
        allow_mock_gripper=True,
        urdf_path=str(ARM_ROOT / "model.urdf"),
        target_frame_name="corenetic_gripper_ee",
        ik_solver="hirol_gaussian_newton",
        ik_tolerance=1e-6,
        ik_max_iterations=200,
        use_otg=False,
    )
    kinematics = _make_kinematics_driver(robot)
    urdf_lower, urdf_upper = _joint_limits(robot, kinematics)
    legacy_lower = np.asarray(legacy_lower, dtype=np.float64).reshape(7)
    legacy_upper = np.asarray(legacy_upper, dtype=np.float64).reshape(7)
    wall_band = np.asarray(wall_band, dtype=np.float64).reshape(7)
    if not (np.allclose(urdf_lower, legacy_lower) and np.allclose(urdf_upper, legacy_upper)):
        raise RuntimeError("URDF joint limits differ from the sealed native safety limits")
    safe_lower = legacy_lower + wall_band
    safe_upper = legacy_upper - wall_band
    preferred_seed = IK_INITIAL_JOINTS_RAD.copy()
    target_matrices = [
        _pose_matrix(target.position_xyz, target.quaternion_xyzw) for target in targets
    ]
    best: dict[str, object] | None = None
    first_pose_safe = 0
    complete_safe = 0
    failure_counts: dict[str, int] = {}
    preferred_failure_reason: str | None = None
    attempted_seed_count = 0

    def reject(reason: str) -> None:
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    for seed_name, initial_seed in _safe_multistart_seeds(
        safe_lower, safe_upper, preferred_seed
    ):
        attempted_seed_count += 1
        knots: list[np.ndarray] = []
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        joint_steps_abs: list[float] = []
        joint_steps_l2: list[float] = []
        seed = initial_seed
        failed = False
        seed_failure_reason: str | None = None
        for order, (target, target_matrix) in enumerate(zip(targets, target_matrices, strict=True)):
            try:
                solution, _status = _solve_target(kinematics, seed, target_matrix)
            except Exception:
                seed_failure_reason = "solver_exception"
                reject("solver_exception")
                failed = True
                break
            if solution is None:
                seed_failure_reason = "ik_failed"
                reject("ik_failed")
                failed = True
                break
            solution = np.asarray(solution, dtype=np.float64).reshape(7)
            if not np.isfinite(solution).all():
                seed_failure_reason = "nonfinite_solution"
                reject("nonfinite_solution")
                failed = True
                break
            if np.any(solution < legacy_lower - 1e-5) or np.any(solution > legacy_upper + 1e-5):
                seed_failure_reason = "physical_joint_limit"
                reject("physical_joint_limit")
                failed = True
                break
            if _legacy_wall_margin(solution, safe_lower, safe_upper) <= 0.0:
                seed_failure_reason = "legacy_wall"
                reject("legacy_wall")
                failed = True
                break
            actual = _fk_pose(kinematics, solution)
            if actual is None:
                seed_failure_reason = "fk_failed"
                reject("fk_failed")
                failed = True
                break
            position_error = float(np.linalg.norm(actual[:3, 3] - target_matrix[:3, 3]))
            orientation_error = _rotation_error_deg(actual, target_matrix)
            if position_error > POSITION_TOLERANCE_M or orientation_error > ORIENTATION_TOLERANCE_DEG:
                seed_failure_reason = "residual"
                reject("residual")
                failed = True
                break
            if order == 0:
                first_pose_safe += 1
            else:
                delta = solution - seed
                step_abs = float(np.max(np.abs(delta)))
                step_l2 = float(np.linalg.norm(delta))
                if step_abs > MAX_JOINT_STEP_ABS_RAD or step_l2 > MAX_JOINT_STEP_L2_RAD:
                    seed_failure_reason = "branch_jump"
                    reject("branch_jump")
                    failed = True
                    break
                joint_steps_abs.append(step_abs)
                joint_steps_l2.append(step_l2)
            knots.append(solution)
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            seed = solution
        if failed:
            if seed_name == "preferred":
                preferred_failure_reason = seed_failure_reason
            continue
        complete_safe += 1
        candidate_q = np.asarray(knots, dtype=np.float64)
        candidate_margin = _legacy_wall_margin(candidate_q, safe_lower, safe_upper)
        candidate = {
            "seed_name": seed_name,
            "q": candidate_q,
            "minimum_wall_margin_rad": candidate_margin,
            "position_errors": position_errors,
            "orientation_errors": orientation_errors,
            "joint_steps_abs": joint_steps_abs,
            "joint_steps_l2": joint_steps_l2,
        }
        # The operator-supplied current start pose owns the IK branch whenever
        # it yields a complete safe path.  Halton seeds remain a fail-closed
        # fallback only when that requested branch cannot solve the episode.
        if seed_name == "preferred":
            best = candidate
            break
        if best is None or candidate_margin > float(best["minimum_wall_margin_rad"]):
            best = candidate

    if best is None:
        raise RuntimeError(
            "no deterministic IK branch satisfies the complete legacy joint-wall envelope; "
            f"first_pose_safe={first_pose_safe}, failures={failure_counts}"
        )
    q, duplicates_removed = _drop_consecutive_duplicate_knots(np.asarray(best["q"]))
    if len(q) < 2:
        raise RuntimeError("native replay requires at least two distinct IK knots")
    return q, {
        "input_pose_count": len(targets),
        "joint_knot_count": len(q),
        "consecutive_duplicate_knots_removed": duplicates_removed,
        "seed_strategy": "requested_current_start_pose_then_halton_maximum_wall_margin_fallback",
        "requested_initial_pose_name": IK_INITIAL_POSE_NAME,
        "requested_initial_joint_rad": IK_INITIAL_JOINTS_RAD.tolist(),
        "requested_initial_pose_source": "panda_py.Panda(192.168.11.102).get_state().q",
        "requested_initial_seed_failure_reason": preferred_failure_reason,
        "selected_seed": (
            IK_INITIAL_POSE_NAME if best["seed_name"] == "preferred" else best["seed_name"]
        ),
        "seed_candidates": MULTISTART_SEED_COUNT + 1,
        "seed_candidates_attempted": attempted_seed_count,
        "first_pose_safe_candidates": first_pose_safe,
        "complete_safe_candidates": complete_safe,
        "candidate_failure_counts": failure_counts,
        "legacy_safe_lower_rad": safe_lower.tolist(),
        "legacy_safe_upper_rad": safe_upper.tolist(),
        "minimum_waypoint_wall_margin_rad": best["minimum_wall_margin_rad"],
        "start_joint_rad": q[0].tolist(),
        "end_joint_rad": q[-1].tolist(),
        "urdf_path": str(ARM_ROOT / "model.urdf"),
        "target_frame": "corenetic_gripper_ee",
        "ik_solver": "hirol_gaussian_newton",
        "max_position_error_m": max(best["position_errors"], default=0.0),
        "max_orientation_error_deg": max(best["orientation_errors"], default=0.0),
        "max_joint_step_abs_rad": max(best["joint_steps_abs"], default=0.0),
        "max_joint_step_l2_rad": max(best["joint_steps_l2"], default=0.0),
    }


def _save_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _network_probe_result(stdout: str, stderr: str, returncode: int) -> dict[str, object]:
    loss_match = re.search(r"([\d.]+)% packet loss", stdout)
    rtt_match = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", stdout)
    metrics: dict[str, float] = {}
    if loss_match:
        metrics["packet_loss_pct"] = float(loss_match[1])
    if rtt_match:
        metrics.update(
            rtt_min_ms=float(rtt_match[1]),
            rtt_avg_ms=float(rtt_match[2]),
            rtt_max_ms=float(rtt_match[3]),
            rtt_mdev_ms=float(rtt_match[4]),
        )
    passed = bool(
        returncode == 0
        and loss_match
        and rtt_match
        and metrics["packet_loss_pct"] == 0.0
        and metrics["rtt_max_ms"] <= 0.90
        and metrics["rtt_mdev_ms"] <= 0.10
    )
    return {
        "passed": passed,
        "metrics": metrics,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


def _network_gate(out_dir: Path) -> None:
    attempts: list[dict[str, object]] = []
    for attempt in range(1, NETWORK_GATE_ATTEMPTS + 1):
        probe = subprocess.run(
            [
                "ping",
                "-I",
                "192.168.11.100",
                "-i",
                "0.001",
                "-c",
                "10000",
                "-s",
                "1200",
                "-q",
                ROBOT_IP,
            ],
            text=True,
            capture_output=True,
            timeout=25,
            check=False,
        )
        result = _network_probe_result(probe.stdout, probe.stderr, probe.returncode)
        result["attempt"] = attempt
        attempts.append(result)
        if result["passed"]:
            _save_json(
                out_dir / "network.json",
                {
                    "passed": True,
                    "accepted_attempt": attempt,
                    "thresholds": {
                        "packet_loss_pct": 0.0,
                        "rtt_max_ms": 0.90,
                        "rtt_mdev_ms": 0.10,
                    },
                    "attempts": attempts,
                },
            )
            return
        print(
            f"FCI network gate attempt {attempt}/{NETWORK_GATE_ATTEMPTS} failed; "
            f"metrics={result['metrics']}. Retrying without relaxing thresholds.",
            flush=True,
        )
    _save_json(
        out_dir / "network.json",
        {
            "passed": False,
            "accepted_attempt": None,
            "thresholds": {
                "packet_loss_pct": 0.0,
                "rtt_max_ms": 0.90,
                "rtt_mdev_ms": 0.10,
            },
            "attempts": attempts,
        },
    )
    raise RuntimeError(
        f"original FCI network gate failed {NETWORK_GATE_ATTEMPTS} consecutive attempts; no motion"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        type=int,
        nargs="?",
        default=None,
        help="legacy positional episode_index; prefer --episode-index",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="episode_index to replay (default: 0)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"dataset containing the generated sidecar (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--gripper-width-mm", type=float, required=True)
    parser.add_argument(
        "--gripper-mode",
        choices=("off", "dataset"),
        default="off",
        help="off leaves the gripper untouched; dataset follows recorded box_gripper.distance_m",
    )
    parser.add_argument(
        "--gripper-command-rate-hz",
        type=float,
        default=CORENETIC_COMMAND_RATE_HZ,
        help=f"Corenetic command rate limit (default: {CORENETIC_COMMAND_RATE_HZ:g} Hz)",
    )
    parser.add_argument(
        "--replay-speed-factor",
        type=float,
        default=NATIVE_SPEED_FACTOR,
        help=f"native replay speed factor in (0, 1] (default: {NATIVE_SPEED_FACTOR:g})",
    )
    parser.add_argument(
        "--start-speed-factor",
        type=float,
        default=START_SPEED_FACTOR,
        help=f"move-to-trajectory-start speed factor in (0, 1] (default: {START_SPEED_FACTOR:g})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="initial native planner waypoints per chunk (2..200; default: 10; timeouts split automatically)",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--unchecked-execution",
        action="store_true",
        help="box0909-compatible execution without project-side motion, state, network, or geometry audits",
    )
    args = parser.parse_args(argv)
    if args.episode is not None and args.episode_index is not None and args.episode != args.episode_index:
        parser.error("positional episode and --episode-index disagree")
    args.episode_index = (
        args.episode_index if args.episode_index is not None else args.episode if args.episode is not None else 0
    )
    if args.episode_index < 0:
        parser.error("--episode-index must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not math.isfinite(args.gripper_width_mm) or not 0.0 <= args.gripper_width_mm <= WIDTH_MAX_MM:
        raise SystemExit(f"--gripper-width-mm must be within [0, {WIDTH_MAX_MM}]")
    for name in ("replay_speed_factor", "start_speed_factor"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be within (0, 1]")
    if (
        not math.isfinite(args.gripper_command_rate_hz)
        or not 0.0 < args.gripper_command_rate_hz <= 100.0
    ):
        raise SystemExit("--gripper-command-rate-hz must be within (0, 100]")
    if args.gripper_mode == "dataset" and not args.unchecked_execution:
        raise SystemExit("--gripper-mode dataset currently requires --unchecked-execution")
    if not 2 <= args.chunk_size <= 200:
        raise SystemExit("--chunk-size must be within [2, 200]")
    dataset_root = args.dataset_root.expanduser().resolve()
    csv_path = dataset_root / SIDECAR_RELATIVE_DIR / f"state_action.{args.side}.csv"

    for path in (REPO_ROOT, REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    native = _load_native_module()
    native.verify_inputs()
    from panda_py import _core as core, libfranka
    from arm_runtime import snapshot, state_checks
    from replay_timed_candidate import urdf_chain

    provenance = _verify_fixed_base_provenance(dataset_root)
    targets, sidecar_summary = _load_episode_targets(csv_path, args.episode_index)
    q, ik_summary = _solve_joint_knots(targets, native.LOWER, native.UPPER, native.WALL_BAND)
    gripper_widths_m = None
    gripper_summary: dict[str, object] = {"mode": args.gripper_mode}
    if args.gripper_mode == "dataset":
        gripper_widths_m, loaded_gripper_summary = _load_episode_gripper_widths(
            dataset_root,
            targets,
            args.episode_index,
        )
        if len(gripper_widths_m) != len(q):
            raise RuntimeError(
                "dataset gripper alignment requires one width per retained IK knot; "
                f"widths={len(gripper_widths_m)}, knots={len(q)}"
            )
        gripper_summary.update(loaded_gripper_summary)
        gripper_summary["command_rate_limit_hz"] = float(args.gripper_command_rate_hz)
    chain = urdf_chain(ARM_ROOT / "model.urdf")
    reference = json.loads((ARM_ROOT / "tool_reference.json").read_text(encoding="utf-8"))
    width_m = float(args.gripper_width_mm) / 1000.0

    out = BUNDLE_ROOT / "logs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_dataset_" + uuid.uuid4().hex[:8]
    )
    out.mkdir(parents=True)
    _save_json(
        out / "dataset_input.json",
        {
            "schema": "p0_native_generated_dataset_replay/v1",
            "dataset_root": str(dataset_root),
            "episode": int(args.episode_index),
            "side": args.side,
            "sidecar_path": str(csv_path),
            "sidecar_sha256": _sha256(csv_path),
            "sidecar": sidecar_summary,
            "provenance": provenance,
            "ik": ik_summary,
            "legacy_cli_gripper_width_mm": float(args.gripper_width_mm),
            "native_speed_factor": float(args.replay_speed_factor),
            "start_speed_factor": float(args.start_speed_factor),
            "chunk_size": int(args.chunk_size),
            "gripper": gripper_summary,
            "execution_mode": (
                "box0909_compatible_unchecked" if args.unchecked_execution else "audited"
            ),
        },
    )
    print(f"日志目录：{out}", flush=True)
    print(
        f"Loaded generated {args.side} trajectory: dataset={dataset_root}, episode={args.episode_index}, "
        f"poses={len(targets)}, native_knots={len(q)}, sidecar_sha256={_sha256(csv_path)}",
        flush=True,
    )

    if args.execute and os.geteuid() != 0:
        raise SystemExit("hardware execution requires sudo")

    with ExitStack() as stack:
        for root in ([BUNDLE_ROOT, ARM_ROOT, SOURCE_ROOT] if args.execute else []):
            lock = stack.enter_context((root / "session.lock").open("a", encoding="utf-8"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        if not args.execute:
            _audit_native_partitions(
                native,
                core,
                q,
                chunk_size=args.chunk_size,
                speed_factor=args.replay_speed_factor,
                width_m=width_m,
                out_dir=out,
            )
            print("Offline dataset IK/native/geometry checks finished. Robot instances=0; gripper instances=0.")
            return 0

        if args.unchecked_execution:
            replay_trajectories = _build_unchecked_trajectories(
                core,
                q,
                args.chunk_size,
                args.replay_speed_factor,
            )
            print(UNCHECKED_RISK_BANNER, flush=True)
            print(
                f"Prepared {len(replay_trajectories)} fixed native replay chunks before FR3 connection. "
                f"Replay speed factor={args.replay_speed_factor:g}; "
                f"start speed factor={args.start_speed_factor:g}; "
                f"gripper mode={args.gripper_mode}. No project-side execution audit was run.",
                flush=True,
            )
            prompt = (
                f"Type YES to execute unchecked dataset={dataset_root.name} "
                f"episode={args.episode_index} side={args.side}: "
            )
            if not sys.stdin.isatty() or input(prompt).strip() != "YES":
                print("Cancelled: no FR3 connection and no motion.", flush=True)
                return 2

            panda = None
            gripper = None

            def unchecked_interrupt(signum, _frame):
                raise KeyboardInterrupt(f"Signal {signum}")

            signal.signal(signal.SIGTERM, unchecked_interrupt)
            try:
                panda = core.Panda(ROBOT_IP, "native_dataset_unchecked", libfranka.RealtimeConfig.kIgnore)
                current_q = np.asarray(panda.get_state().q, dtype=np.float64)
                start_delta_rad = float(np.max(np.abs(current_q - q[0])))
                start_trajectory = core.JointTrajectory(
                    [current_q.tolist(), q[0].tolist()],
                    args.start_speed_factor,
                    0.0,
                    PLANNING_TIMEOUT_S,
                )
                print(
                    f"[HIGH RISK] Starting unaudited move to trajectory start; "
                    f"max joint delta={start_delta_rad:.6f} rad.",
                    flush=True,
                )
                _run_unchecked_trajectory(panda, core, start_trajectory)
                if args.gripper_mode == "dataset":
                    gripper = _make_corenetic_gripper(args.gripper_command_rate_hz)
                    gripper.connect()
                    held_width_m = float(gripper.get_position() * CORENETIC_MAX_WIDTH_M)
                    _save_json(
                        out / "gripper_initialization.json",
                        {
                            "schema": "p0_corenetic_gripper_initialization/v1",
                            "policy": "read_in_collection_mode_then_enter_control_holding_measured_width",
                            "measured_hold_width_m": held_width_m,
                            "first_dataset_width_m": float(gripper_widths_m[0]),
                            "first_dataset_minus_hold_m": float(gripper_widths_m[0] - held_width_m),
                            "dataset_command_sent": False,
                        },
                    )
                    print(
                        "Arm reached the first EE pose. Corenetic control was initialized by holding "
                        f"the measured physical width ({held_width_m * 1000.0:.2f} mm); no dataset "
                        "trajectory command has been sent. Arm and dataset gripper replay will start "
                        "together at replay t=0. "
                        f"First dataset width={gripper_widths_m[0] * 1000.0:.2f} mm.",
                        flush=True,
                    )
                for index, (start, end, trajectory) in enumerate(replay_trajectories):
                    print(
                        f"[HIGH RISK] Starting fixed replay_{index} knots=[{start}, {end}); "
                        "no project-side audit or measured-position replanning.",
                        flush=True,
                    )
                    _run_unchecked_trajectory(
                        panda,
                        core,
                        trajectory,
                        gripper=gripper,
                        gripper_widths_m=(
                            gripper_widths_m[start:end] if gripper_widths_m is not None else None
                        ),
                        joint_knots=(q[start:end] if gripper_widths_m is not None else None),
                    )
                if gripper is not None and gripper_widths_m is not None:
                    _flush_gripper_position(
                        gripper,
                        float(gripper_widths_m[-1]),
                        float(args.gripper_command_rate_hz),
                    )
            finally:
                if panda is not None:
                    panda.stop_controller()
                if gripper is not None:
                    gripper.disconnect()
            print(
                f"Completed unchecked generated-dataset replay; gripper mode={args.gripper_mode}.",
                flush=True,
            )
            return 0

        native_partitions = list(_chunk_indices(len(q), args.chunk_size))
        print(
            "Fixed robot-base provenance verified and auxiliary marker alignment absent. "
            "Redundant nominal pre-audit skipped; every executed segment keeps its native motion-envelope "
            "and full-mesh geometry audit.",
            flush=True,
        )
        prompt = (
            f"将执行 dataset={dataset_root.name} episode={args.episode_index} side={args.side}；"
            "夹爪不受控且保持所填开口、范围清空、急停可用。输入 YES："
        )
        if not sys.stdin.isatty() or input(prompt).strip() != "YES":
            print("Cancelled: no motion", flush=True)
            return 2

        _network_gate(out)
        panda = core.Panda(ROBOT_IP, "native_dataset_arm_only", libfranka.RealtimeConfig.kIgnore)

        def interrupt(signum, _frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        stages = [("start", q[:1], args.start_speed_factor)] + [
            (f"replay_{index}", q[start + 1 : end], args.replay_speed_factor)
            for index, (start, end) in enumerate(native_partitions)
        ]
        try:
            for name, targets_q, speed in stages:
                state = snapshot(panda.get_robot())
                state_checks(state, chain, reference)
                if name == "start" and np.max(np.abs(np.asarray(state["q"]) - q[0])) <= 0.03:
                    print("Already within original start tolerance; no start movement.", flush=True)
                    continue
                folder = out / name
                folder.mkdir()
                path = np.vstack([state["q"], targets_q])
                trajectory = native.prepare(core, path, speed, width_m, folder)
                state_checks(snapshot(panda.get_robot()), chain, reference, state["q"])
                _save_json(folder / "before.json", state)
                native.execute(panda, core, trajectory, targets_q[-1], chain, folder)
                state_checks(snapshot(panda.get_robot()), chain, reference)
        finally:
            panda.stop_controller()
        print("Completed generated-dataset native arm-only replay. No gripper commands.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
