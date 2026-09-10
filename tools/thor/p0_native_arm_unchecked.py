#!/usr/bin/env python3
"""Execute the frozen P0 joint knots without project-side safety gates.

This deliberately does not import or run the trajectory-audit, robot-state,
joint-envelope, TCP, or collision/scene modules from the frozen handoff. It
constructs every replay trajectory before opening the FR3 connection. After
connecting, it reads the current joints once and plans one slow ``start`` move
to the stored first pose. Measured positions are not inserted into subsequent
``replay_*`` plans.

The FR3/libfranka firmware and controller-level protections still apply; this
script cannot and does not disable protections implemented below this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


BUNDLE_ROOT = Path("/home/nvidia/box_api/replay_p0_native_arm_only_20260908")
SOURCE_PATH = Path("/home/nvidia/box_api/replay_p0_once_20260908/timed_plan.json")
P1_ACTIVE_PATH = Path("/home/nvidia/lerobot/outputs/calibration/p1_simple_eye_hand_calibration/active.json")
PYTHON_SITE = Path(
    "/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/"
    "cmeel.prefix/lib/python3.12/site-packages"
)
ROBOT_IP = "192.168.11.102"
ROBOT_NAME = "p0_native_unchecked"
CHUNK_SIZE = 200
SPEED_FACTOR = 0.1
START_SPEED_FACTOR = 0.05
MAX_DEVIATION_RAD = 0.02
PLANNING_TIMEOUT_S = 30.0
STIFFNESS = [300.0, 300.0, 300.0, 300.0, 120.0, 80.0, 30.0]
DAMPING = [25.0, 25.0, 25.0, 25.0, 10.0, 8.0, 5.0]
RISK_BANNER = """\
[HIGH RISK] Unchecked P0 FR3 replay selected.
- Trajectory audit, robot-state safety checks, and collision/scene checks are disabled.
- panda_py reads the current joints once and plans a slow move to the stored trajectory start pose.
- The start path is not audited or collision checked. No measured-position replanning occurs between replay chunks.
- The gripper is not commanded. Clear people, payloads, cables, table hazards, and keep the physical E-stop ready.
- GUI Abort is only a software stop and is not a substitute for the physical E-stop.
"""


def _episode_knots(source: dict[str, object], episode: int) -> np.ndarray:
    record = next(
        (item for item in source.get("episodes", []) if int(item.get("episode", -1)) == episode),
        None,
    )
    if record is None:
        raise ValueError(f"Episode {episode} is absent from the frozen P0 source")
    if "joint_knots" in record:
        knots = np.asarray(record["joint_knots"], dtype=float)
        expected_shape = (int(record["frames"]), 7)
        if knots.shape != expected_shape or not np.isfinite(knots).all():
            raise ValueError("Relocalized P0 source has invalid joint knot data")
        return knots
    coefficients = np.asarray(record["coeff_descending_unit_interval"], dtype=float)
    knots = np.concatenate((coefficients[:, -1, :], coefficients[-1].sum(axis=0)[None]), axis=0)
    expected_shape = (int(record["frames"]), 8)
    if knots.shape != expected_shape or not np.isfinite(knots).all():
        raise ValueError("Frozen P0 source has invalid knot data")
    return knots[:, :7]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source() -> tuple[Path, dict[str, object], str]:
    if not P1_ACTIVE_PATH.exists():
        return SOURCE_PATH, json.loads(SOURCE_PATH.read_text(encoding="utf-8")), "frozen original (no active P1 calibration)"
    active = json.loads(P1_ACTIVE_PATH.read_text(encoding="utf-8"))
    if active.get("schema") != "p1_simple_eye_hand_active/v1":
        raise ValueError(f"Unsupported P1 active pointer schema: {P1_ACTIVE_PATH}")
    plan_path = Path(str(active.get("plan_path", "")))
    calibration_path = Path(str(active.get("calibration_path", "")))
    if not plan_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Active P1 plan or calibration is missing; refusing to use the frozen fallback silently")
    if _sha256(plan_path) != str(active.get("plan_sha256", "")):
        raise ValueError("Active P1 plan hash mismatch")
    if _sha256(calibration_path) != str(active.get("calibration_sha256", "")):
        raise ValueError("Active P1 calibration hash mismatch")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise ValueError("Active P1 calibration no longer has passing status")
    source = json.loads(plan_path.read_text(encoding="utf-8"))
    if source.get("schema") != "p0_relocalized_joint_plan/v1":
        raise ValueError("Active P1 plan is not a relocalized P0 joint plan")
    label = f"P1-relocated base ({calibration.get('world_frame_id', 'unknown world')}; {calibration_path})"
    return plan_path, source, label


def _chunk_waypoints(q: np.ndarray):
    start = 0
    while start < len(q) - 1:
        end = min(len(q), start + CHUNK_SIZE)
        yield q[start:end]
        start = end - 1


def _run_trajectory(panda, core, trajectory) -> None:
    controller = core.NativeJointTrajectoryController(
        trajectory,
        STIFFNESS,
        DAMPING,
        0.001,
    )
    started = False
    try:
        panda.start_controller_guarded(controller)
        started = True
        deadline = time.monotonic() + float(trajectory.get_duration()) + 30.0
        while panda.control_thread_active():
            if time.monotonic() > deadline:
                raise RuntimeError("Native controller timeout")
            time.sleep(0.02)
        panda.stop_controller()
        panda.raise_error()
    finally:
        if started:
            panda.stop_controller()


def _start_trajectory(core, current_q: np.ndarray, target_q: np.ndarray):
    return core.JointTrajectory(
        [current_q.tolist(), target_q.tolist()],
        START_SPEED_FACTOR,
        0.0,
        PLANNING_TIMEOUT_S,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=int, choices=(0, 1))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gripper-width-mm", type=float, required=True)
    args = parser.parse_args()

    if not np.isfinite(args.gripper_width_mm) or not 0.0 <= args.gripper_width_mm <= 89.05:
        parser.error("gripper width must be between 0 and 89.05 mm")

    print(RISK_BANNER, flush=True)
    sys.path[:0] = [str(BUNDLE_ROOT / "python"), str(PYTHON_SITE)]
    from panda_py import _core as core, libfranka

    source_path, source, source_label = _resolve_source()
    q = _episode_knots(source, args.episode)
    trajectories = [
        core.JointTrajectory(
            waypoints.tolist(),
            SPEED_FACTOR,
            MAX_DEVIATION_RAD,
            PLANNING_TIMEOUT_S,
        )
        for waypoints in _chunk_waypoints(q)
    ]
    print(
        f"Loaded Episode {args.episode}: {len(q)} stored frames, "
        f"{len(trajectories)} fixed replay chunks, declared passive gripper width "
        f"{args.gripper_width_mm:g} mm. Source={source_label}; path={source_path}; sha256={_sha256(source_path)}.",
        flush=True,
    )
    print(
        "No trajectory audit, state safety gate, or collision check was run. "
        "Hardware execution will add one slow measured-position start move; "
        "replay chunks remain fixed.",
        flush=True,
    )
    if not args.execute:
        print("Inspection finished without creating an FR3 instance or sending motion commands.", flush=True)
        return 0

    if os.geteuid() != 0:
        parser.error("hardware execution requires sudo")
    if not sys.stdin.isatty():
        parser.error("hardware execution confirmation requires a TTY")
    if input("Type YES to accept the listed risks and execute unchecked FR3 motion: ").strip() != "YES":
        print("Cancelled: no FR3 connection and no motion.", flush=True)
        return 2

    panda = core.Panda(ROBOT_IP, ROBOT_NAME, libfranka.RealtimeConfig.kIgnore)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        current_q = np.asarray(panda.get_state().q, dtype=float)
        start_delta_rad = float(np.max(np.abs(current_q - q[0])))
        start_trajectory = _start_trajectory(core, current_q, q[0])
        print(
            f"[HIGH RISK] Starting unaudited panda_py move to trajectory start at "
            f"speed factor {START_SPEED_FACTOR:g}; max joint delta={start_delta_rad:.6f} rad; "
            "no collision/state safety gate.",
            flush=True,
        )
        _run_trajectory(panda, core, start_trajectory)
        print("Reached the end of the slow trajectory-start controller stage.", flush=True)
        for index, trajectory in enumerate(trajectories):
            print(
                f"[HIGH RISK] Starting fixed replay_{index}; no measured-position "
                "replanning or project-side safety/collision audit.",
                flush=True,
            )
            _run_trajectory(panda, core, trajectory)
    finally:
        panda.stop_controller()

    print("Completed unchecked arm-only P0 replay. No gripper commands were sent.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
