#!/usr/bin/env python3
"""Slowly preposition FR3 for P1 using the FR3-capable native controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


BUNDLE_ROOT = Path("/home/nvidia/box_api/replay_p0_native_arm_only_20260908")
PYTHON_SITE = Path(
    "/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/"
    "cmeel.prefix/lib/python3.12/site-packages"
)
ROBOT_IP = "192.168.11.102"
ROBOT_NAME = "p1_native_preposition"
SPEED_FACTOR = 0.05
PLANNING_TIMEOUT_S = 30.0
STIFFNESS = [300.0, 300.0, 300.0, 300.0, 120.0, 80.0, 30.0]
DAMPING = [25.0, 25.0, 25.0, 25.0, 10.0, 8.0, 5.0]
FR3_LOWER = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_UPPER = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def load_first_target(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not records:
        raise ValueError(f"No records in P1 selected pose document: {path}")
    raw = records[0].get("joint_values_rad", records[0].get("joint_values"))
    target = np.asarray(raw, dtype=float)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("First P1 pose must contain seven finite joint values")
    if np.any(target < FR3_LOWER) or np.any(target > FR3_UPPER):
        raise ValueError(f"First P1 pose exceeds FR3 joint limits: {target.tolist()}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_document", type=Path)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.confirmation != "P1_MOVE_FR3":
        parser.error("exact confirmation P1_MOVE_FR3 is required")
    target = load_first_target(args.pose_document.resolve())

    sys.path[:0] = [str(BUNDLE_ROOT / "python"), str(PYTHON_SITE)]
    from panda_py import _core as core, libfranka

    panda = core.Panda(ROBOT_IP, ROBOT_NAME, libfranka.RealtimeConfig.kIgnore)
    started = False
    try:
        current = np.asarray(panda.get_state().q, dtype=float)
        trajectory = core.JointTrajectory(
            [current.tolist(), target.tolist()], SPEED_FACTOR, 0.0, PLANNING_TIMEOUT_S
        )
        print(
            "[HIGH RISK] P1 native preposition: "
            f"speed factor={SPEED_FACTOR:g}; max joint delta={np.max(np.abs(target-current)):.6f} rad; "
            "no project-side collision/scene audit; physical E-stop required.",
            flush=True,
        )
        controller = core.NativeJointTrajectoryController(trajectory, STIFFNESS, DAMPING, 0.001)
        panda.start_controller_guarded(controller)
        started = True
        deadline = time.monotonic() + float(trajectory.get_duration()) + 30.0
        while panda.control_thread_active():
            if time.monotonic() > deadline:
                raise RuntimeError("P1 native preposition controller timeout")
            time.sleep(0.02)
        panda.stop_controller()
        started = False
        panda.raise_error()
        reached = np.asarray(panda.get_state().q, dtype=float)
        error = float(np.max(np.abs(reached - target)))
        if error > 0.02:
            raise RuntimeError(f"P1 native preposition did not reach target: max error={error:.6f} rad")
        print(f"P1 native preposition complete; max joint error={error:.6f} rad.", flush=True)
    finally:
        if started:
            panda.stop_controller()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
