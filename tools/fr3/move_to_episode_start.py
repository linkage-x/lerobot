#!/usr/bin/env python3
"""Move one FR3 arm to the first joint frame of a solved episode."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


JOINT_COLUMNS = [f"fr3_joint{i}_rad" for i in range(1, 8)]
EXPECTED_EE_MASS_KG = 0.7495964172315478
EXPECTED_EE_COM_M = np.asarray(
    [0.009645203274806327, -0.007980637829292982, 0.06658405690880417],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--robot-ip", default="192.168.11.102")
    parser.add_argument("--speed-factor", type=float, default=0.03)
    parser.add_argument("--success-tolerance-rad", type=float, default=0.03)
    return parser.parse_args()


def load_first_frame(path: Path, episode: int) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["episode_index"]) == episode:
                return np.asarray([float(row[name]) for name in JOINT_COLUMNS], dtype=np.float64)
    raise RuntimeError(f"Episode {episode} not found in {path}")


def main() -> int:
    if os.environ.get("FRANKA_PHYSICAL_WATCHER_CONFIRMED") != "YES":
        raise RuntimeError("A person with immediate E-stop access must be beside the robot.")

    args = parse_args()
    target = load_first_frame(Path(args.csv), args.episode)

    import panda_py
    from panda_py import libfranka

    panda = panda_py.Panda(args.robot_ip, realtime_config=libfranka.RealtimeConfig.kIgnore)
    before_state = panda.get_state()
    if str(before_state.robot_mode) != "RobotMode.kIdle" or list(before_state.current_errors):
        raise RuntimeError(
            f"FR3 is not ready: mode={before_state.robot_mode}, "
            f"errors={list(before_state.current_errors)}"
        )

    mass_error = abs(float(before_state.m_ee) - EXPECTED_EE_MASS_KG)
    com_error = float(np.linalg.norm(np.asarray(before_state.F_x_Cee) - EXPECTED_EE_COM_M))
    if mass_error > 0.05 or com_error > 0.02:
        raise RuntimeError(
            f"End-effector configuration mismatch: mass_error={mass_error}, com_error={com_error}"
        )

    before = np.asarray(before_state.q, dtype=np.float64)
    print(f"[INFO] current_joints_rad={before.tolist()}", flush=True)
    print(f"[INFO] target_joints_rad={target.tolist()}", flush=True)
    print(f"[INFO] initial_gap_rad={float(np.max(np.abs(target - before))):.6f}", flush=True)

    success = panda.move_to_joint_position(
        target,
        speed_factor=float(args.speed_factor),
        stiffness=np.asarray([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]),
        damping=np.asarray([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]),
        dq_threshold=0.001,
        success_threshold=float(args.success_tolerance_rad),
    )

    after_state = panda.get_state()
    after = np.asarray(after_state.q, dtype=np.float64)
    final_error = float(np.max(np.abs(target - after)))
    print(f"[INFO] move_result={bool(success)}", flush=True)
    print(f"[INFO] final_joints_rad={after.tolist()}", flush=True)
    print(f"[INFO] final_error_rad={final_error:.6f}", flush=True)
    print(f"[INFO] robot_mode={after_state.robot_mode}", flush=True)
    print(f"[INFO] current_errors={list(after_state.current_errors)}", flush=True)
    print(f"[INFO] last_motion_errors={list(after_state.last_motion_errors)}", flush=True)
    if not success or final_error > float(args.success_tolerance_rad):
        raise RuntimeError("FR3 did not reach the episode start within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
