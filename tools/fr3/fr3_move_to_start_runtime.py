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

"""
Move the FR3 arm to the workstation XML ``home`` keyframe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_ROBOT_IP = "192.168.1.206"
# First seven qpos values from
# src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.xml:<key name="home">.
# This is the workstation recording start contract; it is deliberately not Panda.move_to_start().
FR3_PIKA_HOME_JOINTS_RAD = (0.0, -0.785, 0.0, -2.355, 0.0, 1.57079, 0.785)
DEFAULT_TOLERANCE_RAD = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the FR3 arm to the workstation XML home keyframe.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument(
        "--tolerance-rad",
        type=float,
        default=DEFAULT_TOLERANCE_RAD,
        help="Maximum allowed read-back joint error after the move.",
    )
    return parser.parse_args(argv)


def check_ping(robot_ip: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["ping", "-c", "1", "-W", "1", robot_ip],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "SKIP", "ping binary unavailable"

    details = completed.stdout.strip() or completed.stderr.strip() or f"returncode={completed.returncode}"
    if completed.returncode == 0:
        return "PASS", details
    return "FAIL", details


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from panda_py import Panda

    robot = None
    try:
        ping_status, ping_details = check_ping(args.robot_ip)
        print(f"fr3_move_to_start=PING status={ping_status} details={ping_details}")
        print(f"fr3_move_to_start=CONNECT robot_ip={args.robot_ip}")
        try:
            robot = Panda(args.robot_ip)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to FR3 at {args.robot_ip}. "
                "libfranka UDP timeouts usually mean the controller is unreachable from the current machine, "
                "the host NIC is not on the robot subnet, the robot is not ready for FCI, "
                "or another libfranka client already holds the session. "
                f"Ping probe: {ping_status}. "
                "Next steps: run "
                f"`python tools/fr3/fr3_hardware_smoke.py --robot-ip={args.robot_ip} --skip-spacemouse-list --skip-spacemouse-open` "
                "or retry with "
                f"`python tools/fr3/fr3_move_to_start.py --runtime host --robot-ip={args.robot_ip}`."
            ) from exc
        target = list(FR3_PIKA_HOME_JOINTS_RAD)
        print(f"fr3_move_to_start=TARGET source=fr3_pika_gripper.xml:keyframe/home q={target}")
        robot.move_to_joint_position(target)
        state = robot.get_state()
        q = getattr(state, "q", None)
        if q is None:
            raise RuntimeError("FR3 move_to_start completed but state.q is unavailable.")
        current = [float(value) for value in q]
        max_error = max(abs(actual - desired) for actual, desired in zip(current, target, strict=True))
        if max_error > float(args.tolerance_rad):
            raise RuntimeError(
                f"FR3 reached a different start pose: max_joint_error={max_error:.4f} rad "
                f"> tolerance={args.tolerance_rad:.4f} rad."
            )
        print("fr3_move_to_start=PASS")
        print("Current joint angles (rad):", current)
        print(f"Max joint error (rad): {max_error:.6f}")
        return 0
    except Exception as exc:
        print(f"fr3_move_to_start=FAIL details={exc}", file=sys.stderr)
        raise
    finally:
        if robot is not None and hasattr(robot, "stop_controller"):
            try:
                robot.stop_controller()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
