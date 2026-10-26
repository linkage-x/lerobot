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
Move the FR3 arm to its SDK-defined start pose from inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import subprocess
import sys



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the FR3 arm to its start pose.")
    # No default IP: with two arms the wrong one is the one that moves (roadmap B2). Callers resolve it
    # from tools/fr3/fr3_arms.json (fr3_move_to_start.py --arm, the gateway).
    parser.add_argument("--robot-ip", required=True, help="FR3 controller IP address.")
    parser.add_argument(
        "--start-joints",
        default=None,
        help="7 comma-separated joint angles (rad), the arm's start in fr3_arms.json; default panda_py's start",
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
        if args.start_joints:
            start = [float(v) for v in args.start_joints.split(",")]
            if len(start) != 7:
                raise ValueError(f"--start-joints needs 7 angles, got {len(start)}")
            print(f"fr3_move_to_start=TARGET q={start}")
            robot.move_to_joint_position(start)
        else:
            robot.move_to_start()
        state = robot.get_state()
        q = getattr(state, "q", None)
        if q is None:
            raise RuntimeError("FR3 move_to_start completed but state.q is unavailable.")
        print("fr3_move_to_start=PASS")
        print("Current joint angles (rad):", list(q))
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
