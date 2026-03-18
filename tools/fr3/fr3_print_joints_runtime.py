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
Print the FR3 arm's current joint angles from inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import math


DEFAULT_ROBOT_IP = "192.168.1.208"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print FR3 current joint angles.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from panda_py import Panda

    robot = None
    try:
        print(f"fr3_print_joints=CONNECT robot_ip={args.robot_ip}")
        robot = Panda(args.robot_ip)
        state = robot.get_state()
        q = list(getattr(state, "q", []))
        if not q:
            raise RuntimeError("state.q is unavailable.")
        q_deg = [math.degrees(v) for v in q]
        print("fr3_print_joints=PASS")
        print(f"Joint angles (rad): {[round(v, 6) for v in q]}")
        print(f"Joint angles (deg): {[round(v, 3) for v in q_deg]}")
        return 0
    finally:
        if robot is not None and hasattr(robot, "stop_controller"):
            try:
                robot.stop_controller()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
