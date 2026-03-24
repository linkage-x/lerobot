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
Open the DAS gripper, hold for a fixed duration, then close it inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import time


DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_GEN_CON_SDK_PATH = "/opt/dependencies/gen_con_sdk_python_release"
DEFAULT_FULLY_OPEN_SUCCESS_THRESHOLD = 0.90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the DAS gripper, wait, then close it.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="DAS controller serial port.")
    parser.add_argument(
        "--gen-con-sdk-path",
        default=DEFAULT_GEN_CON_SDK_PATH,
        help="Path to the Gen Controller SDK root.",
    )
    parser.add_argument(
        "--open-position",
        type=float,
        default=1.0,
        help="Normalized open target. DAS uses 1.0 as fully open.",
    )
    parser.add_argument(
        "--close-position",
        type=float,
        default=0.0,
        help="Normalized close target. DAS uses 0.0 as fully closed.",
    )
    parser.add_argument(
        "--hold-open-s",
        type=float,
        default=10.0,
        help="How long to keep the gripper open before closing.",
    )
    parser.add_argument(
        "--move-timeout-s",
        type=float,
        default=3.0,
        help="Maximum time to wait for the gripper to reach the open/close target.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.02,
        help="Acceptable normalized error when checking the final position.",
    )
    parser.add_argument(
        "--fully-open-success-threshold",
        type=float,
        default=DEFAULT_FULLY_OPEN_SUCCESS_THRESHOLD,
        help="Treat a fully-open command as successful once measured position reaches this threshold.",
    )
    parser.add_argument("--baudrate", type=int, default=921600, help="Controller baudrate.")
    parser.add_argument(
        "--update-frequency-hz",
        type=float,
        default=50.0,
        help="Encoder update frequency requested from the controller.",
    )
    return parser.parse_args(argv)


def _clip_position(position: float) -> float:
    return max(0.0, min(float(position), 1.0))


def _target_reached(
    measured_position: float,
    target_position: float,
    tolerance: float,
    fully_open_success_threshold: float,
) -> bool:
    if abs(measured_position - target_position) <= tolerance:
        return True
    if target_position >= 1.0 - tolerance and measured_position >= fully_open_success_threshold:
        return True
    return False


def _wait_for_target(
    driver,
    target_position: float,
    timeout_s: float,
    tolerance: float,
    fully_open_success_threshold: float,
) -> tuple[bool, float]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_position = float(driver.get_position())
    while time.perf_counter() < deadline:
        last_position = float(driver.get_position())
        if _target_reached(last_position, target_position, tolerance, fully_open_success_threshold):
            return True, last_position
        time.sleep(0.05)
    return False, last_position


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from lerobot.robots.franka_research3.backends import DasGripperHardwareDriver

    open_position = _clip_position(args.open_position)
    close_position = _clip_position(args.close_position)
    tolerance = max(0.0, args.position_tolerance)
    move_timeout_s = max(0.0, args.move_timeout_s)
    hold_open_s = max(0.0, args.hold_open_s)
    fully_open_success_threshold = _clip_position(args.fully_open_success_threshold)

    driver = None
    close_attempted = False
    try:
        driver = DasGripperHardwareDriver(
            serial_port=args.gripper_port,
            gen_con_sdk_path=args.gen_con_sdk_path,
            initial_position=open_position,
            baudrate=args.baudrate,
            update_frequency_hz=args.update_frequency_hz,
        )
        print(
            f"das_gripper=CONNECT port={args.gripper_port} "
            f"sdk_path={args.gen_con_sdk_path} baudrate={args.baudrate}"
        )
        driver.connect()

        initial_position = float(driver.get_position())
        print(f"das_gripper=CONNECTED initial_position={initial_position:.4f}")

        print(f"das_gripper=COMMAND open_position={open_position:.4f}")
        driver.set_position(open_position)
        open_ok, measured_open = _wait_for_target(
            driver,
            open_position,
            move_timeout_s,
            tolerance,
            fully_open_success_threshold,
        )
        print(f"das_gripper=MEASURE open_position={measured_open:.4f}")
        print(f"das_gripper=FEEDBACK open_reached={open_ok}")
        if not open_ok:
            print("das_gripper=FAIL reason=open_target_not_reached")
            return 1

        print(f"das_gripper=HOLD duration_s={hold_open_s:.2f}")
        time.sleep(hold_open_s)

        print(f"das_gripper=COMMAND close_position={close_position:.4f}")
        close_attempted = True
        driver.set_position(close_position)
        close_ok, measured_close = _wait_for_target(
            driver,
            close_position,
            move_timeout_s,
            tolerance,
            fully_open_success_threshold,
        )
        print(f"das_gripper=MEASURE close_position={measured_close:.4f}")
        print(f"das_gripper=FEEDBACK close_reached={close_ok}")
        if not close_ok:
            print("das_gripper=FAIL reason=close_target_not_reached")
            return 1

        print("das_gripper=PASS")
        return 0
    finally:
        if driver is not None:
            if not close_attempted:
                try:
                    print(f"das_gripper=SAFETY close_position={close_position:.4f}")
                    driver.set_position(close_position)
                    time.sleep(0.2)
                except Exception as exc:
                    print(f"das_gripper=SAFETY_FAIL reason={exc}")
            driver.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
