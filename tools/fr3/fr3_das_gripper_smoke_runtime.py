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
Run a direct DAS gripper controller smoke test inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import time


DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_GEN_CON_SDK_PATH = "/opt/dependencies/gen_con_sdk_python_release"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct DAS gripper controller smoke test.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="DAS controller serial port.")
    parser.add_argument(
        "--gen-con-sdk-path",
        default=DEFAULT_GEN_CON_SDK_PATH,
        help="Path to the Gen Controller SDK root.",
    )
    parser.add_argument(
        "--initial-position",
        type=float,
        default=0.5,
        help="Normalized initial position sent during connect.",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        default=0.8,
        help="Normalized intermediate position to command during the smoke test.",
    )
    parser.add_argument(
        "--return-position",
        type=float,
        default=0.2,
        help="Normalized final position to command before exiting.",
    )
    parser.add_argument("--hold-s", type=float, default=0.5, help="Minimum time to wait after each command.")
    parser.add_argument(
        "--feedback-timeout-s",
        type=float,
        default=2.0,
        help="Maximum time to wait for encoder feedback to change after each command.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.01,
        help="Minimum normalized position delta considered as feedback change.",
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


def _wait_for_position_change(driver, baseline_position: float, timeout_s: float, tolerance: float) -> tuple[bool, float]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_position = float(driver.get_position())
    while time.perf_counter() < deadline:
        last_position = float(driver.get_position())
        if abs(last_position - baseline_position) >= tolerance:
            return True, last_position
        time.sleep(0.05)
    return False, last_position


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from lerobot.robots.franka_research3.backends import DasGripperHardwareDriver

    driver = None
    try:
        driver = DasGripperHardwareDriver(
            serial_port=args.gripper_port,
            gen_con_sdk_path=args.gen_con_sdk_path,
            initial_position=_clip_position(args.initial_position),
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

        target_position = _clip_position(args.target_position)
        print(f"das_gripper=COMMAND target_position={target_position:.4f}")
        driver.set_position(target_position)
        target_feedback_ok, measured_target = _wait_for_position_change(
            driver,
            baseline_position=initial_position,
            timeout_s=max(args.hold_s, args.feedback_timeout_s),
            tolerance=max(0.0, args.position_tolerance),
        )
        print(f"das_gripper=MEASURE target_position={measured_target:.4f}")
        print(f"das_gripper=FEEDBACK target_changed={target_feedback_ok}")
        if not target_feedback_ok:
            print("das_gripper=FAIL reason=no_encoder_feedback_after_target_command")
            return 1

        return_position = _clip_position(args.return_position)
        print(f"das_gripper=COMMAND return_position={return_position:.4f}")
        driver.set_position(return_position)
        return_feedback_ok, measured_return = _wait_for_position_change(
            driver,
            baseline_position=measured_target,
            timeout_s=max(args.hold_s, args.feedback_timeout_s),
            tolerance=max(0.0, args.position_tolerance),
        )
        print(f"das_gripper=MEASURE return_position={measured_return:.4f}")
        print(f"das_gripper=FEEDBACK return_changed={return_feedback_ok}")
        if not return_feedback_ok:
            print("das_gripper=FAIL reason=no_encoder_feedback_after_return_command")
            return 1

        print("das_gripper=PASS")
        return 0
    finally:
        if driver is not None:
            driver.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
