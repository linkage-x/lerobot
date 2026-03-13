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
Run a direct Pika gripper smoke test inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import time


DEFAULT_GRIPPER_PORT = "/dev/ttyUSB80"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct Pika gripper smoke test.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="Pika serial port.")
    parser.add_argument("--max-width-mm", type=float, default=90.0, help="Expected fully-open gripper width.")
    parser.add_argument(
        "--target-width-mm",
        type=float,
        default=20.0,
        help="Intermediate width to command during the smoke test.",
    )
    parser.add_argument(
        "--return-width-mm",
        type=float,
        default=90.0,
        help="Final width to command before exiting.",
    )
    parser.add_argument("--hold-s", type=float, default=1.0, help="Sleep after each command.")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="Extra settle time after enable before issuing the first gripper command.",
    )
    parser.add_argument(
        "--feedback-timeout-s",
        type=float,
        default=2.0,
        help="Maximum time to wait for motor position feedback to change after each command.",
    )
    return parser.parse_args(argv)


def _clip_width(width_mm: float, max_width_mm: float) -> float:
    return max(0.0, min(float(width_mm), float(max_width_mm)))


def _snapshot(gripper) -> dict[str, object]:
    motor_data = gripper.get_motor_data()
    motor_status = gripper.get_motor_status()
    latest_data = gripper.serial_comm.get_latest_data()
    return {
        "motor_position_rad": float(motor_data.get("Position", 0.0)),
        "motor_speed_rad_s": float(motor_data.get("Speed", 0.0)),
        "motor_current_ma": float(motor_data.get("Current", 0.0)),
        "status": str(motor_status.get("Status", "0x00")),
        "bus_current_ma": float(motor_status.get("BusCurrent", 0.0)),
        "voltage_v": float(motor_status.get("Voltage", 0.0)),
        "latest_data_keys": sorted(latest_data.keys()),
    }


def _print_snapshot(label: str, snapshot: dict[str, object]) -> None:
    print(
        f"{label} "
        f"position_rad={snapshot['motor_position_rad']:.6f} "
        f"speed_rad_s={snapshot['motor_speed_rad_s']:.6f} "
        f"current_ma={snapshot['motor_current_ma']:.1f} "
        f"status={snapshot['status']} "
        f"bus_current_ma={snapshot['bus_current_ma']:.1f} "
        f"voltage_v={snapshot['voltage_v']:.3f} "
        f"latest_data_keys={snapshot['latest_data_keys']}"
    )


def _wait_for_feedback(gripper, baseline_position_rad: float, timeout_s: float) -> tuple[bool, dict[str, object]]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_snapshot = _snapshot(gripper)
    while time.perf_counter() < deadline:
        last_snapshot = _snapshot(gripper)
        if abs(float(last_snapshot["motor_position_rad"]) - baseline_position_rad) > 1e-4:
            return True, last_snapshot
        time.sleep(0.05)
    return False, last_snapshot


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from pika.gripper import Gripper

    gripper = None
    try:
        print(f"pika_gripper=CONNECT port={args.gripper_port}")
        gripper = Gripper(args.gripper_port)
        if not gripper.connect():
            raise ConnectionError(f"Could not connect to Pika gripper on {args.gripper_port}.")
        if not gripper.enable():
            raise ConnectionError("Could not enable the Pika gripper.")
        time.sleep(max(0.0, args.settle_s))

        initial_width_mm = float(gripper.get_gripper_distance())
        initial_snapshot = _snapshot(gripper)
        print(f"pika_gripper=CONNECTED initial_width_mm={initial_width_mm:.3f}")
        _print_snapshot("pika_gripper=STATE initial", initial_snapshot)

        target_width_mm = _clip_width(args.target_width_mm, args.max_width_mm)
        print(f"pika_gripper=COMMAND target_width_mm={target_width_mm:.3f}")
        baseline_position_rad = float(initial_snapshot["motor_position_rad"])
        gripper.set_gripper_distance(target_width_mm)
        target_feedback_ok, target_snapshot = _wait_for_feedback(
            gripper,
            baseline_position_rad=baseline_position_rad,
            timeout_s=max(args.hold_s, args.feedback_timeout_s),
        )
        measured_target_width_mm = float(gripper.get_gripper_distance())
        print(f"pika_gripper=MEASURE target_width_mm={measured_target_width_mm:.3f}")
        _print_snapshot("pika_gripper=STATE target", target_snapshot)
        print(f"pika_gripper=FEEDBACK target_changed={target_feedback_ok}")
        if not target_feedback_ok:
            print("pika_gripper=FAIL reason=no_position_feedback_after_target_command")
            return 1

        return_width_mm = _clip_width(args.return_width_mm, args.max_width_mm)
        print(f"pika_gripper=COMMAND return_width_mm={return_width_mm:.3f}")
        baseline_position_rad = float(target_snapshot["motor_position_rad"])
        gripper.set_gripper_distance(return_width_mm)
        return_feedback_ok, return_snapshot = _wait_for_feedback(
            gripper,
            baseline_position_rad=baseline_position_rad,
            timeout_s=max(args.hold_s, args.feedback_timeout_s),
        )
        measured_return_width_mm = float(gripper.get_gripper_distance())
        print(f"pika_gripper=MEASURE return_width_mm={measured_return_width_mm:.3f}")
        _print_snapshot("pika_gripper=STATE return", return_snapshot)
        print(f"pika_gripper=FEEDBACK return_changed={return_feedback_ok}")
        if not return_feedback_ok:
            print("pika_gripper=FAIL reason=no_position_feedback_after_return_command")
            return 1

        print("pika_gripper=PASS")
        return 0
    finally:
        if gripper is not None:
            try:
                gripper.disable()
            finally:
                gripper.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
