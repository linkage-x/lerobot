#!/usr/bin/env python3

"""Smoke-test Quest3/Vuer connectivity and hand-tracking data before FR3 control."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np

from lerobot.teleoperators.quest3 import Quest3Teleop, Quest3TeleopConfig
from lerobot.teleoperators.quest3.configuration_quest3 import (
    DEFAULT_QUEST3_CALIBRATION_DIR,
    DEFAULT_QUEST3_CERT_FILE,
    DEFAULT_QUEST3_KEY_FILE,
    Quest3GripperMapping,
    Quest3Hand,
)
from lerobot.utils.utils import init_logging


def _local_ip_hint() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--cert-file", type=Path, default=DEFAULT_QUEST3_CERT_FILE)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_QUEST3_KEY_FILE)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_QUEST3_CALIBRATION_DIR)
    parser.add_argument("--hand", choices=[hand.value for hand in Quest3Hand], default=Quest3Hand.RIGHT.value)
    parser.add_argument(
        "--gripper-mapping",
        choices=[mapping.value for mapping in Quest3GripperMapping],
        default=Quest3GripperMapping.FINGERTIP_DISTANCE.value,
    )
    parser.add_argument("--open-pinch-value", type=float, default=0.111)
    parser.add_argument("--closed-pinch-value", type=float, default=0.004)
    parser.add_argument("--open-fingertip-distance-m", type=float, default=0.085)
    parser.add_argument("--closed-fingertip-distance-m", type=float, default=0.018)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--lost-tracking-timeout-s", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging()

    config = Quest3TeleopConfig(
        host=args.host,
        port=args.port,
        cert_file=args.cert_file,
        key_file=args.key_file,
        calibration_dir=args.calibration_dir,
        hand=Quest3Hand(args.hand),
        gripper_mapping=Quest3GripperMapping(args.gripper_mapping),
        open_pinch_value=args.open_pinch_value,
        closed_pinch_value=args.closed_pinch_value,
        open_fingertip_distance_m=args.open_fingertip_distance_m,
        closed_fingertip_distance_m=args.closed_fingertip_distance_m,
        lost_tracking_timeout_s=args.lost_tracking_timeout_s,
    )
    teleop = Quest3Teleop(config)
    ip_hint = _local_ip_hint()
    print("Quest3 connection smoke starting.")
    print(f"Certificate: {config.cert_file}")
    print(f"Private key: {config.key_file}")
    print(f"Calibration dir: {config.calibration_dir}")
    print(
        "Gripper mapping: {mapping} closed_pinch={closed_pinch:.3f} open_pinch={open_pinch:.3f} "
        "closed_distance={closed:.3f}m open_distance={open:.3f}m".format(
            mapping=config.gripper_mapping.value,
            closed_pinch=config.closed_pinch_value,
            open_pinch=config.open_pinch_value,
            closed=config.closed_fingertip_distance_m,
            open=config.open_fingertip_distance_m,
        )
    )
    print(f"USB reverse URL: https://127.0.0.1:{args.port}?ws=wss://127.0.0.1:{args.port}")
    print(f"Wi-Fi/LAN URL: https://{ip_hint}:{args.port}?ws=wss://{ip_hint}:{args.port}")
    print("If using USB debugging, run: adb reverse tcp:{0} tcp:{0}".format(args.port))

    start = time.perf_counter()
    period_s = 1.0 / max(float(args.print_hz), 1e-6)
    fingertip_distance_min = float("inf")
    fingertip_distance_max = float("-inf")
    try:
        teleop.connect()
        while args.duration_s is None or time.perf_counter() - start < args.duration_s:
            state = teleop.latest_debug_state()
            pose = np.asarray(state["wrist_pose"], dtype=np.float64)
            xyz = pose[:3, 3]
            fingertip_distance = float(state["fingertip_distance_m"])
            if bool(state["tracking_valid"]):
                fingertip_distance_min = min(fingertip_distance_min, fingertip_distance)
                fingertip_distance_max = max(fingertip_distance_max, fingertip_distance)
            print(
                "hand={hand} valid={valid} age={age:.3f}s "
                "pinch={pinch} pinch_value={pinch_value:.3f} pinch_gripper={pinch_gripper:.3f} "
                "squeeze={squeeze} squeeze_value={squeeze_value:.3f} "
                "fingertip_distance_m={fingertip_distance_m:.3f} "
                "distance_range_m=({distance_min:.3f},{distance_max:.3f}) "
                "gripper_unclipped={gripper_unclipped:.3f} gripper={gripper:.3f} "
                "wrist_xyz=({x:.3f}, {y:.3f}, {z:.3f})".format(
                    hand=state["hand"],
                    valid=state["tracking_valid"],
                    age=state["tracking_age_s"],
                    pinch=state["pinch"],
                    pinch_value=state["pinch_value"],
                    pinch_gripper=state["pinch_gripper"],
                    squeeze=state["squeeze"],
                    squeeze_value=state["squeeze_value"],
                    fingertip_distance_m=fingertip_distance,
                    distance_min=fingertip_distance_min if np.isfinite(fingertip_distance_min) else float("nan"),
                    distance_max=fingertip_distance_max if np.isfinite(fingertip_distance_max) else float("nan"),
                    gripper_unclipped=state["gripper_unclipped"],
                    gripper=state["gripper"],
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                )
            )
            time.sleep(period_s)
    finally:
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    main()
