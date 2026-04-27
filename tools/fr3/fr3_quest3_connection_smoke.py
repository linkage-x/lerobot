#!/usr/bin/env python3

"""Smoke-test Quest3/Vuer connectivity — supports both hand-tracking and controller mode."""

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
        default=Quest3GripperMapping.PINCH_VALUE.value,
    )
    parser.add_argument("--open-pinch-value", type=float, default=0.111)
    parser.add_argument("--closed-pinch-value", type=float, default=0.004)
    parser.add_argument("--open-fingertip-distance-m", type=float, default=0.085)
    parser.add_argument("--closed-fingertip-distance-m", type=float, default=0.018)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--lost-tracking-timeout-s", type=float, default=0.25)
    parser.add_argument(
        "--use-controller",
        dest="use_hand_tracking",
        action="store_false",
        help="Use controller (MotionControllers) instead of hand tracking. Right grip=clutch, triggers=gripper.",
    )
    parser.set_defaults(use_hand_tracking=True)
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
        use_hand_tracking=args.use_hand_tracking,
        gripper_mapping=Quest3GripperMapping(args.gripper_mapping),
        open_pinch_value=args.open_pinch_value,
        closed_pinch_value=args.closed_pinch_value,
        open_fingertip_distance_m=args.open_fingertip_distance_m,
        closed_fingertip_distance_m=args.closed_fingertip_distance_m,
        lost_tracking_timeout_s=args.lost_tracking_timeout_s,
    )
    teleop = Quest3Teleop(config)
    ip_hint = _local_ip_hint()
    mode_label = "HAND TRACKING" if args.use_hand_tracking else "CONTROLLER"
    print(f"Quest3 connection smoke — MODE: {mode_label}")
    print(f"Certificate: {config.cert_file}")
    print(f"Private key: {config.key_file}")
    print(f"Calibration dir: {config.calibration_dir}")
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

            hand_valid = bool(state["tracking_valid"])
            ctrl_valid = bool(state.get("controller_valid", False))

            if hand_valid:
                fingertip_distance_min = min(fingertip_distance_min, fingertip_distance)
                fingertip_distance_max = max(fingertip_distance_max, fingertip_distance)

            lines = []
            lines.append(
                "HAND  valid={valid} age={age:.3f}s "
                "pinch={pinch} pinch_val={pinch_value:.3f} squeeze={squeeze} "
                "gripper={gripper:.3f} "
                "wrist=({x:.3f},{y:.3f},{z:.3f})".format(
                    valid=hand_valid,
                    age=state["tracking_age_s"],
                    pinch=state["pinch"],
                    pinch_value=state["pinch_value"],
                    squeeze=state["squeeze_value"],
                    gripper=state["gripper"],
                    x=xyz[0], y=xyz[1], z=xyz[2],
                )
            )
            lines.append(
                "CTRL  valid={valid} age={age:.3f}s "
                "grip={grip:.3f} trigger={trigger:.3f} "
                "btn_a={btn_a} btn_b={btn_b} "
                "pos=({px:.3f},{py:.3f},{pz:.3f})".format(
                    valid=ctrl_valid,
                    age=state.get("controller_age_s", float("inf")),
                    grip=state.get("controller_grip", 0.0),
                    trigger=state.get("controller_trigger", 0.0),
                    btn_a=state.get("controller_button_a", False),
                    btn_b=state.get("controller_button_b", False),
                    px=state.get("controller_pos", np.zeros(3))[0],
                    py=state.get("controller_pos", np.zeros(3))[1],
                    pz=state.get("controller_pos", np.zeros(3))[2],
                )
            )

            if hand_valid:
                lines.append(
                    "GRIP  fingertip={ft:.3f}m range=({dmin:.3f},{dmax:.3f}) unclipped={unclip:.3f}".format(
                        ft=fingertip_distance,
                        dmin=fingertip_distance_min if np.isfinite(fingertip_distance_min) else float("nan"),
                        dmax=fingertip_distance_max if np.isfinite(fingertip_distance_max) else float("nan"),
                        unclip=state["gripper_unclipped"],
                    )
                )
            elif ctrl_valid:
                lines.append(
                    "GRIP  ctrl_gripper={g:.3f}".format(g=state.get("gripper_from_controller", 0.0))
                )

            if not hand_valid and not ctrl_valid:
                lines.append("  (no data from either source)")

            print(" | ".join(lines))
            time.sleep(period_s)
    finally:
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    main()
