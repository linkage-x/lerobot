#!/usr/bin/env python3
"""
Move the FR3 arm to the DAS replay start joint configuration from inside the Docker runtime.
"""

from __future__ import annotations

import argparse


DEFAULT_ROBOT_IP = "192.168.1.208"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the FR3 arm to the DAS replay start joint configuration.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from tools.fr3.fr3_das_replay_real_runtime import move_to_das_start

    print(f"fr3_move_to_das_start=CONNECT robot_ip={args.robot_ip}")
    move_to_das_start(args.robot_ip)
    print("fr3_move_to_das_start=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
