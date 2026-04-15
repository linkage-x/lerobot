#!/usr/bin/env python3
"""
Move the FR3 arm to the DAS replay start joint configuration from inside the Docker runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_ROBOT_IP = "192.168.1.208"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the FR3 arm to the DAS replay start joint configuration.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    return parser.parse_args(argv)


def _import_move_to_das_start():
    try:
        from tools.fr3.fr3_das_replay_real_runtime import move_to_das_start
        return move_to_das_start
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from tools.fr3.fr3_das_replay_real_runtime import move_to_das_start
        return move_to_das_start


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    move_to_das_start = _import_move_to_das_start()

    print(f"fr3_move_to_das_start=CONNECT robot_ip={args.robot_ip}")
    move_to_das_start(args.robot_ip)
    print("fr3_move_to_das_start=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
