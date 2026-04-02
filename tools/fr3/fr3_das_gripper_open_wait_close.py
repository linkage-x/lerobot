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
Run a Docker-based DAS gripper open-wait-close helper from the host machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEFAULT_SERVICE = "lerobot-fr3-sim-teleop"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_GEN_CON_SDK_PATH = "/opt/dependencies/gen_con_sdk_python_release"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the DAS gripper, wait, then close it via Docker.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="DAS controller serial port.")
    parser.add_argument(
        "--gen-con-sdk-path",
        default=DEFAULT_GEN_CON_SDK_PATH,
        help="Path to the Gen Controller SDK root inside the container.",
    )
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Docker compose service to run.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains docker/docker-compose.yml.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.",
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
    parser.add_argument("--baudrate", type=int, default=921600, help="Controller baudrate.")
    parser.add_argument(
        "--update-frequency-hz",
        type=float,
        default=50.0,
        help="Encoder update frequency requested from the controller.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    runtime_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_das_gripper_open_wait_close_runtime.py",
        f"--gripper-port={args.gripper_port}",
        f"--gen-con-sdk-path={args.gen_con_sdk_path}",
        f"--open-position={args.open_position}",
        f"--close-position={args.close_position}",
        f"--hold-open-s={args.hold_open_s}",
        f"--move-timeout-s={args.move_timeout_s}",
        f"--position-tolerance={args.position_tolerance}",
        f"--baudrate={args.baudrate}",
        f"--update-frequency-hz={args.update_frequency_hz}",
    ]
    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        args.service,
        "bash",
        "-lc",
        " ".join(runtime_args),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_docker_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
