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
Run a Docker-based direct DAS gripper controller smoke test.
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
    parser = argparse.ArgumentParser(description="Run a Docker-based direct DAS gripper controller smoke test.")
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
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    runtime_args = [
        "cd /lerobot &&",
        "PYTHONPATH=/lerobot/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_das_gripper_smoke_runtime.py",
        f"--gripper-port={args.gripper_port}",
        f"--gen-con-sdk-path={args.gen_con_sdk_path}",
        f"--initial-position={args.initial_position}",
        f"--target-position={args.target_position}",
        f"--return-position={args.return_position}",
        f"--hold-s={args.hold_s}",
        f"--feedback-timeout-s={args.feedback_timeout_s}",
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
