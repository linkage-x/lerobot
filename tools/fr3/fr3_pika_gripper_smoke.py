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
Run a Docker-based direct Pika gripper smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEFAULT_SERVICE = "lerobot-user"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-based direct Pika gripper smoke test.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="Pika serial port.")
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
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    runtime_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_pika_gripper_smoke_runtime.py",
        f"--gripper-port={args.gripper_port}",
        f"--max-width-mm={args.max_width_mm}",
        f"--target-width-mm={args.target_width_mm}",
        f"--return-width-mm={args.return_width_mm}",
        f"--hold-s={args.hold_s}",
        f"--settle-s={args.settle_s}",
        f"--feedback-timeout-s={args.feedback_timeout_s}",
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
