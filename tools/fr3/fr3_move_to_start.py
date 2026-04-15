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
Run FR3 move_to_start inside the repository Docker or host runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fr3 import fr3_record


DEFAULT_SERVICE = "lerobot-user"
DEFAULT_ROBOT_IP = "192.168.11.102"
DEFAULT_RUNTIME = "docker"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR3 move_to_start inside Docker or on the host.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument(
        "--runtime",
        choices=("docker", "host"),
        default=DEFAULT_RUNTIME,
        help="Runtime to use. 'host' reuses the repository .venv and host-side FR3 environment.",
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
    parser.add_argument("--dry-run", action="store_true", help="Print the selected runtime command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    runtime_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_move_to_start_runtime.py",
        f"--robot-ip={args.robot_ip}",
    ]
    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        "-T",
        args.service,
        "bash",
        "-lc",
        " ".join(runtime_args),
    ]


def build_host_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    host_python = fr3_record._host_python_executable(workspace)
    runtime_script = workspace / "tools" / "fr3" / "fr3_move_to_start_runtime.py"
    return [
        str(host_python),
        str(runtime_script),
        f"--robot-ip={args.robot_ip}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    if args.runtime == "host":
        command = build_host_command(args)
    else:
        command = build_docker_command(args)

    if args.dry_run:
        if args.runtime == "host":
            print(fr3_record.format_host_command_for_display(command, workspace=workspace))
        else:
            print(shlex.join(command))
        return 0

    run_kwargs: dict[str, object] = {"check": False}
    if args.runtime == "host":
        run_kwargs["cwd"] = workspace
        run_kwargs["env"] = fr3_record.build_host_env(workspace)
    completed = subprocess.run(command, **run_kwargs)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
