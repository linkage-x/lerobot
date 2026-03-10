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
Run a reusable Docker-based FR3 real-hardware teleoperation smoke test.

This wraps the conservative first-entry command used for FR3 + SpaceMouse bring-up.
It is intentionally configured for short, low-risk smoke sessions rather than
normal teleoperation or recording.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEFAULT_SERVICE = "lerobot-user"
DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_URDF_PATH = "/lerobot/src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.urdf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-based FR3 teleoperation smoke test.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="Pika serial port to probe before fallback.")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Docker compose service to run.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to mount into the container.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Teleoperation loop frequency.")
    parser.add_argument("--duration", type=float, default=30.0, help="Smoke duration in seconds.")
    parser.add_argument("--device-id", type=int, default=0, help="SpaceMouse device index.")
    parser.add_argument(
        "--tool-mode",
        choices=["binary", "incremental"],
        default="binary",
        help="SpaceMouse gripper button mode.",
    )
    parser.add_argument("--incremental-step", type=float, default=0.01, help="Incremental gripper step size.")
    parser.add_argument("--move-time", type=float, default=0.02, help="Incremental gripper update interval.")
    parser.add_argument("--scale-x", type=float, default=0.0001, help="SpaceMouse X translation scale.")
    parser.add_argument("--scale-y", type=float, default=0.0001, help="SpaceMouse Y translation scale.")
    parser.add_argument("--scale-z", type=float, default=0.0001, help="SpaceMouse Z translation scale.")
    parser.add_argument(
        "--allow-rotation",
        action="store_true",
        help="Enable rotational teleoperation. Disabled by default for first-entry smoke.",
    )
    parser.add_argument(
        "--max-target-delta-pos",
        default="[0.001,0.001,0.001]",
        help="Per-step EE translation clamp passed to the robot config.",
    )
    parser.add_argument(
        "--max-target-delta-rot",
        default="[0.0,0.0,0.0]",
        help="Per-step EE rotation clamp passed to the robot config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    teleop_args = [
        "cd /lerobot &&",
        "PYTHONPATH=/lerobot/src",
        "/lerobot/.venv/bin/lerobot-teleoperate",
        f"--fps={args.fps}",
        f"--teleop_time_s={args.duration}",
        "--robot.type=franka_research3",
        f"--robot.robot_ip={args.robot_ip}",
        f"--robot.gripper_port={args.gripper_port}",
        f"--robot.urdf_path={DEFAULT_URDF_PATH}",
        "--robot.target_frame_name=pika_gripper_ee",
        f"--robot.max_target_delta_pos={args.max_target_delta_pos}",
        f"--robot.max_target_delta_rot={args.max_target_delta_rot}",
        "--teleop.type=spacemouse",
        f"--teleop.device_id={args.device_id}",
        f"--teleop.tool_mode={args.tool_mode}",
        f"--teleop.enable_rotation={'true' if args.allow_rotation else 'false'}",
        f"--teleop.scale_x={args.scale_x}",
        f"--teleop.scale_y={args.scale_y}",
        f"--teleop.scale_z={args.scale_z}",
    ]
    if args.tool_mode == "incremental":
        teleop_args.append(f"--teleop.incremental_step={args.incremental_step}")
        teleop_args.append(f"--teleop.move_time={args.move_time}")

    command = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        args.service,
        "bash",
        "-lc",
        " ".join(teleop_args),
    ]
    return command


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
