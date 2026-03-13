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
DEFAULT_TRANSLATION_SCALE = 0.000615
DEFAULT_ROTATION_SCALE = 0.000648
DEFAULT_TRANSLATION_MAX_TARGET_DELTA_POS = "[0.001,0.001,0.001]"
DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT = "[0.0,0.0,0.0]"
DEFAULT_ROTATION_MAX_TARGET_DELTA_POS = "[0.0,0.0,0.0]"
DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT = "[0.01,0.01,0.01]"
DEFAULT_COMBINED_MAX_TARGET_DELTA_POS = DEFAULT_TRANSLATION_MAX_TARGET_DELTA_POS
DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT = DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT
DEFAULT_ROTATION_ONLY_TRANSLATION_SCALE = 0.0
DEFAULT_WZ_ONLY_TRANSLATION_SCALE = 0.0
DEFAULT_WZ_ONLY_SCALE_WX = 0.0
DEFAULT_WZ_ONLY_SCALE_WY = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-based FR3 teleoperation smoke test.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT, help="Pika serial port to probe before fallback.")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Docker compose service to run.")
    parser.add_argument(
        "--smoke-profile",
        choices=["translation", "rotation", "combined", "wz"],
        default="translation",
        help="Smoke preset. 'rotation' disables translation, 'combined' enables conservative coupled translation + rotation checks, and 'wz' isolates yaw rotation.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to mount into the container.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.",
    )
    parser.add_argument("--fps", type=int, default=200, help="Teleoperation loop frequency.")
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
    parser.add_argument("--translation-scale", type=float, default=None, help="Unified SpaceMouse translation scale.")
    parser.add_argument("--rotation-scale", type=float, default=None, help="Unified SpaceMouse rotation scale.")
    parser.add_argument("--scale-x", type=float, default=None, help="Optional X translation scale override.")
    parser.add_argument("--scale-y", type=float, default=None, help="Optional Y translation scale override.")
    parser.add_argument("--scale-z", type=float, default=None, help="Optional Z translation scale override.")
    parser.add_argument("--scale-wx", type=float, default=None, help="Optional WX rotation scale override.")
    parser.add_argument("--scale-wy", type=float, default=None, help="Optional WY rotation scale override.")
    parser.add_argument("--scale-wz", type=float, default=None, help="Optional WZ rotation scale override.")
    parser.add_argument(
        "--allow-rotation",
        action="store_true",
        help="Enable rotational teleoperation. Disabled by default for first-entry smoke.",
    )
    parser.add_argument(
        "--max-target-delta-pos",
        default=None,
        help="Per-step EE translation clamp passed to the robot config.",
    )
    parser.add_argument(
        "--max-target-delta-rot",
        default=None,
        help="Per-step EE rotation clamp passed to the robot config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    args = parser.parse_args(argv)

    if args.smoke_profile == "rotation":
        if args.translation_scale is None:
            args.translation_scale = DEFAULT_ROTATION_ONLY_TRANSLATION_SCALE
        if args.rotation_scale is None:
            args.rotation_scale = DEFAULT_ROTATION_SCALE
        if args.max_target_delta_pos is None:
            args.max_target_delta_pos = DEFAULT_ROTATION_MAX_TARGET_DELTA_POS
        if args.max_target_delta_rot is None:
            args.max_target_delta_rot = DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT
        args.allow_rotation = True
    elif args.smoke_profile == "wz":
        if args.translation_scale is None:
            args.translation_scale = DEFAULT_WZ_ONLY_TRANSLATION_SCALE
        if args.rotation_scale is None:
            args.rotation_scale = DEFAULT_ROTATION_SCALE
        if args.scale_wx is None:
            args.scale_wx = DEFAULT_WZ_ONLY_SCALE_WX
        if args.scale_wy is None:
            args.scale_wy = DEFAULT_WZ_ONLY_SCALE_WY
        if args.max_target_delta_pos is None:
            args.max_target_delta_pos = DEFAULT_ROTATION_MAX_TARGET_DELTA_POS
        if args.max_target_delta_rot is None:
            args.max_target_delta_rot = DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT
        args.allow_rotation = True
    elif args.smoke_profile == "combined":
        if args.translation_scale is None:
            args.translation_scale = DEFAULT_TRANSLATION_SCALE
        if args.rotation_scale is None:
            args.rotation_scale = DEFAULT_ROTATION_SCALE
        if args.max_target_delta_pos is None:
            args.max_target_delta_pos = DEFAULT_COMBINED_MAX_TARGET_DELTA_POS
        if args.max_target_delta_rot is None:
            args.max_target_delta_rot = DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT
        args.allow_rotation = True
    else:
        if args.translation_scale is None:
            args.translation_scale = DEFAULT_TRANSLATION_SCALE
        if args.rotation_scale is None:
            args.rotation_scale = DEFAULT_ROTATION_SCALE
        if args.max_target_delta_pos is None:
            args.max_target_delta_pos = DEFAULT_TRANSLATION_MAX_TARGET_DELTA_POS
        if args.max_target_delta_rot is None:
            args.max_target_delta_rot = DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT

    return args


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
        f"--teleop.translation_scale={args.translation_scale}",
        f"--teleop.rotation_scale={args.rotation_scale}",
    ]
    for key, value in (
        ("scale_x", args.scale_x),
        ("scale_y", args.scale_y),
        ("scale_z", args.scale_z),
        ("scale_wx", args.scale_wx),
        ("scale_wy", args.scale_wy),
        ("scale_wz", args.scale_wz),
    ):
        if value is not None:
            teleop_args.append(f"--teleop.{key}={value}")
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
