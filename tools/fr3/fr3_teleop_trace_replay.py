#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


DEFAULT_HARDWARE_SERVICE = "lerobot-user"
DEFAULT_SIM_SERVICE = "lerobot-fr3-sim"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-based FR3 teleop trace replay.")
    parser.add_argument("--mode", choices=["sim", "hardware"], required=True)
    parser.add_argument(
        "--trace-profile",
        choices=["translation", "combined", "wz"],
        default="translation",
        help="Trace preset. 'combined' replays coupled translation and rotation pulses, while 'wz' isolates yaw rotation.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root mounted into the container.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Trace output path inside the repo. Defaults to outputs/fr3_traces/<mode>_trace.json.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--step-x", type=float, default=0.0002)
    parser.add_argument("--step-y", type=float, default=0.0002)
    parser.add_argument("--step-z", type=float, default=0.0002)
    parser.add_argument("--step-wx", type=float, default=0.0002)
    parser.add_argument("--step-wy", type=float, default=0.0002)
    parser.add_argument("--step-wz", type=float, default=0.0002)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--move-s", type=float, default=0.75)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--robot-ip", default="192.168.1.206")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--service", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    output_path = args.output.resolve() if args.output is not None else workspace / "outputs" / "fr3_traces" / f"{args.mode}_trace.json"
    repo_relative_output = output_path.relative_to(workspace)
    service = args.service or (DEFAULT_SIM_SERVICE if args.mode == "sim" else DEFAULT_HARDWARE_SERVICE)

    runtime_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_teleop_trace_replay_runtime.py",
        f"--mode={args.mode}",
        f"--trace-profile={args.trace_profile}",
        f"--output=/workspace/{repo_relative_output.as_posix()}",
        f"--fps={args.fps}",
        f"--step-x={args.step_x}",
        f"--step-y={args.step_y}",
        f"--step-z={args.step_z}",
        f"--step-wx={args.step_wx}",
        f"--step-wy={args.step_wy}",
        f"--step-wz={args.step_wz}",
        f"--warmup-s={args.warmup_s}",
        f"--move-s={args.move_s}",
        f"--hold-s={args.hold_s}",
        f"--settle-s={args.settle_s}",
    ]
    if args.mode == "hardware":
        runtime_args.append(f"--robot-ip={args.robot_ip}")
        runtime_args.append(f"--gripper-port={args.gripper_port}")

    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        service,
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
