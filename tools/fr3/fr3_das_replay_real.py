#!/usr/bin/env python3
"""
FR3 DAS 数据集真机重播启动器（宿主机运行）

用法：
    python tools/fr3/fr3_das_replay_real.py --episode 0
    python tools/fr3/fr3_das_replay_real.py --episode 5 --dataset outputs/datasets/lerobotv3_0310_100ep
    python tools/fr3/fr3_das_replay_real.py --episode 0 --dry-run   # 只打印 docker 命令
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"
# 服务名须与 docker/docker-compose.yml 中真机容器的服务名一致（此处复用 sim 容器，仅去掉 X11 挂载）
DEFAULT_SERVICE = "lerobot-fr3-sim"
DEFAULT_ROBOT_IP = "192.168.1.208"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB80"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch FR3 DAS dataset real-robot replay inside the Docker container."
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset path relative to repo root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--fps", type=int, default=30, help="Replay frame rate (default: 30)")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help=f"FR3 robot IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT,
                        help=f"Pika gripper serial port (default: {DEFAULT_GRIPPER_PORT})")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (auto-detected by default)",
    )
    parser.add_argument("--compose-file", type=Path, default=None)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--dry-run", action="store_true", help="Print docker command without running")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file else workspace / "docker" / "docker-compose.yml"

    runtime_args = [
        "cd /lerobot &&",
        "PYTHONPATH=/lerobot/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_das_replay_real_runtime.py",
        f"--episode={args.episode}",
        f"--dataset={shlex.quote(f'/lerobot/{args.dataset}')}",
        f"--fps={args.fps}",
        f"--robot-ip={shlex.quote(args.robot_ip)}",
        f"--gripper-port={shlex.quote(args.gripper_port)}",
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
    cmd = build_docker_command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] docker compose run 失败，退出码 {result.returncode}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
