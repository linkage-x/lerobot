#!/usr/bin/env python3
"""
FR3 DAS 数据集 MuJoCo 重播启动器（宿主机运行）

用法：
    python tools/fr3/fr3_das_replay.py --episode 0
    python tools/fr3/fr3_das_replay.py --episode 5 --dataset outputs/datasets/lerobotv3_0310_100ep
    python tools/fr3/fr3_das_replay.py --episode 0 --no-viewer   # headless，只输出误差统计
    python tools/fr3/fr3_das_replay.py --dry-run                  # 只打印 docker 命令
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"
DEFAULT_SERVICE = "lerobot-fr3-sim"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch FR3 DAS dataset MuJoCo replay inside the sim container.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset path relative to repo root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--fps", type=int, default=30, help="Replay frame rate (default: 30, matches recording)")
    parser.add_argument("--no-viewer", action="store_true", help="Headless mode: skip MuJoCo viewer, only print metrics")
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
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_das_replay_runtime.py",
        f"--episode={args.episode}",
        f"--dataset=/workspace/{args.dataset}",
        f"--fps={args.fps}",
    ]
    if args.no_viewer:
        runtime_args.append("--no-viewer")

    docker_run_extra: list[str] = []
    if not args.no_viewer:
        display = os.environ.get("DISPLAY", ":0")
        docker_run_extra = [
            "-e", f"DISPLAY={display}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        ]

    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        *docker_run_extra,
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
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
