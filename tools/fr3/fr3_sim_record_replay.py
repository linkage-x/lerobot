#!/usr/bin/env python3
"""
FR3 MuJoCo 仿真录制数据集重播启动器（宿主机运行）

用法：
    python tools/fr3/fr3_sim_record_replay.py --dataset outputs/datasets/fr3_sim_record_20260421_072232
    python tools/fr3/fr3_sim_record_replay.py --episode 0 --dataset outputs/datasets/fr3_sim_record_20260421_072232
    python tools/fr3/fr3_sim_record_replay.py --dataset outputs/datasets/fr3_sim_record_20260421_072232 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_SERVICE = "lerobot-fr3-sim-teleop"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch FR3 sim recording dataset replay inside the sim container.")
    parser.add_argument("--episode", type=int, default=None, help="Episode index to replay (default: replay all episodes)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (relative to repo root or absolute)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Replay frame rate (default: 30)")
    parser.add_argument("--no-viewer", action="store_true", help="Headless mode: skip MuJoCo viewer")
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

    if args.dataset:
        if args.dataset.startswith("/"):
            dataset_path = args.dataset
        else:
            dataset_path = f"/workspace/{args.dataset}"
    else:
        dataset_path = "/workspace/outputs/datasets/fr3_sim_record"

    runtime_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_sim_record_replay_runtime.py",
        f"--dataset={dataset_path}",
        f"--fps={args.fps}",
    ]
    if args.episode is not None:
        runtime_args.append(f"--episode={args.episode}")
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
