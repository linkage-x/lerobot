#!/usr/bin/env python3
"""Run offline checkpoint-vs-dataset frame comparisons inside the Docker infer runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

DEFAULT_SERVICE = 'lerobot-infer-fr3-act'
DEFAULT_PROFILE = 'infer'
DEFAULT_CHECKPOINT = Path('outputs/train/2026-03-19/10-48-39_act/checkpoints/060000')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Docker-side checkpoint-vs-dataset frame comparisons.')
    parser.add_argument('--service', default=DEFAULT_SERVICE, help='Docker compose service to run.')
    parser.add_argument('--profile', default=DEFAULT_PROFILE, help='Docker compose profile to enable.')
    parser.add_argument(
        '--workspace',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Repository root to mount into the container.',
    )
    parser.add_argument(
        '--compose-file',
        type=Path,
        default=None,
        help='Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.',
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT, help='Checkpoint directory relative to repo root.')
    parser.add_argument('--dataset-root', default=None, help='Optional dataset root override.')
    parser.add_argument('--episodes', default='0,13', help='Comma-separated episode indices, or all-starts.')
    parser.add_argument('--frame-indices', default='0', help='Comma-separated frame indices to evaluate per episode.')
    parser.add_argument('--device', default=None, help='Optional torch device override inside the container.')
    parser.add_argument('--dry-run', action='store_true', help='Print the Docker command without executing it.')
    return parser.parse_args(argv)


def _to_container_path(path: Path, workspace: Path) -> str:
    path_str = str(path)
    if path_str.startswith('/lerobot/'):
        return path_str

    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(f'Path must live inside {resolved_workspace} or already be a /lerobot path.') from exc

    return f'/lerobot/{relative.as_posix()}'


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / 'docker' / 'docker-compose.yml'
    checkpoint = _to_container_path(args.checkpoint, workspace)

    runtime_args = [
        'cd /lerobot &&',
        'PYTHONPATH=/lerobot/src:/lerobot/tools/fr3',
        '/lerobot/.venv/bin/python',
        'tools/fr3/fr3_check_policy_dataset_frame_runtime.py',
        f'--checkpoint={shlex.quote(checkpoint)}',
        f'--episodes={shlex.quote(args.episodes)}',
        f'--frame-indices={shlex.quote(args.frame_indices)}',
        *([f'--dataset-root={shlex.quote(args.dataset_root)}'] if args.dataset_root is not None else []),
        *([f'--device={shlex.quote(args.device)}'] if args.device is not None else []),
    ]

    return [
        'docker',
        'compose',
        '--profile',
        args.profile,
        '-f',
        str(compose_file),
        'run',
        '--rm',
        args.service,
        'bash',
        '-lc',
        ' '.join(runtime_args),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_docker_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
