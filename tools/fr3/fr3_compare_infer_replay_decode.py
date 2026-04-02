#!/usr/bin/env python3
"""Run offline infer-vs-replay decode comparison inside the Docker infer runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

DEFAULT_SERVICE = 'lerobot-infer-fr3-act'
DEFAULT_PROFILE = 'infer'
DEFAULT_DATASET = Path('outputs/datasets/lerobotv3_0310_100ep')
CONTAINER_WORKSPACE = '/workspace'
LEGACY_CONTAINER_WORKSPACE = '/lerobot'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run offline infer-vs-replay decode comparison inside Docker.')
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
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET, help='Dataset root relative to repo root.')
    parser.add_argument('--episode', type=int, default=0, help='Episode index to sample.')
    parser.add_argument('--source', choices=['state', 'action'], default='action', help='Dataset pose source to compare.')
    parser.add_argument('--frame-indices', default=None, help='Optional comma-separated frame indices to compare.')
    parser.add_argument('--max-frames', type=int, default=8, help='Compare the first N frames when frame-indices is omitted.')
    parser.add_argument(
        '--start-pose-b-xyzquat',
        default=None,
        help='Optional B/E start pose override as x,y,z,qx,qy,qz,qw. Defaults to replay reset pose.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print the Docker command without executing it.')
    return parser.parse_args(argv)


def _to_container_path(path: Path, workspace: Path) -> str:
    path_str = str(path)
    if path_str.startswith(f'{CONTAINER_WORKSPACE}/'):
        return path_str
    if path_str.startswith(f'{LEGACY_CONTAINER_WORKSPACE}/'):
        return f"{CONTAINER_WORKSPACE}/{path_str.removeprefix(f'{LEGACY_CONTAINER_WORKSPACE}/')}"

    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(
            f'Path must live inside {resolved_workspace} or already be a {CONTAINER_WORKSPACE} path.'
        ) from exc

    return f'{CONTAINER_WORKSPACE}/{relative.as_posix()}'


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / 'docker' / 'docker-compose.yml'
    dataset_path = _to_container_path(args.dataset, workspace)

    runtime_args = [
        'cd /workspace &&',
        'PYTHONPATH=/workspace/src:/workspace/tools/fr3',
        '/lerobot/.venv/bin/python',
        'tools/fr3/fr3_compare_infer_replay_decode_runtime.py',
        f'--dataset={shlex.quote(dataset_path)}',
        f'--episode={args.episode}',
        f'--source={shlex.quote(args.source)}',
        *([f'--frame-indices={shlex.quote(args.frame_indices)}'] if args.frame_indices is not None else []),
        f'--max-frames={args.max_frames}',
        *([f'--start-pose-b-xyzquat={shlex.quote(args.start_pose_b_xyzquat)}'] if args.start_pose_b_xyzquat is not None else []),
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
