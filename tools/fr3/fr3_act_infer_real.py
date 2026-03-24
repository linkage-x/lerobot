#!/usr/bin/env python3
"""
Run FR3 ACT real-robot inference inside the Docker hardware runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

DEFAULT_SERVICE = 'lerobot-infer-fr3-act'
DEFAULT_PROFILE = 'infer'
DEFAULT_CHECKPOINT = Path('outputs/train/2026-03-19/10-48-39_act/checkpoints/060000')
DEFAULT_CAMERA_CONFIG = Path('tools/fr3/fr3_act_infer_camera_config.yaml')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Docker-based FR3 ACT real-robot inference.')
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
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help='Checkpoint directory relative to repo root.',
    )
    parser.add_argument(
        '--camera-config',
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help='Camera config YAML relative to repo root. Defaults to the OpenCV-based FR3 inference camera config.',
    )
    parser.add_argument('--dataset-root', default=None, help='Optional dataset root override.')
    parser.add_argument('--policy-fps', type=float, default=None, help='Optional low-rate policy update FPS override.')
    parser.add_argument('--max-steps', type=int, default=None, help='Optional inference loop step limit.')
    parser.add_argument('--preview', action='store_true', help='Print safe runtime targets without sending robot actions.')
    parser.add_argument('--robot-ip', default=None, help='Optional FR3 robot IP override.')
    parser.add_argument('--gripper-port', default=None, help='Optional DAS gripper serial port override.')
    parser.add_argument(
        '--gripper-backend',
        choices=['pika', 'das'],
        default='das',
        help='Hardware gripper backend to use inside the container.',
    )
    parser.add_argument('--first-frame-max-pos-delta-mm', type=float, default=None)
    parser.add_argument('--first-frame-max-rot-delta-deg', type=float, default=None)
    parser.add_argument('--max-step-pos-delta-mm', type=float, default=None)
    parser.add_argument('--max-step-rot-delta-deg', type=float, default=None)
    parser.add_argument(
        '--tactile-fallback',
        choices=['baseline_idle'],
        default=None,
        help='Preview-only tactile fallback to inject baseline no-contact tactile into runtime.',
    )
    parser.add_argument(
        '--debug-step0-dump-dir',
        type=Path,
        default=None,
        help='Optional output directory to dump the exact step0 policy input bundle for offline comparison.',
    )
    parser.add_argument(
        '--no-move-to-das-start',
        dest='move_to_das_start',
        action='store_false',
        help='Skip moving the arm to the DAS start joint configuration before inference.',
    )
    parser.add_argument(
        '--no-align-gripper-to-dataset-start',
        dest='align_gripper_to_dataset_start',
        action='store_false',
        help='Skip physically moving the gripper to the dataset-start mean before policy inference begins.',
    )
    parser.add_argument('--dataset-start-gripper-tolerance', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Print the Docker command without executing it.')
    parser.set_defaults(move_to_das_start=True, align_gripper_to_dataset_start=True)
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
    camera_config = _to_container_path(args.camera_config, workspace)
    debug_step0_dump_dir = (
        _to_container_path(args.debug_step0_dump_dir, workspace) if args.debug_step0_dump_dir is not None else None
    )

    runtime_args = [
        'cd /lerobot &&',
        'PYTHONPATH=/lerobot/src',
        '/lerobot/.venv/bin/python',
        'tools/fr3/fr3_act_infer_real_runtime.py',
        f'--checkpoint={shlex.quote(checkpoint)}',
        f'--camera-config={shlex.quote(camera_config)}',
        f'--gripper-backend={shlex.quote(args.gripper_backend)}',
        *([f'--dataset-root={shlex.quote(args.dataset_root)}'] if args.dataset_root is not None else []),
        *([f'--policy-fps={args.policy_fps}'] if args.policy_fps is not None else []),
        *([f'--max-steps={args.max_steps}'] if args.max_steps is not None else []),
        *(['--preview'] if args.preview else []),
        *([f'--tactile-fallback={shlex.quote(args.tactile_fallback)}'] if args.tactile_fallback is not None else []),
        *([f'--debug-step0-dump-dir={shlex.quote(debug_step0_dump_dir)}'] if debug_step0_dump_dir is not None else []),
        *([] if args.move_to_das_start else ['--no-move-to-das-start']),
        *([] if args.align_gripper_to_dataset_start else ['--no-align-gripper-to-dataset-start']),
        *([f'--dataset-start-gripper-tolerance={args.dataset_start_gripper_tolerance}'] if args.dataset_start_gripper_tolerance is not None else []),
        *([f'--robot-ip={shlex.quote(args.robot_ip)}'] if args.robot_ip is not None else []),
        *([f'--gripper-port={shlex.quote(args.gripper_port)}'] if args.gripper_port is not None else []),
        *(
            [f'--first-frame-max-pos-delta-mm={args.first_frame_max_pos_delta_mm}']
            if args.first_frame_max_pos_delta_mm is not None
            else []
        ),
        *(
            [f'--first-frame-max-rot-delta-deg={args.first_frame_max_rot_delta_deg}']
            if args.first_frame_max_rot_delta_deg is not None
            else []
        ),
        *([f'--max-step-pos-delta-mm={args.max_step_pos_delta_mm}'] if args.max_step_pos_delta_mm is not None else []),
        *([f'--max-step-rot-delta-deg={args.max_step_rot_delta_deg}'] if args.max_step_rot_delta_deg is not None else []),
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
