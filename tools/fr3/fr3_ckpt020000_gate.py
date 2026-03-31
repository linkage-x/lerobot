#!/usr/bin/env python3
"""Prepare or run the standard FR3 checkpoint-020000 gate sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

import fr3_act_infer_real as infer_cli
import fr3_check_policy_dataset_frame as offline_cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CAMERA_CONFIG = Path('tools/fr3/fr3_act_infer_camera_config.yaml')
_DEFAULT_DATASET_ROOT = 'outputs/datasets/lerobotv3_0310_100ep_aligned_ts'
_DEFAULT_EPISODES = '0,13'
_DEFAULT_FRAME_INDICES = '0,1,2,4,8,16,24,32,40'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare or run the ckpt-020000 FR3 validation gates.')
    parser.add_argument('--run-name', default=None, help='Training run directory name under outputs/train.')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Explicit checkpoint path. Overrides --run-name.')
    parser.add_argument('--dataset-root', default=_DEFAULT_DATASET_ROOT)
    parser.add_argument('--camera-config', type=Path, default=_DEFAULT_CAMERA_CONFIG)
    parser.add_argument('--episodes', default=_DEFAULT_EPISODES)
    parser.add_argument('--frame-indices', default=_DEFAULT_FRAME_INDICES)
    parser.add_argument('--preview-max-steps', type=int, default=5)
    parser.add_argument('--real-max-steps', type=int, default=10)
    parser.add_argument('--first-frame-max-pos-delta-mm', type=float, default=20.0)
    parser.add_argument('--first-frame-max-rot-delta-deg', type=float, default=8.0)
    parser.add_argument('--max-step-pos-delta-mm', type=float, default=3.0)
    parser.add_argument('--max-step-rot-delta-deg', type=float, default=2.0)
    parser.add_argument(
        '--debug-step0-dump-dir',
        type=Path,
        default=None,
        help='Optional explicit dump dir for preview step0 bundles.',
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Execute offline and preview gates immediately. Default is print-only.',
    )
    parser.add_argument(
        '--run-real',
        action='store_true',
        help='After preview, also run the short real rollout command.',
    )
    return parser.parse_args(argv)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_name is None:
        raise ValueError('Pass either --checkpoint or --run-name.')
    return Path('outputs/train') / args.run_name / 'checkpoints' / '020000'


def resolve_step0_dump_dir(args: argparse.Namespace, checkpoint: Path) -> Path:
    if args.debug_step0_dump_dir is not None:
        return args.debug_step0_dump_dir
    run_name = args.run_name or checkpoint.parents[1].name
    return Path('outputs/validation') / run_name / 'ckpt_020000_preview_step0'


def build_offline_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    offline_args = offline_cli.parse_args([])
    offline_args.checkpoint = checkpoint
    offline_args.dataset_root = args.dataset_root
    offline_args.episodes = args.episodes
    offline_args.frame_indices = args.frame_indices
    return offline_cli.build_docker_command(offline_args)


def build_preview_command(args: argparse.Namespace, checkpoint: Path, step0_dump_dir: Path) -> list[str]:
    preview_args = infer_cli.parse_args([])
    preview_args.checkpoint = checkpoint
    preview_args.camera_config = args.camera_config
    preview_args.dataset_root = args.dataset_root
    preview_args.preview = True
    preview_args.max_steps = args.preview_max_steps
    preview_args.first_frame_max_pos_delta_mm = args.first_frame_max_pos_delta_mm
    preview_args.first_frame_max_rot_delta_deg = args.first_frame_max_rot_delta_deg
    preview_args.max_step_pos_delta_mm = args.max_step_pos_delta_mm
    preview_args.max_step_rot_delta_deg = args.max_step_rot_delta_deg
    preview_args.debug_step0_dump_dir = step0_dump_dir
    return infer_cli.build_docker_command(preview_args)


def build_real_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    real_args = infer_cli.parse_args([])
    real_args.checkpoint = checkpoint
    real_args.camera_config = args.camera_config
    real_args.dataset_root = args.dataset_root
    real_args.preview = False
    real_args.max_steps = args.real_max_steps
    real_args.first_frame_max_pos_delta_mm = args.first_frame_max_pos_delta_mm
    real_args.first_frame_max_rot_delta_deg = args.first_frame_max_rot_delta_deg
    real_args.max_step_pos_delta_mm = args.max_step_pos_delta_mm
    real_args.max_step_rot_delta_deg = args.max_step_rot_delta_deg
    return infer_cli.build_docker_command(real_args)


def print_command(label: str, command: list[str]) -> None:
    print(f'[{label}]')
    print(shlex.join(command))
    print()


def run_command(label: str, command: list[str]) -> int:
    print(f'[RUN] {label}')
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args)
    step0_dump_dir = resolve_step0_dump_dir(args, checkpoint)

    offline_command = build_offline_command(args, checkpoint)
    preview_command = build_preview_command(args, checkpoint, step0_dump_dir)
    real_command = build_real_command(args, checkpoint)

    print(f'[INFO] checkpoint={checkpoint}')
    print(f'[INFO] dataset_root={args.dataset_root}')
    print(f'[INFO] step0_dump_dir={step0_dump_dir}')
    print()
    print_command('OFFLINE', offline_command)
    print_command('PREVIEW', preview_command)
    print_command('REAL_SHORT', real_command)

    if not args.run:
        return 0

    offline_rc = run_command('offline', offline_command)
    if offline_rc != 0:
        return offline_rc

    preview_rc = run_command('preview', preview_command)
    if preview_rc != 0 or not args.run_real:
        return preview_rc

    return run_command('real_short', real_command)


if __name__ == '__main__':
    raise SystemExit(main())
