#!/usr/bin/env python3
"""
Run FR3 ACT real-robot inference inside the Docker hardware runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

import yaml

DEFAULT_SERVICE = 'lerobot-infer-fr3-act'
DEFAULT_PROFILE = 'infer'
DEFAULT_CHECKPOINT = Path('outputs/train/2026-03-19/10-48-39_act/checkpoints/060000')
DEFAULT_CAMERA_CONFIG = Path('tools/fr3/fr3_act_infer_camera_config.yaml')
CONTAINER_WORKSPACE = '/workspace'
LEGACY_CONTAINER_WORKSPACE = '/lerobot'
ROBOT_INIT_STATE_SHORTHAND_PREFIXES = (
    'joints=',
    'joints:',
    'joint_rad=',
    'joint_rad:',
    'ee_xyzquat=',
    'ee_xyzquat:',
    'xyzquat=',
    'xyzquat:',
    'ee_xyzrotvec=',
    'ee_xyzrotvec:',
    'xyzrotvec=',
    'xyzrotvec:',
)
GRIPPER_BACKEND_CHOICES = ('pika', 'das', 'franka_hand', 'corenetic')


def _normalize_gripper_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == 'box':
        return 'corenetic'
    return normalized


def _normalize_workspace_path(path_value: str) -> str:
    if path_value.startswith(f'{LEGACY_CONTAINER_WORKSPACE}/'):
        return f"{CONTAINER_WORKSPACE}/{path_value.removeprefix(f'{LEGACY_CONTAINER_WORKSPACE}/')}"
    return path_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Docker-based FR3 ACT real-robot inference.')
    parser.add_argument(
        '--inference-config',
        type=Path,
        default=None,
        help='Generated inference YAML from the training dataset view.',
    )
    parser.add_argument('--service', default=None, help='Docker compose service to run.')
    parser.add_argument('--profile', default=None, help='Docker compose profile to enable.')
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
        default=None,
        help='Checkpoint directory relative to repo root.',
    )
    parser.add_argument(
        '--camera-config',
        type=Path,
        default=None,
        help='Camera config YAML relative to repo root. Defaults to the OpenCV-based FR3 inference camera config.',
    )
    parser.add_argument('--dataset-root', default=None, help='Optional dataset root override.')
    parser.add_argument('--policy-fps', type=float, default=None, help='Optional low-rate policy update FPS override.')
    parser.add_argument('--max-steps', type=int, default=None, help='Optional inference loop step limit.')
    parser.add_argument(
        '--preview',
        dest='preview',
        action='store_true',
        default=None,
        help='Print safe runtime targets without sending robot actions.',
    )
    parser.add_argument('--no-preview', dest='preview', action='store_false', help='Disable preview mode from config.')
    parser.add_argument('--robot-ip', default=None, help='Optional FR3 robot IP override.')
    parser.add_argument('--gripper-port', default=None, help='Optional DAS gripper serial port override.')
    parser.add_argument(
        '--gripper-backend',
        type=_normalize_gripper_backend,
        choices=GRIPPER_BACKEND_CHOICES,
        default=None,
        help='Hardware gripper backend to use inside the container.',
    )
    parser.add_argument('--gripper-max-width-mm', type=float, default=None)
    parser.add_argument('--corenetic-bind-ip', dest='corenetic_bind_ip', default=None)
    parser.add_argument('--box-bind-ip', dest='corenetic_bind_ip', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-bind-port', dest='corenetic_bind_port', type=int, default=None)
    parser.add_argument('--box-bind-port', dest='corenetic_bind_port', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-remote-ip', dest='corenetic_remote_ip', default=None)
    parser.add_argument('--box-remote-ip', dest='corenetic_remote_ip', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-remote-port', dest='corenetic_remote_port', type=int, default=None)
    parser.add_argument('--box-remote-port', dest='corenetic_remote_port', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-sdk-dir', dest='corenetic_sdk_dir', default=None)
    parser.add_argument('--box-sdk-dir', dest='corenetic_sdk_dir', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-connect-timeout-s', dest='corenetic_connect_timeout_s', type=float, default=None)
    parser.add_argument('--box-connect-timeout-s', dest='corenetic_connect_timeout_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-poll-interval-s', dest='corenetic_poll_interval_s', type=float, default=None)
    parser.add_argument('--box-poll-interval-s', dest='corenetic_poll_interval_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-stale-threshold-s', dest='corenetic_stale_threshold_s', type=float, default=None)
    parser.add_argument('--box-stale-threshold-s', dest='corenetic_stale_threshold_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        '--no-corenetic-release-mode-on-disconnect',
        dest='corenetic_release_mode_on_disconnect',
        action='store_false',
        default=None,
    )
    parser.add_argument(
        '--no-box-release-mode-on-disconnect',
        dest='corenetic_release_mode_on_disconnect',
        action='store_false',
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--robot-urdf-path', type=Path, default=None)
    parser.add_argument('--target-frame-name', default=None)
    parser.add_argument(
        '--gripper-close-below',
        type=float,
        default=None,
        help=(
            'Optional raw policy gripper threshold. If the model gripper output is below this value, '
            'force the runtime gripper command to 0.'
        ),
    )
    parser.add_argument('--first-frame-max-pos-delta-mm', type=float, default=None)
    parser.add_argument('--first-frame-max-rot-delta-deg', type=float, default=None)
    parser.add_argument('--max-step-pos-delta-mm', type=float, default=None)
    parser.add_argument('--max-step-rot-delta-deg', type=float, default=None)
    parser.add_argument('--max-leash-pos-delta-mm', type=float, default=None)
    parser.add_argument('--max-leash-rot-delta-deg', type=float, default=None)
    parser.add_argument(
        '--use-otg',
        dest='use_otg',
        action='store_true',
        default=None,
        help='Enable FR3 joint-space Ruckig OTG smoothing inside the runtime.',
    )
    parser.add_argument(
        '--no-use-otg',
        dest='use_otg',
        action='store_false',
        help='Disable FR3 joint-space Ruckig OTG smoothing inside the runtime.',
    )
    parser.add_argument('--otg-control-frequency', type=float, default=None)
    parser.add_argument('--otg-async-control-frequency', type=float, default=None)
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
        '--camera-preview-window',
        action='store_true',
        default=None,
        help='Show all policy input camera frames in one OpenCV window inside the runtime.',
    )
    parser.add_argument(
        '--move-to-das-start',
        dest='move_to_das_start',
        action='store_true',
        help=(
            'Move the arm to the DAS rig start joint configuration before inference. Off by '
            'default; see the runtime flag of the same name for why homing to another rig pose '
            'offsets the whole trajectory.'
        ),
    )
    parser.add_argument(
        '--no-move-to-das-start',
        dest='move_to_das_start',
        action='store_false',
        help='Now the default; accepted so existing launchers and configs keep working.',
    )
    parser.add_argument(
        '--no-align-gripper-to-dataset-start',
        dest='align_gripper_to_dataset_start',
        action='store_false',
        help='Skip physically moving the gripper to the dataset-start mean before policy inference begins.',
    )
    parser.add_argument('--align-gripper-to-dataset-start', dest='align_gripper_to_dataset_start', action='store_true')
    parser.add_argument('--dataset-start-gripper-tolerance', type=float, default=None)
    parser.add_argument(
        '--robot-init-state',
        default=None,
        help=(
            'Optional robot startup state before inference. Accepts a YAML/JSON file path, '
            'an inline YAML/JSON object, or shorthand like joints=7 comma-separated radians '
            'or ee_xyzquat=x,y,z,qx,qy,qz,qw.'
        ),
    )
    parser.add_argument(
        '--interactive-rollouts',
        action='store_true',
        default=None,
        help='Use keyboard-driven rollout control: start, stop current rollout, and quit.',
    )
    parser.add_argument('--rollout-start-key', default=None, help='Keyboard key to start a rollout in interactive mode.')
    parser.add_argument('--rollout-stop-key', default=None, help='Keyboard key to stop the current rollout in interactive mode.')
    parser.add_argument('--rollout-quit-key', default=None, help='Keyboard key to quit interactive inference.')
    parser.add_argument(
        '--mujoco-viewer',
        action='store_true',
        default=None,
        help='Open a MuJoCo viewer during real inference and overlay current/target EE markers.',
    )
    parser.add_argument(
        '--mujoco-model',
        type=Path,
        default=None,
        help='Optional MuJoCo XML model path relative to repo root, for example the FR3+Pika XML.',
    )
    parser.add_argument('--mujoco-max-chunk-points', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Print the Docker command without executing it.')
    parser.set_defaults(move_to_das_start=None, align_gripper_to_dataset_start=None)
    return apply_inference_config_defaults(parser.parse_args(argv))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _nested(raw: dict[str, Any], *keys: str) -> Any:
    value: Any = raw
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _path_or_default(value: Any, default: Path | None) -> Path | None:
    return Path(str(value)) if value not in (None, '') else default


def _bool_or_default(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _stringify_config_value(value: Any) -> str | None:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(',', ':'))


def apply_inference_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    raw: dict[str, Any] = {}
    if args.inference_config is not None:
        raw = _read_yaml(args.inference_config)

    args.service = args.service or _nested(raw, 'runtime', 'docker', 'service') or DEFAULT_SERVICE
    args.profile = args.profile or _nested(raw, 'runtime', 'docker', 'profile') or DEFAULT_PROFILE
    args.checkpoint = args.checkpoint or _path_or_default(
        _nested(raw, 'runtime', 'checkpoint') or _nested(raw, 'training', 'checkpoint'),
        DEFAULT_CHECKPOINT,
    )
    args.camera_config = args.camera_config or _path_or_default(
        _nested(raw, 'runtime', 'camera_config'),
        DEFAULT_CAMERA_CONFIG,
    )
    args.dataset_root = args.dataset_root or _nested(raw, 'runtime', 'dataset_root') or _nested(raw, 'training', 'dataset_root')
    args.policy_fps = args.policy_fps if args.policy_fps is not None else _nested(raw, 'runtime', 'policy_fps')
    args.max_steps = args.max_steps if args.max_steps is not None else _nested(raw, 'runtime', 'max_steps')
    args.preview = bool(_nested(raw, 'runtime', 'preview')) if args.preview is None else args.preview
    args.robot_ip = args.robot_ip or _nested(raw, 'runtime', 'hardware', 'robot_ip')
    args.gripper_port = args.gripper_port or _nested(raw, 'runtime', 'hardware', 'gripper_port')
    args.gripper_backend = _normalize_gripper_backend(
        args.gripper_backend or _nested(raw, 'runtime', 'hardware', 'gripper_backend') or 'das'
    )
    args.gripper_max_width_mm = (
        args.gripper_max_width_mm
        if args.gripper_max_width_mm is not None
        else _nested(raw, 'runtime', 'hardware', 'gripper_max_width_mm')
    )
    args.corenetic_bind_ip = (
        args.corenetic_bind_ip
        or _nested(raw, 'runtime', 'hardware', 'corenetic_bind_ip')
        or _nested(raw, 'runtime', 'hardware', 'box_bind_ip')
    )
    args.corenetic_bind_port = (
        args.corenetic_bind_port
        if args.corenetic_bind_port is not None
        else _nested(raw, 'runtime', 'hardware', 'corenetic_bind_port')
        or _nested(raw, 'runtime', 'hardware', 'box_bind_port')
    )
    args.corenetic_remote_ip = (
        args.corenetic_remote_ip
        or _nested(raw, 'runtime', 'hardware', 'corenetic_remote_ip')
        or _nested(raw, 'runtime', 'hardware', 'box_remote_ip')
    )
    args.corenetic_remote_port = (
        args.corenetic_remote_port
        if args.corenetic_remote_port is not None
        else _nested(raw, 'runtime', 'hardware', 'corenetic_remote_port')
        or _nested(raw, 'runtime', 'hardware', 'box_remote_port')
    )
    args.corenetic_sdk_dir = (
        args.corenetic_sdk_dir
        or _nested(raw, 'runtime', 'hardware', 'corenetic_sdk_dir')
        or _nested(raw, 'runtime', 'hardware', 'box_sdk_dir')
    )
    args.corenetic_connect_timeout_s = (
        args.corenetic_connect_timeout_s
        if args.corenetic_connect_timeout_s is not None
        else _nested(raw, 'runtime', 'hardware', 'corenetic_connect_timeout_s')
        or _nested(raw, 'runtime', 'hardware', 'box_connect_timeout_s')
    )
    args.corenetic_poll_interval_s = (
        args.corenetic_poll_interval_s
        if args.corenetic_poll_interval_s is not None
        else _nested(raw, 'runtime', 'hardware', 'corenetic_poll_interval_s')
        or _nested(raw, 'runtime', 'hardware', 'box_poll_interval_s')
    )
    args.corenetic_stale_threshold_s = (
        args.corenetic_stale_threshold_s
        if args.corenetic_stale_threshold_s is not None
        else _nested(raw, 'runtime', 'hardware', 'corenetic_stale_threshold_s')
        or _nested(raw, 'runtime', 'hardware', 'box_stale_threshold_s')
    )
    corenetic_release_default = (
        _nested(raw, 'runtime', 'hardware', 'corenetic_release_mode_on_disconnect')
        if _nested(raw, 'runtime', 'hardware', 'corenetic_release_mode_on_disconnect') is not None
        else _nested(raw, 'runtime', 'hardware', 'box_release_mode_on_disconnect')
    )
    args.corenetic_release_mode_on_disconnect = (
        bool(corenetic_release_default)
        if args.corenetic_release_mode_on_disconnect is None and corenetic_release_default is not None
        else args.corenetic_release_mode_on_disconnect
    )
    args.robot_urdf_path = args.robot_urdf_path or _path_or_default(
        _nested(raw, 'runtime', 'hardware', 'robot_urdf_path'),
        None,
    )
    args.target_frame_name = args.target_frame_name or _nested(raw, 'runtime', 'hardware', 'target_frame_name')
    args.gripper_close_below = (
        args.gripper_close_below
        if args.gripper_close_below is not None
        else _nested(raw, 'runtime', 'control', 'gripper_close_below')
    )
    if args.gripper_close_below is None:
        args.gripper_close_below = _nested(raw, 'runtime', 'gripper_close_below')
    args.first_frame_max_pos_delta_mm = (
        args.first_frame_max_pos_delta_mm
        if args.first_frame_max_pos_delta_mm is not None
        else _nested(raw, 'runtime', 'safety', 'first_frame_max_pos_delta_mm')
    )
    args.first_frame_max_rot_delta_deg = (
        args.first_frame_max_rot_delta_deg
        if args.first_frame_max_rot_delta_deg is not None
        else _nested(raw, 'runtime', 'safety', 'first_frame_max_rot_delta_deg')
    )
    args.max_step_pos_delta_mm = (
        args.max_step_pos_delta_mm
        if args.max_step_pos_delta_mm is not None
        else _nested(raw, 'runtime', 'safety', 'max_step_pos_delta_mm')
    )
    args.max_step_rot_delta_deg = (
        args.max_step_rot_delta_deg
        if args.max_step_rot_delta_deg is not None
        else _nested(raw, 'runtime', 'safety', 'max_step_rot_delta_deg')
    )
    otg_default = _nested(raw, 'runtime', 'control', 'use_otg')
    if otg_default is None:
        otg_default = _nested(raw, 'runtime', 'otg', 'enabled')
    args.use_otg = bool(otg_default) if args.use_otg is None and otg_default is not None else args.use_otg
    args.otg_control_frequency = (
        args.otg_control_frequency
        if args.otg_control_frequency is not None
        else _nested(raw, 'runtime', 'control', 'otg_control_frequency')
    )
    if args.otg_control_frequency is None:
        args.otg_control_frequency = _nested(raw, 'runtime', 'otg', 'control_frequency')
    args.otg_async_control_frequency = (
        args.otg_async_control_frequency
        if args.otg_async_control_frequency is not None
        else _nested(raw, 'runtime', 'control', 'otg_async_control_frequency')
    )
    if args.otg_async_control_frequency is None:
        args.otg_async_control_frequency = _nested(raw, 'runtime', 'otg', 'async_control_frequency')
    args.debug_step0_dump_dir = args.debug_step0_dump_dir or _path_or_default(
        _nested(raw, 'runtime', 'debug_step0_dump_dir'),
        None,
    )
    camera_preview_default = _nested(raw, 'runtime', 'debug', 'camera_preview_window')
    if camera_preview_default is None:
        camera_preview_default = _nested(raw, 'runtime', 'camera_preview_window')
    args.camera_preview_window = (
        bool(camera_preview_default) if args.camera_preview_window is None else args.camera_preview_window
    )
    args.move_to_das_start = (
        _bool_or_default(_nested(raw, 'runtime', 'startup', 'move_to_das_start'), False)
        if args.move_to_das_start is None
        else args.move_to_das_start
    )
    args.align_gripper_to_dataset_start = (
        _bool_or_default(_nested(raw, 'runtime', 'startup', 'align_gripper_to_dataset_start'), True)
        if args.align_gripper_to_dataset_start is None
        else args.align_gripper_to_dataset_start
    )
    args.dataset_start_gripper_tolerance = (
        args.dataset_start_gripper_tolerance
        if args.dataset_start_gripper_tolerance is not None
        else _nested(raw, 'runtime', 'startup', 'dataset_start_gripper_tolerance')
    )
    args.robot_init_state = args.robot_init_state or _stringify_config_value(
        _nested(raw, 'runtime', 'startup', 'robot_init_state')
    )
    interactive_default = _nested(raw, 'runtime', 'interactive_rollouts')
    if interactive_default is None:
        interactive_default = _nested(raw, 'runtime', 'interactive', 'enabled')
    args.interactive_rollouts = bool(interactive_default) if args.interactive_rollouts is None else args.interactive_rollouts
    args.rollout_start_key = args.rollout_start_key or _nested(raw, 'runtime', 'interactive', 'start_key') or 's'
    args.rollout_stop_key = args.rollout_stop_key or _nested(raw, 'runtime', 'interactive', 'stop_key') or 'x'
    args.rollout_quit_key = args.rollout_quit_key or _nested(raw, 'runtime', 'interactive', 'quit_key') or 'q'
    mujoco_default = _nested(raw, 'runtime', 'mujoco', 'enabled')
    if mujoco_default is None:
        mujoco_default = _nested(raw, 'runtime', 'mujoco_viewer')
    args.mujoco_viewer = bool(mujoco_default) if args.mujoco_viewer is None else args.mujoco_viewer
    args.mujoco_model = args.mujoco_model or _path_or_default(_nested(raw, 'runtime', 'mujoco', 'model'), None)
    args.mujoco_max_chunk_points = (
        args.mujoco_max_chunk_points
        if args.mujoco_max_chunk_points is not None
        else _nested(raw, 'runtime', 'mujoco', 'max_chunk_points')
    )
    return args


def _to_container_path(path: Path, workspace: Path) -> str:
    path_str = str(path)
    if path_str.startswith(f'{CONTAINER_WORKSPACE}/'):
        return path_str
    if path_str.startswith(f'{LEGACY_CONTAINER_WORKSPACE}/'):
        return f"{CONTAINER_WORKSPACE}/{path_str.removeprefix(f'{LEGACY_CONTAINER_WORKSPACE}/')}"

    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve() if path.is_absolute() else (resolved_workspace / path).resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(
            f'Path must live inside {resolved_workspace} or already be a {CONTAINER_WORKSPACE} path.'
        ) from exc

    return f'{CONTAINER_WORKSPACE}/{relative.as_posix()}'


def _robot_init_state_to_runtime_arg(value: str, workspace: Path) -> str:
    stripped = str(value).strip()
    if (
        len(stripped) < 512
        and '\n' not in stripped
        and not stripped.lstrip().startswith(('{', '['))
        and not any(stripped.startswith(prefix) for prefix in ROBOT_INIT_STATE_SHORTHAND_PREFIXES)
    ):
        candidate = Path(stripped)
        if candidate.is_absolute() or (workspace / candidate).exists():
            path_for_conversion = candidate if candidate.is_absolute() else workspace / candidate
            return _to_container_path(path_for_conversion, workspace)
    return stripped


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / 'docker' / 'docker-compose.yml'
    checkpoint = _to_container_path(args.checkpoint, workspace)
    camera_config = _to_container_path(args.camera_config, workspace)
    debug_step0_dump_dir = (
        _to_container_path(args.debug_step0_dump_dir, workspace) if args.debug_step0_dump_dir is not None else None
    )
    robot_init_state = (
        _robot_init_state_to_runtime_arg(args.robot_init_state, workspace)
        if args.robot_init_state is not None
        else None
    )
    mujoco_model = _to_container_path(args.mujoco_model, workspace) if args.mujoco_model is not None else None

    runtime_args = [
        'cd /workspace &&',
        'PYTHONPATH=/workspace/src',
        '/lerobot/.venv/bin/python',
        'tools/fr3/fr3_act_infer_real_runtime.py',
        f'--checkpoint={shlex.quote(checkpoint)}',
        f'--camera-config={shlex.quote(camera_config)}',
        f'--gripper-backend={shlex.quote(args.gripper_backend)}',
        *([f"--dataset-root={shlex.quote(_normalize_workspace_path(args.dataset_root))}"] if args.dataset_root is not None else []),
        *([f'--policy-fps={args.policy_fps}'] if args.policy_fps is not None else []),
        *([f'--max-steps={args.max_steps}'] if args.max_steps is not None else []),
        *(['--preview'] if args.preview else []),
        *([f'--tactile-fallback={shlex.quote(args.tactile_fallback)}'] if args.tactile_fallback is not None else []),
        *([f'--debug-step0-dump-dir={shlex.quote(debug_step0_dump_dir)}'] if debug_step0_dump_dir is not None else []),
        *(['--camera-preview-window'] if args.camera_preview_window else []),
        *(['--move-to-das-start'] if args.move_to_das_start else []),
        *([] if args.align_gripper_to_dataset_start else ['--no-align-gripper-to-dataset-start']),
        *([f'--dataset-start-gripper-tolerance={args.dataset_start_gripper_tolerance}'] if args.dataset_start_gripper_tolerance is not None else []),
        *([f'--gripper-close-below={args.gripper_close_below}'] if args.gripper_close_below is not None else []),
        *([f'--robot-init-state={shlex.quote(robot_init_state)}'] if robot_init_state is not None else []),
        *(['--interactive-rollouts'] if args.interactive_rollouts else []),
        f'--rollout-start-key={shlex.quote(args.rollout_start_key)}',
        f'--rollout-stop-key={shlex.quote(args.rollout_stop_key)}',
        f'--rollout-quit-key={shlex.quote(args.rollout_quit_key)}',
        *(['--mujoco-viewer'] if args.mujoco_viewer else []),
        *([f'--mujoco-model={shlex.quote(mujoco_model)}'] if mujoco_model is not None else []),
        *([f'--mujoco-max-chunk-points={args.mujoco_max_chunk_points}'] if args.mujoco_max_chunk_points is not None else []),
        *([f'--robot-ip={shlex.quote(args.robot_ip)}'] if args.robot_ip is not None else []),
        *([f'--gripper-port={shlex.quote(args.gripper_port)}'] if args.gripper_port is not None else []),
        *([f'--gripper-max-width-mm={args.gripper_max_width_mm}'] if args.gripper_max_width_mm is not None else []),
        *([f'--corenetic-bind-ip={shlex.quote(args.corenetic_bind_ip)}'] if args.corenetic_bind_ip is not None else []),
        *([f'--corenetic-bind-port={args.corenetic_bind_port}'] if args.corenetic_bind_port is not None else []),
        *([f'--corenetic-remote-ip={shlex.quote(args.corenetic_remote_ip)}'] if args.corenetic_remote_ip is not None else []),
        *([f'--corenetic-remote-port={args.corenetic_remote_port}'] if args.corenetic_remote_port is not None else []),
        *([f'--corenetic-sdk-dir={shlex.quote(args.corenetic_sdk_dir)}'] if args.corenetic_sdk_dir is not None else []),
        *([f'--corenetic-connect-timeout-s={args.corenetic_connect_timeout_s}'] if args.corenetic_connect_timeout_s is not None else []),
        *([f'--corenetic-poll-interval-s={args.corenetic_poll_interval_s}'] if args.corenetic_poll_interval_s is not None else []),
        *([f'--corenetic-stale-threshold-s={args.corenetic_stale_threshold_s}'] if args.corenetic_stale_threshold_s is not None else []),
        *(['--no-corenetic-release-mode-on-disconnect'] if args.corenetic_release_mode_on_disconnect is False else []),
        *(
            [f'--robot-urdf-path={shlex.quote(_to_container_path(args.robot_urdf_path, workspace))}']
            if args.robot_urdf_path is not None
            else []
        ),
        *([f'--target-frame-name={shlex.quote(args.target_frame_name)}'] if args.target_frame_name is not None else []),
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
        *([f'--max-leash-pos-delta-mm={args.max_leash_pos_delta_mm}'] if args.max_leash_pos_delta_mm is not None else []),
        *([f'--max-leash-rot-delta-deg={args.max_leash_rot_delta_deg}'] if args.max_leash_rot_delta_deg is not None else []),
        *(['--use-otg'] if args.use_otg is True else ['--no-use-otg'] if args.use_otg is False else []),
        *([f'--otg-control-frequency={args.otg_control_frequency}'] if args.otg_control_frequency is not None else []),
        *(
            [f'--otg-async-control-frequency={args.otg_async_control_frequency}']
            if args.otg_async_control_frequency is not None
            else []
        ),
    ]

    return [
        'docker',
        'compose',
        '--profile',
        args.profile,
        '-f',
        str(compose_file),
        'run',
        *([] if args.interactive_rollouts else ['-T']),
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
