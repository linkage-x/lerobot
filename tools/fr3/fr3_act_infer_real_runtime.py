#!/usr/bin/env python3
"""
Minimal FR3 ACT real-robot inference runtime (runs inside Docker).

Execution model:
1. Read the training checkpoint and dataset metadata.
2. Open the FR3 inference cameras via OpenCV.
3. Run low-rate policy inference at the dataset FPS.
4. Convert each absolute EE action to a robot EE command.
5. Hand the command to FrankaResearch3, which performs IK and joint-space OTG
   smoothing before sending high-rate joint targets to the controller.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

# Both roots, before anything below is imported. `lerobot` lives under src/ and the launcher
# exports it on PYTHONPATH, but `tools.*` -- the takeover, the command guard, the control channel
# -- is imported by package path and nothing puts the repo root on sys.path: running a file as a
# script puts *the file's own directory* there, not the working directory. Same bootstrap the
# other FR3 runtimes carry, and for the same reason.
_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
for _bootstrap_path in (str(_BOOTSTRAP_REPO_ROOT / 'src'), str(_BOOTSTRAP_REPO_ROOT)):
    if _bootstrap_path not in sys.path:
        sys.path.insert(0, _bootstrap_path)

import numpy as np
import torch
import yaml

from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
from lerobot.cameras.gmsl2.configuration_gmsl2 import Gmsl2CameraConfig
from lerobot.cameras.hikrobot.configuration_hikrobot import HikrobotCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotObservation
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.processor.core import TransitionKey
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.robots.franka_research3.processor_franka_research3 import (
    DeltaEEToAbsoluteEEAction,
    EE_POSITION_KEYS,
    EE_QUAT_KEYS,
    KeepAbsoluteEEObservation,
    PREV_CMD_GRIPPER_KEY,
    PREV_CMD_POSITION_KEYS,
    PREV_CMD_QUAT_KEYS,
    PREV_CMD_ROTVEC_KEYS,
    _continuous_quaternion,
    delta_ee_rotvec_keys,
    delta_reference_from_action_names,
)
from lerobot.utils.control_utils import predict_action, prepare_observation_for_inference
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

from tools.fr3.command_guard import (
    compute_pose_delta_from_current,
    limit_command_for_safety,
    smooth_robot_command_ema,
)
from tools.fr3.command_guard import PREV_CMD_POSITION_KEYS as _GUARD_PREV_CMD_POSITION_KEYS
from tools.fr3.command_guard import PREV_CMD_ROTVEC_KEYS as _GUARD_PREV_CMD_ROTVEC_KEYS
from tools.fr3.dagger_takeover import (
    ExpertTakeover,
    backend_dates_reports,
    expert_spans,
    motion_gain_for,
    undated_backend_error,
)
from tools.fr3.workspace_fence import resolve_workspace_fence
from tools.fr3.dagger_dataset import (
    DEFAULT_MAX_BUFFERED_FRAMES,
    DaggerEpisodeWriter,
    DaggerFrameBuffer,
    build_dagger_frame,
    dagger_dataset_can_load_locally,
    dagger_dataset_features,
    dagger_dataset_is_unfinalized,
    dagger_dataset_root_is_recreatable,
    image_source_keys,
    sent_command_to_dataset_action,
)
from tools.fr3.interactive_control import InteractiveRolloutKeyboard
from tools.fr3.scene_reset import (
    PoseProbeRequest,
    SceneResetError,
    execute_pose_probe,
    execute_scene_reset,
    pose_probe_request_from_payload,
    scene_reset_request_from_payload,
)
from tools.fr3.live_frames import LiveFrameEmitter

# The guard carries its own copy of the prev_cmd key names so that it can be imported
# without the policy stack. Checked here, the one place both definitions are in scope: a
# rename in the processor would otherwise leave the step guard quietly falling back to the
# measured pose, which is the failure this pair of tuples exists to prevent.
assert _GUARD_PREV_CMD_POSITION_KEYS == PREV_CMD_POSITION_KEYS, (
    'command_guard.PREV_CMD_POSITION_KEYS has drifted from the FR3 processor'
)
assert _GUARD_PREV_CMD_ROTVEC_KEYS == PREV_CMD_ROTVEC_KEYS, (
    'command_guard.PREV_CMD_ROTVEC_KEYS has drifted from the FR3 processor'
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPO_ROOT / 'outputs/train/2026-03-19/10-48-39_act/checkpoints/060000'
_DEFAULT_CAMERA_CONFIG = _REPO_ROOT / 'tools/fr3/fr3_act_infer_camera_config.yaml'
_DEFAULT_ROBOT_IP = '192.168.1.208'
_DEFAULT_GRIPPER_PORT = '/dev/ttyUSB0'
_DEFAULT_GRIPPER_BACKEND = 'das'
_GRIPPER_BACKEND_CHOICES = ('pika', 'das', 'franka_hand', 'corenetic')
_DAS_XML = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.xml'
_PIKA_XML = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml'
_DAS_URDF = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf'
_PIKA_URDF = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf'
_DEFAULT_TACTILE_VALID_MASK_PATH = _REPO_ROOT / 'docs/tactile/tactile_valid_mask_50x10.json'
_DEFAULT_TACTILE_BASELINE_PATH = _REPO_ROOT / 'docs/tactile/idle_baseline.json'
_DEFAULT_STATE_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
_DEFAULT_ACTION_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
_DEFAULT_OPENCV_FOURCC = 'MJPG'
_DEFAULT_OPENCV_BACKEND = Cv2Backends.V4L2
_OBS_IMAGES_PREFIX = 'observation.images.'
_DEFAULT_FIRST_FRAME_MAX_POS_DELTA_MM = 30.0
_DEFAULT_FIRST_FRAME_MAX_ROT_DELTA_DEG = 10.0
# How far the *policy* may ask the EE to move in one step, measured against the pose its action is
# defined relative to (``prev_cmd``). This is the number the training data bounds: in
# eeframe_fr3_spacemouse_20260813_160401 the recorded per-step delta has p50 1.59 mm and p95
# 2.93 mm, and a 5.0 mm magnitude admits 99.90% of demo frames -- so this clips genuine outliers
# and leaves the demonstrated motion alone.
_DEFAULT_MAX_STEP_POS_DELTA_MM = 5.0
_DEFAULT_MAX_STEP_ROT_DELTA_DEG = 3.0

# How far the command may run ahead of where the arm actually is. This is a different quantity
# from the one above and it is not policy aggression: it is servo tracking lag. The command leads,
# the impedance controller follows, and the gap between them is what produces the force that moves
# the arm -- so a healthy moving arm always has one. In the very demonstrations this policy was
# trained on that gap runs to p50 5.71 mm and p95 10.65 mm (max 15.92 mm), which is why the leash
# is sized at 20 mm: it admits 100% of the recorded frames. It exists to catch a command running
# away from an arm that is stuck, blocked, or not tracking at all, which is a failure the step
# limit above cannot see.
#
# Judging the lag by the *step* limit is what pinned a healthy rollout at 299/299 clamped: the lag
# alone (median 4.18 mm on that run) already exceeded the 3 mm the launcher was passing, before
# the policy contributed a single millimetre. The demos themselves would have been clamped on
# 61.1% of their frames by that same test.
_DEFAULT_MAX_LEASH_POS_DELTA_MM = 20.0
_DEFAULT_MAX_LEASH_ROT_DELTA_DEG = 8.0
_DEFAULT_DATASET_START_GRIPPER_TOLERANCE = 0.05
_DEFAULT_USE_OTG = bool(FrankaResearch3Config.__dataclass_fields__['use_otg'].default)
_DEFAULT_OTG_CONTROL_FREQUENCY = float(
    FrankaResearch3Config.__dataclass_fields__['otg_control_frequency'].default
)
_DEFAULT_OTG_ASYNC_CONTROL_FREQUENCY = float(
    FrankaResearch3Config.__dataclass_fields__['otg_async_control_frequency'].default
)
_DEFAULT_CONTROLLER_STIFFNESS = (600.0, 600.0, 600.0, 600.0, 280.0, 180.0, 70.0)
_DEFAULT_CONTROLLER_DAMPING = (50.0, 50.0, 50.0, 50.0, 20.0, 15.0, 10.0)
_DAS_START_JOINTS_RAD = np.array(
    [
        -0.053397256451184094,
        -1.5604194603713035,
        -1.720175311909912,
        -2.119629211414152,
        0.011555741406479218,
        2.1189401256121045,
        -0.9682376640047694,
    ],
    dtype=np.float64,
)
_TACTILE_FALLBACK_CHOICES = ('baseline_idle',)
_RTC_MODE_CHOICES = ('auto', 'enabled', 'disabled')
_RTC_POLICY_TYPES = {'pi0', 'pi05', 'pi0_fast', 'smolvla'}
_DEFAULT_RTC_EXECUTION_HORIZON = 16
_DEFAULT_RTC_MAX_GUIDANCE_WEIGHT = 10.0
_DEFAULT_RTC_PREFIX_ATTENTION_SCHEDULE = RTCAttentionSchedule.EXP
_DEFAULT_RTC_REPLAN_QUEUE_SIZE = 25
_JOINT_NAMES = [
    'fr3_joint1',
    'fr3_joint2',
    'fr3_joint3',
    'fr3_joint4',
    'fr3_joint5',
    'fr3_joint6',
    'fr3_joint7',
]


def _normalize_gripper_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == 'box':
        return 'corenetic'
    return normalized


def _load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = _resolve_repo_path(dataset_root) / 'meta' / 'info.json'
    return json.loads(info_path.read_text(encoding='utf-8'))


def _resolve_dataset_data_file(dataset_root: Path, *, chunk_index: int, file_index: int) -> Path:
    dataset_root = _resolve_repo_path(dataset_root)
    info = _load_dataset_info(dataset_root)
    candidates: list[Path] = []
    data_path_template = info.get('data_path')
    if isinstance(data_path_template, str):
        candidates.append(dataset_root / data_path_template.format(chunk_index=chunk_index, file_index=file_index))
    candidates.extend(
        [
            dataset_root / 'data' / f'chunk-{chunk_index:03d}' / f'file-{file_index:03d}.parquet',
            dataset_root / 'data' / f'chunk-{chunk_index:03d}' / f'file-{file_index:06d}.parquet',
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'Could not resolve data parquet for chunk_index={chunk_index} file_index={file_index} under {dataset_root}'
    )


def _load_observation_state_feature_names(dataset_root: Path, state_key: str = 'observation.state') -> list[str]:
    info = _load_dataset_info(dataset_root)
    names = info.get('features', {}).get(state_key, {}).get('names')
    if not isinstance(names, list):
        return [*EE_POSITION_KEYS, *EE_QUAT_KEYS, 'gripper.pos']
    return [str(name) for name in names]


def _extract_dataset_state_contract_indices(dataset_root: Path, state_key: str = 'observation.state') -> dict[str, int]:
    state_names = _load_observation_state_feature_names(dataset_root, state_key=state_key)
    required_names = ['ee.x', 'ee.y', 'ee.z', 'ee.qx', 'ee.qy', 'ee.qz', 'ee.qw']
    missing_names = [name for name in required_names if name not in state_names]
    if missing_names:
        raise KeyError(f'Dataset {state_key} names are missing required entries: {missing_names}')
    indices = {name: state_names.index(name) for name in required_names}
    if 'gripper.pos' in state_names:
        indices['gripper.pos'] = state_names.index('gripper.pos')
    return indices


def _extract_pose_gripper_from_state_row(
    state_row: np.ndarray,
    *,
    state_indices: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, float | None]:
    state_row = np.asarray(state_row, dtype=np.float64)
    position = np.asarray([state_row[state_indices[key]] for key in EE_POSITION_KEYS], dtype=np.float64)
    quaternion = np.asarray([state_row[state_indices[key]] for key in EE_QUAT_KEYS], dtype=np.float64)
    gripper = None
    if 'gripper.pos' in state_indices:
        gripper = float(state_row[state_indices['gripper.pos']])
    return position, quaternion, gripper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run FR3 ACT real-robot inference inside Docker.')
    parser.add_argument('--checkpoint', type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--camera-config',
        type=Path,
        default=_DEFAULT_CAMERA_CONFIG,
        help='Camera config YAML. Defaults to the OpenCV-based FR3 inference camera config.',
    )
    parser.add_argument('--dataset-root', default=None, help='Optional dataset root override.')
    parser.add_argument(
        '--task-prompt',
        default=None,
        help=(
            'Task prompt to send to language-conditioned policies. If omitted, the runtime uses the '
            'single task stored in the checkpoint dataset/view; multi-task views require this flag.'
        ),
    )
    parser.add_argument('--policy-fps', type=float, default=None, help='Optional low-rate policy update FPS override.')
    parser.add_argument(
        '--policy-n-action-steps',
        type=int,
        default=None,
        help=(
            'Optional runtime override for policy.config.n_action_steps. '
            'Use 1 for receding-horizon inference that replans every control step.'
        ),
    )
    parser.add_argument(
        '--act-temporal-ensemble-coeff',
        type=float,
        default=None,
        help=(
            'Optional ACT temporal ensembling coefficient. The original ACT default is 0.01. '
            'Positive values favor older chunk predictions; negative values favor newer predictions. '
            'When enabled, policy.config.n_action_steps is forced to 1 because ACT must be queried every step.'
        ),
    )
    parser.add_argument(
        '--act-temporal-action-offset',
        type=int,
        default=0,
        help=(
            'When ACT temporal ensembling is enabled, execute this many frames ahead in the ensembled action '
            'sequence instead of the immediate first action. Use 0 for the default immediate action.'
        ),
    )
    parser.add_argument(
        '--act-temporal-stuck-max-offset',
        type=int,
        default=None,
        help=(
            'Optional maximum temporal action offset to use when closed-gripper policy targets are stuck. '
            'When set, the runtime gradually increases the ACT temporal offset toward this value instead of '
            'always executing the immediate/near action from the ensembled chunk.'
        ),
    )
    parser.add_argument(
        '--act-temporal-stuck-offset-step',
        type=int,
        default=2,
        help='Temporal action offset increment applied each time the closed-gripper stuck detector fires.',
    )
    parser.add_argument(
        '--act-temporal-stuck-steps',
        type=int,
        default=12,
        help='Consecutive closed-gripper low-motion steps required before increasing temporal offset.',
    )
    parser.add_argument(
        '--act-temporal-stuck-pos-delta-mm',
        type=float,
        default=3.0,
        help='Unassisted EE target distance below which the policy is considered stuck for temporal offset advance.',
    )
    parser.add_argument(
        '--act-temporal-stuck-closed-gripper-max',
        type=float,
        default=0.05,
        help='Normalized gripper command at or below this value is considered closed for temporal offset advance.',
    )
    parser.add_argument(
        '--rtc-mode',
        choices=_RTC_MODE_CHOICES,
        default='auto',
        help=(
            'Real-Time Chunking mode for flow-matching chunk policies. '
            'auto enables RTC only for pi0/pi0.5/pi0_fast/SmolVLA; disabled preserves the checkpoint default queue.'
        ),
    )
    parser.add_argument('--rtc', dest='rtc_mode', action='store_const', const='enabled', help='Enable RTC.')
    parser.add_argument('--no-rtc', dest='rtc_mode', action='store_const', const='disabled', help='Disable RTC.')
    parser.add_argument('--rtc-auto', dest='rtc_mode', action='store_const', const='auto', help='Auto-enable RTC for supported policies.')
    parser.add_argument(
        '--rtc-execution-horizon',
        type=int,
        default=_DEFAULT_RTC_EXECUTION_HORIZON,
        help='RTC overlap horizon, in policy steps. Typical pi0/pi0.5 values are 8-12.',
    )
    parser.add_argument(
        '--rtc-max-guidance-weight',
        type=float,
        default=_DEFAULT_RTC_MAX_GUIDANCE_WEIGHT,
        help='RTC guidance strength. 10.0 is the recommended starting point for 10-step pi0/pi0.5 inference.',
    )
    parser.add_argument(
        '--rtc-prefix-attention-schedule',
        choices=[schedule.value for schedule in RTCAttentionSchedule],
        default=_DEFAULT_RTC_PREFIX_ATTENTION_SCHEDULE.value,
        help='RTC prefix weighting schedule. EXP is the conservative default for real-robot rollout.',
    )
    parser.add_argument(
        '--rtc-replan-queue-size',
        type=int,
        default=_DEFAULT_RTC_REPLAN_QUEUE_SIZE,
        help=(
            'Request a new action chunk when this many postprocessed actions remain. '
            'For chunk_size=50, 30 replans after about 20 executed steps and leaves overlap for RTC.'
        ),
    )
    parser.add_argument(
        '--rtc-inference-delay-steps',
        type=int,
        default=None,
        help='Optional fixed inference delay in policy steps. Omit to estimate from measured chunk latency.',
    )
    parser.add_argument(
        '--command-ema-alpha',
        type=float,
        default=None,
        help=(
            'Optional EMA smoothing for decoded EE commands before safety clamp. '
            'Use lightly with chunk/RTC policies; heavy values can delay grasp and insertion corrections. '
            '1.0 disables smoothing; smaller values are smoother.'
        ),
    )
    parser.add_argument(
        '--place-assist-offset-base-xyz',
        default=None,
        help=(
            'Optional comma-separated xyz offset in robot base frame, in meters. '
            'When enabled, the offset is ramped into the EE command only after the policy is closed-gripper '
            'and appears stuck for --place-assist-stuck-steps consecutive steps.'
        ),
    )
    parser.add_argument(
        '--place-assist-stuck-steps',
        type=int,
        default=20,
        help='Consecutive closed-gripper low-motion policy steps required before place assist starts.',
    )
    parser.add_argument(
        '--place-assist-stuck-pos-delta-mm',
        type=float,
        default=3.0,
        help='Policy is considered stuck when the unassisted EE target is within this distance of current EE.',
    )
    parser.add_argument(
        '--place-assist-ramp-step-mm',
        type=float,
        default=1.5,
        help='Maximum place-assist offset ramp speed per policy step, in mm.',
    )
    parser.add_argument(
        '--place-assist-closed-gripper-max',
        type=float,
        default=0.05,
        help='Normalized gripper command at or below this value is considered closed for place assist.',
    )
    parser.add_argument('--max-steps', type=int, default=None, help='Optional inference loop step limit.')
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Run policy and print safe targets without sending robot actions.',
    )
    parser.add_argument('--robot-ip', default=_DEFAULT_ROBOT_IP)
    parser.add_argument('--gripper-port', default=_DEFAULT_GRIPPER_PORT)
    parser.add_argument(
        '--gripper-backend',
        type=_normalize_gripper_backend,
        choices=_GRIPPER_BACKEND_CHOICES,
        default=_DEFAULT_GRIPPER_BACKEND,
    )
    parser.add_argument(
        '--gripper-max-width-mm',
        type=float,
        default=90.0,
        help='Physical gripper maximum opening in millimeters, used to normalize dataset gripper units.',
    )
    parser.add_argument('--corenetic-bind-ip', dest='corenetic_bind_ip', default='0.0.0.0', help='Local IP for Corenetic gripper UDP bind.')
    parser.add_argument('--box-bind-ip', dest='corenetic_bind_ip', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-bind-port', dest='corenetic_bind_port', type=int, default=15000, help='Local UDP port for Corenetic gripper.')
    parser.add_argument('--box-bind-port', dest='corenetic_bind_port', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-remote-ip', dest='corenetic_remote_ip', default='192.168.2.60', help='Corenetic gripper MCU IP.')
    parser.add_argument('--box-remote-ip', dest='corenetic_remote_ip', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-remote-port', dest='corenetic_remote_port', type=int, default=15000, help='Corenetic gripper MCU UDP port.')
    parser.add_argument('--box-remote-port', dest='corenetic_remote_port', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-sdk-dir', dest='corenetic_sdk_dir', default='tools/thor/box_sdk', help='Corenetic/BOX SDK directory relative to repo root.')
    parser.add_argument('--box-sdk-dir', dest='corenetic_sdk_dir', help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-connect-timeout-s', dest='corenetic_connect_timeout_s', type=float, default=3.0)
    parser.add_argument('--box-connect-timeout-s', dest='corenetic_connect_timeout_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-poll-interval-s', dest='corenetic_poll_interval_s', type=float, default=0.01)
    parser.add_argument('--box-poll-interval-s', dest='corenetic_poll_interval_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--corenetic-stale-threshold-s', dest='corenetic_stale_threshold_s', type=float, default=1.0)
    parser.add_argument('--box-stale-threshold-s', dest='corenetic_stale_threshold_s', type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        '--no-corenetic-release-mode-on-disconnect',
        dest='corenetic_release_mode_on_disconnect',
        action='store_false',
        help='Do not switch Corenetic gripper back to collection mode on disconnect.',
    )
    parser.add_argument('--no-box-release-mode-on-disconnect', dest='corenetic_release_mode_on_disconnect', action='store_false', help=argparse.SUPPRESS)
    parser.add_argument('--robot-urdf-path', type=Path, default=None, help='Optional FR3 tool URDF override for IK.')
    parser.add_argument('--target-frame-name', default=None, help='Optional IK target frame override.')
    parser.add_argument(
        '--gripper-close-below',
        type=float,
        default=None,
        help=(
            'Optional raw policy gripper threshold. If the model gripper output is below this value, '
            'force the robot gripper command to 0 before unit normalization. Disabled by default.'
        ),
    )
    parser.add_argument(
        '--gripper-change-delay-s',
        type=float,
        default=None,
        help=(
            'Optional minimum seconds between accepted gripper command changes. '
            'Comparison is done in normalized [0, 1] gripper scale.'
        ),
    )
    parser.add_argument(
        '--gripper-change-min-delta',
        type=float,
        default=0.08,
        help=(
            'Minimum normalized [0, 1] difference between observed gripper state and desired command '
            'before accepting a gripper command change when --gripper-change-delay-s is enabled.'
        ),
    )
    parser.add_argument(
        '--gripper-change-settle-tolerance',
        type=float,
        default=0.12,
        help=(
            'Normalized [0, 1] tolerance for considering the observed gripper state settled near the '
            'latched command before another gripper command change is allowed.'
        ),
    )
    parser.add_argument(
        '--gripper-change-settle-timeout-s',
        type=float,
        default=1.5,
        help=(
            'Maximum seconds to wait for observed gripper state to settle near the latched command before '
            'allowing a new gripper change. This prevents permanent lockout when grasping an object.'
        ),
    )
    parser.add_argument(
        '--use-otg',
        dest='use_otg',
        action='store_true',
        default=_DEFAULT_USE_OTG,
        help='Enable FR3 joint-space Ruckig OTG smoothing after IK.',
    )
    parser.add_argument(
        '--no-use-otg',
        dest='use_otg',
        action='store_false',
        help='Disable FR3 joint-space Ruckig OTG smoothing and send IK joint targets directly.',
    )
    parser.add_argument(
        '--otg-control-frequency',
        type=float,
        default=_DEFAULT_OTG_CONTROL_FREQUENCY,
        help='FR3 Ruckig OTG planning frequency in Hz.',
    )
    parser.add_argument(
        '--otg-async-control-frequency',
        type=float,
        default=_DEFAULT_OTG_ASYNC_CONTROL_FREQUENCY,
        help='FR3 joint command sender frequency in Hz when OTG is enabled.',
    )
    parser.add_argument(
        '--controller-stiffness',
        default=None,
        help='Optional comma-separated 7D FR3 joint-position controller stiffness. Disabled when omitted.',
    )
    parser.add_argument(
        '--controller-damping',
        default=None,
        help='Optional comma-separated 7D FR3 joint-position controller damping. Disabled when omitted.',
    )
    parser.add_argument(
        '--controller-filter-coeff',
        type=float,
        default=None,
        help='Optional panda_py joint-position controller filter coefficient.',
    )
    parser.add_argument('--device', default=None, help='Optional torch device override.')
    parser.add_argument('--log-interval', type=int, default=30, help='Step interval for runtime logging.')
    parser.add_argument(
        '--first-frame-max-pos-delta-mm',
        type=float,
        default=_DEFAULT_FIRST_FRAME_MAX_POS_DELTA_MM,
        help='Reject the first policy EE target if any xyz component differs from current EE pose by more than this value.',
    )
    parser.add_argument(
        '--first-frame-max-rot-delta-deg',
        type=float,
        default=_DEFAULT_FIRST_FRAME_MAX_ROT_DELTA_DEG,
        help='Reject the first policy EE target if any relative rotvec component differs from current EE pose by more than this value.',
    )
    parser.add_argument(
        '--max-step-pos-delta-mm',
        type=float,
        default=_DEFAULT_MAX_STEP_POS_DELTA_MM,
        help='Clamp each step xyz delta relative to current EE pose to this limit.',
    )
    parser.add_argument(
        '--max-step-rot-delta-deg',
        type=float,
        default=_DEFAULT_MAX_STEP_ROT_DELTA_DEG,
        help=(
            'Limit the magnitude of each step rotation delta, measured against prev_cmd -- the '
            'pose the policy action is defined relative to.'
        ),
    )
    parser.add_argument(
        '--max-leash-pos-delta-mm',
        type=float,
        default=_DEFAULT_MAX_LEASH_POS_DELTA_MM,
        help=(
            'Limit how far the command may run ahead of the measured EE pose. This bounds servo '
            'tracking lag, not policy motion, so it must stay well above the lag a healthy moving '
            'arm produces; size it from the recorded prev_cmd-vs-measured gap, not from per-step '
            'motion.'
        ),
    )
    parser.add_argument(
        '--max-leash-rot-delta-deg',
        type=float,
        default=_DEFAULT_MAX_LEASH_ROT_DELTA_DEG,
        help='Rotational counterpart of --max-leash-pos-delta-mm.',
    )
    parser.add_argument(
        '--record-config',
        type=str,
        default=None,
        help=(
            "The rig's record config, read for its robot.workspace_min/max fence. The driver clips "
            'every commanded pose to that box and reports the clipped pose back, so a rollout with '
            'a box of its own can stop short of where the demonstrations went without anything '
            'saying so. Name the config the rig records with; omit it only on a rig that has none.'
        ),
    )
    parser.add_argument(
        '--workspace-min',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help=(
            'Override the workspace fence lower corner, in the robot base frame. Requires '
            '--workspace-max, and replaces --record-config entirely rather than per axis.'
        ),
    )
    parser.add_argument(
        '--workspace-max',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Upper corner of --workspace-min.',
    )
    parser.add_argument(
        '--tactile-fallback',
        choices=_TACTILE_FALLBACK_CHOICES,
        default=None,
        help='Preview-only tactile fallback. baseline_idle injects no-contact tactile from baseline/mask assets.',
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
        help='Show the policy input camera frames in one OpenCV window with camera-name labels.',
    )
    parser.add_argument(
        '--no-camera-preview-window',
        dest='camera_preview_window',
        action='store_false',
        help=(
            'Turn the OpenCV preview window back off. The run_pick_place_* launchers enable it '
            'for most modes; a caller with no X display (the GUI gateway) needs a way to say so '
            'without reimplementing the launcher.'
        ),
    )
    parser.add_argument(
        '--move-to-das-start',
        dest='move_to_das_start',
        action='store_true',
        help=(
            'Move the arm to the DAS replay start joint configuration before inference. Off by '
            'default: those joint angles belong to the DAS rig, and the start pose is not cosmetic '
            '-- T_B_Ws is solved from the first observation against the dataset start pose, so it '
            'places the whole trajectory in the workspace. Homing to a pose belonging to a '
            'different rig silently offsets every target. Home with the pose the episodes were '
            'recorded from instead: --robot-init-state, or the launcher own homing step.'
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
    parser.add_argument(
        '--dataset-start-gripper-tolerance',
        type=float,
        default=_DEFAULT_DATASET_START_GRIPPER_TOLERANCE,
        help='Absolute normalized gripper tolerance used for startup diagnostics and optional auto-alignment.',
    )
    parser.add_argument(
        '--robot-init-state',
        default=None,
        help=(
            'Optional robot startup state before inference. Accepts a YAML/JSON file path, '
            'an inline YAML/JSON object, or shorthand like joints=7 comma-separated radians '
            'or ee_xyzquat=x,y,z,qx,qy,qz,qw. File/object examples: '
            '{type: joints, joints_rad: [...], gripper: 0.5} or '
            '{type: ee_xyzquat, xyzquat: [x,y,z,qx,qy,qz,qw], gripper: 0.5}.'
        ),
    )
    parser.add_argument(
        '--interactive-rollouts',
        action='store_true',
        help='Wait for a start/stop/quit signal between rollouts, from a keyboard on a TTY or one command per line on a stdin pipe.',
    )
    parser.add_argument(
        '--preview-jpeg-dir',
        type=Path,
        default=None,
        help=(
            'Publish the frames the policy is being fed as <camera>.jpg in this directory, for '
            'a viewer that is not at the machine. Independent of --camera-preview-window, which '
            'needs a local X display.'
        ),
    )
    parser.add_argument(
        '--preview-jpeg-fps',
        type=float,
        default=5.0,
        help='Upper bound on preview JPEG writes per second (default 5).',
    )
    parser.add_argument(
        '--live-frame-interval',
        type=int,
        default=0,
        help=(
            'Print one live_frame= line every N steps: joint angles, gripper and the commanded '
            'versus measured end-effector position, for a browser to draw the arm while the '
            'rollout is running. 0 (default) prints nothing. The GUI passes 1; a terminal '
            'operator watching the arm itself has no use for it.'
        ),
    )
    parser.add_argument(
        '--rollout-trace-dir',
        type=str,
        default='outputs/rollout_traces',
        help=(
            'Directory for one CSV per interactive rollout holding its per-step end-effector '
            'position and gripper command. On by default because a rollout cannot be repeated: '
            'the object placement it was run against is destroyed by the run itself. Pass an '
            'empty string to write nothing; the summary on the rollout end marker is printed '
            'either way.'
        ),
    )
    parser.add_argument('--rollout-start-key', default='s', help='Interactive key to start a rollout.')
    parser.add_argument('--rollout-stop-key', default='x', help='Interactive key to stop the current rollout.')
    parser.add_argument(
        '--rollout-home-key',
        default='h',
        help='Interactive key to move the arm back to its start pose between rollouts.',
    )
    parser.add_argument('--rollout-quit-key', default='q', help='Interactive key to quit inference.')
    parser.add_argument(
        '--dagger-takeover',
        action='store_true',
        help=(
            'Let the operator take the arm over mid-rollout with a SpaceMouse and hand it back, '
            'so a correction is applied to the state the policy actually walked into. Moving the '
            'device takes the arm; the policy resumes once it goes quiet. Off by '
            'default: it opens a second action source onto a loop that is moving a real arm.'
        ),
    )
    parser.add_argument(
        '--dagger-takeover-release-after-s',
        type=float,
        default=1.0,
        help=(
            'Hand the arm back to the policy once the SpaceMouse has been quiet this long '
            '(default 1 s). 0 disables automatic takeover, leaving only the takeover key.'
        ),
    )
    parser.add_argument(
        '--rollout-takeover-key',
        default='t',
        help=(
            'Interactive key that latches DAgger takeover on, for an operator who wants the arm '
            'held still without moving the device. Only bound with --dagger-takeover.'
        ),
    )
    parser.add_argument(
        '--dagger-spacemouse-device-id',
        type=int,
        default=0,
        help='SpaceMouse device index used for takeover (default 0).',
    )
    parser.add_argument(
        '--dagger-translation-scale',
        type=float,
        default=None,
        help="Override the SpaceMouse translation scale during takeover. Defaults to the recorder's value.",
    )
    parser.add_argument(
        '--dagger-rotation-scale',
        type=float,
        default=None,
        help="Override the SpaceMouse rotation scale during takeover. Defaults to the recorder's value.",
    )
    parser.add_argument(
        '--dagger-dataset-root',
        type=Path,
        default=None,
        help=(
            'Write the steps the operator drove to this LeRobot dataset, one episode per '
            'correction, each frame flagged is_intervention. Created if absent, extended if not. '
            'Without it a takeover steers the arm and leaves no training sample behind.'
        ),
    )
    parser.add_argument(
        '--dagger-dataset-repo-id',
        type=str,
        default=None,
        help='repo_id for --dagger-dataset-root. Defaults to the directory name.',
    )
    parser.add_argument(
        '--dagger-max-buffered-frames',
        type=int,
        default=DEFAULT_MAX_BUFFERED_FRAMES,
        help=(
            'Expert frames held in memory per rollout before the rest are dropped and counted '
            f'(default {DEFAULT_MAX_BUFFERED_FRAMES}, about 15 s of correction at 30 Hz). They are '
            'held rather than written as they happen because save_episode encodes video, and the '
            'end of a correction is when the policy is about to resume driving a real arm.'
        ),
    )
    parser.add_argument(
        '--dagger-min-span-frames',
        type=int,
        default=2,
        help='Corrections shorter than this many frames are a bumped device, not a demonstration (default 2).',
    )
    parser.add_argument(
        '--mujoco-viewer',
        action='store_true',
        help='Open a passive MuJoCo viewer that mirrors real FR3 joints and overlays policy EE targets.',
    )
    parser.add_argument(
        '--mujoco-model',
        type=Path,
        default=None,
        help=(
            'Optional MuJoCo XML model for the viewer. Defaults to the DAS XML for --gripper-backend=das '
            'and the Pika XML for --gripper-backend=pika.'
        ),
    )
    parser.add_argument(
        '--mujoco-max-chunk-points',
        type=int,
        default=64,
        help='Maximum policy action-chunk target points to draw in the MuJoCo viewer.',
    )
    parser.set_defaults(
        move_to_das_start=False,
        align_gripper_to_dataset_start=True,
        corenetic_release_mode_on_disconnect=True,
    )
    return parser.parse_args(argv)


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _parse_optional_float_tuple(
    value: str | None,
    *,
    expected_len: int,
    argument_name: str,
) -> list[float] | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    parts = [part.strip() for part in raw_value.split(',') if part.strip()]
    values = [float(part) for part in parts]
    if len(values) != expected_len:
        raise ValueError(f'{argument_name} expects {expected_len} comma-separated values, got {len(values)}.')
    return values


def resolve_pretrained_model_dir(checkpoint_path: str | Path) -> Path:
    checkpoint_dir = _resolve_repo_path(checkpoint_path)
    pretrained_dir = checkpoint_dir / 'pretrained_model'
    if pretrained_dir.is_dir():
        return pretrained_dir
    if (checkpoint_dir / 'config.json').is_file():
        return checkpoint_dir
    raise FileNotFoundError(f'Could not find pretrained_model/config.json under {checkpoint_dir}')


def load_train_config(pretrained_dir: Path) -> TrainPipelineConfig:
    return TrainPipelineConfig.from_pretrained(pretrained_dir)


def resolve_dataset_root(pretrained_dir: Path, train_cfg: TrainPipelineConfig, dataset_root: str | None) -> Path:
    root_value = dataset_root or train_cfg.dataset.root
    if root_value is None:
        raise ValueError(
            f"No dataset root resolved from {pretrained_dir / 'train_config.json'}. Pass --dataset-root explicitly."
        )
    return _resolve_repo_path(root_value)


def _feature_has_ee_pose(info: dict[str, Any], state_key: str) -> bool:
    names = info.get('features', {}).get(state_key, {}).get('names')
    if not isinstance(names, list):
        return False
    required_names = {'ee.x', 'ee.y', 'ee.z', 'ee.qx', 'ee.qy', 'ee.qz', 'ee.qw'}
    return required_names.issubset({str(name) for name in names})


def _resolve_existing_dataset_root(path_value: str | Path) -> Path:
    candidate = _resolve_repo_path(path_value)
    if candidate.exists():
        return candidate

    basename = Path(path_value).name
    for root in (
        _REPO_ROOT / 'outputs' / 'datasets',
        Path('/home/corenetic/Code/lerobot/outputs/datasets'),
        Path('/home/corenetic/Code/lerobot/data'),
    ):
        replacement = root / basename
        if replacement.exists():
            print(
                '[WARN] alignment_dataset_root_path_remapped='
                f'from={candidate} to={replacement.resolve()}'
            )
            return replacement.resolve()
    return candidate


def _infer_source_state_key(manifest: dict[str, Any], source_info: dict[str, Any]) -> str:
    for key in ('source_state_key', 'alignment_state_key'):
        state_key = manifest.get(key)
        if isinstance(state_key, str) and _feature_has_ee_pose(source_info, state_key):
            return state_key

    action_key = manifest.get('action_key') or manifest.get('action_source_key')
    if isinstance(action_key, str) and action_key.startswith('action.'):
        suffix = action_key.split('.', 1)[1]
        state_key = f'observation.state.{suffix}'
        if _feature_has_ee_pose(source_info, state_key):
            return state_key

    if _feature_has_ee_pose(source_info, 'observation.state'):
        return 'observation.state'

    for state_key in sorted(source_info.get('features', {})):
        if state_key.startswith('observation.state') and _feature_has_ee_pose(source_info, state_key):
            return str(state_key)

    raise KeyError('Could not find an EE-pose observation.state feature in source dataset metadata.')


def resolve_alignment_dataset_root_and_state_key(dataset_root: Path) -> tuple[Path, str]:
    """Resolve a dataset root that contains EE state for start-frame alignment.

    Image-only policy views intentionally omit observation.state from policy metadata.
    Their manifest records the source dataset root, which still contains the EE
    state needed to align dataset-world actions to the live robot base frame.
    """
    dataset_root = _resolve_repo_path(dataset_root)
    try:
        info = _load_dataset_info(dataset_root)
    except FileNotFoundError:
        return dataset_root, 'observation.state'
    if _feature_has_ee_pose(info, 'observation.state'):
        return dataset_root, 'observation.state'

    manifest_path = dataset_root / 'meta' / 'il_view_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(
            f"{dataset_root} has no EE-pose observation.state and no {manifest_path}; "
            "cannot estimate dataset start pose for real-robot inference."
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source_dataset_root = manifest.get('source_dataset_root')
    if not source_dataset_root:
        raise ValueError(f"{manifest_path} does not define source_dataset_root.")
    source_root = _resolve_existing_dataset_root(source_dataset_root)
    source_info = _load_dataset_info(source_root)
    return source_root, _infer_source_state_key(manifest, source_info)


def _load_crop_source_frame_shapes(
    manifest: dict[str, Any], manifest_path: Path, feature_keys: list[str]
) -> dict[str, tuple[int, int]]:
    """The (H, W) each crop was drawn on, read off the recording the view was built from."""
    roots = manifest.get('source_dataset_roots') or []
    if not roots and manifest.get('source_dataset_root'):
        roots = [manifest['source_dataset_root']]
    for root_value in roots:
        try:
            info = _load_dataset_info(_resolve_existing_dataset_root(root_value))
        except (FileNotFoundError, ValueError, OSError):
            continue
        features = info.get('features') or {}
        shapes: dict[str, tuple[int, int]] = {}
        for feature_key in feature_keys:
            shape = (features.get(feature_key) or {}).get('shape')
            if isinstance(shape, (list, tuple)) and len(shape) == 3:
                shapes[feature_key] = (int(shape[0]), int(shape[1]))
        if len(shapes) == len(feature_keys):
            return shapes
    # Not fatal: the bounds check in `apply_camera_crop` still rejects a crop that cannot fit the
    # live frame. It only stops catching a camera opened *larger* than the recording, where the
    # rectangle still fits and quietly frames a different part of the scene.
    print(
        '[WARN] camera_crop_source_frame=unknown '
        f'reason=source_dataset_unreadable manifest={manifest_path}'
    )
    return {}


def load_camera_crop_specs(dataset_root: Path) -> tuple[dict[str, list[int]], dict[str, tuple[int, int]]]:
    """The crop a training view baked into its videos, keyed by image feature.

    A view built with a camera crop stores the cropped pixels in its video and nowhere else --
    its own feature shape *is* the crop's -- so a checkpoint trained on it asks for an image
    that only exists inside that rectangle. The rollout reads whole sensor frames, so unless the
    crop is replayed here it is a training-only transform: the policy gets the entire scene
    squeezed into the rectangle's shape where it was trained on a window of it. Nothing raises,
    because both images carry the shape the policy asked for; the scene is simply scaled and
    shifted away from what the checkpoint saw, and the rollout reaches for the wrong place.

    Returns the crops plus the (H, W) of the recording each was drawn on, so a camera opened at
    another resolution is refused rather than cropped against the wrong pixel grid.
    """
    dataset_root = _resolve_repo_path(dataset_root)
    manifest_path = dataset_root / 'meta' / 'il_view_manifest.json'
    if not manifest_path.is_file():
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw_specs = manifest.get('camera_crop_specs') or {}
    if not isinstance(raw_specs, dict):
        raise ValueError(f'{manifest_path} has a non-object camera_crop_specs: {raw_specs!r}')
    crops: dict[str, list[int]] = {}
    for feature_key, value in raw_specs.items():
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(
                f'{manifest_path} camera_crop_specs[{feature_key!r}] must be [x, y, w, h], got {value!r}'
            )
        try:
            x, y, w, h = (int(part) for part in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{manifest_path} camera_crop_specs[{feature_key!r}] must contain integers, got {value!r}'
            ) from exc
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(
                f'{manifest_path} camera_crop_specs[{feature_key!r}] must have non-negative x/y and '
                f'positive w/h, got {value!r}'
            )
        crops[str(feature_key)] = [x, y, w, h]
    if not crops:
        return {}, {}
    return crops, _load_crop_source_frame_shapes(manifest, manifest_path, sorted(crops))


def move_to_das_start_if_requested(*, robot_ip: str, enabled: bool) -> None:
    if not enabled:
        return

    import panda_py

    print(f'[INFO] 连接 panda_py ({robot_ip})，移动到 DAS 起始关节角...')
    print(f'[INFO] 目标关节角（rad）: {_DAS_START_JOINTS_RAD.tolist()}')
    panda = panda_py.Panda(robot_ip)
    panda.move_to_joint_position(_DAS_START_JOINTS_RAD.tolist())
    del panda
    time.sleep(0.5)
    print('[INFO] 已到达 DAS 起始关节角')


def _parse_numeric_sequence(value: Any, *, expected_len: int, label: str) -> list[float]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('['):
            value = yaml.safe_load(stripped)
        else:
            value = [item.strip() for item in stripped.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{label} must be a list or comma-separated string.')
    values = [float(item) for item in value]
    if len(values) != expected_len:
        raise ValueError(f'{label} must contain {expected_len} values, got {len(values)}.')
    return values


def _load_robot_init_state_payload(spec: str | None) -> dict[str, Any] | None:
    if spec in (None, ''):
        return None

    spec = str(spec).strip()
    path: Path | None = None
    if len(spec) < 512 and '\n' not in spec and not spec.lstrip().startswith(('{', '[')):
        path = _resolve_repo_path(spec)
    if path is not None and path.is_file():
        with path.open('r', encoding='utf-8') as f:
            payload = yaml.safe_load(f) or {}
        if isinstance(payload, dict) and 'robot_init_state' in payload:
            payload = payload['robot_init_state']
        if not isinstance(payload, dict):
            raise ValueError(f'robot-init-state file {path} must contain a mapping/object.')
        return payload

    for prefix in ('joints=', 'joints:', 'joint_rad=', 'joint_rad:'):
        if spec.startswith(prefix):
            return {'type': 'joints', 'joints_rad': spec[len(prefix) :]}
    for prefix in ('ee_xyzquat=', 'ee_xyzquat:', 'xyzquat=', 'xyzquat:'):
        if spec.startswith(prefix):
            return {'type': 'ee_xyzquat', 'xyzquat': spec[len(prefix) :]}
    for prefix in ('ee_xyzrotvec=', 'ee_xyzrotvec:', 'xyzrotvec=', 'xyzrotvec:'):
        if spec.startswith(prefix):
            return {'type': 'ee_xyzrotvec', 'xyzrotvec': spec[len(prefix) :]}

    payload = yaml.safe_load(spec)
    if not isinstance(payload, dict):
        raise ValueError(
            'robot-init-state must be a file path, mapping/object, or shorthand like '
            'joints=q1,...,q7 / ee_xyzquat=x,y,z,qx,qy,qz,qw.'
        )
    return payload


def parse_robot_init_state(spec: str | None) -> dict[str, Any] | None:
    payload = _load_robot_init_state_payload(spec)
    if payload is None:
        return None

    state_type = str(payload.get('type', payload.get('kind', ''))).strip().lower()
    gripper = payload.get('gripper', payload.get('gripper_pos'))
    parsed: dict[str, Any]

    if state_type in {'joint', 'joints', 'joint_positions', 'joint_rad', 'joints_rad'} or 'joints_rad' in payload:
        joints = _parse_numeric_sequence(
            payload.get('joints_rad', payload.get('joints', payload.get('values'))),
            expected_len=7,
            label='robot init joints_rad',
        )
        parsed = {'type': 'joints', 'joints_rad': joints}
    elif state_type in {'ee_pose', 'ee_xyzquat', 'xyzquat'} or 'xyzquat' in payload:
        xyzquat = _parse_numeric_sequence(
            payload.get('xyzquat', payload.get('ee_xyzquat', payload.get('values'))),
            expected_len=7,
            label='robot init xyzquat',
        )
        parsed = {'type': 'ee_xyzquat', 'xyzquat': xyzquat}
    elif state_type in {'ee_xyzrotvec', 'xyzrotvec', 'ee_rotvec'} or 'xyzrotvec' in payload:
        xyzrotvec = _parse_numeric_sequence(
            payload.get('xyzrotvec', payload.get('ee_xyzrotvec', payload.get('values'))),
            expected_len=6,
            label='robot init xyzrotvec',
        )
        parsed = {'type': 'ee_xyzrotvec', 'xyzrotvec': xyzrotvec}
    else:
        raise ValueError(
            "robot-init-state type must be one of 'joints', 'ee_xyzquat', or 'ee_xyzrotvec'."
        )

    if gripper is not None:
        parsed['gripper'] = float(gripper)
    parsed['timeout_s'] = float(payload.get('timeout_s', 20.0))
    parsed['joint_tolerance_rad'] = float(payload.get('joint_tolerance_rad', 0.01))
    parsed['ee_pos_tolerance_m'] = float(payload.get('ee_pos_tolerance_m', 0.005))
    parsed['ee_rot_tolerance_deg'] = float(payload.get('ee_rot_tolerance_deg', 2.0))
    parsed['gripper_tolerance'] = float(payload.get('gripper_tolerance', 0.02))
    return parsed


def _observation_joint_positions(observation: RobotObservation) -> np.ndarray:
    return np.asarray([float(observation[f'joint_{idx}.pos']) for idx in range(1, 8)], dtype=np.float64)


def _observation_pose_rotvec(observation: RobotObservation) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray([float(observation[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
    rotvec = np.asarray([float(observation[key]) for key in ('ee.wx', 'ee.wy', 'ee.wz')], dtype=np.float64)
    return position, rotvec


def _rotation_error_rad(current_rotvec: np.ndarray, target_rotvec: np.ndarray) -> float:
    current = Rotation.from_rotvec(current_rotvec)
    target = Rotation.from_rotvec(target_rotvec)
    return float(np.linalg.norm((target * current.inv()).as_rotvec()))


def _wait_until_robot_init_reached(
    robot: Any,
    init_state: dict[str, Any],
    *,
    target_joints_rad: np.ndarray | None = None,
    target_position: np.ndarray | None = None,
    target_rotvec: np.ndarray | None = None,
    target_gripper: float | None = None,
) -> None:
    deadline = time.perf_counter() + float(init_state['timeout_s'])
    last_status = ''
    while time.perf_counter() < deadline:
        observation = robot.get_observation(include_cameras=False)
        checks: list[bool] = []
        status_parts: list[str] = []
        if target_joints_rad is not None:
            joint_error = float(np.max(np.abs(_observation_joint_positions(observation) - target_joints_rad)))
            checks.append(joint_error <= float(init_state['joint_tolerance_rad']))
            status_parts.append(f'joint_max_err_rad={joint_error:.4f}')
        if target_position is not None and target_rotvec is not None:
            current_position, current_rotvec = _observation_pose_rotvec(observation)
            pos_error_m = float(np.linalg.norm(current_position - target_position))
            rot_error_deg = float(np.rad2deg(_rotation_error_rad(current_rotvec, target_rotvec)))
            checks.append(pos_error_m <= float(init_state['ee_pos_tolerance_m']))
            checks.append(rot_error_deg <= float(init_state['ee_rot_tolerance_deg']))
            status_parts.append(f'ee_pos_err_mm={pos_error_m * 1000.0:.2f}')
            status_parts.append(f'ee_rot_err_deg={rot_error_deg:.2f}')
        if target_gripper is not None:
            gripper_error = float(abs(float(observation['gripper.pos']) - target_gripper))
            checks.append(gripper_error <= float(init_state['gripper_tolerance']))
            status_parts.append(f'gripper_err={gripper_error:.3f}')
        last_status = ' '.join(status_parts)
        if checks and all(checks):
            print(f'[INFO] robot_init_state reached: {last_status}')
            return
        precise_sleep(0.05)
    raise TimeoutError(f'Timed out waiting for robot_init_state: {last_status}')


def move_to_robot_init_state_if_requested(robot: Any, init_state: dict[str, Any] | None) -> None:
    if init_state is None:
        return

    target_gripper = float(init_state['gripper']) if 'gripper' in init_state else None
    if init_state['type'] == 'joints':
        target_joints_rad = np.asarray(init_state['joints_rad'], dtype=np.float64)
        print(
            '[INFO] moving_to_robot_init_state='
            f"type=joints joints_rad={target_joints_rad.tolist()} gripper={target_gripper}"
        )
        robot.send_joint_positions(target_joints_rad, gripper_pos=target_gripper)
        _wait_until_robot_init_reached(
            robot,
            init_state,
            target_joints_rad=target_joints_rad,
            target_gripper=target_gripper,
        )
        return

    if init_state['type'] == 'ee_xyzquat':
        xyzquat = np.asarray(init_state['xyzquat'], dtype=np.float64)
        target_position = xyzquat[:3]
        target_rotvec = Rotation.from_quat(xyzquat[3:7]).as_rotvec()
    else:
        xyzrotvec = np.asarray(init_state['xyzrotvec'], dtype=np.float64)
        target_position = xyzrotvec[:3]
        target_rotvec = xyzrotvec[3:6]

    current_observation = robot.get_observation(include_cameras=False)
    if target_gripper is None:
        target_gripper = float(current_observation['gripper.pos'])
    command = {
        'ee.x': float(target_position[0]),
        'ee.y': float(target_position[1]),
        'ee.z': float(target_position[2]),
        'ee.wx': float(target_rotvec[0]),
        'ee.wy': float(target_rotvec[1]),
        'ee.wz': float(target_rotvec[2]),
        'gripper.pos': float(target_gripper),
    }
    print(
        '[INFO] moving_to_robot_init_state='
        f"type={init_state['type']} xyz={target_position.tolist()} rotvec={target_rotvec.tolist()} "
        f'gripper={target_gripper}'
    )
    robot.send_action(command)
    _wait_until_robot_init_reached(
        robot,
        init_state,
        target_position=target_position,
        target_rotvec=target_rotvec,
        target_gripper=target_gripper,
    )


def home_arm_to_start_pose(robot: Any) -> bool:
    """Put the arm back at the pose the demonstrations started from, between rollouts.

    Interactive mode homes the arm exactly once, in the launcher, before this process exists.
    Every rollout after the first therefore begins wherever the previous one stopped -- and the
    dataset frame is anchored to the start pose (T_B_Ws is solved from the first observation
    against the dataset's start distribution), so a rollout launched from a displaced arm is
    either refused by the first-frame gate or, worse, admitted at the edge of it.

    It has to happen in this process rather than by running fr3_move_to_start_runtime.py again:
    the FR3 accepts one libfranka client and this process is holding it. That script opens its
    own `Panda()`, which is why the launcher runs it *before* exec'ing this runtime and never
    alongside it. `FrankaResearch3.move_to_start()` goes to the same joint target -- the config's
    start pose, held equal to the XML `home` keyframe by
    tests/robots/test_fr3_home_keyframe_contract.py -- and, unlike a second libfranka session,
    it stops and restarts the OTG loop around the move instead of fighting it.

    The gripper is deliberately left where it is. Homing with a peg still in the fingers is
    wrong for the next rollout, but opening them would drop whatever is held from wherever the
    arm stopped, and only the operator standing at the rig can say which of those is worse. The
    position is logged either way, so the choice is on the record.

    Returns whether the arm actually arrived. A failure is reported and handed back rather than
    raised: tearing down a loaded policy -- ten seconds of arm motion paid for with a minute of
    reload and ten gigabytes of VRAM -- is a much larger action than the one that failed, and
    the first-frame gate still stands between a mis-posed arm and a rollout.
    """
    try:
        gripper_pos = float(robot.get_observation(include_cameras=False)['gripper.pos'])
        gripper_text = f'{gripper_pos:.3f}'
    except Exception:
        gripper_text = 'unknown'
    print(f'[INFO] interactive_homing=start gripper_pos={gripper_text} (gripper is left as it is)')
    try:
        robot.move_to_start()
    except Exception as exc:
        print(f'[WARN] interactive_homing=failed details={exc}')
        return False
    print('[INFO] interactive_homing=done')
    return True


def resolve_mujoco_model_path(gripper_backend: str, model_path: str | Path | None) -> Path:
    if model_path is not None:
        return _resolve_repo_path(model_path)
    return _DAS_XML if gripper_backend == 'das' else _PIKA_XML


def resolve_robot_tool_model(gripper_backend: str, urdf_path: str | Path | None, target_frame_name: str | None) -> tuple[Path, str]:
    if urdf_path is not None:
        resolved_urdf = _resolve_repo_path(urdf_path)
    elif gripper_backend == 'das':
        resolved_urdf = _DAS_URDF
    else:
        resolved_urdf = _PIKA_URDF

    if target_frame_name is not None:
        resolved_target_frame = str(target_frame_name)
    elif gripper_backend == 'das':
        resolved_target_frame = 'das_gripper_ee'
    else:
        resolved_target_frame = 'pika_gripper_ee'

    if gripper_backend == 'corenetic' and urdf_path is None:
        print(
            '[WARN] Corenetic gripper is using the Pika FR3 URDF/TCP fallback for IK. '
            'Pass --robot-urdf-path and --target-frame-name when you have the calibrated Corenetic tool model.'
        )
    return resolved_urdf, resolved_target_frame


class FR3InferenceMujocoVisualizer:
    def __init__(self, *, model_path: str | Path, max_chunk_points: int = 64):
        self.model_path = _resolve_repo_path(model_path)
        self.max_chunk_points = max(1, int(max_chunk_points))
        self._mujoco = None
        self._viewer = None
        self._model = None
        self._data = None
        self._joint_qpos_addresses: list[int] = []

    def start(self) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._data = mujoco.MjData(self._model)
        self._joint_qpos_addresses = []
        for joint_name in _JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model {self.model_path}")
            self._joint_qpos_addresses.append(int(self._model.jnt_qposadr[joint_id]))
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        print(f'[INFO] mujoco_viewer=enabled model={self.model_path}')

    @property
    def is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def _sync_joint_state(self, robot_observation: RobotObservation) -> None:
        assert self._mujoco is not None and self._model is not None and self._data is not None
        joint_positions = _observation_joint_positions(robot_observation)
        for qpos_address, joint_position in zip(self._joint_qpos_addresses, joint_positions, strict=True):
            self._data.qpos[qpos_address] = float(joint_position)
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)

    def _add_box(self, scene: Any, pose: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
        assert self._mujoco is not None
        if scene.ngeom >= scene.maxgeom:
            return
        self._mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=self._mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array([0.018, 0.018, 0.018], dtype=np.float64),
            pos=np.asarray(pose[:3, 3], dtype=np.float64),
            mat=np.asarray(pose[:3, :3], dtype=np.float64).reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def _add_sphere(self, scene: Any, point: np.ndarray, rgba: np.ndarray, radius: float = 0.006) -> None:
        assert self._mujoco is not None
        if scene.ngeom >= scene.maxgeom:
            return
        self._mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=self._mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(point, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def _add_capsule(self, scene: Any, start: np.ndarray, end: np.ndarray, rgba: np.ndarray) -> None:
        assert self._mujoco is not None
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        self._mujoco.mjv_initGeom(
            geom,
            type=self._mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.003, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        self._mujoco.mjv_connector(
            geom,
            self._mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.003,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1

    def _draw_chunk_trajectory(self, scene: Any, chunk_poses: list[np.ndarray]) -> None:
        if not chunk_poses:
            return
        available_points = max((scene.maxgeom - scene.ngeom + 1) // 2, 1)
        point_count = min(len(chunk_poses), self.max_chunk_points, available_points)
        if point_count <= 0:
            return
        if len(chunk_poses) > point_count:
            indices = np.linspace(0, len(chunk_poses) - 1, point_count).round().astype(int)
            poses = [chunk_poses[int(index)] for index in indices]
        else:
            poses = chunk_poses
        points = [np.asarray(pose[:3, 3], dtype=np.float64) for pose in poses]
        start_rgba = np.array([0.10, 0.55, 1.00, 0.78], dtype=np.float32)
        end_rgba = np.array([1.00, 0.15, 0.55, 0.92], dtype=np.float32)
        denom = max(len(points) - 1, 1)
        for idx, point in enumerate(points):
            alpha = float(idx / denom)
            rgba = (1.0 - alpha) * start_rgba + alpha * end_rgba
            self._add_sphere(scene, point, rgba)
            if idx > 0:
                self._add_capsule(scene, points[idx - 1], point, rgba)

    def update(
        self,
        *,
        robot_observation: RobotObservation,
        current_ee_pose: np.ndarray,
        target_ee_pose: np.ndarray,
        safe_target_ee_pose: np.ndarray | None = None,
        chunk_ee_poses: list[np.ndarray] | None,
    ) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        self._sync_joint_state(robot_observation)
        with self._viewer.lock():
            scene = self._viewer.user_scn
            scene.ngeom = 0
            self._add_box(scene, current_ee_pose, (1.00, 0.42, 0.12, 0.85))
            self._add_box(scene, target_ee_pose, (0.10, 0.85, 0.35, 0.85))
            if safe_target_ee_pose is not None:
                self._add_box(scene, safe_target_ee_pose, (1.00, 0.92, 0.10, 0.90))
            self._draw_chunk_trajectory(scene, chunk_ee_poses or [])
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass




def _coerce_opencv_index_or_path(value: Any) -> int | Path:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if stripped.isdigit() else Path(stripped)
    if isinstance(value, Path):
        return value
    raise ValueError(f'Unsupported OpenCV camera device identifier: {value!r}')


def _coerce_opencv_backend(value: Any) -> Cv2Backends:
    if value is None:
        return _DEFAULT_OPENCV_BACKEND
    if isinstance(value, Cv2Backends):
        return value
    if isinstance(value, int):
        return Cv2Backends(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return Cv2Backends(int(stripped))
        return Cv2Backends[stripped.upper()]
    raise ValueError(f'Unsupported OpenCV backend identifier: {value!r}')


def _coerce_cv2_rotation(value: Any) -> Cv2Rotation:
    if isinstance(value, Cv2Rotation):
        return value
    if value is None:
        return Cv2Rotation.NO_ROTATION
    if isinstance(value, int):
        return Cv2Rotation(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return Cv2Rotation.NO_ROTATION
        if stripped.lstrip('-').isdigit():
            return Cv2Rotation(int(stripped))
        aliases = {
            'none': Cv2Rotation.NO_ROTATION,
            'no_rotation': Cv2Rotation.NO_ROTATION,
            'rotate_90': Cv2Rotation.ROTATE_90,
            '90': Cv2Rotation.ROTATE_90,
            'rotate_180': Cv2Rotation.ROTATE_180,
            '180': Cv2Rotation.ROTATE_180,
            'rotate_270': Cv2Rotation.ROTATE_270,
            '-90': Cv2Rotation.ROTATE_270,
        }
        normalized = stripped.lower()
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError(f'Unsupported cv2 rotation identifier: {value!r}')


def _coerce_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def load_camera_configs(
    camera_config_path: str | Path,
) -> dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig | Gmsl2CameraConfig]:
    config_path = _resolve_repo_path(camera_config_path)
    with config_path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    camera_entries = raw.get('robot', {}).get('cameras', {})
    if not camera_entries:
        raise ValueError(f'No robot.cameras entries found in {config_path}')

    camera_configs: dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig | Gmsl2CameraConfig] = {}
    for camera_name, cfg in camera_entries.items():
        camera_type = cfg.get('type')
        if camera_type == 'intelrealsense':
            camera_configs[camera_name] = RealSenseCameraConfig(
                serial_number_or_name=str(cfg['serial_number_or_name']),
                width=int(cfg['width']),
                height=int(cfg['height']),
                fps=int(cfg['fps']),
            )
            continue
        if camera_type == 'opencv':
            image_shape = cfg.get('image_shape')
            width = cfg.get('width')
            height = cfg.get('height')
            if image_shape is not None:
                if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 2:
                    raise ValueError(f"opencv camera '{camera_name}' must use image_shape=[height, width]")
                height, width = int(image_shape[0]), int(image_shape[1])
            if width is None or height is None:
                raise ValueError(f"opencv camera '{camera_name}' requires width/height or image_shape")

            device_id = cfg.get('device_id', cfg.get('index_or_path'))
            if device_id is None:
                raise ValueError(f"opencv camera '{camera_name}' requires device_id or index_or_path")

            camera_configs[camera_name] = OpenCVCameraConfig(
                index_or_path=_coerce_opencv_index_or_path(device_id),
                width=int(width),
                height=int(height),
                fps=int(cfg['fps']),
                color_mode=cfg.get('color_mode', ColorMode.RGB),
                warmup_s=int(cfg.get('warmup_s', 1)),
                fourcc=str(cfg.get('fourcc', _DEFAULT_OPENCV_FOURCC)),
                backend=_coerce_opencv_backend(cfg.get('backend', _DEFAULT_OPENCV_BACKEND)),
            )
            continue
        if camera_type == 'hikrobot':
            image_shape = cfg.get('image_shape')
            width = cfg.get('width')
            height = cfg.get('height')
            if image_shape is not None:
                if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 2:
                    raise ValueError(f"hikrobot camera '{camera_name}' must use image_shape=[height, width]")
                height, width = int(image_shape[0]), int(image_shape[1])
            if width is None or height is None:
                raise ValueError(f"hikrobot camera '{camera_name}' requires width/height or image_shape")

            camera_configs[camera_name] = HikrobotCameraConfig(
                serial=str(cfg['serial']) if cfg.get('serial') is not None else None,
                device_index=int(cfg['device_index']) if cfg.get('device_index') is not None else None,
                width=int(width),
                height=int(height),
                fps=int(cfg['fps']),
                color_mode=cfg.get('color_mode', ColorMode.BGR),
                warmup_s=int(cfg.get('warmup_s', 1)),
                transport_layer=str(cfg.get('transport_layer', 'usb')),
                exposure_us=float(cfg['exposure_us']) if cfg.get('exposure_us') is not None else None,
                gain_db=float(cfg['gain_db']) if cfg.get('gain_db') is not None else None,
                timeout_ms=int(cfg.get('timeout_ms', 1000)),
            )
            continue
        if camera_type == 'gmsl2':
            image_shape = cfg.get('image_shape')
            width = cfg.get('width')
            height = cfg.get('height')
            if image_shape is not None:
                if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 2:
                    raise ValueError(f"gmsl2 camera '{camera_name}' must use image_shape=[height, width]")
                height, width = int(image_shape[0]), int(image_shape[1])
            if width is None or height is None:
                raise ValueError(f"gmsl2 camera '{camera_name}' requires width/height or image_shape")
            sensor_id = cfg.get('sensor_id')
            device = cfg.get('device')
            camera_configs[camera_name] = Gmsl2CameraConfig(
                sensor_id=int(sensor_id) if sensor_id is not None else None,
                device=str(device) if device is not None else None,
                pipeline=str(cfg.get('pipeline', 'argus')),
                sensor_mode=int(cfg.get('sensor_mode', 0)),
                v4l2_pixel_format=str(cfg.get('v4l2_pixel_format', 'UYVY')),
                bayer_format=str(cfg.get('bayer_format', 'grbg10le')),
                sync_role=str(cfg.get('sync_role', 'auto')),
                trig_pin=_coerce_int(cfg.get('trig_pin', 0x00020007)),
                apply_sync_at_connect=bool(cfg.get('apply_sync_at_connect', True)),
                exposure_us=int(cfg['exposure_us']) if cfg.get('exposure_us') is not None else None,
                gain=int(cfg['gain']) if cfg.get('gain') is not None else None,
                width=int(width),
                height=int(height),
                fps=int(cfg['fps']),
                color_mode=cfg.get('color_mode', ColorMode.BGR),
                rotation=_coerce_cv2_rotation(cfg.get('rotation', Cv2Rotation.NO_ROTATION)),
                warmup_s=int(cfg.get('warmup_s', 2)),
                timeout_ms=int(cfg.get('timeout_ms', 2000)),
            )
            continue
        raise ValueError(f"Unsupported camera type '{camera_type}' in {config_path} for {camera_name}")
    return camera_configs


def load_dataset_metadata(dataset_root: Path, repo_id: str) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)


def extract_feature_names(feature_entry: dict[str, Any], default_names: list[str]) -> list[str]:
    names = feature_entry.get('names')
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        for key in ('motors', 'dimensions', 'axes'):
            if isinstance(names.get(key), list):
                return [str(name) for name in names[key]]
    return list(default_names)


def extract_required_image_keys(input_features: dict[str, Any]) -> list[str]:
    return [
        feature_key[len(_OBS_IMAGES_PREFIX) :]
        for feature_key, feature in input_features.items()
        if feature_key.startswith(_OBS_IMAGES_PREFIX) and feature.type == FeatureType.VISUAL
    ]


def extract_required_tactile_keys(input_features: dict[str, Any]) -> list[str]:
    return [
        feature_key
        for feature_key, feature in input_features.items()
        if feature_key.startswith('observation.tactile.') and feature.type == FeatureType.STATE
    ]


def validate_camera_keys(*, required_image_keys: list[str], available_camera_keys: list[str]) -> None:
    missing_camera_keys = sorted(set(required_image_keys).difference(available_camera_keys))
    if missing_camera_keys:
        raise ValueError(
            'camera-config is missing policy-required cameras '
            f'{missing_camera_keys}; available cameras: {sorted(available_camera_keys)}'
        )



def _load_tactile_valid_mask(mask_path: str | Path) -> np.ndarray:
    path = _resolve_repo_path(mask_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    mask = np.asarray(payload['mask'], dtype=bool)
    if mask.shape != (50, 10):
        raise ValueError(f'Expected tactile valid mask shape (50, 10), got {mask.shape} from {path}')
    return mask


def _load_tactile_baseline_side(
    baseline_path: str | Path,
    side: str,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    path = _resolve_repo_path(baseline_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('encoding') == 'mask_fill':
        if valid_mask is None:
            raise ValueError(f'valid_mask is required to decode mask_fill tactile baseline from {path}')
        try:
            side_payload = payload['sides'][side]
            valid_value = float(side_payload['valid_value'])
            invalid_value = float(side_payload['invalid_value'])
        except Exception as exc:
            raise ValueError(f'Could not load tactile baseline side={side!r} from {path}') from exc
        baseline = np.full(valid_mask.shape, invalid_value, dtype=np.float32)
        baseline[valid_mask.astype(bool)] = valid_value
        return baseline
    try:
        values = payload['data'][0]['tactiles'][side]
    except Exception as exc:
        raise ValueError(f'Could not load tactile baseline side={side!r} from {path}') from exc
    baseline = np.asarray(values, dtype=np.float32)
    if baseline.size != 500:
        raise ValueError(f'Expected 500 tactile baseline values for side={side!r}, got {baseline.size} from {path}')
    return baseline.reshape(50, 10)


def build_tactile_fallback_observation(fallback_mode: str | None) -> dict[str, np.ndarray] | None:
    if fallback_mode is None:
        return None
    if fallback_mode != 'baseline_idle':
        raise ValueError(f'Unsupported tactile fallback mode: {fallback_mode}')

    valid_mask = _load_tactile_valid_mask(_DEFAULT_TACTILE_VALID_MASK_PATH)
    left_raw = _load_tactile_baseline_side(_DEFAULT_TACTILE_BASELINE_PATH, 'left', valid_mask)
    right_raw = _load_tactile_baseline_side(_DEFAULT_TACTILE_BASELINE_PATH, 'right', valid_mask)
    zeros = np.zeros_like(left_raw, dtype=np.float32)
    return {
        'observation.tactile.left_raw': left_raw.astype(np.float32),
        'observation.tactile.right_raw': right_raw.astype(np.float32),
        'observation.tactile.valid_mask': valid_mask.astype(np.float32),
        'observation.tactile.left_clean': zeros.copy(),
        'observation.tactile.right_clean': zeros.copy(),
    }


def _gripper_feature_unit(feature_name: str | None) -> str | None:
    if not feature_name:
        return None
    name = str(feature_name).lower()
    if name.endswith('.width_mm') or 'width_mm' in name:
        return 'mm'
    if name.endswith('.distance_m') or name.endswith('.width_m') or 'distance_m' in name or 'width_m' in name:
        return 'm'
    if name in {
        'gripper.pos',
        'observation.state.gripper.pos',
        'prev_cmd.gripper.pos',
        'observation.state.prev_cmd.gripper.pos',
    }:
        return 'normalized'
    return None


def normalize_dataset_gripper(
    aperture_value: float,
    cfg: FrankaResearch3Config,
    *,
    feature_name: str | None = None,
) -> float:
    aperture_value = float(max(0.0, aperture_value))
    unit = _gripper_feature_unit(feature_name)
    if unit == 'normalized' and cfg.gripper_backend != 'das':
        return float(np.clip(aperture_value, 0.0, 1.0))

    if cfg.gripper_backend == 'das':
        span_m = float(cfg.das_max_distance_m - cfg.das_min_distance_m)
        if span_m <= 0.0:
            return 0.0
        return float(np.clip((aperture_value - cfg.das_min_distance_m) / span_m, 0.0, 1.0))

    max_width_m = float(cfg.gripper_max_width_mm) / 1000.0
    max_width_mm = float(cfg.gripper_max_width_mm)
    if max_width_m <= 0.0 or max_width_mm <= 0.0:
        return 0.0
    if unit == 'm':
        return float(np.clip(aperture_value / max_width_m, 0.0, 1.0))
    if unit == 'mm':
        return float(np.clip(aperture_value / max_width_mm, 0.0, 1.0))

    # Fallback for legacy checkpoints whose gripper dimension had no unit-bearing name.
    if aperture_value <= max_width_m * 1.25:
        return float(np.clip(aperture_value / max_width_m, 0.0, 1.0))
    if aperture_value <= 1.0:
        return float(np.clip(aperture_value, 0.0, 1.0))
    return float(np.clip(aperture_value / max_width_mm, 0.0, 1.0))


def denormalize_live_gripper_observation(
    gripper_pos: float,
    cfg: FrankaResearch3Config,
    *,
    feature_name: str | None = None,
) -> float:
    gripper_pos = float(np.clip(gripper_pos, 0.0, 1.0))
    unit = _gripper_feature_unit(feature_name)
    if unit == 'normalized' and cfg.gripper_backend != 'das':
        return gripper_pos

    if cfg.gripper_backend == 'das':
        span_m = float(cfg.das_max_distance_m - cfg.das_min_distance_m)
        if span_m <= 0.0:
            return 0.0
        return float(cfg.das_min_distance_m + gripper_pos * span_m)

    max_width_m = float(cfg.gripper_max_width_mm) / 1000.0
    max_width_mm = float(cfg.gripper_max_width_mm)
    if unit == 'mm':
        return float(gripper_pos * max_width_mm)
    if max_width_m <= 0.0:
        return 0.0
    return float(gripper_pos * max_width_m)


def _dataset_feature_name_for_observation_key(state_names: list[str] | None, key: str) -> str | None:
    if not state_names:
        return None
    for name in state_names:
        if _state_name_to_observation_key(str(name)) == key:
            return str(name)
    return None


def convert_gripper_observation_to_dataset_units(
    observation: RobotObservation,
    *,
    robot_cfg: FrankaResearch3Config,
    state_names: list[str] | None = None,
) -> RobotObservation:
    converted_observation = dict(observation)
    for key in ('gripper.pos', PREV_CMD_GRIPPER_KEY):
        if key not in converted_observation:
            continue
        converted_observation[key] = denormalize_live_gripper_observation(
            float(converted_observation[key]),
            robot_cfg,
            feature_name=_dataset_feature_name_for_observation_key(state_names, key) or key,
        )
    return converted_observation


def _state_name_to_observation_key(name: str) -> str:
    aliases = {
        'x': 'ee.x',
        'y': 'ee.y',
        'z': 'ee.z',
        'qx': 'ee.qx',
        'qy': 'ee.qy',
        'qz': 'ee.qz',
        'qw': 'ee.qw',
        'state.ee.x': 'ee.x',
        'state.ee.y': 'ee.y',
        'state.ee.z': 'ee.z',
        'state.ee.qx': 'ee.qx',
        'state.ee.qy': 'ee.qy',
        'state.ee.qz': 'ee.qz',
        'state.ee.qw': 'ee.qw',
        'observation.state.ee.x': 'ee.x',
        'observation.state.ee.y': 'ee.y',
        'observation.state.ee.z': 'ee.z',
        'observation.state.ee.qx': 'ee.qx',
        'observation.state.ee.qy': 'ee.qy',
        'observation.state.ee.qz': 'ee.qz',
        'observation.state.ee.qw': 'ee.qw',
        'observation.state.left.ee.x': 'ee.x',
        'observation.state.left.ee.y': 'ee.y',
        'observation.state.left.ee.z': 'ee.z',
        'observation.state.left.ee.qx': 'ee.qx',
        'observation.state.left.ee.qy': 'ee.qy',
        'observation.state.left.ee.qz': 'ee.qz',
        'observation.state.left.ee.qw': 'ee.qw',
        'observation.state.right.ee.x': 'ee.x',
        'observation.state.right.ee.y': 'ee.y',
        'observation.state.right.ee.z': 'ee.z',
        'observation.state.right.ee.qx': 'ee.qx',
        'observation.state.right.ee.qy': 'ee.qy',
        'observation.state.right.ee.qz': 'ee.qz',
        'observation.state.right.ee.qw': 'ee.qw',
        'gripper': 'gripper.pos',
        'prev_cmd.gripper': PREV_CMD_GRIPPER_KEY,
        'handheld_gripper.pika_left.width_mm': 'gripper.pos',
        'handheld_gripper.pika_right.width_mm': 'gripper.pos',
        'corenetic_gripper.distance_m': 'gripper.pos',
        'box_gripper.distance_m': 'gripper.pos',
        'observation.state.gripper.pos': 'gripper.pos',
        'observation.state.prev_cmd.gripper.pos': PREV_CMD_GRIPPER_KEY,
        'observation.state_raw.handheld_gripper.pika_left.width_mm': 'gripper.pos',
        'observation.state_raw.handheld_gripper.pika_right.width_mm': 'gripper.pos',
        'observation.state_raw.corenetic_gripper.distance_m': 'gripper.pos',
        'observation.state_raw.box_gripper.distance_m': 'gripper.pos',
    }
    return aliases.get(name, name)


def _action_value(action_map: dict[str, float], *keys: str) -> float:
    return _action_value_with_name(action_map, *keys)[0]


def _action_value_with_name(action_map: dict[str, float], *keys: str) -> tuple[float, str]:
    for key in keys:
        if key in action_map:
            return float(action_map[key]), key
    raise KeyError(f'Missing action keys {keys!r} in decoded policy action.')


def extract_action_gripper_raw(action_tensor: torch.Tensor, action_names: list[str]) -> float:
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    if action_np.shape != (len(action_names),):
        raise ValueError(f'Expected policy action shape {(len(action_names),)}, got {action_np.shape}')
    action_map = {name: float(action_np[i]) for i, name in enumerate(action_names)}
    return _action_value(
        action_map,
        'gripper',
        'gripper.pos',
        'observation.state_raw.handheld_gripper.pika_left.width_mm',
        'observation.state_raw.handheld_gripper.pika_right.width_mm',
        'corenetic_gripper.distance_m',
        'observation.state_raw.corenetic_gripper.distance_m',
        'box_gripper.distance_m',
        'observation.state_raw.box_gripper.distance_m',
    )


def apply_camera_crop(
    image: np.ndarray,
    crop: list[int],
    *,
    feature_key: str,
    source_hw: tuple[int, int] | None,
) -> np.ndarray:
    """Take the same rectangle out of a live frame that the training view took out of its video.

    Both mismatches raise rather than warn. A crop is expressed in the recording's own pixels, so
    against any other frame size it is not an approximation of the training view -- it is a
    different part of the scene, and the rollout that follows looks healthy the whole way down.
    """
    x, y, w, h = (int(part) for part in crop)
    frame_h, frame_w = int(image.shape[0]), int(image.shape[1])
    if source_hw is not None and (frame_h, frame_w) != source_hw:
        raise ValueError(
            f'{feature_key}: live frame is {frame_w}x{frame_h} but the training view crop {crop} was '
            f'drawn on {source_hw[1]}x{source_hw[0]}. Open the camera at the recording resolution -- a '
            'crop is in source pixels and frames a different part of the scene at any other size.'
        )
    if x + w > frame_w or y + h > frame_h:
        raise ValueError(
            f'{feature_key}: training view crop {crop} does not fit a {frame_w}x{frame_h} live frame.'
        )
    return np.ascontiguousarray(image[y : y + h, x : x + w])


def resize_image_to_policy_shape(image: np.ndarray, image_feature: Any) -> np.ndarray:
    shape = tuple(getattr(image_feature, 'shape', ()))
    if len(shape) != 3:
        return image

    _, expected_h, expected_w = shape
    if image.shape[:2] == (expected_h, expected_w):
        return np.ascontiguousarray(image)

    import cv2

    interpolation = cv2.INTER_AREA if image.shape[0] > expected_h or image.shape[1] > expected_w else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(image, (expected_w, expected_h), interpolation=interpolation))


def build_policy_observation(
    state_observation: RobotObservation,
    *,
    state_names: list[str],
    input_features: dict[str, Any],
    tactile_fallback_observation: dict[str, np.ndarray] | None = None,
    camera_configs: dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig] | None = None,
    camera_crop_specs: dict[str, list[int]] | None = None,
    camera_crop_source_hw: dict[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    if 'observation.state' in input_features:
        observation['observation.state'] = np.asarray(
            [state_observation[_state_name_to_observation_key(name)] for name in state_names],
            dtype=np.float32,
        )

    for camera_key in extract_required_image_keys(input_features):
        if camera_key not in state_observation:
            raise KeyError(f"Camera '{camera_key}' missing from robot observation.")
        image = np.asarray(state_observation[camera_key], dtype=np.uint8)
        if image.ndim == 3 and image.shape[-1] == 3 and camera_configs is not None and camera_key in camera_configs:
            color_mode = getattr(camera_configs[camera_key], 'color_mode', None)
            try:
                color_mode = ColorMode(color_mode)
            except ValueError:
                color_mode = None
            if color_mode == ColorMode.BGR:
                image = np.ascontiguousarray(image[..., ::-1])
        feature_key = f'{_OBS_IMAGES_PREFIX}{camera_key}'
        raw_shape = tuple(image.shape)
        crop = (camera_crop_specs or {}).get(feature_key)
        if crop is not None:
            image = apply_camera_crop(
                image,
                crop,
                feature_key=feature_key,
                source_hw=(camera_crop_source_hw or {}).get(feature_key),
            )
        logged_shapes = getattr(build_policy_observation, '_logged_image_shapes', set())
        if camera_key not in logged_shapes:
            feature_shape = tuple(getattr(input_features[feature_key], 'shape', ()))
            print(
                '[INFO] policy_image_preprocess '
                f'camera={camera_key} raw_shape_hwc={raw_shape} policy_feature_chw={feature_shape} '
                + (
                    f"crop_xywh={','.join(str(part) for part in crop)} cropped_shape_hwc={tuple(image.shape)} "
                    'method=cv2.crop_then_resize'
                    if crop is not None
                    else 'crop=none method=cv2.resize_no_crop'
                )
            )
            logged_shapes.add(camera_key)
            setattr(build_policy_observation, '_logged_image_shapes', logged_shapes)
        observation[feature_key] = resize_image_to_policy_shape(image, input_features[feature_key])

    required_tactile_keys = extract_required_tactile_keys(input_features)
    missing_tactile_keys = [feature_key for feature_key in required_tactile_keys if feature_key not in state_observation]
    if missing_tactile_keys and tactile_fallback_observation is not None:
        unresolved_tactile_keys = [
            feature_key for feature_key in missing_tactile_keys if feature_key not in tactile_fallback_observation
        ]
        if unresolved_tactile_keys:
            raise KeyError(
                f"Missing tactile observation keys: {unresolved_tactile_keys}; fallback only provides {sorted(tactile_fallback_observation)}"
            )
    elif missing_tactile_keys:
        raise KeyError(f"Missing tactile observation keys: {missing_tactile_keys}")

    for feature_key, feature in input_features.items():
        if feature_key in observation:
            continue
        if feature.type != FeatureType.STATE:
            continue
        if feature_key in state_observation:
            observation[feature_key] = np.asarray(state_observation[feature_key], dtype=np.float32)
            continue
        if tactile_fallback_observation is not None and feature_key in tactile_fallback_observation:
            observation[feature_key] = np.asarray(tactile_fallback_observation[feature_key], dtype=np.float32)
            continue
        observation[feature_key] = np.zeros(tuple(feature.shape), dtype=np.float32)

    return observation


def show_policy_camera_preview_window(
    policy_observation: dict[str, np.ndarray],
    *,
    camera_keys: list[str],
    window_name: str = 'FR3 policy camera inputs',
) -> bool:
    import cv2

    labeled_images: list[np.ndarray] = []
    for camera_key in camera_keys:
        feature_key = f'{_OBS_IMAGES_PREFIX}{camera_key}'
        if feature_key not in policy_observation:
            continue
        image_rgb = np.asarray(policy_observation[feature_key], dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            continue
        image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
        label = camera_key
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_width, label_height = label_size
        cv2.rectangle(image_bgr, (0, 0), (label_width + 18, label_height + 18), (0, 0, 0), thickness=-1)
        cv2.putText(
            image_bgr,
            label,
            (9, label_height + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        labeled_images.append(image_bgr)

    if not labeled_images:
        return True

    target_height = max(image.shape[0] for image in labeled_images)
    resized_images: list[np.ndarray] = []
    for image in labeled_images:
        if image.shape[0] == target_height:
            resized_images.append(image)
            continue
        scale = target_height / float(image.shape[0])
        resized_images.append(cv2.resize(image, (int(round(image.shape[1] * scale)), target_height)))

    canvas = np.ascontiguousarray(np.concatenate(resized_images, axis=1))
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        print(f'[WARN] camera_preview_window=disabled reason=cv2_error: {exc}')
        return False
    if key in (ord('q'), 27):
        cv2.destroyWindow(window_name)
        print('[INFO] camera_preview_window=closed_by_user')
        return False
    return True


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish a file by rename.

    A reader polling these must never see a half-written JPEG, and it polls far more often
    than anything here writes.
    """

    tmp_path = path.with_name(f'.{path.name}.tmp')
    with open(tmp_path, 'wb') as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


class PolicyCameraPreviewSink:
    """Publish the frames the policy is seeing as JPEG files a viewer can poll.

    Exists because the rollout's own preview is an OpenCV window, which needs an X display on
    the machine the robot is wired to and shows the run only to someone standing at it. The
    frames a remote operator most needs to see are precisely the ones the policy is being fed,
    so they are published from the same observation the network receives -- after cropping and
    resizing, not before.

    Encoding happens on a background thread. The caller is a real-robot control loop at the
    dataset's frame rate, and a JPEG encode of two 640x480 frames is a few milliseconds it
    should not spend: `publish` only copies, and a slow or stalled reader can never add
    latency to the arm.
    """

    def __init__(self, output_dir: Path, *, camera_keys: list[str], fps: float = 5.0):
        self.output_dir = Path(output_dir)
        self.camera_keys = list(camera_keys)
        self._min_interval_s = 1.0 / fps if fps > 0 else 0.0
        self._pending: dict[str, np.ndarray] | None = None
        self._pending_immediate = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    def start(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f'[WARN] policy_camera_preview_sink=disabled reason=mkdir_failed: {exc}')
            self._failed = True
            return
        self._thread = threading.Thread(target=self._encode_loop, daemon=True, name='policy-preview-sink')
        self._thread.start()
        print(
            f'[INFO] policy_camera_preview_sink=enabled dir={self.output_dir} '
            f"cameras={','.join(self.camera_keys)}"
        )

    def publish(self, policy_observation: dict[str, np.ndarray], *, immediate: bool = False) -> None:
        """Queue one frame per camera for encoding.

        `immediate` exempts this frame from the publish rate limit. A rollout publishes many
        frames a second and only the newest matters, but the still published when the arm
        parks is the only one that will be written until the next rollout starts -- dropping
        it for arriving too soon after the last rollout frame would leave the viewer with a
        file whose age no longer says anything about the scene it shows.
        """
        if self._failed or self._closed.is_set():
            return
        frames = self._frames_from(policy_observation)
        if not frames:
            return
        with self._lock:
            self._pending = frames
            self._pending_immediate = self._pending_immediate or immediate
        self._wake.set()

    def _frames_from(self, policy_observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera_key in self.camera_keys:
            image = policy_observation.get(f'{_OBS_IMAGES_PREFIX}{camera_key}')
            if image is None:
                continue
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3:
                continue
            # Copied, not referenced: the caller reuses its observation buffers, and the
            # encoder thread would otherwise race with the next capture.
            frames[camera_key] = np.array(image, dtype=np.uint8, copy=True)
        return frames

    def write_still(self, policy_observation: dict[str, np.ndarray], output_dir: Path) -> list[str]:
        """Encode and write one frame per camera on the calling thread.

        The calibration probe uses this instead of `publish`. It is standing at a commanded
        coordinate and about to say so in the log, and a reader that acts on that line has to
        find the frame taken there -- not the one the encoder thread had not got to yet, which
        shows the arm somewhere else and would be measured as if it did not.
        """

        import cv2

        frames = self._frames_from(policy_observation)
        if not frames:
            return []
        directory = Path(output_dir)
        written: list[str] = []
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for camera_key, image_rgb in frames.items():
                ok, buffer = cv2.imencode('.jpg', image_rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if not ok:
                    continue
                _write_bytes_atomic(directory / f'{camera_key}.jpg', buffer.tobytes())
                written.append(camera_key)
        except Exception as exc:  # noqa: BLE001 - a still must never end a session
            print(f'[WARN] policy_camera_still_write_failed dir={directory}: {exc}')
            return written
        return written

    def _encode_loop(self) -> None:
        import cv2

        last_write_s = 0.0
        while not self._closed.is_set():
            if not self._wake.wait(timeout=0.5):
                continue
            self._wake.clear()
            with self._lock:
                frames = self._pending
                immediate = self._pending_immediate
                self._pending = None
                self._pending_immediate = False
            if not frames:
                continue
            now_s = time.monotonic()
            if not immediate and self._min_interval_s and now_s - last_write_s < self._min_interval_s:
                continue
            last_write_s = now_s
            for camera_key, image_rgb in frames.items():
                try:
                    ok, buffer = cv2.imencode('.jpg', image_rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    _write_bytes_atomic(self.output_dir / f'{camera_key}.jpg', buffer.tobytes())
                except Exception as exc:  # noqa: BLE001 - a preview must never end a rollout
                    print(f'[WARN] policy_camera_preview_write_failed camera={camera_key}: {exc}')
                    self._failed = True
                    return

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        for camera_key in self.camera_keys:
            # Remove the frames rather than leaving the last one behind: a stale JPEG sitting
            # in /dev/shm after the run ends is indistinguishable from a live one to anything
            # that only looks at the file, and the viewer would show a finished rollout as
            # still going.
            try:
                (self.output_dir / f'{camera_key}.jpg').unlink()
            except OSError:
                pass


# A rollout's gripper command is the event signal, not its observed width. In this dataset
# `observation.state.gripper.pos` reads 0 on 47% of frames while the command held a clean 1.0,
# so any "did it close?" test keyed on the observation fires on dropouts instead of on grasps.
_TRACE_GRIPPER_CLOSED_BELOW = 0.5
# An open stretch shorter than this does not end a hold. Observed on rollout 9 of
# L4_full48_holdout22_40/030000: the command touched 0.4997 for two steps, went back up, and
# only shut for real 22 steps later, so "the first hold" was a two-sample blip and the rollout
# scored lift 0 with the grasp and release points on top of each other. The same run's real
# hold was itself split by a three-step excursion back over the threshold.
_TRACE_GRIPPER_REOPEN_MIN_STEPS = 5


def _dominant_closed_span(closed: np.ndarray) -> tuple[int, int]:
    """The inclusive `(first, last)` step of the hold that carries the rollout.

    The gripper command is a continuous signal crossed against a threshold, so one hold can
    come back as several runs and a transient can come back as a hold. Runs parted by less
    than `_TRACE_GRIPPER_REOPEN_MIN_STEPS` open steps are one hold, and of the holds that
    remain the longest is the one that did the carrying -- a grasp is a hold, not a blip.
    """
    flags = closed.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    spans: list[list[int]] = [[int(starts[0]), int(ends[0])]]
    for start, end in zip(starts[1:], ends[1:]):
        if int(start) - spans[-1][1] - 1 < _TRACE_GRIPPER_REOPEN_MIN_STEPS:
            spans[-1][1] = int(end)
        else:
            spans.append([int(start), int(end)])
    first, last = max(spans, key=lambda span: span[1] - span[0])
    return first, last


class RolloutGeometryTrace:
    """Where one rollout put the gripper, sampled while it happens and reduced when it ends.

    A rollout cannot be repeated. The peg is placed by hand, and that placement stops existing
    the moment the arm touches it, so anything not written down during the run is gone. This
    keeps every step's end-effector position and gripper command, and derives the two points
    that carry the result -- where the gripper closed, and where it opened again -- only at the
    end, from the buffer it kept.

    Deriving them online would bake a threshold into data that can never be recaptured; deriving
    them from a retained trace leaves the threshold a parameter that can be changed afterwards
    with the same rollouts still in hand.

    Positions are sampled in the dataset's own frame, because the point of recording them is to
    compare them against where the demonstrations grasped, and a comparison across two frames is
    not a comparison.
    """

    def __init__(self, rollout_index: int, *, trace_dir: Path | None = None):
        self.rollout_index = int(rollout_index)
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self._rows: list[tuple[int, float, float, float, float, float, str, str]] = []

    def sample(
        self,
        *,
        step_idx: int,
        position_xyz: np.ndarray,
        gripper_command: float,
        gripper_raw: float,
        command_status: str,
        source: str = 'policy',
    ) -> None:
        try:
            position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
        except (ValueError, TypeError):
            return
        if not np.all(np.isfinite(position)):
            return
        self._rows.append(
            (
                int(step_idx),
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(gripper_command),
                float(gripper_raw),
                str(command_status),
                str(source),
            )
        )

    def summary(self) -> dict[str, Any]:
        """The landing points of this rollout, or what it reached if it never closed."""
        if not self._rows:
            return {'samples': 0, 'closed': False}
        positions = np.asarray([row[1:4] for row in self._rows], dtype=np.float64)
        gripper = np.asarray([row[4] for row in self._rows], dtype=np.float64)
        sources = [row[7] for row in self._rows]
        # A grasp is a falling edge, not a level. A rollout can begin with the gripper already
        # commanded shut -- the start-pose check warns about exactly that when the live gripper
        # does not match the dataset's start contract -- and "first step under the threshold"
        # then returns step 0, putting the grasp at the home pose ~340 mm above the table
        # instead of on the object. Requiring the command to have been open first makes the
        # event a transition, which is what closing on something is.
        open_steps = np.flatnonzero(gripper >= _TRACE_GRIPPER_CLOSED_BELOW)
        first_open = int(open_steps[0]) if open_steps.size else len(self._rows)
        closed = gripper < _TRACE_GRIPPER_CLOSED_BELOW
        closed[:first_open] = False
        result: dict[str, Any] = {'samples': len(self._rows), 'closed': bool(closed.any())}
        # Reduced from the per-step column rather than tracked alongside it, so the marker can
        # never claim a takeover the trace file does not show. `intervened` is separate from
        # the span list because it is the one bit every consumer needs: a rollout the operator
        # drove part of says nothing about the policy's success rate, and folding it into that
        # rate is the same arithmetic mistake as forgetting to grade a round.
        expert_spans_found = expert_spans(sources)
        if expert_spans_found:
            result['intervened'] = True
            result['expert_steps'] = sum(last - first + 1 for first, last in expert_spans_found)
            result['expert_spans'] = expert_spans_found
        # Who was driving at the instant of each landing event. `intervened` says a human was in
        # the rollout somewhere; this says whether they were in *this* event, which is the
        # question a landing point and a grade both actually ask. A grasp the policy made before
        # the operator stepped in is still the policy's datum, and a peg the operator seated is
        # not the policy's success.
        def driver_at(index: int) -> str:
            return 'expert' if sources[index] == 'expert' else 'policy'

        if not closed.any():
            # Never closed. The lowest point it reached is still the landing point worth
            # plotting: it is where the policy decided the object was.
            lowest = int(np.argmin(positions[:, 2]))
            result['approach_xyz'] = positions[lowest].tolist()
            result['approach_by'] = driver_at(lowest)
            return result
        close_idx, hold_end = _dominant_closed_span(closed)
        release_idx = min(hold_end + 1, len(self._rows) - 1)
        apex_z = float(positions[close_idx : release_idx + 1, 2].max())
        result.update(
            {
                'grasp_xyz': positions[close_idx].tolist(),
                'release_xyz': positions[release_idx].tolist(),
                'apex_z': apex_z,
                'lift_m': apex_z - float(positions[close_idx, 2]),
                'descent_m': apex_z - float(positions[release_idx, 2]),
                'held_steps': release_idx - close_idx,
                'grasp_by': driver_at(close_idx),
                'release_by': driver_at(release_idx),
            }
        )
        return result

    def summary_log_fields(self) -> str:
        """The summary as `key=value` fields appended to the rollout's end marker.

        On the end marker rather than in a file of its own because the page already reads that
        line, and a second channel for the same event is a second thing that can be out of step
        with the first.
        """
        summary = self.summary()
        fields = [f"samples={summary.get('samples', 0)}", f"closed={int(bool(summary.get('closed')))}"]
        for key in ('grasp_xyz', 'release_xyz', 'approach_xyz'):
            point = summary.get(key)
            if point is not None:
                fields.append(f"{key}=" + ','.join(f'{value:.4f}' for value in point))
        for key in ('apex_z', 'lift_m', 'descent_m'):
            value = summary.get(key)
            if value is not None:
                fields.append(f'{key}={float(value):.4f}')
        held = summary.get('held_steps')
        if held is not None:
            fields.append(f'held_steps={int(held)}')
        for key in ('grasp_by', 'release_by', 'approach_by'):
            driver = summary.get(key)
            if driver is not None:
                fields.append(f'{key}={driver}')
        if summary.get('intervened'):
            fields.append('intervened=1')
            fields.append(f"expert_steps={int(summary['expert_steps'])}")
            fields.append(
                'expert_spans=' + ';'.join(f'{first}-{last}' for first, last in summary['expert_spans'])
            )
        return ' '.join(fields)

    def write(self) -> None:
        """Persist the raw per-step trace, so the reduction above can be redone later."""
        if self.trace_dir is None or not self._rows:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f'rollout_{self.rollout_index:03d}.csv'
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('step,x,y,z,gripper_cmd,gripper_raw,status,source\n')
                for step_idx, x, y, z, gripper_cmd, gripper_raw, status, source in self._rows:
                    handle.write(
                        f'{step_idx},{x:.6f},{y:.6f},{z:.6f},{gripper_cmd:.4f},{gripper_raw:.4f},'
                        f'{status},{source}\n'
                    )
        except OSError as exc:  # noqa: BLE001 - a trace must never end a rollout
            print(f'[WARN] rollout_trace_write_failed index={self.rollout_index}: {exc}')
            return
        print(f'[INFO] rollout_trace_written index={self.rollout_index} path={path} rows={len(self._rows)}')


def close_camera_preview_window(window_name: str = 'FR3 policy camera inputs') -> None:
    try:
        import cv2

        cv2.destroyWindow(window_name)
    except Exception:
        pass


def _pose_from_position_and_quaternion(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return pose


def _pose_from_position_and_rotvec(position_xyz: np.ndarray, rotvec_xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(rotvec_xyz).as_matrix()
    return pose


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    pose_inv = np.eye(4, dtype=np.float64)
    pose_inv[:3, :3] = rotation.T
    pose_inv[:3, 3] = -rotation.T @ translation
    return pose_inv


def _pose_to_xyzquat(pose: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.asarray(
        [
            pose[0, 3],
            pose[1, 3],
            pose[2, 3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
            quaternion_xyzw[3],
        ],
        dtype=np.float64,
    )


def _pose_from_quaternion_observation(
    observation: RobotObservation,
    *,
    position_keys: tuple[str, str, str] = EE_POSITION_KEYS,
    quaternion_keys: tuple[str, str, str, str] = EE_QUAT_KEYS,
) -> np.ndarray:
    return _pose_from_position_and_quaternion(
        np.asarray([observation[key] for key in position_keys], dtype=np.float64),
        np.asarray([observation[key] for key in quaternion_keys], dtype=np.float64),
    )


def convert_absolute_observation_from_E_to_I(absolute_observation_e: RobotObservation) -> RobotObservation:
    return dict(absolute_observation_e)


def localize_observation_to_start_frame(
    absolute_observation: RobotObservation,
    episode_start_position_xyz: np.ndarray,
    *,
    previous_quaternion_xyzw: np.ndarray | None = None,
) -> tuple[RobotObservation, np.ndarray]:
    absolute_position = np.asarray(
        [absolute_observation['ee.x'], absolute_observation['ee.y'], absolute_observation['ee.z']],
        dtype=np.float64,
    )
    absolute_quaternion_xyzw = np.asarray(
        [absolute_observation['ee.qx'], absolute_observation['ee.qy'], absolute_observation['ee.qz'], absolute_observation['ee.qw']],
        dtype=np.float64,
    )
    local_quaternion_xyzw = _continuous_quaternion(absolute_quaternion_xyzw, previous_quaternion_xyzw)
    local_position = absolute_position - np.asarray(episode_start_position_xyz, dtype=np.float64)

    localized_observation = dict(absolute_observation)
    localized_observation.update(
        {
            'ee.x': float(local_position[0]),
            'ee.y': float(local_position[1]),
            'ee.z': float(local_position[2]),
            'ee.qx': float(local_quaternion_xyzw[0]),
            'ee.qy': float(local_quaternion_xyzw[1]),
            'ee.qz': float(local_quaternion_xyzw[2]),
            'ee.qw': float(local_quaternion_xyzw[3]),
        }
    )
    return localized_observation, local_quaternion_xyzw


def convert_local_command_to_base_frame(
    local_robot_command: dict[str, float],
    episode_start_position_xyz: np.ndarray,
) -> dict[str, float]:
    local_position = np.asarray(
        [local_robot_command['ee.x'], local_robot_command['ee.y'], local_robot_command['ee.z']],
        dtype=np.float64,
    )
    absolute_position = np.asarray(episode_start_position_xyz, dtype=np.float64) + local_position
    base_robot_command = dict(local_robot_command)
    base_robot_command.update(
        {
            'ee.x': float(absolute_position[0]),
            'ee.y': float(absolute_position[1]),
            'ee.z': float(absolute_position[2]),
        }
    )
    return base_robot_command


def _load_episode_data_locations(dataset_root: Path) -> list[tuple[int, int, int]]:
    import pyarrow.parquet as pq

    dataset_root = _resolve_repo_path(dataset_root)
    meta_dir = dataset_root / 'meta' / 'episodes'
    meta_files = sorted(meta_dir.rglob('*.parquet'))
    if not meta_files:
        raise FileNotFoundError(f'No episode metadata parquet files found in {meta_dir}')

    episode_rows: list[tuple[int, int, int]] = []
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file)).to_pydict()
        episode_indices = table['episode_index']
        chunk_indices = table['data/chunk_index']
        file_indices = table['data/file_index']
        for episode_index, chunk_index, file_index in zip(episode_indices, chunk_indices, file_indices, strict=True):
            episode_rows.append((int(episode_index), int(chunk_index), int(file_index)))

    episode_rows.sort(key=lambda item: item[0])
    return episode_rows


def _load_episode_start_state_rows(
    dataset_root: Path,
    *,
    state_key: str = 'observation.state',
) -> list[tuple[int, np.ndarray]]:
    import pyarrow.parquet as pq

    dataset_root = _resolve_repo_path(dataset_root)
    start_state_rows: list[tuple[int, np.ndarray]] = []
    for episode_index, chunk_index, file_index in _load_episode_data_locations(dataset_root):
        data_file = _resolve_dataset_data_file(dataset_root, chunk_index=chunk_index, file_index=file_index)
        table = pq.read_table(str(data_file), columns=['episode_index', state_key]).to_pydict()
        for row_episode_index, state in zip(table['episode_index'], table[state_key], strict=True):
            if int(row_episode_index) != episode_index:
                continue
            start_state_rows.append((episode_index, np.asarray(state, dtype=np.float64)))
            break
        else:
            raise ValueError(f'Episode {episode_index} metadata found, but no rows matched in {data_file}')

    if not start_state_rows:
        raise ValueError(f'No episode starts resolved from {dataset_root}')
    return start_state_rows


def _load_episode_start_gripper_targets(
    dataset_root: Path,
    *,
    state_key: str = 'observation.state',
) -> tuple[dict[int, float], str, str] | None:
    """Return the per-episode gripper target that should define rollout start.

    The first observation can lag the command at episode start: on this FR3/Pika
    dataset, some first-frame sensor readings are still closed while the first
    action and prev_cmd already say "open". For rollout alignment, the command
    contract is what matters: physically put the gripper where the policy was
    commanded to start, not where a stale first sensor sample happened to be.
    """
    import pyarrow.parquet as pq

    dataset_root = _resolve_repo_path(dataset_root)
    info = _load_dataset_info(dataset_root)
    state_names = _load_observation_state_feature_names(dataset_root, state_key=state_key)
    action_names_raw = info.get('features', {}).get('action', {}).get('names')
    action_names = [str(name) for name in action_names_raw] if isinstance(action_names_raw, list) else []

    column = state_key
    value_index: int
    source: str
    feature_name = 'gripper.pos'
    if 'gripper.pos' in action_names:
        column = 'action'
        value_index = action_names.index('gripper.pos')
        source = 'action.gripper.pos'
    elif 'prev_cmd.gripper.pos' in state_names:
        value_index = state_names.index('prev_cmd.gripper.pos')
        source = f'{state_key}.prev_cmd.gripper.pos'
    elif 'gripper.pos' in state_names:
        value_index = state_names.index('gripper.pos')
        source = f'{state_key}.gripper.pos'
    else:
        return None

    targets: dict[int, float] = {}
    for episode_index, chunk_index, file_index in _load_episode_data_locations(dataset_root):
        data_file = _resolve_dataset_data_file(dataset_root, chunk_index=chunk_index, file_index=file_index)
        table = pq.read_table(str(data_file), columns=['episode_index', column]).to_pydict()
        for row_episode_index, values in zip(table['episode_index'], table[column], strict=True):
            if int(row_episode_index) != episode_index:
                continue
            targets[int(episode_index)] = float(np.asarray(values, dtype=np.float64)[value_index])
            break
        else:
            raise ValueError(f'Episode {episode_index} metadata found, but no rows matched in {data_file}')

    if not targets:
        return None
    return targets, source, feature_name


def _load_episode_start_states(
    dataset_root: Path,
    *,
    state_key: str = 'observation.state',
) -> np.ndarray:
    start_state_rows = _load_episode_start_state_rows(dataset_root, state_key=state_key)
    return np.asarray([state for _, state in start_state_rows], dtype=np.float64)


def _quaternion_angle_deg(quaternion_a_xyzw: np.ndarray, quaternion_b_xyzw: np.ndarray) -> float:
    quaternion_a = np.asarray(quaternion_a_xyzw, dtype=np.float64)
    quaternion_b = np.asarray(quaternion_b_xyzw, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(quaternion_a, quaternion_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def estimate_dataset_start_pose_contract(
    dataset_root: Path,
    *,
    state_key: str = 'observation.state',
) -> tuple[np.ndarray, dict[str, Any]]:
    start_state_rows = _load_episode_start_state_rows(dataset_root, state_key=state_key)
    start_states = np.asarray([state for _, state in start_state_rows], dtype=np.float64)
    state_indices = _extract_dataset_state_contract_indices(dataset_root, state_key=state_key)
    positions = np.asarray([[state[state_indices[key]] for key in EE_POSITION_KEYS] for state in start_states], dtype=np.float64)
    quaternions = np.asarray([[state[state_indices[key]] for key in EE_QUAT_KEYS] for state in start_states], dtype=np.float64)
    observation_gripper_values = (
        np.asarray([state[state_indices['gripper.pos']] for state in start_states], dtype=np.float64)
        if 'gripper.pos' in state_indices
        else None
    )
    gripper_targets = _load_episode_start_gripper_targets(dataset_root, state_key=state_key)

    aligned_quaternions = quaternions.copy()
    reference_quaternion = aligned_quaternions[0]
    for idx in range(len(aligned_quaternions)):
        if float(np.dot(aligned_quaternions[idx], reference_quaternion)) < 0.0:
            aligned_quaternions[idx] = -aligned_quaternions[idx]

    mean_quaternion = aligned_quaternions.mean(axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    mean_position = positions.mean(axis=0)
    representative_pose_xyzquat = np.concatenate([mean_position, mean_quaternion], dtype=np.float64)

    rotation_spread_deg = np.asarray(
        [_quaternion_angle_deg(quaternion_xyzw, mean_quaternion) for quaternion_xyzw in aligned_quaternions],
        dtype=np.float64,
    )
    stats: dict[str, Any] = {
        'state_key': state_key,
        'episodes': int(len(start_states)),
        'mean_position_xyz_m': mean_position.copy(),
        'position_std_xyz_mm': positions.std(axis=0) * 1000.0,
        'mean_quaternion_xyzw': mean_quaternion.copy(),
        'rotation_spread_mean_deg': float(rotation_spread_deg.mean()),
        'rotation_spread_p95_deg': float(np.percentile(rotation_spread_deg, 95)),
        'rotation_spread_max_deg': float(rotation_spread_deg.max()),
    }
    if gripper_targets is not None:
        gripper_targets_by_episode, gripper_source, gripper_feature_name = gripper_targets
        ordered_targets = np.asarray(
            [gripper_targets_by_episode[int(episode_index)] for episode_index, _ in start_state_rows],
            dtype=np.float64,
        )
        stats['gripper_mean'] = float(ordered_targets.mean())
        stats['gripper_std'] = float(ordered_targets.std())
        stats['gripper_feature_name'] = gripper_feature_name
        stats['gripper_source'] = gripper_source
        if observation_gripper_values is not None:
            stats['observation_gripper_mean'] = float(observation_gripper_values.mean())
            stats['observation_gripper_std'] = float(observation_gripper_values.std())
    elif observation_gripper_values is not None:
        stats['gripper_mean'] = float(observation_gripper_values.mean())
        stats['gripper_std'] = float(observation_gripper_values.std())
        stats['gripper_feature_name'] = 'gripper.pos'
        stats['gripper_source'] = f'{state_key}.gripper.pos'
    return representative_pose_xyzquat, stats


def summarize_live_start_alignment_to_dataset_starts(
    dataset_root: Path,
    T_B_Ws: np.ndarray,
    live_start_pose_i: np.ndarray,
    *,
    state_key: str = 'observation.state',
    live_gripper: float | None = None,
) -> dict[str, Any]:
    start_state_rows = _load_episode_start_state_rows(dataset_root, state_key=state_key)
    state_indices = _extract_dataset_state_contract_indices(dataset_root, state_key=state_key)
    gripper_targets = _load_episode_start_gripper_targets(dataset_root, state_key=state_key)
    start_gripper_targets_by_episode = gripper_targets[0] if gripper_targets is not None else None
    live_position = np.asarray(live_start_pose_i[:3, 3], dtype=np.float64)
    live_rotation = Rotation.from_matrix(live_start_pose_i[:3, :3])

    position_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    gripper_abs_errors: list[float] = []
    best_episode_index = -1
    best_position_error_mm = float('inf')
    best_rotation_error_deg = float('inf')
    best_gripper_abs_delta = float('nan')
    best_episode_gripper = float('nan')
    best_score = float('inf')

    for episode_index, state in start_state_rows:
        position_xyz, quaternion_xyzw, dataset_gripper = _extract_pose_gripper_from_state_row(
            state,
            state_indices=state_indices,
        )
        if start_gripper_targets_by_episode is not None:
            dataset_gripper = start_gripper_targets_by_episode.get(int(episode_index), dataset_gripper)
        dataset_start_pose_i = _pose_from_position_and_quaternion(position_xyz, quaternion_xyzw)
        predicted_start_pose_i = T_B_Ws @ dataset_start_pose_i
        position_error_mm = float(np.linalg.norm(predicted_start_pose_i[:3, 3] - live_position) * 1000.0)
        rotation_error_deg = float(
            np.degrees(
                np.linalg.norm(
                    (live_rotation.inv() * Rotation.from_matrix(predicted_start_pose_i[:3, :3])).as_rotvec()
                )
            )
        )
        gripper_abs_delta = float('nan')
        if live_gripper is not None and dataset_gripper is not None:
            gripper_abs_delta = float(abs(live_gripper - dataset_gripper))
            gripper_abs_errors.append(gripper_abs_delta)
        position_errors_mm.append(position_error_mm)
        rotation_errors_deg.append(rotation_error_deg)
        score = position_error_mm + rotation_error_deg
        if score < best_score:
            best_score = score
            best_episode_index = int(episode_index)
            best_position_error_mm = position_error_mm
            best_rotation_error_deg = rotation_error_deg
            best_gripper_abs_delta = gripper_abs_delta
            best_episode_gripper = dataset_gripper

    position_errors_mm_arr = np.asarray(position_errors_mm, dtype=np.float64)
    rotation_errors_deg_arr = np.asarray(rotation_errors_deg, dtype=np.float64)
    summary = {
        'best_episode_index': best_episode_index,
        'best_position_error_mm': best_position_error_mm,
        'best_rotation_error_deg': best_rotation_error_deg,
        'median_position_error_mm': float(np.median(position_errors_mm_arr)),
        'p95_position_error_mm': float(np.percentile(position_errors_mm_arr, 95)),
        'median_rotation_error_deg': float(np.median(rotation_errors_deg_arr)),
        'p95_rotation_error_deg': float(np.percentile(rotation_errors_deg_arr, 95)),
    }
    if live_gripper is not None and gripper_abs_errors:
        gripper_abs_errors_arr = np.asarray(gripper_abs_errors, dtype=np.float64)
        summary.update(
            {
                'live_gripper': float(live_gripper),
                'best_episode_gripper': best_episode_gripper,
                'best_gripper_abs_delta': best_gripper_abs_delta,
                'median_gripper_abs_delta': float(np.median(gripper_abs_errors_arr)),
                'p95_gripper_abs_delta': float(np.percentile(gripper_abs_errors_arr, 95)),
            }
        )
    return summary


def convert_base_observation_from_I_to_dataset_frame(
    absolute_observation_i: RobotObservation,
    T_B_Ws: np.ndarray,
    *,
    previous_quaternion_xyzw: np.ndarray | None = None,
) -> tuple[RobotObservation, np.ndarray]:
    dataset_observation_i = dict(absolute_observation_i)
    current_dataset_quaternion_xyzw = previous_quaternion_xyzw
    for position_keys, quaternion_keys in (
        (EE_POSITION_KEYS, EE_QUAT_KEYS),
        (PREV_CMD_POSITION_KEYS, PREV_CMD_QUAT_KEYS),
    ):
        if not all(key in absolute_observation_i for key in position_keys + quaternion_keys):
            continue
        input_quaternion_xyzw = np.asarray([absolute_observation_i[key] for key in quaternion_keys], dtype=np.float64)
        absolute_pose_i = _pose_from_quaternion_observation(
            absolute_observation_i,
            position_keys=position_keys,
            quaternion_keys=quaternion_keys,
        )
        dataset_pose_i = _invert_pose(T_B_Ws) @ absolute_pose_i
        dataset_quaternion_xyzw = Rotation.from_matrix(dataset_pose_i[:3, :3]).as_quat()
        reference_quaternion = (
            current_dataset_quaternion_xyzw if position_keys == EE_POSITION_KEYS else input_quaternion_xyzw
        )
        dataset_quaternion_xyzw = _continuous_quaternion(dataset_quaternion_xyzw, reference_quaternion)
        dataset_observation_i.update(
            {
                position_keys[0]: float(dataset_pose_i[0, 3]),
                position_keys[1]: float(dataset_pose_i[1, 3]),
                position_keys[2]: float(dataset_pose_i[2, 3]),
                quaternion_keys[0]: float(dataset_quaternion_xyzw[0]),
                quaternion_keys[1]: float(dataset_quaternion_xyzw[1]),
                quaternion_keys[2]: float(dataset_quaternion_xyzw[2]),
                quaternion_keys[3]: float(dataset_quaternion_xyzw[3]),
            }
        )
        if position_keys == EE_POSITION_KEYS:
            current_dataset_quaternion_xyzw = dataset_quaternion_xyzw
    if current_dataset_quaternion_xyzw is None:
        raise KeyError('Current EE pose keys are missing from the absolute observation.')
    return dataset_observation_i, current_dataset_quaternion_xyzw


def convert_dataset_command_to_base_frame(
    dataset_robot_command_i: dict[str, float],
    T_B_Ws: np.ndarray,
) -> dict[str, float]:
    dataset_pose_i = _pose_from_position_and_rotvec(
        np.asarray(
            [dataset_robot_command_i['ee.x'], dataset_robot_command_i['ee.y'], dataset_robot_command_i['ee.z']],
            dtype=np.float64,
        ),
        np.asarray(
            [dataset_robot_command_i['ee.wx'], dataset_robot_command_i['ee.wy'], dataset_robot_command_i['ee.wz']],
            dtype=np.float64,
        ),
    )
    base_pose_i = T_B_Ws @ dataset_pose_i
    base_rotvec_xyz = Rotation.from_matrix(base_pose_i[:3, :3]).as_rotvec()

    base_robot_command_i = dict(dataset_robot_command_i)
    base_robot_command_i.update(
        {
            'ee.x': float(base_pose_i[0, 3]),
            'ee.y': float(base_pose_i[1, 3]),
            'ee.z': float(base_pose_i[2, 3]),
            'ee.wx': float(base_rotvec_xyz[0]),
            'ee.wy': float(base_rotvec_xyz[1]),
            'ee.wz': float(base_rotvec_xyz[2]),
        }
    )
    return base_robot_command_i


def convert_base_command_from_I_to_E(base_robot_command_i: dict[str, float]) -> dict[str, float]:
    return dict(base_robot_command_i)


def build_delta_action_reconstructor(action_names: list[str]) -> DeltaEEToAbsoluteEEAction | None:
    """One stateful reconstructor for a delta-action checkpoint, or None for absolute EE.

    Returns the *same* processor step the recorder used to turn its delta back into a robot
    command, so deployment cannot drift from training. Built once at startup rather than per
    step: the step keeps quaternion-sign continuity across frames.

    The workspace clamp is deliberately not passed here. Reconstruction happens in the dataset
    frame, while ``robot_cfg.workspace_min/max`` bound the robot base frame; clamping in the wrong
    frame would silently distort the command. The existing base-frame guards
    (``limit_command_for_safety`` and the robot driver's own clip) still apply.
    """
    reference = delta_reference_from_action_names(action_names)
    if reference is None:
        return None
    print(f'[INFO] action_contract=delta reference={reference}')
    return DeltaEEToAbsoluteEEAction(reference=reference)


def decode_action_to_robot_command(
    action_tensor: torch.Tensor,
    *,
    action_names: list[str],
    robot_cfg: FrankaResearch3Config,
    gripper_close_below: float | None = None,
    delta_reconstructor: DeltaEEToAbsoluteEEAction | None = None,
    dataset_observation_i: RobotObservation | None = None,
) -> dict[str, float]:
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    if action_np.shape != (len(action_names),):
        raise ValueError(f'Expected policy action shape {(len(action_names),)}, got {action_np.shape}')

    action_map = {name: float(action_np[i]) for i, name in enumerate(action_names)}

    if delta_reconstructor is not None:
        # A view built with --action-drop-dims omits the axes the teleop rig locks, so the policy
        # emits no drx/dry. Restore them as the exact zeros the dropped columns held, *before*
        # reconstruction: DeltaEEToAbsoluteEEAction treats an action missing any delta key as one
        # that is already absolute and passes it straight through, which would feed a
        # millimetre-scale increment to the arm as if it were a base-frame target.
        for rotvec_key in delta_ee_rotvec_keys(delta_reconstructor.reference):
            action_map.setdefault(rotvec_key, 0.0)
        # Delta contract: the policy emits an increment against a reference pose that lives in
        # the dataset frame, so it must be rebuilt here, before the dataset -> base -> E
        # conversions downstream.
        if dataset_observation_i is None:
            raise ValueError(
                'A delta-action checkpoint needs the dataset-frame observation to rebuild an '
                'absolute target; dataset_observation_i was not provided.'
            )
        rebuilt = delta_reconstructor(
            {
                TransitionKey.ACTION: dict(action_map),
                TransitionKey.OBSERVATION: dict(dataset_observation_i),
            }
        )[TransitionKey.ACTION]
        rebuilt_rotvec = Rotation.from_quat(
            np.asarray([rebuilt['ee.qx'], rebuilt['ee.qy'], rebuilt['ee.qz'], rebuilt['ee.qw']], dtype=np.float64)
        ).as_rotvec()
        raw_delta_gripper, delta_gripper_feature = _action_value_with_name(action_map, 'gripper', 'gripper.pos')
        if gripper_close_below is not None and raw_delta_gripper < float(gripper_close_below):
            delta_gripper_normalized = 0.0
        else:
            delta_gripper_normalized = normalize_dataset_gripper(
                raw_delta_gripper,
                robot_cfg,
                feature_name=delta_gripper_feature,
            )
        return {
            'ee.x': float(rebuilt['ee.x']),
            'ee.y': float(rebuilt['ee.y']),
            'ee.z': float(rebuilt['ee.z']),
            'ee.wx': float(rebuilt_rotvec[0]),
            'ee.wy': float(rebuilt_rotvec[1]),
            'ee.wz': float(rebuilt_rotvec[2]),
            'gripper.pos': delta_gripper_normalized,
        }

    quaternion_xyzw = np.asarray(
        [
            _action_value(action_map, 'qx', 'ee.qx'),
            _action_value(action_map, 'qy', 'ee.qy'),
            _action_value(action_map, 'qz', 'ee.qz'),
            _action_value(action_map, 'qw', 'ee.qw'),
        ],
        dtype=np.float64,
    )
    rotvec_xyz = Rotation.from_quat(quaternion_xyzw).as_rotvec()
    raw_gripper_value, gripper_feature = _action_value_with_name(
        action_map,
        'gripper',
        'gripper.pos',
        'observation.state.gripper.pos',
        'observation.state_raw.handheld_gripper.pika_left.width_mm',
        'observation.state_raw.handheld_gripper.pika_right.width_mm',
        'corenetic_gripper.distance_m',
        'observation.state_raw.corenetic_gripper.distance_m',
        'box_gripper.distance_m',
        'observation.state_raw.box_gripper.distance_m',
    )
    if gripper_close_below is not None and raw_gripper_value < float(gripper_close_below):
        gripper_normalized = 0.0
    else:
        gripper_normalized = normalize_dataset_gripper(
            raw_gripper_value,
            robot_cfg,
            feature_name=gripper_feature,
        )

    return {
        'ee.x': _action_value(action_map, 'x', 'ee.x'),
        'ee.y': _action_value(action_map, 'y', 'ee.y'),
        'ee.z': _action_value(action_map, 'z', 'ee.z'),
        'ee.wx': float(rotvec_xyz[0]),
        'ee.wy': float(rotvec_xyz[1]),
        'ee.wz': float(rotvec_xyz[2]),
        'gripper.pos': gripper_normalized,
    }


def _pose_from_rotvec_command(robot_command: dict[str, float]) -> np.ndarray:
    return _pose_from_position_and_rotvec(
        np.asarray([robot_command['ee.x'], robot_command['ee.y'], robot_command['ee.z']], dtype=np.float64),
        np.asarray([robot_command['ee.wx'], robot_command['ee.wy'], robot_command['ee.wz']], dtype=np.float64),
    )


def extract_new_action_chunk_for_visualization(
    policy: Any,
    current_action_tensor: torch.Tensor,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> list[torch.Tensor] | None:
    temporal_ensembler = getattr(policy, 'temporal_ensembler', None)
    ensembled_actions = getattr(temporal_ensembler, 'ensembled_actions', None)
    if ensembled_actions is not None:
        action_tensors = [current_action_tensor.detach().cpu()]
        try:
            preview_postprocessor = deepcopy(postprocessor)
            remaining_actions = ensembled_actions.detach()
            for action_idx in range(remaining_actions.shape[1]):
                processed_action = preview_postprocessor(remaining_actions[:, action_idx, :])
                action_tensors.append(processed_action.detach().cpu())
        except Exception as exc:
            print(f'[WARN] mujoco_temporal_ensemble_preview=unavailable reason={type(exc).__name__}: {exc}')
            return None
        return action_tensors

    action_queue = getattr(policy, '_action_queue', None)
    n_action_steps = getattr(getattr(policy, 'config', None), 'n_action_steps', None)
    if action_queue is None or n_action_steps is None:
        return None
    if len(action_queue) != max(int(n_action_steps) - 1, 0):
        return None

    action_tensors = [current_action_tensor.detach().cpu()]
    if len(action_queue) == 0:
        return action_tensors

    try:
        preview_postprocessor = deepcopy(postprocessor)
        for raw_action in list(action_queue):
            processed_action = preview_postprocessor(raw_action)
            action_tensors.append(processed_action.detach().cpu())
    except Exception as exc:
        print(f'[WARN] mujoco_chunk_preview=unavailable reason={type(exc).__name__}: {exc}')
        return None
    return action_tensors



def extract_action_queue_for_visualization(
    action_queue: ActionQueue,
    current_action_tensor: torch.Tensor,
) -> list[torch.Tensor]:
    action_tensors = [current_action_tensor.detach().cpu()]
    with action_queue.lock:
        if action_queue.queue is None:
            return action_tensors
        for queued_action in action_queue.queue[action_queue.last_index :]:
            action_tensors.append(queued_action.detach().cpu())
    return action_tensors

def select_temporal_ensemble_offset_action(
    action_tensor: torch.Tensor,
    *,
    policy: Any,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    offset: int,
) -> torch.Tensor:
    offset = int(offset)
    if offset <= 0:
        return action_tensor
    temporal_ensembler = getattr(policy, 'temporal_ensembler', None)
    ensembled_actions = getattr(temporal_ensembler, 'ensembled_actions', None)
    if ensembled_actions is None:
        raise ValueError('--act-temporal-action-offset requires ACT temporal ensembling to be enabled.')
    if ensembled_actions.shape[1] <= 0:
        return action_tensor

    future_index = min(offset - 1, int(ensembled_actions.shape[1]) - 1)
    raw_future_action = ensembled_actions[:, future_index, :]
    return postprocessor(raw_future_action)


def resolve_temporal_ensemble_max_offset(policy: Any, requested_max_offset: int | None) -> int | None:
    if requested_max_offset is None:
        return None
    requested_max_offset = int(requested_max_offset)
    if requested_max_offset <= 0:
        return None
    temporal_ensembler = getattr(policy, 'temporal_ensembler', None)
    ensembled_actions = getattr(temporal_ensembler, 'ensembled_actions', None)
    if ensembled_actions is None or int(ensembled_actions.shape[1]) <= 0:
        return requested_max_offset
    return min(requested_max_offset, int(ensembled_actions.shape[1]))


def update_temporal_offset_on_stuck(
    temporal_offset_state: dict[str, int],
    *,
    base_offset: int,
    max_offset: int | None,
    offset_step: int,
    stuck_steps: int,
    stuck_pos_delta_m: float,
    closed_gripper_max: float,
    unassisted_command: dict[str, float],
    robot_observation: RobotObservation,
) -> dict[str, Any]:
    if max_offset is None:
        temporal_offset_state['current_offset'] = max(int(base_offset), 0)
        temporal_offset_state['stuck_count'] = 0
        return {'status': 'disabled', 'current_offset': int(temporal_offset_state['current_offset']), 'stuck_count': 0}

    base_offset = max(int(base_offset), 0)
    max_offset = max(int(max_offset), base_offset)
    current_offset = int(temporal_offset_state.get('current_offset', base_offset))
    stuck_count = int(temporal_offset_state.get('stuck_count', 0))

    position_delta, _ = compute_pose_delta_from_current(unassisted_command, robot_observation)
    unassisted_delta_m = float(np.linalg.norm(position_delta))
    closed_gripper = float(unassisted_command['gripper.pos']) <= float(closed_gripper_max)
    low_motion = unassisted_delta_m <= float(stuck_pos_delta_m)

    if closed_gripper and low_motion:
        stuck_count += 1
        if stuck_count >= max(int(stuck_steps), 1):
            current_offset = min(max_offset, current_offset + max(int(offset_step), 1))
            stuck_count = 0
            status = 'advance'
        else:
            status = f'waiting:{stuck_count}/{max(int(stuck_steps), 1)}'
    else:
        if current_offset != base_offset:
            status = 'reset_moving'
        else:
            status = 'base'
        current_offset = base_offset
        stuck_count = 0

    temporal_offset_state['current_offset'] = current_offset
    temporal_offset_state['stuck_count'] = stuck_count
    return {
        'status': status,
        'current_offset': current_offset,
        'stuck_count': stuck_count,
        'unassisted_delta_m': unassisted_delta_m,
        'closed_gripper': closed_gripper,
    }


def build_chunk_ee_poses_for_visualization(
    action_tensors: list[torch.Tensor] | None,
    *,
    action_names: list[str],
    robot_cfg: FrankaResearch3Config,
    T_B_Ws: np.ndarray,
    delta_reference: str | None = None,
    dataset_observation_i: RobotObservation | None = None,
) -> list[np.ndarray] | None:
    """Absolute EE poses for a whole predicted action chunk, for the MuJoCo preview only.

    A delta chunk cannot be decoded frame-independently: each future action's reference is the
    pose the previous one commanded, so the chunk is *integrated* forward from the current
    observation. For ``delta_ee_from_current`` that is an approximation -- the future measured
    poses do not exist yet -- which is acceptable here because this feeds a preview overlay and
    never a robot command.
    """
    if not action_tensors:
        return None
    if delta_reference is not None and dataset_observation_i is None:
        return None
    # A fresh reconstructor per call: the chunk is a hypothetical future, and its quaternion-sign
    # history must not leak into the live control path's reconstructor.
    chunk_reconstructor = (
        DeltaEEToAbsoluteEEAction(reference=delta_reference) if delta_reference is not None else None
    )
    rolling_observation = dict(dataset_observation_i) if dataset_observation_i is not None else None
    chunk_ee_poses: list[np.ndarray] = []
    for action_tensor in action_tensors:
        dataset_robot_command_i = decode_action_to_robot_command(
            action_tensor,
            action_names=action_names,
            robot_cfg=robot_cfg,
            delta_reconstructor=chunk_reconstructor,
            dataset_observation_i=rolling_observation,
        )
        if chunk_reconstructor is not None and rolling_observation is not None:
            # Feed the rebuilt target forward as the next frame's reference.
            rolling_quaternion = Rotation.from_rotvec(
                [
                    dataset_robot_command_i['ee.wx'],
                    dataset_robot_command_i['ee.wy'],
                    dataset_robot_command_i['ee.wz'],
                ]
            ).as_quat()
            for keys, values in (
                (PREV_CMD_POSITION_KEYS, ('ee.x', 'ee.y', 'ee.z')),
                (EE_POSITION_KEYS, ('ee.x', 'ee.y', 'ee.z')),
            ):
                for key, source in zip(keys, values, strict=True):
                    rolling_observation[key] = float(dataset_robot_command_i[source])
            for keys in (PREV_CMD_QUAT_KEYS, EE_QUAT_KEYS):
                for key, value in zip(keys, rolling_quaternion, strict=True):
                    rolling_observation[key] = float(value)
        base_robot_command_i = convert_dataset_command_to_base_frame(dataset_robot_command_i, T_B_Ws)
        robot_command_e = convert_base_command_from_I_to_E(base_robot_command_i)
        chunk_ee_poses.append(_pose_from_rotvec_command(robot_command_e))
    return chunk_ee_poses


def build_hold_command(
    robot_observation: RobotObservation,
    *,
    gripper_pos_override: float | None = None,
) -> dict[str, float]:
    return {
        'ee.x': float(robot_observation['ee.x']),
        'ee.y': float(robot_observation['ee.y']),
        'ee.z': float(robot_observation['ee.z']),
        'ee.wx': float(robot_observation['ee.wx']),
        'ee.wy': float(robot_observation['ee.wy']),
        'ee.wz': float(robot_observation['ee.wz']),
        'gripper.pos': float(
            robot_observation['gripper.pos'] if gripper_pos_override is None else np.clip(gripper_pos_override, 0.0, 1.0)
        ),
    }


def align_gripper_to_dataset_start(
    robot: Any,
    *,
    target_gripper_pos: float,
    tolerance: float,
    max_wait_s: float = 8.0,
    poll_interval_s: float = 0.1,
) -> RobotObservation:
    start_t = time.perf_counter()
    robot_observation = robot.get_observation()
    current_gripper_pos = float(robot_observation['gripper.pos'])
    initial_abs_delta = abs(current_gripper_pos - target_gripper_pos)
    print(
        '[INFO] gripper_start_alignment='
        f'current={current_gripper_pos:.3f} target={target_gripper_pos:.3f} '
        f'abs_delta={initial_abs_delta:.3f} tol={tolerance:.3f}'
    )
    if initial_abs_delta <= tolerance:
        print('[INFO] gripper_start_alignment_status=already_within_tolerance')
        return robot_observation

    deadline = start_t + max_wait_s
    while True:
        robot.send_action(build_hold_command(robot_observation, gripper_pos_override=target_gripper_pos))
        precise_sleep(poll_interval_s)
        robot_observation = robot.get_observation()
        current_gripper_pos = float(robot_observation['gripper.pos'])
        current_abs_delta = abs(current_gripper_pos - target_gripper_pos)
        if current_abs_delta <= tolerance:
            print(
                '[INFO] gripper_start_alignment_status=done '
                f'current={current_gripper_pos:.3f} target={target_gripper_pos:.3f} '
                f'abs_delta={current_abs_delta:.3f} elapsed_s={time.perf_counter() - start_t:.2f}'
            )
            return robot_observation
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                'Timed out aligning gripper to dataset-start mean: '
                f'current={current_gripper_pos:.3f} target={target_gripper_pos:.3f} '
                f'abs_delta={current_abs_delta:.3f} tol={tolerance:.3f}'
            )



def apply_gripper_observation_offset(
    dataset_observation_i: RobotObservation,
    *,
    gripper_offset: float | None,
) -> RobotObservation:
    corrected_observation = dict(dataset_observation_i)
    if gripper_offset is None:
        return corrected_observation
    corrected_observation['gripper.pos'] = float(
        np.clip(float(dataset_observation_i['gripper.pos']) + float(gripper_offset), 0.0, 1.0)
    )
    return corrected_observation


def apply_place_assist_offset(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
    assist_state: dict[str, Any],
    *,
    target_offset_xyz_m: np.ndarray | None,
    stuck_steps: int,
    stuck_pos_delta_m: float,
    ramp_step_m: float,
    closed_gripper_max: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if target_offset_xyz_m is None:
        return dict(robot_command), {'status': 'disabled', 'offset_xyz_m': np.zeros(3, dtype=np.float64)}

    target_offset_xyz_m = np.asarray(target_offset_xyz_m, dtype=np.float64)
    if target_offset_xyz_m.shape != (3,):
        raise ValueError('--place-assist-offset-base-xyz expects exactly 3 values.')

    current_offset = np.asarray(assist_state.get('offset_xyz_m', np.zeros(3, dtype=np.float64)), dtype=np.float64)
    stuck_count = int(assist_state.get('stuck_count', 0))
    position_delta, _ = compute_pose_delta_from_current(robot_command, robot_observation)
    unassisted_delta_m = float(np.linalg.norm(position_delta))
    closed_gripper = float(robot_command['gripper.pos']) <= float(closed_gripper_max)
    low_motion = unassisted_delta_m <= float(stuck_pos_delta_m)

    if not closed_gripper:
        stuck_count = 0
        current_offset = np.zeros(3, dtype=np.float64)
        status = 'reset_open_gripper'
    elif low_motion:
        stuck_count += 1
        if stuck_count >= max(int(stuck_steps), 1):
            remaining = target_offset_xyz_m - current_offset
            remaining_norm = float(np.linalg.norm(remaining))
            max_step = max(float(ramp_step_m), 0.0)
            if remaining_norm > 0.0 and max_step > 0.0:
                scale = min(1.0, max_step / remaining_norm)
                current_offset = current_offset + remaining * scale
            status = 'active'
        else:
            status = f'waiting:{stuck_count}/{max(int(stuck_steps), 1)}'
    else:
        stuck_count = 0
        status = 'armed_moving'

    assist_state['offset_xyz_m'] = current_offset
    assist_state['stuck_count'] = stuck_count

    assisted_command = dict(robot_command)
    assisted_command['ee.x'] = float(assisted_command['ee.x'] + current_offset[0])
    assisted_command['ee.y'] = float(assisted_command['ee.y'] + current_offset[1])
    assisted_command['ee.z'] = float(assisted_command['ee.z'] + current_offset[2])
    return assisted_command, {
        'status': status,
        'offset_xyz_m': current_offset,
        'stuck_count': stuck_count,
        'unassisted_delta_m': unassisted_delta_m,
        'closed_gripper': closed_gripper,
    }


def apply_gripper_change_delay(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
    latch_state: dict[str, float | None],
    *,
    delay_s: float | None,
    min_delta: float,
    settle_tolerance: float,
    settle_timeout_s: float,
) -> tuple[dict[str, float], dict[str, float | str]]:
    desired = float(np.clip(robot_command['gripper.pos'], 0.0, 1.0))
    observed = float(np.clip(float(robot_observation['gripper.pos']), 0.0, 1.0))
    debug: dict[str, float | str] = {
        'observed': observed,
        'desired': desired,
        'delta': abs(desired - observed),
        'status': 'disabled',
    }
    if delay_s is None:
        command = dict(robot_command)
        command['gripper.pos'] = desired
        debug['command'] = desired
        debug['latched'] = desired
        debug['latch_error'] = abs(observed - desired)
        return command, debug

    delay_s = max(0.0, float(delay_s))
    min_delta = max(0.0, float(min_delta))
    settle_tolerance = max(0.0, float(settle_tolerance))
    settle_timeout_s = max(delay_s, float(settle_timeout_s))
    latched = latch_state.get('command')
    if latched is None:
        latched = observed
        latch_state['command'] = latched

    latched = float(np.clip(float(latched), 0.0, 1.0))
    last_change_time_s = latch_state.get('last_change_time_s')
    now_s = time.perf_counter()
    elapsed_s = float('inf') if last_change_time_s is None else now_s - float(last_change_time_s)
    latch_error = abs(observed - latched)
    debug['latched'] = latched
    debug['latch_error'] = latch_error

    if last_change_time_s is not None and elapsed_s < delay_s:
        command = dict(robot_command)
        command['gripper.pos'] = latched
        debug['command'] = latched
        debug['status'] = f'locked_discard:{delay_s - elapsed_s:.2f}s'
        return command, debug

    if last_change_time_s is not None and latch_error > settle_tolerance and elapsed_s < settle_timeout_s:
        command = dict(robot_command)
        command['gripper.pos'] = latched
        debug['command'] = latched
        debug['status'] = f'settling_discard:{settle_timeout_s - elapsed_s:.2f}s'
        return command, debug

    desired_delta_from_observed = abs(desired - observed)
    desired_delta_from_latched = abs(desired - latched)
    should_change = desired_delta_from_observed >= min_delta and desired_delta_from_latched >= 1e-6

    if should_change:
        latched = desired
        latch_state['command'] = latched
        latch_state['last_change_time_s'] = now_s
        debug['status'] = 'change'
    else:
        debug['status'] = 'hold'

    command = dict(robot_command)
    command['gripper.pos'] = latched
    debug['command'] = latched
    return command, debug


def should_reject_first_command(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
    *,
    max_pos_delta_m: float,
    max_rot_delta_rad: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    position_delta, rotation_delta = compute_pose_delta_from_current(robot_command, robot_observation)
    reject = bool(
        np.any(np.abs(position_delta) > float(max_pos_delta_m))
        or np.any(np.abs(rotation_delta) > float(max_rot_delta_rad))
    )
    return reject, position_delta, rotation_delta


def _format_vector(values: np.ndarray, *, scale: float = 1.0) -> str:
    return ', '.join(f'{component * scale:.2f}' for component in values)



def _sanitize_feature_key(feature_key: str) -> str:
    return feature_key.replace('/', '_').replace('.', '__')


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(item) for item in value]
    return value


def _extract_numeric_observation_scalars(observation: RobotObservation) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            continue
        if isinstance(value, (np.floating, float)):
            scalars[key] = float(value)
            continue
        if isinstance(value, (np.integer, int)):
            scalars[key] = int(value)
            continue
        if isinstance(value, (np.bool_, bool)):
            scalars[key] = bool(value)
    return scalars


def dump_step0_capture_bundle(
    output_dir: Path,
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    T_B_Ws: np.ndarray,
    start_alignment_stats: dict[str, Any],
    state_names: list[str],
    action_names: list[str],
    policy_observation: dict[str, np.ndarray],
    robot_observation: RobotObservation,
    absolute_state_observation_e: RobotObservation,
    absolute_state_observation_i: RobotObservation,
    dataset_state_observation_i: RobotObservation,
) -> Path:
    from PIL import Image

    output_dir = _resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_files: dict[str, str] = {}
    image_files: dict[str, str] = {}
    for feature_key, value in sorted(policy_observation.items()):
        array_value = np.asarray(value)
        array_filename = f'{_sanitize_feature_key(feature_key)}.npy'
        np.save(output_dir / array_filename, array_value)
        observation_files[feature_key] = array_filename
        if feature_key.startswith('observation.images.'):
            image = np.asarray(array_value, dtype=np.uint8)
            image_filename = f'{_sanitize_feature_key(feature_key)}.png'
            Image.fromarray(image).save(output_dir / image_filename)
            image_files[feature_key] = image_filename

    metadata = {
        'checkpoint': str(checkpoint),
        'dataset_root': str(dataset_root),
        'dataset_repo_id': dataset_repo_id,
        'state_names': list(state_names),
        'action_names': list(action_names),
        'T_B_Ws': _jsonify_value(T_B_Ws),
        'start_alignment_stats': _jsonify_value(start_alignment_stats),
        'policy_observation_files': observation_files,
        'policy_image_files': image_files,
        'robot_observation_scalars': _extract_numeric_observation_scalars(robot_observation),
        'absolute_state_observation_e_scalars': _extract_numeric_observation_scalars(absolute_state_observation_e),
        'absolute_state_observation_i_scalars': _extract_numeric_observation_scalars(absolute_state_observation_i),
        'dataset_state_observation_i_scalars': _extract_numeric_observation_scalars(dataset_state_observation_i),
    }
    if 'observation.state' in policy_observation:
        metadata['policy_observation_state'] = _jsonify_value(
            np.asarray(policy_observation['observation.state'], dtype=np.float64)
        )
    else:
        metadata['policy_observation_state'] = None
        metadata['policy_observation_state_note'] = 'not present; policy appears to be image-only'
    (output_dir / 'metadata.json').write_text(json.dumps(_jsonify_value(metadata), indent=2), encoding='utf-8')
    return output_dir


def dump_step0_action_debug(
    output_dir: Path,
    *,
    action_names: list[str],
    action_tensor: torch.Tensor,
    dataset_robot_command_i: dict[str, float],
    base_robot_command_i: dict[str, float],
    robot_command: dict[str, float],
    command_to_send: dict[str, float],
    command_status: str,
    position_delta: np.ndarray,
    rotation_delta: np.ndarray,
    clamped: bool,
) -> None:
    output_dir = _resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    payload = {
        'action_names': list(action_names),
        'action_values': {name: float(action_np[idx]) for idx, name in enumerate(action_names)},
        'dataset_robot_command_i': _jsonify_value(dataset_robot_command_i),
        'base_robot_command_i': _jsonify_value(base_robot_command_i),
        'raw_robot_command_e': _jsonify_value(robot_command),
        'safe_command_e': _jsonify_value(command_to_send),
        'command_status': str(command_status),
        'clamped': bool(clamped),
        'position_delta_m': _jsonify_value(position_delta),
        'position_delta_mm': _jsonify_value(np.asarray(position_delta, dtype=np.float64) * 1000.0),
        'rotation_delta_rad': _jsonify_value(rotation_delta),
        'rotation_delta_deg': _jsonify_value(np.rad2deg(rotation_delta)),
    }
    (output_dir / 'step0_action_debug.json').write_text(
        json.dumps(_jsonify_value(payload), indent=2),
        encoding='utf-8',
    )



def _policy_config_type(policy_cfg: Any) -> str:
    policy_type = getattr(policy_cfg, "type", "")
    if callable(policy_type):
        try:
            policy_type = policy_type()
        except TypeError:
            policy_type = ""
    return str(policy_type or "").strip().lower()


def should_enable_rtc_for_policy(policy_cfg: Any, rtc_mode: str) -> bool:
    mode = str(rtc_mode or "disabled").strip().lower()
    if mode not in _RTC_MODE_CHOICES:
        raise ValueError(f"--rtc-mode must be one of {_RTC_MODE_CHOICES}, got {rtc_mode!r}.")
    supports_rtc = hasattr(policy_cfg, "rtc_config") and _policy_config_type(policy_cfg) in _RTC_POLICY_TYPES
    if mode == "disabled":
        return False
    if mode == "enabled" and not supports_rtc:
        raise ValueError(
            f"--rtc-mode=enabled was requested, but policy type {_policy_config_type(policy_cfg)!r} "
            "does not support RTC in this runtime. Use --rtc-mode=auto to keep unsupported policies on their "
            "checkpoint queue."
        )
    return supports_rtc


def configure_rtc_for_policy_config(
    policy_cfg: Any,
    *,
    rtc_mode: str,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
) -> bool:
    enabled = should_enable_rtc_for_policy(policy_cfg, rtc_mode)
    policy_type = _policy_config_type(policy_cfg)
    if not enabled:
        print(f"[INFO] rtc=disabled mode={rtc_mode} policy_type={policy_type or '<unknown>'}")
        return False

    execution_horizon = int(execution_horizon)
    if execution_horizon <= 0:
        raise ValueError("--rtc-execution-horizon must be > 0 when RTC is enabled.")
    max_guidance_weight = float(max_guidance_weight)
    if max_guidance_weight <= 0.0:
        raise ValueError("--rtc-max-guidance-weight must be > 0 when RTC is enabled.")
    schedule = RTCAttentionSchedule(str(prefix_attention_schedule).upper())
    policy_cfg.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=schedule,
    )
    print(
        "[INFO] rtc=enabled "
        f"mode={rtc_mode} policy_type={policy_type} "
        f"execution_horizon={execution_horizon} "
        f"max_guidance_weight={max_guidance_weight:.6g} "
        f"prefix_attention_schedule={schedule.value}"
    )
    return True


def disable_policy_compile_for_online_rollout(policy_cfg: Any) -> None:
    """Keep training-time torch.compile settings out of the real-time rollout path.

    Training checkpoints can legitimately carry ``compile_model=True``. Reusing that
    flag during online robot control is a different contract: the first forward pass
    may spend tens of seconds in TorchInductor compile/autotune and RTC will then
    treat the whole chunk as stale. Rollout should be latency-predictable by default.
    """
    if bool(getattr(policy_cfg, 'compile_model', False)):
        compile_mode = getattr(policy_cfg, 'compile_mode', '<unset>')
        policy_cfg.compile_model = False
        print(
            '[INFO] policy_compile_model_override='
            f'from=True to=False compile_mode={compile_mode} reason=online_rollout_latency'
        )


def _clamp_rtc_delay_steps(delay_steps: int, chunk_len: int) -> int:
    delay_steps = max(int(delay_steps), 0)
    chunk_len = int(chunk_len)
    if chunk_len <= 0:
        return 0
    if delay_steps >= chunk_len:
        clamped = max(chunk_len - 1, 0)
        print(
            "[WARN] rtc_delay_exceeds_chunk "
            f"delay_steps={delay_steps} chunk_len={chunk_len}; using delay_steps={clamped} so one action remains."
        )
        return clamped
    return delay_steps


def resolve_rtc_replan_queue_size(policy: Any, requested_replan_queue_size: int) -> int:
    chunk_size = int(getattr(getattr(policy, "config", None), "chunk_size", 1))
    if chunk_size <= 1:
        return 0
    requested = max(int(requested_replan_queue_size), 0)
    return min(requested, chunk_size - 1)



def build_expert_takeover(args: argparse.Namespace, *, step_period_s: float) -> ExpertTakeover | None:
    """Connect the SpaceMouse the operator will steer with, or explain why there is none.

    Imported here rather than at module scope so a rig with no HID library, or no device
    plugged in, still runs ordinary rollouts. Takeover is the addition; it must not become a
    new way for inference to fail to start.
    """
    if not args.dagger_takeover:
        return None
    from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

    overrides: dict[str, Any] = {'device_id': int(args.dagger_spacemouse_device_id)}
    if args.dagger_translation_scale is not None:
        overrides['translation_scale'] = float(args.dagger_translation_scale)
    if args.dagger_rotation_scale is not None:
        overrides['rotation_scale'] = float(args.dagger_rotation_scale)
    teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(**overrides))
    teleop.connect()
    # The backstop. The pre-flight above already refused this run before the arm existed; this one
    # asks the object actually being handed to the takeover, so no future call path can reach the
    # rig by skipping the early gate. The device is released before the exit so a second attempt
    # finds it free.
    if not backend_dates_reports(teleop):
        driver_module = getattr(type(teleop), '__module__', None)
        driver_file = getattr(sys.modules.get(driver_module), '__file__', None)
        try:
            teleop.disconnect()
        except Exception:  # noqa: BLE001 - already failing; the exit message is what matters
            pass
        raise SystemExit('--dagger-takeover refused: ' + undated_backend_error(driver_module=driver_file))
    release_after_s = float(args.dagger_takeover_release_after_s)
    # `frequency` is the rate the scales are calibrated against, not the device's report rate --
    # see `motion_gain_for`. This loop runs slower, so one reading has to cover proportionally
    # more ground to move the arm at the speed the recorder moved it. `step_period_s` is handed
    # over as well as folded into the gain, because it is the rate this loop is *supposed* to hold
    # and the rate it holds are two different numbers on a rig doing real inference: the takeover
    # measures its own steps and corrects the gain by what it finds, so the operator gets the
    # recorded speed rather than the nominal one.
    motion_gain = motion_gain_for(tick_hz=float(teleop.config.frequency), step_period_s=step_period_s)
    print(
        '[INFO] dagger_takeover=ready '
        f"device_id={overrides['device_id']} "
        f"translation_scale={teleop.config.translation_scale:.6f} "
        f"rotation_scale={teleop.config.rotation_scale:.6f} "
        f'release_after_s={release_after_s:.2f} '
        f'motion_gain={motion_gain:.2f} '
        f'nominal_step_ms={step_period_s * 1000.0:.1f} '
        f'full_deflection_mm_per_step={teleop.config.translation_scale * motion_gain * 1000.0:.1f} '
        # Always yes by the time this prints -- the refusal above is the only other outcome. It is
        # printed anyway because the operator's pre-flight is this banner, and "the field is there
        # and says yes" is a check they can make; an absent field is not.
        'report_timestamps=yes'
    )
    return ExpertTakeover(
        teleop,
        release_after_s=release_after_s,
        motion_gain=motion_gain,
        step_period_s=step_period_s,
    )


def build_dagger_writer(
    args: argparse.Namespace,
    *,
    ds_meta: LeRobotDatasetMetadata,
    fps: float,
) -> tuple[Any, DaggerEpisodeWriter] | None:
    """Open (or extend) the dataset the corrections will be written to.

    The schema is the imitated dataset's own plus ``is_intervention``: a DAgger episode is only
    worth anything if it is shaped like the demonstrations it will be trained beside, and any
    other way of deriving the schema is a way for the two to drift apart.

    Extending rather than recreating is deliberate and is the recorder's own behaviour -- a
    second session of corrections belongs in the same place as the first, and
    ``LeRobotDataset.create`` on an existing root raises rather than appending.
    """
    if not args.dagger_takeover or args.dagger_dataset_root is None:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dagger_dataset_root).expanduser()
    repo_id = str(args.dagger_dataset_repo_id or root.name)
    features = dagger_dataset_features(dict(ds_meta.features))

    def create_dataset() -> Any:
        dataset = LeRobotDataset.create(
            repo_id,
            fps=int(round(fps)),
            root=root,
            features=features,
            use_videos=True,
        )
        print(f'[INFO] dagger_dataset=created root={root} repo_id={repo_id} fps={int(round(fps))}')
        return dataset

    if root.exists() and dagger_dataset_can_load_locally(root):
        dataset = LeRobotDataset(repo_id, root=root)
        # The frames written below are built against `features`, derived from the dataset being
        # imitated. If the root points at a dataset built from something else, every column
        # would be filled from the wrong place -- and a schema mismatch that only shows up as a
        # badly trained policy is worth failing on here instead.
        mismatched = sorted(set(features) ^ set(dataset.meta.features))
        if mismatched:
            raise SystemExit(
                f'--dagger-dataset-root {root} holds a dataset whose schema differs from '
                f'{ds_meta.repo_id}: {mismatched}. Point it at a fresh directory, or at the '
                'DAgger dataset recorded against this same view.'
            )
        print(
            f'[INFO] dagger_dataset=extending root={root} repo_id={repo_id} '
            f'episodes={dataset.meta.total_episodes}'
        )
    else:
        if root.exists():
            if not dagger_dataset_root_is_recreatable(root):
                if dagger_dataset_is_unfinalized(root):
                    raise SystemExit(
                        f'--dagger-dataset-root {root} holds a DAgger session that was killed '
                        'before it closed its dataset: the frames and videos are on disk, but the '
                        'episode metadata was never flushed and the data parquet has no footer, '
                        'so nothing can open it. Move it aside and start a fresh directory -- '
                        'those corrections cannot be recovered.'
                    )
                raise SystemExit(
                    f'--dagger-dataset-root {root} exists but is not a loadable LeRobot dataset '
                    'and contains files other than an empty DAgger metadata shell. Move it aside, '
                    'or point --dagger-dataset-root at a fresh directory.'
                )
            shutil.rmtree(root)
        dataset = create_dataset()
    return dataset, DaggerEpisodeWriter(dataset, min_span_frames=int(args.dagger_min_span_frames))


def build_dagger_action_encoder(
    action_names: list[str],
    *,
    robot_cfg: FrankaResearch3Config,
    gripper_feature_name: str | None,
) -> tuple[Any, Callable[[float], float]]:
    """The two pieces the writer cannot import: the delta encoder and the gripper's units.

    ``AbsoluteEEToDeltaEEAction`` is the step the *recorder* used to turn a command into the
    delta it stored. Using it here rather than a second implementation is what makes a DAgger
    sample and a demonstration sample the same arithmetic; anything else would be a second
    definition of the action space, discoverable only as a policy that has learned an offset.
    """
    reference = delta_reference_from_action_names(action_names)
    encode_delta: Callable[[dict[str, float], dict[str, float]], dict[str, float]] | None = None
    if reference is not None:
        from lerobot.robots.franka_research3.processor_franka_research3 import (
            AbsoluteEEToDeltaEEAction,
        )

        encoder = AbsoluteEEToDeltaEEAction(reference=reference)

        def encode_delta(absolute_action, dataset_observation_i):  # noqa: F811
            return encoder(
                {
                    TransitionKey.ACTION: dict(absolute_action),
                    TransitionKey.OBSERVATION: dict(dataset_observation_i),
                }
            )[TransitionKey.ACTION]

    def denormalize_gripper(value: float) -> float:
        return denormalize_live_gripper_observation(value, robot_cfg, feature_name=gripper_feature_name)

    return encode_delta, denormalize_gripper


def resolve_rollout_task_prompt(ds_meta: LeRobotDatasetMetadata, explicit_task_prompt: str | None) -> str | None:
    if explicit_task_prompt is not None and str(explicit_task_prompt).strip():
        return str(explicit_task_prompt).strip()

    tasks = getattr(ds_meta, "tasks", None)
    if tasks is None:
        return None
    task_prompts = [str(task).strip() for task in list(tasks.index) if str(task).strip()]
    unique_prompts = list(dict.fromkeys(task_prompts))
    if len(unique_prompts) == 1:
        return unique_prompts[0]
    if len(unique_prompts) > 1:
        raise ValueError(
            "This checkpoint dataset/view contains multiple task prompts; pass --task-prompt explicitly "
            f"to avoid running pi0/pi0.5 with the wrong language condition. Prompts: {unique_prompts}"
        )
    return None

def predict_action_chunk_for_rollout(
    observation: dict[str, np.ndarray],
    *,
    policy: Any,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    inference_delay: int,
    prev_chunk_left_over: torch.Tensor | None,
    execution_horizon: int,
    task: str | None = None,
    robot_type: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict one action chunk for a queue-driven rollout.

    The returned first tensor is still in policy/raw action space, because RTC uses it as the
    previous-chunk prefix on the next inference. The second tensor is already postprocessed into
    the dataset action contract and can be decoded one step at a time against the live robot state.
    """
    observation = dict(observation)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        prepared_observation = prepare_observation_for_inference(
            observation,
            device,
            task=task,
            robot_type=robot_type,
        )
        preprocessed_observation = preprocessor(prepared_observation)
        actions = policy.predict_action_chunk(
            preprocessed_observation,
            inference_delay=int(inference_delay),
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=int(execution_horizon),
        )
        original_actions = actions.squeeze(0).detach().clone()
        processed_actions = postprocessor(actions).squeeze(0).detach().cpu().clone()
    return original_actions, processed_actions


def _clone_policy_observation_for_async(observation: dict[str, np.ndarray]) -> dict[str, Any]:
    """Own the observation buffers before handing them to a background planner thread."""
    cloned: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        elif torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = deepcopy(value)
    return cloned


@dataclass(frozen=True)
class AsyncActionChunkPlanResult:
    original_actions: torch.Tensor
    processed_actions: torch.Tensor
    latency_s: float
    action_index_before_inference: int
    guidance_delay_steps: int
    observation_step: int


class AsyncActionChunkPlanner:
    """Run slow chunk inference off the 30 Hz robot command loop.

    RTC assumes the robot keeps executing the previous chunk while the next chunk is being
    predicted. If inference runs synchronously inside the command loop, no actions are actually
    sent during that time; the arm catches the current target and visually pauses. This helper keeps
    the physical control loop causal: one thread plans, the main loop continues sending setpoints.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: AsyncActionChunkPlanResult | None = None
        self._error: BaseException | None = None

    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        observation: dict[str, np.ndarray],
        *,
        predict_kwargs: dict[str, Any],
        action_index_before_inference: int,
        guidance_delay_steps: int,
        observation_step: int,
    ) -> bool:
        observation_snapshot = _clone_policy_observation_for_async(observation)

        def _target() -> None:
            started_at_s = time.perf_counter()
            result: AsyncActionChunkPlanResult | None = None
            error: BaseException | None = None
            try:
                original_actions, processed_actions = predict_action_chunk_for_rollout(
                    observation_snapshot,
                    **predict_kwargs,
                )
                result = AsyncActionChunkPlanResult(
                    original_actions=original_actions,
                    processed_actions=processed_actions,
                    latency_s=time.perf_counter() - started_at_s,
                    action_index_before_inference=int(action_index_before_inference),
                    guidance_delay_steps=int(guidance_delay_steps),
                    observation_step=int(observation_step),
                )
            except BaseException as exc:  # pragma: no cover - propagated on the main thread
                error = exc
            with self._lock:
                self._result = result
                self._error = error

        with self._lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    return False
                # A completed result has to be merged before another request can replace it.
                return False
            self._result = None
            self._error = None
            self._thread = threading.Thread(target=_target, daemon=True, name='RTCActionChunkPlanner')
            self._thread.start()
            return True

    def pop_completed(self) -> AsyncActionChunkPlanResult | None:
        with self._lock:
            thread = self._thread
            if thread is None or thread.is_alive():
                return None
            result = self._result
            error = self._error
            self._thread = None
            self._result = None
            self._error = None
        thread.join(timeout=0.0)
        if error is not None:
            raise RuntimeError('Asynchronous RTC chunk prediction failed.') from error
        if result is None:
            raise RuntimeError('Asynchronous RTC chunk prediction finished without producing actions.')
        return result

    def join(self, timeout_s: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout_s)
        return not thread.is_alive()


def merge_completed_rtc_plan(
    action_queue: ActionQueue,
    result: AsyncActionChunkPlanResult,
    latency_tracker: LatencyTracker,
) -> dict[str, Any]:
    """Merge an async RTC plan using actions actually consumed during inference.

    Wall-clock latency is only an estimate of how many setpoints should have been consumed. The
    robot cares about the setpoints that were really sent, so queue merge uses the queue's
    consumption index. This is the causal quantity that keeps the new chunk aligned with the old
    one.
    """
    consumed_steps = max(int(action_queue.get_action_index()) - int(result.action_index_before_inference), 0)
    merge_delay_steps = _clamp_rtc_delay_steps(consumed_steps, int(result.processed_actions.shape[0]))
    action_queue.merge(
        result.original_actions,
        result.processed_actions,
        real_delay=merge_delay_steps,
        action_index_before_inference=(
            int(result.action_index_before_inference) if merge_delay_steps == consumed_steps else None
        ),
    )
    latency_tracker.add(float(result.latency_s))
    return {
        'status': 'replan_async_merge',
        'latency_s': float(result.latency_s),
        'guidance_delay_steps': int(result.guidance_delay_steps),
        'merge_delay_steps': int(merge_delay_steps),
        'actual_consumed_steps': int(consumed_steps),
        'queue_size': action_queue.qsize(),
        'observation_step': int(result.observation_step),
    }

def load_policy_stack(
    pretrained_dir: Path,
    *,
    ds_meta: LeRobotDatasetMetadata,
    device: torch.device,
    n_action_steps_override: int | None = None,
    act_temporal_ensemble_coeff: float | None = None,
    rtc_mode: str = 'disabled',
    rtc_execution_horizon: int = _DEFAULT_RTC_EXECUTION_HORIZON,
    rtc_max_guidance_weight: float = _DEFAULT_RTC_MAX_GUIDANCE_WEIGHT,
    rtc_prefix_attention_schedule: str = _DEFAULT_RTC_PREFIX_ATTENTION_SCHEDULE.value,
) -> tuple[Any, PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    policy_cfg = load_train_config(pretrained_dir).policy
    if policy_cfg is None:
        raise ValueError(f"No policy config found in {pretrained_dir / 'train_config.json'}")

    disable_policy_compile_for_online_rollout(policy_cfg)

    rtc_enabled = configure_rtc_for_policy_config(
        policy_cfg,
        rtc_mode=rtc_mode,
        execution_horizon=rtc_execution_horizon,
        max_guidance_weight=rtc_max_guidance_weight,
        prefix_attention_schedule=rtc_prefix_attention_schedule,
    )

    if rtc_enabled and act_temporal_ensemble_coeff is not None:
        raise ValueError('--act-temporal-ensemble-coeff and RTC are mutually exclusive rollout smoothers.')

    if n_action_steps_override is not None:
        n_action_steps = int(n_action_steps_override)
        if n_action_steps <= 0:
            raise ValueError('--policy-n-action-steps must be > 0 when provided.')
        chunk_size = int(getattr(policy_cfg, 'chunk_size', n_action_steps))
        if n_action_steps > chunk_size:
            raise ValueError(
                f'--policy-n-action-steps={n_action_steps} exceeds policy chunk_size={chunk_size}.'
            )
        old_n_action_steps = getattr(policy_cfg, 'n_action_steps', None)
        policy_cfg.n_action_steps = n_action_steps
        print(
            '[INFO] policy_n_action_steps_override='
            f'from={old_n_action_steps} to={n_action_steps} chunk_size={chunk_size}'
        )

    if act_temporal_ensemble_coeff is not None:
        if not hasattr(policy_cfg, 'temporal_ensemble_coeff'):
            raise ValueError('--act-temporal-ensemble-coeff is only supported by ACT-style policy configs.')
        temporal_ensemble_coeff = float(act_temporal_ensemble_coeff)
        chunk_size = int(getattr(policy_cfg, 'chunk_size', 1))
        old_temporal_ensemble_coeff = getattr(policy_cfg, 'temporal_ensemble_coeff', None)
        old_n_action_steps = getattr(policy_cfg, 'n_action_steps', None)
        policy_cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy_cfg.n_action_steps = 1
        print(
            '[INFO] act_temporal_ensemble_override='
            f'coeff_from={old_temporal_ensemble_coeff} coeff_to={temporal_ensemble_coeff:.6g} '
            f'n_action_steps_from={old_n_action_steps} n_action_steps_to=1 chunk_size={chunk_size}'
        )

    policy_cfg.device = str(device)
    policy_cfg.pretrained_path = pretrained_dir
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={'device_processor': {'device': str(device)}},
    )
    policy.eval()
    return policy, preprocessor, postprocessor


class TerminateAsKeyboardInterrupt:
    """SIGTERM turned into the exception a rollout already unwinds on, once.

    SIGTERM is how the gateway's stop button reaches this process when the `quit` it wrote to
    stdin has gone unanswered, and Python's default action for it is to die on the spot: no
    `finally`, so no `robot.disconnect()` and no `dataset.finalize()`. A DAgger run killed that
    way leaves its corrections as a parquet with no footer -- 432 recorded frames were lost
    exactly so on 2026-09-02. Raising here makes a signalled shutdown identical to a Ctrl-C one,
    which is the path this file has always handled.

    A repeat is ignored rather than escalating, because the second signal would land *inside*
    that shutdown -- encoding video and closing parquet writers is the slow part of it -- and
    interrupting the close is the failure the guard exists to prevent. SIGKILL stays the
    operator's escalation, and the gateway sends it after its own grace period.
    """

    def __init__(self, emit: Callable[[str], None] = print):
        self._emit = emit
        self._previous: Any = None
        self._installed = False
        self.shutting_down = False

    def install(self) -> None:
        try:
            self._previous = signal.signal(signal.SIGTERM, self)
        except ValueError:
            # Signal handlers can only be installed from the main thread. A harness that drives
            # `run_inference` off it still gets the rollout; it just does not get this.
            return
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        self._installed = False
        try:
            signal.signal(signal.SIGTERM, self._previous)
        except (ValueError, TypeError):
            pass

    def __call__(self, signum: int, _frame: Any) -> None:
        if self.shutting_down:
            self._emit(f'[INFO] shutdown_signal={signum} ignored; the session is already closing.')
            return
        self.shutting_down = True
        self._emit(f'[INFO] shutdown_signal={signum} received; closing the session.')
        raise KeyboardInterrupt


def run_inference(args: argparse.Namespace) -> int:
    pretrained_dir = resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = load_train_config(pretrained_dir)
    dataset_root = resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    alignment_dataset_root, alignment_state_key = resolve_alignment_dataset_root_and_state_key(dataset_root)
    ds_meta = load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    task_prompt = resolve_rollout_task_prompt(ds_meta, args.task_prompt)
    dataset_start_pose_contract_xyzquat, dataset_start_pose_stats = estimate_dataset_start_pose_contract(
        alignment_dataset_root,
        state_key=alignment_state_key,
    )
    camera_configs = load_camera_configs(args.camera_config)
    camera_crop_specs, camera_crop_source_hw = load_camera_crop_specs(dataset_root)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    policy, preprocessor, postprocessor = load_policy_stack(
        pretrained_dir,
        ds_meta=ds_meta,
        device=device,
        n_action_steps_override=args.policy_n_action_steps,
        act_temporal_ensemble_coeff=args.act_temporal_ensemble_coeff,
        rtc_mode=args.rtc_mode,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
    )
    required_image_keys = extract_required_image_keys(policy.config.input_features)
    required_tactile_keys = extract_required_tactile_keys(policy.config.input_features)
    validate_camera_keys(required_image_keys=required_image_keys, available_camera_keys=list(camera_configs))
    if args.tactile_fallback is not None and not args.preview:
        raise ValueError('--tactile-fallback is preview-only. Use it together with --preview.')

    policy_fps = float(args.policy_fps or ds_meta.fps)
    if policy_fps <= 0.0:
        raise ValueError('policy-fps must be positive.')
    rtc_config = getattr(policy.config, 'rtc_config', None)
    rtc_enabled = bool(rtc_config is not None and getattr(rtc_config, 'enabled', False))
    rtc_time_per_step = 1.0 / policy_fps
    rtc_replan_queue_size = (
        resolve_rtc_replan_queue_size(policy, args.rtc_replan_queue_size) if rtc_enabled else 0
    )
    rtc_state: dict[str, Any] = {
        'queue': ActionQueue(rtc_config) if rtc_enabled else None,
        'latency_tracker': LatencyTracker(),
        'last_debug': {'status': 'disabled'},
    }
    if rtc_enabled and args.act_temporal_stuck_max_offset is not None:
        raise ValueError('--act-temporal-stuck-max-offset is ACT-temporal-only; disable it when RTC is enabled.')
    first_frame_max_pos_delta_m = float(args.first_frame_max_pos_delta_mm) / 1000.0
    first_frame_max_rot_delta_rad = np.deg2rad(float(args.first_frame_max_rot_delta_deg))
    max_step_pos_delta_m = float(args.max_step_pos_delta_mm) / 1000.0
    max_step_rot_delta_rad = np.deg2rad(float(args.max_step_rot_delta_deg))
    max_leash_pos_delta_m = float(args.max_leash_pos_delta_mm) / 1000.0
    max_leash_rot_delta_rad = np.deg2rad(float(args.max_leash_rot_delta_deg))
    dataset_start_gripper_tolerance = float(args.dataset_start_gripper_tolerance)
    state_names = (
        extract_feature_names(ds_meta.features['observation.state'], _DEFAULT_STATE_NAMES)
        if 'observation.state' in ds_meta.features
        else []
    )
    action_names = extract_feature_names(ds_meta.features['action'], _DEFAULT_ACTION_NAMES)
    # Which action contract this checkpoint was trained on is read off the dataset's own action
    # feature names, so a delta checkpoint can never be silently driven as an absolute one.
    # Resolved here, before the arm is touched, and the reconstructor is stateful so it must be
    # a single instance for the whole run.
    delta_reference = delta_reference_from_action_names(action_names)
    delta_reconstructor = build_delta_action_reconstructor(action_names)
    robot_init_state = parse_robot_init_state(args.robot_init_state)
    mujoco_model_path = resolve_mujoco_model_path(args.gripper_backend, args.mujoco_model)
    robot_urdf_path, target_frame_name = resolve_robot_tool_model(
        args.gripper_backend,
        args.robot_urdf_path,
        args.target_frame_name,
    )
    controller_stiffness = _parse_optional_float_tuple(
        args.controller_stiffness,
        expected_len=7,
        argument_name='--controller-stiffness',
    )
    controller_damping = _parse_optional_float_tuple(
        args.controller_damping,
        expected_len=7,
        argument_name='--controller-damping',
    )
    place_assist_offset_base_xyz = _parse_optional_float_tuple(
        args.place_assist_offset_base_xyz,
        expected_len=3,
        argument_name='--place-assist-offset-base-xyz',
    )
    place_assist_offset_base_xyz_m = (
        np.asarray(place_assist_offset_base_xyz, dtype=np.float64)
        if place_assist_offset_base_xyz is not None
        else None
    )

    # Refused here, before the arm is constructed and the policy is loaded, because this is a
    # pre-flight and a pre-flight that fires after the operator has homed the arm and waited out a
    # checkpoint load is one they will learn to run past. The property lives on the driver class,
    # so no device has to be plugged in to answer it.
    if args.dagger_takeover:
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

        if not backend_dates_reports(SpaceMouseTeleop):
            raise SystemExit(
                '--dagger-takeover refused: '
                + undated_backend_error(
                    driver_module=getattr(
                        sys.modules.get(SpaceMouseTeleop.__module__), '__file__', None
                    )
                )
            )

    # The fence the driver will clip against, resolved before the config is built so the banner can
    # say where it came from. It stopped being a literal here because a literal cannot be compared
    # with the recording rig's, and the two had silently drifted 50 mm apart on z -- 22 mm above the
    # lowest frame in the demonstrations, which put every grasp inside the wall.
    workspace_min, workspace_max, workspace_fence_source = resolve_workspace_fence(
        record_config_path=args.record_config,
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
    )
    print(
        f'[INFO] workspace_fence min=({workspace_min[0]:.3f}, {workspace_min[1]:.3f}, {workspace_min[2]:.3f}) '
        f'max=({workspace_max[0]:.3f}, {workspace_max[1]:.3f}, {workspace_max[2]:.3f}) '
        f'source={workspace_fence_source}'
    )

    tactile_fallback_observation = build_tactile_fallback_observation(args.tactile_fallback)
    tactile_enabled = bool(required_tactile_keys) and tactile_fallback_observation is None
    robot_cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(robot_urdf_path),
        target_frame_name=target_frame_name,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        stiffness=controller_stiffness,
        damping=controller_damping,
        filter_coeff=args.controller_filter_coeff,
        gripper_max_width_mm=float(args.gripper_max_width_mm),
        corenetic_bind_ip=str(args.corenetic_bind_ip),
        corenetic_bind_port=int(args.corenetic_bind_port),
        corenetic_remote_ip=str(args.corenetic_remote_ip),
        corenetic_remote_port=int(args.corenetic_remote_port),
        corenetic_sdk_dir=str(args.corenetic_sdk_dir),
        corenetic_connect_timeout_s=float(args.corenetic_connect_timeout_s),
        corenetic_poll_interval_s=float(args.corenetic_poll_interval_s),
        corenetic_stale_threshold_s=float(args.corenetic_stale_threshold_s),
        corenetic_release_mode_on_disconnect=bool(args.corenetic_release_mode_on_disconnect),
        use_otg=bool(args.use_otg),
        otg_control_frequency=float(args.otg_control_frequency),
        otg_async_control_frequency=float(args.otg_async_control_frequency),
        das_tactile_frequency_hz=policy_fps if tactile_enabled else None,
        das_tactile_valid_mask_path=str(_DEFAULT_TACTILE_VALID_MASK_PATH) if tactile_enabled else None,
        das_tactile_baseline_path=str(_DEFAULT_TACTILE_BASELINE_PATH) if tactile_enabled else None,
        das_tactile_timeout_s=2.0,
        cameras={name: cfg for name, cfg in camera_configs.items()},
    )

    if robot_init_state is not None and args.move_to_das_start:
        print('[INFO] robot_init_state is set; ignoring the explicit --move-to-das-start request.')
    else:
        move_to_das_start_if_requested(robot_ip=args.robot_ip, enabled=bool(args.move_to_das_start))

    from lerobot.robots.franka_research3 import FrankaResearch3

    robot = FrankaResearch3(robot_cfg)
    dataset_gripper_feature_name = str(dataset_start_pose_stats.get('gripper_feature_name') or 'gripper.pos')
    dataset_start_gripper_mean_normalized: float | None = None
    if 'gripper_mean' in dataset_start_pose_stats:
        dataset_start_gripper_mean_normalized = normalize_dataset_gripper(
            float(dataset_start_pose_stats['gripper_mean']),
            robot_cfg,
            feature_name=dataset_gripper_feature_name,
        )
    state_processor = KeepAbsoluteEEObservation()
    T_B_Ws: np.ndarray | None = None
    start_alignment_stats: dict[str, Any] | None = None
    dataset_start_pose_contract = _pose_from_position_and_quaternion(
        dataset_start_pose_contract_xyzquat[:3],
        dataset_start_pose_contract_xyzquat[3:7],
    )
    previous_dataset_quaternion_xyzw: np.ndarray | None = None
    preview_gripper_offset: float | None = None

    print(f'[INFO] checkpoint={pretrained_dir}')
    print(f'[INFO] dataset_root={dataset_root}')
    if alignment_dataset_root != dataset_root:
        print(f'[INFO] alignment_dataset_root={alignment_dataset_root}')
    print(f'[INFO] alignment_state_key={alignment_state_key}')
    if camera_crop_specs:
        for feature_key in sorted(camera_crop_specs):
            source_hw = camera_crop_source_hw.get(feature_key)
            print(
                f'[INFO] camera_crop {feature_key}='
                + ','.join(str(part) for part in camera_crop_specs[feature_key])
                + (f' source_frame={source_hw[1]}x{source_hw[0]}' if source_hw else ' source_frame=unknown')
            )
    else:
        print('[INFO] camera_crop=<none> (training view was built full frame)')
    print(f'[INFO] policy_device={device}')
    print(f'[INFO] policy_fps={policy_fps:.3f}')
    print('[INFO] task_prompt=' + (json.dumps(task_prompt, ensure_ascii=False) if task_prompt else '<empty>'))
    print('[INFO] policy_image_keys=' + ', '.join(required_image_keys) if required_image_keys else '[INFO] policy_image_keys=<none>')
    print('[INFO] policy_tactile_keys=' + ', '.join(required_tactile_keys) if required_tactile_keys else '[INFO] policy_tactile_keys=<none>')
    print('[INFO] tactile_fallback=' + args.tactile_fallback if args.tactile_fallback is not None else '[INFO] tactile_fallback=<none>')
    print(
        '[INFO] dataset_start_contract='
        f"episodes={dataset_start_pose_stats['episodes']} "
        f"xyz=({dataset_start_pose_contract_xyzquat[0]:+.6f}, {dataset_start_pose_contract_xyzquat[1]:+.6f}, {dataset_start_pose_contract_xyzquat[2]:+.6f}) "
        f"quat=({dataset_start_pose_contract_xyzquat[3]:+.6f}, {dataset_start_pose_contract_xyzquat[4]:+.6f}, "
        f"{dataset_start_pose_contract_xyzquat[5]:+.6f}, {dataset_start_pose_contract_xyzquat[6]:+.6f})"
    )
    dataset_start_spread_line = (
        '[INFO] dataset_start_spread='
        f"xyz_std_mm=({_format_vector(np.asarray(dataset_start_pose_stats['position_std_xyz_mm'], dtype=np.float64))}) "
        f"rot_mean/p95/max_deg="
        f"{dataset_start_pose_stats['rotation_spread_mean_deg']:.2f}/"
        f"{dataset_start_pose_stats['rotation_spread_p95_deg']:.2f}/"
        f"{dataset_start_pose_stats['rotation_spread_max_deg']:.2f}"
    )
    if 'gripper_mean' in dataset_start_pose_stats and 'gripper_std' in dataset_start_pose_stats:
        dataset_start_spread_line += (
            f" gripper_mean/std={dataset_start_pose_stats['gripper_mean']:.3f}/"
            f"{dataset_start_pose_stats['gripper_std']:.3f}"
            f" source={dataset_start_pose_stats.get('gripper_source', '<unknown>')}"
        )
        if 'observation_gripper_mean' in dataset_start_pose_stats and 'observation_gripper_std' in dataset_start_pose_stats:
            dataset_start_spread_line += (
                f" observation_gripper_mean/std={dataset_start_pose_stats['observation_gripper_mean']:.3f}/"
                f"{dataset_start_pose_stats['observation_gripper_std']:.3f}"
            )
    print(dataset_start_spread_line)
    print(f'[INFO] state_frame=absolute_pose({robot_cfg.target_frame_name}) in dataset_world(W_s)')
    print('[INFO] tool_frame_transform=identity; no DAS gripper_base_link<->EE fixed transform is applied')
    print(
        '[INFO] safety='
        f'first_frame<{args.first_frame_max_pos_delta_mm:.1f}mm/{args.first_frame_max_rot_delta_deg:.1f}deg, '
        f'per_step<{args.max_step_pos_delta_mm:.1f}mm/{args.max_step_rot_delta_deg:.1f}deg (vs prev_cmd), '
        f'leash<{args.max_leash_pos_delta_mm:.1f}mm/{args.max_leash_rot_delta_deg:.1f}deg (vs measured), '
        f'preview={args.preview}'
    )
    print(
        '[INFO] joint-space smoothing='
        f"{'enabled' if robot_cfg.use_otg else 'disabled'} "
        f'FR3 OTG @ {robot_cfg.otg_control_frequency:.1f}Hz / sender @ {robot_cfg.otg_async_control_frequency:.1f}Hz'
    )
    print(
        '[INFO] controller_gains='
        f"stiffness={'set' if robot_cfg.stiffness is not None else 'default'} "
        f"damping={'set' if robot_cfg.damping is not None else 'default'} "
        f"filter_coeff={robot_cfg.filter_coeff if robot_cfg.filter_coeff is not None else 'default'}"
    )
    print(
        '[INFO] gripper_close_below='
        + ('disabled' if args.gripper_close_below is None else f'raw<{float(args.gripper_close_below):.6g}->0')
    )
    print(
        '[INFO] gripper_change_delay='
        + (
            'disabled'
            if args.gripper_change_delay_s is None
            else f'{float(args.gripper_change_delay_s):.3f}s min_delta={float(args.gripper_change_min_delta):.3f} normalized'
            f' settle_tol={float(args.gripper_change_settle_tolerance):.3f}'
            f' settle_timeout={float(args.gripper_change_settle_timeout_s):.3f}s'
        )
    )
    print(f'[INFO] act_temporal_action_offset={int(args.act_temporal_action_offset)}')
    print(
        '[INFO] act_temporal_stuck_offset='
        + (
            'disabled'
            if args.act_temporal_stuck_max_offset is None
            else (
                f"base={int(args.act_temporal_action_offset)} "
                f"max={int(args.act_temporal_stuck_max_offset)} "
                f"step={int(args.act_temporal_stuck_offset_step)} "
                f"stuck_steps={int(args.act_temporal_stuck_steps)} "
                f"stuck_pos_delta={float(args.act_temporal_stuck_pos_delta_mm):.2f}mm "
                f"closed_gripper_max={float(args.act_temporal_stuck_closed_gripper_max):.3f}"
            )
        )
    )
    print(
        '[INFO] rtc_queue='
        + (
            'disabled'
            if not rtc_enabled
            else (
                f'replan_when_remaining<={rtc_replan_queue_size} ' 
                f'fixed_delay_steps={args.rtc_inference_delay_steps if args.rtc_inference_delay_steps is not None else "auto"}'
            )
        )
    )
    print(
        '[INFO] command_ema_alpha='
        + ('disabled' if args.command_ema_alpha is None else f'{float(args.command_ema_alpha):.3f}')
    )
    print(
        '[INFO] place_assist='
        + (
            'disabled'
            if place_assist_offset_base_xyz_m is None
            else (
                f"offset_base_xyz_m=({place_assist_offset_base_xyz_m[0]:+.4f}, "
                f"{place_assist_offset_base_xyz_m[1]:+.4f}, {place_assist_offset_base_xyz_m[2]:+.4f}) "
                f"stuck_steps={int(args.place_assist_stuck_steps)} "
                f"stuck_pos_delta={float(args.place_assist_stuck_pos_delta_mm):.2f}mm "
                f"ramp_step={float(args.place_assist_ramp_step_mm):.2f}mm "
                f"closed_gripper_max={float(args.place_assist_closed_gripper_max):.3f}"
            )
        )
    )
    print(
        '[INFO] mujoco_viewer='
        f"{'enabled' if args.mujoco_viewer else 'disabled'} model={mujoco_model_path}"
    )
    print(f"[INFO] camera_preview_window={'enabled' if args.camera_preview_window else 'disabled'}")
    preview_sink: PolicyCameraPreviewSink | None = None
    if args.preview_jpeg_dir is not None:
        preview_sink = PolicyCameraPreviewSink(
            args.preview_jpeg_dir,
            camera_keys=list(required_image_keys),
            fps=float(args.preview_jpeg_fps),
        )
        preview_sink.start()

    def build_camera_preview_observation() -> dict[str, np.ndarray]:
        """The frames a viewer should see, prepared exactly as the policy's are."""

        if preview_sink is None:
            return {}
        try:
            robot_observation = robot.get_observation()
        except Exception as exc:  # noqa: BLE001 - a background frame must not stop rollout control
            print(f'[WARN] camera_preview_snapshot=skipped reason={exc}')
            return {}
        preview_observation: dict[str, np.ndarray] = {}
        for camera_key in required_image_keys:
            if camera_key not in robot_observation:
                continue
            image = np.asarray(robot_observation[camera_key], dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] != 3:
                continue
            if camera_key in camera_configs:
                color_mode = getattr(camera_configs[camera_key], 'color_mode', None)
                try:
                    color_mode = ColorMode(color_mode)
                except ValueError:
                    color_mode = None
                if color_mode == ColorMode.BGR:
                    image = np.ascontiguousarray(image[..., ::-1])
            feature_key = f'{_OBS_IMAGES_PREFIX}{camera_key}'
            crop = camera_crop_specs.get(feature_key) if camera_crop_specs else None
            if crop is not None:
                image = apply_camera_crop(
                    image,
                    crop,
                    feature_key=feature_key,
                    source_hw=camera_crop_source_hw.get(feature_key) if camera_crop_source_hw else None,
                )
            preview_observation[feature_key] = image
        return preview_observation

    def publish_current_camera_preview_snapshot() -> None:
        preview_observation = build_camera_preview_observation()
        if preview_sink is not None and preview_observation:
            preview_sink.publish(preview_observation, immediate=True)
            print('[INFO] scene_reset_preview_snapshot=published', flush=True)

    def write_pose_probe_still(request: PoseProbeRequest) -> None:
        """Freeze the view of the arm standing at one probed coordinate.

        Written to its own directory rather than over the live preview because it is evidence,
        not a view: the operator clicks the tool in it minutes later, by which time the arm has
        retreated and the waiting loop has homed it. The sidecar names the request that
        produced it, so a reader can tell this still from the previous probe's.
        """

        if preview_sink is None or args.preview_jpeg_dir is None:
            print('[WARN] pose_probe_still=skipped reason=no_preview_sink', flush=True)
            return
        preview_observation = build_camera_preview_observation()
        if not preview_observation:
            print('[WARN] pose_probe_still=skipped reason=no_frames', flush=True)
            return
        directory = Path(args.preview_jpeg_dir) / 'probe'
        cameras = preview_sink.write_still(preview_observation, directory)
        if not cameras:
            print('[WARN] pose_probe_still=skipped reason=no_frames_written', flush=True)
            return
        sidecar = {
            'requestId': request.requestId,
            'xyz': [float(value) for value in request.xyz],
            'cameras': cameras,
            'at': time.time(),
        }
        try:
            _write_bytes_atomic(directory / 'probe.json', json.dumps(sidecar, sort_keys=True).encode())
        except OSError as exc:
            print(f'[WARN] pose_probe_still=sidecar_failed details={exc}', flush=True)
            return
        print(
            f"[INFO] pose_probe_still=written request_id={request.requestId} "
            f"cameras={','.join(cameras)} dir={directory}",
            flush=True,
        )

    if args.preview and args.align_gripper_to_dataset_start:
        print('[INFO] preview_gripper_alignment=requested; using virtual observation correction without moving hardware.')
    if args.interactive_rollouts and robot_init_state is None:
        print('[WARN] interactive_rollouts enabled without robot_init_state; rollout reset will hold the current robot state.')

    # Declared here so the finally below can always test it. The arm is not connected here: see
    # the connect inside the try.
    mujoco_visualizer: FR3InferenceMujocoVisualizer | None = None
    def reset_policy_runtime_state() -> None:
        policy.reset()
        state_processor.reset()
        preprocessor.reset()
        postprocessor.reset()
        if rtc_enabled:
            rtc_state['queue'] = ActionQueue(rtc_config)
            rtc_state['latency_tracker'] = LatencyTracker()
            rtc_state['last_debug'] = {'status': 'reset'}

    def run_policy_rollout(
        interactive_keyboard: InteractiveRolloutKeyboard | None = None,
        trace: RolloutGeometryTrace | None = None,
        expert_takeover: ExpertTakeover | None = None,
        dagger_buffer: DaggerFrameBuffer | None = None,
    ) -> str:
        reset_policy_runtime_state()
        T_B_Ws: np.ndarray | None = None
        start_alignment_stats: dict[str, Any] | None = None
        previous_dataset_quaternion_xyzw: np.ndarray | None = None
        # Separate from the observation's continuity state above: this one tracks the sign of
        # the *action* quaternion written to the DAgger dataset, which is a different sequence
        # of rotations and would flip at different frames.
        previous_dagger_quaternion_xyzw: np.ndarray | None = None
        preview_gripper_offset: float | None = None
        latest_chunk_ee_poses: list[np.ndarray] | None = None
        camera_preview_enabled = bool(args.camera_preview_window)
        previous_sent_command: dict[str, float] | None = None
        previous_smoothed_command: dict[str, float] | None = None
        gripper_latch_state: dict[str, float | None] = {'command': None, 'last_change_time_s': None}
        place_assist_state: dict[str, Any] = {
            'offset_xyz_m': np.zeros(3, dtype=np.float64),
            'stuck_count': 0,
        }
        temporal_offset_state: dict[str, int] = {
            'current_offset': max(int(args.act_temporal_action_offset), 0),
            'stuck_count': 0,
        }
        rtc_planner = AsyncActionChunkPlanner() if rtc_enabled else None

        def finish_rollout(status: str) -> str:
            if rtc_planner is not None and not rtc_planner.join(timeout_s=5.0):
                print('[WARN] rtc_async_planner_still_running_after_rollout_stop timeout_s=5.0')
            return status

        step_idx = 0
        while args.max_steps is None or step_idx < args.max_steps:
            if interactive_keyboard is not None and interactive_keyboard.should_stop_rollout():
                return finish_rollout('quit' if interactive_keyboard.quit_requested.is_set() else 'stopped')
            loop_start_t = time.perf_counter()
            robot_observation = robot.get_observation()
            previous_tracking_position_delta: np.ndarray | None = None
            previous_tracking_rotation_delta: np.ndarray | None = None
            if previous_sent_command is not None:
                previous_tracking_position_delta, previous_tracking_rotation_delta = compute_pose_delta_from_current(
                    previous_sent_command,
                    robot_observation,
                )
            absolute_state_observation_e = state_processor.observation(dict(robot_observation))
            absolute_state_observation_i = convert_absolute_observation_from_E_to_I(absolute_state_observation_e)
            live_gripper_dataset_units = denormalize_live_gripper_observation(
                float(robot_observation['gripper.pos']),
                robot_cfg,
                feature_name=dataset_gripper_feature_name,
            )
            if T_B_Ws is None:
                current_start_pose_i = _pose_from_quaternion_observation(absolute_state_observation_i)
                T_B_Ws = current_start_pose_i @ _invert_pose(dataset_start_pose_contract)
                start_alignment_stats = summarize_live_start_alignment_to_dataset_starts(
                    alignment_dataset_root,
                    T_B_Ws,
                    current_start_pose_i,
                    state_key=alignment_state_key,
                    live_gripper=live_gripper_dataset_units,
                )
                dataset_alignment_line = (
                    '[INFO] dataset_start_alignment='
                    f"T(B,W_s).xyz=({T_B_Ws[0,3]:+.6f}, {T_B_Ws[1,3]:+.6f}, {T_B_Ws[2,3]:+.6f}) "
                    f"nearest_ep={start_alignment_stats['best_episode_index']} "
                    f"nearest_pos_mm={start_alignment_stats['best_position_error_mm']:.2f} "
                    f"nearest_rot_deg={start_alignment_stats['best_rotation_error_deg']:.2f} "
                    f"median_pos_mm={start_alignment_stats['median_position_error_mm']:.2f} "
                    f"p95_pos_mm={start_alignment_stats['p95_position_error_mm']:.2f} "
                    f"median_rot_deg={start_alignment_stats['median_rotation_error_deg']:.2f} "
                    f"p95_rot_deg={start_alignment_stats['p95_rotation_error_deg']:.2f}"
                )
                if 'live_gripper' in start_alignment_stats:
                    dataset_gripper_mean = float(dataset_start_pose_stats.get('gripper_mean', float('nan')))
                    dataset_gripper_source = str(dataset_start_pose_stats.get('gripper_source', '<unknown>'))
                    gripper_delta_to_mean = abs(float(start_alignment_stats['live_gripper']) - dataset_gripper_mean)
                    dataset_alignment_line += (
                        f" live_gripper={start_alignment_stats['live_gripper']:.3f}"
                        f" dataset_gripper_target={dataset_gripper_mean:.3f}"
                        f" source={dataset_gripper_source}"
                        f" delta_to_target={gripper_delta_to_mean:.3f}"
                        f" nearest_gripper_abs={start_alignment_stats['best_gripper_abs_delta']:.3f}"
                        f" median_gripper_abs={start_alignment_stats['median_gripper_abs_delta']:.3f}"
                        f" p95_gripper_abs={start_alignment_stats['p95_gripper_abs_delta']:.3f}"
                    )
                print(dataset_alignment_line)
                if (
                    'live_gripper' in start_alignment_stats
                    and 'gripper_mean' in dataset_start_pose_stats
                    and abs(float(start_alignment_stats['live_gripper']) - float(dataset_start_pose_stats['gripper_mean']))
                    > dataset_start_gripper_tolerance
                ):
                    print(
                        '[WARN] live gripper start is far from dataset start contract; '
                        f"live={start_alignment_stats['live_gripper']:.3f} "
                        f"dataset_target={float(dataset_start_pose_stats['gripper_mean']):.3f} "
                        f"source={dataset_start_pose_stats.get('gripper_source', '<unknown>')} "
                        f"abs_delta={abs(float(start_alignment_stats['live_gripper']) - float(dataset_start_pose_stats['gripper_mean'])):.3f} "
                        f"tol={dataset_start_gripper_tolerance:.3f}. "
                        'Use a dataset-like start gripper or --align-gripper-to-dataset-start.'
                    )
                if args.preview and 'live_gripper' in start_alignment_stats and 'gripper_mean' in dataset_start_pose_stats:
                    preview_gripper_offset = float(dataset_start_pose_stats['gripper_mean']) - float(start_alignment_stats['live_gripper'])
                    preview_target_gripper = float(np.clip(float(start_alignment_stats['live_gripper']) + preview_gripper_offset, 0.0, 1.0))
                    print(
                        '[INFO] preview_gripper_alignment=virtual '
                        f"live_start={float(start_alignment_stats['live_gripper']):.3f} "
                        f"target={float(dataset_start_pose_stats['gripper_mean']):.3f} "
                        f"source={dataset_start_pose_stats.get('gripper_source', '<unknown>')} "
                        f"offset={preview_gripper_offset:+.3f} "
                        f"corrected_start={preview_target_gripper:.3f}"
                    )

            assert T_B_Ws is not None
            dataset_state_observation_i, previous_dataset_quaternion_xyzw = convert_base_observation_from_I_to_dataset_frame(
                absolute_state_observation_i,
                T_B_Ws,
                previous_quaternion_xyzw=previous_dataset_quaternion_xyzw,
            )
            dataset_state_observation_i = convert_gripper_observation_to_dataset_units(
                dataset_state_observation_i,
                robot_cfg=robot_cfg,
                state_names=state_names,
            )
            policy_state_observation_i = apply_gripper_observation_offset(
                dataset_state_observation_i,
                gripper_offset=preview_gripper_offset if args.preview else None,
            )
            if args.preview and preview_gripper_offset is not None and step_idx == 0:
                print(
                    '[INFO] preview_policy_obs_gripper='
                    f"raw={float(dataset_state_observation_i['gripper.pos']):.3f} "
                    f"corrected={float(policy_state_observation_i['gripper.pos']):.3f} "
                    f"offset={preview_gripper_offset:+.3f}"
                )
            policy_observation = build_policy_observation(
                policy_state_observation_i,
                state_names=state_names,
                input_features=policy.config.input_features,
                tactile_fallback_observation=tactile_fallback_observation,
                camera_configs=camera_configs,
                camera_crop_specs=camera_crop_specs,
                camera_crop_source_hw=camera_crop_source_hw,
            )
            if camera_preview_enabled:
                camera_preview_enabled = show_policy_camera_preview_window(
                    policy_observation,
                    camera_keys=required_image_keys,
                )
            if preview_sink is not None:
                preview_sink.publish(policy_observation)
            if step_idx == 0 and args.debug_step0_dump_dir is not None:
                if start_alignment_stats is None:
                    raise RuntimeError('start_alignment_stats must be initialized before step0 capture dump.')
                dump_dir = dump_step0_capture_bundle(
                    args.debug_step0_dump_dir,
                    checkpoint=pretrained_dir,
                    dataset_root=dataset_root,
                    dataset_repo_id=train_cfg.dataset.repo_id,
                    T_B_Ws=T_B_Ws,
                    start_alignment_stats=start_alignment_stats,
                    state_names=state_names,
                    action_names=action_names,
                    policy_observation=policy_observation,
                    robot_observation=robot_observation,
                    absolute_state_observation_e=absolute_state_observation_e,
                    absolute_state_observation_i=absolute_state_observation_i,
                    dataset_state_observation_i=dataset_state_observation_i,
                )
                print(f'[INFO] step0_capture_dump={dump_dir}')
            if rtc_enabled:
                action_queue = rtc_state['queue']
                if not isinstance(action_queue, ActionQueue):
                    raise RuntimeError('RTC is enabled but the rollout action queue was not initialized.')
                latency_tracker = rtc_state['latency_tracker']
                if not isinstance(latency_tracker, LatencyTracker):
                    raise RuntimeError('RTC is enabled but the latency tracker was not initialized.')
                if rtc_planner is None:
                    raise RuntimeError('RTC is enabled but the async planner was not initialized.')

                completed_plan = rtc_planner.pop_completed()
                if completed_plan is not None:
                    rtc_state['last_debug'] = merge_completed_rtc_plan(
                        action_queue,
                        completed_plan,
                        latency_tracker,
                    )

                if action_queue.empty():
                    if rtc_planner.running():
                        # This is an overload path, not the intended steady state. Wait for the
                        # in-flight plan rather than crashing, but make the stall explicit in logs.
                        print('[WARN] rtc_queue_starved_waiting_for_async_plan')
                        if not rtc_planner.join(timeout_s=5.0):
                            raise RuntimeError(
                                'RTC action queue starved and async chunk prediction did not finish within 5s.'
                            )
                        completed_plan = rtc_planner.pop_completed()
                        if completed_plan is None:
                            raise RuntimeError('RTC async chunk prediction completed without a mergeable result.')
                        rtc_state['last_debug'] = merge_completed_rtc_plan(
                            action_queue,
                            completed_plan,
                            latency_tracker,
                        )

                    if action_queue.empty():
                        prev_chunk_left_over = action_queue.get_left_over()
                        if prev_chunk_left_over is not None:
                            prev_chunk_left_over = prev_chunk_left_over.detach().clone()
                        tracked_latency = latency_tracker.max() or 0.0
                        guidance_delay_steps = (
                            int(args.rtc_inference_delay_steps)
                            if args.rtc_inference_delay_steps is not None
                            else int(math.ceil(float(tracked_latency) / rtc_time_per_step))
                        )
                        guidance_delay_steps = max(guidance_delay_steps, 0)
                        chunk_start_t = time.perf_counter()
                        original_actions, processed_actions = predict_action_chunk_for_rollout(
                            policy_observation,
                            policy=policy,
                            device=device,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            use_amp=bool(policy.config.use_amp),
                            inference_delay=guidance_delay_steps,
                            prev_chunk_left_over=prev_chunk_left_over,
                            execution_horizon=int(args.rtc_execution_horizon),
                            task=task_prompt,
                            robot_type=robot.name,
                        )
                        chunk_latency_s = time.perf_counter() - chunk_start_t
                        latency_tracker.add(chunk_latency_s)
                        # No queued actions are consumed while the main thread blocks here. Skipping
                        # by wall-clock latency would jump into the middle of the new chunk and create
                        # exactly the discontinuity RTC is supposed to avoid.
                        action_queue.merge(
                            original_actions,
                            processed_actions,
                            real_delay=0,
                            action_index_before_inference=action_queue.get_action_index(),
                        )
                        rtc_state['last_debug'] = {
                            'status': 'replan_sync_bootstrap' if prev_chunk_left_over is None else 'replan_sync_starved',
                            'latency_s': chunk_latency_s,
                            'guidance_delay_steps': guidance_delay_steps,
                            'merge_delay_steps': 0,
                            'actual_consumed_steps': 0,
                            'queue_size': action_queue.qsize(),
                            'prev_leftover': 0 if prev_chunk_left_over is None else int(prev_chunk_left_over.shape[0]),
                        }

                elif action_queue.qsize() <= rtc_replan_queue_size:
                    if rtc_planner.running():
                        rtc_state['last_debug'] = {
                            'status': 'replan_async_pending',
                            'queue_size': action_queue.qsize(),
                        }
                    else:
                        prev_chunk_left_over = action_queue.get_left_over()
                        if prev_chunk_left_over is not None:
                            prev_chunk_left_over = prev_chunk_left_over.detach().clone()
                        tracked_latency = latency_tracker.max() or 0.0
                        guidance_delay_steps = (
                            int(args.rtc_inference_delay_steps)
                            if args.rtc_inference_delay_steps is not None
                            else int(math.ceil(float(tracked_latency) / rtc_time_per_step))
                        )
                        guidance_delay_steps = max(guidance_delay_steps, 0)
                        started = rtc_planner.start(
                            policy_observation,
                            predict_kwargs={
                                'policy': policy,
                                'device': device,
                                'preprocessor': preprocessor,
                                'postprocessor': postprocessor,
                                'use_amp': bool(policy.config.use_amp),
                                'inference_delay': guidance_delay_steps,
                                'prev_chunk_left_over': prev_chunk_left_over,
                                'execution_horizon': int(args.rtc_execution_horizon),
                                'task': task_prompt,
                                'robot_type': robot.name,
                            },
                            action_index_before_inference=action_queue.get_action_index(),
                            guidance_delay_steps=guidance_delay_steps,
                            observation_step=step_idx,
                        )
                        rtc_state['last_debug'] = {
                            'status': 'replan_async_start' if started else 'replan_async_pending',
                            'latency_s': 0.0,
                            'guidance_delay_steps': guidance_delay_steps,
                            'merge_delay_steps': 0,
                            'queue_size': action_queue.qsize(),
                            'prev_leftover': 0 if prev_chunk_left_over is None else int(prev_chunk_left_over.shape[0]),
                        }
                else:
                    rtc_state['last_debug'] = {
                        'status': 'replan_async_pending' if rtc_planner.running() else 'reuse',
                        'queue_size': action_queue.qsize(),
                    }
                action_tensor = action_queue.get()
                if action_tensor is None:
                    raise RuntimeError('RTC action queue is empty after chunk prediction.')
                temporal_offset_used = 0
            else:
                action_tensor = predict_action(
                    policy_observation,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=bool(policy.config.use_amp),
                    task=task_prompt,
                    robot_type=robot.name,
                )
                temporal_offset_used = int(temporal_offset_state.get('current_offset', int(args.act_temporal_action_offset)))
                action_tensor = select_temporal_ensemble_offset_action(
                    action_tensor,
                    policy=policy,
                    postprocessor=postprocessor,
                    offset=temporal_offset_used,
                )
            model_gripper_raw = extract_action_gripper_raw(action_tensor, action_names)
            if mujoco_visualizer is not None:
                maybe_chunk_actions = (
                    extract_action_queue_for_visualization(action_queue, action_tensor)
                    if rtc_enabled and isinstance(rtc_state.get('queue'), ActionQueue)
                    else extract_new_action_chunk_for_visualization(
                        policy,
                        action_tensor,
                        postprocessor,
                    )
                )
                if maybe_chunk_actions is not None:
                    latest_chunk_ee_poses = build_chunk_ee_poses_for_visualization(
                        maybe_chunk_actions,
                        action_names=action_names,
                        robot_cfg=robot_cfg,
                        T_B_Ws=T_B_Ws,
                        delta_reference=delta_reference,
                        dataset_observation_i=dataset_state_observation_i,
                    )
            dataset_robot_command_i = decode_action_to_robot_command(
                action_tensor,
                action_names=action_names,
                robot_cfg=robot_cfg,
                gripper_close_below=args.gripper_close_below,
                delta_reconstructor=delta_reconstructor,
                dataset_observation_i=dataset_state_observation_i,
            )
            base_robot_command_i = convert_dataset_command_to_base_frame(dataset_robot_command_i, T_B_Ws)
            robot_command = convert_base_command_from_I_to_E(base_robot_command_i)
            temporal_offset_debug = update_temporal_offset_on_stuck(
                temporal_offset_state,
                base_offset=int(args.act_temporal_action_offset),
                max_offset=resolve_temporal_ensemble_max_offset(policy, args.act_temporal_stuck_max_offset),
                offset_step=int(args.act_temporal_stuck_offset_step),
                stuck_steps=int(args.act_temporal_stuck_steps),
                stuck_pos_delta_m=float(args.act_temporal_stuck_pos_delta_mm) / 1000.0,
                closed_gripper_max=float(args.act_temporal_stuck_closed_gripper_max),
                unassisted_command=robot_command,
                robot_observation=robot_observation,
            )
            robot_command, place_assist_debug = apply_place_assist_offset(
                robot_command,
                robot_observation,
                place_assist_state,
                target_offset_xyz_m=place_assist_offset_base_xyz_m,
                stuck_steps=int(args.place_assist_stuck_steps),
                stuck_pos_delta_m=float(args.place_assist_stuck_pos_delta_mm) / 1000.0,
                ramp_step_m=float(args.place_assist_ramp_step_mm) / 1000.0,
                closed_gripper_max=float(args.place_assist_closed_gripper_max),
            )
            command_source = 'policy'
            if expert_takeover is not None:
                robot_command, takeover_debug = expert_takeover.command(
                    # `latched` is the manual override, not the ordinary way in: moving the
                    # SpaceMouse is what takes the arm, and letting go of it hands the arm back.
                    latched=interactive_keyboard is not None and interactive_keyboard.takeover_is_engaged(),
                    policy_command=robot_command,
                    previous_sent_command=previous_sent_command,
                    robot_observation=robot_observation,
                )
                command_source = str(takeover_debug['source'])
                if command_source == 'expert':
                    # Both assists read "the policy has stopped making progress" off a command
                    # that is no longer the policy's. While the operator drives, holding still
                    # is an instruction, not a stall -- and an offset accumulated during a
                    # correction would be re-applied to a policy that has since been handed an
                    # arm somewhere else entirely.
                    place_assist_state['stuck_count'] = 0
                    place_assist_state['offset_xyz_m'] = np.zeros(3, dtype=np.float64)
                    temporal_offset_state['stuck_count'] = 0

            # One smoothing and clamping path for both action sources. The expert's target is
            # a delta against `previous_sent_command`, which is the same pose the policy's
            # delta is defined against, so the step guard below bounds the two identically --
            # and the filter state carries across a handoff instead of restarting at it.
            robot_command = smooth_robot_command_ema(
                robot_command,
                previous_smoothed_command,
                alpha=args.command_ema_alpha,
            )
            safe_command, command_guard = limit_command_for_safety(
                robot_command,
                robot_observation,
                max_step_pos_delta_m=max_step_pos_delta_m,
                max_step_rot_delta_rad=max_step_rot_delta_rad,
                max_leash_pos_delta_m=max_leash_pos_delta_m,
                max_leash_rot_delta_rad=max_leash_rot_delta_rad,
            )
            position_delta = command_guard['position_delta']
            rotation_delta = command_guard['rotation_delta']
            command_status = 'pass'
            command_to_send = safe_command
            if step_idx == 0:
                reject_first_command, first_position_delta, first_rotation_delta = should_reject_first_command(
                    robot_command,
                    robot_observation,
                    max_pos_delta_m=first_frame_max_pos_delta_m,
                    max_rot_delta_rad=first_frame_max_rot_delta_rad,
                )
                if reject_first_command:
                    command_status = 'hold_first_frame'
                    command_to_send = build_hold_command(robot_observation)
                    print(
                        '[WARN] step=0 rejecting first policy target and holding current EE pose; '
                        f'pos_delta_mm=({_format_vector(first_position_delta, scale=1000.0)}) '
                        f'rot_delta_deg=({_format_vector(np.rad2deg(first_rotation_delta))})'
                    )
            if command_status == 'pass' and command_guard['status'] != 'pass':
                command_status = command_guard['status']

            gripper_command_before_latch = float(command_to_send['gripper.pos'])
            command_to_send, gripper_latch_debug = apply_gripper_change_delay(
                command_to_send,
                robot_observation,
                gripper_latch_state,
                delay_s=args.gripper_change_delay_s,
                min_delta=float(args.gripper_change_min_delta),
                settle_tolerance=float(args.gripper_change_settle_tolerance),
                settle_timeout_s=float(args.gripper_change_settle_timeout_s),
            )

            previous_smoothed_command = dict(command_to_send)

            if mujoco_visualizer is not None:
                current_ee_pose = _pose_from_quaternion_observation(absolute_state_observation_e)
                target_ee_pose = _pose_from_rotvec_command(robot_command)
                safe_target_ee_pose = _pose_from_rotvec_command(command_to_send)
                mujoco_visualizer.update(
                    robot_observation=robot_observation,
                    current_ee_pose=current_ee_pose,
                    target_ee_pose=target_ee_pose,
                    safe_target_ee_pose=safe_target_ee_pose,
                    chunk_ee_poses=latest_chunk_ee_poses,
                )

            if step_idx == 0 and args.debug_step0_dump_dir is not None:
                dump_step0_action_debug(
                    args.debug_step0_dump_dir,
                    action_names=action_names,
                    action_tensor=action_tensor,
                    dataset_robot_command_i=dataset_robot_command_i,
                    base_robot_command_i=base_robot_command_i,
                    robot_command=robot_command,
                    command_to_send=command_to_send,
                    command_status=command_status,
                    position_delta=position_delta,
                    rotation_delta=rotation_delta,
                    clamped=bool(command_guard['status'] != 'pass'),
                )

            if interactive_keyboard is not None and interactive_keyboard.should_stop_rollout():
                return finish_rollout('quit' if interactive_keyboard.quit_requested.is_set() else 'stopped')

            if not args.preview:
                robot.send_action(command_to_send)
                previous_sent_command = dict(command_to_send)

            if trace is not None:
                trace.sample(
                    step_idx=step_idx,
                    position_xyz=np.asarray(
                        [absolute_state_observation_i[key] for key in EE_POSITION_KEYS],
                        dtype=np.float64,
                    ),
                    gripper_command=float(command_to_send['gripper.pos']),
                    gripper_raw=float(model_gripper_raw),
                    command_status=str(command_status),
                    source=command_source,
                )

            if dagger_buffer is not None:
                # Offered every step, kept only when the operator was driving: the buffer owns
                # where a span starts, so this call site never has to know.
                if command_source == 'expert':
                    dagger_action, previous_dagger_quaternion_xyzw = sent_command_to_dataset_action(
                        command_to_send,
                        T_B_Ws=T_B_Ws,
                        dataset_observation_i=dataset_state_observation_i,
                        encode_delta=dagger_encode_delta,
                        denormalize_gripper=dagger_denormalize_gripper,
                        previous_quaternion_xyzw=previous_dagger_quaternion_xyzw,
                    )
                    dagger_buffer.append(
                        build_dagger_frame(
                            dataset_features=dagger_features,
                            # The dataset-frame observation, not the base-frame one: a rollout
                            # runs with the arm's start pose offset from the dataset's, and an
                            # observation written in the live base frame would carry that offset
                            # into every sample.
                            observation_values={
                                **dataset_state_observation_i,
                                # The images the *policy* was shown, not the robot's raw frames.
                                # The dataset being imitated is a training view, and a view's
                                # images are its own crop of the camera (here 542x286 of the ee
                                # frame and 444x382 of the side one) in RGB. `build_policy_observation`
                                # has already put the live frame through exactly that crop, so
                                # taking its output writes a DAgger sample in the view's own
                                # pixels; taking `robot_observation` writes a 480x640 BGR-or-RGB
                                # frame that `validate_frame` rejects at flush time -- after the
                                # correction has been driven and cannot be repeated.
                                **{
                                    camera_name: policy_observation[f'{_OBS_IMAGES_PREFIX}{camera_name}']
                                    for camera_name in dagger_image_keys
                                },
                            },
                            action_values=dagger_action,
                            task=dagger_task,
                        ),
                        is_expert=True,
                    )
                else:
                    dagger_buffer.append({}, is_expert=False)

            if live_frame_emitter.wants(step_idx):
                # Measured joints with the *commanded* end-effector target beside them, which is
                # the pair that makes a clamp visible: the arm lags a command it is following and
                # stops tracking one it is not, and only the two together tell those apart.
                live_frame_emitter.emit_step(
                    step_idx,
                    # Thunks, not values: reading joints out of an observation can raise on a
                    # robot config that does not report them, and the emitter's guarantee is
                    # that the picture never costs the run.
                    joints_rad=lambda: _observation_joint_positions(robot_observation),
                    gripper=lambda: float(robot_observation['gripper.pos']),
                    source=command_source,
                    status=str(command_status),
                    rollout_index=None if trace is None else int(trace.rollout_index),
                    target_position_m=lambda: [float(command_to_send[key]) for key in EE_POSITION_KEYS],
                    actual_position_m=lambda: [float(robot_observation[key]) for key in EE_POSITION_KEYS],
                )

            elapsed_s = time.perf_counter() - loop_start_t
            sleep_s = max(1.0 / policy_fps - elapsed_s, 0.0)

            if (
                args.preview
                or command_status != 'pass'
                or command_source != 'policy'
                or step_idx % max(args.log_interval, 1) == 0
            ):
                log_message = (
                    ('[PREVIEW] step=' if args.preview else '[INFO] step=')
                    + f"{step_idx} "
                    + f"status={command_status} "
                    + f"raw_ee=({robot_command['ee.x']:.4f}, {robot_command['ee.y']:.4f}, {robot_command['ee.z']:.4f}) "
                    + f"safe_ee=({command_to_send['ee.x']:.4f}, {command_to_send['ee.y']:.4f}, {command_to_send['ee.z']:.4f}) "
                    + f"model_gripper_raw={model_gripper_raw:.2f} "
                    + f"gripper_obs={float(gripper_latch_debug['observed']):.3f} "
                    + f"gripper_des={gripper_command_before_latch:.3f} "
                    + f"gripper_cmd={command_to_send['gripper.pos']:.3f} "
                    + f"gripper_latched={float(gripper_latch_debug['latched']):.3f} "
                    + f"gripper_latch_err={float(gripper_latch_debug['latch_error']):.3f} "
                    + f"gripper_latch={gripper_latch_debug['status']}"
                    + (
                        ''
                        if args.act_temporal_stuck_max_offset is None
                        else (
                            f" temporal_offset={temporal_offset_used}->{int(temporal_offset_debug['current_offset'])} "
                            f"temporal_offset_status={temporal_offset_debug['status']}"
                        )
                    )
                    + (
                        ''
                        if not rtc_enabled
                        else (
                            f" rtc={rtc_state['last_debug'].get('status', 'unknown')}"
                            f" rtc_q={int(rtc_state['last_debug'].get('queue_size', -1))}"
                            f" rtc_lat_ms={float(rtc_state['last_debug'].get('latency_s', 0.0)) * 1000.0:.1f}"
                            f" rtc_delay={int(rtc_state['last_debug'].get('guidance_delay_steps', 0))}"
                            f"->{int(rtc_state['last_debug'].get('merge_delay_steps', 0))}"
                        )
                    )
                    + (
                        ''
                        if place_assist_offset_base_xyz_m is None
                        else (
                            f" place_assist={place_assist_debug['status']} "
                            f"place_assist_xyz_mm=({_format_vector(np.asarray(place_assist_debug['offset_xyz_m']), scale=1000.0)})"
                        )
                    )
                    + (
                        ''
                        if command_status == 'hold_first_frame'
                        else (
                            f" pos_delta_mm=({_format_vector(position_delta, scale=1000.0)}) "
                            f"rot_delta_deg=({_format_vector(np.rad2deg(rotation_delta))})"
                            # What the policy actually asked for, separate from the lag baked into
                            # pos_delta_mm above. Reading only the combined number is what made a
                            # closely-tracking rollout look like a runaway one.
                            + f" step_mm={np.linalg.norm(command_guard['step_position_delta']) * 1000.0:.2f} "
                            + f"step_deg={np.rad2deg(np.linalg.norm(command_guard['step_rotation_delta'])):.2f}"
                        )
                    )
                )
                if previous_tracking_position_delta is not None and previous_tracking_rotation_delta is not None:
                    log_message += (
                        f" prev_cmd_err_mm={np.linalg.norm(previous_tracking_position_delta) * 1000.0:.2f} "
                        f"prev_cmd_err_rot_deg={np.linalg.norm(np.rad2deg(previous_tracking_rotation_delta)):.2f}"
                    )
                if command_source != 'policy':
                    log_message += (
                        f" source={command_source} takeover={takeover_debug.get('status', '')}"
                        f" takeover_gripper={int(bool(takeover_debug.get('gripper_owned')))}"
                        # `reads` is how many SpaceMouse reports this step took off the queue.
                        # One, steadily, is the loop keeping up; a run of larger numbers is the
                        # arm following a hand from several steps ago. `takeover=stale` is the
                        # device saying nothing at all -- see tools/fr3/dagger_takeover.
                        f" takeover_reads={int(takeover_debug.get('reads', 0))}"
                        # The gain this step used and the step it was measured over. `takeover_gain`
                        # steadily above the `motion_gain` printed at startup is the loop missing
                        # its rate -- the same story `loop_ms` tells, measured from the device end.
                        f" takeover_gain={float(takeover_debug.get('gain', 0.0)):.2f}"
                        f" takeover_step_ms={float(takeover_debug.get('step_ms', 0.0)):.1f}"
                        f" step_mm={takeover_debug.get('step_mm', 0.0):.1f}"
                    )
                log_message += f" loop_ms={elapsed_s * 1000.0:.1f} sleep_ms={sleep_s * 1000.0:.1f}"
                print(log_message)

            precise_sleep(sleep_s)
            step_idx += 1
        return finish_rollout('completed')

    rollout_trace_dir = Path(args.rollout_trace_dir).expanduser() if args.rollout_trace_dir else None
    dagger_open = build_dagger_writer(args, ds_meta=ds_meta, fps=policy_fps)
    dagger_dataset_handle, dagger_writer = dagger_open if dagger_open is not None else (None, None)
    dagger_features = dagger_dataset_features(dict(ds_meta.features)) if dagger_writer is not None else {}
    dagger_image_keys = image_source_keys(dagger_features) if dagger_writer is not None else []
    # Checked here rather than at the first written frame, because a DAgger frame is only built
    # while the operator is driving and the buffer is only flushed when the rollout ends: a
    # schema problem found there costs the whole correction (measured: 476 expert steps of an
    # insertion, lost at flush). The images come from the policy observation, so a camera the
    # dataset carries and the policy does not read is one this loop cannot fill.
    missing_dagger_images = [
        camera_name
        for camera_name in dagger_image_keys
        if camera_name not in required_image_keys
    ]
    if missing_dagger_images:
        raise SystemExit(
            'DAgger corrections cannot be written for camera(s) '
            f"{', '.join(missing_dagger_images)}: {ds_meta.repo_id} carries them but the policy "
            'does not read them, so the rollout never builds an image in the view geometry. '
            'Point --dagger-dataset-root at a dataset matching this policy, or record without '
            '--dagger-takeover.'
        )
    dagger_encode_delta, dagger_denormalize_gripper = (
        build_dagger_action_encoder(
            action_names,
            robot_cfg=robot_cfg,
            gripper_feature_name=dataset_gripper_feature_name,
        )
        if dagger_writer is not None
        else (None, lambda value: value)
    )
    # The prompt the corrections are labelled with. A DAgger episode that carries a different
    # task string from the demonstrations is an episode a language-conditioned policy will not
    # associate with the thing it was correcting.
    dagger_task = str(task_prompt or '')
    if dagger_writer is not None and not dagger_task:
        raise SystemExit(
            '--dagger-dataset-root needs a task prompt to label the corrections with; pass '
            '--task-prompt (the dataset has more than one task).'
        )
    live_frame_emitter = LiveFrameEmitter(interval=int(args.live_frame_interval))
    if live_frame_emitter.enabled:
        print(f'[INFO] live_frame_stream=enabled interval={live_frame_emitter.interval}')
    interactive_keyboard: InteractiveRolloutKeyboard | None = None
    expert_takeover: ExpertTakeover | None = None
    if args.dagger_takeover and not args.interactive_rollouts:
        # Interactive mode is the one that has an operator sitting at the rig with a hand on the
        # device. A bounded scripted run has nobody to take the arm from the policy, and a
        # device connected but unattended is worse than one that was never opened.
        raise SystemExit('--dagger-takeover requires --interactive-rollouts.')
    # Installed before the try that owns the shutdown, restored at the end of its finally, so
    # every step of that shutdown runs under it. See the class for why it exists.
    terminate_guard = TerminateAsKeyboardInterrupt()
    terminate_guard.install()
    # Opened before the try so the finally can always close it: a stack created inside the try
    # is undefined if the statement above it raises, and the NameError that follows in the
    # finally would replace the error worth reading.
    dagger_encoding = ExitStack()
    try:
        # The arm is connected here rather than several hundred lines earlier, because this try
        # is what owns `robot.disconnect()`. Connected outside it, any exception in between --
        # and everything above raises: an unloadable --dagger-dataset-root, a camera the policy
        # does not read, a missing task prompt -- unwound the interpreter with the FCI control
        # loop still live. panda_py's C++ control thread is then destroyed while joinable, which
        # is `terminate called without an active exception`: the connection is severed instead
        # of the controller being stopped, and the arm is dropped out of control mid-hold. A
        # path typo should not be able to do that to a running arm.
        robot.connect()
        if args.mujoco_viewer:
            mujoco_visualizer = FR3InferenceMujocoVisualizer(
                model_path=mujoco_model_path,
                max_chunk_points=args.mujoco_max_chunk_points,
            )
            mujoco_visualizer.start()
        if args.align_gripper_to_dataset_start and not args.preview:
            if dataset_start_gripper_mean_normalized is None:
                print(
                    '[WARN] align_gripper_to_dataset_start=fallback_open_gripper '
                    'reason=dataset_start_states_do_not_include_gripper_values target=1.000'
                )
                align_gripper_to_dataset_start(
                    robot,
                    target_gripper_pos=1.0,
                    tolerance=dataset_start_gripper_tolerance,
                )
            else:
                align_gripper_to_dataset_start(
                    robot,
                    target_gripper_pos=float(dataset_start_gripper_mean_normalized),
                    tolerance=dataset_start_gripper_tolerance,
                )
        expert_takeover = build_expert_takeover(args, step_period_s=1.0 / policy_fps)
        # The recorder wraps its whole session in this for the same reason: episodes saved
        # inside it get their video finalized on the way out, and one saved outside it does not.
        if dagger_dataset_handle is not None:
            from lerobot.datasets.video_utils import VideoEncodingManager

            dagger_encoding.enter_context(VideoEncodingManager(dagger_dataset_handle))
        if args.interactive_rollouts:
            interactive_keyboard = InteractiveRolloutKeyboard(
                start_key=args.rollout_start_key,
                stop_key=args.rollout_stop_key,
                home_key=args.rollout_home_key,
                quit_key=args.rollout_quit_key,
                takeover_key=args.rollout_takeover_key if expert_takeover is not None else None,
            )
            interactive_keyboard.start()
            rollout_index = 0
            # Whether anything has displaced the arm since it was last put at the start pose.
            # True at process start because nothing in *this* process has moved it yet -- the
            # launcher's homing step ran before exec. It goes false the moment a rollout ends,
            # which is the case the waiting banner used to get wrong for every rollout but the
            # first, and it is what the operator is deciding on when they reach for home.
            arm_at_start = True
            while not interactive_keyboard.quit_requested.is_set():
                move_to_robot_init_state_if_requested(robot, robot_init_state)
                publish_current_camera_preview_snapshot()
                command = interactive_keyboard.wait_for_command(arm_at_start=arm_at_start)
                if command == 'quit':
                    break
                if command == 'home':
                    arm_at_start = home_arm_to_start_pose(robot)
                    continue
                if command == 'scene_reset':
                    payload = interactive_keyboard.pop_scene_reset_payload()
                    try:
                        request = scene_reset_request_from_payload(payload or {})
                    except SceneResetError as exc:
                        print(f'[WARN] scene_reset=failed details={exc}')
                        arm_at_start = False
                        continue
                    result = execute_scene_reset(robot, request)
                    # Reported by the reset rather than inferred from ok+returnToStart: a failed
                    # reset now homes too, and a successful one whose homing raised does not.
                    arm_at_start = bool(result.get('returnedToStart'))
                    continue
                if command == 'probe_pose':
                    payload = interactive_keyboard.pop_probe_pose_payload()
                    try:
                        probe_request = pose_probe_request_from_payload(payload or {})
                    except SceneResetError as exc:
                        print(f'[WARN] pose_probe=failed details={exc}')
                        continue
                    # Set before the motion, not after it: from here the arm is off the start
                    # pose whatever the probe does next, including failing halfway.
                    arm_at_start = False
                    execute_pose_probe(robot, probe_request, on_arrival=lambda: write_pose_probe_still(probe_request))
                    continue
                rollout_index += 1
                arm_at_start = False
                print(f'[INFO] interactive_rollout_start index={rollout_index}')
                trace = RolloutGeometryTrace(rollout_index, trace_dir=rollout_trace_dir)
                dagger_buffer = (
                    DaggerFrameBuffer(max_frames=int(args.dagger_max_buffered_frames))
                    if dagger_writer is not None
                    else None
                )
                rollout_status = run_policy_rollout(
                    interactive_keyboard,
                    trace=trace,
                    expert_takeover=expert_takeover,
                    dagger_buffer=dagger_buffer,
                )
                print(
                    f'[INFO] interactive_rollout_end index={rollout_index} status={rollout_status} '
                    + trace.summary_log_fields()
                )
                trace.write()
                if dagger_writer is not None and dagger_buffer is not None:
                    # Here, not at the end of each span: save_episode encodes video, and the end
                    # of a span is the moment the operator has let go and the policy is about to
                    # drive the arm again. The loop has stopped by now, so a slow write costs
                    # nothing but the operator's patience.
                    dagger_writer.write(dagger_buffer, rollout_index=rollout_index)
                if rollout_status == 'quit':
                    break
            print('[INFO] interactive_rollouts=stopped')
        else:
            move_to_robot_init_state_if_requested(robot, robot_init_state)
            run_policy_rollout()
    except KeyboardInterrupt:
        print('[INFO] KeyboardInterrupt received, stopping inference loop.')
    finally:
        dagger_encoding.close()
        if interactive_keyboard is not None:
            interactive_keyboard.close()
        if expert_takeover is not None:
            expert_takeover.close()
        if mujoco_visualizer is not None:
            mujoco_visualizer.close()
        if args.camera_preview_window:
            close_camera_preview_window()
        if preview_sink is not None:
            preview_sink.close()
        # Guarded because connect() is now inside this try and can be what failed. disconnect()
        # carries @check_if_not_connected, so calling it on an arm that never came up raises
        # DeviceNotConnectedError from the finally and replaces the error worth reading.
        if robot.is_connected:
            robot.disconnect()
        # Restored last, so every step above ran under the guard rather than under the default
        # action, which would have killed a shutdown that was still in progress.
        terminate_guard.restore()

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_inference(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
