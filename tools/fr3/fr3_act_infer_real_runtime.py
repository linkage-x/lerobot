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
from contextlib import nullcontext
from copy import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.hikrobot.configuration_hikrobot import HikrobotCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotObservation
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.robots.franka_research3.processor_franka_research3 import (
    EE_POSITION_KEYS,
    EE_QUAT_KEYS,
    KeepAbsoluteEEObservation,
    PREV_CMD_GRIPPER_KEY,
    PREV_CMD_POSITION_KEYS,
    PREV_CMD_QUAT_KEYS,
    _continuous_quaternion,
)
from lerobot.utils.control_utils import predict_action
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPO_ROOT / 'outputs/train/2026-03-19/10-48-39_act/checkpoints/060000'
_DEFAULT_CAMERA_CONFIG = _REPO_ROOT / 'tools/fr3/fr3_act_infer_camera_config.yaml'
_DEFAULT_ROBOT_IP = '192.168.1.208'
_DEFAULT_GRIPPER_PORT = '/dev/ttyUSB0'
_DEFAULT_GRIPPER_BACKEND = 'das'
_DAS_URDF = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf'
_PIKA_URDF = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.urdf'
_DEFAULT_TACTILE_VALID_MASK_PATH = _REPO_ROOT / 'docs/tactile/tactile_valid_mask_50x10.json'
_DEFAULT_TACTILE_BASELINE_PATH = _REPO_ROOT / 'docs/tactile/idle_baseline.json'
_DEFAULT_STATE_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
_DEFAULT_ACTION_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
_DEFAULT_OPENCV_FOURCC = 'MJPG'
_DEFAULT_OPENCV_BACKEND = Cv2Backends.V4L2
_OBS_IMAGES_PREFIX = 'observation.images.'
_DEFAULT_FIRST_FRAME_MAX_POS_DELTA_MM = 30.0
_DEFAULT_FIRST_FRAME_MAX_ROT_DELTA_DEG = 10.0
_DEFAULT_MAX_STEP_POS_DELTA_MM = 5.0
_DEFAULT_MAX_STEP_ROT_DELTA_DEG = 3.0
_DEFAULT_DATASET_START_GRIPPER_TOLERANCE = 0.05
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
# Replay uses I=gripper_base_link and E=das_gripper_ee with this fixed DAS extrinsic.
_T_IE = np.array(
    [
        [0.0, 0.0, 1.0, 0.13],
        [0.0, -1.0, 0.0, 0.00],
        [1.0, 0.0, 0.0, -0.04],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)
_T_EI = np.linalg.inv(_T_IE)
_TACTILE_FALLBACK_CHOICES = ('baseline_idle',)


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


def _load_observation_state_feature_names(dataset_root: Path) -> list[str]:
    info = _load_dataset_info(dataset_root)
    names = info.get('features', {}).get('observation.state', {}).get('names')
    if not isinstance(names, list):
        return [*EE_POSITION_KEYS, *EE_QUAT_KEYS, 'gripper.pos']
    return [str(name) for name in names]


def _extract_dataset_state_contract_indices(dataset_root: Path) -> dict[str, int]:
    state_names = _load_observation_state_feature_names(dataset_root)
    required_names = ['ee.x', 'ee.y', 'ee.z', 'ee.qx', 'ee.qy', 'ee.qz', 'ee.qw']
    missing_names = [name for name in required_names if name not in state_names]
    if missing_names:
        raise KeyError(f'Dataset observation.state names are missing required entries: {missing_names}')
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
    parser.add_argument('--policy-fps', type=float, default=None, help='Optional low-rate policy update FPS override.')
    parser.add_argument('--max-steps', type=int, default=None, help='Optional inference loop step limit.')
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Run policy and print safe targets without sending robot actions.',
    )
    parser.add_argument('--preflight', dest='preflight', action='store_true', default=True)
    parser.add_argument('--no-preflight', dest='preflight', action='store_false')
    parser.add_argument('--preflight-max-actions', type=int, default=None)
    parser.add_argument('--preflight-max-step-pos-delta-mm', type=float, default=None)
    parser.add_argument('--preflight-max-step-rot-delta-deg', type=float, default=None)
    parser.add_argument('--preflight-max-step-gripper-delta', type=float, default=None)
    parser.add_argument('--robot-ip', default=_DEFAULT_ROBOT_IP)
    parser.add_argument('--gripper-port', default=_DEFAULT_GRIPPER_PORT)
    parser.add_argument('--gripper-backend', choices=['pika', 'das'], default=_DEFAULT_GRIPPER_BACKEND)
    parser.add_argument('--urdf-path', type=Path, default=None, help='Optional FR3 tool URDF override.')
    parser.add_argument('--target-frame-name', default=None, help='Optional target EE frame override.')
    parser.add_argument(
        '--dataset-frame',
        choices=['tool_base', 'target_ee'],
        default='target_ee',
        help='Frame used by dataset observation/action EE pose.',
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
        help='Clamp each step relative rotvec component to this limit.',
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
        '--no-move-to-das-start',
        dest='move_to_das_start',
        action='store_false',
        help='Skip moving the arm to the DAS replay start joint configuration before inference.',
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
    parser.set_defaults(move_to_das_start=True, align_gripper_to_dataset_start=True)
    return parser.parse_args(argv)


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


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


def load_camera_configs(
    camera_config_path: str | Path,
) -> dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig]:
    config_path = _resolve_repo_path(camera_config_path)
    with config_path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    camera_entries = raw.get('robot', {}).get('cameras', {})
    if not camera_entries:
        raise ValueError(f'No robot.cameras entries found in {config_path}')

    camera_configs: dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig] = {}
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


def normalize_dataset_gripper(dataset_gripper: float, cfg: FrankaResearch3Config) -> float:
    """Convert a dataset gripper value into the robot command's normalized width.

    Pika datasets in this repo store gripper values as normalized width already.
    DAS datasets store aperture in meters and need conversion through the DAS range.
    """
    dataset_gripper = float(max(0.0, dataset_gripper))
    if cfg.gripper_backend == 'das':
        span_m = float(cfg.das_max_distance_m - cfg.das_min_distance_m)
        if span_m <= 0.0:
            return 0.0
        return float(np.clip((dataset_gripper - cfg.das_min_distance_m) / span_m, 0.0, 1.0))
    return float(np.clip(dataset_gripper, 0.0, 1.0))


def denormalize_live_gripper_observation(gripper_pos: float, cfg: FrankaResearch3Config) -> float:
    """Convert live normalized robot gripper observation into dataset units."""
    gripper_pos = float(np.clip(gripper_pos, 0.0, 1.0))
    if cfg.gripper_backend == 'das':
        span_m = float(cfg.das_max_distance_m - cfg.das_min_distance_m)
        if span_m <= 0.0:
            return 0.0
        return float(cfg.das_min_distance_m + gripper_pos * span_m)
    return gripper_pos


def convert_gripper_observation_to_dataset_units(
    observation: RobotObservation,
    *,
    robot_cfg: FrankaResearch3Config,
) -> RobotObservation:
    converted_observation = dict(observation)
    for key in ('gripper.pos', PREV_CMD_GRIPPER_KEY):
        if key not in converted_observation:
            continue
        converted_observation[key] = denormalize_live_gripper_observation(
            float(converted_observation[key]),
            robot_cfg,
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
        'gripper': 'gripper.pos',
        'prev_cmd.gripper': PREV_CMD_GRIPPER_KEY,
    }
    return aliases.get(name, name)


def _action_value(action_map: dict[str, float], *keys: str) -> float:
    for key in keys:
        if key in action_map:
            return float(action_map[key])
    raise KeyError(f'Missing action keys {keys!r} in decoded policy action.')



def predict_action_chunk_for_preflight(
    observation: dict[str, np.ndarray],
    *,
    policy: Any,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == 'cuda' and use_amp else nullcontext(),
    ):
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        if hasattr(policy, 'predict_action_chunk'):
            actions = policy.predict_action_chunk(observation)
        else:
            actions = policy.select_action(observation).unsqueeze(1)
        actions = postprocessor(actions)
    return actions


def run_action_chunk_preflight(
    policy_observation: dict[str, np.ndarray],
    *,
    policy: Any,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    robot_type: str,
    task: str | None,
    action_names: list[str],
    robot_cfg: FrankaResearch3Config,
    robot_observation: RobotObservation,
    T_B_Ws: np.ndarray,
    dataset_frame: str,
    max_actions: int | None,
    first_frame_max_pos_delta_m: float,
    first_frame_max_rot_delta_rad: float,
    max_step_pos_delta_m: float,
    max_step_rot_delta_rad: float,
    max_step_gripper_delta: float,
) -> None:
    chunk = predict_action_chunk_for_preflight(
        policy_observation,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        robot_type=robot_type,
        task=task,
    )
    action_count = int(chunk.shape[1]) if chunk.ndim >= 3 else 1
    check_count = min(action_count, int(max_actions or action_count))
    commands: list[dict[str, float]] = []
    for action_idx in range(check_count):
        action_tensor = chunk[:, action_idx, :] if chunk.ndim >= 3 else chunk
        dataset_command = decode_action_to_robot_command(action_tensor, action_names=action_names, robot_cfg=robot_cfg)
        base_command = convert_dataset_command_to_base_frame(dataset_command, T_B_Ws)
        robot_command = (
            convert_base_command_from_I_to_E(base_command)
            if dataset_frame == 'tool_base'
            else base_command
        )
        commands.append(robot_command)

    first_position_delta = np.asarray(
        [commands[0][key] - robot_observation[key] for key in ('ee.x', 'ee.y', 'ee.z')], dtype=np.float64
    )
    _, current_rotation = _extract_observation_pose(robot_observation)
    _, first_rotation = _extract_command_pose(commands[0])
    first_rotation_delta = (current_rotation.inv() * first_rotation).as_rotvec()
    if np.any(np.abs(first_position_delta) > first_frame_max_pos_delta_m) or np.any(
        np.abs(first_rotation_delta) > first_frame_max_rot_delta_rad
    ):
        raise RuntimeError(
            'Preflight failed: first action is discontinuous relative to current target; '
            f'pos_delta_mm=({_format_vector(first_position_delta, scale=1000.0)}) '
            f'rot_delta_deg=({_format_vector(np.rad2deg(first_rotation_delta))})'
        )

    max_pos_delta = np.zeros(3, dtype=np.float64)
    max_rot_delta = np.zeros(3, dtype=np.float64)
    max_gripper_delta = 0.0
    previous = commands[0]
    z_values = [float(previous['ee.z'])]
    for idx_cmd, command in enumerate(commands[1:], start=1):
        pos_delta = np.asarray([command[key] - previous[key] for key in ('ee.x', 'ee.y', 'ee.z')], dtype=np.float64)
        _, previous_rotation = _extract_command_pose(previous)
        _, command_rotation = _extract_command_pose(command)
        rot_delta = (previous_rotation.inv() * command_rotation).as_rotvec()
        gripper_delta = float(command['gripper.pos'] - previous['gripper.pos'])
        max_pos_delta = np.maximum(max_pos_delta, np.abs(pos_delta))
        max_rot_delta = np.maximum(max_rot_delta, np.abs(rot_delta))
        max_gripper_delta = max(max_gripper_delta, abs(gripper_delta))
        z_values.append(float(command['ee.z']))
        if np.any(np.abs(pos_delta) > max_step_pos_delta_m) or np.any(np.abs(rot_delta) > max_step_rot_delta_rad) or abs(gripper_delta) > max_step_gripper_delta:
            raise RuntimeError(
                f'Preflight failed: action[{idx_cmd}] is discontinuous relative to previous target; '
                f'pos_delta_mm=({_format_vector(pos_delta, scale=1000.0)}) '
                f'rot_delta_deg=({_format_vector(np.rad2deg(rot_delta))}) '
                f'gripper_delta={gripper_delta:+.3f}'
            )
        previous = command
    z_arr = np.asarray(z_values, dtype=np.float64)
    print(
        '[PREFLIGHT] action_chunk=pass '
        f'checked_actions={check_count}/{action_count} '
        f'first_pos_delta_mm=({_format_vector(first_position_delta, scale=1000.0)}) '
        f'first_rot_delta_deg=({_format_vector(np.rad2deg(first_rotation_delta))}) '
        f'max_step_pos_delta_mm=({_format_vector(max_pos_delta, scale=1000.0)}) '
        f'max_step_rot_delta_deg=({_format_vector(np.rad2deg(max_rot_delta))}) '
        f'max_step_gripper_delta={max_gripper_delta:.3f} '
        f'z_min/max/net_mm={float(z_arr.min() * 1000.0):.1f}/{float(z_arr.max() * 1000.0):.1f}/{float((z_arr[-1] - z_arr[0]) * 1000.0):+.1f}'
    )

def build_policy_observation(
    state_observation: RobotObservation,
    *,
    state_names: list[str],
    input_features: dict[str, Any],
    tactile_fallback_observation: dict[str, np.ndarray] | None = None,
    camera_configs: dict[str, OpenCVCameraConfig | RealSenseCameraConfig | HikrobotCameraConfig] | None = None,
) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {
        'observation.state': np.asarray(
            [state_observation[_state_name_to_observation_key(name)] for name in state_names],
            dtype=np.float32,
        )
    }

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
        observation[f'{_OBS_IMAGES_PREFIX}{camera_key}'] = image

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
    absolute_observation_i = dict(absolute_observation_e)
    for position_keys, quaternion_keys in (
        (EE_POSITION_KEYS, EE_QUAT_KEYS),
        (PREV_CMD_POSITION_KEYS, PREV_CMD_QUAT_KEYS),
    ):
        if not all(key in absolute_observation_e for key in position_keys + quaternion_keys):
            continue
        input_quaternion_xyzw = np.asarray([absolute_observation_e[key] for key in quaternion_keys], dtype=np.float64)
        absolute_pose_e = _pose_from_quaternion_observation(
            absolute_observation_e,
            position_keys=position_keys,
            quaternion_keys=quaternion_keys,
        )
        absolute_pose_i = absolute_pose_e @ _T_EI
        quaternion_xyzw = Rotation.from_matrix(absolute_pose_i[:3, :3]).as_quat()
        quaternion_xyzw = _continuous_quaternion(quaternion_xyzw, input_quaternion_xyzw)
        absolute_observation_i.update(
            {
                position_keys[0]: float(absolute_pose_i[0, 3]),
                position_keys[1]: float(absolute_pose_i[1, 3]),
                position_keys[2]: float(absolute_pose_i[2, 3]),
                quaternion_keys[0]: float(quaternion_xyzw[0]),
                quaternion_keys[1]: float(quaternion_xyzw[1]),
                quaternion_keys[2]: float(quaternion_xyzw[2]),
                quaternion_keys[3]: float(quaternion_xyzw[3]),
            }
        )
    return absolute_observation_i


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


def _load_episode_start_state_rows(dataset_root: Path) -> list[tuple[int, np.ndarray]]:
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
    start_state_rows: list[tuple[int, np.ndarray]] = []
    for episode_index, chunk_index, file_index in episode_rows:
        data_file = _resolve_dataset_data_file(dataset_root, chunk_index=chunk_index, file_index=file_index)
        table = pq.read_table(str(data_file), columns=['episode_index', 'observation.state']).to_pydict()
        for row_episode_index, state in zip(table['episode_index'], table['observation.state'], strict=True):
            if int(row_episode_index) != episode_index:
                continue
            start_state_rows.append((episode_index, np.asarray(state, dtype=np.float64)))
            break
        else:
            raise ValueError(f'Episode {episode_index} metadata found, but no rows matched in {data_file}')

    if not start_state_rows:
        raise ValueError(f'No episode starts resolved from {dataset_root}')
    return start_state_rows


def _load_episode_start_states(dataset_root: Path) -> np.ndarray:
    start_state_rows = _load_episode_start_state_rows(dataset_root)
    return np.asarray([state for _, state in start_state_rows], dtype=np.float64)


def _quaternion_angle_deg(quaternion_a_xyzw: np.ndarray, quaternion_b_xyzw: np.ndarray) -> float:
    quaternion_a = np.asarray(quaternion_a_xyzw, dtype=np.float64)
    quaternion_b = np.asarray(quaternion_b_xyzw, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(quaternion_a, quaternion_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def estimate_dataset_start_pose_contract(dataset_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    start_states = _load_episode_start_states(dataset_root)
    state_indices = _extract_dataset_state_contract_indices(dataset_root)
    positions = np.asarray([[state[state_indices[key]] for key in EE_POSITION_KEYS] for state in start_states], dtype=np.float64)
    quaternions = np.asarray([[state[state_indices[key]] for key in EE_QUAT_KEYS] for state in start_states], dtype=np.float64)
    gripper_values = (
        np.asarray([state[state_indices['gripper.pos']] for state in start_states], dtype=np.float64)
        if 'gripper.pos' in state_indices
        else None
    )

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
        'episodes': int(len(start_states)),
        'mean_position_xyz_m': mean_position.copy(),
        'position_std_xyz_mm': positions.std(axis=0) * 1000.0,
        'mean_quaternion_xyzw': mean_quaternion.copy(),
        'rotation_spread_mean_deg': float(rotation_spread_deg.mean()),
        'rotation_spread_p95_deg': float(np.percentile(rotation_spread_deg, 95)),
        'rotation_spread_max_deg': float(rotation_spread_deg.max()),
    }
    if gripper_values is not None:
        stats['gripper_mean'] = float(gripper_values.mean())
        stats['gripper_std'] = float(gripper_values.std())
    return representative_pose_xyzquat, stats


def summarize_live_start_alignment_to_dataset_starts(
    dataset_root: Path,
    T_B_Ws: np.ndarray,
    live_start_pose_i: np.ndarray,
    *,
    live_gripper: float | None = None,
) -> dict[str, Any]:
    start_state_rows = _load_episode_start_state_rows(dataset_root)
    state_indices = _extract_dataset_state_contract_indices(dataset_root)
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
    base_pose_i = _pose_from_position_and_rotvec(
        np.asarray([base_robot_command_i['ee.x'], base_robot_command_i['ee.y'], base_robot_command_i['ee.z']], dtype=np.float64),
        np.asarray([base_robot_command_i['ee.wx'], base_robot_command_i['ee.wy'], base_robot_command_i['ee.wz']], dtype=np.float64),
    )
    base_pose_e = base_pose_i @ _T_IE
    base_rotvec_xyz = Rotation.from_matrix(base_pose_e[:3, :3]).as_rotvec()
    base_robot_command_e = dict(base_robot_command_i)
    base_robot_command_e.update(
        {
            'ee.x': float(base_pose_e[0, 3]),
            'ee.y': float(base_pose_e[1, 3]),
            'ee.z': float(base_pose_e[2, 3]),
            'ee.wx': float(base_rotvec_xyz[0]),
            'ee.wy': float(base_rotvec_xyz[1]),
            'ee.wz': float(base_rotvec_xyz[2]),
        }
    )
    return base_robot_command_e


def decode_action_to_robot_command(
    action_tensor: torch.Tensor,
    *,
    action_names: list[str],
    robot_cfg: FrankaResearch3Config,
) -> dict[str, float]:
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    if action_np.shape != (len(action_names),):
        raise ValueError(f'Expected policy action shape {(len(action_names),)}, got {action_np.shape}')

    action_map = {name: float(action_np[i]) for i, name in enumerate(action_names)}
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
    gripper_normalized = normalize_dataset_gripper(
        _action_value(action_map, 'gripper', 'gripper.pos'),
        robot_cfg,
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


def _extract_observation_pose(robot_observation: RobotObservation) -> tuple[np.ndarray, Rotation]:
    position = np.asarray(
        [robot_observation['ee.x'], robot_observation['ee.y'], robot_observation['ee.z']],
        dtype=np.float64,
    )
    rotation = Rotation.from_rotvec(
        [robot_observation['ee.wx'], robot_observation['ee.wy'], robot_observation['ee.wz']]
    )
    return position, rotation


def _extract_command_pose(robot_command: dict[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.asarray([robot_command['ee.x'], robot_command['ee.y'], robot_command['ee.z']], dtype=np.float64)
    rotation = Rotation.from_rotvec([robot_command['ee.wx'], robot_command['ee.wy'], robot_command['ee.wz']])
    return position, rotation


def compute_pose_delta_from_current(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
) -> tuple[np.ndarray, np.ndarray]:
    current_position, current_rotation = _extract_observation_pose(robot_observation)
    target_position, target_rotation = _extract_command_pose(robot_command)
    position_delta = target_position - current_position
    rotation_delta = (current_rotation.inv() * target_rotation).as_rotvec()
    return position_delta, rotation_delta


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


def clamp_command_relative_to_current(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
    *,
    max_pos_delta_m: float,
    max_rot_delta_rad: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, bool]:
    current_position, current_rotation = _extract_observation_pose(robot_observation)
    position_delta, rotation_delta = compute_pose_delta_from_current(robot_command, robot_observation)
    clamped_position_delta = np.clip(position_delta, -float(max_pos_delta_m), float(max_pos_delta_m))
    clamped_rotation_delta = np.clip(rotation_delta, -float(max_rot_delta_rad), float(max_rot_delta_rad))
    clamped = bool(
        not np.allclose(clamped_position_delta, position_delta)
        or not np.allclose(clamped_rotation_delta, rotation_delta)
    )

    safe_position = current_position + clamped_position_delta
    safe_rotation = current_rotation * Rotation.from_rotvec(clamped_rotation_delta)
    safe_rotvec = safe_rotation.as_rotvec()
    safe_command = dict(robot_command)
    safe_command.update(
        {
            'ee.x': float(safe_position[0]),
            'ee.y': float(safe_position[1]),
            'ee.z': float(safe_position[2]),
            'ee.wx': float(safe_rotvec[0]),
            'ee.wy': float(safe_rotvec[1]),
            'ee.wz': float(safe_rotvec[2]),
        }
    )
    return safe_command, position_delta, rotation_delta, clamped


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
        'policy_observation_state': _jsonify_value(np.asarray(policy_observation['observation.state'], dtype=np.float64)),
        'robot_observation_scalars': _extract_numeric_observation_scalars(robot_observation),
        'absolute_state_observation_e_scalars': _extract_numeric_observation_scalars(absolute_state_observation_e),
        'absolute_state_observation_i_scalars': _extract_numeric_observation_scalars(absolute_state_observation_i),
        'dataset_state_observation_i_scalars': _extract_numeric_observation_scalars(dataset_state_observation_i),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(_jsonify_value(metadata), indent=2), encoding='utf-8')
    return output_dir


def load_policy_stack(
    pretrained_dir: Path,
    *,
    ds_meta: LeRobotDatasetMetadata,
    device: torch.device,
) -> tuple[Any, PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    policy_cfg = load_train_config(pretrained_dir).policy
    if policy_cfg is None:
        raise ValueError(f"No policy config found in {pretrained_dir / 'train_config.json'}")

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


def run_inference(args: argparse.Namespace) -> int:
    pretrained_dir = resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = load_train_config(pretrained_dir)
    dataset_root = resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    ds_meta = load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    dataset_start_pose_contract_xyzquat, dataset_start_pose_stats = estimate_dataset_start_pose_contract(dataset_root)
    camera_configs = load_camera_configs(args.camera_config)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    policy, preprocessor, postprocessor = load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    required_image_keys = extract_required_image_keys(policy.config.input_features)
    required_tactile_keys = extract_required_tactile_keys(policy.config.input_features)
    validate_camera_keys(required_image_keys=required_image_keys, available_camera_keys=list(camera_configs))
    if args.tactile_fallback is not None and not args.preview:
        raise ValueError('--tactile-fallback is preview-only. Use it together with --preview.')

    policy_fps = float(args.policy_fps or ds_meta.fps)
    if policy_fps <= 0.0:
        raise ValueError('policy-fps must be positive.')
    first_frame_max_pos_delta_m = float(args.first_frame_max_pos_delta_mm) / 1000.0
    first_frame_max_rot_delta_rad = np.deg2rad(float(args.first_frame_max_rot_delta_deg))
    max_step_pos_delta_m = float(args.max_step_pos_delta_mm) / 1000.0
    max_step_rot_delta_rad = np.deg2rad(float(args.max_step_rot_delta_deg))
    dataset_start_gripper_tolerance = float(args.dataset_start_gripper_tolerance)
    state_names = extract_feature_names(ds_meta.features['observation.state'], _DEFAULT_STATE_NAMES)
    action_names = extract_feature_names(ds_meta.features['action'], _DEFAULT_ACTION_NAMES)

    tactile_fallback_observation = build_tactile_fallback_observation(args.tactile_fallback)
    tactile_enabled = bool(required_tactile_keys) and tactile_fallback_observation is None
    default_urdf_path = _DAS_URDF if args.gripper_backend == 'das' else _PIKA_URDF
    default_target_frame_name = 'das_gripper_ee' if args.gripper_backend == 'das' else 'pika_gripper_ee'
    urdf_path = _resolve_repo_path(args.urdf_path) if args.urdf_path is not None else default_urdf_path
    target_frame_name = args.target_frame_name or default_target_frame_name
    robot_cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(urdf_path),
        target_frame_name=target_frame_name,
        workspace_min=(0.1, -0.6, 0.05),
        workspace_max=(0.9, 0.6, 0.8),
        das_tactile_frequency_hz=policy_fps if tactile_enabled else None,
        das_tactile_valid_mask_path=str(_DEFAULT_TACTILE_VALID_MASK_PATH) if tactile_enabled else None,
        das_tactile_baseline_path=str(_DEFAULT_TACTILE_BASELINE_PATH) if tactile_enabled else None,
        das_tactile_timeout_s=2.0,
        cameras={name: cfg for name, cfg in camera_configs.items()},
    )

    if args.move_to_das_start and args.gripper_backend != 'das':
        print('[INFO] move_to_das_start skipped because gripper_backend is not das.')
    move_to_das_start_if_requested(robot_ip=args.robot_ip, enabled=bool(args.move_to_das_start and args.gripper_backend == 'das'))

    from lerobot.robots.franka_research3 import FrankaResearch3

    robot = FrankaResearch3(robot_cfg)
    dataset_start_gripper_mean_normalized: float | None = None
    if 'gripper_mean' in dataset_start_pose_stats:
        dataset_start_gripper_mean_normalized = normalize_dataset_gripper(
            float(dataset_start_pose_stats['gripper_mean']),
            robot_cfg,
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
    print(f'[INFO] policy_device={device}')
    print(f'[INFO] policy_fps={policy_fps:.3f}')
    print('[INFO] policy_image_keys=' + ', '.join(required_image_keys) if required_image_keys else '[INFO] policy_image_keys=<none>')
    print('[INFO] policy_tactile_keys=' + ', '.join(required_tactile_keys) if required_tactile_keys else '[INFO] policy_tactile_keys=<none>')
    print('[INFO] tactile_fallback=' + args.tactile_fallback if args.tactile_fallback is not None else '[INFO] tactile_fallback=<none>')
    print(f'[INFO] gripper_backend={args.gripper_backend} target_frame={target_frame_name} urdf_path={urdf_path}')
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
        )
    print(dataset_start_spread_line)
    print(
        '[INFO] state_frame='
        + (
            'absolute_pose(gripper_base_link) in dataset_world(W_s)'
            if args.dataset_frame == 'tool_base'
            else 'absolute_pose(target_ee) in dataset_world(W_s)'
        )
    )
    print(
        '[INFO] safety='
        f'first_frame<{args.first_frame_max_pos_delta_mm:.1f}mm/{args.first_frame_max_rot_delta_deg:.1f}deg, '
        f'per_step<{args.max_step_pos_delta_mm:.1f}mm/{args.max_step_rot_delta_deg:.1f}deg, '
        f'preview={args.preview}'
    )
    print(
        '[INFO] joint-space smoothing='
        f'FR3 OTG @ {robot_cfg.otg_control_frequency:.1f}Hz / sender @ {robot_cfg.otg_async_control_frequency:.1f}Hz'
    )

    if args.preview and args.align_gripper_to_dataset_start:
        print('[INFO] preview_gripper_alignment=requested; using virtual observation correction without moving hardware.')

    robot.connect()
    if args.align_gripper_to_dataset_start and not args.preview:
        if dataset_start_gripper_mean_normalized is None:
            raise ValueError('Dataset start states do not include gripper values; cannot auto-align gripper.')
        align_gripper_to_dataset_start(
            robot,
            target_gripper_pos=float(dataset_start_gripper_mean_normalized),
            tolerance=dataset_start_gripper_tolerance,
        )
    policy.reset()
    state_processor.reset()
    preprocessor.reset()
    postprocessor.reset()

    try:
        step_idx = 0
        while args.max_steps is None or step_idx < args.max_steps:
            loop_start_t = time.perf_counter()
            robot_observation = robot.get_observation()
            absolute_state_observation_e = state_processor.observation(dict(robot_observation))
            absolute_state_observation_i = (
                convert_absolute_observation_from_E_to_I(absolute_state_observation_e)
                if args.dataset_frame == 'tool_base'
                else dict(absolute_state_observation_e)
            )
            live_gripper_dataset_units = denormalize_live_gripper_observation(
                float(robot_observation['gripper.pos']),
                robot_cfg,
            )
            if T_B_Ws is None:
                current_start_pose_i = _pose_from_quaternion_observation(absolute_state_observation_i)
                T_B_Ws = current_start_pose_i @ _invert_pose(dataset_start_pose_contract)
                start_alignment_stats = summarize_live_start_alignment_to_dataset_starts(
                    dataset_root,
                    T_B_Ws,
                    current_start_pose_i,
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
                    gripper_delta_to_mean = abs(float(start_alignment_stats['live_gripper']) - dataset_gripper_mean)
                    dataset_alignment_line += (
                        f" live_gripper={start_alignment_stats['live_gripper']:.3f}"
                        f" dataset_gripper_mean={dataset_gripper_mean:.3f}"
                        f" delta_to_mean={gripper_delta_to_mean:.3f}"
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
                        f"dataset_mean={float(dataset_start_pose_stats['gripper_mean']):.3f} "
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
                        f"target_mean={float(dataset_start_pose_stats['gripper_mean']):.3f} "
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
            )
            if step_idx == 0 and args.preflight:
                run_action_chunk_preflight(
                    policy_observation,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=bool(policy.config.use_amp),
                    robot_type=robot.name,
                    task=getattr(train_cfg, 'task', None),
                    action_names=action_names,
                    robot_cfg=robot_cfg,
                    robot_observation=robot_observation,
                    T_B_Ws=T_B_Ws,
                    dataset_frame=args.dataset_frame,
                    max_actions=args.preflight_max_actions,
                    first_frame_max_pos_delta_m=first_frame_max_pos_delta_m,
                    first_frame_max_rot_delta_rad=first_frame_max_rot_delta_rad,
                    max_step_pos_delta_m=float(args.preflight_max_step_pos_delta_mm or args.max_step_pos_delta_mm) / 1000.0,
                    max_step_rot_delta_rad=np.deg2rad(float(args.preflight_max_step_rot_delta_deg or args.max_step_rot_delta_deg)),
                    max_step_gripper_delta=float(args.preflight_max_step_gripper_delta or 0.05),
                )
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
            action_tensor = predict_action(
                policy_observation,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp),
                robot_type=robot.name,
            )
            dataset_robot_command_i = decode_action_to_robot_command(
                action_tensor,
                action_names=action_names,
                robot_cfg=robot_cfg,
            )
            base_robot_command_i = convert_dataset_command_to_base_frame(dataset_robot_command_i, T_B_Ws)
            robot_command = (
                convert_base_command_from_I_to_E(base_robot_command_i)
                if args.dataset_frame == 'tool_base'
                else base_robot_command_i
            )
            safe_command, position_delta, rotation_delta, clamped = clamp_command_relative_to_current(
                robot_command,
                robot_observation,
                max_pos_delta_m=max_step_pos_delta_m,
                max_rot_delta_rad=max_step_rot_delta_rad,
            )
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
            if command_status == 'pass' and clamped:
                command_status = 'clamped'

            if not args.preview:
                robot.send_action(command_to_send)

            if args.preview or command_status != 'pass' or step_idx % max(args.log_interval, 1) == 0:
                log_message = (
                    ('[PREVIEW] step=' if args.preview else '[INFO] step=')
                    + f"{step_idx} "
                    + f"status={command_status} "
                    + f"raw_ee=({robot_command['ee.x']:.4f}, {robot_command['ee.y']:.4f}, {robot_command['ee.z']:.4f}) "
                    + f"safe_ee=({command_to_send['ee.x']:.4f}, {command_to_send['ee.y']:.4f}, {command_to_send['ee.z']:.4f}) "
                    + f"gripper={command_to_send['gripper.pos']:.3f}"
                    + (
                        ''
                        if command_status == 'hold_first_frame'
                        else (
                            f" pos_delta_mm=({_format_vector(position_delta, scale=1000.0)}) "
                            f"rot_delta_deg=({_format_vector(np.rad2deg(rotation_delta))})"
                        )
                    )
                )
                print(log_message)

            elapsed_s = time.perf_counter() - loop_start_t
            precise_sleep(max(1.0 / policy_fps - elapsed_s, 0.0))
            step_idx += 1
    except KeyboardInterrupt:
        print('[INFO] KeyboardInterrupt received, stopping inference loop.')
    finally:
        robot.disconnect()

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_inference(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
