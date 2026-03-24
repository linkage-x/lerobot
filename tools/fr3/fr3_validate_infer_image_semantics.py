#!/usr/bin/env python3
"""Validate FR3 inference image semantics with live captures, dataset start-frame references, and preview metrics.

This script does four things in one run:
1. Captures live left/right frames from the current FR3 inference camera config.
2. Finds the closest dataset episode starts by current EE pose and exports their first-frame images.
3. Computes normal-vs-swapped image similarity scores against those dataset start frames.
4. Runs offline preview evaluation on the same captured live observations for both normal and swapped image mappings,
   then saves summary JSON/CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import sys
from pathlib import Path
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / 'outputs' / 'analysis' / 'fr3_infer_image_semantics'


@dataclass(frozen=True)
class EpisodeImageReference:
    episode_index: int
    state: Any
    left_image: Any
    right_image: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from tools.fr3 import fr3_act_infer_real_runtime as infer_runtime

    parser = argparse.ArgumentParser(
        description='Validate live FR3 left/right camera semantics against dataset starts and preview metrics.'
    )
    parser.add_argument('--checkpoint', type=Path, default=infer_runtime._DEFAULT_CHECKPOINT)
    parser.add_argument('--camera-config', type=Path, default=infer_runtime._DEFAULT_CAMERA_CONFIG)
    parser.add_argument('--dataset-root', default=None, help='Optional dataset root override.')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--limit-episodes', type=int, default=20, help='How many earliest dataset episodes to consider.')
    parser.add_argument('--top-k', type=int, default=5, help='How many closest episode starts to export and compare.')
    parser.add_argument('--preview-steps', type=int, default=5, help='How many live observations to collect for preview comparison.')
    parser.add_argument('--policy-fps', type=float, default=None, help='Optional preview observation collection FPS override.')
    parser.add_argument('--robot-ip', default=infer_runtime._DEFAULT_ROBOT_IP)
    parser.add_argument('--gripper-port', default=infer_runtime._DEFAULT_GRIPPER_PORT)
    parser.add_argument('--gripper-backend', choices=['pika', 'das'], default=infer_runtime._DEFAULT_GRIPPER_BACKEND)
    parser.add_argument('--device', default=None, help='Optional torch device override.')
    parser.add_argument(
        '--first-frame-max-pos-delta-mm',
        type=float,
        default=infer_runtime._DEFAULT_FIRST_FRAME_MAX_POS_DELTA_MM,
    )
    parser.add_argument(
        '--first-frame-max-rot-delta-deg',
        type=float,
        default=infer_runtime._DEFAULT_FIRST_FRAME_MAX_ROT_DELTA_DEG,
    )
    parser.add_argument(
        '--max-step-pos-delta-mm',
        type=float,
        default=infer_runtime._DEFAULT_MAX_STEP_POS_DELTA_MM,
    )
    parser.add_argument(
        '--max-step-rot-delta-deg',
        type=float,
        default=infer_runtime._DEFAULT_MAX_STEP_ROT_DELTA_DEG,
    )
    return parser.parse_args(argv)


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _default_output_dir() -> Path:
    return (_DEFAULT_OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError('cv2 is required. Run this script inside the infer container or an environment with OpenCV.') from exc
    return cv2


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError('numpy is required. Run this script inside the infer container or an environment with dependencies installed.') from exc
    return np


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError('pyarrow is required to read the dataset parquet files.') from exc
    return pq


def _ensure_hwc_uint8(image: Any):
    np = _require_numpy()

    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f'Expected image with 2 or 3 dims, got shape {arr.shape}')
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _squeeze_to_gray(image: Any):
    cv2 = _require_cv2()

    arr = _ensure_hwc_uint8(image)
    if arr.shape[2] == 1:
        return arr[..., 0]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.shape[2] == 3 else arr[..., 0]


def _resize_to_match(src: Any, shape_hw: tuple[int, int]):
    cv2 = _require_cv2()

    arr = _ensure_hwc_uint8(src)
    target_h, target_w = shape_hw
    if arr.shape[:2] == (target_h, target_w):
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _save_image(path: Path, image: Any) -> Path:
    cv2 = _require_cv2()

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _ensure_hwc_uint8(image)
    if arr.shape[2] == 1:
        ok = cv2.imwrite(str(path), arr[..., 0])
    else:
        ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f'Failed to write image to {path}')
    return path


def _tile_grid(top_left: Any, top_right: Any, bottom_left: Any, bottom_right: Any):
    cv2 = _require_cv2()

    tl = _ensure_hwc_uint8(top_left)
    tr = _resize_to_match(top_right, tl.shape[:2])
    bl = _resize_to_match(bottom_left, tl.shape[:2])
    br = _resize_to_match(bottom_right, tl.shape[:2])
    if tl.shape[2] == 1:
        tl = cv2.cvtColor(tl[..., 0], cv2.COLOR_GRAY2BGR)
    if tr.shape[2] == 1:
        tr = cv2.cvtColor(tr[..., 0], cv2.COLOR_GRAY2BGR)
    if bl.shape[2] == 1:
        bl = cv2.cvtColor(bl[..., 0], cv2.COLOR_GRAY2BGR)
    if br.shape[2] == 1:
        br = cv2.cvtColor(br[..., 0], cv2.COLOR_GRAY2BGR)
    return cv2.vconcat([cv2.hconcat([tl, tr]), cv2.hconcat([bl, br])])


def compute_image_similarity(live_image: Any, dataset_image: Any) -> dict[str, float]:
    np = _require_numpy()

    live_gray = _squeeze_to_gray(live_image).astype(np.float32) / 255.0
    dataset_gray = _squeeze_to_gray(dataset_image).astype(np.float32) / 255.0
    if live_gray.shape != dataset_gray.shape:
        dataset_gray = _resize_to_match(dataset_gray[..., None], live_gray.shape)[:, :, 0].astype(np.float32) / 255.0
    diff = live_gray - dataset_gray
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    live_flat = live_gray.reshape(-1)
    ds_flat = dataset_gray.reshape(-1)
    denom = float(np.linalg.norm(live_flat) * np.linalg.norm(ds_flat))
    cosine = float(np.dot(live_flat, ds_flat) / denom) if denom > 0.0 else 0.0
    return {
        'mae': mae,
        'rmse': rmse,
        'cosine': cosine,
    }


def compute_pair_similarity(live_left: Any, live_right: Any, ds_left: Any, ds_right: Any) -> dict[str, Any]:
    left_metrics = compute_image_similarity(live_left, ds_left)
    right_metrics = compute_image_similarity(live_right, ds_right)
    return {
        'left': left_metrics,
        'right': right_metrics,
        'pair_mae': float((left_metrics['mae'] + right_metrics['mae']) / 2.0),
        'pair_rmse': float((left_metrics['rmse'] + right_metrics['rmse']) / 2.0),
        'pair_cosine': float((left_metrics['cosine'] + right_metrics['cosine']) / 2.0),
    }


def build_robot_config(*, args: argparse.Namespace, camera_configs: dict[str, Any], tactile_enabled: bool, policy_fps: float):
    from lerobot.robots.franka_research3 import FrankaResearch3Config
    from tools.fr3 import fr3_act_infer_real_runtime as infer_runtime

    return FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(infer_runtime._DAS_URDF),
        target_frame_name='das_gripper_ee',
        workspace_min=(0.1, -0.6, 0.05),
        workspace_max=(0.9, 0.6, 0.8),
        das_tactile_frequency_hz=policy_fps if tactile_enabled else None,
        das_tactile_valid_mask_path=str(infer_runtime._DEFAULT_TACTILE_VALID_MASK_PATH) if tactile_enabled else None,
        das_tactile_baseline_path=str(infer_runtime._DEFAULT_TACTILE_BASELINE_PATH) if tactile_enabled else None,
        das_tactile_timeout_s=2.0,
        cameras={name: cfg for name, cfg in camera_configs.items()},
    )


def capture_observation_sequence(
    args: argparse.Namespace,
    *,
    camera_configs: dict[str, Any],
    policy_fps: float,
    tactile_enabled: bool,
) -> tuple[list[dict[str, Any]], Any, Any]:
    np = _require_numpy()
    from lerobot.robots.franka_research3 import FrankaResearch3
    from lerobot.robots.franka_research3.processor_franka_research3 import KeepAbsoluteEEObservation
    from tools.fr3 import fr3_act_infer_real_runtime as infer_runtime

    robot_cfg = build_robot_config(args=args, camera_configs=camera_configs, tactile_enabled=tactile_enabled, policy_fps=policy_fps)
    robot = FrankaResearch3(robot_cfg)
    state_processor = KeepAbsoluteEEObservation()
    records: list[dict[str, Any]] = []
    episode_start_pose = None
    previous_local_quaternion_xyzw = None

    robot.connect()
    state_processor.reset()
    try:
        next_deadline = time.perf_counter()
        for step_idx in range(max(int(args.preview_steps), 1)):
            robot_observation = robot.get_observation()
            absolute_state_observation = state_processor.observation(dict(robot_observation))
            if episode_start_pose is None:
                episode_start_pose = infer_runtime._pose_from_quaternion_observation(absolute_state_observation)
            localized_state_observation, previous_local_quaternion_xyzw = infer_runtime.localize_observation_to_start_frame(
                absolute_state_observation,
                episode_start_pose,
                previous_quaternion_xyzw=previous_local_quaternion_xyzw,
            )
            records.append(
                {
                    'robot_observation': dict(robot_observation),
                    'absolute_state_observation': dict(absolute_state_observation),
                    'localized_state_observation': dict(localized_state_observation),
                }
            )
            if step_idx + 1 < max(int(args.preview_steps), 1):
                next_deadline += 1.0 / float(policy_fps)
                infer_runtime.precise_sleep(max(0.0, next_deadline - time.perf_counter()))
    finally:
        robot.disconnect()

    first_absolute = records[0]['absolute_state_observation']
    current_state = np.asarray(
        [
            first_absolute['ee.x'],
            first_absolute['ee.y'],
            first_absolute['ee.z'],
            first_absolute['ee.qx'],
            first_absolute['ee.qy'],
            first_absolute['ee.qz'],
            first_absolute['ee.qw'],
            first_absolute['gripper.pos'],
        ],
        dtype=np.float64,
    )
    return records, current_state, episode_start_pose


def load_episode_image_references(
    dataset_root: Path,
    *,
    repo_id: str,
    episode_indices: list[int],
) -> list[EpisodeImageReference]:
    np = _require_numpy()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    refs: list[EpisodeImageReference] = []
    for episode_index in episode_indices:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[int(episode_index)], download_videos=False)
        if len(dataset) == 0:
            raise ValueError(f'LeRobotDataset returned no frames for episode {episode_index}')
        item = dataset[0]
        refs.append(
            EpisodeImageReference(
                episode_index=int(episode_index),
                state=np.asarray(item['observation.state'], dtype=np.float64),
                left_image=_ensure_hwc_uint8(item['observation.images.left']),
                right_image=_ensure_hwc_uint8(item['observation.images.right']),
            )
        )
    return refs


def swap_left_right(observation: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(observation)
    if 'left' in observation and 'right' in observation:
        swapped['left'] = observation['right']
        swapped['right'] = observation['left']
    return swapped


def preview_mapping_summary(
    mapping_name: str,
    *,
    records: list[dict[str, Any]],
    episode_start_pose: Any,
    pretrained_dir: Path,
    ds_meta: Any,
    device: Any,
    state_names: list[str],
    action_names: list[str],
    robot_cfg: Any,
    first_frame_max_pos_delta_mm: float,
    first_frame_max_rot_delta_deg: float,
    max_step_pos_delta_mm: float,
    max_step_rot_delta_deg: float,
) -> dict[str, Any]:
    np = _require_numpy()
    import torch
    from tools.fr3 import fr3_act_infer_real_runtime as infer_runtime

    policy, preprocessor, postprocessor = infer_runtime.load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    policy.reset()

    first_frame_max_pos_delta_m = float(first_frame_max_pos_delta_mm) / 1000.0
    first_frame_max_rot_delta_rad = np.deg2rad(float(first_frame_max_rot_delta_deg))
    max_step_pos_delta_m = float(max_step_pos_delta_mm) / 1000.0
    max_step_rot_delta_rad = np.deg2rad(float(max_step_rot_delta_deg))

    status_counts = {'pass': 0, 'clamped': 0, 'hold_first_frame': 0}
    step_summaries: list[dict[str, Any]] = []

    for step_idx, record in enumerate(records):
        localized_state_observation = dict(record['localized_state_observation'])
        if mapping_name == 'swapped':
            localized_state_observation = swap_left_right(localized_state_observation)

        policy_observation = infer_runtime.build_policy_observation(
            localized_state_observation,
            state_names=state_names,
            input_features=policy.config.input_features,
        )
        action_tensor = infer_runtime.predict_action(
            policy_observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(policy.config.use_amp),
            robot_type='franka_research3',
        )
        local_robot_command = infer_runtime.decode_action_to_robot_command(
            action_tensor,
            action_names=action_names,
            robot_cfg=robot_cfg,
        )
        robot_command = infer_runtime.convert_local_command_to_base_frame(local_robot_command, episode_start_pose)
        safe_command, position_delta, rotation_delta, clamped = infer_runtime.clamp_command_relative_to_current(
            robot_command,
            record['robot_observation'],
            max_pos_delta_m=max_step_pos_delta_m,
            max_rot_delta_rad=max_step_rot_delta_rad,
        )
        status = 'pass'
        command_to_send = safe_command
        first_frame_reject = False
        first_position_delta = None
        first_rotation_delta = None
        if step_idx == 0:
            first_frame_reject, first_position_delta, first_rotation_delta = infer_runtime.should_reject_first_command(
                robot_command,
                record['robot_observation'],
                max_pos_delta_m=first_frame_max_pos_delta_m,
                max_rot_delta_rad=first_frame_max_rot_delta_rad,
            )
            if first_frame_reject:
                status = 'hold_first_frame'
                command_to_send = infer_runtime.build_hold_command(record['robot_observation'])
        if status == 'pass' and clamped:
            status = 'clamped'
        status_counts[status] = status_counts.get(status, 0) + 1
        step_summaries.append(
            {
                'step': int(step_idx),
                'status': status,
                'raw_ee_xyz': [float(robot_command['ee.x']), float(robot_command['ee.y']), float(robot_command['ee.z'])],
                'safe_ee_xyz': [float(command_to_send['ee.x']), float(command_to_send['ee.y']), float(command_to_send['ee.z'])],
                'gripper': float(command_to_send['gripper.pos']),
                'position_delta_mm': [float(value * 1000.0) for value in position_delta],
                'rotation_delta_deg': [float(np.degrees(value)) for value in rotation_delta],
                'first_frame_reject': bool(first_frame_reject),
                'first_frame_position_delta_mm': None if first_position_delta is None else [float(value * 1000.0) for value in first_position_delta],
                'first_frame_rotation_delta_deg': None if first_rotation_delta is None else [float(np.degrees(value)) for value in first_rotation_delta],
            }
        )

    def _mean_abs_component(values_key: str) -> float:
        values = []
        for step in step_summaries:
            values.extend(abs(float(v)) for v in step[values_key])
        return float(sum(values) / len(values)) if values else 0.0

    return {
        'mapping': mapping_name,
        'steps': step_summaries,
        'summary': {
            'status_counts': status_counts,
            'hold_first_frame': int(status_counts.get('hold_first_frame', 0)),
            'clamped': int(status_counts.get('clamped', 0)),
            'pass': int(status_counts.get('pass', 0)),
            'mean_abs_position_delta_mm': _mean_abs_component('position_delta_mm'),
            'mean_abs_rotation_delta_deg': _mean_abs_component('rotation_delta_deg'),
        },
    }


def write_similarity_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'episode_index',
                'normal_pair_mae',
                'normal_pair_rmse',
                'normal_pair_cosine',
                'swapped_pair_mae',
                'swapped_pair_rmse',
                'swapped_pair_cosine',
                'preferred_mapping_by_mae',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main(argv: list[str] | None = None) -> int:
    np = _require_numpy()
    import torch
    from tools.fr3 import fr3_act_infer_real_runtime as infer_runtime
    from tools.fr3 import fr3_compare_current_pose_to_dataset_starts as pose_compare

    args = parse_args(argv)
    output_dir = _resolve_repo_path(args.output_dir) if args.output_dir is not None else _default_output_dir()
    live_dir = output_dir / 'live'
    refs_dir = output_dir / 'dataset_refs'
    compare_dir = output_dir / 'comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_dir = infer_runtime.resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = infer_runtime.load_train_config(pretrained_dir)
    dataset_root = infer_runtime.resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    ds_meta = infer_runtime.load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    camera_configs = infer_runtime.load_camera_configs(args.camera_config)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    policy, _, _ = infer_runtime.load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    required_image_keys = infer_runtime.extract_required_image_keys(policy.config.input_features)
    required_tactile_keys = infer_runtime.extract_required_tactile_keys(policy.config.input_features)
    infer_runtime.validate_camera_keys(required_image_keys=required_image_keys, available_camera_keys=list(camera_configs))
    policy_fps = float(args.policy_fps or ds_meta.fps)
    state_names = infer_runtime.extract_feature_names(ds_meta.features['observation.state'], infer_runtime._DEFAULT_STATE_NAMES)
    action_names = infer_runtime.extract_feature_names(ds_meta.features['action'], infer_runtime._DEFAULT_ACTION_NAMES)
    tactile_enabled = bool(required_tactile_keys)

    records, current_state, episode_start_pose = capture_observation_sequence(
        args,
        camera_configs=camera_configs,
        policy_fps=policy_fps,
        tactile_enabled=False,
    )

    first_live = records[0]['robot_observation']
    live_left = _ensure_hwc_uint8(first_live['left'])
    live_right = _ensure_hwc_uint8(first_live['right'])
    live_left_path = _save_image(live_dir / 'live_left.png', live_left)
    live_right_path = _save_image(live_dir / 'live_right.png', live_right)

    start_states = pose_compare.load_episode_start_states(dataset_root, limit_episodes=args.limit_episodes)
    deltas = pose_compare.compute_episode_deltas(
        current_state,
        start_states,
        position_weight_mm=1.0,
        rotation_weight_deg=1.0,
        gripper_weight=50.0,
    )
    top_matches = deltas[: max(int(args.top_k), 1)]
    top_episode_indices = [match.episode_index for match in top_matches]
    refs = load_episode_image_references(dataset_root, repo_id=train_cfg.dataset.repo_id, episode_indices=top_episode_indices)

    similarity_rows_for_json: list[dict[str, Any]] = []
    similarity_rows_for_csv: list[dict[str, Any]] = []

    for ref, delta in zip(refs, top_matches, strict=True):
        normal = compute_pair_similarity(live_left, live_right, ref.left_image, ref.right_image)
        swapped = compute_pair_similarity(live_left, live_right, ref.right_image, ref.left_image)
        preferred_mapping = 'normal' if normal['pair_mae'] <= swapped['pair_mae'] else 'swapped'

        ref_left_path = _save_image(refs_dir / f'episode_{ref.episode_index:03d}_left.png', ref.left_image)
        ref_right_path = _save_image(refs_dir / f'episode_{ref.episode_index:03d}_right.png', ref.right_image)
        compare_normal_path = _save_image(
            compare_dir / f'episode_{ref.episode_index:03d}_normal.png',
            _tile_grid(live_left, ref.left_image, live_right, ref.right_image),
        )
        compare_swapped_path = _save_image(
            compare_dir / f'episode_{ref.episode_index:03d}_swapped.png',
            _tile_grid(live_left, ref.right_image, live_right, ref.left_image),
        )

        similarity_rows_for_json.append(
            {
                'episode_index': int(ref.episode_index),
                'position_delta_mm': [float(value * 1000.0) for value in delta.position_delta_m],
                'position_distance_mm': float(delta.position_distance_m * 1000.0),
                'rotation_delta_deg': float(delta.rotation_delta_deg),
                'gripper_delta': float(delta.gripper_delta),
                'normal': normal,
                'swapped': swapped,
                'preferred_mapping_by_mae': preferred_mapping,
                'saved_images': {
                    'dataset_left': str(ref_left_path),
                    'dataset_right': str(ref_right_path),
                    'compare_normal': str(compare_normal_path),
                    'compare_swapped': str(compare_swapped_path),
                },
            }
        )
        similarity_rows_for_csv.append(
            {
                'episode_index': int(ref.episode_index),
                'normal_pair_mae': f"{normal['pair_mae']:.6f}",
                'normal_pair_rmse': f"{normal['pair_rmse']:.6f}",
                'normal_pair_cosine': f"{normal['pair_cosine']:.6f}",
                'swapped_pair_mae': f"{swapped['pair_mae']:.6f}",
                'swapped_pair_rmse': f"{swapped['pair_rmse']:.6f}",
                'swapped_pair_cosine': f"{swapped['pair_cosine']:.6f}",
                'preferred_mapping_by_mae': preferred_mapping,
            }
        )

    similarity_csv_path = write_similarity_csv(output_dir / 'similarity_table.csv', similarity_rows_for_csv)

    preview_results: dict[str, Any] = {}
    preview_records = records
    preview_episode_start_pose = episode_start_pose
    preview_capture_error = None
    if tactile_enabled:
        try:
            preview_records, _, preview_episode_start_pose = capture_observation_sequence(
                args,
                camera_configs=camera_configs,
                policy_fps=policy_fps,
                tactile_enabled=True,
            )
        except Exception as exc:
            preview_capture_error = f'{type(exc).__name__}: {exc}'

    robot_cfg = build_robot_config(args=args, camera_configs=camera_configs, tactile_enabled=tactile_enabled, policy_fps=policy_fps)
    for mapping_name in ('normal', 'swapped'):
        if preview_capture_error is not None:
            preview_results[mapping_name] = {
                'mapping': mapping_name,
                'error': preview_capture_error,
            }
            continue
        try:
            preview_results[mapping_name] = preview_mapping_summary(
                mapping_name,
                records=preview_records,
                episode_start_pose=preview_episode_start_pose,
                pretrained_dir=pretrained_dir,
                ds_meta=ds_meta,
                device=device,
                state_names=state_names,
                action_names=action_names,
                robot_cfg=robot_cfg,
                first_frame_max_pos_delta_mm=args.first_frame_max_pos_delta_mm,
                first_frame_max_rot_delta_deg=args.first_frame_max_rot_delta_deg,
                max_step_pos_delta_mm=args.max_step_pos_delta_mm,
                max_step_rot_delta_deg=args.max_step_rot_delta_deg,
            )
        except Exception as exc:
            preview_results[mapping_name] = {
                'mapping': mapping_name,
                'error': f'{type(exc).__name__}: {exc}',
            }

    mean_normal_mae = float(np.mean([row['normal']['pair_mae'] for row in similarity_rows_for_json])) if similarity_rows_for_json else 0.0
    mean_swapped_mae = float(np.mean([row['swapped']['pair_mae'] for row in similarity_rows_for_json])) if similarity_rows_for_json else 0.0

    summary = {
        'checkpoint': str(pretrained_dir),
        'dataset_root': str(dataset_root),
        'camera_config': str(_resolve_repo_path(args.camera_config)),
        'policy_image_keys': required_image_keys,
        'policy_tactile_keys': required_tactile_keys,
        'policy_fps': float(policy_fps),
        'current_state': {
            'x': float(current_state[0]),
            'y': float(current_state[1]),
            'z': float(current_state[2]),
            'qx': float(current_state[3]),
            'qy': float(current_state[4]),
            'qz': float(current_state[5]),
            'qw': float(current_state[6]),
            'gripper': float(current_state[7]),
        },
        'live_images': {
            'left': str(live_left_path),
            'right': str(live_right_path),
        },
        'top_pose_matches': [
            {
                'episode_index': int(match.episode_index),
                'position_delta_mm': [float(value * 1000.0) for value in match.position_delta_m],
                'position_distance_mm': float(match.position_distance_m * 1000.0),
                'rotation_delta_deg': float(match.rotation_delta_deg),
                'gripper_delta': float(match.gripper_delta),
                'weighted_score': float(match.weighted_score),
            }
            for match in top_matches
        ],
        'similarity': {
            'csv_path': str(similarity_csv_path),
            'mean_pair_mae': {
                'normal': mean_normal_mae,
                'swapped': mean_swapped_mae,
            },
            'preferred_mapping_by_mean_mae': 'normal' if mean_normal_mae <= mean_swapped_mae else 'swapped',
            'episodes': similarity_rows_for_json,
        },
        'preview_capture_error': preview_capture_error,
        'preview': preview_results,
    }
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[INFO] output_dir={output_dir}')
    print(f'[INFO] live_left={live_left_path}')
    print(f'[INFO] live_right={live_right_path}')
    print(f'[INFO] summary_json={summary_path}')
    print(f'[INFO] similarity_csv={similarity_csv_path}')
    print('[INFO] similarity_table')
    for row in similarity_rows_for_csv:
        print(
            '  '
            + f"ep={row['episode_index']} normal_mae={row['normal_pair_mae']} swapped_mae={row['swapped_pair_mae']} "
            + f"preferred={row['preferred_mapping_by_mae']}"
        )
    print('[INFO] preview_summary')
    for mapping_name in ('normal', 'swapped'):
        mapping = preview_results[mapping_name]
        if 'error' in mapping:
            print(f"  {mapping_name}: error={mapping['error']}")
            continue
        summary_metrics = mapping['summary']
        print(
            '  '
            + f"{mapping_name}: hold_first_frame={summary_metrics['hold_first_frame']} clamped={summary_metrics['clamped']} "
            + f"pass={summary_metrics['pass']} mean_abs_pos_mm={summary_metrics['mean_abs_position_delta_mm']:.2f} "
            + f"mean_abs_rot_deg={summary_metrics['mean_abs_rotation_delta_deg']:.2f}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
