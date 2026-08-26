#!/usr/bin/env python3
"""Compare a dumped live step0 policy input bundle against nearest dataset start/frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import predict_action
from lerobot.utils.rotation import Rotation

import fr3_act_infer_real_runtime as infer_runtime


def _load_metadata(capture_dir: Path) -> dict:
    metadata_path = capture_dir / 'metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Capture metadata not found: {metadata_path}')
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def _state_distance_scores(states: np.ndarray, query_state: np.ndarray) -> np.ndarray:
    std = np.std(states, axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return np.linalg.norm((states - query_state[None, :]) / std[None, :], axis=1)


def _state_error_summary(query_state: np.ndarray, candidate_state: np.ndarray) -> dict[str, float | list[float]]:
    pos_delta_m = np.asarray(query_state[:3] - candidate_state[:3], dtype=np.float64)
    rot_delta_deg = infer_runtime._quaternion_angle_deg(query_state[3:7], candidate_state[3:7])
    gripper_delta_mm = float(abs(query_state[7] - candidate_state[7]) * 1000.0) if len(query_state) > 7 else 0.0
    return {
        'pos_delta_mm_xyz': (pos_delta_m * 1000.0).tolist(),
        'pos_err_mm_norm': float(np.linalg.norm(pos_delta_m) * 1000.0),
        'rot_err_deg': float(rot_delta_deg),
        'gripper_delta_mm': gripper_delta_mm,
    }


def _image_tensor_to_hwc_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.moveaxis(image, 0, -1)
    return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)


def _image_error_metrics(live_image: np.ndarray, dataset_image: np.ndarray) -> tuple[float, float]:
    diff = live_image.astype(np.float32) - dataset_image.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def _load_dataset_frame(dataset_root: Path, repo_id: str, episode_idx: int, frame_idx: int) -> dict:
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode_idx], image_transforms=None)
    if frame_idx < 0 or frame_idx >= len(dataset):
        raise IndexError(f'Frame {frame_idx} out of range for episode {episode_idx} with length {len(dataset)}')
    return dataset[frame_idx]


def _load_all_frame_state_rows(
    dataset_root: Path,
    episode_indices: list[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    import pyarrow.parquet as pq

    dataset_root = infer_runtime._resolve_repo_path(dataset_root)
    meta_dir = dataset_root / 'meta' / 'episodes'
    meta_files = sorted(meta_dir.rglob('*.parquet'))
    if not meta_files:
        raise FileNotFoundError(f'No episode metadata parquet files found in {meta_dir}')

    episode_set = {int(episode_idx) for episode_idx in episode_indices}
    data_files: list[Path] = []
    seen_data_files: set[Path] = set()
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file)).to_pydict()
        for episode_idx, chunk_idx, file_idx in zip(
            table['episode_index'],
            table['data/chunk_index'],
            table['data/file_index'],
            strict=True,
        ):
            episode_idx = int(episode_idx)
            if episode_idx not in episode_set:
                continue
            # Resolved through the runtime rather than formatted here: a v3 view names its
            # parquet `file-000`, older exports `file-000000`, and the info.json `data_path`
            # template is what settles it. Spelling the six-digit form out here made this scan
            # fail on every view built by the Training View page.
            data_file = infer_runtime._resolve_dataset_data_file(
                dataset_root, chunk_index=int(chunk_idx), file_index=int(file_idx)
            )
            if data_file not in seen_data_files:
                seen_data_files.add(data_file)
                data_files.append(data_file)

    states_by_episode: dict[int, list[np.ndarray]] = {int(episode_idx): [] for episode_idx in episode_indices}
    for data_file in sorted(data_files):
        table = pq.read_table(str(data_file), columns=['episode_index', 'observation.state']).to_pydict()
        for row_episode_idx, state in zip(table['episode_index'], table['observation.state'], strict=True):
            row_episode_idx = int(row_episode_idx)
            if row_episode_idx not in states_by_episode:
                continue
            states_by_episode[row_episode_idx].append(np.asarray(state, dtype=np.float64))

    frame_states: list[np.ndarray] = []
    frame_refs: list[tuple[int, int]] = []
    for episode_idx in episode_indices:
        episode_states = states_by_episode.get(int(episode_idx), [])
        if not episode_states:
            raise ValueError(f'No frame states resolved for episode {episode_idx} from {dataset_root}')
        for frame_idx, state in enumerate(episode_states):
            frame_states.append(state)
            frame_refs.append((int(episode_idx), frame_idx))

    if not frame_states:
        raise ValueError(f'No frame states resolved from {dataset_root}')
    return np.asarray(frame_states, dtype=np.float64), frame_refs



def _load_capture_policy_observation(metadata: dict, capture_dir: Path) -> dict[str, np.ndarray]:
    return {
        feature_key: np.asarray(np.load(capture_dir / array_filename))
        for feature_key, array_filename in metadata['policy_observation_files'].items()
    }


def _build_quaternion_observation_from_rotvec_scalars(raw_scalars: dict) -> dict[str, float]:
    raw_pose = infer_runtime._pose_from_position_and_rotvec(
        np.asarray([raw_scalars['ee.x'], raw_scalars['ee.y'], raw_scalars['ee.z']], dtype=np.float64),
        np.asarray([raw_scalars['ee.wx'], raw_scalars['ee.wy'], raw_scalars['ee.wz']], dtype=np.float64),
    )
    quaternion_xyzw = Rotation.from_matrix(raw_pose[:3, :3]).as_quat()
    return {
        'ee.x': float(raw_pose[0, 3]),
        'ee.y': float(raw_pose[1, 3]),
        'ee.z': float(raw_pose[2, 3]),
        'ee.qx': float(quaternion_xyzw[0]),
        'ee.qy': float(quaternion_xyzw[1]),
        'ee.qz': float(quaternion_xyzw[2]),
        'ee.qw': float(quaternion_xyzw[3]),
        'gripper.pos': float(raw_scalars.get('gripper.pos', 0.0)),
    }


def _assemble_policy_observation_from_capture(
    capture_policy_observation: dict[str, np.ndarray],
    dataset_state_observation_i: dict[str, float],
    *,
    state_names: list[str],
    input_features: dict,
) -> dict[str, np.ndarray]:
    state_observation = dict(dataset_state_observation_i)
    for feature_key, value in capture_policy_observation.items():
        if feature_key.startswith('observation.images.'):
            state_observation[feature_key[len('observation.images.'):]] = np.asarray(value)
        elif feature_key.startswith('observation.tactile.'):
            state_observation[feature_key] = np.asarray(value)
    return infer_runtime.build_policy_observation(
        state_observation,
        state_names=state_names,
        input_features=input_features,
        tactile_fallback_observation=None,
    )


def _decode_action_tensor_to_dataset_command_i(action_tensor: torch.Tensor, action_names: list[str]) -> dict[str, float]:
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    action_map = {name: float(action_np[i]) for i, name in enumerate(action_names)}
    quaternion_xyzw = np.asarray(
        [
            infer_runtime._action_value(action_map, 'qx', 'ee.qx'),
            infer_runtime._action_value(action_map, 'qy', 'ee.qy'),
            infer_runtime._action_value(action_map, 'qz', 'ee.qz'),
            infer_runtime._action_value(action_map, 'qw', 'ee.qw'),
        ],
        dtype=np.float64,
    )
    rotvec_xyz = Rotation.from_quat(quaternion_xyzw).as_rotvec()
    return {
        'ee.x': infer_runtime._action_value(action_map, 'x', 'ee.x'),
        'ee.y': infer_runtime._action_value(action_map, 'y', 'ee.y'),
        'ee.z': infer_runtime._action_value(action_map, 'z', 'ee.z'),
        'ee.wx': float(rotvec_xyz[0]),
        'ee.wy': float(rotvec_xyz[1]),
        'ee.wz': float(rotvec_xyz[2]),
    }


def _pose_from_command(command: dict[str, float]) -> np.ndarray:
    return infer_runtime._pose_from_position_and_rotvec(
        np.asarray([command['ee.x'], command['ee.y'], command['ee.z']], dtype=np.float64),
        np.asarray([command['ee.wx'], command['ee.wy'], command['ee.wz']], dtype=np.float64),
    )


def _pose_delta_summary(target_pose: np.ndarray, current_pose: np.ndarray) -> dict[str, list[float] | float]:
    position_delta = np.asarray(target_pose[:3, 3] - current_pose[:3, 3], dtype=np.float64)
    rotation_delta = (Rotation.from_matrix(current_pose[:3, :3]).inv() * Rotation.from_matrix(target_pose[:3, :3])).as_rotvec()
    return {
        'pos_delta_mm_xyz': (position_delta * 1000.0).tolist(),
        'pos_delta_mm_norm': float(np.linalg.norm(position_delta) * 1000.0),
        'rot_delta_deg_xyz': np.rad2deg(rotation_delta).tolist(),
        'rot_delta_deg_norm': float(np.degrees(np.linalg.norm(rotation_delta))),
    }


def _evaluate_live_frame_hypothesis(
    *,
    hypothesis_name: str,
    raw_quaternion_observation: dict[str, float],
    capture_policy_observation: dict[str, np.ndarray],
    dataset_start_pose_contract: np.ndarray,
    state_names: list[str],
    action_names: list[str],
    input_features: dict,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
) -> tuple[dict[str, float], dict[str, list[float] | float]]:
    raw_pose = infer_runtime._pose_from_quaternion_observation(raw_quaternion_observation)
    if hypothesis_name == 'E':
        absolute_state_observation_i = infer_runtime.convert_absolute_observation_from_E_to_I(raw_quaternion_observation)
        current_pose_raw = raw_pose
    elif hypothesis_name == 'I':
        absolute_state_observation_i = dict(raw_quaternion_observation)
        current_pose_raw = raw_pose
    else:
        raise ValueError(f'Unsupported hypothesis {hypothesis_name!r}')

    current_start_pose_i = infer_runtime._pose_from_quaternion_observation(absolute_state_observation_i)
    T_B_Ws = current_start_pose_i @ infer_runtime._invert_pose(dataset_start_pose_contract)
    dataset_state_observation_i, _ = infer_runtime.convert_base_observation_from_I_to_dataset_frame(
        absolute_state_observation_i,
        T_B_Ws,
        previous_quaternion_xyzw=None,
    )
    policy_observation = _assemble_policy_observation_from_capture(
        capture_policy_observation,
        dataset_state_observation_i,
        state_names=state_names,
        input_features=input_features,
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    action_tensor = predict_action(
        policy_observation,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    dataset_command_i = _decode_action_tensor_to_dataset_command_i(action_tensor, action_names)
    base_command_i = infer_runtime.convert_dataset_command_to_base_frame(dataset_command_i, T_B_Ws)
    if hypothesis_name == 'E':
        target_pose_raw = _pose_from_command(infer_runtime.convert_base_command_from_I_to_E(base_command_i))
    else:
        target_pose_raw = _pose_from_command(base_command_i)
    delta_summary = _pose_delta_summary(target_pose_raw, current_pose_raw)
    return dataset_state_observation_i, delta_summary


def _resolve_dataset_root_and_repo_id(metadata: dict, checkpoint: Path, dataset_root_override: str | None) -> tuple[Path, str]:
    if dataset_root_override is not None:
        dataset_root = infer_runtime._resolve_repo_path(dataset_root_override)
    elif metadata.get('dataset_root'):
        dataset_root = infer_runtime._resolve_repo_path(metadata['dataset_root'])
    else:
        pretrained_dir = infer_runtime.resolve_pretrained_model_dir(checkpoint)
        train_cfg = infer_runtime.load_train_config(pretrained_dir)
        dataset_root = infer_runtime.resolve_dataset_root(pretrained_dir, train_cfg, None)
    if metadata.get('dataset_repo_id'):
        repo_id = str(metadata['dataset_repo_id'])
    else:
        pretrained_dir = infer_runtime.resolve_pretrained_model_dir(checkpoint)
        train_cfg = infer_runtime.load_train_config(pretrained_dir)
        repo_id = str(train_cfg.dataset.repo_id)
    return dataset_root, repo_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare a dumped live step0 policy input bundle against nearest dataset start/frame.')
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', default=None)
    args = parser.parse_args(argv)

    capture_dir = infer_runtime._resolve_repo_path(args.capture_dir)
    metadata = _load_metadata(capture_dir)
    dataset_root, repo_id = _resolve_dataset_root_and_repo_id(metadata, args.checkpoint, args.dataset_root)
    capture_policy_observation = _load_capture_policy_observation(metadata, capture_dir)
    policy_state_path = capture_dir / metadata['policy_observation_files']['observation.state']
    live_state = np.asarray(np.load(policy_state_path), dtype=np.float64)

    pretrained_dir = infer_runtime.resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = infer_runtime.load_train_config(pretrained_dir)
    ds_meta = infer_runtime.load_dataset_metadata(dataset_root, repo_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, preprocessor, postprocessor = infer_runtime.load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    dataset_start_pose_contract_xyzquat, _ = infer_runtime.estimate_dataset_start_pose_contract(dataset_root)
    dataset_start_pose_contract = infer_runtime._pose_from_position_and_quaternion(
        dataset_start_pose_contract_xyzquat[:3],
        dataset_start_pose_contract_xyzquat[3:7],
    )
    raw_quaternion_observation = _build_quaternion_observation_from_rotvec_scalars(metadata['robot_observation_scalars'])

    start_rows = infer_runtime._load_episode_start_state_rows(dataset_root)
    start_states = np.asarray([state for _, state in start_rows], dtype=np.float64)
    start_scores = _state_distance_scores(start_states, live_state)
    start_best_idx = int(np.argmin(start_scores))
    start_best_episode, start_best_state = start_rows[start_best_idx]
    start_summary = _state_error_summary(live_state, start_best_state)

    print(f'[INFO] scanning frame states across {len(start_rows)} episodes with single-pass parquet reads')
    frame_states_arr, frame_refs = _load_all_frame_state_rows(
        dataset_root,
        [episode_idx for episode_idx, _ in start_rows],
    )
    frame_scores = _state_distance_scores(frame_states_arr, live_state)
    frame_best_idx = int(np.argmin(frame_scores))
    frame_best_episode, frame_best_frame = frame_refs[frame_best_idx]
    frame_best_state = frame_states_arr[frame_best_idx]
    frame_summary = _state_error_summary(live_state, frame_best_state)

    print(f'[INFO] capture_dir={capture_dir}')
    print(f'[INFO] dataset_root={dataset_root}')
    print(
        '[START_MATCH] '
        f'episode={start_best_episode} frame=0 '
        f'zscore={start_scores[start_best_idx]:.6f} '
        f'pos_err_mm={start_summary["pos_err_mm_norm"]:.3f} '
        f'rot_err_deg={start_summary["rot_err_deg"]:.3f} '
        f'gripper_delta_mm={start_summary["gripper_delta_mm"]:.3f} '
        f'pos_delta_mm_xyz={[round(v, 3) for v in start_summary["pos_delta_mm_xyz"]]}'
    )
    print(
        '[FRAME_MATCH] '
        f'episode={frame_best_episode} frame={frame_best_frame} '
        f'zscore={frame_scores[frame_best_idx]:.6f} '
        f'pos_err_mm={frame_summary["pos_err_mm_norm"]:.3f} '
        f'rot_err_deg={frame_summary["rot_err_deg"]:.3f} '
        f'gripper_delta_mm={frame_summary["gripper_delta_mm"]:.3f} '
        f'pos_delta_mm_xyz={[round(v, 3) for v in frame_summary["pos_delta_mm_xyz"]]}'
    )

    state_names = [str(name) for name in metadata.get('state_names', infer_runtime._DEFAULT_STATE_NAMES)]
    action_names = [str(name) for name in metadata.get('action_names', infer_runtime._DEFAULT_ACTION_NAMES)]
    capture_task = metadata.get('task') if isinstance(metadata.get('task'), str) else None
    # The E-vs-I hypothesis rebuilds the state under each frame convention, and it only knows how
    # to rebuild `ee.*` and `gripper.pos`. A `delta_ee_from_prev_cmd` checkpoint also asks for
    # `prev_cmd.*`, which this path predates. Skipped rather than fatal: the image comparison
    # below is what a crop / camera-placement preflight is here for, and it does not depend on the
    # hypothesis at all.
    for hypothesis_name in ('E', 'I'):
        try:
            hypothesis_state_observation_i, delta_summary = _evaluate_live_frame_hypothesis(
                hypothesis_name=hypothesis_name,
                raw_quaternion_observation=raw_quaternion_observation,
                capture_policy_observation=capture_policy_observation,
                dataset_start_pose_contract=dataset_start_pose_contract,
                state_names=state_names,
                action_names=action_names,
                input_features=policy.config.input_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                use_amp=bool(policy.config.use_amp),
                task=capture_task,
                robot_type=ds_meta.robot_type,
            )
        except KeyError as exc:
            print(
                f'[WARN] hypothesis={hypothesis_name} skipped=unsupported_state_contract '
                f'missing_key={exc.args[0]!r} '
                'reason=_evaluate_live_frame_hypothesis rebuilds only ee.*/gripper.pos'
            )
            continue
        print(
            '[HYPOTHESIS] '
            f'live_ee_frame={hypothesis_name} '
            f'policy_state_xyz=({hypothesis_state_observation_i["ee.x"]:+.6f}, {hypothesis_state_observation_i["ee.y"]:+.6f}, {hypothesis_state_observation_i["ee.z"]:+.6f}) '
            f'first_target_pos_delta_mm_norm={delta_summary["pos_delta_mm_norm"]:.3f} '
            f'first_target_rot_delta_deg_norm={delta_summary["rot_delta_deg_norm"]:.3f} '
            f'first_target_pos_delta_mm_xyz={[round(v, 3) for v in delta_summary["pos_delta_mm_xyz"]]} '
            f'first_target_rot_delta_deg_xyz={[round(v, 3) for v in delta_summary["rot_delta_deg_xyz"]]}'
        )

    for label, episode_idx, frame_idx in (
        ('START_IMAGE', start_best_episode, 0),
        ('FRAME_IMAGE', frame_best_episode, frame_best_frame),
    ):
        dataset_item = _load_dataset_frame(dataset_root, repo_id, episode_idx, frame_idx)
        for image_key, array_filename in sorted(metadata['policy_observation_files'].items()):
            if not image_key.startswith('observation.images.'):
                continue
            live_image = np.asarray(np.load(capture_dir / array_filename), dtype=np.uint8)
            dataset_image = _image_tensor_to_hwc_uint8(dataset_item[image_key])
            mae, rmse = _image_error_metrics(live_image, dataset_image)
            print(
                f'[{label}] episode={episode_idx} frame={frame_idx:4d} '
                f'camera={image_key} '
                f'mae_255={mae:.3f} '
                f'rmse_255={rmse:.3f}'
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
