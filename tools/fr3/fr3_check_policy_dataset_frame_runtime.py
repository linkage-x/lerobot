#!/usr/bin/env python3
"""Offline checkpoint-vs-dataset frame comparison for FR3 ACT inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import predict_action

import fr3_act_infer_real_runtime as infer_runtime


def parse_indices(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(',') if part.strip()]


def parse_episodes(spec: str, dataset_root: Path) -> list[int]:
    if spec.strip() == 'all-starts':
        return [episode_index for episode_index, _ in infer_runtime._load_episode_start_state_rows(dataset_root)]
    return parse_indices(spec)


def dataset_item_to_policy_observation(item: dict, *, input_feature_keys: list[str]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for feature_key in input_feature_keys:
        if feature_key not in item:
            raise KeyError(f'Missing dataset feature {feature_key!r} in item keys {sorted(item)}')
        value = item[feature_key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Expected torch.Tensor for {feature_key}, got {type(value)}')
        array_value = value.detach().cpu().numpy()
        if feature_key.startswith('observation.images.'):
            array_value = np.moveaxis(array_value, 0, -1)
            array_value = np.clip(np.rint(array_value * 255.0), 0.0, 255.0).astype(np.uint8)
        observation[feature_key] = np.asarray(array_value)
    return observation


def action_error(predicted_action: np.ndarray, dataset_action: np.ndarray) -> tuple[float, float, float]:
    pos_err_mm = float(np.linalg.norm(predicted_action[:3] - dataset_action[:3]) * 1000.0)
    rot_err_deg = infer_runtime._quaternion_angle_deg(predicted_action[3:7], dataset_action[3:7])
    grip_err_mm = float(abs(predicted_action[7] - dataset_action[7]) * 1000.0)
    return pos_err_mm, rot_err_deg, grip_err_mm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Offline checkpoint-vs-dataset frame comparison for FR3 ACT inference.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--episodes', default='0,13')
    parser.add_argument('--frame-indices', default='0')
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    pretrained_dir = infer_runtime.resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = infer_runtime.load_train_config(pretrained_dir)
    dataset_root = infer_runtime.resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    ds_meta = infer_runtime.load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    policy, preprocessor, postprocessor = infer_runtime.load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    input_feature_keys = list(policy.config.input_features)
    episodes = parse_episodes(args.episodes, dataset_root)
    frame_indices = parse_indices(args.frame_indices)

    print(f'[INFO] checkpoint={pretrained_dir}')
    print(f'[INFO] dataset_root={dataset_root}')
    print(f'[INFO] policy_device={device}')
    print(f'[INFO] episodes={episodes}')
    print(f'[INFO] frame_indices={frame_indices}')

    pos_errors_mm: list[float] = []
    rot_errors_deg: list[float] = []
    grip_errors_mm: list[float] = []
    detailed_rows: list[tuple[float, float, float, int, int]] = []

    for episode_idx in episodes:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        dataset = LeRobotDataset(train_cfg.dataset.repo_id, root=dataset_root, episodes=[episode_idx], image_transforms=None)
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= len(dataset):
                raise IndexError(f'Frame {frame_idx} out of range for episode {episode_idx} with length {len(dataset)}')
            item = dataset[frame_idx]
            observation = dataset_item_to_policy_observation(item, input_feature_keys=input_feature_keys)
            predicted_action = predict_action(
                observation,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp),
                task=item.get('task') if isinstance(item.get('task'), str) else None,
                robot_type=dataset.meta.robot_type,
            )
            predicted_action_np = np.asarray(predicted_action.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
            dataset_action_np = np.asarray(item['action'].detach().cpu().numpy(), dtype=np.float64)
            pos_err_mm, rot_err_deg, grip_err_mm = action_error(predicted_action_np, dataset_action_np)
            pos_errors_mm.append(pos_err_mm)
            rot_errors_deg.append(rot_err_deg)
            grip_errors_mm.append(grip_err_mm)
            detailed_rows.append((pos_err_mm, rot_err_deg, grip_err_mm, episode_idx, frame_idx))
            print(
                f'[CHECK] episode={episode_idx:3d} frame={frame_idx:4d} '
                f'pos_err_mm={pos_err_mm:.3f} '
                f'rot_err_deg={rot_err_deg:.3f} '
                f'grip_err_mm={grip_err_mm:.3f}'
            )

    pos_errors_arr = np.asarray(pos_errors_mm, dtype=np.float64)
    rot_errors_arr = np.asarray(rot_errors_deg, dtype=np.float64)
    grip_errors_arr = np.asarray(grip_errors_mm, dtype=np.float64)
    print(
        '[SUMMARY] '
        f'count={len(detailed_rows)} '
        f'pos_mm median/p95/max={np.median(pos_errors_arr):.3f}/{np.percentile(pos_errors_arr, 95):.3f}/{np.max(pos_errors_arr):.3f} '
        f'rot_deg median/p95/max={np.median(rot_errors_arr):.3f}/{np.percentile(rot_errors_arr, 95):.3f}/{np.max(rot_errors_arr):.3f} '
        f'grip_mm median/p95/max={np.median(grip_errors_arr):.3f}/{np.percentile(grip_errors_arr, 95):.3f}/{np.max(grip_errors_arr):.3f}'
    )
    for pos_err_mm, rot_err_deg, grip_err_mm, episode_idx, frame_idx in sorted(detailed_rows, reverse=True)[:5]:
        print(
            '[WORST] '
            f'episode={episode_idx:3d} frame={frame_idx:4d} '
            f'pos_err_mm={pos_err_mm:.3f} '
            f'rot_err_deg={rot_err_deg:.3f} '
            f'grip_err_mm={grip_err_mm:.3f}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
