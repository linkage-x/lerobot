#!/usr/bin/env python3
"""Offline action-quality check for FR3 pi05+LoRA checkpoints.

Run inside the lerobot docker image from /workspace with:

PYTHONPATH=/workspace/src:/workspace python scripts/eval_pi05_lora_fr3_offline.py \
  --checkpoint outputs/train/.../checkpoints/003000 \
  --start-frames-only
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.franka_research3 import FrankaResearch3Config
from scipy.spatial.transform import Rotation

from tools.fr3.fr3_act_infer_real_runtime import (
    _DEFAULT_ACTION_NAMES,
    _DEFAULT_STATE_NAMES,
    decode_action_to_robot_command,
    extract_feature_names,
    load_dataset_metadata,
    load_policy_stack,
    load_train_config,
    predict_action_chunk_for_preflight,
    resolve_dataset_root,
    resolve_pretrained_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-id", default="hph/fr3_pick_place_ee2ee_v1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--start-frames-only", action="store_true")
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices. Overrides sampling.")
    parser.add_argument("--fail-first-pos-mm", type=float, default=30.0)
    parser.add_argument("--fail-first-rot-deg", type=float, default=10.0)
    parser.add_argument("--fail-p95-pred-gt-mm", type=float, default=40.0)
    parser.add_argument("--chunk-actions", type=int, default=50)
    return parser.parse_args()


def quat_error_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = np.asarray(a_xyzw, dtype=np.float64)
    b = np.asarray(b_xyzw, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float((Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude() * 180.0 / math.pi)


def image_tensor_to_hwc_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def choose_indices(ds: LeRobotDataset, args: argparse.Namespace) -> list[int]:
    if args.indices:
        return [int(item) for item in args.indices.split(",") if item.strip()]

    chosen: list[int] = []
    if args.start_frames_only:
        for idx in range(len(ds)):
            item = ds[idx]
            if int(item["frame_index"]) == 0:
                chosen.append(idx)
                if len(chosen) >= args.max_samples:
                    break
        return chosen

    if args.max_samples >= len(ds):
        return list(range(len(ds)))
    return np.linspace(0, len(ds) - 1, args.max_samples, dtype=np.int64).tolist()


def summarize(name: str, values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    return (
        f"{name}: mean={array.mean():.2f} p50={np.percentile(array, 50):.2f} "
        f"p95={np.percentile(array, 95):.2f} max={array.max():.2f}"
    )


def main() -> int:
    args = parse_args()
    checkpoint = resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = load_train_config(checkpoint)
    dataset_root = Path(args.dataset_root) if args.dataset_root else resolve_dataset_root(checkpoint, train_cfg, None)
    ds_meta = load_dataset_metadata(dataset_root, args.repo_id)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = LeRobotDataset(args.repo_id, root=str(dataset_root))
    policy, preprocessor, postprocessor = load_policy_stack(checkpoint, ds_meta=ds_meta, device=device)
    action_names = extract_feature_names(ds_meta.features["action"], _DEFAULT_ACTION_NAMES)
    _ = extract_feature_names(ds_meta.features["observation.state"], _DEFAULT_STATE_NAMES)
    robot_cfg = FrankaResearch3Config(robot_ip="0.0.0.0", gripper_backend="pika")

    indices = choose_indices(ds, args)
    if not indices:
        raise RuntimeError("No samples selected.")

    first_pos_current_mm: list[float] = []
    first_rot_current_deg: list[float] = []
    pred_gt_pos_mm: list[float] = []
    pred_gt_rot_deg: list[float] = []
    max_step_pos_mm: list[float] = []

    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] samples={len(indices)} start_frames_only={args.start_frames_only}")

    for sample_no, idx in enumerate(indices):
        item = ds[idx]
        state = item["observation.state"].detach().cpu().numpy().astype(np.float32)
        gt = item["action"].detach().cpu().numpy().astype(np.float32)
        obs = {"observation.state": state}
        for camera_name in ("ee", "side", "front"):
            key = f"observation.images.{camera_name}"
            if key in item:
                obs[key] = image_tensor_to_hwc_uint8(item[key])

        chunk = predict_action_chunk_for_preflight(
            obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(policy.config.use_amp),
            robot_type="franka_research3",
            task=item.get("task", "Pick and place"),
        )
        chunk_np = chunk[0, : args.chunk_actions].detach().cpu().numpy()
        pred = chunk_np[0]

        pred_cmd = decode_action_to_robot_command(chunk[:, 0, :], action_names=action_names, robot_cfg=robot_cfg)
        gt_cmd = decode_action_to_robot_command(item["action"].reshape(1, -1), action_names=action_names, robot_cfg=robot_cfg)
        _ = pred_cmd, gt_cmd

        current_xyz = state[:3].astype(np.float64)
        current_quat = state[6:10].astype(np.float64)
        pred_xyz = pred[:3].astype(np.float64)
        pred_quat = pred[3:7].astype(np.float64)
        gt_xyz = gt[:3].astype(np.float64)
        gt_quat = gt[3:7].astype(np.float64)

        first_pos_current_mm.append(float(np.linalg.norm(pred_xyz - current_xyz) * 1000.0))
        first_rot_current_deg.append(quat_error_deg(current_quat, pred_quat))
        pred_gt_pos_mm.append(float(np.linalg.norm(pred_xyz - gt_xyz) * 1000.0))
        pred_gt_rot_deg.append(quat_error_deg(gt_quat, pred_quat))

        if len(chunk_np) > 1:
            max_step_pos_mm.append(float(np.max(np.linalg.norm(np.diff(chunk_np[:, :3], axis=0), axis=1)) * 1000.0))
        else:
            max_step_pos_mm.append(0.0)

        print(
            "[SAMPLE] "
            f"n={sample_no:03d} idx={idx} ep={int(item['episode_index'])} frame={int(item['frame_index'])} "
            f"first_current_mm={first_pos_current_mm[-1]:.2f} "
            f"pred_gt_mm={pred_gt_pos_mm[-1]:.2f} "
            f"first_current_rot_deg={first_rot_current_deg[-1]:.2f} "
            f"pred_xyz=({pred_xyz[0]:+.4f},{pred_xyz[1]:+.4f},{pred_xyz[2]:+.4f}) "
            f"gt_xyz=({gt_xyz[0]:+.4f},{gt_xyz[1]:+.4f},{gt_xyz[2]:+.4f})"
        )

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    print("[SUMMARY]", summarize("first_current_pos_mm", first_pos_current_mm))
    print("[SUMMARY]", summarize("first_current_rot_deg", first_rot_current_deg))
    print("[SUMMARY]", summarize("pred_gt_pos_mm", pred_gt_pos_mm))
    print("[SUMMARY]", summarize("pred_gt_rot_deg", pred_gt_rot_deg))
    print("[SUMMARY]", summarize("chunk_max_step_pos_mm", max_step_pos_mm))

    failed = False
    if max(first_pos_current_mm) > args.fail_first_pos_mm:
        print(f"[FAIL] first_current_pos max exceeds {args.fail_first_pos_mm:.1f} mm")
        failed = True
    if max(first_rot_current_deg) > args.fail_first_rot_deg:
        print(f"[FAIL] first_current_rot max exceeds {args.fail_first_rot_deg:.1f} deg")
        failed = True
    if np.percentile(pred_gt_pos_mm, 95) > args.fail_p95_pred_gt_mm:
        print(f"[FAIL] pred_gt_pos p95 exceeds {args.fail_p95_pred_gt_mm:.1f} mm")
        failed = True
    if failed:
        return 2
    print("[PASS] offline action quality gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
