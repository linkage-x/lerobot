#!/usr/bin/env python3
"""Offline infer-vs-replay pose decode consistency check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import fr3_act_infer_real_runtime as infer_runtime
import fr3_das_replay_real_runtime as replay_runtime
from lerobot.utils.rotation import Rotation


def parse_xyzquat(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError('Expected 7 comma-separated floats for x,y,z,qx,qy,qz,qw.')
    try:
        return np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected 7 comma-separated floats for x,y,z,qx,qy,qz,qw.') from exc


def parse_frame_indices(value: str | None, *, max_frames: int) -> list[int]:
    if value is None:
        return list(range(max(max_frames, 0)))
    items = [item.strip() for item in value.split(',') if item.strip()]
    return [int(item) for item in items]


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    position_error_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0)
    rotation_error_deg = float(
        np.degrees(
            np.linalg.norm(
                (Rotation.from_matrix(T_a[:3, :3]).inv() * Rotation.from_matrix(T_b[:3, :3])).as_rotvec()
            )
        )
    )
    return position_error_mm, rotation_error_deg


def infer_decode_pose(T_B_Ws: np.ndarray, frame_xyzquat: np.ndarray) -> np.ndarray:
    quaternion_xyzw = np.asarray(frame_xyzquat[3:7], dtype=np.float64)
    rotvec_xyz = Rotation.from_quat(quaternion_xyzw).as_rotvec()
    dataset_robot_command_i = {
        'ee.x': float(frame_xyzquat[0]),
        'ee.y': float(frame_xyzquat[1]),
        'ee.z': float(frame_xyzquat[2]),
        'ee.wx': float(rotvec_xyz[0]),
        'ee.wy': float(rotvec_xyz[1]),
        'ee.wz': float(rotvec_xyz[2]),
        'gripper.pos': 0.0,
    }
    base_robot_command_i = infer_runtime.convert_dataset_command_to_base_frame(dataset_robot_command_i, T_B_Ws)
    base_robot_command_e = infer_runtime.convert_base_command_from_I_to_E(base_robot_command_i)
    return infer_runtime._pose_from_position_and_rotvec(
        np.asarray([base_robot_command_e['ee.x'], base_robot_command_e['ee.y'], base_robot_command_e['ee.z']], dtype=np.float64),
        np.asarray([base_robot_command_e['ee.wx'], base_robot_command_e['ee.wy'], base_robot_command_e['ee.wz']], dtype=np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Offline infer-vs-replay pose decode consistency check.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--source', choices=['state', 'action'], default='action')
    parser.add_argument('--frame-indices', default=None)
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--start-pose-b-xyzquat', type=parse_xyzquat, default=None)
    args = parser.parse_args(argv)

    episode = replay_runtime.load_episode(str(args.dataset), args.episode)
    frames = np.asarray(episode[args.source], dtype=np.float64)
    frame_indices = parse_frame_indices(args.frame_indices, max_frames=args.max_frames)
    frame_indices = [frame_idx for frame_idx in frame_indices if 0 <= frame_idx < len(frames)]
    if not frame_indices:
        raise ValueError('No valid frame indices selected for comparison.')

    start_pose_b_xyzquat = (
        replay_runtime._RESET_POSE_B_XYZQUAT.copy()
        if args.start_pose_b_xyzquat is None
        else np.asarray(args.start_pose_b_xyzquat, dtype=np.float64)
    )
    T_B_E_start = replay_runtime.pose_from_xyzquat(start_pose_b_xyzquat)
    dataset_start_contract_xyzquat, _ = infer_runtime.estimate_dataset_start_pose_contract(args.dataset)
    T_Ws_I0_contract = infer_runtime._pose_from_position_and_quaternion(
        dataset_start_contract_xyzquat[:3],
        dataset_start_contract_xyzquat[3:7],
    )
    T_B_Ws = T_B_E_start @ infer_runtime._T_EI @ infer_runtime._invert_pose(T_Ws_I0_contract)

    print(f'[INFO] dataset={args.dataset}')
    print(f'[INFO] episode={args.episode} source={args.source} frames={frame_indices}')
    print(f'[INFO] T(B,W_s).xyz={np.round(T_B_Ws[:3, 3], 6).tolist()}')

    position_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    for frame_idx in frame_indices:
        frame_xyzquat = np.asarray(frames[frame_idx][:7], dtype=np.float64)
        T_Ws_It = replay_runtime.pose_from_xyzquat(frame_xyzquat)
        replay_pose_e = replay_runtime.ws_to_base(T_B_Ws, T_Ws_It)
        infer_pose_e = infer_decode_pose(T_B_Ws, frame_xyzquat)
        position_error_mm, rotation_error_deg = pose_error(replay_pose_e, infer_pose_e)
        position_errors_mm.append(position_error_mm)
        rotation_errors_deg.append(rotation_error_deg)
        print(
            f'[CHECK] frame={frame_idx:4d} '
            f'pos_err_mm={position_error_mm:.9f} '
            f'rot_err_deg={rotation_error_deg:.9f}'
        )

    max_position_error_mm = float(np.max(position_errors_mm))
    max_rotation_error_deg = float(np.max(rotation_errors_deg))
    print(
        '[RESULT] '
        f'max_pos_err_mm={max_position_error_mm:.9f} '
        f'max_rot_err_deg={max_rotation_error_deg:.9f}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
