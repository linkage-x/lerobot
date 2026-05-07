#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.calibration.fr3_teleop import (
    build_combined_trace_profile,
    build_default_trace_profile,
    build_trace_bundle,
    build_wz_trace_profile,
    make_trace_sample,
    save_trace_bundle,
)
from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, update_passive_viewer_markers
from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config
from lerobot.utils.robot_utils import precise_sleep


DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT = (0.0, 0.0, 0.0)
DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT = (0.01, 0.01, 0.01)
DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lerobot"
    / "robots"
    / "franka_research3"
    / "assets"
    / "franka_fr3"
    / "fr3_pika_gripper_ati.urdf"
)


def _parse_tuple(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a fixed FR3 teleop profile and record TCP traces.")
    parser.add_argument("--mode", choices=["sim", "hardware"], required=True)
    parser.add_argument("--trace-profile", choices=["translation", "combined", "wz"], default="translation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--step-x", type=float, default=0.0002)
    parser.add_argument("--step-y", type=float, default=0.0002)
    parser.add_argument("--step-z", type=float, default=0.0002)
    parser.add_argument("--step-wx", type=float, default=0.0002)
    parser.add_argument("--step-wy", type=float, default=0.0002)
    parser.add_argument("--step-wz", type=float, default=0.0002)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--move-s", type=float, default=0.75)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--keep-viewer-open", action="store_true")
    parser.add_argument("--ik-solver", choices=["hirol_lm", "hirol_gaussian_newton", "placo"], default="hirol_lm")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--target-frame-name", default="pika_gripper_ee")
    parser.add_argument("--workspace-min", type=_parse_tuple, default=(0.2, -0.6, 0.05))
    parser.add_argument("--workspace-max", type=_parse_tuple, default=(0.9, 0.6, 0.8))
    parser.add_argument("--max-target-delta-pos", type=_parse_tuple, default=(0.001, 0.001, 0.001))
    parser.add_argument("--max-target-delta-rot", type=_parse_tuple, default=None)
    args = parser.parse_args()
    if args.max_target_delta_rot is None:
        if args.trace_profile in ("combined", "wz"):
            args.max_target_delta_rot = DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT
        else:
            args.max_target_delta_rot = DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT
    return args


def _pose_from_xyzquat(xyzquat: np.ndarray) -> np.ndarray:
    xyzquat = np.asarray(xyzquat, dtype=np.float64)
    quat = xyzquat[3:7]
    quat_norm = float(np.linalg.norm(quat))
    if not np.all(np.isfinite(xyzquat[:7])) or quat_norm <= 1e-12:
        raise ValueError(f"Invalid 7D pose for replay: {xyzquat[:7].tolist()}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = xyzquat[:3]
    pose[:3, :3] = Rotation.from_quat(quat / quat_norm).as_matrix()
    return pose


def _is_valid_pose7d(pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=np.float64)
    return pose.shape[0] >= 7 and np.all(np.isfinite(pose[:7])) and float(np.linalg.norm(pose[3:7])) > 1e-12


def _sanitize_pose_stream(poses: np.ndarray) -> tuple[np.ndarray, list[int]]:
    sanitized = np.asarray(poses, dtype=np.float64).copy()
    invalid_indices = [idx for idx, pose in enumerate(sanitized) if not _is_valid_pose7d(pose)]
    if not invalid_indices:
        sanitized[:, 3:7] /= np.linalg.norm(sanitized[:, 3:7], axis=1, keepdims=True)
        return sanitized, []

    invalid_index_set = set(invalid_indices)
    valid_indices = [idx for idx in range(len(sanitized)) if idx not in invalid_index_set]
    if not valid_indices:
        raise ValueError("Pose stream contains no valid 7D poses.")

    next_valid_index = valid_indices[0]
    last_valid_pose: np.ndarray | None = None
    for idx in range(len(sanitized)):
        if _is_valid_pose7d(sanitized[idx]):
            last_valid_pose = sanitized[idx].copy()
            sanitized[idx, 3:7] /= np.linalg.norm(sanitized[idx, 3:7])
            continue
        if last_valid_pose is None:
            sanitized[idx] = sanitized[next_valid_index]
        else:
            sanitized[idx] = last_valid_pose
        sanitized[idx, 3:7] /= np.linalg.norm(sanitized[idx, 3:7])
    return sanitized, invalid_indices


def _load_dataset_episode(dataset_path: Path, episode_idx: int) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    meta_dir = dataset_path / "meta" / "episodes"
    meta_files = sorted(meta_dir.rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata parquet files found in {meta_dir}.")

    chunk_idx = None
    file_idx = None
    dataset_from_index = 0
    dataset_to_index = 0
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file)).to_pydict()
        for row_idx, row_episode_idx in enumerate(table["episode_index"]):
            if int(row_episode_idx) != int(episode_idx):
                continue
            chunk_idx = int(table["data/chunk_index"][row_idx])
            file_idx = int(table["data/file_index"][row_idx])
            if "dataset_from_index" in table:
                dataset_from_index = int(table["dataset_from_index"][row_idx])
            if "dataset_to_index" in table:
                dataset_to_index = int(table["dataset_to_index"][row_idx])
            break
        if chunk_idx is not None:
            break
    if chunk_idx is None or file_idx is None:
        raise ValueError(f"Episode {episode_idx} not found in {dataset_path}.")

    candidates = [
        dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet",
        dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:06d}.parquet",
    ]
    data_file = next((candidate for candidate in candidates if candidate.exists()), None)
    if data_file is None:
        raise FileNotFoundError(f"No data parquet found for episode {episode_idx}; tried {candidates}.")

    table = pq.read_table(str(data_file)).to_pydict()
    row_indices = [idx for idx, value in enumerate(table["episode_index"]) if int(value) == int(episode_idx)]
    if not row_indices and dataset_to_index > dataset_from_index:
        row_indices = list(range(dataset_from_index, dataset_to_index))
    if not row_indices:
        raise ValueError(f"Episode {episode_idx} has no rows in {data_file}.")

    if "observation.state" not in table:
        raise KeyError(f"{data_file} is missing observation.state.")
    if "action" not in table:
        raise KeyError(f"{data_file} is missing action; absolute-pose replay needs a target pose stream.")

    states = np.asarray([table["observation.state"][idx] for idx in row_indices], dtype=np.float64)
    actions = np.asarray([table["action"][idx] for idx in row_indices], dtype=np.float64)
    timestamps = np.asarray([table["timestamp"][idx] for idx in row_indices], dtype=np.float64)
    frame_indices = np.asarray([table["frame_index"][idx] for idx in row_indices], dtype=np.int64)
    if states.ndim != 2 or states.shape[1] < 7:
        raise ValueError(f"observation.state must contain at least 7 pose values, got shape {states.shape}.")
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"action must contain at least 7 pose values, got shape {actions.shape}.")
    states, invalid_state_indices = _sanitize_pose_stream(states)
    actions, invalid_action_indices = _sanitize_pose_stream(actions)

    return {
        "episode_index": np.asarray([int(episode_idx)], dtype=np.int64),
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "invalid_state_indices": np.asarray(invalid_state_indices, dtype=np.int64),
        "invalid_action_indices": np.asarray(invalid_action_indices, dtype=np.int64),
        "data_file": np.asarray([str(data_file)]),
    }


def _load_dataset_episode_indices(dataset_path: Path) -> list[int]:
    import pyarrow.parquet as pq

    meta_dir = dataset_path / "meta" / "episodes"
    meta_files = sorted(meta_dir.rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata parquet files found in {meta_dir}.")

    episode_indices: list[int] = []
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file), columns=["episode_index"]).to_pydict()
        episode_indices.extend(int(episode_idx) for episode_idx in table["episode_index"])
    return sorted(episode_indices)


def _load_dataset_fps(dataset_path: Path) -> int | None:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = info.get("fps")
    if fps is None:
        return None
    return int(fps)


def _observation_to_sample(*, profile_step: int, scheduled_time_s: float, measured_time_s: float, action, observation, target_pose):
    joint_positions = np.array([observation[f"joint_{index}.pos"] for index in range(1, 8)], dtype=np.float64)
    ee_position = np.array(
        [observation["ee.x"], observation["ee.y"], observation["ee.z"]],
        dtype=np.float64,
    )
    ee_rotvec = np.array(
        [observation["ee.wx"], observation["ee.wy"], observation["ee.wz"]],
        dtype=np.float64,
    )
    target_position = None
    target_rotvec = None
    if target_pose is not None:
        target_position = np.asarray(target_pose[:3, 3], dtype=np.float64)
        target_rotvec = Rotation.from_matrix(target_pose[:3, :3]).as_rotvec()
    return make_trace_sample(
        profile_step=profile_step,
        scheduled_time_s=scheduled_time_s,
        measured_time_s=measured_time_s,
        action=action,
        joint_positions=joint_positions,
        ee_position=ee_position,
        ee_rotvec=ee_rotvec,
        gripper=float(observation["gripper.pos"]),
        target_position=target_position,
        target_rotvec=target_rotvec,
    )


def _sync_viewer(env: FR3MujocoEnv, viewer, info: dict[str, Any], marker_style: MarkerStyle) -> None:
    if viewer is None or not viewer.is_running():
        return
    with viewer.lock():
        update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
    viewer.sync()


def _hold_viewer_open(viewer) -> None:
    if viewer is None or not viewer.is_running():
        return
    print("fr3_trace_replay_viewer=OPEN press Esc in the MuJoCo viewer to close")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.016)


def _sim_info_to_sample(*, profile_step: int, scheduled_time_s: float, measured_time_s: float, action, info):
    target_pose = np.asarray(info["target_pose"], dtype=np.float64)
    tcp_pose = np.asarray(info["tcp_pose"], dtype=np.float64)
    ee_rotvec = Rotation.from_matrix(tcp_pose[:3, :3]).as_rotvec()
    target_rotvec = Rotation.from_matrix(target_pose[:3, :3]).as_rotvec()
    return make_trace_sample(
        profile_step=profile_step,
        scheduled_time_s=scheduled_time_s,
        measured_time_s=measured_time_s,
        action=action,
        joint_positions=np.asarray(info["joint_positions"], dtype=np.float64),
        ee_position=np.asarray(tcp_pose[:3, 3], dtype=np.float64),
        ee_rotvec=ee_rotvec,
        gripper=float(action["gripper"]) if action is not None else 1.0,
        target_position=np.asarray(target_pose[:3, 3], dtype=np.float64),
        target_rotvec=target_rotvec,
        extra={
            "otg_enabled": bool(info.get("otg_enabled", False)),
            "otg_steps": int(info.get("otg_steps", 0)),
            "sender_steps": int(info.get("sender_steps", 0)),
        },
    )


def run_sim_trace(args: argparse.Namespace, profile: dict[str, object]) -> dict[str, object]:
    env = FR3MujocoEnv(
        FR3MujocoEnvConfig(
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            max_target_delta_pos=args.max_target_delta_pos,
            max_target_delta_rot=args.max_target_delta_rot,
            teleop_control_frequency=float(args.fps),
            max_episode_steps=len(profile["actions"]) + 10,
            ik_solver=args.ik_solver,
        )
    )
    viewer = None
    marker_style = MarkerStyle()
    samples: list[dict[str, object]] = []
    try:
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        _, info = env.reset()
        _sync_viewer(env, viewer, info, marker_style)
        samples.append(
            _sim_info_to_sample(
                profile_step=-1,
                scheduled_time_s=0.0,
                measured_time_s=0.0,
                action=None,
                info=info,
            )
        )
        start_time = time.perf_counter()
        next_deadline = start_time
        for step_index, action in enumerate(profile["actions"]):
            _, _, terminated, truncated, info = env.step_teleop_action(action, control_period_s=1.0 / args.fps)
            measured_time_s = time.perf_counter() - start_time
            _sync_viewer(env, viewer, info, marker_style)
            samples.append(
                _sim_info_to_sample(
                    profile_step=step_index,
                    scheduled_time_s=(step_index + 1) / args.fps,
                    measured_time_s=measured_time_s,
                    action=action,
                    info=info,
                )
            )
            if terminated or truncated:
                break
            if args.viewer:
                next_deadline += 1.0 / args.fps
                precise_sleep(max(next_deadline - time.perf_counter(), 0.0))
        if args.keep_viewer_open:
            _hold_viewer_open(viewer)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    return build_trace_bundle(
        mode="sim",
        profile=profile,
        samples=samples,
        metadata={
            "urdf_path": str(args.urdf_path),
            "target_frame_name": args.target_frame_name,
            "ik_solver": args.ik_solver,
        },
    )


def _dataset_info_to_sample(
    *,
    profile_step: int,
    dataset_episode_index: int,
    scheduled_time_s: float,
    measured_time_s: float,
    info: dict[str, Any],
    action_pose_7d: np.ndarray,
    state_pose_7d: np.ndarray,
    dataset_frame_index: int,
    dataset_timestamp_s: float,
    gripper: float,
) -> dict[str, object]:
    target_pose = np.asarray(info["target_pose"], dtype=np.float64)
    tcp_pose = np.asarray(info["tcp_pose"], dtype=np.float64)
    return make_trace_sample(
        profile_step=profile_step,
        scheduled_time_s=scheduled_time_s,
        measured_time_s=measured_time_s,
        action=None,
        joint_positions=np.asarray(info["joint_positions"], dtype=np.float64),
        ee_position=np.asarray(tcp_pose[:3, 3], dtype=np.float64),
        ee_rotvec=Rotation.from_matrix(tcp_pose[:3, :3]).as_rotvec(),
        gripper=gripper,
        target_position=np.asarray(target_pose[:3, 3], dtype=np.float64),
        target_rotvec=Rotation.from_matrix(target_pose[:3, :3]).as_rotvec(),
        extra={
            "dataset_episode_index": int(dataset_episode_index),
            "dataset_frame_index": int(dataset_frame_index),
            "dataset_timestamp_s": float(dataset_timestamp_s),
            "dataset_action_pose": np.asarray(action_pose_7d[:7], dtype=np.float64).tolist(),
            "dataset_state_pose": np.asarray(state_pose_7d[:7], dtype=np.float64).tolist(),
            "otg_enabled": bool(info.get("otg_enabled", False)),
            "otg_steps": int(info.get("otg_steps", 0)),
            "sender_steps": int(info.get("sender_steps", 0)),
        },
    )


def run_sim_dataset_replay(args: argparse.Namespace) -> dict[str, object]:
    if args.dataset is None:
        raise ValueError("--dataset is required for dataset replay.")
    dataset_path = args.dataset.resolve()
    requested_episode_indices = [int(args.episode)] if args.episode is not None else _load_dataset_episode_indices(dataset_path)
    episodes = [_load_dataset_episode(dataset_path, episode_idx) for episode_idx in requested_episode_indices]
    if not episodes:
        raise ValueError(f"No episodes found in {dataset_path}.")
    total_action_count = int(sum(len(episode["actions"]) for episode in episodes))
    max_episode_steps = max(int(len(episode["actions"])) for episode in episodes) + 10
    dataset_fps = args.fps
    total_timestamp_duration_s = 0.0
    total_timestamp_intervals = 0
    for episode in episodes:
        timestamps = episode["timestamps"]
        if len(timestamps) > 1:
            total_timestamp_duration_s += float(timestamps[-1] - timestamps[0])
            total_timestamp_intervals += len(timestamps) - 1
    if total_timestamp_duration_s > 0 and total_timestamp_intervals > 0:
        dataset_fps = max(int(round(total_timestamp_intervals / total_timestamp_duration_s)), 1)

    env = FR3MujocoEnv(
        FR3MujocoEnvConfig(
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            max_target_delta_pos=args.max_target_delta_pos,
            max_target_delta_rot=args.max_target_delta_rot,
            teleop_control_frequency=float(args.fps),
            max_episode_steps=max_episode_steps,
            ik_solver=args.ik_solver,
        )
    )
    profile = {
        "name": f"dataset_episode_{args.episode}" if args.episode is not None else "dataset_all_episodes",
        "fps": args.fps,
        "actions": [None] * total_action_count,
        "segments": [
            {
                "name": "dataset_replay" if args.episode is not None else "dataset_all_replay",
                "kind": "absolute_pose",
                "start_step": 0,
                "end_step": max(total_action_count - 1, 0),
                "sample_start_index": 0,
                "sample_end_index": total_action_count,
                "step_count": total_action_count,
            }
        ],
    }
    viewer = None
    marker_style = MarkerStyle()
    samples: list[dict[str, object]] = []
    data_files: list[str] = []
    invalid_pose_replacements: list[dict[str, object]] = []
    try:
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        start_time = time.perf_counter()
        next_deadline = start_time
        scheduled_time_offset_s = 0.0
        global_step_index = 0
        for episode in episodes:
            episode_idx = int(episode["episode_index"][0])
            states = episode["states"]
            actions = episode["actions"]
            timestamps = episode["timestamps"]
            frame_indices = episode["frame_indices"]
            data_files.append(str(episode["data_file"][0]))
            if len(episode["invalid_state_indices"]) or len(episode["invalid_action_indices"]):
                invalid_pose_replacements.append(
                    {
                        "episode": episode_idx,
                        "invalid_state_frame_indices": [
                            int(frame_indices[idx]) for idx in episode["invalid_state_indices"].tolist()
                        ],
                        "invalid_action_frame_indices": [
                            int(frame_indices[idx]) for idx in episode["invalid_action_indices"].tolist()
                        ],
                    }
                )

            _, info = env.reset()
            initial_pose = _pose_from_xyzquat(states[0, :7])
            initial_joints = np.asarray(
                env._kinematics.inverse_kinematics(env._get_joint_positions(), initial_pose),
                dtype=np.float64,
            )
            _, info = env.reset(options={"joint_positions": initial_joints[: len(env.cfg.joint_names)]})
            _sync_viewer(env, viewer, info, marker_style)

            first_timestamp = float(timestamps[0]) if len(timestamps) else 0.0
            for episode_step_index, action_pose_7d in enumerate(actions[:, :7]):
                target_pose = _pose_from_xyzquat(action_pose_7d)
                gripper = float(np.clip(actions[episode_step_index, 7], 0.0, 1.0)) if actions.shape[1] > 7 else 1.0
                _, _, terminated, truncated, info = env.step_target_pose(
                    target_pose,
                    gripper=gripper,
                    control_period_s=1.0 / args.fps,
                )
                measured_time_s = time.perf_counter() - start_time
                scheduled_time_s = scheduled_time_offset_s + float(timestamps[episode_step_index] - first_timestamp)
                _sync_viewer(env, viewer, info, marker_style)
                samples.append(
                    _dataset_info_to_sample(
                        profile_step=global_step_index,
                        dataset_episode_index=episode_idx,
                        scheduled_time_s=scheduled_time_s,
                        measured_time_s=measured_time_s,
                        info=info,
                        action_pose_7d=action_pose_7d,
                        state_pose_7d=states[episode_step_index, :7],
                        dataset_frame_index=int(frame_indices[episode_step_index]),
                        dataset_timestamp_s=float(timestamps[episode_step_index]),
                        gripper=gripper,
                    )
                )
                global_step_index += 1
                if terminated or truncated:
                    break
                if args.viewer:
                    next_deadline += 1.0 / args.fps
                    precise_sleep(max(next_deadline - time.perf_counter(), 0.0))
            if len(timestamps) > 1:
                scheduled_time_offset_s += float(timestamps[-1] - first_timestamp) + 1.0 / args.fps
            else:
                scheduled_time_offset_s += len(actions) / args.fps
        if args.keep_viewer_open:
            _hold_viewer_open(viewer)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    return build_trace_bundle(
        mode="sim_dataset",
        profile=profile,
        samples=samples,
        metadata={
            "dataset": str(dataset_path),
            "episode": int(args.episode) if args.episode is not None else None,
            "episode_indices": requested_episode_indices,
            "episode_count": len(requested_episode_indices),
            "dataset_fps_estimate": dataset_fps,
            "dataset_data_files": sorted(set(data_files)),
            "invalid_pose_replacement_count": int(
                sum(
                    len(row["invalid_state_frame_indices"]) + len(row["invalid_action_frame_indices"])
                    for row in invalid_pose_replacements
                )
            ),
            "invalid_pose_replacements": invalid_pose_replacements,
            "urdf_path": str(args.urdf_path),
            "target_frame_name": args.target_frame_name,
            "ik_solver": args.ik_solver,
        },
    )


def run_hardware_trace(args: argparse.Namespace, profile: dict[str, object]) -> dict[str, object]:
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip=args.robot_ip,
            gripper_port=args.gripper_port,
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            max_target_delta_pos=args.max_target_delta_pos,
            max_target_delta_rot=args.max_target_delta_rot,
        )
    )
    samples: list[dict[str, object]] = []
    try:
        robot.connect()
        observation = robot.get_observation()
        samples.append(
            _observation_to_sample(
                profile_step=-1,
                scheduled_time_s=0.0,
                measured_time_s=0.0,
                action=None,
                observation=observation,
                target_pose=getattr(robot, "_last_command_pose", None),
            )
        )
        start_time = time.perf_counter()
        next_deadline = start_time
        for step_index, action in enumerate(profile["actions"]):
            robot.send_action(action)
            next_deadline += 1.0 / args.fps
            precise_sleep(max(next_deadline - time.perf_counter(), 0.0))
            observation = robot.get_observation()
            measured_time_s = time.perf_counter() - start_time
            samples.append(
                _observation_to_sample(
                    profile_step=step_index,
                    scheduled_time_s=(step_index + 1) / args.fps,
                    measured_time_s=measured_time_s,
                    action=action,
                    observation=observation,
                    target_pose=getattr(robot, "_last_command_pose", None),
                )
            )
    finally:
        if robot.is_connected:
            robot.disconnect()
    return build_trace_bundle(
        mode="hardware",
        profile=profile,
        samples=samples,
        metadata={
            "robot_ip": args.robot_ip,
            "gripper_port": args.gripper_port,
            "gripper_is_mock": bool(getattr(robot, "_gripper_is_mock", False)),
            "urdf_path": str(args.urdf_path),
            "target_frame_name": args.target_frame_name,
        },
    )


def _resolve_output_path(args: argparse.Namespace) -> Path:
    output_path = args.output
    if not output_path.exists() and output_path.suffix:
        return output_path
    if output_path.exists() and not output_path.is_dir():
        return output_path
    if output_path.suffix and args.dataset is None:
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    if args.dataset is not None:
        dataset_name = args.dataset.resolve().name
        episode_label = "all" if args.episode is None else f"ep{int(args.episode):03d}"
        return output_path / f"{dataset_name}_{episode_label}_trace.json"
    return output_path / f"{args.mode}_trace.json"


def main() -> int:
    args = parse_args()
    if args.fps is None:
        args.fps = _load_dataset_fps(args.dataset.resolve()) if args.dataset is not None else 60
    if args.fps is None:
        args.fps = 60
    if args.trace_profile == "combined":
        profile = build_combined_trace_profile(
            fps=args.fps,
            step_x=args.step_x,
            step_y=args.step_y,
            step_z=args.step_z,
            step_wx=args.step_wx,
            step_wy=args.step_wy,
            step_wz=args.step_wz,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    elif args.trace_profile == "wz":
        profile = build_wz_trace_profile(
            fps=args.fps,
            step_wz=args.step_wz,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    else:
        profile = build_default_trace_profile(
            fps=args.fps,
            step_x=args.step_x,
            step_y=args.step_y,
            step_z=args.step_z,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    if args.mode == "sim":
        if args.dataset is not None:
            bundle = run_sim_dataset_replay(args)
        else:
            bundle = run_sim_trace(args, profile)
    else:
        bundle = run_hardware_trace(args, profile)
    output_path = _resolve_output_path(args)
    save_trace_bundle(output_path, bundle)
    print(f"fr3_trace_replay={args.mode}")
    print(f"output={output_path}")
    print(f"samples={len(bundle['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
