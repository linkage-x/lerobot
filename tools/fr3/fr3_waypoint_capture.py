#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture FR3 fixed-waypoint image datasets with synchronized EE pose snapshots."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import franka_research3, make_robot_from_config
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_ROTVEC_KEYS = ("ee.wx", "ee.wy", "ee.wz")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
STATE_VECTOR_NAMES = [
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.qx",
    "ee.qy",
    "ee.qz",
    "ee.qw",
    "gripper.pos",
]


@dataclass
class WaypointCaptureDatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 1
    num_episodes: int = 1
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "h264"
    streaming_encoding: bool = True
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = 2

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("`dataset.repo_id` must be a non-empty string.")
        if not self.single_task:
            raise ValueError("`dataset.single_task` must be a non-empty string.")
        if self.fps <= 0:
            raise ValueError(f"`dataset.fps` must be > 0, but {self.fps} is provided.")
        if self.num_episodes <= 0:
            raise ValueError(f"`dataset.num_episodes` must be > 0, but {self.num_episodes} is provided.")


@dataclass
class WaypointCaptureRuntimeConfig:
    settle_time_s: float = 1.0
    settle_timeout_s: float = 8.0
    settle_position_threshold_m: float = 0.003
    settle_angle_threshold_deg: float = 2.0
    settle_consecutive_samples: int = 3
    settle_check_fps: int = 30
    max_read_age_ms: int = 500
    auto_move_to_start_before_episode: bool = False

    def __post_init__(self) -> None:
        if self.settle_time_s < 0:
            raise ValueError(f"`runtime.settle_time_s` must be >= 0, but {self.settle_time_s} is provided.")
        if self.settle_timeout_s <= 0:
            raise ValueError(
                f"`runtime.settle_timeout_s` must be > 0, but {self.settle_timeout_s} is provided."
            )
        if self.settle_position_threshold_m <= 0:
            raise ValueError(
                "`runtime.settle_position_threshold_m` must be > 0, "
                f"but {self.settle_position_threshold_m} is provided."
            )
        if self.settle_angle_threshold_deg <= 0:
            raise ValueError(
                f"`runtime.settle_angle_threshold_deg` must be > 0, but {self.settle_angle_threshold_deg} is provided."
            )
        if self.settle_consecutive_samples <= 0:
            raise ValueError(
                "`runtime.settle_consecutive_samples` must be > 0, "
                f"but {self.settle_consecutive_samples} is provided."
            )
        if self.settle_check_fps <= 0:
            raise ValueError(
                f"`runtime.settle_check_fps` must be > 0, but {self.settle_check_fps} is provided."
            )
        if self.max_read_age_ms <= 0:
            raise ValueError(f"`runtime.max_read_age_ms` must be > 0, but {self.max_read_age_ms} is provided.")


@dataclass
class WaypointCaptureConfig:
    robot: FrankaResearch3Config
    dataset: WaypointCaptureDatasetConfig
    waypoints: list[dict[str, float]]
    runtime: WaypointCaptureRuntimeConfig = field(default_factory=WaypointCaptureRuntimeConfig)
    resume: bool = False


def _normalize_camera_frame_to_rgb(camera: Any, frame: np.ndarray) -> np.ndarray:
    color_mode = getattr(getattr(camera, "config", None), "color_mode", None)
    try:
        color_mode = ColorMode(color_mode)
    except ValueError:
        color_mode = None

    if color_mode == ColorMode.BGR:
        return np.ascontiguousarray(frame[..., ::-1])
    return np.ascontiguousarray(frame)


def _get_latest_timestamp(device: Any) -> float:
    lock = getattr(device, "frame_lock", None) or getattr(device, "read_lock", None)
    if lock is not None:
        with lock:
            timestamp = getattr(device, "latest_timestamp", None)
    else:
        timestamp = getattr(device, "latest_timestamp", None)

    if timestamp is None:
        raise RuntimeError(f"{type(device).__name__} has not produced a timestamp yet.")
    return float(timestamp)


def _build_capture_timestamp_feature_names(camera_names: list[str]) -> list[str]:
    names = ["robot.ee.capture_timestamp_s"]
    for camera_name in camera_names:
        names.append(f"camera.{camera_name}.capture_timestamp_s")
    return names


def _build_dataset_features(robot: Any, *, use_videos: bool) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_VECTOR_NAMES),),
            "names": STATE_VECTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_VECTOR_NAMES),),
            "names": STATE_VECTOR_NAMES,
        },
    }

    camera_names = list(robot.cameras.keys())
    capture_timestamp_names = _build_capture_timestamp_feature_names(camera_names)
    features["observation.device_capture_timestamp"] = {
        "dtype": "float64",
        "shape": (len(capture_timestamp_names),),
        "names": capture_timestamp_names,
    }

    for camera_name, camera in robot.cameras.items():
        height = int(camera.height or camera.config.height or 0)
        width = int(camera.width or camera.config.width or 0)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"Camera '{camera_name}' did not report a valid image size. "
                f"Resolved height={height}, width={width}."
            )
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def _validate_resume_compatibility(dataset: LeRobotDataset, expected_features: dict[str, dict[str, Any]], fps: int) -> None:
    if dataset.fps != fps:
        raise ValueError(
            f"Dataset fps mismatch while resuming: existing dataset uses {dataset.fps}, config requests {fps}."
        )

    existing_signatures = {
        key: (feature["dtype"], tuple(feature["shape"]), feature.get("names"))
        for key, feature in dataset.features.items()
    }
    expected_signatures = {
        key: (feature["dtype"], tuple(feature["shape"]), feature.get("names"))
        for key, feature in expected_features.items()
    }
    if existing_signatures != expected_signatures:
        raise ValueError(
            "Dataset feature mismatch while resuming.\n"
            f"Existing: {existing_signatures}\n"
            f"Expected: {expected_signatures}"
        )


def _create_or_resume_dataset(
    cfg: WaypointCaptureConfig,
    *,
    features: dict[str, dict[str, Any]],
    num_cameras: int,
) -> LeRobotDataset:
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        _validate_resume_compatibility(dataset, features, cfg.dataset.fps)
        if num_cameras > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )
        return dataset

    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type="franka_research3_waypoint_capture",
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )


def _validate_waypoint(index: int, waypoint: dict[str, float]) -> None:
    missing_position = [key for key in EE_POSITION_KEYS if key not in waypoint]
    if missing_position:
        raise ValueError(f"waypoints[{index}] is missing position keys: {missing_position}")

    has_rotvec = all(key in waypoint for key in EE_ROTVEC_KEYS)
    has_quat = all(key in waypoint for key in EE_QUAT_KEYS)
    if has_rotvec == has_quat:
        raise ValueError(
            f"waypoints[{index}] must provide exactly one orientation format: "
            f"{EE_ROTVEC_KEYS} or {EE_QUAT_KEYS}."
        )

    if "gripper.pos" in waypoint:
        gripper = float(waypoint["gripper.pos"])
        if gripper < 0.0 or gripper > 1.0:
            raise ValueError(
                f"waypoints[{index}]['gripper.pos'] must be in [0, 1], got {gripper}."
            )


def _validate_waypoints(waypoints: list[dict[str, float]]) -> None:
    if not waypoints:
        raise ValueError("`waypoints` must contain at least one waypoint.")
    for idx, waypoint in enumerate(waypoints):
        _validate_waypoint(idx, waypoint)


def _quaternion_angle_error_deg(target_xyzw: np.ndarray, current_xyzw: np.ndarray) -> float:
    dot = float(np.dot(np.asarray(target_xyzw, dtype=np.float64), np.asarray(current_xyzw, dtype=np.float64)))
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0))))


def _waypoint_to_robot_target(
    waypoint: dict[str, float],
    *,
    default_gripper_pos: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    target_position_xyz = np.array([float(waypoint[key]) for key in EE_POSITION_KEYS], dtype=np.float64)

    if all(key in waypoint for key in EE_ROTVEC_KEYS):
        target_rotvec = np.array([float(waypoint[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64)
        target_quaternion_xyzw = Rotation.from_rotvec(target_rotvec).as_quat()
    else:
        target_quaternion_xyzw = np.array([float(waypoint[key]) for key in EE_QUAT_KEYS], dtype=np.float64)
        norm = float(np.linalg.norm(target_quaternion_xyzw))
        if norm <= 1e-9:
            raise ValueError("Waypoint quaternion norm must be > 0.")
        target_quaternion_xyzw = target_quaternion_xyzw / norm
        target_rotvec = Rotation.from_quat(target_quaternion_xyzw).as_rotvec()

    gripper_pos = float(np.clip(waypoint.get("gripper.pos", default_gripper_pos), 0.0, 1.0))
    target_action = {
        "ee.x": float(target_position_xyz[0]),
        "ee.y": float(target_position_xyz[1]),
        "ee.z": float(target_position_xyz[2]),
        "ee.wx": float(target_rotvec[0]),
        "ee.wy": float(target_rotvec[1]),
        "ee.wz": float(target_rotvec[2]),
        "gripper.pos": gripper_pos,
    }
    action_vector = np.array(
        [
            float(target_position_xyz[0]),
            float(target_position_xyz[1]),
            float(target_position_xyz[2]),
            float(target_quaternion_xyzw[0]),
            float(target_quaternion_xyzw[1]),
            float(target_quaternion_xyzw[2]),
            float(target_quaternion_xyzw[3]),
            gripper_pos,
        ],
        dtype=np.float32,
    )
    return target_action, target_position_xyz, target_quaternion_xyzw, action_vector


def _extract_observation_state_vector(observation: dict[str, Any]) -> np.ndarray:
    ee_position_xyz = np.array([float(observation[key]) for key in EE_POSITION_KEYS], dtype=np.float32)
    ee_rotvec = np.array([float(observation[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64)
    ee_quaternion_xyzw = Rotation.from_rotvec(ee_rotvec).as_quat().astype(np.float32, copy=False)
    gripper_pos = np.array([float(observation["gripper.pos"])], dtype=np.float32)
    return np.concatenate((ee_position_xyz, ee_quaternion_xyzw, gripper_pos), dtype=np.float32)


def _wait_for_waypoint_settle(
    robot: Any,
    *,
    target_position_xyz: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    runtime_cfg: WaypointCaptureRuntimeConfig,
) -> tuple[bool, float, float]:
    settle_start_t = time.perf_counter()
    consecutive_ok = 0
    last_position_error_m = float("inf")
    last_angle_error_deg = float("inf")
    settle_period_s = 1.0 / float(runtime_cfg.settle_check_fps)

    while True:
        loop_start_t = time.perf_counter()
        observation = robot.get_observation(include_cameras=False)
        current_position_xyz = np.array([float(observation[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
        current_rotvec = np.array([float(observation[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64)
        current_quaternion_xyzw = Rotation.from_rotvec(current_rotvec).as_quat()

        last_position_error_m = float(np.linalg.norm(current_position_xyz - target_position_xyz))
        last_angle_error_deg = _quaternion_angle_error_deg(target_quaternion_xyzw, current_quaternion_xyzw)
        if (
            last_position_error_m <= runtime_cfg.settle_position_threshold_m
            and last_angle_error_deg <= runtime_cfg.settle_angle_threshold_deg
        ):
            consecutive_ok += 1
            if consecutive_ok >= runtime_cfg.settle_consecutive_samples:
                return True, last_position_error_m, last_angle_error_deg
        else:
            consecutive_ok = 0

        if time.perf_counter() - settle_start_t >= runtime_cfg.settle_timeout_s:
            return False, last_position_error_m, last_angle_error_deg

        precise_sleep(max(settle_period_s - (time.perf_counter() - loop_start_t), 0.0))


def _capture_waypoint_frame(
    robot: Any,
    *,
    action_vector: np.ndarray,
    task: str,
    episode_start_time_s: float,
    max_read_age_ms: int,
) -> dict[str, Any]:
    observation = robot.get_observation(include_cameras=True)
    capture_time_s = time.perf_counter()
    frame: dict[str, Any] = {
        "observation.state": _extract_observation_state_vector(observation),
        "action": np.asarray(action_vector, dtype=np.float32),
        "task": task,
    }

    capture_timestamp_values: list[float] = [capture_time_s - episode_start_time_s]
    for camera_name, camera in robot.cameras.items():
        camera_frame = np.asarray(observation[camera_name], dtype=np.uint8)
        frame[f"observation.images.{camera_name}"] = _normalize_camera_frame_to_rgb(camera, camera_frame)
        camera_capture_time_s = _get_latest_timestamp(camera)
        age_ms = (capture_time_s - camera_capture_time_s) * 1e3
        if age_ms > max_read_age_ms:
            raise TimeoutError(
                f"Camera '{camera_name}' frame is too old: {age_ms:.1f} ms "
                f"(max allowed: {max_read_age_ms} ms)."
            )
        capture_timestamp_values.append(camera_capture_time_s - episode_start_time_s)

    frame["observation.device_capture_timestamp"] = np.asarray(capture_timestamp_values, dtype=np.float64)
    return frame


def _move_to_start_if_supported(robot: Any) -> None:
    move_to_start = getattr(robot, "move_to_start", None)
    if not callable(move_to_start):
        logging.warning(
            "Robot '%s' does not expose move_to_start(); continuing without auto reset.",
            getattr(robot, "name", type(robot).__name__),
        )
        return
    logging.info("Moving robot to start pose before recording.")
    move_to_start()


def _print_intro(cfg: WaypointCaptureConfig, robot: Any, dataset: LeRobotDataset) -> None:
    print("FR3 waypoint capture recorder")
    print(f"Dataset repo_id: {cfg.dataset.repo_id}")
    print(f"Dataset root: {dataset.root}")
    print(f"Configured waypoints: {len(cfg.waypoints)}")
    print(f"Configured cameras: {len(robot.cameras)} ({', '.join(robot.cameras.keys()) if robot.cameras else '(none)'})")
    print(f"Target episodes: {cfg.dataset.num_episodes}")


@parser.wrap()
def record(cfg: WaypointCaptureConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    _validate_waypoints(cfg.waypoints)

    robot = make_robot_from_config(cfg.robot)
    dataset: LeRobotDataset | None = None

    try:
        robot.connect()
        features = _build_dataset_features(robot, use_videos=cfg.dataset.video)
        dataset = _create_or_resume_dataset(cfg, features=features, num_cameras=len(robot.cameras))
        _print_intro(cfg, robot, dataset)

        if cfg.runtime.auto_move_to_start_before_episode:
            _move_to_start_if_supported(robot)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes:
                logging.info("Recording episode %d/%d.", recorded_episodes + 1, cfg.dataset.num_episodes)
                episode_start_time_s = time.perf_counter()
                initial_observation = robot.get_observation(include_cameras=False)
                default_gripper_pos = float(initial_observation["gripper.pos"])

                for waypoint_index, waypoint in enumerate(cfg.waypoints):
                    target_action, target_position_xyz, target_quaternion_xyzw, action_vector = _waypoint_to_robot_target(
                        waypoint,
                        default_gripper_pos=default_gripper_pos,
                    )
                    logging.info(
                        "Waypoint %d/%d target xyz=(%.4f, %.4f, %.4f) gripper=%.3f",
                        waypoint_index + 1,
                        len(cfg.waypoints),
                        target_action["ee.x"],
                        target_action["ee.y"],
                        target_action["ee.z"],
                        target_action["gripper.pos"],
                    )
                    robot.send_action(target_action)
                    settled, position_error_m, angle_error_deg = _wait_for_waypoint_settle(
                        robot,
                        target_position_xyz=target_position_xyz,
                        target_quaternion_xyzw=target_quaternion_xyzw,
                        runtime_cfg=cfg.runtime,
                    )
                    if not settled:
                        logging.warning(
                            "Waypoint %d settle timed out: position_error=%.4fm angle_error=%.2fdeg",
                            waypoint_index + 1,
                            position_error_m,
                            angle_error_deg,
                        )

                    if cfg.runtime.settle_time_s > 0:
                        precise_sleep(cfg.runtime.settle_time_s)

                    frame = _capture_waypoint_frame(
                        robot,
                        action_vector=action_vector,
                        task=cfg.dataset.single_task,
                        episode_start_time_s=episode_start_time_s,
                        max_read_age_ms=cfg.runtime.max_read_age_ms,
                    )
                    default_gripper_pos = float(frame["observation.state"][-1])
                    dataset.add_frame(frame)

                dataset.save_episode()
                recorded_episodes += 1
                logging.info(
                    "Saved episode %d/%d with %d frames.",
                    recorded_episodes,
                    cfg.dataset.num_episodes,
                    len(cfg.waypoints),
                )

                if cfg.runtime.auto_move_to_start_before_episode and recorded_episodes < cfg.dataset.num_episodes:
                    _move_to_start_if_supported(robot)

        return dataset
    finally:
        if dataset is not None:
            dataset.finalize()
            if cfg.dataset.push_to_hub:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
