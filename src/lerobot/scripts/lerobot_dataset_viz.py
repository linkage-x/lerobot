#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

- Visualize data on a distant machine and switch episodes from a local terminal:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876 \
    --control-port 9999

local$ rerun rerun+http://IP:9876/proxy
local$ dataset_viz_client.py --host IP --control-port 9999
```

"""

import argparse
import contextlib
import csv
import gc
import logging
import multiprocessing as mp
import os
import queue
import select
import socket
import socketserver
import sys
import termios
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import tty
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.rotation import Rotation
from lerobot.utils.state_feature_names import flatten_feature_name_paths, get_ee_pose_state_indices
from lerobot.utils.utils import init_logging


EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
EE_RULER_AXIS_COLORS = {
    "x": [255, 0, 0, 255],
    "y": [0, 255, 0, 255],
    "z": [0, 0, 255, 255],
}

QUIT_COMMANDS = {"q", "\x03", "\x04"}
MANUALLY_LOGGED_KEYS = {ACTION, OBS_STATE, DONE, REWARD, "next.success"}
AUTO_VIZ_EXCLUDED_KEYS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
_WARNED_NONE_COLLATE_KEYS: set[str] = set()
POSE7_STATE_COLUMNS = ("state_x_m", "state_y_m", "state_z_m", "state_qx", "state_qy", "state_qz", "state_qw")
POSE7_ACTION_COLUMNS = (
    "action_x_m",
    "action_y_m",
    "action_z_m",
    "action_qx",
    "action_qy",
    "action_qz",
    "action_qw",
)
POSE7_SCALAR_NAMES = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
CUBE_BASE_POSE7_COLUMNS = (
    "cube_base_x_m",
    "cube_base_y_m",
    "cube_base_z_m",
    "cube_base_qx",
    "cube_base_qy",
    "cube_base_qz",
    "cube_base_qw",
)
CUBE_CAMERA_POSE7_COLUMNS = (
    "cube_cam_x_m",
    "cube_cam_y_m",
    "cube_cam_z_m",
    "cube_cam_qx",
    "cube_cam_qy",
    "cube_cam_qz",
    "cube_cam_qw",
)


def collate_without_none_features(batch: list[dict]) -> dict:
    none_keys = sorted({key for item in batch for key, value in item.items() if value is None})
    if none_keys:
        new_none_keys = [key for key in none_keys if key not in _WARNED_NONE_COLLATE_KEYS]
        if new_none_keys:
            logging.warning(
                "Skipping %d dataset feature(s) with None values during visualization: %s",
                len(new_none_keys),
                ", ".join(new_none_keys),
            )
            _WARNED_NONE_COLLATE_KEYS.update(new_none_keys)
        batch = [{key: value for key, value in item.items() if key not in none_keys} for item in batch]
    return torch.utils.data.default_collate(batch)


class RawEpisodeVideoReaders:
    def __init__(self, video_paths: dict[str, Path]) -> None:
        import cv2

        self._cv2 = cv2
        self._caps = {}
        self._next_frame_indices = {}
        for key, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logging.warning("Could not open raw episode video for %s: %s", key, path)
                continue
            self._caps[key] = cap
            self._next_frame_indices[key] = 0

    @property
    def keys(self) -> list[str]:
        return list(self._caps)

    def read_frame(self, key: str, frame_index: int) -> np.ndarray | None:
        cap = self._caps.get(key)
        if cap is None:
            return None
        next_frame_index = self._next_frame_indices[key]
        if next_frame_index > frame_index:
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
            next_frame_index = frame_index

        while next_frame_index < frame_index:
            ok, _ = cap.read()
            if not ok:
                logging.warning("Could not skip to frame %d in raw episode video %s", frame_index, key)
                return None
            next_frame_index += 1

        ok, frame_bgr = cap.read()
        if not ok:
            logging.warning("Could not read frame %d from raw episode video %s", frame_index, key)
            return None
        self._next_frame_indices[key] = frame_index + 1
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()


def find_raw_episode_video_paths(root: Path | None, episode_index: int) -> dict[str, Path]:
    if root is None:
        return {}
    episode_dir = root / "episodes" / f"episode_{episode_index:06d}"
    if not episode_dir.exists():
        return {}
    video_paths = {}
    for path in sorted([*episode_dir.glob("*.mp4"), *episode_dir.glob("*.mkv")]):
        video_paths[path.stem] = path
    return video_paths


def filter_raw_episode_video_paths(video_paths: dict[str, Path], raw_video_keys: str | None) -> dict[str, Path]:
    if raw_video_keys is None:
        return video_paths
    requested_keys = [key.strip() for key in raw_video_keys.split(",") if key.strip()]
    if not requested_keys:
        return video_paths
    missing_keys = [key for key in requested_keys if key not in video_paths]
    if missing_keys:
        logging.warning("Requested raw video camera(s) not found: %s", ", ".join(missing_keys))
    return {key: video_paths[key] for key in requested_keys if key in video_paths}


def get_raw_video_frame_count(video_path: Path) -> int | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def log_episode_frame_counts(
    *,
    episode_index: int,
    lowdim_frame_count: int,
    raw_video_paths: dict[str, Path],
    derived_pose_trajectory: dict | None,
    derived_cube_pose_trajectories: list[dict],
) -> None:
    logging.info("Episode %d low-dim frames: %d", episode_index, lowdim_frame_count)
    if raw_video_paths:
        for key, path in raw_video_paths.items():
            frame_count = get_raw_video_frame_count(path)
            if frame_count is None:
                logging.warning("Episode %d raw video %s frames: unreadable (%s)", episode_index, key, path)
            else:
                logging.info("Episode %d raw video %s frames: %d", episode_index, key, frame_count)
    if derived_pose_trajectory is not None:
        logging.info(
            "Episode %d derived pose '%s' frames: state=%d action=%d",
            episode_index,
            derived_pose_trajectory["pose_name"],
            len(derived_pose_trajectory["state_by_frame"]),
            len(derived_pose_trajectory["action_by_frame"]),
        )
    for trajectory in derived_cube_pose_trajectories:
        logging.info(
            "Episode %d derived cube pose '%s' frames: base=%d sources=%s",
            episode_index,
            trajectory["cube_name"],
            len(trajectory["base_by_frame"]),
            ",".join(trajectory["source_names"]),
        )


def _parse_pose7(row: dict[str, str], columns: tuple[str, ...]) -> np.ndarray | None:
    try:
        pose = np.asarray([float(row[column]) for column in columns], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(pose)):
        return None
    return pose


def _pose_name_from_state_action_csv(path: Path) -> str:
    prefix = "state_action."
    if path.name.startswith(prefix) and path.name.endswith(".csv"):
        return path.name[len(prefix) : -len(".csv")]
    return "default"


def load_derived_pose_trajectory(
    root: Path | None,
    episode_index: int,
    derived_name: str | None,
    pose_name: str | None,
) -> dict | None:
    if root is None:
        return None
    derived_root = root / "derived"
    if not derived_root.exists():
        return None

    derived_dirs = [derived_root / derived_name] if derived_name else sorted(path for path in derived_root.iterdir() if path.is_dir())
    csv_paths = []
    for derived_dir in derived_dirs:
        if not derived_dir.exists():
            continue
        if pose_name:
            csv_paths.append(derived_dir / f"state_action.{pose_name}.csv")
        else:
            csv_paths.extend(sorted(derived_dir.glob("state_action.*.csv")))
            csv_paths.extend(sorted(derived_dir.glob("state_action.csv")))

    best = None
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        state_by_frame = {}
        action_by_frame = {}
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row.get("episode_index", -1)) != episode_index:
                        continue
                    frame_index = int(row["frame_index"])
                except (TypeError, ValueError):
                    continue
                state_pose = _parse_pose7(row, POSE7_STATE_COLUMNS)
                action_pose = _parse_pose7(row, POSE7_ACTION_COLUMNS)
                if state_pose is not None:
                    state_by_frame[frame_index] = state_pose
                if action_pose is not None:
                    action_by_frame[frame_index] = action_pose

        num_valid = len(state_by_frame) + len(action_by_frame)
        candidate = {
            "pose_name": pose_name or _pose_name_from_state_action_csv(csv_path),
            "path": csv_path,
            "state_by_frame": state_by_frame,
            "action_by_frame": action_by_frame,
            "num_valid": num_valid,
        }
        if best is None or candidate["num_valid"] > best["num_valid"]:
            best = candidate

    if best is None or best["num_valid"] == 0:
        return None
    return best


def _cube_pose_name_from_csv(path: Path) -> tuple[str, str]:
    parts = path.stem.split(".")
    if len(parts) >= 3 and parts[0] == "cube_pose":
        return parts[1], parts[2]
    return "unknown", path.stem


def load_derived_cube_pose_trajectories(
    root: Path | None,
    episode_index: int,
    derived_name: str | None,
    cube_name: str | None,
) -> list[dict]:
    if root is None:
        return []
    derived_root = root / "derived"
    if not derived_root.exists():
        return []

    derived_dirs = [derived_root / derived_name] if derived_name else sorted(path for path in derived_root.iterdir() if path.is_dir())
    csv_paths = []
    for derived_dir in derived_dirs:
        if not derived_dir.exists():
            continue
        pattern = f"cube_pose.{cube_name}.*.csv" if cube_name else "cube_pose.*.*.csv"
        csv_paths.extend(sorted(derived_dir.glob(pattern)))

    trajectories = []
    for csv_path in csv_paths:
        parsed_cube_name, camera_name = _cube_pose_name_from_csv(csv_path)
        base_by_frame = {}
        camera_by_frame = {}
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row.get("episode_index", -1)) != episode_index:
                        continue
                    frame_index = int(row["frame_index"])
                except (TypeError, ValueError):
                    continue
                base_pose = _parse_pose7(row, CUBE_BASE_POSE7_COLUMNS)
                camera_pose = _parse_pose7(row, CUBE_CAMERA_POSE7_COLUMNS)
                if base_pose is not None:
                    base_by_frame[frame_index] = base_pose
                if camera_pose is not None:
                    camera_by_frame[frame_index] = camera_pose

        if not base_by_frame and not camera_by_frame:
            continue
        trajectories.append(
            {
                "cube_name": parsed_cube_name,
                "camera_name": camera_name,
                "path": csv_path,
                "base_by_frame": base_by_frame,
                "camera_by_frame": camera_by_frame,
            }
        )
    return trajectories


def aggregate_cube_base_trajectories(cube_pose_sources: list[dict]) -> list[dict]:
    trajectories_by_cube = {}
    for source in sorted(cube_pose_sources, key=lambda item: (item["cube_name"], item["camera_name"])):
        cube_name = source["cube_name"]
        trajectory = trajectories_by_cube.setdefault(
            cube_name,
            {
                "cube_name": cube_name,
                "base_by_frame": {},
                "source_count": 0,
                "source_names": [],
            },
        )
        trajectory["source_count"] += 1
        trajectory["source_names"].append(source["camera_name"])
        for frame_index, pose in sorted(source["base_by_frame"].items()):
            trajectory["base_by_frame"].setdefault(frame_index, pose)

    return [trajectory for trajectory in trajectories_by_cube.values() if trajectory["base_by_frame"]]


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def tensor_to_rerun_image_numpy(value: torch.Tensor) -> np.ndarray | None:
    if value.ndim == 2:
        image = value.detach().cpu().numpy()
        if image.dtype == np.bool_:
            return image.astype(np.uint8) * 255
        return image

    if value.ndim == 3 and value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        if value.dtype == torch.bool:
            image = value.detach().cpu().to(torch.uint8).mul(255).permute(1, 2, 0).numpy()
        elif value.dtype == torch.float32 and value.min().item() >= 0.0 and value.max().item() <= 1.0:
            image = to_hwc_uint8_numpy(value)
        else:
            image = value.detach().cpu().permute(1, 2, 0).numpy()
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[..., 0]
        return image

    return None


def build_scalar_entity_paths(
    root: str,
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    width: int,
) -> list[str]:
    flattened_names = flatten_feature_name_paths(feature_names)
    if flattened_names is None or len(flattened_names) != width:
        return [f"{root}/{dim_idx}" for dim_idx in range(width)]
    return [f"{root}/{name}" for name in flattened_names]


def build_device_capture_timestamp_entity_paths(
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    width: int,
) -> list[str]:
    root = "observation/device_capture_timestamp"
    flattened_names = flatten_feature_name_paths(feature_names)
    if flattened_names is None or len(flattened_names) != width:
        return [f"{root}/{dim_idx}" for dim_idx in range(width)]
    entity_paths = []
    for name in flattened_names:
        normalized_name = name.replace(".", "/")
        if normalized_name.endswith("/capture_timestamp_s"):
            normalized_name = normalized_name.removesuffix("/capture_timestamp_s")
        entity_paths.append(f"{root}/{normalized_name}")
    return entity_paths


def get_auto_visualization_keys(batch: dict, camera_keys: list[str] | tuple[str, ...]) -> list[str]:
    excluded_keys = set(camera_keys) | MANUALLY_LOGGED_KEYS | AUTO_VIZ_EXCLUDED_KEYS
    return [
        key
        for key in batch
        if key not in excluded_keys and not key.endswith("_is_pad") and not isinstance(batch[key], str | bytes)
    ]


def log_feature_value(
    key: str,
    value: torch.Tensor,
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    display_compressed_images: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        return

    if key == "observation.device_capture_timestamp":
        value = as_1d_tensor(value)

    if value.ndim == 0:
        rr.log(key, rr.Scalars(value.item()))
        return

    if value.ndim == 1:
        if key == "observation.device_capture_timestamp":
            entity_paths = build_device_capture_timestamp_entity_paths(feature_names, len(value))
        else:
            entity_paths = build_scalar_entity_paths(key, feature_names, len(value))
        for entity_path, val in zip(entity_paths, value, strict=True):
            rr.log(entity_path, rr.Scalars(val.item()))
        return

    image = tensor_to_rerun_image_numpy(value)
    if image is not None:
        entity = rr.Image(image)
        if display_compressed_images and image.dtype == np.uint8:
            entity = entity.compress()
        rr.log(key, entity=entity)
        return

    rr.log(key, rr.Tensor(value.detach().cpu().numpy()))


def as_1d_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    return value


def has_ee_pose(batch: dict, ee_pose_state_indices: dict[str, int] | None = None) -> bool:
    return all(key in batch for key in (*EE_POSITION_KEYS, *EE_QUAT_KEYS)) or (
        ee_pose_state_indices is not None and OBS_STATE in batch
    )


def _to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def make_system_time_anchor(first_relative_timestamp_s: float, now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now().astimezone()
    return now - timedelta(seconds=float(first_relative_timestamp_s))


def to_system_timestamp(anchor: datetime, relative_timestamp_s: float) -> datetime:
    return anchor + timedelta(seconds=float(relative_timestamp_s))


def extract_ee_pose(
    batch: dict,
    index: int,
    ee_pose_state_indices: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if all(key in batch for key in (*EE_POSITION_KEYS, *EE_QUAT_KEYS)):
        position = np.array([_to_float(batch[key][index]) for key in EE_POSITION_KEYS], dtype=np.float32)
        quaternion = np.array([_to_float(batch[key][index]) for key in EE_QUAT_KEYS], dtype=np.float32)
        rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
        return position, rotation
    elif ee_pose_state_indices is not None and OBS_STATE in batch:
        state = batch[OBS_STATE][index]
        position = np.array([_to_float(state[ee_pose_state_indices[key]]) for key in EE_POSITION_KEYS], dtype=np.float32)
        quaternion = np.array([_to_float(state[ee_pose_state_indices[key]]) for key in EE_QUAT_KEYS], dtype=np.float32)
        rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
        return position, rotation
    else:
        raise KeyError("EE pose is unavailable in both top-level batch keys and observation.state.")


def log_ee_pose_3d(
    batch: dict,
    index: int,
    trajectory_positions: list[np.ndarray],
    axis_length: float,
    ee_pose_state_indices: dict[str, int] | None = None,
) -> None:
    position, rotation = extract_ee_pose(batch, index, ee_pose_state_indices=ee_pose_state_indices)
    trajectory_positions.append(position.copy())

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ee_pose/trajectory",
        rr.LineStrips3D(
            [np.asarray(trajectory_positions, dtype=np.float32)],
            colors=[[255, 176, 0, 255]],
            radii=[0.002],
        ),
    )
    rr.log(
        "world/ee_pose/position",
        rr.Points3D([position], colors=[[255, 176, 0, 255]], radii=[0.008]),
    )
    rr.log(
        "world/ee_pose/frame",
        rr.Transform3D(translation=position, mat3x3=rotation, axis_length=axis_length),
    )


def _rotation_from_pose7(pose: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)


def _sorted_pose_positions(pose_by_frame: dict[int, np.ndarray]) -> np.ndarray:
    poses = [pose_by_frame[frame_idx] for frame_idx in sorted(pose_by_frame)]
    return np.asarray([pose[:3] for pose in poses], dtype=np.float32)


def log_derived_pose_trajectory(trajectory: dict) -> None:
    pose_name = trajectory["pose_name"]
    state_positions = _sorted_pose_positions(trajectory["state_by_frame"])
    action_positions = _sorted_pose_positions(trajectory["action_by_frame"])

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if len(state_positions) > 0:
        rr.log(
            f"world/derived_ee_pose/{pose_name}/state_trajectory",
            rr.LineStrips3D([state_positions], colors=[[0, 180, 255, 255]], radii=[0.002]),
            static=True,
        )
    if len(action_positions) > 0:
        rr.log(
            f"world/derived_ee_pose/{pose_name}/action_target_trajectory",
            rr.LineStrips3D([action_positions], colors=[[255, 176, 0, 255]], radii=[0.002]),
            static=True,
        )


def log_derived_pose_frame(trajectory: dict, frame_index: int, axis_length: float) -> None:
    pose_name = trajectory["pose_name"]
    for label, pose_by_frame, color in (
        ("state", trajectory["state_by_frame"], [0, 180, 255, 255]),
        ("action_target", trajectory["action_by_frame"], [255, 176, 0, 255]),
    ):
        pose = pose_by_frame.get(frame_index)
        if pose is None:
            continue
        position = pose[:3]
        rotation = _rotation_from_pose7(pose)
        rr.log(
            f"world/derived_ee_pose/{pose_name}/{label}/position",
            rr.Points3D([position], colors=[color], radii=[0.008]),
        )
        rr.log(
            f"world/derived_ee_pose/{pose_name}/{label}/frame",
            rr.Transform3D(translation=position, mat3x3=rotation, axis_length=axis_length),
        )


def log_derived_pose_scalars(trajectory: dict, frame_index: int) -> None:
    pose_name = trajectory["pose_name"]
    for label, pose_by_frame in (
        ("state", trajectory["state_by_frame"]),
        ("action_target", trajectory["action_by_frame"]),
    ):
        pose = pose_by_frame.get(frame_index)
        if pose is None:
            continue
        for scalar_name, value in zip(POSE7_SCALAR_NAMES, pose, strict=True):
            rr.log(f"derived_ee_pose/{pose_name}/{label}/{scalar_name}", rr.Scalars(float(value)))


def log_derived_cube_pose_trajectories(trajectories: list[dict]) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for trajectory in trajectories:
        cube_name = trajectory["cube_name"]
        base_positions = _sorted_pose_positions(trajectory["base_by_frame"])
        if len(base_positions) > 0:
            rr.log(
                f"world/{cube_name}/cube_pose/trajectory",
                rr.LineStrips3D([base_positions], colors=[[80, 220, 120, 255]], radii=[0.0015]),
                static=True,
            )


def log_derived_cube_pose_frame(trajectories: list[dict], frame_index: int, axis_length: float) -> None:
    for trajectory in trajectories:
        cube_name = trajectory["cube_name"]
        pose = trajectory["base_by_frame"].get(frame_index)
        if pose is None:
            continue
        position = pose[:3]
        rotation = _rotation_from_pose7(pose)
        root = f"world/{cube_name}/cube_pose"
        rr.log(f"{root}/position", rr.Points3D([position], colors=[[80, 220, 120, 255]], radii=[0.006]))
        rr.log(
            f"{root}/frame",
            rr.Transform3D(translation=position, mat3x3=rotation, axis_length=axis_length),
        )


def build_ee_axis_ruler_strips(
    axis: str,
    length: float,
    *,
    minor_tick_step: float = 0.01,
    major_tick_step: float = 0.05,
    minor_tick_height: float = 0.005,
    major_tick_height: float = 0.01,
) -> list[np.ndarray]:
    if length <= 0:
        return []
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported ruler axis: {axis}")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    tick_axis_index = {"x": 1, "y": 2, "z": 0}[axis]

    ruler_end = np.zeros(3, dtype=np.float32)
    ruler_end[axis_index] = length
    strips = [
        np.array(
            [
                np.zeros(3, dtype=np.float32),
                ruler_end,
            ],
            dtype=np.float32,
        )
    ]

    tick_positions = np.arange(0.0, length + minor_tick_step * 0.5, minor_tick_step, dtype=np.float32)
    for tick_position in tick_positions:
        is_major_tick = np.isclose(np.mod(tick_position, major_tick_step), 0.0, atol=1e-6) or np.isclose(
            np.mod(tick_position, major_tick_step), major_tick_step, atol=1e-6
        )
        tick_height = major_tick_height if is_major_tick else minor_tick_height
        tick_start = np.zeros(3, dtype=np.float32)
        tick_end = np.zeros(3, dtype=np.float32)
        tick_start[axis_index] = tick_position
        tick_end[axis_index] = tick_position
        tick_end[tick_axis_index] = tick_height
        strips.append(
            np.array(
                [
                    tick_start,
                    tick_end,
                ],
                dtype=np.float32,
            )
        )

    return strips


def log_ee_ruler(length: float) -> None:
    strips = []
    colors = []
    for axis, color in EE_RULER_AXIS_COLORS.items():
        axis_strips = build_ee_axis_ruler_strips(axis, length)
        strips.extend(axis_strips)
        colors.extend([color] * len(axis_strips))

    if not strips:
        return

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ruler",
        rr.LineStrips3D(
            strips,
            colors=colors,
            radii=[0.0015] * len(strips),
        ),
        static=True,
    )


def should_enable_episode_switch(mode: str, save: bool) -> bool:
    return mode == "distant" and not save


def format_host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def build_public_rerun_connect_url(
    *,
    grpc_port: int,
    server_uri: str,
    public_host: str | None = None,
) -> str:
    if public_host is None:
        return server_uri
    return f"rerun+http://{format_host_for_url(public_host)}:{grpc_port}/proxy"


def build_public_rerun_web_viewer_url(
    *,
    web_port: int,
    connect_url: str,
    public_host: str,
) -> str:
    return f"http://{format_host_for_url(public_host)}:{web_port}?url={quote(connect_url, safe='')}"


def build_episode_switch_visualize_kwargs(cli_kwargs: dict) -> dict:
    visualize_kwargs = cli_kwargs.copy()
    visualize_kwargs.pop("episode_index")
    return visualize_kwargs


def build_episode_process_visualize_kwargs(visualize_kwargs: dict, *, rerun_recording_id: str) -> dict:
    visualize_kwargs = visualize_kwargs.copy()
    visualize_kwargs["rerun_recording_id"] = rerun_recording_id
    return visualize_kwargs


def normalize_control_command(raw_command: str) -> str | None:
    normalized = raw_command.strip().lower()
    if not normalized:
        return None
    first = normalized[0]
    if first == "n":
        return "n"
    if first == "q":
        return "q"
    return None


def enqueue_control_command(command_queue: queue.Queue[str], raw_command: str) -> str | None:
    command = normalize_control_command(raw_command)
    if command is not None:
        command_queue.put(command)
    return command


def get_next_episode_index(current_episode_index: int, total_episodes: int) -> int | None:
    next_episode_index = current_episode_index + 1
    if next_episode_index >= total_episodes:
        return None
    return next_episode_index


@contextlib.contextmanager
def raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_episode_switch_command() -> str:
    if sys.stdin.isatty():
        with raw_terminal_mode():
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], None)
                if ready:
                    return sys.stdin.read(1).lower()

    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    stripped = line.strip().lower()
    return stripped[:1] if stripped else ""


class EpisodeControlTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, command_queue: queue.Queue[str]):
        self.command_queue = command_queue
        super().__init__(server_address, request_handler_class)


class EpisodeControlTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: EpisodeControlTCPServer = self.server  # type: ignore[assignment]
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        logging.info("Dataset viz control client connected: %s", client)
        try:
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                enqueue_control_command(server.command_queue, line.decode("utf-8", errors="ignore"))
        finally:
            logging.info("Dataset viz control client disconnected: %s", client)


def start_control_server(host: str, port: int) -> tuple[callable, callable, int]:
    command_queue: queue.Queue[str] = queue.Queue()
    server = EpisodeControlTCPServer((host, port), EpisodeControlTCPHandler, command_queue)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def read_command() -> str:
        return command_queue.get()

    def shutdown() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    return read_command, shutdown, server.server_address[1]


def terminate_episode_process(process: mp.Process | None, timeout_s: float = 5.0) -> None:
    if process is None:
        return
    if not process.is_alive():
        process.join(timeout=timeout_s)
        return

    process.terminate()
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.kill()
        process.join(timeout=timeout_s)


def is_tcp_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_tcp_port_available(
    host: str,
    port: int,
    *,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.05,
    label: str | None = None,
) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if is_tcp_port_available(host, port):
            return
        time.sleep(poll_interval_s)

    port_label = label or f"{host}:{port}"
    raise RuntimeError(
        f"Port {port_label} is still in use. Stop the previous dataset_viz/rerun process or choose another port."
    )


def create_episode_process(
    ctx: mp.context.BaseContext,
    *,
    repo_id: str,
    root: Path | None,
    episode_index: int,
    tolerance_s: float,
    visualize_kwargs: dict,
) -> mp.Process:
    return ctx.Process(
        target=serve_dataset_episode,
        kwargs={
            "repo_id": repo_id,
            "root": root,
            "episode_index": episode_index,
            "tolerance_s": tolerance_s,
            "visualize_kwargs": visualize_kwargs,
        },
        daemon=False,
    )


def run_episode_switch_loop(
    *,
    start_episode_index: int,
    total_episodes: int,
    launch_episode,
    terminate_episode,
    read_command,
) -> None:
    current_episode_index = start_episode_index
    process = launch_episode(current_episode_index)
    try:
        while True:
            command = read_command()
            if not command:
                continue
            if command in QUIT_COMMANDS:
                break
            if command != "n":
                continue

            next_episode_index = get_next_episode_index(current_episode_index, total_episodes)
            if next_episode_index is None:
                logging.info(
                    "Episode %d is the last available episode (%d total). Ignoring 'n'.",
                    current_episode_index,
                    total_episodes,
                )
                continue

            logging.info("Switching from episode %d to episode %d.", current_episode_index, next_episode_index)
            terminate_episode(process)
            process = launch_episode(next_episode_index)
            current_episode_index = next_episode_index
    finally:
        terminate_episode(process)


def serve_dataset_episode(
    *,
    repo_id: str,
    root: Path | None,
    episode_index: int,
    tolerance_s: float,
    visualize_kwargs: dict,
) -> None:
    init_logging()
    logging.info("Loading dataset for episode %d", episode_index)
    dataset = LeRobotDataset(repo_id, episodes=[episode_index], root=root, tolerance_s=tolerance_s)
    try:
        visualize_dataset(dataset, episode_index=episode_index, **visualize_kwargs)
    finally:
        del dataset
        gc.collect()
        rr.disconnect()


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    public_host: str | None = None,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    ee_axis_length: float = 0.05,
    ee_ruler_length: float = 0.1,
    raw_episode_videos: bool = True,
    raw_video_keys: str | None = None,
    raw_video_frame_stride: int = 1,
    derived_sidecar_poses: bool = True,
    derived_cube_poses: bool = True,
    derived_name: str | None = None,
    derived_pose_name: str | None = None,
    derived_cube_name: str | None = None,
    rerun_recording_id: str | None = None,
    **kwargs,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_without_none_features,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", recording_id=rerun_recording_id, spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        viewer_connect_uri = build_public_rerun_connect_url(
            grpc_port=grpc_port,
            server_uri=server_uri,
            public_host=public_host,
        )
        if viewer_connect_uri != server_uri:
            logging.info("Using public rerun connect URL: %s", viewer_connect_uri)
        rr.serve_web_viewer(open_browser=False, web_port=web_port)
        if public_host is not None:
            logging.info(
                "Open remote web viewer: %s",
                build_public_rerun_web_viewer_url(
                    web_port=web_port,
                    connect_url=viewer_connect_uri,
                    public_host=public_host,
                ),
            )

    logging.info("Logging to Rerun")
    logging.info("World frame convention: +X forward, +Y left, +Z up (URDF base/world frame).")
    log_ee_ruler(ee_ruler_length)

    raw_video_readers = None
    raw_video_paths = {}
    if raw_episode_videos and len(dataset.meta.camera_keys) == 0:
        raw_video_paths = filter_raw_episode_video_paths(
            find_raw_episode_video_paths(dataset.root, episode_index),
            raw_video_keys,
        )
        if raw_video_paths:
            raw_video_readers = RawEpisodeVideoReaders(raw_video_paths)
            if raw_video_readers.keys:
                logging.info(
                    "Logging raw episode videos from %s for cameras: %s (frame_stride=%d)",
                    dataset.root / "episodes" / f"episode_{episode_index:06d}",
                    ", ".join(raw_video_readers.keys),
                    raw_video_frame_stride,
                )

    derived_pose_trajectory = None
    if derived_sidecar_poses:
        derived_pose_trajectory = load_derived_pose_trajectory(
            dataset.root,
            episode_index=episode_index,
            derived_name=derived_name,
            pose_name=derived_pose_name,
        )
        if derived_pose_trajectory is not None:
            logging.info(
                "Logging derived EE pose trajectory '%s' from %s.",
                derived_pose_trajectory["pose_name"],
                derived_pose_trajectory["path"],
            )
            log_derived_pose_trajectory(derived_pose_trajectory)
        else:
            logging.info("No derived EE pose sidecar trajectory found for episode %d.", episode_index)

    derived_cube_pose_trajectories = []
    if derived_cube_poses:
        derived_cube_pose_sources = load_derived_cube_pose_trajectories(
            dataset.root,
            episode_index=episode_index,
            derived_name=derived_name,
            cube_name=derived_cube_name,
        )
        derived_cube_pose_trajectories = aggregate_cube_base_trajectories(derived_cube_pose_sources)
        if derived_cube_pose_trajectories:
            logging.info("Logging %d derived cube pose trajectory/trajectories in world frame.", len(derived_cube_pose_trajectories))
            log_derived_cube_pose_trajectories(derived_cube_pose_trajectories)
        else:
            logging.info("No non-empty derived cube pose sidecar trajectories found for episode %d.", episode_index)

    log_episode_frame_counts(
        episode_index=episode_index,
        lowdim_frame_count=len(dataset),
        raw_video_paths=raw_video_paths,
        derived_pose_trajectory=derived_pose_trajectory,
        derived_cube_pose_trajectories=derived_cube_pose_trajectories,
    )

    first_index = None
    system_time_anchor = None
    ee_trajectory_positions: list[np.ndarray] = []
    state_feature_names = dataset.meta.features.get(OBS_STATE, {}).get("names")
    action_feature_names = dataset.meta.features.get(ACTION, {}).get("names")
    ee_pose_state_indices = get_ee_pose_state_indices(state_feature_names)
    state_name_paths = flatten_feature_name_paths(state_feature_names)
    action_name_paths = flatten_feature_name_paths(action_feature_names)
    action_root = ACTION
    if state_name_paths is not None and action_name_paths == state_name_paths:
        action_root = "action_target"
        logging.info(
            "Action and observation.state share identical feature names; logging actions under '%s/'.",
            action_root,
        )
    action_entity_paths = None
    state_entity_paths = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()
        if system_time_anchor is None:
            system_time_anchor = make_system_time_anchor(batch["timestamp"][0].item())
            logging.info("Rerun timestamp timeline anchored to system time at %s.", system_time_anchor.isoformat())
        # iterate over the batch
        for i in range(len(batch["index"])):
            relative_timestamp_s = batch["timestamp"][i].item()
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("episode_time", duration=relative_timestamp_s)
            rr.set_time("timestamp", timestamp=to_system_timestamp(system_time_anchor, relative_timestamp_s))

            # display each camera image
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                rr.log(key, entity=img_entity)

            if raw_video_readers is not None:
                frame_index = batch["frame_index"][i].item()
                if frame_index % raw_video_frame_stride == 0:
                    for key in raw_video_readers.keys:
                        img = raw_video_readers.read_frame(key, frame_index)
                        if img is None:
                            continue
                        img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                        rr.log(f"raw_video/{key}", entity=img_entity)

            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                action_value = as_1d_tensor(batch[ACTION][i])
                if action_entity_paths is None:
                    action_entity_paths = build_scalar_entity_paths(action_root, action_feature_names, len(action_value))
                for entity_path, val in zip(action_entity_paths, action_value, strict=True):
                    rr.log(entity_path, rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                state_value = as_1d_tensor(batch[OBS_STATE][i])
                if state_entity_paths is None:
                    state_entity_paths = build_scalar_entity_paths("state", state_feature_names, len(state_value))
                for entity_path, val in zip(state_entity_paths, state_value, strict=True):
                    rr.log(entity_path, rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

            for key in get_auto_visualization_keys(batch, dataset.meta.camera_keys):
                feature_names = dataset.meta.features.get(key, {}).get("names")
                log_feature_value(key, batch[key][i], feature_names, display_compressed_images)

            if has_ee_pose(batch, ee_pose_state_indices=ee_pose_state_indices):
                log_ee_pose_3d(
                    batch,
                    i,
                    ee_trajectory_positions,
                    axis_length=ee_axis_length,
                    ee_pose_state_indices=ee_pose_state_indices,
                )

            if derived_pose_trajectory is not None:
                frame_index = batch["frame_index"][i].item()
                log_derived_pose_frame(derived_pose_trajectory, frame_index, axis_length=ee_axis_length)
                log_derived_pose_scalars(derived_pose_trajectory, frame_index)

            if derived_cube_pose_trajectories:
                frame_index = batch["frame_index"][i].item()
                log_derived_cube_pose_frame(derived_cube_pose_trajectories, frame_index, axis_length=ee_axis_length)

    if raw_video_readers is not None:
        raw_video_readers.close()

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help=(
            "Episode to visualize initially. In `--mode distant` without `--save`, "
            "press `n`/`q` in the remote terminal by default, or use `dataset_viz_client.py` "
            "from a local terminal when `--control-port` is set."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine. "
            "In distant mode without `--save`, switch episodes either from the remote terminal with `n`/`q`, "
            "or from a local terminal through `dataset_viz_client.py` when `--control-port` is set."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        help="deprecated, please use --grpc-port instead.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--public-host",
        type=str,
        default=None,
        help=(
            "Optional public hostname or IP advertised to the remote web viewer. "
            "Use this in `--mode distant` when the browser cannot reach the host through 127.0.0.1. "
            "The script will print a full browser URL with `?url=...` for the remote web viewer. "
            "Example: `--public-host 192.168.1.200`."
        ),
    )
    parser.add_argument(
        "--control-host",
        type=str,
        default="0.0.0.0",
        help=(
            "Host interface for the optional TCP control server. "
            "Use together with `--control-port`, then connect from a local terminal with `dataset_viz_client.py`."
        ),
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help=(
            "Optional TCP control port for remote episode switching. "
            "When set in `--mode distant`, the script listens for `n`/`q` commands from `dataset_viz_client.py` "
            "instead of requiring the remote terminal."
        ),
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed images in Rerun instead of uncompressed ones.",
    )
    parser.add_argument(
        "--ee-axis-length",
        type=float,
        default=0.05,
        help="Axis length in meters for the 3D end-effector pose visualization.",
    )
    parser.add_argument(
        "--ee-ruler-length",
        type=float,
        default=0.1,
        help="Length in meters of the static 3D ruler drawn in the world frame. Set to 0 to disable it.",
    )
    parser.add_argument(
        "--no-raw-episode-videos",
        dest="raw_episode_videos",
        action="store_false",
        help=(
            "Disable fallback logging of raw videos from <root>/episodes/episode_XXXX/cam_*.mp4|mkv "
            "when the dataset metadata has no camera/video features."
        ),
    )
    parser.set_defaults(raw_episode_videos=True)
    parser.add_argument(
        "--raw-video-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated raw episode camera keys to log, e.g. cam_02,cam_06. "
            "Defaults to all videos found under <root>/episodes/episode_XXXX."
        ),
    )
    parser.add_argument(
        "--raw-video-frame-stride",
        type=int,
        default=1,
        help="Log every Nth raw video frame. Use a larger value to reduce .rrd size and viewer load.",
    )
    parser.add_argument(
        "--no-derived-sidecar-poses",
        dest="derived_sidecar_poses",
        action="store_false",
        help="Disable fallback logging of EE pose trajectories from <root>/derived/*/state_action*.csv.",
    )
    parser.set_defaults(derived_sidecar_poses=True)
    parser.add_argument(
        "--no-derived-cube-poses",
        dest="derived_cube_poses",
        action="store_false",
        help="Disable fallback logging of non-empty cube pose trajectories from <root>/derived/*/cube_pose.*.*.csv.",
    )
    parser.set_defaults(derived_cube_poses=True)
    parser.add_argument(
        "--derived-name",
        type=str,
        default=None,
        help="Optional derived sidecar directory name under <root>/derived to use for EE pose trajectories.",
    )
    parser.add_argument(
        "--derived-pose-name",
        type=str,
        default=None,
        help="Optional pose name for state_action.<name>.csv, e.g. left, right, or head. Defaults to the first valid sidecar.",
    )
    parser.add_argument(
        "--derived-cube-name",
        type=str,
        default=None,
        help="Optional cube name for cube_pose.<cube>.<camera>.csv, e.g. left, right, or head. Defaults to all non-empty cube pose sidecars.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    if kwargs["ws_port"] is not None:
        logging.warning(
            "--ws-port is deprecated and will be removed in future versions. Please use --grpc-port instead."
        )
        logging.warning("Setting grpc_port to ws_port value.")
        kwargs["grpc_port"] = kwargs.pop("ws_port")
    if kwargs["raw_video_frame_stride"] < 1:
        raise ValueError("--raw-video-frame-stride must be >= 1")

    init_logging()
    if should_enable_episode_switch(args.mode, bool(args.save)):
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        total_episodes = meta.total_episodes
        del meta
        gc.collect()

        if args.episode_index >= total_episodes:
            raise IndexError(f"Episode index {args.episode_index} out of range for dataset with {total_episodes} episodes.")

        if args.control_port is not None:
            wait_for_tcp_port_available(
                args.control_host,
                args.control_port,
                label=f"control server {args.control_host}:{args.control_port}",
            )

        visualize_kwargs = build_episode_switch_visualize_kwargs(kwargs)

        ctx = mp.get_context("spawn")

        def launch_episode(episode_index: int) -> mp.Process:
            episode_visualize_kwargs = build_episode_process_visualize_kwargs(
                visualize_kwargs,
                rerun_recording_id=str(uuid4()),
            )
            wait_for_tcp_port_available("0.0.0.0", visualize_kwargs["grpc_port"], label=f"gRPC {visualize_kwargs['grpc_port']}")
            wait_for_tcp_port_available("0.0.0.0", visualize_kwargs["web_port"], label=f"web viewer {visualize_kwargs['web_port']}")
            process = create_episode_process(
                ctx,
                repo_id=repo_id,
                root=root,
                episode_index=episode_index,
                tolerance_s=tolerance_s,
                visualize_kwargs=episode_visualize_kwargs,
            )
            process.start()
            return process

        if args.control_port is not None:
            read_command, shutdown_control, control_port = start_control_server(args.control_host, args.control_port)
            logging.info(
                "Interactive episode switching enabled through TCP control server on %s:%d.",
                args.control_host,
                control_port,
            )
            logging.info("Use dataset_viz_client.py locally and send `n` / `q` commands over the control port.")
        else:
            logging.info(
                "Interactive episode switching enabled for distant mode. Press 'n' for next episode, 'q' to quit."
            )
            read_command = read_episode_switch_command
            shutdown_control = lambda: None

        try:
            run_episode_switch_loop(
                start_episode_index=args.episode_index,
                total_episodes=total_episodes,
                launch_episode=launch_episode,
                terminate_episode=terminate_episode_process,
                read_command=read_command,
            )
        finally:
            shutdown_control()
        return

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)
    try:
        visualize_dataset(dataset, **vars(args))
    finally:
        del dataset
        gc.collect()
        rr.disconnect()


if __name__ == "__main__":
    main()
