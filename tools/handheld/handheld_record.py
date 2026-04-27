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

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterator

import numpy as np

from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.cameras.hikrobot.configuration_hikrobot import HikrobotCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.handheld_gripper.configs import HandheldGripperConfig
from lerobot.handheld_gripper.pika_sense.configuration_pika_sense import PikaSenseConfig  # noqa: F401
from lerobot.handheld_gripper.utils import make_handheld_grippers_from_configs
from lerobot.tactiles.configs import TactileConfig
from lerobot.tactiles.paxini_gen2 import PaxiniGen2OmegaTactile, PaxiniGen2OmegaTactileConfig  # noqa: F401
from lerobot.utils.import_utils import make_device_from_device_class
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

HANDHELD_TACTILE_WIDTH = 12
HANDHELD_TACTILE_HEIGHT = 10
HANDHELD_TACTILE_SIDE_NAMES = ("left", "right")
HANDHELD_TACTILE_DIMENSION_LABELS = ("x", "y", "z")
HANDHELD_TACTILE_SIDE_TAXELS = HANDHELD_TACTILE_WIDTH * HANDHELD_TACTILE_HEIGHT
HANDHELD_TACTILE_NUM_SIDES = len(HANDHELD_TACTILE_SIDE_NAMES)
HANDHELD_SOFT_SYNC_FEATURE_NAMES = (
    "target_timestamp_s",
    "max_skew_s",
    "oldest_device_lag_s",
    "global_lag_s",
    "timed_out",
)


@dataclass
class HandheldSoftSyncConfig:
    enabled: bool = True
    tolerance_ms: float = 20.0
    wait_timeout_ms: float = 150.0
    poll_interval_ms: float = 1.0
    buffer_duration_s: float = 0.25

    def __post_init__(self) -> None:
        if self.tolerance_ms < 0:
            raise ValueError(
                f"`sensors.soft_sync.tolerance_ms` must be >= 0, but {self.tolerance_ms} is provided."
            )
        if self.wait_timeout_ms < 0:
            raise ValueError(
                f"`sensors.soft_sync.wait_timeout_ms` must be >= 0, but {self.wait_timeout_ms} is provided."
            )
        if self.poll_interval_ms <= 0:
            raise ValueError(
                f"`sensors.soft_sync.poll_interval_ms` must be > 0, but {self.poll_interval_ms} is provided."
            )
        if self.buffer_duration_s <= 0:
            raise ValueError(
                "`sensors.soft_sync.buffer_duration_s` must be > 0, "
                f"but {self.buffer_duration_s} is provided."
            )


@dataclass(frozen=True)
class SoftSyncResult:
    target_timestamp_s: float
    max_skew_s: float
    oldest_device_lag_s: float
    timed_out: bool


@dataclass(frozen=True)
class TimestampedSample:
    timestamp_s: float
    value: Any


@dataclass(frozen=True)
class SoftSyncFrameSelection:
    result: SoftSyncResult
    samples: dict[str, TimestampedSample]


@dataclass(frozen=True)
class CapturedEpisodeFrame:
    frame_index: int
    dataset_timestamp_s: float
    frame: dict[str, Any]


@dataclass(frozen=True)
class EpisodeCaptureComplete:
    result: EpisodeRecordResult | None = None
    error: BaseException | None = None


@dataclass
class HandheldDatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: float = 10.0
    num_episodes: int = 0
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    capture_queue_size: int = 16

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("`dataset.repo_id` must be a non-empty string.")
        if not self.single_task:
            raise ValueError("`dataset.single_task` must be a non-empty string.")
        if self.fps <= 0:
            raise ValueError(f"`dataset.fps` must be > 0, but {self.fps} is provided.")
        if self.episode_time_s <= 0:
            raise ValueError(f"`dataset.episode_time_s` must be > 0, but {self.episode_time_s} is provided.")
        if self.num_episodes < 0:
            raise ValueError(f"`dataset.num_episodes` must be >= 0, but {self.num_episodes} is provided.")
        if self.capture_queue_size <= 0:
            raise ValueError(
                f"`dataset.capture_queue_size` must be > 0, but {self.capture_queue_size} is provided."
            )


@dataclass
class HandheldSensorsConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    tactiles: dict[str, TactileConfig] = field(default_factory=dict)
    handheld_grippers: dict[str, HandheldGripperConfig] = field(default_factory=dict)
    max_read_age_ms: int = 500
    soft_sync: HandheldSoftSyncConfig = field(default_factory=HandheldSoftSyncConfig)

    def __post_init__(self) -> None:
        if not self.cameras and not self.tactiles and not self.handheld_grippers:
            raise ValueError("At least one device must be configured in `sensors`.")
        if self.max_read_age_ms <= 0:
            raise ValueError(
                f"`sensors.max_read_age_ms` must be > 0, but {self.max_read_age_ms} is provided."
            )


@dataclass
class HandheldRecordingConfig:
    sensors: HandheldSensorsConfig
    dataset: HandheldDatasetConfig
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    rerun_save_path: str | Path | None = None
    play_sounds: bool = False
    resume: bool = False
    robot_type: str | None = "handheld_capture"


class EpisodeStopAction(str, Enum):
    TIMED_OUT = "timed_out"
    SAVE_EARLY = "save_early"
    DISCARD_EARLY = "discard_early"
    EXIT = "exit"


@dataclass(frozen=True)
class EpisodeRecordResult:
    recorded_frames: int
    stop_action: EpisodeStopAction


def _make_tactiles_from_configs(tactile_configs: dict[str, TactileConfig]) -> dict[str, Any]:
    tactiles: dict[str, Any] = {}
    for key, cfg in tactile_configs.items():
        if cfg.type == "paxini_gen2_omega":
            tactiles[key] = PaxiniGen2OmegaTactile(cfg)
        else:
            tactiles[key] = make_device_from_device_class(cfg)
    return tactiles


def _validate_handheld_tactile_layout(tactile_name: str, tactile: Any) -> None:
    num_taxels = int(tactile.num_taxels or 0)
    num_dimensions = int(tactile.num_dimensions or 0)
    expected_taxels = HANDHELD_TACTILE_SIDE_TAXELS * HANDHELD_TACTILE_NUM_SIDES
    expected_dimensions = len(HANDHELD_TACTILE_DIMENSION_LABELS)
    if num_taxels != expected_taxels or num_dimensions != expected_dimensions:
        raise ValueError(
            f"Tactile '{tactile_name}' must report shape ({expected_taxels}, {expected_dimensions}) "
            f"for handheld recording, but got ({num_taxels}, {num_dimensions})."
        )


def _build_handheld_tactile_feature_names(tactile_name: str) -> dict[str, dict[str, Any]]:
    height = HANDHELD_TACTILE_HEIGHT
    width = HANDHELD_TACTILE_WIDTH
    side_taxels = HANDHELD_TACTILE_SIDE_TAXELS
    return {
        f"observation.tactile.{tactile_name}.left_xyz": {
            "dtype": "float32",
            "shape": (len(HANDHELD_TACTILE_DIMENSION_LABELS), height, width),
            "names": ["channels", "height", "width"],
        },
        f"observation.tactile.{tactile_name}.right_xyz": {
            "dtype": "float32",
            "shape": (len(HANDHELD_TACTILE_DIMENSION_LABELS), height, width),
            "names": ["channels", "height", "width"],
        },
        f"observation.tactile.{tactile_name}.left_magnitude": {
            "dtype": "float32",
            "shape": (height, width),
            "names": ["height", "width"],
        },
        f"observation.tactile.{tactile_name}.right_magnitude": {
            "dtype": "float32",
            "shape": (height, width),
            "names": ["height", "width"],
        },
        f"observation.tactile.{tactile_name}.raw_xyz": {
            "dtype": "float32",
            "shape": (HANDHELD_TACTILE_NUM_SIDES, side_taxels, len(HANDHELD_TACTILE_DIMENSION_LABELS)),
            "names": ["side", "taxel", "channels"],
        },
    }


def _reshape_handheld_tactile_side(side_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side_hwc = np.asarray(side_frame, dtype=np.float32).reshape(
        HANDHELD_TACTILE_HEIGHT,
        HANDHELD_TACTILE_WIDTH,
        len(HANDHELD_TACTILE_DIMENSION_LABELS),
    )
    side_chw = np.ascontiguousarray(np.transpose(side_hwc, (2, 0, 1)))
    magnitude = np.linalg.norm(side_hwc, axis=-1).astype(np.float32, copy=False)
    return side_chw, magnitude


def _build_handheld_tactile_observation(tactile_name: str, tactile_frame: np.ndarray) -> dict[str, np.ndarray]:
    raw_frame = np.asarray(tactile_frame, dtype=np.float32)
    expected_shape = (
        HANDHELD_TACTILE_SIDE_TAXELS * HANDHELD_TACTILE_NUM_SIDES,
        len(HANDHELD_TACTILE_DIMENSION_LABELS),
    )
    if raw_frame.shape != expected_shape:
        raise ValueError(
            f"Tactile '{tactile_name}' frame must have shape {expected_shape}, but got {tuple(raw_frame.shape)}."
        )

    left_raw, right_raw = np.split(raw_frame, HANDHELD_TACTILE_NUM_SIDES, axis=0)
    left_xyz, left_magnitude = _reshape_handheld_tactile_side(left_raw)
    right_xyz, right_magnitude = _reshape_handheld_tactile_side(right_raw)
    return {
        f"observation.tactile.{tactile_name}.left_xyz": left_xyz,
        f"observation.tactile.{tactile_name}.right_xyz": right_xyz,
        f"observation.tactile.{tactile_name}.left_magnitude": left_magnitude,
        f"observation.tactile.{tactile_name}.right_magnitude": right_magnitude,
        f"observation.tactile.{tactile_name}.raw_xyz": np.stack((left_raw, right_raw)).astype(np.float32, copy=False),
    }


def _build_state_feature_names(
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
) -> list[str]:
    names: list[str] = []
    for gripper_name in handheld_grippers:
        names.append(f"handheld_gripper.{gripper_name}.width_mm")
    return names


def _build_capture_timestamp_feature_names(
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
) -> list[str]:
    names: list[str] = []
    for camera_name in cameras:
        names.append(f"camera.{camera_name}.capture_timestamp_s")
    for tactile_name in tactiles:
        names.append(f"tactile.{tactile_name}.capture_timestamp_s")
    for gripper_name in handheld_grippers:
        names.append(f"handheld_gripper.{gripper_name}.capture_timestamp_s")
    return names


def build_dataset_features(
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    *,
    use_videos: bool,
    include_soft_sync_diagnostics: bool = True,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}

    state_names = _build_state_feature_names(tactiles, handheld_grippers)
    if state_names:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    capture_timestamp_names = _build_capture_timestamp_feature_names(cameras, tactiles, handheld_grippers)
    if capture_timestamp_names:
        features["observation.device_capture_timestamp"] = {
            "dtype": "float64",
            "shape": (len(capture_timestamp_names),),
            "names": capture_timestamp_names,
        }
        if include_soft_sync_diagnostics:
            features["observation.soft_sync"] = {
                "dtype": "float64",
                "shape": (len(HANDHELD_SOFT_SYNC_FEATURE_NAMES),),
                "names": list(HANDHELD_SOFT_SYNC_FEATURE_NAMES),
            }

    for tactile_name, tactile in tactiles.items():
        _validate_handheld_tactile_layout(tactile_name, tactile)
        features.update(_build_handheld_tactile_feature_names(tactile_name))

    for camera_name, camera in cameras.items():
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


def _get_locked_latest_value(device: Any) -> tuple[Any, float] | None:
    lock = getattr(device, "frame_lock", None) or getattr(device, "read_lock", None)
    if lock is None:
        return None

    with lock:
        timestamp = getattr(device, "latest_timestamp", None)
        if timestamp is None:
            return None

        for attr_name in ("latest_frame", "latest_color_frame", "latest_width_mm"):
            value = getattr(device, attr_name, None)
            if value is not None:
                return value, float(timestamp)

    return None


def _read_timestamped_latest(
    device: Any,
    *,
    max_read_age_ms: int,
) -> TimestampedSample:
    locked_value = _get_locked_latest_value(device)
    if locked_value is not None:
        value, timestamp = locked_value
        age_ms = (time.perf_counter() - timestamp) * 1000
        if age_ms > max_read_age_ms:
            raise TimeoutError(
                f"{type(device).__name__} latest sample is stale ({age_ms:.1f} ms > {max_read_age_ms} ms)."
            )
        return TimestampedSample(timestamp_s=timestamp, value=value)

    value = device.read_latest(max_age_ms=max_read_age_ms)
    return TimestampedSample(timestamp_s=_get_latest_timestamp(device), value=value)


class TimestampedSampleBuffer:
    def __init__(
        self,
        *,
        device_name: str,
        device: Any,
        max_read_age_ms: int,
        buffer_duration_s: float,
        poll_interval_s: float,
    ) -> None:
        self.device_name = device_name
        self.device = device
        self.max_read_age_ms = max_read_age_ms
        self.buffer_duration_s = buffer_duration_s
        self.poll_interval_s = poll_interval_s
        self._samples: deque[TimestampedSample] = deque()
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_logged_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name=f"soft-sync-buffer-{self.device_name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest(self) -> TimestampedSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def nearest(
        self,
        target_timestamp_s: float,
        *,
        min_timestamp_s: float | None = None,
    ) -> TimestampedSample | None:
        with self._lock:
            candidates = list(self._samples)

        if min_timestamp_s is not None:
            candidates = [sample for sample in candidates if sample.timestamp_s >= min_timestamp_s]
        if not candidates:
            return None

        return min(candidates, key=lambda sample: abs(sample.timestamp_s - target_timestamp_s))

    def _append(self, sample: TimestampedSample) -> None:
        with self._lock:
            if self._samples and sample.timestamp_s <= self._samples[-1].timestamp_s:
                return

            self._samples.append(sample)
            min_kept_timestamp_s = sample.timestamp_s - self.buffer_duration_s
            while self._samples and self._samples[0].timestamp_s < min_kept_timestamp_s:
                self._samples.popleft()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._append(_read_timestamped_latest(self.device, max_read_age_ms=self.max_read_age_ms))
                self._last_logged_error = None
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                if error_text != self._last_logged_error:
                    logging.debug(
                        "Soft sync buffer for %s has no fresh sample yet: %s",
                        self.device_name,
                        exc,
                    )
                    self._last_logged_error = error_text

            self._stop_event.wait(self.poll_interval_s)


def _iter_capture_devices(
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
) -> Iterator[tuple[str, Any]]:
    for camera_name, camera in cameras.items():
        yield f"camera.{camera_name}", camera
    for tactile_name, tactile in tactiles.items():
        yield f"tactile.{tactile_name}", tactile
    for gripper_name, gripper in handheld_grippers.items():
        yield f"handheld_gripper.{gripper_name}", gripper


def _make_soft_sync_sample_buffers(
    *,
    cfg: HandheldSensorsConfig,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
) -> dict[str, TimestampedSampleBuffer]:
    if not cfg.soft_sync.enabled:
        return {}

    return {
        device_name: TimestampedSampleBuffer(
            device_name=device_name,
            device=device,
            max_read_age_ms=cfg.max_read_age_ms,
            buffer_duration_s=cfg.soft_sync.buffer_duration_s,
            poll_interval_s=cfg.soft_sync.poll_interval_ms / 1000.0,
        )
        for device_name, device in _iter_capture_devices(cameras, tactiles, handheld_grippers)
    }


def _start_soft_sync_sample_buffers(buffers: dict[str, TimestampedSampleBuffer]) -> None:
    for buffer in buffers.values():
        buffer.start()


def _stop_soft_sync_sample_buffers(buffers: dict[str, TimestampedSampleBuffer]) -> None:
    for buffer in buffers.values():
        buffer.stop()


def _compute_soft_sync_result(
    *,
    target_capture_time_s: float,
    episode_start_time_s: float,
    timestamps: list[float],
    timed_out: bool,
) -> SoftSyncResult:
    target_timestamp_s = target_capture_time_s - episode_start_time_s
    if not timestamps:
        return SoftSyncResult(
            target_timestamp_s=target_timestamp_s,
            max_skew_s=0.0,
            oldest_device_lag_s=0.0,
            timed_out=timed_out,
        )

    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    return SoftSyncResult(
        target_timestamp_s=target_timestamp_s,
        max_skew_s=max_timestamp - min_timestamp,
        oldest_device_lag_s=max(0.0, target_capture_time_s - min_timestamp),
        timed_out=timed_out,
    )


def _soft_sync_result_to_array(
    soft_sync_result: SoftSyncResult,
    capture_timestamp_values: list[float],
) -> np.ndarray:
    if capture_timestamp_values:
        max_skew_s = max(capture_timestamp_values) - min(capture_timestamp_values)
        oldest_device_lag_s = max(0.0, soft_sync_result.target_timestamp_s - min(capture_timestamp_values))
        global_lag_s = float(np.median(np.asarray(capture_timestamp_values, dtype=np.float64)))
        global_lag_s -= soft_sync_result.target_timestamp_s
    else:
        max_skew_s = soft_sync_result.max_skew_s
        oldest_device_lag_s = soft_sync_result.oldest_device_lag_s
        global_lag_s = 0.0

    return np.asarray(
        [
            soft_sync_result.target_timestamp_s,
            max_skew_s,
            oldest_device_lag_s,
            global_lag_s,
            1.0 if soft_sync_result.timed_out else 0.0,
        ],
        dtype=np.float64,
    )


def _wait_for_soft_sync_target(
    *,
    cfg: HandheldSoftSyncConfig,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    target_capture_time_s: float,
    episode_start_time_s: float,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> SoftSyncResult:
    devices = list(_iter_capture_devices(cameras, tactiles, handheld_grippers))
    if not cfg.enabled or not devices:
        timestamps = [_get_latest_timestamp(device) for _, device in devices]
        return _compute_soft_sync_result(
            target_capture_time_s=target_capture_time_s,
            episode_start_time_s=episode_start_time_s,
            timestamps=timestamps,
            timed_out=False,
        )

    min_accepted_timestamp_s = target_capture_time_s - (cfg.tolerance_ms / 1000.0)
    deadline_s = now_fn() + (cfg.wait_timeout_ms / 1000.0)
    last_timestamps: list[float] = []
    stale_device_names: list[str] = []

    while True:
        last_timestamps = []
        stale_device_names = []
        for device_name, device in devices:
            try:
                timestamp = _get_latest_timestamp(device)
            except RuntimeError:
                stale_device_names.append(device_name)
                continue

            last_timestamps.append(timestamp)
            if timestamp < min_accepted_timestamp_s:
                stale_device_names.append(device_name)

        if not stale_device_names:
            return _compute_soft_sync_result(
                target_capture_time_s=target_capture_time_s,
                episode_start_time_s=episode_start_time_s,
                timestamps=last_timestamps,
                timed_out=False,
            )

        now_s = now_fn()
        if now_s >= deadline_s:
            logging.warning(
                "Soft sync timed out for target %.6fs; stale devices: %s",
                target_capture_time_s - episode_start_time_s,
                ", ".join(stale_device_names),
            )
            return _compute_soft_sync_result(
                target_capture_time_s=target_capture_time_s,
                episode_start_time_s=episode_start_time_s,
                timestamps=last_timestamps,
                timed_out=True,
            )

        sleep_fn(min(cfg.poll_interval_ms / 1000.0, max(0.0, deadline_s - now_s)))


def _compute_soft_sync_selection(
    *,
    target_capture_time_s: float,
    episode_start_time_s: float,
    samples: dict[str, TimestampedSample],
    timed_out: bool,
) -> SoftSyncFrameSelection:
    result = _compute_soft_sync_result(
        target_capture_time_s=target_capture_time_s,
        episode_start_time_s=episode_start_time_s,
        timestamps=[sample.timestamp_s for sample in samples.values()],
        timed_out=timed_out,
    )
    return SoftSyncFrameSelection(result=result, samples=samples)


def _wait_for_soft_sync_samples(
    *,
    cfg: HandheldSoftSyncConfig,
    buffers: dict[str, TimestampedSampleBuffer],
    target_capture_time_s: float,
    episode_start_time_s: float,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> SoftSyncFrameSelection:
    min_accepted_timestamp_s = target_capture_time_s - (cfg.tolerance_ms / 1000.0)
    deadline_s = now_fn() + (cfg.wait_timeout_ms / 1000.0)

    while True:
        selected_samples: dict[str, TimestampedSample] = {}
        missing_device_names: list[str] = []

        for device_name, buffer in buffers.items():
            sample = buffer.nearest(target_capture_time_s, min_timestamp_s=min_accepted_timestamp_s)
            if sample is None:
                missing_device_names.append(device_name)
                continue
            selected_samples[device_name] = sample

        if not missing_device_names:
            return _compute_soft_sync_selection(
                target_capture_time_s=target_capture_time_s,
                episode_start_time_s=episode_start_time_s,
                samples=selected_samples,
                timed_out=False,
            )

        now_s = now_fn()
        if now_s >= deadline_s:
            fallback_samples = {
                device_name: sample
                for device_name, buffer in buffers.items()
                if (sample := buffer.nearest(target_capture_time_s)) is not None
            }
            logging.warning(
                "Soft sync sample selection timed out for target %.6fs; missing devices: %s",
                target_capture_time_s - episode_start_time_s,
                ", ".join(missing_device_names),
            )
            return _compute_soft_sync_selection(
                target_capture_time_s=target_capture_time_s,
                episode_start_time_s=episode_start_time_s,
                samples=fallback_samples,
                timed_out=True,
            )

        sleep_fn(min(cfg.poll_interval_ms / 1000.0, max(0.0, deadline_s - now_s)))


def collect_dataset_frame(
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    *,
    max_read_age_ms: int,
    episode_start_time_s: float,
    task: str,
    soft_sync_result: SoftSyncResult | None = None,
    soft_sync_samples: dict[str, TimestampedSample] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    state_values: list[float] = []
    capture_timestamp_values: list[float] = []

    for camera_name, camera in cameras.items():
        sample = (soft_sync_samples or {}).get(f"camera.{camera_name}")
        if sample is None:
            sample = _read_timestamped_latest(camera, max_read_age_ms=max_read_age_ms)
        camera_frame = sample.value
        capture_timestamp_values.append(sample.timestamp_s - episode_start_time_s)
        frame[f"observation.images.{camera_name}"] = _normalize_camera_frame_to_rgb(camera, camera_frame)

    for tactile_name, tactile in tactiles.items():
        sample = (soft_sync_samples or {}).get(f"tactile.{tactile_name}")
        if sample is None:
            sample = _read_timestamped_latest(tactile, max_read_age_ms=max_read_age_ms)
        tactile_frame = np.asarray(sample.value, dtype=np.float32)
        capture_timestamp_values.append(sample.timestamp_s - episode_start_time_s)
        frame.update(_build_handheld_tactile_observation(tactile_name, tactile_frame))

    for gripper_name, gripper in handheld_grippers.items():
        sample = (soft_sync_samples or {}).get(f"handheld_gripper.{gripper_name}")
        if sample is None:
            sample = _read_timestamped_latest(gripper, max_read_age_ms=max_read_age_ms)
        width_mm = float(sample.value)
        capture_timestamp_values.append(sample.timestamp_s - episode_start_time_s)
        state_values.append(width_mm)

    if state_values:
        frame["observation.state"] = np.asarray(state_values, dtype=np.float32)
    if capture_timestamp_values:
        frame["observation.device_capture_timestamp"] = np.asarray(capture_timestamp_values, dtype=np.float64)
    if soft_sync_result is not None:
        frame["observation.soft_sync"] = _soft_sync_result_to_array(
            soft_sync_result,
            capture_timestamp_values,
        )
    frame["task"] = task
    return frame


def _init_rerun(cfg: HandheldRecordingConfig) -> bool:
    if not cfg.display_data and cfg.rerun_save_path is None:
        return False

    try:
        import rerun as rr
    except Exception as exc:  # noqa: BLE001
        logging.warning("Rerun is unavailable, continuing without visualization: %s", exc)
        return False

    rr.init("handheld_record", spawn=False)

    rerun_save_path = Path(cfg.rerun_save_path) if cfg.rerun_save_path is not None else None
    if rerun_save_path is not None:
        rerun_save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rerun_save_path))
        logging.info("Rerun recording will be written to %s", rerun_save_path)
        if cfg.display_data:
            logging.info("`rerun_save_path` is set, so the script will record a .rrd file instead of spawning a viewer.")
        return True

    if cfg.display_ip is not None and cfg.display_port is not None:
        rr.connect_grpc(url=f"rerun+http://{cfg.display_ip}:{cfg.display_port}/proxy")
    else:
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)

    return True


def _log_rerun_frame(
    *,
    frame_index: int,
    dataset_timestamp_s: float,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    frame: dict[str, Any],
) -> None:
    try:
        import rerun as rr
    except Exception:  # noqa: BLE001
        return

    rr.set_time("frame_index", sequence=frame_index)
    rr.set_time("timestamp", timestamp=dataset_timestamp_s)

    capture_timestamps = frame.get("observation.device_capture_timestamp")
    capture_index = 0

    for camera_name in cameras:
        rr.log(f"observation/images/{camera_name}", rr.Image(frame[f"observation.images.{camera_name}"]))
        if capture_timestamps is not None:
            rr.log(
                f"observation/device_capture_timestamp/camera/{camera_name}",
                rr.Scalars(float(capture_timestamps[capture_index])),
            )
            capture_index += 1

    for tactile_name in tactiles:
        left_xyz = np.asarray(frame[f"observation.tactile.{tactile_name}.left_xyz"], dtype=np.float32)
        right_xyz = np.asarray(frame[f"observation.tactile.{tactile_name}.right_xyz"], dtype=np.float32)
        rr.log(
            f"observation/tactile/{tactile_name}/left_xyz",
            rr.Image(np.transpose(left_xyz, (1, 2, 0))),
        )
        rr.log(
            f"observation/tactile/{tactile_name}/right_xyz",
            rr.Image(np.transpose(right_xyz, (1, 2, 0))),
        )
        rr.log(
            f"observation/tactile/{tactile_name}/left_magnitude",
            rr.Image(np.asarray(frame[f"observation.tactile.{tactile_name}.left_magnitude"], dtype=np.float32)),
        )
        rr.log(
            f"observation/tactile/{tactile_name}/right_magnitude",
            rr.Image(np.asarray(frame[f"observation.tactile.{tactile_name}.right_magnitude"], dtype=np.float32)),
        )
        rr.log(
            f"observation/tactile/{tactile_name}/raw_xyz",
            rr.Tensor(np.asarray(frame[f"observation.tactile.{tactile_name}.raw_xyz"], dtype=np.float32)),
        )
        if capture_timestamps is not None:
            rr.log(
                f"observation/device_capture_timestamp/tactile/{tactile_name}",
                rr.Scalars(float(capture_timestamps[capture_index])),
            )
            capture_index += 1

    state_vector = frame.get("observation.state")
    state_offset = 0
    if state_vector is None:
        state_vector = np.array([], dtype=np.float32)

    for gripper_name in handheld_grippers:
        if state_offset < len(state_vector):
            rr.log(
                f"observation/handheld_gripper/{gripper_name}/width_mm",
                rr.Scalars(float(state_vector[state_offset])),
            )
            state_offset += 1
        if capture_timestamps is not None:
            rr.log(
                f"observation/device_capture_timestamp/handheld_gripper/{gripper_name}",
                rr.Scalars(float(capture_timestamps[capture_index])),
            )
            capture_index += 1

    soft_sync = frame.get("observation.soft_sync")
    if soft_sync is not None:
        for name, value in zip(HANDHELD_SOFT_SYNC_FEATURE_NAMES, soft_sync, strict=True):
            rr.log(f"observation/soft_sync/{name}", rr.Scalars(float(value)))


def _format_device_summary(devices: dict[str, Any]) -> str:
    if not devices:
        return "(none)"
    return ", ".join(devices)


def _format_target_episodes(num_episodes: int) -> str:
    return "unlimited" if num_episodes == 0 else str(num_episodes)


def _print_intro(
    cfg: HandheldRecordingConfig,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    dataset_root: Path | None,
) -> None:
    print("Handheld multimodal recorder")
    print(
        "This script records Hikrobot camera streams, Paxini Gen2 tactile streams, "
        "and Pika Sense readings into LeRobot v3 format."
    )
    print(f"Dataset repo_id: {cfg.dataset.repo_id}")
    if dataset_root is not None:
        print(f"Dataset root: {dataset_root}")
    print(f"Cameras: {_format_device_summary(cameras)}")
    print(f"Tactiles: {_format_device_summary(tactiles)}")
    print(f"Handheld grippers: {_format_device_summary(handheld_grippers)}")
    print(
        f"Episode length: {cfg.dataset.episode_time_s:.2f}s | Dataset FPS: {cfg.dataset.fps} | "
        f"Target saved episodes: {_format_target_episodes(cfg.dataset.num_episodes)}"
    )
    if cfg.sensors.soft_sync.enabled:
        print(
            "Soft sync: enabled | "
            f"tolerance={cfg.sensors.soft_sync.tolerance_ms:.1f}ms | "
            f"timeout={cfg.sensors.soft_sync.wait_timeout_ms:.1f}ms"
        )
    else:
        print("Soft sync: disabled")


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


def _connect_devices(*device_groups: dict[str, Any]) -> None:
    for devices in device_groups:
        for device in devices.values():
            device.connect()


def _connect_cameras_best_effort(cameras: dict[str, Any]) -> dict[str, Any]:
    connected_cameras: dict[str, Any] = {}

    for camera_name, camera in cameras.items():
        try:
            camera.connect()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Camera '%s' failed to connect and will be skipped: %s", camera_name, exc)
            try:
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
            except Exception as disconnect_exc:  # noqa: BLE001
                logging.warning("Failed to disconnect skipped camera '%s' cleanly: %s", camera_name, disconnect_exc)
            continue

        connected_cameras[camera_name] = camera

    return connected_cameras


def _disconnect_devices(*device_groups: dict[str, Any]) -> None:
    for devices in reversed(device_groups):
        for device in devices.values():
            try:
                device.disconnect()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to disconnect %s cleanly: %s", type(device).__name__, exc)


def _confirm_keep_episode(play_sounds: bool) -> bool:
    logging.info("Waiting for terminal confirmation for the recorded episode.")
    log_say("Save episode", play_sounds)
    while True:
        try:
            response = input("Save current episode? [Y/n]: ").strip().lower()
        except EOFError:
            logging.warning("Terminal input closed while waiting for save confirmation; keeping the episode.")
            return True

        if response in ("", "y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please answer with 'Y' or 'n'.")


def _wait_for_enter(episode_attempt_index: int, play_sounds: bool) -> None:
    log_say(f"Episode {episode_attempt_index} ready", play_sounds)
    input(f"Episode {episode_attempt_index}: press Enter to start recording...")


@contextmanager
def _terminal_cbreak_mode() -> Iterator[bool]:
    if not sys.stdin.isatty():
        yield False
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        termios.tcflush(fd, termios.TCIFLUSH)


def _read_terminal_key_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None

    try:
        pressed = os.read(sys.stdin.fileno(), 1)
    except OSError:
        return None
    if not pressed:
        return None
    if pressed == b"\x1b":
        return "esc"
    return pressed.decode("utf-8", errors="ignore").lower()


def _has_episode_data(dataset: LeRobotDataset) -> bool:
    return dataset.episode_buffer is not None and int(dataset.episode_buffer["size"]) > 0


def _recording_target_reached(dataset: LeRobotDataset, target_num_episodes: int) -> bool:
    return target_num_episodes > 0 and dataset.num_episodes >= target_num_episodes


def _capture_episode_frames(
    *,
    cfg: HandheldRecordingConfig,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    soft_sync_buffers: dict[str, TimestampedSampleBuffer],
    output_queue: Queue[CapturedEpisodeFrame | EpisodeCaptureComplete],
    stop_event: Event,
) -> None:
    period_s = 1.0 / cfg.dataset.fps
    scheduled_timestamps = np.arange(0.0, cfg.dataset.episode_time_s, period_s, dtype=np.float64)
    episode_start_time_s = time.perf_counter()

    try:
        for frame_index, target_timestamp_s in enumerate(scheduled_timestamps):
            if stop_event.is_set():
                output_queue.put(
                    EpisodeCaptureComplete(
                        EpisodeRecordResult(
                            recorded_frames=frame_index,
                            stop_action=EpisodeStopAction.EXIT,
                        )
                    )
                )
                return

            pressed = _read_terminal_key_nonblocking()
            if pressed == "esc":
                logging.info("Detected Esc during recording; stopping recording session.")
                output_queue.put(
                    EpisodeCaptureComplete(
                        EpisodeRecordResult(
                            recorded_frames=frame_index,
                            stop_action=EpisodeStopAction.EXIT,
                        )
                    )
                )
                return
            if pressed == "s":
                logging.info("Detected 's' during recording; stopping early and saving the episode.")
                output_queue.put(
                    EpisodeCaptureComplete(
                        EpisodeRecordResult(
                            recorded_frames=frame_index,
                            stop_action=EpisodeStopAction.SAVE_EARLY,
                        )
                    )
                )
                return
            if pressed == "n":
                logging.info("Detected 'n' during recording; stopping early and discarding the episode.")
                output_queue.put(
                    EpisodeCaptureComplete(
                        EpisodeRecordResult(
                            recorded_frames=frame_index,
                            stop_action=EpisodeStopAction.DISCARD_EARLY,
                        )
                    )
                )
                return

            sleep_s = episode_start_time_s + float(target_timestamp_s) - time.perf_counter()
            if sleep_s > 0:
                precise_sleep(sleep_s)

            soft_sync_result = None
            soft_sync_samples = None
            if cfg.sensors.soft_sync.enabled:
                target_capture_time_s = episode_start_time_s + float(target_timestamp_s)
                if soft_sync_buffers:
                    selection = _wait_for_soft_sync_samples(
                        cfg=cfg.sensors.soft_sync,
                        buffers=soft_sync_buffers,
                        target_capture_time_s=target_capture_time_s,
                        episode_start_time_s=episode_start_time_s,
                    )
                    soft_sync_result = selection.result
                    soft_sync_samples = selection.samples
                else:
                    soft_sync_result = _wait_for_soft_sync_target(
                        cfg=cfg.sensors.soft_sync,
                        cameras=cameras,
                        tactiles=tactiles,
                        handheld_grippers=handheld_grippers,
                        target_capture_time_s=target_capture_time_s,
                        episode_start_time_s=episode_start_time_s,
                    )

            frame = collect_dataset_frame(
                cameras,
                tactiles,
                handheld_grippers,
                max_read_age_ms=cfg.sensors.max_read_age_ms,
                episode_start_time_s=episode_start_time_s,
                task=cfg.dataset.single_task,
                soft_sync_result=soft_sync_result,
                soft_sync_samples=soft_sync_samples,
            )
            output_queue.put(
                CapturedEpisodeFrame(
                    frame_index=frame_index,
                    dataset_timestamp_s=float(target_timestamp_s),
                    frame=frame,
                )
            )

        output_queue.put(
            EpisodeCaptureComplete(
                EpisodeRecordResult(
                    recorded_frames=int(len(scheduled_timestamps)),
                    stop_action=EpisodeStopAction.TIMED_OUT,
                )
            )
        )
    except BaseException as exc:  # noqa: BLE001
        output_queue.put(EpisodeCaptureComplete(error=exc))


def _record_episode(
    *,
    cfg: HandheldRecordingConfig,
    dataset: LeRobotDataset,
    cameras: dict[str, Any],
    tactiles: dict[str, Any],
    handheld_grippers: dict[str, Any],
    soft_sync_buffers: dict[str, TimestampedSampleBuffer],
    rerun_enabled: bool,
) -> EpisodeRecordResult:
    output_queue: Queue[CapturedEpisodeFrame | EpisodeCaptureComplete] = Queue(
        maxsize=cfg.dataset.capture_queue_size
    )
    stop_event = Event()

    with _terminal_cbreak_mode() as keyboard_enabled:
        if keyboard_enabled:
            print("Recording... press 's' to stop+save, 'n' to stop+discard, or Esc to exit.")

        capture_thread = Thread(
            target=_capture_episode_frames,
            kwargs={
                "cfg": cfg,
                "cameras": cameras,
                "tactiles": tactiles,
                "handheld_grippers": handheld_grippers,
                "soft_sync_buffers": soft_sync_buffers,
                "output_queue": output_queue,
                "stop_event": stop_event,
            },
            daemon=True,
            name="handheld-capture-producer",
        )
        capture_thread.start()

        try:
            while True:
                item = output_queue.get()
                if isinstance(item, EpisodeCaptureComplete):
                    if item.error is not None:
                        raise item.error
                    if item.result is None:
                        raise RuntimeError("Episode capture finished without a result.")
                    return item.result

                dataset.add_frame(item.frame)
                if rerun_enabled:
                    _log_rerun_frame(
                        frame_index=item.frame_index,
                        dataset_timestamp_s=item.dataset_timestamp_s,
                        cameras=cameras,
                        tactiles=tactiles,
                        handheld_grippers=handheld_grippers,
                        frame=item.frame,
                    )
        finally:
            stop_event.set()
            capture_thread.join(timeout=2.0)


def _create_or_resume_dataset(
    cfg: HandheldRecordingConfig,
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
        robot_type=cfg.robot_type,
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


@parser.wrap()
def record(cfg: HandheldRecordingConfig) -> LeRobotDataset:
    init_logging()

    all_cameras = make_cameras_from_configs(cfg.sensors.cameras)
    cameras = all_cameras
    tactiles = _make_tactiles_from_configs(cfg.sensors.tactiles)
    handheld_grippers = make_handheld_grippers_from_configs(cfg.sensors.handheld_grippers)
    dataset: LeRobotDataset | None = None
    soft_sync_buffers: dict[str, TimestampedSampleBuffer] = {}

    try:
        cameras = _connect_cameras_best_effort(all_cameras)
        _connect_devices(tactiles, handheld_grippers)
        if not cameras and not tactiles and not handheld_grippers:
            raise RuntimeError("No devices connected successfully.")
        features = build_dataset_features(
            cameras,
            tactiles,
            handheld_grippers,
            use_videos=cfg.dataset.video,
            include_soft_sync_diagnostics=cfg.sensors.soft_sync.enabled,
        )
        dataset = _create_or_resume_dataset(cfg, features=features, num_cameras=len(cameras))
        rerun_enabled = _init_rerun(cfg)

        _print_intro(cfg, cameras, tactiles, handheld_grippers, dataset.root)
        soft_sync_buffers = _make_soft_sync_sample_buffers(
            cfg=cfg.sensors,
            cameras=cameras,
            tactiles=tactiles,
            handheld_grippers=handheld_grippers,
        )
        _start_soft_sync_sample_buffers(soft_sync_buffers)

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If save latency is too high, consider enabling "
                "`dataset.streaming_encoding=true`."
            )

        attempted_episodes = 0
        with VideoEncodingManager(dataset):
            while not _recording_target_reached(dataset, cfg.dataset.num_episodes):
                attempted_episodes += 1
                _wait_for_enter(attempted_episodes, cfg.play_sounds)

                try:
                    record_result = _record_episode(
                        cfg=cfg,
                        dataset=dataset,
                        cameras=cameras,
                        tactiles=tactiles,
                        handheld_grippers=handheld_grippers,
                        soft_sync_buffers=soft_sync_buffers,
                        rerun_enabled=rerun_enabled,
                    )
                except KeyboardInterrupt:
                    logging.info("Recording interrupted by user during episode capture.")
                    if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                        dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                    break

                recorded_frames = record_result.recorded_frames
                print(f"Recorded {recorded_frames} frames for the current episode.")

                if record_result.stop_action == EpisodeStopAction.SAVE_EARLY:
                    if _has_episode_data(dataset):
                        dataset.save_episode()
                        print(
                            "Episode saved early by 's'. "
                            f"Total saved episodes: {dataset.num_episodes}/"
                            f"{_format_target_episodes(cfg.dataset.num_episodes)}"
                        )
                    else:
                        logging.warning("Received 's' but no frames were recorded; discarding empty episode.")
                        if dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                        print("No frames recorded; episode discarded.")
                    continue

                if record_result.stop_action == EpisodeStopAction.DISCARD_EARLY:
                    if _has_episode_data(dataset):
                        dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                    print("Episode discarded early by 'n'.")
                    continue

                if record_result.stop_action == EpisodeStopAction.EXIT:
                    if _has_episode_data(dataset):
                        dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                    print("Recording stopped by Esc. Current episode discarded.")
                    break

                if _confirm_keep_episode(cfg.play_sounds):
                    dataset.save_episode()
                    print(
                        "Episode saved. "
                        f"Total saved episodes: {dataset.num_episodes}/"
                        f"{_format_target_episodes(cfg.dataset.num_episodes)}"
                    )
                else:
                    dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                    print("Episode discarded.")

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        return dataset
    finally:
        _stop_soft_sync_sample_buffers(soft_sync_buffers)
        _disconnect_devices(cameras, tactiles, handheld_grippers)


def main() -> None:
    record()


if __name__ == "__main__":
    main()
