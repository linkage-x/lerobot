#!/usr/bin/env python

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

import sys
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode
from tools.handheld import handheld_record


class FakeCamera:
    def __init__(self, *, height: int, width: int, color_mode: ColorMode, timestamp: float):
        self.height = height
        self.width = width
        self.config = type("Config", (), {"height": height, "width": width, "color_mode": color_mode})()
        self.latest_timestamp = timestamp
        self.frame_lock = Lock()

    def read_latest(self, max_age_ms: int):
        del max_age_ms
        return np.array([[[255, 0, 0]]], dtype=np.uint8)


class FakeTactile:
    def __init__(self, *, num_taxels: int, num_dimensions: int, timestamp: float):
        self.num_taxels = num_taxels
        self.num_dimensions = num_dimensions
        self.latest_timestamp = timestamp
        self.frame_lock = Lock()

    def read_latest(self, max_age_ms: int):
        del max_age_ms
        return np.arange(self.num_taxels * self.num_dimensions, dtype=np.int16).reshape(
            self.num_taxels,
            self.num_dimensions,
        )


class FakeGripper:
    def __init__(self, *, timestamp: float):
        self.latest_timestamp = timestamp
        self.read_lock = Lock()

    def read_latest(self, max_age_ms: int):
        del max_age_ms
        return 12.5


class FakeConnectableCamera:
    def __init__(self, *, should_fail: bool):
        self.should_fail = should_fail
        self.is_connected = False
        self.disconnect_calls = 0

    def connect(self):
        if self.should_fail:
            raise RuntimeError("camera offline")
        self.is_connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False


def test_build_dataset_features_includes_camera_tactile_and_gripper_streams():
    cameras = {"front": FakeCamera(height=720, width=1280, color_mode=ColorMode.RGB, timestamp=0.0)}
    tactiles = {"paxini": FakeTactile(num_taxels=240, num_dimensions=3, timestamp=0.0)}
    handheld_grippers = {"pika": FakeGripper(timestamp=0.0)}

    features = handheld_record.build_dataset_features(
        cameras,
        tactiles,
        handheld_grippers,
        use_videos=True,
    )

    assert features["observation.images.front"]["dtype"] == "video"
    assert features["observation.images.front"]["shape"] == (720, 1280, 3)
    assert features["observation.state"]["shape"] == (1,)
    assert features["observation.state"]["names"] == ["handheld_gripper.pika.width_mm"]
    assert features["observation.device_capture_timestamp"]["names"] == [
        "camera.front.capture_timestamp_s",
        "tactile.paxini.capture_timestamp_s",
        "handheld_gripper.pika.capture_timestamp_s",
    ]
    assert features["observation.tactile.paxini.left_xyz"]["shape"] == (3, 10, 12)
    assert features["observation.tactile.paxini.right_xyz"]["shape"] == (3, 10, 12)
    assert features["observation.tactile.paxini.left_magnitude"]["shape"] == (10, 12)
    assert features["observation.tactile.paxini.right_magnitude"]["shape"] == (10, 12)
    assert features["observation.tactile.paxini.raw_xyz"]["shape"] == (2, 120, 3)


def test_collect_dataset_frame_normalizes_bgr_images_and_preserves_capture_times():
    cameras = {"front": FakeCamera(height=1, width=1, color_mode=ColorMode.BGR, timestamp=10.25)}
    tactiles = {"paxini": FakeTactile(num_taxels=240, num_dimensions=3, timestamp=10.30)}
    handheld_grippers = {"pika": FakeGripper(timestamp=10.40)}

    frame = handheld_record.collect_dataset_frame(
        cameras,
        tactiles,
        handheld_grippers,
        max_read_age_ms=500,
        episode_start_time_s=10.0,
        task="demo",
    )

    assert frame["task"] == "demo"
    assert frame["observation.images.front"][0, 0].tolist() == [0, 0, 255]
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].tolist() == [12.5]
    assert frame["observation.tactile.paxini.left_xyz"].shape == (3, 10, 12)
    assert frame["observation.tactile.paxini.right_xyz"].shape == (3, 10, 12)
    assert frame["observation.tactile.paxini.left_magnitude"].shape == (10, 12)
    assert frame["observation.tactile.paxini.right_magnitude"].shape == (10, 12)
    assert frame["observation.tactile.paxini.raw_xyz"].shape == (2, 120, 3)
    np.testing.assert_array_equal(
        frame["observation.tactile.paxini.raw_xyz"][0, 0],
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        frame["observation.tactile.paxini.left_xyz"][:, 0, 0],
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        frame["observation.tactile.paxini.left_magnitude"][0, 0],
        np.linalg.norm(np.array([0.0, 1.0, 2.0], dtype=np.float32)),
    )
    np.testing.assert_allclose(
        frame["observation.device_capture_timestamp"],
        np.array([0.25, 0.30, 0.40], dtype=np.float64),
        atol=1e-9,
    )


def test_build_handheld_tactile_observation_rejects_unexpected_shape():
    with pytest.raises(ValueError, match="frame must have shape"):
        handheld_record._build_handheld_tactile_observation(
            "paxini",
            np.zeros((120, 3), dtype=np.float32),
        )


def test_connect_cameras_best_effort_skips_failed_camera(caplog: pytest.LogCaptureFixture):
    cameras = {
        "good": FakeConnectableCamera(should_fail=False),
        "bad": FakeConnectableCamera(should_fail=True),
    }

    with caplog.at_level("WARNING"):
        connected = handheld_record._connect_cameras_best_effort(cameras)

    assert list(connected) == ["good"]
    assert cameras["good"].is_connected is True
    assert cameras["bad"].is_connected is False
    assert cameras["bad"].disconnect_calls == 0
    assert "Camera 'bad' failed to connect and will be skipped" in caplog.text


def test_log_rerun_frame_uses_structured_device_capture_timestamp_paths(monkeypatch):
    logged = []

    fake_rerun = SimpleNamespace(
        set_time=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: logged.append((args, kwargs)),
        Image=lambda value: ("Image", value),
        Tensor=lambda value: ("Tensor", value),
        Scalars=lambda value: ("Scalars", value),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)

    handheld_record._log_rerun_frame(
        frame_index=0,
        dataset_timestamp_s=0.25,
        cameras={"front": object()},
        tactiles={"paxini": object()},
        handheld_grippers={"pika": object()},
        frame={
            "observation.images.front": np.zeros((1, 1, 3), dtype=np.uint8),
            "observation.tactile.paxini.left_xyz": np.zeros((3, 10, 12), dtype=np.float32),
            "observation.tactile.paxini.right_xyz": np.zeros((3, 10, 12), dtype=np.float32),
            "observation.tactile.paxini.left_magnitude": np.zeros((10, 12), dtype=np.float32),
            "observation.tactile.paxini.right_magnitude": np.zeros((10, 12), dtype=np.float32),
            "observation.tactile.paxini.raw_xyz": np.zeros((2, 120, 3), dtype=np.float32),
            "observation.state": np.array([12.5], dtype=np.float32),
            "observation.device_capture_timestamp": np.array([0.25, 0.30, 0.40], dtype=np.float64),
        },
    )

    logged_paths = [args[0] for args, _ in logged]

    assert "observation/device_capture_timestamp/camera/front" in logged_paths
    assert "observation/device_capture_timestamp/tactile/paxini" in logged_paths
    assert "observation/device_capture_timestamp/handheld_gripper/pika" in logged_paths
