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

import json
import sys
from queue import Queue
from threading import Event, Lock
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


class FakeSampleBuffer:
    def __init__(self, samples):
        self.samples = list(samples)

    def nearest(self, target_timestamp_s: float, *, min_timestamp_s: float | None = None):
        candidates = self.samples
        if min_timestamp_s is not None:
            candidates = [sample for sample in candidates if sample.timestamp_s >= min_timestamp_s]
        if not candidates:
            return None
        return min(candidates, key=lambda sample: abs(sample.timestamp_s - target_timestamp_s))


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
    assert "observation.soft_sync" not in features
    assert features["observation.tactile.paxini.left_xyz"]["shape"] == (3, 10, 12)
    assert features["observation.tactile.paxini.right_xyz"]["shape"] == (3, 10, 12)
    assert features["observation.tactile.paxini.left_magnitude"]["shape"] == (10, 12)
    assert features["observation.tactile.paxini.right_magnitude"]["shape"] == (10, 12)
    assert features["observation.tactile.paxini.raw_xyz"]["shape"] == (2, 120, 3)


def test_build_dataset_features_can_include_soft_sync_diagnostics_when_requested():
    cameras = {"front": FakeCamera(height=720, width=1280, color_mode=ColorMode.RGB, timestamp=0.0)}

    features = handheld_record.build_dataset_features(
        cameras,
        tactiles={},
        handheld_grippers={},
        use_videos=True,
        include_soft_sync_diagnostics=True,
    )

    assert features["observation.soft_sync"]["names"] == [
        "target_timestamp_s",
        "max_skew_s",
        "oldest_device_lag_s",
        "global_lag_s",
        "timed_out",
    ]


def test_soft_sync_is_disabled_by_default():
    assert handheld_record.HandheldSoftSyncConfig().enabled is False


def test_write_raw_capture_metadata_records_device_and_timestamp_contract(tmp_path):
    cameras = {"front": FakeCamera(height=720, width=1280, color_mode=ColorMode.RGB, timestamp=0.0)}
    cfg = handheld_record.HandheldRecordingConfig(
        sensors=handheld_record.HandheldSensorsConfig(cameras={"front": object()}),
        dataset=handheld_record.HandheldDatasetConfig(
            repo_id="local/test",
            single_task="demo",
            root=tmp_path,
        ),
    )
    features = handheld_record.build_dataset_features(
        cameras,
        tactiles={},
        handheld_grippers={},
        use_videos=True,
    )

    metadata_path = handheld_record.write_raw_capture_metadata(
        dataset_root=tmp_path,
        cfg=cfg,
        cameras=cameras,
        tactiles={},
        handheld_grippers={},
        features=features,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["recording_mode"] == "raw_latest_samples"
    assert payload["soft_sync_applied"] is False
    assert payload["capture"]["device_capture_timestamp_names"] == ["camera.front.capture_timestamp_s"]
    assert "front" in payload["devices"]["cameras"]


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


def test_collect_dataset_frame_records_soft_sync_diagnostics():
    cameras = {"front": FakeCamera(height=1, width=1, color_mode=ColorMode.RGB, timestamp=10.25)}
    handheld_grippers = {"pika": FakeGripper(timestamp=10.24)}

    frame = handheld_record.collect_dataset_frame(
        cameras,
        tactiles={},
        handheld_grippers=handheld_grippers,
        max_read_age_ms=500,
        episode_start_time_s=10.0,
        task="demo",
        soft_sync_result=handheld_record.SoftSyncResult(
            target_timestamp_s=0.25,
            max_skew_s=999.0,
            oldest_device_lag_s=999.0,
            timed_out=True,
        ),
    )

    np.testing.assert_allclose(
        frame["observation.soft_sync"],
        np.array([0.25, 0.01, 0.01, -0.005, 1.0], dtype=np.float64),
        atol=1e-9,
    )


def test_collect_dataset_frame_uses_selected_soft_sync_samples_instead_of_latest_values():
    camera = FakeCamera(height=1, width=1, color_mode=ColorMode.RGB, timestamp=10.40)
    camera.latest_frame = np.array([[[1, 2, 3]]], dtype=np.uint8)
    gripper = FakeGripper(timestamp=10.40)
    gripper.latest_width_mm = 40.0
    selected_camera = handheld_record.TimestampedSample(
        timestamp_s=10.02,
        value=np.array([[[9, 8, 7]]], dtype=np.uint8),
    )
    selected_gripper = handheld_record.TimestampedSample(timestamp_s=10.01, value=11.0)

    frame = handheld_record.collect_dataset_frame(
        {"front": camera},
        tactiles={},
        handheld_grippers={"pika": gripper},
        max_read_age_ms=500,
        episode_start_time_s=10.0,
        task="demo",
        soft_sync_result=handheld_record.SoftSyncResult(
            target_timestamp_s=0.0,
            max_skew_s=0.0,
            oldest_device_lag_s=0.0,
            timed_out=False,
        ),
        soft_sync_samples={
            "camera.front": selected_camera,
            "handheld_gripper.pika": selected_gripper,
        },
    )

    assert frame["observation.images.front"].tolist() == [[[9, 8, 7]]]
    assert frame["observation.state"].tolist() == [11.0]
    np.testing.assert_allclose(
        frame["observation.device_capture_timestamp"],
        np.array([0.02, 0.01], dtype=np.float64),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        frame["observation.soft_sync"],
        np.array([0.0, 0.01, 0.0, 0.015, 0.0], dtype=np.float64),
        atol=1e-9,
    )


def test_wait_for_soft_sync_target_returns_ready_diagnostics_without_timeout():
    cameras = {"front": FakeCamera(height=1, width=1, color_mode=ColorMode.RGB, timestamp=10.0)}
    handheld_grippers = {"pika": FakeGripper(timestamp=9.99)}
    cfg = handheld_record.HandheldSoftSyncConfig(
        enabled=True,
        tolerance_ms=20.0,
        wait_timeout_ms=50.0,
        poll_interval_ms=1.0,
    )

    result = handheld_record._wait_for_soft_sync_target(
        cfg=cfg,
        cameras=cameras,
        tactiles={},
        handheld_grippers=handheld_grippers,
        target_capture_time_s=10.0,
        episode_start_time_s=9.0,
        now_fn=lambda: 10.0,
        sleep_fn=lambda seconds: None,
    )

    assert result.timed_out is False
    assert result.target_timestamp_s == pytest.approx(1.0)
    assert result.max_skew_s == pytest.approx(0.01)
    assert result.oldest_device_lag_s == pytest.approx(0.01)


def test_wait_for_soft_sync_target_times_out_and_returns_latest_diagnostics(caplog):
    cameras = {"front": FakeCamera(height=1, width=1, color_mode=ColorMode.RGB, timestamp=9.0)}
    cfg = handheld_record.HandheldSoftSyncConfig(
        enabled=True,
        tolerance_ms=5.0,
        wait_timeout_ms=3.0,
        poll_interval_ms=1.0,
    )
    clock = {"now": 10.0}

    def now_fn():
        return clock["now"]

    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    with caplog.at_level("WARNING"):
        result = handheld_record._wait_for_soft_sync_target(
            cfg=cfg,
            cameras=cameras,
            tactiles={},
            handheld_grippers={},
            target_capture_time_s=10.0,
            episode_start_time_s=9.0,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )

    assert result.timed_out is True
    assert result.target_timestamp_s == pytest.approx(1.0)
    assert result.max_skew_s == pytest.approx(0.0)
    assert result.oldest_device_lag_s == pytest.approx(1.0)
    assert "Soft sync timed out" in caplog.text


def test_wait_for_soft_sync_samples_selects_nearest_buffered_samples():
    cfg = handheld_record.HandheldSoftSyncConfig(
        enabled=True,
        tolerance_ms=30.0,
        wait_timeout_ms=50.0,
        poll_interval_ms=1.0,
    )
    buffers = {
        "camera.front": FakeSampleBuffer(
            [
                handheld_record.TimestampedSample(timestamp_s=9.95, value="old"),
                handheld_record.TimestampedSample(timestamp_s=10.01, value="near"),
                handheld_record.TimestampedSample(timestamp_s=10.08, value="latest"),
            ]
        ),
        "handheld_gripper.pika": FakeSampleBuffer(
            [
                handheld_record.TimestampedSample(timestamp_s=10.02, value=12.0),
            ]
        ),
    }

    selection = handheld_record._wait_for_soft_sync_samples(
        cfg=cfg,
        buffers=buffers,
        target_capture_time_s=10.0,
        episode_start_time_s=9.0,
        now_fn=lambda: 10.0,
        sleep_fn=lambda seconds: None,
    )

    assert selection.result.timed_out is False
    assert selection.samples["camera.front"].value == "near"
    assert selection.samples["handheld_gripper.pika"].value == 12.0
    assert selection.result.max_skew_s == pytest.approx(0.01)


def test_dataset_num_episodes_zero_means_unlimited():
    cfg = handheld_record.HandheldDatasetConfig(
        repo_id="local/test",
        single_task="demo",
        num_episodes=0,
    )

    assert cfg.num_episodes == 0
    dataset = SimpleNamespace(num_episodes=999)
    assert handheld_record._recording_target_reached(dataset, cfg.num_episodes) is False
    assert handheld_record._format_target_episodes(cfg.num_episodes) == "unlimited"


def test_dataset_num_episodes_rejects_negative_values():
    with pytest.raises(ValueError, match="num_episodes.*>= 0"):
        handheld_record.HandheldDatasetConfig(
            repo_id="local/test",
            single_task="demo",
            num_episodes=-1,
        )


def test_recording_target_reached_only_for_positive_episode_limit():
    assert handheld_record._recording_target_reached(SimpleNamespace(num_episodes=2), 3) is False
    assert handheld_record._recording_target_reached(SimpleNamespace(num_episodes=3), 3) is True
    assert handheld_record._recording_target_reached(SimpleNamespace(num_episodes=4), 3) is True


def test_read_terminal_key_nonblocking_maps_escape(monkeypatch):
    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 0

    monkeypatch.setattr(handheld_record.sys, "stdin", FakeStdin())
    monkeypatch.setattr(handheld_record.select, "select", lambda *args, **kwargs: ([0], [], []))
    monkeypatch.setattr(handheld_record.os, "read", lambda fd, size: b"\x1b")

    assert handheld_record._read_terminal_key_nonblocking() == "esc"


def test_capture_episode_frames_escape_returns_exit_without_frame(monkeypatch):
    cfg = handheld_record.HandheldRecordingConfig(
        sensors=handheld_record.HandheldSensorsConfig(
            cameras={"front": object()},
            soft_sync=handheld_record.HandheldSoftSyncConfig(enabled=False),
        ),
        dataset=handheld_record.HandheldDatasetConfig(
            repo_id="local/test",
            single_task="demo",
            fps=30,
            episode_time_s=1.0,
            num_episodes=0,
        ),
    )
    output_queue = Queue()

    monkeypatch.setattr(handheld_record, "_read_terminal_key_nonblocking", lambda: "esc")

    handheld_record._capture_episode_frames(
        cfg=cfg,
        cameras={"front": object()},
        tactiles={},
        handheld_grippers={},
        soft_sync_buffers={},
        output_queue=output_queue,
        stop_event=Event(),
    )

    item = output_queue.get_nowait()
    assert isinstance(item, handheld_record.EpisodeCaptureComplete)
    assert item.result is not None
    assert item.result.recorded_frames == 0
    assert item.result.stop_action == handheld_record.EpisodeStopAction.EXIT


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
            "observation.soft_sync": np.array([0.25, 0.15, 0.0, 0.01, 0.0], dtype=np.float64),
        },
    )

    logged_paths = [args[0] for args, _ in logged]

    assert "observation/device_capture_timestamp/camera/front" in logged_paths
    assert "observation/device_capture_timestamp/tactile/paxini" in logged_paths
    assert "observation/device_capture_timestamp/handheld_gripper/pika" in logged_paths
    assert "observation/soft_sync/target_timestamp_s" in logged_paths
    assert "observation/soft_sync/max_skew_s" in logged_paths
    assert "observation/soft_sync/global_lag_s" in logged_paths
