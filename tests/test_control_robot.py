#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.cameras import CameraConfig
from lerobot.cameras.hikrobot import HikrobotCameraConfig
from lerobot.cameras.hikrobot.configuration_hikrobot import (
    HIKROBOT_DEFAULT_COLOR_MODE,
    HIKROBOT_DEFAULT_EXPOSURE_US,
    HIKROBOT_DEFAULT_GAIN_DB,
    HIKROBOT_DEFAULT_GAMMA,
    HIKROBOT_DEFAULT_LOCK_WHITE_BALANCE_AFTER_WARMUP,
    HIKROBOT_DEFAULT_WARMUP_S,
)
from lerobot.cameras.hikrobot.configuration_hikrobot import ColorMode
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.robots import RobotConfig
from lerobot.scripts import lerobot_record as lerobot_record_module
from lerobot.scripts import lerobot_teleoperate as lerobot_teleoperate_module
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    _build_record_loop_frequency_warning,
    _confirm_keep_episode,
    _confirm_next_episode,
    _move_robot_to_start,
    record,
    record_loop,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig, SpaceMouseToolMode
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


@dataclass(kw_only=True)
class _HikrobotConfigSurfaceRobot(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()


def _make_hikrobot_camera_config(**overrides) -> HikrobotCameraConfig:
    return HikrobotCameraConfig(
        serial="LEFT123",
        width=1280,
        height=720,
        fps=30,
        **overrides,
    )


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_teleoperate_forces_binary_spacemouse_for_franka_hand(monkeypatch):
    captured = {}

    class FakeRobot:
        def connect(self):
            return None

        def disconnect(self):
            return None

    class FakeTeleop:
        def connect(self):
            return None

        def disconnect(self):
            return None

    def _make_teleop(cfg):
        captured["tool_mode"] = cfg.tool_mode
        return FakeTeleop()

    monkeypatch.setattr(lerobot_teleoperate_module, "init_logging", lambda: None)
    monkeypatch.setattr(
        lerobot_teleoperate_module,
        "make_teleoperator_from_config",
        _make_teleop,
    )
    monkeypatch.setattr(lerobot_teleoperate_module, "make_robot_from_config", lambda cfg: FakeRobot())
    monkeypatch.setattr(
        lerobot_teleoperate_module,
        "make_default_processors",
        lambda: (None, None, None),
    )
    monkeypatch.setattr(lerobot_teleoperate_module, "teleop_loop", lambda **kwargs: None)

    cfg = TeleoperateConfig(
        robot=FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_backend="franka_hand",
            allow_mock_gripper=False,
            urdf_path="/tmp/fr3.urdf",
        ),
        teleop=SpaceMouseTeleopConfig(tool_mode=SpaceMouseToolMode.INCREMENTAL),
        teleop_time_s=0.1,
    )

    teleoperate(cfg)

    assert captured["tool_mode"] == SpaceMouseToolMode.BINARY
    assert cfg.teleop.tool_mode == SpaceMouseToolMode.BINARY


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)


def test_record_higher_control_fps_keeps_dataset_fps(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_high_control_fps",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        control_fps=200,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3


def test_record_loop_warning_omits_dataset_drop_above_dataset_fps():
    warning = _build_record_loop_frequency_warning(actual_fps=75.3, control_fps=200, dataset_fps=30)

    assert "target control FPS (200 Hz)" in warning
    assert "Dataset frames might be dropped" not in warning


def test_record_loop_warning_mentions_dataset_drop_below_dataset_fps():
    warning = _build_record_loop_frequency_warning(actual_fps=8.9, control_fps=200, dataset_fps=30)

    assert "Dataset frames might be dropped" in warning
    assert "dataset.fps (30 Hz)" in warning


def test_record_loop_samples_cameras_only_on_dataset_ticks(monkeypatch):
    class FakeDataset:
        fps = 20
        features = {}

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    class FakeRobot:
        name = "franka_research3"
        robot_type = "franka_research3"

        def __init__(self):
            self.include_cameras_calls = []

        def get_observation(self, *, include_cameras: bool = True):
            self.include_cameras_calls.append(include_cameras)
            observation = {"joint_1.pos": 0.0}
            if include_cameras:
                observation["front"] = "frame"
            return observation

        def send_action(self, action):
            return action

    class FakeTeleop:
        def get_action(self):
            return {"joint_1.pos": 0.1}

    perf_counter_values = iter(
        [
            0.00,
            0.00,
            0.00,
            0.01,
            0.01,
            0.01,
            0.02,
            0.02,
            0.03,
            0.03,
            0.03,
            0.05,
            0.05,
            0.06,
            0.06,
            0.06,
            0.08,
            0.08,
            0.09,
            0.09,
            0.09,
            0.10,
            0.10,
            0.11,
            0.11,
            0.11,
        ]
    )
    monkeypatch.setattr(lerobot_record_module.time, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(lerobot_record_module, "precise_sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lerobot_record_module, "Teleoperator", FakeTeleop)
    monkeypatch.setattr(
        lerobot_record_module,
        "build_dataset_frame",
        lambda features, values, prefix: {f"{prefix}.has_camera": "front" in values},
    )

    dataset = FakeDataset()
    robot = FakeRobot()

    record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=100,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=dataset,
        teleop=FakeTeleop(),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        control_time_s=0.11,
        single_task="test",
        display_data=False,
        display_compressed_images=False,
    )

    assert robot.include_cameras_calls == [True, False, True, False, True]
    assert len(dataset.frames) == 3
    assert all(frame["observation.has_camera"] is True for frame in dataset.frames)


def test_record_loop_normalizes_bgr_camera_frames_to_rgb_for_dataset(monkeypatch):
    class FakeDataset:
        fps = 10
        features = {}

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    class FakeCamera:
        def __init__(self, color_mode):
            self.config = type("Config", (), {"color_mode": color_mode})()

    class FakeRobot:
        name = "franka_research3"
        robot_type = "franka_research3"

        def __init__(self):
            self.cameras = {"front": FakeCamera(ColorMode.BGR)}

        def get_observation(self, *, include_cameras: bool = True):
            observation = {"joint_1.pos": 0.0}
            if include_cameras:
                frame_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
                observation["front"] = frame_bgr
            return observation

        def send_action(self, action):
            return action

    class FakeTeleop:
        def get_action(self):
            return {"joint_1.pos": 0.1}

    perf_counter_values = iter([0.0, 0.0, 0.0, 0.01, 0.01, 0.11, 0.11, 0.12, 0.12, 0.12])
    monkeypatch.setattr(lerobot_record_module.time, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(lerobot_record_module, "precise_sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lerobot_record_module, "Teleoperator", FakeTeleop)
    monkeypatch.setattr(
        lerobot_record_module,
        "build_dataset_frame",
        lambda _features, values, prefix: (
            {f"{prefix}.pixel": values["front"][0, 0].tolist()}
            if prefix == "observation"
            else {f"{prefix}.joint_1.pos": values["joint_1.pos"]}
        ),
    )

    dataset = FakeDataset()

    record_loop(
        robot=FakeRobot(),
        events={"exit_early": False},
        fps=10,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=dataset,
        teleop=FakeTeleop(),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        control_time_s=0.11,
        single_task="test",
        display_data=False,
        display_compressed_images=False,
    )

    assert len(dataset.frames) == 1
    assert dataset.frames[0]["observation.pixel"] == [0, 0, 255]


def test_record_config_uses_hikrobot_recording_profile_defaults(tmp_path):
    robot_cfg = _HikrobotConfigSurfaceRobot(cameras={"front": _make_hikrobot_camera_config()})
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "hikrobot_record_cfg",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )

    camera_cfg = cfg.robot.cameras["front"]
    assert camera_cfg.color_mode == HIKROBOT_DEFAULT_COLOR_MODE
    assert camera_cfg.warmup_s == HIKROBOT_DEFAULT_WARMUP_S
    assert camera_cfg.exposure_us == HIKROBOT_DEFAULT_EXPOSURE_US
    assert camera_cfg.gain_db == HIKROBOT_DEFAULT_GAIN_DB
    assert camera_cfg.gamma == HIKROBOT_DEFAULT_GAMMA
    assert camera_cfg.lock_white_balance_after_warmup is HIKROBOT_DEFAULT_LOCK_WHITE_BALANCE_AFTER_WARMUP


def test_teleoperate_config_uses_hikrobot_recording_profile_defaults():
    robot_cfg = _HikrobotConfigSurfaceRobot(cameras={"front": _make_hikrobot_camera_config()})
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=MockTeleopConfig(),
        teleop_time_s=0.1,
    )

    camera_cfg = cfg.robot.cameras["front"]
    assert camera_cfg.color_mode == HIKROBOT_DEFAULT_COLOR_MODE
    assert camera_cfg.warmup_s == HIKROBOT_DEFAULT_WARMUP_S
    assert camera_cfg.exposure_us == HIKROBOT_DEFAULT_EXPOSURE_US
    assert camera_cfg.gain_db == HIKROBOT_DEFAULT_GAIN_DB
    assert camera_cfg.gamma == HIKROBOT_DEFAULT_GAMMA
    assert camera_cfg.lock_white_balance_after_warmup is HIKROBOT_DEFAULT_LOCK_WHITE_BALANCE_AFTER_WARMUP


def test_explicit_hikrobot_camera_overrides_are_preserved_in_record_config(tmp_path):
    robot_cfg = _HikrobotConfigSurfaceRobot(
        cameras={
            "front": _make_hikrobot_camera_config(
                color_mode=ColorMode.RGB,
                warmup_s=0,
                exposure_us=8000.0,
                gain_db=8.0,
                gamma=None,
                lock_white_balance_after_warmup=False,
            )
        }
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "hikrobot_record_override_cfg",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )

    camera_cfg = cfg.robot.cameras["front"]
    assert camera_cfg.color_mode == ColorMode.RGB
    assert camera_cfg.warmup_s == 0
    assert camera_cfg.exposure_us == 8000.0
    assert camera_cfg.gain_db == 8.0
    assert camera_cfg.gamma is None
    assert camera_cfg.lock_white_balance_after_warmup is False


class StartableRobot:
    name = "startable"

    def __init__(self):
        self.calls = 0

    def move_to_start(self):
        self.calls += 1


def test_move_robot_to_start_calls_robot_method():
    robot = StartableRobot()

    _move_robot_to_start(robot, play_sounds=False)

    assert robot.calls == 1


def test_move_robot_to_start_raises_for_unsupported_robot():
    with pytest.raises(RuntimeError, match="does not support move_to_start"):
        _move_robot_to_start(object(), play_sounds=False)


def test_confirm_next_episode_accepts_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert _confirm_next_episode(play_sounds=False) is True


def test_confirm_next_episode_accepts_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert _confirm_next_episode(play_sounds=False) is False


def test_confirm_keep_episode_accepts_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert _confirm_keep_episode(play_sounds=False) is True


def test_confirm_keep_episode_accepts_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert _confirm_keep_episode(play_sounds=False) is False
