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
from lerobot.robots import RobotConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    _confirm_keep_episode,
    _confirm_next_episode,
    _move_robot_to_start,
    record,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
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
