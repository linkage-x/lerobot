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

import numpy as np

from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
from tools.fr3 import fr3_record_runtime


class FakeProcessor:
    def __init__(self):
        self.reset_calls = 0

    def __call__(self, value):
        return value

    def reset(self):
        self.reset_calls += 1


class FakeDataset:
    def __init__(self):
        self.num_episodes = 0
        self.features = {}
        self.finalized = False

    def save_episode(self):
        self.num_episodes += 1

    def clear_episode_buffer(self):
        pass

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, tags=None, private=False):
        del tags, private


class FakeDatasetFactory:
    @staticmethod
    def create(*args, **kwargs):
        del args, kwargs
        return FakeDataset()


class FakeVideoEncodingManager:
    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class FakeRobot:
    name = "franka_research3"
    observation_features = {}
    cameras = {}

    def __init__(self):
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def move_to_start(self):
        pass


class FakeTeleop:
    action_features = {}

    def __init__(self):
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


def test_record_resets_teleop_action_processor_after_episode(monkeypatch):
    teleop_action_processor = FakeProcessor()
    robot_action_processor = FakeProcessor()
    robot_observation_processor = FakeProcessor()
    robot = FakeRobot()
    teleop = FakeTeleop()

    monkeypatch.setattr(fr3_record_runtime, "init_logging", lambda: None)
    monkeypatch.setattr(fr3_record_runtime, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(fr3_record_runtime, "make_teleoperator_from_config", lambda cfg: teleop)
    monkeypatch.setattr(
        fr3_record_runtime,
        "make_fr3_ee2ee_processors",
        lambda cfg: (teleop_action_processor, robot_action_processor, robot_observation_processor),
    )
    monkeypatch.setattr(fr3_record_runtime, "aggregate_pipeline_dataset_features", lambda **kwargs: {})
    monkeypatch.setattr(fr3_record_runtime, "create_initial_features", lambda **kwargs: {})
    monkeypatch.setattr(fr3_record_runtime, "combine_feature_dicts", lambda *args: {})
    monkeypatch.setattr(fr3_record_runtime, "sanity_check_dataset_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "VideoEncodingManager", FakeVideoEncodingManager)
    monkeypatch.setattr(fr3_record_runtime, "LeRobotDataset", FakeDatasetFactory)
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: None)
    monkeypatch.setattr(
        fr3_record_runtime,
        "init_keyboard_listener",
        lambda: (None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}),
    )
    monkeypatch.setattr(fr3_record_runtime, "log_say", lambda *args, **kwargs: None)

    cfg = RecordConfig(
        robot=FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        ),
        teleop=SpaceMouseTeleopConfig(),
        dataset=DatasetRecordConfig(
            repo_id="local/fr3_test",
            single_task="test",
            root="/tmp/fr3_test",
            fps=30,
            num_episodes=1,
            episode_time_s=1,
            reset_time_s=0,
            video=False,
            push_to_hub=False,
        ),
        auto_move_to_start_after_episode=False,
        move_to_start_after_last_episode=False,
        confirm_next_episode_after_reset=False,
        play_sounds=False,
        display_data=False,
    )

    dataset = fr3_record_runtime.record(cfg)

    assert dataset.num_episodes == 1
    assert teleop_action_processor.reset_calls == 1
    assert dataset.finalized is True


def test_make_fr3_ee2ee_processors_use_delta_hold_for_idle_robot_frames():
    cfg = RecordConfig(
        robot=FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            workspace_min=(0.2, -0.2, 0.1),
            workspace_max=(0.7, 0.2, 0.6),
        ),
        teleop=SpaceMouseTeleopConfig(),
        dataset=DatasetRecordConfig(
            repo_id="local/fr3_test",
            single_task="test",
            root="/tmp/fr3_test",
            fps=30,
            num_episodes=1,
            episode_time_s=1,
            reset_time_s=0,
            video=False,
            push_to_hub=False,
        ),
        auto_move_to_start_after_episode=False,
        move_to_start_after_last_episode=False,
        confirm_next_episode_after_reset=False,
        play_sounds=False,
        display_data=False,
    )

    teleop_action_processor, robot_action_processor, _ = fr3_record_runtime.make_fr3_ee2ee_processors(cfg)
    observation = {
        "ee.x": 0.4,
        "ee.y": 0.1,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "gripper.pos": 0.5,
    }
    idle_action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    processed_for_dataset = teleop_action_processor((idle_action.copy(), observation))
    processed_for_robot = robot_action_processor((processed_for_dataset, observation))

    assert all(key in processed_for_dataset for key in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"))
    assert np.isclose(processed_for_dataset["ee.z"], observation["ee.z"])
    assert processed_for_robot == idle_action
