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
        self.clear_episode_buffer_calls = 0

    def save_episode(self):
        self.num_episodes += 1

    def clear_episode_buffer(self):
        self.clear_episode_buffer_calls += 1

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
        self.call_order = None
        self.move_to_start_calls = 0
        self.send_action_calls = []
        self._observation = {
            "ee.x": 0.4,
            "ee.y": 0.0,
            "ee.z": 0.3,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 0.5,
        }

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return self._observation.copy()

    def send_action(self, action):
        self.send_action_calls.append(action.copy())
        if self.call_order is not None:
            self.call_order.append("send_action")
        return action

    def move_to_start(self):
        self.move_to_start_calls += 1
        if self.call_order is not None:
            self.call_order.append("move_to_start")


class FakeTeleop:
    action_features = {}

    def __init__(self):
        self.is_connected = False
        self.last_gripper = 0.5

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_action(self):
        return {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": self.last_gripper,
        }

    def set_gripper(self, normalized_position: float):
        self.last_gripper = normalized_position


class FakeTargetActionProcessor(FakeProcessor):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return self.output.copy()


class FakeObservationProcessor(FakeProcessor):
    def __call__(self, value):
        return value


class FakeRobotActionProcessor(FakeProcessor):
    def __call__(self, value):
        action, _observation = value
        return action


class FakeSettlingRobot(FakeRobot):
    def __init__(self, observations):
        super().__init__()
        self._observations = [observation.copy() for observation in observations]
        self.send_action_calls = []

    def get_observation(self):
        if len(self._observations) > 1:
            return self._observations.pop(0)
        return self._observations[0].copy()

    def send_action(self, action):
        self.send_action_calls.append(action.copy())
        return action


class FakeTeleopWithAction(FakeTeleop):
    def __init__(self, action):
        super().__init__()
        self._action = action
        self.get_action_calls = 0

    def get_action(self):
        self.get_action_calls += 1
        return self._action.copy()


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
    monkeypatch.setattr(fr3_record_runtime, "_wait_for_episode_start_settle", lambda **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "_confirm_keep_episode", lambda play_sounds: True)
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
    assert robot_observation_processor.reset_calls == 1
    assert dataset.finalized is True


def test_record_waits_for_episode_start_settle_before_record_loop(monkeypatch):
    teleop_action_processor = FakeProcessor()
    robot_action_processor = FakeProcessor()
    robot_observation_processor = FakeProcessor()
    robot = FakeRobot()
    teleop = FakeTeleop()
    call_order = []
    teleop.wait_until_idle = lambda **kwargs: call_order.append("idle") or True

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
    monkeypatch.setattr(
        fr3_record_runtime,
        "_wait_for_episode_start_settle",
        lambda **kwargs: call_order.append("settle"),
    )
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: call_order.append("record"))
    monkeypatch.setattr(fr3_record_runtime, "_confirm_keep_episode", lambda play_sounds: True)
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

    fr3_record_runtime.record(cfg)

    assert call_order == ["idle", "settle", "record"]


def test_record_resets_before_first_episode_when_auto_move_enabled(monkeypatch):
    teleop_action_processor = FakeProcessor()
    robot_action_processor = FakeProcessor()
    robot_observation_processor = FakeProcessor()
    robot = FakeRobot()
    teleop = FakeTeleop()
    call_order = []
    robot.call_order = call_order
    teleop.wait_until_idle = lambda **kwargs: call_order.append("idle") or True

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
    monkeypatch.setattr(
        fr3_record_runtime,
        "_wait_for_episode_start_settle",
        lambda **kwargs: call_order.append("settle"),
    )
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: call_order.append("record"))
    monkeypatch.setattr(fr3_record_runtime, "_confirm_keep_episode", lambda play_sounds: True)
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
        auto_move_to_start_after_episode=True,
        move_to_start_after_last_episode=True,
        confirm_next_episode_after_reset=False,
        play_sounds=False,
        display_data=False,
    )

    fr3_record_runtime.record(cfg)

    assert call_order[:5] == ["move_to_start", "send_action", "idle", "settle", "record"]


def test_record_moves_to_start_opens_gripper_before_keep_confirmation(monkeypatch):
    teleop_action_processor = FakeProcessor()
    robot_action_processor = FakeProcessor()
    robot_observation_processor = FakeProcessor()
    robot = FakeRobot()
    teleop = FakeTeleop()
    call_order = []
    robot.call_order = call_order
    teleop.wait_until_idle = lambda **kwargs: True

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
    monkeypatch.setattr(fr3_record_runtime, "_wait_for_episode_start_settle", lambda **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: call_order.append("record"))
    monkeypatch.setattr(
        fr3_record_runtime,
        "_confirm_keep_episode",
        lambda play_sounds: call_order.append("confirm") or True,
    )
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
        auto_move_to_start_after_episode=True,
        move_to_start_after_last_episode=True,
        confirm_next_episode_after_reset=False,
        play_sounds=False,
        display_data=False,
    )

    dataset = fr3_record_runtime.record(cfg)

    assert call_order == ["move_to_start", "send_action", "record", "move_to_start", "send_action", "confirm"]
    assert robot.move_to_start_calls == 2
    assert robot.send_action_calls[-1]["gripper"] == 1.0
    assert teleop.last_gripper == 1.0
    assert dataset.num_episodes == 1


def test_record_discards_episode_when_keep_confirmation_rejects(monkeypatch):
    teleop_action_processor = FakeProcessor()
    robot_action_processor = FakeProcessor()
    robot_observation_processor = FakeProcessor()
    robot = FakeRobot()
    teleop = FakeTeleop()
    teleop.wait_until_idle = lambda **kwargs: True
    dataset_holder = {}
    keep_responses = iter([False, True])

    def create_dataset(*args, **kwargs):
        del args, kwargs
        dataset = FakeDataset()
        dataset_holder["dataset"] = dataset
        return dataset

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
    monkeypatch.setattr(fr3_record_runtime, "LeRobotDataset", type("Factory", (), {"create": staticmethod(create_dataset)}))
    monkeypatch.setattr(fr3_record_runtime, "_wait_for_episode_start_settle", lambda **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "record_loop", lambda **kwargs: None)
    monkeypatch.setattr(fr3_record_runtime, "_confirm_keep_episode", lambda play_sounds: next(keep_responses))
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

    assert dataset is dataset_holder["dataset"]
    assert dataset.num_episodes == 1
    assert dataset.clear_episode_buffer_calls == 1
    assert teleop_action_processor.reset_calls == 2
    assert robot_observation_processor.reset_calls == 2


def test_wait_for_episode_start_settle_freezes_initial_target_and_waits_until_gripper_catches_up(monkeypatch):
    monkeypatch.setattr(fr3_record_runtime, "precise_sleep", lambda duration_s: None)
    monkeypatch.setattr(fr3_record_runtime, "EPISODE_START_SETTLE_CONSECUTIVE_SAMPLES", 2)
    monkeypatch.setattr(fr3_record_runtime, "EPISODE_START_SETTLE_TIMEOUT_S", 1.0)

    target_action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 1.0,
        "ee.x": 0.4,
        "ee.y": 0.0,
        "ee.z": 0.3,
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
        "gripper.pos": 1.0,
    }
    observations = [
        {
            "ee.x": 0.4,
            "ee.y": 0.0,
            "ee.z": 0.3,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 0.0,
        },
        {
            "ee.x": 0.4,
            "ee.y": 0.0,
            "ee.z": 0.3,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 0.6,
        },
        {
            "ee.x": 0.4,
            "ee.y": 0.0,
            "ee.z": 0.3,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 1.0,
        },
        {
            "ee.x": 0.4,
            "ee.y": 0.0,
            "ee.z": 0.3,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 1.0,
        },
    ]
    robot = FakeSettlingRobot(observations)
    teleop = FakeTeleopWithAction(
        {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": 1.0,
        }
    )
    teleop_action_processor = FakeTargetActionProcessor(target_action)
    robot_action_processor = FakeRobotActionProcessor()
    robot_observation_processor = FakeObservationProcessor()

    fr3_record_runtime._wait_for_episode_start_settle(
        robot=robot,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        events={"stop_recording": False},
        fps=30,
    )

    assert teleop.get_action_calls == 1
    assert len(teleop_action_processor.calls) == 0
    assert len(robot.send_action_calls) == 3
    assert all(action["gripper.pos"] == 1.0 for action in robot.send_action_calls)
    assert all(action["ee.x"] == 0.4 for action in robot.send_action_calls)
    assert teleop_action_processor.reset_calls == 1
    assert robot_observation_processor.reset_calls == 1


def test_wait_for_episode_start_settle_ignores_initial_spacemouse_translation(monkeypatch):
    monkeypatch.setattr(fr3_record_runtime, "precise_sleep", lambda duration_s: None)
    monkeypatch.setattr(fr3_record_runtime, "EPISODE_START_SETTLE_CONSECUTIVE_SAMPLES", 1)
    monkeypatch.setattr(fr3_record_runtime, "EPISODE_START_SETTLE_TIMEOUT_S", 1.0)

    observations = [
        {
            "ee.x": 0.42,
            "ee.y": 0.01,
            "ee.z": 0.31,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 0.4,
        },
        {
            "ee.x": 0.42,
            "ee.y": 0.01,
            "ee.z": 0.31,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
            "gripper.pos": 0.7,
        },
    ]
    robot = FakeSettlingRobot(observations)
    teleop = FakeTeleopWithAction(
        {
            "enabled": True,
            "target_x": -0.05,
            "target_y": 0.02,
            "target_z": 0.01,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.1,
            "gripper": 0.7,
        }
    )
    teleop_action_processor = FakeTargetActionProcessor(
        {
            "enabled": True,
            "target_x": -0.05,
            "target_y": 0.02,
            "target_z": 0.01,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.1,
            "gripper": 0.7,
            "ee.x": 0.37,
            "ee.y": 0.03,
            "ee.z": 0.32,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.05,
            "ee.qw": 0.998749217771909,
            "gripper.pos": 0.7,
        }
    )
    robot_action_processor = FakeRobotActionProcessor()
    robot_observation_processor = FakeObservationProcessor()

    fr3_record_runtime._wait_for_episode_start_settle(
        robot=robot,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        events={"stop_recording": False},
        fps=30,
    )

    assert teleop.get_action_calls == 1
    assert len(teleop_action_processor.calls) == 0
    assert len(robot.send_action_calls) == 1
    assert robot.send_action_calls[0]["ee.x"] == 0.42
    assert robot.send_action_calls[0]["ee.y"] == 0.01
    assert robot.send_action_calls[0]["ee.z"] == 0.31
    assert robot.send_action_calls[0]["gripper.pos"] == 0.7
    assert teleop_action_processor.reset_calls == 1
    assert robot_observation_processor.reset_calls == 1


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

    assert all(key in processed_for_dataset for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"))
    assert np.isclose(processed_for_dataset["ee.z"], observation["ee.z"])
    assert processed_for_robot == idle_action
