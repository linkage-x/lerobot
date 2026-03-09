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

import numpy as np
import pytest

from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config


class DummyArmDriver:
    instances: list["DummyArmDriver"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.connected = False
        self.joint_positions = np.array([0.1, 0.2, 0.3, -1.0, 0.5, 1.2, -0.7], dtype=np.float64)
        self.set_joint_positions_calls: list[np.ndarray] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_positions(self) -> np.ndarray:
        return self.joint_positions.copy()

    def get_ee_pose(self) -> np.ndarray | None:
        return None

    def set_joint_positions(self, joint_positions: np.ndarray) -> None:
        self.set_joint_positions_calls.append(np.asarray(joint_positions, dtype=np.float64))


class DummyGripperDriver:
    instances: list["DummyGripperDriver"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.connected = False
        self.position = 0.25
        self.set_position_calls: list[float] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_position(self) -> float:
        return self.position

    def set_position(self, normalized_position: float) -> None:
        self.set_position_calls.append(normalized_position)
        self.position = normalized_position


class DummyKinematicsDriver:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.forward_pose = np.eye(4, dtype=np.float64)
        self.forward_pose[:3, 3] = np.array([0.4, 0.1, 0.3], dtype=np.float64)
        self.inverse_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.inverse_solution = np.array([0.5, 0.4, 0.3, -0.2, 0.1, 0.0, -0.1], dtype=np.float64)

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        del joint_positions_rad
        return self.forward_pose.copy()

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
        self.inverse_calls.append((current_joint_positions_rad.copy(), desired_pose.copy()))
        return self.inverse_solution.copy()


@pytest.fixture
def robot(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    cfg = FrankaResearch3Config(
        robot_ip="192.168.1.206",
        gripper_port="/dev/ttyUSB80",
        urdf_path="/tmp/fr3.urdf",
        workspace_min=(0.2, -0.3, 0.2),
        workspace_max=(0.6, 0.3, 0.5),
    )
    device = FrankaResearch3(cfg)
    yield device
    if device.is_connected:
        device.disconnect()


def test_connect_disconnect(robot):
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected


def test_get_observation(robot):
    robot.connect()
    observation = robot.get_observation()

    expected_ee_keys = {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"}
    expected_joint_keys = {f"joint_{i}.pos" for i in range(1, 8)}
    assert expected_ee_keys.issubset(observation)
    assert expected_joint_keys.issubset(observation)
    assert observation["ee.x"] == pytest.approx(0.4)
    assert observation["ee.y"] == pytest.approx(0.1)
    assert observation["ee.z"] == pytest.approx(0.3)
    assert observation["gripper.pos"] == pytest.approx(0.25)


def test_send_action_clips_workspace_and_sends_joint_targets(robot):
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 0.5,
        "target_y": 0.5,
        "target_z": 0.5,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 1.2,
    }
    returned = robot.send_action(action)

    assert returned["gripper"] == pytest.approx(1.0)
    assert robot._gripper.set_position_calls[-1] == pytest.approx(1.0)
    assert len(robot._arm.set_joint_positions_calls) == 1
    assert np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.6, 0.3, 0.5]))


def test_send_action_disabled_reuses_last_command_pose(robot):
    robot.connect()
    first_action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    robot.send_action(first_action)
    assert robot._last_command_pose is not None

    second_action = {
        "enabled": False,
        "target_x": 0.4,
        "target_y": 0.4,
        "target_z": 0.4,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.1,
    }
    robot.send_action(second_action)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose, robot._last_command_pose)


def test_send_action_integrates_relative_pose_while_enabled(robot):
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    robot.send_action(action)

    _, first_desired_pose = robot._kinematics.inverse_calls[-2]
    _, second_desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(first_desired_pose[:3, 3], np.array([0.41, 0.08, 0.33]))
    assert np.allclose(second_desired_pose[:3, 3], np.array([0.42, 0.06, 0.36]))


def test_connect_cleans_up_partial_backends(monkeypatch):
    class FailingGripperDriver(DummyGripperDriver):
        def connect(self) -> None:
            self.connected = True
            raise RuntimeError("gripper connect failed")

    DummyArmDriver.instances = []
    FailingGripperDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    with pytest.raises(RuntimeError, match="gripper connect failed"):
        robot.connect()

    assert not robot.is_connected
    assert robot._arm is None
    assert robot._gripper is None
    assert DummyArmDriver.instances[-1].connected is False
    assert FailingGripperDriver.instances[-1].connected is False
