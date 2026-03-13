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
import time

from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config
from lerobot.utils.rotation import Rotation


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


class ReportingArmDriver(DummyArmDriver):
    def get_ee_pose(self) -> np.ndarray | None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array([9.0, 8.0, 7.0], dtype=np.float64)
        return pose


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


class FailingGripperDriver(DummyGripperDriver):
    def connect(self) -> None:
        self.connected = True
        raise RuntimeError("gripper connect failed")


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


class DummyOTGDriver:
    instances: list["DummyOTGDriver"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.reset_calls: list[np.ndarray] = []
        self.step_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self, current_joint_positions: np.ndarray) -> None:
        self.reset_calls.append(np.asarray(current_joint_positions, dtype=np.float64))

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        target = np.asarray(target_joint_positions, dtype=np.float64)
        self.step_calls.append((current.copy(), target.copy()))
        return target.copy()


class FailingOTGDriver(DummyOTGDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise RuntimeError("otg init failed")


@pytest.fixture
def robot(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
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
    assert len(DummyOTGDriver.instances) == 1
    assert len(DummyOTGDriver.instances[-1].reset_calls) == 1
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


def test_get_observation_uses_kinematics_target_frame_even_if_arm_reports_pose(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", ReportingArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    robot.connect()
    observation = robot.get_observation()

    assert observation["ee.x"] == pytest.approx(0.4)
    assert observation["ee.y"] == pytest.approx(0.1)
    assert observation["ee.z"] == pytest.approx(0.3)
    robot.disconnect()


def test_connect_falls_back_to_mock_gripper_when_hardware_unavailable(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    robot.connect()
    assert robot.is_connected
    assert robot._gripper_is_mock is True
    assert robot.get_observation()["gripper.pos"] == pytest.approx(1.0)
    robot.disconnect()


def test_connect_raises_when_mock_gripper_disabled(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            allow_mock_gripper=False,
            urdf_path="/tmp/fr3.urdf",
        )
    )

    with pytest.raises(RuntimeError, match="FR3 gripper hardware unavailable on /dev/ttyUSB80"):
        robot.connect()

    assert robot.is_connected is False
    assert robot._gripper is None


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
    deadline = time.perf_counter() + 0.2
    while (
        (
            len(robot._arm.set_joint_positions_calls) == 0
            or not np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)
        )
        and time.perf_counter() < deadline
    ):
        time.sleep(0.005)

    assert returned["gripper"] == pytest.approx(1.0)
    assert robot._gripper.set_position_calls[-1] == pytest.approx(1.0)
    assert len(robot._arm.set_joint_positions_calls) >= 1
    assert np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.6, 0.3, 0.5]))


def test_send_action_disabled_stops_at_current_pose_after_release(robot):
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
    inverse_call_count = len(robot._kinematics.inverse_calls)
    current_joints = robot._arm.get_joint_positions()

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

    assert len(robot._kinematics.inverse_calls) == inverse_call_count
    assert np.allclose(robot._last_command_pose, robot._kinematics.forward_pose)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, current_joints)


def test_send_action_disabled_after_active_holds_current_otg_command(robot):
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

    latched_otg_command = np.array([0.21, 0.22, 0.23, -0.94, 0.55, 1.18, -0.68], dtype=np.float64)
    with robot._otg_command_lock:
        robot._otg_command_joints = latched_otg_command.copy()

    second_action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    inverse_call_count = len(robot._kinematics.inverse_calls)
    robot.send_action(second_action)

    assert len(robot._kinematics.inverse_calls) == inverse_call_count
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, latched_otg_command)

    robot._arm.joint_positions = np.array([0.05, 0.1, 0.15, -1.1, 0.45, 1.05, -0.8], dtype=np.float64)
    robot.send_action(second_action)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, latched_otg_command)


def test_send_action_disabled_without_previous_command_holds_current_joints(robot):
    robot.connect()
    current_joints = robot._arm.get_joint_positions()
    action = {
        "enabled": False,
        "target_x": 0.4,
        "target_y": 0.4,
        "target_z": 0.4,
        "target_wx": 0.1,
        "target_wy": -0.1,
        "target_wz": 0.2,
        "gripper": 0.1,
    }

    robot.send_action(action)

    assert len(robot._kinematics.inverse_calls) == 0
    assert np.allclose(robot._last_command_pose, robot._kinematics.forward_pose)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, current_joints)


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


def test_send_action_clamps_target_delta_before_workspace(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            max_target_delta_pos=(0.01, 0.02, 0.03),
            max_target_delta_rot=(0.1, 0.2, 0.3),
        )
    )
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 1.0,
        "target_y": -1.0,
        "target_z": 1.0,
        "target_wx": 1.0,
        "target_wy": -1.0,
        "target_wz": 1.0,
        "gripper": 0.5,
    }
    robot.send_action(action)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.41, 0.08, 0.33]))
    rotvec = Rotation.from_matrix(desired_pose[:3, :3]).as_rotvec()
    assert np.allclose(rotvec, np.array([0.1, -0.2, 0.3]))
    robot.disconnect()


def test_send_action_runs_joint_targets_through_otg(monkeypatch):
    class SmoothingOTGDriver(DummyOTGDriver):
        def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
            current = np.asarray(current_joint_positions, dtype=np.float64)
            target = np.asarray(target_joint_positions, dtype=np.float64)
            self.step_calls.append((current.copy(), target.copy()))
            return target - 0.05

    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", SmoothingOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            use_otg=True,
        )
    )
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    robot.send_action(action)
    deadline = time.perf_counter() + 0.2
    expected_command = robot._kinematics.inverse_solution - 0.05
    while (
        (
            len(robot._arm.set_joint_positions_calls) == 0
            or not np.allclose(robot._arm.set_joint_positions_calls[-1], expected_command)
        )
        and time.perf_counter() < deadline
    ):
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 1
    assert np.allclose(robot._arm.set_joint_positions_calls[-1], expected_command)
    robot.disconnect()


def test_otg_continues_running_after_single_send_action(robot):
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    deadline = time.perf_counter() + 0.2
    while len(robot._otg.step_calls) < 2 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 2


def test_otg_sender_runs_faster_than_smoother(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            otg_control_frequency=20.0,
            otg_async_control_frequency=200.0,
        )
    )
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    deadline = time.perf_counter() + 0.25
    while len(robot._otg.step_calls) < 2 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 2
    assert len(robot._arm.set_joint_positions_calls) > len(robot._otg.step_calls)
    robot.disconnect()


def test_connect_cleans_up_partial_backends(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", FailingOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            use_otg=True,
        )
    )

    with pytest.raises(RuntimeError, match="otg init failed"):
        robot.connect()

    assert not robot.is_connected
    assert robot._arm is None
    assert robot._gripper is None
    assert DummyArmDriver.instances[-1].connected is False
    assert DummyGripperDriver.instances[-1].connected is False
