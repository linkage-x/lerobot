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

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class ArmDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions(self) -> np.ndarray: ...

    def get_ee_pose(self) -> np.ndarray | None: ...

    def set_joint_positions(self, joint_positions: np.ndarray) -> None: ...


class GripperDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_position(self) -> float: ...

    def set_position(self, normalized_position: float) -> None: ...


class KinematicsDriver(Protocol):
    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray: ...


@dataclass
class PandaPyArmDriver:
    robot_ip: str
    damping: list[float] | None = None
    stiffness: list[float] | None = None
    filter_coeff: float | None = None

    def __post_init__(self):
        try:
            from panda_py import Panda, controllers
        except Exception as e:  # pragma: no cover - exercised with real hardware only
            raise ImportError(
                "franka_research3 requires panda_py for hardware arm control. "
                "Install panda_py in the runtime environment to use the FR3 hardware backend."
            ) from e

        self._panda_cls = Panda
        self._controllers = controllers
        self._robot = None
        self._controller = None

    def connect(self) -> None:
        self._robot = self._panda_cls(self.robot_ip)
        self._controller = self._controllers.JointPosition()
        if self.damping is not None:
            self._controller.set_damping(self.damping)
        if self.stiffness is not None:
            self._controller.set_stiffness(self.stiffness)
        if self.filter_coeff is not None:
            self._controller.set_filter(self.filter_coeff)
        self._robot.start_controller(self._controller)

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.stop_controller()
            self._robot = None
            self._controller = None

    def get_joint_positions(self) -> np.ndarray:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        return np.asarray(self._robot.get_state().q, dtype=np.float64)

    def get_ee_pose(self) -> np.ndarray | None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        pose = np.asarray(self._robot.get_pose(), dtype=np.float64)
        if pose.shape == (4, 4):
            return pose
        if pose.size == 16:
            return pose.reshape(4, 4)
        return None

    def set_joint_positions(self, joint_positions: np.ndarray) -> None:
        if self._controller is None:
            raise RuntimeError("Arm backend is not connected.")
        self._controller.set_control(np.asarray(joint_positions, dtype=np.float64))


@dataclass
class PikaGripperHardwareDriver:
    serial_port: str
    max_width_mm: float = 90.0

    def __post_init__(self):
        try:
            from pika.gripper import Gripper
        except Exception as e:  # pragma: no cover - exercised with real hardware only
            raise ImportError(
                "franka_research3 requires pika.gripper for hardware gripper control. "
                "Install the Pika gripper SDK in the runtime environment to use this backend."
            ) from e

        self._gripper_cls = Gripper
        self._gripper = None

    def connect(self) -> None:
        self._gripper = self._gripper_cls(self.serial_port)
        if not self._gripper.connect():
            raise ConnectionError(f"Could not connect to Pika gripper on {self.serial_port}.")
        if not self._gripper.enable():
            raise ConnectionError("Could not enable the Pika gripper.")

    def disconnect(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.disable()
            finally:
                self._gripper.disconnect()
                self._gripper = None

    def get_position(self) -> float:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        width_mm = float(self._gripper.get_gripper_distance())
        return float(np.clip(width_mm / self.max_width_mm, 0.0, 1.0))

    def set_position(self, normalized_position: float) -> None:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        target_width_mm = float(np.clip(normalized_position, 0.0, 1.0) * self.max_width_mm)
        self._gripper.set_gripper_distance(target_width_mm)


@dataclass
class PlacoKinematicsDriver:
    urdf_path: str
    target_frame_name: str
    joint_names: list[str]

    def __post_init__(self):
        self._kinematics = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name=self.target_frame_name,
            joint_names=self.joint_names,
        )

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        joint_positions_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64))
        return self._kinematics.forward_kinematics(joint_positions_deg)

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
        current_joint_positions_deg = np.rad2deg(np.asarray(current_joint_positions_rad, dtype=np.float64))
        solution_deg = self._kinematics.inverse_kinematics(current_joint_positions_deg, desired_pose)
        return np.deg2rad(np.asarray(solution_deg, dtype=np.float64))
