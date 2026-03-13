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
import logging
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


class JointOTGDriver(Protocol):
    def reset(self, current_joint_positions: np.ndarray) -> None: ...

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray: ...


def _silence_pika_logs() -> None:
    logging.getLogger("pika.gripper").setLevel(logging.WARNING)
    logging.getLogger("pika.serial_comm").setLevel(logging.WARNING)


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
        _silence_pika_logs()
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
class MockGripperDriver:
    initial_position: float = 1.0

    def __post_init__(self):
        self._position = float(np.clip(self.initial_position, 0.0, 1.0))
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_position(self) -> float:
        if not self._connected:
            raise RuntimeError("Gripper backend is not connected.")
        return self._position

    def set_position(self, normalized_position: float) -> None:
        if not self._connected:
            raise RuntimeError("Gripper backend is not connected.")
        self._position = float(np.clip(normalized_position, 0.0, 1.0))


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


@dataclass
class RuckigOTGDriver:
    dof: int
    dt: float
    max_velocity: list[float]
    max_acceleration: list[float]
    max_jerk: list[float]
    min_position: list[float] | None = None
    max_position: list[float] | None = None
    synchronization: bool = True
    sync_mode: str = "time"

    def __post_init__(self):
        try:
            from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization
        except Exception as e:  # pragma: no cover - exercised in docker / hardware runtime
            raise ImportError(
                "franka_research3 OTG requires ruckig in the runtime environment. "
                "Use the FR3 docker image or install ruckig manually."
            ) from e

        self._result = Result
        self._ruckig = Ruckig(self.dof, self.dt)
        self._input = InputParameter(self.dof)
        self._output = OutputParameter(self.dof)

        self._max_velocity = np.asarray(self.max_velocity, dtype=np.float64)
        self._max_acceleration = np.asarray(self.max_acceleration, dtype=np.float64)
        self._max_jerk = np.asarray(self.max_jerk, dtype=np.float64)
        self._min_position = (
            None if self.min_position is None else np.asarray(self.min_position, dtype=np.float64)
        )
        self._max_position = (
            None if self.max_position is None else np.asarray(self.max_position, dtype=np.float64)
        )
        self._current_velocity = np.zeros(self.dof, dtype=np.float64)
        self._current_acceleration = np.zeros(self.dof, dtype=np.float64)

        self._input.max_velocity = self._max_velocity.tolist()
        self._input.max_acceleration = self._max_acceleration.tolist()
        self._input.max_jerk = self._max_jerk.tolist()
        if self._min_position is not None and self._max_position is not None:
            self._input.min_position = self._min_position.tolist()
            self._input.max_position = self._max_position.tolist()

        if not self.synchronization or self.sync_mode.lower() == "none":
            self._input.synchronization = Synchronization.No
        elif self.sync_mode.lower() == "phase":
            self._input.synchronization = Synchronization.Phase
        else:
            self._input.synchronization = Synchronization.Time

    def reset(self, current_joint_positions: np.ndarray) -> None:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        if current.shape != (self.dof,):
            raise ValueError(f"Expected current_joint_positions shape {(self.dof,)}, got {current.shape}.")

        self._current_velocity.fill(0.0)
        self._current_acceleration.fill(0.0)
        self._input.current_position = current.tolist()
        self._input.current_velocity = self._current_velocity.tolist()
        self._input.current_acceleration = self._current_acceleration.tolist()
        self._input.target_position = current.tolist()
        self._input.target_velocity = [0.0] * self.dof
        self._input.target_acceleration = [0.0] * self.dof

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        target = np.asarray(target_joint_positions, dtype=np.float64)
        if current.shape != (self.dof,) or target.shape != (self.dof,):
            raise ValueError(
                f"Expected current and target shapes {(self.dof,)}, got {current.shape} and {target.shape}."
            )

        if self._min_position is not None and self._max_position is not None:
            target = np.clip(target, self._min_position, self._max_position)

        self._input.current_position = current.tolist()
        self._input.current_velocity = self._current_velocity.tolist()
        self._input.current_acceleration = self._current_acceleration.tolist()
        self._input.target_position = target.tolist()
        self._input.target_velocity = [0.0] * self.dof
        self._input.target_acceleration = [0.0] * self.dof

        result = self._ruckig.update(self._input, self._output)
        if result not in (self._result.Working, self._result.Finished):
            raise RuntimeError(f"Ruckig OTG update failed with result={result}.")

        new_position = np.asarray(self._output.new_position, dtype=np.float64)
        self._current_velocity = np.asarray(self._output.new_velocity, dtype=np.float64)
        self._current_acceleration = np.asarray(self._output.new_acceleration, dtype=np.float64)
        self._output.pass_to_input(self._input)
        return new_position
