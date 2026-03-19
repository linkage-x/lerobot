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
import importlib
import logging
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Protocol

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class ArmDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions(self) -> np.ndarray: ...

    def get_ee_pose(self) -> np.ndarray | None: ...

    def set_joint_positions(self, joint_positions: np.ndarray) -> None: ...

    def move_to_start(self) -> None: ...


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


_DAS_SDK_ENV_VAR = "GEN_CON_SDK_HOME"
_DEFAULT_DAS_SDK_ROOTS = (
    Path("/opt/dependencies/gen_con_sdk_python_release"),
    Path("/opt/gen_con_sdk_python_release"),
)
_DAS_SDK_MODULE_CANDIDATES = (
    "gen_controller_sdk_python",
    "gen_con_sdk_python_release",
    "gen_con_sdk_python_release.gen_controller_sdk_python",
    "dependencies.gen_con_sdk_python_release.gen_controller_sdk_python",
)


def _resolve_das_databus_cls(gen_con_sdk_path: str | None):
    search_roots: list[Path] = []
    for raw_path in (gen_con_sdk_path, os.environ.get(_DAS_SDK_ENV_VAR)):
        if raw_path:
            path = Path(raw_path).expanduser()
            search_roots.extend([path, path.parent])
    for root in _DEFAULT_DAS_SDK_ROOTS:
        search_roots.extend([root, root.parent])

    seen_paths: set[str] = set()
    for root in search_roots:
        root_str = str(root)
        if not root_str or root_str in seen_paths or not root.exists():
            continue
        seen_paths.add(root_str)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    import_errors: list[str] = []
    for module_name in _DAS_SDK_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        databus_cls = getattr(module, "DataBus", None)
        if databus_cls is not None:
            return databus_cls

    raise ImportError(
        "franka_research3 DAS gripper backend requires the gen_controller_sdk_python DataBus. "
        f"Set {_DAS_SDK_ENV_VAR} to the cloned SDK root if it is not installed in the default image. "
        f"Import attempts: {import_errors}"
    )


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
        self._start_controller()

    def _start_controller(self) -> None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        self._controller = self._controllers.JointPosition()
        if self.damping is not None:
            self._controller.set_damping(self.damping)
        if self.stiffness is not None:
            self._controller.set_stiffness(self.stiffness)
        if self.filter_coeff is not None:
            self._controller.set_filter(self.filter_coeff)
        current_joint_positions = np.asarray(self._robot.get_state().q, dtype=np.float64)
        self._controller.set_control(current_joint_positions)
        self._robot.start_controller(self._controller)

    def _stop_controller(self) -> None:
        if self._robot is not None and self._controller is not None:
            self._robot.stop_controller()
            self._controller = None

    def disconnect(self) -> None:
        if self._robot is not None:
            self._stop_controller()
            self._robot = None

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

    def move_to_start(self) -> None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        controller_was_running = self._controller is not None
        if controller_was_running:
            self._stop_controller()
        try:
            self._robot.move_to_start()
        finally:
            if controller_was_running:
                self._start_controller()


@dataclass
class PikaGripperHardwareDriver:
    serial_port: str
    max_width_mm: float = 90.0
    command_rate_limit_hz: float | None = 15.0
    command_deadband_mm: float = 0.5

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
        self._last_command_width_mm: float | None = None
        self._last_command_time_s: float | None = None
        self._pending_command_width_mm: float | None = None

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
        self._last_command_width_mm = None
        self._last_command_time_s = None
        self._pending_command_width_mm = None

    def get_position(self) -> float:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        width_mm = float(self._gripper.get_gripper_distance())
        return float(np.clip(width_mm / self.max_width_mm, 0.0, 1.0))

    def set_position(self, normalized_position: float) -> None:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        target_width_mm = float(np.clip(normalized_position, 0.0, 1.0) * self.max_width_mm)
        if (
            self._last_command_width_mm is not None
            and abs(target_width_mm - self._last_command_width_mm) < self.command_deadband_mm
        ):
            self._pending_command_width_mm = None
            return

        self._pending_command_width_mm = target_width_mm

        now = time.perf_counter()
        if self.command_rate_limit_hz is not None and self._last_command_time_s is not None:
            min_interval_s = 1.0 / self.command_rate_limit_hz
            if now - self._last_command_time_s < min_interval_s:
                return

        pending_width_mm = self._pending_command_width_mm
        if pending_width_mm is None:
            return

        # Collapse repeated control-loop writes into the newest target before touching the serial link.
        self._gripper.set_gripper_distance(pending_width_mm)
        self._last_command_width_mm = pending_width_mm
        self._last_command_time_s = now
        self._pending_command_width_mm = None


@dataclass
class DasGripperHardwareDriver:
    serial_port: str
    gen_con_sdk_path: str | None = None
    baudrate: int = 921600
    update_frequency_hz: float = 50.0
    min_distance_m: float = 0.0
    max_distance_m: float = 0.103
    grasp_threshold_m: float = 0.002
    initial_position: float = 1.0
    command_rate_limit_hz: float | None = 15.0
    command_deadband_m: float = 0.0005

    def __post_init__(self):
        if self.max_distance_m <= self.min_distance_m:
            raise ValueError("Das gripper max_distance_m must be greater than min_distance_m.")
        if self.update_frequency_hz <= 0:
            raise ValueError("Das gripper update_frequency_hz must be positive.")
        if self.baudrate <= 0:
            raise ValueError("Das gripper baudrate must be positive.")
        if self.command_deadband_m < 0:
            raise ValueError("Das gripper command_deadband_m must be non-negative.")
        self._databus_cls = _resolve_das_databus_cls(self.gen_con_sdk_path)
        self._databus = None
        self._lock = threading.Lock()
        self._gripper_state_updated = False
        self._position_m: float | None = None
        self._target_distance_m: float | None = None
        self._last_command_distance_m: float | None = None
        self._last_command_time_s: float | None = None
        self._pending_command_distance_m: float | None = None

    def connect(self) -> None:
        self._databus = self._databus_cls(
            tty_port=self.serial_port,
            baudrate=self.baudrate,
            encoder_freq=self.update_frequency_hz,
            encoder_callback=self._encoder_callback,
        )
        initial_distance_m = self._scale_to_distance(self.initial_position)
        self._databus.set_target_distance(initial_distance_m)
        self._target_distance_m = initial_distance_m
        self._last_command_distance_m = initial_distance_m

        deadline = time.perf_counter() + 2.0
        while not self._gripper_state_updated:
            if time.perf_counter() >= deadline:
                raise TimeoutError("DasController did not receive encoder data within timeout.")
            time.sleep(0.001)

    def disconnect(self) -> None:
        if self._databus is not None:
            stop = getattr(self._databus, "stop", None)
            if callable(stop):
                stop()
            self._databus = None
        self._gripper_state_updated = False
        self._position_m = None
        self._target_distance_m = None
        self._last_command_distance_m = None
        self._last_command_time_s = None
        self._pending_command_distance_m = None

    def _encoder_callback(self, record_data: bytes) -> None:
        try:
            distance_m = float(struct.unpack(">f", record_data)[0])
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to parse DAS encoder update: %s", exc)
            return

        with self._lock:
            self._position_m = float(np.clip(distance_m, self.min_distance_m, self.max_distance_m))
            if self._target_distance_m is not None and self.grasp_threshold_m > 0:
                _ = self._position_m > (self._target_distance_m + self.grasp_threshold_m)
        self._gripper_state_updated = True

    def _scale_to_distance(self, normalized_position: float) -> float:
        normalized = float(np.clip(normalized_position, 0.0, 1.0))
        return self.min_distance_m + normalized * (self.max_distance_m - self.min_distance_m)

    def get_position(self) -> float:
        if self._databus is None:
            raise RuntimeError("Gripper backend is not connected.")
        with self._lock:
            distance_m = self._position_m
        if distance_m is None:
            distance_m = self._target_distance_m
        if distance_m is None:
            raise RuntimeError("Das gripper backend has not received any position update yet.")
        span = self.max_distance_m - self.min_distance_m
        if span <= 0:
            return 0.0
        return float(np.clip((distance_m - self.min_distance_m) / span, 0.0, 1.0))

    def set_position(self, normalized_position: float) -> None:
        if self._databus is None:
            raise RuntimeError("Gripper backend is not connected.")

        target_distance_m = self._scale_to_distance(normalized_position)
        if (
            self._last_command_distance_m is not None
            and abs(target_distance_m - self._last_command_distance_m) < self.command_deadband_m
        ):
            self._pending_command_distance_m = None
            return

        self._pending_command_distance_m = target_distance_m
        now = time.perf_counter()
        if self.command_rate_limit_hz is not None and self._last_command_time_s is not None:
            min_interval_s = 1.0 / self.command_rate_limit_hz
            if now - self._last_command_time_s < min_interval_s:
                return

        pending_distance_m = self._pending_command_distance_m
        if pending_distance_m is None:
            return

        self._databus.set_target_distance(pending_distance_m)
        self._target_distance_m = pending_distance_m
        self._last_command_distance_m = pending_distance_m
        self._last_command_time_s = now
        self._pending_command_distance_m = None


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
