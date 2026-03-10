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

from functools import cached_property
import logging
import threading

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

from ..robot import Robot
from .backends import PandaPyArmDriver, PikaGripperHardwareDriver, PlacoKinematicsDriver, RuckigOTGDriver
from .config_franka_research3 import FrankaResearch3Config

logger = logging.getLogger(__name__)


class FrankaResearch3(Robot):
    config_class = FrankaResearch3Config
    name = "franka_research3"

    arm_driver_cls = PandaPyArmDriver
    gripper_driver_cls = PikaGripperHardwareDriver
    kinematics_driver_cls = PlacoKinematicsDriver
    otg_driver_cls = RuckigOTGDriver

    def __init__(self, config: FrankaResearch3Config):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._arm = None
        self._gripper = None
        self._kinematics = None
        self._otg = None
        self._is_connected = False
        self._reference_pose: np.ndarray | None = None
        self._last_command_pose: np.ndarray | None = None
        self._prev_enabled = False
        self._otg_target_joints: np.ndarray | None = None
        self._otg_target_lock = threading.Lock()
        self._otg_command_joints: np.ndarray | None = None
        self._otg_command_lock = threading.Lock()
        self._otg_thread: threading.Thread | None = None
        self._otg_sender_thread: threading.Thread | None = None
        self._otg_running = False
        self._otg_error: Exception | None = None

    @property
    def _joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(1, 8)]

    def _raise_if_otg_failed(self) -> None:
        if self._otg_error is not None:
            raise RuntimeError("FR3 OTG background loop failed.") from self._otg_error

    def _start_otg_loop(self, initial_joint_positions: np.ndarray) -> None:
        self._otg_target_joints = np.asarray(initial_joint_positions, dtype=np.float64).copy()
        self._otg_command_joints = np.asarray(initial_joint_positions, dtype=np.float64).copy()
        self._otg_error = None
        self._otg_running = True
        self._otg_thread = threading.Thread(
            target=self._otg_loop,
            daemon=True,
            name="FrankaResearch3OTGLoop",
        )
        self._otg_thread.start()
        self._otg_sender_thread = threading.Thread(
            target=self._otg_sender_loop,
            daemon=True,
            name="FrankaResearch3OTGSenderLoop",
        )
        self._otg_sender_thread.start()

    def _stop_otg_loop(self) -> None:
        self._otg_running = False
        if self._otg_thread is not None:
            self._otg_thread.join(timeout=1.0)
        self._otg_thread = None
        if self._otg_sender_thread is not None:
            self._otg_sender_thread.join(timeout=1.0)
        self._otg_sender_thread = None
        with self._otg_target_lock:
            self._otg_target_joints = None
        with self._otg_command_lock:
            self._otg_command_joints = None

    def _otg_loop(self) -> None:
        if self._otg is None or self._arm is None:
            return

        while self._otg_running:
            with self._otg_target_lock:
                target_joints_rad = (
                    None if self._otg_target_joints is None else self._otg_target_joints.copy()
                )
            with self._otg_command_lock:
                command_joints_rad = (
                    None if self._otg_command_joints is None else self._otg_command_joints.copy()
                )

            if target_joints_rad is None or command_joints_rad is None:
                precise_sleep(self.config.otg_dt)
                continue

            try:
                next_command_joints_rad = self._otg.step(command_joints_rad, target_joints_rad)
                with self._otg_command_lock:
                    self._otg_command_joints = np.asarray(next_command_joints_rad, dtype=np.float64).copy()
            except Exception as e:  # pragma: no cover - exercised with real hardware only
                self._otg_error = e
                self._otg_running = False
                logger.exception("FR3 OTG background loop failed")
                break

            precise_sleep(self.config.otg_dt)

    def _otg_sender_loop(self) -> None:
        if self._arm is None:
            return

        while self._otg_running:
            with self._otg_command_lock:
                command_joints_rad = (
                    None if self._otg_command_joints is None else self._otg_command_joints.copy()
                )

            if command_joints_rad is None:
                precise_sleep(self.config.otg_async_dt)
                continue

            try:
                self._arm.set_joint_positions(command_joints_rad)
            except Exception as e:  # pragma: no cover - exercised with real hardware only
                self._otg_error = e
                self._otg_running = False
                logger.exception("FR3 OTG sender loop failed")
                break

            precise_sleep(self.config.otg_async_dt)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        ee_features: dict[str, type] = {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.wx": float,
            "ee.wy": float,
            "ee.wz": float,
            "gripper.pos": float,
        }
        joint_features = {f"{joint}.pos": float for joint in self._joint_names}
        camera_features = {
            name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()
        }
        return {**ee_features, **joint_features, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        arm = self.arm_driver_cls(
            robot_ip=self.config.robot_ip,
            damping=self.config.damping,
            stiffness=self.config.stiffness,
            filter_coeff=self.config.filter_coeff,
        )
        gripper = self.gripper_driver_cls(
            serial_port=self.config.gripper_port,
            max_width_mm=self.config.gripper_max_width_mm,
        )
        kinematics = self.kinematics_driver_cls(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            joint_names=self.config.joint_names,
        )
        otg = None
        connected_cameras = []

        try:
            arm.connect()
            gripper.connect()
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
            if self.config.use_otg:
                otg = self.otg_driver_cls(
                    dof=len(self._joint_names),
                    dt=self.config.otg_dt,
                    max_velocity=list(self.config.otg_max_velocity),
                    max_acceleration=list(self.config.otg_max_acceleration),
                    max_jerk=list(self.config.otg_max_jerk),
                    min_position=list(self.config.otg_min_position),
                    max_position=list(self.config.otg_max_position),
                    synchronization=self.config.otg_synchronization,
                    sync_mode=self.config.otg_sync_mode,
                )
                otg.reset(np.asarray(arm.get_joint_positions(), dtype=np.float64))
        except Exception:
            for camera in reversed(connected_cameras):
                try:
                    camera.disconnect()
                except Exception:
                    pass
            try:
                gripper.disconnect()
            except Exception:
                pass
            try:
                arm.disconnect()
            except Exception:
                pass
            raise

        self._arm = arm
        self._gripper = gripper
        self._kinematics = kinematics
        self._otg = otg
        self._is_connected = True
        if self._otg is not None:
            self._start_otg_loop(np.asarray(arm.get_joint_positions(), dtype=np.float64))
        try:
            self.configure()
        except Exception:
            self.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_joint_positions(self) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")
        return np.asarray(self._arm.get_joint_positions(), dtype=np.float64)

    def _compute_ee_pose(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        if self._arm is None or self._kinematics is None:
            raise RuntimeError("Robot is not connected.")
        ee_pose = self._arm.get_ee_pose()
        if ee_pose is not None:
            return np.asarray(ee_pose, dtype=np.float64)
        return np.asarray(self._kinematics.forward_kinematics(joint_positions_rad), dtype=np.float64)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._raise_if_otg_failed()
        joint_positions_rad = self._read_joint_positions()
        ee_pose = self._compute_ee_pose(joint_positions_rad)
        ee_rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
        gripper_pos = float(self._gripper.get_position())

        observation: RobotObservation = {
            "ee.x": float(ee_pose[0, 3]),
            "ee.y": float(ee_pose[1, 3]),
            "ee.z": float(ee_pose[2, 3]),
            "ee.wx": float(ee_rotvec[0]),
            "ee.wy": float(ee_rotvec[1]),
            "ee.wz": float(ee_rotvec[2]),
            "gripper.pos": gripper_pos,
        }
        for index, joint_position in enumerate(joint_positions_rad, start=1):
            observation[f"joint_{index}.pos"] = float(joint_position)
        for camera_name, camera in self.cameras.items():
            observation[camera_name] = camera.read_latest()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._raise_if_otg_failed()
        joint_positions_rad = self._read_joint_positions()
        current_pose = self._compute_ee_pose(joint_positions_rad)

        enabled = bool(action["enabled"])
        if enabled:
            if not self._prev_enabled or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            delta_pos = np.array(
                [float(action["target_x"]), float(action["target_y"]), float(action["target_z"])],
                dtype=np.float64,
            )
            delta_rot = Rotation.from_rotvec(
                [float(action["target_wx"]), float(action["target_wy"]), float(action["target_wz"])]
            )

            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = self._reference_pose[:3, :3] @ delta_rot.as_matrix()
            desired_pose[:3, 3] = self._reference_pose[:3, 3] + delta_pos
            desired_pose[:3, 3] = np.clip(
                desired_pose[:3, 3],
                np.asarray(self.config.workspace_min, dtype=np.float64),
                np.asarray(self.config.workspace_max, dtype=np.float64),
            )
            self._last_command_pose = desired_pose.copy()
        else:
            if self._last_command_pose is None:
                self._last_command_pose = current_pose.copy()
            desired_pose = self._last_command_pose.copy()

        target_joints_rad = self._kinematics.inverse_kinematics(joint_positions_rad, desired_pose)
        if self._otg is not None:
            with self._otg_target_lock:
                self._otg_target_joints = np.asarray(target_joints_rad, dtype=np.float64).copy()
        else:
            self._arm.set_joint_positions(target_joints_rad)

        gripper_target = float(np.clip(action["gripper"], 0.0, 1.0))
        self._gripper.set_position(gripper_target)

        self._last_command_pose = desired_pose.copy()
        if enabled:
            self._reference_pose = desired_pose.copy()
        else:
            self._reference_pose = None
        self._prev_enabled = enabled
        return {
            "enabled": enabled,
            "target_x": float(action["target_x"]),
            "target_y": float(action["target_y"]),
            "target_z": float(action["target_z"]),
            "target_wx": float(action["target_wx"]),
            "target_wy": float(action["target_wy"]),
            "target_wz": float(action["target_wz"]),
            "gripper": gripper_target,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._stop_otg_loop()
            for camera in self.cameras.values():
                try:
                    camera.disconnect()
                except Exception:
                    pass
            if self._gripper is not None:
                try:
                    self._gripper.disconnect()
                except Exception:
                    pass
            if self._arm is not None:
                try:
                    self._arm.disconnect()
                except Exception:
                    pass
        finally:
            self._arm = None
            self._gripper = None
            self._kinematics = None
            self._otg = None
            self._is_connected = False
            self._reference_pose = None
            self._last_command_pose = None
            self._prev_enabled = False
            self._otg_error = None
