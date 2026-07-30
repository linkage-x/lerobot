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
import time

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

from ..robot import Robot
from .backends import (
    DasGripperHardwareDriver,
    CoreneticGripperHardwareDriver,
    FrankaHandGripperHardwareDriver,
    HirolGaussianNewtonKinematicsDriver,
    HirolLMKinematicsDriver,
    MockGripperDriver,
    PandaPyArmDriver,
    PikaGripperHardwareDriver,
    PlacoKinematicsDriver,
    RuckigOTGDriver,
)
from .config_franka_research3 import FrankaResearch3Config
from .processor_franka_research3 import PREV_CMD_GRIPPER_KEY, PREV_CMD_POSITION_KEYS, PREV_CMD_ROTVEC_KEYS

logger = logging.getLogger(__name__)


class FrankaResearch3(Robot):
    config_class = FrankaResearch3Config
    name = "franka_research3"

    arm_driver_cls = PandaPyArmDriver
    gripper_driver_cls = PikaGripperHardwareDriver
    das_gripper_driver_cls = DasGripperHardwareDriver
    corenetic_gripper_driver_cls = CoreneticGripperHardwareDriver
    franka_hand_gripper_driver_cls = FrankaHandGripperHardwareDriver
    mock_gripper_driver_cls = MockGripperDriver
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
        self._gripper_is_mock = False
        self._reference_pose: np.ndarray | None = None
        self._last_command_pose: np.ndarray | None = None
        self._last_command_gripper: float | None = None
        self._hold_joint_target: np.ndarray | None = None
        self._prev_enabled = False
        self._otg_target_joints: np.ndarray | None = None
        self._otg_target_lock = threading.Lock()
        self._otg_command_joints: np.ndarray | None = None
        self._otg_command_lock = threading.Lock()
        self._otg_thread: threading.Thread | None = None
        self._otg_sender_thread: threading.Thread | None = None
        self._otg_running = False
        self._otg_error: Exception | None = None
        self._state_snapshot_lock = threading.Lock()
        self._last_observation_joint_positions_rad: np.ndarray | None = None
        self._last_observation_ee_pose: np.ndarray | None = None
        self._capture_timestamp_origin_s = time.perf_counter()

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
        prev_cmd_features: dict[str, type] = {
            **{key: float for key in PREV_CMD_POSITION_KEYS},
            **{key: float for key in PREV_CMD_ROTVEC_KEYS},
            PREV_CMD_GRIPPER_KEY: float,
        }
        joint_features = {f"{joint}.pos": float for joint in self._joint_names}
        camera_features = {
            name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()
        }
        tactile_features: dict[str, tuple[int, int]] = {}
        if self.config.das_tactile_valid_mask_path is not None and self.config.das_tactile_baseline_path is not None:
            tactile_features = {
                'observation.tactile.left_raw': (50, 10),
                'observation.tactile.right_raw': (50, 10),
                'observation.tactile.valid_mask': (50, 10),
                'observation.tactile.left_clean': (50, 10),
                'observation.tactile.right_clean': (50, 10),
            }
        return {**ee_features, **prev_cmd_features, **joint_features, **camera_features, **tactile_features}

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

    @property
    def capture_timestamp_feature_names(self) -> tuple[str, ...]:
        return (
            "fr3.arm.capture_timestamp_s",
            f"{self.config.gripper_backend}_gripper.capture_timestamp_s",
            *(f"camera.{name}.capture_timestamp_s" for name in self.cameras),
        )

    def reset_capture_timestamp_origin(self) -> None:
        self._capture_timestamp_origin_s = time.perf_counter()

    def _relative_capture_timestamp(self, timestamp_s: float) -> float:
        return float(timestamp_s - self._capture_timestamp_origin_s)

    def _make_gripper_driver(self):
        if self.config.gripper_backend == "mock":
            return self.mock_gripper_driver_cls(initial_position=1.0)
        if self.config.gripper_backend == "das":
            return self.das_gripper_driver_cls(
                serial_port=self.config.gripper_port,
                gen_con_sdk_path=self.config.gen_con_sdk_path,
                baudrate=self.config.das_baudrate,
                update_frequency_hz=self.config.das_update_frequency_hz,
                tactile_frequency_hz=self.config.das_tactile_frequency_hz,
                tactile_valid_mask_path=self.config.das_tactile_valid_mask_path,
                tactile_baseline_path=self.config.das_tactile_baseline_path,
                tactile_timeout_s=self.config.das_tactile_timeout_s,
                min_distance_m=self.config.das_min_distance_m,
                max_distance_m=self.config.das_max_distance_m,
                grasp_threshold_m=self.config.das_grasp_threshold_m,
                initial_position=self.config.das_initial_position,
                command_rate_limit_hz=self.config.gripper_command_rate_limit_hz,
                command_deadband_m=self.config.gripper_command_deadband_mm / 1000.0,
            )
        if self.config.gripper_backend == "franka_hand":
            return self.franka_hand_gripper_driver_cls(
                robot_ip=self.config.robot_ip,
                command_rate_limit_hz=self.config.gripper_command_rate_limit_hz,
                command_deadband_m=self.config.gripper_command_deadband_mm / 1000.0,
            )
        if self.config.gripper_backend == "corenetic":
            return self.corenetic_gripper_driver_cls(
                bind_ip=self.config.corenetic_bind_ip,
                bind_port=self.config.corenetic_bind_port,
                remote_ip=self.config.corenetic_remote_ip,
                remote_port=self.config.corenetic_remote_port,
                sdk_dir=self.config.corenetic_sdk_dir,
                urdf_relpath=self.config.corenetic_urdf_relpath,
                max_width_m=self.config.gripper_max_width_mm / 1000.0,
                poll_interval_s=self.config.corenetic_poll_interval_s,
                stale_threshold_s=self.config.corenetic_stale_threshold_s,
                connect_timeout_s=self.config.corenetic_connect_timeout_s,
                command_rate_limit_hz=self.config.gripper_command_rate_limit_hz,
                command_deadband_m=self.config.gripper_command_deadband_mm / 1000.0,
                release_mode_on_disconnect=self.config.corenetic_release_mode_on_disconnect,
            )
        return self.gripper_driver_cls(
            serial_port=self.config.gripper_port,
            max_width_mm=self.config.gripper_max_width_mm,
            command_rate_limit_hz=self.config.gripper_command_rate_limit_hz,
            command_deadband_mm=self.config.gripper_command_deadband_mm,
        )

    def _make_kinematics_driver(self):
        kwargs = {
            "urdf_path": self.config.urdf_path,
            "target_frame_name": self.config.target_frame_name,
            "joint_names": self.config.joint_names,
        }
        if self.kinematics_driver_cls is not PlacoKinematicsDriver:
            return self.kinematics_driver_cls(**kwargs)
        if self.config.ik_solver == "hirol_lm":
            return HirolLMKinematicsDriver(
                **kwargs,
                tolerance=self.config.ik_tolerance,
                max_iterations=self.config.ik_max_iterations,
            )
        if self.config.ik_solver == "hirol_gaussian_newton":
            return HirolGaussianNewtonKinematicsDriver(
                **kwargs,
                tolerance=self.config.ik_tolerance,
                max_iterations=self.config.ik_max_iterations,
            )
        return self.kinematics_driver_cls(**kwargs)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        arm = self.arm_driver_cls(
            robot_ip=self.config.robot_ip,
            damping=self.config.damping,
            stiffness=self.config.stiffness,
            filter_coeff=self.config.filter_coeff,
        )
        gripper = None
        kinematics = self._make_kinematics_driver()
        otg = None
        connected_cameras = []

        try:
            arm.connect()
            try:
                gripper = self._make_gripper_driver()
                gripper.connect()
                self._gripper_is_mock = self.config.gripper_backend == "mock"
            except Exception as gripper_error:
                if not self.config.allow_mock_gripper:
                    raise RuntimeError(
                        f"FR3 gripper hardware unavailable on {self.config.gripper_port}."
                    ) from gripper_error
                logger.warning(
                    "FR3 gripper hardware unavailable on %s; falling back to mock gripper: %s",
                    self.config.gripper_port,
                    gripper_error,
                )
                if gripper is not None:
                    try:
                        gripper.disconnect()
                    except Exception:
                        pass
                gripper = self.mock_gripper_driver_cls(initial_position=1.0)
                gripper.connect()
                self._gripper_is_mock = True
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
            if gripper is not None:
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
        self.reset_capture_timestamp_origin()
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

    def _reset_teleop_state(self) -> None:
        self._reference_pose = None
        self._last_command_pose = None
        self._last_command_gripper = None
        self._hold_joint_target = None
        self._prev_enabled = False

    def _cache_observation_state_snapshot(
        self,
        joint_positions_rad: np.ndarray,
        ee_pose: np.ndarray,
    ) -> None:
        with self._state_snapshot_lock:
            self._last_observation_joint_positions_rad = np.asarray(joint_positions_rad, dtype=np.float64).copy()
            self._last_observation_ee_pose = np.asarray(ee_pose, dtype=np.float64).copy()

    def _consume_observation_state_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._state_snapshot_lock:
            joint_positions_rad = self._last_observation_joint_positions_rad
            ee_pose = self._last_observation_ee_pose
            self._last_observation_joint_positions_rad = None
            self._last_observation_ee_pose = None
        if joint_positions_rad is not None:
            joint_positions_rad = np.asarray(joint_positions_rad, dtype=np.float64).copy()
        if ee_pose is not None:
            ee_pose = np.asarray(ee_pose, dtype=np.float64).copy()
        return joint_positions_rad, ee_pose

    def _clear_observation_state_snapshot(self) -> None:
        with self._state_snapshot_lock:
            self._last_observation_joint_positions_rad = None
            self._last_observation_ee_pose = None

    def _read_joint_positions(self) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")
        return np.asarray(self._arm.get_joint_positions(), dtype=np.float64)

    def _get_release_hold_joint_target(self, current_joint_positions_rad: np.ndarray) -> np.ndarray:
        if self._prev_enabled and self._otg is not None:
            with self._otg_command_lock:
                if self._otg_command_joints is not None:
                    return np.asarray(self._otg_command_joints, dtype=np.float64).copy()
            with self._otg_target_lock:
                if self._otg_target_joints is not None:
                    return np.asarray(self._otg_target_joints, dtype=np.float64).copy()
        return np.asarray(current_joint_positions_rad, dtype=np.float64).copy()

    def _compute_ee_pose(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        if self._kinematics is None:
            raise RuntimeError("Robot is not connected.")
        return np.asarray(self._kinematics.forward_kinematics(joint_positions_rad), dtype=np.float64)

    def _make_pose_from_absolute_action(self, action: RobotAction) -> np.ndarray:
        desired_pose = np.eye(4, dtype=np.float64)
        desired_pose[:3, 3] = np.array(
            [float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])],
            dtype=np.float64,
        )
        desired_pose[:3, :3] = Rotation.from_rotvec(
            [float(action["ee.wx"]), float(action["ee.wy"]), float(action["ee.wz"])]
        ).as_matrix()
        desired_pose[:3, 3] = np.clip(
            desired_pose[:3, 3],
            np.asarray(self.config.workspace_min, dtype=np.float64),
            np.asarray(self.config.workspace_max, dtype=np.float64),
        )
        return desired_pose

    def _normalize_gripper_command(self, gripper_target: float) -> float:
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))
        if self.config.gripper_backend == "franka_hand":
            return 1.0 if gripper_target >= 0.5 else 0.0
        return gripper_target

    def _make_prev_command_observation(
        self,
        *,
        current_ee_pose: np.ndarray,
        current_gripper_pos: float,
    ) -> RobotObservation:
        previous_command_pose = current_ee_pose if self._last_command_pose is None else self._last_command_pose
        previous_command_rotvec = Rotation.from_matrix(previous_command_pose[:3, :3]).as_rotvec()
        previous_command_gripper = (
            current_gripper_pos if self._last_command_gripper is None else self._last_command_gripper
        )
        return {
            "prev_cmd.ee.x": float(previous_command_pose[0, 3]),
            "prev_cmd.ee.y": float(previous_command_pose[1, 3]),
            "prev_cmd.ee.z": float(previous_command_pose[2, 3]),
            "prev_cmd.ee.wx": float(previous_command_rotvec[0]),
            "prev_cmd.ee.wy": float(previous_command_rotvec[1]),
            "prev_cmd.ee.wz": float(previous_command_rotvec[2]),
            PREV_CMD_GRIPPER_KEY: float(previous_command_gripper),
        }

    @check_if_not_connected
    def get_observation(self, *, include_cameras: bool = True) -> RobotObservation:
        self._raise_if_otg_failed()
        joint_positions_rad = self._read_joint_positions()
        arm_capture_timestamp_s = time.perf_counter()
        ee_pose = self._compute_ee_pose(joint_positions_rad)
        self._cache_observation_state_snapshot(joint_positions_rad, ee_pose)
        ee_rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
        gripper_pos = float(self._gripper.get_position())
        gripper_capture_timestamp_s = time.perf_counter()

        observation: RobotObservation = {
            "ee.x": float(ee_pose[0, 3]),
            "ee.y": float(ee_pose[1, 3]),
            "ee.z": float(ee_pose[2, 3]),
            "ee.wx": float(ee_rotvec[0]),
            "ee.wy": float(ee_rotvec[1]),
            "ee.wz": float(ee_rotvec[2]),
            "gripper.pos": gripper_pos,
            "fr3.arm.capture_timestamp_s": self._relative_capture_timestamp(arm_capture_timestamp_s),
            f"{self.config.gripper_backend}_gripper.capture_timestamp_s": self._relative_capture_timestamp(
                gripper_capture_timestamp_s
            ),
            **self._make_prev_command_observation(current_ee_pose=ee_pose, current_gripper_pos=gripper_pos),
        }
        for index, joint_position in enumerate(joint_positions_rad, start=1):
            observation[f"joint_{index}.pos"] = float(joint_position)
        get_tactile_observation = getattr(self._gripper, 'get_tactile_observation', None)
        if callable(get_tactile_observation):
            observation.update(get_tactile_observation())
        if include_cameras:
            latest_samples: dict[str, tuple[np.ndarray, float]] = {}
            for camera_name, camera in self.cameras.items():
                read_latest_with_timestamp = getattr(camera, "read_latest_with_timestamp", None)
                if callable(read_latest_with_timestamp):
                    latest_samples[camera_name] = read_latest_with_timestamp(
                        max_age_ms=self.config.camera_max_age_ms
                    )
                else:
                    try:
                        frame = camera.read_latest(max_age_ms=self.config.camera_max_age_ms)
                    except TypeError as exc:
                        if "max_age_ms" not in str(exc):
                            raise
                        frame = camera.read_latest()
                    latest_samples[camera_name] = (
                        frame,
                        float(getattr(camera, "latest_timestamp", time.perf_counter())),
                    )

            if latest_samples:
                # Anchor every camera on the oldest of their latest frames and take each
                # camera's frame closest to that instant.
                #
                # Taking each camera's own newest frame instead is tempting -- it measures 8.5 ms
                # behind the arm read rather than 25 ms -- but it was tried and it breaks
                # recording. Nothing then bounds how far apart the cameras' newest frames are:
                # each camera's background thread delivers independently, and one falling a whole
                # period behind puts the pair 25.1 ms apart, past any guard worth having. It
                # aborted an episode after 21 frames on hardware. Anchoring is what bounds the
                # spread, and that bound is why the guard below can stay tight.
                reference_timestamp_s = min(timestamp for _frame, timestamp in latest_samples.values())
                selected_timestamps: list[float] = []
                for camera_name, camera in self.cameras.items():
                    read_closest = getattr(camera, "read_closest", None)
                    if callable(read_closest):
                        frame, timestamp_s = read_closest(
                            reference_timestamp_s,
                            max_age_ms=self.config.camera_max_age_ms,
                        )
                    else:
                        frame, timestamp_s = latest_samples[camera_name]
                    observation[camera_name] = frame
                    observation[f"camera.{camera_name}.capture_timestamp_s"] = (
                        self._relative_capture_timestamp(timestamp_s)
                    )
                    selected_timestamps.append(timestamp_s)

                camera_skew_ms = (max(selected_timestamps) - min(selected_timestamps)) * 1e3
                if camera_skew_ms > self.config.camera_max_skew_ms:
                    raise RuntimeError(
                        f"FR3 camera skew {camera_skew_ms:.1f} ms exceeds "
                        f"camera_max_skew_ms={self.config.camera_max_skew_ms:.1f}."
                    )
        return observation

    @check_if_not_connected
    def move_to_start(self) -> None:
        self._raise_if_otg_failed()
        self._clear_observation_state_snapshot()
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")
        move_to_start = getattr(self._arm, "move_to_start", None)
        if not callable(move_to_start):
            raise RuntimeError("FR3 arm backend does not support move_to_start().")

        otg_enabled = self._otg is not None
        fallback_joint_positions_rad = self._read_joint_positions()
        if otg_enabled:
            self._stop_otg_loop()

        moved_joint_positions_rad = fallback_joint_positions_rad
        try:
            move_to_start()
            moved_joint_positions_rad = self._read_joint_positions()
        finally:
            self._reset_teleop_state()
            if otg_enabled:
                self._otg.reset(moved_joint_positions_rad)
                self._start_otg_loop(moved_joint_positions_rad)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._raise_if_otg_failed()
        joint_positions_rad, current_pose = self._consume_observation_state_snapshot()
        if joint_positions_rad is None:
            joint_positions_rad = self._read_joint_positions()
        hold_current_joints = False
        if all(key in action for key in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")):
            desired_pose = self._make_pose_from_absolute_action(action)
            enabled = True
            self._hold_joint_target = None
            self._reference_pose = None
            self._prev_enabled = False
        else:
            if current_pose is None:
                current_pose = self._compute_ee_pose(joint_positions_rad)
            enabled = bool(action["enabled"])
            if enabled:
                self._hold_joint_target = None
                if not self._prev_enabled or self._reference_pose is None:
                    self._reference_pose = current_pose.copy()

                delta_pos = np.array(
                    [float(action["target_x"]), float(action["target_y"]), float(action["target_z"])],
                    dtype=np.float64,
                )
                if self.config.max_target_delta_pos is not None:
                    delta_pos = np.clip(
                        delta_pos,
                        -np.asarray(self.config.max_target_delta_pos, dtype=np.float64),
                        np.asarray(self.config.max_target_delta_pos, dtype=np.float64),
                    )
                delta_rot = Rotation.from_rotvec(
                    [float(action["target_wx"]), float(action["target_wy"]), float(action["target_wz"])]
                )
                if self.config.max_target_delta_rot is not None:
                    clamped_delta_rot = np.clip(
                        delta_rot.as_rotvec(),
                        -np.asarray(self.config.max_target_delta_rot, dtype=np.float64),
                        np.asarray(self.config.max_target_delta_rot, dtype=np.float64),
                    )
                    delta_rot = Rotation.from_rotvec(clamped_delta_rot)

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
                if self._hold_joint_target is None:
                    self._hold_joint_target = self._get_release_hold_joint_target(joint_positions_rad)
                target_joints_rad = self._hold_joint_target.copy()
                desired_pose = self._compute_ee_pose(target_joints_rad)
                hold_current_joints = True

        if hold_current_joints:
            target_joints_rad = np.asarray(target_joints_rad, dtype=np.float64).copy()
        else:
            target_joints_rad = self._kinematics.inverse_kinematics(joint_positions_rad, desired_pose)
        if self._otg is not None:
            with self._otg_target_lock:
                self._otg_target_joints = np.asarray(target_joints_rad, dtype=np.float64).copy()
        else:
            self._arm.set_joint_positions(target_joints_rad)

        gripper_key = "gripper.pos" if "gripper.pos" in action else "gripper"
        gripper_target = self._normalize_gripper_command(float(action[gripper_key]))
        self._gripper.set_position(gripper_target)

        self._last_command_pose = desired_pose.copy()
        self._last_command_gripper = gripper_target
        if not all(key in action for key in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")):
            if enabled:
                self._hold_joint_target = None
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

        return {
            "ee.x": float(desired_pose[0, 3]),
            "ee.y": float(desired_pose[1, 3]),
            "ee.z": float(desired_pose[2, 3]),
            "ee.wx": float(action["ee.wx"]),
            "ee.wy": float(action["ee.wy"]),
            "ee.wz": float(action["ee.wz"]),
            "gripper.pos": gripper_target,
        }

    @check_if_not_connected
    def send_joint_positions(
        self,
        joint_positions_rad: np.ndarray,
        *,
        gripper_pos: float | None = None,
    ) -> np.ndarray:
        self._raise_if_otg_failed()
        self._clear_observation_state_snapshot()
        target_joints_rad = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
        if target_joints_rad.shape != (7,):
            raise ValueError(f"Expected 7 joint targets, got shape {target_joints_rad.shape}.")
        if self._otg is not None:
            with self._otg_target_lock:
                self._otg_target_joints = target_joints_rad.copy()
        else:
            if self._arm is None:
                raise RuntimeError("Arm backend is not connected.")
            self._arm.set_joint_positions(target_joints_rad)
        if gripper_pos is not None:
            normalized_gripper_pos = self._normalize_gripper_command(float(gripper_pos))
            self._gripper.set_position(normalized_gripper_pos)
            self._last_command_gripper = normalized_gripper_pos
        self._hold_joint_target = None
        self._reference_pose = None
        self._prev_enabled = False
        self._last_command_pose = self._compute_ee_pose(target_joints_rad)
        return target_joints_rad.copy()

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._clear_observation_state_snapshot()
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
            self._gripper_is_mock = False
            self._reset_teleop_state()
            self._otg_error = None
