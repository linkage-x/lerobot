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

from collections.abc import Iterator, Sequence
import contextlib
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

# How far the tool point may sit from where it was commanded before the arm counts as stalled.
#
# Running out of reach is not an error anywhere in this stack. HirolLMKinematicsDriver clips its
# Newton step to the URDF joint limits and returns the joints it already had, so an unreachable
# target becomes "hold position": no exception, no log line, and the FR3 -- which never receives an
# illegal command -- keeps its status light green. The operator sees an arm that has simply stopped
# moving in one direction, with the SpaceMouse still deflected. That is what this detects.
#
# Deliberately not a config field. It is a diagnostic threshold, not a control parameter, and the
# only reason to raise it is to stop hearing about a stall that is still happening. 5 mm is five
# times max_target_delta_pos, so a single step that IK solves imperfectly stays quiet.
_REACH_STALL_TOLERANCE_M = 0.005
# A stall lasts as long as the operator keeps pushing, and every line the recorder prints becomes
# the GUI's status message. Warn on the leading edge, then at this interval, then once on release.
_REACH_STALL_REWARN_S = 2.0


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
        # Distance between the last commanded tool point and the one the IK solution actually
        # realises. 0.0 whenever the arm is following its command.
        self._reach_stall_error_m: float = 0.0
        self._reach_stall_since_s: float | None = None
        self._reach_stall_last_warned_s: float = 0.0
        self._state_snapshot_lock = threading.Lock()
        self._last_observation_joint_positions_rad: np.ndarray | None = None
        self._last_observation_ee_pose: np.ndarray | None = None
        self._capture_timestamp_origin_s = time.perf_counter()
        # What the config asked for before any Set Home capture overwrote it, so the capture can be
        # undone. `None` is a meaningful value here -- it means "keep the arm backend's own start
        # pose" -- so it is preserved rather than normalised into a joint vector.
        self._default_start_joint_positions: tuple[float, ...] | None = (
            None
            if config.start_joint_positions is None
            else tuple(float(value) for value in config.start_joint_positions)
        )

    @property
    def _joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(1, 8)]

    def _raise_if_otg_failed(self) -> None:
        if self._otg_error is not None:
            raise RuntimeError("FR3 OTG background loop failed.") from self._otg_error

    def _configured_start_joint_positions(self) -> np.ndarray | None:
        if self.config.start_joint_positions is None:
            return None
        target = np.asarray(self.config.start_joint_positions, dtype=np.float64)
        expected_shape = (len(self.config.joint_names),)
        if target.shape != expected_shape:
            raise RuntimeError(f"Configured FR3 start pose shape must be {expected_shape}, got {target.shape}.")
        return target

    def capture_current_start_joint_positions(self, *, require_cached: bool = False) -> tuple[float, ...]:
        """Use the latest observed joint state as the configured return-to-start pose."""
        with self._state_snapshot_lock:
            cached_joint_positions_rad = self._last_observation_joint_positions_rad
            joint_positions_rad = (
                None
                if cached_joint_positions_rad is None
                else np.asarray(cached_joint_positions_rad, dtype=np.float64).copy()
            )
        if joint_positions_rad is None:
            if require_cached:
                raise RuntimeError("No cached FR3 joint observation is available yet.")
            joint_positions_rad = self._read_joint_positions()
        expected_shape = (len(self.config.joint_names),)
        if joint_positions_rad.shape != expected_shape:
            raise RuntimeError(f"FR3 joint pose shape must be {expected_shape}, got {joint_positions_rad.shape}.")
        captured = tuple(float(value) for value in joint_positions_rad)
        self.config.start_joint_positions = captured
        return captured

    def restore_configured_start_joint_positions(self) -> tuple[float, ...] | None:
        """Undo a captured start pose, back to what the config declared when the robot was built.

        Deliberately the *config's* value rather than the `home` keyframe literal: those two are
        held equal by tests/robots/test_fr3_home_keyframe_contract.py, and reading the config means
        a rig that legitimately declares a different start pose resets to its own default instead of
        to this file's idea of one.
        """
        self.config.start_joint_positions = self._default_start_joint_positions
        return self._default_start_joint_positions

    @contextlib.contextmanager
    def _otg_speed_scaled(self, speed_scale: float) -> Iterator[None]:
        """Run the OTG at a fraction of its configured joint ceilings for the duration.

        A no-op for an OTG backend that cannot be told -- the sim twin, the test doubles. A
        trajectory generator that ignores the request is still correct, only not slower, and
        refusing to home on a backend that lacks the knob would be the worse failure.
        """

        limits_scaled_by = None if self._otg is None else getattr(self._otg, "limits_scaled_by", None)
        if speed_scale == 1.0 or not callable(limits_scaled_by):
            yield
            return
        with limits_scaled_by(speed_scale):
            yield

    def _move_to_configured_start(
        self, target_joint_positions_rad: np.ndarray, *, speed_scale: float = 1.0
    ) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")
        set_joint_positions = getattr(self._arm, "set_joint_positions", None)
        if not callable(set_joint_positions):
            raise RuntimeError("FR3 arm backend does not support set_joint_positions().")

        speed_scale = float(speed_scale)
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}.")

        tolerance_rad = float(self.config.start_joint_tolerance_rad)
        # Stretched by the same factor the speed is cut by. Slowing the arm down deliberately
        # must not turn a homing move that works into a timeout that reports the arm as broken.
        deadline_s = time.perf_counter() + float(self.config.start_move_timeout_s) / float(speed_scale)
        command_joint_positions_rad = self._read_joint_positions()

        if self._otg is None:
            set_joint_positions(target_joint_positions_rad)
        else:
            with self._otg_speed_scaled(speed_scale):
                self._otg.reset(command_joint_positions_rad)
                while time.perf_counter() < deadline_s:
                    self._raise_if_otg_failed()
                    command_joint_positions_rad = np.asarray(
                        self._otg.step(command_joint_positions_rad, target_joint_positions_rad),
                        dtype=np.float64,
                    )
                    set_joint_positions(command_joint_positions_rad)
                    if np.max(np.abs(command_joint_positions_rad - target_joint_positions_rad)) <= tolerance_rad:
                        break
                    precise_sleep(self.config.otg_dt)
                else:
                    raise RuntimeError(
                        "Timed out generating FR3 start-pose trajectory before reaching the configured keyframe."
                    )

        set_joint_positions(target_joint_positions_rad)
        while time.perf_counter() < deadline_s:
            observed_joint_positions_rad = self._read_joint_positions()
            max_error_rad = float(np.max(np.abs(observed_joint_positions_rad - target_joint_positions_rad)))
            if max_error_rad <= tolerance_rad:
                logger.info("FR3 reached configured start pose with max_joint_error=%.4f rad", max_error_rad)
                return observed_joint_positions_rad
            precise_sleep(0.05)

        raise RuntimeError(
            "FR3 did not reach the configured start pose within "
            f"{self.config.start_move_timeout_s:.1f}s; "
            f"max_joint_error={max_error_rad:.4f} rad, tolerance={tolerance_rad:.4f} rad."
        )

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
        self._reach_stall_error_m = 0.0
        self._reach_stall_since_s = None
        self._reach_stall_last_warned_s = 0.0

    @property
    def reach_stall_error_m(self) -> float:
        """How far the arm is behind its own command, in metres. 0.0 when it is keeping up.

        Non-zero means the commanded tool point is outside what IK can realise from here -- the
        arm is holding still while something keeps asking it to move. Read by the recorder so the
        operator is told instead of guessing; see _note_reach_tracking.

        Written by the control thread and read by the recorder's progress thread. A plain float
        attribute is enough: the reader wants "is it stalled, and roughly how badly", and a
        one-poll-stale answer to that is the same answer.
        """

        return self._reach_stall_error_m

    @check_if_not_connected
    def iter_reach_errors_m(
        self,
        tool_points: Sequence[Sequence[float]],
        rotvec: Sequence[float],
    ) -> Iterator[float]:
        """How far IK falls short of each tool point, walked in order. Commands nothing.

        The measurement `reach_stall_error_m` publishes during a move -- the distance between a
        commanded tool point and the forward kinematics of the joints IK returned for it -- taken
        on the model before anything is sent. It lets a caller refuse a trajectory outright
        instead of discovering it with the arm half-way through and the peg in the gripper.

        Seeded from the arm's current joints and re-seeded from each solution, because that is
        what execution does: the solver is a local one, so whether a point is reachable depends on
        the configuration the arm arrives in and not on the point alone. Feed it a densely sampled
        path rather than bare waypoints, or the answer describes a journey nothing will take.

        Lazy on purpose. An unreachable point costs the solver its whole iteration budget, and a
        caller that stops at the first failure should not pay for the rest of the path.
        """

        joint_positions_rad = self._read_joint_positions()
        rotation = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
        ik_kwargs: dict[str, float] = {}
        if self.config.ik_orientation_weight is not None:
            ik_kwargs["orientation_weight"] = float(self.config.ik_orientation_weight)
        return self._iter_reach_errors_m(joint_positions_rad, tool_points, rotation, ik_kwargs)

    def _iter_reach_errors_m(
        self,
        joint_positions_rad: np.ndarray,
        tool_points: Sequence[Sequence[float]],
        rotation: np.ndarray,
        ik_kwargs: dict[str, float],
    ) -> Iterator[float]:
        # Split out so the connection check and the joint read above happen when the caller asks,
        # not whenever it gets round to the first point.
        for point in tool_points:
            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = rotation
            desired_pose[:3, 3] = np.asarray(point, dtype=np.float64)
            joint_positions_rad = np.asarray(
                self._kinematics.inverse_kinematics(joint_positions_rad, desired_pose, **ik_kwargs),
                dtype=np.float64,
            )
            realised_pose = self._compute_ee_pose(joint_positions_rad)
            yield float(np.linalg.norm(realised_pose[:3, 3] - desired_pose[:3, 3]))

    def _note_reach_tracking(self, desired_pose: np.ndarray, target_joints_rad: np.ndarray) -> None:
        """Compare where the arm was told to go with where these joints actually put it.

        Purely solver-side: both poses come from the kinematics model, so this measures whether IK
        could realise the command, not whether the arm has physically caught up yet. Servo lag,
        OTG smoothing and a slow gripper are all invisible here, which is what makes a non-zero
        reading mean one specific thing.
        """

        realised_pose = self._compute_ee_pose(target_joints_rad)
        error_m = float(np.linalg.norm(realised_pose[:3, 3] - desired_pose[:3, 3]))
        now_s = time.perf_counter()

        if error_m <= _REACH_STALL_TOLERANCE_M:
            # Published as exactly 0.0 rather than the sub-tolerance residual, so a reader can
            # test the value itself instead of re-deriving the threshold this file chose.
            self._reach_stall_error_m = 0.0
            if self._reach_stall_since_s is not None:
                logger.warning(
                    "FR3 reach limit cleared after %.1fs", now_s - self._reach_stall_since_s
                )
            self._reach_stall_since_s = None
            return

        self._reach_stall_error_m = error_m
        if self._reach_stall_since_s is None:
            self._reach_stall_since_s = now_s
        elif now_s - self._reach_stall_last_warned_s < _REACH_STALL_REWARN_S:
            return
        self._reach_stall_last_warned_s = now_s
        commanded = desired_pose[:3, 3]
        realised = realised_pose[:3, 3]
        logger.warning(
            "FR3 reach limit: commanded tool point (%.4f, %.4f, %.4f) is %.1f mm outside what IK "
            "can reach from here; the arm is holding at (%.4f, %.4f, %.4f). Joint limits, not the "
            "workspace fence -- lift the tool or change its orientation to get the axis back.",
            commanded[0],
            commanded[1],
            commanded[2],
            error_m * 1e3,
            realised[0],
            realised[1],
            realised[2],
        )

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

    def _read_joint_positions_with_timestamp(self) -> tuple[np.ndarray, float]:
        """Joint positions and when they were read from the arm, not when we picked them up.

        The two differ because the driver serves a cache refreshed by its own state reader. A
        timestamp taken here would describe this process's scheduling rather than the arm's, and
        the error is not recoverable later: it sits inside the arm-vs-camera offset and is
        indistinguishable from camera latency.
        """
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")
        read_with_timestamp = getattr(self._arm, "get_joint_positions_with_timestamp", None)
        if callable(read_with_timestamp):
            joint_positions, sampled_at_s = read_with_timestamp()
            return np.asarray(joint_positions, dtype=np.float64), float(sampled_at_s)
        # A backend that cannot say when it sampled: the read instant is the honest upper bound.
        return np.asarray(self._arm.get_joint_positions(), dtype=np.float64), time.perf_counter()

    def _read_gripper_position_with_timestamp(self) -> tuple[float, float]:
        """Gripper position and when it was sampled, for the backends that can say.

        Same failure as the arm, and worse where it applies: the Franka Hand driver polls at
        10 Hz, so a pickup-time stamp can be optimistic by 100 ms. `das` knows its instant too --
        the databus hands it over in a callback.

        `pika` and `corenetic` take the branch below, where the read instant is a true upper
        bound rather than a guess. `pika` reads straight through to the SDK's last parsed frame
        and the SDK records no arrival time. `corenetic` samples do carry a timestamp, but it is
        the BOX MCU's clock -- putting it in this column would mean splicing two time bases
        together with no measured offset between them, which buys a plausible number and loses
        the ability to tell the offset from a real lag.
        """
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        read_with_timestamp = getattr(self._gripper, "get_position_with_timestamp", None)
        if callable(read_with_timestamp):
            gripper_pos, sampled_at_s = read_with_timestamp()
            return float(gripper_pos), float(sampled_at_s)
        return float(self._gripper.get_position()), time.perf_counter()

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

    def _camera_timeout_context(
        self,
        *,
        failed_camera_name: str,
        ee_pose: np.ndarray,
        stage_timings_ms: dict[str, float] | None = None,
    ) -> str:
        now_s = time.perf_counter()
        camera_states: list[str] = []
        for camera_name, camera in self.cameras.items():
            timestamp = getattr(camera, "latest_timestamp", None)
            if timestamp is None:
                age_text = "none"
            else:
                try:
                    age_text = f"{(now_s - float(timestamp)) * 1e3:.1f}"
                except (TypeError, ValueError):
                    age_text = "invalid"
            thread = getattr(camera, "thread", None)
            thread_alive = "unknown" if thread is None else str(bool(thread.is_alive())).lower()
            event = getattr(camera, "new_frame_event", None)
            event_set = "unknown" if event is None else str(bool(event.is_set())).lower()
            camera_states.append(
                f"{camera_name}:age_ms={age_text},thread_alive={thread_alive},event_set={event_set}"
            )
        timing_text = ""
        if stage_timings_ms:
            timing_text = " stage_ms=[" + ",".join(
                f"{name}={elapsed_ms:.1f}" for name, elapsed_ms in stage_timings_ms.items()
            ) + "]"
        return (
            f" FR3 camera_timeout_context failed_camera={failed_camera_name} "
            f"ee_z_m={float(ee_pose[2, 3]):.4f}{timing_text} cameras=[{'; '.join(camera_states)}]"
        )

    @check_if_not_connected
    def get_observation(self, *, include_cameras: bool = True) -> RobotObservation:
        self._raise_if_otg_failed()
        observation_start_s = time.perf_counter()
        joint_positions_rad, arm_capture_timestamp_s = self._read_joint_positions_with_timestamp()
        after_arm_read_s = time.perf_counter()
        ee_pose = self._compute_ee_pose(joint_positions_rad)
        self._cache_observation_state_snapshot(joint_positions_rad, ee_pose)
        ee_rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
        gripper_pos, gripper_capture_timestamp_s = self._read_gripper_position_with_timestamp()
        after_gripper_read_s = time.perf_counter()

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
        before_camera_read_s = time.perf_counter()
        stage_timings_ms = {
            "arm_read": (after_arm_read_s - observation_start_s) * 1e3,
            "gripper_read": (after_gripper_read_s - after_arm_read_s) * 1e3,
            "pre_camera": (before_camera_read_s - observation_start_s) * 1e3,
        }
        if include_cameras:
            latest_samples: dict[str, tuple[np.ndarray, float]] = {}
            for camera_name, camera in self.cameras.items():
                try:
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
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"{exc}"
                        f"{self._camera_timeout_context(
                            failed_camera_name=camera_name,
                            ee_pose=ee_pose,
                            stage_timings_ms=stage_timings_ms,
                        )}"
                    ) from exc

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
                if (
                    camera_skew_ms > self.config.camera_max_skew_ms
                    and getattr(self.config, "camera_skew_hard_fail", True)
                ):
                    raise RuntimeError(
                        f"FR3 camera skew {camera_skew_ms:.1f} ms exceeds "
                        f"camera_max_skew_ms={self.config.camera_max_skew_ms:.1f}."
                    )
        return observation

    @check_if_not_connected
    def move_to_start(self, speed_scale: float = 1.0) -> None:
        """Drive the arm back to its start pose.

        `speed_scale` in (0, 1] runs the move at that fraction of the OTG's configured joint
        ceilings, which are the arm's rated maxima. It defaults to 1.0 so the between-episode
        homing every recorder does is unchanged; the caller that turns it down is the scene
        reset, which reaches homing from half-finished trajectories with the peg still gripped.

        It has no effect on the backend's own `move_to_start()` -- the path taken when no start
        keyframe is configured -- because that motion is generated inside libfranka.
        """

        self._raise_if_otg_failed()
        self._clear_observation_state_snapshot()
        if self._arm is None:
            raise RuntimeError("Arm backend is not connected.")

        configured_start = self._configured_start_joint_positions()
        move_to_start = getattr(self._arm, "move_to_start", None)
        if configured_start is None and not callable(move_to_start):
            raise RuntimeError("FR3 arm backend does not support move_to_start().")

        otg_enabled = self._otg is not None
        fallback_joint_positions_rad = self._read_joint_positions()
        if otg_enabled:
            self._stop_otg_loop()

        moved_joint_positions_rad = fallback_joint_positions_rad
        try:
            if configured_start is None:
                move_to_start()
                moved_joint_positions_rad = self._read_joint_positions()
            else:
                moved_joint_positions_rad = self._move_to_configured_start(
                    configured_start, speed_scale=speed_scale
                )
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
            ik_kwargs: dict[str, float] = {}
            if self.config.ik_orientation_weight is not None:
                ik_kwargs["orientation_weight"] = float(self.config.ik_orientation_weight)
            target_joints_rad = self._kinematics.inverse_kinematics(
                joint_positions_rad, desired_pose, **ik_kwargs
            )
            self._note_reach_tracking(desired_pose, target_joints_rad)
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
