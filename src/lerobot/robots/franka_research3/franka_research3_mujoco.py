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

"""MuJoCo-backed FR3 that satisfies the same ``Robot`` contract as the hardware arm.

The point of this adapter is schema parity: a simulated recording session and a hardware
recording session run through the *same* ``record_loop`` and the *same* ee2ee processor
pipeline, so the two datasets differ only in where the pixels and joint angles came from --
never in feature names, ordering, shapes, or timestamp bookkeeping. A policy trained on sim
data can therefore be replayed against hardware data without a translation layer.

That includes the frame: ``ee.*`` here is in the robot base frame, as it is on the hardware
arm. The scene bolts the arm to a pedestal, so the env converts out of MuJoCo world
coordinates on the way through -- see ``FR3MujocoEnvConfig.base_frame_name``.
"""

from __future__ import annotations

from functools import cached_property
import logging
import time
from typing import Any

import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .config_franka_research3_mujoco import FrankaResearch3MujocoConfig
from .processor_franka_research3 import PREV_CMD_GRIPPER_KEY

logger = logging.getLogger(__name__)

# Gripper reads are instantaneous in simulation, but the recorder must still be able to tell
# which column of ``observation.device_capture_timestamp`` belongs to the gripper. Naming the
# simulated backend explicitly keeps a sim dataset from masquerading as a Pika/DAS capture.
SIM_GRIPPER_BACKEND = "sim"
_WORKSPACE_OBJECT_BODY_NAME = "workspace_object_body"


class FrankaResearch3Mujoco(Robot):
    config_class = FrankaResearch3MujocoConfig
    name = "franka_research3_mujoco"

    def __init__(self, config: FrankaResearch3MujocoConfig):
        super().__init__(config)
        self.config = config
        self._env: Any = None
        self._is_connected = False
        self._last_command_pose: np.ndarray | None = None
        self._last_command_gripper: float | None = None
        self._capture_timestamp_origin_s = time.perf_counter()
        self._rng = np.random.default_rng()
        self._workspace_object_home_xy: np.ndarray | None = None

    # ------------------------------------------------------------------ features ---

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
            "prev_cmd.ee.x": float,
            "prev_cmd.ee.y": float,
            "prev_cmd.ee.z": float,
            "prev_cmd.ee.wx": float,
            "prev_cmd.ee.wy": float,
            "prev_cmd.ee.wz": float,
            PREV_CMD_GRIPPER_KEY: float,
        }
        joint_features = {f"{joint}.pos": float for joint in self._joint_state_names}
        camera_features = {
            name: (self.config.camera_height, self.config.camera_width, 3)
            for name in self.config.camera_names
        }
        return {**ee_features, **prev_cmd_features, **joint_features, **camera_features}

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
    def _joint_state_names(self) -> list[str]:
        return [f"joint_{index}" for index in range(1, len(self.config.joint_names) + 1)]

    @property
    def capture_timestamp_feature_names(self) -> tuple[str, ...]:
        return (
            "fr3.arm.capture_timestamp_s",
            f"{SIM_GRIPPER_BACKEND}_gripper.capture_timestamp_s",
            *(f"camera.{name}.capture_timestamp_s" for name in self.config.camera_names),
        )

    def reset_capture_timestamp_origin(self) -> None:
        self._capture_timestamp_origin_s = time.perf_counter()

    def _relative_capture_timestamp(self, timestamp_s: float) -> float:
        return float(timestamp_s - self._capture_timestamp_origin_s)

    # ---------------------------------------------------------------- lifecycle ---

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise RuntimeError(f"{self} is already connected.")
        # Imported lazily: the env pulls in mujoco, which the hardware-only profiles do not install.
        from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig

        env_kwargs: dict[str, Any] = {
            "target_frame_name": self.config.target_frame_name,
            "joint_names": tuple(self.config.joint_names),
            "camera_names": tuple(self.config.camera_names),
            "camera_name_mapping": dict(self.config.camera_name_mapping),
            "camera_width": int(self.config.camera_width),
            "camera_height": int(self.config.camera_height),
            "enable_cameras": True,
            "initial_joint_positions": tuple(self.config.initial_joint_positions),
            "initial_gripper": float(self.config.initial_gripper),
            "workspace_min": tuple(self.config.workspace_min),
            "workspace_max": tuple(self.config.workspace_max),
            "max_target_delta_pos": self.config.max_target_delta_pos,
            "max_target_delta_rot": self.config.max_target_delta_rot,
            "use_otg": bool(self.config.use_otg),
            "continuous_physics": bool(self.config.continuous_physics),
            "continuous_physics_frequency": self.config.continuous_physics_frequency,
            "teleop_control_frequency": float(self.config.teleop_control_frequency),
            "arm_actuator_kp": self.config.arm_actuator_kp,
            "enable_arm_gravity_compensation": bool(self.config.enable_arm_gravity_compensation),
            "arm_gravity_compensation_scale": float(self.config.arm_gravity_compensation_scale),
            "ik_solver": self.config.ik_solver,
            "ik_tolerance": float(self.config.ik_tolerance),
            "ik_max_iterations": int(self.config.ik_max_iterations),
            # The recorder owns episode length; never let the env truncate a run out from under it.
            "max_episode_steps": 10**9,
        }
        if self.config.urdf_path:
            env_kwargs["urdf_path"] = self.config.urdf_path
        if self.config.sim_xml_path:
            env_kwargs["sim_xml_path"] = self.config.sim_xml_path

        self._env = FR3MujocoEnv(cfg=FR3MujocoEnvConfig(**env_kwargs))
        self._env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        self._last_command_pose = None
        self._last_command_gripper = None
        self._is_connected = True
        self._workspace_object_home_xy = self._read_workspace_object_xy()
        # Warm the renderer: the very first render_with_timestamps() pays EGL context and
        # scene-upload costs (~100ms observed) that would otherwise land on episode 0 frame 0
        # and show up in the sync audit as a bogus camera straggle.
        self._env.render_with_timestamps()
        self.reset_capture_timestamp_origin()

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._is_connected = False

    # --------------------------------------------------------------- observation ---

    def get_observation(self, *, include_cameras: bool = True) -> RobotObservation:
        env = self._require_env()
        info = env._build_info(include_camera_obs=False)
        arm_capture_timestamp_s = time.perf_counter()

        ee_pose = np.asarray(info["ee_pose"], dtype=np.float64)
        joint_positions = np.asarray(info["joint_positions"], dtype=np.float64)
        ee_rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
        # Measured finger opening, not the command echo: the hardware robot reports what the
        # gripper actually did, and a sim dataset that reported the setpoint instead would hide
        # closing lag that a policy trained on it will meet on the real gripper.
        gripper_pos = float(env.measured_gripper_position())
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
            f"{SIM_GRIPPER_BACKEND}_gripper.capture_timestamp_s": self._relative_capture_timestamp(
                gripper_capture_timestamp_s
            ),
            **self._make_prev_command_observation(
                current_ee_pose=ee_pose,
                current_gripper_pos=gripper_pos,
            ),
        }
        for index, joint_position in enumerate(joint_positions, start=1):
            observation[f"joint_{index}.pos"] = float(joint_position)

        if include_cameras:
            frames = env.render_with_timestamps()
            if frames is None:
                raise RuntimeError("MuJoCo renderer was busy; could not capture a camera frame.")
            selected_timestamps: list[float] = []
            for camera_name in self.config.camera_names:
                frame, timestamp_s = frames[camera_name]
                observation[camera_name] = frame
                observation[f"camera.{camera_name}.capture_timestamp_s"] = self._relative_capture_timestamp(
                    timestamp_s
                )
                selected_timestamps.append(timestamp_s)
            camera_skew_ms = (max(selected_timestamps) - min(selected_timestamps)) * 1e3
            if camera_skew_ms > self.config.camera_max_skew_ms:
                # Same failure mode and message shape as the hardware robot: a frame whose
                # cameras disagree by more than the budget is not a valid training sample.
                raise RuntimeError(
                    f"FR3 MuJoCo camera render skew {camera_skew_ms:.1f} ms exceeds "
                    f"camera_max_skew_ms={self.config.camera_max_skew_ms:.1f}."
                )
        return observation

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

    # -------------------------------------------------------------------- action ---

    def send_action(self, action: RobotAction) -> RobotAction:
        env = self._require_env()
        control_period_s = 1.0 / float(self.config.teleop_control_frequency)

        absolute_keys = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
        if all(key in action for key in absolute_keys):
            gripper = float(np.clip(action.get("gripper.pos", action.get("gripper", 0.0)), 0.0, 1.0))
            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, 3] = [float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])]
            desired_pose[:3, :3] = Rotation.from_rotvec(
                [float(action["ee.wx"]), float(action["ee.wy"]), float(action["ee.wz"])]
            ).as_matrix()
            env.step_absolute_pose(
                desired_pose,
                gripper=gripper,
                control_period_s=control_period_s,
                ik_orientation_weight=self.config.ik_orientation_weight,
            )
            self._last_command_pose = desired_pose.copy()
            self._last_command_gripper = gripper
            return dict(action)

        # Idle / motion-disabled frames still arrive as the raw delta action; the env's teleop
        # path already implements hold-current-joints for them, matching the hardware driver.
        _, _, _, _, info = env.step_teleop_action(
            action,
            control_period_s,
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        self._last_command_pose = np.asarray(info["target_pose"], dtype=np.float64).copy()
        self._last_command_gripper = float(info["gripper_command"])
        return dict(action)

    # --------------------------------------------------------------- episode ops ---

    def move_to_start(self) -> None:
        """Reset the scene to the episode start pose.

        The recorder calls this between episodes exactly as it does for hardware, so the sim
        and hardware session scripts stay identical.
        """
        env = self._require_env()
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        if self.config.randomize_workspace_object:
            self._randomize_workspace_object()
        self._last_command_pose = None
        self._last_command_gripper = None

    def _require_env(self) -> Any:
        if not self._is_connected or self._env is None:
            raise RuntimeError(f"{self} is not connected.")
        return self._env

    def render_preview_frame(self, camera_name: str | None = None) -> np.ndarray | None:
        """One rendered frame for a person to look at. Returns None if the renderer was busy.

        Deliberately not ``get_observation(include_cameras=True)``: that fails the whole frame
        when the cameras disagree by more than ``camera_max_skew_ms``, which is correct for a
        training sample and wrong here -- a render hiccup would abort the very replay the preview
        exists to illustrate. A preview owes nothing to the timestamp contract.

        Without a name, prefers a camera looking *at* the arm over one mounted on it: a wrist view
        cannot show you whether the trajectory went where you meant it to.
        """
        env = self._require_env()
        frames = env.render_with_timestamps()
        if not frames:
            return None
        selected = camera_name if camera_name in frames else self._preview_camera_name(frames)
        if selected is None:
            return None
        return frames[selected][0]

    def _preview_camera_name(self, frames: dict[str, Any]) -> str | None:
        mapping = self.config.camera_name_mapping or {}
        for name in frames:
            model_camera = str(mapping.get(name, name)).lower()
            if "wrist" in model_camera or "ee" in model_camera:
                continue
            return name
        return next(iter(frames), None)

    def _workspace_object_body_id(self) -> int:
        env = self._require_env()
        try:
            body_id = int(
                env._mujoco.mj_name2id(
                    env.model, env._mujoco.mjtObj.mjOBJ_BODY, _WORKSPACE_OBJECT_BODY_NAME
                )
            )
        except Exception:  # noqa: BLE001 - model without the body is a valid scene
            return -1
        return body_id

    def _read_workspace_object_xy(self) -> np.ndarray | None:
        body_id = self._workspace_object_body_id()
        if body_id < 0:
            return None
        env = self._require_env()
        joint_address = int(env.model.body_jntadr[body_id])
        if joint_address < 0:
            return None
        qpos_address = int(env.model.jnt_qposadr[joint_address])
        return np.asarray(env.data.qpos[qpos_address : qpos_address + 2], dtype=np.float64).copy()

    def _randomize_workspace_object(self) -> None:
        if self._workspace_object_home_xy is None:
            return
        body_id = self._workspace_object_body_id()
        if body_id < 0:
            return
        env = self._require_env()
        joint_address = int(env.model.body_jntadr[body_id])
        if joint_address < 0:
            return
        qpos_address = int(env.model.jnt_qposadr[joint_address])
        radius = float(self.config.workspace_object_random_radius_m) * np.sqrt(self._rng.random())
        angle = 2.0 * np.pi * self._rng.random()
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)
        env.data.qpos[qpos_address : qpos_address + 2] = self._workspace_object_home_xy + offset
        env._mujoco.mj_forward(env.model, env.data)
