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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


def _default_fr3_urdf_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "fr3_pika_gripper_ati.urdf"
    )


@dataclass
class FR3MujocoEnvConfig:
    urdf_path: str = field(default_factory=_default_fr3_urdf_path)
    target_frame_name: str = "pika_gripper_ee"
    target_marker_name: str = "target"
    target_site_name: str = "target_site"
    tcp_marker_name: str = "TCP"
    tcp_site_name: str = "TCP_site"
    joint_names: tuple[str, ...] = (
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    )
    initial_joint_positions: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    workspace_min: tuple[float, float, float] = (0.2, -0.6, 0.05)
    workspace_max: tuple[float, float, float] = (0.9, 0.6, 0.8)
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None
    max_episode_steps: int = 300
    render_mode: str | None = None


class FR3MujocoEnv(gym.Env):
    """Minimal FR3 MuJoCo environment for pre-teleop validation.

    This environment intentionally starts narrow:
    - absolute joint-position actions only
    - deterministic reset/step
    - observations exposing joints plus FK-based EE pose in the same frame

    The purpose is to validate FR3 assets, joint ranges, and frame consistency
    before wiring in teleop-target and visualization layers.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, cfg: FR3MujocoEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or FR3MujocoEnvConfig()
        self.render_mode = self.cfg.render_mode
        self._mujoco = self._import_mujoco()
        self._kinematics = self._build_kinematics()

        self.model = self._mujoco.MjModel.from_xml_path(self.cfg.urdf_path)
        self.data = self._mujoco.MjData(self.model)
        self._step_count = 0

        self._joint_ids = []
        self._qpos_indices = []
        self._qvel_indices = []
        for joint_name in self.cfg.joint_names:
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
            self._joint_ids.append(joint_id)
            self._qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self._qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

        self._qpos_indices = np.asarray(self._qpos_indices, dtype=np.int64)
        self._qvel_indices = np.asarray(self._qvel_indices, dtype=np.int64)
        self._joint_lower = self.model.jnt_range[self._joint_ids, 0].astype(np.float64)
        self._joint_upper = self.model.jnt_range[self._joint_ids, 1].astype(np.float64)
        self._initial_joint_positions = np.clip(
            np.asarray(self.cfg.initial_joint_positions, dtype=np.float64),
            self._joint_lower,
            self._joint_upper,
        )
        self._workspace_min = np.asarray(self.cfg.workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(self.cfg.workspace_max, dtype=np.float64)
        self._prev_enabled = False
        self._reference_pose: np.ndarray | None = None
        self._last_command_pose: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._tcp_pose: np.ndarray | None = None
        self._last_gripper = 1.0

        self.action_space = spaces.Box(
            low=self._joint_lower.astype(np.float32),
            high=self._joint_upper.astype(np.float32),
            shape=(len(self.cfg.joint_names),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=self._joint_lower.astype(np.float32),
                    high=self._joint_upper.astype(np.float32),
                    shape=(len(self.cfg.joint_names),),
                    dtype=np.float32,
                ),
                "environment_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(12,),
                    dtype=np.float32,
                ),
            }
        )

        self._set_joint_state(self._initial_joint_positions)

    @staticmethod
    def _import_mujoco():
        try:
            import mujoco
        except Exception as e:
            raise ImportError(
                "FR3 MuJoCo simulation requires the `mujoco` Python package. "
                "Install the simulation extra or use the dedicated FR3 sim container."
            ) from e
        return mujoco

    def _build_kinematics(self):
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        return PlacoKinematicsDriver(
            urdf_path=self.cfg.urdf_path,
            target_frame_name=self.cfg.target_frame_name,
            joint_names=list(self.cfg.joint_names),
        )

    def _set_joint_state(self, joint_positions: np.ndarray) -> None:
        self.data.qpos[self._qpos_indices] = np.asarray(joint_positions, dtype=np.float64)
        self.data.qvel[self._qvel_indices] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._update_visualization_state()

    def _get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64).copy()

    def _current_tcp_pose(self) -> np.ndarray:
        return np.asarray(self._kinematics.forward_kinematics(self._get_joint_positions()), dtype=np.float64)

    def _update_visualization_state(self) -> None:
        self._tcp_pose = self._current_tcp_pose()
        if self._target_pose is None:
            self._target_pose = self._tcp_pose.copy()

    def _pose_to_seven_d(self, pose: np.ndarray) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return np.concatenate([pose[:3, 3], quat_xyzw], dtype=np.float64)

    def _build_visualization_info(self) -> dict[str, Any]:
        target_pose = self._target_pose if self._target_pose is not None else self._current_tcp_pose()
        tcp_pose = self._tcp_pose if self._tcp_pose is not None else self._current_tcp_pose()
        return {
            "target_marker_name": self.cfg.target_marker_name,
            "target_site_name": self.cfg.target_site_name,
            "target_pose": target_pose.copy(),
            "target_pose_7d": self._pose_to_seven_d(target_pose),
            "tcp_marker_name": self.cfg.tcp_marker_name,
            "tcp_site_name": self.cfg.tcp_site_name,
            "tcp_pose": tcp_pose.copy(),
            "tcp_pose_7d": self._pose_to_seven_d(tcp_pose),
        }

    def _zero_teleop_action(self) -> dict[str, float | bool]:
        return {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": self._last_gripper,
        }

    def _normalize_teleop_action(self, action: dict[str, Any] | None) -> dict[str, float | bool]:
        merged = self._zero_teleop_action()
        if action is not None:
            merged.update(action)
        merged["enabled"] = bool(merged["enabled"])
        for key in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper"):
            merged[key] = float(merged[key])
        merged["gripper"] = float(np.clip(merged["gripper"], 0.0, 1.0))
        return merged

    def _compute_desired_pose_from_teleop(
        self, current_pose: np.ndarray, action: dict[str, float | bool]
    ) -> tuple[np.ndarray, bool]:
        enabled = bool(action["enabled"])
        hold_current_joints = False

        if enabled:
            if not self._prev_enabled or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            delta_pos = np.array([action["target_x"], action["target_y"], action["target_z"]], dtype=np.float64)
            if self.cfg.max_target_delta_pos is not None:
                limit = np.asarray(self.cfg.max_target_delta_pos, dtype=np.float64)
                delta_pos = np.clip(delta_pos, -limit, limit)

            delta_rot = Rotation.from_rotvec([action["target_wx"], action["target_wy"], action["target_wz"]])
            if self.cfg.max_target_delta_rot is not None:
                limit = np.asarray(self.cfg.max_target_delta_rot, dtype=np.float64)
                delta_rot = Rotation.from_rotvec(np.clip(delta_rot.as_rotvec(), -limit, limit))

            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = self._reference_pose[:3, :3] @ delta_rot.as_matrix()
            desired_pose[:3, 3] = self._reference_pose[:3, 3] + delta_pos
            desired_pose[:3, 3] = np.clip(desired_pose[:3, 3], self._workspace_min, self._workspace_max)
        else:
            if self._last_command_pose is None:
                desired_pose = current_pose.copy()
                hold_current_joints = True
            else:
                desired_pose = self._last_command_pose.copy()

        return desired_pose, hold_current_joints

    def step_teleop_action(self, action: dict[str, Any] | None):
        teleop_action = self._normalize_teleop_action(action)
        current_joints = self._get_joint_positions()
        current_pose = self._current_tcp_pose()
        desired_pose, hold_current_joints = self._compute_desired_pose_from_teleop(current_pose, teleop_action)

        if hold_current_joints:
            target_joints = current_joints.copy()
        else:
            target_joints = np.asarray(self._kinematics.inverse_kinematics(current_joints, desired_pose), dtype=np.float64)
        self._set_joint_state(target_joints)

        self._last_gripper = float(teleop_action["gripper"])
        self._last_command_pose = desired_pose.copy()
        if teleop_action["enabled"]:
            self._reference_pose = desired_pose.copy()
        else:
            self._reference_pose = None
        self._target_pose = desired_pose.copy()
        self._prev_enabled = bool(teleop_action["enabled"])

        self._step_count += 1
        observation = self._build_observation()
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        info = self._build_info()
        info["teleop_action"] = teleop_action.copy()
        return observation, 0.0, terminated, truncated, info

    def _build_observation(self) -> dict[str, np.ndarray]:
        joint_positions = self._get_joint_positions()
        ee_pose = np.asarray(self._kinematics.forward_kinematics(joint_positions), dtype=np.float64)
        env_state = np.concatenate([ee_pose[:3, 3], ee_pose[:3, :3].reshape(-1)], dtype=np.float64)
        return {
            "agent_pos": joint_positions.astype(np.float32),
            "environment_state": env_state.astype(np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_enabled = False
        self._reference_pose = None
        self._last_command_pose = None
        self._target_pose = None
        target_joint_positions = self._initial_joint_positions
        if options and "joint_positions" in options:
            target_joint_positions = np.clip(
                np.asarray(options["joint_positions"], dtype=np.float64),
                self._joint_lower,
                self._joint_upper,
            )
        self._set_joint_state(target_joint_positions)
        return self._build_observation(), self._build_info()

    def step(self, action: np.ndarray):
        target_joint_positions = np.clip(
            np.asarray(action, dtype=np.float64).reshape(len(self.cfg.joint_names)),
            self._joint_lower,
            self._joint_upper,
        )
        self._set_joint_state(target_joint_positions)
        self._last_command_pose = self._tcp_pose.copy()
        self._reference_pose = None
        self._prev_enabled = False
        self._step_count += 1
        observation = self._build_observation()
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        return observation, 0.0, terminated, truncated, self._build_info()

    def _build_info(self) -> dict[str, Any]:
        joint_positions = self._get_joint_positions()
        ee_pose = self._current_tcp_pose()
        info = {
            "joint_positions": joint_positions,
            "ee_pose": ee_pose,
            "target_frame_name": self.cfg.target_frame_name,
        }
        info.update(self._build_visualization_info())
        return info

    def render(self):
        # The first implementation step stays headless. Viewer integration will
        # be added after the local sim gate is stable.
        return None

    def close(self) -> None:
        return None


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: FR3MujocoEnvConfig | None = None,
) -> gym.vector.VectorEnv:
    env_cfg = cfg or FR3MujocoEnvConfig()
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    def _make_one():
        return FR3MujocoEnv(cfg=env_cfg)

    return env_cls([_make_one for _ in range(n_envs)])
