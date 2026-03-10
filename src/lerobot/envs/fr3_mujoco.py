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

    def _get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64).copy()

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
        self._step_count += 1
        observation = self._build_observation()
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        return observation, 0.0, terminated, truncated, self._build_info()

    def _build_info(self) -> dict[str, Any]:
        joint_positions = self._get_joint_positions()
        ee_pose = np.asarray(self._kinematics.forward_kinematics(joint_positions), dtype=np.float64)
        return {
            "joint_positions": joint_positions,
            "ee_pose": ee_pose,
            "target_frame_name": self.cfg.target_frame_name,
        }

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
