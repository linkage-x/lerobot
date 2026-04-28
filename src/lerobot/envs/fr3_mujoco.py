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
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.utils.rotation import Rotation


def _default_fr3_urdf_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "fr3_pika_gripper_ati.urdf"
    )


def _default_fr3_sim_xml_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "fr3_pika_ati_scene.xml"
    )


@dataclass
class FR3MujocoEnvConfig:
    urdf_path: str = field(default_factory=_default_fr3_urdf_path)
    sim_xml_path: str = field(default_factory=_default_fr3_sim_xml_path)
    target_frame_name: str = "pika_task_tcp"
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
    gripper_joint_names: tuple[str, str] = ("gripper_left_joint", "gripper_right_joint")
    camera_names: tuple[str, ...] = (
        "third_person",
        "north_east",
        "side",
        "west",
        "south_west",
        "south_east",
        "wrist",
    )
    camera_name_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "third_person": "third_person_cam",
            "north_east": "north_east_cam",
            "side": "side_cam",
            "west": "west_cam",
            "south_west": "south_west_cam",
            "south_east": "south_east_cam",
            "wrist": "ee_cam",
        }
    )
    scene_geom_names: tuple[str, ...] = (
        "floor",
        "table",
        "workspace_object",
        "peg_hole_base",
        "peg_hole_wall_x_pos",
        "peg_hole_wall_x_neg",
        "peg_hole_wall_y_pos",
        "peg_hole_wall_y_neg",
    )
    camera_height: int = 256
    camera_width: int = 256
    enable_cameras: bool = False
    # initial_joint_positions: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    initial_joint_positions: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 3.14, 0.785)
    initial_gripper: float = 1.0
    gripper_sim_steps: int = 640
    workspace_min: tuple[float, float, float] = (0.2, -0.6, 0.05)
    workspace_max: tuple[float, float, float] = (0.9, 0.6, 1.2)
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None
    use_otg: bool = True
    arm_actuator_kp: float | None = None
    enable_arm_gravity_compensation: bool = True
    arm_gravity_compensation_scale: float = 0.5
    teleop_control_frequency: float = 200.0
    otg_control_frequency: float = 800.0
    otg_async_control_frequency: float = 1000.0
    continuous_physics: bool = False
    continuous_physics_frequency: float | None = None
    otg_max_velocity: tuple[float, ...] = (2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208)
    otg_max_acceleration: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)
    otg_max_jerk: tuple[float, ...] = (4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0)
    otg_min_position: tuple[float, ...] = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
    otg_max_position: tuple[float, ...] = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)
    otg_synchronization: bool = True
    otg_sync_mode: str = "time"
    max_episode_steps: int = 300
    render_mode: str | None = None

    @property
    def teleop_dt(self) -> float:
        return 1.0 / self.teleop_control_frequency

    @property
    def otg_dt(self) -> float:
        return 1.0 / self.otg_control_frequency

    @property
    def otg_async_dt(self) -> float:
        return 1.0 / self.otg_async_control_frequency

    @property
    def continuous_physics_dt(self) -> float:
        if self.use_otg:
            frequency = self.otg_async_control_frequency
        else:
            frequency = self.otg_control_frequency if self.continuous_physics_frequency is None else self.continuous_physics_frequency
        return 1.0 / float(frequency)


class _MujocoArmKinematics:
    def __init__(
        self,
        mujoco,
        model,
        target_frame_name: str,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ):
        self._mujoco = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_frame_name)
        if self._target_body_id < 0:
            raise ValueError(f"Body '{target_frame_name}' not found in MuJoCo model.")
        self._qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self._qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        self._joint_lower = np.asarray(joint_lower, dtype=np.float64)
        self._joint_upper = np.asarray(joint_upper, dtype=np.float64)

    def _set_arm_qpos(self, joint_positions: np.ndarray) -> None:
        self._data.qpos[self._qpos_indices] = np.asarray(joint_positions, dtype=np.float64)
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        self._set_arm_qpos(joint_positions)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self._data.xpos[self._target_body_id]
        pose[:3, :3] = self._data.xmat[self._target_body_id].reshape(3, 3)
        return pose

    def inverse_kinematics(
        self,
        current_joint_positions: np.ndarray,
        desired_pose: np.ndarray,
        *,
        lock_orientation: bool = False,
        orientation_weight: float | None = None,
    ) -> np.ndarray:
        guess = np.clip(np.asarray(current_joint_positions, dtype=np.float64), self._joint_lower, self._joint_upper)
        jac_pos = np.zeros((3, self._model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self._model.nv), dtype=np.float64)
        damping = 0.3
        resolved_orientation_weight = 1.0 if lock_orientation else 0.1
        if orientation_weight is not None:
            resolved_orientation_weight = float(orientation_weight)

        # The task TCP frame can be rotated away from the historical default.
        # Give LM extra iterations so reachable far targets still converge
        # under the updated frame convention.
        for _ in range(400):
            self._set_arm_qpos(guess)
            current_pos = np.asarray(self._data.xpos[self._target_body_id], dtype=np.float64)
            current_rot = np.asarray(self._data.xmat[self._target_body_id], dtype=np.float64).reshape(3, 3)

            pos_error = np.asarray(desired_pose[:3, 3], dtype=np.float64) - current_pos
            rot_error = (
                Rotation.from_matrix(np.asarray(desired_pose[:3, :3], dtype=np.float64) @ current_rot.T).as_rotvec()
            )
            error = np.concatenate([pos_error, rot_error * resolved_orientation_weight], dtype=np.float64)
            if np.linalg.norm(error) < 1e-6:
                break

            self._mujoco.mj_jacBody(self._model, self._data, jac_pos, jac_rot, self._target_body_id)
            jacobian = np.vstack([jac_pos[:, self._qvel_indices], jac_rot[:, self._qvel_indices]])
            lhs = jacobian.T @ jacobian + damping * np.eye(len(self._qvel_indices), dtype=np.float64)
            step = np.linalg.solve(lhs, jacobian.T @ error)
            guess = np.clip(guess + 0.5 * step, self._joint_lower, self._joint_upper)

        return guess


class FR3MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, cfg: FR3MujocoEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or FR3MujocoEnvConfig()
        self.render_mode = self.cfg.render_mode
        self._mujoco = self._import_mujoco()
        self._renderer = None

        self.model = self._mujoco.MjModel.from_xml_path(self.cfg.sim_xml_path)
        self.data = self._mujoco.MjData(self.model)
        self._step_count = 0
        self._physics_lock = threading.RLock()
        self._continuous_physics_stop = threading.Event()
        self._continuous_physics_thread: threading.Thread | None = None
        self._gravity_comp_data = self._mujoco.MjData(self.model) if self.cfg.enable_arm_gravity_compensation else None

        self._joint_ids = []
        self._qpos_indices = []
        self._qvel_indices = []
        self._actuator_ids = []
        for joint_name in self.cfg.joint_names:
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
            self._joint_ids.append(joint_id)
            self._qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self._qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))
            actuator_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"fr3_actuator{len(self._actuator_ids) + 1}",
            )
            self._actuator_ids.append(actuator_id)

        self._qpos_indices = np.asarray(self._qpos_indices, dtype=np.int64)
        self._qvel_indices = np.asarray(self._qvel_indices, dtype=np.int64)
        self._joint_lower = self.model.jnt_range[self._joint_ids, 0].astype(np.float64)
        self._joint_upper = self.model.jnt_range[self._joint_ids, 1].astype(np.float64)
        self._configure_arm_position_actuators()
        self._kinematics = self._build_kinematics()
        self._otg = self._build_otg()
        self._initial_joint_positions = np.clip(
            np.asarray(self.cfg.initial_joint_positions, dtype=np.float64),
            self._joint_lower,
            self._joint_upper,
        )

        self._gripper_joint_indices: dict[str, int] = {}
        self._gripper_qvel_indices: dict[str, int] = {}
        self._gripper_joint_limits: dict[str, tuple[float, float]] = {}
        for key, joint_name in zip(("left", "right"), self.cfg.gripper_joint_names, strict=True):
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Gripper joint '{joint_name}' not found in MuJoCo model.")
            self._gripper_joint_indices[key] = int(self.model.jnt_qposadr[joint_id])
            self._gripper_qvel_indices[key] = int(self.model.jnt_dofadr[joint_id])
            limits = self.model.jnt_range[joint_id].astype(np.float64)
            self._gripper_joint_limits[key] = (float(limits[0]), float(limits[1]))
        self._gripper_actuator_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_ACTUATOR,
            "pika_gripper_actuator",
        )
        if self._gripper_actuator_id < 0:
            raise ValueError("Actuator 'pika_gripper_actuator' not found in MuJoCo model.")
        self._gripper_ctrl_range = self.model.actuator_ctrlrange[self._gripper_actuator_id].astype(np.float64)

        self._workspace_min = np.asarray(self.cfg.workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(self.cfg.workspace_max, dtype=np.float64)
        self._tcp_body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, self.cfg.target_frame_name)
        if self._tcp_body_id < 0:
            raise ValueError(f"TCP body '{self.cfg.target_frame_name}' not found in MuJoCo model.")
        self._gripper_base_body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self._gripper_left_body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "gripper_left_link")
        self._gripper_right_body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "gripper_right_link")
        self._link7_body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "fr3_link7")
        self._table_geom_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._workspace_object_geom_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "workspace_object"
        )
        self._gripper_left_pad_geom_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "gripper_left_pad_collision"
        )
        self._gripper_right_pad_geom_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "gripper_right_pad_collision"
        )
        self._gripper_left_geom_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "gripper_left_collision"
        )
        self._gripper_right_geom_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "gripper_right_collision"
        )
        self._prev_enabled = False
        self._reference_pose: np.ndarray | None = None
        self._last_command_pose: np.ndarray | None = None
        self._hold_joint_target: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._tcp_pose: np.ndarray | None = None
        self._otg_target_joints: np.ndarray | None = None
        self._otg_command_joints: np.ndarray | None = None
        self._servo_target_joints: np.ndarray | None = None
        self._last_gripper = float(np.clip(self.cfg.initial_gripper, 0.0, 1.0))

        observation_dict: dict[str, spaces.Space] = {
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
        if self.cfg.enable_cameras:
            observation_dict["camera_obs"] = spaces.Dict(
                {
                    camera_name: spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.cfg.camera_height, self.cfg.camera_width, 3),
                        dtype=np.uint8,
                    )
                    for camera_name in self.cfg.camera_names
                }
            )
        self.action_space = spaces.Box(
            low=self._joint_lower.astype(np.float32),
            high=self._joint_upper.astype(np.float32),
            shape=(len(self.cfg.joint_names),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(observation_dict)

        self._reset_joint_state(self._initial_joint_positions)
        self._servo_target_joints = self._initial_joint_positions.copy()
        self._set_gripper_command(self._last_gripper)

        self._renderer = None
        self._renderer_owner_thread_id: int | None = None

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
        # Keep teleop IK/FK on the same kinematic model that MuJoCo executes.
        # Mixing URDF/placo kinematics with MuJoCo physics lets target_pose look
        # correct in FK while the simulated TCP moves in a different direction.
        return _MujocoArmKinematics(
            mujoco=self._mujoco,
            model=self.model,
            target_frame_name=self.cfg.target_frame_name,
            qpos_indices=self._qpos_indices,
            qvel_indices=self._qvel_indices,
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
        )

    def _build_otg(self):
        if not self.cfg.use_otg:
            return None
        from lerobot.robots.franka_research3.backends import RuckigOTGDriver

        return RuckigOTGDriver(
            dof=len(self.cfg.joint_names),
            dt=self.cfg.otg_dt,
            max_velocity=list(self.cfg.otg_max_velocity),
            max_acceleration=list(self.cfg.otg_max_acceleration),
            max_jerk=list(self.cfg.otg_max_jerk),
            min_position=list(self.cfg.otg_min_position),
            max_position=list(self.cfg.otg_max_position),
            synchronization=self.cfg.otg_synchronization,
            sync_mode=self.cfg.otg_sync_mode,
        )

    def _configure_arm_position_actuators(self) -> None:
        if self.cfg.arm_actuator_kp is None:
            return
        kp = float(self.cfg.arm_actuator_kp)
        if kp <= 0:
            raise ValueError("arm_actuator_kp must be positive when provided.")
        actuator_ids = np.asarray(self._actuator_ids, dtype=np.int64)
        self.model.actuator_gainprm[actuator_ids, 0] = kp
        self.model.actuator_biasprm[actuator_ids, 1] = -kp

    def _apply_arm_gravity_compensation_locked(self) -> None:
        if not self.cfg.enable_arm_gravity_compensation:
            self.data.qfrc_applied[self._qvel_indices] = 0.0
            return
        if self._gravity_comp_data is None:
            self.data.qfrc_applied[self._qvel_indices] = 0.0
            return
        scale = float(self.cfg.arm_gravity_compensation_scale)
        if scale < 0:
            raise ValueError("arm_gravity_compensation_scale must be non-negative.")

        # First version of arm gravity compensation: evaluate bias forces on a
        # zero-velocity copy of the current state so we only cancel the gravity
        # load and avoid feeding Coriolis/centrifugal terms back into teleop.
        self._gravity_comp_data.qpos[:] = self.data.qpos
        self._gravity_comp_data.qvel[:] = 0.0
        if hasattr(self._gravity_comp_data, "act") and hasattr(self.data, "act"):
            if self._gravity_comp_data.act.shape == self.data.act.shape:
                self._gravity_comp_data.act[:] = self.data.act
        self._gravity_comp_data.ctrl[:] = self.data.ctrl
        self._mujoco.mj_forward(self.model, self._gravity_comp_data)
        self.data.qfrc_applied[self._qvel_indices] = np.asarray(
            scale * self._gravity_comp_data.qfrc_bias[self._qvel_indices],
            dtype=np.float64,
        )

    def _set_arm_target(self, joint_positions: np.ndarray) -> np.ndarray:
        with self._physics_lock:
            arm_positions = np.clip(joint_positions, self._joint_lower, self._joint_upper)
            if self._actuator_ids:
                self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = arm_positions
            return np.asarray(arm_positions, dtype=np.float64)

    def _step_physics(self, physics_steps: int = 1) -> None:
        with self._physics_lock:
            for _ in range(max(int(physics_steps), 1)):
                self._apply_arm_gravity_compensation_locked()
                self._mujoco.mj_step(self.model, self.data)
            self._mujoco.mj_forward(self.model, self.data)
            self._update_visualization_state()

    def _copy_visual_state_locked(self, target_data) -> None:
        target_data.time = self.data.time
        target_data.qpos[:] = self.data.qpos
        target_data.qvel[:] = self.data.qvel
        if hasattr(target_data, "act") and hasattr(self.data, "act") and target_data.act.shape == self.data.act.shape:
            target_data.act[:] = self.data.act
        target_data.ctrl[:] = self.data.ctrl
        if hasattr(target_data, "mocap_pos") and hasattr(self.data, "mocap_pos"):
            target_data.mocap_pos[:] = self.data.mocap_pos
        if hasattr(target_data, "mocap_quat") and hasattr(self.data, "mocap_quat"):
            target_data.mocap_quat[:] = self.data.mocap_quat
        self._mujoco.mj_forward(self.model, target_data)

    def copy_visual_state(self, target_data) -> None:
        with self._physics_lock:
            self._copy_visual_state_locked(target_data)

    def _set_joint_state(self, joint_positions: np.ndarray) -> None:
        """Compatibility wrapper: set arm target then advance one physics step."""
        self._set_arm_target(joint_positions)
        self._step_physics(1)

    def _reset_joint_state(self, joint_positions: np.ndarray) -> None:
        """Direct qpos write for initialization/reset (no physics, matches prior _set_joint_state behavior).

        Used only when placing the arm at startup or after reset — not during normal stepping.
        """
        with self._physics_lock:
            arm_positions = np.asarray(joint_positions, dtype=np.float64)
            previous_qpos = self.data.qpos.copy()
            previous_qvel = self.data.qvel.copy()
            previous_ctrl = self.data.ctrl.copy()

            def apply_arm_state(candidate: np.ndarray) -> bool:
                self.data.qpos[:] = previous_qpos
                self.data.qvel[:] = previous_qvel
                self.data.ctrl[:] = previous_ctrl
                self.data.qpos[self._qpos_indices] = candidate
                self.data.qvel[self._qvel_indices] = 0.0
                if self._actuator_ids:
                    self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = candidate
                self._mujoco.mj_forward(self.model, self.data)
                return not self._has_gripper_table_penetration()

            if not apply_arm_state(arm_positions):
                low, high = 0.0, 1.0
                best = np.asarray(previous_qpos[self._qpos_indices], dtype=np.float64).copy()
                for _ in range(12):
                    mid = 0.5 * (low + high)
                    candidate = np.asarray(previous_qpos[self._qpos_indices], dtype=np.float64) + mid * (
                        arm_positions - np.asarray(previous_qpos[self._qpos_indices], dtype=np.float64)
                    )
                    if apply_arm_state(candidate):
                        best = candidate.copy()
                        low = mid
                    else:
                        high = mid
                apply_arm_state(best)
            self._update_visualization_state()

    def _gripper_joint_targets_from_command(self, gripper_command: float) -> dict[str, float]:
        command = float(np.clip(gripper_command, 0.0, 1.0))
        targets: dict[str, float] = {}
        for key, (lower, upper) in self._gripper_joint_limits.items():
            closed = 0.0
            if lower <= 0.0 <= upper:
                closed = 0.0
            else:
                closed = lower if abs(lower) < abs(upper) else upper
            open_target = lower if abs(lower) > abs(upper) else upper
            targets[key] = float(closed + command * (open_target - closed))
        return targets

    def _gripper_ctrl_from_command(self, gripper_command: float) -> float:
        command = float(np.clip(gripper_command, 0.0, 1.0))
        lower, upper = self._gripper_ctrl_range
        return float(lower + command * (upper - lower))

    def _set_gripper_command(self, gripper_command: float, *, simulate: bool = False) -> None:
        with self._physics_lock:
            self.data.ctrl[self._gripper_actuator_id] = self._gripper_ctrl_from_command(gripper_command)
            if self.cfg.continuous_physics and self._continuous_physics_thread is not None:
                return
            if simulate:
                frozen_arm_target = np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64).copy()
                if self._actuator_ids:
                    self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = frozen_arm_target
                self.data.qvel[:] = 0.0
                self._mujoco.mj_forward(self.model, self.data)
                for _ in range(max(int(self.cfg.gripper_sim_steps), 1)):
                    self.data.qpos[self._qpos_indices] = frozen_arm_target
                    self.data.qvel[self._qvel_indices] = 0.0
                    if self._actuator_ids:
                        self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = frozen_arm_target
                    self._apply_arm_gravity_compensation_locked()
                    self._mujoco.mj_step(self.model, self.data)
                self.data.qpos[self._qpos_indices] = frozen_arm_target
                self.data.qvel[self._qvel_indices] = 0.0
                if self._actuator_ids:
                    self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = frozen_arm_target
                self._mujoco.mj_forward(self.model, self.data)
                self._update_visualization_state()
                return

            targets = self._gripper_joint_targets_from_command(gripper_command)
            for key, qpos_index in self._gripper_joint_indices.items():
                self.data.qpos[qpos_index] = targets[key]
                self.data.qvel[self._gripper_qvel_indices[key]] = 0.0
            self._mujoco.mj_forward(self.model, self.data)
            self._update_visualization_state()

    def _get_gripper_joint_positions(self) -> dict[str, float]:
        return {
            key: float(self.data.qpos[qpos_index])
            for key, qpos_index in self._gripper_joint_indices.items()
        }

    def _has_gripper_table_penetration(self) -> bool:
        if self._table_geom_id < 0:
            return False
        guard_geom_ids = {
            geom_id
            for geom_id in (self._gripper_left_pad_geom_id, self._gripper_right_pad_geom_id)
            if geom_id >= 0
        }
        if not guard_geom_ids:
            guard_geom_ids = {
                geom_id
                for geom_id in (self._gripper_left_geom_id, self._gripper_right_geom_id)
                if geom_id >= 0
            }
        if not guard_geom_ids:
            return False
        for contact_id in range(int(self.data.ncon)):
            contact = self.data.contact[contact_id]
            if float(contact.dist) >= -1e-6:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._table_geom_id not in (geom1, geom2):
                continue
            other_geom = geom2 if geom1 == self._table_geom_id else geom1
            if other_geom in guard_geom_ids:
                return True
        return False

    def _get_joint_positions(self) -> np.ndarray:
        with self._physics_lock:
            return np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64).copy()

    def _current_tcp_pose(self) -> np.ndarray:
        with self._physics_lock:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = np.asarray(self.data.xpos[self._tcp_body_id], dtype=np.float64)
            pose[:3, :3] = np.asarray(self.data.xmat[self._tcp_body_id], dtype=np.float64).reshape(3, 3)
            return pose

    def _update_visualization_state(self) -> None:
        self._tcp_pose = self._current_tcp_pose()
        if self._target_pose is None:
            self._target_pose = self._tcp_pose.copy()

    def _apply_continuous_control_tick_locked(self) -> None:
        if self._otg is not None and self._otg_target_joints is not None:
            command_joints = (
                np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64).copy()
                if self._otg_command_joints is None
                else self._otg_command_joints.copy()
            )
            if np.allclose(command_joints, self._otg_target_joints, atol=1e-6, rtol=0):
                self._otg_command_joints = self._otg_target_joints.copy()
                self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = self._otg_target_joints
            else:
                next_joints = self._otg.step(command_joints, self._otg_target_joints)
                self._otg_command_joints = next_joints.copy()
                self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = np.clip(
                    next_joints, self._joint_lower, self._joint_upper
                )
        elif self._servo_target_joints is not None:
            self.data.ctrl[np.asarray(self._actuator_ids, dtype=np.int64)] = np.clip(
                self._servo_target_joints, self._joint_lower, self._joint_upper
            )

        self._apply_arm_gravity_compensation_locked()
        self._mujoco.mj_step(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        self._update_visualization_state()

    def _continuous_physics_loop(self) -> None:
        dt = self.cfg.continuous_physics_dt
        while not self._continuous_physics_stop.is_set():
            step_start = time.perf_counter()
            with self._physics_lock:
                self._apply_continuous_control_tick_locked()
            sleep_s = max(dt - (time.perf_counter() - step_start), 0.0)
            if self._continuous_physics_stop.wait(sleep_s):
                break

    def _ensure_continuous_physics_thread(self) -> None:
        if not self.cfg.continuous_physics:
            return
        if self._continuous_physics_thread is not None and self._continuous_physics_thread.is_alive():
            return
        self._continuous_physics_stop.clear()
        self._continuous_physics_thread = threading.Thread(
            target=self._continuous_physics_loop,
            name="fr3-mujoco-physics",
            daemon=True,
        )
        self._continuous_physics_thread.start()

    def _stop_continuous_physics_thread(self) -> None:
        if self._continuous_physics_thread is None:
            return
        self._continuous_physics_stop.set()
        self._continuous_physics_thread.join(timeout=1.0)
        self._continuous_physics_thread = None
        self._continuous_physics_stop.clear()

    def _reset_otg_state(self, current_joint_positions: np.ndarray) -> None:
        if self._otg is None:
            self._otg_target_joints = None
            self._otg_command_joints = None
            return
        current = np.asarray(current_joint_positions, dtype=np.float64)
        self._otg.reset(current)
        self._otg_target_joints = current.copy()
        self._otg_command_joints = current.copy()

    def _control_window_step_counts(self, duration_s: float | None) -> tuple[int, int]:
        window_s = self.cfg.teleop_dt if duration_s is None else max(float(duration_s), 0.0)
        physics_steps = max(1, int(np.ceil(window_s / self.cfg.otg_dt)))
        sender_steps = max(1, int(np.ceil(window_s / self.cfg.otg_async_dt)))
        return physics_steps, sender_steps

    def _advance_servo_window(self, target_joint_positions: np.ndarray, duration_s: float | None) -> tuple[int, int]:
        physics_steps, sender_steps = self._control_window_step_counts(duration_s)
        target = np.clip(np.asarray(target_joint_positions, dtype=np.float64), self._joint_lower, self._joint_upper)
        self._set_arm_target(target)
        self._step_physics(physics_steps)
        return physics_steps, sender_steps

    def _advance_otg_window(self, duration_s: float | None) -> tuple[int, int]:
        if self._otg is None or self._otg_target_joints is None:
            return 0, 0

        otg_steps, sender_steps = self._control_window_step_counts(duration_s)
        command_joints = (
            self._get_joint_positions() if self._otg_command_joints is None else self._otg_command_joints.copy()
        )

        # Use the previous OTG command state, matching the real-robot integration.
        # Refeeding lagging measured qpos together with planned velocity/acceleration
        # makes Ruckig replan from an inconsistent state and can reverse direction.
        if np.allclose(command_joints, self._otg_target_joints, atol=1e-6, rtol=0):
            self._otg_command_joints = self._otg_target_joints.copy()
            self._advance_servo_window(self._otg_target_joints, duration_s)
            return otg_steps, sender_steps

        for _ in range(otg_steps):
            next_joints = self._otg.step(command_joints, self._otg_target_joints)
            self._otg_command_joints = next_joints.copy()
            self._set_arm_target(next_joints)
            self._step_physics(1)
            command_joints = next_joints.copy()

        # if self._step_count % 50 == 0:
        #     final_joints = self._get_joint_positions()
        #     err = np.linalg.norm(final_joints - self._otg_target_joints)
        #     print(f"[DEBUG OTG] step={self._step_count} otg_steps={otg_steps} err={err:.6f} target[:3]={self._otg_target_joints[:3]} actual[:3]={final_joints[:3]}")

        return otg_steps, sender_steps

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
            "camera_names": tuple(self.cfg.camera_names),
            "scene_geom_names": tuple(self.cfg.scene_geom_names),
            "gripper_joint_positions": self._get_gripper_joint_positions(),
            "gripper_command": float(self._last_gripper),
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
            self._hold_joint_target = None
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
            if self._hold_joint_target is None:
                self._hold_joint_target = self._get_joint_positions().copy()
            desired_pose = np.asarray(self._kinematics.forward_kinematics(self._hold_joint_target), dtype=np.float64)
            hold_current_joints = True

        return desired_pose, hold_current_joints

    def _build_camera_obs(self) -> dict[str, np.ndarray]:
        return self.render()

    def step_teleop_action(
        self,
        action: dict[str, Any] | None,
        control_period_s: float | None = None,
        *,
        include_camera_obs_in_observation: bool = True,
        include_camera_obs_in_info: bool = True,
    ):
        with self._physics_lock:
            teleop_action = self._normalize_teleop_action(action)
            current_joints = self._get_joint_positions()
            current_pose = self._current_tcp_pose()
            desired_pose, hold_current_joints = self._compute_desired_pose_from_teleop(current_pose, teleop_action)

            if hold_current_joints:
                target_joints = self._hold_joint_target.copy()
                target_joints = np.clip(target_joints, self._joint_lower, self._joint_upper)
                self._servo_target_joints = target_joints.copy()
                self._otg_target_joints = None
                if self.cfg.continuous_physics:
                    otg_steps, sender_steps = 0, 0
                else:
                    self._advance_servo_window(target_joints, control_period_s)
                    otg_steps, sender_steps = 0, 0
            else:
                target_joints = np.asarray(
                    self._kinematics.inverse_kinematics(current_joints, desired_pose, lock_orientation=True),
                    dtype=np.float64,
                )
                target_joints = np.clip(target_joints, self._joint_lower, self._joint_upper)
                if self._otg is not None:
                    self._otg_target_joints = target_joints.copy()
                    self._servo_target_joints = None
                    if self.cfg.continuous_physics:
                        otg_steps, sender_steps = 0, 0
                    else:
                        otg_steps, sender_steps = self._advance_otg_window(control_period_s)
                else:
                    self._servo_target_joints = target_joints.copy()
                    self._otg_target_joints = None
                    if self.cfg.continuous_physics:
                        otg_steps, sender_steps = 0, 0
                    else:
                        self._advance_servo_window(target_joints, control_period_s)
                        otg_steps, sender_steps = 0, 0

            previous_gripper = self._last_gripper
            self._last_gripper = float(teleop_action["gripper"])
            if not np.isclose(previous_gripper, self._last_gripper):
                self._set_gripper_command(self._last_gripper, simulate=not self.cfg.continuous_physics)
            self._last_command_pose = desired_pose.copy()
            if teleop_action["enabled"]:
                self._reference_pose = desired_pose.copy()
            else:
                self._reference_pose = None
            self._target_pose = desired_pose.copy()
            self._prev_enabled = bool(teleop_action["enabled"])

            self._step_count += 1
            observation = self._build_observation(include_camera_obs=include_camera_obs_in_observation)
            terminated = False
            truncated = self._step_count >= self.cfg.max_episode_steps
            info = self._build_info(include_camera_obs=include_camera_obs_in_info)
            info["teleop_action"] = teleop_action.copy()
            info["target_joint_positions"] = target_joints.copy()
            info["otg_enabled"] = self._otg is not None
            info["otg_steps"] = otg_steps
            info["sender_steps"] = sender_steps
            return observation, 0.0, terminated, truncated, info

    def _build_observation(self, *, include_camera_obs: bool = True) -> dict[str, Any]:
        joint_positions = self._get_joint_positions()
        ee_pose = self._current_tcp_pose()
        env_state = np.concatenate([ee_pose[:3, 3], ee_pose[:3, :3].reshape(-1)], dtype=np.float64)
        observation: dict[str, Any] = {
            "agent_pos": joint_positions.astype(np.float32),
            "environment_state": env_state.astype(np.float32),
        }
        if self.cfg.enable_cameras and include_camera_obs:
            observation["camera_obs"] = self._build_camera_obs()
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        include_camera_obs_in_observation: bool = True,
        include_camera_obs_in_info: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._physics_lock:
            super().reset(seed=seed)
            self._step_count = 0
            self._prev_enabled = False
            self._reference_pose = None
            self._last_command_pose = None
            self._hold_joint_target = None
            self._target_pose = None
            self._otg_target_joints = None
            self._otg_command_joints = None
            target_joint_positions = self._initial_joint_positions
            if options and "joint_positions" in options:
                target_joint_positions = np.clip(
                    np.asarray(options["joint_positions"], dtype=np.float64),
                    self._joint_lower,
                    self._joint_upper,
                )
            self._reset_joint_state(target_joint_positions)
            self._reset_otg_state(target_joint_positions)
            self._servo_target_joints = np.asarray(target_joint_positions, dtype=np.float64).copy()
            self._last_gripper = float(np.clip(self.cfg.initial_gripper, 0.0, 1.0))
            self._set_gripper_command(self._last_gripper)
            if self.cfg.continuous_physics:
                self._ensure_continuous_physics_thread()
            return (
                self._build_observation(include_camera_obs=include_camera_obs_in_observation),
                self._build_info(include_camera_obs=include_camera_obs_in_info),
            )

    def step(self, action: np.ndarray):
        with self._physics_lock:
            target_joint_positions = np.clip(
                np.asarray(action, dtype=np.float64).reshape(len(self.cfg.joint_names)),
                self._joint_lower,
                self._joint_upper,
            )
            if self._otg is not None:
                self._otg_target_joints = target_joint_positions.copy()
                self._servo_target_joints = None
                if not self.cfg.continuous_physics:
                    self._advance_otg_window(self.cfg.teleop_dt)
            else:
                self._servo_target_joints = target_joint_positions.copy()
                self._otg_target_joints = None
                if not self.cfg.continuous_physics:
                    self._advance_servo_window(target_joint_positions, self.cfg.teleop_dt)
            self._last_command_pose = self._tcp_pose.copy()
            self._reference_pose = None
            self._hold_joint_target = None
            self._prev_enabled = False
            self._step_count += 1
            observation = self._build_observation()
            terminated = False
            truncated = self._step_count >= self.cfg.max_episode_steps
            return observation, 0.0, terminated, truncated, self._build_info()

    def _build_info(self, *, include_camera_obs: bool = True) -> dict[str, Any]:
        with self._physics_lock:
            joint_positions = self._get_joint_positions()
            ee_pose = self._current_tcp_pose()
            info = {
                "joint_positions": joint_positions,
                "ee_pose": ee_pose,
                "target_frame_name": self.cfg.target_frame_name,
                "arm_gravity_compensation_enabled": bool(self.cfg.enable_arm_gravity_compensation),
            }
            if self.cfg.enable_cameras and include_camera_obs:
                info["camera_obs"] = self._build_camera_obs()
            info.update(self._build_visualization_info())
            return info

    def _get_renderer(self):
        current_thread_id = threading.get_ident()
        if self._renderer is not None and self._renderer_owner_thread_id != current_thread_id:
            self._renderer.close()
            self._renderer = None
            self._renderer_owner_thread_id = None
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model,
                height=self.cfg.camera_height,
                width=self.cfg.camera_width,
            )
            self._renderer_owner_thread_id = current_thread_id
        return self._renderer

    def render(self, *, blocking: bool = True):
        acquired = self._physics_lock.acquire(blocking=blocking)
        if not acquired:
            return None
        try:
            renderer = self._get_renderer()
            frames: dict[str, np.ndarray] = {}
            for camera_name in self.cfg.camera_names:
                model_camera_name = self.cfg.camera_name_mapping.get(camera_name, camera_name)
                renderer.update_scene(self.data, camera=model_camera_name)
                frames[camera_name] = np.asarray(renderer.render()).copy()
            return frames
        finally:
            self._physics_lock.release()

    def close(self) -> None:
        self._stop_continuous_physics_thread()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_owner_thread_id = None
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
