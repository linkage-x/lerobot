#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig
from lerobot.utils.rotation import Rotation


def _default_quest3_pika_xml_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "quest3_pika_gripper_scene.xml"
    )


@dataclass
class Quest3PikaMujocoEnvConfig(FR3MujocoEnvConfig):
    sim_xml_path: str = field(default_factory=_default_quest3_pika_xml_path)
    target_frame_name: str = "pika_task_tcp"
    scene_mode: str = "quest3_pika_gripper"
    quest3_position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quest3_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quest3_recenter_on_first_tracking: bool = True
    quest3_incremental_mode: bool = True
    quest3_follow_orientation: bool = True
    quest3_rotation_alignment_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    quest3_fixed_tcp_quat_xyzw: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    initial_tcp_position: tuple[float, float, float] = (0.48, 0.0, 0.45)
    workspace_min: tuple[float, float, float] = (0.20, -0.80, -0.20)
    workspace_max: tuple[float, float, float] = (1.10, 0.80, 1.40)
    continuous_physics_frequency: float | None = 500.0
    quest3_env_deadband_m: float = 0.0005
    quest3_env_deadband_rad: float = 0.0
    quest3_env_max_step_pos_m: float = 0.30
    quest3_env_max_step_rot_rad: float = 3.14
    quest3_env_filter_alpha_pos: float = 1.0
    quest3_env_filter_alpha_rot: float = 1.0
    quest3_gripper_binary: bool = True
    quest3_gripper_binary_threshold: float = 0.5


class Quest3PikaMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, cfg: Quest3PikaMujocoEnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or Quest3PikaMujocoEnvConfig()
        self.render_mode = self.cfg.render_mode
        self._mujoco = self._import_mujoco()
        self.model = self._mujoco.MjModel.from_xml_path(self.cfg.sim_xml_path)
        self.data = self._mujoco.MjData(self.model)
        self._physics_lock = threading.RLock()
        self._continuous_physics_stop = threading.Event()
        self._continuous_physics_thread: threading.Thread | None = None
        self._renderer = None
        self._renderer_owner_thread_id: int | None = None
        self._step_count = 0

        self._workspace_min = np.asarray(self.cfg.workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(self.cfg.workspace_max, dtype=np.float64)
        self._position_scale = np.asarray(self.cfg.quest3_position_scale, dtype=np.float64)
        self._position_offset = np.asarray(self.cfg.quest3_position_offset, dtype=np.float64)
        self._rotation_alignment = Rotation.from_quat(self.cfg.quest3_rotation_alignment_xyzw)
        self._fixed_tcp_rotation = Rotation.from_quat(self.cfg.quest3_fixed_tcp_quat_xyzw)

        self._tcp_body_id = self._body_id(self.cfg.target_frame_name)
        self._gripper_base_body_id = self._body_id("gripper_base")
        self._gripper_left_body_id = self._body_id("gripper_left_link")
        self._gripper_right_body_id = self._body_id("gripper_right_link")
        self._table_geom_id = self._geom_id("table")
        self._workspace_object_geom_id = self._geom_id("workspace_object")
        self._gripper_left_geom_id = self._geom_id("gripper_left_collision")
        self._gripper_right_geom_id = self._geom_id("gripper_right_collision")
        self._gripper_left_pad_geom_id = -1
        self._gripper_right_pad_geom_id = -1

        self._mocap_id = int(self.model.body_mocapid[self._gripper_base_body_id])
        if self._mocap_id < 0:
            raise ValueError("Body 'gripper_base' must be a mocap body in the Quest3 Pika scene.")

        self._base_to_tcp = self._local_body_pose(self._tcp_body_id)
        self._tcp_to_base = np.linalg.inv(self._base_to_tcp)

        self._gripper_joint_indices: dict[str, int] = {}
        self._gripper_qvel_indices: dict[str, int] = {}
        self._gripper_joint_limits: dict[str, tuple[float, float]] = {}
        for key, joint_name in zip(("left", "right"), self.cfg.gripper_joint_names, strict=True):
            joint_id = self._joint_id(joint_name)
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

        self._target_pose: np.ndarray | None = None
        self._tcp_pose: np.ndarray | None = None
        self._last_gripper = float(np.clip(self.cfg.initial_gripper, 0.0, 1.0))
        self._last_command_pose = self._initial_tcp_pose()
        self._quest3_origin_wrist_pos: np.ndarray | None = None
        self._last_quest3_wrist_pos: np.ndarray | None = None
        self._last_mapped_tcp_pos = np.asarray(self.cfg.initial_tcp_position, dtype=np.float64).copy()
        self._zero_joints = np.zeros(len(self.cfg.joint_names), dtype=np.float64)
        self._prev_clutch_active = False
        self._mocap_baseline_pos: np.ndarray | None = None
        self._mocap_baseline_quat_wxyz: np.ndarray | None = None
        self._prev_filtered_dp: np.ndarray | None = None
        self._prev_filtered_dr_rotvec: np.ndarray | None = None

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.cfg.joint_names),), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.cfg.joint_names),), dtype=np.float32),
                "environment_state": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            }
        )

        self._apply_tcp_pose(self._last_command_pose)
        self._set_gripper_command(self._last_gripper, teleport=True)

    @staticmethod
    def _import_mujoco():
        try:
            import mujoco
        except Exception as e:
            raise ImportError("Quest3 Pika MuJoCo simulation requires the `mujoco` Python package.") from e
        return mujoco

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model.")
        return int(body_id)

    def _joint_id(self, name: str) -> int:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model.")
        return int(joint_id)

    def _geom_id(self, name: str) -> int:
        return int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, name))

    def _local_body_pose(self, body_id: int) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(self.model.body_pos[body_id], dtype=np.float64)
        quat_wxyz = np.asarray(self.model.body_quat[body_id], dtype=np.float64)
        pose[:3, :3] = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
        return pose

    def _initial_tcp_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(self.cfg.initial_tcp_position, dtype=np.float64)
        pose[:3, :3] = self._fixed_tcp_rotation.as_matrix()
        return pose

    def _pose_to_wxyz(self, pose: np.ndarray) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    def _apply_tcp_pose(self, tcp_pose: np.ndarray) -> None:
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64).reshape(4, 4)
        base_pose = tcp_pose @ self._tcp_to_base
        self.data.mocap_pos[self._mocap_id] = base_pose[:3, 3]
        self.data.mocap_quat[self._mocap_id] = self._pose_to_wxyz(base_pose)
        self._mujoco.mj_forward(self.model, self.data)
        self._target_pose = tcp_pose.copy()
        self._tcp_pose = self._current_tcp_pose()

    def _pose_from_quest3_action(self, action: dict[str, Any]) -> np.ndarray | None:
        if not bool(action.get("tracking_valid", False)):
            return None
        wrist_pos = np.array(
            [action.get("wrist_x", 0.0), action.get("wrist_y", 0.0), action.get("wrist_z", 0.0)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(wrist_pos)):
            return None

        self._last_quest3_wrist_pos = wrist_pos.copy()
        if self.cfg.quest3_recenter_on_first_tracking:
            if self._quest3_origin_wrist_pos is None:
                self._quest3_origin_wrist_pos = wrist_pos.copy()
            target_pos = (
                np.asarray(self.cfg.initial_tcp_position, dtype=np.float64)
                + (wrist_pos - self._quest3_origin_wrist_pos) * self._position_scale
                + self._position_offset
            )
        else:
            target_pos = wrist_pos * self._position_scale + self._position_offset

        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.clip(target_pos, self._workspace_min, self._workspace_max)
        self._last_mapped_tcp_pos = pose[:3, 3].copy()
        if self.cfg.quest3_follow_orientation:
            wrist_quat = np.array(
                [
                    action.get("wrist_qx", 0.0),
                    action.get("wrist_qy", 0.0),
                    action.get("wrist_qz", 0.0),
                    action.get("wrist_qw", 1.0),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(wrist_quat) > 1e-6 and np.all(np.isfinite(wrist_quat)):
                pose[:3, :3] = (Rotation.from_quat(wrist_quat) * self._rotation_alignment).as_matrix()
            else:
                pose[:3, :3] = self._fixed_tcp_rotation.as_matrix()
        else:
            pose[:3, :3] = self._fixed_tcp_rotation.as_matrix()
        return pose

    def _current_tcp_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(self.data.xpos[self._tcp_body_id], dtype=np.float64)
        pose[:3, :3] = np.asarray(self.data.xmat[self._tcp_body_id], dtype=np.float64).reshape(3, 3)
        return pose

    def _pose_to_seven_d(self, pose: np.ndarray) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return np.concatenate([pose[:3, 3], quat_xyzw], dtype=np.float64)

    def _get_gripper_joint_positions(self) -> dict[str, float]:
        return {key: float(self.data.qpos[qpos_index]) for key, qpos_index in self._gripper_joint_indices.items()}

    def _gripper_joint_targets_from_command(self, gripper_command: float) -> dict[str, float]:
        command = float(np.clip(gripper_command, 0.0, 1.0))
        targets: dict[str, float] = {}
        for key, (lower, upper) in self._gripper_joint_limits.items():
            closed = 0.0 if lower <= 0.0 <= upper else (lower if abs(lower) < abs(upper) else upper)
            open_target = lower if abs(lower) > abs(upper) else upper
            targets[key] = float(closed + command * (open_target - closed))
        return targets

    def _gripper_ctrl_from_command(self, gripper_command: float) -> float:
        lower, upper = self._gripper_ctrl_range
        command = float(np.clip(gripper_command, 0.0, 1.0))
        return float(upper - command * (upper - lower))

    def _set_gripper_command(self, gripper_command: float, *, simulate: bool = False, teleport: bool = False) -> None:
        if self.cfg.quest3_gripper_binary:
            gripper_command = 1.0 if float(gripper_command) >= float(self.cfg.quest3_gripper_binary_threshold) else 0.0
        self.data.ctrl[self._gripper_actuator_id] = self._gripper_ctrl_from_command(gripper_command)
        if teleport:
            targets = self._gripper_joint_targets_from_command(gripper_command)
            for key, qpos_index in self._gripper_joint_indices.items():
                self.data.qpos[qpos_index] = targets[key]
                self.data.qvel[self._gripper_qvel_indices[key]] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        if simulate:
            self._step_physics(max(int(self.cfg.gripper_sim_steps), 1))

    def _step_physics(self, steps: int) -> None:
        for _ in range(max(int(steps), 1)):
            self._mujoco.mj_step(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        self._tcp_pose = self._current_tcp_pose()

    def _continuous_physics_loop(self) -> None:
        dt = 1.0 / float(self.cfg.continuous_physics_frequency or 500.0)
        while not self._continuous_physics_stop.is_set():
            step_start = time.perf_counter()
            with self._physics_lock:
                self._step_physics(1)
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
            name="quest3-pika-mujoco-physics",
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

    def _build_visualization_info(self) -> dict[str, Any]:
        target_pose = self._target_pose if self._target_pose is not None else self._current_tcp_pose()
        tcp_pose = self._current_tcp_pose()
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

    def copy_visual_state(self, target_data) -> None:
        with self._physics_lock:
            target_data.qpos[:] = self.data.qpos
            target_data.qvel[:] = self.data.qvel
            target_data.ctrl[:] = self.data.ctrl
            if hasattr(target_data, "mocap_pos"):
                target_data.mocap_pos[:] = self.data.mocap_pos
            if hasattr(target_data, "mocap_quat"):
                target_data.mocap_quat[:] = self.data.mocap_quat
            self._mujoco.mj_forward(self.model, target_data)

    def _quat_wxyz_to_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)

    def _quat_xyzw_to_wxyz(self, quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    def _apply_env_filtering(self, dp: np.ndarray, dr_rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm_p = float(np.linalg.norm(dp))
        deadband_p = float(self.cfg.quest3_env_deadband_m)
        if norm_p < deadband_p:
            dp = np.zeros(3, dtype=np.float64)

        max_step_p = float(self.cfg.quest3_env_max_step_pos_m)
        if max_step_p > 0.0 and norm_p > max_step_p:
            dp = dp / norm_p * max_step_p

        alpha_p = float(np.clip(self.cfg.quest3_env_filter_alpha_pos, 0.0, 1.0))
        if self._prev_filtered_dp is not None and alpha_p > 0.0:
            dp = alpha_p * dp + (1.0 - alpha_p) * self._prev_filtered_dp
        self._prev_filtered_dp = dp.copy()

        norm_r = float(np.linalg.norm(dr_rotvec))
        deadband_r = float(self.cfg.quest3_env_deadband_rad)
        if norm_r < deadband_r:
            dr_rotvec = np.zeros(3, dtype=np.float64)

        max_step_r = float(self.cfg.quest3_env_max_step_rot_rad)
        if max_step_r > 0.0 and norm_r > max_step_r:
            dr_rotvec = dr_rotvec / norm_r * max_step_r

        alpha_r = float(np.clip(self.cfg.quest3_env_filter_alpha_rot, 0.0, 1.0))
        if self._prev_filtered_dr_rotvec is not None and alpha_r > 0.0:
            dr_rotvec = alpha_r * dr_rotvec + (1.0 - alpha_r) * self._prev_filtered_dr_rotvec
        self._prev_filtered_dr_rotvec = dr_rotvec.copy()

        return dp, dr_rotvec

    def _slerp_wxyz(self, q0_wxyz: np.ndarray, q1_wxyz: np.ndarray, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, 1.0))
        q0 = np.asarray(q0_wxyz, dtype=np.float64)
        q1 = np.asarray(q1_wxyz, dtype=np.float64)
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
        s1 = np.sin(t * theta_0) / sin_theta_0
        return s0 * q0 + s1 * q1

    def _apply_incremental_mocap(
        self,
        delta_pos: np.ndarray,
        delta_rotvec: np.ndarray,
    ) -> None:
        if self._mocap_baseline_pos is None or self._mocap_baseline_quat_wxyz is None:
            return

        dp, dr = self._apply_env_filtering(delta_pos.copy(), delta_rotvec.copy())
        p_target = self._mocap_baseline_pos + dp
        p_target = np.clip(p_target, self._workspace_min, self._workspace_max)

        q_target_wxyz = self._mocap_baseline_quat_wxyz.copy()
        if np.linalg.norm(dr) > 0.0 and self.cfg.quest3_follow_orientation:
            q_baseline_xyzw = self._quat_wxyz_to_xyzw(self._mocap_baseline_quat_wxyz)
            R_baseline = Rotation.from_quat(q_baseline_xyzw)
            R_delta = Rotation.from_rotvec(dr)
            R_target = R_delta * R_baseline * self._rotation_alignment
            q_target_xyzw = R_target.as_quat()
            q_target_wxyz = self._quat_xyzw_to_wxyz(q_target_xyzw)

        base_pose = np.eye(4, dtype=np.float64)
        base_pose[:3, 3] = p_target
        R_target = Rotation.from_quat(self._quat_wxyz_to_xyzw(q_target_wxyz))
        base_pose[:3, :3] = R_target.as_matrix()
        self.data.mocap_pos[self._mocap_id] = base_pose[:3, 3]
        self.data.mocap_quat[self._mocap_id] = q_target_wxyz
        self._mujoco.mj_forward(self.model, self.data)

        self._target_pose = base_pose.copy()
        self._tcp_pose = self._current_tcp_pose()
        self._last_mapped_tcp_pos = p_target.copy()

    def step_teleop_action(
        self,
        action: dict[str, Any] | None,
        control_period_s: float | None = None,
        *,
        include_camera_obs_in_observation: bool = True,
        include_camera_obs_in_info: bool = True,
    ):
        with self._physics_lock:
            action = {} if action is None else dict(action)
            clutch_active = bool(action.get("clutch_active", False))

            if self.cfg.quest3_incremental_mode:
                if clutch_active and not self._prev_clutch_active:
                    self._mocap_baseline_pos = self.data.mocap_pos[self._mocap_id].copy()
                    self._mocap_baseline_quat_wxyz = self.data.mocap_quat[self._mocap_id].copy()
                    self._prev_filtered_dp = None
                    self._prev_filtered_dr_rotvec = None

                if clutch_active:
                    delta_pos = np.array(
                        [action.get("delta_x", 0.0), action.get("delta_y", 0.0), action.get("delta_z", 0.0)],
                        dtype=np.float64,
                    )
                    delta_rotvec = np.array(
                        [action.get("delta_wx", 0.0), action.get("delta_wy", 0.0), action.get("delta_wz", 0.0)],
                        dtype=np.float64,
                    )
                    if np.all(np.isfinite(delta_pos)) and np.all(np.isfinite(delta_rotvec)):
                        self._apply_incremental_mocap(delta_pos, delta_rotvec)
                else:
                    self._mocap_baseline_pos = None
                    self._mocap_baseline_quat_wxyz = None
                    self._prev_filtered_dp = None
                    self._prev_filtered_dr_rotvec = None
            else:
                target_pose = self._pose_from_quest3_action(action)
                if target_pose is not None:
                    self._last_command_pose = target_pose.copy()
                self._apply_tcp_pose(self._last_command_pose)

            self._prev_clutch_active = clutch_active

            previous_gripper = self._last_gripper
            self._last_gripper = float(np.clip(action.get("gripper", self._last_gripper), 0.0, 1.0))
            if not np.isclose(previous_gripper, self._last_gripper):
                self._set_gripper_command(self._last_gripper)

            if not self.cfg.continuous_physics:
                duration = self.cfg.teleop_dt if control_period_s is None else max(float(control_period_s), 0.0)
                steps = max(1, int(np.ceil(duration / self.model.opt.timestep)))
                self._step_physics(steps)

            self._step_count += 1
            observation = self._build_observation(include_camera_obs=include_camera_obs_in_observation)
            truncated = self._step_count >= self.cfg.max_episode_steps
            info = self._build_info(include_camera_obs=include_camera_obs_in_info)
            info["teleop_action"] = action.copy()
            info["target_joint_positions"] = self._zero_joints.copy()
            info["otg_enabled"] = False
            info["otg_steps"] = 0
            info["sender_steps"] = 0
            return observation, 0.0, False, truncated, info

    def _build_observation(self, *, include_camera_obs: bool = True) -> dict[str, Any]:
        ee_pose = self._current_tcp_pose()
        env_state = np.concatenate([ee_pose[:3, 3], ee_pose[:3, :3].reshape(-1)], dtype=np.float64)
        observation: dict[str, Any] = {
            "agent_pos": self._zero_joints.astype(np.float32),
            "environment_state": env_state.astype(np.float32),
        }
        if self.cfg.enable_cameras and include_camera_obs:
            observation["camera_obs"] = self.render()
        return observation

    def _build_info(self, *, include_camera_obs: bool = True) -> dict[str, Any]:
        info: dict[str, Any] = {
            "joint_positions": self._zero_joints.copy(),
            "ee_pose": self._current_tcp_pose(),
            "target_frame_name": self.cfg.target_frame_name,
            "arm_gravity_compensation_enabled": False,
            "scene_mode": self.cfg.scene_mode,
            "quest3_recentered": self._quest3_origin_wrist_pos is not None,
            "quest3_wrist_position": None if self._last_quest3_wrist_pos is None else self._last_quest3_wrist_pos.copy(),
            "quest3_origin_wrist_position": None
            if self._quest3_origin_wrist_pos is None
            else self._quest3_origin_wrist_pos.copy(),
            "quest3_mapped_tcp_position": self._last_mapped_tcp_pos.copy(),
        }
        if self.cfg.enable_cameras and include_camera_obs:
            info["camera_obs"] = self.render()
        info.update(self._build_visualization_info())
        return info

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
            self._last_command_pose = self._initial_tcp_pose()
            self._quest3_origin_wrist_pos = None
            self._last_quest3_wrist_pos = None
            self._last_mapped_tcp_pos = np.asarray(self.cfg.initial_tcp_position, dtype=np.float64).copy()
            self._prev_clutch_active = False
            self._mocap_baseline_pos = None
            self._mocap_baseline_quat_wxyz = None
            self._prev_filtered_dp = None
            self._prev_filtered_dr_rotvec = None
            if options and "tcp_pose" in options:
                self._last_command_pose = np.asarray(options["tcp_pose"], dtype=np.float64).reshape(4, 4)
            self._last_gripper = float(np.clip(self.cfg.initial_gripper, 0.0, 1.0))
            self._apply_tcp_pose(self._last_command_pose)
            self._set_gripper_command(self._last_gripper)
            self._step_physics(1)
            if self.cfg.continuous_physics:
                self._ensure_continuous_physics_thread()
            return (
                self._build_observation(include_camera_obs=include_camera_obs_in_observation),
                self._build_info(include_camera_obs=include_camera_obs_in_info),
            )

    def step(self, action: np.ndarray):
        return self.step_teleop_action(None)

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
