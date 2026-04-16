#!/usr/bin/env python3
"""
FR3 MuJoCo 仿真录制数据集重播运行时（容器内运行）

从 LeRobotDataset 读取录制的 state/action，在 MuJoCo 仿真中重播，
使用与录制时相同的 FR3 模型和相机配置。

数据格式（observation.state）：
    [ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, prev_cmd.ee.qx, ..., gripper.pos, ee.wx, ..., joint_1.pos, ...]
action：
    [ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, gripper.pos]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.envs.fr3_mujoco import _MujocoArmKinematics
from lerobot.utils.rotation import Rotation

_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_XML = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_ati_scene.xml"

_SIM_PHYSICS_HZ = 800.0
_ARM_GRAVITY_COMPENSATION_SCALE = 0.5
_GRIPPER_JOINT_NAMES = ("gripper_left_joint", "gripper_right_joint")
_GRIPPER_KEYS = ("left", "right")
_GRIPPER_STATE_INDEX = 11
_IK_SEED_JOINTS_RAD = np.array(
    [-0.057898, -1.550287, -1.694779, -2.125869, 0.022876, 2.119851, -0.948924],
    dtype=np.float64,
)


def load_episode(dataset_path: str, episode_idx: int) -> dict[str, np.ndarray]:
    """从 Parquet 数据集读取指定 episode 的 state/action/timestamp."""
    meta_dir = Path(dataset_path) / "meta" / "episodes"
    meta_files = sorted(meta_dir.rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata in {meta_dir}")

    chunk_idx = file_idx = None
    for mf in meta_files:
        t = pq.read_table(str(mf)).to_pydict()
        if episode_idx in t["episode_index"]:
            i = t["episode_index"].index(episode_idx)
            chunk_idx = t["data/chunk_index"][i]
            file_idx = t["data/file_index"][i]
            break
    if chunk_idx is None:
        raise ValueError(f"Episode {episode_idx} not found in {dataset_path}")

    data_file = Path(dataset_path) / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
    t = pq.read_table(str(data_file)).to_pydict()
    mask = [i for i, e in enumerate(t["episode_index"]) if e == episode_idx]

    return {
        "state": np.array([t["observation.state"][i] for i in mask], dtype=np.float64),
        "action": np.array([t["action"][i] for i in mask], dtype=np.float64),
        "timestamp": np.array([t["timestamp"][i] for i in mask], dtype=np.float64),
    }


def list_episode_indices(dataset_path: str) -> list[int]:
    meta_dir = Path(dataset_path) / "meta" / "episodes"
    meta_files = sorted(meta_dir.rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata in {meta_dir}")

    episode_indices: list[int] = []
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file)).to_pydict()
        episode_indices.extend(int(episode_index) for episode_index in table["episode_index"])

    ordered_unique_episode_indices = sorted(set(episode_indices))
    if not ordered_unique_episode_indices:
        raise ValueError(f"No episodes found in {dataset_path}")
    return ordered_unique_episode_indices


def state_joint_positions(states: np.ndarray, frame_idx: int) -> np.ndarray | None:
    if states.ndim != 2 or states.shape[1] < 22:
        return None
    return np.asarray(states[frame_idx, 15:22], dtype=np.float64)


def action_gripper(actions: np.ndarray, frame_idx: int, default: float = 1.0) -> float:
    if actions.ndim == 2 and actions.shape[1] > 7:
        return float(actions[frame_idx, 7])
    return default


def state_gripper(states: np.ndarray, actions: np.ndarray, frame_idx: int, default: float = 1.0) -> float:
    if states.ndim == 2 and states.shape[1] > _GRIPPER_STATE_INDEX:
        return float(states[frame_idx, _GRIPPER_STATE_INDEX])
    if states.ndim == 2 and states.shape[1] > 7:
        return float(states[frame_idx, 7])
    return action_gripper(actions, frame_idx, default=default)


def solve_initial_joints(
    kin: _MujocoArmKinematics,
    target_pose: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    joints = np.asarray(kin.inverse_kinematics(_IK_SEED_JOINTS_RAD.copy(), target_pose), dtype=np.float64)
    joints = np.clip(joints[:7], joint_lower, joint_upper)
    fk = np.asarray(kin.forward_kinematics(joints), dtype=np.float64)
    position_error = float(np.linalg.norm(target_pose[:3, 3] - fk[:3, 3]))
    return joints, fk, position_error


def get_joint_ids(mj, model, joint_names: list[str]) -> list[int]:
    ids = []
    for jname in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not in MuJoCo model")
        ids.append(jid)
    return ids


def get_joint_state_info(mj, model, joint_names: list[str]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = get_joint_ids(mj, model, joint_names)
    qpos_indices = np.asarray([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int64)
    qvel_indices = np.asarray([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=np.int64)
    joint_lower = model.jnt_range[joint_ids, 0].astype(np.float64)
    joint_upper = model.jnt_range[joint_ids, 1].astype(np.float64)
    return joint_ids, qpos_indices, qvel_indices, joint_lower, joint_upper


def get_gripper_state_info(mj, model) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[float, float]], int, np.ndarray]:
    joint_qpos_indices: dict[str, int] = {}
    joint_qvel_indices: dict[str, int] = {}
    joint_limits: dict[str, tuple[float, float]] = {}
    for key, joint_name in zip(_GRIPPER_KEYS, _GRIPPER_JOINT_NAMES, strict=True):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Gripper joint '{joint_name}' not in MuJoCo model")
        joint_qpos_indices[key] = int(model.jnt_qposadr[joint_id])
        joint_qvel_indices[key] = int(model.jnt_dofadr[joint_id])
        limits = model.jnt_range[joint_id].astype(np.float64)
        joint_limits[key] = (float(limits[0]), float(limits[1]))

    actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "pika_gripper_actuator")
    if actuator_id < 0:
        raise ValueError("Actuator 'pika_gripper_actuator' not in MuJoCo model")
    ctrl_range = model.actuator_ctrlrange[actuator_id].astype(np.float64)
    return joint_qpos_indices, joint_qvel_indices, joint_limits, actuator_id, ctrl_range


def get_arm_actuator_ids(mj, model, num_joints: int) -> np.ndarray:
    actuator_ids = []
    for index in range(num_joints):
        actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"fr3_actuator{index + 1}")
        if actuator_id < 0:
            raise ValueError(f"Actuator 'fr3_actuator{index + 1}' not in MuJoCo model")
        actuator_ids.append(int(actuator_id))
    return np.asarray(actuator_ids, dtype=np.int64)


def set_joint_state(mj, model, data, joint_ids: list[int], q_rad: np.ndarray) -> None:
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = q_rad[idx]
        data.qvel[int(model.jnt_dofadr[jid])] = 0.0
    mj.mj_forward(model, data)


def get_joint_positions(model, data, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qpos[int(model.jnt_qposadr[jid])] for jid in joint_ids], dtype=np.float64)


def pose_from_xyzquat(xyzquat: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4x4 SE(3)"""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyzquat[:3]
    T[:3, :3] = Rotation.from_quat(xyzquat[3:7]).as_matrix()
    return T


def rotation_angle_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    trace = float(np.trace(R1.T @ R2))
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def get_body_pose(data, body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float64)
    pose[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return pose


def gripper_joint_targets_from_command(
    gripper_command: float,
    joint_limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    command = float(np.clip(gripper_command, 0.0, 1.0))
    targets: dict[str, float] = {}
    for key, (lower, upper) in joint_limits.items():
        closed = 0.0 if lower <= 0.0 <= upper else (lower if abs(lower) < abs(upper) else upper)
        open_target = lower if abs(lower) > abs(upper) else upper
        targets[key] = float(closed + command * (open_target - closed))
    return targets


def gripper_ctrl_from_command(gripper_command: float, ctrl_range: np.ndarray) -> float:
    command = float(np.clip(gripper_command, 0.0, 1.0))
    lower, upper = ctrl_range
    return float(upper + command * (lower - upper))


def initialize_gripper_state(
    mj,
    model,
    data,
    *,
    gripper_command: float,
    joint_qpos_indices: dict[str, int],
    joint_qvel_indices: dict[str, int],
    joint_limits: dict[str, tuple[float, float]],
    actuator_id: int,
    ctrl_range: np.ndarray,
) -> None:
    data.ctrl[actuator_id] = gripper_ctrl_from_command(gripper_command, ctrl_range)
    targets = gripper_joint_targets_from_command(gripper_command, joint_limits)
    for key, qpos_index in joint_qpos_indices.items():
        data.qpos[qpos_index] = targets[key]
        data.qvel[joint_qvel_indices[key]] = 0.0
    mj.mj_forward(model, data)


def set_gripper_control(
    data,
    *,
    gripper_command: float,
    actuator_id: int,
    ctrl_range: np.ndarray,
) -> None:
    data.ctrl[actuator_id] = gripper_ctrl_from_command(gripper_command, ctrl_range)


def set_arm_target(data, actuator_ids: np.ndarray, joint_positions: np.ndarray, joint_lower: np.ndarray, joint_upper: np.ndarray) -> None:
    data.ctrl[actuator_ids] = np.clip(np.asarray(joint_positions, dtype=np.float64), joint_lower, joint_upper)


def apply_arm_gravity_compensation(
    mj,
    model,
    data,
    gravity_comp_data,
    qvel_indices: np.ndarray,
) -> None:
    gravity_comp_data.qpos[:] = data.qpos
    gravity_comp_data.qvel[:] = 0.0
    if hasattr(gravity_comp_data, "act") and hasattr(data, "act") and gravity_comp_data.act.shape == data.act.shape:
        gravity_comp_data.act[:] = data.act
    gravity_comp_data.ctrl[:] = data.ctrl
    mj.mj_forward(model, gravity_comp_data)
    data.qfrc_applied[qvel_indices] = np.asarray(
        _ARM_GRAVITY_COMPENSATION_SCALE * gravity_comp_data.qfrc_bias[qvel_indices],
        dtype=np.float64,
    )


def step_physics(
    mj,
    model,
    data,
    *,
    gravity_comp_data,
    qvel_indices: np.ndarray,
    num_steps: int,
) -> None:
    for _ in range(max(int(num_steps), 1)):
        apply_arm_gravity_compensation(mj, model, data, gravity_comp_data, qvel_indices)
        mj.mj_step(model, data)
    mj.mj_forward(model, data)


def replay_episode(
    *,
    args: argparse.Namespace,
    episode_idx: int,
    mujoco,
    model,
    mj_data,
    joint_ids: list[int],
    qvel_indices: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    arm_actuator_ids: np.ndarray,
    gripper_qpos_indices: dict[str, int],
    gripper_qvel_indices: dict[str, int],
    gripper_joint_limits: dict[str, tuple[float, float]],
    gripper_actuator_id: int,
    gripper_ctrl_range: np.ndarray,
    gravity_comp_data,
    kin_ee: _MujocoArmKinematics,
    kin_tcp: _MujocoArmKinematics,
    viewer,
) -> int:
    print(f"[INFO] 加载 episode {episode_idx}  dataset={args.dataset}")
    ep = load_episode(args.dataset, episode_idx)
    states = ep["state"]
    actions = ep["action"]
    n_frames = len(states)
    print(f"[INFO] {n_frames} 帧 @ {args.fps} fps")
    if states.ndim != 2 or states.shape[1] < 7:
        raise ValueError(f"observation.state must include at least 7D EE pose, got shape {states.shape}")
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"action must include at least 7D EE pose, got shape {actions.shape}")

    print(f"[DEBUG] 录制 EE state[0]: {states[0][:7]}")
    print(f"[DEBUG] 录制 action[0]: {actions[0][:7]}")
    initial_gripper = state_gripper(states, actions, 0)
    print(f"[DEBUG] 录制 gripper[0]: state/fallback={initial_gripper:.3f} action={action_gripper(actions, 0):.3f}")

    print(f"[DEBUG] state[0] ee pos from xyzquat: {states[0, :3]}")
    T_B_E_recorded_0 = pose_from_xyzquat(states[0, :7])
    print(f"[DEBUG] state[0] xyzquat → T_B_E: pos={T_B_E_recorded_0[:3, 3]}")

    record_joints_0 = state_joint_positions(states, 0)
    if record_joints_0 is not None:
        initial_joint_positions = record_joints_0.copy()
        print(f"[DEBUG] state[0] joint_positions: {record_joints_0}")
        record_FK_ee = np.asarray(kin_ee.forward_kinematics(record_joints_0), dtype=np.float64)
        record_FK_tcp = np.asarray(kin_tcp.forward_kinematics(record_joints_0), dtype=np.float64)
        diff_ee = float(np.linalg.norm(T_B_E_recorded_0[:3, 3] - record_FK_ee[:3, 3]))
        diff_tcp = float(np.linalg.norm(T_B_E_recorded_0[:3, 3] - record_FK_tcp[:3, 3]))
    else:
        print("[INFO] observation.state lacks joint positions; solving initial joints from 7D EE pose")
        joints_ee, record_FK_ee, diff_ee = solve_initial_joints(kin_ee, T_B_E_recorded_0, joint_lower, joint_upper)
        joints_tcp, record_FK_tcp, diff_tcp = solve_initial_joints(kin_tcp, T_B_E_recorded_0, joint_lower, joint_upper)
        initial_joint_positions = joints_tcp if diff_tcp < diff_ee else joints_ee
    print(f"[DEBUG] FK candidate (pika_gripper_ee): {record_FK_ee[:3, 3]}")
    print(f"[DEBUG] FK candidate (pika_task_tcp): {record_FK_tcp[:3, 3]}")
    print(f"[DEBUG] pos diff to pika_gripper_ee: {diff_ee*1000:.1f}mm")
    print(f"[DEBUG] pos diff to pika_task_tcp: {diff_tcp*1000:.1f}mm")

    replay_body_name = "pika_task_tcp" if diff_tcp < diff_ee else "pika_gripper_ee"
    kin_for_replay = kin_tcp if replay_body_name == "pika_task_tcp" else kin_ee
    replay_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, replay_body_name)
    if replay_body_id < 0:
        raise ValueError(f"Body '{replay_body_name}' not in MuJoCo model")
    print(f"[INFO] Using {replay_body_name} for replay (closer match)")

    set_joint_state(mujoco, model, mj_data, joint_ids, initial_joint_positions)
    set_arm_target(mj_data, arm_actuator_ids, initial_joint_positions, joint_lower, joint_upper)
    initialize_gripper_state(
        mujoco,
        model,
        mj_data,
        gripper_command=initial_gripper,
        joint_qpos_indices=gripper_qpos_indices,
        joint_qvel_indices=gripper_qvel_indices,
        joint_limits=gripper_joint_limits,
        actuator_id=gripper_actuator_id,
        ctrl_range=gripper_ctrl_range,
    )

    T_B_E_initial = np.asarray(kin_for_replay.forward_kinematics(initial_joint_positions), dtype=np.float64)
    print(f"[DEBUG] Sim initial EE pose (kin, initial joints): {T_B_E_initial[:3, 3]}")

    current_joints_rad = initial_joint_positions.copy()
    physics_steps_per_frame = max(1, round(_SIM_PHYSICS_HZ / args.fps))

    pos_errors_mm: list[float] = []
    rot_errors_deg: list[float] = []
    previous_gripper_target = initial_gripper
    completed_frames = 0

    print(f"\n[INFO] 开始重播 ({n_frames} 帧)…\n")

    for fi in range(n_frames):
        t0 = time.perf_counter()

        action = actions[fi]
        ee_target = action[:7]
        gripper_target = action_gripper(actions, fi, default=previous_gripper_target)

        T_B_Et_star = pose_from_xyzquat(ee_target)
        target_joints_ik = np.asarray(kin_for_replay.inverse_kinematics(current_joints_rad, T_B_Et_star), dtype=np.float64)
        target_joints_rad = np.clip(target_joints_ik[:7], joint_lower, joint_upper)

        set_arm_target(mj_data, arm_actuator_ids, target_joints_rad, joint_lower, joint_upper)
        set_gripper_control(
            mj_data,
            gripper_command=gripper_target,
            actuator_id=gripper_actuator_id,
            ctrl_range=gripper_ctrl_range,
        )
        step_physics(
            mujoco,
            model,
            mj_data,
            gravity_comp_data=gravity_comp_data,
            qvel_indices=qvel_indices,
            num_steps=physics_steps_per_frame,
        )
        current_joints_rad = get_joint_positions(model, mj_data, joint_ids)

        T_B_Et_sim = get_body_pose(mj_data, replay_body_id)

        state = states[fi]
        ee_recorded = state[:7]
        T_B_Et_recorded = pose_from_xyzquat(ee_recorded)

        pos_err_mm = float(np.linalg.norm(T_B_Et_sim[:3, 3] - T_B_Et_recorded[:3, 3]) * 1000)
        rot_err_deg = rotation_angle_error_deg(T_B_Et_sim[:3, :3], T_B_Et_recorded[:3, :3])
        pos_errors_mm.append(pos_err_mm)
        rot_errors_deg.append(rot_err_deg)
        completed_frames = fi + 1

        if abs(gripper_target - previous_gripper_target) > 1e-6:
            print(f"  frame {fi:4d}/{n_frames}: gripper={gripper_target:.2f}")
            previous_gripper_target = gripper_target

        if fi % 30 == 0:
            print(f"  frame {fi:4d}/{n_frames}: pos_err={pos_err_mm:.2f}mm  rot_err={rot_err_deg:.2f}deg")

        if viewer is not None and not viewer.is_running():
            break
        if viewer is not None:
            viewer.sync()

        dt_s = time.perf_counter() - t0
        sleep_s = max(1.0 / args.fps - dt_s, 0.0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    avg_pos_mm = float(np.mean(pos_errors_mm)) if pos_errors_mm else float("nan")
    max_pos_mm = float(np.max(pos_errors_mm)) if pos_errors_mm else float("nan")
    avg_rot_deg = float(np.mean(rot_errors_deg)) if rot_errors_deg else float("nan")
    max_rot_deg = float(np.max(rot_errors_deg)) if rot_errors_deg else float("nan")
    finished = completed_frames >= n_frames

    print(f"\n[RESULT] 重播{'完成' if finished else '提前结束'}:")
    print(f"  平均位置误差: {avg_pos_mm:.2f} mm")
    print(f"  最大位置误差: {max_pos_mm:.2f} mm")
    print(f"  平均旋转误差: {avg_rot_deg:.2f} deg")
    print(f"  最大旋转误差: {max_rot_deg:.2f} deg")
    print(
        "mujoco_replay_result="
        f"status={'complete' if finished else 'incomplete'} "
        f"completed_frames={completed_frames} total_frames={n_frames} "
        f"avg_pos_mm={avg_pos_mm:.6f} max_pos_mm={max_pos_mm:.6f} "
        f"avg_rot_deg={avg_rot_deg:.6f} max_rot_deg={max_rot_deg:.6f}"
    )

    return 0 if finished else 3


def replay(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer as mj_viewer

    print(f"[INFO] 加载 MuJoCo: {_SIM_XML}")
    model = mujoco.MjModel.from_xml_path(str(_SIM_XML))
    mj_data = mujoco.MjData(model)
    joint_ids, qpos_indices, qvel_indices, joint_lower, joint_upper = get_joint_state_info(mujoco, model, _JOINT_NAMES)
    arm_actuator_ids = get_arm_actuator_ids(mujoco, model, len(_JOINT_NAMES))
    (
        gripper_qpos_indices,
        gripper_qvel_indices,
        gripper_joint_limits,
        gripper_actuator_id,
        gripper_ctrl_range,
    ) = get_gripper_state_info(mujoco, model)
    gravity_comp_data = mujoco.MjData(model)

    kin_ee = _MujocoArmKinematics(
        mujoco=mujoco,
        model=model,
        target_frame_name="pika_gripper_ee",
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )
    kin_tcp = _MujocoArmKinematics(
        mujoco=mujoco,
        model=model,
        target_frame_name="pika_task_tcp",
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )

    episode_indices = [args.episode] if args.episode is not None else list_episode_indices(args.dataset)
    if args.episode is None:
        print(f"[INFO] 未指定 --episode，连续重播全部 episodes: {episode_indices}")

    viewer = None
    if not args.no_viewer:
        viewer = mj_viewer.launch_passive(model, mj_data)
        print("[INFO] MuJoCo Viewer 已启动")

    try:
        for index, episode_idx in enumerate(episode_indices):
            if len(episode_indices) > 1:
                print(f"\n[INFO] ===== Episode {episode_idx} ({index + 1}/{len(episode_indices)}) =====")
            status = replay_episode(
                args=args,
                episode_idx=episode_idx,
                mujoco=mujoco,
                model=model,
                mj_data=mj_data,
                joint_ids=joint_ids,
                qvel_indices=qvel_indices,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                arm_actuator_ids=arm_actuator_ids,
                gripper_qpos_indices=gripper_qpos_indices,
                gripper_qvel_indices=gripper_qvel_indices,
                gripper_joint_limits=gripper_joint_limits,
                gripper_actuator_id=gripper_actuator_id,
                gripper_ctrl_range=gripper_ctrl_range,
                gravity_comp_data=gravity_comp_data,
                kin_ee=kin_ee,
                kin_tcp=kin_tcp,
                viewer=viewer,
            )
            if status != 0:
                return status
            if viewer is not None and not viewer.is_running():
                break
        return 0
    finally:
        if viewer is not None:
            # On some NVIDIA/GLX Docker setups, explicit passive-viewer teardown
            # segfaults after a successful replay. Let process exit close it.
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 sim recording dataset replay runtime")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    status = replay(args)
    if not args.no_viewer:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
