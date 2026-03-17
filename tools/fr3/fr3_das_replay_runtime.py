#!/usr/bin/env python3
"""
FR3 DAS 数据集 MuJoCo 重播运行时（容器内运行）

从 LeRobot v3 Parquet 数据集读取录制的 EE 轨迹，通过坐标变换 + IK + OTG
在 MuJoCo 中重播，并用 Rerun 可视化重播轨迹 vs 录制轨迹。

数据格式（observation.state / action）：
    [x, y, z, qx, qy, qz, qw, gripper]
    坐标系：SLAM world frame W_s（每个 episode 起始位置为原点）

坐标变换链：
    T(B, E_t) = T(B, E_reset) * inv(T(W_s, E_0)) * T(W_s, E_t)

    其中：
        T(B, E_reset) = pose_from_xyzquat(reset_arm_command)
                        录制前机械臂 reset 笛卡尔位姿（reset_space="cartesian"）
        T(W_s, E_0)   = pose_from_xyzquat(state[0]) — 首帧 SLAM 位姿
        T(W_s, E_t)   = pose_from_xyzquat(state[t])
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 路径常量（容器内 /lerobot 挂载到 repo 根）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAS_XML = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.xml"
_DAS_URDF = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf"

_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

# 录制前机械臂 reset 位姿（B 系笛卡尔，reset_space="cartesian"）
# 格式：[x, y, z, qx, qy, qz, qw]（xyzw，标量在后），供 pose_from_xyzquat 使用
# 原始 reset_arm_command 四元数（xyzw）：qx=0.7071068, qy=0, qz=0.7071068, qw=0
# → 绕 [1,0,1]/√2 轴 180° 旋转
_RESET_POSE_B_XYZQUAT = np.array(
    [0.15327496, -0.54249998, 0.27, 0.7071068, 0.0, 0.7071068, 0.0],
    dtype=np.float64,
)

# MuJoCo 仿真起始关节角（从真实 FR3 192.168.1.208 查询，2026-03-17）
# 从真实机器人当前位置出发，IK 收敛更快，消除启动瞬态误差
_IK_SEED_JOINTS_RAD = np.array(
    [-0.057898, -1.550287, -1.694779, -2.125869, 0.022876, 2.119851, -0.948924],
    dtype=np.float64,
)

# OTG 限制
_OTG_CTRL_HZ = 800.0
_OTG_MAX_VEL = [2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208]
_OTG_MAX_ACC = [8.0] * 7
_OTG_MAX_JERK = [4000.0] * 7
_OTG_MIN_POS = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
_OTG_MAX_POS = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]


# ---------------------------------------------------------------------------
# SE(3) 工具
# ---------------------------------------------------------------------------


def pose_from_xyzquat(xyzquat: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4x4 SE(3)（scipy quat 约定：[x,y,z,w]）"""
    from scipy.spatial.transform import Rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyzquat[:3]
    T[:3, :3] = Rotation.from_quat(xyzquat[3:7]).as_matrix()
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """SE(3) 解析逆"""
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rotation_angle_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """两个旋转矩阵之间的角度误差（°）"""
    trace = np.clip(np.trace(R1.T @ R2), -1.0, 3.0)
    return float(np.degrees(np.arccos((trace - 1.0) / 2.0)))


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_episode(dataset_path: str, episode_idx: int) -> dict[str, np.ndarray]:
    """从 Parquet 数据集读取指定 episode 的 state/action/timestamp。

    Returns:
        state:     np.ndarray[N, 8]  — T(W_s, E_t) 在 W_s 系下的 EE pose + gripper
        action:    np.ndarray[N, 8]  — 发送的 EE 目标 + gripper
        timestamp: np.ndarray[N]
    """
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

    data_file = Path(dataset_path) / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:06d}.parquet"
    t = pq.read_table(str(data_file)).to_pydict()
    mask = [i for i, e in enumerate(t["episode_index"]) if e == episode_idx]

    return {
        "state": np.array([t["observation.state"][i] for i in mask], dtype=np.float64),
        "action": np.array([t["action"][i] for i in mask], dtype=np.float64),
        "timestamp": np.array([t["timestamp"][i] for i in mask], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 坐标变换
# ---------------------------------------------------------------------------


def build_T_B_Ws(T_B_E_reset: np.ndarray, T_Ws_E0: np.ndarray) -> np.ndarray:
    """
    计算 T(B, W_s)。

    T(B, W_s) = T(B, E_reset) * T(E_0, W_s)
              = T(B, E_reset) * inv(T(W_s, E_0))

    Args:
        T_B_E_reset: 录制前机械臂 reset 位姿在 B 系下的 4x4 SE(3)
                     由 reset_arm_command（cartesian）直接构造
        T_Ws_E0:     首帧 EE pose 在 W_s 系下（来自 state[0]）
    """
    return T_B_E_reset @ se3_inv(T_Ws_E0)


def ws_to_base(T_B_Ws: np.ndarray, T_Ws_Et: np.ndarray) -> np.ndarray:
    """T(B, E_t) = T(B, W_s) * T(W_s, E_t)"""
    return T_B_Ws @ T_Ws_Et


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------


def get_joint_ids(mj, model, joint_names: list[str]) -> list[int]:
    ids = []
    for jname in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not in MuJoCo model")
        ids.append(jid)
    return ids


def set_joint_state(mj, model, data, joint_ids: list[int], q_rad: np.ndarray) -> None:
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = q_rad[idx]
    mj.mj_forward(model, data)


def get_joint_positions(model, data, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qpos[int(model.jnt_qposadr[jid])] for jid in joint_ids], dtype=np.float64)


# ---------------------------------------------------------------------------
# Rerun 日志
# ---------------------------------------------------------------------------


def init_rerun(episode_idx: int) -> tuple[bool, str]:
    """初始化 Rerun，保存到 /tmp/fr3_replay_ep{idx:03d}.rrd（可跨容器查看）。

    Returns (ok, rrd_path)
    """
    try:
        import rerun as rr
        rrd_path = f"/tmp/fr3_replay_ep{episode_idx:03d}.rrd"
        rr.init(f"fr3_das_replay_ep{episode_idx:03d}", spawn=False)
        rr.save(rrd_path)
        print(f"[INFO] Rerun 记录保存到 {rrd_path}  (用 'rerun {rrd_path}' 查看)")
        return True, rrd_path
    except Exception as e:
        print(f"[WARN] Rerun 不可用: {e}")
        return False, ""


def log_frame(rr_ok: bool, fi: int, ts: float,
              sim_pos: np.ndarray, rec_pos: np.ndarray,
              pos_err_mm: float, rot_err_deg: float,
              gripper: float) -> None:
    if not rr_ok:
        return
    import rerun as rr
    rr.set_time("frame", sequence=fi)
    rr.set_time("timestamp", timestamp=ts)
    rr.log("ee/sim", rr.Points3D([sim_pos], colors=[[0, 220, 0]], radii=[0.005]))
    rr.log("ee/recorded", rr.Points3D([rec_pos], colors=[[220, 100, 0]], radii=[0.005]))
    rr.log("error/pos_mm", rr.Scalars([pos_err_mm]))
    rr.log("error/rot_deg", rr.Scalars([rot_err_deg]))
    rr.log("gripper", rr.Scalars([gripper]))


# ---------------------------------------------------------------------------
# 重播主循环
# ---------------------------------------------------------------------------


def replay(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer as mj_viewer

    # ── 加载数据 ──────────────────────────────────────────────────────
    print(f"[INFO] 加载 episode {args.episode}  dataset={args.dataset}")
    ep = load_episode(args.dataset, args.episode)
    states = ep["state"]       # [N, 8]  T(W_s, E_t) + gripper
    actions = ep["action"]     # [N, 8]  目标 T(W_s, E_t*) + gripper
    timestamps = ep["timestamp"]
    n_frames = len(states)
    print(f"[INFO] {n_frames} 帧 @ {args.fps} fps")

    # ── 初始化 MuJoCo ─────────────────────────────────────────────────
    print(f"[INFO] 加载 MuJoCo: {_DAS_XML}")
    model = mujoco.MjModel.from_xml_path(str(_DAS_XML))
    mj_data = mujoco.MjData(model)
    joint_ids = get_joint_ids(mujoco, model, _JOINT_NAMES)

    # ── 初始化 IK (placo) ──────────────────────────────────────────────
    print(f"[INFO] 初始化 Placo IK: {_DAS_URDF} → das_gripper_ee")
    from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver, RuckigOTGDriver

    kin = PlacoKinematicsDriver(
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        joint_names=_JOINT_NAMES,
    )

    # ── reset 位姿（B 系 SE(3)）──────────────────────────────────────
    T_B_E_reset = pose_from_xyzquat(_RESET_POSE_B_XYZQUAT)

    # ── 计算 T(B, W_s) ────────────────────────────────────────────────
    T_Ws_E0 = pose_from_xyzquat(states[0])
    T_B_Ws = build_T_B_Ws(T_B_E_reset, T_Ws_E0)
    print(f"[INFO] T(B, W_s)  pos={T_B_Ws[:3,3].round(4)}")

    # ── 初始化 OTG ────────────────────────────────────────────────────
    otg_dt = 1.0 / _OTG_CTRL_HZ
    otg_steps_per_frame = max(1, round(_OTG_CTRL_HZ / args.fps))
    otg = RuckigOTGDriver(
        dof=7, dt=otg_dt,
        max_velocity=_OTG_MAX_VEL, max_acceleration=_OTG_MAX_ACC, max_jerk=_OTG_MAX_JERK,
        min_position=_OTG_MIN_POS, max_position=_OTG_MAX_POS,
        synchronization=True, sync_mode="time",
    )

    # ── 从 home 关节角出发，初始化仿真状态 ──────────────────────────
    current_joints_rad = _IK_SEED_JOINTS_RAD.copy()
    set_joint_state(mujoco, model, mj_data, joint_ids, current_joints_rad)
    otg.reset(current_joints_rad)

    # ── MuJoCo Viewer ─────────────────────────────────────────────────
    viewer = None
    if not args.no_viewer:
        viewer = mj_viewer.launch_passive(model, mj_data)
        print("[INFO] MuJoCo Viewer 已启动")

    # ── 初始化 Rerun ──────────────────────────────────────────────────
    rr_ok, rrd_path = init_rerun(args.episode)

    # ── 误差记录 ──────────────────────────────────────────────────────
    pos_errors_mm: list[float] = []
    rot_errors_deg: list[float] = []

    frame_dt = 1.0 / args.fps
    print(f"\n[INFO] 开始重播 ({n_frames} 帧)…\n")

    try:
        for fi in range(n_frames):
            t0 = time.perf_counter()

            # 目标 EE pose：action[fi] → T(W_s, E*_t) → T(B, E*_t)
            T_Ws_Et_star = pose_from_xyzquat(actions[fi])
            T_B_Et_star = ws_to_base(T_B_Ws, T_Ws_Et_star)

            # IK  (PlacoKinematicsDriver: input rad → output rad)
            prev_joints_ik = target_joints_ik if fi > 0 else current_joints_rad  # noqa: F821
            target_joints_ik = np.asarray(kin.inverse_kinematics(current_joints_rad, T_B_Et_star), dtype=np.float64)
            target_joints_rad = np.clip(target_joints_ik[:7], _OTG_MIN_POS, _OTG_MAX_POS)
            if fi > 0:
                ik_delta = np.abs(target_joints_ik[:7] - prev_joints_ik[:7])
                if ik_delta.max() > 0.3:  # rad ≈ 17°，超过此阈值视为 elbow flip
                    print(f"  [IK_JUMP] frame {fi:4d}: max_joint_delta={ik_delta.max():.3f}rad "
                          f"joint_idx={int(np.argmax(ik_delta))}  pos_err={pos_errors_mm[-1] if pos_errors_mm else 0:.1f}mm")
                # OTG 追踪能力检查：每帧最大可追踪关节角 = max_vel * frame_dt
                otg_max_per_frame = np.array(_OTG_MAX_VEL) * (1.0 / args.fps)
                otg_gap = np.abs(target_joints_rad - current_joints_rad) - otg_max_per_frame
                if otg_gap.max() > 0.05:  # 超出 OTG 单帧追踪能力 0.05 rad
                    print(f"  [OTG_LAG] frame {fi:4d}: otg_gap={otg_gap.max():.3f}rad "
                          f"joint_idx={int(np.argmax(otg_gap))}")

            # OTG 平滑步进
            for _ in range(otg_steps_per_frame):
                current_joints_rad = np.asarray(
                    otg.step(current_joints_rad, target_joints_rad), dtype=np.float64
                )

            # 更新 MuJoCo
            set_joint_state(mujoco, model, mj_data, joint_ids, current_joints_rad)

            # 仿真 EE pose（FK）  (PlacoKinematicsDriver: input rad)
            T_B_Et_sim = np.asarray(kin.forward_kinematics(current_joints_rad), dtype=np.float64)
            sim_pos = T_B_Et_sim[:3, 3]
            sim_rot = T_B_Et_sim[:3, :3]

            # 录制参考 EE pose：state[fi] → T(W_s, E_t) → T(B, E_t)
            T_Ws_Et_rec = pose_from_xyzquat(states[fi])
            T_B_Et_rec = ws_to_base(T_B_Ws, T_Ws_Et_rec)
            rec_pos = T_B_Et_rec[:3, 3]
            rec_rot = T_B_Et_rec[:3, :3]

            # 误差（仿真 vs 录制）
            pos_err_mm = float(np.linalg.norm(sim_pos - rec_pos) * 1000)
            rot_err_deg = rotation_angle_error_deg(sim_rot, rec_rot)
            pos_errors_mm.append(pos_err_mm)
            rot_errors_deg.append(rot_err_deg)

            # Rerun
            log_frame(rr_ok, fi, float(timestamps[fi]),
                      sim_pos, rec_pos, pos_err_mm, rot_err_deg, float(actions[fi][7]))

            # Viewer
            if viewer is not None and viewer.is_running():
                viewer.sync()

            # 速率控制
            elapsed = time.perf_counter() - t0
            if (sleep_t := frame_dt - elapsed) > 0:
                time.sleep(sleep_t)

            if fi % 30 == 0:
                print(f"  [{fi:4d}/{n_frames}]  pos_err={pos_err_mm:6.2f}mm  rot_err={rot_err_deg:5.2f}°")

        # ── 统计汇总 ──────────────────────────────────────────────────
        pos_arr = np.array(pos_errors_mm)
        rot_arr = np.array(rot_errors_deg)
        # 稳定期统计（跳过前 30 帧 IK 收敛瞬态）
        _WARMUP = min(30, n_frames)
        pos_stable = pos_arr[_WARMUP:]
        rot_stable = rot_arr[_WARMUP:]
        print("\n" + "=" * 58)
        print(f"  Episode {args.episode} 重播完成   {n_frames} 帧 @ {args.fps} fps")
        print("=" * 58)
        print(f"  【全程】位置误差 (mm)   mean={pos_arr.mean():.2f}  "
              f"max={pos_arr.max():.2f}  p95={np.percentile(pos_arr,95):.2f}")
        print(f"  【全程】旋转误差 (°)    mean={rot_arr.mean():.2f}  "
              f"max={rot_arr.max():.2f}  p95={np.percentile(rot_arr,95):.2f}")
        if len(pos_stable) > 0:
            print(f"  【稳定期 frame {_WARMUP}+】位置误差 (mm)   mean={pos_stable.mean():.2f}  "
                  f"max={pos_stable.max():.2f}  p95={np.percentile(pos_stable,95):.2f}")
            print(f"  【稳定期 frame {_WARMUP}+】旋转误差 (°)    mean={rot_stable.mean():.2f}  "
                  f"max={rot_stable.max():.2f}  p95={np.percentile(rot_stable,95):.2f}")
        print("=" * 58)

        if viewer is not None and viewer.is_running():
            print("[INFO] 重播结束，Viewer 保持打开（按 Esc 关闭）")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.016)

    finally:
        if viewer is not None:
            viewer.close()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 DAS MuJoCo replay runtime (runs inside container)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True, help="数据集绝对路径（容器内）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    return replay(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
