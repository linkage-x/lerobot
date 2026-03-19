#!/usr/bin/env python3
"""
FR3 DAS 数据集真机重播运行时（容器内运行）

坐标变换链与仿真版 fr3_das_replay_runtime.py 完全相同。
区别：MuJoCo 后端替换为 FrankaResearch3 真机驱动；IK/OTG 由 FrankaResearch3 内部处理。

用法（通过 fr3_das_replay_real.py 启动，或直接运行）：
    PYTHONPATH=/lerobot/src python tools/fr3/fr3_das_replay_real_runtime.py \\
        --episode 0 --dataset /lerobot/outputs/datasets/lerobotv3_0310_100ep

数据格式（observation.state / action）：
    [x, y, z, qx, qy, qz, qw, gripper_aperture_m]
    坐标系：SLAM world frame W_s，各帧存储 T(W_s, I_t)
    其中 I = gripper_base_link（DAS 设备机身），E = das_gripper_ee（末端执行器）

坐标变换链：
    T(B, E_t) = T(B, W_s) * T(W_s, I_t) * T(I, E)

    其中：
        T(B, W_s) = T(B, E_reset) * T(E, I) * inv(T(W_s, I_0)_pos_only)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# 路径常量（容器内 /lerobot 挂载到 repo 根）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAS_URDF = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf"

_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

# 录制前机械臂 reset 位姿（B 系笛卡尔，xyzw 标量在后）
# R_reset_new = R_reset @ Ry(+15°)，补偿 IMU 安装倾斜引入的初始 pitch 误差
_RESET_POSE_B_XYZQUAT = np.array(
    [0.15327496, -0.54249998, 0.27, 0.60876143, 0.0, 0.79335334, 0.0],
    dtype=np.float64,
)

# DAS 录制起始关节角（rad），从真实 FR3 192.168.1.208 查询（2026-03-17）
# 重播前必须先将机械臂移动到此构型
_IK_SEED_JOINTS_RAD = np.array(
    # [-0.057892, -1.550292, -1.694795, -2.125873, 0.022874, 2.119849, -0.948928],
    [-0.053397256451184094, -1.5604194603713035, -1.720175311909912, -2.119629211414152, 0.011555741406479218, 2.1189401256121045, -0.9682376640047694],
    dtype=np.float64,
)

# DAS 设备机身→EE 固定外参 T(I, E)
# I = gripper_base_link，E = das_gripper_ee
# R(I,E) = [[0,0,1],[0,-1,0],[1,0,0]]，xyz=[0.13,0,-0.04]
_T_IE = np.array(
    [
        [0.0,  0.0,  1.0,  0.13],
        [0.0, -1.0,  0.0,  0.00],
        [1.0,  0.0,  0.0, -0.04],
        [0.0,  0.0,  0.0,  1.00],
    ],
    dtype=np.float64,
)

# EE base 坐标系 Z 安全下限（米）：低于此值跳过该帧命令，防止撞桌
_MIN_Z_M = 0.15
_DAS_RESET_POSITION = 1.0
_DAS_RESET_TARGET_TOLERANCE = 0.02
_DAS_FULLY_OPEN_SUCCESS_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# SE(3) 工具
# ---------------------------------------------------------------------------


def pose_from_xyzquat(xyzquat: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4x4 SE(3)（scipy quat 约定：scalar-last）"""
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
    """从 Parquet 数据集读取指定 episode 的 state/action/timestamp。"""
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
    if not mask:
        raise ValueError(
            f"Episode {episode_idx} found in metadata but no rows in {data_file}"
        )

    return {
        "state": np.array([t["observation.state"][i] for i in mask], dtype=np.float64),
        "action": np.array([t["action"][i] for i in mask], dtype=np.float64),
        "timestamp": np.array([t["timestamp"][i] for i in mask], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 坐标变换
# ---------------------------------------------------------------------------


def build_T_B_Ws(T_B_E_reset: np.ndarray, T_Ws_I0: np.ndarray) -> np.ndarray:
    """
    T(B, W_s) = T(B, E_reset) * T(E, I) * inv(T(W_s, I_0)_pos_only)

    T_Ws_I0 旋转部分强制为 I，仅保留平移（≈0），使 R(B,Ws)=I，
    消除 IMU 安装倾斜（≈15°Y）引入的 +Z 耦合。
    """
    T_Ws_I0_pos_only = np.eye(4, dtype=np.float64)
    T_Ws_I0_pos_only[:3, 3] = T_Ws_I0[:3, 3]
    return T_B_E_reset @ se3_inv(_T_IE) @ se3_inv(T_Ws_I0_pos_only)


def ws_to_base(T_B_Ws: np.ndarray, T_Ws_It: np.ndarray) -> np.ndarray:
    """T(B, E_t) = T(B, W_s) * T(W_s, I_t) * T(I, E)"""
    return T_B_Ws @ T_Ws_It @ _T_IE


# ---------------------------------------------------------------------------
# Rerun 日志
# ---------------------------------------------------------------------------


def init_rerun(episode_idx: int) -> tuple[bool, str]:
    try:
        import rerun as rr
        rrd_path = f"/tmp/fr3_real_replay_ep{episode_idx:03d}.rrd"
        rr.init(f"fr3_das_real_replay_ep{episode_idx:03d}", spawn=False)
        rr.save(rrd_path)
        print(f"[INFO] Rerun 记录保存到 {rrd_path}  (用 'rerun {rrd_path}' 查看)")
        return True, rrd_path
    except Exception as e:
        print(f"[WARN] Rerun 不可用: {e}")
        return False, ""


def log_frame(rr_ok: bool, fi: int, ts: float,
              hw_pos: np.ndarray, rec_pos: np.ndarray,
              pos_err_mm: float, rot_err_deg: float,
              gripper: float) -> None:
    if not rr_ok:
        return
    import rerun as rr
    rr.set_time("frame", sequence=fi)
    rr.set_time("timestamp", timestamp=ts)
    rr.log("ee/hardware", rr.Points3D([hw_pos], colors=[[0, 220, 0]], radii=[0.005]))
    rr.log("ee/recorded", rr.Points3D([rec_pos], colors=[[220, 100, 0]], radii=[0.005]))
    rr.log("error/pos_mm", rr.Scalars([pos_err_mm]))
    rr.log("error/rot_deg", rr.Scalars([rot_err_deg]))
    rr.log("gripper", rr.Scalars([gripper]))


# ---------------------------------------------------------------------------
# 重播前移动到 DAS 起始关节角
# ---------------------------------------------------------------------------


def move_to_das_start(robot_ip: str) -> None:
    """
    用 panda_py 将机械臂移至 DAS 录制起始关节角（阻塞直到到达）。

    必须在 FrankaResearch3.connect() 之前调用，避免 JointPosition 控制器冲突。
    目标关节角：_IK_SEED_JOINTS_RAD（从真实 FR3 192.168.1.208 查询，2026-03-17）
    """
    import panda_py

    print(f"[INFO] 连接 panda_py ({robot_ip})，移动到 DAS 起始关节角...")
    print(f"[INFO] 目标关节角（rad）: {_IK_SEED_JOINTS_RAD.tolist()}")
    panda = panda_py.Panda(robot_ip)
    panda.move_to_joint_position(_IK_SEED_JOINTS_RAD.tolist())
    del panda
    time.sleep(0.5)   # 等待 Franka 控制器释放控制权，避免 FrankaResearch3.connect() 时双重控制冲突
    print("[INFO] 已到达 DAS 起始关节角")


def reset_das_gripper(robot: "FrankaResearch3", target_position: float, timeout_s: float) -> None:
    """
    重播前显式将 DAS 夹爪 reset 到指定开口，并等待编码器反馈变化。

    这里直接走 gripper driver，避免通过 send_action 触发 arm IK/OTG。
    """
    clipped_target = float(np.clip(target_position, 0.0, 1.0))
    gripper = getattr(robot, "_gripper", None)
    if gripper is None:
        raise RuntimeError("FR3 gripper backend is not connected.")

    initial_position = float(gripper.get_position())
    print(
        f"[INFO] DAS 夹爪 reset: target={clipped_target:.4f} "
        f"(1.0=fully open), current={initial_position:.4f}"
    )
    gripper.set_position(clipped_target)

    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_position = initial_position
    while time.perf_counter() < deadline:
        last_position = float(gripper.get_position())
        if abs(last_position - clipped_target) <= _DAS_RESET_TARGET_TOLERANCE:
            print(f"[INFO] DAS 夹爪已 reset 到 {last_position:.4f}")
            return
        if clipped_target >= 1.0 - _DAS_RESET_TARGET_TOLERANCE and last_position >= _DAS_FULLY_OPEN_SUCCESS_THRESHOLD:
            print(
                f"[INFO] DAS 夹爪已足够打开: measured={last_position:.4f} "
                f"(threshold={_DAS_FULLY_OPEN_SUCCESS_THRESHOLD:.2f})"
            )
            return
        time.sleep(0.05)

    raise TimeoutError(
        f"DAS gripper did not reach reset target {clipped_target:.4f} within {timeout_s:.2f}s "
        f"(last={last_position:.4f})."
    )


def dataset_gripper_aperture_to_normalized(aperture_m: float, cfg: "FrankaResearch3Config") -> float:
    """
    DAS 数据集的第 8 维是夹爪开口距离（米），而不是 [0,1] 归一化值。

    replay 到硬件前，需要按当前 gripper backend 的行程映射回归一化命令。
    """
    aperture_m = float(max(0.0, aperture_m))
    if cfg.gripper_backend == "das":
        span_m = cfg.das_max_distance_m - cfg.das_min_distance_m
        if span_m <= 0:
            return 0.0
        return float(np.clip((aperture_m - cfg.das_min_distance_m) / span_m, 0.0, 1.0))

    max_width_m = cfg.gripper_max_width_mm / 1000.0
    if max_width_m <= 0:
        return 0.0
    return float(np.clip(aperture_m / max_width_m, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 真机重播主循环
# ---------------------------------------------------------------------------


def replay_real(args: argparse.Namespace) -> int:
    from lerobot.robots.franka_research3 import FrankaResearch3
    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config

    # ── 加载数据 ──────────────────────────────────────────────────────
    print(f"[INFO] 加载 episode {args.episode}  dataset={args.dataset}")
    ep = load_episode(args.dataset, args.episode)
    states = ep["state"]
    actions = ep["action"]
    timestamps = ep["timestamp"]
    n_frames = len(states)
    print(f"[INFO] {n_frames} 帧 @ {args.fps} fps")

    # ── 预检查：打印轨迹 Z 范围 ───────────────────────────────────────
    T_B_E_reset = pose_from_xyzquat(_RESET_POSE_B_XYZQUAT)
    T_B_Ws_preview = build_T_B_Ws(T_B_E_reset, pose_from_xyzquat(states[0]))
    target_poses = [ws_to_base(T_B_Ws_preview, pose_from_xyzquat(actions[fi])) for fi in range(n_frames)]
    z_vals = [float(T[2, 3]) for T in target_poses]
    z_min, z_max = min(z_vals), max(z_vals)
    n_below = sum(1 for z in z_vals if z < _MIN_Z_M)
    print(f"[INFO] 轨迹 Z 范围: min={z_min:.4f}m  max={z_max:.4f}m  低于 {_MIN_Z_M}m 的帧数={n_below}")
    if n_below > 0:
        print(f"[WARN] {n_below} 帧目标 Z < {_MIN_Z_M}m，这些帧将被跳过（不发给机器人）")

    # ── 移动到 DAS 起始关节角（先于 connect，避免控制器冲突）──────────
    move_to_das_start(args.robot_ip)

    # ── 初始化真机 ────────────────────────────────────────────────────
    cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        workspace_min=(0.1, -0.6, _MIN_Z_M),  # x min < reset_x(0.153)，避免首帧被 clip 引起 IK 偏移
        workspace_max=(0.9, 0.6, 0.8),
    )
    robot = FrankaResearch3(cfg)
    print(f"[INFO] 连接真机 {args.robot_ip} ...")
    robot.connect()
    print("[INFO] 真机已连接")

    try:
        if args.gripper_backend == "das":
            reset_das_gripper(
                robot,
                target_position=args.reset_gripper_position,
                timeout_s=args.reset_gripper_timeout_s,
            )

        # ── 读取真机实际 EE 完整位姿（position + orientation）──────────
        # 真机版不做 pos_only 简化：
        #   T_B_Ws = T_B_E_actual @ inv(T_IE) @ inv(T_Ws_I0)
        # → t=0 时 T_B_Ws @ states[0] @ T_IE = T_B_E_actual（精确，零首帧误差）
        # pos_only 会让 R_0（~15° IMU 倾斜）通过 T_IE 外参耦合出 ~3.5cm Z 平移残差，
        # 导致首帧命令比实际位置低 3.5cm（"下沉"）。
        obs_init = robot.get_observation()
        T_B_E_actual = np.eye(4, dtype=np.float64)
        T_B_E_actual[:3, 3] = [obs_init["ee.x"], obs_init["ee.y"], obs_init["ee.z"]]
        T_B_E_actual[:3, :3] = Rotation.from_rotvec(
            [obs_init["ee.wx"], obs_init["ee.wy"], obs_init["ee.wz"]]
        ).as_matrix()
        print(f"[INFO] 实际 EE 位置: xyz=[{obs_init['ee.x']:.4f}, {obs_init['ee.y']:.4f}, {obs_init['ee.z']:.4f}]")
        print(f"[INFO] 理论 reset xyz={_RESET_POSE_B_XYZQUAT[:3].round(4)}  "
              f"偏差 dz={obs_init['ee.z'] - _RESET_POSE_B_XYZQUAT[2]:.4f}m")

        # ── 计算 T(B, W_s)（不做 pos_only，使用完整 T_Ws_I0）─────────
        T_Ws_I0 = pose_from_xyzquat(states[0])
        T_B_Ws = T_B_E_actual @ se3_inv(_T_IE) @ se3_inv(T_Ws_I0)
        # 验证：t=0 命令应等于实际 EE 位置
        T_B_E0_check = ws_to_base(T_B_Ws, T_Ws_I0)
        print(f"[INFO] T(B, W_s)  pos={T_B_Ws[:3,3].round(4)}")
        print(f"[INFO] t=0 命令预测 xyz={T_B_E0_check[:3,3].round(4)}  "
              f"（应 ≈ 实际 EE 位置，差值={np.linalg.norm(T_B_E0_check[:3,3] - T_B_E_actual[:3,3])*1000:.2f}mm）")

        # ── 初始化 Rerun ──────────────────────────────────────────────
        rr_ok, _ = init_rerun(args.episode)

        # ── 误差记录 ──────────────────────────────────────────────────
        pos_errors_mm: list[float] = []
        rot_errors_deg: list[float] = []
        skipped_frames: list[int] = []

        assert actions.shape[1] >= 8, (
            f"action 列数 {actions.shape[1]} < 8，期望 [x,y,z,qx,qy,qz,qw,gripper]"
        )
        action_gripper_normalized = np.array(
            [dataset_gripper_aperture_to_normalized(frame[7], cfg) for frame in actions],
            dtype=np.float64,
        )
        print(
            "[INFO] 数据集夹爪语义: aperture_m -> normalized  "
            f"frame0={actions[0][7]:.4f}m->{action_gripper_normalized[0]:.4f}"
        )

        frame_dt = 1.0 / args.fps
        print(f"\n[INFO] 开始真机重播 ({n_frames} 帧)…\n")

        for fi in range(n_frames):
            t0 = time.perf_counter()

            # 目标 EE pose
            T_Ws_Et_star = pose_from_xyzquat(actions[fi])
            T_B_Et_star = ws_to_base(T_B_Ws, T_Ws_Et_star)

            # 前3帧打印命令位置，确认无首帧下沉
            if fi < 3:
                print(f"  [DEBUG] frame {fi}: 命令 xyz={T_B_Et_star[:3,3].round(4)}")

            # 安全检查：Z < 15cm 时跳过，不发给机器人
            target_z = float(T_B_Et_star[2, 3])
            if target_z < _MIN_Z_M:
                print(f"  [WARN] frame {fi:4d}: target Z={target_z:.4f}m < {_MIN_Z_M}m，跳过")
                skipped_frames.append(fi)
                elapsed = time.perf_counter() - t0
                if (sleep_t := frame_dt - elapsed) > 0:
                    time.sleep(sleep_t)
                continue

            # 发送给真机（绝对 EE pose，旋转向量格式）
            rotvec = Rotation.from_matrix(T_B_Et_star[:3, :3]).as_rotvec()
            robot.send_action({
                "ee.x":  float(T_B_Et_star[0, 3]),
                "ee.y":  float(T_B_Et_star[1, 3]),
                "ee.z":  float(T_B_Et_star[2, 3]),
                "ee.wx": float(rotvec[0]),
                "ee.wy": float(rotvec[1]),
                "ee.wz": float(rotvec[2]),
                "gripper.pos": float(action_gripper_normalized[fi]),
            })

            # 读取当前 EE pose（误差统计用）
            obs = robot.get_observation()
            hw_pos = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=np.float64)
            hw_rot = Rotation.from_rotvec([obs["ee.wx"], obs["ee.wy"], obs["ee.wz"]]).as_matrix()

            # 录制参考 pose
            T_B_Et_rec = ws_to_base(T_B_Ws, pose_from_xyzquat(states[fi]))
            rec_pos = T_B_Et_rec[:3, 3]
            rec_rot = T_B_Et_rec[:3, :3]

            pos_err_mm = float(np.linalg.norm(hw_pos - rec_pos) * 1000)
            rot_err_deg = rotation_angle_error_deg(hw_rot, rec_rot)
            pos_errors_mm.append(pos_err_mm)
            rot_errors_deg.append(rot_err_deg)

            log_frame(rr_ok, fi, float(timestamps[fi]),
                      hw_pos, rec_pos, pos_err_mm, rot_err_deg, float(action_gripper_normalized[fi]))

            # 速率控制
            elapsed = time.perf_counter() - t0
            if (sleep_t := frame_dt - elapsed) > 0:
                time.sleep(sleep_t)

            if fi % 30 == 0:
                print(f"  [{fi:4d}/{n_frames}]  pos_err={pos_err_mm:6.2f}mm  rot_err={rot_err_deg:5.2f}°")

        # ── 统计汇总 ──────────────────────────────────────────────────
        if skipped_frames:
            print(f"\n[WARN] 共跳过 {len(skipped_frames)} 帧（Z < {_MIN_Z_M}m）: {skipped_frames[:10]}"
                  f"{'...' if len(skipped_frames) > 10 else ''}")

        pos_arr = np.array(pos_errors_mm)
        if len(pos_arr) == 0:
            print("[WARN] 无有效帧（全部被跳过），无统计数据")
            return 0
        rot_arr = np.array(rot_errors_deg)
        _WARMUP = min(30, len(pos_arr))
        pos_stable = pos_arr[_WARMUP:]
        rot_stable = rot_arr[_WARMUP:]

        print("\n" + "=" * 60)
        print(f"  Episode {args.episode} 真机重播完成   {n_frames} 帧 @ {args.fps} fps")
        print("=" * 60)
        if len(pos_arr) > 0:
            print(f"  【全程】位置误差 (mm)   mean={pos_arr.mean():.2f}  "
                  f"max={pos_arr.max():.2f}  p95={np.percentile(pos_arr,95):.2f}")
            print(f"  【全程】旋转误差 (°)    mean={rot_arr.mean():.2f}  "
                  f"max={rot_arr.max():.2f}  p95={np.percentile(rot_arr,95):.2f}")
        if len(pos_stable) > 0:
            print(f"  【稳定期 frame {_WARMUP}+】位置误差 (mm)   mean={pos_stable.mean():.2f}  "
                  f"max={pos_stable.max():.2f}  p95={np.percentile(pos_stable,95):.2f}")
            print(f"  【稳定期 frame {_WARMUP}+】旋转误差 (°)    mean={rot_stable.mean():.2f}  "
                  f"max={rot_stable.max():.2f}  p95={np.percentile(rot_stable,95):.2f}")
        print("=" * 60)

    finally:
        robot.disconnect()
        print("[INFO] 真机已断开")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FR3 DAS 真机重播运行时（容器内运行，通过 fr3_das_replay_real.py 启动）"
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True, help="数据集绝对路径（容器内）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-ip", default="192.168.1.208")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-backend", choices=["pika", "das"], default="das")
    parser.add_argument("--reset-gripper-position", type=float, default=_DAS_RESET_POSITION)
    parser.add_argument("--reset-gripper-timeout-s", type=float, default=2.0)
    return parser.parse_args(argv)


def main() -> int:
    return replay_real(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
