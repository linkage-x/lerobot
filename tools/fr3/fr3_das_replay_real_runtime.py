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
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# 路径常量（容器内 /lerobot 挂载到 repo 根）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAS_URDF = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf"

_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
_POSE_WITH_GRIPPER_NORMALIZED_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_normalized"]
_POSE_WITH_GRIPPER_APERTURE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_aperture_m"]
_JOINT_POSITION_RAD_NAMES = [f"joint_{joint_idx}_rad" for joint_idx in range(1, 8)]
_REPLAY_STATUS_NAMES = [
    "command_sent",
    "skipped_by_z",
    "blend_alpha",
    "finger_lowest_est_z_m",
    "cmd_ee_z_m",
    "target_ee_z_m",
    "target_dt_s",
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
_DEFAULT_MIN_TOOL_Z_M = 0.18
_DAS_RESET_POSITION = 1.0
_DAS_RESET_TARGET_TOLERANCE = 0.02
_DAS_FULLY_OPEN_SUCCESS_THRESHOLD = 0.90
_PEAK_DIAGNOSTIC_FRAMES = 5
_DEFAULT_OTG_SCALE = 1.0
_DEFAULT_ANALYSIS_OUTPUT_DIR = _REPO_ROOT / "outputs" / "analysis"
_POSE_AXIS_LENGTH_M = 0.015
_POSE_RESTORE_POS_THRESHOLD_MM = 5.0
_POSE_RESTORE_ROT_THRESHOLD_DEG = 1.0
_LEGACY_FIRST_FRAME_TILT_DEG_THRESHOLD = 8.0
_LEGACY_FIRST_FRAME_AXIS_Y_ALIGNMENT_THRESHOLD = 0.9
_LEGACY_START_BLEND_FRAMES = 12
_DEFAULT_LEGACY_Z_OFFSET_M = 0.01
# Conservative swept finger envelope in EE frame, derived from the DAS URDF and
# finger meshes (`das_link5.STL` / `das_link6.STL`) over joint range [0, 0.925].
_FINGER_SWEEP_BBOX_E_MIN = np.array([-0.0350, -0.0707, -0.0976], dtype=np.float64)
_FINGER_SWEEP_BBOX_E_MAX = np.array([0.0018, 0.0699, -0.0227], dtype=np.float64)
_STALL_HW_STEP_MM_THRESHOLD = 0.3
_STALL_POS_ERR_MM_THRESHOLD = 30.0
_STALL_Q_CMD_ERR_DEG_THRESHOLD = 20.0
_STALL_CONSECUTIVE_FRAMES = 3


# ---------------------------------------------------------------------------
# SE(3) 工具
# ---------------------------------------------------------------------------


def _rotation_class():
    from scipy.spatial.transform import Rotation

    return Rotation


def pose_from_xyzquat(xyzquat: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4x4 SE(3)（scipy quat 约定：scalar-last）"""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyzquat[:3]
    T[:3, :3] = _rotation_class().from_quat(xyzquat[3:7]).as_matrix()
    return T


def pose_to_xyzquat(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) → [x, y, z, qx, qy, qz, qw]"""
    xyzquat = np.empty(7, dtype=np.float64)
    xyzquat[:3] = np.asarray(T, dtype=np.float64)[:3, 3]
    xyzquat[3:7] = _rotation_class().from_matrix(np.asarray(T, dtype=np.float64)[:3, :3]).as_quat()
    return xyzquat


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


def quaternion_angle_error_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    """两个四元数之间的最小夹角（°）"""
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def interpolate_pose(T_start: np.ndarray, T_end: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) pose interpolation with linear translation and shortest-path rotation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.asarray(T_start, dtype=np.float64).copy()
    if alpha >= 1.0:
        return np.asarray(T_end, dtype=np.float64).copy()

    T_interp = np.eye(4, dtype=np.float64)
    T_interp[:3, 3] = (1.0 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
    R_start = T_start[:3, :3]
    R_delta = R_start.T @ T_end[:3, :3]
    scaled_delta = _rotation_class().from_matrix(R_delta).as_rotvec() * alpha
    T_interp[:3, :3] = R_start @ _rotation_class().from_rotvec(scaled_delta).as_matrix()
    return T_interp


def bbox_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """Return the 8 corners of an axis-aligned bounding box."""
    return np.asarray(
        [[x, y, z] for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])],
        dtype=np.float64,
    )


def estimate_finger_lowest_z(T_B_E: np.ndarray) -> float:
    """Estimate the world-frame lowest finger point using a conservative EE-frame envelope."""
    safety_points_e = bbox_corners(_FINGER_SWEEP_BBOX_E_MIN, _FINGER_SWEEP_BBOX_E_MAX)
    points_b = (T_B_E[:3, :3] @ safety_points_e.T).T + T_B_E[:3, 3]
    return float(np.min(points_b[:, 2]))


def apply_pose_z_offset(T_B_E: np.ndarray, z_offset_m: float) -> np.ndarray:
    """Apply a constant base-frame Z correction to a pose."""
    if abs(float(z_offset_m)) < 1e-12:
        return np.asarray(T_B_E, dtype=np.float64).copy()
    T_offset = np.asarray(T_B_E, dtype=np.float64).copy()
    T_offset[2, 3] += float(z_offset_m)
    return T_offset


def _bool_like_is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False


def detect_robot_abort_reason(robot: object) -> str | None:
    """Best-effort detection of real-hardware control aborts/reflexes."""
    raise_if_failed = getattr(robot, "_raise_if_otg_failed", None)
    if callable(raise_if_failed):
        try:
            raise_if_failed()
        except Exception as exc:  # pragma: no cover - exercised on hardware
            return f"OTG failed: {exc}"

    arm_backend = getattr(robot, "_arm", None)
    panda_robot = getattr(arm_backend, "_robot", None)
    get_state = getattr(panda_robot, "get_state", None)
    if not callable(get_state):
        return None

    try:
        state = get_state()
    except Exception as exc:  # pragma: no cover - exercised on hardware
        return f"arm state unavailable: {exc}"

    success_rate = getattr(state, "control_command_success_rate", None)
    if success_rate is not None:
        try:
            success_rate = float(success_rate)
        except (TypeError, ValueError):
            success_rate = None
        if success_rate is not None and success_rate < 1.0:
            return f"control_command_success_rate={success_rate:.3f}"

    for attr_name in dir(state):
        attr_lower = attr_name.lower()
        if "error" not in attr_lower and "reflex" not in attr_lower:
            continue
        try:
            attr_value = getattr(state, attr_name)
        except Exception:  # pragma: no cover - defensive
            continue
        if _bool_like_is_true(attr_value):
            return f"robot state flagged {attr_name}=True"
        if hasattr(attr_value, "__dict__"):
            for nested_name, nested_value in vars(attr_value).items():
                if _bool_like_is_true(nested_value):
                    return f"robot state flagged {attr_name}.{nested_name}=True"
    return None


def describe_first_frame_tilt(T_Ws_I0: np.ndarray) -> dict[str, np.ndarray | float | bool]:
    """Classify whether the dataset likely uses the legacy tilted first-frame contract."""
    rotvec = _rotation_class().from_matrix(T_Ws_I0[:3, :3]).as_rotvec()
    angle_rad = float(np.linalg.norm(rotvec))
    angle_deg = float(np.degrees(angle_rad))
    if angle_rad < 1e-9:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = rotvec / angle_rad
    legacy_tilt = (
        angle_deg >= _LEGACY_FIRST_FRAME_TILT_DEG_THRESHOLD
        and abs(float(axis[1])) >= _LEGACY_FIRST_FRAME_AXIS_Y_ALIGNMENT_THRESHOLD
    )
    return {
        "angle_deg": angle_deg,
        "axis": axis,
        "legacy_tilt": legacy_tilt,
    }


def parse_joint_gains(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != len(_JOINT_NAMES):
        raise argparse.ArgumentTypeError(
            f"Expected {len(_JOINT_NAMES)} comma-separated floats for FR3 joint gains."
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected {len(_JOINT_NAMES)} comma-separated floats for FR3 joint gains."
        ) from exc


def parse_joint_limit_values(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != len(_JOINT_NAMES):
        raise argparse.ArgumentTypeError(
            f"Expected {len(_JOINT_NAMES)} comma-separated floats for FR3 OTG limits."
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected {len(_JOINT_NAMES)} comma-separated floats for FR3 OTG limits."
        ) from exc


def positive_scale(value: str) -> float:
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive float.") from exc
    if scale <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return scale


def summarize_metric(name: str, values: np.ndarray, unit: str, prefix: str = "") -> None:
    if len(values) == 0:
        return
    print(
        f"  {prefix}{name} ({unit})   mean={values.mean():.2f}  "
        f"max={values.max():.2f}  p95={np.percentile(values, 95):.2f}"
    )


def resolve_joint_limit_values(
    default_values: tuple[float, ...],
    override_values: list[float] | None,
    scale: float,
) -> tuple[float, ...]:
    base = np.asarray(default_values if override_values is None else override_values, dtype=np.float64)
    return tuple((base * float(scale)).tolist())


def snapshot_otg_debug(robot: "FrankaResearch3") -> tuple[np.ndarray | None, np.ndarray | None]:
    target_joints = None
    command_joints = None
    target_lock = getattr(robot, "_otg_target_lock", None)
    command_lock = getattr(robot, "_otg_command_lock", None)

    if target_lock is not None:
        with target_lock:
            raw_target = getattr(robot, "_otg_target_joints", None)
            if raw_target is not None:
                target_joints = np.asarray(raw_target, dtype=np.float64).copy()
    else:
        raw_target = getattr(robot, "_otg_target_joints", None)
        if raw_target is not None:
            target_joints = np.asarray(raw_target, dtype=np.float64).copy()

    if command_lock is not None:
        with command_lock:
            raw_command = getattr(robot, "_otg_command_joints", None)
            if raw_command is not None:
                command_joints = np.asarray(raw_command, dtype=np.float64).copy()
    else:
        raw_command = getattr(robot, "_otg_command_joints", None)
        if raw_command is not None:
            command_joints = np.asarray(raw_command, dtype=np.float64).copy()

    return target_joints, command_joints


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_episode(dataset_path: str, episode_idx: int) -> dict[str, np.ndarray]:
    """通过 LeRobotDataset 公共接口读取指定 episode 的 state/action/timestamp。"""
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not (dataset_root / "meta").exists():
        raise FileNotFoundError(f"No dataset metadata directory in {dataset_root}")

    dataset = LeRobotDataset(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
        episodes=[episode_idx],
        download_videos=False,
    )
    columns = dataset.get_episode_column_arrays(episode_idx, ["observation.state", "action", "timestamp"])
    return {
        "state": columns["observation.state"].astype(np.float64, copy=False),
        "action": columns["action"].astype(np.float64, copy=False),
        "timestamp": columns["timestamp"].reshape(-1).astype(np.float64, copy=False),
    }


def load_joint_target_sequence(
    csv_path: str,
    *,
    n_frames: int,
    column_prefix: str,
) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Joint target CSV not found: {path}")

    targets_deg: list[np.ndarray] = []
    expected_columns = [f"{column_prefix}_{joint_idx}_deg" for joint_idx in range(1, 8)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in expected_columns if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing joint target columns in {path}: {missing}")
        for expected_frame, row in enumerate(reader):
            frame_text = row.get("frame")
            if frame_text is not None and int(frame_text) != expected_frame:
                raise ValueError(
                    f"Unexpected frame index in {path}: expected {expected_frame}, got {frame_text}"
                )
            targets_deg.append(
                np.array([float(row[column_name]) for column_name in expected_columns], dtype=np.float64)
            )

    targets_deg_arr = np.asarray(targets_deg, dtype=np.float64)
    if targets_deg_arr.shape != (n_frames, 7):
        raise ValueError(
            f"Expected joint target CSV shape {(n_frames, 7)}, got {targets_deg_arr.shape} from {path}"
        )
    return np.deg2rad(targets_deg_arr)


def create_replay_record_dataset(
    output_root: Path,
    *,
    fps: int,
    source_dataset_path: str,
    episode_idx: int,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_NORMALIZED_NAMES)},
            "fps": float(fps),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_NORMALIZED_NAMES)},
            "fps": float(fps),
        },
        "observation.reference_state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_APERTURE_NAMES)},
            "fps": float(fps),
        },
        "observation.reference_action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_APERTURE_NAMES)},
            "fps": float(fps),
        },
        "observation.source_state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_APERTURE_NAMES)},
            "fps": float(fps),
        },
        "observation.source_action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": list(_POSE_WITH_GRIPPER_APERTURE_NAMES)},
            "fps": float(fps),
        },
        "observation.joint_measured": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": list(_JOINT_POSITION_RAD_NAMES)},
            "fps": float(fps),
        },
        "observation.joint_commanded": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": list(_JOINT_POSITION_RAD_NAMES)},
            "fps": float(fps),
        },
        "observation.joint_target": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": list(_JOINT_POSITION_RAD_NAMES)},
            "fps": float(fps),
        },
        "observation.replay_status": {
            "dtype": "float32",
            "shape": (len(_REPLAY_STATUS_NAMES),),
            "names": {"flags": list(_REPLAY_STATUS_NAMES)},
            "fps": float(fps),
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=f"local/{output_root.name}",
        fps=fps,
        root=output_root,
        robot_type="franka_research3_real_replay",
        features=features,
        use_videos=False,
        image_writer_threads=0,
    )
    metadata_path = output_root / "meta" / "replay_source.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_dataset": str(source_dataset_path),
                "source_episode": int(episode_idx),
                "record_layout": {
                    "observation.state": "measured hardware EE pose in base frame + gripper_normalized",
                    "action": "runtime command candidate in base frame + gripper_normalized",
                    "observation.reference_state": "source state mapped into base frame + gripper_aperture_m",
                    "observation.reference_action": "source action mapped into base frame before send + gripper_aperture_m",
                    "observation.source_state": "raw source dataset state in dataset frame + gripper_aperture_m",
                    "observation.source_action": "raw source dataset action in dataset frame + gripper_aperture_m",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return dataset


def make_pose_gripper_row(
    pose_xyzquat: np.ndarray,
    gripper_value: float,
) -> np.ndarray:
    row = np.empty(8, dtype=np.float32)
    row[:7] = np.asarray(pose_xyzquat, dtype=np.float32)
    row[7] = np.float32(gripper_value)
    return row


def build_replay_record_frame(
    *,
    measured_pose_xyzquat: np.ndarray,
    measured_gripper_normalized: float,
    command_pose_xyzquat: np.ndarray,
    command_gripper_normalized: float,
    reference_state_pose_xyzquat: np.ndarray,
    reference_state_gripper_aperture_m: float,
    reference_action_pose_xyzquat: np.ndarray,
    reference_action_gripper_aperture_m: float,
    source_state_row: np.ndarray,
    source_action_row: np.ndarray,
    measured_joints: np.ndarray,
    command_joints: np.ndarray | None,
    target_joints: np.ndarray | None,
    replay_status: np.ndarray,
    task: str,
) -> dict[str, np.ndarray | str]:
    nan_joints = np.full(7, np.nan, dtype=np.float32)
    return {
        "observation.state": make_pose_gripper_row(measured_pose_xyzquat, measured_gripper_normalized),
        "action": make_pose_gripper_row(command_pose_xyzquat, command_gripper_normalized),
        "observation.reference_state": make_pose_gripper_row(
            reference_state_pose_xyzquat,
            reference_state_gripper_aperture_m,
        ),
        "observation.reference_action": make_pose_gripper_row(
            reference_action_pose_xyzquat,
            reference_action_gripper_aperture_m,
        ),
        "observation.source_state": np.asarray(source_state_row, dtype=np.float32).copy(),
        "observation.source_action": np.asarray(source_action_row, dtype=np.float32).copy(),
        "observation.joint_measured": np.asarray(measured_joints, dtype=np.float32).copy(),
        "observation.joint_commanded": (
            np.asarray(command_joints, dtype=np.float32).copy() if command_joints is not None else nan_joints.copy()
        ),
        "observation.joint_target": (
            np.asarray(target_joints, dtype=np.float32).copy() if target_joints is not None else nan_joints.copy()
        ),
        "observation.replay_status": np.asarray(replay_status, dtype=np.float32).copy(),
        "task": task,
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


def build_T_B_Ws_actual_pos_only(T_B_E_actual: np.ndarray, T_Ws_I0: np.ndarray) -> np.ndarray:
    """Use live hardware EE pose for translation alignment while discarding legacy first-frame tilt."""
    T_Ws_I0_pos_only = np.eye(4, dtype=np.float64)
    T_Ws_I0_pos_only[:3, 3] = T_Ws_I0[:3, 3]
    return T_B_E_actual @ se3_inv(_T_IE) @ se3_inv(T_Ws_I0_pos_only)


def ws_to_base(T_B_Ws: np.ndarray, T_Ws_It: np.ndarray) -> np.ndarray:
    """T(B, E_t) = T(B, W_s) * T(W_s, I_t) * T(I, E)"""
    return T_B_Ws @ T_Ws_It @ _T_IE


# ---------------------------------------------------------------------------
# Rerun 日志
# ---------------------------------------------------------------------------


def init_rerun(episode_idx: int, *, output_dir: Path) -> tuple[bool, Path | None]:
    try:
        import rerun as rr

        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / f"fr3_real_replay_ep{episode_idx:03d}.rrd"
        rr.init(f"fr3_das_real_replay_ep{episode_idx:03d}", spawn=False)
        rr.save(str(rrd_path))
        print(f"[INFO] Rerun 记录保存到 {rrd_path}  (宿主机可直接运行: rerun {rrd_path})")
        return True, rrd_path
    except Exception as e:
        print(f"[WARN] Rerun 不可用: {e}")
        return False, None


def log_frame(
    rr_ok: bool,
    fi: int,
    ts: float,
    hw_pos: np.ndarray,
    state_pos: np.ndarray,
    action_pos: np.ndarray,
    pos_err_state_mm: float,
    rot_err_state_deg: float,
    pos_err_action_mm: float,
    rot_err_action_deg: float,
    gripper: float,
    hw_traj: np.ndarray,
    state_traj: np.ndarray,
    action_traj: np.ndarray,
) -> None:
    if not rr_ok:
        return
    import rerun as rr

    rr.set_time("frame", sequence=fi)
    rr.set_time("timestamp", timestamp=ts)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ee/hardware/trajectory",
        rr.LineStrips3D([np.asarray(hw_traj, dtype=np.float32)], colors=[[0, 220, 0, 255]], radii=[0.002]),
    )
    rr.log(
        "world/ee/reference_state/trajectory",
        rr.LineStrips3D([np.asarray(state_traj, dtype=np.float32)], colors=[[220, 100, 0, 255]], radii=[0.002]),
    )
    rr.log(
        "world/ee/reference_action/trajectory",
        rr.LineStrips3D([np.asarray(action_traj, dtype=np.float32)], colors=[[40, 120, 255, 255]], radii=[0.002]),
    )
    rr.log("world/ee/hardware/current", rr.Points3D([hw_pos], colors=[[0, 220, 0]], radii=[0.005]))
    rr.log("world/ee/reference_state/current", rr.Points3D([state_pos], colors=[[220, 100, 0]], radii=[0.005]))
    rr.log("world/ee/reference_action/current", rr.Points3D([action_pos], colors=[[40, 120, 255]], radii=[0.005]))
    rr.log(
        "world/ee/error/action_to_hardware",
        rr.LineStrips3D([np.asarray([action_pos, hw_pos], dtype=np.float32)], colors=[[255, 64, 64, 255]], radii=[0.001]),
    )
    rr.log(
        "world/ee/error/state_to_hardware",
        rr.LineStrips3D([np.asarray([state_pos, hw_pos], dtype=np.float32)], colors=[[255, 191, 0, 255]], radii=[0.001]),
    )
    rr.log("error/state_pos_mm", rr.Scalars([pos_err_state_mm]))
    rr.log("error/state_rot_deg", rr.Scalars([rot_err_state_deg]))
    rr.log("error/action_pos_mm", rr.Scalars([pos_err_action_mm]))
    rr.log("error/action_rot_deg", rr.Scalars([rot_err_action_deg]))
    rr.log("gripper", rr.Scalars([gripper]))


def write_trajectory_csv(
    output_path: Path,
    *,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    hw_positions: np.ndarray,
    state_positions: np.ndarray,
    action_positions: np.ndarray,
    pos_errors_state_mm: np.ndarray,
    rot_errors_state_deg: np.ndarray,
    pos_errors_action_mm: np.ndarray,
    rot_errors_action_deg: np.ndarray,
    state_action_gap_mm: np.ndarray,
    state_action_gap_deg: np.ndarray,
    measured_joints: np.ndarray,
    command_joints: np.ndarray | None,
    target_joints: np.ndarray | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "frame",
        "timestamp_s",
        "hw_x_m",
        "hw_y_m",
        "hw_z_m",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "action_x_m",
        "action_y_m",
        "action_z_m",
        "pos_err_state_mm",
        "rot_err_state_deg",
        "pos_err_action_mm",
        "rot_err_action_deg",
        "state_action_gap_mm",
        "state_action_gap_deg",
    ]
    for joint_idx in range(1, 8):
        header.append(f"q_meas_{joint_idx}_rad")
    if command_joints is not None:
        for joint_idx in range(1, 8):
            header.append(f"q_cmd_{joint_idx}_rad")
    if target_joints is not None:
        for joint_idx in range(1, 8):
            header.append(f"q_target_{joint_idx}_rad")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row_idx, frame_idx in enumerate(frame_indices.tolist()):
            row = {
                "frame": int(frame_idx),
                "timestamp_s": f"{float(timestamps[row_idx]):.6f}",
                "hw_x_m": f"{float(hw_positions[row_idx, 0]):.6f}",
                "hw_y_m": f"{float(hw_positions[row_idx, 1]):.6f}",
                "hw_z_m": f"{float(hw_positions[row_idx, 2]):.6f}",
                "state_x_m": f"{float(state_positions[row_idx, 0]):.6f}",
                "state_y_m": f"{float(state_positions[row_idx, 1]):.6f}",
                "state_z_m": f"{float(state_positions[row_idx, 2]):.6f}",
                "action_x_m": f"{float(action_positions[row_idx, 0]):.6f}",
                "action_y_m": f"{float(action_positions[row_idx, 1]):.6f}",
                "action_z_m": f"{float(action_positions[row_idx, 2]):.6f}",
                "pos_err_state_mm": f"{float(pos_errors_state_mm[row_idx]):.6f}",
                "rot_err_state_deg": f"{float(rot_errors_state_deg[row_idx]):.6f}",
                "pos_err_action_mm": f"{float(pos_errors_action_mm[row_idx]):.6f}",
                "rot_err_action_deg": f"{float(rot_errors_action_deg[row_idx]):.6f}",
                "state_action_gap_mm": f"{float(state_action_gap_mm[row_idx]):.6f}",
                "state_action_gap_deg": f"{float(state_action_gap_deg[row_idx]):.6f}",
            }
            for joint_idx in range(7):
                row[f"q_meas_{joint_idx + 1}_rad"] = f"{float(measured_joints[row_idx, joint_idx]):.9f}"
            if command_joints is not None:
                for joint_idx in range(7):
                    row[f"q_cmd_{joint_idx + 1}_rad"] = f"{float(command_joints[row_idx, joint_idx]):.9f}"
            if target_joints is not None:
                for joint_idx in range(7):
                    row[f"q_target_{joint_idx + 1}_rad"] = f"{float(target_joints[row_idx, joint_idx]):.9f}"
            writer.writerow(row)


def build_hover_text(
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    pos_errors_action_mm: np.ndarray,
    pos_errors_state_mm: np.ndarray,
    state_action_gap_mm: np.ndarray,
) -> list[str]:
    hover_text: list[str] = []
    for idx, frame_idx in enumerate(frame_indices.tolist()):
        hover_text.append(
            "<br>".join(
                [
                    f"frame={int(frame_idx)}",
                    f"t={float(timestamps[idx]):.3f}s",
                    f"hw-action={float(pos_errors_action_mm[idx]):.2f} mm",
                    f"hw-state={float(pos_errors_state_mm[idx]):.2f} mm",
                    f"state-action={float(state_action_gap_mm[idx]):.2f} mm",
                ]
            )
        )
    return hover_text


def build_pose_axis_trace(
    *,
    positions: np.ndarray,
    rotations: np.ndarray,
    axis_index: int,
    axis_length_m: float,
    name: str,
    color: str,
    hover_text: list[str],
) -> dict[str, object]:
    x_vals: list[float | None] = []
    y_vals: list[float | None] = []
    z_vals: list[float | None] = []
    text: list[str | None] = []
    axis_vec = np.zeros(3, dtype=np.float64)
    axis_vec[axis_index] = float(axis_length_m)

    for frame_idx in range(len(positions)):
        start = positions[frame_idx]
        end = start + rotations[frame_idx] @ axis_vec
        x_vals.extend([float(start[0]), float(end[0]), None])
        y_vals.extend([float(start[1]), float(end[1]), None])
        z_vals.extend([float(start[2]), float(end[2]), None])
        text.extend([hover_text[frame_idx], hover_text[frame_idx], None])

    return {
        "type": "scatter3d",
        "mode": "lines",
        "name": name,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "text": text,
        "hovertemplate": "%{text}<extra>" + name + "</extra>",
        "line": {"color": color, "width": 2},
        "showlegend": True,
    }


def write_trajectory_plot_html(
    output_path: Path,
    *,
    episode_idx: int,
    dataset_path: str,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    hw_positions: np.ndarray,
    hw_rotations: np.ndarray,
    state_positions: np.ndarray,
    action_positions: np.ndarray,
    action_rotations: np.ndarray,
    pos_errors_state_mm: np.ndarray,
    rot_errors_state_deg: np.ndarray,
    pos_errors_action_mm: np.ndarray,
    rot_errors_action_deg: np.ndarray,
    state_action_gap_mm: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hover_text = build_hover_text(
        frame_indices,
        timestamps,
        pos_errors_action_mm,
        pos_errors_state_mm,
        state_action_gap_mm,
    )
    peak_count = min(_PEAK_DIAGNOSTIC_FRAMES, len(frame_indices))
    peak_order = np.argsort(pos_errors_action_mm)[-peak_count:][::-1] if peak_count > 0 else np.array([], dtype=np.int64)

    def positions_to_axes(positions: np.ndarray) -> tuple[list[float], list[float], list[float]]:
        return (
            positions[:, 0].astype(float).tolist(),
            positions[:, 1].astype(float).tolist(),
            positions[:, 2].astype(float).tolist(),
        )

    hw_x, hw_y, hw_z = positions_to_axes(hw_positions)
    state_x, state_y, state_z = positions_to_axes(state_positions)
    action_x, action_y, action_z = positions_to_axes(action_positions)

    traces = [
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "hardware",
            "x": hw_x,
            "y": hw_y,
            "z": hw_z,
            "line": {"color": "rgb(0,220,0)", "width": 6},
            "marker": {"color": "rgb(0,220,0)", "size": 3},
            "text": hover_text,
            "hovertemplate": "%{text}<extra>hardware</extra>",
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "name": "reference_state",
            "x": state_x,
            "y": state_y,
            "z": state_z,
            "line": {"color": "rgb(220,100,0)", "width": 4},
            "text": hover_text,
            "hovertemplate": "%{text}<extra>state</extra>",
        },
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "reference_action",
            "x": action_x,
            "y": action_y,
            "z": action_z,
            "line": {"color": "rgb(40,120,255)", "width": 4},
            "marker": {"color": "rgb(40,120,255)", "size": 3},
            "text": hover_text,
            "hovertemplate": "%{text}<extra>action</extra>",
        },
        build_pose_axis_trace(
            positions=hw_positions,
            rotations=hw_rotations,
            axis_index=0,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="hardware x-axis",
            color="rgb(255,80,80)",
            hover_text=hover_text,
        ),
        build_pose_axis_trace(
            positions=hw_positions,
            rotations=hw_rotations,
            axis_index=1,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="hardware y-axis",
            color="rgb(80,220,120)",
            hover_text=hover_text,
        ),
        build_pose_axis_trace(
            positions=hw_positions,
            rotations=hw_rotations,
            axis_index=2,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="hardware z-axis",
            color="rgb(80,140,255)",
            hover_text=hover_text,
        ),
        build_pose_axis_trace(
            positions=action_positions,
            rotations=action_rotations,
            axis_index=0,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="source x-axis",
            color="rgb(255,170,170)",
            hover_text=hover_text,
        ),
        build_pose_axis_trace(
            positions=action_positions,
            rotations=action_rotations,
            axis_index=1,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="source y-axis",
            color="rgb(170,255,190)",
            hover_text=hover_text,
        ),
        build_pose_axis_trace(
            positions=action_positions,
            rotations=action_rotations,
            axis_index=2,
            axis_length_m=_POSE_AXIS_LENGTH_M,
            name="source z-axis",
            color="rgb(170,200,255)",
            hover_text=hover_text,
        ),
    ]

    if peak_count > 0:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "peak hw-action error",
                "x": hw_positions[peak_order, 0].astype(float).tolist(),
                "y": hw_positions[peak_order, 1].astype(float).tolist(),
                "z": hw_positions[peak_order, 2].astype(float).tolist(),
                "marker": {"color": "rgb(255,64,64)", "size": 6, "symbol": "diamond"},
                "text": [hover_text[idx] for idx in peak_order.tolist()],
                "hovertemplate": "%{text}<extra>peak</extra>",
            }
        )

    layout = {
        "title": {
            "text": f"FR3 Replay Episode {episode_idx:03d} 3D Trajectory",
            "x": 0.02,
        },
        "paper_bgcolor": "#0b1020",
        "plot_bgcolor": "#0b1020",
        "font": {"color": "#e8ecf3", "family": "Menlo, Consolas, monospace"},
        "legend": {"orientation": "h", "x": 0.0, "y": 1.02},
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "X (m)", "backgroundcolor": "#11192d", "gridcolor": "#33415c", "zerolinecolor": "#4a5a78"},
            "yaxis": {"title": "Y (m)", "backgroundcolor": "#11192d", "gridcolor": "#33415c", "zerolinecolor": "#4a5a78"},
            "zaxis": {"title": "Z (m)", "backgroundcolor": "#11192d", "gridcolor": "#33415c", "zerolinecolor": "#4a5a78"},
            "camera": {"eye": {"x": 1.4, "y": -1.6, "z": 1.1}},
        },
        "annotations": [
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.0,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#c4ccda"},
                "text": (
                    f"dataset={dataset_path}<br>"
                    f"frames={len(frame_indices)} | peak markers={peak_count}<br>"
                    f"hw-action pos mean/p95={float(pos_errors_action_mm.mean()):.2f}/{float(np.percentile(pos_errors_action_mm,95)):.2f} mm<br>"
                    f"hw-action rot mean/p95={float(rot_errors_action_deg.mean()):.2f}/{float(np.percentile(rot_errors_action_deg,95)):.2f} deg<br>"
                    "drag to orbit, scroll to zoom, hover to inspect frame metrics and pose axes"
                ),
            }
        ],
    }

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FR3 Replay Episode {episode_idx:03d} 3D Trajectory</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #0b1020; color: #e8ecf3; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces, ensure_ascii=False)};
    const layout = {json.dumps(layout, ensure_ascii=False)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
'''
    output_path.write_text(html, encoding="utf-8")


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


def print_peak_diagnostics(
    *,
    frame_indices: np.ndarray,
    pos_errors_state_mm: np.ndarray,
    rot_errors_state_deg: np.ndarray,
    pos_errors_action_mm: np.ndarray,
    rot_errors_action_deg: np.ndarray,
    state_action_gap_mm: np.ndarray,
    state_action_gap_deg: np.ndarray,
    timestamps: np.ndarray,
    ee_frames: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    hw_positions: np.ndarray,
    measured_joints: np.ndarray,
    command_joints: np.ndarray | None,
    target_joints: np.ndarray | None,
) -> None:
    if len(pos_errors_state_mm) == 0:
        return

    peak_score = np.maximum(pos_errors_state_mm, pos_errors_action_mm)
    topk = min(_PEAK_DIAGNOSTIC_FRAMES, len(peak_score))
    peak_indices = np.argsort(peak_score)[-topk:][::-1]
    print("[INFO] 误差峰值诊断（按 max(hw-state, hw-action) 位置误差排序）")
    for local_idx in peak_indices:
        fi = int(frame_indices[local_idx])
        prev_local_idx = max(local_idx - 1, 0)
        ee_quat = np.asarray(ee_frames[local_idx][3:7], dtype=np.float64)
        ee_prev_quat = np.asarray(ee_frames[prev_local_idx][3:7], dtype=np.float64)
        state_quat = np.asarray(states[local_idx][3:7], dtype=np.float64)
        state_prev_quat = np.asarray(states[prev_local_idx][3:7], dtype=np.float64)
        action_quat = np.asarray(actions[local_idx][3:7], dtype=np.float64)
        action_prev_quat = np.asarray(actions[prev_local_idx][3:7], dtype=np.float64)
        ee_step_xyz_mm = float(np.linalg.norm(ee_frames[local_idx][:3] - ee_frames[prev_local_idx][:3]) * 1000.0)
        state_step_xyz_mm = float(np.linalg.norm(states[local_idx][:3] - states[prev_local_idx][:3]) * 1000.0)
        action_step_xyz_mm = float(np.linalg.norm(actions[local_idx][:3] - actions[prev_local_idx][:3]) * 1000.0)
        dt_s = float(max(timestamps[local_idx] - timestamps[prev_local_idx], 1e-9))
        ee_step_rot_deg = quaternion_angle_error_deg(ee_quat, ee_prev_quat)
        state_step_rot_deg = quaternion_angle_error_deg(state_quat, state_prev_quat)
        action_step_rot_deg = quaternion_angle_error_deg(action_quat, action_prev_quat)
        gripper_step_mm = float((actions[local_idx][7] - actions[prev_local_idx][7]) * 1000.0)
        joint_cmd_l2_deg = None
        joint_cmd_max_abs_deg = None
        joint_target_l2_deg = None
        joint_target_max_abs_deg = None
        target_cmd_l2_deg = None
        if command_joints is not None:
            joint_cmd_delta_deg = np.rad2deg(command_joints[local_idx] - measured_joints[local_idx])
            joint_cmd_l2_deg = float(np.linalg.norm(joint_cmd_delta_deg))
            joint_cmd_max_abs_deg = float(np.max(np.abs(joint_cmd_delta_deg)))
        if target_joints is not None:
            joint_target_delta_deg = np.rad2deg(target_joints[local_idx] - measured_joints[local_idx])
            joint_target_l2_deg = float(np.linalg.norm(joint_target_delta_deg))
            joint_target_max_abs_deg = float(np.max(np.abs(joint_target_delta_deg)))
        if target_joints is not None and command_joints is not None:
            target_cmd_l2_deg = float(np.linalg.norm(np.rad2deg(target_joints[local_idx] - command_joints[local_idx])))
        print(
            f"  frame={fi:4d} ts={timestamps[local_idx]:7.3f}s  "
            f"pos_err_state={pos_errors_state_mm[local_idx]:6.2f}mm  "
            f"pos_err_action={pos_errors_action_mm[local_idx]:6.2f}mm"
        )
        print(
            f"    rot_err_state={rot_errors_state_deg[local_idx]:5.2f}deg  "
            f"rot_err_action={rot_errors_action_deg[local_idx]:5.2f}deg  "
            f"state_action_gap={state_action_gap_mm[local_idx]:6.2f}mm/{state_action_gap_deg[local_idx]:5.2f}deg"
        )
        print(
            f"    z: hw={hw_positions[local_idx,2]:+.4f}m  ee_src={ee_frames[local_idx][2]:+.4f}m  "
            f"state={states[local_idx][2]:+.4f}m  action={actions[local_idx][2]:+.4f}m"
        )
        print(
            f"    dz_prev: ee_src={(ee_frames[local_idx][2]-ee_frames[prev_local_idx][2])*1000:+6.2f}mm  "
            f"state={(states[local_idx][2]-states[prev_local_idx][2])*1000:+6.2f}mm  "
            f"action={(actions[local_idx][2]-actions[prev_local_idx][2])*1000:+6.2f}mm"
        )
        print(
            f"    dt_prev={dt_s*1000:6.2f}ms  "
            f"quat_step_prev: ee_src={ee_step_rot_deg:5.2f}deg  "
            f"state={state_step_rot_deg:5.2f}deg  "
            f"action={action_step_rot_deg:5.2f}deg"
        )
        print(
            f"    xyz_step_prev: ee_src={ee_step_xyz_mm:6.2f}mm  "
            f"state={state_step_xyz_mm:6.2f}mm  action={action_step_xyz_mm:6.2f}mm"
        )
        print(
            f"    xyz_speed_prev: ee_src={ee_step_xyz_mm / dt_s:7.2f}mm/s  "
            f"state={state_step_xyz_mm / dt_s:7.2f}mm/s  "
            f"action={action_step_xyz_mm / dt_s:7.2f}mm/s"
        )
        print(
            f"    gripper: aperture={actions[local_idx][7]*1000:6.2f}mm  "
            f"delta_prev={gripper_step_mm:+6.2f}mm"
        )
        if joint_cmd_l2_deg is not None:
            print(
                f"    q_meas_vs_q_cmd: l2={joint_cmd_l2_deg:5.2f}deg  "
                f"max_abs={joint_cmd_max_abs_deg:5.2f}deg"
            )
        if joint_target_l2_deg is not None:
            print(
                f"    q_meas_vs_q_target: l2={joint_target_l2_deg:5.2f}deg  "
                f"max_abs={joint_target_max_abs_deg:5.2f}deg"
            )
        if target_cmd_l2_deg is not None:
            print(f"    q_cmd_vs_q_target: l2={target_cmd_l2_deg:5.2f}deg")


# ---------------------------------------------------------------------------
# 真机重播主循环
# ---------------------------------------------------------------------------


def replay_real(args: argparse.Namespace) -> int:
    from lerobot.robots.franka_research3 import FrankaResearch3
    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config

    if args.joint_targets_csv is not None and not args.allow_experimental_joint_replay:
        raise RuntimeError(
            "Real-hardware replay defaults have been rolled back to the validated "
            "action[t] + OTG path. Add --allow-experimental-joint-replay to use "
            "joint-target CSV replay."
        )
    if args.disable_otg and not args.allow_unsafe_otg_bypass:
        raise RuntimeError(
            "Real-hardware replay keeps OTG enabled by default. Add "
            "--allow-unsafe-otg-bypass to use --disable-otg."
        )

    # ── 加载数据 ──────────────────────────────────────────────────────
    print(f"[INFO] 加载 episode {args.episode}  dataset={args.dataset}")
    ep = load_episode(args.dataset, args.episode)
    states = ep["state"]
    actions = ep["action"]
    timestamps = ep["timestamp"]
    n_frames = len(states)
    print(f"[INFO] {n_frames} 帧 @ {args.fps} fps")
    ee_frames = actions
    joint_target_sequence = None
    if args.joint_targets_csv is not None:
        joint_target_sequence = load_joint_target_sequence(
            args.joint_targets_csv,
            n_frames=n_frames,
            column_prefix=args.joint_target_column_prefix,
        )
        print(
            "[INFO] Joint 重播源: "
            f"{args.joint_targets_csv}  prefix={args.joint_target_column_prefix}"
        )
        print("[INFO] EE 参考源: action[t]  (用于对比统计与 gripper)")
    else:
        print("[INFO] EE 重播源: action[t]")
    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        print(
            "[INFO] 数据集时间戳: "
            f"mean_dt={dt.mean():.5f}s  std={dt.std():.5f}s  "
            f"min={dt.min():.5f}s  max={dt.max():.5f}s"
        )
    print(f"[INFO] 时序源: {args.timing_source}")
    analysis_output_dir = args.analysis_output_dir.resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 分析输出目录: {analysis_output_dir}")
    replay_record_dataset: LeRobotDataset | None = None
    replay_record_task = f"real_replay:{Path(args.dataset).name}:ep{args.episode:03d}"
    if args.record_replay_dataset is not None:
        replay_record_root = args.record_replay_dataset.resolve()
        replay_record_dataset = create_replay_record_dataset(
            replay_record_root,
            fps=args.fps,
            source_dataset_path=args.dataset,
            episode_idx=args.episode,
        )
        print(f"[INFO] Replay 记录数据集输出: {replay_record_root}")

    # ── 预检查：打印轨迹 Z 范围 ───────────────────────────────────────
    T_B_E_reset = pose_from_xyzquat(_RESET_POSE_B_XYZQUAT)
    T_Ws_I0 = pose_from_xyzquat(states[0])
    preview_first_frame_tilt = describe_first_frame_tilt(T_Ws_I0)
    preview_legacy_z_offset_m = (
        float(args.legacy_z_offset_m) if bool(preview_first_frame_tilt["legacy_tilt"]) else 0.0
    )
    T_B_Ws_preview = build_T_B_Ws(T_B_E_reset, T_Ws_I0)
    target_poses = [
        apply_pose_z_offset(
            ws_to_base(T_B_Ws_preview, pose_from_xyzquat(ee_frames[fi])),
            preview_legacy_z_offset_m,
        )
        for fi in range(n_frames)
    ]
    ee_z_vals = [float(T[2, 3]) for T in target_poses]
    ee_z_min, ee_z_max = min(ee_z_vals), max(ee_z_vals)
    tool_lowest_z_vals = [estimate_finger_lowest_z(T) for T in target_poses]
    tool_z_min, tool_z_max = min(tool_lowest_z_vals), max(tool_lowest_z_vals)
    n_below = sum(1 for z in tool_lowest_z_vals if z < args.min_tool_z_m)
    print(
        "[INFO] 轨迹 Z 范围: "
        f"ee_origin min={ee_z_min:.4f}m max={ee_z_max:.4f}m  "
        f"finger_lowest_est min={tool_z_min:.4f}m max={tool_z_max:.4f}m  "
        f"低于 {args.min_tool_z_m:.3f}m 的帧数={n_below}"
    )
    if n_below > 0:
        print(f"[WARN] {n_below} 帧估计手指最低点 Z < {args.min_tool_z_m:.3f}m，这些帧将被跳过（不发给机器人）")

    # ── 移动到 DAS 起始关节角（先于 connect，避免控制器冲突）──────────
    move_to_das_start(args.robot_ip)

    # ── 初始化真机 ────────────────────────────────────────────────────
    otg_max_velocity = resolve_joint_limit_values(
        FrankaResearch3Config.otg_max_velocity,
        args.otg_max_velocity,
        args.otg_velocity_scale,
    )
    otg_max_acceleration = resolve_joint_limit_values(
        FrankaResearch3Config.otg_max_acceleration,
        args.otg_max_acceleration,
        args.otg_acceleration_scale,
    )
    otg_max_jerk = resolve_joint_limit_values(
        FrankaResearch3Config.otg_max_jerk,
        args.otg_max_jerk,
        args.otg_jerk_scale,
    )

    cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        damping=args.damping,
        stiffness=args.stiffness,
        filter_coeff=args.filter_coeff,
        use_otg=not args.disable_otg,
        otg_max_velocity=otg_max_velocity,
        otg_max_acceleration=otg_max_acceleration,
        otg_max_jerk=otg_max_jerk,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        workspace_min=(0.1, -0.6, args.min_tool_z_m),  # x min < reset_x(0.153)，避免首帧被 clip 引起 IK 偏移
        workspace_max=(0.9, 0.6, 0.8),
    )
    robot = FrankaResearch3(cfg)
    print(f"[INFO] 连接真机 {args.robot_ip} ...")
    if args.filter_coeff is not None or args.damping is not None or args.stiffness is not None:
        print(
            "[INFO] Arm 控制器参数: "
            f"filter_coeff={args.filter_coeff} "
            f"damping={args.damping} "
            f"stiffness={args.stiffness}"
        )
    print(
        "[INFO] OTG 参数: "
        f"use_otg={cfg.use_otg} "
        f"vel_scale={args.otg_velocity_scale:.3f} "
        f"acc_scale={args.otg_acceleration_scale:.3f} "
        f"jerk_scale={args.otg_jerk_scale:.3f}"
    )
    print(f"[INFO] OTG max_velocity   = {np.round(np.asarray(cfg.otg_max_velocity), 4).tolist()}")
    print(f"[INFO] OTG max_accel      = {np.round(np.asarray(cfg.otg_max_acceleration), 4).tolist()}")
    print(f"[INFO] OTG max_jerk       = {np.round(np.asarray(cfg.otg_max_jerk), 4).tolist()}")
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
        obs_init = robot.get_observation()
        T_B_E_actual = np.eye(4, dtype=np.float64)
        T_B_E_actual[:3, 3] = [obs_init["ee.x"], obs_init["ee.y"], obs_init["ee.z"]]
        T_B_E_actual[:3, :3] = _rotation_class().from_rotvec(
            [obs_init["ee.wx"], obs_init["ee.wy"], obs_init["ee.wz"]]
        ).as_matrix()
        print(f"[INFO] 实际 EE 位置: xyz=[{obs_init['ee.x']:.4f}, {obs_init['ee.y']:.4f}, {obs_init['ee.z']:.4f}]")
        print(f"[INFO] 理论 reset xyz={_RESET_POSE_B_XYZQUAT[:3].round(4)}  "
              f"偏差 dz={obs_init['ee.z'] - _RESET_POSE_B_XYZQUAT[2]:.4f}m")

        # ── 根据数据集 pose 语义选择初始化模式─────────
        T_Ws_I0 = pose_from_xyzquat(states[0])
        first_frame_tilt = describe_first_frame_tilt(T_Ws_I0)
        first_frame_axis = np.asarray(first_frame_tilt["axis"], dtype=np.float64)
        use_legacy_pos_only = bool(first_frame_tilt["legacy_tilt"])
        legacy_z_offset_m = float(args.legacy_z_offset_m) if use_legacy_pos_only else 0.0
        if use_legacy_pos_only:
            T_B_Ws = build_T_B_Ws_actual_pos_only(T_B_E_actual, T_Ws_I0)
            start_blend_frames = _LEGACY_START_BLEND_FRAMES
            init_mode = "legacy_tilt_pos_only_actual"
        else:
            T_B_Ws = T_B_E_actual @ se3_inv(_T_IE) @ se3_inv(T_Ws_I0)
            start_blend_frames = 0
            init_mode = "full_actual"
        T_B_E0_check = apply_pose_z_offset(
            ws_to_base(T_B_Ws, T_Ws_I0),
            legacy_z_offset_m,
        )
        first_frame_rotvec = _rotation_class().from_matrix(T_Ws_I0[:3, :3]).as_rotvec()
        print(
            "[INFO] 首帧 contract 检测: "
            f"mode={init_mode} "
            f"rot={np.rad2deg(first_frame_rotvec).round(2).tolist()}deg "
            f"angle={float(first_frame_tilt['angle_deg']):.2f}deg "
            f"axis={first_frame_axis.round(3).tolist()}"
        )
        print(f"[INFO] T(B, W_s)  pos={T_B_Ws[:3,3].round(4)}")
        print(f"[INFO] t=0 命令预测 xyz={T_B_E0_check[:3,3].round(4)}  "
              f"（应 ≈ 实际 EE 位置，差值={np.linalg.norm(T_B_E0_check[:3,3] - T_B_E_actual[:3,3])*1000:.2f}mm）")
        if use_legacy_pos_only:
            print(
                "[INFO] 旧数据首帧 tilt 已检测到: "
                f"启动 {start_blend_frames} 帧平滑过渡，抑制首帧下沉/姿态突变"
            )
            print(
                "[INFO] 旧数据 Z 矫正已启用: "
                f"legacy_z_offset={legacy_z_offset_m * 1000.0:.1f}mm"
            )

        # ── 初始化 Rerun ──────────────────────────────────────────────
        rr_ok, rrd_path = init_rerun(args.episode, output_dir=analysis_output_dir)

        # ── 误差记录 ──────────────────────────────────────────────────
        processed_frame_indices: list[int] = []
        pos_errors_state_mm: list[float] = []
        rot_errors_state_deg: list[float] = []
        pos_errors_action_mm: list[float] = []
        rot_errors_action_deg: list[float] = []
        state_action_gap_mm: list[float] = []
        state_action_gap_deg: list[float] = []
        joint_cmd_track_l2_deg: list[float] = []
        joint_cmd_track_max_abs_deg: list[float] = []
        joint_target_track_l2_deg: list[float] = []
        joint_target_track_max_abs_deg: list[float] = []
        joint_target_cmd_l2_deg: list[float] = []
        skipped_frames: list[int] = []
        abort_reason: str | None = None
        abort_frame: int | None = None
        consecutive_stall_frames = 0

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

        hw_positions: list[np.ndarray] = []
        hw_rotations_history: list[np.ndarray] = []
        state_positions_history: list[np.ndarray] = []
        action_positions_history: list[np.ndarray] = []
        action_rotations_history: list[np.ndarray] = []
        measured_joint_history: list[np.ndarray] = []
        command_joint_history: list[np.ndarray] = []
        target_joint_history: list[np.ndarray] = []

        print(f"\n[INFO] 开始真机重播 ({n_frames} 帧)…\n")

        for fi in range(n_frames):
            t0 = time.perf_counter()
            T_Ws_Et_star = pose_from_xyzquat(ee_frames[fi])
            T_B_Et_star = apply_pose_z_offset(
                ws_to_base(T_B_Ws, T_Ws_Et_star),
                legacy_z_offset_m,
            )
            T_B_Et_cmd = T_B_Et_star
            blend_alpha = 1.0
            if start_blend_frames > 0:
                blend_alpha = min(float(fi) / float(start_blend_frames), 1.0)
                T_B_Et_cmd = interpolate_pose(T_B_E_actual, T_B_Et_star, blend_alpha)
            if args.timing_source == "timestamp" and fi + 1 < n_frames:
                target_dt = max(0.0, float(timestamps[fi + 1] - timestamps[fi]))
            else:
                target_dt = 1.0 / args.fps

            # 前3帧打印命令位置，确认启动过渡与目标轨迹
            if fi < 3:
                print(
                    f"  [DEBUG] frame {fi}: cmd xyz={T_B_Et_cmd[:3,3].round(4)}  "
                    f"target xyz={T_B_Et_star[:3,3].round(4)}  blend_alpha={blend_alpha:.2f}"
                )

            # 安全检查：按手指最低点估计 Z，低于阈值则跳过，防止夹爪/手指怼桌
            target_z = estimate_finger_lowest_z(T_B_Et_cmd)
            command_sent = target_z >= args.min_tool_z_m
            if command_sent:
                rotvec = _rotation_class().from_matrix(T_B_Et_cmd[:3, :3]).as_rotvec()
                if joint_target_sequence is not None:
                    robot.send_joint_positions(
                        joint_target_sequence[fi],
                        gripper_pos=float(action_gripper_normalized[fi]),
                    )
                else:
                    robot.send_action({
                        "ee.x":  float(T_B_Et_cmd[0, 3]),
                        "ee.y":  float(T_B_Et_cmd[1, 3]),
                        "ee.z":  float(T_B_Et_cmd[2, 3]),
                        "ee.wx": float(rotvec[0]),
                        "ee.wy": float(rotvec[1]),
                        "ee.wz": float(rotvec[2]),
                        "gripper.pos": float(action_gripper_normalized[fi]),
                    })
            if target_z < args.min_tool_z_m:
                print(
                    f"  [WARN] frame {fi:4d}: finger_lowest_est_z={target_z:.4f}m "
                    f"(ee_z={float(T_B_Et_cmd[2, 3]):.4f}m) < {args.min_tool_z_m:.3f}m，跳过"
                )
                skipped_frames.append(fi)

            # 读取当前 EE pose（误差统计用）
            obs = robot.get_observation()
            hw_pos = np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]], dtype=np.float64)
            hw_rot = _rotation_class().from_rotvec([obs["ee.wx"], obs["ee.wy"], obs["ee.wz"]]).as_matrix()
            hw_pose_xyzquat = pose_to_xyzquat(
                np.block(
                    [
                        [hw_rot, hw_pos.reshape(3, 1)],
                        [np.zeros((1, 3), dtype=np.float64), np.ones((1, 1), dtype=np.float64)],
                    ]
                )
            )
            measured_joints = np.array([obs[f"joint_{joint_idx}.pos"] for joint_idx in range(1, 8)], dtype=np.float64)
            target_joints, command_joints = snapshot_otg_debug(robot)
            T_B_Et_state = apply_pose_z_offset(
                ws_to_base(T_B_Ws, pose_from_xyzquat(states[fi])),
                legacy_z_offset_m,
            )
            state_pose_xyzquat = pose_to_xyzquat(T_B_Et_state)
            action_pose_xyzquat = pose_to_xyzquat(T_B_Et_star)
            command_pose_xyzquat = pose_to_xyzquat(T_B_Et_cmd)
            replay_status = np.array(
                [
                    1.0 if command_sent else 0.0,
                    0.0 if command_sent else 1.0,
                    float(blend_alpha),
                    float(target_z),
                    float(T_B_Et_cmd[2, 3]),
                    float(T_B_Et_star[2, 3]),
                    float(target_dt),
                ],
                dtype=np.float32,
            )
            if replay_record_dataset is not None:
                replay_record_dataset.add_frame(
                    build_replay_record_frame(
                        measured_pose_xyzquat=hw_pose_xyzquat,
                        measured_gripper_normalized=float(obs["gripper.pos"]),
                        command_pose_xyzquat=command_pose_xyzquat,
                        command_gripper_normalized=float(action_gripper_normalized[fi]),
                        reference_state_pose_xyzquat=state_pose_xyzquat,
                        reference_state_gripper_aperture_m=float(states[fi][7]),
                        reference_action_pose_xyzquat=action_pose_xyzquat,
                        reference_action_gripper_aperture_m=float(actions[fi][7]),
                        source_state_row=states[fi],
                        source_action_row=actions[fi],
                        measured_joints=measured_joints,
                        command_joints=command_joints,
                        target_joints=target_joints,
                        replay_status=replay_status,
                        task=replay_record_task,
                    )
                )
            abort_reason = detect_robot_abort_reason(robot)
            if abort_reason is not None:
                abort_frame = fi
                print(f"[ERROR] frame {fi:4d}: 检测到真机控制中止，立即停止 replay: {abort_reason}")
                break
            if not command_sent:
                elapsed = time.perf_counter() - t0
                if (sleep_t := target_dt - elapsed) > 0:
                    time.sleep(sleep_t)
                continue
            hw_positions.append(hw_pos.copy())
            hw_rotations_history.append(hw_rot.copy())
            measured_joint_history.append(measured_joints.copy())
            if command_joints is not None:
                command_joint_history.append(command_joints.copy())
            if target_joints is not None:
                target_joint_history.append(target_joints.copy())

            # 录制参考 pose
            state_pos = T_B_Et_state[:3, 3]
            state_rot = T_B_Et_state[:3, :3]
            action_pos = T_B_Et_star[:3, 3]
            action_rot = T_B_Et_star[:3, :3]

            pos_err_state_mm = float(np.linalg.norm(hw_pos - state_pos) * 1000.0)
            rot_err_state_deg = rotation_angle_error_deg(hw_rot, state_rot)
            pos_err_action_mm = float(np.linalg.norm(hw_pos - action_pos) * 1000.0)
            rot_err_action_deg = rotation_angle_error_deg(hw_rot, action_rot)
            ref_gap_mm = float(np.linalg.norm(state_pos - action_pos) * 1000.0)
            ref_gap_deg = rotation_angle_error_deg(state_rot, action_rot)
            processed_frame_indices.append(fi)
            state_positions_history.append(state_pos.copy())
            action_positions_history.append(action_pos.copy())
            action_rotations_history.append(action_rot.copy())
            pos_errors_state_mm.append(pos_err_state_mm)
            rot_errors_state_deg.append(rot_err_state_deg)
            pos_errors_action_mm.append(pos_err_action_mm)
            rot_errors_action_deg.append(rot_err_action_deg)
            state_action_gap_mm.append(ref_gap_mm)
            state_action_gap_deg.append(ref_gap_deg)
            if command_joints is not None:
                command_delta_deg = np.rad2deg(command_joints - measured_joints)
                joint_cmd_track_l2_deg.append(float(np.linalg.norm(command_delta_deg)))
                joint_cmd_track_max_abs_deg.append(float(np.max(np.abs(command_delta_deg))))
            if target_joints is not None:
                target_delta_deg = np.rad2deg(target_joints - measured_joints)
                joint_target_track_l2_deg.append(float(np.linalg.norm(target_delta_deg)))
                joint_target_track_max_abs_deg.append(float(np.max(np.abs(target_delta_deg))))
            if target_joints is not None and command_joints is not None:
                joint_target_cmd_l2_deg.append(
                    float(np.linalg.norm(np.rad2deg(target_joints - command_joints)))
                )
            if len(hw_positions) >= 2 and command_joints is not None:
                hw_step_mm = float(np.linalg.norm(hw_positions[-1] - hw_positions[-2]) * 1000.0)
                joint_cmd_l2_deg_current = joint_cmd_track_l2_deg[-1] if joint_cmd_track_l2_deg else 0.0
                if (
                    hw_step_mm <= _STALL_HW_STEP_MM_THRESHOLD
                    and pos_err_action_mm >= _STALL_POS_ERR_MM_THRESHOLD
                    and joint_cmd_l2_deg_current >= _STALL_Q_CMD_ERR_DEG_THRESHOLD
                ):
                    consecutive_stall_frames += 1
                else:
                    consecutive_stall_frames = 0
                if consecutive_stall_frames >= _STALL_CONSECUTIVE_FRAMES:
                    abort_reason = (
                        "控制疑似已中止：硬件连续停滞且命令/测量持续大幅偏离 "
                        f"(hw_step<={_STALL_HW_STEP_MM_THRESHOLD:.1f}mm, "
                        f"pos_err>={_STALL_POS_ERR_MM_THRESHOLD:.1f}mm, "
                        f"q_meas_vs_q_cmd>={_STALL_Q_CMD_ERR_DEG_THRESHOLD:.1f}deg)"
                    )
                    abort_frame = fi
                    print(f"[ERROR] frame {fi:4d}: {abort_reason}")
                    break

            log_frame(
                rr_ok,
                fi,
                float(timestamps[fi]),
                hw_pos,
                state_pos,
                action_pos,
                pos_err_state_mm,
                rot_err_state_deg,
                pos_err_action_mm,
                rot_err_action_deg,
                float(action_gripper_normalized[fi]),
                np.asarray(hw_positions, dtype=np.float64),
                np.asarray(state_positions_history, dtype=np.float64),
                np.asarray(action_positions_history, dtype=np.float64),
            )

            elapsed = time.perf_counter() - t0
            if (sleep_t := target_dt - elapsed) > 0:
                time.sleep(sleep_t)

            if fi % 30 == 0:
                print(
                    f"  [{fi:4d}/{n_frames}]  "
                    f"state_err={pos_err_state_mm:6.2f}mm/{rot_err_state_deg:5.2f}°  "
                    f"action_err={pos_err_action_mm:6.2f}mm/{rot_err_action_deg:5.2f}°  "
                    f"state_action_gap={ref_gap_mm:6.2f}mm"
                )

        # ── 统计汇总 ──────────────────────────────────────────────────
        if skipped_frames:
            print(f"\n[WARN] 共跳过 {len(skipped_frames)} 帧（估计手指最低点 Z < {args.min_tool_z_m:.3f}m）: {skipped_frames[:10]}"
                  f"{'...' if len(skipped_frames) > 10 else ''}")

        pos_state_arr = np.array(pos_errors_state_mm, dtype=np.float64)
        if len(pos_state_arr) == 0:
            print("[WARN] 无有效帧（全部被跳过），无统计数据")
            return 0
        frame_idx_arr = np.asarray(processed_frame_indices, dtype=np.int64)
        rot_state_arr = np.array(rot_errors_state_deg, dtype=np.float64)
        pos_action_arr = np.array(pos_errors_action_mm, dtype=np.float64)
        rot_action_arr = np.array(rot_errors_action_deg, dtype=np.float64)
        gap_pos_arr = np.array(state_action_gap_mm, dtype=np.float64)
        gap_rot_arr = np.array(state_action_gap_deg, dtype=np.float64)
        hw_pos_arr = np.asarray(hw_positions, dtype=np.float64)
        hw_rot_arr = np.asarray(hw_rotations_history, dtype=np.float64)
        state_pos_arr = np.asarray(state_positions_history, dtype=np.float64)
        action_pos_arr = np.asarray(action_positions_history, dtype=np.float64)
        action_rot_arr = np.asarray(action_rotations_history, dtype=np.float64)
        measured_joint_arr = np.asarray(measured_joint_history, dtype=np.float64)
        command_joint_arr = (
            np.asarray(command_joint_history, dtype=np.float64)
            if len(command_joint_history) == len(processed_frame_indices)
            else None
        )
        target_joint_arr = (
            np.asarray(target_joint_history, dtype=np.float64)
            if len(target_joint_history) == len(processed_frame_indices)
            else None
        )
        _WARMUP = min(30, len(pos_state_arr))
        pos_state_stable = pos_state_arr[_WARMUP:]
        rot_state_stable = rot_state_arr[_WARMUP:]
        pos_action_stable = pos_action_arr[_WARMUP:]
        rot_action_stable = rot_action_arr[_WARMUP:]
        gap_pos_stable = gap_pos_arr[_WARMUP:]
        gap_rot_stable = gap_rot_arr[_WARMUP:]

        print("\n" + "=" * 60)
        if abort_reason is None:
            print(f"  Episode {args.episode} 真机重播完成   {n_frames} 帧 @ {args.fps} fps")
        else:
            print(f"  Episode {args.episode} 真机重播中止   截止 frame {abort_frame} @ {args.fps} fps")
        print("=" * 60)
        if abort_reason is not None:
            print(f"  [ERROR] 本次统计截止到中止帧 {abort_frame}: {abort_reason}")
        summarize_metric("【全程】跟踪误差 vs action 位置", pos_action_arr, "mm")
        summarize_metric("【全程】跟踪误差 vs action 旋转", rot_action_arr, "°")
        summarize_metric("【全程】复现误差 vs state 位置", pos_state_arr, "mm")
        summarize_metric("【全程】复现误差 vs state 旋转", rot_state_arr, "°")
        summarize_metric("【全程】state-action 源数据位置差", gap_pos_arr, "mm")
        summarize_metric("【全程】state-action 源数据旋转差", gap_rot_arr, "°")
        if len(pos_state_stable) > 0:
            summarize_metric(f"【稳定期 frame {_WARMUP}+】跟踪误差 vs action 位置", pos_action_stable, "mm")
            summarize_metric(f"【稳定期 frame {_WARMUP}+】跟踪误差 vs action 旋转", rot_action_stable, "°")
            summarize_metric(f"【稳定期 frame {_WARMUP}+】复现误差 vs state 位置", pos_state_stable, "mm")
            summarize_metric(f"【稳定期 frame {_WARMUP}+】复现误差 vs state 旋转", rot_state_stable, "°")
            summarize_metric(f"【稳定期 frame {_WARMUP}+】state-action 源数据位置差", gap_pos_stable, "mm")
            summarize_metric(f"【稳定期 frame {_WARMUP}+】state-action 源数据旋转差", gap_rot_stable, "°")
        if joint_cmd_track_l2_deg:
            summarize_metric("【全程】q_meas vs q_cmd 关节 L2", np.asarray(joint_cmd_track_l2_deg), "deg")
            summarize_metric(
                "【全程】q_meas vs q_cmd 单关节 max_abs",
                np.asarray(joint_cmd_track_max_abs_deg),
                "deg",
            )
        if joint_target_track_l2_deg:
            summarize_metric("【全程】q_meas vs q_target 关节 L2", np.asarray(joint_target_track_l2_deg), "deg")
            summarize_metric(
                "【全程】q_meas vs q_target 单关节 max_abs",
                np.asarray(joint_target_track_max_abs_deg),
                "deg",
            )
        if joint_target_cmd_l2_deg:
            summarize_metric("【全程】q_cmd vs q_target 关节 L2", np.asarray(joint_target_cmd_l2_deg), "deg")
        pose_restore_mask = (
            (pos_action_arr <= _POSE_RESTORE_POS_THRESHOLD_MM)
            & (rot_action_arr <= _POSE_RESTORE_ROT_THRESHOLD_DEG)
        )
        print(
            f"  【全程】pose 还原达标 (<={_POSE_RESTORE_POS_THRESHOLD_MM:.1f}mm & "
            f"<={_POSE_RESTORE_ROT_THRESHOLD_DEG:.1f}deg)   "
            f"{int(np.count_nonzero(pose_restore_mask))}/{len(pose_restore_mask)} "
            f"({100.0 * float(np.mean(pose_restore_mask)):.1f}%)"
        )
        if len(pos_action_stable) > 0:
            stable_restore_mask = (
                (pos_action_stable <= _POSE_RESTORE_POS_THRESHOLD_MM)
                & (rot_action_stable <= _POSE_RESTORE_ROT_THRESHOLD_DEG)
            )
            print(
                f"  【稳定期 frame {_WARMUP}+】pose 还原达标 (<={_POSE_RESTORE_POS_THRESHOLD_MM:.1f}mm & "
                f"<={_POSE_RESTORE_ROT_THRESHOLD_DEG:.1f}deg)   "
                f"{int(np.count_nonzero(stable_restore_mask))}/{len(stable_restore_mask)} "
                f"({100.0 * float(np.mean(stable_restore_mask)):.1f}%)"
            )
        print("=" * 60)
        trajectory_csv_path = analysis_output_dir / f"fr3_real_replay_ep{args.episode:03d}_trajectory.csv"
        write_trajectory_csv(
            trajectory_csv_path,
            frame_indices=frame_idx_arr,
            timestamps=timestamps[frame_idx_arr],
            hw_positions=hw_pos_arr,
            state_positions=state_pos_arr,
            action_positions=action_pos_arr,
            pos_errors_state_mm=pos_state_arr,
            rot_errors_state_deg=rot_state_arr,
            pos_errors_action_mm=pos_action_arr,
            rot_errors_action_deg=rot_action_arr,
            state_action_gap_mm=gap_pos_arr,
            state_action_gap_deg=gap_rot_arr,
            measured_joints=measured_joint_arr,
            command_joints=command_joint_arr,
            target_joints=target_joint_arr,
        )
        print(f"[INFO] 轨迹 CSV 已保存: {trajectory_csv_path}")
        trajectory_html_path = analysis_output_dir / f"fr3_real_replay_ep{args.episode:03d}_trajectory.html"
        write_trajectory_plot_html(
            trajectory_html_path,
            episode_idx=args.episode,
            dataset_path=args.dataset,
            frame_indices=frame_idx_arr,
            timestamps=timestamps[frame_idx_arr],
            hw_positions=hw_pos_arr,
            hw_rotations=hw_rot_arr,
            state_positions=state_pos_arr,
            action_positions=action_pos_arr,
            action_rotations=action_rot_arr,
            pos_errors_state_mm=pos_state_arr,
            rot_errors_state_deg=rot_state_arr,
            pos_errors_action_mm=pos_action_arr,
            rot_errors_action_deg=rot_action_arr,
            state_action_gap_mm=gap_pos_arr,
        )
        print(f"[INFO] 3D HTML 已保存: {trajectory_html_path}")
        if rrd_path is not None:
            print(f"[INFO] Rerun 3D 轨迹已保存: {rrd_path}")
        print_peak_diagnostics(
            frame_indices=frame_idx_arr,
            pos_errors_state_mm=pos_state_arr,
            rot_errors_state_deg=rot_state_arr,
            pos_errors_action_mm=pos_action_arr,
            rot_errors_action_deg=rot_action_arr,
            state_action_gap_mm=gap_pos_arr,
            state_action_gap_deg=gap_rot_arr,
            timestamps=timestamps[frame_idx_arr],
            ee_frames=ee_frames[frame_idx_arr],
            states=states[frame_idx_arr],
            actions=actions[frame_idx_arr],
            hw_positions=hw_pos_arr,
            measured_joints=measured_joint_arr,
            command_joints=command_joint_arr,
            target_joints=target_joint_arr,
        )

    finally:
        if replay_record_dataset is not None:
            try:
                if int(replay_record_dataset.episode_buffer["size"]) > 0:
                    replay_record_dataset.save_episode()
                    print(
                        "[INFO] Replay 记录数据集已保存: "
                        f"{args.record_replay_dataset.resolve()}"
                    )
            finally:
                replay_record_dataset.finalize()
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
    parser.add_argument("--timing-source", choices=["fps", "timestamp"], default="timestamp")
    parser.add_argument("--robot-ip", default="192.168.1.208")
    parser.add_argument("--filter-coeff", type=float, default=None)
    parser.add_argument("--damping", type=parse_joint_gains, default=None)
    parser.add_argument("--stiffness", type=parse_joint_gains, default=None)
    parser.add_argument("--otg-max-velocity", type=parse_joint_limit_values, default=None)
    parser.add_argument("--otg-max-acceleration", type=parse_joint_limit_values, default=None)
    parser.add_argument("--otg-max-jerk", type=parse_joint_limit_values, default=None)
    parser.add_argument("--otg-velocity-scale", type=positive_scale, default=_DEFAULT_OTG_SCALE)
    parser.add_argument("--otg-acceleration-scale", type=positive_scale, default=_DEFAULT_OTG_SCALE)
    parser.add_argument("--otg-jerk-scale", type=positive_scale, default=_DEFAULT_OTG_SCALE)
    parser.add_argument(
        "--min-tool-z-m",
        type=positive_scale,
        default=_DEFAULT_MIN_TOOL_Z_M,
        help="Minimum estimated finger-lowest Z in base frame; frames below this threshold are skipped.",
    )
    parser.add_argument(
        "--legacy-z-offset-m",
        type=float,
        default=_DEFAULT_LEGACY_Z_OFFSET_M,
        help="Base-frame Z correction applied only when the legacy first-frame tilt contract is detected.",
    )
    parser.add_argument("--disable-otg", action="store_true")
    parser.add_argument("--joint-targets-csv", type=str, default=None)
    parser.add_argument("--joint-target-column-prefix", type=str, default="bc_joint")
    parser.add_argument("--allow-experimental-joint-replay", action="store_true")
    parser.add_argument("--allow-unsafe-otg-bypass", action="store_true")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-backend", choices=["pika", "das"], default="das")
    parser.add_argument("--reset-gripper-position", type=float, default=_DAS_RESET_POSITION)
    parser.add_argument("--reset-gripper-timeout-s", type=float, default=2.0)
    parser.add_argument("--analysis-output-dir", type=Path, default=_DEFAULT_ANALYSIS_OUTPUT_DIR)
    parser.add_argument(
        "--record-replay-dataset",
        type=Path,
        default=None,
        help="Optional output root for recording the replay run as a LeRobot v3 dataset.",
    )
    return parser.parse_args(argv)


def main() -> int:
    return replay_real(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
