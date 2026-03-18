#!/usr/bin/env python3
"""
计算 _IK_SEED_JOINTS_RAD 偏移后的新关节角。

用法（Docker 容器内）：
    python tools/fr3/compute_seed_joints.py
    python tools/fr3/compute_seed_joints.py --z-delta -0.015
"""
import argparse
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAS_URDF = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf"
_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3",
    "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
_IK_SEED_JOINTS_RAD = np.array(
    [-0.057892, -1.550292, -1.694795, -2.125873, 0.022874, 2.119849, -0.948928],
    dtype=np.float64,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-delta", type=float, default=-0.015,
                        help="base 坐标系 Z 方向偏移量（米，负数=下移，默认 -0.015）")
    args = parser.parse_args()

    from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

    kin = PlacoKinematicsDriver(
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        joint_names=_JOINT_NAMES,
    )

    T_cur = np.array(kin.forward_kinematics(_IK_SEED_JOINTS_RAD.tolist()))
    print(f"当前 EE xyz : {T_cur[:3, 3].round(4)}")

    T_tgt = T_cur.copy()
    T_tgt[2, 3] += args.z_delta

    new_joints = np.array(kin.inverse_kinematics(_IK_SEED_JOINTS_RAD.tolist(), T_tgt))

    T_verify = np.array(kin.forward_kinematics(new_joints.tolist()))
    err_mm = np.linalg.norm(T_verify[:3, 3] - T_tgt[:3, 3]) * 1000
    print(f"目标 EE xyz : {T_tgt[:3, 3].round(4)}  (Z delta={args.z_delta:+.3f}m)")
    print(f"验证 EE xyz : {T_verify[:3, 3].round(4)}  IK误差={err_mm:.2f}mm")
    print(f"\n新 _IK_SEED_JOINTS_RAD = {new_joints.tolist()}")


if __name__ == "__main__":
    main()
