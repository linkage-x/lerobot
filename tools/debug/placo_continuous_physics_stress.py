#!/usr/bin/env python
"""Stress test: RobotKinematics + continuous_physics FK/IK loop.

Run in the FR3 sim container:
  docker run --rm \
    -e DISPLAY=$DISPLAY \
    -e PYTHONPATH=/workspace/src \
    -v /home/hanyu/Codes/lerobot:/workspace \
    lerobot-internal:local \
    python /workspace/tools/debug/placo_continuous_physics_stress.py --duration 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _default_urdf_path() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "src"
        / "lerobot"
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "fr3_pika_gripper_ati.urdf"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stress test RobotKinematics in continuous_physics MuJoCo env. "
            "Repeatedly calls FK/IK from the physics thread to trigger any "
            "destructor / glibc heap issues seen in the doc."
        )
    )
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--urdf-path", default=_default_urdf_path())
    parser.add_argument("--target-frame", default="pika_gripper_ee")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig

    joint_names = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3",
        "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]

    kin = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=joint_names,
    )

    def kin_forward(joints):
        return kin.forward_kinematics(np.rad2deg(joints))

    def kin_inverse(joints, pose):
        return np.deg2rad(kin.inverse_kinematics(np.rad2deg(joints), pose))

    cfg = FR3MujocoEnvConfig(
        continuous_physics=True,
        continuous_physics_frequency=200.0,
        max_episode_steps=1_000_000,
    )
    env = FR3MujocoEnv(cfg)
    env.reset()  # Starts continuous_physics thread internally

    joints_rad = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
    call_count = 0
    error_count = 0
    start = time.monotonic()
    last_report = start

    while time.monotonic() - start < args.duration:
        try:
            pose = kin_forward(joints_rad)
            # Small perturbation
            pose[0, 3] += 0.001
            pose[1, 3] -= 0.001
            solution = kin_inverse(joints_rad, pose)
            call_count += 1
        except Exception as e:
            error_count += 1
            print(f"[{time.monotonic() - start:.1f}s] FK/IK error: {e}", flush=True)

        now = time.monotonic()
        if now - last_report >= 10.0:
            elapsed = now - start
            print(f"[{elapsed:.0f}s] {call_count} calls, {error_count} errors", flush=True)
            last_report = now

    print(f"Done: {call_count} FK/IK calls, {error_count} errors in {args.duration}s")
    print("About to exit - if glibc heap error occurs it will print below this line")
    print("--- exit ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
