#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pprint import pformat

import numpy as np

from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 MuJoCo environment smoke test.")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--async-envs", action="store_true")
    parser.add_argument(
        "--skip-teleop-probe",
        action="store_true",
        help="Skip a single relative-target teleop probe after the zero-action rollout.",
    )
    parser.add_argument(
        "--skip-egl-probe",
        action="store_true",
        help="Skip creating a MuJoCo EGL context before environment startup.",
    )
    return parser.parse_args()


def probe_egl_context() -> str:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    context = mujoco.GLContext(64, 64)
    try:
        return type(context).__name__
    finally:
        free = getattr(context, "free", None)
        if callable(free):
            free()


def main() -> int:
    args = parse_args()
    smoke_info = {}
    if not args.skip_egl_probe:
        smoke_info["mujoco_gl"] = os.environ.setdefault("MUJOCO_GL", "egl")
        smoke_info["egl_context"] = probe_egl_context()

    env = make_env(
        n_envs=args.n_envs,
        use_async_envs=args.async_envs,
        cfg=FR3MujocoEnvConfig(),
    )
    try:
        observation, info = env.reset()
        print("fr3_mujoco_env=READY")
        print(
            pformat(
                {
                    **smoke_info,
                    "obs_keys": sorted(observation.keys()),
                    "info_keys": sorted(info.keys()),
                }
            )
        )
        zero_action = np.zeros((args.n_envs, 7), dtype=np.float32)
        for step_idx in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(zero_action)
            print(
                pformat(
                    {
                        "step": step_idx,
                        "reward": np.asarray(reward).tolist(),
                        "terminated": np.asarray(terminated).tolist(),
                        "truncated": np.asarray(truncated).tolist(),
                    }
                )
            )
        if not args.skip_teleop_probe and not args.async_envs and args.n_envs == 1:
            observation, reward, terminated, truncated, info = env.envs[0].step_teleop_action(
                {
                    "enabled": True,
                    "target_x": 0.002,
                    "target_y": 0.0,
                    "target_z": 0.0,
                    "target_wx": 0.0,
                    "target_wy": 0.0,
                    "target_wz": 0.0,
                    "gripper": 1.0,
                }
            )
            print(
                pformat(
                    {
                        "teleop_probe": "READY",
                        "target_marker_name": info["target_marker_name"],
                        "tcp_marker_name": info["tcp_marker_name"],
                        "target_pose_7d": np.asarray(info["target_pose_7d"]).round(6).tolist(),
                        "tcp_pose_7d": np.asarray(info["tcp_pose_7d"]).round(6).tolist(),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
            )
        elif not args.skip_teleop_probe:
            print("teleop_probe=SKIPPED (requires SyncVectorEnv with n_envs=1)")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
