#!/usr/bin/env python

from pathlib import Path

import numpy as np

from lerobot.envs.fr3_mujoco import FR3MujocoEnv
from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig


def test_default_fr3_mujoco_urdf_path_exists():
    cfg = FR3MujocoEnvConfig()
    assert Path(cfg.urdf_path).is_file()


def test_local_envhub_wrapper_exists():
    wrapper_path = Path("sim/fr3_mujoco_env/env.py")
    assert wrapper_path.is_file()


def test_reset_info_exposes_target_and_tcp_marker_state():
    env = FR3MujocoEnv()
    try:
        _, info = env.reset()
        assert info["target_marker_name"] == "target"
        assert info["tcp_marker_name"] == "TCP"
        np.testing.assert_allclose(info["target_pose"], info["tcp_pose"])
        assert info["target_pose_7d"].shape == (7,)
        assert info["tcp_pose_7d"].shape == (7,)
    finally:
        env.close()


def test_first_disabled_teleop_action_holds_current_joint_state():
    env = FR3MujocoEnv()
    try:
        env.reset()
        before_joints = env._get_joint_positions()
        _, _, _, _, info = env.step_teleop_action({"enabled": False})
        np.testing.assert_allclose(info["joint_positions"], before_joints)
        np.testing.assert_allclose(info["target_pose"], info["tcp_pose"])
    finally:
        env.close()


def test_teleop_action_clips_target_to_workspace():
    cfg = FR3MujocoEnvConfig(
        workspace_min=(0.25, -0.1, 0.2),
        workspace_max=(0.3, 0.1, 0.25),
        max_target_delta_pos=(1.0, 1.0, 1.0),
    )
    env = FR3MujocoEnv(cfg=cfg)
    try:
        env.reset()
        _, _, _, _, info = env.step_teleop_action(
            {
                "enabled": True,
                "target_x": 10.0,
                "target_y": -10.0,
                "target_z": 10.0,
            }
        )
        np.testing.assert_allclose(
            info["target_pose"][:3, 3],
            np.array([0.3, -0.1, 0.25]),
            atol=1e-6,
        )
    finally:
        env.close()
