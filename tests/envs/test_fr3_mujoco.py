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
        assert info["otg_enabled"] is True
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
        assert info["otg_steps"] == 4
        assert info["sender_steps"] == 5
    finally:
        env.close()


def test_fk_ik_round_trip_stays_in_target_frame():
    env = FR3MujocoEnv()
    try:
        env.reset()
        current_joints = env._get_joint_positions()
        current_pose = env._current_tcp_pose()
        ik_joints = env._kinematics.inverse_kinematics(current_joints, current_pose)
        round_trip_pose = env._kinematics.forward_kinematics(ik_joints)
        np.testing.assert_allclose(round_trip_pose, current_pose, atol=1e-5)
    finally:
        env.close()


def test_fk_ik_round_trip_converges_for_far_reachable_target():
    env = FR3MujocoEnv()
    try:
        env.reset()
        current_joints = env._get_joint_positions()
        target_joints = current_joints + np.array([0.35, -0.25, 0.30, -0.20, 0.15, -0.35, 0.25], dtype=np.float64)
        target_pose = env._kinematics.forward_kinematics(target_joints)
        ik_joints = env._kinematics.inverse_kinematics(current_joints, target_pose)
        round_trip_pose = env._kinematics.forward_kinematics(ik_joints)
        np.testing.assert_allclose(round_trip_pose, target_pose, atol=1e-4)
    finally:
        env.close()


def test_teleop_target_lags_tcp_under_otg_then_converges():
    cfg = FR3MujocoEnvConfig(
        max_target_delta_pos=(0.01, 0.01, 0.01),
        otg_max_velocity=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        otg_max_acceleration=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        otg_max_jerk=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    env = FR3MujocoEnv(cfg=cfg)
    try:
        env.reset()
        _, _, _, _, info = env.step_teleop_action(
            {
                "enabled": True,
                "target_x": 0.002,
                "target_y": -0.001,
                "target_z": 0.0015,
            }
        )
        initial_gap = np.linalg.norm(info["target_pose"][:3, 3] - info["tcp_pose"][:3, 3])
        assert initial_gap > 1e-5

        for _ in range(20):
            _, _, _, _, info = env.step_teleop_action({"enabled": False})

        final_gap = np.linalg.norm(info["target_pose"][:3, 3] - info["tcp_pose"][:3, 3])
        assert final_gap < initial_gap
    finally:
        env.close()


def test_disabling_teleop_stops_at_current_tcp_pose():
    cfg = FR3MujocoEnvConfig(
        max_target_delta_pos=(0.01, 0.01, 0.01),
        otg_max_velocity=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        otg_max_acceleration=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        otg_max_jerk=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    env = FR3MujocoEnv(cfg=cfg)
    try:
        env.reset()
        _, _, _, _, info = env.step_teleop_action(
            {
                "enabled": True,
                "target_x": 0.002,
                "target_y": 0.0,
                "target_z": 0.0,
            }
        )
        assert np.linalg.norm(info["target_pose"][:3, 3] - info["tcp_pose"][:3, 3]) > 1e-5

        _, _, _, _, info = env.step_teleop_action({"enabled": False})
        first_disabled_target = info["target_pose"].copy()
        assert np.linalg.norm(first_disabled_target[:3, 3] - info["tcp_pose"][:3, 3]) > 1e-5

        _, _, _, _, next_info = env.step_teleop_action({"enabled": False})
        np.testing.assert_allclose(next_info["target_pose"], first_disabled_target, atol=1e-6)
    finally:
        env.close()


def test_disabled_teleop_keeps_latched_hold_target_across_multiple_steps():
    cfg = FR3MujocoEnvConfig(
        max_target_delta_pos=(0.01, 0.01, 0.01),
        otg_max_velocity=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        otg_max_acceleration=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        otg_max_jerk=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    env = FR3MujocoEnv(cfg=cfg)
    try:
        env.reset()
        _, _, _, _, info = env.step_teleop_action(
            {
                "enabled": True,
                "target_x": 0.002,
                "target_y": 0.0,
                "target_z": 0.0,
            }
        )
        first_release_target = None
        for _ in range(2):
            _, _, _, _, info = env.step_teleop_action({"enabled": False})
            if first_release_target is None:
                first_release_target = info["target_joint_positions"].copy()
            else:
                np.testing.assert_allclose(info["target_joint_positions"], first_release_target)
    finally:
        env.close()
