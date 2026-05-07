#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig


def test_gripper_command_updates_pika_slide_joints_symmetrically():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        _, _, _, _, closed = env.step_teleop_action(
            {"gripper": 0.0, "tracking_valid": True},
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        _, _, _, _, opened = env.step_teleop_action(
            {"gripper": 1.0, "tracking_valid": True},
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        assert abs(opened["gripper_joint_positions"]["left"]) > abs(closed["gripper_joint_positions"]["left"])
        assert abs(opened["gripper_joint_positions"]["right"]) > abs(closed["gripper_joint_positions"]["right"])
        assert abs(opened["gripper_joint_positions"]["left"] + opened["gripper_joint_positions"]["right"]) < 1e-4
        assert opened["gripper_command"] == 1.0
    finally:
        env.close()


def test_scene_uses_finite_actuators_instead_of_target_weld():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        missing_weld_id = env._mujoco.mj_name2id(
            env.model,
            env._mujoco.mjtObj.mjOBJ_EQUALITY,
            "target_to_gripper",
        )
        assert missing_weld_id < 0
        for actuator_name in (
            "gripper_base_x_actuator",
            "gripper_base_y_actuator",
            "gripper_base_z_actuator",
            "gripper_base_yaw_actuator",
            "gripper_base_pitch_actuator",
            "gripper_base_roll_actuator",
            "pika_gripper_actuator",
        ):
            actuator_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            assert actuator_id >= 0
            force_range = env.model.actuator_forcerange[actuator_id]
            assert np.isfinite(force_range).all()
            assert force_range[0] < 0.0 < force_range[1]
    finally:
        env.close()


def test_apply_tcp_pose_sets_actuator_targets_and_can_teleport_to_target():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        pose = env._initial_tcp_pose()
        pose[:3, 3] = np.array([0.52, 0.04, 0.58], dtype=np.float64)

        env._apply_tcp_pose(pose, teleport=True)

        np.testing.assert_allclose(env._current_tcp_pose()[:3, 3], pose[:3, 3], atol=1e-6)
        np.testing.assert_allclose(env._target_pose[:3, 3], pose[:3, 3], atol=1e-6)
    finally:
        env.close()
