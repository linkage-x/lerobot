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


def test_scripted_contact_grasp_can_raise_workspace_object():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        pose = env._initial_tcp_pose()
        pose[:3, 3] = np.array([0.47, 0.0, 0.30], dtype=np.float64)
        env._apply_tcp_pose(pose)
        env._step_physics(120)
        env._set_gripper_command(0.0, simulate=True)

        for z in (0.32, 0.34, 0.36, 0.38):
            pose[:3, 3] = np.array([0.47, 0.0, z], dtype=np.float64)
            env._apply_tcp_pose(pose)
            env._step_physics(240)

        object_z = float(env.data.qpos[2])
        assert object_z > 0.46
    finally:
        env.close()
