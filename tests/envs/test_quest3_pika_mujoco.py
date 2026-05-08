#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig


def test_gripper_command_updates_pika_slide_joints_symmetrically():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        _, _, _, _, opened = env.step_teleop_action(
            {"gripper": 0.0, "tracking_valid": True},
            control_period_s=0.7,
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        _, _, _, _, closed = env.step_teleop_action(
            {"gripper": 1.0, "tracking_valid": True},
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        assert abs(closed["gripper_joint_positions"]["left"]) > abs(opened["gripper_joint_positions"]["left"])
        assert abs(closed["gripper_joint_positions"]["right"]) > abs(opened["gripper_joint_positions"]["right"])
        assert abs(closed["gripper_joint_positions"]["left"] + closed["gripper_joint_positions"]["right"]) < 1e-4
        assert closed["gripper_command"] == 1.0
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
        gripper_actuator_id = env._actuator_id("pika_gripper_actuator")
        np.testing.assert_allclose(env.model.actuator_forcerange[gripper_actuator_id], [-75.0, 75.0])
    finally:
        env.close()


def test_scene_uses_high_friction_box_proxy_finger_contacts():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        for geom_name in ("gripper_left_collision", "gripper_right_collision"):
            geom_id = env._geom_id(geom_name)
            assert env.model.geom_type[geom_id] == env._mujoco.mjtGeom.mjGEOM_BOX
            assert env.model.geom_condim[geom_id] == 6
            assert env.model.geom_friction[geom_id][0] >= 3.5
            assert env.model.geom_solref[geom_id][0] >= 0.004

        object_geom_id = env._geom_id("workspace_object")
        assert env.model.geom_condim[object_geom_id] == 4
        assert env.model.geom_friction[object_geom_id][0] <= 1.1
        assert env.model.geom_solref[object_geom_id][0] >= 0.006
    finally:
        env.close()


def test_table_has_realistic_wooden_table_inertia():
    env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
    try:
        table_body_id = env._body_id("table_body")
        assert 18.0 <= env.model.body_mass[table_body_id] <= 21.0
        np.testing.assert_allclose(
            env.model.body_inertia[table_body_id],
            np.array([2.7883, 2.4909, 1.6110], dtype=np.float64),
            rtol=1e-3,
            atol=1e-3,
        )
    finally:
        env.close()


def test_closed_gripper_can_lift_workspace_object_without_immediate_slip():
    env = Quest3PikaMujocoEnv(
        Quest3PikaMujocoEnvConfig(
            continuous_physics=False,
            enable_cameras=False,
            quest3_release_telemetry_log=False,
        )
    )
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        pose = env._initial_tcp_pose()
        pose[:3, 3] = np.array([0.48, 0.0, 0.42], dtype=np.float64)
        env._apply_tcp_pose(pose, teleport=True)

        qpos_adr = env._workspace_object_qpos_adr
        qvel_adr = env._workspace_object_qvel_adr
        env.data.qpos[qpos_adr : qpos_adr + 3] = np.array([0.48, 0.0, 0.51], dtype=np.float64)
        env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        env._last_gripper = 0.0
        env._set_gripper_command(0.0, teleport=True)

        for command in np.linspace(0.0, 1.0, 20):
            env._set_gripper_command(float(command))
            env._step_physics(10)
        for _ in range(200):
            env._step_physics(1)

        z_before = float(env.data.qpos[qpos_adr + 2])
        for target_z in np.linspace(0.42, 0.57, 80):
            pose[:3, 3] = np.array([0.48, 0.0, target_z], dtype=np.float64)
            env._apply_tcp_pose(pose)
            env._set_gripper_command(1.0)
            env._step_physics(4)
        z_after = float(env.data.qpos[qpos_adr + 2])

        assert z_after - z_before > 0.08
        assert any(
            "gripper" in str(contact["geom1"]) or "gripper" in str(contact["geom2"])
            for contact in env._workspace_object_contacts()
        )
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


def test_release_command_does_not_hidden_settle_object_to_table():
    env = Quest3PikaMujocoEnv(
        Quest3PikaMujocoEnvConfig(
            continuous_physics=False,
            enable_cameras=False,
            quest3_release_telemetry_log=False,
        )
    )
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        pose = env._initial_tcp_pose()
        pose[:3, 3] = np.array([0.95, 0.0, 0.70], dtype=np.float64)
        env._apply_tcp_pose(pose, teleport=True)

        qpos_adr = env._workspace_object_qpos_adr
        qvel_adr = env._workspace_object_qvel_adr
        env.data.qpos[qpos_adr : qpos_adr + 3] = np.array([0.48, 0.0, 0.70], dtype=np.float64)
        env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        env._last_gripper = 1.0
        env._set_gripper_command(1.0, teleport=True)
        env._mujoco.mj_forward(env.model, env.data)

        z_before = float(env.data.qpos[qpos_adr + 2])
        _, _, _, _, info = env.step_teleop_action(
            {"gripper": 0.0, "tracking_valid": True},
            control_period_s=env.model.opt.timestep,
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
        z_after = float(env.data.qpos[qpos_adr + 2])

        assert info["release_telemetry"]["release_event"] is True
        assert z_before - z_after < 0.001
    finally:
        env.close()
