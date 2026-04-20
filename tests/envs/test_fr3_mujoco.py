#!/usr/bin/env python

from pathlib import Path

import numpy as np

from lerobot.envs.fr3_mujoco import FR3MujocoEnv
from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig


def test_default_fr3_mujoco_asset_paths_exist():
    cfg = FR3MujocoEnvConfig()
    assert Path(cfg.urdf_path).is_file()
    assert Path(cfg.sim_xml_path).is_file()


def test_local_envhub_wrapper_exists():
    wrapper_path = Path("sim/fr3_mujoco_env/env.py")
    assert wrapper_path.is_file()


def test_scene_mount_body_lifts_robot_base_to_table_height():
    env = FR3MujocoEnv()
    try:
        mount_body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, "mount")
        assert mount_body_id >= 0
        np.testing.assert_allclose(env.model.body_pos[mount_body_id], np.array([0.0, 0.0, 0.40]), atol=1e-9)
    finally:
        env.close()


def test_reset_info_exposes_target_tcp_marker_state_and_named_cameras():
    env = FR3MujocoEnv()
    try:
        _, info = env.reset()
        assert info["target_marker_name"] == "target"
        assert info["tcp_marker_name"] == "TCP"
        assert info["camera_names"] == ("third_person", "side", "wrist")
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
        np.testing.assert_allclose(info["target_joint_positions"], before_joints)
        assert np.linalg.norm(info["joint_positions"] - before_joints) < 1e-3
        assert info["otg_enabled"] is True
        assert info["otg_steps"] == 0
    finally:
        env.close()


def test_disabled_teleop_action_freezes_hold_target_without_otg_drift():
    env = FR3MujocoEnv()
    try:
        env.reset()
        hold_target = None
        settled_err = None

        for step in range(201):
            _, _, _, _, info = env.step_teleop_action({"enabled": False})
            target = np.asarray(info["target_joint_positions"], dtype=np.float64)
            actual = np.asarray(info["joint_positions"], dtype=np.float64)
            err = float(np.linalg.norm(actual - target))

            if hold_target is None:
                hold_target = target.copy()
            np.testing.assert_allclose(target, hold_target)
            assert info["otg_steps"] == 0

            if step == 50:
                settled_err = err

        assert settled_err is not None
        assert settled_err < 0.02
        assert abs(err - settled_err) < 1e-4
    finally:
        env.close()


def test_disabled_transition_freezes_measured_joints_not_stale_otg_target():
    env = FR3MujocoEnv()
    try:
        env.reset()
        for _ in range(120):
            env.step_teleop_action({"enabled": True, "target_y": 0.002, "target_z": -0.002})

        actual_before_disable = env._get_joint_positions().copy()
        otg_target_before_disable = env._otg_target_joints.copy()

        _, _, _, _, info = env.step_teleop_action({"enabled": False})
        hold_target = np.asarray(info["target_joint_positions"], dtype=np.float64)

        np.testing.assert_allclose(hold_target, actual_before_disable)
        assert np.linalg.norm(hold_target - otg_target_before_disable) > 1e-3
        assert info["otg_steps"] == 0
    finally:
        env.close()


def test_tiny_enabled_deltas_do_not_accumulate_reference_pose_drift():
    env = FR3MujocoEnv()
    try:
        env.reset()

        for _ in range(50):
            current_pose = env._current_tcp_pose().copy()
            _, _, _, _, info = env.step_teleop_action(
                {
                    "enabled": True,
                    "target_z": -5e-5,
                    "target_wz": 5e-5,
                }
            )

            np.testing.assert_allclose(info["target_pose"], current_pose)
            np.testing.assert_allclose(env._reference_pose, current_pose)
    finally:
        env.close()


def test_enabled_motion_reanchors_target_pose_to_measured_tcp_each_step():
    env = FR3MujocoEnv()
    try:
        env.reset()
        delta_z = -0.02

        _, _, _, _, first = env.step_teleop_action({"enabled": True, "target_z": delta_z})
        first_target = np.asarray(first["target_pose"], dtype=np.float64).copy()
        first_tcp = np.asarray(first["tcp_pose"], dtype=np.float64).copy()
        assert np.linalg.norm(first_target[:3, 3] - first_tcp[:3, 3]) > 1e-4

        _, _, _, _, second = env.step_teleop_action({"enabled": True, "target_z": delta_z})
        second_target = np.asarray(second["target_pose"], dtype=np.float64).copy()

        expected = first_tcp.copy()
        expected[2, 3] += delta_z
        expected[2, 3] = np.clip(expected[2, 3], env.cfg.workspace_min[2], env.cfg.workspace_max[2])

        np.testing.assert_allclose(second_target[:3, 3], expected[:3, 3], atol=1e-6)
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


def test_render_returns_named_camera_images_when_enabled():
    cfg = FR3MujocoEnvConfig(enable_cameras=True, camera_height=64, camera_width=64)
    env = FR3MujocoEnv(cfg=cfg)
    try:
        observation, info = env.reset()
        assert sorted(observation["camera_obs"].keys()) == ["side", "third_person", "wrist"]
        assert sorted(env.render().keys()) == ["side", "third_person", "wrist"]
        for image in observation["camera_obs"].values():
            assert image.shape == (64, 64, 3)
            assert image.dtype == np.uint8
        assert info["scene_geom_names"] == ("floor", "table", "workspace_object")
    finally:
        env.close()


def test_gripper_command_updates_pika_slide_joints_symmetrically():
    env = FR3MujocoEnv()
    try:
        env.reset()
        _, _, _, _, closed = env.step_teleop_action({"enabled": False, "gripper": 0.0})
        _, _, _, _, opened = env.step_teleop_action({"enabled": False, "gripper": 1.0})
        assert abs(opened["gripper_joint_positions"]["left"]) > abs(closed["gripper_joint_positions"]["left"])
        assert abs(opened["gripper_joint_positions"]["right"]) > abs(closed["gripper_joint_positions"]["right"])
        assert abs(opened["gripper_joint_positions"]["left"] + opened["gripper_joint_positions"]["right"]) < 1e-4
        assert opened["gripper_command"] == 1.0
    finally:
        env.close()


def test_gripper_fully_closes_without_object():
    env = FR3MujocoEnv()
    try:
        env.reset()
        env.step_teleop_action({"enabled": False, "gripper": 1.0})
        _, _, _, _, closed = env.step_teleop_action({"enabled": False, "gripper": 0.0})
        assert abs(closed["gripper_joint_positions"]["left"]) < 1e-3
        assert abs(closed["gripper_joint_positions"]["right"]) < 1e-3
    finally:
        env.close()


def test_closing_gripper_advances_object_between_fingers_via_contact_dynamics():
    env = FR3MujocoEnv()
    try:
        env.reset()
        object_joint_qposadr = int(env.model.jnt_qposadr[9])
        env.data.qpos[object_joint_qposadr : object_joint_qposadr + 3] = np.array([0.307, 0.0, 0.8015], dtype=np.float64)
        env.data.qpos[object_joint_qposadr + 3 : object_joint_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        env._mujoco.mj_forward(env.model, env.data)
        before = env.data.xpos[env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, "workspace_object_body")].copy()

        env.step_teleop_action({"enabled": False, "gripper": 1.0})
        _, _, _, _, info = env.step_teleop_action({"enabled": False, "gripper": 0.0})

        after = env.data.xpos[env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, "workspace_object_body")].copy()
        assert np.linalg.norm(after - before) > 1e-5
        assert info["gripper_command"] == 0.0
    finally:
        env.close()


def test_gripper_stops_lowering_when_pads_reach_table_height():
    env = FR3MujocoEnv()
    try:
        env.reset()
        current_pose = env._current_tcp_pose().copy()
        lowered_pose = current_pose.copy()
        lowered_pose[2, 3] = 0.46
        lowered_joint_positions = env._kinematics.inverse_kinematics(env._get_joint_positions(), lowered_pose)
        env._set_joint_state(lowered_joint_positions)
        left_pad_geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, "gripper_left_collision")
        right_pad_geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, "gripper_right_collision")
        left_pad_z = float(env.data.geom_xpos[left_pad_geom_id][2])
        right_pad_z = float(env.data.geom_xpos[right_pad_geom_id][2])
        assert left_pad_z > 0.42
        assert right_pad_z > 0.42
        assert env._current_tcp_pose()[2, 3] > 0.53
    finally:
        env.close()


def test_workspace_object_starts_resting_above_table_surface():
    env = FR3MujocoEnv()
    try:
        env.reset()
        object_body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, "workspace_object_body")
        object_pos = np.asarray(env.data.xpos[object_body_id], dtype=np.float64)
        table_top_z = 0.38 + 0.02
        object_half_height = 0.04
        assert object_pos[2] >= table_top_z + object_half_height - 1e-6
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
