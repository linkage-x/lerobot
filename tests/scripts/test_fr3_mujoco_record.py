#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from tools.fr3 import fr3_mujoco_record


def test_get_env_observation_includes_camera_images():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
        "camera_obs": {
            "third_person": np.zeros((480, 640, 3), dtype=np.uint8),
            "side": np.ones((480, 640, 3), dtype=np.uint8),
            "wrist": np.full((480, 640, 3), 2, dtype=np.uint8),
        },
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=1.0)

    assert observation["third_person"].shape == (480, 640, 3)
    assert observation["side"][0, 0, 0] == 1
    assert observation["wrist"][0, 0, 0] == 2


def test_get_env_observation_extracts_ee_pose_and_joints():
    ee_pose = np.eye(4, dtype=np.float64)
    ee_pose[0, 3] = 0.5
    ee_pose[1, 3] = 0.3
    ee_pose[2, 3] = 0.2
    info = {
        "ee_pose": ee_pose,
        "joint_positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64),
        "camera_obs": {},
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=0.5)

    assert observation["ee.x"] == 0.5
    assert observation["ee.y"] == 0.3
    assert observation["ee.z"] == 0.2
    assert observation["gripper.pos"] == 0.5
    for i in range(1, 8):
        assert np.isclose(observation[f"joint_{i}.pos"], float(i) * 0.1)


def test_get_env_observation_without_cameras():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=0.0)

    assert "third_person" not in observation
    assert "side" not in observation
    assert "wrist" not in observation
    assert observation["gripper.pos"] == 0.0


def test_complete_robot_observation_adds_prev_cmd_quats():
    observation = {
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
    }

    completed = fr3_mujoco_record._complete_robot_observation(observation)

    assert completed["prev_cmd.ee.qx"] == 0.0
    assert completed["prev_cmd.ee.qy"] == 0.0
    assert completed["prev_cmd.ee.qz"] == 0.0
    assert completed["prev_cmd.ee.qw"] == 1.0


def test_complete_robot_observation_preserves_existing_prev_cmd():
    observation = {
        "ee.qx": 0.1,
        "ee.qy": 0.2,
        "ee.qz": 0.3,
        "ee.qw": 0.4,
        "prev_cmd.ee.qx": 999.0,
    }

    completed = fr3_mujoco_record._complete_robot_observation(observation)

    assert completed["prev_cmd.ee.qx"] == 999.0
    assert completed["prev_cmd.ee.qy"] == 0.2
    assert completed["prev_cmd.ee.qz"] == 0.3
    assert completed["prev_cmd.ee.qw"] == 0.4


def test_build_robot_observation_features_includes_joints():
    features = fr3_mujoco_record._build_robot_observation_features(include_cameras=False)

    assert features["ee.x"] is float
    assert features["ee.y"] is float
    assert features["ee.z"] is float
    assert features["ee.wx"] is float
    assert features["ee.wy"] is float
    assert features["ee.wz"] is float
    assert features["gripper.pos"] is float
    for i in range(1, 8):
        assert features[f"joint_{i}.pos"] is float
    assert "third_person" not in features
    assert "side" not in features
    assert "wrist" not in features


def test_build_robot_observation_features_with_cameras():
    features = fr3_mujoco_record._build_robot_observation_features(
        include_cameras=True, camera_shape=(480, 640, 3)
    )

    assert features["third_person"] == (480, 640, 3)
    assert features["side"] == (480, 640, 3)
    assert features["wrist"] == (480, 640, 3)
    assert features["gripper.pos"] is float


def test_build_teleop_features_returns_expected_keys():
    features = fr3_mujoco_record._build_teleop_features()

    assert features["enabled"] is bool
    assert features["target_x"] is float
    assert features["target_y"] is float
    assert features["target_z"] is float
    assert features["target_wx"] is float
    assert features["target_wy"] is float
    assert features["target_wz"] is float
    assert features["gripper"] is float
    assert len(features) == 8
