#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from tools.fr3 import fr3_mujoco_ee2ee_record_smoke, fr3_mujoco_teleop


def test_parse_args_accepts_camera_viewer_and_resolution_flags():
    args = fr3_mujoco_teleop.parse_args(
        [
            "--viewer-camera",
            "third_person",
            "--enable-cameras",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
        ]
    )

    assert args.viewer_camera == "third_person"
    assert args.enable_cameras is True
    assert args.camera_width == 640
    assert args.camera_height == 480
    assert args.continuous_physics is True


def test_build_env_config_uses_d435i_like_camera_defaults_when_enabled():
    args = fr3_mujoco_teleop.parse_args(["--enable-cameras", "--camera-width", "640", "--camera-height", "480"])

    cfg = fr3_mujoco_teleop.build_env_config(args)

    assert cfg.enable_cameras is True
    assert cfg.camera_width == 640
    assert cfg.camera_height == 480
    assert cfg.camera_names == ("third_person", "side", "wrist")
    assert cfg.camera_fovy == 42.0
    assert cfg.continuous_physics is True
    assert cfg.continuous_physics_frequency == 800.0


def test_viewer_camera_name_normalization_accepts_none_and_named_camera():
    env_cfg = fr3_mujoco_teleop.build_env_config(fr3_mujoco_teleop.parse_args([]))

    assert fr3_mujoco_teleop.resolve_viewer_camera_name(None, env_cfg) is None
    assert fr3_mujoco_teleop.resolve_viewer_camera_name("side", env_cfg) == "side_cam"


def test_env_info_to_robot_observation_includes_camera_images():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
        "camera_obs": {
            "third_person": np.zeros((480, 640, 3), dtype=np.uint8),
            "side": np.ones((480, 640, 3), dtype=np.uint8),
            "wrist": np.full((480, 640, 3), 2, dtype=np.uint8),
        },
    }

    observation = fr3_mujoco_ee2ee_record_smoke.env_info_to_robot_observation(info, gripper_pos=1.0)

    assert observation["third_person"].shape == (480, 640, 3)
    assert observation["side"][0, 0, 0] == 1
    assert observation["wrist"][0, 0, 0] == 2


def test_build_robot_observation_features_adds_visual_features_for_named_cameras():
    features = fr3_mujoco_ee2ee_record_smoke.build_robot_observation_features(include_cameras=True, camera_shape=(480, 640, 3))

    assert features["third_person"] == (480, 640, 3)
    assert features["side"] == (480, 640, 3)
    assert features["wrist"] == (480, 640, 3)
    assert features["gripper.pos"] is float
