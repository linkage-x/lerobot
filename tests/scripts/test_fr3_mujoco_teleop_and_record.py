#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from tools.fr3 import fr3_mujoco_ee2ee_record_smoke, fr3_mujoco_teleop
from tools.fr3 import fr3_mujoco_runtime


def test_parse_args_accepts_camera_viewer_and_resolution_flags():
    args = fr3_mujoco_teleop.parse_args(
        [
            "--viewer-camera",
            "external",
            "--enable-cameras",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--camera-fps",
            "30",
        ]
    )

    assert args.viewer_camera == "external"
    assert args.enable_cameras is True
    assert args.camera_width == 640
    assert args.camera_height == 480
    assert args.camera_fps == 30.0
    assert args.continuous_physics is True
    assert args.arm_actuator_kp == 20000.0
    assert args.arm_gravity_comp_scale == 0.5
    assert args.use_otg is False
    assert args.enable_rotation is True
    assert args.scale_wx == 0
    assert args.threshold_x == 0.02
    assert args.threshold_y == 0.02
    assert args.threshold_z == 0.02
    assert args.threshold_wx == 0.04
    assert args.threshold_wy == 0.04
    assert args.threshold_wz == 0.04


def test_parse_args_enable_otg_opt_in_overrides_default():
    args = fr3_mujoco_teleop.parse_args(["--enable-otg"])

    assert args.use_otg is True


def test_parse_args_disable_rotation_opt_out_overrides_default():
    args = fr3_mujoco_teleop.parse_args(["--disable-rotation"])

    assert args.enable_rotation is False


def test_build_env_config_uses_d435i_like_camera_defaults_when_enabled():
    args = fr3_mujoco_teleop.parse_args(["--enable-cameras", "--camera-width", "640", "--camera-height", "480"])

    cfg = fr3_mujoco_runtime.build_runtime_env_config(args)

    assert cfg.enable_cameras is True
    assert cfg.camera_width == 640
    assert cfg.camera_height == 480
    assert cfg.camera_names == ("external", "wrist")
    assert cfg.arm_actuator_kp == 20000.0
    assert cfg.arm_gravity_compensation_scale == 0.5
    assert cfg.use_otg is False
    assert cfg.continuous_physics is True
    assert cfg.continuous_physics_frequency == 800.0


def test_viewer_camera_name_normalization_accepts_none_and_named_camera():
    env_cfg = fr3_mujoco_runtime.build_runtime_env_config(fr3_mujoco_teleop.parse_args([]))

    assert fr3_mujoco_runtime.resolve_viewer_camera_name(None, env_cfg) is None
    assert fr3_mujoco_runtime.resolve_viewer_camera_name("external", env_cfg) == "external_cam"


def test_configure_mujoco_gl_backend_switches_to_glfw_for_viewer_plus_cameras(monkeypatch):
    args = fr3_mujoco_teleop.parse_args(["--enable-cameras"])
    monkeypatch.setenv("MUJOCO_GL", "egl")

    backend = fr3_mujoco_runtime.configure_mujoco_gl_backend(args)

    assert backend == "glfw"
    assert fr3_mujoco_runtime.os.environ["MUJOCO_GL"] == "glfw"


def test_configure_mujoco_gl_backend_keeps_existing_backend_without_viewer(monkeypatch):
    args = fr3_mujoco_teleop.parse_args(["--enable-cameras", "--no-viewer"])
    monkeypatch.setenv("MUJOCO_GL", "egl")

    backend = fr3_mujoco_teleop.configure_mujoco_gl_backend(args)

    assert backend == "egl"
    assert fr3_mujoco_teleop.os.environ["MUJOCO_GL"] == "egl"


def test_env_info_to_robot_observation_includes_camera_images():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
        "camera_obs": {
            "external": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist": np.full((480, 640, 3), 2, dtype=np.uint8),
        },
    }

    observation = fr3_mujoco_ee2ee_record_smoke.env_info_to_robot_observation(info, gripper_pos=1.0)

    assert observation["external"].shape == (480, 640, 3)
    assert observation["wrist"][0, 0, 0] == 2


def test_build_robot_observation_features_adds_visual_features_for_named_cameras():
    features = fr3_mujoco_ee2ee_record_smoke.build_robot_observation_features(include_cameras=True, camera_shape=(480, 640, 3))

    assert features["external"] == (480, 640, 3)
    assert features["wrist"] == (480, 640, 3)
    assert features["gripper.pos"] is float
