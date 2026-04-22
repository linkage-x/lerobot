#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pprint import pformat

from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, run_sim_teleop_loop
from lerobot.teleoperators.spacemouse.configuration_spacemouse import (
    SpaceMouseEnableButton,
    SpaceMouseTeleopConfig,
    SpaceMouseToolMode,
)
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

_D435I_COLOR_WIDTH = 640
_D435I_COLOR_HEIGHT = 480
_VIEWER_CAMERA_CHOICES = tuple(FR3MujocoEnvConfig().camera_names)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR3 MuJoCo teleoperation with SpaceMouse and marker viewer.")
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--viewer-camera", choices=_VIEWER_CAMERA_CHOICES, default=None)
    parser.add_argument("--enable-cameras", action="store_true", default=True)
    parser.add_argument("--camera-width", type=int, default=_D435I_COLOR_WIDTH)
    parser.add_argument("--camera-height", type=int, default=_D435I_COLOR_HEIGHT)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--arm-actuator-kp",
        type=float,
        default=20000.0,
        help="Override MuJoCo FR3 arm position actuator kp for teleop stability.",
    )
    parser.add_argument(
        "--arm-gravity-comp-scale",
        type=float,
        default=0.5,
        help="Scale factor for MuJoCo FR3 arm gravity compensation during teleop.",
    )
    parser.add_argument(
        "--disable-continuous-physics",
        dest="continuous_physics",
        action="store_false",
        help="Disable the background MuJoCo physics thread and only step physics during teleop actions.",
    )
    parser.add_argument(
        "--continuous-physics-frequency",
        type=float,
        default=800.0,
        help="Background MuJoCo stepping frequency in Hz when continuous physics is enabled.",
    )
    parser.set_defaults(continuous_physics=True)
    parser.add_argument(
        "--disable-otg",
        dest="use_otg",
        action="store_false",
        help="Disable OTG for MuJoCo teleop (default).",
    )
    parser.add_argument(
        "--enable-otg",
        dest="use_otg",
        action="store_true",
        help="Enable OTG for MuJoCo teleop.",
    )
    parser.set_defaults(use_otg=False)
    parser.add_argument("--tool-mode", choices=[mode.value for mode in SpaceMouseToolMode], default="incremental")
    parser.add_argument("--motion-enable-button", choices=[button.value for button in SpaceMouseEnableButton], default="none")
    parser.add_argument(
        "--disable-rotation",
        dest="enable_rotation",
        action="store_false",
        help="Disable end-effector rotation control (default).",
    )
    parser.add_argument(
        "--enable-rotation",
        dest="enable_rotation",
        action="store_true",
        help="Enable end-effector rotation control.",
    )
    parser.set_defaults(enable_rotation=True)
    parser.add_argument("--translation-scale", type=float, default=0.001845)
    parser.add_argument("--rotation-scale", type=float, default=0.001944)
    parser.add_argument("--scale-x", type=float, default=None)
    parser.add_argument("--scale-y", type=float, default=None)
    parser.add_argument("--scale-z", type=float, default=None)
    parser.add_argument(
        "--scale-wx",
        type=float,
        default=0, # if set, should be a negative value, such as -0.001944,
        help="Per-axis roll scale override. Defaults negative so SpaceMouse roll matches the FR3 sim TCP roll semantics.",
    )
    parser.add_argument("--scale-wy", type=float, default=0)
    parser.add_argument("--scale-wz", type=float, default=None)
    parser.add_argument("--threshold-x", type=float, default=0.02)
    parser.add_argument("--threshold-y", type=float, default=0.02)
    parser.add_argument("--threshold-z", type=float, default=0.02)
    parser.add_argument("--threshold-wx", type=float, default=0.04)
    parser.add_argument("--threshold-wy", type=float, default=0.04)
    parser.add_argument("--threshold-wz", type=float, default=0.04)
    parser.add_argument("--incremental-step", type=float, default=0.02)
    parser.add_argument("--move-time", type=float, default=0.006)
    parser.add_argument("--button-debounce-s", type=float, default=0.0)
    parser.add_argument("--button-release-grace-s", type=float, default=0.01)
    parser.add_argument("--gripper-cmd-min-delta", type=float, default=0.0)
    parser.add_argument("--gripper-cmd-min-interval-s", type=float, default=0.0)
    parser.add_argument("--gripper-cmd-ema-alpha", type=float, default=0.9)
    parser.add_argument("--gripper-cmd-max-rate", type=float, default=12.0)
    parser.add_argument("--sphere-radius", type=float, default=0.012)
    parser.add_argument("--axis-radius", type=float, default=0.003)
    parser.add_argument("--axis-length", type=float, default=0.06)
    return parser.parse_args(argv)


def build_teleop_config(args: argparse.Namespace) -> SpaceMouseTeleopConfig:
    return SpaceMouseTeleopConfig(
        device_id=args.device_id,
        frequency=args.fps,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        scale_z=args.scale_z,
        scale_wx=args.scale_wx,
        scale_wy=args.scale_wy,
        scale_wz=args.scale_wz,
        threshold_x=args.threshold_x,
        threshold_y=args.threshold_y,
        threshold_z=args.threshold_z,
        threshold_wx=args.threshold_wx,
        threshold_wy=args.threshold_wy,
        threshold_wz=args.threshold_wz,
        enable_rotation=args.enable_rotation,
        motion_enable_button=SpaceMouseEnableButton(args.motion_enable_button),
        tool_mode=SpaceMouseToolMode(args.tool_mode),
        incremental_step=args.incremental_step,
        move_time=args.move_time,
        button_debounce_s=args.button_debounce_s,
        button_release_grace_s=args.button_release_grace_s,
        gripper_cmd_min_delta=args.gripper_cmd_min_delta,
        gripper_cmd_min_interval_s=args.gripper_cmd_min_interval_s,
        gripper_cmd_ema_alpha=args.gripper_cmd_ema_alpha,
        gripper_cmd_max_rate=args.gripper_cmd_max_rate,
    )


def build_env_config(args: argparse.Namespace) -> FR3MujocoEnvConfig:
    max_episode_steps = 1_000_000
    if args.duration_s is not None:
        max_episode_steps = max(int(args.duration_s * args.fps) + 100, 1_000)
    return FR3MujocoEnvConfig(
        max_episode_steps=max_episode_steps,
        use_otg=bool(args.use_otg),
        arm_actuator_kp=float(args.arm_actuator_kp),
        arm_gravity_compensation_scale=float(args.arm_gravity_comp_scale),
        enable_cameras=bool(args.enable_cameras),
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        continuous_physics=bool(args.continuous_physics),
        continuous_physics_frequency=float(args.continuous_physics_frequency),
    )


def build_marker_style(args: argparse.Namespace) -> MarkerStyle:
    return MarkerStyle(
        sphere_radius=args.sphere_radius,
        axis_radius=args.axis_radius,
        axis_length=args.axis_length,
    )


def configure_mujoco_gl_backend(args: argparse.Namespace) -> str | None:
    current_backend = os.environ.get("MUJOCO_GL")
    if args.enable_cameras and not args.no_viewer:
        if current_backend is None or current_backend.lower() == "egl":
            os.environ["MUJOCO_GL"] = "glfw"
            return "glfw"
    return os.environ.get("MUJOCO_GL")


def resolve_viewer_camera_name(viewer_camera: str | None, env_cfg: FR3MujocoEnvConfig) -> str | None:
    if viewer_camera is None:
        return None
    return env_cfg.camera_name_mapping.get(viewer_camera, viewer_camera)


def configure_viewer_camera(mujoco, viewer, env: FR3MujocoEnv, viewer_camera: str | None) -> str | None:
    camera_name = resolve_viewer_camera_name(viewer_camera, env.cfg)
    if camera_name is None:
        return None
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Viewer camera '{viewer_camera}' resolved to missing MuJoCo camera '{camera_name}'.")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = int(camera_id)
    viewer.sync()
    return camera_name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mujoco_gl_backend = configure_mujoco_gl_backend(args)
    teleop = SpaceMouseTeleop(build_teleop_config(args))
    env_cfg = build_env_config(args)
    env = FR3MujocoEnv(env_cfg)
    viewer = None
    viewer_data = None

    print(
        pformat(
            {
                "fps": args.fps,
                "duration_s": args.duration_s,
                "viewer": not args.no_viewer,
                "viewer_camera": args.viewer_camera,
                "enable_cameras": args.enable_cameras,
                "camera_width": args.camera_width,
                "camera_height": args.camera_height,
                "camera_fps": args.camera_fps,
                "mujoco_gl": mujoco_gl_backend,
                "arm_actuator_kp": args.arm_actuator_kp,
                "arm_gravity_compensation_scale": args.arm_gravity_comp_scale,
                "use_otg": args.use_otg,
                "continuous_physics": args.continuous_physics,
                "continuous_physics_frequency": args.continuous_physics_frequency,
                "tool_mode": args.tool_mode,
                "motion_enable_button": args.motion_enable_button,
                "enable_rotation": args.enable_rotation,
            }
        )
    )

    try:
        teleop.connect()
        selected_camera_name = None
        if not args.no_viewer:
            import mujoco.viewer
            import mujoco

            viewer_data = mujoco.MjData(env.model)
            env.copy_visual_state(viewer_data)
            viewer = mujoco.viewer.launch_passive(env.model, viewer_data)
            selected_camera_name = configure_viewer_camera(mujoco, viewer, env, args.viewer_camera)
        info = run_sim_teleop_loop(
            env=env,
            teleop=teleop,
            fps=args.fps,
            viewer=viewer,
            viewer_data=viewer_data,
            duration_s=args.duration_s,
            marker_style=build_marker_style(args),
            render_cameras=args.enable_cameras,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )
        print("fr3_mujoco_teleop=READY")
        print(
            pformat(
                {
                    "loop_steps": info["loop_steps"],
                    "target_marker_name": info["target_marker_name"],
                    "tcp_marker_name": info["tcp_marker_name"],
                    "target_site_name": info["target_site_name"],
                    "tcp_site_name": info["tcp_site_name"],
                    "camera_names": info["camera_names"],
                    "viewer_camera": selected_camera_name,
                }
            )
        )
        return 0
    finally:
        if viewer is not None:
            viewer.close()
        try:
            teleop.disconnect()
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
