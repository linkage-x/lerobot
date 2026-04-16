#!/usr/bin/env python

from __future__ import annotations

import argparse
from pprint import pformat

from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, run_sim_teleop_loop
from lerobot.teleoperators.spacemouse.configuration_spacemouse import (
    SpaceMouseEnableButton,
    SpaceMouseTeleopConfig,
    SpaceMouseToolMode,
)
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

_D435I_COLOR_FOVY_DEG = 42.0
_D435I_COLOR_WIDTH = 640
_D435I_COLOR_HEIGHT = 480


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR3 MuJoCo teleoperation with SpaceMouse and marker viewer.")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--viewer-camera", choices=("third_person", "side", "wrist"), default=None)
    parser.add_argument("--enable-cameras", action="store_true")
    parser.add_argument("--camera-width", type=int, default=_D435I_COLOR_WIDTH)
    parser.add_argument("--camera-height", type=int, default=_D435I_COLOR_HEIGHT)
    parser.add_argument("--tool-mode", choices=[mode.value for mode in SpaceMouseToolMode], default="binary")
    parser.add_argument("--motion-enable-button", choices=[button.value for button in SpaceMouseEnableButton], default="none")
    parser.add_argument("--enable-rotation", action="store_true")
    parser.add_argument("--translation-scale", type=float, default=0.000615)
    parser.add_argument("--rotation-scale", type=float, default=0.000648)
    parser.add_argument("--scale-x", type=float, default=None)
    parser.add_argument("--scale-y", type=float, default=None)
    parser.add_argument("--scale-z", type=float, default=None)
    parser.add_argument("--scale-wx", type=float, default=None)
    parser.add_argument("--scale-wy", type=float, default=None)
    parser.add_argument("--scale-wz", type=float, default=None)
    parser.add_argument("--threshold-x", type=float, default=0.02)
    parser.add_argument("--threshold-y", type=float, default=0.02)
    parser.add_argument("--threshold-z", type=float, default=0.02)
    parser.add_argument("--threshold-wx", type=float, default=0.04)
    parser.add_argument("--threshold-wy", type=float, default=0.04)
    parser.add_argument("--threshold-wz", type=float, default=0.04)
    parser.add_argument("--incremental-step", type=float, default=0.02)
    parser.add_argument("--move-time", type=float, default=0.006)
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
    )


def build_env_config(args: argparse.Namespace) -> FR3MujocoEnvConfig:
    max_episode_steps = 1_000_000
    if args.duration_s is not None:
        max_episode_steps = max(int(args.duration_s * args.fps) + 100, 1_000)
    return FR3MujocoEnvConfig(
        max_episode_steps=max_episode_steps,
        enable_cameras=bool(args.enable_cameras),
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        camera_fovy=float(_D435I_COLOR_FOVY_DEG),
    )


def build_marker_style(args: argparse.Namespace) -> MarkerStyle:
    return MarkerStyle(
        sphere_radius=args.sphere_radius,
        axis_radius=args.axis_radius,
        axis_length=args.axis_length,
    )


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
    teleop = SpaceMouseTeleop(build_teleop_config(args))
    env_cfg = build_env_config(args)
    env = FR3MujocoEnv(env_cfg)
    viewer = None

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
                "camera_fovy": _D435I_COLOR_FOVY_DEG,
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

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            selected_camera_name = configure_viewer_camera(mujoco, viewer, env, args.viewer_camera)
        info = run_sim_teleop_loop(
            env=env,
            teleop=teleop,
            fps=args.fps,
            viewer=viewer,
            duration_s=args.duration_s,
            marker_style=build_marker_style(args),
            render_cameras=args.enable_cameras,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
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
