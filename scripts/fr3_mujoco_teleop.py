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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR3 MuJoCo teleoperation with SpaceMouse and marker viewer.")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--tool-mode", choices=[mode.value for mode in SpaceMouseToolMode], default="binary")
    parser.add_argument("--motion-enable-button", choices=[button.value for button in SpaceMouseEnableButton], default="none")
    parser.add_argument("--enable-rotation", action="store_true")
    parser.add_argument("--scale-x", type=float, default=0.0006)
    parser.add_argument("--scale-y", type=float, default=0.0006)
    parser.add_argument("--scale-z", type=float, default=0.0006)
    parser.add_argument("--scale-wx", type=float, default=0.0004)
    parser.add_argument("--scale-wy", type=float, default=0.0004)
    parser.add_argument("--scale-wz", type=float, default=0.0004)
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
    return parser.parse_args()


def build_teleop_config(args: argparse.Namespace) -> SpaceMouseTeleopConfig:
    return SpaceMouseTeleopConfig(
        device_id=args.device_id,
        frequency=args.fps,
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
    return FR3MujocoEnvConfig(max_episode_steps=max_episode_steps)


def build_marker_style(args: argparse.Namespace) -> MarkerStyle:
    return MarkerStyle(
        sphere_radius=args.sphere_radius,
        axis_radius=args.axis_radius,
        axis_length=args.axis_length,
    )


def main() -> int:
    args = parse_args()
    teleop = SpaceMouseTeleop(build_teleop_config(args))
    env = FR3MujocoEnv(build_env_config(args))
    viewer = None

    print(
        pformat(
            {
                "fps": args.fps,
                "duration_s": args.duration_s,
                "viewer": not args.no_viewer,
                "tool_mode": args.tool_mode,
                "motion_enable_button": args.motion_enable_button,
                "enable_rotation": args.enable_rotation,
            }
        )
    )

    try:
        teleop.connect()
        if not args.no_viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        info = run_sim_teleop_loop(
            env=env,
            teleop=teleop,
            fps=args.fps,
            viewer=viewer,
            duration_s=args.duration_s,
            marker_style=build_marker_style(args),
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
