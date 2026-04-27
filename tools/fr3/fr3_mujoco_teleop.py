#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pprint import pformat

from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, run_sim_teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
try:
    from tools.fr3.fr3_mujoco_runtime import (
        build_runtime_env,
        build_runtime_marker_style,
        build_runtime_teleop_config,
        configure_mujoco_gl_backend,
        configure_viewer_camera,
        create_runtime_arg_parser,
    )
except ModuleNotFoundError:
    from fr3_mujoco_runtime import (
        build_runtime_env,
        build_runtime_marker_style,
        build_runtime_teleop_config,
        configure_mujoco_gl_backend,
        configure_viewer_camera,
        create_runtime_arg_parser,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_runtime_arg_parser(
        description="Run FR3 MuJoCo teleoperation with SpaceMouse or Quest3 and marker viewer.",
        include_duration=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mujoco_gl_backend = configure_mujoco_gl_backend(args)
    teleop_cfg = build_runtime_teleop_config(args)
    teleop = make_teleoperator_from_config(teleop_cfg)
    env = build_runtime_env(args, teleop_cfg)
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
                "camera_set": args.camera_set,
                "sim_xml_path": env.cfg.sim_xml_path,
                "camera_width": args.camera_width,
                "camera_height": args.camera_height,
                "camera_fps": args.camera_fps,
                "mujoco_gl": mujoco_gl_backend,
                "teleop_type": teleop_cfg.type,
                "scene_mode": getattr(env.cfg, "scene_mode", "fr3_arm"),
                "arm_actuator_kp": args.arm_actuator_kp,
                "arm_gravity_compensation_scale": args.arm_gravity_comp_scale,
                "use_otg": args.use_otg,
                "continuous_physics": args.continuous_physics,
                "continuous_physics_frequency": args.continuous_physics_frequency,
                "quest3_recenter_on_first_tracking": args.quest3_recenter_on_first_tracking,
                "quest3_follow_orientation": args.quest3_follow_orientation,
                "quest3_position_scale": args.quest3_position_scale,
                "quest3_position_offset": args.quest3_position_offset,
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
            marker_style=build_runtime_marker_style(args),
            render_cameras=args.enable_cameras,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            debug_pose=args.quest3_debug_pose,
            debug_pose_period_s=args.quest3_debug_pose_period_s,
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
