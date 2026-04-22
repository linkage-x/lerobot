#!/usr/bin/env python

"""
FR3 MuJoCo teleoperation via keyboard.

Uses KeyboardEndEffectorTeleop to supply delta_x/y/z (末端位置增量),
adapts its action format to the SpaceMouse-style target_x/y/z/wx/y/z
format expected by FR3MujocoEnv.step_teleop_action(), and runs the
same sim-teleop loop as fr3_mujoco_teleop.py.

Key mapping (same as KeyboardEndEffectorTeleop):
  Arrow keys    delta_x / delta_y
  Shift/Shift_R delta_z (up/down)
  Ctrl_R        gripper open
  Ctrl_L        gripper close
  ESC           quit

Run:
  python fr3_mujoco_keyboard_teleop.py [--fps N] [--duration-s S] [--no-viewer]
         [--viewer-camera third_person|side|wrist] [--enable-cameras]
         [--camera-width W] [--camera-height H]
         [--disable-continuous-physics]
         [--continuous-physics-frequency HZ]
         [--enable-otg]
         [--delta-scale FLOAT]   # multiply keyboard deltas (default 0.01 m/key-step)
"""

from __future__ import annotations

import argparse
import sys
from pprint import pformat

# Add src to path for imports
sys.path.insert(0, "/workspace/src")

from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, run_sim_teleop_loop
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig


_D435I_COLOR_WIDTH = 640
_D435I_COLOR_HEIGHT = 480
_VIEWER_CAMERA_CHOICES = tuple(FR3MujocoEnvConfig().camera_names)


class KeyboardToSpacemouseAdapter:
    """
    Wraps KeyboardEndEffectorTeleop and translates its delta_x/y/z (+ gripper)
    output into the target_x/y/z/wx/y/z + enabled + gripper format expected
    by FR3MujocoEnv.step_teleop_action().

    SpaceMouse actions are in SI units (meters / rad).  KeyboardEndEffectorTeleop
    returns integer ±1 per key press, so we scale by --delta-scale.
    """

    def __init__(self, keyboard_teleop: KeyboardEndEffectorTeleop, delta_scale: float = 0.01):
        self._kbd = keyboard_teleop
        self._scale = delta_scale
        self._enabled = False  # tracks if any movement key is held

    @property
    def sync_gripper_baseline(self):
        return getattr(self._kbd, "sync_gripper_baseline", None)

    def get_action(self) -> dict:
        raw = self._kbd.get_action()
        dx = raw.get("delta_x", 0.0)
        dy = raw.get("delta_y", 0.0)
        dz = raw.get("delta_z", 0.0)
        gripper = raw.get("gripper", 1.0)

        # Any non-zero delta means the user is actively moving
        active = bool(dx or dy or dz)

        return {
            "enabled": active,
            "target_x": float(dx) * self._scale,
            "target_y": float(dy) * self._scale,
            "target_z": float(dz) * self._scale,
            # No rotation from keyboard
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": float(gripper),
        }

    def connect(self):
        self._kbd.connect()

    def disconnect(self):
        self._kbd.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._kbd.is_connected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FR3 MuJoCo teleoperation with keyboard (arrow keys = delta_x/y/z)."
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--viewer-camera",
        choices=_VIEWER_CAMERA_CHOICES,
        default=None,
    )
    parser.add_argument("--enable-cameras", action="store_true")
    parser.add_argument("--camera-width", type=int, default=_D435I_COLOR_WIDTH)
    parser.add_argument("--camera-height", type=int, default=_D435I_COLOR_HEIGHT)
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
        help="Disable the background MuJoCo physics thread.",
    )
    parser.add_argument(
        "--continuous-physics-frequency",
        type=float,
        default=800.0,
        help="Background MuJoCo stepping frequency in Hz.",
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
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=0.01,
        help="Keyboard delta multiplier in meters per key-step (default 0.01 = 1 cm).",
    )
    # Marker style
    parser.add_argument("--sphere-radius", type=float, default=0.012)
    parser.add_argument("--axis-radius", type=float, default=0.003)
    parser.add_argument("--axis-length", type=float, default=0.06)
    return parser.parse_args(argv)


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

    kbd_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)
    kbd_teleop = KeyboardEndEffectorTeleop(kbd_config)
    teleop = KeyboardToSpacemouseAdapter(kbd_teleop, delta_scale=args.delta_scale)

    env_cfg = build_env_config(args)
    env = FR3MujocoEnv(env_cfg)

    print(
        pformat(
            {
                "fps": args.fps,
                "duration_s": args.duration_s,
                "viewer": not args.no_viewer,
                "viewer_camera": args.viewer_camera,
                "enable_cameras": args.enable_cameras,
                "continuous_physics": args.continuous_physics,
                "continuous_physics_frequency": args.continuous_physics_frequency,
                "delta_scale (m/key-step)": args.delta_scale,
                "keyboard mapping": {
                    "Arrow keys": "delta_x / delta_y",
                    "Shift / Shift_R": "delta_z (up / down)",
                    "Ctrl_R": "gripper open",
                    "Ctrl_L": "gripper close",
                    "ESC": "quit",
                },
            }
        )
    )

    try:
        teleop.connect()
        selected_camera_name = None
        viewer = None
        viewer_data = None

        if not args.no_viewer:
            import mujoco.viewer
            import mujoco

            viewer_data = mujoco.MjData(env.model)
            env.copy_visual_state(viewer_data)
            viewer = mujoco.viewer.launch_passive(env.model, viewer_data)
            selected_camera_name = configure_viewer_camera(
                mujoco, viewer, env, args.viewer_camera
            )

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
        )

        print("fr3_mujoco_keyboard_teleop=READY")
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
