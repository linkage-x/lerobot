#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle
from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.quest3.configuration_quest3 import (
    DEFAULT_QUEST3_CALIBRATION_DIR,
    DEFAULT_QUEST3_CERT_FILE,
    DEFAULT_QUEST3_KEY_FILE,
    Quest3GripperMapping,
    Quest3Hand,
    Quest3TeleopConfig,
)
from lerobot.teleoperators.spacemouse.configuration_spacemouse import (
    SpaceMouseEnableButton,
    SpaceMouseTeleopConfig,
    SpaceMouseToolMode,
)

_D435I_COLOR_WIDTH = 640
_D435I_COLOR_HEIGHT = 480
_VIEWER_CAMERA_CHOICES = tuple(FR3MujocoEnvConfig().camera_names)


def create_runtime_arg_parser(
    *,
    description: str,
    add_help: bool = True,
    include_duration: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    parser.add_argument("--fps", type=int, default=200)
    if include_duration:
        parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--teleop-type",
        choices=("spacemouse", "quest3"),
        default="spacemouse",
        help="Teleoperator type for argparse-only FR3 MuJoCo tools. fr3_mujoco_record.py also accepts draccus --teleop.type.",
    )
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
        default=0, #-0.001944,
        help="Per-axis roll scale override. Defaults negative so SpaceMouse roll matches the FR3 sim TCP roll semantics.",
    )
    parser.add_argument("--scale-wy", type=float, default=0)
    parser.add_argument("--scale-wz", type=float, default=0)
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
    parser.add_argument("--quest3-host", default="0.0.0.0")
    parser.add_argument("--quest3-port", type=int, default=8012)
    parser.add_argument("--quest3-cert-file", type=Path, default=DEFAULT_QUEST3_CERT_FILE)
    parser.add_argument("--quest3-key-file", type=Path, default=DEFAULT_QUEST3_KEY_FILE)
    parser.add_argument("--quest3-calibration-dir", type=Path, default=DEFAULT_QUEST3_CALIBRATION_DIR)
    parser.add_argument("--quest3-hand", choices=[hand.value for hand in Quest3Hand], default=Quest3Hand.RIGHT.value)
    parser.add_argument(
        "--quest3-gripper-mapping",
        choices=[mapping.value for mapping in Quest3GripperMapping],
        default=Quest3GripperMapping.PINCH_VALUE.value,
    )
    parser.add_argument("--quest3-open-pinch-value", type=float, default=0.111)
    parser.add_argument("--quest3-closed-pinch-value", type=float, default=0.004)
    parser.add_argument("--quest3-open-fingertip-distance-m", type=float, default=0.085)
    parser.add_argument("--quest3-closed-fingertip-distance-m", type=float, default=0.018)
    parser.add_argument("--quest3-translation-scale", type=float, default=1.0)
    parser.add_argument("--quest3-rotation-scale", type=float, default=1.0)
    parser.add_argument("--quest3-translation-deadband-m", type=float, default=0.002)
    parser.add_argument("--quest3-rotation-deadband-rad", type=float, default=0.02)
    parser.add_argument("--quest3-clutch-source", choices=("pinch", "squeeze", "always"), default="squeeze")
    parser.add_argument("--quest3-clutch-threshold", type=float, default=0.5)
    parser.add_argument("--quest3-lost-tracking-timeout-s", type=float, default=0.25)
    parser.add_argument(
        "--quest3-use-controller",
        dest="quest3_use_hand_tracking",
        action="store_false",
        help="Use Quest3 controllers (MotionControllers) instead of hand tracking. Right grip=clutch, triggers=gripper.",
    )
    parser.set_defaults(quest3_use_hand_tracking=True)
    parser.add_argument(
        "--quest3-scene-mode",
        choices=("pika_gripper", "fr3_arm"),
        default="pika_gripper",
        help="For Quest3 teleop, use a direct Pika gripper scene by default or the full FR3 arm scene.",
    )
    parser.add_argument(
        "--quest3-position-scale",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
        help="Scale Quest3 wrist xyz before mapping into the MuJoCo scene.",
    )
    parser.add_argument(
        "--quest3-position-offset",
        type=float,
        nargs=3,
        metavar=("OX", "OY", "OZ"),
        default=(0.0, 0.0, 0.0),
        help="Offset mapped Quest3 wrist xyz in the MuJoCo scene. In recenter mode this is relative to the initial Pika TCP.",
    )
    parser.add_argument(
        "--quest3-recenter-on-first-tracking",
        dest="quest3_recenter_on_first_tracking",
        action="store_true",
        help="Map the first valid Quest3 wrist pose to the initial Pika TCP, then use relative hand motion.",
    )
    parser.add_argument(
        "--quest3-absolute-origin",
        dest="quest3_recenter_on_first_tracking",
        action="store_false",
        help="Use absolute Quest3 wrist xyz with scale/offset instead of recentering on first tracking.",
    )
    parser.set_defaults(quest3_recenter_on_first_tracking=True)
    parser.add_argument(
        "--quest3-follow-orientation",
        dest="quest3_follow_orientation",
        action="store_true",
        help="Drive Pika TCP orientation from Quest3 wrist orientation.",
    )
    parser.add_argument(
        "--quest3-lock-orientation",
        dest="quest3_follow_orientation",
        action="store_false",
        help="Keep the Pika TCP orientation fixed while following Quest3 wrist position.",
    )
    parser.set_defaults(quest3_follow_orientation=True)
    parser.add_argument(
        "--quest3-rotation-alignment-xyzw",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        default=(0.0, 0.0, 0.0, 1.0),
        help="Additional xyzw quaternion alignment applied to Quest3 wrist orientation.",
    )
    parser.add_argument("--sphere-radius", type=float, default=0.012)
    parser.add_argument("--axis-radius", type=float, default=0.003)
    parser.add_argument("--axis-length", type=float, default=0.06)
    parser.add_argument("--quest3-debug-pose", action="store_true", help="Print Quest3 wrist to MuJoCo TCP mapping diagnostics.")
    parser.add_argument("--quest3-debug-pose-period-s", type=float, default=1.0)
    return parser


def parse_runtime_args(
    argv: list[str] | None = None,
    *,
    description: str,
    include_duration: bool = False,
) -> tuple[argparse.Namespace, list[str]]:
    parser = create_runtime_arg_parser(description=description, add_help=False, include_duration=include_duration)
    return parser.parse_known_args(argv)


def _set_frequency(config: TeleoperatorConfig, frequency: int) -> TeleoperatorConfig:
    if hasattr(config, "frequency"):
        setattr(config, "frequency", int(frequency))
    return config


def _apply_runtime_quest3_overrides(config: Quest3TeleopConfig, args: argparse.Namespace) -> Quest3TeleopConfig:
    config.host = args.quest3_host
    config.port = int(args.quest3_port)
    config.cert_file = args.quest3_cert_file
    config.key_file = args.quest3_key_file
    config.calibration_dir = args.quest3_calibration_dir
    config.hand = Quest3Hand(args.quest3_hand)
    config.use_hand_tracking = bool(args.quest3_use_hand_tracking)
    config.translation_scale = float(args.quest3_translation_scale)
    config.rotation_scale = float(args.quest3_rotation_scale)
    config.translation_deadband_m = float(args.quest3_translation_deadband_m)
    config.rotation_deadband_rad = float(args.quest3_rotation_deadband_rad)
    config.enable_rotation = bool(args.enable_rotation)
    config.clutch_source = str(args.quest3_clutch_source)
    config.clutch_threshold = float(args.quest3_clutch_threshold)
    config.gripper_mapping = Quest3GripperMapping(args.quest3_gripper_mapping)
    config.open_pinch_value = float(args.quest3_open_pinch_value)
    config.closed_pinch_value = float(args.quest3_closed_pinch_value)
    config.open_fingertip_distance_m = float(args.quest3_open_fingertip_distance_m)
    config.closed_fingertip_distance_m = float(args.quest3_closed_fingertip_distance_m)
    config.gripper_cmd_ema_alpha = float(args.gripper_cmd_ema_alpha)
    config.gripper_cmd_max_rate = float(args.gripper_cmd_max_rate)
    config.lost_tracking_timeout_s = float(args.quest3_lost_tracking_timeout_s)
    return config


def build_runtime_teleop_config(
    args: argparse.Namespace,
    *,
    frequency: int | None = None,
    base_config: TeleoperatorConfig | None = None,
) -> TeleoperatorConfig:
    resolved_frequency = args.fps if frequency is None else frequency
    if base_config is not None:
        teleop_type = getattr(base_config, "type", None)
        if teleop_type == "quest3":
            config = copy.deepcopy(base_config)
            _set_frequency(config, resolved_frequency)
            return _apply_runtime_quest3_overrides(config, args)

    if getattr(args, "teleop_type", "spacemouse") == "quest3":
        return Quest3TeleopConfig(
            host=args.quest3_host,
            port=args.quest3_port,
            cert_file=args.quest3_cert_file,
            key_file=args.quest3_key_file,
            calibration_dir=args.quest3_calibration_dir,
            hand=Quest3Hand(args.quest3_hand),
            use_hand_tracking=args.quest3_use_hand_tracking,
            frequency=resolved_frequency,
            translation_scale=args.quest3_translation_scale,
            rotation_scale=args.quest3_rotation_scale,
            translation_deadband_m=args.quest3_translation_deadband_m,
            rotation_deadband_rad=args.quest3_rotation_deadband_rad,
            enable_rotation=args.enable_rotation,
            clutch_source=args.quest3_clutch_source,
            clutch_threshold=args.quest3_clutch_threshold,
            gripper_mapping=Quest3GripperMapping(args.quest3_gripper_mapping),
            initial_gripper=1.0,
            open_pinch_value=args.quest3_open_pinch_value,
            closed_pinch_value=args.quest3_closed_pinch_value,
            open_fingertip_distance_m=args.quest3_open_fingertip_distance_m,
            closed_fingertip_distance_m=args.quest3_closed_fingertip_distance_m,
            gripper_cmd_ema_alpha=args.gripper_cmd_ema_alpha,
            gripper_cmd_max_rate=args.gripper_cmd_max_rate,
            lost_tracking_timeout_s=args.quest3_lost_tracking_timeout_s,
        )

    return SpaceMouseTeleopConfig(
        device_id=args.device_id,
        frequency=resolved_frequency,
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


def build_runtime_env_config(
    args: argparse.Namespace,
    *,
    max_episode_steps: int | None = None,
    control_frequency: int | None = None,
) -> FR3MujocoEnvConfig:
    resolved_control_frequency = args.fps if control_frequency is None else control_frequency
    resolved_max_episode_steps = max_episode_steps
    if resolved_max_episode_steps is None:
        duration_s = getattr(args, "duration_s", None)
        resolved_max_episode_steps = 1_000_000
        if duration_s is not None:
            resolved_max_episode_steps = max(int(duration_s * resolved_control_frequency) + 100, 1_000)
    return FR3MujocoEnvConfig(
        max_episode_steps=resolved_max_episode_steps,
        teleop_control_frequency=float(resolved_control_frequency),
        use_otg=bool(args.use_otg),
        arm_actuator_kp=float(args.arm_actuator_kp),
        arm_gravity_compensation_scale=float(args.arm_gravity_comp_scale),
        enable_cameras=bool(args.enable_cameras),
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        continuous_physics=bool(args.continuous_physics),
        continuous_physics_frequency=float(args.continuous_physics_frequency),
    )


def build_runtime_quest3_pika_env_config(
    args: argparse.Namespace,
    *,
    max_episode_steps: int | None = None,
    control_frequency: int | None = None,
) -> Quest3PikaMujocoEnvConfig:
    resolved_control_frequency = args.fps if control_frequency is None else control_frequency
    resolved_max_episode_steps = max_episode_steps
    if resolved_max_episode_steps is None:
        duration_s = getattr(args, "duration_s", None)
        resolved_max_episode_steps = 1_000_000
        if duration_s is not None:
            resolved_max_episode_steps = max(int(duration_s * resolved_control_frequency) + 100, 1_000)
    return Quest3PikaMujocoEnvConfig(
        max_episode_steps=resolved_max_episode_steps,
        teleop_control_frequency=float(resolved_control_frequency),
        enable_cameras=bool(args.enable_cameras),
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        continuous_physics=bool(args.continuous_physics),
        continuous_physics_frequency=float(args.continuous_physics_frequency),
        quest3_position_scale=tuple(float(v) for v in args.quest3_position_scale),
        quest3_position_offset=tuple(float(v) for v in args.quest3_position_offset),
        quest3_recenter_on_first_tracking=bool(args.quest3_recenter_on_first_tracking),
        quest3_follow_orientation=bool(args.quest3_follow_orientation),
        quest3_rotation_alignment_xyzw=tuple(float(v) for v in args.quest3_rotation_alignment_xyzw),
    )


def should_use_quest3_pika_env(args: argparse.Namespace, teleop_cfg: TeleoperatorConfig) -> bool:
    return getattr(teleop_cfg, "type", None) == "quest3" and getattr(args, "quest3_scene_mode", "pika_gripper") == "pika_gripper"


def build_runtime_env(args: argparse.Namespace, teleop_cfg: TeleoperatorConfig, **kwargs) -> FR3MujocoEnv | Quest3PikaMujocoEnv:
    if should_use_quest3_pika_env(args, teleop_cfg):
        return Quest3PikaMujocoEnv(build_runtime_quest3_pika_env_config(args, **kwargs))
    return FR3MujocoEnv(build_runtime_env_config(args, **kwargs))


def build_runtime_marker_style(args: argparse.Namespace) -> MarkerStyle:
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
