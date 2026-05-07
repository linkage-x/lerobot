#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

FIXED_6_CAMERA_NAMES = ("third_person", "north_east", "side", "west", "south_west", "south_east")
HIKON_8_CAMERA_NAMES = ("hk_01", "hk_02", "hk_03", "hk_04", "hk_05", "hk_06", "hk_07", "hk_08")
HIKON_BOX_SCENE_XML = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lerobot"
    / "robots"
    / "franka_research3"
    / "assets"
    / "franka_fr3"
    / "fr3_pika_ati_box_scene.xml"
)

STATE_VECTOR_NAMES = [
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.qx",
    "ee.qy",
    "ee.qz",
    "ee.qw",
    "gripper.pos",
]
JOINT_VECTOR_NAMES = [f"joint_{i}.pos" for i in range(1, 8)]


def _configure_mujoco_gl_backend(requested_backend: str | None, *, viewer: bool) -> str | None:
    if requested_backend is not None:
        os.environ["MUJOCO_GL"] = requested_backend
    elif viewer:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    return os.environ.get("MUJOCO_GL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_for_explicit = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description=(
            "Generate FR3 MuJoCo moving-ChArUco calibration data in LeRobot format. "
            "This is a simulation-only data-path validator: no teleoperation and no physical camera setup."
        )
    )
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path. CLI values override config values.")
    parser.add_argument("--repo-id", default="local/fr3_mujoco_calibration")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/fr3_mujoco_calibration"))
    parser.add_argument("--task", default="FR3 MuJoCo calibration")
    parser.add_argument("--overwrite", action="store_true", help="Remove --root before creating the dataset.")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--dataset-fps", type=int, default=10)
    parser.add_argument("--control-frequency", type=float, default=20.0)
    parser.add_argument("--max-command-steps", type=int, default=160)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--joint-delta-rad", type=float, default=0.20)
    parser.add_argument("--joint-margin-rad", type=float, default=0.08)
    parser.add_argument(
        "--sample-mode",
        choices=("random_walk", "around_initial", "ergodic_xyz", "ergodic_6d"),
        default="random_walk",
        help=(
            "random_walk samples next target around current joints; around_initial samples around reset joints; "
            "ergodic_xyz/ergodic_6d generate Cartesian targets from SMC ergodic control."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-set", choices=("fixed_6", "default", "hikon_8"), default="fixed_6")
    parser.add_argument(
        "--camera-names",
        default="",
        help="Comma-separated logical MuJoCo camera names. Overrides --camera-set.",
    )
    parser.add_argument("--sim-xml-path", type=Path, default=None)
    parser.add_argument("--arm-actuator-kp", type=float, default=20000.0)
    parser.add_argument("--arm-gravity-comp-scale", type=float, default=0.5)
    parser.add_argument("--enable-otg", "--use-otg", dest="enable_otg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controller-damping", type=float, nargs="+", default=None)
    parser.add_argument("--controller-stiffness", type=float, nargs="+", default=None)
    parser.add_argument("--controller-filter-coeff", type=float, default=None)
    parser.add_argument("--otg-control-frequency", type=float, default=800.0)
    parser.add_argument("--otg-async-control-frequency", type=float, default=1000.0)
    parser.add_argument("--otg-synchronization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--otg-sync-mode", default="time")
    parser.add_argument("--otg-max-velocity", type=float, nargs="+", default=(2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208))
    parser.add_argument("--otg-max-acceleration", type=float, nargs="+", default=(8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0))
    parser.add_argument(
        "--otg-max-jerk",
        type=float,
        nargs="+",
        default=(4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0),
    )
    parser.add_argument(
        "--motion-mode",
        choices=("auto", "teleport", "servo", "cartesian_impedance"),
        default="auto",
        help=(
            "auto uses teleport for sample-mode=random_walk, servo for joint-space modes, "
            "and cartesian_impedance for ergodic Cartesian modes."
        ),
    )
    parser.add_argument("--ergodic-low", type=float, nargs="+", default=None)
    parser.add_argument("--ergodic-high", type=float, nargs="+", default=None)
    parser.add_argument("--ergodic-x0", type=float, nargs="+", default=None)
    parser.add_argument("--ergodic-num-k-per-dim", type=int, default=4)
    parser.add_argument("--ergodic-dt", type=float, default=0.05)
    parser.add_argument("--ergodic-speed", type=float, default=0.04)
    parser.add_argument("--ergodic-boundary", choices=("reflect", "clip", "none"), default="reflect")
    parser.add_argument(
        "--ergodic-position-x0-offset-m",
        type=float,
        nargs=3,
        default=(0.055, -0.045, -0.035),
        help=(
            "Default xyz offset from the initial TCP pose when --ergodic-x0 is null. "
            "A nonzero offset breaks center-start symmetry in the uniform-box ergodic field."
        ),
    )
    parser.add_argument(
        "--ergodic-rotation-x0-rad",
        type=float,
        nargs=3,
        default=(0.08, -0.06, 0.05),
        help=(
            "Default initial local rotvec offset for ergodic_6d when --ergodic-x0 is null. "
            "A nonzero value breaks the uniform-box symmetry that otherwise keeps rx/ry/rz at zero."
        ),
    )
    parser.add_argument(
        "--ergodic-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for SMC ergodic trajectory generation. auto uses CUDA when torch reports it available.",
    )
    parser.add_argument(
        "--ergodic-plan-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate and save the ergodic trajectory visualization/report, then exit before tracking/capture.",
    )
    parser.add_argument(
        "--pause-after-ergodic-plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause after saving the ergodic trajectory preview and before tracking/capture.",
    )
    parser.add_argument("--cartesian-position-tolerance-m", type=float, default=0.004)
    parser.add_argument("--cartesian-orientation-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--cartesian-max-position-step-m", type=float, default=0.015)
    parser.add_argument("--cartesian-max-rotation-step-rad", type=float, default=0.08)
    parser.add_argument("--cartesian-orientation-weight", type=float, default=1.0)
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open a passive MuJoCo viewer during capture. Use --no-viewer to override a config that enables it.",
    )
    parser.add_argument(
        "--viewer-camera",
        default="",
        help="Optional MuJoCo camera to show in the viewer, e.g. third_person_cam or hk_01_cam.",
    )
    parser.add_argument(
        "--viewer-hold-s",
        type=float,
        default=0.05,
        help="Seconds to keep each captured pose visible in the viewer.",
    )
    parser.add_argument(
        "--viewer-pause-every",
        type=int,
        default=0,
        help="If >0, pause for Enter every N samples after updating the viewer.",
    )
    parser.add_argument(
        "--viewer-final-hold-s",
        type=float,
        default=0.0,
        help="Seconds to keep the final viewer open before exiting.",
    )
    parser.add_argument("--continuous-physics", action="store_true")
    parser.add_argument("--continuous-physics-frequency", type=float, default=800.0)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--streaming-encoding", action="store_true", default=True)
    parser.add_argument("--no-streaming-encoding", dest="streaming_encoding", action="store_false")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--mujoco-gl",
        choices=("glfw", "egl", "osmesa"),
        default=None,
        help="Set MUJOCO_GL before creating renderers. Defaults to glfw with --viewer, otherwise egl.",
    )
    args = parser.parse_args(argv)
    explicit_dests = _explicit_cli_dests(argv_for_explicit)
    if args.config_path is not None:
        _apply_config_defaults(args, args.config_path, argv_for_explicit)
    args._explicit_cli_dests = explicit_dests
    return args


def _explicit_cli_dests(argv: list[str]) -> set[str]:
    dests: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token[2:].split("=", 1)[0]
        if not option:
            continue
        dests.add(option.replace("-", "_"))
        if option.startswith("no-"):
            dests.add(option[3:].replace("-", "_"))
    if "use_otg" in dests:
        dests.add("enable_otg")
    if "no_use_otg" in dests:
        dests.add("enable_otg")
    return dests


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Loading --config-path requires PyYAML in the runtime environment.") from exc
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _flatten_calibration_config(cfg: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    sections = {
        "dataset": (
            "repo_id",
            "root",
            "task",
            "overwrite",
            "num_samples",
            "dataset_fps",
        ),
        "sampling": (
            "sample_mode",
            "seed",
            "joint_delta_rad",
            "joint_margin_rad",
        ),
        "motion": (
            "motion_mode",
            "control_frequency",
            "max_command_steps",
            "settle_steps",
            "joint_tolerance_rad",
            "enable_otg",
            "arm_actuator_kp",
            "arm_gravity_comp_scale",
            "continuous_physics",
            "continuous_physics_frequency",
        ),
        "controller": (
            "use_otg",
            "enable_otg",
            "damping",
            "stiffness",
            "filter_coeff",
            "controller_damping",
            "controller_stiffness",
            "controller_filter_coeff",
            "otg_control_frequency",
            "otg_async_control_frequency",
            "otg_synchronization",
            "otg_sync_mode",
            "otg_max_velocity",
            "otg_max_acceleration",
            "otg_max_jerk",
        ),
        "camera": (
            "camera_width",
            "camera_height",
            "camera_set",
            "camera_names",
            "sim_xml_path",
        ),
        "viewer": (
            "viewer",
            "viewer_camera",
            "viewer_hold_s",
            "viewer_pause_every",
            "viewer_final_hold_s",
        ),
        "encoding": (
            "vcodec",
            "streaming_encoding",
            "encoder_threads",
            "encoder_queue_maxsize",
        ),
        "ergodic": (
            "ergodic_low",
            "ergodic_high",
            "ergodic_x0",
            "ergodic_num_k_per_dim",
            "ergodic_dt",
            "ergodic_speed",
            "ergodic_boundary",
            "ergodic_position_x0_offset_m",
            "ergodic_rotation_x0_rad",
            "ergodic_device",
            "ergodic_plan_only",
            "pause_after_ergodic_plan",
        ),
        "cartesian_impedance": (
            "cartesian_position_tolerance_m",
            "cartesian_orientation_tolerance_rad",
            "cartesian_max_position_step_m",
            "cartesian_max_rotation_step_rad",
            "cartesian_orientation_weight",
        ),
        "replay": (
            "command_interval_s",
            "max_command_steps_for_initial_pose",
            "max_translation_step_m",
            "max_rotation_step_deg",
            "position_tolerance_m",
            "orientation_tolerance_deg",
        ),
        "output": ("report_json",),
        "runtime": ("mujoco_gl",),
    }
    for section, keys in sections.items():
        payload = cfg.get(section, {})
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if key in payload:
                if key == "use_otg":
                    flat["enable_otg"] = payload[key]
                elif key == "damping":
                    flat["controller_damping"] = payload[key]
                elif key == "stiffness":
                    flat["controller_stiffness"] = payload[key]
                elif key == "filter_coeff":
                    flat["controller_filter_coeff"] = payload[key]
                elif key == "command_interval_s":
                    flat["control_frequency"] = 1.0 / float(payload[key])
                elif key == "max_command_steps_for_initial_pose":
                    flat["max_command_steps"] = payload[key]
                elif key == "max_translation_step_m":
                    flat["cartesian_max_position_step_m"] = payload[key]
                elif key == "max_rotation_step_deg":
                    flat["cartesian_max_rotation_step_rad"] = np.deg2rad(float(payload[key]))
                elif key == "position_tolerance_m":
                    flat["cartesian_position_tolerance_m"] = payload[key]
                elif key == "orientation_tolerance_deg":
                    flat["cartesian_orientation_tolerance_rad"] = np.deg2rad(float(payload[key]))
                else:
                    flat[key] = payload[key]
    for key, value in cfg.items():
        if key in sections:
            continue
        flat[key] = value
    return flat


def _apply_config_defaults(args: argparse.Namespace, config_path: Path, argv: list[str]) -> None:
    cfg = _load_yaml_config(config_path)
    if "calibration" in cfg and isinstance(cfg["calibration"], dict):
        cfg = cfg["calibration"]
    flat = _flatten_calibration_config(cfg)
    explicit = _explicit_cli_dests(argv)
    for key, value in flat.items():
        if key in explicit or not hasattr(args, key):
            continue
        if key in {"root", "sim_xml_path", "report_json"} and value is not None:
            value = Path(str(value))
        setattr(args, key, value)


def _resolve_motion_mode(args: argparse.Namespace) -> str:
    if args.motion_mode != "auto":
        return str(args.motion_mode)
    if args.sample_mode == "random_walk":
        return "teleport"
    if str(args.sample_mode).startswith("ergodic_"):
        return "cartesian_impedance"
    return "servo"


def _resolve_camera_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.camera_names.strip():
        names = tuple(name.strip() for name in args.camera_names.split(",") if name.strip())
        if not names:
            raise ValueError("--camera-names was provided but no valid camera names were parsed.")
        return names
    if args.camera_set == "fixed_6":
        return FIXED_6_CAMERA_NAMES
    if args.camera_set == "hikon_8":
        return HIKON_8_CAMERA_NAMES
    from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig

    return tuple(FR3MujocoEnvConfig().camera_names)


def _as_float_tuple(value: Any, *, name: str, length: int = 7) -> tuple[float, ...]:
    values = tuple(float(v) for v in value)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(values)}.")
    return values


def _build_env_config(args: argparse.Namespace) -> FR3MujocoEnvConfig:
    from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig

    camera_names = _resolve_camera_names(args)
    camera_name_mapping = dict(FR3MujocoEnvConfig().camera_name_mapping)
    sim_xml_path = args.sim_xml_path
    if args.camera_set == "hikon_8" and not args.camera_names.strip() and sim_xml_path is None:
        sim_xml_path = HIKON_BOX_SCENE_XML
        camera_name_mapping.update({name: f"{name}_cam" for name in HIKON_8_CAMERA_NAMES})

    return FR3MujocoEnvConfig(
        sim_xml_path=str(sim_xml_path.expanduser()) if sim_xml_path is not None else FR3MujocoEnvConfig().sim_xml_path,
        camera_names=camera_names,
        camera_name_mapping=camera_name_mapping,
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        # Keep env.step() lightweight. This tool renders cameras explicitly only
        # at capture samples, so control steps should not build camera_obs.
        enable_cameras=False,
        max_episode_steps=max(int(args.num_samples) * (int(args.max_command_steps) + int(args.settle_steps) + 2), 1000),
        teleop_control_frequency=float(args.control_frequency),
        use_otg=bool(args.enable_otg),
        otg_control_frequency=float(args.otg_control_frequency),
        otg_async_control_frequency=float(args.otg_async_control_frequency),
        otg_synchronization=bool(args.otg_synchronization),
        otg_sync_mode=str(args.otg_sync_mode),
        otg_max_velocity=_as_float_tuple(args.otg_max_velocity, name="otg_max_velocity"),
        otg_max_acceleration=_as_float_tuple(args.otg_max_acceleration, name="otg_max_acceleration"),
        otg_max_jerk=_as_float_tuple(args.otg_max_jerk, name="otg_max_jerk"),
        arm_actuator_kp=float(args.arm_actuator_kp),
        arm_gravity_compensation_scale=float(args.arm_gravity_comp_scale),
        continuous_physics=bool(args.continuous_physics),
        continuous_physics_frequency=float(args.continuous_physics_frequency),
    )


def _controller_summary(args: argparse.Namespace, env: FR3MujocoEnv) -> dict[str, Any]:
    return {
        "use_otg": bool(args.enable_otg),
        "damping": None if args.controller_damping is None else [float(v) for v in args.controller_damping],
        "stiffness": None if args.controller_stiffness is None else [float(v) for v in args.controller_stiffness],
        "filter_coeff": None if args.controller_filter_coeff is None else float(args.controller_filter_coeff),
        "command_interval_s": float(env.cfg.teleop_dt),
        "otg_control_frequency": float(args.otg_control_frequency),
        "otg_async_control_frequency": float(args.otg_async_control_frequency),
        "otg_synchronization": bool(args.otg_synchronization),
        "otg_sync_mode": str(args.otg_sync_mode),
        "otg_max_velocity": [float(v) for v in _as_float_tuple(args.otg_max_velocity, name="otg_max_velocity")],
        "otg_max_acceleration": [
            float(v) for v in _as_float_tuple(args.otg_max_acceleration, name="otg_max_acceleration")
        ],
        "otg_max_jerk": [float(v) for v in _as_float_tuple(args.otg_max_jerk, name="otg_max_jerk")],
        "arm_actuator_kp": float(args.arm_actuator_kp),
        "arm_gravity_comp_scale": float(args.arm_gravity_comp_scale),
    }


def _build_dataset_features(camera_names: tuple[str, ...], *, height: int, width: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_VECTOR_NAMES),),
            "names": STATE_VECTOR_NAMES,
        },
        "observation.joints": {
            "dtype": "float32",
            "shape": (len(JOINT_VECTOR_NAMES),),
            "names": JOINT_VECTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_VECTOR_NAMES),),
            "names": STATE_VECTOR_NAMES,
        },
        "observation.device_capture_timestamp": {
            "dtype": "float64",
            "shape": (1 + len(camera_names),),
            "names": ["robot.ee.capture_timestamp_s"]
            + [f"camera.{camera_name}.capture_timestamp_s" for camera_name in camera_names],
        },
    }
    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _pose_to_state_vector(ee_pose: np.ndarray, gripper_pos: float) -> np.ndarray:
    from lerobot.utils.rotation import Rotation

    quat_xyzw = Rotation.from_matrix(np.asarray(ee_pose[:3, :3], dtype=np.float64)).as_quat()
    return np.asarray(
        [
            float(ee_pose[0, 3]),
            float(ee_pose[1, 3]),
            float(ee_pose[2, 3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
            float(gripper_pos),
        ],
        dtype=np.float32,
    )


def _sample_joint_target(
    env: FR3MujocoEnv,
    rng: np.random.Generator,
    *,
    joint_delta_rad: float,
    joint_margin_rad: float,
    sample_mode: str,
) -> np.ndarray:
    if sample_mode == "random_walk":
        center = np.asarray(env._get_joint_positions(), dtype=np.float64)
    else:
        center = np.asarray(env._initial_joint_positions, dtype=np.float64)
    lower = np.maximum(np.asarray(env._joint_lower, dtype=np.float64) + float(joint_margin_rad), center - float(joint_delta_rad))
    upper = np.minimum(np.asarray(env._joint_upper, dtype=np.float64) - float(joint_margin_rad), center + float(joint_delta_rad))
    bad = lower >= upper
    if np.any(bad):
        lower[bad] = np.asarray(env._joint_lower, dtype=np.float64)[bad]
        upper[bad] = np.asarray(env._joint_upper, dtype=np.float64)[bad]
    return rng.uniform(lower, upper).astype(np.float64)


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= float(max_norm) or norm <= 1e-12:
        return vec
    return vec * (float(max_norm) / norm)


def _default_ergodic_bounds(
    env: FR3MujocoEnv,
    dim: int,
    *,
    position_x0_offset_m: np.ndarray | None = None,
    rotation_x0_rad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_pose = np.asarray(env._current_tcp_pose(), dtype=np.float64)
    xyz = current_pose[:3, 3]
    low_xyz = xyz + np.array([-0.14, -0.16, -0.08], dtype=np.float64)
    high_xyz = xyz + np.array([0.14, 0.16, 0.10], dtype=np.float64)
    workspace_min = np.asarray(env.cfg.workspace_min, dtype=np.float64)
    workspace_max = np.asarray(env.cfg.workspace_max, dtype=np.float64)
    low_xyz = np.maximum(low_xyz, workspace_min)
    high_xyz = np.minimum(high_xyz, workspace_max)
    pos_offset = np.asarray(
        [0.055, -0.045, -0.035] if position_x0_offset_m is None else position_x0_offset_m,
        dtype=np.float64,
    ).reshape(3)
    x0_xyz = np.clip(xyz + pos_offset, low_xyz, high_xyz)
    if dim == 3:
        return low_xyz, high_xyz, x0_xyz
    if dim == 6:
        low = np.concatenate((low_xyz, np.full(3, -0.35, dtype=np.float64)))
        high = np.concatenate((high_xyz, np.full(3, 0.35, dtype=np.float64)))
        rot_x0 = np.asarray(
            [0.08, -0.06, 0.05] if rotation_x0_rad is None else rotation_x0_rad,
            dtype=np.float64,
        ).reshape(3)
        rot_x0 = np.clip(rot_x0, low[3:6], high[3:6])
        x0 = np.concatenate((x0_xyz, rot_x0))
        return low, high, x0
    raise ValueError(f"Unsupported ergodic dim: {dim}")


def _coerce_ergodic_vector(value: Any, *, dim: int, name: str, default: np.ndarray) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64).reshape(dim)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(dim, float(arr[0]), dtype=np.float64)
    if arr.size != dim:
        raise ValueError(f"{name} must have either 1 or {dim} values, got {arr.size}.")
    return arr


def _resolve_ergodic_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        if requested == "cuda":
            raise RuntimeError("--ergodic-device=cuda requires PyTorch with CUDA support.")
        return "cpu"
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--ergodic-device=cuda was requested, but torch.cuda.is_available() is false.")
    return "cuda" if cuda_available else "cpu"


def _build_ergodic_pose_targets(env: FR3MujocoEnv, args: argparse.Namespace) -> tuple[list[np.ndarray], dict[str, Any]]:
    from lerobot.utils.rotation import Rotation

    try:
        from tools.fr3.ergodic.smc import SMCErgodicConfig, run_uniform_box_smc
    except ModuleNotFoundError:
        from ergodic.smc import SMCErgodicConfig, run_uniform_box_smc

    dim = 3 if args.sample_mode == "ergodic_xyz" else 6
    position_x0_offset_m = np.asarray(args.ergodic_position_x0_offset_m, dtype=np.float64).reshape(3)
    rotation_x0_rad = np.asarray(args.ergodic_rotation_x0_rad, dtype=np.float64).reshape(3)
    default_low, default_high, default_x0 = _default_ergodic_bounds(
        env,
        dim,
        position_x0_offset_m=position_x0_offset_m,
        rotation_x0_rad=rotation_x0_rad,
    )
    low = _coerce_ergodic_vector(args.ergodic_low, dim=dim, name="ergodic_low", default=default_low)
    high = _coerce_ergodic_vector(args.ergodic_high, dim=dim, name="ergodic_high", default=default_high)
    x0 = _coerce_ergodic_vector(args.ergodic_x0, dim=dim, name="ergodic_x0", default=default_x0)

    current_pose = np.asarray(env._current_tcp_pose(), dtype=np.float64)
    base_rotation = Rotation.from_matrix(current_pose[:3, :3])
    smc_cfg = SMCErgodicConfig(
        dim=dim,
        low=tuple(float(v) for v in low.tolist()),
        high=tuple(float(v) for v in high.tolist()),
        num_k_per_dim=int(args.ergodic_num_k_per_dim),
        dt=float(args.ergodic_dt),
        speed=float(args.ergodic_speed),
        boundary=str(args.ergodic_boundary),
        seed=int(args.seed),
    )
    ergodic_device = _resolve_ergodic_device(str(args.ergodic_device))
    result = run_uniform_box_smc(smc_cfg, tsteps=int(args.num_samples), x0=x0, device=ergodic_device)
    targets: list[np.ndarray] = []
    for state in result["x_traj"]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(state[:3], dtype=np.float64)
        if dim == 3:
            pose[:3, :3] = base_rotation.as_matrix()
        else:
            pose[:3, :3] = (Rotation.from_rotvec(np.asarray(state[3:6], dtype=np.float64)) * base_rotation).as_matrix()
        targets.append(pose)
    summary = {
        "dim": int(dim),
        "low": [float(v) for v in low.tolist()],
        "high": [float(v) for v in high.tolist()],
        "x0": [float(v) for v in x0.tolist()],
        "num_k_per_dim": int(args.ergodic_num_k_per_dim),
        "dt": float(args.ergodic_dt),
        "speed": float(args.ergodic_speed),
        "boundary": str(args.ergodic_boundary),
        "device": str(result.get("device", ergodic_device)),
        "final_metric": float(result["metric_log"][-1]),
        "x_traj": np.asarray(result["x_traj"], dtype=np.float64),
        "metric_log": np.asarray(result["metric_log"], dtype=np.float64),
    }
    return targets, summary


def _save_ergodic_visualization(
    *,
    targets: list[np.ndarray],
    ergodic_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    if not targets:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.asarray([pose[:3, 3] for pose in targets], dtype=np.float64)
    low = np.asarray(ergodic_summary.get("low", []), dtype=np.float64).reshape(-1)
    high = np.asarray(ergodic_summary.get("high", []), dtype=np.float64).reshape(-1)
    metric_log = np.asarray(ergodic_summary.get("metric_log", []), dtype=np.float64).reshape(-1)
    dim = int(ergodic_summary.get("dim", 3))
    files: dict[str, str] = {}

    fig = plt.figure(figsize=(7, 6), dpi=140, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="black", linewidth=1.0, alpha=0.7)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="C2", s=45, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="C3", s=45, label="end")
    if low.size >= 3 and high.size >= 3:
        ax.set_xlim(float(low[0]), float(high[0]))
        ax.set_ylim(float(low[1]), float(high[1]))
        ax.set_zlim(float(low[2]), float(high[2]))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Ergodic target EE position trajectory")
    ax.legend(loc="upper right")
    path = output_dir / "ergodic_target_ee_xyz_3d.png"
    fig.savefig(path)
    plt.close(fig)
    files["target_ee_xyz_3d_png"] = str(path)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=140, sharex=True, tight_layout=True)
    sample_idx = np.arange(positions.shape[0])
    labels = ("x [m]", "y [m]", "z [m]")
    for axis, label in enumerate(labels):
        axes[axis].plot(sample_idx, positions[:, axis], color=f"C{axis}", linewidth=1.0)
        if low.size >= 3 and high.size >= 3:
            axes[axis].axhline(float(low[axis]), color="0.55", linestyle="--", linewidth=0.8)
            axes[axis].axhline(float(high[axis]), color="0.55", linestyle="--", linewidth=0.8)
        axes[axis].set_ylabel(label)
    axes[-1].set_xlabel("sample")
    path = output_dir / "ergodic_target_ee_xyz_vs_sample.png"
    fig.savefig(path)
    plt.close(fig)
    files["target_ee_xyz_vs_sample_png"] = str(path)

    if dim == 6:
        x_traj = np.asarray(ergodic_summary.get("x_traj", []), dtype=np.float64)
        if x_traj.ndim == 2 and x_traj.shape[1] >= 6:
            fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=140, sharex=True, tight_layout=True)
            for axis, label in enumerate(("rx [rad]", "ry [rad]", "rz [rad]")):
                axes[axis].plot(sample_idx, x_traj[:, axis + 3], color=f"C{axis + 3}", linewidth=1.0)
                if low.size >= 6 and high.size >= 6:
                    axes[axis].axhline(float(low[axis + 3]), color="0.55", linestyle="--", linewidth=0.8)
                    axes[axis].axhline(float(high[axis + 3]), color="0.55", linestyle="--", linewidth=0.8)
                axes[axis].set_ylabel(label)
            axes[-1].set_xlabel("sample")
            path = output_dir / "ergodic_target_rotvec_vs_sample.png"
            fig.savefig(path)
            plt.close(fig)
            files["target_rotvec_vs_sample_png"] = str(path)

    if metric_log.size > 0:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140, tight_layout=True)
        ax.plot(np.arange(metric_log.size), metric_log, color="C0", linewidth=1.0)
        ax.set_xlabel("sample")
        ax.set_ylabel("ergodic metric")
        ax.set_title("SMC ergodic metric")
        path = output_dir / "ergodic_metric_vs_sample.png"
        fig.savefig(path)
        plt.close(fig)
        files["metric_vs_sample_png"] = str(path)

    serializable_summary = {
        key: value
        for key, value in ergodic_summary.items()
        if key not in {"x_traj", "metric_log"}
    }
    npz_path = output_dir / "ergodic_target_trajectory.npz"
    np.savez(
        npz_path,
        target_positions_xyz=positions,
        x_traj=np.asarray(ergodic_summary.get("x_traj", []), dtype=np.float64),
        metric_log=metric_log,
        low=low,
        high=high,
    )
    files["target_trajectory_npz"] = str(npz_path)
    json_path = output_dir / "ergodic_summary.json"
    _save_report(json_path, serializable_summary)
    files["summary_json"] = str(json_path)
    return files


def _drive_to_joint_target(
    env: FR3MujocoEnv,
    target_joints: np.ndarray,
    *,
    max_command_steps: int,
    settle_steps: int,
    joint_tolerance_rad: float,
) -> dict[str, Any]:
    steps_used = 0
    max_abs_error = float("inf")
    best_max_abs_error = float("inf")
    for step in range(1, int(max_command_steps) + 1):
        _, _, _, truncated, info = env.step(np.asarray(target_joints, dtype=np.float64))
        current = np.asarray(info["joint_positions"], dtype=np.float64)
        error = np.asarray(target_joints, dtype=np.float64) - current
        max_abs_error = float(np.max(np.abs(error)))
        best_max_abs_error = min(best_max_abs_error, max_abs_error)
        steps_used = step
        if max_abs_error <= float(joint_tolerance_rad):
            break
        if truncated:
            break

    last_info = info
    for _ in range(max(0, int(settle_steps))):
        _, _, _, _, last_info = env.step(np.asarray(target_joints, dtype=np.float64))

    final_joints = np.asarray(last_info["joint_positions"], dtype=np.float64)
    final_error = np.asarray(target_joints, dtype=np.float64) - final_joints
    final_max_abs_error = float(np.max(np.abs(final_error)))
    return {
        "reached": bool(final_max_abs_error <= float(joint_tolerance_rad)),
        "steps_used": int(steps_used),
        "best_joint_max_abs_error_rad": float(min(best_max_abs_error, final_max_abs_error)),
        "final_joint_max_abs_error_rad": final_max_abs_error,
        "final_joint_l2_error_rad": float(np.linalg.norm(final_error)),
        "final_joint_values_rad": [float(v) for v in final_joints.tolist()],
    }


def _compute_pose_error(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    from lerobot.utils.rotation import Rotation

    pos_error = np.asarray(target_pose[:3, 3], dtype=np.float64) - np.asarray(current_pose[:3, 3], dtype=np.float64)
    rot_error = (
        Rotation.from_matrix(np.asarray(target_pose[:3, :3], dtype=np.float64))
        * Rotation.from_matrix(np.asarray(current_pose[:3, :3], dtype=np.float64)).inv()
    ).as_rotvec()
    return float(np.linalg.norm(pos_error)), float(np.linalg.norm(rot_error))


def _drive_to_cartesian_pose(
    env: FR3MujocoEnv,
    target_pose: np.ndarray,
    *,
    max_command_steps: int,
    settle_steps: int,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    max_position_step_m: float,
    max_rotation_step_rad: float,
    orientation_weight: float,
) -> dict[str, Any]:
    from lerobot.utils.rotation import Rotation

    best_position_error_m = float("inf")
    best_orientation_error_rad = float("inf")
    steps_used = 0
    last_info: dict[str, Any] | None = None
    for step in range(1, int(max_command_steps) + 1):
        current_pose = np.asarray(env._current_tcp_pose(), dtype=np.float64)
        current_rot = Rotation.from_matrix(current_pose[:3, :3])
        target_rot = Rotation.from_matrix(np.asarray(target_pose[:3, :3], dtype=np.float64))
        pos_error_vec = np.asarray(target_pose[:3, 3], dtype=np.float64) - current_pose[:3, 3]
        rot_error_vec = (target_rot * current_rot.inv()).as_rotvec()
        position_error_m = float(np.linalg.norm(pos_error_vec))
        orientation_error_rad = float(np.linalg.norm(rot_error_vec))
        best_position_error_m = min(best_position_error_m, position_error_m)
        best_orientation_error_rad = min(best_orientation_error_rad, orientation_error_rad)
        steps_used = step
        if position_error_m <= position_tolerance_m and orientation_error_rad <= orientation_tolerance_rad:
            break

        command_pose = current_pose.copy()
        command_pose[:3, 3] = current_pose[:3, 3] + _clip_vector_norm(pos_error_vec, max_position_step_m)
        rot_step = _clip_vector_norm(rot_error_vec, max_rotation_step_rad)
        command_pose[:3, :3] = (Rotation.from_rotvec(rot_step) * current_rot).as_matrix()
        current_joints = np.asarray(env._get_joint_positions(), dtype=np.float64)
        target_joints = np.asarray(
            env._kinematics.inverse_kinematics(
                current_joints,
                command_pose,
                lock_orientation=True,
                orientation_weight=float(orientation_weight),
            ),
            dtype=np.float64,
        )
        _, _, _, truncated, last_info = env.step(target_joints)
        if truncated:
            break

    if last_info is None:
        last_info = env._build_info(include_camera_obs=False)
    for _ in range(max(0, int(settle_steps))):
        current_pose = np.asarray(env._current_tcp_pose(), dtype=np.float64)
        current_rot = Rotation.from_matrix(current_pose[:3, :3])
        target_rot = Rotation.from_matrix(np.asarray(target_pose[:3, :3], dtype=np.float64))
        pos_error_vec = np.asarray(target_pose[:3, 3], dtype=np.float64) - current_pose[:3, 3]
        rot_error_vec = (target_rot * current_rot.inv()).as_rotvec()
        position_error_m = float(np.linalg.norm(pos_error_vec))
        orientation_error_rad = float(np.linalg.norm(rot_error_vec))
        best_position_error_m = min(best_position_error_m, position_error_m)
        best_orientation_error_rad = min(best_orientation_error_rad, orientation_error_rad)
        if position_error_m <= position_tolerance_m and orientation_error_rad <= orientation_tolerance_rad:
            break

        command_pose = current_pose.copy()
        command_pose[:3, 3] = current_pose[:3, 3] + _clip_vector_norm(pos_error_vec, max_position_step_m)
        rot_step = _clip_vector_norm(rot_error_vec, max_rotation_step_rad)
        command_pose[:3, :3] = (Rotation.from_rotvec(rot_step) * current_rot).as_matrix()
        current_joints = np.asarray(env._get_joint_positions(), dtype=np.float64)
        target_joints = np.asarray(
            env._kinematics.inverse_kinematics(
                current_joints,
                command_pose,
                lock_orientation=True,
                orientation_weight=float(orientation_weight),
            ),
            dtype=np.float64,
        )
        _, _, _, truncated, last_info = env.step(target_joints)
        if truncated:
            break

    final_pose = np.asarray(env._current_tcp_pose(), dtype=np.float64)
    final_position_error_m, final_orientation_error_rad = _compute_pose_error(final_pose, target_pose)
    return {
        "reached": bool(
            final_position_error_m <= position_tolerance_m
            and final_orientation_error_rad <= orientation_tolerance_rad
        ),
        "steps_used": int(steps_used),
        "best_position_error_m": float(min(best_position_error_m, final_position_error_m)),
        "best_orientation_error_rad": float(min(best_orientation_error_rad, final_orientation_error_rad)),
        "final_position_error_m": float(final_position_error_m),
        "final_orientation_error_rad": float(final_orientation_error_rad),
        "final_joint_values_rad": [float(v) for v in np.asarray(last_info["joint_positions"], dtype=np.float64).tolist()],
    }


def _teleport_to_joint_target(
    env: FR3MujocoEnv,
    target_joints: np.ndarray,
    *,
    settle_steps: int,
    joint_tolerance_rad: float,
) -> dict[str, Any]:
    target = np.clip(np.asarray(target_joints, dtype=np.float64), env._joint_lower, env._joint_upper)
    env._reset_joint_state(target)
    env._reset_otg_state(target)
    env._servo_target_joints = target.copy()
    env._otg_target_joints = None
    for _ in range(max(0, int(settle_steps))):
        env._step_physics(1)
    final_joints = np.asarray(env._get_joint_positions(), dtype=np.float64)
    final_error = target - final_joints
    final_max_abs_error = float(np.max(np.abs(final_error)))
    return {
        "reached": bool(final_max_abs_error <= float(joint_tolerance_rad)),
        "steps_used": 0,
        "best_joint_max_abs_error_rad": final_max_abs_error,
        "final_joint_max_abs_error_rad": final_max_abs_error,
        "final_joint_l2_error_rad": float(np.linalg.norm(final_error)),
        "final_joint_values_rad": [float(v) for v in final_joints.tolist()],
    }


def _capture_frame(
    env: FR3MujocoEnv,
    *,
    task: str,
    episode_start_time_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = env._build_info(include_camera_obs=False)
    ee_pose = np.asarray(info["ee_pose"], dtype=np.float64)
    joints = np.asarray(info["joint_positions"], dtype=np.float32)
    gripper_pos = float(info.get("gripper_command", 1.0))
    state = _pose_to_state_vector(ee_pose, gripper_pos)

    capture_time_s = time.perf_counter()
    frames = env.render()
    if frames is None:
        raise RuntimeError("MuJoCo render returned None.")

    frame: dict[str, Any] = {
        "observation.state": state,
        "observation.joints": joints,
        "action": np.asarray(state, dtype=np.float32),
        "observation.device_capture_timestamp": np.asarray(
            [capture_time_s - episode_start_time_s] * (1 + len(env.cfg.camera_names)),
            dtype=np.float64,
        ),
        "task": task,
    }
    for camera_name in env.cfg.camera_names:
        image = np.asarray(frames[camera_name], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Camera {camera_name!r} produced invalid image shape {image.shape}.")
        frame[f"observation.images.{camera_name}"] = np.ascontiguousarray(image)

    quick_view = {
        "position_xyz_m": [float(v) for v in ee_pose[:3, 3].tolist()],
        "quaternion_xyzw": [float(v) for v in state[3:7].tolist()],
        "joint_values_rad": [float(v) for v in joints.tolist()],
        "num_images": len(env.cfg.camera_names),
    }
    return frame, quick_view


def _save_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _set_viewer_camera(env: FR3MujocoEnv, viewer: Any, viewer_camera: str) -> str:
    camera_name = str(viewer_camera).strip()
    if not camera_name:
        return ""
    mujoco = env._mujoco
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Viewer camera not found in MuJoCo model: {camera_name}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = int(camera_id)
    viewer.sync()
    return camera_name


def _sync_viewer(env: FR3MujocoEnv, viewer: Any, viewer_data: Any) -> None:
    env.copy_visual_state(viewer_data)
    viewer.sync()


def _maybe_hold_for_viewer(args: argparse.Namespace, sample_index: int) -> None:
    if args.viewer_hold_s > 0.0:
        time.sleep(float(args.viewer_hold_s))
    if args.viewer_pause_every > 0 and sample_index % int(args.viewer_pause_every) == 0:
        input(f"Paused after sample {sample_index}. Press Enter to continue...")


def _maybe_pause_after_ergodic_plan(args: argparse.Namespace, ergodic_visualization_files: dict[str, str]) -> None:
    if not bool(args.pause_after_ergodic_plan):
        return
    print(
        pformat(
            {
                "ergodic_plan_ready": True,
                "ergodic_visualizations": ergodic_visualization_files,
                "next_step": "tracking_and_capture",
            }
        )
    )
    try:
        input("Paused after ergodic planning. Press Enter to start tracking/capture...")
    except EOFError:
        print("No stdin available; continuing to tracking/capture.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    if args.dataset_fps <= 0:
        raise ValueError("--dataset-fps must be > 0.")
    if bool(args.ergodic_plan_only) and not str(args.sample_mode).startswith("ergodic_"):
        raise ValueError("--ergodic-plan-only requires sample-mode=ergodic_xyz or ergodic_6d.")
    if bool(args.pause_after_ergodic_plan) and not str(args.sample_mode).startswith("ergodic_"):
        raise ValueError("--pause-after-ergodic-plan requires sample-mode=ergodic_xyz or ergodic_6d.")
    resolved_motion_mode = _resolve_motion_mode(args)
    explicit_dests = getattr(args, "_explicit_cli_dests", set())
    requested_mujoco_gl = args.mujoco_gl
    if args.viewer and "mujoco_gl" not in explicit_dests:
        requested_mujoco_gl = "glfw"
    mujoco_gl_backend = _configure_mujoco_gl_backend(requested_mujoco_gl, viewer=bool(args.viewer))

    root = args.root.expanduser()
    if root.exists() and args.overwrite:
        shutil.rmtree(root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.envs.fr3_mujoco import FR3MujocoEnv

    env_cfg = _build_env_config(args)
    env = FR3MujocoEnv(env_cfg)
    dataset: LeRobotDataset | None = None
    viewer = None
    viewer_data = None
    records: list[dict[str, Any]] = []
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        ergodic_targets: list[np.ndarray] = []
        ergodic_summary: dict[str, Any] = {}
        ergodic_visualization_files: dict[str, str] = {}
        if str(args.sample_mode).startswith("ergodic_"):
            ergodic_targets, ergodic_summary = _build_ergodic_pose_targets(env, args)
            if bool(args.ergodic_plan_only):
                ergodic_visualization_files = _save_ergodic_visualization(
                    targets=ergodic_targets,
                    ergodic_summary=ergodic_summary,
                    output_dir=root / "visualizations" / "ergodic",
                )
                report_path = args.report_json
                if report_path is None:
                    report_path = root / "mujoco_calibration_report.json"
                _save_report(
                    report_path.expanduser(),
                    {
                        "dataset_root": str(root),
                        "repo_id": args.repo_id,
                        "num_samples": int(args.num_samples),
                        "motion_mode": resolved_motion_mode,
                        "sample_mode": args.sample_mode,
                        "ergodic_plan_only": True,
                        "ergodic": {
                            key: value
                            for key, value in ergodic_summary.items()
                            if key not in {"x_traj", "metric_log"}
                        },
                        "ergodic_visualizations": ergodic_visualization_files,
                    },
                )
                print(
                    pformat(
                        {
                            "dataset_root": str(root),
                            "num_samples": args.num_samples,
                            "motion_mode": resolved_motion_mode,
                            "sample_mode": args.sample_mode,
                            "ergodic_plan_only": True,
                            "ergodic": {
                                key: value
                                for key, value in ergodic_summary.items()
                                if key not in {"x_traj", "metric_log"}
                            },
                            "ergodic_visualizations": ergodic_visualization_files,
                            "report_json": str(report_path.expanduser()),
                        }
                    )
                )
                return 0
        selected_viewer_camera = ""
        if args.viewer:
            import mujoco.viewer

            viewer_data = env._mujoco.MjData(env.model)
            env.copy_visual_state(viewer_data)
            viewer = mujoco.viewer.launch_passive(env.model, viewer_data)
            selected_viewer_camera = _set_viewer_camera(env, viewer, args.viewer_camera)

        features = _build_dataset_features(env.cfg.camera_names, height=env.cfg.camera_height, width=env.cfg.camera_width)
        dataset = LeRobotDataset.create(
            args.repo_id,
            int(args.dataset_fps),
            root=root,
            robot_type="franka_research3_mujoco_calibration",
            features=features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=max(1, 2 * len(env.cfg.camera_names)),
            batch_encoding_size=1,
            vcodec=args.vcodec,
            streaming_encoding=bool(args.streaming_encoding),
            encoder_queue_maxsize=int(args.encoder_queue_maxsize),
            encoder_threads=args.encoder_threads,
        )
        if ergodic_targets:
            ergodic_visualization_files = _save_ergodic_visualization(
                targets=ergodic_targets,
                ergodic_summary=ergodic_summary,
                output_dir=Path(dataset.root) / "visualizations" / "ergodic",
            )
            _maybe_pause_after_ergodic_plan(args, ergodic_visualization_files)

        print(
            pformat(
                {
                    "dataset_root": str(dataset.root),
                    "repo_id": args.repo_id,
                    "num_samples": args.num_samples,
                    "camera_names": env.cfg.camera_names,
                    "camera_size": [env.cfg.camera_width, env.cfg.camera_height],
                    "sim_xml_path": env.cfg.sim_xml_path,
                    "mujoco_gl": mujoco_gl_backend,
                    "motion_mode": resolved_motion_mode,
                    "sample_mode": args.sample_mode,
                    "encoding": {
                        "requested_vcodec": str(args.vcodec),
                        "resolved_vcodec": str(dataset.vcodec),
                        "streaming_encoding": bool(args.streaming_encoding),
                        "encoder_threads": args.encoder_threads,
                        "encoder_queue_maxsize": int(args.encoder_queue_maxsize),
                    },
                    "controller": _controller_summary(args, env),
                    "cartesian_tracking": {
                        "position_tolerance_m": float(args.cartesian_position_tolerance_m),
                        "orientation_tolerance_rad": float(args.cartesian_orientation_tolerance_rad),
                        "max_position_step_m": float(args.cartesian_max_position_step_m),
                        "max_rotation_step_rad": float(args.cartesian_max_rotation_step_rad),
                        "orientation_weight": float(args.cartesian_orientation_weight),
                    },
                    "ergodic": {
                        key: value for key, value in ergodic_summary.items() if key not in {"x_traj", "metric_log"}
                    },
                    "ergodic_visualizations": ergodic_visualization_files,
                    "viewer": bool(args.viewer),
                    "viewer_camera": selected_viewer_camera,
                    "note": "simulation-only calibration data path; camera geometry need not match HIKON",
                }
            )
        )

        rng = np.random.default_rng(int(args.seed))
        episode_start_time_s = time.perf_counter()
        with VideoEncodingManager(dataset):
            for index in range(1, int(args.num_samples) + 1):
                record_target: dict[str, Any]
                if str(args.sample_mode).startswith("ergodic_"):
                    target_pose = ergodic_targets[index - 1]
                    record_target = {
                        "pose_matrix": np.asarray(target_pose, dtype=np.float64).tolist(),
                        "position_xyz_m": [float(v) for v in target_pose[:3, 3].tolist()],
                    }
                    if resolved_motion_mode != "cartesian_impedance":
                        raise ValueError(
                            f"sample-mode={args.sample_mode} requires motion-mode=cartesian_impedance or auto."
                        )
                    move_result = _drive_to_cartesian_pose(
                        env,
                        target_pose,
                        max_command_steps=args.max_command_steps,
                        settle_steps=args.settle_steps,
                        position_tolerance_m=args.cartesian_position_tolerance_m,
                        orientation_tolerance_rad=args.cartesian_orientation_tolerance_rad,
                        max_position_step_m=args.cartesian_max_position_step_m,
                        max_rotation_step_rad=args.cartesian_max_rotation_step_rad,
                        orientation_weight=args.cartesian_orientation_weight,
                    )
                else:
                    target_joints = _sample_joint_target(
                        env,
                        rng,
                        joint_delta_rad=args.joint_delta_rad,
                        joint_margin_rad=args.joint_margin_rad,
                        sample_mode=args.sample_mode,
                    )
                    record_target = {"joint_values_rad": [float(v) for v in target_joints.tolist()]}
                    if resolved_motion_mode == "teleport":
                        move_result = _teleport_to_joint_target(
                            env,
                            target_joints,
                            settle_steps=args.settle_steps,
                            joint_tolerance_rad=args.joint_tolerance_rad,
                        )
                    else:
                        move_result = _drive_to_joint_target(
                            env,
                            target_joints,
                            max_command_steps=args.max_command_steps,
                            settle_steps=args.settle_steps,
                            joint_tolerance_rad=args.joint_tolerance_rad,
                        )
                if viewer is not None and viewer_data is not None:
                    _sync_viewer(env, viewer, viewer_data)
                    _maybe_hold_for_viewer(args, index)
                frame, quick_view = _capture_frame(env, task=args.task, episode_start_time_s=episode_start_time_s)
                dataset.add_frame(frame)
                sample_idx = int(dataset.episode_buffer["size"])
                if "final_joint_max_abs_error_rad" in move_result:
                    error_text = f"joint_max_err={move_result['final_joint_max_abs_error_rad']:.5f}rad"
                else:
                    error_text = (
                        f"pos_err={move_result['final_position_error_m']:.5f}m "
                        f"rot_err={move_result['final_orientation_error_rad']:.5f}rad"
                    )
                print(
                    f"[{index:03d}/{int(args.num_samples):03d}] sample #{sample_idx}: "
                    f"move_success={move_result['reached']} {error_text} "
                    f"images={quick_view['num_images']}"
                )
                records.append(
                    {
                        "sample_index": int(index),
                        "target": record_target,
                        "move_result": move_result,
                        "capture_quick_view": quick_view,
                    }
                )

            if dataset.episode_buffer is not None and int(dataset.episode_buffer["size"]) > 0:
                dataset.save_episode()
                print("Saved one MuJoCo calibration episode.")

        report_path = args.report_json
        if report_path is None:
            report_path = root / "mujoco_calibration_report.json"
        _save_report(
            report_path.expanduser(),
            {
                "dataset_root": str(dataset.root),
                "repo_id": args.repo_id,
                "camera_names": list(env.cfg.camera_names),
                "sim_xml_path": env.cfg.sim_xml_path,
                "motion_mode": resolved_motion_mode,
                "sample_mode": args.sample_mode,
                "encoding": {
                    "requested_vcodec": str(args.vcodec),
                    "resolved_vcodec": str(dataset.vcodec),
                    "streaming_encoding": bool(args.streaming_encoding),
                    "encoder_threads": args.encoder_threads,
                    "encoder_queue_maxsize": int(args.encoder_queue_maxsize),
                },
                "controller": _controller_summary(args, env),
                "cartesian_tracking": {
                    "position_tolerance_m": float(args.cartesian_position_tolerance_m),
                    "orientation_tolerance_rad": float(args.cartesian_orientation_tolerance_rad),
                    "max_position_step_m": float(args.cartesian_max_position_step_m),
                    "max_rotation_step_rad": float(args.cartesian_max_rotation_step_rad),
                    "orientation_weight": float(args.cartesian_orientation_weight),
                },
                "ergodic": {
                    key: value for key, value in ergodic_summary.items() if key not in {"x_traj", "metric_log"}
                },
                "ergodic_visualizations": ergodic_visualization_files,
                "records": records,
            },
        )
        print(f"Report saved: {report_path.expanduser()}")
        if viewer is not None and args.viewer_final_hold_s > 0.0:
            _sync_viewer(env, viewer, viewer_data)
            time.sleep(float(args.viewer_final_hold_s))
        return 0
    finally:
        if viewer is not None:
            viewer.close()
        if dataset is not None:
            dataset.finalize()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
