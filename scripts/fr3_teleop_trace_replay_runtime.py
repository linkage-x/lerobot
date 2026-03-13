#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.calibration.fr3_teleop import (
    build_combined_trace_profile,
    build_default_trace_profile,
    build_trace_bundle,
    build_wz_trace_profile,
    make_trace_sample,
    save_trace_bundle,
)
from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config
from lerobot.utils.robot_utils import precise_sleep


DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT = (0.0, 0.0, 0.0)
DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT = (0.01, 0.01, 0.01)
DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "robots"
    / "franka_research3"
    / "assets"
    / "franka_fr3"
    / "fr3_pika_gripper_ati.urdf"
)


def _parse_tuple(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a fixed FR3 teleop profile and record TCP traces.")
    parser.add_argument("--mode", choices=["sim", "hardware"], required=True)
    parser.add_argument("--trace-profile", choices=["translation", "combined", "wz"], default="translation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--step-x", type=float, default=0.0002)
    parser.add_argument("--step-y", type=float, default=0.0002)
    parser.add_argument("--step-z", type=float, default=0.0002)
    parser.add_argument("--step-wx", type=float, default=0.0002)
    parser.add_argument("--step-wy", type=float, default=0.0002)
    parser.add_argument("--step-wz", type=float, default=0.0002)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--move-s", type=float, default=0.75)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--target-frame-name", default="pika_gripper_ee")
    parser.add_argument("--workspace-min", type=_parse_tuple, default=(0.2, -0.6, 0.05))
    parser.add_argument("--workspace-max", type=_parse_tuple, default=(0.9, 0.6, 0.8))
    parser.add_argument("--max-target-delta-pos", type=_parse_tuple, default=(0.001, 0.001, 0.001))
    parser.add_argument("--max-target-delta-rot", type=_parse_tuple, default=None)
    args = parser.parse_args()
    if args.max_target_delta_rot is None:
        if args.trace_profile in ("combined", "wz"):
            args.max_target_delta_rot = DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT
        else:
            args.max_target_delta_rot = DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT
    return args


def _observation_to_sample(*, profile_step: int, scheduled_time_s: float, measured_time_s: float, action, observation, target_pose):
    joint_positions = np.array([observation[f"joint_{index}.pos"] for index in range(1, 8)], dtype=np.float64)
    ee_position = np.array(
        [observation["ee.x"], observation["ee.y"], observation["ee.z"]],
        dtype=np.float64,
    )
    ee_rotvec = np.array(
        [observation["ee.wx"], observation["ee.wy"], observation["ee.wz"]],
        dtype=np.float64,
    )
    target_position = None
    target_rotvec = None
    if target_pose is not None:
        target_position = np.asarray(target_pose[:3, 3], dtype=np.float64)
        target_rotvec = Rotation.from_matrix(target_pose[:3, :3]).as_rotvec()
    return make_trace_sample(
        profile_step=profile_step,
        scheduled_time_s=scheduled_time_s,
        measured_time_s=measured_time_s,
        action=action,
        joint_positions=joint_positions,
        ee_position=ee_position,
        ee_rotvec=ee_rotvec,
        gripper=float(observation["gripper.pos"]),
        target_position=target_position,
        target_rotvec=target_rotvec,
    )


def _sim_info_to_sample(*, profile_step: int, scheduled_time_s: float, measured_time_s: float, action, info):
    target_pose = np.asarray(info["target_pose"], dtype=np.float64)
    tcp_pose = np.asarray(info["tcp_pose"], dtype=np.float64)
    ee_rotvec = Rotation.from_matrix(tcp_pose[:3, :3]).as_rotvec()
    target_rotvec = Rotation.from_matrix(target_pose[:3, :3]).as_rotvec()
    return make_trace_sample(
        profile_step=profile_step,
        scheduled_time_s=scheduled_time_s,
        measured_time_s=measured_time_s,
        action=action,
        joint_positions=np.asarray(info["joint_positions"], dtype=np.float64),
        ee_position=np.asarray(tcp_pose[:3, 3], dtype=np.float64),
        ee_rotvec=ee_rotvec,
        gripper=float(action["gripper"]) if action is not None else 1.0,
        target_position=np.asarray(target_pose[:3, 3], dtype=np.float64),
        target_rotvec=target_rotvec,
        extra={
            "otg_enabled": bool(info.get("otg_enabled", False)),
            "otg_steps": int(info.get("otg_steps", 0)),
            "sender_steps": int(info.get("sender_steps", 0)),
        },
    )


def run_sim_trace(args: argparse.Namespace, profile: dict[str, object]) -> dict[str, object]:
    env = FR3MujocoEnv(
        FR3MujocoEnvConfig(
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            max_target_delta_pos=args.max_target_delta_pos,
            max_target_delta_rot=args.max_target_delta_rot,
            teleop_control_frequency=float(args.fps),
            max_episode_steps=len(profile["actions"]) + 10,
        )
    )
    samples: list[dict[str, object]] = []
    try:
        _, info = env.reset()
        samples.append(
            _sim_info_to_sample(
                profile_step=-1,
                scheduled_time_s=0.0,
                measured_time_s=0.0,
                action=None,
                info=info,
            )
        )
        start_time = time.perf_counter()
        for step_index, action in enumerate(profile["actions"]):
            _, _, terminated, truncated, info = env.step_teleop_action(action, control_period_s=1.0 / args.fps)
            measured_time_s = time.perf_counter() - start_time
            samples.append(
                _sim_info_to_sample(
                    profile_step=step_index,
                    scheduled_time_s=(step_index + 1) / args.fps,
                    measured_time_s=measured_time_s,
                    action=action,
                    info=info,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return build_trace_bundle(
        mode="sim",
        profile=profile,
        samples=samples,
        metadata={
            "urdf_path": str(args.urdf_path),
            "target_frame_name": args.target_frame_name,
        },
    )


def run_hardware_trace(args: argparse.Namespace, profile: dict[str, object]) -> dict[str, object]:
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip=args.robot_ip,
            gripper_port=args.gripper_port,
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            max_target_delta_pos=args.max_target_delta_pos,
            max_target_delta_rot=args.max_target_delta_rot,
        )
    )
    samples: list[dict[str, object]] = []
    try:
        robot.connect()
        observation = robot.get_observation()
        samples.append(
            _observation_to_sample(
                profile_step=-1,
                scheduled_time_s=0.0,
                measured_time_s=0.0,
                action=None,
                observation=observation,
                target_pose=getattr(robot, "_last_command_pose", None),
            )
        )
        start_time = time.perf_counter()
        next_deadline = start_time
        for step_index, action in enumerate(profile["actions"]):
            robot.send_action(action)
            next_deadline += 1.0 / args.fps
            precise_sleep(max(next_deadline - time.perf_counter(), 0.0))
            observation = robot.get_observation()
            measured_time_s = time.perf_counter() - start_time
            samples.append(
                _observation_to_sample(
                    profile_step=step_index,
                    scheduled_time_s=(step_index + 1) / args.fps,
                    measured_time_s=measured_time_s,
                    action=action,
                    observation=observation,
                    target_pose=getattr(robot, "_last_command_pose", None),
                )
            )
    finally:
        if robot.is_connected:
            robot.disconnect()
    return build_trace_bundle(
        mode="hardware",
        profile=profile,
        samples=samples,
        metadata={
            "robot_ip": args.robot_ip,
            "gripper_port": args.gripper_port,
            "gripper_is_mock": bool(getattr(robot, "_gripper_is_mock", False)),
            "urdf_path": str(args.urdf_path),
            "target_frame_name": args.target_frame_name,
        },
    )


def main() -> int:
    args = parse_args()
    if args.trace_profile == "combined":
        profile = build_combined_trace_profile(
            fps=args.fps,
            step_x=args.step_x,
            step_y=args.step_y,
            step_z=args.step_z,
            step_wx=args.step_wx,
            step_wy=args.step_wy,
            step_wz=args.step_wz,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    elif args.trace_profile == "wz":
        profile = build_wz_trace_profile(
            fps=args.fps,
            step_wz=args.step_wz,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    else:
        profile = build_default_trace_profile(
            fps=args.fps,
            step_x=args.step_x,
            step_y=args.step_y,
            step_z=args.step_z,
            warmup_s=args.warmup_s,
            move_s=args.move_s,
            hold_s=args.hold_s,
            settle_s=args.settle_s,
        )
    if args.mode == "sim":
        bundle = run_sim_trace(args, profile)
    else:
        bundle = run_hardware_trace(args, profile)
    save_trace_bundle(args.output, bundle)
    print(f"fr3_trace_replay={args.mode}")
    print(f"output={args.output}")
    print(f"samples={len(bundle['samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
