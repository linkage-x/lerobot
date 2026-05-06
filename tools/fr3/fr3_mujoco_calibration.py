#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate FR3 MuJoCo moving-ChArUco calibration data in LeRobot format. "
            "This is a simulation-only data-path validator: no teleoperation and no physical camera setup."
        )
    )
    parser.add_argument("--repo-id", default="local/fr3_mujoco_calibration")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/fr3_mujoco_calibration"))
    parser.add_argument("--task", default="FR3 MuJoCo calibration")
    parser.add_argument("--overwrite", action="store_true", help="Remove --root before creating the dataset.")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--dataset-fps", type=int, default=10)
    parser.add_argument("--control-frequency", type=float, default=120.0)
    parser.add_argument("--max-command-steps", type=int, default=160)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--joint-delta-rad", type=float, default=0.20)
    parser.add_argument("--joint-margin-rad", type=float, default=0.08)
    parser.add_argument(
        "--sample-mode",
        choices=("random_walk", "around_initial"),
        default="random_walk",
        help="random_walk samples the next target around current joints; around_initial samples every target around reset joints.",
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
    parser.add_argument("--enable-otg", action="store_true")
    parser.add_argument(
        "--motion-mode",
        choices=("auto", "teleport", "servo"),
        default="auto",
        help=(
            "auto uses teleport for sample-mode=random_walk and servo for other sample modes. "
            "teleport writes MuJoCo qpos directly; servo uses env.step() position control."
        ),
    )
    parser.add_argument("--viewer", action="store_true", help="Open a passive MuJoCo viewer during capture.")
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
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--mujoco-gl",
        choices=("glfw", "egl", "osmesa"),
        default=None,
        help="Set MUJOCO_GL before creating renderers. Defaults to glfw with --viewer, otherwise egl.",
    )
    return parser.parse_args(argv)


def _resolve_motion_mode(args: argparse.Namespace) -> str:
    if args.motion_mode != "auto":
        return str(args.motion_mode)
    if args.sample_mode == "random_walk":
        return "teleport"
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
        arm_actuator_kp=float(args.arm_actuator_kp),
        arm_gravity_compensation_scale=float(args.arm_gravity_comp_scale),
        continuous_physics=bool(args.continuous_physics),
        continuous_physics_frequency=float(args.continuous_physics_frequency),
    )


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    if args.dataset_fps <= 0:
        raise ValueError("--dataset-fps must be > 0.")
    resolved_motion_mode = _resolve_motion_mode(args)
    mujoco_gl_backend = _configure_mujoco_gl_backend(args.mujoco_gl, viewer=bool(args.viewer))

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
            encoder_queue_maxsize=30,
            encoder_threads=args.encoder_threads,
        )

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
                target_joints = _sample_joint_target(
                    env,
                    rng,
                    joint_delta_rad=args.joint_delta_rad,
                    joint_margin_rad=args.joint_margin_rad,
                    sample_mode=args.sample_mode,
                )
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
                print(
                    f"[{index:03d}/{int(args.num_samples):03d}] sample #{sample_idx}: "
                    f"move_success={move_result['reached']} "
                    f"joint_max_err={move_result['final_joint_max_abs_error_rad']:.5f}rad "
                    f"images={quick_view['num_images']}"
                )
                records.append(
                    {
                        "sample_index": int(index),
                        "target_joint_values_rad": [float(v) for v in target_joints.tolist()],
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
