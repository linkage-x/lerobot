#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run FR3 MuJoCo simulation recording with the default ee2ee dataset contract."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.franka_research3 import (
    AbsoluteEEActionToRobotAction,
    DeltaActionToAbsoluteEEAction,
    KeepAbsoluteEEObservation,
)
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators.spacemouse.configuration_spacemouse import (
    SpaceMouseEnableButton,
    SpaceMouseTeleopConfig,
    SpaceMouseToolMode,
)
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.constants import ACTION, OBS_STR

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
_SIM_CAMERA_NAMES = FR3MujocoEnvConfig().camera_names
_D435I_IMAGE_SHAPE = (480, 640, 3)


def _get_env_observation(info: dict, gripper_pos: float) -> dict:
    ee_pose = np.asarray(info["ee_pose"], dtype=np.float64)
    joint_positions = np.asarray(info["joint_positions"], dtype=np.float64)
    ee_rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
    observation = {
        "ee.x": float(ee_pose[0, 3]),
        "ee.y": float(ee_pose[1, 3]),
        "ee.z": float(ee_pose[2, 3]),
        "ee.wx": float(ee_rotvec[0]),
        "ee.wy": float(ee_rotvec[1]),
        "ee.wz": float(ee_rotvec[2]),
        "gripper.pos": float(gripper_pos),
    }
    for index, joint_position in enumerate(joint_positions, start=1):
        observation[f"joint_{index}.pos"] = float(joint_position)
    camera_obs = info.get("camera_obs")
    if isinstance(camera_obs, dict):
        for camera_name in _SIM_CAMERA_NAMES:
            image = camera_obs.get(camera_name)
            if image is not None:
                observation[camera_name] = np.asarray(image)
    return observation


def _complete_robot_observation(observation: dict) -> dict:
    completed = dict(observation)
    for axis in ("qx", "qy", "qz", "qw"):
        completed.setdefault(f"prev_cmd.ee.{axis}", completed.get(f"ee.{axis}", 0.0))
    return completed


def _build_robot_observation_features(
    include_cameras: bool = False,
    camera_shape: tuple[int, int, int] = _D435I_IMAGE_SHAPE,
) -> dict:
    features = {
        "ee.x": float,
        "ee.y": float,
        "ee.z": float,
        "ee.wx": float,
        "ee.wy": float,
        "ee.wz": float,
        "gripper.pos": float,
    }
    for index in range(1, 8):
        features[f"joint_{index}.pos"] = float
    if include_cameras:
        for camera_name in _SIM_CAMERA_NAMES:
            features[camera_name] = camera_shape
    return features


def _build_teleop_features() -> dict:
    return {
        "enabled": bool,
        "target_x": float,
        "target_y": float,
        "target_z": float,
        "target_wx": float,
        "target_wy": float,
        "target_wz": float,
        "gripper": float,
    }


def _update_viewer_markers(mujoco, viewer, info: dict, style: MarkerStyle) -> None:
    target_pose = np.asarray(info["target_pose"], dtype=np.float64)
    tcp_pose = np.asarray(info["tcp_pose"], dtype=np.float64)
    scene = viewer.user_scn
    geoms = [
        {"kind": "sphere", "pos": target_pose[:3, 3], "rgba": style.target_rgba, "size": style.sphere_radius},
        {"kind": "sphere", "pos": tcp_pose[:3, 3], "rgba": style.tcp_rgba, "size": style.sphere_radius},
    ]
    if scene.maxgeom < len(geoms):
        raise RuntimeError(f"Viewer supports {scene.maxgeom} geoms, need {len(geoms)}")
    scene.ngeom = 0
    for geom_data in geoms:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([geom_data["size"], 0, 0]),
                            geom_data["pos"], np.eye(3).reshape(-1), geom_data["rgba"])
        scene.ngeom += 1


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop_action_processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[
            DeltaActionToAbsoluteEEAction(
                workspace_min=cfg.robot.workspace_min,
                workspace_max=cfg.robot.workspace_max,
                max_target_delta_pos=cfg.robot.max_target_delta_pos,
                max_target_delta_rot=cfg.robot.max_target_delta_rot,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[AbsoluteEEActionToRobotAction()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[dict, dict](
        steps=[KeepAbsoluteEEObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    teleop_features = _build_teleop_features()
    robot_observation_features = _build_robot_observation_features(
        include_cameras=cfg.dataset.video,
        camera_shape=(480, 640, 3),
    )
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=teleop_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot_observation_features),
            use_videos=cfg.dataset.video,
        ),
        hw_to_dataset_features(robot_observation_features, prefix="observation", use_video=cfg.dataset.video),
    )

    from datetime import datetime
    dataset_root = Path(cfg.dataset.root)
    if not str(dataset_root).startswith("/workspace/"):
        dataset_root = Path("/workspace") / dataset_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = dataset_root.parent / f"{dataset_root.name}_{timestamp}"

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=str(dataset_root),
        robot_type="franka_research3_mujoco",
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(_SIM_CAMERA_NAMES),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )

    env_cfg = FR3MujocoEnvConfig(
        teleop_control_frequency=float(cfg.control_fps or cfg.dataset.fps),
        max_episode_steps=max(int(cfg.dataset.episode_time_s * (cfg.control_fps or cfg.dataset.fps)) + 100, 1000),
        use_otg=False,
        enable_cameras=True,
        camera_width=640,
        camera_height=480,
        continuous_physics=True,
        continuous_physics_frequency=800.0,
    )
    env = FR3MujocoEnv(env_cfg)

    teleop_cfg = SpaceMouseTeleopConfig(
        device_id=0,
        frequency=cfg.control_fps or cfg.dataset.fps,
        translation_scale=0.001845,
        rotation_scale=0.001944,
        enable_rotation=True,
        tool_mode=SpaceMouseToolMode.INCREMENTAL,
        motion_enable_button=SpaceMouseEnableButton.NONE,
    )
    teleop = SpaceMouseTeleop(teleop_cfg)

    marker_style = MarkerStyle()
    viewer = None
    viewer_data = None

    listener, events = None, None
    control_fps = cfg.control_fps or cfg.dataset.fps

    try:
        import mujoco
        import mujoco.viewer

        viewer_data = mujoco.MjData(env.model)
        env.copy_visual_state(viewer_data)
        viewer = mujoco.viewer.launch_passive(env.model, viewer_data)
        print("fr3_mujoco_record=READY")
    except Exception as e:
        logging.warning(f"Could not launch viewer: {e}")

    try:
        teleop.connect()
        listener, events = init_keyboard_listener()

        recorded_episodes = 0
        current_gripper = 1.0

        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            _, info = env.reset()
            sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
            if callable(sync_gripper) and "gripper_command" in info:
                sync_gripper(float(info["gripper_command"]))
                current_gripper = float(info["gripper_command"])

            if viewer is not None:
                copy_visual_state = getattr(env, "copy_visual_state", None)
                if viewer_data is not None and callable(copy_visual_state):
                    copy_visual_state(viewer_data)
                with viewer.lock():
                    _update_viewer_markers(env._mujoco, viewer, info, marker_style)
                viewer.sync()

            timestamp = 0.0
            start_episode_t = time.perf_counter()
            dataset_frame_period_s = 1 / dataset.fps
            next_dataset_frame_t = 0.0

            while timestamp < cfg.dataset.episode_time_s and not events["stop_recording"]:
                loop_start = time.perf_counter()

                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                loop_elapsed_s = time.perf_counter() - start_episode_t
                should_capture = loop_elapsed_s + 1e-9 >= next_dataset_frame_t

                obs_raw = _get_env_observation(info, gripper_pos=current_gripper)
                obs_processed = robot_observation_processor(obs_raw)
                obs_completed = _complete_robot_observation(obs_raw)

                action = teleop.get_action()
                action_processed = teleop_action_processor((action.copy(), obs_raw))
                action_values = action_processed

                _, _, terminated, truncated, info = env.step_teleop_action(
                    action, control_period_s=1.0 / control_fps
                )
                current_gripper = float(action.get("gripper", current_gripper))

                if viewer is not None and not viewer.is_running():
                    break
                if viewer is not None:
                    copy_visual_state = getattr(env, "copy_visual_state", None)
                    if viewer_data is not None and callable(copy_visual_state):
                        copy_visual_state(viewer_data)
                    with viewer.lock():
                        _update_viewer_markers(env._mujoco, viewer, info, marker_style)
                    viewer.sync()

                if should_capture:
                    observation_frame = build_dataset_frame(
                        dataset.features,
                        {**obs_completed, **robot_observation_processor(obs_raw)},
                        prefix=OBS_STR,
                    )
                    action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                    frame = {**observation_frame, **action_frame, "task": cfg.dataset.single_task}
                    dataset.add_frame(frame)

                    while next_dataset_frame_t <= loop_elapsed_s + 1e-9:
                        next_dataset_frame_t += dataset_frame_period_s

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / control_fps - dt_s, 0.0))
                timestamp = time.perf_counter() - start_episode_t

                if terminated or truncated:
                    break

            keep_episode = True
            if keep_episode:
                dataset.save_episode()
                recorded_episodes += 1
            else:
                logging.info("Discarding recorded episode %s.", dataset.num_episodes)
                dataset.clear_episode_buffer()

            teleop_action_processor.reset()
            robot_observation_processor.reset()

        dataset.finalize()

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if viewer is not None:
            viewer.close()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        env.close()

    return dataset


def main() -> None:
    record()


if __name__ == "__main__":
    main()
