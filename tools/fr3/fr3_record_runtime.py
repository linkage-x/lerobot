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

"""Run FR3 recording with the default ee2ee dataset contract."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import franka_research3, make_robot_from_config
from lerobot.robots.franka_research3 import (
    AbsoluteEEActionToRobotAction,
    DeltaActionToAbsoluteEEAction,
    KeepAbsoluteEEObservation,
)
from lerobot.scripts.lerobot_record import (
    RecordConfig,
    _confirm_next_episode,
    _move_robot_to_start,
    record_loop,
)
import lerobot.teleoperators.spacemouse  # noqa: F401
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
EPISODE_START_SETTLE_POSITION_THRESHOLD_M = 0.002
EPISODE_START_SETTLE_ANGLE_THRESHOLD_RAD = np.deg2rad(1.0)
EPISODE_START_SETTLE_GRIPPER_THRESHOLD = 0.02
EPISODE_START_SETTLE_CONSECUTIVE_SAMPLES = 5
EPISODE_START_SETTLE_TIMEOUT_S = 3.0


def make_fr3_ee2ee_processors(cfg: RecordConfig) -> tuple[
    RobotProcessorPipeline[tuple[dict, dict], dict],
    RobotProcessorPipeline[tuple[dict, dict], dict],
    RobotProcessorPipeline[dict, dict],
]:
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
    return teleop_action_processor, robot_action_processor, robot_observation_processor


def _quaternion_angle_error_rad(target_xyzw: np.ndarray, current_xyzw: np.ndarray) -> float:
    dot = float(np.dot(np.asarray(target_xyzw, dtype=np.float64), np.asarray(current_xyzw, dtype=np.float64)))
    return 2.0 * float(np.arccos(np.clip(abs(dot), 0.0, 1.0)))


def _compute_episode_start_settle_errors(observation: dict, target_action: dict) -> tuple[float, float, float]:
    position_error_m = float(
        np.linalg.norm(
            np.array([observation[key] for key in EE_POSITION_KEYS], dtype=np.float64)
            - np.array([target_action[key] for key in EE_POSITION_KEYS], dtype=np.float64)
        )
    )
    angle_error_rad = _quaternion_angle_error_rad(
        np.array([target_action[key] for key in EE_QUAT_KEYS], dtype=np.float64),
        np.array([observation[key] for key in EE_QUAT_KEYS], dtype=np.float64),
    )
    gripper_error = abs(float(observation["gripper.pos"]) - float(target_action["gripper.pos"]))
    return position_error_m, angle_error_rad, gripper_error


def _wait_for_episode_start_settle(
    *,
    robot,
    teleop,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    events: dict[str, bool],
    fps: int,
) -> None:
    initial_observation = robot.get_observation()
    initial_action = teleop.get_action()
    target_action = teleop_action_processor((initial_action, initial_observation))

    settle_start_t = time.perf_counter()
    consecutive_ok = 0
    last_errors: tuple[float, float, float] | None = None
    while not events["stop_recording"]:
        loop_start_t = time.perf_counter()
        current_observation = robot.get_observation()
        current_observation_processed = robot_observation_processor(current_observation)
        robot_action = robot_action_processor((target_action, current_observation))
        robot.send_action(robot_action)

        last_errors = _compute_episode_start_settle_errors(current_observation_processed, target_action)
        position_error_m, angle_error_rad, gripper_error = last_errors
        if (
            position_error_m <= EPISODE_START_SETTLE_POSITION_THRESHOLD_M
            and angle_error_rad <= EPISODE_START_SETTLE_ANGLE_THRESHOLD_RAD
            and gripper_error <= EPISODE_START_SETTLE_GRIPPER_THRESHOLD
        ):
            consecutive_ok += 1
            if consecutive_ok >= EPISODE_START_SETTLE_CONSECUTIVE_SAMPLES:
                logging.info(
                    "Episode start settled in %.2fs (position=%.4fm, angle=%.2fdeg, gripper=%.3f).",
                    time.perf_counter() - settle_start_t,
                    position_error_m,
                    np.rad2deg(angle_error_rad),
                    gripper_error,
                )
                break
        else:
            consecutive_ok = 0

        if time.perf_counter() - settle_start_t >= EPISODE_START_SETTLE_TIMEOUT_S:
            break
        precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_start_t), 0.0))

    if last_errors is not None and consecutive_ok < EPISODE_START_SETTLE_CONSECUTIVE_SAMPLES:
        position_error_m, angle_error_rad, gripper_error = last_errors
        logging.warning(
            "Timed out waiting for episode start settle after %.2fs (position=%.4fm, angle=%.2fdeg, gripper=%.3f).",
            time.perf_counter() - settle_start_t,
            position_error_m,
            np.rad2deg(angle_error_rad),
            gripper_error,
        )

    teleop_action_processor.reset()
    robot_observation_processor.reset()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    if cfg.teleop is None:
        raise ValueError("FR3 default ee2ee recording requires a teleoperator configuration.")
    if cfg.policy is not None:
        raise ValueError("FR3 default ee2ee runtime currently supports teleop recording only, not policy evaluation.")

    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="fr3_recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_fr3_ee2ee_processors(cfg)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=teleop.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    listener, events = None, None
    control_fps = cfg.control_fps or cfg.dataset.fps

    try:
        robot.connect()
        teleop.connect()
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                _wait_for_episode_start_settle(
                    robot=robot,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    events=events,
                    fps=control_fps,
                )
                if events["stop_recording"]:
                    break
                record_loop(
                    robot=robot,
                    events=events,
                    fps=control_fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=None,
                    preprocessor=None,
                    postprocessor=None,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                teleop_action_processor.reset()
                robot_observation_processor.reset()

                should_move_to_start = (
                    cfg.auto_move_to_start_after_episode
                    and not events["stop_recording"]
                    and (
                        recorded_episodes < cfg.dataset.num_episodes
                        or cfg.move_to_start_after_last_episode
                    )
                )
                if should_move_to_start:
                    _move_robot_to_start(robot, cfg.play_sounds)

                should_run_reset_window = (
                    cfg.dataset.reset_time_s > 0
                    and not events["stop_recording"]
                    and recorded_episodes < cfg.dataset.num_episodes
                )
                if should_run_reset_window:
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=control_fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=None,
                        preprocessor=None,
                        postprocessor=None,
                        dataset=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed_images,
                    )

                should_confirm_next_episode = (
                    cfg.confirm_next_episode_after_reset
                    and not events["stop_recording"]
                    and recorded_episodes < cfg.dataset.num_episodes
                )
                if should_confirm_next_episode and not _confirm_next_episode(cfg.play_sounds):
                    events["stop_recording"] = True
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        log_say("Exiting", cfg.play_sounds)

    return dataset


def main() -> None:
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
