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

import argparse
import logging
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import (
    _encode_grid_jpeg,
    _render_camera_frames,
    _start_camera_stream_outputs,
    render_camera_grid,
    update_passive_viewer_markers,
)
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
from lerobot.scripts.lerobot_record import _confirm_keep_episode
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.constants import ACTION, OBS_STR
try:
    from tools.fr3.fr3_mujoco_runtime import (
        build_runtime_env,
        build_runtime_marker_style,
        build_runtime_teleop_config,
        configure_mujoco_gl_backend,
        configure_viewer_camera,
        parse_runtime_args,
    )
except ModuleNotFoundError:
    from fr3_mujoco_runtime import (
        build_runtime_env,
        build_runtime_marker_style,
        build_runtime_teleop_config,
        configure_mujoco_gl_backend,
        configure_viewer_camera,
        parse_runtime_args,
    )

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
_SIM_CAMERA_NAMES = FR3MujocoEnvConfig().camera_names
_D435I_IMAGE_SHAPE = (480, 640, 3)
_WORKSPACE_OBJECT_BODY_NAME = "workspace_object_body"
_WORKSPACE_OBJECT_RANDOM_RADIUS_M = 0.10
_RUNTIME_ARGS: argparse.Namespace | None = None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTAINER_WORKSPACE = "/workspace"
_LEGACY_CONTAINER_WORKSPACE = "/lerobot"


class _LatestEpisodeControlState:
    def __init__(
        self,
        *,
        sample_info: dict[str, Any],
        viewer_info: dict[str, Any],
        action: dict[str, Any],
        sample_gripper: float,
    ) -> None:
        self._lock = threading.Lock()
        self._sample_info = dict(sample_info)
        self._viewer_info = dict(viewer_info)
        self._action = dict(action)
        self._sample_gripper = float(sample_gripper)
        self._loop_steps = 0
        self._terminated = False
        self._truncated = False
        self._exception: BaseException | None = None

    def update(
        self,
        *,
        sample_info: dict[str, Any],
        viewer_info: dict[str, Any],
        action: dict[str, Any],
        sample_gripper: float,
        loop_steps: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        with self._lock:
            self._sample_info = dict(sample_info)
            self._viewer_info = dict(viewer_info)
            self._action = dict(action)
            self._sample_gripper = float(sample_gripper)
            self._loop_steps = int(loop_steps)
            self._terminated = bool(terminated)
            self._truncated = bool(truncated)

    def set_exception(self, error: BaseException) -> None:
        with self._lock:
            self._exception = error

    def snapshot(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, int, bool, bool, BaseException | None]:
        with self._lock:
            return (
                dict(self._sample_info),
                dict(self._viewer_info),
                dict(self._action),
                float(self._sample_gripper),
                int(self._loop_steps),
                bool(self._terminated),
                bool(self._truncated),
                self._exception,
            )


def _zero_teleop_action(gripper: float) -> dict[str, float | bool]:
    return {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": float(gripper),
    }


def _resolve_dataset_root(path_value: str | Path, *, workspace: Path = _REPO_ROOT) -> Path:
    path = Path(path_value).expanduser()
    path_str = str(path)
    resolved_workspace = workspace.resolve()

    if path_str.startswith(f"{_CONTAINER_WORKSPACE}/"):
        return resolved_workspace / path_str.removeprefix(f"{_CONTAINER_WORKSPACE}/")
    if path_str.startswith(f"{_LEGACY_CONTAINER_WORKSPACE}/"):
        return resolved_workspace / path_str.removeprefix(f"{_LEGACY_CONTAINER_WORKSPACE}/")
    if path.is_absolute():
        return path
    return resolved_workspace / path


def _chmod_dataset_tree(root: Path, mode: int = 0o777) -> None:
    if not root.exists():
        return

    failed_paths: list[tuple[Path, OSError]] = []
    for path in (root, *root.rglob("*")):
        try:
            path.chmod(mode)
        except OSError as exc:
            failed_paths.append((path, exc))

    if failed_paths:
        first_path, first_error = failed_paths[0]
        logging.warning(
            "Failed to chmod %s dataset paths under %s to %s. First failure: %s: %s",
            len(failed_paths),
            root,
            oct(mode),
            first_path,
            first_error,
        )


def _get_env_observation(info: dict, gripper_pos: float, camera_obs: dict[str, np.ndarray] | None = None) -> dict:
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
    if camera_obs is None:
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


def _sample_disk_xy(center_xy: np.ndarray, radius_m: float, rng: np.random.Generator) -> np.ndarray:
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2)
    sample_radius = float(radius_m) * np.sqrt(rng.uniform(0.0, 1.0))
    sample_angle = rng.uniform(0.0, 2.0 * np.pi)
    offset = np.array([np.cos(sample_angle), np.sin(sample_angle)], dtype=np.float64) * sample_radius
    return center_xy + offset


def _reset_workspace_object_for_episode(
    env: FR3MujocoEnv,
    rng: np.random.Generator,
    *,
    random_radius_m: float = _WORKSPACE_OBJECT_RANDOM_RADIUS_M,
) -> np.ndarray:
    mujoco = env._mujoco
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, _WORKSPACE_OBJECT_BODY_NAME)
    if body_id < 0:
        raise ValueError(f"Body '{_WORKSPACE_OBJECT_BODY_NAME}' not found in MuJoCo model.")

    joint_id = int(env.model.body_jntadr[body_id])
    if joint_id < 0:
        raise ValueError(f"Body '{_WORKSPACE_OBJECT_BODY_NAME}' has no associated joint to reset.")

    qpos_adr = int(env.model.jnt_qposadr[joint_id])
    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    initial_pos = np.asarray(env.model.body_pos[body_id], dtype=np.float64).copy()
    initial_quat = np.asarray(env.model.body_quat[body_id], dtype=np.float64).copy()
    sampled_xy = _sample_disk_xy(initial_pos[:2], random_radius_m, rng)
    sampled_pos = initial_pos.copy()
    sampled_pos[:2] = sampled_xy

    with env._physics_lock:
        env.data.qpos[qpos_adr : qpos_adr + 3] = sampled_pos
        env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = initial_quat
        env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
    return sampled_pos


def _wait_for_teleop_idle(teleop, *, play_sounds: bool) -> None:
    wait_until_idle = getattr(teleop, "wait_until_idle", None)
    if not callable(wait_until_idle):
        return

    logging.info("Waiting for teleop input to return to idle before starting the episode.")
    log_say("Release teleop input", play_sounds)
    if not wait_until_idle(consecutive_samples=3):
        raise RuntimeError("Teleop input did not return to idle before starting the episode.")


def _run_episode_control_loop(
    *,
    env: FR3MujocoEnv,
    teleop,
    fps: int,
    initial_info: dict[str, Any],
    initial_gripper: float,
    shared_state: _LatestEpisodeControlState,
    stop_event: threading.Event,
) -> None:
    info = dict(initial_info)
    current_gripper = float(initial_gripper)
    loop_steps = 0
    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            action = teleop.get_action()
            sample_info = dict(info)
            sample_gripper = current_gripper
            _, _, terminated, truncated, next_info = env.step_teleop_action(
                action,
                control_period_s=1.0 / fps,
                include_camera_obs_in_observation=False,
                include_camera_obs_in_info=False,
            )
            loop_steps += 1
            shared_state.update(
                sample_info=sample_info,
                viewer_info=next_info,
                action=action,
                sample_gripper=sample_gripper,
                loop_steps=loop_steps,
                terminated=terminated,
                truncated=truncated,
            )
            info = dict(next_info)
            current_gripper = float(action.get("gripper", current_gripper))

            if terminated or truncated:
                stop_event.set()
                break

            sleep_s = max(1.0 / fps - (time.perf_counter() - loop_start), 0.0)
            if sleep_s > 0.0:
                stop_event.wait(sleep_s)
    except BaseException as exc:
        shared_state.set_exception(exc)
        stop_event.set()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    runtime_args = _RUNTIME_ARGS or parse_runtime_args(
        [],
        description="Run FR3 MuJoCo simulation recording with the default ee2ee dataset contract.",
    )[0]
    init_logging()
    mujoco_gl_backend = configure_mujoco_gl_backend(runtime_args)
    logging.info(
        pformat(
            {
                "record_cfg": asdict(cfg),
                "runtime": {
                    "viewer": not runtime_args.no_viewer,
                    "viewer_camera": runtime_args.viewer_camera,
                    "enable_cameras": runtime_args.enable_cameras,
                    "camera_width": runtime_args.camera_width,
                    "camera_height": runtime_args.camera_height,
                    "camera_fps": runtime_args.camera_fps,
                    "mujoco_gl": mujoco_gl_backend,
                    "teleop_type": getattr(cfg.teleop, "type", getattr(runtime_args, "teleop_type", None)),
                    "quest3_scene_mode": getattr(runtime_args, "quest3_scene_mode", None),
                    "tool_mode": runtime_args.tool_mode,
                    "motion_enable_button": runtime_args.motion_enable_button,
                    "enable_rotation": runtime_args.enable_rotation,
                },
            }
        )
    )

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
    dataset_root = _resolve_dataset_root(cfg.dataset.root)
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

    control_fps = int(runtime_args.fps)
    target_episode_frames = int(round(float(cfg.dataset.episode_time_s) * float(cfg.dataset.fps)))
    if target_episode_frames <= 0:
        raise ValueError(
            "dataset.episode_time_s and dataset.fps must produce at least one frame "
            f"(got episode_time_s={cfg.dataset.episode_time_s}, fps={cfg.dataset.fps})."
        )
    teleop_cfg = build_runtime_teleop_config(
        runtime_args,
        frequency=control_fps,
        base_config=cfg.teleop,
    )
    teleop = make_teleoperator_from_config(teleop_cfg)
    env = build_runtime_env(
        runtime_args,
        teleop_cfg,
        control_frequency=control_fps,
        # Dataset length is owned by the fixed-frame capture loop below. Keep a generous
        # simulation step budget so slow rendering does not truncate the episode early.
        max_episode_steps=max(int(cfg.dataset.episode_time_s * control_fps * 3) + control_fps, 1000),
    )

    marker_style = build_runtime_marker_style(runtime_args)
    viewer = None
    viewer_data = None
    selected_camera_name = None
    http_server = None
    latest_frame = None
    screen = None
    camera_renderer = None
    camera_render_data = None
    cv2_module = None
    viewer_period_s = 1.0 / 60.0
    next_viewer_sync = time.perf_counter()

    listener, events = None, None
    rng = np.random.default_rng()

    try:
        import mujoco
        import mujoco.viewer

        if not runtime_args.no_viewer:
            viewer_data = mujoco.MjData(env.model)
            env.copy_visual_state(viewer_data)
            viewer = mujoco.viewer.launch_passive(env.model, viewer_data)
            selected_camera_name = configure_viewer_camera(mujoco, viewer, env, runtime_args.viewer_camera)
        if runtime_args.enable_cameras:
            import cv2

            cv2_module = cv2
            http_server, latest_frame, screen = _start_camera_stream_outputs(
                camera_width=int(runtime_args.camera_width),
                camera_height=int(runtime_args.camera_height),
                camera_names=tuple(env.cfg.camera_names),
            )
            camera_renderer = env._mujoco.Renderer(
                env.model,
                height=int(runtime_args.camera_height),
                width=int(runtime_args.camera_width),
            )
            camera_render_data = env._mujoco.MjData(env.model)
        print("fr3_mujoco_record=READY")
        print(
            pformat(
                {
                    "camera_names": env.cfg.camera_names,
                    "viewer_camera": selected_camera_name,
                    "camera_stream": bool(runtime_args.enable_cameras),
                }
            )
        )
    except Exception as e:
        logging.warning(f"Could not launch viewer: {e}")

    try:
        teleop.connect()
        listener, events = init_keyboard_listener()

        recorded_episodes = 0
        current_gripper = 1.0

        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            _, info = env.reset(
                include_camera_obs_in_observation=False,
                include_camera_obs_in_info=False,
            )
            sampled_object_pos = _reset_workspace_object_for_episode(env, rng)
            info = env._build_info()
            logging.info(
                "Episode %s workspace_object reset to x=%.4f, y=%.4f, z=%.4f within %.3fm radius.",
                dataset.num_episodes,
                sampled_object_pos[0],
                sampled_object_pos[1],
                sampled_object_pos[2],
                _WORKSPACE_OBJECT_RANDOM_RADIUS_M,
            )
            sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
            if callable(sync_gripper) and "gripper_command" in info:
                sync_gripper(float(info["gripper_command"]))
                current_gripper = float(info["gripper_command"])
            _wait_for_teleop_idle(teleop, play_sounds=cfg.play_sounds)

            shared_state = _LatestEpisodeControlState(
                sample_info=info,
                viewer_info=info,
                action=_zero_teleop_action(current_gripper),
                sample_gripper=current_gripper,
            )
            episode_stop_event = threading.Event()
            control_thread = threading.Thread(
                target=_run_episode_control_loop,
                name="fr3-record-control",
                daemon=True,
                kwargs={
                    "env": env,
                    "teleop": teleop,
                    "fps": control_fps,
                    "initial_info": info,
                    "initial_gripper": current_gripper,
                    "shared_state": shared_state,
                    "stop_event": episode_stop_event,
                },
            )
            control_thread.start()

            if viewer is not None:
                copy_visual_state = getattr(env, "copy_visual_state", None)
                if viewer_data is not None and callable(copy_visual_state):
                    copy_visual_state(viewer_data)
                with viewer.lock():
                    update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
                viewer.sync()
                next_viewer_sync = time.perf_counter() + viewer_period_s

            start_episode_t = time.perf_counter()
            dataset_frame_period_s = 1 / dataset.fps
            captured_dataset_frames = 0
            max_capture_lag_s = 0.0
            try:
                for dataset_frame_idx in range(target_episode_frames):
                    target_capture_t = start_episode_t + dataset_frame_idx * dataset_frame_period_s

                    if events["exit_early"]:
                        events["exit_early"] = False
                        episode_stop_event.set()
                        break

                    episode_finished = False
                    while not events["stop_recording"]:
                        sample_info, viewer_info, action, sample_gripper, loop_steps, terminated, truncated, error = (
                            shared_state.snapshot()
                        )
                        if error is not None:
                            raise error

                        if terminated or truncated:
                            episode_stop_event.set()
                            episode_finished = True
                            break

                        now = time.perf_counter()

                        if viewer is not None and not viewer.is_running():
                            episode_stop_event.set()
                            episode_finished = True
                            break
                        if viewer is not None and now >= next_viewer_sync:
                            copy_visual_state = getattr(env, "copy_visual_state", None)
                            if viewer_data is not None and callable(copy_visual_state):
                                copy_visual_state(viewer_data)
                            with viewer.lock():
                                update_passive_viewer_markers(env._mujoco, viewer, viewer_info, marker_style)
                            viewer.sync()
                            next_viewer_sync = now + viewer_period_s

                        if time.perf_counter() + 1e-9 >= target_capture_t:
                            break

                        next_deadline = target_capture_t
                        if viewer is not None:
                            next_deadline = min(next_deadline, next_viewer_sync)
                        precise_sleep(max(min(next_deadline - time.perf_counter(), 0.01), 0.001))

                    if events["stop_recording"] or episode_finished:
                        break

                    sample_info, viewer_info, action, sample_gripper, loop_steps, terminated, truncated, error = (
                        shared_state.snapshot()
                    )
                    if error is not None:
                        raise error
                    if terminated or truncated:
                        episode_stop_event.set()
                        break

                    capture_lag_s = max(time.perf_counter() - target_capture_t, 0.0)
                    max_capture_lag_s = max(max_capture_lag_s, capture_lag_s)
                    if capture_lag_s > dataset_frame_period_s and dataset_frame_idx % max(int(dataset.fps), 1) == 0:
                        logging.warning(
                            "FR3 MuJoCo dataset capture is %.3fs behind the nominal %.3fs frame time "
                            "(frame %s/%s). The episode will keep writing frames to preserve "
                            "dataset.episode_time_s.",
                            capture_lag_s,
                            dataset_frame_period_s,
                            dataset_frame_idx + 1,
                            target_episode_frames,
                        )

                    camera_obs = None
                    if (
                        runtime_args.enable_cameras
                        and camera_renderer is not None
                        and camera_render_data is not None
                    ):
                        env.copy_visual_state(camera_render_data)
                        camera_obs = _render_camera_frames(
                            mujoco=env._mujoco,
                            renderer=camera_renderer,
                            render_data=camera_render_data,
                            camera_names=tuple(env.cfg.camera_names),
                            camera_name_mapping=dict(env.cfg.camera_name_mapping),
                        )
                        if latest_frame is not None:
                            grid = render_camera_grid(
                                camera_obs,
                                int(runtime_args.camera_width),
                                int(runtime_args.camera_height),
                                tuple(env.cfg.camera_names),
                            )
                            latest_frame.set(_encode_grid_jpeg(grid, cv2_module))
                            if screen is not None:
                                try:
                                    import pygame

                                    surf = pygame.surfarray.make_surface(np.transpose(grid, (1, 0, 2)))
                                    screen.fill((0, 0, 0))
                                    screen.blit(surf, (0, 0))
                                    pygame.display.flip()
                                    for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            events["stop_recording"] = True
                                except Exception:
                                    screen = None
                    obs_raw = _get_env_observation(sample_info, gripper_pos=sample_gripper, camera_obs=camera_obs)
                    obs_processed = robot_observation_processor(obs_raw)
                    obs_completed = _complete_robot_observation(obs_raw)
                    action_processed = teleop_action_processor((action.copy(), obs_raw))

                    observation_frame = build_dataset_frame(
                        dataset.features,
                        {**obs_completed, **obs_processed},
                        prefix=OBS_STR,
                    )
                    action_frame = build_dataset_frame(dataset.features, action_processed, prefix=ACTION)
                    frame = {**observation_frame, **action_frame, "task": cfg.dataset.single_task}
                    dataset.add_frame(frame)
                    captured_dataset_frames += 1

                if captured_dataset_frames != target_episode_frames:
                    logging.warning(
                        "Episode captured %s/%s dataset frames before stopping.",
                        captured_dataset_frames,
                        target_episode_frames,
                    )
                elif max_capture_lag_s > dataset_frame_period_s:
                    logging.warning(
                        "Episode preserved the target %s dataset frames, but capture lag peaked at %.3fs.",
                        target_episode_frames,
                        max_capture_lag_s,
                    )
            finally:
                episode_stop_event.set()
                control_thread.join(timeout=1.0)
                _, info, _, current_gripper, _, _, _, error = shared_state.snapshot()
                if error is not None:
                    raise error

            keep_episode = _confirm_keep_episode(cfg.play_sounds)
            if keep_episode:
                dataset.save_episode()
                recorded_episodes += 1
            else:
                logging.info("Discarding recorded episode %s.", dataset.num_episodes)
                dataset.clear_episode_buffer()

            teleop_action_processor.reset()
            robot_observation_processor.reset()

        dataset.finalize()
        _chmod_dataset_tree(dataset_root)

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if http_server is not None:
            http_server.shutdown()
            server_close = getattr(http_server, "server_close", None)
            if callable(server_close):
                server_close()
        if camera_renderer is not None:
            camera_renderer.close()
        if screen is not None:
            import pygame

            pygame.quit()
        if viewer is not None:
            viewer.close()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        env.close()

    return dataset


def main(argv: list[str] | None = None) -> None:
    runtime_args, remaining = parse_runtime_args(
        argv,
        description="Run FR3 MuJoCo simulation recording with the default ee2ee dataset contract.",
    )
    global _RUNTIME_ARGS
    _RUNTIME_ARGS = runtime_args
    sys.argv = [sys.argv[0], *remaining]
    record()


if __name__ == "__main__":
    main()
