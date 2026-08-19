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

"""Replay a recorded FR3 ee2ee episode and score how well the arm tracked it.

``--backend sim`` (the default) is the safety gate in front of real-robot replay: the recorded
``action`` stream is fed back through the *same* :class:`FrankaResearch3Mujoco` robot the sim
recorder uses, and each commanded pose is compared against what the simulated arm actually
reached. If the arm cannot follow the trajectory in simulation -- unreachable pose, joint limit,
IK failure -- it will not follow it on hardware either, and the operator finds out without moving
a real arm.

``--backend real`` runs the identical trajectory, approach phase and scoring against the hardware
``FrankaResearch3`` from the same config. One runtime rather than two is the point: a real replay
that reconstructed its trajectory differently from the run that cleared it would validate one
thing and execute another. The differences are only the ones that must differ -- there is nothing
to render, and the arm is asked to refuse rather than lunge when it cannot reach the start pose.

Both report on the gateway's ``mujoco_replay_result=`` protocol line, so the Episode Replay
page's validation gate works unchanged for workstation datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.robots.franka_research3.processor_franka_research3 import (  # noqa: E402
    DELTA_REFERENCE_CURRENT,
    PREV_CMD_POSITION_KEYS,
    PREV_CMD_QUAT_KEYS,
    delta_ee_position_keys,
    delta_ee_rotvec_keys,
    delta_reference_from_action_names,
)
from lerobot.utils.rotation import Rotation  # noqa: E402

from tools.fr3.fr3_gui_record_runtime import build_sim_robot_config, _resolve_workspace_path  # noqa: E402
from tools.fr3.fr3_tool_frame_check import (  # noqa: E402
    ToolFrameError,
    require_frame,
    workspace_clip_report,
)

ACTION_FEATURE = "action"
OBSERVATION_FEATURE = "observation.state"
DEFAULT_MAX_POSITION_ERROR_MM = 20.0
DEFAULT_MAX_ROTATION_ERROR_DEG = 15.0
_PROGRESS_EVERY = 25


def emit(line: str) -> None:
    print(line, flush=True)


def _preflight_contract(robot_cfg: Any, action_positions: Any) -> list[str]:
    """Refuse to replay a dataset the config cannot mean, before the arm is even connected.

    Replay reads recorded poses and commands them. Two properties of the config decide what those
    numbers mean, and neither is recorded in the dataset:

    * ``target_frame_name`` -- the two FR3 tool frames are 410.85 mm apart and share an orientation,
      so replaying one dataset under the other frame does not fail. It drives the fingertips to
      where the other frame's origin used to be, for the whole episode.
    * ``workspace_min``/``workspace_max`` -- ``_make_pose_from_absolute_action`` clips absolute
      replay commands too, so a box that no longer contains the recorded trajectory does not stop
      the replay. It flattens it against a wall and then the tracking score is computed against the
      flattened version, which is a number about a trajectory nobody recorded.

    Both used to be silent. This is the check the prose in WORKSTATION_RECORDING.md was standing in
    for; a comment is not a mechanism.
    """
    positions = np.asarray(action_positions, dtype=np.float64)
    configured_frame = str(getattr(robot_cfg, "target_frame_name", "") or "")

    if configured_frame:
        try:
            errors = require_frame("this episode", positions[0], configured_frame)
        except ToolFrameError as exc:
            emit(f"ERROR: {exc}")
            emit(
                "Convert the dataset with tools/fr3/fr3_migrate_tool_frame.py, or replay it against "
                "a config whose robot.target_frame_name matches the frame it was recorded in."
            )
            return [str(exc)]
        emit(f"Tool frame: {configured_frame} (episode start {errors[configured_frame] * 1e3:.1f} mm from home)")

    workspace_min = getattr(robot_cfg, "workspace_min", None)
    workspace_max = getattr(robot_cfg, "workspace_max", None)
    if workspace_min is not None and workspace_max is not None:
        outside, worst_m, details = workspace_clip_report(positions, workspace_min, workspace_max)
        if outside:
            reason = (
                f"robot.workspace_min/max would clip {outside} of {len(positions)} commanded poses "
                f"({100.0 * outside / max(len(positions), 1):.1f}%), displacing them by up to "
                f"{worst_m * 1e3:.1f} mm: {'; '.join(details)}"
            )
            emit(f"ERROR: {reason}.")
            emit(
                "The clip applies to absolute replay commands, so this would not fail -- it would "
                "reshape the trajectory and then score the reshaped one. Widen the box for this "
                "replay, or replay a dataset the box was derived for."
            )
            return [reason]

    return []


def _replay_robot_config(record_cfg: Any, args: argparse.Namespace, fps: int) -> Any:
    real_backend = args.backend == "real"
    if real_backend:
        robot_cfg = record_cfg.robot
        if args.robot_ip:
            robot_cfg.robot_ip = args.robot_ip
        robot_cfg.urdf_path = _resolve_workspace_path(str(getattr(robot_cfg, "urdf_path", "") or ""))
        if args.ik_orientation_weight is not None:
            robot_cfg.ik_orientation_weight = float(args.ik_orientation_weight)
        # Cameras are the recorder's business. Opening them here would take the RealSense units
        # away from the live monitor the operator is watching the arm through, and this loop
        # never reads an image.
        robot_cfg.cameras = {}
        return robot_cfg

    robot_cfg = build_sim_robot_config(record_cfg)
    # Each replayed frame must be given the same amount of simulated time the recorder had
    # between frames. Leaving this at the recorder's control frequency would advance physics
    # by a fraction of a frame period per command and report tracking error that is really
    # just an under-integrated servo window.
    robot_cfg.teleop_control_frequency = float(max(fps, 1))
    if args.ik_orientation_weight is not None:
        robot_cfg.ik_orientation_weight = float(args.ik_orientation_weight)
    return robot_cfg


def _settle_to_start_pose(
    robot: Any,
    *,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    target_gripper: float,
    settle_steps: int,
    settle_tolerance_mm: float,
    settle_period_s: float,
) -> float:
    settle_target = {
        "ee.x": float(target_position[0]),
        "ee.y": float(target_position[1]),
        "ee.z": float(target_position[2]),
        **dict(
            zip(
                ("ee.wx", "ee.wy", "ee.wz"),
                (float(v) for v in Rotation.from_quat(target_quaternion).as_rotvec()),
                strict=True,
            )
        ),
        "gripper.pos": float(target_gripper),
    }
    settle_error_mm = float("inf")
    total_steps = max(int(settle_steps), 0)
    for step_index in range(total_steps):
        loop_start_s = time.perf_counter()
        robot.send_action(settle_target)
        settle_observation = robot.get_observation(include_cameras=False)
        settle_error_mm = float(
            np.linalg.norm(
                np.array(
                    [
                        settle_observation["ee.x"],
                        settle_observation["ee.y"],
                        settle_observation["ee.z"],
                    ],
                    dtype=np.float64,
                )
                - target_position
            )
            * 1e3
        )
        if settle_error_mm <= settle_tolerance_mm:
            break
        if settle_period_s > 0.0 and step_index + 1 < total_steps:
            remaining_s = float(settle_period_s) - (time.perf_counter() - loop_start_s)
            if remaining_s > 0.0:
                time.sleep(remaining_s)
    return settle_error_mm


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        raise KeyError(f"Dataset is missing the '{key}' feature.")
    names = feature.get("names")
    if not isinstance(names, list):
        raise KeyError(f"Dataset feature '{key}' has no component names.")
    return [str(name) for name in names]


def load_episode_actions(dataset_root: Path, episode: int) -> dict[str, Any]:
    """Read one episode's action stream and recorded EE observations out of the parquet files."""
    import pyarrow.parquet as pq

    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}.")

    info = _load_info(dataset_root)
    action_names = _feature_names(info, ACTION_FEATURE)
    observation_names = _feature_names(info, OBSERVATION_FEATURE)

    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    for parquet_file in parquet_files:
        table = pq.read_table(
            parquet_file, columns=["episode_index", ACTION_FEATURE, OBSERVATION_FEATURE]
        )
        episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
        mask = episode_indices == int(episode)
        if not mask.any():
            continue
        action_rows = np.asarray(table[ACTION_FEATURE].to_pylist(), dtype=np.float64)
        observation_rows = np.asarray(table[OBSERVATION_FEATURE].to_pylist(), dtype=np.float64)
        actions.append(action_rows[mask])
        observations.append(observation_rows[mask])

    if not actions:
        raise ValueError(f"Episode {episode} has no frames in {dataset_root}.")
    return {
        "actions": np.concatenate(actions, axis=0),
        "observations": np.concatenate(observations, axis=0),
        "action_names": action_names,
        "observation_names": observation_names,
        "fps": int(info.get("fps") or 30),
    }


def _column(names: list[str], values: np.ndarray, key: str) -> np.ndarray | None:
    if key not in names:
        return None
    return values[:, names.index(key)]


def _pose_stream(names: list[str], values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract (positions Nx3, quaternions Nx4 xyzw) from a flattened feature block."""
    position_keys = ("ee.x", "ee.y", "ee.z")
    quat_keys = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
    columns = [_column(names, values, key) for key in position_keys]
    if any(column is None for column in columns):
        return None
    positions = np.stack(columns, axis=1)
    quat_columns = [_column(names, values, key) for key in quat_keys]
    if any(column is None for column in quat_columns):
        return None
    return positions, np.stack(quat_columns, axis=1)


def _rotation_error_deg(target_xyzw: np.ndarray, actual_xyzw: np.ndarray) -> float:
    dot = float(np.dot(target_xyzw, actual_xyzw))
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0))))


def _delta_stream(names: list[str], values: np.ndarray, reference: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation deltas Nx3, rotvec deltas Nx3) from a delta action block."""
    position_columns = [_column(names, values, key) for key in delta_ee_position_keys(reference)]
    rotvec_columns = [_column(names, values, key) for key in delta_ee_rotvec_keys(reference)]
    if any(column is None for column in position_columns + rotvec_columns):
        raise ValueError(f"Action column is missing some {reference!r} delta components.")
    return np.stack(position_columns, axis=1), np.stack(rotvec_columns, axis=1)


def reconstruct_absolute_pose_stream(
    *,
    action_names: list[str],
    actions: np.ndarray,
    observation_names: list[str],
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Absolute (positions, quaternions, source-description) for either action contract.

    An ``absolute_ee`` dataset is read straight out of the action column. A delta dataset is
    rebuilt using the same conventions the recorder used -- world-frame translation, body-frame
    right-multiplied rotation -- with the reference taken from the observation column:

    * ``delta_ee_from_prev_cmd``: the reference is the previous *command*, which during replay is
      the pose we ourselves issued on the previous frame, so the trajectory is the cumulative
      integral of the deltas seeded by frame 0's recorded ``prev_cmd``.
    * ``delta_ee_from_current``: the reference is the *measured* pose on that frame, which the
      dataset stores, so each target is independent -- no integration, no drift.

    Which delta mode produced the dataset is inferred from what the observation actually carries,
    and the choice is reported rather than assumed, because integrating with the wrong reference
    would silently produce a plausible-looking but different trajectory.
    """
    absolute = _pose_stream(action_names, actions)
    if absolute is not None:
        return absolute[0], absolute[1], "absolute_ee action column"

    reference = delta_reference_from_action_names(action_names)
    if reference is None:
        raise ValueError(
            "Dataset action stream is neither absolute EE (ee.x/y/z + ee.qx/qy/qz/qw) nor any "
            f"known delta EE contract; got action names {action_names}."
        )
    delta_positions, delta_rotvecs = _delta_stream(action_names, actions, reference)
    total_frames = int(delta_positions.shape[0])

    measured = _pose_stream(observation_names, observations)
    if measured is None:
        raise ValueError(
            "Rebuilding a delta action stream needs the absolute EE pose in observation.state; "
            f"got observation names {observation_names}."
        )
    measured_positions, measured_quaternions = measured

    positions = np.zeros((total_frames, 3), dtype=np.float64)
    quaternions = np.zeros((total_frames, 4), dtype=np.float64)

    if reference == DELTA_REFERENCE_CURRENT:
        # Reference is the measured pose on the same frame, which the dataset stores. Each target
        # is independent -- no integration, so no accumulated drift.
        for frame_index in range(total_frames):
            rotation = (
                Rotation.from_quat(measured_quaternions[frame_index]).as_matrix()
                @ Rotation.from_rotvec(delta_rotvecs[frame_index]).as_matrix()
            )
            positions[frame_index] = measured_positions[frame_index] + delta_positions[frame_index]
            quaternions[frame_index] = Rotation.from_matrix(rotation).as_quat()
        return positions, quaternions, "delta_ee_from_current, per-frame measured reference"

    # delta_ee_from_prev_cmd: the reference is the pose commanded on the previous frame, which
    # during replay is the pose we ourselves issue, so the trajectory is the cumulative integral
    # of the deltas seeded by frame 0's recorded prev_cmd.
    prev_cmd_positions = [_column(observation_names, observations, key) for key in PREV_CMD_POSITION_KEYS]
    prev_cmd_quaternions = [_column(observation_names, observations, key) for key in PREV_CMD_QUAT_KEYS]
    if not any(column is None for column in prev_cmd_positions + prev_cmd_quaternions):
        seed_position = np.stack(prev_cmd_positions, axis=1)[0].astype(np.float64)
        seed_rotation = Rotation.from_quat(np.stack(prev_cmd_quaternions, axis=1)[0]).as_matrix()
        source = "delta_ee_from_prev_cmd, integrated from observation prev_cmd at frame 0"
    else:
        # No recorded command reference. Seeding from the measured pose is a *different*
        # reference, so say so loudly instead of quietly producing a shifted trajectory.
        seed_position = measured_positions[0].astype(np.float64)
        seed_rotation = Rotation.from_quat(measured_quaternions[0]).as_matrix()
        source = "delta_ee_from_prev_cmd, seeded from the measured pose (no prev_cmd recorded)"
        emit(
            "WARN: dataset records no prev_cmd.ee.* in observation.state; seeding the delta "
            "integration from the measured pose instead. The rebuilt trajectory is offset by the "
            "frame-0 command-vs-measured residual."
        )

    reference_position = seed_position
    reference_rotation = seed_rotation
    for frame_index in range(total_frames):
        reference_position = reference_position + delta_positions[frame_index]
        reference_rotation = reference_rotation @ Rotation.from_rotvec(delta_rotvecs[frame_index]).as_matrix()
        positions[frame_index] = reference_position
        quaternions[frame_index] = Rotation.from_matrix(reference_rotation).as_quat()
    return positions, quaternions, source


def replay_episode(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset.resolve()
    episode_data = load_episode_actions(dataset_root, args.episode)
    actions = episode_data["actions"]
    action_names = episode_data["action_names"]
    total_frames = int(actions.shape[0])

    action_positions, action_quaternions, action_source = reconstruct_absolute_pose_stream(
        action_names=action_names,
        actions=actions,
        observation_names=episode_data["observation_names"],
        observations=episode_data["observations"],
    )
    emit(f"Action source: {action_source}")
    action_gripper = _column(action_names, actions, "gripper.pos")

    from lerobot.scripts.lerobot_record import RecordConfig  # noqa: F401  (config typing only)
    import draccus

    with open(args.config_path) as config_file:
        record_cfg = draccus.load(RecordConfig, config_file)
    fps = int(args.fps or episode_data["fps"] or 30)
    real_backend = args.backend == "real"
    robot_cfg = _replay_robot_config(record_cfg, args, fps)
    contract_failures = _preflight_contract(robot_cfg, action_positions)
    if contract_failures:
        # A verdict about the *contract*, not about the trajectory: no arm was connected and no pose
        # was tracked, so every error field stays at zero and `reasons` carries what went wrong. The
        # gateway's parser sees a well-formed failure instead of a missing result line.
        return {
            "schema_version": 1,
            "dataset": str(dataset_root),
            "episode": int(args.episode),
            "fps": fps,
            "status": "failed",
            "reasons": contract_failures,
            "completed_frames": 0,
            "total_frames": total_frames,
            "avg_position_error_mm": 0.0,
            "max_position_error_mm": 0.0,
            "avg_rotation_error_deg": 0.0,
            "max_rotation_error_deg": 0.0,
            "limits": {
                "max_position_error_mm": float(args.max_position_error_mm),
                "max_rotation_error_deg": float(args.max_rotation_error_deg),
            },
            "backend": str(args.backend),
            "robot_type": "",
            "video_frames": 0,
            "native_video_path": "",
        }
    robot = make_robot_from_config(robot_cfg)

    frame_period_s = 1.0 / max(fps, 1)
    position_errors_mm: list[float] = []
    rotation_errors_deg: list[float] = []
    completed_frames = 0
    video = _PreviewVideoWriter(args.render_video, fps) if args.render_video else None

    emit(
        f"fr3_mujoco_replay dataset={dataset_root.name} episode={args.episode} "
        f"frames={total_frames} fps={fps} backend={args.backend}"
    )
    if real_backend:
        emit(f"Real-robot replay on {getattr(robot_cfg, 'robot_ip', '?')}: the arm will move.")
    robot.connect()
    try:
        # Unscored approach phase: the arm starts wherever it is, which is not where the
        # recording began. Without this, frame 0 is scored against a homing jump and every
        # episode "fails" for a reason that has nothing to do with the trajectory.
        settle_error_mm = _settle_to_start_pose(
            robot,
            target_position=action_positions[0],
            target_quaternion=action_quaternions[0],
            target_gripper=float(action_gripper[0]) if action_gripper is not None else 1.0,
            settle_steps=args.settle_steps,
            settle_tolerance_mm=args.settle_tolerance_mm,
            settle_period_s=frame_period_s if args.realtime else 0.0,
        )
        emit(f"Approach to episode start pose: residual {settle_error_mm:.2f} mm")
        if settle_error_mm > args.settle_tolerance_mm:
            # Reported, not silently folded into the trajectory score: an arm that cannot even
            # reach the start pose says something different from one that drifts mid-episode.
            message = (
                f"could not reach the recorded start pose within {args.settle_steps} steps "
                f"({settle_error_mm:.2f} mm > {args.settle_tolerance_mm:.2f} mm)"
            )
            if real_backend:
                # On hardware this is where it stops. Frame 0 of a trajectory whose start the arm
                # never reached is a step of unknown size taken at full command rate, and the
                # thing that absorbs it is the hardware.
                raise RuntimeError(
                    f"Refusing to replay on the real robot: {message}. The arm is parked where the "
                    "approach left it; nothing from the recorded trajectory was sent."
                )
            emit(f"WARN: {message}; the trajectory may be outside the simulated workspace")

        for frame_index in range(total_frames):
            target_position = action_positions[frame_index]
            target_quaternion = action_quaternions[frame_index]
            target_rotvec = Rotation.from_quat(target_quaternion).as_rotvec()
            command = {
                "ee.x": float(target_position[0]),
                "ee.y": float(target_position[1]),
                "ee.z": float(target_position[2]),
                "ee.wx": float(target_rotvec[0]),
                "ee.wy": float(target_rotvec[1]),
                "ee.wz": float(target_rotvec[2]),
                "gripper.pos": float(action_gripper[frame_index]) if action_gripper is not None else 1.0,
            }
            loop_start_s = time.perf_counter()
            robot.send_action(command)
            observation = robot.get_observation(include_cameras=False)

            actual_position = np.array(
                [observation["ee.x"], observation["ee.y"], observation["ee.z"]], dtype=np.float64
            )
            actual_quaternion = Rotation.from_rotvec(
                [observation["ee.wx"], observation["ee.wy"], observation["ee.wz"]]
            ).as_quat()
            position_errors_mm.append(float(np.linalg.norm(actual_position - target_position) * 1e3))
            rotation_errors_deg.append(_rotation_error_deg(target_quaternion, actual_quaternion))
            completed_frames += 1
            if video is not None:
                video.append(_preview_frame(robot, video))

            if frame_index % _PROGRESS_EVERY == 0 or frame_index == total_frames - 1:
                emit(f"Replayed {completed_frames}/{total_frames} frames")
            if args.realtime:
                remaining_s = frame_period_s - (time.perf_counter() - loop_start_s)
                if remaining_s > 0:
                    time.sleep(remaining_s)
    finally:
        if video is not None:
            video.close()
        if robot.is_connected:
            robot.disconnect()

    if video is not None:
        if video.error is not None:
            emit(f"WARN: preview video not written ({video.error}); the replay metrics are unaffected")
        elif video.frames_written:
            emit(f"fr3_mujoco_replay_video={video.destination}")

    avg_position_mm = float(np.mean(position_errors_mm)) if position_errors_mm else 0.0
    max_position_mm = float(np.max(position_errors_mm)) if position_errors_mm else 0.0
    avg_rotation_deg = float(np.mean(rotation_errors_deg)) if rotation_errors_deg else 0.0
    max_rotation_deg = float(np.max(rotation_errors_deg)) if rotation_errors_deg else 0.0
    passed = (
        completed_frames == total_frames
        and max_position_mm <= args.max_position_error_mm
        and max_rotation_deg <= args.max_rotation_error_deg
    )

    result = {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "episode": int(args.episode),
        "fps": fps,
        "status": "passed" if passed else "failed",
        "completed_frames": completed_frames,
        "total_frames": total_frames,
        "avg_position_error_mm": avg_position_mm,
        "max_position_error_mm": max_position_mm,
        "avg_rotation_error_deg": avg_rotation_deg,
        "max_rotation_error_deg": max_rotation_deg,
        "limits": {
            "max_position_error_mm": float(args.max_position_error_mm),
            "max_rotation_error_deg": float(args.max_rotation_error_deg),
        },
        "backend": str(args.backend),
        "robot_type": robot.name,
        "video_frames": int(video.frames_written) if video is not None else 0,
        "native_video_path": (
            str(video.destination) if video is not None and video.frames_written and not video.error else ""
        ),
    }
    return result


def fr3_mujoco_replay_report_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "derived" / "fr3_mujoco_replay" / f"episode_{int(episode):06d}.json"


def fr3_mujoco_replay_video_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "derived" / "fr3_mujoco_replay" / f"episode_{int(episode):06d}.mp4"


def fr3_real_replay_report_path(dataset_root: Path, episode: int) -> Path:
    """Separate from the sim report on purpose.

    The sim report is the verdict that authorized this run. Writing the hardware result over it
    would destroy the record of what was checked beforehand, and a second real replay would then
    be gated by the first one's outcome.
    """
    return dataset_root / "derived" / "fr3_real_replay" / f"episode_{int(episode):06d}.json"


class _PreviewVideoWriter:
    """Collects rendered frames and writes them as one mp4 beside the report.

    Every failure here is swallowed and reported once. The video is something to look at; the
    verdict is the numbers. Letting a missing encoder or a busy renderer take down the replay
    would trade the thing that matters for the thing that illustrates it.
    """

    def __init__(self, destination: Path, fps: int) -> None:
        self.destination = destination
        self.fps = max(int(fps), 1)
        self.frames_written = 0
        self.error: str | None = None
        self._writer: Any = None

    def _ensure_writer(self, frame: np.ndarray) -> Any:
        if self._writer is None:
            import imageio.v2 as imageio

            self.destination.parent.mkdir(parents=True, exist_ok=True)
            # yuv420p needs even dimensions, and a renderer height of 481 is not a reason to
            # produce a file no browser will decode.
            self._writer = imageio.get_writer(
                str(self.destination),
                fps=self.fps,
                codec="libx264",
                macro_block_size=2,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        return self._writer

    def append(self, frame: np.ndarray | None) -> None:
        if frame is None or self.error is not None:
            return
        try:
            self._ensure_writer(frame).append_data(np.ascontiguousarray(frame))
            self.frames_written += 1
        except Exception as exc:  # noqa: BLE001 - a preview must never fail the replay
            self.error = str(exc)
            self._writer = None

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as exc:  # noqa: BLE001
            self.error = self.error or str(exc)
        finally:
            self._writer = None


def _preview_frame(robot: Any, video: _PreviewVideoWriter) -> np.ndarray | None:
    """Render one preview frame, or record why not and stop asking.

    The encoder's failures were already swallowed, but the renderer's were not: on a machine
    where the offscreen GL context cannot be created (headless box, EGL falling back to a dri2
    screen it cannot open), the very first frame raised and took the whole validation with it --
    losing the verdict over its illustration. The metrics do not need pixels.
    """
    if video.error is not None:
        return None
    try:
        return robot.render_preview_frame()
    except Exception as exc:  # noqa: BLE001 - a preview must never fail the replay
        video.error = f"renderer unavailable: {exc}"
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an FR3 ee2ee episode in MuJoCo.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--config-path", dest="config_path", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--max-position-error-mm", type=float, default=DEFAULT_MAX_POSITION_ERROR_MM)
    parser.add_argument("--max-rotation-error-deg", type=float, default=DEFAULT_MAX_ROTATION_ERROR_DEG)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace the replay at the dataset frame rate instead of running as fast as possible.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=200,
        help="Maximum unscored control steps used to reach the recorded start pose.",
    )
    # 5 mm sits above the steady-state sag of the position-actuated sim arm (~3.7 mm observed
    # under 0.5x gravity compensation) and well below the 20 mm trajectory budget, so the
    # warning fires for genuinely unreachable start poses rather than for normal servo droop.
    parser.add_argument("--settle-tolerance-mm", type=float, default=5.0)
    parser.add_argument(
        "--ik-orientation-weight",
        type=float,
        default=None,
        help=(
            "Absolute-pose IK orientation weight for replay. "
            "Omit to keep the robot backend default."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--render-video",
        dest="render_video",
        type=Path,
        default=None,
        help="Write an mp4 of the simulated arm following the trajectory (preview only).",
    )
    parser.add_argument(
        "--backend",
        choices=("sim", "real"),
        default="sim",
        help="sim replays in MuJoCo and scores the trajectory; real moves the hardware arm.",
    )
    parser.add_argument(
        "--robot-ip",
        dest="robot_ip",
        default="",
        help="Override the config's robot_ip. Only meaningful with --backend real.",
    )
    args = parser.parse_args(argv)
    if args.backend == "real":
        if args.render_video is not None:
            parser.error("--render-video is a MuJoCo preview; there is nothing to render on hardware.")
        # A real arm has one clock and it is the wall clock. Streaming a 30 Hz trajectory as fast
        # as the loop can issue it would ask the hardware for motion the recording never made.
        args.realtime = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = replay_episode(args)
    default_report = (
        fr3_real_replay_report_path(args.dataset.resolve(), args.episode)
        if args.backend == "real"
        else fr3_mujoco_replay_report_path(args.dataset.resolve(), args.episode)
    )
    destination = args.output or default_report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if args.backend == "real":
        emit(
            "fr3_real_replay_result="
            f"status={result['status']} "
            f"completed_frames={result['completed_frames']} "
            f"total_frames={result['total_frames']} "
            f"avg_pos_mm={result['avg_position_error_mm']:.4f} "
            f"max_pos_mm={result['max_position_error_mm']:.4f} "
            f"avg_rot_deg={result['avg_rotation_error_deg']:.4f} "
            f"max_rot_deg={result['max_rotation_error_deg']:.4f}"
        )
        emit(f"fr3_real_replay_report={destination}")
        return 0 if result["status"] == "passed" else 1

    # Exact shape the gateway's validation parser expects; keep it byte-compatible.
    emit(
        "mujoco_replay_result="
        f"status={result['status']} "
        f"completed_frames={result['completed_frames']} "
        f"total_frames={result['total_frames']} "
        f"avg_pos_mm={result['avg_position_error_mm']:.4f} "
        f"max_pos_mm={result['max_position_error_mm']:.4f} "
        f"avg_rot_deg={result['avg_rotation_error_deg']:.4f} "
        f"max_rot_deg={result['max_rotation_error_deg']:.4f}"
    )
    emit(f"fr3_mujoco_replay_report={destination}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
