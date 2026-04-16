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

"""Smoke test the FR3 default ee2ee record path against the MuJoCo environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.rotation import Rotation
from tools.fr3.fr3_record_runtime import make_fr3_ee2ee_processors

_D435I_IMAGE_SHAPE = (480, 640, 3)
_SIM_CAMERA_NAMES = FR3MujocoEnvConfig().camera_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8, help="Number of teleop steps to record.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS used for the smoke dataset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/datasets/fr3_mujoco_ee2ee_smoke"),
        help="Output root for the temporary smoke dataset.",
    )
    parser.add_argument(
        "--keep-dataset",
        action="store_true",
        help="Keep the generated dataset on disk instead of deleting it after verification.",
    )
    parser.add_argument("--camera-width", type=int, default=_D435I_IMAGE_SHAPE[1], help="Sim camera width.")
    parser.add_argument("--camera-height", type=int, default=_D435I_IMAGE_SHAPE[0], help="Sim camera height.")
    return parser.parse_args()


def scripted_teleop_action(step_idx: int) -> dict[str, float | bool]:
    direction = 1.0 if step_idx % 2 == 0 else -1.0
    enabled = step_idx < 6
    return {
        "enabled": enabled,
        "target_x": 0.002 * direction if enabled else 0.0,
        "target_y": -0.001 * direction if enabled else 0.0,
        "target_z": 0.0015 if enabled else 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.01 * direction if enabled else 0.0,
        "gripper": 1.0 if step_idx < 4 else 0.5,
    }


def env_info_to_robot_observation(info: dict[str, object], *, gripper_pos: float) -> dict[str, object]:
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


def build_robot_observation_features(
    *,
    include_cameras: bool = False,
    camera_shape: tuple[int, int, int] = _D435I_IMAGE_SHAPE,
) -> dict[str, type | tuple[int, int, int]]:
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


def complete_robot_observation(observation: dict[str, object]) -> dict[str, object]:
    completed = dict(observation)
    for axis in ("qx", "qy", "qz", "qw"):
        completed.setdefault(f"prev_cmd.ee.{axis}", completed.get(f"ee.{axis}", 0.0))
    return completed


def main() -> int:
    args = parse_args()

    dataset_root = args.output_root.resolve()
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    record_cfg = RecordConfig(
        robot=FrankaResearch3Config(
            urdf_path=str(FR3MujocoEnvConfig().urdf_path),
            workspace_min=FR3MujocoEnvConfig().workspace_min,
            workspace_max=FR3MujocoEnvConfig().workspace_max,
            max_target_delta_pos=FR3MujocoEnvConfig().max_target_delta_pos,
            max_target_delta_rot=FR3MujocoEnvConfig().max_target_delta_rot,
        ),
        teleop=SpaceMouseTeleopConfig(),
        dataset=DatasetRecordConfig(
            repo_id="local/fr3_mujoco_ee2ee_smoke",
            root=dataset_root,
            single_task="FR3 MuJoCo ee2ee smoke",
            fps=args.fps,
            num_episodes=1,
            episode_time_s=args.steps / args.fps,
            reset_time_s=0,
            video=False,
            push_to_hub=False,
        ),
        play_sounds=False,
        display_data=False,
    )
    teleop_action_processor, _, robot_observation_processor = make_fr3_ee2ee_processors(record_cfg)

    teleop_features = {
        "enabled": bool,
        "target_x": float,
        "target_y": float,
        "target_z": float,
        "target_wx": float,
        "target_wy": float,
        "target_wz": float,
        "gripper": float,
    }
    robot_observation_features = build_robot_observation_features(
        include_cameras=True,
        camera_shape=(int(args.camera_height), int(args.camera_width), 3),
    )
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=teleop_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot_observation_features),
            use_videos=False,
        ),
        hw_to_dataset_features(robot_observation_features, prefix=OBS_STR, use_video=False),
    )

    dataset = LeRobotDataset.create(
        repo_id=record_cfg.dataset.repo_id,
        fps=args.fps,
        root=dataset_root,
        robot_type="franka_research3_mujoco",
        features=dataset_features,
        use_videos=False,
        image_writer_threads=0,
    )
    env = FR3MujocoEnv(
        FR3MujocoEnvConfig(
            teleop_control_frequency=float(args.fps),
            max_episode_steps=max(args.steps + 4, 16),
            enable_cameras=True,
            camera_width=int(args.camera_width),
            camera_height=int(args.camera_height),
        )
    )

    try:
        _, info = env.reset()
        current_gripper = 1.0
        for step_idx in range(args.steps):
            observation_raw = env_info_to_robot_observation(info, gripper_pos=current_gripper)
            observation_processed = complete_robot_observation(
                {
                    **observation_raw,
                    **robot_observation_processor(observation_raw),
                }
            )

            teleop_action = scripted_teleop_action(step_idx)
            action_processed = teleop_action_processor((teleop_action.copy(), observation_raw))

            observation_frame = build_dataset_frame(dataset.features, observation_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_processed, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": record_cfg.dataset.single_task})

            _, _, terminated, truncated, info = env.step_teleop_action(teleop_action, control_period_s=1.0 / args.fps)
            current_gripper = float(teleop_action["gripper"])
            if terminated or truncated:
                break

        dataset.save_episode()
        dataset.finalize()

        recorded = LeRobotDataset(record_cfg.dataset.repo_id, root=dataset_root, episodes=[0])
        frame0 = recorded[0]

        required_keys = {"observation.state", "action"}
        required_keys.update({f"observation.images.{camera_name}" for camera_name in _SIM_CAMERA_NAMES})
        missing = required_keys - set(recorded.features)
        if missing:
            raise RuntimeError(f"Missing expected ee2ee features: {sorted(missing)}")

        obs_names = recorded.features[OBS_STR + ".state"]["names"]
        act_names = recorded.features[ACTION]["names"]
        expected_observation_names = [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.qx",
            "ee.qy",
            "ee.qz",
            "ee.qw",
            "prev_cmd.ee.qx",
            "prev_cmd.ee.qy",
            "prev_cmd.ee.qz",
            "prev_cmd.ee.qw",
            "gripper.pos",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "joint_1.pos",
            "joint_2.pos",
            "joint_3.pos",
            "joint_4.pos",
            "joint_5.pos",
            "joint_6.pos",
            "joint_7.pos",
        ]
        expected_action_names = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
        if obs_names != expected_observation_names:
            raise RuntimeError(f"Unexpected observation.state names: {obs_names}")
        if act_names != expected_action_names:
            raise RuntimeError(f"Unexpected action names: {act_names}")

        print("fr3_mujoco_ee2ee_record_smoke=READY")
        print(
            pformat(
                {
                    "dataset_root": str(dataset_root),
                    "num_frames": recorded.num_frames,
                    "features": sorted(recorded.features),
                    "observation_names": obs_names,
                    "action_names": act_names,
                    "image_keys": sorted(key for key in recorded.features if key.startswith("observation.images.")),
                    "frame0_observation_state": np.asarray(frame0["observation.state"]).round(6).tolist(),
                    "frame0_action": np.asarray(frame0["action"]).round(6).tolist(),
                }
            )
        )
    finally:
        env.close()
        if not args.keep_dataset and dataset_root.exists():
            shutil.rmtree(dataset_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
