#!/usr/bin/env python

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

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ObservationProcessorStep, RobotAction, RobotActionProcessorStep, RobotObservation
from lerobot.processor.core import TransitionKey
from lerobot.utils.rotation import Rotation

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_ROTVEC_KEYS = ("ee.wx", "ee.wy", "ee.wz")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
PREV_CMD_POSITION_KEYS = ("prev_cmd.ee.x", "prev_cmd.ee.y", "prev_cmd.ee.z")
PREV_CMD_ROTVEC_KEYS = ("prev_cmd.ee.wx", "prev_cmd.ee.wy", "prev_cmd.ee.wz")
PREV_CMD_QUAT_KEYS = ("prev_cmd.ee.qx", "prev_cmd.ee.qy", "prev_cmd.ee.qz", "prev_cmd.ee.qw")
PREV_CMD_GRIPPER_KEY = "prev_cmd.gripper.pos"


def _canonicalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    dominant_component_index = int(np.argmax(np.abs(quaternion_xyzw)))
    if float(quaternion_xyzw[dominant_component_index]) < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def _continuous_quaternion(quaternion_xyzw: np.ndarray, previous_quaternion_xyzw: np.ndarray | None) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    if previous_quaternion_xyzw is None:
        return _canonicalize_quaternion(quaternion_xyzw)
    if float(np.dot(quaternion_xyzw, previous_quaternion_xyzw)) < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def _pose_from_position_and_rotvec(position_xyz: np.ndarray, rotvec_xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(rotvec_xyz).as_matrix()
    return pose


def _pose_from_position_and_quaternion(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return pose


@dataclass
class KeepAbsoluteEEObservation(ObservationProcessorStep):
    """Keep only absolute EE pose, gripper position, and camera observations using quaternions."""

    _prev_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_prev_cmd_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)

    def _extract_quaternion(
        self,
        observation: RobotObservation,
        *,
        rotvec_keys: tuple[str, str, str],
        quat_keys: tuple[str, str, str, str],
    ) -> np.ndarray | None:
        if all(key in observation for key in rotvec_keys):
            return Rotation.from_rotvec([observation[key] for key in rotvec_keys]).as_quat()
        if all(key in observation for key in quat_keys):
            return np.array([observation[key] for key in quat_keys], dtype=np.float64)
        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        quaternion_xyzw = self._extract_quaternion(
            observation,
            rotvec_keys=EE_ROTVEC_KEYS,
            quat_keys=EE_QUAT_KEYS,
        )
        prev_cmd_quaternion_xyzw = self._extract_quaternion(
            observation,
            rotvec_keys=PREV_CMD_ROTVEC_KEYS,
            quat_keys=PREV_CMD_QUAT_KEYS,
        )

        ignored_keys = set(EE_POSITION_KEYS + EE_ROTVEC_KEYS + EE_QUAT_KEYS + ("gripper.pos",))
        ignored_keys.update(PREV_CMD_POSITION_KEYS + PREV_CMD_ROTVEC_KEYS + PREV_CMD_QUAT_KEYS + (PREV_CMD_GRIPPER_KEY,))

        passthrough_observation = {
            key: value
            for key, value in observation.items()
            if key not in ignored_keys and not key.endswith(".pos")
        }

        processed_observation: RobotObservation = {
            **{key: observation[key] for key in EE_POSITION_KEYS if key in observation},
            "gripper.pos": observation["gripper.pos"],
            **passthrough_observation,
        }

        if quaternion_xyzw is not None:
            quaternion_xyzw = _continuous_quaternion(quaternion_xyzw, self._prev_quaternion_xyzw)
            self._prev_quaternion_xyzw = quaternion_xyzw.copy()
            processed_observation.update(
                {key: float(value) for key, value in zip(EE_QUAT_KEYS, quaternion_xyzw, strict=True)}
            )

        if all(key in observation for key in PREV_CMD_POSITION_KEYS):
            processed_observation.update({key: observation[key] for key in PREV_CMD_POSITION_KEYS})
        if PREV_CMD_GRIPPER_KEY in observation:
            processed_observation[PREV_CMD_GRIPPER_KEY] = observation[PREV_CMD_GRIPPER_KEY]
        if prev_cmd_quaternion_xyzw is not None:
            prev_cmd_quaternion_xyzw = _continuous_quaternion(
                prev_cmd_quaternion_xyzw,
                self._prev_prev_cmd_quaternion_xyzw,
            )
            self._prev_prev_cmd_quaternion_xyzw = prev_cmd_quaternion_xyzw.copy()
            processed_observation.update(
                {key: float(value) for key, value in zip(PREV_CMD_QUAT_KEYS, prev_cmd_quaternion_xyzw, strict=True)}
            )

        return processed_observation

    def reset(self) -> None:
        self._prev_quaternion_xyzw = None
        self._prev_prev_cmd_quaternion_xyzw = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for key in list(observation_features):
            if (
                key.endswith(".pos")
                and key not in {"gripper.pos", PREV_CMD_GRIPPER_KEY}
                and key not in EE_POSITION_KEYS
                and key not in PREV_CMD_POSITION_KEYS
            ):
                observation_features.pop(key, None)
        for key in EE_ROTVEC_KEYS + PREV_CMD_ROTVEC_KEYS:
            observation_features.pop(key, None)
        gripper_feature = observation_features.pop("gripper.pos", None)
        prev_cmd_gripper_feature = observation_features.pop(PREV_CMD_GRIPPER_KEY, None)
        for key in EE_QUAT_KEYS:
            observation_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        for key in PREV_CMD_QUAT_KEYS:
            observation_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        if gripper_feature is not None:
            observation_features["gripper.pos"] = gripper_feature
        if prev_cmd_gripper_feature is not None:
            observation_features[PREV_CMD_GRIPPER_KEY] = prev_cmd_gripper_feature
        return features


@dataclass
class DeltaActionToAbsoluteEEAction(RobotActionProcessorStep):
    """Convert FR3 delta teleop actions into absolute EE action targets using quaternions."""

    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None

    _reference_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_command_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _prev_output_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)

    def _pose_from_observation(self, observation: RobotObservation) -> np.ndarray:
        return _pose_from_position_and_rotvec(
            position_xyz=np.array([observation[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            rotvec_xyz=np.array([observation[key] for key in EE_ROTVEC_KEYS], dtype=np.float64),
        )

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to map FR3 delta teleop commands to absolute EE actions.")

        current_pose = self._pose_from_observation(observation)
        enabled = bool(action.pop("enabled"))
        gripper = float(np.clip(action.pop("gripper"), 0.0, 1.0))
        raw_target_x = float(action.pop("target_x"))
        raw_target_y = float(action.pop("target_y"))
        raw_target_z = float(action.pop("target_z"))
        raw_target_wx = float(action.pop("target_wx"))
        raw_target_wy = float(action.pop("target_wy"))
        raw_target_wz = float(action.pop("target_wz"))

        if enabled:
            if not self._prev_enabled or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            delta_pos = np.array(
                [raw_target_x, raw_target_y, raw_target_z],
                dtype=np.float64,
            )
            if self.max_target_delta_pos is not None:
                delta_pos = np.clip(
                    delta_pos,
                    -np.asarray(self.max_target_delta_pos, dtype=np.float64),
                    np.asarray(self.max_target_delta_pos, dtype=np.float64),
                )

            delta_rotvec = np.array(
                [raw_target_wx, raw_target_wy, raw_target_wz],
                dtype=np.float64,
            )
            if self.max_target_delta_rot is not None:
                delta_rotvec = np.clip(
                    delta_rotvec,
                    -np.asarray(self.max_target_delta_rot, dtype=np.float64),
                    np.asarray(self.max_target_delta_rot, dtype=np.float64),
                )
            delta_rot = Rotation.from_rotvec(delta_rotvec)

            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = self._reference_pose[:3, :3] @ delta_rot.as_matrix()
            desired_pose[:3, 3] = self._reference_pose[:3, 3] + delta_pos
            desired_pose[:3, 3] = np.clip(
                desired_pose[:3, 3],
                np.asarray(self.workspace_min, dtype=np.float64),
                np.asarray(self.workspace_max, dtype=np.float64),
            )
            self._last_command_pose = desired_pose.copy()
            self._reference_pose = desired_pose.copy()
        else:
            desired_pose = self._last_command_pose.copy() if self._last_command_pose is not None else current_pose.copy()
            self._reference_pose = None

        desired_quaternion_xyzw = Rotation.from_matrix(desired_pose[:3, :3]).as_quat()
        desired_quaternion_xyzw = _continuous_quaternion(desired_quaternion_xyzw, self._prev_output_quaternion_xyzw)
        self._prev_output_quaternion_xyzw = desired_quaternion_xyzw.copy()
        self._prev_enabled = enabled
        return {
            "enabled": enabled,
            "target_x": raw_target_x,
            "target_y": raw_target_y,
            "target_z": raw_target_z,
            "target_wx": raw_target_wx,
            "target_wy": raw_target_wy,
            "target_wz": raw_target_wz,
            "gripper": gripper,
            "ee.x": float(desired_pose[0, 3]),
            "ee.y": float(desired_pose[1, 3]),
            "ee.z": float(desired_pose[2, 3]),
            "ee.qx": float(desired_quaternion_xyzw[0]),
            "ee.qy": float(desired_quaternion_xyzw[1]),
            "ee.qz": float(desired_quaternion_xyzw[2]),
            "ee.qw": float(desired_quaternion_xyzw[3]),
            "gripper.pos": gripper,
        }

    def reset(self) -> None:
        self._reference_pose = None
        self._last_command_pose = None
        self._prev_enabled = False
        self._prev_output_quaternion_xyzw = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in ("enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper"):
            action_features.pop(key, None)

        for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"):
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@dataclass
class AbsoluteEEActionToRobotAction(RobotActionProcessorStep):
    """Adapt ee2ee record actions so idle frames still use the robot's hold-current-joints path."""

    def action(self, action: RobotAction) -> RobotAction:
        if not all(key in action for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw")):
            return action

        gripper = float(np.clip(action.get("gripper.pos", action.get("gripper", 0.0)), 0.0, 1.0))
        enabled = bool(action.get("enabled", True))
        if not enabled:
            return {
                "enabled": False,
                "target_x": float(action.get("target_x", 0.0)),
                "target_y": float(action.get("target_y", 0.0)),
                "target_z": float(action.get("target_z", 0.0)),
                "target_wx": float(action.get("target_wx", 0.0)),
                "target_wy": float(action.get("target_wy", 0.0)),
                "target_wz": float(action.get("target_wz", 0.0)),
                "gripper": gripper,
            }

        desired_pose = _pose_from_position_and_quaternion(
            position_xyz=np.array([action[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            quaternion_xyzw=np.array([action[key] for key in EE_QUAT_KEYS], dtype=np.float64),
        )
        desired_rotvec = Rotation.from_matrix(desired_pose[:3, :3]).as_rotvec()

        return {
            "ee.x": float(action["ee.x"]),
            "ee.y": float(action["ee.y"]),
            "ee.z": float(action["ee.z"]),
            "ee.wx": float(desired_rotvec[0]),
            "ee.wy": float(desired_rotvec[1]),
            "ee.wz": float(desired_rotvec[2]),
            "gripper.pos": gripper,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
