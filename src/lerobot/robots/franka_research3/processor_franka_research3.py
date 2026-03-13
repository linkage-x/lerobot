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


@dataclass
class KeepAbsoluteEEObservation(ObservationProcessorStep):
    """Keep only absolute EE pose, gripper position, and camera observations."""

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return {
            key: value
            for key, value in observation.items()
            if key.startswith("ee.") or key == "gripper.pos" or not key.endswith(".pos")
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for key in list(observation_features):
            if key.endswith(".pos") and key != "gripper.pos" and not key.startswith("ee."):
                observation_features.pop(key, None)
        return features


@dataclass
class DeltaActionToAbsoluteEEAction(RobotActionProcessorStep):
    """Convert FR3 delta teleop actions into absolute EE action targets."""

    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None

    _reference_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_command_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)

    def _pose_from_observation(self, observation: RobotObservation) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array(
            [observation["ee.x"], observation["ee.y"], observation["ee.z"]],
            dtype=np.float64,
        )
        pose[:3, :3] = Rotation.from_rotvec(
            [observation["ee.wx"], observation["ee.wy"], observation["ee.wz"]]
        ).as_matrix()
        return pose

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to map FR3 delta teleop commands to absolute EE actions.")

        current_pose = self._pose_from_observation(observation)
        enabled = bool(action.pop("enabled"))
        gripper = float(np.clip(action.pop("gripper"), 0.0, 1.0))

        if enabled:
            if not self._prev_enabled or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            delta_pos = np.array(
                [float(action.pop("target_x")), float(action.pop("target_y")), float(action.pop("target_z"))],
                dtype=np.float64,
            )
            if self.max_target_delta_pos is not None:
                delta_pos = np.clip(
                    delta_pos,
                    -np.asarray(self.max_target_delta_pos, dtype=np.float64),
                    np.asarray(self.max_target_delta_pos, dtype=np.float64),
                )

            delta_rotvec = np.array(
                [float(action.pop("target_wx")), float(action.pop("target_wy")), float(action.pop("target_wz"))],
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

            # Consume the disabled delta payload to keep the action contract explicit.
            for key in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"):
                action.pop(key, None)

        desired_rotvec = Rotation.from_matrix(desired_pose[:3, :3]).as_rotvec()
        self._prev_enabled = enabled
        return {
            "ee.x": float(desired_pose[0, 3]),
            "ee.y": float(desired_pose[1, 3]),
            "ee.z": float(desired_pose[2, 3]),
            "ee.wx": float(desired_rotvec[0]),
            "ee.wy": float(desired_rotvec[1]),
            "ee.wz": float(desired_rotvec[2]),
            "gripper.pos": gripper,
        }

    def reset(self) -> None:
        self._reference_pose = None
        self._last_command_pose = None
        self._prev_enabled = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in ("enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper"):
            action_features.pop(key, None)

        for key in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"):
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
