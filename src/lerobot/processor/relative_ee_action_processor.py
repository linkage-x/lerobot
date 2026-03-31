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

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.rotation import Rotation

from .core import PolicyAction, RobotObservation, TransitionKey
from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry

_ABSOLUTE_ACTION_DIM = 8
_RELATIVE_ACTION_DIM = 8

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
EE_ROTVEC_KEYS = ("ee.wx", "ee.wy", "ee.wz")


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    pose_inv = np.eye(4, dtype=np.float64)
    pose_inv[:3, :3] = rotation.T
    pose_inv[:3, 3] = -rotation.T @ translation
    return pose_inv


def _pose_from_position_and_quaternion(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix()
    return pose


def _pose_from_position_and_rotvec(position_xyz: np.ndarray, rotvec_xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec_xyz, dtype=np.float64)).as_matrix()
    return pose


def _continuous_quaternion(quaternion_xyzw: np.ndarray, previous_quaternion_xyzw: np.ndarray | None) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    if previous_quaternion_xyzw is None:
        return -quaternion_xyzw if float(quaternion_xyzw[3]) < 0.0 else quaternion_xyzw
    if float(np.dot(quaternion_xyzw, previous_quaternion_xyzw)) < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def _anchor_pose_from_observation(observation: RobotObservation) -> np.ndarray:
    if observation is None:
        raise ValueError("Relative EE action processors require an observation in the transition.")

    if all(key in observation for key in EE_QUAT_KEYS):
        return _pose_from_position_and_quaternion(
            np.asarray([observation[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            np.asarray([observation[key] for key in EE_QUAT_KEYS], dtype=np.float64),
        )

    if all(key in observation for key in EE_ROTVEC_KEYS):
        return _pose_from_position_and_rotvec(
            np.asarray([observation[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            np.asarray([observation[key] for key in EE_ROTVEC_KEYS], dtype=np.float64),
        )

    raise ValueError(
        "Observation must contain EE pose as either xyz+quaternion or xyz+rotvec. "
        f"Missing quaternion keys: {[key for key in EE_QUAT_KEYS if key not in observation]}; "
        f"missing rotvec keys: {[key for key in EE_ROTVEC_KEYS if key not in observation]}"
    )


def _restore_action_tensor(array: np.ndarray, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array.reshape(*shape)).to(device=device, dtype=dtype)


@dataclass
@ProcessorStepRegistry.register("absolute_to_relative_ee_action_processor")
class AbsoluteToRelativeEEActionProcessorStep(PolicyActionProcessorStep):
    """Convert absolute xyz+quaternion+gripper actions into current-frame-relative xyz+quaternion+gripper actions."""

    observation_state_key: str = "observation.state"
    state_pose_indices: tuple[int, int, int, int, int, int, int] = field(default_factory=lambda: (0, 1, 2, 3, 4, 5, 6))

    def get_config(self) -> dict:
        return asdict(self)

    def __call__(self, transition):
        if transition.get(TransitionKey.ACTION) is None:
            return transition.copy()
        return super().__call__(transition)

    def _infer_observation_sample_count(self, observation: RobotObservation) -> int:
        if observation is None:
            raise ValueError("Relative EE action processors require an observation in the transition.")

        if self.observation_state_key in observation:
            state_value = observation[self.observation_state_key]
            if isinstance(state_value, torch.Tensor):
                state_array = state_value.detach().cpu().numpy()
            else:
                state_array = np.asarray(state_value, dtype=np.float64)
            if state_array.ndim <= 1:
                return 1
            return int(np.prod(state_array.shape[:-1]))

        if all(key in observation for key in EE_POSITION_KEYS) and (
            all(key in observation for key in EE_QUAT_KEYS) or all(key in observation for key in EE_ROTVEC_KEYS)
        ):
            return 1

        raise ValueError(
            f"Observation must provide either '{self.observation_state_key}' or raw EE pose keys for relative conversion."
        )

    def _anchor_poses_from_observation(self, observation: RobotObservation, num_samples: int) -> list[np.ndarray]:
        if observation is None:
            raise ValueError("Relative EE action processors require an observation in the transition.")

        if self.observation_state_key in observation:
            state_value = observation[self.observation_state_key]
            if isinstance(state_value, torch.Tensor):
                state_array = state_value.detach().cpu().numpy()
            else:
                state_array = np.asarray(state_value, dtype=np.float64)
            if state_array.shape[-1] <= max(self.state_pose_indices):
                raise ValueError(
                    f"Observation state shape {tuple(state_array.shape)} does not cover pose indices {self.state_pose_indices}."
                )
            state_array = state_array.reshape(-1, state_array.shape[-1])
            anchor_poses = [
                _pose_from_position_and_quaternion(
                    row[list(self.state_pose_indices[:3])],
                    row[list(self.state_pose_indices[3:7])],
                )
                for row in state_array
            ]
        elif all(key in observation for key in EE_POSITION_KEYS) and (
            all(key in observation for key in EE_QUAT_KEYS) or all(key in observation for key in EE_ROTVEC_KEYS)
        ):
            anchor_poses = [_anchor_pose_from_observation(observation)]
        else:
            raise ValueError(
                f"Observation must provide either '{self.observation_state_key}' or raw EE pose keys for relative conversion."
            )

        if len(anchor_poses) == 1 and num_samples > 1:
            return [anchor_poses[0].copy() for _ in range(num_samples)]
        if len(anchor_poses) != num_samples:
            raise ValueError(
                f"Anchor sample count {len(anchor_poses)} does not match action sample count {num_samples}."
            )
        return anchor_poses

    @staticmethod
    def _reshape_action_samples(action: torch.Tensor, expected_dim: int, num_anchors: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int]]:
        if action.shape[-1] != expected_dim:
            raise ValueError(f"Expected action dim {expected_dim}, got shape {tuple(action.shape)}.")

        original_shape = tuple(action.shape)
        action_array = action.detach().cpu().numpy()

        if action.ndim == 1:
            sample_count, sequence_length = 1, 1
            flat = action_array.reshape(sample_count, sequence_length, expected_dim)
        elif action.ndim == 2 and action.shape[0] == num_anchors:
            sample_count, sequence_length = action.shape[0], 1
            flat = action_array.reshape(sample_count, sequence_length, expected_dim)
        elif action.ndim == 2:
            sample_count, sequence_length = 1, action.shape[0]
            flat = action_array.reshape(sample_count, sequence_length, expected_dim)
        else:
            sample_count = int(np.prod(action.shape[:-2]))
            sequence_length = action.shape[-2]
            flat = action_array.reshape(sample_count, sequence_length, expected_dim)

        return np.asarray(flat, dtype=np.float64), original_shape, (sample_count, sequence_length)

    def action(self, action: PolicyAction) -> PolicyAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        observation_sample_count = self._infer_observation_sample_count(observation)
        flat_actions, original_shape, (sample_count, sequence_length) = self._reshape_action_samples(
            action, _ABSOLUTE_ACTION_DIM, num_anchors=observation_sample_count
        )
        anchor_poses = self._anchor_poses_from_observation(observation, num_samples=sample_count)

        relative_actions = np.zeros((sample_count, sequence_length, _RELATIVE_ACTION_DIM), dtype=np.float64)
        for sample_idx in range(sample_count):
            anchor_inv = _invert_pose(anchor_poses[sample_idx])
            previous_relative_quaternion = None
            for step_idx in range(sequence_length):
                absolute_action = flat_actions[sample_idx, step_idx]
                absolute_pose = _pose_from_position_and_quaternion(absolute_action[:3], absolute_action[3:7])
                relative_pose = anchor_inv @ absolute_pose
                relative_quaternion = Rotation.from_matrix(relative_pose[:3, :3]).as_quat()
                relative_quaternion = _continuous_quaternion(relative_quaternion, previous_relative_quaternion)
                previous_relative_quaternion = relative_quaternion.copy()
                relative_actions[sample_idx, step_idx, :3] = relative_pose[:3, 3]
                relative_actions[sample_idx, step_idx, 3:7] = relative_quaternion
                relative_actions[sample_idx, step_idx, 7] = absolute_action[7]

        return _restore_action_tensor(relative_actions, original_shape, device=action.device, dtype=action.dtype)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("relative_to_absolute_ee_action_processor")
class RelativeToAbsoluteEEActionProcessorStep(PolicyActionProcessorStep):
    """Convert current-frame-relative xyz+quaternion+gripper actions back into absolute xyz+quaternion+gripper actions."""

    observation_state_key: str = "observation.state"
    state_pose_indices: tuple[int, int, int, int, int, int, int] = field(default_factory=lambda: (0, 1, 2, 3, 4, 5, 6))

    def get_config(self) -> dict:
        return asdict(self)

    def __call__(self, transition):
        if transition.get(TransitionKey.ACTION) is None:
            return transition.copy()
        return super().__call__(transition)

    def action(self, action: PolicyAction) -> PolicyAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        anchor_helper = AbsoluteToRelativeEEActionProcessorStep(
            observation_state_key=self.observation_state_key,
            state_pose_indices=self.state_pose_indices,
        )
        observation_sample_count = anchor_helper._infer_observation_sample_count(observation)
        flat_actions, original_shape, (sample_count, sequence_length) = anchor_helper._reshape_action_samples(
            action, _RELATIVE_ACTION_DIM, num_anchors=observation_sample_count
        )
        anchor_poses = anchor_helper._anchor_poses_from_observation(observation, num_samples=sample_count)

        absolute_actions = np.zeros((sample_count, sequence_length, _ABSOLUTE_ACTION_DIM), dtype=np.float64)
        for sample_idx in range(sample_count):
            previous_absolute_quaternion = None
            for step_idx in range(sequence_length):
                relative_action = flat_actions[sample_idx, step_idx]
                relative_pose = _pose_from_position_and_quaternion(relative_action[:3], relative_action[3:7])
                absolute_pose = anchor_poses[sample_idx] @ relative_pose
                absolute_quaternion = Rotation.from_matrix(absolute_pose[:3, :3]).as_quat()
                absolute_quaternion = _continuous_quaternion(absolute_quaternion, previous_absolute_quaternion)
                previous_absolute_quaternion = absolute_quaternion.copy()
                absolute_actions[sample_idx, step_idx, :3] = absolute_pose[:3, 3]
                absolute_actions[sample_idx, step_idx, 3:7] = absolute_quaternion
                absolute_actions[sample_idx, step_idx, 7] = relative_action[7]

        return _restore_action_tensor(absolute_actions, original_shape, device=action.device, dtype=action.dtype)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
