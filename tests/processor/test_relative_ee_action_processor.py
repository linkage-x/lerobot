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

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteToRelativeEEActionProcessorStep,
    DataProcessorPipeline,
    RelativeToAbsoluteEEActionProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, policy_action_to_transition, transition_to_policy_action
from lerobot.utils.rotation import Rotation


def _make_observation() -> dict[str, float]:
    quaternion_xyzw = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_quat()
    return {
        "ee.x": 0.5,
        "ee.y": -0.1,
        "ee.z": 0.3,
        "ee.qx": float(quaternion_xyzw[0]),
        "ee.qy": float(quaternion_xyzw[1]),
        "ee.qz": float(quaternion_xyzw[2]),
        "ee.qw": float(quaternion_xyzw[3]),
    }


def _make_absolute_chunk() -> torch.Tensor:
    pose0_quaternion = Rotation.from_rotvec([0.25, -0.1, 0.55]).as_quat()
    pose1_quaternion = Rotation.from_rotvec([0.4, -0.3, 0.6]).as_quat()
    return torch.tensor(
        [
            [
                [0.7, 0.0, 0.35, *pose0_quaternion.tolist(), 0.2],
                [0.8, 0.1, 0.4, *pose1_quaternion.tolist(), 0.6],
            ]
        ],
        dtype=torch.float32,
    )


def _make_observation_state_batch() -> torch.Tensor:
    observation0 = _make_observation()
    quaternion1 = Rotation.from_rotvec([-0.15, 0.05, 0.2]).as_quat()
    observation1 = {
        "ee.x": -0.2,
        "ee.y": 0.3,
        "ee.z": 0.4,
        "ee.qx": float(quaternion1[0]),
        "ee.qy": float(quaternion1[1]),
        "ee.qz": float(quaternion1[2]),
        "ee.qw": float(quaternion1[3]),
    }
    return torch.tensor(
        [
            [
                observation0["ee.x"],
                observation0["ee.y"],
                observation0["ee.z"],
                observation0["ee.qx"],
                observation0["ee.qy"],
                observation0["ee.qz"],
                observation0["ee.qw"],
                0.2,
            ],
            [
                observation1["ee.x"],
                observation1["ee.y"],
                observation1["ee.z"],
                observation1["ee.qx"],
                observation1["ee.qy"],
                observation1["ee.qz"],
                observation1["ee.qw"],
                0.7,
            ],
        ],
        dtype=torch.float32,
    )


def _assert_quaternion_pose_match(expected: torch.Tensor, actual: torch.Tensor, atol: float = 1e-5) -> None:
    assert torch.allclose(actual[..., :3], expected[..., :3], atol=atol)
    assert torch.allclose(actual[..., 7], expected[..., 7], atol=atol)
    quat_dot = torch.sum(actual[..., 3:7] * expected[..., 3:7], dim=-1).abs()
    assert torch.allclose(quat_dot, torch.ones_like(quat_dot), atol=atol)


def test_absolute_to_relative_ee_action_round_trip_for_chunk_tensor():
    absolute_to_relative = AbsoluteToRelativeEEActionProcessorStep()
    relative_to_absolute = RelativeToAbsoluteEEActionProcessorStep()

    absolute_chunk = _make_absolute_chunk()
    transition = create_transition(observation=_make_observation(), action=absolute_chunk)

    relative_chunk = absolute_to_relative(transition)[TransitionKey.ACTION]
    restored_chunk = relative_to_absolute(
        create_transition(observation=_make_observation(), action=relative_chunk)
    )[TransitionKey.ACTION]

    assert relative_chunk.shape == (1, 2, 8)
    _assert_quaternion_pose_match(absolute_chunk, restored_chunk)


def test_relative_to_absolute_ee_action_round_trip_for_single_action_tensor():
    absolute_to_relative = AbsoluteToRelativeEEActionProcessorStep()
    relative_to_absolute = RelativeToAbsoluteEEActionProcessorStep()

    absolute_action = _make_absolute_chunk()[0, 0]
    relative_action = absolute_to_relative(
        create_transition(observation=_make_observation(), action=absolute_action)
    )[TransitionKey.ACTION]
    restored_action = relative_to_absolute(
        create_transition(observation=_make_observation(), action=relative_action)
    )[TransitionKey.ACTION]

    assert relative_action.shape == (8,)
    _assert_quaternion_pose_match(absolute_action, restored_action)


def test_relative_ee_action_round_trip_for_batched_single_step_actions_with_observation_state():
    absolute_to_relative = AbsoluteToRelativeEEActionProcessorStep()
    relative_to_absolute = RelativeToAbsoluteEEActionProcessorStep()

    observation_state = _make_observation_state_batch()
    absolute_actions = torch.tensor(
        [
            [0.7, 0.0, 0.35, *Rotation.from_rotvec([0.25, -0.1, 0.55]).as_quat().tolist(), 0.2],
            [-0.15, 0.35, 0.45, *Rotation.from_rotvec([-0.12, 0.07, 0.31]).as_quat().tolist(), 0.6],
        ],
        dtype=torch.float32,
    )

    transition = create_transition(
        observation={"observation.state": observation_state},
        action=absolute_actions,
    )

    relative_actions = absolute_to_relative(transition)[TransitionKey.ACTION]
    restored_actions = relative_to_absolute(
        create_transition(observation={"observation.state": observation_state}, action=relative_actions)
    )[TransitionKey.ACTION]

    assert relative_actions.shape == (2, 8)
    _assert_quaternion_pose_match(absolute_actions, restored_actions)


def test_absolute_to_relative_ee_action_matches_expected_local_translation_for_identity_anchor():
    processor = AbsoluteToRelativeEEActionProcessorStep()
    transition = create_transition(
        observation={
            "ee.x": 0.0,
            "ee.y": 0.0,
            "ee.z": 0.0,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
        },
        action=torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.4]], dtype=torch.float32),
    )

    relative_action = processor(transition)[TransitionKey.ACTION]

    assert torch.allclose(relative_action[0, :3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
    assert torch.allclose(relative_action[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(relative_action[0, 7], torch.tensor(0.4), atol=1e-6)


def test_absolute_to_relative_ee_action_canonicalizes_quaternion_sign_for_single_action():
    processor = AbsoluteToRelativeEEActionProcessorStep()
    transition = create_transition(
        observation={
            "ee.x": 0.0,
            "ee.y": 0.0,
            "ee.z": 0.0,
            "ee.qx": 0.0,
            "ee.qy": 0.0,
            "ee.qz": 0.0,
            "ee.qw": 1.0,
        },
        action=torch.tensor([[1.0, 2.0, 3.0, -0.0, -0.0, -0.0, -1.0, 0.4]], dtype=torch.float32),
    )

    relative_action = processor(transition)[TransitionKey.ACTION]

    assert torch.allclose(relative_action[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_relative_ee_action_processor_updates_action_feature_shape():
    absolute_features = {
        PipelineFeatureType.ACTION: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))
        },
        PipelineFeatureType.OBSERVATION: {},
    }

    absolute_to_relative = AbsoluteToRelativeEEActionProcessorStep()
    relative_to_absolute = RelativeToAbsoluteEEActionProcessorStep()

    transformed_relative = absolute_to_relative.transform_features(absolute_features)
    transformed_absolute = relative_to_absolute.transform_features(absolute_features)

    assert transformed_relative[PipelineFeatureType.ACTION]["action"].shape == (8,)
    assert transformed_absolute[PipelineFeatureType.ACTION]["action"].shape == (8,)


def test_relative_ee_action_processor_pipeline_save_and_load(tmp_path):
    pipeline = DataProcessorPipeline(
        steps=[AbsoluteToRelativeEEActionProcessorStep()],
        name="relative_ee_preprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    pipeline.save_pretrained(tmp_path, config_filename="preprocessor.json")
    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="preprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    loaded_step = loaded.steps[0]
    assert isinstance(loaded_step, AbsoluteToRelativeEEActionProcessorStep)
