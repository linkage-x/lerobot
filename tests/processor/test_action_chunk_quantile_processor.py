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

from lerobot.processor import (
    ActionChunkQuantileNormalizerProcessorStep,
    ActionChunkQuantileUnnormalizerProcessorStep,
    DataProcessorPipeline,
)
from lerobot.processor.converters import create_transition, policy_action_to_transition, transition_to_policy_action


def _make_offset_stats():
    return [
        {
            "offset": 0,
            "q02": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
            "q98": [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.5, 0.9],
        },
        {
            "offset": 1,
            "q02": [10.0, 10.0, 10.0, 0.2, 0.2, 0.2, 0.8, 0.2],
            "q98": [14.0, 14.0, 14.0, 0.6, 0.6, 0.6, 1.2, 0.6],
        },
    ]


def _make_step_kwargs():
    return {
        "chunk_size": 2,
        "n_action_steps": 2,
        "action_dim": 8,
        "lower_quantile": 0.02,
        "upper_quantile": 0.98,
        "offset_stats": _make_offset_stats(),
        "action_feature_names": {"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
    }


def test_action_chunk_quantile_round_trip_for_chunk_tensor():
    normalizer = ActionChunkQuantileNormalizerProcessorStep(**_make_step_kwargs())
    unnormalizer = ActionChunkQuantileUnnormalizerProcessorStep(**_make_step_kwargs())

    action_chunk = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 1.25, 0.5],
                [12.0, 12.0, 12.0, 0.4, 0.4, 0.4, 1.0, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    transition = create_transition(action=action_chunk)

    normalized = normalizer(transition)["action"]
    restored = unnormalizer(create_transition(action=normalized))["action"]

    assert torch.allclose(normalized[0, 0], torch.zeros(8), atol=1e-6)
    assert torch.allclose(normalized[0, 1], torch.zeros(8), atol=1e-6)
    assert torch.allclose(restored[..., :3], action_chunk[..., :3], atol=1e-6)
    assert torch.allclose(restored[..., 7:], action_chunk[..., 7:], atol=1e-6)
    assert torch.allclose(torch.linalg.norm(restored[..., 3:7], dim=-1), torch.ones_like(restored[..., 0]), atol=1e-6)


def test_action_chunk_quantile_unnormalizer_tracks_single_action_offsets_and_reset():
    unnormalizer = ActionChunkQuantileUnnormalizerProcessorStep(**_make_step_kwargs())

    normalized_action = torch.zeros(1, 8, dtype=torch.float32)

    first = unnormalizer(create_transition(action=normalized_action))["action"]
    second = unnormalizer(create_transition(action=normalized_action))["action"]
    third = unnormalizer(create_transition(action=normalized_action))["action"]

    assert torch.allclose(first[0, :3], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(second[0, :3], torch.tensor([12.0, 12.0, 12.0]))
    assert torch.allclose(third[0, :3], torch.tensor([1.0, 1.0, 1.0]))

    unnormalizer.reset()
    again = unnormalizer(create_transition(action=normalized_action))["action"]
    assert torch.allclose(again[0, :3], torch.tensor([1.0, 1.0, 1.0]))


def test_action_chunk_quantile_processor_pipeline_save_and_load(tmp_path):
    step = ActionChunkQuantileUnnormalizerProcessorStep(**_make_step_kwargs())
    pipeline = DataProcessorPipeline(
        steps=[step],
        name="quantile_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    pipeline.save_pretrained(tmp_path, config_filename="postprocessor.json")
    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    output = loaded(torch.zeros(1, 8, dtype=torch.float32))
    assert torch.allclose(output[0, :3], torch.tensor([1.0, 1.0, 1.0]))
