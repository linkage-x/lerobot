#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    ActionChunkQuantileNormalizerProcessorStep,
    ActionChunkQuantileUnnormalizerProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def resolve_action_chunk_stats_path(config: ACTConfig, dataset_root: str | Path | None) -> Path | None:
    if not config.action_chunk_quantile_normalization:
        return None

    if config.action_chunk_stats_path is not None:
        configured_path = Path(config.action_chunk_stats_path)
        if configured_path.is_absolute():
            return configured_path
        if configured_path.exists():
            return configured_path
        if dataset_root is not None:
            return Path(dataset_root) / configured_path
        return configured_path

    if dataset_root is None:
        return None

    return Path(dataset_root) / "meta" / f"policy_action_chunk_stats.chunk{config.chunk_size}.json"


def load_action_chunk_stats(config: ACTConfig, dataset_root: str | Path | None) -> dict[str, Any] | None:
    stats_path = resolve_action_chunk_stats_path(config, dataset_root)
    if stats_path is None:
        return None
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Action chunk stats file not found at {stats_path}. "
            "Disable action_chunk_quantile_normalization or provide a valid action_chunk_stats_path."
        )
    with stats_path.open(encoding="utf-8") as f:
        stats = json.load(f)

    if stats.get("method") != "quantile_per_offset":
        raise ValueError(f"Unsupported action chunk stats method: {stats.get('method')!r}")
    if int(stats.get("chunk_size", -1)) != config.chunk_size:
        raise ValueError(
            f"Chunk stats chunk_size={stats.get('chunk_size')} does not match policy chunk_size={config.chunk_size}."
        )
    return stats


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    action_chunk_stats: dict[str, Any] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    normalize_observation_keys = set(config.input_features) if config.input_features else None
    if (
        normalize_observation_keys is not None
        and config.use_tactile
        and config.tactile_use_valid_mask
        and config.tactile_valid_mask_feature_key
    ):
        normalize_observation_keys.discard(config.tactile_valid_mask_feature_key)

    normalization_map = dict(config.normalization_mapping)
    if config.action_chunk_quantile_normalization:
        normalization_map[FeatureType.ACTION] = NormalizationMode.IDENTITY

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=normalization_map,
            stats=dataset_stats,
            device=config.device,
            normalize_observation_keys=normalize_observation_keys,
        ),
    ]
    output_steps = []

    if config.action_chunk_quantile_normalization:
        if action_chunk_stats is None:
            raise ValueError("action_chunk_quantile_normalization is enabled but action_chunk_stats is missing.")
        action_feature = config.output_features[ACTION]
        shared_kwargs = {
            "feature_key": ACTION,
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
            "action_dim": action_feature.shape[-1],
            "lower_quantile": float(action_chunk_stats["lower_quantile"]),
            "upper_quantile": float(action_chunk_stats["upper_quantile"]),
            "offset_stats": action_chunk_stats["offset_stats"],
            "action_feature_names": action_chunk_stats.get("action_names"),
            "clip": bool(config.action_chunk_quantile_clip),
        }
        input_steps.append(ActionChunkQuantileNormalizerProcessorStep(**shared_kwargs))
        output_steps.append(ActionChunkQuantileUnnormalizerProcessorStep(**shared_kwargs))

    output_steps.extend(
        [
            UnnormalizerProcessorStep(
                features=config.output_features, norm_map=normalization_map, stats=dataset_stats
            ),
            DeviceProcessorStep(device="cpu"),
        ]
    )

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
