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
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import ACTION
from lerobot.utils.state_feature_names import get_ee_pose_state_indices

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


def _normalize_quantile_name(value: float) -> str:
    quantile_int = int(round(float(value) * 100))
    return f"q{quantile_int:02d}"


@dataclass
class _ActionChunkQuantileMixin:
    feature_key: str = ACTION
    chunk_size: int = 1
    n_action_steps: int = 1
    action_dim: int = 0
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    offset_stats: list[dict[str, Any]] = field(default_factory=list)
    action_feature_names: list[str] | dict[str, Any] | None = None
    quaternion_indices: list[int] | None = None
    clip: bool = False
    eps: float = 1e-8

    _lower_stats: Tensor = field(init=False, repr=False)
    _upper_stats: Tensor = field(init=False, repr=False)
    _next_inference_offset: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        lower_key = _normalize_quantile_name(self.lower_quantile)
        upper_key = _normalize_quantile_name(self.upper_quantile)
        if not self.offset_stats:
            raise ValueError("offset_stats must be provided for action chunk quantile normalization.")

        sorted_stats = sorted(self.offset_stats, key=lambda item: int(item["offset"]))
        if len(sorted_stats) != self.chunk_size:
            raise ValueError(
                f"Expected {self.chunk_size} offset stats entries, got {len(sorted_stats)} for {self.feature_key!r}."
            )

        lower_stats = []
        upper_stats = []
        for expected_offset, stats in enumerate(sorted_stats):
            actual_offset = int(stats["offset"])
            if actual_offset != expected_offset:
                raise ValueError(
                    f"Offset stats must be contiguous and start at 0. Expected offset {expected_offset}, got {actual_offset}."
                )
            if lower_key not in stats or upper_key not in stats:
                raise ValueError(f"Offset {actual_offset} is missing {lower_key}/{upper_key} statistics.")
            lower_stats.append(stats[lower_key])
            upper_stats.append(stats[upper_key])

        self.offset_stats = sorted_stats
        self._lower_stats = torch.as_tensor(lower_stats, dtype=torch.float32)
        self._upper_stats = torch.as_tensor(upper_stats, dtype=torch.float32)

        if self.action_dim <= 0:
            self.action_dim = int(self._lower_stats.shape[-1])
        if int(self._lower_stats.shape[-1]) != self.action_dim:
            raise ValueError(
                f"Action dim mismatch. Expected {self.action_dim}, got stats with dim {self._lower_stats.shape[-1]}."
            )

        if self.quaternion_indices is None and self.action_feature_names is not None:
            ee_pose_indices = get_ee_pose_state_indices(self.action_feature_names, strict=False)
            if ee_pose_indices is not None:
                self.quaternion_indices = [
                    ee_pose_indices["ee.qx"],
                    ee_pose_indices["ee.qy"],
                    ee_pose_indices["ee.qz"],
                    ee_pose_indices["ee.qw"],
                ]

    def get_config(self) -> dict[str, Any]:
        return {
            "feature_key": self.feature_key,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.action_dim,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "offset_stats": self.offset_stats,
            "action_feature_names": self.action_feature_names,
            "quaternion_indices": self.quaternion_indices,
            "clip": self.clip,
            "eps": self.eps,
        }

    def reset(self) -> None:
        self._next_inference_offset = 0

    def _quantile_transform(self, tensor: Tensor, lower: Tensor, upper: Tensor, *, inverse: bool) -> Tensor:
        lower = lower.to(device=tensor.device, dtype=tensor.dtype)
        upper = upper.to(device=tensor.device, dtype=tensor.dtype)
        denom = upper - lower
        denom = torch.where(denom == 0, torch.full_like(denom, self.eps), denom)

        if inverse:
            if self.clip:
                tensor = tensor.clamp(-1.0, 1.0)
            output = (tensor + 1.0) * denom / 2.0 + lower
            return self._renormalize_quaternion(output)

        output = 2.0 * (tensor - lower) / denom - 1.0
        if self.clip:
            output = output.clamp(-1.0, 1.0)
        return output

    def _renormalize_quaternion(self, tensor: Tensor) -> Tensor:
        if not self.quaternion_indices:
            return tensor

        renormalized = tensor.clone()
        quat = renormalized[..., list(self.quaternion_indices)]
        quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        quat_norm = torch.where(quat_norm == 0, torch.ones_like(quat_norm), quat_norm)
        renormalized[..., list(self.quaternion_indices)] = quat / quat_norm
        return renormalized

    def _transform_chunk(self, action: Tensor, *, inverse: bool) -> Tensor:
        view_shape = [1] * (action.dim() - 2) + [self.chunk_size, self.action_dim]
        lower = self._lower_stats.view(*view_shape)
        upper = self._upper_stats.view(*view_shape)
        return self._quantile_transform(action, lower, upper, inverse=inverse)

    def _transform_single_action(self, action: Tensor, *, inverse: bool, advance_offset: bool) -> Tensor:
        offset = self._next_inference_offset
        transformed = self._quantile_transform(
            action,
            self._lower_stats[offset],
            self._upper_stats[offset],
            inverse=inverse,
        )
        if advance_offset:
            self._next_inference_offset = (self._next_inference_offset + 1) % max(1, self.n_action_steps)
        return transformed

    def _transform_action(self, action: Tensor, *, inverse: bool, advance_offset: bool) -> Tensor:
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim} for {self.feature_key!r}, got shape {tuple(action.shape)}."
            )

        if action.dim() >= 2 and action.shape[-2] == self.chunk_size:
            return self._transform_chunk(action, inverse=inverse)

        return self._transform_single_action(action, inverse=inverse, advance_offset=advance_offset)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="action_chunk_quantile_normalizer_processor")
class ActionChunkQuantileNormalizerProcessorStep(_ActionChunkQuantileMixin, PolicyActionProcessorStep):
    def action(self, action: PolicyAction) -> PolicyAction:
        return self._transform_action(action, inverse=False, advance_offset=False)


@dataclass
@ProcessorStepRegistry.register(name="action_chunk_quantile_unnormalizer_processor")
class ActionChunkQuantileUnnormalizerProcessorStep(_ActionChunkQuantileMixin, PolicyActionProcessorStep):
    def action(self, action: PolicyAction) -> PolicyAction:
        return self._transform_action(action, inverse=True, advance_offset=True)
