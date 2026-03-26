#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

EE_POSE_STATE_NAME_ALIASES = {
    "ee.x": ("ee.x", "x"),
    "ee.y": ("ee.y", "y"),
    "ee.z": ("ee.z", "z"),
    "ee.qx": ("ee.qx", "qx"),
    "ee.qy": ("ee.qy", "qy"),
    "ee.qz": ("ee.qz", "qz"),
    "ee.qw": ("ee.qw", "qw"),
}


def flatten_feature_name_paths(
    feature_names: list[str] | tuple[str, ...] | dict[str, Any] | None,
    prefix: str = "",
) -> list[str] | None:
    if feature_names is None:
        return None
    if isinstance(feature_names, (list, tuple)):
        return [f"{prefix}/{name}" if prefix else str(name) for name in feature_names]
    if isinstance(feature_names, dict):
        flattened = []
        for key, value in feature_names.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            child_paths = flatten_feature_name_paths(value, child_prefix)
            if child_paths is not None:
                flattened.extend(child_paths)
        return flattened
    raise TypeError(f"Unsupported feature names structure: {type(feature_names)!r}")


def resolve_feature_name_indices(
    feature_names: list[str] | tuple[str, ...] | dict[str, Any] | None,
    required_aliases: dict[str, tuple[str, ...]],
    *,
    strict: bool = True,
) -> dict[str, int] | None:
    flattened_names = flatten_feature_name_paths(feature_names)
    if flattened_names is None:
        if strict:
            raise ValueError("Feature names metadata is required to resolve named state indices.")
        return None

    aliases = {}
    for idx, name in enumerate(flattened_names):
        aliases.setdefault(name, idx)
        aliases.setdefault(name.split("/")[-1], idx)

    indices = {}
    missing = []
    for canonical_name, candidate_aliases in required_aliases.items():
        for alias in candidate_aliases:
            if alias in aliases:
                indices[canonical_name] = aliases[alias]
                break
        else:
            missing.append(canonical_name)

    if missing:
        if strict:
            raise ValueError(
                f"Could not resolve required feature names {missing}. Available state features: {flattened_names}"
            )
        return None

    return indices


def get_ee_pose_state_indices(
    state_feature_names: list[str] | tuple[str, ...] | dict[str, Any] | None,
    *,
    strict: bool = False,
) -> dict[str, int] | None:
    return resolve_feature_name_indices(
        state_feature_names,
        EE_POSE_STATE_NAME_ALIASES,
        strict=strict,
    )
