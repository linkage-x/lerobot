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

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import draccus
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.processor_act import load_action_chunk_stats
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Inspect one ACT training batch and print raw absolute action, processed relative action, and masked observation.state."
    )
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=None, help="Single-sample shortcut. If set with --num-samples>1, it becomes the start index.")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated dataset indices to inspect, e.g. '0,100,5000'.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--action-step", type=int, default=0)
    parser.add_argument("--max-action-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jsonl-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_known_args(argv)


def _default_feature_names(feature_key: str, width: int) -> list[str] | None:
    if width == 8:
        return ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    if feature_key == OBS_STATE and width == 7:
        return ["x", "y", "z", "wx", "wy", "wz", "gripper"]
    if feature_key == ACTION and width == 7:
        return ["x", "y", "z", "wx", "wy", "wz", "gripper"]
    return None


def _feature_names(feature_key: str, feature_meta: dict[str, Any] | None, width: int) -> list[str]:
    if feature_meta is None:
        default_names = _default_feature_names(feature_key, width)
        return default_names if default_names is not None else [str(i) for i in range(width)]
    names = feature_meta.get("names")
    if isinstance(names, list) and len(names) == width:
        return [str(name) for name in names]
    default_names = _default_feature_names(feature_key, width)
    return default_names if default_names is not None else [str(i) for i in range(width)]


def _vector_to_named_dict(vector: torch.Tensor, names: list[str]) -> dict[str, float]:
    vector = vector.detach().cpu().to(dtype=torch.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape {tuple(vector.shape)}")
    if vector.shape[0] != len(names):
        raise ValueError(f"Expected {len(names)} names, got vector shape {tuple(vector.shape)}")
    return {name: float(vector[i].item()) for i, name in enumerate(names)}


def _mean_abs_named_dict(named_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not named_dicts:
        return {}
    keys = list(named_dicts[0])
    return {
        key: float(sum(abs(sample[key]) for sample in named_dicts) / len(named_dicts))
        for key in keys
    }


def _max_abs_named_dict(named_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not named_dicts:
        return {}
    keys = list(named_dicts[0])
    return {
        key: float(max(abs(sample[key]) for sample in named_dicts))
        for key in keys
    }


def _select_state_vector(batch: dict[str, Any], *, sample_index: int) -> torch.Tensor:
    state = batch[OBS_STATE]
    if state.ndim < 2:
        raise ValueError(f"Expected batched observation.state, got shape {tuple(state.shape)}")
    return state[sample_index]


def _select_action_vector(batch: dict[str, Any], *, sample_index: int, action_step: int) -> torch.Tensor:
    action = batch[ACTION]
    if action.ndim == 2:
        return action[sample_index]
    if action.ndim >= 3:
        return action[sample_index, action_step]
    raise ValueError(f"Expected batched action tensor, got shape {tuple(action.shape)}")


def _select_action_head(
    batch: dict[str, Any],
    *,
    sample_index: int,
    start_action_step: int,
    max_action_steps: int,
    action_names: list[str],
) -> list[dict[str, Any]]:
    action = batch[ACTION]
    if action.ndim == 2:
        return [{"step": 0, "values": _vector_to_named_dict(action[sample_index], action_names)}]
    if action.ndim < 3:
        raise ValueError(f"Expected batched action tensor, got shape {tuple(action.shape)}")
    total_steps = int(action.shape[1])
    if start_action_step < 0 or start_action_step >= total_steps:
        raise IndexError(f"action_step {start_action_step} out of range for action shape {tuple(action.shape)}")
    steps_to_show = min(max_action_steps, total_steps - start_action_step)
    return [
        {
            "step": start_action_step + offset,
            "values": _vector_to_named_dict(action[sample_index, start_action_step + offset], action_names),
        }
        for offset in range(steps_to_show)
    ]


def _run_preprocessor_stages(
    preprocessor: DataProcessorPipeline[dict[str, Any], dict[str, Any]],
    raw_batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transition = preprocessor.to_transition(deepcopy(raw_batch))
    before_qoff_transition = None

    for processor_step in preprocessor.steps:
        if (
            before_qoff_transition is None
            and type(processor_step).__name__ == "ActionChunkQuantileNormalizerProcessorStep"
        ):
            before_qoff_transition = deepcopy(transition)
        transition = processor_step(transition)

    if before_qoff_transition is None:
        before_qoff_transition = deepcopy(transition)

    return (
        preprocessor.to_output(before_qoff_transition),
        preprocessor.to_output(transition),
    )


def build_batch_inspection_report(
    raw_batch: dict[str, Any],
    relative_batch_before_qoff: dict[str, Any],
    processed_batch_after_qoff: dict[str, Any],
    masked_batch: dict[str, Any],
    *,
    state_names: list[str],
    action_names: list[str],
    sample_index: int,
    dataset_index: int | None,
    action_step: int,
    max_action_steps: int,
    policy_cfg: Any,
    preprocessor_steps: list[str],
) -> dict[str, Any]:
    raw_action_head = _select_action_head(
        raw_batch,
        sample_index=sample_index,
        start_action_step=action_step,
        max_action_steps=max_action_steps,
        action_names=action_names,
    )
    relative_action_head = _select_action_head(
        relative_batch_before_qoff,
        sample_index=sample_index,
        start_action_step=action_step,
        max_action_steps=max_action_steps,
        action_names=action_names,
    )
    processed_action_head = _select_action_head(
        processed_batch_after_qoff,
        sample_index=sample_index,
        start_action_step=action_step,
        max_action_steps=max_action_steps,
        action_names=action_names,
    )

    return {
        "contract": {
            "policy_type": getattr(policy_cfg, "type", None),
            "relative_ee_action": bool(getattr(policy_cfg, "relative_ee_action", False)),
            "mask_ee_pose_in_state": bool(getattr(policy_cfg, "mask_ee_pose_in_state", False)),
            "chunk_size": getattr(policy_cfg, "chunk_size", None),
            "n_action_steps": getattr(policy_cfg, "n_action_steps", None),
            "n_obs_steps": getattr(policy_cfg, "n_obs_steps", None),
        },
        "selection": {
            "sample_index": sample_index,
            "dataset_index": dataset_index,
            "action_step": action_step,
            "max_action_steps": max_action_steps,
        },
        "preprocessor_steps": preprocessor_steps,
        "raw_observation_state": _vector_to_named_dict(
            _select_state_vector(raw_batch, sample_index=sample_index),
            state_names,
        ),
        "preprocessed_observation_state": _vector_to_named_dict(
            _select_state_vector(relative_batch_before_qoff, sample_index=sample_index),
            state_names,
        ),
        "masked_observation_state": _vector_to_named_dict(
            _select_state_vector(masked_batch, sample_index=sample_index),
            state_names,
        ),
        "raw_absolute_action": _vector_to_named_dict(
            _select_action_vector(raw_batch, sample_index=sample_index, action_step=action_step),
            action_names,
        ),
        "relative_action_before_qoff": _vector_to_named_dict(
            _select_action_vector(relative_batch_before_qoff, sample_index=sample_index, action_step=action_step),
            action_names,
        ),
        "processed_action_after_qoff": _vector_to_named_dict(
            _select_action_vector(processed_batch_after_qoff, sample_index=sample_index, action_step=action_step),
            action_names,
        ),
        "raw_absolute_action_head": raw_action_head,
        "relative_action_before_qoff_head": relative_action_head,
        "processed_action_after_qoff_head": processed_action_head,
        "raw_absolute_action_selected_step_head": raw_action_head,
        "relative_action_before_qoff_selected_step_head": relative_action_head,
        "processed_action_after_qoff_selected_step_head": processed_action_head,
        # Backward-compatible aliases. These now explicitly refer to the fully processed
        # training target after qoff when qoff is enabled.
        "processed_relative_action": _vector_to_named_dict(
            _select_action_vector(processed_batch_after_qoff, sample_index=sample_index, action_step=action_step),
            action_names,
        ),
        "processed_relative_action_head": processed_action_head,
        "processed_relative_action_selected_step_head": processed_action_head,
    }


def build_batch_inspection_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("Cannot build summary from an empty report list.")

    raw_observation_states = [report["raw_observation_state"] for report in reports]
    preprocessed_states = [report["preprocessed_observation_state"] for report in reports]
    masked_states = [report["masked_observation_state"] for report in reports]
    raw_actions = [report["raw_absolute_action"] for report in reports]
    relative_actions_before_qoff = [report["relative_action_before_qoff"] for report in reports]
    processed_actions_after_qoff = [report["processed_action_after_qoff"] for report in reports]
    masked_state_mean_abs = _mean_abs_named_dict(masked_states)
    masked_state_max_abs = _max_abs_named_dict(masked_states)
    always_zero_masked_keys = [
        key for key, max_abs in masked_state_max_abs.items() if max_abs == 0.0
    ]

    return {
        "contract": reports[0]["contract"],
        "selection": {
            "sample_count": len(reports),
            "dataset_indices": [report["selection"]["dataset_index"] for report in reports],
            "action_step": reports[0]["selection"]["action_step"],
            "max_action_steps": reports[0]["selection"]["max_action_steps"],
        },
        "preprocessor_steps": reports[0]["preprocessor_steps"],
        "raw_observation_state_mean_abs": _mean_abs_named_dict(raw_observation_states),
        "preprocessed_observation_state_mean_abs": _mean_abs_named_dict(preprocessed_states),
        "masked_observation_state_mean_abs": masked_state_mean_abs,
        "masked_observation_state_max_abs": masked_state_max_abs,
        "masked_observation_state_always_zero_keys": always_zero_masked_keys,
        "raw_absolute_action_mean_abs": _mean_abs_named_dict(raw_actions),
        "raw_absolute_action_max_abs": _max_abs_named_dict(raw_actions),
        "relative_action_before_qoff_mean_abs": _mean_abs_named_dict(relative_actions_before_qoff),
        "relative_action_before_qoff_max_abs": _max_abs_named_dict(relative_actions_before_qoff),
        "processed_action_after_qoff_mean_abs": _mean_abs_named_dict(processed_actions_after_qoff),
        "processed_action_after_qoff_max_abs": _max_abs_named_dict(processed_actions_after_qoff),
        "gripper_delta_mean_abs": float(
            sum(
                abs(raw["gripper"] - relative["gripper"])
                for raw, relative in zip(raw_actions, relative_actions_before_qoff)
            )
            / len(reports)
        ),
        "relative_action_before_qoff_rotation_identity_error_mean_abs": {
            "qx": float(sum(abs(action["qx"]) for action in relative_actions_before_qoff) / len(reports)),
            "qy": float(sum(abs(action["qy"]) for action in relative_actions_before_qoff) / len(reports)),
            "qz": float(sum(abs(action["qz"]) for action in relative_actions_before_qoff) / len(reports)),
            "qw_error": float(sum(abs(1.0 - action["qw"]) for action in relative_actions_before_qoff) / len(reports)),
        },
        # Backward-compatible aliases for the final training target.
        "processed_relative_action_mean_abs": _mean_abs_named_dict(processed_actions_after_qoff),
        "processed_relative_action_max_abs": _max_abs_named_dict(processed_actions_after_qoff),
        "processed_relative_rotation_identity_error_mean_abs": {
            "qx": float(sum(abs(action["qx"]) for action in processed_actions_after_qoff) / len(reports)),
            "qy": float(sum(abs(action["qy"]) for action in processed_actions_after_qoff) / len(reports)),
            "qz": float(sum(abs(action["qz"]) for action in processed_actions_after_qoff) / len(reports)),
            "qw_error": float(sum(abs(1.0 - action["qw"]) for action in processed_actions_after_qoff) / len(reports)),
        },
    }


def _resolve_sample_indices(args: argparse.Namespace, dataset_length: int) -> list[int]:
    if args.sample_indices:
        sample_indices = [int(token.strip()) for token in args.sample_indices.split(",") if token.strip()]
    else:
        start_index = args.sample_index if args.sample_index is not None else args.start_index
        sample_indices = [start_index + i * args.sample_stride for i in range(args.num_samples)]

    if not sample_indices:
        raise ValueError("No sample indices were resolved for inspection.")

    invalid = [index for index in sample_indices if index < 0 or index >= dataset_length]
    if invalid:
        raise IndexError(f"Sample indices out of range for dataset length {dataset_length}: {invalid}")
    return sample_indices


def _write_jsonl(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for report in reports:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")


def load_train_config(config_path: Path, cli_overrides: list[str]) -> TrainPipelineConfig:
    register_third_party_plugins()
    return draccus.parse(TrainPipelineConfig, config_path=str(config_path), args=cli_overrides)


def main(argv: list[str] | None = None) -> int:
    args, cli_overrides = parse_args(argv)
    cfg = load_train_config(args.config_path, cli_overrides)
    cfg.validate()
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size
    cfg.policy.device = args.device

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs: dict[str, Any] = {"dataset_stats": dataset.meta.stats}
    if cfg.policy.type == "act" and getattr(cfg.policy, "action_chunk_quantile_normalization", False):
        dataset_root = Path(cfg.dataset.root) if cfg.dataset.root is not None else dataset.root
        processor_kwargs["action_chunk_stats"] = load_action_chunk_stats(cfg.policy, dataset_root)

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": args.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    sample_indices = _resolve_sample_indices(args, len(dataset))
    reports: list[dict[str, Any]] = []
    preprocessor_steps = [type(step).__name__ for step in preprocessor.steps]

    for dataset_index in sample_indices:
        raw_batch = default_collate([dataset[dataset_index]])
        relative_batch_before_qoff, processed_batch_after_qoff = _run_preprocessor_stages(preprocessor, raw_batch)
        masked_batch = (
            policy._mask_robot_state_features(deepcopy(relative_batch_before_qoff))
            if hasattr(policy, "_mask_robot_state_features")
            else deepcopy(relative_batch_before_qoff)
        )

        state_names = _feature_names(OBS_STATE, dataset.meta.features.get(OBS_STATE), raw_batch[OBS_STATE].shape[-1])
        action_names = _feature_names(ACTION, dataset.meta.features.get(ACTION), raw_batch[ACTION].shape[-1])
        reports.append(
            build_batch_inspection_report(
                raw_batch,
                relative_batch_before_qoff,
                processed_batch_after_qoff,
                masked_batch,
                state_names=state_names,
                action_names=action_names,
                sample_index=0,
                dataset_index=dataset_index,
                action_step=args.action_step,
                max_action_steps=args.max_action_steps,
                policy_cfg=cfg.policy,
                preprocessor_steps=preprocessor_steps,
            )
        )

    summary = build_batch_inspection_summary(reports)

    if args.jsonl_path is not None:
        _write_jsonl(args.jsonl_path, reports)
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if len(reports) == 1 and args.jsonl_path is None and args.summary_path is None:
        print(json.dumps(reports[0], ensure_ascii=False, indent=2))
    else:
        output: dict[str, Any] = {"summary": summary}
        if args.jsonl_path is not None:
            output["jsonl_path"] = str(args.jsonl_path)
        if args.summary_path is not None:
            output["summary_path"] = str(args.summary_path)
        if len(reports) > 1 and args.jsonl_path is None:
            output["reports"] = reports
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
