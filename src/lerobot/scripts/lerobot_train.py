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
import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

# Silence HuggingFace tokenizers fork/parallelism warnings and avoid potential deadlocks.
# If the user has already set TOKENIZERS_PARALLELISM explicitly, we respect their choice.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import (
    IMAGENET_STATS,
    ImageTransforms,
    make_dataset,
    resolve_delta_timestamps,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.ot_train import make_ot_dataloader
from lerobot.policies.ot_train.ot_loss import OTLossConfig, OTFeatureSpec, compute_ot_loss_for_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    ot_src_obs: dict[str, torch.Tensor] | None = None,
    ot_tgt_obs: dict[str, torch.Tensor] | None = None,
    ot_loss_cfg: OTLossConfig | None = None,
    lambda_ot: float = 0.0,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        ot_src_obs / ot_tgt_obs: Optional raw observations for feature-based OT loss.
        ot_loss_cfg: Optional OTLossConfig describing OT feature terms.
        lambda_ot: Global weight for the OT loss term.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Standard BC loss on the main batch
        loss, output_dict = policy.forward(batch)

        # Optional: add OT loss when OT observations and a loss config are provided.
        ot_info: dict[str, float] = {}

        if lambda_ot != 0.0:
            # Feature-based OT loss using raw src / tgt observations and an
            # OTLossConfig. This implements the "canonical" OT described in
            # OT_LOSS_DESIGN.md.
            if ot_src_obs is not None and ot_tgt_obs is not None and ot_loss_cfg is not None:
                ot_loss, ot_metrics = compute_ot_loss_for_policy(
                    policy=policy,
                    src_obs=ot_src_obs,
                    tgt_obs=ot_tgt_obs,
                    cfg=ot_loss_cfg,
                )
                loss = loss + float(lambda_ot) * ot_loss
                ot_info["ot_loss"] = float(ot_loss.item())
                for k, v in ot_metrics.items():
                    # Avoid duplicating ot_loss as another key; keep only train/ot_loss
                    if k == "ot_loss":
                        continue
                    if isinstance(v, (int, float)):
                        # Log OT metrics with their native names (e.g. 'ot_pi_sum', 'ot_cost/…')
                        ot_info[k] = float(v)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    # attach OT metrics if any
    if ot_info and lambda_ot != 0.0:
        if output_dict is None:
            output_dict = {}
        output_dict.update(ot_info)
    return train_metrics, output_dict



def _slice_batch(batch: dict, n: int) -> dict:
    """Take the first n samples along batch dim for every tensor leaf in a nested dict."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = _slice_batch(v, n)
        elif hasattr(v, "shape") and v.__class__.__name__ == "Tensor":
            out[k] = v[:n]
        else:
            out[k] = v
    return out


_CONCAT_DEBUG_LIMIT = 8  # emit at most N debug lines from concat alignment
_concat_debug_count = 0


def _concat_two_batches(b1: dict, b2: dict) -> dict:
    """Concatenate two nested-batch dicts along batch dim for matching tensor leaves.

    If a key exists in only one input, keep that value.
    """
    keys = set(b1.keys()) | set(b2.keys())
    out = {}
    # Only align ranks for scalar/vector-like keys; never try to expand images/videos.
    RANK_ALIGN_KEYS = {"action", "observation.state", "observation.environment_state"}

    def _infer_batch_size(d: dict) -> int | None:
        """Try to infer batch size from a nested dict by looking for a Tensor leaf.

        Prefer the 'action' key when available.
        """
        try:
            v = d.get("action", None)
            if hasattr(v, "shape") and v.__class__.__name__ == "Tensor":
                return int(v.shape[0])
        except Exception:
            pass
        for _v in d.values():
            if isinstance(_v, dict):
                bs = _infer_batch_size(_v)
                if bs is not None:
                    return bs
            elif hasattr(_v, "shape") and _v.__class__.__name__ == "Tensor":
                return int(_v.shape[0])
        return None
    for k in keys:
        in1 = k in b1; in2 = k in b2
        if in1 and in2:
            v1, v2 = b1[k], b2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                out[k] = _concat_two_batches(v1, v2)
            elif (
                hasattr(v1, "shape")
                and v1.__class__.__name__ == "Tensor"
                and hasattr(v2, "shape")
                and v2.__class__.__name__ == "Tensor"
            ):
                # When mixing batches coming from different DataLoaders, Accelerate may have already
                # moved one to the GPU while the other is still on CPU. Ensure both tensors are on
                # the same device before concatenating along the batch dimension.
                import torch as _torch

                if v1.device != v2.device:
                    # Prefer the non-CPU device if any; otherwise keep CPU.
                    if v1.device.type != "cpu":
                        v2 = v2.to(v1.device, non_blocking=(v1.device.type == "cuda"))
                    elif v2.device.type != "cpu":
                        v1 = v1.to(v2.device, non_blocking=(v2.device.type == "cuda"))
                    # If both are CPU, do nothing.
                    global _concat_debug_count
                    if _concat_debug_count < _CONCAT_DEBUG_LIMIT:
                        logging.info(f"concat-align device for key='{k}': {v1.device} vs {v2.device}")
                        _concat_debug_count += 1

                # Align ranks if needed (e.g., (B, T, D) with (B, D)).
                if v1.dim() != v2.dim() and k in RANK_ALIGN_KEYS:
                    # Handle common 3D vs 2D mismatch for whitelisted keys only.
                    if v1.dim() == 3 and v2.dim() == 2 and v1.size(-1) == v2.size(-1):
                        v2 = v2.unsqueeze(1).repeat(1, v1.size(1), 1)
                        if _concat_debug_count < _CONCAT_DEBUG_LIMIT:
                            logging.info(
                                f"concat-align rank (expand v2) key='{k}': {tuple(v1.shape)} vs {tuple(v2.shape)}"
                            )
                            _concat_debug_count += 1
                    elif v1.dim() == 2 and v2.dim() == 3 and v1.size(-1) == v2.size(-1):
                        v1 = v1.unsqueeze(1).repeat(1, v2.size(1), 1)
                        if _concat_debug_count < _CONCAT_DEBUG_LIMIT:
                            logging.info(
                                f"concat-align rank (expand v1) key='{k}': {tuple(v1.shape)} vs {tuple(v2.shape)}"
                            )
                            _concat_debug_count += 1
                    # Otherwise fall through; torch.cat will raise if shapes incompatible.

                out[k] = _torch.cat([v1, v2], dim=0)
            else:
                out[k] = v1
        elif in1:
            v1 = b1[k]
            # If only present in batch1 and tensor-like, try to pad with sensible defaults for batch2.
            if hasattr(v1, "shape") and v1.__class__.__name__ == "Tensor":
                import torch as _torch
                # Special-case known mask: action_is_pad -> default False
                if k == "action_is_pad" and v1.dim() >= 1:
                    bs2 = _infer_batch_size(b2) or 0
                    pad_shape = (bs2,) + tuple(v1.shape[1:])
                    pad = _torch.zeros(pad_shape, dtype=v1.dtype, device=v1.device)
                    out[k] = _torch.cat([v1, pad], dim=0)
                    if _concat_debug_count < _CONCAT_DEBUG_LIMIT:
                        logging.info(
                            f"concat-pad missing action_is_pad (left present) with zeros; new shape={tuple(out[k].shape)}"
                        )
                        _concat_debug_count += 1
                else:
                    out[k] = v1
            else:
                out[k] = v1
        else:
            v2 = b2[k]
            if hasattr(v2, "shape") and v2.__class__.__name__ == "Tensor":
                import torch as _torch
                if k == "action_is_pad" and v2.dim() >= 1:
                    bs1 = _infer_batch_size(b1) or 0
                    pad_shape = (bs1,) + tuple(v2.shape[1:])
                    pad = _torch.zeros(pad_shape, dtype=v2.dtype, device=v2.device)
                    out[k] = _torch.cat([pad, v2], dim=0)
                    if _concat_debug_count < _CONCAT_DEBUG_LIMIT:
                        logging.info(
                            f"concat-pad missing action_is_pad (right present) with zeros; new shape={tuple(out[k].shape)}"
                        )
                        _concat_debug_count += 1
                else:
                    out[k] = v2
            else:
                out[k] = v2
    return out



def _build_ot_loss_config_from_object(loss_cfg: object | None) -> OTLossConfig | None:
    """Convert a generic JSON/YAML-style object into an `OTLossConfig`.

    This lets us keep `OTConfig.loss_config` typed as a plain object in the
    config tree (to avoid import cycles) while still feeding a strongly-typed
    `OTLossConfig` instance into the OT loss module at training time.
    """
    if loss_cfg is None:
        return None

    # Already a fully-formed config (e.g. set programmatically).
    if isinstance(loss_cfg, OTLossConfig):
        return loss_cfg

    # JSON / dict-style representation coming from train_config.{json,yaml}.
    if isinstance(loss_cfg, dict):
        raw_features = loss_cfg.get("features", [])
        reg = float(loss_cfg.get("reg", 0.01))
        tau_src_val = loss_cfg.get("tau_src", None)
        tau_tgt_val = loss_cfg.get("tau_tgt", None)
        tau_src = float(tau_src_val) if tau_src_val is not None else None
        tau_tgt = float(tau_tgt_val) if tau_tgt_val is not None else None
        heuristic = bool(loss_cfg.get("heuristic", False))

        features: list[OTFeatureSpec] = []
        for feat_cfg in raw_features:
            # Allow mixing dicts and already-constructed OTFeatureSpec instances.
            if isinstance(feat_cfg, OTFeatureSpec):
                features.append(feat_cfg)
                continue
            if not isinstance(feat_cfg, dict):
                raise TypeError(
                    f"Unsupported OT feature spec type {type(feat_cfg)}; expected dict or OTFeatureSpec."
                )

            # dim_slice is represented in JSON as [start, stop]; optionally also
            # accept a mapping {"start": ..., "stop": ...} for flexibility.
            dim_slice_cfg = feat_cfg.get("dim_slice", None)
            dim_slice = None
            if isinstance(dim_slice_cfg, (list, tuple)):
                if len(dim_slice_cfg) != 2:
                    raise ValueError(
                        f"dim_slice list must have length 2 [start, stop], got {dim_slice_cfg!r}"
                    )
                start, stop = dim_slice_cfg
                dim_slice = slice(start, stop)
            elif isinstance(dim_slice_cfg, dict):
                # e.g. {"start": 0, "stop": 8}
                start = dim_slice_cfg.get("start", None)
                stop = dim_slice_cfg.get("stop", None)
                dim_slice = slice(start, stop)
            elif dim_slice_cfg is None:
                dim_slice = None
            else:
                raise TypeError(
                    f"Unsupported dim_slice type {type(dim_slice_cfg)}; expected [start, stop], "
                    f"{{'start': ..., 'stop': ...}}, or null."
                )

            features.append(
                OTFeatureSpec(
                    src_key=feat_cfg["src_key"],
                    tgt_key=feat_cfg.get("tgt_key", feat_cfg["src_key"]),
                    dim_slice=dim_slice,
                    use_learned_embed=bool(feat_cfg.get("use_learned_embed", False)),
                    embed_name=feat_cfg.get("embed_name"),
                    weight_embed=float(feat_cfg.get("weight_embed", 1.0)),
                    weight_label=float(feat_cfg.get("weight_label", 1.0)),
                    term_name=feat_cfg.get("term_name"),
                )
            )

        return OTLossConfig(
            features=features,
            reg=reg,
            tau_src=tau_src,
            tau_tgt=tau_tgt,
            heuristic=heuristic,
        )

    raise TypeError(
        f"Unsupported ot.loss_config type {type(loss_cfg)}; expected dict, OTLossConfig, or None."
    )


def build_train_val_episode_split(cfg: TrainPipelineConfig) -> tuple[list[int], list[int]]:
    """Split available episodes into train/val by episode index.

    Follows the convention used in `offline_eval_ckpts.py`: the last ~10% of the
    available episodes are used as validation. If `cfg.dataset.episodes` is set,
    the split is performed within that subset for fresh runs. On resume, we treat
    `cfg.dataset.episodes` as the training episodes and reconstruct validation
    episodes from the full metadata.
    """
    if cfg.dataset is None:
        raise ValueError("Offline dataset config is required to build a train/val split.")

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )

    # When resuming, assume that cfg.dataset.episodes already corresponds to the
    # train split from a previous run and reconstruct validation episodes as the
    # complement in the full metadata.
    if cfg.resume and cfg.dataset.episodes is not None:
        train_eps = list(cfg.dataset.episodes)
        all_eps = list(range(ds_meta.total_episodes))
        train_set = set(train_eps)
        val_eps = [ep for ep in all_eps if ep not in train_set]
        return train_eps, val_eps

    # Fresh run: build the split from the available episodes.
    if cfg.dataset.episodes is None:
        available_eps = list(range(ds_meta.total_episodes))
    else:
        available_eps = list(cfg.dataset.episodes)

    num_eps = len(available_eps)
    if num_eps == 0:
        raise ValueError("No episodes available in dataset to split into train/val.")

    n_val = max(1, num_eps // 10)

    if num_eps == 1:
        train_eps = available_eps
        val_eps = []
    else:
        val_eps = available_eps[-n_val:]
        train_eps = available_eps[:-n_val]
        if len(train_eps) == 0:
            train_eps = [available_eps[0]]
            val_eps = available_eps[1:]

    return train_eps, val_eps


@torch.no_grad()
def offline_eval_split(
    dataloader: torch.utils.data.DataLoader,
    policy: PreTrainedPolicy,
    preprocessor,
    accelerator: Accelerator,
    max_batches: int = 20,
) -> dict | None:
    """Run a quick offline eval on a held-out validation split."""
    if dataloader is None:
        return None

    model = accelerator.unwrap_model(policy)
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    n = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = preprocessor(batch)
        with accelerator.autocast():
            loss, out = model.forward(batch)
        total_loss += float(loss.item())
        # Some policies (e.g., diffusion) return None as output_dict
        total_l1 += float(out.get("l1_loss", 0.0)) if isinstance(out, dict) else 0.0
        n += 1

    if was_training:
        model.train()

    if n == 0:
        return None

    return {
        "offline_eval/avg_loss": total_loss / n,
        "offline_eval/avg_l1": total_l1 / n,
        "offline_eval/n_batches": n,
    }


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Build a deterministic train/val split at the episode level so that
    # validation episodes are never seen during training.
    train_episodes, val_episodes = build_train_val_episode_split(cfg)
    cfg.dataset.episodes = train_episodes

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info(
            "Creating dataset (train episodes only): "
            f"{len(train_episodes)} train, {len(val_episodes)} val"
        )
        dataset = make_dataset(cfg)

        # Optional: create a small hold-out for offline eval when no env is configured.
        # We keep this lightweight and self-contained to provide an eval curve similar to original ACT
        # when users train purely offline on recorded datasets.
        offline_eval_enabled = (
            cfg.eval_freq > 0 and cfg.env is None and not cfg.dataset.streaming and getattr(dataset, "num_episodes", 0) >= 2
        )
        if offline_eval_enabled:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.factory import resolve_delta_timestamps, ImageTransforms, IMAGENET_STATS

            num_eps = int(dataset.num_episodes)
            # 10% episodes as validation (at least 1)
            n_val = max(1, num_eps // 10)
            # use last episodes as val to keep split deterministic
            val_eps = list(range(num_eps - n_val, num_eps))
            train_eps = list(range(0, num_eps - n_val))
            if len(train_eps) == 0:
                # fallback: keep at least 1 train episode
                train_eps = [0]
                val_eps = [i for i in range(1, num_eps)]

            # Rebuild train/val datasets with identical transforms and delta timestamps
            img_tf = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
            delta_ts = resolve_delta_timestamps(cfg.policy, dataset.meta)
            train_dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=train_eps,
                image_transforms=img_tf,
                delta_timestamps=delta_ts,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
            val_dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=val_eps,
                image_transforms=img_tf,
                delta_timestamps=delta_ts,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
            # Apply ImageNet stats override if requested (same as make_dataset)
            if cfg.dataset.use_imagenet_stats:
                import torch as _torch
                for key in train_dataset.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        train_dataset.meta.stats[key][stats_type] = _torch.tensor(stats, dtype=_torch.float32)
                for key in val_dataset.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        val_dataset.meta.stats[key][stats_type] = _torch.tensor(stats, dtype=_torch.float32)
            # Replace dataset with train split for the rest of the pipeline
            dataset = train_dataset
            logging.info(
                f"Offline eval enabled (no env). Split episodes -> train: {len(train_eps)}, val: {len(val_eps)}"
            )
        else:
            val_dataset = None

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
        offline_eval_enabled = (
            cfg.eval_freq > 0 and cfg.env is None and not cfg.dataset.streaming and getattr(dataset, "num_episodes", 0) >= 2
        )
        if offline_eval_enabled:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.factory import resolve_delta_timestamps, ImageTransforms, IMAGENET_STATS

            num_eps = int(dataset.num_episodes)
            n_val = max(1, num_eps // 10)
            val_eps = list(range(num_eps - n_val, num_eps))
            train_eps = list(range(0, num_eps - n_val))
            if len(train_eps) == 0:
                train_eps = [0]
                val_eps = [i for i in range(1, num_eps)]

            img_tf = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
            delta_ts = resolve_delta_timestamps(cfg.policy, dataset.meta)
            train_dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=train_eps,
                image_transforms=img_tf,
                delta_timestamps=delta_ts,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
            val_dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=val_eps,
                image_transforms=img_tf,
                delta_timestamps=delta_ts,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
            if cfg.dataset.use_imagenet_stats:
                import torch as _torch
                for key in train_dataset.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        train_dataset.meta.stats[key][stats_type] = _torch.tensor(stats, dtype=_torch.float32)
                for key in val_dataset.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        val_dataset.meta.stats[key][stats_type] = _torch.tensor(stats, dtype=_torch.float32)
            dataset = train_dataset
        else:
            val_dataset = None

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # Build a validation dataloader on held-out episodes for offline eval when no env is used.
    val_dataloader: torch.utils.data.DataLoader | None = None
    if cfg.env is None and len(val_episodes) > 0 and is_main_process:
        img_tf = (
            ImageTransforms(cfg.dataset.image_transforms)
            if cfg.dataset.image_transforms.enable
            else None
        )
        delta_ts = resolve_delta_timestamps(cfg.policy, dataset.meta)

        val_dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=val_episodes,
            image_transforms=img_tf,
            delta_timestamps=delta_ts,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )

        if cfg.dataset.use_imagenet_stats:
            for key in val_dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    val_dataset.meta.stats[key][stats_type] = torch.tensor(
                        stats, dtype=torch.float32
                    )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=device.type == "cuda",
        )

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Optional: build OT dataloader and loss config if enabled
    ot_dataloader = None
    ot_loss_cfg: OTLossConfig | None = None
    if cfg.ot.enable:
        if cfg.ot.src_repo_id is None or cfg.ot.pair_info_path is None:
            raise ValueError("OT enabled but src_repo_id or pair_info_path is None")
        # Make the OT source dataset use the same image transforms and stats as the
        # main (target) dataset to keep visual distributions consistent for OT
        # embeddings. This mirrors make_dataset's behavior.
        img_tf_src = (
            ImageTransforms(cfg.dataset.image_transforms)
            if cfg.dataset.image_transforms.enable
            else None
        )
        # 让 BC 源数据集在时间维与主数据集一致（对齐 action 窗口与 _is_pad 掩码），以匹配 ot-sim2real 的做法：
        # 使用与主数据相同的 delta_timestamps（由 policy 的 *delta_indices 推导）。
        ds_src_meta = LeRobotDatasetMetadata(
            cfg.ot.src_repo_id,
            root=cfg.ot.src_root if cfg.ot.src_root else cfg.dataset.root,
            revision=cfg.dataset.revision,
        )
        delta_timestamps_src = resolve_delta_timestamps(cfg.policy, ds_src_meta)

        ds_src = LeRobotDataset(
            cfg.ot.src_repo_id,
            root=cfg.ot.src_root if cfg.ot.src_root else cfg.dataset.root,
            image_transforms=img_tf_src,
            delta_timestamps=delta_timestamps_src,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        if cfg.dataset.use_imagenet_stats:
            # Align source dataset camera stats to ImageNet like the main dataset
            for key in ds_src.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds_src.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        ot_batch_size = max(1, int(cfg.batch_size * cfg.ot.batch_ratio))
        ot_dataloader = make_ot_dataloader(
            ds_src=ds_src,
            ds_tgt=dataset,
            pair_info_path=cfg.ot.pair_info_path,
            batch_size=ot_batch_size,
            obs_keys=cfg.ot.obs_keys,
            action_key="action",
            base_index_src=cfg.ot.base_index_src,
            base_index_tgt=cfg.ot.base_index_tgt,
            window_size=cfg.ot.window_size if cfg.ot.window_size > 0 else None,
            sharpness=getattr(cfg.ot, 'sharpness', 0.0),
            no_window=getattr(cfg.ot, 'no_window', False),
            topk_src_episodes=getattr(cfg.ot, 'topk_src_episodes', None),
            num_workers=cfg.num_workers,
        )
        # Mix source data into BC: make a small BC-src dataloader and split the main batch.
        bc_tgt_bs = (cfg.batch_size + 1) // 2
        bc_src_bs = max(1, cfg.batch_size - bc_tgt_bs)
        bc_src_dataloader = torch.utils.data.DataLoader(
            ds_src,
            num_workers=cfg.num_workers,
            batch_size=bc_src_bs,
            shuffle=True,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        # Convert any JSON/YAML-style OT loss config into a strongly-typed
        # OTLossConfig instance understood by the OT loss utilities.
        ot_loss_cfg = _build_ot_loss_config_from_object(cfg.ot.loss_config)

    # Optional validation dataloader for offline eval (no env)
    if cfg.env is None and val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=max(1, cfg.num_workers // 2),
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
    else:
        val_dataloader = None

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if val_dataloader is not None:
        policy, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, val_dataloader, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    # Note: ot_dataloader stays on CPU workers (no .to(device)); yielded tensors are moved by preprocessor
    dl_iter = cycle(dataloader)
    # Prepare OT iterator if enabled
    if cfg.ot.enable and ot_dataloader is not None:
        ot_iter = cycle(ot_dataloader)
    else:
        ot_iter = None

    if "bc_src_dataloader" in locals():
        bc_src_iter = cycle(bc_src_dataloader)
    else:
        bc_src_iter = None

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        raw_batch = next(dl_iter)
        if bc_src_iter is not None:
            raw_batch_tgt = _slice_batch(raw_batch, bc_tgt_bs)
            raw_batch_src = next(bc_src_iter)
            raw_batch = _concat_two_batches(raw_batch_tgt, raw_batch_src)
        batch = preprocessor(raw_batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # If OT is enabled, extract raw src / tgt observations for feature-based OT.
        ot_src_obs: dict[str, torch.Tensor] | None = None
        ot_tgt_obs: dict[str, torch.Tensor] | None = None
        if cfg.ot.enable and ot_iter is not None:
            try:
                ot_pair = next(ot_iter)
                # Ensure OT loss can reference 'action' as a feature (to align with
                # ot-sim2real diffusion_policy_ot config where label='action'). We
                # inject actions into the obs dicts so that OTFeatureSpec with
                # src_key / tgt_key == 'action' is supported.
                ot_src_obs = dict(ot_pair["src"]["obs"])  # shallow copy
                ot_tgt_obs = dict(ot_pair["tgt"]["obs"])  # shallow copy
                if "actions" in ot_pair["src"] and ot_pair["src"]["actions"] is not None:
                    ot_src_obs["action"] = ot_pair["src"]["actions"]
                if "actions" in ot_pair["tgt"] and ot_pair["tgt"]["actions"] is not None:
                    ot_tgt_obs["action"] = ot_pair["tgt"]["actions"]
            except StopIteration:
                ot_src_obs = None
                ot_tgt_obs = None

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            ot_src_obs=ot_src_obs,
            ot_tgt_obs=ot_tgt_obs,
            ot_loss_cfg=ot_loss_cfg,
            lambda_ot=cfg.ot.lambda_ot if cfg.ot.enable else 0.0,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir
                        / "eval"
                        / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(
                        eval_info["overall"]["video_paths"][0],
                        step,
                        mode="eval",
                    )

            accelerator.wait_for_everyone()
        elif (cfg.env is None) and (val_dataloader is not None) and is_eval_step:
            # Offline eval on hold-out split: report avg loss and avg L1 over a handful of batches
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(
                    f"Offline eval (no env) at step {step} on hold-out episodes"
                )
            policy.eval()
            n_batches = 0
            total_loss = 0.0
            total_l1 = 0.0
            max_eval_batches = 20  # keep it light
            with torch.no_grad(), accelerator.autocast():
                for i, batch in enumerate(val_dataloader):
                    if i >= max_eval_batches:
                        break
                    batch = preprocessor(batch)
                    # Use policy.forward: in eval mode KL is disabled by design; report L1 as eval loss
                    loss, output_dict = policy.forward(batch)
                    total_loss += float(loss.item())
                    # Diffusion returns None for output_dict; guard accordingly
                    total_l1 += float(output_dict.get("l1_loss", 0.0)) if isinstance(output_dict, dict) else 0.0
                    n_batches += 1

            if n_batches > 0 and is_main_process:
                avg_loss = total_loss / n_batches
                avg_l1 = total_l1 / n_batches
                logging.info(
                    f"Offline eval: avg_loss={avg_loss:.4f}, avg_l1={avg_l1:.4f} over {n_batches} batches"
                )
                if wandb_logger:
                    wandb_logger.log_dict(
                        {
                            "offline_eval/avg_loss": avg_loss,
                            "offline_eval/avg_l1": avg_l1,
                            "offline_eval/n_batches": n_batches,
                        },
                        step,
                        mode="eval",
                    )
            policy.train()

        elif cfg.env is None and is_eval_step:
            # Offline eval on held-out validation episodes when no environment is configured.
            if is_main_process and val_dataloader is not None:
                eval_dict = offline_eval_split(
                    val_dataloader, policy, preprocessor, accelerator, max_batches=20
                )
                if eval_dict is not None:
                    logging.info(
                        "Offline eval at step %d: avg_loss=%.4f avg_l1=%.4f n_batches=%d",
                        step,
                        eval_dict["offline_eval/avg_loss"],
                        eval_dict["offline_eval/avg_l1"],
                        eval_dict["offline_eval/n_batches"],
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(eval_dict, step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        # if cfg.policy.push_to_hub:
        #     unwrapped_policy = accelerator.unwrap_model(policy)
        #     unwrapped_policy.push_model_to_hub(cfg)
        #     preprocessor.push_to_hub(cfg.policy.repo_id)
        #     postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
