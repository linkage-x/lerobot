#!/usr/bin/env python

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


DEFAULT_STEPS = ["040000", "060000", "100000"]
DEPLOYMENT_RANKING_WEIGHTS = {
    "first_action_l1": 0.25,
    "jerk_alignment": 0.20,
    "episode_p90_action_l1": 0.20,
    "episode_p90_first_action_l1": 0.15,
    "episode_p90_jerk_alignment": 0.15,
    "action_l1": 0.05,
}


@dataclass
class EpisodeMetrics:
    episode_index: int
    num_sequences: int
    num_valid_actions: int
    action_l1_mean: float
    first_action_l1_mean: float
    pred_jerk_l1_mean: float
    gt_jerk_l1_mean: float
    jerk_ratio_pred_over_gt: float
    tactile_ablation_l1_mean: float
    tactile_ablation_first_action_l1_mean: float
    tactile_ablation_relative_to_action_l1: float


@dataclass
class CheckpointRanking:
    checkpoint: str
    overall_rank: int
    composite_score: float
    action_l1_rank: int
    first_action_l1_rank: int
    jerk_alignment_rank: int
    episode_p90_action_l1_rank: int
    episode_p90_first_action_l1_rank: int
    episode_p90_jerk_alignment_rank: int


@dataclass
class CheckpointMetrics:
    checkpoint: str
    num_batches: int
    num_sequences: int
    num_valid_actions: int
    action_l1_mean: float
    first_action_l1_mean: float
    pred_jerk_l1_mean: float
    gt_jerk_l1_mean: float
    jerk_ratio_pred_over_gt: float
    tactile_ablation_l1_mean: float
    tactile_ablation_first_action_l1_mean: float
    tactile_ablation_relative_to_action_l1: float
    episode_p90_action_l1_mean: float
    episode_p90_first_action_l1_mean: float
    episode_p90_jerk_alignment: float
    episode_max_action_l1_mean: float
    episode_max_first_action_l1_mean: float
    episode_metrics: list[EpisodeMetrics]


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value_sum: float, count: int) -> None:
        self.total += float(value_sum)
        self.count += int(count)

    @property
    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


class EpisodeAccumulator:
    def __init__(self, episode_index: int) -> None:
        self.episode_index = episode_index
        self.num_sequences = 0
        self.num_valid_actions = 0
        self.action_l1 = RunningMean()
        self.first_action_l1 = RunningMean()
        self.pred_jerk = RunningMean()
        self.gt_jerk = RunningMean()
        self.tactile_ablation_l1 = RunningMean()
        self.tactile_ablation_first = RunningMean()

    def to_metrics(self) -> EpisodeMetrics:
        gt_jerk_mean = self.gt_jerk.mean
        action_l1_mean = self.action_l1.mean
        tactile_ablation_mean = self.tactile_ablation_l1.mean
        return EpisodeMetrics(
            episode_index=self.episode_index,
            num_sequences=self.num_sequences,
            num_valid_actions=self.num_valid_actions,
            action_l1_mean=action_l1_mean,
            first_action_l1_mean=self.first_action_l1.mean,
            pred_jerk_l1_mean=self.pred_jerk.mean,
            gt_jerk_l1_mean=gt_jerk_mean,
            jerk_ratio_pred_over_gt=(
                self.pred_jerk.mean / gt_jerk_mean
                if gt_jerk_mean and not math.isnan(gt_jerk_mean)
                else float("nan")
            ),
            tactile_ablation_l1_mean=tactile_ablation_mean,
            tactile_ablation_first_action_l1_mean=self.tactile_ablation_first.mean,
            tactile_ablation_relative_to_action_l1=(
                tactile_ablation_mean / action_l1_mean
                if self.action_l1.count > 0 and action_l1_mean != 0
                else float("nan")
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline comparison for FR3 ACT checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing checkpoints/.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=DEFAULT_STEPS,
        help="Checkpoint step directories to compare, e.g. 040000 060000 100000.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation dataloader batch size. Defaults to the checkpoint train_config value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Evaluation dataloader worker count. Defaults to the checkpoint train_config value.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum number of dataloader batches to evaluate. 0 means full dataset.",
    )
    parser.add_argument(
        "--tactile-keys",
        nargs="*",
        default=None,
        help="Optional tactile keys to zero during ablation. Defaults to checkpoint config tactile keys.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to save a Markdown comparison report.",
    )
    return parser.parse_args()


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = [item.clone() if isinstance(item, torch.Tensor) else item for item in value]
        else:
            cloned[key] = value
    return cloned


def zero_tactile_keys(batch: dict[str, Any], tactile_keys: list[str]) -> dict[str, Any]:
    ablated = clone_batch(batch)
    for key in tactile_keys:
        if key not in ablated:
            continue
        value = ablated[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Tactile key `{key}` is expected to be a tensor, got {type(value)}")
        ablated[key] = torch.zeros_like(value)
    return ablated


def jerk_stats(actions: torch.Tensor, action_is_pad: torch.Tensor) -> tuple[float, int]:
    total = 0.0
    count = 0
    valid_mask = ~action_is_pad
    for seq, seq_valid in zip(actions, valid_mask, strict=True):
        valid_len = int(seq_valid.sum().item())
        if valid_len < 3:
            continue
        jerk = seq[2:valid_len] - 2 * seq[1 : valid_len - 1] + seq[: valid_len - 2]
        total += jerk.abs().sum().item()
        count += jerk.numel()
    return total, count


def masked_l1_stats(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor) -> tuple[float, int]:
    valid_mask = (~action_is_pad).unsqueeze(-1)
    diff = (pred - target).abs() * valid_mask
    count = int(valid_mask.sum().item()) * pred.shape[-1]
    return diff.sum().item(), count


def first_action_l1_stats(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor) -> tuple[float, int]:
    valid_first = ~action_is_pad[:, 0]
    if not valid_first.any():
        return 0.0, 0
    diff = (pred[:, 0] - target[:, 0]).abs()[valid_first]
    return diff.sum().item(), diff.numel()


def sample_masked_l1_stats(
    pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor
) -> tuple[float, int]:
    valid_mask = ~action_is_pad
    if not valid_mask.any():
        return 0.0, 0
    diff = (pred - target).abs()[valid_mask]
    return diff.sum().item(), diff.numel()


def sample_first_action_l1_stats(
    pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor
) -> tuple[float, int]:
    if bool(action_is_pad[0].item()):
        return 0.0, 0
    diff = (pred[0] - target[0]).abs()
    return diff.sum().item(), diff.numel()


def sample_jerk_stats(actions: torch.Tensor, action_is_pad: torch.Tensor) -> tuple[float, int]:
    valid_len = int((~action_is_pad).sum().item())
    if valid_len < 3:
        return 0.0, 0
    jerk = actions[2:valid_len] - 2 * actions[1 : valid_len - 1] + actions[: valid_len - 2]
    return jerk.abs().sum().item(), jerk.numel()


def build_dataset_from_checkpoint(
    checkpoint_dir: Path, dataset_root: Path, batch_size: int | None, num_workers: int | None, device: str
) -> tuple[torch.utils.data.DataLoader, int, int]:
    train_cfg_path = checkpoint_dir / "train_config.json"
    train_cfg = TrainPipelineConfig.from_pretrained(
        train_cfg_path,
        cli_args=[
            f"--dataset.root={dataset_root}",
            *([f"--batch_size={batch_size}"] if batch_size is not None else []),
            *([f"--num_workers={num_workers}"] if num_workers is not None else []),
        ],
    )
    dataset = make_dataset(train_cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        drop_last=False,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )
    return dataloader, train_cfg.batch_size, train_cfg.num_workers


def load_policy_bundle(
    checkpoint_dir: Path, device: str
) -> tuple[Any, Any, Any, list[str]]:
    policy_cfg = PreTrainedConfig.from_pretrained(
        checkpoint_dir,
        cli_overrides=[f"--device={device}", "--use_amp=false"],
    )
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(checkpoint_dir, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    tactile_keys = getattr(policy_cfg, "tactile_feature_keys", []) or []
    return policy.eval(), preprocessor, postprocessor, tactile_keys


def evaluate_checkpoint(
    checkpoint_dir: Path,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int,
    tactile_keys_override: list[str] | None,
) -> CheckpointMetrics:
    policy, preprocessor, postprocessor, checkpoint_tactile_keys = load_policy_bundle(checkpoint_dir, device)
    tactile_keys = tactile_keys_override if tactile_keys_override is not None else checkpoint_tactile_keys
    total_batches = len(dataloader)

    action_l1 = RunningMean()
    first_action_l1 = RunningMean()
    pred_jerk = RunningMean()
    gt_jerk = RunningMean()
    tactile_ablation_l1 = RunningMean()
    tactile_ablation_first = RunningMean()

    num_batches = 0
    num_sequences = 0
    num_valid_actions = 0
    episode_accumulators: dict[int, EpisodeAccumulator] = {}

    with torch.inference_mode():
        for batch_idx, raw_batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                effective_total = min(total_batches, max_batches) if max_batches > 0 else total_batches
                print(
                    f"  batch {batch_idx + 1}/{effective_total} "
                    f"(checkpoint={checkpoint_dir.parent.name})"
                )

            proc_batch = preprocessor(clone_batch(raw_batch))
            pred_actions = policy.predict_action_chunk(proc_batch)
            pred_actions = postprocessor(pred_actions)
            pred_actions = pred_actions.detach()

            gt_actions = raw_batch["action"].to(pred_actions.device)
            action_is_pad = raw_batch["action_is_pad"].to(pred_actions.device).bool()
            episode_indices = raw_batch["episode_index"].tolist()

            ablated_actions = None

            if tactile_keys:
                ablated_proc_batch = zero_tactile_keys(proc_batch, tactile_keys)
                ablated_actions = policy.predict_action_chunk(ablated_proc_batch)
                ablated_actions = postprocessor(ablated_actions)
                ablated_actions = ablated_actions.detach()

            for sample_idx, episode_index in enumerate(episode_indices):
                pred_seq = pred_actions[sample_idx]
                gt_seq = gt_actions[sample_idx]
                pad_seq = action_is_pad[sample_idx]

                l1_sum, l1_count = sample_masked_l1_stats(pred_seq, gt_seq, pad_seq)
                first_sum, first_count = sample_first_action_l1_stats(pred_seq, gt_seq, pad_seq)
                pred_jerk_sum, pred_jerk_count = sample_jerk_stats(pred_seq, pad_seq)
                gt_jerk_sum, gt_jerk_count = sample_jerk_stats(gt_seq, pad_seq)

                action_l1.update(l1_sum, l1_count)
                first_action_l1.update(first_sum, first_count)
                pred_jerk.update(pred_jerk_sum, pred_jerk_count)
                gt_jerk.update(gt_jerk_sum, gt_jerk_count)

                episode_accumulator = episode_accumulators.setdefault(
                    int(episode_index), EpisodeAccumulator(int(episode_index))
                )
                episode_accumulator.num_sequences += 1
                episode_accumulator.num_valid_actions += int((~pad_seq).sum().item())
                episode_accumulator.action_l1.update(l1_sum, l1_count)
                episode_accumulator.first_action_l1.update(first_sum, first_count)
                episode_accumulator.pred_jerk.update(pred_jerk_sum, pred_jerk_count)
                episode_accumulator.gt_jerk.update(gt_jerk_sum, gt_jerk_count)

                if ablated_actions is not None:
                    ablated_seq = ablated_actions[sample_idx]
                    ablation_sum, ablation_count = sample_masked_l1_stats(ablated_seq, pred_seq, pad_seq)
                    ablation_first_sum, ablation_first_count = sample_first_action_l1_stats(
                        ablated_seq, pred_seq, pad_seq
                    )
                    tactile_ablation_l1.update(ablation_sum, ablation_count)
                    tactile_ablation_first.update(ablation_first_sum, ablation_first_count)
                    episode_accumulator.tactile_ablation_l1.update(ablation_sum, ablation_count)
                    episode_accumulator.tactile_ablation_first.update(
                        ablation_first_sum, ablation_first_count
                    )

            num_batches += 1
            num_sequences += int(gt_actions.shape[0])
            num_valid_actions += int((~action_is_pad).sum().item())

    pred_jerk_mean = pred_jerk.mean
    gt_jerk_mean = gt_jerk.mean
    episode_metrics = [
        episode_accumulators[episode_index].to_metrics()
        for episode_index in sorted(episode_accumulators)
    ]
    (
        episode_p90_action_l1_mean,
        episode_p90_first_action_l1_mean,
        episode_p90_jerk_alignment,
        episode_max_action_l1_mean,
        episode_max_first_action_l1_mean,
    ) = summarize_episode_metrics(episode_metrics)
    return CheckpointMetrics(
        checkpoint=checkpoint_dir.parent.name,
        num_batches=num_batches,
        num_sequences=num_sequences,
        num_valid_actions=num_valid_actions,
        action_l1_mean=action_l1.mean,
        first_action_l1_mean=first_action_l1.mean,
        pred_jerk_l1_mean=pred_jerk_mean,
        gt_jerk_l1_mean=gt_jerk_mean,
        jerk_ratio_pred_over_gt=pred_jerk_mean / gt_jerk_mean if gt_jerk_mean and not math.isnan(gt_jerk_mean) else float("nan"),
        tactile_ablation_l1_mean=tactile_ablation_l1.mean,
        tactile_ablation_first_action_l1_mean=tactile_ablation_first.mean,
        tactile_ablation_relative_to_action_l1=(
            tactile_ablation_l1.mean / action_l1.mean if action_l1.count > 0 and action_l1.mean != 0 else float("nan")
        ),
        episode_p90_action_l1_mean=episode_p90_action_l1_mean,
        episode_p90_first_action_l1_mean=episode_p90_first_action_l1_mean,
        episode_p90_jerk_alignment=episode_p90_jerk_alignment,
        episode_max_action_l1_mean=episode_max_action_l1_mean,
        episode_max_first_action_l1_mean=episode_max_first_action_l1_mean,
        episode_metrics=episode_metrics,
    )


def print_summary(metrics: list[CheckpointMetrics]) -> None:
    headers = [
        "ckpt",
        "action_l1",
        "first_l1",
        "pred_jerk",
        "gt_jerk",
        "jerk_ratio",
        "ablation_l1",
        "ablation_first",
        "ablation/action",
    ]
    rows = [
        [
            metric.checkpoint,
            f"{metric.action_l1_mean:.6f}",
            f"{metric.first_action_l1_mean:.6f}",
            f"{metric.pred_jerk_l1_mean:.6f}",
            f"{metric.gt_jerk_l1_mean:.6f}",
            f"{metric.jerk_ratio_pred_over_gt:.4f}",
            f"{metric.tactile_ablation_l1_mean:.6f}",
            f"{metric.tactile_ablation_first_action_l1_mean:.6f}",
            f"{metric.tactile_ablation_relative_to_action_l1:.4f}",
        ]
        for metric in metrics
    ]

    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def format_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def safe_sort_value(value: float, reverse: bool = False) -> tuple[bool, float]:
    if math.isnan(value):
        return True, 0.0
    return False, -value if reverse else value


def jerk_alignment_score(jerk_ratio: float) -> float:
    if math.isnan(jerk_ratio) or jerk_ratio <= 0:
        return float("inf")
    return abs(math.log(jerk_ratio))


def percentile(values: list[float], q: float) -> float:
    clean_values = sorted(value for value in values if not math.isnan(value))
    if not clean_values:
        return float("nan")
    if len(clean_values) == 1:
        return clean_values[0]

    position = (len(clean_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean_values[lower]

    weight = position - lower
    return clean_values[lower] * (1 - weight) + clean_values[upper] * weight


def summarize_episode_metrics(
    episode_metrics: list[EpisodeMetrics],
) -> tuple[float, float, float, float, float]:
    action_values = [item.action_l1_mean for item in episode_metrics]
    first_action_values = [item.first_action_l1_mean for item in episode_metrics]
    jerk_alignment_values = [jerk_alignment_score(item.jerk_ratio_pred_over_gt) for item in episode_metrics]
    return (
        percentile(action_values, 0.9),
        percentile(first_action_values, 0.9),
        percentile(jerk_alignment_values, 0.9),
        max((value for value in action_values if not math.isnan(value)), default=float("nan")),
        max((value for value in first_action_values if not math.isnan(value)), default=float("nan")),
    )


def rank_checkpoints(metrics: list[CheckpointMetrics]) -> list[CheckpointRanking]:
    action_l1_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(metrics, key=lambda item: safe_sort_value(item.action_l1_mean)), start=1
        )
    }
    first_action_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(metrics, key=lambda item: safe_sort_value(item.first_action_l1_mean)), start=1
        )
    }
    jerk_alignment_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(metrics, key=lambda item: safe_sort_value(jerk_alignment_score(item.jerk_ratio_pred_over_gt))),
            start=1,
        )
    }
    episode_p90_action_l1_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(
                metrics,
                key=lambda item: safe_sort_value(item.episode_p90_action_l1_mean),
            ),
            start=1,
        )
    }
    episode_p90_first_action_l1_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(metrics, key=lambda item: safe_sort_value(item.episode_p90_first_action_l1_mean)),
            start=1,
        )
    }
    episode_p90_jerk_alignment_ranks = {
        metric.checkpoint: rank
        for rank, metric in enumerate(
            sorted(metrics, key=lambda item: safe_sort_value(item.episode_p90_jerk_alignment)),
            start=1,
        )
    }

    ranked = []
    for metric in metrics:
        composite_score = (
            DEPLOYMENT_RANKING_WEIGHTS["action_l1"] * action_l1_ranks[metric.checkpoint]
            + DEPLOYMENT_RANKING_WEIGHTS["first_action_l1"] * first_action_ranks[metric.checkpoint]
            + DEPLOYMENT_RANKING_WEIGHTS["jerk_alignment"] * jerk_alignment_ranks[metric.checkpoint]
            + DEPLOYMENT_RANKING_WEIGHTS["episode_p90_action_l1"]
            * episode_p90_action_l1_ranks[metric.checkpoint]
            + DEPLOYMENT_RANKING_WEIGHTS["episode_p90_first_action_l1"]
            * episode_p90_first_action_l1_ranks[metric.checkpoint]
            + DEPLOYMENT_RANKING_WEIGHTS["episode_p90_jerk_alignment"]
            * episode_p90_jerk_alignment_ranks[metric.checkpoint]
        )
        ranked.append(
            CheckpointRanking(
                checkpoint=metric.checkpoint,
                overall_rank=0,
                composite_score=composite_score,
                action_l1_rank=action_l1_ranks[metric.checkpoint],
                first_action_l1_rank=first_action_ranks[metric.checkpoint],
                jerk_alignment_rank=jerk_alignment_ranks[metric.checkpoint],
                episode_p90_action_l1_rank=episode_p90_action_l1_ranks[metric.checkpoint],
                episode_p90_first_action_l1_rank=episode_p90_first_action_l1_ranks[metric.checkpoint],
                episode_p90_jerk_alignment_rank=episode_p90_jerk_alignment_ranks[metric.checkpoint],
            )
        )

    ranked.sort(
        key=lambda item: (
            safe_sort_value(item.composite_score),
            safe_sort_value(next(metric.action_l1_mean for metric in metrics if metric.checkpoint == item.checkpoint)),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item.overall_rank = rank
    return ranked


def generate_markdown_report(
    metrics: list[CheckpointMetrics],
    rankings: list[CheckpointRanking],
    run_dir: Path,
    dataset_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
) -> str:
    lines = [
        "# Offline Checkpoint Comparison",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Device: `{device}`",
        f"- Batch size: `{batch_size}`",
        f"- Num workers: `{num_workers}`",
        f"- Max batches: `{max_batches}`",
        "",
        "This ranking is a deployment-oriented heuristic offline score.",
        "It prioritizes first-step accuracy, action smoothness, and episode-level tail behavior.",
        "Use it to narrow candidates before real robot evaluation.",
        "",
        "## Deployment-Oriented Ranking",
        "",
        "Weights:",
        f"- first_action_l1: `{DEPLOYMENT_RANKING_WEIGHTS['first_action_l1']}`",
        f"- jerk_alignment: `{DEPLOYMENT_RANKING_WEIGHTS['jerk_alignment']}`",
        f"- episode_p90_action_l1: `{DEPLOYMENT_RANKING_WEIGHTS['episode_p90_action_l1']}`",
        f"- episode_p90_first_action_l1: `{DEPLOYMENT_RANKING_WEIGHTS['episode_p90_first_action_l1']}`",
        f"- episode_p90_jerk_alignment: `{DEPLOYMENT_RANKING_WEIGHTS['episode_p90_jerk_alignment']}`",
        f"- action_l1: `{DEPLOYMENT_RANKING_WEIGHTS['action_l1']}`",
        "",
        "Notes:",
        "- `jerk_alignment` is `abs(log(pred_jerk / gt_jerk))`; lower is better.",
        "- `episode_p90_*` emphasizes deployment tail-risk instead of only average behavior.",
        "- `tactile_ablation` is still reported below as a diagnostic, but it no longer drives ranking.",
        "",
    ]

    ranking_rows = [
        [
            str(item.overall_rank),
            item.checkpoint,
            format_float(item.composite_score, digits=3),
            str(item.action_l1_rank),
            str(item.first_action_l1_rank),
            str(item.jerk_alignment_rank),
            str(item.episode_p90_action_l1_rank),
            str(item.episode_p90_first_action_l1_rank),
            str(item.episode_p90_jerk_alignment_rank),
        ]
        for item in rankings
    ]
    lines.append(
        make_markdown_table(
            [
                "rank",
                "checkpoint",
                "composite_score",
                "action_l1_rank",
                "first_action_l1_rank",
                "jerk_alignment_rank",
                "episode_p90_action_l1_rank",
                "episode_p90_first_action_l1_rank",
                "episode_p90_jerk_alignment_rank",
            ],
            ranking_rows,
        )
    )
    lines.extend(["", "## Aggregate Metrics", ""])

    aggregate_rows = [
        [
            metric.checkpoint,
            format_float(metric.action_l1_mean),
            format_float(metric.first_action_l1_mean),
            format_float(metric.pred_jerk_l1_mean),
            format_float(metric.gt_jerk_l1_mean),
            format_float(metric.jerk_ratio_pred_over_gt, digits=4),
            format_float(metric.episode_p90_action_l1_mean),
            format_float(metric.episode_p90_first_action_l1_mean),
            format_float(metric.episode_p90_jerk_alignment),
            format_float(metric.episode_max_action_l1_mean),
            format_float(metric.episode_max_first_action_l1_mean),
            format_float(metric.tactile_ablation_l1_mean),
            format_float(metric.tactile_ablation_first_action_l1_mean),
            format_float(metric.tactile_ablation_relative_to_action_l1, digits=4),
            str(metric.num_sequences),
        ]
        for metric in metrics
    ]
    lines.append(
        make_markdown_table(
            [
                "checkpoint",
                "action_l1",
                "first_l1",
                "pred_jerk",
                "gt_jerk",
                "jerk_ratio",
                "episode_p90_action_l1",
                "episode_p90_first_l1",
                "episode_p90_jerk_alignment",
                "episode_max_action_l1",
                "episode_max_first_l1",
                "ablation_l1",
                "ablation_first",
                "ablation/action",
                "num_sequences",
            ],
            aggregate_rows,
        )
    )

    for metric in metrics:
        lines.extend(["", f"## Checkpoint {metric.checkpoint}", ""])
        lines.append(
            make_markdown_table(
                [
                    "episode",
                    "num_sequences",
                    "num_valid_actions",
                    "action_l1",
                    "first_l1",
                    "pred_jerk",
                    "gt_jerk",
                    "jerk_ratio",
                    "ablation_l1",
                    "ablation/action",
                ],
                [
                    [
                        str(item.episode_index),
                        str(item.num_sequences),
                        str(item.num_valid_actions),
                        format_float(item.action_l1_mean),
                        format_float(item.first_action_l1_mean),
                        format_float(item.pred_jerk_l1_mean),
                        format_float(item.gt_jerk_l1_mean),
                        format_float(item.jerk_ratio_pred_over_gt, digits=4),
                        format_float(item.tactile_ablation_l1_mean),
                        format_float(item.tactile_ablation_relative_to_action_l1, digits=4),
                    ]
                    for item in metric.episode_metrics
                ],
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dirs = [args.run_dir / "checkpoints" / step / "pretrained_model" for step in args.steps]
    missing = [str(path) for path in checkpoint_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint directories: {missing}")

    dataloader, effective_batch_size, effective_num_workers = build_dataset_from_checkpoint(
        checkpoint_dirs[0],
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(
        "Evaluation dataloader: "
        f"batch_size={effective_batch_size}, num_workers={effective_num_workers}, "
        f"max_batches={args.max_batches if args.max_batches > 0 else 'full'}"
    )

    metrics = []
    for checkpoint_dir in checkpoint_dirs:
        print(f"Evaluating checkpoint {checkpoint_dir.parent.name} ...")
        metrics.append(
            evaluate_checkpoint(
                checkpoint_dir=checkpoint_dir,
                dataloader=dataloader,
                device=args.device,
                max_batches=args.max_batches,
                tactile_keys_override=args.tactile_keys,
            )
        )

    print_summary(metrics)
    rankings = rank_checkpoints(metrics)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_dir": str(args.run_dir),
            "dataset_root": str(args.dataset_root),
            "device": args.device,
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
            "max_batches": args.max_batches,
            "checkpoints": [asdict(metric) for metric in metrics],
            "rankings": [asdict(ranking) for ranking in rankings],
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved metrics to {args.output_json}")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        report = generate_markdown_report(
            metrics=metrics,
            rankings=rankings,
            run_dir=args.run_dir,
            dataset_root=args.dataset_root,
            device=args.device,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            max_batches=args.max_batches,
        )
        args.output_markdown.write_text(report)
        print(f"Saved Markdown report to {args.output_markdown}")


if __name__ == "__main__":
    main()
