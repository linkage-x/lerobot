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
"""Offline policy inference on a LeRobot dataset episode.

Unlike ``lerobot_eval.py``, this script does not create a robot or simulator environment. It loads a
trained policy checkpoint, takes observations step-by-step from a LeRobot dataset, predicts action chunks,
and plots the predicted chunks as a 3D matplotlib trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device, init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root to read observations from.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help=(
            "Training config file or directory. If a directory is passed, the script looks for "
            "train_config.generated.json or train_config.json. If omitted, it uses the checkpoint's "
            "pretrained_model/train_config.json when available."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint path. Accepts a pretrained_model directory, one checkpoint step directory, "
            "a checkpoints directory, or a checkpoints/last symlink."
        ),
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to evaluate.")
    parser.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="First episode-local step to evaluate. Defaults to the first step.",
    )
    parser.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="Last episode-local step to evaluate, inclusive. Defaults to the last step.",
    )
    parser.add_argument(
        "--trajectory-dims",
        default="auto",
        help=(
            "Comma-separated action dimensions used as XYZ. Use indices like '0,1,2', names like "
            "'ee.x,ee.y,ee.z', or 'auto' to infer from action feature names."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Repo id used when constructing LeRobotDataset. Defaults to the training config repo_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_offline"),
        help="Directory where the plot, npz, and metadata json are written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override, for example 'cuda', 'cuda:0', or 'cpu'. Defaults to checkpoint config.",
    )
    parser.add_argument("--use-amp", action="store_true", help="Use torch autocast during inference.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window after saving.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for the saved PNG figure.",
    )
    return parser.parse_args()


def resolve_pretrained_model_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    checkpoint = checkpoint.resolve()
    if (checkpoint / "config.json").is_file() and (checkpoint / "model.safetensors").is_file():
        return checkpoint

    pretrained = checkpoint / PRETRAINED_MODEL_DIR
    if (pretrained / "config.json").is_file() and (pretrained / "model.safetensors").is_file():
        return pretrained.resolve()

    if checkpoint.name == "checkpoints":
        last = checkpoint / "last"
        if last.exists():
            return resolve_pretrained_model_path(last)

        step_dirs = sorted(
            [path for path in checkpoint.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name),
        )
        if step_dirs:
            return resolve_pretrained_model_path(step_dirs[-1])

    raise FileNotFoundError(
        f"Could not resolve a pretrained_model directory from {checkpoint}. Expected config.json and "
        "model.safetensors under the path or under <checkpoint>/pretrained_model."
    )


def resolve_train_config_path(train_config: Path | None, pretrained_model_path: Path) -> Path:
    if train_config is None:
        candidate = pretrained_model_path / TRAIN_CONFIG_NAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"--train-config was not provided and {candidate} does not exist. Pass the training config path."
        )

    train_config = train_config.expanduser()
    if train_config.is_file():
        return train_config.resolve()

    if not train_config.is_dir():
        raise FileNotFoundError(f"Training config path does not exist: {train_config}")

    candidates = [
        train_config / "train_config.generated.json",
        train_config / TRAIN_CONFIG_NAME,
        train_config / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find train_config.generated.json or {TRAIN_CONFIG_NAME} under {train_config}"
    )


def load_train_config(path: Path) -> TrainPipelineConfig:
    with draccus.config_type("json"):
        return draccus.parse(TrainPipelineConfig, str(path), args=[])


def make_dataset_image_transforms(cfg: TrainPipelineConfig) -> ImageTransforms | None:
    if cfg.dataset.image_transforms.enable:
        return ImageTransforms(cfg.dataset.image_transforms)
    return None


def resolve_observation_delta_timestamps(
    policy_cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list[float]] | None:
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    if delta_timestamps is None:
        return None

    input_keys = set(policy_cfg.input_features or {})
    filtered = {key: value for key, value in delta_timestamps.items() if key in input_keys}
    return filtered or None


def load_policy(
    policy_cfg: PreTrainedConfig,
    pretrained_model_path: Path,
    ds_meta: LeRobotDatasetMetadata,
):
    policy_cls = get_policy_class(policy_cfg.type)
    return policy_cls.from_pretrained(
        pretrained_model_path,
        config=policy_cfg,
        dataset_stats=ds_meta.stats,
        dataset_meta=ds_meta,
    )


def batched_observation_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    batch = default_collate([sample])
    batch.pop(ACTION, None)
    batch.pop(f"{ACTION}_is_pad", None)
    return batch


def _queue_action_items(policy) -> list[torch.Tensor]:
    if hasattr(policy, "_action_queue"):
        return list(policy._action_queue)
    if hasattr(policy, "_queues") and ACTION in policy._queues:
        return list(policy._queues[ACTION])
    return []


def postprocess_action_chunk(
    action_chunk: torch.Tensor,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> torch.Tensor:
    if action_chunk.ndim == 2:
        action_chunk = action_chunk.unsqueeze(1)

    try:
        processed = postprocessor(action_chunk)
    except Exception:
        processed_steps = [postprocessor(action_chunk[:, step]) for step in range(action_chunk.shape[1])]
        processed = torch.stack(processed_steps, dim=1)

    if processed.ndim == 2:
        processed = processed.unsqueeze(1)
    if processed.ndim != 3:
        raise ValueError(f"Expected postprocessed action chunk shape (B, T, D), got {tuple(processed.shape)}")
    return processed


def predict_action_chunk(
    policy,
    preprocessed_batch: dict[str, Any],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> np.ndarray:
    policy.reset()
    first_error: Exception | None = None
    try:
        action_chunk = policy.predict_action_chunk(dict(preprocessed_batch))
        if action_chunk.ndim not in (2, 3):
            raise ValueError(f"Unexpected action chunk shape {tuple(action_chunk.shape)}")
    except Exception as exc:
        first_error = exc
        policy.reset()
        first_action = policy.select_action(dict(preprocessed_batch))
        if first_action.ndim == 1:
            first_action = first_action.unsqueeze(0)

        action_items = [first_action, *_queue_action_items(policy)]
        if len(action_items) == 0:
            raise RuntimeError("Policy did not return an action chunk or populate an action queue.") from first_error
        action_chunk = torch.stack(action_items, dim=1)

    action_chunk = postprocess_action_chunk(action_chunk, postprocessor)
    return action_chunk[0].detach().cpu().float().numpy()


def flatten_feature_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    elif isinstance(names, dict):
        flat = []

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, list):
                for item in value:
                    flat.append(f"{prefix}.{item}" if prefix else str(item))
            else:
                flat.append(f"{prefix}.{value}" if prefix else str(value))

        visit("", names)
        return flat
    else:
        return []


def resolve_action_names(ds_meta: LeRobotDatasetMetadata, train_cfg: TrainPipelineConfig, action_dim: int) -> list[str]:
    candidates = [flatten_feature_names(ds_meta.features.get(ACTION, {}).get("names"))]

    if train_cfg.dataset.root is not None:
        train_info_path = Path(train_cfg.dataset.root) / "meta" / "info.json"
        if train_info_path.is_file():
            with train_info_path.open() as f:
                train_info = json.load(f)
            candidates.append(flatten_feature_names(train_info.get("features", {}).get(ACTION, {}).get("names")))

    for names in candidates:
        if len(names) == action_dim:
            return names
    return [str(index) for index in range(action_dim)]


def parse_trajectory_dims(spec: str, action_names: list[str], action_dim: int) -> tuple[int, int, int]:
    if spec == "auto":
        lower_to_index = {name.lower(): idx for idx, name in enumerate(action_names)}
        candidates = [
            ("ee.x", "ee.y", "ee.z"),
            ("x", "y", "z"),
            ("position.x", "position.y", "position.z"),
            ("pos.x", "pos.y", "pos.z"),
        ]
        for candidate in candidates:
            if all(name in lower_to_index for name in candidate):
                return tuple(lower_to_index[name] for name in candidate)  # type: ignore[return-value]

        suffix_matches = []
        for suffix in (".x", ".y", ".z"):
            matches = [idx for idx, name in enumerate(action_names) if name.lower().endswith(suffix)]
            suffix_matches.append(matches[0] if matches else None)
        if all(index is not None for index in suffix_matches):
            return tuple(int(index) for index in suffix_matches)  # type: ignore[return-value]

        if action_dim >= 3:
            return 0, 1, 2
        raise ValueError(f"Cannot infer XYZ dims from action_dim={action_dim}")

    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if len(tokens) != 3:
        raise ValueError("--trajectory-dims must contain exactly three comma-separated values")

    lower_to_index = {name.lower(): idx for idx, name in enumerate(action_names)}
    dims = []
    for token in tokens:
        if token.isdigit():
            index = int(token)
        else:
            key = token.lower()
            if key not in lower_to_index:
                raise KeyError(f"Action dimension name {token!r} not found in {action_names}")
            index = lower_to_index[key]
        if index < 0 or index >= action_dim:
            raise IndexError(f"Action dimension index {index} out of range for action_dim={action_dim}")
        dims.append(index)
    return tuple(dims)  # type: ignore[return-value]


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_action_chunks(
    chunks: list[np.ndarray],
    step_ids: list[int],
    gt_points: np.ndarray | None,
    gt_step_ids: list[int],
    trajectory_dims: tuple[int, int, int],
    action_names: list[str],
    episode_index: int,
    episode_length: int,
    output_path: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save the offline action chunk plot. "
            "Install it with `pip install 'lerobot[matplotlib-dep]'` or install matplotlib directly."
        ) from exc

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    all_points = []

    if gt_points is not None and len(gt_points) > 0:
        all_points.append(gt_points)
        ax.plot(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            color="black",
            linestyle="--",
            linewidth=2.8,
            alpha=0.78,
            label="GT action trajectory",
        )
        gt_start = gt_points[0]
        gt_end = gt_points[-1]
        ax.scatter(gt_start[0], gt_start[1], gt_start[2], color="black", marker="s", s=44, depthshade=False)
        ax.scatter(gt_end[0], gt_end[1], gt_end[2], color="black", marker="D", s=44, depthshade=False)
        ax.text(gt_start[0], gt_start[1], gt_start[2], f"GT {gt_step_ids[0]}", color="black", fontsize=8)
        ax.text(gt_end[0], gt_end[1], gt_end[2], f"GT {gt_step_ids[-1]}", color="black", fontsize=8)

    for chunk_idx, (chunk, step_id) in enumerate(zip(chunks, step_ids, strict=True)):
        points = chunk[:, trajectory_dims]
        all_points.append(points)
        color = cmap(chunk_idx % cmap.N)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.8, alpha=0.9)

        start = points[0]
        end = points[-1]
        tail_step = min(step_id + len(points) - 1, episode_length - 1)
        ax.scatter(start[0], start[1], start[2], color=color, marker="o", s=34, depthshade=False)
        ax.scatter(end[0], end[1], end[2], color=color, marker="X", s=54, depthshade=False)
        ax.text(start[0], start[1], start[2], f"{step_id}", color=color, fontsize=8)
        ax.text(end[0], end[1], end[2], f"{tail_step}", color=color, fontsize=8)

    points = np.concatenate(all_points, axis=0)
    set_axes_equal(ax, points)
    x_name, y_name, z_name = [action_names[index] for index in trajectory_dims]
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title(
        f"Offline action chunks | episode={episode_index} | steps={step_ids[0]}..{step_ids[-1]}"
    )
    ax.grid(True)
    if gt_points is not None and len(gt_points) > 0:
        ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def write_interactive_action_chunks_html(
    chunks: list[np.ndarray],
    step_ids: list[int],
    gt_points: np.ndarray | None,
    gt_step_ids: list[int],
    trajectory_dims: tuple[int, int, int],
    action_names: list[str],
    episode_index: int,
    episode_length: int,
    output_path: Path,
) -> None:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    series = []
    all_points = []
    gt_payload = None
    if gt_points is not None and len(gt_points) > 0:
        all_points.append(gt_points.astype(float))
        gt_payload = {
            "start_step": int(gt_step_ids[0]),
            "end_step": int(gt_step_ids[-1]),
            "color": "#111111",
            "points": np.round(gt_points.astype(float), 6).tolist(),
        }

    for chunk_idx, (chunk, step_id) in enumerate(zip(chunks, step_ids, strict=True)):
        points = chunk[:, trajectory_dims].astype(float)
        all_points.append(points)
        series.append(
            {
                "step": int(step_id),
                "tail_step": int(min(step_id + len(points) - 1, episode_length - 1)),
                "color": palette[chunk_idx % len(palette)],
                "points": np.round(points, 6).tolist(),
            }
        )

    points = np.concatenate(all_points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = ((mins + maxs) / 2.0).tolist()
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0.0:
        radius = 1.0

    payload = {
        "title": f"Offline action chunks | episode={episode_index} | steps={step_ids[0]}..{step_ids[-1]}",
        "axisLabels": [action_names[index] for index in trajectory_dims],
        "center": center,
        "radius": radius,
        "gt": gt_payload,
        "series": series,
    }

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Offline action chunks</title>
  <style>
    :root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #16181d; }
    .bar {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 12px 16px; border-bottom: 1px solid #d9dde5; background: white;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    button {
      border: 1px solid #c5cad4; background: #fff; border-radius: 6px; padding: 6px 10px;
      cursor: pointer; font-size: 13px;
    }
    button:hover { background: #f0f2f5; }
    .hint { color: #5c6370; font-size: 13px; }
    .wrap { height: calc(100vh - 58px); min-height: 520px; position: relative; }
    canvas { width: 100%; height: 100%; display: block; background: #ffffff; }
    .legend {
      position: absolute; right: 14px; top: 14px; padding: 8px 10px; background: rgba(255,255,255,0.88);
      border: 1px solid #d9dde5; border-radius: 6px; font-size: 12px; color: #333;
    }
  </style>
</head>
<body>
  <div class="bar">
    <h1 id="title"></h1>
    <button id="reset">Reset view</button>
    <span class="hint">Drag to rotate. Wheel to zoom. Shift+drag to pan.</span>
  </div>
  <div class="wrap">
    <canvas id="view"></canvas>
    <div class="legend"><b>GT</b>: black dashed line<br><b>Predicted chunks</b>: colored lines<br>circle: chunk start<br>X: chunk end<br>text: episode step id</div>
  </div>
  <script>
    const DATA = __DATA__;
    document.getElementById("title").textContent = DATA.title;

    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    let theta = -0.75;
    let phi = 0.55;
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let shiftDrag = false;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function rotatePoint(point) {
      const x0 = point[0] - DATA.center[0];
      const y0 = point[1] - DATA.center[1];
      const z0 = point[2] - DATA.center[2];
      const cy = Math.cos(theta), sy = Math.sin(theta);
      const cp = Math.cos(phi), sp = Math.sin(phi);
      const x1 = cy * x0 + sy * z0;
      const z1 = -sy * x0 + cy * z0;
      const y1 = cp * y0 - sp * z1;
      const z2 = sp * y0 + cp * z1;
      return [x1, y1, z2];
    }

    function project(point) {
      const rect = canvas.getBoundingClientRect();
      const rotated = rotatePoint(point);
      const scale = 0.42 * Math.min(rect.width, rect.height) / DATA.radius * zoom;
      return {
        x: rect.width / 2 + panX + rotated[0] * scale,
        y: rect.height / 2 + panY - rotated[1] * scale,
        z: rotated[2],
      };
    }

    function drawLine(points, color, dashed = false, width = 2, alpha = 0.92) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const p = project(points[i]);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = alpha;
      if (dashed) ctx.setLineDash([10, 7]);
      ctx.stroke();
      ctx.restore();
    }

    function drawSquare(point, color, text) {
      const p = project(point);
      ctx.fillStyle = color;
      ctx.fillRect(p.x - 5, p.y - 5, 10, 10);
      ctx.fillText(String(text), p.x + 8, p.y - 8);
    }

    function drawDiamond(point, color, text) {
      const p = project(point);
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = color;
      ctx.fillRect(-5, -5, 10, 10);
      ctx.restore();
      ctx.fillText(String(text), p.x + 8, p.y + 5);
    }

    function drawCircle(point, color, text) {
      const p = project(point);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.fillText(String(text), p.x + 7, p.y - 7);
    }

    function drawX(point, color, text) {
      const p = project(point);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(p.x - 6, p.y - 6);
      ctx.lineTo(p.x + 6, p.y + 6);
      ctx.moveTo(p.x + 6, p.y - 6);
      ctx.lineTo(p.x - 6, p.y + 6);
      ctx.stroke();
      ctx.fillText(String(text), p.x + 8, p.y + 4);
    }

    function drawAxes() {
      const labels = DATA.axisLabels;
      const r = DATA.radius;
      const c = DATA.center;
      const axes = [
        [[c[0] - r, c[1], c[2]], [c[0] + r, c[1], c[2]], labels[0]],
        [[c[0], c[1] - r, c[2]], [c[0], c[1] + r, c[2]], labels[1]],
        [[c[0], c[1], c[2] - r], [c[0], c[1], c[2] + r], labels[2]],
      ];
      ctx.save();
      ctx.strokeStyle = "#aeb4bf";
      ctx.fillStyle = "#4b5563";
      ctx.lineWidth = 1;
      ctx.font = "13px Arial";
      for (const axis of axes) {
        const a = project(axis[0]);
        const b = project(axis[1]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.fillText(axis[2], b.x + 6, b.y - 6);
      }
      ctx.restore();
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.font = "12px Arial";
      ctx.textBaseline = "middle";
      drawAxes();
      if (DATA.gt) drawLine(DATA.gt.points, DATA.gt.color, true, 3, 0.78);
      const ordered = DATA.series.slice().sort((a, b) => {
        const za = a.points.reduce((sum, p) => sum + rotatePoint(p)[2], 0) / a.points.length;
        const zb = b.points.reduce((sum, p) => sum + rotatePoint(p)[2], 0) / b.points.length;
        return za - zb;
      });
      for (const chunk of ordered) drawLine(chunk.points, chunk.color);
      if (DATA.gt) {
        ctx.fillStyle = DATA.gt.color;
        drawSquare(DATA.gt.points[0], DATA.gt.color, "GT " + DATA.gt.start_step);
        drawDiamond(DATA.gt.points[DATA.gt.points.length - 1], DATA.gt.color, "GT " + DATA.gt.end_step);
      }
      for (const chunk of DATA.series) {
        ctx.fillStyle = chunk.color;
        drawCircle(chunk.points[0], chunk.color, chunk.step);
        drawX(chunk.points[chunk.points.length - 1], chunk.color, chunk.tail_step);
      }
    }

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      shiftDrag = event.shiftKey;
      lastX = event.clientX;
      lastY = event.clientY;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (shiftDrag) {
        panX += dx;
        panY += dy;
      } else {
        theta += dx * 0.01;
        phi = Math.max(-1.45, Math.min(1.45, phi + dy * 0.01));
      }
      draw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoom *= Math.exp(-event.deltaY * 0.001);
      zoom = Math.max(0.15, Math.min(20, zoom));
      draw();
    }, { passive: false });
    document.getElementById("reset").addEventListener("click", () => {
      theta = -0.75;
      phi = 0.55;
      zoom = 1.0;
      panX = 0;
      panY = 0;
      draw();
    });
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html.replace("__DATA__", json.dumps(payload)), encoding="utf-8")


def main() -> None:
    args = parse_args()
    init_logging()
    register_third_party_plugins()
    set_seed(args.seed)

    pretrained_model_path = resolve_pretrained_model_path(args.checkpoint)
    train_config_path = resolve_train_config_path(args.train_config, pretrained_model_path)
    train_cfg = load_train_config(train_config_path)

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_model_path)
    if args.device is None:
        device = get_safe_torch_device(policy_cfg.device, log=True)
    elif args.device == "auto":
        device = auto_select_torch_device()
    else:
        device = get_safe_torch_device(args.device, log=True)
    policy_cfg.device = str(device)
    policy_cfg.use_amp = bool(args.use_amp or policy_cfg.use_amp)
    policy_cfg.pretrained_path = pretrained_model_path

    dataset_root = args.dataset_root.expanduser().resolve()
    repo_id = args.dataset_repo_id or train_cfg.dataset.repo_id or dataset_root.name
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root, revision=train_cfg.dataset.revision)
    if args.episode_index < 0 or args.episode_index >= ds_meta.total_episodes:
        raise IndexError(
            f"Episode index {args.episode_index} out of range. Dataset has {ds_meta.total_episodes} episodes."
        )

    delta_timestamps = resolve_observation_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=[args.episode_index],
        delta_timestamps=delta_timestamps,
        image_transforms=make_dataset_image_transforms(train_cfg),
        revision=train_cfg.dataset.revision,
        video_backend=train_cfg.dataset.video_backend,
        tolerance_s=train_cfg.tolerance_s,
    )

    episode_meta = ds_meta.episodes[args.episode_index]
    episode_length = int(episode_meta["dataset_to_index"] - episode_meta["dataset_from_index"])
    step_start = 0 if args.step_start is None else args.step_start
    step_end = episode_length - 1 if args.step_end is None else args.step_end
    if step_start < 0 or step_end < step_start or step_end >= episode_length:
        raise ValueError(
            f"Invalid step range [{step_start}, {step_end}] for episode length {episode_length}. "
            "step_end is inclusive."
        )
    step_ids = list(range(step_start, step_end + 1))

    action_dim = int(policy_cfg.output_features[ACTION].shape[0])
    action_names = resolve_action_names(ds_meta, train_cfg, action_dim)
    trajectory_dims = parse_trajectory_dims(args.trajectory_dims, action_names, action_dim)

    logging.info("Loading policy from %s", pretrained_model_path)
    policy = load_policy(policy_cfg, pretrained_model_path, ds_meta)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": str(device)}}
    rename_map = getattr(train_cfg, "rename_map", None)
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_model_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    chunks: list[np.ndarray] = []
    context = torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext()
    with torch.no_grad(), context:
        for step_id in tqdm(step_ids, desc="Offline inference"):
            sample = dataset[step_id]
            batch = batched_observation_from_sample(sample)
            preprocessed_batch = preprocessor(batch)
            chunks.append(predict_action_chunk(policy, preprocessed_batch, postprocessor))

    gt_tail_step = min(step_end + max(chunk.shape[0] for chunk in chunks) - 1, episode_length - 1)
    episode_actions = dataset.get_episode_column_arrays(args.episode_index, [ACTION])[ACTION]
    if episode_actions.ndim != 2:
        episode_actions = episode_actions.reshape(episode_actions.shape[0], -1)
    max_trajectory_dim = max(trajectory_dims)
    if episode_actions.shape[1] <= max_trajectory_dim:
        raise ValueError(
            f"Dataset action dim {episode_actions.shape[1]} cannot provide trajectory dims {trajectory_dims}."
        )
    gt_step_ids = list(range(step_start, gt_tail_step + 1))
    gt_points = episode_actions[step_start : gt_tail_step + 1, trajectory_dims].astype(np.float32)

    run_name = f"{dataset_root.name}_ep{args.episode_index:04d}_steps{step_start:04d}-{step_end:04d}"
    output_dir = args.output_dir / run_name
    plot_path = output_dir / "action_chunks_3d.png"
    html_path = output_dir / "action_chunks_3d_interactive.html"
    npz_path = output_dir / "action_chunks.npz"
    metadata_path = output_dir / "metadata.json"

    plot_action_chunks(
        chunks,
        step_ids,
        gt_points,
        gt_step_ids,
        trajectory_dims,
        action_names,
        args.episode_index,
        episode_length,
        plot_path,
        dpi=args.dpi,
        show=args.show,
    )
    write_interactive_action_chunks_html(
        chunks,
        step_ids,
        gt_points,
        gt_step_ids,
        trajectory_dims,
        action_names,
        args.episode_index,
        episode_length,
        html_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        chunks=np.asarray(chunks, dtype=np.float32),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        gt_points=gt_points,
        gt_step_ids=np.asarray(gt_step_ids, dtype=np.int64),
        trajectory_dims=np.asarray(trajectory_dims, dtype=np.int64),
    )
    metadata = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "train_config": str(train_config_path),
        "checkpoint": str(pretrained_model_path),
        "episode_index": args.episode_index,
        "episode_length": episode_length,
        "step_start": step_start,
        "step_end": step_end,
        "gt_step_start": gt_step_ids[0],
        "gt_step_end": gt_step_ids[-1],
        "trajectory_dims": list(trajectory_dims),
        "trajectory_dim_names": [action_names[index] for index in trajectory_dims],
        "action_names": action_names,
        "plot_path": str(plot_path),
        "html_path": str(html_path),
        "npz_path": str(npz_path),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved 3D action chunk plot to: {plot_path}")
    print(f"Saved interactive 3D action chunk viewer to: {html_path}")
    print(f"Saved predicted chunks to: {npz_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
