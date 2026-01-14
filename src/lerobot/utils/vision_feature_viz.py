#!/usr/bin/env python
"""Visualize and compare backbone feature maps across multiple configs.

Examples:
  python src/lerobot/utils/vision_feature_viz.py \
    --config_paths \
      src/lerobot/scripts/train_config/experiments/test_backbone/act_rn18.json \
      src/lerobot/scripts/train_config/experiments/test_backbone/act_rn50.json \
      src/lerobot/scripts/train_config/experiments/test_backbone/act_rn101.json \
    --sample_index 0 --output_dir /tmp/vision_viz

  python src/lerobot/utils/vision_feature_viz.py --image_path /path/to/image.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _parse_config(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("dataset", {}), cfg.get("policy", {})


def _resolve_weights(weight_name: str | None):
    if not weight_name or weight_name == "None":
        return None
    if "." in weight_name:
        class_name, member = weight_name.split(".", 1)
        weight_cls = getattr(torchvision.models, class_name, None)
        if weight_cls is None:
            raise ValueError(f"Unknown weights class: {class_name}")
        return getattr(weight_cls, member)
    weight_cls = getattr(torchvision.models, weight_name, None)
    if weight_cls is None:
        raise ValueError(f"Unknown weights: {weight_name}")
    return weight_cls.DEFAULT


def _build_backbone(backbone: str, weights_name: str | None, replace_final_stride: bool):
    weights = _resolve_weights(weights_name)
    model = getattr(torchvision.models, backbone)(
        replace_stride_with_dilation=[False, False, bool(replace_final_stride)],
        weights=weights,
        norm_layer=FrozenBatchNorm2d,
    )
    return IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"})


def _normalize_for_backbone(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.float32:
        image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def _select_image_from_dataset(
    dataset_cfg: dict,
    sample_index: int,
    camera_key: str | None,
    apply_image_transforms: bool,
) -> tuple[torch.Tensor, str]:
    repo_id = dataset_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("dataset.repo_id is required in the config when no --image_path is provided.")
    root = dataset_cfg.get("root")
    episodes = dataset_cfg.get("episodes")
    revision = dataset_cfg.get("revision")
    video_backend = dataset_cfg.get("video_backend")

    image_transforms = None
    if apply_image_transforms and dataset_cfg.get("image_transforms", {}).get("enable", False):
        image_transforms = ImageTransforms(_build_image_transforms_config(dataset_cfg))

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=None,
        image_transforms=image_transforms,
        revision=revision,
        video_backend=video_backend,
    )

    if camera_key is None:
        if not dataset.meta.camera_keys:
            raise ValueError("No camera keys found in dataset metadata.")
        camera_key = dataset.meta.camera_keys[0]

    item = dataset[sample_index]
    if camera_key not in item:
        raise ValueError(f"Camera key '{camera_key}' not found in dataset item. Available: {list(item.keys())}")

    image = item[camera_key]
    if image.ndim == 4:
        image = image[-1]
    return image, camera_key


def _load_image_from_path(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    return image


def _build_image_transforms_config(dataset_cfg: dict) -> ImageTransformsConfig:
    cfg_dict = dict(dataset_cfg.get("image_transforms", {}))
    tfs = cfg_dict.get("tfs")
    if tfs:
        parsed_tfs = {}
        for name, tf_cfg in tfs.items():
            if isinstance(tf_cfg, dict):
                parsed_tfs[name] = ImageTransformConfig(**tf_cfg)
            else:
                parsed_tfs[name] = tf_cfg
        cfg_dict["tfs"] = parsed_tfs
    return ImageTransformsConfig(**cfg_dict)


def _normalize_map(feature_map: torch.Tensor) -> np.ndarray:
    fmap = feature_map.detach().cpu().float()
    fmin = float(fmap.min())
    fmax = float(fmap.max())
    if fmax - fmin < 1e-6:
        return np.zeros_like(fmap.numpy())
    return ((fmap - fmin) / (fmax - fmin)).numpy()


def _save_channel_grid(
    feature_map: torch.Tensor,
    indices: list[int],
    out_path: Path,
    cols: int = 4,
    title: str | None = None,
):
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(indices):
            continue
        ch = indices[i]
        ax.imshow(_normalize_map(feature_map[ch]), cmap="viridis")
        ax.set_title(f"ch {ch}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backbone feature maps across configs.")
    parser.add_argument(
        "--config_paths",
        nargs="+",
        default=[
            "src/lerobot/scripts/train_config/experiments/test_backbone/act_rn18.json",
            "src/lerobot/scripts/train_config/experiments/test_backbone/act_rn50.json",
            "src/lerobot/scripts/train_config/experiments/test_backbone/act_rn101.json",
        ],
    )
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--camera_key", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="vision_feature_viz")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--apply_image_transforms", action="store_true")
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.config_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels
    if labels is None:
        labels = [p.stem for p in config_paths]
    if len(labels) != len(config_paths):
        raise ValueError("--labels must match the number of --config_paths.")

    dataset_cfg, _ = _parse_config(config_paths[0])

    if args.image_path:
        image = _load_image_from_path(Path(args.image_path))
        image_label = Path(args.image_path).name
    else:
        image, camera_key = _select_image_from_dataset(
            dataset_cfg,
            args.sample_index,
            args.camera_key,
            args.apply_image_transforms,
        )
        image_label = f"dataset_idx_{args.sample_index}_{camera_key}"

    image = image.detach().cpu()
    image_np = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(image_np, 0.0, 1.0))
    plt.title(f"input: {image_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "input.png", dpi=150)
    plt.close()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    mean_maps = []
    max_maps = []

    for label, cfg_path in zip(labels, config_paths):
        _, policy_cfg = _parse_config(cfg_path)
        backbone = policy_cfg.get("vision_backbone", "resnet18")
        weights_name = policy_cfg.get("pretrained_backbone_weights")
        replace_stride = policy_cfg.get("replace_final_stride_with_dilation", False)

        backbone_model = _build_backbone(backbone, weights_name, replace_stride)
        backbone_model.to(device).eval()

        image_norm = _normalize_for_backbone(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = backbone_model(image_norm)["feature_map"].squeeze(0)

        mean_maps.append(_normalize_map(features.mean(dim=0)))
        max_maps.append(_normalize_map(features.max(dim=0).values))

        channel_var = torch.var(features, dim=(1, 2))
        topk_idx = torch.argsort(channel_var, descending=True)[: args.topk].tolist()
        _save_channel_grid(
            features,
            topk_idx,
            output_dir / f"{label}_top{args.topk}.png",
            cols=4,
            title=f"{label} top-{args.topk} channels",
        )

    fig, axes = plt.subplots(2, len(labels), figsize=(4 * len(labels), 8), squeeze=False)
    for col, label in enumerate(labels):
        axes[0, col].imshow(mean_maps[col], cmap="viridis")
        axes[0, col].set_title(f"{label} mean")
        axes[0, col].axis("off")
        axes[1, col].imshow(max_maps[col], cmap="viridis")
        axes[1, col].set_title(f"{label} max")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_maps_summary.png", dpi=150)
    plt.close(fig)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
