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

"""Convenience CLI for manually removing low-quality episodes from a LeRobot dataset.

This script intentionally delegates the dataset rewrite to ``lerobot-edit-dataset``'s
delete_episodes path, which uses the public LeRobot dataset tools API.
"""

import argparse
import ast
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_edit_dataset import (
    DeleteEpisodesConfig,
    EditDatasetConfig,
    handle_delete_episodes,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def parse_episode_indices(values: list[str]) -> list[int]:
    """Parse episode indices from CLI tokens.

    Accepts all of these forms:
      --episodes 1 2 3
      --episodes 1,2,3
      --episodes "[1, 2, 3]"
    """
    parsed: list[int] = []

    for value in values:
        text = value.strip()
        if not text:
            continue

        if text.startswith("[") and text.endswith("]"):
            items = ast.literal_eval(text)
            if not isinstance(items, (list, tuple)):
                raise ValueError("--episodes list syntax must contain a list or tuple of integers")
            parsed.extend(_coerce_episode_index(item) for item in items)
            continue

        for item in text.replace(",", " ").split():
            parsed.append(_coerce_episode_index(item))

    unique: list[int] = []
    seen: set[int] = set()
    for episode_index in parsed:
        if episode_index < 0:
            raise ValueError(f"Episode indices must be >= 0, got {episode_index}")
        if episode_index not in seen:
            unique.append(episode_index)
            seen.add(episode_index)

    if not unique:
        raise ValueError("At least one episode index must be provided")

    return unique


def _coerce_episode_index(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid episode index: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("-"):
            digits = text[1:]
        else:
            digits = text
        if digits.isdigit():
            return int(text)
    raise ValueError(f"Invalid episode index: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete manually selected bad episodes from a LeRobot dataset.",
    )
    parser.add_argument("--repo-id", required=True, help="Input dataset repo_id, e.g. user/dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Exact local dataset root containing meta/, data/, videos/. "
            "Defaults to $HF_LEROBOT_HOME/repo_id."
        ),
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help=(
            "Episode indices to delete. Accepted forms: "
            "1 4 7, 1,4,7, or \"[1, 4, 7]\"."
        ),
    )
    parser.add_argument(
        "--new-repo-id",
        default=None,
        help="Optional output repo_id. If omitted, the input repo_id is reused.",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=None,
        help="Optional output dataset root. If omitted, follows lerobot-edit-dataset path semantics.",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push the cleaned dataset after deletion.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without writing files.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    return parser.parse_args(argv)


def build_edit_config(args: argparse.Namespace, episode_indices: list[int]) -> EditDatasetConfig:
    return EditDatasetConfig(
        repo_id=args.repo_id,
        root=str(args.root) if args.root is not None else None,
        new_repo_id=args.new_repo_id,
        new_root=str(args.new_root) if args.new_root is not None else None,
        push_to_hub=args.push_to_hub,
        operation=DeleteEpisodesConfig(episode_indices=episode_indices),
    )


def _default_output_root(
    repo_id: str,
    root: Path | None,
    new_repo_id: str | None,
    new_root: Path | None,
) -> Path:
    if new_root is not None:
        return new_root
    if new_repo_id is not None:
        return HF_LEROBOT_HOME / new_repo_id
    if root is not None:
        return root
    return HF_LEROBOT_HOME / repo_id


def _print_plan(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    episode_indices: list[int],
    output_root: Path,
) -> None:
    total_episodes = dataset.meta.total_episodes
    total_frames = dataset.meta.total_frames
    removed_frames = sum(dataset.meta.episodes[index]["length"] for index in episode_indices)
    remaining_episodes = total_episodes - len(episode_indices)
    remaining_frames = total_frames - removed_frames

    print(f"Dataset: {dataset.repo_id}")
    print(f"Root: {dataset.root}")
    print(f"Delete episodes: {episode_indices}")
    print(f"Episodes: {total_episodes} -> {remaining_episodes}")
    print(f"Frames: {total_frames} -> {remaining_frames}")
    print(f"Output repo_id: {args.new_repo_id or args.repo_id}")
    print(f"Output root: {output_root}")
    if output_root == dataset.root:
        backup_root = dataset.root.with_name(dataset.root.name + "_old")
        print(f"In-place edit: original dataset will be moved to {backup_root}")


def _validate_episode_indices(dataset: LeRobotDataset, episode_indices: list[int]) -> None:
    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = sorted(set(episode_indices) - valid_indices)
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")
    if len(episode_indices) == dataset.meta.total_episodes:
        raise ValueError("Cannot delete all episodes from dataset")


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = parse_args(argv)
    episode_indices = parse_episode_indices(args.episodes)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    _validate_episode_indices(dataset, episode_indices)
    output_root = _default_output_root(args.repo_id, args.root, args.new_repo_id, args.new_root)
    _print_plan(dataset, args, episode_indices, output_root)

    if args.dry_run:
        print("Dry run: no files were changed.")
        return 0

    if not args.yes:
        response = input("Delete these episodes using LeRobot dataset_tools.delete_episodes? [y/N] ")
        if response.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    cfg = build_edit_config(args, episode_indices)
    handle_delete_episodes(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
