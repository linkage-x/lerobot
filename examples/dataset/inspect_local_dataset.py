#!/usr/bin/env python
"""
Quickly inspect a local LeRobot dataset without touching the hub.

Examples:
  python examples/dataset/inspect_local_dataset.py --repo-id left_fr3_ip_216ep_ft2ee --root /data/fr3_lerobot
  python examples/dataset/inspect_local_dataset.py --repo-id my_ds --root /path/to/cache --sample-episodes 5
"""

import argparse
from pprint import pprint

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot dataset.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo_id (folder name under root).")
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory that contains the dataset folder. If omitted, uses the default cache location.",
    )
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=3,
        help="Load this many episodes to probe shapes. Set <=0 to skip loading sample data.",
    )
    parser.add_argument(
        "--list-lengths",
        action="store_true",
        help="Print per-episode frame counts and basic stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Inspecting repo_id='{args.repo_id}' root='{args.root}'")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    print("\nMetadata")
    print("--------")
    print(f"total_episodes: {meta.total_episodes}")
    print(f"total_frames:   {meta.total_frames}")
    print(f"fps:            {meta.fps}")
    print(f"robot_type:     {meta.robot_type}")
    print(f"camera_keys:    {meta.camera_keys}")
    print("features:")
    pprint(meta.features)

    if args.list_lengths:
        eps = meta.episodes
        if {"dataset_from_index", "dataset_to_index"} <= set(eps.column_names):
            starts = eps["dataset_from_index"]
            ends = eps["dataset_to_index"]
            lengths = [int(b) - int(a) for a, b in zip(starts, ends)]
            total = sum(lengths)
            print("\nEpisode lengths (frames):")
            print(lengths)
            print(
                f"count={len(lengths)}, min={min(lengths)}, max={max(lengths)}, "
                f"mean={total/len(lengths):.2f}"
            )
        else:
            print("\nEpisode lengths unavailable (dataset_from_index/to_index not found in metadata).")

    if args.sample_episodes <= 0:
        return

    n_eps = args.sample_episodes
    if meta.total_episodes is not None:
        n_eps = min(n_eps, meta.total_episodes)

    print(f"\nLoading dataset with first {n_eps} episode(s) to probe shapes...")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(n_eps)))
    print(f"num_episodes: {ds.num_episodes}")
    print(f"num_frames:   {ds.num_frames}")
    print(f"keys:         {list(ds.features.keys())}")

    if len(ds) == 0:
        print("Dataset is empty; skipping sample fetch.")
        return

    first_item = ds[0]
    print("\nSample item shapes:")
    for k, v in first_item.items():
        try:
            shape = tuple(v.shape)
        except Exception:
            shape = "N/A"
        print(f"  {k}: {shape}")


if __name__ == "__main__":
    main()
