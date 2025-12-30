#!/usr/bin/env python3
"""
Analyze local (offline) wandb runs under outputs/train/* and emit compact JSON summaries.

Usage:
  python utils/auto_loop/analyze_offline.py \
      --outputs-root outputs/train \
      --out-json src/lerobot/scripts/train_config/reports/data/offline_summary.json

This does not require WANDB_API_KEY nor network access, and helps dry-run the loop.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .common import load_history_from_run_dir, scan_offline_runs, summarize_series_dict, save_json
except Exception:  # script mode fallback
    from utils.auto_loop.common import load_history_from_run_dir, scan_offline_runs, summarize_series_dict, save_json  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs/train")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out-json", default="src/lerobot/scripts/train_config/reports/data/offline_summary.json")
    args = ap.parse_args()

    root = Path(args.outputs_root)
    runs = scan_offline_runs(root)[: args.limit]
    results: Dict[str, Dict[str, Any]] = {}
    for run_dir in runs:
        series = load_history_from_run_dir(run_dir)
        if not series:
            continue
        results[str(run_dir)] = summarize_series_dict(series)

    out_path = Path(args.out_json)
    save_json(out_path, results)
    print(f"Wrote {out_path} with {len(results)} runs")


if __name__ == "__main__":
    main()
