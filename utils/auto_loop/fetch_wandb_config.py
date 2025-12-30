#!/usr/bin/env python3
"""
Fetch run.config from W&B and save as a local training JSON.

Usage:
  python utils/auto_loop/fetch_wandb_config.py \
    --entity kjust-pinduoduo --project lerobot \
    --run iztuqxzc --run xx5st9ij --run ithx0mon \
    --out-dir src/lerobot/scripts/train_config

Notes:
  - Requires network + WANDB_API_KEY
  - We log a faithful copy of the nested config that training expects.
  - You can still override steps/log_freq/eval_freq at runtime via CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def unflatten(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        cur = out
        parts = k.split(sep)
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                cur[p] = v
            else:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]  # type: ignore[assignment]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--out-dir", default="src/lerobot/scripts/train_config")
    args = ap.parse_args()

    try:
        from wandb.apis.public import Api
    except Exception as e:
        raise SystemExit("wandb API not available. Ensure WANDB_API_KEY and network access.") from e

    api = Api()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rid in args.run:
        run_path = f"{args.entity}/{args.project}/{rid}"
        run = api.run(run_path)
        cfg_obj = run.config
        # cfg_obj is a Config object (mapping-like). Convert to plain dict.
        cfg_flat: Dict[str, Any] = {}
        for k in cfg_obj.keys():
            # skip internal wandb keys (start with _) if any
            if k.startswith("_"):
                continue
            cfg_flat[k] = cfg_obj[k]
        # Many wandb setups store nested dicts but also flatten with dots.
        # If there are dots, build nested dict, but preserve existing nested dicts.
        if any("." in k for k in cfg_flat.keys()):
            nested = unflatten(cfg_flat)
        else:
            nested = cfg_flat

        # Save
        out_path = out_dir / f"fromwandb_{rid}.json"
        with open(out_path, "w") as f:
            json.dump(nested, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

