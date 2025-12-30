#!/usr/bin/env python3
"""
Generate small-step experiment variants from a base config and start training.

Default is sandbox-friendly: we only print the planned commands unless --exec is provided.

Example:
  python utils/auto_loop/generate_and_run.py \
    --base-cfg src/lerobot/scripts/train_config/act_fr3_ot.json \
    --reason "pi low" \
    --changes 'ot.loss_config.reg@mul=1.6' 'ot.loss_config.tau_src@set=0.5' 'ot.loss_config.tau_tgt@set=0.5' \
    --steps 2000 --log-freq 50 --eval-freq 200 \
    --exec
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from .rules import apply_changes
    from .common import tiny_id
except Exception:  # script mode fallback
    from utils.auto_loop.rules import apply_changes  # type: ignore
    from utils.auto_loop.common import tiny_id  # type: ignore


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _dump_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def build_train_cmd(cfg_path: Path, steps: int, log_freq: int, eval_freq: int, extra_args: List[str]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={cfg_path}",
        f"--steps={steps}",
        f"--log_freq={log_freq}",
        f"--eval_freq={eval_freq}",
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--reason", default="variant")
    ap.add_argument("--changes", nargs="*", default=[], help="dotted edits with @op, e.g. a.b@mul=1.2")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--eval-freq", type=int, default=200)
    ap.add_argument("--out-dir", default="src/lerobot/scripts/train_config")
    ap.add_argument("--exec", action="store_true", help="actually launch training")
    ap.add_argument("--extra-args", nargs="*", default=[], help="passed through to training script")
    ap.add_argument("--inherit-dataset-from", default="", help="path to a cfg to copy dataset/ot.src_root/ot.pair_info_path from")
    args = ap.parse_args()

    base = Path(args.base_cfg)
    cfg = _load_json(base)

    # parse changes from CLI
    changes: Dict[str, Any] = {}
    for ch in args.changes:
        if "=" not in ch:
            continue
        key, sval = ch.split("=", 1)
        try:
            val: Any = float(sval)
        except Exception:
            val = sval
        changes[key] = val

    new_cfg = apply_changes(cfg, changes)

    # Inherit dataset roots from a baseline cfg if requested
    if args.inherit_dataset_from:
        try:
            base_ds = _load_json(Path(args.inherit_dataset_from))
            if "dataset" in base_ds and isinstance(base_ds["dataset"], dict):
                new_cfg.setdefault("dataset", {})
                for k in ["root", "repo_id", "revision", "streaming", "use_imagenet_stats", "video_backend"]:
                    if k in base_ds["dataset"]:
                        new_cfg["dataset"][k] = base_ds["dataset"][k]
            if "ot" in base_ds and isinstance(base_ds["ot"], dict):
                new_cfg.setdefault("ot", {})
                for k in ["src_root", "src_repo_id", "pair_info_path"]:
                    if k in base_ds["ot"]:
                        new_cfg["ot"][k] = base_ds["ot"][k]
        except Exception as e:
            print(f"[warn] failed to inherit dataset from {args.inherit_dataset_from}: {e}")
    tag = tiny_id(6)
    base_stem = base.stem
    new_name = f"{base_stem}_{tag}.json"
    out_path = Path(args.out_dir) / new_name
    _dump_json(out_path, new_cfg)
    print(f"Wrote variant: {out_path} ({args.reason})")

    cmd = build_train_cmd(out_path, steps=args.steps, log_freq=args.log_freq, eval_freq=args.eval_freq, extra_args=args.extra_args)
    print("Train cmd:", shlex.join(cmd))
    if args.__dict__.get("exec", False):
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
