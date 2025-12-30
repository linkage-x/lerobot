#!/usr/bin/env python3
"""
Online W&B analysis (requires network + WANDB_API_KEY).

Example:
  python utils/auto_loop/analyze_wandb.py \
    --entity kjust-pinduoduo --project lerobot \
    --group-substr act --max-runs 20 \
    --out-json src/lerobot/scripts/train_config/reports/data/online_summary.json

Note: Will gracefully exit with code 0 if wandb import/API fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

try:
    from .common import save_json
    from .rules import decide
except Exception:  # script mode fallback
    from utils.auto_loop.common import save_json  # type: ignore
    from utils.auto_loop.rules import decide  # type: ignore


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def summarize_history(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    # rows are dicts with keys including _step and metrics
    series: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        st = r.get("_step")
        if st is None:
            continue
        try:
            st = int(st)
        except Exception:
            continue
        for k in keys:
            if k not in r:
                continue
            v = _to_float(r[k])
            if v is None:
                continue
            series.setdefault(k, []).append((st, v))
    # sort
    for k in list(series.keys()):
        series[k] = sorted(series[k], key=lambda kv: kv[0])

    def summarize(vals: List[Tuple[int, float]], mode: str = "min") -> Dict[str, Any]:
        if not vals:
            return {"first": None, "last": None, "best": None, "best_step": None, "delta": None, "pct": None}
        first = vals[0][1]
        last = vals[-1][1]
        if mode == "max":
            best_step, best_val = max(vals, key=lambda kv: kv[1])
        else:
            best_step, best_val = min(vals, key=lambda kv: kv[1])
        delta = last - first
        pct = 0.0 if first == 0 else (delta / first) * 100.0
        return {
            "first": first,
            "last": last,
            "best": best_val,
            "best_step": best_step,
            "delta": delta,
            "pct": pct,
        }

    summary: Dict[str, Any] = {}
    for k in keys:
        mode = "max" if k.endswith("pi_sum") or k.endswith("pi_diag") or k.endswith("lr") else "min"
        summary[k] = summarize(series.get(k, []), mode)
    # steps
    steps = 0
    for _, vals in series.items():
        if vals:
            steps = max(steps, vals[-1][0])
    summary["steps"] = steps
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--group-substr", default="")
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--max-runs", type=int, default=50)
    ap.add_argument("--out-json", default="src/lerobot/scripts/train_config/reports/data/online_summary.json")
    ap.add_argument("--history-limit", type=int, default=2500, help="fetch this many rows per run")
    args = ap.parse_args()

    try:
        import wandb  # noqa: F401
        from wandb.apis.public import Api
    except Exception:
        print("wandb API not available; skipping.")
        sys.exit(0)

    api = Api()
    proj = f"{args.entity}/{args.project}"
    runs = api.runs(proj)
    out: Dict[str, Any] = {}
    keys = [
        "train/loss",
        "train/l1_loss",
        "train/kld_loss",
        "train/ot_loss",
        "train/ot_ot_loss",
        "train/ot_ot_pi_sum",
        "train/ot_ot_pi_diag",
        "train/ot_ot_cost/observation.state",
        "train/grad_norm",
        "train/lr",
        "eval/offline_eval/avg_loss",
        "eval/offline_eval/avg_l1",
    ]

    sel: List[Any] = []
    for r in runs:
        if args.group_substr and args.group_substr not in (r.group or ""):
            continue
        if args.tag and not set(args.tag).issubset(set(r.tags or [])):
            continue
        sel.append(r)
        if len(sel) >= args.max_runs:
            break

    for r in sel:
        rows = list(r.history(samples=args.history_limit))
        summary = summarize_history(rows, keys)
        out[r.id] = summary

    # Optional: propose next actions via rules for each run
    proposals: Dict[str, Any] = {}
    for k, summ in out.items():
        decisions = [d.__dict__ for d in decide(summ)]
        proposals[k] = decisions
    out_obj = {"summary": out, "proposals": proposals}
    save_json(args.out_json, out_obj)
    print(f"Wrote {args.out_json} with {len(out)} runs")


if __name__ == "__main__":
    main()
