#!/usr/bin/env python3
"""
Update ACT-OT experiment report from Weights & Biases runs.

Usage examples:
  # Append three runs to the default report file
  python utils/ot_report_update.py \
    --project kjust-pinduoduo/lerobot \
    --runs RUN_ID_1 RUN_ID_2 RUN_ID_3 \
    --title "Batch update"

Environment:
  - WANDB_API_KEY should be set for private projects.

The script appends a structured section to:
  src/lerobot/scripts/train_config/reports/act_ot.md
creating the directory if needed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append ACT-OT W&B runs to experiment report.")
    p.add_argument("--project", default="kjust-pinduoduo/lerobot", help="W&B project path (entity/project)")
    p.add_argument("--runs", nargs="+", help="One or more W&B run IDs")
    p.add_argument(
        "--file",
        default="src/lerobot/scripts/train_config/reports/act_ot.md",
        help="Report file to append to (will be created if missing)",
    )
    p.add_argument("--title", default=None, help="Optional section title to include in the report")
    return p.parse_args()


def _login_wandb() -> None:
    import wandb  # lazy import

    # Do not leak any API key; rely on env var if needed.
    wandb.login(anonymous="never")


def _collect_runs(project: str, run_ids: List[str]) -> Dict[str, Any]:
    import wandb  # lazy import

    api = wandb.Api(timeout=60)

    PRIMARY = [
        "train/loss",
        "train/l1_loss",
        "train/kld_loss",
        "eval/offline_eval/avg_l1",
        "eval/offline_eval/avg_loss",
    ]
    OTHERS = ["train/grad_norm", "train/lr", "train/update_s", "train/dataloading_s"]
    OT_MAP = {
        "train/ot_loss": ["train/ot_loss"],
        "train/ot_pi_sum": ["train/ot_pi_sum", "train/ot_ot_pi_sum"],
        "train/ot_pi_diag": ["train/ot_pi_diag", "train/ot_ot_pi_diag"],
        "train/ot_cost/observation.state": [
            "train/ot_cost/observation.state",
            "train/ot_ot_cost/observation.state",
        ],
    }

    def get_series(row: Dict[str, Any], keys: List[str]) -> float:
        for k in keys:
            v = row.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return v
        return float("nan")

    def summarize(run_path: str) -> Dict[str, Any]:
        run = api.run(run_path)
        rows = [row for row in run.scan_history(page_size=10000)]
        steps = [row.get("_step", i) for i, row in enumerate(rows)]

        # Gather metrics
        metrics: Dict[str, List[float]] = {}
        for k in PRIMARY + OTHERS:
            seq: List[float] = []
            for row in rows:
                v = row.get(k, float("nan"))
                seq.append(v if isinstance(v, (int, float)) and math.isfinite(v) else float("nan"))
            metrics[k] = seq
        for k, ks in OT_MAP.items():
            seq = [get_series(row, ks) for row in rows]
            metrics[k] = seq

        def first(vals: List[float]) -> float:
            for x in vals:
                if isinstance(x, (int, float)) and math.isfinite(x):
                    return x
            return float("nan")

        def last(vals: List[float]) -> float:
            for x in reversed(vals):
                if isinstance(x, (int, float)) and math.isfinite(x):
                    return x
            return float("nan")

        def bestmin(vals: List[float]):
            fs = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float)) and math.isfinite(v)]
            return (None, float("nan")) if not fs else min(fs, key=lambda t: t[1])

        summ: Dict[str, Dict[str, Any]] = {}
        for k, vals in metrics.items():
            f = first(vals)
            l = last(vals)
            bi, bv = bestmin(vals)
            s: Dict[str, Any] = {
                "first": f,
                "last": l,
                "best": bv,
                "best_step": (steps[bi] if bi is not None else None),
            }
            if math.isfinite(f) and math.isfinite(l) and f != 0:
                s["delta"] = l - f
                s["pct"] = 100.0 * (l - f) / (f + 1e-12)
            summ[k] = s

        # Extract OT params from config (best effort)
        try:
            cfg = dict(run.config)
        except Exception:
            cfg = {}

        def get_from_config(d: Dict[str, Any], path: List[str], default=None):
            cur: Any = d
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return default
            return cur

        ot_params = {
            "window_size": get_from_config(cfg, ["ot", "window_size"]),
            "lambda_ot": get_from_config(cfg, ["ot", "lambda_ot"]),
            "reg": get_from_config(cfg, ["ot", "loss_config", "reg"]),
            "tau_src": get_from_config(cfg, ["ot", "loss_config", "tau_src"]),
            "tau_tgt": get_from_config(cfg, ["ot", "loss_config", "tau_tgt"]),
            "heuristic": get_from_config(cfg, ["ot", "loss_config", "heuristic"]),
            "weight_label": None,
            "weight_embed": None,
        }
        feats = get_from_config(cfg, ["ot", "loss_config", "features"], []) or []
        if isinstance(feats, list) and feats:
            f0 = feats[0]
            if isinstance(f0, dict):
                ot_params["weight_label"] = f0.get("weight_label")
                ot_params["weight_embed"] = f0.get("weight_embed")

        return {
            "name": run.name,
            "url": run.url,
            "summary": summ,
            "ot_params": ot_params,
        }

    return {rid: summarize(f"{project}/{rid}") for rid in run_ids}


def _append_report(report_file: Path, runs_info: Dict[str, Any], title: str | None) -> None:
    def fmt_line(s: Dict[str, Any], key: str, label: str) -> str:
        x = s.get(key, {})
        f = x.get("first", float("nan"))
        l = x.get("last", float("nan"))
        b = x.get("best", float("nan"))
        bs = x.get("best_step")
        return f"  - {label}: first={f:.4g}, last={l:.4g}, best={b:.4g}@{bs}"

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines.append("\n\n---\n")
    lines.append(f"## ACT-OT Report Update ({now})")
    if title:
        lines.append(f"- Note: {title}")
    for rid, info in runs_info.items():
        s = info["summary"]
        p = info["ot_params"]
        lines.append("")
        lines.append(f"### {info['name']} ({rid})")
        lines.append(f"- Link: {info['url']}")
        lines.append("- Config:")
        lines.append(
            f"  - window_size={p['window_size']}, reg={p['reg']}, tau=({p['tau_src']},{p['tau_tgt']}), heuristic={p['heuristic']}"
        )
        lines.append(
            f"  - weight_embed={p['weight_embed']}, weight_label={p['weight_label']}, lambda_ot={p['lambda_ot']}"
        )
        lines.append("- Metrics:")
        lines.append(fmt_line(s, "train/loss", "train/loss"))
        lines.append(fmt_line(s, "train/l1_loss", "train/l1_loss"))
        lines.append(fmt_line(s, "eval/offline_eval/avg_l1", "eval/avg_l1"))
        lines.append(fmt_line(s, "train/ot_loss", "ot_loss"))
        lines.append(fmt_line(s, "train/ot_pi_sum", "ot_pi_sum"))
        lines.append(fmt_line(s, "train/ot_pi_diag", "ot_pi_diag"))
        lines.append(fmt_line(s, "train/ot_cost/observation.state", "ot_cost(state)"))

    report_file.parent.mkdir(parents=True, exist_ok=True)
    if report_file.exists():
        existing = report_file.read_text(encoding="utf-8")
    else:
        existing = "# ACT-OT Experiment Report\n"
    report_file.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _login_wandb()
    runs_info = _collect_runs(args.project, args.runs)
    _append_report(Path(args.file), runs_info, args.title)


if __name__ == "__main__":
    main()

