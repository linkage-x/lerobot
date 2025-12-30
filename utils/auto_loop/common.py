import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Minimal helper to parse wandb offline logs or local JSON summaries.


@dataclass
class MetricSummary:
    first: Optional[float]
    last: Optional[float]
    best: Optional[float]
    best_step: Optional[int]
    delta: Optional[float]
    pct: Optional[float]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def summarize_series(values: List[Tuple[int, float]], mode: str = "min") -> MetricSummary:
    # values: list of (step, scalar), sorted by step
    if not values:
        return MetricSummary(None, None, None, None, None, None)
    first = values[0][1]
    last = values[-1][1]
    if mode == "min":
        best_step, best_val = min(values, key=lambda kv: kv[1])
    else:
        best_step, best_val = max(values, key=lambda kv: kv[1])
    delta = None
    pct = None
    if first is not None and last is not None:
        delta = last - first
        try:
            pct = 0.0 if first == 0 else (delta / first) * 100.0
        except Exception:
            pct = None
    return MetricSummary(first, last, best_val, best_step, delta, pct)


def find_wandb_file(run_dir: Path, filename: str) -> Optional[Path]:
    # Common offline structure: <run_dir>/wandb/latest-run/files/<filename>
    latest = run_dir / "wandb" / "latest-run" / "files" / filename
    if latest.exists():
        return latest
    # Fallback: search within wandb dir
    for p in (run_dir / "wandb").rglob(filename):
        return p
    return None


def load_history_from_run_dir(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Extract step->metric series from a W&B run directory.

    Preferred source: wandb-history.jsonl (offline/local file stream).
    Fallback: wandb-summary.json (final summary only) when history.jsonl is missing
              e.g. with newer W&B versions or pure online mode.

    Returns a dict: metric_name -> list[(step, value)], step-sorted.
    """
    series: Dict[str, List[Tuple[int, float]]] = {}

    # 1) Try the line-delimited history first
    history_path = find_wandb_file(run_dir, "wandb-history.jsonl")
    if history_path is not None and history_path.exists():
        with open(history_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                step = rec.get("_step")
                if step is None:
                    # support custom step keys like "train/Optimization step"
                    step = rec.get("train/Optimization step", rec.get("Optimization step"))
                if step is None:
                    continue
                try:
                    step = int(step)
                except Exception:
                    continue
                for k, v in rec.items():
                    if k.startswith("_"):
                        continue
                    val = _safe_float(v)
                    if val is None:
                        continue
                    lst = series.setdefault(k, [])
                    lst.append((step, val))
        # sort by step
        for k in list(series.keys()):
            series[k] = sorted(series[k], key=lambda kv: kv[0])
        if series:
            return series

    # 2) Fallback to final summary if no history.jsonl is present
    summary_path = find_wandb_file(run_dir, "wandb-summary.json")
    if summary_path is not None and summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summ = json.load(f)
        except Exception:
            summ = None
        if isinstance(summ, dict):
            # Use a single pseudo-step (1) for all numeric keys
            for k, v in summ.items():
                val = _safe_float(v)
                if val is None:
                    continue
                series.setdefault(k, []).append((1, float(val)))
    return series


def scan_offline_runs(outputs_root: Path) -> List[Path]:
    """Find candidate run directories that contain a wandb folder.

    Default training output dir pattern in this repo is outputs/train/YYYY-MM-DD/HH-MM-SS_*/
    """
    if not outputs_root.exists():
        return []
    candidates: List[Path] = []
    for p in outputs_root.rglob("wandb"):
        run_dir = p.parent
        candidates.append(run_dir)
    # de-duplicate and sort by mtime desc
    uniq = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq


# Canonical metric aliases to be robust to naming variants across experiments.
# We will report the first alias that exists in the history.
METRIC_ALIASES: dict[str, list[str]] = {
    # Core losses
    "train/loss": ["train/loss"],
    "train/l1_loss": ["train/l1_loss"],
    "train/kld_loss": ["train/kld_loss"],
    # OT loss
    "train/ot_loss": ["train/ot_loss", "train/ot_ot_loss", "train/ot"],
    # OT mass/diagonal
    "train/ot_pi_sum": ["train/ot_pi_sum", "train/ot_ot_pi_sum"],
    "train/ot_pi_diag": ["train/ot_pi_diag", "train/ot_ot_pi_diag"],
    # OT costs (per-term)
    "train/ot_cost/img_third_person": [
        "train/ot_cost/img_third_person",
        "train/ot_ot_cost/observation.images.third_person_cam_color",
    ],
    "train/ot_cost/img_side": [
        "train/ot_cost/img_side",
        "train/ot_ot_cost/observation.images.side_cam_color",
    ],
    "train/ot_cost/img_ee": [
        "train/ot_cost/img_ee",
        "train/ot_ot_cost/observation.images.ee_cam_color",
    ],
    "train/ot_cost/action_lbl": [
        "train/ot_cost/action_lbl",
        "train/ot_ot_cost/action",
        "train/ot_ot_cost/observation.state",  # legacy state proxy
    ],
    # Optim/logging
    "train/grad_norm": ["train/grad_norm"],
    "train/lr": ["train/lr"],
    # Eval (wandb logger prefixes eval/)
    "eval/offline_eval/avg_loss": ["eval/offline_eval/avg_loss"],
    "eval/offline_eval/avg_l1": ["eval/offline_eval/avg_l1"],
}


def summarize_series_dict(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        vals: list[tuple[int, float]] = []
        for alias in aliases:
            if alias in series and series[alias]:
                vals = series[alias]
                break
        # Determine mode
        mode = "max" if canonical.endswith("pi_sum") or canonical.endswith("pi_diag") or canonical.endswith("lr") else "min"
        ms = summarize_series(vals, mode=mode)
        summary[canonical] = ms.__dict__
    # basic step count
    steps = 0
    for vals in series.values():
        if vals:
            steps = max(steps, vals[-1][0])
    summary["steps"] = steps
    return summary


def load_compare_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def tiny_id(n: int) -> str:
    import random
    import string

    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))
