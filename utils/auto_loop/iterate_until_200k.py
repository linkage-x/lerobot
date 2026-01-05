#!/usr/bin/env python3
"""
Iterative experiment controller: run small rounds until a 200k-worthy config emerges.

Policy:
- Start from provided cfgs (4 recommended). Each round runs 10k steps with online W&B.
- After each round, analyze latest loop outputs and check per-run criteria:
    * 0.05 <= train/ot_pi_sum (last) <= 0.30
    * train/ot_loss (last) > 0 (non-tiny)
    * eval/offline_eval/avg_l1 improved by >=10% vs first (pct <= -10)
    * train/grad_norm (last) < 60
  If any run satisfies all, stop and print the winning cfg + latest loop run dir.
- Otherwise, generate next-round cfgs via rules.decide (variants-per-run=1) with safety clamps:
    * Clamp ot.lambda_ot >= 0.02
    * Keep integer fields as ints (apply_changes already does this)
  and launch the next round.

Usage:
  python -m utils.auto_loop.iterate_until_200k \
    --start-cfgs <cfg1> <cfg2> <cfg3> <cfg4> \
    --gpus 0,1,2,3 --entity <wandb_entity> --project <wandb_project>

The script runs synchronously and exits when a winner is found (printing a summary),
or after --max-rounds.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .analyze_offline import scan_offline_runs, load_history_from_run_dir
from .common import summarize_series_dict
from .rules import decide


def _latest_loop_runs(outputs_root: Path) -> Tuple[str, List[Path]]:
    runs = scan_offline_runs(outputs_root)
    by_loop: Dict[str, List[Path]] = {}
    for r in runs:
        m = re.search(r"outputs/train/(loop_\d+)/", str(r))
        if m:
            by_loop.setdefault(m.group(1), []).append(r)
    if not by_loop:
        return "", []
    latest = sorted(by_loop)[-1]
    return latest, sorted(by_loop[latest])


def _worthy(summary: Dict[str, Any], pi_min: float, pi_max: float, eval_improve_req: float) -> bool:
    # Extract last values safely
    def g(path: List[str], key: str) -> float | None:
        cur: Any = summary
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        v = cur.get(key) if isinstance(cur, dict) else None
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    pi_sum_last = g(["train/ot_pi_sum"], "last")
    ot_loss_last = g(["train/ot_loss"], "last")
    eval_pct = g(["eval/offline_eval/avg_l1"], "pct")
    grad_last = g(["train/grad_norm"], "last")

    if pi_sum_last is None or ot_loss_last is None or eval_pct is None or grad_last is None:
        return False
    if not (pi_min <= pi_sum_last <= pi_max):
        return False
    if not (ot_loss_last > 1e-9):
        return False
    if not (eval_pct <= eval_improve_req):
        return False
    if not (grad_last < 60.0):
        return False
    return True


def _materialize_variant(base_cfg: Path, reason: str, changes: Dict[str, Any]) -> Path | None:
    cli = [f"{k}={v}" for k, v in changes.items()]
    cmd = [
        sys.executable,
        "-m",
        "utils.auto_loop.generate_and_run",
        "--base-cfg",
        str(base_cfg),
        "--reason",
        reason,
        "--changes",
        *cli,
        "--steps",
        "10000",
        "--log-freq",
        "100",
        "--eval-freq",
        "500",
        "--out-dir",
        "src/lerobot/scripts/train_config",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    path: Path | None = None
    for ln in out.stdout.splitlines():
        if ln.startswith("Wrote variant: "):
            target = ln.split("Wrote variant: ")[1].split(" ")[0].strip()
            path = Path(target)
            break
    # Clamp lambda_ot if present
    if path and path.exists():
        try:
            data = json.loads(path.read_text())
            lam = float(data.get("ot", {}).get("lambda_ot", 0.1))
            if lam < 0.02:
                data["ot"]["lambda_ot"] = 0.02
                path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-cfgs", nargs="+", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--eval-freq", type=int, default=500)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--pi-min", type=float, default=0.05)
    ap.add_argument("--pi-max", type=float, default=0.30)
    ap.add_argument("--eval-improve-pct", type=float, default=-10.0)  # <= -10% vs first
    args = ap.parse_args()

    outputs_root = Path("outputs/train")
    current_cfgs = [Path(p) for p in args.start_cfgs]

    for r in range(1, args.max_rounds + 1):
        # Launch one round synchronously
        cmd = [
            sys.executable,
            "utils/auto_loop/autorun.py",
            "--cfgs",
            *[str(p) for p in current_cfgs],
            "--steps",
            str(args.steps),
            "--log-freq",
            str(args.log_freq),
            "--eval-freq",
            str(args.eval_freq),
            "--concurrency",
            str(min(len(current_cfgs), 4)),
            "--gpus",
            args.gpus,
            "--rounds",
            "1",
            "--exec",
            "--wandb-online",
            "--wandb-entity",
            args.entity,
            "--wandb-project",
            args.project,
        ]
        print(f"[iter] Round {r}: launch {len(current_cfgs)} cfgs -> {args.steps} steps")
        subprocess.run(cmd, check=False)

        # Analyze latest loop
        loop_id, run_dirs = _latest_loop_runs(outputs_root)
        print(f"[iter] Latest loop {loop_id} with {len(run_dirs)} runs")
        # Check winners
        winners: List[Tuple[Path, Dict[str, Any]]] = []
        for rd in run_dirs:
            series = load_history_from_run_dir(rd)
            if not series:
                continue
            summ = summarize_series_dict(series)
            if _worthy(summ, args.pi_min, args.pi_max, args.eval_improve_pct):
                winners.append((rd, summ))
        if winners:
            print("[iter] Found 200k-worthy candidate(s):")
            for rd, summ in winners:
                print(" - run:", rd)
                print("   pi_sum_last=", summ.get("train/ot_pi_sum", {}).get("last"))
                print("   eval_l1_last=", summ.get("eval/offline_eval/avg_l1", {}).get("last"),
                      "pct=", summ.get("eval/offline_eval/avg_l1", {}).get("pct"))
            return

        # Otherwise, generate next-round variants via rules (variants-per-run=1)
        next_cfgs: List[Path] = []
        for base in current_cfgs:
            # Try to get the decision from any corresponding run (best-effort);
            # if not available, fall back to default gentle.
            # Here we just call decide on the aggregate of the latest loop's runs,
            # then apply the first decision for uniformity.
            # More sophisticated mapping can be added if needed.
            any_summ: Dict[str, Any] | None = None
            for rd in run_dirs:
                series = load_history_from_run_dir(rd)
                if series:
                    any_summ = summarize_series_dict(series)
                    break
            prop = decide(any_summ or {}) if any_summ else []
            if not prop:
                prop = []
            changes = prop[0].changes if prop else {"ot.loss_config.reg@mul": 1.2, "ot.window_size@max": 20}
            newp = _materialize_variant(base, prop[0].reason if prop else "gentle", changes)
            if newp:
                next_cfgs.append(newp)
        if not next_cfgs:
            print("[iter] No next cfgs generated; stopping.")
            return
        current_cfgs = next_cfgs


if __name__ == "__main__":
    main()

