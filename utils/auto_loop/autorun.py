#!/usr/bin/env python3
"""
Lightweight experiment loop runner.

Features
- Queue multiple training configs, run each for a small number of steps, with optional concurrency.
- Force safe defaults: offline wandb, local output_dir under outputs/train, configurable device.
- After each run finishes, parse offline wandb history and write/update a summary JSON.

Example
  python utils/auto_loop/autorun.py \
    --cfgs \
      src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015.json \
      src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json \
    --steps 2000 --concurrency 2 --gpus 0,1 --exec

Note: Without --exec we only print the plan.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .common import load_history_from_run_dir, summarize_series_dict, save_json, tiny_id
    from .rules import decide, apply_changes
except Exception:  # script mode fallback for `python utils/auto_loop/autorun.py`
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root (so `utils` becomes importable)
    from utils.auto_loop.common import load_history_from_run_dir, summarize_series_dict, save_json, tiny_id  # type: ignore
    from utils.auto_loop.rules import decide, apply_changes  # type: ignore


DEFAULT_CFGLIST = [
    # curated in exp_plan.md
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015.json",
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json",
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_nowindow_tau2_reg015_lambda015_action001.json",
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_w6_sharp1_lambda016_action001.json",
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005.json",
    "src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20.json",
]


@dataclass
class Job:
    cfg_path: Path
    out_dir: Path
    device: str
    gpu_id: Optional[str]
    steps: int
    log_freq: int
    eval_freq: int
    extra: List[str]


def build_cmd(job: Job) -> List[str]:
    # Use accelerate launch + script path per user guidance
    args = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "src/lerobot/scripts/lerobot_train.py",
        f"--config_path={job.cfg_path}",
        f"--steps={job.steps}",
        f"--log_freq={job.log_freq}",
        f"--eval_freq={job.eval_freq}",
        f"--output_dir={job.out_dir}",
        f"--policy.device={job.device}",
    ]
    # Default to offline wandb unless caller specified a mode explicitly
    has_mode = any(str(x).startswith("--wandb.mode=") for x in job.extra)
    if not has_mode:
        args.extend(["--wandb.mode=offline", "--wandb.enable=true", "--wandb.disable_artifact=true"])
    args.extend(job.extra)
    return args


def run_queue(jobs: List[Job], concurrency: int, gpus: List[str], dry_run: bool, summary_out: Path) -> Dict[str, Dict[str, Any]]:
    proc_slots: Dict[int, subprocess.Popen] = {}
    job_slots: Dict[int, Job] = {}
    pending = queue.Queue()
    for jb in jobs:
        pending.put(jb)

    def _start(slot: int, jb: Job) -> None:
        env = os.environ.copy()
        if jb.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = jb.gpu_id
        # Route wandb caches to a workspace-local writable dir to avoid permission issues under ~/.cache
        cache_root = Path(".wandb_cache").resolve()
        (cache_root / "logs").mkdir(parents=True, exist_ok=True)
        (cache_root / "config").mkdir(parents=True, exist_ok=True)
        env["WANDB_CACHE_DIR"] = str(cache_root)
        env["WANDB_CONFIG_DIR"] = str(cache_root / "config")
        # The training code sets wandb.init(dir=output_dir), but set WANDB_DIR as a fallback
        env["WANDB_DIR"] = str(jb.out_dir)
        # Avoid sentry DNS noise if the environment restricts it
        env["WANDB_SENTRY_ENABLED"] = "false"
        cmd = build_cmd(jb)
        print(f"[slot {slot}] {shlex.join(cmd)}")
        if dry_run:
            return
        # Do NOT pre-create jb.out_dir; training validates that it must not exist when resume=False
        # Log to a temp file outside output_dir to avoid the pre-create.
        tmp_logs = Path(".codex_tmp"); tmp_logs.mkdir(parents=True, exist_ok=True)
        log_path = tmp_logs / f"slot{slot}_{jb.cfg_path.stem}.log"
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        proc_slots[slot] = proc
        job_slots[slot] = jb

    # Fill initial slots
    for s in range(concurrency):
        if pending.empty():
            break
        jb = pending.get()
        _start(s, jb)

    # Monitor
    per_run_summary: Dict[str, Dict[str, Any]] = {}
    while proc_slots:
        time.sleep(5)
        finished: List[int] = []
        for s, p in list(proc_slots.items()):
            ret = p.poll()
            if ret is not None:
                finished.append(s)
        # Close finished and start new ones
        for s in finished:
            proc_slots.pop(s, None)
            jb = job_slots.pop(s, None)
            if jb is not None:
                # Try to summarize offline metrics
                series = load_history_from_run_dir(jb.out_dir)
                if series:
                    summary = summarize_series_dict(series)
                    per_run_summary[str(jb.out_dir)] = summary
                    root = summary_out
                    root.parent.mkdir(parents=True, exist_ok=True)
                    existing: Dict[str, Any] = {}
                    if root.exists():
                        try:
                            with open(root, "r") as f:
                                existing = json.load(f)
                        except Exception:
                            existing = {}
                    existing[str(jb.out_dir)] = summary
                    save_json(root, existing)
                    print(f"[slot {s}] summarized -> {root}")
        for s in range(concurrency):
            if s in proc_slots:
                continue
            if pending.empty():
                continue
            _start(s, pending.get())
    
    return per_run_summary

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--eval-freq", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids; if empty, run on CPU")
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra CLI args to training script (repeatable)")
    # Convenience flags to pass wandb online settings without using --extra
    ap.add_argument("--wandb-online", action="store_true", help="force wandb online mode")
    ap.add_argument("--wandb-entity", default="", help="wandb entity when online")
    ap.add_argument("--wandb-project", default="", help="wandb project when online")
    ap.add_argument("--inherit-dataset-from", default="", help="baseline cfg path whose dataset/ot roots will be inherited by spawned jobs")
    ap.add_argument("--output-root", default="outputs/train")
    ap.add_argument("--summary-out", default="src/lerobot/scripts/train_config/reports/data/autorun_summary.json")
    ap.add_argument("--rounds", type=int, default=1, help="number of iterative rounds")
    ap.add_argument("--variants-per-run", type=int, default=2, help="max variants generated per completed run when deciding")
    ap.add_argument("--decide", action="store_true", help="after each round, propose variants via rules and schedule next round")
    ap.add_argument("--markdown-out", default="src/lerobot/scripts/train_config/reports/act_ot.md", help="append per-round markdown summary here")
    args = ap.parse_args()

    cfgs = [Path(p) for p in (args.cfgs or DEFAULT_CFGLIST)]
    device = "cuda" if args.gpus else "cpu"
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.gpus else []
    out_root = Path(args.output_root) / f"loop_{int(time.time())}"
    jid = tiny_id(4)

    jobs: List[Job] = []
    for i, cfg in enumerate(cfgs):
        name = cfg.stem
        sub = out_root / f"{i:02d}_{name}_{jid}"
        gpu_id = gpus[i % len(gpus)] if gpus else None
        extra = list(args.extra)
        if args.wandb_online:
            extra.extend([
                "--wandb.mode=online",
                f"--wandb.entity={args.wandb_entity}" if args.wandb_entity else "",
                f"--wandb.project={args.wandb_project}" if args.wandb_project else "",
            ])
            extra = [x for x in extra if x]
        jobs.append(
            Job(
                cfg_path=cfg,
                out_dir=sub,
                device=device,
                gpu_id=gpu_id,
                steps=args.steps,
                log_freq=args.log_freq,
                eval_freq=args.eval_freq,
                extra=(extra + (["--inherit-dataset-from", args.inherit_dataset_from] if args.inherit_dataset_from else [])),
            )
        )

    dry = not args.exec
    if dry:
        print("[DRY-RUN] Planned jobs:")
        for j in jobs:
            gpu = j.gpu_id if j.gpu_id is not None else "cpu"
            print(f"  - {j.cfg_path} -> {j.out_dir} on {gpu} for {j.steps} steps")
    completed_summaries: Dict[str, Dict[str, Any]] = {}
    current_jobs = jobs

    def append_round_markdown(rnum: int, jobs: List[Job]) -> None:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append(f"\n\n### Auto Loop Round {rnum}\n")
        for jb in jobs:
            out = jb.out_dir
            # Try wandb latest-run files
            meta = out / "wandb/latest-run/files/wandb-metadata.json"
            summ = out / "wandb/latest-run/files/wandb-summary.json"
            url = None
            # Fallback: parse slot log
            slot_log = Path(f".codex_tmp/slot0_{jb.cfg_path.stem}.log")
            # Try to discover any slot log matching this stem
            for sl in Path('.codex_tmp').glob(f"slot*_{jb.cfg_path.stem}.log"):
                try:
                    txt = sl.read_text(errors='ignore')
                    for ln in txt.splitlines():
                        if 'Track this run' in ln and 'https://' in ln:
                            url = ln.split('-->')[-1].strip()
                            break
                except Exception:
                    pass
                if url:
                    break
            metrics = {}
            try:
                import json as _json
                if summ.exists():
                    metrics = _json.loads(summ.read_text())
            except Exception:
                metrics = {}
            def gv(k: str):
                v = metrics.get(k)
                return f"{v:.4f}" if isinstance(v, (int, float)) else (str(v) if v is not None else "-")
            lines.append(f"- cfg: `{jb.cfg_path}` | out: `{out}`")
            if url:
                lines.append(f"  - W&B: {url}")
            lines.append(
                "  - eval_l1: {} | train_l1: {} | ot_pi_sum: {} | ot_loss: {} | grad_norm: {}".format(
                    gv('eval/offline_eval/avg_l1'), gv('train/l1_loss'), gv('train/ot_pi_sum'), gv('train/ot_loss'), gv('train/grad_norm')
                )
            )
        with open(md_path, 'a') as f:
            f.write("\n".join(lines))

    for r in range(max(1, args.rounds)):
        print(f"=== Round {r+1}/{max(1, args.rounds)} ===")
        round_summary = run_queue(
            current_jobs,
            concurrency=max(1, args.concurrency),
            gpus=gpus,
            dry_run=dry,
            summary_out=Path(args.summary_out),
        )
        completed_summaries.update(round_summary)
        # Append per-round markdown summary
        try:
            append_round_markdown(r+1, current_jobs)
        except Exception as e:
            print(f"[warn] markdown summary failed: {e}")

        if not args.decide or r == args.rounds - 1:
            break

        # Generate next round jobs from decisions
        next_jobs: List[Job] = []
        for jb in current_jobs:
            out_key = str(jb.out_dir)
            summ = round_summary.get(out_key)
            if not summ:
                continue
            decisions = decide(summ)
            # load base cfg
            try:
                with open(jb.cfg_path, "r") as f:
                    base_cfg = json.load(f)
            except Exception:
                continue
            cnt = 0
            for dc in decisions:
                if cnt >= max(1, args.variants_per_run):
                    break
                new_cfg = apply_changes(base_cfg, dc.changes)
                variants_dir = Path("src/lerobot/scripts/train_config")
                variants_dir.mkdir(parents=True, exist_ok=True)
                tag = tiny_id(5)
                new_name = f"{jb.cfg_path.stem}_{tag}.json"
                new_path = variants_dir / new_name
                try:
                    with open(new_path, "w") as f:
                        json.dump(new_cfg, f, indent=2)
                except Exception:
                    continue

                # Schedule job
                name = new_path.stem
                sub = out_root / f"r{r+1}_{name}_{tiny_id(4)}"
                gpu_id = gpus[(len(next_jobs)) % len(gpus)] if gpus else None
                next_jobs.append(
                    Job(
                        cfg_path=new_path,
                        out_dir=sub,
                        device=device,
                        gpu_id=gpu_id,
                        steps=args.steps,
                        log_freq=args.log_freq,
                        eval_freq=args.eval_freq,
                        extra=list(args.extra),
                    )
                )
                cnt += 1
        if not next_jobs:
            print("No next-round jobs were generated by rules; stopping.")
            break
        current_jobs = next_jobs



if __name__ == "__main__":
    main()
