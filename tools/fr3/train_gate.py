#!/usr/bin/env python
"""Checkpoint gate: probe every checkpoint a training run saves, and stop the run once the answer is in.

The loop, per new `checkpoints/NNNNNN`:
1. Wait until it is fully written (pretrained_model + training_state, untouched for a minute).
2. Run `holdout_probe.py` on it as a capped side job (`run/user_job.sh --probe`), beside the trainer.
   The trainer is not paused: the probe's MemoryMax keeps any OOM inside the probe's own cgroup,
   and pausing would mean stopping and resuming, which costs more than the ~4 min of shared GPU.
   Too little memory free means waiting and retrying, never touching the trainer.
3. `decide` on the probes so far, and either let the run go on or stop it.

Stopping goes through the gateway (`POST /api/training/stop`) when the gateway is following this
job, so the Training page shows "stopped", and through `systemctl --user stop <unit>` otherwise.
A stopped run keeps every checkpoint it saved; `--resume` continues it.

Everything is appended to `<log-dir>/gate.jsonl`; lines starting with `GATE` go to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_SETTLE_S = 60.0
MEMORY_RETRY_S = 300.0
MAX_PROBE_ATTEMPTS = 3


@dataclass(frozen=True)
class GateRules:
    along_min: float = 0.5  # median share of the funnel's xy motion toward the peg
    cos_min: float = 0.8  # median direction agreement with the funnel's xy motion
    plateau: float = 0.05  # a passing checkpoint that gains less than this on a passing one: done
    fail_step: int = 20000  # by this step ...
    fail_along: float = 0.2  # ... a run still below this is not learning the correction


def passed(summary: dict[str, Any], rules: GateRules) -> bool:
    return summary["along"] >= rules.along_min and summary["cos"] >= rules.cos_min


def best_step(history: list[tuple[int, dict[str, Any]]], rules: GateRules) -> int:
    """The passing checkpoint that goes furthest toward the peg; failing all, the furthest one."""
    pool = [h for h in history if passed(h[1], rules)] or history
    return max(pool, key=lambda h: (h[1]["along"], h[0]))[0]


def decide(history: list[tuple[int, dict[str, Any]]], total_steps: int, rules: GateRules) -> tuple[str, str]:
    """(action, reason) after the last probe in `history` (step order). Action is one of
    `continue`, `stop_pass`, `stop_fail`, `done`."""
    step, cur = history[-1]
    if step >= total_steps:
        return "done", f"{step} is the last checkpoint"
    if passed(cur, rules) and len(history) >= 2:
        prev_step, prev = history[-2]
        gain = cur["along"] - prev["along"]
        if step + (step - prev_step) >= total_steps:
            # Stopping now saves one interval and loses the only checkpoint at the end of the LR decay.
            return "continue", f"plateau at {step}, but the next checkpoint is the last ({total_steps}): train it out"
        if passed(prev, rules) and gain < rules.plateau:
            return "stop_pass", (
                f"{prev_step} and {step} both pass and along gained {gain:+.3f} < {rules.plateau}: plateau"
            )
    if step >= rules.fail_step and cur["along"] < rules.fail_along:
        return "stop_fail", (
            f"along {cur['along']:.3f} < {rules.fail_along} at {step} (>= {rules.fail_step}): "
            "the relabel is not being learned; check the data before spending more steps"
        )
    return "continue", f"along {cur['along']:.3f} cos {cur['cos']:.3f} ({'pass' if passed(cur, rules) else 'below gate'})"


def ready_checkpoints(run_dir: Path, now: float) -> list[int]:
    """Steps whose checkpoint is complete and has not been written to for CKPT_SETTLE_S."""
    out = []
    for d in sorted((run_dir / "checkpoints").glob("[0-9]" * 6)):
        model = d / "pretrained_model"
        weights = [model / "adapter_model.safetensors", model / "model.safetensors"]
        if not (d / "training_state").is_dir() or not any(w.is_file() for w in weights):
            continue
        newest = max(p.stat().st_mtime for p in d.rglob("*"))
        if now - newest >= CKPT_SETTLE_S:
            out.append(int(d.name))
    return out


class Gate:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.train_root: Path = args.train_root.resolve()
        self.run_dir = self.train_root / "outputs/train" / args.job
        self.log_dir: Path = (args.log_dir or self.train_root / "outputs/logs/gate" / args.job).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.log_dir / "gate.jsonl"
        self.rules = GateRules(args.along_min, args.cos_min, args.plateau, args.fail_step, args.fail_along)
        if args.total_steps:
            self.total_steps = args.total_steps
        else:
            cfg_path = self.train_root / args.view / "runs" / args.job / "train_config.generated.json"
            self.total_steps = int(json.loads(cfg_path.read_text())["steps"])

    def say(self, event: str, **fields: Any) -> None:
        entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with self.journal.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print("GATE " + json.dumps(entry, ensure_ascii=False), flush=True)

    def history(self) -> list[tuple[int, dict[str, Any]]]:
        if not self.journal.is_file():
            return []
        probes = {}
        for line in self.journal.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "probe":
                probes[e["step"]] = e["summary"]
        return sorted(probes.items())

    def trainer_alive(self) -> bool:
        r = subprocess.run(["systemctl", "--user", "is-active", "-q", self.args.unit])
        return r.returncode == 0

    def probe(self, step: int) -> dict[str, Any] | None:
        ckpt = f"outputs/train/{self.args.job}/checkpoints/{step:06d}/pretrained_model"
        out = self.log_dir / f"probe_{step:06d}.json"
        name = f"gate-probe-{step:06d}"
        for attempt in range(1, MAX_PROBE_ATTEMPTS + 1):
            cmd = [
                "bash", str(REPO_ROOT / "run/user_job.sh"), "start", name, "--probe",
                "--need-gb", str(self.args.need_gb), "--log", str(self.log_dir / f"probe_{step:06d}.log"), "--",
                str(REPO_ROOT / ".venv-fr3/bin/python"), "-u", str(REPO_ROOT / "tools/fr3/holdout_probe.py"),
                "--train-root", str(self.train_root), "--view", self.args.view, "--out", str(out),
                f"{step:06d}={ckpt}",
            ]  # fmt: skip
            started = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
            if started.returncode != 0:
                self.say("probe_waiting", step=step, attempt=attempt, why=started.stderr.strip()[:300])
                time.sleep(MEMORY_RETRY_S)
                continue
            while subprocess.run(["systemctl", "--user", "is-active", "-q", f"job-{name}"]).returncode == 0:
                time.sleep(10)
            if out.is_file():
                return json.loads(out.read_text())["summary"][f"{step:06d}"]
            self.say("probe_failed", step=step, attempt=attempt, log=str(self.log_dir / f"probe_{step:06d}.log"))
            time.sleep(MEMORY_RETRY_S)  # mostly an OOM at its cap: retrying at once meets the same free memory
        return None

    def stop_training(self, reason: str) -> str:
        try:
            with urllib.request.urlopen(f"{self.args.gateway}/api/training/status", timeout=10) as r:
                t = json.load(r)["training"]
            if t.get("jobName") == self.args.job and t.get("state") in ("starting", "running"):
                req = urllib.request.Request(f"{self.args.gateway}/api/training/stop", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})  # fmt: skip
                with urllib.request.urlopen(req, timeout=60) as r:
                    return "gateway: " + str(json.load(r).get("message", ""))[:200]
        except OSError as exc:  # gateway down or unreachable: the unit is still ours to stop
            reason = f"{reason}; gateway unreachable ({exc})"
        subprocess.run(["systemctl", "--user", "stop", self.args.unit], check=True)
        return f"systemctl stop {self.args.unit} ({reason})"

    def run(self) -> None:
        self.say("start", job=self.args.job, total_steps=self.total_steps, rules=self.rules.__dict__,
                 auto_stop=self.args.auto_stop)  # fmt: skip
        while True:
            done = {s for s, _ in self.history()}
            failed = set()
            pending = [s for s in ready_checkpoints(self.run_dir, time.time()) if s not in done]
            for step in pending:
                summary = self.probe(step)
                if summary is None:
                    failed.add(step)
                    continue
                self.say("probe", step=step, summary=summary)
                history = self.history()
                action, why = decide(history, self.total_steps, self.rules)
                best = best_step(history, self.rules)
                self.say("decision", step=step, action=action, why=why, best=best)
                if action in ("stop_pass", "stop_fail", "done"):
                    if action != "done" and self.trainer_alive():
                        how = self.stop_training(action) if self.args.auto_stop else "auto-stop off: left running"
                        self.say("stopped", step=step, action=action, how=how, best=best)
                    return
            if not self.trainer_alive() and not [
                s for s in ready_checkpoints(self.run_dir, time.time() + CKPT_SETTLE_S) if s not in done | failed
            ]:
                history = self.history()
                self.say("trainer_gone", best=best_step(history, self.rules) if history else None)
                return
            time.sleep(self.args.poll_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--job", required=True, help="training job name (outputs/train/<job>)")
    ap.add_argument("--view", required=True, help="the job's relabelled view, relative to --train-root")
    ap.add_argument("--unit", required=True, help="systemd user unit the trainer runs in")
    ap.add_argument("--total-steps", type=int, default=0, help="default: the job's train_config.generated.json in --view")
    ap.add_argument("--gateway", default="http://10.77.0.156:8765")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--need-gb", type=int, default=26,
                    help="RAM a probe needs free before it starts (its cap is at least this minus 6 GB, and a probe "
                         "peaks at ~16 GB anon since the 10-26 pi_gemma fix; it was 39 GB before)")  # fmt: skip
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--no-auto-stop", dest="auto_stop", action="store_false",
                    help="log the decision but leave the run going")  # fmt: skip
    d = GateRules()
    for field in ("along_min", "cos_min", "plateau", "fail_along"):
        ap.add_argument(f"--{field.replace('_', '-')}", type=float, default=getattr(d, field))
    ap.add_argument("--fail-step", type=int, default=d.fail_step)
    Gate(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
