#!/usr/bin/env python3
"""Judge a batch of finished rollouts offline, and score the judge against people's labels.

    # every grasp-loop run and the GUI rollout log, with the no-API mock provider
    python tools/fr3/fr3_judge_rollouts.py judge \\
        --grasp-loop 'outputs/analysis/grasp_loop/grasp_2026*.jsonl' \\
        --rollout-log outputs/rollouts/rollout_log.jsonl --provider mock

    # after filling the `human_label` column of the CSV it wrote
    python tools/fr3/fr3_judge_rollouts.py benchmark --decisions <out>.jsonl --labels <out>.csv

`judge` writes `<out>.jsonl` (one decision record per line: the decision log) and `<out>.csv`
beside it (the same decisions as a labelling sheet: fill `human_label` with one of the task's
options). It only reads the rollout records; it never touches a robot, a dataset, or the log it
reads. `benchmark` joins the two on (episode, span, task) and reports accuracy, the confusion
matrix, per-class precision/recall, calibration, and what each auto-accept threshold would cost.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fr3.judgment import (  # noqa: E402
    AMBIGUOUS,
    PROVIDERS,
    TASKS,
    UNKNOWN,
    DecisionLog,
    FeatureSnapshot,
    JudgeConfig,
    RolloutJudge,
    decision_record,
)
from tools.fr3.judgment_features import APPLIES, RULES, load_grasp_loop_run, load_rollout_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "analysis" / "judge"

CSV_FIELDS = (
    "episode_id",
    "expert_span_id",
    "task_name",
    "decision",
    "confidence",
    "route",
    "decided_by",
    "status",
    "alternatives",
    "options",
    "operator_hint",
    "human_label",
    "human_notes",
    "features_sha256",
)


def expand(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern)) or [pattern]
        # A grasp run's force trace sits beside it with the same prefix; it holds no trials.
        paths += [Path(m) for m in matched if not m.endswith("_force.jsonl")]
    return paths


def collect(grasp_loop: Iterable[str], rollout_logs: Iterable[str]) -> list[FeatureSnapshot]:
    snaps: list[FeatureSnapshot] = []
    for path in expand(grasp_loop):
        snaps += load_grasp_loop_run(path)
    for path in expand(rollout_logs):
        snaps += load_rollout_log(path)
    return snaps


def judge_batch(
    judge: RolloutJudge, snaps: list[FeatureSnapshot], tasks: list[str], log: DecisionLog
) -> tuple[list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    tally: Counter = Counter()
    for snap in snaps:
        for task in tasks:
            decision = judge.evaluate(task, snap)
            if decision.source == "not_applicable":
                tally[(task, "not_applicable")] += 1
                continue
            record = decision_record(snap, decision, judge.provider)
            log.append(record)
            records.append(record)
            tally[(task, f"{decision.source}/{decision.route}")] += 1
    return records, tally


def operator_hint(human: dict[str, Any]) -> str:
    """What people already said, shown to the labeller as context (not a label in the taxonomy)."""

    parts = [f"{k}={v}" for k, v in human.items() if k != "note"]
    if human.get("note"):
        parts.append(f"note={str(human['note'])[:120]}")
    return "; ".join(parts)


def write_sheet(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            task = TASKS[r["task_name"]]
            writer.writerow(
                {
                    "episode_id": r["episode_id"],
                    "expert_span_id": r["expert_span_id"] or "",
                    "task_name": r["task_name"],
                    "decision": r["decision"],
                    "confidence": "" if r["confidence"] is None else r["confidence"],
                    "route": r["route"] or "",
                    "decided_by": r["decided_by"],
                    "status": r["status"],
                    "alternatives": json.dumps(r["alternatives"]),
                    "options": "|".join(task.labels) + ("" if task.fallback in task.labels else f"|{task.fallback}"),
                    "operator_hint": operator_hint(r.get("human") or {}),
                    "human_label": "",
                    "human_notes": "",
                    "features_sha256": r["features_sha256"],
                }
            )


def cmd_judge(args: argparse.Namespace) -> int:
    config = JudgeConfig.load(
        args.config,
        provider=args.provider,
        timeout_s=args.timeout_s,
        use_rules=False if args.no_rules else None,
    )
    provider = PROVIDERS[config.provider]()
    ok, why = provider.available()
    if not ok:
        # Refused up front: a batch of nothing but fallbacks is not a result anyone wants.
        print(f"provider {config.provider!r} is not available: {why}", file=sys.stderr)
        return 2
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        print(f"unknown tasks: {', '.join(unknown)}; known: {', '.join(TASKS)}", file=sys.stderr)
        return 2
    snaps = collect(args.grasp_loop, args.rollout_log)
    if args.limit:
        snaps = snaps[: args.limit]
    if not snaps:
        print("no rollouts found in the given inputs", file=sys.stderr)
        return 2
    out = args.out or DEFAULT_OUT_DIR / f"judge_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    if out.exists():
        print(f"{out} exists; pass a new --out", file=sys.stderr)
        return 2
    judge = RolloutJudge(provider, config, rules=RULES, applies=APPLIES)
    records, tally = judge_batch(judge, snaps, tasks, DecisionLog(out))
    sheet = out.with_suffix(".csv")
    write_sheet(sheet, records)
    print(f"snapshots={len(snaps)} decisions={len(records)} provider={provider.name} model={provider.model}")
    for task in tasks:
        row = {k[1]: v for k, v in sorted(tally.items()) if k[0] == task}
        print(f"  {task}: {row}")
    print(out)
    print(sheet)
    return 0


# --------------------------------------------------------------------------------- benchmark ---


def read_labels(path: Path) -> dict[tuple[str, str, str], str]:
    labels: dict[tuple[str, str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("human_label") or "").strip().upper()
            if label:
                labels[(row["episode_id"], row.get("expert_span_id") or "", row["task_name"])] = label
    return labels


def read_decisions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def benchmark(
    decisions: list[dict[str, Any]],
    labels: dict[tuple[str, str, str], str],
    *,
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    bins: int = 5,
) -> dict[str, Any]:
    """Per task: how the provider's own answers compare with the people's.

    Rule-decided rows are scored separately (they are the code's answers, not the provider's),
    and fallbacks are counted as abstentions -- routed to a person, so never wrong, never right.
    """

    by_task: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for d in decisions:
        key = (d["episode_id"], d.get("expert_span_id") or "", d["task_name"])
        if key in labels:
            by_task[d["task_name"]].append((d, labels[key]))

    report: dict[str, Any] = {}
    for task, pairs in sorted(by_task.items()):
        provider = [(d, y) for d, y in pairs if d["decided_by"] == "provider"]
        rules = [(d, y) for d, y in pairs if d["decided_by"] == "rule"]
        abstained = sum(1 for d, _ in pairs if d["decided_by"] == "fallback")
        classes = sorted({y for _, y in provider} | {d["decision"] for d, _ in provider})
        confusion = {y: {p: 0 for p in classes} for y in classes}
        for d, y in provider:
            confusion[y][d["decision"]] += 1
        per_class = {}
        for c in classes:
            tp = confusion[c][c]
            predicted = sum(confusion[y][c] for y in classes)
            actual = sum(confusion[c].values())
            per_class[c] = {
                "precision": round(tp / predicted, 3) if predicted else None,
                "recall": round(tp / actual, 3) if actual else None,
                "support": actual,
            }
        calibration = []
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            inside = [(d, y) for d, y in provider if lo <= d["confidence"] < hi or (b == bins - 1 and d["confidence"] == 1.0)]
            if inside:
                calibration.append(
                    {
                        "bin": [round(lo, 2), round(hi, 2)],
                        "n": len(inside),
                        "mean_confidence": round(sum(d["confidence"] for d, _ in inside) / len(inside), 3),
                        "accuracy": round(sum(d["decision"] == y for d, y in inside) / len(inside), 3),
                    }
                )
        sweep = []
        for t in thresholds:
            accepted = [(d, y) for d, y in provider if d["confidence"] >= t]
            sweep.append(
                {
                    "auto_accept_at": t,
                    "coverage": round(len(accepted) / len(provider), 3) if provider else None,
                    "accuracy_accepted": round(sum(d["decision"] == y for d, y in accepted) / len(accepted), 3) if accepted else None,
                    "to_review": len(provider) - len(accepted),
                }
            )
        report[task] = {
            "labelled": len(pairs),
            "provider": {
                "n": len(provider),
                "accuracy": round(sum(d["decision"] == y for d, y in provider) / len(provider), 3) if provider else None,
                "confusion": confusion,
                "per_class": per_class,
                "calibration": calibration,
                "threshold_sweep": sweep,
            },
            "rules": {
                "n": len(rules),
                "accuracy": round(sum(d["decision"] == y for d, y in rules) / len(rules), 3) if rules else None,
                "disagreements": [
                    {"episode_id": d["episode_id"], "rule": d["decision"], "human": y} for d, y in rules if d["decision"] != y
                ],
            },
            "abstained": abstained,
        }
    return report


def cmd_benchmark(args: argparse.Namespace) -> int:
    labels = read_labels(args.labels)
    if not labels:
        print(f"no human_label filled in {args.labels}", file=sys.stderr)
        return 2
    valid = {t: set(TASKS[t].labels) | {AMBIGUOUS, UNKNOWN} for t in TASKS}
    bad = sorted({(k[2], v) for k, v in labels.items() if k[2] in valid and v not in valid[k[2]]})
    if bad:
        print("labels outside their task's options: " + ", ".join(f"{t}={v}" for t, v in bad), file=sys.stderr)
        return 2
    report = benchmark(read_decisions(args.decisions), labels)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(args.out)
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    judge = sub.add_parser("judge", help="judge a batch of rollouts; write the decision log and a labelling sheet")
    judge.add_argument("--grasp-loop", nargs="*", default=[], help="grasp-loop row files or globs")
    judge.add_argument("--rollout-log", nargs="*", default=[], help="rollout_log.jsonl files")
    judge.add_argument("--tasks", default=",".join(TASKS), help=f"comma-separated, of: {', '.join(TASKS)}")
    judge.add_argument("--provider", choices=sorted(PROVIDERS), default=None, help="default: the config's, else mock")
    judge.add_argument("--config", type=Path, default=None, help="JSON with JudgeConfig fields")
    judge.add_argument("--timeout-s", type=float, default=None)
    judge.add_argument("--no-rules", action="store_true", help="put every case to the provider (for a benchmark)")
    judge.add_argument("--limit", type=int, default=0, help="judge only the first N snapshots")
    judge.add_argument("--out", type=Path, default=None, help="decision log path (.jsonl); the .csv goes beside it")
    judge.set_defaults(func=cmd_judge)

    bench = sub.add_parser("benchmark", help="score a decision log against a filled labelling sheet")
    bench.add_argument("--decisions", type=Path, required=True)
    bench.add_argument("--labels", type=Path, required=True, help="the judge's CSV with human_label filled in")
    bench.add_argument("--out", type=Path, default=None)
    bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
