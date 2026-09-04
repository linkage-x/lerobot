#!/usr/bin/env python3
"""Compare how much the arm actually moved, per control step, across checkpoints.

A rollout is graded by its outcome, and an outcome is one bit per attempt. That is too coarse
to tell "the policy aims at the wrong place" from "the policy has stopped moving", and the two
call for opposite fixes. The per-step traces under ``outputs/rollout_traces/`` already hold the
answer -- this reads them and says which one happened.

Three things this is careful about, because each one has already misled a reading of these
files:

* **Only the policy's own steps count.** A trace interleaves policy steps with operator
  takeovers (``source``), and a takeover is by construction decisive and large. Pooling them
  flatters exactly the checkpoints that needed the most rescuing.

* **A displacement needs two adjacent policy steps.** Differencing row *i* against row *i-1*
  across a takeover, or across a gap in the step counter, attributes the operator's motion --
  or a whole missing interval -- to the policy.

* **``x, y, z`` is the observed end-effector position, not the command.** The runtime samples
  ``absolute_state_observation_i`` (fr3_act_infer_real_runtime.py:5231), so this measures what
  the arm did, which is only the same as what the policy asked for while the safety guard is
  passing commands through. Steps the leash clamped are therefore dropped by default and
  counted in the report: on a clamped step a large command shows up as a small motion, and
  keeping it would understate exactly the checkpoints that were trying to move.

The metric that separates the two failure modes is the fraction of steps below ~0.05 mm. At
30 fps that is an arm which, for that control tick, did nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROLLOUT_LOG = ROOT / "outputs" / "rollouts" / "rollout_log.jsonl"
DEFAULT_TRACES_ROOT = ROOT / "outputs" / "rollout_traces"

#: The step counted as "the arm did not move this tick", in millimetres. Not a tuned number:
#: the FR3's commanded resolution and the recorded demonstrations both put a deliberate motion
#: an order of magnitude above this, so anything under it is indistinguishable from holding.
STILL_MM = 0.05

#: A second, looser threshold. A policy can be creeping rather than frozen, and the two look
#: different here while looking identical at STILL_MM alone.
CREEP_MM = 0.2

#: The band a gripper command falls in when the policy is committing to neither open nor
#: closed. A delta-action policy that has learned to hesitate shows up here before it shows up
#: anywhere else, because the gripper is the one dimension with no momentum to hide behind.
GRIPPER_MID = (0.05, 0.95)

#: The guard's verdict on a step, as written by `limit_command_for_safety`. Anything else means
#: the motion on that row is the guard's, not the policy's.
UNCLAMPED_STATUS = "pass"

#: How much of the regression has to be undone before the fix is called confirmed. Fixed here,
#: and deliberately fixed *before* any candidate was measured: a threshold chosen after seeing
#: the number is not a threshold. Expressed as a fraction of the gap between the regressed run
#: and the baseline, so it does not depend on the absolute rate of either.
RECOVERY_CONFIRMED = 0.80
RECOVERY_PARTIAL = 0.40

#: `logPath` ends in the session stamp the trace directory is named after.
_SESSION_RE = re.compile(r"_(\d{8}_\d{6})\.log$")


class ComparisonError(RuntimeError):
    """A refusal the operator can fix."""


@dataclass
class TraceRow:
    step: int
    x: float
    y: float
    z: float
    gripper_cmd: float
    gripper_raw: float
    status: str
    source: str


@dataclass
class RolloutMotion:
    """One rollout, reduced to the policy-owned motion it produced."""

    name: str
    outcome: str
    policy_steps: int
    steps_mm: list[float] = field(default_factory=list)
    gripper_cmd: list[float] = field(default_factory=list)
    gripper_raw: list[float] = field(default_factory=list)
    clamped_pairs: int = 0

    @property
    def still_frac(self) -> float:
        """This rollout's own still-step rate, so the pooled number can be read against its spread."""
        return _fraction(self.steps_mm, STILL_MM)

    @property
    def usable_pairs(self) -> int:
        return len(self.steps_mm)


@dataclass
class GroupMotion:
    """A checkpoint's rollouts, pooled."""

    label: str
    checkpoint: str
    rollouts: list[RolloutMotion] = field(default_factory=list)
    missing_traces: int = 0

    @property
    def steps_mm(self) -> list[float]:
        return [value for rollout in self.rollouts for value in rollout.steps_mm]

    @property
    def gripper_cmd(self) -> list[float]:
        return [value for rollout in self.rollouts for value in rollout.gripper_cmd]

    @property
    def gripper_raw(self) -> list[float]:
        return [value for rollout in self.rollouts for value in rollout.gripper_raw]


def read_trace(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            try:
                rows.append(
                    TraceRow(
                        step=int(record["step"]),
                        x=float(record["x"]),
                        y=float(record["y"]),
                        z=float(record["z"]),
                        gripper_cmd=float(record["gripper_cmd"]),
                        gripper_raw=float(record["gripper_raw"]),
                        status=str(record["status"]),
                        source=str(record["source"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ComparisonError(f"{path} is not a rollout trace: {exc}") from exc
    return rows


def motion_from_trace(
    rows: Sequence[TraceRow],
    *,
    name: str,
    outcome: str = "",
    include_clamped: bool = False,
) -> RolloutMotion:
    """Reduce one trace to the per-step displacements the policy is answerable for.

    A pair contributes only when both of its rows are the policy's and their step counters are
    adjacent. Everything else -- a takeover, the step after a takeover, a gap where rows were
    dropped -- carries someone else's motion or an unknown interval.
    """
    motion = RolloutMotion(name=name, outcome=outcome, policy_steps=0)
    previous: TraceRow | None = None
    for row in rows:
        is_policy = row.source == "policy"
        if is_policy:
            motion.policy_steps += 1
            motion.gripper_cmd.append(row.gripper_cmd)
            motion.gripper_raw.append(row.gripper_raw)
        if previous is not None and is_policy and previous.source == "policy":
            if row.step - previous.step == 1:
                # Either end clamped makes the interval between them the guard's, not the policy's.
                clamped = not (row.status == previous.status == UNCLAMPED_STATUS)
                if clamped and not include_clamped:
                    motion.clamped_pairs += 1
                else:
                    motion.steps_mm.append(
                        1000.0
                        * math.dist((previous.x, previous.y, previous.z), (row.x, row.y, row.z))
                    )
        previous = row
    return motion


def trace_path_for(entry: dict[str, Any], traces_root: Path) -> Path | None:
    """Where this rollout-log entry's per-step CSV lives, if it can be named at all.

    The log records the *session* log path and an index within that session; the trace writer
    names its directory after the same stamp. Entries predating the trace writer resolve to a
    path that does not exist, which is why the caller counts them rather than failing.
    """
    match = _SESSION_RE.search(str(entry.get("logPath") or ""))
    index = entry.get("rolloutIndex")
    if not match or not isinstance(index, int):
        return None
    return traces_root / f"session_{match.group(1)}" / f"rollout_{index:03d}.csv"


def load_group(
    label: str,
    checkpoint: str,
    entries: Iterable[dict[str, Any]],
    traces_root: Path,
    *,
    outcomes: Sequence[str] | None = None,
    include_clamped: bool = False,
) -> GroupMotion:
    group = GroupMotion(label=label, checkpoint=checkpoint)
    for entry in entries:
        if entry.get("checkpointId") != checkpoint:
            continue
        outcome = str(entry.get("outcome") or "")
        if outcomes and outcome not in outcomes:
            continue
        path = trace_path_for(entry, traces_root)
        if path is None or not path.is_file():
            group.missing_traces += 1
            continue
        group.rollouts.append(
            motion_from_trace(
                read_trace(path),
                name=f"{path.parent.name}/{path.stem}",
                outcome=outcome,
                include_clamped=include_clamped,
            )
        )
    if not group.rollouts:
        raise ComparisonError(
            f"No usable trace for {label} ({checkpoint}): "
            f"{group.missing_traces} logged rollouts have no trace file on disk."
        )
    return group


def _fraction(values: Sequence[float], below: float) -> float:
    return sum(1 for value in values if value < below) / len(values) if values else float("nan")


def _mid_band_fraction(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(1 for value in values if GRIPPER_MID[0] < value < GRIPPER_MID[1]) / len(values)


def summarise(group: GroupMotion) -> dict[str, Any]:
    steps = group.steps_mm
    per_rollout = [rollout.policy_steps for rollout in group.rollouts]
    # Kept per rollout as well as pooled: a pooled rate says nothing about whether a candidate
    # landing between two groups is distinguishable from either, and that is precisely the case
    # a single-variable retrain is most likely to produce.
    still_by_rollout = sorted(
        rollout.still_frac for rollout in group.rollouts if rollout.steps_mm
    )
    return {
        "label": group.label,
        "checkpoint": group.checkpoint,
        "rollouts": len(group.rollouts),
        "missing_traces": group.missing_traces,
        "policy_steps": sum(per_rollout),
        "usable_pairs": len(steps),
        "clamped_pairs": sum(rollout.clamped_pairs for rollout in group.rollouts),
        "median_mm": statistics.median(steps) if steps else float("nan"),
        "p90_mm": (
            statistics.quantiles(steps, n=10)[8] if len(steps) >= 10 else float("nan")
        ),
        "max_mm": max(steps) if steps else float("nan"),
        "still_frac": _fraction(steps, STILL_MM),
        "creep_frac": _fraction(steps, CREEP_MM),
        "steps_per_rollout_median": statistics.median(per_rollout) if per_rollout else 0,
        "steps_per_rollout_min": min(per_rollout) if per_rollout else 0,
        "steps_per_rollout_max": max(per_rollout) if per_rollout else 0,
        "gripper_cmd_mid_frac": _mid_band_fraction(group.gripper_cmd),
        "gripper_raw_mid_frac": _mid_band_fraction(group.gripper_raw),
        "still_frac_by_rollout": still_by_rollout,
        "still_frac_min": still_by_rollout[0] if still_by_rollout else float("nan"),
        "still_frac_max": still_by_rollout[-1] if still_by_rollout else float("nan"),
        "still_frac_median": (
            statistics.median(still_by_rollout) if still_by_rollout else float("nan")
        ),
        "outcomes": {
            outcome: sum(1 for r in group.rollouts if r.outcome == outcome)
            for outcome in sorted({r.outcome for r in group.rollouts})
        },
    }


def recovery(baseline: dict[str, Any], regressed: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """How much of the regression the candidate undid, on the still-step fraction.

    Reported as a fraction of the gap the regression opened, so it means the same thing whether
    the baseline sits at 10% or at 30%. A gap that never opened is not something a candidate can
    close, and is refused rather than divided by.
    """
    gap = regressed["still_frac"] - baseline["still_frac"]
    if not gap > 0:
        raise ComparisonError(
            f"{regressed['label']} is not worse than {baseline['label']} on the still-step "
            f"fraction ({regressed['still_frac']:.3f} vs {baseline['still_frac']:.3f}); there is "
            "no regression for a candidate to recover."
        )
    fraction = (regressed["still_frac"] - candidate["still_frac"]) / gap
    if fraction >= RECOVERY_CONFIRMED:
        verdict = "CONFIRMED"
    elif fraction >= RECOVERY_PARTIAL:
        verdict = "PARTIAL"
    else:
        verdict = "NOT RECOVERED"
    return {"recovered": fraction, "verdict": verdict, "gap": gap}


def format_report(summaries: Sequence[dict[str, Any]], verdict: dict[str, Any] | None) -> str:
    rows: list[tuple[str, ...]] = [("metric", *[s["label"] for s in summaries])]

    def add(name: str, key: str, fmt: str) -> None:
        rows.append((name, *[format(s[key], fmt) for s in summaries]))

    rows.append(("rollouts", *[f"{s['rollouts']} ({s['missing_traces']} no trace)" for s in summaries]))
    rows.append(("policy steps", *[f"{s['policy_steps']}" for s in summaries]))
    rows.append(("usable step pairs", *[f"{s['usable_pairs']}" for s in summaries]))
    rows.append(("clamped pairs dropped", *[f"{s['clamped_pairs']}" for s in summaries]))
    add("median step (mm)", "median_mm", ".3f")
    add("p90 step (mm)", "p90_mm", ".3f")
    add("max step (mm)", "max_mm", ".3f")
    add(f"frac < {STILL_MM} mm", "still_frac", ".1%")
    add(f"frac < {CREEP_MM} mm", "creep_frac", ".1%")
    rows.append(
        (
            "policy steps / rollout",
            *[
                f"{s['steps_per_rollout_median']} ({s['steps_per_rollout_min']}-{s['steps_per_rollout_max']})"
                for s in summaries
            ],
        )
    )
    rows.append(
        (
            f"per-rollout <{STILL_MM} mm",
            *[
                f"{s['still_frac_median']:.1%} ({s['still_frac_min']:.1%}-{s['still_frac_max']:.1%})"
                for s in summaries
            ],
        )
    )
    add("gripper cmd mid-band", "gripper_cmd_mid_frac", ".1%")
    add("gripper raw mid-band", "gripper_raw_mid_frac", ".1%")
    rows.append(
        (
            "outcomes",
            *[", ".join(f"{k}:{v}" for k, v in s["outcomes"].items()) or "-" for s in summaries],
        )
    )

    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))

    if verdict is not None:
        lines.append("")
        lines.append(
            f"recovered {verdict['recovered']:.0%} of the {verdict['gap']:.1%} regression "
            f"in the <{STILL_MM} mm fraction -> {verdict['verdict']} "
            f"(confirmed >= {RECOVERY_CONFIRMED:.0%}, partial >= {RECOVERY_PARTIAL:.0%})"
        )
    return "\n".join(lines)


def parse_group(value: str) -> tuple[str, str]:
    label, separator, checkpoint = value.partition("=")
    if not separator or not label.strip() or not checkpoint.strip():
        raise argparse.ArgumentTypeError(
            f"--group wants LABEL=CHECKPOINT_ID (got {value!r}), for example "
            "baseline=L4_full48_holdout22_40/030000"
        )
    return label.strip(), checkpoint.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        type=parse_group,
        required=True,
        metavar="LABEL=CHECKPOINT_ID",
        help="A column of the report. Repeat; order is kept.",
    )
    parser.add_argument("--rollout-log", type=Path, default=DEFAULT_ROLLOUT_LOG)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument(
        "--outcomes",
        default="",
        help="Comma-separated outcomes to keep (default: every graded rollout). Filtering by "
        "outcome selects on the thing being explained, so prefer reading the outcome mix the "
        "report prints.",
    )
    parser.add_argument(
        "--include-clamped",
        action="store_true",
        help="Keep step pairs the safety leash limited. Off by default: on those rows the "
        "recorded motion is the guard's, not the policy's.",
    )
    parser.add_argument(
        "--recovery",
        default="",
        metavar="BASELINE,REGRESSED,CANDIDATE",
        help="Three group labels to judge against the thresholds fixed in this file.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summaries as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.rollout_log.is_file():
        raise ComparisonError(f"No rollout log at {args.rollout_log}")
    entries = [
        json.loads(line) for line in args.rollout_log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    outcomes = tuple(part.strip() for part in args.outcomes.split(",") if part.strip())

    summaries = [
        summarise(
            load_group(
                label,
                checkpoint,
                entries,
                args.traces_root,
                outcomes=outcomes,
                include_clamped=args.include_clamped,
            )
        )
        for label, checkpoint in args.groups
    ]

    verdict = None
    if args.recovery:
        wanted = [part.strip() for part in args.recovery.split(",")]
        if len(wanted) != 3:
            raise ComparisonError("--recovery wants three group labels: BASELINE,REGRESSED,CANDIDATE")
        by_label = {summary["label"]: summary for summary in summaries}
        missing = [label for label in wanted if label not in by_label]
        if missing:
            raise ComparisonError(f"--recovery names groups that were not compared: {', '.join(missing)}")
        verdict = recovery(*(by_label[label] for label in wanted))

    if args.json:
        print(json.dumps({"groups": summaries, "recovery": verdict}, indent=2))
    else:
        print(format_report(summaries, verdict))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
