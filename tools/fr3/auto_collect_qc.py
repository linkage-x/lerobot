"""Reading a night back before believing it: did the arm follow, and is the data shaped like data.

Fifty cycles with nobody touching the rig is half of phase one's criterion. The other half is
that the frames are worth training on, and those two are almost independent -- an arm can run all
night, smoothly, and record something unusable. So this pass runs over the shards *before* any
encoding time is spent, and it answers questions that have measured comparisons rather than
opinions behind them.

**Did the arm actually follow its own setpoint?** Every row stores the command that was sent and
the pose the arm was at when it was chosen, so their difference is exactly the quantity
``command_guard`` calls the leash -- ``target_position - current_position``, the gap that makes
the impedance controller produce force. The demonstrations this policy is trained on run that gap
to p50 5.71 mm, p95 10.65 mm, max 15.92 mm, which is why the deployment leash sits at 20 mm. A
night whose gap looks like theirs was tracking. A night whose gap is flat zero was not recording
what it thought it was; a night whose gap runs to the leash had an arm that was blocked, stalled,
or being dragged.

**Is the motion shaped like the motion the policy already learns from?** The step distribution is
compared to the demonstrations' own (p50 1.59 mm, p95 2.93 mm) rather than only to the guard that
clips at 5.0 mm, because data that never trips the guard but sits at its ceiling is data whose
every frame is a 99.9th-percentile step by the standard of the set it is about to be mixed into.

**And are the still runs shorter than the ones a demonstration contains?** A recorded leg
republishes its setpoint through the whole gripper settle, so an auto episode carries a run of
identical commands that no human produced deliberately. The demonstrations' own runs average 17
frames and reach 39; on a delta action a zero is self-reinforcing, which is why
``dagger_dataset`` truncates the human tail at 17. Whatever the conversion decides to do about it,
the number has to be known before the conversion, not after.

**Are the gaps explained?** Every unrecorded leg emits a marker. A stretch of wall clock between
two episodes with no marker across it is not an environment operation, it is missing data, and
the two are indistinguishable once the process has exited.

**And the verdict itself is a model, so it gets held-out data rather than trust.** Every number
above is counted by ``grasp_verdict``, a classifier reading one scalar -- the gripper width -- and
an acceptance criterion counted by an unvalidated classifier measures the classifier, not the rig.
This is not hypothetical: that rule already had a failure mode nobody had written down, since an
*open* hand reads 1.0, sails over the floor that separates a clamped peg from a hand shut on air,
and was reported as holding the peg. It was found by accident. ``build_review_sample`` exists so
the next one is found on purpose: it draws a stratified sample of cycles, hands over the frames at
the moment of each verdict together with what the classifier said, and ``verdict_agreement`` reads
the labels back as false positives and false negatives per verdict type. The card already asks for
>=95% agreement with human labelling; until now there was no way to measure it.

Nothing here needs the arm, a camera, a GPU or a dataset. It needs the shards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

from tools.fr3.collection_recorder import (
    DEMO_STEP_P50_MM,
    DEMO_STEP_P95_MM,
    DEPLOYMENT_STEP_LIMIT_MM,
    STILL_STEP_MM,
    iter_rows,
    read_shards,
)


# How far the command leads the arm in the demonstrations this policy is trained on, in
# millimetres, and the leash sized to admit all of them. `fr3_act_infer_real_runtime` carries the
# derivation; the numbers are repeated rather than imported because that module pulls in the whole
# policy stack and this one must run on a laptop with a copy of the shards.
DEMO_LEAD_P50_MM = 5.71
DEMO_LEAD_P95_MM = 10.65
DEMO_LEAD_MAX_MM = 15.92
DEPLOYMENT_LEASH_MM = 20.0
# A gap this small over a whole night means the stored state and the stored command are the same
# number, which is what a recorder that published *after* the send would produce. It is a
# recording fault, not a well-tracking arm: a moving impedance-controlled arm always lags.
SUSPICIOUSLY_PERFECT_LEAD_MM = 0.05
# The demonstrations' own runs of consecutive still frames: mean 17, longest 39. `dagger_dataset`
# truncates a human still run at the mean for the reason its docstring gives.
DEMO_STILL_RUN_MEAN = 17
DEMO_STILL_RUN_MAX = 39
# How long a stretch of wall clock between two rows may pass with no marker explaining it, in
# seconds. Generous on purpose: the unrecorded legs of a cycle take tens of seconds, and every one
# of them emits a marker, so this only fires on a stretch nothing accounted for at all.
UNEXPLAINED_GAP_S = 5.0
# How long a *marked* gap may run before it is reported anyway, in seconds.
#
# A marker at the head of a gap is how a legitimate unrecorded leg looks: `_place_the_peg` emits
# one and then walks four waypoints that take tens of seconds between them. But "a marker came
# first" cannot be the whole rule, or a five-minute hang is invisible for as long as something
# announced itself before it. A leg that took two minutes is a stall whether or not it said so.
MARKED_GAP_LIMIT_S = 120.0


@dataclass
class EpisodeReport:
    index: int
    kind: str = ""
    verdict: str = ""
    auditOk: bool = True
    frames: int = 0
    phases: dict[str, int] = field(default_factory=dict)
    offsetMm: float = 0.0
    startT: float | None = None
    endT: float | None = None
    closed: bool = False
    leadMm: list[float] = field(default_factory=list)
    stepMm: list[float] = field(default_factory=list)
    longestStillRun: int = 0
    stepsOverLimit: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "verdict": self.verdict,
            "auditOk": self.auditOk,
            "frames": self.frames,
            "phases": dict(self.phases),
            "offsetMm": self.offsetMm,
            "seconds": None if self.startT is None or self.endT is None else self.endT - self.startT,
            "closed": self.closed,
            "leadMm": _distribution(self.leadMm),
            "stepMm": _distribution([step for step in self.stepMm if step >= STILL_STEP_MM]),
            "longestStillRun": self.longestStillRun,
            "stepsOverLimit": self.stepsOverLimit,
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _lead_mm(row: dict[str, Any]) -> float | None:
    """How far the command led the arm on this frame: `command_guard`'s own leash quantity."""

    state, action = row.get("state") or {}, row.get("sent_action") or {}
    try:
        return 1000.0 * math.dist(
            (float(state["ee.x"]), float(state["ee.y"]), float(state["ee.z"])),
            (float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_session(root: Path | str) -> dict[str, Any]:
    path = Path(root) / "session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_report(root: Path | str, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything the shards can be asked without an arm, a camera or a dataset."""

    root = Path(root)
    shards = read_shards(root)
    rows = list(iter_rows(root) if rows is None else rows)

    episodes: dict[int, EpisodeReport] = {}
    order: list[int] = []
    lead_all: list[float] = []
    step_all: list[float] = []
    camera_skew: dict[str, list[float]] = {}
    camera_failures: dict[str, int] = {}
    frames_without_images = 0
    frames_outside_episode = 0
    unexplained_gaps: list[dict[str, Any]] = []
    stalled_legs: list[dict[str, Any]] = []
    marker_counts: dict[str, int] = {}

    still_run = 0
    previous_t: float | None = None
    previous_was_marker = False
    open_episode: EpisodeReport | None = None

    for row in rows:
        kind = row.get("kind")
        t = row.get("t")
        # Measured across every row, not only across frames. The gap that matters most sits
        # between two *markers* -- an episode that ended and the next one that started five
        # minutes later -- and a check that only looked at frames could not see it at all.
        if previous_t is not None and t is not None:
            gap = float(t) - float(previous_t)
            if not previous_was_marker and gap >= UNEXPLAINED_GAP_S:
                unexplained_gaps.append({"fromT": previous_t, "toT": t, "seconds": gap})
            elif previous_was_marker and gap >= MARKED_GAP_LIMIT_S:
                stalled_legs.append({"fromT": previous_t, "toT": t, "seconds": gap})
        if kind == "marker":
            marker = str(row.get("marker"))
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
            if marker == "episode_start":
                index = int(row.get("cycle", row.get("episode", -1)))
                report = episodes.setdefault(index, EpisodeReport(index=index))
                if index not in order:
                    order.append(index)
                report.kind = str(row.get("cycleKind") or "")
                report.offsetMm = float(row.get("offsetMm") or 0.0)
                report.startT = t
                open_episode = report
                still_run = 0
            elif marker == "episode_end":
                index = int(row.get("cycle", row.get("episode", -1)))
                report = episodes.setdefault(index, EpisodeReport(index=index))
                if index not in order:
                    order.append(index)
                report.verdict = str(row.get("verdict") or "")
                report.auditOk = bool(row.get("auditOk", True))
                report.endT = t
                report.closed = True
                open_episode = None
                still_run = 0
            previous_was_marker = True
            previous_t = t if t is not None else previous_t
            continue

        if kind != "frame":
            continue

        previous_was_marker = False
        previous_t = t if t is not None else previous_t

        episode_index = int(row.get("episode", -1))
        report = open_episode
        if report is None or report.index != episode_index:
            report = episodes.get(episode_index)
        if report is None:
            frames_outside_episode += 1
            still_run = 0
        else:
            report.frames += 1
            phase = str(row.get("phase") or "")
            report.phases[phase] = report.phases.get(phase, 0) + 1

        lead = _lead_mm(row)
        if lead is not None:
            lead_all.append(lead)
            if report is not None:
                report.leadMm.append(lead)

        step = row.get("stepMm")
        if step is not None:
            step = float(step)
            step_all.append(step)
            if report is not None:
                report.stepMm.append(step)
                if step > DEPLOYMENT_STEP_LIMIT_MM:
                    report.stepsOverLimit += 1
            still_run = still_run + 1 if step < STILL_STEP_MM else 0
            if report is not None:
                report.longestStillRun = max(report.longestStillRun, still_run)

        images = row.get("images") or {}
        if not images:
            frames_without_images += 1
        for name, skew in (row.get("cameraSkewMs") or {}).items():
            camera_skew.setdefault(name, []).append(float(skew))
        for name in (row.get("cameraErrors") or {}):
            camera_failures[name] = camera_failures.get(name, 0) + 1

    moving = [step for step in step_all if step >= STILL_STEP_MM]
    report_episodes = [episodes[index].payload() for index in order]
    verdict_counts: dict[str, int] = {}
    for episode in report_episodes:
        verdict_counts[episode["verdict"] or "?"] = verdict_counts.get(episode["verdict"] or "?", 0) + 1

    return {
        "root": str(root),
        "session": read_session(root),
        "shards": {
            "opened": len(shards),
            "closed": sum(1 for shard in shards if shard["closed"]),
            "unfooted": [shard["path"].name for shard in shards if not shard["closed"]],
        },
        "rows": {
            "total": len(rows),
            "frames": sum(1 for row in rows if row.get("kind") == "frame"),
            "markers": sum(1 for row in rows if row.get("kind") == "marker"),
            "markerCounts": marker_counts,
        },
        "episodes": report_episodes,
        "verdicts": verdict_counts,
        "tracking": {
            **_distribution(lead_all),
            "overLeash": sum(1 for lead in lead_all if lead > DEPLOYMENT_LEASH_MM),
            "demoP50Mm": DEMO_LEAD_P50_MM,
            "demoP95Mm": DEMO_LEAD_P95_MM,
            "demoMaxMm": DEMO_LEAD_MAX_MM,
            "leashMm": DEPLOYMENT_LEASH_MM,
        },
        "steps": {
            **_distribution(moving),
            "all": len(step_all),
            "stillFraction": (len(step_all) - len(moving)) / len(step_all) if step_all else None,
            "overLimit": sum(1 for step in step_all if step > DEPLOYMENT_STEP_LIMIT_MM),
            "demoP50Mm": DEMO_STEP_P50_MM,
            "demoP95Mm": DEMO_STEP_P95_MM,
            "limitMm": DEPLOYMENT_STEP_LIMIT_MM,
            "longestStillRun": max((episode["longestStillRun"] for episode in report_episodes), default=0),
            "demoStillRunMean": DEMO_STILL_RUN_MEAN,
            "demoStillRunMax": DEMO_STILL_RUN_MAX,
        },
        "cameras": {
            "skewMs": {name: _distribution(values) for name, values in camera_skew.items()},
            "failures": camera_failures,
            "framesWithoutImages": frames_without_images,
        },
        "continuity": {
            "framesOutsideEpisode": frames_outside_episode,
            "episodesWithoutEnd": [
                episode["index"] for episode in report_episodes if not episode["closed"]
            ],
            "unexplainedGaps": unexplained_gaps,
            "stalledLegs": stalled_legs,
        },
    }


def build_review_sample(
    report: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    *,
    per_verdict: int = 5,
    seed: int = 0,
    frames_per_cycle: int = 3,
) -> dict[str, Any]:
    """A stratified sample of cycles for a person to label, with what the classifier said.

    Stratified by verdict rather than drawn uniformly, and that is the whole point: the verdict
    that matters is the rare one. A uniform sample of a good night is fifty `held` cycles and no
    information about whether `empty` is ever wrong, which is the direction that costs a night.

    The frames handed over are the last ones of the grasp -- the close and the lift, where the
    fingers are around the peg or around nothing and a person can see which. Their image paths are
    included verbatim so the reviewer opens a file rather than reconstructing one.
    """

    import random

    rows = list(rows)
    by_verdict: dict[str, list[int]] = {}
    for episode in report["episodes"]:
        by_verdict.setdefault(episode["verdict"] or "?", []).append(episode["index"])

    rng = random.Random(seed)
    chosen: list[int] = []
    for verdict in sorted(by_verdict):
        indices = sorted(by_verdict[verdict])
        rng.shuffle(indices)
        chosen.extend(sorted(indices[:per_verdict]))
    chosen_set = set(chosen)

    frames_by_cycle: dict[int, list[dict[str, Any]]] = {}
    verdict_by_cycle: dict[int, dict[str, Any]] = {}
    open_cycle: int | None = None
    for row in rows:
        if row.get("kind") == "marker":
            marker = row.get("marker")
            index = int(row.get("cycle", -1))
            if marker == "episode_start":
                open_cycle = index if index in chosen_set else None
            elif marker == "episode_end" and index in chosen_set:
                verdict_by_cycle[index] = {
                    "verdict": row.get("verdict"),
                    "widthNormalized": row.get("widthNormalized"),
                    "auditOk": row.get("auditOk"),
                }
                open_cycle = None
            continue
        if row.get("kind") != "frame" or open_cycle is None:
            continue
        # Only the grasp itself. The approach frames show an empty table either way.
        if str(row.get("phase")) in {"close_gripper", "lift_8cm_after_grasp"}:
            frames_by_cycle.setdefault(open_cycle, []).append(row)

    items = []
    for index in chosen:
        frames = frames_by_cycle.get(index, [])
        picked = frames[-frames_per_cycle:] if frames else []
        detail = verdict_by_cycle.get(index, {})
        items.append(
            {
                "cycle": index,
                "classifier": detail.get("verdict"),
                "widthNormalized": detail.get("widthNormalized"),
                "frames": [
                    {"t": frame.get("t"), "phase": frame.get("phase"), "images": frame.get("images") or {}}
                    for frame in picked
                ],
                # Filled in by the reviewer. Left null rather than pre-filled with the classifier's
                # answer, because a label that starts as the prediction is a label that agrees with
                # it by default.
                "humanLabel": None,
            }
        )
    return {
        "root": report["root"],
        "seed": seed,
        "perVerdict": per_verdict,
        "verdictCounts": {verdict: len(indices) for verdict, indices in by_verdict.items()},
        "items": items,
    }


def verdict_agreement(sample: dict[str, Any]) -> dict[str, Any]:
    """Read the labels back as agreement, and as false positives and negatives per verdict type.

    Per type, not pooled. A pooled agreement of 96% on a night that is 95% `held` says nothing
    about whether the rig can recognise an empty close, and the empty close is the one that ends a
    run. `held` is treated as the positive class because it is the one the acceptance criterion
    counts and the one whose false positives inflate it.
    """

    labelled = [item for item in sample["items"] if item.get("humanLabel")]
    if not labelled:
        return {"labelled": 0, "agreement": None, "byVerdict": {}, "note": "no humanLabel filled in"}

    agree = sum(1 for item in labelled if item["humanLabel"] == item["classifier"])
    by_verdict: dict[str, dict[str, int]] = {}
    for item in labelled:
        said, truth = item["classifier"], item["humanLabel"]
        for verdict in {said, truth}:
            bucket = by_verdict.setdefault(str(verdict), {"said": 0, "truth": 0, "fp": 0, "fn": 0})
        by_verdict[str(said)]["said"] += 1
        by_verdict[str(truth)]["truth"] += 1
        if said != truth:
            by_verdict[str(said)]["fp"] += 1
            by_verdict[str(truth)]["fn"] += 1
    for bucket in by_verdict.values():
        bucket["fpRate"] = bucket["fp"] / bucket["said"] if bucket["said"] else None
        bucket["fnRate"] = bucket["fn"] / bucket["truth"] if bucket["truth"] else None
    return {
        "labelled": len(labelled),
        "unlabelled": len(sample["items"]) - len(labelled),
        "agreement": agree / len(labelled),
        "byVerdict": by_verdict,
    }


def phase_one_verdict(report: dict[str, Any], *, cycles_required: int = 50) -> dict[str, Any]:
    """Phase one's criterion, both halves, as a list of checks that each say why.

    Deliberately not one boolean. The run count and the data quality fail for different reasons
    and are fixed in different places, and a single red light would hide which.
    """

    episodes = report["episodes"]
    tracking = report["tracking"]
    steps = report["steps"]
    continuity = report["continuity"]
    held = sum(1 for episode in episodes if episode["verdict"] == "held")

    checks: list[dict[str, Any]] = [
        {
            "name": "uninterrupted_cycles",
            "ok": held >= cycles_required,
            "detail": f"{held} of {cycles_required} cycles ended holding the peg",
        },
        {
            "name": "every_shard_closed",
            "ok": not report["shards"]["unfooted"],
            "detail": f"unfooted shards: {report['shards']['unfooted'] or 'none'}",
        },
        {
            "name": "no_step_over_the_guard",
            "ok": steps["overLimit"] == 0,
            "detail": f"{steps['overLimit']} commands above the {steps['limitMm']:.1f} mm step guard",
        },
        {
            # Both directions. A gap at the leash means the arm was not following; a gap of
            # essentially zero over a whole night means the stored state is the stored command,
            # which is a recorder fault that looks like perfect tracking.
            "name": "the_arm_was_following",
            "ok": bool(
                tracking["n"]
                and tracking["overLeash"] == 0
                and (tracking["p50"] or 0.0) > SUSPICIOUSLY_PERFECT_LEAD_MM
            ),
            "detail": (
                f"lead p50 {tracking['p50']} mm, p95 {tracking['p95']} mm, max {tracking['max']} mm "
                f"(demos p50 {tracking['demoP50Mm']}, p95 {tracking['demoP95Mm']}, "
                f"max {tracking['demoMaxMm']}; leash {tracking['leashMm']}), "
                f"{tracking['overLeash']} frames over the leash"
            ),
        },
        {
            "name": "no_unexplained_gaps",
            "ok": not continuity["unexplainedGaps"] and continuity["framesOutsideEpisode"] == 0,
            "detail": (
                f"{len(continuity['unexplainedGaps'])} gaps with no marker across them, "
                f"{continuity['framesOutsideEpisode']} frames outside any episode"
            ),
        },
        {
            "name": "no_stalled_legs",
            "ok": not continuity["stalledLegs"],
            "detail": (
                f"{len(continuity['stalledLegs'])} marked gaps over {MARKED_GAP_LIMIT_S:.0f}s "
                "-- a leg that announced itself and then took minutes is still a stall"
            ),
        },
        {
            "name": "every_episode_closed",
            "ok": not continuity["episodesWithoutEnd"],
            "detail": f"episodes with no end marker: {continuity['episodesWithoutEnd'] or 'none'}",
        },
        {
            # A warning rather than a failure: the conversion can truncate, and the number is here
            # so the decision is made knowing it rather than discovered in a trained policy.
            "name": "still_runs_within_the_demonstrations",
            "ok": steps["longestStillRun"] <= DEMO_STILL_RUN_MAX,
            "detail": (
                f"longest still run {steps['longestStillRun']} frames "
                f"(demos mean {steps['demoStillRunMean']}, longest {steps['demoStillRunMax']}; "
                f"dagger_dataset truncates a human run at {steps['demoStillRunMean']})"
            ),
        },
    ]
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def describe_report(report: dict[str, Any], verdict: dict[str, Any] | None = None) -> str:
    """The night in the form somebody actually reads before deciding to keep it."""

    steps, tracking, cameras = report["steps"], report["tracking"], report["cameras"]
    lines = [
        f"root={report['root']}",
        f"shards={report['shards']['opened']} closed={report['shards']['closed']} "
        f"unfooted={report['shards']['unfooted'] or 'none'}",
        f"rows={report['rows']['total']} frames={report['rows']['frames']} markers={report['rows']['markers']}",
        f"episodes={len(report['episodes'])} verdicts={report['verdicts']}",
        f"lead_mm p50={tracking['p50']} p95={tracking['p95']} max={tracking['max']} "
        f"over_leash={tracking['overLeash']} (demo p50={tracking['demoP50Mm']} p95={tracking['demoP95Mm']})",
        f"step_mm p50={steps['p50']} p95={steps['p95']} max={steps['max']} over_limit={steps['overLimit']} "
        f"(demo p50={steps['demoP50Mm']} p95={steps['demoP95Mm']})",
        f"still_fraction={steps['stillFraction']} longest_still_run={steps['longestStillRun']} "
        f"(demo longest {steps['demoStillRunMax']})",
        f"camera_skew_ms={ {name: value['p95'] for name, value in cameras['skewMs'].items()} } "
        f"failures={cameras['failures'] or 'none'} frames_without_images={cameras['framesWithoutImages']}",
        f"continuity gaps={len(report['continuity']['unexplainedGaps'])} "
        f"stalled_legs={len(report['continuity']['stalledLegs'])} "
        f"outside_episode={report['continuity']['framesOutsideEpisode']} "
        f"open_episodes={report['continuity']['episodesWithoutEnd'] or 'none'}",
    ]
    if verdict is not None:
        lines.append(f"phase_one={'PASS' if verdict['ok'] else 'FAIL'}")
        for check in verdict["checks"]:
            lines.append(f"  [{'ok' if check['ok'] else 'FAIL'}] {check['name']}: {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="A collection session directory (the one holding shard_*).")
    parser.add_argument("--cycles-required", type=int, default=50)
    parser.add_argument("--json", default="", help="Also write the full report here.")
    parser.add_argument(
        "--review-sample",
        default="",
        help="Write a stratified verdict-review manifest here for a person to label.",
    )
    parser.add_argument("--review-per-verdict", type=int, default=5)
    parser.add_argument(
        "--review-read",
        default="",
        help="Read a filled-in review manifest back and report agreement and FP/FN per verdict.",
    )
    args = parser.parse_args(argv)

    if args.review_read:
        sample = json.loads(Path(args.review_read).read_text(encoding="utf-8"))
        print(json.dumps(verdict_agreement(sample), ensure_ascii=False, indent=2), flush=True)
        return 0

    report = build_report(args.root)
    verdict = phase_one_verdict(report, cycles_required=int(args.cycles_required))
    print(describe_report(report, verdict), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.review_sample:
        from tools.fr3.collection_recorder import iter_rows

        sample = build_review_sample(
            report, iter_rows(args.root), per_verdict=int(args.review_per_verdict)
        )
        Path(args.review_sample).write_text(
            json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[INFO] review_sample items={len(sample['items'])} -> {args.review_sample} "
            "(fill humanLabel, then re-run with --review-read)",
            flush=True,
        )
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
