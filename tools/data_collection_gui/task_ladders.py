#!/usr/bin/env python
"""Grading ladders: what a rollout achieved, as the furthest precondition it reached.

`success` / `failure` is one bit, and one bit cannot say where a rollout broke. The 20-rollout
batch of 2026-08-31 came back 20/20 `failure` while breaking in three different places -- three
rounds never touched the peg, eight knocked it over while closing, eight carried it and could
not insert it, one inserted it partway. Those are not categories. They are consecutive links in
one chain, each of which has to hold before the next can be attempted, so the grade that carries
all of that information is simply *how far along the chain the rollout got*.

The chain itself is not per-task guesswork. Its links are the moments the set of active contacts
changes: free motion, contact with the object, the object moving with the end effector, contact
with the target, the constraint satisfied, the gripper gone. That sequence is readable off a
single demonstration, which is why a new task needs a ladder file and no new code, and why the
ordinals below are fixed rather than renumbered per task: `stage 3` means "the object moved with
the end effector" whether the task is inserting a peg, opening a drawer or pouring a cup.

Three things are kept apart on purpose:

* `stage` -- how far it got. Ordinal, and deliberately not a score: the step from 2 to 3 is
  "does the constraint hold at all" and the step from 5 to 6 is "is it precise enough". Those
  are not the same size, so these numbers must never be averaged. Report the funnel -- the
  fraction reaching at least each stage -- and read the largest drop between neighbours as the
  next thing to work on.
* `blocker` -- why it stopped there. The same stage fails for different reasons, and merging
  them counts two different work items as one.
* `inDistribution` -- whether the attempt was inside what the demonstrations cover. A failure
  outside it is not evidence about the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LADDER_DIR = Path("tools") / "fr3" / "task_ladders"


@dataclass(frozen=True)
class Stage:
    """One link of the chain, in the shared vocabulary rather than a task's own words."""

    id: str
    ordinal: int
    label: str
    criterion: str


# Ordinals are the shared scale and are never renumbered by a task. A task that lacks a link
# (a wiping task secures no object) omits it; the ordinals of the links it does have stay put,
# so `stage` means the same thing across tasks and across time.
STAGE_VOCABULARY: tuple[Stage, ...] = (
    Stage("not_reached", 0, "未接近", "末端从未进入物体邻域"),
    Stage("approach", 1, "接近", "进入物体邻域，但未建立接触"),
    Stage("contact", 2, "接触", "建立接触，但未形成可承载的约束"),
    Stage("secure", 3, "获取", "物体随末端一起运动"),
    Stage("transport", 4, "转移", "保持约束到达目标邻域"),
    Stage("target_contact", 5, "目标接触", "与目标建立约束，部分满足"),
    Stage("constraint_met", 6, "约束满足", "目标约束完全满足"),
    Stage("release_stable", 7, "释放稳定", "撤出后约束仍然满足"),
)
STAGES_BY_ID = {stage.id: stage for stage in STAGE_VOCABULARY}

# Why it stopped. Shared for the same reason the stages are: "the grasp pose is off" should mean
# one thing across tasks, or the tallies cannot be compared.
BLOCKER_VOCABULARY: tuple[str, ...] = (
    "object_pose_offset",
    "perception",
    "policy_action",
    "hardware",
    "operator_stop",
    "out_of_distribution",
    "unknown",
)

# Not task properties, so a ladder is not allowed to forbid them: any task can be interrupted by
# a person, and any task can produce a round nobody could explain. Leaving these out of a ladder
# would push those rounds into a wrong blocker or out of the record entirely.
ALWAYS_ALLOWED_BLOCKERS: tuple[str, ...] = ("operator_stop", "unknown")

# `aborted` is not a grade. It records that the round is not evidence about the policy -- someone
# walked into the cell, or an acceptance step required stopping early -- so it survives as an
# explicit override of the derived outcome, and the stage is still stored for what it is worth.
ABORTED = "aborted"


class LadderError(ValueError):
    """A ladder file, or a grade against it, that cannot be trusted to mean anything."""


@dataclass(frozen=True)
class Ladder:
    task: str
    label: str
    stages: tuple[Stage, ...]
    instances: dict[str, str]
    terminal: Stage
    blockers: tuple[str, ...]
    # The operator picking a stage from a menu needs the task's own words for it, not the
    # vocabulary's abstract one: "闭合时碰到了销但把它推倒" is a thing they watched happen,
    # "建立接触，但未形成可承载的约束" is a definition they have to translate first.
    blocker_labels: dict[str, str]
    blocker_instances: dict[str, str]

    def stage(self, stage_id: str) -> Stage:
        if stage_id not in self.instances:
            known = ", ".join(s.id for s in self.stages)
            raise LadderError(f"Stage {stage_id!r} is not part of task {self.task!r} (has: {known}).")
        return STAGES_BY_ID[stage_id]

    def by_ordinal(self, ordinal: int) -> Stage:
        for stage in self.stages:
            if stage.ordinal == ordinal:
                return stage
        known = ", ".join(str(s.ordinal) for s in self.stages)
        raise LadderError(f"Task {self.task!r} has no stage {ordinal} (has: {known}).")

    def describe(self) -> dict[str, Any]:
        """The ladder as the grading UI needs it: the menu an operator picks from.

        Serialised from the ladder file rather than restated in the page, so a task that grows
        a stage grows the menu with it and there is no second place to keep in step.
        """
        return {
            "task": self.task,
            "label": self.label,
            "terminal": self.terminal.ordinal,
            "stages": [
                {
                    "id": stage.id,
                    "ordinal": stage.ordinal,
                    "label": stage.label,
                    "criterion": stage.criterion,
                    "instance": self.instances.get(stage.id, ""),
                }
                for stage in self.stages
            ],
            "blockers": [
                {
                    "id": blocker,
                    "label": self.blocker_labels.get(blocker, blocker),
                    "instance": self.blocker_instances.get(blocker, ""),
                }
                for blocker in self.blockers
            ],
        }


def load_ladder(path: Path) -> Ladder:
    """Parse one ladder file, refusing anything whose stages are not the shared vocabulary."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise LadderError(f"Cannot read ladder {path}: {error}") from error
    if not isinstance(raw, dict):
        raise LadderError(f"Ladder {path} is not a mapping.")

    task = str(raw.get("task") or "").strip()
    if not task:
        raise LadderError(f"Ladder {path} declares no task id.")

    entries = raw.get("stages")
    if not isinstance(entries, list) or not entries:
        raise LadderError(f"Ladder {path} declares no stages.")
    stages: list[Stage] = []
    instances: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise LadderError(f"Ladder {path} has a stage that is not a mapping.")
        stage_id = str(entry.get("id") or "").strip()
        if stage_id not in STAGES_BY_ID:
            known = ", ".join(STAGES_BY_ID)
            raise LadderError(f"Ladder {path} uses unknown stage {stage_id!r}. Known stages: {known}.")
        if stage_id in instances:
            raise LadderError(f"Ladder {path} declares stage {stage_id!r} twice.")
        stages.append(STAGES_BY_ID[stage_id])
        instances[stage_id] = str(entry.get("instance") or "").strip()
    # Sorted by the shared ordinal rather than by file order, so a ladder cannot accidentally
    # assert an order the vocabulary does not have.
    stages.sort(key=lambda stage: stage.ordinal)

    terminal_id = str(raw.get("terminal") or stages[-1].id).strip()
    if terminal_id not in instances:
        raise LadderError(f"Ladder {path} sets terminal {terminal_id!r}, which is not one of its stages.")
    terminal = STAGES_BY_ID[terminal_id]
    if terminal.ordinal != stages[-1].ordinal:
        raise LadderError(
            f"Ladder {path} sets terminal {terminal_id!r} below its last stage {stages[-1].id!r}: "
            "a stage past success is a stage the grade would never be able to reward."
        )

    blockers: list[str] = []
    blocker_labels: dict[str, str] = {}
    blocker_instances: dict[str, str] = {}
    for entry in raw.get("blockers") or []:
        mapping = entry if isinstance(entry, dict) else {"id": entry}
        blocker_id = str(mapping.get("id") or "").strip()
        if blocker_id not in BLOCKER_VOCABULARY:
            known = ", ".join(BLOCKER_VOCABULARY)
            raise LadderError(f"Ladder {path} uses unknown blocker {blocker_id!r}. Known blockers: {known}.")
        blockers.append(blocker_id)
        blocker_labels[blocker_id] = str(mapping.get("label") or blocker_id).strip()
        blocker_instances[blocker_id] = str(mapping.get("instance") or "").strip()

    declared = tuple(blockers) or BLOCKER_VOCABULARY
    return Ladder(
        task=task,
        label=str(raw.get("label") or task),
        stages=tuple(stages),
        instances=instances,
        terminal=terminal,
        blockers=declared + tuple(b for b in ALWAYS_ALLOWED_BLOCKERS if b not in declared),
        blocker_labels=blocker_labels,
        blocker_instances=blocker_instances,
    )


def list_ladders(repo_root: Path) -> list[Ladder]:
    """Every ladder the repo ships, so the page can offer them without being told the names.

    A file that will not parse is left out rather than raising: one broken ladder must not take
    the grading UI down for the task the operator is actually running.
    """
    found: list[Ladder] = []
    for path in sorted((Path(repo_root) / LADDER_DIR).glob("*.yaml")):
        try:
            found.append(load_ladder(path))
        except LadderError:
            continue
    return found


def find_ladder(repo_root: Path, task: str) -> Ladder:
    task = str(task or "").strip()
    # Refused rather than joined: this becomes a path, and a rollout grade is not a reason to
    # let a request name a file outside the ladder directory.
    if not task or "/" in task or "\\" in task or task.startswith("."):
        raise LadderError(f"Task ladder id {task!r} is not a bare name.")
    path = repo_root / LADDER_DIR / f"{task}.yaml"
    if not path.is_file():
        raise LadderError(f"No ladder for task {task!r} at {path}.")
    return load_ladder(path)


def normalize_grade(payload: dict[str, Any], ladder: Ladder | None) -> dict[str, Any]:
    """The graded fields of one rollout, or `{}` when it was graded the old way.

    Without a ladder there is nothing to check a stage against, so a stage is refused rather
    than stored as a number whose meaning no reader could recover. With one, `outcome` is
    *derived* -- success is reaching the terminal stage and nothing else -- because an outcome
    stored beside a stage is an outcome that can contradict it, which is exactly how the 20-round
    batch ended up with a partially inserted peg filed under `failure` and readable only in prose.
    """
    has_stage = payload.get("stage") is not None or payload.get("stageId") is not None
    if not has_stage:
        return {}
    if ladder is None:
        raise LadderError("A stage was supplied without a task ladder to interpret it against.")

    stage_id = payload.get("stageId")
    if stage_id is not None:
        stage = ladder.stage(str(stage_id).strip())
    else:
        try:
            ordinal = int(payload["stage"])
        except (TypeError, ValueError) as error:
            raise LadderError(f"Stage must be an integer ordinal (got {payload['stage']!r}).") from error
        stage = ladder.by_ordinal(ordinal)
    if payload.get("stage") is not None and stage_id is not None and int(payload["stage"]) != stage.ordinal:
        raise LadderError(
            f"Stage {payload['stage']!r} and stageId {stage_id!r} disagree "
            f"({stage.id} is stage {stage.ordinal})."
        )

    graded: dict[str, Any] = {
        "taskLadder": ladder.task,
        "stage": stage.ordinal,
        "stageId": stage.id,
        # Stored rather than looked up at read time so the record stays readable on its own --
        # a map drawing "how far along the chain" needs to know where the chain ends, and a
        # grade should keep meaning what it meant when it was made even if the ladder later
        # grows a stage.
        "terminalStage": ladder.terminal.ordinal,
    }

    blocker = payload.get("blocker")
    if blocker not in (None, ""):
        blocker = str(blocker).strip()
        if blocker not in ladder.blockers:
            known = ", ".join(ladder.blockers)
            raise LadderError(f"Blocker {blocker!r} is not one of: {known}.")
        graded["blocker"] = blocker
    elif stage.ordinal < ladder.terminal.ordinal:
        # A rollout that fell short stopped for a reason. Recording "unknown" is allowed;
        # recording nothing loses the question ever having been asked.
        graded["blocker"] = "unknown"

    in_distribution = payload.get("inDistribution")
    if in_distribution is not None:
        graded["inDistribution"] = bool(in_distribution)

    requested = str(payload.get("outcome") or "").strip()
    derived = "success" if stage.ordinal >= ladder.terminal.ordinal else "failure"
    if requested == ABORTED:
        graded["outcome"] = ABORTED
    elif requested and requested != derived:
        raise LadderError(
            f"Outcome {requested!r} contradicts stage {stage.id} of task {ladder.task!r}, "
            f"which grades as {derived!r}. Send the stage and let the outcome follow from it."
        )
    else:
        graded["outcome"] = derived
    return graded


def stage_funnel(entries: list[dict[str, Any]], ladder: Ladder) -> list[dict[str, Any]]:
    """How many graded rollouts reached at least each stage, and where they were lost.

    The funnel is the report, not the mean: the largest `lost` is the next thing to work on.
    Rollouts marked `aborted` are excluded -- they were stopped for reasons that say nothing
    about the policy -- as are ungraded ones, which is why `graded` is returned alongside.
    """
    graded = [
        entry
        for entry in entries
        if entry.get("taskLadder") == ladder.task
        and isinstance(entry.get("stage"), int)
        and entry.get("outcome") != ABORTED
    ]
    rows: list[dict[str, Any]] = []
    previous = len(graded)
    for stage in ladder.stages:
        if stage.ordinal == 0:
            continue
        reached = sum(1 for entry in graded if int(entry["stage"]) >= stage.ordinal)
        rows.append(
            {
                "stage": stage.ordinal,
                "stageId": stage.id,
                "label": stage.label,
                "reached": reached,
                "lost": previous - reached,
                "graded": len(graded),
            }
        )
        previous = reached
    return rows
