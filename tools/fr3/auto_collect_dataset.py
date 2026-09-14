"""Turning a night of shards into episodes a policy can be trained on, and refusing the rest.

The recorder's job was to write down what happened, including the parts that went wrong. This
file's job is the opposite: decide what of it is a training sample. The two are separate passes on
purpose -- deletion is irreversible and a selection rule that ran inside the recorder would have
thrown away, at three in the morning, the frames a later question turns out to need.

Four rules decide, and each of them is a decision the card behind this work made explicitly.

**Only the recorded legs.** A cycle's fetch, place, displacement and retrieval are the environment
moving, not the task being performed. They were never published to the recorder in the first
place, so this is a check rather than a filter -- but it is checked, because a frame from an
unrecorded leg reaching a dataset would teach the policy to drive the peg back out of the hole.

**Only episodes that ended holding the peg, unless asked otherwise.** The failures stay on disk and
stay labelled; they are excluded here, not deleted, because behaviour cloning trains on expert
data and a later value-based pass needs exactly the trajectories behaviour cloning must not see.

**Only episodes whose every command was inside the deployment guard.** An episode the step audit
invalidated contains at least one command a rollout would clip, and training on a command the
runtime refuses teaches a policy to speak a language its own runtime does not.

**And the dead-time rule is the one the human path already uses, imported rather than reimplemented.**
``DaggerFrameBuffer`` decides it, so auto-collected stillness and human stillness are trimmed by
the same test, measured the same way -- against the last frame that was *kept*, not against the
previous step, so a slow drift survives and a genuinely motionless run does not. This matters more
here than there: a recorded leg republishes its setpoint through the whole grasp settle, about 19
identical commands at 30 Hz, and the demonstrations' own runs average 17. Note what the imported
rule gets right that a position-only rule would not -- the gripper is one of its three channels,
because closing the fingers while the arm holds still is the single most important frame of a
grasp, and is not dead time at all.

**The dataset is written separately from the demonstrations.** Not merged, because the card's
mixing decision -- sampling weights, or staged training -- is a decision about *proportions*, and
it cannot be made after two sets of frames have been concatenated into one directory. Provenance
travels in a column so that it survives whatever merge is eventually chosen.

What this file deliberately does not own is the schema. Features come from the dataset being
imitated, images must already be in that dataset's geometry, and the action must be in its action
space -- the same three things ``dagger_dataset`` refuses to guess, injected here for the same
reason: they need the policy chain, and a converter that could only be tested on the rig is a
converter whose reasoning errors are found by the rig. What is testable without any of it is the
selection, which is where every rule above lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable

from tools.fr3.collection_recorder import DEPLOYMENT_STEP_LIMIT_MM


# The column that says where a frame came from. A third value is needed because auto-collected
# data is neither a demonstration nor a human correction, and a dataset that can only say "human /
# not human" cannot express the mixture this data is supposed to enter as.
PROVENANCE_KEY = "provenance"
PROVENANCE_AUTO = "auto_collect"

# The legs that are training data. Everything else a cycle does is the environment moving, and is
# listed here rather than inferred so that adding a leg to the loop cannot quietly add it to the
# dataset.
RECORDED_PHASES = ("approach_above_peg", "descend_to_peg", "close_gripper", "lift_8cm_after_grasp")

# Which cycle outcomes become training episodes. `held` only, by default: the rest are kept on
# disk, labelled, and excluded here.
DEFAULT_KEEP_VERDICTS = ("held",)
# The demonstrations' own mean run of consecutive still frames. `dagger_dataset`'s number and its
# argument; repeated as a default here so that a converter run without the dataset stack still
# reports the same trim it would have applied. Measured on the 50 insert demonstrations
# (`fr3_spacemouse-insert__delta_ee_from_prev_cmd`, 2026-09-11): 262 runs of consecutive frames
# under 0.05 mm, mean 17.1, median 17, p90 31, max 39. A trim at 17 therefore leaves a run no
# longer than the demonstrations' own average one, which is the whole claim the number makes.
DEFAULT_MAX_STILL_FRAMES = 17
# An episode shorter than this is an artefact rather than an approach -- a cycle that halted part
# way, or one whose frames were mostly decimated.
DEFAULT_MIN_EPISODE_FRAMES = 8

# Which dead-time rule actually ran. The two are not interchangeable: the fallback has no rotation
# comparison, so it trims a superset of what the human path would trim. Naming the one that
# produced the numbers is the difference between a report a reader can act on and one whose
# meaning depends on what happened to be installed on the machine that produced it.
STILL_RULE_BUFFER = "dagger_buffer"
STILL_RULE_FALLBACK = "position_gripper_only"
STILL_RULE_OFF = "off"


class AutoCollectDatasetError(RuntimeError):
    """A conversion that must not run."""


@dataclass(frozen=True)
class SelectionPolicy:
    keepVerdicts: tuple[str, ...] = DEFAULT_KEEP_VERDICTS
    requireAuditOk: bool = True
    maxStillFrames: int | None = DEFAULT_MAX_STILL_FRAMES
    minEpisodeFrames: int = DEFAULT_MIN_EPISODE_FRAMES
    recordedPhases: tuple[str, ...] = RECORDED_PHASES
    stepLimitMm: float = DEPLOYMENT_STEP_LIMIT_MM


@dataclass
class SelectedEpisode:
    index: int
    kind: str
    verdict: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    stillDropped: int = 0
    offsetMm: float = 0.0
    stillRule: str = STILL_RULE_OFF

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "verdict": self.verdict,
            "frames": len(self.frames),
            "stillDropped": self.stillDropped,
            "stillRule": self.stillRule,
            "offsetMm": self.offsetMm,
        }


@dataclass
class Rejection:
    index: int
    reason: str
    detail: str = ""

    def payload(self) -> dict[str, Any]:
        return {"index": self.index, "reason": self.reason, "detail": self.detail}


def _sent_command(row: dict[str, Any]) -> dict[str, float] | None:
    action = row.get("sent_action")
    if not isinstance(action, dict):
        return None
    try:
        return {key: float(action[key]) for key in
                ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos")}
    except (KeyError, TypeError, ValueError):
        return None


StillFilter = Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], int]]


def _still_filter(max_still_frames: int | None) -> tuple[StillFilter, str]:
    """The dead-time rule, preferring the one the human path already applies.

    `DaggerFrameBuffer` is imported lazily because it pulls in numpy, pyarrow and
    `lerobot.datasets`, and the selection pass is meant to be runnable on a machine that only has
    a copy of the shards. When it is not importable the fallback applies the same three-channel
    test against the same constants -- and the name of whichever ran is returned with it, so that
    a report produced without the dataset stack is never mistaken for one produced with it.
    """

    if max_still_frames is None or max_still_frames < 0:
        return (lambda frames: (list(frames), 0)), STILL_RULE_OFF

    try:
        from tools.fr3.dagger_dataset import DaggerFrameBuffer

        def with_buffer(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
            buffer = DaggerFrameBuffer(max_frames=len(frames) + 1, max_still_frames=max_still_frames)
            for frame in frames:
                buffer.append(frame, is_expert=True, sent_command=_sent_command(frame))
            kept = [frame for span in buffer.spans() for frame in span]
            return kept, buffer.still_frames_dropped

        return with_buffer, STILL_RULE_BUFFER
    except Exception:  # noqa: BLE001 - the stack is absent, not broken
        from tools.fr3.collection_recorder import STILL_STEP_MM

        # Position and gripper only: without scipy there is no rotation comparison here, and a
        # frame whose stillness cannot be fully tested is motion until proven otherwise -- the
        # same direction `_is_dead_time` errs in.
        def without_buffer(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
            kept: list[dict[str, Any]] = []
            dropped = 0
            run = 0
            reference: dict[str, float] | None = None
            for frame in frames:
                command = _sent_command(frame)
                still = False
                if command is not None and reference is not None:
                    moved_mm = 1000.0 * math.dist(
                        (command["ee.x"], command["ee.y"], command["ee.z"]),
                        (reference["ee.x"], reference["ee.y"], reference["ee.z"]),
                    )
                    still = moved_mm < STILL_STEP_MM and abs(
                        command["gripper.pos"] - reference["gripper.pos"]
                    ) < 1e-3
                if still:
                    run += 1
                    if run > max_still_frames:
                        dropped += 1
                        continue
                else:
                    run = 0
                    reference = command
                kept.append(frame)
            return kept, dropped

        return without_buffer, STILL_RULE_FALLBACK


def select_episodes(
    rows: Iterable[dict[str, Any]],
    policy: SelectionPolicy | None = None,
) -> tuple[list[SelectedEpisode], list[Rejection]]:
    """Which frames become training samples, and which episodes are refused and why.

    Pure: rows in, decisions out. Every rule the module docstring lists is applied here and
    nowhere else, so a change to what gets trained on is a change to one function that can be
    checked without a robot, a camera or a dataset.
    """

    policy = policy or SelectionPolicy()
    recorded = set(policy.recordedPhases)
    still_filter, still_rule = _still_filter(policy.maxStillFrames)

    open_episode: SelectedEpisode | None = None
    episodes: dict[int, SelectedEpisode] = {}
    order: list[int] = []
    closed: set[int] = set()
    rejections: list[Rejection] = []
    stray_phase_frames: dict[int, set[str]] = {}
    over_limit: dict[int, int] = {}

    for row in rows:
        kind = row.get("kind")
        if kind == "marker":
            marker = str(row.get("marker"))
            index = int(row.get("cycle", row.get("episode", -1)))
            if marker == "episode_start":
                episode = SelectedEpisode(
                    index=index,
                    kind=str(row.get("cycleKind") or ""),
                    verdict="",
                    offsetMm=float(row.get("offsetMm") or 0.0),
                )
                episodes[index] = episode
                if index not in order:
                    order.append(index)
                open_episode = episode
            elif marker == "episode_end":
                episode = episodes.get(index)
                if episode is not None:
                    episode.verdict = str(row.get("verdict") or "")
                    if not bool(row.get("auditOk", True)):
                        over_limit[index] = over_limit.get(index, 0) or 1
                    closed.add(index)
                open_episode = None
            continue
        if kind != "frame":
            continue
        if open_episode is None:
            continue
        phase = str(row.get("phase") or "")
        if phase not in recorded:
            stray_phase_frames.setdefault(open_episode.index, set()).add(phase)
            continue
        step = row.get("stepMm")
        if step is not None and float(step) > policy.stepLimitMm:
            over_limit[open_episode.index] = over_limit.get(open_episode.index, 0) + 1
        open_episode.frames.append(row)

    selected: list[SelectedEpisode] = []
    for index in order:
        episode = episodes[index]
        if index not in closed:
            # Not "assume it ended well". An episode with no end marker is one whose verdict was
            # never written, which is exactly the episode that was in flight when something went
            # wrong.
            rejections.append(Rejection(index, "no_end_marker", "the run stopped inside this episode"))
            continue
        if stray_phase_frames.get(index):
            rejections.append(
                Rejection(
                    index,
                    "unrecorded_phase_present",
                    f"frames from {sorted(stray_phase_frames[index])} reached the recorder",
                )
            )
            continue
        if episode.verdict not in policy.keepVerdicts:
            rejections.append(Rejection(index, f"verdict_{episode.verdict or 'missing'}", "kept on disk, not trained on"))
            continue
        if policy.requireAuditOk and over_limit.get(index):
            rejections.append(
                Rejection(index, "step_over_guard", f"{over_limit[index]} commands above {policy.stepLimitMm:.1f} mm")
            )
            continue
        kept, still_dropped = still_filter(episode.frames)
        episode.frames = kept
        episode.stillDropped = still_dropped
        episode.stillRule = still_rule
        if len(episode.frames) < policy.minEpisodeFrames:
            rejections.append(
                Rejection(index, "too_short", f"{len(episode.frames)} frames after the dead-time trim")
            )
            continue
        selected.append(episode)
    return selected, rejections


def selection_summary(
    selected: Iterable[SelectedEpisode], rejections: Iterable[Rejection]
) -> dict[str, Any]:
    selected = list(selected)
    rejections = list(rejections)
    reasons: dict[str, int] = {}
    for rejection in rejections:
        reasons[rejection.reason] = reasons.get(rejection.reason, 0) + 1
    kinds: dict[str, int] = {}
    for episode in selected:
        kinds[episode.kind or "?"] = kinds.get(episode.kind or "?", 0) + 1
    # `n/a` rather than a rule name when nothing was selected: no rule trimmed anything, and
    # naming one would describe a pass that did not happen. More than one cannot occur in a
    # single pass, but it is joined rather than silently reduced to the first.
    rules = sorted({episode.stillRule for episode in selected if episode.stillRule})
    return {
        "episodes": len(selected),
        "frames": sum(len(episode.frames) for episode in selected),
        "stillDropped": sum(episode.stillDropped for episode in selected),
        "stillRule": "+".join(rules) if rules else "n/a",
        "byKind": kinds,
        "rejected": len(rejections),
        "rejectedByReason": reasons,
        "selected": [episode.payload() for episode in selected],
        "rejections": [rejection.payload() for rejection in rejections],
    }


def describe_selection(summary: dict[str, Any]) -> str:
    lines = [
        f"episodes={summary['episodes']} frames={summary['frames']} "
        f"still_dropped={summary['stillDropped']} still_rule={summary['stillRule']} "
        f"by_kind={summary['byKind']}",
        f"rejected={summary['rejected']} {summary['rejectedByReason'] or ''}",
    ]
    for episode in summary["selected"]:
        lines.append(
            f"  [{episode['index']:03d}] {episode['kind']:<8} {episode['verdict']:<8} "
            f"frames={episode['frames']:4d} still_dropped={episode['stillDropped']:3d} "
            f"offset_mm={episode['offsetMm']:5.1f}"
        )
    for rejection in summary["rejections"]:
        lines.append(f"  [{rejection['index']:03d}] REJECTED {rejection['reason']}: {rejection['detail']}")
    return "\n".join(lines)


class AutoCollectEpisodeWriter:
    """Writes selected episodes into a dataset, one episode per cycle.

    The dataset and the three things that need the policy chain are passed in, for
    ``dagger_dataset``'s stated reason: they need a GPU-scale install, and a writer that can only
    be exercised on the rig is a writer whose mistakes are found by the rig.

    ``build_frame`` receives one row and must return a dataset frame -- images already in the
    dataset's geometry, action already in its action space. It is the only place that knows the
    schema, and it is expected to raise on a mismatch rather than to fill a missing column,
    because a frame that is merely similar to a recorded one trains a policy on a schema it will
    not meet again.
    """

    def __init__(
        self,
        dataset: Any,
        build_frame: Callable[[dict[str, Any], SelectedEpisode], dict[str, Any]],
        *,
        emit: Callable[[str], None] = print,
    ):
        self._dataset = dataset
        self._build_frame = build_frame
        self._emit = emit

    def write(self, episodes: Iterable[SelectedEpisode]) -> dict[str, Any]:
        written = 0
        frames_written = 0
        for episode in episodes:
            for row in episode.frames:
                self._dataset.add_frame(self._build_frame(row, episode))
            # `parallel_encoding=False` for the reason the recorder gives at its own call site:
            # save_episode has been observed never to return with two cameras and parallel
            # encoding on.
            self._dataset.save_episode(parallel_encoding=False)
            written += 1
            frames_written += len(episode.frames)
        summary = {"episodes": written, "frames": frames_written}
        self._emit(f"[INFO] auto_collect_dataset_written episodes={written} frames={frames_written}")
        return summary


def main(argv: list[str] | None = None) -> int:
    """Selection only. Writing needs the dataset stack and is driven from a runtime, not from here.

    A dry run that needs nothing but the shards is the useful half at this stage: it says exactly
    which episodes a conversion would take and why it would refuse the rest, which is the question
    worth answering before any encoding time is spent.
    """

    import argparse
    import json
    from pathlib import Path

    from tools.fr3.collection_recorder import iter_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="A collection session directory (the one holding shard_*).")
    parser.add_argument(
        "--keep-verdicts",
        default=",".join(DEFAULT_KEEP_VERDICTS),
        help="Comma-separated cycle outcomes to train on. The rest stay on disk, labelled.",
    )
    parser.add_argument("--max-still-frames", type=int, default=DEFAULT_MAX_STILL_FRAMES)
    parser.add_argument("--min-episode-frames", type=int, default=DEFAULT_MIN_EPISODE_FRAMES)
    parser.add_argument("--allow-audit-failures", action="store_true")
    parser.add_argument("--closed-shards-only", action="store_true",
                        help="Ignore the shard that was in flight. Use when reading a live run.")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    policy = SelectionPolicy(
        keepVerdicts=tuple(part.strip() for part in str(args.keep_verdicts).split(",") if part.strip()),
        requireAuditOk=not args.allow_audit_failures,
        maxStillFrames=int(args.max_still_frames),
        minEpisodeFrames=int(args.min_episode_frames),
    )
    selected, rejections = select_episodes(
        iter_rows(args.root, closed_only=bool(args.closed_shards_only)), policy
    )
    summary = selection_summary(selected, rejections)
    print(describe_selection(summary), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
