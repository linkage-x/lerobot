"""What a step looks like: how big, how it varies, and whether it slows down near the target.

Four numbers decide whether auto-collected motion is shaped like the demonstrations, and they are
here rather than in a notebook because the roadmap's acceptance criterion names them and a
criterion that cannot be re-read off a command is a criterion nobody will check twice.

**Why four and not one.** A constant-speed script already matches the demonstrations' *median*
step -- 2.00 mm against a measured 2.02 -- so a criterion written on the median alone passes a
teacher that is wrong in the only place that matters. What separates them is the shape:

- **dispersion** (``cv``): the demonstrations' moving steps have CV 0.55; a constant has 0.
- **temporal structure** (``r1``): lag-1 autocorrelation of the step magnitude within an episode
  is 0.92 in the demonstrations, because a hand ramps up and slows down. Per-frame independent
  noise, whatever its distribution, gives ~0 -- which is why adding jitter to a constant does not
  make the data more human, it moves the difference from the first moment to the second.
- **deceleration** (``nearFarRatio``): the median step at 20-50 mm of remaining travel is 26% of
  the median step beyond 200 mm. This is the one that is not cosmetic. Whatever teaches the last
  few centimetres teaches the handoff error, and a teacher that keeps commanding 2 mm there is
  teaching the wrong thing regardless of whether anything can tell it apart from a human.

The band in ``DEMO_BAND`` was measured, not chosen: see its comment for the view and the date.
Reading a *collection session* and reading a *training view* are the same four numbers over two
different files, so both live here -- comparing them across two scripts would compare the scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

# This module is named in an acceptance criterion as a command, so `python tools/fr3/step_profile.py`
# has to work and not only `python -m tools.fr3.step_profile`. Running a file by path puts its own
# directory on `sys.path` and not the repo root, which would leave `--view` working and `--session`
# raising ImportError -- one mode silently unavailable is worse than neither.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Steps below this are the arm holding still, not travelling. Deliberately *not* the dead-time
# rule's threshold: a frame can be too small to be travel and still be a command worth training
# on, so the two questions get two numbers.
MOVING_MM = 0.1

# The dead-time rule's own threshold, used for the still-run statistic and nothing else. It has to
# be this one and not `MOVING_MM`: the only reason to report the runs is to compare them with the
# `k` that rule trims at, and a run measured at a different threshold is not the run it trims.
try:  # pragma: no cover - exercised by whichever import path the machine has
    from tools.fr3.dagger_dataset import STILL_STEP_MM as STILL_MM
except Exception:  # noqa: BLE001
    STILL_MM = 0.05

# Remaining-travel buckets, far to near. The near bucket is 20-50 mm because that is where the
# demonstrations' deceleration is unambiguous; below it they rise again (a final re-approach),
# which is why no criterion here asks the curve to be monotone.
BUCKETS_MM: tuple[tuple[float, float], ...] = (
    (200.0, math.inf),
    (100.0, 200.0),
    (50.0, 100.0),
    (20.0, 50.0),
    (10.0, 20.0),
    (5.0, 10.0),
    (2.0, 5.0),
    (0.0, 2.0),
)
FAR_BUCKET = (200.0, math.inf)
NEAR_BUCKET = (20.0, 50.0)


@dataclass(frozen=True)
class Band:
    """What a pass looks like. Inclusive bounds; `None` means the criterion has no bound there."""

    p50Mm: tuple[float, float]
    cv: tuple[float, float]
    r1Min: float
    nearFarMax: float


# Measured 2026-09-11 on `outputs/exports/training_views/fr3_spacemouse-insert__delta_ee_from_prev_cmd`
# (50 episodes, 24945 frames): moving p50 2.02 mm, CV 0.55, median within-episode r1 0.92 (0.89 on
# moving frames), near/far ratio 0.72/2.79 = 0.26. The band is those numbers with room, not a
# target to tune against -- a run outside it is a run to explain, not a run to nudge.
DEMO_BAND = Band(p50Mm=(1.8, 2.2), cv=(0.45, 0.65), r1Min=0.8, nearFarMax=0.40)

# The legs of a collection cycle that are training data. Same tuple `auto_collect_dataset` selects
# on; imported there rather than duplicated when that module is importable.
try:  # pragma: no cover - exercised by whichever import path the machine has
    from tools.fr3.auto_collect_dataset import RECORDED_PHASES
except Exception:  # noqa: BLE001
    RECORDED_PHASES = ("approach_above_peg", "descend_to_peg", "close_gripper", "lift_8cm_after_grasp")


@dataclass
class EpisodeSteps:
    """One episode's per-frame step magnitude and how far it still had to travel.

    Both in millimetres, same length, one entry per frame. `remainingMm` is measured to the
    episode's own final position rather than to a nominal target: the target is not recorded in
    every source this reads, and what the criterion is about is the *shape* of the approach.
    """

    stepMm: list[float] = field(default_factory=list)
    remainingMm: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Profile:
    frames: int
    movingFrames: int
    p10Mm: float
    p50Mm: float
    p90Mm: float
    cv: float
    r1: float
    buckets: list[dict[str, Any]]
    nearFarRatio: float
    stillRuns: dict[str, float]

    def payload(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "movingFrames": self.movingFrames,
            "p10Mm": self.p10Mm,
            "p50Mm": self.p50Mm,
            "p90Mm": self.p90Mm,
            "cv": self.cv,
            "r1": self.r1,
            "buckets": self.buckets,
            "nearFarRatio": self.nearFarRatio,
            "stillRuns": self.stillRuns,
        }


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if len(values) else float("nan")


def _bucket_p50(episodes: Sequence[EpisodeSteps], lo: float, hi: float) -> tuple[int, float]:
    steps: list[float] = []
    for episode in episodes:
        for step, remaining in zip(episode.stepMm, episode.remainingMm):
            if lo <= remaining < hi:
                steps.append(step)
    if not steps:
        return 0, float("nan")
    return len(steps), float(np.percentile(np.asarray(steps), 50))


def profile(episodes: Sequence[EpisodeSteps]) -> Profile:
    """The four numbers, plus the distributions they are read off.

    `r1` is the *median of the per-episode* autocorrelations, not the autocorrelation of every
    episode concatenated: the joins between episodes are not steps anybody took, and on 50 short
    sequences they are enough to move the number.
    """

    all_steps = np.asarray([s for episode in episodes for s in episode.stepMm], dtype=float)
    moving = all_steps[all_steps >= MOVING_MM]

    r1s: list[float] = []
    for episode in episodes:
        steps = np.asarray(episode.stepMm, dtype=float)
        steps = steps[steps >= MOVING_MM]
        if len(steps) > 30 and steps.std() > 0:
            r1s.append(float(np.corrcoef(steps[:-1], steps[1:])[0, 1]))

    buckets: list[dict[str, Any]] = []
    for lo, hi in BUCKETS_MM:
        count, p50 = _bucket_p50(episodes, lo, hi)
        label = f">={lo:.0f}" if math.isinf(hi) else f"{lo:.0f}-{hi:.0f}"
        buckets.append({"remainingMm": label, "frames": count, "p50Mm": p50})

    _, far = _bucket_p50(episodes, *FAR_BUCKET)
    _, near = _bucket_p50(episodes, *NEAR_BUCKET)
    ratio = float(near / far) if far and math.isfinite(far) and math.isfinite(near) and far > 0 else float("nan")

    runs: list[int] = []
    for episode in episodes:
        run = 0
        for step in episode.stepMm:
            if step < STILL_MM:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)

    return Profile(
        frames=len(all_steps),
        movingFrames=len(moving),
        p10Mm=_percentile(moving, 10),
        p50Mm=_percentile(moving, 50),
        p90Mm=_percentile(moving, 90),
        cv=float(moving.std() / moving.mean()) if len(moving) and moving.mean() > 0 else float("nan"),
        r1=float(np.median(r1s)) if r1s else float("nan"),
        buckets=buckets,
        nearFarRatio=ratio,
        stillRuns={
            "runs": len(runs),
            "mean": float(np.mean(runs)) if runs else float("nan"),
            "median": float(np.median(runs)) if runs else float("nan"),
            "p90": float(np.percentile(runs, 90)) if runs else float("nan"),
            "max": float(max(runs)) if runs else float("nan"),
        },
    )


def check(result: Profile, band: Band = DEMO_BAND) -> list[tuple[str, bool, str]]:
    """Each criterion, whether it passed, and the reading that decided it.

    A `nan` fails rather than passes: "could not be measured" and "measured and within band" are
    the two answers this must never collapse, because the first one is what an empty or malformed
    session produces and it would otherwise read as a clean bill.
    """

    def within(value: float, bounds: tuple[float, float]) -> bool:
        return math.isfinite(value) and bounds[0] <= value <= bounds[1]

    return [
        ("moving p50 mm", within(result.p50Mm, band.p50Mm), f"{result.p50Mm:.2f} in {band.p50Mm}"),
        ("cv", within(result.cv, band.cv), f"{result.cv:.2f} in {band.cv}"),
        ("r1", math.isfinite(result.r1) and result.r1 >= band.r1Min, f"{result.r1:.2f} >= {band.r1Min}"),
        (
            "near/far p50 ratio",
            math.isfinite(result.nearFarRatio) and result.nearFarRatio <= band.nearFarMax,
            f"{result.nearFarRatio:.2f} <= {band.nearFarMax}",
        ),
    ]


def describe(result: Profile, checks: Sequence[tuple[str, bool, str]] | None = None) -> str:
    lines = [
        f"frames={result.frames} moving={result.movingFrames} "
        f"({result.movingFrames / result.frames:.1%})" if result.frames else "frames=0",
        f"  moving step   p10={result.p10Mm:.2f} p50={result.p50Mm:.2f} p90={result.p90Mm:.2f} mm  "
        f"cv={result.cv:.2f}",
        f"  autocorr r1   {result.r1:.2f} (median over episodes, moving frames)",
        f"  near/far p50  {result.nearFarRatio:.2f} "
        f"(20-50 mm remaining vs >=200 mm)",
        f"  still runs    n={result.stillRuns['runs']:.0f} mean={result.stillRuns['mean']:.1f} "
        f"median={result.stillRuns['median']:.0f} p90={result.stillRuns['p90']:.0f} "
        f"max={result.stillRuns['max']:.0f} (< {STILL_MM} mm, the dead-time rule's threshold)",
        "  step p50 by remaining travel:",
    ]
    for bucket in result.buckets:
        if bucket["frames"]:
            lines.append(
                f"    {bucket['remainingMm']:>8} mm  n={bucket['frames']:6d}  p50={bucket['p50Mm']:.2f} mm"
            )
    for name, passed, detail in checks or []:
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return "\n".join(lines)


def episodes_from_view(root: str) -> list[EpisodeSteps]:
    """A training view: the step is the action itself, because the action space is the delta.

    Read through pandas rather than the dataset class on purpose -- this has to run on a machine
    that has the parquet and nothing else.
    """

    import glob

    import pandas as pd

    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}/data")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    action = np.stack(frame["action"].to_numpy())
    state = np.stack(frame["observation.state"].to_numpy())
    episode_index = frame["episode_index"].to_numpy()

    episodes: list[EpisodeSteps] = []
    for index in np.unique(episode_index):
        selected = episode_index == index
        xyz = state[selected, :3]
        remaining = np.linalg.norm(xyz - xyz[-1], axis=1) * 1000.0
        steps = np.linalg.norm(action[selected, :3], axis=1) * 1000.0
        episodes.append(EpisodeSteps(stepMm=list(steps), remainingMm=list(remaining)))
    return episodes


def episodes_from_rows(rows: Iterable[dict[str, Any]], phases: Sequence[str] = RECORDED_PHASES) -> list[EpisodeSteps]:
    """A collection session: the step is the difference between successive commands.

    The first frame of an episode has no predecessor and is *dropped* rather than counted as a
    zero step -- a zero there would land in the still bucket and pull every statistic this file
    reports toward the script looking more still than it is.
    """

    recorded = set(phases)
    open_positions: list[tuple[float, float, float]] | None = None
    episodes: list[EpisodeSteps] = []

    def close(positions: list[tuple[float, float, float]] | None) -> None:
        if not positions or len(positions) < 2:
            return
        array = np.asarray(positions, dtype=float)
        steps = np.linalg.norm(np.diff(array, axis=0), axis=1) * 1000.0
        remaining = np.linalg.norm(array[1:] - array[-1], axis=1) * 1000.0
        episodes.append(EpisodeSteps(stepMm=list(steps), remainingMm=list(remaining)))

    for row in rows:
        kind = row.get("kind")
        if kind == "marker":
            marker = str(row.get("marker"))
            if marker == "episode_start":
                close(open_positions)
                open_positions = []
            elif marker == "episode_end":
                close(open_positions)
                open_positions = None
            continue
        if kind != "frame" or open_positions is None:
            continue
        if str(row.get("phase") or "") not in recorded:
            continue
        action = row.get("sent_action")
        if not isinstance(action, dict):
            continue
        try:
            open_positions.append((float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])))
        except (KeyError, TypeError, ValueError):
            continue
    close(open_positions)
    return episodes


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--view", help="A training view directory (reads data/**/*.parquet).")
    source.add_argument("--session", help="A collection session directory (reads shard_*).")
    parser.add_argument("--closed-shards-only", action="store_true",
                        help="Ignore the shard in flight. Use when reading a live run.")
    parser.add_argument("--check", action="store_true",
                        help="Compare against the demonstrations' band and exit non-zero on a miss.")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    if args.view:
        episodes = episodes_from_view(args.view)
    else:
        from tools.fr3.collection_recorder import iter_rows

        episodes = episodes_from_rows(iter_rows(args.session, closed_only=bool(args.closed_shards_only)))

    result = profile(episodes)
    checks = check(result) if args.check else None
    print(describe(result, checks), flush=True)
    if args.json:
        from pathlib import Path

        payload = result.payload()
        if checks is not None:
            payload["checks"] = [{"name": n, "pass": p, "detail": d} for n, p, d in checks]
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if checks is not None and not all(passed for _name, passed, _detail in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
