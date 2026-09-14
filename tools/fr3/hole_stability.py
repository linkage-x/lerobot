"""How much did the hole move, and did it wander or drift? The two have different fixes.

The fixture was never bolted down. That is one sentence and it reaches almost every terminal
number this project has, because a hole that can be nudged is not the hole the geometry describes.

**It is a candidate explanation for a number the roadmap treats as a puzzle.** The capture radius
measured out at 4.2 mm against a nominal single-side clearance of 2.5 mm. A chamferless peg
landing on the edge of a compliant hole can push the hole into alignment, so the effective capture
radius is genuinely larger than the clearance allows -- no wide confidence interval required. If
that is what happened, bolting the fixture down *shrinks* the capture radius toward 2.5 mm, and
the eight-point 7 mm search ring was sized against 4.2. The roadmap already states what that
costs: the ring covers a 9.7 mm disc at 4.2 mm and "leaves a dead annulus between 2.5 and 4.5 mm
if it is really the nominal clearance."

**And it decides whether the existing demonstrations are one task or several.** A moving hole does
not by itself mislabel a demonstration -- the operator saw the hole and compensated. What matters
is whether the hole sat in a *different place* from episode to episode, because then two frames
that look alike carry different correct terminal actions, which is the same unobserved-variable
disease this project already diagnosed in search controllers, and it puts a floor on precision
that no quantity of data removes.

So the question is not "should the demonstrations be re-collected". It is "how far did the hole
move", which is measurable from data already on disk, in minutes, and answers three questions at
once.

**The measurement needs no new apparatus.** When the peg is seated, the tool is held laterally by
the hole, so the tool pose at that instant *is* a reading of where the hole was -- which is
exactly how ``terminal_trials`` re-reads its reference from physical contact. Every seated descent
ever logged is therefore a hole reading that was never read as one.

**Scatter and drift are separated because they are different faults.** A steady drift across a
session is fixture creep: the hole is somewhere definite at any moment, and re-reading the
reference every few trials tracks it -- which the terminal loop already does. Scatter that
survives removing the drift is compliance: the hole moved *during* insertions, is not in a
definite place, and no amount of re-reading recovers it. A single spread number pools the two and
recommends the wrong fix for whichever one is really there.

Ordering matters and is not free. ``episode_index`` in a training view is not chronological, so a
drift test run on it is a drift test on an arbitrary permutation -- recover the true order from
``il_view_manifest.json`` and pass it in, or pass timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


# Single-side clearance of this peg and hole, in millimetres. Everything here is judged against it
# because it is the length the task is actually played out over.
NOMINAL_CLEARANCE_MM = 2.5
# The capture radius measured while the fixture was loose, and the interval it carries. Quoted so a
# re-measurement on a bolted fixture can be compared to the number the search pattern was sized
# against rather than to a memory of it.
LOOSE_CAPTURE_RADIUS_MM = 4.2
LOOSE_CAPTURE_RADIUS_CI_MM = (2.5, 8.8)
# The ring E7-C flies, and how many landings it puts on it.
SEARCH_RING_MM = 7.0
SEARCH_POINTS = 8
# How many readings before the drift/wander split is allowed to produce a verdict.
#
# Not a convention. Separating them fits a line per axis, which spends two degrees of freedom, so
# at three readings the residual has one left and "the wander that survives removing the drift" is
# whatever the fit could not absorb -- a number that will happily read as compliance on a fixture
# that never moved. Eight leaves six, which is still thin and is why the report says so.
MIN_READINGS_FOR_VERDICT = 8


class HoleStabilityError(ValueError):
    """Input that cannot be read as a set of hole readings."""


@dataclass(frozen=True)
class Reading:
    """One measurement of where the hole was, and when."""

    order: float
    x: float
    y: float
    source: str = ""


def readings_from_trial_rows(rows: Iterable[dict[str, Any]]) -> list[Reading]:
    """Hole readings out of a `terminal_trials` / `terminal_servo` JSONL log, best source first.

    If the log contains a summary row it carries `referenceUpdates`, and those are strictly better
    than anything that can be reconstructed from the trial rows: a reference trial re-grips the peg
    while it stands in the hole, so its reading has the grasp offset re-zeroed against the hole
    rather than added to it. Preferring them here rather than leaving the choice to the caller is
    deliberate -- the two kinds of reading are not interchangeable and mixing them would pool a
    clean measurement with a confounded one.

    Falling back to seated stops keeps a log without a summary readable. Only seated ones: a
    descent that stopped on the face measures the face, and a slip measures nothing.
    """

    rows = list(rows)
    for row in rows:
        if row.get("kind") == "summary" and row.get("referenceUpdates"):
            return readings_from_reference_updates(row)

    readings: list[Reading] = []
    for index, row in enumerate(rows):
        if row.get("kind") not in (None, "trial"):
            continue
        if row.get("verdict") != "seated" or not row.get("ok", True):
            continue
        xyz = row.get("stoppedAtXyz")
        if not xyz or len(xyz) < 2:
            continue
        readings.append(
            Reading(
                order=float(row.get("elapsedS", index)),
                x=float(xyz[0]),
                y=float(xyz[1]),
                source=f"trial[{row.get('index', index)}]",
            )
        )
    return readings


def readings_from_reference_updates(summary: dict[str, Any]) -> list[Reading]:
    """Hole readings out of a `terminal_trials` run summary, which is where the clean ones are.

    This is the instrument. A reference trial seats the peg with the search on and then re-grips it
    *while it stands in the hole*, so the pose it was re-gripped at is the hole plus whatever bias
    the peg carries in the fingers -- and since the peg is re-gripped at exactly that pose, the
    bias is re-zeroed against the hole itself rather than accumulating. Every other source of hole
    readings in this project carries an unknown grasp offset on top of the hole position and cannot
    separate the two. This one does not.

    `terminal_trials` already records these as `referenceUpdates` and has done since it was
    written. They have never been read as a hole measurement.
    """

    readings: list[Reading] = []
    for update in summary.get("referenceUpdates") or []:
        to_xyz = update.get("toXyz")
        if not to_xyz or len(to_xyz) < 2:
            continue
        readings.append(
            Reading(
                order=float(update.get("elapsedS", update.get("index", len(readings)))),
                x=float(to_xyz[0]),
                y=float(to_xyz[1]),
                source=f"reference[{update.get('index', len(readings))}]",
            )
        )
    return readings


def seating_pose_from_trace(
    steps: Sequence[dict[str, Any]],
    *,
    release_gripper: float = 0.5,
) -> tuple[float, float] | None:
    """Where the tool was when it let the peg go, in a per-step demonstration trace.

    The release is the marker rather than the deepest point, and the difference is the whole
    reliability of the reading: the deepest point of a *failed* insertion is the peg standing on
    the face, which is not a hole reading, while an operator only opens the fingers once the peg is
    in. Searched forward from the deepest point so an open gripper on the way down -- the approach
    -- is never mistaken for the release.
    """

    if not steps:
        return None
    try:
        depths = [float(step["z"]) for step in steps]
    except (KeyError, TypeError, ValueError) as exc:
        raise HoleStabilityError(f"trace step missing a usable z: {exc}") from exc
    deepest = depths.index(min(depths))
    for step in steps[deepest:]:
        try:
            if float(step["gripper_cmd"]) > release_gripper:
                return (float(step["x"]), float(step["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _fit_line(order: list[float], values: list[float]) -> tuple[float, float]:
    """Least squares slope and intercept. Slope is the drift; the residual is the wander."""

    n = len(order)
    mean_o = sum(order) / n
    mean_v = sum(values) / n
    denominator = sum((o - mean_o) ** 2 for o in order)
    if denominator <= 0.0:
        return 0.0, mean_v
    slope = sum((o - mean_o) * (v - mean_v) for o, v in zip(order, values, strict=True)) / denominator
    return slope, mean_v - slope * mean_o


def hole_stability(readings: Sequence[Reading]) -> dict[str, Any]:
    """Spread, drift, and the spread that survives removing the drift.

    The last of the three is the number that decides things. Total spread says how far apart two
    readings can be; residual spread says how far apart they are *after* a per-session reference
    re-read would have corrected for creep, which is the error a terminal controller is actually
    left holding.
    """

    if len(readings) < 3:
        raise HoleStabilityError(
            f"{len(readings)} readings is not a spread. Three is the minimum at which drift and "
            "wander can be told apart at all, and the answer will not be worth acting on until "
            "there are many more."
        )

    order = [reading.order for reading in readings]
    xs = [reading.x for reading in readings]
    ys = [reading.y for reading in readings]
    n = len(readings)

    centre = (sum(xs) / n, sum(ys) / n)
    radial = [1000.0 * math.dist((x, y), centre) for x, y in zip(xs, ys, strict=True)]

    slope_x, intercept_x = _fit_line(order, xs)
    slope_y, intercept_y = _fit_line(order, ys)
    residual_x = [x - (slope_x * o + intercept_x) for o, x in zip(order, xs, strict=True)]
    residual_y = [y - (slope_y * o + intercept_y) for o, y in zip(order, ys, strict=True)]
    residual_radial = [
        1000.0 * math.hypot(rx, ry) for rx, ry in zip(residual_x, residual_y, strict=True)
    ]

    span = (max(order) - min(order)) or 1.0
    drift_mm = 1000.0 * math.hypot(slope_x * span, slope_y * span)

    def stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
            "max": ordered[-1],
        }

    total = stats(radial)
    residual = stats(residual_radial)
    degrees_of_freedom = n - 2
    verdict = (
        classify_stability(total["p95"], residual["p95"], drift_mm)
        if n >= MIN_READINGS_FOR_VERDICT
        else {
            "label": "insufficient",
            "detail": (
                f"{n} readings: the drift fit spends two degrees of freedom per axis, leaving "
                f"{degrees_of_freedom}, which is not enough for the residual to mean anything. The "
                "numbers above are reported so they can be looked at; they are not a verdict, and "
                f"in particular they cannot distinguish a compliant fixture from a fit with nothing "
                "left over. Collect at least "
                f"{MIN_READINGS_FOR_VERDICT} readings -- `terminal_trials` produces one per "
                "reference trial by construction."
            ),
            "searchRingStillCovers": _ring_covers(LOOSE_CAPTURE_RADIUS_MM),
        }
    )
    return {
        "readings": n,
        "degreesOfFreedom": degrees_of_freedom,
        "minReadingsForVerdict": MIN_READINGS_FOR_VERDICT,
        "centreXy": list(centre),
        "orderSpan": span,
        # How far a reading can sit from the mean hole position, before any correction.
        "totalRadialMm": total,
        # How far the hole moved end to end along the fitted line. This is creep, and the terminal
        # loop's periodic reference re-read already tracks it.
        "driftMm": drift_mm,
        # What is left once the creep is removed. This is the part a re-read cannot recover.
        "residualRadialMm": residual,
        "clearanceMm": NOMINAL_CLEARANCE_MM,
        "looseCaptureRadiusMm": LOOSE_CAPTURE_RADIUS_MM,
        "verdict": verdict,
    }


def classify_stability(total_p95_mm: float, residual_p95_mm: float, drift_mm: float) -> dict[str, Any]:
    """Name which of the two faults this is, because they are fixed in different places.

    The threshold is the clearance itself rather than a fraction of it. A residual wander at or
    above the clearance means two insertions aimed identically can miss and hit, which is the
    definition of the correct action not being determined by the observation.
    """

    wandering = residual_p95_mm >= NOMINAL_CLEARANCE_MM
    creeping = drift_mm >= NOMINAL_CLEARANCE_MM and residual_p95_mm < total_p95_mm
    if wandering:
        label = "compliant"
        detail = (
            f"residual wander p95 {residual_p95_mm:.2f} mm is at or above the {NOMINAL_CLEARANCE_MM} mm "
            "clearance after the drift is removed. Re-reading the reference cannot recover this: "
            "the hole was not in a definite place. Two demonstrations that look alike can carry "
            "different correct terminal actions, and the capture radius measured on this fixture "
            f"({LOOSE_CAPTURE_RADIUS_MM} mm) is likely larger than a bolted one will give."
        )
    elif creeping:
        label = "creeping"
        detail = (
            f"the hole moved {drift_mm:.2f} mm end to end but only wanders {residual_p95_mm:.2f} mm "
            "around its own track. That is creep, it is tracked by re-reading the reference from "
            "contact every few trials, and the demonstrations are of one task."
        )
    else:
        label = "stable"
        detail = (
            f"total spread p95 {total_p95_mm:.2f} mm is inside the {NOMINAL_CLEARANCE_MM} mm clearance. "
            "The fixture being loose did not move the hole enough to matter over this set."
        )
    return {
        "label": label,
        "detail": detail,
        # What it means for the pattern E7-C flies, which was sized against the loose number.
        "searchRingStillCovers": _ring_covers(LOOSE_CAPTURE_RADIUS_MM if label != "compliant" else NOMINAL_CLEARANCE_MM),
    }


def _ring_covers(capture_radius_mm: float) -> dict[str, Any]:
    """Whether eight landings on a 7 mm ring leave a gap at this capture radius.

    The gap between adjacent landings is the chord; a point halfway between two landings is
    covered only if half that chord is inside the capture radius. And the centre has to be covered
    too, which it is only if the ring radius itself is inside the capture radius of a landing.
    """

    chord_mm = 2.0 * SEARCH_RING_MM * math.sin(math.pi / SEARCH_POINTS)
    between_ok = chord_mm / 2.0 <= capture_radius_mm
    centre_ok = SEARCH_RING_MM <= capture_radius_mm + capture_radius_mm
    return {
        "captureRadiusMm": capture_radius_mm,
        "ringMm": SEARCH_RING_MM,
        "points": SEARCH_POINTS,
        "gapBetweenLandingsMm": chord_mm,
        "covered": bool(between_ok and centre_ok),
        "note": (
            "covered" if between_ok and centre_ok
            else f"a dead annulus remains: half the {chord_mm:.2f} mm chord is outside a "
                 f"{capture_radius_mm:.2f} mm capture radius"
        ),
    }


def describe(report: dict[str, Any]) -> str:
    total, residual = report["totalRadialMm"], report["residualRadialMm"]
    verdict = report["verdict"]
    ring = verdict["searchRingStillCovers"]
    return "\n".join(
        [
            f"readings={report['readings']} dof={report['degreesOfFreedom']} "
            f"centre=({report['centreXy'][0]:+.4f}, {report['centreXy'][1]:+.4f})",
            f"total_radial_mm p50={total['p50']:.2f} p95={total['p95']:.2f} max={total['max']:.2f}",
            f"drift_mm={report['driftMm']:.2f} (end to end along the fitted line)",
            f"residual_radial_mm p50={residual['p50']:.2f} p95={residual['p95']:.2f} max={residual['max']:.2f} "
            f"(what a reference re-read cannot recover)",
            f"clearance_mm={report['clearanceMm']}  loose_capture_radius_mm={report['looseCaptureRadiusMm']}",
            f"VERDICT: {verdict['label']} -- {verdict['detail']}",
            f"search ring: {ring['note']} (ring {ring['ringMm']} mm, {ring['points']} points, "
            f"gap {ring['gapBetweenLandingsMm']:.2f} mm, capture {ring['captureRadiusMm']:.2f} mm)",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference-updates",
        default="",
        help=(
            "A terminal_trials summary JSON. Its referenceUpdates are the only hole readings on "
            "this rig with the grasp offset re-zeroed against the hole itself -- prefer them."
        ),
    )
    parser.add_argument(
        "--trials",
        default="",
        help="A terminal_trials / terminal_servo JSONL log. Seated stops become hole readings.",
    )
    parser.add_argument(
        "--traces",
        nargs="*",
        default=[],
        help="Per-step demonstration or rollout traces (CSV with x,y,z,gripper_cmd).",
    )
    parser.add_argument(
        "--order",
        default="",
        help=(
            "JSON list giving the true chronological order of --traces. Required for a drift test "
            "on a training view, whose episode_index is not chronological -- recover it from "
            "il_view_manifest.json."
        ),
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    readings: list[Reading] = []
    if args.reference_updates:
        # A summary JSON, or a JSONL whose summary row holds them.
        readings.extend(
            readings_from_reference_updates(
                json.loads(Path(args.reference_updates).read_text(encoding="utf-8"))
            )
        )
    if args.trials:
        with Path(args.trials).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        readings.extend(readings_from_trial_rows(rows))
    if args.traces:
        order = json.loads(Path(args.order).read_text(encoding="utf-8")) if args.order else None
        if order is not None and len(order) != len(args.traces):
            raise HoleStabilityError(
                f"--order has {len(order)} entries for {len(args.traces)} traces. A drift test on a "
                "mismatched order is a drift test on an arbitrary permutation."
            )
        for index, path in enumerate(args.traces):
            with Path(path).open(encoding="utf-8") as handle:
                steps = list(csv.DictReader(handle))
            pose = seating_pose_from_trace(steps)
            if pose is None:
                print(f"[WARN] {path}: no release after the deepest point; not a hole reading", flush=True)
                continue
            readings.append(
                Reading(
                    order=float(order[index]) if order is not None else float(index),
                    x=pose[0],
                    y=pose[1],
                    source=Path(path).name,
                )
            )

    report = hole_stability(readings)
    print(describe(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["verdict"]["label"] not in {"compliant", "insufficient"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
