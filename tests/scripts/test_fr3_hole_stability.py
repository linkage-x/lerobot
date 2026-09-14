"""Did the hole wander, or did it creep? The two look alike in one number and are fixed differently.

Creep is tracked: the hole is somewhere definite at any instant, and re-reading the reference from
contact every few trials follows it, which the terminal loop already does. Wander is not: the hole
moved during insertions, so two identical aims can miss and hit, and nothing recovers it. A spread
statistic that pools them recommends the wrong fix for whichever one is really present, which is
why every test here is about keeping them apart.
"""

import math

import pytest

from tools.fr3.hole_stability import (
    NOMINAL_CLEARANCE_MM,
    HoleStabilityError,
    Reading,
    describe,
    hole_stability,
    readings_from_trial_rows,
    seating_pose_from_trace,
)


def _readings(offsets, *, x0=0.36, y0=-0.14):
    return [Reading(order=float(i), x=x0 + dx / 1000.0, y=y0 + dy / 1000.0) for i, (dx, dy) in enumerate(offsets)]


def test_a_steady_creep_is_named_creep_and_not_compliance():
    """It moved 8 mm end to end and never wandered off its own track: a reference re-read tracks it."""

    report = hole_stability(_readings([(i * 0.4, 0.0) for i in range(21)]))
    assert report["driftMm"] == pytest.approx(8.0, abs=0.1)
    assert report["residualRadialMm"]["p95"] < 0.01
    assert report["verdict"]["label"] == "creeping"
    assert "re-reading the reference" in report["verdict"]["detail"]


def test_wander_that_survives_removing_the_drift_is_named_compliance():
    """This is the one that cannot be corrected, and the one that makes demos several tasks."""

    offsets = [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)] * 5
    report = hole_stability(_readings(offsets))
    assert report["driftMm"] < 1.0, "there is no trend here, only scatter"
    assert report["residualRadialMm"]["p95"] >= NOMINAL_CLEARANCE_MM
    assert report["verdict"]["label"] == "compliant"


def test_a_hole_that_barely_moved_is_not_reported_as_a_problem():
    offsets = [(0.3, 0.0), (-0.2, 0.1), (0.0, 0.3), (0.1, -0.2)] * 5
    report = hole_stability(_readings(offsets))
    assert report["verdict"]["label"] == "stable"


def test_creep_and_wander_together_are_separated_rather_than_pooled():
    """The whole point: a big total spread does not say which fault produced it."""

    offsets = [(i * 0.5 + (3.0 if i % 2 else -3.0), 0.0) for i in range(21)]
    report = hole_stability(_readings(offsets))
    assert report["totalRadialMm"]["p95"] > 5.0, "pooled, this looks alarming"
    assert report["driftMm"] == pytest.approx(10.0, abs=0.5)
    assert report["residualRadialMm"]["p95"] >= NOMINAL_CLEARANCE_MM
    assert report["verdict"]["label"] == "compliant", "the wander is still there under the creep"


def test_a_compliant_fixture_puts_the_search_ring_back_in_question():
    """The ring was sized against a capture radius a loose fixture may have inflated."""

    loose = hole_stability(_readings([(0.3, 0.0), (-0.2, 0.1), (0.0, 0.3), (0.1, -0.2)] * 5))
    assert loose["verdict"]["searchRingStillCovers"]["covered"] is True

    compliant = hole_stability(_readings([(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)] * 5))
    ring = compliant["verdict"]["searchRingStillCovers"]
    assert ring["captureRadiusMm"] == pytest.approx(NOMINAL_CLEARANCE_MM)
    assert ring["covered"] is False
    assert "dead annulus" in ring["note"]
    # The geometry that decides it: half the chord between adjacent landings.
    assert ring["gapBetweenLandingsMm"] == pytest.approx(2 * 7.0 * math.sin(math.pi / 8), abs=0.01)


def test_two_readings_are_not_a_spread():
    with pytest.raises(HoleStabilityError, match="not a spread"):
        hole_stability(_readings([(0.0, 0.0), (1.0, 0.0)]))


# -- turning logs that already exist into hole readings ---------------------------------------


def test_only_seated_stops_are_hole_readings():
    """A stop on the face measures the face; a slip measures nothing."""

    rows = [
        {"kind": "trial", "ok": True, "index": 0, "verdict": "seated", "stoppedAtXyz": [0.36, -0.14, 0.052]},
        {"kind": "trial", "ok": True, "index": 1, "verdict": "standing", "stoppedAtXyz": [0.40, -0.10, 0.057]},
        {"kind": "trial", "ok": True, "index": 2, "verdict": "slip", "stoppedAtXyz": [0.41, -0.09, 0.053]},
        {"kind": "trial", "ok": False, "index": 3, "error": "servo failed"},
        {"kind": "grasp", "stage": "regrip"},
        {"kind": "trial", "ok": True, "index": 4, "verdict": "seated", "stoppedAtXyz": [0.3602, -0.1398, 0.052]},
    ]
    readings = readings_from_trial_rows(rows)
    assert [reading.source for reading in readings] == ["trial[0]", "trial[4]"]


def test_the_release_is_the_marker_and_not_the_deepest_point():
    """The deepest point of a failed insertion is the peg standing on the face."""

    steps = [
        {"x": 0.30, "y": -0.10, "z": 0.20, "gripper_cmd": 1.0},   # approach, fingers open
        {"x": 0.36, "y": -0.14, "z": 0.052, "gripper_cmd": 0.0},  # deepest, still holding
        {"x": 0.361, "y": -0.141, "z": 0.053, "gripper_cmd": 0.0},
        {"x": 0.3612, "y": -0.1412, "z": 0.055, "gripper_cmd": 1.0},  # released here
    ]
    assert seating_pose_from_trace(steps) == pytest.approx((0.3612, -0.1412))


def test_a_descent_that_never_released_is_not_a_reading():
    steps = [
        {"x": 0.30, "y": -0.10, "z": 0.20, "gripper_cmd": 1.0},
        {"x": 0.36, "y": -0.14, "z": 0.052, "gripper_cmd": 0.0},
    ]
    assert seating_pose_from_trace(steps) is None


def test_the_report_names_the_clearance_it_is_judging_against():
    text = describe(hole_stability(_readings([(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)] * 5)))
    assert "clearance_mm=2.5" in text and "VERDICT: compliant" in text
    assert "residual_radial_mm" in text and "drift_mm=" in text


def test_three_readings_do_not_get_a_verdict_because_the_fit_has_nothing_left_over():
    """Two degrees of freedom per axis go into the drift fit. At n=3 the residual is an artefact.

    This is not caution for its own sake: run on the three usable readings in this project's entire
    175-rollout history, the tool cheerfully returned "compliant", which would have been read as
    evidence that the fixture moved during insertions when it is equally consistent with a fit that
    simply could not absorb three points.
    """

    report = hole_stability(_readings([(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0)]))
    assert report["readings"] == 3 and report["degreesOfFreedom"] == 1
    assert report["verdict"]["label"] == "insufficient"
    assert "degrees of freedom" in report["verdict"]["detail"]
    # The numbers are still reported -- they are just not a conclusion.
    assert report["residualRadialMm"]["p95"] > 0.0


def test_reference_updates_are_the_clean_readings_and_are_already_logged():
    """`terminal_trials` has recorded these since it was written; nobody read them as hole positions."""

    from tools.fr3.hole_stability import readings_from_reference_updates

    summary = {
        "referenceUpdates": [
            {"index": 0, "fromXyz": [0.3599, -0.1333, 0.052], "toXyz": [0.3601, -0.1331, 0.052],
             "stepMm": 0.28, "elapsedS": 12.0},
            {"index": 9, "fromXyz": [0.3601, -0.1331, 0.052], "toXyz": [0.3607, -0.1329, 0.052],
             "stepMm": 0.63, "elapsedS": 240.0},
        ]
    }
    readings = readings_from_reference_updates(summary)
    assert [reading.source for reading in readings] == ["reference[0]", "reference[9]"]
    assert readings[1].order == 240.0 and readings[1].x == pytest.approx(0.3607)


def test_a_trials_log_with_a_summary_uses_the_clean_readings_and_not_the_confounded_ones():
    """The two kinds are not interchangeable, so the choice is not left to the caller."""

    rows = [
        {"kind": "trial", "ok": True, "index": 0, "verdict": "seated", "stoppedAtXyz": [0.40, -0.10, 0.05]},
        {"kind": "trial", "ok": True, "index": 1, "verdict": "seated", "stoppedAtXyz": [0.41, -0.11, 0.05]},
        {"kind": "summary", "haltedOn": "schedule_complete", "referenceUpdates": [
            {"index": 0, "toXyz": [0.3601, -0.1331, 0.052], "stepMm": 0.3, "elapsedS": 10.0},
            {"index": 9, "toXyz": [0.3604, -0.1330, 0.052], "stepMm": 0.4, "elapsedS": 200.0},
        ]},
    ]
    readings = readings_from_trial_rows(rows)
    assert [reading.source for reading in readings] == ["reference[0]", "reference[9]"]
    assert readings[0].x == pytest.approx(0.3601), "the seated stops were used instead"


def test_a_trials_log_without_a_summary_still_yields_what_it_can():
    rows = [
        {"kind": "trial", "ok": True, "index": 0, "verdict": "seated", "stoppedAtXyz": [0.40, -0.10, 0.05]},
        {"kind": "summary", "haltedOn": "grasp_empty"},
    ]
    assert [r.source for r in readings_from_trial_rows(rows)] == ["trial[0]"]
