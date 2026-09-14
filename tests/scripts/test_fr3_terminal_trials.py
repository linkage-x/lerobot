"""E6-lite: does the loop close, and does it stop when it can no longer read its own state?

Two kinds of thing are worth testing here. The schedule decides what the run can conclude -- a
sweep that is monotone, or aimed down one bearing, produces rows that cannot separate the offset
from the afternoon -- and it is fixed before the arm moves, so it can be checked exactly. The
loop itself is judged on the one property that makes an unattended run worth starting: every
state it cannot interpret ends the run rather than producing another row.

The rig fake is the servo suite's hole with a gripper added. That is the whole physics the loop
depends on: a peg goes in within the capture radius and stands on the face outside it, and the
fingers hold a peg only if they closed where the peg was left.
"""

import math

import pytest

import tools.fr3.scene_reset as scene_reset
import tools.fr3.terminal_servo as terminal_servo
import tools.fr3.terminal_trials as terminal_trials
from tools.fr3.terminal_servo import TerminalServoRequest
from tools.fr3.terminal_trials import (
    TerminalTrialError,
    assert_tool_is_level,
    resume_state,
    tool_axis_tilt_deg,
    TerminalTrialsRequest,
    build_trial_schedule,
    classify_stop,
    run_terminal_trials,
    summarize_by_offset,
    validate_terminal_trials,
)

from tests.scripts.test_fr3_terminal_servo import FakeHoleRobot


SEATED = (0.3599, -0.1333, 0.0523)
PICK = (0.3640, -0.1370, 0.0550)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(terminal_servo, "precise_sleep", lambda seconds: None)


class FakeTrialRig(FakeHoleRobot):
    """The servo suite's fixture, plus a peg that can be let go of and picked back up.

    The gripper is modelled on what this rig actually reports: a clamped peg reads its own
    thickness rather than the closed command, an empty close reads nothing, and an open command
    is the only thing that puts the peg down. That last rule matters -- the servo commands the
    *measured* width while it carries the peg, and a fake that treated any non-zero command as
    "let go" would drop the peg on the first step of every descent.
    """

    PEG_WIDTH = 0.31

    def __init__(self, hole_xy, *, capture_m=0.0042, peg_xyz=PICK, held=False, face_z=SEATED[2] + 0.005):
        super().__init__(hole_xy, capture_m=capture_m, face_z=face_z)
        self.peg_xyz = tuple(float(value) for value in peg_xyz)
        self.held = bool(held)
        self.grasp_reach_m = 0.005
        self.gripper = 1.0
        self.closes = []

    def _fingers_on_peg(self):
        return (
            math.hypot(self.xyz[0] - self.peg_xyz[0], self.xyz[1] - self.peg_xyz[1]) <= self.grasp_reach_m
            and abs(self.xyz[2] - self.peg_xyz[2]) <= self.grasp_reach_m
        )

    def send_action(self, action):
        result = super().send_action(action)
        commanded = float(action["gripper.pos"])
        if commanded >= 0.9:
            if self.held:
                self.held = False
                self.peg_xyz = tuple(self.xyz)
            self.gripper = commanded
        elif commanded <= 0.05:
            if not self.held and self._fingers_on_peg():
                self.held = True
            self.closes.append((tuple(self.xyz), self.held))
            self.gripper = self.PEG_WIDTH if self.held else 0.0
        else:
            self.gripper = self.PEG_WIDTH if self.held else commanded
        return result


def _servo(**overrides):
    fields = dict(
        xyz=SEATED,
        handoffZ=0.12,
        controlPeriodS=0.01,
        timeoutS=2.0,
        settleS=0.01,
        openSettleS=0.0,
    )
    fields.update(overrides)
    return TerminalServoRequest(**fields)


def _request(**overrides):
    fields = dict(
        servo=_servo(),
        offsetsMm=(0.0, 8.0),
        repeats=1,
        controlEvery=4,
        seed=0,
        pickXyz=PICK,
        graspSettleS=0.01,
        controlPeriodS=0.01,
        timeoutS=2.0,
    )
    fields.update(overrides)
    return TerminalTrialsRequest(**fields)


# --- the schedule -----------------------------------------------------------------------------


def test_the_sweep_is_shuffled_so_a_drifting_fixture_cannot_pose_as_a_capture_radius():
    request = _request(offsetsMm=(0.0, 2.0, 4.0, 6.0, 8.0), repeats=4, controlEvery=100)
    offsets = [spec.offsetMm for spec in build_trial_schedule(request) if spec.kind == "offset"]
    assert sorted(offsets) == sorted([value for value in request.offsetsMm for _ in range(4)])
    assert offsets != sorted(offsets)


def test_each_offset_lands_on_its_own_bearing_so_anisotropy_stays_testable():
    request = _request(offsetsMm=(6.0,), repeats=8, controlEvery=100)
    bearings = [spec.bearingDeg for spec in build_trial_schedule(request) if spec.kind == "offset"]
    assert len(set(bearings)) == len(bearings)
    assert max(bearings) - min(bearings) > 90.0


def test_the_same_seed_plans_the_same_run():
    assert build_trial_schedule(_request(seed=5)) == build_trial_schedule(_request(seed=5))
    assert build_trial_schedule(_request(seed=5)) != build_trial_schedule(_request(seed=6))


def test_a_reference_trial_comes_first_and_then_every_control_interval():
    specs = build_trial_schedule(_request(offsetsMm=(4.0,), repeats=9, controlEvery=3))
    kinds = [spec.kind for spec in specs]
    assert kinds[0] == "reference"
    gaps = [index for index, kind in enumerate(kinds) if kind == "reference"]
    assert gaps == [0, 4, 8]


def test_reference_trials_carry_the_search_and_offset_trials_carry_what_was_asked_for():
    specs = build_trial_schedule(_request(searchRingM=0.0, referenceRingM=0.007))
    assert {spec.searchRingM for spec in specs if spec.kind == "reference"} == {0.007}
    assert {spec.searchRingM for spec in specs if spec.kind == "offset"} == {0.0}


# --- reading one descent ----------------------------------------------------------------------


def test_creep_is_read_before_depth_so_a_slipped_peg_is_not_an_insertion():
    # 1.4 mm above target is squarely inside the seated cluster; 18 s of settle is what said
    # otherwise on 2026-09-10.
    assert classify_stop(1.4, 4.0, seated_mm=3.0, slip_mm=3.0, standing_mm=4.9) == "slip"
    assert classify_stop(1.4, 0.1, seated_mm=3.0, slip_mm=3.0, standing_mm=4.9) == "seated"


def test_a_stop_between_the_two_measured_clusters_is_reported_rather_than_assigned():
    assert classify_stop(4.0, 0.1, seated_mm=3.0, slip_mm=3.0, standing_mm=4.9) == "ambiguous"
    assert classify_stop(5.2, 0.1, seated_mm=3.0, slip_mm=3.0, standing_mm=4.9) == "standing"


def test_the_two_clusters_may_not_be_asserted_to_touch():
    with pytest.raises(TerminalTrialError):
        validate_terminal_trials(_request(standingMm=2.0), build_trial_schedule(_request()))


# --- refusing a run before it starts ------------------------------------------------------------


def test_an_offset_that_leaves_the_fence_is_refused_before_the_first_trial():
    request = _request(offsetsMm=(0.0, 40.0), repeats=1)
    with pytest.raises(Exception):
        validate_terminal_trials(
            request,
            build_trial_schedule(request),
            # Tight enough that no bearing survives: a 40 mm offset needs both axes inside
            # 10 mm of the aim, which no direction satisfies.
            workspace_min=(0.350, -0.145, 0.0),
            workspace_max=(0.370, -0.125, 0.30),
        )


def test_a_reference_trial_without_a_search_is_refused():
    request = _request(referenceRingM=0.0)
    with pytest.raises(TerminalTrialError):
        validate_terminal_trials(request, build_trial_schedule(request))


def test_the_plan_reports_how_far_the_run_can_reach():
    request = _request(offsetsMm=(0.0, 8.0), repeats=1, searchRingM=0.007)
    qc = validate_terminal_trials(request, build_trial_schedule(request))
    assert qc["widestOffsetMm"] == 8.0
    assert qc["widestCommandedMm"] == pytest.approx(15.0)


# --- the loop ------------------------------------------------------------------------------------


def test_the_loop_closes_on_itself_and_needs_nobody_between_trials():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=3, controlEvery=100)
    summary = run_terminal_trials(robot, request)
    assert summary["haltedOn"] == "schedule_complete"
    assert summary["trials"] == 4  # one reference plus three offsets
    assert summary["seated"] == 4
    assert robot.held is True


def test_the_run_answers_p_seat_against_offset_rather_than_one_more_success_rate():
    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    request = _request(offsetsMm=(2.0, 8.0), repeats=3, controlEvery=100)
    summary = run_terminal_trials(robot, request)
    buckets = {bucket["offsetMm"]: bucket for bucket in summary["byOffsetMm"]}
    assert buckets[2.0]["seatedFraction"] == 1.0
    assert buckets[8.0]["seatedFraction"] == 0.0


def test_reference_trials_are_left_out_of_the_curve_they_are_the_control_for():
    rows = [
        {"kind": "trial", "trialKind": "reference", "offsetMm": 0.0, "verdict": "seated"},
        {"kind": "trial", "trialKind": "offset", "offsetMm": 0.0, "verdict": "standing"},
    ]
    assert summarize_by_offset(rows) == [
        {
            "offsetMm": 0.0,
            "n": 1,
            "seated": 0,
            "standing": 1,
            "slip": 0,
            "ambiguous": 0,
            "failed": 0,
            "decided": 1,
            "seatedFraction": 0.0,
        }
    ]


def test_a_hole_that_crept_is_re_read_from_contact_rather_than_trusted_from_the_command_line():
    """The estimate is where the demonstrations said the hole was; the hole has since moved 5 mm.

    Five is past the capture radius, so every trial aimed at the stale estimate misses and the
    run would report a capture radius of zero. What saves it is that the reference trial carries
    the search: it finds the hole from a landing on the ring, and the estimate is re-read from
    where that seating happened.

    The fake does not model the lateral hold a seated peg puts on the tool, so the estimate here
    lands on the ring landing rather than on the hole centre. That is the conservative case --
    on the arm the tool is dragged the rest of the way -- and the property that matters holds
    either way: after the reference, zero-offset trials are inside the capture radius again.
    """

    crept = (SEATED[0] + 0.005, SEATED[1])
    robot = FakeTrialRig(hole_xy=crept)
    request = _request(offsetsMm=(0.0,), repeats=2, controlEvery=1)
    summary = run_terminal_trials(robot, request)
    assert summary["referenceUpdates"], "the reference trial never moved the estimate"
    assert (
        math.hypot(summary["referenceXyz"][0] - crept[0], summary["referenceXyz"][1] - crept[1])
        <= robot.capture_m
    )
    offsets = [row for row in summary["rows"] if row.get("trialKind") == "offset"]
    assert offsets and all(row["verdict"] == "seated" for row in offsets)


def test_a_correction_bigger_than_the_search_could_have_found_stops_the_run():
    robot = FakeTrialRig(hole_xy=(SEATED[0] + 0.005, SEATED[1]))
    request = _request(offsetsMm=(0.0,), repeats=1, maxReferenceStepM=0.0001)
    summary = run_terminal_trials(robot, request)
    assert summary["haltedOn"] == "reference_step_too_large"
    assert summary["ok"] is False


def test_fingers_that_close_on_nothing_end_the_run_instead_of_filling_the_night_with_rows():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=4, controlEvery=100)

    original = robot.send_action
    state = {"trials": 0}

    def knock_the_peg_over(action):
        result = original(action)
        # As soon as the peg has been put down for the second time, move it out of reach: this
        # is a peg that toppled after release, which is the failure an unattended cycle has to
        # notice rather than re-run ninety times with an empty hand.
        if not robot.held and float(action["gripper.pos"]) >= 0.9:
            state["trials"] += 1
            if state["trials"] == 2:
                robot.peg_xyz = (robot.peg_xyz[0] + 0.05, robot.peg_xyz[1], robot.peg_xyz[2])
        return result

    robot.send_action = knock_the_peg_over
    summary = run_terminal_trials(robot, request)
    assert summary["haltedOn"] == "grasp_empty"
    assert summary["trials"] < 5


def test_a_reference_that_cannot_find_the_hole_is_retried_once_and_then_ends_the_run():
    # 30 mm away is outside anything the 7 mm ring can reach, so the search exhausts.
    robot = FakeTrialRig(hole_xy=(SEATED[0] + 0.030, SEATED[1]))
    request = _request(offsetsMm=(0.0,), repeats=4, controlEvery=100)
    summary = run_terminal_trials(robot, request)
    assert summary["haltedOn"] == "reference_lost"
    references = [row for row in summary["rows"] if row.get("trialKind") == "reference"]
    assert len(references) == 2
    assert all(row["verdict"] != "seated" for row in references)


def test_the_run_stops_when_it_is_out_of_time_rather_than_mid_descent():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=6, controlEvery=100, maxSeconds=1e-9)
    summary = run_terminal_trials(robot, request)
    assert summary["haltedOn"] == "time_budget"
    assert summary["ok"] is True
    assert summary["trials"] == 0


def test_the_arm_is_left_holding_the_peg_at_carrying_height_when_a_run_stops():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=1, controlEvery=100)
    summary = run_terminal_trials(robot, request)
    assert summary["parked"] is True
    assert robot.held is True
    assert robot.xyz[2] >= SEATED[2] + request.servo.retreatM - 1e-6


def test_a_servo_that_refuses_its_own_qc_ends_the_run_with_the_reason_on_the_row():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=2, controlEvery=100)

    calls = {"n": 0}
    real = terminal_trials.execute_terminal_servo

    def fail_on_the_second(robot_arg, servo_request):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"ok": False, "error": "trajectory_qc_failed: contrived"}
        return real(robot_arg, servo_request)

    terminal_trials.execute_terminal_servo = fail_on_the_second
    try:
        summary = run_terminal_trials(robot, request)
    finally:
        terminal_trials.execute_terminal_servo = real
    assert summary["haltedOn"] == "servo_failed"
    assert summary["rows"][-1]["error"] == "trajectory_qc_failed: contrived"


def test_every_row_is_handed_over_as_it_happens_rather_than_at_the_end():
    robot = FakeTrialRig(hole_xy=SEATED[:2])
    request = _request(offsetsMm=(0.0,), repeats=2, controlEvery=100)
    seen = []
    summary = run_terminal_trials(robot, request, on_row=seen.append)
    assert [row["kind"] for row in seen].count("trial") == summary["trials"]
    assert seen[-1]["kind"] == "summary"
    assert seen[0]["kind"] == "grasp"


def test_the_brake_stops_between_trials_so_the_loop_ends_holding_the_peg():
    """Stopped mid-descent, the peg is somewhere the next trial cannot re-grip it from -- and the
    loop closing at all depends on every trial ending with the peg in the fingers."""

    robot = FakeTrialRig(SEATED[:2])
    request = _request(offsetsMm=(0.0, 4.0), repeats=3, controlEvery=8)
    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] > 2

    summary = run_terminal_trials(robot, request, should_stop=should_stop)
    assert summary["haltedOn"] == "stop_requested"
    assert summary["trials"] == 2
    # A deliberate stop is not a fault, and the peg is still held.
    assert summary["ok"] is True
    assert robot.held is True


def test_a_trial_that_failed_before_a_verdict_is_not_arithmetic_but_is_not_lost_either():
    """Hit for real on 2026-09-11: a failed offset trial crashed the summary that reads it.

    The row a failure writes has no verdict, because nothing was classified. Putting it in `n`
    would give it a vote in a seated fraction it never answered; dropping it would hide that the
    offset is short of its scheduled repeats. It gets its own column, and the summary survives.
    """

    rows = [
        {"kind": "trial", "trialKind": "reference", "ok": True, "offsetMm": 0.0, "verdict": "seated"},
        {"kind": "trial", "trialKind": "offset", "ok": True, "offsetMm": 4.0, "verdict": "seated"},
        {"kind": "trial", "trialKind": "offset", "ok": True, "offsetMm": 4.0, "verdict": "standing"},
        {"kind": "trial", "trialKind": "offset", "ok": False, "offsetMm": 4.0,
         "error": "scene reset step align_above_target timed out"},
    ]
    buckets = summarize_by_offset(rows)
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket["offsetMm"] == 4.0
    assert bucket["n"] == 2 and bucket["failed"] == 1
    assert bucket["seated"] == 1 and bucket["standing"] == 1
    assert bucket["decided"] == 2 and bucket["seatedFraction"] == 0.5


def test_a_failure_row_with_no_offset_at_all_does_not_crash_the_summary():
    """Older rows, written before the offset was attached to a failure, must still be readable."""

    rows = [{"kind": "trial", "trialKind": "offset", "ok": False, "error": "timed out"}]
    assert summarize_by_offset(rows) == []


# --- the tool has to be level, and the run has to say so ----------------------


def _rotvec_leaning(degrees: float) -> tuple[float, float, float]:
    """A tool pointing down, leaned by `degrees`: a rotation of pi - theta about x."""

    return (math.pi - math.radians(degrees), 0.0, 0.0)


def test_a_tool_pointing_straight_down_reads_as_level_whichever_way_it_points():
    assert tool_axis_tilt_deg(_rotvec_leaning(0.0)) == pytest.approx(0.0, abs=1e-9)
    # Pointing *up* is just as level. The question is how far the axis leans, not which end of
    # it the peg is on, and the loop's tool points down.
    assert tool_axis_tilt_deg((0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)


def test_a_leaning_tool_reads_its_lean():
    assert tool_axis_tilt_deg(_rotvec_leaning(15.14)) == pytest.approx(15.14, abs=1e-6)


def test_a_leaning_run_is_refused_with_the_depth_it_would_have_wedged_at():
    """Measured 2026-09-11: the arm sat at 15.1 deg and five trials read as seated anyway.

    At that angle a peg wedges around 18 mm into a 2.5 mm clearance hole, so `above_target`
    stopped 2-3 mm short every time -- just inside the 3 mm threshold. The run looked healthy
    and was measuring where the peg jammed.
    """

    request = _request(maxTiltDeg=2.0)
    with pytest.raises(TerminalTrialError) as caught:
        assert_tool_is_level(_rotvec_leaning(15.14), request)
    message = str(caught.value)
    assert "15.14 deg" in message and "18 mm" in message
    assert "Home the arm" in message, "a refusal has to name the remedy"


def test_a_tilt_check_can_be_turned_off_to_measure_the_tilt_deliberately():
    assert assert_tool_is_level(_rotvec_leaning(15.14), _request(maxTiltDeg=0.0)) == pytest.approx(15.14)


# --- a grasp gets more than one try, but not for the failure that needs eyes ---


def test_a_grasp_that_takes_on_the_second_try_does_not_end_the_run(monkeypatch):
    widths = iter([0.02, 0.31])
    monkeypatch.setattr(terminal_trials, "_close_and_lift", lambda *a, **k: next(widths))
    width, verdict, attempts = terminal_trials._grasp_until_held(
        object(), _request(graspAttempts=3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
    )
    assert verdict == "held" and attempts == 2 and width == pytest.approx(0.31)


def test_a_peg_that_is_no_longer_there_is_still_a_halt_after_the_retries(monkeypatch):
    """No number of closes at the same pose finds a peg that fell over somewhere else."""

    calls = []
    monkeypatch.setattr(terminal_trials, "_close_and_lift",
                        lambda *a, **k: (calls.append(1), 0.02)[1])
    _width, verdict, attempts = terminal_trials._grasp_until_held(
        object(), _request(graspAttempts=3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
    )
    assert verdict == "empty" and attempts == 3 and len(calls) == 3


def test_one_attempt_is_the_default_so_nothing_retries_that_did_not_before(monkeypatch):
    calls = []
    monkeypatch.setattr(terminal_trials, "_close_and_lift",
                        lambda *a, **k: (calls.append(1), 0.02)[1])
    terminal_trials._grasp_until_held(object(), _request(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None)
    assert len(calls) == 1


# --- resuming an interrupted night --------------------------------------------


def test_a_resumed_run_skips_what_was_answered_and_keeps_the_hole_it_paid_for():
    rows = [
        {"kind": "trial", "index": 0, "ok": True, "verdict": "seated", "referenceAccepted": True,
         "referenceXyz": [0.36, -0.133, 0.052]},
        {"kind": "trial", "index": 1, "ok": True, "verdict": "standing"},
        {"kind": "summary", "haltedOn": "grasp_empty", "referenceXyz": [0.3605, -0.1332, 0.052]},
    ]
    done, reference = resume_state(rows)
    assert done == {0, 1}
    assert reference == pytest.approx((0.3605, -0.1332, 0.052))


def test_a_trial_that_answered_nothing_is_run_again_rather_than_counted_as_done():
    """It is the trial that stopped the run, so treating it as done shrinks the schedule."""

    rows = [
        {"kind": "trial", "index": 0, "ok": True, "verdict": "seated"},
        {"kind": "trial", "index": 1, "ok": False, "error": "align_above_target timed out"},
    ]
    done, _reference = resume_state(rows)
    assert done == {0}


def test_a_night_with_no_reference_row_resumes_on_the_nominal_pose_rather_than_inventing_one():
    done, reference = resume_state([{"kind": "trial", "index": 3, "ok": True, "verdict": "seated"}])
    assert done == {3} and reference is None


# --- the reference trial re-reads the hole's height too -----------------------


def test_a_reference_trial_corrects_the_holes_height_and_not_only_its_xy():
    """Measured 2026-09-11: seven seated trials read 1.50-2.03 mm "above target".

    Half a millimetre of spread on a two millimetre offset is the estimate being wrong, not the
    peg landing differently. Carrying the typed-in z forward kept that bias for the whole run and
    spent two of the three millimetres the seated threshold has to work with.
    """

    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    summary = run_terminal_trials(robot, _request(offsetsMm=(0.0,), repeats=1, controlEvery=1))
    updates = summary["referenceUpdates"]
    assert updates, "a seated reference trial is what re-reads the hole"
    before, after = updates[0]["fromXyz"], updates[0]["toXyz"]
    seated_rows = [row for row in summary["rows"]
                   if row.get("kind") == "trial" and row.get("verdict") == "seated"]
    assert seated_rows
    assert after[2] == pytest.approx(seated_rows[0]["stoppedAtXyz"][2])
    assert "referenceStepZMm" in seated_rows[0]
    assert abs(after[2] - before[2]) == pytest.approx(
        seated_rows[0]["referenceStepZMm"] / 1000.0, abs=1e-9
    )


class FakeFlooredTrialRig(FakeTrialRig):
    """A fixture whose bottom is where it is, not where the request thought it was.

    The plain rig lets the peg follow the commanded z down, so aiming low just lands low and no
    estimate is ever wrong. The rig this models had a bottom two millimetres above the number it
    was given, which is the whole reason the reference trial has to re-read the height.
    """

    def __init__(self, *args, floor_z, **kwargs):
        super().__init__(*args, **kwargs)
        self.floor_z = float(floor_z)

    def send_action(self, action):
        clamped = dict(action)
        clamped["ee.z"] = max(float(action["ee.z"]), self.floor_z)
        return super().send_action(clamped)


def test_a_height_correction_larger_than_the_hole_could_have_moved_is_refused():
    """The same bound the lateral step gets: a large vertical step is not a correction either."""

    low = _servo(xyz=(SEATED[0], SEATED[1], SEATED[2] - 0.002))
    robot = FakeFlooredTrialRig(hole_xy=SEATED[:2], capture_m=0.0042, floor_z=SEATED[2])
    summary = run_terminal_trials(
        robot, _request(servo=low, offsetsMm=(0.0,), repeats=1, controlEvery=1,
                        maxReferenceStepM=0.001)
    )
    assert summary["haltedOn"] == "reference_step_too_large"


def test_that_same_two_millimetre_correction_is_accepted_under_the_normal_bound():
    low = _servo(xyz=(SEATED[0], SEATED[1], SEATED[2] - 0.002))
    robot = FakeFlooredTrialRig(hole_xy=SEATED[:2], capture_m=0.0042, floor_z=SEATED[2])
    summary = run_terminal_trials(
        robot, _request(servo=low, offsetsMm=(0.0,), repeats=1, controlEvery=1)
    )
    updates = summary["referenceUpdates"]
    assert updates, summary["haltedOn"]
    assert updates[0]["toXyz"][2] - updates[0]["fromXyz"][2] == pytest.approx(0.002, abs=3e-4)


# --- the peg is never left standing ------------------------------------------


def _step_names(captured: str) -> list[str]:
    return [
        line.split("name=")[1].split()[0]
        for line in captured.splitlines()
        if "scene_reset_step=start" in line
    ]


def test_regripping_in_place_closes_the_fingers_before_the_retreat(capsys):
    """A peg that did not seat stands off-centre; left alone through the retreat it falls.

    That is not an edge case -- it is every trial outside the capture radius, which is the half
    of the sweep the sweep exists to produce. Three runs on 2026-09-11 ended that way.
    """

    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    summary = run_terminal_trials(robot, _request(regripInPlace=True, controlEvery=1))
    assert summary["haltedOn"] == "schedule_complete", summary["haltedOn"]
    names = _step_names(capsys.readouterr().out)
    assert "regrip_after_release" in names
    # And the loop does not descend again for a peg it is already holding: the only fetch is the
    # one that starts the run.
    assert names.count("descend_8cm_to_pick") == 1
    assert names.index("descend_8cm_to_pick") < names.index("regrip_after_release")


def test_the_older_cycle_is_unchanged_when_regripping_in_place_is_off(capsys):
    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    summary = run_terminal_trials(robot, _request(controlEvery=1))
    assert summary["haltedOn"] == "schedule_complete", summary["haltedOn"]
    names = _step_names(capsys.readouterr().out)
    assert "regrip_after_release" not in names
    assert names.count("descend_8cm_to_pick") > 1, "one fetch per trial is the older cycle"


def test_a_peg_that_did_not_seat_is_not_stood_on_the_face_at_all():
    """Where the peg is actually lost, measured 2026-09-11 across three runs.

    A peg that stopped on the face is balanced on the rim, off-centre, and it falls the instant
    the fingers open -- a re-close at the same pose read 0.076 and caught air. So the fix is not
    to shorten the gap between release and re-grip; there is no gap. It is not to let go.
    """

    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    summary = run_terminal_trials(
        robot, _request(offsetsMm=(10.0,), repeats=2, controlEvery=100,
                        releaseOnlyWhenSeated=True, regripInPlace=True)
    )
    assert summary["haltedOn"] == "schedule_complete", summary["haltedOn"]
    assert robot.held, "the last trial did not seat, so the fingers never opened"
    offset_rows = [row for row in summary["rows"]
                   if row.get("kind") == "trial" and row.get("trialKind") == "offset"]
    assert offset_rows and all(row["verdict"] != "seated" for row in offset_rows), (
        "10 mm is outside this fixture's capture radius by design"
    )


def test_a_peg_that_seated_is_still_let_go_of():
    robot = FakeTrialRig(hole_xy=SEATED[:2], capture_m=0.0042)
    summary = run_terminal_trials(
        robot, _request(offsetsMm=(0.0,), repeats=2, controlEvery=1,
                        releaseOnlyWhenSeated=True, regripInPlace=True)
    )
    assert summary["haltedOn"] == "schedule_complete", summary["haltedOn"]
    assert summary["seated"] >= 1


def test_the_runtime_wires_the_contact_thresholds_into_the_descent():
    """They set the axial force a jammed peg sees, so a flag that does not arrive is invisible.

    The peg slides in the fingers when the arm leans on it and the gripper's force cannot be
    raised, so the only lever is how far the setpoint travels past a peg that has already
    stopped: the growth threshold decides whether there is contact, the hold decides how long to
    keep pressing after that, and the stall is the backstop. A default that silently overrode
    any of them would be read as "the change did not help".
    """

    from tools.fr3.fr3_terminal_trials_runtime import build_request, parse_args

    args = parse_args([
        "--hole-pose", "0.3599,-0.1333,0.0523",
        "--contact-lag-mm", "3", "--contact-hold-s", "0.1", "--contact-stall-mm", "8",
    ])
    servo = build_request(args).servo
    assert servo.contactLagM == pytest.approx(0.003)
    assert servo.contactHoldS == pytest.approx(0.1)
    assert servo.contactStallM == pytest.approx(0.008)


def test_the_contact_thresholds_keep_their_old_values_when_nothing_asks():
    from tools.fr3.fr3_terminal_trials_runtime import build_request, parse_args
    from tools.fr3.terminal_servo import (
        TERMINAL_SERVO_CONTACT_HOLD_S,
        TERMINAL_SERVO_CONTACT_LAG_M,
        TERMINAL_SERVO_CONTACT_STALL_M,
    )

    servo = build_request(parse_args(["--hole-pose", "0.3599,-0.1333,0.0523"])).servo
    assert servo.contactLagM == pytest.approx(TERMINAL_SERVO_CONTACT_LAG_M)
    assert servo.contactHoldS == pytest.approx(TERMINAL_SERVO_CONTACT_HOLD_S)
    assert servo.contactStallM == pytest.approx(TERMINAL_SERVO_CONTACT_STALL_M)


def test_the_runtime_wires_the_descent_speed_and_keeps_the_modules_own_default():
    from tools.fr3.fr3_terminal_trials_runtime import build_request, parse_args
    from tools.fr3.terminal_servo import TERMINAL_SERVO_MAX_SPEED_MS

    base = ["--hole-pose", "0.3599,-0.1333,0.0523"]
    assert build_request(parse_args(base)).servo.maxSpeedMs == pytest.approx(
        TERMINAL_SERVO_MAX_SPEED_MS
    )
    slowed = build_request(parse_args(base + ["--descent-speed-ms", "0.01"])).servo
    assert slowed.maxSpeedMs == pytest.approx(0.01)
