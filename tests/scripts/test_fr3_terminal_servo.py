"""E5's terminal controller: does it hand over on the right descent, and read the wrong one right?

Two things are worth testing here and they are different in kind. The arming state machine
decides *which* descent the policy is taken off, and getting it wrong hands the arm over while
it is still reaching for the peg. The descent itself decides what the run reports, and its whole
value is that a peg stopped by the fixture face and a peg that went into the hole come back as
different numbers rather than as the same timeout.
"""

from collections import deque
import math

import pytest

import tools.fr3.scene_reset as scene_reset
import tools.fr3.terminal_servo as terminal_servo
from tools.fr3.scene_reset import SceneResetError
from tools.fr3.terminal_servo import (
    TerminalServoError,
    TerminalServoRequest,
    execute_terminal_servo,
    parse_terminal_servo_pose,
    terminal_servo_arming,
    terminal_servo_search_offsets,
    terminal_servo_search_path,
    terminal_servo_waypoints,
    validate_terminal_servo_trajectory,
)

from tests.scripts.test_fr3_scene_reset import FakeRobot


SEATED = (0.3599, -0.1333, 0.0523)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    """Walk the setpoint at wall-clock speed rather than the arm's, as the reset's suite does."""

    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    # The descent keeps its real speed cap -- the cap is one of the things under test -- so what
    # is removed instead is the waiting between setpoints. The loop still measures how long it
    # has been held up in wall-clock seconds, which is the reading that matters.
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(terminal_servo, "precise_sleep", lambda seconds: None)


def _request(**overrides):
    fields = dict(xyz=SEATED, handoffZ=0.12, controlPeriodS=0.001, timeoutS=2.0)
    fields.update(overrides)
    return TerminalServoRequest(**fields)


def _descending(z, **overrides):
    """A robot carrying the peg, sitting at height `z` over the target's XY."""

    robot = FakeRobot()
    robot.xyz = (SEATED[0] - 0.01, SEATED[1] + 0.008, z)
    robot.gripper = 0.25
    for key, value in overrides.items():
        setattr(robot, key, value)
    return robot


class FakeFixtureRobot(FakeRobot):
    """An arm whose peg lands on something at `floor_z` and is held there.

    Deliberately not `FakeStalledRobot`: that one publishes `reach_stall_error_m`, because IK
    could not realise the pose. A peg standing on the fixture face is a pose the arm can reach
    perfectly well and an object in the way, so nothing on the driver reports it and the only
    evidence is the tool sitting above its own setpoint.
    """

    def __init__(self, floor_z):
        super().__init__()
        self.floor_z = float(floor_z)

    def send_action(self, action):
        realised_z = max(float(action["ee.z"]), self.floor_z)
        return super().send_action({**action, "ee.z": realised_z})


class FakeLaggingRobot(FakeRobot):
    """An arm that realises the setpoint it was given one step ago, and nothing is in its way."""

    def __init__(self):
        super().__init__()
        self._pending = None

    def send_action(self, action):
        realised = self._pending if self._pending is not None else dict(action)
        self._pending = dict(action)
        return super().send_action({**action, "ee.z": realised["ee.z"]})


class FakeDelayedRobot(FakeRobot):
    """An arm that answers the setpoint it was given `steps` control periods ago.

    The real one does. Every terminal servo of 2026-09-10 sat 5.2-5.9 mm above a setpoint that
    moved 1.7 mm a step -- three steps of delay, with the peg still five centimetres above the
    fixture -- and the descent read that standing lag as contact and stopped 15 mm in, eight
    times out of eight. `floor_z` optionally puts something in the way as well, because a late
    arm that really is blocked is the case that still has to come back as contact.
    """

    def __init__(self, steps, floor_z=None):
        super().__init__()
        self._steps = int(steps)
        self._pending = deque()
        self.floor_z = None if floor_z is None else float(floor_z)

    def send_action(self, action):
        self._pending.append(float(action["ee.z"]))
        # Before the delay line fills, the arm is still where it was: a delay, not a shrunk step.
        realised = self._pending.popleft() if len(self._pending) > self._steps else self.xyz[2]
        if self.floor_z is not None:
            realised = max(realised, self.floor_z)
        return super().send_action({**action, "ee.z": realised})


class FakeSeatedRobot(FakeRobot):
    """An arm that below `from_z` holds a fixed offset from every setpoint, and stays there.

    A seated peg does exactly this. The hole owns the tool's XY, so the arm sits off the pose it
    was told for as long as that pose is commanded, and no amount of waiting closes the gap.
    Above `from_z` the arm is free, so the lateral leg still converges and the descent starts
    from a clean pose -- the offset is the descent's, not the approach's.
    """

    def __init__(self, from_z, offset_xyz):
        super().__init__()
        self.from_z = float(from_z)
        self.offset = tuple(float(value) for value in offset_xyz)

    def send_action(self, action):
        if float(action["ee.z"]) >= self.from_z:
            return super().send_action(action)
        return super().send_action({
            **action,
            "ee.x": action["ee.x"] + self.offset[0],
            "ee.y": action["ee.y"] + self.offset[1],
            "ee.z": action["ee.z"] + self.offset[2],
        })


def _delay_steps(request, lag_m):
    """How many control periods of delay stand the arm `lag_m` behind at this request's speed."""

    return max(1, round(lag_m / (request.maxSpeedMs * request.controlPeriodS)))


# --- which descent hands over ---------------------------------------------------------------


def test_the_reach_for_the_peg_does_not_hand_the_arm_over():
    """The pick descends through the handoff height too, on its way to z = 0.046."""

    armed = False
    for z in (0.30, 0.20, 0.12, 0.08, 0.046):
        armed, fired = terminal_servo_arming(armed, commanded_gripper=1.0, observed_z=z, handoff_z=0.12)
        assert not fired


def test_the_carry_to_the_fixture_does():
    armed = False
    fired = False
    # Grasped on the table, lifted over the handoff height, carried across, and brought down.
    for gripper, z in ((1.0, 0.046), (0.0, 0.046), (0.0, 0.10), (0.0, 0.25), (0.0, 0.14), (0.0, 0.11)):
        armed, fired = terminal_servo_arming(armed, commanded_gripper=gripper, observed_z=z, handoff_z=0.12)
    assert fired


def test_holding_the_peg_below_the_height_is_not_enough_on_its_own():
    """Grasped and already low: nothing above the handoff height has happened yet."""

    armed, fired = terminal_servo_arming(False, commanded_gripper=0.0, observed_z=0.05, handoff_z=0.12)
    assert (armed, fired) == (False, False)


def test_dropping_the_peg_disarms_until_it_has_been_lifted_again():
    armed, _ = terminal_servo_arming(False, commanded_gripper=0.0, observed_z=0.25, handoff_z=0.12)
    assert armed
    armed, fired = terminal_servo_arming(armed, commanded_gripper=1.0, observed_z=0.25, handoff_z=0.12)
    assert (armed, fired) == (False, False)
    armed, fired = terminal_servo_arming(armed, commanded_gripper=0.0, observed_z=0.10, handoff_z=0.12)
    assert not fired


# --- what the descent reports ----------------------------------------------------------------


def test_a_peg_that_goes_in_reaches_the_seated_depth():
    robot = _descending(0.118)
    result = execute_terminal_servo(robot, _request())
    assert result["ok"] and result["stoppedOn"] == "target"
    assert result["seatedDepthErrorMm"] == pytest.approx(0.0, abs=2.0)


def test_a_peg_stopped_on_the_fixture_face_reports_the_height_it_stopped_at():
    """The reading E5 is run for: how far above the seated depth the peg was refused."""

    result = execute_terminal_servo(_blocked_at(0.0623), _request())
    assert result["ok"] and result["stoppedOn"] == "contact"
    assert result["seatedDepthErrorMm"] == pytest.approx(10.0, abs=1.5)


def _blocked_at(floor_z):
    robot = FakeFixtureRobot(floor_z)
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    return robot


def test_being_held_up_is_reported_rather_than_raised():
    """Contact is a result. Reporting it as a failure would make a miss look like a broken run."""

    assert execute_terminal_servo(_descending(0.118), _request())["ok"]
    assert execute_terminal_servo(_blocked_at(0.0623), _request())["ok"]


def test_one_step_of_tracking_lag_is_not_contact():
    """At 0.05 m/s the setpoint moves 1.7 mm a step; an arm a step behind is not blocked."""

    robot = FakeLaggingRobot()
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, _request())
    assert result["stoppedOn"] == "target"


def test_a_standing_answer_delay_is_not_contact():
    """The 2026-09-10 failure. An arm that is late is not an arm that is blocked.

    Six millimetres of standing lag -- more than the 5.2-5.9 mm those runs actually reported,
    and half again the contact threshold -- with nothing under the peg for the whole descent.
    The old rule compared that figure against the threshold directly and stopped every run
    about 15 mm in; this one compares how much it grows, and a delay does not grow.
    """

    # Zero hold because this suite runs the loop unpaced: 0.3 s of wall clock never elapses
    # inside a descent that takes two milliseconds, which is why the rule this replaced passed
    # its own tests for three weeks and then stopped eight runs in a row on the arm.
    request = _request(contactHoldS=0.0)
    robot = FakeDelayedRobot(_delay_steps(request, 0.006))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "target"
    assert result["seatedDepthErrorMm"] == pytest.approx(0.0, abs=2.0)
    # The lag was real and would have tripped the rule this replaced. It is reported, not acted
    # on, so a descent that stops can be read against what the arm's delay was that day.
    assert result["lagMm"] == pytest.approx(6.0, abs=1.0)
    assert result["lagMm"] > 1000.0 * request.contactLagM


def test_a_late_arm_that_is_really_blocked_still_reports_contact():
    """The delay must not buy a blocked peg any extra travel into the fixture."""

    request = _request()
    robot = FakeDelayedRobot(_delay_steps(request, 0.006), floor_z=0.080)
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "contact"
    assert result["seatedDepthErrorMm"] == pytest.approx(27.7, abs=1.5)


def test_the_backstop_catches_a_peg_already_against_the_fixture():
    """A peg touching at the handoff height is inside the warm-up, where growth is not read.

    So the absolute lag has to remain a stop of its own. Without it the arm would drive the
    setpoint the whole 68 mm down onto a peg that never moved.
    """

    request = _request()
    robot = FakeFixtureRobot(0.1195)
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "contact"
    assert result["descentMm"] == pytest.approx(0.5, abs=0.3)
    assert result["heldUpMm"] == pytest.approx(1000.0 * request.contactStallM, abs=1.5)


def test_a_clean_descent_reports_no_growth_and_no_settling():
    """The reference row. Everything the slip columns say is said against this one.

    A descent with nothing in the way peaks at no growth and closes what lag it has the
    instant the setpoint parks. On 2026-09-10 the three descents graded as insertions did
    exactly that -- 0.1 s of settling each -- and the two graded as misses took 8.1 s and
    18.1 s, which is the only column that separated them.
    """

    result = execute_terminal_servo(_descending(0.118), _request())
    assert result["stoppedOn"] == "target"
    assert result["peakGrowthMm"] == pytest.approx(0.0, abs=0.5)
    assert result["settleSeconds"] < 0.5
    assert result["settleMm"] == pytest.approx(0.0, abs=0.5)


def test_a_blocked_descent_leaves_its_growth_in_the_peak():
    """`heldUpGrowthMm` is the growth at the stop; the peak is what the descent ever saw."""

    result = execute_terminal_servo(_blocked_at(0.0623), _request())
    assert result["stoppedOn"] == "contact"
    assert result["peakGrowthMm"] >= 1000.0 * _request().contactLagM


def test_a_peg_held_off_sideways_still_counts_as_reaching_the_depth():
    """The descent's question is depth. Being held laterally is the reading, not the failure.

    A peg in a hole owns the tool's XY, so a 3-D arrival tolerance makes the insertions time
    out. Four of 2026-09-10's nine descents did, at 1.8-8.6 mm of lateral standoff.
    """

    request = _request()
    robot = FakeSeatedRobot(0.115, (0.006, 0.0, 0.0))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "target"
    assert result["seatedDepthErrorMm"] == pytest.approx(0.0, abs=2.0)
    assert result["lateralErrorMm"] == pytest.approx(6.0, abs=0.5)


def test_a_depth_it_cannot_close_gives_up_in_the_settle_time_not_the_timeout():
    """What the operator saw twice: the arm reached the bottom and leaned on it for 20 s."""

    request = _request(timeoutS=8.0, settleS=0.2)
    robot = FakeSeatedRobot(0.115, (0.0, 0.0, 0.003))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "timeout"
    assert result["seatedDepthErrorMm"] == pytest.approx(3.0, abs=0.5)
    # The full deadline is timeoutS plus the travel, well over 11 s. It gave up in a fraction.
    assert result["descentSeconds"] < 1.0


def test_the_peg_is_let_go_where_it_stopped_not_where_it_was_aimed():
    """Re-commanding an unreachable target while the fingers open leans on the fixture."""

    robot = _blocked_at(0.0623)
    execute_terminal_servo(robot, _request())
    opening = next(a for a in robot.actions if a["gripper.pos"] >= 0.9)
    assert opening["ee.z"] == pytest.approx(0.0623, abs=0.001)


def test_the_wrist_angle_is_reported_rather_than_commanded():
    """E5 asks whether tool position determines the peg tip; turning the wrist would confound it."""

    robot = _descending(0.118)
    robot.rotvec = (0.98, 0.05, -0.02)
    result = execute_terminal_servo(robot, _request())
    assert result["handoffRotvec"] == pytest.approx([0.98, 0.05, -0.02])
    assert {(a["ee.wx"], a["ee.wy"], a["ee.wz"]) for a in robot.actions} == {(0.98, 0.05, -0.02)}


# --- what is refused before the arm moves ------------------------------------------------------


def test_the_lateral_move_happens_at_the_handoff_height_never_at_fixture_height():
    """A peg travelling sideways at fixture height is the one path that can shear it off."""

    across, down, up = terminal_servo_waypoints(_request(), (0.30, -0.10, 0.118))
    assert across[0] == "align_above_target" and across[1] == (SEATED[0], SEATED[1], 0.12)
    assert down[1] == SEATED
    assert up[1][2] > SEATED[2]


def test_a_target_under_the_floor_is_refused():
    with pytest.raises(TerminalServoError, match="below the"):
        validate_terminal_servo_trajectory(_request(xyz=(0.3599, -0.1333, 0.020)))


def test_a_handoff_at_or_below_the_target_leaves_no_descent_to_own():
    with pytest.raises(TerminalServoError, match="above the target"):
        validate_terminal_servo_trajectory(_request(handoffZ=0.04))


def test_the_descent_may_not_be_driven_faster_than_the_speed_this_module_sets():
    with pytest.raises(TerminalServoError, match="no force control"):
        validate_terminal_servo_trajectory(_request(maxSpeedMs=0.15))


def test_a_target_outside_the_workspace_never_reaches_the_arm():
    robot = _descending(0.118)
    robot.config = type("Cfg", (), {"workspace_min": (0.18, -0.10, 0.0), "workspace_max": (0.62, 0.45, 0.50)})()
    result = execute_terminal_servo(robot, _request())
    assert not result["ok"] and "trajectory_qc_failed" in result["error"]
    assert robot.actions == []


def test_a_terminal_servo_error_is_a_scene_reset_error_so_the_rollout_loop_already_catches_it():
    assert issubclass(TerminalServoError, SceneResetError)


@pytest.mark.parametrize("text", ["0.36,-0.13", "0.36,-0.13,0.05,0.1", "0.36,-0.13,nope", "", "nan,0,0"])
def test_a_pose_that_is_not_three_finite_metres_is_refused_on_the_command_line(text):
    with pytest.raises(TerminalServoError):
        parse_terminal_servo_pose(text)


def test_the_pose_reads_as_metres_in_the_order_it_was_typed():
    assert parse_terminal_servo_pose(" 0.3599, -0.1333 ,0.0523 ") == (0.3599, -0.1333, 0.0523)


class FakeHoleRobot(FakeRobot):
    """A fixture with a hole in it, offset from where the servo is aiming.

    The peg stops on the face wherever the tool is, except within `capture_m` of the hole in XY,
    where it goes down as far as it is told. That one rule is the whole of E7 route C's physics:
    the arm has no way of knowing which of the two it is doing until it reads the height it
    stopped at, and the two heights do not overlap.
    """

    def __init__(self, hole_xy, capture_m=0.0042, face_z=SEATED[2] + 0.005):
        super().__init__()
        self.hole_xy = (float(hole_xy[0]), float(hole_xy[1]))
        self.capture_m = float(capture_m)
        self.face_z = float(face_z)
        self.landed_on = []

    def send_action(self, action):
        over_hole = (
            (float(action["ee.x"]) - self.hole_xy[0]) ** 2
            + (float(action["ee.y"]) - self.hole_xy[1]) ** 2
        ) <= self.capture_m**2
        floor_z = -1.0 if over_hole else self.face_z
        realised_z = max(float(action["ee.z"]), floor_z)
        if realised_z <= self.face_z:
            self.landed_on.append("hole" if over_hole else "face")
        return super().send_action({**action, "ee.z": realised_z})


class FakeCreepingRobot(FakeRobot):
    """A peg that slides in the fingers rather than an arm that is blocked.

    Modelled on what the two runs graded as slips actually logged, which is not a stall: their
    lag sat a millimetre or so over the free-descent figure and their growth was *negative*, so
    the peg was sliding almost as fast as the setpoint was moving. What gives them away is the
    tail -- 8.1 s and 18.1 s to close the last millimetres against 0.1 s for a real insertion.
    A first-order give reproduces both halves at once: a constant trail while the setpoint moves
    at a constant speed, and an exponential close once it parks.
    """

    def __init__(self, floor_z, give=0.005):
        super().__init__()
        self.floor_z = float(floor_z)
        self.give = float(give)

    def send_action(self, action):
        commanded_z = float(action["ee.z"])
        if commanded_z < self.floor_z:
            self.floor_z += (commanded_z - self.floor_z) * self.give
        return super().send_action({**action, "ee.z": max(commanded_z, self.floor_z)})


def _searching(**overrides):
    fields = dict(searchRingM=0.007, settleS=0.05)
    fields.update(overrides)
    return _request(**fields)


def test_the_search_is_off_by_default_so_a_run_is_still_the_control_arm():
    assert terminal_servo_search_offsets(_request()) == ((0.0, 0.0),)
    robot = _descending(0.12)
    result = execute_terminal_servo(robot, _request())
    assert result["ok"] is True
    assert result["searchStoppedOn"] in {"seated", "exhausted"}
    assert result["searchLandings"] == 1
    assert result["searchOffsetMm"] == pytest.approx(0.0)


def test_eight_landings_leave_no_gap_wider_than_the_capture_radius():
    """The ring is only worth flying if every point in the disc is inside somebody's capture."""

    offsets = terminal_servo_search_offsets(_searching())
    assert len(offsets) == 9 and offsets[0] == (0.0, 0.0)
    capture_m = 0.0042
    worst = 0.0
    for step in range(0, 360, 3):
        for radius_mm in range(0, 10):
            point = (
                radius_mm / 1000.0 * math.cos(math.radians(step)),
                radius_mm / 1000.0 * math.sin(math.radians(step)),
            )
            nearest = min(math.dist(point, offset) for offset in offsets)
            worst = max(worst, nearest)
    assert worst < capture_m


def test_the_nominal_pose_is_tried_first_and_a_hit_ends_the_search():
    robot = FakeHoleRobot(hole_xy=(SEATED[0], SEATED[1]))
    robot.xyz = (SEATED[0] - 0.01, SEATED[1] + 0.008, 0.12)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, _searching())
    assert result["searchStoppedOn"] == "seated"
    assert result["searchIndex"] == 0
    assert len(result["searchAttempts"]) == 1
    assert result["searchOffsetMm"] == pytest.approx(0.0)


def test_a_miss_steps_onto_the_ring_until_it_seats():
    """The hole is 7 mm away, which the fixed pose cannot reach and one of the landings can."""

    hole = (SEATED[0] + 0.007, SEATED[1])
    robot = FakeHoleRobot(hole_xy=hole)
    robot.xyz = (SEATED[0] - 0.01, SEATED[1] + 0.008, 0.12)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, _searching())
    assert result["searchStoppedOn"] == "seated"
    assert result["searchIndex"] >= 1
    assert result["searchAttempts"][0]["aboveTargetMm"] > 3.0
    assert result["searchAttempts"][-1]["aboveTargetMm"] <= 3.0
    assert result["searchOffsetMm"] == pytest.approx(7.0, abs=0.5)
    # And it was let go over the hole it found, not over the pose it was aimed at.
    assert result["searchLandingXyz"][0] == pytest.approx(hole[0], abs=0.001)


def test_a_hole_no_landing_can_reach_reports_exhausted_rather_than_seated():
    robot = FakeHoleRobot(hole_xy=(SEATED[0] + 0.030, SEATED[1]))
    robot.xyz = (SEATED[0] - 0.01, SEATED[1] + 0.008, 0.12)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, _searching())
    assert result["searchStoppedOn"] == "exhausted"
    assert len(result["searchAttempts"]) == 9
    assert all(attempt["aboveTargetMm"] > 3.0 for attempt in result["searchAttempts"])


def test_creep_ends_the_search_because_the_ruler_changed_length():
    """A peg sliding in the fingers invalidates every landing computed from the tool pose."""

    robot = FakeCreepingRobot(floor_z=SEATED[2] + 0.010)
    robot.xyz = (SEATED[0] - 0.01, SEATED[1] + 0.008, 0.12)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, _searching(settleS=0.5, searchSlipM=0.001))
    # It reached the seated depth -- and is still not an insertion, which is the point.
    assert result["stoppedOn"] == "target"
    assert result["seatedDepthErrorMm"] <= 3.0
    assert result["searchStoppedOn"] == "slip"
    assert result["settleMm"] >= 1.0
    assert len(result["searchAttempts"]) == 1


def test_a_lift_shorter_than_the_run_up_is_refused():
    request = _searching(searchLiftM=0.005)
    with pytest.raises(TerminalServoError, match="growth test needs to arm"):
        validate_terminal_servo_trajectory(request)


def test_a_ring_of_two_landings_is_refused():
    with pytest.raises(TerminalServoError, match="too few for a ring"):
        validate_terminal_servo_trajectory(_searching(searchPoints=2))


def test_a_landing_outside_the_workspace_never_reaches_the_arm():
    request = _searching(searchRingM=0.007)
    with pytest.raises(SceneResetError, match="search_"):
        validate_terminal_servo_trajectory(
            request,
            workspace_min=(SEATED[0] - 0.003, -1.0, 0.0),
            workspace_max=(1.0, 1.0, 1.0),
        )


def test_the_search_path_is_only_checked_when_there_is_a_search():
    """With the search off the QC still checks exactly the three legs it always checked."""

    path = terminal_servo_search_path(_request(), (0.30, 0.0, 0.20))
    assert [name for name, _xyz in path] == [
        "align_above_target",
        "descend_to_target",
        "retreat_after_release",
    ]


def test_a_looser_step_tolerance_does_not_loosen_the_descent_that_decides_seated():
    """The trap that made this a separate field rather than a bigger `toleranceM`.

    The descent reads `toleranceM` for its own "reached target z" test, and `searchSeatedM` is
    3 mm. Had the positioning steps been loosened by raising the shared number to 4 mm, a peg
    stopped 3 mm high would have been called `target` instead of `timeout` -- a seated
    classification handed out for free, on the one measurement this module exists to make.
    """

    request = _request(timeoutS=8.0, settleS=0.2, stepToleranceM=0.004)
    robot = FakeSeatedRobot(0.115, (0.0, 0.0, 0.003))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    result = execute_terminal_servo(robot, request)
    assert result["stoppedOn"] == "timeout", "the descent must keep its own 2.0 mm criterion"
    assert result["seatedDepthErrorMm"] == pytest.approx(3.0, abs=0.5)


def test_a_step_tolerance_is_refused_before_the_arm_moves_when_it_is_not_a_tolerance():
    """Refused the way every other bad request is: a failed result, not a half-run descent."""

    result = execute_terminal_servo(_descending(0.118), _request(stepToleranceM=-0.001))
    assert result["ok"] is False
    assert "stepToleranceM must be positive" in result["error"]

def test_the_regrip_follows_a_released_peg_down_rather_than_closing_where_it_let_go(capsys):
    """Measured 2026-09-11: a clean seating, released, re-gripped at the same z -- and empty.

    The fingers were holding the peg up; opening them lets it drop to its own seated depth. A
    re-grip at the release height therefore closes above it, which is what the random 1/2/3
    attempt counts in the one table that finished were really showing.
    """

    robot = FakeSeatedRobot(0.115, (0.0, 0.0, 0.0))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    execute_terminal_servo(robot, _request(regripGripper=0.0, regripDropM=0.004))
    starts = [line for line in capsys.readouterr().out.splitlines()
              if "scene_reset_step=start" in line and "regrip_after_release" in line]
    assert starts, "the re-grip step has to run for this to mean anything"
    released_z = float(starts[0].split("xyz=")[1].split(",")[2].split()[0])
    assert released_z < SEATED[2] + 0.010, released_z


def test_a_regrip_drop_can_never_drive_the_fingers_below_the_servos_own_floor(capsys):
    robot = FakeSeatedRobot(0.115, (0.0, 0.0, 0.0))
    robot.xyz = (SEATED[0], SEATED[1], 0.118)
    robot.gripper = 0.25
    request = _request(regripGripper=0.0, regripDropM=0.5, minZ=0.03)
    execute_terminal_servo(robot, request)
    starts = [line for line in capsys.readouterr().out.splitlines()
              if "scene_reset_step=start" in line and "regrip_after_release" in line]
    assert starts
    assert float(starts[0].split("xyz=")[1].split(",")[2].split()[0]) == pytest.approx(0.03, abs=1e-6)


def test_a_negative_regrip_drop_is_refused_because_a_released_peg_does_not_rise():
    result = execute_terminal_servo(_descending(0.118), _request(regripDropM=-0.001))
    assert result["ok"] is False
    assert "regripDropM" in result["error"]
