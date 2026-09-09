"""E5's terminal controller: does it hand over on the right descent, and read the wrong one right?

Two things are worth testing here and they are different in kind. The arming state machine
decides *which* descent the policy is taken off, and getting it wrong hands the arm over while
it is still reaching for the peg. The descent itself decides what the run reports, and its whole
value is that a peg stopped by the fixture face and a peg that went into the hole come back as
different numbers rather than as the same timeout.
"""

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
    terminal_servo_waypoints,
    validate_terminal_servo_trajectory,
)

from tests.scripts.test_fr3_scene_reset import FakeRobot


SEATED = (0.3599, -0.1333, 0.0523)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    """Walk the setpoint at wall-clock speed rather than the arm's, as the reset's suite does."""

    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    # The descent keeps its real 0.05 m/s -- the cap is one of the things under test -- so what
    # is removed instead is the waiting between setpoints. The loop still measures how long it
    # has been held up in wall-clock seconds, which is the reading that matters.
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(terminal_servo, "precise_sleep", lambda seconds: None)


def _request(**overrides):
    fields = dict(xyz=SEATED, handoffZ=0.12, maxSpeedMs=0.05, controlPeriodS=0.001, timeoutS=2.0)
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
