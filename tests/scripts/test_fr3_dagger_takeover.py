"""What happens to the arm when the operator takes it over mid-rollout, and hands it back.

None of this needs a robot or a SpaceMouse: the takeover is a function from (last command,
device reading) to the next command, and that is where every decision worth checking lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.utils.rotation import Rotation

from tools.fr3.dagger_takeover import ExpertTakeover, expert_spans

POLICY_COMMAND = {
    "ee.x": 0.400,
    "ee.y": -0.100,
    "ee.z": 0.070,
    "ee.wx": 0.0,
    "ee.wy": 0.0,
    "ee.wz": 0.0,
    "gripper.pos": 1.0,
}
LAST_SENT = {
    "ee.x": 0.380,
    "ee.y": -0.090,
    "ee.z": 0.060,
    "ee.wx": 0.0,
    "ee.wy": 0.0,
    "ee.wz": 0.0,
    "gripper.pos": 0.8,
}
# Where the arm actually got to. Behind the command it was chasing, as a moving arm always is.
OBSERVATION = {
    "ee.x": 0.360,
    "ee.y": -0.080,
    "ee.z": 0.050,
    "ee.wx": 0.0,
    "ee.wy": 0.0,
    "ee.wz": 0.0,
    "gripper.pos": 0.8,
}


def mouse_action(**overrides):
    action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.8,
    }
    action.update(overrides)
    return action


class FakeSpaceMouse:
    """Replays a scripted list of readings, then keeps returning the last one."""

    def __init__(self, readings=None):
        self.readings = list(readings or [])
        self._last = mouse_action()
        self.disconnected = False

    def get_action(self):
        if self.readings:
            self._last = self.readings.pop(0)
        return self._last

    def disconnect(self):
        self.disconnected = True


class BrokenSpaceMouse:
    def get_action(self):
        raise OSError("device went away")

    def disconnect(self):
        raise OSError("still gone")


def take(takeover, *, engaged=True, policy_command=None, previous_sent_command=LAST_SENT):
    return takeover.command(
        engaged=engaged,
        policy_command=dict(POLICY_COMMAND if policy_command is None else policy_command),
        previous_sent_command=previous_sent_command,
        robot_observation=OBSERVATION,
    )


def test_a_rollout_nobody_took_over_returns_the_policy_s_own_command():
    # Byte-identical, not merely equivalent. Takeover is an addition to a loop that is moving a
    # real arm, so a run that never engages it has to be the run that would have happened
    # without it -- otherwise every rollout in the batch carries a change nobody asked for.
    takeover = ExpertTakeover(FakeSpaceMouse())

    command, debug = take(takeover, engaged=False)

    assert command == POLICY_COMMAND
    assert debug["source"] == "policy"
    assert takeover.engaged is False


def test_taking_over_without_moving_re_issues_the_command_already_in_flight():
    # The handoff itself must not be a motion. An operator reaches for the control because
    # something is going wrong, and a jolt at that instant is the worst possible response.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action()]))

    command, debug = take(takeover)

    assert debug["source"] == "expert"
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"])
    assert command["ee.y"] == pytest.approx(LAST_SENT["ee.y"])
    assert command["ee.z"] == pytest.approx(LAST_SENT["ee.z"])


def test_the_expert_s_delta_is_measured_from_the_command_not_from_where_the_arm_got_to():
    # The single decision this file exists to pin down. The policy's action is a delta against
    # `prev_cmd`, so the expert's is too: one reference means one step guard, and it means the
    # handoff costs nothing. Anchored to the *measured* pose instead, engaging would command a
    # step backwards of the whole tracking lag -- 20 mm in x here, and larger the faster the
    # arm is moving.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_x=0.002)]))

    command, _ = take(takeover)

    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)
    assert command["ee.x"] != pytest.approx(OBSERVATION["ee.x"] + 0.002)


def test_the_reference_follows_what_was_sent_so_a_clamped_step_cannot_wind_up():
    # Second step, after the previous command came back shortened by the safety clamp. The
    # expert integrates onto whatever was actually sent, so the target never runs away from an
    # arm the operator is watching -- push harder and the arm moves, rather than the number.
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(enabled=True, target_x=0.002), mouse_action(enabled=True, target_x=0.002)])
    )
    take(takeover)
    clamped = dict(LAST_SENT, **{"ee.x": LAST_SENT["ee.x"] + 0.001})

    command, _ = take(takeover, previous_sent_command=clamped)

    assert command["ee.x"] == pytest.approx(clamped["ee.x"] + 0.002)


def test_rotation_composes_the_way_the_recorder_composes_it():
    # `desired_R = reference_R @ delta_R`, matching processor_franka_research3. Left-multiplying
    # would turn the tool about the base axes during takeover and about the tool axes during
    # recording: one device, two behaviours, and a correction that does not mean what the demo
    # it will be trained beside means.
    reference = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    sent = dict(LAST_SENT)
    rotvec = reference.as_rotvec()
    sent.update({"ee.wx": rotvec[0], "ee.wy": rotvec[1], "ee.wz": rotvec[2]})
    delta = Rotation.from_rotvec(np.array([0.1, 0.0, 0.0]))
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_wx=0.1)]))

    command, _ = take(takeover, previous_sent_command=sent)

    got = Rotation.from_rotvec(
        np.array([command["ee.wx"], command["ee.wy"], command["ee.wz"]])
    ).as_matrix()
    assert got == pytest.approx((reference * delta).as_matrix(), abs=1e-9)
    assert not np.allclose(got, (delta * reference).as_matrix(), atol=1e-6)


def test_taking_over_does_not_by_itself_move_the_gripper():
    # The device reports an absolute gripper position, so passing it through on the first
    # engaged step would snap the fingers to wherever the operator's buttons last left them --
    # at a moment when there is often already something between them.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(gripper=0.0)]))

    command, debug = take(takeover)

    assert command["gripper.pos"] == pytest.approx(LAST_SENT["gripper.pos"])
    assert debug["gripper_owned"] is False


def test_the_device_owns_the_gripper_once_the_operator_presses_a_button():
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(gripper=0.8), mouse_action(gripper=0.0), mouse_action(gripper=0.0)])
    )
    take(takeover)

    command, debug = take(takeover)

    assert command["gripper.pos"] == pytest.approx(0.0)
    assert debug["gripper_owned"] is True
    # And it stays owned: a button released back to where it started is not a handback.
    assert take(takeover)[1]["gripper_owned"] is True


def test_letting_go_and_taking_over_again_re_reads_the_gripper_from_the_policy():
    # State from the last engagement must not survive into the next one. The policy has been
    # driving in between, and its gripper is the one the arm is actually holding.
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(gripper=0.8), mouse_action(gripper=0.0), mouse_action(gripper=0.0)])
    )
    take(takeover)
    take(takeover)
    handed_back, debug = take(takeover, engaged=False)
    assert handed_back == POLICY_COMMAND
    assert debug["source"] == "policy"

    reengaged = dict(LAST_SENT, **{"gripper.pos": 0.35})
    command, debug = take(takeover, previous_sent_command=reengaged)

    assert command["gripper.pos"] == pytest.approx(0.35)
    assert debug["gripper_owned"] is False


def test_a_device_that_cannot_be_read_holds_position_rather_than_handing_control_back():
    # Silently returning the policy's command would be the one surprise worth ruling out: the
    # operator asked for control precisely because the policy was doing the wrong thing, and
    # they have no way of knowing they no longer have it.
    takeover = ExpertTakeover(BrokenSpaceMouse())

    command, debug = take(takeover)

    assert debug["status"] == "poll_failed"
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"])
    assert command["ee.z"] == pytest.approx(LAST_SENT["ee.z"])
    assert command["gripper.pos"] == pytest.approx(LAST_SENT["gripper.pos"])


def test_taking_over_on_the_very_first_step_falls_back_to_where_the_arm_is():
    # Nothing has been sent yet, so there is no command to be continuous with. The measured
    # pose is the only honest anchor, and it is also the correct one: the arm is standing still.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_z=-0.001)]))

    command, _ = take(takeover, previous_sent_command=None)

    assert command["ee.z"] == pytest.approx(OBSERVATION["ee.z"] - 0.001)


def test_a_non_finite_reading_is_treated_as_no_motion():
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_x=float("nan"))]))

    command, _ = take(takeover)

    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"])


def test_closing_releases_the_device_even_when_it_refuses():
    # This runs in the same `finally` that puts the robot down.
    takeover = ExpertTakeover(BrokenSpaceMouse())
    take(takeover)

    takeover.close()

    assert takeover.engaged is False


def test_spans_are_read_off_the_source_column():
    assert expert_spans([]) == []
    assert expert_spans(["policy", "policy"]) == []
    assert expert_spans(["expert", "expert"]) == [(0, 1)]
    # A takeover still running when the rollout was stopped is still a takeover.
    assert expert_spans(["policy", "expert"]) == [(1, 1)]
