"""What happens to the arm when the operator takes it over mid-rollout, and hands it back.

None of this needs a robot or a SpaceMouse: the takeover is a function from (last command,
device reading) to the next command, and that is where every decision worth checking lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.utils.rotation import Rotation

from tools.fr3.dagger_takeover import ExpertTakeover, expert_spans, motion_gain_for

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
    """Replays a scripted list of reports, one per call, then goes quiet.

    Going quiet is this double's own convention, not the device's: measured on the rig, an empty
    hidraw queue makes ``PySpaceMouseDriver.poll`` return the *last state* rather than None (0
    empty returns in 620k reads over 2 s). ``ExpertTakeover`` reads once per step and scales that
    reading, so what the device does on a second read inside one step no longer reaches the arm
    -- see :class:`RepeatingSpaceMouse`, which holds the real behaviour.
    """

    def __init__(self, readings=None):
        self.readings = list(readings or [])
        self.polls = 0
        self._gripper = mouse_action()["gripper"]
        self.disconnected = False

    def get_action(self):
        self.polls += 1
        if not self.readings:
            return mouse_action(gripper=self._gripper)
        reading = self.readings.pop(0)
        if reading.get("gripper") is not None:
            self._gripper = reading["gripper"]
        return reading

    def disconnect(self):
        self.disconnected = True


class FakeClock:
    """Time the test moves by hand. Automatic handback is a duration, so it needs one."""

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class BrokenSpaceMouse:
    def get_action(self):
        raise OSError("device went away")

    def disconnect(self):
        raise OSError("still gone")


def take(takeover, *, latched=True, policy_command=None, previous_sent_command=LAST_SENT):
    return takeover.command(
        latched=latched,
        policy_command=dict(POLICY_COMMAND if policy_command is None else policy_command),
        previous_sent_command=previous_sent_command,
        robot_observation=OBSERVATION,
    )


def test_a_rollout_nobody_took_over_returns_the_policy_s_own_command():
    # Byte-identical, not merely equivalent. Takeover is an addition to a loop that is moving a
    # real arm, so a run that never engages it has to be the run that would have happened
    # without it -- otherwise every rollout in the batch carries a change nobody asked for.
    takeover = ExpertTakeover(FakeSpaceMouse())

    command, debug = take(takeover, latched=False)

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
        FakeSpaceMouse(
            [
                mouse_action(enabled=True, target_x=0.002),
                mouse_action(enabled=True, target_x=0.002),
            ]
        )
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
    clock = FakeClock()
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(gripper=0.8), mouse_action(gripper=0.0), mouse_action(gripper=0.0)]),
        clock=clock,
    )
    take(takeover)
    take(takeover)
    clock.advance(2.0)  # the operator let go, and the release window has passed
    handed_back, debug = take(takeover, latched=False)
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


# --- taking the arm by moving the device, and giving it back by not ---------------------------


def test_moving_the_device_takes_the_arm_with_no_key_pressed():
    # The feature, in one test. The operator reaches for the SpaceMouse *because* something is
    # going wrong; a key to press first is a thing to find at the worst possible moment, and a
    # key left latched hands the next rollout to a device nobody is holding.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_x=0.002)]))

    command, debug = take(takeover, latched=False)

    assert debug["source"] == "expert"
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)


def test_a_device_sitting_still_never_takes_the_arm():
    # The other half of it: a SpaceMouse plugged in and untouched must leave the rollout exactly
    # as it would have been without one, for every step of it.
    takeover = ExpertTakeover(FakeSpaceMouse())

    for _ in range(50):
        command, debug = take(takeover, latched=False)
        assert command == POLICY_COMMAND
        assert debug["source"] == "policy"


def test_pressing_a_gripper_button_is_input_too():
    # An operator who takes the arm to close the fingers has not moved the puck at all.
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(gripper=0.8), mouse_action(gripper=0.6)]))
    take(takeover, latched=False)  # the first reading is a baseline, not a change

    command, debug = take(takeover, latched=False)

    assert debug["source"] == "expert"
    # And the button that took the arm owns the gripper on that same step: measuring ownership
    # from the post-press reading would need a second press to notice the first.
    assert debug["gripper_owned"] is True
    assert command["gripper.pos"] == pytest.approx(0.6)


def test_the_arm_goes_back_to_the_policy_once_the_device_has_been_quiet_long_enough():
    clock = FakeClock()
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(enabled=True, target_x=0.002)]), release_after_s=1.0, clock=clock
    )
    assert take(takeover, latched=False)[1]["source"] == "expert"

    clock.advance(0.9)
    assert take(takeover, latched=False)[1]["source"] == "expert"

    clock.advance(0.2)
    command, debug = take(takeover, latched=False)

    assert debug["source"] == "policy"
    assert command == POLICY_COMMAND


def test_input_that_keeps_arriving_keeps_the_arm():
    # The release window is measured from the last report, not from the first: an operator making
    # a slow correction must not have the policy resume underneath them halfway through it.
    clock = FakeClock()
    takeover = ExpertTakeover(
        FakeSpaceMouse([mouse_action(enabled=True, target_x=0.002), mouse_action()] * 8),
        release_after_s=1.0,
        clock=clock,
    )

    for _ in range(8):
        clock.advance(0.9)
        assert take(takeover, latched=False)[1]["source"] == "expert"


def test_the_key_still_latches_the_arm_past_the_release_window():
    # For an operator who wants the arm held still while they think, which is exactly the case
    # automatic takeover cannot express: holding still and letting go look identical to a device.
    clock = FakeClock()
    takeover = ExpertTakeover(FakeSpaceMouse(), release_after_s=1.0, clock=clock)

    assert take(takeover, latched=True)[1]["source"] == "expert"
    clock.advance(30.0)
    assert take(takeover, latched=True)[1]["source"] == "expert"
    assert take(takeover, latched=False)[1]["source"] == "policy"


def test_release_after_zero_leaves_only_the_key():
    takeover = ExpertTakeover(FakeSpaceMouse([mouse_action(enabled=True, target_x=0.01)]), release_after_s=0.0)

    command, debug = take(takeover, latched=False)

    assert debug["source"] == "policy"
    assert command == POLICY_COMMAND
    assert takeover.auto_enabled is False


def test_a_dead_device_does_not_hand_the_arm_back_by_itself():
    # Nor does it take it: a device that cannot be read has said nothing about what the operator
    # is doing, so it neither engages nor releases.
    takeover = ExpertTakeover(BrokenSpaceMouse())

    command, debug = take(takeover, latched=False)
    assert debug["source"] == "policy"
    assert command == POLICY_COMMAND

    assert take(takeover, latched=True)[1]["status"] == "poll_failed"
    assert take(takeover, latched=False)[1]["source"] == "expert"


# --- moving the arm at the speed the recorder moved it -----------------------------------------


class RepeatingSpaceMouse:
    """A puck held off centre: the same reading, and a new report behind every one of them.

    ``PySpaceMouseDriver.poll`` returns the device's cached state when the hidraw queue is empty,
    so "read again" and "read a new report" are indistinguishable from the values alone -- which
    is why the reading is dated. A deflected puck keeps reporting at ~126 Hz, so a loop at 30 Hz
    sees a fresh report every step, and the timestamp advances with them.

    ``dated=False`` is a backend that does not timestamp its reports at all.
    """

    def __init__(self, reading, *, dated=True, tick_s=1.0 / 126.0):
        self.reading = reading
        self.polls = 0
        self.tick_s = float(tick_s)
        self.last_report_timestamp = 100.0 if dated else None

    def get_action(self):
        self.polls += 1
        if self.last_report_timestamp is not None:
            self.last_report_timestamp += self.tick_s
        return dict(self.reading)

    def disconnect(self):
        pass


class SilentSpaceMouse:
    """The device once the hand comes off: the last report, handed back forever, undated.

    The puck at rest sends nothing, so the hidraw queue stays empty and ``poll`` keeps returning
    the state of the last report the operator did produce -- which is not zero unless the puck's
    own resting position falls inside the deadband. Only the frozen timestamp tells the arm that
    nobody is driving any more.

    ``gripper_step`` is the other half of the device: a button held down produces no new reports
    either, and the teleoperator's own state machine keeps moving the gripper while it is held.
    """

    def __init__(self, reading, *, timestamp=100.0, gripper_step=0.0):
        self.reading = dict(reading)
        self.last_report_timestamp = float(timestamp)
        self.gripper_step = float(gripper_step)
        self.polls = 0

    def get_action(self):
        self.polls += 1
        if self.gripper_step:
            self.reading["gripper"] = max(0.0, self.reading["gripper"] - self.gripper_step)
        return dict(self.reading)

    def disconnect(self):
        pass


def test_the_device_is_read_once_per_control_step():
    # Reading it twice measures nothing extra -- the second read is a copy -- and the old drain
    # summed those copies, so a puck off centre by any amount asked for 32x what the hand did.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.001))

    take(ExpertTakeover(device), latched=False)

    assert device.polls == 1


def test_one_reading_covers_the_whole_control_step():
    # The scales are metres per recorder tick at 200 Hz. A 30 Hz loop that applied one reading
    # unscaled would move the arm at a sixth of the speed the same hand produced while recording.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.001))
    gain = motion_gain_for(tick_hz=200.0, step_period_s=1.0 / 30.0)
    takeover = ExpertTakeover(device, motion_gain=gain)

    command, debug = take(takeover, latched=False)

    assert gain == pytest.approx(200.0 / 30.0)
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.001 * gain)
    assert debug["step_mm"] == pytest.approx(1.0 * gain)


def test_holding_the_puck_still_asks_for_the_recorder_speed_not_the_clamp():
    # Full deflection is 0.000615 m/tick * 200 = 0.123 m/s, which at 30 Hz is 4.1 mm per step --
    # under the 5 mm step guard. Every takeover step arriving clamped is the signature of the
    # summing bug, so pin the number that says it is gone.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.000615))
    takeover = ExpertTakeover(device, motion_gain=motion_gain_for(tick_hz=200.0, step_period_s=1.0 / 30.0))

    _, debug = take(takeover, latched=False)

    assert debug["step_mm"] == pytest.approx(4.1, abs=0.05)


def test_gain_of_one_leaves_a_reading_alone():
    # The default. A caller that has not told `ExpertTakeover` its loop rate gets the raw reading
    # rather than a speed silently chosen for it.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.001))

    command, _ = take(ExpertTakeover(device), latched=False)

    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.001)


def test_a_zero_gain_still_reads_the_hand_as_engaged():
    # `moved` is a property of the reading, not of what reaches the arm: a gain that scales the
    # motion away must not read as an operator who let go and hand the arm back mid-correction.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.001))

    _, debug = take(ExpertTakeover(device, motion_gain=0.0), latched=False)

    assert debug["source"] == "expert"
    assert debug["moved"] is True


# --- letting go ---------------------------------------------------------------------------------
#
# The first MuJoCo rehearsal ran one takeover for 1196 steps to the end of the rollout and handed
# back 271 mm from where it engaged, with the operator's hand nowhere near the device. Nothing was
# wrong with the SpaceMouse: it had stopped reporting, exactly as it should, and the driver went on
# handing back the last thing it had said.


def test_a_device_that_has_gone_quiet_stops_moving_the_arm():
    # The bug, in one test. The last report the operator produced is applied once -- it is theirs
    # -- and every read after it is that same report again, which is not a request to keep going.
    gain = motion_gain_for(tick_hz=200.0, step_period_s=1.0 / 30.0)
    device = SilentSpaceMouse(mouse_action(enabled=True, target_x=0.002))
    takeover = ExpertTakeover(device, motion_gain=gain)

    first, _ = take(takeover, latched=False)
    assert first["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002 * gain)

    standing_still = dict(LAST_SENT, **{"ee.x": first["ee.x"]})
    for _ in range(30):
        command, debug = take(takeover, latched=False, previous_sent_command=standing_still)
        assert debug["status"] == "stale"
        assert debug["step_mm"] == pytest.approx(0.0)
        assert command["ee.x"] == pytest.approx(standing_still["ee.x"])


def test_the_arm_goes_back_to_the_policy_after_the_device_goes_quiet():
    # And the other consequence of counting copies as input: they reset the idle timer, so the
    # takeover that could not be moved out of also could never be released.
    clock = FakeClock()
    device = SilentSpaceMouse(mouse_action(enabled=True, target_x=0.002))
    takeover = ExpertTakeover(device, release_after_s=1.0, clock=clock)
    assert take(takeover, latched=False)[1]["source"] == "expert"

    clock.advance(1.1)
    command, debug = take(takeover, latched=False)

    assert debug["source"] == "policy"
    assert command == POLICY_COMMAND


def test_a_puck_held_off_centre_keeps_driving_however_long_it_is_held():
    # The half that must not break: a deflected puck reports continuously, so holding it still is
    # a rate command that goes on meaning what it says. Gating motion on freshness must not turn
    # a steady push into a stall.
    clock = FakeClock()
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.002))
    takeover = ExpertTakeover(device, release_after_s=1.0, clock=clock)

    for _ in range(5):
        clock.advance(0.9)
        command, debug = take(takeover, latched=False)
        assert debug["source"] == "expert"
        assert debug["status"] == "moving"
        assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)


def test_a_held_gripper_button_keeps_the_arm_while_the_puck_says_nothing():
    # The deliberate exemption. A button held down produces no new reports either, so gating the
    # gripper on report freshness would hand the arm back mid-close, with the fingers on the
    # object and the operator still pressing.
    clock = FakeClock()
    device = SilentSpaceMouse(mouse_action(gripper=0.8), gripper_step=0.05)
    takeover = ExpertTakeover(device, release_after_s=1.0, clock=clock)
    take(takeover, latched=False)  # the first reading is a baseline, not a change

    for _ in range(4):
        clock.advance(0.9)
        _, debug = take(takeover, latched=False)
        assert debug["source"] == "expert"
        # Still nothing from the puck: it is the button alone holding the arm.
        assert debug["status"] == "stale"


def test_a_backend_that_does_not_date_its_reports_is_taken_at_its_word():
    # Without a timestamp there is no way to tell a copy from a report, and refusing to act on a
    # reading that might be new would leave the operator pushing a dead puck. Old behaviour, and
    # the reason every scripted double in this file still means what it did.
    device = RepeatingSpaceMouse(mouse_action(enabled=True, target_x=0.002), dated=False)
    takeover = ExpertTakeover(device)

    command, debug = take(takeover, latched=False)

    assert debug["source"] == "expert"
    assert debug["status"] == "moving"
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)
