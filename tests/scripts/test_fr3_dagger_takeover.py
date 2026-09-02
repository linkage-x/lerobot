"""What happens to the arm when the operator takes it over mid-rollout, and hands it back.

None of this needs a robot or a SpaceMouse: the takeover is a function from (last command,
device reading) to the next command, and that is where every decision worth checking lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.utils.rotation import Rotation

from tools.fr3.dagger_takeover import (
    MAX_STEP_PERIOD_RATIO,
    ExpertTakeover,
    backend_dates_reports,
    expert_spans,
    motion_gain_for,
    undated_backend_error,
)

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


# --- moving the arm at the speed the recorder moved it, and at the time the hand moved it -------


class QueuedSpaceMouse:
    """The driver, as it behaves: one queued report per read, then copies of the last one.

    ``PySpaceMouseDriver.poll`` takes a single report out of the kernel's hidraw queue per call,
    and once that queue is empty it returns the device's cached state instead -- carrying the
    timestamp it already had, which is the only way to tell the copy from a report.

    ``arrive()`` is the device putting reports into the queue: what the puck does between two
    control steps, which is about four reports at 30 Hz while it is displaced, and none at all
    once the operator's hand comes off.
    """

    def __init__(self, *arrivals, gripper_step=0.0):
        self._queue = [dict(reading) for reading in arrivals]
        self._current = mouse_action()
        self.last_report_timestamp = 100.0
        self.gripper_step = float(gripper_step)
        self.polls = 0

    def arrive(self, *readings, times=1):
        for _ in range(times):
            self._queue.extend(dict(reading) for reading in readings)
        return self

    def get_action(self):
        self.polls += 1
        if self._queue:
            self._current = self._queue.pop(0)
            self.last_report_timestamp += 1.0 / 126.0
        elif self.gripper_step:
            # The teleoperator's own state machine, which needs no report: a button held down
            # keeps moving the gripper, and moves it on the copies too.
            self._current["gripper"] = max(0.0, self._current["gripper"] - self.gripper_step)
        return dict(self._current)

    def disconnect(self):
        pass


class UndatedSpaceMouse:
    """A backend that does not timestamp its reports: every read might be new, and none can be
    told apart. The same reading forever is all it offers."""

    def __init__(self, reading):
        self.reading = dict(reading)
        self.polls = 0

    def get_action(self):
        self.polls += 1
        return dict(self.reading)

    def disconnect(self):
        pass


def test_the_queue_is_emptied_every_step_and_the_newest_report_is_the_one_that_moves_the_arm():
    # Both halves of the read, in one test. The device sends ~126 reports a second against a loop
    # at 30, so a step that reads once is steering the arm with a hand from several steps ago --
    # and after the operator lets go, with the tail of a queue nobody is adding to any more. But
    # the drain must not add up what it passed over: this is a rate control, and four reports of
    # 1 mm is a hand asking for 1 mm, not for 4.
    device = QueuedSpaceMouse(
        mouse_action(enabled=True, target_x=0.001),
        mouse_action(enabled=True, target_x=0.001),
        mouse_action(enabled=True, target_x=0.001),
        mouse_action(enabled=True, target_x=0.004),
    )

    command, debug = take(ExpertTakeover(device), latched=False)

    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.004)
    assert debug["step_mm"] == pytest.approx(4.0)
    # Four reports plus the read that finds the queue empty.
    assert device.polls == 5
    assert debug["reads"] == 5


def test_one_reading_covers_the_whole_control_step():
    # The scales are metres per recorder tick at 200 Hz. A 30 Hz loop that applied one reading
    # unscaled would move the arm at a sixth of the speed the same hand produced while recording.
    device = QueuedSpaceMouse(mouse_action(enabled=True, target_x=0.001))
    gain = motion_gain_for(tick_hz=200.0, step_period_s=1.0 / 30.0)
    takeover = ExpertTakeover(device, motion_gain=gain)

    command, debug = take(takeover, latched=False)

    assert gain == pytest.approx(200.0 / 30.0)
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.001 * gain)
    assert debug["step_mm"] == pytest.approx(1.0 * gain)


def test_holding_the_puck_still_asks_for_the_recorder_speed_not_the_clamp():
    # Full deflection is 0.000615 m/tick * 200 = 0.123 m/s, which at 30 Hz is 4.1 mm per step --
    # under the 5 mm step guard. Every takeover step arriving clamped is the signature of the
    # summing bug, so pin the number that says it is gone: four reports of full deflection in one
    # step is still one step of full deflection.
    device = QueuedSpaceMouse().arrive(mouse_action(enabled=True, target_x=0.000615), times=4)
    takeover = ExpertTakeover(device, motion_gain=motion_gain_for(tick_hz=200.0, step_period_s=1.0 / 30.0))

    _, debug = take(takeover, latched=False)

    assert debug["step_mm"] == pytest.approx(4.1, abs=0.05)


def test_gain_of_one_leaves_a_reading_alone():
    # The default. A caller that has not told `ExpertTakeover` its loop rate gets the raw reading
    # rather than a speed silently chosen for it.
    device = QueuedSpaceMouse(mouse_action(enabled=True, target_x=0.001))

    command, _ = take(ExpertTakeover(device), latched=False)

    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.001)


def test_a_zero_gain_still_reads_the_hand_as_engaged():
    # `moved` is a property of the reading, not of what reaches the arm: a gain that scales the
    # motion away must not read as an operator who let go and hand the arm back mid-correction.
    device = QueuedSpaceMouse(mouse_action(enabled=True, target_x=0.001))

    _, debug = take(ExpertTakeover(device, motion_gain=0.0), latched=False)

    assert debug["source"] == "expert"
    assert debug["moved"] is True


# --- moving at the recorded speed on a loop that misses its rate ---------------------------------
#
# The puck is a velocity: full deflection is 0.000615 m per 200 Hz tick, so 0.123 m/s, and that is
# what it meant for every frame in the dataset. A gain sized from the loop's *nominal* period only
# reproduces that while the loop keeps its nominal period. On the rig a step also carries two
# camera reads and a forward pass, and the operator has no way to ask for the missing speed --
# they are already against the stop.

FULL_DEFLECTION_M = 0.000615
RECORDED_SPEED_MM_S = FULL_DEFLECTION_M * 200.0 * 1000.0  # 123 mm/s


def drive_at_full_deflection(takeover, device, clock, *, seconds):
    """One control step: the reports that arrived during it, then the step itself."""
    device.arrive(mouse_action(enabled=True, target_x=FULL_DEFLECTION_M))
    clock.advance(seconds)
    return take(takeover, latched=False)[1]


def full_deflection_takeover(clock, *, nominal_hz=60.0):
    device = QueuedSpaceMouse()
    takeover = ExpertTakeover(
        device,
        motion_gain=motion_gain_for(tick_hz=200.0, step_period_s=1.0 / nominal_hz),
        step_period_s=1.0 / nominal_hz,
        clock=clock,
    )
    return device, takeover


def test_a_loop_that_keeps_its_rate_moves_the_arm_at_the_recorded_speed():
    # The number this whole section is about, in the units the operator feels it in.
    clock = FakeClock()
    device, takeover = full_deflection_takeover(clock)

    drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 60.0)
    debug = drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 60.0)

    assert debug["gain"] == pytest.approx(200.0 / 60.0)
    assert debug["step_mm"] * 60.0 == pytest.approx(RECORDED_SPEED_MM_S, abs=0.5)


def test_a_slow_step_moves_the_arm_further_so_the_speed_stays_the_one_the_hand_asked_for():
    # A loop settled at 30 Hz against a 60 Hz nominal. Each step is held twice as long, so it has
    # to cover twice the ground -- otherwise the same hand drives the arm at half the speed it
    # drove it while recording, which is the one thing the operator cannot compensate for.
    clock = FakeClock()
    device, takeover = full_deflection_takeover(clock)

    drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 30.0)
    debug = drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 30.0)

    assert debug["gain"] == pytest.approx(200.0 / 30.0)
    assert debug["step_ms"] == pytest.approx(1000.0 / 30.0)
    assert debug["step_mm"] == pytest.approx(4.1, abs=0.05)
    # The invariant, not the millimetres: the speed is the same one the 60 Hz loop produced.
    assert debug["step_mm"] * 30.0 == pytest.approx(RECORDED_SPEED_MM_S, abs=0.5)


def test_a_stalled_step_arrives_as_a_slow_step_rather_than_as_a_lunge():
    # A second of dead time is not a control step, and honouring it literally would ask for 123 mm
    # in one command -- with the operator's hand still where it was before the freeze. The bound
    # is what makes the correction safe to apply without knowing why the loop stopped.
    clock = FakeClock()
    device, takeover = full_deflection_takeover(clock)

    drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 60.0)
    debug = drive_at_full_deflection(takeover, device, clock, seconds=1.0)

    assert debug["gain"] == pytest.approx(MAX_STEP_PERIOD_RATIO * 200.0 / 60.0)
    assert debug["step_mm"] == pytest.approx(4.0 * 2.05, abs=0.05)


def test_the_first_step_of_a_run_is_taken_at_the_nominal_rate():
    # There is no previous step to measure, and the gap since the last rollout -- homing, waiting
    # for the operator to press s -- is not one. Nominal is the honest answer, not a measurement.
    clock = FakeClock()
    device, takeover = full_deflection_takeover(clock)
    clock.advance(45.0)

    debug = drive_at_full_deflection(takeover, device, clock, seconds=0.0)

    assert debug["gain"] == pytest.approx(200.0 / 60.0)
    assert debug["step_ms"] == pytest.approx(0.0)


def test_a_caller_that_names_no_step_period_keeps_the_gain_it_was_given():
    # Sim replays, tests, anything whose loop rate is not a measured quantity. Correcting a gain
    # by a clock nobody paced against would be inventing a speed, not reproducing one.
    clock = FakeClock()
    device = QueuedSpaceMouse()
    takeover = ExpertTakeover(device, motion_gain=2.0, clock=clock)

    drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 60.0)
    debug = drive_at_full_deflection(takeover, device, clock, seconds=1.0 / 6.0)

    assert debug["gain"] == pytest.approx(2.0)
    assert debug["step_mm"] == pytest.approx(FULL_DEFLECTION_M * 2.0 * 1000.0)


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
    device = QueuedSpaceMouse(mouse_action(enabled=True, target_x=0.002))
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
    device = QueuedSpaceMouse(mouse_action(enabled=True, target_x=0.002))
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
    device = QueuedSpaceMouse()
    takeover = ExpertTakeover(device, release_after_s=1.0, clock=clock)

    for _ in range(5):
        clock.advance(0.9)
        device.arrive(mouse_action(enabled=True, target_x=0.002), times=4)
        command, debug = take(takeover, latched=False)
        assert debug["source"] == "expert"
        assert debug["status"] == "moving"
        assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)


def test_a_held_gripper_button_keeps_the_arm_while_the_puck_says_nothing():
    # The deliberate exemption. A button held down produces no new reports either, so gating the
    # gripper on report freshness would hand the arm back mid-close, with the fingers on the
    # object and the operator still pressing.
    clock = FakeClock()
    device = QueuedSpaceMouse(gripper_step=0.05)
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
    # reading that might be new would leave the operator pushing a dead puck. Old behaviour, on
    # purpose -- including the single read, because draining an undated device would sum copies.
    device = UndatedSpaceMouse(mouse_action(enabled=True, target_x=0.002))
    takeover = ExpertTakeover(device)

    command, debug = take(takeover, latched=False)

    assert debug["source"] == "expert"
    assert debug["status"] == "moving"
    assert device.polls == 1
    assert command["ee.x"] == pytest.approx(LAST_SENT["ee.x"] + 0.002)


class ConnectedButSilentSpaceMouse(QueuedSpaceMouse):
    """A patched driver at the moment it connects: dated, but nothing has arrived yet.

    This is the state every rig is in when the banner prints -- the operator's hand is not on the
    puck -- and it is the case a check on the *value* of the timestamp gets backwards.
    """

    def __init__(self):
        super().__init__()
        self.last_report_timestamp = None


def test_a_driver_that_carries_no_timestamp_at_all_is_the_undated_one():
    assert backend_dates_reports(QueuedSpaceMouse()) is True
    assert backend_dates_reports(UndatedSpaceMouse(mouse_action())) is False


def test_a_dated_driver_that_has_not_heard_from_the_device_yet_still_counts_as_dated():
    # The refusal fires at connect, before any report exists. Reading the timestamp's value here
    # would refuse every correctly patched rig on the rig's normal startup state.
    device = ConnectedButSilentSpaceMouse()

    assert device.last_report_timestamp is None
    assert backend_dates_reports(device) is True


def test_the_takeover_says_whether_its_own_device_dates_its_reports():
    # Asked of the object because the answer changes what its other numbers mean: a handback gap
    # measured on an undated device is the runaway's length, not the operator's excursion.
    assert ExpertTakeover(QueuedSpaceMouse()).dates_reports is True
    assert ExpertTakeover(UndatedSpaceMouse(mouse_action())).dates_reports is False


def test_the_refusal_explains_the_import_rather_than_blaming_the_device():
    # The device is almost never the cause -- the wrong copy of lerobot is -- so the message has
    # to carry the path that was loaded and the variable that decides it.
    message = undated_backend_error(driver_module="/home/hanyu/Codes/lerobot/src/lerobot/x.py")

    assert "/home/hanyu/Codes/lerobot/src/lerobot/x.py" in message
    assert "PYTHONPATH" in message
    # And it must not read as "the rehearsal is broken too": that path is deliberately allowed.
    assert "dagger_sim" in message
