"""Two action sources, one guard: what the arm is actually sent across a takeover.

``test_fr3_dagger_takeover.py`` checks what the operator's device turns into. This file checks
what happens to that command afterwards -- through the same EMA and the same step/leash clamp the
rollout runs -- because the failure this whole path has to avoid lives in the seam between them:
a handoff that steps further in one tick than the policy is ever allowed to.

The MuJoCo rehearsal (``tools/fr3/dagger_sim_dryrun.py``) runs exactly this chain against a
simulated arm. What it adds is timing and a human hand; what it cannot add is a reason to leave
the arithmetic untested until then.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.fr3.command_guard import (
    limit_command_for_safety,
    observation_with_prev_cmd,
    smooth_robot_command_ema,
)
from tools.fr3.dagger_takeover import ExpertTakeover

MAX_STEP_POS_M = 0.006
MAX_STEP_ROT_RAD = float(np.deg2rad(3.0))
MAX_LEASH_POS_M = 0.060
MAX_LEASH_ROT_RAD = float(np.deg2rad(25.0))


class FakeSpaceMouse:
    """One report per call, then quiet -- this double's convention, not the device's.

    The hardware repeats its last state instead of reporting empty, which is why
    ``ExpertTakeover`` reads once per step and scales that reading rather than draining. These
    tests are about *when* the arm changes hands, so one scripted report per step is the clearest
    way to say "the operator did something on this step"; the rate model is pinned in
    ``test_fr3_dagger_takeover.py``.
    """

    def __init__(self, *readings):
        self._readings = list(readings)
        self._gripper = mouse()["gripper"]
        self.disconnected = False

    def get_action(self):
        if not self._readings:
            return mouse(gripper=self._gripper)
        reading = self._readings.pop(0)
        self._gripper = reading["gripper"]
        return reading

    def disconnect(self):
        self.disconnected = True


def mouse(**overrides):
    action = {
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


def command(x=0.400, y=-0.100, z=0.070, gripper=1.0):
    return {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "gripper.pos": gripper,
    }


def observation(x=0.395, y=-0.099, z=0.068, gripper=1.0):
    """Where the arm got to. Behind the command it is chasing, as a moving arm always is."""
    return {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "gripper.pos": gripper,
    }


def send(robot_command, robot_observation, previous_sent_command):
    """One pass of the loop's command path: prev_cmd reference, then clamp."""
    guard_observation = observation_with_prev_cmd(robot_observation, previous_sent_command)
    return limit_command_for_safety(
        robot_command,
        guard_observation,
        max_step_pos_delta_m=MAX_STEP_POS_M,
        max_step_rot_delta_rad=MAX_STEP_ROT_RAD,
        max_leash_pos_delta_m=MAX_LEASH_POS_M,
        max_leash_rot_delta_rad=MAX_LEASH_ROT_RAD,
    )


def position(item):
    return np.array([item["ee.x"], item["ee.y"], item["ee.z"]], dtype=np.float64)


def test_the_instant_control_changes_hands_the_arm_is_sent_nowhere_new():
    """The operator reaches for the SpaceMouse *because* something is going wrong, often with the
    gripper already on the object. Engaging must not itself be a motion command."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse()))
    last_sent = command()

    engaged, _ = takeover.command(
        latched=True,
        policy_command=command(x=0.420),
        previous_sent_command=last_sent,
        robot_observation=observation(),
    )
    sent, guard = send(engaged, observation(), last_sent)

    assert position(sent) == pytest.approx(position(last_sent))
    assert guard["status"] == "pass"
    assert float(np.linalg.norm(guard["step_position_delta"])) == pytest.approx(0.0)


def test_the_operator_is_bounded_by_the_same_step_limit_as_the_policy():
    """A SpaceMouse held over hard is a large delta per tick. Two sources, one limit: otherwise
    the rehearsal would be verifying a guard that only applies to one of them."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse(target_x=0.05)))
    last_sent = command()

    engaged, _ = takeover.command(
        latched=True,
        policy_command=command(),
        previous_sent_command=last_sent,
        robot_observation=observation(),
    )
    sent, guard = send(engaged, observation(), last_sent)

    assert guard["status"] == "step_limited"
    assert float(np.linalg.norm(position(sent) - position(last_sent))) == pytest.approx(
        MAX_STEP_POS_M, abs=1e-9
    )


def test_handing_back_from_somewhere_the_policy_never_went_is_walked_back_not_lunged():
    """The rehearsal's point, in one test. The operator moves the arm 50 mm; the source that
    resumes has a command from before that, so the gap is real. The clamp turns it into a walk of
    step-sized ticks -- with no prev_cmd reference it would be a single 50 mm command instead."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse()))
    corrected = command(x=0.450)  # where the operator left the arm
    takeover.command(
        latched=True,
        policy_command=command(),
        previous_sent_command=corrected,
        robot_observation=observation(x=0.449),
    )

    resumed = command(x=0.400)  # the frozen source, 50 mm behind
    handback, _ = takeover.command(
        latched=False,
        policy_command=resumed,
        previous_sent_command=corrected,
        robot_observation=observation(x=0.449),
    )
    sent, guard = send(handback, observation(x=0.449), corrected)

    assert handback == resumed  # released: the policy's command, untouched
    assert guard["status"] == "step_limited"
    step_mm = float(np.linalg.norm(position(sent) - position(corrected))) * 1000.0
    assert step_mm == pytest.approx(MAX_STEP_POS_M * 1000.0, abs=1e-6)


def test_a_walk_back_takes_the_number_of_ticks_the_gap_divided_by_the_limit_says():
    """Not a tautology: it is the check that the reference *advances*. Anchored to the measured
    pose instead, a lagging arm would let each tick start from further back and the walk would
    never converge."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse()))
    target = command(x=0.400)
    sent = command(x=0.430)  # 30 mm to reconcile, at 6 mm a tick
    ticks = 0

    while abs(sent["ee.x"] - target["ee.x"]) > 1e-6 and ticks < 20:
        handback, _ = takeover.command(
            latched=False,
            policy_command=target,
            previous_sent_command=sent,
            robot_observation=observation(x=sent["ee.x"] - 0.001),
        )
        sent, _ = send(handback, observation(x=sent["ee.x"] - 0.001), sent)
        ticks += 1

    assert ticks == 5
    assert sent["ee.x"] == pytest.approx(0.400)


def test_the_smoothing_filter_does_not_restart_at_the_handoff():
    """The EMA carries the same previous command across the boundary, so the first expert tick is
    filtered like any other. A filter reset at the handoff would put a step change through a path
    whose whole purpose is to have none."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse(target_x=0.01)))
    last_sent = command()

    engaged, _ = takeover.command(
        latched=True,
        policy_command=command(),
        previous_sent_command=last_sent,
        robot_observation=observation(),
    )
    smoothed = smooth_robot_command_ema(engaged, last_sent, alpha=0.5)

    assert smoothed["ee.x"] == pytest.approx(0.405)


def test_without_the_prev_cmd_reference_the_step_guard_measures_the_wrong_thing():
    """Why the rehearsal writes its own last command into the observation: the simulated arm does
    not report prev_cmd, and the fallback measures from the *measured* pose. That silently folds
    servo lag into the step, so the same command is clamped differently in the rehearsal than on
    hardware -- and the rehearsal would be rehearsing a guard nobody runs."""
    lagging = observation(x=0.380)  # 20 mm of tracking lag
    asked = command(x=0.402)  # 2 mm past the last command: well inside the step limit

    _, with_reference = send(asked, lagging, command())
    _, without_reference = limit_command_for_safety(
        asked,
        lagging,
        max_step_pos_delta_m=MAX_STEP_POS_M,
        max_step_rot_delta_rad=MAX_STEP_ROT_RAD,
        max_leash_pos_delta_m=MAX_LEASH_POS_M,
        max_leash_rot_delta_rad=MAX_LEASH_ROT_RAD,
    )

    assert with_reference["status"] == "pass"
    assert with_reference["has_prev_cmd_reference"] is True
    assert without_reference["status"] == "step_limited"
    assert without_reference["has_prev_cmd_reference"] is False


def test_the_gripper_the_policy_asked_for_survives_the_clamp():
    """The clamp bounds pose, not grip. An operator who has taken over with the object between
    the fingers must not have the hold rewritten by a guard about how far the arm may move."""
    takeover = ExpertTakeover(FakeSpaceMouse(mouse(gripper=0.8)))
    last_sent = command(gripper=0.0)

    engaged, _ = takeover.command(
        latched=True,
        policy_command=command(x=0.500, gripper=1.0),
        previous_sent_command=last_sent,
        robot_observation=observation(gripper=0.02),
    )
    sent, guard = send(engaged, observation(gripper=0.02), last_sent)

    assert guard["status"] == "pass"
    assert sent["gripper.pos"] == 0.0


def test_the_handback_the_operator_never_asked_for_is_walked_back_like_any_other():
    """Automatic handback makes this the common case rather than the rare one. Every pause of
    more than a second now ends a takeover, so the walk back from wherever the operator left the
    arm happens many times in a rollout instead of once -- through the same clamp, at the same
    six millimetres a tick, with no key pressed to cause it."""

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    takeover = ExpertTakeover(FakeSpaceMouse(mouse(target_x=0.03), mouse()), clock=clock)
    last_sent = command()

    corrected, debug = takeover.command(
        latched=False,
        policy_command=command(),
        previous_sent_command=last_sent,
        robot_observation=observation(),
    )
    assert debug["source"] == "expert"
    sent, _ = send(corrected, observation(), last_sent)

    clock.now = 1.5  # the operator's hand comes off the device
    resumed, debug = takeover.command(
        latched=False,
        policy_command=command(),
        previous_sent_command=sent,
        robot_observation=observation(x=sent["ee.x"] - 0.001),
    )
    handback, guard = send(resumed, observation(x=sent["ee.x"] - 0.001), sent)

    assert debug["source"] == "policy"
    assert guard["status"] == "step_limited"
    assert float(np.linalg.norm(position(handback) - position(sent))) == pytest.approx(
        MAX_STEP_POS_M, abs=1e-9
    )
