"""The GT grasp funnel (roadmap v14 step 2): does it put the fingers where P0 says they grasp?

Four things the funnel has to get right, each of which a simpler one gets wrong: the policy's
close is never executed, because a close 20 mm too high is the failure being fixed; the sideways
move never happens at peg height; the close waits for the arm to stop; and every step is recorded
next to what the policy asked for, because those pairs are the later correction data.

The arm here is a first-order lag on the setpoint, so "arrived" and "stopped" are different
moments, as they are on the rig.
"""

import json
import math

import pytest

import tools.fr3.grasp_loop as grasp_loop
import tools.fr3.scene_reset as scene_reset
from tools.fr3.grasp_funnel import ALIGN, CLOSE, DESCEND, FUNNEL_ALIGN_Z_SLACK_M, SEARCH, FunnelConfig, GraspFunnel
from tools.fr3.grasp_loop import GraspHandover, arm_for_trial, completed_trials, run_grasp_loop, summarize_grasp_loop

from tests.scripts.test_fr3_grasp_loop import GraspRig, _put_back, _request

PEG = (0.43, -0.15, 0.058)
ROT = (3.14, 0.0, 0.0)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(grasp_loop, "precise_sleep", lambda seconds: None)


def _command(xyz, gripper=1.0):
    return {
        "ee.x": xyz[0], "ee.y": xyz[1], "ee.z": xyz[2],
        "ee.wx": ROT[0], "ee.wy": ROT[1], "ee.wz": ROT[2],
        "gripper.pos": gripper,
    }


def _simulate(funnel, policy_target, *, start=(0.40, -0.12, 0.20), close_at_z=None, steps=900, lag=0.4, push=None):
    """Drive a lagging arm with the funnel between a straight-line policy and the arm.

    The policy heads for `policy_target` and asks to close once it is below `close_at_z`.
    `push(step, xyz)` may move the arm, which is how a disturbance mid-descent is staged.
    """

    ee = list(start)
    executed = []
    for step in range(steps):
        policy_xyz = tuple(ee[k] + max(-0.004, min(0.004, policy_target[k] - ee[k])) for k in range(3))
        gripper = 0.0 if close_at_z is not None and ee[2] <= close_at_z else 1.0
        command = funnel.step(step, tuple(ee), ROT, _command(policy_xyz, gripper), policy_gripper_raw=gripper)
        executed.append(command)
        for k, key in enumerate(("ee.x", "ee.y", "ee.z")):
            ee[k] += lag * (command[key] - ee[k])
        if push is not None:
            ee = list(push(step, tuple(ee)))
        if funnel.state == CLOSE:
            break
    return executed, tuple(ee)


def test_a_policy_that_closes_high_and_off_centre_is_closed_on_the_peg_instead():
    # The 09-22 failure: 25 mm off in XY, closing at dz +20.
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    target = (PEG[0] + 0.025, PEG[1], PEG[2] + 0.020)
    executed, ee = _simulate(funnel, target, close_at_z=PEG[2] + 0.021)

    assert funnel.state == CLOSE
    # The funnel has the arm at +60 mm, long before the policy's close at +20.
    assert funnel.entryReason == "reached_align_z"
    assert math.hypot(ee[0] - PEG[0], ee[1] - PEG[1]) * 1000 <= 8.0
    assert (ee[2] - PEG[2]) * 1000 == pytest.approx(-6.0, abs=1.0)
    # The policy's close was never executed; only the funnel's, on its last step.
    assert all(c["gripper.pos"] == 1.0 for c in executed[:-1])
    assert executed[-1]["gripper.pos"] == 0.0
    record = funnel.trial_record()
    assert record["xyErrorAtEntryMm"] > 8.0 and record["xyErrorAtCloseMm"] <= 8.0
    assert record["funnelCloseDzMm"] == pytest.approx(-6.0, abs=1.0)


def test_a_policy_that_asks_to_close_above_the_align_height_hands_over_there():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    _simulate(funnel, (PEG[0] + 0.03, PEG[1], PEG[2] + 0.10), close_at_z=PEG[2] + 0.101)
    assert funnel.entryReason == "policy_closed"
    assert funnel.entryXyz[2] > funnel.config.alignZ
    assert funnel.state == CLOSE and funnel.xyErrorAtCloseMm <= 8.0


def test_nothing_moves_sideways_below_the_align_height_while_outside_the_capture_radius():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    # The policy dives straight down 20 mm off the peg, never asking to close.
    target = (PEG[0], PEG[1] + 0.020, PEG[2] + 0.010)
    _simulate(funnel, target, start=(PEG[0], PEG[1] + 0.020, 0.20))
    align_z = funnel.config.alignZ
    for record in funnel.steps:
        if record["funnel_state"] in (ALIGN,) and record["xy_error_mm"] > 8.0 and record["ee_xyz"][2] < align_z - FUNNEL_ALIGN_Z_SLACK_M:
            # Below the align height and outside the radius: the setpoint may only go up.
            assert record["executed_action"][:2] == pytest.approx(record["ee_xyz"][:2], abs=0.004)
    assert funnel.state == CLOSE
    assert funnel.trial_record()["prematureDescend"] is True
    assert funnel.blockedSteps > 0


def test_an_arm_that_parks_short_of_the_align_height_still_moves_across():
    """09-23 trial 9, 09-28 trials 9 and 19: ALIGN entered just under alignZ and 50 mm off; the
    setpoint went up to alignZ, the arm stopped 2.4 mm short of it -- this arm's dead band -- and
    "up first" judged on the measurement waited out the whole budget without moving across."""

    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    align_z = funnel.config.alignZ
    ee = [PEG[0] + 0.05, PEG[1], align_z - 0.0004]
    for step in range(400):
        command = funnel.step(step, tuple(ee), ROT, _command((ee[0], ee[1], ee[2] - 0.004)), policy_gripper_raw=1.0)
        target = (command["ee.x"], command["ee.y"], command["ee.z"])
        ee[0] += 0.4 * (target[0] - ee[0])
        ee[1] += 0.4 * (target[1] - ee[1])
        # Rising, the arm never gets closer than 2.4 mm to where it was sent.
        ee[2] = min(ee[2] + 0.4 * (target[2] - ee[2]), target[2] - 0.0024) if target[2] > ee[2] else target[2]
        if funnel.state == CLOSE:
            break
    assert funnel.state == CLOSE
    assert funnel.xyErrorAtCloseMm is not None and funnel.xyErrorAtCloseMm <= 8.0


def test_the_close_waits_for_the_arm_to_stop():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    _simulate(funnel, (PEG[0], PEG[1], PEG[2] + 0.02), close_at_z=PEG[2] + 0.021, lag=0.15)
    close_index = next(i for i, r in enumerate(funnel.steps) if r["grasp_triggered"])
    window = funnel.steps[close_index - funnel.config.settleSteps : close_index + 1]
    speeds = [
        math.dist(a["ee_xyz"], b["ee_xyz"]) * 1000 / funnel.config.controlPeriodS
        for a, b in zip(window, window[1:])
    ]
    # The recorded positions are rounded to 0.01 mm, which is 0.3 mm/s per axis at 30 Hz.
    assert max(speeds) < funnel.config.settleSpeedMmS + 0.6


def test_leaving_the_capture_radius_during_the_descent_goes_back_to_aligning():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    shoved = []

    def push(step, xyz):
        if funnel.state == DESCEND and not shoved and xyz[2] < PEG[2] + 0.03:
            shoved.append(step)
            return (xyz[0] + 0.015, xyz[1], xyz[2])
        return xyz

    _simulate(funnel, (PEG[0], PEG[1], PEG[2] + 0.02), close_at_z=PEG[2] + 0.021, push=push)
    assert shoved and funnel.realigns == 1
    assert funnel.state == CLOSE and funnel.xyErrorAtCloseMm <= 8.0


def test_the_policy_drives_until_it_reaches_the_align_height():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    policy = (0.41, -0.11, 0.15)
    command = funnel.step(0, (0.40, -0.12, 0.20), ROT, _command(policy, 1.0))
    assert funnel.state == SEARCH and not funnel.active
    assert (command["ee.x"], command["ee.y"], command["ee.z"]) == policy
    # ...but never its close, which is the funnel's.
    command = funnel.step(1, (0.40, -0.12, 0.20), ROT, _command(policy, 0.3))
    assert command["gripper.pos"] == 1.0 and funnel.state == ALIGN


def test_every_step_is_recorded_with_the_v13_schema():
    funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    _simulate(funnel, (PEG[0] + 0.01, PEG[1], PEG[2] + 0.02), close_at_z=PEG[2] + 0.021)
    keys = {
        "funnel_state", "transition_reason", "policy_raw_action", "executed_action", "servo_residual_xy",
        "peg_xy", "peg_source", "ee_xyz", "xy_error_mm", "policy_gripper_raw", "executed_gripper",
        "descend_blocked", "grasp_triggered",
    }
    assert all(keys <= set(record) for record in funnel.steps)
    assert json.loads(json.dumps(funnel.steps)) == funnel.steps
    assert [r["transition_reason"] for r in funnel.steps if r["transition_reason"]] == [
        "reached_align_z", "aligned", "settled_at_close_height",
    ]


# ------------------------------------------------------------------------------- loop ---


def test_interleaved_arms_put_one_of_each_in_every_pair_and_a_resume_agrees():
    request = _request(arms="AB", seed=3)
    arms = [arm_for_trial(request, t) for t in range(40)]
    assert all(sorted(arms[i : i + 2]) == ["A", "B"] for i in range(0, 40, 2))
    assert arms == [arm_for_trial(request, t) for t in range(40)]
    assert arms[0::2] != ["A"] * 20  # the order within a pair is drawn, not fixed
    assert arm_for_trial(_request(arms="B"), 5) == "B"


def test_the_funnel_budget_starts_when_the_funnel_takes_the_arm():
    handover = GraspHandover(maxPolicySteps=100, funnelMaxSteps=50)
    handover.funnel = GraspFunnel(FunnelConfig(pegXyz=PEG))
    assert handover.timed_out(100)
    handover.funnel.step(90, (PEG[0], PEG[1], 0.10), ROT, _command((PEG[0], PEG[1], 0.10)))
    assert handover.funnel.active
    assert not handover.timed_out(120) and handover.timed_out(140)


def _funnel_aware_policy(robot, miss_m):
    """The loop's stand-in policy, routed through the funnel when the trial carries one."""

    def run(trial, handover):
        px, py, pz = robot.peg_xyz
        aim = (px + miss_m, py, pz + 0.020)
        for step in range(600):
            xyz = robot.xyz
            policy_xyz = tuple(xyz[k] + max(-0.004, min(0.004, aim[k] - xyz[k])) for k in range(3))
            gripper = 0.0 if abs(xyz[2] - aim[2]) < 0.001 else 1.0
            command = _command(policy_xyz, gripper)
            if handover.funnel is not None:
                command = handover.funnel.step(step, xyz, robot.rotvec, command)
            robot.send_action(command)
            if handover.observe(step, robot.xyz, command["gripper.pos"]):
                return "grasp_handover"
            if handover.timed_out(step):
                return "grasp_timeout"
        return "grasp_timeout"

    return run


def test_a_policy_that_misses_by_20_mm_fails_alone_and_is_held_with_the_funnel(tmp_path):
    robot = GraspRig()
    robot.reach_m = 0.010
    out = tmp_path / "grasp.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=6, arms="AB", seed=1),
        run_policy_trial=_funnel_aware_policy(robot, 0.020),
        out_path=out,
        # Every A trial misses, and a missed peg goes back to the fixture by hand.
        wait_for_operator=_put_back(robot, []),
    )
    rows = completed_trials(out)
    by_arm = {arm: [r["verdict"] for r in rows if r["arm"] == arm] for arm in "AB"}
    assert by_arm["B"] == ["held"] * 3
    assert "held" not in by_arm["A"]
    b = [r for r in rows if r["arm"] == "B"]
    assert all(r["xyErrorAtCloseMm"] <= 8.0 and r["closeAboveTargetMm"] == pytest.approx(-6.0, abs=1.0) for r in b)
    # The correction data is written beside the row file, one file per funnel trial.
    for row in b:
        lines = open(row["funnelSteps"], encoding="utf-8").read().splitlines()
        # Steps go on after the close until the handover's settle steps have passed.
        records = [json.loads(line) for line in lines]
        assert sum(r["grasp_triggered"] for r in records) == 1
        assert records[-1]["funnel_state"] == CLOSE and records[-1]["executed_gripper"] == 0.0
    summary = result["summary"]["byArm"]
    assert summary["B"]["held"] == 3 and summary["A"]["held"] == 0
    assert summary["B"]["funnel"]["entered"] == 3


def test_a_single_arm_run_still_summarises_under_by_arm():
    rows = [{"verdict": "held"}, {"verdict": "empty"}]
    summary = summarize_grasp_loop(rows)
    assert summary["held"] == 1 and summary["byArm"]["A"]["graded"] == 2


def test_a_high_takeover_descends_fast_to_60_mm_then_slow_and_still_closes_on_the_peg():
    cfg = FunnelConfig(pegXyz=PEG, alignDzMm=250.0)
    funnel = GraspFunnel(cfg)
    executed, final = _simulate(funnel, (PEG[0] + 0.05, PEG[1] - 0.04, PEG[2]), start=(0.40, -0.12, 0.40), lag=0.5)
    assert funnel.state == CLOSE and funnel.entryReason == "reached_align_z"
    assert funnel.entryXyz[2] <= cfg.alignZ + 1e-9
    assert math.hypot(final[0] - PEG[0], final[1] - PEG[1]) * 1000.0 <= cfg.captureXyMm
    descend = [s for s in funnel.steps if s["funnel_state"] == DESCEND]
    steps = [
        (a["executed_action"][2], a["executed_action"][2] - b["executed_action"][2])
        for a, b in zip(descend, descend[1:])
    ]
    fast = cfg.fastDescendSpeedMS * cfg.controlPeriodS
    slow = cfg.descendSpeedMS * cfg.controlPeriodS
    above = [dz for z, dz in steps if z > cfg.slowBelowZ + fast]
    below = [dz for z, dz in steps if z < cfg.slowBelowZ and dz > 1e-9]
    # The step log rounds to 0.01 mm, hence the tolerance.
    assert above and all(dz == pytest.approx(fast, abs=2e-5) for dz in above)
    assert below and all(dz <= slow + 2e-5 for dz in below)
    assert funnel.trial_record()["funnelAlignDzMm"] == 250.0


def test_a_60_mm_takeover_descends_at_the_terminal_speed_throughout():
    cfg = FunnelConfig(pegXyz=PEG)
    funnel = GraspFunnel(cfg)
    _simulate(funnel, (PEG[0] + 0.02, PEG[1], PEG[2]), start=(0.40, -0.12, 0.20))
    descend = [s["executed_action"][2] for s in funnel.steps if s["funnel_state"] == DESCEND]
    drops = [a - b for a, b in zip(descend, descend[1:]) if a - b > 1e-9]
    assert drops and max(drops) <= cfg.descendSpeedMS * cfg.controlPeriodS + 2e-5
