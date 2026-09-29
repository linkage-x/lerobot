"""The grasp-only loop: does it grade by the lift, and does it always know where the peg is?

The loop is only worth running unattended if two things hold. Its verdict has to be the one the
09-22 hand grades agree with -- width *after* a lift, never at the close -- and every trial has
to start from a peg whose position the loop actually knows, or the next reset closes on air and
the run fills with rows that measure the staging instead of the policy.

The rig fake is the scene-reset suite's arm with a peg on the table: fingers that close within
reach of the peg take it, and it moves with them until they open.
"""

import json
import math

import numpy as np

import pytest

import tools.fr3.grasp_loop as grasp_loop
import tools.fr3.scene_reset as scene_reset
from tools.fr3.grasp_loop import (
    GraspHandover,
    GraspLoopRequest,
    completed_trials,
    grasp_handover_due,
    load_mask_strokes,
    run_grasp_loop,
    summarize_grasp_loop,
    validate_grasp_loop_request,
    wilson_interval,
)
from tools.fr3.scene_reset import SceneResetError, SceneResetStroke

from tests.scripts.test_fr3_scene_reset import FakeRobot


PICK = (0.3640, -0.1370, 0.0550)
TARGET_Z = 0.058


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(grasp_loop, "precise_sleep", lambda seconds: None)


class GraspRig(FakeRobot):
    """A peg on a table. Widths are the 09-22 readings: 0.31 clamped, 0.006 shut on air."""

    HELD_WIDTH = 0.31
    EMPTY_WIDTH = 0.006

    def __init__(self, peg_xyz=PICK):
        super().__init__()
        self.peg_xyz = tuple(peg_xyz)
        self.held = False
        self.closed = False
        self.reach_m = 0.005
        # A standing peg can be taken anywhere along its length; sideways it has to be between
        # the fingers.
        self.reach_z_m = 0.012

    def send_action(self, action):
        result = super().send_action(action)
        commanded = float(action["gripper.pos"])
        if commanded >= 0.5:
            if self.held:
                self.peg_xyz = tuple(self.xyz)
            self.held = self.closed = False
            self.gripper = commanded
            return result
        if not self.closed:
            self.closed = True
            self.held = (
                math.dist(self.xyz[:2], self.peg_xyz[:2]) <= self.reach_m
                and abs(self.xyz[2] - self.peg_xyz[2]) <= self.reach_z_m
            )
        if self.held:
            self.peg_xyz = tuple(self.xyz)
        self.gripper = self.HELD_WIDTH if self.held else self.EMPTY_WIDTH
        return result


def _request(**overrides):
    fields = dict(
        pickXyz=PICK,
        targetZ=TARGET_Z,
        strokes=(SceneResetStroke(x=0.43, y=-0.15, radiusM=0.03),),
        trials=3,
        closeSettleSteps=3,
        seed=7,
    )
    fields.update(overrides)
    return GraspLoopRequest(**fields)


def _policy(robot, plan):
    """A stand-in policy: per trial, either reach the peg (with an XY miss) and close, or wander.

    `plan` items: ("grasp", miss_m), ("knock", None) -- closes on air after shoving the peg away --
    ("air", (sideways_m, above_m)) -- closes on air clear of the peg -- or ("wander", None) --
    never closes.
    """

    plan = list(plan)

    def run(trial, handover: GraspHandover) -> str:
        kind, miss = plan[trial]
        rotvec = robot.rotvec
        if kind == "knock":
            robot.peg_xyz = (robot.peg_xyz[0] + 0.05, robot.peg_xyz[1], robot.peg_xyz[2])
            kind, miss = "grasp", 0.02
        if kind == "wander":
            for step in range(handover.maxPolicySteps + 1):
                if step >= handover.maxPolicySteps:
                    return "grasp_timeout"
                handover.observe(step, robot.xyz, 1.0)
            return "grasp_timeout"
        px, py, pz = robot.peg_xyz
        if kind == "air":
            # Closes in mid-air: `miss` is (sideways, above) the peg.
            at = (px + miss[0], py, pz + miss[1])
        else:
            at = (px + miss, py, pz)
        for step in range(100):
            gripper = 1.0 if step < 5 else 0.2
            robot.send_action({
                "ee.x": at[0], "ee.y": at[1], "ee.z": at[2],
                "ee.wx": rotvec[0], "ee.wy": rotvec[1], "ee.wz": rotvec[2],
                "gripper.pos": gripper,
            })
            if handover.observe(step, robot.xyz, gripper):
                return "grasp_handover"
        return "grasp_timeout"

    return run


def _trials(path):
    return completed_trials(path)


# ---------------------------------------------------------------------------- trigger ---


def test_a_two_step_blip_does_not_hand_over_and_does_not_leave_its_height_behind():
    handover = GraspHandover(closeSettleSteps=3)
    assert not handover.observe(0, (0.4, 0.0, 0.10), 0.4997)
    assert not handover.observe(1, (0.4, 0.0, 0.10), 0.4997)
    assert not handover.observe(2, (0.4, 0.0, 0.10), 0.9)
    assert not handover.observe(3, (0.4, 0.0, 0.07), 0.2)
    assert not handover.observe(4, (0.4, 0.0, 0.07), 0.2)
    assert handover.observe(5, (0.4, 0.0, 0.08), 0.2)
    # The closure height is where the held close began, not the blip and not the handover.
    assert handover.closeStep == 3
    assert handover.closeXyz == (0.4, 0.0, 0.07)
    assert handover.handoverStep == 5
    assert handover.commandedGripper == 0.2


def test_the_trigger_reads_the_command_threshold_strictly():
    assert grasp_handover_due(4, commanded_gripper=0.5, closed_below=0.5, settle_steps=3) == (0, False)
    assert grasp_handover_due(2, commanded_gripper=0.49, closed_below=0.5, settle_steps=3) == (3, True)


# ------------------------------------------------------------------------------- mask ---


def test_the_mask_can_be_rebuilt_from_the_reset_targets_a_rollout_log_recorded(tmp_path):
    log = tmp_path / "rollout_log.jsonl"
    log.write_text(
        "\n".join([
            json.dumps({"rolloutIndex": 1, "resetTarget": [0.47, -0.12, 0.058]}),
            json.dumps({"rolloutIndex": 2}),
            "not json",
            json.dumps({"rolloutIndex": 3, "resetTarget": [0.36, -0.25, 0.058]}),
        ])
    )
    strokes = load_mask_strokes(log, radius_m=0.01)
    assert [(s.x, s.y, s.radiusM) for s in strokes] == [(0.47, -0.12, 0.01), (0.36, -0.25, 0.01)]


def test_the_gateway_mask_format_loads_as_is(tmp_path):
    mask = tmp_path / "scene_reset_mask.json"
    mask.write_text(json.dumps({"strokes": [{"x": 0.4, "y": -0.1, "radiusM": 0.02}], "updatedAt": "x"}))
    assert load_mask_strokes(mask) == (SceneResetStroke(x=0.4, y=-0.1, radiusM=0.02),)


def test_a_mask_entirely_outside_the_fence_is_refused_before_anything_moves():
    request = _request(strokes=(SceneResetStroke(x=0.9, y=0.0, radiusM=0.01),))
    with pytest.raises(SceneResetError, match="no stroke"):
        validate_grasp_loop_request(request, workspace_min=(0.18, -0.45, 0.0), workspace_max=(0.70, 0.45, 0.70))


# ------------------------------------------------------------------------------- loop ---


def _put_back(robot, asked):
    def operator(message):
        asked.append(message)
        robot.peg_xyz = PICK
        return True

    return operator


def test_a_held_peg_is_regripped_the_scripts_way_and_a_missed_one_goes_back_to_the_fixture(tmp_path):
    robot = GraspRig()
    out = tmp_path / "grasp.jsonl"
    asked = []
    result = run_grasp_loop(
        robot,
        _request(trials=3),
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.02), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert [r["verdict"] for r in rows] == ["held", "empty", "held"]
    # The first comes from the fixture; the second from the script's re-grip of the first; the
    # third from the fixture again, because nobody re-picks a peg the policy missed.
    assert [r["staging"] for r in rows] == ["fixture", "regrip", "fixture"]
    assert len(asked) == 1 and "fixture" in asked[0]
    assert rows[0]["widthLifted"] == GraspRig.HELD_WIDTH
    assert rows[0]["regripWidth"] == GraspRig.HELD_WIDTH
    assert "regripWidth" not in rows[1]
    assert rows[1]["widthLifted"] == GraspRig.EMPTY_WIDTH
    assert rows[1]["lateralMm"] == pytest.approx(20.0, abs=0.5)
    assert result["halted"] == ""
    assert result["summary"]["held"] == 2 and result["summary"]["graded"] == 3
    # A run never ends with the peg in the air.
    assert not robot.held


def test_the_regrip_is_the_scripts_close_at_its_own_height_and_the_peg_is_pressed_then_let_go(tmp_path, monkeypatch):
    robot = GraspRig()
    request = _request(trials=2)
    slept = []
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: slept.append((len(robot.actions), seconds)))
    run_grasp_loop(
        robot,
        request,
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.0)]),
        out_path=tmp_path / "g.jsonl",
    )
    closes = [a for a in robot.actions if a["gripper.pos"] == request.closedGripper]
    assert any(a["ee.z"] == pytest.approx(request.regripZ) for a in closes)
    # Carried to the second target in the script's grip and let go where it touches.
    opens = [
        i for i, (prev, a) in enumerate(zip(robot.actions, robot.actions[1:]), start=1)
        if prev["gripper.pos"] == request.closedGripper and a["gripper.pos"] >= 0.5
    ]
    press_z = request.regripZ - grasp_loop.GRASP_LOOP_PLACE_PRESS_M
    pressed = [i for i in opens if robot.actions[i]["ee.z"] == pytest.approx(press_z)]
    # ...after a second held there, still closed.
    assert pressed and all((i, request.placeDwellS) in slept for i in pressed)


def test_every_descent_onto_the_table_waits_at_a_hover_first(tmp_path, capsys):
    robot = GraspRig()
    run_grasp_loop(robot, _request(trials=2), run_policy_trial=_policy(robot, [("grasp", 0.0)] * 2), out_path=tmp_path / "g.jsonl")
    names = [
        line.split(" name=")[1].split()[0]
        for line in capsys.readouterr().out.splitlines()
        if "scene_reset_step=start" in line
    ]
    descents = [i for i, n in enumerate(names) if n in ("descend_8cm_to_pick", "descend_8cm_to_place")]
    # Fixture pick and place, set-down, re-grip, carried place, park: every one hovers first.
    assert len(descents) >= 6
    assert all(names[i - 1] == "settle_above_" + names[i].rsplit("_", 1)[1] for i in descents)


class WrenchRig(GraspRig):
    """The same rig, reporting a wrench: a spring pushing up once the tool is below the table."""

    TABLE_Z = 0.052

    @property
    def external_wrench(self):
        return (0.1, -0.2, 3.0 + 2000.0 * max(0.0, self.TABLE_Z - self.xyz[2]), 0.0, 0.0, 0.0)


def test_the_force_trace_records_every_scripted_step_and_changes_no_command(tmp_path):
    plan = [("grasp", 0.0), ("grasp", 0.02)]
    plain, traced = GraspRig(), WrenchRig()
    run_grasp_loop(plain, _request(trials=2), run_policy_trial=_policy(plain, plan), out_path=tmp_path / "a.jsonl",
                   wait_for_operator=_put_back(plain, []))
    out = tmp_path / "b.jsonl"
    run_grasp_loop(traced, _request(trials=2), run_policy_trial=_policy(traced, plan), out_path=out,
                   wait_for_operator=_put_back(traced, []))
    # Read-only: the arm is sent the same commands as without a wrench to read. Compared with
    # consecutive repeats collapsed: how many ticks a step holds a setpoint follows the wall clock.
    def distinct(actions):
        return [a for i, a in enumerate(actions) if i == 0 or a != actions[i - 1]]

    assert distinct(traced.actions) == distinct(plain.actions)
    assert not (tmp_path / "a_force.jsonl").exists()
    records = [json.loads(line) for line in (tmp_path / "b_force.jsonl").read_text().splitlines()]
    names = {r["name"] for r in records}
    assert {"settle_above_place", "descend_8cm_to_place", "dwell_before_open", "open_gripper", "settle_after_open"} <= names
    assert all(r["outcome"] == "done" and r["requestId"].startswith("grasp_loop_") for r in records)
    assert all(len(row) == len(r["columns"]) for r in records for row in r["samples"])
    fz = records[0]["columns"].index("fz")
    assert all(row[fz] >= 3.0 for r in records for row in r["samples"])
    # The sink belongs to the run: nothing outside it is traced.
    assert scene_reset._force_trace_path is None


def test_a_press_past_the_step_tolerance_is_refused():
    with pytest.raises(SceneResetError, match="placePressM"):
        validate_grasp_loop_request(_request(placePressM=0.010))


def test_a_peg_that_tips_on_the_set_down_is_handed_to_the_operator(tmp_path):
    class TippingRig(GraspRig):
        releases = 0

        def send_action(self, action):
            was_held = self.held
            result = super().send_action(action)
            if was_held and not self.held:
                self.releases += 1
                if self.releases == 2:
                    # The first release is the fixture reset's; the second, out of the policy's
                    # grip, tips the peg over and it rolls off.
                    self.peg_xyz = (self.peg_xyz[0] + 0.04, self.peg_xyz[1], TARGET_Z)
            return result

    robot = TippingRig()
    asked = []
    out = tmp_path / "g.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert rows[0]["verdict"] == "held" and rows[0]["regripWidth"] == GraspRig.EMPTY_WIDTH
    assert len(asked) == 1
    assert rows[1]["staging"] == "fixture"


def test_the_peg_is_where_the_fingers_opened_not_where_they_were_sent(tmp_path):
    class ShortRig(GraspRig):
        """Settles 2 mm short in x, inside even the hover tolerance, as the real arm does."""

        def get_observation(self, *, include_cameras=False):
            observation = super().get_observation(include_cameras=include_cameras)
            observation["ee.x"] -= 0.002
            return observation

    robot = ShortRig()
    out = tmp_path / "g.jsonl"
    run_grasp_loop(robot, _request(trials=1), run_policy_trial=_policy(robot, [("grasp", 0.0)]), out_path=out)
    (row,) = _trials(out)
    assert row["placeOffsetMm"] == pytest.approx(2.0, abs=0.1)
    assert row["pegXyz"][0] == pytest.approx(row["targetXyz"][0] - 0.002, abs=1e-5)


def test_the_lift_keeps_the_policys_own_close_rather_than_squeezing_harder(tmp_path):
    robot = GraspRig()
    policy = _policy(robot, [("grasp", 0.0)])
    handed_over_at = []

    def trial(index, handover):
        status = policy(index, handover)
        handed_over_at.append(len(robot.actions))
        return status

    run_grasp_loop(robot, _request(trials=1), run_policy_trial=trial, out_path=tmp_path / "g.jsonl")
    after = robot.actions[handed_over_at[0]:]
    carried = [a for a in after[: next(i for i, a in enumerate(after) if a["gripper.pos"] >= 0.5)]]
    assert carried and all(a["gripper.pos"] == 0.2 for a in carried)


def test_the_verdict_is_taken_after_the_lift_and_not_at_the_close(tmp_path, monkeypatch):
    # Fingers that shut on the peg and lose it as it rises: clamped at the close, empty carried.
    class SlippingRig(GraspRig):
        def send_action(self, action):
            before = self.xyz[2]
            result = super().send_action(action)
            if self.held and self.xyz[2] > before + 0.01:
                self.held = False
                self.peg_xyz = (self.peg_xyz[0], self.peg_xyz[1], TARGET_Z)
                self.gripper = self.EMPTY_WIDTH
            return result

    robot = SlippingRig()
    run_grasp_loop(robot, _request(trials=1), run_policy_trial=_policy(robot, [("grasp", 0.0)]), out_path=tmp_path / "g.jsonl")
    (row,) = _trials(tmp_path / "g.jsonl")
    assert row["widthAtClose"] == GraspRig.HELD_WIDTH
    assert row["verdict"] == "empty"


def test_a_knocked_peg_halts_an_unattended_run_instead_of_closing_on_air(tmp_path):
    robot = GraspRig()
    out = tmp_path / "g.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=3),
        run_policy_trial=_policy(robot, [("knock", None), ("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
    )
    assert [r["verdict"] for r in _trials(out)] == ["empty"]
    assert result["halted"] == "peg_lost"


def test_an_attended_run_waits_for_the_peg_to_go_back_in_the_fixture(tmp_path):
    robot = GraspRig()
    asked = []
    operator = _put_back(robot, asked)
    out = tmp_path / "g.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_policy(robot, [("knock", None), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=operator,
    )
    assert len(asked) == 1 and "fixture" in asked[0]
    assert [r["verdict"] for r in _trials(out)] == ["empty", "held"]
    assert result["halted"] == ""


def test_a_policy_that_never_closes_is_graded_a_failure_not_dropped(tmp_path):
    robot = GraspRig()
    out = tmp_path / "g.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=1, maxPolicySteps=20),
        run_policy_trial=_policy(robot, [("wander", None)]),
        out_path=out,
    )
    (row,) = _trials(out)
    assert row["verdict"] == "no_close"
    assert summarize_grasp_loop([row])["graded"] == 1


def test_a_rerun_resumes_after_the_trials_already_answered(tmp_path):
    out = tmp_path / "g.jsonl"
    robot = GraspRig()
    run_grasp_loop(robot, _request(trials=1), run_policy_trial=_policy(robot, [("grasp", 0.0)]), out_path=out)
    robot = GraspRig()
    run_grasp_loop(robot, _request(trials=2), run_policy_trial=_policy(robot, [None, ("grasp", 0.0)]), out_path=out)
    assert [r["trial"] for r in _trials(out)] == [0, 1]


def test_the_stop_file_is_read_at_trial_boundaries(tmp_path):
    robot = GraspRig()
    out = tmp_path / "g.jsonl"
    calls = iter([False, True])
    result = run_grasp_loop(
        robot,
        _request(trials=3),
        run_policy_trial=_policy(robot, [("grasp", 0.0)] * 3),
        out_path=out,
        stop_requested=lambda: next(calls),
    )
    assert len(_trials(out)) == 1
    assert result["halted"] == "stop_requested"
    assert not robot.held


def test_a_motion_fault_is_written_down_and_ends_the_run(tmp_path, monkeypatch):
    robot = GraspRig()
    out = tmp_path / "g.jsonl"

    def broken(*args, **kwargs):
        raise TimeoutError("scene reset step lift_8cm_after_grasp timed out")

    monkeypatch.setattr(grasp_loop, "check_grasp", broken)
    result = run_grasp_loop(robot, _request(trials=2), run_policy_trial=_policy(robot, [("grasp", 0.0)] * 2), out_path=out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert result["halted"] == "motion_fault"
    assert any(r["kind"] == "halt" and "timed out" in r["error"] for r in rows)
    assert _trials(out) == []


# ---------------------------------------------------------------------------- summary ---


def test_wilson_matches_the_textbook_value():
    low, high = wilson_interval(11, 38)
    assert low == pytest.approx(0.170, abs=0.002)
    assert high == pytest.approx(0.448, abs=0.002)


def test_operator_stops_are_not_counted_as_policy_outcomes():
    rows = [
        {"verdict": "held", "closeAboveTargetMm": 10.0, "lateralMm": 2.0},
        {"verdict": "empty", "closeAboveTargetMm": 30.0, "lateralMm": 9.0},
        {"verdict": "not_graded"},
    ]
    summary = summarize_grasp_loop(rows)
    assert summary["graded"] == 2 and summary["held"] == 1
    assert summary["closeAboveTargetMm"] == {"held": 10.0, "empty": 30.0}


# ---------------------------------------------------------------------------- control ---


def test_the_control_channel_answers_the_operator_wait_and_the_boundary_stop():
    import threading

    lines: list[str] = []
    from tools.fr3.grasp_loop import GraspLoopControl

    class Pipe:
        def __init__(self):
            self.queue: list[str] = []
            self.ready = threading.Condition()
            self.closed = False

        def put(self, line):
            with self.ready:
                self.queue.append(line)
                self.ready.notify()

        def close(self):
            with self.ready:
                self.closed = True
                self.ready.notify()

        def __iter__(self):
            while True:
                with self.ready:
                    while not self.queue and not self.closed:
                        self.ready.wait()
                    if self.queue:
                        yield self.queue.pop(0)
                    else:
                        return

    pipe = Pipe()
    control = GraspLoopControl(pipe, log=lines.append)
    control.start()
    answered = []
    waiter = threading.Thread(target=lambda: answered.append(control.wait_for_operator("put it back")))
    waiter.start()
    pipe.put("continue\n")
    waiter.join(timeout=2)
    assert answered == [True]
    assert not control.stop_requested()
    pipe.put("stop\n")
    pipe.close()
    for _ in range(100):
        if control.stop_requested():
            break
        threading.Event().wait(0.01)
    assert control.stop_requested()
    # A closed channel means nobody is there: the wait answers at once, and answers no.
    assert control.wait_for_operator("again") is False
    assert "[INFO] grasp_loop_stop=requested" in lines


def test_empty_fingers_that_stop_at_a_partial_close_are_not_graded_held(tmp_path):
    # 2026-09-23: the policy's close command sits at ~0.06, and empty fingers stop there, not at
    # zero. Four such closes 31-162 mm off the peg were graded held on the width alone.
    class CommandTrackingRig(GraspRig):
        def send_action(self, action):
            result = super().send_action(action)
            if not self.held and float(action["gripper.pos"]) < 0.5:
                self.gripper = max(self.EMPTY_WIDTH, float(action["gripper.pos"]))
            return result

    robot = CommandTrackingRig()

    def policy(trial, handover):
        at = (robot.peg_xyz[0] + 0.05, robot.peg_xyz[1], robot.peg_xyz[2] + 0.02)
        for step in range(100):
            gripper = 1.0 if step < 5 else 0.07
            robot.send_action({"ee.x": at[0], "ee.y": at[1], "ee.z": at[2], "ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0, "gripper.pos": gripper})
            if handover.observe(step, robot.xyz, gripper):
                return "grasp_handover"
        return "grasp_timeout"

    out = tmp_path / "g.jsonl"
    run_grasp_loop(robot, _request(trials=1), run_policy_trial=policy, out_path=out)
    (row,) = _trials(out)
    assert row["widthLifted"] == pytest.approx(0.07)
    assert row["verdict"] == "empty"
    assert row["widthOverCommand"] == pytest.approx(0.0, abs=1e-3)


def test_held_needs_the_fingers_stopped_wider_than_their_command():
    held = lambda width, cmd: grasp_loop.grasp_is_held(width, cmd, held_width=0.025, blocked_margin=0.05)
    # The 2026-09-23 pilot, as (width lifted, command).
    assert held(0.3102, 0.0) and held(0.3088, 0.0012)
    assert not any(held(w, c) for w, c in [(0.0573, 0.0562), (0.075, 0.0744), (0.0622, 0.0666), (0.4913, 0.4964), (0.006, 0.0)])


def test_a_miss_that_never_came_low_near_the_peg_is_picked_up_by_the_script(tmp_path):
    """09-24 trial 8: the pure policy closed 35 mm above the peg and 49 mm off. The peg was still
    standing; asking a person to put it back cost a trial's worth of their time for nothing."""

    robot = GraspRig()
    asked = []
    out = tmp_path / "g.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=3),
        run_policy_trial=_policy(robot, [("air", (0.05, 0.035)), ("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert [r["verdict"] for r in rows] == ["empty", "held", "held"]
    assert rows[0]["pegUntouched"] is True and rows[0]["lowestNearPegMm"] is None
    assert [r["staging"] for r in rows] == ["fixture", "repick", "regrip"]
    assert asked == []


def test_a_miss_that_came_low_near_the_peg_still_goes_to_the_operator(tmp_path):
    robot = GraspRig()
    asked = []
    out = tmp_path / "g.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=2),
        # 30 mm off at the grasp height: inside the fingers' reach, below the peg top.
        run_policy_trial=_policy(robot, [("air", (0.03, 0.0)), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert rows[0]["pegUntouched"] is False and rows[0]["lowestNearPegMm"] == pytest.approx(0.0, abs=0.5)
    assert [r["staging"] for r in rows] == ["fixture", "fixture"]
    assert len(asked) == 1


def test_a_repick_that_comes_up_empty_hands_the_peg_to_the_operator(tmp_path):
    """The path said clear, but the peg is not there (the fake's knock moves it without the tool
    going near): the re-pick closes on air, and a person is asked rather than trusting it."""

    robot = GraspRig()
    asked = []
    out = tmp_path / "g.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_policy(robot, [("knock", None), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert rows[0]["pegUntouched"] is True
    assert rows[1]["staging"] == "fixture" and len(asked) == 1
    assert not robot.held


class ReflexRig(GraspRig):
    """The rig with libfranka's loop: it can die, and `recover_control_loop` brings it back."""

    def __init__(self):
        super().__init__()
        self.control_loop_alive = True
        self.recoveries = 0

    def recover_control_loop(self):
        self.recoveries += 1
        self.control_loop_alive = True


def _reflex_policy(robot, plan):
    """Like `_policy`, plus ("reflex", None): the arm is driven into the peg top and the loop dies."""

    inner = _policy(robot, [p if p[0] != "reflex" else ("wander", None) for p in plan])

    def run(trial, handover):
        if plan[trial][0] == "reflex":
            robot.control_loop_alive = False
            return "control_loop_died"
        return inner(trial, handover)

    return run


def test_a_reflex_in_the_policys_segment_is_a_collision_and_recovers_only_on_the_operators_word(tmp_path):
    robot = ReflexRig()
    asked = []
    out = tmp_path / "g.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_reflex_policy(robot, [("reflex", None), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    rows = _trials(out)
    assert [r["verdict"] for r in rows] == ["collision", "held"]
    # First the reflex -- recover, open, up, home -- then the peg, which it may have knocked.
    assert asked[0].startswith(grasp_loop.GRASP_LOOP_REFLEX_PROMPT) and "fixture" in asked[1]
    assert robot.recoveries == 1 and robot.gripper >= 0.5
    assert result["halted"] == ""
    # Counted against the arm, like a miss.
    summary = result["summary"]
    assert summary["graded"] == 2 and summary["held"] == 1 and summary["collision"] == 1


def test_an_unattended_reflex_halts_without_moving_the_arm(tmp_path):
    robot = ReflexRig()
    out = tmp_path / "g.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_reflex_policy(robot, [("reflex", None), ("grasp", 0.0)]),
        out_path=out,
    )
    moved = len(robot.actions)
    assert result["halted"] == "control_loop_died"
    assert robot.recoveries == 0 and not robot.control_loop_alive
    assert [r["verdict"] for r in _trials(out)] == ["collision"]
    assert len(robot.actions) == moved


def test_a_reflex_in_a_scripted_step_is_recovered_and_is_not_charged_to_the_policy(tmp_path):
    class DiesOnSetDown(ReflexRig):
        def send_action(self, action):
            if self.held and action["ee.z"] < 0.07 and not getattr(self, "died", False):
                self.died = True
                self.control_loop_alive = False
                return dict(action)
            return super().send_action(action)

    robot = DiesOnSetDown()
    asked = []
    out = tmp_path / "g.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=2),
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
        wait_for_operator=_put_back(robot, asked),
    )
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["kind"] for line in lines if line["kind"] in ("reflex", "trial")] == ["reflex", "trial"]
    assert robot.recoveries == 1 and asked[0].startswith(grasp_loop.GRASP_LOOP_REFLEX_PROMPT)
    assert result["halted"] == ""


class PadRig(FakeRobot):
    """Fingers that centre the peg only along their closing axis (tool y), on a pad 20 mm wide
    (tool x). `knock_once_m` is one bad close: the first grasp the fingers make lands that far off
    across the pad, as 09-28's did. Homing puts the wrist back, as the real keyframe move does."""

    PAD_HALF_M = 0.010

    def __init__(self, knock_once_m=0.0):
        super().__init__()
        self.peg_xyz = PICK
        self.held = self.closed = False
        self.offset = (0.0, 0.0)
        self.knock_once_m = knock_once_m
        self.offsets_at_close = []

    def move_to_start(self):
        super().move_to_start()
        self.rotvec = (0.0, 0.0, 0.0)

    def send_action(self, action):
        result = super().send_action(action)
        if float(action["gripper.pos"]) >= 0.5:
            if self.held:
                self.peg_xyz = (self.xyz[0] + self.offset[0], self.xyz[1] + self.offset[1], self.xyz[2])
            self.held = self.closed = False
            return result
        if not self.closed:
            self.closed = True
            turn = _rotation(self.rotvec)
            width_axis, closing_axis = turn.apply(np.array([1.0, 0.0, 0.0]))[:2], turn.apply(np.array([0.0, 1.0, 0.0]))[:2]
            d = np.array(self.peg_xyz[:2]) - np.array(self.xyz[:2])
            across = float(d @ width_axis) + self.knock_once_m
            self.knock_once_m = 0.0
            self.held = (
                abs(across) <= self.PAD_HALF_M
                and abs(float(d @ closing_axis)) <= 0.02
                and abs(self.xyz[2] - self.peg_xyz[2]) <= 0.012
            )
            if self.held:
                # Squeezed onto the tool along the closing axis; left where it was across the pad.
                self.offset = tuple(float(v) for v in across * width_axis)
                self.offsets_at_close.append(abs(across))
        if self.held:
            self.peg_xyz = (self.xyz[0] + self.offset[0], self.xyz[1] + self.offset[1], self.xyz[2])
        self.gripper = GraspRig.HELD_WIDTH if self.held else GraspRig.EMPTY_WIDTH
        return result


def _rotation(rotvec):
    from lerobot.utils.rotation import Rotation

    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))


def _aim_at_the_record(robot):
    """A funnel-like arm B: closes exactly where the loop says the peg is."""

    def run(trial, handover):
        at = handover.pegXyz
        for step in range(100):
            gripper = 1.0 if step < 5 else 0.0
            robot.send_action({
                "ee.x": at[0], "ee.y": at[1], "ee.z": at[2] - 0.006,
                "ee.wx": robot.rotvec[0], "ee.wy": robot.rotvec[1], "ee.wz": robot.rotvec[2],
                "gripper.pos": gripper,
            })
            if handover.observe(step, robot.xyz, gripper):
                return "grasp_handover"
        return "grasp_timeout"

    return run


@pytest.mark.parametrize(("turn", "stays_off"), [(grasp_loop.GRASP_LOOP_REGRIP_TURN_RAD, False), (0.0, True)])
def test_one_off_centre_close_is_squeezed_out_by_the_turned_regrip(tmp_path, monkeypatch, turn, stays_off):
    """09-28 (photo 11:25): one close put the peg 6 mm off across the pad, and every close after it
    kept it there -- each aimed at the recorded tool point, which never sees the offset. With the
    re-grip closing across the funnel's close, the next close squeezes it out."""

    monkeypatch.setattr(grasp_loop, "GRASP_LOOP_REGRIP_TURN_RAD", turn)
    robot = PadRig(knock_once_m=0.006)
    out = tmp_path / "g.jsonl"
    run_grasp_loop(robot, _request(trials=6), run_policy_trial=_aim_at_the_record(robot), out_path=out)
    assert [r["verdict"] for r in _trials(out)] == ["held"] * 6
    later = robot.offsets_at_close[3:]
    if stays_off:
        assert all(v == pytest.approx(0.006, abs=1e-4) for v in later)
    else:
        assert all(v <= 1e-4 for v in later)


class HighTableRig(GraspRig):
    """The rig with a force estimate, and a table the carried peg meets 4 mm above the place
    height -- 09-28 trial 35. The arm goes where it is told; the estimate reads the overlap as a
    5 N/mm push."""

    CONTACT_Z = None

    def send_action(self, action):
        was_closed = self.closed
        result = super().send_action(action)
        if self.closed and not was_closed:
            self.grip_z = self.xyz[2]
        return result

    @property
    def external_wrench(self):
        # Only a carried peg meets the table early; at the height it was taken it stands on it.
        if not self.held or self.CONTACT_Z is None or abs(self.xyz[2] - getattr(self, "grip_z", -1.0)) < 1e-6:
            return (0.1, 0.1, 1.0, 0.0, 0.0, 0.0)
        return (0.1, 0.1, 1.0 - 5000.0 * max(0.0, self.CONTACT_Z - self.xyz[2]), 0.0, 0.0, 0.0)


def test_a_set_down_that_meets_the_table_early_stops_pushing_and_lets_go_there(tmp_path, capsys, monkeypatch):
    # 1 mm a tick, so the cap is crossed a millimetre or two into the table rather than a jump.
    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 0.03)
    request = _request(trials=2)
    robot = HighTableRig()
    HighTableRig.CONTACT_Z = request.regripZ + 0.004
    try:
        run_grasp_loop(robot, request, run_policy_trial=_policy(robot, [("grasp", 0.0)] * 2), out_path=tmp_path / "g.jsonl")
    finally:
        HighTableRig.CONTACT_Z = None
    out = capsys.readouterr().out
    assert "stopped_on=contact" in out
    contact = request.regripZ + 0.004
    stops = [float(line.rsplit("z=", 1)[1]) for line in out.splitlines() if "stopped_on=contact" in line]
    # Every set-down stopped within the cap's worth (7 N at 5 N/mm) plus one tick of the table.
    assert stops and all(z >= contact - 0.0014 - 0.0011 for z in stops)
    # And let go where it stopped, not back down at the target.
    opens = [
        a for prev, a in zip(robot.actions, robot.actions[1:])
        if prev["gripper.pos"] == request.closedGripper and a["gripper.pos"] >= 0.5 and a["ee.z"] < 0.07
    ]
    assert opens and all(a["ee.z"] >= contact - 0.0025 for a in opens)
    assert [r["verdict"] for r in _trials(tmp_path / "g.jsonl")] == ["held", "held"]


# ------------------------------------------------------------------------- end to end ---


from tools.fr3.terminal_servo import TerminalServoRequest  # noqa: E402

import tools.fr3.terminal_servo as terminal_servo  # noqa: E402


class InsertRig(GraspRig):
    """The grasp rig with the fixture's hole under the pick: v14 step 4's whole task.

    A held peg carried over the block stops on its face unless the tool is within `capture_m`
    of the hole, where it goes down as far as it is told; let go of there, it drops back into
    the fixture at the pick. That is all the servo needs to tell "in" from "on the face".
    """

    def __init__(self, hole_xy=PICK[:2], capture_m=0.0042, face_above_m=0.035):
        super().__init__()
        self.hole_xy = tuple(hole_xy)
        self.capture_m = capture_m
        self.face_above_m = face_above_m
        self.face_z = None  # set by the test to the aim's height + face_above_m
        self.opened_at = []

    def _over_hole(self, x, y):
        return math.dist((x, y), self.hole_xy) <= self.capture_m

    def send_action(self, action):
        x, y, z = float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])
        near_block = math.dist((x, y), self.hole_xy) <= 0.03
        # Only a peg coming down from above meets the face; one already below it (the fixture's
        # own peg, lifted out) is in the hole.
        from_above = self.xyz[2] >= self.face_z - 1e-6 if self.face_z is not None else False
        if self.held and from_above and near_block and not self._over_hole(x, y):
            action = {**action, "ee.z": max(z, self.face_z)}
        was_held = self.held
        result = super().send_action(action)
        if was_held and not self.held:
            self.opened_at.append(tuple(self.xyz))
            if self._over_hole(self.xyz[0], self.xyz[1]) and self.xyz[2] < 0.1:
                self.peg_xyz = PICK
        return result


def _insert_servo(**overrides):
    fields = dict(
        xyz=(PICK[0], PICK[1], TARGET_Z),
        handoffZ=0.12,
        searchRingM=0.007,
        controlPeriodS=0.01,
        timeoutS=2.0,
        settleS=0.01,
        openSettleS=0.0,
    )
    fields.update(overrides)
    return TerminalServoRequest(**fields)


@pytest.fixture
def _fast_servo(monkeypatch):
    monkeypatch.setattr(terminal_servo, "precise_sleep", lambda seconds: None)


def _insert_rig(**kwargs):
    robot = InsertRig(**kwargs)
    # The policy closes at the peg's table height (TARGET_Z), 6 mm above the script's regripZ, so
    # the aim is raised by that; the face is 35 mm over the aim, as the mouth is on the rig.
    robot.face_z = TARGET_Z + 0.006 + robot.face_above_m
    return robot


def _grades(robot, answers, asked):
    answers = list(answers)

    def grade(message):
        asked.append((message, robot.held))
        return answers.pop(0)

    return grade


def test_a_held_grasp_is_carried_into_the_hole_and_let_go_only_on_the_operators_in(tmp_path, _fast_servo):
    robot = _insert_rig()
    asked = []
    out = tmp_path / "e2e.jsonl"
    result = run_grasp_loop(
        robot,
        _request(trials=2, insertServo=_insert_servo()),
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
        ask_grade=_grades(robot, ["in", "in"], asked),
    )
    rows = _trials(out)
    assert [r["verdict"] for r in rows] == ["held", "held"]
    assert [r["inserted"] for r in rows] == [True, True]
    # Asked while the peg was still in the fingers, with the servo's own reading in the question.
    assert all(held for _message, held in asked) and len(asked) == 2
    assert asked[0][0].startswith("grade:") and "auto=seated" in asked[0][0]
    # Put in the hole, not re-gripped: the second trial is staged from the fixture again.
    assert [r["staging"] for r in rows] == ["fixture", "fixture"]
    assert "regripWidth" not in rows[0]
    insert = rows[0]["insert"]
    assert insert["grade"] == "in" and insert["autoVerdict"] == "seated" and insert["released"]
    assert insert["searchIndex"] == 0
    assert insert["raisedMm"] == pytest.approx(6.0, abs=0.1)
    assert insert["aimXyz"][2] == pytest.approx(TARGET_Z + 0.006)
    assert robot.move_to_start_calls >= 2
    e2e = result["summary"]["endToEnd"]
    assert e2e["inserted"] == 2 and e2e["graded"] == 2 and e2e["insertedOfHeld"] == [2, 2]
    assert e2e["firstLanding"] == 2
    assert not robot.held


def test_out_keeps_hold_takes_the_peg_back_to_its_table_spot_and_regrips_it_there(tmp_path, _fast_servo):
    robot = _insert_rig()
    out = tmp_path / "e2e.jsonl"
    asked = []
    result = run_grasp_loop(
        robot,
        _request(trials=2, insertServo=_insert_servo()),
        run_policy_trial=_policy(robot, [("grasp", 0.0), ("grasp", 0.0)]),
        out_path=out,
        ask_grade=_grades(robot, ["out", "in"], asked),
    )
    rows = _trials(out)
    assert [r["inserted"] for r in rows] == [False, True]
    assert rows[0]["insert"]["released"] is False
    # Nothing was let go of over the block on "out".
    assert all(math.dist(at[:2], robot.hole_xy) > 0.03 for at in robot.opened_at[:1])
    # Set down where it was picked and taken the script's way, so the next trial is a regrip.
    assert rows[0]["regripWidth"] == GraspRig.HELD_WIDTH
    assert rows[1]["staging"] == "regrip"
    assert result["summary"]["endToEnd"]["inserted"] == 1


def test_nobody_grading_lets_the_servos_seated_verdict_decide(tmp_path, _fast_servo):
    robot = _insert_rig()
    out = tmp_path / "e2e.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=1, insertServo=_insert_servo()),
        run_policy_trial=_policy(robot, [("grasp", 0.0)]),
        out_path=out,
    )
    row = _trials(out)[0]
    assert row["inserted"] is True and row["insert"]["grade"] is None
    assert not robot.held and robot.peg_xyz == PICK


def test_a_first_landing_on_the_face_is_found_by_the_search(tmp_path, _fast_servo):
    # The hole 7 mm off the aim, past the capture radius: the ring finds it.
    robot = _insert_rig(hole_xy=(PICK[0] + 0.007, PICK[1]))
    out = tmp_path / "e2e.jsonl"
    run_grasp_loop(
        robot,
        _request(trials=1, insertServo=_insert_servo()),
        run_policy_trial=_policy(robot, [("grasp", 0.0)]),
        out_path=out,
        ask_grade=_grades(robot, ["in"], []),
    )
    insert = _trials(out)[0]["insert"]
    assert insert["inserted"] is True and insert["searchIndex"] > 0


def test_a_missed_grasp_is_a_miss_of_the_whole_task_and_is_never_carried_to_the_hole(tmp_path, _fast_servo):
    robot = _insert_rig()
    out = tmp_path / "e2e.jsonl"
    asked = []
    result = run_grasp_loop(
        robot,
        _request(trials=1, insertServo=_insert_servo()),
        run_policy_trial=_policy(robot, [("grasp", 0.02)]),
        out_path=out,
        ask_grade=_grades(robot, [], asked),
        wait_for_operator=_put_back(robot, []),
    )
    row = _trials(out)[0]
    assert row["verdict"] == "empty" and row["inserted"] is False and "insert" not in row
    assert asked == []
    assert result["summary"]["endToEnd"] == {
        "graded": 1, "inserted": 0, "rate": 0.0, "wilson95": [0.0, 0.793],
        "insertedOfHeld": [0, 0], "firstLanding": 0,
    }


def test_an_ungraded_insertion_is_left_out_of_the_end_to_end_rate():
    rows = [
        {"verdict": "held", "inserted": True, "insert": {"searchIndex": 0}},
        {"verdict": "held", "inserted": None, "insert": {"searchIndex": 0}},
        {"verdict": "empty", "inserted": False},
    ]
    e2e = summarize_grasp_loop(rows)["endToEnd"]
    assert (e2e["graded"], e2e["inserted"], e2e["insertedOfHeld"]) == (2, 1, [1, 1])
    # A grasp-only run has no end-to-end reading at all.
    assert "endToEnd" not in summarize_grasp_loop([{"verdict": "held"}])


def test_an_insertion_target_that_is_not_the_fixture_is_refused():
    request = _request(insertServo=_insert_servo(xyz=(PICK[0] + 0.03, PICK[1], TARGET_Z)))
    with pytest.raises(SceneResetError, match="pickXyz"):
        validate_grasp_loop_request(request)
    validate_grasp_loop_request(_request(insertServo=_insert_servo(xyz=(0.3597, -0.1328, 0.058))))


def test_the_request_record_carries_the_insertion_servo_as_plain_data():
    record = grasp_loop._request_record(_request(insertServo=_insert_servo()))
    assert json.loads(json.dumps(record))["insertServo"]["searchRingM"] == 0.007
    assert grasp_loop._request_record(_request())["insertServo"] is None


def test_only_in_or_out_answers_a_grade_and_a_stop_answers_nobody():
    import queue
    import threading

    from tools.fr3.grasp_loop import GraspLoopControl

    lines: list[str] = []
    words: "queue.Queue[str | None]" = queue.Queue()

    def stream():
        while (word := words.get()) is not None:
            yield word

    control = GraspLoopControl(stream(), log=lines.append)
    control.start()
    answers = []
    asker = threading.Thread(target=lambda: answers.append(control.ask_grade("grade: trial 1")))
    asker.start()
    words.put("continue\n")
    asker.join(timeout=0.5)
    assert asker.is_alive() and answers == []
    words.put("in\n")
    asker.join(timeout=2)
    assert answers == ["in"]
    assert "[ATTENTION] grasp_loop_needs_operator grade: trial 1" in lines
    assert "[INFO] grasp_loop_operator=in" in lines

    asker = threading.Thread(target=lambda: answers.append(control.ask_grade("grade: trial 2")))
    asker.start()
    words.put("stop\n")
    asker.join(timeout=2)
    assert answers == ["in", None]
    words.put(None)
