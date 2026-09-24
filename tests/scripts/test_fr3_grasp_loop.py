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
    or ("wander", None) -- never closes.
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
