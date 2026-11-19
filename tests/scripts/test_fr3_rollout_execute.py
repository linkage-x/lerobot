# ruff: noqa: N803, N806  -- T, T_obs: the usual names of transforms
"""The relative-TCP execution layer, closed loop against a simulated arm (no hardware, no panda_py)."""

from __future__ import annotations

import queue
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import (
    fr3_kinematics as kin,
    relative_tcp as rt,
    rollout_execute as rx,
    rollout_execute_report as rep,
)

Q0 = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
STEP_DOWN_M = 0.004  # the fake policy: 4 mm down per 33 ms step (120 mm/s)


def _contract(horizon: int = 32) -> rt.PolicyContract:
    cam = rt.CameraInput("cam_10", 1920, 1080, [0, 0, 1920, 1080], 32, 18)
    return rt.PolicyContract(hand="right", fps=60, frame_stride=2, horizon=horizon, history=2, cameras=[cam],
                             width_name="box_gripper.distance_m")


def _fk(q: np.ndarray) -> np.ndarray:
    return kin.NOMINAL.fk(np.asarray(q, dtype=float)[None])[0]


def _moved_along_tcp_z(q: np.ndarray) -> float:
    """How far the TCP went along its own starting z axis (the fake policy moves along -z of the TCP)."""
    return float((np.linalg.inv(_fk(Q0)) @ _fk(q))[2, 3])


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(s, 1e-4)


class FakeArm:
    """Follows its command with a first-order lag (tau 15 ms); records what it was told."""

    def __init__(self, clock: Clock, q0=Q0, f_ext=None):
        self.clock, self.q, self.dq = clock, np.array(q0, dtype=float), np.zeros(7)
        self.cmd = self.q.copy()
        self.t = clock()
        self.f_ext = f_ext or (lambda t: np.zeros(6))
        self.bias = 0.05
        self.started = self.stopped = False
        self.commands: list[np.ndarray] = []

    def _advance(self):
        dt = self.clock() - self.t
        self.t = self.clock()
        if dt <= 0:
            return
        a = 1.0 - np.exp(-dt / 0.015)
        new = self.q + a * (self.cmd - self.q)
        self.dq = (new - self.q) / dt
        self.q = new

    def read(self):
        self._advance()
        # + a small constant: a real estimate is never exactly 0 (exactly 0 means no estimate, which stops a run)
        return self.q.copy(), self.dq.copy(), np.asarray(self.f_ext(self.clock()), dtype=float) + self.bias

    def start(self, q):
        self.started = True
        self.cmd = np.array(q, dtype=float)

    def command(self, q, dq):
        self._advance()
        self.cmd = np.array(q, dtype=float)
        self.commands.append(self.cmd.copy())

    def check(self):
        pass

    def stop(self):
        self.stopped = True


def _plan(fail_step: int | None = None):
    def plan(T_obs, actions, seed):
        targets, widths = rt.targets_from_actions(T_obs, actions)
        q, ok = kin.ik_retry(kin.NOMINAL, targets, seed, early_exit=True, rest=seed)  # as PlanContext.evaluate
        why = ["" if o else "IK" for o in ok]
        ok = ok.copy()
        if fail_step is not None:
            ok[fail_step - 1] = False
            why[fail_step - 1] = f"step {fail_step}: IK not reachable inside the joint walls"
        return {"T": targets, "q": q, "ok": ok, "why": why, "widths": widths}

    return plan


def _policy(c: rt.PolicyContract, width=lambda k: 0.088, step=(0.0, 0.0, -STEP_DOWN_M)):
    def predict(_sample):
        T = np.repeat(np.eye(4)[None], c.horizon, axis=0)
        T[:, :3, 3] = np.outer(np.arange(1, c.horizon + 1), step)
        return np.concatenate([rt.encode(T), np.array([[width(k)] for k in range(c.horizon)])], axis=1)

    return predict


def _run(stage: rx.Stage, *, age_s=0.15, stale_frames=False, fail_step=None, f_ext=None, limits=rx.Limits(), width=lambda k: 0.088,
         gripper=None, stop_flag=lambda: "", record=None, start_check=None, step=(0.0, 0.0, -STEP_DOWN_M), arm_cls=None):
    c = _contract()
    clock = Clock()
    arm = (arm_cls or FakeArm)(clock, f_ext=f_ext)
    history = rx.PoseHistory()
    seen: list[dict] = []

    def observe(min_t):
        t_img = clock() - age_s if stale_frames else max(min_t, clock() - age_s)
        return rx.Observation(t_img, {"cam_10": np.zeros((18, 32, 3), np.uint8)}, 0.088, None,
                              {"box_stale_s": 0.02})

    pred = rx.Predictor(contract=c, history=history, observe=observe, predict=_policy(c, width, step), plan=_plan(fail_step),
                        sample=lambda h, o: {"history": h}, limits=limits, clock=clock)
    # synchronous: each request is answered before the loop's next tick (the thread is not started)
    pred.request = lambda *, min_t, seed: pred.results.put(pred.once(min_t, seed))
    result = rx.run_loop(arm=arm, predictor=pred, stage=stage, n_exec=16, limits=limits, fk_tcp=_fk,
                         wall_lower=kin._WALL_LOWER, wall_upper=kin._WALL_UPPER, history=history,
                         stop_flag=stop_flag, log=lambda **kw: seen.append(kw), gripper=gripper, clock=clock,
                         sleep=clock.sleep, record=record, start_check=start_check)
    return result, arm, seen


def test_stage_one_runs_exactly_one_chunks_n_action_steps_then_holds():
    result, arm, seen = _run(rx.STAGES[1])
    assert result.completed and not result.stopped and result.chunks_accepted == 1
    assert arm.started and arm.stopped
    assert _moved_along_tcp_z(arm.q) == pytest.approx(-16 * STEP_DOWN_M, abs=1e-3)  # 16 steps of 4 mm, then held there
    chunks = [e for e in seen if e.get("event") == "chunk"]
    assert len(chunks) == 1 and chunks[0]["k0"] == 1 and chunks[0]["end"] == 16


def test_closed_loop_joins_each_late_chunk_at_its_first_future_step_and_keeps_the_speed():
    stage = replace(rx.STAGES[2], max_s=1.2)
    result, arm, seen = _run(stage, age_s=0.15)
    assert result.completed, result.stopped
    chunks = [e for e in seen if e.get("event") == "chunk" and e["accepted"]]
    assert len(chunks) >= 3
    # a frame 150 ms old: the steps whose time passed (~4-5 of them) are skipped, not executed late
    assert all(e["k0"] >= 5 for e in chunks[1:])
    # the arm kept the policy's speed: ~120 mm/s over the run (no slowing down, no jumps back)
    # 120 mm/s while running (each tick is 5 ms); each switch gives back ~2 mm: the new chunk is anchored on
    # the measured pose, which trails the command by the controller's lag (15 ms here)
    z = np.array([_moved_along_tcp_z(q) for q in arm.commands[: result.ticks]])
    speed = -(z[-1] - z[20]) / ((len(z) - 21) * 0.005)
    assert speed == pytest.approx(STEP_DOWN_M * 30, rel=0.1)
    steps = np.abs(np.diff(np.stack(arm.commands), axis=0)).max(axis=1)  # the largest joint, per tick
    # radians per 5 ms tick: no switch jumps (each step of a new chunk is seeded from the running one; 0.016
    # rad when seeded from one posture -- the elbow's redundancy resolved differently at the switch)
    assert steps[: result.ticks].max() < 0.012


def test_a_chunk_whose_needed_step_is_not_executable_is_refused_and_the_arm_never_moves():
    result, arm, _ = _run(rx.STAGES[1], fail_step=5)
    assert not result.completed and "step 5" in result.stopped and "IK" in result.stopped
    np.testing.assert_allclose(arm.q, Q0, atol=1e-6)


def test_a_start_away_from_where_the_demonstrations_began_is_refused_before_the_arm_is_taken():
    seen_q = []
    result, arm, _ = _run(rx.STAGES[1], start_check=lambda q: seen_q.append(q) or "start: the TCP is 150 mm from where any training episode began")
    assert not result.completed and result.stopped.startswith("start:") and result.chunks_accepted == 0
    assert not arm.started and not arm.commands  # the controller was never started
    np.testing.assert_allclose(seen_q[0], Q0)


def test_a_failing_tail_step_ends_the_chunk_there_but_does_not_refuse_it():
    result, arm, _ = _run(rx.STAGES[1], fail_step=20)
    assert result.completed  # stage 1 runs 16 steps; step 20 is past what it executes


def test_contact_stops_and_holds_where_the_arm_is():
    result, arm, _ = _run(replace(rx.STAGES[2], max_s=2.0), f_ext=lambda t: np.array([0, 0, 20.0 if t > 1000.6 else 0.0, 0, 0, 0]))
    assert "external force" in result.stopped and not result.completed
    np.testing.assert_allclose(arm.commands[-1], arm.q, atol=1e-3)


def test_a_force_swing_in_motion_does_not_stop_but_a_lasting_force_or_an_impact_does():
    # 11-19: O_F_ext swings +-10 N for a few ticks with the TCP's acceleration, nothing touched
    swing = lambda t: np.array([18.0 if 1000.60 < t < 1000.611 else 0.0, 0, 0, 0, 0, 0])  # noqa: E731
    result, _, _ = _run(replace(rx.STAGES[2], max_s=1.0), f_ext=swing)
    assert result.completed, result.stopped
    impact = lambda t: np.array([30.0 if 1000.60 < t < 1000.606 else 0.0, 0, 0, 0, 0, 0])  # noqa: E731
    result, _, _ = _run(replace(rx.STAGES[2], max_s=1.0), f_ext=impact)
    assert "in one tick (limit 25)" in result.stopped
    lasting = lambda t: np.array([16.0 if t > 1000.6 else 0.0, 0, 0, 0, 0, 0])  # noqa: E731
    result, _, _ = _run(replace(rx.STAGES[2], max_s=1.0), f_ext=lasting)
    assert "over 50 ms (limit 15)" in result.stopped


def test_a_chunk_brakes_to_its_last_step_over_several_steps():
    q = np.zeros((33, 7))
    q[:, 0] = np.cumsum(np.r_[0.0, np.full(32, 0.045)])  # 1.35 rad/s to the end (end = 20)
    T = np.repeat(np.eye(4)[None], 33, axis=0)
    ch = rx.Chunk(0, 0.0, 0.0, 1 / 30, T, q, np.ones(33, bool), [""] * 33, np.zeros(32))
    ch.prepare(rx.Limits(), 20)
    ts = np.arange(0.0, ch.offsets[20], 0.0005)
    dq = np.array([ch.command(t)[1] for t in ts])
    assert np.abs(ch.command(ch.offsets[20])[1]).max() == 0.0
    # planned at BRAKE_SHARE of it; the cubic between steps adds up to about 2x of that
    assert (np.abs(np.diff(dq[:, 0])) / 0.0005).max() < 1.5 * rx.STOP_ACCELERATION[0]
    assert (ch.stretch()[:20] > 1.01).sum() >= 3  # spread over several steps, not all in the last


def test_a_first_chunk_starts_from_rest_within_the_stop_acceleration():
    q = np.zeros((6, 7))
    q[1:, 0] = 0.012 + 0.002 * np.arange(5)  # step 1: 12 mrad at once (the policy's 1-2 mm offset), then slow
    off = rx.retime(q, 1 / 30, rx.Limits(), still_start=True, end=5)
    T = np.repeat(np.eye(4)[None], 6, axis=0)
    ch = rx.Chunk(0, 0.0, 0.0, 1 / 30, T, q, np.ones(6, bool), [""] * 6, np.zeros(5), still_start=True)
    ch.prepare(rx.Limits(), 5)
    np.testing.assert_allclose(ch.offsets, off)
    assert off[1] > 1 / 30  # the first step takes longer than 33 ms
    ts = np.arange(0.0, off[1], 0.0005)
    dq = np.array([ch.command(t)[1] for t in ts])
    # planned at BRAKE_SHARE of it; an uneven step (12 mrad, then 2) lets the cubic add about half again
    assert (np.abs(np.diff(dq, axis=0)) / 0.0005 / rx.STOP_ACCELERATION).max() < 1.6


def test_inputs_too_old_stop_before_anything_moves():
    result, arm, _ = _run(rx.STAGES[2], age_s=0.6, stale_frames=True)
    assert "ms old" in result.stopped
    np.testing.assert_allclose(arm.q, Q0, atol=1e-6)


def test_the_operator_stop_ends_the_run():
    clock_calls = {"n": 0}

    def flag():
        clock_calls["n"] += 1
        return "stopped by the operator" if clock_calls["n"] > 60 else ""

    result, arm, _ = _run(replace(rx.STAGES[2], max_s=5.0), stop_flag=flag)
    assert result.stopped == "stopped by the operator" and arm.stopped


def test_stage_three_closes_the_gripper_ahead_of_the_predicted_width():
    sent: list[str] = []
    result, _, seen = _run(replace(rx.STAGES[3], max_s=1.0), width=lambda k: 0.088 if k < 12 else 0.03,
                           gripper=sent.append)
    assert result.completed, result.stopped
    assert sent[:1] == ["close"]


def _speeds(commands: list[np.ndarray], ticks: int) -> np.ndarray:
    return np.diff(np.stack(commands[:ticks]), axis=0) / 0.005  # rad/s per joint, per 5 ms tick


def test_a_step_too_fast_for_a_joint_is_slowed_down_not_refused():
    # 24 mm per step along the TCP's y: more joint motion per step than allowed at Q0
    lim = rx.Limits(step_mm=36.0)
    stage = replace(rx.STAGES[2], max_s=1.3)  # from rest it ramps up first
    result, arm, seen = _run(stage, limits=lim, step=(0.0, 0.024, 0.0), age_s=0.05)
    assert result.completed, result.stopped
    assert result.chunks_accepted >= 2  # through a switch (blended) as well
    v = _speeds(arm.commands, result.ticks)
    vmax = rx.joint_speed_limit(lim, 1 / 30)
    assert (np.abs(v) / vmax).max() < 1.02  # never commanded faster than the limit
    assert (np.abs(v) / vmax).max() > 0.75  # ... and it ran close to it (ramps up from rest, brakes ahead)
    # what 11-19's stage 2 met: refusing such a step (no slowing allowed) stops the run
    result, _, _ = _run(stage, limits=replace(lim, step_stretch_max=1.0), step=(0.0, 0.024, 0.0), age_s=0.05)
    assert not result.completed and "in one step" in result.stopped


def test_a_re_posture_is_refused_however_slowly_it_could_be_run():
    # 11-18: IK pulled to the middle of the joint range re-postures the arm in step 1 for a 4 mm target
    T0 = _fk(Q0)
    T = np.repeat(T0[None], 3, axis=0)
    T[1:, 2, 3] -= [0.002, 0.004]
    q_mid, ok = kin.ik(kin.NOMINAL, T[1:], Q0)  # rest: the range middle
    q_held, _ = kin.ik(kin.NOMINAL, T[1:], Q0, rest=Q0)
    assert ok.all()
    held = np.vstack([Q0, q_held])
    ok, _ = rx.assess(T, held, np.ones(2, bool), ["", ""], rx.Limits())
    assert ok.all() and rx.reposture(held).max() < 0.01
    mid = np.vstack([Q0, q_mid])
    ok, why = rx.assess(T, mid, np.ones(2, bool), ["", ""], rx.Limits(step_stretch_max=1e9))  # speed not the issue
    assert not ok[1] and "re-postures" in why[1] and "IK branch change" in why[1]


def test_retime_keeps_the_policys_timing_unless_a_joint_needs_longer():
    q = np.zeros((6, 7))
    q[:, 0] = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]  # 0.3 rad/s: well under 1.5
    off = rx.retime(q, 1 / 30, rx.Limits(), still_start=False, end=5)
    # the policy's timing, except where it brakes to rest on the last step (within STOP_ACCELERATION)
    np.testing.assert_allclose(off[:4], np.arange(4) / 30, atol=1e-9)
    assert off[5] > 5 / 30
    q[3:, 0] += 0.06  # step 3 asks 0.07 rad in 33 ms
    off = rx.retime(q, 1 / 30, rx.Limits(), still_start=False, end=5)
    h = np.diff(off)
    assert h[2] > 1.4 / 30  # the fast step itself
    assert 1 / 30 < h[1] < h[2]  # and the arm speeds up into it within STOP_ACCELERATION, not in one knot


def test_a_sharp_change_of_direction_is_passed_slowly_or_ends_the_chunk_before_it():
    """11-19 (160814): j2 through 0, the IK turned j1 around at 29 rad/s² under every speed limit."""
    dt = 1 / 30
    smooth = np.zeros((12, 7))
    smooth[:, 0] = 0.02 * np.arange(12)  # 0.6 rad/s, straight on
    assert np.isinf(rx.turn_speed_cap(smooth)[1:-1]).all() or (rx.turn_speed_cap(smooth)[1:-1] > 1e3).all()
    np.testing.assert_allclose(np.diff(rx.retime(smooth, dt, rx.Limits(), still_start=False, end=0))[:8], dt, atol=1e-9)

    turn = smooth.copy()
    turn[6:, 0] = turn[5, 0] - 0.008 * np.arange(1, 7)  # turns from +0.6 to -0.24 rad/s at step 5
    a = np.abs(turn[6, 0] - 2 * turn[5, 0] + turn[4, 0]) / dt**2  # 25 rad/s², over STOP_ACCELERATION (7.5)
    cap = rx.turn_speed_cap(turn)[5]
    assert cap * dt == pytest.approx(np.sqrt(rx.STOP_ACCELERATION[0] / (a * dt * dt)) * dt)
    h = np.diff(rx.retime(turn, dt, rx.Limits(), still_start=False, end=0))
    assert h[4] > dt and h[5] > dt  # slowed into and out of the turn
    T = np.repeat(np.eye(4)[None], 12, axis=0)
    ok, why = rx.assess(T, turn, np.ones(11, bool), [""] * 11, rx.Limits(), dt=dt)
    assert ok[:6].all() and ok[5]  # x1.8 slowed: within step_stretch_max, run through slowly
    sharp = smooth.copy()
    sharp[6:, 0] = sharp[5, 0] - 0.03 * np.arange(1, 7)  # +0.6 to -0.9 rad/s: x2.6 slowed
    ok, why = rx.assess(T, sharp, np.ones(11, bool), [""] * 11, rx.Limits(), dt=dt)
    assert ok[:6].all() and not ok[6] and "changes direction at step 5" in why[6] and "j1" in why[6]


def test_a_stop_brakes_every_joint_to_rest_together_then_holds():
    v0 = np.array([2.0, -1.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    dur, at = rx.stop_motion(np.zeros(7), v0)
    assert pytest.approx(3.0 / 15.0) == dur  # the joint that needs longest (j5, 3 rad/s), at its rating
    _, dq_mid = at(dur / 2)
    np.testing.assert_allclose(dq_mid, v0 / 2)  # together: the direction of motion kept
    q_end, dq_end = at(dur + 1.0)
    np.testing.assert_allclose(dq_end, 0.0)
    np.testing.assert_allclose(q_end, v0 * dur / 2)
    assert rx.stop_motion(np.ones(7), np.zeros(7))[0] == 0.0


def test_a_stop_in_motion_brakes_instead_of_stepping_the_speed_to_zero():
    ticks = []
    result, _, _ = _run(replace(rx.STAGES[2], max_s=2.0), f_ext=lambda t: np.array([0, 0, 20.0 if t > 1000.6 else 0.0, 0, 0, 0]),
                        record=lambda item: ticks.append(item["tick"]) if "tick" in item else None)
    assert "external force" in result.stopped
    hold = np.array([t["q_cmd"] for t in ticks if t.get("phase") == 2])
    steps = np.abs(np.diff(hold, axis=0)).max(axis=1)
    assert steps[0] > 0 and steps[-1] == 0.0  # the command still moves right after the stop (braking), then stands


def test_no_external_force_estimate_refuses_the_start_and_one_that_goes_silent_stops():
    def silent_arm(clock, f_ext=None):
        arm = FakeArm(clock, f_ext=f_ext)
        arm.bias = 0.0
        return arm

    result, arm, _ = _run(rx.STAGES[2], arm_cls=silent_arm)
    assert "no external force estimate" in result.stopped and not arm.started

    def says_its_mode(clock, f_ext=None):
        arm = silent_arm(clock, f_ext)
        arm.condition = lambda: "robot mode kUserStopped"
        return arm

    result, _, _ = _run(rx.STAGES[2], arm_cls=says_its_mode)
    assert "standing still; robot mode kUserStopped)" in result.stopped  # the robot's own state, in the message

    class GoesSilent(FakeArm):
        def read(self):
            q, dq, f = super().read()
            return q, dq, (np.zeros(6) if self.clock() > 1000.5 else f)

    result, arm, _ = _run(replace(rx.STAGES[2], max_s=2.0), arm_cls=GoesSilent)
    assert "read exactly 0" in result.stopped and arm.started


def test_the_command_passes_every_step_with_continuous_speed_never_over_the_limit():
    rng = np.random.default_rng(0)
    q = np.cumsum(np.c_[rng.uniform(0.0, 0.08, (33, 1)), rng.uniform(-0.03, 0.03, (33, 6))], axis=0)  # uneven, fast
    q[0] = 0.0
    T = np.repeat(np.eye(4)[None], 33, axis=0)
    ch = rx.Chunk(0, 0.0, 0.0, 1 / 30, T, q, np.ones(33, bool), [""] * 33, np.zeros(32), still_start=True)
    ch.prepare(rx.Limits(), 32)
    for k in range(33):
        np.testing.assert_allclose(ch.command(ch.offsets[k])[0], q[k], atol=1e-12)
    ts = np.arange(0.0, ch.offsets[-1], 0.001)
    dq = np.array([ch.command(t)[1] for t in ts])
    vmax = rx.joint_speed_limit(rx.Limits(), 1 / 30)
    assert (np.abs(dq) / vmax).max() <= 1.0 + 1e-3
    for k in range(1, 32):  # the speed is continuous through every step (a linear command jumps there)
        before, after = ch.command(ch.offsets[k] - 1e-7)[1], ch.command(ch.offsets[k] + 1e-7)[1]
        assert np.abs(before - after).max() < 1e-3
    assert np.abs(ch.command(0.0)[1]).max() == 0.0 and np.abs(ch.command(ch.offsets[-1])[1]).max() == 0.0
    # monotone: between two steps no joint leaves their range
    for k in range(32):
        seg = np.array([ch.command(t)[0] for t in np.linspace(ch.offsets[k], ch.offsets[k + 1], 20)])
        lo, hi = np.minimum(q[k], q[k + 1]) - 1e-12, np.maximum(q[k], q[k + 1]) + 1e-12
        assert ((seg >= lo) & (seg <= hi)).all()


def test_a_newer_chunk_takes_over_without_a_jump():
    T = np.repeat(np.eye(4)[None], 5, axis=0)
    qa = np.outer(np.arange(5), np.full(7, 0.01))
    qb = qa + 0.02  # the newer prediction: 0.02 rad off the running one
    old = rx.Chunk(0, 0.0, 0.0, 1 / 30, T, qa, np.ones(5, bool), [""] * 5, np.zeros(4))
    new = rx.Chunk(1, 0.0, 0.0, 1 / 30, T, qb, np.ones(5, bool), [""] * 5, np.zeros(4))
    old.prepare(rx.Limits(), 4)
    new.prepare(rx.Limits(), 4)
    t0, t1 = 0.03, 0.03 + rx.BLEND_S * 0.5
    np.testing.assert_allclose(rx.blend(old, new, t0, t0, t1)[0], old.command(t0)[0])
    np.testing.assert_allclose(rx.blend(old, new, t1, t0, t1)[0], new.command(t1)[0])
    ts = np.linspace(t0, t1, 200)
    qs = np.array([rx.blend(old, new, t, t0, t1)[0] for t in ts])
    assert np.abs(np.diff(qs, axis=0)).max() < 0.02 / 50  # spread over the window, not one tick


def test_assess_ends_a_chunk_at_its_first_step_too_fast():
    T = np.repeat(np.eye(4)[None], 5, axis=0)
    T[:, 0, 3] = [0, 0.005, 0.010, 0.040, 0.045]  # step 3 jumps 30 mm
    q = np.zeros((5, 7))
    ok, why = rx.assess(T, q, np.ones(4, bool), [""] * 4, rx.Limits())
    assert ok.tolist() == [True, True, True, False, False]
    assert "30.0 mm in one step" in why[3] and why[4] == "after step 3"


def test_step_limits_are_the_checkpoints_own_under_the_ceilings():
    assert rx.step_limits(0.0362, 4.49) == {"step_mm": 36.2, "step_deg": 4.49}
    assert rx.step_limits(0.09, 20.0) == {"step_mm": rx.STEP_MM_CEILING, "step_deg": rx.STEP_DEG_CEILING}
    assert rx.step_limits(None, None) == {}  # no distribution: the Limits defaults stand
    lim = rx.Limits(**rx.step_limits(0.0362, 4.49))
    assert lim.step_mm == 36.2 and lim.near_step_mm == 15.0


def test_a_fast_step_passes_in_free_space_and_fails_near_the_carts_table_cameras():
    lim = rx.Limits(step_mm=36.0)
    T = np.repeat(np.eye(4)[None], 3, axis=0)
    T[1, 0, 3], T[2, 0, 3] = 0.025, 0.050  # 25 mm per step
    q = np.zeros((3, 7))
    ok, _ = rx.assess(T, q, np.ones(2, bool), ["", ""], lim, np.array([0.30, 0.30]))
    assert ok.all()
    ok, why = rx.assess(T, q, np.ones(2, bool), ["", ""], lim, np.array([0.30, 0.06]))
    assert ok.tolist() == [True, True, False]
    assert "25.0 mm in one step 60 mm from the carts/table/cameras (limit 15 closer than 100 mm)" in why[2]
    ok, why = rx.assess(T, q, np.ones(2, bool), ["", ""], rx.Limits(), None)  # the old 15 mm default
    assert not ok[1] and "limit 15" in why[1]


def test_joint_limits_scale_with_each_joints_own_speed_limit():
    T = np.repeat(np.eye(4)[None], 2, axis=0)
    q = np.zeros((2, 7))
    q[1, 6] = 0.08  # j7: within 0.05 x 2.0
    ok, _ = rx.assess(T, q, np.ones(1, bool), [""], rx.Limits())
    assert ok.all()
    q[1, 0] = 0.06  # j1: over 0.05, but within x2 of it: slowed down (retime), not refused
    ok, _ = rx.assess(T, q, np.ones(1, bool), [""], rx.Limits())
    assert ok.all()
    q[1, 0] = 0.11  # more than twice: an IK branch change, refused
    ok, why = rx.assess(T, q, np.ones(1, bool), [""], rx.Limits())
    assert not ok[1] and "j1 moves 0.110 rad in one step (limit 0.050 x2 slowed down" in why[1]
    kw = {"f_ext": np.zeros(6), "f0": np.zeros(6), "T_meas": np.eye(4), "T_cmd": np.eye(4), "loop_dt": 0.005,
          "limits": rx.Limits(), "wall_lower": kin._WALL_LOWER, "wall_upper": kin._WALL_UPPER,
          "q": np.array([0, 0, 0, -1.5, 0, 1.5, 0.0])}
    assert rx.tick_problem(dq=np.array([0, 0, 0, 0, 0, 0, 3.0]), **kw) == ("", "")  # j7 under 1.9 x 2.0
    kind, why = rx.tick_problem(dq=np.array([0, 0, 2.0, 0, 0, 0, 0]), **kw)
    assert kind == "speed" and "j3 turns at 2.00 rad/s (limit 1.90" in why


def test_history_is_read_at_the_observation_instant():
    h = rx.PoseHistory()
    for i in range(11):
        T = np.eye(4)
        T[0, 3] = 0.01 * i
        h.add(10.0 + 0.01 * i, T, np.full(7, float(i)))
    assert h.at(10.055)[0, 3] == pytest.approx(0.055)
    assert h.q_at(10.055)[0] == pytest.approx(5.5)
    with pytest.raises(RuntimeError, match="starts"):
        h.at(9.0)
    with pytest.raises(RuntimeError, match="no FK within"):
        h.at(10.5)


def test_the_shared_history_reads_back_like_the_in_process_one():
    """The loop writes its FK into shared memory; the predictor's process reads it (wrapping the ring)."""
    local, shared = rx.PoseHistory(span_s=0.5), rx.SharedPoseHistory(span_s=0.5, capacity=64)
    try:
        reader = rx.SharedPoseHistory.attach(shared.spec())
        for i in range(150):  # more than the ring holds
            T = np.eye(4)
            T[0, 3] = 0.001 * i
            for h in (local, shared):
                h.add(10.0 + 0.005 * i, T, np.full(7, float(i)))
        for t in (10.6, 10.6025, 10.745, 10.75, 10.76):
            np.testing.assert_allclose(reader.at(t), local.at(t))
            np.testing.assert_allclose(reader.q_at(t), local.q_at(t))
        np.testing.assert_allclose(reader.poses(10.74, [-0.02, -0.01, 0.0]), local.poses(10.74, [-0.02, -0.01, 0.0]))
        with pytest.raises(RuntimeError, match="starts"):
            reader.at(10.0)  # older than the ring (and the span)
        with pytest.raises(RuntimeError, match="no FK within"):
            reader.at(11.0)
        reader.close()
    finally:
        shared.close()


def _process_parts(*, fail: str = "", hang_s: float = 0.0):
    """ProcessPredictor's build, in the spawned process: the fake policy and plan, frames on real time."""
    import time

    if fail:
        raise RuntimeError(fail)
    c = _contract()

    def observe(min_t):
        time.sleep(hang_s)
        return rx.Observation(max(min_t, time.monotonic() - 0.03), {"cam_10": np.zeros((18, 32, 3), np.uint8)}, 0.088,
                              None, {"box_stale_s": 0.02})

    return {"contract": c, "limits": rx.Limits(), "observe": observe, "predict": _policy(c), "plan": _plan(),
            "sample": lambda h, o: {}, "info": {"n_exec": 16, "first_width": 0.088}}


def _fill(history, seconds: float = 0.5) -> None:
    import time

    now = time.monotonic()
    for k in range(int(seconds * 200) + 1):
        history.add(now - seconds + k * 0.005, _fk(Q0), Q0)


def _wait_result(pred, timeout_s: float = 30.0):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return pred.results.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
    raise AssertionError("no prediction from the predictor's process")


def test_the_predictor_runs_in_its_own_process_and_plans_like_the_thread():
    history = rx.SharedPoseHistory()
    pred = rx.ProcessPredictor((__name__, "_process_parts", {}), history=history)
    try:
        info = pred.start()
        assert info["n_exec"] == 16 and pred.offsets_s == [-4 / 60, -2 / 60, 0.0]
        _fill(history)
        import time

        pred.request(min_t=time.monotonic() - 0.2, seed=Q0)
        first = _wait_result(pred)
        assert first.error == "" and first.chunk is not None
        assert list(first.timing) == ["requested", "observe", "predict", "plan", "chunk"]
        _fill(history)
        pred.request(min_t=first.obs.t_img + 1e-3, seed=first.chunk)  # a replan: seeded from the running chunk
        second = _wait_result(pred)
        assert second.error == "" and second.chunk is not None and pred.busy[0] == ""

        local = rx.PoseHistory()
        _fill(local)
        thread = rx.Predictor(contract=_contract(), history=local, observe=_process_parts()["observe"],
                              predict=_policy(_contract()), plan=_plan(), sample=lambda h, o: {}, limits=rx.Limits())
        same = thread.serve(time.monotonic() - 0.2, Q0, time.monotonic())
        np.testing.assert_allclose(first.chunk.q, same.chunk.q, atol=1e-9)
    finally:
        pred.close()
        history.close()
    assert not pred.process.is_alive()


def test_a_predictor_process_that_cannot_load_says_why_before_anything_moves():
    history = rx.SharedPoseHistory()
    try:
        with pytest.raises(RuntimeError, match="no such checkpoint"):
            rx.ProcessPredictor((__name__, "_process_parts", {"fail": "no such checkpoint"}), history=history).start()
    finally:
        history.close()


def _process_dies(**_kw):
    import os

    os._exit(3)  # noqa: SLF001 -- a crash while loading (CUDA, a segfault): no message comes back


def test_a_predictor_process_that_dies_while_loading_says_so():
    history = rx.SharedPoseHistory()
    try:
        with pytest.raises(RuntimeError, match=r"ended before it was ready \(exit code 3\)"):
            rx.ProcessPredictor((__name__, "_process_dies", {}), history=history).start()
    finally:
        history.close()


def test_the_predictor_process_ending_is_a_prediction_error_that_stops_the_run():
    history = rx.SharedPoseHistory()
    pred = rx.ProcessPredictor((__name__, "_process_parts", {"hang_s": 30.0}), history=history)
    try:
        pred.start()
        import time

        pred.request(min_t=time.monotonic(), seed=Q0)
        time.sleep(0.3)
        assert pred.busy[0] == "observe"  # the loop's record says where it was
        pred.process.kill()
        out = _wait_result(pred)
        assert out.chunk is None and "predictor's process ended" in out.error
        with pytest.raises(queue.Empty):
            pred.results.get_nowait()  # said once
    finally:
        pred.close()
        history.close()


def test_the_loop_thread_asks_for_real_time_priority_and_says_when_refused(monkeypatch):
    import os

    calls = []
    monkeypatch.setattr(os, "sched_setscheduler", lambda pid, policy, param: calls.append((pid, policy, param.sched_priority)))
    assert rx.realtime_thread() == ""
    assert calls == [(0, os.SCHED_FIFO, rx.RT_PRIORITY)]

    def refuse(*_a):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "sched_setscheduler", refuse)
    assert "Operation not permitted" in rx.realtime_thread()
    rx.normal_thread()  # never raises


def test_gripper_hysteresis_and_interval():
    g = rx.GripperDecider(0.088)
    assert g.decide(0.055, 0.0) == ""
    assert g.decide(0.045, 0.0) == "close"
    assert g.decide(0.065, 0.1) == ""  # too soon
    assert g.decide(0.055, 1.0) == ""  # inside the band
    assert g.decide(0.065, 1.0) == "open"


def test_execute_without_attended_loads_nothing_and_moves_nothing(tmp_path: Path, monkeypatch, capsys):
    from tools.fr3 import relative_tcp_policy as rtp, rollout_relative_tcp as ro

    def nothing(*_a, **_k):
        raise AssertionError("nothing may be loaded without --attended")

    monkeypatch.setattr(rtp, "load_checkpoint", nothing)
    monkeypatch.setattr(ro, "execute", nothing)
    rc = ro.main(["execute", "--checkpoint", str(tmp_path / "ckpt"), "--record", str(tmp_path / "rec"),
                  "--arm", "right", "--stage", "1"])
    assert rc == 1 and "a hand on the e-stop" in capsys.readouterr().err
    assert not (tmp_path / "rec").exists()


def test_predictor_results_queue_is_a_queue():
    assert isinstance(rx.Predictor(contract=_contract(), history=rx.PoseHistory(), observe=None, predict=None,
                                   plan=None, sample=None, limits=rx.Limits()).results, queue.Queue)


def test_ik_early_exit_converges_to_the_posture_nearest_the_seed():
    T0 = _fk(Q0)
    targets = np.repeat(T0[None], 32, axis=0)
    targets[:, 2, 3] -= np.linspace(0.005, 0.16, 32)
    q_fast, ok_fast = kin.ik_retry(kin.NOMINAL, targets, Q0, early_exit=True)
    q_full, ok_full = kin.ik_retry(kin.NOMINAL, targets, Q0)
    assert ok_fast.all() and ok_full.all()
    np.testing.assert_allclose(kin.NOMINAL.fk(q_fast)[:, :3, 3], targets[:, :3, 3], atol=1e-4)
    # nearest the seed: never further from it than the 400-iteration answer, and smooth step to step
    assert np.abs(q_fast - Q0).sum() <= np.abs(q_full - Q0).sum() + 1e-6
    assert np.abs(np.diff(q_fast, axis=0)).max() < 0.03


def test_an_unreachable_target_does_not_hold_a_plan_up_for_the_whole_iteration_budget():
    """11-19 (163001): 3 targets that never converged kept a 32-target batch iterating 400 times (138 ms)."""
    import time

    targets = np.repeat(kin.NOMINAL.fk(Q0[None]), 32, axis=0)
    targets[-3:, 0, 3] += 2.0  # 2 m away: out of reach
    t = time.perf_counter()
    _, ok = kin.ik_retry(kin.NOMINAL, targets, Q0, early_exit=True, rest=Q0)
    fast = time.perf_counter() - t
    t = time.perf_counter()
    kin.ik(kin.NOMINAL, targets, Q0, early_exit=True, rest=Q0)  # one try at the full 400
    full = time.perf_counter() - t
    assert ok[:-3].all() and not ok[-3:].any()
    assert fast < full  # five tries at EARLY_EXIT_ITERS cost less than one at 400


def test_ik_for_execution_holds_the_posture_it_starts_from():
    """11-18 execute_20261118_171718: the arm stood at the wrist near j6 = pi, j5/j7 away from their middles; the
    first target, 1.3 mm off, moved j5 by 0.149 rad toward the middle and the chunk was refused. With the seed as
    the nullspace's rest the same step stays a millimetre's worth of joint motion."""
    q_arm = np.array([0.9338, -0.8253, -1.253, -2.5983, -0.2913, 3.1651, 0.4173])
    target = _fk(q_arm)
    target[:3, 3] += [-0.0005, -0.0012, -0.0005]
    q_mid, ok_mid = kin.ik_retry(kin.NOMINAL, target[None], q_arm, early_exit=True)
    q_hold, ok_hold = kin.ik_retry(kin.NOMINAL, target[None], q_arm, early_exit=True, rest=q_arm)
    assert ok_mid.all() and ok_hold.all()
    assert np.abs(q_mid[0] - q_arm)[[4, 6]].max() > 0.05  # the re-posture toward the middle
    assert np.abs(q_hold[0] - q_arm).max() < 0.02


def test_rollout_plans_pass_the_seed_as_the_ik_rest():
    from tools.fr3 import rollout_relative_tcp as ro

    seen = {}

    class Kin:
        NOMINAL = None

        @staticmethod
        def ik_retry(arm, flange, seed, *, early_exit, rest):
            seen.update(seed=seed, rest=rest, early_exit=early_exit)
            return np.repeat(np.asarray(seed, dtype=float)[None], len(flange), axis=0), np.ones(len(flange), bool)

    class Model:
        r = np.zeros(0)

        def place(self, arm, q):
            return np.zeros((0, 3)), np.zeros((0, 3))

        def table_mask(self):
            return np.zeros(0, bool)

    class Col:
        def obstacle_gaps(self, *a):
            return np.zeros(0)

    plan = object.__new__(ro.PlanContext)
    plan.kin, plan.T_tool, plan.T_base_world, plan.distribution = Kin, np.eye(4), None, None
    plan.model, plan.col, plan.obstacles, plan.problems = Model(), Col(), [], []
    actions = np.zeros((4, rt.POSE_DIM + 1))
    actions[:, 3] = actions[:, 7] = 1.0  # identity rotation columns r00, r11
    plan.evaluate(_fk(Q0), actions, Q0, early_exit=True)
    np.testing.assert_array_equal(seen["rest"], Q0)
    assert seen["early_exit"] is True


def test_the_preview_check_names_the_joint_step_execution_would_refuse():
    from tools.fr3 import rollout_relative_tcp as ro

    jump = {"j5": 0.0}

    class Kin:
        NOMINAL = kin.NOMINAL

        @staticmethod
        def ik_retry(arm, flange, seed, *, early_exit, rest):
            q = np.repeat(np.asarray(seed, dtype=float)[None], len(flange), axis=0)
            q[:, 4] += jump["j5"]
            return q, np.ones(len(flange), bool)

    class Model:
        r = np.zeros(0)

        def place(self, arm, q):
            return np.zeros((0, 3)), np.zeros((0, 3))

        def table_mask(self):
            return np.zeros(0, bool)

    class Col:
        def obstacle_gaps(self, *a):
            return np.zeros(0)

    plan = object.__new__(ro.PlanContext)
    plan.kin, plan.T_tool, plan.T_base_world, plan.distribution = Kin, np.eye(4), None, None
    plan.model, plan.col, plan.obstacles, plan.problems = Model(), Col(), [], []
    actions = np.zeros((32, rt.POSE_DIM + 1))
    actions[:, 3] = actions[:, 7] = 1.0
    assert plan.check(Q0, actions, None, 16)["step_limit"] == ""
    jump["j5"] = 0.149  # 11-18's re-posture: within x2 of j5's 0.10 per step -- execution slows it, passes
    assert plan.check(Q0, actions, None, 16)["step_limit"] == ""
    jump["j5"] = 0.25
    out = plan.check(Q0, actions, None, 16)
    assert out["execution_steps"] == 16 and "j5 moves 0.250 rad" in out["step_limit"]
    assert any(r.startswith("execution: step 1: j5") for r in out["reasons"]) and not out["ok"]


# ------------------------------------------------------ a move to a training start ---


def test_a_joint_move_is_a_straight_minimum_jerk_line_within_its_speed_and_acceleration():
    q1 = Q0 + np.array([0.6, -0.2, 0.3, 0.1, 0.0, 0.0, 0.9])
    T, at = rx.joint_move(Q0, q1)
    ts = np.linspace(0.0, T, 2001)
    q = np.array([at(t)[0] for t in ts])
    dq = np.array([at(t)[1] for t in ts])
    np.testing.assert_allclose(q[0], Q0) and np.testing.assert_allclose(q[-1], q1)
    np.testing.assert_allclose(dq[0], 0.0, atol=1e-12) and np.testing.assert_allclose(dq[-1], 0.0, atol=1e-12)
    assert (np.abs(dq) <= rx.MOVE_SPEED_RAD_S * rx.JOINT_SCALE + 1e-6).all()
    acc = np.diff(dq, axis=0) / np.diff(ts)[:, None]
    assert (np.abs(acc) <= rx.MOVE_ACCELERATION * 1.01).all()
    s = (q[:, [0, 6]] - Q0[[0, 6]]) / (q1 - Q0)[[0, 6]]  # every joint the same share of its way: a straight line
    np.testing.assert_allclose(s[:, 0], s[:, 1], atol=1e-9)


def _move(q_goal, *, arm_cls=FakeArm, f_ext=None, stop_flag=lambda: "", q_planned=None, record=None):
    clock = Clock()
    arm = arm_cls(clock, f_ext=f_ext)
    seen = []
    result = rx.run_joint_move(arm=arm, q_goal=q_goal, limits=rx.Limits(), fk_tcp=_fk, wall_lower=kin._WALL_LOWER,
                               wall_upper=kin._WALL_UPPER, stop_flag=stop_flag, log=lambda **kw: seen.append(kw),
                               clock=clock, sleep=clock.sleep, q_planned=q_planned, record=record)
    return result, arm, seen


def test_the_move_to_a_start_gets_there_and_lets_go():
    goal = Q0 + np.array([0.2, -0.1, 0.1, 0.05, 0.0, 0.0, 0.3])
    result, arm, seen = _move(goal, q_planned=Q0)
    assert result.completed and result.stopped == "" and arm.started and arm.stopped
    np.testing.assert_allclose(arm.q, goal, atol=1e-3)
    assert seen[0]["event"] == "move" and seen[0]["duration_s"] > 1.0


def test_the_move_to_a_start_refuses_a_moved_arm_and_a_dead_force_estimate_and_brakes_on_a_stop():
    goal = Q0 + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result, arm, _ = _move(goal, q_planned=Q0 + 0.05)
    assert "moved since the move was planned" in result.stopped and not arm.started

    def silent(clock, f_ext=None):
        a = FakeArm(clock, f_ext=f_ext)
        a.bias = 0.0
        return a

    result, arm, _ = _move(goal, arm_cls=silent)
    assert "no external force estimate" in result.stopped and not arm.started

    ticks = []
    result, arm, _ = _move(goal, f_ext=lambda t: np.array([0, 0, 20.0 if t > 1000.8 else 0.0, 0, 0, 0]),
                           record=lambda item: ticks.append(item["tick"]) if "tick" in item else None)
    assert "external force" in result.stopped and not result.completed and arm.stopped
    hold = np.array([t["q_cmd"] for t in ticks if t.get("phase") == 2])
    assert np.abs(np.diff(hold, axis=0)).max(axis=1)[-1] == 0.0  # braked to rest, then held
    assert np.abs(arm.q - goal).max() > 0.05  # it did not go on

    clock_stop = {"n": 0}

    def operator():
        clock_stop["n"] += 1
        return "stopped by the operator" if clock_stop["n"] > 100 else ""

    result, arm, _ = _move(goal, stop_flag=operator)
    assert result.stopped == "stopped by the operator" and arm.stopped


class _StartDist:
    def __init__(self, T_world: list[np.ndarray], opening: list[np.ndarray] | None = None):
        self.start_p = np.array([T[:3, 3] for T in T_world])
        self.start_q = np.array([rt.matrix_to_quat_xyzw(T[:3, :3]) for T in T_world])
        self.summary = {}
        self.opening = np.array([rt.matrix_to_pose7(T) for T in opening or []]).reshape(-1, 7)

    def opening_of(self, row):
        return self.opening

    def start_problem(self, T_world, width):
        d = np.linalg.norm(self.start_p - T_world[:3, 3], axis=1).min()
        return "" if d < 0.02 else f"start: {d * 1e3:.0f} mm from any training start"


def _start_plan(T_world: list[np.ndarray], gap=lambda path: np.full(len(path), 0.5), opening=None):
    from tools.fr3 import rollout_relative_tcp as ro

    class Model:
        r = np.zeros(1)

        def place(self, arm, path):
            return path, path

        def table_mask(self):
            return None

    class Col:
        def obstacle_gaps(self, obstacles, A, B, r, mask):
            return gap(A)[:, None]

    plan = object.__new__(ro.PlanContext)
    plan.kin, plan.T_tool, plan.T_base_world = kin, np.eye(4), np.eye(4)
    plan.model, plan.col, plan.obstacles, plan.problems = Model(), Col(), [], []
    plan.distribution = _StartDist(T_world, opening)
    return plan


def test_the_start_is_a_real_training_start_reached_clear_of_the_shoulder_singularity():
    q_start = np.array([1.1, -0.5, -1.0, -2.25, 0.1, 2.9, 0.7])  # j2 well away from 0
    T_start = _fk(q_start) @ np.eye(4)
    plan = _start_plan([T_start])
    q_now = np.array([1.0, -0.6, -0.9, -2.3, 0.1, 2.85, 0.65])
    out = plan.plan_to_start(q_now, rng=np.random.default_rng(0))
    assert out["ok"] and out["index"] == 0 and not out["near_shoulder_singularity"]
    np.testing.assert_allclose(_fk(np.array(out["q_goal"]))[:3, 3], T_start[:3, 3], atol=1e-3)
    assert abs(out["q_goal"][1]) >= 0.25 and out["room_rad"] >= 0.25

    # 11-19 (170940): standing in the other shoulder branch (j2 ~ 0, j3 near its wall), the start is still
    # reached in the posture with room in every joint, not the branch the arm happens to be in
    flipped = np.array([2.14, -0.02, -2.11, -2.09, -0.02, 2.4, 0.8])
    out = plan.plan_to_start(flipped, rng=np.random.default_rng(0))
    q_goal = np.array(out["q_goal"])
    walls = np.minimum(q_goal - kin._WALL_LOWER, kin._WALL_UPPER - q_goal).min()
    assert out["ok"] and out["room_rad"] >= 0.3 and walls >= 0.3 and abs(q_goal[1]) >= 0.3
    assert not out["followed_opening"]  # no opening stored: the room at the start

    # with the episode's opening (12 cm down from the start), the room is the smallest along it
    down = []
    for k in range(13):
        T = T_start.copy()
        T[2, 3] -= 0.01 * k
        down.append(T)
    out = _start_plan([T_start], opening=down).plan_to_start(q_now, rng=np.random.default_rng(0))
    assert out["ok"] and out["followed_opening"] and out["room_rad"] <= out["room_at_start_rad"] + 1e-9
    away = [T_start] + [T_start @ _shift(2.0)]  # an opening the arm cannot follow
    out = _start_plan([T_start], opening=away).plan_to_start(q_now, rng=np.random.default_rng(0))
    assert not out["ok"] and "follow the episode's opening" in out["why"]

    blocked = _start_plan([T_start], gap=lambda path: np.linspace(0.3, 0.004, len(path)))
    out = blocked.plan_to_start(q_now, rng=np.random.default_rng(0))
    assert not out["ok"] and "from the carts/table/cameras" in out["why"]

    beyond = q_now.copy()
    beyond[5] = -0.2  # past panda_py's wall on j6
    out = plan.plan_to_start(beyond, rng=np.random.default_rng(0))
    assert not out["ok"] and "walls now" in out["why"]


def _shift(dx: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = dx
    return T


def test_moving_to_a_start_without_attended_loads_nothing_and_moves_nothing(tmp_path: Path, monkeypatch, capsys):
    from tools.fr3 import rollout_relative_tcp as ro

    def nothing(*_a, **_k):
        raise AssertionError("nothing may run without --attended")

    monkeypatch.setattr(ro, "to_start", nothing)
    rc = ro.main(["to-start", "--checkpoint", str(tmp_path / "ckpt"), "--record", str(tmp_path / "rec"), "--arm", "right"])
    assert rc == 1 and "a hand on" in capsys.readouterr().err and not (tmp_path / "rec").exists()


# ------------------------------------------------------- the report on a record ---


def _recorded(tmp_path: Path, stage: rx.Stage, **kw):
    """A simulated run written by the real Recorder, with the record.json execute() writes around it."""
    import json

    limits = kw.pop("limits", rx.Limits())
    meta = {"schema": rx.RECORD_SCHEMA, "kind": "execute", "checkpoint": "ckpt", "contract": _contract().to_dict(),
            "arm": "right", "n_action_steps": 16, "frame_source": "/dev/shm/lerobot_online_sync",
            "tcp_tool": np.eye(4).tolist(), **rx.stage_dict(stage, limits)}
    recorder = rx.Recorder(tmp_path)
    result, arm, seen = _run(stage, limits=limits, record=recorder, **kw)
    recorder.close()
    meta["result"] = {"completed": result.completed, "stopped": result.stopped, "chunks": result.chunks_accepted,
                      "duration_s": round(result.duration_s, 3), "ticks": result.ticks}
    (tmp_path / "record.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
    return result, arm


def test_report_reads_a_closed_loop_run_back(tmp_path: Path):
    result, _ = _recorded(tmp_path, replace(rx.STAGES[2], max_s=2.0), age_s=0.15)
    r = rep.analyze(tmp_path)
    assert r["result"]["completed"] and r["chunks"]["accepted"] == result.chunks_accepted >= 3
    assert r["inputs"]["age_ms"]["p50"] == pytest.approx(150.0, abs=6.0)
    assert r["timing"]["late_steps"]["min"] >= 4  # a 150 ms old frame: 4-5 steps of 33 ms already past
    assert r["timing"]["loop_hz"] == pytest.approx(rx.LOOP_HZ, rel=0.05)
    # the fake arm lags its command by 15 ms at 120 mm/s: ~2 mm behind the policy's targets, never more than a few
    assert 0.5 < r["motion"]["target_mm"]["p50"] < 3.0 and r["motion"]["target_mm"]["max"] < 5.0
    assert r["motion"]["tracking_mm"]["max"] < 5.0
    assert r["motion"]["net_mm"] > 180.0  # 2 s at ~120 mm/s
    age = next(h for h in r["headroom"] if h["what"] == "frame age")
    assert age["limit"] == 400.0 and age["use"] == pytest.approx(age["worst"] / 400.0, abs=1e-3)
    text = rep.format_text(r)
    assert "TCP vs target" in text and "fewer than" not in text


def test_report_shows_the_tick_that_stopped_the_run_and_the_hold_after_it(tmp_path: Path):
    class Overshoots(FakeArm):  # 11-19 120252: 2 mm behind at the last recorded tick, 20.4 mm at the stop's
        def read(self):
            q, dq, f = super().read()
            if self.clock() > 1000.6:
                q = q + np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0])
            return q, dq, f

    result, _ = _recorded(tmp_path, replace(rx.STAGES[2], max_s=2.0), arm_cls=Overshoots)
    assert "off its command" in result.stopped
    r = rep.analyze(tmp_path)
    assert r["motion"]["tracking_mm"]["max"] < 5.0  # the run's own ticks: the stop's is reported apart
    trip = r["stop"]["trip"]
    assert trip["tracking_mm"] > 20.0 and 0.0 < trip["loop_dt_ms"] < 10.0
    assert r["stop"]["hold"]["ticks"] >= int(rx.HOLD_S * rx.LOOP_HZ) - 2
    assert r["timing"]["loop_ms"]["max"] < 10.0  # the hold's ticks do not count as loop periods
    text = rep.format_text(r)
    assert "the stop's tick" in text and "hold after stop" in text


def test_predictor_records_when_each_phase_began():
    c = _contract()
    clock = Clock()
    history = rx.PoseHistory()
    for k in range(40):
        history.add(clock() - 0.5 + k * 0.0125, _fk(Q0), Q0)

    def observe(min_t):
        clock.sleep(0.02)
        return rx.Observation(clock() - 0.03, {"cam_10": np.zeros((18, 32, 3), np.uint8)}, 0.088, None, {"box_stale_s": 0.02})

    def predict(sample):
        clock.sleep(0.03)
        return _policy(c)(sample)

    pred = rx.Predictor(contract=c, history=history, observe=observe, predict=predict, plan=_plan(), sample=lambda h, o: {},
                        limits=rx.Limits(), clock=clock)
    pred.request(min_t=clock() - 1.0, seed=Q0)
    pred.close()
    pred._run()  # in this thread: the request, then the close
    out = pred.results.get_nowait()
    assert out.chunk is not None and pred.busy == ("", pred.busy[1])
    tm = out.timing
    assert list(tm) == ["requested", "observe", "predict", "plan", "chunk"]
    assert tm["predict"] - tm["observe"] == pytest.approx(0.02) and tm["plan"] - tm["predict"] == pytest.approx(0.03)


def test_report_on_a_refused_first_chunk_says_the_arm_never_moved(tmp_path: Path):
    _recorded(tmp_path, rx.STAGES[1], fail_step=5)
    r = rep.analyze(tmp_path)
    assert r["chunks"] == {"accepted": 0, "refused": 1, "refusals": [r["chunks"]["refusals"][0]]}
    assert "step 5" in r["chunks"]["refusals"][0]
    assert "never commanded" in r["motion"]["note"]
    assert "never commanded" in rep.format_text(r)


def test_report_on_stage_one_warns_that_one_prediction_sets_no_limit(tmp_path: Path):
    _recorded(tmp_path, rx.STAGES[1])
    r = rep.analyze(tmp_path)
    assert r["inputs"]["age_ms"]["n"] == 1
    assert "fewer than 5 predictions" in rep.format_text(r)


def test_report_lists_the_gripper_commands(tmp_path: Path):
    sent: list[str] = []
    _recorded(tmp_path, replace(rx.STAGES[3], max_s=1.0), width=lambda k: 0.088 if k < 12 else 0.03, gripper=sent.append)
    r = rep.analyze(tmp_path)
    assert sent and [g["action"] for g in r["gripper"]] == sent


def test_report_pools_several_runs(tmp_path: Path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _recorded(a, replace(rx.STAGES[2], max_s=0.8))
    _recorded(b, replace(rx.STAGES[2], max_s=0.8))
    out = tmp_path / "r.json"
    assert rep.main([str(a), str(b), "--json", str(out)]) == 0
    import json

    data = json.loads(out.read_text())
    n = sum(run["inputs"]["age_ms"]["n"] for run in data["runs"])
    assert data["pooled"]["age_ms"]["n"] == n and "pooled over 2 runs" in capsys.readouterr().out


def test_report_uses_the_tool_frame_the_run_recorded():
    T_tool = np.eye(4)
    T_tool[2, 3] = 0.1
    fk = rep.tcp_fk({"tcp_tool": T_tool.tolist(), "arm": "right"})
    np.testing.assert_allclose(fk(np.stack([Q0, Q0])), np.stack([_fk(Q0) @ T_tool] * 2), atol=1e-12)


def test_report_refuses_a_preview_record(tmp_path: Path):
    (tmp_path / "record.json").write_text('{"kind": "preview"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an execute record"):
        rep.analyze(tmp_path)


def test_a_takeover_blends_longer_over_a_larger_gap_so_its_own_speed_stays_small():
    lim, dt = rx.Limits(), 1 / 30
    vmax = rx.joint_speed_limit(lim, dt)
    assert rx.blend_duration(np.zeros(7), lim, dt) == rx.BLEND_S
    gap = np.zeros(7)
    gap[0] = 0.03
    T = rx.blend_duration(gap, lim, dt)
    assert T > rx.BLEND_S and 1.5 * gap[0] / T <= rx.BLEND_SPEED_SHARE * vmax[0] + 1e-9
    assert rx.blend_duration(np.full(7, 1.0), lim, dt) == rx.BLEND_MAX_S
    # a chunk keeps that share free: a takeover at full speed stays under the cap (11:40: 1.64 rad/s, cap 1.5)
    q = np.outer(np.arange(33), np.r_[0.06, np.zeros(6)])  # wants 1.8 rad/s on j1
    T = np.repeat(np.eye(4)[None], 33, axis=0)
    old = rx.Chunk(0, 0.0, 0.0, dt, T, q, np.ones(33, bool), [""] * 33, np.zeros(32))
    new = rx.Chunk(1, 0.0, 0.0, dt, T, q + gap, np.ones(33, bool), [""] * 33, np.zeros(32))
    old.prepare(lim, 32)
    new.prepare(lim, 32)
    t0 = old.offsets[5]
    t1 = t0 + rx.blend_duration(new.command(t0)[0] - old.command(t0)[0], lim, dt)
    v = np.array([rx.blend(old, new, t, t0, t1)[1] for t in np.linspace(t0, t1, 200)])
    assert np.abs(v[:, 0]).max() <= vmax[0] + 1e-6
