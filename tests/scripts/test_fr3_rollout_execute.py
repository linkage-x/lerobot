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
        return self.q.copy(), self.dq.copy(), np.asarray(self.f_ext(self.clock()), dtype=float)

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
         gripper=None, stop_flag=lambda: "", record=None, start_check=None, step=(0.0, 0.0, -STEP_DOWN_M)):
    c = _contract()
    clock = Clock()
    arm = FakeArm(clock, f_ext=f_ext)
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
    stage = replace(rx.STAGES[2], max_s=0.8)
    result, arm, seen = _run(stage, limits=lim, step=(0.0, 0.024, 0.0), age_s=0.05)
    assert result.completed, result.stopped
    assert result.chunks_accepted >= 2  # through a switch (blended) as well
    v = _speeds(arm.commands, result.ticks)
    vmax = rx.joint_speed_limit(lim, 1 / 30)
    assert (np.abs(v) / vmax).max() < 1.02  # never commanded faster than the limit
    assert (np.abs(v) / vmax).max() > 0.9  # ... and it did run at it: slowed only as much as needed
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
    # the policy's timing, except the last step (the arm comes to rest there within STOP_ACCELERATION)
    np.testing.assert_allclose(off[:5], np.arange(5) / 30, atol=1e-9)
    q[3:, 0] += 0.06  # step 3 asks 0.07 rad in 33 ms
    off = rx.retime(q, 1 / 30, rx.Limits(), still_start=False, end=5)
    assert np.diff(off)[2] > 1.4 / 30 and np.diff(off)[0] == pytest.approx(1 / 30)


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
    assert rx.tick_problem(dq=np.array([0, 0, 0, 0, 0, 0, 3.0]), **kw) == ("", "")  # j7 under 1.6 x 2.0
    kind, why = rx.tick_problem(dq=np.array([0, 0, 2.0, 0, 0, 0, 0]), **kw)
    assert kind == "speed" and "j3 turns at 2.00 rad/s (limit 1.60" in why


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
