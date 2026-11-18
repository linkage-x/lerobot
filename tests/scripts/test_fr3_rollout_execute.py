# ruff: noqa: N803, N806  -- T, T_obs: the usual names of transforms
"""The relative-TCP execution layer, closed loop against a simulated arm (no hardware, no panda_py)."""

from __future__ import annotations

import queue
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import fr3_kinematics as kin, relative_tcp as rt, rollout_execute as rx

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
        q, ok = kin.ik_retry(kin.NOMINAL, targets, seed, early_exit=True)
        why = ["" if o else "IK" for o in ok]
        ok = ok.copy()
        if fail_step is not None:
            ok[fail_step - 1] = False
            why[fail_step - 1] = f"step {fail_step}: IK not reachable inside the joint walls"
        return {"T": targets, "q": q, "ok": ok, "why": why, "widths": widths}

    return plan


def _policy(c: rt.PolicyContract, width=lambda k: 0.088):
    def predict(_sample):
        T = np.repeat(np.eye(4)[None], c.horizon, axis=0)
        T[:, 2, 3] = -STEP_DOWN_M * np.arange(1, c.horizon + 1)
        return np.concatenate([rt.encode(T), np.array([[width(k)] for k in range(c.horizon)])], axis=1)

    return predict


def _run(stage: rx.Stage, *, age_s=0.15, stale_frames=False, fail_step=None, f_ext=None, limits=rx.Limits(), width=lambda k: 0.088,
         gripper=None, stop_flag=lambda: ""):
    c = _contract()
    clock = Clock()
    arm = FakeArm(clock, f_ext=f_ext)
    history = rx.PoseHistory()
    seen: list[dict] = []

    def observe(min_t):
        t_img = clock() - age_s if stale_frames else max(min_t, clock() - age_s)
        return rx.Observation(t_img, {"cam_10": np.zeros((18, 32, 3), np.uint8)}, 0.088, None,
                              {"box_stale_s": 0.02})

    pred = rx.Predictor(contract=c, history=history, observe=observe, predict=_policy(c, width), plan=_plan(fail_step),
                        sample=lambda h, o: {"history": h}, limits=limits, clock=clock)
    # synchronous: each request is answered before the loop's next tick (the thread is not started)
    pred.request = lambda *, min_t, seed: pred.results.put(pred.once(min_t, seed))
    result = rx.run_loop(arm=arm, predictor=pred, stage=stage, n_exec=16, limits=limits, fk_tcp=_fk,
                         wall_lower=kin._WALL_LOWER, wall_upper=kin._WALL_UPPER, history=history,
                         stop_flag=stop_flag, log=lambda **kw: seen.append(kw), gripper=gripper, clock=clock,
                         sleep=clock.sleep)
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


def test_assess_ends_a_chunk_at_its_first_step_too_fast():
    T = np.repeat(np.eye(4)[None], 5, axis=0)
    T[:, 0, 3] = [0, 0.005, 0.010, 0.040, 0.045]  # step 3 jumps 30 mm
    q = np.zeros((5, 7))
    ok, why = rx.assess(T, q, np.ones(4, bool), [""] * 4, rx.Limits())
    assert ok.tolist() == [True, True, True, False, False]
    assert "30.0 mm in one step" in why[3] and why[4] == "after step 3"


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


def test_execute_reads_the_unlock_before_loading_anything(tmp_path: Path, monkeypatch):
    from tools.fr3 import real_execution_unlock, relative_tcp_policy as rtp, rollout_relative_tcp as ro

    def nothing(*_a, **_k):
        raise AssertionError("nothing may be loaded before the unlock is read")

    monkeypatch.setattr(rtp, "load_checkpoint", nothing)
    with pytest.raises(real_execution_unlock.RealExecutionLockedError, match="executing a policy on the arm is locked"):
        ro.execute(tmp_path / "ckpt", tmp_path / "rec", arm="right", stage=1, gateway="http://x", repo_root=tmp_path)
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
