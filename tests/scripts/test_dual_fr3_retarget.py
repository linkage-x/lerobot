"""dual_fr3_retarget (roadmap C3): both arms' joint paths for a whole episode, optimised together."""

import math

import numpy as np
import pytest

from tools.fr3 import dual_fr3_collision as col, dual_fr3_retarget as rg, fr3_kinematics as kin

TOOL = kin.urdf_fixed_transform("corenetic_socket_ee")
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _placed(spacing: float) -> np.ndarray:
    T = np.eye(4)
    T[1, 3] = spacing
    return T


def _paths(n: int, turn: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    s = np.sin(np.linspace(0.0, np.pi, n))
    q = np.tile(kin.Q_READY, (n, 1))
    q[:, 0] += turn
    q[:, 3] += 0.25 * s
    return q, q * MIRROR


def _problem(q_r, q_l, spacing, **settings) -> rg.Problem:
    targets = {"right": kin.NOMINAL.fk(q_r) @ TOOL, "left": kin.NOMINAL.fk(q_l) @ TOOL}
    return rg.Problem(
        targets,
        {k: np.ones(len(q_r)) for k in targets},
        TOOL,
        _placed(spacing),
        col.load(),
        [],
        rg.Settings(**settings),
    )


def _swivel(q: np.ndarray, sign: float, steps: int, h: float = 0.02) -> np.ndarray:
    """Move ``q`` along the arm's self-motion (the TCP stays, the elbow swings)."""
    q = q.copy()
    for _ in range(steps):
        _, J = kin.NOMINAL.kinematics(q, TOOL[:3, 3])
        v = np.linalg.svd(J)[2][:, -1, :]
        v *= np.sign(v[:, 2:3] + 1e-12) * sign
        q = np.clip(q + h * v, kin.Q_LOWER, kin.Q_UPPER)
    return q


def test_the_block_tridiagonal_solve_is_the_dense_one() -> None:
    rng = np.random.default_rng(0)
    n, k, c = 6, 3, -2.0
    D = np.array([(lambda A: A @ A.T + 10 * np.eye(k))(rng.normal(size=(k, k))) for _ in range(n)])
    rhs = rng.normal(size=(n, k))
    dense = np.zeros((n * k, n * k))
    for t in range(n):
        dense[t * k : (t + 1) * k, t * k : (t + 1) * k] = D[t]
        if t:
            dense[t * k : (t + 1) * k, (t - 1) * k : t * k] = c * np.eye(k)
            dense[(t - 1) * k : t * k, t * k : (t + 1) * k] = c * np.eye(k)
    assert np.allclose(rg._block_tridiagonal_solve(D, c, rhs).ravel(), np.linalg.solve(dense, rhs.ravel()))


def test_a_reachable_pair_comes_back_unchanged_and_executable() -> None:
    q_r, q_l = _paths(40)
    targets = {"right": kin.NOMINAL.fk(q_r) @ TOOL, "left": kin.NOMINAL.fk(q_l) @ TOOL}
    t = np.arange(40) / 60.0
    result = rg.retarget(
        targets, {k: np.ones(40, bool) for k in targets}, tool=TOOL, T_right_left=_placed(0.8), obstacles=[],
        t_raw=t, speed=1 / 3, settings=rg.Settings(d_safe_m=0.05),
    )  # fmt: skip
    assert result["executable"], result["reasons"]
    after = result["retarget"]
    for side in ("right", "left"):
        assert after["arms"][side]["change_position_mm"]["max"] < 0.5
        assert after["arms"][side]["change_rotation_deg"]["max"] < 0.5
        assert result["start"][side]["frames_tracked"] == 40
    assert result["timing"]["reference_duration_s"] == pytest.approx(3 * t[-1])
    assert result["arrays"]["q"].shape == (40, 14)


def test_two_elbows_in_each_other_are_swung_apart_without_moving_the_hands() -> None:
    q_r, q_l = _paths(30, turn=-0.5)
    spacing, d_safe = 0.45, 0.05
    tangled = _swivel(q_l, -1.0, 40)  # same TCP path, the left elbow swung toward the right arm
    model = col.load()
    assert col.clearance(model, q_r, tangled, _placed(spacing))["arm_arm_m"].min() < 0.02
    problem = _problem(q_r, q_l, spacing, d_safe_m=d_safe)
    Q, _ = rg.optimise(problem, np.concatenate([q_r, tangled], axis=1))
    m = rg.metrics(problem, Q)
    assert m["min_arm_arm_gap_mm"] >= d_safe * 1e3 - 0.5
    for side in ("right", "left"):
        assert m["arms"][side]["change_position_mm"]["max"] < 0.5
        assert m["arms"][side]["change_rotation_deg"]["max"] < 0.5
    assert rg.verdict(m, problem.settings) == (True, [])


def test_a_target_past_the_reach_is_not_executable_and_says_why() -> None:
    q_r, q_l = _paths(30)
    targets = {"right": kin.NOMINAL.fk(q_r) @ TOOL, "left": kin.NOMINAL.fk(q_l) @ TOOL}
    targets["left"] = targets["left"].copy()
    targets["left"][:, 0, 3] += np.linspace(0.0, 0.8, 30)  # walking out of the reach
    result = rg.retarget(
        targets, {k: np.ones(30, bool) for k in targets}, tool=TOOL, T_right_left=_placed(0.8), obstacles=[],
        t_raw=np.arange(30) / 60.0, speed=1 / 3, settings=rg.Settings(d_safe_m=0.05),
    )  # fmt: skip
    assert not result["executable"]
    assert any(r.startswith("left: retarget moves the TCP") for r in result["reasons"])
    assert result["retarget"]["arms"]["right"]["change_position_mm"]["max"] < 0.5


def test_the_hands_relation_is_read_without_the_lever_between_them() -> None:
    q_r, q_l = _paths(10)
    problem = _problem(q_r, q_l, 0.8)
    turn = np.eye(4)
    turn[:3, :3] = kin._rpy_matrix(math.radians(0.5), 0.0, 0.0)
    problem.targets["left"] = problem.targets["left"] @ turn  # the left wrist 0.5 deg off, in place
    m = rg.metrics(problem, np.concatenate([q_r, q_l], axis=1))
    assert m["t_lr_change_position_mm"]["max"] < 1e-6  # not 0.5 deg x 0.8 m = 7 mm
    assert m["t_lr_change_rotation_deg"]["max"] == pytest.approx(0.5, abs=1e-6)


def test_retime_slows_a_segment_to_the_velocity_limit_and_never_speeds_up() -> None:
    n = 20
    Q = np.zeros((n, 14))
    Q[10:, 0] = 1.0  # a 1 rad step of j1 between frames 9 and 10
    t_raw = np.arange(n) / 60.0
    t_ref = rg.retime(Q, t_raw, speed=1.0, velocity_scale=0.5)
    dt = np.diff(t_ref)
    assert dt[9] == pytest.approx(1.0 / (0.5 * rg.FR3_VELOCITY_LIMITS[0]))
    assert np.allclose(np.delete(dt, 9), 1 / 60.0)
    q_cmd, index = rg.resample(Q, t_ref, 1 / 60.0)
    assert index[0] == 0 and index[-1] <= n - 1 and np.all(np.diff(index) > 0)
    assert np.abs(np.diff(q_cmd[:, 0])).max() <= 0.5 * rg.FR3_VELOCITY_LIMITS[0] / 60.0 + 1e-9


def test_d_safe_is_built_from_the_placement_and_the_execution_tails() -> None:
    d = rg.derive_d_safe(1.0)
    assert d["d_safe_m"] == pytest.approx(max(3 * 1.0 + 2 * rg.EXECUTION_TAIL_MM, rg.D_SAFE_FLOOR_MM) / 1e3)
    assumed = rg.derive_d_safe(None)
    assert not assumed["placement_sigma_known"] and assumed["d_safe_m"] > d["d_safe_m"]
    assert rg.derive_d_safe(0.0, execution_tail_mm=(1.0, 1.0))["d_safe_m"] == pytest.approx(
        rg.D_SAFE_FLOOR_MM / 1e3
    )
