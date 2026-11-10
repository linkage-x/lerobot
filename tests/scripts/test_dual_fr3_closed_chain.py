"""dual_fr3_closed_chain (roadmap D6): two hands on one rigid object.

The geometry is checked against what it must be in closed form -- a placement error that is a translation only
costs nothing while the object only translates, and ``(I - R) p`` once it turns -- and the springs by how they
scale: the squeeze with the stiffness, the sag and the friction deadband against it.
"""

# ruff: noqa: N806  (T_right_left as in the runtime)
import math

import numpy as np
import pytest

from tools.fr3 import dual_fr3_closed_chain as cc, fr3_kinematics as kin

TOOL = kin.urdf_fixed_transform("corenetic_gripper_ee")
SPACING = 0.6


def _yaw(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rigid_pair(turn_deg: float = 0.0, n: int = 30):
    """Both hands on one object 0.3 m wide that moves 0.08 m along x and turns ``turn_deg`` about z, through
    IK: (q_right, q_left, T_right_left), every frame held."""
    T_RL = np.eye(4)
    T_RL[1, 3] = SPACING
    down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    s = np.linspace(0.0, 1.0, n)
    hands = {"right": [], "left": []}
    for si in s:
        obj = np.eye(4)
        obj[:3, :3] = _yaw(math.radians(turn_deg) * si)
        obj[:3, 3] = [0.5 + 0.08 * si, SPACING / 2, 0.3]
        for arm, side in (("right", -0.15), ("left", 0.15)):
            grasp = np.eye(4)
            grasp[:3, :3] = down
            grasp[:3, 3] = [0.0, side, 0.0]
            hands[arm].append(obj @ grasp)
    T_left = np.linalg.inv(T_RL) @ np.asarray(hands["left"])
    q = {}
    for arm, T in (("right", np.asarray(hands["right"])), ("left", T_left)):
        qa, ok = kin.ik_retry(kin.NOMINAL, T @ np.linalg.inv(TOOL), kin.Q_READY)
        assert ok.all()
        q[arm] = qa
    return q["right"], q["left"], T_RL


def test_held_segments_and_the_compliance_ramp() -> None:
    held = np.zeros(40, bool)
    held[10:20] = True
    held[30:35] = True
    assert cc.held_segments(held) == [(10, 20), (30, 35)]
    profile = cc.compliance_profile(held, 0.4, 4)
    assert np.all(profile[10:20] == 0.4) and np.all(profile[30:35] == 0.4)
    assert profile[5] == 1.0 and profile[24] == 1.0  # away from the segments
    assert np.all(np.diff(profile[5:11]) < 0)  # soft before the fingers seat
    assert np.all(np.diff(profile[19:25]) > 0)  # and stiff again after the release
    assert np.all(cc.compliance_profile(held, 1.0, 4) == 1.0)
    assert cc.held_segments(np.zeros(5, bool)) == []


def test_the_tcp_compliance_is_a_spring_that_scales_with_the_joints() -> None:
    q = kin.Q_READY[None]
    C = cc.tcp_compliance(q, TOOL)[0]
    assert np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) > 0)
    assert np.allclose(cc.tcp_compliance(q, TOOL, cc.JOINT_STIFFNESS * 0.5)[0], 2.0 * C)
    # translation at the ready pose: of the order of 1-10 mm per 10 N, as the arm is under its default gains
    assert 1e-4 < C[2, 2] < 1e-2


def test_a_rigid_plan_in_its_own_placement_asks_nothing_of_the_object() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=20.0)
    m = cc.relative_mismatch(q_r, q_l, np.ones(len(q_r), bool), T_RL, TOOL)
    assert np.abs(m["twist"]).max() < 1e-4  # the IK's tolerance
    assert len(m["frames"]) == len(q_r) and np.all(m["segment"] == 0)


def test_a_placement_translation_costs_nothing_until_the_object_turns() -> None:
    eps = np.eye(4)
    eps[:3, 3] = [0.01, -0.02, 0.005]
    held = None
    for turn, expected in ((0.0, 0.0), (20.0, None)):
        q_r, q_l, T_RL = _rigid_pair(turn_deg=turn)
        held = np.ones(len(q_r), bool)
        m = cc.relative_mismatch(q_r, q_l, held, T_RL, TOOL, T_true=T_RL @ eps)
        last = m["twist"][-1]
        if expected is not None:
            assert np.abs(last).max() < 1e-4
            continue
        # turned by R about z: the left hand is (I - R) p off, p the error (here in the right base's axes)
        R = _yaw(math.radians(turn))
        assert np.linalg.norm(last[:3]) == pytest.approx(
            np.linalg.norm((np.eye(3) - R) @ eps[:3, 3]), rel=0.02
        )
        assert np.abs(last[3:]).max() < 1e-4  # a translation turns nothing


def test_a_skew_costs_its_hand_s_travel() -> None:
    q_r, q_l, T_RL = _rigid_pair()
    held = np.ones(len(q_r), bool)
    m = cc.relative_mismatch(q_r, q_l, held, T_RL, TOOL, left_delay_frames=2.0)
    step = 0.08 / (len(q_r) - 1)
    assert np.linalg.norm(m["twist"][len(q_r) // 2, :3]) == pytest.approx(2 * step, rel=0.05)


def test_the_squeeze_scales_with_the_stiffness_and_the_sag_against_it() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=20.0)
    held = np.ones(len(q_r), bool)
    eps = np.eye(4)
    eps[:3, 3] = [0.0, 0.01, 0.0]
    m = cc.relative_mismatch(q_r, q_l, held, T_RL, TOOL, T_true=T_RL @ eps)
    chain = cc.ClosedChain(q_r, q_l, m["frames"], m["midpoint"], T_RL, TOOL)
    w1 = chain.internal_wrench(m["twist"])
    assert np.allclose(chain.internal_wrench(m["twist"], 0.5), 0.5 * w1)
    assert np.allclose(chain.sag_mm(1.0, 0.5), 2.0 * chain.sag_mm(1.0))
    assert np.all(chain.sag_mm(2.0) > chain.sag_mm(1.0))
    # a softer grasp in series takes the squeeze down; a rigid one changes nothing
    assert np.allclose(chain.internal_wrench(m["twist"], 1.0, None), w1)
    soft = np.linalg.norm(chain.internal_wrench(m["twist"], 1.0, 2000.0)[:, :3], axis=1)
    assert np.all(soft <= np.linalg.norm(w1[:, :3], axis=1) + 1e-9)
    # the grip holds the force's lever to the midpoint too
    assert np.all(chain.grasp_moments(w1) >= 0.0)


def test_a_run_that_followed_its_plan_measures_no_squeeze() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=10.0)
    held = np.zeros(len(q_r), bool)
    held[5:25] = True
    run = cc.from_run(q_r, q_l, held, T_RL, TOOL)
    assert run["held_frames"] == 20
    assert run["segments"][0]["frames"] == [5, 25] and run["segments"][0]["force_n_max"] < 0.05
    bent = q_l.copy()
    bent[15:25, 0] += 0.005  # the left's j1 lags 0.29 deg in the second half
    worse = cc.from_run(q_r, bent, held, T_RL, TOOL)["segments"][0]
    assert worse["mismatch_mm_max"] > 1.0 and worse["force_n_max"] > 0.5
    assert cc.from_run(q_r[:0], q_l[:0], held[:0], T_RL, TOOL) == {"held_frames": 0}


def test_the_budget_finds_the_stiffest_scale_inside_every_limit() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=20.0)
    plan = {
        "q_right": q_r,
        "q_left": q_l,
        "held": np.ones(len(q_r), bool),
        "T_right_left": T_RL,
        "tool": TOOL,
        "fps": 20.0,
    }
    report = cc.budget(plan, samples=40)
    assert report["held_frames"] == len(q_r) and report["plan_only"]["mismatch_mm_max"] < 0.1
    tight, loose = report["scenarios"]["assumed"], report["scenarios"]["measured_b3"]
    assert loose["mismatch_mm_p95"] < tight["mismatch_mm_p95"]
    forces = [r["force_n_p95"] for r in tight["scales"]]
    assert forces == sorted(forces, reverse=True)  # softer, less squeeze
    for scenario in (tight, loose):
        ok = [r for r in scenario["scales"] if r["ok"]]
        if ok:
            assert scenario["scale"] == ok[0]["scale"]
        else:
            assert "grasps" in scenario and "no joint stiffness fits" in scenario["verdict"]
    nothing = cc.budget({**plan, "held": np.zeros(len(q_r), bool)})
    assert nothing["held_frames"] == 0 and "nothing closes the chain" in nothing["verdict"]


def test_a_held_plan_round_trips(tmp_path) -> None:
    q_r, q_l, T_RL = _rigid_pair()
    path = tmp_path / "plan.npz"
    np.savez(
        path,
        q_right=q_r,
        q_left=q_l,
        held=np.ones(len(q_r), np.uint8),
        T_right_left=T_RL,
        tool=TOOL,
        fps=20.0,
    )
    plan = cc.read_plan(path)
    assert plan["held"].dtype == bool and plan["fps"] == 20.0
    np.savez(tmp_path / "bad.npz", q_right=q_r)
    with pytest.raises(ValueError, match="not a held plan"):
        cc.read_plan(tmp_path / "bad.npz")
