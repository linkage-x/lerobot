"""urdf_kinematics (roadmap docs/monte02_left_arm_roadmap.html, M1.3): any serial chain read from a URDF.

On the FR3 it must give what the DH table of fr3_kinematics gives; on Monte02 (tilted shoulder axis, trunk
joints locked) its FK, Jacobian and IK are checked against themselves. The Monte02 tests skip without the
private opencv_kalibr submodule.
"""

# ruff: noqa: N806
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import fr3_kinematics as fr3, urdf_kinematics as uk

MONTE02 = Path(__file__).resolve().parents[2] / (
    "third_party/opencv_kalibr/monte02_h_v1_description/urdf/monte02_h_v1_gmsl2cam.urdf"
)
needs_monte02 = pytest.mark.skipif(not MONTE02.exists(), reason="opencv_kalibr submodule not checked out")
TRUNK = {"joint_t1_knee_pitch": 0.0, "joint_t2_hip_pitch": 0.0, "joint_t3_waist_yaw": 0.0}
LEFT_ARM = tuple(
    f"joint_l{i}_{n}"
    for i, n in enumerate(
        ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll"],
        start=1,
    )
)


def _random_q(chain: uk.Chain, n: int, seed: int = 0) -> np.ndarray:
    lo, hi = chain.q_limits()
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, chain.dof))


# --- FR3: the URDF chain is the DH table --------------------------------------------------------------------


def test_fr3_urdf_chain_is_the_dh_table() -> None:
    chain = uk.load_chain(fr3.FR3_URDF, "fr3_link0", fr3.FLANGE_LINK)
    assert chain.joint_names == tuple(f"fr3_joint{i}" for i in range(1, 8))
    q = np.random.default_rng(1).uniform(fr3.Q_LOWER, fr3.Q_UPPER, size=(50, 7))
    point = np.array([0.01, -0.02, 0.2])
    T, J = chain.kinematics(q, point)
    T_dh, J_dh = fr3.NOMINAL.kinematics(q, point)
    np.testing.assert_allclose(T, T_dh, atol=1e-9)
    np.testing.assert_allclose(J, J_dh, atol=1e-9)


def test_fr3_tool_frame_is_the_fixed_chain_after_the_flange() -> None:
    chain = uk.load_chain(fr3.FR3_URDF, "fr3_link0", "corenetic_socket_ee")
    q = np.random.default_rng(2).uniform(fr3.Q_LOWER, fr3.Q_UPPER, size=(20, 7))
    tool = fr3.urdf_fixed_transform("corenetic_socket_ee")
    np.testing.assert_allclose(chain.fk(q), fr3.NOMINAL.fk(q) @ tool, atol=1e-9)
    np.testing.assert_allclose(uk.fixed_transform(fr3.FR3_URDF, fr3.FLANGE_LINK, "corenetic_socket_ee"), tool, atol=1e-12)


def test_fr3_ik_agrees_with_fr3_kinematics() -> None:
    chain = uk.load_chain(fr3.FR3_URDF, "fr3_link0", fr3.FLANGE_LINK)
    limits = (fr3.Q_LOWER, fr3.Q_UPPER)
    rng = np.random.default_rng(3)
    q_true = np.clip(fr3.Q_READY + rng.normal(0.0, 0.3, size=(40, 7)), fr3.Q_LOWER, fr3.Q_UPPER)
    T_target = fr3.NOMINAL.fk(q_true)
    q, ok = uk.ik(chain, T_target, fr3.Q_READY, limits=limits)
    q_ref, ok_ref = fr3.ik(fr3.NOMINAL, T_target, fr3.Q_READY)
    np.testing.assert_array_equal(ok, ok_ref)
    np.testing.assert_allclose(q[ok], q_ref[ok], atol=1e-9)
    assert ok.mean() > 0.9


# --- chain construction -------------------------------------------------------------------------------------


def test_load_chain_refuses_what_would_silently_be_another_robot() -> None:
    with pytest.raises(ValueError, match="does not descend"):
        uk.load_chain(fr3.FR3_URDF, "fr3_link3", "fr3_link1")
    with pytest.raises(ValueError, match="no link"):
        uk.load_chain(fr3.FR3_URDF, "fr3_link0", "fr3_link9")
    with pytest.raises(ValueError, match="not between"):
        uk.load_chain(fr3.FR3_URDF, "fr3_link0", "fr3_link4", locked={"fr3_joint6": 0.0})
    with pytest.raises(ValueError, match="outside its limits"):
        uk.load_chain(fr3.FR3_URDF, "fr3_link0", "fr3_link8", locked={"fr3_joint4": 0.5})
    with pytest.raises(ValueError, match="mimic"):
        uk.load_chain(fr3.FR3_URDF, "link_gripper_base", "link_gripper_tip_left")
    with pytest.raises(ValueError, match="not fixed"):
        uk.fixed_transform(fr3.FR3_URDF, "fr3_link0", "fr3_link8")


def test_locking_a_joint_is_the_full_chain_at_that_value() -> None:
    full = uk.load_chain(fr3.FR3_URDF, "fr3_link0", fr3.FLANGE_LINK)
    locked = uk.load_chain(fr3.FR3_URDF, "fr3_link0", fr3.FLANGE_LINK, locked={"fr3_joint1": 0.4})
    assert locked.dof == 6 and "fr3_joint1" not in locked.joint_names
    q6 = _random_q(locked, 10)
    q7 = np.hstack([np.full((10, 1), 0.4), q6])
    np.testing.assert_allclose(locked.fk(q6), full.fk(q7), atol=1e-12)
    assert locked.fingerprint != full.fingerprint
    relocked = uk.load_chain(fr3.FR3_URDF, "fr3_link0", fr3.FLANGE_LINK, locked={"fr3_joint1": 0.41})
    assert relocked.fingerprint != locked.fingerprint


# --- Monte02 left arm ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def left_arm() -> uk.Chain:
    return uk.load_chain(MONTE02, "link_b0_base", "lt_corenetic_socket_ee", locked=TRUNK)


@needs_monte02
def test_monte02_left_arm_is_seven_joints_with_the_urdf_limits(left_arm: uk.Chain) -> None:
    assert left_arm.joint_names == LEFT_ARM
    np.testing.assert_allclose(left_arm.lower[5:], [-1.5708, -1.5708])
    np.testing.assert_allclose(left_arm.upper[5:], [1.5708, 1.5708])
    np.testing.assert_allclose(left_arm.axes[0], [0.0, -0.99027, -0.13917], atol=1e-5)  # the tilted shoulder


@needs_monte02
def test_monte02_locked_trunk_is_the_ten_joint_chain(left_arm: uk.Chain) -> None:
    full = uk.load_chain(MONTE02, "link_b0_base", "lt_corenetic_socket_ee")
    assert full.joint_names[:3] == tuple(TRUNK)
    trunk = np.array([0.2, -0.3, 0.5])
    bent = uk.load_chain(MONTE02, "link_b0_base", "lt_corenetic_socket_ee", locked=dict(zip(TRUNK, trunk, strict=True)))
    q = _random_q(left_arm, 10)
    np.testing.assert_allclose(bent.fk(q), full.fk(np.hstack([np.tile(trunk, (10, 1)), q])), atol=1e-12)
    # the arm alone, from its own base, then the trunk's fixed+locked part in front of it
    arm = uk.load_chain(MONTE02, "link_l0_arm_base", "lt_corenetic_socket_ee")
    base = uk.load_chain(MONTE02, "link_b0_base", "link_l0_arm_base", locked=TRUNK)
    assert base.dof == 0
    np.testing.assert_allclose(left_arm.fk(q), base.fk(np.zeros((1, 0))) @ arm.fk(q), atol=1e-12)


@needs_monte02
def test_monte02_socket_is_the_fixed_chain_after_the_sensor(left_arm: uk.Chain) -> None:
    to_sensor = uk.load_chain(MONTE02, "link_b0_base", "link_l_sensor_ft", locked=TRUNK)
    q = _random_q(left_arm, 5)
    tool = uk.fixed_transform(MONTE02, "link_l_sensor_ft", "lt_corenetic_socket_ee")
    np.testing.assert_allclose(tool[:3, 3], [-0.00206, -0.00096, 0.18324], atol=1e-5)
    np.testing.assert_allclose(left_arm.fk(q), to_sensor.fk(q) @ tool, atol=1e-12)


@needs_monte02
def test_monte02_jacobian_is_the_derivative_of_fk(left_arm: uk.Chain) -> None:
    q = _random_q(left_arm, 6, seed=4)
    point = np.array([0.0, 0.03, -0.01])
    T, J = left_arm.kinematics(q, point)
    h = 1e-6
    for i in range(left_arm.dof):
        dq = np.zeros(left_arm.dof)
        dq[i] = h
        Tp, Tm = left_arm.fk(q + dq), left_arm.fk(q - dq)
        dp = (left_arm.point(q + dq, point) - left_arm.point(q - dq, point)) / (2 * h)
        dR = (Tp[:, :3, :3] - Tm[:, :3, :3]) / (2 * h) @ T[:, :3, :3].transpose(0, 2, 1)
        w = np.stack([dR[:, 2, 1], dR[:, 0, 2], dR[:, 1, 0]], axis=1)
        np.testing.assert_allclose(J[:, :3, i], dp, atol=1e-7)
        np.testing.assert_allclose(J[:, 3:, i], w, atol=1e-7)


@needs_monte02
def test_monte02_ik_round_trip(left_arm: uk.Chain) -> None:
    q_true = _random_q(left_arm, 60, seed=5)
    T_target = left_arm.fk(q_true)
    lo, hi = left_arm.q_limits()
    q, ok = uk.ik_retry(left_arm, T_target, (lo + hi) / 2)
    pos, rot = uk.pose_error(T_target, left_arm.fk(q))
    assert ok.mean() > 0.9, ok.mean()
    assert pos[ok].max() < uk.IK_TOL_MM and rot[ok].max() < uk.IK_TOL_DEG
    assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)
    # a control loop: seeded from a posture nearby, early exit, pulled toward where it stands
    q_near = np.clip(q_true + 0.01, lo, hi)
    q2, ok2 = uk.ik_retry(left_arm, T_target, q_near, early_exit=True, rest=q_near)
    assert ok2.mean() > 0.95
    # stays on the posture it was seeded from; one pose with shoulder roll on its limit slides 0.13 rad along
    # the self-motion (7 joints, 6 constraints), so not every pose
    assert np.quantile(np.abs(q2[ok2] - q_true[ok2]).max(axis=1), 0.95) < 0.02


@needs_monte02
def test_monte02_out_of_reach_does_not_converge(left_arm: uk.Chain) -> None:
    T = left_arm.fk(np.zeros((1, 7)))
    T[0, :3, 3] += [3.0, 0.0, 0.0]
    _, ok = uk.ik_retry(left_arm, T, np.zeros(7))
    assert not ok.any()
