"""The FR3 replay plans its start for the whole trajectory inside the limits panda_py enforces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config  # noqa: E402
from lerobot.robots.franka_research3.backends import (  # noqa: E402
    PANDA_PY_JOINT_LIMITS_LOWER,
    PANDA_PY_JOINT_LIMITS_UPPER,
    HirolGaussianNewtonKinematicsDriver,
)
from lerobot.robots.franka_research3.trajectory_start import (  # noqa: E402
    _better,
    _Branch,
    plan_trajectory_start,
)

_URDF = (
    Path(__file__).resolve().parents[2]
    / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
)
_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
_FR3_START = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0])


def _ik(**limits) -> HirolGaussianNewtonKinematicsDriver:
    return HirolGaussianNewtonKinematicsDriver(
        urdf_path=str(_URDF), target_frame_name="corenetic_gripper_ee", joint_names=_JOINTS, **limits
    )


def _panda_ik() -> HirolGaussianNewtonKinematicsDriver:
    return _ik(position_limits_lower=PANDA_PY_JOINT_LIMITS_LOWER, position_limits_upper=PANDA_PY_JOINT_LIMITS_UPPER)


def test_ik_stops_at_the_panda_py_wall_not_the_urdf_range():
    urdf_lower, urdf_upper = _ik().joint_limits
    lower, upper = _panda_ik().joint_limits
    # j6: the FR3 goes to 4.5169, panda_py walls it at 3.7525 (10-09 abort at frame 479).
    assert urdf_upper[5] == pytest.approx(4.5169)
    assert upper[5] == pytest.approx(3.7525)
    # The intersection: never wider than either.
    assert np.all(lower >= urdf_lower) and np.all(lower >= PANDA_PY_JOINT_LIMITS_LOWER)
    assert np.all(upper <= urdf_upper) and np.all(upper <= PANDA_PY_JOINT_LIMITS_UPPER)
    beyond = _FR3_START.copy()
    beyond[5] = 4.1
    target = _ik().forward_kinematics(beyond)
    assert _panda_ik().inverse_kinematics(beyond, target)[5] <= 3.7525


def test_fr3_robot_ik_respects_the_controller_wall_by_default():
    cfg = FrankaResearch3Config(
        robot_ip="192.168.1.206",
        urdf_path=str(_URDF),
        target_frame_name="corenetic_gripper_ee",
        ik_solver="hirol_gaussian_newton",
    )
    assert FrankaResearch3(cfg)._make_kinematics_driver().joint_limits[1][5] == pytest.approx(3.7525)
    cfg.ik_respect_controller_joint_limits = False
    assert FrankaResearch3(cfg)._make_kinematics_driver().joint_limits[1][5] == pytest.approx(4.5169)


def _poses_along(q_from, q_to, frames=30):
    fk = _ik()
    return [fk.forward_kinematics(q_from + t * (np.asarray(q_to) - q_from)) for t in np.linspace(0.0, 1.0, frames)]


def test_plan_tracks_every_frame_with_room_to_the_limits():
    ik = _panda_ik()
    lower, upper = ik.joint_limits
    q_from = np.array([0.3, -0.5, 0.2, -2.2, 0.3, 2.5, 0.5])
    poses = _poses_along(q_from, [0.5, -0.3, 0.1, -2.0, 0.2, 3.5, 0.7])

    plan = plan_trajectory_start(ik, poses, lower, upper, seeds=[_FR3_START], num_random_seeds=16)

    assert plan.feasible and plan.failed_at_frame == -1
    assert plan.min_limit_margin_rad >= 0.1
    assert plan.max_position_error_m < 0.002
    start = ik.forward_kinematics(np.asarray(plan.joints_rad))
    assert np.allclose(start[:3, 3], poses[0][:3, 3], atol=1e-4)
    assert plan.to_dict()["joints_rad"] == pytest.approx(plan.joints_rad)
    # Deterministic: the preview and a rerun hand the real replay the same joints.
    again = plan_trajectory_start(ik, poses, lower, upper, seeds=[_FR3_START], num_random_seeds=16)
    assert again.joints_rad == pytest.approx(plan.joints_rad)


def test_plan_says_so_when_no_start_reaches_the_trajectory():
    ik = _panda_ik()
    lower, upper = ik.joint_limits
    far = np.eye(4)
    far[:3, 3] = [2.0, 0.0, 0.5]
    plan = plan_trajectory_start(ik, [far], lower, upper, seeds=[_FR3_START], num_random_seeds=4)
    assert not plan.feasible
    assert "frame 0" in plan.reason


def test_feasible_branch_nearest_the_reference_wins_over_a_wider_margin():
    reference = np.zeros(7)
    near = _Branch(joints_rad=np.full(7, 0.1), min_margin_rad=0.11)
    far = _Branch(joints_rad=np.full(7, 1.0), min_margin_rad=0.30)
    stuck = _Branch(joints_rad=np.zeros(7), failed_at_frame=400)
    stuck_later = _Branch(joints_rad=np.full(7, 2.0), failed_at_frame=600)
    assert _better(near, far, reference)
    assert _better(far, stuck, reference)  # any branch that finishes beats one that does not
    assert _better(stuck_later, stuck, reference)  # among failures, the one that got furthest
