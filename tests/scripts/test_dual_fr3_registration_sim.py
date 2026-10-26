"""dual_fr3_registration_sim (roadmap G3): the kinematics, the fits, and that a noiseless world comes out exact."""

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import dual_fr3_registration_sim as sim, fr3_kinematics as kin

REPO = Path(__file__).resolve().parents[2]


def _payload_identification():
    path = REPO / "src" / "lerobot" / "robots" / "franka_research3" / "payload_identification.py"
    spec = importlib.util.spec_from_file_location("_fr3_payload_identification", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def test_fk_and_jacobian_are_payload_identifications() -> None:
    reference = _payload_identification().flange_kinematics
    q = np.random.default_rng(0).uniform(kin.Q_LOWER, kin.Q_UPPER, (6, 7))
    T, J = sim.NOMINAL.kinematics(q)
    for i in range(len(q)):
        T_ref, J_ref = reference(q[i])
        assert np.allclose(T[i], T_ref) and np.allclose(J[i], J_ref)


def test_ik_reaches_a_pose_and_seat_moves_the_tool_point_onto_a_target() -> None:
    R_down, seed = sim._down(sim.NOMINAL, np.array([0.45, 0.2, 0.25]))
    cup = np.array([0.0, 0.0, 0.1])
    target = sim._poses_at(sim.dwell_rotations(R_down, 4, 25.0), np.array([0.45, 0.2, 0.25]), cup)
    q, ok = sim.ik_retry(sim.NOMINAL, target, seed)
    assert ok.all()
    assert np.allclose(sim.NOMINAL.point(q, cup), [0.45, 0.2, 0.25], atol=1e-7)
    other = sim.perturbed(np.random.default_rng(1), 5.0)
    moved = sim.seat(other, q, cup, np.tile([0.45, 0.2, 0.25], (len(q), 1)))
    assert np.allclose(other.point(moved, cup), [0.45, 0.2, 0.25], atol=1e-9)


def test_kinematic_scale_is_linear_enough_to_calibrate() -> None:
    rng = np.random.default_rng(2)
    q = rng.uniform(kin.Q_LOWER, kin.Q_UPPER, (200, 7))
    small = np.mean([sim.kinematic_rms(sim.NOMINAL, sim.perturbed(np.random.default_rng(s), 1.0), q) for s in range(8)])
    large = np.mean([sim.kinematic_rms(sim.NOMINAL, sim.perturbed(np.random.default_rng(s), 4.0), q) for s in range(8)])
    assert large / small == pytest.approx(4.0, rel=0.02)


def test_laser_station_fit_recovers_station_and_smr() -> None:
    rng = np.random.default_rng(3)
    q = rng.uniform(kin.Q_LOWER, kin.Q_UPPER, (12, 7))
    T_m = sim.NOMINAL.fk(q)
    T_S_B = sim.rigid(math.radians(150), (2.0, -0.3, -0.2), roll=0.1)
    o = np.array([0.03, -0.01, 0.04])
    p_S = sim.apply(T_S_B, T_m[:, :3, 3] + T_m[:, :3, :3] @ o)
    T, o_fit = sim.laser_station_fit(T_m, p_S)
    assert np.allclose(T, T_S_B, atol=1e-9) and np.allclose(o_fit, o, atol=1e-9)


@pytest.fixture(scope="module")
def noiseless_setup() -> sim.Setup:
    config = replace(
        sim.Config(name="noiseless"),
        seat_sigma_mm=0.0,
        fk_rms_mm=0.0,
        world_sigma_mm=0.0,
        laser_sigma_mm=0.0,
    )
    return sim.build_setup(config)


def test_a_noiseless_world_registers_exactly(noiseless_setup: sim.Setup) -> None:
    assert len(noiseless_setup.usable_balls) >= 4
    row = sim.run_trial(noiseless_setup, np.random.default_rng(4))
    for method in sim.METHODS:
        assert row[method]["tlr_max_mm"] < 1e-3, method
        assert row[method]["reg_max_mm"] < 1e-3, method
    assert row["floor_p95_mm"] < 1e-6
    check = row["closure_check"]
    assert not check["false_alarm"]
    # with exact edges the threshold is world_provenance's floor, which both injected errors exceed
    assert check["threshold_mm"] == pytest.approx(sim.wp.CLOSURE_MIN_THRESHOLD_MM)
    assert check["caught_5mm"] and check["caught_10mm"]
