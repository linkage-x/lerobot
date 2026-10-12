# ruff: noqa: N806  (matrix names: T, J, R, A follow the maths)
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.franka_research3.backends import PANDA_PY_JOINT_LIMITS_LOWER, PANDA_PY_JOINT_LIMITS_UPPER
from lerobot.robots.franka_research3.payload_identification import (
    default_poses,
    fit_difference,
    fit_load,
    flange_kinematics,
    reach_groups,
    urdf_link_load,
    visit_sequence,
)

_ASSETS = Path(__file__).resolve().parents[2] / "src/lerobot/robots/franka_research3/assets/franka_fr3"
_XML = _ASSETS / "fr3_corenetic_gripper.xml"
_URDF = _ASSETS / "fr3_corenetic_gripper.urdf"


def _sim_run(tmp_path: Path, name: str, **kw) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    args = argparse.Namespace(
        xml=_XML,
        without_smr=False,
        noise_nm=0.0,
        bias_nm=0.0,
        arm_model_error=0.0,
        hysteresis_nm=0.0,
        approach_rad=0.03,
        seed=0,
        out=tmp_path / f"{name}.json",
    )
    for k, v in kw.items():
        setattr(args, k, v)
    runtime.simulate(args)
    run = json.loads(args.out.read_text())
    p = run["poses"]
    return (
        run,
        np.array([s["q"] for s in p]),
        np.array([s["tau_J"] for s in p]),
        np.array([s["g_arm"] for s in p]),
    )


def test_default_poses_stay_inside_panda_py_joint_walls():
    poses = default_poses()
    assert poses.shape == (32, 7)
    margin = 0.05
    assert np.all(poses >= np.array(PANDA_PY_JOINT_LIMITS_LOWER) + margin)
    assert np.all(poses <= np.array(PANDA_PY_JOINT_LIMITS_UPPER) - margin)
    assert len(np.unique(reach_groups(poses))) == 3


def test_flange_kinematics_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    f8 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link8")
    rng = np.random.default_rng(0)
    for q in [*default_poses()[::5], *rng.uniform(-1.5, 1.5, (3, 7))]:
        data.qpos[:] = 0.0
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        T, _ = flange_kinematics(q)
        np.testing.assert_allclose(T[:3, 3], data.xpos[f8], atol=1e-9)
        np.testing.assert_allclose(T[:3, :3], data.xmat[f8].reshape(3, 3), atol=1e-9)


def test_fit_recovers_the_mujoco_tool_exactly(tmp_path):
    pytest.importorskip("mujoco")
    run, qs, tau, g_arm = _sim_run(tmp_path, "ideal")
    est = fit_load(qs, tau, g_arm)
    truth = run["sim"]["truth"]
    assert est.mass_kg == pytest.approx(truth["mass_kg"], abs=1e-6)
    np.testing.assert_allclose(est.com_flange_m * 1000, truth["com_flange_mm"], atol=0.01)


def test_per_reach_offsets_absorb_the_bare_arm_model_error(tmp_path):
    """Global offsets let a 1 % arm-model error into the load; per-reach offsets keep most of it out.

    What is left (~13 g here) is the error of links 5-7, which move with the wrist inside a reach.
    """
    pytest.importorskip("mujoco")
    run, qs, tau, g_arm = _sim_run(tmp_path, "arm", arm_model_error=0.01, bias_nm=0.3)
    truth = run["sim"]["truth"]["mass_kg"]
    glob = fit_load(qs, tau, g_arm)
    per_reach = fit_load(qs, tau, g_arm, groups=reach_groups(qs))
    assert abs(glob.mass_kg - truth) > 0.04
    assert abs(per_reach.mass_kg - truth) < 0.3 * abs(glob.mass_kg - truth)


def test_smr_difference_cancels_offsets_and_arm_model_error(tmp_path):
    pytest.importorskip("mujoco")
    kw = {"arm_model_error": 0.05, "bias_nm": 0.3}
    _, qs, tau_in, _ = _sim_run(tmp_path, "smr_in", **kw)
    _, _, tau_out, _ = _sim_run(tmp_path, "smr_out", without_smr=True, **kw)
    est = fit_difference(qs, tau_in, tau_out)
    m, c = urdf_link_load(_URDF, "link_corenetic_smr")
    assert est.mass_kg == pytest.approx(m, abs=1e-6)
    np.testing.assert_allclose(est.com_flange_m, c, atol=1e-5)


def test_two_way_approach_cancels_joint_friction_hysteresis(tmp_path):
    """10-12 on the FR3: measured once per pose, +-0.1..0.4 Nm of friction hysteresis moved the load by
    hundreds of grams depending on how it was modelled. Reaching each pose from both sides cancels it."""
    pytest.importorskip("mujoco")
    kw = {"hysteresis_nm": 0.25, "bias_nm": 0.3}
    run, qs, tau, g_arm = _sim_run(tmp_path, "two_way", **kw)
    truth = run["sim"]["truth"]
    est = fit_load(qs, tau, g_arm)
    assert est.mass_kg == pytest.approx(truth["mass_kg"], abs=1e-3)
    np.testing.assert_allclose(est.com_flange_m * 1000, truth["com_flange_mm"], atol=0.5)
    h = np.mean([np.subtract(p["tau_J_up"], p["tau_J_down"]) / 2 for p in run["poses"]], axis=0)
    np.testing.assert_allclose(h, run["sim"]["hysteresis_nm"], atol=1e-4)  # stored rounded

    run, qs, tau, g_arm = _sim_run(tmp_path, "one_way", approach_rad=0.0, **kw)
    assert abs(fit_load(qs, tau, g_arm).mass_kg - truth["mass_kg"]) > 0.05


def test_visit_sequence_reaches_every_pose_from_both_sides():
    poses = default_poses()
    seq = visit_sequence(poses, 0.03, PANDA_PY_JOINT_LIMITS_LOWER, PANDA_PY_JOINT_LIMITS_UPPER)
    assert len(seq) == 4 * len(poses)
    for k in range(0, len(seq), 4):
        (pre_up, t0, i), (q_up, tag_up, _), (pre_dn, t1, _), (q_dn, tag_dn, _) = seq[k : k + 4]
        assert (t0, tag_up, t1, tag_dn) == (None, "up", None, "down")
        np.testing.assert_array_equal(q_up, poses[i])
        assert np.all(q_up - pre_up > 0) and np.all(q_dn - pre_dn < 0)


def test_urdf_smr_link_in_flange_frame():
    m, c = urdf_link_load(_URDF, "link_corenetic_smr")
    assert m == pytest.approx(0.14)
    np.testing.assert_allclose(c * 1000, [49.464, -49.815, 103.891], atol=0.01)


class _FakeArm:
    """move_to_joint_position fails with the given errors first, then succeeds."""

    def __init__(self, failures: list[dict]):
        self.failures = list(failures)
        self.errors: dict = {}
        self.recovered = 0

    def move_to_joint_position(self, q, speed_factor):
        if self.failures:
            self.errors = self.failures.pop(0)
            return False
        self.errors = {}
        return True

    def recover(self):
        self.recovered += 1

    def read_once(self):
        return argparse.Namespace(last_motion_errors=argparse.Namespace(**self.errors))


def _move(arm: _FakeArm) -> bool:
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    args = argparse.Namespace(speed_factor=0.1, max_retries=3)
    return runtime._move(arm, arm, np.zeros(7), args)


def test_move_retries_only_the_communication_reflex(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    comm = {"communication_constraints_violation": True}
    arm = _FakeArm([comm, comm])
    assert _move(arm) and arm.recovered == 2

    collision = {"communication_constraints_violation": False, "cartesian_reflex": True}
    arm = _FakeArm([collision])
    assert not _move(arm) and arm.recovered == 0

    arm = _FakeArm([comm] * 4)  # gives up after max_retries
    assert not _move(arm) and arm.recovered == 3


def test_desk_config_keeps_the_template_tcp_and_writes_the_identified_load(tmp_path):
    pytest.importorskip("mujoco")
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    run, *_ = _sim_run(tmp_path, "no_smr", without_smr=True, hysteresis_nm=0.25)
    template = {
        "id": "old",
        "name": "old",
        "inertial": {"mass": 0.75},
        "tcp": {
            "rotation": {"pitch": 0, "roll": 0, "yaw": -0.785},
            "translation": {"x": 0, "y": 0, "z": 0.172},
        },
        "deviceId": "ee-gripper",
    }
    (tmp_path / "template.json").write_text(json.dumps(template))
    args = argparse.Namespace(
        run=tmp_path / "no_smr.json",
        template=tmp_path / "template.json",
        name="gripper + cam",
        out=tmp_path / "desk.json",
        urdf=_URDF,
    )
    runtime.desk_config(args)
    cfg = json.loads(args.out.read_text())
    assert cfg["id"] != "old" and cfg["name"] == "gripper + cam"
    assert cfg["tcp"] == template["tcp"] and cfg["deviceId"] == "ee-gripper"
    truth = run["sim"]["truth"]
    assert cfg["inertial"]["mass"] == pytest.approx(truth["mass_kg"], abs=1e-3)
    com = [cfg["inertial"]["centerOfMass"][k] * 1000 for k in "xyz"]
    np.testing.assert_allclose(com, truth["com_flange_mm"], atol=0.5)
    assert set(cfg["inertial"]["inertia"]) == {"x11", "x12", "x13", "x22", "x23", "x33"}
