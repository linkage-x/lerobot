# ruff: noqa: N806  (matrix names: T, J, R, A follow the maths)
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.franka_research3.backends import (
    PANDA_PY_JOINT_LIMITS_LOWER,
    PANDA_PY_JOINT_LIMITS_UPPER,
    PANDA_PY_JOINT_POSITION_START,
)
from lerobot.robots.franka_research3.payload_identification import (
    default_poses,
    fit_difference,
    fit_load,
    flange_kinematics,
    franka_external_wrench,
    load_regressor,
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


def test_desk_config_writes_one_config_per_smr_state_with_the_template_tcp_on_request(tmp_path):
    pytest.importorskip("mujoco")
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    # no sensor offsets: then the load the robot's model needs is the physical one
    run_in, *_ = _sim_run(tmp_path, "smr_in", hysteresis_nm=0.25)
    run_out, *_ = _sim_run(tmp_path, "smr_out", without_smr=True, hysteresis_nm=0.25)
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
        runs=[tmp_path / "smr_in.json", tmp_path / "smr_out.json"],
        template=tmp_path / "template.json",
        name_smr="gripper + cam + smr",
        name_no_smr="gripper + cam",
        out_dir=tmp_path / "desk",
        stem="load",
        urdf=_URDF,
        tcp_template=True,
    )
    runtime.desk_config(args)
    for stem, run, name in (
        ("load_smr", run_in, "gripper + cam + smr"),
        ("load_no_smr", run_out, "gripper + cam"),
    ):
        cfg = json.loads((tmp_path / "desk" / f"{stem}.json").read_text())
        assert cfg["id"] != "old" and cfg["name"] == name
        assert cfg["tcp"] == template["tcp"] and cfg["deviceId"] == "ee-gripper"
        truth = run["sim"]["truth"]
        assert cfg["inertial"]["mass"] == pytest.approx(truth["mass_kg"], abs=1e-3)
        com = [cfg["inertial"]["centerOfMass"][k] * 1000 for k in "xyz"]
        np.testing.assert_allclose(com, truth["com_flange_mm"], atol=0.5)
        assert set(cfg["inertial"]["inertia"]) == {"x11", "x12", "x13", "x22", "x23", "x33"}


def test_franka_external_wrench_of_an_unconfigured_mass_is_its_weight():
    # 0.3 kg more than configured, 50 mm off the flange axis: the robot reports +m*g up (the force it holds
    # the extra mass with) and the matching moment about the base origin.
    m, c = 0.3, np.array([0.05, -0.02, 0.06])
    for q in default_poses()[::7]:
        T, _ = flange_kinematics(q)
        w = franka_external_wrench(q, load_regressor(q) @ np.r_[m, m * c])
        f = np.array([0.0, 0.0, m * 9.81])
        np.testing.assert_allclose(w[:3], f, atol=1e-9)
        np.testing.assert_allclose(w[3:], np.cross(T[:3, :3] @ c + T[:3, 3], f), atol=1e-9)


def test_six_d_poses_turn_every_tool_axis_up_inside_the_joint_walls():
    mujoco = pytest.importorskip("mujoco")
    from lerobot.robots.franka_research3.payload_identification import pose_set, six_d_poses

    poses = six_d_poses()
    lo, hi = np.asarray(PANDA_PY_JOINT_LIMITS_LOWER), np.asarray(PANDA_PY_JOINT_LIMITS_UPPER)
    model = mujoco.MjModel.from_xml_path(str(_XML))
    # the FR3's own limits (the MJCF's) as well as panda_py's walls: the FR3's j6 stops at 0.5445
    lo_all, hi_all = np.maximum(lo, model.jnt_range[:7, 0]), np.minimum(hi, model.jnt_range[:7, 1])
    assert np.all(poses > lo_all + 0.1) and np.all(poses < hi_all - 0.1)
    # the upper arm stays in the start pose: only the wrist moves
    np.testing.assert_allclose(poses[:, :4], np.tile(poses[0, :4], (len(poses), 1)))
    visit_sequence(poses, 0.03, lo_all, hi_all)  # every pose reachable from both sides
    data = mujoco.MjData(model)
    ft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_sensor_ft")
    faces = set()
    for q in poses:
        data.qpos[:] = 0.0
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        up = data.xmat[ft].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        k = int(np.argmax(np.abs(up)))
        assert abs(up[k]) > 0.85
        faces.add(f"{'xyz'[k]}{'+' if up[k] > 0 else '-'}")
    assert faces == {"x+", "x-", "y+", "y-", "z+", "z-"}
    assert len(pose_set("payload+six_d")) == len(default_poses()) + len(poses)


def test_desk_config_of_the_bare_gripper_leaves_the_adaptor_out(tmp_path):
    pytest.importorskip("mujoco")
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    run, *_ = _sim_run(tmp_path, "gripper", tool_config="gripper", poses="payload+six_d", hysteresis_nm=0.2)
    assert run["tool_config"] == "gripper" and not run["smr_present"]
    full, *_ = _sim_run(tmp_path, "full")
    m_adaptor = full["sim"]["truth"]["mass_kg"] - run["sim"]["truth"]["mass_kg"]
    assert m_adaptor == pytest.approx(0.046529 + 0.1871 + 0.14, abs=1e-6)  # mount + camera case + SMR
    (tmp_path / "template.json").write_text(json.dumps({"id": "old", "tcp": {"z": 0.172}, "deviceId": "g"}))
    args = argparse.Namespace(
        runs=[tmp_path / "gripper.json"],
        template=tmp_path / "template.json",
        name=["gripper=Corenetic gripper"],
        name_smr=None,
        name_no_smr=None,
        out_dir=tmp_path / "desk",
        stem="load",
        urdf=_URDF,
        report=tmp_path / "report.json",
    )
    assert runtime.desk_config(args) == 0
    cfg = json.loads((tmp_path / "desk" / "load_gripper.json").read_text())
    assert cfg["name"] == "Corenetic gripper"
    assert cfg["inertial"]["mass"] == pytest.approx(run["sim"]["truth"]["mass_kg"], abs=1e-3)
    entry = json.loads((tmp_path / "report.json").read_text())["configs"]["gripper"]
    assert entry["n_poses"] == len(default_poses()) + 13
    from lerobot.robots.franka_research3.payload_identification import urdf_subtree_inertial

    m_full_cad = urdf_subtree_inertial(_URDF)[0]
    assert m_full_cad - entry["cad"]["mass_kg"] == pytest.approx(m_adaptor, abs=1e-3)
    # a run of a configuration without a name is refused
    args.name = []
    with pytest.raises(SystemExit, match="no Desk name"):
        runtime.desk_config(args)


def test_desk_tcp_defaults_to_the_new_tips_and_takes_an_fr3_pivot_through_the_urdf(tmp_path):
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    template = {"rotation": {"roll": 0, "pitch": 0, "yaw": -0.7853981633974483}, "translation": {"x": 0, "y": 0, "z": 0.17199}}
    tcp, src = runtime.desk_tcp(template, urdf=_URDF)
    assert src["kind"] == "estimate" and tcp["translation"] == {"x": 0.0, "y": 0.0, "z": runtime.NEW_TIP_TCP_Z_M}
    assert tcp["rotation"] == template["rotation"]
    assert runtime.desk_tcp(template, keep_template=True, urdf=_URDF)[0] == template
    assert runtime.desk_tcp(template, tcp_z_m=0.2, urdf=_URDF)[0]["translation"]["z"] == 0.2
    # corenetic_gripper_ee is the template's TCP (z 171.99 mm, yaw -45 deg): a socket 18 mm further out
    # along its z and 1 mm along its x lands at z 189.99 and x/y rotated by -45 deg in the flange frame
    fit = tmp_path / "x_pivot.fit.json"
    fit.write_text(json.dumps({"frame": "corenetic_gripper_ee", "robot_ip": "ip", "fk": {"s_ee_mm": [1.0, 0.0, 18.0], "s_sigma_mm": [0.1, 0.1, 0.2]}}))
    tcp, src = runtime.desk_tcp(template, pivot=fit, urdf=_URDF)
    assert src["kind"] == "pivot" and src["s_ee_mm"] == [1.0, 0.0, 18.0]
    t = tcp["translation"]
    assert t["z"] == pytest.approx(0.18999, abs=2e-5)
    assert (t["x"], t["y"]) == pytest.approx((0.001 * math.cos(-math.pi / 4), 0.001 * math.sin(-math.pi / 4)), abs=2e-5)
    assert tcp["rotation"] == template["rotation"]
    fit.write_text(json.dumps({"fk": {}}))
    with pytest.raises(SystemExit):
        runtime.desk_tcp(template, pivot=fit, urdf=_URDF)


def test_six_d_zero_pose_points_the_fingertips_straight_up_inside_both_limit_sets():
    mujoco = pytest.importorskip("mujoco")
    from lerobot.robots.franka_research3.payload_identification import pose_set, six_d_zero_pose

    q = six_d_zero_pose()
    model = mujoco.MjModel.from_xml_path(str(_XML))
    lo = np.maximum(PANDA_PY_JOINT_LIMITS_LOWER, model.jnt_range[:7, 0])
    hi = np.minimum(PANDA_PY_JOINT_LIMITS_UPPER, model.jnt_range[:7, 1])
    assert np.all(q > lo + 0.3) and np.all(q < hi - 0.3)
    start = np.array(PANDA_PY_JOINT_POSITION_START)
    assert q[[0, 2, 4]].tolist() == [0.0, 0.0, 0.0] and q[6] == pytest.approx(start[6])  # planar, j7 unturned
    data = mujoco.MjData(model)
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    ft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_sensor_ft")
    assert data.xmat[ft].reshape(3, 3)[2, 2] == pytest.approx(1.0, abs=1e-9)  # sensor z (fingertips) up
    route = pose_set("six_d_zero")
    np.testing.assert_array_equal(route[0], q)
    np.testing.assert_array_equal(route[-1], start)  # check-poses covers the way back too


def test_the_zero_route_is_the_only_way_between_its_waypoints():
    from lerobot.robots.franka_research3.payload_identification import (
        SIX_D_ZERO_ROUTE,
        six_d_zero_moves,
        six_d_zero_waypoints,
    )

    points = six_d_zero_waypoints()
    np.testing.assert_allclose(points["start"], PANDA_PY_JOINT_POSITION_START)
    assert set(SIX_D_ZERO_ROUTE) == set(points)
    moves = six_d_zero_moves()
    assert frozenset(("start", "six_d_zero")) in moves and frozenset(("check_xplus", "start")) in moves
    # from the zero pose straight to a side pose: a path check-poses never sampled
    assert frozenset(("six_d_zero", "check_xplus")) not in moves
    assert frozenset(("six_d_zero", "check_yminus")) not in moves
    # the re-check poses move only the wrist from the start pose
    for name in ("check_xplus", "check_yminus"):
        assert np.array_equal(points[name][:4], points["start"][:4])


def test_move_waypoint_only_moves_along_the_checked_route(monkeypatch, capsys):
    import sys
    import types

    from lerobot.robots.franka_research3.payload_identification import six_d_zero_waypoints
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    points = six_d_zero_waypoints()
    arm = {"q": points["start"].copy(), "moves": []}

    class Robot:
        def read_once(self):
            return types.SimpleNamespace(q=arm["q"].tolist(), last_motion_errors=None)

    class Panda:
        def __init__(self, ip):
            pass

        def get_robot(self):
            return Robot()

        def move_to_joint_position(self, q, speed_factor):
            arm["moves"].append(speed_factor)
            arm["q"] = np.asarray(q)
            return True

        def stop_controller(self):
            pass

    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(Panda=Panda))

    def run(to):
        return runtime.move_waypoint(argparse.Namespace(robot_ip="x", to=to, speed_factor=0.1, max_retries=0))

    assert run("check_xplus") == 0 and "REACHED check_xplus" in capsys.readouterr().out
    # from a side pose straight up: check-poses never sampled that path
    assert run("six_d_zero") == 1 and "not a move of the checked route" in capsys.readouterr().out
    assert run("start") == 0 and run("six_d_zero") == 0
    np.testing.assert_allclose(arm["q"], points["six_d_zero"])
    arm["q"] = points["start"] + 0.2  # pushed away / left after a replay
    assert run("six_d_zero") == 1 and "no waypoint" in capsys.readouterr().out
    assert len(arm["moves"]) == 3


def test_a_cold_box_gets_a_reminder_not_a_refusal():
    from tools.fr3 import fr3_payload_identification_runtime as runtime

    text = runtime.six_d_warmup_warning(23 * 60)
    assert "才上电 23 分钟" in text and "预热 30 分钟" in text and "照常继续" in text
    assert runtime.six_d_warmup_warning(31 * 60) == ""
    assert runtime.six_d_warmup_warning(None) == ""  # no BOX clock: nothing to say
