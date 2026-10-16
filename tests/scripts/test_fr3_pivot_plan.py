"""fr3_pivot_plan: the pivot planned as dwells -- where the arm can turn the insert about the ball, and
whether the planned dwells measure the socket."""
# ruff: noqa: N802, N806  (matrix names follow the maths)

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("mujoco")

from tests.scripts.test_fr3_pivot_auto import _fake_panda_module, _FakeClock  # noqa: E402
from tools.fr3 import (
    fr3_pivot_auto as pa,  # noqa: E402
    fr3_pivot_plan as pp,  # noqa: E402
    fr3_tcp_pivot as tp,  # noqa: E402
)

# A coarse grid: the same checks, a few seconds instead of a minute.
COARSE = {"heading_span_deg": 45.0, "heading_step_deg": 15.0, "pitch_step_deg": 10.0, "roll_step_deg": 10.0}
# The ball on its stand on the table: the stand's foot is 92.2 mm below the ball centre.
BALL = [0.45, 0.0, 0.20]
TABLE_Z = 0.20 - 0.0922


@pytest.fixture(scope="module")
def kin():
    return pa.Kinematics()


@pytest.fixture(scope="module")
def stand():
    return pp.Stand.load(spacing_mm=2.0)


@pytest.fixture(scope="module")
def centre(kin):
    # test_fr3_pivot_auto's capture: the gripper about level (26° down), heading 90° off radial
    q = np.array([-0.63018, 0.22631, 0.14601, -2.21704, 0.87887, 1.63422, -1.68999])
    T = kin.fk(q)
    assert T[:3, 3] == pytest.approx(BALL, abs=1e-4)
    return {"schema": pa.CENTRE_SCHEMA, "q": q.tolist(), "centre_m": T[:3, 3].tolist()}


def test_socket_angles_name_the_tool_heading_its_pitch_and_the_roll_about_it():
    R = pp.socket_rotation(math.radians(30), 0.0, 0.0)
    assert R[:, 2] == pytest.approx([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0], abs=1e-12)
    assert R[:, 0] == pytest.approx([0, 0, 1], abs=1e-12)  # the cup axis up
    R = pp.socket_rotation(0.0, math.radians(10), 0.0)
    assert math.degrees(math.asin(R[2, 2])) == pytest.approx(10)  # tool tip up
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform([-math.pi, -0.6, -1.2], [math.pi, 0.6, 1.2])
        b = pp.socket_angles(pp.socket_rotation(*a))[0]
        assert np.remainder(a - b + math.pi, 2 * math.pi) - math.pi == pytest.approx([0, 0, 0], abs=1e-9)


def test_stand_clearance_matches_the_insert_cad(stand):
    def clear(pitch, roll):
        return stand.socket_clearance(pp.socket_rotation(0.7, math.radians(pitch), math.radians(roll)))

    assert clear(0, 0) == pytest.approx(8.7, abs=0.3)  # cup over the neck: 12.68 cup vs 5 mm neck
    assert (
        clear(-25, 0) == pytest.approx(0.6, abs=0.3) and clear(-30, 0) < 0
    )  # the tool tipped down onto the neck
    assert clear(0, 60) == pytest.approx(1.4, abs=0.3) and clear(0, 70) < 0
    assert clear(-20, 40) < 0 < clear(-20, 30)  # pitch and roll together
    # the stand is turned about the vertical: the heading does not matter
    for h in (-2.0, 0.3, 2.5):
        assert stand.socket_clearance(
            pp.socket_rotation(h, math.radians(-20), math.radians(30))
        ) == pytest.approx(clear(-20, 30), abs=0.05)


def test_fit_sigma_needs_turns_about_two_axes_and_falls_with_more():
    one_axis = [pp.socket_rotation(math.radians(h), 0.0, 0.0) for h in (0, 20, 40)]
    assert np.isinf(pp.fit_sigma(one_axis)).all()
    star = [
        pp.socket_rotation(math.radians(h), math.radians(p), math.radians(r))
        for h, p, r in [(0, 0, 0), (30, 0, 0), (0, -20, 0), (0, 0, 30), (30, -20, 30)]
    ]
    sig = pp.fit_sigma(star)
    assert np.isfinite(sig).all()
    assert pp.fit_sigma(star * 2) == pytest.approx(sig / math.sqrt(2))


def _seated_run(kin, centre, path, delta_mm):
    """The true socket is ``delta_mm`` off corenetic_socket_ee and the ball holds it there."""
    D = np.eye(4)
    D[:3, 3] = np.asarray(delta_mm) / 1e3
    q0 = np.asarray(centre["q"])
    ball = (kin.fk(q0) @ D)[:3, 3]
    qs, q = [], q0
    for R in path["R"][::10]:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, ball
        q, ep, _ = kin.ik(T @ np.linalg.inv(D), q, q0)
        assert ep < 1e-6
        qs.append(q)
    return np.asarray(qs)


def test_planned_dwells_pass_every_check_and_measure_the_true_socket(kin, stand, centre, tmp_path):
    scene = pa.SceneCheck(table_z_m=TABLE_Z)
    params = pp.DwellParams(dwells=8)
    path, qs, names, info, S = pp.plan_dwells(
        centre, kin, scene, stand, params, pp.SurveyParams(table_z_m=TABLE_Z, **COARSE)
    )
    assert info["problems"] == [], info
    assert len(info["dwells_deg"]) == 8 and np.isfinite(info["sigma_per_unit"]["s"]).all()
    assert names[0] == "press" and names[-1] == "release" and names[-2] == "return"
    summary = pa.check_path(path, qs, np.asarray(centre["q"]), kin, scene, params.rate_hz, table_z_m=TABLE_Z)
    assert summary["passed"], summary["problems"]
    assert max(summary["max_joint_speed_deg_s"]) <= pa.MAX_JOINT_SPEED_DEG_S
    assert summary["max_joint_speed_deg_s"][5] <= pa.MAX_J6_SPEED_DEG_S
    more, stand_info = pp.stand_problems(S, path, qs)
    assert more == []
    # the capture has the tool 26° down, the socket already 0.6 mm off the neck: moves may come that
    # close again (less 0.2 mm), not closer
    assert S.checker.min_mm["socket"] == pytest.approx(S.checker.start["socket_mm"] - pp.START_SLACK_MM)
    assert S.checker.min_mm["socket"] < pp.SOCKET_CLEARANCE_MM
    assert stand_info["socket_stand_mm"] >= S.checker.min_mm["socket"]
    # starts and ends unpressed at the capture, pressed 3 mm down in between
    assert path["p"][0] == pytest.approx(BALL, abs=1e-4) and path["p"][-1] == pytest.approx(BALL, abs=1e-4)
    dwell = names.index("dwell1")
    i = int(np.argmax(path["phase"] == dwell))
    assert path["p"][i] - (S.centre_m - path["R"][i] @ S.anchor_m) == pytest.approx([0, 0, -0.003], abs=1e-9)
    # every dwell is held still for dwell_s
    held = (path["phase"] == dwell).sum() / params.rate_hz
    assert held == pytest.approx(params.dwell_s, abs=0.02)

    # The ball holds the true socket wherever the plan turns it: the fit finds the socket, not the frame.
    delta = [0.4, -0.7, 1.2]
    qs_seated = _seated_run(kin, centre, path, delta)
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    T = np.asarray([gripper.fk(q) for q in qs_seated])
    run = {
        "mode": "commanded",
        "q": qs_seated.tolist(),
        "O_T_EE": [list(m.T.reshape(-1)) for m in T],
        "F_ext": [[0, 0, -1.8, 0, 0, 0]] * len(qs_seated),
        "F_ext_frame": "O",
        "completed": True,
    }
    run_path = tmp_path / "20261015_120000_pivot.json"
    run_path.write_text(json.dumps(run))
    out = tmp_path / "fit.json"
    assert tp.main(["fit", str(run_path), "--json", str(out)]) == 0
    fit = json.loads(out.read_text())
    assert fit["socket_frame"]["s_ee_mm"] == pytest.approx(delta, abs=0.05)


def test_a_capture_that_already_breaks_a_check_is_named(kin, stand, centre):
    scene = pa.SceneCheck(table_z_m=TABLE_Z)
    q = np.asarray(centre["q"]).copy()
    q[1] = math.radians(-100.5)  # 0.5° inside panda_py's j2 wall
    T = kin.fk(q)
    moved = {**centre, "q": q.tolist(), "centre_m": T[:3, 3].tolist()}
    path, qs, _, info, _ = pp.plan_dwells(
        moved, kin, scene, stand, pp.DwellParams(dwells=8), pp.SurveyParams(table_z_m=TABLE_Z, **COARSE)
    )
    assert len(qs) == 0
    assert any("起始姿态本身不过检查" in p and "limit: j2" in p for p in info["problems"])


def test_plan_cli_writes_a_dwell_plan_execute_accepts(kin, centre, tmp_path, monkeypatch):
    coarse = pp.SurveyParams
    monkeypatch.setattr(pp, "SurveyParams", lambda **kw: coarse(**{**COARSE, **kw}))
    centre_file = tmp_path / "20261015_120000_centre.json"
    centre_file.write_text(json.dumps(centre))
    out = tmp_path / "20261015_120100_plan.json"
    rc = pa.main(["plan", str(centre_file), "--dwells", "8", "--table-z-m", f"{TABLE_Z}", "--out", str(out)])
    summary = json.loads(out.with_name("20261015_120100_plan.summary.json").read_text())
    assert rc == 0, summary["problems"]
    assert summary["kind"] == "dwells" and summary["passed"] and len(summary["dwells_deg"]) == 8
    plan = json.loads(out.read_text())
    assert plan["schema"] == pa.PLAN_SCHEMA and len(plan["q"]) == len(plan["t"]) == len(plan["quat_xyzw"])
    assert plan["phase_names"][max(plan["phase"])] == "release"

    # execute follows it (a Panda that tracks perfectly) and writes a run fit can read
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    module, calls = _fake_panda_module(gripper, centre["q"])
    monkeypatch.setitem(sys.modules, "panda_py", module)
    clock = _FakeClock()
    monkeypatch.setattr(pa.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(pa.time, "sleep", clock.sleep)
    handlers = {sig: pa.signal.getsignal(sig) for sig in (pa.signal.SIGINT, pa.signal.SIGTERM)}
    run = tmp_path / "20261015_120200_pivot.json"
    try:
        assert pa.main(["execute", str(out), "--out", str(run)]) == 0
    finally:
        for sig, h in handlers.items():
            pa.signal.signal(sig, h)
    recorded = json.loads(run.read_text())
    assert recorded["completed"] and calls["stopped"] == 1
    assert tp.main(["fit", str(run), "--json", str(tmp_path / "f.json")]) == 0
