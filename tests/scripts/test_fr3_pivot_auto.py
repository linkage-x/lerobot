"""fr3_pivot_auto: a planned, pressed pivot measures the socket -- not the frame it was planned on."""
# ruff: noqa: N806  (matrix names follow the maths)

from __future__ import annotations

import json
import types

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from tools.fr3 import (
    fr3_pivot_auto as pa,  # noqa: E402
    fr3_tcp_pivot as tp,  # noqa: E402
)

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
FAST = {"speed_deg_s": 25.0, "ramp_s": 0.5}  # fewer samples; the geometry is the same


@pytest.fixture(scope="module")
def kin():
    return pa.Kinematics()


@pytest.fixture(scope="module")
def centre(kin):
    # As in the 10-13 photo: the gripper reaching in about level (26° down), the cup facing down onto a
    # ball 80 mm above the table (fr3_base z = 120 mm), 0.45 m out, heading 90° off radial.
    q = np.array([-0.63018, 0.22631, 0.14601, -2.21704, 0.87887, 1.63422, -1.68999])
    T = kin.fk(q)
    assert T[:3, 3] == pytest.approx([0.45, 0.0, 0.20], abs=1e-4)
    assert T[2, 2] == pytest.approx(-0.434, abs=0.01)  # tool axis 26° below level
    return {"schema": pa.CENTRE_SCHEMA, "q": q.tolist(), "centre_m": T[:3, 3].tolist()}


def _socket_offset(delta_mm) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(delta_mm) / 1e3
    return T


def _seated_run(kin, centre, path, delta_mm):
    """The true socket is ``delta_mm`` off corenetic_socket_ee; the ball holds it, the stiff rotation follows."""
    D = _socket_offset(delta_mm)
    q0 = np.asarray(centre["q"])
    ball = (kin.fk(q0) @ D)[:3, 3]  # seated where the centre was taken
    qs, q = [], q0
    for R in path["R"]:
        T_true = np.eye(4)
        T_true[:3, :3], T_true[:3, 3] = R, ball
        q, ep, _ = kin.ik(T_true @ np.linalg.inv(D), q, q0)
        assert ep < 1e-6
        qs.append(q)
    return np.asarray(qs)


def _run_file(tmp_path, name, qs, bias, gripper_kin):
    T = np.asarray([gripper_kin.fk(q) for q in qs])
    run = {
        "robot_ip": "x",
        "note": name,
        "created": "2026-10-13T20:00:00",
        "mode": "commanded",
        "plan": f"{name}_plan.json",
        "centre": "c.json",
        "bias_mm": bias,
        "press_mm": 3.0,
        "completed": True,
        "t": list(range(len(qs))),
        "q": qs.tolist(),
        "O_T_EE": [list(m.T.reshape(-1)) for m in T],
        "F_ext": [[0.0, 0.0, -1.8, 0, 0, 0]] * len(qs),
        "F_ext_frame": "O",
        "phase": [1] * len(qs),
    }
    path = tmp_path / f"{name}_pivot.json"
    path.write_text(json.dumps(run))
    out = tmp_path / f"{name}.fit.json"
    assert tp.main(["fit", str(path), "--json", str(out)]) == 0
    return json.loads(out.read_text())


def test_the_planned_star_passes_its_checks_and_keeps_the_wrist_slow(kin, centre):
    path, qs, summary = pa.plan(centre, pa.PlanParams(), kin, None)
    assert summary["passed"], summary["problems"]
    assert 80 < summary["duration_s"] < 100
    assert max(summary["max_joint_speed_deg_s"]) < 8
    assert summary["table_clearance_mm"] is None  # no scene here; see the MuJoCo test
    # starts and ends where the socket was seated, nothing pressed
    assert np.allclose(qs[0], centre["q"], atol=1e-6) and np.allclose(qs[-1], centre["q"], atol=1e-4)
    assert np.allclose(path["p"][0], centre["centre_m"]) and np.allclose(path["p"][-1], centre["centre_m"])
    # mid-pivot the command is 3 mm down into the ball (the cup faces down, across the tool axis)
    mid = len(path["t"]) // 2
    assert path["p"][mid] - np.asarray(centre["centre_m"]) == pytest.approx([0, 0, -0.003], abs=1e-7)


def test_seated_runs_find_the_true_socket_whatever_the_bias_and_a_followed_command_does_not(kin, centre, tmp_path):
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    delta = [0.4, -0.7, 1.2]  # corenetic_socket_ee is this far off the real socket centre
    fits = {}
    for name, bias in (("plus", [3.0, 0, 0]), ("minus", [-3.0, 0, 0])):
        params = pa.PlanParams(bias_mm=bias, **FAST)
        path, qs_cmd, summary = pa.plan(centre, params, kin, None)
        assert summary["ik_worst_mm"] < 1e-3  # FAST is over the speed limits; only the geometry matters here
        fits[f"seated_{name}"] = _run_file(tmp_path, f"seated_{name}", _seated_run(kin, centre, path, delta), bias, gripper)
        fits[f"followed_{name}"] = _run_file(tmp_path, f"followed_{name}", qs_cmd, bias, gripper)

    for name in ("plus", "minus"):
        sock = fits[f"seated_{name}"]["socket_frame"]["s_ee_mm"]
        assert sock == pytest.approx(delta, abs=0.05)
        assert fits[f"seated_{name}"]["commanded"]["contact"]["in_contact_fraction"] == 1.0
    dc_seated = np.subtract(fits["seated_plus"]["fk"]["c_base_mm"], fits["seated_minus"]["fk"]["c_base_mm"])
    dc_followed = np.subtract(fits["followed_plus"]["fk"]["c_base_mm"], fits["followed_minus"]["fk"]["c_base_mm"])
    assert np.linalg.norm(dc_seated) < 0.05
    assert dc_followed == pytest.approx([6.0, 0.0, 0.0], abs=0.05)
    # a followed command measures the very frame it was planned on (the press is a fixed point below it)
    assert fits["followed_plus"]["socket_frame"]["s_ee_mm"] == pytest.approx([0, 0, 0], abs=0.05)


def test_the_planner_rejects_a_start_past_the_wall_and_names_it(kin, centre):
    q = np.asarray(centre["q"]).copy()
    q[1] = np.radians(-100.5)  # 0.5° inside panda_py's j2 wall
    _, _, summary = pa.plan({**centre, "q": q.tolist()}, pa.PlanParams(**FAST), kin, None)
    assert not summary["passed"]
    assert any("j2 离 panda_py 的墙" in p for p in summary["problems"])


def test_mujoco_scene_reports_table_clearance_and_new_contacts(kin, centre):
    pytest.importorskip("mujoco")
    scene = pa.SceneCheck()
    _, _, summary = pa.plan(centre, pa.PlanParams(**FAST), kin, scene)
    assert summary["mujoco_checked"] and summary["table_clearance_mm"] > pa.MIN_TABLE_CLEARANCE_MM
    assert summary["table_z_mm"] == 120.0 and summary["centre_above_table_mm"] == pytest.approx(80.0, abs=0.1)
    high = pa.SceneCheck(table_z_m=0.17)  # the ball only 30 mm above the table: a tilted fingertip hits it
    _, _, summary = pa.plan(centre, pa.PlanParams(**FAST), kin, high, table_z_m=0.17)
    assert not summary["passed"] and any("最低离桌面" in p and "z = 170 mm" in p for p in summary["problems"])


def test_a_centre_below_the_table_is_a_mis_capture(kin, centre):
    _, _, summary = pa.plan(centre, pa.PlanParams(**FAST), kin, None, table_z_m=0.25)
    assert any("球心只比桌面高 -50 mm" in p for p in summary["problems"])


def test_monitor_and_start_checks():
    m = pa.Monitor()
    q = HOME
    assert m.check(q, [0, 0, 2, 0, 0, 0], [0, 0, 0], [0, 0, 0.003]) == ""
    assert "外力" in m.check(q, [0, 12, 12, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert "偏离指令" in m.check(q, [0, 0, 0, 0, 0, 0], [0, 0, 0], [0.02, 0, 0])
    near = HOME.copy()
    near[5] = np.radians(214.5)
    assert "j6 离 panda_py 的墙" in m.check(near, [0] * 6, [0, 0, 0], [0, 0, 0])
    elbow = HOME.copy()
    elbow[3] += np.radians(6)
    assert m.check(elbow, [0] * 6, [0, 0, 0], [0, 0, 0]) == ""  # no plan given: not checked
    assert "j4 偏离规划 6.0°" in m.check(elbow, [0] * 6, [0, 0, 0], [0, 0, 0], HOME)
    plan = {"q": [HOME.tolist()], "p": [[0.5, 0, 0.1]]}
    assert pa.start_problems(HOME, [0.5, 0, 0.1], plan) == []
    moved = HOME.copy()
    moved[3] += np.radians(3)
    assert "j4 离规划起点差 3.0°" in pa.start_problems(moved, [0.5, 0, 0.1], plan)[0]


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += max(s, 0.001)


def _fake_panda_module(kin_gripper, q_start, *, force_after=None):
    """A Panda that tracks the command perfectly (q = the nullspace target) and reports a 1.8 N press."""
    calls = {"started": 0, "stopped": 0, "set": 0}

    class State:
        def __init__(self, q, force):
            self.q = list(q)
            self.dq = [0.0] * 7
            self.O_T_EE = list(kin_gripper.fk(q).T.reshape(-1))
            self.O_F_ext_hat_K = [0.0, 0.0, -force, 0.0, 0.0, 0.0]  # what the arm applies: down onto the ball

    class Ctrl:
        def __init__(self, **kw):
            self.kw = kw
            calls["nullspace"] = kw.get("nullspace_stiffness")
            self.q = np.asarray(q_start)

        def set_control(self, position, orientation, q_nullspace):
            calls["set"] += 1
            self.q = np.asarray(q_nullspace)

    class Robot:
        def read_once(self):
            return State(q_start, 0.0)

    class Panda:
        def __init__(self, ip):
            self.ctrl = None

        def get_robot(self):
            return Robot()

        def start_controller(self, ctrl):
            calls["started"] += 1
            self.ctrl = ctrl

        def stop_controller(self):
            calls["stopped"] += 1

        def get_state(self):
            force = 20.0 if force_after is not None and calls["set"] > force_after else 1.8
            return State(self.ctrl.q, force)

        def raise_error(self):
            pass

    module = types.ModuleType("panda_py")
    module.Panda = Panda
    module.controllers = types.SimpleNamespace(CartesianImpedance=Ctrl)
    return module, calls


def _plan_file(tmp_path, kin, centre):
    path, qs, summary = pa.plan(centre, pa.PlanParams(tilt_deg=10, spin_deg=10, **FAST), kin, None)
    out = tmp_path / "p.json"
    params = pa.PlanParams(tilt_deg=10, spin_deg=10, **FAST)
    out.write_text(
        json.dumps(
            {
                "schema": pa.PLAN_SCHEMA,
                "centre": "c.json",
                "params": pa.asdict(params),
                **summary,
                "passed": True,  # FAST is over the real speed limits on purpose
                "t": path["t"].tolist(),
                "q": qs.tolist(),
                "p": path["p"].tolist(),
                "quat_xyzw": [pa.quat_xyzw(R).tolist() for R in path["R"]],
                "phase": path["phase"].tolist(),
                "phase_names": [n for n, _, _ in pa._phases(params)],
            }
        )
    )
    return out


def test_execute_follows_the_plan_and_writes_a_fittable_run(kin, centre, tmp_path, monkeypatch):
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    module, calls = _fake_panda_module(gripper, centre["q"])
    monkeypatch.setitem(__import__("sys").modules, "panda_py", module)
    clock = _FakeClock()
    monkeypatch.setattr(pa.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(pa.time, "sleep", clock.sleep)
    handlers = {sig: pa.signal.getsignal(sig) for sig in (pa.signal.SIGINT, pa.signal.SIGTERM)}
    plan = _plan_file(tmp_path, kin, centre)
    out = tmp_path / "20261013_200000_pivot.json"
    try:
        assert pa.main(["execute", str(plan), "--out", str(out)]) == 0
    finally:
        for sig, h in handlers.items():
            pa.signal.signal(sig, h)
    run = json.loads(out.read_text())
    assert calls["started"] == 1 and calls["stopped"] == 1 and calls["set"] > 100
    assert calls["nullspace"] == pa.NULLSPACE_STIFFNESS  # holds the elbow where MuJoCo checked it
    assert run["mode"] == "commanded" and run["completed"] and len(run["q"]) == len(run["F_ext"])
    assert tp.contact_summary(run)["press_median_n"] == 1.8


def test_execute_stops_on_force_and_keeps_nothing_to_fit(kin, centre, tmp_path, monkeypatch, capsys):
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    module, calls = _fake_panda_module(gripper, centre["q"], force_after=50)
    monkeypatch.setitem(__import__("sys").modules, "panda_py", module)
    clock = _FakeClock()
    monkeypatch.setattr(pa.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(pa.time, "sleep", clock.sleep)
    handlers = {sig: pa.signal.getsignal(sig) for sig in (pa.signal.SIGINT, pa.signal.SIGTERM)}
    plan = _plan_file(tmp_path, kin, centre)
    out = tmp_path / "20261013_200000_pivot.json"
    try:
        assert pa.main(["execute", str(plan), "--out", str(out)]) == 4
    finally:
        for sig, h in handlers.items():
            pa.signal.signal(sig, h)
    assert not out.exists() and out.with_suffix(".aborted.json").exists()
    assert calls["stopped"] == 1
    assert "外力 20.0 N" in capsys.readouterr().out


def _fake_hand_panda(kin_gripper, qs_hand, *, force=1.8, force_after=None):
    """A Panda whose joints the operator moves along ``qs_hand``; records what the spring was set to."""
    calls = {"started": 0, "stopped": 0, "set": [], "impedance": None, "reads": 0}

    class State:
        def __init__(self, q, f):
            self.q = list(q)
            self.dq = [0.0] * 7
            self.O_T_EE = list(kin_gripper.fk(q).T.reshape(-1))
            self.O_F_ext_hat_K = [0.0, 0.0, -f, 0.0, 0.0, 0.0]

    class Ctrl:
        def __init__(self, impedance, **kw):
            calls["impedance"] = np.asarray(impedance)

        def set_control(self, position, orientation, q_nullspace):
            calls["set"].append(np.asarray(position))

    class Robot:
        def read_once(self):
            return State(qs_hand[0], 0.0)

    class Panda:
        def __init__(self, ip):
            pass

        def get_robot(self):
            return Robot()

        def start_controller(self, ctrl):
            calls["started"] += 1

        def stop_controller(self):
            calls["stopped"] += 1

        def get_state(self):
            i = min(calls["reads"], len(qs_hand) - 1)
            calls["reads"] += 1
            f = 25.0 if force_after is not None and i > force_after else force
            return State(qs_hand[i], f)

        def raise_error(self):
            pass

    module = types.ModuleType("panda_py")
    module.Panda = Panda
    module.controllers = types.SimpleNamespace(CartesianImpedance=Ctrl)
    return module, calls


def _guide_setup(tmp_path, monkeypatch, kin, centre, **fake):
    gripper = pa.Kinematics(frame="corenetic_gripper_ee")
    # The operator turns the gripper through the star while the ball holds the (true) socket.
    path, _, _ = pa.plan(centre, pa.PlanParams(**FAST), kin, None)
    qs_hand = _seated_run(kin, centre, path, [0.4, -0.7, 1.2])
    module, calls = _fake_hand_panda(gripper, qs_hand, **fake)
    monkeypatch.setitem(__import__("sys").modules, "panda_py", module)
    clock = _FakeClock()
    monkeypatch.setattr(pa.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(pa.time, "sleep", clock.sleep)
    centre_file = tmp_path / "20261013_210000_centre.json"
    # captured with the arm seated: the socket FK there (it is off the true socket by the offset)
    centre_file.write_text(json.dumps({**centre, "centre_m": kin.fk(qs_hand[0])[:3, 3].tolist()}))
    return centre_file, calls


def _run_guide(argv):
    handlers = {sig: pa.signal.getsignal(sig) for sig in (pa.signal.SIGINT, pa.signal.SIGTERM, pa.signal.SIGUSR1)}
    try:
        return pa.main(argv)
    finally:
        for sig, h in handlers.items():
            pa.signal.signal(sig, h)


def test_guide_holds_the_socket_pressed_with_rotation_free_and_writes_a_fittable_run(kin, centre, tmp_path, monkeypatch):
    centre_file, calls = _guide_setup(tmp_path, monkeypatch, kin, centre)
    out = tmp_path / "20261013_210100_pivot.json"
    rc = _run_guide(["guide", str(centre_file), "--duration-s", "30", "--bias-mm", "3", "0", "0", "--out", str(out)])
    assert rc == 0
    assert np.allclose(np.diag(calls["impedance"]), [600, 600, 600, 0, 0, 0])  # rotation is the operator's
    run = json.loads(out.read_text())
    assert run["mode"] == "guided" and run["completed"] and set(run["phase"]) == {0, 1, 2}
    # while guiding, the spring centre is the captured centre + bias + 3 mm along the tool axis
    c0 = np.asarray(json.loads(centre_file.read_text())["centre_m"])
    i = run["phase"].index(1) + 5
    assert np.asarray(run["cmd_socket"][i]) == pytest.approx(c0 + [0.003, 0, -0.003], abs=1e-7)
    assert run["tilt_max_deg"] == pytest.approx(20, abs=0.5) and run["spin_range_deg"] == pytest.approx(60, abs=1)
    fitted = tmp_path / "g.fit.json"
    assert tp.main(["fit", str(out), "--json", str(fitted)]) == 0
    report = json.loads(fitted.read_text())
    assert report["commanded"]["kind"] == "guided" and report["commanded"]["contact"]["press_median_n"] == 1.8
    assert report["socket_frame"]["s_ee_mm"] == pytest.approx([0.4, -0.7, 1.2], abs=0.05)


def test_guide_refuses_a_socket_that_left_the_centre_and_stops_on_force(kin, centre, tmp_path, monkeypatch, capsys):
    centre_file, calls = _guide_setup(tmp_path, monkeypatch, kin, centre, force_after=300)
    moved = json.loads(centre_file.read_text())
    moved["centre_m"] = (np.asarray(moved["centre_m"]) + [0.005, 0, 0]).tolist()
    far = tmp_path / "20261013_210200_centre.json"
    far.write_text(json.dumps(moved))
    assert _run_guide(["guide", str(far)]) == 2
    assert "离取到的球心 5.0 mm" in capsys.readouterr().out
    assert calls["started"] == 0

    out = tmp_path / "20261013_210300_pivot.json"
    assert _run_guide(["guide", str(centre_file), "--duration-s", "30", "--out", str(out)]) == 4
    assert calls["stopped"] == 1 and not out.exists() and out.with_suffix(".aborted.json").exists()
    assert "外力 25.0 N" in capsys.readouterr().out
    assert _run_guide(["guide", str(centre_file), "--duration-s", "5"]) == 2


def test_guide_progress_reads_tilt_and_spin():
    R0 = np.eye(3)
    seen = [pa.rot((1, 0, 0), np.radians(25)), pa.rot((0, 0, 1), np.radians(60)), pa.rot((0, 0, 1), np.radians(-40))]
    tilt, spin = pa.guide_progress(R0, seen)
    assert tilt == pytest.approx(25, abs=1e-6) and spin == pytest.approx(100, abs=1e-6)


def test_guide_anchored_on_the_fitted_ball_centre_stops_dragging_the_socket_frame(kin, centre, tmp_path, monkeypatch):
    """The spring held at the true socket (s mm off the frame): its pull no longer turns with the gripper."""
    true_s = np.array([0.4, -0.7, 1.2])
    errors = {}
    for anchored in (False, True):
        centre_file, calls = _guide_setup(tmp_path, monkeypatch, kin, centre)
        out = tmp_path / f"20261013_2104{int(anchored)}0_pivot.json"
        argv = ["guide", str(centre_file), "--duration-s", "30", "--out", str(out)]
        if anchored:
            argv += ["--anchor-mm", *[str(x) for x in true_s], "--anchor-from", "x.fit.json"]
        assert _run_guide(argv) == 0
        run = json.loads(out.read_text())
        ball = (kin.fk(np.asarray(centre["q"])) @ _socket_offset(true_s))[:3, 3]
        guided = [i for i, ph in enumerate(run["phase"]) if ph == 1]
        # the spring's pull is commanded - actual of the frame it acts on; the hand keeps the true socket on the ball
        pulls = []
        for i in guided:
            sock = kin.fk(run["q"][i])
            assert np.allclose(sock[:3, :3] @ true_s / 1e3 + sock[:3, 3], ball, atol=1e-6)
            pulls.append(np.asarray(run["cmd_socket"][i]) - sock[:3, 3])
        pulls = np.asarray(pulls) * 1e3
        errors[anchored] = np.linalg.norm(pulls - [0, 0, -3], axis=1)
        assert run["anchor_mm"] == (true_s.tolist() if anchored else [0.0, 0.0, 0.0])
    assert errors[False].max() > 0.3  # the frame origin swings about the ball centre as the hand turns
    assert errors[True].max() < 1e-3  # held at the ball centre: only the press is left
