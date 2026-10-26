"""dual_fr3_replay_runtime's real-arm driver (roadmap D1) on a stand-in panda_py (``fake_panda_py``).

What the stand-in cannot show is what B1 is for: libfranka's real-time loop on Thor, two arms' traffic on one
link, the arm's own reflexes. What it does show: the calls the runtime makes, in its order, on a control loop
that runs beside them, the D2 stamps they carry, and every refusal before an arm moves.
"""

# ruff: noqa: N806  (T_right_left as in the runtime)
import copy

import numpy as np
import pytest

from tests.scripts.test_dual_fr3_replay_runtime import FPS, MIRROR, TOOL, _path
from tools.fr3 import dual_fr3_replay_runtime as rt, fr3_kinematics as kin

STUB = "tests.scripts.fake_panda_py"


def _driver(host: str = "10.0.0.1", payload=None, period_s: float = 1.0 / FPS) -> rt.RealFr3Driver:
    return rt.RealFr3Driver(
        robot_ip=host,
        tool=TOOL.tolist(),
        period_s=period_s,
        arm="right",
        payload=payload,
        panda_py_module=STUB,
    )


def _real_plans(hosts=("10.0.0.1", "10.0.0.2"), amplitude: float = 0.3):
    q_r = _path(amplitude)
    T_right_left = np.eye(4)
    T_right_left[1, 3] = 0.8
    plans = []
    for name, q, host in (("right", q_r, hosts[0]), ("left", q_r * MIRROR, hosts[1])):
        spec = {
            "kind": "real",
            "arm": name,
            "robot_ip": host,
            "payload": None,
            "tool": TOOL.tolist(),
            "start_joints": q[0].tolist(),
            "panda_py_module": STUB,
        }
        plans.append(rt.ArmPlan(name, kin.NOMINAL.fk(q) @ TOOL, spec, joints=q))
    return plans, T_right_left


def test_the_driver_moves_to_the_plan_s_start_then_follows_its_joints() -> None:
    q = _path(0.3)
    driver = _driver()
    start = driver.move_to_start(kin.NOMINAL.fk(q[:1])[0] @ TOOL, q[0].tolist())
    assert start["reached"] and start["move_to_joint_position"]
    assert 0.0 < start["filter_coeff"] < 0.2  # a third of a 20 ms period, per millisecond
    assert start["desk_m_ee_kg"] == 0.0
    stamps = []
    for k in range(len(q)):
        driver.command(None, k, q[k])
        rt._sleep_until(rt.now_ns() + int(1e9 / FPS))
        _, q_now = driver.observe()
        stamps.append(driver.stamp())
    assert np.abs(q_now - q[-1]).max() < 0.02  # caught up within a frame or so
    robot_s = np.array([s[0] for s in stamps])
    assert np.all(np.diff(robot_s) > 0)  # the loop's clock, not a frozen cache
    assert np.diff(robot_s).mean() == pytest.approx(1.0 / FPS, abs=0.004)
    ctrl = driver._ctrl
    driver.hold()
    assert driver._ctrl is None and not driver._panda._run.is_set()  # controller stopped
    assert np.allclose(ctrl.velocity, 0.0)  # held where it was, not still moving


def test_the_velocity_feed_forward_is_the_step_from_the_last_frame() -> None:
    q = _path(0.3)
    driver = _driver()
    driver.move_to_start(kin.NOMINAL.fk(q[:1])[0] @ TOOL, q[0].tolist())
    driver.command(None, 0, q[0])
    assert np.allclose(driver._ctrl.velocity, 0.0)
    driver.command(None, 1, q[1])
    assert np.allclose(driver._ctrl.velocity, (q[1] - q[0]) * FPS)
    driver.hold()


def test_what_a_real_arm_refuses_before_it_moves() -> None:
    q0 = kin.Q_READY.tolist()
    T0 = kin.NOMINAL.fk(kin.Q_READY[None])[0] @ TOOL
    with pytest.raises(rt.ArmFaultError, match="has none"):
        _driver().move_to_start(T0, None)
    with pytest.raises(rt.ArmFaultError, match="j1 are past panda_py's walls now"):
        _driver("10.0.0.1#q=1:2.95").move_to_start(T0, q0)
    with pytest.raises(rt.ArmFaultError, match="j4 are past the FR3 joint limits now"):
        _driver("10.0.0.1#q=4:-3.0504").move_to_start(T0, q0)  # the new arm's transport pose, 10-26
    declared = {"mass_kg": 0.0, "com_m": [0.0, 0.0, 0.0], "source": "bare flange"}
    with pytest.raises(rt.ArmFaultError, match="does NOT match"):
        _driver("10.0.0.1#m_ee=1.111", payload=declared).move_to_start(T0, q0)
    matching = _driver(payload=declared)
    assert matching.move_to_start(T0, q0)["reached"]
    matching.hold()  # a controller left running outlives the test and ticks whatever clock the next one fakes
    driver = _driver()
    with pytest.raises(rt.ArmFaultError, match="before move_to_start"):
        driver.command(None, 0, kin.Q_READY)
    driver.move_to_start(T0, q0)
    with pytest.raises(rt.ArmFaultError, match="not TCP poses"):
        driver.command(T0, 0, None)
    beyond = kin.Q_READY.copy()
    beyond[5] = kin.Q_UPPER[5] + 0.01
    with pytest.raises(rt.ArmFaultError, match="j6 past the walls"):
        driver.command(None, 0, beyond)
    driver.hold()


def test_a_dead_control_loop_is_seen_with_or_without_libfranka_s_error() -> None:
    T0 = kin.NOMINAL.fk(kin.Q_READY[None])[0] @ TOOL
    for host, error in (("10.0.0.1#reflex@0.1", "cartesian_reflex"), ("10.0.0.1#stall@0.1", "has not moved")):
        driver = _driver(host)
        driver.move_to_start(T0, kin.Q_READY.tolist())
        with pytest.raises((RuntimeError, rt.ArmFaultError), match=error):
            for _ in range(100):
                driver.observe()
                rt._sleep_until(rt.now_ns() + 10_000_000)
        driver.hold()


def test_two_real_arms_run_together_and_d2_reads_their_clocks(tmp_path) -> None:
    plans, T_right_left = _real_plans()
    report = rt.run(
        plans, fps=FPS, T_right_left=T_right_left, run_dir=tmp_path / "run", go=lambda: True, start_lead_s=0.1
    )
    assert report["stop"] is None, report["stop"]
    for row in report["arms"].values():
        assert row["outcome"] == "completed" and row["frames_commanded"] == len(plans[0].targets)
        assert row["start"]["reached"] and row["start"]["robot_ip"].startswith("10.0.0.")
    assert report["clock"]["status"] == "passed", report["clock"]
    # each stand-in arm keeps its own robot clock; D2 fits them apart
    arms = report["clock"]["arms"]
    assert arms["right"]["clock"]["n"] == arms["left"]["clock"]["n"] == len(plans[0].targets)


def test_a_reflex_on_one_real_arm_stops_the_other_within_one_period(tmp_path) -> None:
    plans, T_right_left = _real_plans(hosts=("10.0.0.1", "10.0.0.2#reflex@0.4"))
    report = rt.run(
        plans, fps=FPS, T_right_left=T_right_left, run_dir=tmp_path / "run", go=lambda: True, start_lead_s=0.1
    )
    left, right = report["arms"]["left"], report["arms"]["right"]
    assert left["outcome"] == "fault" and "cartesian_reflex" in left["reason"]
    assert report["stop"]["by"] == 2
    assert right["outcome"] == "stopped" and right["commands_after_stop"] <= 1
    # the faulted arm sent one command it never read back; D2 still pairs the frames both measured
    assert report["clock"]["status"] in ("passed", "failed"), report["clock"]


def test_real_plans_must_be_joint_paths_within_the_retarget_s_speed() -> None:
    plans, _ = _real_plans()
    assert rt.real_plan_problems(plans, FPS) == []
    assert "j1 moves" in rt.real_plan_problems(plans, FPS * 20)[0]  # the same path, 20x faster
    plans[1].joints = None
    assert "left: no joint path" in rt.real_plan_problems(plans, FPS)[0]


def test_the_real_backend_needs_both_arms_and_panda_py() -> None:
    from tools.fr3 import fr3_arms

    both = fr3_arms.load()  # both in service since 10-26 (B1)
    one = copy.deepcopy(both)
    one["arms"]["right"].update(in_service=False, why_not="B1: not cabled")
    assert rt.real_backend_refusal(one, panda_py_module=STUB).startswith("the right arm is not in service")
    assert rt.real_backend_refusal(both, panda_py_module=STUB) == ""
    assert "does not import" in rt.real_backend_refusal(both, panda_py_module="no_such_panda_py")
    same = copy.deepcopy(both)
    same["arms"]["left"]["robot_ip"] = same["arms"]["right"]["robot_ip"]
    assert "both arms are" in rt.real_backend_refusal(same, panda_py_module=STUB)
    entries = rt.real_arm_entries(both)
    assert entries["left"]["robot_ip"] == "192.168.11.102" and entries["right"]["robot_ip"] == "192.168.1.210"


def test_the_cli_takes_real_arms_only_on_a_retarget(capsys, tmp_path) -> None:
    assert rt.main([str(tmp_path), "--backend", "real"]) == 2
    assert "give --retarget" in capsys.readouterr().out
    assert (
        rt.main([str(tmp_path), "--backend", "real", "--retarget", "absolute", "--fault", "left:raise@3"])
        == 2
    )
    assert "--fault is for the fake arms" in capsys.readouterr().out
