"""dual_fr3_move_to_start: both FR3s home, their paths checked against each other first.

The plan on the capsule model with the registry's 0.66 m placement, at poses found by sampling (each named
for the branch it takes); then the whole run -- two arm processes, the check, the answer, the move -- on the
stand-in panda_py (``fake_panda_py``), whose move is instant.
"""

# ruff: noqa: N806  (T_ for poses, as in the tool)
import json

import numpy as np
import pytest

from tools.fr3 import dual_fr3_move_to_start as dh, dual_fr3_scene as scene, fr3_arms, fr3_kinematics as kin

STUB = "tests.scripts.fake_panda_py"
HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
GOALS = {"right": HOME, "left": HOME}
T_RIGHT_LEFT = np.array(fr3_arms.placement({"placement": {
    "left_in_right_m": [0.0, 0.66, 0.0], "left_yaw_deg": 0.0, "sigma_mm": 20.0, "source": "test"}})["T_right_left"])  # fmt: skip
GATE_M = 0.06


def _plan(q_right, q_left, **kw):
    return dh.plan_home({"right": q_right, "left": q_left}, GOALS, T_RIGHT_LEFT, d_gate_m=GATE_M, **kw)


def test_the_path_is_the_joint_line_in_small_steps() -> None:
    q0 = HOME.copy()
    q0[0] = 1.0
    path = dh.joint_path(q0, HOME)
    assert np.allclose(path[0], q0) and np.allclose(path[-1], HOME)
    assert np.abs(np.diff(path, axis=0)).max() <= dh.STEP_RAD + 1e-12
    assert len(dh.joint_path(HOME, HOME)) == 2


def test_arms_at_their_start_stay_put_and_one_arm_away_goes_alone() -> None:
    assert _plan(HOME, HOME)["stages"] == []
    q = HOME.copy()
    q[0] = 0.6
    plan = _plan(q, HOME)
    assert plan["chosen"] == "together" and plan["stages"] == [["right"]]
    assert plan["arms"]["right"]["max_joint_move_deg"] == pytest.approx(34.4, abs=0.1)
    assert not plan["arms"]["left"]["moving"]


def test_two_arms_that_would_meet_on_the_way_go_one_after_the_other() -> None:
    q_r = np.array([1.093, 0.392, 0.0, -1.545, 0.0, 1.287, 0.785])
    q_l = np.array([-0.832, -0.164, 0.0, -1.278, 0.0, 2.44, 0.785])
    plan = _plan(q_r, q_l)
    # together they pass through each other somewhere on the grid of timings; the left first, too
    assert plan["options"]["together"]["min_gap_mm"] < 0.0
    assert plan["options"]["left_then_right"]["min_gap_mm"] < 0.0
    assert plan["options"]["right_then_left"]["min_gap_mm"] >= plan["gate_mm"] == 60.0
    assert plan["chosen"] == "right_then_left" and plan["stages"] == [["right"], ["left"]]


def test_arms_apart_now_but_crossing_on_every_path_are_refused() -> None:
    # 32 mm apart where they stand, and each arm's straight way home goes through the other
    q_r = np.array([1.06, 0.155, 0.0, -1.728, 0.0, 1.22, 0.785])
    q_l = np.array([-0.817, 0.063, 0.0, -1.18, 0.0, 1.413, 0.785])
    plan = _plan(q_r, q_l)
    assert plan["start_gap_mm"] == pytest.approx(32.3, abs=0.5)
    # already inside the 60 mm gate: held to "never closer than now" instead
    assert plan["gate_mm"] == pytest.approx(plan["start_gap_mm"] - dh.NEVER_CLOSER_MARGIN_M * 1e3, abs=0.1)
    assert plan["stages"] is None and "guide them apart in Desk" in plan["reasons"][-1]


def test_a_path_through_the_table_and_a_joint_past_the_wall_are_refused() -> None:
    # a table raised to 0.45 m: the folded arm is in it already, and its way home goes deeper -- refused,
    # where a way out (never closer than now) would pass
    q = HOME.copy()
    q[1], q[3] = -1.0, -2.95
    low = _plan(q, HOME, obstacles=[scene.table_obstacle(top_m=0.45, edge_m=0.2)])
    assert low["stages"] is None and any("from the table on the way" in r for r in low["reasons"])
    assert low["arms"]["right"]["min_env_gap_mm"] < -180.0
    past = HOME.copy()
    past[5] = kin._WALL_UPPER[5] + 0.01
    assert any("past panda_py's walls" in r for r in _plan(past, HOME)["reasons"])


# ------------------------------------------------------------------ the run, on the stand-in ---


def _registry(right_ip: str = "10.0.0.1", left_ip: str = "10.0.0.2", placement: bool = True) -> dict:
    arm = {"in_service": True, "payload": None, "start_joints_rad": kin.Q_READY.tolist()}
    reg = {"arms": {"right": {**arm, "robot_ip": right_ip}, "left": {**arm, "robot_ip": left_ip}}}
    if placement:
        reg["placement"] = {"left_in_right_m": [0.0, 0.66, 0.0], "left_yaw_deg": 0.0, "sigma_mm": 20.0, "source": "t"}
    return reg


def _lines(out: str, prefix: str) -> list[dict]:
    return [json.loads(line.removeprefix(prefix)) for line in out.splitlines() if line.startswith(prefix)]


def test_the_run_checks_then_moves_the_arm_that_is_away(capsys) -> None:
    code = dh.home(_registry(right_ip="10.0.0.1#q=1:0.5"), go=lambda: "go", panda_py_module=STUB)
    out = capsys.readouterr().out
    assert code == 0, out
    (plan,) = _lines(out, dh.PLAN_LINE)
    assert plan["stages"] == [["right"]] and plan["ip"] == {"right": "10.0.0.1#q=1:0.5", "left": "10.0.0.2"}
    assert "no payload declared" in plan["payload"]["right"]
    (result,) = _lines(out, dh.RESULT_LINE)
    assert result["ok"] and result["stages"][0]["right"]["max_error_deg"] < 1e-6


def test_the_check_and_an_abort_move_nothing_and_no_placement_is_refused(capsys) -> None:
    assert dh.home(_registry(right_ip="10.0.0.1#q=1:0.5"), check=True, panda_py_module=STUB) == 0
    assert dh.home(_registry(right_ip="10.0.0.1#q=1:0.5"), go=lambda: "abort", panda_py_module=STUB) == 3
    out = capsys.readouterr().out
    assert len(_lines(out, dh.PLAN_LINE)) == 2 and not _lines(out, dh.RESULT_LINE)
    assert "not moved (abort)" in out
    assert dh.home(_registry(placement=False), go=lambda: "go", panda_py_module=STUB) == 2
    assert "declares no placement" in capsys.readouterr().out


def test_the_registry_placement_is_validated() -> None:
    bad = {"left_in_right_m": [0.0, 0.66], "left_yaw_deg": 0.0, "sigma_mm": -1.0, "source": ""}
    problems = fr3_arms._placement_problems(bad)
    assert len(problems) == 3
    T = np.array(fr3_arms.placement({"placement": {**bad, "left_in_right_m": [0.1, 0.66, 0.0], "left_yaw_deg": 90.0}})["T_right_left"])
    assert np.allclose(T[:3, 3], [0.1, 0.66, 0.0]) and np.allclose(T[:3, 0], [0.0, 1.0, 0.0])
