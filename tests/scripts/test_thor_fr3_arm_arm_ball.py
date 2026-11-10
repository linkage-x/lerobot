"""fr3_arm_arm_ball: the two FR3 bases joined by balls both arms pivoted about (roadmap B3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.thor.gmsl2 import fr3_arm_arm_ball as aa, register_fr3_base as reg, world_provenance as wp

LEFT_IP, RIGHT_IP = "192.168.11.102", "192.168.1.210"
# the right base 0.66 m to the left base's -y, turned 3 deg about z (fr3_arms.json placement, roughly)
_c, _s = math.cos(math.radians(3)), math.sin(math.radians(3))
T_LEFT_RIGHT = np.array([[_c, -_s, 0, 0.004], [_s, _c, 0, -0.66], [0, 0, 1, 0.002], [0, 0, 0, 1.0]])
# balls in the right base: reachable by both, spread over a plane and in height
BALLS_RIGHT = [
    [0.45, 0.30, 0.05],
    [0.55, 0.45, 0.05],
    [0.40, 0.55, 0.20],
    [0.60, 0.25, 0.25],
    [0.50, 0.40, 0.35],
]


def _fit(centre_m, ip: str, *, joint: bool = False) -> dict:
    fk = {
        "c_base_mm": (np.asarray(centre_m) * 1e3).tolist(),
        "c_sigma_mm": [0.05] * 3,
        "s_ee_mm": [0, 0, -10],
        "residual_rms_mm": 0.1,
        "residual_p95_mm": 0.2,
        "n_used": 300,
        "n_total": 4000,
        "span_deg": 40.0,
        "weak_singular": 0.3,
    }
    if joint:
        return {
            "schema": "fr3_pivot_joint/v1",
            "runs": ["x_pivot.json"],
            "shared_centre": True,
            "fk": {**fk, "c_base_mm": [fk["c_base_mm"]], "c_sigma_mm": [fk["c_sigma_mm"]]},
        }
    return {"robot_ip": ip, "fk": fk}


def _pairs(*, nudge_mm=None, wrong_left_ip: str = "") -> list:
    pairs = []
    for i, c_right in enumerate(BALLS_RIGHT):
        c_left = (T_LEFT_RIGHT @ np.r_[c_right, 1.0])[:3]
        if nudge_mm and i == nudge_mm[0]:
            c_left = c_left + np.asarray(nudge_mm[1]) / 1e3  # the ball moved between the two arms' pivots
        left_ip = wrong_left_ip or LEFT_IP
        left = aa.side_row(_fit(c_left, left_ip), f"l{i}.fit.json", left_ip, LEFT_IP, "左臂")
        right = aa.side_row(_fit(c_right, RIGHT_IP), f"r{i}.fit.json", RIGHT_IP, RIGHT_IP, "右臂")
        pairs.append((left, right))
    return pairs


def test_the_balls_recover_the_base_to_base_edge_and_it_goes_into_the_graph():
    report = aa.solve(_pairs())

    assert report["verdict"] == "ok", report["reasons"]
    edge = report["edge"]
    assert (edge["from_world_frame_id"], edge["to_world_frame_id"]) == ("fr3_base", "fr3_base_right")
    # T_to_from takes fr3_base coordinates into fr3_base_right
    assert np.allclose(edge["T_to_from"], np.linalg.inv(T_LEFT_RIGHT), atol=1e-9)
    q = edge["quality"]
    assert q["positions"] == 5 and q["leave_one_out_max_mm"] == pytest.approx(0.0, abs=1e-6)
    assert q["baseline_m"] == pytest.approx(0.66, abs=0.01)
    assert edge["sigma_mm"] == aa.MIN_SIGMA_MM  # a perfect leave-one-out claims the floor, not zero
    assert wp.edge_method_kind(edge) == "arm_arm_ball"
    assert edge["source"]["robot_ips"] == {"fr3_base": LEFT_IP, "fr3_base_right": RIGHT_IP}
    assert report["workspace_points_base_m"][0] == BALLS_RIGHT[0]

    # apply: fr3_base_right becomes a node, reached from the world through fr3_base; no reference check
    reg.check_reference(report, Path("/nonexistent"))  # base to base: passes without a world_reference.json
    world_edge = {
        "from_world_frame_id": "world_a",
        "to_world_frame_id": "fr3_base",
        "method": "test",
        "T_to_from": np.eye(4).tolist(),
    }
    graph = {
        "nodes": [{"world_frame_id": "world_a"}, {"world_frame_id": "fr3_base", "reason": "robot_base"}],
        "edges": [world_edge],
    }
    updated = reg.apply_report(report, graph)
    assert "fr3_base_right" in {n["world_frame_id"] for n in updated["nodes"]}
    transform, hops = wp.world_transform(updated, "world_a", "fr3_base_right")
    assert [h["to_world_frame_id"] for h in hops] == ["fr3_base", "fr3_base_right"]
    assert np.allclose(transform, np.linalg.inv(T_LEFT_RIGHT), atol=1e-9)


def test_a_ball_that_moved_between_the_arms_fails_leave_one_out():
    report = aa.solve(_pairs(nudge_mm=(2, [6.0, 0.0, 0.0])))
    assert report["verdict"] == "rejected"
    assert any("留一预测最差" in r for r in report["reasons"])
    assert report["edge"]["quality"]["leave_one_out_max_mm"] > aa.GATE_LOO_MAX_MM


def test_a_pivot_from_the_wrong_arm_or_too_few_balls_is_refused():
    report = aa.solve(_pairs(wrong_left_ip=RIGHT_IP))
    assert report["verdict"] == "rejected" and "edge" not in report
    assert any("是在 192.168.1.210 上采的" in r for r in report["reasons"])

    report = aa.solve(_pairs()[:3])
    assert report["verdict"] == "rejected" and any("≥ 4" in r for r in report["reasons"])

    line = [
        (
            aa.side_row(_fit([0.4 + 0.1 * i, 0, 0.1], LEFT_IP), "l", LEFT_IP, LEFT_IP, "左臂"),
            aa.side_row(_fit([0.4 + 0.1 * i, 0, 0.1], RIGHT_IP), "r", RIGHT_IP, RIGHT_IP, "右臂"),
        )
        for i in range(4)
    ]
    report = aa.solve(line)
    assert report["verdict"] == "rejected" and any("共线" in r for r in report["reasons"])


def test_cli_reads_fits_and_joint_fits_and_takes_the_ip_from_the_runs(tmp_path):
    pivots = tmp_path / "fr3_tcp_pivot"
    (pivots / "joint").mkdir(parents=True)
    args = []
    for i, c_right in enumerate(BALLS_RIGHT):
        c_left = (T_LEFT_RIGHT @ np.r_[c_right, 1.0])[:3]
        left = pivots / f"2026110{i}_100000_pivot.fit.json"
        left.write_text(json.dumps(_fit(c_left, LEFT_IP)))
        if i == 0:
            # a joint fit records no IP; its run's fit does
            (pivots / "x_pivot.fit.json").write_text(json.dumps({"robot_ip": RIGHT_IP}))
            right = pivots / "joint" / "20261103_100000_joint.json"
            right.write_text(json.dumps(_fit(c_right, "", joint=True)))
        else:
            right = pivots / f"2026110{i}_110000_pivot.fit.json"
            right.write_text(json.dumps(_fit(c_right, RIGHT_IP)))
        args += ["--position", str(left), str(right)]
    out = tmp_path / "armarm.json"

    assert aa.main([*args, "--left-ip", LEFT_IP, "--right-ip", RIGHT_IP, "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["verdict"] == "ok" and report["positions"][0]["right"]["robot_ip"] == RIGHT_IP
    assert aa.main([*args, "--left-ip", RIGHT_IP, "--right-ip", RIGHT_IP, "--out", str(out)]) == 1
