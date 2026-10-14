"""fr3_base_ball_check: a ball both the FR3 and the handheld pivoted about referees a world -> fr3_base edge."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.thor.gmsl2 import fr3_base_ball_check as bc


def _rot(rotvec) -> np.ndarray:
    w = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(w)
    K = np.zeros((3, 3)) if theta == 0 else np.cross(np.eye(3), w / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _rigid(rotvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rot(rotvec), t
    return T


def _quat_xyzw(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    return [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]


def _sweep(n: int = 240, deg: float = 30.0) -> list[np.ndarray]:
    """Tilt about x, about y, and spin about z: the motion the panel asks for."""
    a = np.radians(deg) * np.sin(np.linspace(0, 6 * np.pi, n))
    b = np.radians(deg) * np.cos(np.linspace(0, 4 * np.pi, n))
    c = np.radians(60) * np.sin(np.linspace(0, 2 * np.pi, n))
    return [_rot([a[i], 0, 0]) @ _rot([0, b[i], 0]) @ _rot([0, 0, c[i]]) for i in range(n)]


T_WORLD_BASE = _rigid([0.02, -0.01, 1.5], [1.2, 0.0, 0.74])
LIVE_ERROR = _rigid([0.0, 0.0, np.radians(1.0)], [0.003, -0.002, 0.004])  # the live edge is off by this


def _fr3_fit(ball_world: np.ndarray) -> dict:
    c_base = np.linalg.inv(T_WORLD_BASE) @ np.r_[ball_world, 1.0]
    return {
        "fk": {
            "s_ee_mm": [0.0, 0.0, -10.3],
            "s_sigma_mm": [0.02, 0.02, 0.02],
            "c_base_mm": (c_base[:3] * 1e3).tolist(),
            "c_sigma_mm": [0.03, 0.03, 0.03],
            "residual_rms_mm": 0.1,
            "residual_p95_mm": 0.2,
            "n_used": 400,
            "n_total": 4500,
            "span_deg": 35.0,
            "weak_singular": 0.3,
        },
        "socket_frame": {"s_ee_mm": [0.0, 0.0, 0.0]},
    }


def _handheld_episode(
    dataset: Path, episode: int, ball_world: np.ndarray, s_tcp: np.ndarray, *, world="world_a", noise_m=0.0007, slid=None
):
    ep_dir = dataset / "episodes" / f"episode_{episode:06d}"
    ep_dir.mkdir(parents=True)
    (ep_dir / "meta.json").write_text(
        json.dumps({"episode_index": episode, "world_frame": {"status": "ok", "world_frame_id": world}})
    )
    sidecar = dataset / "derived" / bc.SIDECAR_NAME
    sidecar.mkdir(parents=True, exist_ok=True)
    path = sidecar / "state_action.right.csv"
    new = not path.exists()
    rng = np.random.default_rng(episode)
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["episode_index", "frame_index", "solve_ok", "state_x_m", "state_y_m", "state_z_m",
                        "state_qx", "state_qy", "state_qz", "state_qw"])
        for i, R in enumerate(_sweep()):
            t = ball_world - R @ s_tcp + rng.normal(0, noise_m, 3)  # labels scatter ~1 mm
            if slid is not None and slid[0] <= i < slid[1]:
                t = t - R @ np.array([0.006, 0.0, 0.0])  # the insert slid sideways on the ball
            w.writerow([episode, i, "True", *t, *_quat_xyzw(R)])
        w.writerow([episode, 999, "False", "nan", "nan", "nan", "nan", "nan", "nan", "nan"])


def _graph(T_fr3_base_world: np.ndarray) -> dict:
    return {
        "nodes": [{"world_frame_id": "world_a"}, {"world_frame_id": "fr3_base"}],
        "edges": [{"from_world_frame_id": "world_a", "to_world_frame_id": "fr3_base",
                   "T_to_from": T_fr3_base_world.tolist(), "method": "fr3_single_tag_hand_eye", "created_utc": "x"}],
    }


BALLS = [np.array([1.0, -0.3, 0.05]), np.array([1.3, 0.0, 0.05]), np.array([1.05, 0.3, 0.12])]


def test_the_ball_finds_what_the_live_edge_is_off_by_and_clears_the_right_edge(tmp_path):
    dataset = tmp_path / "ds"
    s_tcp = np.array([0.001, -0.0005, 0.002])  # production's TCP is 2 mm off the socket: must not matter
    rows = []
    for i, ball in enumerate(BALLS):
        _handheld_episode(dataset, i, ball, s_tcp)
        rows.append(bc.position_row(_fr3_fit(ball), f"p{i}.fit.json", dataset, i))
    assert all(r["usable"] for r in rows), [r["reasons"] for r in rows]
    assert rows[0]["handheld"]["s_mm"] == pytest.approx(s_tcp * 1e3, abs=0.3)

    T_live_world_base = LIVE_ERROR @ T_WORLD_BASE
    right = tmp_path / "laser__right.json"
    right.write_text(json.dumps({"edge": {"from_world_frame_id": "world_a", "to_world_frame_id": "fr3_base",
                                          "T_to_from": np.linalg.inv(T_WORLD_BASE).tolist()}}))
    report = bc.check(rows, _graph(np.linalg.inv(T_live_world_base)), [right])

    live, cand = report["edges"]
    assert live["name"] == "live" and cand["name"] == "laser__right.json"
    for p, ball in zip(live["positions"], BALLS, strict=True):
        expected = (LIVE_ERROR[:3, :3] @ ball + LIVE_ERROR[:3, 3] - ball) * 1e3
        assert p["error_mm"] == pytest.approx(expected, abs=0.3)
    assert live["rms_mm"] > 4.0
    assert cand["max_mm"] < 0.3
    ball_edge = report["ball_edges"]["world_a"]
    assert np.asarray(ball_edge["T_world_base"]) == pytest.approx(T_WORLD_BASE, abs=1e-3)
    json.dumps(report)


def _joint_fit(ball_world: np.ndarray, *, shared: bool = True) -> dict:
    fk = _fr3_fit(ball_world)["fk"]
    centres = [fk["c_base_mm"]] if shared else [fk["c_base_mm"], fk["c_base_mm"]]
    return {
        "schema": bc.FR3_JOINT_SCHEMA,
        "runs": ["a_pivot.json", "b_pivot.json"],
        "shared_centre": shared,
        "fk": {**fk, "c_base_mm": centres, "c_sigma_mm": [[0.2, 0.2, 0.2]] * len(centres)},
    }


def test_a_joint_fit_with_one_shared_centre_is_one_ball_position(tmp_path):
    # 10-14: one run's centre sigma is ~1 mm, nine runs on one ball together ~0.5 mm
    dataset = tmp_path / "ds"
    _handheld_episode(dataset, 0, BALLS[0], np.zeros(3))
    _handheld_episode(dataset, 1, BALLS[1], np.zeros(3))
    row = bc.position_row(_joint_fit(BALLS[0]), "20261014_142837_joint.json", dataset, 0)
    assert row["usable"], row["reasons"]
    assert row["fr3"]["centre_mm"] == pytest.approx(_fr3_fit(BALLS[0])["fk"]["c_base_mm"])
    assert row["fr3"]["centre_sigma_mm"] == [0.2, 0.2, 0.2]
    assert row["fr3_runs"] == ["a_pivot.json", "b_pivot.json"]
    apart = bc.position_row(_joint_fit(BALLS[1], shared=False), "x_joint.json", dataset, 1)
    assert not apart["usable"] and "每次一个球心" in apart["reasons"][0] and len(apart["reasons"]) == 1


def test_a_stretch_with_the_insert_slid_off_the_ball_is_dropped_not_fitted(tmp_path):
    # 10-14: 2 mm labels, and the insert ~6 mm off the ball for a fifth of the pivot
    dataset = tmp_path / "ds"
    _handheld_episode(dataset, 0, BALLS[0], np.zeros(3), noise_m=0.0015, slid=(40, 90))
    row = bc.position_row(_fr3_fit(BALLS[0]), "p.fit.json", dataset, 0)
    assert row["usable"], row["reasons"]
    assert np.linalg.norm(np.asarray(row["handheld"]["centre_mm"]) - BALLS[0] * 1e3) < 1.0
    _, R, t = bc._pick_cube(dataset / "derived" / bc.SIDECAR_NAME, 0, "")
    loose = bc.fit_pivot(R, t, reject_mm=bc.HANDHELD_REJECT_MM, reject_k=3.0)
    assert np.linalg.norm(loose.c_base_m - BALLS[0]) > 1.5e-3  # what 3x the median kept


def test_a_position_without_a_world_stamp_or_labels_is_named_not_scored(tmp_path):
    dataset = tmp_path / "ds"
    _handheld_episode(dataset, 0, BALLS[0], np.zeros(3), world="")
    (dataset / "episodes" / "episode_000001").mkdir(parents=True)
    rows = [
        bc.position_row(_fr3_fit(BALLS[0]), "a.fit.json", dataset, 0),
        bc.position_row(_fr3_fit(BALLS[1]), "b.fit.json", dataset, 1),
        bc.position_row({"fk": {}}, "c.json", dataset, 0),
    ]
    assert [r["usable"] for r in rows] == [False, False, False]
    assert "world 戳" in rows[0]["reasons"][0]
    assert "先生成 EE 轨迹" in rows[1]["reasons"][-1]
    assert "fk.c_base_mm" in rows[2]["reasons"][0]
    assert bc.check(rows, _graph(np.eye(4)), [])["edges"] == []


def test_too_little_rotation_is_a_warning(tmp_path):
    dataset = tmp_path / "ds"
    ep_dir = dataset / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "meta.json").write_text(json.dumps({"world_frame": {"status": "ok", "world_frame_id": "world_a"}}))
    sidecar = dataset / "derived" / bc.SIDECAR_NAME
    sidecar.mkdir(parents=True)
    with (sidecar / "state_action.right.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_index", "frame_index", "state_x_m", "state_y_m", "state_z_m",
                    "state_qx", "state_qy", "state_qz", "state_qw"])
        for i, R in enumerate(_sweep(deg=5.0)):
            w.writerow([0, i, *(BALLS[0] - R @ np.array([0, 0, 0.05])), *_quat_xyzw(R)])
    row = bc.position_row(_fr3_fit(BALLS[0]), "a.fit.json", dataset, 0)
    assert row["usable"]
    assert any("转得不够" in w for w in row["warnings"])


def test_a_handheld_episode_that_is_not_a_pivot_is_refused(tmp_path):
    dataset = tmp_path / "ds"
    _handheld_episode(dataset, 0, BALLS[0], np.zeros(3))
    path = dataset / "derived" / bc.SIDECAR_NAME / "state_action.right.csv"
    lines = path.read_text().splitlines()
    with path.open("a", newline="") as f:  # the ball moved 40 mm, pivoted twice as long there
        w = csv.writer(f)
        for i, R in enumerate(_sweep() + _sweep()):
            w.writerow([0, 1000 + i, "True", *(BALLS[0] + [0.04, 0, 0]), *_quat_xyzw(R)])
    assert len(lines) > 200
    row = bc.position_row(_fr3_fit(BALLS[0]), "a.fit.json", dataset, 0)
    assert not row["usable"]
    assert "球中途动过" in row["reasons"][0]


def test_cli_writes_the_report(tmp_path, capsys):
    dataset = tmp_path / "ds"
    _handheld_episode(dataset, 0, BALLS[0], np.zeros(3))
    fit = tmp_path / "p0.fit.json"
    fit.write_text(json.dumps(_fr3_fit(BALLS[0])))
    world = tmp_path / "world"
    world.mkdir()
    (world / "world_graph.json").write_text(json.dumps(_graph(np.linalg.inv(LIVE_ERROR @ T_WORLD_BASE))))
    out = tmp_path / "check.json"

    assert bc.main(["--position", str(fit), str(dataset), "0", "--world-dir", str(world), "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["schema"] == bc.SCHEMA and report["edges"][0]["name"] == "live"
    assert "edge live: mean" in capsys.readouterr().out


FOUR_BALLS = [*BALLS, np.array([1.1, -0.1, 0.3])]  # the fourth higher: tilts become visible


def _rows(tmp_path, balls, s_tcp=np.array([0.002, -0.003, 0.002])):
    dataset = tmp_path / "ds"
    rows = []
    for i, ball in enumerate(balls):
        _handheld_episode(dataset, i, ball, s_tcp)
        rows.append(bc.position_row(_fr3_fit(ball), f"p{i}.fit.json", dataset, i))
    return rows


def test_one_handheld_lever_arm_is_fitted_over_all_positions(tmp_path):
    s_tcp = np.array([0.002, -0.003, 0.002])
    rows = _rows(tmp_path, FOUR_BALLS, s_tcp)

    report = bc.check(rows, _graph(np.linalg.inv(T_WORLD_BASE)), [])

    shared = report["handheld_shared"]["world_a"]
    assert shared["positions"] == [0, 1, 2, 3]
    assert shared["s_mm"] == pytest.approx(s_tcp * 1e3, abs=0.2)
    for row, ball in zip(report["positions"], FOUR_BALLS, strict=True):
        assert "_poses" not in row  # the poses only live until the shared fit is done
        assert row["handheld_shared"]["centre_mm"] == pytest.approx(ball * 1e3, abs=0.3)
    # the edge is scored on the shared-s centres; the right edge is right everywhere
    assert report["edges"][0]["max_mm"] < 0.5
    json.dumps(report)

    each = bc.check(_rows(tmp_path / "b", FOUR_BALLS), _graph(np.eye(4)), [], handheld_s="per-position")
    assert each["handheld_shared"] == {} and "handheld_shared" not in each["positions"][0]
    assert any("各自的杠杆臂" in w for w in each["ball_edges"]["world_a"]["warnings"])


def test_a_ball_edge_is_judged_by_predicting_each_position_from_the_others(tmp_path):
    rows = _rows(tmp_path, FOUR_BALLS)
    three = bc.ball_edge(rows[:3], "world_a")
    assert three["verdict"] == "rejected" and "≥ 4" in three["reasons"][0]
    assert three["leave_one_out_mm"] == []

    edge = bc.ball_edge(rows, "world_a")
    assert edge["verdict"] == "ok", edge["reasons"]
    assert np.asarray(edge["T_world_base"]) == pytest.approx(T_WORLD_BASE, abs=1e-3)
    assert len(edge["leave_one_out_mm"]) == 4 and edge["leave_one_out_max_mm"] < 1.0
    assert edge["scale_pct"] == pytest.approx(0.0, abs=0.1)

    # one ball the camera world puts 10 mm off: the others cannot predict it, and the verdict says so
    rows[3]["handheld"]["centre_mm"] = (np.asarray(rows[3]["handheld"]["centre_mm"]) + [0.0, 0.0, 10.0]).tolist()
    bad = bc.ball_edge(rows, "world_a")
    assert bad["verdict"] == "rejected" and "留一预测最差" in bad["reasons"][0]


def test_edge_writes_the_ball_edge_as_a_candidate_apply_accepts(tmp_path, capsys):
    from tools.thor.gmsl2 import register_fr3_base

    world = tmp_path / "world"
    world.mkdir()
    reference = world / "world_reference.json"
    reference.write_text(json.dumps({"world_frame_id": "world_a"}))
    graph = _graph(np.linalg.inv(LIVE_ERROR @ T_WORLD_BASE))
    (world / "world_graph.json").write_text(json.dumps(graph))
    dataset = tmp_path / "ds"
    args = []
    for i, ball in enumerate(FOUR_BALLS):
        _handheld_episode(dataset, i, ball, np.zeros(3))
        fit = tmp_path / f"p{i}.fit.json"
        fit.write_text(json.dumps(_fr3_fit(ball)))
        args += ["--position", str(fit), str(dataset), str(i)]
    check = tmp_path / "check_x.json"
    assert bc.main([*args, "--world-dir", str(world), "--out", str(check)]) == 0
    assert json.loads(check.read_text())["reference_world"] == "world_a"
    out = tmp_path / "ball_check_x.json"

    assert bc.main(["edge", "--check", str(check), "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["schema"] == register_fr3_base.REPORT_SCHEMA and report["verdict"] == "ok"
    edge = report["edge"]
    assert edge["method"].startswith(bc.BALL_EDGE_METHOD)
    assert np.asarray(edge["T_to_from"]) == pytest.approx(np.linalg.inv(T_WORLD_BASE), abs=1e-3)
    assert len(edge["source"]["positions"]) == 4 and len(report["workspace_points_base_m"]) == 4
    register_fr3_base.check_reference(report, world)
    updated = register_fr3_base.apply_report(report, graph, replace=True)
    assert updated["edges"][-1]["supersedes"][0]["method"] == "fr3_single_tag_hand_eye"
    assert "verdict: ok" in capsys.readouterr().out

    # scored by a later check, the ball edge is marked as fitted on balls (in-sample)
    rows = _rows(tmp_path / "again", FOUR_BALLS, np.zeros(3))
    scored = bc.check(rows, graph, [out])["edges"][1]
    assert scored["from_balls"] and scored["max_mm"] < 0.5

    # a check that did not record the reference (older tool) is written, but not swappable
    old = json.loads(check.read_text())
    del old["world_reference_sha256"]
    check.write_text(json.dumps(old))
    assert bc.main(["edge", "--check", str(check), "--world", "world_a", "--out", str(out)]) == 2
    assert "哈希" in json.loads(out.read_text())["reasons"][0]
