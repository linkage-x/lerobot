"""register_fr3_base: the FR3 base enters the world graph as a measured edge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.thor.gmsl2 import register_fr3_base as reg, world_provenance as wp


def _report(**overrides) -> dict:
    report = {
        "schema": reg.REPORT_SCHEMA,
        "verdict": "ok",
        "reasons": [],
        "edge": {
            "from_world_frame_id": "world_a",
            "to_world_frame_id": "fr3_base",
            "T_to_from": [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "method": "test",
        },
        "target_node": {"world_frame_id": "fr3_base", "reason": "robot_base"},
    }
    report.update(overrides)
    return report


def test_apply_adds_the_edge_and_the_target_node() -> None:
    graph = {"version": 1, "nodes": [{"world_frame_id": "world_a"}], "edges": []}
    updated = reg.apply_report(_report(), graph)
    assert [n["world_frame_id"] for n in updated["nodes"]] == ["world_a", "fr3_base"]
    T, _ = wp.world_transform(updated, "world_a", "fr3_base")
    assert T[0][3] == pytest.approx(0.5)
    assert graph["edges"] == []  # the input is not mutated


def test_apply_refuses_a_rejected_report() -> None:
    with pytest.raises(ValueError, match="verdict is 'rejected'"):
        reg.apply_report(_report(verdict="rejected", reasons=["too few"]), {"nodes": [{"world_frame_id": "world_a"}]})


def test_apply_refuses_an_unknown_source_world() -> None:
    with pytest.raises(ValueError, match="world_a is not a node"):
        reg.apply_report(_report(), {"nodes": [{"world_frame_id": "world_b"}], "edges": []})


def test_apply_refuses_a_second_edge_between_connected_worlds() -> None:
    graph = reg.apply_report(_report(), {"nodes": [{"world_frame_id": "world_a"}], "edges": []})
    with pytest.raises(ValueError, match="already connected"):
        reg.apply_report(_report(), graph)


def test_solve_recovers_a_known_edge_from_a_synthetic_p0_run(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)

    def T_of(rotvec, t):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        T[:3, 3] = t
        return T

    def look_at(position, target):
        z = np.asarray(target, float) - position
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        T[:3, 3] = position
        return T

    T_world_base = T_of([0.2, -2.0, 0.4], [1.2, 0.1, 0.7])
    T_tcp_tag = T_of([3.1, 0.0, 0.1], [0.0, 0.0, 0.05])
    scale, size = 0.99, 0.16
    centre_base = np.array([0.5, 0.0, 0.3])
    centre_world = (T_world_base @ np.r_[centre_base, 1.0])[:3]
    cams = {
        f"cam_{i:02d}": look_at(centre_world + np.array([1.5 * np.cos(a), 1.5 * np.sin(a), 0.9]), centre_world)
        for i, a in enumerate(np.linspace(0, 2 * np.pi, 5, endpoint=False))
    }
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    D = np.array([[-0.05], [0.0], [0.0], [0.0]])

    root = tmp_path / "repo"
    world_dir = root / "world"
    world_dir.mkdir(parents=True)
    (world_dir / wp.WORLD_REFERENCE_FILE).write_text(
        json.dumps(
            {
                "world_frame_id": "world_a",
                "calibration_id": "calib_x_extrinsics",
                "cameras": {name: {"T_world_camera": T.tolist()} for name, T in cams.items()},
            }
        )
    )
    intr_dir = root / "outputs" / "calibration" / "calib_x_intrinsics"
    intr_dir.mkdir(parents=True)
    rows = []
    for name in cams:
        path = intr_dir / f"{name}.json"
        path.write_text(json.dumps({"model": "opencv_fisheye", "camera_matrix": K.tolist(), "dist_coeffs": D.ravel().tolist(), "image_width": 1920, "image_height": 1080}))
        rows.append({"camera_name": name, "status": "ok", "intrinsics_json": str(path)})
    (intr_dir / "summary.json").write_text(json.dumps({"cameras": rows}))

    h = size * scale / 2
    X = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    records = []
    for i in range(40):
        T_base_tcp = T_of(rng.normal(0, 0.3, 3) + [np.pi, 0, 0], centre_base + rng.normal(0, 0.12, 3))
        T_world_tag = T_world_base @ T_base_tcp @ T_tcp_tag
        entry = {}
        for name, T_world_c in cams.items():
            T_c_tag = np.linalg.inv(T_world_c) @ T_world_tag
            if (T_c_tag[:3, :3] @ [0, 0, 1])[2] > -0.2:  # tag faces away
                continue
            rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T_c_tag[:3, :3]))
            uv, _ = cv2.fisheye.projectPoints(X.reshape(1, 4, 3), rvec, np.ascontiguousarray(T_c_tag[:3, 3]).reshape(3, 1), K, D)
            uv = uv.reshape(4, 2) + rng.normal(0, 0.3, (4, 2))
            entry[name] = {"detections": [{"tag_id": 6, "corners_px": uv.tolist(), "image_width": 1920, "image_height": 1080}]}
        records.append({"capture_index": i, "T_base_tcp": T_base_tcp.tolist(), "cameras": entry})
    run = root / "outputs" / "calibration" / "p0" / "run"
    (run / "camera_calibration").mkdir(parents=True)
    (run / "captures.json").write_text(json.dumps({"records": records}))
    # The P0 solve's own (slightly wrong) cameras seed the initial guess.
    perturb = T_of([0.01, -0.01, 0.02], [0.01, 0.0, -0.01])
    (run / "camera_calibration" / "summary.json").write_text(
        json.dumps(
            {
                "marker": {"marker_size_m": size},
                "joint_solution": {
                    "tool_to_board": {"matrix_4x4": (T_tcp_tag @ perturb).tolist()},
                    "cameras": {
                        name: {"base_to_camera": {"matrix_4x4": (perturb @ np.linalg.inv(T_world_base) @ T).tolist()}}
                        for name, T in cams.items()
                    },
                },
            }
        )
    )

    out = tmp_path / "report.json"
    rc = reg.main(["--world-dir", str(world_dir), "solve", "--run", str(run), "--intrinsics", str(intr_dir / "summary.json"), "--out", str(out)])
    report = json.loads(out.read_text())
    assert rc == 0, report["reasons"]
    edge = report["edge"]
    assert (edge["from_world_frame_id"], edge["to_world_frame_id"]) == ("world_a", "fr3_base")
    # T_to_from takes world coordinates into the FR3 base.
    assert np.allclose(edge["T_to_from"], np.linalg.inv(T_world_base), atol=2e-3)
    assert report["edge"]["quality"]["tag_scale"] == pytest.approx(scale, abs=2e-3)
    assert report["edge"]["source"]["p0_run"] == "outputs/calibration/p0/run"


def test_solve_refuses_intrinsics_from_another_calibration(tmp_path: Path) -> None:
    world_dir = tmp_path / "world"
    world_dir.mkdir()
    (world_dir / wp.WORLD_REFERENCE_FILE).write_text(json.dumps({"world_frame_id": "w", "calibration_id": "calib_a_extrinsics", "cameras": {}}))
    intr = tmp_path / "calib_b_intrinsics"
    intr.mkdir()
    (intr / "summary.json").write_text(json.dumps({"cameras": []}))
    with pytest.raises(SystemExit, match="not the run the world's cameras were solved with"):
        reg.main(["--world-dir", str(world_dir), "solve", "--run", str(tmp_path), "--intrinsics", str(intr / "summary.json"), "--out", str(tmp_path / "r.json")])
