"""World provenance: the one field an episode cannot be given afterwards."""

import csv
import json
import math
from pathlib import Path

import pytest

from tools.thor.gmsl2 import world_provenance as wp


def _write_reference(root: Path, **overrides) -> Path:
    payload = {
        "version": 1,
        "world_frame_id": "world_20260819_031843",
        "created_utc": "2026-08-19T03:18:43Z",
        "calibration_id": "thor_gmsl2_selfcal_0804_fisheye_extrinsics",
        "parent_world_frame_id": None,
        "revisions": [],
        "cameras": {"cam_06": {}, "cam_07": {}},
    }
    payload.update(overrides)
    path = root / wp.WORLD_SUBDIR / wp.WORLD_REFERENCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reads_the_frozen_reference(tmp_path: Path) -> None:
    path = _write_reference(tmp_path)

    block = wp.read_world_provenance(tmp_path)

    assert block["status"] == wp.STATUS_OK
    assert block["world_frame_id"] == "world_20260819_031843"
    assert block["calibration_id"] == "thor_gmsl2_selfcal_0804_fisheye_extrinsics"
    assert block["reference_cameras"] == ["cam_06", "cam_07"]
    assert block["reference_path"] == "tools/thor/gmsl2/world/world_reference.json"
    assert len(block["reference_sha256"]) == 64
    # The hash is of the file, so an edit to it is visible even when the id is
    # unchanged -- that is the audit trail, not the contract.
    before = block["reference_sha256"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert wp.read_world_provenance(tmp_path)["reference_sha256"] != before


def test_missing_reference_is_stamped_not_defaulted(tmp_path: Path) -> None:
    block = wp.read_world_provenance(tmp_path)

    assert block["status"] == wp.STATUS_MISSING
    assert block["world_frame_id"] == ""
    # The remedy has to be in the message: re-running freeze here is the exact
    # mistake the mechanism exists to prevent, so the note names it.
    assert "freeze" in block["note"]
    assert wp.describe(block).startswith("WARNING")


def test_unparseable_reference_does_not_masquerade_as_a_world(tmp_path: Path) -> None:
    path = tmp_path / wp.WORLD_SUBDIR / wp.WORLD_REFERENCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated", encoding="utf-8")

    block = wp.read_world_provenance(tmp_path)

    assert block["status"] == wp.STATUS_UNREADABLE
    assert block["world_frame_id"] == ""


def test_reference_without_an_id_is_incomplete(tmp_path: Path) -> None:
    _write_reference(tmp_path, world_frame_id="")

    block = wp.read_world_provenance(tmp_path)

    assert block["status"] == wp.STATUS_INCOMPLETE
    assert block["world_frame_id"] == ""


def test_registration_disagreeing_with_the_reference_is_flagged(tmp_path: Path) -> None:
    _write_reference(tmp_path)
    (tmp_path / wp.WORLD_SUBDIR / wp.WORLD_REGISTRATION_FILE).write_text(
        json.dumps(
            {
                "world_continuity_state": "BROKEN",
                "generated_utc": "2026-08-20T00:00:00Z",
                "calibration_id": "calib_20260820",
                "world_frame_id": "world_20260820_000000",
            }
        ),
        encoding="utf-8",
    )

    block = wp.read_world_provenance(tmp_path)

    assert block["last_registration"]["matches_reference"] is False
    assert "DISAGREES WITH REFERENCE" in wp.describe(block)


def test_single_world_passes_through(tmp_path: Path) -> None:
    _write_reference(tmp_path)
    block = wp.read_world_provenance(tmp_path)

    assert wp.assert_single_world([("ep0", block), ("ep1", block)]) == "world_20260819_031843"


def test_all_unstamped_is_allowed_and_reports_no_world() -> None:
    # Historical episodes predate the stamp. Refusing them would make every old
    # dataset unexportable without making anyone safer.
    assert wp.assert_single_world([("ep0", None), ("ep1", {})]) == ""


def test_two_worlds_are_refused(tmp_path: Path) -> None:
    _write_reference(tmp_path)
    first = wp.read_world_provenance(tmp_path)
    _write_reference(tmp_path, world_frame_id="world_20260901_120000")
    second = wp.read_world_provenance(tmp_path)

    with pytest.raises(wp.MixedWorldError) as excinfo:
        wp.assert_single_world([("ep0", first), ("ep1", second)])
    assert "world_20260819_031843" in str(excinfo.value)
    assert "world_20260901_120000" in str(excinfo.value)


def test_stamped_mixed_with_unstamped_is_refused(tmp_path: Path) -> None:
    # "Might be the same world" is not a coordinate system: an unstamped episode
    # cannot be proven to belong to the stamped one's frame.
    _write_reference(tmp_path)
    block = wp.read_world_provenance(tmp_path)

    with pytest.raises(wp.MixedWorldError) as excinfo:
        wp.assert_single_world([("ep0", block), ("ep_legacy", None)])
    assert "<unstamped>" in str(excinfo.value)


def test_a_failed_read_never_counts_as_a_world(tmp_path: Path) -> None:
    # A missing-reference block still has a world_frame_id key; it must not be
    # treated as an id just because the block exists.
    missing = wp.read_world_provenance(tmp_path)
    assert wp.world_frame_id_of(missing) == ""
    assert wp.assert_single_world([("ep0", missing)]) == ""


def test_repo_reference_is_readable() -> None:
    # The checked-in reference is the one Thor records against; if this stops
    # parsing, every episode recorded from this tree is unstamped.
    repo = Path(__file__).resolve().parents[2]
    block = wp.read_world_provenance(repo)
    assert block["status"] == wp.STATUS_OK
    # The id changes on every promotion that mints an island; what must hold is
    # that the world graph knows it, or restamping and cross-world edges cannot.
    graph = json.loads((repo / wp.WORLD_SUBDIR / "world_graph.json").read_text(encoding="utf-8"))
    assert block["world_frame_id"] in {node["world_frame_id"] for node in graph["nodes"]}


def test_lr3_writer_stamps_info_json(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

    _write_reference(tmp_path / "repo")
    block = wp.read_world_provenance(tmp_path / "repo")
    writer = lr3.Lr3Writer(
        tmp_path / "ds",
        repo_id="repo",
        task="pick",
        fps=2,
        world_frame=block,
    )
    writer.finalize()

    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["world_frame"]["world_frame_id"] == "world_20260819_031843"


def test_lr3_writer_without_provenance_says_unstamped(tmp_path: Path) -> None:
    # Not an omission: a reader must be able to tell "no world" from "this file
    # cannot say", which is the same distinction sidecar v3 draws for camera_set.
    pytest.importorskip("pyarrow")
    from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

    writer = lr3.Lr3Writer(tmp_path / "ds", repo_id="repo", task="pick", fps=2)
    writer.finalize()

    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["world_frame"]["status"] == "unstamped"
    assert info["world_frame"]["world_frame_id"] == ""


def _episode_meta(root: Path, name: str, block) -> None:
    ep = root / "episodes" / name
    ep.mkdir(parents=True, exist_ok=True)
    meta = {"episode_index": 0}
    if block is not None:
        meta["world_frame"] = block
    (ep / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_inspect_dataset_is_the_smoke_check(tmp_path: Path) -> None:
    _write_reference(tmp_path / "repo")
    block = wp.read_world_provenance(tmp_path / "repo")
    ds = tmp_path / "ds"
    _episode_meta(ds, "episode_000000", block)
    _episode_meta(ds, "episode_000001", block)

    code, lines = wp.inspect_dataset(ds)

    assert code == 0
    assert any("world_20260819_031843" in line for line in lines)
    assert lines[-1].startswith("OK:")


def test_inspect_dataset_fails_on_an_unstamped_episode(tmp_path: Path) -> None:
    _write_reference(tmp_path / "repo")
    block = wp.read_world_provenance(tmp_path / "repo")
    ds = tmp_path / "ds"
    _episode_meta(ds, "episode_000000", block)
    _episode_meta(ds, "episode_000001", None)

    code, lines = wp.inspect_dataset(ds)

    assert code == 1
    assert any("UNSTAMPED" in line for line in lines)


def test_inspect_dataset_reports_an_empty_dataset(tmp_path: Path) -> None:
    code, lines = wp.inspect_dataset(tmp_path)
    assert code == 2
    assert any("no episodes" in line for line in lines)


def _graph(root: Path) -> Path:
    path = root / wp.WORLD_SUBDIR / "world_graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "nodes": [
                    {
                        "world_frame_id": "world_20260923_143048",
                        "created_utc": "2026-09-28T07:04:22Z",
                        "calibration_id": "calib_20260923_cam13refit_extrinsics",
                        "parent_world_frame_id": "world_20260819_031843",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_restamp_keeps_the_original_and_is_idempotent(tmp_path: Path) -> None:
    from tools.thor.gmsl2 import restamp_world as rw

    old = {"world_frame_id": "world_20260819_031843", "status": "ok", "reference_sha256": "d7de"}
    _episode_meta(tmp_path, "episode_000000", old)
    _episode_meta(tmp_path, "episode_000001", {"world_frame_id": "world_20260928_063531", "status": "ok"})
    node = rw.load_world_node(_graph(tmp_path), "world_20260923_143048")

    dry, _ = rw.restamp([tmp_path], expect_from="world_20260819_031843", node=node, reason="r", apply=False)
    assert len(dry) == 1
    assert wp.inspect_dataset(tmp_path)[0] == 1  # dry run wrote nothing: still two worlds

    changed, skipped = rw.restamp(
        [tmp_path], expect_from="world_20260819_031843", node=node, reason="mount re-installed", apply=True
    )
    assert len(changed) == 1 and len(skipped) == 1  # the episode from a third world is left alone
    block = json.loads((tmp_path / "episodes/episode_000000/meta.json").read_text())["world_frame"]
    assert wp.world_frame_id_of(block) == "world_20260923_143048"
    assert block["restamp"]["original"] == old
    assert block["restamp"]["reason"] == "mount re-installed"
    assert "reference_sha256" not in block  # never copied from the file that was wrong

    again, _ = rw.restamp([tmp_path], expect_from="world_20260819_031843", node=node, reason="r", apply=True)
    assert again == []


def test_restamp_refuses_a_world_that_is_not_in_the_graph(tmp_path: Path) -> None:
    from tools.thor.gmsl2 import restamp_world as rw

    with pytest.raises(KeyError):
        rw.load_world_node(_graph(tmp_path), "world_20260923_999999")


def test_restamp_rewrites_a_symlinked_meta_once(tmp_path: Path) -> None:
    from tools.thor.gmsl2 import restamp_world as rw

    src = tmp_path / "src"
    _episode_meta(src, "episode_000000", {"world_frame_id": "world_20260819_031843", "status": "ok"})
    derived = tmp_path / "derived" / "episodes" / "episode_000000"
    derived.mkdir(parents=True)
    (derived / "meta.json").symlink_to(src / "episodes/episode_000000/meta.json")
    node = rw.load_world_node(_graph(tmp_path), "world_20260923_143048")

    changed, _ = rw.restamp(
        [src, tmp_path / "derived"], expect_from="world_20260819_031843", node=node, reason="r", apply=True
    )

    assert len(changed) == 1
    assert (derived / "meta.json").is_symlink()


# ------------------------------------------------- cross-world registration ---


def _rz(deg: float, t=(0.0, 0.0, 0.0)) -> list[list[float]]:
    import math

    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0, t[0]], [s, c, 0.0, t[1]], [0.0, 0.0, 1.0, t[2]], [0.0, 0.0, 0.0, 1.0]]


def _edge(a: str, b: str, T: list[list[float]]) -> dict:
    return {"from_world_frame_id": a, "to_world_frame_id": b, "T_to_from": T, "method": f"{a}->{b}"}


def test_world_transform_composes_and_walks_edges_backwards() -> None:
    np = pytest.importorskip("numpy")
    a_to_b, b_to_c = _rz(30, (1, 0, 0)), _rz(-75, (0, 2, 0.5))
    graph = {"edges": [_edge("a", "b", a_to_b), _edge("c", "b", np.linalg.inv(b_to_c).tolist())]}

    T, path = wp.world_transform(graph, "a", "c")
    assert np.allclose(T, np.asarray(b_to_c) @ np.asarray(a_to_b), atol=1e-12)
    assert [hop["traversed_reversed"] for hop in path] == [False, True]
    back, _ = wp.world_transform(graph, "c", "a")
    assert np.allclose(np.asarray(back) @ np.asarray(T), np.eye(4), atol=1e-12)
    assert wp.world_transform(graph, "a", "a") == ([[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)], [])


def test_unconnected_worlds_have_no_transform_not_an_identity() -> None:
    assert wp.world_transform({"edges": [_edge("a", "b", _rz(10))]}, "a", "z") is None
    assert wp.world_transform({"edges": []}, "a", "b") is None


def test_a_non_rigid_edge_is_refused() -> None:
    bad = _rz(10)
    bad[0][0] *= 1.01
    with pytest.raises(ValueError, match="not a rigid transform"):
        wp.world_transform({"edges": [_edge("a", "b", bad)]}, "a", "b")


def test_transform_pose7_matches_the_matrix_product() -> None:
    np = pytest.importorskip("numpy")

    def mat(pose):
        x, y, z, qx, qy, qz, qw = pose
        T = np.eye(4)
        T[:3, :3] = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
        T[:3, 3] = [x, y, z]
        return T

    q = np.array([0.3, -0.5, 0.1, 0.8])
    q /= np.linalg.norm(q)
    pose = [0.4, -0.2, 0.9, *q]
    # A 121 deg edge like the real one, plus a near-180 deg one (Shepperd's other branches).
    for T in (_rz(121, (1.2, 0.01, 0.74)), [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]):
        out = wp.transform_pose7(T, pose)
        assert np.allclose(mat(out), np.asarray(T) @ mat(pose), atol=1e-12)
    nan_pose = [float("nan")] * 7
    assert all(v != v for v in wp.transform_pose7(_rz(10), nan_pose))


def test_repo_graph_reaches_the_fr3_base_from_the_current_world() -> None:
    # The FR3 base enters as an edge, not as a redefinition of the world: the
    # reference keeps naming the camera island and the graph says how to get out.
    repo = Path(__file__).resolve().parents[2]
    block = wp.read_world_provenance(repo)
    found = wp.world_transform(wp.read_world_graph(repo), block["world_frame_id"], "fr3_base")
    assert found is not None
    T, path = found
    assert path and path[-1]["to_world_frame_id"] == "fr3_base"


def test_reexpress_pose_csv_moves_every_pose_group_by_its_episode(tmp_path):
    src = tmp_path / "state_action.right.csv"
    src.write_text(
        "episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw,"
        "action_x_m,action_y_m,action_z_m,action_qx,action_qy,action_qz,action_qw,gripper\n"
        "0,0,0.1,0.2,0.3,0,0,0,1,1.0,0.0,0.0,0,0,0,1,0.5\n"
        "0,1,nan,nan,nan,nan,nan,nan,nan,1.0,0.0,0.0,0,0,0,1,0.5\n",
        encoding="utf-8",
    )
    # 90 deg about z, then (1, 2, 3).
    T = [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
    dst = tmp_path / "out" / "state_action.right.csv"

    assert wp.reexpress_pose_csv(src, dst, {0: T}) == 2

    rows = list(csv.DictReader(dst.open()))
    assert [float(rows[0][k]) for k in ("state_x_m", "state_y_m", "state_z_m")] == pytest.approx([0.8, 2.1, 3.3])
    assert float(rows[0]["state_qz"]) == pytest.approx(math.sqrt(0.5))
    assert [float(rows[0][k]) for k in ("action_x_m", "action_y_m", "action_z_m")] == pytest.approx([1.0, 3.0, 3.0])
    assert rows[0]["gripper"] == "0.5"
    # A gap stays a gap rather than becoming the edge's translation.
    assert math.isnan(float(rows[1]["state_x_m"]))

    with pytest.raises(RuntimeError, match="episode 0 has no transform"):
        wp.reexpress_pose_csv(src, dst, {1: T})
