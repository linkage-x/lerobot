"""World provenance: the one field an episode cannot be given afterwards."""

import json
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
        json.dumps({
            "world_continuity_state": "BROKEN",
            "generated_utc": "2026-08-20T00:00:00Z",
            "calibration_id": "calib_20260820",
            "world_frame_id": "world_20260820_000000",
        }),
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
    block = wp.read_world_provenance(Path(__file__).resolve().parents[2])
    assert block["status"] == wp.STATUS_OK
    assert block["world_frame_id"] == "world_20260819_031843"


def test_lr3_writer_stamps_info_json(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

    _write_reference(tmp_path / "repo")
    block = wp.read_world_provenance(tmp_path / "repo")
    writer = lr3.Lr3Writer(
        tmp_path / "ds", repo_id="repo", task="pick", fps=2, world_frame=block,
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
