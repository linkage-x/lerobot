"""The comparison, the pointer write, and the staleness gate.

These lock down behaviour that a seven-day production incident paid for: a solve
whose result was never loaded because the only way to load it was a hand edit,
and two candidate calibrations that no automatic rule could have chosen between.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.data_collection_gui import calibration_promotion as promotion


def _pose(x: float, y: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> list[list[float]]:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def _write_run(
    root: Path,
    name: str,
    poses: dict[str, list[list[float]]],
    *,
    world_state: str = "CONTINUOUS",
    world_id: str = "world_a",
    rmse: float = 0.25,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "bundle_rmse_px": rmse,
                "joint_solution": {
                    "num_cameras": len(poses),
                    "cameras": {
                        cam: {"status": "ok", "base_to_camera": {"matrix_4x4": matrix}}
                        for cam, matrix in poses.items()
                    },
                },
                "world": {
                    "world_frame_id": world_id,
                    "reference_world_frame_id": "world_a",
                    "world_continuity_state": world_state,
                    "reason": "stable_cluster",
                    "stable_cameras": sorted(poses),
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_a_rig_that_did_not_move_reports_no_movement(tmp_path):
    poses = {"cam_a": _pose(0.0), "cam_b": _pose(1.0), "cam_c": _pose(0.0, 1.0)}
    _write_run(tmp_path, "live", poses)
    _write_run(tmp_path, "candidate", poses)
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    assert comparison["ok"]
    assert comparison["medianBaselineShiftMm"] == 0.0
    assert comparison["medianRotationDeg"] == 0.0
    assert promotion.promotion_blockers(comparison) == []


def test_moving_the_whole_rig_is_not_reported_as_cameras_moving(tmp_path):
    """The comparison must be blind to a change of world origin.

    Otherwise every re-registration would look like every camera moving at once,
    and the operator would learn to ignore the one number that matters.
    """
    live = {"cam_a": _pose(0.0), "cam_b": _pose(1.0), "cam_c": _pose(0.0, 1.0)}
    shifted = {cam: _pose(matrix[0][3] + 5.0, matrix[1][3] + 5.0) for cam, matrix in live.items()}
    _write_run(tmp_path, "live", live)
    _write_run(tmp_path, "candidate", shifted)
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    assert comparison["medianBaselineShiftMm"] == pytest.approx(0.0, abs=1e-6)


# Cameras spread around a workspace rather than in a line, which is what the
# real rig looks like: seven cameras ringing a table. The arrangement matters --
# see the degeneracy test below for what a collinear one would hide.
_RING = {
    "cam_a": _pose(0.0, 0.0),
    "cam_b": _pose(2.0, 0.0),
    "cam_c": _pose(2.0, 2.0),
    "cam_d": _pose(0.0, 2.0),
}


def test_the_median_singles_out_the_camera_that_moved(tmp_path):
    """One camera moving changes all of its own baselines and one of everyone else's.

    This is why the per-camera row is a median over partners rather than a max:
    with a max, every camera in the rig reports the mover's displacement and the
    table says nothing about who moved.
    """
    moved = dict(_RING)
    moved["cam_b"] = _pose(2.02, 0.0)  # 20 mm, along one baseline and across others
    _write_run(tmp_path, "live", _RING)
    _write_run(tmp_path, "candidate", moved)
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    rows = {row["camera"]: row for row in comparison["cameras"]}
    assert comparison["cameras"][0]["camera"] == "cam_b"
    # cam_b's three baselines change by 20.0, 14.2 and 0.1 mm -- the last one
    # because cam_c happens to sit square-on to the direction of travel. The
    # median rides over that degenerate partner and still reports the move.
    assert rows["cam_b"]["medianBaselineShiftMm"] > 5.0
    for camera in ("cam_a", "cam_c", "cam_d"):
        # Each of the three sees exactly one changed baseline out of three, so
        # the median leaves them at zero however large the mover's excursion.
        assert rows[camera]["medianBaselineShiftMm"] == pytest.approx(0.0, abs=1e-6)
    # ...while a max would have put the mover's displacement on other cameras'
    # rows too, which is the presentation this metric is chosen to avoid.
    assert rows["cam_a"]["maxBaselineShiftMm"] > 5.0
    assert rows["cam_d"]["maxBaselineShiftMm"] > 5.0


def test_a_translation_perpendicular_to_a_collinear_rig_is_under_reported(tmp_path):
    """A known blind spot of baseline lengths, recorded rather than hidden.

    Moving a camera at right angles to a baseline changes that baseline's length
    only to second order (1 m + 20 mm sideways is 1.0002 m), so a rig whose
    cameras were all in a line would barely register such a move. Two reasons
    this is acceptable rather than a defect to fix with a full SE(3) alignment:
    the production rig is a ring around a workspace, not a line -- and on real
    0820-vs-0902 data this metric picked out the same camera (cam_07) that an
    alignment-based residual did. The alternative needs an SVD, and the gateway
    has no numpy.

    If the rig ever does become near-collinear, this test is the place that says
    what stops working.
    """
    collinear = {"cam_a": _pose(0.0), "cam_b": _pose(1.0), "cam_c": _pose(2.0)}
    moved = dict(collinear)
    moved["cam_b"] = _pose(1.0, 0.02)  # 20 mm perpendicular to every baseline
    _write_run(tmp_path, "live", collinear)
    _write_run(tmp_path, "candidate", moved)
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    rows = {row["camera"]: row for row in comparison["cameras"]}
    # cam_b is still ranked first, but reports 0.2 mm for a 20 mm move.
    assert comparison["cameras"][0]["camera"] == "cam_b"
    assert rows["cam_b"]["medianBaselineShiftMm"] == pytest.approx(0.2, abs=0.02)


def test_a_changed_world_blocks_the_promotion(tmp_path):
    poses = {"cam_a": _pose(0.0), "cam_b": _pose(1.0)}
    _write_run(tmp_path, "live", poses, world_id="world_a")
    _write_run(tmp_path, "candidate", poses, world_id="world_b", world_state="RECONCILED")
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    kinds = {item["kind"] for item in promotion.promotion_blockers(comparison)}
    assert kinds == {"world_continuity", "world_frame_changed"}


def test_dropping_a_camera_blocks_the_promotion(tmp_path):
    _write_run(tmp_path, "live", {"cam_a": _pose(0.0), "cam_b": _pose(1.0)})
    _write_run(tmp_path, "candidate", {"cam_a": _pose(0.0)})
    comparison = promotion.compare_runs(
        promotion.load_run(tmp_path / "live"), promotion.load_run(tmp_path / "candidate")
    )
    assert [item["kind"] for item in promotion.promotion_blockers(comparison)] == ["cameras_removed"]
    assert comparison["removedCameras"] == ["cam_b"]


def test_an_unreadable_run_is_a_blocker_not_a_crash(tmp_path):
    (tmp_path / "candidate").mkdir()
    comparison = promotion.compare_runs(
        promotion.RunPoses(name="live", cameras={"cam_a": tuple(map(tuple, _pose(0.0)))}),
        promotion.load_run(tmp_path / "candidate"),
    )
    assert comparison["ok"] is False
    assert promotion.promotion_blockers(comparison)[0]["kind"] == "unreadable"


CONFIG = """calibration:
  # Do NOT repoint world_reference.json at this run: it is the historical anchor.
  intrinsics_run_name: old_intrinsics
  fixed_camera_run_name: old_extrinsics
  auxiliary_marker_run_name: ""
cube_tracker:
  camera_model: opencv_fisheye
"""


def test_the_pointer_write_keeps_every_comment_in_the_file():
    """A YAML round-trip would delete the reasoning above these two lines.

    That comment block is where the "do not repoint world_reference.json"
    warning lives, so losing it costs more than the promotion gains.
    """
    updated, changes = promotion.rewrite_pointers(CONFIG, {"extrinsics": "new_extrinsics"})
    assert "Do NOT repoint world_reference.json" in updated
    assert "fixed_camera_run_name: new_extrinsics" in updated
    assert "intrinsics_run_name: old_intrinsics" in updated
    assert changes == [
        {"kind": "extrinsics", "key": "fixed_camera_run_name", "from": "old_extrinsics", "to": "new_extrinsics"}
    ]


def test_both_pointers_can_move_in_one_write():
    updated, changes = promotion.rewrite_pointers(
        CONFIG, {"intrinsics": "new_intrinsics", "extrinsics": "new_extrinsics"}
    )
    assert "intrinsics_run_name: new_intrinsics" in updated
    assert "fixed_camera_run_name: new_extrinsics" in updated
    assert len(changes) == 2


def test_an_ambiguous_key_refuses_rather_than_guessing():
    """Writing the occurrence production does not read looks exactly like success."""
    doubled = CONFIG + "\nother:\n  fixed_camera_run_name: somewhere_else\n"
    with pytest.raises(promotion.PointerWriteError, match="2 处"):
        promotion.rewrite_pointers(doubled, {"extrinsics": "new_extrinsics"})


def test_a_missing_key_refuses():
    with pytest.raises(promotion.PointerWriteError, match="找不到"):
        promotion.rewrite_pointers("cube_tracker:\n  camera_model: x\n", {"extrinsics": "y"})


def test_the_config_is_replaced_in_one_step(tmp_path):
    path = tmp_path / "tracking.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    updated, _ = promotion.rewrite_pointers(CONFIG, {"extrinsics": "new_extrinsics"})
    promotion.write_config_atomically(path, updated)
    assert "new_extrinsics" in path.read_text(encoding="utf-8")
    # No temp files left behind to be picked up as config by anything globbing.
    assert [p.name for p in tmp_path.iterdir()] == ["tracking.yaml"]


def test_staleness_ignores_the_live_run_and_anything_older(tmp_path):
    root = tmp_path / "calibration"
    live = _write_run(root, "live_extrinsics", {"cam_a": _pose(0.0)})
    newer = _write_run(root, "newer_extrinsics", {"cam_a": _pose(0.0)})
    older = _write_run(root, "older_extrinsics", {"cam_a": _pose(0.0)})
    import os

    os.utime(live, (1_700_000_100, 1_700_000_100))
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_200, 1_700_000_200))
    found = promotion.promotable_runs(root, suffix="_extrinsics", live_run="live_extrinsics")
    assert [row["run"] for row in found] == ["newer_extrinsics"]


def test_an_experiment_run_is_never_offered_for_promotion(tmp_path):
    """A gate that nags about every experiment is a gate that gets ignored."""
    root = tmp_path / "calibration"
    live = _write_run(root, "live_extrinsics", {"cam_a": _pose(0.0)})
    experiment = _write_run(root, "exp_extrinsics", {"cam_a": _pose(0.0)})
    import os

    os.utime(live, (1_700_000_100, 1_700_000_100))
    promotion.write_experiment_marker(experiment, reason="solved without exporting")
    os.utime(experiment, (1_700_000_200, 1_700_000_200))
    assert promotion.promotable_runs(root, suffix="_extrinsics", live_run="live_extrinsics") == []


def test_the_stale_message_names_the_run_production_is_actually_loading():
    text = promotion.stale_pointer_refusal(
        [{"run": "calib_0902_extrinsics", "updatedAt": "2026-09-02 10:38"}],
        kind_label="外参",
        live_run="calib_0820_extrinsics",
    )
    assert "calib_0820_extrinsics" in text
    assert "calib_0902_extrinsics" in text
    assert promotion.stale_pointer_refusal([], kind_label="外参", live_run="x") == ""


def test_mixed_lens_models_block_an_intrinsics_promotion(tmp_path):
    run = tmp_path / "candidate"
    for camera, model in (("cam_06", "opencv_fisheye"), ("cam_07", "opencv_rational")):
        directory = run / "converted" / f"{camera}_serial"
        directory.mkdir(parents=True)
        (directory / "intrinsics_producer.json").write_text(
            json.dumps({"camera_name": camera, "model": model}), encoding="utf-8"
        )
    comparison = promotion.compare_intrinsics_runs(
        promotion.IntrinsicsRun(name="live", cameras=["cam_06", "cam_07"]),
        promotion.load_intrinsics_run(run),
        tracker_model="opencv_fisheye",
    )
    assert comparison["mixedModels"] == ["opencv_fisheye", "opencv_rational"]
    assert [item["kind"] for item in promotion.intrinsics_blockers(comparison)] == ["mixed_models"]


def test_a_lens_run_the_tracker_cannot_load_blocks(tmp_path):
    run = tmp_path / "candidate"
    directory = run / "converted" / "cam_06_serial"
    directory.mkdir(parents=True)
    (directory / "intrinsics_producer.json").write_text(
        json.dumps({"camera_name": "cam_06", "model": "opencv_rational"}), encoding="utf-8"
    )
    comparison = promotion.compare_intrinsics_runs(
        promotion.IntrinsicsRun(name="live", cameras=["cam_06"]),
        promotion.load_intrinsics_run(run),
        tracker_model="opencv_fisheye",
    )
    kinds = [item["kind"] for item in promotion.intrinsics_blockers(comparison)]
    assert kinds == ["model_mismatch"]


def test_the_promotion_log_records_the_evidence_not_just_the_decision(tmp_path):
    record = promotion.promotion_record(
        changes=[{"kind": "extrinsics", "key": "fixed_camera_run_name", "from": "a", "to": "b"}],
        comparison={
            "medianBaselineShiftMm": 0.233,
            "medianRotationDeg": 0.0911,
            "worstPair": {"a": "cam_07", "b": "cam_14", "shiftMm": 0.832},
            "candidateWorld": {"worldFrameId": "world_a", "continuityState": "CONTINUOUS"},
            "addedCameras": [],
            "removedCameras": [],
        },
        acknowledged=["world_frame_changed"],
        note="fresh calibration for today's session",
    )
    assert record["evidence"]["medianBaselineShiftMm"] == 0.233
    assert record["acknowledgedBlockers"] == ["world_frame_changed"]
    path = tmp_path / "promotions.jsonl"
    promotion.append_promotion_log(path, record)
    promotion.append_promotion_log(path, record)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def _write_intrinsics_run(root: Path, name: str, models: dict[str, str]) -> Path:
    run = root / name
    for camera, model in models.items():
        directory = run / "converted" / f"{camera}_serial"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "intrinsics_producer.json").write_text(
            json.dumps({"camera_name": camera, "model": model}), encoding="utf-8"
        )
    (run / "summary.json").write_text(json.dumps({"camera_model": next(iter(models.values()))}), encoding="utf-8")
    return run


def test_the_two_spellings_of_one_model_are_the_same_model():
    """`camera_model: fisheye` in the config vs `"opencv_fisheye"` on disk.

    Comparing these raw is worse than not comparing: it marks every correct lens
    run as a mismatch, which either disables the check or silently empties the
    staleness gate of real candidates.
    """
    assert promotion.normalize_model("fisheye") == promotion.normalize_model("opencv_fisheye")
    assert promotion.normalize_model("rational") == promotion.normalize_model("opencv_rational")
    assert promotion.normalize_model("fisheye") != promotion.normalize_model("opencv_rational")


def test_a_matching_lens_run_is_not_reported_as_a_model_mismatch(tmp_path):
    run = _write_intrinsics_run(tmp_path, "candidate", {"cam_06": "opencv_fisheye"})
    comparison = promotion.compare_intrinsics_runs(
        promotion.IntrinsicsRun(name="live", cameras=["cam_06"]),
        promotion.load_intrinsics_run(run),
        tracker_model="fisheye",  # as the tracking config actually spells it
    )
    assert promotion.intrinsics_blockers(comparison) == []


def test_the_rational_twin_never_shows_up_as_a_promotable_lens_run(tmp_path):
    """Found against the real rig, not invented.

    `thor_gmsl2_selfcal_0804_fisheye_intrinsics` and `..._rational_intrinsics`
    were exported from one report seconds apart. Production loads the fisheye
    one, so the rational twin reads as "newer" forever, and the gate would have
    warned on every single trajectory run about a calibration the tracker would
    refuse to load.
    """
    root = tmp_path / "calibration"
    live = _write_intrinsics_run(root, "selfcal_fisheye_intrinsics", {"cam_06": "opencv_fisheye"})
    twin = _write_intrinsics_run(root, "selfcal_rational_intrinsics", {"cam_06": "opencv_rational"})
    newer = _write_intrinsics_run(root, "later_fisheye_intrinsics", {"cam_06": "opencv_fisheye"})
    import os

    os.utime(live, (1_700_000_100, 1_700_000_100))
    os.utime(twin, (1_700_000_101, 1_700_000_101))  # exported a second later
    os.utime(newer, (1_700_000_900, 1_700_000_900))

    found = promotion.promotable_runs(
        root, suffix="_intrinsics", live_run="selfcal_fisheye_intrinsics", require_model="fisheye"
    )
    assert [row["run"] for row in found] == ["later_fisheye_intrinsics"]
    # Without the filter the twin comes back, which is the bug this pins.
    unfiltered = promotion.promotable_runs(
        root, suffix="_intrinsics", live_run="selfcal_fisheye_intrinsics"
    )
    assert "selfcal_rational_intrinsics" in [row["run"] for row in unfiltered]
