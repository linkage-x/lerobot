from __future__ import annotations

import pytest

from tools.thor.gmsl2 import label_provenance as lp

_T = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _fr3_world(transform, created):
    return {
        "world_frame_id": "fr3_base",
        "status": "ok",
        "reexpressed_from": {"world_frame_id": "world_a", "status": "ok"},
        "T_target_source": transform,
        "edges": [{"from_world_frame_id": "world_a", "to_world_frame_id": "fr3_base", "created_utc": created}],
    }


def _session(**overrides):
    labels = {
        "tracked": True,
        "intrinsics": "intr",
        "extrinsics": "calib_1014",
        "camera_state": "ok",
        "marker_to_tcp": "bundle_0929.json",
        "tcp_link": "corenetic_gripper_ee",
        "sidecar_chain": "{}",
        "box_pairing": ["raw_sof"],
    }
    labels.update(overrides)
    return labels


def _info(world, sessions, mixed=()):
    return {"world_frame": world, "label_provenance": lp.provenance_block(world, sessions, list(mixed))}


def test_box_pairing_reads_the_recorder_and_the_realignment():
    recorded_old = {"box_camera_alignment": {"mode": "sensor_timestamp_sof", "exposure_fraction": 0.0}}
    recorded_new = {
        "box_camera_alignment": {
            "mode": "sensor_timestamp_sof",
            "exposure_fraction": -0.5,
            "box_targets": {"box1": {"target": "mid_exposure_plus_d_box"}, "box2": {"target": "raw_sof"}},
        }
    }
    realigned = {"boxes": {"box1": {"target": "mid_exposure_plus_d_box", "source": "episode"}}}
    assert lp.box_pairing(recorded_old, None) == "raw_sof"
    assert lp.box_pairing(recorded_new, None) == "mid_exposure_plus_d_box"
    assert lp.box_pairing(recorded_old, realigned) == "mid_exposure_plus_d_box"
    assert lp.box_pairing({}, None) == "unrecorded"


def test_two_exports_through_different_edges_do_not_combine():
    # Both say fr3_base; the 10-14 and 10-15 ball edges differ by millimetres.
    old = _info(_fr3_world(_T, "2026-10-14T09:17:42Z"), {"s1": _session()})
    shifted = [row[:] for row in _T]
    shifted[0][3] = 0.104
    new = _info(_fr3_world(shifted, "2026-10-15T03:08:47Z"), {"s2": _session()})

    lp.assert_compatible_datasets([("a", old), ("b", old)])
    with pytest.raises(lp.MixedLabelsError, match="edge"):
        lp.assert_compatible_datasets([("a", old), ("b", new)])


def test_stamped_and_pre_stamp_exports_do_not_combine():
    world = _fr3_world(_T, "2026-10-15T03:08:47Z")
    stamped = _info(world, {"s1": _session()})
    legacy = {"world_frame": world, "tcp_frame": {"link": "corenetic_gripper_ee"}}
    with pytest.raises(lp.MixedLabelsError, match="extrinsics"):
        lp.assert_compatible_datasets([("new", stamped), ("old", legacy)])
    # Two pre-stamp exports through the same edge still combine.
    lp.assert_compatible_datasets([("old1", legacy), ("old2", legacy)])


def test_a_mixed_export_trains_only_on_its_own():
    world = _fr3_world(_T, "2026-10-15T03:08:47Z")
    sessions = {"s1": _session(), "s2": _session(camera_state="stale")}
    mixed = _info(world, sessions, mixed=["camera_state"])
    lp.assert_compatible_datasets([("mixed", mixed)])
    with pytest.raises(lp.MixedLabelsError, match="--allow-mixed-labels"):
        lp.assert_compatible_datasets([("mixed", mixed), ("again", mixed)])


def test_session_chain_ignores_untracked_sessions_but_not_their_box_pairing():
    sessions = {"a": _session(), "b": {"tracked": False, "box_pairing": ["raw_sof"]}}
    assert lp.assert_single_session_chain(sessions) == []
    sessions["b"]["box_pairing"] = ["mid_exposure_plus_d_box"]
    with pytest.raises(lp.MixedLabelsError, match="box_pairing"):
        lp.assert_single_session_chain(sessions)


def test_warnings_name_stale_cameras_and_a_superseded_edge():
    key = lp.dataset_key(_fr3_world(_T, "2026-10-14T09:17:42Z"), {"s": _session(camera_state="stale")})
    current = lp.dataset_key(_fr3_world(_T, "2026-10-15T03:08:47Z"), {})["edge"]
    warnings = lp.label_warnings(key, current)
    assert any("camera state stale" in w for w in warnings)
    assert any("created 2026-10-14T09:17:42Z; now 2026-10-15T03:08:47Z" in w for w in warnings)
    assert lp.label_warnings(lp.dataset_key(_fr3_world(_T, "x"), {"s": _session()}), None) == []


_BUNDLE_0929 = "third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/marker_to_tcp_calibration_20260929.json"


def _tracked_session(tmp_path, bundle_path, hands=("left", "right"), box_alignment=None):
    import json

    session = tmp_path / "s"
    sidecar = session / "derived" / lp.SIDECAR
    sidecar.mkdir(parents=True)
    for cube in hands:
        (sidecar / f"state_action.{cube}.csv").write_text("episode_index\n", encoding="utf-8")
    (session / "meta").mkdir()
    info = {"derived": {lp.SIDECAR: {"camera_state": {"verdict": "ok"}}}}
    if box_alignment is not None:
        info["box_alignment"] = box_alignment
    (session / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    summary = {"ee_from_cube": {"marker_to_tcp_calibration_path": bundle_path}}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return session, run


def test_the_production_left_hand_is_unvalidated_and_says_on_which_links():
    """Bundle 0929 + today's delay table: left is a camera pivot from 08-25, flagged stale, no d_box."""
    import json

    bundle = json.loads((lp._REPO_ROOT / _BUNDLE_0929).read_text(encoding="utf-8"))
    delays = json.loads((lp._REPO_ROOT / lp.BOX_TIME_DELAYS).read_text(encoding="utf-8"))
    left = lp.hand_validation("left", bundle, delays, {})
    assert left["status"] == lp.UNVALIDATED and left["box_device_id"] == "box1672693301"
    reasons = " | ".join(left["reasons"])
    assert "never checked by the laser tracker" in reasons
    assert "STALE RISK 2026-09-28" in reasons
    assert "d_box of box1672693301 not measured" in reasons
    right = lp.hand_validation("right", bundle, delays, {})
    assert right == {
        "status": lp.MEASURED, "reasons": [], "box_device_id": "box1819152274", "d_box": "table",
        "marker_to_tcp_laser": True,
    }  # fmt: skip


def test_an_episode_s_own_d_box_counts_and_a_missing_bundle_is_a_reason():
    bundle = {
        "cubes": {"left": {"device_id": "box1", "pivot": {"source": "laser_tracker"}}},
    }
    realigned = {"0": {"boxes": {"box1": {"target": "mid_exposure_plus_d_box", "source": "episode"}}}}
    assert lp.hand_validation("left", bundle, {"boxes": {}}, realigned)["d_box"] == "episode"
    assert lp.hand_validation("left", bundle, {"boxes": {}}, realigned)["status"] == lp.MEASURED
    missing = lp.hand_validation("left", None, None, {})
    assert missing["status"] == lp.UNVALIDATED and "not readable" in missing["reasons"][0]


def test_the_export_carries_each_hand_s_status_outside_the_key(tmp_path):
    session, run = _tracked_session(tmp_path, _BUNDLE_0929)
    labels = lp.session_labels(session, run, {0: {}}, lp._REPO_ROOT)
    assert {cube: hand["status"] for cube, hand in labels["hands"].items()} == {
        "left": lp.UNVALIDATED,
        "right": lp.MEASURED,
    }
    block = lp.provenance_block(None, {"s": labels}, [])
    assert block["hands"]["left"]["sessions"] == {"s": lp.UNVALIDATED}
    assert "hands" not in block["key"]  # validation is how far to trust labels, not which labels
    right_only, run_r = _tracked_session(tmp_path / "r", _BUNDLE_0929, hands=("right",))
    other = lp.session_labels(right_only, run_r, {0: {}}, lp._REPO_ROOT)
    assert lp.provenance_block(None, {"r": other}, [])["key"] == block["key"]
    (warning,) = lp.hand_warnings(block)
    assert warning.startswith("left hand labels are unvalidated (marker->TCP never checked by the laser")
    assert lp.hand_warnings(lp.provenance_block(None, {"r": other}, [])) == []
