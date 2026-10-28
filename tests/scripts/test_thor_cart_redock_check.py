"""cart_redock_check (roadmap B5'): did a cart come back where it was registered?

Synthetic fisheye cameras look at tags on two carts and a table; the observations are exact projections plus a
little noise, so what the check reports can be held against the motion that was applied.  The camera layout
copies the two traps the first real captures (10-27) showed: a camera pair 95 mm apart (cam_03/cam_13) and a
camera whose calibrated pose went stale (cam_09 after its mount was changed).
"""

# ruff: noqa: N803, N806  (R, T as in the tool)
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.thor.gmsl2 import cart_redock_check as rc

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
D = np.array([-0.07, -0.007, 0.003, -0.0007])
EDGE = 0.076
TAGS = {
    # left cart: a strip along its table side, like the real 20-29, and one tag off the strip
    20: (-0.25, 0.40, -0.10), 21: (0.0, 0.40, -0.10), 22: (0.25, 0.40, -0.10), 23: (0.10, 0.25, -0.10),
    40: (-0.25, -0.40, -0.10), 41: (0.0, -0.45, -0.10), 42: (0.25, -0.30, -0.10),
    45: (-0.30, 0.0, 0.0), 46: (0.30, 0.10, 0.0), 47: (0.0, -0.10, 0.0),
}  # fmt: skip


def _layout(tmp_path: Path) -> dict:
    layout = {
        "schema": "cart_tags/v1",
        "dictionary": "DICT_6X6_50",
        "groups": {
            "left_cart": {"kind": "cart", "base": "fr3_base", "ids": [20, 21, 22, 23]},
            "right_cart": {"kind": "cart", "base": "fr3_base_right", "ids": [40, 41, 42]},
            "table": {"kind": "static", "ids": [45, 46, 47]},
        },
        "black_edge_mm": {str(t): 76 for t in TAGS},
        "thresholds": {"cart_reach_m": 1.2, "cart_mm": 2.0, "static_px": 0.6, "static_reach_m": 1.0},
    }
    path = tmp_path / "cart_tags.json"
    path.write_text(json.dumps(layout))
    return rc.load_layout(path)


def _look_at(position, target=(0.0, 0.0, -0.05)) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    z = np.asarray(target) - position
    z /= np.linalg.norm(z)
    y = np.array([0.0, 0.0, -1.0]) - (np.array([0.0, 0.0, -1.0]) @ z) * z
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    T[:3, 3] = position
    return T


def _cameras() -> dict:
    poses = {
        "cam_03": _look_at((0.05, -1.2, 1.1)),
        "cam_13": _look_at((-0.045, -1.2, 1.1)),  # 95 mm from cam_03
        "cam_07": _look_at((-1.2, 0.0, 1.0)),
        "cam_09": _look_at((1.2, 0.0, 1.0)),
        "cam_14": _look_at((0.0, 1.2, 1.1)),
    }
    return {n: {"K": K, "D": D, "size": (1920, 1080), "T_world_camera": T} for n, T in poses.items()}


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _corners(motion: dict | None = None) -> dict[int, np.ndarray]:
    """World corners of every tag; ``motion`` maps a tag id to the rigid (R, t) it underwent."""
    h = EDGE / 2
    out = {}
    for tag, c in TAGS.items():
        P = np.asarray(c) + np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
        if motion and tag in motion:
            R, t = motion[tag]
            P = P @ R.T + t
        out[tag] = P
    return out


def _detections(cams, corners, true_poses=None, noise_px=0.03, seed=0, hide=()) -> dict:
    """What each camera sees: exact fisheye projections from its *true* pose, plus noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, cam in cams.items():
        seen_by = dict(cam, T_world_camera=(true_poses or {}).get(name, cam["T_world_camera"]))
        tags = {}
        for tag, P in corners.items():
            if (name, tag) in hide:
                continue
            uv = rc.project(seen_by, P)
            if np.all((uv > 0) & (uv < [1920, 1080])):
                tags[tag] = {"corners_px": uv + rng.normal(0.0, noise_px, uv.shape)}
        out[name] = {"tags": tags}
    return out


def _capture(cams, layout, detections, extrinsics_sha="e" * 64) -> dict:
    m = rc.measure_from_detections(cams, detections, layout)
    return {
        "schema": rc.MEASUREMENT_SCHEMA,
        "measured_utc": "2026-10-27T00:00:00+00:00",
        "episode": "synthetic",
        "calibration": {
            "extrinsics_run": "calib_x_extrinsics",
            "extrinsics_sha256": extrinsics_sha,
            "intrinsics_sha256": "i" * 64,
        },
        **m,
    }


def test_the_projection_is_opencv_s_fisheye() -> None:
    cam = _cameras()["cam_07"]
    X = _corners()[21]
    T_cw = np.linalg.inv(cam["T_world_camera"])
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    tvec = np.ascontiguousarray(T_cw[:3, 3]).reshape(3, 1)
    uv, _ = cv2.fisheye.projectPoints(X.reshape(1, -1, 3), rvec, tvec, K, D.reshape(4, 1))
    assert np.allclose(rc.project(cam, X), uv.reshape(-1, 2), atol=1e-6)


def test_an_unmoved_scene_passes_and_the_tags_land_where_they_are(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, _corners(), seed=2))
    for tag, P in _corners().items():
        got = np.asarray(ref["tags"][str(tag)]["corners_world_m"])
        assert np.abs(got - P).max() < 5e-4, tag
        assert ref["tags"][str(tag)]["edge_mm"] == pytest.approx(76.0, abs=0.5)
    report = rc.compare(cams, ref, cur, layout)
    assert report["verdict"] == "ok" and report["absolute_replay_allowed"]
    for g in report["groups"].values():
        assert g["within_reach_bound_mm"] < 1.0, g  # mostly |ω| x 1.2 m: noise in the rotation, not motion


def test_a_cart_that_came_back_3_mm_and_0_2_deg_off_is_refused_and_its_motion_recovered(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    R, t = _rot_z(0.2), np.array([0.003, -0.001, 0.0])
    moved = _corners(dict.fromkeys((40, 41, 42), (R, t)))
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, moved, seed=2))
    report = rc.compare(cams, ref, cur, layout)
    assert report["verdict"] == "cart_moved" and not report["absolute_replay_allowed"]
    right, left = report["groups"]["right_cart"], report["groups"]["left_cart"]
    assert right["status"] == "moved" and left["status"] == "ok"
    assert right["rotation_deg"] == pytest.approx(0.2, abs=0.02)
    Delta = np.asarray(right["delta_world"]["matrix_4x4"])
    assert not right["delta_world"]["validated_for_correction"]
    # Δ moves the reference corners onto the moved ones -- and so would move the registered base
    for tag in (40, 41, 42):
        P = _corners()[tag]
        assert np.abs(P @ Delta[:3, :3].T + Delta[:3, 3] - moved[tag]).max() < 3e-4


def test_when_the_table_moves_no_cart_result_is_trusted(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    shifted = _corners(dict.fromkeys((45, 46, 47), (np.eye(3), np.array([0.002, 0.0, 0.0]))))
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, shifted, seed=2))
    report = rc.compare(cams, ref, cur, layout)
    assert report["verdict"] == "cameras_or_table_moved"
    assert report["static_check"]["status"] == "table_or_all_cameras_moved"
    assert report["static_check"]["cameras_ok"] == [] and report["static_check"]["tags_bumped"] == []
    assert {report["groups"][g]["status"] for g in ("left_cart", "right_cart")} == {"untrusted"}
    assert report["groups"]["left_cart"]["status_untrusted"] == "ok"
    assert report["groups"]["table"]["status"] == "info"  # its 3D motion is shown, the pixels decide


def _turned(cams, name, deg) -> dict:
    """``name``'s true pose after its mount turned ``deg`` about the vertical -- the calibration still says the old."""
    T = cams[name]["T_world_camera"].copy()
    T[:3, :3] = _rot_z(deg) @ T[:3, :3]
    return {name: T}


def test_a_camera_that_moved_since_the_reference_is_left_out_of_the_carts(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(
        cams, layout, _detections(cams, _corners(), true_poses=_turned(cams, "cam_07", 0.1), seed=2)
    )
    report = rc.compare(cams, ref, cur, layout)
    static = report["static_check"]
    assert static["status"] == "ok" and static["cameras_moved"] == ["cam_07"]
    assert all(s > 0.6 for s in static["cameras"]["cam_07"]["shift_px"].values())
    assert "cam_07" in report["cameras_left_out_of_carts"]
    for g in ("left_cart", "right_cart"):
        assert all("cam_07" not in c for c in report["groups"][g]["tag_cameras"].values())
    assert report["verdict"] == "ok"  # the carts did not move; the other cameras say so


def test_a_bumped_table_tag_is_told_from_a_moved_camera(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    bumped = _corners({46: (np.eye(3), np.array([0.004, 0.0, 0.0]))})
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, bumped, seed=2))
    static = rc.compare(cams, ref, cur, layout)["static_check"]
    assert static["tags_bumped"] == [46] and static["cameras_moved"] == []
    assert static["status"] == "ok"


def test_a_camera_that_sees_no_table_tag_cannot_vouch_and_is_left_out(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    hidden = {("cam_14", t) for t in (45, 46, 47)}
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1, hide=hidden))
    cur = _capture(cams, layout, _detections(cams, _corners(), seed=2, hide=hidden))
    report = rc.compare(cams, ref, cur, layout)
    assert report["static_check"]["cameras_unverified"] == ["cam_14"]
    assert report["cameras_left_out_of_carts"] == ["cam_14"] and report["verdict"] == "ok"


def test_a_three_tag_cart_keeps_all_three(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    restuck = _corners({41: (np.eye(3), np.array([0.003, 0.0, 0.0]))})
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, restuck, seed=2))
    right = rc.compare(cams, ref, cur, layout)["groups"]["right_cart"]
    # two tags left would fix no rotation about the line through them; the moved tag shows as misfit instead
    assert right["tags"] == [40, 41, 42] and right["tags_dropped"] == []
    assert right["fit_rms_mm"] > 0.5


def test_a_measurement_from_before_the_static_check_still_compares(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, _corners(), seed=2))
    for v in ref["cameras"].values():
        del v["corners_px"]  # older measurements kept pixels only per triangulated tag
    px = rc.camera_pixels(ref)
    assert np.allclose(px["cam_07"][45], ref["tags"]["45"]["observations_px"]["cam_07"])
    assert rc.compare(cams, ref, cur, layout)["verdict"] == "ok"


def test_the_table_datum_keeps_world_corners_and_camera_pixels(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    m = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    m.update(dictionary="DICT_6X6_50")
    m["calibration"]["world_frame_id"] = "world_x"
    datum = rc.freeze_datum(m, layout, "synthetic")
    assert sorted(datum["tags"]) == ["45", "46", "47"] and datum["world_frame_id"] == "world_x"
    assert datum["plane"]["rms_mm"] < 0.5 and abs(datum["plane"]["normal_world"][2]) > 0.999
    assert 0.6 + EDGE < datum["plane"]["extent_m"][0] < 0.75  # 45..46 plus a tag, along the principal axis
    assert set(datum["camera_pixels"]["cam_03"]) == {"45", "46", "47"}
    later = _capture(
        cams, layout, _detections(cams, _corners(), true_poses=_turned(cams, "cam_09", 0.1), seed=2)
    )
    check = rc.check_cameras(json.loads(json.dumps(datum)), later, 0.6)
    assert check["cameras_moved"] == ["cam_09"] and check["status"] == "ok"


def test_a_datum_needs_three_table_tags(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    hidden = {(n, 47) for n in cams}
    m = _capture(cams, layout, _detections(cams, _corners(), seed=1, hide=hidden))
    m.update(dictionary="DICT_6X6_50")
    with pytest.raises(ValueError, match="need 3"):
        rc.freeze_datum(m, layout, "synthetic")


def test_a_camera_on_a_stale_pose_is_left_out_of_every_tag(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    stale = cams["cam_09"]["T_world_camera"].copy()
    stale[:3, :3] = _rot_z(1.0) @ stale[:3, :3]  # the mount sagged; the calibration did not
    det = _detections(cams, _corners(), true_poses={"cam_09": stale}, seed=1)
    ref = _capture(cams, layout, det)
    assert ref["cameras_excluded"] == ["cam_09"]
    assert "stale or moved pose" in ref["cameras"]["cam_09"]["why"]
    for tag, P in _corners().items():
        entry = ref["tags"][str(tag)]
        assert "cam_09" not in entry["cameras"]
        if entry["status"] == "ok":
            assert np.abs(np.asarray(entry["corners_world_m"]) - P).max() < 1e-3, tag
    cur = _capture(cams, layout, _detections(cams, _corners(), true_poses={"cam_09": stale}, seed=2))
    assert rc.compare(cams, ref, cur, layout)["verdict"] == "ok"


def test_both_captures_are_triangulated_by_the_cameras_they_share(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    # a calibration that is slightly off for cam_07 makes the 3-camera and 2-camera points differ
    true = {"cam_07": cams["cam_07"]["T_world_camera"] @ np.diag([1.0, 1.0, 1.0, 1.0])}
    true["cam_07"] = true["cam_07"].copy()
    true["cam_07"][:3, 3] += [0.0, 0.0015, 0.0]
    ref = _capture(cams, layout, _detections(cams, _corners(), true_poses=true, seed=1))
    hidden = {("cam_07", 21)}
    cur = _capture(cams, layout, _detections(cams, _corners(), true_poses=true, seed=2, hide=hidden))
    assert "cam_07" in ref["tags"]["21"]["cameras"] and "cam_07" not in cur["tags"]["21"]["cameras"]
    report = rc.compare(cams, ref, cur, layout)
    left = report["groups"]["left_cart"]
    assert "cam_07" not in left["tag_cameras"]["21"]
    assert left["tag_residual_mm"]["21"] < 0.3 and report["verdict"] == "ok"


def test_the_close_camera_pair_alone_does_not_carry_a_tag(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    only_pair = {(n, 47) for n in ("cam_07", "cam_09", "cam_14")}
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1, hide=only_pair))
    cur = _capture(cams, layout, _detections(cams, _corners(), seed=2, hide=only_pair))
    assert sorted(ref["tags"]["47"]["cameras"]) == ["cam_03", "cam_13"]
    assert rc.ray_angle_deg(cams, ["cam_03", "cam_13"], np.array(TAGS[47])) < rc.MIN_RAY_ANGLE_DEG
    table = rc.compare(cams, ref, cur, layout)["groups"]["table"]
    assert 47 in table["tags_unpaired"] and 47 not in table["tags"]


def test_a_tag_that_moved_on_its_cart_is_dropped_not_taken_for_the_cart(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    restuck = _corners({22: (np.eye(3), np.array([0.005, 0.0, 0.0]))})
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    cur = _capture(cams, layout, _detections(cams, restuck, seed=2))
    left = rc.compare(cams, ref, cur, layout)["groups"]["left_cart"]
    assert [d["tag"] for d in left["tags_dropped"]] == [22]
    assert left["status"] == "ok" and left["tags"] == [20, 21, 23]


def test_a_reference_from_other_camera_poses_is_refused(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    ref = _capture(cams, layout, _detections(cams, _corners(), seed=1), extrinsics_sha="a" * 64)
    cur = _capture(cams, layout, _detections(cams, _corners(), seed=2), extrinsics_sha="b" * 64)
    with pytest.raises(ValueError, match="Take a new reference"):
        rc.compare(cams, ref, cur, layout)


def test_a_layout_cannot_put_one_tag_in_two_groups(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"dictionary": "DICT_6X6_50", "groups": {"a": {"ids": [1, 2]}, "b": {"ids": [2]}}})
    )
    with pytest.raises(SystemExit, match="tag 2"):
        rc.load_layout(path)


def test_cameras_load_from_a_copied_run_whose_paths_point_at_thor(tmp_path) -> None:
    intr_dir = tmp_path / "calib_x_intrinsics"
    (intr_dir / "converted" / "cam_07_SN").mkdir(parents=True)
    (intr_dir / "converted" / "cam_07_SN" / "intrinsics_producer.json").write_text(
        json.dumps({"model": "opencv_fisheye", "image_width": "1920", "image_height": "1080",
                    "camera_matrix": json.dumps(K.tolist()), "dist_coeffs": json.dumps([D.tolist()])})
    )  # fmt: skip
    thor_path = "/home/nvidia/lerobot/outputs/calibration/calib_x_intrinsics/converted/cam_07_SN/intrinsics_producer.json"
    (intr_dir / "summary.json").write_text(
        json.dumps({"cameras": [{"camera_name": "cam_07", "status": "ok", "intrinsics_json": thor_path},
                                {"camera_name": "cam_02", "status": "failed", "intrinsics_json": "x"}]})
    )  # fmt: skip
    (tmp_path / "calib_x_extrinsics").mkdir()
    T = _cameras()["cam_07"]["T_world_camera"]
    (tmp_path / "calib_x_extrinsics" / "summary.json").write_text(
        json.dumps({"world": {"world_frame_id": "world_x"},
                    "joint_solution": {"cameras": {"cam_07": {"status": "ok", "base_to_camera": {"matrix_4x4": T.tolist()}}}}})
    )  # fmt: skip
    cams, identity = rc.load_cameras(tmp_path, "calib_x")
    assert list(cams) == ["cam_07"] and identity["world_frame_id"] == "world_x"
    assert np.allclose(cams["cam_07"]["K"], K) and np.allclose(cams["cam_07"]["T_world_camera"], T)
    assert identity["extrinsics_run"] == "calib_x_extrinsics" and len(identity["extrinsics_sha256"]) == 64


def test_detection_averages_the_frames_and_keeps_only_layout_tags(tmp_path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    frame = np.full((540, 960), 255, np.uint8)
    placed = {}
    for tag, (x, y) in {40: (200, 150), 7: (600, 300)}.items():  # 7 is the rig's, not a cart tag
        side = 160
        img = cv2.aruco.generateImageMarker(dictionary, tag, side)
        frame[y : y + side, x : x + side] = img
        placed[tag] = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], float) - 0.5
    video = tmp_path / "cam_07.mkv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 30, (960, 540))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1/mkv")
    for _ in range(20):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    det = rc.detect_video(video, "DICT_6X6_50", {40, 41}, stride=2)
    assert det["sampled"] == 10 and det["size"] == (960, 540)
    assert list(det["tags"]) == [40]
    assert det["tags"][40]["frames"] == 10
    assert np.abs(det["tags"][40]["corners_px"] - placed[40]).max() < 1.0


def _frozen_datum(tmp_path, cams, layout) -> Path:
    m = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    m.update(dictionary="DICT_6X6_50")
    m["calibration"]["world_frame_id"] = "world_x"
    path = tmp_path / "table_datum.json"
    path.write_text(json.dumps(rc.freeze_datum(m, layout, "synthetic")))
    return path


def _recorded_session(tmp_path, episodes: dict[str, str]) -> Path:
    """A recorded dataset: one dir per episode, a stub video per camera, meta.json naming its calibration."""
    root = tmp_path / "thor_gmsl2_11ch_v1_20261028_090000"
    for name, run in episodes.items():
        ep = root / "episodes" / name
        ep.mkdir(parents=True)
        (ep / "meta.json").write_text(json.dumps({"world_frame": {"calibration_id": run}}))
        for cam in _cameras():
            (ep / f"{cam}.mkv").write_bytes(b"0")
    return root


def test_a_session_is_checked_episode_by_episode_and_only_new_videos_are_read(tmp_path, monkeypatch) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _frozen_datum(tmp_path, cams, layout)
    seen = {
        "episode_000000": _detections(cams, _corners(), seed=2),
        "episode_000001": _detections(cams, _corners(), true_poses=_turned(cams, "cam_09", 0.1), seed=3),
    }
    reads = []

    def fake_detect(episode, cameras, dictionary, ids, *, stride, max_samples):
        reads.append(episode.name)
        return {n: d for n, d in seen[episode.name].items() if n in cameras}

    monkeypatch.setattr(rc, "detect_episode", fake_detect)
    dataset = _recorded_session(tmp_path, {"episode_000000": "calib_x", "episode_000001": "calib_x"})
    argv = [
        "--layout",
        str(tmp_path / "cart_tags.json"),
        "session",
        str(dataset),
        "--datum",
        str(datum),
        "--write",
    ]
    assert rc.main(argv) == 1  # a camera moved
    s = json.loads((dataset / rc.SESSION_CHECK_PATH).read_text())
    assert s["episode_verdicts"] == {"episode_000000": "ok", "episode_000001": "cameras_moved"}
    assert s["verdict"] == "cameras_moved" and s["cameras_moved"] == ["cam_09"]
    assert s["worst_shift_px"]["cam_09"] > 0.6 > s["worst_shift_px"]["cam_07"]

    # Again: nothing decoded.  The calibration stamp is read again though -- an episode recorded on other
    # extrinsics says nothing about whether its labels fit, however still its cameras were.
    (dataset / "episodes" / "episode_000000" / "meta.json").write_text(
        json.dumps({"world_frame": {"calibration_id": "calib_y"}})
    )
    assert rc.main(argv) == 1 and reads == ["episode_000000", "episode_000001"]
    s = json.loads((dataset / rc.SESSION_CHECK_PATH).read_text())
    assert s["episode_verdicts"]["episode_000000"] == "other_calibration"
    assert s["episodes_detail"]["episode_000000"]["pixel_verdict"] == "ok"
    assert s["extrinsics_runs"] == ["calib_x", "calib_y"]

    # A re-recorded episode (its videos changed) is read again.
    seen["episode_000001"] = _detections(cams, _corners(), seed=4)
    (dataset / "episodes" / "episode_000001" / "cam_09.mkv").write_bytes(b"00")
    assert rc.main(argv) == 0 and reads[-1] == "episode_000001" and len(reads) == 3
    s = json.loads((dataset / rc.SESSION_CHECK_PATH).read_text())
    assert s["verdict"] == "other_calibration" and s["cameras_moved"] == []


def test_a_session_whose_cameras_see_no_table_tag_is_unchecked_not_ok(tmp_path, monkeypatch) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _frozen_datum(tmp_path, cams, layout)
    monkeypatch.setattr(rc, "detect_episode", lambda *a, **k: {n: {"tags": {}} for n in a[1]})
    dataset = _recorded_session(tmp_path, {"episode_000000": "calib_x"})
    s = rc.check_session(dataset, datum, layout)
    assert s["verdict"] == "unchecked" and s["episodes_checked"] == 0
    assert s["episodes_detail"]["episode_000000"]["status"] == "missing"


def test_a_session_only_decodes_the_cameras_the_datum_has_table_pixels_for(tmp_path, monkeypatch) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum_path = _frozen_datum(tmp_path, cams, layout)
    datum = json.loads(datum_path.read_text())
    datum["camera_pixels"]["cam_14"] = {}  # like the real cam_06/08/14
    datum_path.write_text(json.dumps(datum))
    asked = []

    def fake_detect(episode, cameras, *a, **k):
        asked.extend(cameras)
        return {
            n: d for n, d in _detections(cams, _corners(), seed=2).items() if n in cameras and n != "cam_07"
        }

    monkeypatch.setattr(rc, "detect_episode", fake_detect)
    s = rc.check_session(_recorded_session(tmp_path, {"episode_000000": "calib_x"}), datum_path, layout)
    assert "cam_14" not in asked
    detail = s["episodes_detail"]["episode_000000"]
    assert detail["cameras_not_recorded"] == ["cam_07"] and "cam_14" not in detail["cameras"]
    assert s["verdict"] == "ok"


def _rigid(deg: float, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rot_z(deg), t
    return T


def _anchored_datum(tmp_path, cams, layout) -> dict:
    m = _capture(cams, layout, _detections(cams, _corners(), seed=1))
    m.update(dictionary="DICT_6X6_50")
    m["calibration"]["world_frame_id"] = "world_x"
    return json.loads(json.dumps(rc.freeze_datum(m, layout, "synthetic", cams)))


def _in_gauge(cams, T_world_session) -> dict:
    """The same cameras as a fresh bundle reports them: in its own gauge S."""
    T_sw = np.linalg.inv(T_world_session)
    return {n: dict(c, T_world_camera=T_sw @ c["T_world_camera"]) for n, c in cams.items()}


def test_the_table_places_a_bundle_in_the_world_without_the_camera_consensus(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _anchored_datum(tmp_path, cams, layout)
    assert sorted(datum["anchor_cameras"]) == sorted(cams)  # every synthetic camera sees the table
    T_ws = _rigid(30.0, [2.0, -1.0, 0.5])
    # cam_09 was bumped since the datum: its new pixels come with its new pose, so it still counts
    true = dict(cams, cam_09=dict(cams["cam_09"], T_world_camera=_turned(cams, "cam_09", 1.0)["cam_09"]))
    a = rc.table_anchor(datum, _in_gauge(true, T_ws), _detections(true, _corners(), seed=5))
    assert a["status"] == "measured" and a["tags"] == [45, 46, 47]
    # where the table is (0.03 px of noise in each of two captures, and nothing else)
    P_s = (
        np.concatenate(list(_corners().values())) @ np.linalg.inv(T_ws)[:3, :3].T + np.linalg.inv(T_ws)[:3, 3]
    )
    T = np.asarray(a["T_world_session"])
    assert np.abs(P_s @ T[:3, :3].T + T[:3, 3] - np.concatenate(list(_corners().values()))).max() < 3e-4
    # world_registration's anchor: the table centre, in W and in S
    assert np.allclose(
        np.asarray(a["T_world_session"]) @ np.asarray(a["T_session_anchor"]), a["T_world_anchor"]
    )
    assert abs(np.asarray(a["T_world_anchor"])[2, 3]) < 1e-3


def test_a_table_tag_that_moved_spoils_a_three_tag_anchor(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _anchored_datum(tmp_path, cams, layout)
    moved = _corners({47: (np.eye(3), np.array([0.02, 0.0, 0.0]))})
    a = rc.table_anchor(datum, _in_gauge(cams, np.eye(4)), _detections(cams, moved, seed=5))
    assert a["status"] == "inconsistent" and "moved" in a["why"]
    hidden = {(n, 47) for n in cams}
    a = rc.table_anchor(datum, cams, _detections(cams, _corners(), seed=5, hide=hidden))
    assert a["status"] == "insufficient"


def test_a_datum_without_its_calibration_cannot_anchor(tmp_path) -> None:
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _anchored_datum(tmp_path, cams, layout)
    del datum["anchor_cameras"]
    assert rc.table_anchor(datum, cams, _detections(cams, _corners()))["status"] == "no_datum_calibration"


def test_the_anchor_reads_the_bundle_in_its_own_gauge(tmp_path) -> None:
    intr = tmp_path / "intrinsics_by_serial" / "converted" / "cam_07_SN"
    intr.mkdir(parents=True)
    (intr / "intrinsics_producer.json").write_text(
        json.dumps({"camera_name": "cam_07", "model": "opencv_fisheye", "image_width": 1920, "image_height": 1080,
                    "camera_matrix": K.tolist(), "dist_coeffs": [D.tolist()]})
    )  # fmt: skip
    T = _cameras()["cam_07"]["T_world_camera"]
    report = tmp_path / "extrinsics_report.json"
    report.write_text(
        json.dumps({"camera_model": "fisheye", "reference": "cam_07", "intrinsics_source": "/home/nvidia/x",
                    "T_ref_cam": {"cam_07": T.tolist(), "cam_02": T.tolist()}})
    )  # fmt: skip
    cams, bundle = rc.load_bundle_cameras(report, tmp_path / "intrinsics_by_serial")
    assert list(cams) == ["cam_07"] and np.allclose(cams["cam_07"]["T_world_camera"], T)
    assert bundle["sha256"] == rc._sha256(report) and bundle["reference"] == "cam_07"


def test_episodes_on_a_new_calibration_are_held_against_its_own_capture(tmp_path, monkeypatch) -> None:
    """After a recalibration the datum's pixels are stale for whatever moved; the run's capture is not."""
    cams, layout = _cameras(), _layout(tmp_path)
    datum = _frozen_datum(tmp_path, cams, layout)
    bumped = _turned(cams, "cam_09", 0.1)  # cam_09 moved, then the rig was recalibrated
    seen = {"episode_000000": _detections(cams, _corners(), true_poses=bumped, seed=2)}
    monkeypatch.setattr(
        rc,
        "detect_episode",
        lambda ep, cameras, *a, **k: {n: d for n, d in seen[ep.name].items() if n in cameras},
    )
    dataset = _recorded_session(tmp_path, {"episode_000000": "calib_y"})
    root = tmp_path / "calibration"

    before = rc.check_session(dataset, datum, layout, calibration_root=root)
    assert before["verdict"] == "other_calibration" and before["extrinsics_runs_unreferenced"] == ["calib_y"]
    assert before["episodes_detail"]["episode_000000"]["pixel_verdict"] == "cameras_moved"

    capture = _detections(cams, _corners(), true_poses=bumped, seed=3)
    (root / "calib_y_extrinsics").mkdir(parents=True)
    (root / "calib_y_extrinsics" / rc.ANCHOR_FILE).write_text(
        json.dumps({"schema": rc.ANCHOR_SCHEMA, "status": "insufficient", "camera_pixels": {
            n: {str(t): v["corners_px"].tolist() for t, v in d["tags"].items()} for n, d in capture.items()}})
    )  # fmt: skip
    after = rc.check_session(dataset, datum, layout, before, calibration_root=root)
    detail = after["episodes_detail"]["episode_000000"]
    assert after["verdict"] == "ok" and detail["reference"]["kind"] == "calibration_capture"
    assert after["extrinsics_runs_unreferenced"] == [] and after["worst_shift_px"]["cam_09"] < 0.6
