"""camera_motion: which cameras moved since their extrinsics, and which ones the tracking leaves out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.thor.gmsl2 import camera_motion as cm

cv2 = pytest.importorskip("cv2")
RUN = "calib_20261102_195715_extrinsics"
CAMERAS = ["cam_03", "cam_06", "cam_07", "cam_08", "cam_09", "cam_12", "cam_13", "cam_14"]


def _texture(seed: int = 0) -> np.ndarray:
    """A room-like background: blobs and edges SIFT can hold on to."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (480, 640)).astype(np.uint8), (0, 0), 2.0)
    for _ in range(60):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 440))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))), int(rng.integers(0, 255)), -1)
    return img


def _shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _dataset(root: Path, name: str, state: dict | None, episodes: int = 2) -> Path:
    ds = root / name
    for e in range(episodes):
        ep = ds / "episodes" / f"episode_{e:06d}"
        ep.mkdir(parents=True)
        for cam in CAMERAS:
            (ep / f"{cam}.mkv").write_bytes(b"")
    (ds / "meta").mkdir(parents=True)
    info = {"derived": {cm.SIDECAR: {"camera_state": state}}} if state is not None else {}
    (ds / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return ds


def test_background_shift_reads_a_two_pixel_move_and_a_still_camera():
    ref = _texture()
    still = cm.background_shift(ref, ref.copy())
    moved = cm.background_shift(ref, _shifted(ref, 0.0, -2.4))  # cam_14, 11-13
    assert still["inliers"] >= cm.MIN_INLIERS and still["shift_px"] < 0.1
    assert moved["shift_px"] == pytest.approx(2.4, abs=0.15)
    assert moved["shift_vector_px"][1] == pytest.approx(-2.4, abs=0.15)
    assert cm.verdict_of(still) == "stable" and cm.verdict_of(moved) == "moved"
    blank = cm.background_shift(ref, np.full_like(ref, 128))  # scene gone: nothing to match
    assert cm.verdict_of(blank) == "unverified"


def test_the_selection_leaves_out_moved_and_unverified_cameras_only_when_something_moved():
    stable = dict.fromkeys(CAMERAS, "stable")
    assert cm.select(stable) == ([], "none_moved")
    # an unverified camera alone does not trigger anything: every ordinary recording stays as it was
    assert cm.select({**stable, "cam_14": "unverified"}) == ([], "none_moved")
    nov13 = {**stable, "cam_03": "moved", "cam_13": "moved", "cam_14": "moved"}
    assert cm.select(nov13) == (["cam_03", "cam_13", "cam_14"], "excluded")
    # moved, and one more that cannot be shown to be in place: that one goes too
    assert cm.select({**nov13, "cam_12": "unverified"}) == (["cam_03", "cam_12", "cam_13", "cam_14"], "excluded")
    # too few verified cameras to track on: keep them all and let the camera state call it stale
    assert cm.select({**nov13, "cam_12": "moved", "cam_09": "unverified"}) == ([], "too_few_stable")


def test_the_reference_is_the_latest_earlier_full_camera_ok_recording_on_the_same_extrinsics(tmp_path):
    ok = {"verdict": "ok", "extrinsics_run_name": RUN}
    _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261112_175344", ok)
    best = _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261112_184911", ok)
    _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261112_190000", {"verdict": "ok", "extrinsics_run_name": "calib_older"})
    _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261113_155317", {"verdict": "stale", "extrinsics_run_name": RUN})
    # already tracked without its moved cameras: ok, but it saw the moved cameras where they are now
    _dataset(
        tmp_path,
        "thor_gmsl2_9ch_v1_20261113_183417",
        {**ok, "excluded_cameras": ["cam_03", "cam_13", "cam_14"]},
    )
    _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261117_090000", ok)  # later than the target
    target = _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261116_110051", None)
    assert cm.find_reference(target, RUN) == best
    assert cm.find_reference(target, "calib_newer") is None


def test_check_writes_the_selection_with_its_evidence(tmp_path, monkeypatch):
    ok = {"verdict": "ok", "extrinsics_run_name": RUN}
    ref = _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261112_184911", ok)
    target = _dataset(tmp_path, "thor_gmsl2_9ch_v1_20261113_183417", None)
    moved = {"cam_03": (-1.2, 1.1), "cam_13": (0.0, -1.4), "cam_14": (0.0, -2.3)}
    backgrounds = {cam: _texture(i) for i, cam in enumerate(CAMERAS)}

    def fake_median(video: Path, samples: int = cm.SAMPLES):
        cam = video.stem
        img = backgrounds[cam]
        if ref in video.parents:
            return img
        return _shifted(img, *moved.get(cam, (0.0, 0.0)))

    monkeypatch.setattr(cm, "median_image", fake_median)
    record = cm.check(target, RUN, CAMERAS)
    assert record["reference"]["dataset"] == str(ref) and record["reference"]["episode"] == "episode_000001"
    assert record["moved"] == ["cam_03", "cam_13", "cam_14"]
    assert record["excluded_cameras"] == ["cam_03", "cam_13", "cam_14"] and record["decision"] == "excluded"
    assert set(record["cameras"]["cam_14"]["by_episode"]) == {"episode_000000", "episode_000001"}
    assert "tracking without cam_03, cam_13, cam_14" in record["message"]
    json.dumps(record, allow_nan=False)

    nothing = cm.check(_dataset(tmp_path / "alone", "thor_gmsl2_9ch_v1_20261113_183417", None), RUN, CAMERAS)
    assert nothing["decision"] == "no_reference" and nothing["excluded_cameras"] == []


def test_the_cli_reads_the_cameras_and_run_from_the_tracking_config(tmp_path, monkeypatch):
    calib = tmp_path / "calibration" / RUN
    calib.mkdir(parents=True)
    (calib / "summary.json").write_text(
        json.dumps({"joint_solution": {"cameras": {c: {} for c in CAMERAS}}}), encoding="utf-8"
    )
    config = tmp_path / "tracking.yaml"
    config.write_text(
        f"calibration:\n  root_dir: {tmp_path / 'calibration'}\n  fixed_camera_run_name: {RUN}\n", encoding="utf-8"
    )
    target = _dataset(tmp_path / "datasets", "thor_gmsl2_9ch_v1_20261113_183417", None)
    seen = {}

    def fake_check(dataset, run, cameras, **_kw):
        seen.update(dataset=dataset, run=run, cameras=cameras)
        return {"cameras": {}, "excluded_cameras": ["cam_14"], "moved": ["cam_14"], "message": "m"}

    monkeypatch.setattr(cm, "check", fake_check)
    assert cm.main([str(target), "--config", str(config), "--write"]) == 0
    assert seen == {"dataset": target, "run": RUN, "cameras": CAMERAS}
    written = json.loads((target / cm.OUTPUT).read_text(encoding="utf-8"))
    assert written["excluded_cameras"] == ["cam_14"]
