"""The side camera's tape check: the fit, the verdicts, and the gateway's keeping of them."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.data_collection_gui import gateway  # noqa: E402
from tools.fr3 import side_cam_check  # noqa: E402

TABLE_BGR = (118, 124, 94)  # the teal mat, as the side camera sees it
TAPE_BGR = (90, 170, 230)  # yellow tape, H ~18
GEAR_BGR = (20, 60, 230)  # orange-red gear, H ~6


def scene(dx: float = 0.0, dy: float = 0.0, deg: float = 0.0, *, cover_side: bool = False, gear_shift=(0, 0)):
    """A 640x480 still: the tape square seen at an angle round a gear, moved as a camera move would."""

    image = np.zeros((480, 640, 3), np.uint8)
    image[:] = TABLE_BGR
    square = np.array([[340, 190], [425, 265], [330, 335], [255, 260]], np.float64)
    cv2.polylines(image, [square.astype(np.int32)], True, TAPE_BGR, 9)
    cv2.circle(image, (335 + gear_shift[0], 262 + gear_shift[1]), 50, GEAR_BGR, -1)
    if cover_side:  # the arm or the peg across one side
        cv2.rectangle(image, (290, 290), (345, 345), (40, 40, 40), -1)
    matrix = cv2.getRotationMatrix2D((320, 240), deg, 1.0)
    matrix[:, 2] += [dx, dy]
    return cv2.warpAffine(image, matrix, (640, 480), borderMode=cv2.BORDER_REPLICATE)


def test_the_tape_mask_keeps_the_lines_and_drops_the_gear():
    mask = side_cam_check.tape_mask(scene())
    assert mask[262, 335] == 0  # gear centre
    assert mask[228, 383] > 0 or mask[227, 382] > 0  # on the upper-right side, between corners
    assert (mask > 0).sum() > 1000


def test_an_unmoved_camera_is_ok_even_with_a_side_covered_and_the_gear_moved():
    reference = scene()
    for current in (scene(), scene(cover_side=True), scene(gear_shift=(6, -4))):
        result, overlay = side_cam_check.check_images(reference, current)
        assert result["verdict"] == "ok", result
        assert result["pose"]["shiftPx"] <= 1.0
        assert overlay.shape == reference.shape


@pytest.mark.parametrize(
    ("dx", "dy", "deg", "expected"),
    [(0, -5, 0.0, "shifted"), (12, 6, 0.0, "moved"), (-15, 10, -2.0, "moved"), (0, 0, 2.5, "moved")],
)
def test_a_moved_camera_is_reported_with_the_shift_it_made(dx, dy, deg, expected):
    result, _ = side_cam_check.check_images(scene(), scene(dx, dy, deg))
    assert result["verdict"] == expected, result
    pose = result["pose"]
    if deg == 0.0:
        assert pose["dxPx"] == pytest.approx(dx, abs=1.0)
        assert pose["dyPx"] == pytest.approx(dy, abs=1.0)
    assert pose["rotDeg"] == pytest.approx(deg, abs=0.5)


def test_no_tape_in_view_is_unknown_not_ok():
    blank = scene()
    blank[150:360, 220:460] = TABLE_BGR
    result, _ = side_cam_check.check_images(scene(), blank)
    assert result["verdict"] == "unknown"


def test_a_preview_at_another_size_is_compared_in_reference_pixels():
    small = cv2.resize(scene(0, -5), (320, 240), interpolation=cv2.INTER_AREA)
    result, _ = side_cam_check.check_images(scene(), small)
    assert result["verdict"] == "shifted"
    assert result["pose"]["dyPx"] == pytest.approx(-5, abs=1.5)


def test_an_unsettled_exposure_is_unknown_and_says_so_not_tape_covered():
    # 11-13: the first frame of a freshly opened side camera, B34 G54 R4 against ~110 settled.
    dark = (scene().astype(np.float64) * np.array([0.3, 0.42, 0.04])).astype(np.uint8)
    result, _ = side_cam_check.check_images(scene(), dark)
    assert result["verdict"] == "unknown"
    assert "曝光" in result["reason"] and "遮挡" not in result["reason"]
    assert min(result["exposureRatios"]) < side_cam_check.EXPOSURE_MIN_RATIO


def test_a_settled_exposure_a_little_off_the_reference_is_still_checked():
    brighter = np.clip(scene().astype(np.float64) * 1.12, 0, 255).astype(np.uint8)
    result, _ = side_cam_check.check_images(scene(), brighter)
    assert result["verdict"] == "ok", result


def _preview_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    import importlib  # noqa: PLC0415

    from tools.fr3 import fr3_realsense_preview  # noqa: PLC0415

    return importlib.reload(fr3_realsense_preview)


def test_the_preview_holds_frames_back_until_the_exposure_stops_changing(monkeypatch):
    preview = _preview_module(monkeypatch)
    settle = preview.ExposureSettle(min_s=0.5, max_s=3.0)
    frame = np.full((48, 64, 3), 110, np.uint8)
    t = 0.0
    for level in (5, 20, 45, 70, 90, 100, 106, 109):  # auto exposure ramping up, 30 fps
        assert not settle.update(np.full_like(frame, level), t)
        t += 1 / 30
    while t < 0.5:  # stable, but not yet min_s
        assert not settle.update(frame, t)
        t += 1 / 30
    released = None
    for i in range(preview.SETTLE_WINDOW + 1):
        if settle.update(frame, t):
            released = i
            break
        t += 1 / 30
    assert released is not None
    assert settle.update(np.full_like(frame, 3), t + 0.1)  # once settled, stays settled


def test_the_preview_gives_up_waiting_at_max_s(monkeypatch):
    preview = _preview_module(monkeypatch)
    settle = preview.ExposureSettle(min_s=0.5, max_s=3.0)
    t, level = 0.0, 10
    while t < 3.0:
        assert not settle.update(np.full((48, 64, 3), level, np.uint8), t)
        t += 1 / 30
        level = 10 + (level + 7) % 200  # never stable
    assert settle.update(np.full((48, 64, 3), level, np.uint8), 3.0)


def _jpeg(image) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return data.tobytes()


def test_the_gateway_checks_against_the_default_reference_and_keeps_every_result(tmp_path: Path, monkeypatch):
    default = tmp_path / gateway._SIDE_CAM_DEFAULT_REFERENCE
    default.parent.mkdir(parents=True)
    cv2.imwrite(str(default), scene())
    logs: list[tuple[str, str]] = []
    state = SimpleNamespace(repo_root=tmp_path, log=lambda level, text: logs.append((level, text)))
    stills = iter([_jpeg(scene()), _jpeg(scene(12, 6))])
    monkeypatch.setattr(gateway, "_camera_still_bytes", lambda _state, _key: (next(stills), 200, ""))

    status = gateway._side_cam_check_status(state, "side")
    assert status["reference"]["kind"] == "default"
    assert status["last"] is None

    first = gateway._run_side_cam_check(state, {"camera": "side"})
    assert first["result"]["verdict"] == "ok"
    second = gateway._run_side_cam_check(state, {"camera": "side"})
    assert second["result"]["verdict"] == "moved"
    assert second["last"]["verdict"] == "moved"
    assert [row["verdict"] for row in second["history"]] == ["ok", "moved"]
    folder = tmp_path / gateway._SIDE_CAM_CHECK_SUBDIR / "side"
    assert (folder / "last_overlay.jpg").is_file()
    assert logs[-1][0] == "warn"


def test_another_camera_has_no_default_reference(tmp_path: Path, monkeypatch):
    state = SimpleNamespace(repo_root=tmp_path, log=lambda *_: None)
    monkeypatch.setattr(gateway, "_camera_still_bytes", lambda _state, _key: (_jpeg(scene()), 200, ""))
    assert gateway._side_cam_check_status(state, "ee")["reference"] is None
    with pytest.raises(ValueError, match="基准"):
        gateway._run_side_cam_check(state, {"camera": "ee"})


def test_replacing_the_reference_keeps_the_old_one_and_refuses_a_still_without_tape(tmp_path: Path, monkeypatch):
    state = SimpleNamespace(repo_root=tmp_path, log=lambda *_: None)
    blank = scene()
    blank[150:360, 220:460] = TABLE_BGR
    stills = iter([_jpeg(blank), _jpeg(scene()), _jpeg(scene(0, -5)), _jpeg(scene(0, -5))])
    monkeypatch.setattr(gateway, "_camera_still_bytes", lambda _state, _key: (next(stills), 200, ""))

    with pytest.raises(ValueError, match="胶带像素"):
        gateway._set_side_cam_reference(state, {"camera": "side"})
    first = gateway._set_side_cam_reference(state, {"camera": "side"})
    assert first["reference"]["kind"] == "saved"
    gateway._set_side_cam_reference(state, {"camera": "side"})
    folder = tmp_path / gateway._SIDE_CAM_CHECK_SUBDIR / "side"
    assert len(list(folder.glob("reference_until_*.jpg"))) == 1
    assert json.loads((folder / "reference.json").read_text(encoding="utf-8"))["source"] == "captured"
    # Checked against the new reference, the shifted still is where the reference now is.
    assert gateway._run_side_cam_check(state, {"camera": "side"})["result"]["verdict"] == "ok"
