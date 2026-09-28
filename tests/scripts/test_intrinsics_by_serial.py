"""Intrinsics follow the module serial, not the port."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.thor.gmsl2 import camera_eeprom as ce, intrinsics_by_serial as ibs


def _producer(run: Path, camera: str, serial: str, fx: float, cx: float, cy: float, model="opencv_fisheye") -> Path:
    path = run / "converted" / f"{camera}_{serial}" / "intrinsics_producer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "camera_name": camera, "camera_serial": serial, "model": model,
        "camera_matrix": [[fx, 0, cx], [0, fx, cy], [0, 0, 1]], "dist_coeffs": [[0.1, 0.0, 0.0, 0.0]],
    }))
    return path


def _eeprom_run(root: Path, name: str, stamp: str, cameras: dict[str, str]) -> Path:
    run = root / name
    rows = []
    for camera, serial in cameras.items():
        path = _producer(run, camera, serial, 1000.0, 960.0, 540.0)
        rows.append({"camera_name": camera, "camera_serial": serial, "status": "ok", "intrinsics_json": str(path)})
    (run / "summary.json").write_text(json.dumps(
        {"timestamp_utc": stamp, "serial_source": "eeprom", "cameras": rows}))
    return run


def _episodes(root: Path, identities: list[dict[str, str | None]]) -> Path:
    for i, identity in enumerate(identities):
        episode = root / "episodes" / f"episode_{i:06d}"
        episode.mkdir(parents=True)
        (episode / "meta.json").write_text(json.dumps({"camera_identity": {
            cam: {"serial": serial, "answered": serial is not None} for cam, serial in identity.items()
        }}))
    return root / "episodes"


def test_port_serials_come_from_the_capture_and_a_mid_capture_swap_is_refused(tmp_path):
    episodes = _episodes(tmp_path / "a", [{"cam_07": "SN-B", "cam_09": None}, {"cam_07": "SN-B"}])
    assert ibs.capture_port_serials(episodes) == {"cam_07": "SN-B"}

    swapped = _episodes(tmp_path / "b", [{"cam_07": "SN-B"}, {"cam_07": "SN-C"}])
    with pytest.raises(ValueError, match="cam_07"):
        ibs.capture_port_serials(swapped)

    assert ibs.capture_port_serials(_episodes(tmp_path / "c", [{}])) == {}


def test_a_moved_module_takes_its_own_lens_to_the_new_port(tmp_path):
    calib = tmp_path / "calibration"
    _eeprom_run(calib, "old_intrinsics", "2026-09-01T00:00:00Z", {"cam_07": "SN-B", "cam_09": "SN-C"})
    # SN-B was later re-fitted on port 14: the newer run wins for SN-B only
    newer = _eeprom_run(calib, "new_intrinsics", "2026-09-20T00:00:00Z", {"cam_14": "SN-B"})

    sources, missing = ibs.resolve_sources(["SN-B", "SN-C", "SN-X"], calib)

    assert missing == ["SN-X"]
    assert sources["SN-B"].run == "new_intrinsics" and sources["SN-B"].camera == "cam_14"
    assert sources["SN-C"].run == "old_intrinsics"

    staged = tmp_path / "staged"
    summary = ibs.stage_run({"cam_09": "SN-B", "cam_07": "SN-C", "cam_04": "SN-X"}, sources, staged)

    assert summary["serial_source"] == "eeprom"
    assert [row["camera_name"] for row in summary["cameras"]] == ["cam_07", "cam_09"]
    moved = json.loads((staged / "converted" / "cam_09_SN-B" / "intrinsics_producer.json").read_text())
    original = json.loads((newer / "converted" / "cam_14_SN-B" / "intrinsics_producer.json").read_text())
    assert moved["camera_name"] == "cam_09" and moved["camera_serial"] == "SN-B"
    assert moved["camera_matrix"] == original["camera_matrix"]
    assert moved["intrinsics_origin"]["run"] == "new_intrinsics"
    assert moved["intrinsics_origin"]["camera"] == "cam_14"
    # and the staged run is itself a source for the next solve
    assert ibs.eeprom_run_sources(tmp_path)["SN-B"].run == "staged"


def test_legacy_runs_are_found_through_the_registry(tmp_path):
    calib = tmp_path / "calibration"
    _producer(calib / "legacy_intrinsics", "cam_13", "WRONG-SERIAL", 1004.0, 925.0, 531.0)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"serials": {"SN-057": {
        "run": "legacy_intrinsics", "camera": "cam_13",
        "producer": "converted/cam_13_WRONG-SERIAL/intrinsics_producer.json"}}}))

    sources, missing = ibs.resolve_sources(["SN-057"], calib, registry)

    assert missing == []
    assert sources["SN-057"].how == "registry" and sources["SN-057"].camera == "cam_13"


def test_mixed_models_cannot_be_staged_into_one_run(tmp_path):
    calib = tmp_path / "calibration"
    _eeprom_run(calib, "a", "1", {"cam_01": "SN-A"})
    _producer(calib / "b", "cam_02", "SN-B", 1000, 960, 540, model="opencv_rational")
    (calib / "b" / "summary.json").write_text(json.dumps({"timestamp_utc": "2", "serial_source": "eeprom", "cameras": [
        {"camera_name": "cam_02", "camera_serial": "SN-B", "status": "ok",
         "intrinsics_json": str(calib / "b" / "converted" / "cam_02_SN-B" / "intrinsics_producer.json")}]}))
    sources, _ = ibs.resolve_sources(["SN-A", "SN-B"], calib)
    with pytest.raises(ValueError, match="模型不一致"):
        ibs.stage_run({"cam_01": "SN-A", "cam_02": "SN-B"}, sources, tmp_path / "s")


def test_registry_ties_each_calibrated_camera_to_the_nearest_module(tmp_path):
    run = tmp_path / "run_0804"
    _producer(run, "cam_06", "WRONG1", 1001.0, 964.0, 556.0)
    _producer(run, "cam_13", "WRONG2", 1004.0, 926.0, 531.0)
    _producer(run, "cam_14", "WRONG3", 1000.0, 700.0, 300.0)  # nothing near
    factory = {
        "SN-024": {"fx": 998.9, "fy": 998.8, "cx": 964.7, "cy": 557.4},
        "SN-057": {"fx": 1004.6, "fy": 1004.6, "cx": 925.4, "cy": 530.8},
    }

    registry, notes = ce.register_intrinsics([run], factory, match_px=8.0)

    assert registry["serials"]["SN-024"]["camera"] == "cam_06"
    assert registry["serials"]["SN-057"]["producer"] == "converted/cam_13_WRONG2/intrinsics_producer.json"
    assert any("cam_14" in note for note in notes)
    assert registry["factory"] == factory
