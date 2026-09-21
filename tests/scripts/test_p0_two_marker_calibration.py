import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import yaml

from tools.thor.p0_two_marker_calibration import (
    DEFAULT_RECORDER_CONFIG,
    _build_cpu_isolation_plan,
    _camera_id,
    _excluded_sensor_ids,
    _locked_sensor_ids,
    _make_recorder_config,
    _irq_cpu_from_interrupts,
    _load_resume_records,
    _normalize_camera_name,
    _operator_key,
    _parse_camera_aliases,
    _resolve_resume_run,
    _resume_counts,
    _run_robot_only_control_test,
    _select_sensor_ids,
)


def _write_resume_capture(run_dir: Path, capture_index: int = 0) -> None:
    image_dir = run_dir / "captures" / f"capture_{capture_index:03d}"
    image_dir.mkdir(parents=True)
    cameras = {}
    for camera, valid in (("cam_06", True), ("cam_07", False)):
        image_path = image_dir / f"{camera}.jpg"
        image_path.write_bytes(b"jpeg-placeholder")
        cameras[camera] = {
            "calibration_camera": camera,
            "image": str(image_path),
            "valid": valid,
            "detections": (
                [
                    {
                        "tag_id": 6,
                        "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "image_width": 10,
                        "image_height": 10,
                    }
                ]
                if valid
                else []
            ),
            "error": "",
        }
    payload = {
        "schema": "fr3_base_single_tag_camera_captures/v1",
        "marker": {"family": "tag36h11", "id": 6, "marker_size_m": 0.16},
        "records": [
            {
                "capture_index": capture_index,
                "created_utc": "2026-09-21T00:00:00+00:00",
                "sync": {},
                "T_base_tcp": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "joint_values_rad": [0.0] * 7,
                "cameras": cameras,
            }
        ],
    }
    (run_dir / "captures.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resume_loads_committed_captures_and_maps_counts_by_physical_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    run_dir = root / "manual_run_20260921T000000Z"
    _write_resume_capture(run_dir, capture_index=4)

    assert _resolve_resume_run(root, "latest") == run_dir.resolve()
    records = _load_resume_records(run_dir)
    assert [record["capture_index"] for record in records] == [4]
    assert _resume_counts(
        records,
        {"cam_05": "cam_06", "cam_07": "cam_07"},
    ) == {"cam_05": 1, "cam_07": 0}


def test_resume_rejects_changed_camera_identity_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "manual_run_20260921T000000Z"
    _write_resume_capture(run_dir)
    records = _load_resume_records(run_dir)
    try:
        _resume_counts(records, {"cam_06": "cam_06", "cam_08": "cam_08"})
    except ValueError as exc:
        assert "camera identities differ" in str(exc)
    else:
        raise AssertionError("expected changed resume camera identities to fail")


def test_cpu_isolation_reserves_robot_irq_and_neighbor_for_control() -> None:
    interrupts = """           CPU0       CPU1       CPU2       CPU3
 42:          0          0       8000          0 ITS-MSI enP2p1s0-0
"""
    irq_cpu = _irq_cpu_from_interrupts(interrupts, "enP2p1s0", {0, 1, 2, 3})
    assert irq_cpu == 2
    plan = _build_cpu_isolation_plan(
        {0, 1, 2, 3},
        interface="enP2p1s0",
        irq_cpu=irq_cpu,
        requested_control_cpu=None,
        disabled=False,
    )
    assert plan.robot_irq_cpu == 2
    assert plan.robot_control_cpu == 3
    assert plan.vision_cpus == (0, 1)


def test_cpu_isolation_honors_explicit_control_cpu() -> None:
    plan = _build_cpu_isolation_plan(
        {0, 1, 2, 3},
        interface="enP2p1s0",
        irq_cpu=2,
        requested_control_cpu=1,
        disabled=False,
    )
    assert plan.robot_control_cpu == 1
    assert plan.vision_cpus == (0, 3)


def test_camera_aliases_accept_ids_and_reject_duplicate_calibrated_identity() -> None:
    assert _normalize_camera_name("5") == "cam_05"
    assert _normalize_camera_name("cam_6") == "cam_06"
    assert _parse_camera_aliases(["5=6", "cam_08=cam_09"]) == {
        "cam_05": "cam_06",
        "cam_08": "cam_09",
    }

    try:
        _parse_camera_aliases(["5=6", "7=6"])
    except ValueError as exc:
        assert "same calibrated camera" in str(exc)
    else:
        raise AssertionError("expected duplicate calibrated camera aliases to fail")


def test_standalone_recorder_config_detects_current_locked_camera_ids(tmp_path: Path) -> None:
    config_path = _make_recorder_config(
        DEFAULT_RECORDER_CONFIG,
        tmp_path / "run",
        Path("/dev/shm/p0_two_marker_test"),
        6,
    )
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cameras = payload["sensors"]["cameras"]
        assert cameras["detect_all"] is True
        assert cameras["sensor_ids"] == []
        assert cameras["online_sync"]["frame_bus_every_n"] == 6
    finally:
        shutil.rmtree(config_path.parent)


def test_explicit_sensor_ids_and_operator_keys(tmp_path: Path) -> None:
    config_path = _make_recorder_config(
        DEFAULT_RECORDER_CONFIG,
        tmp_path / "run",
        Path("/dev/shm/p0_two_marker_test"),
        6,
        [3, 6, 7],
    )
    try:
        cameras = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sensors"]["cameras"]
        assert cameras["detect_all"] is False
        assert cameras["sensor_ids"] == [3, 6, 7]
    finally:
        shutil.rmtree(config_path.parent)

    assert _camera_id("cam_02") == 2
    assert _camera_id("7") == 7
    assert _excluded_sensor_ids([]) == [2]
    assert _excluded_sensor_ids(["cam_07", "2"]) == [2, 7]
    assert _operator_key("Return", "\r") == 13
    assert _operator_key("KP_Enter", "") == 13
    assert _operator_key("Escape", "") == 27
    assert _operator_key("q", "q") == ord("q")
    assert _operator_key("x", "x") is None


def test_locked_sensor_ids_parses_lock_check_output(tmp_path: Path) -> None:
    script = tmp_path / "tools/thor/gmsl2/check_max96726_locks.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")

    def runner(cmd, **kwargs):
        assert cmd == [str(script)]
        assert kwargs["timeout"] == 30
        return SimpleNamespace(returncode=0, stdout="LOCKED_VIDEO_IDS=2,3,6,14\n", stderr="")

    assert _locked_sensor_ids(tmp_path, _runner=runner) == [2, 3, 6, 14]


def test_selects_locked_cameras_except_explicit_exclusions() -> None:
    selected, ignored = _select_sensor_ids(
        [2, 3, 5, 6, 7],
        [2],
    )
    assert selected == [3, 5, 6, 7]
    assert ignored == {"cam_02": "excluded by P0 policy"}


def test_robot_only_control_test_reads_controller_health() -> None:
    class Panda:
        def __init__(self):
            self.error_checks = 0

        def raise_error(self):
            self.error_checks += 1

        def get_state(self):
            return SimpleNamespace(
                control_command_success_rate=0.999,
                robot_mode=SimpleNamespace(name="kMove"),
            )

    panda = Panda()
    robot = SimpleNamespace(_arm=SimpleNamespace(_robot=panda))
    _run_robot_only_control_test(robot, 0.03)
    assert panda.error_checks >= 2
