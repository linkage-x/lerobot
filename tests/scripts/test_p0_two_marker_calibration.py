from pathlib import Path
import shutil
from types import SimpleNamespace

import yaml

from tools.thor.p0_two_marker_calibration import (
    DEFAULT_RECORDER_CONFIG,
    _camera_id,
    _excluded_sensor_ids,
    _locked_sensor_ids,
    _make_recorder_config,
    _normalize_camera_name,
    _operator_key,
    _parse_camera_aliases,
    _run_robot_only_control_test,
    _select_sensor_ids,
)


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
