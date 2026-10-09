from __future__ import annotations

from pathlib import Path

import yaml

from tools.thor.prepare_p0_calibration_recorder_config import (
    build_config,
    parse_locked_ids,
    parse_sensor_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    REPO_ROOT
    / "third_party/opencv_kalibr/fr3_calibration/host/thor_gmsl2_calibration.yaml"
)


def test_selects_all_current_locks_except_requested_ids() -> None:
    locked = [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]
    payload = build_config(TEMPLATE, locked, [1, 4, 10])

    cameras = payload["sensors"]["cameras"]
    assert cameras["detect_all"] is False
    assert cameras["sensor_ids"] == [3, 6, 7, 8, 9, 12, 13, 14]
    assert payload["p0_calibration_camera_selection"]["locked_sensor_ids"] == locked


def test_parses_lock_script_output() -> None:
    assert parse_locked_ids("header\nLOCKED_VIDEO_IDS=1,3,14\n") == [1, 3, 14]


def test_sensor_ids_are_unique_and_bounded() -> None:
    assert parse_sensor_ids("10,1,10,4") == [10, 1, 4]


def test_template_remains_valid_yaml() -> None:
    assert isinstance(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")), dict)
