from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools/thor/run_p0_two_marker_calibration_local.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(REPO_ROOT / ".venv/bin/python")
    env["P0_LOCKED_CAMERA_IDS"] = "1,3,4,6,7,8,9,10,12,13,14"
    env["DISPLAY"] = ":99"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_teaching_dry_run_uses_current_recorder_and_key_output() -> None:
    result = _run("teaching", "--key", "p0_test", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "teaching_pose_recorder.py" in result.stdout
    assert "outputs/datasets/p0_test/teaching_pose_records.json" in result.stdout
    assert "p0_two_marker_calibration.py" not in result.stdout


def test_capture_dry_run_uses_thor_batch_capture_and_forwards_limit() -> None:
    result = _run("capture", "--key=p0_test", "--max-records", "17", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "batch_execute_pose_and_capture_thor_gmsl2.py" in result.stdout
    assert "--execution.max_records=17" in result.stdout
    assert "fr3_execute_pose_thor_gmsl2_apriltag_p0_test_merged" in result.stdout
    assert "selected=[3, 6, 7, 8, 9, 12, 13, 14]" in result.stdout
    assert "DISPLAY/WAYLAND_DISPLAY cleared" in result.stdout
    assert "--thor.recorder_config_path=" in result.stdout


def test_capture_accepts_standalone_calibration_captures_json() -> None:
    source = "outputs/calibration/p0_single_tag_camera_calibration/manual_run_example/captures.json"
    result = _run(
        "capture",
        "--key=p0_replay",
        "--input-json",
        source,
        "--max-records=all",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"--input.json_path={REPO_ROOT / source}" in result.stdout
    assert "--execution.max_records" not in result.stdout


def test_hardware_run_requires_explicit_authorization() -> None:
    result = _run("capture", "--key", "p0_test")

    assert result.returncode == 2
    assert "requires --execute" in result.stderr


def test_rejects_non_dataset_key() -> None:
    result = _run("teaching", "--key", "../escape", "--dry-run")

    assert result.returncode == 2
    assert "simple dataset name" in result.stderr
