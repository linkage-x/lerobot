"""Real replay holds at the trajectory start until the gateway says go."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pinocchio")

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py"
)
_spec = importlib.util.spec_from_file_location("replay_cube_pose_real", _SCRIPT)
replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = replay
_spec.loader.exec_module(replay)


def _wait_with_stdin(monkeypatch, text: str | None, timeout_s: float = 5.0) -> str:
    read_fd, write_fd = os.pipe()
    if text is not None:
        os.write(write_fd, text.encode())
    os.close(write_fd)
    stdin = os.fdopen(read_fd, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    cfg = replay.ReplayRuntimeConfig(hold_at_trajectory_start=True, trajectory_start_timeout_s=timeout_s)
    try:
        return replay._wait_at_trajectory_start({"reached": True, "final_position_error_m": 0.003}, cfg)
    finally:
        stdin.close()


def test_go_releases_the_trajectory(monkeypatch, capsys):
    assert _wait_with_stdin(monkeypatch, "go\n") == "go"
    assert "REPLAY_AT_TRAJECTORY_START reached=True position_error_mm=3.00" in capsys.readouterr().out


def test_anything_but_go_keeps_the_arm_at_the_start(monkeypatch):
    assert _wait_with_stdin(monkeypatch, "abort\n") == "abort"
    assert _wait_with_stdin(monkeypatch, "") == "stdin_closed"
    # Unknown lines are ignored, not taken as go.
    assert _wait_with_stdin(monkeypatch, "start\n") == "stdin_closed"


def test_the_hold_times_out(monkeypatch):
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    cfg = replay.ReplayRuntimeConfig(hold_at_trajectory_start=True, trajectory_start_timeout_s=0.2)
    try:
        assert replay._wait_at_trajectory_start({"reached": True}, cfg) == "timeout"
    finally:
        os.close(write_fd)
        stdin.close()
