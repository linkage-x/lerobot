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


def test_csv_trajectory_carries_the_recorded_gripper_opening(tmp_path):
    csv_path = tmp_path / "state_action.right.csv"
    csv_path.write_text(
        "episode_index,frame_index,gripper_width_m,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw\n"
        "0,0,0.09,0.5,0,0.3,0,0,0,1\n"
        "0,1,0.0323,0.5,0,0.3,0,0,0,1\n"
        "0,2,nan,0.5,0,0.3,0,0,0,1\n",
        encoding="utf-8",
    )
    cfg = replay.TrajectoryInputConfig(source="csv", csv_path=str(csv_path), pose_prefix="state")
    episodes, summary = replay._load_csv_episode_trajectories(cfg)
    # Metres over the corenetic jaw's 90 mm: the arm's gripper is commanded the same opening.
    assert [t.gripper_pos for t in episodes[0].targets] == pytest.approx([1.0, 0.0323 / 0.09, None], nan_ok=True)
    assert episodes[0].targets[2].gripper_pos is None
    assert summary["gripper_width_rows"] == 2
    runtime = replay.ReplayRuntimeConfig(use_dataset_gripper_for_replay=True, gripper_pos=1.0)
    assert replay._get_target_gripper_command(episodes[0].targets[1], runtime, 1.0) == pytest.approx(0.0323 / 0.09)
    assert replay._get_target_gripper_command(episodes[0].targets[2], runtime, 1.0) == 1.0


def test_trigger_pressed_home_replays_the_gripper_shut_not_at_the_tube_width(tmp_path):
    csv_path = tmp_path / "state_action.right.csv"
    # 09 Oct ep0: the jaw follows the trigger, then stops on the tube at 32.3 mm while the
    # trigger keeps going to 100 %.
    csv_path.write_text(
        "episode_index,frame_index,gripper_width_m,gripper_trigger_pct,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw\n"
        "0,0,0.0887,0.0,0.5,0,0.3,0,0,0,1\n"
        "0,1,0.0386,75.8,0.5,0,0.3,0,0,0,1\n"
        "0,2,0.0336,92.8,0.5,0,0.3,0,0,0,1\n"
        "0,3,0.0323,100.0,0.5,0,0.3,0,0,0,1\n"
        "0,4,0.0372,nan,0.5,0,0.3,0,0,0,1\n",
        encoding="utf-8",
    )
    cfg = replay.TrajectoryInputConfig(
        source="csv", csv_path=str(csv_path), pose_prefix="state", grasp_trigger_close_pct=90.0
    )
    episodes, summary = replay._load_csv_episode_trajectories(cfg)
    assert [t.gripper_pos for t in episodes[0].targets] == pytest.approx(
        [0.0887 / 0.09, 0.0386 / 0.09, 0.0, 0.0, 0.0372 / 0.09]
    )
    assert summary["grasp_rows"] == 2
    # Off by default: the recorded width throughout.
    plain, _ = replay._load_csv_episode_trajectories(
        replay.TrajectoryInputConfig(source="csv", csv_path=str(csv_path), pose_prefix="state")
    )
    assert plain[0].targets[3].gripper_pos == pytest.approx(0.0323 / 0.09)
    with pytest.raises(ValueError, match="grasp_closed_width_m"):
        replay.TrajectoryInputConfig(source="csv", csv_path=str(csv_path), grasp_closed_width_m=0.2)


def test_thor_profile_replays_the_dataset_gripper():
    import yaml

    profile = yaml.safe_load(_SCRIPT.with_name("replay_cube_pose_in_robot_base.thor.yaml").read_text())
    assert profile["replay"]["use_dataset_gripper_for_replay"] is True
    assert profile["input"]["grasp_trigger_close_pct"] == 90
    assert profile["input"]["grasp_closed_width_m"] == 0.0


def test_external_torques_are_nan_when_the_robot_cannot_report_them():
    class _Robot:
        pass

    class _Fr3:
        def get_external_joint_torques(self):
            return [0, 0, 0, 0, 0, 9.0, 0]

    assert all(v != v for v in replay._external_joint_torques(_Robot()))
    assert replay._external_joint_torques(_Fr3())[5] == 9.0
