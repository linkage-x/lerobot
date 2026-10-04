"""BOX UI command separation and retry gates, without launching hardware."""
from __future__ import annotations

import io
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tools.data_collection_gui import gateway
from tools.thor.gmsl2 import thor_record


@pytest.fixture
def state(tmp_path, monkeypatch):
    value = gateway.GatewayState(
        repo_root=tmp_path, config_path=tmp_path / "config.yaml",
        config={"recorder": {"script": "tools/thor/gmsl2/thor_record.py"},
                "dataset": {"root": str(tmp_path / "dataset"), "fps": 60},
                "fr3_teleop": {"enabled": True}},
        recording=gateway.RecordingStatus(state="armed"),
        replay=gateway.ReplayStatus(),
    )
    value.fr3_teleop = gateway.ThorFr3Status(enabled=True)
    value.devices = [{"id": "fr3", "state": "idle"}, {"id": "spacemouse", "state": "idle"},
                     {"id": "cam_00", "kind": "camera", "state": "running"}]
    written = []
    monkeypatch.setattr(gateway, "_ensure_recorder_running", lambda _: object())
    monkeypatch.setattr(gateway, "_write_recorder_stdin", lambda _, text: written.append(text))
    return value, written


def test_f_then_e_are_separate_commands_and_s_d_leave_arm_active(state):
    value, written = state
    with pytest.raises(RuntimeError, match="Press F"):
        gateway._start_episode(value)
    gateway._start_thor_fr3(value)
    assert written == ["fr3_start\n"] and value.recording.state == "armed"
    assert value.fr3_teleop.state == "starting"
    with pytest.raises(RuntimeError, match="Press F"):
        gateway._start_episode(value)
    gateway._apply_recorder_output(value, 'FR3_LIVE {"state":"running","telemetry":{"q":[0,0,0,0,0,0,0]}}')
    gateway._start_episode(value)
    assert written[-1] == "\n" and value.recording.state == "recording"
    gateway._stop_recorder(value, "save")
    assert written[-1] == "save\n" and value.fr3_teleop.state == "running"
    value.recording.state = "recording"
    gateway._stop_recorder(value, "discard")
    assert written[-1] == "n\n" and value.fr3_teleop.state == "running"
    assert "fr3_stop\n" not in written


def test_native_error_warns_and_allows_f_again_without_sensor_reconnect(state):
    value, written = state
    gateway._apply_recorder_output(value, 'FR3_LIVE {"state":"error","message":"clear error then F","telemetry":{},"pid":null}')
    assert value.recording.state == "armed"
    assert value.devices[-1]["state"] == "running"
    assert value.devices[0]["state"] == "error"
    gateway._start_thor_fr3(value)
    assert written == ["fr3_start\n"]


@pytest.mark.parametrize("recorder_state", ["idle", "connecting", "recording", "review", "saving", "discarding"])
def test_f_requires_connected_idle_sensors(state, recorder_state):
    value, written = state
    value.recording.state = recorder_state
    with pytest.raises(RuntimeError, match="Connect sensors"):
        gateway._start_thor_fr3(value)
    assert not written


@pytest.mark.parametrize("session_name", ["calibration_session", "tracker_mount_session", "marker_tcp_session"])
def test_f_cannot_move_during_a_calibration_capture(state, session_name):
    value, written = state
    capture = getattr(value, session_name)
    capture.active, capture.stage = True, "capture"
    with pytest.raises(RuntimeError, match="calibration"):
        gateway._start_thor_fr3(value)
    assert not written


def test_f_cannot_overlap_physical_replay(state):
    value, written = state
    value.replay_process = SimpleNamespace(poll=lambda: None)
    value.replay_process_kind = "real"
    with pytest.raises(RuntimeError, match="physical replay"):
        gateway._start_thor_fr3(value)
    assert not written


@pytest.mark.parametrize("fr3_state", ["starting", "moving_to_start", "running", "stopping"])
def test_calibration_capture_requires_stopped_fr3(state, fr3_state):
    value, written = state
    value.fr3_teleop.state = fr3_state
    with pytest.raises(RuntimeError, match="Stop FR3"):
        gateway._start_episode(value, capture_root=Path("/tmp/calibration"))
    assert not written


def test_calibration_capture_does_not_require_f(state):
    value, written = state
    gateway._start_episode(value, capture_root=Path("/tmp/calibration"))
    assert written == ["capture_root:/tmp/calibration\n", "\n"]


def test_sensor_connected_status_does_not_claim_arm_running(state):
    value, _ = state
    gateway._set_active_device_states(value, "running")
    gateway._sync_thor_fr3_devices(value)
    assert [device["state"] for device in value.devices] == ["idle", "idle", "running"]


def test_c_uses_selected_cameras_and_box_without_starting_fr3(state, monkeypatch):
    value, _ = state
    value.recording.state = "idle"
    value.devices = [
        {"id": f"cam_{sid:02d}", "kind": "camera", "state": "idle", "config": {"sensor_id": sid}}
        for sid in (6, 7, 8)
    ] + [
        {"id": "box_gripper", "kind": "box_collection", "state": "idle"},
        {"id": "fr3", "kind": "robot", "state": "idle"},
        {"id": "spacemouse", "kind": "teleoperator", "state": "idle"},
    ]
    commands = []
    monkeypatch.setattr(gateway.subprocess, "Popen", lambda command, **_: commands.append(command) or SimpleNamespace(pid=42, poll=lambda: None))
    monkeypatch.setattr(gateway, "_venv_python", lambda *_args, **_kwargs: Path("/tmp/python"))
    monkeypatch.setattr(gateway, "_recorder_env", lambda *_args: {})
    monkeypatch.setattr(gateway, "_start_output_reader", lambda *_args: None)
    gateway._connect_recorder(value, camera_ids=[6, 7], box_enabled=True, laser_tracker=False)
    assert "--sensor-ids=6,7" in commands[0]
    assert "--no-box" not in commands[0]
    assert value.recording.selectedCameraIds == ["cam_06", "cam_07"]
    assert value.recording.boxEnabled is True
    assert [device["state"] for device in value.devices] == ["warning", "warning", "idle", "warning", "idle", "idle"]
    gateway._mark_connected_devices(value, "camera", "cam_06, cam_07")
    gateway._set_active_device_states(value, "running")
    assert [device["state"] for device in value.devices] == ["running", "running", "idle", "running", "idle", "idle"]


def test_camera_only_c_disables_f_and_rejects_unavailable_camera(state, monkeypatch):
    value, written = state
    value.recording.state = "idle"
    value.devices = [{"id": "cam_06", "kind": "camera", "state": "idle", "config": {"sensor_id": 6}}]
    with pytest.raises(ValueError, match="not available"):
        gateway._connect_recorder(value, camera_ids=[7], box_enabled=False)
    commands = []
    monkeypatch.setattr(gateway.subprocess, "Popen", lambda command, **_: commands.append(command) or SimpleNamespace(pid=42, poll=lambda: None))
    monkeypatch.setattr(gateway, "_venv_python", lambda *_args, **_kwargs: Path("/tmp/python"))
    monkeypatch.setattr(gateway, "_recorder_env", lambda *_args: {})
    monkeypatch.setattr(gateway, "_start_output_reader", lambda *_args: None)
    gateway._connect_recorder(value, camera_ids=[6], box_enabled=False)
    assert "--sensor-ids=6" in commands[0] and "--no-box" in commands[0]
    value.recording.state = "armed"
    with pytest.raises(RuntimeError, match="requires the BOX gripper"):
        gateway._start_thor_fr3(value)
    assert not written


def test_box_only_configuration_keeps_e_without_f(state):
    value, written = state
    value.fr3_teleop.enabled = False
    gateway._start_episode(value)
    assert written == ["\n"]


def test_f_commands_are_out_of_band_and_do_not_start_camera_episode(monkeypatch):
    seen, queue = [], []
    monkeypatch.setattr(thor_record.sys, "stdin", io.StringIO("fr3_start\nfr3_stop\n\nsave\nq\n"))
    thor_record._read_stdin_loop(queue, threading.Event(), on_fr3_teleop=seen.append)
    assert seen == [True, False, False]
    assert [command.kind for command in queue] == ["start", "save", "quit"]


def test_original_box_reader_ignores_fr3_commands_without_callback(monkeypatch):
    queue = []
    monkeypatch.setattr(thor_record.sys, "stdin", io.StringIO("fr3_start\nfr3_stop\nq\n"))
    thor_record._read_stdin_loop(queue, threading.Event())
    assert [command.kind for command in queue] == ["quit"]


def test_fr3_marker_capture_before_f_uses_its_own_root_and_episode_index(state, tmp_path):
    value, written = state
    marker_root = tmp_path / "marker_tcp_capture"
    (marker_root / "episodes/episode_000002").mkdir(parents=True)
    value.marker_tcp_session = gateway.MarkerTcpSession(
        active=True, stage="capture", sessionRoot=str(marker_root), sessionName="marker_test",
    )
    value.recording.datasetRoot = str(tmp_path / "task_dataset")
    value.recording.episodeIndex = 8
    result = gateway._marker_tcp_record_sample(value, "start", box_id="box-a", condition="pivot_01")
    assert result["ok"] is True
    assert value.fr3_teleop.state == "idle"
    assert written[0] == f"capture_root:{marker_root}\n"
    assert "fr3_start\n" not in written
    sample = value.marker_tcp_session.samples[0]
    assert sample.datasetRoot == str(marker_root) and sample.episodeIndex == 3
    assert gateway._marker_tcp_record_sample(value, "save")["ok"] is True
    assert sample.datasetRoot == str(marker_root) and sample.episodeIndex == 3


@pytest.mark.parametrize("motion_state", ["starting", "moving_to_start", "running", "stopping"])
def test_fr3_marker_capture_requires_motion_to_stop(state, tmp_path, motion_state):
    value, written = state
    value.fr3_teleop.state = motion_state
    value.marker_tcp_session = gateway.MarkerTcpSession(
        active=True, stage="capture", sessionRoot=str(tmp_path / "marker"),
    )
    result = gateway._marker_tcp_record_sample(value, "start", box_id="box-a", condition="pivot_01")
    assert result["ok"] is False
    assert "Stop FR3" in result["error"]
    assert not written and not value.marker_tcp_session.samples


def test_box_only_marker_capture_retains_legacy_task_root(state, tmp_path):
    value, written = state
    value.fr3_teleop.enabled = False
    value.recording.datasetRoot = str(tmp_path / "task_dataset")
    value.recording.episodeIndex = 8
    value.marker_tcp_session = gateway.MarkerTcpSession(
        active=True, stage="capture", sessionRoot=str(tmp_path / "marker"),
    )
    result = gateway._marker_tcp_record_sample(value, "start", box_id="box-a", condition="pivot_01")
    assert result["ok"] is True
    assert not any(line.startswith("capture_root:") for line in written)
    sample = value.marker_tcp_session.samples[0]
    assert sample.datasetRoot == value.recording.datasetRoot and sample.episodeIndex == 8


@pytest.mark.parametrize("command", ["q\n", "quit\n", "exit\n", ""])
def test_exit_and_parent_eof_stop_fr3_before_camera_cleanup_queue(monkeypatch, command):
    queue = []
    seen = []
    def stop(active):
        assert not queue  # arm stop precedes the main thread's camera cleanup
        seen.append(active)
    monkeypatch.setattr(thor_record.sys, "stdin", io.StringIO(command))
    thor_record._read_stdin_loop(queue, threading.Event(), on_fr3_teleop=stop)
    assert seen == [False] and [item.kind for item in queue] == ["quit"]
