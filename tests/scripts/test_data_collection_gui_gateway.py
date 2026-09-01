from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import threading
import time

import numpy as np
import pytest
import yaml

from tools.data_collection_gui import gateway


def _write_minimal_episode_dataset(dataset_root: Path, total_episodes: int = 3) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": total_episodes,
        "total_frames": total_episodes * 2,
        "features": {
            "observation.state": {
                "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"],
            },
            "action": {
                "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"],
            },
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = {
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "observation.state": [],
        "action": [],
    }
    for episode in range(total_episodes):
        for frame in range(2):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["timestamp"].append(frame / 30.0)
            pose = [0.3 + 0.001 * frame, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.5]
            rows["observation.state"].append(pose)
            rows["action"].append(pose)
    pq.write_table(pa.table(rows), dataset_root / "data" / "chunk-000" / "file-000.parquet")


def test_dataset_scan_signature_tracks_v3_finalization_without_root_mtime_change(tmp_path):
    datasets_root = tmp_path / "outputs" / "datasets"
    dataset_root = datasets_root / "thor_gmsl2_7ch_v1_20260720_151325"
    dataset_root.mkdir(parents=True)
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(dataset_root, ns=(fixed_ns, fixed_ns))
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=datasets_root,
    )
    before = gateway._dataset_scan_signature(state)

    info_path = dataset_root / "meta" / "info.json"
    info_path.parent.mkdir()
    info_path.write_text(json.dumps({"total_episodes": 1, "total_frames": 345, "fps": 60}), encoding="utf-8")
    os.utime(dataset_root, ns=(fixed_ns, fixed_ns))

    assert gateway._dataset_scan_signature(state) != before


def test_dataset_stats_refresh_defers_while_recorder_active(tmp_path, monkeypatch):
    dataset_root = tmp_path / "outputs" / "datasets" / "active_session"
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(
            repoId="local/test",
            datasetRoot=str(dataset_root),
            pid=1234,
        ),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    def fail_scan(_state):
        raise AssertionError("active recorder should defer dataset scans")

    monkeypatch.setattr(gateway, "_dataset_scan_signature", fail_scan)

    gateway._refresh_dataset_stats_cache(state)

    assert state.dataset_cache_ready is False


def test_dataset_stats_refresh_resumes_after_recorder_exit(tmp_path, monkeypatch):
    dataset_root = tmp_path / "outputs" / "datasets" / "finished_session"
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test", datasetRoot=str(dataset_root), pid=None),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    monkeypatch.setattr(gateway, "_dataset_scan_signature", lambda _state: (("finished",),))
    monkeypatch.setattr(gateway, "_recorded_dataset_items", lambda _state: [{"path": str(dataset_root)}])
    monkeypatch.setattr(gateway, "_read_recorded_trajectory", lambda _state: ([{"frame": 0}], {"dataStatus": "loaded"}))

    gateway._refresh_dataset_stats_cache(state)

    assert state.dataset_cache_ready is True
    assert state.cached_recorded_datasets == [{"path": str(dataset_root)}]
    assert state.cached_trajectory == [{"frame": 0}]
    assert state.cached_trajectory_meta == {"dataStatus": "loaded"}


def test_make_state_loads_handheld_config_contract():
    state = gateway.make_state(Path.cwd(), Path("tools/handheld/handheld_record_example.yaml"))

    snapshot = gateway._snapshot(state)

    assert snapshot["configSummary"]["repoId"] == "local/handheld_multimodal_v1"
    assert snapshot["recording"]["repoId"] == "local/handheld_multimodal_v1"
    assert snapshot["recording"]["targetFrames"] == 300
    assert snapshot["replay"]["fps"] == 30
    assert any(device["id"] == "cam_0" for device in snapshot["devices"])
    assert any(device["kind"] == "handheld_gripper" for device in snapshot["devices"])


def test_default_config_is_thor_gmsl2_box():
    assert str(gateway.DEFAULT_CONFIG_PATH) == "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml"
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    snapshot = gateway._snapshot(state)

    assert snapshot["configSummary"]["repoId"] == "local/thor_gmsl2_Nch_v1"
    assert snapshot["configSummary"]["fps"] == 60
    devices_by_kind: dict[str, list[str]] = {}
    for device in snapshot["devices"]:
        devices_by_kind.setdefault(device["kind"], []).append(device["id"])
    # detected-camera GMSL2 rig (detect_all => sids 0..15 placeholder before connect).
    assert "camera" in devices_by_kind
    assert all(cid.startswith("cam_") for cid in devices_by_kind["camera"])
    assert len(devices_by_kind["camera"]) >= 11
    # Box collection sensors are surfaced as a distinct device kind.
    assert "box_collection" in devices_by_kind
    assert {"box_gripper", "box_imu", "box_trigger"}.issubset(set(devices_by_kind["box_collection"]))
    # Old Hikrobot / Pika devices are no longer in the default rig.
    assert "handheld_gripper" not in devices_by_kind


def _calibration_gateway_state(tmp_path: Path) -> gateway.GatewayState:
    dataset_root = tmp_path / "outputs" / "datasets" / "thor_gmsl2_Nch_v1"
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={
            "dataset": {
                "repo_id": "local/test",
                "root": str(dataset_root),
                "fps": 60,
                "episode_time_s": 20,
            },
            "recorder": {"script": "tools/thor/gmsl2/thor_record.py"},
        },
        recording=gateway.RecordingStatus(
            repoId="local/test",
            datasetRoot=str(dataset_root),
            state="armed",
            savedEpisodes=2,
        ),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )
    state.devices = [
        {"id": "cam_06", "kind": "camera"},
        {"id": "cam_07", "kind": "camera"},
    ]
    return state


def _open_calibration_segment(state: gateway.GatewayState, monkeypatch) -> None:
    def fake_start_episode(fake_state: gateway.GatewayState, episode_time_s: float | None = None) -> None:
        fake_state.recording.state = "recording"
        fake_state.recording.frameIndex = 0

    monkeypatch.setattr(gateway, "_start_episode", fake_start_episode)
    assert gateway._start_calibration_session(state)["ok"] is True
    assert gateway._calibration_step_record(state, "start")["ok"] is True


def _refuse_stop_recorder(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("must not drive a recorder that already closed the episode")

    monkeypatch.setattr(gateway, "_stop_recorder", fail)


def _capture_recorder_stdin(monkeypatch) -> list[str]:
    written: list[str] = []
    monkeypatch.setattr(gateway, "_ensure_recorder_running", lambda _state: object())
    monkeypatch.setattr(gateway, "_write_recorder_stdin", lambda _proc, text: written.append(text))
    return written


def test_start_episode_asks_the_gmsl2_recorder_for_a_specific_length(tmp_path, monkeypatch):
    state = _calibration_gateway_state(tmp_path)
    written = _capture_recorder_stdin(monkeypatch)

    gateway._start_episode(state, 30)

    # The length has to reach the recorder before the start newline, or the
    # episode is already running under the config's length.
    assert written == ["episode_time:30\n", "\n"]
    # 30 s x 60 fps: targetFrames is what flips the recorder to "review", so it
    # has to follow the length actually asked for.
    assert state.recording.targetFrames == 1800
    assert state.recording.state == "recording"


def test_start_episode_without_an_override_restores_the_config_length(tmp_path, monkeypatch):
    state = _calibration_gateway_state(tmp_path)
    state.recording.targetFrames = 1800  # left over from a calibration sweep
    written = _capture_recorder_stdin(monkeypatch)

    gateway._start_episode(state)

    assert written == ["\n"]
    assert state.recording.targetFrames == 1200  # 20 s x 60 fps from the config


def test_start_episode_refuses_a_length_the_recorder_cannot_honour(tmp_path, monkeypatch):
    # The FR3 runtime queues unrecognised stdin lines as commands, so an
    # episode_time line there would be noise in its state machine rather than a
    # longer episode. Say so instead of silently recording the wrong length.
    state = _calibration_gateway_state(tmp_path)
    state.config.pop("recorder")
    written = _capture_recorder_stdin(monkeypatch)

    with pytest.raises(RuntimeError, match="episode length"):
        gateway._start_episode(state, 30)

    assert written == []


def test_calibration_session_defaults_to_30s_sweeps_and_accepts_an_override(tmp_path):
    state = _calibration_gateway_state(tmp_path)

    assert gateway._start_calibration_session(state)["ok"] is True
    assert state.calibration_session.episodeTimeS == 30.0

    assert gateway._cancel_calibration_session(state)["ok"] is True
    assert gateway._start_calibration_session(state, "", "45")["ok"] is True
    assert state.calibration_session.episodeTimeS == 45.0


def test_calibration_segment_length_is_editable_between_sweeps(tmp_path):
    state = _calibration_gateway_state(tmp_path)
    assert gateway._start_calibration_session(state)["ok"] is True

    assert gateway._set_calibration_segment_seconds(state, "60")["ok"] is True
    assert state.calibration_session.episodeTimeS == 60.0
    # Out of range and non-numeric both keep the value that was working.
    assert gateway._set_calibration_segment_seconds(state, "1")["ok"] is False
    assert gateway._set_calibration_segment_seconds(state, "600")["ok"] is False
    assert gateway._set_calibration_segment_seconds(state, "abc")["ok"] is False
    assert state.calibration_session.episodeTimeS == 60.0


def test_calibration_segment_length_is_locked_while_a_sweep_records(tmp_path, monkeypatch):
    # The recorder was already told how long this episode runs; accepting a new
    # number would describe the segment on screen wrongly.
    state = _calibration_gateway_state(tmp_path)
    _open_calibration_segment(state, monkeypatch)

    assert gateway._set_calibration_segment_seconds(state, "60")["ok"] is False
    assert state.calibration_session.episodeTimeS == 30.0


def test_calibration_start_asks_the_recorder_for_the_session_length(tmp_path, monkeypatch):
    state = _calibration_gateway_state(tmp_path)
    seen: list[float | None] = []

    def fake_start_episode(fake_state: gateway.GatewayState, episode_time_s: float | None = None) -> None:
        seen.append(episode_time_s)
        fake_state.recording.state = "recording"

    monkeypatch.setattr(gateway, "_start_episode", fake_start_episode)
    assert gateway._start_calibration_session(state, "", "45")["ok"] is True

    assert gateway._calibration_step_record(state, "start")["ok"] is True

    assert seen == [45.0]


def test_calibration_save_ends_a_live_segment_early(tmp_path, monkeypatch):
    state = _calibration_gateway_state(tmp_path)
    _open_calibration_segment(state, monkeypatch)
    stopped: list[str] = []
    monkeypatch.setattr(gateway, "_stop_recorder", lambda _s, action: stopped.append(action))

    result = gateway._calibration_step_record(state, "save")

    assert result["ok"] is True
    # What the button promises: end the open episode now rather than waiting out
    # the configured episode_time_s.
    assert stopped == ["save"]
    step = state.calibration_session.steps[0]
    assert step.status == "captured"
    assert step.episodeIndex == 2
    assert state.calibration_session.currentIndex == 1
    # The solve has to run over the dataset the recorder is writing into, not the
    # calib_<ts> label the session made up before knowing it.
    assert state.calibration_session.datasetRoot == state.recording.datasetRoot
    assert state.calibration_session.datasetName == "thor_gmsl2_Nch_v1"


def test_calibration_save_registers_a_segment_the_recorder_already_auto_saved(tmp_path, monkeypatch):
    # dataset.episode_time_s is 10 s on the GMSL2 rig, so the recorder saves the
    # episode and re-arms long before the operator is done waving the board.
    # Clicking 保存本段 then used to come back "Cannot save while recorder is
    # armed" and lose a segment that was already written.
    state = _calibration_gateway_state(tmp_path)
    _open_calibration_segment(state, monkeypatch)
    _refuse_stop_recorder(monkeypatch)
    state.recording.state = "armed"
    state.recording.savedEpisodes = 3

    result = gateway._calibration_step_record(state, "save")

    assert result["ok"] is True
    step = state.calibration_session.steps[0]
    assert step.status == "captured"
    assert step.episodeIndex == 2
    assert "自动" in step.note
    assert state.calibration_session.currentIndex == 1


def test_calibration_save_refuses_when_the_segment_never_landed(tmp_path, monkeypatch):
    # Recorder closed the episode without writing it (frame-sync gate failure,
    # stream exit): savedEpisodes never moved, so there is nothing to mark
    # captured and saying "captured" would feed the solve a segment that is not
    # on disk.
    state = _calibration_gateway_state(tmp_path)
    _open_calibration_segment(state, monkeypatch)
    _refuse_stop_recorder(monkeypatch)
    state.recording.state = "armed"

    result = gateway._calibration_step_record(state, "save")

    assert result["ok"] is False
    assert state.calibration_session.steps[0].status == "pending"
    assert state.calibration_session.currentIndex == 0


def test_calibration_discard_admits_an_auto_saved_segment_cannot_be_taken_back(tmp_path, monkeypatch):
    state = _calibration_gateway_state(tmp_path)
    _open_calibration_segment(state, monkeypatch)
    _refuse_stop_recorder(monkeypatch)
    state.recording.state = "armed"
    state.recording.savedEpisodes = 3

    result = gateway._calibration_step_record(state, "discard")

    assert result["ok"] is True
    step = state.calibration_session.steps[0]
    # Re-recording is still the right next move, but the solver reads every
    # episode under the dataset, so the operator has to know this one stays.
    assert step.status == "pending"
    assert "无法撤回" in state.calibration_session.message


def _marker_tcp_gateway_state(tmp_path: Path) -> gateway.GatewayState:
    dataset_root = tmp_path / "outputs" / "datasets" / "marker_tcp_raw"
    return gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(
            repoId="local/test",
            datasetRoot=str(dataset_root),
            state="armed",
            episodeIndex=7,
        ),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )


def _write_static_transform(path: Path, *, x_m: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "T_ee_cube": [
                    [1.0, 0.0, 0.0, x_m],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "num_samples_used": 12,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_marker_tcp_sample_save_tolerates_recorder_returning_to_armed(tmp_path, monkeypatch):
    state = _marker_tcp_gateway_state(tmp_path)
    assert gateway._start_marker_tcp_session(state)["ok"] is True

    def fake_start_episode(fake_state):
        fake_state.recording.state = "recording"
        fake_state.recording.frameIndex = 0

    monkeypatch.setattr(gateway, "_start_episode", fake_start_episode)
    result = gateway._marker_tcp_record_sample(state, "start", side="left", condition="same_mount_01")
    assert result["ok"] is True
    sample = state.marker_tcp_session.samples[0]
    assert sample.status == "recording"
    assert sample.episodeIndex == 7

    state.recording.state = "armed"
    result = gateway._marker_tcp_record_sample(state, "save", side="left", condition="same_mount_01")

    assert result["ok"] is True
    assert state.marker_tcp_session.pendingSampleId == ""
    assert state.marker_tcp_session.samples[0].status == "saved"
    assert state.marker_tcp_session.samples[0].episodeIndex == 7


def test_marker_tcp_sample_records_box_id_target(tmp_path, monkeypatch):
    state = _marker_tcp_gateway_state(tmp_path)
    assert gateway._start_marker_tcp_session(state)["ok"] is True

    def fake_start_episode(fake_state):
        fake_state.recording.state = "recording"
        fake_state.recording.frameIndex = 0

    monkeypatch.setattr(gateway, "_start_episode", fake_start_episode)
    result = gateway._marker_tcp_record_sample(
        state,
        "start",
        box_id="box1819152274",
        condition="same_mount_01",
    )

    assert result["ok"] is True
    sample = state.marker_tcp_session.samples[0]
    assert sample.boxId == "box1819152274"
    assert sample.side == "box1819152274"
    assert result["markerTcp"]["samples"][0]["boxId"] == "box1819152274"


def test_marker_tcp_registers_static_transforms_and_writes_report(tmp_path, monkeypatch):
    state = _marker_tcp_gateway_state(tmp_path)
    monkeypatch.syspath_prepend(str(Path.cwd() / "third_party" / "opencv_kalibr"))
    assert gateway._start_marker_tcp_session(state)["ok"] is True
    a = _write_static_transform(tmp_path / "a" / "static_transform.json", x_m=0.0)
    b = _write_static_transform(tmp_path / "b" / "static_transform.json", x_m=0.001)

    assert gateway._register_marker_tcp_static_transform(state, path_arg=str(a), side="left", condition="same_mount_01")["ok"] is True
    assert gateway._register_marker_tcp_static_transform(state, path_arg=str(b), side="left", condition="remount_01")["ok"] is True
    result = gateway._run_marker_tcp_repeatability_report(state)

    assert result["ok"] is True
    report_path = Path(state.marker_tcp_session.reportPath)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_bundles"] == 2
    assert report["translation_error_mm"]["max"] == pytest.approx(0.5)


def test_workstation_profile_exposes_fr3_teleop_contract():
    state = gateway.make_state(
        Path.cwd(),
        Path("tools/fr3/fr3_record_config.yaml"),
        profile="workstation",
    )

    snapshot = gateway._snapshot(state)

    assert snapshot["deployment"]["profile"] == "workstation"
    assert snapshot["deployment"]["defaultRoute"] == "teleoperation"
    assert {"fr3", "pika", "spacemouse", "realsense", "mujoco"}.issubset(
        snapshot["deployment"]["capabilities"]
    )
    assert {"camera", "robot", "gripper", "teleoperator"}.issubset(
        {device["kind"] for device in snapshot["devices"]}
    )
    devices_by_id = {device["id"]: device for device in snapshot["devices"]}
    assert devices_by_id["fr3"]["label"] == "Franka Research 3"
    # A persistent /dev/serial alias, not a bare /dev/ttyUSB* whose number moves on replug.
    # by-path is preferred over by-id here because the gripper's CH340 adapter reports no
    # USB serial number, so its by-id name is not unique when several are attached.
    assert devices_by_id["pika"]["config"]["port"].startswith("/dev/serial/by-")
    assert snapshot["teleop"]["urdfPath"].endswith("fr3_pika_gripper.urdf")
    assert snapshot["teleop"]["simXmlPath"].endswith("fr3_pika_gripper_scene.xml")
    assert "ati" not in snapshot["teleop"]["urdfPath"].lower()
    assert [view["id"] for view in snapshot["teleop"]["cameraViews"]] == ["external", "wrist"]
    assert [view["deviceId"] for view in snapshot["teleop"]["cameraViews"]] == ["side", "ee"]
    assert snapshot["replay"]["realRobotIp"] == "192.168.1.206"
    assert snapshot["replay"]["realEndEffectorMode"] == "pika_gripper_ee"


def test_workstation_teleop_prefers_fr3_virtualenv(tmp_path: Path):
    default_python = tmp_path / ".venv" / "bin" / "python"
    fr3_python = tmp_path / ".venv-fr3" / "bin" / "python"
    default_python.parent.mkdir(parents=True)
    fr3_python.parent.mkdir(parents=True)
    default_python.touch()
    fr3_python.touch()
    state = gateway.make_state(
        Path.cwd(),
        Path("tools/fr3/fr3_record_config.yaml"),
        profile="workstation",
    )
    state.repo_root = tmp_path

    command = gateway._fr3_sim_teleop_command(state)

    assert command[0] == str(fr3_python)
    assert "--no-viewer" in command
    assert "--viewer-camera" not in command
    assert gateway._venv_python(tmp_path) == default_python
    assert gateway._venv_python3(tmp_path, prefer_fr3=True) == fr3_python


def test_workstation_real_teleop_command_uses_record_config_without_preflight_gate():
    state = gateway.make_state(
        Path.cwd(),
        Path("tools/fr3/fr3_record_config.yaml"),
        profile="workstation",
    )

    command = gateway._fr3_real_teleop_command(state)

    assert command[0].endswith((".venv-fr3/bin/python", ".venv/bin/python"))
    assert command[1:3] == ["-m", "tools.fr3.fr3_real_teleop_runtime"]
    assert command[3] == f"--config_path={state.config_path}"
    assert all("preflight" not in part for part in command)


def test_start_workstation_real_teleop_does_not_call_hardware_preflight(monkeypatch):
    state = gateway.make_state(
        Path.cwd(),
        Path("tools/fr3/fr3_record_config.yaml"),
        profile="workstation",
    )
    captured = {}

    class FakeProcess:
        pid = 4242
        stdout = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway, "_start_teleop_output_reader", lambda *_args: None)
    monkeypatch.setattr(
        gateway,
        "_run_real_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("preflight must not run")),
    )

    gateway._start_fr3_real_teleop(state)

    assert state.teleop.state == "starting"
    assert state.teleop.backend == "real"
    assert state.teleop.realRobotReady is False
    assert state.teleop.pid == 4242
    assert captured["command"] == gateway._fr3_real_teleop_command(state)


def test_gmsl2_device_preview_uses_recorder_owned_frames_only():
    gmsl2_state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)

    assert gateway._state_is_gmsl2(gmsl2_state) is True
    assert gateway._should_use_recorder_camera_preview(gmsl2_state) is True

    handheld_state = gateway.make_state(Path.cwd(), Path("tools/handheld/handheld_record_example.yaml"))

    assert gateway._state_is_gmsl2(handheld_state) is False
    assert gateway._should_use_recorder_camera_preview(handheld_state) is False

    handheld_state.camera_preview_suspended = True

    assert gateway._should_use_recorder_camera_preview(handheld_state) is True


def test_previews_suspended_for_connect_keeps_flag_on_success():
    """Success path: recorder now owns the cameras, so the suspend flag must
    stay set (later cleared by _stop_recorder / _snapshot on recorder exit)."""
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    assert state.camera_preview_suspended is False
    with gateway._previews_suspended_for_connect(state):
        assert state.camera_preview_suspended is True
    assert state.camera_preview_suspended is True


def test_previews_suspended_for_connect_resets_flag_on_any_exception():
    """Failure path: any error before the recorder takes the cameras (a
    blocking terminate(), the settle sleep, or _connect_recorder raising) must
    reset the flag so the operator can keep inspecting the Device Manager grid.
    This is the regression the context manager closes — previously only the
    hand-written connect except reset it, and the preflight ran outside it."""
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    try:
        with gateway._previews_suspended_for_connect(state):
            assert state.camera_preview_suspended is True
            raise RuntimeError("connect blew up mid-preflight")
    except RuntimeError:
        pass
    assert state.camera_preview_suspended is False


def test_gmsl2_timeline_ignores_replay_warmup_for_splitmux_episode(tmp_path):
    dataset_root = tmp_path / "gmsl2"
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "cam_00.mkv").write_bytes(b"0" * 2048)
    (ep_dir / "meta.json").write_text(
        json.dumps({
            "duration_s": 10.0,
            "video": {"fps": 60, "replay_warmup_s": 1.5},
        }),
        encoding="utf-8",
    )

    timeline = gateway._read_gmsl2_timeline(dataset_root, episode=0)

    assert timeline["videoWarmupS"] == 0.0
    assert timeline["totalFrames"] == 600
    assert timeline["cameraKeys"] == ["cam_00"]
    assert timeline["frames"][0]["timestamp"] == 0


def test_gmsl2_timeline_exposes_per_camera_video_offsets(tmp_path):
    dataset_root = tmp_path / "gmsl2"
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    t0 = 1000.0
    (ep_dir / "cam_00.mkv").write_bytes(b"0" * 2048)
    (ep_dir / "cam_01.mkv").write_bytes(b"1" * 2048)
    (ep_dir / "meta.json").write_text(
        json.dumps(
            {
                "duration_s": 1.0,
                "video": {"fps": 60},
                "sync_reference": {
                    "t0_wall_s": t0,
                    "camera_first_wall_s": {"cam_00": t0 + 0.10, "cam_01": t0 + 0.14},
                },
            }
        ),
        encoding="utf-8",
    )

    timeline = gateway._read_gmsl2_timeline(dataset_root, episode=0)

    assert timeline["frames"][0]["timestamp"] == pytest.approx(0.12)
    assert timeline["cameraVideoOffsetsS"] == {"cam_00": pytest.approx(0.10), "cam_01": pytest.approx(0.14)}
    assert gateway._gmsl2_camera_video_offsets_s(
        {
            "sync_reference": {
                "t0_wall_s": t0,
                "camera_first_wall_s": {"cam_00": t0 + 0.10},
            }
        },
        ["observation.images.cam_00"],
    ) == {"observation.images.cam_00": pytest.approx(0.10)}


def test_resolve_gmsl2_video_path_accepts_lerobot_feature_key(tmp_path, monkeypatch):
    dataset_root = tmp_path / "gmsl2"
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    mkv = ep_dir / "cam_00.mkv"
    mkv.write_bytes(b"0" * 2048)
    (ep_dir / "meta.json").write_text(json.dumps({"duration_s": 1.0}), encoding="utf-8")
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", episode=0),
        datasets_root=dataset_root.parent,
    )
    monkeypatch.setattr(gateway, "_remux_mkv_to_mp4", lambda *_args, **_kwargs: None)

    assert gateway._resolve_video_path(state, dataset_root, "observation.images.cam_00") is None


def test_processing_item_and_qc_include_online_sync_manifest(tmp_path):
    dataset_root = tmp_path / "gmsl2_v3"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "meta.json").write_text(json.dumps({"duration_s": 2 / 30.0, "video": {"fps": 30}}), encoding="utf-8")
    (ep_dir / "online_sync_manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "failure": "",
                "actual_frames": 2,
                "sync_source": "sof_tsc_ns",
                "tolerance_ns": 1_000_000,
                "frame_count_by_camera": {"cam_00": 2, "cam_01": 2},
                "max_abs_delta_ns_by_camera": {"cam_00": 12_000, "cam_01": 18_000},
                "active_cameras": ["cam_00", "cam_01"],
            }
        ),
        encoding="utf-8",
    )

    item = gateway._processing_item_from_dataset(dataset_root)
    qc = gateway._run_qc(dataset_root)

    assert item["onlineSync"]["status"] == "pass"
    assert item["onlineSync"]["actualFrames"] == 2
    assert item["onlineSync"]["maxSofDeltaMs"] == pytest.approx(0.018)
    check = next(check for check in qc["checks"] if check["name"] == "online_sync_manifest")
    assert check["status"] == "pass"
    assert qc["online_sync"]["episodes"][0]["frameCountByCamera"] == {"cam_00": 2, "cam_01": 2}


def test_lerobot_v3_gmsl2_timeline_ignores_replay_warmup(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "cam_00.mkv").write_bytes(b"0" * 2048)
    (ep_dir / "meta.json").write_text(
        json.dumps({
            "duration_s": 10.0,
            "video": {"fps": 30, "replay_warmup_s": 1.0},
        }),
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    timeline = gateway._read_dataset_timeline(state, dataset_root, episode=0)

    assert timeline["videoWarmupS"] == 0.0
    assert timeline["totalFrames"] == 2
    assert timeline["cameraKeys"] == ["cam_00"]
    assert [frame["timestamp"] for frame in timeline["frames"]] == [0.0, 1 / 30.0]


def test_cached_mp4_rejects_short_duration(tmp_path, monkeypatch):
    mkv = tmp_path / "cam_00.mkv"
    mp4 = tmp_path / "cam_00.mp4"
    mkv.write_bytes(b"m" * 100_000)
    mp4.write_bytes(b"p" * 80_000)
    newer = mkv.stat().st_mtime + 1
    import os
    os.utime(mp4, (newer, newer))
    monkeypatch.setattr(gateway, "_probe_video_duration_s", lambda _path: 0.93)

    assert gateway._cached_mp4_is_usable(mp4, mkv, expected_duration_s=10.0) is False


def test_remux_writes_tmp_then_replaces_mp4(tmp_path, monkeypatch):
    mkv = tmp_path / "cam_00.mkv"
    mp4 = tmp_path / "cam_00.mp4"
    mkv.write_bytes(b"m" * 100_000)
    mp4.write_bytes(b"bad-cache")
    calls: list[Path] = []

    def fake_cache_ok(candidate: Path, _mkv: Path, _expected: float | None = None) -> bool:
        calls.append(candidate)
        return candidate.name.startswith(".cam_00.") and candidate.suffix == ".mp4" and candidate.exists()

    class FakeResult:
        returncode = 0

    def fake_run(cmd, **_kwargs):
        assert cmd[:5] == ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        output = Path(cmd[-1])
        assert output.name.startswith(".cam_00.")
        output.write_bytes(b"good-remux")
        return FakeResult()

    monkeypatch.setattr(gateway, "_cached_mp4_is_usable", fake_cache_ok)
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    result = gateway._remux_mkv_to_mp4(mkv, expected_duration_s=10.0)

    assert result == mp4
    assert mp4.read_bytes() == b"good-remux"
    assert any(path == mp4 for path in calls)
    assert any(path.name.startswith(".cam_00.") for path in calls)
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_gmsl2_timeline_includes_touch_heatmap_samples(tmp_path):
    dataset_root = tmp_path / "gmsl2"
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    (ep_dir / "cam_00.mkv").write_bytes(b"0" * 2048)
    (ep_dir / "meta.json").write_text(
        json.dumps({
            "duration_s": 1.0,
            "video": {"fps": 2, "replay_warmup_s": 0.5},
        }),
        encoding="utf-8",
    )
    left_fx = [0.0] * 239
    left_fy = [0.0] * 239
    left_fz = [0.0] * 239
    right_fz = [0.0] * 239
    left_fx[0] = 2.0
    left_fy[0] = -3.0
    left_fz[0] = 7.0
    right_fz[238] = 11.0
    rows = [
        {
            "sid": "box_touch_left",
            "t_rel_s": 0.5,
            "data": {"timestamp": 101, "fx_0p1N": left_fx, "fy_0p1N": left_fy, "fz_0p1N": left_fz},
        },
        {
            "sid": "box_touch_right",
            "t_rel_s": 0.5,
            "data": {"timestamp": 202, "fz_0p1N": right_fz},
        },
    ]
    with (ep_dir / "box_sensors.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    timeline = gateway._read_gmsl2_timeline(dataset_root, episode=0)

    touch = timeline["frames"][1]["touch"]
    assert touch["left"]["timestamp"] == 101
    assert touch["left"]["fx"][0] == 2.0
    assert touch["left"]["fy"][0] == -3.0
    assert touch["left"]["fz"][0] == 7.0
    assert touch["left"]["activePoints"] == 1
    assert touch["right"]["timestamp"] == 202
    assert touch["right"]["fz"][238] == 11.0


def test_touch_from_parquet_row_includes_shear_axes():
    fx = [0.0] * 239
    fy = [0.0] * 239
    fz = [0.0] * 239
    fx[3] = 5.0
    fy[3] = -4.0
    row = {
        "timestamp": 1.25,
        "observation.touch.box_touch_left.fx_0p1N": fx,
        "observation.touch.box_touch_left.fy_0p1N": fy,
        "observation.touch.box_touch_left.fz_0p1N": fz,
    }

    touch = gateway._touch_from_parquet_row(row, {})

    assert touch["left"]["timestamp"] == 1_250_000
    assert touch["left"]["fx"][3] == 5.0
    assert touch["left"]["fy"][3] == -4.0
    assert touch["left"]["fz"][3] == 0.0
    assert touch["left"]["activePoints"] == 1

def test_box_collection_devices_use_remote_endpoint_in_detail():
    config = {
        "sensors": {"cameras": {"defaults": {"fps": 60}, "detect_all": False, "sensor_ids": [0, 4]}},
        "box_collection": {
            "enabled": True,
            "remote_ip": "10.20.30.40",
            "remote_port": 15000,
            "poll_interval_s": 0.05,
            "expected_devices": ["box_gripper", "box_imu"],
        },
    }
    devices = gateway._device_statuses(config)
    box_devices = [d for d in devices if d["kind"] == "box_collection"]
    assert [d["id"] for d in box_devices] == ["box_gripper", "box_imu"]
    assert all(d["detail"] == "UDP 10.20.30.40:15000" for d in box_devices)
    assert all(d["fps"] == 20 for d in box_devices)  # 1 / 0.05 -> 20 Hz


def test_box_collection_disabled_hides_devices():
    config = {
        "sensors": {"cameras": {"defaults": {"fps": 60}}},
        "box_collection": {"enabled": False, "expected_devices": ["box_gripper"]},
    }
    devices = gateway._device_statuses(config)
    assert not any(d["kind"] == "box_collection" for d in devices)


def test_recorder_output_marks_box_collection_devices(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "cam_00", "kind": "camera", "label": "cam_00", "state": "warning", "fps": 60, "latencyMs": 0, "detail": ""},
            {"id": "box_gripper", "kind": "box_collection", "label": "g", "state": "warning", "fps": 20, "latencyMs": 0, "detail": ""},
            {"id": "box_imu", "kind": "box_collection", "label": "i", "state": "warning", "fps": 20, "latencyMs": 0, "detail": ""},
            {"id": "box_trigger", "kind": "box_collection", "label": "t", "state": "warning", "fps": 20, "latencyMs": 0, "detail": ""},
        ],
    )

    gateway._apply_recorder_output(state, "Cameras: cam_00")
    gateway._apply_recorder_output(state, "Box devices: box_gripper, box_imu")

    states = {device["id"]: device["state"] for device in state.devices}
    assert states["cam_00"] == "running"
    assert states["box_gripper"] == "running"
    assert states["box_imu"] == "running"
    # Devices listed in the YAML but not in the connected announcement
    # transition to "error" so the operator sees what's missing.
    assert states["box_trigger"] == "error"


def test_box_devices_json_roster_replaces_static_rows_and_marks_live(tmp_path):
    # BOX_DEVICES_JSON (broadcast discovery at Connect) swaps the static
    # single-box rows for one row per (discovered box × sensor); a following
    # "Box devices:" line marks them live by namespaced id.
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "cam_00", "kind": "camera", "label": "cam_00", "state": "warning", "fps": 60, "latencyMs": 0, "detail": ""},
            {"id": "box_gripper", "kind": "box_collection", "label": "g", "state": "idle", "fps": 20, "latencyMs": 0, "detail": ""},
        ],
    )
    roster = [
        {"device_id": 1, "sn": "box0", "ip": "192.168.2.61", "data_port": 15000,
         "box_id": "box0", "expected_devices": ["box_gripper", "box_imu"],
         "capability_names": ["box_gripper", "box_imu"]},
        {"device_id": 2, "sn": "box1", "ip": "192.168.2.62", "data_port": 15000,
         "box_id": "box1", "expected_devices": ["box_gripper"],
         "capability_names": ["box_gripper"]},
    ]
    gateway._apply_recorder_output(state, "BOX_DEVICES_JSON " + json.dumps(roster))

    box_ids = {d["id"] for d in state.devices if d["kind"] == "box_collection"}
    assert box_ids == {"box0/box_gripper", "box0/box_imu", "box1/box_gripper"}
    # The static unnamespaced row is gone; the camera row survives.
    assert "box_gripper" not in box_ids
    assert any(d["id"] == "cam_00" for d in state.devices)
    assert state.box_devices_roster == roster

    gateway._apply_recorder_output(state, "Box devices: box0/box_gripper, box1/box_gripper")
    states = {d["id"]: d["state"] for d in state.devices if d["kind"] == "box_collection"}
    assert states["box0/box_gripper"] == "running"
    assert states["box1/box_gripper"] == "running"
    assert states["box0/box_imu"] == "error"  # discovered but not reporting yet


def test_recorder_script_picks_thor_when_configured(tmp_path):
    repo_root = tmp_path
    (repo_root / "tools" / "thor" / "gmsl2").mkdir(parents=True)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={
            "recorder": {"script": "tools/thor/gmsl2/thor_record.py"},
            "dataset": {"repo_id": "local/test", "fps": 60},
        },
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )
    script, flag = gateway._recorder_script(state)
    assert script == repo_root / "tools" / "thor" / "gmsl2" / "thor_record.py"
    assert flag == "--config-path"


def test_recorder_script_defaults_to_handheld_for_legacy_configs(tmp_path):
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )
    script, flag = gateway._recorder_script(state)
    assert script == tmp_path / "tools" / "handheld" / "handheld_record.py"
    assert flag == "--config_path"


def test_device_statuses_include_camera_resolution_and_ports():
    config = {
        "sensors": {
            "cameras": {"front": {"type": "opencv", "width": 640, "height": 480, "fps": 30}},
            "handheld_grippers": {"pika": {"type": "pika_sense", "port": "/dev/ttyUSB0", "fps": 120}},
        }
    }

    devices = gateway._device_statuses(config)

    front = next(device for device in devices if device["id"] == "front")
    pika = next(device for device in devices if device["id"] == "pika")
    assert front["detail"] == "640x480"
    assert pika["label"] == "pika_sense /dev/ttyUSB0"
    assert pika["fps"] == 120


def test_box_live_output_updates_preview_without_log_noise(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )

    gateway._apply_recorder_output(
        state,
        'BOX_LIVE {"sensors":{"box_touch_left":{"timestamp":7,"fz_0p1N":[1,2,3]},"box_six_d_force":{"fxyz_mxyz":[1,2,3,4,5,6]}},"status":{"queue_size":2},"received_at_s":12.5}',
    )

    payload = gateway._box_preview_payload(state, "box_touch_left")
    assert payload["active"] is True
    assert payload["sensor"]["timestamp"] == 7
    assert payload["status"]["queue_size"] == 2
    assert state.recording.lastOutput == ""
    assert state.recording.recentOutput == []
    assert state.events == []


def test_recorder_tlv_ignored_output_is_noise(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )

    gateway._apply_recorder_output(state, "[liwp][box] tlv ignored: type=0x7")

    assert state.recording.lastOutput == ""
    assert state.recording.recentOutput == []
    assert state.events == []


def test_recorder_output_updates_status_and_event_log(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 30, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )

    gateway._apply_recorder_output(state, "Recorded 42 frames for the current episode.")
    gateway._apply_recorder_output(state, "Episode saved. Total saved episodes: 3/unlimited")

    assert state.recording.frameIndex == 0
    assert state.recording.savedEpisodes == 3
    assert state.recording.episodeIndex == 3
    assert state.recording.state == "saving"
    assert state.events[0].message.startswith("recorder: Episode saved")


def test_recorder_output_marks_connected_and_failed_devices(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 30, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "front", "kind": "camera", "label": "front", "state": "warning", "fps": 30, "latencyMs": 0, "detail": ""},
            {"id": "side", "kind": "camera", "label": "side", "state": "warning", "fps": 30, "latencyMs": 0, "detail": ""},
            {"id": "pika", "kind": "handheld_gripper", "label": "pika", "state": "warning", "fps": 120, "latencyMs": 0, "detail": ""},
        ],
    )

    gateway._apply_recorder_output(state, "Cameras: front")
    gateway._apply_recorder_output(state, "Handheld grippers: pika")
    gateway._apply_recorder_output(state, "Episode 1 ready")

    device_states = {device["id"]: device["state"] for device in state.devices}
    assert device_states == {"front": "running", "side": "error", "pika": "running"}
    assert state.recording.state == "armed"


def test_recorder_output_marks_gmsl2_failed_stream_camera_error(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test", state="connecting"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "cam_02", "kind": "camera", "label": "cam_02", "state": "running", "fps": 60, "latencyMs": 0, "detail": ""},
            {"id": "cam_03", "kind": "camera", "label": "cam_03", "state": "running", "fps": 60, "latencyMs": 0, "detail": ""},
        ],
    )

    gateway._apply_recorder_output(
        state,
        "2026-06-03 09:21:35,120 WARNING persistent_session "
        "[cam_00] failed to reach PLAYING (+12.01s)",
    )

    device_states = {device["id"]: device["state"] for device in state.devices}
    assert device_states == {"cam_02": "running", "cam_03": "running"}
    assert state.recording.state == "error"

    gateway._apply_recorder_output(
        state,
        "2026-06-03 09:15:39,981 WARNING persistent_session "
        "connect stable window failed after sid=2 first pass: "
        "cam_02(bus EOS (upstream stopped delivering buffers))",
    )

    device_states = {device["id"]: device["state"] for device in state.devices}
    assert device_states == {"cam_02": "error", "cam_03": "running"}
    assert state.recording.state == "error"

    gateway._apply_recorder_output(
        state,
        "2026-06-02 03:39:27,829 WARNING persistent_session connect() partial success: "
        "10/11 streams up; failed: cam_03(bus EOS (upstream stopped delivering buffers))",
    )

    device_states = {device["id"]: device["state"] for device in state.devices}
    assert device_states == {"cam_02": "error", "cam_03": "error"}
    assert state.recording.state == "error"


def test_snapshot_replays_recent_recorder_failures_into_red_status(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(
            repoId="local/test",
            state="connecting",
            recentOutput=[
                "2026-06-03 09:21:49,130 WARNING persistent_session "
                "[cam_02] failed to reach PLAYING (+26.03s)",
            ],
        ),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "cam_02", "kind": "camera", "label": "cam_02", "state": "running", "fps": 60, "latencyMs": 0, "detail": ""},
        ],
    )

    snapshot = gateway._snapshot(state)

    assert snapshot["recording"]["state"] == "error"
    assert snapshot["devices"][0]["state"] == "error"


def test_recorder_error_state_recovers_when_episode_ready(tmp_path):
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 60, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test", state="connecting"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        devices=[
            {"id": "cam_02", "kind": "camera", "label": "cam_02", "state": "running", "fps": 60, "latencyMs": 0, "detail": ""},
        ],
    )

    gateway._apply_recorder_output(state, "NvBufSurfaceFromFd Failed")
    assert state.recording.state == "error"

    gateway._apply_recorder_output(state, "Episode 0 ready")
    assert state.recording.state == "armed"


def test_recorder_env_adds_repo_import_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    env = gateway._recorder_env(tmp_path)
    paths = env["PYTHONPATH"].split(":")

    assert paths[:2] == [str(tmp_path / "src"), str(tmp_path)]
    assert paths[2] == "/existing/path"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_mujoco_replay_command_uses_selected_cube_sidecar_and_episode(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "fr3_sim_record_20260421_072232"
    dataset_root.mkdir(parents=True)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 30, "episode_time_s": 10}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", episode=2, fps=60),
    )

    command = gateway._mujoco_replay_command(state, dataset_root)

    assert command[1] == str(
        repo_root
        / "third_party"
        / "opencv_kalibr"
        / "fr3_data_collection_replay"
        / "replay_cube_pose_in_robot_base_mujoco.py"
    )
    assert command[command.index("--dataset-root") + 1] == str(dataset_root)
    assert command[command.index("--cube") + 1] == "left"
    assert command[command.index("--episode-index") + 1] == "2"
    assert command[command.index("--fps") + 1] == "60"
    assert command[command.index("--robot-spacing-m") + 1] == "0.9"
    assert "--no-viewer" in command
    report_path = Path(command[command.index("--report-json") + 1])
    assert report_path.name == "mujoco_preview.left.episode_000002.json"
    video_path = Path(command[command.index("--render-video") + 1])
    assert video_path.name == "mujoco_preview.left.episode_000002.mp4"

    both_command = gateway._mujoco_replay_command(state, dataset_root, "both")
    assert both_command[both_command.index("--cube") + 1] == "both"
    both_report = Path(both_command[both_command.index("--report-json") + 1])
    assert both_report.name == "mujoco_preview.both.episode_000002.json"
    both_video = Path(both_command[both_command.index("--render-video") + 1])
    assert both_video.name == "mujoco_preview.both.episode_000002.mp4"


def test_save_annotation_persists_episode_metadata(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    dataset_root.mkdir(parents=True)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "single_task": "Pick up the cube", "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset=str(dataset_root), datasetRoot=str(dataset_root), episode=3),
        selected_replay_root=dataset_root,
    )
    monkeypatch.setattr(gateway, "_resolve_known_dataset", lambda _state, _path: dataset_root)

    gateway._save_annotation(
        state,
        {
            "datasetRoot": str(dataset_root),
            "episode": 3,
            "taskPrompt": "Pick up the red cube and place it in the fixture",
            "outcome": "success",
            "quality": "good",
            "includeInTraining": True,
            "tags": ["red-cube", "clean"],
            "notes": "smooth trajectory",
            "annotator": "operator-a",
        },
    )

    annotation = gateway._active_annotation(state)

    assert annotation["source"] == "manual"
    assert annotation["taskPrompt"] == "Pick up the red cube and place it in the fixture"
    assert annotation["outcome"] == "success"
    assert annotation["quality"] == "good"
    assert annotation["includeInTraining"] is True
    assert annotation["tags"] == ["red-cube", "clean"]
    assert (dataset_root / "meta" / "gui_annotations.json").is_file()


def test_replay_episode_selection_defaults_to_first_and_can_switch(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=3)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    gateway._select_replay_dataset(state, str(dataset_root))
    first_timeline = gateway._read_dataset_timeline(state, dataset_root)
    gateway._select_replay_episode(state, "1")
    selected_timeline = gateway._read_dataset_timeline(state, dataset_root)

    assert state.replay.episodeOptions == [0, 1, 2]
    assert first_timeline["episode"] == 0
    assert selected_timeline["episode"] == 1
    assert [frame["frame"] for frame in selected_timeline["frames"]] == [0, 1]


def test_processing_items_accepts_dataset_root_as_datasets_root(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(repo_root / "missing"), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test", datasetRoot=str(repo_root / "missing")),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root,
    )

    items = gateway._processing_items(state)

    assert [item["path"] for item in items] == [str(dataset_root)]
    assert items[0]["name"] == "episode_set"


def test_set_datasets_root_creates_missing_directory(tmp_path):
    repo_root = tmp_path / "repo"
    missing_root = repo_root / "data"
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )

    created = gateway._set_datasets_root(state, str(missing_root))

    assert created is True
    assert missing_root.is_dir()
    assert state.datasets_root == missing_root.resolve()


def test_replay_timeline_includes_generated_cube_pose_sidecars(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=2)
    sidecar = dataset_root / "derived" / gateway.DEFAULT_TRAJ_SIDECAR_NAME
    sidecar.mkdir(parents=True)
    for cube, offset in (("left", 0.0), ("right", 0.1), ("head", 0.2)):
        (sidecar / f"state_action.{cube}.csv").write_text(
            "\n".join(
                [
                    "episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw",
                    f"1,0,{0.3 + offset},0.0,0.2,0.0,0.0,0.0,1.0",
                    f"1,1,{0.31 + offset},0.0,0.21,0.0,0.0,0.0,1.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    timeline = gateway._read_dataset_timeline(state, dataset_root, episode=1)

    assert timeline["cubePoseNames"] == ["left", "right", "head"]
    assert timeline["frames"][0]["cubePoses"]["left"]["x"] == 0.3
    assert timeline["frames"][1]["cubePoses"]["right"]["x"] == pytest.approx(0.41)
    assert timeline["frames"][0]["cubePoses"]["head"]["qw"] == 1.0


def test_gmsl2_timeline_surfaces_ee_pose_sidecar_without_v3_parquet(tmp_path):
    # Camera-only (--no-box) datasets have raw episodes but no data/chunk parquet,
    # so _read_dataset_timeline takes the _read_gmsl2_timeline path. The AprilTag
    # EE-pose sidecar produced by run_april_cube_tracking_* (no v3 parquet needed)
    # must still reach the replay view, with timestamps on the PWM-synced camera
    # grid (pts_offset + N/fps).
    dataset_root = tmp_path / "outputs" / "datasets" / "cam_only_set"
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    t0 = 1000.0
    ep_dir.joinpath("meta.json").write_text(
        json.dumps(
            {
                "duration_s": 0.05,  # 0.05 * 60 -> 3 frames
                "video": {"fps": 60},
                "sync_reference": {
                    "t0_wall_s": t0,
                    # mean(first_wall - t0) = (0.01 + 0.03) / 2 = 0.02
                    "camera_first_wall_s": {"cam_00": t0 + 0.01, "cam_01": t0 + 0.03},
                },
            }
        ),
        encoding="utf-8",
    )
    ep_dir.joinpath("cam_00.mkv").write_bytes(b"\0" * 2048)  # >1024 -> counts as a camera
    sidecar = dataset_root / "derived" / gateway.DEFAULT_TRAJ_SIDECAR_NAME
    sidecar.mkdir(parents=True)
    (sidecar / "state_action.left.csv").write_text(
        "\n".join(
            [
                "episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw",
                "0,0,0.30,0.0,0.20,0.0,0.0,0.0,1.0",
                "0,1,0.31,0.0,0.21,0.0,0.0,0.0,1.0",
                "0,2,0.32,0.0,0.22,0.0,0.0,0.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    # No v3 parquet -> the gmsl2 timeline path is exercised.
    assert gateway._has_gmsl2_episodes(dataset_root)
    assert not gateway._has_lerobot_v3_data(dataset_root)

    timeline = gateway._read_dataset_timeline(state, dataset_root, episode=0)

    assert timeline["cubePoseNames"] == ["left"]
    assert timeline["frames"][0]["cubePoses"]["left"]["x"] == pytest.approx(0.30)
    assert timeline["frames"][2]["cubePoses"]["left"]["x"] == pytest.approx(0.32)
    # PWM grid: timestamp = pts_offset(0.02) + N / fps(60)
    assert timeline["frames"][0]["timestamp"] == pytest.approx(0.02)
    assert timeline["frames"][1]["timestamp"] == pytest.approx(0.02 + 1.0 / 60.0)


def test_replay_timeline_includes_camera_cube_overlays(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    tracking_run = tmp_path / "outputs" / "tracking_analysis" / f"episode_set{gateway.DEFAULT_TRACKING_RUN_SUFFIX}"
    per_camera = tracking_run / "per_camera"
    per_camera.mkdir(parents=True)
    intrinsics = tmp_path / "calibration" / "intrinsics.json"
    intrinsics.parent.mkdir(parents=True)
    fixed_summary = tmp_path / "calibration" / "fixed_summary.json"
    intrinsics.write_text(
        json.dumps({"camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]}),
        encoding="utf-8",
    )
    fixed_summary.write_text(
        json.dumps(
            {
                "joint_solution": {
                    "status": "ok",
                    "cameras": {
                        "hk_01": {
                            "base_to_camera": {
                                "matrix_4x4": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tracking_run / "summary.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "calibration_inputs": {"fixed_camera_summary": str(fixed_summary)},
                "cube_tracker": {"cube_size_cm": 7.0},
                "active_streams": [
                    {
                        "stream_key": "cam_0",
                        "camera_name": "hk_01",
                        "serial": "S1",
                        "intrinsics_path": str(intrinsics),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (per_camera / "camera_S1_records.csv").write_text(
        "\n".join(
            [
                "frame_global_index,episode_index,frame_index,cube_name,camera_serial,camera_name,stream_key,cube_detected,cube_num_markers,cube_reprojection_rmse_px,used_for_fusion,cube_base_x_m,cube_base_y_m,cube_base_z_m,cube_base_qx,cube_base_qy,cube_base_qz,cube_base_qw",
                "0,0,0,left,S1,hk_01,cam_0,1,2,1.25,1,0.0,0.0,1.0,0.0,0.0,0.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    timeline = gateway._read_dataset_timeline(state, dataset_root, episode=0)
    overlay = timeline["frames"][0]["videoOverlays"]["observation.images.cam_0"][0]

    assert overlay["cubeName"] == "left"
    assert overlay["numMarkers"] == 2
    assert overlay["usedForFusion"] is True
    assert overlay["corners"][0] is not None
    assert overlay["axes"]["origin"] == pytest.approx([320.0, 240.0])


def test_traj_gen_starts_april_tracking_with_selected_dataset_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    runner_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_RUNNER
    config_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_CONFIG
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("#!/usr/bin/env bash\necho tracking\n", encoding="utf-8")
    config_path.write_text("input:\n  dataset_root: /wrong/from/yaml\n", encoding="utf-8")
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 1234
        stdout = []

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway, "_start_traj_gen_output_reader", lambda *_args: None)

    gateway._queue_traj_gen(state, dataset_root)

    command = launched["command"]
    assert command[0] == "bash"
    assert str(runner_path) in command
    assert "--config" in command
    assert str(config_path) in command
    assert "--dataset-root" in command
    assert command[command.index("--dataset-root") + 1] == str(dataset_root)
    assert str(dataset_root) in state.processing_processes
    item = gateway._processing_item_from_dataset(dataset_root)
    assert item["status"] == "running"
    assert "AprilTag cube tracking" in item["message"]


def test_exported_v3_dataset_is_replay_selectable_but_blocks_real_robot(tmp_path):
    repo_root = tmp_path / "repo"
    recorded_root = repo_root / "outputs" / "datasets" / "recorded_set"
    exported_root = repo_root / "outputs" / "exports" / "exported_set"
    _write_minimal_episode_dataset(recorded_root, total_episodes=1)
    _write_minimal_episode_dataset(exported_root, total_episodes=2)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(recorded_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", fps=30),
        datasets_root=recorded_root.parent,
        exports_root=exported_root.parent,
    )

    replay_items = gateway._recorded_dataset_items(state)
    exported_item = next(item for item in replay_items if item["path"] == str(exported_root))
    assert exported_item["datasetKind"] == "exported"

    processing_paths = {item["path"] for item in gateway._processing_items(state)}
    assert str(recorded_root) in processing_paths
    assert str(exported_root) not in processing_paths

    gateway._select_replay_dataset(state, str(exported_root))
    assert state.replay.datasetRoot == str(exported_root)
    assert state.replay.datasetKind == "exported"
    assert state.replay.episodeOptions == [0, 1]
    snapshot = gateway._snapshot(state)
    assert snapshot["replay"]["datasetRoot"] == str(exported_root)
    assert snapshot["replay"]["datasetKind"] == "exported"

    timeline = gateway._read_dataset_timeline(state, exported_root, episode=1)
    assert timeline["datasetKind"] == "exported"
    assert timeline["episode"] == 1
    assert timeline["totalFrames"] == 2

    with pytest.raises(RuntimeError, match="Real-robot replay is disabled for exported datasets"):
        gateway._require_mujoco_validation(state)


def test_replay_realsense_preview_paths_keep_full_camera_key(tmp_path):
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "fr3.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
    )

    image_path, status_path = gateway._realsense_preview_paths(state, "observation.images.ee")

    assert image_path.name.endswith("_observation_images_ee.jpg")
    assert status_path.name.endswith("_observation_images_ee.json")


def test_replay_realsense_matches_dataset_camera_keys_to_configured_serials(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "fr3_spacemouse"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.ee": {"dtype": "video"},
                    "observation.images.side": {"dtype": "video"},
                    "observation.images.unconfigured": {"dtype": "video"},
                }
            }
        ),
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "fr3.yaml",
        config={
            "robot": {
                "cameras": {
                    "ee": {
                        "type": "IntelRealSense",
                        "serial_number_or_name": "RS_EE",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                    "side": {
                        "type": "IntelRealSense",
                        "serial_number_or_name": "RS_SIDE",
                        "width": 848,
                        "height": 480,
                        "fps": 6,
                    },
                    "other": {
                        "type": "OpenCV",
                        "serial_number_or_name": "NOT_RS",
                    },
                }
            }
        },
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(dataset=str(dataset_root), datasetRoot=str(dataset_root)),
        selected_replay_root=dataset_root,
        profile="workstation",
    )

    matches = gateway._replay_realsense_camera_matches(state, dataset_root)

    assert [(match["cameraKey"], match["configKey"], match["serial"]) for match in matches] == [
        ("observation.images.ee", "ee", "RS_EE"),
        ("observation.images.side", "side", "RS_SIDE"),
    ]
    assert matches[0]["fps"] == 15
    assert matches[1]["fps"] == 6


def test_replay_realsense_status_reports_all_matched_dataset_cameras(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "fr3_spacemouse"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.images.ee": {"dtype": "video"}}}),
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "fr3.yaml",
        config={
            "robot": {
                "cameras": {
                    "ee": {"type": "IntelRealSense", "serial_number_or_name": "RS_EE"},
                }
            }
        },
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(dataset=str(dataset_root), datasetRoot=str(dataset_root)),
        selected_replay_root=dataset_root,
        profile="workstation",
    )

    status = gateway._realsense_preview_status(state)

    assert status["running"] is False
    assert status["cameras"] == [
        {
            "available": None,
            "running": False,
            "error": "Preview starts with real-robot replay",
            "cameraKey": "observation.images.ee",
            "configKey": "ee",
            "serial": "RS_EE",
        }
    ]


def test_mujoco_validation_is_recommended_for_preflight_but_required_for_real_replay(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=2)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={
            "dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30},
            "replay": {
                "robot_ip": "192.168.1.99",
                "gripper_port": "/dev/ttyUSB9",
                "real_preflight_enabled": False,
            },
        },
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", fps=30),
        datasets_root=dataset_root.parent,
    )

    gateway._select_replay_dataset(state, str(dataset_root))
    gateway._preflight_replay(state)
    gateway._start_dry_run_replay(state)
    assert "strongly recommended" in state.replay.message
    assert state.replay.state == "dry_run"
    assert state.replay.safety == "ready"

    try:
        gateway._require_mujoco_validation(state)
    except RuntimeError as exc:
        assert "MuJoCo validation required" in str(exc)
    else:
        raise AssertionError("real replay should require a passed MuJoCo validation")

    state.replay.mujocoValidation = gateway._new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=0,
    )
    gateway._apply_mujoco_replay_output(
        state,
        "mujoco_replay_result status=complete completed_frames=2 total_frames=2 "
        "avg_pos_mm=2.0 max_pos_mm=4.0 avg_rot_deg=1.0 max_rot_deg=3.0",
    )
    gateway._finish_mujoco_validation(state, 0)
    gateway._preflight_replay(state)
    command = gateway._real_replay_command(
        state,
        dataset_root,
        "left",
        "192.168.1.99",
    )

    assert state.replay.mujocoValidation["status"] == "passed"
    assert state.replay.safety == "ready"
    assert "current MuJoCo validation" in state.replay.message
    assert command[1].endswith(
        "third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py"
    )
    assert "--input.source=csv" in command
    assert f"--input.csv_path={dataset_root}/derived/april_cube_tracking_in_robot_base/state_action.left.csv" in command
    assert "--input.dataset_pose_name=left" in command
    assert "--robot.robot_ip=192.168.1.99" in command
    assert "--replay.episode_index=0" in command
    assert "--replay.initial_pose_mode=current" in command
    assert "--replay.fail_on_unreached_initial_pose=true" in command
    assert "--end_effector.mode=robot_config" in command

    pika_command = gateway._real_replay_command(
        state,
        dataset_root,
        "left",
        "192.168.1.206",
        "pika_gripper_ee",
    )
    assert "--robot.gripper_backend=pika" in pika_command
    assert "--robot.allow_mock_gripper=false" in pika_command
    assert "--robot.target_frame_name=pika_task_tcp" in pika_command
    assert any(part.endswith("fr3_pika_gripper.urdf") for part in pika_command)

    bare_command = gateway._real_replay_command(
        state,
        dataset_root,
        "left",
        "192.168.1.99",
        "fr3_ee",
    )
    assert "--end_effector.mode=fr3_ee" in bare_command

    preflight_command = gateway._real_preflight_command(state, "192.168.1.99")
    assert preflight_command[0] == str(gateway._mujoco_replay_python(state))
    assert (
        f"--config-path={repo_root}/third_party/opencv_kalibr/"
        "fr3_data_collection_replay/replay_cube_pose_in_robot_base.thor.yaml"
    ) in preflight_command
    assert "--skip-host-imports" in preflight_command
    assert "--skip-hikrobot" in preflight_command
    assert "--skip-gripper" in preflight_command


def test_approve_mujoco_report_rechecks_metrics_instead_of_bypassing_failure(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    sidecar_dir = dataset_root / "derived" / gateway.DEFAULT_TRAJ_SIDECAR_NAME
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "state_action.left.csv").write_text(
        "episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw\n"
        "0,0,0.30,0.0,0.20,0.0,0.0,0.0,1.0\n"
        "0,1,0.301,0.0,0.20,0.0,0.0,0.0,1.0\n",
        encoding="utf-8",
    )
    report_path = gateway._mujoco_preview_report_path(dataset_root, 0, "left")
    video_path = gateway._mujoco_preview_video_path(dataset_root, 0, "left")
    report_path.write_text(
        json.dumps({
            "dataset_root": str(dataset_root),
            "cube_mode": "left",
            "episode_index": 0,
            "fps": 30,
            "robots": {
                "left": {
                    "frames": [{"frame_index": 0}, {"frame_index": 1}],
                    "metrics": {
                        "avg_position_error_mm": 23.6,
                        "max_position_error_mm": 104.3,
                        "avg_rotation_error_deg": 6.2,
                        "max_rotation_error_deg": 29.3,
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    video_path.write_bytes(b"rendered")
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(
            dataset=str(dataset_root), datasetRoot=str(dataset_root), episode=0, fps=30,
            totalFrames=2, recordedFrames=4, mujocoCubeMode="left",
        ),
        selected_replay_root=dataset_root,
    )

    gateway._approve_mujoco_report(state, "left")

    assert state.replay.mujocoValidation["status"] == "failed"
    assert state.replay.mujocoValidation["maxPositionErrorMm"] == pytest.approx(104.3)
    assert state.replay.safety == "fault"
    with pytest.raises(RuntimeError, match="MuJoCo validation required"):
        gateway._require_mujoco_validation(state)
    assert gateway._require_mujoco_validation(
        state,
        cube_mode="left",
        allow_failed_override=True,
    ) == dataset_root.resolve()

    passing_report = json.loads(report_path.read_text(encoding="utf-8"))
    passing_report["robots"]["left"]["metrics"].update({
        "avg_position_error_mm": 2.0,
        "max_position_error_mm": 4.0,
        "avg_rotation_error_deg": 1.0,
        "max_rotation_error_deg": 3.0,
    })
    report_path.write_text(json.dumps(passing_report), encoding="utf-8")
    video_path.write_bytes(b"passing-render")

    gateway._approve_mujoco_report(state, "left")

    assert state.replay.mujocoValidation["status"] == "passed"
    assert gateway._require_mujoco_validation(state) == dataset_root.resolve()


def test_run_qc_includes_fr3_ik_result_in_overall_status(monkeypatch, tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    monkeypatch.setattr(
        gateway,
        "_run_fr3_ik_qc",
        lambda *args, **kwargs: {
            "status": "fail",
            "message": "left: 0/1 trajectories reachable; 80.00% poses reachable",
            "cubes": [{"cube": "left", "numUnreachableTrajectories": 1, "reachableRatio": 0.8}],
        },
    )

    qc = gateway._run_qc(dataset_root, repo_root=tmp_path, ik_python=Path(sys.executable))

    assert qc["status"] == "fail"
    assert qc["ik_evaluation"]["cubes"][0]["cube"] == "left"
    ik_check = next(check for check in qc["checks"] if check["name"] == "fr3_ik_reachability")
    assert ik_check["status"] == "fail"


def test_run_qc_preserves_virtualenv_python_symlink(monkeypatch, tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    base_python = tmp_path / "python-base"
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "fr3-venv" / "bin" / "python3"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    captured: dict[str, Path] = {}

    def fake_ik_qc(*args, **kwargs):
        del args
        captured["python"] = kwargs["python_executable"]
        return {"status": "skipped", "message": "not needed", "cubes": []}

    monkeypatch.setattr(gateway, "_run_fr3_ik_qc", fake_ik_qc)

    gateway._run_qc(dataset_root, repo_root=tmp_path, ik_python=venv_python)

    assert captured["python"] == venv_python
    assert captured["python"] != venv_python.resolve()


def test_fr3_ik_qc_runs_each_available_arm_sidecar(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    sidecar_dir = dataset_root / "derived" / gateway.DEFAULT_TRAJ_SIDECAR_NAME
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "state_action.left.csv").write_text(
        "state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw\n"
        "0.4,0.0,0.3,0.0,0.0,0.0,1.0\n",
        encoding="utf-8",
    )
    script = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.py"
    config = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.thor.yaml"
    script.parent.mkdir(parents=True)
    script.write_text("# verifier\n", encoding="utf-8")
    config.write_text("robot: {}\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        del kwargs
        report_path = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--validation.report_json_path=")))
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps({
            "summary": {
                "num_targets": 10,
                "num_unreachable": 2,
                "reachable_ratio": 0.8,
                "reason_counts": {"ok": 8, "fk_residual": 2},
                "ik_error_stats": {"mean_position_error_m": 0.004},
                "episode_summary": [{
                    "episode_index": 3,
                    "num_targets": 10,
                    "num_reachable": 8,
                    "num_unreachable": 2,
                    "reachable_ratio": 0.8,
                    "trajectory_reachable": True,
                    "ik_trajectory_label": "reachable",
                    "unreachable_duration_s": 0.1,
                    "max_consecutive_unreachable_timesteps": 2,
                    "max_position_error_m": 0.004,
                    "max_orientation_error_deg": 1.5,
                }],
                "trajectory_reachability": {
                    "total_trajectories": 1,
                    "num_unreachable_trajectories": 0,
                },
            }
        }), encoding="utf-8")
        report_path.with_name("verify_fr3_cube_pose_ik_error_over_time.png").write_bytes(b"plot")
        return subprocess.CompletedProcess(command, 0, stdout="Offline FR3 IK validation finished")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    result = gateway._run_fr3_ik_qc(
        dataset_root,
        repo_root=repo_root,
        python_executable=Path(sys.executable),
        fps=60,
    )

    assert result["status"] == "warn"
    assert [cube["cube"] for cube in result["cubes"]] == ["left"]
    assert result["cubes"][0]["reachableRatio"] == pytest.approx(0.8)
    assert result["cubes"][0]["reachableEpisodeIndices"] == [3]
    assert result["cubes"][0]["unreachableEpisodeIndices"] == []
    assert result["cubes"][0]["episodes"][0]["maxPositionErrorMm"] == pytest.approx(4.0)
    assert result["cubes"][0]["plotAvailable"] is True


def test_fr3_ik_qc_skips_sidecar_without_finite_poses(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    sidecar_dir = dataset_root / "derived" / gateway.DEFAULT_TRAJ_SIDECAR_NAME
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "state_action.right.csv").write_text(
        "state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw\n"
        "nan,nan,nan,nan,nan,nan,nan\n",
        encoding="utf-8",
    )
    script = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.py"
    config = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.thor.yaml"
    script.parent.mkdir(parents=True)
    script.write_text("# verifier\n", encoding="utf-8")
    config.write_text("robot: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verifier must not run")),
    )

    result = gateway._run_fr3_ik_qc(
        dataset_root,
        repo_root=repo_root,
        python_executable=Path(sys.executable),
        fps=60,
    )

    assert result["status"] == "pass"
    assert result["cubes"] == [{
        "cube": "right",
        "status": "skipped",
        "message": "no finite EE target poses in trajectory sidecar",
    }]


def test_real_replay_rejects_two_cube_mode(tmp_path):
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
    )

    with pytest.raises(ValueError, match="must be left or right"):
        gateway._start_real_replay(state, "both", "192.168.1.99")


def test_real_preflight_failure_is_preserved_in_panel_log(monkeypatch, tmp_path):
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "camera_config.yaml",
        config={"replay": {"real_preflight_enabled": True}},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
    )
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="[PASS] fr3_ping: reachable\n[FAIL] fr3_arm: connection refused\n"
        ),
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        gateway._run_real_preflight(state, ["192.168.1.99"])

    joined = "\n".join(state.replay.realReplayLog)
    assert "replay_cube_pose_in_robot_base.thor.yaml" in joined
    assert "[PASS] fr3_ping: reachable" in joined
    assert "[FAIL] fr3_arm: connection refused" in joined


def _gripper_jump_contract_state(tmp_path: Path, *, profile: str) -> tuple[gateway.GatewayState, Path]:
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", fps=30, episode=0),
        datasets_root=dataset_root.parent,
        profile=profile,
    )
    return state, dataset_root


def test_workstation_trajectory_contract_allows_full_gripper_travel(monkeypatch, tmp_path):
    state, dataset_root = _gripper_jump_contract_state(tmp_path, profile="workstation")
    monkeypatch.setattr(
        gateway,
        "_read_dataset_timeline",
        lambda *_args, **_kwargs: {
            "frames": [
                {"eePose": {"x": 0.3, "y": 0.0, "z": 0.2, "gripper": 0.0}},
                {"eePose": {"x": 0.301, "y": 0.0, "z": 0.2, "gripper": 1.0}},
            ]
        },
    )

    contract = gateway._trajectory_contract_for_episode(state, dataset_root)

    assert contract["status"] == "passed"
    gripper_check = next(check for check in contract["checks"] if check["name"] == "gripper_range_step")
    assert gripper_check["maxStep"] == 1.0
    assert gripper_check["stepThreshold"] == 1.0


def test_thor_trajectory_contract_keeps_conservative_gripper_step(monkeypatch, tmp_path):
    state, dataset_root = _gripper_jump_contract_state(tmp_path, profile="thor")
    monkeypatch.setattr(
        gateway,
        "_read_dataset_timeline",
        lambda *_args, **_kwargs: {
            "frames": [
                {"eePose": {"x": 0.3, "y": 0.0, "z": 0.2, "gripper": 0.0}},
                {"eePose": {"x": 0.301, "y": 0.0, "z": 0.2, "gripper": 1.0}},
            ]
        },
    )

    contract = gateway._trajectory_contract_for_episode(state, dataset_root)

    assert contract["status"] == "failed"
    assert "max gripper step 1.000 > 0.350" in contract["failures"]


def test_mujoco_validation_fails_when_metrics_exceed_threshold(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={
            "dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30},
            "replay": {"mujoco_max_position_error_mm": 3.0, "mujoco_max_rotation_error_deg": 5.0},
        },
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", fps=30),
        datasets_root=dataset_root.parent,
    )

    gateway._select_replay_dataset(state, str(dataset_root))
    state.replay.mujocoValidation = gateway._new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=0,
    )
    gateway._apply_mujoco_replay_output(
        state,
        "mujoco_replay_result=status=complete completed_frames=2 total_frames=2 "
        "avg_pos_mm=2.0 max_pos_mm=4.0 avg_rot_deg=1.0 max_rot_deg=3.0",
    )
    gateway._finish_mujoco_validation(state, 0)

    assert state.replay.mujocoValidation["status"] == "failed"
    assert "max position error" in state.replay.mujocoValidation["message"]
    assert state.replay.safety == "fault"


def test_mujoco_validation_requires_structured_result(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test", fps=30),
        datasets_root=dataset_root.parent,
    )

    gateway._select_replay_dataset(state, str(dataset_root))
    state.replay.mujocoValidation = gateway._new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=0,
    )
    gateway._apply_mujoco_replay_output(state, "  最大位置误差: 1.00 mm")
    gateway._apply_mujoco_replay_output(state, "  最大旋转误差: 1.00 deg")
    gateway._finish_mujoco_validation(state, 0)

    assert state.replay.mujocoValidation["status"] == "failed"
    assert "missing structured mujoco_replay_result" in state.replay.mujocoValidation["message"]


def test_count_completed_episodes_strips_namespace_and_matches_timestamped_dirs(tmp_path):
    repo_root = tmp_path / "repo"
    datasets_root = repo_root / "outputs" / "datasets"
    _write_minimal_episode_dataset(datasets_root / "pick_and_place", total_episodes=3)
    _write_minimal_episode_dataset(datasets_root / "pick_and_place_20260528_103422", total_episodes=2)
    _write_minimal_episode_dataset(datasets_root / "fold_towel", total_episodes=5)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
        datasets_root=datasets_root,
    )

    # "local/" namespace stripped; exact + timestamped captures counted, others ignored.
    assert gateway._count_completed_episodes(state, "local/pick_and_place") == 5
    assert gateway._count_completed_episodes(state, "local/fold_towel") == 5
    assert gateway._count_completed_episodes(state, "") == 0


def test_tasks_with_progress_reflects_recorded_episodes(tmp_path):
    repo_root = tmp_path / "repo"
    datasets_root = repo_root / "outputs" / "datasets"
    _write_minimal_episode_dataset(datasets_root / "pick_and_place", total_episodes=4)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
        datasets_root=datasets_root,
    )
    gateway._create_task(
        state,
        {
            "name": "Pick and Place",
            "targetEpisodes": 100,
            "datasetRepoId": "local/pick_and_place",
            "completedEpisodes": 0,
        },
    )

    tasks = gateway._tasks_with_progress(state)

    assert len(tasks) == 1
    assert tasks[0]["completedEpisodes"] == 4
    assert tasks[0]["targetEpisodes"] == 100


def _task_state(tmp_path):
    repo_root = tmp_path / "repo"
    datasets_root = repo_root / "outputs" / "datasets"
    datasets_root.mkdir(parents=True)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={
            "dataset": {
                "repo_id": "local/thor_gmsl2_Nch_v1",
                "root": "outputs/datasets/thor_gmsl2_Nch_v1",
                "single_task": "default capture",
                "fps": 60,
            },
            "cameras": {"width": 1920, "height": 1080},
        },
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
        datasets_root=datasets_root,
    )
    return state, datasets_root


def test_build_task_overlay_patches_only_dataset_and_aligns_root_with_repo_id(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    task = {
        "id": "task-1",
        "name": "Pick and Place",
        "description": "pick the cube",
        "datasetRepoId": "local/pick_and_place",
    }

    overlay = gateway._build_task_overlay_config(state.config, task, datasets_root)

    assert overlay["dataset"]["repo_id"] == "local/pick_and_place"
    # root basename must equal repo_id trailing segment so episodes are counted.
    assert Path(overlay["dataset"]["root"]).name == "pick_and_place"
    assert overlay["dataset"]["single_task"] == "pick the cube"
    # non-dataset blocks untouched; base config not mutated.
    assert overlay["cameras"] == state.config["cameras"]
    assert state.config["dataset"]["repo_id"] == "local/thor_gmsl2_Nch_v1"


def test_resolve_recorder_config_path_uses_overlay_for_active_task(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    gateway._create_task(
        state,
        {
            "name": "Pick and Place",
            "targetEpisodes": 100,
            "datasetRepoId": "local/pick_and_place",
        },
    )
    task_id = gateway._read_tasks(state)[0]["id"]
    gateway._set_active_task(state, task_id)

    config_path = gateway._resolve_recorder_config_path(state)

    assert config_path != state.config_path
    assert config_path.is_file()
    assert Path(state.recording.datasetRoot).name == "pick_and_place"
    overlay = gateway._load_yaml(config_path)
    assert overlay["dataset"]["repo_id"] == "local/pick_and_place"


def test_resolve_recorder_config_path_falls_back_without_active_task(tmp_path):
    state, _ = _task_state(tmp_path)

    config_path = gateway._resolve_recorder_config_path(state)

    assert config_path == state.config_path
    assert state.recording.datasetRoot == "outputs/datasets/thor_gmsl2_Nch_v1"


def test_set_active_task_rejects_task_without_repo_id(tmp_path):
    state, _ = _task_state(tmp_path)
    gateway._create_task(state, {"name": "No Dataset", "targetEpisodes": 10})
    task_id = gateway._read_tasks(state)[0]["id"]

    with pytest.raises(ValueError, match="dataset repo id"):
        gateway._set_active_task(state, task_id)
    assert state.active_task_id is None


def test_recorded_task_dataset_counts_toward_task_end_to_end(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    gateway._create_task(
        state,
        {"name": "Pick and Place", "targetEpisodes": 100, "datasetRepoId": "local/pick_and_place"},
    )
    task_id = gateway._read_tasks(state)[0]["id"]
    gateway._set_active_task(state, task_id)
    gateway._resolve_recorder_config_path(state)

    # Simulate the recorder writing a timestamped capture under the overlay root.
    _write_minimal_episode_dataset(datasets_root / "pick_and_place_20260601_120000", total_episodes=7)

    tasks = gateway._tasks_with_progress(state)
    assert tasks[0]["completedEpisodes"] == 7


def test_delete_active_task_clears_binding(tmp_path):
    state, _ = _task_state(tmp_path)
    gateway._create_task(
        state,
        {"name": "Pick and Place", "targetEpisodes": 100, "datasetRepoId": "local/pick_and_place"},
    )
    task_id = gateway._read_tasks(state)[0]["id"]
    gateway._set_active_task(state, task_id)

    gateway._delete_task(state, task_id)

    assert state.active_task_id is None


def test_export_command_builds_args_from_task(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    task = {
        "id": "task-1",
        "name": "Pick and Place",
        "description": "pick the cube",
        "datasetRepoId": "local/pick_and_place",
    }

    command, out_root = gateway._export_command(state, task)

    assert out_root == state.exports_root / "pick_and_place"
    assert "--base-name" in command and command[command.index("--base-name") + 1] == "pick_and_place"
    assert command[command.index("--repo-id") + 1] == "local/pick_and_place"
    assert command[command.index("--task") + 1] == "pick the cube"
    assert command[command.index("--datasets-root") + 1] == str(datasets_root)
    assert "--overwrite" in command
    assert command[1].endswith("tools/thor/gmsl2/export_v3.py")


def test_export_command_rejects_task_without_repo_id(tmp_path):
    state, _ = _task_state(tmp_path)
    with pytest.raises(ValueError, match="dataset repo id"):
        gateway._export_command(state, {"id": "t", "name": "x", "datasetRepoId": ""})


def _write_qc_pass_gmsl2_session(dataset_root: Path, cams: tuple[str, ...] = ()) -> None:
    episode = dataset_root / "episodes" / "episode_000000"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(
        json.dumps({"video": {"fps": 60, "height": 480, "width": 640}}),
        encoding="utf-8",
    )
    if cams:
        for cam in cams:
            (episode / f"{cam}.mkv").write_bytes(b"0" * 2048)
        (episode / "online_sync_manifest.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "actual_frames": 1,
                    "active_cameras": list(cams),
                    "frame_count_by_camera": {cam: 1 for cam in cams},
                }
            ),
            encoding="utf-8",
        )
    gateway._write_processing_meta(
        dataset_root,
        {
            "active_version": "v1",
            "versions": {"v1": {"qc": {"status": "pass", "summary": "ok"}}},
        },
    )


def test_approved_dataset_export_command_uses_actual_camera_count_for_output_name(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    session = datasets_root / "thor_gmsl2_11ch_v1_20260713_075106"
    cams = ("cam_00", "cam_06", "cam_07", "cam_08", "cam_09", "cam_12", "cam_13", "cam_14")
    _write_qc_pass_gmsl2_session(session, cams=cams)

    command, out_root = gateway._approved_dataset_export_command(state, session)

    assert command[command.index("--base-name") + 1] == "thor_gmsl2_11ch_v1_20260713_075106"
    assert command[command.index("--output-name") + 1] == "thor_gmsl2_8ch_v1_20260713_075106"
    assert command[command.index("--repo-id") + 1] == "local/thor_gmsl2_8ch_v1_20260713_075106"
    assert out_root == state.exports_root / "thor_gmsl2_8ch_v1_20260713_075106"


def test_approved_dataset_export_command_scopes_to_selected_session(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    session = datasets_root / "pick_and_place_20260601_101046"
    _write_qc_pass_gmsl2_session(session)
    gateway._write_tasks(
        state,
        [
            {
                "id": "task-1",
                "name": "Pick and Place",
                "description": "pick cube carefully",
                "datasetRepoId": "local/pick_and_place",
            }
        ],
    )

    command, out_root = gateway._approved_dataset_export_command(state, session)

    assert out_root == state.exports_root / "pick_and_place_20260601_101046"
    assert command[command.index("--datasets-root") + 1] == str(datasets_root)
    assert command[command.index("--base-name") + 1] == "pick_and_place_20260601_101046"
    assert command[command.index("--repo-id") + 1] == "local/pick_and_place_20260601_101046"
    assert command[command.index("--task") + 1] == "pick cube carefully"


def test_start_approved_dataset_export_copies_qc_pass_lerobot_v3_dataset(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    dataset_root = datasets_root / "approved_v3"
    _write_minimal_episode_dataset(dataset_root, total_episodes=2)
    gateway._write_processing_meta(
        dataset_root,
        {
            "active_version": "v1",
            "versions": {"v1": {"qc": {"status": "pass", "summary": "ok"}}},
        },
    )

    gateway._start_approved_dataset_export(state, str(dataset_root))

    out_root = state.exports_root / "approved_v3"
    assert state.dataset_export.state == "complete"
    assert state.dataset_export.datasetRoot == str(dataset_root)
    assert state.dataset_export.outputPath == str(out_root)
    assert state.dataset_export.selectedEpisodes == 2
    assert (out_root / "meta" / "info.json").is_file()
    assert (out_root / "data" / "chunk-000" / "file-000.parquet").is_file()


def test_start_approved_dataset_export_consolidates_raw_gmsl2_even_with_parquet(tmp_path, monkeypatch):
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    dataset_root = datasets_root / "pick_and_place_20260601_101046"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    _write_qc_pass_gmsl2_session(dataset_root)
    (dataset_root / "episodes" / "episode_000000" / "cam_00.mkv").write_bytes(b"0" * 2048)

    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4321
        stdout = []

        def poll(self):
            return None

    class FakeThread:
        def __init__(self, *args, **kwargs):
            launched["thread_args"] = args
            launched["thread_kwargs"] = kwargs

        def start(self):
            launched["thread_started"] = True

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway, "Thread", FakeThread)

    gateway._start_approved_dataset_export(state, str(dataset_root))

    command = launched["command"]
    assert command[1].endswith("tools/thor/gmsl2/export_v3.py")
    assert command[command.index("--datasets-root") + 1] == str(datasets_root)
    assert command[command.index("--base-name") + 1] == dataset_root.name
    assert "--overwrite" in command
    assert state.dataset_export.state == "exporting"
    assert state.dataset_export.outputPath == str(state.exports_root / dataset_root.name)
    assert launched["thread_started"] is True
    assert not (state.exports_root / dataset_root.name / "meta" / "info.json").exists()


def test_start_approved_dataset_export_rejects_non_qc_pass_dataset(tmp_path):
    state, datasets_root = _task_state(tmp_path)
    session = datasets_root / "pick_and_place_20260601_101046"
    episode = session / "episodes" / "episode_000000"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text("{}", encoding="utf-8")
    (episode / "cam_0.mkv").write_bytes(b"0" * 2048)

    with pytest.raises(ValueError, match="pass QC"):
        gateway._start_approved_dataset_export(state, str(session))


def test_apply_export_output_tracks_progress_and_terminal_state(tmp_path):
    state, _ = _task_state(tmp_path)
    state.dataset_export = gateway.DatasetExportStatus(state="exporting")

    gateway._apply_export_output(state, "Export plan: 9 episodes from 2 session(s) -> local/pick_and_place")
    assert state.dataset_export.selectedEpisodes == 9

    gateway._apply_export_output(state, "Episode 0 written (5 frames) from s/episode_000000")
    gateway._apply_export_output(state, "Episode 1 written (4 frames) from s/episode_000000")
    assert state.dataset_export.totalFrames == 9

    gateway._apply_export_output(state, "Export complete: 2 episodes at /x/pick_and_place")
    assert state.dataset_export.state == "complete"


def test_apply_export_output_marks_error(tmp_path):
    state, _ = _task_state(tmp_path)
    state.dataset_export = gateway.DatasetExportStatus(state="exporting")
    gateway._apply_export_output(state, "ERROR: No recorded sessions found")
    assert state.dataset_export.state == "error"


def test_snapshot_includes_dataset_export_and_active_task(tmp_path):
    state, _ = _task_state(tmp_path)
    snap = gateway._snapshot(state)
    assert "datasetExport" in snap
    assert snap["datasetExport"]["state"] == "idle"
    assert snap["activeTaskId"] == ""


def test_recorder_preview_frame_reads_fresh_tmpfs_jpeg(tmp_path, monkeypatch):
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    frame = b"\xff\xd8fake-jpeg\xff\xd9"
    (preview_dir / "cam_02.jpg").write_bytes(frame)
    monkeypatch.setattr(gateway, "_RECORDER_PREVIEW_DIR", preview_dir)
    monkeypatch.setattr(gateway, "_RECORDER_PREVIEW_STALE_S", 10.0)

    assert gateway._recorder_preview_frame("cam_02") == frame
    assert gateway._recorder_preview_frame("cam_03") is None


def test_recorder_failure_summary_prefers_error_over_last_stdout():
    recording = gateway.RecordingStatus(repoId="local/test")
    recording.recentOutput = [
        "Connecting: spawning 11 persistent pipelines...",
        "ERROR: persistent pipeline connect failed: connect exceeded global deadline 120.0s",
        "CONSUMER: Waiting until producer is connected...",
    ]
    recording.lastOutput = "CONSUMER: Waiting until producer is connected..."

    summary = gateway._recorder_failure_summary(recording)

    assert summary == (
        "ERROR: persistent pipeline connect failed: "
        "connect exceeded global deadline 120.0s"
    )


def test_recorder_failure_summary_uses_argus_failure_keyword():
    recording = gateway.RecordingStatus(repoId="local/test")
    recording.recentOutput = [
        "Camera index = 10",
        "nvbuf_utils: dmabuf_fd -1 mapped entry NOT found",
        "CONSUMER: Waiting until producer is connected...",
    ]
    recording.lastOutput = "CONSUMER: Waiting until producer is connected..."

    summary = gateway._recorder_failure_summary(recording)

    assert summary == "nvbuf_utils: dmabuf_fd -1 mapped entry NOT found"


# ---------------------------------------------------------------------------
# _maybe_send_preview_demand — viewer-demand heartbeat to the recorder
#
# On-demand recorder previews depend on the gateway pinging the recorder while
# the Device Manager grid polls camera.jpg. The write is debounced to ~1/s so a
# grid of 11 tiles polling at ~5fps doesn't flood the recorder's stdin.
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass


class _FakeRecorderProcess:
    pid = 4321

    def __init__(self):
        self.stdin = _FakeStdin()

    def poll(self):
        return None


def _preview_demand_state(tmp_path):
    return gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        # Heartbeats only go to a GMSL2 recorder (recorder.script gates it).
        config={
            "dataset": {"repo_id": "local/test", "root": str(tmp_path), "fps": 30},
            "recorder": {"script": "tools/thor/gmsl2/thor_record.py"},
        },
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=tmp_path,
    )


def test_maybe_send_preview_demand_writes_heartbeat(tmp_path):
    state = _preview_demand_state(tmp_path)
    state.process = _FakeRecorderProcess()
    gateway._maybe_send_preview_demand(state)
    assert state.process.stdin.writes == ["preview_demand\n"]


def test_maybe_send_preview_demand_debounces_rapid_polls(tmp_path, monkeypatch):
    state = _preview_demand_state(tmp_path)
    state.process = _FakeRecorderProcess()
    clock = {"t": 1000.0}
    monkeypatch.setattr(gateway.time, "monotonic", lambda: clock["t"])

    # A burst of polls inside the debounce window sends exactly one heartbeat.
    for _ in range(20):
        gateway._maybe_send_preview_demand(state)
    assert state.process.stdin.writes == ["preview_demand\n"]

    # Once the interval elapses, the next poll heartbeats again.
    clock["t"] += gateway._RECORDER_PREVIEW_DEMAND_INTERVAL_S + 0.01
    gateway._maybe_send_preview_demand(state)
    assert state.process.stdin.writes == ["preview_demand\n", "preview_demand\n"]


def test_maybe_send_preview_demand_noop_without_recorder(tmp_path):
    state = _preview_demand_state(tmp_path)
    state.process = None
    gateway._maybe_send_preview_demand(state)  # must not raise
    assert state.recorder_preview_demand_sent_s == 0.0


def test_maybe_send_preview_demand_skips_non_gmsl2_recorder(tmp_path):
    # The handheld recorder reads stdin as keypresses; it must never receive a
    # preview_demand heartbeat.
    state = _preview_demand_state(tmp_path)
    state.config = {"dataset": {"repo_id": "local/test", "root": str(tmp_path), "fps": 30},
                    "recorder": {"script": "tools/handheld/handheld_record.py"}}
    state.process = _FakeRecorderProcess()
    gateway._maybe_send_preview_demand(state)
    assert state.process.stdin.writes == []


def _training_view_state(tmp_path):
    """Workstation state with one recording and one training view built from it."""
    state, datasets_root = _task_state(tmp_path)
    state.exports_root = tmp_path / "repo" / "outputs" / "exports"
    state.profile = "workstation"
    def add_cameras(root: Path) -> None:
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for camera_key in ("observation.images.ee", "observation.images.side"):
            info["features"][camera_key] = {"dtype": "video"}
            chunk = root / "videos" / camera_key / "chunk-000"
            chunk.mkdir(parents=True)
            (chunk / "file-000.mp4").write_bytes(b"0" * 2048)
        info_path.write_text(json.dumps(info), encoding="utf-8")

    dataset_root = datasets_root / "fr3_spacemouse"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    add_cameras(dataset_root)
    view_root = state.exports_root / gateway.TRAINING_VIEWS_DIR_NAME / "fr3_spacemouse__delta_ee_from_prev_cmd"
    _write_minimal_episode_dataset(view_root, total_episodes=1)
    add_cameras(view_root)
    (view_root / "meta" / "il_view_manifest.json").write_text(
        json.dumps(
            {
                "source_dataset_root": str(dataset_root),
                "action_mode": "delta_ee_from_prev_cmd",
                "total_episodes": 1,
                "total_rows": 2,
            }
        ),
        encoding="utf-8",
    )
    gateway._invalidate_replay_candidates_memo()
    return state, dataset_root, view_root


def test_training_view_under_exports_is_a_replay_candidate(tmp_path):
    # The grouping directory is not a dataset root, so a single-level scan of the exports root
    # hid every built view -- and every endpoint gated on this list refused to serve it.
    state, _dataset_root, view_root = _training_view_state(tmp_path)

    candidates = gateway._complete_replay_dataset_candidates(state)

    assert view_root in candidates
    assert gateway._dataset_kind(state, view_root) == "training_view"
    assert gateway._resolve_known_dataset(state, str(view_root)) == view_root


def test_select_replay_dataset_accepts_a_training_view(tmp_path):
    state, _dataset_root, view_root = _training_view_state(tmp_path)

    gateway._select_replay_dataset(state, str(view_root))

    assert state.selected_replay_root == view_root
    assert state.replay.datasetRoot == str(view_root)
    assert state.replay.datasetKind == "training_view"


def test_recorded_dataset_items_link_a_view_to_its_source(tmp_path):
    state, dataset_root, view_root = _training_view_state(tmp_path)

    items = {item["path"]: item for item in gateway._recorded_dataset_items(state)}

    view_item = items[str(view_root)]
    assert view_item["datasetKind"] == "training_view"
    assert view_item["viewOf"] == str(dataset_root)
    assert view_item["viewOfName"] == dataset_root.name
    assert view_item["actionContract"] == "delta_ee_from_prev_cmd"
    # "Latest" drives one-click actions on fresh captures; a derived view must not claim it even
    # though it is the newest directory on disk.
    assert view_item["isLatest"] is False
    assert items[str(dataset_root)]["isLatest"] is True


def test_training_view_command_names_the_job_after_dataset_and_contract(tmp_path):
    # Without an explicit job name the builder falls back to a fixed legacy name, so every view
    # would generate configs training into -- and overwriting -- the same output directory.
    state, dataset_root, _view_root = _training_view_state(tmp_path)

    command, view_root = gateway._training_view_command(state, dataset_root, "delta_ee_from_prev_cmd")

    assert view_root == state.exports_root / gateway.TRAINING_VIEWS_DIR_NAME / "fr3_spacemouse__delta_ee_from_prev_cmd"
    assert command[command.index("--job-name") + 1] == "fr3_spacemouse__delta_ee_from_prev_cmd"
    assert command[command.index("--view-root") + 1] == str(view_root)
    assert "--prepare-only" in command


def test_start_training_view_refuses_to_build_from_another_view(tmp_path):
    state, _dataset_root, view_root = _training_view_state(tmp_path)

    with pytest.raises(ValueError, match="already a training view"):
        gateway._start_training_view(state, str(view_root), "absolute_ee")


def test_training_view_build_reports_the_view_not_the_training_output_dir(tmp_path):
    """Prepare-only prints a training output dir last; that must not become the status line.

    The directory is not created by a prepare-only run, and its name comes from the builder's
    own job naming -- reporting it told the operator about a job they never asked for.
    """
    state, dataset_root, view_root = _training_view_state(tmp_path)
    state.dataset_export = gateway.DatasetExportStatus(
        state="exporting",
        target="delta_ee_from_prev_cmd",
        datasetRoot=str(dataset_root),
        outputPath="",
    )

    class FakeProcess:
        stdout = [
            f"[prepare] dataset view: {view_root}\n",
            f"[prepare] train config: {view_root}/train_config.generated.json\n",
            "[prepare] training output dir (created when this view is trained): outputs/train/x\n",
        ]

        def wait(self):
            return 0

    process = FakeProcess()
    state.export_process = process

    gateway._read_export_output(state, process)

    assert state.dataset_export.state == "complete"
    assert state.dataset_export.outputPath == str(view_root)
    assert state.dataset_export.message == "View ready: 1 episode(s) · 2 frames · delta_ee_from_prev_cmd"


def test_export_completion_leaves_non_view_exports_untouched(tmp_path):
    state, _datasets_root = _task_state(tmp_path)
    state.dataset_export = gateway.DatasetExportStatus(state="exporting", target="lerobot_v3")

    class FakeProcess:
        stdout = ["Episode 0 written (12 frames)\n"]

        def wait(self):
            return 0

    process = FakeProcess()
    state.export_process = process

    gateway._read_export_output(state, process)

    assert state.dataset_export.state == "complete"
    assert state.dataset_export.message == "Episode 0 written (12 frames)"
    assert state.dataset_export.totalFrames == 12


def test_touch_payload_accepts_every_known_pad_width_and_tags_the_model():
    # The BOX SDK hands all pads over in a fixed 239-slot array, so box_client
    # cuts each frame to the pad fitted. The replay/preview path must accept
    # those narrower frames instead of dropping them as malformed.
    m2020 = gateway._touch_payload(
        {"timestamp": 5, "model": "m2020", "points": 9,
         "fz_0p1N": [0, 0, 0, 0, 90, 0, 0, 0, 0], "fx_0p1N": [0] * 9, "fy_0p1N": [0] * 9}
    )
    assert m2020 is not None
    assert m2020["model"] == "m2020"
    assert m2020["points"] == 9
    assert len(m2020["fz"]) == 9
    assert m2020["maxFz"] == 90
    assert m2020["activePoints"] == 1

    # Untagged frames (archived before box_client tagged the model) fall back
    # to the width, which is unambiguous for the pads we ship.
    legacy = gateway._touch_payload({"timestamp": 6, "fz_0p1N": [0.0] * 239})
    assert legacy is not None
    assert legacy["model"] == "paxini_l5325"
    assert legacy["points"] == 239

    untagged_m2020 = gateway._touch_payload({"timestamp": 7, "fz_0p1N": [0.0] * 9})
    assert untagged_m2020 is not None
    assert untagged_m2020["model"] == "m2020"

    # A width matching no pad is a truncated payload, not a smaller pad.
    assert gateway._touch_payload({"timestamp": 8, "fz_0p1N": [0.0] * 100}) is None
    assert gateway._touch_payload({"timestamp": 9, "fz_0p1N": []}) is None


def test_touch_payload_from_fz_pads_to_the_nearest_pad_width():
    # A short fz column is padded up to the smallest pad that fits, never
    # inflated back to 239 -- that would put an M2020 frame on the Paxini
    # layout in the replay view.
    short = gateway._touch_payload_from_fz([1.0, 2.0, 3.0])
    assert short is not None
    assert short["points"] == 9
    assert short["model"] == "m2020"
    assert short["fz"][:3] == [1.0, 2.0, 3.0]
    assert short["fz"][3:] == [0.0] * 6

    wide = gateway._touch_payload_from_fz([0.0] * 200)
    assert wide is not None
    assert wide["points"] == 239
    assert wide["model"] == "paxini_l5325"

    assert gateway._touch_payload_from_fz([]) is None


# --- canonical world frame (roadmap 2.4) -------------------------------------


def _world_gateway_state(tmp_path: Path) -> gateway.GatewayState:
    """A repo root the world CLI can actually be run against.

    ``third_party/opencv_kalibr`` is linked in rather than faked: the gateway
    puts it on PYTHONPATH itself, so a stub would test the stub.
    """
    (tmp_path / "third_party").mkdir(parents=True, exist_ok=True)
    real = Path(__file__).resolve().parents[2] / "third_party" / "opencv_kalibr"
    link = tmp_path / "third_party" / "opencv_kalibr"
    if not link.exists():
        link.symlink_to(real, target_is_directory=True)
    return gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(tmp_path / "ds"), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test", datasetRoot=str(tmp_path / "ds")),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )


def _write_bundle_report(path: Path, *, shift_m: float = 0.0, moved: str | None = None) -> Path:
    """A four-camera bundle in its own gauge, optionally with one camera bumped."""
    offsets = {"cam_00": [0, 0, 0], "cam_01": [1, 0, 0], "cam_02": [0, 1, 0], "cam_03": [1, 1, 0.2]}
    poses = {}
    for name, offset in offsets.items():
        matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        for axis in range(3):
            matrix[axis][3] = float(offset[axis]) - shift_m
        if name == moved:
            matrix[0][3] += 0.05
        poses[name] = matrix
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"reference": "cam_00", "T_ref_cam": poses, "rmse_px": 0.244}), encoding="utf-8"
    )
    return path


def test_world_frame_payload_says_so_when_nothing_has_been_frozen(tmp_path):
    state = _world_gateway_state(tmp_path)

    payload = gateway._world_frame_payload(state)

    assert payload["ok"] is True
    assert payload["reference"] == {"exists": False}
    assert payload["registration"] is None


def test_latest_bundle_report_prefers_the_run_just_solved(tmp_path):
    """A newest-file scan would register whichever calibration happens to be
    newest on disk, which is not necessarily the one that was just produced."""
    state = _world_gateway_state(tmp_path)
    _write_bundle_report(tmp_path / "outputs" / "metrology" / "old_run" / "extrinsics_report.json")
    wanted = _write_bundle_report(
        tmp_path / "outputs" / "metrology" / "the_run" / "extrinsics_report.json"
    )
    state.calibration.outputPath = str(wanted.parent)

    assert gateway._latest_bundle_report(state) == wanted


def test_registering_without_a_frozen_world_explains_the_missing_step(tmp_path):
    state = _world_gateway_state(tmp_path)
    _write_bundle_report(tmp_path / "outputs" / "metrology" / "run" / "extrinsics_report.json")

    result = gateway._register_world(state)

    assert result["ok"] is False
    assert "冻结" in result["error"]


def test_freeze_then_register_keeps_the_same_world_across_a_gauge_change(tmp_path, monkeypatch):
    """The end-to-end point of Phase 2.4, through the endpoints the GUI calls."""
    state = _world_gateway_state(tmp_path)
    monkeypatch.setattr(gateway, "_cv2_python", lambda repo_root: Path(sys.executable))
    first = _write_bundle_report(tmp_path / "outputs" / "metrology" / "run_a" / "extrinsics_report.json")
    state.calibration.outputPath = str(first.parent)

    frozen = gateway._freeze_world_reference(state)
    assert frozen["ok"] is True, frozen.get("error")
    world_id = frozen["reference"]["world_frame_id"]

    # Same rig, re-solved in a different gauge, with one camera bumped.
    second = _write_bundle_report(
        tmp_path / "outputs" / "metrology" / "run_b" / "extrinsics_report.json",
        shift_m=3.0,
        moved="cam_03",
    )
    state.calibration.outputPath = str(second.parent)
    result = gateway._register_world(state, assume_stable=["cam_00", "cam_01", "cam_02"])

    assert result["ok"] is True, result.get("error")
    registration = result["registration"]
    assert registration["world_continuity_state"] == "CONTINUOUS"
    assert registration["world_frame_id"] == world_id
    assert registration["consensus"]["moved_cameras"] == ["cam_03"]


def test_freezing_twice_is_refused_because_it_would_be_a_different_world(tmp_path, monkeypatch):
    state = _world_gateway_state(tmp_path)
    monkeypatch.setattr(gateway, "_cv2_python", lambda repo_root: Path(sys.executable))
    report = _write_bundle_report(tmp_path / "outputs" / "metrology" / "run" / "extrinsics_report.json")
    state.calibration.outputPath = str(report.parent)

    assert gateway._freeze_world_reference(state)["ok"] is True
    again = gateway._freeze_world_reference(state)

    assert again["ok"] is False
    assert "already defines world" in again["error"]


def _write_rig_check_result(
    state: gateway.GatewayState,
    *,
    generated_utc: str,
    ok: list[str],
    moved: list[str] = (),
    overall: str = "moved",
) -> Path:
    cameras = {name: {"verdict": "ok", "status": "measured"} for name in ok}
    cameras.update({name: {"verdict": "moved", "status": "measured"} for name in moved})
    path = gateway._rig_check_root(state) / "last_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"generated_utc": generated_utc, "overall": overall, "cameras": cameras}),
        encoding="utf-8",
    )
    return path


def _frozen_world(tmp_path, monkeypatch) -> gateway.GatewayState:
    state = _world_gateway_state(tmp_path)
    monkeypatch.setattr(gateway, "_cv2_python", lambda repo_root: Path(sys.executable))
    report = _write_bundle_report(tmp_path / "outputs" / "metrology" / "run_a" / "extrinsics_report.json")
    state.calibration.outputPath = str(report.parent)
    assert gateway._freeze_world_reference(state)["ok"] is True
    return state


def test_the_rig_self_check_decides_which_cameras_are_stable(tmp_path, monkeypatch):
    """The self-check resolves ~1.7 mm at 1 m; the geometric consensus ~1 cm.

    Letting the coarser measurement overrule the finer one would throw away the
    only evidence that can see a small bump at all.
    """
    state = _frozen_world(tmp_path, monkeypatch)
    _write_rig_check_result(
        state, generated_utc="2099-01-01T00:00:00Z", ok=["cam_00", "cam_01", "cam_02"], moved=["cam_03"]
    )
    _write_bundle_report(
        tmp_path / "outputs" / "metrology" / "run_b" / "extrinsics_report.json", shift_m=3.0, moved="cam_03"
    )
    state.calibration.outputPath = str(tmp_path / "outputs" / "metrology" / "run_b")

    result = gateway._register_world(state)

    assert result["stableSource"]["origin"] == "rig_check"
    assert result["registration"]["consensus"]["stable_cameras"] == ["cam_00", "cam_01", "cam_02"]
    assert result["registration"]["consensus"]["moved_cameras"] == ["cam_03"]


def test_a_self_check_older_than_the_frozen_world_is_not_evidence(tmp_path, monkeypatch):
    """It describes movement since a baseline that predates the world itself."""
    state = _frozen_world(tmp_path, monkeypatch)
    _write_rig_check_result(
        state, generated_utc="2000-01-01T00:00:00Z", ok=["cam_00", "cam_01", "cam_02", "cam_03"]
    )
    _write_bundle_report(tmp_path / "outputs" / "metrology" / "run_b" / "extrinsics_report.json", shift_m=3.0)
    state.calibration.outputPath = str(tmp_path / "outputs" / "metrology" / "run_b")

    result = gateway._register_world(state)

    assert result["stableSource"]["origin"] == "geometry"
    assert "冻结时间" in result["stableSource"]["reason"]


def test_an_inconclusive_self_check_is_refused_rather_than_read_as_ok(tmp_path, monkeypatch):
    state = _frozen_world(tmp_path, monkeypatch)
    _write_rig_check_result(
        state,
        generated_utc="2099-01-01T00:00:00Z",
        ok=["cam_00", "cam_01", "cam_02", "cam_03"],
        overall="inconclusive",
    )
    _write_bundle_report(tmp_path / "outputs" / "metrology" / "run_b" / "extrinsics_report.json", shift_m=3.0)
    state.calibration.outputPath = str(tmp_path / "outputs" / "metrology" / "run_b")

    result = gateway._register_world(state)

    assert result["stableSource"]["origin"] == "geometry"
    assert "无法判定" in result["stableSource"]["reason"]


def test_an_explicit_operator_choice_outranks_the_self_check(tmp_path, monkeypatch):
    state = _frozen_world(tmp_path, monkeypatch)
    _write_rig_check_result(
        state, generated_utc="2099-01-01T00:00:00Z", ok=["cam_00", "cam_01", "cam_02"], moved=["cam_03"]
    )
    _write_bundle_report(tmp_path / "outputs" / "metrology" / "run_b" / "extrinsics_report.json", shift_m=3.0)
    state.calibration.outputPath = str(tmp_path / "outputs" / "metrology" / "run_b")

    result = gateway._register_world(state, assume_stable=["cam_01", "cam_02", "cam_03"])

    assert result["stableSource"]["origin"] == "operator"
    assert result["registration"]["consensus"]["stable_cameras"] == ["cam_01", "cam_02", "cam_03"]


def test_the_frozen_world_is_somewhere_git_actually_tracks():
    """The one calibration artefact that cannot be regenerated.

    Re-running `freeze` mints a new `world_frame_id` for the same physical
    frame, orphaning the ID stamped into every episode recorded so far — so
    restoring `world_reference.json` from git is the only recovery there is.
    Putting it back under `outputs/` would look tidy (it is, after all, produced
    by a tool) and would silently make it disposable: that tree is 7 GB of
    regenerable artefacts and is deleted to reclaim space.

    Guards the .gitignore side too, which is the half a path constant cannot.
    """
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    reference = gateway._WORLD_SUBDIR / gateway._WORLD_REFERENCE_FILE
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(reference)],
        cwd=repo_root,
        check=False,
    )
    assert ignored.returncode != 0, f"{reference} is gitignored; it cannot be regenerated"

    # The volatile half must stay out: it is rewritten by every continuity check.
    for volatile in (gateway._WORLD_REGISTRATION_FILE, gateway._WORLD_STABLE_SOURCE_FILE):
        path = gateway._WORLD_SUBDIR / volatile
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=repo_root, check=False
        )
        assert result.returncode == 0, f"{path} would be committed on every check"


# ---------------------------------------------------------------------------
# Solve progress, and refusing a solve that cannot finish
# ---------------------------------------------------------------------------


def _solve_state(tmp_path: Path) -> gateway.GatewayState:
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(tmp_path / "ds"), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test", datasetRoot=str(tmp_path / "ds")),
        replay=gateway.ReplayStatus(dataset="local/test"),
    )
    return state


def _charuco_capture(tmp_path: Path, *, episodes: int, cameras: int) -> Path:
    dataset = tmp_path / "outputs" / "datasets" / "calib_capture"
    for episode in range(episodes):
        directory = dataset / "episodes" / f"episode_{episode:03d}"
        directory.mkdir(parents=True)
        for camera in range(cameras):
            (directory / f"cam_{camera:02d}.mkv").write_bytes(b"")
    return dataset


def test_the_bar_advances_only_where_a_step_can_say_how_far_it_is(tmp_path):
    """Detection counts videos; the bundle counts nothing and must not pretend."""
    weights = (0.8, 0.16, 0.04)

    assert gateway._solve_fraction(1, 0, 100, weights) == pytest.approx(0.0)
    assert gateway._solve_fraction(1, 50, 100, weights) == pytest.approx(0.4)
    assert gateway._solve_fraction(1, 100, 100, weights) == pytest.approx(0.8)
    # Step 2 reports no units, so it sits at its own start boundary rather than
    # creeping on a timer -- a bar that moves on its own stops being evidence.
    assert gateway._solve_fraction(2, 0, 0, weights) == pytest.approx(0.8)
    assert gateway._solve_fraction(3, 0, 0, weights) == pytest.approx(0.96)
    assert gateway._solve_fraction(0, 0, 0, weights) == 0.0
    # A capture whose video count was under-estimated still cannot overrun.
    assert gateway._solve_fraction(1, 300, 100, weights) == pytest.approx(0.8)


def test_no_eta_is_offered_before_there_is_anything_to_extrapolate_from():
    assert gateway._solve_eta_s(0.0, 30.0) == 0.0
    assert gateway._solve_eta_s(0.01, 30.0) == 0.0  # dominated by process startup
    assert gateway._solve_eta_s(0.25, 300.0) == pytest.approx(900.0)
    assert gateway._solve_eta_s(1.0, 300.0) == 0.0


def test_detection_output_is_read_as_one_unit_per_video():
    """The columns detect_charuco prints, and the lines that are not videos."""
    done, detail = gateway._solve_progress_line("episode_000 cam_06     512             41")
    assert done is True
    assert "episode_000" in detail and "cam_06" in detail and "512" in detail

    # A camera whose video will not open still finished its unit; not counting
    # it would leave the bar permanently short of 100%.
    done, detail = gateway._solve_progress_line("episode_000 cam_00     -    -  <-- 视频打不开")
    assert done is True
    assert "打不开" in detail

    assert gateway._solve_progress_line("episode          camera    frames  median corners") == (False, "")
    assert gateway._solve_progress_line("-" * 52) == (False, "")
    assert gateway._solve_progress_line("   ") == (False, "")

    # The bundle's own prose is not a unit, but it is the only sign of life
    # during the minutes it spends inside least_squares.
    done, detail = gateway._solve_progress_line("sync frames: 812  cameras: ['cam_06']")
    assert done is False
    assert detail.startswith("sync frames")


def test_the_detection_bar_is_scaled_by_the_videos_it_will_open(tmp_path):
    dataset = _charuco_capture(tmp_path, episodes=3, cameras=4)
    assert gateway._charuco_video_count(dataset / "episodes") == 12

    # detect_charuco also accepts a directory holding the videos directly, and
    # the count has to follow it there or the bar would read 0/0.
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "cam_06.mkv").write_bytes(b"")
    (flat / "cam_07.mp4").write_bytes(b"")
    assert gateway._charuco_video_count(flat) == 2


def test_a_solve_step_reports_its_output_while_it_is_still_running(tmp_path):
    """Read at the end this is a log; read as it arrives it is the bar."""
    state = _solve_state(tmp_path)
    seen: list[str] = []
    script = "import sys\nfor i in range(3):\n    print('line', i)\n    sys.stdout.flush()\n"

    proc = gateway._calibration_step(
        state,
        Path(sys.executable),
        ["-c", script],
        label="测试步骤…",
        timeout=60,
        on_line=seen.append,
    )

    assert proc is not None
    assert proc.returncode == 0
    assert seen == ["line 0", "line 1", "line 2"]
    assert proc.stdout.splitlines() == seen


def test_a_step_that_prints_nothing_and_never_exits_is_still_killed(tmp_path):
    """The deadline cannot be checked between lines: a hung step prints none."""
    state = _solve_state(tmp_path)

    proc = gateway._calibration_step(
        state,
        Path(sys.executable),
        ["-c", "import time; time.sleep(120)"],
        label="卡住的步骤…",
        timeout=1,
    )

    assert proc is None
    assert state.calibration.state == "failed"
    assert "1s" in state.calibration.message


def test_stderr_is_still_captured_when_the_step_fails(tmp_path):
    """The failure message is built from it, so streaming must not drop it."""
    state = _solve_state(tmp_path)

    proc = gateway._calibration_step(
        state,
        Path(sys.executable),
        ["-c", "import sys; sys.stderr.write('ModuleNotFoundError: boom\\n'); sys.exit(2)"],
        label="测试步骤…",
        timeout=60,
    )

    assert proc is not None
    assert proc.returncode == 2
    assert "ModuleNotFoundError: boom" in proc.stderr


def test_a_solve_is_refused_up_front_when_the_interpreter_cannot_import_scipy(tmp_path, monkeypatch):
    """The 2026-08-20 failure: detection ran to completion -- tens of minutes --
    and only then did the bundle die on `No module named 'scipy'`."""
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    monkeypatch.setattr(
        gateway, "_solve_python", lambda _root: (Path("/opt/venv/bin/python3"), ["scipy.optimize"])
    )
    started: list[str] = []
    monkeypatch.setattr(gateway, "_run_extrinsics_calibration", lambda *a, **k: started.append("ran"))

    result = gateway._start_extrinsics_calibration(state, str(dataset))

    assert result["ok"] is False
    assert "scipy" in result["error"]
    # Actionable, not merely accurate: the interpreter and the install command.
    assert "/opt/venv/bin/python3" in result["error"]
    assert "pip install scipy" in result["error"]
    assert started == []
    assert state.calibration.state != "running"


def test_a_refused_solve_says_so_inside_the_guided_session(tmp_path, monkeypatch):
    """The wizard shows its own message; a refusal returned only to the caller
    leaves it reading "可以解算" while nothing at all is happening."""
    state = _calibration_gateway_state(tmp_path)
    monkeypatch.setattr(gateway, "_start_episode", lambda *a, **k: None)
    assert gateway._start_calibration_session(state)["ok"] is True
    monkeypatch.setattr(
        gateway, "_solve_python", lambda _root: (Path("/opt/venv/bin/python3"), ["scipy.optimize"])
    )

    result = gateway._start_extrinsics_calibration(state, str(tmp_path / "nope"))

    assert result["ok"] is False
    assert state.calibration_session.message == result["error"]


def test_the_module_probe_reports_only_what_is_actually_missing(tmp_path):
    missing = gateway._missing_modules(
        Path(sys.executable), ["json", "definitely_not_a_module", "os"]
    )
    assert missing == ["definitely_not_a_module"]

    # An interpreter that cannot even run is missing everything: it is unusable
    # either way, and the caller only decides whether to use it.
    assert gateway._missing_modules(tmp_path / "no-such-python", ["json"]) == ["json"]


def test_the_solve_puts_a_bar_on_screen_from_the_click(tmp_path, monkeypatch):
    """Not from whenever the worker thread happens to get scheduled."""
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    monkeypatch.setattr(gateway, "_solve_python", lambda _root: (Path(sys.executable), []))
    monkeypatch.setattr(gateway, "_run_extrinsics_calibration", lambda *a, **k: None)

    assert gateway._start_extrinsics_calibration(state, str(dataset))["ok"] is True

    progress = state.calibration.progress
    assert progress.stepIndex == 1
    assert progress.stepCount == 3
    assert progress.startedAt > 0


def test_elapsed_is_read_from_the_rig_clock_at_request_time(tmp_path, monkeypatch):
    """The bundle prints nothing for minutes; if elapsed only advanced on output
    the page would look frozen exactly when the operator needs to see it is not.
    Computing it in the browser is not an option either -- the rig's clock and
    the operator's have been observed minutes apart."""
    state = _solve_state(tmp_path)
    state.calibration.state = "running"
    state.calibration.progress = gateway.CalibrationProgress(
        stepIndex=1, stepCount=3, label="检测 ChArUco 角点…", fraction=0.25, startedAt=1000.0
    )
    monkeypatch.setattr(gateway.time, "time", lambda: 1300.0)

    payload = gateway._calibration_payload(state)

    assert payload["progress"]["elapsedS"] == pytest.approx(300.0)
    assert payload["progress"]["etaS"] == pytest.approx(900.0)

    # Once it stops, the clock stops with it rather than counting up forever.
    gateway._finish_solve_progress(state, complete=True)
    state.calibration.state = "complete"
    frozen = gateway._calibration_payload(state)
    assert frozen["progress"]["elapsedS"] == pytest.approx(300.0)
    assert frozen["progress"]["etaS"] == 0.0
    assert frozen["progress"]["fraction"] == 1.0


def test_a_failed_solve_stops_its_clock_too(tmp_path, monkeypatch):
    state = _solve_state(tmp_path)
    state.calibration.progress = gateway.CalibrationProgress(
        stepIndex=2, stepCount=3, fraction=0.8, startedAt=1000.0, etaS=120.0
    )
    monkeypatch.setattr(gateway.time, "time", lambda: 1100.0)

    gateway._fail_calibration(state, "多相机联合 BA… 失败：boom")

    assert state.calibration.state == "failed"
    assert state.calibration.progress.elapsedS == pytest.approx(100.0)
    assert state.calibration.progress.etaS == 0.0


def test_the_bar_walks_all_three_steps_of_a_real_solve(tmp_path, monkeypatch):
    """The wiring, not the arithmetic: each step must declare how many units it
    has before it runs, and the export must not leave the bar short of done."""
    state = _solve_state(tmp_path)
    state.calibration.state = "running"
    state.calibration.progress = gateway.CalibrationProgress(startedAt=1000.0)
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    report_path = tmp_path / "outputs" / "metrology" / "run_x" / "extrinsics_report.json"
    intrinsics = tmp_path / gateway._CALIB_INTRINSICS_REPORT
    intrinsics.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.write_text("{}", encoding="utf-8")

    steps: list[tuple[str, int]] = []
    finished: list[tuple[str, int]] = []

    def fake_step(_state, _python, args, *, label, timeout, on_line=None):
        steps.append((label, _state.calibration.progress.total))
        module = next((arg for arg in args if arg.startswith("metrology.cli.")), "")
        if module.endswith("detect_charuco"):
            assert on_line is not None
            for episode in range(2):
                for camera in range(3):
                    on_line(f"episode_{episode:03d} cam_{camera:02d}   400   40")
        finished.append((label, _state.calibration.progress.done))
        if module.endswith("calibrate_extrinsics"):
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"rmse_px": 0.2, "num_frames": 400, "per_camera_rmse": {"cam_00": 0.2}}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(["python"], 0, "", "")

    monkeypatch.setattr(gateway, "_calibration_step", fake_step)

    gateway._run_extrinsics_calibration(state, dataset, "run_x", Path(sys.executable))

    assert [label for label, _ in steps] == ["检测 ChArUco 角点…", "多相机联合 BA…", "导出生产标定…"]
    # Only the detection step knows its own size; it is 2 episodes x 3 cameras.
    assert [total for _, total in steps] == [6, 0, 0]
    assert finished[0] == ("检测 ChArUco 角点…", 6)  # every video counted, none twice
    assert state.calibration.state == "complete", state.calibration.message
    assert state.calibration.progress.fraction == 1.0


# ---------------------------------------------------------------------------
# Which capture gets solved, and reusing its detections on a retry
# ---------------------------------------------------------------------------


def test_a_failed_solve_can_be_retried_on_the_same_capture(tmp_path):
    """The 2026-08-20 dead end: the wizard records into a dataset named after
    the rig (thor_gmsl2_10ch_v1_...), the solve failed, and the fallback scan
    only finds directories with "calib" in the name -- so an intact 11-episode
    capture became unreachable the moment its first solve failed."""
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    renamed = dataset.parent / "thor_gmsl2_10ch_v1_20260820_152528"
    dataset.rename(renamed)
    state.calibration_session = gateway.CalibrationSession(
        active=True, stage="failed", datasetName=renamed.name, datasetRoot=str(renamed)
    )

    resolved, source = gateway._solve_dataset(state)

    assert resolved == renamed
    assert source == "session"


def test_leaving_the_wizard_does_not_orphan_what_it_recorded(tmp_path):
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    renamed = dataset.parent / "thor_gmsl2_10ch_v1_20260820_152528"
    dataset.rename(renamed)
    state.calibration_session = gateway.CalibrationSession(
        active=True, stage="failed", datasetName=renamed.name, datasetRoot=str(renamed)
    )

    assert gateway._cancel_calibration_session(state)["ok"] is True

    assert gateway._solve_dataset(state) == (renamed, "manual")


def test_a_named_capture_is_never_silently_replaced_by_another(tmp_path):
    """Solving a different capture than the one asked for produces a result
    nobody can trace back to its input."""
    state = _solve_state(tmp_path)
    _charuco_capture(tmp_path, episodes=1, cameras=2)  # a perfectly good fallback
    state.calibration.solveDatasetRoot = str(tmp_path / "gone")

    resolved, source = gateway._solve_dataset(state)

    assert resolved is None
    assert source == "missing"

    result = gateway._start_extrinsics_calibration(state)
    assert result["ok"] is False
    assert "读不到" in result["error"]


def test_picking_a_capture_requires_it_to_actually_hold_episodes(tmp_path):
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    empty = tmp_path / "outputs" / "datasets" / "not_a_capture"
    empty.mkdir(parents=True)

    assert gateway._set_solve_dataset(state, str(empty))["ok"] is False
    assert gateway._set_solve_dataset(state, str(dataset))["ok"] is True
    assert state.calibration.solveDatasetRoot == str(dataset)
    # Clearing restores the automatic choice rather than wedging on the old one.
    assert gateway._set_solve_dataset(state, "")["ok"] is True
    assert state.calibration.solveDatasetRoot == ""


def test_the_capture_being_solved_is_always_in_the_dropdown(tmp_path):
    """A capture recorded moments ago is not in the dataset scan yet; if the
    list omitted it the dropdown would contradict the label above it."""
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    state.calibration.solveDatasetRoot = str(dataset)

    payload = gateway._solve_payload(state)

    assert payload["datasetRoot"] == str(dataset)
    assert payload["source"] == "manual"
    assert [item["path"] for item in payload["candidates"]] == [str(dataset)]
    assert payload["candidates"][0]["episodes"] == 2


def test_detections_are_reused_when_the_videos_have_not_changed(tmp_path):
    """They are a pure function of (video, stride, board), and producing them is
    the expensive half of a solve. Re-deriving them on every retry is what made
    the missing-scipy failure cost half an hour a second time."""
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    episodes = dataset / "episodes"
    detections = tmp_path / "det"
    detections.mkdir()

    assert gateway._reusable_detections(episodes, detections) is None  # nothing yet

    for stem, _video in gateway._capture_videos(episodes):
        (detections / f"{stem}.npz").write_bytes(b"")
    gateway._write_detection_manifest(episodes, detections)

    assert gateway._reusable_detections(episodes, detections) == 6


def test_a_capture_that_changed_is_detected_again_rather_than_half_reused(tmp_path):
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    episodes = dataset / "episodes"
    detections = tmp_path / "det"
    detections.mkdir()
    for stem, _video in gateway._capture_videos(episodes):
        (detections / f"{stem}.npz").write_bytes(b"")
    gateway._write_detection_manifest(episodes, detections)

    # An episode deleted after the fact. Reusing here would let a recording the
    # operator threw away keep voting on the extrinsics.
    shutil.rmtree(episodes / "episode_001")
    assert gateway._reusable_detections(episodes, detections) is None

    # And a video re-recorded under the same name.
    restored = _charuco_capture(tmp_path / "again", episodes=2, cameras=3) / "episodes"
    gateway._write_detection_manifest(restored, detections)
    os.utime(restored / "episode_000" / "cam_00.mkv", (1, 1))
    assert gateway._reusable_detections(restored, detections) is None


def test_stale_npz_are_cleared_before_a_re_detection(tmp_path):
    detections = tmp_path / "det"
    detections.mkdir()
    (detections / "episode_000__cam_00.npz").write_bytes(b"")
    (detections / gateway._DETECTION_MANIFEST).write_text("{}", encoding="utf-8")

    gateway._clear_detections(detections)

    assert list(detections.iterdir()) == []


def test_a_retry_skips_detection_and_says_so(tmp_path, monkeypatch):
    state = _solve_state(tmp_path)
    state.calibration.state = "running"
    state.calibration.progress = gateway.CalibrationProgress(startedAt=1000.0)
    dataset = _charuco_capture(tmp_path, episodes=2, cameras=3)
    episodes = dataset / "episodes"
    detections = gateway._detections_dir(state, dataset)
    detections.mkdir(parents=True)
    for stem, _video in gateway._capture_videos(episodes):
        (detections / f"{stem}.npz").write_bytes(b"")
    gateway._write_detection_manifest(episodes, detections)
    intrinsics = tmp_path / gateway._CALIB_INTRINSICS_REPORT
    intrinsics.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.write_text("{}", encoding="utf-8")
    report = tmp_path / "outputs" / "metrology" / "run_y" / "extrinsics_report.json"

    labels: list[str] = []

    def fake_step(_state, _python, args, *, label, timeout, on_line=None):
        labels.append(label)
        module = next((arg for arg in args if arg.startswith("metrology.cli.")), "")
        if module.endswith("calibrate_extrinsics"):
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"rmse_px": 0.2, "per_camera_rmse": {}}), encoding="utf-8")
        return subprocess.CompletedProcess(["python"], 0, "", "")

    monkeypatch.setattr(gateway, "_calibration_step", fake_step)

    gateway._run_extrinsics_calibration(state, dataset, "run_y", Path(sys.executable))

    assert labels == ["多相机联合 BA…", "导出生产标定…"]  # detection never ran
    assert state.calibration.state == "complete"


def test_forcing_a_re_detection_ignores_the_cache(tmp_path, monkeypatch):
    """The escape hatch for the case the fingerprint cannot see: a video that
    changed without its mtime changing, or a detector that was itself fixed."""
    state = _solve_state(tmp_path)
    state.calibration.state = "running"
    state.calibration.progress = gateway.CalibrationProgress(startedAt=1000.0)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    episodes = dataset / "episodes"
    detections = gateway._detections_dir(state, dataset)
    detections.mkdir(parents=True)
    for stem, _video in gateway._capture_videos(episodes):
        (detections / f"{stem}.npz").write_bytes(b"")
    gateway._write_detection_manifest(episodes, detections)
    intrinsics = tmp_path / gateway._CALIB_INTRINSICS_REPORT
    intrinsics.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.write_text("{}", encoding="utf-8")

    labels: list[str] = []

    def fake_step(_state, _python, args, *, label, timeout, on_line=None):
        labels.append(label)
        return subprocess.CompletedProcess(["python"], 1, "", "boom")

    monkeypatch.setattr(gateway, "_calibration_step", fake_step)

    gateway._run_extrinsics_calibration(
        state, dataset, "run_z", Path(sys.executable), force_redetect=True
    )

    assert labels[0] == "检测 ChArUco 角点…"
    # The manifest is written only on success, so a failed re-detection cannot
    # leave a directory that the next attempt would trust.
    assert not (detections / gateway._DETECTION_MANIFEST).is_file()


def test_the_solve_can_refit_intrinsics_from_a_second_capture(tmp_path, monkeypatch):
    """The wizard records one intrinsics sweep per camera, and until now the
    solve read none of them: it ran detect -> calibrate_extrinsics -> export and
    reused whatever intrinsics run production already pointed at."""
    state = _solve_state(tmp_path)
    state.calibration.state = "running"
    state.calibration.progress = gateway.CalibrationProgress(startedAt=1000.0)
    extrinsics = _charuco_capture(tmp_path, episodes=1, cameras=3)
    intrinsics = _charuco_capture(tmp_path / "i", episodes=7, cameras=3)
    report = tmp_path / "outputs" / "metrology" / "run_i" / "extrinsics_report.json"

    steps: list[tuple[str, str]] = []

    def fake_step(_state, _python, args, *, label, timeout, on_line=None):
        module = next((arg for arg in args if arg.startswith("metrology.cli.")), "")
        steps.append((label, module))
        if module.endswith("calibrate_intrinsics"):
            Path(args[args.index("--out") + 1]).write_text("{}", encoding="utf-8")
        if module.endswith("calibrate_extrinsics"):
            # It must be solved against the intrinsics just fitted, not the
            # production run: shipping a bundle fitted to one set of intrinsics
            # alongside another is a mismatch nothing downstream can detect.
            assert "--intrinsics-report" in args
            assert args[args.index("--intrinsics-report") + 1].endswith("intrinsics_report.json")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"rmse_px": 0.2, "per_camera_rmse": {}}), encoding="utf-8")
        if module.endswith("export_production_calibration"):
            # And the new intrinsics have to be exported, or the run would ship
            # a bundle whose lenses live only under outputs/.
            assert "--intrinsics-report" in args
        return subprocess.CompletedProcess(["python"], 0, "", "")

    monkeypatch.setattr(gateway, "_calibration_step", fake_step)

    gateway._run_extrinsics_calibration(
        state, extrinsics, "run_i", Path(sys.executable), intrinsics_dataset=intrinsics
    )

    assert [module.split(".")[-1] for _label, module in steps] == [
        "detect_charuco",
        "calibrate_intrinsics",
        "detect_charuco",
        "calibrate_extrinsics",
        "export_production_calibration",
    ]
    assert state.calibration.state == "complete", state.calibration.message
    assert state.calibration.intrinsicsRun == "run_i_intrinsics"
    assert state.calibration.progress.stepCount == 5


def test_asking_to_refit_intrinsics_without_a_capture_is_refused_up_front(tmp_path, monkeypatch):
    state = _solve_state(tmp_path)
    dataset = _charuco_capture(tmp_path, episodes=1, cameras=2)
    state.calibration.solveDatasetRoot = str(dataset)
    monkeypatch.setattr(gateway, "_solve_python", lambda _root: (Path(sys.executable), []))
    monkeypatch.setattr(gateway, "_run_extrinsics_calibration", lambda *a, **k: None)

    result = gateway._start_extrinsics_calibration(state, refit_intrinsics=True)

    assert result["ok"] is False
    assert "内参采集" in result["error"]
    assert "四角" in result["hint"]
    assert state.calibration.state != "running"


def test_the_bar_is_weighted_by_the_video_each_step_has_to_decode(tmp_path):
    """An intrinsics capture is one sweep per camera: seven times the video of
    the extrinsics sweep. A fixed split would sit at 40% with 90% of the work
    still ahead."""
    intrinsics_heavy = gateway._solve_weights([70, 10])
    assert sum(intrinsics_heavy) == pytest.approx(1.0)
    assert intrinsics_heavy[0] == pytest.approx(0.85 * 70 / 80)
    assert intrinsics_heavy[2] == pytest.approx(0.85 * 10 / 80)
    # Detection still dominates when it is the only capture.
    assert gateway._solve_weights([10]) == pytest.approx([0.85, 0.13, 0.02])


def test_each_capture_keeps_its_own_detections(tmp_path):
    """Intrinsics and extrinsics are different recordings; sharing a detection
    directory would let one overwrite the other's corners."""
    state = _solve_state(tmp_path)
    a = _charuco_capture(tmp_path, episodes=1, cameras=2)
    b = _charuco_capture(tmp_path / "b", episodes=7, cameras=2)

    assert gateway._detections_dir(state, a) != gateway._detections_dir(state, b)


def test_the_intrinsics_capture_is_selected_separately(tmp_path):
    state = _solve_state(tmp_path)
    extrinsics = _charuco_capture(tmp_path, episodes=1, cameras=2)
    intrinsics = _charuco_capture(tmp_path / "i", episodes=7, cameras=2)

    assert gateway._set_solve_dataset(state, str(extrinsics))["ok"] is True
    assert gateway._set_solve_dataset(state, str(intrinsics), "intrinsics")["ok"] is True

    payload = gateway._solve_payload(state)
    assert payload["datasetRoot"] == str(extrinsics)
    assert payload["intrinsicsDatasetRoot"] == str(intrinsics)
    assert payload["intrinsicsEpisodes"] == 7
    # Both must stay selectable in the one dropdown the panel offers.
    assert {item["path"] for item in payload["candidates"]} == {str(extrinsics), str(intrinsics)}


def test_the_weights_follow_the_order_the_steps_actually_run_in(tmp_path):
    """The intrinsics fit sits *between* the two detections, so its weight
    cannot be appended after both of them."""
    weights = gateway._solve_weights([70, 10])

    assert weights[0] == pytest.approx(0.85 * 70 / 80)  # detect the intrinsics capture
    assert weights[1] == pytest.approx(0.05)  # fit intrinsics
    assert weights[2] == pytest.approx(0.85 * 10 / 80)  # detect the extrinsics capture
    assert sum(weights) == pytest.approx(1.0)
    # A bar 74% of the way through step 1 of 5 is not 15% done overall.
    assert gateway._solve_fraction(1, 70, 70, weights) == pytest.approx(0.74375)


def test_a_blind_camera_does_not_take_the_whole_intrinsics_fit_down(tmp_path):
    """detect_charuco deliberately writes a file for a camera that saw nothing.
    On this rig cam_01/02/03 point away from the board and detect zero frames in
    every episode; aborting every other camera's fit over that is the wrong
    failure mode."""
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "opencv_kalibr"))
    from metrology.cli import calibrate_intrinsics

    detections = tmp_path / "det"
    detections.mkdir()
    np.savez(
        detections / "episode_000000__cam_01.npz",
        image_size=np.asarray([1920, 1080], dtype=np.int64),
        frames=np.asarray([], dtype=np.int32),
        n_per_frame=np.asarray([], dtype=np.int32),
        charuco_ids=np.zeros((0,), np.int32),
        charuco_pts=np.zeros((0, 2), np.float32),
        aruco_n_per_frame=np.asarray([], dtype=np.int32),
        aruco_ids=np.zeros((0,), np.int32),
        aruco_pts=np.zeros((0, 4, 2), np.float32),
        total_frames=np.asarray([1826], dtype=np.int64),
    )
    report = detections / "intrinsics_report.json"

    assert calibrate_intrinsics.main(["--detections", str(detections), "--out", str(report)]) == 0

    entry = json.loads(report.read_text(encoding="utf-8"))["cameras"]["cam_01"]
    assert entry["frames_detected"] == 0
    # No K under any model, which is what load_intrinsics_map skips on.
    assert entry["models"] == {}


def test_a_refit_reports_edge_coverage_next_to_the_residual(tmp_path):
    """Reprojection cannot see the failure that forced the 0804 recapture: a
    distortion fit is perfectly happy to be self-consistent over the middle of
    the frame it was given, and say nothing about the corners it never saw."""
    report = tmp_path / "intrinsics_report.json"
    report.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam_06": {
                        "observed_radius_fraction": 0.96,
                        "models": {"fisheye": {"monotonic_across_frame": True}},
                    },
                    "cam_08": {
                        "observed_radius_fraction": 0.62,
                        "models": {"fisheye": {"monotonic_across_frame": True}},
                    },
                    "cam_09": {
                        "observed_radius_fraction": 0.91,
                        "models": {"fisheye": {"monotonic_across_frame": False}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    cameras = [
        {"id": "cam_06", "reprojectionPx": 0.2, "status": "pass"},
        {"id": "cam_08", "reprojectionPx": 0.2, "status": "pass"},
        {"id": "cam_09", "reprojectionPx": 0.2, "status": "pass"},
    ]

    gateway._annotate_intrinsics_coverage(cameras, report)

    assert cameras[0]["status"] == "pass"  # 96% is what the good recapture reached
    assert cameras[1]["status"] == "warn"  # 62% is what had to be redone
    assert "62%" in cameras[1]["intrinsicsNote"]
    # A model that folds inside its own frame is worse than thin coverage: those
    # pixels map to the wrong ray, so a good residual there means nothing.
    assert cameras[2]["status"] == "fail"
    assert "折返" in cameras[2]["intrinsicsNote"]


def test_coverage_annotation_survives_a_report_it_cannot_read(tmp_path):
    cameras = [{"id": "cam_06", "reprojectionPx": 0.2, "status": "pass"}]
    gateway._annotate_intrinsics_coverage(cameras, tmp_path / "missing.json")
    assert cameras == [{"id": "cam_06", "reprojectionPx": 0.2, "status": "pass"}]


def _intrinsics_gateway_state(tmp_path: Path, run: str) -> gateway.GatewayState:
    state = _marker_tcp_gateway_state(tmp_path)
    state.calibration.intrinsicsRun = run
    return state


def _write_producer_intrinsics(
    root: Path, camera: str, *, coverage: float | None, fold_deg: float, corner_deg: float
) -> None:
    directory = root / "converted" / f"{camera}_SERIAL"
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "camera_name": camera,
        "camera_serial": "SERIAL",
        "image_width": 1920,
        "image_height": 1080,
        "camera_matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [[-0.07, -0.005, 0.001, 0.0]],
        "model": "opencv_fisheye",
    }
    if coverage is not None:
        payload["self_calibration"] = {
            "observed_radius_fraction": coverage,
            "radial_fold_deg": fold_deg,
            "corner_bearing_deg": corner_deg,
            "frames_used": 200,
            "heldout_time_block_rmse_px": 0.16,
        }
    (directory / "intrinsics_producer.json").write_text(json.dumps(payload), encoding="utf-8")


def test_intrinsics_coverage_reports_each_camera_against_the_target(tmp_path):
    run = "thor_gmsl2_selfcal_0804_fisheye_intrinsics"
    root = tmp_path / "outputs" / "calibration" / run
    # cam_06 as it really is in the 0804 production set: short of the recapture
    # floor and folding only 2.1 deg outside its own corner.
    _write_producer_intrinsics(root, "cam_06", coverage=0.7876, fold_deg=80.23, corner_deg=78.13)
    _write_producer_intrinsics(root, "cam_08", coverage=0.9558, fold_deg=float("inf"), corner_deg=76.86)

    payload = gateway._intrinsics_coverage_payload(_intrinsics_gateway_state(tmp_path, run))

    assert payload["ok"] is True
    assert payload["run"] == run
    assert payload["coverageTarget"] == 0.90
    by_camera = {entry["camera"]: entry for entry in payload["cameras"]}
    assert by_camera["cam_06"]["coverage"] == pytest.approx(0.7876)
    assert by_camera["cam_06"]["foldMarginDeg"] == pytest.approx(2.10, abs=0.01)
    assert by_camera["cam_06"]["foldsInsideFrame"] is False
    # An infinite fold limit means the model never folds. Reporting it as a
    # number would make "never folds" indistinguishable from a huge margin, and
    # JSON cannot carry inf anyway.
    assert by_camera["cam_08"]["foldMarginDeg"] is None


def test_intrinsics_coverage_leaves_an_unmeasured_camera_unmeasured(tmp_path):
    """A vendor file has no self-calibration record, and must not read as passing."""
    run = "vendor_intrinsics"
    root = tmp_path / "outputs" / "calibration" / run
    _write_producer_intrinsics(root, "cam_01", coverage=None, fold_deg=0.0, corner_deg=0.0)

    payload = gateway._intrinsics_coverage_payload(_intrinsics_gateway_state(tmp_path, run))

    entry = payload["cameras"][0]
    assert entry["camera"] == "cam_01"
    assert "coverage" not in entry


def test_intrinsics_coverage_reports_a_missing_run_instead_of_an_empty_table(tmp_path):
    payload = gateway._intrinsics_coverage_payload(_intrinsics_gateway_state(tmp_path, "no_such_run"))

    assert payload["cameras"] == []
    assert "找不到内参目录" in payload["error"]


# --- marker->TCP solve + EE trajectory bundle override -----------------------

MARKER_TCP_TRACKING_CONFIG = """
calibration:
  root_dir: outputs/calibration
  intrinsics_run_name: intr_run
  fixed_camera_run_name: extr_run
cube_tracker:
  aruco_dictionary: DICT_APRILTAG_36h11
  marker_size_cm: 5.6
  cubes:
    - name: left
      handedness: right_hand
      marker_ids: [null, 2, 0, 3, 4, 1]
      cube_size_cm_xyz: [7.192, 7.167, 7.109]
ee_from_cube:
  mode: calibrated_marker_to_tcp
  marker_to_tcp_calibration_path: config/marker_to_tcp_calibration.json
"""

# R_cube_tcp for the 0812 left cube: a proper rotation, so the solve inherits it
# rather than refusing for lack of rotation evidence.
INHERITED_ROTATION = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]


def _marker_tcp_bundle(rotation, translation_m=(0.0, 0.1019, 0.0085)):
    transform = [list(row) + [value] for row, value in zip(rotation, translation_m, strict=True)]
    transform.append([0.0, 0.0, 0.0, 1.0])
    return {
        "schema": "marker_rig_to_tcp_calibration/v1",
        "calibration_id": "pivot_20260812",
        "cubes": {
            "left": {"device_id": "box1672693301", "T_cube_tcp": transform},
            "right": {"device_id": "box1819152274", "T_cube_tcp": transform},
        },
    }


def _marker_tcp_solve_repo(tmp_path: Path) -> tuple[gateway.GatewayState, Path, Path]:
    """A repo laid out with everything the solve reads before it shells out."""
    repo_root = tmp_path / "repo"
    config_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_CONFIG
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(MARKER_TCP_TRACKING_CONFIG, encoding="utf-8")
    runner_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_RUNNER
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("#!/usr/bin/env bash\necho tracking\n", encoding="utf-8")

    bundle_path = config_path.parent / "marker_to_tcp_calibration.json"
    bundle_path.write_text(json.dumps(_marker_tcp_bundle(INHERITED_ROTATION)), encoding="utf-8")
    # ee_from_cube.marker_to_tcp_calibration_path is relative to the repo root.
    config_path.write_text(
        MARKER_TCP_TRACKING_CONFIG.replace(
            "config/marker_to_tcp_calibration.json",
            str(bundle_path.relative_to(repo_root)),
        ),
        encoding="utf-8",
    )

    summary_path = repo_root / "outputs" / "calibration" / "extr_run" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"joint_solution": {"cameras": {"cam_06": {}, "cam_07": {}, "cam_13": {}}}}),
        encoding="utf-8",
    )

    dataset_root = repo_root / "outputs" / "datasets" / "marker_tcp_raw"
    for episode in (3, 5):
        episode_dir = dataset_root / "episodes" / f"episode_{episode:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        # cam_99 is recorded but never calibrated: it must not reach the detector.
        for camera in ("cam_06", "cam_07", "cam_13", "cam_99"):
            (episode_dir / f"{camera}.mkv").write_bytes(b"")

    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 60}},
        recording=gateway.RecordingStatus(repoId="local/test", datasetRoot=str(dataset_root)),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )
    session_root = repo_root / "outputs" / "calibration" / "marker_tcp" / "session"
    session_root.mkdir(parents=True, exist_ok=True)
    state.marker_tcp_session = gateway.MarkerTcpSession(
        active=True,
        sessionName="session",
        sessionRoot=str(session_root),
        stage="capture",
        samples=[
            gateway.MarkerTcpSample(
                id=f"sample_{n:03d}",
                side="box1672693301",
                boxId="box1672693301",
                condition=f"same_mount_{n:02d}",
                status="saved",
                datasetRoot=str(dataset_root),
                episodeIndex=episode,
            )
            for n, episode in enumerate((3, 5), start=1)
        ],
    )
    return state, dataset_root, bundle_path


def _stub_marker_tcp_chain(monkeypatch, calls: list[list[str]]):
    """Run the solve without cv2/scipy: record commands, fake their outputs."""
    monkeypatch.setattr(gateway, "_marker_tcp_python", lambda _state: Path("/usr/bin/python3"))

    def fake_run(state, command, *, label, log_path, timeout_s=7200):
        calls.append(list(command))
        if "metrology.cli.build_rig_layout_from_cube" in command:
            out = Path(command[command.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "schema": "marker_layout_resolved/v1",
                        "layout_id": "left_resolved",
                        "units": "m",
                        "rig_frame_is_cube_frame": True,
                        "T_rig_cube": np.eye(4).tolist(),
                        "markers": [],
                    }
                ),
                encoding="utf-8",
            )
        elif "metrology.cli.detect_rig_markers" in command:
            Path(command[command.index("--out") + 1]).write_bytes(b"")
        elif "metrology.cli.track_marker_rig_in_base" in command:
            Path(command[command.index("--out-dir") + 1]).mkdir(parents=True, exist_ok=True)
        elif "metrology.cli.pivot_marker_tcp_calibration" in command:
            Path(command[command.index("--out") + 1]).write_text(
                json.dumps(
                    {
                        "fit": {"residual_mm": {"p95": 3.02}},
                        "socket_moved_between_episodes": False,
                        "primary_fit": "shared_anchor",
                    }
                ),
                encoding="utf-8",
            )
            emitted = Path(command[command.index("--emit-marker-to-tcp") + 1])
            emitted.write_text(json.dumps(_marker_tcp_bundle(INHERITED_ROTATION)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gateway, "_run_marker_tcp_command", fake_run)


def test_marker_tcp_solve_runs_the_metrology_chain_and_writes_a_production_bundle(
    tmp_path, monkeypatch
):
    state, dataset_root, _ = _marker_tcp_solve_repo(tmp_path)
    calls: list[list[str]] = []
    _stub_marker_tcp_chain(monkeypatch, calls)

    result = gateway._run_marker_tcp_solve(
        state, box_id="box1672693301", socket_beyond_tcp_mm="0", background=False
    )

    assert result["ok"] is True, result.get("error")
    steps = [
        next(part for part in command if part.startswith("metrology.cli.")) for command in calls
    ]
    assert steps == [
        "metrology.cli.build_rig_layout_from_cube",
        "metrology.cli.detect_rig_markers",
        "metrology.cli.track_marker_rig_in_base",
        "metrology.cli.pivot_marker_tcp_calibration",
    ]

    detect = calls[1]
    # Both saved episodes are solved together, renumbered into the subset dataset.
    assert detect[detect.index("--episodes") + 1 : detect.index("--cameras")] == ["0", "1"]
    # Recorded-but-uncalibrated cameras never reach the detector.
    cameras = detect[detect.index("--cameras") + 1 : detect.index("--dictionary")]
    assert cameras == ["cam_06", "cam_07", "cam_13"]
    assert detect[detect.index("--dictionary") + 1] == "DICT_APRILTAG_36h11"

    pivot = calls[3]
    assert pivot[pivot.index("--cube") + 1] == "left"
    assert pivot[pivot.index("--device-id") + 1] == "box1672693301"
    assert pivot[pivot.index("--socket-beyond-tcp-mm") + 1] == "0"
    # A single pivot point cannot observe rotation, so it is inherited verbatim
    # and its provenance is recorded rather than being silently re-fitted.
    assert pivot[pivot.index("--rotation-cube-tcp") + 1] == "0,0,1;-1,0,0;0,-1,0"
    assert "inherited from existing production bundle" in pivot[pivot.index("--rotation-source") + 1]

    session = state.marker_tcp_session
    # Solving one BOX must not close the session: the operator still has to be
    # able to record another sample or solve the second BOX.
    assert session.stage == "capture"
    assert Path(session.solvePath).is_file()
    summary = json.loads(Path(session.solveSummaryPath).read_text(encoding="utf-8"))
    assert summary["boxId"] == "box1672693301"
    assert summary["cubeName"] == "left"
    assert summary["rigFrameIsCubeFrame"] is True
    assert summary["cameras"] == ["cam_06", "cam_07", "cam_13"]
    assert summary["pivotP95Mm"] == 3.02
    assert [entry["sourceEpisodeIndex"] for entry in summary["episodes"]] == [3, 5]
    # The layout ships next to the bundle so a non-identity rig frame can travel
    # with it into the production tracker.
    assert (Path(session.solvePath).parent / gateway.DEFAULT_MARKER_LAYOUT_NAME).is_file()


def test_marker_tcp_solve_refuses_a_box_the_production_bundle_does_not_cover(tmp_path, monkeypatch):
    state, _, _ = _marker_tcp_solve_repo(tmp_path)
    _stub_marker_tcp_chain(monkeypatch, [])

    result = gateway._run_marker_tcp_solve(state, box_id="box_unknown", background=False)

    assert result["ok"] is False
    assert "box_unknown" in result["error"]


def test_queue_traj_gen_writes_a_marker_tcp_override_config_and_records_it(tmp_path, monkeypatch):
    state, dataset_root, bundle_path = _marker_tcp_solve_repo(tmp_path)
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4321
        stdout = []

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        return FakeProcess()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway, "_start_traj_gen_output_reader", lambda *_args: None)

    gateway._queue_traj_gen(state, dataset_root, marker_to_tcp_calibration_path=bundle_path)

    command = launched["command"]
    override_path = Path(command[command.index("--config") + 1])
    # The base config is left alone; the override lives under the dataset's meta/.
    assert override_path != state.repo_root / gateway.DEFAULT_EE_TRAJECTORY_CONFIG
    assert override_path.parent == dataset_root / "meta"
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    assert override["ee_from_cube"]["mode"] == "calibrated_marker_to_tcp"
    assert override["ee_from_cube"]["marker_to_tcp_calibration_path"] == str(bundle_path)
    # An identity rig frame needs no layout override: production keeps generating
    # cube corners from the same numbers.
    assert "marker_layout_path" not in override["cube_tracker"]

    item = gateway._processing_item_from_dataset(dataset_root)
    assert item["markerTcpCalibrationPath"] == str(bundle_path)
    assert str(bundle_path) in item["message"]


def test_queue_traj_gen_carries_a_cad_rig_frame_layout_into_the_tracker(tmp_path, monkeypatch):
    """A bundle solved in a CAD rig frame is not sufficient on its own.

    The tracker would otherwise report a cube-frame pose while the bundle's
    T_rig_tcp is expressed in the rig frame -- a silent frame error of exactly
    the CAD rotation, with no symptom until the EE labels are wrong.
    """
    state, dataset_root, _ = _marker_tcp_solve_repo(tmp_path)
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)

    solve_dir = Path(state.marker_tcp_session.sessionRoot) / "solve_cad"
    solve_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = solve_dir / "marker_to_tcp_calibration.json"
    bundle_path.write_text(json.dumps(_marker_tcp_bundle(INHERITED_ROTATION)), encoding="utf-8")
    layout_path = solve_dir / gateway.DEFAULT_MARKER_LAYOUT_NAME
    layout_path.write_text(
        json.dumps({"rig_frame_is_cube_frame": False, "markers": []}), encoding="utf-8"
    )

    class FakeProcess:
        pid = 4322
        stdout = []

        def poll(self):
            return None

    launched: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        launched["command"] = command
        return FakeProcess()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway, "_start_traj_gen_output_reader", lambda *_args: None)

    gateway._queue_traj_gen(state, dataset_root, marker_to_tcp_calibration_path=bundle_path)

    command = launched["command"]
    override = yaml.safe_load(
        Path(command[command.index("--config") + 1]).read_text(encoding="utf-8")
    )
    assert override["cube_tracker"]["marker_layout_path"] == str(layout_path)


def test_marker_tcp_solve_returns_immediately_and_reports_progress_through_the_session(
    tmp_path, monkeypatch
):
    """The POST must not hold the connection open for the whole solve.

    Detection alone decodes every frame of every camera; a synchronous response
    would sit past any browser or proxy timeout and read as a failure even when
    the solve succeeded. The panel already polls the snapshot, so the request
    only has to validate and hand off.
    """
    state, _, _ = _marker_tcp_solve_repo(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_run(_state, command, *, label, log_path, timeout_s=7200):
        started.set()
        assert release.wait(timeout=10)
        raise RuntimeError("stopped on purpose")

    monkeypatch.setattr(gateway, "_marker_tcp_python", lambda _state: Path("/usr/bin/python3"))
    monkeypatch.setattr(gateway, "_run_marker_tcp_command", blocking_run)

    result = gateway._run_marker_tcp_solve(state, box_id="box1672693301")

    assert result["ok"] is True
    assert started.wait(timeout=10)
    assert state.marker_tcp_session.stage == "solving"
    # A second request while one is running is refused rather than racing it.
    second = gateway._run_marker_tcp_solve(state, box_id="box1672693301")
    assert second["ok"] is False
    assert "已有解算在进行中" in second["error"]

    release.set()
    for _ in range(100):
        if state.marker_tcp_session.stage == "failed":
            break
        time.sleep(0.05)
    assert state.marker_tcp_session.stage == "failed"
    assert "stopped on purpose" in state.marker_tcp_session.message


def test_marker_tcp_solve_rejects_a_bad_socket_offset_before_spawning_anything(tmp_path, monkeypatch):
    state, _, _ = _marker_tcp_solve_repo(tmp_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("no subprocess should be reached for an invalid request")

    monkeypatch.setattr(gateway, "_run_marker_tcp_command", explode)

    result = gateway._run_marker_tcp_solve(state, box_id="box1672693301", socket_beyond_tcp_mm="很多")

    assert result["ok"] is False
    assert "必须是数字" in result["error"]


# --- calibration pointer: what was solved vs what production loads -----------


def _pointer_state(tmp_path: Path, *, config: str | None, solved_intr: str, solved_extr: str):
    state = _marker_tcp_gateway_state(tmp_path)
    state.calibration.intrinsicsRun = solved_intr
    state.calibration.extrinsicsRun = solved_extr
    if config is not None:
        path = tmp_path / gateway._TRACKING_CONFIG
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config, encoding="utf-8")
    gateway._PRODUCTION_RUNS_CACHE.clear()
    return state


_POINTER_CONFIG = """
calibration:
  root_dir: outputs/calibration
  intrinsics_run_name: intr_0804
  fixed_camera_run_name: extr_0804
"""


def test_production_runs_are_read_from_the_config_not_from_memory(tmp_path):
    """The two must come from different places or they cannot be compared.

    `state.calibration.*Run` starts equal to the config and is then overwritten
    by whatever a solve produced. Reading the file separately is the only way
    the panel can tell "solved" from "live".
    """
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_NEW", solved_extr="extr_NEW")

    production = gateway._production_calibration_runs(state)

    assert production["intrinsicsRun"] == "intr_0804"
    assert production["extrinsicsRun"] == "extr_0804"
    assert production["error"] == ""


def test_pointer_mismatch_flags_the_run_that_drifted(tmp_path):
    """The 2026-08-20 failure: new extrinsics solved, production never repointed."""
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_0804", solved_extr="calib_0820_extrinsics")

    mismatch = gateway._calibration_pointer_mismatch(state, gateway._production_calibration_runs(state))

    assert [f["kind"] for f in mismatch["fields"]] == ["extrinsics"]
    assert mismatch["fields"][0]["solved"] == "calib_0820_extrinsics"
    assert mismatch["fields"][0]["production"] == "extr_0804"
    # The message has to say the solve did not promote anything, not merely
    # that two strings differ -- "they differ" reads as a display glitch.
    assert "不会" in mismatch["message"]
    assert str(gateway._TRACKING_CONFIG) in mismatch["message"]
    assert mismatch["configPath"] == str(gateway._TRACKING_CONFIG)


def test_pointer_mismatch_is_empty_when_they_agree(tmp_path):
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_0804", solved_extr="extr_0804")

    assert gateway._calibration_pointer_mismatch(state, gateway._production_calibration_runs(state)) == {}


def test_pointer_mismatch_is_silent_when_the_config_cannot_be_read(tmp_path):
    """An unreadable config is not evidence of a mismatch.

    We do not know what production loads, and asserting a disagreement we
    cannot see would put a false alarm on top of a broken deployment.
    """
    state = _pointer_state(tmp_path, config=None, solved_intr="intr_NEW", solved_extr="extr_NEW")

    production = gateway._production_calibration_runs(state)
    assert production["error"]
    assert gateway._calibration_pointer_mismatch(state, production) == {}


def test_production_runs_follow_an_edit_rather_than_caching_boot_state(tmp_path):
    """The pointer is edited by hand and rewritten by deploys.

    A value cached at startup would keep reporting the old run as live, which
    is precisely the class of error this comparison is meant to surface.
    """
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_0804", solved_extr="calib_0820_extrinsics")
    assert gateway._calibration_pointer_mismatch(state, gateway._production_calibration_runs(state))

    path = tmp_path / gateway._TRACKING_CONFIG
    path.write_text(_POINTER_CONFIG.replace("extr_0804", "calib_0820_extrinsics"), encoding="utf-8")
    os.utime(path, (time.time() + 1, time.time() + 1))

    assert gateway._production_calibration_runs(state)["extrinsicsRun"] == "calib_0820_extrinsics"
    assert gateway._calibration_pointer_mismatch(state, gateway._production_calibration_runs(state)) == {}


def test_calibration_payload_carries_both_pointers(tmp_path):
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_0804", solved_extr="calib_0820_extrinsics")

    payload = gateway._calibration_payload(state)

    assert payload["extrinsicsRun"] == "calib_0820_extrinsics"
    assert payload["production"]["extrinsicsRun"] == "extr_0804"
    assert payload["pointerMismatch"]["fields"]


def test_calibration_payload_omits_pointer_mismatch_when_they_agree(tmp_path):
    """Absent, not an empty object.

    `{}` is truthy in the browser, so a client testing for the key found a
    "mismatch" with no `fields` and crashed the calibration page on the healthy
    path. The key documents itself as present only on disagreement; send that.
    """
    state = _pointer_state(tmp_path, config=_POINTER_CONFIG, solved_intr="intr_0804", solved_extr="extr_0804")

    payload = gateway._calibration_payload(state)

    assert "pointerMismatch" not in payload
