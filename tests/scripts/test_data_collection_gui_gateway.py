from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def _write_sync_report(dataset_root: Path, *, status: str, schema_version: int) -> None:
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "fr3_sync_report.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "report_kind": "fr3_sync_audit",
                "status": status,
                "clock_semantics": "hardware_mixed",
                "total_frames": 300,
                "failures": []
                if status == "pass"
                else ["12/300 frame(s) exceed the 20.0 ms within-camera skew budget"],
                "cross_modality_bias_ms": {"camera.ee.capture_timestamp_s": -23.3},
                "summary": {"within_group_skew_over_budget_frames": 0 if status == "pass" else 12},
                "skew_evaluation": {
                    "budgets_ms": {"within_group": 20.0, "residual": 36.7, "bias": 60.0},
                    "within_group": {"camera": {"p95_ms": 7.8}},
                    "residual": {"p95_ms": 12.8},
                    "raw_all_device": {"p95_ms": 36.6},
                },
            }
        ),
        encoding="utf-8",
    )


def test_run_qc_fails_a_dataset_whose_timestamp_sync_failed(tmp_path):
    """QC is the export gate, so the alignment verdict has to be inside it, not beside it."""
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    _write_sync_report(dataset_root, status="fail", schema_version=3)

    qc = gateway._run_qc(dataset_root)

    assert qc["status"] == "fail"
    check = next(check for check in qc["checks"] if check["name"] == "timestamp_sync")
    assert check["status"] == "fail"
    assert "within-camera skew budget" in check["message"]
    assert qc["timestamp_sync"]["biasMs"]["camera.ee.capture_timestamp_s"] == pytest.approx(-23.3)
    assert qc["timestamp_sync"]["rawSkewP95Ms"] == pytest.approx(36.6)


def test_run_qc_passes_a_dataset_whose_timestamp_sync_passed(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    _write_sync_report(dataset_root, status="pass", schema_version=3)

    qc = gateway._run_qc(dataset_root)

    assert qc["status"] == "pass"
    check = next(check for check in qc["checks"] if check["name"] == "timestamp_sync")
    assert check["status"] == "pass"


def test_run_qc_does_not_believe_a_pre_v3_sync_verdict(tmp_path):
    """A v2 report judged the raw all-device spread, which failed every hardware episode.

    Recomputing is the only honest option, and this dataset carries no capture-timestamp column,
    so the recompute finds nothing to judge and QC stays silent rather than importing a verdict
    that was produced by a rule this gateway no longer applies.
    """
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    _write_sync_report(dataset_root, status="fail", schema_version=2)

    qc = gateway._run_qc(dataset_root)

    assert [check for check in qc["checks"] if check["name"] == "timestamp_sync"] == []
    assert qc["timestamp_sync"] is None
    assert qc["status"] == "pass"


def test_recorded_datasets_report_episodes_excluded_by_review(tmp_path):
    """The page has to show the exclusion before Build View, not explain it afterwards."""
    datasets_root = tmp_path / "outputs" / "datasets"
    dataset_root = datasets_root / "fr3_spacemouse_20260811_170748"
    _write_minimal_episode_dataset(dataset_root, total_episodes=3)
    (dataset_root / "meta" / "gui_annotations.json").write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": {
                    "0": {"episode": 0, "includeInTraining": True},
                    "2": {"episode": 2, "includeInTraining": False},
                },
            }
        ),
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=datasets_root,
        profile="workstation",
    )

    item = next(item for item in gateway._recorded_dataset_items(state) if item["name"] == dataset_root.name)

    assert item["excludedEpisodes"] == [2]


def test_building_a_view_from_only_excluded_episodes_is_refused(tmp_path):
    datasets_root = tmp_path / "outputs" / "datasets"
    dataset_root = datasets_root / "fr3_spacemouse_20260811_170748"
    _write_minimal_episode_dataset(dataset_root, total_episodes=2)
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.ee"] = {"dtype": "video", "shape": [64, 64, 3]}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    video_dir = dataset_root / "videos" / "observation.images.ee" / "chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").write_bytes(b"\0" * 16)
    (dataset_root / "meta" / "gui_annotations.json").write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": {
                    str(episode): {"episode": episode, "includeInTraining": False}
                    for episode in range(2)
                },
            }
        ),
        encoding="utf-8",
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=datasets_root,
        profile="workstation",
    )

    with pytest.raises(ValueError, match="nothing to build"):
        gateway._start_training_view(state, str(dataset_root), "delta_ee_from_prev_cmd")


def _qc_warned_state(tmp_path: Path) -> tuple[gateway.GatewayState, Path]:
    """A dataset whose QC ran and warned -- the state that used to look like "QC pending"."""
    datasets_root = tmp_path / "outputs" / "datasets"
    dataset_root = datasets_root / "thor_gmsl2_v1_20260811_170748"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    gateway._write_processing_meta_qc(
        dataset_root,
        {
            "status": "warn",
            "summary": "6 pass · 1 warn · 0 fail · 2 frames",
            "valid_frames_pct": 100.0,
            "checks": [
                {"name": "schema", "status": "pass", "message": "1 parquet file(s), 2 rows"},
                {
                    "name": "frame_count",
                    "status": "warn",
                    "message": "parquet has 2 rows but info.json declares 3",
                },
            ],
            "completed_at": gateway._now_iso(),
        },
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=datasets_root,
    )
    return state, dataset_root


def test_a_qc_warning_is_its_own_status_not_qc_pending(tmp_path):
    _, dataset_root = _qc_warned_state(tmp_path)

    item = gateway._processing_item_from_dataset(dataset_root)

    assert item["status"] == "qc_warn"
    # The message names the warning instead of reading like QC never ran.
    assert "info.json declares 3" in item["message"]


def test_export_refuses_a_warned_dataset_until_the_warning_is_acknowledged(tmp_path):
    state, dataset_root = _qc_warned_state(tmp_path)

    with pytest.raises(ValueError, match="confirm to export anyway"):
        gateway._start_approved_dataset_export(state, str(dataset_root))

    # The refusal has to carry the warnings, or the confirmation is uninformed.
    try:
        gateway._start_approved_dataset_export(state, str(dataset_root))
    except ValueError as exc:
        assert "frame_count" in str(exc)


def test_export_proceeds_on_a_warned_dataset_once_acknowledged(tmp_path, monkeypatch):
    state, dataset_root = _qc_warned_state(tmp_path)
    exported: dict[str, Path] = {}
    monkeypatch.setattr(
        gateway,
        "_copy_approved_v3_dataset_export",
        lambda _state, root, _item: exported.setdefault("root", root),
    )

    gateway._start_approved_dataset_export(state, str(dataset_root), acknowledge_warnings=True)

    assert exported["root"] == dataset_root
    # Overriding a warning is a decision, so it is logged with what was overridden.
    assert any(
        entry.level == "warn" and "frame_count" in entry.message for entry in state.events
    )


def test_export_still_refuses_a_dataset_whose_qc_never_ran(tmp_path):
    datasets_root = tmp_path / "outputs" / "datasets"
    dataset_root = datasets_root / "thor_gmsl2_v1_20260811_180000"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=datasets_root,
    )

    with pytest.raises(ValueError, match="must pass QC"):
        gateway._start_approved_dataset_export(state, str(dataset_root), acknowledge_warnings=True)


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
