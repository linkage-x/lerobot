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

    assert gateway._resolve_video_path(state, dataset_root, "observation.images.cam_00") == mkv


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
        location_args = [part for part in cmd if isinstance(part, str) and part.startswith("location=")]
        output = Path(location_args[-1].split("=", 1)[1])
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
    left_fz = [0.0] * 239
    right_fz = [0.0] * 239
    left_fz[0] = 7.0
    right_fz[238] = 11.0
    rows = [
        {
            "sid": "box_touch_left",
            "t_rel_s": 0.5,
            "data": {"timestamp": 101, "fz_0p1N": left_fz},
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
    assert touch["left"]["fz"][0] == 7.0
    assert touch["left"]["activePoints"] == 1
    assert touch["right"]["timestamp"] == 202
    assert touch["right"]["fz"][238] == 11.0


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
    (sidecar_dir / "state_action.left.csv").write_text("header\n", encoding="utf-8")
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
                "trajectory_reachability": {
                    "total_trajectories": 1,
                    "num_unreachable_trajectories": 0,
                },
            }
        }), encoding="utf-8")
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
