from __future__ import annotations

import json
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

    assert snapshot["configSummary"]["repoId"] == "local/thor_gmsl2_11ch_v1"
    assert snapshot["configSummary"]["fps"] == 60
    devices_by_kind: dict[str, list[str]] = {}
    for device in snapshot["devices"]:
        devices_by_kind.setdefault(device["kind"], []).append(device["id"])
    # 11-camera GMSL2 rig (detect_all => sids 0..15 placeholder before connect).
    assert "camera" in devices_by_kind
    assert all(cid.startswith("cam_") for cid in devices_by_kind["camera"])
    assert len(devices_by_kind["camera"]) >= 11
    # Box collection sensors are surfaced as a distinct device kind.
    assert "box_collection" in devices_by_kind
    assert {"box_gripper", "box_imu", "box_trigger"}.issubset(set(devices_by_kind["box_collection"]))
    # Old Hikrobot / Pika devices are no longer in the default rig.
    assert "handheld_gripper" not in devices_by_kind


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


def test_recorder_env_adds_repo_import_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    env = gateway._recorder_env(tmp_path)
    paths = env["PYTHONPATH"].split(":")

    assert paths[:2] == [str(tmp_path / "src"), str(tmp_path)]
    assert paths[2] == "/existing/path"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_mujoco_replay_command_uses_repo_relative_dataset_path(tmp_path):
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

    assert command[1] == str(repo_root / "tools" / "fr3" / "fr3_sim_record_replay.py")
    assert "--dataset=outputs/datasets/fr3_sim_record_20260421_072232" in command
    assert "--episode=2" in command
    assert "--fps=60" in command
    assert "--no-viewer" not in command


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
    sidecar = dataset_root / "derived" / "hikon_cube_tracking_in_robot_base"
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


def test_replay_timeline_includes_camera_cube_overlays(tmp_path):
    dataset_root = tmp_path / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    tracking_run = tmp_path / "outputs" / "tracking_analysis" / "episode_set_tracking_in_robot_base"
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


def test_traj_gen_starts_hikon_tracking_with_selected_dataset_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    script_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_SCRIPT
    config_path = repo_root / gateway.DEFAULT_EE_TRAJECTORY_CONFIG
    script_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('tracking')\n", encoding="utf-8")
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
    assert str(script_path) in command
    assert "--config" in command
    assert str(config_path) in command
    assert "--dataset-root" in command
    assert command[command.index("--dataset-root") + 1] == str(dataset_root)
    assert str(dataset_root) in state.processing_processes
    item = gateway._processing_item_from_dataset(dataset_root)
    assert item["status"] == "running"
    assert "Hikon cube tracking" in item["message"]


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
        "mujoco_replay_result=status=complete completed_frames=2 total_frames=2 "
        "avg_pos_mm=2.0 max_pos_mm=4.0 avg_rot_deg=1.0 max_rot_deg=3.0",
    )
    gateway._finish_mujoco_validation(state, 0)
    gateway._preflight_replay(state)
    command = gateway._real_replay_command(state, dataset_root)

    assert state.replay.mujocoValidation["status"] == "passed"
    assert state.replay.safety == "ready"
    assert "current MuJoCo validation" in state.replay.message
    assert "--dataset=outputs/datasets/episode_set" in command
    assert "--episode=0" in command
    assert "--robot-ip=192.168.1.99" in command
    assert "--gripper-port=/dev/ttyUSB9" in command


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
                "repo_id": "local/thor_gmsl2_11ch_v1",
                "root": "outputs/datasets/thor_gmsl2_11ch_v1",
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
    assert state.config["dataset"]["repo_id"] == "local/thor_gmsl2_11ch_v1"


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
    assert state.recording.datasetRoot == "outputs/datasets/thor_gmsl2_11ch_v1"


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
