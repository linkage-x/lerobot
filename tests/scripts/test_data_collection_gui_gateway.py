from __future__ import annotations

import json
from pathlib import Path

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


def test_traj_gen_is_explicitly_not_implemented(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "outputs" / "datasets" / "episode_set"
    _write_minimal_episode_dataset(dataset_root, total_episodes=1)
    state = gateway.GatewayState(
        repo_root=repo_root,
        config_path=repo_root / "config.yaml",
        config={"dataset": {"repo_id": "local/test", "root": str(dataset_root), "fps": 30}},
        recording=gateway.RecordingStatus(repoId="local/test"),
        replay=gateway.ReplayStatus(dataset="local/test"),
        datasets_root=dataset_root.parent,
    )

    try:
        gateway._queue_traj_gen(state, dataset_root)
    except NotImplementedError as exc:
        assert "待实现" in str(exc)
        assert "Generate EE Trajectory" in str(exc)
    else:
        raise AssertionError("traj-gen should report that it is not implemented")


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
