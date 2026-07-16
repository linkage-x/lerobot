import json
from pathlib import Path

from tools.thor.gmsl2 import online_sync_burnin as burnin


def _write_sidecar(path: Path, camera: str, rows: int) -> None:
    path.write_text(
        "camera,encoded_frame_index,local_frame_number,sensor_timestamp_ns,sof_tsc_ns\n"
        + "".join(
            f"{camera},{i},{i + 1},{1000 + i},{1000000 + i * 16666667}\n"
            for i in range(rows)
        )
    )


def _write_episode(
    dataset_root: Path,
    *,
    rows: int,
    actual_frames: int,
    manifest_ok: bool = True,
    failure: str = "",
) -> Path:
    ep_dir = dataset_root / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    for camera in ("cam_06", "cam_07"):
        _write_sidecar(ep_dir / f"{camera}.argus_frame_metadata.csv", camera, rows)
    (ep_dir / "online_sync_manifest.json").write_text(json.dumps({
        "ok": manifest_ok,
        "failure": failure,
        "target_frames": actual_frames,
        "actual_frames": actual_frames,
        "active_cameras": ["cam_06", "cam_07"],
        "frame_count_by_camera": {"cam_06": actual_frames, "cam_07": actual_frames},
        "max_abs_delta_ns_by_camera": {"cam_06": 0, "cam_07": 5000},
        "tolerance_ns": 1_000_000,
    }))
    return ep_dir


def test_analyze_dataset_accepts_equal_online_sync_counts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_episode(dataset_root, rows=60, actual_frames=60)

    report = burnin.analyze_dataset(
        dataset_root,
        expected_episodes=1,
        expected_frames=60,
        tolerance_ns=1_000_000,
        run_ffprobe=False,
    )

    assert report["summary"]["ok"] is True
    assert report["episodes"][0]["sidecar_rows_by_camera"] == {
        "cam_06": 60,
        "cam_07": 60,
    }


def test_analyze_dataset_rejects_sidecar_count_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_episode(dataset_root, rows=59, actual_frames=60)

    report = burnin.analyze_dataset(
        dataset_root,
        expected_episodes=1,
        expected_frames=60,
        tolerance_ns=1_000_000,
        run_ffprobe=False,
    )

    assert report["summary"]["ok"] is False
    assert "episode_000000: cam_06 sidecar rows 59 != 60" in report["summary"]["failures"]


def test_analyze_dataset_surfaces_manifest_failure_detail(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_episode(
        dataset_root,
        rows=8,
        actual_frames=8,
        manifest_ok=False,
        failure=(
            "missing or out-of-tolerance full SOF cluster after recording start: "
            "cam_12: timed out waiting for Argus buffer after 1000 ms"
        ),
    )

    report = burnin.analyze_dataset(
        dataset_root,
        expected_episodes=1,
        expected_frames=60,
        tolerance_ns=1_000_000,
        run_ffprobe=False,
    )

    assert report["summary"]["ok"] is False
    assert any("cam_12: timed out waiting" in failure for failure in report["summary"]["failures"])


def test_run_ui_burnin_fail_fast_on_recorder_error(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "tools/thor/gmsl2"
    script_dir.mkdir(parents=True)
    (script_dir / "thor_record.py").write_text(
        "import sys, time\n"
        "print('Dataset root: /tmp/fake_online_sync_dataset', flush=True)\n"
        "print('Episode 0 ready', flush=True)\n"
        "sys.stdin.readline()\n"
        "print('ERROR: Online sync failed; episode will be discarded.', flush=True)\n"
        "time.sleep(30)\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sensors:\n"
        "  cameras:\n"
        "    defaults: {}\n"
        "box_collection:\n"
        "  enabled: false\n"
        "dataset: {}\n"
    )

    result = burnin.run_ui_burnin(
        config_path=config_path,
        repo_root=repo_root,
        dataset_root=tmp_path / "dataset",
        log_dir=tmp_path / "logs",
        episodes=10,
        episode_time_s=60.0,
        fps=60,
        sensor_ids=[6, 7],
        detect_all=False,
        preview=False,
        box_enabled=False,
        no_auto_recover=True,
        skip_argus_probe=False,
        run_timeout_s=5.0,
        continue_on_failure=False,
        debug=False,
    )

    assert result.ready_count == 1
    assert result.saved_count == 0
    assert result.protocol_failure == "ERROR: Online sync failed; episode will be discarded."
    assert result.timed_out is False
