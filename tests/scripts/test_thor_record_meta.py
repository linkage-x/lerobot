import importlib.util
import json
import sys
from pathlib import Path

from tools.thor.box_sdk import box_client as bc
from tools.thor.gmsl2 import argus_frame_sync as afs
from tools.thor.gmsl2 import gmsl2_record as gr
from tools.thor.gmsl2 import persistent_session as ps


def _load_thor_record_module():
    path = Path("tools/thor/gmsl2/thor_record.py")
    spec = importlib.util.spec_from_file_location("tools.thor.gmsl2.thor_record", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _recorder_config(tmp_path: Path) -> gr.RecorderConfig:
    return gr.RecorderConfig(
        cameras=gr.CameraDefaults(recorder_backend="argus_metadata"),
        hardware_sync=gr.HardwareSync(),
        argus_frame_sync=gr.ArgusFrameSync(enabled=True, required=True),
        online_sync=gr.ArgusOnlineSync(),
        repo_id="local/test",
        single_task="test",
        dataset_root=tmp_path / "dataset",
        fps=60,
        num_episodes=1,
        episode_time_s=1.0,
        detect_all=False,
        sensor_ids=[6],
        exclude_sensor_ids=[],
        name_prefix="cam",
        spawn_stagger_s=0.0,
        connect_stable_s=0.0,
        connect_timeout_s=1.0,
        connect_first_fragment_timeout_s=0.0,
        two_phase_connect=False,
        stop_on_stream_exit=True,
        recording_preview_enabled=False,
        recording_preview_on_demand=True,
        recording_preview_idle_ttl_s=0.0,
        recording_preview_stagger_s=0.0,
        recording_preview_stale_s=0.0,
        recording_preview_watchdog_s=0.0,
        stream_health_poll_s=0.0,
        warmup_roll_s=0.0,
        warmup_keep_last_n=0,
    )


def test_thor_record_meta_records_connect_stream_errors(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    ep_dir = tmp_path / "episode_000000"
    ep_dir.mkdir()
    handle = ps.EpisodeHandle(
        idx=0,
        directory=ep_dir,
        t0_wall_s=100.0,
        t0_mono_s=10.0,
        stop_wall_s=101.0,
        fragments={
            "cam_06": ps.FragmentInfo(
                sid=6,
                name="cam_06",
                fragment_id=0,
                path=ep_dir / "cam_06.mkv",
                first_pts_s=None,
                first_wall_s=100.0,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )
    connect_errors = [
        ps.StreamError(
            sid=3,
            name="cam_03",
            message="Argus metadata preflight failed; dropping camera: cam_03 timeout",
        )
    ]

    meta_path = thor_record._write_episode_meta(
        handle,
        _recorder_config(tmp_path),
        locked=[3, 6],
        argus_failed=[],
        connect_stream_errors=connect_errors,
        box_cfg=bc.BoxFleetConfig(enabled=False),
        box_snapshots=[],
        stop_reason="save",
        wallclock_start_utc="2026-07-03T00:00:00+00:00",
        wallclock_end_utc="2026-07-03T00:00:01+00:00",
    )

    meta = json.loads(meta_path.read_text())
    assert meta["active_camera_sids"] == [6]
    assert meta["argus_failed_sids"] == []
    assert meta["connect_failed_sids"] == [3]
    assert meta["connect_stream_errors"] == [
        {
            "sid": 3,
            "name": "cam_03",
            "message": "Argus metadata preflight failed; dropping camera: cam_03 timeout",
        }
    ]


def test_wallclock_start_is_derived_from_recording_origin() -> None:
    thor_record = _load_thor_record_module()

    assert thor_record._wallclock_utc_from_wall_s(100.0) == "1970-01-01T00:01:40+00:00"


def test_argus_frame_sync_uses_stop_marker_to_trim_drain_frames(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    ep_dir = tmp_path / "episode_000000"
    ep_dir.mkdir()
    cfg = _recorder_config(tmp_path)
    cfg.cameras.recorder_backend = "gstreamer_splitmux"
    cfg.argus_frame_sync.reference_strategy = "camera"
    cfg.argus_frame_sync.reference_camera = "cam_06"
    cfg.argus_frame_sync.tolerance_ms = 1.0

    def rows(camera: str, values: list[int]) -> list[afs.ArgusFrameMetadata]:
        return [
            afs.ArgusFrameMetadata(
                camera=camera,
                encoded_frame_index=i,
                local_frame_number=i + 1,
                sensor_timestamp_ns=sof - 1_000,
                sof_tsc_ns=sof,
            )
            for i, sof in enumerate(values)
        ]

    afs.write_frame_metadata_csv(
        afs.frame_metadata_sidecar_path(ep_dir, "cam_06"),
        rows("cam_06", [100_000_000, 116_666_667, 133_333_334, 150_000_001]),
    )
    afs.write_frame_metadata_csv(
        afs.frame_metadata_sidecar_path(ep_dir, "cam_07"),
        rows("cam_07", [100_010_000, 116_676_667, 133_343_334, 150_010_001]),
    )
    (ep_dir / "argus_recording_markers.json").write_text(json.dumps({
        "reference_camera": "cam_06",
        "start_sof_tsc_ns": 100_000_000,
        "stop_sof_tsc_ns_exclusive": 133_333_335,
    }))

    handle = ps.EpisodeHandle(
        idx=0,
        directory=ep_dir,
        t0_wall_s=100.0,
        t0_mono_s=10.0,
        stop_wall_s=101.0,
        fragments={
            "cam_06": ps.FragmentInfo(
                sid=6,
                name="cam_06",
                fragment_id=0,
                path=ep_dir / "cam_06.mkv",
                first_pts_s=None,
                first_wall_s=100.0,
                state=ps.FragmentState.EPISODE,
            ),
            "cam_07": ps.FragmentInfo(
                sid=7,
                name="cam_07",
                fragment_id=0,
                path=ep_dir / "cam_07.mkv",
                first_pts_s=None,
                first_wall_s=100.0,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )

    payload, failure = thor_record._evaluate_argus_frame_sync(handle, cfg)

    assert failure is None
    assert payload["ok"] is True
    assert payload["recording_markers"]["stop_sof_tsc_ns_exclusive"] == 133_333_335
    assert payload["reference_frame_count"] == 3
    assert payload["frame_count_by_camera"] == {"cam_06": 3, "cam_07": 3}


def _online_sync_handle(tmp_path: Path) -> ps.EpisodeHandle:
    ep_dir = tmp_path / "episode_000000"
    ep_dir.mkdir()
    return ps.EpisodeHandle(
        idx=0,
        directory=ep_dir,
        t0_wall_s=100.0,
        t0_mono_s=10.0,
        fragments={
            "cam_06": ps.FragmentInfo(
                sid=6,
                name="cam_06",
                fragment_id=0,
                path=ep_dir / "cam_06.mkv",
                first_pts_s=None,
                first_wall_s=100.0,
                state=ps.FragmentState.EPISODE,
            ),
            "cam_07": ps.FragmentInfo(
                sid=7,
                name="cam_07",
                fragment_id=0,
                path=ep_dir / "cam_07.mkv",
                first_pts_s=None,
                first_wall_s=100.0,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )


def _write_online_sync_sidecars(ep_dir: Path, *, rows_per_camera: int) -> None:
    for camera in ("cam_06", "cam_07"):
        rows = [
            afs.ArgusFrameMetadata(
                camera=camera,
                encoded_frame_index=i,
                local_frame_number=i + 1,
                sensor_timestamp_ns=100_000_000 + i,
                sof_tsc_ns=100_000_000 + i * 16_666_667,
            )
            for i in range(rows_per_camera)
        ]
        afs.write_frame_metadata_csv(afs.frame_metadata_sidecar_path(ep_dir, camera), rows)


def test_online_sync_manifest_gate_accepts_valid_manifest(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    handle = _online_sync_handle(tmp_path)
    cfg = _recorder_config(tmp_path)
    cfg.cameras.recorder_backend = "argus_online_sync"
    _write_online_sync_sidecars(handle.directory, rows_per_camera=3)
    (handle.directory / "online_sync_manifest.json").write_text(json.dumps({
        "ok": True,
        "actual_frames": 3,
        "frame_count_by_camera": {"cam_06": 3, "cam_07": 3},
        "max_abs_delta_ns_by_camera": {"cam_06": 0, "cam_07": 4_000},
    }))

    payload, failure = thor_record._evaluate_online_sync_manifest(handle, cfg)

    assert failure is None
    assert payload["ok"] is True
    assert payload["actual_frames"] == 3
    assert payload["sidecar_counts"] == {"cam_06": 3, "cam_07": 3}


def test_online_sync_manifest_reports_sof_to_monotonic_clock_bridge(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    handle = _online_sync_handle(tmp_path)
    cfg = _recorder_config(tmp_path)
    cfg.cameras.recorder_backend = "argus_online_sync"
    for camera, acquire_delay_ns in (("cam_06", 2_000_000), ("cam_07", 1_000_000)):
        rows = []
        for index in range(3):
            sof_tsc_ns = 1_000_000_000 + index * 16_666_667
            eof_tsc_ns = sof_tsc_ns + 14_000_000
            rows.append(afs.ArgusFrameMetadata(
                camera=camera,
                encoded_frame_index=index,
                local_frame_number=index + 1,
                sensor_timestamp_ns=sof_tsc_ns - 100_000_000,
                sof_tsc_ns=sof_tsc_ns,
                eof_tsc_ns=eof_tsc_ns,
                host_acquired_monotonic_ns=eof_tsc_ns - 100_000_000 + acquire_delay_ns,
            ))
        afs.write_frame_metadata_csv(afs.frame_metadata_sidecar_path(handle.directory, camera), rows)
    (handle.directory / "online_sync_manifest.json").write_text(json.dumps({
        "ok": True,
        "actual_frames": 3,
        "frame_count_by_camera": {"cam_06": 3, "cam_07": 3},
        "max_abs_delta_ns_by_camera": {"cam_06": 0, "cam_07": 4_000},
    }))

    payload, failure = thor_record._evaluate_online_sync_manifest(handle, cfg)

    assert failure is None
    bridge = payload["camera_clock_bridge"]
    assert bridge["scale"] == 1.0
    assert bridge["offset_ns"] == -100_000_000
    assert bridge["sample_count"] == 6
    assert bridge["clock_pair_residual_ns"]["max"] == 0
    assert bridge["host_acquire_delay_from_sof_ns"]["min"] == 15_000_000
    assert bridge["host_acquire_delay_from_sof_ns"]["max"] == 16_000_000


def test_online_sync_manifest_gate_rejects_missing_manifest(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    handle = _online_sync_handle(tmp_path)
    cfg = _recorder_config(tmp_path)
    cfg.cameras.recorder_backend = "argus_online_sync"

    payload, failure = thor_record._evaluate_online_sync_manifest(handle, cfg)

    assert failure == "online_sync_missing_manifest"
    assert payload["ok"] is False
    assert payload["failures"] == ["missing online_sync_manifest.json"]


def test_online_sync_manifest_gate_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    thor_record = _load_thor_record_module()
    handle = _online_sync_handle(tmp_path)
    cfg = _recorder_config(tmp_path)
    cfg.cameras.recorder_backend = "argus_online_sync"
    _write_online_sync_sidecars(handle.directory, rows_per_camera=3)
    (handle.directory / "online_sync_manifest.json").write_text(json.dumps({
        "ok": True,
        "actual_frames": 3,
        "frame_count_by_camera": {"cam_06": 3, "cam_07": 2},
        "max_abs_delta_ns_by_camera": {"cam_06": 0, "cam_07": 4_000},
    }))

    payload, failure = thor_record._evaluate_online_sync_manifest(handle, cfg)

    assert failure == "online_sync_failed"
    assert payload["ok"] is False
    assert "cam_07 manifest frame count 2 != 3" in payload["failures"]
