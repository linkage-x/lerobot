from pathlib import Path

from tools.thor.gmsl2 import argus_metadata_session as ams
from tools.thor.gmsl2 import gmsl2_record as gr
from tools.thor.gmsl2 import persistent_session as ps


def _recorder_config(tmp_path: Path, *, recorder_backend: str) -> gr.RecorderConfig:
    return gr.RecorderConfig(
        cameras=gr.CameraDefaults(recorder_backend=recorder_backend),
        hardware_sync=gr.HardwareSync(),
        argus_frame_sync=gr.ArgusFrameSync(),
        repo_id="local/test",
        single_task="test",
        dataset_root=tmp_path / "dataset",
        fps=60,
        num_episodes=1,
        episode_time_s=1.0,
        detect_all=False,
        sensor_ids=[6],
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


def _streams(*sids: int) -> list[ps.StreamConfig]:
    return [ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in sids]


class _PreflightSession(ams.ArgusMetadataCameraSession):
    def __init__(self, tmp_path: Path, outcomes: list[Exception | None]):
        super().__init__(
            _streams(6, 7),
            tmp_path / "warmup",
            repo_root=tmp_path,
            binary_path=tmp_path / "argus_metadata_video_recorder",
            auto_build=False,
        )
        self.outcomes = list(outcomes)
        self.calls: list[tuple[int, ...]] = []

    def _run_preflight_for_streams(self, streams: list[ps.StreamConfig]) -> None:
        self.calls.append(tuple(stream.sid for stream in streams))
        if not self.outcomes:
            return
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome


class _PreviewSession(ams.ArgusMetadataCameraSession):
    def __init__(self, tmp_path: Path):
        stream = ps.StreamConfig(
            sid=6,
            name="cam_06",
            preview_jpeg_path=str(tmp_path / "cam_06.jpg"),
        )
        super().__init__(
            [stream],
            tmp_path / "warmup",
            repo_root=tmp_path,
            binary_path=tmp_path / "argus_metadata_video_recorder",
            auto_build=False,
        )
        self.ensure_calls = 0

    def _ensure_preview(self, stream: ps.StreamConfig) -> None:
        self.ensure_calls += 1


class _StartOrderSession(_PreviewSession):
    def __init__(self, tmp_path: Path):
        super().__init__(tmp_path)
        self._active_sids = [6]
        self.recording_active_seen_by_disable: bool | None = None
        self.binary_path = tmp_path / "missing_argus_metadata_video_recorder"

    def disable_previews(self) -> None:
        self.recording_active_seen_by_disable = self._recording_active
        super().disable_previews()


class _DisconnectOrderSession(_StartOrderSession):
    pass


class _FakePreviewProc:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if not self.terminated and not self.killed else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class _ExitedTextProc:
    returncode = 0

    def poll(self) -> int:
        return 0


def test_camera_defaults_select_argus_metadata_backend() -> None:
    assert gr.CameraDefaults().recorder_backend == "argus_metadata"


def test_legacy_gmsl2_cli_rejects_argus_metadata_backend(tmp_path: Path) -> None:
    cfg = _recorder_config(tmp_path, recorder_backend="argus_metadata")

    message = gr._legacy_cli_backend_error(cfg)

    assert message is not None
    assert "legacy standalone GStreamer/splitmux CLI" in message
    assert "thor_record.py" in message


def test_legacy_gmsl2_cli_accepts_splitmux_backend(tmp_path: Path) -> None:
    cfg = _recorder_config(tmp_path, recorder_backend="gstreamer_splitmux")

    assert gr._legacy_cli_backend_error(cfg) is None


def test_argus_metadata_session_derives_name_prefix() -> None:
    streams = [
        ps.StreamConfig(sid=6, name="cam_06"),
        ps.StreamConfig(sid=7, name="cam_07"),
    ]

    assert ams.ArgusMetadataCameraSession._name_prefix_for_streams(streams) == "cam"


def test_argus_metadata_session_supports_custom_name_prefix() -> None:
    streams = [
        ps.StreamConfig(sid=6, name="thorcam_06"),
        ps.StreamConfig(sid=7, name="thorcam_07"),
    ]

    assert ams.ArgusMetadataCameraSession._name_prefix_for_streams(streams) == "thorcam"


def test_argus_metadata_session_rejects_non_sid_suffix() -> None:
    streams = [ps.StreamConfig(sid=6, name="front_left")]

    try:
        ams.ArgusMetadataCameraSession._name_prefix_for_streams(streams)
    except ValueError as exc:
        assert "requires stream names to end with" in str(exc)
    else:
        raise AssertionError("expected invalid stream name to raise ValueError")


def test_argus_metadata_session_record_command_passes_name_prefix(tmp_path: Path) -> None:
    session = ams.ArgusMetadataCameraSession(
        [
            ps.StreamConfig(sid=6, name="thorcam_06"),
            ps.StreamConfig(sid=7, name="thorcam_07"),
        ],
        tmp_path / "warmup",
        repo_root=tmp_path,
        binary_path=Path("/tmp/argus_metadata_video_recorder"),
        auto_build=False,
    )

    cmd = session._build_record_command(session._stream_cfgs, tmp_path / "episode", frames=2)

    assert cmd[cmd.index("--name-prefix") + 1] == "thorcam"
    assert cmd[cmd.index("--sids") + 1] == "6,7"
    assert cmd[cmd.index("--frames") + 1] == "2"
    assert cmd[cmd.index("--container") + 1] == "mkv"


def test_argus_metadata_session_keeps_connect_stable_delay(tmp_path: Path) -> None:
    session = ams.ArgusMetadataCameraSession(
        [ps.StreamConfig(sid=6, name="cam_06")],
        tmp_path / "warmup",
        repo_root=tmp_path,
        binary_path=Path("/tmp/argus_metadata_video_recorder"),
        auto_build=False,
        connect_stable_s=2.0,
    )

    assert session.connect_stable_s == 2.0


def test_argus_metadata_session_record_command_passes_container(tmp_path: Path) -> None:
    session = ams.ArgusMetadataCameraSession(
        [
            ps.StreamConfig(sid=6, name="cam_06", codec="h265", container="mp4"),
            ps.StreamConfig(sid=7, name="cam_07", codec="h265", container="mp4"),
        ],
        tmp_path / "warmup",
        repo_root=tmp_path,
        binary_path=Path("/tmp/argus_metadata_video_recorder"),
        auto_build=False,
    )

    cmd = session._build_record_command(session._stream_cfgs, tmp_path / "episode", frames=2)

    assert cmd[cmd.index("--codec") + 1] == "h265"
    assert cmd[cmd.index("--container") + 1] == "mp4"


def test_argus_metadata_stop_episode_uses_container_suffix(tmp_path: Path) -> None:
    session = ams.ArgusMetadataCameraSession(
        [
            ps.StreamConfig(sid=6, name="cam_06", codec="h265", container="mp4"),
        ],
        tmp_path / "warmup",
        repo_root=tmp_path,
        binary_path=Path("/tmp/argus_metadata_video_recorder"),
        auto_build=False,
    )
    session._proc = _ExitedTextProc()
    handle = ps.EpisodeHandle(
        idx=0,
        directory=tmp_path / "episode_000000",
        t0_wall_s=100.0,
        t0_mono_s=10.0,
    )

    session.stop_episode(handle)

    assert handle.fragments["cam_06"].path.name == "cam_06.mp4"


def test_argus_metadata_preview_command_writes_jpeg_file() -> None:
    stream = ps.StreamConfig(sid=6, name="cam_06", width=1920, height=1080, fps=60)

    cmd = ams.ArgusMetadataCameraSession._preview_command(stream, Path("/dev/shm/cam_06.jpg"))

    assert cmd[:4] == ["gst-launch-1.0", "-q", "nvarguscamerasrc", "sensor-id=6"]
    assert "jpegenc" in cmd
    assert "multifilesink" in cmd
    assert "location=/dev/shm/cam_06.jpg" in cmd
    assert "max-files=1" in cmd
    assert "video/x-raw,framerate=5/1" in cmd


def test_argus_metadata_preview_not_enabled_while_recording(tmp_path: Path) -> None:
    session = _PreviewSession(tmp_path)
    session._recording_active = True

    session.enable_previews()

    assert session.ensure_calls == 0


def test_argus_metadata_disable_previews_terminates_and_unlinks(tmp_path: Path) -> None:
    session = _PreviewSession(tmp_path)
    proc = _FakePreviewProc()
    preview_path = Path(session._stream_cfgs[0].preview_jpeg_path)
    preview_path.write_bytes(b"stale jpeg")
    session._preview_procs["cam_06"] = proc

    session.disable_previews()

    assert proc.terminated
    assert session._preview_procs == {}
    assert not preview_path.exists()


def test_argus_metadata_start_marks_recording_before_disabling_previews(tmp_path: Path) -> None:
    session = _StartOrderSession(tmp_path)

    try:
        session.start_episode(tmp_path / "episode_000000", 0)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected missing recorder binary to raise FileNotFoundError")

    assert session.recording_active_seen_by_disable is True
    assert session._recording_active is False


def test_argus_metadata_disconnect_keeps_recording_active_until_cleanup(tmp_path: Path) -> None:
    session = _DisconnectOrderSession(tmp_path)
    session._recording_active = True

    session.disconnect()

    assert session.recording_active_seen_by_disable is True
    assert session._recording_active is False


def test_argus_metadata_preflight_keeps_full_set_when_group_passes(tmp_path: Path) -> None:
    session = _PreflightSession(tmp_path, outcomes=[None])

    selected = session._preflight_streams(_streams(6, 7))

    assert [stream.sid for stream in selected] == [6, 7]
    assert session.calls == [(6, 7)]
    assert session.poll_errors() == []


def test_argus_metadata_preflight_drops_only_bad_camera(tmp_path: Path) -> None:
    session = _PreflightSession(
        tmp_path,
        outcomes=[
            RuntimeError("group failed"),
            None,
            RuntimeError("cam_07 timeout"),
            None,
        ],
    )

    selected = session._preflight_streams(_streams(6, 7))
    errors = session.poll_errors()

    assert [stream.sid for stream in selected] == [6]
    assert session.calls == [(6, 7), (6,), (7,), (6,)]
    assert len(errors) == 1
    assert errors[0].sid == 7
    assert errors[0].name == "cam_07"
    assert "dropping camera" in errors[0].message


def test_argus_metadata_preflight_drops_camera_named_by_group_error(tmp_path: Path) -> None:
    session = _PreflightSession(
        tmp_path,
        outcomes=[
            RuntimeError("cam_07: timed out waiting for frame metadata"),
            None,
        ],
    )

    selected = session._preflight_streams(_streams(6, 7))
    errors = session.poll_errors()

    assert [stream.sid for stream in selected] == [6]
    assert session.calls == [(6, 7), (6,)]
    assert len(errors) == 1
    assert errors[0].sid == 7
    assert "cam_07" in errors[0].message


def test_argus_metadata_preflight_raises_when_survivors_fail_together(tmp_path: Path) -> None:
    session = _PreflightSession(
        tmp_path,
        outcomes=[
            RuntimeError("group failed"),
            None,
            RuntimeError("cam_07 timeout"),
            RuntimeError("survivor group failed"),
        ],
    )

    try:
        session._preflight_streams(_streams(6, 7))
    except RuntimeError as exc:
        assert "survivor group failed" in str(exc)
    else:
        raise AssertionError("expected survivor group failure to raise")


def test_argus_metadata_preflight_raises_when_every_camera_fails(tmp_path: Path) -> None:
    session = _PreflightSession(
        tmp_path,
        outcomes=[
            RuntimeError("group failed"),
            RuntimeError("cam_06 timeout"),
            RuntimeError("cam_07 timeout"),
        ],
    )

    try:
        session._preflight_streams(_streams(6, 7))
    except RuntimeError as exc:
        assert "failed for every camera" in str(exc)
        assert "cam_06" in str(exc)
        assert "cam_07" in str(exc)
    else:
        raise AssertionError("expected all-camera preflight failure to raise")
