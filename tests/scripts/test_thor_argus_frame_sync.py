from pathlib import Path

from tools.thor.gmsl2.argus_frame_sync import (
    ArgusFrameMetadata,
    CameraFrameWindow,
    align_episode_frames,
    camera_frame_windows,
    frame_metadata_sidecar_path,
    read_frame_metadata_csv,
    write_alignment_report_json,
    write_frame_metadata_csv,
)
from tools.thor.gmsl2.argus_video_materialize import build_ffmpeg_select_command
from tools.thor.gmsl2.argus_video_materialize import _select_materialization_encoder


def _rows(camera: str, sof_values: list[int], *, local_start: int = 1) -> list[ArgusFrameMetadata]:
    return [
        ArgusFrameMetadata(
            camera=camera,
            encoded_frame_index=i,
            local_frame_number=local_start + i,
            sensor_timestamp_ns=sof - 20_000_000,
            sof_tsc_ns=sof,
            eof_tsc_ns=sof + 14_677_000,
            internal_frame_count=local_start + i + 2,
        )
        for i, sof in enumerate(sof_values)
    ]


def test_align_episode_frames_uses_sof_tsc_not_local_frame_number() -> None:
    period = 16_666_667
    cam06 = _rows("cam_06", [100_000_000 + i * period for i in range(4)])
    # cam_07 local frame numbers start at the same value, but the timestamp
    # sequence is one trigger period behind cam_06's local numbering.
    cam07 = _rows(
        "cam_07",
        [100_000_000 - period + 6_000 + i * period for i in range(5)],
    )

    alignment = align_episode_frames(
        {"cam_06": cam06, "cam_07": cam07},
        reference_camera="cam_06",
        tolerance_ns=50_000,
    )

    assert alignment.ok
    assert alignment.frame_count_by_camera() == {"cam_06": 4, "cam_07": 4}
    assert alignment.cameras["cam_07"].max_abs_delta_ns == 6_000

    # The first cam_06 reference frame matches cam_07 encoded frame 1, not
    # encoded frame 0. This is the important production behavior: local frame
    # number equality is not used as the sync contract.
    first_cam07_match = alignment.cameras["cam_07"].matches[0]
    assert first_cam07_match.encoded_frame_index == 1
    assert first_cam07_match.local_frame_number == 2
    assert first_cam07_match.delta_ns == 6_000

    windows = camera_frame_windows(alignment)
    assert windows["cam_06"].start_frame_index == 0
    assert windows["cam_06"].stop_frame_index == 4
    assert windows["cam_07"].start_frame_index == 1
    assert windows["cam_07"].stop_frame_index == 5
    assert windows["cam_07"].frame_count == 4


def test_align_episode_frames_fails_when_nearest_frame_exceeds_tolerance() -> None:
    cam06 = _rows("cam_06", [100_000_000, 116_666_667])
    cam07 = _rows("cam_07", [100_900_000, 117_566_667])

    alignment = align_episode_frames(
        {"cam_06": cam06, "cam_07": cam07},
        reference_camera="cam_06",
        tolerance_ns=500_000,
    )

    assert not alignment.ok
    assert alignment.frame_count_by_camera() == {"cam_06": 0, "cam_07": 0}
    assert "no synchronized frame set within tolerance" in alignment.failures


def test_align_episode_frames_drops_boundary_frames() -> None:
    cam06 = _rows("cam_06", [100_000_000, 116_666_667, 133_333_334])
    cam07 = _rows("cam_07", [100_003_000])

    alignment = align_episode_frames(
        {"cam_06": cam06, "cam_07": cam07},
        reference_camera="cam_06",
        tolerance_ns=50_000,
    )

    assert alignment.ok
    assert alignment.frame_count_by_camera() == {"cam_06": 1, "cam_07": 1}
    assert alignment.accepted_reference_indices == [0]
    assert alignment.dropped_reference_indices == [1, 2]


def test_align_episode_frames_fails_on_interior_drops() -> None:
    cam06 = _rows("cam_06", [100_000_000, 116_666_667, 133_333_334])
    cam07 = _rows("cam_07", [100_003_000, 133_336_000])

    alignment = align_episode_frames(
        {"cam_06": cam06, "cam_07": cam07},
        reference_camera="cam_06",
        tolerance_ns=50_000,
    )

    assert not alignment.ok
    assert alignment.frame_count_by_camera() == {"cam_06": 2, "cam_07": 2}
    assert alignment.accepted_reference_indices == [0, 2]
    assert alignment.dropped_reference_indices == [1]
    assert "dropped 1 reference frames inside synchronized window" in alignment.failures


def test_frame_metadata_csv_round_trip_and_report(tmp_path: Path) -> None:
    rows = _rows("cam_06", [100_000_000, 116_666_667])
    sidecar = frame_metadata_sidecar_path(tmp_path, "cam_06")
    write_frame_metadata_csv(sidecar, rows)

    loaded = read_frame_metadata_csv(sidecar)
    assert loaded == rows

    alignment = align_episode_frames({"cam_06": loaded}, reference_camera="cam_06")
    report = tmp_path / "argus_frame_alignment.json"
    write_alignment_report_json(report, alignment)

    text = report.read_text(encoding="utf-8")
    assert '"ok": true' in text
    assert '"reference_camera": "cam_06"' in text


def test_build_ffmpeg_select_command_uses_exact_frame_window() -> None:
    window = CameraFrameWindow(
        camera="cam_07",
        start_frame_index=3,
        stop_frame_index=9,
        frame_count=6,
    )

    cmd = build_ffmpeg_select_command(
        Path("/tmp/cam_07.mkv"),
        Path("/tmp/cam_07_aligned.mkv"),
        window,
        fps=60,
        codec="h265",
    )

    assert "select=between(n\\,3\\,8),setpts=N/(60*TB)" in cmd
    assert cmd[cmd.index("-frames:v") + 1] == "6"
    assert "libx265" in cmd


def test_build_ffmpeg_select_command_accepts_preselected_encoder() -> None:
    window = CameraFrameWindow(
        camera="cam_06",
        start_frame_index=0,
        stop_frame_index=2,
        frame_count=2,
    )

    cmd = build_ffmpeg_select_command(
        Path("/tmp/cam_06.mkv"),
        Path("/tmp/cam_06_aligned.mkv"),
        window,
        fps=60,
        codec="h265",
        encoder="libx265",
    )

    assert cmd[cmd.index("-c:v") + 1] == "libx265"


def test_select_materialization_encoder_reports_missing_ffmpeg_encoder() -> None:
    try:
        _select_materialization_encoder("h265", available_encoders={"libx264"})
    except RuntimeError as exc:
        assert "libx265" in str(exc)
        assert "Argus-aligned videos" in str(exc)
    else:
        raise AssertionError("expected missing materialization encoder to fail")
