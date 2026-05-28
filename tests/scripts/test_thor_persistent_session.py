"""Unit tests for tools.thor.gmsl2.persistent_session.

These tests do NOT require a working GStreamer install — they mock out the
``gi.repository.Gst`` module and exercise only:

* ``build_pipeline_desc`` string assembly (real-hardware + test-source paths)
* The format-location callback's state machine
* ``write_episode_meta`` schema

The real GStreamer plumbing (PLAYING transitions, split-now emit, async
finalize) is verified by the standalone demo on Thor hardware.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from tools.thor.gmsl2 import persistent_session as ps


# ---------------------------------------------------------------------------
# build_pipeline_desc
# ---------------------------------------------------------------------------


def test_pipeline_desc_real_hardware_h265():
    cfg = ps.StreamConfig(sid=7, name="cam_07")
    desc = ps.build_pipeline_desc(cfg, "/tmp/cam_07_warmup_%05d.mkv")
    assert "nvarguscamerasrc sensor-id=7" in desc
    assert "nvv4l2h265enc" in desc
    assert "h265parse" in desc
    assert "splitmuxsink name=mux_7" in desc
    assert "async-finalize=true" in desc
    assert "max-size-time=0 max-size-bytes=0" in desc
    assert "/tmp/cam_07_warmup_%05d.mkv" in desc
    # Splitmuxsink can only cut on an IDR; iframeinterval alone is not enough.
    assert "idrinterval=" in desc
    iframe = f"iframeinterval={cfg.iframe_interval}"
    idr = f"idrinterval={cfg.iframe_interval}"
    assert iframe in desc and idr in desc


def test_pipeline_desc_real_hardware_h264_with_exposure_and_gain():
    cfg = ps.StreamConfig(
        sid=2, name="cam_02", codec="h264", exposure_us=8000, gain=4,
    )
    desc = ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    assert "nvv4l2h264enc" in desc
    assert "h264parse" in desc
    assert "exposuretimerange=\"8000000 8000000\"" in desc
    assert "gainrange=\"4 4\"" in desc


def test_pipeline_desc_test_source_uses_software_encoder():
    cfg = ps.StreamConfig(sid=0, name="cam_00", use_test_source=True, codec="h264")
    desc = ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    assert "videotestsrc" in desc
    assert "x264enc" in desc
    assert "nvarguscamerasrc" not in desc
    assert "nvv4l2h264enc" not in desc
    # The overlay text helps tell streams apart on the dev host.
    assert "cam_00" in desc


# ---------------------------------------------------------------------------
# format-location callback state machine
# ---------------------------------------------------------------------------


def _make_stream(tmp_path, sid=0, name="cam_00") -> ps._Stream:
    cfg = ps.StreamConfig(sid=sid, name=name)
    warmup = tmp_path / "warmup"
    warmup.mkdir(exist_ok=True)
    # The session is only used as a host for `_on_fragment_opened`; we don't
    # need its real lifecycle methods here.
    fake_session = SimpleNamespace(
        _Gst=SimpleNamespace(MessageType=SimpleNamespace()),
        _on_fragment_opened=lambda stream, info: None,
        _record_error=lambda err: None,
    )
    return ps._Stream(cfg, warmup, fake_session)  # type: ignore[arg-type]


def _make_first_sample(pts_ns: int):
    """Mimic the minimal Gst.Sample / Gst.Buffer surface format-location uses."""
    buf = SimpleNamespace(pts=pts_ns)
    return SimpleNamespace(get_buffer=lambda: buf)


def test_format_location_warmup_returns_warmup_path(tmp_path):
    stream = _make_stream(tmp_path)
    sample = _make_first_sample(int(0.5 * 1e9))
    path = stream._on_format_location_full(None, 3, sample)
    assert path.startswith(str(tmp_path / "warmup"))
    assert path.endswith("_00003.mkv")
    assert stream.fragment_history[-1].state == ps.FragmentState.WARMUP
    assert stream.fragment_history[-1].first_pts_s == 0.5
    # WARMUP fragments must not be advertised as "last episode fragment".
    assert stream.last_episode_fragment is None


def test_format_location_episode_returns_episode_path_and_records_pts(tmp_path):
    stream = _make_stream(tmp_path, sid=2, name="cam_02")
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    stream.state = ps.FragmentState.EPISODE
    stream.current_episode_dir = episode_dir
    sample = _make_first_sample(int(12.345 * 1e9))
    path = stream._on_format_location_full(None, 4, sample)
    assert path == str(episode_dir / "cam_02.mkv")
    info = stream.last_episode_fragment
    assert info is not None
    assert info.fragment_id == 4
    assert info.first_pts_s == 12.345
    assert info.state == ps.FragmentState.EPISODE


def test_format_location_handles_clock_time_none(tmp_path):
    stream = _make_stream(tmp_path)
    # Gst.CLOCK_TIME_NONE is 2**64 - 1; we filter that out.
    sample = _make_first_sample(2**64 - 1)
    stream._on_format_location_full(None, 0, sample)
    assert stream.fragment_history[-1].first_pts_s is None


def test_format_location_handles_missing_buffer(tmp_path):
    stream = _make_stream(tmp_path)
    sample = SimpleNamespace(get_buffer=lambda: None)
    stream._on_format_location_full(None, 0, sample)
    assert stream.fragment_history[-1].first_pts_s is None


# ---------------------------------------------------------------------------
# write_episode_meta schema
# ---------------------------------------------------------------------------


def test_write_episode_meta_emits_sync_reference_with_first_pts(tmp_path):
    session = ps.PersistentCameraSession(
        streams=[], warmup_dir=tmp_path / "w",
    )
    ep_dir = tmp_path / "episode_000003"
    ep_dir.mkdir()
    handle = ps.EpisodeHandle(
        idx=3, directory=ep_dir,
        t0_wall_s=1716700000.0, t0_mono_s=12345.0,
        stop_wall_s=1716700005.5,
        fragments={
            "cam_02": ps.FragmentInfo(
                sid=2, name="cam_02", fragment_id=1,
                path=ep_dir / "cam_02.mkv",
                first_pts_s=0.0166, first_wall_s=1716700000.05,
                state=ps.FragmentState.EPISODE,
            ),
            "cam_07": ps.FragmentInfo(
                sid=7, name="cam_07", fragment_id=1,
                path=ep_dir / "cam_07.mkv",
                first_pts_s=0.0333, first_wall_s=1716700000.07,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )
    path = session.write_episode_meta(handle)
    assert path == ep_dir / "meta.json"
    meta = json.loads(path.read_text())
    assert meta["episode_index"] == 3
    assert meta["duration_s"] == 5.5
    sync = meta["sync_reference"]
    assert sync["split_now_wall_s"] == 1716700000.0
    assert sync["camera_first_pts_s"] == {"cam_02": 0.0166, "cam_07": 0.0333}
    assert sync["camera_first_wall_s"]["cam_02"] == 1716700000.05
    # cameras[] preserves per-stream entries.
    names = {entry["name"] for entry in meta["cameras"]}
    assert names == {"cam_02", "cam_07"}


# ---------------------------------------------------------------------------
# discard_episode unlinks fragment files
# ---------------------------------------------------------------------------


def test_discard_episode_removes_fragment_files(tmp_path):
    session = ps.PersistentCameraSession(
        streams=[], warmup_dir=tmp_path / "w",
    )
    ep_dir = tmp_path / "episode_000007"
    ep_dir.mkdir()
    file_a = ep_dir / "cam_02.mkv"
    file_b = ep_dir / "cam_07.mkv"
    file_a.write_bytes(b"fake mkv a")
    file_b.write_bytes(b"fake mkv b")
    handle = ps.EpisodeHandle(
        idx=7, directory=ep_dir,
        t0_wall_s=time.time(), t0_mono_s=time.monotonic(),
        fragments={
            "cam_02": ps.FragmentInfo(
                sid=2, name="cam_02", fragment_id=1, path=file_a,
                first_pts_s=0.0, first_wall_s=0.0,
                state=ps.FragmentState.EPISODE,
            ),
            "cam_07": ps.FragmentInfo(
                sid=7, name="cam_07", fragment_id=1, path=file_b,
                first_pts_s=0.0, first_wall_s=0.0,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )
    # No streams attached, so the in-flight discard branch is a no-op; the
    # public contract is "the EPISODE files are gone".
    session.discard_episode(handle)
    assert not file_a.exists()
    assert not file_b.exists()


# ---------------------------------------------------------------------------
# _next_episode_index
# ---------------------------------------------------------------------------


def test_next_episode_index_empty(tmp_path):
    assert ps._next_episode_index(tmp_path) == 0


def test_cleanup_warmup_files_keeps_recent_per_sid(tmp_path):
    warmup = tmp_path / "warmup"
    warmup.mkdir()
    # 5 fragments for sid 0, 3 for sid 7
    for i in range(5):
        (warmup / f"cam_00_warmup_{i:05d}.mkv").write_bytes(b"x")
    for i in range(3):
        (warmup / f"cam_07_warmup_{i:05d}.mkv").write_bytes(b"x")
    # Non-matching files should be ignored.
    (warmup / "other_file.mkv").write_bytes(b"x")
    session = ps.PersistentCameraSession(streams=[], warmup_dir=warmup)
    deleted = session.cleanup_warmup_files(keep_last_n=2)
    assert deleted == 3 + 1  # sid 0: 5-2=3 deleted; sid 7: 3-2=1 deleted
    remaining = sorted(p.name for p in warmup.glob("cam_*_warmup_*.mkv"))
    assert remaining == [
        "cam_00_warmup_00003.mkv",
        "cam_00_warmup_00004.mkv",
        "cam_07_warmup_00001.mkv",
        "cam_07_warmup_00002.mkv",
    ]
    # Non-matching file untouched.
    assert (warmup / "other_file.mkv").exists()


def test_cleanup_warmup_files_zero_keeps_none(tmp_path):
    warmup = tmp_path / "warmup"
    warmup.mkdir()
    for i in range(3):
        (warmup / f"cam_00_warmup_{i:05d}.mkv").write_bytes(b"x")
    session = ps.PersistentCameraSession(streams=[], warmup_dir=warmup)
    deleted = session.cleanup_warmup_files(keep_last_n=0)
    assert deleted == 3
    assert not list(warmup.glob("cam_*_warmup_*.mkv"))


def test_next_episode_index_skips_non_episode_dirs(tmp_path):
    (tmp_path / "episode_000000").mkdir()
    (tmp_path / "episode_000004").mkdir()
    (tmp_path / "episode_garbage").mkdir()
    (tmp_path / "random_dir").mkdir()
    assert ps._next_episode_index(tmp_path) == 5
