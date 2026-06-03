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

from tools.thor.gmsl2 import gmsl2_record as gr
from tools.thor.gmsl2 import persistent_session as ps
from tools.thor.gmsl2 import persistent_session_worker as psw


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


def test_pipeline_desc_real_hardware_h264_with_exposure_and_argus_gain():
    cfg = ps.StreamConfig(
        sid=2, name="cam_02", codec="h264", exposure_us=8000, argus_gain=4,
    )
    desc = ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    assert "nvv4l2h264enc" in desc
    assert "h264parse" in desc
    assert "exposuretimerange=\"8000000 8000000\"" in desc
    assert "gainrange=\"4 4\"" in desc


def test_pipeline_desc_driver_gain_does_not_become_argus_gainrange():
    cfg = ps.StreamConfig(sid=2, name="cam_02", exposure_us=9999, gain=320)
    desc = ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    assert "exposuretimerange=\"9999000 9999000\"" in desc
    assert "gainrange" not in desc


def test_pipeline_desc_rejects_invalid_argus_gain():
    cfg = ps.StreamConfig(sid=2, name="cam_02", argus_gain=320)
    try:
        ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    except ValueError as exc:
        assert "argus_gain must be <= 4.0" in str(exc)
    else:
        raise AssertionError("expected invalid argus_gain to raise ValueError")


def test_gmsl2_record_pipeline_driver_gain_does_not_become_argus_gainrange(tmp_path):
    defaults = gr.CameraDefaults(exposure_us=9999, gain=320)
    cmd = gr.build_pipeline(2, tmp_path / "cam_02.mkv", defaults)
    assert "exposuretimerange=9999000 9999000" in cmd
    assert not any(part.startswith("gainrange=") for part in cmd)


def test_gmsl2_record_pipeline_uses_explicit_argus_gain(tmp_path):
    defaults = gr.CameraDefaults(argus_gain=4.0)
    cmd = gr.build_pipeline(2, tmp_path / "cam_02.mkv", defaults)
    assert "gainrange=4 4" in cmd


def test_pipeline_desc_with_recorder_preview_defers_preview_branch():
    cfg = ps.StreamConfig(
        sid=7, name="cam_07",
        preview_jpeg_path="/dev/shm/lerobot_preview/cam_07.jpg",
    )
    desc = ps.build_pipeline_desc(cfg, "/tmp/w.mkv")
    assert "tee name=t_7" in desc
    assert "queue name=recq_7" in desc
    assert "splitmuxsink name=mux_7" in desc
    assert desc.index("nvv4l2h265enc") < desc.index("h265parse")
    assert desc.index("h265parse") < desc.index("tee name=t_7")
    assert "prevq_7" not in desc
    assert "prevvalve_7" not in desc
    assert "nvvidconv name=prevconv_7" not in desc
    assert "jpegenc name=prevenc_7" not in desc
    assert "appsink name=preview_7" not in desc


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


_GST_CLOCK_TIME_NONE = 2**64 - 1


def _make_first_sample(pts_ns: int):
    """Mimic the minimal Gst.Sample / Gst.Buffer surface format-location uses."""
    buf = SimpleNamespace(pts=pts_ns)
    return SimpleNamespace(get_buffer=lambda: buf)


def _warmup_cfg(tmp_path, sid=0, name="cam_00"):
    warmup = tmp_path / "warmup"
    warmup.mkdir(exist_ok=True)
    return ps.StreamConfig(sid=sid, name=name), warmup


def test_fragment_dict_warmup_returns_warmup_path(tmp_path):
    cfg, warmup = _warmup_cfg(tmp_path)
    sample = _make_first_sample(int(0.5 * 1e9))
    info = psw._fragment_dict(
        cfg, 3, sample, ps.FragmentState.WARMUP, None, warmup,
        _GST_CLOCK_TIME_NONE,
    )
    assert info["path"].startswith(str(warmup))
    assert info["path"].endswith("_00003.mkv")
    assert info["state"] == ps.FragmentState.WARMUP.value
    assert info["first_pts_s"] == 0.5


def test_fragment_dict_episode_returns_episode_path_and_records_pts(tmp_path):
    cfg, warmup = _warmup_cfg(tmp_path, sid=2, name="cam_02")
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    sample = _make_first_sample(int(12.345 * 1e9))
    info = psw._fragment_dict(
        cfg, 4, sample, ps.FragmentState.EPISODE, episode_dir, warmup,
        _GST_CLOCK_TIME_NONE,
    )
    assert info["path"] == str(episode_dir / "cam_02.mkv")
    assert info["fragment_id"] == 4
    assert info["first_pts_s"] == 12.345
    assert info["state"] == ps.FragmentState.EPISODE.value


def test_fragment_dict_handles_clock_time_none(tmp_path):
    cfg, warmup = _warmup_cfg(tmp_path)
    sample = _make_first_sample(_GST_CLOCK_TIME_NONE)
    info = psw._fragment_dict(
        cfg, 0, sample, ps.FragmentState.WARMUP, None, warmup,
        _GST_CLOCK_TIME_NONE,
    )
    assert info["first_pts_s"] is None


def test_fragment_dict_handles_missing_buffer(tmp_path):
    cfg, warmup = _warmup_cfg(tmp_path)
    sample = SimpleNamespace(get_buffer=lambda: None)
    info = psw._fragment_dict(
        cfg, 0, sample, ps.FragmentState.WARMUP, None, warmup,
        _GST_CLOCK_TIME_NONE,
    )
    assert info["first_pts_s"] is None


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
