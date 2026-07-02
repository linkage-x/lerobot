"""Unit tests for the multi-process redesign of ``PersistentCameraSession``.

These tests prove the structural fix for the PR2 regression — namely:

  * the post-PR2 in-process model held N Argus CaptureSession's on one
    RPC socket and deadlocked under load (root-cause analysis in
    ``tools/thor/DEPLOYMENT.md``);
  * the new design spawns one ``persistent_session_worker`` subprocess
    per camera so the daemon sees N independent clients.

The tests deliberately do **not** spin up real subprocesses or import
``gi.repository``. They drive ``_apply_event_to_proxy`` and the public
``PersistentCameraSession`` surface with hand-crafted ``_StreamProxy``
instances, which is both fast and runnable on dev hosts without Argus.

What's verified
---------------

* event-dispatch contract (``_apply_event_to_proxy`` handles every event
  type the worker can emit, including unknown events without raising)
* ``_StreamProxy`` lifecycle helpers (``wait_ready``, ``drain_errors``,
  ``start_episode`` / ``stop_episode`` command emission)
* ``PersistentCameraSession.connect()`` raises a clear ``RuntimeError``
  when ANY proxy fails to reach PLAYING — isolation in action
* ``PersistentCameraSession.poll_errors()`` aggregates errors across all
  proxies and clears them
* ``PersistentCameraSession.stop_episode()`` waits for each proxy's
  ``episode_done`` event and collects the resulting ``FragmentInfo``
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.thor.gmsl2 import persistent_session as ps


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Minimal stand-in for mp.Process — never actually forks."""

    last_kwargs = None
    all_kwargs = []

    def __init__(self, *args, **kwargs):
        self._alive = True
        self.args = args
        self.kwargs = kwargs
        _FakeProcess.last_kwargs = kwargs
        _FakeProcess.all_kwargs.append(kwargs)

    def start(self):
        return None

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def is_alive(self):
        return self._alive


class _FakeCtx:
    """multiprocessing context substitute that hands out plain queues and
    a non-forking Process class."""

    def Queue(self):
        # The reader thread treats ``queue.Empty`` identically to
        # ``mp.queues.Empty`` (it imports the stdlib one), so we can use a
        # thread-safe queue.Queue with no behavior changes.
        return queue.Queue()

    def Process(self, *args, **kwargs):
        return _FakeProcess(*args, **kwargs)


def _make_proxy(tmp_path, sid=2, name="cam_02", *, on_fragment_opened=None):
    """Build a _StreamProxy WITHOUT spawning a real child process.

    Tests can then push events through _apply_event_to_proxy directly or
    drive the proxy's state machine manually.
    """
    cfg = ps.StreamConfig(sid=sid, name=name)
    return ps._StreamProxy(
        cfg, tmp_path / "warmup", _FakeCtx(),
        on_fragment_opened=on_fragment_opened,
    )


def _frag_event(sid, name, fragment_id, *, state, path,
                first_pts_s=0.5, first_wall_s=None):
    return (
        "fragment",
        {
            "sid": sid,
            "name": name,
            "fragment_id": fragment_id,
            "path": str(path),
            "first_pts_s": first_pts_s,
            "first_wall_s": first_wall_s if first_wall_s is not None else time.time(),
            "state": state.value,
        },
    )


# ---------------------------------------------------------------------------
# _apply_event_to_proxy — event dispatch contract
# ---------------------------------------------------------------------------


def test_stream_proxy_preview_commands_enqueue_worker_messages(tmp_path):
    proxy = _make_proxy(tmp_path)
    proxy.enable_preview()
    proxy.disable_preview()
    assert proxy.cmd_q.get_nowait() == ("enable_preview",)
    assert proxy.cmd_q.get_nowait() == ("disable_preview",)


def test_stream_proxy_spawn_passes_ready_timeout_to_worker(tmp_path):
    _FakeProcess.last_kwargs = None
    _FakeProcess.all_kwargs = []
    cfg = ps.StreamConfig(sid=2, name="cam_02")
    proxy = ps._StreamProxy(
        cfg, tmp_path / "warmup", _FakeCtx(), ready_timeout_s=0.42,
    )
    proxy.spawn()
    assert _FakeProcess.last_kwargs["kwargs"] == {
        "ready_timeout_s": 0.42, "two_phase": False,
    }
    proxy.disconnect()


def test_session_preview_controls_skip_streams_without_preview_path(tmp_path):
    cfg_preview = ps.StreamConfig(
        sid=2, name="cam_02",
        preview_jpeg_path="/dev/shm/lerobot_preview/cam_02.jpg",
    )
    cfg_plain = ps.StreamConfig(sid=3, name="cam_03")
    session = ps.PersistentCameraSession([cfg_preview, cfg_plain], tmp_path / "warmup")
    session._streams = {
        2: ps._StreamProxy(cfg_preview, tmp_path / "warmup", _FakeCtx()),
        3: ps._StreamProxy(cfg_plain, tmp_path / "warmup", _FakeCtx()),
    }
    session.enable_previews(stagger_s=0.0)
    session.disable_previews()
    assert session._streams[2].cmd_q.get_nowait() == ("enable_preview",)
    assert session._streams[2].cmd_q.get_nowait() == ("disable_preview",)
    assert session._streams[3].cmd_q.empty()


def test_apply_event_playing_sets_ready_evt(tmp_path):
    proxy = _make_proxy(tmp_path)
    assert not proxy.ready_evt.is_set()
    ps._apply_event_to_proxy(proxy, ("playing",))
    assert proxy.ready_evt.is_set()
    assert proxy.errors == []


def test_apply_event_fragment_warmup_appends_history_not_last_episode(tmp_path):
    proxy = _make_proxy(tmp_path)
    evt = _frag_event(
        2, "cam_02", 3, state=ps.FragmentState.WARMUP,
        path=tmp_path / "warmup" / "cam_02_warmup_00003.mkv",
    )
    ps._apply_event_to_proxy(proxy, evt)
    assert len(proxy.fragment_history) == 1
    assert proxy.fragment_history[-1].state == ps.FragmentState.WARMUP
    assert proxy.last_episode_fragment is None


def test_apply_event_fragment_episode_records_last_episode_fragment(tmp_path):
    proxy = _make_proxy(tmp_path)
    ep_dir = tmp_path / "episode_000000"
    evt = _frag_event(
        2, "cam_02", 5, state=ps.FragmentState.EPISODE,
        path=ep_dir / "cam_02.mkv", first_pts_s=12.345,
    )
    ps._apply_event_to_proxy(proxy, evt)
    info = proxy.last_episode_fragment
    assert info is not None
    assert info.fragment_id == 5
    assert info.first_pts_s == 12.345
    assert info.state == ps.FragmentState.EPISODE


def test_apply_event_on_fragment_opened_callback_fires_with_proxy(tmp_path):
    seen = []
    proxy = _make_proxy(
        tmp_path, on_fragment_opened=lambda p, info: seen.append((p.cfg.sid, info)),
    )
    evt = _frag_event(
        2, "cam_02", 1, state=ps.FragmentState.WARMUP,
        path=tmp_path / "warmup" / "cam_02_warmup_00001.mkv",
    )
    ps._apply_event_to_proxy(proxy, evt)
    assert len(seen) == 1
    assert seen[0][0] == 2
    assert isinstance(seen[0][1], ps.FragmentInfo)


def test_apply_event_error_appends_and_unblocks_ready(tmp_path):
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ("error", "NvArgusCameraSrc: TIMEOUT (6)", ""))
    # ready_evt must flip so connect() does not block forever waiting for a
    # dead worker — this is the exact deadlock condition we are fixing.
    assert proxy.ready_evt.is_set()
    assert len(proxy.errors) == 1
    assert "TIMEOUT" in proxy.errors[0].message
    # Sticky failure marker drives the preview watchdog (see refresh_stale_previews).
    assert proxy.recording_failed


def test_apply_event_eos_appends_and_unblocks_ready(tmp_path):
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ("eos",))
    assert proxy.ready_evt.is_set()
    assert len(proxy.errors) == 1
    assert "EOS" in proxy.errors[0].message
    assert proxy.recording_failed


def test_apply_event_episode_done_with_fragment_updates_last(tmp_path):
    proxy = _make_proxy(tmp_path)
    ep_dir = tmp_path / "episode_000007"
    payload = {
        "sid": 2, "name": "cam_02", "fragment_id": 1,
        "path": str(ep_dir / "cam_02.mkv"),
        "first_pts_s": 0.02, "first_wall_s": time.time(),
        "state": ps.FragmentState.EPISODE.value,
    }
    ps._apply_event_to_proxy(proxy, ("episode_done", payload))
    assert proxy.episode_done_evt.is_set()
    assert proxy.last_episode_fragment is not None
    assert proxy.last_episode_fragment.fragment_id == 1


def test_apply_event_episode_done_without_fragment_only_sets_event(tmp_path):
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ("episode_done", None))
    assert proxy.episode_done_evt.is_set()
    assert proxy.last_episode_fragment is None


def test_apply_event_disconnected_sets_disconnected_evt(tmp_path):
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ("disconnected",))
    assert proxy.disconnected_evt.is_set()


def test_apply_event_ignores_garbage(tmp_path):
    """Unknown / malformed events must not raise."""
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ())
    ps._apply_event_to_proxy(proxy, "nope")  # type: ignore[arg-type]
    ps._apply_event_to_proxy(proxy, ("mystery_event", 1, 2, 3))
    # Nothing should have been recorded.
    assert proxy.errors == []
    assert not proxy.ready_evt.is_set()
    assert not proxy.recording_failed


# ---------------------------------------------------------------------------
# refresh_stale_previews — a dead recording stream must not spin the watchdog
#
# Regression guard for the 2026-06-03 "Preview stale: restarted ..." storm:
# once all cameras EOS while armed/idle, the preview JPEGs go permanently stale
# and the watchdog used to disable+enable the (futile) preview branch on every
# dead pipeline every 2s forever. recording_failed now gates that.
# ---------------------------------------------------------------------------


def _session_with_preview_proxy(tmp_path, sid=2, name="cam_02"):
    jpg = tmp_path / f"{name}.jpg"
    jpg.write_bytes(b"x")
    cfg = ps.StreamConfig(sid=sid, name=name, preview_jpeg_path=str(jpg))
    session = ps.PersistentCameraSession([cfg], tmp_path / "warmup")
    proxy = ps._StreamProxy(cfg, tmp_path / "warmup", _FakeCtx())
    session._streams = {sid: proxy}
    return session, proxy, jpg


def _set_jpeg_age(jpg, age_s):
    stamp = time.time() - age_s
    os.utime(jpg, (stamp, stamp))


def test_refresh_stale_previews_restarts_branch_when_recording_alive(tmp_path):
    session, proxy, jpg = _session_with_preview_proxy(tmp_path)
    _set_jpeg_age(jpg, age_s=10.0)
    restarted = session.refresh_stale_previews(max_age_s=3.0)
    # Recording alive + stale preview -> bounce the lossy branch (legit case).
    assert restarted == ["cam_02"]
    assert proxy.cmd_q.get_nowait() == ("disable_preview",)
    assert proxy.cmd_q.get_nowait() == ("enable_preview",)


def test_refresh_stale_previews_skips_dead_recording_stream(tmp_path):
    session, proxy, jpg = _session_with_preview_proxy(tmp_path)
    _set_jpeg_age(jpg, age_s=10.0)
    proxy.recording_failed = True
    restarted = session.refresh_stale_previews(max_age_s=3.0)
    # Dead upstream -> nothing restarted, no worker command enqueued.
    assert restarted == []
    assert proxy.cmd_q.empty()


def test_refresh_stale_previews_reports_dead_stream_once_no_storm(tmp_path):
    session, proxy, jpg = _session_with_preview_proxy(tmp_path)
    _set_jpeg_age(jpg, age_s=10.0)
    proxy.recording_failed = True
    assert not proxy._preview_down_logged
    session.refresh_stale_previews(max_age_s=3.0)
    assert proxy._preview_down_logged  # reported on first detection only
    # Many subsequent ticks stay silent and restart nothing -> the storm is gone.
    for _ in range(5):
        assert session.refresh_stale_previews(max_age_s=3.0) == []
    assert proxy.cmd_q.empty()


def test_refresh_stale_previews_resets_down_flag_when_jpeg_fresh(tmp_path):
    session, proxy, jpg = _session_with_preview_proxy(tmp_path)
    _set_jpeg_age(jpg, age_s=10.0)
    proxy.recording_failed = True
    session.refresh_stale_previews(max_age_s=3.0)
    assert proxy._preview_down_logged
    # A fresh JPEG (preview producing frames again) resets the one-shot guard so
    # a later failure is reported anew instead of being permanently muted.
    _set_jpeg_age(jpg, age_s=0.0)
    session.refresh_stale_previews(max_age_s=3.0)
    assert not proxy._preview_down_logged


# ---------------------------------------------------------------------------
# _StreamProxy lifecycle helpers
# ---------------------------------------------------------------------------


def test_stream_proxy_wait_ready_returns_true_on_playing_event(tmp_path):
    proxy = _make_proxy(tmp_path)
    threading.Timer(0.05, lambda: ps._apply_event_to_proxy(proxy, ("playing",))).start()
    assert proxy.wait_ready(timeout_s=1.0) is True
    assert proxy.errors == []


def test_stream_proxy_wait_ready_returns_false_on_error_event(tmp_path):
    proxy = _make_proxy(tmp_path)
    threading.Timer(
        0.05,
        lambda: ps._apply_event_to_proxy(proxy, ("error", "bad", "")),
    ).start()
    assert proxy.wait_ready(timeout_s=1.0) is False
    assert any("bad" in e.message for e in proxy.errors)


def test_stream_proxy_wait_ready_timeout_records_synthetic_error(tmp_path):
    proxy = _make_proxy(tmp_path)
    # No event will arrive.
    assert proxy.wait_ready(timeout_s=0.1) is False
    assert any(
        "worker did not become ready" in e.message for e in proxy.errors
    )


def test_stream_proxy_drain_errors_consumes_buffer(tmp_path):
    proxy = _make_proxy(tmp_path)
    proxy.errors.append(ps.StreamError(sid=2, name="cam_02", message="a"))
    proxy.errors.append(ps.StreamError(sid=2, name="cam_02", message="b"))
    out = proxy.drain_errors()
    assert [e.message for e in out] == ["a", "b"]
    assert proxy.drain_errors() == []


def test_format_stream_error_includes_debug_source():
    err = ps.StreamError(
        sid=3,
        name="cam_03",
        message="Internal data stream error.",
        debug="src=nvarguscamerasrc3; NvBufSurfaceFromFd Failed.",
    )

    text = ps.format_stream_error(err)

    assert text.startswith("cam_03(Internal data stream error.")
    assert "src=nvarguscamerasrc3" in text
    assert "NvBufSurfaceFromFd Failed" in text


def test_stream_proxy_start_episode_emits_command(tmp_path):
    proxy = _make_proxy(tmp_path)
    ep_dir = tmp_path / "episode_000003"
    proxy.start_episode(ep_dir)
    cmd = proxy.cmd_q.get(timeout=0.5)
    assert cmd == ("start_episode", str(ep_dir))
    # start_episode also resets episode-tracking state so a stale fragment
    # from the previous episode does not leak into the next handle.
    assert proxy.last_episode_fragment is None
    assert not proxy.episode_done_evt.is_set()


def test_stream_proxy_stop_episode_emits_command(tmp_path):
    proxy = _make_proxy(tmp_path)
    proxy.stop_episode()
    assert proxy.cmd_q.get(timeout=0.5) == ("stop_episode",)


# ---------------------------------------------------------------------------
# PersistentCameraSession — isolation properties
# ---------------------------------------------------------------------------


def _connect_with_fake_workers(
    session: ps.PersistentCameraSession,
    behaviors: dict[int, str],
) -> None:
    """Drive session.connect() without forking real subprocesses.

    ``behaviors`` maps sid -> "playing" | "error" | "timeout". We swap the
    ctx factory for one that hands out plain queues + non-forking Process
    objects, then patch _StreamProxy.spawn to deferred-push the matching
    event via a threading.Timer (so wait_ready exercises its real timing
    path instead of being short-circuited).
    """
    session._ctx_factory = lambda: _FakeCtx()

    def fake_spawn(self):
        behavior = behaviors.get(self.cfg.sid, "playing")
        if behavior == "playing":
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(self, ("playing",)),
            ).start()
        elif behavior == "error":
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "synthetic test failure", ""),
                ),
            ).start()
        # "timeout" -> no event posted; wait_ready will time out

    with patch.object(ps._StreamProxy, "spawn", fake_spawn):
        session.connect()


def test_connect_partial_failure_keeps_successful_proxies(tmp_path):
    """If 1/3 cameras fails, connect() returns success with 2 active
    streams and leaves the error in poll_errors() for the recorder."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3, 4)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    _connect_with_fake_workers(
        session, {2: "playing", 3: "error", 4: "playing"},
    )
    # No raise. Active sids drop cam_03; cam_02 and cam_04 are still up.
    assert session.active_sids == [2, 4]
    errs = session.poll_errors()
    messages = [e.message for e in errs]
    assert any("synthetic test failure" in m for m in messages)
    assert all(e.sid == 3 for e in errs)


def test_connect_partial_failure_handles_timeout(tmp_path):
    """One camera never reports PLAYING — must surface as a timeout error
    without hanging the whole session."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.15,
    )

    _connect_with_fake_workers(session, {2: "playing", 3: "timeout"})
    assert session.active_sids == [2]
    errs = session.poll_errors()
    assert any("did not become ready" in e.message for e in errs)
    assert all(e.sid == 3 for e in errs)


def test_connect_raises_when_all_proxies_fail(tmp_path):
    """If every camera fails, connect() must raise — otherwise the caller
    would happily proceed with zero active streams."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.15,
    )

    try:
        _connect_with_fake_workers(session, {2: "error", 3: "timeout"})
    except RuntimeError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected connect() to raise when all proxies fail")

    assert "failed on all" in msg
    assert "cam_02" in msg and "cam_03" in msg
    assert session._streams == {}


def test_active_sids_starts_empty(tmp_path):
    session = ps.PersistentCameraSession([], tmp_path / "warmup")
    assert session.active_sids == []


def test_connect_stable_window_retries_post_playing_error(tmp_path):
    cfgs = [ps.StreamConfig(sid=2, name="cam_02")]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3, connect_stable_s=0.05,
    )

    spawn_count = {"n": 0}

    def flaky_after_playing(self):
        spawn_count["n"] += 1
        attempt = spawn_count["n"]
        ps._apply_event_to_proxy(self, ("playing",))
        if attempt == 1:
            threading.Timer(
                0.01,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "post-playing timeout", ""),
                ),
            ).start()

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", flaky_after_playing):
        session.connect()

    assert session.active_sids == [2]
    assert spawn_count["n"] == 2
    assert session.poll_errors() == []


def test_connect_retries_playing_stream_without_first_fragment(tmp_path):
    cfgs = [ps.StreamConfig(sid=2, name="cam_02")]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0,
        ready_timeout_s=0.3,
        first_fragment_timeout_s=0.05,
    )
    spawn_count = {"n": 0}

    def spawn_without_fragment_first(self):
        spawn_count["n"] += 1
        attempt = spawn_count["n"]
        ps._apply_event_to_proxy(self, ("playing",))
        if attempt == 2:
            ps._apply_event_to_proxy(self, ("fragment", {
                "sid": 2,
                "name": "cam_02",
                "fragment_id": 0,
                "path": str(tmp_path / "warmup" / "cam_02_warmup_00000.mkv"),
                "first_pts_s": 0.0,
                "first_wall_s": time.time(),
                "state": ps.FragmentState.WARMUP.value,
            }))

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", spawn_without_fragment_first):
        session.connect()

    assert spawn_count["n"] == 2
    assert session.active_sids == [2]
    assert session.poll_errors() == []


def test_connect_retry_rescues_flaky_sid(tmp_path):
    """A sid that fails on first spawn but succeeds on retry must end up
    active and produce no surviving error. This mirrors the recover_argus.sh
    "sid=3 fails first probe, passes after daemon restart + retry" pattern
    that motivated PR4."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    spawn_count: dict[int, int] = {}

    def flaky_spawn(self):
        sid = self.cfg.sid
        spawn_count[sid] = spawn_count.get(sid, 0) + 1
        if sid == 3 and spawn_count[sid] == 1:
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "first-try transient", ""),
                ),
            ).start()
        else:
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(self, ("playing",)),
            ).start()

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", flaky_spawn):
        session.connect()

    assert session.active_sids == [2, 3]
    assert spawn_count == {2: 1, 3: 2}, "sid=3 must be retried exactly once"
    assert session.poll_errors() == [], (
        "rescued sid must not surface a residual error to the caller"
    )


def test_connect_retry_drops_sid_that_keeps_failing(tmp_path):
    """If a sid fails BOTH the original spawn and the retry, it's dropped
    and the error stays in poll_errors() for the recorder to warn about."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    def fail_sid_3_always(self):
        if self.cfg.sid == 3:
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "permanent failure", ""),
                ),
            ).start()
        else:
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(self, ("playing",)),
            ).start()

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", fail_sid_3_always):
        session.connect()

    assert session.active_sids == [2]
    errs = session.poll_errors()
    assert any(
        e.sid == 3 and "permanent failure" in e.message for e in errs
    )


def test_connect_retries_run_sequentially_not_in_parallel(tmp_path):
    """Retry one sid at a time so Argus/NVMM allocation storms do not repeat.

    Each retry posts the rescue event after a fixed delay. Sequential retry
    should take roughly N * delay; a parallel retry would complete near delay.
    """
    fail_sids = (2, 3, 4, 5, 7)
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in fail_sids
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=1.0,
    )

    spawn_count: dict[int, int] = {}
    rescue_delay_s = 0.05

    def flaky_spawn(self):
        sid = self.cfg.sid
        spawn_count[sid] = spawn_count.get(sid, 0) + 1
        if spawn_count[sid] == 1:
            threading.Timer(
                0.01,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "first-try", ""),
                ),
            ).start()
        else:
            threading.Timer(
                rescue_delay_s,
                lambda: ps._apply_event_to_proxy(self, ("playing",)),
            ).start()

    session._ctx_factory = lambda: _FakeCtx()
    t0 = time.monotonic()
    with patch.object(ps._StreamProxy, "spawn", flaky_spawn):
        session.connect()
    elapsed = time.monotonic() - t0

    assert sorted(session.active_sids) == sorted(fail_sids)
    assert all(spawn_count[s] == 2 for s in fail_sids)
    serial_lower_bound = len(fail_sids) * rescue_delay_s
    assert elapsed >= serial_lower_bound * 0.8, (
        f"Phase 3 retry appears parallel: elapsed={elapsed:.3f}s, "
        f"serial should take at least ~{serial_lower_bound:.3f}s"
    )

def test_parallel_retry_drops_only_truly_dead_sids(tmp_path):
    """When some retries succeed and others permanently fail concurrently,
    the success path must keep its sids active and the failure path must
    leave only those sids' errors in poll_errors(). No cross-contamination."""
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}")
        for sid in (2, 3, 4, 5)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    spawn_count: dict[int, int] = {}

    def half_flaky_spawn(self):
        sid = self.cfg.sid
        spawn_count[sid] = spawn_count.get(sid, 0) + 1
        if spawn_count[sid] == 1:
            threading.Timer(
                0.01,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "first-try fail", ""),
                ),
            ).start()
        else:
            # Even sids rescued, odd sids stay dead.
            if sid % 2 == 0:
                threading.Timer(
                    0.05,
                    lambda: ps._apply_event_to_proxy(self, ("playing",)),
                ).start()
            else:
                threading.Timer(
                    0.05,
                    lambda: ps._apply_event_to_proxy(
                        self, ("error", "permanent dead", ""),
                    ),
                ).start()

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", half_flaky_spawn):
        session.connect()

    assert sorted(session.active_sids) == [2, 4]
    errs = session.poll_errors()
    dead_sids = sorted(e.sid for e in errs)
    assert dead_sids == [3, 5]
    assert all("permanent dead" in e.message for e in errs)


def test_connect_retry_does_not_retry_more_than_once_per_sid(tmp_path):
    """Bound the recovery effort: a permanently dead sid must result in
    exactly 2 spawn calls (original + 1 retry), never more, so a hardware
    failure can't trap connect() in an infinite loop."""
    cfgs = [ps.StreamConfig(sid=3, name="cam_03")]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.15,
    )

    spawn_count = {"n": 0}

    def always_fail(self):
        spawn_count["n"] += 1
        threading.Timer(
            0.02,
            lambda: ps._apply_event_to_proxy(self, ("error", "no camera", "")),
        ).start()

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", always_fail):
        try:
            session.connect()
        except RuntimeError:
            pass  # all-failed expected

    assert spawn_count["n"] == 2, (
        f"spawn must be called exactly 2x (original + 1 retry), got {spawn_count['n']}"
    )


def test_connect_spawns_and_waits_each_stream_before_next_spawn(tmp_path):
    """Connect must not start all Argus clients before waiting for PLAYING.

    The hardware-stable strategy is spawn -> wait_ready -> optional stable
    window -> next spawn, which bounds concurrent CaptureSession/NVMM setup.
    """
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3, 4)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    events: list[str] = []

    def recording_spawn(self):
        events.append(f"spawn:{self.cfg.sid}")
        ps._apply_event_to_proxy(self, ("playing",))

    real_wait_ready = ps._StreamProxy.wait_ready

    def recording_wait_ready(self, timeout_s):
        events.append(f"wait:{self.cfg.sid}")
        return real_wait_ready(self, timeout_s)

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", recording_spawn), \
         patch.object(ps._StreamProxy, "wait_ready", recording_wait_ready):
        session.connect()

    assert events == [
        "spawn:2", "wait:2",
        "spawn:3", "wait:3",
        "spawn:4", "wait:4",
    ]


# ---------------------------------------------------------------------------
# Two-phase connect (PR7): spawn-all-to-PAUSED, then serialize PLAYING
# ---------------------------------------------------------------------------


def _two_phase_connect(session, behaviors):
    """Drive a two-phase connect() with fake workers.

    ``behaviors`` maps sid -> (pause, play, retry), each "ok"/"fail":
      * pause  -> Phase 1 result (reach PAUSED, or fail before it)
      * play   -> Phase 2 result (reach PLAYING after play(), or fail)
      * retry  -> single-phase restart_stream result (straight to PLAYING)
    Returns the ordered list of spawn/play/respawn events so tests can assert
    that every worker is spawned to PAUSED before any is told to play.
    """
    session._ctx_factory = lambda: _FakeCtx()
    order = []

    def _emit(proxy, ok, playing):
        if ok:
            evt = ("playing",) if playing else ("paused",)
        else:
            evt = ("error", f"sid={proxy.cfg.sid} fail", "")
        threading.Timer(0.02, lambda: ps._apply_event_to_proxy(proxy, evt)).start()

    def fake_spawn(self, *, two_phase=False):
        b = behaviors.get(self.cfg.sid, ("ok", "ok", "ok"))
        if two_phase:
            order.append(f"spawn:{self.cfg.sid}")
            _emit(self, b[0] == "ok", playing=False)
        else:  # restart_stream path is single-phase -> straight to PLAYING
            order.append(f"respawn:{self.cfg.sid}")
            _emit(self, b[2] == "ok", playing=True)

    def fake_play(self):
        b = behaviors.get(self.cfg.sid, ("ok", "ok", "ok"))
        order.append(f"play:{self.cfg.sid}")
        _emit(self, b[1] == "ok", playing=True)

    with patch.object(ps._StreamProxy, "spawn", fake_spawn), \
         patch.object(ps._StreamProxy, "play", fake_play):
        session.connect()
    return order


def test_two_phase_connect_spawns_all_to_paused_before_any_play(tmp_path):
    """The whole point of PR7: every worker is spawned (to PAUSED, overlapping
    python/Gst.init) before the first PAUSED->PLAYING is triggered, and the
    PLAYING bring-ups are then issued one at a time."""
    cfgs = [ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3, 4)]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, connect_stable_s=0.0, ready_timeout_s=0.3,
        two_phase_connect=True,
    )
    order = _two_phase_connect(session, {})

    spawn_idx = [i for i, e in enumerate(order) if e.startswith("spawn:")]
    play_idx = [i for i, e in enumerate(order) if e.startswith("play:")]
    assert len(spawn_idx) == 3 and len(play_idx) == 3
    assert max(spawn_idx) < min(play_idx), (
        f"all PAUSED spawns must precede the first play(): {order}"
    )
    assert session.active_sids == [2, 3, 4]
    assert session.poll_errors() == []


def test_two_phase_connect_drops_sid_that_fails_to_pause(tmp_path):
    """A worker that dies before PAUSED is dropped; the retry (single-phase)
    also fails here, so its error surfaces and the others stay active."""
    cfgs = [ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3, 4)]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, connect_stable_s=0.0, ready_timeout_s=0.2,
        two_phase_connect=True,
    )
    order = _two_phase_connect(
        session, {3: ("fail", "ok", "fail")},
    )
    # cam_03 never gets a play() in Phase 2 (it was dropped in Phase 1) but is
    # retried single-phase (respawn), which also fails.
    assert "play:3" not in order
    assert "respawn:3" in order
    assert session.active_sids == [2, 4]
    errs = session.poll_errors()
    assert any(e.sid == 3 for e in errs)


def test_two_phase_connect_play_failure_rescued_by_retry(tmp_path):
    """A worker that reaches PAUSED but fails its first PAUSED->PLAYING is
    rescued by the single-phase restart in the retry round."""
    cfgs = [ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, connect_stable_s=0.0, ready_timeout_s=0.3,
        two_phase_connect=True,
    )
    order = _two_phase_connect(
        session, {3: ("ok", "fail", "ok")},
    )
    assert "play:3" in order and "respawn:3" in order
    assert sorted(session.active_sids) == [2, 3]
    assert session.poll_errors() == []


def test_persistent_session_poll_errors_aggregates_across_proxies(tmp_path):
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(cfgs, tmp_path / "warmup")
    p2 = _make_proxy(tmp_path, sid=2, name="cam_02")
    p3 = _make_proxy(tmp_path, sid=3, name="cam_03")
    session._streams = {2: p2, 3: p3}

    p2.errors.append(ps.StreamError(sid=2, name="cam_02", message="bus EOS"))
    p3.errors.append(ps.StreamError(sid=3, name="cam_03", message="TIMEOUT (6)"))
    p3.errors.append(ps.StreamError(sid=3, name="cam_03", message="UNAVAILABLE (3)"))

    errs = session.poll_errors()
    messages = sorted(e.message for e in errs)
    assert messages == ["TIMEOUT (6)", "UNAVAILABLE (3)", "bus EOS"]
    # poll_errors must drain — a second poll should return nothing.
    assert session.poll_errors() == []


def test_persistent_session_stop_episode_collects_fragments_from_each_proxy(tmp_path):
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}") for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup", finalize_grace_s=0.0,
    )
    p2 = _make_proxy(tmp_path, sid=2, name="cam_02")
    p3 = _make_proxy(tmp_path, sid=3, name="cam_03")
    session._streams = {2: p2, 3: p3}

    ep_dir = tmp_path / "episode_000004"
    ep_dir.mkdir()

    def _ack_start(proxy):
        cmd = proxy.cmd_q.get(timeout=0.5)
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "start", cmd[2], 123, time.monotonic_ns(),
        ))

    threading.Thread(target=_ack_start, args=(p2,), daemon=True).start()
    threading.Thread(target=_ack_start, args=(p3,), daemon=True).start()
    handle = session.start_episode(ep_dir, 4)

    # Simulate the workers reporting episode_done with a fragment after a
    # short delay (mimicking finalize_grace_s).
    def _later(proxy, sid):
        time.sleep(0.05)
        ps._apply_event_to_proxy(proxy, (
            "episode_done",
            {
                "sid": sid, "name": f"cam_{sid:02d}", "fragment_id": 1,
                "path": str(ep_dir / f"cam_{sid:02d}.mkv"),
                "first_pts_s": 0.03, "first_wall_s": time.time(),
                "state": ps.FragmentState.EPISODE.value,
            },
        ))

    threading.Thread(target=_later, args=(p2, 2), daemon=True).start()
    threading.Thread(target=_later, args=(p3, 3), daemon=True).start()

    def _ack_stop(proxy):
        cmd = proxy.cmd_q.get(timeout=0.5)
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "stop", cmd[1], 456, time.monotonic_ns(),
        ))

    threading.Thread(target=_ack_stop, args=(p2,), daemon=True).start()
    threading.Thread(target=_ack_stop, args=(p3,), daemon=True).start()
    session.stop_episode(handle)

    assert set(handle.fragments.keys()) == {"cam_02", "cam_03"}
    assert handle.fragments["cam_02"].fragment_id == 1
    assert handle.fragments["cam_03"].path == ep_dir / "cam_03.mkv"


def test_persistent_session_start_stop_commands_share_frame_grid(tmp_path):
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}", fps=60)
        for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs,
        tmp_path / "warmup",
        target_slice_guard_s=0.05,
    )
    p2 = _make_proxy(tmp_path, sid=2, name="cam_02")
    p3 = _make_proxy(tmp_path, sid=3, name="cam_03")
    session._streams = {2: p2, 3: p3}

    ep_dir = tmp_path / "episode_000005"
    start_cmds = {}

    def _capture_start(proxy, key):
        cmd = proxy.cmd_q.get(timeout=0.5)
        start_cmds[key] = cmd
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "start", cmd[2], 123, time.monotonic_ns(),
        ))

    threading.Thread(target=_capture_start, args=(p2, 2), daemon=True).start()
    threading.Thread(target=_capture_start, args=(p3, 3), daemon=True).start()
    handle = session.start_episode(ep_dir, 5)
    start_cmd_2 = start_cmds[2]
    start_cmd_3 = start_cmds[3]

    assert start_cmd_2[0] == "start_episode_at"
    assert start_cmd_3[0] == "start_episode_at"
    assert start_cmd_2[2:] == start_cmd_3[2:]
    assert start_cmd_2[3] == handle.start_frame_index
    assert start_cmd_2[4] == handle.frame_period_ns

    stop_cmds = {}

    def _capture_stop(proxy, key):
        cmd = proxy.cmd_q.get(timeout=0.5)
        stop_cmds[key] = cmd
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "stop", cmd[1], 456, time.monotonic_ns(),
        ))

    threading.Thread(target=_capture_stop, args=(p2, 2), daemon=True).start()
    threading.Thread(target=_capture_stop, args=(p3, 3), daemon=True).start()
    session.schedule_stop_episode(handle)
    stop_cmd_2 = stop_cmds[2]
    stop_cmd_3 = stop_cmds[3]

    assert stop_cmd_2[0] == "stop_episode_at"
    assert stop_cmd_3[0] == "stop_episode_at"
    assert stop_cmd_2[1:] == stop_cmd_3[1:]
    assert stop_cmd_2[2] == handle.expected_video_frames
    assert stop_cmd_2[3] == handle.stop_frame_index
    assert stop_cmd_2[4] == handle.frame_period_ns
    assert handle.expected_video_frames == (
        handle.stop_frame_index - handle.start_frame_index
    )
    assert handle.stop_mono_ns == session._target_for_frame_index(handle.stop_frame_index)


def test_persistent_session_missing_start_split_ack_marks_episode_invalid(tmp_path):
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}", fps=60)
        for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs,
        tmp_path / "warmup",
        target_slice_guard_s=0.01,
    )
    p2 = _make_proxy(tmp_path, sid=2, name="cam_02")
    p3 = _make_proxy(tmp_path, sid=3, name="cam_03")
    session._streams = {2: p2, 3: p3}

    def _ack_start(proxy):
        cmd = proxy.cmd_q.get(timeout=0.5)
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "start", cmd[2], 123, time.monotonic_ns(),
        ))

    threading.Thread(target=_ack_start, args=(p2,), daemon=True).start()
    handle = session.start_episode(tmp_path / "episode_000006", 6)

    assert not handle.split_schedule_ok
    assert any("start:" in failure for failure in handle.split_schedule_failures)
    assert any("cam_03" in failure for failure in handle.split_schedule_failures)


def test_persistent_session_missing_stop_split_ack_marks_episode_invalid(tmp_path):
    cfgs = [
        ps.StreamConfig(sid=sid, name=f"cam_{sid:02d}", fps=60)
        for sid in (2, 3)
    ]
    session = ps.PersistentCameraSession(
        cfgs,
        tmp_path / "warmup",
        target_slice_guard_s=0.01,
    )
    p2 = _make_proxy(tmp_path, sid=2, name="cam_02")
    p3 = _make_proxy(tmp_path, sid=3, name="cam_03")
    session._streams = {2: p2, 3: p3}

    def _ack_start(proxy):
        cmd = proxy.cmd_q.get(timeout=0.5)
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "start", cmd[2], 123, time.monotonic_ns(),
        ))

    threading.Thread(target=_ack_start, args=(p2,), daemon=True).start()
    threading.Thread(target=_ack_start, args=(p3,), daemon=True).start()
    handle = session.start_episode(tmp_path / "episode_000007", 7)
    assert handle.split_schedule_ok

    def _ack_stop(proxy):
        cmd = proxy.cmd_q.get(timeout=0.5)
        ps._apply_event_to_proxy(proxy, (
            "split_scheduled", "stop", cmd[1], 456, time.monotonic_ns(),
        ))

    threading.Thread(target=_ack_stop, args=(p2,), daemon=True).start()
    session.schedule_stop_episode(handle)

    assert not handle.split_schedule_ok
    assert any("stop:" in failure for failure in handle.split_schedule_failures)
    assert any("cam_03" in failure for failure in handle.split_schedule_failures)


def test_persistent_session_start_episode_before_connect_raises(tmp_path):
    session = ps.PersistentCameraSession([], tmp_path / "warmup")
    try:
        session.start_episode(tmp_path / "ep", 0)
    except RuntimeError as exc:
        assert "connect()" in str(exc)
    else:
        raise AssertionError("expected start_episode before connect to raise")


# ---------------------------------------------------------------------------
# Worker-side isolation — independent processes have independent failures
# ---------------------------------------------------------------------------


def test_one_failing_proxy_does_not_block_others_from_becoming_ready(tmp_path):
    """The isolation contract.

    The pre-rewrite single-process model collapsed under PR2's load: a
    stuck Argus session on one camera deadlocked the Python thread for all
    cameras. With one subprocess per camera, an error on sid=3 has no path
    to delay sid=2 reaching PLAYING.

    We capture each proxy's ready_evt state right after its wait_ready
    completes (via the patched spawn closure) to prove cam_02 became ready
    even though cam_03 will eventually fail. If isolation were broken, the
    test would time out instead of completing.
    """
    cfgs = [
        ps.StreamConfig(sid=2, name="cam_02"),
        ps.StreamConfig(sid=3, name="cam_03"),
    ]
    session = ps.PersistentCameraSession(
        cfgs, tmp_path / "warmup",
        spawn_stagger_s=0.0, ready_timeout_s=0.3,
    )

    observed_ready: dict[int, bool] = {}

    def fake_spawn(self):
        if self.cfg.sid == 2:
            threading.Timer(
                0.02,
                lambda: ps._apply_event_to_proxy(self, ("playing",)),
            ).start()
        elif self.cfg.sid == 3:
            threading.Timer(
                0.05,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "sid=3 broke", ""),
                ),
            ).start()

    # Wrap wait_ready so we observe what each proxy saw before connect()
    # tore everything down on failure.
    real_wait_ready = ps._StreamProxy.wait_ready

    def observing_wait_ready(self, timeout_s):
        result = real_wait_ready(self, timeout_s)
        observed_ready[self.cfg.sid] = result
        return result

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", fake_spawn), \
         patch.object(ps._StreamProxy, "wait_ready", observing_wait_ready):
        # Partial failures no longer raise — cam_02 survives, cam_03 is dropped.
        session.connect()

    assert observed_ready[2] is True, "cam_02 must reach PLAYING despite cam_03's failure"
    assert observed_ready[3] is False
    assert session.active_sids == [2], "successful proxy must stay live after partial failure"
    errs = session.poll_errors()
    assert any("sid=3 broke" in e.message for e in errs)


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def test_discard_episode_unlinks_all_handle_fragments(tmp_path):
    """discard_episode no longer needs to touch any worker — stop_episode
    already closed each EPISODE fragment, so discard is pure file deletion."""
    cfgs = [ps.StreamConfig(sid=2, name="cam_02")]
    session = ps.PersistentCameraSession(cfgs, tmp_path / "warmup")

    ep_dir = tmp_path / "episode_000009"
    ep_dir.mkdir()
    file_a = ep_dir / "cam_02.mkv"
    file_a.write_bytes(b"fake")
    handle = ps.EpisodeHandle(
        idx=9, directory=ep_dir,
        t0_wall_s=time.time(), t0_mono_s=time.monotonic(),
        fragments={
            "cam_02": ps.FragmentInfo(
                sid=2, name="cam_02", fragment_id=1, path=file_a,
                first_pts_s=0.0, first_wall_s=0.0,
                state=ps.FragmentState.EPISODE,
            ),
        },
    )
    session.discard_episode(handle)
    assert not file_a.exists()
