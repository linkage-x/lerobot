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
    assert _FakeProcess.last_kwargs["kwargs"] == {"ready_timeout_s": 0.42}
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


def test_apply_event_eos_appends_and_unblocks_ready(tmp_path):
    proxy = _make_proxy(tmp_path)
    ps._apply_event_to_proxy(proxy, ("eos",))
    assert proxy.ready_evt.is_set()
    assert len(proxy.errors) == 1
    assert "EOS" in proxy.errors[0].message


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


def test_connect_retries_run_in_parallel_not_serially(tmp_path):
    """Phase 3 retry must overlap restart_stream calls across sids, not
    chain them. We measure wall-clock cost vs. theoretical serial cost.

    Each retry's spawn posts the rescue event after a fixed delay; if
    retries serialize, total time = N * delay; if parallel, total ≈ delay.
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
    rescue_delay_s = 0.2

    def flaky_spawn(self):
        sid = self.cfg.sid
        spawn_count[sid] = spawn_count.get(sid, 0) + 1
        if spawn_count[sid] == 1:
            # First attempt: fail immediately so Phase 1+2 returns fast.
            threading.Timer(
                0.01,
                lambda: ps._apply_event_to_proxy(
                    self, ("error", "first-try", ""),
                ),
            ).start()
        else:
            # Retry attempt: succeed only after rescue_delay_s. If retries
            # run serially, total Phase 3 time >= N * rescue_delay_s.
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
    # Generous bound: serial would be N * rescue_delay = 1.0s. Parallel
    # should be well under 2x rescue_delay even on a loaded CI runner.
    serial_lower_bound = len(fail_sids) * rescue_delay_s
    assert elapsed < serial_lower_bound * 0.6, (
        f"Phase 3 retry appears serial: elapsed={elapsed:.3f}s, "
        f"serial would have taken ~{serial_lower_bound:.3f}s"
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


def test_rolling_spawn_calls_spawn_for_all_streams_before_first_wait_ready(tmp_path):
    """The wall-clock win comes from spawning all workers (Phase 1) before
    waiting for any of them (Phase 2). Verify that order."""
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
        # Post "playing" immediately so wait_ready returns fast for all.
        ps._apply_event_to_proxy(self, ("playing",))

    real_wait_ready = ps._StreamProxy.wait_ready

    def recording_wait_ready(self, timeout_s):
        events.append(f"wait:{self.cfg.sid}")
        return real_wait_ready(self, timeout_s)

    session._ctx_factory = lambda: _FakeCtx()
    with patch.object(ps._StreamProxy, "spawn", recording_spawn), \
         patch.object(ps._StreamProxy, "wait_ready", recording_wait_ready):
        session.connect()

    # All spawn:* events must come before any wait:* event.
    spawn_indexes = [i for i, e in enumerate(events) if e.startswith("spawn:")]
    wait_indexes = [i for i, e in enumerate(events) if e.startswith("wait:")]
    assert spawn_indexes and wait_indexes
    assert max(spawn_indexes) < min(wait_indexes), (
        f"rolling spawn must finish Phase 1 before Phase 2; got order: {events}"
    )


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
    handle = session.start_episode(ep_dir, 4)
    # Drain the start-episode commands from each proxy's cmd_q so the test
    # doesn't have to assert on them here.
    p2.cmd_q.get(timeout=0.5)
    p3.cmd_q.get(timeout=0.5)

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

    session.stop_episode(handle)
    # stop_episode also pushes its own command — drain so the test is clean.
    p2.cmd_q.get(timeout=0.5)
    p3.cmd_q.get(timeout=0.5)

    assert set(handle.fragments.keys()) == {"cam_02", "cam_03"}
    assert handle.fragments["cam_02"].fragment_id == 1
    assert handle.fragments["cam_03"].path == ep_dir / "cam_03.mkv"


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
