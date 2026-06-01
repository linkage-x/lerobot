"""Persistent GStreamer pipeline session for GMSL2 multi-camera capture.

This is PR1 of the "Option A" plan in
``tools/data_collection_gui/docs/option_a_persistent_pipeline_design.md``:
a standalone module + interactive demo that proves the splitmuxsink slicing
path works **without** touching ``thor_record.py``. The recorder integration
lands in PR2 once this module is burn-in tested.

What it does
------------

Each camera gets a long-lived ``Gst.Pipeline``::

    nvarguscamerasrc sensor-id=N do-timestamp=true ...
      ! video/x-raw(memory:NVMM),format=NV12,width=W,height=H,framerate=F/1
      ! nvv4l2h{265,264}enc bitrate=... iframeinterval=... ...
      ! h{265,264}parse
      ! splitmuxsink name=mux_N
          muxer-factory=matroskamux
          async-finalize=true
          max-size-time=0 max-size-bytes=0
          location=<warmup-dir>/cam_NN_warmup_%05d.mkv

The pipeline is created with the Python ``gi.repository.Gst`` binding so we
can ``emit("split-now")`` on the splitmuxsink at episode boundaries — that's
the part ``gst-launch-1.0`` cannot do from CLI.

State machine per stream::

    WARMUP   -- format-location returns /tmp/cam_NN_warmup_*.mkv
       │
       └── start_episode() -> emit("force-IDR"), emit("split-now")
              │
              ▼
    EPISODE  -- format-location returns <episode-dir>/cam_NN.mkv
       │       first-sample PTS recorded into the EpisodeHandle for L3b
       │       alignment downstream
       │
       └── stop_episode() -> emit("split-now")
              │
              ▼
    WARMUP   -- back to /tmp until next start_episode()

Discard
-------

Discard is implemented as "split now + delete the just-written fragment":

    discard_episode(handle):
        emit("split-now") on all muxes  # closes the EPISODE fragment
        wait_for_finalize()              # async-finalize must drain to disk
        unlink <episode-dir>/cam_*.mkv

Standalone demo
---------------

Run this module directly to spawn N pipelines and drive them from stdin::

    # Real hardware (Thor):
    python3 -m tools.thor.gmsl2.persistent_session \\
        --sids 0,2,7 \\
        --episode-root /tmp/pcs_episodes \\
        --warmup-dir /tmp/pcs_warmup \\
        --episode-time-s 5

    # Dev host (no nvarguscamerasrc): videotestsrc + x264enc fallback
    python3 -m tools.thor.gmsl2.persistent_session \\
        --sids 0,1 --use-test-source \\
        --episode-root /tmp/pcs_episodes \\
        --warmup-dir /tmp/pcs_warmup

Interactive commands on stdin:
    <Enter>     start a timed episode
    s           save (no-op; episodes auto-finish when the timer expires)
    d           discard the most recent episode
    q           quit (drains pipelines cleanly)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue as queue_module
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("persistent_session")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FragmentState(Enum):
    WARMUP = "warmup"
    EPISODE = "episode"


@dataclass
class StreamConfig:
    """Per-stream parameters resolved from the YAML or CLI."""

    sid: int
    name: str
    width: int = 1920
    height: int = 1080
    fps: int = 60
    codec: str = "h265"           # h264 | h265
    bitrate_kbps: int = 20000
    iframe_interval: int = 30
    preset_level: int = 1
    control_rate: int = 1
    sensor_mode: int = 0
    exposure_us: int = 0
    # Driver-unit V4L2 gain. This is intentionally not forwarded to
    # nvarguscamerasrc; Argus gainrange uses a different 0..4 float scale.
    gain: int = 0
    argus_gain: float = 0.0
    use_test_source: bool = False  # dev-host fallback (videotestsrc + x264enc)


@dataclass
class FragmentInfo:
    """One file segment produced by splitmuxsink."""

    sid: int
    name: str
    fragment_id: int
    path: Path
    first_pts_s: float | None  # filled from format-location-full first-sample
    first_wall_s: float        # host wall time when the new file opened
    state: FragmentState


@dataclass
class EpisodeHandle:
    """Identifies a single episode across all streams."""

    idx: int
    directory: Path
    t0_wall_s: float           # host wall time just before split-now emit
    t0_mono_s: float
    # name -> fragment opened in EPISODE state for that stream
    fragments: dict[str, FragmentInfo] = field(default_factory=dict)
    stop_wall_s: float = 0.0


@dataclass
class StreamError:
    sid: int
    name: str
    message: str
    debug: str = ""
    wall_s: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline_desc(stream: StreamConfig, warmup_location: str) -> str:
    """Render the parse_launch description for one persistent pipeline."""

    parser = "h265parse" if stream.codec == "h265" else "h264parse"
    muxer_factory = "matroskamux"

    if stream.use_test_source:
        # Dev fallback: software videotestsrc + x264enc.
        # `is-live=true` matches the camera framerate semantics, and we
        # use `gop` rather than the nvenc iframeinterval property.
        encoder = (
            f"x264enc tune=zerolatency speed-preset=ultrafast "
            f"bitrate={stream.bitrate_kbps} key-int-max={stream.iframe_interval}"
        )
        if stream.codec == "h265":
            encoder = (
                f"x265enc tune=zerolatency speed-preset=ultrafast "
                f"bitrate={stream.bitrate_kbps} key-int-max={stream.iframe_interval}"
            )
        source = (
            f"videotestsrc is-live=true pattern=ball "
            f"! video/x-raw,format=I420,width={stream.width},height={stream.height},"
            f"framerate={stream.fps}/1 "
            f"! timeoverlay halignment=left valignment=top text=\"cam_{stream.sid:02d}\" "
        )
    else:
        # Real hardware: nvarguscamerasrc + nvv4l2h{265,264}enc.
        enc_factory = "nvv4l2h265enc" if stream.codec == "h265" else "nvv4l2h264enc"
        exposure_clause = ""
        if stream.exposure_us > 0:
            exposure_clause = (
                f"exposuretimerange=\"{stream.exposure_us * 1000} "
                f"{stream.exposure_us * 1000}\" "
            )
        gain_clause = ""
        if stream.argus_gain > 0:
            if stream.argus_gain > 4.0:
                raise ValueError(
                    f"argus_gain must be <= 4.0 for nvarguscamerasrc, got {stream.argus_gain}"
                )
            gain_clause = f"gainrange=\"{stream.argus_gain:g} {stream.argus_gain:g}\" "

        source = (
            f"nvarguscamerasrc sensor-id={stream.sid} "
            f"sensor-mode={stream.sensor_mode} do-timestamp=true "
            f"{exposure_clause}{gain_clause}"
            f"! video/x-raw(memory:NVMM),format=NV12,width={stream.width},"
            f"height={stream.height},framerate={stream.fps}/1 "
        )
        # nvv4l2h{264,265}enc distinguishes I-frame interval (iframeinterval)
        # from IDR-frame interval (idrinterval). splitmuxsink can only cut on
        # an IDR boundary, so we pin both to the same period — otherwise the
        # default idrinterval (~256 frames on JetPack 6) makes split-now
        # spread the actual cut across several seconds across cameras.
        encoder = (
            f"{enc_factory} bitrate={stream.bitrate_kbps * 1000} "
            f"iframeinterval={stream.iframe_interval} "
            f"idrinterval={stream.iframe_interval} "
            f"preset-level={stream.preset_level} "
            f"control-rate={stream.control_rate} insert-sps-pps=1"
        )

    return (
        f"{source} "
        f"! {encoder} "
        f"! {parser} "
        f"! splitmuxsink name=mux_{stream.sid} "
        f"  muxer-factory={muxer_factory} "
        f"  async-finalize=true "
        f"  max-size-time=0 max-size-bytes=0 "
        f"  location=\"{warmup_location}\""
    )


# ---------------------------------------------------------------------------
# Per-stream worker proxy
# ---------------------------------------------------------------------------
#
# Each pipeline lives in its own subprocess (see persistent_session_worker.py).
# The parent process holds one _StreamProxy per camera which owns:
#   - the child mp.Process
#   - the two mp.Queue's the protocol uses
#   - a reader thread that drains evt_q into local state
#
# The proxy turns child-side events into parent-side state:
#   ready_evt      set when the child reports "playing" OR any terminal error
#                  (so connect() never blocks waiting for a dead worker)
#   episode_done_evt set after a stop_episode finalize round trips
#   last_episode_fragment   most recent EPISODE fragment opened
#   errors         queued StreamError's (drained by poll_errors())
#
# This file intentionally does NOT import gi.repository. All Gst-dependent
# code lives in the worker module, which is only imported inside the child.


def _apply_event_to_proxy(proxy: "_StreamProxy", event: tuple) -> None:
    """Dispatch one worker event into a proxy's state.

    Pulled out as a module-level free function so the unit tests can drive
    arbitrary event sequences without spinning up a real subprocess. The
    reader thread is the only place that calls this in production.
    """
    if not isinstance(event, tuple) or not event:
        return
    kind = event[0]
    if kind == "playing":
        proxy.ready_evt.set()
        return
    if kind == "fragment":
        info_dict = event[1]
        info = FragmentInfo(
            sid=int(info_dict["sid"]),
            name=str(info_dict["name"]),
            fragment_id=int(info_dict["fragment_id"]),
            path=Path(info_dict["path"]),
            first_pts_s=info_dict["first_pts_s"],
            first_wall_s=float(info_dict["first_wall_s"]),
            state=FragmentState(info_dict["state"]),
        )
        proxy.fragment_history.append(info)
        if info.state == FragmentState.EPISODE:
            proxy.last_episode_fragment = info
        if proxy._on_fragment_opened_cb is not None:
            try:
                proxy._on_fragment_opened_cb(proxy, info)
            except Exception as exc:
                logger.warning("on_fragment_opened callback failed: %s", exc)
        return
    if kind == "error":
        message = str(event[1]) if len(event) > 1 else "unspecified error"
        debug = str(event[2]) if len(event) > 2 else ""
        proxy.errors.append(StreamError(
            sid=proxy.cfg.sid, name=proxy.cfg.name,
            message=message, debug=debug,
        ))
        proxy.ready_evt.set()  # unblock connect() — terminal failure
        return
    if kind == "eos":
        proxy.errors.append(StreamError(
            sid=proxy.cfg.sid, name=proxy.cfg.name,
            message="bus EOS (upstream stopped delivering buffers)",
        ))
        proxy.ready_evt.set()
        return
    if kind == "episode_done":
        info_dict = event[1] if len(event) > 1 else None
        if info_dict is not None:
            proxy.last_episode_fragment = FragmentInfo(
                sid=int(info_dict["sid"]),
                name=str(info_dict["name"]),
                fragment_id=int(info_dict["fragment_id"]),
                path=Path(info_dict["path"]),
                first_pts_s=info_dict["first_pts_s"],
                first_wall_s=float(info_dict["first_wall_s"]),
                state=FragmentState(info_dict["state"]),
            )
        proxy.episode_done_evt.set()
        return
    if kind == "disconnected":
        proxy.disconnected_evt.set()
        return
    logger.warning("[%s] unknown worker event %r", proxy.cfg.name, event)


class _StreamProxy:
    """Parent-side handle for one worker subprocess."""

    def __init__(
        self,
        cfg: StreamConfig,
        warmup_dir: Path,
        ctx: Any,
        *,
        on_fragment_opened: Callable[["_StreamProxy", FragmentInfo], None] | None = None,
    ):
        self.cfg = cfg
        self.warmup_dir = warmup_dir
        self._ctx = ctx
        self.cmd_q: Any = ctx.Queue()
        self.evt_q: Any = ctx.Queue()
        self.proc: Any = None
        self.reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self.ready_evt = threading.Event()
        self.episode_done_evt = threading.Event()
        self.disconnected_evt = threading.Event()
        self.last_episode_fragment: FragmentInfo | None = None
        self.fragment_history: list[FragmentInfo] = []
        self.errors: list[StreamError] = []
        self._on_fragment_opened_cb = on_fragment_opened

    # -- lifecycle --------------------------------------------------------

    def spawn(self) -> None:
        # Imported lazily so the parent can still be imported on machines
        # without gi.repository (e.g. dev hosts where only build_pipeline_desc
        # is unit-tested).
        from tools.thor.gmsl2 import persistent_session_worker as worker

        self.proc = self._ctx.Process(
            target=worker.run_worker,
            args=(self.cfg, self.warmup_dir, self.cmd_q, self.evt_q),
            name=f"ps-worker-{self.cfg.sid:02d}",
            daemon=True,
        )
        self.proc.start()
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"ps-reader-{self.cfg.sid:02d}",
            daemon=True,
        )
        self.reader_thread.start()

    def wait_ready(self, timeout_s: float) -> bool:
        """Block until the worker reports PLAYING or a terminal failure.

        Returns True iff the worker reached PLAYING with no error queued.
        """
        if not self.ready_evt.wait(timeout_s):
            # No event from the child within the deadline — record a
            # synthetic error so poll_errors() surfaces it to the operator.
            self.errors.append(StreamError(
                sid=self.cfg.sid, name=self.cfg.name,
                message=f"worker did not become ready within {timeout_s:.1f}s",
            ))
            return False
        return not self.errors

    def start_episode(self, episode_dir: Path) -> None:
        self.episode_done_evt.clear()
        self.last_episode_fragment = None
        self.cmd_q.put(("start_episode", str(episode_dir)))

    def stop_episode(self) -> None:
        self.cmd_q.put(("stop_episode",))

    def wait_episode_done(self, timeout_s: float) -> bool:
        return self.episode_done_evt.wait(timeout_s)

    def disconnect(self, *, grace_s: float = 3.0) -> None:
        try:
            self.cmd_q.put(("disconnect",))
        except Exception:
            pass
        if self.proc is None:
            return
        # First give the child a chance to drain async-finalize cleanly.
        self.proc.join(timeout=grace_s)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=1.0)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=1.0)
        self._reader_stop.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)

    # -- error/event drain ----------------------------------------------

    def drain_errors(self) -> list[StreamError]:
        out, self.errors = self.errors, []
        return out

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            try:
                evt = self.evt_q.get(timeout=0.2)
            except queue_module.Empty:
                continue
            except (EOFError, OSError):
                return
            _apply_event_to_proxy(self, evt)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class PersistentCameraSession:
    """N persistent GStreamer pipelines with on-demand episode slicing.

    Each pipeline runs in its own subprocess (see persistent_session_worker)
    so nvargus-daemon sees N independent RPC clients instead of one client
    holding N CaptureSession's — that was the post-PR2 regression that
    deadlocked set_state(PLAYING) and produced the empty-episode-dir bug.
    """

    def __init__(
        self,
        streams: list[StreamConfig],
        warmup_dir: Path,
        *,
        spawn_stagger_s: float = 1.0,
        finalize_grace_s: float = 0.5,
        ready_timeout_s: float = 12.0,
        on_fragment_opened: Callable[["_StreamProxy", FragmentInfo], None] | None = None,
    ):
        self._stream_cfgs = list(streams)
        self.warmup_dir = warmup_dir
        self.spawn_stagger_s = float(spawn_stagger_s)
        self.finalize_grace_s = float(finalize_grace_s)
        self.ready_timeout_s = float(ready_timeout_s)
        self._on_fragment_opened_cb = on_fragment_opened
        self._streams: dict[int, _StreamProxy] = {}
        self._errors: list[StreamError] = []
        self._lock = threading.Lock()
        # Subprocess context is chosen at connect() time; tests can also
        # inject a fake by patching self._ctx_factory before calling connect.
        self._ctx_factory: Callable[[], Any] = lambda: mp.get_context("spawn")
        self.connect_duration_s: float = 0.0

    # -- lifecycle --------------------------------------------------------

    @property
    def active_sids(self) -> list[int]:
        """sids whose worker reached PLAYING and is still up.

        Updated as ``connect()`` drops failed workers and as
        ``restart_stream()`` swaps them in/out.
        """
        return sorted(self._streams)

    def connect(self) -> None:
        if self._streams:
            raise RuntimeError("connect() called twice without disconnect()")
        self.warmup_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._ctx_factory()
        for cfg in self._stream_cfgs:
            self._streams[cfg.sid] = _StreamProxy(
                cfg, self.warmup_dir, ctx,
                on_fragment_opened=self._on_fragment_opened_cb,
            )

        t0 = time.time()
        sids = sorted(self._streams)

        # Phase 1 — rolling spawn.
        #   Each worker is its own subprocess (separate RPC socket to
        #   nvargus-daemon), so the daemon sees independent clients and
        #   isolation holds. We still pace SPAWNS by spawn_stagger_s to
        #   bound the daemon's per-tick OpenCaptureSession load, but we do
        #   NOT wait for PLAYING here. Workers run their Argus open in
        #   parallel from this point on — that's where the wall-clock win
        #   comes from vs. the strictly-serial version (which made
        #   connect() take stagger * N * (Argus open time) seconds).
        for i, sid in enumerate(sids):
            if i > 0 and self.spawn_stagger_s > 0:
                time.sleep(self.spawn_stagger_s)
            proxy = self._streams[sid]
            proxy.spawn()
            logger.info(
                "[%s] spawned (+%.2fs)", proxy.cfg.name, time.time() - t0,
            )

        # Phase 2 — serial wait_ready.
        #   By the time Phase 1 finishes, the earliest-spawned workers have
        #   typically already reached PLAYING (their wait_ready returns
        #   immediately because ready_evt is already set). Only the last
        #   few stragglers actually block here.
        for sid in sids:
            proxy = self._streams[sid]
            ok = proxy.wait_ready(self.ready_timeout_s)
            elapsed = time.time() - t0
            if ok:
                logger.info("[%s] PLAYING (+%.2fs)", proxy.cfg.name, elapsed)
            else:
                logger.warning(
                    "[%s] failed to reach PLAYING (+%.2fs)", proxy.cfg.name, elapsed,
                )

        # Phase 3 — parallel retry round.
        #   recover_argus.sh proved daemon-side per-sid retry works: when
        #   a single sid's first OpenCaptureSession is mid-air during a
        #   daemon state-machine glitch, restarting that sid alone usually
        #   succeeds. PR3's restart_stream(sid) is the equivalent at the
        #   worker level (terminate the worker -> daemon drops its
        #   sessions -> respawn -> wait_ready).
        #
        #   We retry in parallel: each worker has its own subprocess and
        #   its own RPC socket to nvargus-daemon, so concurrent
        #   terminate+respawn is exactly N independent "client disconnect
        #   + new client connect" pairs — daemon-side serialization holds
        #   per socket, not across them, which is the multi-process
        #   isolation argument PR3 was designed around. Wall-clock cost
        #   collapses from sum(per-sid) to max(per-sid) (~5s instead of
        #   ~25-30s on the 7-fail external test).
        #
        #   At most once per sid per connect() so a permanently-dead
        #   camera doesn't loop.
        first_round_errors = self.poll_errors()
        if first_round_errors:
            retry_sids = sorted({e.sid for e in first_round_errors})
            logger.warning(
                "connect first pass: %d/%d failed; retrying %s in parallel",
                len(retry_sids), len(self._streams), retry_sids,
            )
            live_retry_sids = [sid for sid in retry_sids if sid in self._streams]
            if live_retry_sids:
                with ThreadPoolExecutor(
                    max_workers=len(live_retry_sids),
                    thread_name_prefix="pcs-retry",
                ) as ex:
                    list(ex.map(self.restart_stream, live_retry_sids))
            # Only what survived the retry counts toward partial failure.
            errors = self.poll_errors()
        else:
            errors = []

        self.connect_duration_s = time.time() - t0

        # Partial-failure policy: drop failed workers, keep the rest live.
        # connect() only raises when EVERY worker failed; otherwise it
        # returns success and the caller is expected to poll_errors() to
        # see who fell off and emit a warning to the operator.
        if errors:
            failed_sids = {e.sid for e in errors}
            for sid in list(self._streams):
                if sid in failed_sids:
                    try:
                        self._streams[sid].disconnect(grace_s=1.0)
                    except Exception:
                        pass
                    del self._streams[sid]
            if not self._streams:
                bad = ", ".join(
                    f"{e.name}({(e.message or '')[:60]})" for e in errors
                )
                raise RuntimeError(
                    f"connect() failed on all {len(errors)} stream(s): {bad}"
                )
            # Partial: leave the errors in the bag so poll_errors() can
            # surface them to the caller.
            with self._lock:
                self._errors.extend(errors)
            bad = ", ".join(
                f"{e.name}({(e.message or '')[:60]})" for e in errors
            )
            logger.warning(
                "connect() partial success: %d/%d streams up; failed: %s",
                len(self._streams), len(self._stream_cfgs), bad,
            )
            return

        logger.info(
            "connected %d streams in %.2fs (stagger=%.2fs)",
            len(self._streams), self.connect_duration_s, self.spawn_stagger_s,
        )

    def disconnect(self) -> None:
        for proxy in self._streams.values():
            try:
                proxy.disconnect()
            except Exception as exc:
                logger.warning("[%s] disconnect failed: %s", proxy.cfg.name, exc)
        self._streams.clear()

    # -- episode control -------------------------------------------------

    def start_episode(self, episode_dir: Path, idx: int) -> EpisodeHandle:
        if not self._streams:
            raise RuntimeError("start_episode() before connect()")
        episode_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            t0_wall = time.time()
            t0_mono = time.monotonic()
            for proxy in self._streams.values():
                proxy.start_episode(episode_dir)
            handle = EpisodeHandle(
                idx=idx, directory=episode_dir,
                t0_wall_s=t0_wall, t0_mono_s=t0_mono,
            )
        return handle

    def stop_episode(self, handle: EpisodeHandle) -> EpisodeHandle:
        with self._lock:
            handle.stop_wall_s = time.time()
            for proxy in self._streams.values():
                proxy.stop_episode()
        # Each worker runs its own finalize_grace_s sleep before emitting
        # episode_done; we just have to wait for those events to land.
        per_worker_timeout = self.finalize_grace_s + 1.5
        for proxy in self._streams.values():
            if not proxy.wait_episode_done(per_worker_timeout):
                logger.warning(
                    "[%s] episode_done not received within %.1fs",
                    proxy.cfg.name, per_worker_timeout,
                )
        for proxy in self._streams.values():
            if proxy.last_episode_fragment is not None:
                handle.fragments[proxy.cfg.name] = proxy.last_episode_fragment
                proxy.last_episode_fragment = None
        return handle

    def discard_episode(self, handle: EpisodeHandle) -> None:
        # stop_episode already had each worker close its EPISODE fragment;
        # all we need to do here is delete the .mkv files on disk.
        for info in handle.fragments.values():
            try:
                info.path.unlink(missing_ok=True)
                logger.info("discarded fragment %s", info.path)
            except OSError as exc:
                logger.warning("failed to unlink %s: %s", info.path, exc)

    # -- maintenance -----------------------------------------------------

    def cleanup_warmup_files(self, *, keep_last_n: int = 3) -> int:
        """Delete warmup-state fragments, keeping the most recent N per camera.

        splitmuxsink doesn't auto-rotate fragments in async-finalize mode,
        so this prevents the warmup directory from growing unbounded across
        long sessions. Call once per episode.

        Returns the number of files deleted.
        """
        if not self.warmup_dir.is_dir():
            return 0
        per_sid: dict[int, list[Path]] = {}
        for path in self.warmup_dir.glob("cam_*_warmup_*.mkv"):
            try:
                # cam_00_warmup_00003.mkv -> sid=0
                sid = int(path.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            per_sid.setdefault(sid, []).append(path)
        deleted = 0
        for paths in per_sid.values():
            paths.sort(key=lambda p: p.name)
            for old in paths[:-keep_last_n] if keep_last_n > 0 else paths:
                try:
                    old.unlink()
                    deleted += 1
                except OSError:
                    pass
        return deleted

    def restart_stream(self, sid: int) -> bool:
        """Kill and respawn the worker for one stream.

        Useful when ``poll_errors()`` flags that stream as failed: terminates
        the child process (which drops its Argus session daemon-side),
        spawns a fresh worker, and waits for it to reach PLAYING. Returns
        True on success.
        """
        if sid not in self._streams:
            return False
        old = self._streams[sid]
        try:
            old.disconnect(grace_s=1.0)
        except Exception as exc:
            logger.warning("restart_stream sid=%s disconnect raised: %s", sid, exc)
        ctx = self._ctx_factory()
        new = _StreamProxy(
            old.cfg, self.warmup_dir, ctx,
            on_fragment_opened=self._on_fragment_opened_cb,
        )
        self._streams[sid] = new
        try:
            new.spawn()
            ok = new.wait_ready(self.ready_timeout_s)
            logger.info("restart_stream sid=%s -> %s", sid, "ok" if ok else "FAIL")
            return ok
        except Exception as exc:
            logger.warning("restart_stream sid=%s spawn raised: %s", sid, exc)
            return False

    # -- diagnostics -----------------------------------------------------

    def poll_errors(self) -> list[StreamError]:
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
            for proxy in self._streams.values():
                errors.extend(proxy.drain_errors())
        return errors

    # -- meta helpers ----------------------------------------------------

    def write_episode_meta(self, handle: EpisodeHandle) -> Path:
        meta = {
            "episode_index": handle.idx,
            "t0_wall_s": handle.t0_wall_s,
            "t0_mono_s": handle.t0_mono_s,
            "stop_wall_s": handle.stop_wall_s,
            "duration_s": handle.stop_wall_s - handle.t0_wall_s,
            "sync_reference": {
                "split_now_wall_s": handle.t0_wall_s,
                "camera_first_pts_s": {
                    name: info.first_pts_s for name, info in handle.fragments.items()
                },
                "camera_first_wall_s": {
                    name: info.first_wall_s for name, info in handle.fragments.items()
                },
                "note": (
                    "split_now_wall_s is host time when split-now was emitted. "
                    "camera_first_pts_s is the PTS of the first frame in the new "
                    "fragment (extracted from splitmuxsink format-location-full "
                    "first_sample). Use these to align cameras to BOX wall-clock "
                    "samples without re-running ffprobe."
                ),
            },
            "cameras": [
                {
                    "name": name,
                    "sid": info.sid,
                    "fragment_id": info.fragment_id,
                    "file": info.path.name,
                    "first_pts_s": info.first_pts_s,
                    "first_wall_s": info.first_wall_s,
                }
                for name, info in handle.fragments.items()
            ],
        }
        meta_path = handle.directory / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta_path


# ---------------------------------------------------------------------------
# Demo CLI
# ---------------------------------------------------------------------------


def _read_stdin_loop(queue: list[str], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, ValueError):
            stop.set()
            return
        if not line:
            stop.set()
            return
        queue.append(line.strip().lower())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="PR1 demo: persistent GStreamer pipelines with splitmuxsink slicing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sids", required=True,
                    help="comma-separated camera sensor ids, e.g. '0,2,7'")
    ap.add_argument("--episode-root", type=Path, required=True,
                    help="root directory where episode_NNNNNN/cam_XX.mkv go")
    ap.add_argument("--warmup-dir", type=Path, required=True,
                    help="directory for warmup-state fragments (gets cleaned each run)")
    ap.add_argument("--episode-time-s", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--codec", choices=["h264", "h265"], default="h265")
    ap.add_argument("--bitrate-kbps", type=int, default=20000)
    ap.add_argument("--iframe-interval", type=int, default=30,
                    help="encoder GOP size; smaller = lower split latency")
    ap.add_argument("--spawn-stagger-s", type=float, default=1.0)
    ap.add_argument("--use-test-source", action="store_true",
                    help="dev host fallback: videotestsrc + x{264,265}enc instead of nvarguscamerasrc")
    ap.add_argument("--name-prefix", default="cam")
    ap.add_argument("--auto-episodes", type=int, default=0,
                    help="non-interactive mode: record this many episodes back-to-back then quit. "
                         "Useful for SSH-driven burn-in where stdin is piped.")
    ap.add_argument("--inter-episode-gap-s", type=float, default=0.5,
                    help="how long to dwell in WARMUP between auto episodes")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args(argv)


def _run_demo(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    sids = [int(s) for s in args.sids.split(",") if s.strip()]
    if not sids:
        print("error: --sids parsed empty", file=sys.stderr)
        return 2

    # Reset the warmup dir so disk doesn't fill up across runs.
    if args.warmup_dir.exists():
        shutil.rmtree(args.warmup_dir, ignore_errors=True)
    args.warmup_dir.mkdir(parents=True, exist_ok=True)
    args.episode_root.mkdir(parents=True, exist_ok=True)

    streams = [
        StreamConfig(
            sid=sid,
            name=f"{args.name_prefix}_{sid:02d}",
            width=args.width, height=args.height, fps=args.fps,
            codec=args.codec, bitrate_kbps=args.bitrate_kbps,
            iframe_interval=args.iframe_interval,
            use_test_source=args.use_test_source,
        )
        for sid in sids
    ]

    session = PersistentCameraSession(
        streams, args.warmup_dir, spawn_stagger_s=args.spawn_stagger_s,
    )

    print(f"connecting {len(sids)} streams (stagger {args.spawn_stagger_s}s)...", flush=True)
    session.connect()
    print(f"connected in {session.connect_duration_s:.2f}s", flush=True)

    def record_one_episode(idx: int) -> tuple[EpisodeHandle, float]:
        ep_dir = args.episode_root / f"episode_{idx:06d}"
        t_split_start = time.monotonic()
        handle = session.start_episode(ep_dir, idx)
        split_emit_ms = (time.monotonic() - t_split_start) * 1000
        print(
            f"episode {idx} started -> {ep_dir} (split-now emit "
            f"in {split_emit_ms:.1f}ms)",
            flush=True,
        )
        ep_t0 = time.monotonic()
        while time.monotonic() - ep_t0 < args.episode_time_s:
            time.sleep(0.05)
        actual = time.monotonic() - ep_t0
        session.stop_episode(handle)
        session.write_episode_meta(handle)
        sizes = ", ".join(
            f"{name}={_human_size(info.path)}@{info.first_pts_s if info.first_pts_s is not None else -1:.3f}s"
            for name, info in handle.fragments.items()
        )
        print(
            f"episode {idx} stopped after {actual:.2f}s. fragments: {sizes}",
            flush=True,
        )
        return handle, actual

    ep_idx = _next_episode_index(args.episode_root)
    rc = 0

    if args.auto_episodes > 0:
        print(f"auto mode: recording {args.auto_episodes} episodes back-to-back", flush=True)
        try:
            for _ in range(args.auto_episodes):
                record_one_episode(ep_idx)
                ep_idx += 1
                if args.inter_episode_gap_s > 0:
                    time.sleep(args.inter_episode_gap_s)
                for err in session.poll_errors():
                    print(f"!! stream error sid={err.sid} {err.name}: {err.message}", flush=True)
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            rc = 130
        finally:
            session.disconnect()
            print("disconnected.", flush=True)
        return rc

    print(
        "interactive commands: <Enter>=start a {:.1f}s episode, d=discard last, q=quit"
        .format(args.episode_time_s),
        flush=True,
    )

    cmd_queue: list[str] = []
    stop = threading.Event()
    reader = threading.Thread(
        target=_read_stdin_loop, args=(cmd_queue, stop),
        name="pcs-stdin", daemon=True,
    )
    reader.start()

    last_handle: EpisodeHandle | None = None
    try:
        while not stop.is_set():
            if not cmd_queue:
                for err in session.poll_errors():
                    print(f"!! stream error sid={err.sid} {err.name}: {err.message}", flush=True)
                time.sleep(0.05)
                continue
            cmd = cmd_queue.pop(0)
            if cmd == "q":
                break
            if cmd == "d":
                if last_handle is None:
                    print("nothing to discard", flush=True)
                    continue
                session.discard_episode(last_handle)
                print(f"discarded episode {last_handle.idx}", flush=True)
                last_handle = None
                continue
            if cmd not in ("", "e"):
                print(f"unknown command: {cmd!r}", flush=True)
                continue
            last_handle, _ = record_one_episode(ep_idx)
            ep_idx += 1
    except KeyboardInterrupt:
        print("interrupted; draining", flush=True)
        rc = 130
    finally:
        stop.set()
        session.disconnect()
        print("disconnected.", flush=True)
    return rc


def _next_episode_index(episode_root: Path) -> int:
    if not episode_root.is_dir():
        return 0
    existing = [
        int(p.name.removeprefix("episode_"))
        for p in episode_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
        and p.name.removeprefix("episode_").isdigit()
    ]
    return (max(existing) + 1) if existing else 0


def _human_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "missing"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


if __name__ == "__main__":
    sys.exit(_run_demo(_parse_args()))
