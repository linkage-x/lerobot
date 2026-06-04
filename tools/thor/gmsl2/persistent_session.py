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
    # Recorder-owned live preview. The initial pipeline only includes the
    # recording branch plus an encoded-stream tee; the worker attaches/removes
    # the preview branch after connect() so Argus raw-surface readiness is not
    # perturbed by preview conversion.
    preview_jpeg_path: str | None = None


# Bounded wait for splitmuxsink to finalize the just-closed EPISODE fragment
# to disk. GStreamer 1.18+ posts a ``splitmuxsink-fragment-closed`` element
# message when async-finalize completes; the worker waits for it instead of
# blind-sleeping. This is only the fallback ceiling — in the normal case the
# message lands within a few ms of the stop split-now and the worker proceeds
# immediately. Kept module-level so the parent's stop_episode timeout stays in
# sync with the worker's worst-case finalize wait.
FINALIZE_FRAGMENT_TIMEOUT_S = 3.0


# Latest recorder-owned preview frames. The gateway serves these files while
# the recorder owns the cameras, avoiding a second nvarguscamerasrc client.
PREVIEW_DIR = Path("/dev/shm/lerobot_preview")
PREVIEW_WIDTH = 480
PREVIEW_FPS = 5


def preview_frame_path(name: str) -> Path:
    return PREVIEW_DIR / f"{name}.jpg"


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


def format_stream_error(error: StreamError, *, max_detail: int = 180) -> str:
    """Compact one-line error text for operator-facing logs."""
    message = " ".join(str(error.message or "").split())
    debug = " ".join(str(error.debug or "").split())
    detail = message or "unknown error"
    if debug:
        detail = f"{detail} | {debug}"
    if len(detail) > max_detail:
        detail = detail[: max(0, max_detail - 3)] + "..."
    return f"{error.name}({detail})"


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

    encode_chain = f"! {encoder} ! {parser}"
    sink_chain = (
        f"! splitmuxsink name=mux_{stream.sid} "
        f"  muxer-factory={muxer_factory} "
        f"  async-finalize=true "
        f"  max-size-time=0 max-size-bytes=0 "
        f"  location=\"{warmup_location}\""
    )

    if not stream.preview_jpeg_path:
        return f"{source} {encode_chain} {sink_chain}"

    sid = stream.sid
    return (
        f"{source} ! tee name=t_{sid} "
        f"t_{sid}. ! queue name=recq_{sid} {encode_chain} {sink_chain}"
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
    if kind == "paused":
        proxy.paused_evt.set()
        return
    if kind == "playing":
        # A worker that reached PLAYING necessarily passed PAUSED; set both so
        # a two-phase wait_paused() can never miss a fast worker.
        proxy.paused_evt.set()
        proxy.ready_evt.set()
        return
    if kind == "fragment":
        info_dict = event[1]
        proxy.first_fragment_evt.set()
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
        proxy.recording_failed = True
        # Unblock connect() on terminal failure — both the two-phase PAUSED
        # wait and the PLAYING wait, so a dead worker never hangs either phase.
        proxy.paused_evt.set()
        proxy.ready_evt.set()
        return
    if kind == "eos":
        proxy.errors.append(StreamError(
            sid=proxy.cfg.sid, name=proxy.cfg.name,
            message="bus EOS (upstream stopped delivering buffers)",
        ))
        proxy.recording_failed = True
        proxy.paused_evt.set()
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
        ready_timeout_s: float = 6.0,
        on_fragment_opened: Callable[["_StreamProxy", FragmentInfo], None] | None = None,
    ):
        self.cfg = cfg
        self.warmup_dir = warmup_dir
        self._ctx = ctx
        self.ready_timeout_s = float(ready_timeout_s)
        self.cmd_q: Any = ctx.Queue()
        self.evt_q: Any = ctx.Queue()
        self.proc: Any = None
        self.reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self.ready_evt = threading.Event()
        self.paused_evt = threading.Event()
        self.first_fragment_evt = threading.Event()
        self.episode_done_evt = threading.Event()
        self.disconnected_evt = threading.Event()
        self.last_episode_fragment: FragmentInfo | None = None
        self.fragment_history: list[FragmentInfo] = []
        self.errors: list[StreamError] = []
        self._on_fragment_opened_cb = on_fragment_opened
        # Sticky "the recording pipeline died" marker. Unlike ``errors`` (drained
        # by poll_errors), this survives error draining so the preview watchdog
        # can tell "lossy preview branch wedged, recording still alive" (restart
        # the branch) from "upstream nvarguscamerasrc is gone" (restarting the
        # branch is futile). Set on any error/EOS worker event; reset only by
        # replacing the proxy via restart_stream.
        self.recording_failed = False
        # One-shot guard so refresh_stale_previews reports a dead stream once
        # instead of every watchdog tick.
        self._preview_down_logged = False

    # -- lifecycle --------------------------------------------------------

    def spawn(self, *, two_phase: bool = False) -> None:
        # Imported lazily so the parent can still be imported on machines
        # without gi.repository (e.g. dev hosts where only build_pipeline_desc
        # is unit-tested).
        from tools.thor.gmsl2 import persistent_session_worker as worker

        self.proc = self._ctx.Process(
            target=worker.run_worker,
            args=(self.cfg, self.warmup_dir, self.cmd_q, self.evt_q),
            kwargs={
                "ready_timeout_s": self.ready_timeout_s,
                "two_phase": two_phase,
            },
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

    def wait_first_fragment(self, timeout_s: float) -> bool:
        """Block until the worker opens its first warmup fragment.

        A worker can report PLAYING before Argus/NVMM failures surface. The
        first splitmux fragment is the stronger signal that encoded camera
        buffers are actually flowing through the production path.
        """
        if timeout_s <= 0:
            return True
        if self.first_fragment_evt.wait(timeout_s):
            return True
        self.errors.append(StreamError(
            sid=self.cfg.sid, name=self.cfg.name,
            message=f"no warmup fragment within {timeout_s:.1f}s after PLAYING",
        ))
        self.ready_evt.set()
        return False

    def wait_paused(self, timeout_s: float) -> bool:
        """Block until a two-phase worker reports PAUSED or a terminal failure.

        Returns True iff the worker reached PAUSED with no error queued. This
        is the cheap Phase-1 gate: PAUSED opens no Argus session, so all
        workers can be waited on concurrently.
        """
        if not self.paused_evt.wait(timeout_s):
            self.errors.append(StreamError(
                sid=self.cfg.sid, name=self.cfg.name,
                message=f"worker did not reach PAUSED within {timeout_s:.1f}s",
            ))
            return False
        return not self.errors

    def play(self) -> None:
        """Release a two-phase worker from PAUSED into the PLAYING bring-up."""
        self.cmd_q.put(("play",))

    def start_episode(self, episode_dir: Path) -> None:
        self.episode_done_evt.clear()
        self.last_episode_fragment = None
        self.cmd_q.put(("start_episode", str(episode_dir)))

    def stop_episode(self) -> None:
        self.cmd_q.put(("stop_episode",))

    def enable_preview(self) -> None:
        self.cmd_q.put(("enable_preview",))

    def disable_preview(self) -> None:
        self.cmd_q.put(("disable_preview",))

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
        connect_stable_s: float = 0.0,
        first_fragment_timeout_s: float = 0.0,
        two_phase_connect: bool = False,
        on_fragment_opened: Callable[["_StreamProxy", FragmentInfo], None] | None = None,
    ):
        self._stream_cfgs = list(streams)
        self.warmup_dir = warmup_dir
        self.spawn_stagger_s = float(spawn_stagger_s)
        self.finalize_grace_s = float(finalize_grace_s)
        self.ready_timeout_s = float(ready_timeout_s)
        self.connect_stable_s = max(0.0, float(connect_stable_s))
        self.first_fragment_timeout_s = max(0.0, float(first_fragment_timeout_s))
        # When True, connect() spawns all workers to PAUSED at once (overlapping
        # their python/Gst.init startup) and then serializes only the
        # PAUSED->PLAYING Argus bring-up. Default keeps the proven one-at-a-time
        # spawn-then-PLAYING path.
        self.two_phase_connect = bool(two_phase_connect)
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

    def _new_stream_proxy(self, cfg: StreamConfig, ctx: Any) -> _StreamProxy:
        return _StreamProxy(
            cfg, self.warmup_dir, ctx,
            ready_timeout_s=self.ready_timeout_s,
            on_fragment_opened=self._on_fragment_opened_cb,
        )

    def _drop_streams(self, sids: set[int]) -> None:
        for sid in sorted(sids):
            proxy = self._streams.get(sid)
            if proxy is None:
                continue
            try:
                proxy.disconnect(grace_s=1.0)
            except Exception:
                pass
            self._streams.pop(sid, None)

    def _wait_connect_stable(self, *, reason: str) -> list[StreamError]:
        if self.connect_stable_s <= 0 or not self._streams:
            return []
        time.sleep(self.connect_stable_s)
        errors = self.poll_errors()
        if errors:
            logger.warning(
                "connect stable window failed after %s: %s",
                reason, ", ".join(format_stream_error(e) for e in errors),
            )
        return errors

    def _wait_proxy_ready_and_stable(
        self, proxy: _StreamProxy, *, t0: float, reason: str,
    ) -> tuple[bool, list[StreamError]]:
        ok = proxy.wait_ready(self.ready_timeout_s)
        elapsed = time.time() - t0
        if ok and not proxy.wait_first_fragment(self.first_fragment_timeout_s):
            ok = False
        if ok:
            logger.info("[%s] PLAYING (+%.2fs)", proxy.cfg.name, elapsed)
        else:
            logger.warning(
                "[%s] failed to reach PLAYING (+%.2fs)",
                proxy.cfg.name, elapsed,
            )
        errors = self.poll_errors()
        if ok:
            errors.extend(self._wait_connect_stable(reason=reason))
        failed_sids = {e.sid for e in errors}
        if not ok and proxy.cfg.sid not in failed_sids:
            errors.append(StreamError(
                sid=proxy.cfg.sid,
                name=proxy.cfg.name,
                message=(
                    f"did not reach PLAYING within {self.ready_timeout_s:.1f}s "
                    "(stuck before ready)"
                ),
            ))
            failed_sids.add(proxy.cfg.sid)
        if failed_sids:
            self._drop_streams(failed_sids)
        return ok and proxy.cfg.sid not in failed_sids, errors

    def _first_pass_serial(
        self, ctx: Any, cfg_by_sid: dict[int, StreamConfig],
        sids: list[int], t0: float,
    ) -> list[StreamError]:
        """Proven path: spawn one Argus client, wait until it reaches PLAYING,
        then give already-live streams a small stability window before moving
        to the next. This avoids overlapping N CaptureSession/NVMM/NVENC
        bring-ups, which produces dmabuf_fd=-1 / NvBufSurfaceFromFd failures."""
        first_round_errors: list[StreamError] = []
        for i, sid in enumerate(sids):
            if i > 0 and self.spawn_stagger_s > 0:
                time.sleep(self.spawn_stagger_s)
            proxy = self._new_stream_proxy(cfg_by_sid[sid], ctx)
            self._streams[sid] = proxy
            proxy.spawn()
            logger.info("[%s] spawned (+%.2fs)", proxy.cfg.name, time.time() - t0)
            _ok, errors = self._wait_proxy_ready_and_stable(
                proxy, t0=t0, reason=f"sid={sid} first pass",
            )
            first_round_errors.extend(errors)
        return first_round_errors

    def _first_pass_two_phase(
        self, ctx: Any, cfg_by_sid: dict[int, StreamConfig],
        sids: list[int], t0: float,
    ) -> list[StreamError]:
        """Two-phase path. Phase 1: spawn every worker to PAUSED concurrently
        (PAUSED opens no Argus session, so there is no daemon contention and
        the ~1.2s python+Gst.init startup overlaps across cameras). Phase 2:
        serialize PAUSED->PLAYING exactly like the serial path — same ordering,
        same per-camera stability window — so the contentious Argus bring-up is
        never concurrent. The net win is the overlapped startup, not any change
        to how the risky transition is sequenced."""
        first_round_errors: list[StreamError] = []

        # Phase 1: spawn all, no stagger — they touch nothing shared yet.
        for sid in sids:
            proxy = self._new_stream_proxy(cfg_by_sid[sid], ctx)
            self._streams[sid] = proxy
            proxy.spawn(two_phase=True)
        logger.info("two-phase: spawned %d workers (+%.2fs)", len(sids), time.time() - t0)

        # Wait for each worker to reach PAUSED (concurrent; cheap).
        paused_failed: set[int] = set()
        for sid in sids:
            proxy = self._streams.get(sid)
            if proxy is None:
                continue
            if not proxy.wait_paused(self.ready_timeout_s):
                paused_failed.add(sid)
        if paused_failed:
            errors = self.poll_errors()
            seen = {e.sid for e in errors}
            for sid in paused_failed:
                if sid not in seen:
                    errors.append(StreamError(
                        sid=sid, name=cfg_by_sid[sid].name,
                        message="worker failed before reaching PAUSED",
                    ))
            self._drop_streams(paused_failed)
            first_round_errors.extend(errors)
        logger.info(
            "two-phase: %d/%d PAUSED (+%.2fs)",
            len(self._streams), len(sids), time.time() - t0,
        )

        # Phase 2: serialize the PAUSED->PLAYING Argus bring-up.
        for i, sid in enumerate(sids):
            proxy = self._streams.get(sid)
            if proxy is None:  # dropped in Phase 1
                continue
            if i > 0 and self.spawn_stagger_s > 0:
                time.sleep(self.spawn_stagger_s)
            proxy.play()
            logger.info("[%s] play (+%.2fs)", proxy.cfg.name, time.time() - t0)
            _ok, errors = self._wait_proxy_ready_and_stable(
                proxy, t0=t0, reason=f"sid={sid} first pass (two-phase)",
            )
            first_round_errors.extend(errors)
        return first_round_errors

    def connect(self) -> None:
        if self._streams:
            raise RuntimeError("connect() called twice without disconnect()")
        self.warmup_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._ctx_factory()
        cfg_by_sid = {cfg.sid: cfg for cfg in self._stream_cfgs}

        t0 = time.time()
        sids = sorted(cfg_by_sid)
        retry_sids: set[int] = set()

        if self.two_phase_connect:
            first_round_errors = self._first_pass_two_phase(ctx, cfg_by_sid, sids, t0)
        else:
            first_round_errors = self._first_pass_serial(ctx, cfg_by_sid, sids, t0)
        retry_sids.update(e.sid for e in first_round_errors)

        # Final all-live stability window before retry. PLAYING alone is not
        # sufficient on Argus: recent hardware logs show EOS/TIMEOUT can land
        # immediately after readiness.
        final_first_errors = self._wait_connect_stable(reason="first pass all-live")
        if final_first_errors:
            first_round_errors.extend(final_first_errors)
            retry_sids.update(e.sid for e in final_first_errors)
            self._drop_streams({e.sid for e in final_first_errors})

        retry_errors: list[StreamError] = []
        if retry_sids:
            retry_queue = sorted(sid for sid in retry_sids if sid in cfg_by_sid)
            retried: set[int] = set()
            logger.warning(
                "connect first pass: %d/%d failed; retrying %s sequentially",
                len(retry_queue), len(cfg_by_sid), retry_queue,
            )
            while retry_queue:
                sid = retry_queue.pop(0)
                if sid in retried or sid not in cfg_by_sid:
                    continue
                retried.add(sid)
                if self.spawn_stagger_s > 0:
                    time.sleep(self.spawn_stagger_s)
                ok = self.restart_stream(sid)
                errors = self.poll_errors()
                failed_sids = {e.sid for e in errors}
                if not ok and sid not in failed_sids:
                    errors.append(StreamError(
                        sid=sid, name=cfg_by_sid[sid].name,
                        message="restart_stream did not reach PLAYING",
                    ))
                    failed_sids.add(sid)
                if failed_sids:
                    self._drop_streams(failed_sids)
                for err in errors:
                    if err.sid in retried:
                        retry_errors.append(err)
                    elif err.sid not in retry_queue:
                        retry_queue.append(err.sid)

        final_errors = self._wait_connect_stable(reason="post-retry all-live")
        if final_errors:
            failed_sids = {e.sid for e in final_errors}
            self._drop_streams(failed_sids)
            retry_errors.extend(final_errors)

        self.connect_duration_s = time.time() - t0

        # Only retry-surviving errors are surfaced. First-pass errors for sids
        # that were rescued are intentionally swallowed.
        errors = retry_errors
        if errors:
            failed_sids = {e.sid for e in errors}
            self._drop_streams(failed_sids)
            if not self._streams:
                bad = ", ".join(format_stream_error(e) for e in errors)
                raise RuntimeError(
                    f"connect() failed on all {len(errors)} stream(s): {bad}"
                )
            with self._lock:
                self._errors.extend(errors)
            bad = ", ".join(format_stream_error(e) for e in errors)
            logger.warning(
                "connect() partial success: %d/%d streams up; failed: %s",
                len(self._streams), len(self._stream_cfgs), bad,
            )
            return

        logger.info(
            "connected %d streams in %.2fs (stagger=%.2fs stable=%.2fs)",
            len(self._streams), self.connect_duration_s,
            self.spawn_stagger_s, self.connect_stable_s,
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
        # Each worker waits (bounded by FINALIZE_FRAGMENT_TIMEOUT_S) for the
        # splitmuxsink-fragment-closed message before emitting episode_done;
        # we just have to wait for those events to land. Cover the worker's
        # worst case (message never arrives -> message timeout + fallback
        # grace) plus a margin.
        per_worker_timeout = (
            FINALIZE_FRAGMENT_TIMEOUT_S + self.finalize_grace_s + 1.5
        )
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

    # -- live preview ----------------------------------------------------

    def enable_previews(self, *, stagger_s: float = 0.5) -> None:
        """Enable recorder-owned preview branches after connect().

        The worker dynamically attaches each branch to the recording tee, so
        connect() only has to prove the recording path can reach PLAYING. We
        still stagger branch creation to avoid a burst of nvvidconv/jpegenc
        allocation across all cameras.
        """
        for i, proxy in enumerate(self._streams.values()):
            if not proxy.cfg.preview_jpeg_path:
                continue
            if i > 0 and stagger_s > 0:
                time.sleep(stagger_s)
            proxy.enable_preview()

    def wait_preview_frames(self, *, timeout_s: float = 5.0) -> list[str]:
        """Wait briefly for recorder-owned preview JPEGs to appear.

        Returns camera names whose preview path did not produce a frame before
        the timeout. This is only a preview readiness check; stream health is
        still reported through poll_errors().
        """
        deadline = time.time() + max(0.0, timeout_s)
        missing: set[str] = {
            proxy.cfg.name
            for proxy in self._streams.values()
            if proxy.cfg.preview_jpeg_path
        }
        while missing and time.time() < deadline:
            for proxy in list(self._streams.values()):
                if proxy.cfg.name not in missing or not proxy.cfg.preview_jpeg_path:
                    continue
                path = Path(proxy.cfg.preview_jpeg_path)
                try:
                    if path.stat().st_size > 0:
                        missing.discard(proxy.cfg.name)
                except OSError:
                    pass
            if missing:
                time.sleep(0.1)
        return sorted(missing)

    def disable_previews(self) -> None:
        for proxy in self._streams.values():
            if proxy.cfg.preview_jpeg_path:
                proxy.disable_preview()

    def refresh_stale_previews(self, *, max_age_s: float) -> list[str]:
        """Restart recorder-owned preview branches whose JPEG stopped updating.

        This only touches the lossy preview branch in each worker; it does not
        restart the recording pipeline. A stale preview JPEG has two very
        different causes that must be handled differently:

        * The lossy preview branch alone wedged while the recording pipeline is
          still PLAYING -> restarting just that branch is the right, cheap fix.
        * The whole recording pipeline died (Argus error / bus EOS), so the
          upstream ``nvarguscamerasrc`` is gone -> restarting the preview branch
          can never produce a frame. Doing it every watchdog tick spins forever
          (the 2026-06-03 "Preview stale: restarted ..." storm, where 11 dead
          cameras were "restarted" every 2s indefinitely).

        ``proxy.recording_failed`` (set on any worker error/EOS event) tells the
        two apart: dead streams are reported once, then skipped. Their recovery
        belongs to poll_errors() / an operator reconnect, not this watchdog.
        """
        if max_age_s <= 0:
            return []
        now = time.time()
        restarted: list[str] = []
        for proxy in list(self._streams.values()):
            path_raw = proxy.cfg.preview_jpeg_path
            if not path_raw:
                continue
            path = Path(path_raw)
            try:
                age_s = now - path.stat().st_mtime
            except OSError:
                age_s = max_age_s + 1.0
            if age_s <= max_age_s:
                proxy._preview_down_logged = False
                continue
            if proxy.recording_failed:
                # Upstream camera is dead; restarting the preview fork is futile
                # and the source of the restart storm. Report once, then stay
                # quiet until the cameras are reconnected (a fresh proxy clears
                # the flag). NOTE: keep this message free of the gateway's
                # recorder-failure keywords so it stays purely informational.
                if not proxy._preview_down_logged:
                    proxy._preview_down_logged = True
                    logger.warning(
                        "[%s] preview down: recording stream is dead; preview "
                        "cannot recover until the cameras are reconnected",
                        proxy.cfg.name,
                    )
                continue
            logger.warning(
                "[%s] recorder preview stale (age %.1fs); restarting preview branch",
                proxy.cfg.name, age_s,
            )
            proxy.disable_preview()
            proxy.enable_preview()
            restarted.append(proxy.cfg.name)
        return restarted

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
        True on success. The retry path is intentionally sequential and uses
        the same post-PLAYING stability window as the first pass.
        """
        cfg_by_sid = {cfg.sid: cfg for cfg in self._stream_cfgs}
        old = self._streams.get(sid)
        if old is not None:
            cfg = old.cfg
            try:
                old.disconnect(grace_s=1.0)
            except Exception as exc:
                logger.warning("restart_stream sid=%s disconnect raised: %s", sid, exc)
        else:
            cfg = cfg_by_sid.get(sid)
            if cfg is None:
                return False
        ctx = self._ctx_factory()
        new = self._new_stream_proxy(cfg, ctx)
        self._streams[sid] = new
        try:
            new.spawn()
            ok, errors = self._wait_proxy_ready_and_stable(
                new, t0=time.time(), reason=f"sid={sid} retry",
            )
            if errors:
                self._drop_streams({e.sid for e in errors})
                with self._lock:
                    self._errors.extend(errors)
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
    ap.add_argument("--two-phase", action="store_true",
                    help="PR7: spawn all workers to PAUSED concurrently, then "
                         "serialize PAUSED->PLAYING (overlaps per-worker startup)")
    ap.add_argument("--connect-stable-s", type=float, default=0.0,
                    help="post-PLAYING stability window per camera before next")
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
        connect_stable_s=args.connect_stable_s,
        two_phase_connect=args.two_phase,
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
