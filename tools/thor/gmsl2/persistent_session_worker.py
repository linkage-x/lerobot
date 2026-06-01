"""Single-stream GStreamer worker for ``PersistentCameraSession``.

Runs as a child process spawned by the parent ``PersistentCameraSession``.
One worker owns exactly one ``nvarguscamerasrc`` -> ... -> ``splitmuxsink``
pipeline plus the GLib MainLoop driving its bus.

Why a child process at all
--------------------------
PR2 (``6cfa9236``) moved 11 capture pipelines from independent gst-launch
subprocesses into one Python process. ``nvargus-daemon`` then sees a single
RPC client holding 11 ``Argus::CaptureSession`` instances on one socket and
falls over under that load: ``0x00000005`` socket errors,
``UNAVAILABLE/TIMEOUT/CANCELLED`` cascades, and ``set_state(PLAYING)``
deadlocks that hang the parent Python thread.

Wrapping each pipeline in its own subprocess restores the pre-PR2
``daemon-client`` relationship (N clients, one daemon) while keeping PR2's
"connect once, slice per-episode with split-now" UX.

Protocol
--------
Parent <-> child use two ``multiprocessing.Queue``\\s, passed as args::

    cmd_q  parent -> child:
        ("start_episode", str(episode_dir))
        ("stop_episode",)
        ("disconnect",)

    evt_q  child -> parent:
        ("playing",)
        ("fragment", FragmentInfo dict)         # warmup or episode
        ("episode_done", FragmentInfo dict|None) # after stop_episode finalize
        ("error", message: str, debug: str)
        ("eos",)
        ("disconnected",)

Fragments are serialized as plain ``dict``\\s (path -> str, enum -> str) so
they survive ``pickle`` across the spawn boundary; the parent rehydrates
them into ``persistent_session.FragmentInfo``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MpQueue
from pathlib import Path
from typing import Any

from tools.thor.gmsl2.persistent_session import (
    FragmentState,
    StreamConfig,
    build_pipeline_desc,
)

logger = logging.getLogger("ps_worker")


@dataclass
class _WorkerState:
    """Mutable per-worker state shared between the command thread and the
    GLib MainLoop callbacks."""

    cfg: StreamConfig
    warmup_dir: Path
    state: FragmentState = FragmentState.WARMUP
    current_episode_dir: Path | None = None
    last_episode_fragment: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _fragment_dict(
    cfg: StreamConfig,
    fragment_id: int,
    first_sample: Any,
    state: FragmentState,
    current_episode_dir: Path | None,
    warmup_dir: Path,
    gst_clock_time_none: int,
) -> dict:
    """Render a fragment event payload as a picklable dict.

    Pulled out as a free function so the unit tests can exercise the
    pts-extraction + path-selection logic without spinning up Gst.
    """
    first_pts_s: float | None = None
    try:
        if first_sample is not None:
            buf = first_sample.get_buffer()
            if buf is not None and buf.pts is not None:
                if buf.pts < gst_clock_time_none:
                    first_pts_s = float(buf.pts) / 1e9
    except Exception as exc:
        logger.debug("[%s] could not parse first_sample pts: %s", cfg.name, exc)

    if state == FragmentState.EPISODE and current_episode_dir is not None:
        path = current_episode_dir / f"{cfg.name}.mkv"
    else:
        path = warmup_dir / f"cam_{cfg.sid:02d}_warmup_{int(fragment_id):05d}.mkv"

    return {
        "sid": cfg.sid,
        "name": cfg.name,
        "fragment_id": int(fragment_id),
        "path": str(path),
        "first_pts_s": first_pts_s,
        "first_wall_s": time.time(),
        "state": state.value,
    }


def run_worker(
    cfg: StreamConfig,
    warmup_dir: Path,
    cmd_q: "MpQueue[Any]",
    evt_q: "MpQueue[Any]",
    *,
    ready_timeout_s: float = 6.0,
) -> int:
    """Child-process entrypoint. Returns an exit code."""

    # Lazy import inside the child so the parent never has to depend on gi.
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # type: ignore

    Gst.init(None)

    state = _WorkerState(cfg=cfg, warmup_dir=warmup_dir)

    warmup_template = str(warmup_dir / f"cam_{cfg.sid:02d}_warmup_%05d.mkv")
    desc = build_pipeline_desc(cfg, warmup_template)
    pipeline = Gst.parse_launch(desc)
    splitmux = pipeline.get_by_name(f"mux_{cfg.sid}")
    if splitmux is None:
        evt_q.put(("error", f"splitmuxsink mux_{cfg.sid} not found", ""))
        return 2

    encoder = None
    it = pipeline.iterate_elements()
    while True:
        result, elem = it.next()
        if result == Gst.IteratorResult.DONE or elem is None:
            break
        if result != Gst.IteratorResult.OK:
            continue
        factory = elem.get_factory()
        name = factory.get_name() if factory else ""
        if name in {"nvv4l2h265enc", "nvv4l2h264enc", "x264enc", "x265enc"}:
            encoder = elem

    def on_format_location_full(_mux, fragment_id, first_sample, *_user):
        with state.lock:
            cur_state = state.state
            cur_dir = state.current_episode_dir
        info = _fragment_dict(
            cfg, fragment_id, first_sample, cur_state, cur_dir,
            warmup_dir, Gst.CLOCK_TIME_NONE,
        )
        if cur_state == FragmentState.EPISODE:
            with state.lock:
                state.last_episode_fragment = info
        evt_q.put(("fragment", info))
        return info["path"]

    splitmux.connect("format-location-full", on_format_location_full)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            evt_q.put(("error", str(err), str(debug or "")))
        elif t == Gst.MessageType.EOS:
            evt_q.put(("eos",))

    bus.connect("message", on_bus)

    glib_loop = GLib.MainLoop.new(None, False)
    glib_thread = threading.Thread(
        target=glib_loop.run, name=f"ps-worker-glib-{cfg.sid}", daemon=True,
    )
    glib_thread.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        evt_q.put(("error", "set_state(PLAYING) returned FAILURE", ""))
        glib_loop.quit()
        return 3

    change, gst_state, _pending = pipeline.get_state(
        timeout=int(ready_timeout_s * Gst.SECOND),
    )
    if change == Gst.StateChangeReturn.FAILURE or gst_state != Gst.State.PLAYING:
        nick = gst_state.value_nick if hasattr(gst_state, "value_nick") else str(gst_state)
        evt_q.put((
            "error",
            f"did not reach PLAYING within {ready_timeout_s:.1f}s (stuck at {nick})",
            "",
        ))
        pipeline.set_state(Gst.State.NULL)
        glib_loop.quit()
        return 4

    evt_q.put(("playing",))

    finalize_grace_s = 0.5
    while True:
        cmd = cmd_q.get()
        if not isinstance(cmd, tuple) or len(cmd) == 0:
            continue
        kind = cmd[0]
        if kind == "start_episode":
            ep_dir = Path(cmd[1])
            with state.lock:
                state.state = FragmentState.EPISODE
                state.current_episode_dir = ep_dir
                state.last_episode_fragment = None
            if encoder is not None:
                try:
                    encoder.emit("force-IDR")
                except Exception as exc:
                    logger.debug("[%s] force-IDR not supported: %s", cfg.name, exc)
            try:
                splitmux.emit("split-now")
            except Exception as exc:
                evt_q.put(("error", f"split-now failed: {exc}", ""))
        elif kind == "stop_episode":
            with state.lock:
                state.state = FragmentState.WARMUP
                state.current_episode_dir = None
            try:
                splitmux.emit("split-now")
            except Exception as exc:
                logger.warning("[%s] split-now (stop) failed: %s", cfg.name, exc)
            # async-finalize must drain to disk before we report which file
            # we wrote. splitmuxsink in the 1.20 line does not expose a
            # finalize-done signal so we sleep a small grace window.
            time.sleep(finalize_grace_s)
            with state.lock:
                fragment = state.last_episode_fragment
                state.last_episode_fragment = None
            evt_q.put(("episode_done", fragment))
        elif kind == "disconnect":
            pipeline.set_state(Gst.State.NULL)
            glib_loop.quit()
            evt_q.put(("disconnected",))
            return 0
        else:
            logger.warning("[%s] ignoring unknown command %r", cfg.name, kind)
