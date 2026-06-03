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
import os
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MpQueue
from pathlib import Path
from typing import Any

from tools.thor.gmsl2.persistent_session import (
    PREVIEW_FPS,
    PREVIEW_WIDTH,
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

    preview_path = cfg.preview_jpeg_path
    preview_tmp_path = f"{preview_path}.{os.getpid()}.tmp" if preview_path else None
    preview_lock = threading.Lock()
    preview_elements: list[Any] = []
    preview_tee_pad: Any | None = None

    def _drop_preview_frame() -> None:
        if not preview_path:
            return
        for path in (preview_path, preview_tmp_path):
            if not path:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass

    def _make_element(factory: str, name: str):
        elem = Gst.ElementFactory.make(factory, name)
        if elem is None:
            raise RuntimeError(f"GStreamer element {factory!r} is unavailable")
        return elem

    def _link_elements(elements: list[Any]) -> None:
        for left, right in zip(elements, elements[1:]):
            if not left.link(right):
                raise RuntimeError(
                    f"could not link {left.get_name()} -> {right.get_name()}"
                )

    def _pad_link_return_name(ret: Any) -> str:
        nick = getattr(ret, "value_nick", None)
        if nick:
            return str(nick)
        try:
            return Gst.PadLinkReturn(ret).value_nick
        except Exception:
            return str(ret)

    def _pad_link_ok(ret: Any) -> bool:
        if ret == Gst.PadLinkReturn.OK:
            return True
        try:
            return int(ret) == int(Gst.PadLinkReturn.OK)
        except Exception:
            return False

    def _link_tee_to_queue(tee_pad: Any, queue_sink_pad: Any) -> None:
        ret = tee_pad.link(queue_sink_pad)
        if _pad_link_ok(ret) or (tee_pad.is_linked() and queue_sink_pad.is_linked()):
            return
        link_full = getattr(tee_pad, "link_full", None)
        if callable(link_full):
            fallback_ret = link_full(queue_sink_pad, Gst.PadLinkCheck.NOTHING)
            if (
                _pad_link_ok(fallback_ret)
                or (tee_pad.is_linked() and queue_sink_pad.is_linked())
            ):
                logger.info(
                    "[%s] preview tee link used fallback after %s",
                    cfg.name, _pad_link_return_name(ret),
                )
                return
            ret = fallback_ret
        try:
            src_caps = tee_pad.query_caps(None).to_string()
        except Exception:
            src_caps = "<unknown>"
        try:
            sink_caps = queue_sink_pad.query_caps(None).to_string()
        except Exception:
            sink_caps = "<unknown>"
        raise RuntimeError(
            "could not link tee to preview queue "
            f"(ret={_pad_link_return_name(ret)}, "
            f"tee_caps={src_caps}, queue_caps={sink_caps})"
        )

    def _request_tee_src_pad(tee):
        get_request_pad = getattr(tee, "get_request_pad", None)
        if callable(get_request_pad):
            return get_request_pad("src_%u")
        tmpl = tee.get_pad_template("src_%u")
        if tmpl is None:
            return None
        return tee.request_pad(tmpl, None, None)

    def _release_preview_branch_locked() -> None:
        nonlocal preview_elements, preview_tee_pad
        tee = pipeline.get_by_name(f"t_{cfg.sid}")
        if preview_tee_pad is not None:
            if preview_elements:
                q_sink = preview_elements[0].get_static_pad("sink")
                if q_sink is not None:
                    try:
                        preview_tee_pad.unlink(q_sink)
                    except Exception:
                        pass
            if tee is not None:
                try:
                    tee.release_request_pad(preview_tee_pad)
                except Exception:
                    pass
            preview_tee_pad = None
        for elem in reversed(preview_elements):
            try:
                elem.set_state(Gst.State.NULL)
            except Exception:
                pass
        for elem in list(preview_elements):
            try:
                pipeline.remove(elem)
            except Exception:
                pass
        preview_elements = []

    def _disable_preview_branch() -> None:
        with preview_lock:
            _release_preview_branch_locked()
            _drop_preview_frame()

    def _enable_preview_branch() -> None:
        nonlocal preview_elements, preview_tee_pad
        if not preview_path:
            return
        with preview_lock:
            if preview_elements:
                return
            tee = pipeline.get_by_name(f"t_{cfg.sid}")
            if tee is None:
                logger.warning("[%s] preview tee not found", cfg.name)
                return
            try:
                Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("[%s] preview dir create failed: %s", cfg.name, exc)
                return

            elements: list[Any] = []
            tee_pad = None
            try:
                pw = PREVIEW_WIDTH
                ph = max(2, int(round(pw * cfg.height / max(cfg.width, 1))) // 2 * 2)
                q = _make_element("queue", f"prevq_{cfg.sid}")
                q.set_property("leaky", 2)  # downstream
                q.set_property("max-size-buffers", 1)
                q.set_property("max-size-bytes", 0)
                q.set_property("max-size-time", 0)
                elements.append(q)

                if cfg.use_test_source:
                    elements.extend([
                        _make_element("videoconvert", f"prevvc_{cfg.sid}"),
                        _make_element("videoscale", f"prevscale_{cfg.sid}"),
                    ])
                else:
                    # Preview is fed from the raw pre-encoder tee. Avoid opening
                    # an extra H26x decoder per camera; 11 decoder branches can
                    # exhaust VIC/NVDEC contexts on Thor. nvvidconv only does the
                    # low-resolution colorspace/scale step needed for JPEG.
                    elements.append(_make_element("nvvidconv", f"prevconv_{cfg.sid}"))

                size_filter = _make_element("capsfilter", f"prevcaps_{cfg.sid}")
                size_filter.set_property(
                    "caps",
                    Gst.Caps.from_string(
                        f"video/x-raw,format=I420,width={pw},height={ph}"
                    ),
                )
                elements.append(size_filter)

                elements.append(_make_element("videorate", f"prevrate_{cfg.sid}"))
                fps_filter = _make_element("capsfilter", f"prevfps_{cfg.sid}")
                fps_filter.set_property(
                    "caps", Gst.Caps.from_string(f"video/x-raw,framerate={PREVIEW_FPS}/1"),
                )
                elements.append(fps_filter)

                enc = _make_element("jpegenc", f"prevenc_{cfg.sid}")
                enc.set_property("quality", 60)
                elements.append(enc)

                appsink = _make_element("appsink", f"preview_{cfg.sid}")
                appsink.set_property("emit-signals", True)
                appsink.set_property("max-buffers", 1)
                appsink.set_property("drop", True)
                appsink.set_property("sync", False)

                def on_preview_sample(sink):
                    sample = sink.emit("pull-sample")
                    if sample is None:
                        return Gst.FlowReturn.OK
                    buf = sample.get_buffer()
                    ok, mapinfo = buf.map(Gst.MapFlags.READ)
                    if not ok:
                        return Gst.FlowReturn.OK
                    try:
                        data = bytes(mapinfo.data)
                    finally:
                        buf.unmap(mapinfo)
                    try:
                        if preview_tmp_path is not None:
                            with open(preview_tmp_path, "wb") as fh:
                                fh.write(data)
                            os.replace(preview_tmp_path, preview_path)
                    except OSError as exc:
                        logger.debug("[%s] preview write failed: %s", cfg.name, exc)
                    return Gst.FlowReturn.OK

                appsink.connect("new-sample", on_preview_sample)
                elements.append(appsink)

                for elem in elements:
                    pipeline.add(elem)
                _link_elements(elements)
                tee_pad = _request_tee_src_pad(tee)
                if tee_pad is None:
                    raise RuntimeError("could not request tee src pad")
                q_sink = q.get_static_pad("sink")
                if q_sink is None:
                    raise RuntimeError("preview queue sink pad missing")
                _link_tee_to_queue(tee_pad, q_sink)
                for elem in elements:
                    elem.sync_state_with_parent()
                preview_elements = elements
                preview_tee_pad = tee_pad
                if encoder is not None:
                    try:
                        encoder.emit("force-IDR")
                    except Exception as exc:
                        logger.debug("[%s] force-IDR (preview) not supported: %s", cfg.name, exc)
                logger.info("[%s] preview branch enabled", cfg.name)
            except Exception as exc:
                if tee_pad is not None:
                    try:
                        tee.release_request_pad(tee_pad)
                    except Exception:
                        pass
                for elem in reversed(elements):
                    try:
                        elem.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                for elem in elements:
                    try:
                        pipeline.remove(elem)
                    except Exception:
                        pass
                logger.warning("[%s] preview branch enable failed: %s", cfg.name, exc)
                _drop_preview_frame()

    def _set_preview_enabled(enabled: bool) -> None:
        if enabled:
            _enable_preview_branch()
        else:
            _disable_preview_branch()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            src = message.src
            src_name = src.get_name() if src is not None else ""
            err, debug = message.parse_error()
            if src_name.startswith("prev") or src_name.startswith("preview"):
                logger.warning("[%s] preview branch error (ignored): %s", cfg.name, err)
                _set_preview_enabled(False)
                return
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
            # Mirror start_episode: splitmuxsink can only cut on an IDR
            # boundary, and idrinterval=60 means the natural cadence is
            # ~1 frame/sec. Without force-IDR here the EPISODE fragment
            # loses 0-1s of frames at the tail because splitmux silently
            # waits for the next natural IDR before swapping out. Users
            # saw the slider freeze ~60 frames short of duration_s * fps.
            if encoder is not None:
                try:
                    encoder.emit("force-IDR")
                except Exception as exc:
                    logger.debug("[%s] force-IDR (stop) not supported: %s", cfg.name, exc)
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
        elif kind == "enable_preview":
            _set_preview_enabled(True)
        elif kind == "disable_preview":
            _set_preview_enabled(False)
        elif kind == "disconnect":
            _set_preview_enabled(False)
            pipeline.set_state(Gst.State.NULL)
            glib_loop.quit()
            evt_q.put(("disconnected",))
            return 0
        else:
            logger.warning("[%s] ignoring unknown command %r", cfg.name, kind)
