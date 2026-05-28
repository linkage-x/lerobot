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
# Lazy GStreamer import
# ---------------------------------------------------------------------------


def _gst_module():
    """Import gi.repository.Gst lazily so unit tests can mock the module."""
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # type: ignore

    Gst.init(None)
    return Gst, GLib


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
    gain: int = 0
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
        if stream.gain > 0:
            gain_clause = f"gainrange=\"{stream.gain} {stream.gain}\" "

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
# Stream handle
# ---------------------------------------------------------------------------


class _Stream:
    """One persistent pipeline, owned by the session."""

    def __init__(self, cfg: StreamConfig, warmup_dir: Path, session: "PersistentCameraSession"):
        self.cfg = cfg
        self.warmup_dir = warmup_dir
        self.session = session
        self.state: FragmentState = FragmentState.WARMUP
        self.current_episode_dir: Path | None = None
        self.fragment_history: list[FragmentInfo] = []
        # The most recently opened fragment per (state, episode_dir):
        self.last_episode_fragment: FragmentInfo | None = None
        # `gi` objects
        self.pipeline: Any = None
        self.splitmux: Any = None
        self.encoder: Any = None
        self.bus: Any = None
        self.bus_watch_id: int | None = None

    @property
    def warmup_location_template(self) -> str:
        return str(self.warmup_dir / f"cam_{self.cfg.sid:02d}_warmup_%05d.mkv")

    def build(self, Gst: Any) -> None:
        desc = build_pipeline_desc(self.cfg, self.warmup_location_template)
        logger.debug("[%s] pipeline desc: %s", self.cfg.name, desc)
        self.pipeline = Gst.parse_launch(desc)
        self.splitmux = self.pipeline.get_by_name(f"mux_{self.cfg.sid}")
        if self.splitmux is None:
            raise RuntimeError(f"[{self.cfg.name}] splitmuxsink not found by name")
        # nvv4l2h{265,264}enc exposes the `force-IDR` action signal so we can
        # cut the IDR latency on episode boundaries down to ~one frame.
        self.encoder = None
        it = self.pipeline.iterate_elements()
        while True:
            result, elem = it.next()
            if result == Gst.IteratorResult.DONE or elem is None:
                break
            if result != Gst.IteratorResult.OK:
                continue
            factory = elem.get_factory()
            name = factory.get_name() if factory else ""
            if name in {"nvv4l2h265enc", "nvv4l2h264enc", "x264enc", "x265enc"}:
                self.encoder = elem
        self.splitmux.connect("format-location-full", self._on_format_location_full)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus_watch_id = self.bus.connect("message", self._on_bus_message)

    def start(self, Gst: Any) -> None:
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"[{self.cfg.name}] pipeline set_state(PLAYING) returned FAILURE")

    def stop(self, Gst: Any) -> None:
        if self.bus is not None and self.bus_watch_id is not None:
            try:
                self.bus.disconnect(self.bus_watch_id)
            except Exception:
                pass
            self.bus.remove_signal_watch()
            self.bus_watch_id = None
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

    def force_split(self) -> None:
        """Ask the encoder for an IDR and the muxer for a new fragment."""
        if self.encoder is not None:
            try:
                self.encoder.emit("force-IDR")
            except Exception as exc:
                # x264enc / x265enc may not expose force-IDR on all platforms;
                # the muxer will still cut on the next natural keyframe.
                logger.debug("[%s] force-IDR not supported: %s", self.cfg.name, exc)
        try:
            self.splitmux.emit("split-now")
        except Exception as exc:
            logger.warning("[%s] split-now emit failed: %s", self.cfg.name, exc)

    # -- callbacks --------------------------------------------------------

    def _on_format_location_full(self, _mux, fragment_id, first_sample, *_user) -> str:
        # `first_sample` is a Gst.Sample for the buffer that triggered the
        # split; its PTS is the actual first frame of the new fragment.
        first_pts_s: float | None = None
        try:
            if first_sample is not None:
                buf = first_sample.get_buffer()
                if buf is not None and buf.pts is not None:
                    # Gst.CLOCK_TIME_NONE == 2**64 - 1
                    if buf.pts < (1 << 60):
                        first_pts_s = float(buf.pts) / 1e9
        except Exception as exc:
            logger.debug("[%s] could not parse first_sample pts: %s", self.cfg.name, exc)

        wall_s = time.time()
        state = self.state
        if state == FragmentState.EPISODE and self.current_episode_dir is not None:
            path = self.current_episode_dir / f"{self.cfg.name}.mkv"
        else:
            path = self.warmup_dir / f"cam_{self.cfg.sid:02d}_warmup_{fragment_id:05d}.mkv"

        info = FragmentInfo(
            sid=self.cfg.sid,
            name=self.cfg.name,
            fragment_id=int(fragment_id),
            path=path,
            first_pts_s=first_pts_s,
            first_wall_s=wall_s,
            state=state,
        )
        self.fragment_history.append(info)
        if state == FragmentState.EPISODE:
            self.last_episode_fragment = info
        self.session._on_fragment_opened(self, info)
        logger.debug(
            "[%s] format-location-full fragment=%d state=%s pts=%.6fs -> %s",
            self.cfg.name, fragment_id, state.value,
            first_pts_s if first_pts_s is not None else -1.0,
            path,
        )
        return str(path)

    def _on_bus_message(self, _bus, message) -> None:
        Gst = self.session._Gst
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.session._record_error(StreamError(
                sid=self.cfg.sid, name=self.cfg.name,
                message=str(err), debug=str(debug or ""),
            ))
            logger.warning("[%s] bus ERROR: %s | %s", self.cfg.name, err, debug)
        elif t == Gst.MessageType.EOS:
            logger.info("[%s] bus EOS", self.cfg.name)
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("[%s] bus WARNING: %s | %s", self.cfg.name, err, debug)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class PersistentCameraSession:
    """N persistent GStreamer pipelines with on-demand episode slicing."""

    def __init__(
        self,
        streams: list[StreamConfig],
        warmup_dir: Path,
        *,
        spawn_stagger_s: float = 1.0,
        finalize_grace_s: float = 0.5,
        on_fragment_opened: Callable[["_Stream", FragmentInfo], None] | None = None,
    ):
        self._stream_cfgs = list(streams)
        self.warmup_dir = warmup_dir
        self.spawn_stagger_s = float(spawn_stagger_s)
        self.finalize_grace_s = float(finalize_grace_s)
        self._on_fragment_opened_cb = on_fragment_opened
        self._streams: dict[int, _Stream] = {}
        self._errors: list[StreamError] = []
        self._lock = threading.Lock()
        self._Gst: Any = None
        self._GLib: Any = None
        self._glib_loop: Any = None
        self._glib_thread: threading.Thread | None = None
        self.connect_duration_s: float = 0.0

    # -- lifecycle --------------------------------------------------------

    def connect(self) -> None:
        if self._streams:
            raise RuntimeError("connect() called twice without disconnect()")
        Gst, GLib = _gst_module()
        self._Gst, self._GLib = Gst, GLib
        self.warmup_dir.mkdir(parents=True, exist_ok=True)
        # The signal-watch bus delivers messages on the default GLib main
        # context, so we need a MainLoop running on a dedicated thread for
        # the callbacks to actually fire.
        self._glib_loop = GLib.MainLoop.new(None, False)
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run, name="pcs-glib-mainloop", daemon=True,
        )
        self._glib_thread.start()

        t0 = time.time()
        for cfg in self._stream_cfgs:
            stream = _Stream(cfg, self.warmup_dir, self)
            stream.build(Gst)
            self._streams[cfg.sid] = stream
        # Stagger only the PLAYING transition (where Argus actually does
        # buffer allocation) — building the elements is cheap.
        for i, sid in enumerate(sorted(self._streams)):
            if i > 0 and self.spawn_stagger_s > 0:
                time.sleep(self.spawn_stagger_s)
            self._streams[sid].start(Gst)
            logger.info("[%s] PLAYING (+%.2fs)", self._streams[sid].cfg.name, time.time() - t0)
        # Brief warm-up so the encoders are actually producing data before
        # the first start_episode().
        time.sleep(0.5)
        self.connect_duration_s = time.time() - t0
        logger.info(
            "connected %d streams in %.2fs (stagger=%.2fs)",
            len(self._streams), self.connect_duration_s, self.spawn_stagger_s,
        )

    def disconnect(self) -> None:
        Gst = self._Gst
        if Gst is None:
            return
        for stream in self._streams.values():
            try:
                stream.stop(Gst)
            except Exception as exc:
                logger.warning("[%s] stop failed: %s", stream.cfg.name, exc)
        self._streams.clear()
        if self._glib_loop is not None and self._glib_loop.is_running():
            self._glib_loop.quit()
        if self._glib_thread is not None:
            self._glib_thread.join(timeout=2.0)
        self._glib_loop = None
        self._glib_thread = None

    # -- episode control -------------------------------------------------

    def start_episode(self, episode_dir: Path, idx: int) -> EpisodeHandle:
        if not self._streams:
            raise RuntimeError("start_episode() before connect()")
        episode_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            t0_wall = time.time()
            t0_mono = time.monotonic()
            for stream in self._streams.values():
                stream.state = FragmentState.EPISODE
                stream.current_episode_dir = episode_dir
            for stream in self._streams.values():
                stream.force_split()
            handle = EpisodeHandle(
                idx=idx, directory=episode_dir,
                t0_wall_s=t0_wall, t0_mono_s=t0_mono,
            )
        return handle

    def stop_episode(self, handle: EpisodeHandle) -> EpisodeHandle:
        with self._lock:
            handle.stop_wall_s = time.time()
            for stream in self._streams.values():
                stream.state = FragmentState.WARMUP
                stream.current_episode_dir = None
            for stream in self._streams.values():
                stream.force_split()
        # Wait for async-finalize to actually flush the just-closed EPISODE
        # fragments to disk. splitmuxsink doesn't expose a finalize-done
        # signal in the GStreamer 1.20 line, so we sleep a small grace
        # window. Real fix lands when we add a pad probe for EOS on the
        # muxer's src pad in PR2.
        time.sleep(self.finalize_grace_s)
        # Now snapshot each stream's last EPISODE fragment into the handle.
        for stream in self._streams.values():
            if stream.last_episode_fragment is not None:
                handle.fragments[stream.cfg.name] = stream.last_episode_fragment
                stream.last_episode_fragment = None
        return handle

    def discard_episode(self, handle: EpisodeHandle) -> None:
        # `stop_episode` already closed the EPISODE fragment; just delete
        # the files it produced.
        for info in handle.fragments.values():
            try:
                info.path.unlink(missing_ok=True)
                logger.info("discarded fragment %s", info.path)
            except OSError as exc:
                logger.warning("failed to unlink %s: %s", info.path, exc)
        # If discard is called *while still in EPISODE state*, flip back to
        # WARMUP and split first.
        with self._lock:
            need_split = any(
                s.state == FragmentState.EPISODE for s in self._streams.values()
            )
            if need_split:
                for stream in self._streams.values():
                    stream.state = FragmentState.WARMUP
                    stream.current_episode_dir = None
                    stream.force_split()
        if need_split:
            time.sleep(self.finalize_grace_s)
            for stream in self._streams.values():
                if stream.last_episode_fragment is not None:
                    try:
                        stream.last_episode_fragment.path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    stream.last_episode_fragment = None

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
        """NULL -> PLAYING the pipeline for one stream.

        Useful when ``poll_errors()`` flags that stream as failed: drops the
        Argus session, rebuilds the pipeline (cheap; the elements are already
        constructed), and brings it back to PLAYING. Returns True on success.
        """
        Gst = self._Gst
        if Gst is None or sid not in self._streams:
            return False
        stream = self._streams[sid]
        try:
            stream.pipeline.set_state(Gst.State.NULL)
            stream.pipeline.get_state(timeout=Gst.SECOND)
            ret = stream.pipeline.set_state(Gst.State.PLAYING)
            ok = ret != Gst.StateChangeReturn.FAILURE
            logger.info("restart_stream sid=%s -> %s", sid, "ok" if ok else "FAIL")
            return ok
        except Exception as exc:
            logger.warning("restart_stream sid=%s raised: %s", sid, exc)
            return False

    # -- diagnostics -----------------------------------------------------

    def poll_errors(self) -> list[StreamError]:
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        return errors

    def _record_error(self, err: StreamError) -> None:
        with self._lock:
            self._errors.append(err)

    def _on_fragment_opened(self, stream: _Stream, info: FragmentInfo) -> None:
        if self._on_fragment_opened_cb is not None:
            try:
                self._on_fragment_opened_cb(stream, info)
            except Exception as exc:
                logger.warning("on_fragment_opened callback failed: %s", exc)

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
