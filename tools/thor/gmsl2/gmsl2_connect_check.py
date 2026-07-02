#!/usr/bin/env python3
"""GMSL2 multi-camera "Connect" reproducer — standalone, for vendor debugging.

供应商参考用：把数据采集 GUI 里 "Connect" 按钮做的事单文件复现，方便在
GMSL2 摄像头 / serializer-deserializer / 驱动侧排查 "部分相机拉不起来 /
拉起后立刻掉" 的问题。本脚本 **不依赖** lerobot 仓库、box_sdk、gateway，
只需要 Jetson/Thor 上已有的 GStreamer + python3-gi + nvarguscamerasrc。

What it does
------------
Connect spawns one ``nvarguscamerasrc`` capture pipeline **per camera, each in
its own subprocess** (so nvargus-daemon sees N independent RPC clients), brings
them all to PLAYING, and reports which cameras came up and which failed with
their underlying Argus/NVMM error. This script reproduces exactly that, with
the same pipeline the production recorder uses:

    nvarguscamerasrc sensor-id=N sensor-mode=M do-timestamp=true
      ! video/x-raw(memory:NVMM),format=NV12,width=W,height=H,framerate=F/1
      ! nvv4l2h265enc bitrate=.. iframeinterval=.. idrinterval=.. \
            preset-level=1 control-rate=1 insert-sps-pps=1
      ! h265parse
      ! fakesink              (or splitmuxsink with --record DIR)

Two bring-up strategies (both used in production):
  * serial     : spawn one camera, wait until PLAYING, hold a stability
                 window, then the next. Avoids overlapping Argus/NVMM bring-ups.
  * two-phase  : spawn all cameras to PAUSED first (PAUSED opens no Argus
                 session — it is a no-op for a live source), then serialize
                 only the PAUSED->PLAYING transition.

Why this is useful to the camera/driver vendor
----------------------------------------------
On this rig, after a clean ``nvargus-daemon`` restart, some cameras still fail
to stream on with kernel errors such as::

    ar0234c 20-0023: i2c write failed, 0x3060 = 00
    ar0234c 20-0023: Error turning on streaming
    (Argus) Sensor GUID NN is in error state / waitForIdle() timed out
    NvBufSurfaceFromFd Failed / dmabuf_fd -1

Those originate below GStreamer (sensor / CSI / serializer-deserializer /
RCE / driver), not in application code. This script isolates the camera
bring-up so the failure can be reproduced and correlated with kernel and
nvargus-daemon logs.

Usage
-----
    # All 11 cameras on this rig, default serial bring-up:
    python3 gmsl2_connect_check.py --sids 0,2,3,4,5,7,9,10,11,14,15

    # Two-phase bring-up (overlaps per-process startup, same PLAYING ordering):
    python3 gmsl2_connect_check.py --sids 0,2,3,4,5,7,9,10,11,14,15 --two-phase

    # Cheapest repro of just the sensor stream-on (no encoder / NVMM pressure):
    python3 gmsl2_connect_check.py --sids 0,2,3 --pipeline argus-only

    # Probe a single suspect camera in a loop to see if it is intermittent:
    for i in $(seq 1 10); do python3 gmsl2_connect_check.py --sids 10; done

Recommended logs to capture alongside a failing run (send to vendor):
    journalctl -u nvargus-daemon --since "<connect start time>"
    journalctl -k --since "<connect start time>" | \
        grep -E "ar0234|max96|nvcsi|vi-output|tegra-vi|i2c|streaming|timeout"

This script writes nothing to disk unless ``--record DIR`` is given, and it
always tears the pipelines down on exit (so it does not leak Argus sessions).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as queue_module
import sys
import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pipeline construction (kept identical to the production recorder)
# ---------------------------------------------------------------------------


def build_pipeline_desc(
    *,
    sid: int,
    sensor_mode: int,
    width: int,
    height: int,
    fps: int,
    codec: str,
    bitrate_kbps: int,
    iframe_interval: int,
    pipeline_kind: str,
    record_dir: str | None,
) -> str:
    """Render the gst-launch description for one camera's pipeline."""
    source = (
        f"nvarguscamerasrc sensor-id={sid} sensor-mode={sensor_mode} "
        f"do-timestamp=true "
        f"! video/x-raw(memory:NVMM),format=NV12,width={width},"
        f"height={height},framerate={fps}/1"
    )

    if pipeline_kind == "argus-only":
        # Exercises only the sensor stream-on + NVMM source pool; no encoder.
        # Smallest reproducer for "Error turning on streaming" / i2c failures.
        return f"{source} ! fakesink sync=false"

    enc_factory = "nvv4l2h265enc" if codec == "h265" else "nvv4l2h264enc"
    parser = "h265parse" if codec == "h265" else "h264parse"
    encoder = (
        f"{enc_factory} bitrate={bitrate_kbps * 1000} "
        f"iframeinterval={iframe_interval} idrinterval={iframe_interval} "
        f"preset-level=1 control-rate=1 insert-sps-pps=1"
    )

    if record_dir:
        # Full production parity: actually write a per-camera MKV via
        # splitmuxsink (adds the NVMM->encoder->muxer load path).
        location = f"{record_dir}/cam_{sid:02d}_%05d.mkv"
        sink = (
            f"splitmuxsink name=mux_{sid} muxer-factory=matroskamux "
            f"async-finalize=true max-size-time=0 max-size-bytes=0 "
            f'location="{location}"'
        )
    else:
        # Same encoder/NVMM pressure as production, but discard the output.
        sink = "fakesink sync=false"

    return f"{source} ! {encoder} ! {parser} ! {sink}"


# ---------------------------------------------------------------------------
# Per-camera worker subprocess
# ---------------------------------------------------------------------------
#
# Event protocol (worker -> parent), all tagged with sid:
#   ("paused", sid)
#   ("playing", sid)
#   ("error", sid, message, debug)
#   ("eos", sid)
# Command protocol (parent -> worker):
#   ("play",)   -- two-phase only: release PAUSED -> PLAYING
#   ("stop",)   -- tear down to NULL and exit


def run_worker(
    sid: int,
    desc: str,
    ready_timeout_s: float,
    two_phase: bool,
    cmd_q,
    evt_q,
) -> int:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst  # type: ignore

    Gst.init(None)

    try:
        pipeline = Gst.parse_launch(desc)
    except GLib.Error as exc:  # malformed pipeline / missing element
        evt_q.put(("error", sid, f"parse_launch failed: {exc}", ""))
        return 2

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            evt_q.put(("error", sid, str(err), str(debug or "")))
        elif t == Gst.MessageType.EOS:
            # On nvarguscamerasrc, an unexpected EOS means the source stopped
            # delivering buffers (sensor dropped out) — report it as a fault.
            evt_q.put(("eos", sid))

    bus.connect("message", on_bus)

    loop = GLib.MainLoop.new(None, False)
    threading.Thread(target=loop.run, name=f"glib-{sid}", daemon=True).start()

    def _bring_to(state, label) -> bool:
        ret = pipeline.set_state(state)
        if ret == Gst.StateChangeReturn.FAILURE:
            evt_q.put(("error", sid, f"set_state({label}) returned FAILURE", ""))
            return False
        change, gst_state, _pending = pipeline.get_state(
            timeout=int(ready_timeout_s * Gst.SECOND)
        )
        if change == Gst.StateChangeReturn.FAILURE or gst_state != state:
            nick = getattr(gst_state, "value_nick", str(gst_state))
            evt_q.put((
                "error", sid,
                f"did not reach {label} within {ready_timeout_s:.1f}s (stuck at {nick})",
                "",
            ))
            return False
        return True

    if two_phase:
        # PAUSED is a no-op for a live source (NO_PREROLL, no Argus session);
        # get_state returns PAUSED. Report it and wait to be released.
        if not _bring_to(Gst.State.PAUSED, "PAUSED"):
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
            return 4
        evt_q.put(("paused", sid))
        while True:
            cmd = cmd_q.get()
            if not isinstance(cmd, tuple) or not cmd:
                continue
            if cmd[0] == "play":
                break
            if cmd[0] == "stop":
                pipeline.set_state(Gst.State.NULL)
                loop.quit()
                return 0

    # PAUSED|READY -> PLAYING : this is where the Argus CaptureSession is
    # created and the NVMM surfaces are allocated (the part that fails).
    if not _bring_to(Gst.State.PLAYING, "PLAYING"):
        pipeline.set_state(Gst.State.NULL)
        loop.quit()
        return 4

    evt_q.put(("playing", sid))

    # Keep running (so the bus can still surface a post-PLAYING EOS/ERROR —
    # the "reaches PLAYING then immediately drops" pattern) until told to stop.
    while True:
        cmd = cmd_q.get()
        if isinstance(cmd, tuple) and cmd and cmd[0] == "stop":
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
            return 0


# ---------------------------------------------------------------------------
# Parent-side per-camera handle
# ---------------------------------------------------------------------------


@dataclass
class CamProc:
    sid: int
    cmd_q: object
    proc: object = None
    playing_evt: threading.Event = field(default_factory=threading.Event)
    paused_evt: threading.Event = field(default_factory=threading.Event)
    failed_evt: threading.Event = field(default_factory=threading.Event)
    play_wall_s: float = 0.0
    playing_wall_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def send(self, cmd: tuple) -> None:
        try:
            self.cmd_q.put(cmd)
        except Exception:
            pass


class ConnectCheck:
    def __init__(self, args):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.evt_q = self.ctx.Queue()
        self.cams: dict[int, CamProc] = {}
        self._stop_reader = threading.Event()
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._t0 = 0.0

    # -- event plumbing --------------------------------------------------

    def _read_events(self) -> None:
        while not self._stop_reader.is_set():
            try:
                evt = self.evt_q.get(timeout=0.2)
            except queue_module.Empty:
                continue
            except (EOFError, OSError):
                return
            self._dispatch(evt)

    def _dispatch(self, evt: tuple) -> None:
        kind, sid = evt[0], evt[1]
        cam = self.cams.get(sid)
        if cam is None:
            return
        now = time.monotonic() - self._t0
        if kind == "paused":
            cam.paused_evt.set()
        elif kind == "playing":
            cam.playing_wall_s = now
            cam.playing_evt.set()
            print(f"  [cam_{sid:02d}] PLAYING   (+{now:6.2f}s)", flush=True)
        elif kind == "error":
            msg = evt[2]
            debug = evt[3] if len(evt) > 3 else ""
            cam.errors.append(msg + (f" | {debug}" if debug else ""))
            cam.failed_evt.set()
            cam.paused_evt.set()  # unblock any pending PAUSED wait
            print(f"  [cam_{sid:02d}] ERROR     (+{now:6.2f}s) {msg}", flush=True)
            if debug:
                print(f"           debug: {debug}", flush=True)
        elif kind == "eos":
            cam.errors.append("bus EOS (source stopped delivering buffers)")
            cam.failed_evt.set()
            cam.paused_evt.set()
            print(f"  [cam_{sid:02d}] EOS       (+{now:6.2f}s) source stopped", flush=True)

    # -- spawning --------------------------------------------------------

    def _spawn(self, sid: int, two_phase: bool) -> None:
        cmd_q = self.ctx.Queue()
        desc = build_pipeline_desc(
            sid=sid,
            sensor_mode=self.args.sensor_mode,
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
            codec=self.args.codec,
            bitrate_kbps=self.args.bitrate_kbps,
            iframe_interval=self.args.iframe_interval,
            pipeline_kind=self.args.pipeline,
            record_dir=self.args.record,
        )
        cam = CamProc(sid=sid, cmd_q=cmd_q)
        proc = self.ctx.Process(
            target=run_worker,
            args=(sid, desc, self.args.ready_timeout_s, two_phase, cmd_q, self.evt_q),
            name=f"cam-{sid:02d}",
            daemon=True,
        )
        cam.proc = proc
        self.cams[sid] = cam
        proc.start()

    # -- connect strategies ---------------------------------------------

    def connect(self) -> None:
        self._t0 = time.monotonic()
        self._reader.start()
        sids = self.args.sids
        print(
            f"Connect: {len(sids)} cameras, pipeline={self.args.pipeline}, "
            f"mode={'two-phase' if self.args.two_phase else 'serial'}, "
            f"stagger={self.args.stagger_s}s stable={self.args.stable_s}s\n",
            flush=True,
        )
        if self.args.two_phase:
            self._connect_two_phase(sids)
        else:
            self._connect_serial(sids)

    def _connect_serial(self, sids: list[int]) -> None:
        for i, sid in enumerate(sids):
            if i > 0 and self.args.stagger_s > 0:
                time.sleep(self.args.stagger_s)
            self._spawn(sid, two_phase=False)
            self.cams[sid].play_wall_s = time.monotonic() - self._t0
            self._wait_one(sid)
            self._stable_window()

    def _connect_two_phase(self, sids: list[int]) -> None:
        # Phase 1: spawn all to PAUSED concurrently (no Argus, no contention).
        for sid in sids:
            self._spawn(sid, two_phase=True)
        n_paused = 0
        for sid in sids:
            if self.cams[sid].paused_evt.wait(self.args.ready_timeout_s):
                if not self.cams[sid].failed_evt.is_set():
                    n_paused += 1
        print(
            f"  Phase 1: {n_paused}/{len(sids)} reached PAUSED "
            f"(+{time.monotonic() - self._t0:.2f}s)\n",
            flush=True,
        )
        # Phase 2: serialize PAUSED->PLAYING.
        play_order = [s for s in sids if not self.cams[s].failed_evt.is_set()]
        for i, sid in enumerate(play_order):
            if i > 0 and self.args.stagger_s > 0:
                time.sleep(self.args.stagger_s)
            self.cams[sid].play_wall_s = time.monotonic() - self._t0
            self.cams[sid].send(("play",))
            self._wait_one(sid)
            self._stable_window()

    def _wait_one(self, sid: int) -> None:
        cam = self.cams[sid]
        deadline = time.monotonic() + self.args.ready_timeout_s
        while time.monotonic() < deadline:
            if cam.playing_evt.is_set() or cam.failed_evt.is_set():
                return
            time.sleep(0.02)
        if not cam.playing_evt.is_set():
            cam.errors.append(f"no PLAYING/ERROR within {self.args.ready_timeout_s:.1f}s")
            cam.failed_evt.set()

    def _stable_window(self) -> None:
        if self.args.stable_s > 0:
            time.sleep(self.args.stable_s)

    # -- hold + teardown -------------------------------------------------

    def hold(self) -> None:
        if self.args.hold_s <= 0:
            return
        print(
            f"\nHolding PLAYING cameras for {self.args.hold_s}s to catch "
            f"late EOS/ERROR (cameras that drop after reaching PLAYING)...",
            flush=True,
        )
        time.sleep(self.args.hold_s)

    def teardown(self) -> None:
        for cam in self.cams.values():
            cam.send(("stop",))
        for cam in self.cams.values():
            if cam.proc is None:
                continue
            cam.proc.join(timeout=3.0)
            if cam.proc.is_alive():
                cam.proc.terminate()
                cam.proc.join(timeout=1.0)
            if cam.proc.is_alive():
                cam.proc.kill()
                cam.proc.join(timeout=1.0)
        self._stop_reader.set()

    # -- summary ---------------------------------------------------------

    def summary(self) -> int:
        up = [s for s, c in self.cams.items() if c.playing_evt.is_set() and not c.failed_evt.is_set()]
        down = [s for s in self.cams if s not in up]
        print("\n" + "=" * 64)
        print(f"RESULT: {len(up)}/{len(self.cams)} cameras up")
        print(f"  PLAYING : {','.join(f'cam_{s:02d}' for s in sorted(up)) or '(none)'}")
        print(f"  FAILED  : {','.join(f'cam_{s:02d}' for s in sorted(down)) or '(none)'}")
        if down:
            print("\nFailure details:")
            for s in sorted(down):
                cam = self.cams[s]
                first = cam.errors[0] if cam.errors else "reached PLAYING then no error recorded"
                print(f"  cam_{s:02d}: {first}")
        # Per-camera bring-up timing (PLAYING latency from its play/spawn).
        print("\nBring-up timing (PLAYING latency per camera):")
        for s in sorted(self.cams):
            cam = self.cams[s]
            if cam.playing_evt.is_set():
                dt = cam.playing_wall_s - cam.play_wall_s
                print(f"  cam_{s:02d}: +{cam.playing_wall_s:6.2f}s (play->playing {dt:.2f}s)")
            else:
                print(f"  cam_{s:02d}: never reached PLAYING")
        print("=" * 64, flush=True)
        return 0 if up and not down else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Standalone GMSL2 'Connect' reproducer for vendor debugging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--sids", default="0,2,3,4,5,7,9,10,11,14,15",
        help="comma-separated camera sensor-ids to bring up",
    )
    ap.add_argument(
        "--pipeline", choices=["full", "argus-only"], default="full",
        help="'full' = nvarguscamerasrc+nvenc (production load); "
             "'argus-only' = nvarguscamerasrc+fakesink (sensor stream-on only)",
    )
    ap.add_argument("--two-phase", action="store_true",
                    help="spawn all to PAUSED first, then serialize PLAYING")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="write per-camera MKV via splitmuxsink into DIR "
                         "(full production parity; off by default = fakesink)")
    ap.add_argument("--sensor-mode", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--codec", choices=["h264", "h265"], default="h265")
    ap.add_argument("--bitrate-kbps", type=int, default=40000)
    ap.add_argument("--iframe-interval", type=int, default=1)
    ap.add_argument("--stagger-s", type=float, default=1.0,
                    help="delay between sequential camera bring-ups")
    ap.add_argument("--stable-s", type=float, default=2.0,
                    help="hold after each PLAYING to catch immediate EOS/TIMEOUT")
    ap.add_argument("--ready-timeout-s", type=float, default=12.0,
                    help="per-camera deadline to reach PAUSED / PLAYING")
    ap.add_argument("--hold-s", type=float, default=5.0,
                    help="after all bring-ups, keep cameras PLAYING this long "
                         "to catch ones that drop after connecting")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.sids = [int(s) for s in args.sids.split(",") if s.strip() != ""]
    if not args.sids:
        print("error: --sids parsed empty", file=sys.stderr)
        return 2
    if args.record:
        import os
        os.makedirs(args.record, exist_ok=True)

    try:
        import gi  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"error: python3-gi (GStreamer bindings) not available: {exc}", file=sys.stderr)
        return 2

    check = ConnectCheck(args)
    rc = 1
    try:
        check.connect()
        check.hold()
        rc = check.summary()
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        rc = 130
    finally:
        check.teardown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
