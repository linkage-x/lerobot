"""Thor data-collection orchestrator: 11 x GMSL2 cameras + BOX 采集板.

This is the recorder script spawned by ``tools/data_collection_gui/gateway.py``
when the operator presses *Connect* on the LeRobot data-collection GUI. It
is a thin shell around two things that already exist in the repo:

  * ``tools/thor/gmsl2/gmsl2_record.py`` — drives 11 hardware-encoded GStreamer
    pipelines; pixel data never enters Python.
  * ``tools/thor/box_sdk/box_client.py`` — wraps the vendored ``box_sdk`` wheel
    over UDP/15000 to read the gripper / IMU / trigger / 6D force / two
    Paxini touch pads from the BOX MCU.

The protocol on stdout / stdin matches what ``gateway._apply_recorder_output``
already understands so the existing GUI plumbing keeps working:

  * ``Dataset root: <path>``                       → gateway shows path
  * ``Cameras: cam_00, cam_03, ...``               → marks each camera id
  * ``Box devices: box_gripper, box_imu, ...``     → marks each box sensor id
  * ``Episode <N> ready``                          → arms the GUI
  * ``Recorded <K> frames ...``                    → frame progress
  * ``Episode saved. Total saved episodes: N/<budget>``
  * ``Episode discarded`` / ``Recording stopped``

stdin commands accepted (the gateway writes lines to our stdin):

  * ``\\n`` (empty line)   → start an episode
  * ``y`` or ``yes``       → save the current / just-finished episode
  * ``n`` or ``no``        → discard
  * ``q`` or ``quit``      → exit the program cleanly

The recorder always runs in *streaming* mode — it never returns frames to
Python. Frame counts shown to the operator are derived from elapsed time *
configured fps; the authoritative timestamps live inside each MKV PTS
stream and the ``meta.json`` sidecar we write per episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python tools/thor/gmsl2/thor_record.py` invocation without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import gmsl2_record as gr  # noqa: E402
from tools.thor.gmsl2 import persistent_session as ps  # noqa: E402
from tools.thor.gmsl2 import thor_lerobot_v3 as lr3  # noqa: E402
from tools.thor.box_sdk import box_client as bc  # noqa: E402

logger = logging.getLogger("thor_record")


_STDIN_HINT = (
    "stdin commands: <Enter>=start episode, y=save, n=discard, q=quit"
)


@dataclass
class StdinCommand:
    kind: str  # "start" | "save" | "discard" | "quit"
    raw: str


def _emit(text: str) -> None:
    """Single point for protocol lines so flushing is consistent."""
    print(text, flush=True)


def _read_stdin_loop(queue: list[StdinCommand], stop: threading.Event) -> None:
    while not stop.is_set():
        line = sys.stdin.readline()
        if line == "":  # parent closed our stdin
            queue.append(StdinCommand(kind="quit", raw=""))
            return
        stripped = line.strip().lower()
        if stripped == "":
            queue.append(StdinCommand(kind="start", raw=line))
        elif stripped in ("y", "yes", "save"):
            queue.append(StdinCommand(kind="save", raw=line))
        elif stripped in ("n", "no", "discard"):
            queue.append(StdinCommand(kind="discard", raw=line))
        elif stripped in ("q", "quit", "exit"):
            queue.append(StdinCommand(kind="quit", raw=line))
            return
        else:
            logger.warning("ignoring unrecognized stdin command: %r", line)


def _wait_for_command(queue: list[StdinCommand], stop: threading.Event,
                      accept: tuple[str, ...]) -> StdinCommand:
    """Block until the operator queues one of the accepted commands."""
    while not stop.is_set():
        if queue:
            cmd = queue.pop(0)
            if cmd.kind in accept:
                return cmd
            logger.info("ignoring %s while waiting for %s", cmd.kind, accept)
        else:
            time.sleep(0.05)
    return StdinCommand(kind="quit", raw="")


def _drain_until(queue: list[StdinCommand], accept: tuple[str, ...],
                 stop: threading.Event, deadline_s: float) -> StdinCommand | None:
    """Non-blocking poll for an accepted command, with a deadline.

    Non-accepted commands (e.g. ``start`` queued while we're waiting for
    save/discard) are left in the queue so the next main-loop iteration
    can pick them up — otherwise a GUI that emits "Enter Enter" to
    "save current + start next" would lose the second Enter.
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end and not stop.is_set():
        if queue:
            if queue[0].kind in accept:
                return queue.pop(0)
            # First queued cmd is something else (operator moved on); fall
            # through to the caller's default behavior.
            return None
        time.sleep(0.05)
    return None


# ----------------------------------------------------------------- meta ---


def _pts_offset_from_handle(handle: ps.EpisodeHandle) -> float | None:
    """Per-camera (wall_s - split_now_wall_s), averaged across streams.

    This replaces the legacy ffprobe-based first-PTS extraction. PR1 burn-in
    on Thor showed that splitmuxsink's `format-location-full` callback's
    `first_sample.pts` is an unreliable cross-stream anchor (pipeline
    clocks are per-stream, so `first_pts_s` varies by up to 10s across
    cameras even when they actually started ~20ms apart). The real
    cross-stream clock is host wall time, captured into FragmentInfo at
    callback time.

    We return one scalar (the mean per-camera delay between split-now and
    the first sample actually opening on disk) so callers that previously
    passed ``pts_offset_s`` to ``thor_lerobot_v3.write_box_lerobot_v3_episode``
    keep working unchanged. The full per-camera breakdown is preserved in
    meta.json under ``sync_reference.camera_first_wall_s``.
    """
    if not handle.fragments:
        return None
    deltas = [
        info.first_wall_s - handle.t0_wall_s
        for info in handle.fragments.values()
        if info.first_wall_s > 0
    ]
    if not deltas:
        return None
    avg = sum(deltas) / len(deltas)
    logger.info(
        "pts_offset (avg first_wall - split_now across %d cams): %.4fs",
        len(deltas), avg,
    )
    return avg


def _write_sensor_samples(
    ep_dir: Path,
    samples: dict[str, list[bc.SensorSample]],
    t0_wall_s: float,
) -> Path:
    """Write high-frequency per-sensor samples to box_sensors.jsonl."""
    all_samples = bc.BoxClient.serialize_recorded_samples(samples, t0_wall_s)
    path = ep_dir / "box_sensors.jsonl"
    with open(path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    per_sensor = {}
    for s in all_samples:
        sid = s["sid"]
        per_sensor[sid] = per_sensor.get(sid, 0) + 1
    logger.info("wrote %d sensor samples to %s (%s)", len(all_samples), path, per_sensor)
    return path


def _write_episode_meta(
    handle: ps.EpisodeHandle,
    cfg: gr.RecorderConfig,
    locked: list[int],
    argus_failed: list[int],
    box_cfg: bc.BoxClientConfig,
    box_snapshots: list[dict[str, Any]],
    stop_reason: str,
    wallclock_start_utc: str,
    wallclock_end_utc: str,
) -> Path:
    """Write per-episode meta.json under the persistent-pipeline model.

    Mirrors the schema produced by gr.write_episode_meta (so downstream
    consumers don't break), but the ``sync_reference`` block is the new
    PR2 model:

      * ``split_now_wall_s``: host wall time when start_episode() emitted
        split-now on every splitmuxsink
      * ``camera_first_wall_s``: per-camera wall time when the new fragment
        actually opened (from format-location-full callback). This is the
        anchor downstream consumers should align BOX/touch samples to.
      * ``camera_first_pts_s``: per-camera buffer PTS of the first frame in
        the new fragment. Useful for single-stream analysis but NOT
        comparable across cameras (pipeline clocks are independent).

    The legacy ``camera_spawn_wall_s`` / ``camera_spawn_offset_s`` fields
    are gone; the closest replacement is ``camera_first_wall_s``.
    """
    fragments = handle.fragments
    duration_s = max(0.0, handle.stop_wall_s - handle.t0_wall_s)
    meta = {
        "episode_index": handle.idx,
        "repo_id": cfg.repo_id,
        "single_task": cfg.single_task,
        "wallclock_start_utc": wallclock_start_utc,
        "wallclock_end_utc": wallclock_end_utc,
        "duration_s": duration_s,
        "recording_stop_reason": stop_reason,
        "video": {
            "fps": cfg.cameras.fps,
            "width": cfg.cameras.width,
            "height": cfg.cameras.height,
            "codec": cfg.cameras.codec,
            "container": cfg.cameras.container,
            "bitrate_kbps": cfg.cameras.bitrate_kbps,
            "iframe_interval": cfg.cameras.iframe_interval,
            "idr_interval": cfg.cameras.iframe_interval,
            "preset_level": cfg.cameras.preset_level,
            "control_rate": cfg.cameras.control_rate,
            "color_format": "NV12 (YUV420)",
            "replay_warmup_s": float(cfg.cameras.replay_warmup_s),
            "pipeline": "nvarguscamerasrc | nvv4l2h{265,264}enc | splitmuxsink (persistent)",
        },
        "hardware_sync": {
            "enabled": cfg.hardware_sync.enabled,
            "pwm_chip": cfg.hardware_sync.pwm_chip,
            "pwm_id": cfg.hardware_sync.pwm_id,
            "pwm_fps": cfg.hardware_sync.fps,
            "trig_pin": f"0x{cfg.hardware_sync.trig_pin:08x}",
        },
        "sync_reference": {
            "t0_wall_s": handle.t0_wall_s,
            "t0_mono_s": handle.t0_mono_s,
            "split_now_wall_s": handle.t0_wall_s,
            "camera_first_wall_s": {
                name: info.first_wall_s for name, info in fragments.items()
            },
            "camera_first_pts_s": {
                name: info.first_pts_s for name, info in fragments.items()
            },
            "note": (
                "Persistent-pipeline model (PR2). split_now_wall_s is host "
                "time when start_episode() emitted split-now. "
                "camera_first_wall_s is the host time each splitmuxsink "
                "actually opened its new fragment — use this as the "
                "cross-camera alignment anchor (~20ms spread in PR1 "
                "burn-in). camera_first_pts_s is per-stream buffer PTS "
                "and is NOT cross-camera comparable. BOX snapshots carry "
                "t_relative_s = time.time() - split_now_wall_s."
            ),
        },
        "max96726_locked_sids": locked,
        "argus_failed_sids": argus_failed,
        "spawn_stagger_s": cfg.spawn_stagger_s,
        "stop_on_stream_exit": cfg.stop_on_stream_exit,
        "cameras": [
            {
                "sensor_id": info.sid,
                "name": name,
                "file": info.path.name,
                "fragment_id": info.fragment_id,
                "first_pts_s": info.first_pts_s,
                "first_wall_s": info.first_wall_s,
            }
            for name, info in fragments.items()
        ],
    }
    if box_cfg.enabled:
        meta["box_collection"] = {
            "config": asdict(box_cfg),
            "snapshots": box_snapshots,
        }
    meta_path = handle.directory / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def _stream_configs(usable: list[int], cfg: gr.RecorderConfig) -> list[ps.StreamConfig]:
    """Translate RecorderConfig + sids -> per-stream PersistentCameraSession config."""
    return [
        ps.StreamConfig(
            sid=sid,
            name=f"{cfg.name_prefix}_{sid:02d}",
            width=cfg.cameras.width,
            height=cfg.cameras.height,
            fps=cfg.cameras.fps,
            codec=cfg.cameras.codec,
            bitrate_kbps=cfg.cameras.bitrate_kbps,
            iframe_interval=cfg.cameras.iframe_interval,
            preset_level=cfg.cameras.preset_level,
            control_rate=cfg.cameras.control_rate,
            sensor_mode=cfg.cameras.sensor_mode,
            exposure_us=cfg.cameras.exposure_us,
            gain=cfg.cameras.gain,
        )
        for sid in usable
    ]


# ----------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Thor GMSL2 + BOX recorder for the GUI gateway")
    ap.add_argument("--config-path", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--skip-hardware-sync", action="store_true",
                    help="don't touch the PWM/v4l2 trig path -- useful for "
                         "bring-up on dev hosts without the SG16A")
    ap.add_argument("--skip-argus-probe", action="store_true",
                    help="trust the MAX96726 lock list verbatim")
    ap.add_argument("--no-box", action="store_true",
                    help="ignore the YAML box_collection block (camera-only)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    cfg = gr.load_config(args.config_path)
    raw_yaml: dict[str, Any] = {}
    with args.config_path.open() as f:
        import yaml
        raw_yaml = yaml.safe_load(f) or {}
    box_cfg = bc.from_yaml_dict(raw_yaml.get("box_collection") if not args.no_box else None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.dataset_root = cfg.dataset_root.parent / f"{cfg.dataset_root.name}_{stamp}"

    repo_root = args.repo_root.resolve()
    if args.skip_hardware_sync:
        cfg.hardware_sync.enabled = False
    gr.maybe_setup_sync(cfg, repo_root)

    _emit(f"Dataset root: {cfg.dataset_root}")

    _emit("Connecting: detecting MAX96726 locked cameras...")
    locked = gr.detect_locked_sids(cfg, repo_root)
    logger.info("MAX96726 locked sids: %s", locked)
    if not locked:
        _emit("ERROR: no locked GMSL2 cameras detected")
        return 1
    _emit(f"Connecting: {len(locked)} cameras locked, probing Argus ISP...")

    if args.skip_argus_probe:
        usable = list(locked)
        argus_failed: list[int] = []
    else:
        usable, argus_failed = gr.probe_argus(
            locked, cfg.cameras.width, cfg.cameras.height, cfg.cameras.fps,
        )
    if not usable:
        _emit("ERROR: no cameras stream through nvargus")
        return 1
    _emit(f"Connecting: {len(usable)}/{len(locked)} cameras verified")

    camera_ids = [f"{cfg.name_prefix}_{sid:02d}" for sid in usable]
    _emit(f"Cameras: {', '.join(camera_ids)}")

    if cfg.hardware_sync.enabled:
        gr._clamp_exposure_for_pwm_period(cfg)
        gr.apply_per_camera_controls(
            usable, cfg.cameras,
            trig_mode=cfg.hardware_sync.sensor_trig_mode,
            trig_pin=cfg.hardware_sync.trig_pin,
        )

    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    warmup_dir = cfg.dataset_root / "_warmup"

    # Start BOX SDK BEFORE GStreamer pipelines: BOX SDK loads its own
    # native .so + UDP listener; starting it while 11 nvarguscamerasrc
    # pipelines and a GLib MainLoop are already running triggers
    # SIGABRT on the first incoming UDP packet (see
    # tools/data_collection_gui/docs/development_status.md).
    box = bc.BoxClient(box_cfg)
    box_started = box.start() if box_cfg.enabled else False
    if box_started:
        # Wait for the rate-estimation window (2s) to fill so we can report
        # real per-sensor frequencies instead of the poll rate.
        time.sleep(2.5)
        present = box.connected_devices()
        _emit(f"Box devices: {', '.join(present) if present else '(none)'}")
        rates = box.observed_rates()
        if any(rates.values()):
            parts = [f"{sid}={hz:.0f}" for sid, hz in rates.items() if hz > 0]
            _emit(f"Box rates: {', '.join(parts)}")
    elif box_cfg.enabled:
        _emit("Box devices: (none)")
        logger.warning("box_collection enabled but BoxClient.start() returned False")

    # Persistent GStreamer pipelines: spawn the N nvarguscamerasrc
    # pipelines once here, then slice on demand inside the loop. This
    # replaces the per-episode `gr.EpisodeSession(...)` model that paid
    # ~stagger * N seconds of warmup before every StartEpisode.
    pcs = ps.PersistentCameraSession(
        _stream_configs(usable, cfg),
        warmup_dir,
        spawn_stagger_s=cfg.spawn_stagger_s,
    )
    _emit(f"Connecting: spawning {len(usable)} persistent pipelines...")
    pcs.connect()
    _emit(f"Connected {len(usable)} pipelines in {pcs.connect_duration_s:.1f}s")

    lr3_writer: lr3.Lr3Writer | None = None
    if box_cfg.enabled:
        try:
            lr3_writer = lr3.open_box_lerobot_v3_writer(
                cfg.dataset_root,
                repo_id=cfg.repo_id,
                task=cfg.single_task,
                fps=cfg.fps,
            )
            if lr3_writer is None:
                logger.warning("BOX LeRobot v3 writer disabled; pyarrow is unavailable")
        except Exception as exc:
            logger.warning("failed to open BOX LeRobot v3 writer: %s", exc)

    # Stdin reader thread -- gateway writes single lines per command.
    stop_event = threading.Event()
    cmd_queue: list[StdinCommand] = []
    stdin_thread = threading.Thread(
        target=_read_stdin_loop, args=(cmd_queue, stop_event),
        daemon=True, name="thor-record-stdin",
    )
    stdin_thread.start()

    def _sigint(_sig, _frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    saved = 0
    ep_idx = gr._next_episode_index(cfg.dataset_root)
    budget = cfg.num_episodes if cfg.num_episodes > 0 else None
    budget_str = str(budget) if budget else "unlimited"
    logger.info(_STDIN_HINT)
    _emit(f"Episode {ep_idx} ready")

    rc = 0
    try:
        while not stop_event.is_set():
            cmd = _wait_for_command(cmd_queue, stop_event, accept=("start", "quit"))
            if cmd.kind == "quit":
                break

            ep_dir = cfg.dataset_root / "episodes" / f"episode_{ep_idx:06d}"
            wall_start = datetime.now(timezone.utc).isoformat()
            t0_split_start_mono = time.monotonic()
            handle = pcs.start_episode(ep_dir, ep_idx)
            t_start = handle.t0_wall_s
            split_emit_ms = (time.monotonic() - t0_split_start_mono) * 1000
            logger.info(
                "episode %d started @ %s -> %s (split-now emit %.1fms)",
                ep_idx, wall_start, ep_dir, split_emit_ms,
            )
            if box_started:
                box.start_recording(t_start)
            box_snapshots: list[dict[str, Any]] = []
            target_s = cfg.episode_time_s if cfg.episode_time_s > 0 else float("inf")
            last_progress_at = 0.0
            box_sample_at = 0.0
            stop_episode = False
            stop_reason = "operator_save"
            while True:
                now = time.monotonic()
                elapsed = time.time() - t_start
                if elapsed >= target_s:
                    stop_reason = "duration_reached"
                    break
                if cmd_queue:
                    nxt = cmd_queue[0]
                    if nxt.kind in ("save", "discard", "quit"):
                        cmd_queue.pop(0)
                        stop_reason = nxt.kind
                        if nxt.kind == "quit":
                            stop_episode = True
                            stop_event.set()
                        break
                # Persistent-pipeline equivalent of "stream exited early":
                # GLib bus dispatches ERROR messages into pcs._errors via the
                # signal-watch on its MainLoop thread. PR1 burn-in measured
                # dispatch at ~0.14ms so this poll loop sees them within
                # ~50ms of the actual error.
                stream_errs = pcs.poll_errors()
                if stream_errs and cfg.stop_on_stream_exit:
                    if box_started:
                        snap = box.read()
                        snap["t_relative_s"] = elapsed
                        box_snapshots.append(snap)
                    details = ", ".join(
                        f"{e.name}({(e.message or '')[:60]})" for e in stream_errs
                    )
                    logger.warning("streams reported errors: %s", details)
                    _emit(f"Stream exited early: {details}")
                    stop_reason = "stream_exit"
                    break
                if now - last_progress_at > 0.5:
                    approx_frames = int(elapsed * cfg.cameras.fps)
                    _emit(f"Recorded {approx_frames} frames for the current episode.")
                    last_progress_at = now
                if box_started and now - box_sample_at > 0.5:
                    snap = box.read()
                    snap["t_relative_s"] = elapsed
                    box_snapshots.append(snap)
                    box_sample_at = now
                time.sleep(0.05)

            capture_end_wall_s = time.time()
            capture_end_mono_s = time.monotonic()
            wall_end = datetime.now(timezone.utc).isoformat()
            duration_s = capture_end_wall_s - t_start
            if box_started:
                recorded_samples = box.stop_recording()
            else:
                recorded_samples = {}

            pcs.stop_episode(handle)
            cleanup_duration_s = max(0.0, time.monotonic() - capture_end_mono_s)
            pts_offset = _pts_offset_from_handle(handle)

            decision = stop_reason
            if decision == "duration_reached":
                # Auto-save when the episode wall clock ran out unless the operator
                # discards explicitly within a short window.
                decision_cmd = _drain_until(
                    cmd_queue,
                    accept=("save", "discard", "quit"),
                    stop=stop_event,
                    deadline_s=0.2,
                )
                decision = (decision_cmd.kind if decision_cmd else "save")

            if decision in ("save", "operator_save", "stream_exit"):
                meta_path = _write_episode_meta(
                    handle, cfg, locked, argus_failed, box_cfg, box_snapshots,
                    decision, wall_start, wall_end,
                )
                try:
                    payload = json.loads(meta_path.read_text())
                    payload["cleanup_duration_s"] = cleanup_duration_s
                    payload["split_emit_ms"] = split_emit_ms
                    meta_path.write_text(json.dumps(payload, indent=2))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("failed to annotate cleanup duration: %s", exc)
                if recorded_samples:
                    _write_sensor_samples(ep_dir, recorded_samples, t_start)
                sensor_data = {
                    sid: [{"t_rel_s": s.wall_time_s - t_start,
                           "wall_s": s.wall_time_s,
                           "data": s.data}
                          for s in slist]
                    for sid, slist in recorded_samples.items()
                } if recorded_samples else None
                try:
                    if lr3_writer is not None:
                        v3_path = lr3_writer.append_episode(
                            episode_index=ep_idx,
                            snapshots=box_snapshots,
                            duration_s=duration_s,
                            sensor_samples=sensor_data,
                            t0_wall_s=t_start,
                            pts_offset_s=pts_offset,
                        )
                        if v3_path is not None:
                            logger.info("wrote BOX LeRobot v3 rows: %s", v3_path)
                    elif box_snapshots or sensor_data:
                        logger.warning("BOX LeRobot v3 rows skipped; writer is unavailable")
                except Exception as exc:
                    logger.warning("failed to write BOX LeRobot v3 rows: %s", exc)
                saved += 1
                if decision == "stream_exit":
                    _emit("Episode saved with stream exits.")
                else:
                    _emit("Episode saved.")
                _emit(f"Total saved episodes: {saved}/{budget_str}")
                ep_idx += 1
            else:
                # Discard: delete the per-camera .mkv fragments the
                # splitmuxsink just closed, then drop the (now-empty)
                # episode directory so it doesn't show up in dataset scans.
                pcs.discard_episode(handle)
                try:
                    if ep_dir.is_dir():
                        # Remove ep_dir only if nothing else slipped in.
                        leftovers = list(ep_dir.iterdir())
                        if not leftovers:
                            ep_dir.rmdir()
                        else:
                            import shutil
                            shutil.rmtree(ep_dir, ignore_errors=True)
                except OSError as exc:
                    logger.warning("failed to clean discarded episode dir: %s", exc)
                _emit("Episode discarded")

            # Keep the warmup directory bounded across long sessions.
            pcs.cleanup_warmup_files(keep_last_n=3)

            if stop_episode or stop_event.is_set():
                break
            if budget is not None and saved >= budget:
                break
            _emit(f"Episode {ep_idx} ready")
    finally:
        stop_event.set()
        if lr3_writer is not None:
            try:
                lr3_writer.finalize()
            except Exception as exc:
                logger.warning("failed to finalize BOX LeRobot v3 writer: %s", exc)
        try:
            pcs.disconnect()
        except Exception as exc:
            logger.warning("pcs.disconnect on shutdown: %s", exc)
        try:
            box.stop()
        except Exception as exc:
            logger.warning("box.stop on shutdown: %s", exc)
        _emit("Recording stopped")
    return rc


if __name__ == "__main__":
    sys.exit(main())
