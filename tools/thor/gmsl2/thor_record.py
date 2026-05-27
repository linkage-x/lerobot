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
    """Non-blocking poll for an accepted command, with a deadline."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end and not stop.is_set():
        if queue:
            cmd = queue.pop(0)
            if cmd.kind in accept:
                return cmd
        time.sleep(0.05)
    return None


# ----------------------------------------------------------------- meta ---


def _extract_pts_offset(streams: list[gr.CameraStream]) -> float | None:
    """Extract the first-frame PTS from the earliest camera's MKV.

    All cameras share the same PWM trigger so the inter-frame interval is
    identical.  Only the absolute offset of the first frame matters — use the
    camera with the smallest spawn_stagger (earliest start).  Returns
    ``None`` when ffprobe is unavailable or the MKV is empty/corrupt.
    """
    candidates = sorted(streams, key=lambda s: s.started_at)
    for stream in candidates:
        if not stream.file.exists() or stream.file.stat().st_size < 1024:
            continue
        pts_list = lr3.extract_pts(stream.file)
        if pts_list:
            logger.info("PTS offset from %s: %.6fs (first of %d frames)",
                        stream.name, pts_list[0], len(pts_list))
            return pts_list[0]
    logger.warning("could not extract PTS from any camera MKV")
    return None


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
    ep: gr.EpisodeResult,
    cfg: gr.RecorderConfig,
    locked: list[int],
    argus_failed: list[int],
    box_cfg: bc.BoxClientConfig,
    box_snapshots: list[dict[str, Any]],
    stop_reason: str,
) -> Path:
    meta_path = gr.write_episode_meta(ep, cfg, locked, argus_failed)
    payload = json.loads(meta_path.read_text())
    payload["recording_stop_reason"] = stop_reason
    if box_cfg.enabled:
        # Augment the GMSL2 meta with box_collection info so downstream tools
        # have everything in one file.
        payload["box_collection"] = {
            "config": asdict(box_cfg),
            "snapshots": box_snapshots,
        }
    meta_path.write_text(json.dumps(payload, indent=2))
    return meta_path


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
            session = gr.EpisodeSession(usable, cfg)
            wall_start = datetime.now(timezone.utc).isoformat()
            t0_wall = time.time()
            t0_mono = time.monotonic()
            streams = session.start(ep_dir)
            t_start = time.time()
            logger.info("episode %d started @ %s -> %s", ep_idx, wall_start, ep_dir)
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
                dead = [s for s in streams if s.proc and s.proc.poll() is not None]
                if dead and cfg.stop_on_stream_exit:
                    if box_started:
                        snap = box.read()
                        snap["t_relative_s"] = elapsed
                        box_snapshots.append(snap)
                    details = ", ".join(
                        f"{s.name}(rc={s.proc.poll()}, log={s.log_file.name})" for s in dead
                    )
                    logger.warning("streams exited early: %s", details)
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

            session.stop(streams)
            recorded_samples = box.stop_recording() if box_started else {}
            pts_offset = _extract_pts_offset(streams)
            wall_end = datetime.now(timezone.utc).isoformat()
            duration_s = time.time() - t_start

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
                ep = gr.EpisodeResult(
                    index=ep_idx,
                    directory=ep_dir,
                    wallclock_start_utc=wall_start,
                    wallclock_end_utc=wall_end,
                    duration_s=duration_s,
                    streams=streams,
                    t0_wall_s=t0_wall,
                    t0_mono_s=t0_mono,
                )
                _write_episode_meta(
                    ep, cfg, locked, argus_failed, box_cfg, box_snapshots, decision,
                )
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
                    v3_path = lr3.write_box_lerobot_v3_episode(
                        cfg.dataset_root,
                        repo_id=cfg.repo_id,
                        task=cfg.single_task,
                        fps=cfg.fps,
                        episode_index=ep_idx,
                        snapshots=box_snapshots,
                        duration_s=duration_s,
                        sensor_samples=sensor_data,
                        t0_wall_s=t_start,
                        pts_offset_s=pts_offset,
                    )
                    if v3_path is not None:
                        logger.info("wrote BOX LeRobot v3 rows: %s", v3_path)
                    elif box_snapshots:
                        logger.warning("BOX LeRobot v3 rows skipped; pyarrow is unavailable")
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
                # Discard: nuke the episode directory we just produced so it
                # doesn't show up in dataset scans.
                try:
                    import shutil
                    shutil.rmtree(ep_dir, ignore_errors=True)
                except Exception as exc:
                    logger.warning("failed to clean discarded episode dir: %s", exc)
                _emit("Episode discarded")

            if stop_episode or stop_event.is_set():
                break
            if budget is not None and saved >= budget:
                break
            _emit(f"Episode {ep_idx} ready")
    finally:
        stop_event.set()
        try:
            box.stop()
        except Exception as exc:
            logger.warning("box.stop on shutdown: %s", exc)
        _emit("Recording stopped")
    return rc


if __name__ == "__main__":
    sys.exit(main())
