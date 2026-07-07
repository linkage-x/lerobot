"""Thor data-collection orchestrator: 11 x GMSL2 cameras + BOX 采集板.

This is the recorder script spawned by ``tools/data_collection_gui/gateway.py``
when the operator presses *Connect* on the LeRobot data-collection GUI. It is
the stable UI protocol boundary around:

  * ``argus_metadata_session.py`` — the default production camera backend. It
    owns all selected cameras through Libargus, records encoded video, and
    writes one per-frame metadata sidecar per camera.
  * ``persistent_session.py`` — explicit legacy fallback for
    ``recorder_backend: gstreamer_splitmux``.
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

The recorder always runs in *streaming* mode; frame pixels never enter Python.
Progress shown to the operator is estimated from elapsed time * configured fps.
For the default ``argus_metadata`` backend, authoritative cross-camera sync
comes from the Libargus SOF timestamp sidecars and
``argus_frame_alignment.json`` written for each saved episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Allow `python tools/thor/gmsl2/thor_record.py` invocation without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import gmsl2_record as gr  # noqa: E402
from tools.thor.gmsl2 import argus_frame_sync as afs  # noqa: E402
from tools.thor.gmsl2 import argus_metadata_session as ams  # noqa: E402
from tools.thor.gmsl2 import argus_online_sync_session as aos  # noqa: E402
from tools.thor.gmsl2 import argus_video_materialize as avm  # noqa: E402
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


# Healthy BOX sensors stream at ~199 Hz (touch ~50 Hz). A box that answers
# discovery but has stalled/aged out shows 0 Hz; a degraded one limps at ~8 Hz.
# 30 Hz cleanly separates healthy (>=50) from both failure modes and is the floor
# below which recorded box data is too sparse to be usable.
_MIN_HEALTHY_STREAM_HZ = 30.0


def _streaming_sensor_ids(observed_rates: dict[str, float]) -> list[str]:
    """Sensor ids that are actually pushing data (observed update rate > 0 Hz).

    ``observed_rates`` is derived from MCU-timestamp *advancement* (not raw poll
    count), so an empty result distinguishes a BOX that answers discovery but is
    not streaming -- including the "frozen cache" state where get_sensor_cache
    keeps returning the same stale snapshot -- from one that is streaming.
    """
    return sorted(sid for sid, hz in observed_rates.items() if hz > 0)


def _box_stream_health(
    observed_rates: dict[str, float],
    *,
    min_hz: float = _MIN_HEALTHY_STREAM_HZ,
) -> tuple[bool, float]:
    """Return ``(healthy, peak_hz)`` for the pre-record BOX health check.

    Healthy only when the fastest sensor clears ``min_hz`` -- this catches both a
    fully stalled box (0 Hz: discovery OK but no data) and a *degraded* one (e.g.
    ~8 Hz instead of ~199 Hz) that would otherwise record sparse, unusable box
    data and "no touch sample" frames. The box data stream can age out; a host
    re-arm helps but cannot always revive it (see box_client + ts_sync.md), so we
    warn the operator to power-cycle instead of failing silently.
    """
    peak_hz = max(observed_rates.values(), default=0.0)
    return (peak_hz >= min_hz, peak_hz)


def _connect_session_with_deadline(
    session: ps.PersistentCameraSession,
    *,
    timeout_s: float | None,
) -> tuple[bool, str]:
    """Run PersistentCameraSession.connect() inside a wall-clock budget.

    Argus failures can leave worker startup stuck long enough that the GUI only
    sees a meaningless last stdout line.  This helper gives Connect one bounded
    budget and forces session teardown when the budget is exhausted.
    """
    if timeout_s is not None and timeout_s <= 0:
        return False, "connect exceeded global deadline before starting attempt"

    if timeout_s is None:
        try:
            session.connect()
        except RuntimeError as exc:
            return False, str(exc)
        return True, ""

    done = threading.Event()
    result: dict[str, Exception] = {}

    def _target() -> None:
        try:
            session.connect()
        except Exception as exc:  # noqa: BLE001 - convert worker failures to GUI text
            result["error"] = exc
        finally:
            done.set()

    started = time.monotonic()
    worker = threading.Thread(
        target=_target, name="thor-connect-deadline", daemon=True,
    )
    worker.start()
    if done.wait(timeout_s):
        exc = result.get("error")
        if exc is not None:
            return False, str(exc)
        return True, ""

    elapsed = time.monotonic() - started
    try:
        session.disconnect()
    except Exception as exc:  # noqa: BLE001 - timeout path should still report timeout
        logger.warning("pcs.disconnect after connect deadline: %s", exc)
    return (
        False,
        f"connect exceeded global deadline {timeout_s:.1f}s "
        f"(elapsed {elapsed:.1f}s); session teardown requested",
    )


def _emit_box_live(box: bc.BoxPool, *, last_emit_s: float, min_interval_s: float = 0.1) -> float:
    now = time.monotonic()
    if now - last_emit_s < min_interval_s:
        return last_emit_s
    try:
        snap = box.read()
    except Exception as exc:  # noqa: BLE001 - preview must never break recording
        logger.debug("box live read failed: %s", exc)
        return now
    _emit("BOX_LIVE " + json.dumps(snap, separators=(",", ":")))
    return now


_FORCE_CHANNEL_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def _fmt_force_vec(vec: list[float] | None) -> str:
    if not vec:
        return "n/a"
    n = min(len(_FORCE_CHANNEL_LABELS), len(vec))
    return ", ".join(f"{_FORCE_CHANNEL_LABELS[i]}={vec[i]:.4f}" for i in range(n))


def _run_six_d_force_cali(box: bc.BoxPool) -> None:
    """Trigger a 6D force-sensor calibration and stream progress as CALI_LOG lines.

    Runs on its own thread: the stdin reader must not block on the ~0.5s+ SDK
    round-trip. Emits a terminal ``CALI_DONE ok|error`` line so the gateway can
    mark the run complete and the frontend can stop spinning.
    """
    _emit("CALI_LOG 6D force sensor calibration requested")
    try:
        results = box.calibrate_six_d_force()
    except Exception as exc:  # noqa: BLE001 - calibration must never crash the recorder
        _emit(f"CALI_LOG ERROR calibration raised: {exc}")
        _emit("CALI_DONE error")
        return
    ok_all = bool(results)
    for res in results:
        label = res.get("box_id") or "box"
        if res.get("ok"):
            _emit(f"CALI_LOG [{label}] calibration OK (rc={res.get('rc')})")
        else:
            ok_all = False
            _emit(
                f"CALI_LOG [{label}] calibration FAILED (rc={res.get('rc')}): "
                f"{res.get('error') or 'unknown error'}"
            )
        before, after = res.get("before"), res.get("after")
        if before is not None or after is not None:
            _emit(f"CALI_LOG [{label}] before: {_fmt_force_vec(before)}")
            _emit(f"CALI_LOG [{label}] after:  {_fmt_force_vec(after)}")
    _emit(f"CALI_DONE {'ok' if ok_all else 'error'}")


def _read_stdin_loop(
    queue: list[StdinCommand],
    stop: threading.Event,
    on_demand: Callable[[], None] | None = None,
    on_calibrate: Callable[[], None] | None = None,
) -> None:
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
        elif stripped == "preview_demand":
            # Out-of-band viewer-demand heartbeat from the gateway (a frontend
            # tile is polling camera.jpg). This is a continuous signal, not a
            # start/save/discard/quit command, so apply it as a side-effect
            # instead of enqueuing FSM noise that _wait_for_command would log
            # and drop.
            if on_demand is not None:
                on_demand()
        elif stripped == "cali_6dforce":
            # Out-of-band 6D force-sensor calibration request from the gateway
            # (Device Manager button). Like preview_demand it's a side-effect,
            # not an FSM start/save/quit command, so fire the callback instead
            # of enqueuing it. The callback offloads the actual SDK call to its
            # own thread so reading further stdin lines never blocks on it.
            if on_calibrate is not None:
                on_calibrate()
        else:
            logger.warning("ignoring unrecognized stdin command: %r", line)


def _preview_demand_decision(
    *,
    on_demand: bool,
    active: bool,
    last_demand_mono: float,
    now_mono: float,
    ttl_s: float,
) -> bool:
    """Whether recorder-owned previews should be attached right now.

    ``on_demand`` False -> legacy always-on: previews were eagerly enabled at
    Connect and must never be auto-toggled, so the desired state just mirrors
    the current one. ``on_demand`` True -> previews track viewer demand: keep
    them attached iff a demand heartbeat arrived within ``ttl_s`` (the gateway
    sends one roughly once a second while a Device Manager tile is polling).
    """
    if not on_demand:
        return active
    if last_demand_mono <= 0.0:
        return False
    return (now_mono - last_demand_mono) <= ttl_s


def _wait_for_command(
    queue: list[StdinCommand],
    stop: threading.Event,
    accept: tuple[str, ...],
    on_wait: Callable[[], None] | None = None,
) -> StdinCommand:
    """Block until the operator queues one of the accepted commands."""
    while not stop.is_set():
        if queue:
            cmd = queue.pop(0)
            if cmd.kind in accept:
                return cmd
            logger.info("ignoring %s while waiting for %s", cmd.kind, accept)
        else:
            if on_wait is not None:
                on_wait()
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


def _wallclock_utc_from_wall_s(wall_s: float) -> str:
    return datetime.fromtimestamp(float(wall_s), timezone.utc).isoformat()


def _write_episode_meta(
    handle: ps.EpisodeHandle,
    cfg: gr.RecorderConfig,
    locked: list[int],
    argus_failed: list[int],
    connect_stream_errors: list[ps.StreamError],
    box_cfg: bc.BoxFleetConfig,
    box_snapshots: list[dict[str, Any]],
    stop_reason: str,
    wallclock_start_utc: str,
    wallclock_end_utc: str,
) -> Path:
    """Write per-episode meta.json under the persistent-pipeline model.

    Mirrors the schema produced by gr.write_episode_meta (so downstream
    consumers don't break), but the ``sync_reference`` block is the new
    PR2 model:

      * ``t0_wall_s`` / ``t0_mono_s``: recording origin (host wall / monotonic
        clock) shared by cameras and BOX; BOX samples carry
        ``t_relative_s = wall - t0_wall_s``.
      * ``camera_first_wall_s``: per-camera wall time when the new fragment
        actually opened (from format-location-full callback). This is the
        cross-camera anchor downstream consumers align BOX/touch samples to.

    Per-stream first PTS is kept per-camera in ``cameras[].first_pts_s`` (it is
    single-stream only — NOT cross-camera comparable — so it is intentionally
    not duplicated into ``sync_reference``).

    The legacy ``camera_spawn_wall_s`` / ``camera_spawn_offset_s`` fields
    are gone; the closest replacement is ``camera_first_wall_s``.
    """
    fragments = handle.fragments
    duration_s = max(0.0, handle.stop_wall_s - handle.t0_wall_s)
    connect_error_payload = [
        {"sid": err.sid, "name": err.name, "message": err.message}
        for err in connect_stream_errors
    ]
    active_camera_sids = sorted(info.sid for info in fragments.values())
    connect_failed_sids = sorted({
        *argus_failed,
        *(err.sid for err in connect_stream_errors if err.sid >= 0),
    })
    meta = {
        "episode_index": handle.idx,
        "repo_id": cfg.repo_id,
        "single_task": cfg.single_task,
        "wallclock_start_utc": wallclock_start_utc,
        "wallclock_end_utc": wallclock_end_utc,
        "duration_s": duration_s,
        "recording_stop_reason": stop_reason,
        "video": {
            "recorder_backend": cfg.cameras.recorder_backend,
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
            "pipeline": (
                "Libargus CaptureSession | nveglstreamsrc | nvv4l2h{265,264}enc "
                "| mux + per-frame Argus metadata sidecar"
                if cfg.cameras.recorder_backend == "argus_metadata"
                else "nvarguscamerasrc | nvv4l2h{265,264}enc | splitmuxsink (persistent)"
            ),
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
            "camera_first_wall_s": {
                name: info.first_wall_s for name, info in fragments.items()
            },
            "note": (
                "For recorder_backend=argus_metadata, cross-camera alignment "
                "comes from <camera>.argus_frame_metadata.csv SOF TSC values "
                "and argus_frame_alignment.json. For the legacy persistent "
                "splitmux backend, t0_wall_s is the host time when "
                "start_episode() emitted split-now; camera_first_wall_s is "
                "only a host-side fragment-open time and is not a hardware "
                "frame timestamp."
            ),
        },
        "max96726_locked_sids": locked,
        "argus_failed_sids": argus_failed,
        "connect_failed_sids": connect_failed_sids,
        "connect_stream_errors": connect_error_payload,
        "active_camera_sids": active_camera_sids,
        "spawn_stagger_s": cfg.spawn_stagger_s,
        "connect_stable_s": cfg.connect_stable_s,
        "connect_timeout_s": cfg.connect_timeout_s,
        "connect_first_fragment_timeout_s": cfg.connect_first_fragment_timeout_s,
        "stop_on_stream_exit": cfg.stop_on_stream_exit,
        "argus_frame_sync": {
            "enabled": cfg.argus_frame_sync.enabled,
            "required": cfg.argus_frame_sync.required,
            "reference_strategy": cfg.argus_frame_sync.reference_strategy,
            "reference_camera": cfg.argus_frame_sync.reference_camera or None,
            "tolerance_ms": cfg.argus_frame_sync.tolerance_ms,
            "cadence_tolerance_ms": cfg.argus_frame_sync.cadence_tolerance_ms,
            "report_name": cfg.argus_frame_sync.report_name,
            "materialize_verify": cfg.argus_frame_sync.materialize_verify,
            "materialize_workers": cfg.argus_frame_sync.materialize_workers,
        },
        "online_sync": {
            "enabled": cfg.online_sync.enabled,
            "sync_source": cfg.online_sync.sync_source,
            "tolerance_ms": cfg.online_sync.tolerance_ms,
            "startup_full_clusters": cfg.online_sync.startup_full_clusters,
            "frame_timeout_ms": cfg.online_sync.frame_timeout_ms,
            "missing_frame_policy": cfg.online_sync.missing_frame_policy,
            "stop_mode": cfg.online_sync.stop_mode,
        },
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


def _evaluate_argus_frame_sync(
    handle: ps.EpisodeHandle,
    cfg: gr.RecorderConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the per-frame Argus SOF alignment gate for one episode.

    Returns ``(payload, failure_reason)``. ``payload`` is suitable for
    ``meta.json``. ``failure_reason`` is non-None only when the episode should
    not be saved.
    """

    sync_cfg = cfg.argus_frame_sync
    if not sync_cfg.enabled:
        return None, None

    camera_names = sorted(handle.fragments)
    payload: dict[str, Any] = {
        "enabled": True,
        "required": sync_cfg.required,
        "reference_strategy": sync_cfg.reference_strategy,
        "reference_camera": sync_cfg.reference_camera or None,
        "tolerance_ms": sync_cfg.tolerance_ms,
        "cadence_tolerance_ms": sync_cfg.cadence_tolerance_ms,
        "report_name": sync_cfg.report_name,
        "materialize_verify": sync_cfg.materialize_verify,
        "materialize_workers": sync_cfg.materialize_workers,
        "sidecars": {},
    }
    if not camera_names:
        payload["ok"] = False
        payload["failures"] = ["no camera fragments in episode handle"]
        return payload, "argus_frame_sync_failed" if sync_cfg.required else None

    missing: list[str] = []
    sidecars: dict[str, Path] = {}
    for camera in camera_names:
        path = afs.frame_metadata_sidecar_path(handle.directory, camera)
        payload["sidecars"][camera] = str(path)
        if not path.exists():
            missing.append(camera)
        else:
            sidecars[camera] = path

    if missing:
        payload["ok"] = False
        payload["missing_sidecars"] = missing
        payload["failures"] = [
            "missing Argus frame metadata sidecars for: " + ", ".join(missing)
        ]
        failure = "argus_frame_sync_missing_sidecars" if sync_cfg.required else None
        return payload, failure

    try:
        markers_path = handle.directory / "argus_recording_markers.json"
        markers: dict[str, Any] = {}
        start_sof_tsc_ns: int | None = None
        stop_sof_tsc_ns: int | None = None
        if markers_path.exists():
            markers = json.loads(markers_path.read_text())
            raw_start_sof = markers.get("start_sof_tsc_ns")
            if raw_start_sof:
                start_sof_tsc_ns = int(raw_start_sof)
            raw_stop_sof = (
                markers.get("stop_sof_tsc_ns_exclusive")
                or markers.get("stop_sof_tsc_ns")
            )
            if raw_stop_sof:
                stop_sof_tsc_ns = int(raw_stop_sof)

        frames_by_camera = {
            camera: afs.read_frame_metadata_csv(path, camera=camera)
            for camera, path in sidecars.items()
        }
        alignment = afs.align_episode_frames(
            frames_by_camera,
            reference_camera=(
                sync_cfg.reference_camera
                or (markers.get("reference_camera") if sync_cfg.reference_strategy == "camera" else None)
                or None
            ),
            reference_strategy=sync_cfg.reference_strategy,
            tolerance_ns=int(round(sync_cfg.tolerance_ms * 1_000_000)),
            expected_period_ns=int(round(1_000_000_000 / max(int(cfg.cameras.fps), 1))),
            cadence_tolerance_ns=int(round(sync_cfg.cadence_tolerance_ms * 1_000_000)),
            start_sof_tsc_ns=start_sof_tsc_ns,
            stop_sof_tsc_ns=stop_sof_tsc_ns,
        )
        report_path = handle.directory / sync_cfg.report_name
        afs.write_alignment_report_json(report_path, alignment)
        frame_windows = {
            camera: asdict(window)
            for camera, window in afs.camera_frame_windows(alignment).items()
        } if alignment.ok else {}
        materialized_videos = None
        if alignment.ok and cfg.cameras.recorder_backend == "argus_metadata":
            materialized_videos = avm.materialize_aligned_videos(
                handle.directory,
                {name: info.path for name, info in handle.fragments.items()},
                frames_by_camera,
                alignment,
                fps=cfg.cameras.fps,
                codec=cfg.cameras.codec,
                verify_frame_counts=sync_cfg.materialize_verify,
                max_workers=sync_cfg.materialize_workers,
            )
    except Exception as exc:  # noqa: BLE001 - convert metadata failures to episode gate
        payload["ok"] = False
        payload["failures"] = [f"Argus frame sync evaluation failed: {exc}"]
        failure = "argus_frame_sync_failed" if sync_cfg.required else None
        return payload, failure

    payload.update({
        "ok": alignment.ok,
        "reference_strategy": sync_cfg.reference_strategy,
        "reference_camera": alignment.reference_camera,
        "tolerance_ns": alignment.tolerance_ns,
        "expected_period_ns": alignment.expected_period_ns,
        "cadence_tolerance_ns": alignment.cadence_tolerance_ns,
        "recording_markers": markers,
        "reference_frame_count": alignment.reference_frame_count,
        "accepted_reference_indices": alignment.accepted_reference_indices,
        "dropped_reference_indices": alignment.dropped_reference_indices,
        "drop_reasons": alignment.drop_reasons,
        "frame_windows": frame_windows,
        "materialized_videos": materialized_videos,
        "materialize_verify": sync_cfg.materialize_verify,
        "materialize_workers": sync_cfg.materialize_workers,
        "frame_count_by_camera": alignment.frame_count_by_camera(),
        "max_abs_delta_ns_by_camera": {
            name: camera_alignment.max_abs_delta_ns
            for name, camera_alignment in alignment.cameras.items()
        },
        "failures": alignment.failures,
        "report_path": str(report_path),
    })
    failure = "argus_frame_sync_failed" if sync_cfg.required and not alignment.ok else None
    return payload, failure


def _sidecar_data_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _evaluate_online_sync_manifest(
    handle: ps.EpisodeHandle,
    cfg: gr.RecorderConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the encoder-front online-sync contract before saving."""

    manifest_path = handle.directory / "online_sync_manifest.json"
    payload: dict[str, Any] = {
        "enabled": True,
        "manifest": str(manifest_path),
        "sync_source": cfg.online_sync.sync_source,
        "tolerance_ms": cfg.online_sync.tolerance_ms,
        "tolerance_ns": int(round(cfg.online_sync.tolerance_ms * 1_000_000)),
        "sidecar_counts": {},
        "failures": [],
    }
    if not manifest_path.exists():
        payload["ok"] = False
        payload["failures"] = ["missing online_sync_manifest.json"]
        return payload, "online_sync_missing_manifest"

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        payload["ok"] = False
        payload["failures"] = [f"failed to read online sync manifest: {exc}"]
        return payload, "online_sync_manifest_invalid"

    payload["manifest_payload"] = manifest
    failures: list[str] = []
    if not manifest.get("ok"):
        failures.append(str(manifest.get("failure") or "online_sync_manifest.ok is false"))

    actual_frames = int(manifest.get("actual_frames") or 0)
    frame_count_by_camera = manifest.get("frame_count_by_camera") or {}
    active_cameras = sorted(handle.fragments)
    if actual_frames <= 0:
        failures.append("online sync actual_frames is zero")
    for camera in active_cameras:
        count = int(frame_count_by_camera.get(camera) or 0)
        if count != actual_frames:
            failures.append(f"{camera} manifest frame count {count} != {actual_frames}")
        sidecar = afs.frame_metadata_sidecar_path(handle.directory, camera)
        sidecar_count = _sidecar_data_row_count(sidecar)
        payload["sidecar_counts"][camera] = sidecar_count
        if sidecar_count != actual_frames:
            failures.append(f"{camera} sidecar rows {sidecar_count} != {actual_frames}")

    tolerance_ns = int(round(cfg.online_sync.tolerance_ms * 1_000_000))
    for camera, delta in (manifest.get("max_abs_delta_ns_by_camera") or {}).items():
        try:
            if int(delta) > tolerance_ns:
                failures.append(f"{camera} max SOF delta {delta} ns > {tolerance_ns} ns")
        except (TypeError, ValueError):
            failures.append(f"{camera} invalid SOF delta {delta!r}")

    payload["ok"] = not failures
    payload["actual_frames"] = actual_frames
    payload["frame_count_by_camera"] = frame_count_by_camera
    payload["max_abs_delta_ns_by_camera"] = manifest.get("max_abs_delta_ns_by_camera") or {}
    payload["failures"] = failures
    return payload, None if not failures else "online_sync_failed"


# ----------------------------------------------------------------- auto-recover ---
#
# When the operator presses Connect, the recorder may find that nvargus-daemon
# is in a wedged state from a previous crashed session: some sids time out
# during PLAYING, retries can't rescue them, and the operator currently has
# to ssh in and run `tools/thor/gmsl2/recover_argus.sh` by hand. This block
# wraps that recovery script as a fallback inside Connect.


@dataclass
class AutoRecoverConfig:
    enabled: bool = True
    # Path passed as ``--sdk`` to recover_argus.sh. When None, the recorder
    # falls back to ``cfg.hardware_sync.sdk_dir`` resolved against repo root.
    sdk_dir: str | None = None
    # Trigger recover when fewer than ``threshold_fraction`` of the expected
    # cameras came up — 0.6 means "if 7+ of 11 fall off, recover".
    threshold_fraction: float = 0.6
    # Hard cap on the number of recover+retry rounds inside one Connect
    # call so a permanently broken daemon doesn't trap the operator.
    max_attempts: int = 1
    # Wall-clock budget for the recover_argus.sh invocation itself.
    timeout_s: float = 300.0


def _auto_recover_from_yaml(yaml_dict: dict[str, Any] | None) -> AutoRecoverConfig:
    """Build an AutoRecoverConfig from the optional YAML ``auto_recover`` block."""
    if not isinstance(yaml_dict, dict):
        return AutoRecoverConfig()
    return AutoRecoverConfig(
        enabled=bool(yaml_dict.get("enabled", True)),
        sdk_dir=yaml_dict.get("sdk_dir"),
        threshold_fraction=float(yaml_dict.get("threshold_fraction", 0.6)),
        max_attempts=int(yaml_dict.get("max_attempts", 1)),
        timeout_s=float(yaml_dict.get("timeout_s", 300.0)),
    )


def _resolve_recover_sdk_dir(
    auto_cfg: AutoRecoverConfig,
    fallback_sdk_dir: str,
    repo_root: Path,
) -> Path:
    """Resolve which directory to pass as ``--sdk`` to recover_argus.sh.

    Explicit ``auto_cfg.sdk_dir`` wins. Otherwise we reuse the hardware-sync
    SDK path so deployments don't have to repeat themselves in YAML.
    """
    chosen = auto_cfg.sdk_dir or fallback_sdk_dir
    expanded = Path(os.path.expanduser(chosen))
    if not expanded.is_absolute():
        expanded = (repo_root / expanded).resolve()
    return expanded


def _run_recover_argus(
    repo_root: Path, sdk_dir: Path, *, timeout_s: float = 300.0,
    _runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[bool, str]:
    """Run ``tools/thor/gmsl2/recover_argus.sh --sdk <sdk_dir>``.

    Returns ``(ok, tail)`` where ``tail`` is up to 400 chars of stderr/stdout
    captured from the script — useful for emitting a one-line failure
    summary to the GUI without flooding the recorder log.

    ``_runner`` is injectable for tests so we don't have to spawn a real
    subprocess just to verify the rc-handling logic.
    """
    script = repo_root / "tools" / "thor" / "gmsl2" / "recover_argus.sh"
    if not script.is_file():
        return False, f"recover_argus.sh not found at {script}"
    # recover_argus.sh is also used manually, where killing stale gateway /
    # recorder processes is useful. From inside thor_record.py that would kill
    # this recorder and the GUI gateway that spawned it, so keep only the
    # Argus/module/probe recovery actions here.
    cmd = ["bash", str(script), "--sdk", str(sdk_dir), "--skip-kill"]
    runner = _runner or subprocess.run
    try:
        r = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"recover_argus.sh timed out after {timeout_s:.0f}s"
    except Exception as exc:
        return False, f"recover_argus.sh raised: {exc}"
    if r.returncode == 0:
        return True, ""
    tail = (r.stderr or r.stdout or "").strip()
    return False, tail[-400:]


def _should_trigger_recovery(
    active_count: int, expected_count: int, threshold_fraction: float,
) -> bool:
    """Decide whether a partial-success outcome is bad enough to recover.

    ``expected_count`` is what passed Argus probe / detect-locked, NOT 11.
    ``active_count`` is ``len(pcs.active_sids)`` after connect() returned.
    """
    if expected_count <= 0:
        return False
    return active_count < expected_count * threshold_fraction


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
            container=cfg.cameras.container,
            bitrate_kbps=cfg.cameras.bitrate_kbps,
            iframe_interval=cfg.cameras.iframe_interval,
            preset_level=cfg.cameras.preset_level,
            control_rate=cfg.cameras.control_rate,
            sensor_mode=cfg.cameras.sensor_mode,
            exposure_us=cfg.cameras.exposure_us,
            gain=cfg.cameras.gain,
            argus_gain=cfg.cameras.argus_gain,
            preview_jpeg_path=(
                str(ps.preview_frame_path(f"{cfg.name_prefix}_{sid:02d}"))
                if cfg.recording_preview_enabled else None
            ),
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
    ap.add_argument("--no-auto-recover", action="store_true",
                    help="disable the automatic recover_argus.sh round triggered "
                         "by a wedged nvargus-daemon (default: enabled)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    cfg = gr.load_config(args.config_path)
    if cfg.cameras.recorder_backend == "argus_metadata":
        cfg.argus_frame_sync.enabled = True
        cfg.argus_frame_sync.required = True
    raw_yaml: dict[str, Any] = {}
    with args.config_path.open() as f:
        import yaml
        raw_yaml = yaml.safe_load(f) or {}
    box_cfg = bc.fleet_from_yaml_dict(raw_yaml.get("box_collection") if not args.no_box else None)
    auto_cfg = _auto_recover_from_yaml(raw_yaml.get("auto_recover"))
    if args.no_auto_recover:
        auto_cfg.enabled = False

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

    # Start BOX SDK BEFORE the camera recorder: BOX SDK loads its own native
    # .so + UDP listener; starting it after camera capture is already under
    # heavy Argus/GStreamer load has previously triggered SIGABRT on the first
    # incoming UDP packet (see tools/data_collection_gui/docs/development_status.md).
    box = bc.BoxPool(box_cfg)
    box_started = box.start() if box_cfg.enabled else False
    if box_started:
        # Surface the discovered BOX roster (device_id / sn / ip / capabilities)
        # so the gateway renders one GUI row per (discovered box × sensor)
        # instead of relying on the static YAML config -- this is what makes
        # "click Connect -> list all box devices" reflect the real subnet.
        roster = box.discovered_devices()
        if roster:
            _emit("BOX_DEVICES_JSON " + json.dumps(roster, separators=(",", ":")))
        # Wait for the rate-estimation window (2s) to fill so we can report
        # real per-sensor frequencies instead of the poll rate.
        time.sleep(2.5)
        # Use observed_rates() (real UDP arrivals) as the source of truth for
        # "Box devices:" instead of connected_devices() (SDK's static
        # registry). connected_devices() reports whatever was registered at
        # init time and stays the same even when the BOX MCU is unplugged,
        # which let the gateway keep BOX entries green after a physical
        # disconnect. With this change, unplugging BOX -> rates all 0 ->
        # emit "Box devices: (none)" -> gateway flips all box rows to red.
        rates = box.observed_rates()
        live_sids = _streaming_sensor_ids(rates)
        _emit(f"Box devices: {', '.join(live_sids) if live_sids else '(none)'}")
        if live_sids:
            parts = [f"{sid}={rates[sid]:.0f}" for sid in live_sids]
            _emit(f"Box rates: {', '.join(parts)}")
        healthy, peak_hz = _box_stream_health(rates)
        if not healthy:
            # Discovery succeeded (box answered :15001) but the data stream on
            # :15000 is stalled (0 Hz) or degraded (e.g. ~8 Hz vs ~199 Hz) and a
            # host re-arm can't always revive it. Warn loudly so the operator
            # power-cycles instead of recording empty/sparse box episodes.
            _emit(
                f"WARNING: BOX stream unhealthy (peak {peak_hz:.0f} Hz, expected ~199); "
                "recordings will have missing/sparse box data. Power-cycle the box and reconnect."
            )
    elif box_cfg.enabled:
        _emit("Box devices: (none)")
        logger.warning("box_collection enabled but BoxClient.start() returned False")

    # Session backend: production default is the Libargus metadata-integrated
    # recorder. The legacy persistent GStreamer/splitmux backend remains
    # available only when explicitly selected in YAML.
    def _make_pcs():
        streams = _stream_configs(usable, cfg)
        if cfg.cameras.recorder_backend == "argus_metadata":
            return ams.ArgusMetadataCameraSession(
                streams,
                warmup_dir,
                repo_root=repo_root,
                connect_timeout_s=max(30.0, float(cfg.connect_timeout_s or 0.0)),
                connect_stable_s=cfg.connect_stable_s,
            )
        if cfg.cameras.recorder_backend == "argus_online_sync":
            target_frames = 0
            if cfg.episode_time_s > 0:
                target_frames = int(round(float(cfg.episode_time_s) * int(cfg.cameras.fps)))
            return aos.ArgusOnlineSyncCameraSession(
                streams,
                warmup_dir,
                repo_root=repo_root,
                connect_timeout_s=max(30.0, float(cfg.connect_timeout_s or 0.0)),
                connect_stable_s=cfg.connect_stable_s,
                target_frames=target_frames,
                tolerance_ms=cfg.online_sync.tolerance_ms,
                startup_full_clusters=cfg.online_sync.startup_full_clusters,
                frame_timeout_ms=cfg.online_sync.frame_timeout_ms,
                preflight_timeout_s=cfg.online_sync.preflight_timeout_s,
                single_preflight_timeout_s=cfg.online_sync.single_preflight_timeout_s,
                missing_frame_policy=cfg.online_sync.missing_frame_policy,
                stop_mode=cfg.online_sync.stop_mode,
            )
        return ps.PersistentCameraSession(
            streams,
            warmup_dir,
            spawn_stagger_s=cfg.spawn_stagger_s,
            connect_stable_s=cfg.connect_stable_s,
            first_fragment_timeout_s=cfg.connect_first_fragment_timeout_s,
            two_phase_connect=cfg.two_phase_connect,
        )

    connect_timeout_s = max(0.0, float(cfg.connect_timeout_s))
    connect_deadline_at = (
        time.monotonic() + connect_timeout_s if connect_timeout_s > 0 else None
    )

    def _connect_deadline_remaining_s() -> float | None:
        if connect_deadline_at is None:
            return None
        return max(0.0, connect_deadline_at - time.monotonic())

    def _attempt_connect() -> tuple[ps.PersistentCameraSession | None, str]:
        remaining_s = _connect_deadline_remaining_s()
        if remaining_s is not None and remaining_s <= 0:
            return (
                None,
                f"connect exceeded global deadline {connect_timeout_s:.1f}s "
                "before starting attempt",
            )
        new_pcs = _make_pcs()
        suffix = (
            f" (deadline remaining {remaining_s:.1f}s)"
            if remaining_s is not None else ""
        )
        _emit(f"Connecting: spawning {len(usable)} persistent pipelines...{suffix}")
        ok, message = _connect_session_with_deadline(
            new_pcs, timeout_s=remaining_s,
        )
        if ok:
            return new_pcs, ""
        try:
            new_pcs.disconnect()
        except Exception as cleanup_exc:
            logger.warning(
                "pcs.disconnect after connect failure: %s", cleanup_exc,
            )
        return None, message

    pcs, attempt_error = _attempt_connect()
    expected = len(usable)
    recover_attempts = 0

    while True:
        # Decide whether the current outcome warrants an auto-recover round.
        recover_reason: str | None = None
        if pcs is None:
            recover_reason = f"connect raised: {attempt_error}"
        else:
            active = len(pcs.active_sids)
            if _should_trigger_recovery(active, expected, auto_cfg.threshold_fraction):
                recover_reason = (
                    f"only {active}/{expected} cameras up "
                    f"(threshold {int(auto_cfg.threshold_fraction * 100)}%)"
                )
        if (recover_reason is None
                or not auto_cfg.enabled
                or recover_attempts >= auto_cfg.max_attempts):
            break

        # Tear down whatever partial session we have so recover_argus.sh
        # doesn't see lingering workers holding Argus sockets.
        if pcs is not None:
            try:
                pcs.disconnect()
            except Exception as cleanup_exc:
                logger.warning(
                    "pcs.disconnect before auto-recover: %s", cleanup_exc,
                )
            pcs = None

        remaining_s = _connect_deadline_remaining_s()
        if remaining_s is not None and remaining_s <= 0:
            attempt_error = (
                f"connect exceeded global deadline {connect_timeout_s:.1f}s "
                "before auto-recover"
            )
            break

        sdk_dir = _resolve_recover_sdk_dir(
            auto_cfg, cfg.hardware_sync.sdk_dir, repo_root,
        )
        recover_timeout_s = auto_cfg.timeout_s
        if remaining_s is not None:
            recover_timeout_s = min(recover_timeout_s, remaining_s)
        _emit(f"Auto-recover: {recover_reason}; running recover_argus.sh "
              f"(sdk={sdk_dir}, timeout={recover_timeout_s:.1f}s)")
        ok, tail = _run_recover_argus(
            repo_root, sdk_dir, timeout_s=recover_timeout_s,
        )
        recover_attempts += 1
        if not ok:
            recover_tail = tail or 'see recorder log'
            _emit(f"Auto-recover failed: {recover_tail}")
            attempt_error = (
                f"auto-recover failed: {recover_tail}; "
                f"previous connect outcome: {recover_reason}"
            )
            break
        _emit("Auto-recover OK; retrying connect")
        pcs, attempt_error = _attempt_connect()

    if pcs is None:
        _emit(f"ERROR: persistent pipeline connect failed: {attempt_error}")
        if box_started:
            try:
                box.stop()
            except Exception as cleanup_exc:
                logger.warning("box.stop after connect failure: %s", cleanup_exc)
        return 1

    # Partial-failure path: connect() returns success even when some workers
    # fell off, leaving their errors in pcs.poll_errors(). Surface those to
    # the operator and re-emit the active camera list so the GUI's status
    # reflects what is actually recording.
    connect_errors = pcs.poll_errors()
    if connect_errors:
        bad = ", ".join(ps.format_stream_error(e) for e in connect_errors)
        _emit(f"WARNING: {len(connect_errors)} stream(s) failed: {bad}")
        active_camera_ids = [
            f"{cfg.name_prefix}_{sid:02d}" for sid in pcs.active_sids
        ]
        _emit(f"Cameras (active): {', '.join(active_camera_ids)}")
    _emit(f"Connected {len(pcs.active_sids)} pipelines in {pcs.connect_duration_s:.1f}s")

    # --- recorder-owned preview lifecycle (on-demand) -------------------
    # Previews are a lossy tee branch off each recording pipeline. Keeping 11
    # nvvidconv+jpegenc branches attached and running 24/7 adds steady VIC/NVMM
    # load even when nobody is looking at the Device Manager grid, and that idle
    # load is a known aggravator of the GMSL2/Argus stream-on fragility
    # (DEPLOYMENT.md §6.2 / the 2026-06-03 armed-idle EOS storm). So mirror the
    # idle-preview TTL: only attach preview branches while the gateway reports a
    # viewer is polling camera.jpg, and reclaim them after
    # recording_preview_idle_ttl_s of silence. The gateway sends a debounced
    # "preview_demand" heartbeat on the snapshot route; the stdin reader records
    # the latest one and _preview_control_loop (below) drives enable/disable.
    preview_stagger_s = max(0.0, cfg.recording_preview_stagger_s)
    preview_stale_s = max(0.0, cfg.recording_preview_stale_s)
    preview_ttl_s = max(0.0, cfg.recording_preview_idle_ttl_s)
    preview_on_demand = cfg.recording_preview_enabled and cfg.recording_preview_on_demand
    preview_grace_s = max(
        preview_stale_s, preview_stagger_s * max(1, len(pcs.active_sids)) + 2.0
    )
    preview_demand_at = [0.0]  # monotonic of last viewer-demand heartbeat; 0 = none
    preview_demand_lock = threading.Lock()

    def _note_preview_demand() -> None:
        with preview_demand_lock:
            preview_demand_at[0] = time.monotonic()

    if cfg.recording_preview_enabled and not preview_on_demand:
        # Legacy always-on: eager enable at Connect with operator feedback.
        _emit("Preview: enabling recorder-owned camera previews...")
        pcs.enable_previews(stagger_s=preview_stagger_s)
        missing_previews = pcs.wait_preview_frames(timeout_s=max(5.0, preview_grace_s))
        if missing_previews:
            _emit(f"Preview warning: no initial frame for {', '.join(missing_previews)}")
        else:
            _emit("Preview: recorder-owned camera previews enabled")
    elif preview_on_demand:
        _emit(
            "Preview: on-demand (cameras stream preview only while the "
            "Device Manager grid is open)"
        )

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

    # Stdin reader thread -- gateway writes single lines per command. The
    # on_demand callback is fired for "preview_demand" heartbeats (out of band
    # from the start/save/quit FSM).
    stop_event = threading.Event()
    cmd_queue: list[StdinCommand] = []

    def _trigger_six_d_force_cali() -> None:
        # Offload to a dedicated thread so the stdin reader keeps draining lines
        # (including preview_demand heartbeats) during the SDK round-trip.
        threading.Thread(
            target=_run_six_d_force_cali, args=(box,),
            daemon=True, name="thor-record-cali",
        ).start()

    stdin_thread = threading.Thread(
        target=_read_stdin_loop,
        args=(cmd_queue, stop_event, _note_preview_demand, _trigger_six_d_force_cali),
        daemon=True, name="thor-record-stdin",
    )
    stdin_thread.start()

    def _preview_control_loop() -> None:
        # Owns the entire preview lifecycle OFF the command loop so the
        # staggered enable (which sleeps between cameras) can never delay a
        # start/save/quit. ``active`` mirrors what we've asked the workers to do.
        # The stale watchdog only runs while attached and past the grace window;
        # dead recording streams are skipped inside refresh_stale_previews
        # (recording_failed gate), so a camera EOS no longer spins the watchdog.
        active = cfg.recording_preview_enabled and not preview_on_demand
        grace_until = (time.monotonic() + preview_grace_s) if active else 0.0
        last_refresh = 0.0
        refresh_interval = max(0.0, cfg.recording_preview_watchdog_s)
        while not stop_event.wait(0.5):
            if not cfg.recording_preview_enabled:
                continue
            with preview_demand_lock:
                last_demand = preview_demand_at[0]
            now = time.monotonic()
            desired = _preview_demand_decision(
                on_demand=preview_on_demand, active=active,
                last_demand_mono=last_demand, now_mono=now, ttl_s=preview_ttl_s,
            )
            if desired and not active:
                _emit("Preview: viewer detected; enabling camera previews")
                pcs.enable_previews(stagger_s=preview_stagger_s)
                active = True
                grace_until = time.monotonic() + preview_grace_s
            elif not desired and active:
                pcs.disable_previews()
                active = False
                _emit("Preview: no viewer; disabling camera previews to cut idle load")
            if not active or time.monotonic() < grace_until:
                continue
            if refresh_interval > 0 and time.monotonic() - last_refresh < refresh_interval:
                continue
            last_refresh = time.monotonic()
            restarted = pcs.refresh_stale_previews(max_age_s=preview_stale_s)
            if restarted:
                _emit(f"Preview stale: restarted {', '.join(restarted)}")

    preview_thread: threading.Thread | None = None
    if cfg.recording_preview_enabled:
        preview_thread = threading.Thread(
            target=_preview_control_loop, daemon=True, name="thor-record-preview",
        )
        preview_thread.start()

    def _sigint(_sig, _frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    saved = 0
    ep_idx = gr._next_episode_index(cfg.dataset_root)
    budget = cfg.num_episodes if cfg.num_episodes > 0 else None
    budget_str = str(budget) if budget else "unlimited"
    # The stdin hint is a CLI affordance for a human running this in a terminal.
    # Under the GUI gateway our stdin is a pipe (not a TTY) and the gateway
    # drives commands programmatically, so emitting it there only leaks a
    # misleading "press <Enter>" line into the frontend log. Match the
    # isatty() guard handheld_record.py already uses for interactive prompts.
    if sys.stdin.isatty():
        logger.info(_STDIN_HINT)
    _emit(f"Episode {ep_idx} ready")

    rc = 0
    last_box_live_at = 0.0
    last_stream_health_at = 0.0
    last_warmup_roll_at = 0.0

    def _format_stream_errors(errors: list[ps.StreamError]) -> str:
        return ", ".join(ps.format_stream_error(e) for e in errors)

    def _poll_stream_health(*, context: str) -> list[ps.StreamError]:
        nonlocal rc, last_stream_health_at
        poll_s = max(0.0, cfg.stream_health_poll_s)
        now = time.monotonic()
        if poll_s > 0 and now - last_stream_health_at < poll_s:
            return []
        last_stream_health_at = now
        stream_errs = pcs.poll_errors()
        if not stream_errs:
            return []
        details = _format_stream_errors(stream_errs)
        logger.warning("stream health errors during %s: %s", context, details)
        _emit(f"WARNING: {len(stream_errs)} stream(s) failed: {details}")
        if context == "idle":
            logger.warning(
                "stream health errors while armed; recorder remains alive until "
                "operator starts, quits, or reconnects"
            )
        return stream_errs

    def _tick_connected_idle() -> None:
        nonlocal last_box_live_at, last_warmup_roll_at
        _poll_stream_health(context="idle")
        # Bound the throwaway warmup stream while we sit armed but not recording.
        # splitmuxsink never auto-rotates (max-size-time=0), so the open warmup
        # fragment grows until the disk fills if we only ever cleaned up at
        # episode boundaries (the 2026-06 Thor _warmup blow-up: gateway left
        # connected over a weekend -> 120G). Roll then prune on a timer here.
        roll_s = max(0.0, cfg.warmup_roll_s)
        now = time.monotonic()
        if roll_s > 0 and now - last_warmup_roll_at >= roll_s:
            last_warmup_roll_at = now
            try:
                pcs.roll_warmup()
                deleted = pcs.cleanup_warmup_files(keep_last_n=cfg.warmup_keep_last_n)
                if deleted:
                    logger.info("idle warmup maintenance pruned %d fragment(s)", deleted)
            except Exception as exc:  # noqa: BLE001 - maintenance must not break idle
                logger.warning("idle warmup maintenance failed: %s", exc)
        # Preview lifecycle (enable/disable on demand + stale watchdog) runs in
        # its own thread (_preview_control_loop) so its staggered enable never
        # blocks command handling here.
        if box_started:
            last_box_live_at = _emit_box_live(box, last_emit_s=last_box_live_at)

    try:
        while not stop_event.is_set():
            cmd = _wait_for_command(
                cmd_queue, stop_event, accept=("start", "quit"), on_wait=_tick_connected_idle,
            )
            if cmd.kind == "quit":
                break

            ep_dir = cfg.dataset_root / "episodes" / f"episode_{ep_idx:06d}"
            requested_wall_start = datetime.now(timezone.utc).isoformat()
            t0_split_start_mono = time.monotonic()
            handle = pcs.start_episode(ep_dir, ep_idx)
            t_start = handle.t0_wall_s
            wall_start = _wallclock_utc_from_wall_s(t_start)
            split_emit_ms = (time.monotonic() - t0_split_start_mono) * 1000
            logger.info(
                "episode %d started @ %s -> %s (requested %s, start %.1fms)",
                ep_idx, wall_start, ep_dir, requested_wall_start, split_emit_ms,
            )
            if box_started:
                # Pre-record health check: if the box stalled/degraded between
                # Connect and Start (or mid-session), it answers discovery but
                # pushes no (or too-sparse) data, so this episode would record
                # empty/sparse box data. Surface it instead of failing silently --
                # the operator can discard + power-cycle.
                healthy, peak_hz = _box_stream_health(box.observed_rates())
                if not healthy:
                    _emit(
                        f"WARNING: BOX stream unhealthy at episode start (peak {peak_hz:.0f} Hz); "
                        "this episode will have missing/sparse box data. Discard it, "
                        "power-cycle the box, and reconnect."
                    )
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
                stream_errs = _poll_stream_health(context="recording")
                if stream_errs and cfg.stop_on_stream_exit:
                    if box_started:
                        snap = box.read()
                        snap["t_relative_s"] = elapsed
                        box_snapshots.append(snap)
                    details = _format_stream_errors(stream_errs)
                    _emit(f"Stream exited early: {details}")
                    stop_reason = "stream_exit"
                    break
                # Preview lifecycle (incl. stale watchdog) runs in
                # _preview_control_loop across the whole session, recording included.
                if now - last_progress_at > 0.5:
                    approx_frames = int(elapsed * cfg.cameras.fps)
                    _emit(f"Recorded {approx_frames} frames for the current episode.")
                    last_progress_at = now
                if box_started:
                    last_box_live_at = _emit_box_live(box, last_emit_s=last_box_live_at)
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

            frame_sync_payload: dict[str, Any] | None = None
            online_sync_payload: dict[str, Any] | None = None
            if (
                decision in ("save", "operator_save", "stream_exit")
                and cfg.cameras.recorder_backend == "argus_metadata"
                and cfg.argus_frame_sync.enabled
            ):
                frame_sync_payload, frame_sync_failure = _evaluate_argus_frame_sync(handle, cfg)
                if frame_sync_failure is not None:
                    details = ""
                    if frame_sync_payload:
                        failures = frame_sync_payload.get("failures") or []
                        details = "; ".join(str(f) for f in failures)
                    logger.error("Argus frame sync gate failed: %s %s", frame_sync_failure, details)
                    _emit(
                        "ERROR: Argus frame sync failed; episode will be discarded. "
                        f"{details or frame_sync_failure}"
                    )
                    decision = frame_sync_failure

            if (
                decision in ("save", "operator_save", "stream_exit")
                and cfg.cameras.recorder_backend == "argus_online_sync"
            ):
                online_sync_payload, online_sync_failure = _evaluate_online_sync_manifest(handle, cfg)
                if online_sync_failure is not None:
                    details = ""
                    if online_sync_payload:
                        failures = online_sync_payload.get("failures") or []
                        details = "; ".join(str(f) for f in failures)
                    logger.error("online sync gate failed: %s %s", online_sync_failure, details)
                    _emit(
                        "ERROR: Online sync failed; episode will be discarded. "
                        f"{details or online_sync_failure}"
                    )
                    decision = online_sync_failure

            if decision in ("save", "operator_save", "stream_exit"):
                meta_path = _write_episode_meta(
                    handle, cfg, locked, argus_failed, connect_errors,
                    box_cfg, box_snapshots, decision, wall_start, wall_end,
                )
                try:
                    payload = json.loads(meta_path.read_text())
                    payload["cleanup_duration_s"] = cleanup_duration_s
                    payload["split_emit_ms"] = split_emit_ms
                    if frame_sync_payload is not None:
                        payload["argus_frame_sync"] = frame_sync_payload
                    if online_sync_payload is not None:
                        payload["online_sync"] = online_sync_payload
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
        # Stop the preview controller before tearing down pcs so it can't call
        # enable/disable/refresh on a session being disconnected underneath it.
        if preview_thread is not None:
            preview_thread.join(timeout=max(2.0, preview_grace_s))
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
