"""GMSL2 recorder configuration and legacy CLI for Thor GMSL2 cameras.

The production GUI entrypoint is ``thor_record.py``. By default the shared
configuration selects the Libargus online-sync backend there, which aligns
same-frame Argus buffers by SOF metadata before hardware encoding. The legacy
standalone CLI in this file still runs one ``gst-launch-1.0`` process per
camera::

    nvarguscamerasrc sensor-id=N do-timestamp=true sensor-mode=M
      ! video/x-raw(memory:NVMM),format=NV12,width=W,height=H,framerate=F/1
      ! nvv4l2h265enc bitrate=... iframeinterval=... preset-level=... control-rate=...
      ! h265parse
      ! matroskamux
      ! filesink location=<dataset>/episodes/episode_NNNNNN/<name>.mkv

The legacy CLI is kept for diagnostics and fallback capture. It does not pull
Libargus frame metadata inline; the production sync paths live in
``argus_online_sync_session.py``, ``argus_metadata_session.py``, and
``thor_record.py``.

Usage:
    cd ~/lerobot
    PYTHONPATH=src:. python -m tools.thor.gmsl2.gmsl2_record \\
        --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml

Press Ctrl+C to stop the current episode and exit cleanly. The first SIGINT
ends the in-flight episode; the second exits the program.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger("gmsl2_record")


# ---------------------------------------------------------------- config ----


@dataclass
class CameraDefaults:
    pipeline: str = "argus"
    recorder_backend: str = "argus_online_sync"
    sensor_mode: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 60
    exposure_us: int = 0
    # Driver-unit V4L2 gain. Keep separate from Argus gainrange, which uses
    # a 0..4 float scale and rejects values such as the AR0234 driver gain 320.
    gain: int = 0
    argus_gain: float = 0.0
    replay_warmup_s: float = 0.0
    codec: str = "h265"
    bitrate_kbps: int = 20000
    iframe_interval: int = 60
    preset_level: int = 1
    control_rate: int = 1
    profile: str = "main"
    container: str = "mkv"

    def __post_init__(self) -> None:
        self.recorder_backend = self.recorder_backend.lower()
        self.codec = self.codec.lower()
        self.container = self.container.lower()
        if self.recorder_backend not in {
            "gstreamer_splitmux",
            "argus_metadata",
            "argus_online_sync",
        }:
            raise ValueError(
                "recorder_backend must be 'gstreamer_splitmux', "
                f"'argus_metadata', or 'argus_online_sync', got {self.recorder_backend!r}"
            )
        if self.codec not in {"h265", "h264"}:
            raise ValueError(f"codec must be 'h265' or 'h264', got {self.codec!r}")
        if self.container not in {"mkv", "mp4"}:
            raise ValueError(f"container must be 'mkv' or 'mp4', got {self.container!r}")


@dataclass
class HardwareSync:
    """How to arm the SG16A hardware trigger before recording.

    The empirically working flow on this board, per the user's bring-up, is
    just::

        cd <sdk_dir> && sudo sh pwm.sh

    (which exports/configures ``pwmchip6/pwm0`` on the current Thor rig).
    ``setup_script`` is interpreted relative to ``sdk_dir`` when it is a
    bare filename; absolute paths are honored as-is. Module loading and
    ``trig_mode=1`` are assumed to already be in place -- if not, run
    ``tools/thor/gmsl2/setup_sync.sh`` once at cold boot.
    """

    enabled: bool = True
    sdk_dir: str = "tools/thor/gmsl2/sdk"
    setup_script: str = "pwm.sh"
    fps: int = 60
    # `sensor_trig_mode` decides what we push to each camera before capture:
    #   * 1 / "slave": frames are gated by the PWM edge. Cross-camera frames
    #     are physically locked to the same edge (sub-microsecond alignment).
    #     This is the default -- the SG16A's PWM-to-trigger routing has been
    #     verified working. NOTE: per-frame `exposure_us` must fit within one
    #     PWM period or the sensor falls back to ~0.8 fps; the recorder
    #     auto-clamps to 0.85 * (1e6 / fps) when this constraint is violated.
    #   * 0 / "free_run": cameras run at the dtbo-locked rate without trigger
    #     gating; alignment is only as good as the pipeline clock.
    sensor_trig_mode: int = 1
    # Fraction of the PWM period above which we refuse to honor the
    # configured exposure_us in slave mode (clamped down + warning logged).
    exposure_period_fraction_max: float = 0.85
    # Informational fields recorded in the episode meta.json.
    pwm_chip: str = "pwmchip6"
    pwm_id: int = 0
    trig_pin: int = 0x00020007


@dataclass
class ArgusFrameSync:
    """Per-frame SOF timestamp synchronization gate.

    The metadata-integrated Libargus backend writes one
    ``<camera>.argus_frame_metadata.csv`` sidecar per episode. When enabled,
    ``thor_record.py`` aligns all cameras by SOF TSC before saving.
    """

    enabled: bool = False
    required: bool = False
    reference_strategy: str = "virtual_cluster"
    reference_camera: str = ""
    tolerance_ms: float = 1.0
    cadence_tolerance_ms: float = 1.0
    report_name: str = "argus_frame_alignment.json"
    materialize_verify: bool = False
    materialize_workers: int = 2

    def __post_init__(self) -> None:
        self.reference_strategy = self.reference_strategy.lower()
        if self.reference_strategy not in {"virtual_cluster", "camera"}:
            raise ValueError(
                "argus_frame_sync.reference_strategy must be "
                f"'virtual_cluster' or 'camera', got {self.reference_strategy!r}"
            )


@dataclass
class ArgusOnlineSync:
    """Online encoder-front synchronization settings.

    ``argus_online_sync`` uses these settings inside the C++ recorder. Frames
    are aligned before they are pushed to the hardware encoder, so post-save
    materialization is not part of this backend's contract.
    """

    enabled: bool = True
    sync_source: str = "sof_tsc_ns"
    tolerance_ms: float = 1.0
    startup_full_clusters: int = 30
    frame_timeout_ms: int = 1000
    preflight_timeout_s: float = 30.0
    single_preflight_timeout_s: float = 10.0
    missing_frame_policy: str = "fail_episode"
    stop_mode: str = "full_cluster"
    frame_bus_dir: str = ""
    frame_bus_every_n: int = 1

    def __post_init__(self) -> None:
        self.sync_source = str(self.sync_source).lower()
        self.missing_frame_policy = str(self.missing_frame_policy).lower()
        self.stop_mode = str(self.stop_mode).lower()
        self.frame_bus_dir = str(self.frame_bus_dir)
        self.frame_timeout_ms = int(self.frame_timeout_ms)
        self.preflight_timeout_s = float(self.preflight_timeout_s)
        self.single_preflight_timeout_s = float(self.single_preflight_timeout_s)
        self.frame_bus_every_n = int(self.frame_bus_every_n)
        if self.sync_source != "sof_tsc_ns":
            raise ValueError(
                "online_sync.sync_source currently supports only 'sof_tsc_ns', "
                f"got {self.sync_source!r}"
            )
        if self.frame_timeout_ms <= 0:
            raise ValueError(
                "online_sync.frame_timeout_ms must be > 0, "
                f"got {self.frame_timeout_ms!r}"
            )
        if self.preflight_timeout_s <= 0:
            raise ValueError(
                "online_sync.preflight_timeout_s must be > 0, "
                f"got {self.preflight_timeout_s!r}"
            )
        if self.single_preflight_timeout_s <= 0:
            raise ValueError(
                "online_sync.single_preflight_timeout_s must be > 0, "
                f"got {self.single_preflight_timeout_s!r}"
            )
        if self.frame_bus_every_n <= 0:
            raise ValueError(
                "online_sync.frame_bus_every_n must be > 0, "
                f"got {self.frame_bus_every_n!r}"
            )
        if self.missing_frame_policy != "fail_episode":
            raise ValueError(
                "online_sync.missing_frame_policy currently supports only "
                f"'fail_episode', got {self.missing_frame_policy!r}"
            )
        if self.stop_mode != "full_cluster":
            raise ValueError(
                "online_sync.stop_mode currently supports only 'full_cluster', "
                f"got {self.stop_mode!r}"
            )


@dataclass
class RecorderConfig:
    cameras: CameraDefaults
    hardware_sync: HardwareSync
    argus_frame_sync: ArgusFrameSync
    online_sync: ArgusOnlineSync
    repo_id: str
    single_task: str
    dataset_root: Path
    fps: int
    num_episodes: int
    episode_time_s: float
    detect_all: bool
    sensor_ids: list[int]
    name_prefix: str
    spawn_stagger_s: float
    connect_stable_s: float
    connect_timeout_s: float
    connect_first_fragment_timeout_s: float
    two_phase_connect: bool
    stop_on_stream_exit: bool
    recording_preview_enabled: bool
    recording_preview_on_demand: bool
    recording_preview_idle_ttl_s: float
    recording_preview_stagger_s: float
    recording_preview_stale_s: float
    recording_preview_watchdog_s: float
    stream_health_poll_s: float
    # While armed-but-idle, roll + prune the throwaway warmup fragments every
    # this many seconds so the _warmup dir can't fill the disk between episodes
    # (0 disables idle maintenance). keep_last_n bounds how many fragments per
    # camera survive each prune.
    warmup_roll_s: float
    warmup_keep_last_n: int


def _resolve(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _filter_kwargs(cls: type, src: dict) -> dict:
    fields = getattr(cls, "__annotations__", {}).keys()
    return {k: v for k, v in src.items() if k in fields}


def load_config(path: Path) -> RecorderConfig:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    sensors = raw.get("sensors", {}) or {}
    cams = sensors.get("cameras", {}) or {}
    defaults_raw = cams.get("defaults", {}) or {}
    defaults = CameraDefaults(**_filter_kwargs(CameraDefaults, defaults_raw))

    hw_raw = sensors.get("hardware_sync", {}) or {}
    if isinstance(hw_raw.get("trig_pin"), str):
        hw_raw["trig_pin"] = int(hw_raw["trig_pin"], 0)
    hwsync = HardwareSync(**_filter_kwargs(HardwareSync, hw_raw))
    frame_sync_raw = cams.get("argus_frame_sync", {}) or {}
    frame_sync = ArgusFrameSync(**_filter_kwargs(ArgusFrameSync, frame_sync_raw))
    online_sync_raw = cams.get("online_sync", {}) or {}
    online_sync = ArgusOnlineSync(**_filter_kwargs(ArgusOnlineSync, online_sync_raw))

    ds = raw.get("dataset", {}) or {}

    return RecorderConfig(
        cameras=defaults,
        hardware_sync=hwsync,
        argus_frame_sync=frame_sync,
        online_sync=online_sync,
        repo_id=str(ds.get("repo_id", "local/gmsl2")),
        single_task=str(ds.get("single_task", "GMSL2 capture")),
        dataset_root=_resolve(ds.get("root", "outputs/datasets/gmsl2")),
        fps=int(ds.get("fps", defaults.fps)),
        num_episodes=int(ds.get("num_episodes", 0)),
        episode_time_s=float(ds.get("episode_time_s", 10)),
        detect_all=bool(cams.get("detect_all", True)),
        sensor_ids=list(cams.get("sensor_ids", []) or []),
        name_prefix=str(cams.get("name_prefix", "cam")),
        spawn_stagger_s=float(cams.get("spawn_stagger_s", 0.0)),
        connect_stable_s=float(cams.get("connect_stable_s", 2.0)),
        connect_timeout_s=float(cams.get("connect_timeout_s", 120.0)),
        connect_first_fragment_timeout_s=float(cams.get("connect_first_fragment_timeout_s", 0.0)),
        two_phase_connect=bool(cams.get("two_phase_connect", False)),
        stop_on_stream_exit=bool(cams.get("stop_on_stream_exit", True)),
        recording_preview_enabled=bool(cams.get("recording_preview_enabled", True)),
        recording_preview_on_demand=bool(cams.get("recording_preview_on_demand", True)),
        recording_preview_idle_ttl_s=float(cams.get("recording_preview_idle_ttl_s", 6.0)),
        recording_preview_stagger_s=float(cams.get("recording_preview_stagger_s", 0.5)),
        recording_preview_stale_s=float(cams.get("recording_preview_stale_s", 3.0)),
        recording_preview_watchdog_s=float(cams.get("recording_preview_watchdog_s", 2.0)),
        stream_health_poll_s=float(cams.get("stream_health_poll_s", 1.0)),
        warmup_roll_s=float(ds.get("warmup_roll_s", 30.0)),
        warmup_keep_last_n=int(ds.get("warmup_keep_last_n", 3)),
    )


# ---------------------------------------------------------------- detect ----


def detect_locked_sids(cfg: RecorderConfig, repo_root: Path) -> list[int]:
    if not cfg.detect_all:
        return list(cfg.sensor_ids)
    script = repo_root / "tools" / "thor" / "gmsl2" / "check_max96726_locks.sh"
    if not script.exists():
        raise FileNotFoundError(f"missing lock-check script: {script}")
    logger.info("running lock-check: %s", script)
    p = subprocess.run([str(script)], capture_output=True, text=True, timeout=30)
    if p.returncode not in (0, 1):  # 1 = no locks, still informative
        raise RuntimeError(
            f"lock check failed: rc={p.returncode}\nstdout={p.stdout}\nstderr={p.stderr}"
        )
    for line in p.stdout.splitlines():
        if line.startswith("LOCKED_VIDEO_IDS="):
            csv = line.split("=", 1)[1].strip()
            return [int(x) for x in csv.split(",") if x.strip()] if csv else []
    raise RuntimeError("lock-check script did not emit LOCKED_VIDEO_IDS=")


def probe_argus(sids: list[int], width: int, height: int, fps: int,
                timeout_s: float = 8) -> tuple[list[int], list[int]]:
    """For each sid, try a tiny nvarguscamerasrc capture. Returns (ok, fail)."""
    ok: list[int] = []
    fail: list[int] = []
    for sid in sids:
        caps = (
            f"video/x-raw(memory:NVMM),format=NV12,"
            f"width={width},height={height},framerate={fps}/1"
        )
        cmd = [
            "gst-launch-1.0", "-q",
            "nvarguscamerasrc", f"sensor-id={sid}", "num-buffers=10",
            "!", caps,
            "!", "fakesink", "sync=false", "async=false",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        try:
            stdout, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            fail.append(sid)
            logger.warning("sid=%s argus probe timed out (killed pid %s)", sid, proc.pid)
            continue
        if proc.returncode == 0:
            ok.append(sid)
        else:
            fail.append(sid)
            tail = (stdout or "").strip().splitlines()[-1:][:1]
            logger.warning("sid=%s argus probe failed: %s",
                           sid, tail[0][:120] if tail else "no output")
    return ok, fail


# ---------------------------------------------------------------- gst ----


def _enc_element(c: CameraDefaults) -> list[str]:
    bitrate_bps = int(c.bitrate_kbps) * 1000
    if c.codec == "h265":
        return [
            "nvv4l2h265enc",
            f"bitrate={bitrate_bps}",
            f"iframeinterval={c.iframe_interval}",
            f"preset-level={c.preset_level}",
            f"control-rate={c.control_rate}",
            "insert-sps-pps=1",
        ]
    return [
        "nvv4l2h264enc",
        f"bitrate={bitrate_bps}",
        f"iframeinterval={c.iframe_interval}",
        f"preset-level={c.preset_level}",
        f"control-rate={c.control_rate}",
        "insert-sps-pps=1",
    ]


def _parser_element(c: CameraDefaults) -> str:
    return "h265parse" if c.codec == "h265" else "h264parse"


def _muxer_element(c: CameraDefaults) -> str:
    return "matroskamux" if c.container == "mkv" else "qtmux"


def build_pipeline(sid: int, out_path: Path, c: CameraDefaults) -> list[str]:
    caps = (
        f"video/x-raw(memory:NVMM),format=NV12,"
        f"width={c.width},height={c.height},framerate={c.fps}/1"
    )
    cmd: list[str] = [
        "gst-launch-1.0", "-e",
        "nvarguscamerasrc",
        f"sensor-id={sid}",
        f"sensor-mode={c.sensor_mode}",
        "do-timestamp=true",
    ]
    if c.exposure_us > 0:
        cmd += [f"exposuretimerange={c.exposure_us * 1000} {c.exposure_us * 1000}"]
    if c.argus_gain > 0:
        if c.argus_gain > 4.0:
            raise ValueError(
                f"argus_gain must be <= 4.0 for nvarguscamerasrc, got {c.argus_gain}"
            )
        cmd += [f"gainrange={c.argus_gain:g} {c.argus_gain:g}"]
    cmd += ["!", caps, "!"]
    cmd += _enc_element(c)
    cmd += ["!", _parser_element(c)]
    cmd += ["!", _muxer_element(c)]
    cmd += ["!", "filesink", f"location={out_path}", "sync=false"]
    return cmd


# ---------------------------------------------------------------- session ----


@dataclass
class CameraStream:
    sid: int
    name: str
    file: Path
    log_file: Path
    cmd: list[str]
    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    stopped_at: float = 0.0
    exit_code: int | None = None


@dataclass
class EpisodeResult:
    index: int
    directory: Path
    wallclock_start_utc: str
    wallclock_end_utc: str
    duration_s: float
    streams: list[CameraStream] = field(default_factory=list)
    t0_wall_s: float = 0.0
    t0_mono_s: float = 0.0


class EpisodeSession:
    """One per-episode batch of N gst-launch subprocesses."""

    def __init__(self, sids: list[int], cfg: RecorderConfig):
        self.sids = sids
        self.cfg = cfg

    def start(self, episode_dir: Path) -> list[CameraStream]:
        episode_dir.mkdir(parents=True, exist_ok=True)
        streams: list[CameraStream] = []
        t0 = time.time()
        for i, sid in enumerate(self.sids):
            if i > 0 and self.cfg.spawn_stagger_s > 0:
                time.sleep(self.cfg.spawn_stagger_s)
            name = f"{self.cfg.name_prefix}_{sid:02d}"
            ext = self.cfg.cameras.container
            file_path = episode_dir / f"{name}.{ext}"
            log_path = episode_dir / f"{name}.gst.log"
            cmd = build_pipeline(sid, file_path, self.cfg.cameras)
            logger.info("[%s] spawn +%.3fs -> %s", name, time.time() - t0, file_path)
            logger.debug("[%s] cmd: %s", name, shlex.join(cmd))
            log_fh = open(log_path, "w")
            log_fh.write("# " + shlex.join(cmd) + "\n")
            log_fh.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            streams.append(CameraStream(
                sid=sid, name=name, file=file_path, log_file=log_path,
                cmd=cmd, proc=proc, started_at=time.time(),
            ))
        # Give the encoders a moment to actually open the camera.
        time.sleep(1.0)
        logger.info("all %d streams spawned in %.3fs", len(streams), time.time() - t0)
        return streams

    def stop(self, streams: list[CameraStream], grace_s: float = 8.0) -> None:
        # SIGINT lets `gst-launch -e` emit EOS so muxers finalize cleanly.
        for s in streams:
            if s.proc and s.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(s.proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
        for s in streams:
            if not s.proc:
                continue
            try:
                s.exit_code = s.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] SIGINT ignored after %.1fs; SIGTERM", s.name, grace_s)
                try:
                    os.killpg(os.getpgid(s.proc.pid), signal.SIGTERM)
                    s.exit_code = s.proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(s.proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    s.exit_code = -9
            s.stopped_at = time.time()


# ---------------------------------------------------------------- meta ----


def write_episode_meta(
    ep: EpisodeResult,
    cfg: RecorderConfig,
    locked_sids: list[int],
    argus_failed_sids: list[int],
) -> Path:
    meta = {
        "episode_index": ep.index,
        "repo_id": cfg.repo_id,
        "single_task": cfg.single_task,
        "wallclock_start_utc": ep.wallclock_start_utc,
        "wallclock_end_utc": ep.wallclock_end_utc,
        "duration_s": ep.duration_s,
        "video": {
            "fps": cfg.cameras.fps,
            "width": cfg.cameras.width,
            "height": cfg.cameras.height,
            "codec": cfg.cameras.codec,
            "container": cfg.cameras.container,
            "bitrate_kbps": cfg.cameras.bitrate_kbps,
            "iframe_interval": cfg.cameras.iframe_interval,
            "preset_level": cfg.cameras.preset_level,
            "control_rate": cfg.cameras.control_rate,
            "color_format": "NV12 (YUV420)",
            "replay_warmup_s": float(cfg.cameras.replay_warmup_s),
            "pipeline": "nvarguscamerasrc | nvv4l2h{265,264}enc | matroskamux | filesink",
        },
        "hardware_sync": {
            "enabled": cfg.hardware_sync.enabled,
            "pwm_chip": cfg.hardware_sync.pwm_chip,
            "pwm_id": cfg.hardware_sync.pwm_id,
            "pwm_fps": cfg.hardware_sync.fps,
            "trig_pin": f"0x{cfg.hardware_sync.trig_pin:08x}",
        },
        "sync_reference": {
            "t0_wall_s": ep.t0_wall_s,
            "t0_mono_s": ep.t0_mono_s,
            "camera_spawn_wall_s": {
                s.name: s.started_at for s in ep.streams
            },
            "camera_spawn_offset_s": {
                s.name: s.started_at - ep.t0_wall_s for s in ep.streams
            },
            "note": (
                "t0_wall_s is time.time() at episode start (before first camera spawn). "
                "camera_spawn_offset_s is each camera's gst-launch spawn time relative to t0. "
                "MKV PTS tracks start from the pipeline clock origin; subtract spawn_offset "
                "to align all cameras to t0. BOX snapshots carry t_relative_s = time.time() - t0 "
                "and per-sensor MCU timestamps. Use t_relative_s for camera-BOX alignment."
            ),
        },
        "max96726_locked_sids": locked_sids,
        "argus_failed_sids": argus_failed_sids,
        "spawn_stagger_s": cfg.spawn_stagger_s,
        "stop_on_stream_exit": cfg.stop_on_stream_exit,
        "cameras": [
            {
                "sensor_id": s.sid,
                "name": s.name,
                "file": s.file.name,
                "log_file": s.log_file.name,
                "started_wallclock_s": s.started_at,
                "stopped_wallclock_s": s.stopped_at,
                "duration_s": (s.stopped_at - s.started_at) if s.stopped_at else None,
                "exit_code": s.exit_code,
            }
            for s in ep.streams
        ],
    }
    meta_path = ep.directory / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


# ---------------------------------------------------------------- main ----


def _resolve_sdk_path(sdk_dir: str, repo_root: Path) -> Path:
    """Resolve `sdk_dir` against the repo root for relative paths.

    Absolute paths and ``~`` are honored as-is.
    """
    p = Path(os.path.expanduser(sdk_dir))
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _clamp_exposure_for_pwm_period(cfg: RecorderConfig) -> None:
    """Cap ``cameras.exposure_us`` so the sensor doesn't miss PWM triggers.

    AR0234 in slave mode (`trig_mode=1`) requires `exposure + readout` to fit
    within one PWM period. Going over makes the sensor drop to ~0.8 fps. We
    clamp to `exposure_period_fraction_max * period_us` and warn loudly.

    No-op in free-run mode -- the sensor uses its own internal timing.
    """
    if cfg.hardware_sync.sensor_trig_mode != 1:
        return
    if cfg.cameras.exposure_us <= 0:
        return
    fps = max(1, int(cfg.hardware_sync.fps))
    period_us = 1_000_000 // fps
    frac = float(cfg.hardware_sync.exposure_period_fraction_max)
    cap = int(period_us * frac)
    if cfg.cameras.exposure_us > cap:
        logger.warning(
            "exposure_us=%d exceeds %.0f%% of PWM period (%d us @ %d Hz); "
            "clamping to %d to avoid AR0234 trigger miss",
            cfg.cameras.exposure_us, frac * 100, period_us, fps, cap,
        )
        cfg.cameras.exposure_us = cap


def apply_per_camera_controls(
    sids: list[int],
    cameras: CameraDefaults,
    trig_mode: int,
    trig_pin: int,
) -> None:
    """Push trig_mode / exposure / gain to each selected /dev/videoN via v4l2-ctl.

    This puts the cameras into the deterministic state that the user verified
    works for HW recording (``trig_mode=0`` + explicit exposure/gain).
    """
    for sid in sids:
        ctrls = [
            f"sensor_mode={cameras.sensor_mode}",
            f"trig_pin=0x{trig_pin:08x}",
            f"trig_mode={int(trig_mode)}",
        ]
        if cameras.exposure_us > 0:
            ctrls.append(f"exposure={int(cameras.exposure_us)}")
        if cameras.gain > 0:
            ctrls.append(f"gain={int(cameras.gain)}")
        cmd = ["sudo", "-n", "v4l2-ctl", "-d", f"/dev/video{sid}", "-c", ",".join(ctrls)]
        logger.info("v4l2-ctl /dev/video%d -c %s", sid, ",".join(ctrls))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            logger.warning("v4l2-ctl on sid=%d failed: rc=%s stderr=%s",
                           sid, r.returncode, r.stderr.strip())


def maybe_setup_sync(cfg: RecorderConfig, repo_root: Path) -> None:
    """Run the SDK's hardware-trigger script (defaults to ``pwm.sh``).

    The script is launched with ``sudo sh <path>`` from inside ``sdk_dir`` so
    relative paths inside the SDK resolve correctly. We export ``FPS`` so the
    vendored ``pwm.sh`` programs the requested frequency.
    """
    if not cfg.hardware_sync.enabled:
        return
    sdk = _resolve_sdk_path(cfg.hardware_sync.sdk_dir, repo_root)
    script_field = Path(os.path.expanduser(cfg.hardware_sync.setup_script))
    if script_field.is_absolute():
        script_path = script_field
        cwd = script_path.parent
    else:
        script_path = sdk / script_field
        cwd = sdk
    if not script_path.exists():
        raise FileNotFoundError(
            f"hardware_sync setup script not found: {script_path} "
            f"(sdk_dir={sdk}, setup_script={cfg.hardware_sync.setup_script})"
        )
    fps_env = str(int(cfg.hardware_sync.fps))
    logger.info(
        "hardware_sync.enabled=true; running sudo -E sh %s (cwd=%s, FPS=%s)",
        script_path, cwd, fps_env,
    )
    env = {**os.environ, "FPS": fps_env}
    r = subprocess.run(
        ["sudo", "-nE", "sh", str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"hardware_sync setup_script failed: rc={r.returncode}\n"
            f"stdout={r.stdout}\nstderr={r.stderr}"
        )
    if r.stdout.strip():
        logger.info("pwm: %s", r.stdout.strip().splitlines()[-1])


def _legacy_cli_backend_error(cfg: RecorderConfig) -> str | None:
    """Return an error when this legacy CLI cannot honor the configured backend."""

    if cfg.cameras.recorder_backend == "gstreamer_splitmux":
        return None
    return (
        "tools.thor.gmsl2.gmsl2_record is the legacy standalone "
        "GStreamer/splitmux CLI and cannot run "
        f"recorder_backend={cfg.cameras.recorder_backend!r}. Use "
        "tools/thor/gmsl2/thor_record.py for the production "
        "metadata-integrated recorder, or set "
        "sensors.cameras.defaults.recorder_backend: gstreamer_splitmux "
        "for this legacy CLI."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GMSL2 hardware-encoded multi-camera recorder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-path", required=True, type=Path)
    ap.add_argument("--num-episodes", type=int, default=None,
                    help="override dataset.num_episodes (0 = unlimited)")
    ap.add_argument("--episode-time-s", type=float, default=None,
                    help="override dataset.episode_time_s")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--sensor-ids", type=str, default=None,
                    help="comma-separated sids to record; overrides detection")
    ap.add_argument("--spawn-stagger-s", type=float, default=None,
                    help="delay between gst-launch spawns; useful when Argus "
                         "stalls during many simultaneous camera opens")
    ap.add_argument("--ignore-stream-exit", action="store_true",
                    help="diagnostic mode: keep the episode running even if "
                         "one gst-launch stream exits early")
    ap.add_argument("--skip-argus-probe", action="store_true",
                    help="trust the MAX96726 lock list, skip nvargus open-probe")
    ap.add_argument("--skip-hardware-sync", action="store_true",
                    help="do not run the hardware_sync.setup_script even if the "
                         "config enables it; useful when the PWM wire to the "
                         "SG16A trigger pin is not in place and you need to "
                         "record in free-run mode")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_config(args.config_path)
    if args.num_episodes is not None:
        cfg.num_episodes = args.num_episodes
    if args.episode_time_s is not None:
        cfg.episode_time_s = args.episode_time_s
    if args.sensor_ids:
        cfg.detect_all = False
        cfg.sensor_ids = [int(x) for x in args.sensor_ids.split(",") if x.strip()]
    if args.spawn_stagger_s is not None:
        cfg.spawn_stagger_s = max(0.0, args.spawn_stagger_s)
    if args.ignore_stream_exit:
        cfg.stop_on_stream_exit = False

    backend_error = _legacy_cli_backend_error(cfg)
    if backend_error is not None:
        logger.error(backend_error)
        return 2

    repo_root = args.repo_root.resolve()
    if args.skip_hardware_sync:
        logger.warning("--skip-hardware-sync set; not arming PWM trigger source")
        cfg.hardware_sync.enabled = False
    maybe_setup_sync(cfg, repo_root)

    locked = detect_locked_sids(cfg, repo_root)
    logger.info("MAX96726 locked sids: %s", locked)
    if not locked:
        logger.error("no locked cameras detected; aborting")
        return 1

    if args.skip_argus_probe:
        usable = list(locked)
        argus_failed: list[int] = []
    else:
        usable, argus_failed = probe_argus(
            locked, cfg.cameras.width, cfg.cameras.height, cfg.cameras.fps,
        )
        logger.info("argus-streamable sids: %s ; argus-failing sids: %s", usable, argus_failed)
    if not usable:
        logger.error("no cameras stream through nvargus; aborting")
        return 1

    # Push the deterministic per-camera control state (trig_mode + exposure +
    # gain) -- skipped when hardware_sync is disabled to leave the kernel
    # driver's defaults in place.
    if cfg.hardware_sync.enabled:
        _clamp_exposure_for_pwm_period(cfg)
        apply_per_camera_controls(
            usable, cfg.cameras,
            trig_mode=cfg.hardware_sync.sensor_trig_mode,
            trig_pin=cfg.hardware_sync.trig_pin,
        )

    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    logger.info("dataset root: %s", cfg.dataset_root)

    stop_requested = False
    finish_requested = False

    def _sigint(_signum, _frame):
        nonlocal stop_requested, finish_requested
        if not stop_requested:
            logger.info("SIGINT: ending current episode")
            stop_requested = True
        else:
            logger.info("SIGINT (2): exiting after this episode")
            finish_requested = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    ep_idx = _next_episode_index(cfg.dataset_root)
    while not finish_requested:
        if cfg.num_episodes and ep_idx >= cfg.num_episodes:
            break
        ep_dir = cfg.dataset_root / "episodes" / f"episode_{ep_idx:06d}"
        session = EpisodeSession(usable, cfg)
        wall_start = datetime.now(timezone.utc).isoformat()
        t0_wall = time.time()
        t0_mono = time.monotonic()
        streams = session.start(ep_dir)
        logger.info("episode %d started @ %s -> %s", ep_idx, wall_start, ep_dir)
        capture_start = time.time()
        end_at = capture_start + cfg.episode_time_s if cfg.episode_time_s > 0 else float("inf")
        stop_requested = False
        reported_dead: set[str] = set()
        while time.time() < end_at and not stop_requested:
            dead = [s for s in streams if s.proc and s.proc.poll() is not None]
            if dead:
                new_dead = [s.name for s in dead if s.name not in reported_dead]
                if new_dead:
                    logger.warning("%d streams exited early: %s",
                                   len(dead), [s.name for s in dead])
                    reported_dead.update(new_dead)
                if cfg.stop_on_stream_exit:
                    break
            time.sleep(0.1)
        session.stop(streams)
        wall_end = datetime.now(timezone.utc).isoformat()
        ep = EpisodeResult(
            index=ep_idx,
            directory=ep_dir,
            wallclock_start_utc=wall_start,
            wallclock_end_utc=wall_end,
            duration_s=time.time() - t0_wall,
            streams=streams,
            t0_wall_s=t0_wall,
            t0_mono_s=t0_mono,
        )
        meta_path = write_episode_meta(ep, cfg, locked, argus_failed)
        ok_count = sum(1 for s in streams if s.exit_code in (0, None))
        logger.info("episode %d saved (%s); %d/%d streams cleanly exited; meta=%s",
                    ep_idx, ep_dir, ok_count, len(streams), meta_path)
        ep_idx += 1

    return 0


def _next_episode_index(dataset_root: Path) -> int:
    eps_dir = dataset_root / "episodes"
    if not eps_dir.exists():
        return 0
    indices = []
    for child in eps_dir.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            try:
                indices.append(int(child.name.split("_", 1)[1]))
            except ValueError:
                continue
    return (max(indices) + 1) if indices else 0


if __name__ == "__main__":
    sys.exit(main())
