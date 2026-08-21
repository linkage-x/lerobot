#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import ipaddress
import json
import math
import os
import queue
import re
import select
import signal
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
import dataclasses
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, Thread
from typing import Any, Iterable, Sequence
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from tools.data_collection_gui import checkpoints as checkpoint_backend
from tools.data_collection_gui import rollout as rollout_backend
from tools.data_collection_gui import training as training_backend

DEFAULT_CONFIG_PATH = Path("tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml")
DEFAULT_RECORDER_SCRIPT = Path("tools/handheld/handheld_record.py")
# Gateway-driven FR3 SpaceMouse recorder. Handles both the hardware arm and its MuJoCo twin
# behind one `--backend` switch, so both produce byte-identical dataset schemas.
WORKSTATION_RECORDER_SCRIPT = Path("tools/fr3/fr3_gui_record_runtime.py")
RECORD_BACKENDS = ("real", "sim")
DEFAULT_RECORD_BACKEND = "real"
# Action contracts the workstation Training View page can build. Recording always stores
# absolute EE; the delta contracts are derived offline from consecutive dataset frames.
TRAINING_VIEW_ACTION_MODES = ("absolute_ee", "delta_ee_from_prev_cmd", "delta_ee_from_current")
DEFAULT_TRAINING_VIEW_ACTION_MODE = "delta_ee_from_prev_cmd"
# Rates a training view may be built at. 30 is the default and the rate the FR3 baseline was
# recorded at, so views built from newer 60 fps sessions land on the same per-frame action scale
# and can be merged with it. 0 means "whatever the source is", which then requires every source
# in one build to already agree -- offered because a rig that only ever records at one rate has
# no reason to resample, not because mixing rates is ever safe.
TRAINING_VIEW_FPS_CHOICES = (30, 15, 60, 0)
DEFAULT_TRAINING_VIEW_FPS = 30
# How often a running job's progress is checkpointed to disk. Long enough that a job logging
# thirteen lines a second is not writing a file thirteen times a second; short enough that a
# gateway restarted mid-run reports a step count from the same minute.
TRAINING_STATUS_PERSIST_INTERVAL_S = 10.0
DEFAULT_DATASETS_ROOT = Path("outputs/datasets")
DEFAULT_EXPORTS_ROOT = Path("outputs/exports")
# Training views are grouped one level below the exports root. The grouping directory is not
# itself a dataset, so every scan that walks the exports root has to descend into it explicitly.
TRAINING_VIEWS_DIR_NAME = "training_views"
DEPLOYMENT_PROFILES: dict[str, dict[str, Any]] = {
    "thor": {
        "label": "Thor Acquisition",
        "defaultRoute": "live-record",
        "capabilities": ["gmsl2", "box", "imu", "tactile", "force_torque", "recording"],
    },
    "workstation": {
        "label": "FR3 Teleoperation Workstation",
        "defaultRoute": "teleoperation",
        "capabilities": ["fr3", "pika", "spacemouse", "realsense", "mujoco", "recording"],
    },
}
DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM = 20.0
DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG = 15.0
DEFAULT_WORKSTATION_REPLAY_IK_ORIENTATION_WEIGHT = 0.012
DEFAULT_WORKSTATION_REAL_SETTLE_STEPS = 300
DEFAULT_WORKSTATION_REAL_SETTLE_TOLERANCE_MM = 6.0
DEFAULT_REPLAY_MAX_EE_STEP_MM = 120.0
DEFAULT_REPLAY_MAX_GRIPPER_STEP = 0.35
DEFAULT_WORKSTATION_REPLAY_MAX_GRIPPER_STEP = 1.0
DEFAULT_REAL_PREFLIGHT_TIMEOUT_S = 30.0
DEFAULT_CUBE_REPLAY_ROBOT_IP = "192.168.11.102"
# EE trajectory generation now tracks gmsl2 (Thor) datasets with AprilTag cubes
# instead of the legacy Hikon-camera route. The gateway runs on Thor, so it
# invokes the local runner directly (no SSH / copy-back) -- the runner picks the
# opencv_kalibr venv that actually has cv2/pupil_apriltags and wires PYTHONPATH.
DEFAULT_EE_TRAJECTORY_RUNNER = Path("third_party/opencv_kalibr/run_april_cube_tracking_local.sh")
DEFAULT_EE_TRAJECTORY_CONFIG = Path(
    "third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml"
)
# Algorithm id + on-disk sidecar/analysis names produced by the april thor config
# (save_to_dataset.sidecar_dir and output.run_name_suffix). Kept in sync here so
# the gateway reads back exactly what the tracker writes.
DEFAULT_EE_TRAJECTORY_ALGORITHM = "april_cube_tracking_in_robot_base"
DEFAULT_TRAJ_SIDECAR_NAME = "april_cube_tracking_in_robot_base"
DEFAULT_TRACKING_RUN_SUFFIX = "_thor_april_tracking_in_robot_base"
DEFAULT_CUBE_TRAJECTORY_NAMES = ("left", "right", "head")
DEFAULT_MUJOCO_CUBE_MODE = "left"
MUJOCO_CUBE_MODES = ("left", "right", "both")
DEFAULT_MUJOCO_ROBOT_SPACING_M = 0.9
DEFAULT_CUBE_SIZE_M = 0.07
CUBE_OVERLAY_COLORS = {
    "left": "#c2410c",
    "right": "#0f766e",
    "head": "#2563eb",
}


@dataclass
class EventLogItem:
    id: str
    time: str
    level: str
    message: str


_RECORDER_OUTPUT_RING_CAP = 300


@dataclass
class RecordingStatus:
    state: str = "idle"
    datasetRoot: str = ""
    repoId: str = ""
    episodeIndex: int = 0
    savedEpisodes: int = 0
    frameIndex: int = 0
    targetFrames: int = 300
    queueDepth: int = 0
    message: str = "Gateway ready"
    pid: int | None = None
    lastOutput: str = ""
    # Ring buffer of recent recorder stdout lines. Bounded to
    # _RECORDER_OUTPUT_RING_CAP entries so the frontend's polling snapshot
    # can show every line the recorder emitted between polls, instead of
    # only the most recent one (which is what lastOutput captures and is
    # what was losing 9+ lines per Connect cycle).
    recentOutput: list[str] = field(default_factory=list)
    # Workstation profile only: which robot the recorder is driving ("real" or "sim").
    backend: str = DEFAULT_RECORD_BACKEND
    # Latest per-episode timestamp-synchronisation verdict parsed from the recorder's SYNC
    # lines. Kept on the recording status (not a separate poll) so the operator sees an
    # alignment problem on the episode that caused it, while the rig is still set up.
    syncStatus: str = "unknown"
    syncSummary: str = ""
    syncReportPath: str = ""
    syncWarnings: list[str] = field(default_factory=list)


@dataclass
class ReplayStatus:
    state: str = "idle"
    dataset: str = ""
    episode: int = 0
    frameIndex: int = 0
    totalFrames: int = 300
    fps: int = 30
    trackingErrorMm: float = 0.0
    safety: str = "locked"
    message: str = "Replay gateway ready"
    datasetRoot: str = ""
    datasetKind: str = "recorded"
    sourcePath: str = ""
    dataStatus: str = "missing"
    trajectoryKind: str = "none"
    totalEpisodes: int = 0
    episodeOptions: list[int] = field(default_factory=list)
    recordedFrames: int = 0
    diagnostics: list[str] = field(default_factory=list)
    pid: int | None = None
    lastOutput: str = ""
    mujocoCubeMode: str = DEFAULT_MUJOCO_CUBE_MODE
    mujocoValidation: dict[str, Any] = field(default_factory=dict)
    realCubeMode: str = "right"
    realRobotIp: str = ""
    realEndEffectorMode: str = "corenetic_gripper_ee"
    mujocoOverrideAccepted: bool = False
    realReplayLog: list[str] = field(default_factory=list)
    # Bumped whenever the on-disk dataset content changes under a stable
    # (datasetRoot, episode) selection — e.g. deleting an episode keeps the
    # selection on the same slot but swaps in different frames/videos. The UI
    # keys its timeline fetch on this so it refetches instead of showing stale
    # data for the now-different episode.
    revision: int = 0


@dataclass
class TeleopStatus:
    state: str = "idle"
    backend: str = "mujoco"
    inputDevice: str = "spacemouse"
    robotModel: str = "fr3_pika_gripper"
    urdfPath: str = ""
    simXmlPath: str = ""
    # Filled from the config on every snapshot, so this is only what an unconstructed status
    # would claim. Deliberately empty rather than a frame name: a stale default here is what
    # made the idle Teleoperation page name the wrong tool frame.
    targetFrameName: str = ""
    pid: int | None = None
    message: str = "FR3 Pika MuJoCo teleop is ready"
    lastOutput: str = ""
    command: list[str] = field(default_factory=list)
    realRobotReady: bool = False
    cameraViews: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "external", "label": "External", "source": "D435I", "fps": 30, "deviceId": "side"},
            {"id": "wrist", "label": "Wrist", "source": "D405", "fps": 30, "deviceId": "ee"},
        ]
    )


@dataclass
class DatasetExportStatus:
    state: str = "idle"  # idle | exporting | complete | error
    target: str = "lerobot_v3"
    datasetRoot: str = ""
    # Every recording that went into the running build. `datasetRoot` keeps the first one so the
    # existing single-source readers still work, but a merge that reported only its first source
    # would describe a training set that is mostly not what it names.
    datasetRoots: list[str] = field(default_factory=list)
    outputPath: str = ""
    selectedEpisodes: int = 0
    totalFrames: int = 0
    includeRaw: bool = True
    includeDebug: bool = False
    includeTraining: bool = True
    message: str = "Select a task to consolidate its sessions into one v3 dataset"
    manifest: list[str] = field(default_factory=list)
    pid: int | None = None
    taskId: str = ""


@dataclass
class CalibrationStatus:
    state: str = "idle"
    pattern: str = "ChArUco 12x9 · 30 mm (charuco_400)"
    lastRunAt: str = ""
    message: str = "Run calibration to refresh extrinsics"
    cameras: list[dict[str, Any]] = field(default_factory=list)
    outputPath: str = ""
    # Which calibration runs production is currently pointed at. The self-check
    # records these with its baseline: a baseline that outlives the calibration
    # it was taken against compares against a rig that no longer exists.
    intrinsicsRun: str = ""
    extrinsicsRun: str = ""


@dataclass
class CalibrationStep:
    """One capture in the guided calibration.

    Intrinsics need one sweep per camera because the binding constraint is how
    much of *that* camera's frame the board reaches; extrinsics need a single
    sweep that several cameras see at once, because the constraint is
    co-visibility. They cannot share a capture.
    """

    kind: str  # "intrinsics" | "extrinsics"
    camera: str  # "" for the extrinsics step
    status: str = "pending"  # pending | recording | captured | skipped
    episodeIndex: int = -1
    note: str = ""


@dataclass
class CalibrationSession:
    active: bool = False
    stage: str = "idle"  # idle | capture | ready | solving | done | failed
    datasetName: str = ""
    datasetRoot: str = ""
    steps: list[CalibrationStep] = field(default_factory=list)
    currentIndex: int = 0
    message: str = ""


@dataclass
class MarkerTcpSample:
    id: str
    side: str
    condition: str
    source: str = "recording"  # recording | static_transform
    status: str = "pending"  # pending | recording | saved | discarded | registered
    datasetRoot: str = ""
    episodeIndex: int = -1
    staticTransformPath: str = ""
    note: str = ""
    createdAt: str = ""


@dataclass
class MarkerTcpSession:
    active: bool = False
    sessionName: str = ""
    sessionRoot: str = ""
    stage: str = "idle"  # idle | capture | reporting | done | failed
    samples: list[MarkerTcpSample] = field(default_factory=list)
    pendingSampleId: str = ""
    message: str = ""
    reportPath: str = ""


@dataclass
class GatewayState:
    repo_root: Path
    config_path: Path
    config: dict[str, Any]
    recording: RecordingStatus
    replay: ReplayStatus
    profile: str = "thor"
    datasets_root: Path | None = None
    exports_root: Path | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    calibration: CalibrationStatus = field(default_factory=CalibrationStatus)
    calibration_session: CalibrationSession = field(default_factory=CalibrationSession)
    marker_tcp_session: MarkerTcpSession = field(default_factory=MarkerTcpSession)
    dataset_export: DatasetExportStatus = field(default_factory=DatasetExportStatus)
    teleop: TeleopStatus = field(default_factory=TeleopStatus)
    training: training_backend.TrainingRunStatus = field(
        default_factory=training_backend.TrainingRunStatus
    )
    # Bytes, not str: the run writes to its log file, and this object exists only to be
    # polled for liveness and asked for an exit code.
    training_process: subprocess.Popen[bytes] | None = None
    training_persisted_s: float = 0.0
    rollout: rollout_backend.RolloutStatus = field(default_factory=rollout_backend.RolloutStatus)
    # Unlike training, this one keeps its stdin: it is the operator's start/stop channel, and
    # holding it is what lets the gateway steer a rollout. It also means a dead gateway ends
    # the rollout, which for a moving arm is the outcome to want.
    rollout_process: subprocess.Popen[bytes] | None = None
    export_process: subprocess.Popen[str] | None = None
    events: list[EventLogItem] = field(default_factory=list)
    selected_replay_root: Path | None = None
    active_task_id: str | None = None
    process: subprocess.Popen[str] | None = None
    runtime_recording_config: dict[str, Any] | None = None
    runtime_recording_config_path: Path | None = None
    # SpaceMouse 6D gain overrides set from the Teleoperation page, empty until an operator changes
    # one. Kept out of the YAML on purpose: these are tuned live against the arm, and a session's
    # worth of experimenting should not rewrite the file that defines the recording contract.
    runtime_teleop_gains: dict[str, float | None] = field(default_factory=dict)
    replay_process: subprocess.Popen[str] | None = None
    replay_process_kind: str = ""
    teleop_process: subprocess.Popen[str] | None = None
    teleop_started_at_s: float | None = None
    realsense_preview_process: subprocess.Popen[str] | None = None
    realsense_preview_processes: dict[str, subprocess.Popen[str]] = field(default_factory=dict)
    processing_processes: dict[str, subprocess.Popen[str]] = field(default_factory=dict)
    processing_starting: set[str] = field(default_factory=set)
    process_started_at_s: float | None = None
    replay_started_at_s: float | None = None
    log_dir: Path | None = None
    gateway_log_path: Path | None = None
    recorder_log_path: Path | None = None
    device_preview: dict[str, Any] = field(default_factory=dict)
    # BOX 6D force-sensor calibration: the recorder streams CALI_LOG/CALI_DONE
    # lines on stdout after a `cali_6dforce` stdin command; the reader thread
    # appends them here and the Device Manager polls them into its log box.
    # Guarded by its own lock since the stdout reader and the GET handler run on
    # different threads.
    box_cali_running: bool = False
    box_cali_log: list[dict[str, Any]] = field(default_factory=list)
    box_cali_lock: Lock = field(default_factory=Lock)
    # Touch calibration streams on its own TOUCHCALI_LOG/TOUCHCALI_DONE channel
    # into a separate buffer so its log box shows only touch lines (the 6D force
    # and touch buttons now live in separate viewers and must not cross-show).
    box_touch_cali_running: bool = False
    box_touch_cali_log: list[dict[str, Any]] = field(default_factory=list)
    box_touch_cali_lock: Lock = field(default_factory=Lock)
    # Roster of BOX devices the recorder discovered by broadcast at Connect
    # (BOX_DEVICES_JSON). When non-empty it replaces the static YAML-derived
    # box_collection rows so the Device Manager lists exactly the boxes actually
    # on the subnet (one row per discovered box × sensor).
    box_devices_roster: list[dict[str, Any]] = field(default_factory=list)
    # One live preview pipeline per camera device id, so the Device Manager grid
    # can show many cameras at once. Each pipeline's reader thread keeps only the
    # latest JPEG in `camera_preview_frames`; the HTTP layer serves that cached
    # frame as a short snapshot request (NOT a long-lived MJPEG stream), so 11
    # tiles don't exhaust the browser's ~6-connections-per-origin limit. A
    # dedicated spawn lock serializes nvarguscamerasrc Argus opens to dodge the
    # NVMM dmabuf race (NvBufSurfaceFromFd Failed). Pipelines self-terminate
    # once no tile has polled them for `_PREVIEW_IDLE_TTL_S`.
    camera_preview_processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    camera_preview_frames: dict[str, tuple[bytes, float]] = field(default_factory=dict)
    camera_preview_last_access: dict[str, float] = field(default_factory=dict)
    camera_preview_lock: Lock = field(default_factory=Lock)
    camera_preview_spawn_lock: Lock = field(default_factory=Lock)
    camera_preview_last_spawn_s: float = 0.0
    # Set while the recorder owns the cameras so snapshot polls don't respawn a
    # preview pipeline that would re-occupy a sensor the recorder needs.
    camera_preview_suspended: bool = False
    # monotonic() of the last "preview_demand" heartbeat written to recorder
    # stdin; debounces those writes (see _maybe_send_preview_demand).
    recorder_preview_demand_sent_s: float = 0.0
    lock: Lock = field(default_factory=Lock)
    # Recorder stdout lines are pushed here by the reader thread WITHOUT taking
    # `lock`, so a slow snapshot can never back up the pipe and freeze the
    # recorder + camera worker subprocesses. A dedicated consumer thread applies
    # them under `lock`.
    recorder_output_queue: "queue.Queue[tuple[Any, str]]" = field(default_factory=queue.Queue)
    # Cached results of the expensive dataset filesystem scan (298G / 600+
    # episodes on Thor takes 4-12s). A background thread refreshes these OFF the
    # lock; `_snapshot` only reads the cache, so it never walks the dataset tree
    # while holding `lock` (which previously starved the recorder-stdout drain
    # and every camera.jpg preview request for seconds).
    cached_recorded_datasets: list[dict[str, Any]] = field(default_factory=list)
    cached_trajectory: list[Any] = field(default_factory=list)
    cached_trajectory_meta: dict[str, Any] = field(default_factory=dict)
    cached_processing_items: list[dict[str, Any]] = field(default_factory=list)
    dataset_cache_ready: bool = False
    processing_cache_ready: bool = False
    # Cheap fingerprint of dataset dirs plus key sentinel files. This lets the
    # refresher skip the expensive 253-dataset walk when nothing changed while
    # still noticing a just-recorded dataset becoming complete after meta/parquet
    # files are finalized under an already-created top-level directory.
    dataset_scan_signature: tuple = ()
    # Processing is intentionally cached separately from recordedDatasets and
    # trajectory. Trajectory scans are expensive; processing status changes often
    # during EE generation and must not force a full dataset/trajectory rescan.
    processing_scan_signature: tuple = ()

    def log(self, level: str, message: str) -> None:
        self.events.insert(
            0,
            EventLogItem(
                id=f"{time.time_ns()}-{level}",
                time=time.strftime("%H:%M:%S"),
                level=level,
                message=message,
            ),
        )
        del self.events[24:]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to load handheld GUI config files.") from exc

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(loaded).__name__}.")
    return loaded


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset") or {}
    return dataset if isinstance(dataset, dict) else {}


def _replay_config(config: dict[str, Any]) -> dict[str, Any]:
    replay = config.get("replay") or {}
    return replay if isinstance(replay, dict) else {}


def _teleop_config(config: dict[str, Any]) -> dict[str, Any]:
    teleop = config.get("teleop") or {}
    return teleop if isinstance(teleop, dict) else {}


# The SpaceMouse's 6D gain surface, in the order the UI shows it. `translation_scale` and
# `rotation_scale` are the global gains; the six per-axis entries override them one axis at a time
# and are `None` when unset, which is how SpaceMouseTeleopConfig asks for "use the global". A 0.0 is
# therefore *not* the same as unset -- it disables that axis, which is how this rig runs roll and
# pitch: fr3_record_config.yaml pins scale_wx/scale_wy to 0 so the wrist holds the orientation an
# episode starts in, leaving x/y/z and yaw under the SpaceMouse.
FR3_TELEOP_GLOBAL_GAINS = ("translation_scale", "rotation_scale")
FR3_TELEOP_AXIS_GAINS = ("scale_x", "scale_y", "scale_z", "scale_wx", "scale_wy", "scale_wz")
FR3_TELEOP_GAIN_FIELDS = FR3_TELEOP_GLOBAL_GAINS + FR3_TELEOP_AXIS_GAINS

# Per device tick, so the ceiling is a rate: the recorder runs the SpaceMouse at 200 Hz, making
# 0.01 worth 2 m/s or 2 rad/s at full stick deflection. That is already past anything this rig has
# ever been driven at (the largest value in the repo is the sim's 0.001845), so it is a guard
# against a typo'd extra digit rather than a tuning limit. The real safety net downstream is
# `robot.max_target_delta_pos` / `max_target_delta_rot`, which clamp per control step regardless of
# what gain produced the command.
FR3_TELEOP_GAIN_ABS_MAX = 0.01

# `tools/fr3/fr3_mujoco_teleop.py` does not read the recorder's YAML -- it takes gains as CLI flags
# with their own defaults, which differ from the hardware config on every axis that matters:
# 3x the translation gain, and all three rotation axes zeroed. Mirrored here so the UI can show the
# operator what the sim will actually do, and pinned to the parser by
# tests/scripts/test_data_collection_gui_gateway.py so the two cannot drift apart silently.
FR3_SIM_TELEOP_GAIN_DEFAULTS: dict[str, float | None] = {
    "translation_scale": 0.001845,
    "rotation_scale": 0.001944,
    "scale_x": None,
    "scale_y": None,
    "scale_z": None,
    "scale_wx": 0.0,
    "scale_wy": 0.0,
    "scale_wz": 0.0,
}

# The global gain is not what an unset axis gets. `SpaceMouseTeleopConfig` multiplies it by a
# per-axis calibration first (teleop_spacemouse.py, TRANSLATION_/ROTATION_AXIS_CALIBRATION), so an
# unset z moves at 59% of `translation_scale` -- and an explicitly set axis *replaces* the
# calibrated value rather than scaling it, meaning typing the global's own number into z is not a
# no-op. The panel needs both halves of that to state what an axis will really do, so the vectors
# are mirrored here and pinned to the teleoperator's source by
# tests/scripts/test_data_collection_gui_gateway.py.
FR3_TELEOP_AXIS_CALIBRATION: dict[str, float] = {
    "scale_x": 1.0,
    "scale_y": 0.9414634146341463,
    "scale_z": 0.5902439024390244,
    "scale_wx": 1.0,
    "scale_wy": 0.9490740740740741,
    "scale_wz": 0.9259259259259259,
}


def _teleop_gain_defaults(config: dict[str, Any]) -> dict[str, float | None]:
    """The gains the recorder config asks for, with the teleoperator's own fallbacks."""

    teleop = _teleop_config(config)
    defaults: dict[str, float | None] = {}
    for field_name in FR3_TELEOP_GAIN_FIELDS:
        raw = teleop.get(field_name)
        if raw is None or raw == "":
            defaults[field_name] = None
            continue
        try:
            defaults[field_name] = float(raw)
        except (TypeError, ValueError):
            defaults[field_name] = None
    return defaults


def _parse_teleop_gain_overrides(payload: Any) -> dict[str, float | None]:
    """Validate a UI gain payload into the subset that should override the config.

    A key that is absent, `null` or `""` means "leave this one alone"; the caller drops it rather
    than writing a `None`, because `None` is itself a meaningful config value (it means "fall back
    to the global gain") and the UI has no way to distinguish the two intents.
    """

    if not isinstance(payload, dict):
        raise ValueError("Teleop gains must be a JSON object.")
    unknown = sorted(set(payload) - set(FR3_TELEOP_GAIN_FIELDS))
    if unknown:
        raise ValueError(f"Unknown teleop gain(s): {', '.join(unknown)}")

    overrides: dict[str, float | None] = {}
    for field_name in FR3_TELEOP_GAIN_FIELDS:
        if field_name not in payload:
            continue
        raw = payload[field_name]
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number, got {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {raw!r}")
        if abs(value) > FR3_TELEOP_GAIN_ABS_MAX:
            raise ValueError(
                f"{field_name} must be within +/-{FR3_TELEOP_GAIN_ABS_MAX} "
                f"(200 Hz x {FR3_TELEOP_GAIN_ABS_MAX} is already 2 m/s at full deflection)"
            )
        if field_name in FR3_TELEOP_GLOBAL_GAINS and value <= 0.0:
            raise ValueError(
                f"{field_name} is the global gain and must be positive; zero an individual axis "
                "with scale_x .. scale_wz instead"
            )
        overrides[field_name] = value
    return overrides


def _effective_teleop_gains(state: GatewayState) -> dict[str, float | None]:
    gains = _teleop_gain_defaults(state.config)
    gains.update(state.runtime_teleop_gains or {})
    return gains


def _teleop_gains_payload(state: GatewayState) -> dict[str, Any]:
    overrides = state.runtime_teleop_gains or {}
    return {
        "values": _effective_teleop_gains(state),
        "configDefaults": _teleop_gain_defaults(state.config),
        "simDefaults": dict(FR3_SIM_TELEOP_GAIN_DEFAULTS),
        "axisCalibration": dict(FR3_TELEOP_AXIS_CALIBRATION),
        "overridden": sorted(overrides),
        "absMax": FR3_TELEOP_GAIN_ABS_MAX,
    }


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_config(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return default


def _query_flag(query: dict[str, list[str]], key: str) -> bool:
    """Truthiness of a query-string flag, absent meaning false."""
    return (query.get(key, [""])[0] or "").strip().lower() in ("1", "true", "yes", "on")


def _mujoco_validation_thresholds(config: dict[str, Any]) -> tuple[float, float]:
    replay = _replay_config(config)
    return (
        _float_config(replay, "mujoco_max_position_error_mm", DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM),
        _float_config(replay, "mujoco_max_rotation_error_deg", DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG),
    )


def _new_mujoco_validation(
    state: GatewayState,
    *,
    status: str = "not_run",
    dataset_root: Path | None = None,
    episode: int | None = None,
    message: str = "Run MuJoCo replay before real-robot replay.",
) -> dict[str, Any]:
    max_pos_mm, max_rot_deg = _mujoco_validation_thresholds(state.config)
    return {
        "status": status,
        "datasetRoot": str(dataset_root) if dataset_root is not None else "",
        "episode": int(state.replay.episode if episode is None else episode),
        "fps": int(state.replay.fps or 30),
        "exitCode": None,
        "completedFrames": 0,
        "totalFrames": 0,
        "avgPositionErrorMm": None,
        "maxPositionErrorMm": None,
        "avgRotationErrorDeg": None,
        "maxRotationErrorDeg": None,
        "maxPositionThresholdMm": max_pos_mm,
        "maxRotationThresholdDeg": max_rot_deg,
        "hasStructuredResult": False,
        "trajectoryContract": {},
        "isCurrentForSelection": False,
        "message": message,
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cubeMode": str(state.replay.mujocoCubeMode),
    }


def _invalidate_mujoco_validation(state: GatewayState, message: str) -> None:
    state.replay.mujocoValidation = _new_mujoco_validation(state, message=message)


def _validation_store_path(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "gui_replay_validations.json"


def _read_validation_store(dataset_root: Path) -> dict[str, Any]:
    path = _validation_store_path(dataset_root)
    if not path.is_file():
        return {"validations": []}
    try:
        with path.open("r", encoding="utf-8") as validation_file:
            payload = json.load(validation_file)
    except (OSError, json.JSONDecodeError):
        return {"validations": []}
    if not isinstance(payload, dict):
        return {"validations": []}
    validations = payload.get("validations")
    if not isinstance(validations, list):
        payload["validations"] = []
    return payload


def _write_validation_store(dataset_root: Path, validation: dict[str, Any]) -> None:
    path = _validation_store_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_validation_store(dataset_root)
    validations = [item for item in payload.get("validations", []) if isinstance(item, dict)]
    key = (
        str(validation.get("datasetRoot") or ""),
        int(validation.get("episode") or 0),
        int(validation.get("fps") or 0),
        float(validation.get("maxPositionThresholdMm") or 0.0),
        float(validation.get("maxRotationThresholdDeg") or 0.0),
        str(validation.get("cubeMode") or DEFAULT_MUJOCO_CUBE_MODE),
    )
    filtered = [
        item
        for item in validations
        if (
            str(item.get("datasetRoot") or ""),
            int(item.get("episode") or 0),
            int(item.get("fps") or 0),
            float(item.get("maxPositionThresholdMm") or 0.0),
            float(item.get("maxRotationThresholdDeg") or 0.0),
            str(item.get("cubeMode") or DEFAULT_MUJOCO_CUBE_MODE),
        )
        != key
    ]
    payload["validations"] = [validation, *filtered][:50]
    with path.open("w", encoding="utf-8") as validation_file:
        json.dump(payload, validation_file, ensure_ascii=False, indent=2)
        validation_file.write("\n")


def _load_persisted_mujoco_validation(state: GatewayState, dataset_root: Path, episode: int) -> dict[str, Any] | None:
    max_pos_mm, max_rot_deg = _mujoco_validation_thresholds(state.config)
    for item in _read_validation_store(dataset_root).get("validations", []):
        if not isinstance(item, dict):
            continue
        try:
            matches = (
                Path(str(item.get("datasetRoot") or "")).resolve() == dataset_root.resolve()
                and int(item.get("episode")) == int(episode)
                and int(item.get("fps")) == int(state.replay.fps or 30)
                and float(item.get("maxPositionThresholdMm")) == max_pos_mm
                and float(item.get("maxRotationThresholdDeg")) == max_rot_deg
                and str(item.get("cubeMode") or DEFAULT_MUJOCO_CUBE_MODE) == str(state.replay.mujocoCubeMode)
            )
        except (TypeError, ValueError, OSError):
            matches = False
        if matches:
            return dict(item)
    return None


def _refresh_mujoco_validation_current(state: GatewayState) -> None:
    validation = state.replay.mujocoValidation
    if not validation:
        return
    try:
        dataset_root = _active_replay_dataset_root(state)
        validation["isCurrentForSelection"] = _mujoco_validation_is_for_active_episode(state, dataset_root)
    except RuntimeError:
        validation["isCurrentForSelection"] = False


def _target_frames(config: dict[str, Any]) -> int:
    dataset = _dataset_config(config)
    fps = int(dataset.get("fps") or 30)
    episode_time_s = float(dataset.get("episode_time_s") or 10.0)
    return max(1, int(round(fps * episode_time_s)))


def _config_summary(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    dataset = _dataset_config(config)
    sensors = config.get("sensors") if isinstance(config.get("sensors"), dict) else {}
    soft_sync = sensors.get("soft_sync") if isinstance(sensors.get("soft_sync"), dict) else {}
    cameras = sensors.get("cameras") if isinstance(sensors.get("cameras"), dict) else {}
    cam_defaults = cameras.get("defaults") if isinstance(cameras.get("defaults"), dict) else {}
    hw_sync = sensors.get("hardware_sync") if isinstance(sensors.get("hardware_sync"), dict) else {}
    recorder = config.get("recorder") if isinstance(config.get("recorder"), dict) else {}
    recorder_script = str(recorder.get("script") or "")
    is_gmsl = "gmsl" in recorder_script or "defaults" in cameras

    vcodec = str(dataset.get("vcodec") or cam_defaults.get("codec") or "")
    is_fr3 = "fr3_gui_record_runtime" in recorder_script or "tools/fr3/" in recorder_script or "tools\\fr3\\" in recorder_script
    return {
        "configPath": str(config_path),
        "repoId": str(dataset.get("repo_id") or ""),
        "root": str(dataset.get("root") or ""),
        "fps": int(dataset.get("fps") or 30),
        "episodeTimeS": float(dataset.get("episode_time_s") or 10.0),
        "targetFrames": _target_frames(config),
        "numEpisodes": int(dataset.get("num_episodes") or 0),
        "video": bool(dataset.get("video", True)),
        "streamingEncoding": bool(dataset.get("streaming_encoding", False)),
        "vcodec": vcodec,
        "softSync": bool(soft_sync.get("enabled", False)),
        "rerun": {
            "displayData": bool(config.get("display_data", False)),
            "savePath": str(config.get("rerun_save_path") or ""),
        },
        "recorderScript": recorder_script,
        "rigType": "fr3" if is_fr3 else "gmsl2" if is_gmsl else "handheld",
        "hardwareSync": {
            "enabled": bool(hw_sync.get("enabled", False)),
            "fps": int(hw_sync.get("fps") or 0),
            "trigMode": int(hw_sync.get("sensor_trig_mode") or 0),
            "pwmChip": str(hw_sync.get("pwm_chip") or ""),
            "pwmId": int(hw_sync.get("pwm_id") or 0),
        },
        "cameraDefaults": {
            "codec": str(cam_defaults.get("codec") or ""),
            "bitrateKbps": int(cam_defaults.get("bitrate_kbps") or 0),
            "width": int(cam_defaults.get("width") or 0),
            "height": int(cam_defaults.get("height") or 0),
            "pipeline": str(cam_defaults.get("pipeline") or ""),
            "exposureUs": int(cam_defaults.get("exposure_us") or 0),
            "gain": int(cam_defaults.get("gain") or 0),
            "iframeInterval": int(cam_defaults.get("iframe_interval") or 0),
            "container": str(cam_defaults.get("container") or ""),
        },
    }


def _recording_status_from_config(config: dict[str, Any]) -> RecordingStatus:
    dataset = _dataset_config(config)
    return RecordingStatus(
        datasetRoot=str(dataset.get("root") or ""),
        repoId=str(dataset.get("repo_id") or ""),
        targetFrames=_target_frames(config),
        message="Gateway ready to launch handheld recorder",
    )


def _replay_status_from_config(config: dict[str, Any]) -> ReplayStatus:
    dataset = _dataset_config(config)
    replay = _replay_config(config)
    robot = config.get("robot") if isinstance(config.get("robot"), dict) else {}
    default_robot_ip = str(replay.get("robot_ip") or robot.get("robot_ip") or DEFAULT_CUBE_REPLAY_ROBOT_IP)
    return ReplayStatus(
        dataset=str(dataset.get("repo_id") or ""),
        datasetRoot=str(dataset.get("root") or ""),
        totalFrames=_target_frames(config),
        fps=int(dataset.get("fps") or 30),
        realRobotIp=default_robot_ip,
        realEndEffectorMode=(
            "pika_gripper_ee"
            if str(robot.get("gripper_backend") or "") == "pika"
            or "pika" in str(robot.get("urdf_path") or "").lower()
            else "corenetic_gripper_ee"
        ),
    )


_BOX_COLLECTION_DEVICE_LABELS: dict[str, str] = {
    "box_gripper": "BOX gripper (distance)",
    "box_imu": "BOX IMU (acc/gyr/euler/quat)",
    "box_trigger": "BOX trigger travel",
    "box_six_d_force": "BOX 6D force",
    "box_touch_left": "Paxini touch pad L",
    "box_touch_right": "Paxini touch pad R",
}


def _detect_locked_sids(repo_root: Path | None) -> list[int] | None:
    if repo_root is None:
        return None
    script = repo_root / "tools" / "thor" / "gmsl2" / "check_max96726_locks.sh"
    if not script.exists():
        return None
    try:
        result = subprocess.run(
            [str(script)], capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode not in (0, 1):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("LOCKED_VIDEO_IDS="):
            csv = line.split("=", 1)[1].strip()
            return [int(x) for x in csv.split(",") if x.strip()] if csv else []
    return None


def _device_statuses(config: dict[str, Any], repo_root: Path | None = None) -> list[dict[str, Any]]:
    sensors = config.get("sensors") or {}
    if not isinstance(sensors, dict):
        sensors = {}

    devices: list[dict[str, Any]] = []
    cameras_section = sensors.get("cameras")

    if isinstance(cameras_section, dict) and "defaults" in cameras_section:
        defaults = cameras_section.get("defaults") if isinstance(cameras_section.get("defaults"), dict) else {}
        prefix = str(cameras_section.get("name_prefix") or "cam")
        sensor_ids = cameras_section.get("sensor_ids") or []
        detect_all = bool(cameras_section.get("detect_all", False))
        if sensor_ids:
            slot_ids = [int(x) for x in sensor_ids]
        elif detect_all:
            locked = _detect_locked_sids(repo_root)
            if locked is not None:
                slot_ids = locked
            else:
                slot_ids = list(range(16))
        else:
            slot_ids = []
        for sid in slot_ids:
            devices.append(
                {
                    "id": f"{prefix}_{sid:02d}",
                    "kind": "camera",
                    "label": f"GMSL2 sensor-id {sid}",
                    "state": "idle",
                    "fps": int(defaults.get("fps") or 0),
                    "latencyMs": 0,
                    "detail": _format_device_detail(defaults),
                    "config": {**defaults, "sensor_id": sid},
                }
            )
    else:
        # Mapping-style camera section (Hikrobot / OpenCV / RealSense).
        section = cameras_section or {}
        if isinstance(section, dict):
            for device_id, raw_device in section.items():
                device = raw_device if isinstance(raw_device, dict) else {}
                devices.append(_make_mapping_device(device_id, device, "camera"))

    for section_name, kind in (
        ("tactiles", "tactile"),
        ("handheld_grippers", "handheld_gripper"),
    ):
        section = sensors.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for device_id, raw_device in section.items():
            device = raw_device if isinstance(raw_device, dict) else {}
            devices.append(_make_mapping_device(device_id, device, kind))

    robot = config.get("robot") if isinstance(config.get("robot"), dict) else {}
    robot_cameras = robot.get("cameras") if isinstance(robot.get("cameras"), dict) else {}
    existing_ids = {str(device.get("id") or "") for device in devices}
    for device_id, raw_device in robot_cameras.items():
        if str(device_id) in existing_ids:
            continue
        device = raw_device if isinstance(raw_device, dict) else {}
        devices.append(_make_mapping_device(str(device_id), device, "camera"))

    if robot:
        devices.append(
            {
                "id": "fr3",
                "kind": "robot",
                "label": "Franka Research 3",
                "state": "idle",
                "fps": int(config.get("control_fps") or 0),
                "latencyMs": 0,
                "detail": f"robot {robot.get('robot_ip', '?')}",
                "config": robot,
            }
        )
        gripper_port = robot.get("gripper_port")
        if gripper_port:
            devices.append(
                {
                    "id": "pika",
                    "kind": "gripper",
                    "label": "Pika gripper",
                    "state": "idle",
                    "fps": int(config.get("control_fps") or 0),
                    "latencyMs": 0,
                    "detail": str(gripper_port),
                    "config": {"port": gripper_port, "backend": robot.get("gripper_backend", "pika")},
                }
            )
    teleop = config.get("teleop") if isinstance(config.get("teleop"), dict) else {}
    if teleop:
        devices.append(
            {
                "id": str(teleop.get("type") or "teleoperator"),
                "kind": "teleoperator",
                "label": "SpaceMouse" if teleop.get("type") == "spacemouse" else str(teleop.get("type") or "Teleoperator"),
                "state": "idle",
                "fps": int(config.get("control_fps") or 0),
                "latencyMs": 0,
                "detail": f"USB input device {teleop.get('device_id', 0)}",
                "config": teleop,
            }
        )

    box_cfg = config.get("box_collection")
    if isinstance(box_cfg, dict) and box_cfg.get("enabled", True):
        # Accept both the legacy flat single-box block and the new multi-box
        # `boxes:` list. A single empty-id box keeps bare sensor IDs so existing
        # rigs render identically; with >1 box each row's id is namespaced.
        raw_boxes = box_cfg.get("boxes")
        if isinstance(raw_boxes, list):
            box_entries = [b for b in raw_boxes if isinstance(b, dict)]
        else:
            box_entries = [box_cfg]
        for box in box_entries:
            devices.extend(_box_collection_devices(box))
    return devices


def _set_workstation_device_probe(
    devices: list[dict[str, Any]],
    device_id: str,
    *,
    detected: bool,
    detail: str,
    warning: bool = False,
) -> None:
    device = next((item for item in devices if item.get("id") == device_id), None)
    if device is None:
        return
    device["state"] = "warning" if warning else ("running" if detected else "error")
    device["detail"] = detail


def _usb_physical_device_count(vendor_id: str, product_id: str) -> int:
    count = 0
    for vendor_path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_path.read_text().strip().lower() != vendor_id.lower():
                continue
            if vendor_path.with_name("idProduct").read_text().strip().lower() == product_id.lower():
                count += 1
        except OSError:
            continue
    return count


def _probe_workstation_devices(state: GatewayState) -> None:
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    robot_ip = str(robot.get("robot_ip") or "")
    if robot_ip:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", robot_ip],
                capture_output=True,
                text=True,
                timeout=2,
            )
            reachable = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            reachable = False
        fci_ready = False
        fci_detail = ""
        if reachable:
            try:
                fci_probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "from panda_py import Panda; import sys; "
                            "arm=Panda(sys.argv[1]); state=arm.get_state(); print(len(state.q))"
                        ),
                        robot_ip,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                fci_ready = fci_probe.returncode == 0 and fci_probe.stdout.strip() == "7"
                failure = fci_probe.stderr.strip().splitlines()[-1] if fci_probe.stderr.strip() else "state read failed"
                fci_detail = "FCI state read ready" if fci_ready else f"FCI unavailable: {failure}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                fci_detail = f"FCI unavailable: {exc}"
        _set_workstation_device_probe(
            state.devices,
            "fr3",
            detected=fci_ready,
            warning=reachable and not fci_ready,
            detail=f"{robot_ip} reachable; {fci_detail}" if reachable else f"{robot_ip} unreachable",
        )

    gripper_port = str(robot.get("gripper_port") or "")
    if gripper_port:
        port = Path(gripper_port)
        exists = port.exists()
        accessible = exists and os.access(port, os.R_OK | os.W_OK)
        _set_workstation_device_probe(
            state.devices,
            "pika",
            detected=accessible,
            warning=exists and not accessible,
            detail=(
                f"{gripper_port} read/write ready"
                if accessible
                else f"{gripper_port} exists but lacks read/write permission"
                if exists
                else f"{gripper_port} not found"
            ),
        )

    spacemouse_count = _usb_physical_device_count("256f", "c635")
    _set_workstation_device_probe(
        state.devices,
        "spacemouse",
        detected=spacemouse_count > 0,
        detail=(
            f"3Dconnexion SpaceMouse Compact 256f:c635 ({spacemouse_count} physical device)"
            if spacemouse_count
            else "3Dconnexion SpaceMouse Compact 256f:c635 not found"
        ),
    )

    detected_realsense: dict[str, str] = {}
    try:
        import pyrealsense2 as rs

        for device in rs.context().query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            detected_realsense[str(serial)] = str(name)
    except Exception as exc:  # noqa: BLE001
        state.log("warn", f"RealSense probe failed: {exc}")

    cameras = robot.get("cameras") if isinstance(robot.get("cameras"), dict) else {}
    for camera_name, camera in cameras.items():
        if not isinstance(camera, dict) or str(camera.get("type", "")).lower() != "intelrealsense":
            continue
        serial = str(camera.get("serial_number_or_name") or "")
        model = detected_realsense.get(serial)
        resolution = f"{camera.get('width', '?')}x{camera.get('height', '?')}@{camera.get('fps', '?')}"
        _set_workstation_device_probe(
            state.devices,
            str(camera_name),
            detected=model is not None,
            detail=f"{model} serial {serial} {resolution}" if model else f"RealSense serial {serial} not found",
        )


def _box_collection_devices(box: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one frontend row per expected sensor for a single BOX config."""
    box_id = str(box.get("box_id", "") or "")
    expected = box.get("expected_devices") or list(_BOX_COLLECTION_DEVICE_LABELS)
    try:
        poll_hz = int(round(1.0 / float(box.get("poll_interval_s") or 0.05)))
    except (TypeError, ValueError, ZeroDivisionError):
        poll_hz = 0
    detail = f"UDP {box.get('remote_ip', '?')}:{box.get('remote_port', 15000)}"
    out: list[dict[str, Any]] = []
    for sensor_id in expected:
        sid = str(sensor_id)
        device_id = f"{box_id}/{sid}" if box_id else sid
        out.append(
            {
                "id": device_id,
                "kind": "box_collection",
                "label": _BOX_COLLECTION_DEVICE_LABELS.get(sid, sid),
                "state": "idle",
                "fps": poll_hz,
                "latencyMs": 0,
                "detail": detail,
                "config": {
                    "box_id": box_id,
                    "remote_ip": box.get("remote_ip", ""),
                    "remote_port": box.get("remote_port", 15000),
                    "poll_interval_s": box.get("poll_interval_s", 0.05),
                    "bind_ip": box.get("bind_ip", ""),
                    "sensor_id": sid,
                },
            }
        )
    return out


def _box_devices_from_roster(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Device Manager rows from the recorder's discovered BOX roster.

    One row per (discovered box × sensor). Mirrors :func:`_box_collection_devices`
    so the frontend renders discovered boxes identically to configured ones, but
    sourced from the live broadcast enumeration (device_id / sn / ip) rather than
    static YAML. A single box with an empty ``box_id`` keeps bare sensor IDs so
    legacy single-box rigs render and namespace exactly as before.
    """
    out: list[dict[str, Any]] = []
    for entry in roster:
        if not isinstance(entry, dict):
            continue
        box_id = str(entry.get("box_id", "") or "")
        sensors = entry.get("expected_devices") or entry.get("capability_names") or []
        ip = str(entry.get("ip", "") or "?")
        sn = str(entry.get("sn", "") or "")
        device_num = entry.get("device_id")
        detail_id = sn or (f"id={device_num}" if device_num is not None else "")
        detail = f"UDP {ip}:{entry.get('data_port', 15000)}"
        if detail_id:
            detail = f"{detail} ({detail_id})"
        for sensor_id in sensors:
            sid = str(sensor_id)
            row_id = f"{box_id}/{sid}" if box_id else sid
            out.append(
                {
                    "id": row_id,
                    "kind": "box_collection",
                    "label": _BOX_COLLECTION_DEVICE_LABELS.get(sid, sid),
                    "state": "idle",
                    "fps": 0,
                    "latencyMs": 0,
                    "detail": detail,
                    "config": {
                        "box_id": box_id,
                        "device_id": device_num,
                        "sn": sn,
                        # Firmware version as reported by the box's discovery
                        # broadcast (0 when the wheel/firmware predates it). The
                        # calibration center surfaces this as the box's version.
                        "fw_version": entry.get("fw_version", 0),
                        "remote_ip": ip,
                        "data_port": entry.get("data_port", 15000),
                        "sensor_id": sid,
                    },
                }
            )
    return out


def _make_mapping_device(device_id: str, device: dict[str, Any], kind: str) -> dict[str, Any]:
    fps = int(device.get("fps") or 0)
    detail_parts = [str(device.get("type") or kind)]
    if "serial" in device:
        detail_parts.append(str(device["serial"]))
    if "port" in device:
        detail_parts.append(str(device["port"]))
    if "index_or_path" in device:
        detail_parts.append(str(device["index_or_path"]))
    if "serial_number_or_name" in device:
        detail_parts.append(str(device["serial_number_or_name"]))
    return {
        "id": str(device_id),
        "kind": kind,
        "label": " ".join(detail_parts),
        "state": "idle",
        "fps": fps,
        "latencyMs": 0,
        "detail": _format_device_detail(device),
        "config": dict(device),
    }


def _format_device_detail(device: dict[str, Any]) -> str:
    width = device.get("width")
    height = device.get("height")
    transport = device.get("transport_layer")
    details = []
    if width and height:
        details.append(f"{width}x{height}")
    if transport:
        details.append(str(transport))
    return " ".join(details) or str(device.get("type") or "device")


def _resolve_dataset_root(repo_root: Path, raw_root: str | Path | None) -> Path | None:
    if not raw_root:
        return None
    root = Path(raw_root)
    return root if root.is_absolute() else repo_root / root


def _dataset_data_files(dataset_root: Path) -> list[Path]:
    parquets = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if parquets:
        return parquets
    return sorted((dataset_root / "episodes").glob("episode_*/*.mkv"))


def _path_modified_s(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _dataset_modified_s(dataset_root: Path) -> float:
    candidates = [dataset_root, dataset_root / "meta" / "info.json", dataset_root / "episodes"]
    data_files = _dataset_data_files(dataset_root)
    candidates.extend(data_files[:5])
    return max((_path_modified_s(path) for path in candidates), default=0.0)


def _dataset_name_prefixes(name: str) -> set[str]:
    prefixes = {name}
    timestamped_match = re.match(r"^(?P<base>.+)_\d{8}_\d{6}(?:_\d{2})?$", name)
    if timestamped_match:
        prefixes.add(timestamped_match.group("base"))
    return prefixes


def _has_gmsl2_episodes(path: Path) -> bool:
    eps_dir = path / "episodes"
    if not eps_dir.is_dir():
        return False
    return any(eps_dir.glob("episode_*/meta.json"))


def _has_lerobot_v3_data(path: Path) -> bool:
    return any((path / "data").glob("chunk-*/*.parquet"))


def _is_dataset_root(path: Path) -> bool:
    return path.is_dir() and ((path / "meta" / "info.json").is_file() or _has_gmsl2_episodes(path))


def _scan_datasets_root(state: GatewayState) -> list[Path]:
    root = state.datasets_root
    if root is None or not root.is_dir():
        return []
    if _is_dataset_root(root):
        return [root]
    return [entry for entry in root.iterdir() if _is_dataset_root(entry)]


def _scan_exports_root(state: GatewayState) -> list[Path]:
    root = _task_exports_root(state)
    if not root.is_dir():
        return []
    if _is_dataset_root(root):
        return [root]
    entries = [entry for entry in root.iterdir() if _is_dataset_root(entry)]
    # Training views sit one level deeper (exports/training_views/<dataset>__<contract>). A
    # single-level scan skipped the grouping directory -- which is not a dataset root -- and so
    # hid every built view from replay selection, video serving and every other endpoint that
    # gates on this candidate list.
    views_root = _training_views_root(state)
    if views_root.is_dir():
        entries.extend(entry for entry in views_root.iterdir() if _is_dataset_root(entry))
    return entries


def _dataset_root_candidates(state: GatewayState) -> list[Path]:
    dataset = _dataset_config(state.config)
    configured_root = _resolve_dataset_root(state.repo_root, dataset.get("root"))
    live_root = _resolve_dataset_root(state.repo_root, state.recording.datasetRoot)

    candidates: list[Path] = []
    for root in (live_root, configured_root):
        if root is None:
            continue
        parent = root.parent
        if parent.exists():
            for prefix in _dataset_name_prefixes(root.name):
                exact = parent / prefix
                if exact.exists() and exact.is_dir() and exact not in candidates:
                    candidates.append(exact)
                for sibling in parent.glob(f"{prefix}_*"):
                    if sibling.is_dir() and sibling not in candidates:
                        candidates.append(sibling)
        if root.exists() and root not in candidates:
            candidates.append(root)
    for entry in _scan_datasets_root(state):
        if entry not in candidates:
            candidates.append(entry)
    return sorted(candidates, key=_dataset_modified_s, reverse=True)


def _dataset_kind(state: GatewayState, dataset_root: Path) -> str:
    try:
        resolved = dataset_root.resolve()
    except OSError:
        return "recorded"
    # A training view is a re-expression of a recording, not a recording: keeping it its own kind
    # is what lets the UI nest it under its source instead of offering it as capture output.
    try:
        resolved.relative_to(_training_views_root(state).resolve())
        return "training_view"
    except (OSError, ValueError):
        pass
    try:
        resolved.relative_to(_task_exports_root(state).resolve())
    except (OSError, ValueError):
        return "recorded"
    return "exported"


_REPLAY_CANDIDATES_MEMO: tuple[float, tuple[Any, ...], list[Path]] | None = None
_REPLAY_CANDIDATES_TTL_S = 3.0
_PROCESSING_STALE_RUNNING_S = 120.0


def _invalidate_replay_candidates_memo() -> None:
    global _REPLAY_CANDIDATES_MEMO
    _REPLAY_CANDIDATES_MEMO = None


def _path_memo_key(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _complete_replay_candidates_memo_key(state: GatewayState) -> tuple[Any, ...]:
    return (
        _path_memo_key(state.repo_root),
        _path_memo_key(state.datasets_root),
        _path_memo_key(_task_exports_root(state)),
        _dataset_scan_signature(state),
    )


def _complete_replay_dataset_candidates(state: GatewayState) -> list[Path]:
    # Short-TTL memo: this globs every dataset (253 on Thor) and is the single
    # chokepoint reached by several _snapshot helpers (annotation resolve,
    # replay candidates, known-dataset resolve) on every ~1s poll. Key it by the
    # cheap dataset signature so a just-finalized dataset bypasses the memo.
    global _REPLAY_CANDIDATES_MEMO
    now = time.monotonic()
    memo_key = _complete_replay_candidates_memo_key(state)
    memo = _REPLAY_CANDIDATES_MEMO
    if memo is not None and now - memo[0] < _REPLAY_CANDIDATES_TTL_S and memo[1] == memo_key:
        return list(memo[2])
    candidates = list(_complete_dataset_candidates(state))
    for entry in _scan_exports_root(state):
        if entry not in candidates and _dataset_is_complete(entry):
            candidates.append(entry)
    result = sorted(candidates, key=_dataset_modified_s, reverse=True)
    _REPLAY_CANDIDATES_MEMO = (now, memo_key, result)
    return list(result)


def _replay_dataset_candidates(state: GatewayState) -> list[Path]:
    candidates = _complete_replay_dataset_candidates(state)
    selected = state.selected_replay_root.resolve() if state.selected_replay_root is not None else None
    if selected is None:
        return candidates
    selected_matches = [candidate for candidate in candidates if candidate.resolve() == selected]
    other_candidates = [candidate for candidate in candidates if candidate.resolve() != selected]
    return [*selected_matches, *other_candidates]


def _select_replay_dataset(state: GatewayState, raw_path: str) -> None:
    if not raw_path:
        raise ValueError("Missing dataset path.")
    requested = _resolve_dataset_root(state.repo_root, raw_path)
    if requested is None:
        raise ValueError("Missing dataset path.")
    requested = requested.resolve()
    candidates = _complete_replay_dataset_candidates(state)
    matched = next((candidate for candidate in candidates if candidate.resolve() == requested), None)
    if matched is None:
        raise ValueError(f"Dataset is not in the recorded dataset list: {requested}")

    state.selected_replay_root = matched
    episode_options = _dataset_episode_indices(matched, _load_dataset_info(matched))
    state.replay.state = "idle"
    state.replay.safety = "locked"
    state.replay.episodeOptions = episode_options
    state.replay.episode = episode_options[0] if episode_options else 0
    state.replay.frameIndex = 0
    state.replay.trackingErrorMm = 0.0
    state.replay.datasetRoot = str(matched)
    state.replay.datasetKind = _dataset_kind(state, matched)
    state.replay.dataset = str(matched)
    # Not the recorder's current dataset.fps: that describes the next recording, and a dataset
    # recorded before the rate was changed replays at the wrong speed and scores an under-integrated
    # servo window if it is used. See _replay_fps.
    declared_fps = _dataset_declared_fps(matched)
    if declared_fps is not None and declared_fps != state.replay.fps:
        state.log(
            "info",
            f"Replay fps {state.replay.fps} -> {declared_fps} (from {matched.name}/meta/info.json)",
        )
        state.replay.fps = declared_fps
    state.replay.message = f"Selected {_dataset_kind(state, matched)} dataset: {matched.name}"
    _invalidate_mujoco_validation(state, "Dataset changed; run MuJoCo replay again before real-robot replay.")
    persisted_validation = _load_persisted_mujoco_validation(state, matched, state.replay.episode)
    if persisted_validation is not None:
        state.replay.mujocoValidation = persisted_validation
        _refresh_mujoco_validation_current(state)
        state.replay.message = f"Selected {_dataset_kind(state, matched)} dataset: {matched.name}; MuJoCo validation restored"
    state.cached_recorded_datasets = _recorded_dataset_items(state)
    state.cached_trajectory, state.cached_trajectory_meta = _read_recorded_trajectory(state)
    state.dataset_cache_ready = True
    state.log("info", f"Selected replay dataset {matched}")


def _select_replay_episode(state: GatewayState, raw_episode: str) -> None:
    if raw_episode == "":
        raise ValueError("Missing episode index.")
    try:
        episode = int(raw_episode)
    except ValueError as exc:
        raise ValueError(f"Invalid episode index: {raw_episode}") from exc
    dataset_root = state.selected_replay_root or _resolve_known_dataset(state, state.replay.datasetRoot or state.replay.dataset)
    if dataset_root is None:
        raise ValueError("Select a recorded dataset before selecting an episode.")
    episode_options = _dataset_episode_indices(dataset_root, _load_dataset_info(dataset_root))
    if episode_options and episode not in episode_options:
        raise ValueError(f"Episode {episode} is not available for {dataset_root.name}.")
    state.selected_replay_root = dataset_root
    state.replay.episode = episode
    state.replay.episodeOptions = episode_options
    state.replay.state = "idle"
    state.replay.safety = "locked"
    state.replay.frameIndex = 0
    state.replay.trackingErrorMm = 0.0
    state.replay.datasetKind = _dataset_kind(state, dataset_root)
    state.replay.message = f"Selected episode {episode} from {dataset_root.name}"
    _invalidate_mujoco_validation(state, "Episode changed; run MuJoCo replay again before real-robot replay.")
    persisted_validation = _load_persisted_mujoco_validation(state, dataset_root, episode)
    if persisted_validation is not None:
        state.replay.mujocoValidation = persisted_validation
        _refresh_mujoco_validation_current(state)
        state.replay.message = f"Selected episode {episode} from {dataset_root.name}; MuJoCo validation restored"
    state.cached_trajectory, state.cached_trajectory_meta = _read_recorded_trajectory(state)
    state.dataset_cache_ready = True
    state.log("info", f"Selected replay episode {episode} for {dataset_root}")


def _delete_replay_episode(state: GatewayState, raw_episode: str) -> None:
    if raw_episode == "":
        raise ValueError("Missing episode index.")
    try:
        episode = int(raw_episode)
    except ValueError as exc:
        raise ValueError(f"Invalid episode index: {raw_episode}") from exc
    dataset_root = state.selected_replay_root or _resolve_known_dataset(
        state, state.replay.datasetRoot or state.replay.dataset
    )
    if dataset_root is None:
        raise ValueError("Select a recorded dataset before deleting an episode.")
    episode_options = _dataset_episode_indices(dataset_root, _load_dataset_info(dataset_root))
    if episode not in episode_options:
        raise ValueError(f"Episode {episode} is not available for {dataset_root.name}.")
    if len(episode_options) <= 1:
        raise ValueError("Cannot delete the only remaining episode in the dataset.")

    # old episode index -> new contiguous index for every episode we keep. This
    # is the same construction dataset_tools.delete_episodes uses internally, so
    # the v3 core and the side metadata stay in lockstep.
    kept = [ep for ep in episode_options if ep != episode]
    mapping = {old: new for new, old in enumerate(kept)}

    # Dispatch by what the replay timeline actually reads. When a v3 parquet is
    # present the timeline reads *it* (see _read_dataset_timeline's
    # `not _has_lerobot_v3_data` guard), so reindexing the parquet/meta is what
    # makes the per-episode frames change. gmsl2 episode dirs (the mkv videos)
    # are renumbered alongside inside the v3 path for hybrid datasets. Only a
    # pure gmsl2 dataset (no parquet) takes the dir-only path.
    if _has_lerobot_v3_data(dataset_root):
        _delete_lerobot_v3_episode(dataset_root, episode, mapping)
    elif _has_gmsl2_episodes(dataset_root):
        _delete_gmsl2_episode(dataset_root, episode, mapping)
    else:
        raise ValueError(f"Dataset layout not supported for deletion: {dataset_root.name}")

    # Remap episode-keyed side files at the final root for both formats. Derived
    # EE-pose trajectories are intentionally dropped by the per-format helpers
    # (they can't be trusted after a frame renumber) and are regenerated later.
    _reindex_episode_side_metadata(dataset_root, mapping, episode)

    # gmsl2 renumber renames meta.json paths, so stale frame-count memo entries
    # would otherwise linger; the memo is cheap to rebuild.
    _GMSL2_EP_FRAMES_MEMO.clear()

    new_options = _dataset_episode_indices(dataset_root, _load_dataset_info(dataset_root))
    state.selected_replay_root = dataset_root
    state.replay.episodeOptions = new_options
    # Deleting shifts the survivors down, so the slot we stay on (see below) now
    # holds a different episode's frames/videos. Bump the revision so the UI
    # refetches the timeline even though (datasetRoot, episode) is unchanged.
    state.replay.revision += 1
    # Stay on the slot the deleted episode occupied (the next episode shifts into
    # it); clamp to the new last episode when we deleted the tail.
    next_episode = min(episode, new_options[-1]) if new_options else 0
    state.replay.episode = next_episode
    state.replay.state = "idle"
    state.replay.safety = "locked"
    state.replay.frameIndex = 0
    state.replay.trackingErrorMm = 0.0
    state.replay.datasetRoot = str(dataset_root)
    state.replay.dataset = str(dataset_root)
    state.replay.message = (
        f"Deleted episode {episode} from {dataset_root.name}; "
        f"{len(new_options)} episode(s) remain (reindexed)."
    )
    # _snapshot re-derives episodeOptions/totalEpisodes/message from the
    # background scan cache, which still describes the pre-delete tree until the
    # ~4s refresher runs — so without this the response (and every poll until
    # then) would still report the old episode count. We already hold state.lock
    # here, so recompute the cache inline rather than calling
    # _refresh_dataset_stats_cache (which re-acquires the lock and would
    # deadlock). The scan functions are read-only on state.
    state.selected_replay_root = dataset_root
    state.replay.episode = next_episode
    state.cached_recorded_datasets = _recorded_dataset_items(state)
    state.cached_trajectory, state.cached_trajectory_meta = _read_recorded_trajectory(state)
    state.dataset_cache_ready = True
    _invalidate_mujoco_validation(
        state, "Episode deleted; run MuJoCo replay again before real-robot replay."
    )
    persisted_validation = _load_persisted_mujoco_validation(state, dataset_root, next_episode)
    if persisted_validation is not None:
        state.replay.mujocoValidation = persisted_validation
        _refresh_mujoco_validation_current(state)
    state.log("warn", f"Deleted replay episode {episode} from {dataset_root}")


def _delete_lerobot_v3_episode(dataset_root: Path, episode: int, mapping: dict[int, int]) -> None:
    # The gateway runs under a slim python (pyarrow + pandas, but no
    # datasets/torch/av), so lerobot.datasets.dataset_tools.delete_episodes is
    # unimportable here. Reindex the v3 parquet + meta in place with pyarrow
    # instead. This is only correct when there are no *embedded* v3 videos to
    # re-cut; the gmsl2-hybrid datasets keep their mkv videos in episodes/ (the
    # v3 rep is data-only), which the dir renumber below handles.
    info = _load_dataset_info(dataset_root)
    video_keys = [
        key
        for key, feature in (info.get("features") or {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if video_keys and not _has_gmsl2_episodes(dataset_root):
        raise ValueError(
            "Deleting an episode from a packed v3 dataset with embedded videos is not "
            "supported from the gateway (needs ffmpeg/av to re-cut the concatenated "
            "clips); use scripts/lerobot_edit_dataset.py instead."
        )

    _reindex_v3_data_and_meta_inplace(dataset_root, episode, mapping)

    # Hybrid datasets keep the raw gmsl2 per-episode dirs (mkv videos + remux
    # caches); renumber them to match the reindexed parquet.
    if _has_gmsl2_episodes(dataset_root):
        _renumber_gmsl2_episode_dirs(dataset_root, episode, mapping)
    # Derived AprilTag EE-pose sidecars are indexed by the old episode numbers
    # and can't be trusted after renumbering; drop them so replay regenerates.
    derived = dataset_root / "derived"
    if derived.is_dir():
        shutil.rmtree(derived, ignore_errors=True)


def _reindex_v3_data_and_meta_inplace(
    dataset_root: Path, deleted_episode: int, mapping: dict[int, int]
) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def _atomic_write_table(table: "pa.Table", path: Path) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)

    def _replace_col(table: "pa.Table", name: str, values: list) -> "pa.Table":
        idx = table.schema.get_field_index(name)
        if idx < 0:
            return table
        arr = pa.array(values, type=table.schema.field(name).type)
        return table.set_column(idx, name, arr)

    # 1) data parquet: drop the deleted episode's rows, remap episode_index for
    #    the survivors, and rebuild the contiguous global `index` across files.
    running = 0
    for data_file in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(data_file)
        if "episode_index" in table.column_names:
            table = table.filter(pc.not_equal(table["episode_index"], deleted_episode))
            table = _replace_col(
                table,
                "episode_index",
                [mapping.get(int(e), int(e)) for e in table["episode_index"].to_pylist()],
            )
        n = table.num_rows
        if "index" in table.column_names:
            table = _replace_col(table, "index", list(range(running, running + n)))
        running += n
        _atomic_write_table(table, data_file)
    total_frames = running

    # 2) meta/episodes: drop the deleted row, remap episode_index, and recompute
    #    the global row ranges from the surviving per-episode lengths.
    meta_ep_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if meta_ep_files:
        tables = [pq.read_table(f) for f in meta_ep_files]
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        combined = combined.filter(pc.not_equal(combined["episode_index"], deleted_episode))
        new_ep = [mapping[int(e)] for e in combined["episode_index"].to_pylist()]
        order = sorted(range(len(new_ep)), key=lambda i: new_ep[i])
        combined = combined.take(pa.array(order))
        combined = _replace_col(combined, "episode_index", sorted(new_ep))
        lengths = [int(x) for x in combined["length"].to_pylist()]
        starts: list[int] = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
        combined = _replace_col(combined, "dataset_from_index", starts)
        combined = _replace_col(combined, "dataset_to_index", [s + length for s, length in zip(starts, lengths)])
        # Collapse to a single meta/episodes file (index 0/0) so the layout stays
        # consistent after removing a row.
        combined = _replace_col(combined, "meta/episodes/chunk_index", [0] * combined.num_rows)
        combined = _replace_col(combined, "meta/episodes/file_index", [0] * combined.num_rows)
        _atomic_write_table(combined, meta_ep_files[0])
        for extra in meta_ep_files[1:]:
            extra.unlink()

    # 3) info.json: episode/frame totals and the single-split range.
    info_path = dataset_root / "meta" / "info.json"
    info = _load_dataset_info(dataset_root)
    if info:
        info["total_episodes"] = max(int(info.get("total_episodes") or 0) - 1, 0)
        info["total_frames"] = total_frames
        splits = info.get("splits")
        if isinstance(splits, dict) and "train" in splits:
            splits["train"] = f"0:{info['total_episodes']}"
        tmp = info_path.with_name(f".info.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, info_path)


def _renumber_gmsl2_episode_dirs(dataset_root: Path, episode: int, mapping: dict[int, int]) -> None:
    # gmsl2 datasets store one self-contained directory per episode and derive
    # the episode index purely from the directory name, so renumbering is a
    # rmtree of the victim plus a contiguous rename of the survivors.
    eps_dir = dataset_root / "episodes"
    victim = eps_dir / f"episode_{episode:06d}"
    if victim.is_dir():
        shutil.rmtree(victim)
    # Rename in ascending old-index order so every destination slot is free
    # before we move into it.
    for old in sorted(k for k in mapping if k > episode):
        src = eps_dir / f"episode_{old:06d}"
        dst = eps_dir / f"episode_{mapping[old]:06d}"
        if src.is_dir() and not dst.exists():
            os.replace(src, dst)


def _delete_gmsl2_episode(dataset_root: Path, episode: int, mapping: dict[int, int]) -> None:
    # Pure gmsl2 dataset (no v3 parquet): the episode dirs *are* the dataset.
    _renumber_gmsl2_episode_dirs(dataset_root, episode, mapping)
    # Derived AprilTag EE-pose sidecars are indexed by the old episode numbers
    # and can't be trusted after renumbering; drop them so replay regenerates.
    derived = dataset_root / "derived"
    if derived.is_dir():
        shutil.rmtree(derived, ignore_errors=True)


def _reindex_episode_side_metadata(
    dataset_root: Path, mapping: dict[int, int], deleted_episode: int
) -> None:
    # Annotations: { "annotations": { "<episode>": {..., "episode": <n>} } }
    try:
        store = _read_annotation_store(dataset_root)
        annotations = store.get("annotations")
        if isinstance(annotations, dict):
            remapped: dict[str, Any] = {}
            for key, value in annotations.items():
                try:
                    old = int(key)
                except (TypeError, ValueError):
                    continue
                if old == deleted_episode or old not in mapping:
                    continue
                new = mapping[old]
                if isinstance(value, dict):
                    value = {**value, "episode": new}
                remapped[str(new)] = value
            store["annotations"] = remapped
            _write_annotation_store(dataset_root, store)
    except (OSError, json.JSONDecodeError):
        pass

    # Mujoco validations: { "validations": [ {..., "episode": n, "datasetRoot": p} ] }
    try:
        payload = _read_validation_store(dataset_root)
        validations = payload.get("validations")
        if isinstance(validations, list):
            reindexed: list[dict[str, Any]] = []
            for item in validations:
                if not isinstance(item, dict):
                    continue
                try:
                    old = int(item.get("episode"))
                except (TypeError, ValueError):
                    continue
                if old == deleted_episode or old not in mapping:
                    continue
                reindexed.append({**item, "episode": mapping[old], "datasetRoot": str(dataset_root)})
            payload["validations"] = reindexed
            path = _validation_store_path(dataset_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp_path.replace(path)
    except (OSError, json.JSONDecodeError):
        pass


def _load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        with info_path.open("r", encoding="utf-8") as info_file:
            info = json.load(info_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return info if isinstance(info, dict) else {}


def _dataset_declared_fps(dataset_root: Path) -> int | None:
    """The rate the dataset was actually recorded at, from its own metadata, or None if it cannot say."""
    fps = _load_dataset_info(dataset_root).get("fps")
    try:
        value = int(fps)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _replay_fps(state: GatewayState, dataset_root: Path) -> int:
    """The frame rate a replay of *this dataset* must run at.

    `ReplayStatus.fps` is seeded from the recorder config's `dataset.fps`, which describes what the
    *next* recording will do -- it is not a property of the episode being replayed, and the two
    diverge the moment the recording rate is changed. Replaying at the wrong rate does not fail
    visibly; it does two quiet things instead. The preview video is encoded at that rate, so it runs
    against the timeline at fps_used/fps_recorded speed. And `fr3_gui_replay_runtime` sets the sim's
    `teleop_control_frequency` from it, so every command is given a fraction of the simulated time
    the recorder had between frames and the tracking score measures an under-integrated servo window
    rather than the trajectory. The dataset is the authority on its own frame rate.
    """
    return _dataset_declared_fps(dataset_root) or int(state.replay.fps or 30)


def _feature_names(info: dict[str, Any], column: str) -> list[str]:
    feature = (info.get("features") or {}).get(column) or {}
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        indexed = [(index, name) for name, index in names.items() if isinstance(index, int)]
        return [str(name) for _, name in sorted(indexed)]
    return []


def _first_finite(values: Any, default: float = 0.0) -> float:
    if values is None:
        return default
    if isinstance(values, (int, float)):
        value = float(values)
        return value if value == value else default
    if isinstance(values, list):
        for item in values:
            value = _first_finite(item, default=float("nan"))
            if value == value:
                return value
    return default


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        value = float(values)
        return [value] if value == value else []
    if isinstance(values, list):
        floats: list[float] = []
        for item in values:
            floats.extend(_as_float_list(item))
        return floats
    return []


def _axis_indices(names: list[str], values: list[float]) -> tuple[int | None, int | None, int | None]:
    lowered = [name.lower() for name in names]

    def match_axis(axis: str) -> int | None:
        patterns = (
            f".{axis}",
            f"_{axis}",
            f"/{axis}",
            f"position.{axis}",
            f"position_{axis}",
            f"translation.{axis}",
            f"translation_{axis}",
            f"ee_{axis}",
            f"tcp_{axis}",
        )
        for index, name in enumerate(lowered):
            if index >= len(values):
                continue
            if name == axis or any(pattern in name for pattern in patterns):
                return index
        return None

    x_index = match_axis("x")
    y_index = match_axis("y")
    z_index = match_axis("z")
    if x_index is None or y_index is None:
        if len(values) >= 3:
            return 0, 1, 2
    return x_index, y_index, z_index


def _gripper_width(values: list[float], names: list[str]) -> float:
    for index, name in enumerate(name.lower() for name in names):
        if index < len(values) and ("width_mm" in name or "gripper" in name):
            return values[index]
    return values[0] if values else 0.0


def _spread_ms(values: Any) -> float:
    timestamps = [value for value in _as_float_list(values) if value > 0]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)) * 1000.0


def _normalize_series(values: list[float], low: float = 8.0, high: float = 92.0) -> list[float]:
    if not values:
        return []
    finite_values = [value for value in values if value == value]
    if not finite_values:
        return [50.0 for _ in values]
    min_value = min(finite_values)
    max_value = max(finite_values)
    if abs(max_value - min_value) < 1e-9:
        return [50.0 for _ in values]
    scale = (high - low) / (max_value - min_value)
    return [low + (value - min_value) * scale if value == value else 50.0 for value in values]


def _dataset_replay_meta(state: GatewayState, dataset_root: Path, info: dict[str, Any]) -> dict[str, Any]:
    data_files = _dataset_data_files(dataset_root)
    episode_options = _dataset_episode_indices(dataset_root, info)
    if _has_gmsl2_episodes(dataset_root):
        total_episodes, total_frames = _gmsl2_dataset_stats(dataset_root)
    else:
        total_episodes = int(info.get("total_episodes") or 0)
        total_frames = int(info.get("total_frames") or 0)
    return {
        "datasetRoot": str(dataset_root),
        "datasetKind": _dataset_kind(state, dataset_root),
        "sourcePath": str(data_files[-1]) if data_files else "",
        "totalEpisodes": total_episodes,
        "episodeOptions": episode_options,
        "recordedFrames": total_frames,
    }


def _dataset_episode_indices(dataset_root: Path, info: dict[str, Any] | None = None) -> list[int]:
    if _has_gmsl2_episodes(dataset_root):
        indices = []
        for d in _gmsl2_episode_dirs(dataset_root):
            try:
                indices.append(int(d.name.split("_", 1)[1]))
            except (ValueError, IndexError):
                pass
        return sorted(indices)
    try:
        import pyarrow.parquet as pq
    except Exception:
        total_episodes = int((info or _load_dataset_info(dataset_root)).get("total_episodes") or 0)
        return list(range(total_episodes))

    episode_indices: set[int] = set()
    has_rows = False
    for data_file in _dataset_data_files(dataset_root):
        try:
            parquet = pq.ParquetFile(data_file)
            columns = parquet.schema_arrow.names
            if "episode_index" not in columns:
                if parquet.metadata and parquet.metadata.num_rows > 0:
                    has_rows = True
                continue
            table = pq.read_table(data_file, columns=["episode_index"])
            for value in table["episode_index"].to_pylist():
                if value is not None:
                    episode_indices.add(int(value))
        except Exception:
            continue
    if episode_indices:
        return sorted(episode_indices)
    total_episodes = int((info or _load_dataset_info(dataset_root)).get("total_episodes") or 0)
    if total_episodes:
        return list(range(total_episodes))
    return [0] if has_rows else []


def _episode_metadata_row(dataset_root: Path, episode: int) -> dict[str, Any] | None:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    for meta_file in sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet")):
        try:
            table = pq.read_table(meta_file)
        except Exception:
            continue
        if "episode_index" not in table.column_names:
            continue
        for row in table.to_pylist():
            try:
                if int(row.get("episode_index")) == int(episode):
                    return row
            except (TypeError, ValueError):
                continue
    return None


def _dataset_child_from_template(dataset_root: Path, template: str, **values: Any) -> Path | None:
    try:
        relative = template.format(**values)
    except (KeyError, IndexError, ValueError):
        return None
    candidate = (dataset_root / relative).resolve()
    try:
        candidate.relative_to(dataset_root.resolve())
    except ValueError:
        return None
    return candidate


def _resolve_data_file_for_episode(dataset_root: Path, info: dict[str, Any], episode: int) -> Path | None:
    row = _episode_metadata_row(dataset_root, episode)
    if row is None:
        return None
    try:
        chunk_index = int(row["data/chunk_index"])
        file_index = int(row["data/file_index"])
    except (KeyError, TypeError, ValueError):
        return None

    template = str(info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    candidates = [
        _dataset_child_from_template(
            dataset_root,
            template,
            chunk_index=chunk_index,
            file_index=file_index,
        ),
        dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet",
        dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:06d}.parquet",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _resolve_video_file_for_episode(
    dataset_root: Path,
    info: dict[str, Any],
    camera_key: str,
    episode: int,
) -> Path | None:
    row = _episode_metadata_row(dataset_root, episode)
    if row is None:
        return None
    try:
        chunk_index = int(row[f"videos/{camera_key}/chunk_index"])
        file_index = int(row[f"videos/{camera_key}/file_index"])
    except (KeyError, TypeError, ValueError):
        return None

    template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    candidates = [
        _dataset_child_from_template(
            dataset_root,
            template,
            video_key=camera_key,
            chunk_index=chunk_index,
            file_index=file_index,
        ),
        dataset_root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4",
        dataset_root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:06d}.mp4",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _selected_episode_for_dataset(state: GatewayState, dataset_root: Path, episode_options: list[int]) -> int:
    if not episode_options:
        return int(state.replay.episode or 0)
    selected_dataset = state.selected_replay_root
    replay_dataset = _resolve_dataset_root(state.repo_root, state.replay.datasetRoot or state.replay.dataset)
    is_active = False
    for candidate in (selected_dataset, replay_dataset):
        if candidate is None:
            continue
        try:
            if candidate.resolve() == dataset_root.resolve():
                is_active = True
                break
        except OSError:
            continue
    if is_active and int(state.replay.episode or 0) in episode_options:
        return int(state.replay.episode or 0)
    return episode_options[0]


def _active_replay_episode(state: GatewayState, dataset_root: Path) -> int | None:
    """The episode the operator picked, but only if they picked it on *this* dataset.

    `_selected_episode_for_dataset` answers a related question -- which of these candidate
    episodes to show -- and falls back to the first one when the pick is not among them.
    This answers "is there an explicit pick here at all", which is what decides whether a
    scan is allowed to hold itself to that episode instead of snapping to a neighbour.
    """
    replay_dataset = _resolve_dataset_root(state.repo_root, state.replay.datasetRoot or state.replay.dataset)
    for candidate in (state.selected_replay_root, replay_dataset):
        if candidate is None:
            continue
        try:
            if candidate.resolve() == dataset_root.resolve():
                return int(state.replay.episode or 0)
        except OSError:
            continue
    return None


def _recorded_dataset_status(dataset_root: Path) -> str:
    data_files = _dataset_data_files(dataset_root)
    if not data_files:
        return "missing"
    if data_files[0].suffix == ".mkv":
        valid = sum(1 for f in data_files if f.stat().st_size > 1024)
        return "loaded" if valid > 0 else "empty"
    has_empty_file = False
    for data_file in data_files:
        try:
            if data_file.stat().st_size == 0:
                has_empty_file = True
                continue
            with data_file.open("rb") as parquet_file:
                parquet_file.seek(-4, os.SEEK_END)
                if parquet_file.read(4) != b"PAR1":
                    return "unfinalized"
        except OSError:
            return "unreadable"
    return "empty" if has_empty_file and len(data_files) == 1 else "loaded"


def _action_has_ee_pose(info: dict[str, Any]) -> bool:
    features = info.get("features") or {}
    if not isinstance(features, dict):
        return False
    for key in ("action", "observation.state"):
        feature = features.get(key) or {}
        names = feature.get("names") if isinstance(feature, dict) else None
        if isinstance(names, list):
            lowered = {str(n).lower() for n in names}
            if all(any(name.endswith(suffix) for name in lowered) for suffix in ("ee.x", "ee.y", "ee.z")):
                return True
    return False


def _load_processing_meta(dataset_root: Path) -> dict[str, Any] | None:
    meta_path = _processing_meta_path(dataset_root)
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            payload = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _processing_meta_path(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "processing.json"


def _write_processing_meta(dataset_root: Path, payload: dict[str, Any]) -> None:
    meta_path = _processing_meta_path(dataset_root)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(meta_path)


def _online_sync_manifest_summary(dataset_root: Path) -> dict[str, Any] | None:
    if not _has_gmsl2_episodes(dataset_root):
        return None
    ep_dirs = _gmsl2_episode_dirs(dataset_root)
    if not ep_dirs:
        return None
    episodes: list[dict[str, Any]] = []
    present = 0
    ok_count = 0
    failed_count = 0
    missing_count = 0
    actual_frames_total = 0
    max_delta_ns: int | None = None
    failure_reasons: list[str] = []
    frame_count_mismatch = 0
    for ep_dir in ep_dirs:
        match = re.search(r"episode_(\d+)$", ep_dir.name)
        ep_index = int(match.group(1)) if match else len(episodes)
        manifest_path = ep_dir / "online_sync_manifest.json"
        item: dict[str, Any] = {
            "episode": ep_index,
            "present": manifest_path.is_file(),
            "ok": False,
            "actualFrames": None,
            "frameCountByCamera": {},
            "maxSofDeltaMs": None,
            "failure": "missing online_sync_manifest.json",
        }
        if not manifest_path.is_file():
            missing_count += 1
            failure_reasons.append(f"episode {ep_index}: missing online_sync_manifest.json")
            episodes.append(item)
            continue
        present += 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failed_count += 1
            item["failure"] = f"invalid online_sync_manifest.json: {exc}"
            failure_reasons.append(f"episode {ep_index}: {item['failure']}")
            episodes.append(item)
            continue
        if not isinstance(manifest, dict):
            failed_count += 1
            item["failure"] = "invalid online_sync_manifest.json payload"
            failure_reasons.append(f"episode {ep_index}: {item['failure']}")
            episodes.append(item)
            continue
        counts = manifest.get("frame_count_by_camera") if isinstance(manifest.get("frame_count_by_camera"), dict) else {}
        max_by_camera = manifest.get("max_abs_delta_ns_by_camera") if isinstance(manifest.get("max_abs_delta_ns_by_camera"), dict) else {}
        try:
            actual_frames = int(manifest.get("actual_frames") or 0)
        except (TypeError, ValueError):
            actual_frames = 0
        deltas: list[int] = []
        for value in max_by_camera.values():
            try:
                delta = int(value)
            except (TypeError, ValueError):
                continue
            deltas.append(delta)
            max_delta_ns = delta if max_delta_ns is None else max(max_delta_ns, delta)
        mismatch = False
        for camera, value in counts.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                mismatch = True
                continue
            if actual_frames and count != actual_frames:
                mismatch = True
        if mismatch:
            frame_count_mismatch += 1
        ok = bool(manifest.get("ok")) and actual_frames > 0 and not mismatch
        failure = str(manifest.get("failure") or "").strip()
        if not ok and not failure:
            failure = "manifest ok=false" if not manifest.get("ok") else "frame count mismatch"
        if ok:
            ok_count += 1
            actual_frames_total += actual_frames
        else:
            failed_count += 1
            failure_reasons.append(f"episode {ep_index}: {failure}")
        item.update({
            "ok": ok,
            "actualFrames": actual_frames,
            "frameCountByCamera": {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))},
            "maxSofDeltaMs": (max(deltas) / 1_000_000.0) if deltas else None,
            "failure": failure,
        })
        episodes.append(item)
    summary: dict[str, Any] = {
        "present": present,
        "missing": missing_count,
        "ok": ok_count,
        "failed": failed_count,
        "totalEpisodes": len(ep_dirs),
        "actualFrames": actual_frames_total,
        "maxSofDeltaMs": (max_delta_ns / 1_000_000.0) if max_delta_ns is not None else None,
        "frameCountMismatch": frame_count_mismatch,
        "failureReasons": failure_reasons[:8],
        "episodes": episodes[:12],
    }
    if present == 0:
        summary["status"] = "missing"
        summary["message"] = "No online_sync_manifest.json files found"
    elif failed_count or missing_count:
        summary["status"] = "fail"
        summary["message"] = f"{ok_count}/{len(ep_dirs)} episodes passed online-sync manifest checks"
    else:
        summary["status"] = "pass"
        max_delta = summary["maxSofDeltaMs"]
        suffix = f", max SOF delta {max_delta:.3f} ms" if isinstance(max_delta, (int, float)) else ""
        summary["message"] = f"{ok_count}/{len(ep_dirs)} episodes passed online-sync manifest checks{suffix}"
    return summary


def _online_sync_manifest_check(dataset_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    summary = _online_sync_manifest_summary(dataset_root)
    if summary is None:
        return None, None
    status = str(summary.get("status") or "missing")
    if status == "pass":
        check_status = "pass"
    elif status == "missing":
        check_status = "warn"
    else:
        check_status = "fail"
    return summary, {
        "name": "online_sync_manifest",
        "status": check_status,
        "message": summary.get("message") or "online sync manifest unavailable",
    }


def _qc_warning_messages(qc_or_item: dict[str, Any]) -> list[str]:
    """Every warning a QC run raised, in check order.

    Reads either a stored QC result (``checks``) or a snapshot processing item (``qcChecks``),
    because the export gate and the UI have to name the same warnings.
    """
    checks = qc_or_item.get("checks")
    if not isinstance(checks, list):
        checks = qc_or_item.get("qcChecks")
    if not isinstance(checks, list):
        return []
    return [
        f"{check.get('name')}: {check.get('message')}"
        for check in checks
        if isinstance(check, dict) and str(check.get("status") or "").lower() == "warn"
    ]


def _processing_item_from_dataset(
    dataset_root: Path,
    *,
    attached_processes: set[str] | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    info = _load_dataset_info(dataset_root)
    modified_s = _dataset_modified_s(dataset_root)
    if _has_gmsl2_episodes(dataset_root):
        total_episodes, total_frames = _gmsl2_dataset_stats(dataset_root)
    else:
        total_episodes = int(info.get("total_episodes") or 0)
        total_frames = int(info.get("total_frames") or 0)
    base_item = {
        "path": str(dataset_root),
        "name": dataset_root.name,
        "trajectoryVersion": None,
        "qcSummary": "QC not run",
        "message": "No processing metadata yet",
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_s)) if modified_s else "",
        "totalEpisodes": total_episodes,
        "totalFrames": total_frames,
        "validFramesPct": None,
        "logTail": [],
        "onlineSync": _online_sync_manifest_summary(dataset_root),
        "qcChecks": [],
        "ikEvaluation": None,
        "timestampSync": None,
    }

    meta = _load_processing_meta(dataset_root)
    if meta:
        active_version = meta.get("active_version")
        versions = meta.get("versions") if isinstance(meta.get("versions"), dict) else {}
        current_job = meta.get("current_job") if isinstance(meta.get("current_job"), dict) else {}
        version_info = versions.get(active_version) if isinstance(active_version, str) else None
        qc = version_info.get("qc") if isinstance(version_info, dict) else None
        if isinstance(current_job, dict) and current_job.get("status") in ("queued", "running"):
            status = str(current_job["status"])
            message = current_job.get("message") or f"{current_job.get('kind') or 'job'} {status}"
            if attached_processes is not None and str(dataset_root) not in attached_processes:
                updated_s = _parse_iso_epoch_s(current_job.get("updated_at"))
                age_s = (time.time() if now_s is None else now_s) - updated_s if updated_s is not None else None
                if age_s is None or age_s > _PROCESSING_STALE_RUNNING_S:
                    status = "error"
                    message = (
                        "EE trajectory job is stale: metadata says running, "
                        "but this gateway is not attached to a live process"
                    )
        elif isinstance(current_job, dict) and current_job.get("status") in ("failed", "error"):
            status = "error"
            message = current_job.get("message") or f"{current_job.get('kind') or 'job'} failed"
        elif isinstance(qc, dict):
            qc_status = str(qc.get("status") or "").lower()
            if qc_status == "pass":
                status = "qc_pass"
            elif qc_status in ("fail", "failed"):
                status = "qc_failed"
            elif qc_status == "warn":
                # Its own state, not `pose_ready`. A QC that ran and warned is not a QC that has
                # not run: the two used to be the same status, so a single warn removed the
                # dataset from Dataset Export while the page still read "QC pending" -- an
                # export blocked for a reason nobody was shown.
                status = "qc_warn"
            else:
                status = "pose_ready"
            message = (
                qc.get("reason")
                or qc.get("message")
                or (_qc_warning_messages(qc)[0] if qc_status == "warn" else None)
                or "QC available"
            )
        elif version_info is not None:
            status = "pose_ready"
            message = "EE trajectory generated; QC pending"
        else:
            status = "pose_missing"
            message = "Awaiting trajectory generation"
        return {
            **base_item,
            "status": status,
            "trajectoryVersion": active_version if isinstance(active_version, str) else None,
            "qcSummary": (qc.get("summary") if isinstance(qc, dict) and qc.get("summary") else None)
                or (f"valid {qc['valid_frames_pct']}%" if isinstance(qc, dict) and qc.get("valid_frames_pct") is not None else base_item["qcSummary"]),
            "message": message,
            "validFramesPct": float(qc["valid_frames_pct"]) if isinstance(qc, dict) and qc.get("valid_frames_pct") is not None else None,
            "logTail": list(current_job.get("log_tail") or []) if isinstance(current_job, dict) else [],
            "qcChecks": list(qc.get("checks") or []) if isinstance(qc, dict) else [],
            "ikEvaluation": qc.get("ik_evaluation") if isinstance(qc, dict) else None,
            "timestampSync": qc.get("timestamp_sync") if isinstance(qc, dict) else None,
        }

    if not _dataset_data_files(dataset_root):
        return {
            **base_item,
            "status": "pose_missing",
            "message": "No parquet found under data/",
            "qcSummary": "No data files",
        }
    if _action_has_ee_pose(info):
        return {
            **base_item,
            "status": "pose_ready",
            "trajectoryVersion": "v1",
            "qcSummary": "QC not run",
            "message": "EE pose present in observation.state/action; ready for QC",
        }
    return {
        **base_item,
        "status": "pose_missing",
        "message": "EE trajectory not derived from raw observations yet",
    }


def _dataset_is_complete(dataset_root: Path) -> bool:
    if _has_gmsl2_episodes(dataset_root):
        return any(
            f.stat().st_size > 1024
            for f in (dataset_root / "episodes").glob("episode_*/*.mkv")
        )
    info = _load_dataset_info(dataset_root)
    if not info:
        return False
    data_files = _dataset_data_files(dataset_root)
    if not data_files:
        return False
    has_finalized = False
    for data_file in data_files:
        try:
            if data_file.stat().st_size <= 4:
                continue
            with data_file.open("rb") as parquet_file:
                parquet_file.seek(-4, os.SEEK_END)
                if parquet_file.read(4) == b"PAR1":
                    has_finalized = True
                    break
        except OSError:
            continue
    if not has_finalized:
        return False
    for camera_key in _camera_keys(info):
        camera_dir = dataset_root / "videos" / camera_key
        if not camera_dir.is_dir():
            return False
        if not any(camera_dir.glob("chunk-*/*.mp4")):
            return False
    return True


def _complete_dataset_candidates(state: GatewayState) -> list[Path]:
    return [root for root in _dataset_root_candidates(state) if _dataset_is_complete(root)]


def _processing_items(
    state: GatewayState,
    *,
    attached_processes: set[str] | None = None,
) -> list[dict[str, Any]]:
    now_s = time.time()
    return [
        _processing_item_from_dataset(root, attached_processes=attached_processes, now_s=now_s)
        for root in _complete_dataset_candidates(state)
    ]


def _set_datasets_root(state: GatewayState, raw_path: str) -> bool:
    requested = raw_path.strip()
    if not requested:
        raise ValueError("missing path")
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = state.repo_root / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"cannot resolve path: {exc}") from exc
    created = False
    if not resolved.is_dir():
        if resolved.exists():
            raise ValueError(f"not a directory: {resolved}")
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"failed to create directory: {exc}") from exc
        created = True
    state.datasets_root = resolved
    state.selected_replay_root = None
    state.dataset_cache_ready = False
    state.processing_cache_ready = False
    state.dataset_scan_signature = ()
    state.processing_scan_signature = ()
    return created


# --- Camera self-check -------------------------------------------------------
#
# Answers "has a fixed camera been bumped since it was calibrated?" by comparing
# one frame per camera against a baseline captured at calibration time. The
# analysis lives in metrology.cli.rig_shift_check; this side only grabs frames
# and shells out, because the gateway deliberately carries no cv2.
#
# Frames come one at a time through the existing preview pipeline. Argus opens
# are already serialized by the preview spawn lock, and asking for eleven at once
# is the exact pattern that wedges it.

_RIG_CHECK_SUBDIR = Path("outputs") / "metrology" / "rig_check"
_RIG_CHECK_FRAME_TIMEOUT_S = 12.0


def _rig_check_root(state: GatewayState) -> Path:
    return state.repo_root / _RIG_CHECK_SUBDIR


def _cv2_python(repo_root: Path) -> Path | None:
    """First interpreter that can actually import cv2.

    Checked rather than assumed: the gateway runs on the system python, which
    has no cv2, and a missing import surfaces here as a clear message instead of
    a traceback from a subprocess.
    """
    candidates = [
        _venv_python3(repo_root, prefer_fr3=True),
        Path("/home/nvidia/Code/infer/.venv-fr3/bin/python3"),
        Path(sys.executable),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        try:
            probe = subprocess.run(
                [key, "-c", "import cv2"], capture_output=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return candidate
    return None


def _cameras_busy_reason(state: GatewayState) -> str:
    """Why the preview pipelines cannot open the cameras right now, if they can't.

    The recorder holds Argus exclusively from Connect onwards -- including the
    'armed' state, which is connected but not yet writing. Without this the
    self-check would spend a full timeout per camera and then report an empty
    result, which reads like a broken check rather than a busy rig.
    """
    if state.camera_preview_suspended:
        return "相机已被录制器占用（Connect 之后 Argus 是独占的）。请先在采集页 Disconnect，再运行自检。"
    if str(state.recording.state) in {"armed", "recording", "saving", "discarding", "review"}:
        return (
            f"录制会话处于 {state.recording.state} 状态，相机被占用。"
            "请先 Disconnect 释放相机，再运行自检。"
        )
    return ""


def _capture_camera_frames(state: GatewayState, out_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    """One JPEG per configured camera, captured sequentially."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    captured: list[str] = []
    failed: list[dict[str, str]] = []
    cameras = [device for device in state.devices if device.get("kind") == "camera"]
    for device in cameras:
        device_id = str(device.get("id"))
        params = _camera_preview_params(state, device_id)
        if params is None:
            failed.append({"camera": device_id, "reason": "no preview parameters"})
            continue
        sensor_id, sensor_mode, width, height, fps = params
        with state.camera_preview_lock:
            state.camera_preview_last_access[device_id] = time.time()
        if not _ensure_camera_preview(
            state,
            device_id=device_id,
            sensor_id=sensor_id,
            sensor_mode=sensor_mode,
            source_width=width,
            source_height=height,
            source_fps=fps,
        ):
            failed.append({"camera": device_id, "reason": "preview pipeline unavailable"})
            continue

        deadline = time.monotonic() + _RIG_CHECK_FRAME_TIMEOUT_S
        frame: bytes | None = None
        while time.monotonic() < deadline:
            with state.camera_preview_lock:
                cached = state.camera_preview_frames.get(device_id)
                state.camera_preview_last_access[device_id] = time.time()
            if cached is not None:
                frame = cached[0]
                break
            time.sleep(0.1)
        if frame is None:
            failed.append({"camera": device_id, "reason": "no frame within timeout"})
            continue
        (out_dir / f"{device_id}.jpg").write_bytes(frame)
        captured.append(device_id)
    return captured, failed


def _rig_check_baseline_meta(state: GatewayState) -> dict[str, Any]:
    path = _rig_check_root(state) / "baseline" / "baseline.json"
    if not path.is_file():
        return {"exists": False}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": False}
    meta["exists"] = True
    return meta


def _capture_rig_check_baseline(state: GatewayState) -> dict[str, Any]:
    """Record what the cameras see right now as the reference."""
    busy = _cameras_busy_reason(state)
    if busy:
        return {"ok": False, "error": busy}
    baseline_dir = _rig_check_root(state) / "baseline"
    captured, failed = _capture_camera_frames(state, baseline_dir)
    if not captured:
        state.log("warn", "Rig-check baseline: no camera frames captured")
        return {"ok": False, "error": "no camera frames captured", "failed": failed}

    meta = {
        "captured_at": _now_iso(),
        "cameras": captured,
        "failed": failed,
        # Which calibration this baseline belongs to. A baseline outliving the
        # calibration it was taken against is meaningless, so the pairing is
        # recorded rather than assumed.
        "intrinsics_run": state.calibration.intrinsicsRun,
        "extrinsics_run": state.calibration.extrinsicsRun,
    }
    (baseline_dir / "baseline.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    state.log("info", f"Rig-check baseline captured for {len(captured)} camera(s)")
    return {"ok": True, "baseline": meta}


def _run_rig_check(state: GatewayState) -> dict[str, Any]:
    busy = _cameras_busy_reason(state)
    if busy:
        return {"ok": False, "error": busy}
    root = _rig_check_root(state)
    baseline_dir = root / "baseline"
    if not any(baseline_dir.glob("*.jpg")):
        return {
            "ok": False,
            "error": "no baseline captured yet",
            "hint": "标定完成后先采集一次基线，自检才有比较对象。",
        }

    python = _cv2_python(state.repo_root)
    if python is None:
        return {"ok": False, "error": "no interpreter with cv2 available for the self-check"}

    current_dir = root / "current"
    captured, failed = _capture_camera_frames(state, current_dir)
    if not captured:
        return {"ok": False, "error": "no camera frames captured", "failed": failed}

    result_path = root / "last_result.json"
    intrinsics_run = state.calibration.intrinsicsRun
    command = [
        str(python),
        "-m",
        "metrology.cli.rig_shift_check",
        "--baseline",
        str(baseline_dir),
        "--current",
        str(current_dir),
        "--out",
        str(result_path),
    ]
    if intrinsics_run:
        intrinsics_dir = state.repo_root / "outputs" / "calibration" / intrinsics_run
        if intrinsics_dir.is_dir():
            command += ["--intrinsics-dir", str(intrinsics_dir)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(state.repo_root / "third_party" / "opencv_kalibr")
    try:
        proc = subprocess.run(
            command,
            cwd=str(state.repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        state.log("warn", f"Rig-check failed to run: {exc}")
        return {"ok": False, "error": str(exc)}

    if proc.returncode != 0 or not result_path.is_file():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        state.log("warn", f"Rig-check exited {proc.returncode}")
        return {"ok": False, "error": detail[-1] if detail else f"exit {proc.returncode}"}

    report = json.loads(result_path.read_text(encoding="utf-8"))
    report["failed_captures"] = failed
    report["baseline"] = _rig_check_baseline_meta(state)
    level = "warn" if report.get("overall") in {"moved", "suspect", "inconclusive"} else "info"
    state.log(level, f"Rig self-check: {report.get('overall')} — {report.get('guidance', '')}")
    return {"ok": True, "report": report}


def _last_rig_check(state: GatewayState) -> dict[str, Any]:
    path = _rig_check_root(state) / "last_result.json"
    report: dict[str, Any] | None = None
    if path.is_file():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    return {"ok": True, "report": report, "baseline": _rig_check_baseline_meta(state)}


# --- Extrinsics calibration --------------------------------------------------
#
# Runs the same three steps the 0804 calibration ran by hand, on an already
# recorded ChArUco sweep: detect corners, bundle-adjust the camera poses with no
# robot in the loop, then export the run layout production reads.
#
# Intrinsics are reused, not re-fitted. Moving a camera changes where it is, not
# what its lens does, and re-fitting intrinsics needs a different capture anyway
# (per-camera sweeps that reach the frame edge, not one shared sweep).

_CALIB_INTRINSICS_REPORT = Path("outputs") / "metrology" / "calib_final" / "intrinsics_report.json"
# The tracking config is the authority on which calibration production consumes.
# Reading it means the GUI reports what is actually in use rather than a second,
# independently drifting record of it.
_TRACKING_CONFIG = (
    Path("third_party") / "opencv_kalibr" / "hikon_cube_tracking_offline"
    / "config_thor" / "april_cube_tracking_in_robot_base_thor.yaml"
)


def _load_active_calibration_runs(state: GatewayState) -> None:
    import yaml  # local, matching how the rest of this module pulls it in

    path = state.repo_root / _TRACKING_CONFIG
    if not path.is_file():
        return
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        state.log("warn", f"Could not read tracking config for calibration runs: {exc}")
        return
    calib = config.get("calibration") or {}
    tracker = config.get("cube_tracker") or {}
    state.calibration.intrinsicsRun = str(calib.get("intrinsics_run_name", "") or "").strip()
    state.calibration.extrinsicsRun = str(calib.get("fixed_camera_run_name", "") or "").strip()
    model = str(tracker.get("camera_model", "") or "").strip()
    if state.calibration.intrinsicsRun:
        state.calibration.message = (
            f"生产使用 {state.calibration.intrinsicsRun}"
            + (f" · {model}" if model else "")
        )
        state.log(
            "info",
            f"Active calibration: {state.calibration.intrinsicsRun} / {state.calibration.extrinsicsRun}",
        )


def _calibration_session_payload(state: GatewayState) -> dict[str, Any]:
    session = state.calibration_session
    return {
        "active": session.active,
        "stage": session.stage,
        "datasetName": session.datasetName,
        "datasetRoot": session.datasetRoot,
        "currentIndex": session.currentIndex,
        "message": session.message,
        "recorderState": state.recording.state,
        "steps": [
            {
                "kind": step.kind,
                "camera": step.camera,
                "status": step.status,
                "episodeIndex": step.episodeIndex,
                "note": step.note,
            }
            for step in session.steps
        ],
    }


def _start_calibration_session(state: GatewayState, cameras_arg: str = "") -> dict[str, Any]:
    if state.calibration_session.active:
        return {"ok": False, "error": "标定会话已在进行中"}
    if state.calibration.state == "running":
        return {"ok": False, "error": "上一次标定解算尚未结束"}

    wanted = [c.strip() for c in cameras_arg.split(",") if c.strip()]
    cameras = [str(d.get("id")) for d in state.devices if d.get("kind") == "camera"]
    if wanted:
        cameras = [c for c in cameras if c in set(wanted)]
    if not cameras:
        return {"ok": False, "error": "配置里没有相机"}

    name = f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
    datasets_dir = _task_datasets_dir(state) or (state.datasets_root or state.repo_root / "outputs" / "datasets")
    # One episode per camera, then one shared episode for the rig. The order is
    # the order the operator will walk the room in.
    steps = [CalibrationStep(kind="intrinsics", camera=camera) for camera in cameras]
    steps.append(CalibrationStep(kind="extrinsics", camera=""))

    state.calibration_session = CalibrationSession(
        active=True,
        stage="capture",
        datasetName=name,
        datasetRoot=str(Path(datasets_dir) / name),
        steps=steps,
        currentIndex=0,
        message="按提示逐台录制；被遮挡或不需要的相机可以跳过。",
    )
    state.log("info", f"Calibration session {name} started with {len(cameras)} camera(s)")
    return {"ok": True, "session": _calibration_session_payload(state)}


def _calibration_session_advance(state: GatewayState) -> None:
    """Move to the first step still needing a capture, or declare the set ready."""
    session = state.calibration_session
    for index, step in enumerate(session.steps):
        if step.status in {"pending", "recording"}:
            session.currentIndex = index
            return
    session.currentIndex = len(session.steps)
    captured = [s for s in session.steps if s.status == "captured"]
    extrinsics_done = any(s.kind == "extrinsics" and s.status == "captured" for s in session.steps)
    if not extrinsics_done:
        session.stage = "capture"
        session.message = "外参采集被跳过了——没有它无法解算相机之间的位姿。请补录。"
        return
    session.stage = "ready"
    session.message = f"采集完成（{len(captured)} 段），可以开始解算。"


def _calibration_step_record(state: GatewayState, action: str) -> dict[str, Any]:
    session = state.calibration_session
    if not session.active or session.stage != "capture":
        return {"ok": False, "error": "没有进行中的采集步骤"}
    if session.currentIndex >= len(session.steps):
        return {"ok": False, "error": "所有步骤已完成"}
    step = session.steps[session.currentIndex]

    try:
        if action == "start":
            # Checked here so the operator gets the next action rather than the
            # recorder's own English precondition message.
            if state.recording.state in {"idle", "error"}:
                return {
                    "ok": False,
                    "error": "相机还没连接。请先到「采集」页点 Connect，等相机全部就绪后再回来录制。",
                }
            _start_episode(state)
            step.status = "recording"
            session.message = (
                f"正在录制 {step.camera or '外参'}——按提示挥板，完成后点「保存本段」。"
            )
        elif action in {"save", "discard"}:
            _stop_recorder(state, action)
            if action == "save":
                step.status = "captured"
                step.episodeIndex = int(state.recording.savedEpisodes)
                _calibration_session_advance(state)
            else:
                step.status = "pending"
                session.message = "本段已丢弃，可以重录。"
        else:
            return {"ok": False, "error": f"未知动作 {action}"}
    except (RuntimeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "session": _calibration_session_payload(state)}


def _calibration_step_skip(state: GatewayState) -> dict[str, Any]:
    session = state.calibration_session
    if not session.active or session.currentIndex >= len(session.steps):
        return {"ok": False, "error": "没有可跳过的步骤"}
    step = session.steps[session.currentIndex]
    if step.status == "recording":
        return {"ok": False, "error": "正在录制中，请先保存或丢弃"}
    step.status = "skipped"
    # A skipped camera keeps whatever intrinsics it already has; a skipped
    # extrinsics capture leaves nothing to solve, which advance() calls out.
    step.note = "已跳过：沿用现有内参" if step.kind == "intrinsics" else "已跳过"
    _calibration_session_advance(state)
    return {"ok": True, "session": _calibration_session_payload(state)}


def _cancel_calibration_session(state: GatewayState) -> dict[str, Any]:
    if state.recording.state in {"recording", "saving", "discarding"}:
        return {"ok": False, "error": "正在录制中，请先保存或丢弃当前段"}
    state.calibration_session = CalibrationSession()
    state.log("info", "Calibration session cancelled")
    return {"ok": True, "session": _calibration_session_payload(state)}


def _marker_tcp_root(state: GatewayState) -> Path:
    return state.repo_root / "outputs" / "metrology" / "marker_tcp_repeatability"


def _marker_tcp_session_path(state: GatewayState) -> Path | None:
    session = state.marker_tcp_session
    if not session.sessionRoot:
        return None
    return Path(session.sessionRoot) / "session.json"


def _write_marker_tcp_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _save_marker_tcp_session(state: GatewayState) -> None:
    path = _marker_tcp_session_path(state)
    if path is None:
        return
    _write_marker_tcp_json(path, asdict(state.marker_tcp_session))


def _marker_tcp_session_payload(state: GatewayState) -> dict[str, Any]:
    session = state.marker_tcp_session
    return asdict(session)


def _valid_marker_tcp_side(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError("side must be left or right")
    return normalized


def _start_marker_tcp_session(state: GatewayState) -> dict[str, Any]:
    if state.marker_tcp_session.active:
        return {"ok": False, "error": "marker→TCP 采集会话已在进行中"}
    name = f"marker_tcp_{time.strftime('%Y%m%d_%H%M%S')}"
    root = _marker_tcp_root(state) / name
    root.mkdir(parents=True, exist_ok=True)
    state.marker_tcp_session = MarkerTcpSession(
        active=True,
        sessionName=name,
        sessionRoot=str(root),
        stage="capture",
        message="采集 left/right UMI 的重复性样本；本页只记录条件与原始录制，不假设已解出 TCP 外参。",
    )
    _save_marker_tcp_session(state)
    state.log("info", f"Marker→TCP repeatability session started: {root}")
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _cancel_marker_tcp_session(state: GatewayState) -> dict[str, Any]:
    if state.recording.state in {"recording", "saving", "discarding"} and state.marker_tcp_session.pendingSampleId:
        return {"ok": False, "error": "正在录制 marker→TCP 样本，请先保存或丢弃当前段"}
    state.marker_tcp_session = MarkerTcpSession()
    state.log("info", "Marker→TCP repeatability session cancelled")
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _marker_tcp_pending_sample(state: GatewayState) -> MarkerTcpSample | None:
    pending_id = state.marker_tcp_session.pendingSampleId
    if not pending_id:
        return None
    for sample in state.marker_tcp_session.samples:
        if sample.id == pending_id:
            return sample
    return None


def _marker_tcp_record_sample(state: GatewayState, action: str, *, side: str = "", condition: str = "") -> dict[str, Any]:
    session = state.marker_tcp_session
    if not session.active or session.stage not in {"capture", "failed"}:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    action = str(action or "start").strip().lower()
    try:
        if action == "start":
            if session.pendingSampleId:
                return {"ok": False, "error": "已有样本正在录制，请先保存或丢弃"}
            side_norm = _valid_marker_tcp_side(side)
            condition_text = str(condition or "").strip()
            if not condition_text:
                return {"ok": False, "error": "condition 不能为空，例如 same_mount_01 / remount_03 / light_push_x"}
            if state.recording.state in {"idle", "error"}:
                return {"ok": False, "error": "相机还没连接。请先到「采集」页 Connect，再回来采 marker→TCP 样本。"}
            _start_episode(state)
            sample = MarkerTcpSample(
                id=f"sample_{len(session.samples) + 1:03d}",
                side=side_norm,
                condition=condition_text,
                source="recording",
                status="recording",
                datasetRoot=state.recording.datasetRoot,
                episodeIndex=int(state.recording.episodeIndex),
                createdAt=datetime.now(timezone.utc).isoformat(),
            )
            session.samples.append(sample)
            session.pendingSampleId = sample.id
            session.message = f"正在录制 {side_norm} · {condition_text}；结束后保存或丢弃本段。"
        elif action in {"save", "discard"}:
            sample = _marker_tcp_pending_sample(state)
            if sample is None:
                return {"ok": False, "error": "没有正在录制的 marker→TCP 样本"}
            episode_index = sample.episodeIndex if sample.episodeIndex >= 0 else int(state.recording.episodeIndex)
            if state.recording.state == "armed":
                # The recorder can auto-complete a fixed-duration episode and return to
                # armed before the operator clicks save/discard. Finish this panel's
                # sample bookkeeping without sending another command to an idle recorder.
                state.recording.message = "Recorder is armed; marker→TCP sample finalized in calibration page"
            else:
                _stop_recorder(state, action)
            sample.datasetRoot = state.recording.datasetRoot or sample.datasetRoot
            sample.episodeIndex = episode_index
            if action == "save":
                sample.status = "saved"
                sample.note = "raw recording saved; register static_transform.json after solving nominal transform"
                session.message = "样本已保存。生成 static_transform.json 后在本页登记路径，再生成 repeatability 报告。"
            else:
                sample.status = "discarded"
                sample.note = "discard requested; ignored by repeatability report"
                session.message = "样本已丢弃，可以重录。"
            session.pendingSampleId = ""
        else:
            return {"ok": False, "error": f"未知动作 {action}"}
    except (RuntimeError, ValueError) as exc:
        session.stage = "failed"
        session.message = str(exc)
        _save_marker_tcp_session(state)
        return {"ok": False, "error": str(exc), "markerTcp": _marker_tcp_session_payload(state)}
    session.stage = "capture"
    _save_marker_tcp_session(state)
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _register_marker_tcp_static_transform(
    state: GatewayState,
    *,
    path_arg: str,
    side: str,
    condition: str,
) -> dict[str, Any]:
    session = state.marker_tcp_session
    if not session.active:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    try:
        side_norm = _valid_marker_tcp_side(side)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    path_text = str(path_arg or "").strip()
    if not path_text:
        return {"ok": False, "error": "path 不能为空"}
    p = Path(path_text).expanduser()
    if not p.is_absolute():
        p = state.repo_root / p
    if not p.is_file():
        return {"ok": False, "error": f"static_transform.json not found: {p}"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        transform_keys = ("T_ee_cube", "T_marker_tcp", "T_marker_to_tcp", "T_cube_ee")
        if not any(key in data for key in transform_keys):
            return {"ok": False, "error": "JSON 里没有 T_ee_cube / T_marker_tcp / T_marker_to_tcp / T_cube_ee"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"无法读取 static transform: {exc}"}
    condition_text = str(condition or p.parent.name).strip() or p.parent.name
    sample = MarkerTcpSample(
        id=f"sample_{len(session.samples) + 1:03d}",
        side=side_norm,
        condition=condition_text,
        source="static_transform",
        status="registered",
        staticTransformPath=str(p),
        createdAt=datetime.now(timezone.utc).isoformat(),
    )
    session.samples.append(sample)
    session.stage = "capture"
    session.message = f"已登记 {side_norm} · {condition_text}: {p.name}"
    _save_marker_tcp_session(state)
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _run_marker_tcp_repeatability_report(state: GatewayState) -> dict[str, Any]:
    session = state.marker_tcp_session
    if not session.active:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    paths = [sample.staticTransformPath for sample in session.samples if sample.status == "registered" and sample.staticTransformPath]
    if len(paths) < 2:
        return {"ok": False, "error": "至少登记 2 个 static_transform.json 后才能生成 repeatability 报告"}
    try:
        metrology_root = state.repo_root / "third_party" / "opencv_kalibr"
        if str(metrology_root) not in sys.path:
            sys.path.insert(0, str(metrology_root))
        from metrology.cli.marker_tcp_repeatability import load_transform_bundle, summarize_repeatability

        bundles = [load_transform_bundle(Path(path)) for path in paths]
        report = summarize_repeatability(bundles)
        out_path = Path(session.sessionRoot) / "repeatability_report.json"
        _write_marker_tcp_json(out_path, report)
        session.reportPath = str(out_path)
        session.stage = "done"
        p95 = report.get("translation_error_mm", {}).get("p95")
        session.message = f"repeatability report complete: translation p95={p95:.3f} mm" if isinstance(p95, (int, float)) else "repeatability report complete"
        state.log("info", f"Marker→TCP repeatability report written: {out_path}")
    except Exception as exc:  # noqa: BLE001
        session.stage = "failed"
        session.message = str(exc)
        _save_marker_tcp_session(state)
        return {"ok": False, "error": str(exc), "markerTcp": _marker_tcp_session_payload(state)}
    _save_marker_tcp_session(state)
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _newest_calibration_dataset(state: GatewayState) -> Path | None:
    """Most recent recorded sweep that looks like a calibration capture."""
    roots = [state.datasets_root] if state.datasets_root else []
    roots.append(state.repo_root / "outputs" / "datasets")
    for root in roots:
        if not root or not root.is_dir():
            continue
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "episodes").is_dir() and "calib" in path.name.lower()
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _unmoved_cameras(state: GatewayState) -> list[str]:
    """Cameras the last self-check found still in place.

    These define the base frame during the re-export. A camera that was bumped
    must not vote on where the frame is -- aligning onto its stale pose would
    drag the whole rig toward it.
    """
    path = _rig_check_root(state) / "last_result.json"
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [name for name, entry in (report.get("cameras") or {}).items() if entry.get("verdict") == "ok"]


def _fail_calibration(state: GatewayState, message: str) -> None:
    """One place that marks a calibration failed, so the wizard cannot be left
    sitting on "solving" by a path that forgot to update it."""
    state.calibration.state = "failed"
    state.calibration.message = message
    state.log("warn", f"Calibration failed: {message}")
    if state.calibration_session.active:
        state.calibration_session.stage = "failed"
        state.calibration_session.message = message


def _calibration_step(
    state: GatewayState, python: Path, args: list[str], *, label: str, timeout: int
) -> subprocess.CompletedProcess[str] | None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(state.repo_root / "third_party" / "opencv_kalibr")
    state.calibration.message = label
    state.log("info", f"Calibration: {label}")
    try:
        return subprocess.run(
            [str(python), *args],
            cwd=str(state.repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail_calibration(state, f"{label} 失败：{exc}")
        return None


def _run_extrinsics_calibration(state: GatewayState, dataset: Path, run_name: str) -> None:
    python = _cv2_python(state.repo_root)
    if python is None:
        _fail_calibration(state, "找不到带 cv2 的解释器，无法标定")
        return

    # Solve against the intrinsics production is actually using, not the
    # metrology report: that report lives under outputs/, which is excluded from
    # the deploy sync and so is simply absent on the rig. Using the active run
    # also guarantees the bundle and the shipped intrinsics agree.
    calib_root = state.repo_root / "outputs" / "calibration"
    intrinsics_run = calib_root / (state.calibration.intrinsicsRun or "")
    intrinsics_source: list[str]
    if state.calibration.intrinsicsRun and intrinsics_run.is_dir():
        intrinsics_source = ["--intrinsics-run", str(intrinsics_run)]
    elif (state.repo_root / _CALIB_INTRINSICS_REPORT).is_file():
        intrinsics_source = ["--intrinsics-report", str(state.repo_root / _CALIB_INTRINSICS_REPORT)]
    else:
        _fail_calibration(state, (
            "找不到内参：既没有已激活的内参 run（calibration.intrinsics_run_name），"
            f"也没有 {_CALIB_INTRINSICS_REPORT}"
        ))
        return

    work = state.repo_root / "outputs" / "metrology" / run_name
    detections = work / "det_extr"
    base_run = calib_root / (state.calibration.extrinsicsRun or "")

    steps: list[tuple[str, list[str], int]] = [
        (
            "检测 ChArUco 角点…",
            [
                "-m", "metrology.cli.detect_charuco",
                "--episodes", str(dataset / "episodes"),
                "--out", str(detections),
                "--stride", "2",
            ],
            3600,
        ),
        (
            "多相机联合 BA…",
            [
                "-m", "metrology.cli.calibrate_extrinsics",
                "--detections", str(detections),
                *intrinsics_source,
                "--out", str(work),
            ],
            3600,
        ),
    ]
    for label, args, timeout in steps:
        proc = _calibration_step(state, python, args, label=label, timeout=timeout)
        if proc is None:
            return
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            _fail_calibration(state, f"{label} 失败：{detail[-1] if detail else proc.returncode}")
            return

    export_args = [
        "-m", "metrology.cli.export_production_calibration",
        "--extrinsics-report", str(work / "extrinsics_report.json"),
        "--name", run_name,
    ]
    # Only emit intrinsics when there is no production run to keep. Re-solving
    # extrinsics does not touch the lenses.
    keep_intrinsics_run = bool(state.calibration.intrinsicsRun) and intrinsics_run.is_dir()
    if not keep_intrinsics_run:
        export_args += [
            "--intrinsics-report", str(state.repo_root / _CALIB_INTRINSICS_REPORT),
            "--model", "fisheye",
        ]
    if base_run.is_dir():
        export_args += ["--base-extrinsics", str(base_run)]
        unmoved = _unmoved_cameras(state)
        if unmoved:
            export_args += ["--align-cameras", *unmoved]
            state.log("info", f"Base-frame alignment restricted to unmoved cameras: {', '.join(unmoved)}")
    serial_map = state.repo_root / "tools" / "thor" / "gmsl2" / "camera_serial_map.yaml"
    if serial_map.is_file():
        export_args += ["--serial-map", str(serial_map)]

    proc = _calibration_step(state, python, export_args, label="导出生产标定…", timeout=600)
    if proc is None:
        return
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        _fail_calibration(state, f"导出失败：{detail[-1] if detail else proc.returncode}")
        return

    report_path = work / "extrinsics_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail_calibration(state, f"读取 BA 结果失败：{exc}")
        return

    per_camera = report.get("per_camera_rmse") or {}
    cameras = [
        {
            "id": name,
            # Real reprojection residual in pixels, not a stand-in. The old mock
            # reported millimetres it had invented.
            "reprojectionPx": round(float(rmse), 4),
            "status": "pass" if rmse < 0.5 else ("warn" if rmse < 1.0 else "fail"),
        }
        for name, rmse in sorted(per_camera.items())
    ]
    failed = [c for c in cameras if c["status"] == "fail"]
    state.calibration.cameras = cameras
    state.calibration.state = "failed" if failed else "complete"
    state.calibration.lastRunAt = _now_iso()
    state.calibration.outputPath = str(work)
    if not keep_intrinsics_run:
        state.calibration.intrinsicsRun = f"{run_name}_intrinsics"
    state.calibration.extrinsicsRun = f"{run_name}_extrinsics"
    state.calibration.message = (
        f"BA 重投影 {float(report.get('rmse_px', float('nan'))):.4f} px，"
        f"{report.get('num_frames', 0)} 帧 / {len(cameras)} 相机"
        + ("；有相机残差过大" if failed else "")
    )
    state.log(
        "warn" if failed else "info",
        f"Extrinsics calibration {state.calibration.state}: {state.calibration.message}",
    )
    if state.calibration_session.active:
        state.calibration_session.stage = "failed" if failed else "done"
        state.calibration_session.message = state.calibration.message
    # The baseline belongs to the calibration it was taken against, so a new
    # calibration invalidates it rather than silently keeping the old frames.
    baseline_meta = _rig_check_root(state) / "baseline" / "baseline.json"
    if baseline_meta.is_file():
        baseline_meta.unlink()
        state.log("info", "Rig-check baseline cleared; capture a new one for the new calibration")


def _start_extrinsics_calibration(state: GatewayState, dataset_arg: str = "") -> dict[str, Any]:
    if state.calibration.state == "running":
        return {"ok": False, "error": "标定已在进行中"}
    session = state.calibration_session
    if dataset_arg:
        dataset = _resolve_dataset_root(state.repo_root, dataset_arg)
    elif session.active and session.stage == "ready" and session.datasetRoot:
        # Prefer what the guided session just recorded over whatever happens to
        # be the newest calibration-looking directory on disk.
        dataset = _resolve_dataset_root(state.repo_root, session.datasetRoot)
    else:
        dataset = _newest_calibration_dataset(state)
    if dataset is None or not (dataset / "episodes").is_dir():
        return {
            "ok": False,
            "error": "找不到可用的 ChArUco 采集",
            "hint": "先用采集页录一段挥板数据（数据集名含 calib），再运行外参标定。",
        }

    run_name = f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
    state.calibration.state = "running"
    state.calibration.message = f"处理 {dataset.name}…"
    state.calibration.cameras = []
    if state.calibration_session.active:
        state.calibration_session.stage = "solving"
        state.calibration_session.message = f"正在解算 {dataset.name}…" 
    Thread(
        target=_run_extrinsics_calibration,
        args=(state, dataset, run_name),
        daemon=True,
        name=f"extrinsics-calibration-{run_name}",
    ).start()
    return {"ok": True, "dataset": str(dataset), "run": run_name}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _parse_iso_epoch_s(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if text.endswith("Z"):
            return time.mktime(time.strptime(text[:-1], "%Y-%m-%dT%H:%M:%S")) - time.timezone
        return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))
    except (OverflowError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Task management (local JSON store)
# ---------------------------------------------------------------------------

def _tasks_store_path(state: GatewayState) -> Path:
    return state.repo_root / "outputs" / "tasks.json"


def _read_tasks(state: GatewayState) -> list[dict[str, Any]]:
    path = _tasks_store_path(state)
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(loaded, dict):
        tasks = loaded.get("tasks", [])
        return tasks if isinstance(tasks, list) else []
    return []


def _write_tasks(state: GatewayState, tasks: list[dict[str, Any]]) -> None:
    path = _tasks_store_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "tasks": tasks}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _normalize_task(payload: dict[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    status = str(payload.get("status") or "pending")
    if status not in {"pending", "in_progress", "completed", "paused"}:
        status = "pending"
    raw_tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    tags = [str(t).strip() for t in raw_tags if str(t).strip()][:12]
    now = _now_iso()
    return {
        "id": task_id or str(payload.get("id") or f"task-{time.time_ns()}"),
        "name": str(payload.get("name") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "targetEpisodes": max(0, int(payload.get("targetEpisodes", 0))),
        "completedEpisodes": max(0, int(payload.get("completedEpisodes", 0))),
        "status": status,
        "assignee": str(payload.get("assignee") or "").strip(),
        "datasetRepoId": str(payload.get("datasetRepoId") or "").strip(),
        "tags": tags,
        "createdAt": str(payload.get("createdAt") or now),
        "updatedAt": now,
    }


def _dataset_episode_count(dataset_root: Path) -> int:
    if _has_gmsl2_episodes(dataset_root):
        episodes, _ = _gmsl2_dataset_stats(dataset_root)
        return episodes
    return int(_load_dataset_info(dataset_root).get("total_episodes") or 0)


def _count_completed_episodes(state: GatewayState, repo_id: str) -> int:
    if not repo_id:
        return 0
    # datasetRepoId carries a namespace ("local/pick_and_place") but on-disk
    # dataset directories use only the trailing name, optionally suffixed with a
    # capture timestamp ("pick_and_place_20260528_103422"). Match on the base
    # name against each candidate's name prefixes so both forms count.
    base_name = repo_id.split("/")[-1].strip()
    if not base_name:
        return 0
    # Read from the cached dataset scan (maintained off-snapshot by the
    # background refresher) instead of walking every dataset on disk. This is
    # called for EACH task inside _snapshot on every ~1s poll; the old
    # iterdir()+glob was O(tasks x datasets) FS ops (9 x 253 here) and made
    # /api/snapshot take 1-5s -- the gateway-slow / preview-demand-starved root.
    items = state.cached_recorded_datasets if state.dataset_cache_ready else _recorded_dataset_items(state)
    total = 0
    for item in items:
        if str(item.get("datasetKind") or "recorded") != "recorded":
            continue
        name = str(item.get("name") or "")
        if base_name in _dataset_name_prefixes(name):
            total += int(item.get("totalEpisodes") or 0)
    return total


def _tasks_with_progress(state: GatewayState) -> list[dict[str, Any]]:
    tasks = _read_tasks(state)
    for task in tasks:
        if task.get("datasetRepoId"):
            task["completedEpisodes"] = _count_completed_episodes(state, task["datasetRepoId"])
    return tasks


def _create_task(state: GatewayState, payload: dict[str, Any]) -> dict[str, Any]:
    tasks = _read_tasks(state)
    task = _normalize_task(payload)
    if not task["name"]:
        raise ValueError("Task name is required.")
    tasks.append(task)
    _write_tasks(state, tasks)
    state.log("info", f"Created task: {task['name']}")
    return task


def _update_task(state: GatewayState, payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("id") or "")
    if not task_id:
        raise ValueError("Task id is required.")
    tasks = _read_tasks(state)
    for i, existing in enumerate(tasks):
        if existing.get("id") == task_id:
            merged = {**existing, **payload}
            tasks[i] = _normalize_task(merged, task_id=task_id)
            tasks[i]["createdAt"] = existing.get("createdAt", tasks[i]["createdAt"])
            _write_tasks(state, tasks)
            state.log("info", f"Updated task: {tasks[i]['name']}")
            return tasks[i]
    raise ValueError(f"Task not found: {task_id}")


def _delete_task(state: GatewayState, task_id: str) -> None:
    tasks = _read_tasks(state)
    filtered = [t for t in tasks if t.get("id") != task_id]
    if len(filtered) == len(tasks):
        raise ValueError(f"Task not found: {task_id}")
    _write_tasks(state, filtered)
    if state.active_task_id == task_id:
        state.active_task_id = None
    state.log("info", f"Deleted task: {task_id}")


_ACTIVE_TASK_OVERLAY_NAME = ".active_task_config.yaml"
# Teleop gets its own overlay file rather than sharing the recorder's: the two are resolved
# independently and a teleop-only overlay landing on the recorder's path would hand the recorder a
# config it never asked for.
_TELEOP_OVERLAY_NAME = ".teleop_gains_config.yaml"


def _find_task(state: GatewayState, task_id: str | None) -> dict[str, Any] | None:
    if not task_id:
        return None
    for task in _read_tasks(state):
        if task.get("id") == task_id:
            return task
    return None


def _recorder_is_running(state: GatewayState) -> bool:
    return state.process is not None and state.process.poll() is None


def _set_active_task(state: GatewayState, task_id: str) -> dict[str, Any] | None:
    """Bind (or clear) the task that the next Connect records into.

    Passing an empty id clears the binding (records into the YAML default).
    The binding is consumed at Connect time when the overlay config is built,
    so switching while a recorder process is live is rejected: dataset_root is
    fixed at recorder spawn and cannot change mid-session.
    """

    task_id = (task_id or "").strip()
    if not task_id:
        if _recorder_is_running(state) and state.active_task_id is not None:
            raise ValueError("Disconnect the recorder before unbinding the active task.")
        state.active_task_id = None
        state.log("info", "Cleared active recording task")
        return None
    task = _find_task(state, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")
    if not str(task.get("datasetRepoId") or "").strip():
        raise ValueError("Task has no dataset repo id; set one before recording into it.")
    if _recorder_is_running(state) and state.active_task_id not in (None, task_id):
        raise ValueError("Disconnect the recorder before binding a different task.")
    state.active_task_id = task_id
    state.log("info", f"Recording bound to task: {task['name']} ({task['datasetRepoId']})")
    return task


def _task_datasets_dir(state: GatewayState) -> Path | None:
    """Directory under which task datasets live and are counted from."""
    if state.datasets_root is not None:
        return state.datasets_root
    base = _resolve_dataset_root(state.repo_root, _dataset_config(state.config).get("root"))
    return base.parent if base is not None else None


def _build_task_overlay_config(
    base_config: dict[str, Any], task: dict[str, Any], datasets_dir: Path
) -> dict[str, Any]:
    """Deep-copy ``base_config`` and patch only ``dataset.*`` so the recorder
    writes into the task's dataset.

    The on-disk ``root`` basename is kept equal to the ``repo_id`` trailing
    segment because ``_count_completed_episodes`` attributes episodes to a task
    by matching that name against the (timestamp-suffixed) directory names.
    """

    repo_id = str(task.get("datasetRepoId") or "").strip()
    if not repo_id:
        raise ValueError("Task has no datasetRepoId.")
    name = repo_id.split("/")[-1].strip()
    if not name:
        raise ValueError(f"Task datasetRepoId has no name segment: {repo_id!r}")
    overlay = copy.deepcopy(base_config)
    dataset = overlay.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
        overlay["dataset"] = dataset
    dataset["repo_id"] = repo_id
    dataset["root"] = str(datasets_dir / name)
    prompt = str(task.get("description") or task.get("name") or "").strip()
    if prompt:
        dataset["single_task"] = prompt
    return overlay


def _apply_teleop_gain_overrides(config: dict[str, Any], overrides: dict[str, float | None]) -> None:
    """Write UI gain overrides into a config mapping in place."""

    if not overrides:
        return
    teleop = config.get("teleop")
    if not isinstance(teleop, dict):
        teleop = {}
        config["teleop"] = teleop
    for field_name, value in overrides.items():
        teleop[field_name] = value


def _write_overlay_config(
    state: GatewayState, overlay: dict[str, Any], *, name: str = _ACTIVE_TASK_OVERLAY_NAME
) -> Path:
    import yaml

    path = state.repo_root / "outputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as overlay_file:
        yaml.safe_dump(overlay, overlay_file, sort_keys=False, allow_unicode=True)
    return path


def _parse_episode_time_override(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        episode_time_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid episode duration: {value!r}") from exc
    if not math.isfinite(episode_time_s) or episode_time_s < 1.0 or episode_time_s > 600.0:
        raise ValueError("Episode duration must be between 1 and 600 seconds.")
    return episode_time_s


def _parse_recording_fps_override(value: str | float | int | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid recording FPS: {value!r}") from exc
    if not math.isfinite(fps) or fps < 1 or fps > 120 or abs(fps - round(fps)) > 1e-9:
        raise ValueError("Recording FPS must be an integer between 1 and 120.")
    return int(round(fps))


def _clear_runtime_recording_config(state: GatewayState) -> None:
    state.runtime_recording_config = None
    state.runtime_recording_config_path = None
    dataset_config = _dataset_config(state.config)
    state.recording.datasetRoot = str(dataset_config.get("root") or state.recording.datasetRoot)
    state.recording.repoId = str(dataset_config.get("repo_id") or state.recording.repoId)
    state.recording.targetFrames = _target_frames(state.config)


def _resolve_recorder_config_path(
    state: GatewayState, *, episode_time_s: float | None = None, recording_fps: int | None = None
) -> Path:
    """Config path to spawn the recorder with.

    Returns an overlay config when a task binding or per-session UI duration is active, else the
    repo's literal config file. Sets ``state.recording`` and runtime config fields to whichever
    dataset config the recorder will actually use.
    """

    active_task = _find_task(state, state.active_task_id)
    has_task = active_task is not None and bool(str(active_task.get("datasetRepoId") or "").strip())
    # Gains tuned on the Teleoperation page have to reach the recorder too: the operator drives the
    # same SpaceMouse through the same teleoperator while recording, so a session tuned at one gain
    # and recorded at another would put demonstrations in the dataset that no live teleop produced.
    gain_overrides = dict(state.runtime_teleop_gains or {})
    overlay: dict[str, Any] | None = None
    if has_task:
        datasets_dir = _task_datasets_dir(state)
        if datasets_dir is None:
            raise RuntimeError(
                "Cannot record into a task without a datasets root; start the gateway "
                "with --datasets-root or set dataset.root in the config."
            )
        overlay = _build_task_overlay_config(state.config, active_task, datasets_dir)
    else:
        state.active_task_id = None
        if episode_time_s is not None or recording_fps is not None or gain_overrides:
            overlay = copy.deepcopy(state.config)

    if episode_time_s is not None or recording_fps is not None:
        if overlay is None:
            overlay = copy.deepcopy(state.config)
        dataset = overlay.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {}
            overlay["dataset"] = dataset
        if episode_time_s is not None:
            dataset["episode_time_s"] = float(episode_time_s)
        if recording_fps is not None:
            dataset["fps"] = int(recording_fps)

    if gain_overrides:
        if overlay is None:
            overlay = copy.deepcopy(state.config)
        _apply_teleop_gain_overrides(overlay, gain_overrides)

    config = overlay if overlay is not None else state.config
    config_path = _write_overlay_config(state, overlay) if overlay is not None else state.config_path
    dataset_config = _dataset_config(config)
    state.runtime_recording_config = config
    state.runtime_recording_config_path = config_path
    state.recording.datasetRoot = str(dataset_config.get("root") or "")
    state.recording.repoId = str(dataset_config.get("repo_id") or state.recording.repoId)
    state.recording.targetFrames = _target_frames(config)

    if has_task:
        state.log(
            "info",
            f"Recording into task '{active_task['name']}' dataset {active_task['datasetRepoId']} "
            f"(config {config_path})",
        )
    if episode_time_s is not None:
        state.log("info", f"Recording episode duration set to {episode_time_s:g}s for this session")
    if recording_fps is not None:
        state.log("info", f"Recording FPS set to {recording_fps:g} for this session")
    if gain_overrides:
        state.log(
            "info",
            "Recording with SpaceMouse gain overrides "
            + ", ".join(f"{name}={value:g}" for name, value in sorted(gain_overrides.items())),
        )
    return config_path


# ----------------------------------------------------- task v3 consolidation ---


def _export_is_running(state: GatewayState) -> bool:
    return state.export_process is not None and state.export_process.poll() is None


def _task_exports_root(state: GatewayState) -> Path:
    if state.exports_root is not None:
        return state.exports_root
    return state.repo_root / "outputs" / "exports"


def _export_command(state: GatewayState, task: dict[str, Any]) -> tuple[list[str], Path]:
    """Build the export_v3 subprocess command and return (cmd, out_root).

    Consolidates every session whose name shares the task's repo_id trailing
    segment into one LeRobot v3 dataset under the exports root.
    """

    repo_id = str(task.get("datasetRepoId") or "").strip()
    if not repo_id:
        raise ValueError("Task has no dataset repo id; nothing to export.")
    base_name = repo_id.split("/")[-1].strip()
    datasets_dir = _task_datasets_dir(state)
    if datasets_dir is None:
        raise RuntimeError(
            "Cannot export without a datasets root; start the gateway with --datasets-root."
        )
    exports_root = _task_exports_root(state)
    out_root = exports_root / base_name
    task_prompt = str(task.get("description") or task.get("name") or base_name).strip()
    script = state.repo_root / "tools" / "thor" / "gmsl2" / "export_v3.py"
    command = [
        str(_venv_python(state.repo_root)),
        str(script),
        "--datasets-root", str(datasets_dir),
        "--exports-root", str(exports_root),
        "--base-name", base_name,
        "--repo-id", repo_id,
        "--task", task_prompt,
        "--overwrite",
    ]
    return command, out_root


def _matching_task_for_dataset(state: GatewayState, dataset_root: Path) -> dict[str, Any] | None:
    prefixes = _dataset_name_prefixes(dataset_root.name)
    for task in _read_tasks(state):
        repo_id = str(task.get("datasetRepoId") or "").strip()
        base_name = repo_id.split("/")[-1].strip()
        if base_name and base_name in prefixes:
            return task
    return None


def _approved_dataset_export_command(state: GatewayState, dataset_root: Path) -> tuple[list[str], Path]:
    """Build an export_v3 command scoped to one approved raw GMSL2 session."""

    if not _has_gmsl2_episodes(dataset_root):
        raise ValueError("Approved raw export requires a GMSL2 session dataset.")
    base_name = dataset_root.name
    output_name = _dataset_name_with_actual_camera_count(dataset_root)
    exports_root = _task_exports_root(state)
    out_root = exports_root / output_name
    task = _matching_task_for_dataset(state, dataset_root)
    task_prompt = str((task or {}).get("description") or (task or {}).get("name") or output_name).strip()
    namespace = str((task or {}).get("datasetRepoId") or "local").split("/")[0] or "local"
    repo_id = f"{namespace}/{output_name}"
    script = state.repo_root / "tools" / "thor" / "gmsl2" / "export_v3.py"
    command = [
        str(_venv_python(state.repo_root)),
        str(script),
        "--datasets-root", str(dataset_root.parent),
        "--exports-root", str(exports_root),
        "--base-name", base_name,
        "--output-name", output_name,
        "--repo-id", repo_id,
        "--task", task_prompt,
        "--overwrite",
    ]
    return command, out_root


def _start_task_export(state: GatewayState, task_id: str) -> None:
    if _export_is_running(state):
        raise RuntimeError("An export is already running; wait for it to finish.")
    task = _find_task(state, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")
    command, out_root = _export_command(state, task)
    state.export_process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_recorder_env(state.repo_root),
        start_new_session=True,
    )
    state.dataset_export = DatasetExportStatus(
        state="exporting",
        target="lerobot_v3",
        datasetRoot=str(_task_datasets_dir(state) or ""),
        outputPath=str(out_root),
        message=f"Consolidating sessions for {task['name']}…",
        pid=state.export_process.pid,
        taskId=task_id,
    )
    state.log("info", f"Started v3 export for task {task['name']} -> {out_root}")
    Thread(
        target=_read_export_output,
        args=(state, state.export_process),
        daemon=True,
        name=f"task-export-output-{state.export_process.pid}",
    ).start()


def _copy_approved_v3_dataset_export(
    state: GatewayState, dataset_root: Path, processing_item: dict[str, Any]
) -> None:
    exports_root = _task_exports_root(state)
    out_root = exports_root / dataset_root.name
    try:
        if dataset_root.resolve() == out_root.resolve():
            raise ValueError("Export output path is the same as the source dataset path.")
    except OSError as exc:
        raise ValueError(f"Cannot resolve export paths: {exc}") from exc
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_root, out_root)
    state.dataset_export = DatasetExportStatus(
        state="complete",
        target="lerobot_v3",
        datasetRoot=str(dataset_root),
        outputPath=str(out_root),
        selectedEpisodes=int(processing_item.get("totalEpisodes") or 0),
        totalFrames=int(processing_item.get("totalFrames") or 0),
        message=f"Export complete: approved dataset copied to {out_root}",
    )
    state.log("info", f"Exported approved LeRobot v3 dataset {dataset_root} -> {out_root}")


def _dataset_task_base_name(name: str) -> str:
    """The task a recording belongs to, from its directory name.

    ``pick_and_place_20260819_171756`` -> ``pick_and_place``. Sessions of one task differ only
    by their capture timestamp, so the base name is what several recordings of the same task
    have in common -- and therefore what the view built from all of them should be called.
    """
    prefixes = _dataset_name_prefixes(name) - {name}
    return next(iter(prefixes), name)


def _training_view_name(dataset_roots: Sequence[Path], action_mode: str) -> str:
    """The directory a build writes to, derived only from what went into it.

    Deterministic on purpose: rebuilding the same selection lands on the same name and replaces
    it, which is what "this task gained a session, rebuild the view" should do. A different
    selection produces a different name instead of silently overwriting a view that described
    other frames.

    Restricted to ``[A-Za-z0-9._-]`` because the name is not only a directory. It becomes the
    training job name, which `_start_training_run` refuses outside that set, and the trailing
    half of the ``local/<name>`` repo id, which `huggingface_hub.validate_repo_id` refuses on
    the same grounds. A name that only works as a directory is a view that builds and then
    cannot be trained.
    """
    bases: list[str] = []
    for root in dataset_roots:
        base = _dataset_task_base_name(root.name)
        if base not in bases:
            bases.append(base)
    if len(dataset_roots) == 1:
        # One source keeps its own full name, timestamp included: there is nothing to merge, and
        # collapsing it to the task name would make a single-session view claim the whole task.
        stem = dataset_roots[0].name
    else:
        stem = "-".join(sorted(bases))
    return _safe_training_view_name(f"{stem}__{action_mode}")


_UNSAFE_VIEW_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_training_view_name(name: str) -> str:
    """Fold anything a task name may carry into the character set a job name accepts."""
    cleaned = _UNSAFE_VIEW_NAME_CHARS.sub("-", name).strip("-.")
    return cleaned or "training_view"


def _training_view_camera_keys(dataset_roots: Sequence[Path]) -> list[str]:
    """Camera features every selected recording has.

    The intersection rather than the first dataset's set: a camera missing from one source
    would fail deep inside the builder, after it had already started writing the view.
    """
    shared: set[str] | None = None
    for root in dataset_roots:
        info = _load_dataset_info(root) or {}
        features = info.get("features") if isinstance(info.get("features"), dict) else {}
        keys = {
            key
            for key, feature in features.items()
            if key.startswith("observation.images.")
            and isinstance(feature, dict)
            and feature.get("dtype") in ("video", "image")
        }
        shared = keys if shared is None else (shared & keys)
    return sorted(shared or set())


def _training_view_command(
    state: GatewayState,
    dataset_roots: Path | Sequence[Path],
    action_mode: str,
    *,
    camera_crops: dict[str, list[int]] | None = None,
    view_fps: int = DEFAULT_TRAINING_VIEW_FPS,
) -> tuple[list[str], Path]:
    """Build the prepare-only training-view command for an FR3 workstation dataset.

    The workstation recorder already writes a LeRobot v3 dataset, so there is no raw->v3 export
    to run (that is the Thor GMSL2 path). What a workstation operator needs instead is the
    *training view*: the same episodes with the action column expressed in whichever contract the
    policy will be trained on. Delta contracts are derived here, by differencing consecutive
    dataset frames, because a delta computed during capture would span one control tick rather
    than one frame.
    """
    if action_mode not in TRAINING_VIEW_ACTION_MODES:
        raise ValueError(
            f"action_mode must be one of {TRAINING_VIEW_ACTION_MODES}, got {action_mode!r}"
        )
    # A lone Path is the single-source call this function grew out of. Normalised rather than
    # rejected, and checked explicitly because `list()` of a Path raises while `list()` of a str
    # would silently produce one entry per character.
    dataset_roots = [dataset_roots] if isinstance(dataset_roots, (str, Path)) else list(dataset_roots)
    dataset_roots = [Path(root) for root in dataset_roots]
    if not dataset_roots:
        raise ValueError("Select at least one recording to build a training view from.")
    for dataset_root in dataset_roots:
        if not _has_lerobot_v3_data(dataset_root):
            raise ValueError(
                f"{dataset_root.name} is not a LeRobot v3 dataset; nothing to build a view from."
            )
    # Cameras and state keys come from the dataset, not from the script's defaults: those
    # defaults name another rig's cameras (observation.images.cam_1/cam_3) and would fail on
    # every FR3 recording, which uses the config's own camera keys (ee/side).
    camera_keys = _training_view_camera_keys(dataset_roots)
    if not camera_keys:
        names = ", ".join(root.name for root in dataset_roots)
        raise ValueError(
            f"No camera feature is shared by every selected recording ({names}); "
            "a view cannot be built from sources that disagree on their cameras."
        )

    view_name = _training_view_name(dataset_roots, action_mode)
    view_root = _training_views_root(state) / view_name
    command = [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_train_il_policy.py"),
        # An explicit list, never a parent directory: the builder would expand a directory into
        # every dataset inside it, and the view would then hold recordings nobody ticked.
        "--dataset-roots", *[str(root) for root in dataset_roots],
        "--view-root", str(view_root),
        # The job name is what the generated train/inference configs use for their training
        # output dir and checkpoint path. Left to the script's default it is a fixed legacy
        # name, so every view built here would train into -- and overwrite -- the same
        # directory regardless of source dataset or action contract.
        "--job-name", view_name,
        "--repo-id", f"local/{view_name}",
        "--cameras", ",".join(camera_keys),
        "--state-keys", "observation.state",
        "--action-mode", action_mode,
        # The default append selector pulls a handheld-gripper column that FR3 datasets do not
        # have; the FR3 action already carries its own gripper.
        "--action-append-selectors", "",
        "--action-append-names", "",
        # Explicit even though it is the script's default: this command is what the event log
        # records, and "the reviewer's exclusions were applied" has to be visible there.
        "--respect-annotations",
        # The rate the view is resampled to. Explicit rather than left to the script's default
        # because it is the one export setting that silently changes what the action column
        # *means*: the action is a per-frame delta, so a view built at 30 fps from 60 fps frames
        # has twice the per-frame displacement of the recording it came from. Two views built at
        # different rates cannot be merged, and nothing downstream would notice.
        "--view-fps", str(int(view_fps)),
        "--overwrite-view",
        # Build the view only; training is a separate, deliberate step.
        "--prepare-only",
    ]
    if camera_crops:
        command.extend(["--camera-crops", json.dumps(camera_crops, separators=(",", ":"))])
    return command, view_root


def _training_views_root(state: GatewayState) -> Path:
    return _task_exports_root(state) / TRAINING_VIEWS_DIR_NAME


def _training_view_fps_conflict(dataset_roots: Sequence[Path], view_fps: int) -> str | None:
    """Why this rate cannot express these recordings, or None if it can.

    Run before the build rather than left to the builder, which raises the same conditions but
    only after the operator has watched a merge start. Only integer decimation is possible: the
    action column is a per-frame delta, so keeping the nearest frame at a non-divisor rate would
    swing every delta between one and two source intervals.
    """
    rates: dict[int, list[str]] = {}
    for root in dataset_roots:
        fps = int((_load_dataset_info(root) or {}).get("fps") or 0)
        if fps <= 0:
            return f"{root.name} has no usable fps in meta/info.json."
        rates.setdefault(fps, []).append(root.name)
    if view_fps <= 0:
        if len(rates) > 1:
            listed = "; ".join(f"{fps} fps: {', '.join(names)}" for fps, names in sorted(rates.items()))
            return (
                f"The selected recordings disagree on their rate ({listed}). Pick a view rate "
                "they can all be decimated to instead of keeping the source rate."
            )
        return None
    for fps, names in sorted(rates.items()):
        if fps < view_fps:
            return (
                f"{', '.join(names)} is {fps} fps, below the requested {view_fps} fps. "
                "Upsampling would invent frames; pick a lower view rate."
            )
        if fps % view_fps != 0:
            divisors = ", ".join(str(fps // n) for n in range(1, 5) if fps % n == 0)
            return (
                f"{', '.join(names)} is {fps} fps, which {view_fps} fps does not divide. "
                f"Pick a divisor of the source rate (for example {divisors})."
            )
    return None


def _start_training_view(
    state: GatewayState,
    raw_paths: str | Sequence[str],
    action_mode: str,
    *,
    acknowledge_warnings: bool = False,
    camera_crops: dict[str, list[int]] | None = None,
    view_fps: int = DEFAULT_TRAINING_VIEW_FPS,
) -> None:
    """Workstation counterpart of the Thor v3 export: build a policy-ready training view.

    Several recordings can go into one view because that is the only moment they can be
    combined: the view renumbers its episodes and computes meta/stats.json over the whole set,
    so two views built separately share neither an episode index space nor a normalisation, and
    adding a session later means rebuilding from every source at once.
    """
    if _export_is_running(state):
        raise RuntimeError("A view build is already running; wait for it to finish.")
    # A single string still works, and is spelled out because iterating one as a sequence would
    # hand each character to the resolver and report "dataset not found: /" .
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [str(raw_paths)]
    dataset_roots: list[Path] = []
    for raw_path in raw_paths:
        dataset_root = _resolve_known_dataset(state, raw_path)
        if dataset_root is None:
            raise ValueError(f"Dataset not found in the recorded dataset list: {raw_path}")
        # Views are replay candidates now, so they are resolvable here. Re-expressing an already
        # re-expressed action column would silently compose two contracts.
        if _dataset_kind(state, dataset_root) == "training_view":
            raise ValueError(
                f"{dataset_root.name} is already a training view; build from the recording instead."
            )
        if dataset_root not in dataset_roots:
            dataset_roots.append(dataset_root)
    if not dataset_roots:
        raise ValueError("Select at least one recording to build a training view from.")

    selected_episodes = 0
    pending_warnings: list[str] = []
    for dataset_root in dataset_roots:
        processing_item = _processing_item_from_dataset(dataset_root)
        # The same QC gate the Thor export enforces, and for a stronger reason: on this profile
        # the view *is* the export -- it is the last step before a policy trains on these frames,
        # and nothing downstream looks at QC again. The timestamp-sync verdict in particular only
        # exists inside a QC run, so an ungated build let a dataset whose modalities disagreed
        # reach training with its verdict sitting in a file no one had opened. Every source is
        # gated, not just the first: one unchecked recording in a merge is enough to poison the
        # whole training set, and it would be invisible once the episodes were renumbered.
        status = str(processing_item.get("status") or "")
        warnings = _qc_warning_messages(processing_item)
        if status == "qc_warn":
            # A warning is the operator's call to make, but only with the warnings in front of
            # them -- the same rule the replay gate uses for a failed MuJoCo score.
            pending_warnings.extend(f"{dataset_root.name}: {message}" for message in warnings)
            if not warnings:
                pending_warnings.append(f"{dataset_root.name}: see the QC summary for details.")
        elif status != "qc_pass":
            raise ValueError(
                f"{dataset_root.name} must pass QC before a training view is built "
                f"(status: {status or 'unknown'}). Run QC on the Dataset Processing page."
            )
        excluded = _annotation_excluded_episodes(dataset_root)
        total_episodes = int(processing_item.get("totalEpisodes") or 0)
        selected_episodes += max(0, total_episodes - len(excluded))

    if pending_warnings and not acknowledge_warnings:
        raise ValueError(
            "QC passed with warnings; confirm to build the view anyway. " + " | ".join(pending_warnings)
        )
    if pending_warnings:
        state.log(
            "warn",
            f"Building a training view over {len(pending_warnings)} QC warning(s): "
            + " | ".join(pending_warnings),
        )
    if selected_episodes <= 0:
        names = ", ".join(root.name for root in dataset_roots)
        raise ValueError(
            f"No episode is left to build a view from ({names}): every one is either absent or "
            "marked as not for training."
        )
    conflict = _training_view_fps_conflict(dataset_roots, view_fps)
    if conflict:
        raise ValueError(conflict)

    command, view_root = _training_view_command(
        state, dataset_roots, action_mode, camera_crops=camera_crops, view_fps=view_fps
    )
    source_label = (
        dataset_roots[0].name
        if len(dataset_roots) == 1
        else f"{len(dataset_roots)} recordings ({', '.join(root.name for root in dataset_roots)})"
    )
    state.export_process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_recorder_env(state.repo_root),
        start_new_session=True,
    )
    state.dataset_export = DatasetExportStatus(
        state="exporting",
        target=action_mode,
        datasetRoot=str(dataset_roots[0]),
        datasetRoots=[str(root) for root in dataset_roots],
        outputPath=str(view_root),
        selectedEpisodes=selected_episodes,
        totalFrames=0,
        message=f"Building {action_mode} training view from {source_label}…",
        pid=state.export_process.pid,
    )
    state.log(
        "info",
        f"Started {action_mode} training view build {source_label} -> {view_root} "
        f"({selected_episodes} episode(s) after exclusions)",
    )
    Thread(
        target=_read_export_output,
        args=(state, state.export_process),
        daemon=True,
        name=f"training-view-output-{state.export_process.pid}",
    ).start()


def _start_approved_dataset_export(
    state: GatewayState, raw_path: str, *, acknowledge_warnings: bool = False
) -> None:
    if _export_is_running(state):
        raise RuntimeError("An export is already running; wait for it to finish.")
    dataset_root = _resolve_known_dataset(state, raw_path)
    if dataset_root is None:
        raise ValueError("Dataset not found in the approved/candidate dataset list.")
    processing_item = _processing_item_from_dataset(dataset_root)
    status = str(processing_item.get("status") or "")
    warnings = _qc_warning_messages(processing_item)
    if status == "qc_warn":
        # A warning is a judgement call the operator is allowed to make, but only with the
        # warnings in front of them -- the same rule the replay gate uses for a failed MuJoCo
        # score. Silently blocking taught operators that Run QC breaks the export.
        if not acknowledge_warnings:
            raise ValueError(
                "QC passed with warnings; confirm to export anyway. "
                + (" | ".join(warnings) if warnings else "See the QC summary for details.")
            )
        state.log(
            "warn",
            f"Exporting {dataset_root.name} over {len(warnings)} QC warning(s): "
            + (" | ".join(warnings) if warnings else "unspecified"),
        )
    elif status != "qc_pass":
        raise ValueError(f"Dataset must pass QC before export (status: {status or 'unknown'}).")
    if not _has_gmsl2_episodes(dataset_root):
        if _has_lerobot_v3_data(dataset_root):
            _copy_approved_v3_dataset_export(state, dataset_root, processing_item)
            return
        raise ValueError("Approved dataset export supports LeRobot v3 datasets or raw GMSL2 session datasets.")
    command, out_root = _approved_dataset_export_command(state, dataset_root)
    state.export_process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_recorder_env(state.repo_root),
        start_new_session=True,
    )
    state.dataset_export = DatasetExportStatus(
        state="exporting",
        target="lerobot_v3",
        datasetRoot=str(dataset_root),
        outputPath=str(out_root),
        selectedEpisodes=int(processing_item.get("totalEpisodes") or 0),
        totalFrames=0,
        message=f"Exporting approved dataset {dataset_root.name}…",
        pid=state.export_process.pid,
    )
    state.log("info", f"Started approved dataset v3 export for {dataset_root} -> {out_root}")
    Thread(
        target=_read_export_output,
        args=(state, state.export_process),
        daemon=True,
        name=f"dataset-export-output-{state.export_process.pid}",
    ).start()


def _apply_export_output(state: GatewayState, output: str) -> None:
    state.dataset_export.message = output
    match = re.search(r"Export plan: (\d+) episodes", output)
    if match:
        state.dataset_export.selectedEpisodes = int(match.group(1))
    # The training-view builder announces where the view landed. Trust that over the path this
    # gateway predicted, so the UI links to what was actually written.
    view_match = re.match(r"\[prepare\] dataset view: (.+)$", output)
    if view_match:
        state.dataset_export.outputPath = view_match.group(1).strip()
    if output.startswith("Episode ") and "written" in output:
        frames = re.search(r"\((\d+) frames\)", output)
        if frames:
            state.dataset_export.totalFrames += int(frames.group(1))
    if output.startswith("Export complete"):
        state.dataset_export.state = "complete"
    elif output.startswith("ERROR:"):
        state.dataset_export.state = "error"
    state.log("info", f"export: {output}")


def _training_view_completion_message(state: GatewayState) -> str | None:
    """Summarize a finished training-view build from the view itself.

    Without this the status line keeps whatever the builder printed last, which in prepare-only
    mode is the training output dir -- a directory that does not exist yet, named after a job
    nobody on this page asked for. The manifest inside the view is the authoritative record of
    what was built.
    """
    if state.dataset_export.target not in TRAINING_VIEW_ACTION_MODES:
        return None
    contract = state.dataset_export.target
    manifest: dict[str, Any] = {}
    if state.dataset_export.outputPath:
        manifest = _load_json_file(Path(state.dataset_export.outputPath) / "meta" / "il_view_manifest.json")
    episodes = int(manifest.get("total_episodes") or state.dataset_export.selectedEpisodes or 0)
    frames = int(manifest.get("total_rows") or state.dataset_export.totalFrames or 0)
    state.dataset_export.selectedEpisodes = episodes
    state.dataset_export.totalFrames = frames
    # Named in the completion line, not only in the manifest: a view with fewer episodes than the
    # recording it came from has to say so where the operator is already looking.
    dropped = sorted(
        {
            int(episode)
            for excluded in (manifest.get("excluded_episodes") or {}).values()
            for episode in excluded
        }
    )
    excluded_note = f" · {len(dropped)} excluded by review {dropped}" if dropped else ""
    return f"View ready: {episodes} episode(s) · {frames} frames · {contract}{excluded_note}"


def _read_export_output(state: GatewayState, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output = line.strip()
        if not output:
            continue
        with state.lock:
            if state.export_process is not process:
                return
            _apply_export_output(state, output)
    return_code = process.wait()
    with state.lock:
        if state.export_process is process and state.dataset_export.state == "exporting":
            state.dataset_export.state = "error" if return_code else "complete"
            if return_code:
                state.dataset_export.message = f"Export exited with code {return_code}"
            else:
                summary = _training_view_completion_message(state)
                if summary is not None:
                    state.dataset_export.message = summary
                # A just-built view has to show up in the replay candidate list now, not up to
                # one memo TTL (or one 10s stats-refresh cycle) later, or the operator reads
                # "complete" while the dataset list still has no view to open.
                _invalidate_replay_candidates_memo()
                state.dataset_cache_ready = False


# ------------------------------------------------------------------ training runs ---


def _training_view_entries(state: GatewayState) -> list[dict[str, Any]]:
    """Training views this gateway has built, newest first, with what a run needs to pick one.

    Reads each view's own manifest rather than inferring from the directory name: the fps
    and action contract are what decide whether two views are interchangeable, and the
    name only carries the action mode.
    """
    root = _training_views_root(state)
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for view_root in sorted(root.iterdir(), key=lambda item: item.name):
        if not view_root.is_dir():
            continue
        info = _load_dataset_info(view_root) or {}
        manifest = _load_json_file(view_root / "meta" / "il_view_manifest.json") or {}
        if not info:
            continue
        cameras = [key for key in (info.get("features") or {}) if key.startswith("observation.images.")]
        entries.append(
            {
                "name": view_root.name,
                "root": str(view_root),
                "repoId": str(manifest.get("repo_id") or f"local/{view_root.name}"),
                "episodes": int(info.get("total_episodes") or 0),
                "frames": int(info.get("total_frames") or 0),
                "fps": int(info.get("fps") or 0),
                "actionMode": str(manifest.get("action_mode") or ""),
                "cameras": sorted(cameras),
                "sourceFps": manifest.get("source_fps") or {},
                "frameStride": manifest.get("frame_stride") or {},
                # What this view was built with, so the settings can be reused rather than
                # retyped. The manifest is the only record of them: the crop is baked into the
                # view's video and the page that drew it keeps nothing after a reload.
                "cameraCrops": manifest.get("camera_crop_specs") or {},
                "sourceRoots": [str(root) for root in (manifest.get("source_dataset_roots") or [])],
                "excludedEpisodes": {
                    str(root): list(episodes)
                    for root, episodes in (manifest.get("excluded_episodes") or {}).items()
                },
                "buildId": str(manifest.get("build_id") or ""),
                "sourceDigest": str(manifest.get("source_digest") or ""),
                "modifiedAt": datetime.fromtimestamp(view_root.stat().st_mtime, timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
    entries.sort(key=lambda item: item["modifiedAt"], reverse=True)
    return entries


def _training_is_running(state: GatewayState) -> bool:
    """Whether a run is on the GPU, whether or not this gateway started it.

    Asking only about `training_process` would answer "no" for a run re-adopted after a
    restart -- it has a pid and no Popen object -- and the caller uses this to decide whether
    starting another one is allowed. Two ACT runs on one 24 GB card is not a state either of
    them survives.
    """
    process = state.training_process
    if process is not None:
        return process.poll() is None
    pid = state.training.pid
    return (
        state.training.state in ("syncing", "starting", "running")
        and bool(pid)
        and _process_is_alive(int(pid))
    )


def _apply_training_output(state: GatewayState, line: str) -> None:
    status = state.training
    # Progress is read off every line, including the tqdm bars, because those carry the exact
    # step counter. Only the bars that say nothing else are then dropped from the visible tail:
    # they arrive about thirteen times a second, and keeping them would push every real log
    # line out of the window before an operator could read it.
    found = training_backend.parse_progress_line(line)
    if "step" in found:
        status.step = found["step"]
        # Only out of "starting". A stopped run keeps writing for as long as it takes to die,
        # and letting those lines put it back into "running" would lose the operator's stop and
        # then report the resulting non-zero exit as a failure rather than as what they asked for.
        if status.state in ("starting", "syncing"):
            status.state = "running"
    if "totalSteps" in found:
        status.totalSteps = found["totalSteps"]
    if "loss" in found:
        status.loss = found["loss"]
    if "wandbUrl" in found and not status.wandbUrl:
        status.wandbUrl = found["wandbUrl"]
    # Checkpointed periodically rather than per line: output arrives about thirteen times a
    # second, and the point of the file is that a gateway restarted mid-run knows roughly where
    # the run had got to -- not that it knows to the step.
    now = time.monotonic()
    if now - state.training_persisted_s >= TRAINING_STATUS_PERSIST_INTERVAL_S:
        state.training_persisted_s = now
        _persist_training_status(state)
    if training_backend.is_progress_bar_noise(line):
        return
    message = training_backend.strip_progress_prefix(line)
    status.lastLines = [*status.lastLines[-39:], message]
    status.message = message[:300]


# What lerobot_train prints once it has run every step it was asked for. Matched on the
# trainer's own words rather than on a checkpoint existing, because checkpoints are also
# written at every --save-freq: a run killed at step 15000 leaves one too.
_TRAINING_SUCCESS_MARKER = "End of training"


def _training_log_reports_success(log_path: Path) -> bool:
    """Whether the trainer said it finished, read from the tail of its log."""
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return False
    return _TRAINING_SUCCESS_MARKER in tail


def _follow_training_run(
    state: GatewayState,
    log_path: Path,
    pid: int,
    process: subprocess.Popen[str] | None = None,
    *,
    from_start: bool,
) -> None:
    """Follow a run by reading the log file it writes, and record how it ended.

    The run writes to that file itself; the gateway holds no pipe to it. That is the whole
    point. A pipe makes the gateway's lifetime the run's lifetime -- kill the gateway and the
    read end closes, so the next line the trainer writes takes SIGPIPE and the job dies. Which
    is precisely what a deploy did, six thousand steps into a twenty-thousand step run.

    Reading the file instead lets the gateway be restarted, or crash, while the run continues,
    and makes re-attaching after a restart the same code path as following a run this process
    started -- the only difference being whether there is a `process` to ask for an exit code.
    """
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            if not from_start:
                # A re-attached run's backlog is already summarized in the persisted status;
                # replaying thousands of lines would only rewrite it with the same numbers.
                handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    # tqdm redraws with carriage returns, so one "line" off the file can hold
                    # many bar updates. Split them out, or the regexes match the first and the
                    # displayed step lags by however many updates shared the line.
                    for part in line.replace("\r", "\n").splitlines():
                        output = part.rstrip()
                        if not output:
                            continue
                        with state.lock:
                            if state.training.pid != pid:
                                return
                            _apply_training_output(state, output)
                    continue
                alive = process.poll() is None if process is not None else _process_is_alive(pid)
                if not alive:
                    # One more pass: the run may have written its last lines between the read
                    # above and the exit observed here.
                    remaining = handle.read()
                    for part in remaining.replace("\r", "\n").splitlines():
                        output = part.rstrip()
                        if not output:
                            continue
                        with state.lock:
                            if state.training.pid != pid:
                                return
                            _apply_training_output(state, output)
                    break
                time.sleep(1.0)
    except OSError as exc:
        with state.lock:
            if state.training.pid == pid:
                state.training.message = f"Could not follow training log: {exc}"

    return_code = process.wait() if process is not None else None
    with state.lock:
        if state.training.pid != pid:
            return
        state.training.finishedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if state.training.state == "stopped":
            pass
        elif return_code is None:
            # No exit code: this gateway inherited the run rather than starting it. Ask the
            # trainer instead -- lerobot_train prints a line of its own when it finishes all
            # its steps. That is evidence, not a guess; without it a run that completed
            # perfectly gets reported as a failure purely because the gateway was restarted
            # somewhere in the middle of it.
            if _training_log_reports_success(log_path):
                state.training.state = "complete"
                state.training.message = (
                    f"Training finished; checkpoints under {state.training.outputDir}"
                )
            else:
                state.training.state = "error"
                state.training.message = (
                    "Training process ended while this gateway was not attached to it, and its "
                    f"log does not report finishing; check {state.training.logPath}."
                )
        elif return_code == 0:
            state.training.state = "complete"
            state.training.message = f"Training finished; checkpoints under {state.training.outputDir}"
        else:
            state.training.state = "error"
            state.training.message = f"Training exited with code {return_code}"
        _persist_training_status(state)
        state.log(
            "info" if state.training.state == "complete" else "error",
            f"Training run {state.training.jobName} finished"
            + (
                f" with code {return_code}"
                if return_code is not None
                else f" (no exit code; log says {state.training.state})"
            ),
        )


def _training_run_state_path(state: GatewayState) -> Path:
    return state.repo_root / "outputs" / "logs" / "training" / "current_run.json"


def _persist_training_status(state: GatewayState) -> None:
    """Record the run on disk so it outlives this gateway process.

    Training runs for hours; the gateway is restarted by every deploy and does not survive a
    crash. Both already leave the run itself alive -- it is started in its own session -- so
    without this the GPU stays busy with a job the page can no longer show, report or stop.
    """
    path = _training_run_state_path(state)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temp file: a gateway killed mid-write would otherwise leave a
        # truncated file, and the next one would start by failing to parse it.
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(asdict(state.training), indent=2), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        state.log("warn", f"Could not persist training run status: {exc}")


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by someone else. Treated as alive so the page reports it rather than
        # claiming the GPU is free.
        return True
    return True


def _restore_training_run(state: GatewayState) -> None:
    """Re-adopt a run left behind by a previous gateway, if it is still going."""
    path = _training_run_state_path(state)
    if not path.is_file():
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        state.log("warn", f"Ignoring unreadable training run status: {exc}")
        return
    if not isinstance(stored, dict):
        return
    fields = {field.name for field in dataclasses.fields(training_backend.TrainingRunStatus)}
    status = training_backend.TrainingRunStatus(
        **{key: value for key, value in stored.items() if key in fields}
    )
    if status.state not in ("syncing", "starting", "running") or not status.pid:
        # A run that had already finished is kept for display, not followed.
        state.training = status
        return
    if not _process_is_alive(int(status.pid)):
        status.state = "error"
        status.message = (
            f"Training run {status.jobName} was interrupted (pid {status.pid} is gone). "
            f"Its log is at {status.logPath}."
        )
        status.finishedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state.training = status
        _persist_training_status(state)
        state.log("warn", status.message)
        return

    state.training = status
    state.log("info", f"Re-attached to training run {status.jobName} (pid {status.pid})")
    log_path = Path(status.logPath)
    if not log_path.is_file():
        state.training.message = (
            f"{status.jobName} is still running (pid {status.pid}), but its log file is gone; "
            "progress cannot be followed from here."
        )
        return
    Thread(
        target=_follow_training_run,
        args=(state, log_path, int(status.pid)),
        kwargs={"from_start": False},
        daemon=True,
        name=f"training-tail-{status.pid}",
    ).start()


def _unique_job_name(repo_root: Path, base: str) -> str:
    """`base`, or the first `base-N` no run directory has taken yet.

    The timestamp in `base` already separates any two runs a person starts, because a person
    cannot click twice inside one second. This closes the rest of the gap: a run that dies during
    startup and is restarted by a script within the same second would land on the directory the
    dead one left behind, and the operator would be back to reading a FileExistsError from
    lerobot_train -- the exact failure the stamp exists to remove.

    Exact for the host this gateway runs on, which is where `local` training writes. A remote
    host's outputs/train is not visible from here, so there the timestamp is the whole guarantee.
    """
    train_root = repo_root / "outputs" / "train"
    candidate = base
    suffix = 2
    while (train_root / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _start_training_run(state: GatewayState, payload: dict[str, Any]) -> dict[str, Any]:
    if _training_is_running(state):
        raise ValueError("A training run is already in progress; stop it before starting another.")

    host = training_backend.resolve_host(state.repo_root, str(payload.get("hostId") or ""))
    view_name = str(payload.get("viewName") or "").strip()
    views = {entry["name"]: entry for entry in _training_view_entries(state)}
    if view_name not in views:
        raise ValueError(f"Unknown training view {view_name!r}. Build it on the Dataset Export page first.")
    view = views[view_name]
    if view["episodes"] < 1:
        raise ValueError(f"{view_name} has no episodes to train on.")

    policy = str(payload.get("policy") or "act").strip() or "act"
    run_name = str(payload.get("jobName") or "").strip() or f"{view_name}__{policy}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("Job name may only contain letters, digits, '.', '_' and '-'.")
    # Every start gets its own directory, by construction rather than by the operator
    # remembering to rename. lerobot_train refuses to run when its output_dir already exists and
    # it is not resuming (src/lerobot/configs/train.py), and upstream stays clear of that by
    # timestamping the path it picks. This helper overrides output_dir with
    # outputs/train/<job_name>, and the name above is a function of (view, policy) -- two
    # coordinates that are *meant* to hold still while you retrain the same data with more steps
    # or a different chunk size. So the second run of any view collided by construction, and the
    # only advice the error could give was "rename it yourself".
    #
    # Same stamp as the log file below on purpose: the run directory and the log that recorded it
    # are the two halves an operator pairs up afterwards, and a second `datetime.now()` would
    # drift them apart across a midnight or a slow sync.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_name = _unique_job_name(state.repo_root, f"{run_name}__{stamp}")

    # A remote host trains from its own checkout, so the code it runs is whatever was last
    # synced there. Doing it here rather than leaving it to the operator is the difference
    # between "training the fix" and "training whatever was there".
    sync_result: dict[str, Any] = {"skipped": True}
    if host.kind == "remote":
        state.training.state = "syncing"
        state.training.message = f"Syncing code to {host.sshTarget}…"
        sync_result = training_backend.sync_repo_to_host(state.repo_root, host)
        if not sync_result.get("ok"):
            state.training.state = "error"
            state.training.message = str(sync_result.get("message") or "sync failed")
            raise ValueError(state.training.message)

    wandb_enabled = bool(payload.get("wandbEnabled"))
    wandb_key = training_backend.read_wandb_key(host.id) if wandb_enabled else ""
    if wandb_enabled and not wandb_key:
        raise ValueError(
            f"W&B logging is on but no API key is stored for {host.label}. Add one on the Training page."
        )
    if wandb_enabled:
        training_backend.push_wandb_key(host, wandb_key)

    # The view path is the *host's*: a remote machine's repo lives elsewhere, and the local
    # absolute path would silently miss.
    view_root = view["root"]
    if host.kind == "remote":
        try:
            relative = Path(view_root).relative_to(state.repo_root)
        except ValueError:
            raise ValueError(
                f"{view_name} lives outside the repo ({view_root}); a remote host cannot resolve it."
            ) from None
        view_root = f"{host.repoDir}/{relative.as_posix()}"

    argv = training_backend.build_train_argv(
        host=host,
        view_root=view_root,
        repo_id=view["repoId"],
        job_name=job_name,
        policy=policy,
        steps=int(payload.get("steps") or 20000),
        batch_size=int(payload.get("batchSize") or 8),
        num_workers=int(payload.get("numWorkers") or 4),
        save_freq=int(payload.get("saveFreq") or 5000),
        log_freq=int(payload.get("logFreq") or 100),
        device=str(payload.get("device") or "auto"),
        use_amp=bool(payload.get("useAmp")),
        policy_config=str(payload.get("policyConfig") or ""),
        wandb_enabled=wandb_enabled,
        wandb_project=str(payload.get("wandbProject") or "lerobot"),
        wandb_entity=str(payload.get("wandbEntity") or ""),
    )
    command, env = training_backend.build_launch_command(
        state.repo_root, host, argv, wandb_key=wandb_key
    )

    log_dir = state.repo_root / "outputs" / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{job_name}.log"

    # The run's output goes straight to the log file, not through a pipe this gateway holds.
    # With a pipe, the gateway's death closes the read end and the trainer takes SIGPIPE on its
    # next line -- so restarting the gateway (which every deploy does) killed the job. Writing
    # to the file makes the log the record and the gateway merely a reader of it, which is also
    # what lets a restarted gateway pick the run back up.
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=state.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    state.training_process = process
    state.training = training_backend.TrainingRunStatus(
        state="starting",
        hostId=host.id,
        hostLabel=host.label,
        viewName=view_name,
        viewRoot=view_root,
        policy=policy,
        jobName=job_name,
        outputDir=f"outputs/train/{job_name}",
        totalSteps=int(payload.get("steps") or 20000),
        message=f"Training {policy} on {view_name} at {host.label}",
        pid=process.pid,
        startedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        logPath=str(log_path),
        wandbEnabled=wandb_enabled,
    )
    state.log("info", f"Started training {policy} on {view_name} at {host.label} (pid {process.pid})")
    _persist_training_status(state)
    Thread(
        target=_follow_training_run,
        args=(state, log_path, process.pid, process),
        kwargs={"from_start": True},
        daemon=True,
        name=f"training-output-{process.pid}",
    ).start()
    return {"ok": True, "training": asdict(state.training), "sync": sync_result}


def _stop_training_run(state: GatewayState) -> dict[str, Any]:
    process = state.training_process
    # A run re-adopted after a gateway restart has no Popen object -- the pipe died with the
    # gateway that opened it -- but it is still on the GPU, so "stop" has to reach it by pid.
    # Without this the only way to end such a run would be to ssh in and kill it by hand.
    pid = process.pid if process is not None else state.training.pid
    running = process.poll() is None if process is not None else bool(pid and _process_is_alive(int(pid)))
    if not pid or not running:
        return {"ok": True, "training": asdict(state.training), "message": "No training run is active."}
    state.training.state = "stopped"
    state.training.message = "Stopping training run…"
    try:
        # The whole session: a remote run is an ssh client with the trainer on the far end,
        # and killing only the client would leave the GPU busy with an orphan.
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        if process is not None:
            process.terminate()
    state.log("warn", f"Stopped training run {state.training.jobName}")
    _persist_training_status(state)
    return {"ok": True, "training": asdict(state.training)}


# ------------------------------------------------------- checkpoints & rollout ---


def _rig_contract(state: GatewayState) -> checkpoint_backend.RigContract:
    """What this rig is right now, as the thing a checkpoint has to agree with.

    Read from the gateway's own config and the inference camera file rather than from the
    rollout launcher, so the comparison is against the hardware as configured and not against
    a second copy of the same numbers.
    """
    try:
        robot_ip = _real_robot_ip(state)
    except ValueError:
        robot_ip = ""
    import yaml  # local, matching how the rest of this module pulls it in

    camera_config_rel = "tools/fr3/fr3_il_infer_realsense_camera_config.yaml"
    camera_keys: list[str] = []
    camera_path = state.repo_root / camera_config_rel
    if camera_path.is_file():
        try:
            loaded = yaml.safe_load(camera_path.read_text(encoding="utf-8")) or {}
            cameras = ((loaded.get("robot") or {}).get("cameras") or {})
            camera_keys = sorted(str(key) for key in cameras)
        except (OSError, yaml.YAMLError):
            camera_keys = []
    return checkpoint_backend.RigContract(
        robotIp=robot_ip,
        targetFrameName=_fr3_target_frame_name(state),
        cameraKeys=camera_keys,
        cameraConfigPath=camera_config_rel,
    )


def _checkpoint_entries(state: GatewayState, host_id: str) -> dict[str, Any]:
    """Every checkpoint on one host, each already judged against this rig.

    The judgement travels with the listing rather than being computed when a rollout starts,
    because the point of showing it is to stop an operator from picking a checkpoint that
    cannot be rolled out -- which is information they need while choosing, not after.
    """
    host = training_backend.resolve_host(state.repo_root, host_id)
    report = checkpoint_backend.scan_host(state.repo_root, host)
    rig = _rig_contract(state)
    outcomes = checkpoint_backend.outcome_summary(
        checkpoint_backend.load_rollout_outcomes(state.repo_root)
    )
    entries: list[dict[str, Any]] = []
    for raw in report.get("checkpoints") or []:
        entry = dict(raw)
        entry["hostId"] = host.id
        entry["hostLabel"] = host.label
        entry["contract"] = checkpoint_backend.parse_inference_contract(
            str(entry.pop("inferenceConfigText", "") or "")
        )
        issues = checkpoint_backend.check_contract(entry, rig=rig, local=host.kind == "local")
        entry["issues"] = [asdict(issue) for issue in issues]
        entry["verdict"] = checkpoint_backend.verdict_for(issues)
        entry["outcomes"] = outcomes.get(str(entry.get("id") or ""), None)
        entries.append(entry)
    return {
        "ok": bool(report.get("ok")),
        "error": report.get("error", ""),
        "detail": report.get("detail", []),
        "host": asdict(host),
        "rig": asdict(rig),
        "checkpoints": entries,
    }


def _rollout_is_running(state: GatewayState) -> bool:
    process = state.rollout_process
    return process is not None and process.poll() is None


def _guard_checkpoint_deletion(state: GatewayState, checkpoint_ids: list[str]) -> None:
    """Refuse to delete weights that something on this machine is still holding.

    Two live holders, and neither fails loudly when the directory disappears underneath it. A
    rollout has already loaded the policy, so it keeps driving the arm from weights that are no
    longer on disk and only the next restart reveals what happened. A training run owns the job
    directory it is still saving steps into, and deleting one of them leaves a run whose
    checkpoint series has a hole in it.

    Checked here rather than in the backend because liveness is gateway state, not disk state.
    One click could always do this to a single checkpoint; select-all makes it one click for
    every checkpoint at once, which is what turns a sharp edge into a guard worth having.
    """
    ids = {item for item in checkpoint_ids if item}
    if not ids:
        return
    if _rollout_is_running(state) and state.rollout.checkpointId in ids:
        raise checkpoint_backend.CheckpointError(
            f"{state.rollout.checkpointId} is loaded by the rollout running right now. "
            "Stop it on the Rollout page before deleting it."
        )
    if _training_is_running(state):
        job_name = state.training.jobName
        live = sorted(item for item in ids if item.split("/", 1)[0] == job_name)
        if live:
            raise checkpoint_backend.CheckpointError(
                f"{job_name} is training right now and still writing checkpoints; "
                f"{', '.join(live)} belong(s) to it. Stop the run first."
            )


def _apply_rollout_output(state: GatewayState, line: str) -> None:
    status = state.rollout
    parsed = rollout_backend.parse_rollout_line(line)
    for key, value in parsed.items():
        if key == "state":
            # A stop the operator already asked for is not undone by a line the runtime wrote
            # before it noticed. Same reasoning as the training follower.
            if status.state in ("stopped", "error"):
                continue
            status.state = str(value)
        else:
            setattr(status, key, value)
    if parsed.get("commandStatus") == "step_limited":
        status.clampedSteps += 1
    elif parsed.get("commandStatus") == "leash_limited":
        status.leashedSteps += 1
    if not rollout_backend.is_noise(line):
        status.lastLines = (status.lastLines + [line])[-80:]


def _follow_rollout_run(state: GatewayState, log_path: Path, process: subprocess.Popen[bytes]) -> None:
    """Read the rollout's log file and record how the session ended.

    Same file-tailing shape as the training follower and for the same reason -- the gateway
    must not hold the process's stdout pipe. Rollout differs only in holding *stdin*, which is
    the control channel and is supposed to die with the gateway.
    """
    pid = process.pid
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                line = handle.readline()
                if line:
                    for part in line.replace("\r", "\n").splitlines():
                        output = part.rstrip()
                        if not output:
                            continue
                        with state.lock:
                            if state.rollout.pid != pid:
                                return
                            _apply_rollout_output(state, output)
                    continue
                if process.poll() is not None:
                    remaining = handle.read()
                    for part in remaining.replace("\r", "\n").splitlines():
                        output = part.rstrip()
                        if not output:
                            continue
                        with state.lock:
                            if state.rollout.pid != pid:
                                return
                            _apply_rollout_output(state, output)
                    break
                time.sleep(0.4)
    except OSError as exc:
        with state.lock:
            if state.rollout.pid == pid:
                state.rollout.message = f"Could not follow rollout log: {exc}"

    return_code = process.wait()
    with state.lock:
        if state.rollout.pid != pid:
            return
        state.rollout.finishedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if state.rollout.state == "stopped":
            pass
        elif return_code == 0:
            state.rollout.state = "complete"
            state.rollout.message = f"Rollout session finished ({state.rollout.mode})."
        else:
            state.rollout.state = "error"
            state.rollout.message = (
                f"Rollout exited with code {return_code}; see {state.rollout.logPath}."
            )
        state.log(
            "info" if state.rollout.state == "complete" else "error",
            f"Rollout {state.rollout.mode} on {state.rollout.checkpointId} "
            f"finished with code {return_code}",
        )


def _start_rollout(state: GatewayState, payload: dict[str, Any]) -> dict[str, Any]:
    if state.profile != "workstation":
        raise ValueError("Rollouts run on the workstation profile, which is where the FR3 is.")
    if _rollout_is_running(state):
        raise ValueError("A rollout is already running; stop it before starting another.")
    if _training_is_running(state) and str(payload.get("mode") or "") != "env":
        # Both want the GPU, and the rollout is the one with a deadline: a policy starved of
        # inference time still sends commands, just later than the arm expects them.
        raise ValueError(
            "A training run is using the GPU. Stop it before rolling out, or the policy will "
            "miss its control deadlines."
        )

    mode_id = str(payload.get("mode") or "").strip()
    mode = rollout_backend.MODES_BY_ID.get(mode_id)
    if mode is None:
        raise ValueError(f"Unknown rollout mode {mode_id!r}.")
    if mode.movesArm and not bool(payload.get("confirmMotion")):
        raise ValueError(
            f"{mode.label} moves the arm. Confirm motion before starting it."
        )
    if mode.id == "real_debug" and not os.environ.get("DISPLAY"):
        # Refused rather than started: the viewer is the entire difference between this mode
        # and `real`, and without a display the operator would home the arm and run rollouts
        # waiting for a window that is never going to open.
        raise ValueError(
            "real_debug opens a MuJoCo viewer on the rig's own screen, and this gateway has no "
            "X display. Log in graphically on the workstation and redeploy, or use "
            "'Interactive rollouts' — it is the same rollout without the viewer."
        )

    checkpoint_id = str(payload.get("checkpointId") or "")
    checkpoint_backend.validate_checkpoint_id(checkpoint_id)
    listing = _checkpoint_entries(state, training_backend.LOCAL_HOST_ID)
    selected = next(
        (item for item in listing["checkpoints"] if item.get("id") == checkpoint_id), None
    )
    if selected is None:
        raise ValueError(
            f"No checkpoint {checkpoint_id} on this machine. Fetch it from its training host first."
        )
    blocking = [issue for issue in selected["issues"] if issue["level"] == "block"]
    if blocking and not bool(payload.get("overrideContract")):
        raise ValueError(
            "This checkpoint does not match the rig: "
            + " ".join(issue["message"] for issue in blocking)
        )

    contract = selected.get("contract") or {}
    rig = listing["rig"]
    # The checkpoint's own recorded frame wins over the rig default. It is the frame its dataset
    # was anchored to, and that is what the action deltas mean.
    target_frame = str(contract.get("targetFrameName") or rig.get("targetFrameName") or "")
    camera_config = str(contract.get("cameraConfig") or rig.get("cameraConfigPath") or "")

    command, env = rollout_backend.build_rollout_command(
        state.repo_root,
        mode=mode.id,
        checkpoint_path=str(selected.get("path") or ""),
        dataset_root=str((selected.get("view") or {}).get("root") or ""),
        target_frame_name=target_frame,
        robot_ip=str(rig.get("robotIp") or ""),
        camera_config=camera_config,
        max_steps=int(payload.get("maxSteps") or 0),
        move_to_start=bool(payload.get("moveToStart", True)),
        base_env=_tool_env(state.repo_root),
    )

    log_dir = state.repo_root / "outputs" / "logs" / "rollout"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"rollout_{checkpoint_id.replace('/', '_')}_{mode.id}_{stamp}.log"

    try:
        rollout_backend.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        state.log("warn", f"Could not create rollout preview directory: {exc}")

    # stdout to the file, stdin held as a pipe. The asymmetry is the design: the gateway must
    # not own the run's output (that is what killed training runs on every deploy), but it must
    # own its input, because that is the only way to stop a moving arm from a browser.
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=state.repo_root,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    state.rollout_process = process
    state.rollout = rollout_backend.RolloutStatus(
        state="starting",
        mode=mode.id,
        checkpointId=checkpoint_id,
        checkpointPath=str(selected.get("path") or ""),
        policy=str(selected.get("policyType") or ""),
        datasetRoot=str((selected.get("view") or {}).get("root") or ""),
        targetFrameName=target_frame,
        robotIp=str(rig.get("robotIp") or ""),
        cameraKeys=list(selected.get("cameras") or []),
        interactive=mode.interactive,
        movesArm=mode.movesArm,
        maxSteps=int(payload.get("maxSteps") or 0),
        pid=process.pid,
        message=f"Starting {mode.label} on {checkpoint_id}…",
        startedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        logPath=str(log_path),
        previewDir=str(rollout_backend.PREVIEW_DIR),
    )
    state.log(
        "warn" if mode.movesArm else "info",
        f"Started rollout {mode.id} on {checkpoint_id} "
        f"(tool_frame={target_frame}, pid={process.pid})",
    )
    Thread(
        target=_follow_rollout_run,
        args=(state, log_path, process),
        daemon=True,
        name=f"rollout-output-{process.pid}",
    ).start()
    return {"ok": True, "rollout": asdict(state.rollout)}


def _send_rollout_control(state: GatewayState, command: str) -> dict[str, Any]:
    """Write one control word to the rollout's stdin.

    The runtime reads this pipe one line at a time (InteractiveRolloutKeyboard's pipe backend),
    so a word here is exactly a keypress there.
    """
    allowed = {"start", "stop", "quit"}
    if command not in allowed:
        raise ValueError(f"Rollout control must be one of {', '.join(sorted(allowed))}.")
    process = state.rollout_process
    if process is None or process.poll() is not None or process.stdin is None:
        raise ValueError("No rollout is running.")
    if not state.rollout.interactive:
        raise ValueError(
            f"{state.rollout.mode} is not an interactive mode; it runs to completion on its own."
        )
    try:
        process.stdin.write(f"{command}\n".encode())
        process.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise ValueError(f"Rollout is no longer accepting control commands: {exc}") from exc
    if command == "start":
        state.rollout.message = "Start sent."
    elif command == "stop":
        state.rollout.message = "Stop sent; ending the current rollout."
    else:
        state.rollout.state = "stopped"
        state.rollout.message = "Quit sent; ending the rollout session."
    state.log("info", f"Rollout control: {command}")
    return {"ok": True, "rollout": asdict(state.rollout)}


def _stop_rollout(state: GatewayState) -> dict[str, Any]:
    process = state.rollout_process
    if process is None or process.poll() is not None:
        return {"ok": True, "rollout": asdict(state.rollout), "message": "No rollout is running."}
    state.rollout.state = "stopped"
    state.rollout.message = "Stopping rollout…"
    # Ask first: the runtime's quit path disconnects the robot through its own `finally`, which
    # releases the arm in a controlled way. SIGTERM to the group is the fallback for a process
    # that is no longer reading its stdin.
    try:
        if process.stdin is not None:
            process.stdin.write(b"quit\n")
            process.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    state.log("warn", f"Stopped rollout {state.rollout.mode} on {state.rollout.checkpointId}")
    return {"ok": True, "rollout": asdict(state.rollout)}


def _record_rollout_outcome(state: GatewayState, payload: dict[str, Any]) -> dict[str, Any]:
    entry = checkpoint_backend.append_rollout_outcome(
        state.repo_root,
        {
            "checkpointId": str(payload.get("checkpointId") or state.rollout.checkpointId),
            "outcome": str(payload.get("outcome") or ""),
            "mode": str(payload.get("mode") or state.rollout.mode),
            "steps": int(payload.get("steps") or state.rollout.step),
            "note": str(payload.get("note") or ""),
            "logPath": str(payload.get("logPath") or state.rollout.logPath),
        },
    )
    # Clearing the prompt is what makes it fire once per rollout rather than on every poll.
    state.rollout.pendingOutcomeFor = 0
    state.log("info", f"Recorded rollout outcome {entry['outcome']} for {entry['checkpointId']}")
    return {"ok": True, "entry": entry}


def _rollout_preview_frame(camera_key: str) -> bytes | None:
    path = rollout_backend.PREVIEW_DIR / f"{camera_key}.jpg"
    try:
        stat_result = path.stat()
        if time.time() - stat_result.st_mtime > rollout_backend.PREVIEW_STALE_S:
            return None
        frame = path.read_bytes()
    except OSError:
        return None
    return frame or None


def _serve_rollout_camera_snapshot(
    handler: BaseHTTPRequestHandler, *, state: GatewayState, camera_key: str
) -> None:
    if camera_key not in set(state.rollout.cameraKeys):
        _json_response(handler, HTTPStatus.NOT_FOUND, {"error": f"unknown rollout camera: {camera_key}"})
        return
    frame = _rollout_preview_frame(camera_key)
    if frame is None:
        _json_response(
            handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no rollout preview frame yet"}
        )
        return
    try:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Content-Length", str(len(frame)))
        handler.end_headers()
        handler.wfile.write(frame)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _annotation_store_path(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "gui_annotations.json"


def _read_annotation_store(dataset_root: Path) -> dict[str, Any]:
    path = _annotation_store_path(dataset_root)
    if not path.is_file():
        return {"version": 1, "annotations": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "annotations": {}}
    if not isinstance(loaded, dict):
        return {"version": 1, "annotations": {}}
    annotations = loaded.get("annotations")
    if not isinstance(annotations, dict):
        loaded["annotations"] = {}
    loaded.setdefault("version", 1)
    return loaded


def _write_annotation_store(dataset_root: Path, store: dict[str, Any]) -> None:
    path = _annotation_store_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _dataset_task_prompt(dataset_root: Path, config: dict[str, Any]) -> str:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(tasks_path), columns=["task"]).to_pydict()
            tasks = [str(task).strip() for task in table.get("task", []) if str(task).strip()]
            if tasks:
                return tasks[0]
        except Exception:
            pass
    return str(_dataset_config(config).get("single_task") or "").strip()


def _annotation_excluded_episodes(dataset_root: Path) -> list[int]:
    """Episodes the operator marked as not for training, in view order.

    The training-view builder reads the same store for itself, so this is what the *page* shows,
    not what it enforces -- an exclusion the operator cannot see before pressing Build View is
    how a training set quietly stops matching the recording it names.
    """
    store = _read_annotation_store(dataset_root)
    annotations = store.get("annotations") if isinstance(store, dict) else None
    if not isinstance(annotations, dict):
        return []
    excluded: set[int] = set()
    for key, annotation in annotations.items():
        if not isinstance(annotation, dict) or annotation.get("includeInTraining", True):
            continue
        try:
            excluded.add(int(annotation.get("episode", key)))
        except (TypeError, ValueError):
            continue
    return sorted(excluded)


def _normalize_annotation(
    payload: dict[str, Any],
    *,
    dataset_root: Path,
    episode: int,
    default_prompt: str,
    source: str,
) -> dict[str, Any]:
    outcome = str(payload.get("outcome") or "unreviewed")
    if outcome not in {"unreviewed", "success", "failure", "partial"}:
        outcome = "unreviewed"
    quality = str(payload.get("quality") or "unreviewed")
    if quality not in {"unreviewed", "good", "needs_review", "bad"}:
        quality = "unreviewed"
    raw_tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()][:12]
    review_status = str(payload.get("reviewStatus") or "pending")
    if review_status not in {"pending", "approved", "rejected"}:
        review_status = "pending"
    raw_segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    segments = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        try:
            segments.append({
                "id": str(seg.get("id") or f"{len(segments)}"),
                "startFrame": max(0, int(seg.get("startFrame", 0))),
                "endFrame": max(0, int(seg.get("endFrame", 0))),
                "description": str(seg.get("description") or "").strip(),
            })
        except (TypeError, ValueError):
            continue
    return {
        "datasetRoot": str(dataset_root),
        "episode": episode,
        "taskPrompt": str(payload.get("taskPrompt") or default_prompt).strip(),
        "outcome": outcome,
        "quality": quality,
        "includeInTraining": bool(payload.get("includeInTraining", True)),
        "tags": tags,
        "notes": str(payload.get("notes") or "").strip(),
        "annotator": str(payload.get("annotator") or "").strip(),
        "updatedAt": str(payload.get("updatedAt") or ""),
        "source": source,
        "segments": segments,
        "reviewStatus": review_status,
        "reviewComment": str(payload.get("reviewComment") or "").strip(),
    }


def _active_annotation(state: GatewayState) -> dict[str, Any]:
    dataset_root = state.selected_replay_root or _resolve_known_dataset(
        state,
        state.replay.datasetRoot or state.replay.dataset,
    )
    if dataset_root is None:
        candidates = _replay_dataset_candidates(state)
        dataset_root = candidates[0] if candidates else None
    if dataset_root is None:
        return {
            **_normalize_annotation(
                {},
                dataset_root=Path("."),
                episode=int(state.replay.episode or 0),
                default_prompt=str(_dataset_config(state.config).get("single_task") or ""),
                source="default",
            ),
            "datasetRoot": state.replay.datasetRoot or state.replay.dataset or "",
        }
    episode = int(state.replay.episode or 0)
    store = _read_annotation_store(dataset_root)
    raw = store.get("annotations", {}).get(str(episode))
    if isinstance(raw, dict):
        return _normalize_annotation(
            raw,
            dataset_root=dataset_root,
            episode=episode,
            default_prompt=_dataset_task_prompt(dataset_root, state.config),
            source="manual",
        )
    return _normalize_annotation(
        {},
        dataset_root=dataset_root,
        episode=episode,
        default_prompt=_dataset_task_prompt(dataset_root, state.config),
        source="dataset" if _dataset_task_prompt(dataset_root, state.config) else "default",
    )


def _save_annotation(state: GatewayState, payload: dict[str, Any]) -> None:
    raw_dataset = str(payload.get("datasetRoot") or state.replay.datasetRoot or state.replay.dataset or "").strip()
    dataset_root = _resolve_known_dataset(state, raw_dataset) or state.selected_replay_root
    if dataset_root is None:
        raise ValueError("Annotation dataset is not in the recorded dataset list.")
    episode = int(payload.get("episode", state.replay.episode or 0))
    default_prompt = _dataset_task_prompt(dataset_root, state.config)
    annotation = _normalize_annotation(
        payload,
        dataset_root=dataset_root,
        episode=episode,
        default_prompt=default_prompt,
        source="manual",
    )
    annotation["updatedAt"] = _now_iso()
    store = _read_annotation_store(dataset_root)
    annotations = store.setdefault("annotations", {})
    annotations[str(episode)] = annotation
    store["updatedAt"] = annotation["updatedAt"]
    _write_annotation_store(dataset_root, store)
    state.log("info", f"Saved annotation for {dataset_root.name} episode {episode}")


FR3_SYNC_REPORT_RELATIVE_PATH = Path("meta") / "fr3_sync_report.json"
# The verdict rule changed in v3: it moved off the raw all-device spread (which is dominated by
# the cameras' constant offset from the arm read and failed every hardware episode) onto the
# within-group / residual / bias split. A v2 report's `status` therefore cannot be believed here.
FR3_SYNC_REPORT_MIN_SCHEMA = 3


def _fr3_sync_report_check(dataset_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fold the timestamp-sync audit into QC, recomputing it when it is missing or stale.

    QC used to be silent about alignment: a dataset whose cameras and arm disagreed still passed
    every check here and reached Dataset Export, while the only verdict on it sat in a file the
    export path never read. Recomputing rather than trusting a stale file matters for the same
    reason -- an interrupted session never reaches ``finalize()``, so it has no report at all.

    Returns ``(report, check)``. Both are ``None`` when there is nothing to judge (no
    capture-timestamp column, or no numpy in this interpreter -- the Thor gateway runs a bare
    system python and its datasets carry no such column anyway).
    """
    report_path = dataset_root / FR3_SYNC_REPORT_RELATIVE_PATH
    report: dict[str, Any] | None = None
    recomputed = False
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and int(loaded.get("schema_version") or 0) >= FR3_SYNC_REPORT_MIN_SCHEMA:
            report = loaded

    if report is None:
        try:
            from tools.fr3.fr3_sync_audit import write_fr3_sync_report
        except Exception:  # noqa: BLE001 - numpy is optional in the gateway's interpreter
            return None, None
        try:
            report, _ = write_fr3_sync_report(
                dataset_root,
                # No sensor rate is available here -- a dataset records dataset.fps, not the
                # camera's, and this gateway's config may not be the one that recorded it. The
                # residual is measured and reported without a verdict rather than judged against
                # a budget borrowed from another rig.
                residual_tolerance_ms=None,
            )
        except Exception:  # noqa: BLE001 - a dataset without the column is not a QC failure
            return None, None
        recomputed = True

    status = str(report.get("status") or "").lower()
    failures = [str(item) for item in (report.get("failures") or [])]
    source = "recomputed" if recomputed else "meta/fr3_sync_report.json"
    if status == "pass":
        message = f"capture timestamps within budget ({source})"
    else:
        message = failures[0] if failures else f"timestamp sync reported {status or 'no verdict'}"
        if len(failures) > 1:
            message = f"{message} (+{len(failures) - 1} more)"
    return report, {
        "name": "timestamp_sync",
        "status": "pass" if status == "pass" else "fail",
        "message": message,
        "details": {
            "clock_semantics": report.get("clock_semantics"),
            "budgets_ms": (report.get("skew_evaluation") or {}).get("budgets_ms"),
            "bias_ms": report.get("cross_modality_bias_ms"),
            "failures": failures,
        },
    }


def _timestamp_sync_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """The operator-facing digest of a sync report: budgets, what was measured, and the offsets.

    The offsets are carried through deliberately. They are the reason the verdict is a three-way
    split rather than one spread, so a panel that showed only pass/fail would hide the very
    quantity that makes the verdict readable.
    """
    if not isinstance(report, dict):
        return None
    evaluation = report.get("skew_evaluation") if isinstance(report.get("skew_evaluation"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    within_group = evaluation.get("within_group") if isinstance(evaluation.get("within_group"), dict) else {}
    residual = evaluation.get("residual") if isinstance(evaluation.get("residual"), dict) else {}
    return {
        "status": str(report.get("status") or "unknown"),
        "clockSemantics": str(report.get("clock_semantics") or ""),
        "totalFrames": int(report.get("total_frames") or 0),
        "budgetsMs": evaluation.get("budgets_ms") or {},
        "groupSkewP95Ms": max(
            (float(entry.get("p95_ms") or 0.0) for entry in within_group.values()),
            default=None,
        ),
        "groupSkewOverBudgetFrames": int(summary.get("within_group_skew_over_budget_frames") or 0),
        "residualSkewP95Ms": float(residual.get("p95_ms")) if residual.get("p95_ms") is not None else None,
        "residualSkewOverBudgetFrames": summary.get("residual_skew_over_budget_frames"),
        "gridLagOverBudgetFrames": int(summary.get("global_lag_over_tolerance_frames") or 0),
        "rawSkewP95Ms": float((evaluation.get("raw_all_device") or {}).get("p95_ms") or 0.0),
        "biasMs": report.get("cross_modality_bias_ms") or {},
        "failures": [str(item) for item in (report.get("failures") or [])],
    }


def _run_fr3_ik_qc(
    dataset_root: Path,
    *,
    repo_root: Path,
    python_executable: Path,
    fps: int,
) -> dict[str, Any]:
    sidecar_dir = dataset_root / "derived" / DEFAULT_TRAJ_SIDECAR_NAME
    cube_paths = {
        cube: sidecar_dir / f"state_action.{cube}.csv"
        for cube in ("left", "right")
        if (sidecar_dir / f"state_action.{cube}.csv").is_file()
    }
    if not cube_paths:
        return {
            "status": "skipped",
            "message": "No left/right FR3 EE trajectory sidecar is available for offline IK evaluation.",
            "cubes": [],
        }

    script_path = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.py"
    config_path = repo_root / "third_party" / "opencv_kalibr" / "verification" / "verify_fr3_cube_pose_ik.thor.yaml"
    if not script_path.is_file() or not config_path.is_file():
        return {
            "status": "fail",
            "message": "FR3 IK verifier or Thor configuration is missing.",
            "cubes": [],
        }

    cube_results: list[dict[str, Any]] = []
    for cube, csv_path in cube_paths.items():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                has_valid_pose = any(_pose_from_csv_row(row) is not None for row in csv.DictReader(csv_file))
        except (OSError, csv.Error) as exc:
            cube_results.append({
                "cube": cube,
                "status": "fail",
                "message": f"could not read trajectory sidecar: {exc}",
            })
            continue
        if not has_valid_pose:
            cube_results.append({
                "cube": cube,
                "status": "skipped",
                "message": "no finite EE target poses in trajectory sidecar",
            })
            continue

        output_dir = sidecar_dir / "ik_qc" / cube
        report_path = output_dir / "verify_fr3_cube_pose_ik_report.json"
        rows_path = output_dir / "verify_fr3_cube_pose_ik_rows.csv"
        command = [
            str(python_executable),
            str(script_path),
            f"--config_path={config_path}",
            f"--input.csv_path={csv_path}",
            f"--input.dataset_pose_name={cube}",
            f"--replay.replay_fps={max(int(fps), 1)}",
            f"--validation.report_json_path={report_path}",
            f"--validation.report_csv_path={rows_path}",
            "--validation.write_episode_labels_to_dataset=false",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_tool_env(repo_root),
                timeout=900.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            cube_results.append({
                "cube": cube,
                "status": "fail",
                "message": "offline IK evaluation timed out after 900 seconds",
                "reportPath": str(report_path),
            })
            continue
        output_tail = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()][-6:]
        report = _load_json_file(report_path)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        if result.returncode != 0 or not summary:
            cube_results.append({
                "cube": cube,
                "status": "fail",
                "message": output_tail[-1] if output_tail else f"verifier exited with code {result.returncode}",
                "exitCode": result.returncode,
                "reportPath": str(report_path),
                "outputTail": output_tail,
            })
            continue

        trajectory = summary.get("trajectory_reachability") if isinstance(summary.get("trajectory_reachability"), dict) else {}
        raw_episode_summary = summary.get("episode_summary") if isinstance(summary.get("episode_summary"), list) else []
        episodes: list[dict[str, Any]] = []
        for raw_episode in raw_episode_summary:
            if not isinstance(raw_episode, dict):
                continue
            reachable = bool(raw_episode.get("trajectory_reachable"))
            episodes.append({
                "episodeIndex": int(raw_episode.get("episode_index") or 0),
                "status": "reachable" if reachable else "unreachable",
                "label": str(raw_episode.get("ik_trajectory_label") or ("reachable" if reachable else "unreachable")),
                "numTargets": int(raw_episode.get("num_targets") or 0),
                "numReachable": int(raw_episode.get("num_reachable") or 0),
                "numUnreachable": int(raw_episode.get("num_unreachable") or 0),
                "reachableRatio": float(raw_episode.get("reachable_ratio") or 0.0),
                "unreachableDurationS": float(raw_episode.get("unreachable_duration_s") or 0.0),
                "maxConsecutiveUnreachableTimesteps": int(
                    raw_episode.get("max_consecutive_unreachable_timesteps") or 0
                ),
                "maxPositionErrorMm": float(raw_episode.get("max_position_error_m") or 0.0) * 1000.0,
                "maxOrientationErrorDeg": float(raw_episode.get("max_orientation_error_deg") or 0.0),
            })
        total_trajectories = int(
            trajectory.get("num_trajectories")
            or trajectory.get("total_trajectories")
            or 0
        )
        unreachable_trajectories = int(trajectory.get("num_unreachable_trajectories") or 0)
        reachable_trajectories = int(
            trajectory.get("num_reachable_trajectories")
            if trajectory.get("num_reachable_trajectories") is not None
            else max(0, total_trajectories - unreachable_trajectories)
        )
        unreachable_targets = int(summary.get("num_unreachable") or 0)
        reachable_ratio = float(summary.get("reachable_ratio") or 0.0)
        status = "fail" if unreachable_trajectories else ("warn" if unreachable_targets else "pass")
        plot_path = report_path.with_name("verify_fr3_cube_pose_ik_error_over_time.png")
        cube_results.append({
            "cube": cube,
            "status": status,
            "message": (
                f"{reachable_trajectories}/{total_trajectories} trajectories reachable; "
                f"{reachable_ratio * 100.0:.2f}% poses reachable"
            ),
            "numTargets": int(summary.get("num_targets") or 0),
            "numUnreachableTargets": unreachable_targets,
            "numUnreachableTrajectories": unreachable_trajectories,
            "reachableRatio": reachable_ratio,
            "reasonCounts": summary.get("reason_counts") or {},
            "ikErrorStats": summary.get("ik_error_stats") or {},
            "episodes": episodes,
            "reachableEpisodeIndices": [row["episodeIndex"] for row in episodes if row["status"] == "reachable"],
            "unreachableEpisodeIndices": [row["episodeIndex"] for row in episodes if row["status"] == "unreachable"],
            "plotAvailable": plot_path.is_file(),
            "reportPath": str(report_path),
            "rowsPath": str(rows_path),
        })

    status = "pass"
    if any(row.get("status") == "fail" for row in cube_results):
        status = "fail"
    elif any(row.get("status") == "warn" for row in cube_results):
        status = "warn"
    message = "; ".join(f"{row['cube']}: {row['message']}" for row in cube_results)
    return {"status": status, "message": message, "cubes": cube_results}


def _run_qc(
    dataset_root: Path,
    *,
    repo_root: Path | None = None,
    ik_python: Path | None = None,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "summary": f"pyarrow unavailable: {exc}",
            "valid_frames_pct": 0.0,
            "checks": [],
            "completed_at": _now_iso(),
        }

    info = _load_dataset_info(dataset_root)
    if not info:
        return {
            "status": "fail",
            "summary": "missing meta/info.json",
            "valid_frames_pct": 0.0,
            "checks": [{"name": "info", "status": "fail", "message": "meta/info.json not found"}],
            "completed_at": _now_iso(),
        }

    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    cube_pose_names = _cube_names_for_timeline(dataset_root, info)
    camera_keys = _camera_keys(info)
    declared_total = int(info.get("total_frames") or 0)

    data_files = _dataset_data_files(dataset_root)
    if not data_files:
        return {
            "status": "fail",
            "summary": "no parquet files under data/",
            "valid_frames_pct": 0.0,
            "checks": [{"name": "data_files", "status": "fail", "message": "data/chunk-*/*.parquet missing"}],
            "completed_at": _now_iso(),
        }

    checks: list[dict[str, Any]] = []
    online_sync_summary, online_sync_check = _online_sync_manifest_check(dataset_root)
    if online_sync_check is not None:
        checks.append(online_sync_check)
    sync_report, sync_check = _fr3_sync_report_check(dataset_root)
    if sync_check is not None:
        checks.append(sync_check)
    total_rows = 0
    invalid_rows = 0
    schema_failed_files = 0
    all_rows: list[dict[str, Any]] = []

    for data_file in data_files:
        try:
            table = pq.read_table(data_file)
        except Exception as exc:  # noqa: BLE001
            checks.append({
                "name": "schema",
                "status": "fail",
                "message": f"{data_file.name}: parquet read failed ({exc})",
            })
            schema_failed_files += 1
            continue

        missing_columns = [col for col in ("observation.state", "timestamp", "frame_index") if col not in table.column_names]
        if missing_columns:
            checks.append({
                "name": "schema",
                "status": "fail",
                "message": f"{data_file.name}: missing columns {missing_columns}",
            })
            schema_failed_files += 1
            continue

        rows = table.to_pylist()
        total_rows += len(rows)
        all_rows.extend(rows)

    if schema_failed_files == 0 and all_rows:
        checks.append({
            "name": "schema",
            "status": "pass",
            "message": f"{len(data_files)} parquet file(s), {total_rows} rows",
        })

    if all_rows:
        by_episode: dict[int, list[dict[str, Any]]] = {}
        for row in all_rows:
            ep = int(row.get("episode_index") or 0)
            by_episode.setdefault(ep, []).append(row)
        for episode_rows in by_episode.values():
            episode_rows.sort(key=lambda row: int(row.get("frame_index") or 0))

        gap_count = 0
        first_gap: tuple[int, int] | None = None
        backward_count = 0
        first_backward: tuple[int, int] | None = None
        for ep_index, episode_rows in by_episode.items():
            frame_indices = [int(row.get("frame_index") or 0) for row in episode_rows]
            for i in range(1, len(frame_indices)):
                if frame_indices[i] - frame_indices[i - 1] != 1:
                    gap_count += 1
                    if first_gap is None:
                        first_gap = (ep_index, frame_indices[i])
            timestamps = [float(row.get("timestamp") or 0.0) for row in episode_rows]
            for i in range(1, len(timestamps)):
                if timestamps[i] + 1e-6 < timestamps[i - 1]:
                    backward_count += 1
                    if first_backward is None:
                        first_backward = (ep_index, frame_indices[i])

        if gap_count:
            checks.append({
                "name": "frame_continuity",
                "status": "fail",
                "message": f"{gap_count} frame gaps across {len(by_episode)} episodes; first ep={first_gap[0]} frame={first_gap[1]}",
            })
            invalid_rows += gap_count
        else:
            checks.append({
                "name": "frame_continuity",
                "status": "pass",
                "message": f"{len(by_episode)} episodes contiguous ({total_rows} frames)",
            })

        if backward_count:
            checks.append({
                "name": "timestamp_monotonic",
                "status": "fail",
                "message": f"{backward_count} backward timestamps; first ep={first_backward[0]} frame={first_backward[1]}",
            })
            invalid_rows += backward_count
        else:
            checks.append({"name": "timestamp_monotonic", "status": "pass", "message": "monotonic non-decreasing per episode"})

        ee_axis_indices = _named_indices(state_names, ("ee.x", "ee.y", "ee.z"))
        if ee_axis_indices is not None:
            jumps = 0
            max_delta = 0.0
            for episode_rows in by_episode.values():
                prev: tuple[float, float, float] | None = None
                for row in episode_rows:
                    values = _as_float_list(row.get("observation.state"))
                    if max(ee_axis_indices) >= len(values):
                        continue
                    current = (values[ee_axis_indices[0]], values[ee_axis_indices[1]], values[ee_axis_indices[2]])
                    if prev is not None:
                        delta = ((current[0] - prev[0]) ** 2 + (current[1] - prev[1]) ** 2 + (current[2] - prev[2]) ** 2) ** 0.5
                        max_delta = max(max_delta, delta)
                        if delta > 0.05:
                            jumps += 1
                    prev = current
            if jumps:
                severity = "warn" if jumps <= max(1, total_rows // 100) else "fail"
                checks.append({
                    "name": "ee_continuity",
                    "status": severity,
                    "message": f"{jumps} consecutive frames moved > 5 cm (max {max_delta * 100:.1f} cm)",
                })
                if severity == "fail":
                    invalid_rows += jumps
            else:
                checks.append({
                    "name": "ee_continuity",
                    "status": "pass",
                    "message": f"max step {max_delta * 100:.1f} cm",
                })

        declared_episodes = int(info.get("total_episodes") or 0)
        if declared_episodes and declared_episodes != len(by_episode):
            checks.append({
                "name": "episode_count",
                "status": "warn",
                "message": f"parquet has {len(by_episode)} episodes but info.json declares {declared_episodes}",
            })

        quat_indices = _named_indices(state_names, ("ee.qx", "ee.qy", "ee.qz", "ee.qw"))
        if quat_indices is not None:
            bad = 0
            for row in all_rows:
                values = _as_float_list(row.get("observation.state"))
                if max(quat_indices) >= len(values):
                    continue
                norm_sq = sum(values[idx] ** 2 for idx in quat_indices)
                if norm_sq <= 0:
                    bad += 1
                    continue
                norm = norm_sq ** 0.5
                if not (0.95 <= norm <= 1.05):
                    bad += 1
            if bad:
                checks.append({
                    "name": "quat_norm",
                    "status": "fail",
                    "message": f"{bad} frames with |q| outside [0.95, 1.05]",
                })
                invalid_rows += bad
            else:
                checks.append({"name": "quat_norm", "status": "pass", "message": "unit quaternions"})

        gripper_idx = _first_named_index(state_names, ("gripper.pos",))
        gripper_source = "observation.state"
        if gripper_idx is None:
            gripper_idx = _first_named_index(action_names, ("gripper.pos",))
            gripper_source = "action"
        if gripper_idx is not None:
            out_of_range = 0
            for row in all_rows:
                values = _as_float_list(row.get(gripper_source))
                if gripper_idx >= len(values):
                    continue
                value = values[gripper_idx]
                if value != value or not (-0.05 <= value <= 1.05):
                    out_of_range += 1
            if out_of_range:
                checks.append({
                    "name": "gripper_range",
                    "status": "fail",
                    "message": f"{out_of_range} frames outside [0, 1] in {gripper_source}",
                })
                invalid_rows += out_of_range
            else:
                checks.append({"name": "gripper_range", "status": "pass", "message": f"in [0, 1] ({gripper_source})"})

        if camera_keys:
            missing_cams: list[str] = []
            for cam in camera_keys:
                cam_dir = dataset_root / "videos" / cam
                if not cam_dir.is_dir() or not any(cam_dir.glob("chunk-*/*.mp4")):
                    missing_cams.append(cam)
            if missing_cams:
                checks.append({
                    "name": "video_presence",
                    "status": "fail",
                    "message": f"missing mp4 for: {missing_cams[:3]}{'…' if len(missing_cams) > 3 else ''}",
                })
            else:
                checks.append({
                    "name": "video_presence",
                    "status": "pass",
                    "message": f"{len(camera_keys)} camera streams present",
                })

        if declared_total and total_rows != declared_total:
            checks.append({
                "name": "frame_count",
                "status": "warn",
                "message": f"parquet has {total_rows} rows but info.json declares {declared_total}",
            })

    ik_evaluation = _run_fr3_ik_qc(
        dataset_root,
        repo_root=(repo_root or Path.cwd()).resolve(),
        # Do not resolve this path: venv ``bin/python`` is commonly a symlink
        # to the base interpreter, and dereferencing it bypasses pyvenv.cfg and
        # all packages installed in the selected FR3 environment.
        python_executable=Path(ik_python or sys.executable).expanduser(),
        fps=int(info.get("fps") or 30),
    )
    if ik_evaluation["status"] != "skipped":
        checks.append({
            "name": "fr3_ik_reachability",
            "status": ik_evaluation["status"],
            "message": ik_evaluation["message"],
            "details": {"cubes": ik_evaluation["cubes"]},
        })

    valid_rows = max(0, total_rows - invalid_rows)

    overall = "pass"
    for check in checks:
        if check.get("status") == "fail":
            overall = "fail"
            break
        if check.get("status") == "warn" and overall == "pass":
            overall = "warn"

    valid_pct = (valid_rows / total_rows * 100.0) if total_rows else 0.0
    summary_pass = sum(1 for check in checks if check.get("status") == "pass")
    summary_warn = sum(1 for check in checks if check.get("status") == "warn")
    summary_fail = sum(1 for check in checks if check.get("status") == "fail")
    summary = f"{summary_pass} pass · {summary_warn} warn · {summary_fail} fail · {total_rows} frames"

    return {
        "status": overall,
        "summary": summary,
        "valid_frames_pct": round(valid_pct, 1),
        "checks": checks,
        "timestamp_sync": _timestamp_sync_summary(sync_report),
        "online_sync": online_sync_summary,
        "ik_evaluation": ik_evaluation,
        "completed_at": _now_iso(),
    }


def _named_indices(names: list[str], required: tuple[str, ...]) -> tuple[int, ...] | None:
    indices: list[int] = []
    for required_name in required:
        match = _first_named_index(names, (required_name,))
        if match is None:
            return None
        indices.append(match)
    return tuple(indices)


def _first_named_index(names: list[str], suffixes: tuple[str, ...]) -> int | None:
    lowered = [name.lower() for name in names]
    for index, name in enumerate(lowered):
        for suffix in suffixes:
            if name == suffix or name.endswith("." + suffix) or name.endswith("_" + suffix):
                return index
    return None


def _write_processing_meta_qc(dataset_root: Path, qc_result: dict[str, Any]) -> dict[str, Any]:
    existing = _load_processing_meta(dataset_root) or {}
    versions = existing.get("versions") if isinstance(existing.get("versions"), dict) else {}
    active_version = existing.get("active_version") if isinstance(existing.get("active_version"), str) else None
    if not active_version:
        active_version = "v1" if "v1" not in versions else f"v{len(versions) + 1}"
    version_entry = versions.get(active_version) if isinstance(versions.get(active_version), dict) else {}
    versions[active_version] = {
        "created_at": version_entry.get("created_at") or _now_iso(),
        "algorithm": version_entry.get("algorithm") or "identity-mvp",
        "qc": qc_result,
    }
    updated = {
        **existing,
        "active_version": active_version,
        "versions": versions,
        "current_job": {
            "id": f"qc-{int(time.time())}",
            "kind": "qc",
            "status": "complete",
            "completed_at": qc_result.get("completed_at"),
            "log_tail": [
                f"[qc] {check.get('name')}: {check.get('status')} - {check.get('message')}"
                for check in qc_result.get("checks", [])
            ][-12:],
        },
    }
    _write_processing_meta(dataset_root, updated)
    return updated


def _next_processing_version(versions: dict[str, Any]) -> str:
    index = 1
    while f"v{index}" in versions:
        index += 1
    return f"v{index}"


def _ee_trajectory_command(state: GatewayState, dataset_root: Path) -> list[str]:
    runner_path = state.repo_root / DEFAULT_EE_TRAJECTORY_RUNNER
    config_path = state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG
    if not runner_path.is_file():
        raise FileNotFoundError(f"EE trajectory runner not found: {runner_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"EE trajectory config not found: {config_path}")
    return [
        "bash",
        str(runner_path),
        "--dataset-root",
        str(dataset_root),
        "--config",
        str(config_path),
    ]


def _update_traj_gen_meta(
    dataset_root: Path,
    *,
    job_id: str,
    status: str,
    command: list[str] | None = None,
    message: str,
    log_tail: list[str] | None = None,
    version: str | None = None,
    exit_code: int | None = None,
) -> None:
    existing = _load_processing_meta(dataset_root) or {}
    current_job = existing.get("current_job") if isinstance(existing.get("current_job"), dict) else {}
    inherited_tail = [str(line) for line in current_job.get("log_tail", [])] if isinstance(current_job, dict) else []
    tail = inherited_tail if log_tail is None else [str(line) for line in log_tail]
    job = {
        "id": job_id,
        "kind": "traj-gen",
        "status": status,
        "message": message,
        "updated_at": _now_iso(),
        "log_tail": tail[-24:],
    }
    if command is not None:
        job["command"] = command
    elif isinstance(current_job, dict) and isinstance(current_job.get("command"), list):
        job["command"] = current_job["command"]
    if exit_code is not None:
        job["exit_code"] = exit_code
    if status == "running" and not job.get("started_at"):
        job["started_at"] = (current_job.get("started_at") if isinstance(current_job, dict) else None) or _now_iso()
    if status in ("complete", "failed", "error"):
        job["completed_at"] = _now_iso()

    updated = {**existing, "current_job": job}
    if version is not None:
        versions = updated.get("versions") if isinstance(updated.get("versions"), dict) else {}
        versions[version] = {
            "created_at": _now_iso(),
            "algorithm": DEFAULT_EE_TRAJECTORY_ALGORITHM,
            "dataset_root": str(dataset_root),
            "sidecar_dir": str(dataset_root / "derived" / DEFAULT_TRAJ_SIDECAR_NAME),
            "command": job.get("command") or command or [],
            "qc": versions.get(version, {}).get("qc") if isinstance(versions.get(version), dict) else None,
        }
        updated["active_version"] = version
        updated["versions"] = versions
    _write_processing_meta(dataset_root, updated)


def _start_traj_gen_output_reader(state: GatewayState, dataset_root: Path, process: subprocess.Popen[str], job_id: str) -> None:
    thread = Thread(
        target=_read_traj_gen_output,
        args=(state, dataset_root, process, job_id),
        daemon=True,
        name=f"ee-trajectory-output-{process.pid}",
    )
    thread.start()


def _read_traj_gen_output(
    state: GatewayState,
    dataset_root: Path,
    process: subprocess.Popen[str],
    job_id: str,
) -> None:
    log_tail: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            output = line.strip()
            if not output:
                continue
            log_tail = [*log_tail, output][-24:]
            with state.lock:
                still_current = state.processing_processes.get(str(dataset_root)) is process
            if not still_current:
                return
            _update_traj_gen_meta(
                dataset_root,
                job_id=job_id,
                status="running",
                message=output,
                log_tail=log_tail,
            )
            _refresh_cached_processing_item(state, dataset_root)
    exit_code = process.wait()
    with state.lock:
        still_current = state.processing_processes.get(str(dataset_root)) is process
        if still_current:
            state.processing_processes.pop(str(dataset_root), None)
    if not still_current:
        return

    existing = _load_processing_meta(dataset_root) or {}
    versions = existing.get("versions") if isinstance(existing.get("versions"), dict) else {}
    if exit_code == 0:
        version = _next_processing_version(versions)
        message = "EE trajectory generated from AprilTag cube tracking"
        _update_traj_gen_meta(
            dataset_root,
            job_id=job_id,
            status="complete",
            message=message,
            log_tail=[*log_tail, f"[traj-gen] complete exit_code={exit_code}"][-24:],
            version=version,
            exit_code=exit_code,
        )
        _refresh_cached_processing_item(state, dataset_root)
        with state.lock:
            state.log("info", f"Generated EE trajectory for {dataset_root.name} as {version}")
    else:
        message = f"EE trajectory generation failed with exit code {exit_code}"
        _update_traj_gen_meta(
            dataset_root,
            job_id=job_id,
            status="failed",
            message=message,
            log_tail=[*log_tail, f"[traj-gen] failed exit_code={exit_code}"][-24:],
            exit_code=exit_code,
        )
        _refresh_cached_processing_item(state, dataset_root)
        with state.lock:
            state.log("warn", f"{message}: {dataset_root}")


def _queue_traj_gen(state: GatewayState, dataset_root: Path) -> None:
    key = str(dataset_root)
    with state.lock:
        running = state.processing_processes.get(key)
        if key in state.processing_starting:
            state.log("info", f"EE trajectory generation already starting for {dataset_root.name}")
            return
        if running is not None and running.poll() is None:
            state.log("info", f"EE trajectory generation already running for {dataset_root.name}")
            return
        state.processing_processes.pop(key, None)
        state.processing_starting.add(key)

    job_id = f"traj-gen-{int(time.time())}"
    command: list[str] = []
    try:
        command = _ee_trajectory_command(state, dataset_root)
        _update_traj_gen_meta(
            dataset_root,
            job_id=job_id,
            status="running",
            command=command,
            message=f"Running AprilTag cube tracking for {dataset_root.name}",
            log_tail=[f"[traj-gen] {' '.join(command)}"],
        )
        _refresh_cached_processing_item(state, dataset_root)
        process = subprocess.Popen(
            command,
            cwd=state.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_tool_env(state.repo_root),
            start_new_session=True,
        )
    except OSError as exc:
        with state.lock:
            state.processing_starting.discard(key)
        _update_traj_gen_meta(
            dataset_root,
            job_id=job_id,
            status="failed",
            command=command,
            message=f"Failed to start EE trajectory generation: {exc}",
            log_tail=[f"[traj-gen] failed to start: {exc}"],
        )
        _refresh_cached_processing_item(state, dataset_root)
        raise
    except Exception:
        with state.lock:
            state.processing_starting.discard(key)
        raise
    with state.lock:
        state.processing_starting.discard(key)
        state.processing_processes[key] = process
        state.log("info", f"Started EE trajectory generation pid={process.pid} dataset={dataset_root}")
    _refresh_cached_processing_item(state, dataset_root)
    _start_traj_gen_output_reader(state, dataset_root, process, job_id)


def _gmsl2_episode_dirs(dataset_root: Path) -> list[Path]:
    eps_dir = dataset_root / "episodes"
    if not eps_dir.is_dir():
        return []
    return sorted(
        (d for d in eps_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")),
        key=lambda p: p.name,
    )


def _name_with_camera_count(name: str, camera_count: int) -> str:
    if camera_count <= 0:
        return name
    label = f"{camera_count}ch"
    if re.search(r"(?<![A-Za-z0-9])(?:\d+|[Nn])ch(?![A-Za-z0-9])", name):
        return re.sub(r"(?<![A-Za-z0-9])(?:\d+|[Nn])ch(?![A-Za-z0-9])", label, name, count=1)
    return name


def _gmsl2_camera_names(dataset_root: Path) -> list[str]:
    names: set[str] = set()
    for ep_dir in _gmsl2_episode_dirs(dataset_root):
        manifest_path = ep_dir / "online_sync_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            active = manifest.get("active_cameras")
            if isinstance(active, list):
                names.update(str(cam) for cam in active if str(cam).strip())
            counts = manifest.get("frame_count_by_camera")
            if isinstance(counts, dict):
                names.update(str(cam) for cam in counts if str(cam).strip())
        names.update(p.stem for p in ep_dir.glob("cam_*.mkv"))
    return sorted(names)


def _dataset_name_with_actual_camera_count(dataset_root: Path) -> str:
    return _name_with_camera_count(dataset_root.name, len(_gmsl2_camera_names(dataset_root)))


# Per-episode frame-count memo keyed by meta.json path -> (mtime, frames).
# An episode's meta.json is immutable once written, so completed sessions are
# never re-parsed; only the actively-recording episode's meta changes. This
# turns a full 600-episode rescan from "open+parse 600 JSON files" into "stat
# 600 files" (tens of ms), which keeps the background refresher cheap.
_GMSL2_EP_FRAMES_MEMO: dict[str, tuple[float, int]] = {}


def _gmsl2_episode_frame_count(ep_dir: Path) -> int:
    meta_path = ep_dir / "meta.json"
    try:
        mtime = meta_path.stat().st_mtime
    except OSError:
        return 0  # no meta.json yet (episode mid-write)
    key = str(meta_path)
    cached = _GMSL2_EP_FRAMES_MEMO.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with meta_path.open() as f:
            ep_meta = json.load(f)
        dur = float(ep_meta.get("duration_s") or 0)
        fps = int(ep_meta.get("video", {}).get("fps") or 60)
        frames = int(dur * fps)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        frames = 0
    _GMSL2_EP_FRAMES_MEMO[key] = (mtime, frames)
    return frames


def _gmsl2_dataset_stats(dataset_root: Path) -> tuple[int, int]:
    ep_dirs = _gmsl2_episode_dirs(dataset_root)
    return len(ep_dirs), sum(_gmsl2_episode_frame_count(ep_dir) for ep_dir in ep_dirs)


def _training_view_item_fields(dataset_root: Path, dataset_kind: str) -> dict[str, Any]:
    """Source dataset and action contract of a training view, for nesting it under its source.

    The manifest the builder writes is authoritative; the ``<dataset>__<contract>`` directory
    name is the fallback for views built before the manifest carried the action mode.
    """
    if dataset_kind != "training_view":
        return {}
    manifest = _load_json_file(dataset_root / "meta" / "il_view_manifest.json")
    source = str(manifest.get("source_dataset_root") or "")
    contract = str(manifest.get("action_mode") or "")
    if "__" in dataset_root.name:
        name_source, _, name_contract = dataset_root.name.partition("__")
    else:
        name_source, name_contract = dataset_root.name, ""
    return {
        "viewOf": source,
        "viewOfName": Path(source).name if source else name_source,
        "actionContract": contract or name_contract,
    }


def _recorded_dataset_items(state: GatewayState) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    candidates = _complete_replay_dataset_candidates(state)
    # "Latest" drives one-click actions on freshly captured data (Live Record -> Open in Replay),
    # so a derived training view must never claim it even when it is the newest directory.
    latest_recorded = next(
        (root for root in candidates if _dataset_kind(state, root) != "training_view"),
        None,
    )
    for dataset_root in candidates:
        info = _load_dataset_info(dataset_root)
        data_files = _dataset_data_files(dataset_root)
        modified_s = _dataset_modified_s(dataset_root)
        is_gmsl2 = _has_gmsl2_episodes(dataset_root)
        if is_gmsl2:
            total_episodes, total_frames = _gmsl2_dataset_stats(dataset_root)
        else:
            total_episodes = int(info.get("total_episodes") or 0)
            total_frames = int(info.get("total_frames") or 0)
        dataset_kind = _dataset_kind(state, dataset_root)
        items.append(
            {
                "path": str(dataset_root),
                "name": dataset_root.name,
                "datasetKind": dataset_kind,
                "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_s)) if modified_s else "",
                "updatedAtMs": int(modified_s * 1000),
                "totalEpisodes": total_episodes,
                "totalFrames": total_frames,
                # The rate these frames were captured at. Surfaced because the training view is
                # built by decimating it, and only integer ratios are possible -- the page needs
                # this to say that 60 -> 25 is impossible before the operator clicks, rather than
                # after the exporter has been started and refused.
                "fps": int(info.get("fps") or 0),
                "dataStatus": _recorded_dataset_status(dataset_root),
                "sourcePath": str(data_files[-1]) if data_files else "",
                "isLatest": latest_recorded is not None and dataset_root == latest_recorded,
                "excludedEpisodes": _annotation_excluded_episodes(dataset_root),
                "cameraFeatures": _camera_feature_items(info),
                **_training_view_item_fields(dataset_root, dataset_kind),
            }
        )
    return items


def _parquet_status_from_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "magic bytes" in message or "footer" in message:
        return "unfinalized"
    return "unreadable"


def _pose_xyz(pose: Any) -> tuple[float, float, float] | None:
    if not isinstance(pose, dict):
        return None
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    return x, y, z


def _dataset_has_replay_pose_hint(dataset_root: Path, info: dict[str, Any]) -> bool:
    return bool(
        _preferred_exported_ee_pose_cube(info)
        or _action_has_ee_pose(info)
        or _sidecar_cube_pose_files(dataset_root)
    )


def _trajectory_points_from_timeline(timeline: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    frames = timeline.get("frames") if isinstance(timeline, dict) else []
    frames = frames if isinstance(frames, list) else []
    state_names = timeline.get("stateNames") if isinstance(timeline, dict) else []
    state_names = [str(name) for name in state_names] if isinstance(state_names, list) else []
    fps = int(timeline.get("fps") or 30) if isinstance(timeline, dict) else 30

    raw_x: list[float] = []
    raw_y: list[float] = []
    raw_z: list[float] = []
    points: list[dict[str, Any]] = []
    used_pose = False
    previous_frame: int | None = None
    previous_timestamp: float | None = None

    for row_index, frame_row in enumerate(frames):
        if not isinstance(frame_row, dict):
            continue
        frame = int(frame_row.get("frame") if frame_row.get("frame") is not None else row_index)
        timestamp = _first_finite(frame_row.get("timestamp"), default=frame / max(fps, 1))
        state_values = _as_float_list(frame_row.get("state"))
        pose_xyz = _pose_xyz(frame_row.get("eePose"))
        if pose_xyz is not None:
            used_pose = True
            raw_x.append(pose_xyz[0])
            raw_y.append(pose_xyz[1])
            raw_z.append(pose_xyz[2])
        else:
            raw_x.append(float(row_index))
            raw_y.append(_gripper_width(state_values, state_names))
            raw_z.append(0.0)

        point: dict[str, Any] = {
            "frame": frame,
            "x": 0.0,
            "y": 0.0,
            "z": raw_z[-1],
            "gripperWidthMm": _gripper_width(state_values, state_names),
            "skewMs": 0.0,
        }
        if previous_frame is not None and previous_timestamp is not None:
            if frame - previous_frame > 1 or timestamp - previous_timestamp > 1.5 / max(fps, 1):
                point["event"] = "gap"
        previous_frame = frame
        previous_timestamp = timestamp
        points.append(point)

    if not points:
        return [], False

    normalized_x = _normalize_series(raw_x)
    normalized_y = _normalize_series(raw_y)
    normalized_z = _normalize_series(raw_z, low=0.0, high=100.0)
    for point, x_value, y_value, z_value in zip(points, normalized_x, normalized_y, normalized_z, strict=True):
        point["x"] = x_value
        point["y"] = 100.0 - y_value
        point["z"] = z_value
    return points, used_pose


def _read_recorded_trajectory(state: GatewayState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception:  # noqa: BLE001
        dataset_roots = _replay_dataset_candidates(state)
        latest = dataset_roots[0] if dataset_roots else None
        latest_info = _load_dataset_info(latest) if latest is not None else {}
        latest_meta = _dataset_replay_meta(state, latest, latest_info) if latest is not None else {}
        return [], {
            "datasetRoot": latest_meta.get("datasetRoot") or "",
            "datasetKind": latest_meta.get("datasetKind") or "recorded",
            "sourcePath": latest_meta.get("sourcePath") or "",
            "totalEpisodes": latest_meta.get("totalEpisodes") or 0,
            "recordedFrames": latest_meta.get("recordedFrames") or 0,
            "dataStatus": "unreadable",
            "message": "pyarrow is required to load recorded trajectory data",
            "diagnostics": ["Install pyarrow in the active Python environment."],
        }

    best_meta: dict[str, Any] | None = None
    for dataset_root in _replay_dataset_candidates(state):
        info = _load_dataset_info(dataset_root)
        dataset_meta = _dataset_replay_meta(state, dataset_root, info)
        if best_meta is None:
            best_meta = {**dataset_meta, "dataStatus": "missing", "message": "No recorded parquet files found"}

        if _has_gmsl2_episodes(dataset_root):
            timeline = _read_dataset_timeline(state, dataset_root, None)
            frames = timeline.get("frames") if isinstance(timeline, dict) else []
            frames = frames if isinstance(frames, list) else []
            if frames:
                points, used_pose = _trajectory_points_from_timeline(timeline)
                return points, {
                    **dataset_meta,
                    "sourcePath": str(timeline.get("sourcePath") or dataset_meta.get("sourcePath") or dataset_root),
                    "datasetRoot": str(dataset_root),
                    "episode": int(timeline.get("episode") or 0),
                    "frames": len(points),
                    "dataStatus": "loaded",
                    "trajectoryKind": "pose" if used_pose else "gripper_width",
                    "message": (
                        f"Loaded GMSL2 replay episode {int(timeline.get('episode') or 0)} from {dataset_root} ({len(points)} frames with EE pose)"
                        if used_pose
                        else f"Loaded GMSL2 replay episode {int(timeline.get('episode') or 0)} from {dataset_root} ({len(points)} frames)"
                    ),
                }

        data_files = _dataset_data_files(dataset_root)
        # Which file holds the episode that was asked for. A view merged from several
        # recordings gets one parquet per source, and this loop is otherwise newest-first:
        # the newest file answered every request with *its* own first episode, so on the
        # merged view picking episode 14 came back as episode 18 -- and a replay would have
        # driven the arm through episode 18. meta/episodes is v3's own record of where an
        # episode lives, so ask it, and only fall back to mtime order when it cannot say.
        requested_episode = _active_replay_episode(state, dataset_root)
        preferred_file = (
            _resolve_data_file_for_episode(dataset_root, info, requested_episode)
            if requested_episode is not None
            else None
        )
        ordered_files = sorted(data_files, key=lambda path: path.stat().st_mtime, reverse=True)
        if preferred_file is None:
            # No authoritative answer (a recording still in flight has no meta/episodes yet),
            # so leave the old newest-first behaviour alone rather than guessing.
            requested_episode = None
        else:
            ordered_files = [preferred_file, *(path for path in ordered_files if path != preferred_file)]

        for data_file in ordered_files:
            try:
                parquet = pq.ParquetFile(data_file)
                column_names = parquet.schema_arrow.names
                pose_columns = [
                    f"{prefix}.ee_pose.{cube_name}.base"
                    for prefix in ("observation", "action")
                    for cube_name in DEFAULT_CUBE_TRAJECTORY_NAMES
                ]
                wanted_columns = [
                    column
                    for column in (
                        "episode_index",
                        "frame_index",
                        "timestamp",
                        "action",
                        "observation.state",
                        "observation.device_capture_timestamp",
                        *pose_columns,
                    )
                    if column in column_names
                ]
                if not wanted_columns:
                    continue
                table = pq.read_table(data_file, columns=wanted_columns)
            except Exception as exc:  # noqa: BLE001
                status = _parquet_status_from_error(exc)
                if best_meta is None or best_meta.get("dataStatus") == "missing":
                    best_meta = {
                        **dataset_meta,
                        "sourcePath": str(data_file),
                        "dataStatus": status,
                        "message": (
                            "Recorded dataset exists, but its parquet file is missing the final footer. "
                            "This usually means the recorder did not call dataset.finalize()."
                            if status == "unfinalized"
                            else "Recorded dataset exists, but its parquet file could not be read."
                        ),
                        "diagnostics": [str(exc)],
                    }
                continue

            if table.num_rows == 0:
                if best_meta is None or best_meta.get("dataStatus") == "missing":
                    best_meta = {
                        **dataset_meta,
                        "sourcePath": str(data_file),
                        "dataStatus": "empty",
                        "message": "Recorded parquet file is empty",
                    }
                continue

            episode = 0
            if "episode_index" in table.column_names:
                episodes = [int(value) for value in table["episode_index"].to_pylist() if value is not None]
                if episodes:
                    episode_options = sorted(set(episodes))
                    if requested_episode is not None and requested_episode not in episode_options:
                        # Another source's parquet. Falling back to its first episode here is
                        # exactly what made a merged view report the wrong episode as loaded.
                        continue
                    episode = (
                        requested_episode
                        if requested_episode is not None
                        else _selected_episode_for_dataset(state, dataset_root, episode_options)
                    )
                    table = table.filter(pc.equal(table["episode_index"], episode))
                    if table.num_rows == 0:
                        continue

            rows = table.to_pylist()
            rows.sort(key=lambda row: int(row.get("frame_index") or 0))
            if not rows:
                continue

            state_names = _feature_names(info, "observation.state")
            action_names = _feature_names(info, "action")
            exported_ee_pose_cube = _preferred_exported_ee_pose_cube(info)
            raw_x: list[float] = []
            raw_y: list[float] = []
            raw_z: list[float] = []
            points: list[dict[str, Any]] = []
            used_pose = False

            for row_index, row in enumerate(rows):
                frame = int(row.get("frame_index") if row.get("frame_index") is not None else row_index)
                timestamp = _first_finite(row.get("timestamp"), default=frame / max(state.replay.fps, 1))
                state_values = _as_float_list(row.get("observation.state"))
                action_values = _as_float_list(row.get("action"))

                pose_xyz = _pose_xyz(_ee_pose_from_row(row, action_names, state_names, exported_pose_cube=exported_ee_pose_cube))
                vector_values = action_values if len(action_values) >= 3 else state_values
                vector_names = action_names if len(action_values) >= 3 else state_names
                x_index, y_index, z_index = _axis_indices(vector_names, vector_values)

                if pose_xyz is not None:
                    used_pose = True
                    raw_x.append(pose_xyz[0])
                    raw_y.append(pose_xyz[1])
                    raw_z.append(pose_xyz[2])
                elif x_index is not None and y_index is not None and x_index < len(vector_values) and y_index < len(vector_values):
                    used_pose = True
                    raw_x.append(vector_values[x_index])
                    raw_y.append(vector_values[y_index])
                    raw_z.append(vector_values[z_index] if z_index is not None and z_index < len(vector_values) else 0.0)
                else:
                    raw_x.append(float(row_index))
                    raw_y.append(_gripper_width(state_values, state_names))
                    raw_z.append(0.0)

                point: dict[str, Any] = {
                    "frame": frame,
                    "x": 0.0,
                    "y": 0.0,
                    "z": raw_z[-1],
                    "gripperWidthMm": _gripper_width(state_values, state_names),
                    "skewMs": _spread_ms(row.get("observation.device_capture_timestamp")),
                }

                if row_index > 0:
                    previous = rows[row_index - 1]
                    previous_frame = int(previous.get("frame_index") if previous.get("frame_index") is not None else row_index - 1)
                    previous_timestamp = _first_finite(previous.get("timestamp"), default=previous_frame / max(state.replay.fps, 1))
                    if frame - previous_frame > 1 or timestamp - previous_timestamp > 1.5 / max(state.replay.fps, 1):
                        point["event"] = "gap"
                if point["skewMs"] > 50.0:
                    point["event"] = "timeout"
                points.append(point)

            normalized_x = _normalize_series(raw_x)
            normalized_y = _normalize_series(raw_y)
            normalized_z = _normalize_series(raw_z, low=0.0, high=100.0)
            for point, x_value, y_value, z_value in zip(points, normalized_x, normalized_y, normalized_z, strict=True):
                point["x"] = x_value
                point["y"] = 100.0 - y_value
                point["z"] = z_value

            return points, {
                **dataset_meta,
                "sourcePath": str(data_file),
                "datasetRoot": str(dataset_root),
                "episode": episode,
                "frames": len(points),
                "dataStatus": "loaded",
                "trajectoryKind": "pose" if used_pose else "gripper_width",
                "message": (
                    f"Loaded recorded episode {episode} from {dataset_root} ({len(points)} frames)"
                    if used_pose
                    else f"Loaded recorded episode {episode}: frame timeline and gripper width ({len(points)} frames)"
                ),
            }

    return [], best_meta or {"dataStatus": "missing", "message": "No recorded dataset found", "diagnostics": []}


def _resolve_known_dataset(state: GatewayState, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    requested = _resolve_dataset_root(state.repo_root, raw_path)
    if requested is None:
        return None
    try:
        resolved = requested.resolve()
    except OSError:
        return None
    candidates = _complete_replay_dataset_candidates(state)
    for candidate in candidates:
        try:
            if candidate.resolve() == resolved:
                return candidate
        except OSError:
            continue
    return None


def _camera_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    if not isinstance(features, dict):
        return []
    keys: list[str] = []
    for name, feature in features.items():
        if isinstance(feature, dict) and feature.get("dtype") in ("video", "image"):
            keys.append(str(name))
    return keys


def _camera_feature_items(info: dict[str, Any]) -> list[dict[str, Any]]:
    features = info.get("features") or {}
    if not isinstance(features, dict):
        return []
    items: list[dict[str, Any]] = []
    for key in _camera_keys(info):
        feature = features.get(key) if isinstance(features.get(key), dict) else {}
        shape = feature.get("shape") if isinstance(feature.get("shape"), list) else []
        height = int(shape[0]) if len(shape) >= 1 and isinstance(shape[0], (int, float)) else 0
        width = int(shape[1]) if len(shape) >= 2 and isinstance(shape[1], (int, float)) else 0
        items.append({"key": key, "width": width, "height": height})
    return items


def _parse_training_view_fps(raw: str) -> int:
    """Validate the requested view rate before it becomes a command-line argument.

    Refused here rather than left to the exporter so the operator gets the answer in the
    click that asked for it, instead of in a build log minutes later.
    """
    if not raw.strip():
        return DEFAULT_TRAINING_VIEW_FPS
    try:
        fps = int(raw)
    except ValueError as exc:
        raise ValueError(f"view_fps must be an integer, got {raw!r}") from exc
    if fps not in TRAINING_VIEW_FPS_CHOICES:
        raise ValueError(
            f"view_fps must be one of {TRAINING_VIEW_FPS_CHOICES} (0 = keep the source rate), got {fps}"
        )
    return fps


def _parse_training_view_camera_crops(raw: str) -> dict[str, list[int]]:
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("camera_crops must be a JSON object keyed by camera feature") from exc
    if not isinstance(payload, dict):
        raise ValueError("camera_crops must be a JSON object keyed by camera feature")
    crops: dict[str, list[int]] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError(f"camera_crops[{key!r}] must be [x,y,w,h]")
        try:
            crops[str(key)] = [int(part) for part in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera_crops[{key!r}] must contain integers") from exc
    return crops


def _fixed_pose7_from_row(row: dict[str, Any], key: str) -> dict[str, float] | None:
    values = _as_float_list(row.get(key))
    if len(values) < 3:
        return None
    x, y, z = values[:3]
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    qx = values[3] if len(values) > 3 and math.isfinite(values[3]) else 0.0
    qy = values[4] if len(values) > 4 and math.isfinite(values[4]) else 0.0
    qz = values[5] if len(values) > 5 and math.isfinite(values[5]) else 0.0
    qw = values[6] if len(values) > 6 and math.isfinite(values[6]) else 1.0
    return {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


def _preferred_exported_ee_pose_cube(info: dict[str, Any]) -> str | None:
    features = info.get("features") or {}
    if not isinstance(features, dict):
        return None
    for cube_name in DEFAULT_CUBE_TRAJECTORY_NAMES:
        if (
            f"observation.ee_pose.{cube_name}.base" in features
            or f"action.ee_pose.{cube_name}.base" in features
        ):
            return cube_name
    return None


def _preferred_cube_pose_name(names: Iterable[str]) -> str | None:
    available = set(names)
    for cube_name in DEFAULT_CUBE_TRAJECTORY_NAMES:
        if cube_name in available:
            return cube_name
    return next(iter(available), None)


def _ee_pose_from_row(
    row: dict[str, Any],
    action_names: list[str],
    state_names: list[str],
    *,
    exported_pose_cube: str | None = None,
) -> dict[str, Any] | None:
    action_values = _as_float_list(row.get("action"))
    state_values = _as_float_list(row.get("observation.state"))

    # Exported datasets store EE pose as independent fixed-size pose columns.
    # Use one stable source for the whole timeline; per-frame fallback between
    # left/right/head mixes different tracked objects into one impossible path.
    pose = None
    if exported_pose_cube is not None:
        pose = _fixed_pose7_from_row(row, f"observation.ee_pose.{exported_pose_cube}.base")
        if pose is None:
            pose = _fixed_pose7_from_row(row, f"action.ee_pose.{exported_pose_cube}.base")

    # Legacy v3 datasets embed EE pose dimensions inside observation.state/action.
    if pose is None:
        pose = _extract_ee_axes(state_names, state_values) or _extract_ee_axes(action_names, action_values)
    if pose is None:
        return None

    # Gripper is reported independently — prefer state, fall back to action.
    gripper = _extract_gripper(state_names, state_values)
    if gripper is None:
        gripper = _extract_gripper(action_names, action_values)

    return {**pose, "gripper": gripper}


def _force_vector_from_state(state_names: list[str], state_values: list[float]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for axis in ("fx", "fy", "fz"):
        target = f"box_six_d_force.{axis}"
        try:
            index = state_names.index(target)
        except ValueError:
            return None
        if index >= len(state_values):
            return None
        value = float(state_values[index])
        values[axis] = value if math.isfinite(value) else 0.0
    magnitude = math.sqrt(values["fx"] ** 2 + values["fy"] ** 2 + values["fz"] ** 2)
    return {"x": values["fx"], "y": values["fy"], "z": values["fz"], "magnitude": magnitude}


def _ee_pose_from_cube_poses(
    cube_poses: dict[str, dict[str, Any]],
    cube_name: str | None,
) -> dict[str, Any] | None:
    return cube_poses.get(cube_name) if cube_name is not None else None


def _cube_pose_from_parquet_row(row: dict[str, Any], info: dict[str, Any], cube_name: str) -> dict[str, Any] | None:
    state_key = f"observation.state.{cube_name}"
    action_key = f"action.{cube_name}"
    state_values = _as_float_list(row.get(state_key))
    action_values = _as_float_list(row.get(action_key))
    state_names = _feature_names(info, state_key)
    action_names = _feature_names(info, action_key)
    pose = _extract_ee_axes(state_names, state_values) or _extract_ee_axes(action_names, action_values)
    if pose is None:
        return None
    return pose


def _csv_float(row: dict[str, Any], key: str) -> float | None:
    raw_value = row.get(key)
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pose_from_csv_row(row: dict[str, Any]) -> dict[str, float] | None:
    candidates = (
        ("cube_base_x_m", "cube_base_y_m", "cube_base_z_m", "cube_base_qx", "cube_base_qy", "cube_base_qz", "cube_base_qw"),
        ("state_x_m", "state_y_m", "state_z_m", "state_qx", "state_qy", "state_qz", "state_qw"),
        ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"),
        (
            "ee_est_base_x_m",
            "ee_est_base_y_m",
            "ee_est_base_z_m",
            "ee_est_base_qx",
            "ee_est_base_qy",
            "ee_est_base_qz",
            "ee_est_base_qw",
        ),
    )
    for x_key, y_key, z_key, qx_key, qy_key, qz_key, qw_key in candidates:
        x = _csv_float(row, x_key)
        y = _csv_float(row, y_key)
        z = _csv_float(row, z_key)
        if x is None or y is None or z is None:
            continue
        qx = _csv_float(row, qx_key)
        qy = _csv_float(row, qy_key)
        qz = _csv_float(row, qz_key)
        qw = _csv_float(row, qw_key)
        return {
            "x": x,
            "y": y,
            "z": z,
            "qx": qx if qx is not None else 0.0,
            "qy": qy if qy is not None else 0.0,
            "qz": qz if qz is not None else 0.0,
            "qw": qw if qw is not None else 1.0,
        }
    return None


def _sidecar_cube_pose_files(dataset_root: Path) -> dict[str, Path]:
    sidecar_dir = dataset_root / "derived" / DEFAULT_TRAJ_SIDECAR_NAME
    files: dict[str, Path] = {}
    for cube_name in DEFAULT_CUBE_TRAJECTORY_NAMES:
        candidate = sidecar_dir / f"state_action.{cube_name}.csv"
        if candidate.is_file():
            files[cube_name] = candidate
    for candidate in sidecar_dir.glob("state_action.*.csv"):
        name = candidate.name.removeprefix("state_action.").removesuffix(".csv")
        if name:
            files.setdefault(name, candidate)
    return files


def _read_sidecar_cube_poses(dataset_root: Path, episode: int) -> dict[str, dict[int, dict[str, float]]]:
    cube_files = _sidecar_cube_pose_files(dataset_root)
    cube_poses: dict[str, dict[int, dict[str, float]]] = {}
    for cube_name, csv_path in cube_files.items():
        poses_by_frame: dict[int, dict[str, float]] = {}
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    try:
                        row_episode = int(float(row.get("episode_index", "0") or 0))
                        frame_index = int(float(row.get("frame_index", "0") or 0))
                    except ValueError:
                        continue
                    if row_episode != episode:
                        continue
                    pose = _pose_from_csv_row(row)
                    if pose is not None:
                        poses_by_frame[frame_index] = pose
        except OSError:
            continue
        if poses_by_frame:
            cube_poses[cube_name] = poses_by_frame
    return cube_poses


def _tracking_run_dir(state: GatewayState, dataset_root: Path) -> Path:
    return state.repo_root / "outputs" / "tracking_analysis" / f"{dataset_root.name}{DEFAULT_TRACKING_RUN_SUFFIX}"


def _mat4_inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [[float(matrix[r][c]) for c in range(3)] for r in range(3)]
    translation = [float(matrix[r][3]) for r in range(3)]
    rotation_t = [[rotation[c][r] for c in range(3)] for r in range(3)]
    inv_translation = [-sum(rotation_t[r][c] * translation[c] for c in range(3)) for r in range(3)]
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inv_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inv_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inv_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(float(a[r][k]) * float(b[k][c]) for k in range(4)) for c in range(4)] for r in range(4)]


def _transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]] | None:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12 or not math.isfinite(norm):
        return None
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _pose_matrix_from_csv_row(row: dict[str, Any], prefix: str) -> list[list[float]] | None:
    x = _csv_float(row, f"{prefix}_x_m")
    y = _csv_float(row, f"{prefix}_y_m")
    z = _csv_float(row, f"{prefix}_z_m")
    qx = _csv_float(row, f"{prefix}_qx")
    qy = _csv_float(row, f"{prefix}_qy")
    qz = _csv_float(row, f"{prefix}_qz")
    qw = _csv_float(row, f"{prefix}_qw")
    if None in (x, y, z, qx, qy, qz, qw):
        return None
    rotation = _quat_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw))
    if rotation is None:
        return None
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(x)],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(y)],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _project_point(
    camera_matrix: list[list[float]],
    dist_coeffs: list[float],
    point_cam: tuple[float, float, float],
) -> list[float] | None:
    x, y, z = point_cam
    if z <= 1e-6 or not all(math.isfinite(value) for value in (x, y, z)):
        return None
    fx = float(camera_matrix[0][0])
    fy = float(camera_matrix[1][1])
    cx = float(camera_matrix[0][2])
    cy = float(camera_matrix[1][2])
    xn = x / z
    yn = y / z
    if dist_coeffs:
        coeffs = [float(value) for value in dist_coeffs]
        k1 = coeffs[0] if len(coeffs) > 0 else 0.0
        k2 = coeffs[1] if len(coeffs) > 1 else 0.0
        p1 = coeffs[2] if len(coeffs) > 2 else 0.0
        p2 = coeffs[3] if len(coeffs) > 3 else 0.0
        k3 = coeffs[4] if len(coeffs) > 4 else 0.0
        k4 = coeffs[5] if len(coeffs) > 5 else 0.0
        k5 = coeffs[6] if len(coeffs) > 6 else 0.0
        k6 = coeffs[7] if len(coeffs) > 7 else 0.0
        s1 = coeffs[8] if len(coeffs) > 8 else 0.0
        s2 = coeffs[9] if len(coeffs) > 9 else 0.0
        s3 = coeffs[10] if len(coeffs) > 10 else 0.0
        s4 = coeffs[11] if len(coeffs) > 11 else 0.0
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        if abs(denominator) > 1e-12:
            radial /= denominator
        x_tangential = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        y_tangential = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xn = xn * radial + x_tangential + s1 * r2 + s2 * r4
        yn = yn * radial + y_tangential + s3 * r2 + s4 * r4
    return [fx * xn + cx, fy * yn + cy]


def _load_camera_poses_in_base(summary_path: Path) -> dict[str, list[list[float]]]:
    summary = _load_json_file(summary_path)
    poses: dict[str, list[list[float]]] = {}
    joint_cameras = ((summary.get("joint_solution") or {}).get("cameras") or {}) if isinstance(summary, dict) else {}
    if isinstance(joint_cameras, dict):
        for camera_name, camera_info in joint_cameras.items():
            matrix = (((camera_info or {}).get("base_to_camera") or {}).get("matrix_4x4") or [])
            if isinstance(matrix, list) and len(matrix) == 4 and all(isinstance(row, list) and len(row) == 4 for row in matrix):
                poses[str(camera_name)] = [[float(value) for value in row] for row in matrix]
    if poses:
        return poses
    cameras = summary.get("cameras") or {}
    if isinstance(cameras, dict):
        for camera_name, camera_info in cameras.items():
            matrix = (((camera_info or {}).get("base_to_camera") or {}).get("matrix_4x4") or [])
            if isinstance(matrix, list) and len(matrix) == 4 and all(isinstance(row, list) and len(row) == 4 for row in matrix):
                poses[str(camera_name)] = [[float(value) for value in row] for row in matrix]
    return poses


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _flatten_float_list(values: Any) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, list):
        flattened: list[float] = []
        for item in values:
            flattened.extend(_flatten_float_list(item))
        return flattened
    return []


def _load_camera_intrinsics(path: Path) -> tuple[list[list[float]], list[float]] | None:
    payload = _load_json_file(path)
    matrix = payload.get("camera_matrix")
    if isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(row, list) and len(row) == 3 for row in matrix):
        try:
            return [[float(value) for value in row] for row in matrix], _flatten_float_list(payload.get("dist_coeffs"))
        except (TypeError, ValueError):
            return None
    return None


def _cube_corners(size_m: float) -> list[tuple[float, float, float]]:
    half = float(size_m) / 2.0
    return [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]


def _cube_overlay_from_row(
    row: dict[str, Any],
    *,
    camera_matrix: list[list[float]],
    dist_coeffs: list[float],
    t_cam_base: list[list[float]],
    cube_size_m: float,
) -> dict[str, Any] | None:
    if int(float(row.get("cube_detected", "0") or 0)) <= 0:
        return None
    t_base_cube = _pose_matrix_from_csv_row(row, "cube_base")
    if t_base_cube is None:
        return None
    t_cam_cube = _mat4_mul(t_cam_base, t_base_cube)
    corners = [_project_point(camera_matrix, dist_coeffs, _transform_point(t_cam_cube, point)) for point in _cube_corners(cube_size_m)]
    axis_points = [
        (0.0, 0.0, 0.0),
        (cube_size_m, 0.0, 0.0),
        (0.0, cube_size_m, 0.0),
        (0.0, 0.0, cube_size_m),
    ]
    axes = [_project_point(camera_matrix, dist_coeffs, _transform_point(t_cam_cube, point)) for point in axis_points]
    finite = [point for point in corners if point is not None]
    label = None
    if finite:
        label = [sum(point[0] for point in finite) / len(finite), sum(point[1] for point in finite) / len(finite)]
    elif axes[0] is not None:
        label = axes[0]
    cube_name = str(row.get("cube_name") or "cube")
    return {
        "cubeName": cube_name,
        "color": CUBE_OVERLAY_COLORS.get(cube_name, "#ffffff"),
        "corners": corners,
        "axes": {
            "origin": axes[0],
            "x": axes[1],
            "y": axes[2],
            "z": axes[3],
        },
        "label": label,
        "detected": int(float(row.get("cube_detected", "0") or 0)),
        "numMarkers": int(float(row.get("cube_num_markers", "0") or 0)),
        "rmsePx": _csv_float(row, "cube_reprojection_rmse_px"),
        "usedForFusion": int(float(row.get("used_for_fusion", "0") or 0)) > 0,
    }


def _read_video_cube_overlays(state: GatewayState, dataset_root: Path, episode: int) -> dict[int, dict[str, list[dict[str, Any]]]]:
    tracking_run = _tracking_run_dir(state, dataset_root)
    summary = _load_json_file(tracking_run / "summary.json")
    if not summary:
        return {}
    fixed_summary = Path(str(((summary.get("calibration_inputs") or {}).get("fixed_camera_summary") or "")))
    if not fixed_summary.is_absolute():
        fixed_summary = state.repo_root / fixed_summary
    camera_poses = _load_camera_poses_in_base(fixed_summary)
    cube_size_m = float(((summary.get("cube_tracker") or {}).get("cube_size_cm") or DEFAULT_CUBE_SIZE_M * 100.0)) / 100.0
    overlays: dict[int, dict[str, list[dict[str, Any]]]] = {}
    active_streams = summary.get("active_streams") if isinstance(summary.get("active_streams"), list) else []
    for stream in active_streams:
        if not isinstance(stream, dict):
            continue
        stream_key = str(stream.get("stream_key") or "")
        camera_name = str(stream.get("camera_name") or "")
        serial = str(stream.get("serial") or "")
        if not stream_key or not serial:
            continue
        intrinsics = _load_camera_intrinsics(Path(str(stream.get("intrinsics_path") or "")))
        t_base_cam = camera_poses.get(camera_name)
        if intrinsics is None or t_base_cam is None:
            continue
        camera_matrix, dist_coeffs = intrinsics
        t_cam_base = _mat4_inverse_rigid(t_base_cam)
        per_camera_csv = tracking_run / "per_camera" / f"camera_{serial}_records.csv"
        if not per_camera_csv.is_file():
            continue
        try:
            with per_camera_csv.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    try:
                        row_episode = int(float(row.get("episode_index", "-1") or -1))
                        frame_index = int(float(row.get("frame_index", "0") or 0))
                    except ValueError:
                        continue
                    if row_episode != int(episode):
                        continue
                    overlay = _cube_overlay_from_row(
                        row,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        t_cam_base=t_cam_base,
                        cube_size_m=cube_size_m,
                    )
                    if overlay is None:
                        continue
                    overlays.setdefault(frame_index, {}).setdefault(f"observation.images.{stream_key}", []).append(overlay)
        except OSError:
            continue
    return overlays


def _cube_names_for_timeline(dataset_root: Path, info: dict[str, Any]) -> list[str]:
    names = list(DEFAULT_CUBE_TRAJECTORY_NAMES)
    for name in _sidecar_cube_pose_files(dataset_root):
        if name not in names:
            names.append(name)
    features = info.get("features") or {}
    if isinstance(features, dict):
        for key in features:
            for prefix in ("observation.state.", "action."):
                if isinstance(key, str) and key.startswith(prefix):
                    cube_name = key[len(prefix) :]
                    if cube_name and cube_name not in names:
                        names.append(cube_name)
    return names


def _extract_ee_axes(names: list[str], values: list[float]) -> dict[str, float] | None:
    if not names or not values:
        return None
    lowered = [name.lower() for name in names]
    keys: dict[str, int | None] = {
        "x": None, "y": None, "z": None,
        "qx": None, "qy": None, "qz": None, "qw": None,
    }
    suffixes = {
        "x": ("ee.x", ".x", "_x_m"),
        "y": ("ee.y", ".y", "_y_m"),
        "z": ("ee.z", ".z", "_z_m"),
        "qx": ("ee.qx", ".qx", "_qx", "quat.x", "quat_x"),
        "qy": ("ee.qy", ".qy", "_qy", "quat.y", "quat_y"),
        "qz": ("ee.qz", ".qz", "_qz", "quat.z", "quat_z"),
        "qw": ("ee.qw", ".qw", "_qw", "quat.w", "quat_w"),
    }
    for index, name in enumerate(lowered):
        if index >= len(values):
            continue
        for key, candidates in suffixes.items():
            if keys[key] is None and any(name.endswith(suffix) or name == suffix.lstrip(".") for suffix in candidates):
                keys[key] = index
    if keys["x"] is None or keys["y"] is None or keys["z"] is None:
        return None
    return {
        "x": float(values[keys["x"]]),
        "y": float(values[keys["y"]]),
        "z": float(values[keys["z"]]),
        "qx": float(values[keys["qx"]]) if keys["qx"] is not None and keys["qx"] < len(values) else 0.0,
        "qy": float(values[keys["qy"]]) if keys["qy"] is not None and keys["qy"] < len(values) else 0.0,
        "qz": float(values[keys["qz"]]) if keys["qz"] is not None and keys["qz"] < len(values) else 0.0,
        "qw": float(values[keys["qw"]]) if keys["qw"] is not None and keys["qw"] < len(values) else 1.0,
    }


def _extract_gripper(names: list[str], values: list[float]) -> float | None:
    if not names or not values:
        return None
    for index, raw_name in enumerate(names):
        if index >= len(values):
            continue
        name = raw_name.lower()
        if name == "gripper.pos" or name.endswith(".gripper.pos") or name == "gripper_pos" or name.endswith("_gripper.pos") or name == "gripper":
            return float(values[index])
    return None


def _load_camera_controls(dataset_root: Path) -> dict[str, Any] | None:
    """Load an optional recorder camera-controls sidecar for the replay UI."""
    path = dataset_root / "meta" / "camera_controls.json"
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), dict):
        return None
    return payload


def _empty_timeline(
    dataset_root: Path,
    *,
    fps: int = 0,
    episode: int | None = None,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    camera_keys: list[str] | None = None,
    cube_pose_names: list[str] | None = None,
    error: str | None = None,
    dataset_kind: str = "recorded",
) -> dict[str, Any]:
    """Return a fully-shaped ReplayTimeline payload with no frames.

    Every ``/api/replay/timeline`` response must satisfy the ``ReplayTimeline``
    contract the frontend declares -- in particular ``frames`` must be a list,
    not absent -- otherwise ``ReplayInspector`` blows up when it dereferences
    ``timeline.frames[currentFrame]`` before its own early-return guard runs.
    """
    payload: dict[str, Any] = {
        "datasetRoot": str(dataset_root),
        "datasetKind": dataset_kind,
        "name": dataset_root.name,
        "episode": int(episode) if episode is not None else 0,
        "totalFrames": 0,
        "frames": [],
        "fps": int(fps or 0),
        "stateNames": list(state_names or []),
        "actionNames": list(action_names or []),
        "cameraKeys": list(camera_keys or []),
        "cubePoseNames": list(cube_pose_names or []),
        "videoTemplate": "",
        "videoChunkIndex": 0,
        "videoFileIndex": 0,
        "sourcePath": "",
        "videoWarmupS": 0.0,
        "cameraVideoOffsetsS": {},
        "cameraControls": _load_camera_controls(dataset_root),
    }
    if error:
        payload["error"] = error
    return payload


def _gmsl2_replay_warmup_s(ep_meta: dict[str, Any]) -> float:
    video = ep_meta.get("video") if isinstance(ep_meta.get("video"), dict) else {}
    return max(0.0, _first_finite([
        video.get("replay_warmup_s"),
        video.get("warmup_s"),
        ep_meta.get("replay_warmup_s"),
    ]))


def _load_episode_meta(ep_dir: Path) -> dict[str, Any]:
    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _box_snapshots_from_episode_meta(ep_meta: dict[str, Any]) -> list[dict[str, Any]]:
    box_meta = ep_meta.get("box_collection") if isinstance(ep_meta, dict) else None
    snapshots = box_meta.get("snapshots") if isinstance(box_meta, dict) else None
    if not isinstance(snapshots, list):
        return []
    return [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]


def _box_snapshot_rows_for_replay(
    ep_meta: dict[str, Any],
    *,
    fps: int,
    total_frames: int,
) -> tuple[list[str], list[str], dict[int, dict[str, list[float]]]] | None:
    snapshots = _box_snapshots_from_episode_meta(ep_meta)
    if not snapshots or total_frames <= 0:
        return None
    try:
        from tools.thor.gmsl2 import thor_lerobot_v3 as lr3
    except Exception:
        return None
    box_ids = lr3.box_ids_from_snapshots(snapshots)
    rows = lr3._build_episode_rows(
        fps=fps,
        episode_index=0,
        snapshots=snapshots,
        duration_s=total_frames / max(int(fps), 1),
        box_ids=box_ids,
    )
    if not rows:
        return None
    by_frame: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        frame_index = int(row.get("frame_index") or 0)
        by_frame[frame_index] = {
            "state": _as_float_list(row.get("observation.state")),
            "action": [],
        }
    if not any(any(abs(value) > 0.0 for value in entry["state"]) for entry in by_frame.values()):
        return None
    state_names = list(lr3.box_state_names(box_ids))
    return state_names, [], by_frame


def _rows_vector_all_zero(rows: list[dict[str, Any]], column: str) -> bool:
    saw_value = False
    for row in rows:
        values = _as_float_list(row.get(column))
        if values:
            saw_value = True
        if any(abs(value) > 0.0 for value in values):
            return False
    return saw_value


def _touch_payload_from_axes(
    fz_values: Any,
    *,
    fx_values: Any = None,
    fy_values: Any = None,
    timestamp: int = 0,
    expected_count: int | None = 239,
) -> dict[str, Any] | None:
    fz = _as_float_list(fz_values)
    if expected_count is not None:
        fz = fz[:expected_count]
        if len(fz) != expected_count:
            return None
    if not fz:
        return None

    count = len(fz)
    fx = _as_float_list(fx_values)[:count] if fx_values is not None else []
    fy = _as_float_list(fy_values)[:count] if fy_values is not None else []
    if fx_values is not None and len(fx) < count:
        fx = []
    if fy_values is not None and len(fy) < count:
        fy = []

    active_points = 0
    for index, z_value in enumerate(fz):
        x_value = fx[index] if fx else 0.0
        y_value = fy[index] if fy else 0.0
        if abs(x_value) > 0.0 or abs(y_value) > 0.0 or abs(z_value) > 0.0:
            active_points += 1

    payload: dict[str, Any] = {
        "timestamp": int(timestamp),
        "fz": fz,
        "maxFz": max(fz) if fz else 0.0,
        "activePoints": active_points,
    }
    if fx:
        payload["fx"] = fx
    if fy:
        payload["fy"] = fy
    return payload


def _touch_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    timestamp = int(_first_finite(data.get("timestamp"), default=0.0))
    return _touch_payload_from_axes(
        data.get("fz_0p1N"),
        fx_values=data.get("fx_0p1N"),
        fy_values=data.get("fy_0p1N"),
        timestamp=timestamp,
    )


_TOUCH_EXPORT_COLUMNS = {
    side: {
        axis: f"observation.touch.box_touch_{side}.{axis}_0p1N"
        for axis in ("fx", "fy", "fz")
    }
    for side in ("left", "right")
}


def _touch_payload_from_fz(values: Any, *, timestamp: int = 0) -> dict[str, Any] | None:
    fz = _as_float_list(values)[:239]
    if not fz:
        return None
    if len(fz) < 239:
        fz.extend([0.0] * (239 - len(fz)))
    return _touch_payload_from_axes(fz, timestamp=timestamp, expected_count=None)


def _touch_timestamp_from_parquet_row(row: dict[str, Any], info: dict[str, Any], sensor: str) -> int:
    timestamps = _as_float_list(row.get("box.timestamps"))
    if timestamps:
        names = _feature_names(info, "box.timestamps")
        wanted = f"{sensor}.timestamp"
        for index, name in enumerate(names):
            if index < len(timestamps) and (name == wanted or name.endswith(f".{wanted}")):
                return int(timestamps[index])
    return int(_first_finite(row.get("timestamp"), default=0.0) * 1_000_000)


def _touch_from_parquet_row(row: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    touch: dict[str, Any] = {}
    for side, columns in _TOUCH_EXPORT_COLUMNS.items():
        fz_column = columns["fz"]
        if fz_column not in row or row.get(fz_column) is None:
            continue
        sensor = "box_touch_left" if side == "left" else "box_touch_right"
        payload = _touch_payload_from_axes(
            row.get(fz_column),
            fx_values=row.get(columns["fx"]),
            fy_values=row.get(columns["fy"]),
            timestamp=_touch_timestamp_from_parquet_row(row, info, sensor),
            expected_count=None,
        )
        if payload is not None:
            touch[side] = payload
    return touch


def _touch_key_from_sid(sensor_id: str) -> str | None:
    suffix_by_sid = {"box_touch_left": "left", "box_touch_right": "right"}
    if sensor_id in suffix_by_sid:
        return suffix_by_sid[sensor_id]
    if "/" not in sensor_id:
        return None
    box_id, bare = sensor_id.split("/", 1)
    suffix = suffix_by_sid.get(bare)
    if not box_id or suffix is None:
        return None
    return f"{box_id}.{suffix}"


def _read_touch_samples_from_snapshots(ep_meta: dict[str, Any]) -> dict[str, list[tuple[float, dict[str, Any]]]]:
    samples: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for snapshot in _box_snapshots_from_episode_meta(ep_meta):
        sensors = snapshot.get("sensors")
        if not isinstance(sensors, dict):
            continue
        t_rel_s = _first_finite(snapshot.get("t_relative_s"), default=float("nan"))
        if t_rel_s != t_rel_s:
            continue
        for sid, data in sensors.items():
            key = _touch_key_from_sid(str(sid))
            if key is None:
                continue
            payload = _touch_payload(data)
            if payload is None:
                continue
            payload["tRelS"] = t_rel_s
            samples.setdefault(key, []).append((t_rel_s, payload))
    for key in samples:
        samples[key].sort(key=lambda item: item[0])
    return samples


def _read_touch_samples(ep_dir: Path) -> dict[str, list[tuple[float, dict[str, Any]]]]:
    samples: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    path = ep_dir / "box_sensors.jsonl"
    if path.is_file():
        try:
            with path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    key = _touch_key_from_sid(str(row.get("sid") or ""))
                    if key is None:
                        continue
                    t_rel_s = _first_finite(row.get("t_rel_s"), default=float("nan"))
                    if t_rel_s != t_rel_s:
                        continue
                    payload = _touch_payload(row.get("data"))
                    if payload is None:
                        continue
                    payload["tRelS"] = t_rel_s
                    samples.setdefault(key, []).append((t_rel_s, payload))
        except OSError:
            samples = {}
    if not samples:
        samples = _read_touch_samples_from_snapshots(_load_episode_meta(ep_dir))
    for key in samples:
        samples[key].sort(key=lambda item: item[0])
    return samples


def _nearest_touch_payload(
    samples: list[tuple[float, dict[str, Any]]],
    target_s: float,
    *,
    max_age_s: float = 0.25,
) -> dict[str, Any] | None:
    if not samples:
        return None
    times = [item[0] for item in samples]
    index = bisect.bisect_left(times, target_s)
    candidates: list[tuple[float, dict[str, Any]]] = []
    if index < len(samples):
        candidates.append(samples[index])
    if index > 0:
        candidates.append(samples[index - 1])
    if not candidates:
        return None
    sample_t, payload = min(candidates, key=lambda item: abs(item[0] - target_s))
    if abs(sample_t - target_s) > max_age_s:
        return None
    return payload


def _attach_touch_frames(frames: list[dict[str, Any]], ep_dir: Path, *, video_warmup_s: float = 0.0) -> None:
    samples = _read_touch_samples(ep_dir)
    if not any(samples.values()):
        return
    for frame in frames:
        target_s = _first_finite(frame.get("timestamp"), default=0.0) + max(0.0, video_warmup_s)
        touch: dict[str, Any] = {}
        for key, key_samples in samples.items():
            payload = _nearest_touch_payload(key_samples, target_s)
            if payload is not None:
                touch[key] = payload
        if touch:
            frame["touch"] = touch


def _gmsl2_pts_offset_s(ep_meta: dict[str, Any]) -> float:
    """Pipeline-start delay correction for the camera frame-time grid.

    ``pts_offset = mean(camera_first_wall_s[cam] - t0_wall_s)`` -- the same value
    the recorder bakes into the v3 parquet ``timestamp`` column (ts_sync.md
    §5.1), so EE-pose timestamps stay on the PWM-synced camera axis whether the
    replay timeline is read from parquet or from raw gmsl2 episodes. Returns 0.0
    when the ``sync_reference`` anchors are absent.
    """
    sync_reference = ep_meta.get("sync_reference") if isinstance(ep_meta, dict) else None
    if not isinstance(sync_reference, dict):
        return 0.0
    t0_wall_s = sync_reference.get("t0_wall_s")
    camera_first_wall_s = sync_reference.get("camera_first_wall_s")
    if not isinstance(t0_wall_s, (int, float)) or not isinstance(camera_first_wall_s, dict):
        return 0.0
    deltas = [
        float(wall_s) - float(t0_wall_s)
        for wall_s in camera_first_wall_s.values()
        if isinstance(wall_s, (int, float))
    ]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _camera_stem_from_key(camera_key: str) -> str:
    return camera_key.rsplit(".", 1)[-1]


def _gmsl2_camera_first_offsets_s(ep_meta: dict[str, Any]) -> dict[str, float]:
    sync_reference = ep_meta.get("sync_reference") if isinstance(ep_meta, dict) else None
    if not isinstance(sync_reference, dict):
        return {}
    t0_wall_s = sync_reference.get("t0_wall_s")
    camera_first_wall_s = sync_reference.get("camera_first_wall_s")
    if not isinstance(t0_wall_s, (int, float)) or not isinstance(camera_first_wall_s, dict):
        return {}
    offsets: dict[str, float] = {}
    for camera, wall_s in camera_first_wall_s.items():
        if not isinstance(wall_s, (int, float)):
            continue
        offset_s = float(wall_s) - float(t0_wall_s)
        if math.isfinite(offset_s):
            offsets[str(camera)] = offset_s
    return offsets


def _gmsl2_camera_video_offsets_s(ep_meta: dict[str, Any], camera_keys: list[str]) -> dict[str, float]:
    """Map replay camera keys to their file-local zero point in t0-relative time.

    Timeline timestamps are on the shared t0-relative axis, while browser
    ``video.currentTime`` is local to each remuxed camera file. For a camera
    whose first frame landed at ``camera_first_wall_s - t0_wall_s == offset``,
    the frontend must seek to ``timeline_timestamp - offset``.
    """
    raw_offsets = _gmsl2_camera_first_offsets_s(ep_meta)
    if not raw_offsets:
        return {}
    offsets: dict[str, float] = {}
    for key in camera_keys:
        offset = raw_offsets.get(key)
        if offset is None:
            offset = raw_offsets.get(_camera_stem_from_key(key))
        if offset is not None:
            offsets[key] = offset
    return offsets

def _read_gmsl2_timeline(dataset_root: Path, episode: int | None = None) -> dict[str, Any]:
    ep_dirs = _gmsl2_episode_dirs(dataset_root)
    if not ep_dirs:
        return _empty_timeline(dataset_root, error="no episodes found")
    ep_idx = episode if episode is not None else 0
    ep_dir = dataset_root / "episodes" / f"episode_{ep_idx:06d}"
    if not ep_dir.is_dir():
        return _empty_timeline(dataset_root, episode=ep_idx, error=f"episode_{ep_idx:06d} not found")
    meta_path = ep_dir / "meta.json"
    ep_meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            with meta_path.open() as f:
                ep_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    fps = int(ep_meta.get("video", {}).get("fps") or ep_meta.get("fps") or 60)
    duration_s = float(ep_meta.get("duration_s") or 10)
    # PR3+ EPISODE mkv files come from splitmuxsink split-now boundaries, so
    # they no longer contain the pre-episode warmup frames the
    # `replay_warmup_s` field was originally designed to skip. Frame 0 in
    # the timeline now corresponds to video.currentTime == 0 directly. If
    # we kept subtracting warmup here totalFrames would lose
    # replay_warmup_s * fps frames AND the frontend would offset by the
    # same amount when reverse-computing frame from currentTime — the
    # slider used to freeze ~1s short of the end because both layers
    # double-counted the same trim. Force 0 so the math collapses cleanly.
    video_warmup_s = 0.0
    total_frames = max(0, int(duration_s * fps))
    mkv_files = sorted(ep_dir.glob("*.mkv"))
    camera_keys = [f.stem for f in mkv_files if f.stat().st_size > 1024]

    # PWM-synced camera frame-time grid (ts_sync.md §5.1): frame N time =
    # pts_offset + N/fps in the t0-relative domain. pts_offset corrects the
    # pipeline-start delay and matches the v3 parquet `timestamp` column, so
    # EE-pose timestamps agree whether the timeline is read from parquet or here.
    pts_offset_s = _gmsl2_pts_offset_s(ep_meta)

    # AprilTag EE-pose sidecar (derived/april_cube_tracking_in_robot_base) is
    # produced straight from the raw episodes by run_april_cube_tracking_* and
    # needs no v3 parquet. Surface it here too -- otherwise camera-only datasets
    # (no BOX parquet -> this gmsl2 path) would generate an EE pose that never
    # reaches the replay view.
    sidecar_cube_poses = _read_sidecar_cube_poses(dataset_root, ep_idx)
    cube_pose_names = [n for n in DEFAULT_CUBE_TRAJECTORY_NAMES if n in sidecar_cube_poses]
    cube_pose_names += [n for n in sidecar_cube_poses if n not in cube_pose_names]
    ee_pose_cube_name = _preferred_cube_pose_name(cube_pose_names)
    box_fallback = _box_snapshot_rows_for_replay(ep_meta, fps=fps, total_frames=total_frames)
    state_names = box_fallback[0] if box_fallback else []
    action_names = box_fallback[1] if box_fallback else []
    box_rows = box_fallback[2] if box_fallback else {}

    frames: list[dict[str, Any]] = []
    for i in range(total_frames):
        cube_poses: dict[str, dict[str, Any]] = {}
        for cube_name in cube_pose_names:
            cube_pose = sidecar_cube_poses.get(cube_name, {}).get(i)
            if cube_pose is not None:
                cube_poses[cube_name] = cube_pose
        state_values = list(box_rows.get(i, {}).get("state", []))
        action_values = list(box_rows.get(i, {}).get("action", []))
        frame = {
            "frame": i,
            "timestamp": pts_offset_s + i / max(fps, 1),
            "state": state_values,
            "action": action_values,
            "eePose": _ee_pose_from_cube_poses(cube_poses, ee_pose_cube_name) or {},
            "cubePoses": cube_poses,
        }
        force_vector = _force_vector_from_state(state_names, state_values)
        if force_vector is not None:
            frame["forceVector"] = force_vector
        frames.append(frame)
    _attach_touch_frames(frames, ep_dir, video_warmup_s=video_warmup_s)
    return {
        "datasetRoot": str(dataset_root),
        "datasetKind": "recorded",
        "name": dataset_root.name,
        "episode": ep_idx,
        "totalFrames": total_frames,
        "fps": fps,
        "stateNames": state_names,
        "actionNames": action_names,
        "cubePoseNames": [n for n in cube_pose_names if any(n in f.get("cubePoses", {}) for f in frames)],
        "cameraKeys": camera_keys,
        "videoTemplate": "",
        "videoChunkIndex": 0,
        "videoFileIndex": 0,
        "frames": frames,
        "sourcePath": str(ep_dir),
        "videoWarmupS": video_warmup_s,
        "cameraVideoOffsetsS": _gmsl2_camera_video_offsets_s(ep_meta, camera_keys),
        "cameraControls": _load_camera_controls(dataset_root),
    }


def _read_dataset_timeline(state: GatewayState, dataset_root: Path, episode: int | None = None) -> dict[str, Any]:
    if _has_gmsl2_episodes(dataset_root) and not _has_lerobot_v3_data(dataset_root):
        return _read_gmsl2_timeline(dataset_root, episode)
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        return _empty_timeline(dataset_root, error=f"pyarrow unavailable: {exc}", dataset_kind=_dataset_kind(state, dataset_root))

    info = _load_dataset_info(dataset_root)
    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    cube_pose_names = _cube_names_for_timeline(dataset_root, info)
    camera_keys = _camera_keys(info)
    if _has_gmsl2_episodes(dataset_root) and not camera_keys:
        ep_idx = episode if episode is not None else int(state.replay.episode or 0)
        ep_dir = dataset_root / "episodes" / f"episode_{ep_idx:06d}"
        camera_keys = [f.stem for f in sorted(ep_dir.glob("*.mkv")) if f.stat().st_size > 1024]
    fps = int(info.get("fps") or state.replay.fps or 30)
    data_files = _dataset_data_files(dataset_root)
    if not data_files:
        return _empty_timeline(
            dataset_root,
            fps=fps,
            state_names=state_names,
            action_names=action_names,
            camera_keys=camera_keys,
            cube_pose_names=cube_pose_names,
            error="no parquet files",
            dataset_kind=_dataset_kind(state, dataset_root),
        )

    episode_options = _dataset_episode_indices(dataset_root, info)
    selected_episode = episode if episode is not None else _selected_episode_for_dataset(state, dataset_root, episode_options)
    if episode_options and selected_episode not in episode_options:
        return _empty_timeline(
            dataset_root,
            fps=fps,
            episode=selected_episode,
            state_names=state_names,
            action_names=action_names,
            camera_keys=camera_keys,
            cube_pose_names=cube_pose_names,
            error=f"episode {selected_episode} not found",
            dataset_kind=_dataset_kind(state, dataset_root),
        )

    data_file = _resolve_data_file_for_episode(dataset_root, info, selected_episode) or data_files[-1]
    table = pq.read_table(data_file)
    if "episode_index" in table.column_names:
        table = table.filter(pc.equal(table["episode_index"], selected_episode))
        if table.num_rows == 0:
            return _empty_timeline(
                dataset_root,
                fps=fps,
                episode=selected_episode,
                state_names=state_names,
                action_names=action_names,
                camera_keys=camera_keys,
                cube_pose_names=cube_pose_names,
                error=f"episode {selected_episode} not found in {data_file.name}",
                dataset_kind=_dataset_kind(state, dataset_root),
            )
        episode = selected_episode
    else:
        episode = 0

    video_warmup_s = 0.0
    # GMSL2 splitmux episode files are already cut at the playable fragment
    # boundary. The legacy replay_warmup_s field should not be applied to
    # LeRobot v3 timelines; doing so maps a 10s/603-frame episode to only
    # about 9s/540 displayed frames.

    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index") or 0))
    sidecar_cube_poses = _read_sidecar_cube_poses(dataset_root, int(episode or 0))
    video_cube_overlays = _read_video_cube_overlays(state, dataset_root, int(episode or 0))
    for cube_name in sidecar_cube_poses:
        if cube_name not in cube_pose_names:
            cube_pose_names.append(cube_name)
    exported_ee_pose_cube = _preferred_exported_ee_pose_cube(info)
    sidecar_ee_pose_cube = _preferred_cube_pose_name(cube_pose_names)

    ep_dir: Path | None = None
    ep_meta: dict[str, Any] = {}
    box_fallback: tuple[list[str], list[str], dict[int, dict[str, list[float]]]] | None = None
    if _has_gmsl2_episodes(dataset_root):
        ep_dir = dataset_root / "episodes" / f"episode_{int(episode or 0):06d}"
        ep_meta = _load_episode_meta(ep_dir) if ep_dir.is_dir() else {}
        if ep_dir.is_dir() and (not state_names or _rows_vector_all_zero(rows, "observation.state")):
            box_fallback = _box_snapshot_rows_for_replay(
                ep_meta,
                fps=fps,
                total_frames=len(rows),
            )
            if box_fallback is not None:
                state_names = box_fallback[0]
                action_names = box_fallback[1]
    box_rows = box_fallback[2] if box_fallback is not None else {}

    frames: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        frame_index = int(row.get("frame_index") if row.get("frame_index") is not None else row_index)
        timestamp = _first_finite(row.get("timestamp"), default=frame_index / max(fps, 1))
        fallback_box = box_rows.get(frame_index)
        state_values = list(fallback_box.get("state", [])) if fallback_box else _as_float_list(row.get("observation.state"))
        action_values = list(fallback_box.get("action", [])) if fallback_box else _as_float_list(row.get("action"))
        pose = _ee_pose_from_row(row, action_names, state_names, exported_pose_cube=exported_ee_pose_cube) or {}
        cube_poses: dict[str, dict[str, Any]] = {}
        for cube_name in cube_pose_names:
            cube_pose = sidecar_cube_poses.get(cube_name, {}).get(frame_index)
            if cube_pose is None:
                cube_pose = _cube_pose_from_parquet_row(row, info, cube_name)
            if cube_pose is not None:
                cube_poses[cube_name] = cube_pose
        if not pose:
            pose = _ee_pose_from_cube_poses(cube_poses, sidecar_ee_pose_cube) or {}
        frame = {
            "frame": frame_index,
            "timestamp": timestamp,
            "state": state_values,
            "action": action_values,
            "eePose": pose,
            "cubePoses": cube_poses,
            "videoOverlays": video_cube_overlays.get(frame_index, {}),
        }
        force_vector = _force_vector_from_state(state_names, state_values)
        if force_vector is not None:
            frame["forceVector"] = force_vector
        touch = _touch_from_parquet_row(row, info)
        if touch:
            frame["touch"] = touch
        frames.append(frame)

    if ep_dir is not None and ep_dir.is_dir():
        _attach_touch_frames(frames, ep_dir, video_warmup_s=video_warmup_s)

    video_template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    return {
        "datasetRoot": str(dataset_root),
        "datasetKind": _dataset_kind(state, dataset_root),
        "name": dataset_root.name,
        "episode": episode,
        "totalFrames": len(frames),
        "fps": fps,
        "stateNames": state_names,
        "actionNames": action_names,
        "cubePoseNames": [name for name in cube_pose_names if any(name in frame.get("cubePoses", {}) for frame in frames)],
        "cameraKeys": camera_keys,
        "videoTemplate": video_template,
        "videoChunkIndex": 0,
        "videoFileIndex": 0,
        "frames": frames,
        "sourcePath": str(data_file),
        "videoWarmupS": video_warmup_s,
        "cameraVideoOffsetsS": _gmsl2_camera_video_offsets_s(ep_meta, camera_keys),
        "cameraControls": _load_camera_controls(dataset_root),
    }


def _probe_video_duration_s(video_path: Path, *, timeout_s: float = 5.0) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _cached_mp4_is_usable(mp4_path: Path, mkv_path: Path, expected_duration_s: float | None = None) -> bool:
    try:
        mp4_stat = mp4_path.stat()
        mkv_stat = mkv_path.stat()
    except OSError:
        return False
    if mp4_stat.st_size <= 4096 or mp4_stat.st_mtime < mkv_stat.st_mtime:
        return False

    if expected_duration_s is not None and expected_duration_s > 0:
        duration_s = _probe_video_duration_s(mp4_path)
        if duration_s is not None:
            min_duration_s = max(0.5, min(expected_duration_s * 0.75, expected_duration_s - 0.5))
            return duration_s >= min_duration_s
        # If ffprobe is unavailable, reject obviously truncated cache files.
        if mkv_stat.st_size > 0 and mp4_stat.st_size < mkv_stat.st_size * 0.12:
            return False

    return True


_REPLAY_REMUX_SEMAPHORE = BoundedSemaphore(2)


def _remux_mkv_to_mp4(mkv_path: Path, expected_duration_s: float | None = None) -> Path | None:
    mp4_path = mkv_path.with_suffix(".mp4")
    if mp4_path.is_file() and _cached_mp4_is_usable(mp4_path, mkv_path, expected_duration_s):
        return mp4_path
    if mp4_path.exists():
        try:
            mp4_path.unlink()
        except OSError:
            return None

    tmp_path = mp4_path.with_name(f".{mp4_path.stem}.{os.getpid()}.tmp.mp4")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            return None

    if shutil.which("ffmpeg") is None:
        return None

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(mkv_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    timeout_s = 60.0
    if expected_duration_s is not None and expected_duration_s > 0:
        timeout_s = max(60.0, min(300.0, expected_duration_s * 12.0))
    try:
        with _REPLAY_REMUX_SEMAPHORE:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except (subprocess.TimeoutExpired, OSError):
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return None
    if result.returncode != 0 or not _cached_mp4_is_usable(tmp_path, mkv_path, expected_duration_s):
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return None
    try:
        tmp_path.replace(mp4_path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return None
    return mp4_path


def _resolve_video_path(
    state: GatewayState,
    dataset_root: Path,
    camera_key: str,
    episode: int | None = None,
) -> Path | None:
    selected_episode = int(state.replay.episode if episode is None else episode)
    if _has_gmsl2_episodes(dataset_root):
        ep_dir = dataset_root / "episodes" / f"episode_{selected_episode:06d}"
        mkv = ep_dir / f"{_camera_stem_from_key(camera_key)}.mkv"
        if not mkv.is_file():
            return None
        expected_duration_s = None
        meta_path = ep_dir / "meta.json"
        if meta_path.is_file():
            try:
                with meta_path.open() as f:
                    expected_duration_s = _first_finite(json.load(f).get("duration_s"), default=0.0)
            except (OSError, json.JSONDecodeError):
                expected_duration_s = None
        return _remux_mkv_to_mp4(mkv, expected_duration_s=expected_duration_s)

    info = _load_dataset_info(dataset_root)
    episode_video = _resolve_video_file_for_episode(dataset_root, info, camera_key, selected_episode)
    if episode_video is not None:
        return episode_video

    template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    candidate = _dataset_child_from_template(
        dataset_root,
        template,
        video_key=camera_key,
        chunk_index=0,
        file_index=selected_episode,
    )
    if candidate is not None and candidate.is_file():
        return candidate

    camera_dir = dataset_root / "videos" / camera_key
    if camera_dir.is_dir():
        for mp4 in sorted(camera_dir.glob("chunk-*/*.mp4")):
            return mp4
    return None


# ---------------------------------------------------------------- frame preview ---
#
# One still frame out of a recording, for the crop picker. A v3 dataset concatenates every
# episode of a chunk into a single mp4 -- 16 FR3 episodes is a 42 MB file -- so pointing the
# browser at the video URL and letting it seek costs tens of megabytes per camera before the
# operator can draw a box. Decoding the one frame here costs ~30 KB.

_FRAME_PREVIEW_SEMAPHORE = BoundedSemaphore(2)
_FRAME_PREVIEW_CACHE: dict[tuple[str, int, int, str], bytes] = {}
_FRAME_PREVIEW_CACHE_LIMIT = 48
_FRAME_PREVIEW_CACHE_LOCK = Lock()


def _episode_video_window(dataset_root: Path, camera_key: str, episode: int) -> tuple[float, int]:
    """Where `episode` starts inside its (possibly shared) video file, and how long it is.

    v3 concatenates episodes into one file per chunk and records the boundary per camera in
    meta/episodes. Without it every episode of a chunk previews as the same first frame.
    """
    row = _episode_metadata_row(dataset_root, episode)
    if row is None:
        return 0.0, 0
    try:
        start_s = max(0.0, float(row[f"videos/{camera_key}/from_timestamp"]))
    except (KeyError, TypeError, ValueError):
        start_s = 0.0
    try:
        frames = max(0, int(row["length"]))
    except (KeyError, TypeError, ValueError):
        frames = 0
    return start_s, frames


def _frame_preview_source(
    state: GatewayState, dataset_root: Path, camera_key: str, episode: int
) -> tuple[Path, float, int] | None:
    """Video file to decode, the episode's offset inside it, and its frame count."""
    if _has_gmsl2_episodes(dataset_root):
        ep_dir = dataset_root / "episodes" / f"episode_{episode:06d}"
        mkv = ep_dir / f"{_camera_stem_from_key(camera_key)}.mkv"
        if not mkv.is_file():
            return None
        # Decoded straight from the mkv: _resolve_video_path would remux the whole episode to
        # mp4 first, which is minutes of ffmpeg for a single still.
        return mkv, 0.0, _gmsl2_episode_frame_count(ep_dir)
    video_path = _resolve_video_path(state, dataset_root, camera_key, episode)
    if video_path is None:
        return None
    start_s, frames = _episode_video_window(dataset_root, camera_key, episode)
    return video_path, start_s, frames


def _frame_preview_episode_lengths(dataset_root: Path) -> dict[int, int]:
    """episode_index -> frame count, read from meta/episodes in one pass."""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}
    lengths: dict[int, int] = {}
    for meta_file in sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet")):
        try:
            table = pq.read_table(meta_file, columns=["episode_index", "length"])
        except Exception:
            continue
        for row in table.to_pylist():
            try:
                lengths[int(row["episode_index"])] = int(row["length"])
            except (KeyError, TypeError, ValueError):
                continue
    return lengths


def _frame_preview_info(dataset_root: Path) -> dict[str, Any]:
    """What the crop picker needs to address a frame: cameras, rate, and episode lengths."""
    info = _load_dataset_info(dataset_root)
    gmsl2 = _has_gmsl2_episodes(dataset_root)
    lengths = {} if gmsl2 else _frame_preview_episode_lengths(dataset_root)
    episodes: list[dict[str, int]] = []
    for episode in _dataset_episode_indices(dataset_root, info):
        frames = (
            _gmsl2_episode_frame_count(dataset_root / "episodes" / f"episode_{episode:06d}")
            if gmsl2
            else int(lengths.get(episode, 0))
        )
        episodes.append({"episode": int(episode), "frames": frames})
    return {
        "path": str(dataset_root),
        "name": dataset_root.name,
        "fps": int(info.get("fps") or 0),
        "cameras": _camera_feature_items(info),
        "episodes": episodes,
    }


def _frame_preview_timestamp(start_s: float, frame_index: int, frames: int, fps: int) -> tuple[int, float]:
    """Clamp a requested frame to the episode and turn it into a seek timestamp.

    Clamped rather than refused: the slider's range comes from a `frame-info` response that can
    be one poll older than the recording it describes, and a stale last frame should still show
    the last frame.
    """
    clamped = max(0, frame_index if frames <= 0 else min(frame_index, frames - 1))
    return clamped, start_s + (clamped / fps if fps > 0 else 0.0)


def _extract_video_frame_jpeg(video_path: Path, timestamp_s: float) -> bytes | None:
    if shutil.which("ffmpeg") is None:
        return None
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # -ss ahead of -i seeks by index to the keyframe before the target and decodes forward
        # from there. After -i it decodes from the start of the file instead, which on a
        # five-minute chunk is seconds of work for every step of the frame slider.
        "-ss", f"{max(0.0, timestamp_s):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "3",
        "-",
    ]
    try:
        with _FRAME_PREVIEW_SEMAPHORE:
            result = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _frame_preview_jpeg(video_path: Path, timestamp_s: float) -> bytes | None:
    """Cached `_extract_video_frame_jpeg`. Dragging the frame slider revisits the same handful
    of timestamps, and every miss is another ffmpeg process."""
    try:
        stat = video_path.stat()
    except OSError:
        return None
    key = (str(video_path), int(stat.st_mtime_ns), int(stat.st_size), f"{timestamp_s:.3f}")
    with _FRAME_PREVIEW_CACHE_LOCK:
        cached = _FRAME_PREVIEW_CACHE.get(key)
    if cached is not None:
        return cached
    frame = _extract_video_frame_jpeg(video_path, timestamp_s)
    if frame is None:
        return None
    with _FRAME_PREVIEW_CACHE_LOCK:
        _FRAME_PREVIEW_CACHE[key] = frame
        while len(_FRAME_PREVIEW_CACHE) > _FRAME_PREVIEW_CACHE_LIMIT:
            _FRAME_PREVIEW_CACHE.pop(next(iter(_FRAME_PREVIEW_CACHE)))
    return frame


def _serve_static_file(handler: BaseHTTPRequestHandler, asset_path: Path, content_type: str) -> None:
    file_size = asset_path.stat().st_size
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(file_size))
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    with asset_path.open("rb") as asset_file:
        while True:
            chunk = asset_file.read(64 * 1024)
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return


def _terminate_preview_proc(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=1.5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def _stop_all_camera_previews(state: GatewayState) -> None:
    with state.camera_preview_lock:
        procs = list(state.camera_preview_processes.values())
        state.camera_preview_processes.clear()
    for proc in procs:
        _terminate_preview_proc(proc)


@contextmanager
def _previews_suspended_for_connect(state: GatewayState):
    """Hold ``camera_preview_suspended`` across the Connect preflight + spawn.

    Setting the flag makes the snapshot route reject preview polls (409) so a
    concurrent poll can't respawn a Device Manager preview pipeline in the gap
    before the recorder's ``nvarguscamerasrc`` takes the sensors. The previous
    code set the flag inline before the request's ``try`` and relied on four
    scattered reset sites (``_snapshot`` on recorder exit, ``_stop_recorder``,
    and the connect ``except``); any new early-return in Connect that forgot to
    reset would wedge previews at 409 forever.

    Lifecycle here is total: on **any** exception before the recorder is
    running the flag is reset so the operator can keep inspecting the grid; on
    success it stays set (the recorder now owns the cameras) and is later
    cleared by ``_stop_recorder`` / ``_snapshot`` when the recorder exits.
    """
    with state.lock:
        state.camera_preview_suspended = True
    ok = False
    try:
        yield
        ok = True
    finally:
        if not ok:
            with state.lock:
                state.camera_preview_suspended = False


def _camera_preview_stagger_s(state: GatewayState) -> float:
    """Gap between consecutive preview Argus opens.

    Honors a dedicated ``cameras.preview_spawn_stagger_s`` knob; otherwise falls
    back to the recording ``spawn_stagger_s`` capped at 0.5s so the grid still
    fills quickly while keeping concurrent Argus opens serialized.
    """
    cams = state.config.get("cameras") if isinstance(state.config.get("cameras"), dict) else {}
    raw = cams.get("preview_spawn_stagger_s")
    if raw is None:
        try:
            base = float(cams.get("spawn_stagger_s", 0.5))
        except (TypeError, ValueError):
            base = 0.5
        return max(0.0, min(base, 0.5))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.5


# A preview pipeline is reaped once no tile has polled its snapshot for this
# long. The frontend polls every ~250ms, so 5s tolerates a few dropped polls
# / a page that is briefly backgrounded without leaking Argus sessions.
_PREVIEW_IDLE_TTL_S = 5.0
# How long a snapshot request waits for the first frame after a cold spawn
# (Argus open + ISP/AWB settle) before giving up with 503.
_PREVIEW_FIRST_FRAME_TIMEOUT_S = 6.0
# Recorder-owned preview files are produced at ~5 fps; tolerate short pauses
# without showing a stale frame as live.
_RECORDER_PREVIEW_STALE_S = 2.0
_RECORDER_PREVIEW_DIR = Path("/dev/shm/lerobot_preview")
# Debounce interval for the viewer-demand heartbeat sent to the recorder while
# the Device Manager grid polls camera.jpg. Must stay well under the recorder's
# recording_preview_idle_ttl_s (default 6s) so a steadily-polling grid never
# lets previews lapse.
_RECORDER_PREVIEW_DEMAND_INTERVAL_S = 1.0


def _state_is_gmsl2(state: GatewayState) -> bool:
    recorder = state.config.get("recorder") if isinstance(state.config.get("recorder"), dict) else {}
    sensors = state.config.get("sensors") if isinstance(state.config.get("sensors"), dict) else {}
    cameras = sensors.get("cameras") if isinstance(sensors.get("cameras"), dict) else {}
    recorder_script = str(recorder.get("script") or "")
    return "gmsl" in recorder_script or "defaults" in cameras


def _should_use_recorder_camera_preview(state: GatewayState) -> bool:
    return (
        state.process is not None
        or state.camera_preview_suspended
        or _state_is_gmsl2(state)
    )


def _recorder_preview_frame(device_id: str) -> bytes | None:
    path = _RECORDER_PREVIEW_DIR / f"{device_id}.jpg"
    try:
        stat = path.stat()
        if time.time() - stat.st_mtime > _RECORDER_PREVIEW_STALE_S:
            return None
        frame = path.read_bytes()
    except OSError:
        return None
    return frame or None


def _serve_recorder_camera_preview_snapshot(
    handler: BaseHTTPRequestHandler, *, state: GatewayState, device_id: str,
) -> None:
    known = any(
        d.get("id") == device_id and d.get("kind") == "camera"
        for d in state.devices
    )
    if not known:
        _json_response(handler, HTTPStatus.NOT_FOUND, {"error": f"camera not found: {device_id}"})
        return
    # Signal viewer demand so the recorder attaches preview branches on demand.
    # Do this BEFORE reading the frame: when previews are idle-reclaimed the JPEG
    # is absent and this poll (which would 503) is precisely what re-enables them.
    _maybe_send_preview_demand(state)
    frame = _recorder_preview_frame(device_id)
    if frame is None:
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no recorder preview frame yet"})
        return
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(len(frame)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    try:
        handler.wfile.write(frame)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _camera_preview_cmd(
    *,
    sensor_id: int,
    sensor_mode: int,
    source_width: int,
    source_height: int,
    source_fps: int,
) -> list[str]:
    output_width = 480
    output_height = max(1, round(output_width * source_height / max(source_width, 1)))
    output_fps = 10
    caps_in = (
        "video/x-raw(memory:NVMM),"
        f"format=NV12,width={source_width},height={source_height},framerate={source_fps}/1"
    )
    caps_out = f"video/x-raw,format=I420,width={output_width},height={output_height}"
    # Raw concatenated JPEGs on stdout (no multipartmux): the reader thread
    # splits them on SOI/EOI markers and caches the latest one.
    return [
        "gst-launch-1.0",
        "-q",
        "nvarguscamerasrc",
        f"sensor-id={sensor_id}",
        f"sensor-mode={sensor_mode}",
        "do-timestamp=true",
        "!",
        caps_in,
        "!",
        "nvvidconv",
        "!",
        caps_out,
        "!",
        "videorate",
        "!",
        f"video/x-raw,framerate={output_fps}/1",
        "!",
        "jpegenc",
        "quality=65",
        "!",
        "fdsink",
        "fd=1",
    ]


def _realsense_device_preview_cmd(state: GatewayState, device: dict[str, Any]) -> list[str]:
    config = device.get("config") if isinstance(device.get("config"), dict) else {}
    return [
        str(_venv_python3(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_realsense_preview.py"),
        "--serial",
        str(config.get("serial_number_or_name") or ""),
        "--width",
        str(int(config.get("width") or 640)),
        "--height",
        str(int(config.get("height") or 480)),
        "--fps",
        str(int(config.get("fps") or 30)),
    ]


def _ensure_realsense_device_preview(
    state: GatewayState,
    *,
    device_id: str,
    device: dict[str, Any],
) -> bool:
    script = state.repo_root / "tools" / "fr3" / "fr3_realsense_preview.py"
    if not script.is_file():
        return False
    with state.camera_preview_lock:
        proc = state.camera_preview_processes.get(device_id)
        if proc is not None and proc.poll() is None:
            return True
    with state.camera_preview_spawn_lock:
        with state.camera_preview_lock:
            proc = state.camera_preview_processes.get(device_id)
            if proc is not None and proc.poll() is None:
                return True
        proc = subprocess.Popen(
            _realsense_device_preview_cmd(state, device),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=state.repo_root,
            env=_tool_env(state.repo_root),
            start_new_session=True,
        )
        with state.camera_preview_lock:
            state.camera_preview_processes[device_id] = proc
    Thread(
        target=_camera_preview_reader,
        args=(state, device_id, proc),
        daemon=True,
        name=f"realsense-preview-{device_id}",
    ).start()
    return True


def _serve_realsense_device_preview_snapshot(
    handler: BaseHTTPRequestHandler,
    *,
    state: GatewayState,
    device_id: str,
    device: dict[str, Any],
) -> None:
    if state.camera_preview_suspended:
        _json_response(handler, HTTPStatus.CONFLICT, {"error": "camera preview suspended while recorder connects"})
        return
    with state.camera_preview_lock:
        state.camera_preview_last_access[device_id] = time.time()
    if not _ensure_realsense_device_preview(state, device_id=device_id, device=device):
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "RealSense preview unavailable"})
        return
    deadline = time.monotonic() + _PREVIEW_FIRST_FRAME_TIMEOUT_S
    frame: bytes | None = None
    while time.monotonic() < deadline:
        with state.camera_preview_lock:
            cached = state.camera_preview_frames.get(device_id)
            state.camera_preview_last_access[device_id] = time.time()
        if cached is not None:
            frame = cached[0]
            break
        time.sleep(0.1)
    if frame is None:
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no RealSense preview frame yet"})
        return
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(len(frame)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    try:
        handler.wfile.write(frame)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _camera_preview_reader(state: GatewayState, device_id: str, proc: subprocess.Popen[bytes]) -> None:
    """Pump JPEG frames from one preview pipeline into the latest-frame cache.

    Splits the raw concatenated JPEG stream on SOI (FFD8) / EOI (FFD9) markers
    and keeps only the most recent complete frame. Uses ``select`` so it wakes
    even when the pipeline stalls, letting it self-terminate on the idle TTL
    (which releases the Argus session — important so a later Connect doesn't
    find the camera still occupied).
    """
    buf = bytearray()
    stdout = proc.stdout
    assert stdout is not None
    fd = stdout.fileno()
    try:
        while True:
            if proc.poll() is not None:
                break
            with state.camera_preview_lock:
                last = state.camera_preview_last_access.get(device_id, 0.0)
            if time.time() - last > _PREVIEW_IDLE_TTL_S:
                break
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue
            chunk = os.read(fd, 256 * 1024)
            if not chunk:
                break
            buf += chunk
            # Extract every complete JPEG currently buffered; keep the last.
            latest: bytes | None = None
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi < 0:
                    buf.clear()
                    break
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if eoi < 0:
                    if soi > 0:
                        del buf[:soi]
                    break
                latest = bytes(buf[soi : eoi + 2])
                del buf[: eoi + 2]
            if latest is not None:
                with state.camera_preview_lock:
                    state.camera_preview_frames[device_id] = (latest, time.time())
    finally:
        _terminate_preview_proc(proc)
        with state.camera_preview_lock:
            if state.camera_preview_processes.get(device_id) is proc:
                state.camera_preview_processes.pop(device_id, None)
            state.camera_preview_frames.pop(device_id, None)


def _ensure_camera_preview(
    state: GatewayState,
    *,
    device_id: str,
    sensor_id: int,
    sensor_mode: int,
    source_width: int,
    source_height: int,
    source_fps: int,
) -> bool:
    """Spawn the preview pipeline for ``device_id`` if it isn't already running.

    Returns False only when gst-launch is unavailable. The Argus open is
    serialized + staggered across cameras to dodge the NVMM dmabuf race.
    """
    if shutil.which("gst-launch-1.0") is None:
        return False
    with state.camera_preview_lock:
        proc = state.camera_preview_processes.get(device_id)
        if proc is not None and proc.poll() is None:
            return True
    cmd = _camera_preview_cmd(
        sensor_id=sensor_id,
        sensor_mode=sensor_mode,
        source_width=source_width,
        source_height=source_height,
        source_fps=source_fps,
    )
    stagger = _camera_preview_stagger_s(state)
    with state.camera_preview_spawn_lock:
        # Re-check under the spawn lock: another request may have raced us.
        with state.camera_preview_lock:
            proc = state.camera_preview_processes.get(device_id)
            if proc is not None and proc.poll() is None:
                return True
        if stagger > 0:
            gap = stagger - (time.monotonic() - state.camera_preview_last_spawn_s)
            if gap > 0:
                time.sleep(gap)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(Path.cwd()),
            start_new_session=True,
        )
        state.camera_preview_last_spawn_s = time.monotonic()
        with state.camera_preview_lock:
            state.camera_preview_processes[device_id] = proc
    Thread(
        target=_camera_preview_reader,
        args=(state, device_id, proc),
        daemon=True,
        name=f"camera-preview-{device_id}",
    ).start()
    return True


def _serve_camera_preview_snapshot(
    handler: BaseHTTPRequestHandler,
    *,
    state: GatewayState,
    device_id: str,
    sensor_id: int,
    sensor_mode: int,
    source_width: int,
    source_height: int,
    source_fps: int,
) -> None:
    with state.camera_preview_lock:
        state.camera_preview_last_access[device_id] = time.time()
    ok = _ensure_camera_preview(
        state,
        device_id=device_id,
        sensor_id=sensor_id,
        sensor_mode=sensor_mode,
        source_width=source_width,
        source_height=source_height,
        source_fps=source_fps,
    )
    if not ok:
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "gst-launch-1.0 not found"})
        return
    # Wait briefly for a frame (cold spawn must open Argus + settle ISP/AWB).
    deadline = time.monotonic() + _PREVIEW_FIRST_FRAME_TIMEOUT_S
    frame: bytes | None = None
    while True:
        with state.camera_preview_lock:
            cached = state.camera_preview_frames.get(device_id)
            # Keep the pipeline marked live while we wait so the reader's TTL
            # check doesn't reap it out from under a slow first frame.
            state.camera_preview_last_access[device_id] = time.time()
        if cached is not None:
            frame = cached[0]
            break
        with state.camera_preview_lock:
            proc = state.camera_preview_processes.get(device_id)
        if proc is not None and proc.poll() is not None:
            break  # pipeline died before producing a frame
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if frame is None:
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no preview frame yet"})
        return
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(len(frame)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    try:
        handler.wfile.write(frame)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _camera_preview_params(state: GatewayState, device_id: str) -> tuple[int, int, int, int, int] | None:
    device = next((d for d in state.devices if d.get("id") == device_id and d.get("kind") == "camera"), None)
    if device is None:
        return None
    config = device.get("config") if isinstance(device.get("config"), dict) else {}
    try:
        sensor_id = int(config.get("sensor_id", str(device_id).rsplit("_", 1)[-1]))
    except (TypeError, ValueError):
        return None
    sensor_mode = int(config.get("sensor_mode") or 0)
    width = int(config.get("width") or 1920)
    height = int(config.get("height") or 1080)
    fps = int(config.get("fps") or 60)
    return sensor_id, max(sensor_mode, 0), max(width, 1), max(height, 1), max(fps, 1)


def _box_preview_payload(state: GatewayState, device_id: str) -> dict[str, Any]:
    box = state.device_preview.get("box")
    if not isinstance(box, dict):
        box = {}
    sensors = box.get("sensors") if isinstance(box.get("sensors"), dict) else {}
    updated_at = float(box.get("updatedAt") or 0.0)
    stale_s = max(0.0, time.time() - updated_at) if updated_at else None
    return {
        "active": bool(box.get("active")) and (stale_s is None or stale_s < 2.0),
        "deviceId": device_id,
        "updatedAt": updated_at,
        "staleS": stale_s,
        "receivedAtS": box.get("received_at_s"),
        "receivedWallTimeS": box.get("received_wall_time_s"),
        "status": box.get("status") if isinstance(box.get("status"), dict) else {},
        "sensor": sensors.get(device_id) if isinstance(sensors, dict) else None,
        "sensors": sensors,
    }


def _serve_video(handler: BaseHTTPRequestHandler, video_path: Path) -> None:
    file_size = video_path.stat().st_size
    range_header = handler.headers.get("Range") or handler.headers.get("range")
    start = 0
    end = file_size - 1
    status = HTTPStatus.OK
    if range_header:
        match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
        if match:
            start_raw, end_raw = match.group(1), match.group(2)
            if start_raw == "" and end_raw != "":
                length = int(end_raw)
                start = max(0, file_size - length)
                end = file_size - 1
            else:
                start = int(start_raw) if start_raw else 0
                end = int(end_raw) if end_raw else file_size - 1
            end = min(end, file_size - 1)
            if start > end:
                handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                handler.send_header("Content-Range", f"bytes */{file_size}")
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.end_headers()
    with video_path.open("rb") as video_file:
        video_file.seek(start)
        remaining = length
        chunk_size = 64 * 1024
        while remaining > 0:
            chunk = video_file.read(min(chunk_size, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            remaining -= len(chunk)


def _snapshot(state: GatewayState) -> dict[str, Any]:
    teleop_process = state.teleop_process
    if teleop_process is not None and teleop_process.poll() is not None:
        state.log("info" if teleop_process.returncode == 0 else "warn", f"FR3 teleop exited with code {teleop_process.returncode}")
        state.teleop_process = None
        state.teleop_started_at_s = None
        state.teleop.pid = None
        state.teleop.realRobotReady = False
        state.teleop.state = "idle" if teleop_process.returncode == 0 else "error"
        if state.teleop.lastOutput:
            state.teleop.message = f"FR3 teleop exited with code {teleop_process.returncode}: {state.teleop.lastOutput}"
        else:
            state.teleop.message = f"FR3 teleop exited with code {teleop_process.returncode}"

    process = state.process
    if process is not None and process.poll() is not None:
        state.log("warn", f"Handheld recorder exited with code {process.returncode}")
        exited_from = state.recording.state
        state.process = None
        state.camera_preview_suspended = False
        state.process_started_at_s = None
        _clear_runtime_recording_config(state)
        state.recording.state = "idle" if process.returncode == 0 else "error"
        state.recording.pid = None
        state.recording.frameIndex = 0 if process.returncode == 0 else state.recording.frameIndex
        state.recording.queueDepth = 0
        if isinstance(state.device_preview.get("box"), dict):
            state.device_preview["box"] = {**state.device_preview["box"], "active": False}
        summary = (
            _recorder_failure_summary(state.recording)
            if process.returncode != 0 else state.recording.lastOutput
        )
        if summary:
            state.recording.message = f"Recorder exited with code {process.returncode}: {summary}"
        else:
            state.recording.message = f"Recorder exited with code {process.returncode}"
        _set_all_device_states(
            state,
            "idle" if process.returncode == 0 or exited_from in ("idle", "discarding") else "error",
        )

    replay_process = state.replay_process
    if replay_process is not None and replay_process.poll() is not None:
        replay_kind = state.replay_process_kind or "mujoco"
        label = "MuJoCo replay" if replay_kind == "mujoco" else "Real robot replay"
        state.log("info" if replay_process.returncode == 0 else "warn", f"{label} exited with code {replay_process.returncode}")
        state.replay_process = None
        state.replay_process_kind = ""
        state.replay_started_at_s = None
        state.replay.pid = None
        if replay_kind == "mujoco":
            _finish_mujoco_validation(state, replay_process.returncode)
        else:
            _stop_realsense_preview(state)
            _append_real_replay_log(
                state,
                "complete" if replay_process.returncode == 0 else "error",
                f"real replay exited with code {replay_process.returncode}",
            )
            state.replay.safety = "locked"
            state.replay.state = "complete" if replay_process.returncode == 0 else "aborted"
            if state.replay.lastOutput:
                state.replay.message = f"{label} exited with code {replay_process.returncode}: {state.replay.lastOutput}"
            else:
                state.replay.message = f"{label} exited with code {replay_process.returncode}"

    for line in state.recording.recentOutput:
        _mark_failed_camera_devices(state, _failed_camera_ids_from_recorder_output(line))
    if state.recording.state == "connecting" and any(
        _recorder_output_is_failure(line) for line in state.recording.recentOutput
    ):
        state.recording.state = "error"
    recording_state = state.recording.state
    elapsed_s = None
    if state.process_started_at_s is not None:
        elapsed_s = max(0.0, time.monotonic() - state.process_started_at_s)
    # Read the dataset scan results from the cache the background refresher
    # maintains off-lock. NEVER scan the dataset tree here: _snapshot runs under
    # state.lock, and walking 298G/600ep would block the recorder-stdout drain
    # and all camera.jpg requests for seconds (the preview-freeze root cause).
    recorded_datasets = list(state.cached_recorded_datasets)
    trajectory = list(state.cached_trajectory)
    trajectory_meta = dict(state.cached_trajectory_meta)
    selected_replay_root = state.selected_replay_root.resolve() if state.selected_replay_root is not None else None
    if selected_replay_root is not None:
        try:
            cached_root = Path(str(trajectory_meta.get("datasetRoot") or "")).resolve()
        except OSError:
            cached_root = None
        if cached_root != selected_replay_root:
            selected_info = _load_dataset_info(selected_replay_root)
            selected_meta = _dataset_replay_meta(state, selected_replay_root, selected_info)
            trajectory = []
            trajectory_meta = {
                **selected_meta,
                "dataStatus": _recorded_dataset_status(selected_replay_root),
                "trajectoryKind": "pose" if _dataset_has_replay_pose_hint(selected_replay_root, selected_info) else "none",
                "message": f"Selected {_dataset_kind(state, selected_replay_root)} dataset: {selected_replay_root.name}",
            }
    if recorded_datasets and not trajectory_meta.get("datasetRoot"):
        latest_dataset = recorded_datasets[0]
        trajectory_meta = {
            **trajectory_meta,
            "datasetRoot": latest_dataset["path"],
            "datasetKind": latest_dataset.get("datasetKind") or "recorded",
            "sourcePath": latest_dataset["sourcePath"],
            "totalEpisodes": latest_dataset["totalEpisodes"],
            "recordedFrames": latest_dataset["totalFrames"],
        }
    state.replay.datasetRoot = str(trajectory_meta.get("datasetRoot") or state.replay.datasetRoot)
    state.replay.datasetKind = str(trajectory_meta.get("datasetKind") or state.replay.datasetKind or "recorded")
    state.replay.sourcePath = str(trajectory_meta.get("sourcePath") or "")
    state.replay.dataStatus = str(trajectory_meta.get("dataStatus") or "missing")
    state.replay.trajectoryKind = str(trajectory_meta.get("trajectoryKind") or "none")
    state.replay.totalEpisodes = int(trajectory_meta.get("totalEpisodes") or 0)
    state.replay.episodeOptions = [
        int(episode) for episode in trajectory_meta.get("episodeOptions", []) if isinstance(episode, int)
    ]
    state.replay.recordedFrames = int(trajectory_meta.get("recordedFrames") or len(trajectory))
    diagnostics = trajectory_meta.get("diagnostics") or []
    state.replay.diagnostics = [str(item) for item in diagnostics] if isinstance(diagnostics, list) else [str(diagnostics)]
    if not state.replay.mujocoValidation:
        state.replay.mujocoValidation = _new_mujoco_validation(state)
    if trajectory:
        state.replay.episode = int(trajectory_meta.get("episode") or 0)
        state.replay.totalFrames = len(trajectory)
        state.replay.dataset = str(trajectory_meta.get("datasetRoot") or state.replay.dataset)
        if state.replay.state == "idle":
            state.replay.message = str(trajectory_meta.get("message") or f"Loaded recorded episode {state.replay.episode}")
    elif state.replay.state == "idle":
        state.replay.totalFrames = state.replay.recordedFrames
        state.replay.message = str(trajectory_meta.get("message") or "No recorded trajectory loaded")
    _refresh_mujoco_validation_current(state)

    return {
        "deployment": {"profile": state.profile, **DEPLOYMENT_PROFILES[state.profile]},
        "gateway": {
            "configPath": str(state.config_path),
            "pid": state.recording.pid,
            "state": "online",
            "processElapsedS": elapsed_s,
            "datasetsRoot": str(state.datasets_root) if state.datasets_root else "",
        },
        "configSummary": _config_summary(
            state.runtime_recording_config or state.config,
            state.runtime_recording_config_path or state.config_path,
        ),
        "devices": [
            {**device, "state": "running" if recording_state == "recording" and device["state"] != "error" else device["state"]}
            for device in state.devices
        ],
        "recording": asdict(state.recording),
        # The tool frame is a property of the config, not of a session, so both pages report it
        # whether or not anything is running. It used to be spelled out as a literal in the replay
        # page and to sit at a stale dataclass default on an idle teleop status -- exactly the
        # label an operator would trust while replaying a dataset recorded 411 mm away in the
        # other frame.
        "replay": {**asdict(state.replay), "targetFrameName": _fr3_target_frame_name(state)},
        "teleop": {**asdict(state.teleop), "targetFrameName": _fr3_target_frame_name(state)},
        "teleopGains": _teleop_gains_payload(state),
        "annotation": _active_annotation(state),
        "calibration": asdict(state.calibration),
        "calibrationSession": _calibration_session_payload(state),
        "markerTcp": _marker_tcp_session_payload(state),
        "recordedDatasets": recorded_datasets,
        "processing": list(state.cached_processing_items),
        "trajectory": trajectory,
        "events": [asdict(event) for event in state.events],
        "tasks": _tasks_with_progress(state),
        "activeTaskId": state.active_task_id or "",
        "datasetExport": asdict(state.dataset_export),
        "training": asdict(state.training),
    }


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    raw_body = handler.rfile.read(length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def _venv_python(repo_root: Path, *, prefer_fr3: bool = False) -> Path:
    names = (".venv-fr3", ".venv", "venv") if prefer_fr3 else (".venv", "venv")
    for name in names:
        candidate = repo_root / name / "bin" / "python"
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _venv_python3(repo_root: Path, *, prefer_fr3: bool = False) -> Path:
    names = (".venv-fr3", ".venv", "venv") if prefer_fr3 else (".venv", "venv")
    for name in names:
        for executable in ("python3", "python"):
            candidate = repo_root / name / "bin" / executable
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


def _recorder_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["LEROBOT_GUI_HEADLESS"] = "1"
    return env


def _tool_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _fr3_pika_asset_paths(repo_root: Path) -> tuple[Path, Path]:
    asset_root = repo_root / "src" / "lerobot" / "robots" / "franka_research3" / "assets" / "franka_fr3"
    return asset_root / "fr3_pika_gripper.urdf", asset_root / "fr3_pika_gripper_scene.xml"


# Mirrors FrankaResearch3Config.target_frame_name, and is only reached by a config that does not
# set the key -- the workstation's does. Pinned to the dataclass by
# tests/scripts/test_data_collection_gui_gateway.py so the two cannot answer differently.
FR3_DEFAULT_TARGET_FRAME_NAME = "pika_gripper_ee"


def _fr3_target_frame_name(state: GatewayState) -> str:
    """The tool frame, from the robot config rather than a literal.

    Recording, MuJoCo replay and real replay have to name the *same* frame: `pika_task_tcp` and
    `pika_gripper_ee` are 411 mm apart on the same URDF and share an orientation, so a mismatch is
    a silent 411 mm offset rather than a failure. Several of these commands used to spell the frame
    out, which made them the only places that would not follow `robot.target_frame_name` if the rig
    ever switched -- exactly the direction a switch would be attempted from.

    The fallback is `FrankaResearch3Config.target_frame_name`'s own default rather than a literal
    picked here. A config that omits the key gets that frame from the robot class no matter what
    this function says, so any other answer would hand the sim teleop and the replay a different
    frame from the one the recorder is using -- the same silent offset, arrived at from the inside.
    """
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    return str(robot.get("target_frame_name") or "").strip() or FR3_DEFAULT_TARGET_FRAME_NAME


def _teleop_gain_cli_args(state: GatewayState) -> list[str]:
    """Gain flags for the sim teleop script, and only for gains the operator actually set.

    `fr3_mujoco_teleop.py` carries its own defaults (see FR3_SIM_TELEOP_GAIN_DEFAULTS), so passing
    nothing leaves the sim exactly as it was. Passing a per-axis flag as an explicit `None` is not
    possible over argparse, so an override that clears an axis back to "follow the global gain" is
    expressed by simply not sending that flag -- which is what dropping `None` values here does.
    """

    args: list[str] = []
    for field_name, value in sorted((state.runtime_teleop_gains or {}).items()):
        if value is None:
            continue
        args.extend([f"--{field_name.replace('_', '-')}", repr(float(value))])
    return args


def _fr3_sim_teleop_command(state: GatewayState) -> list[str]:
    urdf_path, sim_xml_path = _fr3_pika_asset_paths(state.repo_root)
    return [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_mujoco_teleop.py"),
        "--teleop-type",
        "spacemouse",
        "--urdf-path",
        str(urdf_path),
        "--sim-xml-path",
        str(sim_xml_path),
        "--target-frame-name",
        _fr3_target_frame_name(state),
        "--no-viewer",
        "--enable-cameras",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-fps",
        "30",
        "--disable-otg",
        *_teleop_gain_cli_args(state),
    ]


def _resolve_teleop_config_path(state: GatewayState) -> Path:
    """Config path to spawn the real teleop with: the repo's file, or an overlay carrying gains."""

    overrides = dict(state.runtime_teleop_gains or {})
    if not overrides:
        return state.config_path
    overlay = copy.deepcopy(state.config)
    _apply_teleop_gain_overrides(overlay, overrides)
    path = _write_overlay_config(state, overlay, name=_TELEOP_OVERLAY_NAME)
    state.log(
        "info",
        "Real teleop launching with SpaceMouse gain overrides "
        + ", ".join(f"{name}={value:g}" for name, value in sorted(overrides.items())),
    )
    return path


def _fr3_real_teleop_command(state: GatewayState) -> list[str]:
    return [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        "-m",
        "tools.fr3.fr3_real_teleop_runtime",
        f"--config_path={_resolve_teleop_config_path(state)}",
    ]


def _read_teleop_process_output(state: GatewayState, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output = line.strip()
        if not output:
            continue
        with state.lock:
            if state.teleop_process is not process:
                return
            if output.startswith("libEGL warning:"):
                state.log("warn", f"fr3 teleop: {output}")
                continue
            state.teleop.lastOutput = output
            if output == "fr3_real_teleop=READY":
                state.teleop.state = "running"
                state.teleop.realRobotReady = True
                state.teleop.message = "FR3, Pika gripper, and SpaceMouse are connected"
            elif output.startswith("Camera streams:"):
                state.teleop.message = "External and wrist simulation views are live"
            else:
                state.teleop.message = output
            state.log("info", f"fr3 teleop: {output}")


def _start_teleop_output_reader(state: GatewayState, process: subprocess.Popen[str]) -> None:
    Thread(
        target=_read_teleop_process_output,
        args=(state, process),
        daemon=True,
        name=f"fr3-teleop-output-{process.pid}",
    ).start()


def _start_fr3_sim_teleop(state: GatewayState) -> None:
    process = state.teleop_process
    if process is not None and process.poll() is None:
        state.teleop.message = "FR3 MuJoCo teleop is already running"
        return
    urdf_path, sim_xml_path = _fr3_pika_asset_paths(state.repo_root)
    if not urdf_path.is_file():
        raise RuntimeError(f"Missing FR3 Pika URDF: {urdf_path}")
    if not sim_xml_path.is_file():
        raise RuntimeError(f"Missing FR3 Pika MuJoCo scene: {sim_xml_path}")
    command = _fr3_sim_teleop_command(state)
    env = _tool_env(state.repo_root)
    env["MUJOCO_GL"] = "egl"
    process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    state.teleop_process = process
    state.teleop_started_at_s = time.monotonic()
    state.teleop = TeleopStatus(
        state="running",
        backend="mujoco",
        inputDevice="spacemouse",
        robotModel="fr3_pika_gripper",
        urdfPath=str(urdf_path),
        simXmlPath=str(sim_xml_path),
        targetFrameName=_fr3_target_frame_name(state),
        pid=process.pid,
        message="FR3 Pika MuJoCo teleop started; two camera streams are rendering in the web UI",
        command=command,
        realRobotReady=False,
    )
    state.log("info", f"Started FR3 Pika MuJoCo teleop pid={process.pid}")
    _start_teleop_output_reader(state, process)


def _start_fr3_real_teleop(state: GatewayState) -> None:
    process = state.teleop_process
    if process is not None and process.poll() is None:
        state.teleop.message = "An FR3 teleop session is already active"
        return
    urdf_path, sim_xml_path = _fr3_pika_asset_paths(state.repo_root)
    if not urdf_path.is_file():
        raise RuntimeError(f"Missing FR3 Pika URDF: {urdf_path}")
    if not state.config_path.is_file():
        raise RuntimeError(f"Missing workstation FR3 config: {state.config_path}")

    command = _fr3_real_teleop_command(state)
    process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_tool_env(state.repo_root),
        start_new_session=True,
    )
    state.teleop_process = process
    state.teleop_started_at_s = time.monotonic()
    state.teleop = TeleopStatus(
        state="starting",
        backend="real",
        inputDevice="spacemouse",
        robotModel="fr3_pika_gripper",
        urdfPath=str(urdf_path),
        simXmlPath=str(sim_xml_path),
        targetFrameName=_fr3_target_frame_name(state),
        pid=process.pid,
        message="Connecting to FR3 192.168.1.206, Pika gripper, and SpaceMouse",
        command=command,
        realRobotReady=False,
    )
    state.log("info", f"Started FR3 Pika real teleop without an FCI preflight gate pid={process.pid}")
    _start_teleop_output_reader(state, process)


def _stop_fr3_teleop(state: GatewayState) -> None:
    process = state.teleop_process
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1.0)
        state.log("warn", f"Stopped FR3 teleop pid={process.pid}")
    state.teleop_process = None
    state.teleop_started_at_s = None
    state.teleop.state = "idle"
    state.teleop.pid = None
    state.teleop.realRobotReady = False
    state.teleop.message = "FR3 Pika teleop stopped"


def _serve_teleop_camera_snapshot(
    handler: BaseHTTPRequestHandler,
    *,
    state: GatewayState,
    view_id: str,
) -> None:
    allowed_views = {str(view.get("id")) for view in state.teleop.cameraViews}
    if view_id not in allowed_views:
        _json_response(handler, HTTPStatus.NOT_FOUND, {"error": f"unknown teleop camera view: {view_id}"})
        return
    process = state.teleop_process
    if process is None or process.poll() is not None or state.teleop.state != "running":
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "MuJoCo teleop is not running"})
        return
    try:
        with urlopen(f"http://127.0.0.1:18765/camera/{view_id}.jpg", timeout=1.0) as response:
            payload = response.read()
    except (OSError, URLError, TimeoutError) as exc:
        _json_response(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"camera frame unavailable: {exc}"})
        return
    try:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass


def _set_all_device_states(state: GatewayState, device_state: str) -> None:
    for device in state.devices:
        device["state"] = device_state


def _set_active_device_states(state: GatewayState, device_state: str) -> None:
    for device in state.devices:
        if device.get("state") != "error":
            device["state"] = device_state


def _mark_connected_devices(state: GatewayState, kind: str, summary: str) -> None:
    configured = [device for device in state.devices if device.get("kind") == kind]
    if not configured:
        return

    connected_ids = {
        item.strip()
        for item in summary.split(",")
        if item.strip() and item.strip() != "(none)"
    }
    if not connected_ids:
        for device in configured:
            device["state"] = "error"
        return

    for device in configured:
        device["state"] = "running" if str(device.get("id")) in connected_ids else "error"


def _mark_failed_camera_devices(state: GatewayState, failed_ids: set[str]) -> None:
    if not failed_ids:
        return
    for device in state.devices:
        if device.get("kind") == "camera" and str(device.get("id")) in failed_ids:
            device["state"] = "error"


def _failed_camera_ids_from_recorder_output(output: str) -> set[str]:
    failed_ids: set[str] = set()

    failed_camera_match = re.search(r"Camera '([^']+)' failed to connect", output)
    if failed_camera_match:
        failed_ids.add(failed_camera_match.group(1))

    stream_failed_match = re.search(r"\bstream\(s\) failed:\s*(?P<failed>.+)$", output)
    if stream_failed_match:
        failed_ids.update(re.findall(r"\bcam_\d+\b", stream_failed_match.group("failed")))

    partial_success_match = re.search(r"\bpartial success:.*\bfailed:\s*(?P<failed>.+)$", output)
    if partial_success_match:
        failed_ids.update(re.findall(r"\bcam_\d+\b", partial_success_match.group("failed")))

    if any(token in output for token in ("connect stable window failed", "bus EOS", "stream(s) failed", "failed to reach PLAYING")):
        failed_ids.update(re.findall(r"\bcam_\d+\b(?=\()", output))
        failed_ids.update(re.findall(r"\[(cam_\d+)\]", output))

    return failed_ids


def _recorder_script(state: GatewayState) -> tuple[Path, str]:
    """Resolve which recorder process the gateway should spawn for this config.

    Returns ``(script_path, config_flag)`` so callers can build the command.
    Legacy handheld configs (no ``recorder`` block) keep the old
    ``--config_path`` underscore flag; the Thor GMSL2 recorder uses the
    hyphenated ``--config-path`` argparse convention.
    """

    raw = state.config.get("recorder") if isinstance(state.config.get("recorder"), dict) else {}
    script_path = raw.get("script") if isinstance(raw, dict) else None
    if script_path:
        script = Path(str(script_path))
        if not script.is_absolute():
            script = state.repo_root / script
        flag = str(raw.get("config_flag") or "--config-path")
        return script, flag
    if state.profile == "workstation":
        # The FR3 workstation config is a lerobot RecordConfig, which draccus parses strictly;
        # it cannot carry a `recorder:` block to point here, so the profile selects the script.
        return state.repo_root / WORKSTATION_RECORDER_SCRIPT, "--config_path"
    return state.repo_root / DEFAULT_RECORDER_SCRIPT, "--config_path"


def _default_log_dir(repo_root: Path) -> Path:
    return repo_root / "outputs" / "logs" / "data_collection_gui"


def _timestamp_for_log() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_log_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _new_recorder_log_path(state: GatewayState) -> Path:
    log_dir = _ensure_log_dir(state.log_dir or _default_log_dir(state.repo_root))
    return log_dir / f"recorder_{_timestamp_for_log()}.log"


def _append_line(path: Path | None, line: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
    except OSError:
        pass


def _ensure_recorder_running(state: GatewayState) -> subprocess.Popen[str]:
    process = state.process
    if process is None or process.poll() is not None:
        state.process = None
        state.recording.pid = None
        raise RuntimeError("Connect devices before starting an episode.")
    return process


# Serialize all writes to the recorder's stdin pipe. The control commands
# (connect "\n", save, discard, quit) come from request-handler threads, while
# the preview-demand heartbeat is sent from the camera.jpg snapshot threads;
# without a lock two writes could interleave at the byte level and corrupt a
# command line. One recorder process exists at a time, so a module-level lock
# is sufficient.
_RECORDER_STDIN_LOCK = Lock()


def _write_recorder_stdin(process: subprocess.Popen[str], text: str) -> None:
    if process.stdin is None:
        raise RuntimeError("Handheld recorder stdin is unavailable.")
    with _RECORDER_STDIN_LOCK:
        process.stdin.write(text)
        process.stdin.flush()


def _maybe_send_preview_demand(state: GatewayState) -> None:
    """Tell the recorder a viewer is polling camera.jpg, debounced to ~1/s.

    The recorder attaches preview tee branches on demand and reclaims them
    after an idle TTL (recording_preview_idle_ttl_s), so this heartbeat must
    keep arriving while the Device Manager grid is open. Debouncing caps the
    stdin chatter at one line per second no matter how many tiles poll. Sent
    even when the JPEG does not exist yet (a poll that 503s is exactly what
    must wake the previews up).
    """
    process = state.process
    if process is None or process.stdin is None:
        return
    # Only the GMSL2 recorder understands "preview_demand" and produces
    # recorder-owned preview JPEGs. The handheld recorder reads stdin byte by
    # byte as keypresses, so a heartbeat there would be mis-handled — never
    # send it to a non-GMSL2 recorder.
    if not _state_is_gmsl2(state):
        return
    now = time.monotonic()
    with _RECORDER_STDIN_LOCK:
        if now - state.recorder_preview_demand_sent_s < _RECORDER_PREVIEW_DEMAND_INTERVAL_S:
            return
        state.recorder_preview_demand_sent_s = now
        try:
            process.stdin.write("preview_demand\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass


def _trigger_box_six_d_force_cali(
    state: GatewayState, *, origin: bool = False, box_id: str = ""
) -> dict[str, Any]:
    """Ask the running GMSL2 recorder to calibrate the BOX 6D force sensor.

    The recorder owns the live box connection, so the command rides its stdin
    and progress comes back as CALI_LOG/CALI_DONE stdout lines. ``origin=False``
    (default) requests software zeroing (``cali_6dforce``); ``origin=True``
    requests the MCU-side TLV origin calibration (``cali_6dforce_origin``).
    ``box_id`` restricts the run to a single box (empty = whole fleet), sent as a
    ``:<box_id>`` suffix on the stdin command. Returns a small status dict for
    the POST response.
    """
    process = state.process
    if process is None or process.poll() is not None:
        return {"ok": False, "error": "recorder is not connected; press Connect first"}
    if not _state_is_gmsl2(state):
        return {"ok": False, "error": "6D force calibration requires the GMSL2/BOX recorder"}
    base_cmd = "cali_6dforce_origin" if origin else "cali_6dforce"
    stdin_cmd = f"{base_cmd}:{box_id}\n" if box_id else f"{base_cmd}\n"
    scope = f" [{box_id}]" if box_id else ""
    label = ("6D force MCU-origin" if origin else "6D force software-zero") + scope
    with state.box_cali_lock:
        state.box_cali_running = True
        state.box_cali_log.append(
            {"ts": time.time(), "line": f"{label} calibration command sent to recorder", "done": False}
        )
        del state.box_cali_log[:-200]
    try:
        _write_recorder_stdin(process, stdin_cmd)
    except (BrokenPipeError, ValueError, OSError, RuntimeError) as exc:
        with state.box_cali_lock:
            state.box_cali_running = False
            state.box_cali_log.append(
                {"ts": time.time(), "line": f"failed to send command: {exc}", "done": True}
            )
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _trigger_box_touch_cali(state: GatewayState, *, box_id: str = "") -> dict[str, Any]:
    """Ask the running GMSL2 recorder to calibrate (re-zero) the BOX touch pads.

    Mirrors :func:`_trigger_box_six_d_force_cali` but rides its own
    TOUCHCALI_LOG/TOUCHCALI_DONE stdout channel into ``box_touch_cali_log`` so
    the touch viewer never shows 6D force lines (and vice versa). ``box_id``
    restricts the run to a single box (empty = whole fleet); a box's two pads
    share one MCU-side re-zero, so this is per-box, not per-pad.
    """
    process = state.process
    if process is None or process.poll() is not None:
        return {"ok": False, "error": "recorder is not connected; press Connect first"}
    if not _state_is_gmsl2(state):
        return {"ok": False, "error": "touch calibration requires the GMSL2/BOX recorder"}
    scope = f" [{box_id}]" if box_id else ""
    with state.box_touch_cali_lock:
        state.box_touch_cali_running = True
        state.box_touch_cali_log.append(
            {"ts": time.time(), "line": f"touch calibration command sent to recorder{scope}", "done": False}
        )
        del state.box_touch_cali_log[:-200]
    try:
        _write_recorder_stdin(process, f"calitouch:{box_id}\n" if box_id else "calitouch\n")
    except (BrokenPipeError, ValueError, OSError, RuntimeError) as exc:
        with state.box_touch_cali_lock:
            state.box_touch_cali_running = False
            state.box_touch_cali_log.append(
                {"ts": time.time(), "line": f"failed to send command: {exc}", "done": True}
            )
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _box_cali_log_payload(state: GatewayState) -> dict[str, Any]:
    with state.box_cali_lock:
        return {"running": state.box_cali_running, "lines": list(state.box_cali_log)}


def _box_touch_cali_log_payload(state: GatewayState) -> dict[str, Any]:
    with state.box_touch_cali_lock:
        return {"running": state.box_touch_cali_running, "lines": list(state.box_touch_cali_log)}


def _connect_recorder(
    state: GatewayState,
    *,
    backend: str | None = None,
    episode_time_s: float | None = None,
    recording_fps: int | None = None,
) -> None:
    if state.process is not None and state.process.poll() is None:
        state.recording.message = "Devices are already connected"
        return

    is_workstation = state.profile == "workstation"
    if backend is not None:
        if backend not in RECORD_BACKENDS:
            raise ValueError(f"Recording backend must be one of {RECORD_BACKENDS}, got {backend!r}")
        if not is_workstation and backend != DEFAULT_RECORD_BACKEND:
            raise ValueError("Only the workstation profile can choose a recording backend")
        state.recording.backend = backend

    config_path = _resolve_recorder_config_path(
        state, episode_time_s=episode_time_s, recording_fps=recording_fps
    )
    recorder_script, config_flag = _recorder_script(state)
    command = [
        # The FR3 stack (panda_py, placo, mujoco) lives in .venv-fr3 on the workstation.
        str(_venv_python(state.repo_root, prefer_fr3=is_workstation)),
        str(recorder_script),
        f"{config_flag}={config_path}",
    ]
    # Thor GMSL2 recorder: skip its own nvarguscamerasrc probe pass.
    # Operators run tools/thor/gmsl2/recover_argus.sh before Connect, which
    # already opens each sensor once through Argus; re-probing here adds an
    # extra 11x open/close cycle that has been observed to destabilise
    # nvargus-daemon (sids that recover saw as OK time out 8s later, then
    # the first PLAYING transition deadlocks the Python thread).
    if "thor_record" in str(recorder_script):
        command.append("--skip-argus-probe")
    env = _recorder_env(state.repo_root)
    if is_workstation:
        command.append(f"--backend={state.recording.backend}")
        # The FR3 recorder only ever writes a local dataset -- it neither pulls nor pushes to the
        # Hub. Left online, any local metadata read that misses falls back to huggingface.co, and
        # that connect has no timeout on a workstation with no route to it: the recorder hangs
        # before its first output line and stops reading its own stdin, so the GUI cannot even
        # cancel it. Offline turns that class of hang into an immediate error. Not set for Thor,
        # whose handheld recorder honours dataset.push_to_hub.
        env["HF_HUB_OFFLINE"] = "1"
        if state.recording.backend == "sim":
            # Headless MuJoCo rendering: the recorder runs detached from any X session.
            env["MUJOCO_GL"] = "egl"
    recorder_log_path = _new_recorder_log_path(state)
    state.process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    state.process_started_at_s = time.monotonic()
    state.recording.state = "connecting"
    state.recording.pid = state.process.pid
    state.recording.frameIndex = 0
    state.recording.queueDepth = 0
    state.recording.message = (
        f"Starting FR3 {state.recording.backend} recorder"
        if is_workstation
        else "Connecting handheld devices"
    )
    # A new session's alignment verdict starts blank; carrying the previous run's over would
    # let a stale "pass" vouch for data it never saw.
    state.recording.syncStatus = "unknown"
    state.recording.syncSummary = ""
    state.recording.syncReportPath = ""
    state.recording.syncWarnings = []
    state.recorder_log_path = recorder_log_path
    _append_line(recorder_log_path, f"# command: {' '.join(command)}")
    _append_line(recorder_log_path, f"# cwd: {state.repo_root}")
    # Drop the previous session's log lines so the frontend doesn't show
    # stale output from a prior crashed recorder mixed in with the new one.
    state.recording.lastOutput = ""
    state.recording.recentOutput = []
    # Force the first viewer poll of the new session to send a demand heartbeat.
    state.recorder_preview_demand_sent_s = 0.0
    _set_all_device_states(state, "warning")
    if is_workstation and state.recording.backend == "sim":
        # Nothing physical is opened in a sim session. Leaving the hardware rows at "warning"
        # (the Connect default) would imply the gateway is still waiting on an FR3 that was
        # never asked to come up; the SpaceMouse row stays in the warning->running flow
        # because that device really is opened.
        for device in state.devices:
            if device.get("kind") in ("robot", "gripper", "camera"):
                device["state"] = "idle"
                device["detail"] = "simulated in MuJoCo"
    state.log("info", f"Started {state.profile} recorder pid={state.process.pid} log={recorder_log_path}")
    _start_output_reader(state, state.process)


def _start_episode(state: GatewayState) -> None:
    process = _ensure_recorder_running(state)
    if state.recording.state not in ("armed", "idle"):
        raise RuntimeError(f"Cannot start an episode while recorder is {state.recording.state}.")

    _write_recorder_stdin(process, "\n")
    state.recording.state = "recording"
    state.recording.frameIndex = 0
    state.recording.queueDepth = 1
    state.recording.message = "Episode start queued"
    _set_active_device_states(state, "running")
    state.log("info", "Queued episode start")


def _start_output_reader(state: GatewayState, process: subprocess.Popen[str]) -> None:
    thread = Thread(
        target=_read_process_output,
        args=(state, process),
        daemon=True,
        name=f"handheld-recorder-output-{process.pid}",
    )
    thread.start()


def _start_replay_output_reader(state: GatewayState, process: subprocess.Popen[str]) -> None:
    thread = Thread(
        target=_read_replay_process_output,
        args=(state, process),
        daemon=True,
        name=f"mujoco-replay-output-{process.pid}",
    )
    thread.start()


def _consume_recorder_output(state: GatewayState) -> None:
    """Apply queued recorder stdout lines under the lock, off the pipe path.

    The reader thread (``_read_process_output``) only drains the pipe and
    enqueues; this single long-lived consumer is the only place that takes
    ``state.lock`` for recorder output. If a snapshot briefly holds the lock the
    queue just buffers in memory — the pipe is still drained, so the recorder
    and camera workers never block on stdout. Started once at gateway startup.
    """
    q = state.recorder_output_queue
    while True:
        process, output = q.get()
        if not output:
            continue
        try:
            with state.lock:
                if state.process is not process:
                    continue  # line from an already-replaced recorder
                _apply_recorder_output(state, output)
        except Exception as exc:  # never let the consumer thread die
            state.log("warn", f"recorder output consumer error: {exc}")


def _path_stat_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _newest_shallow_file_signature(root: Path, pattern: str) -> tuple[str, int, int]:
    newest: tuple[str, int, int] = ("", 0, 0)
    try:
        candidates = root.glob(pattern)
        for path in candidates:
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_mtime_ns, path.name)
            newest_key = (newest[1], newest[0])
            if key > newest_key:
                newest = (path.name, stat.st_mtime_ns, stat.st_size)
    except OSError:
        pass
    return newest


def _latest_raw_episode_signature(episodes_root: Path) -> tuple[str, int, int, int, int, str, int, int]:
    latest_name = ""
    latest_path: Path | None = None
    try:
        with os.scandir(episodes_root) as it:
            for entry in it:
                try:
                    if entry.is_dir() and entry.name.startswith("episode_") and entry.name > latest_name:
                        latest_name = entry.name
                        latest_path = Path(entry.path)
                except OSError:
                    continue
    except OSError:
        return ("", 0, 0, 0, 0, "", 0, 0)
    if latest_path is None:
        return ("", 0, 0, 0, 0, "", 0, 0)
    ep_mtime_ns, ep_size = _path_stat_signature(latest_path)
    meta_mtime_ns, meta_size = _path_stat_signature(latest_path / "meta.json")
    mkv_name, mkv_mtime_ns, mkv_size = _newest_shallow_file_signature(latest_path, "*.mkv")
    return (latest_name, ep_mtime_ns, ep_size, meta_mtime_ns, meta_size, mkv_name, mkv_mtime_ns, mkv_size)


def _dataset_scan_signature(state: GatewayState) -> tuple:
    """Cheap fingerprint of dataset dirs plus completion sentinel files.

    A recorder may create the top-level dataset directory before the episode is
    complete, then later finalize ``meta/info.json`` and parquet files without
    changing the top-level directory mtime. Include those shallow sentinels so
    the cache notices the dataset becoming replayable while still avoiding the
    full per-episode scan used by ``_recorded_dataset_items``.
    """
    roots = [state.datasets_root, _task_exports_root(state), _training_views_root(state)]
    sig: list[tuple[Any, ...]] = []
    for root in roots:
        if root is None or not root.is_dir():
            continue
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if not entry.is_dir():
                            continue
                        entry_stat = entry.stat()
                    except OSError:
                        continue
                    dataset_root = Path(entry.path)
                    info_mtime_ns, info_size = _path_stat_signature(dataset_root / "meta" / "info.json")
                    data_mtime_ns, data_size = _path_stat_signature(dataset_root / "data")
                    parquet_name, parquet_mtime_ns, parquet_size = _newest_shallow_file_signature(
                        dataset_root / "data", "chunk-*/*.parquet"
                    )
                    episodes_mtime_ns, episodes_size = _path_stat_signature(dataset_root / "episodes")
                    raw_episode_sig: tuple[str, int, int, int, int, str, int, int]
                    if info_mtime_ns == 0 and parquet_mtime_ns == 0:
                        raw_episode_sig = _latest_raw_episode_signature(dataset_root / "episodes")
                    else:
                        raw_episode_sig = ("", 0, 0, 0, 0, "", 0, 0)
                    sig.append(
                        (
                            entry.path,
                            entry_stat.st_mtime_ns,
                            entry_stat.st_size,
                            info_mtime_ns,
                            info_size,
                            data_mtime_ns,
                            data_size,
                            parquet_name,
                            parquet_mtime_ns,
                            parquet_size,
                            episodes_mtime_ns,
                            episodes_size,
                            *raw_episode_sig,
                        )
                    )
        except OSError:
            continue
    sig.sort()
    return tuple(sig)


def _processing_scan_signature(state: GatewayState) -> tuple:
    """Fingerprint processing metadata without walking trajectory outputs.

    It intentionally includes ``meta/processing.json`` mtime/size so EE trajectory
    progress and completion can update the Processing page without invalidating
    the heavier recorded dataset / trajectory cache.
    """
    sig: list[tuple[str, int, int, int]] = []
    for root in _complete_dataset_candidates(state):
        try:
            root_mtime_ns = root.stat().st_mtime_ns
        except OSError:
            root_mtime_ns = 0
        meta_path = _processing_meta_path(root)
        try:
            meta_stat = meta_path.stat()
            meta_mtime_ns = meta_stat.st_mtime_ns
            meta_size = meta_stat.st_size
        except OSError:
            meta_mtime_ns = 0
            meta_size = 0
        sig.append((str(root), root_mtime_ns, meta_mtime_ns, meta_size))
    sig.sort()
    return tuple(sig)


def _processing_cache_has_inflight(state: GatewayState) -> bool:
    return any(
        item.get("status") in ("queued", "running")
        for item in state.cached_processing_items
    )


def _refresh_cached_processing_item(state: GatewayState, dataset_root: Path) -> None:
    with state.lock:
        attached = set(state.processing_processes.keys())
    item = _processing_item_from_dataset(dataset_root, attached_processes=attached, now_s=time.time())
    target = str(dataset_root)
    with state.lock:
        items = list(state.cached_processing_items)
        for index, existing in enumerate(items):
            if existing.get("path") == target:
                items[index] = item
                break
        else:
            items.insert(0, item)
        state.cached_processing_items = items
        state.processing_cache_ready = True


def _refresh_processing_cache(state: GatewayState) -> None:
    signature = _processing_scan_signature(state)
    with state.lock:
        attached = set(state.processing_processes.keys())
        unchanged = (
            state.processing_cache_ready
            and signature == state.processing_scan_signature
            and not _processing_cache_has_inflight(state)
        )
    if unchanged:
        return
    processing = _processing_items(state, attached_processes=attached)
    with state.lock:
        state.cached_processing_items = processing
        state.processing_cache_ready = True
        state.processing_scan_signature = signature


def _refresh_dataset_stats_cache(state: GatewayState) -> None:
    """Compute the expensive dataset scan OFF the lock and publish the result.

    ``_recorded_dataset_items`` / ``_read_recorded_trajectory`` walk the dataset
    tree (253 datasets / 298G / 600+ episodes on Thor => 4-12s of GIL-held CPU)
    and are read-only on ``state``. Even off ``state.lock`` that scan holds the
    GIL and starves every HTTP handler (snapshot/camera.jpg/cali-log polls),
    which is what made the UI freeze. So first take a cheap dir-mtime fingerprint
    and skip the whole walk when nothing changed; only rescan on a real change.
    """
    with state.lock:
        recorder_active = state.recording.pid is not None and bool(state.recording.datasetRoot)
    if recorder_active:
        # LeRobot/parquet files are not guaranteed readable until the recorder
        # process exits and closes/finalizes them. Live recording status is
        # updated from recorder stdout; defer historical dataset scans so the
        # UI does not surface transient "missing footer" parquet warnings.
        return
    signature = _dataset_scan_signature(state)
    with state.lock:
        selected_before = str(state.selected_replay_root.resolve()) if state.selected_replay_root is not None else ""
        episode_before = int(state.replay.episode or 0)
        unchanged = state.dataset_cache_ready and signature == state.dataset_scan_signature
    if unchanged:
        return
    items = _recorded_dataset_items(state)
    trajectory, meta = _read_recorded_trajectory(state)
    with state.lock:
        selected_now = str(state.selected_replay_root.resolve()) if state.selected_replay_root is not None else ""
        episode_now = int(state.replay.episode or 0)
        if selected_now != selected_before or episode_now != episode_before:
            # A user changed replay selection while this slow scan was running.
            # Publishing the stale meta would make _snapshot jump back to the old
            # latest dataset/episode, so drop this scan and let the next cycle
            # recompute for the current selection.
            return
        state.cached_recorded_datasets = items
        state.cached_trajectory = trajectory
        state.cached_trajectory_meta = meta
        state.dataset_cache_ready = True
        state.dataset_scan_signature = signature


def _dataset_stats_refresher(state: GatewayState, interval_s: float = 10.0) -> None:
    while True:
        try:
            _refresh_dataset_stats_cache(state)
        except Exception as exc:  # keep refreshing despite transient FS errors
            state.log("warn", f"dataset stats refresh failed: {exc}")
        try:
            _refresh_processing_cache(state)
        except Exception as exc:
            state.log("warn", f"processing cache refresh failed: {exc}")
        time.sleep(interval_s)


def _start_background_workers(state: GatewayState) -> None:
    """Launch the gateway's always-on helper threads (output consumer + dataset
    stats refresher). Idempotent-ish: intended to be called once from main()."""
    Thread(
        target=_consume_recorder_output, args=(state,),
        daemon=True, name="recorder-output-consumer",
    ).start()
    Thread(
        target=_dataset_stats_refresher, args=(state,),
        daemon=True, name="dataset-stats-refresher",
    ).start()


def _read_process_output(state: GatewayState, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    # Drain the recorder's stdout pipe as fast as the OS delivers lines and
    # NEVER take state.lock here. The old code did `with state.lock:
    # _apply_recorder_output(...)` per line; when a snapshot held the lock for
    # seconds (scanning 298G), this thread stalled, the 64KB pipe filled, and
    # the recorder + all camera worker subprocesses blocked on their stdout
    # writes — freezing preview production and the stale-preview watchdog.
    # Now the reader only appends to the log file and hands the line to a queue
    # that a dedicated consumer applies under the lock.
    for line in process.stdout:
        raw = line.rstrip("\n")
        _append_line(state.recorder_log_path, raw)
        output = raw.strip()
        if not output:
            continue
        state.recorder_output_queue.put((process, output))


def _read_replay_process_output(state: GatewayState, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output = line.strip()
        if not output:
            continue
        with state.lock:
            if state.replay_process is not process:
                return
            state.replay.lastOutput = output
            state.replay.message = output
            if state.replay_process_kind == "mujoco":
                _apply_mujoco_replay_output(state, output)
                state.log("info", f"mujoco replay: {output}")
            else:
                _append_real_replay_log(state, "replay", output)
                state.log("info", f"real replay: {output}")


def _set_mujoco_validation_metric(validation: dict[str, Any], key: str, value: str) -> None:
    try:
        validation[key] = float(value)
    except ValueError:
        return


def _apply_mujoco_replay_output(state: GatewayState, output: str) -> None:
    validation = state.replay.mujocoValidation or _new_mujoco_validation(state, status="running")
    result_match = re.search(
        r"mujoco_replay_result(?:=|\s+)status=(?P<status>\w+)\s+"
        r"completed_frames=(?P<completed>\d+)\s+total_frames=(?P<total>\d+)\s+"
        r"avg_pos_mm=(?P<avg_pos>[0-9.]+)\s+max_pos_mm=(?P<max_pos>[0-9.]+)\s+"
        r"avg_rot_deg=(?P<avg_rot>[0-9.]+)\s+max_rot_deg=(?P<max_rot>[0-9.]+)",
        output,
    )
    if result_match:
        validation["hasStructuredResult"] = True
        validation["completedFrames"] = int(result_match.group("completed"))
        validation["totalFrames"] = int(result_match.group("total"))
        _set_mujoco_validation_metric(validation, "avgPositionErrorMm", result_match.group("avg_pos"))
        _set_mujoco_validation_metric(validation, "maxPositionErrorMm", result_match.group("max_pos"))
        _set_mujoco_validation_metric(validation, "avgRotationErrorDeg", result_match.group("avg_rot"))
        _set_mujoco_validation_metric(validation, "maxRotationErrorDeg", result_match.group("max_rot"))
        validation["message"] = "MuJoCo replay metrics received"
    else:
        metric_patterns = (
            ("avgPositionErrorMm", r"平均位置误差:\s*([0-9.]+)\s*mm"),
            ("maxPositionErrorMm", r"最大位置误差:\s*([0-9.]+)\s*mm"),
            ("avgRotationErrorDeg", r"平均旋转误差:\s*([0-9.]+)\s*deg"),
            ("maxRotationErrorDeg", r"最大旋转误差:\s*([0-9.]+)\s*deg"),
        )
        for key, pattern in metric_patterns:
            metric_match = re.search(pattern, output)
            if metric_match:
                _set_mujoco_validation_metric(validation, key, metric_match.group(1))
    validation["updatedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state.replay.mujocoValidation = validation


def _mujoco_validation_is_for_active_episode(state: GatewayState, dataset_root: Path) -> bool:
    validation = state.replay.mujocoValidation or {}
    validation_episode = validation.get("episode")
    validation_fps = validation.get("fps")
    max_pos_mm, max_rot_deg = _mujoco_validation_thresholds(state.config)
    validation_pos_threshold = validation.get("maxPositionThresholdMm")
    validation_rot_threshold = validation.get("maxRotationThresholdDeg")
    return (
        validation.get("status") == "passed"
        and Path(str(validation.get("datasetRoot") or "")).resolve() == dataset_root.resolve()
        and int(validation_episode if validation_episode is not None else -1) == int(state.replay.episode)
        and int(validation_fps if validation_fps is not None else 0) == int(state.replay.fps or 30)
        and float(validation_pos_threshold if validation_pos_threshold is not None else -1.0) == max_pos_mm
        and float(validation_rot_threshold if validation_rot_threshold is not None else -1.0) == max_rot_deg
        and str(validation.get("cubeMode") or DEFAULT_MUJOCO_CUBE_MODE) == str(state.replay.mujocoCubeMode)
    )


def _distance_mm(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    keys = ("x", "y", "z")
    try:
        return math.sqrt(sum((float(a[key]) - float(b[key])) ** 2 for key in keys)) * 1000.0
    except (KeyError, TypeError, ValueError):
        return None


def _trajectory_contract_for_episode(state: GatewayState, dataset_root: Path) -> dict[str, Any]:
    replay = _replay_config(state.config)
    max_ee_step_mm = _float_config(replay, "trajectory_max_ee_step_mm", DEFAULT_REPLAY_MAX_EE_STEP_MM)
    default_gripper_step = (
        DEFAULT_WORKSTATION_REPLAY_MAX_GRIPPER_STEP
        if state.profile == "workstation"
        else DEFAULT_REPLAY_MAX_GRIPPER_STEP
    )
    max_gripper_step = _float_config(replay, "trajectory_max_gripper_step", default_gripper_step)
    min_z_value = replay.get("trajectory_min_z_m")
    max_z_value = replay.get("trajectory_max_z_m")
    min_z = float(min_z_value) if min_z_value is not None else None
    max_z = float(max_z_value) if max_z_value is not None else None

    timeline = _read_dataset_timeline(state, dataset_root, state.replay.episode)
    frames = timeline.get("frames") if isinstance(timeline, dict) else []
    frames = frames if isinstance(frames, list) else []
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    if not frames:
        failures.append("no timeline frames")

    cube_mode = str(state.replay.mujocoCubeMode or DEFAULT_MUJOCO_CUBE_MODE)
    selected_cubes = ("left", "right") if cube_mode == "both" else (cube_mode,)
    has_selected_cube_poses = any(
        isinstance(frame, dict)
        and any(isinstance((frame.get("cubePoses") or {}).get(cube), dict) for cube in selected_cubes)
        for frame in frames
    )
    if has_selected_cube_poses:
        pose_sequences = [
            [
                (frame.get("cubePoses") or {}).get(cube)
                for frame in frames
                if isinstance(frame, dict) and (frame.get("cubePoses") or {}).get(cube)
            ]
            for cube in selected_cubes
        ]
        expected_pose_count = len(frames) * len(selected_cubes)
    else:
        pose_sequences = [[frame.get("eePose") for frame in frames if isinstance(frame, dict) and frame.get("eePose")]]
        expected_pose_count = len(frames)
    poses = [pose for sequence in pose_sequences for pose in sequence]
    pose_count = len([pose for pose in poses if isinstance(pose, dict) and {"x", "y", "z"}.issubset(pose.keys())])
    if pose_count != expected_pose_count:
        failures.append(f"missing EE pose for {expected_pose_count - pose_count}/{expected_pose_count} cube-frames")
    checks.append({
        "name": "ee_pose_present",
        "status": "pass" if pose_count == expected_pose_count and frames else "fail",
        "value": pose_count,
        "cubeMode": cube_mode if has_selected_cube_poses else "dataset_default",
    })

    max_step = 0.0
    z_values: list[float] = []
    gripper_values: list[float] = []
    for sequence in pose_sequences:
        previous_pose: dict[str, Any] | None = None
        for pose in sequence:
            if not isinstance(pose, dict):
                continue
            if "z" in pose:
                try:
                    z_values.append(float(pose["z"]))
                except (TypeError, ValueError):
                    pass
            if pose.get("gripper") is not None:
                try:
                    gripper_values.append(float(pose["gripper"]))
                except (TypeError, ValueError):
                    pass
            if previous_pose is not None:
                step = _distance_mm(previous_pose, pose)
                if step is not None:
                    max_step = max(max_step, step)
            previous_pose = pose
    if max_step > max_ee_step_mm:
        failures.append(f"max EE step {max_step:.2f}mm > {max_ee_step_mm:.2f}mm")
    checks.append({"name": "max_ee_step", "status": "pass" if max_step <= max_ee_step_mm else "fail", "valueMm": max_step, "thresholdMm": max_ee_step_mm})

    if gripper_values:
        gripper_min = min(gripper_values)
        gripper_max = max(gripper_values)
        gripper_steps = [abs(curr - prev) for prev, curr in zip(gripper_values, gripper_values[1:], strict=False)]
        max_gripper_delta = max(gripper_steps) if gripper_steps else 0.0
        if gripper_min < 0.0 or gripper_max > 1.0:
            failures.append(f"gripper range [{gripper_min:.3f}, {gripper_max:.3f}] outside [0, 1]")
        if max_gripper_delta > max_gripper_step:
            failures.append(f"max gripper step {max_gripper_delta:.3f} > {max_gripper_step:.3f}")
        checks.append({
            "name": "gripper_range_step",
            "status": "pass" if 0.0 <= gripper_min <= gripper_max <= 1.0 and max_gripper_delta <= max_gripper_step else "fail",
            "min": gripper_min,
            "max": gripper_max,
            "maxStep": max_gripper_delta,
            "stepThreshold": max_gripper_step,
        })

    if z_values and (min_z is not None or max_z is not None):
        z_min = min(z_values)
        z_max = max(z_values)
        z_failures: list[str] = []
        if min_z is not None and z_min < min_z:
            z_failures.append(f"min Z {z_min:.4f}m < {min_z:.4f}m")
        if max_z is not None and z_max > max_z:
            z_failures.append(f"max Z {z_max:.4f}m > {max_z:.4f}m")
        failures.extend(z_failures)
        checks.append({"name": "z_bounds", "status": "pass" if not z_failures else "fail", "min": z_min, "max": z_max, "minThreshold": min_z, "maxThreshold": max_z})

    return {
        "status": "failed" if failures else "passed",
        "frames": len(frames),
        "checks": checks,
        "failures": failures,
    }


def _finish_mujoco_validation(state: GatewayState, exit_code: int | None) -> None:
    validation = state.replay.mujocoValidation or _new_mujoco_validation(state)
    validation["exitCode"] = exit_code
    validation["updatedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    max_pos = validation.get("maxPositionErrorMm")
    max_rot = validation.get("maxRotationErrorDeg")
    max_pos_threshold = float(validation.get("maxPositionThresholdMm") or DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM)
    max_rot_threshold = float(validation.get("maxRotationThresholdDeg") or DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG)
    completed = int(validation.get("completedFrames") or 0)
    total = int(validation.get("totalFrames") or 0)

    reasons: list[str] = []
    if exit_code != 0:
        reasons.append(f"exit code {exit_code}")
    if not validation.get("hasStructuredResult"):
        reasons.append("missing structured mujoco_replay_result")
    if max_pos is None or max_rot is None:
        reasons.append("missing replay metrics")
    else:
        if float(max_pos) > max_pos_threshold:
            reasons.append(f"max position error {float(max_pos):.2f}mm > {max_pos_threshold:.2f}mm")
        if float(max_rot) > max_rot_threshold:
            reasons.append(f"max rotation error {float(max_rot):.2f}deg > {max_rot_threshold:.2f}deg")
    if total > 0 and completed < total:
        reasons.append(f"incomplete episode {completed}/{total} frames")

    dataset_root_value = str(validation.get("datasetRoot") or "")
    try:
        dataset_root = Path(dataset_root_value).resolve()
    except OSError:
        dataset_root = None
    if dataset_root is None or not dataset_root_value:
        reasons.append("missing validation dataset root")
    else:
        try:
            contract = _trajectory_contract_for_episode(state, dataset_root)
        except Exception as exc:  # noqa: BLE001
            contract = {"status": "failed", "checks": [], "failures": [f"trajectory contract error: {exc}"]}
        validation["trajectoryContract"] = contract
        if contract.get("status") != "passed":
            reasons.extend(str(reason) for reason in contract.get("failures", []))

    # MuJoCo validation is the robot-replay gate. A failed validation means the selected
    # trajectory is not authorized for real motion, so surface it as a safety fault until the
    # operator reruns validation or explicitly uses the failed-validation override path.
    if reasons:
        validation["status"] = "failed"
        validation["message"] = "MuJoCo validation failed: " + "; ".join(reasons)
        state.replay.state = "aborted"
        state.replay.safety = "fault"
    else:
        validation["status"] = "passed"
        validation["message"] = (
            f"MuJoCo validation passed: max {float(max_pos):.2f}mm / {float(max_rot):.2f}deg "
            f"within {max_pos_threshold:.2f}mm / {max_rot_threshold:.2f}deg"
        )
        state.replay.state = "complete"
        state.replay.safety = "ready"
    state.replay.mujocoValidation = validation
    _refresh_mujoco_validation_current(state)
    if dataset_root is not None and _is_dataset_root(dataset_root):
        try:
            _write_validation_store(dataset_root, state.replay.mujocoValidation)
        except OSError as exc:
            state.log("warn", f"Failed to persist MuJoCo validation: {exc}")
    state.replay.message = validation["message"]


_RECORDER_NOISE_PREFIXES = (
    "[TLV_LOG_UPLOAD]",
    "[liwp][box] tlv ignored:",
    "GST_ARGUS:",
    "NvMMLite",
)


_RECORDER_FAILURE_KEYWORDS = (
    "connect exceeded global deadline",
    "persistent pipeline connect failed",
    "Auto-recover failed",
    "recover_argus.sh timed out",
    "connect stable window failed",
    "failed to reach PLAYING",
    "did not reach PLAYING",
    "connect() partial success",
    "stream(s) failed",
    "restart_stream",
    "NvBufSurfaceFromFd Failed",
    "dmabuf_fd -1",
    "Failed to create CaptureSession",
    "Argus Error Status",
    "Error turning on streaming",
    "TIMEOUT",
    "bus EOS",
    "CONSUMER: ERROR OCCURRED",
)


def _compact_recorder_summary(line: str, *, max_len: int = 240) -> str:
    summary = " ".join(line.strip().split())
    if len(summary) <= max_len:
        return summary
    return summary[: max(0, max_len - 3)].rstrip() + "..."


def _recorder_output_is_failure(line: str) -> bool:
    return line.startswith("ERROR:") or any(
        token in line for token in _RECORDER_FAILURE_KEYWORDS
    )


def _recorder_failure_summary(recording: RecordingStatus, *, max_len: int = 240) -> str:
    """Pick the most useful recorder failure line for process-exit UI text.

    Recorder stdout often ends with generic Argus chatter such as
    "CONSUMER: Waiting until producer is connected...".  For a failed
    process, prefer explicit protocol errors and then known Argus/GStreamer
    failure signatures from the recent-output ring.
    """
    lines = [line.strip() for line in recording.recentOutput if line.strip()]
    for line in reversed(lines):
        if line.startswith("ERROR:"):
            return _compact_recorder_summary(line, max_len=max_len)
    for line in reversed(lines):
        if _recorder_output_is_failure(line):
            return _compact_recorder_summary(line, max_len=max_len)
    if recording.lastOutput:
        return _compact_recorder_summary(recording.lastOutput, max_len=max_len)
    return ""


def _apply_recorder_output(state: GatewayState, output: str) -> None:
    if any(output.startswith(p) for p in _RECORDER_NOISE_PREFIXES):
        return
    if output.startswith("BOX_LIVE "):
        try:
            payload = json.loads(output.removeprefix("BOX_LIVE ").strip())
        except json.JSONDecodeError:
            state.log("warn", "recorder: malformed BOX_LIVE payload")
            return
        if isinstance(payload, dict):
            state.device_preview["box"] = {**payload, "active": True, "updatedAt": time.time()}
        return
    if output.startswith("BOX_DEVICES_JSON "):
        try:
            roster = json.loads(output.removeprefix("BOX_DEVICES_JSON ").strip())
        except json.JSONDecodeError:
            state.log("warn", "recorder: malformed BOX_DEVICES_JSON payload")
            return
        if isinstance(roster, list):
            _apply_box_roster(state, roster)
        return
    if output.startswith("TOUCHCALI_LOG ") or output.startswith("TOUCHCALI_DONE "):
        done = output.startswith("TOUCHCALI_DONE ")
        line = output.removeprefix("TOUCHCALI_DONE " if done else "TOUCHCALI_LOG ").strip()
        with state.box_touch_cali_lock:
            state.box_touch_cali_log.append({"ts": time.time(), "line": line, "done": done})
            del state.box_touch_cali_log[:-200]  # keep the buffer bounded
            if done:
                state.box_touch_cali_running = False
        return
    if output.startswith("CALI_LOG ") or output.startswith("CALI_DONE "):
        done = output.startswith("CALI_DONE ")
        line = output.removeprefix("CALI_DONE " if done else "CALI_LOG ").strip()
        with state.box_cali_lock:
            state.box_cali_log.append({"ts": time.time(), "line": line, "done": done})
            del state.box_cali_log[:-200]  # keep the buffer bounded
            if done:
                state.box_cali_running = False
        return
    if output.startswith("SYNC "):
        _apply_sync_audit_output(state, output.removeprefix("SYNC ").strip())
        return
    state.recording.lastOutput = output
    state.recording.message = output
    # Append to the ring buffer so the frontend can render every line the
    # recorder wrote between two snapshot polls. Without this, all the
    # rapid bursts (Phase 1 spawn × 11, Phase 2 wait_ready × 11, parallel
    # retry × N, Auto-recover decision) collapse to whatever line happened
    # to land last in the poll window.
    state.recording.recentOutput.append(output)
    if len(state.recording.recentOutput) > _RECORDER_OUTPUT_RING_CAP:
        del state.recording.recentOutput[
            : len(state.recording.recentOutput) - _RECORDER_OUTPUT_RING_CAP
        ]
    state.log("info", f"recorder: {output}")

    _mark_failed_camera_devices(state, _failed_camera_ids_from_recorder_output(output))
    if _recorder_output_is_failure(output):
        state.recording.state = "error"

    for prefix, kind in (
        ("Cameras:", "camera"),
        ("Tactiles:", "tactile"),
        ("Handheld grippers:", "handheld_gripper"),
        ("Box devices:", "box_collection"),
        # Workstation FR3 rig: the recorder reports which of these actually came up, so a
        # sim session never marks the physical arm or gripper as connected.
        ("Robots:", "robot"),
        ("Grippers:", "gripper"),
        ("Teleoperators:", "teleoperator"),
    ):
        if output.startswith(prefix):
            _mark_connected_devices(state, kind, output.removeprefix(prefix).strip())

    if output.startswith("Box rates:"):
        _apply_box_rates(state, output.removeprefix("Box rates:").strip())

    recorded_match = re.search(r"Recorded\s+(\d+)\s+frames", output)
    if recorded_match:
        recorded_frames = int(recorded_match.group(1))
        state.recording.frameIndex = recorded_frames
        if recorded_frames >= state.recording.targetFrames:
            state.recording.state = "review"
            state.recording.queueDepth = 0

    if output.startswith("Episode review:"):
        state.recording.state = "review"
        state.recording.queueDepth = 0

    if output.startswith("Start pose captured:"):
        state.recording.message = output

    saved_match = re.search(r"Total saved episodes:\s*(\d+)", output)
    if saved_match:
        saved_episodes = int(saved_match.group(1))
        state.recording.savedEpisodes = saved_episodes
        state.recording.episodeIndex = saved_episodes
        state.recording.frameIndex = 0
        state.recording.queueDepth = 0

    root_match = re.search(r"Dataset root:\s*(.+)", output)
    if root_match:
        dataset_root = root_match.group(1).strip()
        state.recording.datasetRoot = dataset_root

    ready_match = re.search(r"Episode\s+(\d+)\s+ready", output)
    if ready_match:
        state.recording.state = "armed"
        state.recording.frameIndex = 0
        state.recording.queueDepth = 0
        _set_active_device_states(state, "running")

    if "Episode saved" in output:
        state.recording.state = "saving"
        state.recording.message = "Episode saved; finalizing dataset"
    elif "Episode discarded" in output or "Recording stopped" in output:
        state.recording.state = "discarding"
        state.recording.frameIndex = 0
    elif "Input stream closed; stopping recording session." in output:
        state.recording.message = "Recorder input closed; finalizing dataset"


_SYNC_WARNING_CAP = 12


def _apply_sync_audit_output(state: GatewayState, payload: str) -> None:
    """Fold one ``SYNC ...`` recorder line into the recording status.

    The recorder emits three shapes: a ``status=... skew_p95_ms=...`` digest, a
    ``report=<path>`` pointer once the on-disk report exists, and ``WARN:``/``bias_...``
    detail lines. Warnings are kept as warnings on purpose -- an alignment violation must be
    loudly visible without tearing down a live recording session.
    """
    if payload.startswith("report="):
        state.recording.syncReportPath = payload.removeprefix("report=").strip()
        state.log("info", f"sync report written: {state.recording.syncReportPath}")
        return
    if payload.startswith("WARN:"):
        warning = payload.removeprefix("WARN:").strip()
        state.recording.syncWarnings.append(warning)
        del state.recording.syncWarnings[:-_SYNC_WARNING_CAP]
        state.log("warn", f"timestamp sync: {warning}")
        return
    if payload.startswith("audit unavailable"):
        state.recording.syncStatus = "unavailable"
        state.recording.syncSummary = payload
        state.log("warn", f"timestamp sync: {payload}")
        return
    if payload.startswith("bias_vs_arm_ms["):
        state.log("info", f"timestamp sync: {payload}")
        return

    fields = dict(
        item.split("=", 1) for item in payload.split() if "=" in item
    )
    status = str(fields.get("status") or "").strip()
    if status in ("pass", "fail"):
        # A new episode's verdict supersedes the previous one's warnings.
        if status == "pass":
            state.recording.syncWarnings = []
        state.recording.syncStatus = status
    state.recording.syncSummary = payload
    state.log("info" if status == "pass" else "warn", f"timestamp sync: {payload}")


def _apply_box_roster(state: GatewayState, roster: list[dict[str, Any]]) -> None:
    """Replace the box_collection device rows with the discovered roster.

    The recorder emits this once at Connect after broadcast discovery. We swap
    the static YAML-derived box rows for one row per (discovered box × sensor)
    so the Device Manager lists exactly the boxes on the subnet. A subsequent
    ``Box devices:`` line then marks each row live/error by id.
    """
    state.box_devices_roster = roster
    new_rows = _box_devices_from_roster(roster)
    if not new_rows:
        return
    # Drop existing box rows, keep every other device, append the discovered set.
    state.devices = [d for d in state.devices if d.get("kind") != "box_collection"]
    state.devices.extend(new_rows)
    labels = ", ".join(
        f"{e.get('box_id') or e.get('sn') or e.get('device_id')}" for e in roster
    )
    state.log("info", f"discovered {len(roster)} BOX device(s): {labels}")


def _apply_box_rates(state: GatewayState, rates_str: str) -> None:
    """Parse ``box_imu=200, box_gripper=200, ...`` and update device fps."""
    for part in rates_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        sid, hz_str = part.split("=", 1)
        sid = sid.strip()
        try:
            hz = int(round(float(hz_str)))
        except (TypeError, ValueError):
            continue
        for device in state.devices:
            if device.get("kind") == "box_collection" and device.get("id") == sid:
                device["fps"] = hz


def _dataset_arg_for_container_replay(repo_root: Path, dataset_root: Path) -> str:
    resolved_root = repo_root.resolve()
    resolved_dataset = dataset_root.resolve()
    try:
        return str(resolved_dataset.relative_to(resolved_root))
    except ValueError:
        return str(resolved_dataset)


def _active_replay_dataset_root(state: GatewayState) -> Path:
    requested = state.selected_replay_root or _resolve_known_dataset(state, state.replay.datasetRoot or state.replay.dataset)
    if requested is not None:
        return requested.resolve()
    candidates = _replay_dataset_candidates(state)
    if not candidates:
        raise RuntimeError("No recorded dataset is available for MuJoCo replay.")
    return candidates[0].resolve()


FR3_MUJOCO_REPLAY_DIR = "fr3_mujoco_replay"


def _fr3_mujoco_replay_report_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "derived" / FR3_MUJOCO_REPLAY_DIR / f"episode_{int(episode):06d}.json"


def _fr3_mujoco_replay_video_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "derived" / FR3_MUJOCO_REPLAY_DIR / f"episode_{int(episode):06d}.mp4"


def _mujoco_preview_report_path(dataset_root: Path, episode: int, cube_mode: str) -> Path:
    return (
        dataset_root
        / "derived"
        / DEFAULT_TRAJ_SIDECAR_NAME
        / f"mujoco_preview.{cube_mode}.episode_{int(episode):06d}.json"
    )


def _fr3_mujoco_preview_payload(dataset_root: Path, episode: int) -> dict[str, Any] | None:
    """The FR3 replay report, shaped as the inspector's preview payload.

    The two routes disagree about what a replay report is: Thor's is per-cube and carries the
    per-robot frame lists the inspector never reads, while the FR3 runtime writes one report per
    episode. Adapting here keeps that difference out of the frontend, which only needs to know
    whether there is a video to show.
    """
    report = _load_json_file(_fr3_mujoco_replay_report_path(dataset_root, episode))
    if not report:
        return None
    video_path = _fr3_mujoco_replay_video_path(dataset_root, episode)
    return {
        "schema_version": int(report.get("schema_version") or 1),
        "dataset_root": str(dataset_root),
        "episode_index": int(report.get("episode", episode)),
        "fps": int(report.get("fps") or 0),
        "native_video_path": str(video_path) if video_path.is_file() else "",
        "status": str(report.get("status") or ""),
        "max_position_error_mm": report.get("max_position_error_mm"),
        "max_rotation_error_deg": report.get("max_rotation_error_deg"),
        # No cubes on this rig, and no second arm to place beside the first.
        "robots": {},
        "robot_spacing_m": 0.0,
    }


def _mujoco_preview_video_path(dataset_root: Path, episode: int, cube_mode: str) -> Path:
    return (
        dataset_root
        / "derived"
        / DEFAULT_TRAJ_SIDECAR_NAME
        / f"mujoco_preview.{cube_mode}.episode_{int(episode):06d}.mp4"
    )


def _mujoco_replay_python(state: GatewayState) -> Path:
    # Thor's gateway venv intentionally stays small and does not contain
    # MuJoCo. The FR3 inference environment carries MuJoCo plus the hardware-
    # equivalent IK backend used by cube replay. Keep local development on the
    # repository venv when the Thor-specific environment is absent.
    thor_python = Path("/home/nvidia/Code/infer/.venv-fr3/bin/python3")
    return thor_python if thor_python.is_file() else _venv_python(state.repo_root)


def _fr3_mujoco_replay_command(state: GatewayState, dataset_root: Path) -> list[str]:
    """Workstation MuJoCo validation: replay the recorded FR3 EE command stream.

    The Thor route tracks AprilTag cubes through a scene; a workstation dataset has no cubes
    and instead carries the arm's own absolute EE actions, so validation means feeding those
    back through the simulated arm and scoring the tracking error.
    """
    return [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_gui_replay_runtime.py"),
        "--dataset",
        str(dataset_root),
        "--episode",
        str(state.replay.episode),
        "--config-path",
        str(state.config_path),
        "--fps",
        str(_replay_fps(state, dataset_root)),
        "--max-position-error-mm",
        str(DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM),
        "--max-rotation-error-deg",
        str(DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG),
        "--ik-orientation-weight",
        str(DEFAULT_WORKSTATION_REPLAY_IK_ORIENTATION_WEIGHT),
        # Without this the run produces numbers and nothing to look at: the inspector's video
        # panel stays on its placeholder, which reads as "the replay did not happen".
        "--render-video",
        str(_fr3_mujoco_replay_video_path(dataset_root, state.replay.episode)),
    ]


def _mujoco_replay_command(state: GatewayState, dataset_root: Path, cube_mode: str | None = None) -> list[str]:
    if state.profile == "workstation":
        return _fr3_mujoco_replay_command(state, dataset_root)
    selected_cube_mode = str(cube_mode or state.replay.mujocoCubeMode or DEFAULT_MUJOCO_CUBE_MODE)
    if selected_cube_mode not in MUJOCO_CUBE_MODES:
        raise ValueError(f"MuJoCo cube mode must be one of {MUJOCO_CUBE_MODES}, got {selected_cube_mode!r}")
    report_path = _mujoco_preview_report_path(dataset_root, state.replay.episode, selected_cube_mode)
    video_path = _mujoco_preview_video_path(dataset_root, state.replay.episode, selected_cube_mode)
    command = [
        str(_mujoco_replay_python(state)),
        str(
            state.repo_root
            / "third_party"
            / "opencv_kalibr"
            / "fr3_data_collection_replay"
            / "replay_cube_pose_in_robot_base_mujoco.py"
        ),
        "--dataset-root",
        str(dataset_root),
        "--cube",
        selected_cube_mode,
        "--episode-index",
        str(state.replay.episode),
        "--fps",
        str(state.replay.fps or 30),
        "--pose-prefix",
        "state",
        "--ik-solver",
        "hardware",
        "--robot-spacing-m",
        str(DEFAULT_MUJOCO_ROBOT_SPACING_M),
        "--report-json",
        str(report_path),
        "--render-video",
        str(video_path),
        "--no-viewer",
    ]
    return command


def _approve_mujoco_report(state: GatewayState, cube_mode: str) -> None:
    """Re-evaluate a rendered report; this never overrides failed metrics."""
    if state.replay_process is not None and state.replay_process.poll() is None:
        raise RuntimeError("Wait for the active replay process to finish before checking MuJoCo results.")
    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")
    selected_cube_mode = str(cube_mode).strip().lower()
    if selected_cube_mode not in MUJOCO_CUBE_MODES:
        raise ValueError(f"MuJoCo cube mode must be one of {MUJOCO_CUBE_MODES}, got {selected_cube_mode!r}")

    report_path = _mujoco_preview_report_path(dataset_root, state.replay.episode, selected_cube_mode)
    video_path = _mujoco_preview_video_path(dataset_root, state.replay.episode, selected_cube_mode)
    report = _load_json_file(report_path)
    if not report:
        raise RuntimeError(f"Run MuJoCo {selected_cube_mode} first; no report exists for this episode.")
    if not video_path.is_file():
        raise RuntimeError(f"MuJoCo report exists but its native video is missing: {video_path}")
    if Path(str(report.get("dataset_root") or "")).resolve() != dataset_root.resolve():
        raise RuntimeError("MuJoCo report belongs to a different dataset.")
    if int(report.get("episode_index", -1)) != int(state.replay.episode):
        raise RuntimeError("MuJoCo report belongs to a different episode.")
    if str(report.get("cube_mode") or "") != selected_cube_mode:
        raise RuntimeError("MuJoCo report belongs to a different cube selection.")
    if int(report.get("fps", 0)) != int(state.replay.fps or 30):
        raise RuntimeError("MuJoCo report FPS does not match the selected episode.")

    selected_cubes = ("left", "right") if selected_cube_mode == "both" else (selected_cube_mode,)
    sidecar_dir = dataset_root / "derived" / DEFAULT_TRAJ_SIDECAR_NAME
    newest_input_mtime = max((sidecar_dir / f"state_action.{cube}.csv").stat().st_mtime for cube in selected_cubes)
    if report_path.stat().st_mtime < newest_input_mtime or video_path.stat().st_mtime < newest_input_mtime:
        raise RuntimeError("MuJoCo output is older than the EE trajectory. Run MuJoCo again before passing it.")

    robots = report.get("robots") if isinstance(report.get("robots"), dict) else {}
    robot_rows = [robots.get(cube) for cube in selected_cubes]
    if any(not isinstance(row, dict) for row in robot_rows):
        raise RuntimeError("MuJoCo report is incomplete for the selected cube mode.")
    frame_counts = [len(row.get("frames") or []) for row in robot_rows]
    metric_rows = [row.get("metrics") if isinstance(row.get("metrics"), dict) else {} for row in robot_rows]
    if any(not metrics for metrics in metric_rows):
        raise RuntimeError("MuJoCo report is missing error metrics.")
    # totalFrames is scoped to the selected episode; recordedFrames is the
    # dataset-wide total and would incorrectly reject multi-episode datasets.
    total_frames = int(state.replay.totalFrames or 0) * len(selected_cubes)
    completed_frames = sum(frame_counts)
    weighted_denominator = max(completed_frames, 1)

    state.replay.mujocoCubeMode = selected_cube_mode
    validation = _new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=state.replay.episode,
        message="Checking saved MuJoCo metrics and trajectory contract.",
    )
    validation.update({
        "hasStructuredResult": True,
        "completedFrames": completed_frames,
        "totalFrames": total_frames,
        "avgPositionErrorMm": sum(
            float(metrics["avg_position_error_mm"]) * count for metrics, count in zip(metric_rows, frame_counts, strict=True)
        ) / weighted_denominator,
        "maxPositionErrorMm": max(float(metrics["max_position_error_mm"]) for metrics in metric_rows),
        "avgRotationErrorDeg": sum(
            float(metrics["avg_rotation_error_deg"]) * count for metrics, count in zip(metric_rows, frame_counts, strict=True)
        ) / weighted_denominator,
        "maxRotationErrorDeg": max(float(metrics["max_rotation_error_deg"]) for metrics in metric_rows),
        "cubeMode": selected_cube_mode,
    })
    state.replay.mujocoValidation = validation
    _finish_mujoco_validation(state, 0)
    state.log("info" if validation.get("status") == "passed" else "warn", validation["message"])


def _validated_robot_ip(raw: str, label: str) -> str:
    value = str(raw).strip()
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid IP address, got {value!r}") from exc
    if parsed.version != 4:
        raise ValueError(f"{label} must be an IPv4 address, got {value!r}")
    return value


def _fr3_real_replay_command(state: GatewayState, dataset_root: Path, robot_ip: str) -> list[str]:
    """Drive the hardware FR3 through the episode's own recorded EE actions.

    The same runtime as the MuJoCo gate, with ``--backend real``. Sharing it is the point: a real
    replay that rebuilt its trajectory by different code from the run that cleared it would have
    validated one thing and executed another.
    """
    return [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_gui_replay_runtime.py"),
        "--backend",
        "real",
        "--dataset",
        str(dataset_root),
        "--episode",
        str(state.replay.episode),
        "--config-path",
        str(state.config_path),
        "--fps",
        str(_replay_fps(state, dataset_root)),
        "--robot-ip",
        str(robot_ip),
        "--max-position-error-mm",
        str(DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM),
        "--max-rotation-error-deg",
        str(DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG),
        "--ik-orientation-weight",
        str(DEFAULT_WORKSTATION_REPLAY_IK_ORIENTATION_WEIGHT),
        "--settle-steps",
        str(DEFAULT_WORKSTATION_REAL_SETTLE_STEPS),
        "--settle-tolerance-mm",
        str(DEFAULT_WORKSTATION_REAL_SETTLE_TOLERANCE_MM),
    ]


def _real_replay_command(
    state: GatewayState,
    dataset_root: Path,
    cube_mode: str,
    robot_ip: str,
    end_effector_mode: str = "corenetic_gripper_ee",
) -> list[str]:
    if state.profile == "workstation":
        return _fr3_real_replay_command(state, dataset_root, robot_ip)
    # run/deploy.sh hosts the gateway on Thor itself. Calling the developer-side
    # run_replay_cube_pose_on_thor.sh wrapper here would make Thor SSH back into
    # itself and depend on an unrelated self-SSH key. Invoke the same underlying
    # replay runtime locally with the exact selected sidecar and episode.
    csv_path = (
        dataset_root
        / "derived"
        / DEFAULT_TRAJ_SIDECAR_NAME
        / f"state_action.{cube_mode}.csv"
    )
    command = [
        str(_mujoco_replay_python(state)),
        str(
            state.repo_root
            / "third_party"
            / "opencv_kalibr"
            / "fr3_data_collection_replay"
            / "replay_cube_pose_in_robot_base.py"
        ),
        "--config_path",
        str(
            state.repo_root
            / "third_party"
            / "opencv_kalibr"
            / "fr3_data_collection_replay"
            / "replay_cube_pose_in_robot_base.thor.yaml"
        ),
        "--input.source=csv",
        f"--input.csv_path={csv_path}",
        "--input.pose_prefix=state",
        f"--input.dataset_pose_name={cube_mode}",
        f"--robot.robot_ip={robot_ip}",
        f"--replay.episode_index={int(state.replay.episode)}",
        "--replay.initial_pose_mode=current",
        "--replay.fail_on_unreached_initial_pose=true",
        f"--end_effector.mode={'fr3_ee' if end_effector_mode == 'fr3_ee' else 'robot_config'}",
    ]
    if end_effector_mode == "pika_gripper_ee":
        robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
        urdf_path, _sim_xml_path = _fr3_pika_asset_paths(state.repo_root)
        command.extend(
            [
                "--robot.gripper_backend=pika",
                f"--robot.gripper_port={robot.get('gripper_port') or '/dev/ttyUSB0'}",
                "--robot.allow_mock_gripper=false",
                f"--robot.urdf_path={urdf_path}",
                f"--robot.target_frame_name={_fr3_target_frame_name(state)}",
            ]
        )
    return command


def _real_robot_ip(state: GatewayState) -> str:
    """The robot this profile drives, from its own config -- with no fallback.

    There used to be one: 192.168.1.208, which is the *DAS* rig's arm (see
    tools/fr3/fr3_das_replay_real.py and the rest of the fr3_*_das_* tooling). Reaching it
    meant the loaded profile had not declared a robot, and the gateway would then send real
    motion commands to a different rig's address -- succeeding if something answered there.
    A profile that drives a robot names it; one that does not should say so here.
    """
    replay = _replay_config(state.config)
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    robot_ip = replay.get("robot_ip") or robot.get("robot_ip")
    if not robot_ip:
        raise ValueError(
            "No robot IP in this profile's config: set robot.robot_ip (or replay.robot_ip) in "
            f"{state.config_path}."
        )
    return str(robot_ip)


def _real_preflight_command(state: GatewayState, robot_ip: str | None = None) -> list[str]:
    replay = _replay_config(state.config)
    config_path = (
        state.config_path
        if state.profile == "workstation"
        else state.repo_root
        / "third_party"
        / "opencv_kalibr"
        / "fr3_data_collection_replay"
        / "replay_cube_pose_in_robot_base.thor.yaml"
    )
    command = [
        str(
            _venv_python(state.repo_root, prefer_fr3=True)
            if state.profile == "workstation"
            else _mujoco_replay_python(state)
        ),
        str(state.repo_root / "tools" / "fr3" / "fr3_record_preflight.py"),
        f"--workspace={state.repo_root}",
        f"--config-path={config_path}",
        f"--robot-ip={robot_ip or _real_robot_ip(state)}",
    ]
    if _bool_config(replay, "real_preflight_skip_host_imports", True):
        command.append("--skip-host-imports")
    if _bool_config(replay, "real_preflight_skip_hikrobot", True):
        command.append("--skip-hikrobot")
    if _bool_config(replay, "real_preflight_skip_gripper", True):
        command.append("--skip-gripper")
    if _bool_config(replay, "real_preflight_skip_arm", False):
        command.append("--skip-arm")
    if _bool_config(replay, "real_preflight_skip_ping", False):
        command.append("--skip-ping")
    return command


def _append_real_replay_log(state: GatewayState, stage: str, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [{stage}] {message}"
    state.replay.realReplayLog.append(line)
    del state.replay.realReplayLog[:-120]


def _real_preflight_env(state: GatewayState) -> dict[str, str]:
    env = _tool_env(state.repo_root)
    python_path = _mujoco_replay_python(state)
    venv_root = python_path.parent.parent
    cmeel_libs = sorted((venv_root / "lib").glob("python*/site-packages/cmeel.prefix/lib"))
    ld_entries = [str(path) for path in cmeel_libs]
    ld_entries.extend(path for path in ("/usr/local/lib", "/opt/MVS/lib/64", "/opt/MVS/lib") if Path(path).exists())
    if env.get("LD_LIBRARY_PATH"):
        ld_entries.append(env["LD_LIBRARY_PATH"])
    if ld_entries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(ld_entries))
    return env


def _run_real_preflight(state: GatewayState, robot_ips: Iterable[str] | None = None) -> None:
    replay = _replay_config(state.config)
    if not _bool_config(replay, "real_preflight_enabled", True):
        state.log("warn", "Real-robot preflight skipped by replay.real_preflight_enabled=false")
        return
    timeout_s = _float_config(replay, "real_preflight_timeout_s", DEFAULT_REAL_PREFLIGHT_TIMEOUT_S)
    targets = list(dict.fromkeys(robot_ips or [_real_robot_ip(state)]))
    for robot_ip in targets:
        command = _real_preflight_command(state, robot_ip)
        state.replay.message = f"Running real-robot preflight checks for {robot_ip}"
        _append_real_replay_log(state, "preflight", f"checking robot {robot_ip}")
        _append_real_replay_log(state, "preflight", f"config={next(arg.split('=', 1)[1] for arg in command if arg.startswith('--config-path='))}")
        try:
            result = subprocess.run(
                command,
                cwd=state.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_real_preflight_env(state),
                timeout=max(timeout_s, 1.0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _append_real_replay_log(state, "error", f"preflight timed out after {timeout_s:.1f}s")
            raise RuntimeError(f"Real-robot preflight timed out for {robot_ip} after {timeout_s:.1f}s") from exc
        output_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        for line in output_lines:
            _append_real_replay_log(state, "preflight", line)
        if output_lines:
            state.replay.lastOutput = output_lines[-1]
        if result.returncode != 0:
            details = output_lines[-1] if output_lines else f"exit code {result.returncode}"
            raise RuntimeError(f"Real-robot preflight failed for {robot_ip}: {details}")
        _append_real_replay_log(state, "preflight", f"robot {robot_ip} passed")


def _failed_mujoco_validation_is_override_eligible(
    state: GatewayState,
    dataset_root: Path,
    cube_mode: str,
) -> bool:
    validation = state.replay.mujocoValidation or {}
    completed = int(validation.get("completedFrames") or 0)
    total = int(validation.get("totalFrames") or 0)
    return (
        validation.get("status") == "failed"
        and int(validation.get("exitCode") if validation.get("exitCode") is not None else -1) == 0
        and bool(validation.get("hasStructuredResult"))
        and validation.get("maxPositionErrorMm") is not None
        and validation.get("maxRotationErrorDeg") is not None
        and total > 0
        and completed >= total
        and Path(str(validation.get("datasetRoot") or "")).resolve() == dataset_root.resolve()
        and int(validation.get("episode", -1)) == int(state.replay.episode)
        and int(validation.get("fps", 0)) == int(state.replay.fps or 30)
        and str(validation.get("cubeMode") or "") == cube_mode
    )


def _require_mujoco_validation(
    state: GatewayState,
    *,
    cube_mode: str | None = None,
    allow_failed_override: bool = False,
) -> Path:
    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")
    if _dataset_kind(state, dataset_root) == "exported":
        raise RuntimeError(
            "Real-robot replay is disabled for exported datasets because exported action is next-frame observation.state, not a verified robot command stream."
        )
    selected_cube = str(cube_mode or state.replay.mujocoCubeMode)
    if not _mujoco_validation_is_for_active_episode(state, dataset_root) and not (
        allow_failed_override
        and _failed_mujoco_validation_is_override_eligible(state, dataset_root, selected_cube)
    ):
        validation = state.replay.mujocoValidation or {}
        message = str(validation.get("message") or "Run MuJoCo replay successfully before real-robot replay.")
        raise RuntimeError(f"MuJoCo validation required for this dataset/episode: {message}")
    return dataset_root


def _require_replay_dataset(state: GatewayState) -> Path:
    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")
    return dataset_root


def _mujoco_recommendation_suffix(state: GatewayState, dataset_root: Path) -> str:
    if _mujoco_validation_is_for_active_episode(state, dataset_root):
        return "current MuJoCo validation is available"
    return "MuJoCo replay is strongly recommended before Preflight and still required before Real Robot"


def _preflight_replay(state: GatewayState) -> None:
    dataset_root = _require_replay_dataset(state)
    _run_real_preflight(state)
    state.replay.state = "armed"
    state.replay.safety = "ready"
    state.replay.message = (
        f"Preflight gate passed for episode {state.replay.episode} from {dataset_root.name}; "
        f"hardware checks passed; {_mujoco_recommendation_suffix(state, dataset_root)}"
    )
    state.log("info", state.replay.message)


def _start_dry_run_replay(state: GatewayState) -> None:
    dataset_root = _require_replay_dataset(state)
    state.selected_replay_root = dataset_root
    state.replay.state = "dry_run"
    state.replay.safety = "ready"
    state.replay.pid = None
    state.replay.frameIndex = 0
    state.replay.datasetRoot = str(dataset_root)
    state.replay.dataset = str(dataset_root)
    state.replay.message = (
        f"Dry-run replay ready for {dataset_root.name} episode {state.replay.episode}; "
        f"{_mujoco_recommendation_suffix(state, dataset_root)}"
    )
    state.log("info", state.replay.message)


def _start_mujoco_replay(state: GatewayState, cube_mode: str = DEFAULT_MUJOCO_CUBE_MODE) -> None:
    if state.replay_process is not None and state.replay_process.poll() is None:
        state.replay.message = "MuJoCo replay is already running"
        return

    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")
    cube_mode = str(cube_mode).strip().lower()
    if cube_mode not in MUJOCO_CUBE_MODES:
        raise ValueError(f"MuJoCo cube mode must be one of {MUJOCO_CUBE_MODES}, got {cube_mode!r}")
    if state.profile != "workstation":
        # Thor validates against AprilTag cube trajectories generated from the camera stream.
        # A workstation dataset carries the arm's own EE actions instead, so there is no
        # sidecar to require -- gating on one would make FR3 replay permanently unreachable.
        required_cubes = ("left", "right") if cube_mode == "both" else (cube_mode,)
        episode_poses = _read_sidecar_cube_poses(dataset_root, state.replay.episode)
        missing_cubes = [cube for cube in required_cubes if not episode_poses.get(cube)]
        if missing_cubes:
            raise RuntimeError(
                f"Selected dataset episode {state.replay.episode} has no valid generated EE trajectory for: "
                f"{', '.join(missing_cubes)}. "
                "Run Generate EE Trajectory first."
            )

    state.replay.mujocoCubeMode = cube_mode
    command = _mujoco_replay_command(state, dataset_root, cube_mode)
    state.replay.mujocoValidation = _new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=state.replay.episode,
        message="MuJoCo replay is running; real-robot replay remains locked until metrics pass.",
    )
    process_env = _tool_env(state.repo_root)
    process_env.setdefault("MUJOCO_GL", "egl")
    process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=process_env,
        start_new_session=True,
    )
    state.replay_process = process
    state.replay_process_kind = "mujoco"
    state.replay_started_at_s = time.monotonic()
    state.selected_replay_root = dataset_root
    state.replay.state = "sim_replay"
    state.replay.safety = "locked"
    state.replay.pid = process.pid
    state.replay.frameIndex = 0
    state.replay.datasetRoot = str(dataset_root)
    state.replay.dataset = str(dataset_root)
    # "cube mode" is a Thor concept; a workstation replay tracks the arm's own EE command
    # stream, so naming a cube there would describe something that does not exist.
    replay_subject = "EE command" if state.profile == "workstation" else cube_mode
    state.replay.message = (
        f"MuJoCo {replay_subject} replay started for {dataset_root.name} episode {state.replay.episode}; "
        "waiting for validation metrics"
    )
    state.log(
        "info",
        f"Started MuJoCo replay pid={process.pid} dataset={dataset_root} episode={state.replay.episode} cube={cube_mode}",
    )
    _start_replay_output_reader(state, process)


def _realsense_preview_token(camera_key: str) -> str:
    token = re.sub(r"[^A-Za-z0-9-]+", "_", str(camera_key).strip())
    return token.strip("_") or "default"


def _realsense_preview_paths(state: GatewayState, camera_key: str = "default") -> tuple[Path, Path]:
    token = _realsense_preview_token(camera_key)
    root = Path("/tmp") / f"lerobot_realsense_replay_{os.getpid()}_{token}"
    return Path(str(root) + ".jpg"), Path(str(root) + ".json")


def _realsense_preview_python(state: GatewayState) -> Path:
    candidate = state.repo_root / "third_party" / "opencv_kalibr" / ".venv" / "bin" / "python3"
    return (
        candidate
        if candidate.is_file()
        else _venv_python3(state.repo_root, prefer_fr3=state.profile == "workstation")
    )


def _replay_realsense_camera_matches(state: GatewayState, dataset_root: Path | None = None) -> list[dict[str, Any]]:
    root = dataset_root or state.selected_replay_root or _resolve_known_dataset(state, state.replay.datasetRoot or state.replay.dataset)
    if root is None:
        return []
    try:
        camera_keys = _camera_keys(_load_dataset_info(root))
    except Exception:
        return []
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    configured = robot.get("cameras") if isinstance(robot.get("cameras"), dict) else {}
    matches: list[dict[str, Any]] = []
    for camera_key in camera_keys:
        stem = _camera_stem_from_key(camera_key)
        config_key = camera_key if camera_key in configured else stem
        camera_cfg = configured.get(config_key) if isinstance(configured.get(config_key), dict) else None
        if not camera_cfg or str(camera_cfg.get("type", "")).lower() != "intelrealsense":
            continue
        serial = str(camera_cfg.get("serial_number_or_name") or "").strip()
        if not serial:
            continue
        matches.append(
            {
                "cameraKey": camera_key,
                "configKey": str(config_key),
                "serial": serial,
                "width": int(camera_cfg.get("width") or 640),
                "height": int(camera_cfg.get("height") or 480),
                "fps": min(int(camera_cfg.get("fps") or 15), 15),
            }
        )
    return matches


def _terminate_process_group(process: subprocess.Popen[str], timeout_s: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout_s)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def _stop_realsense_preview(state: GatewayState) -> None:
    processes = dict(state.realsense_preview_processes)
    if state.realsense_preview_process is not None:
        processes.setdefault("default", state.realsense_preview_process)
    for process in processes.values():
        _terminate_process_group(process)
    state.realsense_preview_process = None
    state.realsense_preview_processes = {}


def _start_realsense_preview(state: GatewayState, dataset_root: Path | None = None) -> None:
    matches = _replay_realsense_camera_matches(state, dataset_root)
    running_keys = {key for key, process in state.realsense_preview_processes.items() if process.poll() is None}
    desired_keys = {str(match["cameraKey"]) for match in matches}
    if running_keys == desired_keys and running_keys:
        return
    _stop_realsense_preview(state)
    for match in matches:
        camera_key = str(match["cameraKey"])
        image_path, status_path = _realsense_preview_paths(state, camera_key)
        image_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        command = [
            str(_realsense_preview_python(state)),
            str(state.repo_root / "tools" / "data_collection_gui" / "realsense_live_preview.py"),
            "--output", str(image_path),
            "--status", str(status_path),
            "--serial", str(match["serial"]),
            "--width", str(match["width"]),
            "--height", str(match["height"]),
            "--fps", str(match["fps"]),
        ]
        state.realsense_preview_processes[camera_key] = subprocess.Popen(
            command,
            cwd=state.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _realsense_preview_status(state: GatewayState) -> dict[str, Any]:
    matches = _replay_realsense_camera_matches(state)
    by_key = {str(match["cameraKey"]): match for match in matches}
    cameras: list[dict[str, Any]] = []
    for camera_key, match in by_key.items():
        _image_path, status_path = _realsense_preview_paths(state, camera_key)
        payload = _load_json_file(status_path)
        process = state.realsense_preview_processes.get(camera_key)
        process_running = process is not None and process.poll() is None
        if not payload:
            payload = {
                "available": None,
                "running": process_running,
                "error": "Waiting for RealSense detection" if process_running else "Preview starts with real-robot replay",
            }
        payload = dict(payload)
        payload.update({"cameraKey": camera_key, "configKey": match["configKey"], "serial": match["serial"]})
        payload["running"] = bool(payload.get("running")) and process_running
        cameras.append(payload)
    if not cameras:
        return {
            "available": None,
            "running": False,
            "cameras": [],
            "error": "No dataset camera stream matches a configured RealSense camera.",
        }
    first = cameras[0]
    return {
        "available": any(camera.get("available") is True for camera in cameras),
        "running": any(bool(camera.get("running")) for camera in cameras),
        "serial": first.get("serial"),
        "width": first.get("width"),
        "height": first.get("height"),
        "fps": first.get("fps"),
        "error": "; ".join(str(camera.get("error")) for camera in cameras if camera.get("error")),
        "cameras": cameras,
    }


def _start_real_replay(
    state: GatewayState,
    cube_mode: str = "right",
    robot_ip: str = "",
    end_effector_mode: str = "corenetic_gripper_ee",
    override_mujoco_failure: bool = False,
) -> None:
    if state.replay_process is not None and state.replay_process.poll() is None:
        state.replay.message = "Replay process is already running"
        return

    cube_mode = str(cube_mode).strip().lower()
    if cube_mode not in ("left", "right"):
        raise ValueError(f"Real replay cube mode must be left or right, got {cube_mode!r}")
    end_effector_mode = str(end_effector_mode).strip().lower()
    if end_effector_mode not in ("pika_gripper_ee", "corenetic_gripper_ee", "fr3_ee"):
        raise ValueError("Real replay end effector must be pika_gripper_ee, corenetic_gripper_ee, or fr3_ee")
    workstation = state.profile == "workstation"
    state.replay.realReplayLog = []
    _append_real_replay_log(
        state,
        "request",
        f"{'source=recorded EE actions' if workstation else f'cube={cube_mode}'} "
        f"episode={state.replay.episode} robot={robot_ip or _real_robot_ip(state)} target={end_effector_mode}",
    )
    dataset_root = _require_mujoco_validation(
        state,
        cube_mode=cube_mode,
        allow_failed_override=bool(override_mujoco_failure),
    )
    if not workstation:
        # Both checks below are about the cube sidecars. The workstation replays the episode's own
        # action column, which the validation it just cleared was scored against -- there is no
        # second artefact that could be missing or belong to a different cube.
        validation_mode = str(
            (state.replay.mujocoValidation or {}).get("cubeMode") or state.replay.mujocoCubeMode
        )
        if validation_mode != cube_mode:
            raise RuntimeError(
                f"Run and pass MuJoCo {cube_mode} for this dataset/episode before real replay; "
                f"current validation is for {validation_mode}."
            )
        episode_poses = _read_sidecar_cube_poses(dataset_root, state.replay.episode)
        if not episode_poses.get(cube_mode):
            raise RuntimeError(f"No valid generated EE trajectory for: {cube_mode}")

    selected_ip = _validated_robot_ip(robot_ip or _real_robot_ip(state), "Robot IP")
    _append_real_replay_log(state, "validation", "trajectory and MuJoCo decision accepted")

    # Re-run hardware checks against the exact IP selected in this panel;
    # a prior preflight for a different arm must never authorize motion here.
    _start_realsense_preview(state, dataset_root)
    _append_real_replay_log(state, "camera", "RealSense monitor requested")
    try:
        _run_real_preflight(state, [selected_ip])
    except Exception:
        _stop_realsense_preview(state)
        raise
    state.replay.safety = "ready"
    state.replay.realCubeMode = cube_mode
    state.replay.realRobotIp = selected_ip
    state.replay.realEndEffectorMode = end_effector_mode
    state.replay.mujocoOverrideAccepted = bool(
        override_mujoco_failure and (state.replay.mujocoValidation or {}).get("status") == "failed"
    )
    if state.replay.mujocoOverrideAccepted:
        validation = state.replay.mujocoValidation or {}
        state.log(
            "warn",
            "Operator explicitly accepted failed MuJoCo validation for real replay: "
            f"max_position={validation.get('maxPositionErrorMm')}mm "
            f"max_rotation={validation.get('maxRotationErrorDeg')}deg",
        )
    command = _real_replay_command(state, dataset_root, cube_mode, selected_ip, end_effector_mode)
    _append_real_replay_log(state, "launch", "preflight passed; starting initial-pose-first replay process")
    try:
        process = subprocess.Popen(
            command,
            cwd=state.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_real_preflight_env(state),
            start_new_session=True,
        )
    except Exception:
        _stop_realsense_preview(state)
        _append_real_replay_log(state, "error", "failed to spawn real replay process")
        raise
    state.replay_process = process
    state.replay_process_kind = "real"
    state.replay_started_at_s = time.monotonic()
    state.selected_replay_root = dataset_root
    state.replay.state = "replaying"
    state.replay.safety = "active"
    state.replay.pid = process.pid
    state.replay.frameIndex = 0
    state.replay.datasetRoot = str(dataset_root)
    state.replay.dataset = str(dataset_root)
    state.replay.message = (
        f"Real robot {cube_mode} replay started for episode {state.replay.episode} from {dataset_root.name}; "
        "RealSense preview requested"
    )
    state.log(
        "warn",
        f"Started real robot replay pid={process.pid} dataset={dataset_root} "
        f"episode={state.replay.episode} cube={cube_mode} robot_ip={selected_ip} target={end_effector_mode}",
    )
    _start_replay_output_reader(state, process)


def _abort_replay(state: GatewayState) -> None:
    process = state.replay_process
    replay_kind = state.replay_process_kind or "replay"
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1.0)
        state.log("warn", f"Terminated {replay_kind} replay pid={process.pid}")
    if replay_kind == "real":
        _stop_realsense_preview(state)
        _append_real_replay_log(state, "abort", "operator stopped real replay")
    if replay_kind == "mujoco" and state.replay.mujocoValidation:
        state.replay.mujocoValidation["status"] = "failed"
        state.replay.mujocoValidation["message"] = "MuJoCo validation aborted before completion"
        state.replay.mujocoValidation["updatedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state.replay_process = None
    state.replay_process_kind = ""
    state.replay_started_at_s = None
    state.replay.state = "aborted"
    state.replay.safety = "locked"
    state.replay.pid = None
    state.replay.message = "Replay aborted; command stream stopped"


def _capture_recorder_start_pose(state: GatewayState) -> None:
    if state.profile != "workstation":
        raise RuntimeError("Dynamic start pose capture is only available for the FR3 workstation recorder.")
    if state.recording.state not in ("armed", "recording", "review"):
        raise RuntimeError(f"Cannot capture start pose while recorder is {state.recording.state}.")
    process = _ensure_recorder_running(state)
    try:
        _write_recorder_stdin(process, "set_start_pose\n")
    except BrokenPipeError as exc:
        raise RuntimeError("Recorder input is closed.") from exc
    state.recording.message = "Start pose capture requested"
    state.log("info", "Requested FR3 start pose capture")


def _reset_recorder_start_pose(state: GatewayState) -> None:
    """Undo a Set Home, back to the start pose the recorder was launched with.

    The capture only ever lived in the running recorder's config object -- neither it nor this
    writes the YAML -- so the reset is symmetric: it restores what that process read at startup, and
    a fresh recorder starts from the file regardless.
    """
    if state.profile != "workstation":
        raise RuntimeError("Dynamic start pose capture is only available for the FR3 workstation recorder.")
    if state.recording.state not in ("armed", "recording", "review"):
        raise RuntimeError(f"Cannot reset start pose while recorder is {state.recording.state}.")
    process = _ensure_recorder_running(state)
    try:
        _write_recorder_stdin(process, "reset_start_pose\n")
    except BrokenPipeError as exc:
        raise RuntimeError("Recorder input is closed.") from exc
    state.recording.message = "Start pose reset requested"
    state.log("info", "Requested FR3 start pose reset to the configured default")


def _stop_recorder(state: GatewayState, action: str) -> None:
    # Recorder is going away (or already gone): re-allow Device Manager previews.
    state.camera_preview_suspended = False
    try:
        process = _ensure_recorder_running(state)
    except RuntimeError:
        state.process = None
        state.recording.state = "idle"
        state.recording.pid = None
        state.recording.message = "Recorder is not running"
        return

    if action in ("save", "discard") and state.recording.state not in ("recording", "review"):
        raise RuntimeError(f"Cannot {action} while recorder is {state.recording.state}.")

    if action == "save":
        try:
            _write_recorder_stdin(process, "save\n")
            state.recording.state = "saving"
            state.recording.message = "Save requested; waiting for next episode"
            state.log("info", "Requested recorder save")
            return
        except BrokenPipeError:
            pass

    if action == "discard":
        try:
            _write_recorder_stdin(process, "n\n")
            state.recording.state = "discarding"
            state.recording.message = "Discard requested; waiting for next episode"
            state.log("warn", "Requested recorder discard")
            return
        except BrokenPipeError:
            pass

    if action == "exit":
        if state.recording.state == "connecting":
            os.killpg(process.pid, signal.SIGTERM)
            state.process = None
            state.process_started_at_s = None
            _clear_runtime_recording_config(state)
            state.recording.state = "idle"
            state.recording.pid = None
            state.recording.message = "Recorder process terminated during connect"
            _set_all_device_states(state, "idle")
            state.log("warn", "Terminated handheld recorder process during connect")
            return
        try:
            if state.recording.state == "recording":
                _write_recorder_stdin(process, "q\n")
            elif state.recording.state == "review":
                _write_recorder_stdin(process, "n\nexit\n")
            else:
                _write_recorder_stdin(process, "exit\n")
            state.recording.state = "discarding" if state.recording.frameIndex else "idle"
            state.recording.message = "Exit requested; waiting for recorder shutdown"
            state.log("warn", "Requested recorder exit")
            return
        except BrokenPipeError:
            pass

    os.killpg(process.pid, signal.SIGTERM)
    state.process = None
    state.process_started_at_s = None
    _clear_runtime_recording_config(state)
    state.recording.state = "idle"
    state.recording.pid = None
    state.recording.message = "Recorder process terminated"
    _set_all_device_states(state, "idle")
    state.log("warn", "Terminated handheld recorder process")


class DataCollectionGuiHandler(BaseHTTPRequestHandler):
    server: "DataCollectionGuiServer"

    def do_OPTIONS(self) -> None:
        _json_response(self, HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/teleop/camera.jpg":
            if self.server.state.profile != "workstation":
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "teleoperation is unavailable"})
                return
            view_id = query.get("view", [""])[0]
            _serve_teleop_camera_snapshot(self, state=self.server.state, view_id=view_id)
            return
        if path == "/api/training/hosts":
            state = self.server.state
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "hosts": [asdict(host) for host in training_backend.all_hosts(state.repo_root)],
                },
            )
            return
        if path == "/api/training/machine":
            state = self.server.state
            host_id = query.get("host", [training_backend.LOCAL_HOST_ID])[0]
            try:
                host = training_backend.resolve_host(state.repo_root, host_id)
            except training_backend.TrainingError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            # Probing shells out and can block for seconds on an unreachable host, so it runs
            # without the state lock: holding it would stall every other page's polling.
            report = training_backend.probe_machine(state.repo_root, host)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": bool(report.get("ok")),
                    "host": asdict(host),
                    "machine": report,
                    "wandb": training_backend.wandb_status(host.id),
                },
            )
            return
        if path == "/api/training/views":
            with self.server.state.lock:
                _json_response(
                    self, HTTPStatus.OK, {"ok": True, "views": _training_view_entries(self.server.state)}
                )
            return
        if path == "/api/training/status":
            with self.server.state.lock:
                _json_response(
                    self, HTTPStatus.OK, {"ok": True, "training": asdict(self.server.state.training)}
                )
            return
        if path == "/api/training/wandb":
            host_id = query.get("host", [training_backend.LOCAL_HOST_ID])[0]
            _json_response(
                self, HTTPStatus.OK, {"ok": True, "wandb": training_backend.wandb_status(host_id)}
            )
            return
        if path == "/api/rollout/camera.jpg":
            camera_key = query.get("camera", [""])[0]
            _serve_rollout_camera_snapshot(self, state=self.server.state, camera_key=camera_key)
            return
        if path == "/api/checkpoints":
            state = self.server.state
            host_id = query.get("host", [training_backend.LOCAL_HOST_ID])[0]
            # Outside the lock like the machine probe: a remote scan shells out over ssh and can
            # block for seconds on an unreachable host.
            try:
                _json_response(self, HTTPStatus.OK, _checkpoint_entries(state, host_id))
            except training_backend.TrainingError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        if path == "/api/rollout/status":
            state = self.server.state
            with state.lock:
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "rollout": asdict(state.rollout),
                        "modes": [asdict(mode) for mode in rollout_backend.ROLLOUT_MODES],
                        "rig": asdict(_rig_contract(state)),
                        "trainingBusy": _training_is_running(state),
                    },
                )
            return
        if path == "/api/rollout/outcomes":
            state = self.server.state
            entries = checkpoint_backend.load_rollout_outcomes(state.repo_root)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "entries": list(reversed(entries)),
                    "summary": checkpoint_backend.outcome_summary(entries),
                },
            )
            return
        if path == "/api/calibration/rig-check":
            _json_response(self, HTTPStatus.OK, _last_rig_check(self.server.state))
            return
        if path == "/api/calibration/marker-tcp":
            with self.server.state.lock:
                _json_response(self, HTTPStatus.OK, {"ok": True, "markerTcp": _marker_tcp_session_payload(self.server.state)})
            return
        if path.startswith("/api/assets/pika/"):
            name = path[len("/api/assets/pika/"):]
            allowed = {
                "pika_gripper_base_link.STL",
                "pika_gripper_left_link.STL",
                "pika_gripper_right_link.STL",
            }
            if name not in allowed:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "asset not allowed"})
                return
            asset_path = (
                self.server.state.repo_root
                / "src" / "lerobot" / "robots" / "franka_research3"
                / "assets" / "franka_fr3" / "assets" / name
            )
            if not asset_path.is_file():
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"asset missing: {name}"})
                return
            _serve_static_file(self, asset_path, "model/stl")
            return
        if path == "/api/device-preview/box":
            device_id = query.get("device", [""])[0]
            if not device_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing device"})
                return
            with self.server.state.lock:
                _json_response(self, HTTPStatus.OK, _box_preview_payload(self.server.state, device_id))
                return
        if path == "/api/device/box-cali-log":
            # 6D force calibration log buffer (polled by the force tile's log box).
            # Uses its own box_cali_lock inside the payload helper, so it never
            # contends with the main state lock or the recorder-stdout drain.
            _json_response(self, HTTPStatus.OK, _box_cali_log_payload(self.server.state))
            return
        if path == "/api/device/box-touch-cali-log":
            # Separate touch calibration log buffer (polled by the touch viewer).
            _json_response(self, HTTPStatus.OK, _box_touch_cali_log_payload(self.server.state))
            return
        if path == "/api/device-preview/camera.jpg":
            device_id = query.get("key", [""])[0]
            if not device_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing key"})
                return
            use_recorder_preview = _should_use_recorder_camera_preview(self.server.state)
            device = next(
                (
                    item
                    for item in self.server.state.devices
                    if item.get("id") == device_id and item.get("kind") == "camera"
                ),
                None,
            )
            config = (
                device.get("config")
                if isinstance(device, dict) and isinstance(device.get("config"), dict)
                else {}
            )
            if (
                self.server.state.profile == "workstation"
                and str(config.get("type", "")).lower() == "intelrealsense"
            ):
                _serve_realsense_device_preview_snapshot(
                    self,
                    state=self.server.state,
                    device_id=device_id,
                    device=device,
                )
                return
            params = None if use_recorder_preview else _camera_preview_params(self.server.state, device_id)
            if use_recorder_preview:
                _serve_recorder_camera_preview_snapshot(
                    self, state=self.server.state, device_id=device_id,
                )
                return
            if params is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"camera not found: {device_id}"})
                return
            sensor_id, sensor_mode, source_width, source_height, source_fps = params
            _serve_camera_preview_snapshot(
                self,
                state=self.server.state,
                device_id=device_id,
                sensor_id=sensor_id,
                sensor_mode=sensor_mode,
                source_width=source_width,
                source_height=source_height,
                source_fps=source_fps,
            )
            return
        if path == "/api/replay/realsense-status":
            with self.server.state.lock:
                payload = _realsense_preview_status(self.server.state)
            _json_response(self, HTTPStatus.OK, payload)
            return
        if path == "/api/replay/realsense.jpg":
            camera_key = (query.get("key", [""])[0] or "").strip()
            with self.server.state.lock:
                if not camera_key:
                    matches = _replay_realsense_camera_matches(self.server.state)
                    camera_key = str(matches[0]["cameraKey"]) if matches else "default"
                image_path, _status_path = _realsense_preview_paths(self.server.state, camera_key)
            if not image_path.is_file():
                _json_response(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "No RealSense preview frame available"})
                return
            _serve_static_file(self, image_path, "image/jpeg")
            return
        if path == "/api/processing/ik-plot":
            requested = query.get("path", [""])[0]
            cube = str(query.get("cube", [""])[0]).strip().lower()
            if cube not in {"left", "right"}:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "cube must be left or right"})
                return
            with self.server.state.lock:
                dataset_root = _resolve_known_dataset(self.server.state, requested)
            if dataset_root is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                return
            plot_path = (
                dataset_root
                / "derived"
                / DEFAULT_TRAJ_SIDECAR_NAME
                / "ik_qc"
                / cube
                / "verify_fr3_cube_pose_ik_error_over_time.png"
            )
            if not plot_path.is_file():
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"IK plot not found for {cube}"})
                return
            _serve_static_file(self, plot_path, "image/png")
            return
        if path == "/api/replay/frame-info":
            requested = query.get("path", [""])[0]
            with self.server.state.lock:
                dataset_root = _resolve_known_dataset(self.server.state, requested)
            if dataset_root is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                return
            _json_response(self, HTTPStatus.OK, _frame_preview_info(dataset_root))
            return
        if path == "/api/replay/frame.jpg":
            requested = query.get("path", [""])[0]
            camera_key = (query.get("key", [""])[0] or "").strip()
            if not camera_key:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing camera key"})
                return
            try:
                episode = int(query.get("episode", ["0"])[0] or 0)
                requested_frame = int(query.get("frame", ["0"])[0] or 0)
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "episode and frame must be integers"})
                return
            with self.server.state.lock:
                dataset_root = _resolve_known_dataset(self.server.state, requested)
            if dataset_root is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                return
            # ffmpeg and the metadata reads stay outside state.lock: this endpoint is driven by a
            # slider, and holding the lock for a decode would stall every snapshot poll behind it.
            source = _frame_preview_source(self.server.state, dataset_root, camera_key, episode)
            if source is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"video not found: {camera_key}"})
                return
            video_path, start_s, frames = source
            fps = _dataset_declared_fps(dataset_root) or 0
            frame_index, timestamp_s = _frame_preview_timestamp(start_s, requested_frame, frames, fps)
            jpeg = _frame_preview_jpeg(video_path, timestamp_s)
            if jpeg is None:
                _json_response(
                    self,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": f"could not decode frame {frame_index} of {camera_key}"},
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            # A decoded frame of a finished recording never changes, and the picker re-requests
            # the frame it just showed every time the operator steps back to it.
            self.send_header("Cache-Control", "private, max-age=300")
            self.send_header("X-Frame-Index", str(frame_index))
            self.send_header("X-Frame-Count", str(frames))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(jpeg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/api/replay/video":
            requested = query.get("path", [""])[0]
            camera_key = query.get("key", [""])[0]
            if not camera_key:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing camera key"})
                return
            requested_episode = query.get("episode", [None])[0]
            try:
                episode = int(requested_episode) if requested_episode not in (None, "") else None
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"invalid episode: {requested_episode}"})
                return
            with self.server.state.lock:
                dataset_root = _resolve_known_dataset(self.server.state, requested)
                resolved_episode = int(self.server.state.replay.episode if episode is None else episode)
            video_path = (
                _resolve_video_path(self.server.state, dataset_root, camera_key, resolved_episode)
                if dataset_root
                else None
            )
            if video_path is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"video not found: {camera_key}"})
                return
            _serve_video(self, video_path)
            return
        if path == "/api/replay/mujoco-video":
            requested = query.get("path", [""])[0]
            cube_mode = str(query.get("cube", [DEFAULT_MUJOCO_CUBE_MODE])[0]).strip().lower()
            if cube_mode not in MUJOCO_CUBE_MODES:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"invalid cube mode: {cube_mode}"})
                return
            try:
                episode = int(query.get("episode", ["0"])[0])
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid episode"})
                return
            with self.server.state.lock:
                workstation = self.server.state.profile == "workstation"
                dataset_root = (
                    _resolve_known_dataset(self.server.state, requested)
                    or self.server.state.selected_replay_root
                )
                if dataset_root is None:
                    video_path = None
                elif workstation:
                    video_path = _fr3_mujoco_replay_video_path(dataset_root, episode)
                else:
                    video_path = _mujoco_preview_video_path(dataset_root, episode, cube_mode)
            if video_path is None or not video_path.is_file():
                _json_response(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Run MuJoCo {cube_mode} replay for this dataset and episode first."},
                )
                return
            _serve_video(self, video_path)
            return
        with self.server.state.lock:
            if path in ("/api/health", "/api/snapshot"):
                _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                return
            if path == "/api/handheld/config":
                _json_response(self, HTTPStatus.OK, self.server.state.config)
                return
            if path == "/api/tasks":
                _json_response(self, HTTPStatus.OK, {"tasks": _tasks_with_progress(self.server.state)})
                return
            if path == "/api/replay/timeline":
                requested = query.get("path", [""])[0]
                dataset_root = _resolve_known_dataset(self.server.state, requested)
                if dataset_root is None and requested:
                    # Only an *absent* path falls back. A path that was sent and did not resolve
                    # is a caller error, and answering it with whatever happens to be selected
                    # returns another dataset's timeline under the name that was asked for.
                    # A path is easy to mangle in transit -- a raw "+" in a query string decodes
                    # to a space -- so this has to fail loudly rather than plausibly.
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {requested}"})
                    return
                if dataset_root is None:
                    dataset_root = self.server.state.selected_replay_root
                if dataset_root is None:
                    candidates = _dataset_root_candidates(self.server.state)
                    dataset_root = candidates[0] if candidates else None
                if dataset_root is None:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "no dataset available"})
                    return
                try:
                    requested_episode = query.get("episode", [None])[0]
                    episode = int(requested_episode) if requested_episode not in (None, "") else None
                    timeline = _read_dataset_timeline(self.server.state, dataset_root, episode)
                except Exception as exc:  # noqa: BLE001
                    _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                    return
                _json_response(self, HTTPStatus.OK, timeline)
                return
            if path == "/api/replay/mujoco-preview":
                requested = query.get("path", [""])[0]
                cube_mode = str(query.get("cube", [DEFAULT_MUJOCO_CUBE_MODE])[0]).strip().lower()
                if cube_mode not in MUJOCO_CUBE_MODES:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"invalid cube mode: {cube_mode}"})
                    return
                try:
                    episode = int(query.get("episode", ["0"])[0])
                except ValueError:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid episode"})
                    return
                dataset_root = _resolve_known_dataset(self.server.state, requested) or self.server.state.selected_replay_root
                if dataset_root is None:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not found"})
                    return
                workstation = self.server.state.profile == "workstation"
                try:
                    report = (
                        _fr3_mujoco_preview_payload(dataset_root, episode)
                        if workstation
                        else _load_json_file(_mujoco_preview_report_path(dataset_root, episode, cube_mode))
                    )
                except OSError as exc:
                    _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                    return
                if not report:
                    detail = (
                        "Run MuJoCo replay for this dataset and episode first."
                        if workstation
                        else f"Run MuJoCo {cube_mode} replay for this dataset and episode first."
                    )
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": detail})
                    return
                _json_response(self, HTTPStatus.OK, report)
                return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        # Recording is available in both profiles now: Thor drives the handheld/GMSL2 rig,
        # the workstation drives the FR3 SpaceMouse recorder. Only the BOX-sensor calibration
        # endpoints remain Thor-specific, since no BOX exists on the workstation.
        thor_only_paths = {
            "/api/device/calibrate-6dforce",
            "/api/device/calibrate-6dforce-origin",
            "/api/device/calibrate-touch",
            # Task-scoped v3 consolidation only exists for the multi-session GMSL2 rig; a
            # workstation dataset is already v3 and uses /api/datasets/export for its
            # training view instead.
            "/api/tasks/export",
        }
        if self.server.state.profile != "thor" and path in thor_only_paths:
            _json_response(
                self,
                HTTPStatus.CONFLICT,
                {"error": f"{path} is unavailable in the workstation profile"},
            )
            return
        if path.startswith("/api/training/"):
            state = self.server.state
            body = _read_json_body(self)
            try:
                if path == "/api/training/hosts/add":
                    host = training_backend.add_remote_host(
                        label=str(body.get("label") or ""),
                        ssh_target=str(body.get("sshTarget") or ""),
                        repo_dir=str(body.get("repoDir") or ""),
                        python_path=str(body.get("pythonPath") or ""),
                    )
                    _json_response(self, HTTPStatus.OK, {"ok": True, "host": asdict(host)})
                    return
                if path == "/api/training/hosts/remove":
                    training_backend.remove_remote_host(str(body.get("hostId") or ""))
                    _json_response(self, HTTPStatus.OK, {"ok": True})
                    return
                if path == "/api/training/wandb":
                    host_id = str(body.get("hostId") or training_backend.LOCAL_HOST_ID)
                    if body.get("clear"):
                        training_backend.clear_wandb_key(host_id)
                    else:
                        training_backend.set_wandb_key(host_id, str(body.get("apiKey") or ""))
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        {"ok": True, "wandb": training_backend.wandb_status(host_id)},
                    )
                    return
                if path == "/api/training/sync":
                    host = training_backend.resolve_host(state.repo_root, str(body.get("hostId") or ""))
                    result = training_backend.sync_repo_to_host(state.repo_root, host)
                    _json_response(self, HTTPStatus.OK, {"ok": bool(result.get("ok")), "sync": result})
                    return
                if path == "/api/training/start":
                    # Started outside the lock for the same reason the probe is: a remote start
                    # runs an rsync first, and that can take a minute.
                    result = _start_training_run(state, body)
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if path == "/api/training/stop":
                    with state.lock:
                        _json_response(self, HTTPStatus.OK, _stop_training_run(state))
                    return
            except training_backend.TrainingError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown route {path}"})
            return
        if path.startswith("/api/checkpoints/"):
            state = self.server.state
            body = _read_json_body(self)
            try:
                if path == "/api/checkpoints/fetch":
                    # Outside the lock: this is an rsync of a few hundred megabytes, and holding
                    # the lock for it would freeze every other page's polling for its duration.
                    host = training_backend.resolve_host(state.repo_root, str(body.get("hostId") or ""))
                    listing = _checkpoint_entries(state, host.id)
                    checkpoint_id = str(body.get("checkpointId") or "")
                    selected = next(
                        (item for item in listing["checkpoints"] if item.get("id") == checkpoint_id),
                        None,
                    )
                    if selected is None:
                        raise checkpoint_backend.CheckpointError(
                            f"No checkpoint {checkpoint_id} on {host.label}."
                        )
                    result = checkpoint_backend.fetch_checkpoint(state.repo_root, host, selected)
                    with state.lock:
                        state.log("info", result["message"])
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if path == "/api/checkpoints/delete":
                    checkpoint_id = str(body.get("checkpointId") or "")
                    with state.lock:
                        _guard_checkpoint_deletion(state, [checkpoint_id])
                    result = checkpoint_backend.delete_checkpoint(state.repo_root, checkpoint_id)
                    with state.lock:
                        state.log("warn", result["message"])
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if path == "/api/checkpoints/delete-many":
                    raw_ids = body.get("checkpointIds")
                    checkpoint_ids = (
                        [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []
                    )
                    # Guarded before the first rmtree, not per id: a batch that includes the
                    # running job should be refused whole, rather than half-deleted and then
                    # stopped at the one that mattered.
                    with state.lock:
                        _guard_checkpoint_deletion(state, checkpoint_ids)
                    result = checkpoint_backend.delete_checkpoints(state.repo_root, checkpoint_ids)
                    with state.lock:
                        state.log("warn", result["message"])
                    _json_response(self, HTTPStatus.OK, result)
                    return
            except (checkpoint_backend.CheckpointError, training_backend.TrainingError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown route {path}"})
            return
        if path.startswith("/api/rollout/"):
            state = self.server.state
            body = _read_json_body(self)
            try:
                if path == "/api/rollout/start":
                    with state.lock:
                        _json_response(self, HTTPStatus.OK, _start_rollout(state, body))
                    return
                if path == "/api/rollout/control":
                    with state.lock:
                        _json_response(
                            self,
                            HTTPStatus.OK,
                            _send_rollout_control(state, str(body.get("command") or "")),
                        )
                    return
                if path == "/api/rollout/stop":
                    with state.lock:
                        _json_response(self, HTTPStatus.OK, _stop_rollout(state))
                    return
                if path == "/api/rollout/outcome":
                    with state.lock:
                        _json_response(self, HTTPStatus.OK, _record_rollout_outcome(state, body))
                    return
            except (
                rollout_backend.RolloutError,
                checkpoint_backend.CheckpointError,
                training_backend.TrainingError,
                ValueError,
            ) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown route {path}"})
            return
        if path == "/api/handheld/record/connect":
            # Free the cameras before the recorder opens them: live Device
            # Manager previews hold Argus sessions on the same sensor-ids, and
            # leaving them up makes the recorder's nvarguscamerasrc open hang.
            # The whole preflight + spawn runs inside the suspension context
            # manager so the flag is reset on any failure (terminate(), sleep,
            # or _connect_recorder raising), not just the ones a hand-written
            # except remembered to cover.
            state = self.server.state
            requested_backend = (query.get("backend", [""])[0] or "").strip().lower() or None
            raw_episode_time_s = (
                query.get("episode_time_s", query.get("episodeTimeS", [""]))[0] or ""
            ).strip()
            raw_fps = (query.get("fps", query.get("recording_fps", [""]))[0] or "").strip()
            try:
                episode_time_s = _parse_episode_time_override(raw_episode_time_s)
                recording_fps = _parse_recording_fps_override(raw_fps)
                with _previews_suspended_for_connect(state):
                    # Done outside the state lock (terminate() blocks).
                    _stop_all_camera_previews(state)
                    if state.profile == "workstation":
                        # The RealSense teleop preview owns the same USB devices the hardware
                        # recorder is about to open; leaving it up makes the recorder's
                        # pipeline_start fail with "device busy".
                        _stop_realsense_preview(state)
                    settle_s = _camera_preview_stagger_s(state)
                    if settle_s > 0:
                        time.sleep(settle_s)
                    with state.lock:
                        _connect_recorder(
                            state,
                            backend=requested_backend,
                            episode_time_s=episode_time_s,
                            recording_fps=recording_fps,
                        )
                        response = _snapshot(state)
                _json_response(self, HTTPStatus.OK, response)
            except Exception as exc:  # noqa: BLE001
                state.log("warn", f"{path} failed: {exc}")
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path == "/api/processing/qc":
            state = self.server.state
            requested = (query.get("path", [""])[0] or "").strip()
            try:
                with state.lock:
                    dataset_root = _resolve_known_dataset(state, requested)
                if dataset_root is None:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                    return
                qc_result = _run_qc(
                    dataset_root,
                    repo_root=state.repo_root,
                    ik_python=_mujoco_replay_python(state),
                )
                try:
                    _write_processing_meta_qc(dataset_root, qc_result)
                except OSError as exc:
                    _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"failed to persist QC: {exc}"})
                    return
                _refresh_cached_processing_item(state, dataset_root)
                with state.lock:
                    state.log(
                        "info" if qc_result["status"] == "pass" else "warn",
                        f"QC {qc_result['status']} for {dataset_root.name}: {qc_result['summary']}",
                    )
                    response = _snapshot(state)
                _json_response(self, HTTPStatus.OK, response)
            except Exception as exc:  # noqa: BLE001
                with state.lock:
                    state.log("warn", f"{path} failed: {exc}")
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path == "/api/processing/traj-gen":
            state = self.server.state
            requested = (query.get("path", [""])[0] or "").strip()
            try:
                if state.profile == "workstation":
                    _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {"error": "FR3 workstation datasets already contain recorded EE trajectories; run QC directly."})
                    return
                with state.lock:
                    dataset_root = _resolve_known_dataset(state, requested)
                if dataset_root is None:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                    return
                _queue_traj_gen(state, dataset_root)
                with state.lock:
                    response = _snapshot(state)
                _json_response(self, HTTPStatus.OK, response)
            except NotImplementedError as exc:
                _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                with state.lock:
                    state.log("warn", f"{path} failed: {exc}")
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        try:
            with self.server.state.lock:
                if path == "/api/handheld/record/start":
                    _start_episode(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/handheld/record/stop-save":
                    _stop_recorder(self.server.state, "save")
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/handheld/record/stop-discard":
                    _stop_recorder(self.server.state, "discard")
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/handheld/record/reset-start-pose":
                    _reset_recorder_start_pose(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/handheld/record/set-start-pose":
                    _capture_recorder_start_pose(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/handheld/record/exit":
                    _stop_recorder(self.server.state, "exit")
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/preflight":
                    _preflight_replay(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/select-dataset":
                    _select_replay_dataset(self.server.state, query.get("path", [""])[0])
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/select-episode":
                    _select_replay_episode(self.server.state, query.get("episode", [""])[0])
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/delete-episode":
                    _delete_replay_episode(self.server.state, query.get("episode", [""])[0])
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/start-mujoco":
                    _start_mujoco_replay(
                        self.server.state,
                        query.get("cube", [DEFAULT_MUJOCO_CUBE_MODE])[0],
                    )
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/approve-mujoco":
                    _approve_mujoco_report(
                        self.server.state,
                        query.get("cube", [DEFAULT_MUJOCO_CUBE_MODE])[0],
                    )
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/start-real":
                    _start_real_replay(
                        self.server.state,
                        query.get("cube", ["right"])[0],
                        query.get("robot_ip", [""])[0],
                        query.get("end_effector", ["pika_gripper_ee"])[0],
                        str(query.get("override_mujoco_failure", ["false"])[0]).strip().lower()
                        in ("1", "true", "yes", "on"),
                    )
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/teleop/start-sim":
                    if self.server.state.profile != "workstation":
                        _json_response(
                            self, HTTPStatus.CONFLICT, {"error": "teleoperation is unavailable in the Thor profile"}
                        )
                        return
                    _start_fr3_sim_teleop(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/teleop/start-real":
                    if self.server.state.profile != "workstation":
                        _json_response(
                            self, HTTPStatus.CONFLICT, {"error": "teleoperation is unavailable in the Thor profile"}
                        )
                        return
                    _start_fr3_real_teleop(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/teleop/stop":
                    if self.server.state.profile != "workstation":
                        _json_response(
                            self, HTTPStatus.CONFLICT, {"error": "teleoperation is unavailable in the Thor profile"}
                        )
                        return
                    _stop_fr3_teleop(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/teleop/gains":
                    if self.server.state.profile != "workstation":
                        _json_response(
                            self, HTTPStatus.CONFLICT, {"error": "teleoperation is unavailable in the Thor profile"}
                        )
                        return
                    try:
                        overrides = _parse_teleop_gain_overrides(_read_json_body(self))
                    except ValueError as exc:
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    state = self.server.state
                    # A new gain only reaches the arm on the next spawn: the teleoperator reads its
                    # config once at construction. Say so rather than letting the operator turn a
                    # knob and wait for a running session to respond to it.
                    state.runtime_teleop_gains = overrides
                    if overrides:
                        state.log(
                            "info",
                            "SpaceMouse gain override set: "
                            + ", ".join(f"{name}={value:g}" for name, value in sorted(overrides.items()))
                            + "; applies to the next teleop or recording session",
                        )
                    else:
                        state.log("info", "SpaceMouse gains reset to the recorder config")
                    _json_response(self, HTTPStatus.OK, _snapshot(state))
                    return
                if path == "/api/calibration/run":
                    dataset = (query.get("dataset", [""])[0] or "").strip()
                    result = _start_extrinsics_calibration(self.server.state, dataset)
                    if not result.get("ok"):
                        _json_response(self, HTTPStatus.CONFLICT, result)
                        return
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/calibration/session/start":
                    result = _start_calibration_session(
                        self.server.state, (query.get("cameras", [""])[0] or "").strip()
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/start":
                    result = _start_marker_tcp_session(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/cancel":
                    result = _cancel_marker_tcp_session(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/record":
                    result = _marker_tcp_record_sample(
                        self.server.state,
                        (query.get("action", ["start"])[0] or "start").strip(),
                        side=(query.get("side", [""])[0] or "").strip(),
                        condition=(query.get("condition", [""])[0] or "").strip(),
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/register":
                    result = _register_marker_tcp_static_transform(
                        self.server.state,
                        path_arg=(query.get("path", [""])[0] or "").strip(),
                        side=(query.get("side", [""])[0] or "").strip(),
                        condition=(query.get("condition", [""])[0] or "").strip(),
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/report":
                    result = _run_marker_tcp_repeatability_report(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/session/record":
                    action = (query.get("action", ["start"])[0] or "start").strip()
                    result = _calibration_step_record(self.server.state, action)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/session/skip":
                    result = _calibration_step_skip(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/session/cancel":
                    result = _cancel_calibration_session(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/rig-check":
                    result = _run_rig_check(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/rig-check/baseline":
                    result = _capture_rig_check_baseline(self.server.state)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/device/calibrate-6dforce":
                    box_id = (query.get("box_id", [""])[0] or "").strip()
                    result = _trigger_box_six_d_force_cali(self.server.state, box_id=box_id)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/device/calibrate-6dforce-origin":
                    box_id = (query.get("box_id", [""])[0] or "").strip()
                    result = _trigger_box_six_d_force_cali(self.server.state, origin=True, box_id=box_id)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/device/calibrate-touch":
                    box_id = (query.get("box_id", [""])[0] or "").strip()
                    result = _trigger_box_touch_cali(self.server.state, box_id=box_id)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/processing/datasets-root":
                    requested = (query.get("path", [""])[0] or "").strip()
                    try:
                        created = _set_datasets_root(self.server.state, requested)
                    except ValueError as exc:
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    resolved = self.server.state.datasets_root
                    message = (
                        f"Datasets root did not exist; created {resolved}"
                        if created
                        else f"Datasets root changed to {resolved}"
                    )
                    self.server.state.log("info", message)
                    snapshot = _snapshot(self.server.state)
                    if created:
                        snapshot["notice"] = message
                    _json_response(self, HTTPStatus.OK, snapshot)
                    return
                if path == "/api/annotation/save":
                    _save_annotation(self.server.state, _read_json_body(self))
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/abort":
                    _abort_replay(self.server.state)
                    self.server.state.log("warn", "Replay aborted")
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/tasks/create":
                    task = _create_task(self.server.state, _read_json_body(self))
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/tasks/update":
                    task = _update_task(self.server.state, _read_json_body(self))
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/tasks/delete":
                    task_id = (query.get("id", [""])[0] or "").strip()
                    _delete_task(self.server.state, task_id)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/tasks/activate":
                    task_id = (query.get("id", [""])[0] or "").strip()
                    _set_active_task(self.server.state, task_id)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/tasks/export":
                    task_id = (query.get("id", [""])[0] or "").strip()
                    _start_task_export(self.server.state, task_id)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/datasets/export":
                    requested = (query.get("path", [""])[0] or "").strip()
                    if self.server.state.profile == "workstation":
                        # The workstation recorder already writes v3, so there is no raw->v3
                        # export here; the equivalent step is building the training view.
                        # `paths` is repeated once per selected recording rather than joined:
                        # dataset paths are absolute and a separator would have to be one that
                        # cannot appear in a path.
                        requested_paths = [
                            value.strip() for value in query.get("paths", []) if value.strip()
                        ] or ([requested] if requested else [])
                        _start_training_view(
                            self.server.state,
                            requested_paths,
                            (query.get("action_mode", [DEFAULT_TRAINING_VIEW_ACTION_MODE])[0] or "").strip()
                            or DEFAULT_TRAINING_VIEW_ACTION_MODE,
                            acknowledge_warnings=_query_flag(query, "acknowledge_warnings"),
                            camera_crops=_parse_training_view_camera_crops(
                                (query.get("camera_crops", [""])[0] or "").strip()
                            ),
                            view_fps=_parse_training_view_fps(
                                (query.get("view_fps", [""])[0] or "").strip()
                            ),
                        )
                    else:
                        _start_approved_dataset_export(
                            self.server.state,
                            requested,
                            acknowledge_warnings=_query_flag(query, "acknowledge_warnings"),
                        )
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
        except Exception as exc:  # noqa: BLE001
            if path == "/api/replay/start-real":
                _append_real_replay_log(self.server.state, "error", str(exc))
            self.server.state.log("warn", f"{path} failed: {exc}")
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[data-collection-gateway] {format % args}\n")


class DataCollectionGuiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: GatewayState):
        super().__init__(server_address, DataCollectionGuiHandler)
        self.state = state


def make_state(
    repo_root: Path,
    config_path: Path,
    datasets_root: Path | None = None,
    exports_root: Path | None = None,
    log_dir: Path | None = None,
    gateway_log_path: Path | None = None,
    profile: str = "thor",
) -> GatewayState:
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError(f"Unknown deployment profile: {profile}")
    resolved_root = repo_root.resolve()
    resolved_config = config_path if config_path.is_absolute() else resolved_root / config_path
    config = _load_yaml(resolved_config)
    resolved_datasets_root: Path | None = None
    if datasets_root is not None:
        resolved_datasets_root = datasets_root if datasets_root.is_absolute() else resolved_root / datasets_root
        resolved_datasets_root = resolved_datasets_root.resolve()
    resolved_exports_root: Path | None = None
    if exports_root is not None:
        resolved_exports_root = exports_root if exports_root.is_absolute() else resolved_root / exports_root
        resolved_exports_root = resolved_exports_root.resolve()
    state = GatewayState(
        repo_root=resolved_root,
        config_path=resolved_config,
        config=config,
        profile=profile,
        recording=_recording_status_from_config(config),
        replay=_replay_status_from_config(config),
        datasets_root=resolved_datasets_root,
        exports_root=resolved_exports_root,
        log_dir=log_dir,
        gateway_log_path=gateway_log_path,
        devices=_device_statuses(config, resolved_root),
    )
    state.replay.mujocoValidation = _new_mujoco_validation(state)
    _load_active_calibration_runs(state)
    _restore_training_run(state)
    if profile == "workstation":
        urdf_path, sim_xml_path = _fr3_pika_asset_paths(resolved_root)
        state.teleop.urdfPath = str(urdf_path)
        state.teleop.simXmlPath = str(sim_xml_path)
        state.teleop.message = "FR3 Pika MuJoCo teleop is ready on the workstation"
    state.log("info", f"Loaded {profile} config {resolved_config}")
    if resolved_datasets_root is not None:
        state.log("info", f"Scanning datasets under {resolved_datasets_root}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local HTTP gateway for the LeRobot data collection GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=tuple(DEPLOYMENT_PROFILES), default="thor")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--exports-root", type=Path, default=DEFAULT_EXPORTS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


def _setup_gateway_log(repo_root: Path, requested_log_dir: Path | None) -> tuple[Path, Path]:
    log_dir = requested_log_dir or _default_log_dir(repo_root)
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir
    log_dir = _ensure_log_dir(log_dir.resolve())
    log_path = log_dir / f"gateway_{_timestamp_for_log()}_{os.getpid()}.log"
    # Replace process stdout/stderr with a line-buffered file so gateway and
    # HTTP handler diagnostics survive when launched from the frontend/dev shell.
    fh = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = fh
    sys.stderr = fh
    return log_dir, log_path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    log_dir, gateway_log_path = _setup_gateway_log(repo_root, args.log_dir)
    state = make_state(
        repo_root, args.config_path, args.datasets_root, args.exports_root,
        log_dir=log_dir, gateway_log_path=gateway_log_path, profile=args.profile,
    )
    if state.profile == "workstation":
        _probe_workstation_devices(state)
    server = DataCollectionGuiServer((args.host, args.port), state)
    _start_background_workers(state)
    print(f"Data collection GUI gateway listening on http://{args.host}:{args.port}")
    print(f"Gateway log: {gateway_log_path}")
    try:
        server.serve_forever()
    finally:
        with state.lock:
            if state.process is not None and state.process.poll() is None:
                os.killpg(state.process.pid, signal.SIGTERM)
            for process in list(state.processing_processes.values()):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            if state.teleop_process is not None and state.teleop_process.poll() is None:
                os.killpg(state.teleop_process.pid, signal.SIGTERM)
                state.teleop_process = None
            state.processing_processes.clear()
        server.server_close()


if __name__ == "__main__":
    main()
