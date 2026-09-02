#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
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
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, Thread, Timer
from typing import Any, Callable, Iterable, Sequence
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from tools.data_collection_gui import calibration_promotion as promotion

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
DEFAULT_REAL_ROBOT_IP = "192.168.1.208"
DEFAULT_CUBE_REPLAY_ROBOT_IP = "192.168.11.102"
# EE trajectory generation now tracks gmsl2 (Thor) datasets with AprilTag cubes
# instead of the legacy Hikon-camera route. The gateway runs on Thor, so it
# invokes the local runner directly (no SSH / copy-back) -- the runner picks the
# opencv_kalibr venv that actually has cv2/pupil_apriltags and wires PYTHONPATH.
DEFAULT_EE_TRAJECTORY_RUNNER = Path("third_party/opencv_kalibr/run_april_cube_tracking_local.sh")
DEFAULT_EE_TRAJECTORY_CONFIG = Path(
    "third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml"
)
# Resolved marker layout the solve writes next to its production bundle, so a
# non-identity rig frame travels with the bundle instead of being lost.
DEFAULT_MARKER_LAYOUT_NAME = "marker_layout_resolved.json"
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
    targetFrameName: str = "pika_task_tcp"
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
class CalibrationProgress:
    """How far the solve has got, in units the operator can check against.

    The solve is three subprocesses, and the first one decodes every frame of
    every recorded video: on an 11-camera capture that is tens of minutes with
    nothing on screen but "正在解算…". It also used to fail *after* that wait
    -- the bundle imports scipy, which the rig's interpreter did not have -- so
    "is it alive" and "how much is left" were both unanswerable.
    """

    stepIndex: int = 0  # 1-based; 0 while nothing is running
    stepCount: int = 0
    label: str = ""
    done: int = 0
    total: int = 0  # 0 = this step reports no unit of its own
    fraction: float = 0.0  # overall, 0..1
    # Wall-clock share of each step in *this* run's plan; see _solve_weights.
    weights: list[float] = field(default_factory=list)
    detail: str = ""
    startedAt: float = 0.0  # epoch seconds; 0 = not started
    elapsedS: float = 0.0
    etaS: float = 0.0  # 0 = no basis to extrapolate from yet


@dataclass
class CalibrationStatus:
    state: str = "idle"
    pattern: str = "ChArUco 12x9 · 30 mm (charuco_400)"
    lastRunAt: str = ""
    message: str = "Run calibration to refresh extrinsics"
    cameras: list[dict[str, Any]] = field(default_factory=list)
    outputPath: str = ""
    progress: CalibrationProgress = field(default_factory=CalibrationProgress)
    # The capture the next solve reads, when the operator has named one.
    # Empty means "work it out" -- see _solve_dataset.
    solveDatasetRoot: str = ""
    # The capture intrinsics are re-fitted from, when the operator asks for it.
    # A different recording from the extrinsics one: intrinsics need a per-camera
    # sweep that reaches that camera's frame edges, extrinsics need one sweep
    # several cameras watch at once. They cannot share a capture.
    intrinsicsDatasetRoot: str = ""
    # Which calibration runs production is currently pointed at. The self-check
    # records these with its baseline: a baseline that outlives the calibration
    # it was taken against compares against a rig that no longer exists.
    intrinsicsRun: str = ""
    extrinsicsRun: str = ""
    # Whether the last finished solve was written into production. A solve run
    # in experiment mode leaves the production pointers alone, so "complete"
    # alone no longer implies "this is what the rig is using".
    lastRunExported: bool = True


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
    # How long each sweep records, in seconds. Independent of the config's
    # dataset.episode_time_s because a calibration segment is paced by a human
    # walking a board around a rig, not by whatever an ordinary demonstration
    # episode is worth. Sent to the recorder per episode.
    episodeTimeS: float = 30.0


@dataclass
class MarkerTcpSample:
    id: str
    side: str
    condition: str
    boxId: str = ""
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
    stage: str = "idle"  # idle | capture | solving | reporting | done | failed
    samples: list[MarkerTcpSample] = field(default_factory=list)
    pendingSampleId: str = ""
    message: str = ""
    reportPath: str = ""
    solvePath: str = ""
    solveSummaryPath: str = ""
    pivotReportPath: str = ""
    trackingRunPath: str = ""


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
    export_process: subprocess.Popen[str] | None = None
    events: list[EventLogItem] = field(default_factory=list)
    selected_replay_root: Path | None = None
    active_task_id: str | None = None
    process: subprocess.Popen[str] | None = None
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


# Recorder default when a config omits dataset.episode_time_s. 10 s was too
# short for anything an operator performs in front of the rig (a calibration
# board sweep, a demonstration), so segments kept being cut mid-motion.
_DEFAULT_EPISODE_TIME_S = 20.0


def _episode_time_s(config: dict[str, Any]) -> float:
    return float(_dataset_config(config).get("episode_time_s") or _DEFAULT_EPISODE_TIME_S)


def _target_frames_for_seconds(config: dict[str, Any], seconds: float) -> int:
    fps = int(_dataset_config(config).get("fps") or 30)
    return max(1, int(round(fps * seconds)))


def _target_frames(config: dict[str, Any]) -> int:
    return _target_frames_for_seconds(config, _episode_time_s(config))


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
    return {
        "configPath": str(config_path),
        "repoId": str(dataset.get("repo_id") or ""),
        "root": str(dataset.get("root") or ""),
        "fps": int(dataset.get("fps") or 30),
        "episodeTimeS": _episode_time_s(config),
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
        "rigType": "gmsl2" if is_gmsl else "handheld",
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
    # Pad vendor/geometry is not fixed (Paxini L5325 239-taxel, M2020 3x3, ...)
    # and is reported per frame as `model`; keep the static label neutral.
    "box_touch_left": "BOX touch pad L",
    "box_touch_right": "BOX touch pad R",
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


def _path_modified_ns(path: Path) -> int:
    """Integer mtime, for comparing a file against a record of what it was."""
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


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


def _as_float(value: Any) -> float | None:
    """A finite float, or None. NaN and inf are absences, not measurements."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


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
        "markerTcpCalibrationPath": "",
        "onlineSync": _online_sync_manifest_summary(dataset_root),
        "qcChecks": [],
        "ikEvaluation": None,
    }

    meta = _load_processing_meta(dataset_root)
    if meta:
        active_version = meta.get("active_version")
        versions = meta.get("versions") if isinstance(meta.get("versions"), dict) else {}
        current_job = meta.get("current_job") if isinstance(meta.get("current_job"), dict) else {}
        version_info = versions.get(active_version) if isinstance(active_version, str) else None
        qc = version_info.get("qc") if isinstance(version_info, dict) else None
        marker_tcp_path = ""
        if isinstance(current_job, dict):
            marker_tcp_path = str(current_job.get("marker_to_tcp_calibration_path", "") or "")
        if not marker_tcp_path and isinstance(version_info, dict):
            marker_tcp_path = str(version_info.get("marker_to_tcp_calibration_path", "") or "")
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
            else:
                status = "pose_ready"
            message = qc.get("reason") or qc.get("message") or "QC available"
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
            "markerTcpCalibrationPath": marker_tcp_path,
            "qcChecks": list(qc.get("checks") or []) if isinstance(qc, dict) else [],
            "ikEvaluation": qc.get("ik_evaluation") if isinstance(qc, dict) else None,
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


# What the solve actually imports. cv2 decodes the video and finds the corners;
# scipy's sparse least-squares *is* the bundle adjustment. They are probed
# together because an interpreter carrying only cv2 gets all the way through
# detection -- tens of minutes -- before the bundle dies on "No module named
# 'scipy'", which is exactly what the rig did on 2026-08-20. Submodules rather
# than top-level names: a partial scipy install imports and then fails here.
_SOLVE_REQUIRED_IMPORTS = ("cv2", "numpy", "scipy.optimize", "scipy.sparse")

_IMPORT_PROBE = (
    "import importlib, sys\n"
    "missing = []\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception:\n"
    "        missing.append(name)\n"
    "print(' '.join(missing))\n"
)


def _missing_modules(python: Path, modules: Sequence[str]) -> list[str]:
    """Which of `modules` this interpreter cannot import.

    An interpreter that cannot even run the probe is missing all of them: the
    caller only has to decide whether to use it, and it cannot be used.
    """
    try:
        probe = subprocess.run(
            [str(python), "-c", _IMPORT_PROBE, *modules],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return list(modules)
    if probe.returncode != 0:
        return list(modules)
    return probe.stdout.split()


def _solve_python(repo_root: Path) -> tuple[Path | None, list[str]]:
    """Interpreter for the metrology CLIs, and whatever it still cannot import.

    Prefers one that can import everything the solve needs; falls back to the
    closest so the refusal can name the interpreter and the module, which is
    what makes it fixable, instead of a traceback from a step that has already
    burned the whole detection pass.
    """
    candidates = [
        _venv_python3(repo_root, prefer_fr3=True),
        Path("/home/nvidia/Code/infer/.venv-fr3/bin/python3"),
        Path(sys.executable),
    ]
    best: tuple[Path, list[str]] | None = None
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        missing = _missing_modules(candidate, _SOLVE_REQUIRED_IMPORTS)
        if not missing:
            return candidate, []
        if best is None or len(missing) < len(best[1]):
            best = (candidate, missing)
    if best is None:
        return None, list(_SOLVE_REQUIRED_IMPORTS)
    return best


def _missing_modules_message(python: Path, missing: Sequence[str]) -> str:
    """Say which package is missing and how to install it, not just that it is."""
    packages = sorted({name.split(".")[0] for name in missing})
    return (
        f"解算用的解释器缺少 {'、'.join(packages)}：{python}。"
        f"在这台机器上执行 `{python} -m pip install {' '.join(packages)}` 后重试。"
    )


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


# --- Canonical world frame (roadmap Phase 2.4) -------------------------------
#
# A bundle adjustment fixes its gauge on whichever camera it likes, so exporting
# its poses straight out redefines the world on every re-solve: yesterday's
# absolute trajectories keep their numbers and quietly lose their meaning. The
# frozen world_reference.json is what makes `world_frame_id` mean something, and
# stable-camera consensus is what carries it across a recalibration.
#
# Everything here shells out to metrology.cli.world_registration for the same
# reason the rig self-check does: the gateway carries no numerical stack.

# Tracked in git, deliberately not under outputs/. The frozen reference cannot
# be regenerated -- re-freezing mints a new world_frame_id for the same physical
# frame and orphans the ID stamped into every episode recorded so far -- and
# outputs/ is 7 GB of regenerable artefacts that gets deleted to reclaim space.
_WORLD_SUBDIR = Path("tools") / "thor" / "gmsl2" / "world"
_WORLD_REFERENCE_FILE = "world_reference.json"
_WORLD_GRAPH_FILE = "world_graph.json"
_WORLD_REGISTRATION_FILE = "world_registration.json"
# Which evidence chose the stable cameras. Written by the gateway rather than by
# the CLI, because it is the gateway that knows the rig self-check exists.
_WORLD_STABLE_SOURCE_FILE = "world_stable_source.json"


def _world_root(state: GatewayState) -> Path:
    return state.repo_root / _WORLD_SUBDIR


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_bundle_report(state: GatewayState) -> Path | None:
    """The newest self-calibration bundle -- the current session's camera graph.

    Preferring the run this gateway just solved over a directory scan matters
    when several calibrations sit side by side: registering the wrong one would
    report a healthy world for a rig nobody is using.
    """
    if state.calibration.outputPath:
        candidate = Path(state.calibration.outputPath) / "extrinsics_report.json"
        if candidate.is_file():
            return candidate
    root = state.repo_root / "outputs" / "metrology"
    if not root.is_dir():
        return None
    reports = sorted(root.glob("*/extrinsics_report.json"), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


def _rig_check_stable_cameras(state: GatewayState, reference_created_utc: str) -> dict[str, Any]:
    """Which cameras the image-based self-check says did not move, if usable.

    The two checks are not equally sensitive and it is worth being explicit
    about which is which. The self-check compares each camera's *view* against
    a baseline and resolves about 2 px -- roughly 1.7 mm at 1 m. The geometric
    consensus below compares camera-to-camera geometry between two independent
    solves, which on this rig disagree by up to 6.4 mm with nothing touched, so
    its floor is around a centimetre.

    So the self-check is the better detector of "was this camera bumped", and
    handing its verdict over as the declared stable set puts the sensitive
    measurement in charge of the decision. The geometric per-camera residual
    then becomes an independent check on that decision rather than a restatement
    of it -- the same reason the exporter leaves orientation out of its fit.

    Refused rather than used when it cannot carry that weight: too few cleared
    cameras to define a frame, a verdict that already says it could not tell, or
    a result older than the world it would be judging movement against.
    """
    meta: dict[str, Any] = {"origin": "geometry", "cameras": [], "moved": []}
    report = _read_json_file(_rig_check_root(state) / "last_result.json")
    if not report:
        meta["reason"] = "没有相机自检结果"
        return meta

    cameras = report.get("cameras") or {}
    ok = sorted(name for name, entry in cameras.items() if (entry or {}).get("verdict") == "ok")
    moved = sorted(name for name, entry in cameras.items() if (entry or {}).get("verdict") == "moved")
    generated = str(report.get("generated_utc", ""))
    meta.update({"generatedUtc": generated, "rigCheckOverall": report.get("overall", "")})

    if str(report.get("overall", "")) == "inconclusive":
        meta["reason"] = "相机自检判定为「无法判定」，不能当作未移动的证据"
        return meta
    if len(ok) < 3:
        meta["reason"] = f"相机自检只有 {len(ok)} 台判为未移动，不足以定义世界系"
        return meta
    # Lexicographic works on the "%Y-%m-%dT%H:%M:%SZ" both sides write. An
    # unparseable or missing stamp means the ordering cannot be established,
    # which is a reason not to trust it rather than a reason to assume it.
    if not generated or not reference_created_utc or generated <= reference_created_utc:
        meta["reason"] = "相机自检结果不晚于世界系冻结时间，描述的不是冻结之后的状态"
        return meta

    meta.update({"origin": "rig_check", "cameras": ok, "moved": moved})
    return meta


# How much of the frame radius the board has to reach before the distortion
# model is constrained rather than extrapolated. Not a number picked for this
# panel: metrology.cli.calibrate_intrinsics prints its own "边缘无数据，畸变模型
# 在其余部分是外推" warning below exactly this, and the two must not disagree
# about what "covered" means.
#
# There is deliberately no second, lower threshold that would declare a camera
# due for a re-shoot. One used to live here at 0.80 and it was wrong: coverage
# costs nothing where the camera does not work, and the least-covered camera on
# this rig (cam_06, 79%) never sees the cube past 52% of its frame radius.
# Whether an extrapolated band matters is a question about the workspace, which
# this endpoint cannot see, so it reports the measurement and stops.
_COVERAGE_TARGET = 0.90
# A model that folds back on itself inside its own frame has pixels with no
# unique ray. A margin this small means the fold sits just outside the corner,
# so it is not biting yet but nothing was measured out there either.
_FOLD_MARGIN_WARN_DEG = 5.0


def _intrinsics_coverage_payload(state: GatewayState) -> dict[str, Any]:
    """Per-camera edge coverage of the intrinsics production is actually using.

    Read from the shipped producer JSONs rather than from any fresh fit: the
    question this answers is "is what production consumes good enough", and a
    re-fit that has not been adopted cannot answer it.

    Coverage is the one property of an intrinsics set that a reprojection score
    cannot express. Held-out RMSE is computed where the board went, so a lens
    whose outer ring was never sampled scores just as well as one that was
    fully covered -- the model is simply extrapolating over the rest of the
    frame with nothing to contradict it.
    """
    run = (state.calibration.intrinsicsRun or "").strip()
    payload: dict[str, Any] = {
        "ok": True,
        "run": run,
        "coverageTarget": _COVERAGE_TARGET,
        "foldMarginWarnDeg": _FOLD_MARGIN_WARN_DEG,
        "cameras": [],
    }
    if not run:
        payload["error"] = "生产配置未指定内参 run（calibration.intrinsics_run_name）"
        return payload

    root = state.repo_root / "outputs" / "calibration" / run / "converted"
    payload["source"] = str(root)
    if not root.is_dir():
        payload["error"] = f"找不到内参目录：{root}"
        return payload

    cameras: list[dict[str, Any]] = []
    for directory in sorted(p for p in root.glob("*") if p.is_dir()):
        data = _read_json_file(directory / "intrinsics_producer.json")
        if not data:
            continue
        name = str(data.get("camera_name") or directory.name.split("_")[0])
        entry: dict[str, Any] = {
            "camera": name,
            "serial": str(data.get("camera_serial") or ""),
            "model": str(data.get("model") or ""),
        }
        # Absent for intrinsics that did not come from a metrology self-cal
        # (vendor files carry no self_calibration block). Reporting nothing is
        # the honest answer there; a missing measurement is not a passing one.
        self_cal = data.get("self_calibration")
        if isinstance(self_cal, dict):
            fold = self_cal.get("radial_fold_deg")
            bearing = self_cal.get("corner_bearing_deg")
            margin = None
            if isinstance(fold, (int, float)) and isinstance(bearing, (int, float)):
                # inf means the model never folds, which is not a large margin
                # but the absence of a fold; the frontend renders it as such.
                margin = None if math.isinf(float(fold)) else float(fold) - float(bearing)
            entry.update({
                "coverage": _as_float(self_cal.get("observed_radius_fraction")),
                "foldMarginDeg": margin,
                "foldsInsideFrame": bool(margin is not None and margin <= 0.0),
                "framesUsed": int(self_cal.get("frames_used") or 0),
                "heldoutRmsePx": _as_float(self_cal.get("heldout_time_block_rmse_px")),
            })
        cameras.append(entry)

    payload["cameras"] = cameras
    return payload


def _world_frame_payload(state: GatewayState) -> dict[str, Any]:
    """Everything the calibration page needs to describe the world's state."""
    root = _world_root(state)
    reference = _read_json_file(root / _WORLD_REFERENCE_FILE)
    registration = _read_json_file(root / _WORLD_REGISTRATION_FILE)
    graph = _read_json_file(root / _WORLD_GRAPH_FILE) or {}

    reference_summary: dict[str, Any] | None = None
    if reference:
        reference_summary = {
            "exists": True,
            "world_frame_id": reference.get("world_frame_id", ""),
            "created_utc": reference.get("created_utc", ""),
            "calibration_id": reference.get("calibration_id", ""),
            "definition": reference.get("definition", ""),
            "cameras": sorted((reference.get("cameras") or {}).keys()),
            "revisions": reference.get("revisions") or [],
        }
    else:
        reference_summary = {"exists": False}

    bundle = _latest_bundle_report(state)
    return {
        "ok": True,
        "reference": reference_summary,
        "registration": registration,
        "stableSource": _read_json_file(root / _WORLD_STABLE_SOURCE_FILE) or {"origin": "geometry"},
        "graph": {
            "worlds": len(graph.get("nodes") or []),
            "edges": len(graph.get("edges") or []),
            "nodes": graph.get("nodes") or [],
        },
        "currentBundle": str(bundle) if bundle else "",
        "extrinsicsRun": state.calibration.extrinsicsRun,
    }


def _run_world_cli(state: GatewayState, args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    python = _cv2_python(state.repo_root)
    if python is None:
        return 1, "找不到可用的 Python 解释器（需要 numpy）"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(state.repo_root / "third_party" / "opencv_kalibr")
    command = [
        str(python),
        "-m",
        "metrology.cli.world_registration",
        "--world-dir",
        str(_world_root(state)),
        *args,
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=str(state.repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def _freeze_world_reference(state: GatewayState, *, replace: bool = False) -> dict[str, Any]:
    """Declare the current calibration to be the canonical world.

    Deliberately manual. Freezing is the act that gives ``world_frame_id`` its
    meaning, and doing it as a side effect of calibrating would defeat the point
    -- the frame is supposed to stop moving when the calibration is redone.
    """
    source = _latest_bundle_report(state)
    if source is None:
        run = state.repo_root / "outputs" / "calibration" / (state.calibration.extrinsicsRun or "")
        if not run.is_dir():
            return {
                "ok": False,
                "error": "找不到可冻结的标定：既没有自标定 BA 结果，也没有已激活的外参 run。",
            }
        source = run
    args = [
        "freeze",
        "--extrinsics",
        str(source),
        "--calibration-id",
        state.calibration.extrinsicsRun or str(source),
        "--definition",
        "canonical camera-rig world (roadmap 2.4), frozen from " + str(source),
    ]
    if replace:
        args.append("--replace")
    code, output = _run_world_cli(state, args)
    if code != 0:
        state.log("warn", f"World freeze failed: {output.splitlines()[-1] if output else code}")
        # Spread first: the payload carries its own "ok", and letting it land
        # last would turn every failure into a success.
        return {**_world_frame_payload(state), "ok": False, "error": output or f"exit {code}"}
    state.log("info", f"Canonical world frozen from {source}")
    return {**_world_frame_payload(state), "ok": True, "output": output}


def _register_world(
    state: GatewayState,
    *,
    apply_result: bool = False,
    assume_stable: list[str] | None = None,
    bundle: Path | None = None,
    use_rig_check: bool = True,
) -> dict[str, Any]:
    """Check (and optionally commit) whether this session is still in the same world.

    Precedence for "which cameras did not move": an explicit operator choice
    first, then the image-based self-check when it is usable (see
    :func:`_rig_check_stable_cameras`), and only then the geometric consensus
    working it out alone. Whichever was used is reported back in
    ``stableSource`` -- picking the stable set is the decision the whole
    registration turns on, so it must never be invisible.
    """
    root = _world_root(state)
    if not (root / _WORLD_REFERENCE_FILE).is_file():
        return {
            **_world_frame_payload(state),
            "ok": False,
            "error": "尚未冻结基准世界系。先在本面板点「冻结为基准世界系」，之后每次标定才有比较对象。",
        }
    bundle = bundle or _latest_bundle_report(state)
    if bundle is None:
        return {
            **_world_frame_payload(state),
            "ok": False,
            "error": "找不到自标定 BA 结果（extrinsics_report.json）。先跑一次外参标定。",
        }

    stable_source: dict[str, Any] = {"origin": "operator" if assume_stable else "geometry", "cameras": list(assume_stable or [])}
    if not assume_stable and use_rig_check:
        reference = _read_json_file(root / _WORLD_REFERENCE_FILE) or {}
        stable_source = _rig_check_stable_cameras(state, str(reference.get("created_utc", "")))
        if stable_source["origin"] == "rig_check":
            assume_stable = list(stable_source["cameras"])

    args = ["register", "--current", str(bundle), "--calibration-id", state.calibration.extrinsicsRun or ""]
    if assume_stable:
        args += ["--assume-stable", *assume_stable]
    if apply_result:
        args.append("--apply")
    code, output = _run_world_cli(state, args)
    # Exit code 2 is "continuity broken", which is a verdict rather than a
    # failure: the registration ran and its answer is written out.
    if code not in (0, 2):
        state.log("warn", f"World registration failed: {output.splitlines()[-1] if output else code}")
        return {**_world_frame_payload(state), "ok": False, "error": output or f"exit {code}"}

    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / _WORLD_STABLE_SOURCE_FILE).write_text(
            json.dumps(stable_source, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # not worth failing a good registration over
        state.log("warn", f"Could not record the stable-camera source: {exc}")

    payload = _world_frame_payload(state)
    payload["stableSource"] = stable_source
    registration = payload.get("registration") or {}
    world_state = registration.get("world_continuity_state", "?")
    state.log(
        "info" if code == 0 else "warn",
        f"World registration: {world_state} — {registration.get('guidance', '')}",
    )
    return {**payload, "ok": True, "output": output}


def _world_registration_for_export(state: GatewayState, bundle: Path) -> dict[str, Any] | None:
    """Read-only registration run before the production export.

    Its only job is to decide whether the export may keep the canonical
    ``world_frame_id`` or must mint a new island. It never commits: adopting a
    moved camera into the reference is a decision an operator makes after
    reading the verdict, not a side effect of a calibration finishing.
    """
    if not (_world_root(state) / _WORLD_REFERENCE_FILE).is_file():
        return None
    result = _register_world(state, apply_result=False, bundle=bundle)
    if not result.get("ok"):
        return None
    return result.get("registration")


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


# Cached by mtime rather than read once at boot: the pointer is edited by hand
# and rewritten by every deploy, so a value captured at startup would report a
# stale answer as the current one -- which is the exact failure this comparison
# exists to catch.
_PRODUCTION_RUNS_CACHE: dict[str, Any] = {}


def _production_calibration_runs(state: GatewayState) -> dict[str, str]:
    """What the tracker config says production will actually load, read now.

    ``state.calibration.*Run`` is not this. It starts equal to the config at
    boot, and a solve overwrites it in memory with the run it just produced --
    which never reaches the config, because nothing writes those keys. Reading
    the file separately is what makes the two comparable at all.
    """
    import yaml

    path = state.repo_root / _TRACKING_CONFIG
    out: dict[str, str] = {"configPath": str(_TRACKING_CONFIG), "intrinsicsRun": "", "extrinsicsRun": "", "error": ""}
    try:
        mtime = path.stat().st_mtime_ns
    except OSError as exc:
        out["error"] = f"读不到生产配置：{exc}"
        return out
    cached = _PRODUCTION_RUNS_CACHE.get("value")
    if cached is not None and _PRODUCTION_RUNS_CACHE.get("mtime") == mtime:
        return dict(cached)
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        calib = config.get("calibration") or {}
        out["intrinsicsRun"] = str(calib.get("intrinsics_run_name", "") or "").strip()
        out["extrinsicsRun"] = str(calib.get("fixed_camera_run_name", "") or "").strip()
    except (OSError, yaml.YAMLError) as exc:
        out["error"] = f"生产配置解析失败：{exc}"
        return out
    _PRODUCTION_RUNS_CACHE["mtime"] = mtime
    _PRODUCTION_RUNS_CACHE["value"] = dict(out)
    return out


def _calibration_pointer_mismatch(state: GatewayState, production: dict[str, str]) -> dict[str, Any]:
    """Where the last solve's output and the production pointer disagree.

    Empty when they agree. This is deliberately not silenced after the operator
    has seen it once: on 2026-08-20 a solve produced a calibration that recorded
    "cam_09 has moved" in its own summary, the panel showed it as live, and
    production went on loading the previous run for seven days -- the reprojection
    gate then discarding cam_09 from 1675 of 1680 frames. Nothing in the UI said
    so, because the only place the new name was written was this process's memory.
    """
    if production.get("error"):
        return {}
    fields = (
        ("intrinsics", "内参", state.calibration.intrinsicsRun, production.get("intrinsicsRun", "")),
        ("extrinsics", "外参", state.calibration.extrinsicsRun, production.get("extrinsicsRun", "")),
    )
    differing = [
        {"kind": kind, "label": label, "solved": solved, "production": live}
        for kind, label, solved, live in fields
        if solved and live and solved != live
    ]
    if not differing:
        return {}
    names = " / ".join(item["label"] for item in differing)
    return {
        "fields": differing,
        "configPath": production.get("configPath", ""),
        "message": (
            f"最近解出的{names} run 与生产实际加载的不是同一个。"
            f"解算<b>不会</b>自动改生产指针——要生效必须编辑 {production.get('configPath', '')}，"
            "否则下一条轨迹仍然用旧标定。"
        ),
    }


_PROMOTION_LOG = Path("outputs") / "calibration" / "promotions.jsonl"


def _calibration_root(state: GatewayState) -> Path:
    return state.repo_root / "outputs" / "calibration"


def _tracker_camera_model(state: GatewayState) -> str:
    """What projection model the tracking config expects its intrinsics to be."""
    import yaml

    path = state.repo_root / _TRACKING_CONFIG
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return ""
    tracker = config.get("cube_tracker") or {}
    return str(tracker.get("camera_model", "") or "").strip()


def _promotion_candidates(state: GatewayState, production: dict[str, str]) -> dict[str, str]:
    """Which run each pointer would be promoted to, if the operator says so.

    Two sources, in order. The last solve in this process comes first because it
    is what the operator just watched finish. Failing that, the newest run on
    disk -- which is what survives a gateway restart, and a restart is precisely
    what erased the evidence in August: the panel reloaded the config, agreed
    with production, and the seven days of wrong calibration went unremarked.
    """
    candidates: dict[str, str] = {}
    for kind, live in (
        ("intrinsics", production.get("intrinsicsRun", "")),
        ("extrinsics", production.get("extrinsicsRun", "")),
    ):
        solved = (
            state.calibration.intrinsicsRun if kind == "intrinsics" else state.calibration.extrinsicsRun
        ) or ""
        if solved and solved != live:
            candidates[kind] = solved
            continue
        newer = promotion.promotable_runs(
            _calibration_root(state),
            suffix="_intrinsics" if kind == "intrinsics" else "_extrinsics",
            live_run=live,
            require_model=_tracker_camera_model(state) if kind == "intrinsics" else "",
        )
        if newer:
            candidates[kind] = newer[0]["run"]
    return candidates


# Keyed on what can change the answer: the config's pointers, the set of runs on
# disk, and whichever run the last solve produced. Without it every snapshot poll
# re-reads two run summaries and stats the whole calibration directory, and the
# panel polls continuously while a session is open.
_PROMOTION_REVIEW_CACHE: dict[str, Any] = {}


def _promotion_review_key(state: GatewayState, production: dict[str, str]) -> tuple[Any, ...]:
    root = _calibration_root(state)
    try:
        runs_mtime = root.stat().st_mtime_ns if root.is_dir() else 0
    except OSError:
        runs_mtime = 0
    return (
        # The repo root is part of the identity, not decoration: without it two
        # states sharing pointer names would read each other's cached review.
        str(state.repo_root),
        production.get("intrinsicsRun", ""),
        production.get("extrinsicsRun", ""),
        state.calibration.intrinsicsRun,
        state.calibration.extrinsicsRun,
        runs_mtime,
    )


def _promotion_review(state: GatewayState, production: dict[str, str]) -> dict[str, Any]:
    """The comparison the operator has to see before promoting anything.

    Built on every snapshot rather than on demand, because a review the operator
    has to go and ask for is a review that gets skipped -- and the whole reason
    this exists is that the one step which was optional (editing a YAML file by
    hand) is the step that did get skipped, for seven days.

    Note what is absent: any ranking, score or recommendation. The reprojection
    RMSE is carried through only so it can be shown labelled as not-a-criterion.
    It ranked the two August runs backwards (0804 scored 0.244 px against 0820's
    0.273 px, and 0804 was the one missing a moved camera), so a panel that
    sorted by it would have argued for the wrong run.
    """
    if production.get("error"):
        return {}
    key = _promotion_review_key(state, production)
    if _PROMOTION_REVIEW_CACHE.get("key") == key:
        return copy.deepcopy(_PROMOTION_REVIEW_CACHE["value"])
    candidates = _promotion_candidates(state, production)
    if not candidates:
        _PROMOTION_REVIEW_CACHE.update({"key": key, "value": {}})
        return {}

    root = _calibration_root(state)
    review: dict[str, Any] = {"candidates": candidates, "configPath": production.get("configPath", "")}

    if "extrinsics" in candidates:
        live_run = production.get("extrinsicsRun", "")
        comparison = promotion.compare_runs(
            promotion.load_run(root / live_run, live_run) if live_run else promotion.RunPoses(),
            promotion.load_run(root / candidates["extrinsics"], candidates["extrinsics"]),
        )
        review["extrinsics"] = comparison
        review["extrinsicsBlockers"] = promotion.promotion_blockers(comparison)

    if "intrinsics" in candidates:
        live_run = production.get("intrinsicsRun", "")
        comparison = promotion.compare_intrinsics_runs(
            promotion.load_intrinsics_run(root / live_run, live_run)
            if live_run
            else promotion.IntrinsicsRun(),
            promotion.load_intrinsics_run(root / candidates["intrinsics"], candidates["intrinsics"]),
            tracker_model=_tracker_camera_model(state),
        )
        review["intrinsics"] = comparison
        review["intrinsicsBlockers"] = promotion.intrinsics_blockers(comparison)

    _PROMOTION_REVIEW_CACHE.update({"key": key, "value": copy.deepcopy(review)})
    return review


def _promote_calibration(
    state: GatewayState,
    kinds: Sequence[str],
    *,
    acknowledge: Sequence[str] = (),
    note: str = "",
) -> dict[str, Any]:
    """Write the chosen runs into the tracking config, or say why not.

    This is the only writer of those two keys in the repository. Before this
    existed the sole way to make a solve take effect was a hand edit, which is
    the kind of step that gets skipped precisely when it matters most -- after a
    long solve, at the end of a session, by someone who has already seen the
    panel say the new run is live.
    """
    if state.calibration.state == "running":
        return {"ok": False, "error": "解算正在运行，等它结束再提升"}

    wanted = [kind for kind in kinds if kind in promotion.POINTER_KEYS]
    if not wanted:
        return {"ok": False, "error": "没有指定要提升什么（intrinsics / extrinsics）"}

    _PRODUCTION_RUNS_CACHE.clear()
    _PROMOTION_REVIEW_CACHE.clear()
    production = _production_calibration_runs(state)
    if production.get("error"):
        return {"ok": False, "error": production["error"]}

    review = _promotion_review(state, production)
    candidates = review.get("candidates") or {}
    missing = [kind for kind in wanted if kind not in candidates]
    if missing:
        return {
            "ok": False,
            "error": "没有可提升的" + "、".join("内参" if k == "intrinsics" else "外参" for k in missing)
            + " run——生产加载的已经是最新的了",
        }

    acknowledged = {str(k) for k in acknowledge}
    outstanding: list[dict[str, str]] = []
    for kind in wanted:
        for blocker in review.get(f"{kind}Blockers") or []:
            if blocker["kind"] not in acknowledged:
                outstanding.append({**blocker, "target": kind})
    if outstanding:
        return {
            "ok": False,
            "error": "提升被拦下：" + " ".join(item["message"] for item in outstanding),
            "blockers": outstanding,
            "hint": "这些是需要人确认的风险，不是错误。确认无误后带上 acknowledge 再提升。",
        }

    path = state.repo_root / _TRACKING_CONFIG
    try:
        original = path.read_text(encoding="utf-8")
        updated, changes = promotion.rewrite_pointers(
            original, {kind: candidates[kind] for kind in wanted}
        )
    except (OSError, promotion.PointerWriteError) as exc:
        return {"ok": False, "error": f"改写生产配置失败：{exc}"}
    if updated == original:
        return {"ok": False, "error": "生产配置没有变化——指针已经指向这些 run 了"}

    try:
        promotion.write_config_atomically(path, updated)
    except OSError as exc:
        return {"ok": False, "error": f"写生产配置失败：{exc}"}

    record = promotion.promotion_record(
        changes=changes,
        comparison=review.get("extrinsics") or review.get("intrinsics") or {},
        acknowledged=sorted(acknowledged),
        note=note,
    )
    try:
        promotion.append_promotion_log(state.repo_root / _PROMOTION_LOG, record)
    except OSError as exc:
        # The config write already succeeded and is what production reads, so a
        # log that could not be appended is worth a warning, not a rollback.
        state.log("warn", f"Promotion succeeded but the log could not be written: {exc}")

    _PRODUCTION_RUNS_CACHE.clear()
    _PROMOTION_REVIEW_CACHE.clear()
    _load_active_calibration_runs(state)
    summary = "，".join(f"{change['key']} → {change['to']}" for change in changes)
    state.log("info", f"Calibration promoted: {summary}")
    state.calibration.message = f"已提升为生产标定：{summary}"
    return {"ok": True, "changes": changes, "record": record}


class StaleCalibrationError(RuntimeError):
    """Trajectory generation would silently use a calibration that is not the newest."""

    def __init__(self, message: str, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


def _stale_calibration_gate(state: GatewayState) -> dict[str, Any]:
    """Whether a newer promotable calibration exists than the one production loads.

    This is the only one of the three checks that fires without anyone looking at
    a panel, and it is the one that would actually have caught the August
    incident: promotion is a button somebody has to press, and the whole lesson
    of those seven days is that the manual step is the step that gets missed.
    Here the trajectory itself refuses to be generated against a stale pointer.

    Deliberately not phrased as an error. Generating against the older
    calibration is sometimes exactly right -- reproducing an earlier result, for
    one -- so this asks rather than forbids. What is never right is not being
    told which calibration a trajectory was built on.
    """
    production = _production_calibration_runs(state)
    if production.get("error"):
        return {}
    root = _calibration_root(state)
    # Lens runs the tracker's declared model rules out are not "newer", they are
    # unusable -- see promotable_runs on why this matters more than it sounds.
    model = _tracker_camera_model(state)
    stale = {
        kind: promotion.promotable_runs(
            root,
            suffix=suffix,
            live_run=production.get(key, ""),
            require_model=model if kind == "intrinsics" else "",
        )
        for kind, suffix, key in (
            ("intrinsics", "_intrinsics", "intrinsicsRun"),
            ("extrinsics", "_extrinsics", "extrinsicsRun"),
        )
    }
    messages = [
        promotion.stale_pointer_refusal(
            rows,
            kind_label="内参" if kind == "intrinsics" else "外参",
            live_run=production.get("intrinsicsRun" if kind == "intrinsics" else "extrinsicsRun", ""),
        )
        for kind, rows in stale.items()
    ]
    messages = [text for text in messages if text]
    if not messages:
        return {}
    return {
        "message": " ".join(messages),
        "stale": {kind: rows for kind, rows in stale.items() if rows},
        "live": {
            "intrinsicsRun": production.get("intrinsicsRun", ""),
            "extrinsicsRun": production.get("extrinsicsRun", ""),
            "configPath": production.get("configPath", ""),
        },
    }


def _calibration_session_payload(state: GatewayState) -> dict[str, Any]:
    session = state.calibration_session
    return {
        "active": session.active,
        "stage": session.stage,
        "datasetName": session.datasetName,
        "datasetRoot": session.datasetRoot,
        "currentIndex": session.currentIndex,
        "message": session.message,
        "episodeTimeS": session.episodeTimeS,
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


# A sweep shorter than this cannot cover a frame's corners at a walking pace;
# longer than this is a mis-typed number, and every camera's segment is decoded
# frame by frame during the solve.
_CALIBRATION_SEGMENT_MIN_S = 5.0
_CALIBRATION_SEGMENT_MAX_S = 300.0
_CALIBRATION_SEGMENT_DEFAULT_S = 30.0


def _parse_calibration_segment_seconds(raw: str, fallback: float) -> float:
    """Validate an operator-supplied segment length, in seconds."""
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        seconds = float(text)
    except ValueError as exc:
        raise ValueError(f"每段时长要填数字，收到 {raw!r}") from exc
    if not _CALIBRATION_SEGMENT_MIN_S <= seconds <= _CALIBRATION_SEGMENT_MAX_S:
        raise ValueError(
            f"每段时长要在 {_CALIBRATION_SEGMENT_MIN_S:g}–{_CALIBRATION_SEGMENT_MAX_S:g} 秒之间"
        )
    return seconds


def _set_calibration_segment_seconds(state: GatewayState, seconds_arg: str) -> dict[str, Any]:
    """Change the length of the sweeps still to be recorded."""
    session = state.calibration_session
    if not session.active:
        return {"ok": False, "error": "没有进行中的标定会话"}
    step = session.steps[session.currentIndex] if session.currentIndex < len(session.steps) else None
    if step is not None and step.status == "recording":
        # The recorder was already told how long this one runs; changing the
        # number now would describe the segment on screen wrongly.
        return {"ok": False, "error": "本段正在录制，先保存或丢弃再改时长"}
    try:
        session.episodeTimeS = _parse_calibration_segment_seconds(seconds_arg, session.episodeTimeS)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    state.log("info", f"Calibration segment length set to {session.episodeTimeS:g}s")
    return {"ok": True, "session": _calibration_session_payload(state)}


def _start_calibration_session(
    state: GatewayState, cameras_arg: str = "", seconds_arg: str = ""
) -> dict[str, Any]:
    if state.calibration_session.active:
        return {"ok": False, "error": "标定会话已在进行中"}
    if state.calibration.state == "running":
        return {"ok": False, "error": "上一次标定解算尚未结束"}
    try:
        episode_time_s = _parse_calibration_segment_seconds(
            seconds_arg, _CALIBRATION_SEGMENT_DEFAULT_S
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

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
        episodeTimeS=episode_time_s,
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


# Recorder states in which the current episode is still open, i.e. there is
# something for save/discard to end. See _calibration_step_record.
_EPISODE_OPEN_STATES = frozenset({"recording", "review"})


def _calibration_segment_written(state: GatewayState, step: CalibrationStep) -> bool:
    """Whether the recorder already wrote this step's episode to disk.

    ``savedEpisodes`` counts the episodes the recorder wrote this session and
    ``step.episodeIndex`` was that count when this segment started, so a higher
    count means this segment landed. ``saving`` covers the sliver between the
    recorder's "Episode saved." line and the "Total saved episodes: N" line that
    carries the new count.
    """
    if state.recording.state == "saving":
        return True
    return step.episodeIndex >= 0 and int(state.recording.savedEpisodes) > step.episodeIndex


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
            _start_episode(state, session.episodeTimeS)
            step.status = "recording"
            step.note = ""
            # The index this segment will take, read before the recorder confirms:
            # savedEpisodes is the count already written, i.e. the 0-based index of
            # the one just started. It is also the reference point that tells us
            # later whether the recorder wrote this segment on its own.
            step.episodeIndex = int(state.recording.savedEpisodes)
            session.message = (
                f"正在录制 {step.camera or '外参'}——按提示挥板，"
                f"{session.episodeTimeS:g} 秒后自动收尾，挥完也可以提前点「保存本段」。"
            )
        elif action in {"save", "discard"}:
            # A segment does not necessarily end when the operator says so. The
            # recorder closes the episode itself once dataset.episode_time_s
            # elapses (thor_record: "duration_reached" -> auto-save -> "Episode N
            # ready"), and on the GMSL2 rig that is 10 s -- shorter than a board
            # sweep feels, so the recorder is usually back at "armed" by the time
            # the operator clicks. Driving it then hit the recorder's own
            # precondition ("Cannot save while recorder is armed") and dropped the
            # bookkeeping for a segment already sitting on disk. So only send
            # save/discard while the episode is still open; otherwise register
            # what the recorder already decided.
            episode_open = state.recording.state in _EPISODE_OPEN_STATES
            if episode_open:
                _stop_recorder(state, action)
                written = action == "save"
            else:
                written = _calibration_segment_written(state, step)
            if action == "save":
                if not written:
                    step.status = "pending"
                    session.message = "本段没有落盘，请重录一段。"
                    return {
                        "ok": False,
                        "error": (
                            f"这一段没有保存成功（录制器状态 {state.recording.state}）。"
                            "请重录一段；如果反复失败，去「采集」页看录制器输出。"
                        ),
                        "session": _calibration_session_payload(state),
                    }
                step.status = "captured"
                if not episode_open:
                    step.note = f"录满 {session.episodeTimeS:g}s 自动收尾并保存"
                # Point the solve at what the recorder actually wrote. The session
                # names a calib_<ts> dataset when it starts, but the recorder's
                # dataset root was fixed when it was spawned at Connect, so that
                # name is a label and not a path: _start_extrinsics_calibration
                # would resolve it and find no episodes/ under it.
                if state.recording.datasetRoot:
                    session.datasetRoot = state.recording.datasetRoot
                    session.datasetName = Path(state.recording.datasetRoot).name
                _calibration_session_advance(state)
            else:
                step.status = "pending"
                if written:
                    # Nothing here can un-write it, and the solver reads every
                    # episode under the dataset, so say so rather than implying
                    # the segment is gone.
                    step.note = "上一段已被录制器自动保存，无法撤回；解算时仍会被读入"
                    session.message = (
                        "本段在你点「丢弃」之前已被录制器按固定时长自动保存，无法撤回；"
                        "可以重录一段，解算会把两段都读进去。"
                    )
                else:
                    step.note = ""
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
    # Leaving the wizard must not orphan what it recorded. The episodes are on
    # disk either way; without this the only pointer to them goes with the
    # session, and the fallback scan will not find a capture whose name does not
    # happen to contain "calib".
    if state.calibration_session.datasetRoot:
        state.calibration.solveDatasetRoot = state.calibration_session.datasetRoot
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


def _marker_tcp_target_label(*, box_id: str = "", side: str = "") -> tuple[str, str]:
    box_id_norm = str(box_id or "").strip()
    if box_id_norm:
        return box_id_norm, box_id_norm
    # Backward compatibility for older saved sessions/tests/API callers. New UI
    # sends box_id and uses boxId for display; side is kept only as legacy label.
    side_norm = str(side or "").strip().lower()
    if side_norm in {"left", "right"}:
        return "", side_norm
    raise ValueError("box_id 不能为空，请选择 BOX ID")


def _marker_tcp_slug(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "").strip())
    text = text.strip("._-")
    return text or "marker_tcp"


def _resolve_user_path(state: GatewayState, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = state.repo_root / path
    return path


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _write_yaml_mapping(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _default_marker_tcp_bundle_path(state: GatewayState) -> Path | None:
    config_path = state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG
    if not config_path.is_file():
        return None
    try:
        cfg = _load_yaml_mapping(config_path)
    except Exception as exc:  # noqa: BLE001
        state.log("warn", f"Could not read marker→TCP bundle path from {config_path}: {exc}")
        return None
    ee_cfg = cfg.get("ee_from_cube") if isinstance(cfg.get("ee_from_cube"), dict) else {}
    raw = str(ee_cfg.get("marker_to_tcp_calibration_path", "") or "").strip()
    return _resolve_user_path(state, raw) if raw else None


def _load_marker_tcp_calibration_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    if not schema.startswith("marker_rig_to_tcp_calibration/"):
        raise ValueError(f"{path}: unexpected marker→TCP schema {schema!r}")
    cubes = data.get("cubes")
    if not isinstance(cubes, dict) or not cubes:
        raise ValueError(f"{path}: marker→TCP bundle has no cubes")
    return data


def _resolve_marker_tcp_calibration_file(state: GatewayState, raw_path: str) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = _resolve_user_path(state, text)
    if not path.is_file():
        raise FileNotFoundError(f"marker→TCP calibration bundle not found: {path}")
    _load_marker_tcp_calibration_bundle(path)
    return path


def _marker_tcp_cube_for_box_id(state: GatewayState, box_id: str) -> tuple[str, dict[str, Any], Path | None]:
    target = str(box_id or "").strip()
    default_path = _default_marker_tcp_bundle_path(state)
    if default_path is not None and default_path.is_file():
        bundle = _load_marker_tcp_calibration_bundle(default_path)
        cubes = bundle.get("cubes", {}) if isinstance(bundle.get("cubes"), dict) else {}
        if target in cubes and isinstance(cubes[target], dict):
            return target, cubes[target], default_path
        for name, entry in cubes.items():
            if isinstance(entry, dict) and str(entry.get("device_id", "")).strip() == target:
                return str(name), entry, default_path
    legacy = target.lower()
    if legacy in {"left", "right"}:
        return legacy, {}, default_path
    raise ValueError(
        f"默认 production marker→TCP bundle 中找不到 BOX ID {target!r}。"
        "请先登记该 BOX，或提供包含对应 T_cube_tcp/T_marker_tcp 的 CAD/真值 JSON。"
    )


def _coerce_mat4(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a 4x4 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{label} must be a 4x4 matrix")
        values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{label} contains non-finite values")
        matrix.append(values)
    return matrix


def _rotation_from_transform(value: Any, *, label: str) -> list[list[float]]:
    matrix = _coerce_mat4(value, label=label)
    rotation = [row[:3] for row in matrix[:3]]
    det = _det3(rotation)
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"{label} rotation is not proper (det={det:.6f})")
    return rotation


def _det3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _rotation_arg(rotation: list[list[float]]) -> str:
    return ";".join(",".join(f"{float(value):.12g}" for value in row) for row in rotation)


def _rotation_from_marker_tcp_truth(
    state: GatewayState,
    *,
    truth_path: Path | None,
    cube_name: str,
    box_id: str,
) -> tuple[list[list[float]] | None, str]:
    if truth_path is None:
        return None, ""
    try:
        data = json.loads(truth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 CAD/真值 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"CAD/真值 JSON root must be a mapping: {truth_path}")

    cubes = data.get("cubes") if isinstance(data.get("cubes"), dict) else {}
    candidate_names = []
    if cube_name in cubes:
        candidate_names.append(cube_name)
    if box_id in cubes and box_id not in candidate_names:
        candidate_names.append(box_id)
    for name, entry in cubes.items():
        if isinstance(entry, dict) and str(entry.get("device_id", "")).strip() == box_id and str(name) not in candidate_names:
            candidate_names.append(str(name))
    for name in candidate_names:
        entry = cubes.get(name)
        if isinstance(entry, dict) and "T_cube_tcp" in entry:
            rotation = _rotation_from_transform(entry["T_cube_tcp"], label=f"{truth_path}:cubes.{name}.T_cube_tcp")
            return rotation, f"operator supplied CAD/truth rotation from {truth_path}: cubes.{name}.T_cube_tcp"

    for key in ("T_cube_tcp", "T_marker_tcp", "T_marker_to_tcp", "T_rig_tcp"):
        if key in data:
            rotation = _rotation_from_transform(data[key], label=f"{truth_path}:{key}")
            return rotation, f"operator supplied CAD/truth rotation from {truth_path}: {key}"

    schema = str(data.get("schema", ""))
    if schema.startswith("marker_rig_cad/") or schema.startswith("marker_layout/"):
        state.log(
            "warn",
            f"{truth_path} contains marker geometry but no TCP rotation transform; "
            "using the existing production bundle rotation if available.",
        )
        return None, f"CAD/layout geometry recorded from {truth_path}; no TCP rotation key present"
    return None, f"CAD/truth JSON recorded from {truth_path}; no supported TCP transform key present"


def _select_marker_tcp_rotation(
    state: GatewayState,
    *,
    truth_path: Path | None,
    cube_name: str,
    box_id: str,
    existing_entry: dict[str, Any],
    existing_bundle_path: Path | None,
) -> tuple[list[list[float]], str, str]:
    truth_rotation, truth_note = _rotation_from_marker_tcp_truth(
        state, truth_path=truth_path, cube_name=cube_name, box_id=box_id
    )
    if truth_rotation is not None:
        return truth_rotation, truth_note, truth_note
    if existing_entry and "T_cube_tcp" in existing_entry:
        rotation = _rotation_from_transform(existing_entry["T_cube_tcp"], label=f"{existing_bundle_path}:cubes.{cube_name}.T_cube_tcp")
        source = (
            f"rotation inherited from existing production bundle {existing_bundle_path}; "
            "the fixed-point pivot solve updates translation only because a single pivot point cannot observe rotation"
        )
        return rotation, source, truth_note
    raise ValueError(
        "无法确定 marker rig→TCP 的旋转。单点 pivot 只能估计 TCP 原点；"
        "请提供包含 T_cube_tcp/T_marker_tcp/T_marker_to_tcp 的 CAD/真值 JSON，"
        "或先让默认 production bundle 包含该 BOX 的旋转。"
    )


# Every module the metrology solve chain imports at top level: cv2 for the
# detector, scipy for the bundle adjustment, yaml/numpy everywhere.
_MARKER_TCP_REQUIRED_MODULES = ("cv2", "numpy", "scipy", "yaml")


def _marker_tcp_python_candidates(repo_root: Path) -> list[Path]:
    home = Path.home()
    return [
        repo_root / "third_party" / "opencv_kalibr" / ".venv" / "bin" / "python3",
        repo_root / ".venv-fr3" / "bin" / "python",
        home / "Code" / "infer" / ".venv-fr3" / "bin" / "python",
        home / "Codes" / "infer" / ".venv-fr3" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python3",
        Path(sys.executable),
        Path("/usr/bin/python3"),
    ]


def _marker_tcp_python(state: GatewayState) -> Path:
    """Pick an interpreter that can actually run the metrology solve chain.

    Probing beats guessing here: on Thor the repo's own ``.venv`` has cv2 but no
    scipy, so the obvious choice decodes 7 cameras x ~1800 frames of 1080p and
    only then dies importing ``scipy.optimize`` inside the bundle adjustment.
    A one-second import check up front turns that into an error message.
    """
    tried: list[str] = []
    seen: set[str] = set()
    for candidate in _marker_tcp_python_candidates(state.repo_root):
        if not candidate.is_file() or str(candidate) in seen:
            continue
        seen.add(str(candidate))
        probe = subprocess.run(
            [str(candidate), "-c", f"import {', '.join(_MARKER_TCP_REQUIRED_MODULES)}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
        missing = (probe.stderr or "").strip().splitlines()
        tried.append(f"{candidate}: {missing[-1] if missing else 'import failed'}")
    raise RuntimeError(
        "找不到能运行 metrology 解算链的 python（需要 "
        + ", ".join(_MARKER_TCP_REQUIRED_MODULES)
        + "）。已尝试：\n"
        + "\n".join(f"  - {line}" for line in tried)
    )


def _marker_tcp_tool_env(state: GatewayState) -> dict[str, str]:
    env = _tool_env(state.repo_root)
    metrology_root = state.repo_root / "third_party" / "opencv_kalibr"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(metrology_root), existing]) if existing else str(metrology_root)
    return env


def _run_marker_tcp_command(
    state: GatewayState,
    command: list[str],
    *,
    label: str,
    log_path: Path,
    timeout_s: int = 7200,
) -> subprocess.CompletedProcess[str]:
    session = state.marker_tcp_session
    session.stage = "solving"
    session.message = label
    _save_marker_tcp_session(state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
    try:
        proc = subprocess.run(
            command,
            cwd=str(state.repo_root),
            env=_marker_tcp_tool_env(state),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output if isinstance(exc.output, str) else ""
        with log_path.open("a", encoding="utf-8") as handle:
            if output:
                handle.write(output)
            handle.write(f"\n[TIMEOUT] {label} exceeded {timeout_s}s\n")
        raise RuntimeError(f"{label} 超时：{timeout_s}s 内没有结束") from exc
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(proc.stdout or "")
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        raise RuntimeError(f"{label} 失败(exit={proc.returncode})" + (f": {tail}" if tail else ""))
    return proc


def _marker_tcp_saved_samples(session: MarkerTcpSession, box_id: str) -> list[MarkerTcpSample]:
    target = str(box_id or "").strip()
    samples = []
    for sample in session.samples:
        if sample.status != "saved" or sample.episodeIndex < 0 or not sample.datasetRoot:
            continue
        if target and target not in {str(sample.boxId or "").strip(), str(sample.side or "").strip()}:
            continue
        samples.append(sample)
    return sorted(samples, key=lambda item: (item.datasetRoot, item.episodeIndex, item.id))


def _marker_tcp_subset_dataset(
    state: GatewayState,
    *,
    solve_dir: Path,
    samples: list[MarkerTcpSample],
) -> tuple[Path, list[dict[str, Any]]]:
    roots = {sample.datasetRoot for sample in samples if sample.datasetRoot}
    if len(roots) != 1:
        raise ValueError("本次解算要求样本来自同一个 datasetRoot；请对每个录制会话分别解算")
    dataset_root = _resolve_user_path(state, next(iter(roots)))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"datasetRoot not found: {dataset_root}")

    subset_root = solve_dir / "input_dataset"
    episodes_root = subset_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, Any]] = []
    seen: set[int] = set()
    for sample in samples:
        episode = int(sample.episodeIndex)
        if episode in seen:
            continue
        seen.add(episode)
        source = dataset_root / "episodes" / f"episode_{episode:06d}"
        if not source.is_dir():
            raise FileNotFoundError(f"episode dir not found for sample {sample.id}: {source}")
        if not any(source.glob("cam_*.mkv")):
            raise FileNotFoundError(f"episode has no cam_*.mkv videos: {source}")
        dest = episodes_root / f"episode_{len(mapping):06d}"
        try:
            os.symlink(source.resolve(), dest, target_is_directory=True)
            link_mode = "symlink"
        except OSError:
            shutil.copytree(source, dest, symlinks=True)
            link_mode = "copytree"
        mapping.append(
            {
                "sampleId": sample.id,
                "condition": sample.condition,
                "sourceDatasetRoot": str(dataset_root),
                "sourceEpisodeIndex": episode,
                "subsetEpisodeIndex": len(mapping),
                "sourceEpisodeDir": str(source),
                "subsetEpisodeDir": str(dest),
                "mode": link_mode,
            }
        )
    if not mapping:
        raise ValueError("没有可用于解算的 saved 样本")
    _write_marker_tcp_json(subset_root / "marker_tcp_episode_map.json", {"episodes": mapping})
    return subset_root, mapping


def _marker_tcp_tracking_cameras(state: GatewayState, subset_root: Path, episodes: list[int]) -> list[str]:
    """Cameras to detect on: those recorded in every episode AND calibrated.

    Detection is the expensive half of the solve (1080p decode per camera per
    frame), and a camera with no extrinsics contributes nothing to the pose --
    ``track_marker_rig_in_base`` skips it outright. Intersecting up front keeps
    the cost proportional to what the solve can actually use.
    """
    per_episode: list[set[str]] = []
    for episode in episodes:
        episode_dir = subset_root / "episodes" / f"episode_{episode:06d}"
        per_episode.append({path.stem for path in episode_dir.glob("cam_*.mkv")})
    recorded = set.intersection(*per_episode) if per_episode else set()
    if not recorded:
        raise FileNotFoundError(f"选中的 episode 里没有所有段共有的 cam_*.mkv：{subset_root}")

    cfg = _load_yaml_mapping(state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG)
    calib = cfg.get("calibration") if isinstance(cfg.get("calibration"), dict) else {}
    summary_path = (
        _resolve_user_path(state, str(calib.get("root_dir", "outputs/calibration")))
        / str(calib.get("fixed_camera_run_name", "")).strip()
        / "summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"外参 summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    joint = summary.get("joint_solution") if isinstance(summary.get("joint_solution"), dict) else {}
    calibrated = joint.get("cameras") if isinstance(joint.get("cameras"), dict) else {}
    usable = sorted(recorded & set(calibrated))
    if not usable:
        raise ValueError(
            f"录制到的相机 {sorted(recorded)} 和已标定的相机 {sorted(calibrated)} 没有交集，无法解算"
        )
    return usable


def _marker_tcp_aruco_dictionary(state: GatewayState) -> str:
    """The dictionary the rig's markers are drawn from, per the production config.

    Detecting with the wrong dictionary does not fail loudly -- every id decodes
    to something else and the solve happily fits nothing -- so this reads the one
    place production already declares it instead of carrying a default.
    """
    cfg = _load_yaml_mapping(state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG)
    tracker = cfg.get("cube_tracker") if isinstance(cfg.get("cube_tracker"), dict) else {}
    dictionary = str(tracker.get("aruco_dictionary", "") or "").strip()
    if not dictionary:
        raise ValueError(
            f"cube_tracker.aruco_dictionary 未设置：{state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG}"
        )
    return dictionary


def _marker_tcp_layout_command(
    state: GatewayState,
    *,
    python_bin: Path,
    cube_name: str,
    out_path: Path,
    cad_path: Path | None,
) -> list[str]:
    command = [
        str(python_bin),
        "-m",
        "metrology.cli.build_rig_layout_from_cube",
        "--tracking-config",
        str(state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG),
        "--cube",
        cube_name,
        "--layout-id",
        f"{cube_name}_{out_path.parent.name}",
        "--out",
        str(out_path),
    ]
    if cad_path is not None:
        command += ["--cad-json", str(cad_path)]
    return command


def _cad_declares_rig_placement(cad_path: Path | None, cube_name: str) -> bool:
    """Whether a CAD file actually places the cube in a rig frame.

    The panel's one "CAD / 真值 JSON" field feeds two different consumers -- the
    rotation picker and the layout resolver -- and only some CAD files carry a
    ``T_rig_cube``. Handing a rotation-only file to the resolver would make it
    exit rather than fall through to the identity default, so check first.
    """
    if cad_path is None:
        return False
    try:
        data = json.loads(cad_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    keys = {"T_rig_cube", "T_rig_from_cube", "T_cube_rig", "T_cube_from_rig"}
    if keys & set(data):
        return True
    cubes = data.get("cubes")
    entry = cubes.get(cube_name) if isinstance(cubes, dict) else None
    return isinstance(entry, dict) and bool(keys & set(entry))


def _compose_rig_rotation(
    rotation_cube_tcp: list[list[float]],
    layout_path: Path,
) -> tuple[list[list[float]], bool]:
    """Carry an inherited ``R_cube_tcp`` into the frame the layout is expressed in.

    With the default identity placement the rig frame *is* the cube frame and
    this is a no-op. When CAD moves the cube into a separate rig frame, the
    tracker reports poses of the rig, so the rotation the bundle needs is
    ``R_rig_tcp = R_rig_cube @ R_cube_tcp`` -- writing the un-composed one would
    be a silent frame error of exactly the CAD rotation.
    """
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if bool(layout.get("rig_frame_is_cube_frame", True)):
        return rotation_cube_tcp, False
    T_rig_cube = _coerce_mat4(layout.get("T_rig_cube"), label=f"{layout_path}:T_rig_cube")
    R_rig_cube = [row[:3] for row in T_rig_cube[:3]]
    composed = [
        [sum(R_rig_cube[i][k] * rotation_cube_tcp[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    return composed, True


def _write_ee_trajectory_override_config(
    state: GatewayState,
    dataset_root: Path,
    marker_to_tcp_calibration_path: Path,
) -> Path:
    base_config_path = state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG
    if not base_config_path.is_file():
        raise FileNotFoundError(f"EE trajectory config not found: {base_config_path}")
    cfg = _load_yaml_mapping(base_config_path)
    ee_cfg = cfg.setdefault("ee_from_cube", {})
    if not isinstance(ee_cfg, dict):
        raise ValueError("ee_from_cube must be a mapping in EE trajectory config")
    ee_cfg["mode"] = "calibrated_marker_to_tcp"
    ee_cfg["marker_to_tcp_calibration_path"] = str(marker_to_tcp_calibration_path)
    # A bundle solved in a CAD rig frame is only half the story: the production
    # tracker still generates analytic cube corners unless it is handed the same
    # layout, and pairing a rig-frame T_rig_tcp with a cube-frame pose is a
    # silent frame error. The solve drops the layout next to the bundle exactly
    # so this can pick it up; an identity placement needs no override.
    layout_path = marker_to_tcp_calibration_path.parent / DEFAULT_MARKER_LAYOUT_NAME
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if not bool(layout.get("rig_frame_is_cube_frame", True)):
            tracker_cfg = cfg.setdefault("cube_tracker", {})
            if not isinstance(tracker_cfg, dict):
                raise ValueError("cube_tracker must be a mapping in EE trajectory config")
            tracker_cfg["marker_layout_path"] = str(layout_path)
    digest = hashlib.sha1(str(marker_to_tcp_calibration_path).encode("utf-8")).hexdigest()[:10]
    path = dataset_root / "meta" / f"ee_trajectory_config_marker_tcp_{digest}.yaml"
    _write_yaml_mapping(path, cfg)
    return path


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
        message="按 BOX ID 采集 UMI marker→TCP 样本；保存录制段后可直接解算 marker rig→TCP 并写生产 bundle。",
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


def _marker_tcp_record_sample(
    state: GatewayState,
    action: str,
    *,
    side: str = "",
    box_id: str = "",
    condition: str = "",
) -> dict[str, Any]:
    session = state.marker_tcp_session
    if not session.active or session.stage not in {"capture", "failed"}:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    action = str(action or "start").strip().lower()
    try:
        if action == "start":
            if session.pendingSampleId:
                return {"ok": False, "error": "已有样本正在录制，请先保存或丢弃"}
            box_id_norm, target_label = _marker_tcp_target_label(box_id=box_id, side=side)
            condition_text = str(condition or "").strip()
            if not condition_text:
                return {"ok": False, "error": "condition 不能为空，例如 same_mount_01 / remount_03 / light_push_x"}
            if state.recording.state in {"idle", "error"}:
                return {"ok": False, "error": "相机还没连接。请先到「采集」页 Connect，再回来采 marker→TCP 样本。"}
            _start_episode(state)
            sample = MarkerTcpSample(
                id=f"sample_{len(session.samples) + 1:03d}",
                side=target_label,
                condition=condition_text,
                boxId=box_id_norm,
                source="recording",
                status="recording",
                datasetRoot=state.recording.datasetRoot,
                episodeIndex=int(state.recording.episodeIndex),
                createdAt=datetime.now(timezone.utc).isoformat(),
            )
            session.samples.append(sample)
            session.pendingSampleId = sample.id
            session.message = f"正在录制 {target_label} · {condition_text}；结束后保存或丢弃本段。"
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
                sample.note = "raw recording saved; use solve to estimate marker rig->TCP, or register an external static_transform.json"
                session.message = "样本已保存。可继续录制同一 BOX 的其它 pivot 段，或直接点击解算写入生产 bundle。"
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
    box_id: str = "",
    condition: str = "",
) -> dict[str, Any]:
    session = state.marker_tcp_session
    if not session.active:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    try:
        box_id_norm, target_label = _marker_tcp_target_label(box_id=box_id, side=side)
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
        side=target_label,
        condition=condition_text,
        boxId=box_id_norm,
        source="static_transform",
        status="registered",
        staticTransformPath=str(p),
        createdAt=datetime.now(timezone.utc).isoformat(),
    )
    session.samples.append(sample)
    session.stage = "capture"
    session.message = f"已登记 {target_label} · {condition_text}: {p.name}"
    _save_marker_tcp_session(state)
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _run_marker_tcp_solve(
    state: GatewayState,
    *,
    box_id: str = "",
    cad_path: str = "",
    socket_beyond_tcp_mm: str = "0",
    background: bool = True,
) -> dict[str, Any]:
    """Validate the request, then run the solve chain off the request thread.

    Everything that can be checked cheaply -- the BOX, its cube in the bundle,
    the saved samples, the socket offset, the rotation provenance -- is checked
    here so a bad request still gets a real error response. The four subprocess
    steps that follow take minutes (detection alone decodes every frame of every
    camera), which no browser or proxy will hold a POST open for, so they run in
    a worker and the panel follows ``stage``/``message`` through the snapshot it
    already polls. ``background=False`` runs them inline, for tests.
    """
    session = state.marker_tcp_session
    if not session.active:
        return {"ok": False, "error": "没有进行中的 marker→TCP 采集会话"}
    if session.pendingSampleId:
        return {"ok": False, "error": "还有样本正在录制，请先保存或丢弃当前段"}
    if session.stage == "solving":
        return {"ok": False, "error": "已有解算在进行中，请等它结束"}
    try:
        box_id_norm, target_label = _marker_tcp_target_label(box_id=box_id)
        cube_name, existing_entry, existing_bundle_path = _marker_tcp_cube_for_box_id(state, box_id_norm)
        samples = _marker_tcp_saved_samples(session, box_id_norm)
        if not samples:
            return {"ok": False, "error": f"{target_label} 没有 saved 样本，先录制并保存至少 1 段 pivot 数据"}
        try:
            socket_mm = float(str(socket_beyond_tcp_mm or "0").strip())
        except ValueError as exc:
            raise ValueError("球心到 TCP 偏置必须是数字，单位 mm；球心与 TCP 重合时填 0") from exc
        if not math.isfinite(socket_mm):
            raise ValueError("球心到 TCP 偏置必须是有限数字")

        truth_path = None
        if str(cad_path or "").strip():
            truth_path = _resolve_user_path(state, cad_path)
            if not truth_path.is_file():
                raise FileNotFoundError(f"CAD/真值 JSON not found: {truth_path}")
        rotation, rotation_source, truth_note = _select_marker_tcp_rotation(
            state,
            truth_path=truth_path,
            cube_name=cube_name,
            box_id=box_id_norm,
            existing_entry=existing_entry,
            existing_bundle_path=existing_bundle_path,
        )
    except Exception as exc:  # noqa: BLE001
        session.stage = "failed"
        session.message = str(exc)
        _save_marker_tcp_session(state)
        return {"ok": False, "error": str(exc), "markerTcp": _marker_tcp_session_payload(state)}

    plan = {
        "box_id_norm": box_id_norm,
        "target_label": target_label,
        "cube_name": cube_name,
        "existing_bundle_path": existing_bundle_path,
        "samples": samples,
        "socket_mm": socket_mm,
        "truth_path": truth_path,
        "rotation": rotation,
        "rotation_source": rotation_source,
        "truth_note": truth_note,
    }
    if not background:
        return _marker_tcp_solve_worker(state, plan)

    session.stage = "solving"
    session.message = f"{target_label}：解算已开始，正在准备数据集子集…"
    _save_marker_tcp_session(state)
    Thread(
        target=_marker_tcp_solve_worker,
        args=(state, plan),
        daemon=True,
        name=f"marker-tcp-solve-{_marker_tcp_slug(box_id_norm)}",
    ).start()
    return {"ok": True, "markerTcp": _marker_tcp_session_payload(state)}


def _marker_tcp_solve_worker(state: GatewayState, plan: dict[str, Any]) -> dict[str, Any]:
    session = state.marker_tcp_session
    box_id_norm = plan["box_id_norm"]
    target_label = plan["target_label"]
    cube_name = plan["cube_name"]
    existing_bundle_path = plan["existing_bundle_path"]
    samples = plan["samples"]
    socket_mm = plan["socket_mm"]
    truth_path = plan["truth_path"]
    rotation = plan["rotation"]
    rotation_source = plan["rotation_source"]
    truth_note = plan["truth_note"]
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        solve_dir = Path(session.sessionRoot) / f"solve_{_marker_tcp_slug(box_id_norm)}_{timestamp}"
        solve_dir.mkdir(parents=True, exist_ok=True)
        log_path = solve_dir / "solve.log"
        python_bin = _marker_tcp_python(state)
        tracking_config_path = state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG
        subset_root, episode_map = _marker_tcp_subset_dataset(state, solve_dir=solve_dir, samples=samples)
        episodes = [int(entry["subsetEpisodeIndex"]) for entry in episode_map]
        cameras = _marker_tcp_tracking_cameras(state, subset_root, episodes)

        # 1. Resolve the layout. The cube geometry did not change when the rig
        #    moved, so the corner table comes from the production cube template
        #    rather than being re-estimated; only its placement can come from CAD.
        layout_path = solve_dir / DEFAULT_MARKER_LAYOUT_NAME
        _run_marker_tcp_command(
            state,
            _marker_tcp_layout_command(
                state,
                python_bin=python_bin,
                cube_name=cube_name,
                out_path=layout_path,
                cad_path=truth_path if _cad_declares_rig_placement(truth_path, cube_name) else None,
            ),
            label=f"解算 {target_label}: resolve marker layout",
            log_path=log_path,
        )
        if not layout_path.is_file():
            raise FileNotFoundError(f"resolved marker layout was not written: {layout_path}")
        rotation, rig_frame_composed = _compose_rig_rotation(rotation, layout_path)
        if rig_frame_composed:
            rotation_source = (
                f"{rotation_source}; composed into the CAD rig frame as R_rig_tcp = R_rig_cube @ R_cube_tcp "
                f"using T_rig_cube from {layout_path}"
            )

        # 2. Detect once, cache the corners, then track. Splitting these is what
        #    lets a re-solve skip the expensive half.
        detections_path = solve_dir / "detections.npz"
        _run_marker_tcp_command(
            state,
            [
                str(python_bin),
                "-m",
                "metrology.cli.detect_rig_markers",
                "--dataset-root",
                str(subset_root),
                "--episodes",
                *[str(episode) for episode in episodes],
                "--cameras",
                *cameras,
                "--dictionary",
                _marker_tcp_aruco_dictionary(state),
                "--out",
                str(detections_path),
            ],
            label=f"解算 {target_label}: detect markers ({len(cameras)} cameras x {len(episodes)} episodes)",
            log_path=log_path,
        )
        if not detections_path.is_file():
            raise FileNotFoundError(f"marker detections were not written: {detections_path}")

        # 3. Corner-level BA over every camera at once. Writes the same two CSVs
        #    the production cube tracker writes, so the pivot tool below runs on
        #    a distributed rig and on this relocated cube unchanged.
        tracking_run_dir = solve_dir / "tracking_run"
        _run_marker_tcp_command(
            state,
            [
                str(python_bin),
                "-m",
                "metrology.cli.track_marker_rig_in_base",
                "--detections",
                str(detections_path),
                "--layout-json",
                str(layout_path),
                "--config",
                str(tracking_config_path),
                "--rig-name",
                cube_name,
                "--out-dir",
                str(tracking_run_dir),
            ],
            label=f"解算 {target_label}: track marker rig in robot base",
            log_path=log_path,
        )
        if not tracking_run_dir.is_dir():
            raise FileNotFoundError(f"tracking run was not written: {tracking_run_dir}")

        bundle_path = solve_dir / "marker_to_tcp_calibration.json"
        if existing_bundle_path is not None and existing_bundle_path.is_file():
            shutil.copy2(existing_bundle_path, bundle_path)
        pivot_report_path = solve_dir / "pivot_report.json"
        calibration_id = f"pivot_{timestamp}_{box_id_norm}"
        pivot_command = [
            str(python_bin),
            "-m",
            "metrology.cli.pivot_marker_tcp_calibration",
            str(tracking_run_dir),
            "--cube",
            cube_name,
            "--fps",
            str(float(state.config.get("dataset", {}).get("fps") or 60.0)),
            "--out",
            str(pivot_report_path),
            "--emit-marker-to-tcp",
            str(bundle_path),
            "--calibration-id",
            calibration_id,
            "--device-id",
            box_id_norm,
            "--socket-beyond-tcp-mm",
            f"{socket_mm:.9g}",
            "--rotation-cube-tcp",
            _rotation_arg(rotation),
            "--rotation-source",
            rotation_source,
        ]
        _run_marker_tcp_command(
            state,
            pivot_command,
            label=f"解算 {target_label}: pivot fit + write production bundle",
            log_path=log_path,
        )
        if not bundle_path.is_file():
            raise FileNotFoundError(f"marker→TCP bundle was not written: {bundle_path}")
        pivot_report = json.loads(pivot_report_path.read_text(encoding="utf-8")) if pivot_report_path.is_file() else {}
        summary_path = solve_dir / "solve_summary.json"
        solve_summary = {
            "schema": "marker_tcp_gui_solve/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boxId": box_id_norm,
            "cubeName": cube_name,
            "targetLabel": target_label,
            "datasetRoot": episode_map[0]["sourceDatasetRoot"],
            "episodes": episode_map,
            "subsetDatasetRoot": str(subset_root),
            "trackingConfigPath": str(tracking_config_path),
            "trackingRunPath": str(tracking_run_dir),
            "markerLayoutPath": str(layout_path),
            "rigFrameIsCubeFrame": not rig_frame_composed,
            "detectionsPath": str(detections_path),
            "cameras": cameras,
            "solverPython": str(python_bin),
            "pivotReportPath": str(pivot_report_path),
            "productionBundlePath": str(bundle_path),
            "cadTruthPath": "" if truth_path is None else str(truth_path),
            "cadTruthNote": truth_note,
            "socketBeyondTcpMm": socket_mm,
            "rotationSource": rotation_source,
            "socketMovedBetweenEpisodes": bool(pivot_report.get("socket_moved_between_episodes", False)),
            "primaryFit": pivot_report.get("primary_fit", ""),
            "pivotP95Mm": (pivot_report.get("fit") or {}).get("residual_mm", {}).get("p95") if isinstance(pivot_report, dict) else None,
            "logPath": str(log_path),
        }
        _write_marker_tcp_json(summary_path, solve_summary)

        session.solvePath = str(bundle_path)
        session.solveSummaryPath = str(summary_path)
        session.pivotReportPath = str(pivot_report_path)
        session.trackingRunPath = str(tracking_run_dir)
        # Back to "capture", not "done": solving one BOX does not close the
        # session, and "done" would lock out recording the next sample or the
        # second BOX. The result lives in solvePath and the message.
        session.stage = "capture"
        p95 = solve_summary.get("pivotP95Mm")
        warning = "；注意 socket 在 episode 间移动" if solve_summary["socketMovedBetweenEpisodes"] else ""
        session.message = (
            f"marker→TCP 解算完成：{target_label} -> {bundle_path}"
            + (f"，pivot p95={float(p95):.2f} mm" if isinstance(p95, (int, float)) else "")
            + warning
        )
        state.log("info", f"Marker→TCP solve written: {bundle_path}")
    except Exception as exc:  # noqa: BLE001
        session.stage = "failed"
        session.message = str(exc)
        _save_marker_tcp_session(state)
        return {"ok": False, "error": str(exc), "markerTcp": _marker_tcp_session_payload(state)}
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


# --------------------------------------------------------------------------- #
# hand-eye (AX = XB): the rotation half of marker rig -> TCP                    #
# --------------------------------------------------------------------------- #
#
# The pivot fixture above measures translation and is structurally blind to
# rotation -- a ball-and-socket joint is a 3-DoF spherical pair, so the mounting
# rotation lives in the null space of what it observes and no number of frames
# recovers it. The production bundle therefore still carries a *declared*
# rotation_sigma_deg = 2.0, which on the current lever arm is 9.1 mm: the single
# largest line in a 3 mm budget, and the only one that was never measured.
#
# Hand-eye is the fixture that does observe it. Mount the BOX on the FR3 flange,
# drive it to a set of poses, and read each pose twice -- once from FK, once from
# the camera rig. This gateway side only shells out; everything real is in
# metrology.hand_eye, which takes pose pairs and nothing else and is therefore
# not blocked by the gripper redesign that blocks the acquisition.
_HAND_EYE_REQUIRED_MODULES = ("numpy",)


def _hand_eye_python(state: GatewayState) -> Path:
    """An interpreter that can run the hand-eye solve.

    Deliberately a much weaker requirement than ``_marker_tcp_python``: the
    solver is numpy-only, with the SO(3) log/exp written out rather than
    imported from scipy, precisely so that the machine holding the data can
    always run it. On Thor neither venv has scipy.
    """
    tried: list[str] = []
    seen: set[str] = set()
    for candidate in _marker_tcp_python_candidates(state.repo_root):
        if not candidate.is_file() or str(candidate) in seen:
            continue
        seen.add(str(candidate))
        probe = subprocess.run(
            [str(candidate), "-c", f"import {', '.join(_HAND_EYE_REQUIRED_MODULES)}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
        missing = (probe.stderr or "").strip().splitlines()
        tried.append(f"{candidate}: {missing[-1] if missing else 'import failed'}")
    raise RuntimeError(
        "找不到能运行 hand-eye 解算的 python（只需要 numpy）。已尝试：\n"
        + "\n".join(f"  - {line}" for line in tried)
    )


def _hand_eye_output_root(state: GatewayState) -> Path:
    root = state.repo_root / "outputs" / "metrology" / "hand_eye"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_hand_eye_command(state: GatewayState, args: list[str], *, timeout_s: int = 900) -> dict[str, Any]:
    """Run the CLI and hand back its report plus its exit code.

    The exit code is carried through rather than collapsed into ok/not-ok. Its
    whole point is to distinguish "not observable" (3) from "mis-associated" (5)
    from "solved but out of budget" (4) from "solved with no uncertainty at
    all" (6), and a panel that only knew ok/failed would be the thing that
    turns a refusal back into a confident number.
    """
    python = _hand_eye_python(state)
    command = [str(python), "-m", "metrology.cli.hand_eye_calibration", *args]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(state.repo_root),
        env=_marker_tcp_tool_env(state),
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-20000:],
        "stderr": (proc.stderr or "")[-8000:],
        "command": command,
    }


def _run_hand_eye_solve(
    state: GatewayState,
    *,
    pairs_path: str = "",
    t_flange_box_path: str = "",
    lever_mm: str = "",
    pairing: str = "all",
) -> dict[str, Any]:
    raw = str(pairs_path or "").strip()
    if not raw:
        return {"ok": False, "error": "需要一个 pose-pair JSON 路径"}
    try:
        pairs = _resolve_user_path(state, raw)
        if not pairs.is_file():
            raise FileNotFoundError(f"pose-pair JSON 不存在: {pairs}")
        cad = None
        if str(t_flange_box_path or "").strip():
            cad = _resolve_user_path(state, t_flange_box_path)
            if not cad.is_file():
                raise FileNotFoundError(f"T_flange_box JSON 不存在: {cad}")
        lever = float(str(lever_mm).strip()) if str(lever_mm or "").strip() else None
        if lever is not None and (not math.isfinite(lever) or lever <= 0):
            raise ValueError("杠杆臂必须是正数，单位 mm")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _hand_eye_output_root(state) / f"hand_eye_{stamp}.json"
    args = [str(pairs), "--out", str(out_path), "--pairing", pairing]
    if cad is not None:
        args += ["--t-flange-box", str(cad)]
    if lever is not None:
        args += ["--lever-mm", str(lever)]

    try:
        run = _run_hand_eye_command(state, args)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    report: dict[str, Any] | None = None
    if out_path.is_file():
        try:
            report = json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001
            report = None

    verdict = (report or {}).get("verdict") or {}
    why = verdict.get("why") if isinstance(verdict, dict) else None
    if run["returncode"] == 0:
        error = ""
    elif isinstance(why, str) and why.strip():
        error = why.strip()
    else:
        tail = (run["stderr"] or "").strip().splitlines()
        error = tail[-1] if tail else "hand-eye 解算失败"

    # returncode 0 is the only "solved and accepted" outcome; every other code is
    # a specific, named refusal, and it is carried through rather than collapsed
    # so the panel can say which one happened.
    return {
        "ok": run["returncode"] == 0,
        "returncode": run["returncode"],
        "verdict": verdict,
        "report": report,
        "reportPath": str(out_path) if out_path.is_file() else "",
        "stdout": run["stdout"],
        "stderr": run["stderr"],
        "error": error,
    }


def _run_hand_eye_plan(
    state: GatewayState,
    *,
    poses: str = "",
    pose_noise_deg: str = "",
    pose_noise_mm: str = "",
    trials: str = "",
    lever_mm: str = "",
) -> dict[str, Any]:
    """Size the capture that does not exist yet.

    This is the half of the item that is useful *today*: the acquisition is
    blocked on the gripper-rig design, so what can be produced now is its
    specification -- how many poses, at what per-pose orientation noise, to get
    the rotation inside the budget.
    """
    try:
        pose_list = str(poses or "6,8,12,16,24,32").strip()
        for part in pose_list.split(","):
            if part.strip() and int(part) <= 0:
                raise ValueError("位姿数必须是正整数")
        noise = float(str(pose_noise_deg).strip()) if str(pose_noise_deg or "").strip() else 0.10
        noise_mm = float(str(pose_noise_mm).strip()) if str(pose_noise_mm or "").strip() else 0.5
        trial_count = int(str(trials).strip()) if str(trials or "").strip() else 40
        trial_count = max(5, min(trial_count, 200))
        lever = float(str(lever_mm).strip()) if str(lever_mm or "").strip() else 102.3
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"参数不合法: {exc}"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _hand_eye_output_root(state) / f"hand_eye_plan_{stamp}.json"
    args = [
        "--plan",
        "--plan-poses", pose_list,
        "--plan-pose-noise-deg", str(noise),
        "--plan-pose-noise-mm", str(noise_mm),
        "--plan-trials", str(trial_count),
        "--lever-mm", str(lever),
        "--out", str(out_path),
    ]
    try:
        run = _run_hand_eye_command(state, args, timeout_s=600)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    if run["returncode"] != 0:
        tail = (run["stderr"] or "").strip().splitlines()
        return {"ok": False, "error": tail[-1] if tail else "capture plan 失败", "stdout": run["stdout"]}

    payload = json.loads(out_path.read_text()) if out_path.is_file() else {}
    return {
        "ok": True,
        "plan": payload,
        "planPath": str(out_path) if out_path.is_file() else "",
        "stdout": run["stdout"],
    }


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


def _solve_dataset(state: GatewayState, dataset_arg: str = "") -> tuple[Path | None, str]:
    """Which capture the next solve reads, and on whose authority.

    Order: what the caller named, then what the operator picked, then what the
    guided session recorded, then the newest directory that looks like a sweep.
    The session's capture counts *whatever stage the session is in* -- a solve
    that failed leaves its episodes on disk, and still pointing at them is the
    whole of being able to retry. Requiring stage "ready" here is what made a
    failed solve unretryable: the fallback scan wants "calib" in the directory
    name, and the recorder names captures after the rig, not after the wizard.
    """
    session = state.calibration_session
    # A named capture is an instruction, not a preference: if it cannot be read
    # the answer is "missing", never a quietly substituted other dataset. An
    # operator who picked one capture and got another solved would have no way
    # to tell from the result.
    for raw in (dataset_arg, state.calibration.solveDatasetRoot):
        if not raw:
            continue
        resolved = _resolve_dataset_root(state.repo_root, str(raw))
        if resolved is None or not (resolved / "episodes").is_dir():
            return None, "missing"
        return resolved, "manual"
    if session.active and session.datasetRoot:
        resolved = _resolve_dataset_root(state.repo_root, session.datasetRoot)
        if resolved is not None and (resolved / "episodes").is_dir():
            return resolved, "session"
    newest = _newest_calibration_dataset(state)
    return (newest, "auto") if newest is not None else (None, "none")


def _set_solve_dataset(state: GatewayState, path_arg: str, kind: str = "extrinsics") -> dict[str, Any]:
    """Point one half of the solve at a capture, or clear the choice."""
    if state.calibration.state == "running":
        return {"ok": False, "error": "解算进行中，结束后再换数据集"}
    field_name = "intrinsicsDatasetRoot" if kind == "intrinsics" else "solveDatasetRoot"
    text = str(path_arg or "").strip()
    if not text:
        setattr(state.calibration, field_name, "")
        return {"ok": True, "solve": _solve_payload(state)}
    resolved = _resolve_dataset_root(state.repo_root, text)
    if resolved is None or not (resolved / "episodes").is_dir():
        return {"ok": False, "error": f"{text} 里没有 episodes/，不是一份可解算的采集"}
    setattr(state.calibration, field_name, str(resolved))
    state.log("info", f"Calibration {kind} dataset set to {resolved}")
    return {"ok": True, "solve": _solve_payload(state)}


def _solve_candidates(state: GatewayState) -> list[dict[str, Any]]:
    """Captures that could be solved: anything recorded with an episodes/ tree.

    Built from the dataset scan the snapshot already keeps, so listing them
    costs one stat each rather than a second walk of the datasets root.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in state.cached_recorded_datasets:
        path = str(item.get("path") or "")
        if not path or path in seen or not (Path(path) / "episodes").is_dir():
            continue
        seen.add(path)
        candidates.append(
            {
                "path": path,
                "name": str(item.get("name") or Path(path).name),
                "episodes": int(item.get("totalEpisodes") or 0),
                "updatedAt": str(item.get("updatedAt") or ""),
            }
        )
    # A capture recorded moments ago may not be in the scan yet, and the one the
    # solve is actually pointed at must always be selectable -- otherwise the
    # dropdown would silently disagree with the label above it.
    for extra in (
        state.calibration.solveDatasetRoot,
        state.calibration.intrinsicsDatasetRoot,
        state.calibration_session.datasetRoot,
    ):
        resolved = _resolve_dataset_root(state.repo_root, extra) if extra else None
        if resolved is None or str(resolved) in seen or not (resolved / "episodes").is_dir():
            continue
        seen.add(str(resolved))
        candidates.append(
            {
                "path": str(resolved),
                "name": resolved.name,
                "episodes": len([p for p in (resolved / "episodes").glob("episode_*") if p.is_dir()]),
                "updatedAt": "",
            }
        )
    return sorted(candidates, key=lambda item: item["updatedAt"], reverse=True)


def _production_intrinsics_cameras(state: GatewayState) -> list[str]:
    """Cameras the intrinsics run in production actually ships a lens for.

    Read from the run directory rather than the metrology report: ``outputs/``
    is excluded from the deploy sync, so on the rig the run is the only one of
    the two that exists.
    """
    run = (state.calibration.intrinsicsRun or "").strip()
    if not run:
        return []
    directory = state.repo_root / "outputs" / "calibration" / run
    summary = _read_json_file(directory / "summary.json")
    if isinstance(summary, dict):
        names = [
            str(row.get("camera_name") or "")
            for row in (summary.get("cameras") or [])
            if isinstance(row, dict)
        ]
        if any(names):
            return sorted(n for n in names if n)
    # Older runs, or one written without a summary: the per-camera directories
    # are named "<camera>_<serial>", and the camera name is the cam_NN prefix.
    found = set()
    for path in directory.glob("converted/*/intrinsics_producer.json"):
        parts = path.parent.name.split("_")
        if len(parts) >= 2:
            found.add(f"{parts[0]}_{parts[1]}")
    return sorted(found)


def _capture_cameras(dataset: Path) -> list[str]:
    """Camera names with video in a capture, i.e. the ones a fit would cover."""
    return sorted({video.stem for _, video in _capture_videos(dataset / "episodes")})


def _intrinsics_preflight(state: GatewayState, intrinsics: Path | None) -> dict[str, Any]:
    """Whether re-fitting intrinsics from this capture can survive the export.

    ``export_production_calibration`` writes a whole intrinsics run from one
    report and has no way to carry a camera forward from the run already in
    production, so every camera with video in the capture must come out of the
    fit with a usable model -- one that saw no board takes the export down at
    the last step, after the entire decode. On this rig cam_02/cam_03 point
    away from the board area and detect nothing in every episode, which makes
    "re-fit and export from a full-rig capture" structurally impossible rather
    than unlucky.

    Blocking applies only when production already ships intrinsics: a first
    calibration of a fresh rig has nothing to extend and nothing to lose.
    """
    production = _production_intrinsics_cameras(state)
    if intrinsics is None or not production:
        return {"cameras": [], "production": production, "uncalibrated": [], "blocking": False}
    cameras = _capture_cameras(intrinsics)
    uncalibrated = [name for name in cameras if name not in set(production)]
    return {
        "cameras": cameras,
        "production": production,
        "uncalibrated": uncalibrated,
        "blocking": bool(uncalibrated),
    }


def _preflight_message(preflight: dict[str, Any]) -> str:
    """The refusal, naming the cameras and the two ways past it."""
    names = "、".join(preflight.get("uncalibrated") or [])
    kept = len(preflight.get("production") or [])
    return (
        f"重算内参并导出会在最后一步失败：这份采集里 {names} 没有在产内参，"
        f"导出时它们必须各自拟合出可用的模型，任何一台看不到板都会让整轮作废（已解码的部分全部白跑）。"
        f"当前在产内参只有 {kept} 台，导出也不会把它们保留下来。"
    )


def _solve_payload(state: GatewayState) -> dict[str, Any]:
    """What the panel needs to say which capture will be solved, and offer others."""
    dataset, source = _solve_dataset(state)
    candidates = _solve_candidates(state)
    episodes = next(
        (item["episodes"] for item in candidates if dataset is not None and item["path"] == str(dataset)),
        0,
    )
    intrinsics = _resolve_dataset_root(state.repo_root, state.calibration.intrinsicsDatasetRoot)
    intrinsics_ok = intrinsics is not None and (intrinsics / "episodes").is_dir()
    return {
        "datasetRoot": str(dataset) if dataset else "",
        "datasetName": dataset.name if dataset else "",
        "episodes": episodes,
        "source": source,
        "candidates": candidates,
        "intrinsicsDatasetRoot": str(intrinsics) if intrinsics_ok else "",
        "intrinsicsDatasetName": intrinsics.name if intrinsics_ok else "",
        "intrinsicsEpisodes": next(
            (
                item["episodes"]
                for item in candidates
                if intrinsics_ok and item["path"] == str(intrinsics)
            ),
            0,
        ),
        # What the solve falls back to when intrinsics are not re-fitted.
        "intrinsicsRun": state.calibration.intrinsicsRun,
        # Whether re-fitting from that capture could survive its own export.
        "intrinsicsPreflight": _intrinsics_preflight(
            state, intrinsics if intrinsics_ok else None
        ),
    }


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
    _finish_solve_progress(state)
    state.log("warn", f"Calibration failed: {message}")
    if state.calibration_session.active:
        state.calibration_session.stage = "failed"
        state.calibration_session.message = message


# Wall-clock share of each solve step. These exist only to make one honest bar
# out of three subprocesses: ChArUco detection decodes every frame of every
# recorded video, while the bundle and the export work on corner sets that are
# already orders of magnitude smaller. Measured roughly on the 0804 capture
# (11 cameras x 12 episodes): ~35 min detect, ~7 min bundle, seconds to export.
_SOLVE_STEP_WEIGHTS: tuple[float, ...] = (0.80, 0.16, 0.04)


def _solve_fraction(
    step_index: int, done: int, total: int, weights: Sequence[float] = _SOLVE_STEP_WEIGHTS
) -> float:
    """Overall progress, 0..1, from the running step and its own units.

    A step that reports no units of its own (``total <= 0``) sits at its start
    boundary rather than inventing movement. The bar is allowed to stand still;
    it is not allowed to promise progress that has not happened -- an operator
    who has learned the bar creeps on its own stops reading it.
    """
    if step_index <= 0 or not weights:
        return 0.0
    index = min(step_index, len(weights)) - 1
    before = sum(weights[:index])
    within = max(0.0, min(1.0, done / total)) if total > 0 else 0.0
    return max(0.0, min(1.0, before + weights[index] * within))


def _solve_eta_s(fraction: float, elapsed_s: float) -> float:
    """Seconds left at the rate achieved so far; 0 when there is no basis yet.

    Below a few percent the extrapolation is dominated by interpreter startup
    and would swing by tens of minutes between polls, which reads as broken.
    """
    if fraction < 0.03 or fraction >= 1.0 or elapsed_s <= 0:
        return 0.0
    return max(0.0, elapsed_s * (1.0 - fraction) / fraction)


def _solve_progress_line(line: str) -> tuple[bool, str]:
    """Read one line of a metrology CLI as (finished a unit?, what to show).

    Kept apart from the subprocess plumbing so the two shapes it has to
    recognise -- ``detect_charuco``'s per-video table and the bundle's prose --
    can be pinned by tests instead of by watching a 40-minute run.
    """
    text = line.strip()
    if not text or set(text) <= {"-"}:
        return False, ""
    parts = text.split()
    if parts[0] == "episode" and len(parts) > 1 and parts[1] == "camera":
        return False, ""  # the table header, not a video
    if len(parts) >= 3 and parts[1].startswith("cam_"):
        # "<episode> <camera> <kept frames> <median corners>", one per video.
        if parts[2] == "-":
            return True, f"{parts[0]} · {parts[1]} · 视频打不开"
        if parts[2].isdigit():
            return True, f"{parts[0]} · {parts[1]} · {parts[2]} 帧可用"
    return False, text[:120]


# Detections are a pure function of (video, stride, board), and producing them
# is the expensive half of a solve -- tens of minutes of frame-by-frame decode.
# Keying them by *run* meant every retry redid all of it, which is exactly the
# bill the 2026-08-20 failure presented: the bundle died on a missing scipy
# after the detection pass had already finished and been thrown away.
_DETECTION_STRIDE = 2
_DETECTION_MANIFEST = "manifest.json"


def _capture_videos(episodes: Path) -> list[tuple[str, Path]]:
    """(npz stem, video) for every video the detection step will read.

    Enumerated exactly the way ``detect_charuco`` does -- ``episode_*``
    subdirectories, or the directory itself when the videos sit in it directly,
    and its ``<episode>__<camera>`` npz naming -- so "is this capture already
    detected" is answered against the same files it would write.
    """
    directories = sorted(p for p in episodes.glob("episode_*") if p.is_dir()) or [episodes]
    videos: list[tuple[str, Path]] = []
    for directory in directories:
        for video in sorted(directory.glob("cam_*.mkv")) + sorted(directory.glob("cam_*.mp4")):
            videos.append((f"{directory.name}__{video.stem}", video))
    return videos


def _charuco_video_count(episodes: Path) -> int:
    """How many videos the detection step will open, so its bar has a scale."""
    return len(_capture_videos(episodes))


def _detections_dir(state: GatewayState, dataset: Path) -> Path:
    """Where a capture's corners live. Named after the capture, not the run.

    The path digest is not decoration: intrinsics and extrinsics are two
    different captures solved in the same run, and two datasets roots can hold
    directories of the same name. Keying on the name alone would let one
    capture's corners be read as another's, which no later step could detect.
    """
    digest = hashlib.sha1(str(dataset.resolve()).encode("utf-8")).hexdigest()[:8]
    return state.repo_root / "outputs" / "metrology" / "detections" / f"{dataset.name}__{digest}"


def _detection_fingerprint(episodes: Path) -> dict[str, Any]:
    return {
        "stride": _DETECTION_STRIDE,
        "videos": {stem: _path_modified_ns(video) for stem, video in _capture_videos(episodes)},
    }


def _reusable_detections(episodes: Path, detections: Path) -> int | None:
    """How many npz can be reused as they are, or None if detection must re-run.

    Reuse requires the same set of videos, each with the mtime it had when it
    was detected, and every npz still present. Anything else re-runs the whole
    step: ``detect_charuco`` has no incremental mode, and a half-stale directory
    would be bundled without anyone noticing -- a deleted episode still voting.
    """
    manifest_path = detections / _DETECTION_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current = _detection_fingerprint(episodes)
    if manifest.get("stride") != current["stride"] or manifest.get("videos") != current["videos"]:
        return None
    stems = current["videos"]
    if not stems or any(not (detections / f"{stem}.npz").is_file() for stem in stems):
        return None
    return len(stems)


def _clear_detections(detections: Path) -> None:
    """Drop the previous corners before detecting again.

    The bundle reads whatever npz the directory holds, so an episode that has
    since been deleted would keep voting on the extrinsics if its file survived.
    """
    if not detections.is_dir():
        return
    for path in [*detections.glob("*.npz"), detections / _DETECTION_MANIFEST]:
        with suppress(OSError):
            path.unlink()


def _write_detection_manifest(episodes: Path, detections: Path) -> None:
    manifest = _detection_fingerprint(episodes)
    manifest["generatedUtc"] = _now_iso()
    with suppress(OSError):
        detections.mkdir(parents=True, exist_ok=True)
        (detections / _DETECTION_MANIFEST).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def _solve_weights(detect_videos: Sequence[int], *, export: bool = True) -> list[float]:
    """Wall-clock share of each step, given how much video each detection reads.

    Decoding dominates everything else by orders of magnitude -- an intrinsics
    capture is one sweep per camera, so its detection alone can be seven times
    the extrinsics one -- and a fixed split would put the bar at 40% while 90%
    of the work was still ahead. The fits get the remainder in the order they
    run: intrinsics fit, bundle, export.

    ``export=False`` drops the export step. Its weight is redistributed rather
    than left as a gap, so the bar still reaches 1.0 by finishing work instead
    of by being set there at the end.
    """
    videos = [max(0, int(count)) for count in detect_videos]
    total = sum(videos)
    share = [0.85 * (count / total) if total else 0.85 / max(len(videos), 1) for count in videos]
    # In step order, which interleaves: the intrinsics fit runs between the two
    # detections, so its weight cannot simply be appended after both shares.
    weights = (
        [share[0], 0.05, share[1], 0.08, 0.02]
        if len(share) == 2
        else [share[0], 0.13, 0.02]
    )
    if export:
        return weights
    weights = weights[:-1]
    scale = 1.0 / sum(weights)
    return [w * scale for w in weights]


def _progress_weights(state: GatewayState) -> Sequence[float]:
    return state.calibration.progress.weights or _SOLVE_STEP_WEIGHTS


def _begin_solve_step(
    state: GatewayState, index: int, count: int, label: str, total: int = 0
) -> None:
    progress = state.calibration.progress
    progress.stepIndex = index
    progress.stepCount = count
    progress.label = label
    progress.done = 0
    progress.total = max(0, total)
    progress.detail = ""
    progress.fraction = _solve_fraction(index, 0, progress.total, _progress_weights(state))


def _advance_solve_step(state: GatewayState, *, detail: str = "", unit_done: bool = True) -> None:
    progress = state.calibration.progress
    if unit_done:
        progress.done += 1
        if progress.total:
            progress.done = min(progress.done, progress.total)
    if detail:
        progress.detail = detail
    progress.fraction = _solve_fraction(
        progress.stepIndex, progress.done, progress.total, _progress_weights(state)
    )


def _complete_solve_step(state: GatewayState) -> None:
    """Mark the running step's own units as all accounted for."""
    progress = state.calibration.progress
    progress.done = progress.total
    progress.fraction = _solve_fraction(
        progress.stepIndex, progress.done, progress.total, _progress_weights(state)
    )


def _finish_solve_progress(state: GatewayState, *, complete: bool = False) -> None:
    """Freeze the clock on a solve that has stopped, either way it stopped."""
    progress = state.calibration.progress
    if progress.startedAt > 0:
        progress.elapsedS = round(max(0.0, time.time() - progress.startedAt), 1)
    progress.etaS = 0.0
    if complete:
        progress.fraction = 1.0
        progress.done = progress.total


def _calibration_payload(state: GatewayState) -> dict[str, Any]:
    """Calibration status with the elapsed clock read at request time.

    Elapsed cannot be computed in the browser: the rig's clock and the
    operator's have been observed minutes apart. Nor can it be advanced only
    when the solve prints a line -- the bundle adjustment prints nothing for
    minutes at a stretch, which is precisely when the operator needs to see
    that something is still alive.
    """
    payload = asdict(state.calibration)
    payload["solve"] = _solve_payload(state)
    production = _production_calibration_runs(state)
    payload["production"] = production
    # Only when there is one. An empty dict on the wire reads as "a mismatch
    # object exists" to any client that checks for the key, which is how the
    # panel came to dereference `.fields` on the agreeing case and white-screen.
    mismatch = _calibration_pointer_mismatch(state, production)
    if mismatch:
        payload["pointerMismatch"] = mismatch
    # The comparison is built unconditionally rather than behind a button: a
    # review the operator has to request is a review that gets skipped, and the
    # step that got skipped for seven days was exactly the optional one.
    review = _promotion_review(state, production)
    if review:
        payload["promotion"] = review
    progress = payload.get("progress")
    running = state.calibration.state == "running"
    if running and isinstance(progress, dict) and float(progress.get("startedAt") or 0.0) > 0:
        elapsed = max(0.0, time.time() - float(progress["startedAt"]))
        progress["elapsedS"] = round(elapsed, 1)
        progress["etaS"] = round(_solve_eta_s(float(progress.get("fraction") or 0.0), elapsed), 1)
    return payload


def _calibration_step(
    state: GatewayState,
    python: Path,
    args: list[str],
    *,
    label: str,
    timeout: int,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run one metrology CLI, reporting its output while it is still running.

    Streamed rather than ``subprocess.run``-ed because the detection step takes
    tens of minutes and prints one line per video as it goes: read at the end
    that is a log, read as it arrives it is the progress bar.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(state.repo_root / "third_party" / "opencv_kalibr")
    state.calibration.message = label
    state.log("info", f"Calibration: {label}")
    command = [str(python), *args]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(state.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        _fail_calibration(state, f"{label} 失败：{exc}")
        return None

    # stderr is drained by a thread rather than read at the end: the failure
    # message is built from its last line, and a step that fills the pipe buffer
    # while nobody reads it blocks instead of failing.
    errors: list[str] = []
    drain: Thread | None = None
    if process.stderr is not None:
        drain = Thread(
            target=lambda stream=process.stderr: errors.append(stream.read()),
            daemon=True,
            name="calibration-stderr",
        )
        drain.start()

    # The deadline is enforced by killing the process, not by checking a clock
    # between lines: a step that hangs produces no lines, which is the one case
    # a per-line check cannot catch.
    killed: list[bool] = []

    def _expire() -> None:
        killed.append(True)
        process.kill()

    watchdog = Timer(timeout, _expire)
    watchdog.start()
    lines: list[str] = []
    try:
        if process.stdout is not None:
            for raw in process.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                if on_line is not None:
                    on_line(line)
        process.wait()
    finally:
        watchdog.cancel()
    if drain is not None:
        drain.join(timeout=10)

    if killed:
        _fail_calibration(state, f"{label} 失败：{timeout}s 内没有结束，已终止")
        return None
    return subprocess.CompletedProcess(
        command, process.returncode, "\n".join(lines), "".join(errors)
    )


# How much of a camera's frame radius the board actually reached. The 0804
# calibration had to be recaptured from scratch because this came out at 48-71%:
# the distortion fit is an extrapolation past whatever the sweep covered, and it
# folded over inside the frame. The recapture that fixed it reached 96%. The
# reprojection residual cannot see this at all -- a fit is happy to be
# self-consistent over the middle of the frame.
_INTRINSICS_COVERAGE_WARN = 0.85


def _annotate_intrinsics_coverage(cameras: list[dict[str, Any]], report_path: Path) -> None:
    """Attach each camera's edge coverage from a calibrate_intrinsics report.

    Read rather than recomputed: the report already states it per camera, and a
    second implementation of "how far out did the board get" could disagree with
    the one the fit was judged by.
    """
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    entries = report.get("cameras") or {}
    for camera in cameras:
        entry = entries.get(camera["id"]) or {}
        coverage = entry.get("observed_radius_fraction")
        if coverage is None:
            continue
        camera["coverage"] = round(float(coverage), 3)
        fisheye = (entry.get("models") or {}).get("fisheye") or {}
        folds = fisheye.get("monotonic_across_frame") is False
        if folds:
            # The distortion model turns back on itself inside the frame: pixels
            # out there map to the wrong ray, or to none at all.
            camera["status"] = "fail"
            camera["intrinsicsNote"] = "畸变模型在画幅内折返——这一台要重录内参"
        elif float(coverage) < _INTRINSICS_COVERAGE_WARN:
            if camera["status"] == "pass":
                camera["status"] = "warn"
            camera["intrinsicsNote"] = (
                f"板子只走到画幅半径的 {float(coverage):.0%}，边角是外推的"
            )


def _run_extrinsics_calibration(
    state: GatewayState,
    dataset: Path,
    run_name: str,
    python: Path | None = None,
    *,
    force_redetect: bool = False,
    intrinsics_dataset: Path | None = None,
    export_production: bool = True,
) -> None:
    if python is None:
        python, missing = _solve_python(state.repo_root)
        if python is None:
            _fail_calibration(state, "找不到带 cv2 的解释器，无法标定")
            return
        if missing:
            _fail_calibration(state, _missing_modules_message(python, missing))
            return

    calib_root = state.repo_root / "outputs" / "calibration"
    intrinsics_run = calib_root / (state.calibration.intrinsicsRun or "")
    intrinsics_source: list[str] = []
    if intrinsics_dataset is None:
        # Solve against the intrinsics production is actually using, not the
        # metrology report: that report lives under outputs/, which is excluded
        # from the deploy sync and so is simply absent on the rig. Using the
        # active run also guarantees the bundle and the shipped intrinsics agree.
        if state.calibration.intrinsicsRun and intrinsics_run.is_dir():
            intrinsics_source = ["--intrinsics-run", str(intrinsics_run)]
        elif (state.repo_root / _CALIB_INTRINSICS_REPORT).is_file():
            intrinsics_source = [
                "--intrinsics-report", str(state.repo_root / _CALIB_INTRINSICS_REPORT)
            ]
        else:
            _fail_calibration(state, (
                "找不到内参：既没有已激活的内参 run（calibration.intrinsics_run_name），"
                f"也没有 {_CALIB_INTRINSICS_REPORT}"
            ))
            return

    work = state.repo_root / "outputs" / "metrology" / run_name
    base_run = calib_root / (state.calibration.extrinsicsRun or "")
    # Two captures, two detection passes. Weighted by video count because that
    # is what the time goes into: an intrinsics capture is one sweep per camera.
    plan = [intrinsics_dataset, dataset] if intrinsics_dataset is not None else [dataset]
    state.calibration.progress.weights = _solve_weights(
        [_charuco_video_count(capture / "episodes") for capture in plan],
        export=export_production,
    )
    step_count = (5 if intrinsics_dataset is not None else 3) - (0 if export_production else 1)

    # Detection is the only step that can say how much work it has: one unit per
    # video, counted the same way it enumerates them. The fits report no units,
    # so they move the bar only at their boundaries.
    def _on_line(line: str) -> None:
        unit_done, detail = _solve_progress_line(line)
        if unit_done or detail:
            _advance_solve_step(state, detail=detail, unit_done=unit_done)

    def _run(label: str, args: list[str], timeout: int) -> bool:
        proc = _calibration_step(
            state, python, args, label=label, timeout=timeout, on_line=_on_line
        )
        if proc is None:
            return False
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            _fail_calibration(state, f"{label} 失败：{detail[-1] if detail else proc.returncode}")
            return False
        return True

    def _detect(label: str, capture: Path, step_index: int) -> Path | None:
        """Corners for one capture, reusing whatever is already on disk."""
        capture_episodes = capture / "episodes"
        directory = _detections_dir(state, capture)
        reusable = None if force_redetect else _reusable_detections(capture_episodes, directory)
        if reusable is not None:
            _begin_solve_step(
                state, step_index, step_count, f"{label}（复用已有检测 {reusable} 个视频）", reusable
            )
            _complete_solve_step(state)
            state.log("info", f"Reusing {reusable} ChArUco detections from {directory}")
            return directory
        _clear_detections(directory)
        _begin_solve_step(state, step_index, step_count, label, _charuco_video_count(capture_episodes))
        args = [
            "-m", "metrology.cli.detect_charuco",
            "--episodes", str(capture_episodes),
            "--out", str(directory),
            "--stride", str(_DETECTION_STRIDE),
        ]
        if not _run(label, args, 3600):
            return None
        # Written only after the step succeeded, so a killed or crashed run
        # leaves a directory that is re-detected rather than half-reused.
        _write_detection_manifest(capture_episodes, directory)
        return directory

    fitted_intrinsics: Path | None = None
    if intrinsics_dataset is not None:
        intrinsics_detections = _detect("检测内参采集…", intrinsics_dataset, 1)
        if intrinsics_detections is None:
            return
        _begin_solve_step(state, 2, step_count, "拟合内参…")
        fitted_intrinsics = work / "intrinsics_report.json"
        work.mkdir(parents=True, exist_ok=True)
        fit_args = [
            "-m", "metrology.cli.calibrate_intrinsics",
            "--detections", str(intrinsics_detections),
            "--out", str(fitted_intrinsics),
        ]
        if not _run("拟合内参…", fit_args, 3600):
            return
        # fisheye, not rational: the report holds both, and production declares
        # fisheye (cube_tracker.camera_model). Shipping the other one would be a
        # mismatch nothing downstream can detect.
        intrinsics_source = ["--intrinsics-report", str(fitted_intrinsics), "--model", "fisheye"]

    detect_index = 3 if intrinsics_dataset is not None else 1
    detections = _detect("检测外参采集…" if intrinsics_dataset is not None else "检测 ChArUco 角点…",
                         dataset, detect_index)
    if detections is None:
        return

    _begin_solve_step(state, detect_index + 1, step_count, "多相机联合 BA…")
    bundle_args = [
        "-m", "metrology.cli.calibrate_extrinsics",
        "--detections", str(detections),
        *intrinsics_source,
        "--out", str(work),
    ]
    if not _run("多相机联合 BA…", bundle_args, 3600):
        return

    # Experiment mode stops here: the numbers are the deliverable and the
    # production pointers are left exactly as they were. Skipping the export
    # is also the only way a re-fit from a full-rig capture can finish at all,
    # since the exporter refuses a report that is missing any camera it sees.
    if not export_production:
        state.log("info", f"Experiment solve {run_name}: export skipped, production unchanged")
    else:
        export_args = [
            "-m", "metrology.cli.export_production_calibration",
            "--extrinsics-report", str(work / "extrinsics_report.json"),
            "--name", run_name,
        ]
        # Only emit intrinsics when they were just re-fitted, or when there is no
        # production run to keep. Re-solving extrinsics alone does not touch lenses.
        keep_intrinsics_run = (
            fitted_intrinsics is None
            and bool(state.calibration.intrinsicsRun)
            and intrinsics_run.is_dir()
        )
        if fitted_intrinsics is not None:
            export_args += ["--intrinsics-report", str(fitted_intrinsics), "--model", "fisheye"]
        elif not keep_intrinsics_run:
            export_args += [
                "--intrinsics-report", str(state.repo_root / _CALIB_INTRINSICS_REPORT),
                "--model", "fisheye",
            ]
        world_reference = _world_root(state) / _WORLD_REFERENCE_FILE
        registration = _world_registration_for_export(state, work / "extrinsics_report.json")
        if world_reference.is_file() and registration is not None:
            # The canonical world, not an older run's frame. Inheriting from a base
            # run re-inherits its error every time (7.9 mm RMS / 2.17 deg for the
            # 0720 legacy frame); registering onto W does not.
            export_args += ["--world-reference", str(world_reference)]
            if str(registration.get("world_continuity_state")) not in {"CONTINUOUS", "RECONNECTED"}:
                # A new island is the *safe* direction: the old world_frame_id keeps
                # meaning what it meant, and this run says plainly that it is not in
                # it. Reusing the old ID here is the one thing that would corrupt
                # history, so the export is allowed to proceed only under a new one.
                export_args += ["--allow-world-break"]
                state.log(
                    "warn",
                    "World continuity "
                    f"{registration.get('world_continuity_state')}: exporting under a new world_frame_id. "
                    f"{registration.get('guidance', '')}",
                )
        elif base_run.is_dir():
            export_args += ["--base-extrinsics", str(base_run)]
            unmoved = _unmoved_cameras(state)
            if unmoved:
                export_args += ["--align-cameras", *unmoved]
                state.log("info", f"Base-frame alignment restricted to unmoved cameras: {', '.join(unmoved)}")
        serial_map = state.repo_root / "tools" / "thor" / "gmsl2" / "camera_serial_map.yaml"
        if serial_map.is_file():
            export_args += ["--serial-map", str(serial_map)]

        _begin_solve_step(state, step_count, step_count, "导出生产标定…")
        if not _run("导出生产标定…", export_args, 600):
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
    if fitted_intrinsics is not None:
        _annotate_intrinsics_coverage(cameras, fitted_intrinsics)
    failed = [c for c in cameras if c["status"] == "fail"]
    state.calibration.cameras = cameras
    state.calibration.state = "failed" if failed else "complete"
    _finish_solve_progress(state, complete=True)
    state.calibration.lastRunAt = _now_iso()
    state.calibration.outputPath = str(work)
    state.calibration.lastRunExported = export_production
    # No run directories were written in experiment mode, so the pointers must
    # keep naming the calibration production is actually loading. Claiming this
    # run instead would make an experiment look like a deployment.
    if export_production and not keep_intrinsics_run:
        state.calibration.intrinsicsRun = f"{run_name}_intrinsics"
    if export_production:
        state.calibration.extrinsicsRun = f"{run_name}_extrinsics"
    # These two assignments are in-memory only. Nothing in this repo writes
    # `intrinsics_run_name` / `fixed_camera_run_name` back to the tracker config,
    # so a solve that finishes here has produced files production will not load
    # until somebody edits that file -- and a gateway restart quietly reverts the
    # panel to the config's answer, erasing even the appearance of a new run.
    # Say so at the moment the operator is watching, rather than leaving the
    # panel implying the new calibration is live.
    _PRODUCTION_RUNS_CACHE.clear()
    mismatch = (
        _calibration_pointer_mismatch(state, _production_calibration_runs(state))
        if export_production
        else None
    )
    if mismatch:
        state.log(
            "warn",
            "Solved calibration is NOT live: "
            + ", ".join(f"{f['kind']} {f['solved']} vs production {f['production']}" for f in mismatch["fields"])
            + f". Edit {mismatch['configPath']} to promote it.",
        )
    state.calibration.message = (
        f"BA 重投影 {float(report.get('rmse_px', float('nan'))):.4f} px，"
        f"{report.get('num_frames', 0)} 帧 / {len(cameras)} 相机"
        + ("；有相机残差过大" if failed else "")
        + ("" if export_production else "。实验模式：未导出，生产标定未改动。")
    )
    state.log(
        "warn" if failed else "info",
        f"{'Extrinsics' if export_production else 'Experiment'} calibration "
        f"{state.calibration.state}: {state.calibration.message}",
    )
    if state.calibration_session.active:
        state.calibration_session.stage = "failed" if failed else "done"
        state.calibration_session.message = state.calibration.message
    # The baseline belongs to the calibration it was taken against, so a new
    # calibration invalidates it rather than silently keeping the old frames.
    # An experiment shipped nothing, so the baseline still describes the rig
    # production is running and must survive.
    baseline_meta = _rig_check_root(state) / "baseline" / "baseline.json"
    if export_production and baseline_meta.is_file():
        baseline_meta.unlink()
        state.log("info", "Rig-check baseline cleared; capture a new one for the new calibration")


def _refuse_solve(state: GatewayState, error: str, *, hint: str = "") -> dict[str, Any]:
    """Refuse to start, and put the reason where the operator is looking.

    A guided session shows its own message, so a refusal returned only to the
    caller leaves the wizard reading "采集完成，可以解算" while nothing happens
    -- the same "is it working?" this whole change is about.
    """
    if state.calibration_session.active:
        state.calibration_session.message = error
    state.log("warn", f"Calibration not started: {error}")
    payload: dict[str, Any] = {"ok": False, "error": error}
    if hint:
        payload["hint"] = hint
    return payload


def _start_extrinsics_calibration(
    state: GatewayState,
    dataset_arg: str = "",
    *,
    force_redetect: bool = False,
    refit_intrinsics: bool = False,
    export_production: bool = True,
) -> dict[str, Any]:
    if state.calibration.state == "running":
        return {"ok": False, "error": "标定已在进行中"}
    dataset, source = _solve_dataset(state, dataset_arg)
    if dataset is None:
        if source == "missing":
            return _refuse_solve(
                state,
                "指定的采集读不到（目录不存在，或里面没有 episodes/）",
                hint="在「将解算」下拉里换一份采集。",
            )
        return _refuse_solve(
            state,
            "找不到可用的 ChArUco 采集",
            hint="先用采集页录一段挥板数据，或在「将解算」下拉里选一份带 episodes/ 的采集。",
        )

    # Checked before anything starts, not when the bundle gets there: an
    # interpreter without scipy still runs the detection pass to completion and
    # only then reports a missing module, so the operator waits out the entire
    # expensive half of a solve to be told it was never going to work.
    python, missing = _solve_python(state.repo_root)
    if python is None:
        return _refuse_solve(state, "找不到可用的 Python 解释器，无法解算")
    if missing:
        return _refuse_solve(
            state,
            _missing_modules_message(python, missing),
            hint="解算要先检测 ChArUco 角点再做 BA，缺模块的话会白等一遍检测，所以在这里就拦下来。",
        )

    # Re-fitting intrinsics is a different capture from the extrinsics sweep,
    # and asking for it without one is a mistake worth catching before the run.
    intrinsics_dataset: Path | None = None
    if refit_intrinsics:
        intrinsics_dataset = _resolve_dataset_root(
            state.repo_root, state.calibration.intrinsicsDatasetRoot
        )
        if intrinsics_dataset is None or not (intrinsics_dataset / "episodes").is_dir():
            return _refuse_solve(
                state,
                "勾了「同时重算内参」，但没有选可用的内参采集",
                hint="内参要的是逐台相机各录一段、板子走到画面四角的采集，和外参那一段不是同一份。",
            )
        # Refused here rather than discovered at the export, which is the last
        # step: by then the whole capture has been decoded twice and the hour
        # is spent. See _intrinsics_preflight for why this is structural.
        preflight = _intrinsics_preflight(state, intrinsics_dataset)
        if export_production and preflight["blocking"]:
            return _refuse_solve(
                state,
                _preflight_message(preflight),
                hint="改用「只解算，不导出」跑这一轮：BA 会把这些相机一起解出来并给出残差，"
                "只是不写进生产。要真正把它们并进生产内参，需要导出器支持从在产 run 承接未重拟的相机。",
            )

    run_name = f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
    state.calibration.state = "running"
    state.calibration.message = f"处理 {dataset.name}…"
    state.calibration.cameras = []
    # Started here rather than in the thread so the bar is on screen from the
    # click, instead of appearing whenever the thread happens to get scheduled.
    state.calibration.progress = CalibrationProgress(
        stepIndex=1,
        stepCount=(5 if intrinsics_dataset is not None else 3)
        - (0 if export_production else 1),
        label="准备解算…",
        startedAt=time.time(),
    )
    if state.calibration_session.active:
        state.calibration_session.stage = "solving"
        state.calibration_session.message = f"正在解算 {dataset.name}…"
    Thread(
        target=_run_extrinsics_calibration,
        args=(state, dataset, run_name, python),
        kwargs={
            "force_redetect": force_redetect,
            "intrinsics_dataset": intrinsics_dataset,
            "export_production": export_production,
        },
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


def _write_overlay_config(state: GatewayState, overlay: dict[str, Any]) -> Path:
    import yaml

    path = state.repo_root / "outputs" / _ACTIVE_TASK_OVERLAY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as overlay_file:
        yaml.safe_dump(overlay, overlay_file, sort_keys=False, allow_unicode=True)
    return path


def _resolve_recorder_config_path(state: GatewayState) -> Path:
    """Config path to spawn the recorder with.

    Returns the active task's overlay config when one is bound, else the
    repo's literal config file (current behaviour). Sets
    ``state.recording.datasetRoot`` to whichever dataset root will be used.
    """

    active_task = _find_task(state, state.active_task_id)
    if active_task is None or not str(active_task.get("datasetRepoId") or "").strip():
        state.active_task_id = None
        state.recording.datasetRoot = str(_dataset_config(state.config).get("root") or "")
        return state.config_path
    datasets_dir = _task_datasets_dir(state)
    if datasets_dir is None:
        raise RuntimeError(
            "Cannot record into a task without a datasets root; start the gateway "
            "with --datasets-root or set dataset.root in the config."
        )
    overlay = _build_task_overlay_config(state.config, active_task, datasets_dir)
    overlay_path = _write_overlay_config(state, overlay)
    state.recording.datasetRoot = str(_dataset_config(overlay).get("root") or "")
    state.log(
        "info",
        f"Recording into task '{active_task['name']}' dataset {active_task['datasetRepoId']} "
        f"(config {overlay_path})",
    )
    return overlay_path


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


def _training_view_command(
    state: GatewayState, dataset_root: Path, action_mode: str
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
    if not _has_lerobot_v3_data(dataset_root):
        raise ValueError(f"{dataset_root.name} is not a LeRobot v3 dataset; nothing to build a view from.")
    # Cameras and state keys come from the dataset, not from the script's defaults: those
    # defaults name another rig's cameras (observation.images.cam_1/cam_3) and would fail on
    # every FR3 recording, which uses the config's own camera keys (ee/side).
    info = _load_dataset_info(dataset_root) or {}
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    camera_keys = [
        key
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and isinstance(feature, dict)
        and feature.get("dtype") in ("video", "image")
    ]
    if not camera_keys:
        raise ValueError(f"{dataset_root.name} has no camera features to build a training view from.")

    view_name = f"{dataset_root.name}__{action_mode}"
    view_root = _training_views_root(state) / view_name
    command = [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_train_il_policy.py"),
        "--dataset-root", str(dataset_root),
        "--view-root", str(view_root),
        # The job name is what the generated train/inference configs use for their training
        # output dir and checkpoint path. Left to the script's default it is a fixed legacy
        # name, so every view built here would train into -- and overwrite -- the same
        # directory regardless of source dataset or action contract.
        "--job-name", view_name,
        "--repo-id", f"local/{view_name}",
        "--cameras", ",".join(sorted(camera_keys)),
        "--state-keys", "observation.state",
        "--action-mode", action_mode,
        # The default append selector pulls a handheld-gripper column that FR3 datasets do not
        # have; the FR3 action already carries its own gripper.
        "--action-append-selectors", "",
        "--action-append-names", "",
        "--overwrite-view",
        # Build the view only; training is a separate, deliberate step.
        "--prepare-only",
    ]
    return command, view_root


def _training_views_root(state: GatewayState) -> Path:
    return _task_exports_root(state) / TRAINING_VIEWS_DIR_NAME


def _start_training_view(state: GatewayState, raw_path: str, action_mode: str) -> None:
    """Workstation counterpart of the Thor v3 export: build a policy-ready training view."""
    if _export_is_running(state):
        raise RuntimeError("A view build is already running; wait for it to finish.")
    dataset_root = _resolve_known_dataset(state, raw_path)
    if dataset_root is None:
        raise ValueError("Dataset not found in the recorded dataset list.")
    # Views are replay candidates now, so they are resolvable here. Re-expressing an already
    # re-expressed action column would silently compose two contracts.
    if _dataset_kind(state, dataset_root) == "training_view":
        raise ValueError(f"{dataset_root.name} is already a training view; build from the recording instead.")
    command, view_root = _training_view_command(state, dataset_root, action_mode)
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
        datasetRoot=str(dataset_root),
        outputPath=str(view_root),
        selectedEpisodes=int(_processing_item_from_dataset(dataset_root).get("totalEpisodes") or 0),
        totalFrames=0,
        message=f"Building {action_mode} training view from {dataset_root.name}…",
        pid=state.export_process.pid,
    )
    state.log("info", f"Started {action_mode} training view build {dataset_root} -> {view_root}")
    Thread(
        target=_read_export_output,
        args=(state, state.export_process),
        daemon=True,
        name=f"training-view-output-{state.export_process.pid}",
    ).start()


def _start_approved_dataset_export(state: GatewayState, raw_path: str) -> None:
    if _export_is_running(state):
        raise RuntimeError("An export is already running; wait for it to finish.")
    dataset_root = _resolve_known_dataset(state, raw_path)
    if dataset_root is None:
        raise ValueError("Dataset not found in the approved/candidate dataset list.")
    processing_item = _processing_item_from_dataset(dataset_root)
    if processing_item.get("status") != "qc_pass":
        raise ValueError("Dataset must pass QC before export.")
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
    return f"View ready: {episodes} episode(s) · {frames} frames · {contract}"


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


def _ee_trajectory_command(
    state: GatewayState,
    dataset_root: Path,
    *,
    marker_to_tcp_calibration_path: Path | None = None,
) -> list[str]:
    runner_path = state.repo_root / DEFAULT_EE_TRAJECTORY_RUNNER
    config_path = state.repo_root / DEFAULT_EE_TRAJECTORY_CONFIG
    if not runner_path.is_file():
        raise FileNotFoundError(f"EE trajectory runner not found: {runner_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"EE trajectory config not found: {config_path}")
    if marker_to_tcp_calibration_path is not None:
        config_path = _write_ee_trajectory_override_config(state, dataset_root, marker_to_tcp_calibration_path)
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
    marker_to_tcp_calibration_path: str | Path | None = None,
    calibration: dict[str, str] | None = None,
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
    marker_path_text = (
        str(marker_to_tcp_calibration_path)
        if marker_to_tcp_calibration_path is not None
        else str(current_job.get("marker_to_tcp_calibration_path", "") or "")
        if isinstance(current_job, dict)
        else ""
    )
    if marker_path_text:
        job["marker_to_tcp_calibration_path"] = marker_path_text
    # Carried forward across status updates the same way the command is: the
    # completion update does not re-read the config, and a stamp that vanished
    # when the job finished would be missing from exactly the record that lasts.
    stamp = calibration if calibration is not None else current_job.get("calibration")
    if isinstance(stamp, dict) and stamp:
        job["calibration"] = {
            "intrinsicsRun": str(stamp.get("intrinsicsRun", "") or ""),
            "extrinsicsRun": str(stamp.get("extrinsicsRun", "") or ""),
            "configPath": str(stamp.get("configPath", "") or ""),
        }
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
            "marker_to_tcp_calibration_path": marker_path_text,
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


def _queue_traj_gen(
    state: GatewayState,
    dataset_root: Path,
    *,
    marker_to_tcp_calibration_path: Path | None = None,
    allow_stale_calibration: bool = False,
) -> None:
    # Checked here rather than only in the HTTP handler so that every path into
    # trajectory generation goes through it, including any future automatic one.
    if not allow_stale_calibration:
        gate = _stale_calibration_gate(state)
        if gate:
            raise StaleCalibrationError(gate["message"], gate)
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
        command = _ee_trajectory_command(
            state,
            dataset_root,
            marker_to_tcp_calibration_path=marker_to_tcp_calibration_path,
        )
        marker_path_text = "" if marker_to_tcp_calibration_path is None else str(marker_to_tcp_calibration_path)
        _update_traj_gen_meta(
            dataset_root,
            job_id=job_id,
            status="running",
            command=command,
            message=(
                f"Running AprilTag cube tracking for {dataset_root.name}"
                + (f" with marker→TCP bundle {marker_path_text}" if marker_path_text else "")
            ),
            # Which calibration this trajectory was built on, recorded next to
            # the trajectory itself. Read months later this is the only thing
            # that can answer "was this produced before or after the repoint".
            calibration=_production_calibration_runs(state),
            log_tail=[f"[traj-gen] {' '.join(command)}"],
            marker_to_tcp_calibration_path=marker_path_text or None,
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
            marker_to_tcp_calibration_path=marker_to_tcp_calibration_path,
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


def _gmsl2_dataset_stats(dataset_root: Path) -> tuple[int, int]:
    ep_dirs = _gmsl2_episode_dirs(dataset_root)
    total_frames = 0
    for ep_dir in ep_dirs:
        meta_path = ep_dir / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime
        except OSError:
            continue  # no meta.json yet (episode mid-write)
        key = str(meta_path)
        cached = _GMSL2_EP_FRAMES_MEMO.get(key)
        if cached is not None and cached[0] == mtime:
            total_frames += cached[1]
            continue
        frames = 0
        try:
            with meta_path.open() as f:
                ep_meta = json.load(f)
            dur = float(ep_meta.get("duration_s") or 0)
            fps = int(ep_meta.get("video", {}).get("fps") or 60)
            frames = int(dur * fps)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            frames = 0
        _GMSL2_EP_FRAMES_MEMO[key] = (mtime, frames)
        total_frames += frames
    return len(ep_dirs), total_frames


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
                "dataStatus": _recorded_dataset_status(dataset_root),
                "sourcePath": str(data_files[-1]) if data_files else "",
                "isLatest": latest_recorded is not None and dataset_root == latest_recorded,
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
        for data_file in sorted(data_files, key=lambda path: path.stat().st_mtime, reverse=True):
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
                    episode = _selected_episode_for_dataset(state, dataset_root, episode_options)
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
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            keys.append(str(name))
    return keys


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


# Touch pad geometry, mirrored from box_client so the replay/preview paths work
# on hosts where the ARM-only box_sdk wheel is absent. box_client.TOUCH_MODELS is
# the source of truth; this falls back to it when importable so the two cannot
# drift apart silently.
_TOUCH_MODEL_WIDTHS: dict[str, int] = {"paxini_l5325": 239, "m2020": 9}
try:  # pragma: no cover - exercised on Thor, not in host unit tests
    from tools.thor.box_sdk.box_client import TOUCH_MODELS as _BOX_TOUCH_MODELS

    _TOUCH_MODEL_WIDTHS = {
        name: int(spec["points"]) for name, spec in _BOX_TOUCH_MODELS.items()
    }
except Exception:  # noqa: BLE001 - keep the mirrored table
    pass

_TOUCH_KNOWN_WIDTHS = frozenset(_TOUCH_MODEL_WIDTHS.values())
# Sentinel for "any pad geometry we know", distinct from an explicit width and
# from None (= accept whatever came in).
_TOUCH_ANY_KNOWN_WIDTH = -1


def _touch_model_for_width(width: int) -> str | None:
    for name, points in _TOUCH_MODEL_WIDTHS.items():
        if points == int(width):
            return name
    return None


def _touch_payload_from_axes(
    fz_values: Any,
    *,
    fx_values: Any = None,
    fy_values: Any = None,
    timestamp: int = 0,
    model: str | None = None,
    expected_count: int | None = _TOUCH_ANY_KNOWN_WIDTH,
) -> dict[str, Any] | None:
    fz = _as_float_list(fz_values)
    if expected_count is _TOUCH_ANY_KNOWN_WIDTH:
        # Accept any pad geometry box_client knows about (239-taxel Paxini,
        # 9-taxel M2020, ...) but still reject a frame that is short for every
        # one of them -- that is a truncated payload, not a smaller pad.
        if len(fz) not in _TOUCH_KNOWN_WIDTHS:
            return None
    elif expected_count is not None:
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
        # Pad geometry, so the frontend picks a layout instead of inferring one
        # from array length. Frames archived before box_client tagged the model
        # fall back to the width, which is unambiguous for the pads we ship.
        "model": str(model) if model else (_touch_model_for_width(count) or "unknown"),
        "points": count,
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
        model=data.get("model") if isinstance(data.get("model"), str) else None,
    )


_TOUCH_EXPORT_COLUMNS = {
    side: {
        axis: f"observation.touch.box_touch_{side}.{axis}_0p1N"
        for axis in ("fx", "fy", "fz")
    }
    for side in ("left", "right")
}


def _touch_payload_from_fz(values: Any, *, timestamp: int = 0) -> dict[str, Any] | None:
    """Build an fz-only payload, padding up to the nearest known pad width.

    Callers here hand over a bare fz column whose width is whatever the dataset
    stored, so a short frame is padded to the smallest pad that fits rather
    than to a fixed 239 (which would inflate a 9-taxel M2020 frame back into a
    Paxini-shaped one and put the UI on the wrong layout).
    """

    fz = _as_float_list(values)
    if not fz:
        return None
    target = min((w for w in sorted(_TOUCH_KNOWN_WIDTHS) if w >= len(fz)), default=None)
    if target is None:
        fz = fz[: max(_TOUCH_KNOWN_WIDTHS)]
    elif len(fz) < target:
        fz.extend([0.0] * (target - len(fz)))
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
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()


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
        state.recording.datasetRoot = str(_dataset_config(state.config).get("root") or state.recording.datasetRoot)
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
        "configSummary": _config_summary(state.config, state.config_path),
        "devices": [
            {**device, "state": "running" if recording_state == "recording" and device["state"] != "error" else device["state"]}
            for device in state.devices
        ],
        "recording": asdict(state.recording),
        "replay": asdict(state.replay),
        "teleop": asdict(state.teleop),
        "annotation": _active_annotation(state),
        "calibration": _calibration_payload(state),
        "calibrationSession": _calibration_session_payload(state),
        "markerTcp": _marker_tcp_session_payload(state),
        "recordedDatasets": recorded_datasets,
        "processing": list(state.cached_processing_items),
        "trajectory": trajectory,
        "events": [asdict(event) for event in state.events],
        "tasks": _tasks_with_progress(state),
        "activeTaskId": state.active_task_id or "",
        "datasetExport": asdict(state.dataset_export),
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
        "pika_task_tcp",
        "--no-viewer",
        "--enable-cameras",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-fps",
        "30",
        "--disable-otg",
    ]


def _fr3_real_teleop_command(state: GatewayState) -> list[str]:
    return [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        "-m",
        "tools.fr3.fr3_real_teleop_runtime",
        f"--config_path={state.config_path}",
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
        targetFrameName="pika_task_tcp",
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
        targetFrameName="pika_task_tcp",
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


def _connect_recorder(state: GatewayState, *, backend: str | None = None) -> None:
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

    config_path = _resolve_recorder_config_path(state)
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


def _start_episode(state: GatewayState, episode_time_s: float | None = None) -> None:
    """Queue one episode, optionally overriding how long the recorder runs it.

    ``episode_time_s`` asks the recorder for a specific length for this episode
    only (``episode_time:<seconds>``), instead of the config's
    ``dataset.episode_time_s``. The calibration wizard needs this: a board sweep
    has to last as long as the operator was told to wave, and respawning the
    recorder to change a config value would re-open eleven Argus cameras.
    Only the GMSL2 recorder implements the command -- the FR3 runtime queues
    unrecognised stdin lines as commands, so sending it there would be noise in
    its state machine.
    """
    process = _ensure_recorder_running(state)
    if state.recording.state not in ("armed", "idle"):
        raise RuntimeError(f"Cannot start an episode while recorder is {state.recording.state}.")

    if episode_time_s is not None and episode_time_s > 0:
        if not _state_is_gmsl2(state):
            raise RuntimeError(
                "This recorder cannot change the episode length per episode; "
                "it always uses dataset.episode_time_s."
            )
        _write_recorder_stdin(process, f"episode_time:{episode_time_s:g}\n")
        # Keep the frame budget in step with the length actually asked for:
        # targetFrames is what flips the recorder to "review", so leaving it at
        # the config value would declare a 30 s segment finished at 20 s.
        state.recording.targetFrames = _target_frames_for_seconds(state.config, episode_time_s)
    else:
        state.recording.targetFrames = _target_frames(state.config)
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
        str(state.replay.fps or 30),
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
        str(state.replay.fps or 30),
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
                "--robot.target_frame_name=pika_task_tcp",
            ]
        )
    return command


def _real_robot_ip(state: GatewayState) -> str:
    replay = _replay_config(state.config)
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    return str(replay.get("robot_ip") or robot.get("robot_ip") or DEFAULT_REAL_ROBOT_IP)


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
        if path == "/api/calibration/rig-check":
            _json_response(self, HTTPStatus.OK, _last_rig_check(self.server.state))
            return
        if path == "/api/calibration/world-frame":
            _json_response(self, HTTPStatus.OK, _world_frame_payload(self.server.state))
            return
        if path == "/api/calibration/intrinsics-coverage":
            _json_response(self, HTTPStatus.OK, _intrinsics_coverage_payload(self.server.state))
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
                dataset_root = _resolve_known_dataset(self.server.state, requested) or (
                    self.server.state.selected_replay_root
                )
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
            try:
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
                        _connect_recorder(state, backend=requested_backend)
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
            marker_tcp_raw = (
                query.get("marker_to_tcp_calibration_path", query.get("markerTcpCalibrationPath", [""]))[0]
                or ""
            ).strip()
            try:
                with state.lock:
                    dataset_root = _resolve_known_dataset(state, requested)
                if dataset_root is None:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                    return
                marker_tcp_path = _resolve_marker_tcp_calibration_file(state, marker_tcp_raw)
                allow_stale = (
                    query.get("allow_stale_calibration", query.get("allowStaleCalibration", [""]))[0] or ""
                ).strip() in {"1", "true", "yes"}
                _queue_traj_gen(
                    state,
                    dataset_root,
                    marker_to_tcp_calibration_path=marker_tcp_path,
                    allow_stale_calibration=allow_stale,
                )
                with state.lock:
                    response = _snapshot(state)
                _json_response(self, HTTPStatus.OK, response)
            except StaleCalibrationError as exc:
                _json_response(
                    self,
                    HTTPStatus.CONFLICT,
                    {
                        "ok": False,
                        "error": str(exc),
                        "staleCalibration": exc.detail,
                        "hint": "先在标定中心提升，或带 allow_stale_calibration=1 明确用旧标定生成。",
                    },
                )
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
                if path == "/api/calibration/run":
                    dataset = (query.get("dataset", [""])[0] or "").strip()
                    truthy = {"1", "true", "yes"}
                    force = (query.get("force_redetect", [""])[0] or "").strip() in truthy
                    refit = (query.get("refit_intrinsics", [""])[0] or "").strip() in truthy
                    # Exporting is the default so an unaware caller still ships;
                    # experiment mode has to be asked for.
                    experiment = (query.get("experiment", [""])[0] or "").strip() in truthy
                    result = _start_extrinsics_calibration(
                        self.server.state,
                        dataset,
                        force_redetect=force,
                        refit_intrinsics=refit,
                        export_production=not experiment,
                    )
                    if not result.get("ok"):
                        _json_response(self, HTTPStatus.CONFLICT, result)
                        return
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/calibration/promote":
                    kinds = [
                        item
                        for value in query.get("kind", [])
                        for item in value.split(",")
                        if item.strip()
                    ]
                    acknowledge = [
                        item
                        for value in query.get("acknowledge", [])
                        for item in value.split(",")
                        if item.strip()
                    ]
                    result = _promote_calibration(
                        self.server.state,
                        [k.strip() for k in kinds],
                        acknowledge=[a.strip() for a in acknowledge],
                        note=(query.get("note", [""])[0] or "").strip(),
                    )
                    if not result.get("ok"):
                        _json_response(self, HTTPStatus.CONFLICT, result)
                        return
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/calibration/dataset":
                    result = _set_solve_dataset(
                        self.server.state,
                        (query.get("path", [""])[0] or "").strip(),
                        (query.get("kind", [""])[0] or "extrinsics").strip(),
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/session/start":
                    result = _start_calibration_session(
                        self.server.state,
                        (query.get("cameras", [""])[0] or "").strip(),
                        (query.get("seconds", [""])[0] or "").strip(),
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
                        box_id=(query.get("box_id", query.get("boxId", [""]))[0] or "").strip(),
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
                        box_id=(query.get("box_id", query.get("boxId", [""]))[0] or "").strip(),
                        condition=(query.get("condition", [""])[0] or "").strip(),
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/marker-tcp/solve":
                    result = _run_marker_tcp_solve(
                        self.server.state,
                        box_id=(query.get("box_id", query.get("boxId", [""]))[0] or "").strip(),
                        cad_path=(query.get("cad_path", query.get("cadPath", [""]))[0] or "").strip(),
                        socket_beyond_tcp_mm=(
                            query.get("socket_beyond_tcp_mm", query.get("socketBeyondTcpMm", ["0"]))[0]
                            or "0"
                        ).strip(),
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/hand-eye/solve":
                    result = _run_hand_eye_solve(
                        self.server.state,
                        pairs_path=(query.get("pairs_path", query.get("pairsPath", [""]))[0] or "").strip(),
                        t_flange_box_path=(
                            query.get("t_flange_box_path", query.get("tFlangeBoxPath", [""]))[0] or ""
                        ).strip(),
                        lever_mm=(query.get("lever_mm", query.get("leverMm", [""]))[0] or "").strip(),
                        pairing=(query.get("pairing", ["all"])[0] or "all").strip(),
                    )
                    # A refusal is a real answer here, not a server error: the
                    # report is written either way and the panel renders the
                    # verdict, so this stays 200 and carries returncode.
                    _json_response(self, HTTPStatus.OK, result)
                    return
                if path == "/api/calibration/hand-eye/plan":
                    result = _run_hand_eye_plan(
                        self.server.state,
                        poses=(query.get("poses", [""])[0] or "").strip(),
                        pose_noise_deg=(
                            query.get("pose_noise_deg", query.get("poseNoiseDeg", [""]))[0] or ""
                        ).strip(),
                        pose_noise_mm=(
                            query.get("pose_noise_mm", query.get("poseNoiseMm", [""]))[0] or ""
                        ).strip(),
                        trials=(query.get("trials", [""])[0] or "").strip(),
                        lever_mm=(query.get("lever_mm", query.get("leverMm", [""]))[0] or "").strip(),
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
                if path == "/api/calibration/session/duration":
                    result = _set_calibration_segment_seconds(
                        self.server.state, (query.get("seconds", [""])[0] or "").strip()
                    )
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
                if path == "/api/calibration/world-frame/freeze":
                    result = _freeze_world_reference(
                        self.server.state,
                        replace=(query.get("replace", [""])[0] or "").strip() in {"1", "true", "yes"},
                    )
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    _json_response(self, status, result)
                    return
                if path == "/api/calibration/world-frame/register":
                    stable = [
                        name.strip()
                        for name in (query.get("stable", [""])[0] or "").split(",")
                        if name.strip()
                    ]
                    result = _register_world(
                        self.server.state,
                        apply_result=(query.get("apply", [""])[0] or "").strip() in {"1", "true", "yes"},
                        assume_stable=stable or None,
                        use_rig_check=(query.get("rigcheck", ["1"])[0] or "1").strip()
                        not in {"0", "false", "no"},
                    )
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
                        _start_training_view(
                            self.server.state,
                            requested,
                            (query.get("action_mode", [DEFAULT_TRAINING_VIEW_ACTION_MODE])[0] or "").strip()
                            or DEFAULT_TRAINING_VIEW_ACTION_MODE,
                        )
                    else:
                        _start_approved_dataset_export(self.server.state, requested)
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
