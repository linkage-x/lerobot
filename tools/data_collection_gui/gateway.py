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
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Iterable
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

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
DEFAULT_REPLAY_MAX_EE_STEP_MM = 120.0
DEFAULT_REPLAY_MAX_GRIPPER_STEP = 0.35
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
    # Whether this profile can actually drive a real arm from a recorded episode. The real path
    # replays the cube sidecars an AprilTag rig produces; a workstation dataset has none, so the
    # request dies inside the gateway. Say so up front instead of offering a live-looking button.
    realReplaySupported: bool = True
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
class CalibrationStatus:
    state: str = "idle"
    pattern: str = "ChArUco 5x7 (mock)"
    lastRunAt: str = ""
    message: str = "Run calibration to refresh extrinsics"
    cameras: list[dict[str, Any]] = field(default_factory=list)
    outputPath: str = ""


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
    return [entry for entry in root.iterdir() if _is_dataset_root(entry)]


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
        dataset_root.resolve().relative_to(_task_exports_root(state).resolve())
    except (OSError, ValueError):
        return "recorded"
    return "exported"


_REPLAY_CANDIDATES_MEMO: tuple[float, tuple[Any, ...], list[Path]] | None = None
_REPLAY_CANDIDATES_TTL_S = 3.0
_PROCESSING_STALE_RUNNING_S = 120.0


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


def _mock_calibrate_cameras(state: GatewayState) -> None:
    cameras = [device for device in state.devices if device.get("kind") == "camera"]
    if not cameras:
        state.calibration.state = "failed"
        state.calibration.message = "No cameras configured in handheld config"
        state.calibration.cameras = []
        state.calibration.lastRunAt = _now_iso()
        state.log("warn", "Mock calibration aborted: no cameras configured")
        return

    seed = int(time.time()) % 997
    results: list[dict[str, Any]] = []
    for index, device in enumerate(cameras):
        # Deterministic-ish pseudo errors so the table feels alive without RNG imports.
        reprojection = round(0.6 + ((seed + index * 17) % 23) / 25.0, 3)  # mm
        baseline = round(80.0 + index * 95.0 + ((seed + index * 13) % 41), 1)  # mm
        if reprojection < 1.2:
            status = "pass"
        elif reprojection < 1.8:
            status = "warn"
        else:
            status = "fail"
        results.append(
            {
                "id": str(device.get("id")),
                "reprojectionMm": reprojection,
                "baselineMm": baseline,
                "status": status,
            }
        )

    overall_fail = any(item["status"] == "fail" for item in results)
    state.calibration.state = "failed" if overall_fail else "complete"
    state.calibration.lastRunAt = _now_iso()
    state.calibration.cameras = results
    state.calibration.outputPath = str(state.repo_root / "outputs" / "calibration" / f"mock_{int(time.time())}.json")
    if overall_fail:
        state.calibration.message = "Mock calibration finished with at least one fail; review cameras"
    else:
        state.calibration.message = f"Mock calibration completed for {len(results)} cameras"
    state.log(
        "warn" if overall_fail else "info",
        f"Mock calibration {state.calibration.state}: {state.calibration.message}",
    )


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

    view_root = _training_views_root(state) / f"{dataset_root.name}__{action_mode}"
    command = [
        str(_venv_python(state.repo_root, prefer_fr3=True)),
        str(state.repo_root / "tools" / "fr3" / "fr3_train_il_policy.py"),
        "--dataset-root", str(dataset_root),
        "--view-root", str(view_root),
        "--repo-id", f"local/{dataset_root.name}__{action_mode}",
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
    root = state.exports_root or (state.repo_root / DEFAULT_EXPORTS_ROOT)
    return root / "training_views"


def _start_training_view(state: GatewayState, raw_path: str, action_mode: str) -> None:
    """Workstation counterpart of the Thor v3 export: build a policy-ready training view."""
    if _export_is_running(state):
        raise RuntimeError("A view build is already running; wait for it to finish.")
    dataset_root = _resolve_known_dataset(state, raw_path)
    if dataset_root is None:
        raise ValueError("Dataset not found in the recorded dataset list.")
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
    if output.startswith("Episode ") and "written" in output:
        frames = re.search(r"\((\d+) frames\)", output)
        if frames:
            state.dataset_export.totalFrames += int(frames.group(1))
    if output.startswith("Export complete"):
        state.dataset_export.state = "complete"
    elif output.startswith("ERROR:"):
        state.dataset_export.state = "error"
    state.log("info", f"export: {output}")


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


def _recorded_dataset_items(state: GatewayState) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, dataset_root in enumerate(_complete_replay_dataset_candidates(state)):
        info = _load_dataset_info(dataset_root)
        data_files = _dataset_data_files(dataset_root)
        modified_s = _dataset_modified_s(dataset_root)
        is_gmsl2 = _has_gmsl2_episodes(dataset_root)
        if is_gmsl2:
            total_episodes, total_frames = _gmsl2_dataset_stats(dataset_root)
        else:
            total_episodes = int(info.get("total_episodes") or 0)
            total_frames = int(info.get("total_frames") or 0)
        items.append(
            {
                "path": str(dataset_root),
                "name": dataset_root.name,
                "datasetKind": _dataset_kind(state, dataset_root),
                "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_s)) if modified_s else "",
                "updatedAtMs": int(modified_s * 1000),
                "totalEpisodes": total_episodes,
                "totalFrames": total_frames,
                "dataStatus": _recorded_dataset_status(dataset_root),
                "sourcePath": str(data_files[-1]) if data_files else "",
                "isLatest": index == 0,
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

    cmd = [
        "gst-launch-1.0", "-q",
        "filesrc", f"location={mkv_path}",
        "!", "matroskademux",
        "!", "h265parse",
        "!", "nvv4l2decoder",
        "!", "nvv4l2h264enc", "bitrate=10000000",
        "iframeinterval=60", "idrinterval=60", "insert-sps-pps=1", "insert-vui=1",
        "!", "h264parse", "config-interval=-1",
        "!", "mp4mux", "faststart=true",
        "!", "filesink", f"location={tmp_path}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
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
        return _remux_mkv_to_mp4(mkv, expected_duration_s=expected_duration_s) or mkv

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
        "calibration": asdict(state.calibration),
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
    roots = [state.datasets_root, _task_exports_root(state)]
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
    max_gripper_step = _float_config(replay, "trajectory_max_gripper_step", DEFAULT_REPLAY_MAX_GRIPPER_STEP)
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

    # The verdict belongs to `mujocoValidation`, not to `safety`. `safety` describes whether the
    # hardware path is authorized, and only the real preflight can say that -- a recorded
    # trajectory scoring 25 mm instead of 20 mm is a data verdict, not a faulted rig. Writing
    # "fault" here mislabelled it and, worse, withheld the robot-free controls that are exactly
    # what an operator reaches for next; writing "ready" on a pass claimed an authorization no
    # hardware check had granted. It stays "locked" from the start of the run either way.
    if reasons:
        validation["status"] = "failed"
        validation["message"] = "MuJoCo validation failed: " + "; ".join(reasons)
        state.replay.state = "aborted"
    else:
        validation["status"] = "passed"
        validation["message"] = (
            f"MuJoCo validation passed: max {float(max_pos):.2f}mm / {float(max_rot):.2f}deg "
            f"within {max_pos_threshold:.2f}mm / {max_rot_threshold:.2f}deg"
        )
        state.replay.state = "complete"
    state.replay.mujocoValidation = validation
    _refresh_mujoco_validation_current(state)
    if dataset_root is not None and _is_dataset_root(dataset_root):
        try:
            _write_validation_store(dataset_root, state.replay.mujocoValidation)
        except OSError as exc:
            state.log("warn", f"Failed to persist MuJoCo validation: {exc}")
    state.replay.message = validation["message"]


_RECORDER_NOISE_PREFIXES = ("[TLV_LOG_UPLOAD]", "GST_ARGUS:", "NvMMLite")


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


def _real_replay_command(
    state: GatewayState,
    dataset_root: Path,
    cube_mode: str,
    robot_ip: str,
    end_effector_mode: str = "corenetic_gripper_ee",
) -> list[str]:
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
    command = [
        str(_mujoco_replay_python(state)),
        str(state.repo_root / "tools" / "fr3" / "fr3_record_preflight.py"),
        f"--workspace={state.repo_root}",
        f"--config-path={state.repo_root / 'third_party' / 'opencv_kalibr' / 'fr3_data_collection_replay' / 'replay_cube_pose_in_robot_base.thor.yaml'}",
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


def _realsense_preview_paths(state: GatewayState) -> tuple[Path, Path]:
    root = Path("/tmp") / f"lerobot_realsense_replay_{os.getpid()}"
    return root.with_suffix(".jpg"), root.with_suffix(".json")


def _realsense_preview_python(state: GatewayState) -> Path:
    candidate = state.repo_root / "third_party" / "opencv_kalibr" / ".venv" / "bin" / "python3"
    return (
        candidate
        if candidate.is_file()
        else _venv_python3(state.repo_root, prefer_fr3=state.profile == "workstation")
    )


def _stop_realsense_preview(state: GatewayState) -> None:
    process = state.realsense_preview_process
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
    state.realsense_preview_process = None


def _start_realsense_preview(state: GatewayState) -> None:
    process = state.realsense_preview_process
    if process is not None and process.poll() is None:
        return
    image_path, status_path = _realsense_preview_paths(state)
    image_path.unlink(missing_ok=True)
    status_path.unlink(missing_ok=True)
    command = [
        str(_realsense_preview_python(state)),
        str(state.repo_root / "tools" / "data_collection_gui" / "realsense_live_preview.py"),
        "--output", str(image_path),
        "--status", str(status_path),
    ]
    state.realsense_preview_process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _realsense_preview_status(state: GatewayState) -> dict[str, Any]:
    _image_path, status_path = _realsense_preview_paths(state)
    payload = _load_json_file(status_path)
    process = state.realsense_preview_process
    process_running = process is not None and process.poll() is None
    if not payload:
        return {
            "available": None,
            "running": process_running,
            "error": "Waiting for RealSense detection" if process_running else "Preview starts with real-robot replay",
        }
    payload["running"] = bool(payload.get("running")) and process_running
    return payload


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

    if not state.replay.realReplaySupported:
        # Fail here rather than 60 lines down inside _read_sidecar_cube_poses, whose
        # "No valid generated EE trajectory" reads like a missing file the operator could go
        # produce. On this profile there is no producer: the real path replays the cube sidecars
        # an AprilTag rig writes, and nothing in the FR3 workstation pipeline writes them.
        raise RuntimeError(
            "Real-robot replay is not wired for the workstation profile: it drives the AprilTag "
            "cube sidecars (derived/april_cube_tracking_in_robot_base/state_action.<cube>.csv), "
            "which a workstation recording never produces. MuJoCo replay is the validation path "
            "here."
        )

    cube_mode = str(cube_mode).strip().lower()
    if cube_mode not in ("left", "right"):
        raise ValueError(f"Real replay cube mode must be left or right, got {cube_mode!r}")
    end_effector_mode = str(end_effector_mode).strip().lower()
    if end_effector_mode not in ("pika_gripper_ee", "corenetic_gripper_ee", "fr3_ee"):
        raise ValueError("Real replay end effector must be pika_gripper_ee, corenetic_gripper_ee, or fr3_ee")
    state.replay.realReplayLog = []
    _append_real_replay_log(
        state,
        "request",
        f"cube={cube_mode} episode={state.replay.episode} robot={robot_ip or _real_robot_ip(state)} target={end_effector_mode}",
    )
    dataset_root = _require_mujoco_validation(
        state,
        cube_mode=cube_mode,
        allow_failed_override=bool(override_mujoco_failure),
    )
    validation_mode = str((state.replay.mujocoValidation or {}).get("cubeMode") or state.replay.mujocoCubeMode)
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
    _start_realsense_preview(state)
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
            with self.server.state.lock:
                image_path, _status_path = _realsense_preview_paths(self.server.state)
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
                video_path = (
                    _resolve_video_path(self.server.state, dataset_root, camera_key, episode)
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
            try:
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
                    _mock_calibrate_cameras(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
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
    state.replay.realReplaySupported = profile != "workstation"
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
