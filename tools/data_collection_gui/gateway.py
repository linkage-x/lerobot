#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_CONFIG_PATH = Path("tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml")
DEFAULT_RECORDER_SCRIPT = Path("tools/handheld/handheld_record.py")
DEFAULT_DATASETS_ROOT = Path("outputs/datasets")
DEFAULT_MUJOCO_MAX_POSITION_ERROR_MM = 20.0
DEFAULT_MUJOCO_MAX_ROTATION_ERROR_DEG = 15.0
DEFAULT_REPLAY_MAX_EE_STEP_MM = 120.0
DEFAULT_REPLAY_MAX_GRIPPER_STEP = 0.35
DEFAULT_REAL_PREFLIGHT_TIMEOUT_S = 30.0
DEFAULT_REAL_ROBOT_IP = "192.168.1.208"


@dataclass
class EventLogItem:
    id: str
    time: str
    level: str
    message: str


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
    sourcePath: str = ""
    dataStatus: str = "missing"
    trajectoryKind: str = "none"
    totalEpisodes: int = 0
    episodeOptions: list[int] = field(default_factory=list)
    recordedFrames: int = 0
    diagnostics: list[str] = field(default_factory=list)
    pid: int | None = None
    lastOutput: str = ""
    mujocoValidation: dict[str, Any] = field(default_factory=dict)


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
    datasets_root: Path | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    calibration: CalibrationStatus = field(default_factory=CalibrationStatus)
    events: list[EventLogItem] = field(default_factory=list)
    selected_replay_root: Path | None = None
    process: subprocess.Popen[str] | None = None
    replay_process: subprocess.Popen[str] | None = None
    replay_process_kind: str = ""
    process_started_at_s: float | None = None
    replay_started_at_s: float | None = None
    lock: Lock = field(default_factory=Lock)

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
    return ReplayStatus(
        dataset=str(dataset.get("repo_id") or ""),
        datasetRoot=str(dataset.get("root") or ""),
        totalFrames=_target_frames(config),
        fps=int(dataset.get("fps") or 30),
    )


_BOX_COLLECTION_DEVICE_LABELS: dict[str, str] = {
    "box_gripper": "BOX gripper (distance)",
    "box_imu": "BOX IMU (acc/gyr/euler/quat)",
    "box_trigger": "BOX trigger travel",
    "box_six_d_force": "BOX 6D force",
    "box_touch_left": "Paxini touch pad L",
    "box_touch_right": "Paxini touch pad R",
}


def _device_statuses(config: dict[str, Any]) -> list[dict[str, Any]]:
    sensors = config.get("sensors") or {}
    if not isinstance(sensors, dict):
        sensors = {}

    devices: list[dict[str, Any]] = []
    cameras_section = sensors.get("cameras")

    # GMSL2 11-channel rig: `cameras` is a flat config block (not a mapping
    # of device-id to device) and the sensor_id list is detected at connect
    # time. Surface the configured slots up front so the GUI shows what to
    # expect; the recorder then narrows to `usable` via the "Cameras:" line.
    if isinstance(cameras_section, dict) and "defaults" in cameras_section:
        defaults = cameras_section.get("defaults") if isinstance(cameras_section.get("defaults"), dict) else {}
        prefix = str(cameras_section.get("name_prefix") or "cam")
        sensor_ids = cameras_section.get("sensor_ids") or []
        detect_all = bool(cameras_section.get("detect_all", False))
        if sensor_ids:
            slot_ids = [int(x) for x in sensor_ids]
        elif detect_all:
            slot_ids = list(range(16))  # MAX96726 has up to 16 logical sids
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

    box_cfg = config.get("box_collection")
    if isinstance(box_cfg, dict) and box_cfg.get("enabled", True):
        expected = box_cfg.get("expected_devices") or list(_BOX_COLLECTION_DEVICE_LABELS)
        poll_hz = 0
        try:
            poll_hz = int(round(1.0 / float(box_cfg.get("poll_interval_s") or 0.05)))
        except (TypeError, ValueError, ZeroDivisionError):
            poll_hz = 0
        detail = f"UDP {box_cfg.get('remote_ip', '?')}:{box_cfg.get('remote_port', 15000)}"
        for sensor_id in expected:
            label = _BOX_COLLECTION_DEVICE_LABELS.get(str(sensor_id), str(sensor_id))
            devices.append(
                {
                    "id": str(sensor_id),
                    "kind": "box_collection",
                    "label": label,
                    "state": "idle",
                    "fps": poll_hz,
                    "latencyMs": 0,
                    "detail": detail,
                }
            )
    return devices


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
    return sorted((dataset_root / "data").glob("chunk-*/*.parquet"))


def _path_modified_s(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _dataset_modified_s(dataset_root: Path) -> float:
    candidates = [dataset_root, dataset_root / "meta" / "info.json", *_dataset_data_files(dataset_root)]
    return max((_path_modified_s(path) for path in candidates), default=0.0)


def _dataset_name_prefixes(name: str) -> set[str]:
    prefixes = {name}
    timestamped_match = re.match(r"^(?P<base>.+)_\d{8}_\d{6}(?:_\d{2})?$", name)
    if timestamped_match:
        prefixes.add(timestamped_match.group("base"))
    return prefixes


def _is_dataset_root(path: Path) -> bool:
    return path.is_dir() and (path / "meta" / "info.json").is_file()


def _scan_datasets_root(state: GatewayState) -> list[Path]:
    root = state.datasets_root
    if root is None or not root.is_dir():
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


def _replay_dataset_candidates(state: GatewayState) -> list[Path]:
    candidates = _complete_dataset_candidates(state)
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
    candidates = _complete_dataset_candidates(state)
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
    state.replay.dataset = str(matched)
    state.replay.message = f"Selected recorded dataset: {matched.name}"
    _invalidate_mujoco_validation(state, "Dataset changed; run MuJoCo replay again before real-robot replay.")
    persisted_validation = _load_persisted_mujoco_validation(state, matched, state.replay.episode)
    if persisted_validation is not None:
        state.replay.mujocoValidation = persisted_validation
        _refresh_mujoco_validation_current(state)
        state.replay.message = f"Selected recorded dataset: {matched.name}; MuJoCo validation restored"
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
    state.replay.message = f"Selected episode {episode} from {dataset_root.name}"
    _invalidate_mujoco_validation(state, "Episode changed; run MuJoCo replay again before real-robot replay.")
    persisted_validation = _load_persisted_mujoco_validation(state, dataset_root, episode)
    if persisted_validation is not None:
        state.replay.mujocoValidation = persisted_validation
        _refresh_mujoco_validation_current(state)
        state.replay.message = f"Selected episode {episode} from {dataset_root.name}; MuJoCo validation restored"
    state.log("info", f"Selected replay episode {episode} for {dataset_root}")


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


def _dataset_replay_meta(dataset_root: Path, info: dict[str, Any]) -> dict[str, Any]:
    data_files = _dataset_data_files(dataset_root)
    episode_options = _dataset_episode_indices(dataset_root, info)
    return {
        "datasetRoot": str(dataset_root),
        "sourcePath": str(data_files[-1]) if data_files else "",
        "totalEpisodes": int(info.get("total_episodes") or 0),
        "episodeOptions": episode_options,
        "recordedFrames": int(info.get("total_frames") or 0),
    }


def _dataset_episode_indices(dataset_root: Path, info: dict[str, Any] | None = None) -> list[int]:
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
    meta_path = dataset_root / "meta" / "processing.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as meta_file:
            payload = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _processing_item_from_dataset(dataset_root: Path) -> dict[str, Any]:
    info = _load_dataset_info(dataset_root)
    modified_s = _dataset_modified_s(dataset_root)
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
    }

    meta = _load_processing_meta(dataset_root)
    if meta:
        active_version = meta.get("active_version")
        versions = meta.get("versions") if isinstance(meta.get("versions"), dict) else {}
        current_job = meta.get("current_job") if isinstance(meta.get("current_job"), dict) else {}
        version_info = versions.get(active_version) if isinstance(active_version, str) else None
        qc = version_info.get("qc") if isinstance(version_info, dict) else None
        if isinstance(current_job, dict) and current_job.get("status") in ("queued", "running"):
            status = current_job["status"]
            message = current_job.get("message") or f"{current_job.get('kind') or 'job'} {status}"
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


def _processing_items(state: GatewayState) -> list[dict[str, Any]]:
    return [_processing_item_from_dataset(root) for root in _complete_dataset_candidates(state)]


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


def _run_qc(dataset_root: Path) -> dict[str, Any]:
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

        missing_columns = [col for col in ("action", "observation.state", "timestamp", "frame_index") if col not in table.column_names]
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
    meta_path = dataset_root / "meta" / "processing.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
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
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)
    return updated


def _queue_traj_gen(_state: GatewayState, dataset_root: Path) -> None:
    raise NotImplementedError(f"待实现：Generate EE Trajectory 功能尚未接入，dataset={dataset_root}")


def _recorded_dataset_items(state: GatewayState) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, dataset_root in enumerate(_complete_dataset_candidates(state)):
        info = _load_dataset_info(dataset_root)
        data_files = _dataset_data_files(dataset_root)
        modified_s = _dataset_modified_s(dataset_root)
        items.append(
            {
                "path": str(dataset_root),
                "name": dataset_root.name,
                "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_s)) if modified_s else "",
                "updatedAtMs": int(modified_s * 1000),
                "totalEpisodes": int(info.get("total_episodes") or 0),
                "totalFrames": int(info.get("total_frames") or 0),
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


def _read_recorded_trajectory(state: GatewayState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception:  # noqa: BLE001
        dataset_roots = _replay_dataset_candidates(state)
        latest = dataset_roots[0] if dataset_roots else None
        latest_info = _load_dataset_info(latest) if latest is not None else {}
        latest_meta = _dataset_replay_meta(latest, latest_info) if latest is not None else {}
        return [], {
            "datasetRoot": latest_meta.get("datasetRoot") or "",
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
        dataset_meta = _dataset_replay_meta(dataset_root, info)
        if best_meta is None:
            best_meta = {**dataset_meta, "dataStatus": "missing", "message": "No recorded parquet files found"}
        data_files = _dataset_data_files(dataset_root)
        for data_file in sorted(data_files, key=lambda path: path.stat().st_mtime, reverse=True):
            try:
                parquet = pq.ParquetFile(data_file)
                column_names = parquet.schema_arrow.names
                wanted_columns = [
                    column
                    for column in (
                        "episode_index",
                        "frame_index",
                        "timestamp",
                        "action",
                        "observation.state",
                        "observation.device_capture_timestamp",
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

                vector_values = action_values if len(action_values) >= 3 else state_values
                vector_names = action_names if len(action_values) >= 3 else state_names
                x_index, y_index, z_index = _axis_indices(vector_names, vector_values)

                if x_index is not None and y_index is not None and x_index < len(vector_values) and y_index < len(vector_values):
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
    candidates = _complete_dataset_candidates(state)
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


def _ee_pose_from_row(row: dict[str, Any], action_names: list[str], state_names: list[str]) -> dict[str, Any] | None:
    action_values = _as_float_list(row.get("action"))
    state_values = _as_float_list(row.get("observation.state"))

    # observation.state holds the actual robot pose (smooth); action holds commanded
    # waypoints that jump between targets. Prefer state for the EE transform.
    pose = _extract_ee_axes(state_names, state_values) or _extract_ee_axes(action_names, action_values)
    if pose is None:
        return None

    # Gripper is reported independently — prefer state, fall back to action.
    gripper = _extract_gripper(state_names, state_values)
    if gripper is None:
        gripper = _extract_gripper(action_names, action_values)

    return {**pose, "gripper": gripper}


def _extract_ee_axes(names: list[str], values: list[float]) -> dict[str, float] | None:
    if not names or not values:
        return None
    lowered = [name.lower() for name in names]
    keys: dict[str, int | None] = {
        "x": None, "y": None, "z": None,
        "qx": None, "qy": None, "qz": None, "qw": None,
    }
    suffixes = {
        "x": ("ee.x", ".x", "_x"),
        "y": ("ee.y", ".y", "_y"),
        "z": ("ee.z", ".z", "_z"),
        "qx": ("ee.qx", ".qx", "_qx", "quat.x"),
        "qy": ("ee.qy", ".qy", "_qy", "quat.y"),
        "qz": ("ee.qz", ".qz", "_qz", "quat.z"),
        "qw": ("ee.qw", ".qw", "_qw", "quat.w"),
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
    error: str | None = None,
) -> dict[str, Any]:
    """Return a fully-shaped ReplayTimeline payload with no frames.

    Every ``/api/replay/timeline`` response must satisfy the ``ReplayTimeline``
    contract the frontend declares -- in particular ``frames`` must be a list,
    not absent -- otherwise ``ReplayInspector`` blows up when it dereferences
    ``timeline.frames[currentFrame]`` before its own early-return guard runs.
    """
    payload: dict[str, Any] = {
        "datasetRoot": str(dataset_root),
        "name": dataset_root.name,
        "episode": int(episode) if episode is not None else 0,
        "totalFrames": 0,
        "frames": [],
        "fps": int(fps or 0),
        "stateNames": list(state_names or []),
        "actionNames": list(action_names or []),
        "cameraKeys": list(camera_keys or []),
        "videoTemplate": "",
        "videoChunkIndex": 0,
        "videoFileIndex": 0,
        "sourcePath": "",
    }
    if error:
        payload["error"] = error
    return payload


def _read_dataset_timeline(state: GatewayState, dataset_root: Path, episode: int | None = None) -> dict[str, Any]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        return _empty_timeline(dataset_root, error=f"pyarrow unavailable: {exc}")

    info = _load_dataset_info(dataset_root)
    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    camera_keys = _camera_keys(info)
    fps = int(info.get("fps") or state.replay.fps or 30)
    data_files = _dataset_data_files(dataset_root)
    if not data_files:
        return _empty_timeline(
            dataset_root,
            fps=fps,
            state_names=state_names,
            action_names=action_names,
            camera_keys=camera_keys,
            error="no parquet files",
        )

    data_file = data_files[-1]
    table = pq.read_table(data_file)
    if "episode_index" in table.column_names:
        episodes = [int(value) for value in table["episode_index"].to_pylist() if value is not None]
        episode_options = sorted(set(episodes))
        selected_episode = episode if episode is not None else _selected_episode_for_dataset(state, dataset_root, episode_options)
        if episode_options and selected_episode not in episode_options:
            return _empty_timeline(
                dataset_root,
                fps=fps,
                episode=selected_episode,
                state_names=state_names,
                action_names=action_names,
                camera_keys=camera_keys,
                error=f"episode {selected_episode} not found",
            )
        table = table.filter(pc.equal(table["episode_index"], selected_episode))
        episode = selected_episode
    else:
        episode = 0

    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index") or 0))

    frames: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        frame_index = int(row.get("frame_index") if row.get("frame_index") is not None else row_index)
        timestamp = _first_finite(row.get("timestamp"), default=frame_index / max(fps, 1))
        state_values = _as_float_list(row.get("observation.state"))
        action_values = _as_float_list(row.get("action"))
        pose = _ee_pose_from_row(row, action_names, state_names) or {}
        frames.append(
            {
                "frame": frame_index,
                "timestamp": timestamp,
                "state": state_values,
                "action": action_values,
                "eePose": pose,
            }
        )

    video_template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    return {
        "datasetRoot": str(dataset_root),
        "name": dataset_root.name,
        "episode": episode,
        "totalFrames": len(frames),
        "fps": fps,
        "stateNames": state_names,
        "actionNames": action_names,
        "cameraKeys": camera_keys,
        "videoTemplate": video_template,
        "videoChunkIndex": 0,
        "videoFileIndex": 0,
        "frames": frames,
        "sourcePath": str(data_file),
    }


def _resolve_video_path(state: GatewayState, dataset_root: Path, camera_key: str) -> Path | None:
    info = _load_dataset_info(dataset_root)
    template = str(info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    relative = template.format(video_key=camera_key, chunk_index=0, file_index=0)
    candidate = (dataset_root / relative).resolve()
    try:
        candidate.relative_to(dataset_root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        # fall back to scanning chunk-* directories for the first mp4
        camera_dir = dataset_root / "videos" / camera_key
        if camera_dir.is_dir():
            for mp4 in sorted(camera_dir.glob("chunk-*/*.mp4")):
                return mp4
        return None
    return candidate


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
    process = state.process
    if process is not None and process.poll() is not None:
        state.log("warn", f"Handheld recorder exited with code {process.returncode}")
        exited_from = state.recording.state
        state.process = None
        state.process_started_at_s = None
        state.recording.state = "idle" if process.returncode == 0 else "error"
        state.recording.pid = None
        state.recording.frameIndex = 0 if process.returncode == 0 else state.recording.frameIndex
        state.recording.queueDepth = 0
        if state.recording.lastOutput:
            state.recording.message = f"Recorder exited with code {process.returncode}: {state.recording.lastOutput}"
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
            state.replay.safety = "locked"
            state.replay.state = "complete" if replay_process.returncode == 0 else "aborted"
            if state.replay.lastOutput:
                state.replay.message = f"{label} exited with code {replay_process.returncode}: {state.replay.lastOutput}"
            else:
                state.replay.message = f"{label} exited with code {replay_process.returncode}"

    recording_state = state.recording.state
    elapsed_s = None
    if state.process_started_at_s is not None:
        elapsed_s = max(0.0, time.monotonic() - state.process_started_at_s)
    recorded_datasets = _recorded_dataset_items(state)
    trajectory, trajectory_meta = _read_recorded_trajectory(state)
    if recorded_datasets and not trajectory_meta.get("datasetRoot"):
        latest_dataset = recorded_datasets[0]
        trajectory_meta = {
            **trajectory_meta,
            "datasetRoot": latest_dataset["path"],
            "sourcePath": latest_dataset["sourcePath"],
            "totalEpisodes": latest_dataset["totalEpisodes"],
            "recordedFrames": latest_dataset["totalFrames"],
        }
    state.replay.datasetRoot = str(trajectory_meta.get("datasetRoot") or state.replay.datasetRoot)
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
        "annotation": _active_annotation(state),
        "calibration": asdict(state.calibration),
        "recordedDatasets": recorded_datasets,
        "processing": _processing_items(state),
        "trajectory": trajectory,
        "events": [asdict(event) for event in state.events],
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


def _venv_python(repo_root: Path) -> Path:
    for name in (".venv", "venv"):
        candidate = repo_root / name / "bin" / "python"
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
    return state.repo_root / DEFAULT_RECORDER_SCRIPT, "--config_path"


def _ensure_recorder_running(state: GatewayState) -> subprocess.Popen[str]:
    process = state.process
    if process is None or process.poll() is not None:
        state.process = None
        state.recording.pid = None
        raise RuntimeError("Connect devices before starting an episode.")
    return process


def _write_recorder_stdin(process: subprocess.Popen[str], text: str) -> None:
    if process.stdin is None:
        raise RuntimeError("Handheld recorder stdin is unavailable.")
    process.stdin.write(text)
    process.stdin.flush()


def _connect_recorder(state: GatewayState) -> None:
    if state.process is not None and state.process.poll() is None:
        state.recording.message = "Devices are already connected"
        return

    state.recording.datasetRoot = str(_dataset_config(state.config).get("root") or "")
    recorder_script, config_flag = _recorder_script(state)
    command = [
        str(_venv_python(state.repo_root)),
        str(recorder_script),
        f"{config_flag}={state.config_path}",
    ]
    state.process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_recorder_env(state.repo_root),
        start_new_session=True,
    )
    state.process_started_at_s = time.monotonic()
    state.recording.state = "connecting"
    state.recording.pid = state.process.pid
    state.recording.frameIndex = 0
    state.recording.queueDepth = 0
    state.recording.message = "Connecting handheld devices"
    _set_all_device_states(state, "warning")
    state.log("info", f"Started handheld recorder pid={state.process.pid}")
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


def _read_process_output(state: GatewayState, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output = line.strip()
        if not output:
            continue
        with state.lock:
            if state.process is not process:
                return
            _apply_recorder_output(state, output)


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
                state.log("info", f"real replay: {output}")


def _set_mujoco_validation_metric(validation: dict[str, Any], key: str, value: str) -> None:
    try:
        validation[key] = float(value)
    except ValueError:
        return


def _apply_mujoco_replay_output(state: GatewayState, output: str) -> None:
    validation = state.replay.mujocoValidation or _new_mujoco_validation(state, status="running")
    result_match = re.search(
        r"mujoco_replay_result=status=(?P<status>\w+)\s+"
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

    poses = [frame.get("eePose") for frame in frames if isinstance(frame, dict) and frame.get("eePose")]
    pose_count = len([pose for pose in poses if isinstance(pose, dict) and {"x", "y", "z"}.issubset(pose.keys())])
    if pose_count != len(frames):
        failures.append(f"missing EE pose for {len(frames) - pose_count}/{len(frames)} frames")
    checks.append({"name": "ee_pose_present", "status": "pass" if pose_count == len(frames) and frames else "fail", "value": pose_count})

    max_step = 0.0
    previous_pose: dict[str, Any] | None = None
    z_values: list[float] = []
    gripper_values: list[float] = []
    for pose in poses:
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

    if reasons:
        validation["status"] = "failed"
        validation["message"] = "MuJoCo validation failed: " + "; ".join(reasons)
        state.replay.safety = "fault"
        state.replay.state = "aborted"
    else:
        validation["status"] = "passed"
        validation["message"] = (
            f"MuJoCo validation passed: max {float(max_pos):.2f}mm / {float(max_rot):.2f}deg "
            f"within {max_pos_threshold:.2f}mm / {max_rot_threshold:.2f}deg"
        )
        state.replay.safety = "ready"
        state.replay.state = "complete"
    state.replay.mujocoValidation = validation
    _refresh_mujoco_validation_current(state)
    if dataset_root is not None and _is_dataset_root(dataset_root):
        try:
            _write_validation_store(dataset_root, state.replay.mujocoValidation)
        except OSError as exc:
            state.log("warn", f"Failed to persist MuJoCo validation: {exc}")
    state.replay.message = validation["message"]


def _apply_recorder_output(state: GatewayState, output: str) -> None:
    state.recording.lastOutput = output
    state.recording.message = output
    state.log("info", f"recorder: {output}")

    failed_camera_match = re.search(r"Camera '([^']+)' failed to connect", output)
    if failed_camera_match:
        failed_id = failed_camera_match.group(1)
        for device in state.devices:
            if device.get("kind") == "camera" and device.get("id") == failed_id:
                device["state"] = "error"

    for prefix, kind in (
        ("Cameras:", "camera"),
        ("Tactiles:", "tactile"),
        ("Handheld grippers:", "handheld_gripper"),
        ("Box devices:", "box_collection"),
    ):
        if output.startswith(prefix):
            _mark_connected_devices(state, kind, output.removeprefix(prefix).strip())

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


def _mujoco_replay_command(state: GatewayState, dataset_root: Path) -> list[str]:
    command = [
        str(_venv_python(state.repo_root)),
        str(state.repo_root / "tools" / "fr3" / "fr3_sim_record_replay.py"),
        f"--dataset={_dataset_arg_for_container_replay(state.repo_root, dataset_root)}",
        f"--fps={state.replay.fps or 30}",
    ]
    if state.replay.episode >= 0:
        command.append(f"--episode={state.replay.episode}")
    return command


def _real_replay_command(state: GatewayState, dataset_root: Path) -> list[str]:
    replay = _replay_config(state.config)
    command = [
        str(_venv_python(state.repo_root)),
        str(state.repo_root / "tools" / "fr3" / "fr3_das_replay_real.py"),
        f"--dataset={_dataset_arg_for_container_replay(state.repo_root, dataset_root)}",
        f"--episode={state.replay.episode}",
        f"--fps={state.replay.fps or 30}",
    ]
    option_map = {
        "timing_source": "--timing-source",
        "robot_ip": "--robot-ip",
        "filter_coeff": "--filter-coeff",
        "damping": "--damping",
        "stiffness": "--stiffness",
        "otg_max_velocity": "--otg-max-velocity",
        "otg_max_acceleration": "--otg-max-acceleration",
        "otg_max_jerk": "--otg-max-jerk",
        "otg_velocity_scale": "--otg-velocity-scale",
        "otg_acceleration_scale": "--otg-acceleration-scale",
        "otg_jerk_scale": "--otg-jerk-scale",
        "gripper_port": "--gripper-port",
        "gripper_backend": "--gripper-backend",
        "reset_gripper_position": "--reset-gripper-position",
        "reset_gripper_timeout_s": "--reset-gripper-timeout-s",
        "analysis_output_dir": "--analysis-output-dir",
        "compose_file": "--compose-file",
        "service": "--service",
    }
    for config_key, cli_key in option_map.items():
        if replay.get(config_key) is not None:
            command.append(f"{cli_key}={replay[config_key]}")
    if replay.get("disable_otg"):
        command.append("--disable-otg")
    return command


def _real_robot_ip(state: GatewayState) -> str:
    replay = _replay_config(state.config)
    robot = state.config.get("robot") if isinstance(state.config.get("robot"), dict) else {}
    return str(replay.get("robot_ip") or robot.get("robot_ip") or DEFAULT_REAL_ROBOT_IP)


def _real_preflight_command(state: GatewayState) -> list[str]:
    replay = _replay_config(state.config)
    command = [
        str(_venv_python(state.repo_root)),
        str(state.repo_root / "tools" / "fr3" / "fr3_record_preflight.py"),
        f"--workspace={state.repo_root}",
        f"--config-path={state.config_path}",
        f"--robot-ip={_real_robot_ip(state)}",
    ]
    if _bool_config(replay, "real_preflight_skip_hikrobot", True):
        command.append("--skip-hikrobot")
    if _bool_config(replay, "real_preflight_skip_gripper", True):
        command.append("--skip-gripper")
    if _bool_config(replay, "real_preflight_skip_arm", False):
        command.append("--skip-arm")
    if _bool_config(replay, "real_preflight_skip_ping", False):
        command.append("--skip-ping")
    return command


def _run_real_preflight(state: GatewayState) -> None:
    replay = _replay_config(state.config)
    if not _bool_config(replay, "real_preflight_enabled", True):
        state.log("warn", "Real-robot preflight skipped by replay.real_preflight_enabled=false")
        return
    timeout_s = _float_config(replay, "real_preflight_timeout_s", DEFAULT_REAL_PREFLIGHT_TIMEOUT_S)
    command = _real_preflight_command(state)
    state.replay.message = "Running real-robot preflight checks"
    try:
        result = subprocess.run(
            command,
            cwd=state.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_tool_env(state.repo_root),
            timeout=max(timeout_s, 1.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Real-robot preflight timed out after {timeout_s:.1f}s") from exc
    output_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if output_lines:
        state.replay.lastOutput = output_lines[-1]
    if result.returncode != 0:
        details = output_lines[-1] if output_lines else f"exit code {result.returncode}"
        raise RuntimeError(f"Real-robot preflight failed: {details}")


def _require_mujoco_validation(state: GatewayState) -> Path:
    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")
    if not _mujoco_validation_is_for_active_episode(state, dataset_root):
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
    return "MuJoCo replay is strongly recommended before Preflight/Dry Run and still required before Real Robot"


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


def _start_mujoco_replay(state: GatewayState) -> None:
    if state.replay_process is not None and state.replay_process.poll() is None:
        state.replay.message = "MuJoCo replay is already running"
        return

    dataset_root = _active_replay_dataset_root(state)
    if not _is_dataset_root(dataset_root):
        raise RuntimeError(f"Selected replay dataset is not finalized: {dataset_root}")

    command = _mujoco_replay_command(state, dataset_root)
    state.replay.mujocoValidation = _new_mujoco_validation(
        state,
        status="running",
        dataset_root=dataset_root,
        episode=state.replay.episode,
        message="MuJoCo replay is running; real-robot replay remains locked until metrics pass.",
    )
    process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_tool_env(state.repo_root),
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
    state.replay.message = f"MuJoCo replay started for {dataset_root.name}; waiting for validation metrics"
    state.log("info", f"Started MuJoCo replay pid={process.pid} dataset={dataset_root}")
    _start_replay_output_reader(state, process)


def _start_real_replay(state: GatewayState) -> None:
    if state.replay_process is not None and state.replay_process.poll() is None:
        state.replay.message = "Replay process is already running"
        return

    dataset_root = _require_mujoco_validation(state)
    command = _real_replay_command(state, dataset_root)
    process = subprocess.Popen(
        command,
        cwd=state.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_tool_env(state.repo_root),
        start_new_session=True,
    )
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
    state.replay.message = f"Real robot replay started for episode {state.replay.episode} from {dataset_root.name}"
    state.log("warn", f"Started real robot replay pid={process.pid} dataset={dataset_root} episode={state.replay.episode}")
    _start_replay_output_reader(state, process)


def _start_dry_run_replay(state: GatewayState) -> None:
    dataset_root = _require_replay_dataset(state)
    state.replay.state = "dry_run"
    state.replay.safety = "ready"
    state.replay.frameIndex = 0
    state.replay.message = (
        f"Dry-run started for episode {state.replay.episode} from {dataset_root.name}; "
        f"{_mujoco_recommendation_suffix(state, dataset_root)}"
    )
    state.log("info", state.replay.message)


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
            _write_recorder_stdin(process, "s\n" if state.recording.state == "recording" else "y\n")
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
        if path == "/api/replay/video":
            requested = query.get("path", [""])[0]
            camera_key = query.get("key", [""])[0]
            if not camera_key:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing camera key"})
                return
            with self.server.state.lock:
                dataset_root = _resolve_known_dataset(self.server.state, requested)
                video_path = _resolve_video_path(self.server.state, dataset_root, camera_key) if dataset_root else None
            if video_path is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"video not found: {camera_key}"})
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
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        try:
            with self.server.state.lock:
                if path == "/api/handheld/record/connect":
                    _connect_recorder(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
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
                if path == "/api/replay/start":
                    _start_dry_run_replay(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/start-mujoco":
                    _start_mujoco_replay(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/replay/start-real":
                    _start_real_replay(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/calibration/run":
                    _mock_calibrate_cameras(self.server.state)
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/processing/qc":
                    requested = (query.get("path", [""])[0] or "").strip()
                    dataset_root = _resolve_known_dataset(self.server.state, requested)
                    if dataset_root is None:
                        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                        return
                    qc_result = _run_qc(dataset_root)
                    try:
                        _write_processing_meta_qc(dataset_root, qc_result)
                    except OSError as exc:
                        _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"failed to persist QC: {exc}"})
                        return
                    self.server.state.log(
                        "info" if qc_result["status"] == "pass" else "warn",
                        f"QC {qc_result['status']} for {dataset_root.name}: {qc_result['summary']}",
                    )
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
                    return
                if path == "/api/processing/traj-gen":
                    requested = (query.get("path", [""])[0] or "").strip()
                    dataset_root = _resolve_known_dataset(self.server.state, requested)
                    if dataset_root is None:
                        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "dataset not in candidate list"})
                        return
                    try:
                        _queue_traj_gen(self.server.state, dataset_root)
                    except NotImplementedError as exc:
                        _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {"error": str(exc)})
                        return
                    _json_response(self, HTTPStatus.OK, _snapshot(self.server.state))
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
        except Exception as exc:  # noqa: BLE001
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[data-collection-gateway] {format % args}\n")


class DataCollectionGuiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: GatewayState):
        super().__init__(server_address, DataCollectionGuiHandler)
        self.state = state


def make_state(repo_root: Path, config_path: Path, datasets_root: Path | None = None) -> GatewayState:
    resolved_root = repo_root.resolve()
    resolved_config = config_path if config_path.is_absolute() else resolved_root / config_path
    config = _load_yaml(resolved_config)
    resolved_datasets_root: Path | None = None
    if datasets_root is not None:
        resolved_datasets_root = datasets_root if datasets_root.is_absolute() else resolved_root / datasets_root
        resolved_datasets_root = resolved_datasets_root.resolve()
    state = GatewayState(
        repo_root=resolved_root,
        config_path=resolved_config,
        config=config,
        recording=_recording_status_from_config(config),
        replay=_replay_status_from_config(config),
        datasets_root=resolved_datasets_root,
        devices=_device_statuses(config),
    )
    state.replay.mujocoValidation = _new_mujoco_validation(state)
    state.log("info", f"Loaded handheld config {resolved_config}")
    if resolved_datasets_root is not None:
        state.log("info", f"Scanning datasets under {resolved_datasets_root}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local HTTP gateway for the LeRobot data collection GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = make_state(args.repo_root, args.config_path, args.datasets_root)
    server = DataCollectionGuiServer((args.host, args.port), state)
    print(f"Data collection GUI gateway listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        with state.lock:
            if state.process is not None and state.process.poll() is None:
                os.killpg(state.process.pid, signal.SIGTERM)
        server.server_close()


if __name__ == "__main__":
    main()
