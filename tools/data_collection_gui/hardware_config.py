"""Persist portable device setups and compile them into existing LeRobot runtime YAMLs.

The source YAML is never overwritten. Addresses live in instance parameters; mounting
geometry is a separate object and is retained when only an address changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import draccus
import yaml

from lerobot.hardware import get_model
from lerobot.hardware.catalog import model_fields, validate_parameters

SETUP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEVICE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MOUNTING_FIELDS = {
    "urdf_path",
    "target_frame_name",
    "start_joint_positions",
    "workspace_min",
    "workspace_max",
}


class HardwareConfigError(ValueError):
    pass


def store_root(repo_root: Path, source: Path) -> Path:
    namespace = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    return repo_root / "outputs" / "hardware" / namespace


def _id(value: Any) -> str:
    if not isinstance(value, str) or not SETUP_ID.fullmatch(value):
        raise HardwareConfigError("Configuration id must use 1–64 letters, digits, '_' or '-'.")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_setup(root: Path, setup_id: str) -> dict[str, Any]:
    path = root / f"{_id(setup_id)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HardwareConfigError(f"Could not read configuration {setup_id}: {exc}") from exc


def active_setup(root: Path) -> dict[str, Any] | None:
    pointer = root / "active.json"
    if not pointer.exists():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8"))
    except (KeyError, ValueError, OSError) as exc:
        raise HardwareConfigError(f"Saved hardware configuration is invalid: {exc}") from exc


def list_setups(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.glob("*.json")):
        if path.name == "active.json":
            continue
        try:
            item = read_setup(root, path.stem)
            result.append(
                {
                    "id": item["document"]["id"],
                    "label": item["document"]["label"],
                    "revision": item["revision"],
                }
            )
        except (ValueError, KeyError):
            continue
    return result


def _instance(kind: str, config: dict[str, Any]) -> dict[str, Any] | None:
    name = config.get("type")
    if not name:
        return None
    try:
        specs = model_fields(get_model(kind, str(name)))
        allowed = {f["name"] for f in specs}
        params = {f["name"]: copy.deepcopy(f["default"]) for f in specs if f["default"] is not None}
        params.update({k: copy.deepcopy(v) for k, v in config.items() if k in allowed})
    except ValueError:
        params = {k: copy.deepcopy(v) for k, v in config.items() if k not in {"type", "cameras"}}
    return {"type": name, "params": params}


def _read_field(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return copy.deepcopy(current)


def _write_field(config: dict[str, Any], path: str, value: Any) -> None:
    segments = path.split(".")
    target = config
    for segment in segments[:-1]:
        target = target.setdefault(segment, {})
    target[segments[-1]] = value


def from_runtime(config: dict[str, Any]) -> dict[str, Any]:
    robot = config.get("robot") or {}
    sensors = config.get("sensors") or {}
    cameras = robot.get("cameras", {}) if robot else sensors.get("cameras", {})
    specialized = isinstance(cameras, dict) and "defaults" in cameras
    end_effector = None
    backend = robot.get("gripper_backend", "pika" if robot.get("type") == "franka_research3" else "")
    if backend:
        try:
            model = get_model("end_effector", backend)
            defaults = {
                f["name"]: copy.deepcopy(f["default"])
                for f in model_fields(model)
                if f["default"] is not None
            }
            defaults.update(
                {
                    name: _read_field(robot, key)
                    for name, key in model.robot_fields.items()
                    if _read_field(robot, key) is not None
                }
            )
            end_effector = {"type": model.type, "params": defaults}
        except ValueError:
            end_effector = {"type": backend, "params": {}}
    return {
        "id": "my-setup",
        "label": "My hardware setup",
        "version": 1,
        "robot": _instance("robot", robot),
        "endEffector": end_effector,
        "teleop": _instance("teleoperator", config.get("teleop") or {}),
        "mounting": {k: copy.deepcopy(v) for k, v in robot.items() if k in MOUNTING_FIELDS},
        "cameras": []
        if specialized
        else [
            {"name": name, **item}
            for name, cfg in cameras.items()
            if isinstance(cfg, dict) and (item := _instance("camera", cfg))
        ],
        "specializedCameras": specialized,
    }


def _validate_instance(kind: str, instance: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(instance, dict) or not isinstance(instance.get("type"), str):
        raise HardwareConfigError(f"Select a {kind} model.")
    model = get_model(kind, instance["type"])
    values = validate_parameters(model, instance.get("params", {}))
    return model, values


def _decode(model: Any, values: dict[str, Any]) -> None:
    cls = model.config_type()
    if cls:
        draccus.decode(cls, values)


def _path_in_repo(value: str, repo_root: Path) -> Path:
    for prefix in ("/workspace/", "/lerobot/"):
        if value.startswith(prefix):
            return repo_root / value[len(prefix) :]
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def compile_setup(document: Any, base: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(document, dict) or document.get("version") != 1:
        raise HardwareConfigError("Hardware configuration must be a version 1 object.")
    _id(document.get("id"))
    if not isinstance(document.get("label"), str) or not document["label"].strip():
        raise HardwareConfigError("Configuration name is required.")
    if len(document["label"]) > 120:
        raise HardwareConfigError("Configuration name must be at most 120 characters.")
    # Also rejects NaN/Infinity before they can enter numeric validation or persisted JSON.
    json.dumps(document, allow_nan=False)
    result = copy.deepcopy(base)
    warnings: list[str] = []
    cameras: dict[str, Any] = {}
    identities: set[tuple[str, str]] = set()
    camera_entries = document.get("cameras", [])
    if not isinstance(camera_entries, list):
        raise HardwareConfigError("Cameras must be a list.")
    for entry in camera_entries:
        if not isinstance(entry, dict):
            raise HardwareConfigError("Each camera must be an object.")
        name = entry.get("name", "")
        if not isinstance(name, str) or not DEVICE_NAME.fullmatch(name) or name in cameras:
            raise HardwareConfigError("Camera role names must be unique and use letters, digits, '_' or '-'.")
        model, values = _validate_instance("camera", entry)
        for key in ("width", "height", "fps"):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0:
                raise HardwareConfigError(f"Camera {name}: {key} must be positive.")
        _decode(model, values)
        identity = next(
            (
                values.get(k)
                for k in (
                    "serial_number_or_name",
                    "serial",
                    "device_index",
                    "index_or_path",
                    "sensor_id",
                    "device",
                    "ip",
                    "host",
                    "url",
                    "endpoint",
                )
                if values.get(k) is not None and values.get(k) != ""
            ),
            None,
        )
        if model.identity_fields:
            identity = json.dumps([(key, values.get(key)) for key in model.identity_fields], sort_keys=True)
        if identity is None:
            raise HardwareConfigError(f"Camera {name}: specify its serial, index, port or sensor id.")
        binding = (model.type, str(identity))
        if binding in identities:
            raise HardwareConfigError(
                f"Camera {name}: this physical device is already bound to another camera role."
            )
        identities.add(binding)
        cameras[name] = {"type": model.type, **values}

    if document.get("robot") is None:
        if document.get("endEffector"):
            raise HardwareConfigError("An end effector must be attached to a robot.")
        if base.get("robot"):
            raise HardwareConfigError("Select a robot for this recording workflow.")
        if document.get("specializedCameras"):
            if cameras:
                raise HardwareConfigError(
                    "Select a robot before replacing the specialized Thor camera pipeline."
                )
            warnings.append("Existing Thor camera discovery and BOX configuration are preserved.")
        else:
            result.setdefault("sensors", {})["cameras"] = cameras
        return result, warnings

    model, robot = _validate_instance("robot", document["robot"])
    if document.get("specializedCameras"):
        raise HardwareConfigError("Choose camera instances for the selected robot.")
    robot["type"] = model.type
    robot["cameras"] = cameras
    tool_instance = document.get("endEffector")
    if tool_instance:
        tool, params = _validate_instance("end_effector", tool_instance)
        if model.type not in tool.compatible_robots:
            raise HardwareConfigError(f"{tool.label} is not registered for {model.label}.")
        if tool.end_effector_factory:
            _decode(tool, params)
        for name, target in tool.robot_fields.items():
            if name in params:
                _write_field(robot, target, params[name])
        robot["gripper_backend"] = tool.type
        robot["allow_mock_gripper"] = False
        mounting = {**tool.mounting, **document.get("mounting", {})}
    else:
        if model.type == "franka_research3":
            raise HardwareConfigError("Select an end effector for Franka Research 3.")
        mounting = document.get("mounting", {})
    if not isinstance(mounting, dict) or set(mounting) - MOUNTING_FIELDS:
        raise HardwareConfigError("Unknown mounting fields.")
    robot.update(mounting)
    if model.type == "franka_research3":
        if not str(robot.get("robot_ip", "")).strip():
            raise HardwareConfigError("Robot IP is required.")
        urdf = str(robot.get("urdf_path", ""))
        path = _path_in_repo(urdf, repo_root)
        if not urdf or not path.is_file():
            raise HardwareConfigError(f"Robot URDF does not exist on the gateway: {urdf}")
        import xml.etree.ElementTree as ET

        links = {node.get("name") for node in ET.parse(path).getroot().findall("link")}
        if robot.get("target_frame_name") not in links:
            raise HardwareConfigError("Selected TCP frame is not a link in the selected URDF.")
        robot["urdf_path"] = str(path.resolve())
        warnings.append(
            "Mounting presets describe standard hardware. Check TCP, workspace and home after changing installation geometry."
        )
        if robot["gripper_backend"] == "franka_hand":
            warnings.append(
                "Franka Hand currently uses binary open/grasp commands; simulation for this assembly is unavailable."
            )
    if model.assemble:
        robot = model.assemble(copy.deepcopy(robot))
        if not isinstance(robot, dict) or robot.get("type") != model.type:
            raise HardwareConfigError("Hardware assembly must return the selected robot type and its config.")
    # Run the driver's dataclass validation, but never instantiate/connect a hardware driver.
    _decode(model, {k: v for k, v in robot.items() if k != "type"})
    teleop_model, teleop = _validate_instance("teleoperator", document.get("teleop"))
    _decode(teleop_model, teleop)
    if (
        tool_instance
        and "binary_gripper" in tool.capabilities
        and teleop_model.type == "spacemouse"
        and teleop.get("tool_mode") != "binary"
    ):
        raise HardwareConfigError("This end effector requires SpaceMouse tool_mode=binary.")
    if model.type == "franka_research3" and teleop_model.type not in {"spacemouse", "quest3", "nintendo"}:
        raise HardwareConfigError("FR3 requires a Cartesian teleoperator (SpaceMouse, Quest 3 or Nintendo).")
    if (
        model.type.startswith("so")
        and model.type.endswith("_follower")
        and teleop_model.type != model.type.replace("follower", "leader")
    ):
        raise HardwareConfigError("Select the SO leader matching this robot model.")
    result["robot"] = robot
    result["teleop"] = {"type": teleop_model.type, **teleop}
    # Specialized capture blocks are not members of LeRobot's strict RecordConfig.
    for key in ("sensors", "recorder", "box_collection", "hardware_sync", "replay"):
        result.pop(key, None)
    allowed = {
        "robot",
        "teleop",
        "dataset",
        "policy",
        "control_fps",
        "auto_move_to_start_after_episode",
        "move_to_start_after_last_episode",
        "confirm_next_episode_after_reset",
        "display_data",
        "display_compressed_images",
        "play_sounds",
        "resume",
    }
    result = {k: v for k, v in result.items() if k in allowed}
    result.pop("policy", None)
    result.setdefault("dataset", {}).setdefault("single_task", "Record demonstrations")
    result["dataset"].setdefault("repo_id", "local/demonstrations")
    result["dataset"].setdefault("root", str(repo_root / "outputs/datasets/demonstrations"))
    result["dataset"].setdefault("fps", 30)
    if model.type != "franka_research3":
        result["auto_move_to_start_after_episode"] = False
        result["move_to_start_after_last_episode"] = False
        result["confirm_next_episode_after_reset"] = False
        warnings.append(
            "Native-schema recording is supported. FR3-specific replay, training-view conversion and rollout are unavailable for this robot."
        )
    return result, warnings


def save_setup(
    root: Path,
    document: dict[str, Any],
    runtime: dict[str, Any],
    warnings: list[str],
    expected_revision: str | None,
) -> dict[str, Any]:
    document = copy.deepcopy(document)
    for key, kind in (("robot", "robot"), ("endEffector", "end_effector"), ("teleop", "teleoperator")):
        if document.get(key):
            model, values = _validate_instance(kind, document[key])
            document[key]["params"] = values
            if kind == "end_effector":
                document["mounting"] = {**model.mounting, **document.get("mounting", {})}
    for camera in document["cameras"]:
        _, camera["params"] = _validate_instance("camera", camera)
    setup_id = _id(document["id"])
    path = root / f"{setup_id}.json"
    if path.exists() and read_setup(root, setup_id)["revision"] != expected_revision:
        raise HardwareConfigError("This configuration changed in another browser. Reload it before saving.")
    body = {"document": document, "runtime": runtime, "warnings": warnings}
    revision = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    saved = {**body, "revision": revision}
    _atomic_write(path, json.dumps(saved, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return saved


def activate_setup(root: Path, saved: dict[str, Any]) -> Path:
    setup_id = _id(saved["document"]["id"])
    runtime_path = root / f"{setup_id}.yaml"
    _atomic_write(runtime_path, yaml.safe_dump(saved["runtime"], sort_keys=False, allow_unicode=True))
    # Persist the applied revision, independent of subsequently saved drafts.
    _atomic_write(root / "active.json", json.dumps(saved, ensure_ascii=False, allow_nan=False))
    return runtime_path
