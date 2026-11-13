"""Extensible, SDK-free hardware descriptions. Driver config classes remain authoritative."""

from __future__ import annotations

import copy
import importlib
import types
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Union, get_args, get_origin, get_type_hints

from .interfaces import EndEffectorDriver


@dataclass(frozen=True)
class HardwareModel:
    kind: str
    type: str
    label: str
    # A class or module:Class reference; resolving it imports configuration, never a driver instance.
    config_class: type | str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    primary: tuple[str, ...] = ()
    compatible_robots: tuple[str, ...] = ()
    # End-effector instance parameter -> existing RobotConfig field.
    robot_fields: dict[str, str] = field(default_factory=dict)
    mounting: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    description: str = ""
    # Plugins can declare additional validation without teaching the GUI about the device.
    validate: Callable[[dict[str, Any]], None] | None = None
    recording_processors: Callable[..., Any] | None = None
    end_effector_factory: Callable[[Any], EndEffectorDriver] | None = None
    identity_fields: tuple[str, ...] = ()
    # Convert the composed robot/tool/camera dictionary to a driver's native config shape.
    assemble: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def config_type(self) -> type | None:
        if isinstance(self.config_class, str):
            module, name = self.config_class.split(":", 1)
            return getattr(importlib.import_module(module), name)
        return self.config_class


_MODELS: dict[tuple[str, str], HardwareModel] = {}
_BUILTINS_LOADED = False
_PLUGINS_LOADED = False
_CATALOG_LOCK = RLock()


def _locked(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _CATALOG_LOCK:
            return function(*args, **kwargs)

    return wrapped


@_locked
def register_model(model: HardwareModel) -> None:
    """Register a form/runtime adapter from an installed lerobot_robot_* plugin's __init__.py."""
    if model.kind not in {"robot", "end_effector", "camera", "teleoperator"}:
        raise ValueError(f"Unknown hardware kind {model.kind!r}")
    key = (model.kind, model.type)
    if key in _MODELS:
        raise ValueError(f"Hardware model already registered: {key}")
    _MODELS[key] = model


def _builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    fr3 = "lerobot.robots.franka_research3.config_franka_research3:FrankaResearch3Config"
    tool_fields = (
        "gripper_port",
        "gripper_backend",
        "allow_mock_gripper",
        "gripper_max_width_mm",
        "gripper_command_rate_limit_hz",
        "gripper_command_deadband_mm",
        "gripper_config",
    )
    register_model(
        HardwareModel(
            "robot",
            "franka_research3",
            "Franka Research 3",
            fr3,
            defaults={"robot_ip": "", "allow_mock_gripper": False},
            exclude=(*tool_fields, "cameras", "urdf_path", "target_frame_name", "start_joint_positions"),
            primary=("robot_ip", "id", "realtime_config"),
            capabilities=("record", "teleoperate", "cartesian", "joint_position"),
        )
    )
    for name, label in (("so100_follower", "SO-100"), ("so101_follower", "SO-101")):
        register_model(
            HardwareModel(
                "robot",
                name,
                label,
                "lerobot.robots.so_follower.config_so_follower:"
                + ("SO101FollowerConfig" if name == "so101_follower" else "SO100FollowerConfig"),
                exclude=("cameras",),
                primary=("port", "id", "calibration_dir"),
                capabilities=("record", "teleoperate", "joint_position"),
                description="Integrated gripper; use the matching SO leader teleoperator.",
            )
        )
    asset = "src/lerobot/robots/franka_research3/assets/franka_fr3/"
    for name, label, prefix, mapping, mount in (
        (
            "pika",
            "Pika",
            "",
            {"port": "gripper_port", "max_width_mm": "gripper_max_width_mm"},
            {"urdf_path": asset + "fr3_pika_gripper.urdf", "target_frame_name": "pika_gripper_ee"},
        ),
        (
            "franka_hand",
            "Franka Hand",
            "",
            {},
            {"urdf_path": asset + "fr3_franka_hand.urdf", "target_frame_name": "fr3_hand_tcp"},
        ),
        (
            "das",
            "DAS",
            "das_",
            {"port": "gripper_port"},
            {"urdf_path": asset + "fr3_das_ati.urdf", "target_frame_name": "das_gripper_ee"},
        ),
        (
            "corenetic",
            "Corenetic",
            "corenetic_",
            {"max_width_mm": "gripper_max_width_mm"},
            {"urdf_path": asset + "fr3_corenetic_gripper.urdf", "target_frame_name": "corenetic_gripper_ee"},
        ),
    ):
        common = {
            "command_rate_limit_hz": "gripper_command_rate_limit_hz",
            "command_deadband_mm": "gripper_command_deadband_mm",
        }
        if prefix:
            cls = HardwareModel("robot", "_", "_", fr3).config_type()
            common.update({f.name: f.name for f in fields(cls) if f.name.startswith(prefix)})
        common.update(mapping)
        register_model(
            HardwareModel(
                "end_effector",
                name,
                label,
                fr3,
                robot_fields=common,
                compatible_robots=("franka_research3",),
                mounting=mount,
                primary=tuple(mapping) or ("command_rate_limit_hz",),
                capabilities=("binary_gripper",) if name == "franka_hand" else ("position_gripper",),
                description="Uses the selected arm's IP." if name == "franka_hand" else "",
            )
        )
    for name, label, module, cls, primary, defaults in (
        (
            "intelrealsense",
            "Intel RealSense",
            "realsense.configuration_realsense",
            "RealSenseCameraConfig",
            ("serial_number_or_name", "width", "height", "fps"),
            {},
        ),
        (
            "hikrobot",
            "Hikrobot",
            "hikrobot.configuration_hikrobot",
            "HikrobotCameraConfig",
            ("serial", "transport_layer", "width", "height", "fps"),
            {"transport_layer": "gige"},
        ),
        (
            "opencv",
            "OpenCV / USB camera",
            "opencv.configuration_opencv",
            "OpenCVCameraConfig",
            ("index_or_path", "width", "height", "fps"),
            {"index_or_path": 0},
        ),
        (
            "gmsl2",
            "GMSL2 / Jetson",
            "gmsl2.configuration_gmsl2",
            "Gmsl2CameraConfig",
            ("sensor_id", "device", "pipeline", "width", "height", "fps"),
            {},
        ),
    ):
        register_model(
            HardwareModel(
                "camera",
                name,
                label,
                f"lerobot.cameras.{module}:{cls}",
                defaults={"width": 640, "height": 480, "fps": 30, **defaults},
                primary=primary,
            )
        )
    for name, label, config in (
        ("spacemouse", "SpaceMouse", "spacemouse.configuration_spacemouse:SpaceMouseTeleopConfig"),
        ("so100_leader", "SO-100 leader", "so_leader.config_so_leader:SO100LeaderConfig"),
        ("so101_leader", "SO-101 leader", "so_leader.config_so_leader:SO101LeaderConfig"),
        ("quest3", "Quest 3", "quest3.configuration_quest3:Quest3TeleopConfig"),
        ("nintendo", "Nintendo", "nintendo.configuration_nintendo:NintendoTeleopConfig"),
    ):
        register_model(
            HardwareModel(
                "teleoperator",
                name,
                label,
                f"lerobot.teleoperators.{config}",
                primary=("port", "device_id", "id", "tool_mode"),
            )
        )


@_locked
def get_model(kind: str, name: str) -> HardwareModel:
    _builtins()
    try:
        return _MODELS[kind, name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {kind} model {name!r}; install/register its hardware adapter first."
        ) from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def model_fields(model: HardwareModel) -> list[dict[str, Any]]:
    cls = model.config_type()
    if cls is None:
        return []
    hints = get_type_hints(cls)
    by_name = {f.name: f for f in fields(cls)}
    mapping = model.robot_fields or {name: name for name in by_name}
    result = []
    for name, config_name in mapping.items():
        if name in model.exclude or (model.include and name not in model.include):
            continue
        if (
            model.kind == "robot"
            and model.type == "franka_research3"
            and (name.startswith(("das_", "corenetic_")) or name == "gen_con_sdk_path")
        ):
            continue
        f = by_name[config_name] if config_name in by_name else by_name[name]
        hint = hints.get(f.name, str)
        args = get_args(hint)
        nullable = type(None) in args
        candidates = [t for t in args if t is not type(None)] if nullable else [hint]
        concrete = candidates[0] if len(candidates) == 1 else hint
        default = f.default
        if default is MISSING and f.default_factory is not MISSING:
            default = f.default_factory()
        options = (
            [_plain(item) for item in concrete]
            if isinstance(concrete, type) and issubclass(concrete, Enum)
            else None
        )
        if options:
            kind = "select"
        elif concrete is bool:
            kind = "boolean"
        elif concrete is int:
            kind = "integer"
        elif concrete is float:
            kind = "number"
        elif get_origin(concrete) in (types.UnionType, Union) and set(get_args(concrete)) <= {str, int, Path}:
            kind = "device"
        elif concrete in (str, Path):
            kind = "string"
        else:
            kind = "json"
        result.append(
            {
                "name": name,
                "label": name.replace("_", " "),
                "kind": kind,
                "required": default is MISSING and name not in model.defaults,
                "nullable": nullable,
                "advanced": name not in model.primary,
                "default": _plain(model.defaults.get(name, None if default is MISSING else default)),
                "options": options,
            }
        )
    return result


def validate_parameters(model: HardwareModel, params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError(f"{model.label}: parameters must be an object")
    specs = {f["name"]: f for f in model_fields(model)}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"{model.label}: unknown parameters {sorted(unknown)}")
    values = {name: copy.deepcopy(f["default"]) for name, f in specs.items() if f["default"] is not None}
    values.update(copy.deepcopy(params))
    for name, spec in specs.items():
        value = values.get(name)
        if spec["required"] and (value is None or value == ""):
            raise ValueError(f"{model.label}: {name} is required")
        if value is None:
            if name in values and not spec["nullable"]:
                raise ValueError(f"{model.label}: {name} cannot be null")
            continue
        kind = spec["kind"]
        if kind == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{model.label}: {name} must be a boolean")
        if kind in {"integer", "number"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{model.label}: {name} must be a number")
        if kind == "integer" and int(value) != value:
            raise ValueError(f"{model.label}: {name} must be an integer")
        if kind == "select" and value not in spec["options"]:
            raise ValueError(f"{model.label}: invalid {name}: {value!r}")
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{model.label}: {name} must be a string")
        if kind == "device" and (isinstance(value, bool) or not isinstance(value, (str, int))):
            raise ValueError(f"{model.label}: {name} must be a string or device index")
    if model.validate:
        model.validate(values)
    return values


@_locked
def catalog() -> list[dict[str, Any]]:
    global _PLUGINS_LOADED
    _builtins()
    if not _PLUGINS_LOADED:
        _PLUGINS_LOADED = True
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    result = []
    for model in _MODELS.values():
        try:
            specs = model_fields(model)
            error = ""
        except (ImportError, AttributeError, TypeError) as exc:
            specs, error = [], str(exc)
        result.append(
            {
                "kind": model.kind,
                "type": model.type,
                "label": model.label,
                "fields": specs,
                "compatibleRobots": list(model.compatible_robots),
                "mounting": model.mounting,
                "capabilities": list(model.capabilities),
                "description": model.description,
                "available": not error,
                "error": error,
            }
        )
    return result
