#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
import logging
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Protocol

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class ArmDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions(self) -> np.ndarray: ...

    def get_ee_pose(self) -> np.ndarray | None: ...

    def set_joint_positions(self, joint_positions: np.ndarray) -> None: ...

    def move_to_start(self) -> None: ...


class GripperDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_position(self) -> float: ...

    def set_position(self, normalized_position: float) -> None: ...


class KinematicsDriver(Protocol):
    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray: ...


class JointOTGDriver(Protocol):
    def reset(self, current_joint_positions: np.ndarray) -> None: ...

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray: ...


def _silence_pika_logs() -> None:
    logging.getLogger("pika.gripper").setLevel(logging.WARNING)
    logging.getLogger("pika.serial_comm").setLevel(logging.WARNING)


_DAS_SDK_ENV_VAR = "GEN_CON_SDK_HOME"
_DEFAULT_DAS_SDK_ROOTS = (
    Path("/opt/dependencies/gen_con_sdk_python_release"),
    Path("/opt/gen_con_sdk_python_release"),
)
_DAS_SDK_MODULE_CANDIDATES = (
    "gen_controller_sdk_python",
    "gen_con_sdk_python_release",
    "gen_con_sdk_python_release.gen_controller_sdk_python",
    "dependencies.gen_con_sdk_python_release.gen_controller_sdk_python",
)


def _resolve_das_databus_cls(gen_con_sdk_path: str | None):
    search_roots: list[Path] = []
    for raw_path in (gen_con_sdk_path, os.environ.get(_DAS_SDK_ENV_VAR)):
        if raw_path:
            path = Path(raw_path).expanduser()
            search_roots.extend([path, path.parent])
    for root in _DEFAULT_DAS_SDK_ROOTS:
        search_roots.extend([root, root.parent])

    seen_paths: set[str] = set()
    for root in search_roots:
        root_str = str(root)
        if not root_str or root_str in seen_paths or not root.exists():
            continue
        seen_paths.add(root_str)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    import_errors: list[str] = []
    for module_name in _DAS_SDK_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        databus_cls = getattr(module, "DataBus", None)
        if databus_cls is not None:
            return databus_cls

    raise ImportError(
        "franka_research3 DAS gripper backend requires the gen_controller_sdk_python DataBus. "
        f"Set {_DAS_SDK_ENV_VAR} to the cloned SDK root if it is not installed in the default image. "
        f"Import attempts: {import_errors}"
    )


@dataclass
class PandaPyArmDriver:
    robot_ip: str
    damping: list[float] | None = None
    stiffness: list[float] | None = None
    filter_coeff: float | None = None

    def __post_init__(self):
        try:
            from panda_py import Panda, controllers
        except Exception as e:  # pragma: no cover - exercised with real hardware only
            raise ImportError(
                "franka_research3 requires panda_py for hardware arm control. "
                "Install panda_py in the runtime environment to use the FR3 hardware backend."
            ) from e

        self._panda_cls = Panda
        self._controllers = controllers
        self._robot = None
        self._controller = None

    def connect(self) -> None:
        self._robot = self._panda_cls(self.robot_ip)
        self._start_controller()

    def _start_controller(self) -> None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        self._controller = self._controllers.JointPosition()
        if self.damping is not None:
            self._controller.set_damping(self.damping)
        if self.stiffness is not None:
            self._controller.set_stiffness(self.stiffness)
        if self.filter_coeff is not None:
            self._controller.set_filter(self.filter_coeff)
        current_joint_positions = np.asarray(self._robot.get_state().q, dtype=np.float64)
        self._controller.set_control(current_joint_positions)
        self._robot.start_controller(self._controller)

    def _stop_controller(self) -> None:
        if self._robot is not None and self._controller is not None:
            self._robot.stop_controller()
            self._controller = None

    def disconnect(self) -> None:
        if self._robot is not None:
            self._stop_controller()
            self._robot = None

    def get_joint_positions(self) -> np.ndarray:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        return np.asarray(self._robot.get_state().q, dtype=np.float64)

    def get_ee_pose(self) -> np.ndarray | None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        pose = np.asarray(self._robot.get_pose(), dtype=np.float64)
        if pose.shape == (4, 4):
            return pose
        if pose.size == 16:
            return pose.reshape(4, 4)
        return None

    def set_joint_positions(self, joint_positions: np.ndarray) -> None:
        if self._controller is None:
            raise RuntimeError("Arm backend is not connected.")
        self._controller.set_control(np.asarray(joint_positions, dtype=np.float64))

    def move_to_start(self) -> None:
        if self._robot is None:
            raise RuntimeError("Arm backend is not connected.")
        controller_was_running = self._controller is not None
        if controller_was_running:
            self._stop_controller()
        try:
            self._robot.move_to_start()
        finally:
            if controller_was_running:
                self._start_controller()


@dataclass
class PikaGripperHardwareDriver:
    serial_port: str
    max_width_mm: float = 90.0
    command_rate_limit_hz: float | None = 15.0
    command_deadband_mm: float = 0.5

    def __post_init__(self):
        _silence_pika_logs()
        try:
            from pika.gripper import Gripper
        except Exception as e:  # pragma: no cover - exercised with real hardware only
            raise ImportError(
                "franka_research3 requires pika.gripper for hardware gripper control. "
                "Install the Pika gripper SDK in the runtime environment to use this backend."
            ) from e

        self._gripper_cls = Gripper
        self._gripper = None
        self._last_command_width_mm: float | None = None
        self._last_command_time_s: float | None = None
        self._pending_command_width_mm: float | None = None

    def connect(self) -> None:
        self._gripper = self._gripper_cls(self.serial_port)
        if not self._gripper.connect():
            raise ConnectionError(f"Could not connect to Pika gripper on {self.serial_port}.")
        if not self._gripper.enable():
            raise ConnectionError("Could not enable the Pika gripper.")

    def disconnect(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.disable()
            finally:
                self._gripper.disconnect()
                self._gripper = None
        self._last_command_width_mm = None
        self._last_command_time_s = None
        self._pending_command_width_mm = None

    def get_position(self) -> float:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        width_mm = float(self._gripper.get_gripper_distance())
        return float(np.clip(width_mm / self.max_width_mm, 0.0, 1.0))

    def set_position(self, normalized_position: float) -> None:
        if self._gripper is None:
            raise RuntimeError("Gripper backend is not connected.")
        target_width_mm = float(np.clip(normalized_position, 0.0, 1.0) * self.max_width_mm)
        if (
            self._last_command_width_mm is not None
            and abs(target_width_mm - self._last_command_width_mm) < self.command_deadband_mm
        ):
            self._pending_command_width_mm = None
            return

        self._pending_command_width_mm = target_width_mm

        now = time.perf_counter()
        if self.command_rate_limit_hz is not None and self._last_command_time_s is not None:
            min_interval_s = 1.0 / self.command_rate_limit_hz
            if now - self._last_command_time_s < min_interval_s:
                return

        pending_width_mm = self._pending_command_width_mm
        if pending_width_mm is None:
            return

        # Collapse repeated control-loop writes into the newest target before touching the serial link.
        self._gripper.set_gripper_distance(pending_width_mm)
        self._last_command_width_mm = pending_width_mm
        self._last_command_time_s = now
        self._pending_command_width_mm = None


_TACTILE_IMAGE_SHAPE = (50, 10)
_TACTILE_SIDE_COUNT = int(np.prod(_TACTILE_IMAGE_SHAPE))
_TACTILE_VALID_COUNT = 448
_TACTILE_COMPRESSED_SIDE_VALID_COUNT = _TACTILE_VALID_COUNT // 2
_TACTILE_INVALID_VALUE = 255.0


def _load_tactile_valid_mask(mask_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(mask_path).read_text(encoding="utf-8"))
    mask = np.asarray(payload.get("mask"), dtype=np.float32)
    if tuple(mask.shape) != _TACTILE_IMAGE_SHAPE:
        raise ValueError(
            f"DAS tactile valid mask must have shape {_TACTILE_IMAGE_SHAPE}, got {tuple(mask.shape)} from {mask_path}."
        )
    valid_count = int(mask.astype(bool).sum())
    if valid_count != _TACTILE_VALID_COUNT:
        raise ValueError(
            f"DAS tactile valid mask must contain {_TACTILE_VALID_COUNT} valid cells, got {valid_count} from {mask_path}."
        )
    return mask


def _load_tactile_baselines(baseline_path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    try:
        baseline_data = payload["data"][0]["tactiles"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid DAS tactile baseline format in {baseline_path}.") from exc

    baselines: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        values = np.asarray(baseline_data[side], dtype=np.float32)
        if values.size != _TACTILE_SIDE_COUNT:
            raise ValueError(
                f"DAS tactile baseline '{side}' must contain {_TACTILE_SIDE_COUNT} values, got {values.size} from {baseline_path}."
            )
        baselines[side] = values.reshape(_TACTILE_IMAGE_SHAPE)
    return baselines


def _scatter_tactile_valid_values(valid_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = np.asarray(valid_values, dtype=np.float32).reshape(-1)
    valid_flat = valid_mask.astype(bool).reshape(-1)
    if valid_values.size != int(valid_flat.sum()):
        raise ValueError(
            f"Expected {int(valid_flat.sum())} tactile valid values, got {valid_values.size}."
        )
    dense = np.full(valid_flat.shape, _TACTILE_INVALID_VALUE, dtype=np.float32)
    dense[valid_flat] = valid_values
    return dense.reshape(_TACTILE_IMAGE_SHAPE)


def _build_tactile_horizontal_mirror_pairs(valid_mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    valid = valid_mask.astype(bool)
    rows, cols = valid.shape
    if cols % 2 != 0:
        raise ValueError(f"Expected an even tactile width for horizontal mirror expansion, got {cols}.")

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    half_width = cols // 2
    for row in range(rows):
        for col in range(half_width):
            mirror_col = cols - 1 - col
            if not valid[row, col]:
                if valid[row, mirror_col]:
                    raise ValueError(
                        "DAS tactile valid mask must be horizontally symmetric for bilateral compressed decoding."
                    )
                continue
            if not valid[row, mirror_col]:
                raise ValueError(
                    "DAS tactile valid mask must be horizontally symmetric for bilateral compressed decoding."
                )
            pairs.append(((row, col), (row, mirror_col)))

    if len(pairs) != _TACTILE_COMPRESSED_SIDE_VALID_COUNT:
        raise ValueError(
            f"Expected {_TACTILE_COMPRESSED_SIDE_VALID_COUNT} compressed tactile pairs, got {len(pairs)}."
        )
    return pairs


def _expand_tactile_horizontal_mirror_values(valid_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = np.asarray(valid_values, dtype=np.float32).reshape(-1)
    pairs = _build_tactile_horizontal_mirror_pairs(valid_mask)
    if valid_values.size != len(pairs):
        raise ValueError(f"Expected {len(pairs)} compressed tactile values, got {valid_values.size}.")

    dense = np.full(valid_mask.shape, _TACTILE_INVALID_VALUE, dtype=np.float32)
    for value, (left_pos, right_pos) in zip(valid_values, pairs, strict=True):
        dense[left_pos] = value
        dense[right_pos] = value
    return dense


def _decode_tactile_direct_spatial_split(record_data: bytes, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.frombuffer(record_data, dtype=np.uint8).astype(np.float32)
    if values.size != _TACTILE_VALID_COUNT:
        raise ValueError(f"Expected {_TACTILE_VALID_COUNT} tactile bytes, got {values.size}.")

    combined = _scatter_tactile_valid_values(values, valid_mask)
    pairs = _build_tactile_horizontal_mirror_pairs(valid_mask)
    left = np.full(valid_mask.shape, _TACTILE_INVALID_VALUE, dtype=np.float32)
    right = np.full(valid_mask.shape, _TACTILE_INVALID_VALUE, dtype=np.float32)
    for left_pos, right_pos in pairs:
        left_value = combined[left_pos]
        right_value = combined[right_pos]
        left[left_pos] = left_value
        left[right_pos] = left_value
        right[left_pos] = right_value
        right[right_pos] = right_value
    return left, right


def _decode_float32_payload(record_data: bytes, count: int) -> np.ndarray:
    if len(record_data) != count * 4:
        raise ValueError(f"Expected {count * 4} bytes for {count} float32 values, got {len(record_data)}.")

    candidates: list[tuple[float, np.ndarray]] = []
    for endian in ("<", ">"):
        try:
            values = np.asarray(struct.unpack(f"{endian}{count}f", record_data), dtype=np.float32)
        except struct.error:
            continue
        finite_ratio = float(np.isfinite(values).mean())
        range_ratio = float(((values >= -1e-3) & (values <= (_TACTILE_INVALID_VALUE + 1e-3))).mean())
        candidates.append((finite_ratio + range_ratio, values))

    if not candidates:
        raise ValueError("Could not decode tactile float32 payload.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _decode_tactile_record(record_data: bytes, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_valid_values = _TACTILE_VALID_COUNT * 2
    expected_dense_values = _TACTILE_SIDE_COUNT * 2

    if len(record_data) == _TACTILE_VALID_COUNT:
        return _decode_tactile_direct_spatial_split(record_data, valid_mask)

    if len(record_data) == expected_valid_values:
        values = np.frombuffer(record_data, dtype=np.uint8).astype(np.float32)
        left_valid = values[:_TACTILE_VALID_COUNT]
        right_valid = values[_TACTILE_VALID_COUNT:]
        return (
            _scatter_tactile_valid_values(left_valid, valid_mask),
            _scatter_tactile_valid_values(right_valid, valid_mask),
        )

    if len(record_data) == expected_valid_values * 4:
        values = _decode_float32_payload(record_data, expected_valid_values)
        left_valid = values[:_TACTILE_VALID_COUNT]
        right_valid = values[_TACTILE_VALID_COUNT:]
        return (
            _scatter_tactile_valid_values(left_valid, valid_mask),
            _scatter_tactile_valid_values(right_valid, valid_mask),
        )

    if len(record_data) == expected_dense_values:
        values = np.frombuffer(record_data, dtype=np.uint8).astype(np.float32).reshape(2, *_TACTILE_IMAGE_SHAPE)
        return values[0], values[1]

    if len(record_data) == expected_dense_values * 4:
        values = _decode_float32_payload(record_data, expected_dense_values).reshape(2, *_TACTILE_IMAGE_SHAPE)
        return values[0], values[1]

    raise ValueError(
        "Unsupported DAS tactile payload length: "
        f"got {len(record_data)} bytes, expected one of "
        f"{{{expected_valid_values}, {expected_valid_values * 4}, {expected_dense_values}, {expected_dense_values * 4}}}."
    )


@dataclass
class DasGripperHardwareDriver:
    serial_port: str
    gen_con_sdk_path: str | None = None
    baudrate: int = 921600
    update_frequency_hz: float = 50.0
    tactile_frequency_hz: float | None = None
    tactile_valid_mask_path: str | None = None
    tactile_baseline_path: str | None = None
    tactile_timeout_s: float = 2.0
    min_distance_m: float = 0.0
    max_distance_m: float = 0.103
    grasp_threshold_m: float = 0.002
    initial_position: float = 1.0
    command_rate_limit_hz: float | None = 15.0
    command_deadband_m: float = 0.0005

    def __post_init__(self):
        if self.max_distance_m <= self.min_distance_m:
            raise ValueError("Das gripper max_distance_m must be greater than min_distance_m.")
        if self.update_frequency_hz <= 0:
            raise ValueError("Das gripper update_frequency_hz must be positive.")
        if self.tactile_frequency_hz is not None and self.tactile_frequency_hz <= 0:
            raise ValueError("Das gripper tactile_frequency_hz must be positive when provided.")
        if self.baudrate <= 0:
            raise ValueError("Das gripper baudrate must be positive.")
        if self.command_deadband_m < 0:
            raise ValueError("Das gripper command_deadband_m must be non-negative.")
        if self.tactile_timeout_s <= 0:
            raise ValueError("Das gripper tactile_timeout_s must be positive.")
        if (self.tactile_valid_mask_path is None) != (self.tactile_baseline_path is None):
            raise ValueError("DAS tactile valid mask and baseline paths must be provided together.")
        self._databus_cls = _resolve_das_databus_cls(self.gen_con_sdk_path)
        self._databus = None
        self._lock = threading.Lock()
        self._tactile_lock = threading.Lock()
        self._gripper_state_updated = False
        self._tactile_state_updated = False
        self._position_m: float | None = None
        self._target_distance_m: float | None = None
        self._last_command_distance_m: float | None = None
        self._last_command_time_s: float | None = None
        self._pending_command_distance_m: float | None = None
        self._latest_tactile_observation: dict[str, np.ndarray] | None = None

        self._tactile_valid_mask: np.ndarray | None = None
        self._tactile_baselines: dict[str, np.ndarray] | None = None
        if self.tactile_valid_mask_path is not None and self.tactile_baseline_path is not None:
            self._tactile_valid_mask = _load_tactile_valid_mask(self.tactile_valid_mask_path)
            self._tactile_baselines = _load_tactile_baselines(self.tactile_baseline_path)
            if self.tactile_frequency_hz is None:
                self.tactile_frequency_hz = self.update_frequency_hz

    @property
    def tactile_enabled(self) -> bool:
        return self._tactile_valid_mask is not None and self._tactile_baselines is not None

    def connect(self) -> None:
        self._databus = self._databus_cls(
            tty_port=self.serial_port,
            baudrate=self.baudrate,
            encoder_freq=self.update_frequency_hz,
            tactile_freq=self.tactile_frequency_hz if self.tactile_enabled else None,
            tactile_callback=self._tactile_callback if self.tactile_enabled else None,
            encoder_callback=self._encoder_callback,
        )
        initial_distance_m = self._scale_to_distance(self.initial_position)
        self._databus.set_target_distance(initial_distance_m)
        self._target_distance_m = initial_distance_m
        self._last_command_distance_m = initial_distance_m

        deadline = time.perf_counter() + max(2.0, self.tactile_timeout_s)
        while True:
            tactile_ready = (not self.tactile_enabled) or self._tactile_state_updated
            if self._gripper_state_updated and tactile_ready:
                break
            if time.perf_counter() >= deadline:
                if not self._gripper_state_updated:
                    raise TimeoutError("DasController did not receive encoder data within timeout.")
                raise TimeoutError("DasController did not receive tactile data within timeout.")
            time.sleep(0.001)

    def disconnect(self) -> None:
        if self._databus is not None:
            stop = getattr(self._databus, "stop", None)
            if callable(stop):
                stop()
            self._databus = None
        self._gripper_state_updated = False
        self._tactile_state_updated = False
        self._position_m = None
        self._target_distance_m = None
        self._last_command_distance_m = None
        self._last_command_time_s = None
        self._pending_command_distance_m = None
        with self._tactile_lock:
            self._latest_tactile_observation = None

    def _encoder_callback(self, record_data: bytes) -> None:
        try:
            distance_m = float(struct.unpack(">f", record_data)[0])
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to parse DAS encoder update: %s", exc)
            return

        with self._lock:
            self._position_m = float(np.clip(distance_m, self.min_distance_m, self.max_distance_m))
            if self._target_distance_m is not None and self.grasp_threshold_m > 0:
                _ = self._position_m > (self._target_distance_m + self.grasp_threshold_m)
        self._gripper_state_updated = True

    def _tactile_callback(self, record_data: bytes) -> None:
        if self._tactile_valid_mask is None or self._tactile_baselines is None:
            return
        try:
            left_raw, right_raw = _decode_tactile_record(record_data, self._tactile_valid_mask)
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to parse DAS tactile update: %s", exc)
            return

        valid_mask = self._tactile_valid_mask.astype(np.float32)
        left_clean = (left_raw - self._tactile_baselines["left"]) * valid_mask
        right_clean = (right_raw - self._tactile_baselines["right"]) * valid_mask
        left_clean[valid_mask == 0.0] = 0.0
        right_clean[valid_mask == 0.0] = 0.0

        with self._tactile_lock:
            self._latest_tactile_observation = {
                "observation.tactile.left_raw": np.asarray(left_raw, dtype=np.float32).copy(),
                "observation.tactile.right_raw": np.asarray(right_raw, dtype=np.float32).copy(),
                "observation.tactile.valid_mask": valid_mask.copy(),
                "observation.tactile.left_clean": np.asarray(left_clean, dtype=np.float32).copy(),
                "observation.tactile.right_clean": np.asarray(right_clean, dtype=np.float32).copy(),
            }
        self._tactile_state_updated = True

    def _scale_to_distance(self, normalized_position: float) -> float:
        normalized = float(np.clip(normalized_position, 0.0, 1.0))
        return self.min_distance_m + normalized * (self.max_distance_m - self.min_distance_m)

    def get_position(self) -> float:
        if self._databus is None:
            raise RuntimeError("Gripper backend is not connected.")
        with self._lock:
            distance_m = self._position_m
        if distance_m is None:
            distance_m = self._target_distance_m
        if distance_m is None:
            raise RuntimeError("Das gripper backend has not received any position update yet.")
        span = self.max_distance_m - self.min_distance_m
        if span <= 0:
            return 0.0
        return float(np.clip((distance_m - self.min_distance_m) / span, 0.0, 1.0))

    def get_tactile_observation(self) -> dict[str, np.ndarray]:
        if not self.tactile_enabled:
            return {}
        if self._databus is None:
            raise RuntimeError("Gripper backend is not connected.")
        with self._tactile_lock:
            tactile = self._latest_tactile_observation
            if tactile is None:
                raise RuntimeError("Das gripper backend has not received any tactile update yet.")
            return {key: value.copy() for key, value in tactile.items()}

    def set_position(self, normalized_position: float) -> None:
        if self._databus is None:
            raise RuntimeError("Gripper backend is not connected.")

        target_distance_m = self._scale_to_distance(normalized_position)
        if (
            self._last_command_distance_m is not None
            and abs(target_distance_m - self._last_command_distance_m) < self.command_deadband_m
        ):
            self._pending_command_distance_m = None
            return

        self._pending_command_distance_m = target_distance_m
        now = time.perf_counter()
        if self.command_rate_limit_hz is not None and self._last_command_time_s is not None:
            min_interval_s = 1.0 / self.command_rate_limit_hz
            if now - self._last_command_time_s < min_interval_s:
                return

        pending_distance_m = self._pending_command_distance_m
        if pending_distance_m is None:
            return

        self._databus.set_target_distance(pending_distance_m)
        self._target_distance_m = pending_distance_m
        self._last_command_distance_m = pending_distance_m
        self._last_command_time_s = now
        self._pending_command_distance_m = None


@dataclass
class MockGripperDriver:
    initial_position: float = 1.0

    def __post_init__(self):
        self._position = float(np.clip(self.initial_position, 0.0, 1.0))
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_position(self) -> float:
        if not self._connected:
            raise RuntimeError("Gripper backend is not connected.")
        return self._position

    def set_position(self, normalized_position: float) -> None:
        if not self._connected:
            raise RuntimeError("Gripper backend is not connected.")
        self._position = float(np.clip(normalized_position, 0.0, 1.0))


@dataclass
class PlacoKinematicsDriver:
    urdf_path: str
    target_frame_name: str
    joint_names: list[str]

    def __post_init__(self):
        self._kinematics = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name=self.target_frame_name,
            joint_names=self.joint_names,
        )

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        joint_positions_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64))
        return self._kinematics.forward_kinematics(joint_positions_deg)

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
        current_joint_positions_deg = np.rad2deg(np.asarray(current_joint_positions_rad, dtype=np.float64))
        solution_deg = self._kinematics.inverse_kinematics(current_joint_positions_deg, desired_pose)
        return np.deg2rad(np.asarray(solution_deg, dtype=np.float64))


@dataclass
class RuckigOTGDriver:
    dof: int
    dt: float
    max_velocity: list[float]
    max_acceleration: list[float]
    max_jerk: list[float]
    min_position: list[float] | None = None
    max_position: list[float] | None = None
    synchronization: bool = True
    sync_mode: str = "time"

    def __post_init__(self):
        try:
            from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization
        except Exception as e:  # pragma: no cover - exercised in docker / hardware runtime
            raise ImportError(
                "franka_research3 OTG requires ruckig in the runtime environment. "
                "Use the FR3 docker image or install ruckig manually."
            ) from e

        self._result = Result
        self._ruckig = Ruckig(self.dof, self.dt)
        self._input = InputParameter(self.dof)
        self._output = OutputParameter(self.dof)

        self._max_velocity = np.asarray(self.max_velocity, dtype=np.float64)
        self._max_acceleration = np.asarray(self.max_acceleration, dtype=np.float64)
        self._max_jerk = np.asarray(self.max_jerk, dtype=np.float64)
        self._min_position = (
            None if self.min_position is None else np.asarray(self.min_position, dtype=np.float64)
        )
        self._max_position = (
            None if self.max_position is None else np.asarray(self.max_position, dtype=np.float64)
        )
        self._current_velocity = np.zeros(self.dof, dtype=np.float64)
        self._current_acceleration = np.zeros(self.dof, dtype=np.float64)

        self._input.max_velocity = self._max_velocity.tolist()
        self._input.max_acceleration = self._max_acceleration.tolist()
        self._input.max_jerk = self._max_jerk.tolist()
        if self._min_position is not None and self._max_position is not None:
            self._input.min_position = self._min_position.tolist()
            self._input.max_position = self._max_position.tolist()

        if not self.synchronization or self.sync_mode.lower() == "none":
            self._input.synchronization = Synchronization.No
        elif self.sync_mode.lower() == "phase":
            self._input.synchronization = Synchronization.Phase
        else:
            self._input.synchronization = Synchronization.Time

    def reset(self, current_joint_positions: np.ndarray) -> None:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        if current.shape != (self.dof,):
            raise ValueError(f"Expected current_joint_positions shape {(self.dof,)}, got {current.shape}.")

        self._current_velocity.fill(0.0)
        self._current_acceleration.fill(0.0)
        self._input.current_position = current.tolist()
        self._input.current_velocity = self._current_velocity.tolist()
        self._input.current_acceleration = self._current_acceleration.tolist()
        self._input.target_position = current.tolist()
        self._input.target_velocity = [0.0] * self.dof
        self._input.target_acceleration = [0.0] * self.dof

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        target = np.asarray(target_joint_positions, dtype=np.float64)
        if current.shape != (self.dof,) or target.shape != (self.dof,):
            raise ValueError(
                f"Expected current and target shapes {(self.dof,)}, got {current.shape} and {target.shape}."
            )

        if self._min_position is not None and self._max_position is not None:
            target = np.clip(target, self._min_position, self._max_position)

        self._input.current_position = current.tolist()
        self._input.current_velocity = self._current_velocity.tolist()
        self._input.current_acceleration = self._current_acceleration.tolist()
        self._input.target_position = target.tolist()
        self._input.target_velocity = [0.0] * self.dof
        self._input.target_acceleration = [0.0] * self.dof

        result = self._ruckig.update(self._input, self._output)
        if result not in (self._result.Working, self._result.Finished):
            raise RuntimeError(f"Ruckig OTG update failed with result={result}.")

        new_position = np.asarray(self._output.new_position, dtype=np.float64)
        self._current_velocity = np.asarray(self._output.new_velocity, dtype=np.float64)
        self._current_acceleration = np.asarray(self._output.new_acceleration, dtype=np.float64)
        self._output.pass_to_input(self._input)
        return new_position
