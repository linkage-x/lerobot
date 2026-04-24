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

import inspect
import logging
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from ..teleoperator import Teleoperator
from .configuration_quest3 import Quest3GripperMapping, Quest3Hand, Quest3TeleopConfig

logger = logging.getLogger(__name__)


def _safe_mat_update(previous: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    candidate = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    det = np.linalg.det(candidate[:3, :3])
    if not np.all(np.isfinite(candidate)) or not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return previous.copy(), False
    return candidate, True


class Quest3Teleop(Teleoperator):
    config_class = Quest3TeleopConfig
    name = "quest3"

    # Same OpenXR <-> robot basis convention as the existing HIROL Quest3 integration.
    T_ROBOT_OPENXR = np.array([[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    T_OPENXR_ROBOT = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    WRIST_INDEX = 0
    THUMB_TIP_INDEX = 4
    INDEX_TIP_INDEX = 9

    def __init__(self, config: Quest3TeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._vuer = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._hand_mats: dict[str, np.ndarray] = {
            "left": np.tile(np.eye(4, dtype=np.float64), (25, 1, 1)),
            "right": np.tile(np.eye(4, dtype=np.float64), (25, 1, 1)),
        }
        self._hand_states: dict[str, dict[str, float | bool]] = {
            "left": {"pinch": False, "pinch_value": 0.0, "squeeze": False, "squeeze_value": 0.0},
            "right": {"pinch": False, "pinch_value": 0.0, "squeeze": False, "squeeze_value": 0.0},
        }
        self._last_update_s: dict[str, float] = {"left": float("-inf"), "right": float("-inf")}
        self._baseline_pose: np.ndarray | None = None
        self._last_clutch_active = False
        self._last_gripper = float(np.clip(config.initial_gripper, 0.0, 1.0))
        self._filtered_gripper = self._last_gripper
        self._last_filtered_gripper_time = float("-inf")
        self._last_hand_parse_warning_s = float("-inf")

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper": float,
            "tracking_valid": bool,
            "wrist_x": float,
            "wrist_y": float,
            "wrist_z": float,
            "wrist_qx": float,
            "wrist_qy": float,
            "wrist_qz": float,
            "wrist_qw": float,
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        self._baseline_pose = None

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        try:
            from vuer import Vuer
            from vuer.schemas import Hands, MotionControllers
        except ImportError as exc:
            raise ImportError(
                "Quest3 teleop requires the optional 'vuer' package. Install it in the runtime environment "
                "before using teleop.type=quest3."
            ) from exc

        vuer_kwargs: dict[str, Any] = {
            "host": self.config.host,
            "port": int(self.config.port),
            "queries": dict(grid=False),
            "queue_len": 3,
        }
        if self.config.cert_file is not None:
            vuer_kwargs["cert"] = str(self.config.cert_file)
        if self.config.key_file is not None:
            vuer_kwargs["key"] = str(self.config.key_file)

        self._vuer = Vuer(**vuer_kwargs)
        self._vuer.add_handler("HAND_MOVE")(self._on_hand_move)
        self._vuer.add_handler("CONTROLLER_MOVE")(self._on_controller_move)

        async def main_scene(session, fps=60):
            del fps
            if self.config.use_hand_tracking:
                session.upsert(Hands(stream=True, key="hands", hideLeft=False, hideRight=False))
            else:
                session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True))
            while True:
                import asyncio

                await asyncio.sleep(1.0)

        self._vuer.spawn(start=False)(main_scene)
        self._thread = threading.Thread(target=self._vuer.run, name="quest3-vuer", daemon=True)
        self._thread.start()
        self._is_connected = True

    def _selected_hand(self) -> str:
        return "left" if self.config.hand == Quest3Hand.LEFT else "right"

    def _zero_action(self) -> RobotAction:
        return {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": self._filtered_gripper,
            "tracking_valid": False,
            "wrist_x": 0.0,
            "wrist_y": 0.0,
            "wrist_z": 0.0,
            "wrist_qx": 0.0,
            "wrist_qy": 0.0,
            "wrist_qz": 0.0,
            "wrist_qw": 1.0,
        }

    def _wrist_action_fields(self, pose: np.ndarray, *, tracking_valid: bool) -> dict[str, float | bool]:
        quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return {
            "tracking_valid": bool(tracking_valid),
            "wrist_x": float(pose[0, 3]),
            "wrist_y": float(pose[1, 3]),
            "wrist_z": float(pose[2, 3]),
            "wrist_qx": float(quat_xyzw[0]),
            "wrist_qy": float(quat_xyzw[1]),
            "wrist_qz": float(quat_xyzw[2]),
            "wrist_qw": float(quat_xyzw[3]),
        }

    def _hand_pose_robot_frame(self, hand: str) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool], float]:
        with self._lock:
            hand_mats = self._hand_mats[hand].copy()
            states = dict(self._hand_states[hand])
            last_update_s = float(self._last_update_s[hand])

        wrist_openxr, wrist_valid = _safe_mat_update(np.eye(4, dtype=np.float64), hand_mats[self.WRIST_INDEX])
        if not wrist_valid:
            return np.eye(4, dtype=np.float64), hand_mats, states, last_update_s

        wrist_robot = self.T_ROBOT_OPENXR @ wrist_openxr @ self.T_OPENXR_ROBOT
        return wrist_robot, hand_mats, states, last_update_s

    def _clutch_active(self, states: dict[str, float | bool]) -> bool:
        source = self.config.clutch_source
        threshold = float(self.config.clutch_threshold)
        if source == "pinch":
            return bool(states.get("pinch", False)) or float(states.get("pinch_value", 0.0)) >= threshold
        if source == "squeeze":
            return bool(states.get("squeeze", False)) or float(states.get("squeeze_value", 0.0)) >= threshold
        if source == "always":
            return True
        return False

    def _gripper_from_fingertips(self, hand_mats: np.ndarray) -> float:
        return float(np.clip(self._gripper_from_fingertips_unclipped(hand_mats), 0.0, 1.0))

    def _gripper_from_fingertips_unclipped(self, hand_mats: np.ndarray) -> float:
        distance = self._fingertip_distance(hand_mats)
        open_d = float(self.config.open_fingertip_distance_m)
        closed_d = float(self.config.closed_fingertip_distance_m)
        denom = max(open_d - closed_d, 1e-6)
        return float((distance - closed_d) / denom)

    def _fingertip_distance(self, hand_mats: np.ndarray) -> float:
        thumb = np.asarray(hand_mats[self.THUMB_TIP_INDEX][:3, 3], dtype=np.float64)
        index = np.asarray(hand_mats[self.INDEX_TIP_INDEX][:3, 3], dtype=np.float64)
        return float(np.linalg.norm(thumb - index))

    def _raw_gripper(self, states: dict[str, float | bool], hand_mats: np.ndarray) -> float:
        if self.config.gripper_mapping == Quest3GripperMapping.FINGERTIP_DISTANCE:
            return self._gripper_from_fingertips(hand_mats)
        return self._gripper_from_pinch_value(float(states.get("pinch_value", 0.0)))

    def _gripper_from_pinch_value(self, pinch_value: float) -> float:
        open_value = float(self.config.open_pinch_value)
        closed_value = float(self.config.closed_pinch_value)
        denom = open_value - closed_value
        if abs(denom) < 1e-6:
            return float(np.clip(pinch_value, 0.0, 1.0))
        return float(np.clip((pinch_value - closed_value) / denom, 0.0, 1.0))

    def _filter_gripper_command(self, value: float) -> float:
        raw_value = float(np.clip(value, 0.0, 1.0))
        now = time.perf_counter()
        last_value = float(self._filtered_gripper)
        last_time = float(self._last_filtered_gripper_time)
        filtered = raw_value
        if np.isfinite(last_time):
            if self.config.gripper_cmd_max_rate > 0.0:
                step_dt = 1.0 / max(float(self.config.frequency), 1.0)
                max_delta = float(self.config.gripper_cmd_max_rate) * step_dt
                delta = filtered - last_value
                if abs(delta) > max_delta:
                    filtered = last_value + np.sign(delta) * max_delta
            if self.config.gripper_cmd_ema_alpha > 0.0:
                alpha = float(np.clip(self.config.gripper_cmd_ema_alpha, 0.0, 1.0))
                filtered = alpha * filtered + (1.0 - alpha) * last_value
        self._filtered_gripper = float(np.clip(filtered, 0.0, 1.0))
        self._last_filtered_gripper_time = now
        return self._filtered_gripper

    def sync_gripper_baseline(self, normalized_position: float) -> float:
        value = float(np.clip(normalized_position, 0.0, 1.0))
        self._last_gripper = value
        self._filtered_gripper = value
        self._last_filtered_gripper_time = time.perf_counter()
        return value

    def set_gripper(self, normalized_position: float) -> None:
        self.sync_gripper_baseline(normalized_position)

    def latest_debug_state(self) -> dict[str, Any]:
        hand = self._selected_hand()
        pose, hand_mats, states, last_update_s = self._hand_pose_robot_frame(hand)
        age_s = time.perf_counter() - last_update_s
        return {
            "hand": hand,
            "tracking_age_s": age_s,
            "tracking_valid": age_s <= float(self.config.lost_tracking_timeout_s),
            "wrist_pose": pose.copy(),
            "pinch": states.get("pinch", False),
            "pinch_value": states.get("pinch_value", 0.0),
            "pinch_gripper": self._gripper_from_pinch_value(float(states.get("pinch_value", 0.0))),
            "squeeze": states.get("squeeze", False),
            "squeeze_value": states.get("squeeze_value", 0.0),
            "fingertip_distance_m": self._fingertip_distance(hand_mats),
            "gripper_unclipped": self._gripper_from_fingertips_unclipped(hand_mats),
            "gripper": self._raw_gripper(states, hand_mats),
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        hand = self._selected_hand()
        pose, hand_mats, states, last_update_s = self._hand_pose_robot_frame(hand)
        tracking_valid = time.perf_counter() - last_update_s <= float(self.config.lost_tracking_timeout_s)
        if not tracking_valid:
            self._baseline_pose = None
            self._last_clutch_active = False
            return self._zero_action()

        raw_gripper = self._raw_gripper(states, hand_mats)
        filtered_gripper = self._filter_gripper_command(raw_gripper)
        clutch_active = self._clutch_active(states)
        if not clutch_active:
            self._baseline_pose = None
            self._last_clutch_active = False
            action = self._zero_action()
            action["gripper"] = filtered_gripper
            action.update(self._wrist_action_fields(pose, tracking_valid=True))
            return action

        if self._baseline_pose is None or not self._last_clutch_active:
            self._baseline_pose = pose.copy()
            self._last_clutch_active = True

        delta_pos = (pose[:3, 3] - self._baseline_pose[:3, 3]) * float(self.config.translation_scale)
        delta_pos = np.where(np.abs(delta_pos) >= float(self.config.translation_deadband_m), delta_pos, 0.0)

        delta_rotvec = np.zeros(3, dtype=np.float64)
        if self.config.enable_rotation:
            delta_rot = self._baseline_pose[:3, :3].T @ pose[:3, :3]
            delta_rotvec = Rotation.from_matrix(delta_rot).as_rotvec() * float(self.config.rotation_scale)
            delta_rotvec = np.where(
                np.abs(delta_rotvec) >= float(self.config.rotation_deadband_rad),
                delta_rotvec,
                0.0,
            )

        action = {
            "enabled": bool(np.any(delta_pos) or np.any(delta_rotvec)),
            "target_x": float(delta_pos[0]),
            "target_y": float(delta_pos[1]),
            "target_z": float(delta_pos[2]),
            "target_wx": float(delta_rotvec[0]),
            "target_wy": float(delta_rotvec[1]),
            "target_wz": float(delta_rotvec[2]),
            "gripper": filtered_gripper,
        }
        action.update(self._wrist_action_fields(pose, tracking_valid=True))
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self._vuer is not None:
                close_ws = getattr(self._vuer, "close_ws", None)
                if callable(close_ws):
                    signature = inspect.signature(close_ws)
                    if len(signature.parameters) == 0:
                        close_ws()
                    else:
                        logger.debug("Skipping Vuer.close_ws during disconnect; this Vuer version requires ws_id.")
        finally:
            self._vuer = None
            self._thread = None
            self._is_connected = False
            self._baseline_pose = None
            self._last_clutch_active = False

    async def _on_hand_move(self, event, session, fps=60):
        del session, fps
        try:
            now = time.perf_counter()
            value = event.value if isinstance(event.value, dict) else {}
            left_mats = self._extract_hand_mats(value.get("left"))
            right_mats = self._extract_hand_mats(value.get("right"))
            left_state = self._extract_hand_state(value.get("leftState"))
            right_state = self._extract_hand_state(value.get("rightState"))
            with self._lock:
                if left_mats is not None:
                    self._hand_mats["left"] = left_mats
                    self._hand_states["left"] = left_state
                    self._last_update_s["left"] = now
                if right_mats is not None:
                    self._hand_mats["right"] = right_mats
                    self._hand_states["right"] = right_state
                    self._last_update_s["right"] = now
        except Exception:
            now = time.perf_counter()
            if now - self._last_hand_parse_warning_s > 2.0:
                self._last_hand_parse_warning_s = now
                logger.exception(
                    "Failed to parse Quest3 HAND_MOVE event. payload=%s",
                    self._summarize_payload(getattr(event, "value", None)),
                )

    async def _on_controller_move(self, event, session, fps=60):
        del event, session, fps

    @staticmethod
    def _extract_hand_mats(payload: Any) -> np.ndarray | None:
        if payload is None:
            return None
        ext_data = getattr(payload, "data", None)
        if isinstance(ext_data, bytes):
            decoded = Quest3Teleop._decode_msgpack_ext_float_array(ext_data)
            return None if decoded is None else Quest3Teleop._extract_hand_mats(decoded)
        if isinstance(payload, dict):
            for key in ("data", "values", "value", "array", "buffer"):
                if key in payload:
                    payload = payload[key]
                    nested_ext_data = getattr(payload, "data", None)
                    if isinstance(nested_ext_data, bytes):
                        decoded = Quest3Teleop._decode_msgpack_ext_float_array(nested_ext_data)
                        return None if decoded is None else Quest3Teleop._extract_hand_mats(decoded)
                    break
            else:
                numeric_keys = [key for key in payload if str(key).isdigit()]
                if numeric_keys:
                    payload = [payload[key] for key in sorted(numeric_keys, key=lambda item: int(str(item)))]
        array = np.asarray(payload, dtype=np.float64)
        if array.shape == (25, 4, 4):
            return array.copy()
        if array.shape == (25, 16):
            return array.reshape(25, 4, 4, order="F")
        values = array.reshape(-1)
        if values.size < 25 * 16:
            return None
        return values[: 25 * 16].reshape(25, 4, 4, order="F")

    @staticmethod
    def _decode_msgpack_ext_float_array(data: bytes) -> np.ndarray | None:
        target_values = 25 * 16
        target_f32_bytes = target_values * np.dtype("<f4").itemsize
        target_f64_bytes = target_values * np.dtype("<f8").itemsize
        for dtype, target_bytes in (("<f4", target_f32_bytes), ("<f8", target_f64_bytes), (">f4", target_f32_bytes), (">f8", target_f64_bytes)):
            itemsize = np.dtype(dtype).itemsize
            max_offset = min(16, max(0, len(data) - target_bytes) + 1)
            for offset in range(max_offset):
                usable = len(data) - offset
                if usable < target_bytes or usable % itemsize != 0:
                    continue
                array = np.frombuffer(data, dtype=dtype, count=target_values, offset=offset)
                if np.all(np.isfinite(array)):
                    return array.astype(np.float64, copy=False)
        return None

    @staticmethod
    def _extract_hand_state(payload: Any) -> dict[str, float | bool]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "pinch": bool(payload.get("pinch", False)),
            "pinch_value": float(payload.get("pinchValue", 0.0)),
            "squeeze": bool(payload.get("squeeze", False)),
            "squeeze_value": float(payload.get("squeezeValue", 0.0)),
        }

    @staticmethod
    def _summarize_payload(payload: Any) -> str:
        if not isinstance(payload, dict):
            return f"type={type(payload).__name__}"
        parts = []
        for key, value in payload.items():
            if isinstance(value, dict):
                parts.append(f"{key}:dict(keys={list(value)[:8]})")
            elif hasattr(value, "code") and hasattr(value, "data"):
                data = getattr(value, "data", b"")
                parts.append(f"{key}:ExtType(code={getattr(value, 'code', None)}, data_len={len(data)})")
            elif isinstance(value, (list, tuple)):
                parts.append(f"{key}:{type(value).__name__}(len={len(value)})")
            else:
                parts.append(f"{key}:{type(value).__name__}")
        return "{" + ", ".join(parts[:8]) + "}"
