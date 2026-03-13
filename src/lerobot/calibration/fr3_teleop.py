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

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ACTION_KEYS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper",
)
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ROT_AXIS_TO_INDEX = {"wx": 0, "wy": 1, "wz": 2}


def _normalize_action(action: dict[str, Any] | None = None, *, gripper: float = 1.0) -> dict[str, float | bool]:
    normalized: dict[str, float | bool] = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": float(np.clip(gripper, 0.0, 1.0)),
    }
    if action is not None:
        normalized.update(action)
    normalized["enabled"] = bool(normalized["enabled"])
    for key in ACTION_KEYS[1:]:
        normalized[key] = float(normalized[key])
    normalized["gripper"] = float(np.clip(normalized["gripper"], 0.0, 1.0))
    return normalized


def build_default_trace_profile(
    *,
    fps: int,
    step_x: float = 0.0002,
    step_y: float = 0.0002,
    step_z: float = 0.0002,
    warmup_s: float = 0.5,
    move_s: float = 0.75,
    hold_s: float = 0.5,
    settle_s: float = 1.0,
    gripper: float = 1.0,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    actions: list[dict[str, float | bool]] = []
    segments: list[dict[str, Any]] = []

    def add_segment(
        *,
        name: str,
        duration_s: float,
        action: dict[str, Any] | None = None,
        axis: str | None = None,
        sign: int = 0,
        kind: str = "hold",
    ) -> None:
        normalized_action = _normalize_action(action, gripper=gripper)
        step_count = max(int(round(duration_s * fps)), 1)
        start_step = len(actions)
        actions.extend(deepcopy(normalized_action) for _ in range(step_count))
        end_step = len(actions) - 1
        segments.append(
            {
                "name": name,
                "kind": kind,
                "axis": axis,
                "sign": int(sign),
                "start_step": start_step,
                "end_step": end_step,
                "sample_start_index": start_step,
                "sample_end_index": end_step + 1,
                "step_count": step_count,
                "command": deepcopy(normalized_action),
            }
        )

    add_segment(name="warmup_hold", duration_s=warmup_s)
    for axis_name, step_value in (("x", step_x), ("y", step_y), ("z", step_z)):
        add_segment(
            name=f"{axis_name}_plus",
            duration_s=move_s,
            axis=axis_name,
            sign=1,
            kind="translate",
            action={"enabled": True, f"target_{axis_name}": abs(step_value)},
        )
        add_segment(name=f"{axis_name}_hold_after_plus", duration_s=hold_s)
        add_segment(
            name=f"{axis_name}_minus",
            duration_s=move_s,
            axis=axis_name,
            sign=-1,
            kind="translate",
            action={"enabled": True, f"target_{axis_name}": -abs(step_value)},
        )
        add_segment(name=f"{axis_name}_hold_after_minus", duration_s=hold_s)
    add_segment(name="settle_hold", duration_s=settle_s)

    return {
        "name": "default_xyz_translation_pulses",
        "fps": fps,
        "actions": actions,
        "segments": segments,
    }


def build_combined_trace_profile(
    *,
    fps: int,
    step_x: float = 0.0002,
    step_y: float = 0.0002,
    step_z: float = 0.0002,
    step_wx: float = 0.0002,
    step_wy: float = 0.0002,
    step_wz: float = 0.0002,
    warmup_s: float = 0.5,
    move_s: float = 0.75,
    hold_s: float = 0.5,
    settle_s: float = 1.0,
    gripper: float = 1.0,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    actions: list[dict[str, float | bool]] = []
    segments: list[dict[str, Any]] = []
    coupled_axes = (
        ("x", step_x, "wx", step_wx),
        ("y", step_y, "wy", step_wy),
        ("z", step_z, "wz", step_wz),
    )

    def add_segment(
        *,
        name: str,
        duration_s: float,
        action: dict[str, Any] | None = None,
        axis: str | None = None,
        rotation_axis: str | None = None,
        sign: int = 0,
        kind: str = "hold",
    ) -> None:
        normalized_action = _normalize_action(action, gripper=gripper)
        step_count = max(int(round(duration_s * fps)), 1)
        start_step = len(actions)
        actions.extend(deepcopy(normalized_action) for _ in range(step_count))
        end_step = len(actions) - 1
        segments.append(
            {
                "name": name,
                "kind": kind,
                "axis": axis,
                "rotation_axis": rotation_axis,
                "sign": int(sign),
                "start_step": start_step,
                "end_step": end_step,
                "sample_start_index": start_step,
                "sample_end_index": end_step + 1,
                "step_count": step_count,
                "command": deepcopy(normalized_action),
            }
        )

    add_segment(name="warmup_hold", duration_s=warmup_s)
    for axis_name, step_value, rotation_axis_name, rotation_step_value in coupled_axes:
        add_segment(
            name=f"{axis_name}_{rotation_axis_name}_plus",
            duration_s=move_s,
            axis=axis_name,
            rotation_axis=rotation_axis_name,
            sign=1,
            kind="combined",
            action={
                "enabled": True,
                f"target_{axis_name}": abs(step_value),
                f"target_{rotation_axis_name}": abs(rotation_step_value),
            },
        )
        add_segment(name=f"{axis_name}_{rotation_axis_name}_hold_after_plus", duration_s=hold_s)
        add_segment(
            name=f"{axis_name}_{rotation_axis_name}_minus",
            duration_s=move_s,
            axis=axis_name,
            rotation_axis=rotation_axis_name,
            sign=-1,
            kind="combined",
            action={
                "enabled": True,
                f"target_{axis_name}": -abs(step_value),
                f"target_{rotation_axis_name}": -abs(rotation_step_value),
            },
        )
        add_segment(name=f"{axis_name}_{rotation_axis_name}_hold_after_minus", duration_s=hold_s)
    add_segment(name="settle_hold", duration_s=settle_s)

    return {
        "name": "combined_xyz_wxyz_pulses",
        "fps": fps,
        "actions": actions,
        "segments": segments,
    }


def build_wz_trace_profile(
    *,
    fps: int,
    step_wz: float = 0.0002,
    warmup_s: float = 0.5,
    move_s: float = 0.75,
    hold_s: float = 0.5,
    settle_s: float = 1.0,
    gripper: float = 1.0,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    actions: list[dict[str, float | bool]] = []
    segments: list[dict[str, Any]] = []

    def add_segment(
        *,
        name: str,
        duration_s: float,
        action: dict[str, Any] | None = None,
        rotation_axis: str | None = None,
        sign: int = 0,
        kind: str = "hold",
    ) -> None:
        normalized_action = _normalize_action(action, gripper=gripper)
        step_count = max(int(round(duration_s * fps)), 1)
        start_step = len(actions)
        actions.extend(deepcopy(normalized_action) for _ in range(step_count))
        end_step = len(actions) - 1
        segments.append(
            {
                "name": name,
                "kind": kind,
                "axis": None,
                "rotation_axis": rotation_axis,
                "sign": int(sign),
                "start_step": start_step,
                "end_step": end_step,
                "sample_start_index": start_step,
                "sample_end_index": end_step + 1,
                "step_count": step_count,
                "command": deepcopy(normalized_action),
            }
        )

    add_segment(name="warmup_hold", duration_s=warmup_s)
    add_segment(
        name="wz_plus",
        duration_s=move_s,
        rotation_axis="wz",
        sign=1,
        kind="rotate",
        action={"enabled": True, "target_wz": abs(step_wz)},
    )
    add_segment(name="wz_hold_after_plus", duration_s=hold_s)
    add_segment(
        name="wz_minus",
        duration_s=move_s,
        rotation_axis="wz",
        sign=-1,
        kind="rotate",
        action={"enabled": True, "target_wz": -abs(step_wz)},
    )
    add_segment(name="wz_hold_after_minus", duration_s=hold_s)
    add_segment(name="settle_hold", duration_s=settle_s)

    return {
        "name": "wz_rotation_pulses",
        "fps": fps,
        "actions": actions,
        "segments": segments,
    }


def make_trace_sample(
    *,
    profile_step: int,
    scheduled_time_s: float,
    measured_time_s: float,
    action: dict[str, Any] | None,
    joint_positions: np.ndarray,
    ee_position: np.ndarray,
    ee_rotvec: np.ndarray,
    gripper: float,
    target_position: np.ndarray | None = None,
    target_rotvec: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "profile_step": int(profile_step),
        "scheduled_time_s": float(scheduled_time_s),
        "measured_time_s": float(measured_time_s),
        "action": _normalize_action(action, gripper=gripper) if action is not None else None,
        "joint_positions": np.asarray(joint_positions, dtype=np.float64).tolist(),
        "ee_position": np.asarray(ee_position, dtype=np.float64).tolist(),
        "ee_rotvec": np.asarray(ee_rotvec, dtype=np.float64).tolist(),
        "gripper": float(gripper),
    }
    if target_position is not None:
        sample["target_position"] = np.asarray(target_position, dtype=np.float64).tolist()
    if target_rotvec is not None:
        sample["target_rotvec"] = np.asarray(target_rotvec, dtype=np.float64).tolist()
    if extra:
        sample.update(extra)
    return sample


def build_trace_bundle(
    *,
    mode: str,
    profile: dict[str, Any],
    samples: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_metadata = {
        "mode": mode,
        "fps": profile["fps"],
        "profile_name": profile["name"],
        "profile_segments": deepcopy(profile["segments"]),
        "profile_step_count": len(profile["actions"]),
        "sample_count": len(samples),
    }
    if metadata:
        bundle_metadata.update(metadata)
    return {"metadata": bundle_metadata, "samples": deepcopy(samples)}


def save_trace_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")


def load_trace_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_translation_traces(
    reference_trace: dict[str, Any],
    measured_trace: dict[str, Any],
    *,
    minimum_displacement_m: float = 1e-6,
) -> dict[str, Any]:
    ref_segments = reference_trace["metadata"]["profile_segments"]
    measured_segments = measured_trace["metadata"]["profile_segments"]
    if len(ref_segments) != len(measured_segments):
        raise ValueError("Reference and measured traces must use the same segment layout.")

    segment_results: list[dict[str, Any]] = []
    axis_results: dict[str, dict[str, Any]] = {}

    for ref_segment, measured_segment in zip(ref_segments, measured_segments, strict=True):
        if ref_segment["name"] != measured_segment["name"]:
            raise ValueError("Trace segments do not match by name.")
        axis_name = ref_segment["axis"]
        if ref_segment["kind"] != "translate" or axis_name is None:
            continue

        axis_index = AXIS_TO_INDEX[axis_name]
        ref_start = reference_trace["samples"][ref_segment["sample_start_index"]]["ee_position"][axis_index]
        ref_end = reference_trace["samples"][ref_segment["sample_end_index"]]["ee_position"][axis_index]
        measured_start = measured_trace["samples"][measured_segment["sample_start_index"]]["ee_position"][axis_index]
        measured_end = measured_trace["samples"][measured_segment["sample_end_index"]]["ee_position"][axis_index]

        ref_disp = float(ref_end - ref_start)
        measured_disp = float(measured_end - measured_start)
        if abs(measured_disp) < minimum_displacement_m:
            scale_multiplier = None
            status = "measured_too_small"
        elif np.sign(ref_disp) != np.sign(measured_disp):
            scale_multiplier = None
            status = "direction_mismatch"
        else:
            scale_multiplier = float(ref_disp / measured_disp)
            status = "ok"

        result = {
            "name": ref_segment["name"],
            "axis": axis_name,
            "sign": ref_segment["sign"],
            "reference_displacement_m": ref_disp,
            "measured_displacement_m": measured_disp,
            "scale_multiplier": scale_multiplier,
            "status": status,
        }
        segment_results.append(result)

        axis_summary = axis_results.setdefault(
            axis_name,
            {
                "axis": axis_name,
                "reference_total_displacement_m": 0.0,
                "measured_total_displacement_m": 0.0,
                "segment_count": 0,
                "valid_segment_count": 0,
                "segment_scale_multipliers": [],
            },
        )
        axis_summary["reference_total_displacement_m"] += abs(ref_disp)
        axis_summary["measured_total_displacement_m"] += abs(measured_disp)
        axis_summary["segment_count"] += 1
        if scale_multiplier is not None:
            axis_summary["valid_segment_count"] += 1
            axis_summary["segment_scale_multipliers"].append(scale_multiplier)

    for axis_name, axis_summary in axis_results.items():
        multipliers = axis_summary["segment_scale_multipliers"]
        axis_summary["suggested_scale_multiplier"] = (
            float(np.mean(multipliers)) if multipliers else None
        )
        axis_summary["reference_total_displacement_m"] = float(axis_summary["reference_total_displacement_m"])
        axis_summary["measured_total_displacement_m"] = float(axis_summary["measured_total_displacement_m"])
        axis_summary["segment_scale_multipliers"] = [float(value) for value in multipliers]
        axis_summary["status"] = "ok" if multipliers else "insufficient_signal"

    return {
        "reference_mode": reference_trace["metadata"].get("mode"),
        "measured_mode": measured_trace["metadata"].get("mode"),
        "axis_summaries": axis_results,
        "segment_summaries": segment_results,
    }


def compare_pose_traces(
    reference_trace: dict[str, Any],
    measured_trace: dict[str, Any],
    *,
    minimum_translation_displacement_m: float = 1e-6,
    minimum_rotation_displacement_rad: float = 1e-6,
) -> dict[str, Any]:
    ref_segments = reference_trace["metadata"]["profile_segments"]
    measured_segments = measured_trace["metadata"]["profile_segments"]
    if len(ref_segments) != len(measured_segments):
        raise ValueError("Reference and measured traces must use the same segment layout.")

    translation_segment_results: list[dict[str, Any]] = []
    rotation_segment_results: list[dict[str, Any]] = []
    translation_axis_results: dict[str, dict[str, Any]] = {}
    rotation_axis_results: dict[str, dict[str, Any]] = {}

    for ref_segment, measured_segment in zip(ref_segments, measured_segments, strict=True):
        if ref_segment["name"] != measured_segment["name"]:
            raise ValueError("Trace segments do not match by name.")

        if ref_segment["kind"] in ("translate", "combined") and ref_segment.get("axis") is not None:
            axis_name = ref_segment["axis"]
            axis_index = AXIS_TO_INDEX[axis_name]
            ref_start = reference_trace["samples"][ref_segment["sample_start_index"]]["ee_position"][axis_index]
            ref_end = reference_trace["samples"][ref_segment["sample_end_index"]]["ee_position"][axis_index]
            measured_start = measured_trace["samples"][measured_segment["sample_start_index"]]["ee_position"][axis_index]
            measured_end = measured_trace["samples"][measured_segment["sample_end_index"]]["ee_position"][axis_index]

            ref_disp = float(ref_end - ref_start)
            measured_disp = float(measured_end - measured_start)
            if abs(measured_disp) < minimum_translation_displacement_m:
                scale_multiplier = None
                status = "measured_too_small"
            elif np.sign(ref_disp) != np.sign(measured_disp):
                scale_multiplier = None
                status = "direction_mismatch"
            else:
                scale_multiplier = float(ref_disp / measured_disp)
                status = "ok"

            result = {
                "name": ref_segment["name"],
                "axis": axis_name,
                "sign": ref_segment["sign"],
                "reference_displacement_m": ref_disp,
                "measured_displacement_m": measured_disp,
                "scale_multiplier": scale_multiplier,
                "status": status,
            }
            translation_segment_results.append(result)

            axis_summary = translation_axis_results.setdefault(
                axis_name,
                {
                    "axis": axis_name,
                    "reference_total_displacement_m": 0.0,
                    "measured_total_displacement_m": 0.0,
                    "segment_count": 0,
                    "valid_segment_count": 0,
                    "segment_scale_multipliers": [],
                },
            )
            axis_summary["reference_total_displacement_m"] += abs(ref_disp)
            axis_summary["measured_total_displacement_m"] += abs(measured_disp)
            axis_summary["segment_count"] += 1
            if scale_multiplier is not None:
                axis_summary["valid_segment_count"] += 1
                axis_summary["segment_scale_multipliers"].append(scale_multiplier)

        if ref_segment["kind"] in ("combined", "rotate") and ref_segment.get("rotation_axis") is not None:
            axis_name = ref_segment["rotation_axis"]
            ref_start_rot = np.asarray(
                reference_trace["samples"][ref_segment["sample_start_index"]]["ee_rotvec"],
                dtype=np.float64,
            )
            ref_end_rot = np.asarray(
                reference_trace["samples"][ref_segment["sample_end_index"]]["ee_rotvec"],
                dtype=np.float64,
            )
            measured_start_rot = np.asarray(
                measured_trace["samples"][measured_segment["sample_start_index"]]["ee_rotvec"],
                dtype=np.float64,
            )
            measured_end_rot = np.asarray(
                measured_trace["samples"][measured_segment["sample_end_index"]]["ee_rotvec"],
                dtype=np.float64,
            )

            ref_relative_rot = Rotation.from_rotvec(ref_end_rot) * Rotation.from_rotvec(ref_start_rot).inv()
            measured_relative_rot = Rotation.from_rotvec(measured_end_rot) * Rotation.from_rotvec(measured_start_rot).inv()
            ref_disp = float(np.linalg.norm(ref_relative_rot.as_rotvec()))
            measured_disp = float(np.linalg.norm(measured_relative_rot.as_rotvec()))
            if measured_disp < minimum_rotation_displacement_rad:
                scale_multiplier = None
                status = "measured_too_small"
            else:
                scale_multiplier = float(ref_disp / measured_disp)
                status = "ok"

            result = {
                "name": ref_segment["name"],
                "axis": axis_name,
                "sign": ref_segment["sign"],
                "reference_displacement_rad": ref_disp,
                "measured_displacement_rad": measured_disp,
                "scale_multiplier": scale_multiplier,
                "status": status,
            }
            rotation_segment_results.append(result)

            axis_summary = rotation_axis_results.setdefault(
                axis_name,
                {
                    "axis": axis_name,
                    "reference_total_displacement_rad": 0.0,
                    "measured_total_displacement_rad": 0.0,
                    "segment_count": 0,
                    "valid_segment_count": 0,
                    "segment_scale_multipliers": [],
                },
            )
            axis_summary["reference_total_displacement_rad"] += abs(ref_disp)
            axis_summary["measured_total_displacement_rad"] += abs(measured_disp)
            axis_summary["segment_count"] += 1
            if scale_multiplier is not None:
                axis_summary["valid_segment_count"] += 1
                axis_summary["segment_scale_multipliers"].append(scale_multiplier)

    for axis_summary in translation_axis_results.values():
        multipliers = axis_summary["segment_scale_multipliers"]
        axis_summary["suggested_scale_multiplier"] = (
            float(np.mean(multipliers)) if multipliers else None
        )
        axis_summary["reference_total_displacement_m"] = float(axis_summary["reference_total_displacement_m"])
        axis_summary["measured_total_displacement_m"] = float(axis_summary["measured_total_displacement_m"])
        axis_summary["segment_scale_multipliers"] = [float(value) for value in multipliers]
        axis_summary["status"] = "ok" if multipliers else "insufficient_signal"

    for axis_summary in rotation_axis_results.values():
        multipliers = axis_summary["segment_scale_multipliers"]
        axis_summary["suggested_scale_multiplier"] = (
            float(np.mean(multipliers)) if multipliers else None
        )
        axis_summary["reference_total_displacement_rad"] = float(axis_summary["reference_total_displacement_rad"])
        axis_summary["measured_total_displacement_rad"] = float(axis_summary["measured_total_displacement_rad"])
        axis_summary["segment_scale_multipliers"] = [float(value) for value in multipliers]
        axis_summary["status"] = "ok" if multipliers else "insufficient_signal"

    return {
        "reference_mode": reference_trace["metadata"].get("mode"),
        "measured_mode": measured_trace["metadata"].get("mode"),
        "translation_axis_summaries": translation_axis_results,
        "translation_segment_summaries": translation_segment_results,
        "rotation_axis_summaries": rotation_axis_results,
        "rotation_segment_summaries": rotation_segment_results,
    }
