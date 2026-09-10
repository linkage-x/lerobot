"""Pure geometry for the P1 simple eye-hand calibration.

The hardware runner deliberately lives outside this module.  Keeping the
solver here makes the frame convention and the acceptance test independently
testable without importing camera or robot SDKs.

Notation: ``T_a_b`` maps coordinates expressed in frame ``b`` into frame
``a``.  Every observation obeys

    T_world_tag = T_world_base @ T_base_tcp @ T_tcp_tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EyeHandObservation:
    pose_index: int
    camera: str
    tag_id: int
    T_base_tcp: np.ndarray
    T_world_tag: np.ndarray
    reprojection_rmse_px: float = 0.0


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -(out[:3, :3] @ T[:3, 3])
    return out


def transform_from_vec6(value: Sequence[float]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix()
    T[:3, 3] = value[3:]
    return T


def vec6_from_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def transform_error(predicted: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = invert_transform(predicted) @ observed
    return delta[:3, 3], Rotation.from_matrix(delta[:3, :3]).as_rotvec()


def _initial_tag_pose(observations: Sequence[EyeHandObservation], tag_id: int) -> np.ndarray:
    candidates = [
        invert_transform(obs.T_base_tcp) @ obs.T_world_tag
        for obs in observations
        if obs.tag_id == tag_id
    ]
    if not candidates:
        raise ValueError(f"No observations for tag {tag_id}")
    translations = np.asarray([T[:3, 3] for T in candidates], dtype=np.float64)
    rotations = Rotation.from_matrix(np.asarray([T[:3, :3] for T in candidates]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotations.mean().as_matrix()
    T[:3, 3] = np.median(translations, axis=0)
    return T


def solve_eye_hand(
    observations: Sequence[EyeHandObservation],
    *,
    tag_ids: Sequence[int] = (56, 57),
    initial_T_world_base: np.ndarray | None = None,
    translation_scale_m: float = 0.002,
    rotation_scale_deg: float = 0.2,
    max_nfev: int = 1200,
) -> dict:
    """Jointly solve one base pose and one rigid TCP pose per AprilTag.

    A robust loss is applied to camera/tag observations, so one poor camera
    detection cannot pull the base by an arbitrary amount.  The returned
    residuals remain in physical units for operator-facing quality gates.
    """
    observations = list(observations)
    tag_ids = tuple(int(v) for v in tag_ids)
    if len(observations) < 6:
        raise ValueError("At least 6 camera/tag observations are required")
    pose_count = len({obs.pose_index for obs in observations})
    if pose_count < 3:
        raise ValueError("At least 3 distinct robot poses are required")
    seen = {obs.tag_id for obs in observations}
    missing = [tag_id for tag_id in tag_ids if tag_id not in seen]
    if missing:
        raise ValueError(f"Missing observations for tag ids: {missing}")

    T_world_base0 = np.eye(4) if initial_T_world_base is None else np.asarray(initial_T_world_base, dtype=float)
    # Estimate tool/tag in the initial base frame.  This initialization remains
    # close when the robot was translated or modestly rotated after P0 capture.
    in_initial_base: list[EyeHandObservation] = []
    T_base_world0 = invert_transform(T_world_base0)
    for obs in observations:
        in_initial_base.append(
            EyeHandObservation(
                pose_index=obs.pose_index,
                camera=obs.camera,
                tag_id=obs.tag_id,
                T_base_tcp=obs.T_base_tcp,
                T_world_tag=T_base_world0 @ obs.T_world_tag,
                reprojection_rmse_px=obs.reprojection_rmse_px,
            )
        )
    x0 = [vec6_from_transform(T_world_base0)]
    x0.extend(vec6_from_transform(_initial_tag_pose(in_initial_base, tag_id)) for tag_id in tag_ids)
    x0_arr = np.concatenate(x0)
    tag_offset = {tag_id: 6 + 6 * i for i, tag_id in enumerate(tag_ids)}
    rot_scale = np.deg2rad(float(rotation_scale_deg))

    def unpack(x: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        base = transform_from_vec6(x[:6])
        tags = {tag_id: transform_from_vec6(x[offset : offset + 6]) for tag_id, offset in tag_offset.items()}
        return base, tags

    def residual(x: np.ndarray, physical: bool = False) -> np.ndarray:
        base, tags = unpack(x)
        rows: list[np.ndarray] = []
        for obs in observations:
            pred = base @ obs.T_base_tcp @ tags[obs.tag_id]
            trans, rot = transform_error(pred, obs.T_world_tag)
            if physical:
                rows.append(np.r_[trans, rot])
            else:
                rows.append(np.r_[trans / translation_scale_m, rot / rot_scale])
        return np.concatenate(rows)

    fit = least_squares(residual, x0_arr, loss="huber", f_scale=1.0, max_nfev=int(max_nfev))
    T_world_base, T_tcp_tags = unpack(fit.x)
    physical = residual(fit.x, physical=True).reshape(-1, 6)
    trans_mm = np.linalg.norm(physical[:, :3], axis=1) * 1000.0
    rot_deg = np.linalg.norm(physical[:, 3:], axis=1) * 180.0 / np.pi
    return {
        "T_world_base": T_world_base,
        "T_tcp_tags": T_tcp_tags,
        "num_observations": len(observations),
        "num_robot_poses": pose_count,
        "num_cameras": len({obs.camera for obs in observations}),
        "observations_per_tag": {str(tag_id): sum(obs.tag_id == tag_id for obs in observations) for tag_id in tag_ids},
        "robot_poses_per_tag": {
            str(tag_id): len({obs.pose_index for obs in observations if obs.tag_id == tag_id}) for tag_id in tag_ids
        },
        "translation_residual_mm": {
            "rms": float(np.sqrt(np.mean(np.square(trans_mm)))),
            "median": float(np.median(trans_mm)),
            "p95": float(np.percentile(trans_mm, 95)),
            "max": float(np.max(trans_mm)),
        },
        "rotation_residual_deg": {
            "rms": float(np.sqrt(np.mean(np.square(rot_deg)))),
            "median": float(np.median(rot_deg)),
            "p95": float(np.percentile(rot_deg, 95)),
            "max": float(np.max(rot_deg)),
        },
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "cost": float(fit.cost),
            "nfev": int(fit.nfev),
        },
    }


def assess_solution(
    solution: Mapping[str, object],
    *,
    min_robot_poses: int = 15,
    min_cameras: int = 2,
    min_robot_poses_per_tag: int = 8,
    max_translation_rms_mm: float = 5.0,
    max_rotation_rms_deg: float = 2.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(solution["num_robot_poses"]) < min_robot_poses:
        reasons.append(f"robot poses {solution['num_robot_poses']} < {min_robot_poses}")
    if int(solution["num_cameras"]) < min_cameras:
        reasons.append(f"cameras {solution['num_cameras']} < {min_cameras}")
    for tag_id, count in solution.get("robot_poses_per_tag", {}).items():  # type: ignore[union-attr]
        if int(count) < min_robot_poses_per_tag:
            reasons.append(f"tag {tag_id} robot poses {count} < {min_robot_poses_per_tag}")
    trans = float(solution["translation_residual_mm"]["rms"])  # type: ignore[index]
    rot = float(solution["rotation_residual_deg"]["rms"])  # type: ignore[index]
    if trans > max_translation_rms_mm:
        reasons.append(f"translation RMS {trans:.3f} mm > {max_translation_rms_mm:.3f} mm")
    if rot > max_rotation_rms_deg:
        reasons.append(f"rotation RMS {rot:.3f} deg > {max_rotation_rms_deg:.3f} deg")
    if not bool(solution["optimizer"]["success"]):  # type: ignore[index]
        reasons.append("optimizer did not converge")
    return not reasons, reasons


def select_diverse_pose_records(records: Sequence[dict], count: int) -> list[dict]:
    """Deterministic farthest-point selection over TCP translation/orientation.

    Existing AprilTag teach poses are safer operator knowledge than newly
    invented Cartesian targets.  This only chooses a diverse subset; it does
    not modify or extrapolate a taught pose.
    """
    rows = list(records)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    features: list[np.ndarray] = []
    for index, row in enumerate(rows):
        pose = row.get("pose", {}) if isinstance(row, dict) else {}
        xyz = np.asarray(pose.get("position_xyz_m", []), dtype=float).reshape(-1)
        rot = np.asarray(pose.get("rotvec_xyz_rad", []), dtype=float).reshape(-1)
        if xyz.size == 3 and rot.size == 3 and np.isfinite(np.r_[xyz, rot]).all():
            features.append(np.r_[xyz / 0.10, rot / np.deg2rad(30.0)])
        else:
            joints = np.asarray(row.get("joint_values_rad", []), dtype=float).reshape(-1)
            if joints.size != 7:
                features.append(np.array([float(index)]))
            else:
                features.append(joints / 0.5)
    width = max(v.size for v in features)
    if any(v.size != width for v in features):
        # Mixed legacy schemas: evenly spaced is deterministic and preserves
        # the exact taught targets.
        idx = np.linspace(0, len(rows) - 1, count).round().astype(int)
        return [rows[int(i)] for i in idx]
    X = np.asarray(features)
    selected = [0, len(rows) - 1]
    distances = np.minimum(
        np.linalg.norm(X - X[selected[0]], axis=1),
        np.linalg.norm(X - X[selected[1]], axis=1),
    )
    while len(selected) < count:
        distances[selected] = -1.0
        nxt = int(np.argmax(distances))
        selected.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(X - X[nxt], axis=1))
    # Diversity decides membership only.  Execution keeps the original taught
    # order so the robot follows the operator's scan rather than jumping in
    # farthest-point order across the workspace.
    return [rows[i] for i in sorted(selected)]


def matrix_payload(T: np.ndarray) -> list[list[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()
