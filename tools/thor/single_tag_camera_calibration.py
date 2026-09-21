"""Fixed-camera extrinsics from one rigid EE AprilTag and existing intrinsics.

The capture process supplies synchronized tag corners and measured FR3 TCP
poses. Transform notation is ``T_a_b``: coordinates in ``b`` mapped into
``a``. Every accepted observation obeys::

    T_base_tcp(i) @ T_tcp_tag = T_base_camera @ T_camera_tag(i)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from third_party.opencv_kalibr.realsense_base_extrinsics_calibration.calibrate_fixed_cameras_in_base_from_moving_charuco import (
    average_transform_least_squares,
    build_relative_pairs,
    estimate_fixed_camera_in_base,
    estimate_joint_fixed_cameras_in_base,
    invert_transform,
    summarize_residuals,
    transform_residual,
    transform_to_payload,
)


FISHEYE_MODELS = frozenset({"fisheye", "equidistant", "opencv_fisheye"})


def tag_object_points(marker_size_m: float) -> np.ndarray:
    half = float(marker_size_m) / 2.0
    return np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _camera_views(records: list[dict[str, Any]], camera: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for record in records:
        for live_camera, item in record.get("cameras", {}).items():
            if str(item.get("calibration_camera", live_camera)) != camera:
                continue
            detections = [
                detection
                for detection in item.get("detections", [])
                if int(detection.get("tag_id", -1)) == 6
            ]
            if not detections:
                continue
            detection = detections[0]
            corners = np.asarray(detection.get("corners_px", []), dtype=np.float64).reshape(-1, 2)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                continue
            width = int(detection.get("image_width", 0))
            height = int(detection.get("image_height", 0))
            if width <= 0 or height <= 0:
                continue
            views.append(
                {
                    "capture_index": int(record["capture_index"]),
                    "corners": corners,
                    "width": width,
                    "height": height,
                    "T_base_tcp": _matrix(record["T_base_tcp"], "T_base_tcp"),
                }
            )
    views.sort(key=lambda item: item["capture_index"])
    return views


def _resolve_intrinsics_path(summary_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(summary_path.parent / path)
        if "outputs" in summary_path.parts:
            outputs_index = summary_path.parts.index("outputs")
            repo_root = Path(*summary_path.parts[:outputs_index])
            candidates.append(repo_root / path)
    if "per_camera" in path.parts:
        candidates.append(summary_path.parent.joinpath(*path.parts[path.parts.index("per_camera") :]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"intrinsics file referenced by {summary_path} does not exist: {raw_path}")


def load_existing_fisheye_intrinsics(summary_path: Path) -> dict[str, dict[str, Any]]:
    summary_path = summary_path.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("cameras", []):
        if not isinstance(row, dict) or str(row.get("status", "")).lower() != "ok":
            continue
        camera = str(row.get("camera_name", "")).strip()
        path = _resolve_intrinsics_path(summary_path, str(row.get("intrinsics_json", "")))
        data = json.loads(path.read_text(encoding="utf-8"))
        model = str(data.get("model", "")).lower()
        if model not in FISHEYE_MODELS:
            raise ValueError(
                f"{camera}: expected existing fisheye intrinsics, but {path} declares model={model!r}"
            )
        K = np.asarray(data.get("camera_matrix", []), dtype=np.float64)
        D = np.asarray(data.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
        width = int(data.get("image_width", data.get("width", 0)))
        height = int(data.get("image_height", data.get("height", 0)))
        if K.shape != (3, 3) or D.size != 4 or width <= 0 or height <= 0:
            raise ValueError(f"{camera}: invalid OpenCV fisheye intrinsics in {path}")
        result[camera] = {
            "camera": camera,
            "model": model,
            "K": K,
            "D": D.reshape(4, 1),
            "width": width,
            "height": height,
            "intrinsics_path": str(path),
        }
    if not result:
        raise RuntimeError(f"no usable fisheye intrinsics in {summary_path}")
    return result


def _fit_for_views(intrinsics: dict[str, Any], views: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = {(view["width"], view["height"]) for view in views}
    if len(sizes) != 1:
        raise RuntimeError(f"inconsistent image sizes in captures: {sorted(sizes)}")
    width, height = next(iter(sizes))
    source_width = int(intrinsics["width"])
    source_height = int(intrinsics["height"])
    source_aspect = source_width / source_height
    capture_aspect = width / height
    if abs(source_aspect - capture_aspect) > 1e-3:
        raise RuntimeError(
            f"capture size {width}x{height} does not match intrinsics aspect "
            f"{source_width}x{source_height}"
        )
    K = np.asarray(intrinsics["K"], dtype=np.float64).copy()
    K[0, :] *= width / source_width
    K[1, :] *= height / source_height
    K[2, :] = (0.0, 0.0, 1.0)
    return {**intrinsics, "K": K, "width": width, "height": height, "views": views}


def fisheye_tag_pose(
    corners_px: np.ndarray, fit: dict[str, Any], marker_size_m: float
) -> tuple[np.ndarray, float]:
    points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    K = np.asarray(fit["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(fit["D"], dtype=np.float64).reshape(4, 1)
    normalized = cv2.fisheye.undistortPoints(points.reshape(-1, 1, 2), K, D).reshape(4, 2)
    obj = tag_object_points(marker_size_m)
    ok, rvec, tvec = cv2.solvePnP(
        obj, normalized, np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0:
        raise RuntimeError("fisheye tag solvePnP failed")
    projected, _ = cv2.fisheye.projectPoints(obj.reshape(1, 4, 3), rvec, tvec, K, D)
    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    T_camera_tag = np.eye(4, dtype=np.float64)
    T_camera_tag[:3, :3] = rotation
    T_camera_tag[:3, 3] = np.asarray(tvec).reshape(3)
    return T_camera_tag, rmse


def _filter_joint_observation_outliers(
    observations_by_camera: dict[str, list[dict[str, Any]]],
    T_tcp_tag: np.ndarray,
    camera_poses: dict[str, np.ndarray],
    *,
    max_rotation_deg: float,
    max_translation_m: float,
    min_samples: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Trim observations inconsistent with the shared rigid tag/camera model.

    Reprojection error alone cannot reject planar-PnP pose flips: a wrong pose
    can still explain four tag corners with a small pixel error, especially near
    a fronto-parallel view.  The robot supplies an independent constraint, so
    trim only after a robust initial joint solve and then solve once more.
    """
    filtered: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for camera, samples in sorted(observations_by_camera.items()):
        T_base_camera = camera_poses[camera]
        kept = []
        rejected = []
        for sample in samples:
            predicted = sample["T_b_tool"] @ T_tcp_tag
            observed = T_base_camera @ sample["T_c_board"]
            rotation_deg, translation_m = transform_residual(predicted, observed)
            reasons = []
            if rotation_deg > max_rotation_deg:
                reasons.append("rotation")
            if translation_m > max_translation_m:
                reasons.append("translation")
            if reasons:
                rejected.append(
                    {
                        "capture_index": int(sample["frame_index"]),
                        "reprojection_rmse_px": float(sample["reprojection_rmse_px"]),
                        "rotation_residual_deg": float(rotation_deg),
                        "translation_residual_m": float(translation_m),
                        "reasons": reasons,
                    }
                )
            else:
                kept.append(sample)
        if len(kept) < min_samples:
            raise RuntimeError(
                f"{camera}: robust filtering leaves {len(kept)} observations; "
                f"need {min_samples} (input={len(samples)}, rejected={len(rejected)})"
            )
        filtered[camera] = kept
        reports[camera] = {
            "num_input_observations": len(samples),
            "num_kept_observations": len(kept),
            "num_rejected_outliers": len(rejected),
            "rejected_outliers": rejected,
        }
    return filtered, reports


def solve_fixed_camera_extrinsics(
    fits: dict[str, dict[str, Any]],
    *,
    marker_size_m: float = 0.16,
    max_tag_rmse_px: float = 2.0,
    min_samples: int = 20,
    max_nfev: int = 1200,
    max_joint_rotation_residual_deg: float = 3.0,
    max_joint_translation_residual_m: float = 0.020,
) -> dict[str, Any]:
    observations_by_camera: dict[str, list[dict[str, Any]]] = {}
    initial_camera_poses: dict[str, np.ndarray] = {}
    tool_tag_candidates: list[np.ndarray] = []
    camera_details: dict[str, Any] = {}
    for camera, fit in sorted(fits.items()):
        samples = []
        for view in fit["views"]:
            try:
                T_camera_tag, rmse = fisheye_tag_pose(view["corners"], fit, marker_size_m)
            except (RuntimeError, cv2.error):
                continue
            if rmse > max_tag_rmse_px:
                continue
            samples.append(
                {
                    "frame_index": int(view["capture_index"]),
                    "T_b_tool": view["T_base_tcp"],
                    "T_c_board": T_camera_tag,
                    "reprojection_rmse_px": rmse,
                }
            )
        pairs = build_relative_pairs(samples, [1, 3, 10, 20], 5.0, 0.01)
        if len(samples) < min_samples or len(pairs) < min_samples:
            raise RuntimeError(
                f"{camera}: insufficient extrinsics data: samples={len(samples)}, pairs={len(pairs)}, "
                f"need {min_samples} each"
            )
        T_base_camera, solver = estimate_fixed_camera_in_base(pairs, max_nfev=max_nfev)
        candidates = [
            invert_transform(sample["T_b_tool"]) @ T_base_camera @ sample["T_c_board"]
            for sample in samples
        ]
        observations_by_camera[camera] = samples
        initial_camera_poses[camera] = T_base_camera
        tool_tag_candidates.extend(candidates)
        camera_details[camera] = {
            "num_samples": len(samples),
            "num_motion_pairs": len(pairs),
            "initial_solver": solver,
            "intrinsics_source_camera": fit["intrinsics_source_camera"],
            "intrinsics_path": fit["intrinsics_path"],
            "temporary_intrinsics_reuse": bool(fit["temporary_intrinsics_reuse"]),
        }

    if len(initial_camera_poses) < 2:
        raise RuntimeError("need at least two cameras for joint fixed-camera calibration")
    initial_tool_tag = average_transform_least_squares(tool_tag_candidates)
    T_tcp_tag, camera_poses, initial_joint_solver = estimate_joint_fixed_cameras_in_base(
        observations_by_camera, initial_tool_tag, initial_camera_poses, max_nfev=max_nfev
    )
    filtered_observations, outlier_reports = _filter_joint_observation_outliers(
        observations_by_camera,
        T_tcp_tag,
        camera_poses,
        max_rotation_deg=max_joint_rotation_residual_deg,
        max_translation_m=max_joint_translation_residual_m,
        min_samples=min_samples,
    )
    num_rejected = sum(
        report["num_rejected_outliers"] for report in outlier_reports.values()
    )
    if num_rejected:
        T_tcp_tag, camera_poses, solver = estimate_joint_fixed_cameras_in_base(
            filtered_observations,
            T_tcp_tag,
            camera_poses,
            max_nfev=max_nfev,
        )
    else:
        solver = dict(initial_joint_solver)
    solver["robust_refinement"] = {
        "applied": bool(num_rejected),
        "max_rotation_residual_deg": float(max_joint_rotation_residual_deg),
        "max_translation_residual_m": float(max_joint_translation_residual_m),
        "num_input_observations": int(
            sum(len(samples) for samples in observations_by_camera.values())
        ),
        "num_kept_observations": int(
            sum(len(samples) for samples in filtered_observations.values())
        ),
        "num_rejected_outliers": int(num_rejected),
        "initial_solver": dict(initial_joint_solver),
    }
    all_residuals = []
    for camera, T_base_camera in camera_poses.items():
        residuals = []
        for sample in filtered_observations[camera]:
            predicted = sample["T_b_tool"] @ T_tcp_tag
            observed = T_base_camera @ sample["T_c_board"]
            residual = transform_residual(predicted, observed)
            residuals.append(residual)
            all_residuals.append(residual)
        final_pairs = build_relative_pairs(
            filtered_observations[camera],
            [1, 3, 10, 20],
            5.0,
            0.01,
        )
        if len(final_pairs) < min_samples:
            raise RuntimeError(
                f"{camera}: robust filtering leaves {len(final_pairs)} motion pairs; "
                f"need {min_samples}"
            )
        camera_details[camera]["base_to_camera"] = transform_to_payload(T_base_camera)
        camera_details[camera]["num_input_samples"] = outlier_reports[camera][
            "num_input_observations"
        ]
        camera_details[camera]["num_samples"] = outlier_reports[camera][
            "num_kept_observations"
        ]
        camera_details[camera]["num_motion_pairs"] = len(final_pairs)
        camera_details[camera]["num_rejected_outliers"] = outlier_reports[camera][
            "num_rejected_outliers"
        ]
        camera_details[camera]["rejected_outliers"] = outlier_reports[camera][
            "rejected_outliers"
        ]
        camera_details[camera]["sample_residuals"] = summarize_residuals(residuals)
    residual_summary = summarize_residuals(all_residuals)
    reasons = []
    if not solver.get("success"):
        reasons.append(f"joint optimizer failed: {solver.get('message', '')}")
    if float(residual_summary["translation_m_mean"] or 0.0) > 0.010:
        reasons.append("joint mean translation residual exceeds 10 mm")
    if float(residual_summary["rotation_deg_mean"] or 0.0) > 1.0:
        reasons.append("joint mean rotation residual exceeds 1 degree")
    return {
        "status": "passed" if not reasons else "failed_quality_gate",
        "quality_gate_reasons": reasons,
        "frame_equation": "T_base_tcp(i) @ T_tcp_tag = T_base_camera @ T_camera_tag(i)",
        "tool_to_board": transform_to_payload(T_tcp_tag),
        "cameras": camera_details,
        "solver": solver,
        "robust_filter": solver["robust_refinement"],
        "sample_residuals": residual_summary,
    }


def calibrate_and_write(
    records: list[dict[str, Any]],
    output_dir: Path,
    intrinsics_summary_path: Path,
    *,
    marker_size_m: float = 0.16,
    min_frames: int = 20,
    intrinsics_fallbacks: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    intrinsics_fallbacks = dict(intrinsics_fallbacks or {"cam_03": "cam_13"})
    existing = load_existing_fisheye_intrinsics(intrinsics_summary_path)
    cameras = sorted(
        {
            str(item.get("calibration_camera", live_camera))
            for record in records
            for live_camera, item in record.get("cameras", {}).items()
            if any(int(det.get("tag_id", -1)) == 6 for det in item.get("detections", []))
        }
    )
    if len(cameras) < 2:
        raise RuntimeError("fewer than two cameras detected tag 6")
    fits: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        source_camera = intrinsics_fallbacks.get(camera, camera)
        if source_camera not in existing:
            raise RuntimeError(f"{camera}: no existing fisheye intrinsics for source {source_camera}")
        views = _camera_views(records, camera)
        if len(views) < min_frames:
            raise RuntimeError(f"{camera}: only {len(views)} tag views; need {min_frames}")
        fits[camera] = _fit_for_views(existing[source_camera], views)
        fits[camera].update(
            {
                "camera": camera,
                "intrinsics_source_camera": source_camera,
                "temporary_intrinsics_reuse": source_camera != camera,
            }
        )

    extrinsics = solve_fixed_camera_extrinsics(
        fits, marker_size_m=marker_size_m, min_samples=min_frames
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    payload = {
        "schema": "thor_fixed_cameras_in_current_fr3_base/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": extrinsics["status"],
        "world": {
            "world_frame_id": "fr3_base",
            "definition": "current FR3 base frame at calibration time",
        },
        "marker": {"family": "tag36h11", "id": 6, "marker_size_m": marker_size_m},
        "intrinsics": {
            "mode": "existing_fisheye",
            "summary": str(intrinsics_summary_path.expanduser().resolve()),
            "fallbacks": intrinsics_fallbacks,
        },
        "joint_solution": {
            "status": "ok" if extrinsics["status"] == "passed" else extrinsics["status"],
            "frame_equation": extrinsics["frame_equation"],
            "tool_to_board": extrinsics["tool_to_board"],
            "solver": extrinsics["solver"],
            "robust_filter": extrinsics["robust_filter"],
            "sample_residuals": extrinsics["sample_residuals"],
            "cameras": {
                camera: {
                    "base_to_camera": detail["base_to_camera"],
                    "num_observations": detail["num_samples"],
                    "num_input_observations": detail["num_input_samples"],
                    "num_rejected_outliers": detail["num_rejected_outliers"],
                    "rejected_outliers": detail["rejected_outliers"],
                    "num_motion_pairs": detail["num_motion_pairs"],
                    "sample_residuals": detail["sample_residuals"],
                    "intrinsics_source_camera": detail["intrinsics_source_camera"],
                    "intrinsics_path": detail["intrinsics_path"],
                    "temporary_intrinsics_reuse": detail["temporary_intrinsics_reuse"],
                }
                for camera, detail in extrinsics["cameras"].items()
            },
        },
        "quality_gate_reasons": extrinsics["quality_gate_reasons"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if extrinsics["status"] != "passed":
        raise RuntimeError(
            "fixed-camera extrinsics quality gate failed: "
            + "; ".join(extrinsics["quality_gate_reasons"])
        )
    return output_path, payload
