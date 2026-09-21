#!/usr/bin/env python3
"""Offline per-observation diagnostics for P0 single-tag calibration captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from third_party.opencv_kalibr.realsense_base_extrinsics_calibration.calibrate_fixed_cameras_in_base_from_moving_charuco import (
    transform_residual,
)
from tools.thor.single_tag_camera_calibration import (
    _camera_views,
    _fit_for_views,
    fisheye_tag_pose,
    load_existing_fisheye_intrinsics,
    solve_fixed_camera_extrinsics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--intrinsics-summary", type=Path, required=True)
    parser.add_argument("--trim-rotation-deg", type=float, default=0.0)
    parser.add_argument("--trim-translation-m", type=float, default=0.0)
    args = parser.parse_args()

    records = json.loads(args.captures.read_text(encoding="utf-8"))["records"]
    existing = load_existing_fisheye_intrinsics(args.intrinsics_summary)
    cameras = sorted(
        {
            str(item.get("calibration_camera", live_camera))
            for record in records
            for live_camera, item in record["cameras"].items()
            if item.get("detections")
        }
    )
    fits = {}
    rows_by_camera = {}
    for camera in cameras:
        source_camera = "cam_13" if camera == "cam_03" else camera
        views = _camera_views(records, camera)
        fit = _fit_for_views(existing[source_camera], views)
        fit.update(
            camera=camera,
            intrinsics_source_camera=source_camera,
            temporary_intrinsics_reuse=source_camera != camera,
        )
        fits[camera] = fit
        rows = []
        for view in views:
            T_camera_tag, rmse = fisheye_tag_pose(view["corners"], fit, 0.16)
            points = view["corners"]
            area = abs(cv2.contourArea(points.astype(np.float32)))
            edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            ray = T_camera_tag[:3, 3] / np.linalg.norm(T_camera_tag[:3, 3])
            incidence_deg = math.degrees(
                math.acos(np.clip(abs(float(T_camera_tag[:3, 2] @ ray)), 0.0, 1.0))
            )
            image_size = np.array([view["width"], view["height"]], dtype=np.float64)
            radius = float(np.linalg.norm((points.mean(axis=0) - image_size / 2.0) / (image_size / 2.0)))
            rows.append(
                {
                    "index": view["capture_index"],
                    "rmse": rmse,
                    "area": area,
                    "min_edge": float(edges.min()),
                    "incidence": incidence_deg,
                    "radius": radius,
                    "T_camera_tag": T_camera_tag,
                    "T_base_tcp": view["T_base_tcp"],
                    "view": view,
                }
            )
        rows_by_camera[camera] = rows

    result = solve_fixed_camera_extrinsics(fits, marker_size_m=0.16, min_samples=20)
    if args.trim_rotation_deg > 0.0 or args.trim_translation_m > 0.0:
        T_tcp_tag = np.asarray(result["tool_to_board"]["matrix_4x4"])
        trimmed_fits = {}
        for camera in cameras:
            T_base_camera = np.asarray(
                result["cameras"][camera]["base_to_camera"]["matrix_4x4"]
            )
            kept_views = []
            for row in rows_by_camera[camera]:
                rotation_deg, translation_m = transform_residual(
                    row["T_base_tcp"] @ T_tcp_tag,
                    T_base_camera @ row["T_camera_tag"],
                )
                keep_rotation = (
                    args.trim_rotation_deg <= 0.0
                    or rotation_deg <= args.trim_rotation_deg
                )
                keep_translation = (
                    args.trim_translation_m <= 0.0
                    or translation_m <= args.trim_translation_m
                )
                if row["rmse"] <= 2.0 and keep_rotation and keep_translation:
                    kept_views.append(row["view"])
            trimmed_fits[camera] = {**fits[camera], "views": kept_views}
            print(
                f"TRIM {camera}: {len(fits[camera]['views'])} -> {len(kept_views)}",
                flush=True,
            )
        result = solve_fixed_camera_extrinsics(
            trimmed_fits,
            marker_size_m=0.16,
            min_samples=20,
        )
        print(
            "TRIMMED RESULT "
            + json.dumps(
                {
                    "status": result["status"],
                    "quality_gate_reasons": result["quality_gate_reasons"],
                    "sample_residuals": result["sample_residuals"],
                }
            ),
            flush=True,
        )
    T_tcp_tag = np.asarray(result["tool_to_board"]["matrix_4x4"])
    for camera in cameras:
        T_base_camera = np.asarray(result["cameras"][camera]["base_to_camera"]["matrix_4x4"])
        rows = rows_by_camera[camera]
        for row in rows:
            row["rotation_deg"], row["translation_m"] = transform_residual(
                row["T_base_tcp"] @ T_tcp_tag,
                T_base_camera @ row["T_camera_tag"],
            )
        print(f"CAMERA {camera} observations={len(rows)}")
        for key in (
            "rotation_deg",
            "translation_m",
            "rmse",
            "incidence",
            "area",
            "min_edge",
            "radius",
        ):
            values = np.asarray([row[key] for row in rows])
            quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
            print(f"  {key}: {np.round(quantiles, 4).tolist()}")
        rotation = np.asarray([row["rotation_deg"] for row in rows])
        correlations = {}
        for key in ("rmse", "incidence", "area", "min_edge", "radius"):
            values = np.asarray([row[key] for row in rows])
            correlations[key] = round(float(np.corrcoef(rotation, values)[0, 1]), 3)
        print(f"  rotation correlations: {correlations}")
        top = []
        for row in sorted(rows, key=lambda item: item["rotation_deg"], reverse=True)[:10]:
            top.append(
                {
                    "capture": row["index"],
                    "rotation_deg": round(row["rotation_deg"], 3),
                    "translation_mm": round(row["translation_m"] * 1000.0, 2),
                    "reprojection_px": round(row["rmse"], 3),
                    "incidence_deg": round(row["incidence"], 2),
                    "area_px2": round(row["area"], 1),
                    "min_edge_px": round(row["min_edge"], 1),
                    "radius": round(row["radius"], 3),
                }
            )
        print(f"  worst: {json.dumps(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
