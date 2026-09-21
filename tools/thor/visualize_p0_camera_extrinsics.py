#!/usr/bin/env python3
"""Render current-FR3-base camera extrinsics in XY, XZ, YZ, and XYZ views."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
AXIS_NAMES = ("x", "y", "z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_matrix(camera: str, detail: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(detail.get("base_to_camera", {}).get("matrix_4x4", []), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{camera}: base_to_camera.matrix_4x4 must be a finite 4x4 matrix")
    return matrix


def _set_equal_3d(ax: Any, points: np.ndarray, margin: float = 0.12) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0 + margin, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _frustum_points(T_base_camera: np.ndarray, depth: float) -> np.ndarray:
    half_width = depth * 0.55
    half_height = depth * 0.38
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ],
        dtype=np.float64,
    )
    return (T_base_camera[:3, :3] @ local.T).T + T_base_camera[:3, 3]


def _draw_2d_view(
    ax: Any,
    cameras: list[dict[str, Any]],
    dims: tuple[int, int],
    labels: tuple[str, str],
    colors: list[Any],
    optical_axis_length: float,
) -> None:
    horizontal, vertical = dims
    ax.scatter([0.0], [0.0], marker="*", s=180, color="black", zorder=5)
    ax.annotate("FR3 base", (0.0, 0.0), xytext=(6, 6), textcoords="offset points", fontsize=9)
    for index, (camera, color) in enumerate(zip(cameras, colors, strict=True)):
        position = camera["matrix"][:3, 3]
        optical = camera["matrix"][:3, 2]
        ax.scatter([position[horizontal]], [position[vertical]], s=48, color=color, zorder=4)
        ax.arrow(
            position[horizontal],
            position[vertical],
            optical[horizontal] * optical_axis_length,
            optical[vertical] * optical_axis_length,
            color=color,
            width=0.0025,
            head_width=0.025,
            length_includes_head=True,
            zorder=3,
        )
        offset_y = 7 if index % 2 == 0 else -13
        ax.annotate(
            camera["name"],
            (position[horizontal], position[vertical]),
            xytext=(6, offset_y),
            textcoords="offset points",
            fontsize=8,
            color=color,
            weight="bold",
        )
    ax.set_xlabel(f"FR3 base {labels[0]} (m)")
    ax.set_ylabel(f"FR3 base {labels[1]} (m)")
    ax.set_title(f"{labels[0].upper()}{labels[1].upper()} view — arrow = camera optical +Z")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)


def _draw_3d_view(
    ax: Any,
    cameras: list[dict[str, Any]],
    colors: list[Any],
    axis_length: float,
    frustum_depth: float,
) -> np.ndarray:
    all_points = [np.zeros(3)]
    for axis_index, (axis_name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS, strict=True)):
        vector = np.zeros(3)
        vector[axis_index] = axis_length * 1.3
        ax.quiver(0.0, 0.0, 0.0, *vector, color=color, linewidth=2.5, arrow_length_ratio=0.18)
        ax.text(*vector, f"base {axis_name}", color=color, fontsize=8)
        all_points.append(vector)
    for camera, color in zip(cameras, colors, strict=True):
        matrix = camera["matrix"]
        position = matrix[:3, 3]
        all_points.append(position)
        ax.scatter(*position, s=52, color=color, depthshade=False)
        ax.text(*(position + np.array([0.015, 0.015, 0.015])), camera["name"], color=color, fontsize=8)
        for axis_index, axis_color in enumerate(AXIS_COLORS):
            vector = matrix[:3, axis_index] * axis_length
            ax.quiver(
                *position,
                *vector,
                color=axis_color,
                linewidth=1.1,
                arrow_length_ratio=0.2,
            )
            all_points.append(position + vector)
        frustum = _frustum_points(matrix, frustum_depth)
        all_points.extend(frustum)
        for corner in range(1, 5):
            ax.plot(
                [frustum[0, 0], frustum[corner, 0]],
                [frustum[0, 1], frustum[corner, 1]],
                [frustum[0, 2], frustum[corner, 2]],
                color=color,
                alpha=0.75,
                linewidth=1.0,
            )
        loop = [1, 2, 3, 4, 1]
        ax.plot(frustum[loop, 0], frustum[loop, 1], frustum[loop, 2], color=color, alpha=0.75)
    ax.set_xlabel("FR3 base X (m)")
    ax.set_ylabel("FR3 base Y (m)")
    ax.set_zlabel("FR3 base Z (m)")
    ax.set_title("XYZ view — camera axes: X red, Y green, optical Z blue")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.3)
    points = np.asarray(all_points)
    _set_equal_3d(ax, points)
    return points


def visualize(summary_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    summary_path = summary_path.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"calibration is not passing: status={summary.get('status')!r}")
    camera_details = summary.get("joint_solution", {}).get("cameras", {})
    if not isinstance(camera_details, dict) or not camera_details:
        raise ValueError("calibration contains no cameras")
    cameras = []
    for name, detail in sorted(camera_details.items()):
        matrix = _camera_matrix(name, detail)
        cameras.append({"name": name, "detail": detail, "matrix": matrix})

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "camera_extrinsics_xy_xz_yz_xyz.png"
    visualization_summary_path = output_dir / "camera_extrinsics_visualization.json"
    calibration_copy_path = output_dir / "calibration_summary.json"

    positions = np.asarray([camera["matrix"][:3, 3] for camera in cameras])
    spread = max(float(np.max(np.ptp(np.vstack([np.zeros(3), positions]), axis=0))), 0.5)
    axis_length = float(np.clip(spread * 0.11, 0.08, 0.16))
    frustum_depth = axis_length * 1.15
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(cameras))]

    figure = plt.figure(figsize=(16, 13), constrained_layout=True)
    ax_xy = figure.add_subplot(2, 2, 1)
    ax_xz = figure.add_subplot(2, 2, 2)
    ax_yz = figure.add_subplot(2, 2, 3)
    ax_xyz = figure.add_subplot(2, 2, 4, projection="3d")
    _draw_2d_view(ax_xy, cameras, (0, 1), ("x", "y"), colors, axis_length * 1.4)
    _draw_2d_view(ax_xz, cameras, (0, 2), ("x", "z"), colors, axis_length * 1.4)
    _draw_2d_view(ax_yz, cameras, (1, 2), ("y", "z"), colors, axis_length * 1.4)
    plot_points = _draw_3d_view(ax_xyz, cameras, colors, axis_length, frustum_depth)
    residuals = summary.get("joint_solution", {}).get("sample_residuals", {})
    figure.suptitle(
        "Fixed cameras in current FR3 base\n"
        f"rotation mean={float(residuals.get('rotation_deg_mean', 0.0)):.3f} deg, "
        f"translation mean={float(residuals.get('translation_m_mean', 0.0)) * 1000.0:.2f} mm",
        fontsize=15,
    )
    figure.savefig(image_path, dpi=180, facecolor="white")
    plt.close(figure)

    payload = {
        "schema": "fr3_base_camera_extrinsics_visualization/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_calibration": str(summary_path),
        "source_calibration_sha256": _sha256(summary_path),
        "source_status": summary["status"],
        "world_frame_id": summary.get("world", {}).get("world_frame_id", "fr3_base"),
        "transform_convention": "T_base_camera maps camera coordinates into FR3 base",
        "camera_axis_convention": "OpenCV camera +X right, +Y down, +Z optical forward",
        "visualization_png": str(image_path),
        "bounds_base_xyz_m": {
            "min": plot_points.min(axis=0).tolist(),
            "max": plot_points.max(axis=0).tolist(),
        },
        "joint_sample_residuals": residuals,
        "robust_filter": summary.get("joint_solution", {}).get("robust_filter", {}),
        "cameras": {},
    }
    for camera in cameras:
        name = camera["name"]
        detail = camera["detail"]
        matrix = camera["matrix"]
        payload["cameras"][name] = {
            "T_base_camera": matrix.tolist(),
            "position_base_xyz_m": matrix[:3, 3].tolist(),
            "optical_axis_base_xyz": matrix[:3, 2].tolist(),
            "camera_x_axis_base_xyz": matrix[:3, 0].tolist(),
            "camera_y_axis_base_xyz": matrix[:3, 1].tolist(),
            "rpy_deg": detail["base_to_camera"].get("rpy_deg"),
            "quaternion_xyzw": detail["base_to_camera"].get("quaternion_xyzw"),
            "num_input_observations": detail.get("num_input_observations"),
            "num_observations": detail.get("num_observations"),
            "num_rejected_outliers": detail.get("num_rejected_outliers"),
            "sample_residuals": detail.get("sample_residuals"),
            "intrinsics_source_camera": detail.get("intrinsics_source_camera"),
            "temporary_intrinsics_reuse": detail.get("temporary_intrinsics_reuse", False),
        }
    visualization_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, calibration_copy_path)
    return image_path, visualization_summary_path, calibration_copy_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else summary_path.parent / "visualization"
    )
    image_path, visualization_summary_path, calibration_copy_path = visualize(
        summary_path, output_dir
    )
    print(f"[VISUALIZATION] png={image_path}")
    print(f"[VISUALIZATION] json={visualization_summary_path}")
    print(f"[VISUALIZATION] calibration_summary={calibration_copy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
