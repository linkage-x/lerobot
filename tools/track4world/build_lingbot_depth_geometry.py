#!/usr/bin/env python3
"""Experimental per-camera LingBot-Depth geometry for Thor videos.

The released v0.5 model requires a depth tensor. A zero tensor permits an
RGB-only *trial*, but its metric scale has not been validated by the authors.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def camera_matrices(intrinsics_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(intrinsics_path.read_text())
    source_width, source_height = int(data["image_width"]), int(data["image_height"])
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("Invalid image dimensions")
    intrinsic = np.asarray(data["camera_matrix"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Invalid camera matrix")
    intrinsic = np.diag([width / source_width, height / source_height, 1.0]) @ intrinsic
    distortion = np.asarray(data.get("dist_coeffs", []), dtype=np.float64)
    return intrinsic, distortion


def base_from_camera(extrinsics_path: Path, camera: str) -> np.ndarray:
    data = json.loads(extrinsics_path.read_text())
    joint = data.get("joint_solution", {})
    if joint.get("status") != "ok":
        raise ValueError("Joint camera calibration did not pass")
    item = joint.get("cameras", {}).get(camera, {})
    transform = np.asarray(item.get("base_to_camera", {}).get("matrix_4x4", []), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Missing base-to-camera transform for {camera}")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"Invalid base-to-camera transform for {camera}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.99:
        raise ValueError(f"Invalid base-to-camera rotation for {camera}")
    return transform


def depth_to_base_points(depth: np.ndarray, intrinsic: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    rays = np.stack(((x - intrinsic[0, 2]) / intrinsic[0, 0],
                     (y - intrinsic[1, 2]) / intrinsic[1, 1],
                     np.ones_like(x)), axis=-1)
    points = rays * depth[..., None]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points[~np.isfinite(depth)] = np.nan
    return points.astype(np.float32)


def ffs_depth_from_base_points(points: np.ndarray, valid: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Recover rectified camera Z from FFS robot-base points at the same pixels."""
    camera_z_axis_in_base = transform[:3, 2]
    depth = (points - transform[:3, 3]) @ camera_z_axis_in_base
    depth = depth.astype(np.float32)
    depth[~valid | ~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def enable_single_image_sdpa_fallback(model: object) -> bool:
    """Run LingBot's one-item token list through PyTorch attention without xFormers.

    LingBot's ViT passes a list even for batch size one. Its nested block only
    accepts that list with xFormers, although a one-item list needs no block
    diagonal attention bias. Passing its sole tensor through the ordinary block
    preserves depth-token masking and uses MemEffAttention's PyTorch SDPA path.
    """
    from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock, XFORMERS_AVAILABLE

    if XFORMERS_AVAILABLE:
        return False

    def forward_one(self: NestedTensorBlock, x: object) -> object:
        if isinstance(x, list):
            if len(x) != 1:
                raise RuntimeError("PyTorch attention fallback only supports one image at a time")
            return [Block.forward(self, x[0])]
        return Block.forward(self, x)

    for module in model.modules():
        if isinstance(module, NestedTensorBlock):
            module.forward = types.MethodType(forward_one, module)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--camera", required=True, choices=("cam_03", "cam_13"))
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lingbot-repo", required=True, type=Path)
    parser.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    parser.add_argument("--input-depth", type=Path,
                        help="Optional [frames,H,W] float32 meters .npy for this camera only")
    parser.add_argument("--ffs-geometry", type=Path,
                        help="FFS per-camera geometry aligned with an already rectified --video")
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--resolution-level", type=int, default=5, choices=range(1, 10))
    args = parser.parse_args()
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("Invalid depth range")
    if args.ffs_geometry and args.input_depth:
        parser.error("Choose FFS geometry or an independent .npy depth input")
    if not args.ffs_geometry and (not args.intrinsics or not args.extrinsics):
        parser.error("--intrinsics and --extrinsics are required without FFS geometry")
    if not (args.lingbot_repo / "mdm/model/v2.py").is_file():
        parser.error(f"LingBot-Depth checkout is missing: {args.lingbot_repo}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"Cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {args.video}")
    ffs_points = ffs_valid = None
    if args.ffs_geometry:
        ffs_manifest = json.loads((args.ffs_geometry / "manifest.json").read_text())
        if (ffs_manifest.get("camera_key") != args.camera or
                ffs_manifest.get("coordinate_frame") != "robot_base" or
                ffs_manifest.get("frame_count") != frame_count or
                (ffs_manifest.get("height"), ffs_manifest.get("width")) != (height, width)):
            raise ValueError(f"FFS geometry does not match {args.camera} video")
        ffs_points = np.load(args.ffs_geometry / "ensemble_points.npy", mmap_mode="r", allow_pickle=False)
        ffs_valid = np.load(args.ffs_geometry / "ensemble_valid.npy", mmap_mode="r", allow_pickle=False)
        if ffs_points.shape != (frame_count, height, width, 3) or ffs_valid.shape != (frame_count, height, width):
            raise ValueError("FFS array shapes do not match video")
        report_path = Path(ffs_manifest["stereo_calibration_report"])
        report = json.loads(report_path.read_text())
        projection_key = "P_rect_left" if args.camera == "cam_13" else "P_rect_right"
        projection = np.asarray(report[projection_key], dtype=np.float64)
        if projection.shape != (3, 4):
            raise ValueError(f"Invalid FFS {projection_key}")
        new_intrinsic = projection[:, :3]
        transform = np.asarray(ffs_manifest["T_output_rectified_camera"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("Invalid FFS camera-to-base transform")
        map_x = map_y = None
    else:
        intrinsic, distortion = camera_matrices(args.intrinsics, width, height)
        new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, height), 0, (width, height))
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsic, distortion, None, new_intrinsic,
                                                    (width, height), cv2.CV_32FC1)
        transform = base_from_camera(args.extrinsics, args.camera)
    depth_input = None
    if args.input_depth:
        depth_input = np.load(args.input_depth, mmap_mode="r", allow_pickle=False)
        if depth_input.shape != (frame_count, height, width):
            raise ValueError(f"Input depth shape {depth_input.shape} != {(frame_count, height, width)}")

    sys.path.insert(0, str(args.lingbot_repo.resolve()))
    import torch
    from mdm.model.v2 import MDMModel

    if not torch.cuda.is_available():
        raise RuntimeError("LingBot-Depth experiment requires an available CUDA GPU")
    model = MDMModel.from_pretrained(args.model).cuda().eval()
    if enable_single_image_sdpa_fallback(model):
        print("xFormers unavailable; using PyTorch SDPA for single-image LingBot inference", flush=True)
    geometry_dir = args.output_root / f"{args.camera}_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    (geometry_dir / "manifest.json").unlink(missing_ok=True)
    preview_dir = args.output_root / f"{args.camera}_depth_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    video_output = args.output_root / f"{args.camera}_rectified_8fps.mp4"
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot write {video_output}")
    points = np.lib.format.open_memmap(geometry_dir / "ensemble_points.npy", mode="w+",
                                       dtype=np.float32, shape=(frame_count, height, width, 3))
    valid_output = np.lib.format.open_memmap(geometry_dir / "ensemble_valid.npy", mode="w+",
                                             dtype=bool, shape=(frame_count, height, width))
    depth_output = np.lib.format.open_memmap(geometry_dir / "depth_m.npy", mode="w+",
                                             dtype=np.float32, shape=(frame_count, height, width))
    depth_prior_output = np.lib.format.open_memmap(geometry_dir / "depth_input_m.npy", mode="w+",
                                                   dtype=np.float32, shape=(frame_count, height, width))
    normalized_intrinsic = new_intrinsic.astype(np.float32).copy()
    normalized_intrinsic[0] /= width
    normalized_intrinsic[1] /= height
    intrinsic_tensor = torch.from_numpy(normalized_intrinsic)[None].cuda()
    preview_frames = set(np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int))
    try:
        for frame_index in range(frame_count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {frame_index}/{frame_count}: {args.video}")
            undistorted = (bgr if args.ffs_geometry else
                           cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR))
            writer.write(undistorted)
            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb.copy()).cuda().float().permute(2, 0, 1)[None] / 255.0
            if ffs_points is not None:
                raw_depth = ffs_depth_from_base_points(np.asarray(ffs_points[frame_index]),
                                                       np.asarray(ffs_valid[frame_index]), transform)
            elif depth_input is None:
                raw_depth = np.zeros((height, width), dtype=np.float32)
            else:
                # The supplied map is aligned to the original RGB frame.
                raw_depth = cv2.remap(np.asarray(depth_input[frame_index], dtype=np.float32),
                                      map_x, map_y, cv2.INTER_NEAREST)
                raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
                raw_depth[raw_depth < 0] = 0
            depth_prior_output[frame_index] = raw_depth
            depth_tensor = torch.from_numpy(raw_depth.copy()).cuda()[None]
            with torch.inference_mode():
                result = model.infer(image, depth_in=depth_tensor,
                                     intrinsics=intrinsic_tensor,
                                     resolution_level=args.resolution_level)
            depth = result["depth"].squeeze(0).float().cpu().numpy()
            valid = np.isfinite(depth) & (depth >= args.min_depth_m) & (depth <= args.max_depth_m)
            if "mask" in result:
                valid &= result["mask"].squeeze(0).cpu().numpy().astype(bool)
            depth = np.where(valid, depth, np.nan).astype(np.float32)
            depth_output[frame_index] = depth
            points[frame_index] = depth_to_base_points(depth, new_intrinsic, transform)
            valid_output[frame_index] = valid
            if frame_index in preview_frames:
                def colorize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
                    normalized = np.clip((np.nan_to_num(values, nan=args.max_depth_m)
                                          - args.min_depth_m) / (args.max_depth_m - args.min_depth_m), 0, 1)
                    color = cv2.applyColorMap((255 * (1 - normalized)).astype(np.uint8),
                                              cv2.COLORMAP_TURBO)
                    color[~mask] = 0
                    return color

                prior_valid = np.isfinite(raw_depth) & (raw_depth > 0)
                cv2.imwrite(str(preview_dir / f"frame_{frame_index:04d}.jpg"),
                            np.concatenate((undistorted, colorize(raw_depth, prior_valid),
                                            colorize(depth, valid)), axis=1))
            if frame_index % 16 == 0 or frame_index == frame_count - 1:
                print(f"{args.camera}: {frame_index + 1}/{frame_count} frames", flush=True)
    finally:
        capture.release()
        writer.release()
        points.flush(); valid_output.flush(); depth_output.flush(); depth_prior_output.flush()
    manifest = {
        "camera_key": args.camera, "source_video": str(video_output.resolve()),
        "fps": fps, "frame_count": frame_count, "height": height, "width": width,
        "coordinate_frame": "robot_base", "method": "LingBot-Depth v0.5 per-camera experiment",
        "input_depth": (str(args.ffs_geometry.resolve()) if args.ffs_geometry else
                        str(args.input_depth.resolve()) if args.input_depth else "zeros"),
        "input_depth_method": ("Fast-FoundationStereo" if args.ffs_geometry else
                               "per-camera supplied array" if args.input_depth else "zeros"),
        "metric_scale_verified": False,
        "scale_note": ("Depth input is in meters; output still needs validation against a reference."
                       if depth_input is not None or ffs_points is not None else
                       "RGB-only zero-depth input is unsupported by upstream examples; metric scale is unverified."),
        "model": args.model,
        "intrinsics": str(report_path.resolve()) if args.ffs_geometry else str(args.intrinsics.resolve()),
        "extrinsics": str(args.extrinsics.resolve()) if args.extrinsics else None,
        "T_output_camera": transform.tolist(),
        "K_undistorted": new_intrinsic.tolist(),
    }
    (geometry_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
