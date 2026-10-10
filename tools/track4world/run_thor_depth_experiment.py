#!/usr/bin/env python3
"""Select FFS stereo or independent LingBot-Depth for Thor cam_03/cam_13."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRACK_SCRIPTS = ROOT / "third_party/Track4World/scripts"
sys.path.insert(0, str(TRACK_SCRIPTS))
from run_thor_sam3_recovery import (  # noqa: E402
    EXTRINSICS, SERIALS, TRACK_PYTHON, file_stamp, intrinsics_for, read_json,
)

CAMERAS = ("cam_03", "cam_13")
FILTER = "setpts=N/(60*TB),fps=8,scale=576:320:force_original_aspect_ratio=increase,crop=576:320"


def episode_inputs(dataset: Path, index: int, cameras: tuple[str, ...]) -> tuple[dict, dict]:
    episode = dataset / "episodes" / f"episode_{index:06d}"
    metadata_path = episode / "meta.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing episode metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("episode_index") != index or metadata.get("video", {}).get("fps") != 60:
        raise ValueError(f"{episode}: expected matching episode index and 60 fps video")
    videos = {camera: episode / f"{camera}.mkv" for camera in cameras}
    for video in videos.values():
        if not video.is_file():
            raise ValueError(f"Missing camera video: {video}")
    return ({camera: intrinsics_for(camera, episode, metadata) for camera in cameras}, videos)


def ffs_scene(dataset: Path, index: int) -> Path:
    return (TRACK_SCRIPTS.parent / "results" / dataset.name / f"episode_{index:06d}"
            / "stereo_cam13_cam03_robot_base")


def ensure_ffs_geometry(dataset: Path, indices: list[int], cameras: list[str], python: Path) -> None:
    missing = []
    for index in indices:
        scene = ffs_scene(dataset, index)
        for camera in cameras:
            geometry = scene / f"{camera}_geometry"
            if not all(path.is_file() for path in (
                scene / f"{camera}_rectified_8fps.mp4", scene / "stereo_calibration_report.json",
                geometry / "manifest.json", geometry / "ensemble_points.npy",
                geometry / "ensemble_valid.npy",
            )):
                missing.append(index)
                break
    if missing:
        command = [sys.executable, str(TRACK_SCRIPTS / "run_thor_sam3_recovery.py"),
                   "--dataset-root", str(dataset), "--track-python", str(python),
                   "--episodes", *(str(index) for index in missing), "--prepare-only"]
        subprocess.run(command, check=True, cwd=ROOT)


def prepare_camera(index: int, camera: str, intrinsics: dict, video: Path, dataset: Path,
                   run_root: Path, python: Path, lingbot_repo: Path, model: str,
                   input_mode: str, input_depth_dir: Path | None,
                   resolution_level: int) -> None:
    scene = run_root / f"episode_{index:06d}" / "stereo_cam13_cam03_lingbot_independent"
    input_dir = scene / "inputs"
    calibration_dir = scene / "calibration"
    input_video = input_dir / f"{camera}_8fps.mp4"
    intrinsic_path = calibration_dir / f"{camera}_factory_intrinsics.json"
    geometry = scene / f"{camera}_geometry"
    outputs = [scene / f"{camera}_rectified_8fps.mp4",
               *(geometry / name for name in ("manifest.json", "ensemble_points.npy",
                                               "ensemble_valid.npy", "depth_m.npy",
                                               "depth_input_m.npy"))]
    ffs_geometry = (ffs_scene(dataset, index) / f"{camera}_geometry"
                    if input_mode == "ffs" else None)
    source_video = (ffs_scene(dataset, index) / f"{camera}_rectified_8fps.mp4"
                    if ffs_geometry else video)
    input_depth = (input_depth_dir / f"episode_{index:06d}" / f"{camera}.npy"
                   if input_mode == "npy" else None)
    if input_depth is not None and not input_depth.is_file():
        raise ValueError(f"Missing independent depth input: {input_depth}")
    builder = Path(__file__).with_name("build_lingbot_depth_geometry.py")
    signature = {
        "camera": camera, "source_video": file_stamp(source_video), "intrinsics": intrinsics,
        "extrinsics": file_stamp(EXTRINSICS), "builder": file_stamp(builder),
        "model": model,
        "model_checkpoint": file_stamp(Path(model)) if Path(model).is_file() else None,
        "lingbot_source": file_stamp(lingbot_repo / "mdm/model/v2.py"),
        "input_depth_mode": input_mode,
        "input_depth": file_stamp(input_depth) if input_depth else None,
        "ffs_geometry": ({name: file_stamp(ffs_geometry / name) for name in
                          ("manifest.json", "ensemble_points.npy", "ensemble_valid.npy")}
                         if ffs_geometry else None),
        "resolution_level": resolution_level, "ffmpeg_filter": FILTER,
    }
    cache = geometry / "preparation.json"
    if cache.is_file() and read_json(cache) == signature and all(path.is_file() for path in outputs):
        print(f"Episode {index}/{camera}: reusing LingBot-Depth result", flush=True)
        return
    cache.unlink(missing_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_path.write_text(json.dumps(intrinsics, indent=2) + "\n")
    if ffs_geometry is None:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                        "-vf", FILTER, "-an", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(input_video)], check=True)
    command = [str(python), str(builder), "--video", str(source_video if ffs_geometry else input_video),
               "--camera", camera, "--output-root", str(scene),
               "--lingbot-repo", str(lingbot_repo), "--model", model,
               "--resolution-level", str(resolution_level)]
    if ffs_geometry:
        command.extend(["--ffs-geometry", str(ffs_geometry)])
    else:
        command.extend(["--intrinsics", str(intrinsic_path), "--extrinsics", str(EXTRINSICS)])
    if input_depth is not None:
        command.extend(["--input-depth", str(input_depth)])
    subprocess.run(command, check=True, cwd=ROOT)
    if not all(path.is_file() for path in outputs):
        raise RuntimeError(f"Episode {index}/{camera}: geometry output incomplete")
    cache.write_text(json.dumps(signature, indent=2) + "\n")


def run_independent_tracking(args: argparse.Namespace, dataset: Path, indices: list[int],
                             cameras: list[str], run_root: Path, python: Path,
                             output_prefix: str) -> None:
    # One camera per process keeps Track4World's two-camera fusion branch inactive.
    for camera in cameras:
        command = [sys.executable, str(TRACK_SCRIPTS / "run_interactive_sam3_recovery.py"),
                   "--source-format", "thor", "--dataset-root", str(dataset),
                   "--run-root", str(run_root), "--track-python", str(python),
                   "--episodes", *(str(index) for index in indices), "--cameras", camera,
                   "--output-name", f"{output_prefix}_{camera}"]
        if args.reuse_selections:
            command.append("--reuse-selections")
        if args.select_again:
            command.append("--select-again")
        subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-backend", choices=("ffs", "lingbot-depth"), default="ffs")
    parser.add_argument("--dataset-root", type=Path, required=True)
    episode_mode = parser.add_mutually_exclusive_group()
    episode_mode.add_argument("--episodes", type=int, nargs="+", default=None)
    episode_mode.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS),
                        help="LingBot-Depth cameras; each is processed independently")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--track-python", type=Path, default=TRACK_PYTHON)
    parser.add_argument("--lingbot-repo", type=Path, default=ROOT / "third_party/lingbot-depth")
    parser.add_argument("--lingbot-model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    parser.add_argument("--lingbot-input", choices=("ffs", "zero", "npy"), default="ffs",
                        help="Initial depth for LingBot-Depth (default: existing FFS depth)")
    parser.add_argument("--input-depth-dir", type=Path,
                        help="Per-camera .npy maps for --lingbot-input npy")
    parser.add_argument("--resolution-level", type=int, default=5, choices=range(1, 10))
    parser.add_argument("--reuse-selections", action="store_true")
    parser.add_argument("--select-again", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset_root.expanduser().resolve()
    if not (dataset / "episodes").is_dir():
        parser.error(f"Thor dataset has no episodes/: {dataset}")
    if args.all_episodes:
        indices = sorted(int(path.name.removeprefix("episode_"))
                         for path in (dataset / "episodes").glob("episode_[0-9]*")
                         if path.is_dir() and path.name.removeprefix("episode_").isdigit())
    else:
        indices = args.episodes if args.episodes is not None else [1]
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        parser.error("Choose distinct, nonnegative episode indices")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("Each camera may be selected once")
    if args.reuse_selections and args.select_again:
        parser.error("Selection flags are mutually exclusive")
    if (args.lingbot_input == "npy") != (args.input_depth_dir is not None):
        parser.error("--lingbot-input npy requires --input-depth-dir; other modes do not use it")
    if args.depth_backend == "ffs":
        if args.cameras != list(CAMERAS) or args.lingbot_input != "ffs":
            parser.error("--cameras and --lingbot-input apply only to LingBot-Depth")
        command = [sys.executable, str(TRACK_SCRIPTS / "run_thor_sam3_recovery.py"),
                   "--dataset-root", str(dataset), "--track-python", str(args.track_python)]
        command += (["--all-episodes"] if args.all_episodes else
                    ["--episodes", *(str(index) for index in indices)])
        command.append("--check-only" if args.check_only else "--prepare-only")
        subprocess.run(command, check=True, cwd=ROOT)
        if not args.check_only and not args.prepare_only:
            run_independent_tracking(args, dataset, indices, list(CAMERAS),
                                     TRACK_SCRIPTS.parent / "results" / dataset.name,
                                     args.track_python.expanduser().absolute(), "sam3_selected_ffs")
        return

    if not EXTRINSICS.is_file():
        parser.error(f"Camera calibration is missing: {EXTRINSICS}")
    calibration = read_json(EXTRINSICS)
    if calibration.get("status") != "passed" or calibration.get("joint_solution", {}).get("status") != "ok":
        parser.error(f"Camera calibration did not pass: {EXTRINSICS}")
    for camera in args.cameras:
        item = calibration["joint_solution"].get("cameras", {}).get(camera, {})
        if SERIALS[camera] not in str(item.get("intrinsics_path", "")):
            parser.error(f"Calibration does not match {camera} serial {SERIALS[camera]}")
    try:
        selected = {index: episode_inputs(dataset, index, tuple(args.cameras)) for index in indices}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"LingBot-Depth independent cameras: {', '.join(args.cameras)}\n"
          f"Episodes: {', '.join(map(str, indices))}", flush=True)
    if args.check_only:
        print("Camera inputs and calibration passed checks.")
        return
    # Keep a Conda or venv launcher symlink; resolving it can switch interpreters.
    python = args.track_python.expanduser().absolute()
    lingbot_repo = args.lingbot_repo.expanduser().resolve()
    if not python.is_file():
        parser.error(f"Python interpreter is missing: {python}")
    if not (lingbot_repo / "mdm/model/v2.py").is_file():
        parser.error(f"LingBot-Depth checkout is missing: {lingbot_repo}")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    run_root = TRACK_SCRIPTS.parent / "results" / f"{dataset.name}_lingbot_depth"
    if args.lingbot_input == "zero":
        print("Zero-depth RGB-only trial: upstream metric scale is unverified.", flush=True)
    elif args.lingbot_input == "ffs":
        ensure_ffs_geometry(dataset, indices, args.cameras, python)
    for index in indices:
        intrinsics, videos = selected[index]
        for camera in args.cameras:
            prepare_camera(index, camera, intrinsics[camera], videos[camera], dataset, run_root,
                           python, lingbot_repo, args.lingbot_model, args.lingbot_input,
                           args.input_depth_dir,
                           args.resolution_level)
    if args.prepare_only:
        print(f"Per-camera depth outputs: {run_root}")
        return
    run_independent_tracking(args, dataset, indices, args.cameras, run_root, python,
                             "sam3_selected")


if __name__ == "__main__":
    main()
