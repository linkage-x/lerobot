#!/usr/bin/env python3
"""Interactive two-marker FR3-base calibration for Thor GMSL2 cameras.

This process is intentionally independent of the data-collection GUI.  It is
started from the host through ``run_p0_two_marker_calibration.sh`` but runs on
Thor, where both the Argus cameras and FR3 are connected.  The live viewer
reads the recorder-owned synchronized frame bus, so it never opens a second
camera session.

Controls in the OpenCV window:
    Enter  capture the displayed synchronized camera cluster + measured FR3 pose
    q      finish and solve (only after the minimum observation gate is ready)
    f      force a solve attempt before the readiness gate
    Esc    abort without replacing the active calibration
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.robots import franka_research3, make_robot_from_config  # noqa: E402
from lerobot.robots.franka_research3 import FrankaResearch3Config  # noqa: E402
from third_party.opencv_kalibr.fr3_calibration.execute_pose_and_capture import (  # noqa: E402
    JOINT_VECTOR_NAMES,
)
from third_party.opencv_kalibr.fr3_calibration.execute_pose_and_capture_thor_gmsl2 import (  # noqa: E402
    ThorRecorderClient,
)
from tools.thor.gmsl2.online_sync_frame_client import (  # noqa: E402
    OnlineSyncCluster,
    ThorOnlineSyncFrameClient,
)
from tools.thor.p1_eye_hand_core import (  # noqa: E402
    EyeHandObservation,
    assess_solution,
    matrix_payload,
    solve_base_with_fixed_tcp_tags,
    solve_eye_hand,
)
from tools.thor.p1_simple_eye_hand_calibration import (  # noqa: E402
    BACKING_SIZE_M,
    DEFAULT_EXTRINSICS,
    DEFAULT_INTRINSICS,
    DEFAULT_P0_SOURCE,
    DEFAULT_ROOT,
    TAG_IDS,
    TAG_SIZE_M,
    _intrinsics_index,
    _sha256,
    _tag_pose,
    _world_cameras,
    _write_json,
    create_retargeted_p0_plan,
)

DEFAULT_RECORDER_CONFIG = (
    REPO_ROOT / "third_party/opencv_kalibr/fr3_calibration/host/thor_gmsl2_calibration.yaml"
)
DEFAULT_URDF = (
    REPO_ROOT
    / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
)
WINDOW_NAME = "P0 two-marker calibration (Enter=capture, q=solve, Esc=abort)"
MIN_ROBOT_POSES = 15
MIN_POSES_PER_TAG = 8
MIN_CAMERAS = 2


@dataclass
class CameraResult:
    camera: str
    image_bgr: np.ndarray
    annotated_bgr: np.ndarray
    detections: list[dict[str, Any]]
    error: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _matrix_from_payload(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _load_active(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    active_path = root / "active.json"
    if not active_path.is_file():
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    calibration_path = Path(str(active.get("calibration_path", "")))
    plan_path = Path(str(active.get("plan_path", "")))
    if not calibration_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(f"Active calibration references missing files: {active_path}")
    if active.get("calibration_sha256") and _sha256(calibration_path) != active["calibration_sha256"]:
        raise ValueError(f"Active calibration SHA-256 mismatch: {calibration_path}")
    if active.get("plan_sha256") and _sha256(plan_path) != active["plan_sha256"]:
        raise ValueError(f"Active P0 plan SHA-256 mismatch: {plan_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise ValueError(f"Active calibration is not passing: {calibration_path}")
    return calibration_path, calibration, active


def _choose_existing(mode: str, active: tuple[Path, dict[str, Any], dict[str, Any]] | None) -> bool:
    """Return True when the existing calibration should be reused and execution should end."""
    if active is None:
        if mode == "reuse":
            raise FileNotFoundError("--existing reuse requested, but no active calibration exists")
        return False
    path, calibration, _ = active
    quality = calibration.get("quality", {})
    print(
        f"[EXISTING] {path}\n"
        f"  created={calibration.get('created_utc', 'unknown')} "
        f"poses={quality.get('num_robot_poses', '?')} cameras={quality.get('num_cameras', '?')}",
        flush=True,
    )
    if mode == "reuse":
        return True
    if mode == "recalibrate":
        return False
    if not sys.stdin.isatty():
        raise RuntimeError("An active calibration exists; use --existing reuse or --existing recalibrate")
    while True:
        answer = input("Use existing calibration and exit? [Y=reuse / r=recalibrate]: ").strip().lower()
        if answer in {"", "y", "yes", "u", "use", "reuse"}:
            return True
        if answer in {"r", "redo", "recalibrate"}:
            return False


def _activate_teaching_mode(robot: Any) -> None:
    arm = getattr(robot, "_arm", None)
    if arm is None:
        raise RuntimeError("FR3 arm backend is unavailable")
    stop_otg = getattr(robot, "_stop_otg_loop", None)
    if callable(stop_otg):
        stop_otg()
    arm.damping = [0.0] * 7
    arm.stiffness = [0.0] * 7
    stop_controller = getattr(arm, "_stop_controller", None)
    start_controller = getattr(arm, "_start_controller", None)
    if callable(stop_controller):
        stop_controller()
    if callable(start_controller):
        start_controller()
    print("[ROBOT] teaching mode active: all joint stiffness/damping = 0", flush=True)


def _make_robot(robot_ip: str, urdf: Path) -> Any:
    if not urdf.is_file():
        raise FileNotFoundError(f"FR3 URDF not found: {urdf}")
    _ = franka_research3
    cfg = FrankaResearch3Config(
        id="p0_two_marker_calibration",
        robot_ip=robot_ip,
        gripper_backend="mock",
        allow_mock_gripper=True,
        urdf_path=str(urdf),
        target_frame_name="fr3_ee",
        cameras={},
        use_otg=False,
        stiffness=[0.0] * 7,
        damping=[0.0] * 7,
    )
    return make_robot_from_config(cfg)


def _make_recorder_config(template: Path, run_dir: Path, frame_bus_dir: Path, every_n: int) -> Path:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Recorder config root must be a mapping: {template}")
    cameras = payload.setdefault("sensors", {}).setdefault("cameras", {})
    cameras.setdefault("defaults", {})["recorder_backend"] = "argus_online_sync"
    online = cameras.setdefault("online_sync", {})
    online["enabled"] = True
    online["frame_bus_dir"] = str(frame_bus_dir)
    online["frame_bus_every_n"] = max(1, int(every_n))
    payload.setdefault("box_collection", {})["enabled"] = False
    dataset = payload.setdefault("dataset", {})
    dataset["root"] = str(run_dir / "unused_recorder_dataset")
    dataset["num_episodes"] = 0
    dataset["single_task"] = "P0 interactive two-marker calibration"
    config_dir = Path(tempfile.mkdtemp(prefix="p0_two_marker_cfg_", dir="/tmp"))
    path = config_dir / "recorder.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _start_recorder(config_path: Path, *, skip_hardware_sync: bool) -> ThorRecorderClient:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/thor/gmsl2/thor_record.py"),
        "--config-path",
        str(config_path),
        "--repo-root",
        str(REPO_ROOT),
        "--no-box",
        "--skip-argus-probe",
    ]
    if skip_hardware_sync:
        cmd.append("--skip-hardware-sync")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    recorder = ThorRecorderClient(cmd, cwd=REPO_ROOT, env=env)
    print(f"[THOR] command: {' '.join(cmd)}", flush=True)
    recorder.start(wait_ready_timeout_s=240.0)
    return recorder


def _new_detector() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return (dictionary, parameters)


def _detect_camera(
    camera: str,
    image_bgr: np.ndarray,
    intrinsics: dict[str, Any],
    T_world_camera: np.ndarray,
    detection_scale: float,
    max_rmse_px: float,
) -> CameraResult:
    if 0.0 < detection_scale < 1.0:
        work = cv2.resize(image_bgr, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    else:
        work = image_bgr.copy()
    annotated = work.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    detector = _new_detector()
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    detections: list[dict[str, Any]] = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for marker_corners, marker_id_raw in zip(corners, ids.reshape(-1), strict=True):
            marker_id = int(marker_id_raw)
            if marker_id not in TAG_IDS:
                continue
            pose = _tag_pose(marker_corners, intrinsics, gray.shape)
            if pose is None:
                continue
            T_camera_tag, rmse = pose
            if rmse > max_rmse_px:
                continue
            detections.append(
                {
                    "tag_id": marker_id,
                    "reprojection_rmse_px": float(rmse),
                    "T_camera_tag": matrix_payload(T_camera_tag),
                    "T_world_tag": matrix_payload(T_world_camera @ T_camera_tag),
                }
            )
    return CameraResult(camera, image_bgr, annotated, detections)


def _process_cluster(
    cluster: OnlineSyncCluster,
    intrinsics: dict[str, dict[str, Any]],
    world_cameras: dict[str, np.ndarray],
    pool: ThreadPoolExecutor,
    detection_scale: float,
    max_rmse_px: float,
) -> dict[str, CameraResult]:
    cameras = sorted(cluster.frames)

    def task(camera: str) -> CameraResult:
        try:
            image_bgr = cv2.cvtColor(cluster.frames[camera].as_rgb(), cv2.COLOR_RGB2BGR)
            missing = []
            if camera not in intrinsics:
                missing.append("intrinsics")
            if camera not in world_cameras:
                missing.append("extrinsics")
            if missing:
                return CameraResult(
                    camera,
                    image_bgr,
                    image_bgr.copy(),
                    [],
                    "missing " + "/".join(missing),
                )
            return _detect_camera(
                camera,
                image_bgr,
                intrinsics[camera],
                world_cameras[camera],
                detection_scale,
                max_rmse_px,
            )
        except Exception as exc:  # keep one camera failure visible without killing the viewer
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            return CameraResult(camera, blank, blank.copy(), [], str(exc))

    return {result.camera: result for result in pool.map(task, cameras)}


def _draw_mosaic(
    results: dict[str, CameraResult],
    counts: dict[str, int],
    target: int,
    status_message: str,
) -> np.ndarray:
    tile_w, tile_h = 480, 300
    cameras = sorted(results)
    cols = min(3, max(1, len(cameras)))
    rows = (len(cameras) + cols - 1) // cols
    header_h = 88
    canvas = np.zeros((rows * tile_h + header_h, cols * tile_w, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "ENTER capture | q solve | f force solve | ESC abort",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GREEN=56+57  YELLOW=one marker  RED=no valid marker",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_message[:120],
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (100, 230, 255),
        1,
        cv2.LINE_AA,
    )
    for index, camera in enumerate(cameras):
        result = results[camera]
        tile = cv2.resize(result.annotated_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tag_ids = {int(item["tag_id"]) for item in result.detections}
        color = (0, 200, 0) if tag_ids == set(TAG_IDS) else ((0, 210, 255) if tag_ids else (0, 0, 220))
        y0 = header_h + (index // cols) * tile_h
        x0 = (index % cols) * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + tile_w - 3, y0 + tile_h - 3), color, 5)
        label = f"{camera} tags={sorted(tag_ids)} valid={counts.get(camera, 0)}/{target}"
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + tile_w - 4, y0 + 34), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + 10, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        if result.error:
            cv2.putText(canvas, result.error[:55], (x0 + 10, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 1)
    return canvas


def _robot_sample(robot: Any, settle_s: float, max_joint_delta_rad: float) -> tuple[np.ndarray, list[float]]:
    first = robot.get_observation(include_cameras=False)
    time.sleep(max(0.0, settle_s))
    second = robot.get_observation(include_cameras=False)
    q1 = np.asarray([float(first[name]) for name in JOINT_VECTOR_NAMES])
    q2 = np.asarray([float(second[name]) for name in JOINT_VECTOR_NAMES])
    delta = float(np.max(np.abs(q2 - q1)))
    if delta > max_joint_delta_rad:
        raise RuntimeError(
            f"robot still moving: max joint change {delta:.5f} rad > {max_joint_delta_rad:.5f} rad"
        )
    rotvec = np.asarray([float(second[key]) for key in ("ee.wx", "ee.wy", "ee.wz")])
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T_base_tcp[:3, 3] = [float(second[key]) for key in ("ee.x", "ee.y", "ee.z")]
    return T_base_tcp, q2.tolist()


def _observations_from_records(records: list[dict[str, Any]]) -> list[EyeHandObservation]:
    observations: list[EyeHandObservation] = []
    for record in records:
        T_base_tcp = _matrix_from_payload(record["T_base_tcp"], "T_base_tcp")
        for camera, camera_record in record.get("cameras", {}).items():
            for detection in camera_record.get("detections", []):
                observations.append(
                    EyeHandObservation(
                        pose_index=int(record["capture_index"]),
                        camera=str(camera),
                        tag_id=int(detection["tag_id"]),
                        T_base_tcp=T_base_tcp,
                        T_world_tag=_matrix_from_payload(detection["T_world_tag"], "T_world_tag"),
                        reprojection_rmse_px=float(detection["reprojection_rmse_px"]),
                    )
                )
    return observations


def _ready(observations: list[EyeHandObservation]) -> tuple[bool, str]:
    poses = len({obs.pose_index for obs in observations})
    cameras = len({obs.camera for obs in observations})
    per_tag = {
        tag_id: len({obs.pose_index for obs in observations if obs.tag_id == tag_id})
        for tag_id in TAG_IDS
    }
    ready = (
        poses >= MIN_ROBOT_POSES
        and cameras >= MIN_CAMERAS
        and min(per_tag.values(), default=0) >= MIN_POSES_PER_TAG
    )
    return ready, f"poses={poses}/{MIN_ROBOT_POSES}, cameras={cameras}/{MIN_CAMERAS}, tag poses={per_tag}"


def _save_capture(
    run_dir: Path,
    records: list[dict[str, Any]],
    results: dict[str, CameraResult],
    cluster: OnlineSyncCluster,
    robot: Any,
    settle_s: float,
    max_joint_delta_rad: float,
) -> dict[str, Any]:
    if not any(result.detections for result in results.values()):
        raise RuntimeError("no valid tag 56/57 detection in the displayed cluster")
    T_base_tcp, joints = _robot_sample(robot, settle_s, max_joint_delta_rad)
    capture_index = len(records)
    image_dir = run_dir / "captures" / f"capture_{capture_index:03d}"
    temp_image_dir = image_dir.with_name(image_dir.name + ".tmp")
    shutil.rmtree(temp_image_dir, ignore_errors=True)
    temp_image_dir.mkdir(parents=True, exist_ok=False)
    camera_payload: dict[str, Any] = {}
    try:
        for camera, result in sorted(results.items()):
            image_path = image_dir / f"{camera}.jpg"
            temp_image_path = temp_image_dir / image_path.name
            if not cv2.imwrite(
                str(temp_image_path), result.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to save {image_path}")
            camera_payload[camera] = {
                "image": str(image_path),
                "valid": bool(result.detections),
                "detections": result.detections,
                "error": result.error,
            }
        temp_image_dir.replace(image_dir)
    except Exception:
        shutil.rmtree(temp_image_dir, ignore_errors=True)
        raise
    record = {
        "capture_index": capture_index,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sync": {
            "publish_seq": cluster.publish_seq,
            "logical_frame_index": cluster.logical_frame_index,
            "sync_timestamp_ns": cluster.sync_timestamp_ns,
            "max_delta_ns": cluster.max_delta_ns,
        },
        "T_base_tcp": matrix_payload(T_base_tcp),
        "joint_values_rad": joints,
        "cameras": camera_payload,
    }
    records.append(record)
    _write_json(run_dir / "captures.json", {"schema": "p0_two_marker_captures/v1", "records": records})
    return record


def _solve_and_activate(
    run_dir: Path,
    root: Path,
    records: list[dict[str, Any]],
    world_id: str,
    intrinsics_summary: Path,
    extrinsics_summary: Path,
    p0_source: Path,
    fixed_tcp: dict[int, np.ndarray] | None,
    fixed_tcp_source: Path | None,
    solver_workers: int,
) -> Path:
    observations = _observations_from_records(records)
    if fixed_tcp:
        print("[SOLVE] reusing fixed marker->EE transforms; solving only T_world_base", flush=True)
        solution = solve_base_with_fixed_tcp_tags(observations, fixed_tcp, workers=solver_workers)
        solve_mode = "fixed_marker_to_ee"
    else:
        print("[SOLVE] first run: jointly solving T_world_base and marker->EE transforms", flush=True)
        solution = solve_eye_hand(observations)
        solve_mode = "joint_base_and_marker_to_ee"
    passed, reasons = assess_solution(solution)
    T_world_base = solution.pop("T_world_base")
    T_tcp_tags = solution.pop("T_tcp_tags")
    calibration_path = run_dir / "calibration.json"
    payload = {
        "schema": "p0_two_marker_calibration/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed_quality_gate",
        "world_frame_id": world_id,
        "frame_equation": "T_world_tag = T_world_base @ T_base_tcp @ T_tcp_tag",
        "tcp_frame": "fr3_ee",
        "solve_mode": solve_mode,
        "marker": {
            "family": "tag36h11",
            "ids": list(TAG_IDS),
            "marker_size_m": TAG_SIZE_M,
            "backing_size_m": BACKING_SIZE_M,
        },
        "inputs": {
            "captures_json": str(run_dir / "captures.json"),
            "intrinsics_summary": str(intrinsics_summary),
            "intrinsics_sha256": _sha256(intrinsics_summary),
            "extrinsics_summary": str(extrinsics_summary),
            "extrinsics_sha256": _sha256(extrinsics_summary),
            "fixed_marker_to_ee_source": str(fixed_tcp_source) if fixed_tcp_source else None,
            "fixed_marker_to_ee_source_sha256": (
                _sha256(fixed_tcp_source) if fixed_tcp_source else None
            ),
        },
        "T_world_base": matrix_payload(T_world_base),
        "T_tcp_tag": {str(tag_id): matrix_payload(T) for tag_id, T in T_tcp_tags.items()},
        "quality": solution,
        "quality_gate_reasons": reasons,
        "valid_captures_per_camera": {
            camera: sum(
                bool(record.get("cameras", {}).get(camera, {}).get("valid")) for record in records
            )
            for camera in sorted(
                {
                    camera
                    for record in records
                    for camera in record.get("cameras", {})
                }
            )
        },
    }
    _write_json(calibration_path, payload)
    if not passed:
        raise RuntimeError("calibration quality gate failed: " + "; ".join(reasons))

    candidate_path = run_dir / "p0_relocalized_timed_plan.json"
    create_retargeted_p0_plan(p0_source, T_world_base, candidate_path, calibration_path)
    _write_json(
        root / "active.json",
        {
            "schema": "p1_simple_eye_hand_active/v1",
            "producer": "p0_two_marker_calibration",
            "activated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "calibration_path": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "plan_path": str(candidate_path),
            "plan_sha256": _sha256(candidate_path),
        },
    )
    print(f"[DONE] passing calibration activated: {calibration_path}", flush=True)
    return calibration_path


def _fixed_tcp_from_active(active: tuple[Path, dict[str, Any], dict[str, Any]] | None) -> dict[int, np.ndarray] | None:
    if active is None:
        return None
    raw = active[1].get("T_tcp_tag", {})
    if not isinstance(raw, dict):
        return None
    result = {int(tag_id): _matrix_from_payload(value, f"T_tcp_tag[{tag_id}]") for tag_id, value in raw.items()}
    return result if set(TAG_IDS).issubset(result) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", choices=("ask", "reuse", "recalibrate"), default="ask")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize connecting FR3 in zero-stiffness teaching mode",
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--robot-ip", default="192.168.11.102")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--recorder-config", type=Path, default=DEFAULT_RECORDER_CONFIG)
    parser.add_argument("--intrinsics-summary", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics-summary", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--p0-source", type=Path, default=DEFAULT_P0_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-per-camera", type=int, default=15)
    parser.add_argument("--detection-workers", type=int, default=7)
    parser.add_argument("--solver-workers", type=int, default=7)
    parser.add_argument("--frame-bus-every-n", type=int, default=6)
    parser.add_argument("--detection-scale", type=float, default=0.5)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=3.0)
    parser.add_argument("--settle-time-s", type=float, default=0.12)
    parser.add_argument("--max-settle-joint-delta-rad", type=float, default=0.003)
    parser.add_argument("--enable-hardware-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    active = _load_active(root)
    if _choose_existing(args.existing, active):
        print("[DONE] existing calibration retained; cameras and robot were not connected", flush=True)
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "would_recalibrate": True,
                    "output_root": str(root),
                    "fixed_marker_to_ee_available": _fixed_tcp_from_active(active) is not None,
                    "recorder_config": str(args.recorder_config.expanduser().resolve()),
                    "intrinsics_summary": str(args.intrinsics_summary.expanduser().resolve()),
                    "extrinsics_summary": str(args.extrinsics_summary.expanduser().resolve()),
                },
                indent=2,
            )
        )
        return 0
    if not args.execute or args.confirmation != "P0_TWO_MARKER_TEACHING":
        parser.error("recalibration requires --execute --confirmation P0_TWO_MARKER_TEACHING")
    if args.target_per_camera <= 0 or args.detection_workers <= 0 or args.solver_workers <= 0:
        parser.error("target/count worker options must be positive")

    print(
        "[SAFETY] FR3 will enter zero-stiffness teaching mode. Move it by hand, stop completely, "
        "then press Enter in the camera window. Keep the physical E-stop ready.",
        flush=True,
    )
    run_dir = root / f"manual_run_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_bus_dir = Path(f"/dev/shm/p0_two_marker_{os.getpid()}")
    recorder_config = _make_recorder_config(
        args.recorder_config.expanduser().resolve(), run_dir, frame_bus_dir, args.frame_bus_every_n
    )
    intrinsics_summary = args.intrinsics_summary.expanduser().resolve()
    extrinsics_summary = args.extrinsics_summary.expanduser().resolve()
    intrinsics = _intrinsics_index(intrinsics_summary)
    world_id, world_cameras = _world_cameras(extrinsics_summary)
    fixed_tcp = _fixed_tcp_from_active(active)
    robot = _make_robot(args.robot_ip, args.urdf.expanduser().resolve())
    recorder: ThorRecorderClient | None = None
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    should_solve = False
    try:
        recorder = _start_recorder(recorder_config, skip_hardware_sync=not args.enable_hardware_sync)
        frame_client = ThorOnlineSyncFrameClient(frame_bus_dir)
        cluster = frame_client.get_latest(timeout_s=30.0)
        if cluster is None:
            raise RuntimeError(f"no synchronized frame cluster appeared under {frame_bus_dir}")
        robot.connect()
        _activate_teaching_mode(robot)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1440, 900)
        last_seq = -1
        results: dict[str, CameraResult] = {}
        ui_message = "Move FR3 by hand, stop completely, then press Enter"
        with ThreadPoolExecutor(
            max_workers=args.detection_workers, thread_name_prefix="apriltag-camera"
        ) as pool:
            while True:
                next_cluster = frame_client.get_latest(timeout_s=0.2, min_publish_seq=last_seq + 1)
                if next_cluster is not None:
                    cluster = next_cluster
                    last_seq = cluster.publish_seq
                    results = _process_cluster(
                        cluster,
                        intrinsics,
                        world_cameras,
                        pool,
                        args.detection_scale,
                        args.max_reprojection_rmse_px,
                    )
                    for camera in results:
                        counts.setdefault(camera, 0)
                if results:
                    cv2.imshow(
                        WINDOW_NAME,
                        _draw_mosaic(results, counts, args.target_per_camera, ui_message),
                    )
                key = cv2.waitKey(1) & 0xFF
                if key in (10, 13):
                    try:
                        record = _save_capture(
                            run_dir,
                            records,
                            results,
                            cluster,
                            robot,
                            args.settle_time_s,
                            args.max_settle_joint_delta_rad,
                        )
                        valid = [camera for camera, item in record["cameras"].items() if item["valid"]]
                        for camera in valid:
                            counts[camera] += 1
                        observations = _observations_from_records(records)
                        _, readiness = _ready(observations)
                        ui_message = f"Captured #{len(records)}: {readiness}"
                        print(
                            f"[CAPTURE {len(records):03d}] valid cameras={valid}; counts={counts}; {readiness}",
                            flush=True,
                        )
                    except Exception as exc:
                        ui_message = f"REJECTED: {exc}"
                        print(f"[REJECTED] {exc}", flush=True)
                elif key == ord("q"):
                    ready, readiness = _ready(_observations_from_records(records))
                    if ready:
                        print(f"[READY] {readiness}", flush=True)
                        should_solve = True
                        break
                    ui_message = f"NOT READY: {readiness}"
                    print(f"[NOT READY] {readiness}; continue capture or press f to force", flush=True)
                elif key == ord("f"):
                    print("[FORCE] attempting solve with current accepted observations", flush=True)
                    should_solve = True
                    break
                elif key == 27:
                    print("[ABORT] active calibration was not changed", flush=True)
                    break
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            try:
                if recorder is not None:
                    recorder.stop()
            finally:
                shutil.rmtree(frame_bus_dir, ignore_errors=True)
                shutil.rmtree(recorder_config.parent, ignore_errors=True)

    if not should_solve:
        return 130
    _solve_and_activate(
        run_dir,
        root,
        records,
        world_id,
        intrinsics_summary,
        extrinsics_summary,
        args.p0_source.expanduser().resolve(),
        fixed_tcp,
        active[0] if fixed_tcp and active is not None else None,
        args.solver_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
