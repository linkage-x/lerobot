#!/usr/bin/env python3
"""Relocalize a moved FR3 base from wrist-mounted AprilTags 56 and 57.

The current production camera extrinsics define the world frame.  The capture
stage replays a diverse subset of existing, operator-taught calibration poses;
at every stop Thor records the cameras and the FR3 records its measured TCP.
The solve stage estimates ``T_world_base`` and the two fixed ``T_tcp_tag``
transforms.  A passing solve also creates (but does not activate) a P0 joint
plan retargeted into the new robot base.

This is not a collision checker.  ``run --execute`` moves a real robot and is
therefore deliberately guarded by an exact confirmation string.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from p1_eye_hand_core import (  # noqa: E402
    EyeHandObservation,
    assess_solution,
    invert_transform,
    matrix_payload,
    select_diverse_pose_records,
    solve_eye_hand,
)
DEFAULT_POSES = Path("/home/nvidia/lerobot/outputs/calibration/thor_gmsl2_extrinisics_robot_base_0720")
DEFAULT_INTRINSICS = Path(
    "/home/nvidia/lerobot/outputs/calibration/thor_gmsl2_selfcal_0804_fisheye_intrinsics/summary.json"
)
DEFAULT_EXTRINSICS = Path("/home/nvidia/lerobot/outputs/calibration/calib_20260902_103833_extrinsics/summary.json")
DEFAULT_P0_SOURCE = Path("/home/nvidia/box_api/replay_p0_once_20260908/timed_plan.json")
DEFAULT_ROOT = Path("/home/nvidia/lerobot/outputs/calibration/p1_simple_eye_hand_calibration")
CAPTURE_TEMPLATE = REPO_ROOT / "third_party/opencv_kalibr/fr3_calibration/host/execute_pose_and_capture_thor_gmsl2_apriltag.host.yaml"
CAPTURE_SCRIPT = REPO_ROOT / "third_party/opencv_kalibr/fr3_calibration/execute_pose_and_capture_thor_gmsl2.py"
PREPOSITION_SCRIPT = REPO_ROOT / "tools/thor/p1_native_preposition.py"
TAG_IDS = (56, 57)
TAG_SIZE_M = 0.055
BACKING_SIZE_M = 0.070
# Read from the stationary FR3 at the operator-identified table-contact pose on
# 2026-09-10. P1 database targets must keep the measured TCP at least 150 mm
# above that height. Both values use the FR3-base ``fr3_ee`` pose contract.
TABLE_CONTACT_TCP_Z_M = 0.10046900307468043
MIN_TCP_CLEARANCE_M = 0.150
MIN_TCP_Z_M = TABLE_CONTACT_TCP_Z_M + MIN_TCP_CLEARANCE_M
MIN_VISIBLE_CAMERAS = 4
CAMERA_PROXIMITY_POOL_MULTIPLIER = 2.5
PANDA_FR3_COMPAT_LOWER = np.asarray([-2.7437, -1.7628, -2.8973, -3.0421, -2.8065, 0.5445, -2.8973])
PANDA_FR3_COMPAT_UPPER = np.asarray([2.7437, 1.7628, 2.8973, -0.1518, 2.8065, 3.7525, 2.8973])


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(stage: str, message: str, **extra: Any) -> None:
    print("P1_STATUS " + json.dumps({"stage": stage, "message": message, **extra}, ensure_ascii=False), flush=True)


def _load_pose_records(path: Path) -> list[dict]:
    if path.is_dir() and (path / "summary.json").is_file():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        camera_centers: dict[str, np.ndarray] = {}
        for camera, camera_summary in summary.get("cameras", {}).items():
            matrix = np.asarray(
                camera_summary.get("base_to_camera", {}).get("matrix_4x4", []), dtype=float
            )
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                camera_centers[str(camera)] = matrix[:3, 3]
        dataset_root = Path(str(summary.get("episode", "")))
        if not dataset_root.is_dir():
            dataset_root = REPO_ROOT / "outputs/datasets" / dataset_root.name
        parquet_path = dataset_root / "data/chunk-000/file-000.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"P1 extrinsics pose database parquet not found: {parquet_path}")

        visibility: dict[int, dict[str, float]] = {}
        for detection_path in sorted(path.glob("cam_*/apriltag_detections.csv")):
            camera = detection_path.parent.name
            with detection_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("detected") != "1" or row.get("reject_reason") != "ok":
                        continue
                    visibility.setdefault(int(row["frame_index"]), {})[camera] = float(
                        row["reprojection_rmse_px"]
                    )

        records: list[dict] = []
        for row in pq.read_table(parquet_path).to_pylist():
            state = np.asarray(row["observation.state"], dtype=float)
            joints = np.asarray(row["observation.joints"], dtype=float)
            if state.shape != (8,) or joints.shape != (7,) or not np.isfinite(np.r_[state, joints]).all():
                raise ValueError(f"Invalid P1 database row at frame {row.get('frame_index')}")
            rotation = Rotation.from_quat(state[3:7])
            frame_index = int(row["frame_index"])
            visible = visibility.get(frame_index, {})
            visible_camera_distances = {
                camera: float(np.linalg.norm(state[:3] - camera_centers[camera]))
                for camera in visible
                if camera in camera_centers
            }
            nearest_camera = (
                min(visible_camera_distances, key=visible_camera_distances.get)
                if visible_camera_distances
                else None
            )
            records.append(
                {
                    "index": frame_index + 1,
                    "database_frame_index": frame_index,
                    "pose": {
                        "position_xyz_m": state[:3].tolist(),
                        "rotvec_xyz_rad": rotation.as_rotvec().tolist(),
                        "quaternion_xyzw": rotation.as_quat().tolist(),
                        "euler_xyz_rad": rotation.as_euler("xyz").tolist(),
                        "rotation_matrix": rotation.as_matrix().tolist(),
                    },
                    "joint_values_rad": joints.tolist(),
                    "gripper_pos": float(state[7]),
                    "extrinsics_importance": {
                        "visible_cameras": sorted(visible),
                        "visible_camera_count": len(visible),
                        "mean_reprojection_rmse_px": (
                            float(np.mean(list(visible.values()))) if visible else None
                        ),
                        "visible_camera_distance_m": visible_camera_distances,
                        "nearest_visible_camera": nearest_camera,
                        "nearest_visible_camera_distance_m": (
                            visible_camera_distances[nearest_camera] if nearest_camera is not None else None
                        ),
                    },
                }
            )
        if not records:
            raise ValueError(f"No pose/joint rows in P1 extrinsics database: {parquet_path}")
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("records", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No teaching pose records in {path}")
    return [dict(row) for row in rows]


def select_extrinsics_pose_records(records: list[dict], count: int) -> list[dict]:
    """Choose high, camera-near, diverse poses with strong historical visibility."""
    candidates = [
        record
        for record in records
        if int(record.get("extrinsics_importance", {}).get("visible_camera_count", 0))
        >= MIN_VISIBLE_CAMERAS
        and float(record.get("pose", {}).get("position_xyz_m", [0.0, 0.0, -np.inf])[2])
        >= MIN_TCP_Z_M
        and record.get("extrinsics_importance", {}).get("nearest_visible_camera_distance_m")
        is not None
    ]
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} database poses have TCP z >= {MIN_TCP_Z_M:.6f} m and were "
            f"visible in at least {MIN_VISIBLE_CAMERAS} cameras; need {count}"
        )
    # Restrict diversity sampling to the camera-nearest portion of the valid
    # database. A 2.5x pool preserves broad TCP translation/orientation
    # excitation and rare-camera coverage while rejecting the farthest poses.
    proximity_pool_count = min(
        len(candidates), max(count, int(np.ceil(count * CAMERA_PROXIMITY_POOL_MULTIPLIER)))
    )
    proximity_pool = sorted(
        candidates,
        key=lambda record: (
            float(record["extrinsics_importance"]["nearest_visible_camera_distance_m"]),
            -int(record["extrinsics_importance"]["visible_camera_count"]),
            float(record["extrinsics_importance"]["mean_reprojection_rmse_px"]),
            int(record.get("database_frame_index", record.get("index", 0))),
        ),
    )[:proximity_pool_count]
    proximity_pool.sort(
        key=lambda record: int(record.get("database_frame_index", record.get("index", 0)))
    )
    selected = select_diverse_pose_records(proximity_pool, count)
    cameras = sorted(
        {
            camera
            for record in records
            for camera in record.get("extrinsics_importance", {}).get("visible_cameras", [])
        }
    )
    minimum_per_camera = max(3, count // 10)
    coverage = {
        camera: sum(
            camera in record.get("extrinsics_importance", {}).get("visible_cameras", [])
            for record in selected
        )
        for camera in cameras
    }
    weak = {camera: seen for camera, seen in coverage.items() if seen < minimum_per_camera}
    if weak:
        raise RuntimeError(f"Selected database poses have insufficient per-camera coverage: {weak}")
    for ordinal, record in enumerate(selected, start=1):
        joints = np.asarray(record.get("joint_values_rad"), dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"Selected database pose #{ordinal} has invalid joints")
        if np.any(joints < PANDA_FR3_COMPAT_LOWER) or np.any(joints > PANDA_FR3_COMPAT_UPPER):
            raise RuntimeError(
                f"Selected database pose #{ordinal} exceeds the FR3/Panda-controller common joint range"
            )
    return selected


def prepare_capture(run_dir: Path, source_poses: Path, count: int) -> tuple[Path, Path]:
    source_records = _load_pose_records(source_poses)
    records = (
        select_extrinsics_pose_records(source_records, count)
        if source_poses.is_dir()
        else select_diverse_pose_records(source_records, count)
    )
    selected_path = run_dir / "selected_teaching_pose_records.json"
    # execute_pose_and_capture intentionally rejects a list root so pose files
    # retain provenance.  Keep that contract for the generated subset too.
    corenetic_urdf = (
        REPO_ROOT
        / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
    ).resolve()
    if not corenetic_urdf.is_file():
        raise FileNotFoundError(f"P1 FR3 Corenetic URDF not found: {corenetic_urdf}")
    _write_json(
        selected_path,
        {
            "schema_version": 1,
            "generated_by": "P1_simple_eye_hand_calibration",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_pose_database": str(source_poses),
            "selection": "tcp_z_filter_then_nearest_visible_camera_pool_then_diverse_tcp_original_database_order",
            "selection_policy": {
                "table_contact_tcp_z_m": TABLE_CONTACT_TCP_Z_M,
                "minimum_tcp_clearance_m": MIN_TCP_CLEARANCE_M,
                "minimum_tcp_z_m": MIN_TCP_Z_M,
                "minimum_visible_cameras": MIN_VISIBLE_CAMERAS,
                "camera_proximity_metric": "tcp_to_nearest_historically_visible_camera_center_m",
                "camera_proximity_pool_multiplier": CAMERA_PROXIMITY_POOL_MULTIPLIER,
                "camera_proximity_pool_count": min(
                    len(source_records),
                    max(count, int(np.ceil(count * CAMERA_PROXIMITY_POOL_MULTIPLIER))),
                ),
            },
            "pose_contract": "exact database observation.state pose and observation.joints; no pitch transform or IK",
            "records": records,
        },
    )
    config = yaml.safe_load(CAPTURE_TEMPLATE.read_text(encoding="utf-8"))
    # The legacy AprilTag capture template names an arm-only URDF that is not
    # present in this checkout.  The production P0 Corenetic model contains
    # the same ``fr3_ee`` contract as a zero-offset alias of the FR3 flange.
    # Joint-space teach poses remain unchanged; this defines the measured TCP
    # used by the eye-hand solve without introducing a Franka-hand transform.
    config["robot"]["urdf_path"] = str(corenetic_urdf)
    config["robot"]["target_frame_name"] = "fr3_ee"
    # Eye-hand capture needs only the arm.  Keep gripper commands in-process
    # and inert so this calibration cannot change the physical gripper width.
    config["robot"]["gripper_backend"] = "mock"
    config["input"] = {"key": run_dir.name, "json_path": str(selected_path)}
    # The FR3-native preposition stage already reaches the first selected pose.
    # Do not start the legacy Panda-limited controller from the initial state.
    config["execution"]["home_before_start"] = False
    config["execution"]["control_mode"] = "joint_space"
    config["execution"]["record_inserted_interpolation_waypoints"] = False
    config["execution"]["max_command_steps"] = 240
    config["execution"]["joint_tolerance_rad"] = 0.02
    config["execution"]["fail_on_unreached_pose"] = True
    config["execution"]["max_records"] = int(count)
    config["execution"]["report_json_path"] = str(run_dir / "capture_report.json")
    config["dataset"]["repo_id"] = f"local/{run_dir.name}"
    config["dataset"]["single_task"] = "P1 simple eye-hand calibration tags 56 and 57"
    config["dataset"]["root"] = str(run_dir / "capture")
    config["thor"]["raw_dataset_root"] = str(run_dir / "capture_raw")
    # Each sample is captured only after the arm has stopped and is pose is
    # constant for the whole short episode.  Cross-camera PWM triggering adds
    # no constraint here, and some Thor boots intentionally do not expose the
    # pwm-gpio chip.  Avoid making this static-pose calibration depend on it.
    config["thor"]["skip_hardware_sync"] = True
    # The real capture open is the authoritative camera check.  Avoid an extra
    # 11-camera Argus probe/open/close cycle immediately before it; operators
    # use recover_argus.sh once after boot instead.
    config["thor"]["skip_argus_probe"] = True
    config["output"]["overwrite_existing"] = False
    config["output"]["confirm_overwrite_if_exists"] = False
    config_path = run_dir / "capture_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return selected_path, config_path


def run_capture(config_path: Path) -> Path:
    cmd = [sys.executable, str(CAPTURE_SCRIPT), f"--config_path={config_path}"]
    _status("capturing", "FR3 pose-and-stop capture started", command=cmd)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(config["dataset"]["root"])


def run_native_preposition(selected_path: Path) -> None:
    python = Path("/home/nvidia/Code/infer/.venv-fr3/bin/python")
    bundle_python = Path("/home/nvidia/box_api/replay_p0_native_arm_only_20260908/python")
    cmeel_python = Path(
        "/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/"
        "cmeel.prefix/lib/python3.12/site-packages"
    )
    ld_library_path = ":".join(
        (
            "/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib",
            "/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib",
            "/usr/local/lib",
        )
    )
    command = [
        "sudo", "-n", "env",
        f"PYTHONPATH={bundle_python}:{cmeel_python}",
        f"LD_LIBRARY_PATH={ld_library_path}",
        str(python), str(PREPOSITION_SCRIPT), str(selected_path),
        "--confirmation", "P1_MOVE_FR3",
    ]
    _status("prepositioning", "Slow FR3-native move to first selected pose started", command=command)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _intrinsics_index(summary_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("cameras", []):
        if not isinstance(item, dict) or str(item.get("status", "")).lower() != "ok":
            continue
        camera = str(item.get("camera_name", ""))
        raw = Path(str(item.get("intrinsics_json", "")))
        if not raw.exists() and "per_camera" in raw.parts:
            raw = summary_path.parent.joinpath(*raw.parts[raw.parts.index("per_camera") :])
        data = json.loads(raw.read_text(encoding="utf-8"))
        out[camera] = {
            "path": str(raw),
            "K": np.asarray(data["camera_matrix"], dtype=float),
            "D": np.asarray(data["dist_coeffs"], dtype=float).reshape(-1, 1),
            "model": str(data.get("model", "rational")).lower(),
            "width": int(data.get("image_width", data.get("width", 0))),
            "height": int(data.get("image_height", data.get("height", 0))),
        }
    if not out:
        raise RuntimeError(f"No usable intrinsics in {summary_path}")
    return out


def _world_cameras(summary_path: Path) -> tuple[str, dict[str, np.ndarray]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    world = payload.get("world", {})
    world_id = str(world.get("world_frame_id", ""))
    cameras = payload.get("joint_solution", {}).get("cameras", {})
    result: dict[str, np.ndarray] = {}
    for name, item in cameras.items():
        matrix = np.asarray(item.get("base_to_camera", {}).get("matrix_4x4", []), dtype=float)
        if matrix.shape == (4, 4):
            result[str(name)] = matrix
    if not world_id or not result:
        raise RuntimeError(f"Current extrinsics lacks canonical-world camera poses: {summary_path}")
    return world_id, result


def _load_capture_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No capture parquet under {dataset_root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path).to_pydict()
        for index in range(len(table["index"])):
            rows.append({key: value[index] for key, value in table.items()})
    return rows


def _tcp_pose(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["observation.state"], dtype=float).reshape(-1)
    if state.size < 7 or not np.isfinite(state[:7]).all():
        raise ValueError("Capture row has invalid measured TCP state")
    T = np.eye(4)
    T[:3, 3] = state[:3]
    T[:3, :3] = Rotation.from_quat(state[3:7]).as_matrix()
    return T


def _frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        raise RuntimeError(f"Cannot decode frame {index} from {video}")
    return image


def _tag_pose(corners: np.ndarray, intr: dict[str, Any], image_shape: tuple[int, int]) -> tuple[np.ndarray, float] | None:
    h = TAG_SIZE_M / 2.0
    obj = np.asarray([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    K = intr["K"].copy()
    ih, iw = image_shape
    if intr["width"] and intr["height"] and (iw != intr["width"] or ih != intr["height"]):
        K[0, :] *= iw / intr["width"]
        K[1, :] *= ih / intr["height"]
    D = intr["D"]
    fisheye = intr["model"] in {"fisheye", "equidistant", "opencv_fisheye"}
    if fisheye:
        solve_points = cv2.fisheye.undistortPoints(points.reshape(-1, 1, 2), K, D.reshape(4, 1)).reshape(-1, 2)
        solve_K, solve_D = np.eye(3), np.zeros(5)
    else:
        solve_points, solve_K, solve_D = points, K, D
    ok, rvec, tvec = cv2.solvePnP(obj, solve_points, solve_K, solve_D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0:
        return None
    if fisheye:
        projected, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), rvec, tvec, K, D.reshape(4, 1))
    else:
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - points) ** 2, axis=1))))
    Rm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T, rmse


def collect_observations(
    dataset_root: Path,
    intrinsics_summary: Path,
    extrinsics_summary: Path,
    *,
    max_reprojection_rmse_px: float = 3.0,
) -> tuple[str, list[EyeHandObservation], list[dict[str, Any]]]:
    rows = _load_capture_rows(dataset_root)
    intrinsics = _intrinsics_index(intrinsics_summary)
    world_id, world_cameras = _world_cameras(extrinsics_summary)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector: Any = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, "DetectorParameters_create")
            else cv2.aruco.DetectorParameters()
        )
        detector = {"dictionary": dictionary, "parameters": parameters}
    observations: list[EyeHandObservation] = []
    diagnostics: list[dict[str, Any]] = []
    for camera in sorted(set(intrinsics) & set(world_cameras)):
        videos = sorted((dataset_root / "videos" / f"observation.images.{camera}").glob("chunk-*/*.mp4"))
        if not videos:
            continue
        # The P1 capture writes one video containing one representative frame
        # per stopped robot pose.
        video = videos[0]
        for pose_index, row in enumerate(rows):
            image = _frame(video, pose_index)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if isinstance(detector, dict):
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, detector["dictionary"], parameters=detector["parameters"]
                )
            else:
                corners, ids, _ = detector.detectMarkers(gray)
            found = 0
            if ids is not None:
                for marker_corners, marker_id_raw in zip(corners, ids.reshape(-1), strict=True):
                    marker_id = int(marker_id_raw)
                    if marker_id not in TAG_IDS:
                        continue
                    result = _tag_pose(marker_corners, intrinsics[camera], gray.shape)
                    if result is None:
                        continue
                    T_camera_tag, rmse = result
                    if rmse > max_reprojection_rmse_px:
                        continue
                    observations.append(
                        EyeHandObservation(
                            pose_index=pose_index,
                            camera=camera,
                            tag_id=marker_id,
                            T_base_tcp=_tcp_pose(row),
                            T_world_tag=world_cameras[camera] @ T_camera_tag,
                            reprojection_rmse_px=rmse,
                        )
                    )
                    found += 1
            diagnostics.append({"pose_index": pose_index, "camera": camera, "accepted_tags": found})
    return world_id, observations, diagnostics


def _episode_knots(source: dict[str, Any], record: dict[str, Any]) -> np.ndarray:
    if "joint_knots" in record:
        return np.asarray(record["joint_knots"], dtype=float)
    coefficients = np.asarray(record["coeff_descending_unit_interval"], dtype=float)
    return np.concatenate((coefficients[:, -1, :], coefficients[-1].sum(axis=0)[None]), axis=0)[:, :7]


def _chain_fk(chain: dict[str, Any], q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for origin, axis, angle in zip(chain["origins"], chain["axes"], q, strict=True):
        T = T @ np.asarray(origin, dtype=float)
        joint = np.eye(4)
        joint[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=float) * float(angle)).as_matrix()
        T = T @ joint
    return T @ np.asarray(chain["tail"], dtype=float)


def _chain_fk_jacobian(chain: dict[str, Any], q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    axes_world: list[np.ndarray] = []
    origins_world: list[np.ndarray] = []
    for origin, axis, angle in zip(chain["origins"], chain["axes"], q, strict=True):
        T = T @ np.asarray(origin, dtype=float)
        origins_world.append(T[:3, 3].copy())
        axes_world.append(T[:3, :3] @ np.asarray(axis, dtype=float))
        joint = np.eye(4)
        joint[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=float) * float(angle)).as_matrix()
        T = T @ joint
    T = T @ np.asarray(chain["tail"], dtype=float)
    J = np.zeros((6, len(axes_world)), dtype=float)
    for index, (axis, origin) in enumerate(zip(axes_world, origins_world, strict=True)):
        J[:3, index] = np.cross(axis, T[:3, 3] - origin)
        J[3:, index] = axis
    return T, J


def _chain_ik(chain: dict[str, Any], seed: np.ndarray, desired: np.ndarray) -> np.ndarray:
    lower = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
    upper = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
    seed = np.clip(np.asarray(seed, dtype=float), lower + 1e-8, upper - 1e-8)

    q = seed.copy()
    for _ in range(120):
        actual, J = _chain_fk_jacobian(chain, q)
        trans_error = desired[:3, 3] - actual[:3, 3]
        rot_error = Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()
        if np.linalg.norm(trans_error) <= 2e-5 and np.linalg.norm(rot_error) <= np.deg2rad(0.01):
            return q
        error = np.r_[trans_error, rot_error]
        damping = 2e-4
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), error)
        dq = np.clip(dq, -0.08, 0.08)
        q = np.clip(q + dq, lower + 1e-8, upper - 1e-8)
    actual = _chain_fk(chain, q)
    delta = invert_transform(desired) @ actual
    position_mm = np.linalg.norm(delta[:3, 3]) * 1000.0
    orientation_deg = np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi
    raise RuntimeError(
        f"P0 retarget IK did not converge: position={position_mm:.3f} mm orientation={orientation_deg:.3f} deg"
    )


def create_retargeted_p0_plan(
    source_path: Path,
    T_world_base_new: np.ndarray,
    output_path: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    chain = source.get("chain", {})
    if chain.get("tcp_frame") != "corenetic_gripper_ee" or len(chain.get("origins", [])) != 7:
        raise ValueError("P0 source does not carry the expected corenetic_gripper_ee kinematic chain")
    T_base_new_world = invert_transform(T_world_base_new)
    episodes: list[dict[str, Any]] = []
    max_joint_step = 0.0
    source_max_joint_step = 0.0
    max_position_error_mm = 0.0
    max_orientation_error_deg = 0.0
    for record in source.get("episodes", []):
        old_q = _episode_knots(source, record)
        if len(old_q) > 1:
            source_max_joint_step = max(source_max_joint_step, float(np.max(np.abs(np.diff(old_q, axis=0)))))
        new_q: list[list[float]] = []
        for q in old_q:
            T_world_tcp = _chain_fk(chain, q)  # old world/base alignment is identity
            desired = T_base_new_world @ T_world_tcp
            # The original knot is the closest known branch seed.  Seeding
            # solely from the prior solution can make a long, almost-static
            # segment consume the numerical budget even when the original
            # knot is already the exact solution (identity base move).
            solved = _chain_ik(chain, q, desired)
            actual = _chain_fk(chain, solved)
            delta = invert_transform(desired) @ actual
            max_position_error_mm = max(max_position_error_mm, float(np.linalg.norm(delta[:3, 3]) * 1000.0))
            max_orientation_error_deg = max(
                max_orientation_error_deg,
                float(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi),
            )
            if new_q:
                max_joint_step = max(max_joint_step, float(np.max(np.abs(solved - np.asarray(new_q[-1])))))
            new_q.append(solved.tolist())
        episodes.append({"episode": int(record["episode"]), "frames": len(new_q), "joint_knots": new_q})
    quality = {
        "max_fk_position_error_mm": max_position_error_mm,
        "max_fk_orientation_error_deg": max_orientation_error_deg,
        "max_consecutive_joint_step_rad": max_joint_step,
        "source_max_consecutive_joint_step_rad": source_max_joint_step,
    }
    continuity_limit = max(0.10, 5.0 * source_max_joint_step)
    failures = []
    if max_position_error_mm > 1.0:
        failures.append(f"IK FK position error {max_position_error_mm:.3f} mm > 1.000 mm")
    if max_orientation_error_deg > 0.2:
        failures.append(f"IK FK orientation error {max_orientation_error_deg:.3f} deg > 0.200 deg")
    if max_joint_step > continuity_limit:
        failures.append(f"IK joint discontinuity {max_joint_step:.4f} rad > {continuity_limit:.4f} rad")
    quality["passed"] = not failures
    quality["failure_reasons"] = failures
    quality["joint_continuity_limit_rad"] = continuity_limit
    if failures:
        raise RuntimeError("P0 retarget numerical quality gate failed: " + "; ".join(failures))
    plan = {
        "schema": "p0_relocalized_joint_plan/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "calibration_path": str(calibration_path),
        "T_world_base": matrix_payload(T_world_base_new),
        "old_T_world_base_assumption": matrix_payload(np.eye(4)),
        "episodes": episodes,
        "numerical_ik_quality": quality,
    }
    _write_json(output_path, plan)
    return plan


def solve_run(
    run_dir: Path,
    dataset_root: Path,
    intrinsics_summary: Path,
    extrinsics_summary: Path,
    p0_source: Path,
) -> Path:
    _status("detecting", "Detecting tags 56 and 57 with current production intrinsics/extrinsics")
    world_id, observations, diagnostics = collect_observations(dataset_root, intrinsics_summary, extrinsics_summary)
    solution = solve_eye_hand(observations)
    passed, reasons = assess_solution(solution)
    calibration_path = run_dir / "calibration.json"
    payload = {
        "schema": "p1_simple_eye_hand_calibration/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed_quality_gate",
        "world_frame_id": world_id,
        "frame_equation": "T_world_tag = T_world_base @ T_base_tcp @ T_tcp_tag",
        "tcp_frame": "fr3_ee",
        "marker": {"family": "tag36h11", "ids": list(TAG_IDS), "marker_size_m": TAG_SIZE_M, "backing_size_m": BACKING_SIZE_M},
        "inputs": {
            "dataset_root": str(dataset_root),
            "intrinsics_summary": str(intrinsics_summary),
            "intrinsics_sha256": _sha256(intrinsics_summary),
            "extrinsics_summary": str(extrinsics_summary),
            "extrinsics_sha256": _sha256(extrinsics_summary),
        },
        "T_world_base": matrix_payload(solution.pop("T_world_base")),
        "T_tcp_tag": {str(k): matrix_payload(v) for k, v in solution.pop("T_tcp_tags").items()},
        "quality": solution,
        "quality_gate_reasons": reasons,
        "detection_diagnostics": diagnostics,
    }
    _write_json(calibration_path, payload)
    if not passed:
        _status("failed", "Calibration quality gate failed", calibrationPath=str(calibration_path), reasons=reasons)
        raise RuntimeError("P1 calibration quality gate failed: " + "; ".join(reasons))
    candidate_path = run_dir / "p0_relocalized_timed_plan.json"
    _status("retargeting", "Calibration passed; generating an inactive P0 candidate")
    create_retargeted_p0_plan(
        p0_source,
        np.asarray(payload["T_world_base"], dtype=float),
        candidate_path,
        calibration_path,
    )
    _status("ready", "Candidate is ready for explicit activation", calibrationPath=str(calibration_path), candidatePath=str(candidate_path))
    return calibration_path


def activate(run_dir: Path, root: Path) -> Path:
    calibration_path = run_dir / "calibration.json"
    candidate_path = run_dir / "p0_relocalized_timed_plan.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed" or not candidate_path.is_file():
        raise RuntimeError("Only a passing calibration with a generated P0 candidate can be activated")
    active_path = root / "active.json"
    _write_json(
        active_path,
        {
            "schema": "p1_simple_eye_hand_active/v1",
            "activated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "calibration_path": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "plan_path": str(candidate_path),
            "plan_sha256": _sha256(candidate_path),
        },
    )
    _status("active", "P1 base calibration and relocalized P0 plan activated", activePath=str(active_path))
    return active_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="capture 50 stopped database poses, solve, and create an inactive P0 plan")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirmation", default="")
    run.add_argument("--count", type=int, default=50)
    run.add_argument("--source-poses", type=Path, default=DEFAULT_POSES)
    run.add_argument("--intrinsics-summary", type=Path, default=DEFAULT_INTRINSICS)
    run.add_argument("--extrinsics-summary", type=Path, default=DEFAULT_EXTRINSICS)
    run.add_argument("--p0-source", type=Path, default=DEFAULT_P0_SOURCE)
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    solve = sub.add_parser("solve", help="solve an already captured P1 dataset")
    solve.add_argument("dataset_root", type=Path)
    solve.add_argument("--run-dir", type=Path, required=True)
    solve.add_argument("--intrinsics-summary", type=Path, default=DEFAULT_INTRINSICS)
    solve.add_argument("--extrinsics-summary", type=Path, default=DEFAULT_EXTRINSICS)
    solve.add_argument("--p0-source", type=Path, default=DEFAULT_P0_SOURCE)
    active = sub.add_parser("activate", help="atomically activate a passing run")
    active.add_argument("run_dir", type=Path)
    active.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    if args.command == "activate":
        activate(args.run_dir.resolve(), args.output_root.resolve())
        return 0
    if args.command == "solve":
        solve_run(
            args.run_dir.resolve(), args.dataset_root.resolve(), args.intrinsics_summary.resolve(),
            args.extrinsics_summary.resolve(), args.p0_source.resolve(),
        )
        return 0
    if not args.execute:
        parser.error("run commands a real FR3; pass --execute after reviewing the risk warning")
    if args.confirmation != "P1_MOVE_FR3":
        parser.error("exact --confirmation P1_MOVE_FR3 is required")
    print(
        "[HIGH RISK] P1 calibration will move FR3 through existing taught poses. "
        "No project scene/collision model validates those paths after the robot base was moved. "
        "Clear people, payloads, cables and obstacles; hold the physical E-stop. GUI Abort is software-only.",
        flush=True,
    )
    root = args.output_root.resolve()
    run_dir = root / f"run_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _status(
        "preparing",
        f"Selecting {args.count} camera-near multi-camera poses with TCP z >= {MIN_TCP_Z_M:.6f} m",
        runDir=str(run_dir),
        tableContactTcpZM=TABLE_CONTACT_TCP_Z_M,
        minimumTcpClearanceM=MIN_TCP_CLEARANCE_M,
        minimumTcpZM=MIN_TCP_Z_M,
    )
    selected_path, config_path = prepare_capture(run_dir, args.source_poses.resolve(), args.count)
    run_native_preposition(selected_path)
    dataset_root = run_capture(config_path)
    solve_run(run_dir, dataset_root, args.intrinsics_summary.resolve(), args.extrinsics_summary.resolve(), args.p0_source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
