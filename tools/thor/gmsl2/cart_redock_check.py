"""Did a cart come back where it was registered?  Tags on the carts, seen by the fixed cameras (roadmap B5').

The two FR3s stand on carts that are pushed away while a person records two-hand demos on the table and pushed
back for replay.  A base is registered once (B3 for T_LR, one world edge); every re-dock after that is checked
here, against a reference capture taken right after that registration:

* ``measure`` reads one recorded episode (``cam_XX.mkv`` from the GMSL2 recorder), detects the DICT_6X6_50 tags
  of ``cart_tags.json`` on every calibrated camera (sub-pixel corners, a robust mean over the sampled frames), and
  triangulates each corner in the world with the cameras held at the calibration's poses (fisheye).  A camera
  whose reprojection residual stands out is dropped from that corner and named in the output.
* ``compare`` fits, per group, the rigid motion ``Δ`` that maps the reference corners onto the current ones
  (Kabsch).  Tags and base are fixed to the same cart, so ``T_world_base_now = Δ · T_world_base_ref``.  A cart
  passes when ``Δ`` moves no point within ``cart_reach_m`` of its tags by more than ``cart_mm`` -- a bound on
  what the arm's workspace saw, rotation included (``|t_c| + |ω|·r``, with ``t_c`` the motion of the tags'
  centroid).
* The table tags (``kind: static``, 45-49, fixed since 10-27) check the cameras first, in the image: each camera's
  pixel corners of each table tag, reference against current.  A camera whose table tags shifted moved and is
  left out of the carts; a tag that shifted in every camera that is otherwise steady was bumped and is left out of
  the check; when every camera that sees the table shifted, the table (or the whole rig) moved and no cart result
  is trusted.  A camera that sees no table tag cannot be checked and is left out of the carts too.  Pixels, not a
  3D fit of the table: 10-27 every camera held its table tags within 0.3 px over 2.5 h, yet a Kabsch of the two
  tags left after one noisy corner (45 and 48, 1.4 m apart) read 0.17 deg about the line joining them.
* ``freeze-datum`` keeps one table measurement as the table datum (``world/table_datum.json``): the corners in
  the world and each camera's pixel fingerprint.  ``cameras`` checks a later capture against it -- no reference
  capture, no calibration needed.
* ``session`` does that for every episode of recorded datasets (the first ~0.5 s of each camera that has table
  pixels in the datum) and keeps it in ``<dataset>/derived/table_check.json``; ``export_v3`` runs it for each
  session it exports and records the verdicts.  Each episode is held against the table pixels of the calibration
  it was recorded on (that run's ``table_anchor.json``, or the datum for the datum's run): matching pixels on
  other extrinsics only say the camera sat where some other calibration puts it.
* ``anchor`` places a new calibration's bundle in the world by the table tags alone (``T_world_session``, from
  the sweep it was solved from and the datum's own calibration, ``anchor_cameras``): the independent datum
  ``world_registration``/``export_production_calibration --anchor`` take, which sees the whole rig moving as one.
  Its ``table_anchor.json`` also keeps the sweep's table pixels; the gateway copies it into the exported run.

Why this can be trusted when single-tag hand-eye was ~10 mm off: that error is mostly systematic (camera poses,
the tag's printed size), and the same cameras looking at the same tags from the same place make the same error
twice -- it cancels in ``Δ``.  What is left grows with how far the cart moved, so a small ``Δ`` is a reliable
"did not move"; using a large ``Δ`` to *correct* the base is a different claim that needs the laser (an SMR on
each cart) before it is allowed.  ``Δ`` is written out, marked as not validated for that use.

The reference is only valid with the calibration it was measured with: ``compare`` refuses two measurements
made with different camera poses (re-take the reference after a new calibration goes live).

numpy + cv2 (Thor: ``third_party/opencv_kalibr/.venv``); the calibration is read from the run directories the
production tracker uses (``outputs/calibration/<run>_extrinsics|_intrinsics/summary.json``).
"""

# ruff: noqa: N803, N806  (matrix names: R, T, P, X, K follow the maths)
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LAYOUT_PATH = Path(__file__).with_name("cart_tags.json")
DATUM_PATH = Path(__file__).with_name("world") / "table_datum.json"
MEASUREMENT_SCHEMA = "cart_redock_measurement/v1"
REPORT_SCHEMA = "cart_redock_report/v2"
DATUM_SCHEMA = "table_datum/v1"
CAMERA_CHECK_SCHEMA = "table_camera_check/v1"
#: a datum needs this many table tags triangulated, so that leaving one out still fixes the rotation
DATUM_MIN_TAGS = 3
#: a (camera, tag) must be detected in this share of the sampled frames to count
MIN_SEEN_FRACTION = 0.5
#: a frame whose corners sit this far from the per-corner median is occluded or moving and is not averaged
FRAME_OUTLIER_PX = 1.0
#: cameras whose reprojections of one corner agree within this are triangulated together; the rest are dropped
INLIER_PX = 3.0
#: rays meeting at less than this angle fix depth badly (cam_03/cam_13 are 95 mm apart: 3.5 deg at 1.5 m puts
#: 0.2 px of noise at ~5 mm in depth); such a pair cannot seed a consensus or carry a tag alone
MIN_RAY_ANGLE_DEG = 10.0
#: a camera dropped from more than this share of its corners disagrees with the others everywhere (a stale or
#: moved pose) and is left out of every tag
CAMERA_OUTLIER_SHARE = 0.5
#: a tag the other tags of its group predict worse than this, while they agree among themselves within
#: 1/TAG_DROP_FACTOR of it, moved on its cart (re-stuck, peeled) and is not taken for the cart
TAG_DROP_MM = 1.0
TAG_DROP_FACTOR = 3.0
SESSION_CHECK_SCHEMA = "table_session_check/v2"
#: where ``session`` writes, under the dataset root (beside the tracker's sidecar; a Replay delete drops both)
SESSION_CHECK_PATH = Path("derived") / "table_check.json"
#: frames read per camera and episode: every SESSION_STRIDE-th of the first ones, ~0.5 s at 60 fps.  10-27
#: 15 frames at stride 2 held every table tag within 0.31 px of the datum's 60-frame mean
SESSION_SAMPLES = 15
SESSION_STRIDE = 2
#: per-episode verdicts, the first that applies wins when a session is summed up.  ``other_calibration``: the
#: episode was recorded on extrinsics that have no table pixels of their own (no ``table_anchor.json`` in the
#: run, and not the datum's run), so pixels matching the datum do not say the extrinsics it was labelled with
#: fit (10-27 cam_09/12 sat where 1027 puts them, not where 1014 did)
EPISODE_VERDICTS = ("table_or_all_cameras_moved", "cameras_moved", "other_calibration", "ok", "unchecked")
ANCHOR_SCHEMA = "table_anchor/v1"
#: written next to the bundle by ``anchor``, and copied into the exported extrinsics run by the gateway: the
#: table pixels a calibration was captured with are what its episodes are checked against
ANCHOR_FILE = "table_anchor.json"
#: frames read per camera, spread over the whole calibration sweep: the board and whoever carries it hide
#: different table tags at different times.  10-27 185829 (5402 frames): 120 frames kept 45/46/48 in 74-120 of them
ANCHOR_SAMPLES = 120
#: a table tag the anchor fit misses by more than this moved on the table; with three tags there is no telling
#: which one, so no anchor.  10-27 185829 against the datum: 0.45-1.38 mm
ANCHOR_TAG_MM = 3.0


# ------------------------------------------------------------------ cameras ---


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path_text: str, run_dir: Path) -> Path:
    """An intrinsics_json path as summary.json wrote it, or the same file under a copied run dir."""
    path = Path(path_text)
    if path.exists():
        return path
    parts = path.parts
    if run_dir.name in parts:
        return run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])
    raise FileNotFoundError(f"{path_text} (nor under {run_dir})")


def load_cameras(
    calibration_root: Path, extrinsics_run: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The calibrated cameras of one run: fisheye K, D and T_world_camera; and what identifies the run."""
    extrinsics_run = (
        extrinsics_run if extrinsics_run.endswith("_extrinsics") else f"{extrinsics_run}_extrinsics"
    )
    intrinsics_run = extrinsics_run.removesuffix("_extrinsics") + "_intrinsics"
    extr_path = calibration_root / extrinsics_run / "summary.json"
    intr_path = calibration_root / intrinsics_run / "summary.json"
    extr = json.loads(extr_path.read_text(encoding="utf-8"))
    intr = json.loads(intr_path.read_text(encoding="utf-8"))
    poses = {
        name: np.asarray(entry["base_to_camera"]["matrix_4x4"], dtype=np.float64)  # T_world_camera
        for name, entry in ((extr.get("joint_solution") or {}).get("cameras") or {}).items()
        if str(entry.get("status", "ok")).lower() == "ok"
    }
    cams: dict[str, dict[str, Any]] = {}
    for row in intr.get("cameras") or []:
        name = str(row["camera_name"])
        if str(row.get("status", "")).lower() != "ok" or name not in poses:
            continue
        data = json.loads(
            _resolve(str(row["intrinsics_json"]), calibration_root / intrinsics_run).read_text()
        )
        if str(data.get("model", "")).lower() not in {"opencv_fisheye", "fisheye", "equidistant"}:
            raise SystemExit(f"{name}: expected fisheye intrinsics, got {data.get('model')!r}")
        cams[name] = {
            "K": np.asarray(_maybe_json(data["camera_matrix"]), dtype=np.float64).reshape(3, 3),
            "D": np.asarray(_maybe_json(data["dist_coeffs"]), dtype=np.float64).reshape(4),
            "size": (int(data["image_width"]), int(data["image_height"])),
            "T_world_camera": poses[name],
        }
    if not cams:
        raise SystemExit(f"{extrinsics_run}: no camera has both ok intrinsics and an ok pose")
    identity = {
        "extrinsics_run": extrinsics_run,
        "intrinsics_run": intrinsics_run,
        "extrinsics_sha256": _sha256(extr_path),
        "intrinsics_sha256": _sha256(intr_path),
        "world_frame_id": (extr.get("world") or {}).get("world_frame_id"),
        "cameras": sorted(cams),
    }
    return cams, identity


def _fisheye_intrinsics(source: Path) -> dict[str, dict[str, Any]]:
    """K and D per camera from what a bundle names as its ``intrinsics_source``: a run directory of
    ``intrinsics_producer.json`` files (production, or the by-serial staging) or a ``calibrate_intrinsics`` report."""
    out: dict[str, dict[str, Any]] = {}
    if source.is_dir():
        paths = sorted(source.glob("converted/*/intrinsics_producer.json")) or sorted(
            source.glob("*/intrinsics_producer.json")
        )
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            if str(data.get("model", "")).lower() not in {"opencv_fisheye", "fisheye", "equidistant"}:
                raise SystemExit(f"{path}: expected fisheye intrinsics, got {data.get('model')!r}")
            out[str(data.get("camera_name") or path.parent.name)] = {
                "K": np.asarray(_maybe_json(data["camera_matrix"]), dtype=np.float64).reshape(3, 3),
                "D": np.asarray(_maybe_json(data["dist_coeffs"]), dtype=np.float64).reshape(4),
                "size": (int(data["image_width"]), int(data["image_height"])),
            }
        return out
    report = json.loads(source.read_text(encoding="utf-8"))
    for name, entry in (report.get("cameras") or {}).items():
        block = (entry.get("models") or {}).get("fisheye") or {}
        if "K" in block:
            out[name] = {
                "K": np.asarray(block["K"], dtype=np.float64).reshape(3, 3),
                "D": np.asarray(block["D"], dtype=np.float64).reshape(4),
            }
    return out


def load_bundle_cameras(
    report_path: Path, intrinsics: Path | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """A bundle's cameras in its own gauge S, before any registration: fisheye K, D and ``T_ref_cam``.

    The pose sits under ``T_world_camera`` because that is what ``project``/``triangulate`` read; the "world" here
    is the bundle's gauge.  ``intrinsics`` overrides the report's ``intrinsics_source`` (a copied bundle whose
    paths point at Thor).
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("camera_model", "fisheye")) != "fisheye":
        raise SystemExit(f"{report_path}: a {report.get('camera_model')} bundle; the table tags need fisheye")
    source = Path(intrinsics or report["intrinsics_source"])
    lenses = _fisheye_intrinsics(source)
    cams = {
        name: {**lenses[name], "T_world_camera": np.asarray(T, dtype=np.float64).reshape(4, 4)}
        for name, T in (report.get("T_ref_cam") or {}).items()
        if name in lenses
    }
    if not cams:
        raise SystemExit(f"{report_path}: no camera has both a pose and fisheye intrinsics in {source}")
    identity = {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "reference": report.get("reference"),
        "intrinsics_source": str(source),
        "cameras": sorted(cams),
    }
    return cams, identity


def _maybe_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def project(cam: dict[str, Any], X_world: np.ndarray) -> np.ndarray:
    """OpenCV fisheye (equidistant) projection of world points, (N, 3) -> (N, 2) pixels."""
    T_cw = np.linalg.inv(cam["T_world_camera"])
    Xc = np.asarray(X_world, dtype=np.float64).reshape(-1, 3) @ T_cw[:3, :3].T + T_cw[:3, 3]
    a, b = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r = np.hypot(a, b)
    theta = np.arctan(r)
    k1, k2, k3, k4 = cam["D"]
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
    K = cam["K"]
    return np.stack(
        [K[0, 0] * a * scale + K[0, 1] * b * scale + K[0, 2], K[1, 1] * b * scale + K[1, 2]], axis=1
    )


def _ray(cam: dict[str, Any], uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    xy = cv2.fisheye.undistortPoints(
        np.asarray(uv, dtype=np.float64).reshape(1, 1, 2),
        cam["K"],
        cam["D"].reshape(4, 1),
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
    ).reshape(2)
    d = cam["T_world_camera"][:3, :3] @ np.array([xy[0], xy[1], 1.0])
    return cam["T_world_camera"][:3, 3].copy(), d / np.linalg.norm(d)


def triangulate(
    cams: dict[str, dict[str, Any]], obs: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, float]]:
    """One point from >= 2 cameras' pixels: ray midpoint, then Gauss-Newton on the fisheye reprojection."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for name, uv in obs.items():
        o, d = _ray(cams[name], uv)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    X = np.linalg.solve(A, b)
    names = list(obs)
    target = np.concatenate([np.asarray(obs[n], dtype=np.float64).reshape(2) for n in names])

    def residual(x):
        return np.concatenate([project(cams[n], x[None])[0] for n in names]) - target

    for _ in range(10):
        r = residual(X)
        J = np.empty((r.size, 3))
        for k in range(3):
            step = np.zeros(3)
            step[k] = 1e-6
            J[:, k] = (residual(X + step) - r) / 1e-6
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + dx
        if np.linalg.norm(dx) < 1e-9:
            break
    r = residual(X).reshape(-1, 2)
    return X, {n: float(np.linalg.norm(r[i])) for i, n in enumerate(names)}


def ray_angle_deg(cams: dict[str, dict[str, Any]], names: Any, X: np.ndarray) -> float:
    """The widest angle at X between two of the cameras' centres: how well they fix X along depth."""
    import itertools

    dirs = [cams[n]["T_world_camera"][:3, 3] - X for n in names]
    dirs = [d / np.linalg.norm(d) for d in dirs]
    angles = [np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))) for a, b in itertools.combinations(dirs, 2)]
    return float(max(angles, default=0.0))


def triangulate_robust(
    cams: dict[str, dict[str, Any]], obs: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    """``triangulate`` on the largest set of cameras that agree within ``INLIER_PX``; the rest are dropped.

    The set is grown from every pair (ties: the pair that agrees best), not by dropping the worst camera of
    the joint fit: there a camera that is off pulls the point towards itself and spreads its error over
    everyone (10-27: cam_09 on stale poses put ~15 px on all four cameras of a tag, and two stale cameras
    against two good ones leave nothing standing out).
    """
    import itertools

    names = list(obs)
    if len(names) < 3:
        X, res = triangulate(cams, obs)
        return X, res, []

    def agreeing(X):
        return [
            n for n in names if np.linalg.norm(project(cams[n], X[None])[0] - np.asarray(obs[n])) <= INLIER_PX
        ]

    best_key, best = None, names
    for pair in itertools.combinations(names, 2):
        X, res = triangulate(cams, {n: obs[n] for n in pair})
        angle = ray_angle_deg(cams, pair, X)
        if angle < MIN_RAY_ANGLE_DEG:
            continue  # a short baseline places the point anywhere along depth; everyone else then "disagrees"
        inliers = agreeing(X)
        key = (len(inliers), angle)
        if best_key is None or key > best_key:
            best_key, best = key, inliers
    if len(best) < 2:
        X, res = triangulate(cams, obs)  # nothing agrees: keep all, the residual says so
        return X, res, []
    for _ in range(2):
        X, _ = triangulate(cams, {n: obs[n] for n in best})
        refined = agreeing(X)
        if len(refined) < 2 or set(refined) == set(best):
            break
        best = refined
    X, res = triangulate(cams, {n: obs[n] for n in best})
    return X, res, [n for n in names if n not in best]


# ---------------------------------------------------------------- detection ---


def load_layout(path: Path = LAYOUT_PATH) -> dict[str, Any]:
    layout = json.loads(path.read_text(encoding="utf-8"))
    seen: dict[int, str] = {}
    for group, spec in layout["groups"].items():
        for tag in spec["ids"]:
            if tag in seen:
                raise SystemExit(f"{path.name}: tag {tag} is in both {seen[tag]} and {group}")
            seen[tag] = group
    layout["group_of"] = seen
    return layout


def _detector(dictionary: str):
    import cv2

    params = cv2.aruco.DetectorParameters()
    # CONTOUR, not SUBPIX: on the recorder's H.265 frames SUBPIX snaps between two solutions ~0.7 px apart
    # from frame to frame (10-27, cam_07 tag 40); CONTOUR stays within ~0.1 px.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)


def detect_video(
    path: Path, dictionary: str, ids: set[int], stride: int = 5, max_samples: int = 60
) -> dict[str, Any]:
    """Per tag: the robust mean of its corners over the sampled frames, and how steady they were."""
    import cv2

    detector = _detector(dictionary)
    cap = cv2.VideoCapture(str(path))
    corners: dict[int, list[np.ndarray]] = {}
    sampled = 0
    size = None
    index = -1
    while sampled < max_samples:
        if not cap.grab():
            break
        index += 1
        if index % stride:
            continue
        ok, image = cap.retrieve()
        if not ok:
            break
        sampled += 1
        size = (image.shape[1], image.shape[0])
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, found_ids, _ = detector.detectMarkers(gray)
        for c, tag in zip(found, [] if found_ids is None else found_ids.ravel(), strict=True):
            if int(tag) in ids:
                corners.setdefault(int(tag), []).append(np.asarray(c, dtype=np.float64).reshape(4, 2))
    cap.release()
    tags: dict[int, dict[str, Any]] = {}
    for tag, stack in corners.items():
        arr = np.stack(stack)
        if len(arr) < MIN_SEEN_FRACTION * sampled:
            continue
        med = np.median(arr, axis=0)
        off = np.linalg.norm(arr - med, axis=2).max(axis=1)
        keep = arr[off <= FRAME_OUTLIER_PX]
        if len(keep) < MIN_SEEN_FRACTION * sampled:
            continue
        tags[tag] = {
            "corners_px": keep.mean(axis=0),
            "frames": int(len(keep)),
            "spread_px": float(np.linalg.norm(keep - keep.mean(axis=0), axis=2).std()),
        }
    return {"sampled": sampled, "size": size, "tags": tags}


# ------------------------------------------------------------------ measure ---


def measure_from_detections(
    cams: dict[str, dict[str, Any]], detections: dict[str, dict[str, Any]], layout: dict[str, Any]
) -> dict[str, Any]:
    """Triangulate every tag corner seen by >= 2 calibrated cameras that agree.

    Two passes: the first finds cameras that disagree with the others on most of their corners (left out of
    every tag, with the share named), the second triangulates without them.
    """
    first = _triangulate_tags(cams, detections, layout, excluded=set())
    excluded = {
        n: round(v["dropped_corners"] / v["corners"], 3)
        for n, v in first["cameras"].items()
        if v["corners"] >= 8 and v["dropped_corners"] > CAMERA_OUTLIER_SHARE * v["corners"]
    }
    result = _triangulate_tags(cams, detections, layout, excluded=set(excluded)) if excluded else first
    for n, share in excluded.items():
        result["cameras"][n].update(
            excluded=True,
            why=f"disagreed with the other cameras on {share:.0%} of its corners: stale or moved pose",
        )
    result["cameras_excluded"] = sorted(excluded)
    # every camera's pixels of every tag it saw, triangulated or not: what the static check compares
    for n, d in detections.items():
        result["cameras"].setdefault(n, {})["corners_px"] = {
            str(t): np.asarray(v["corners_px"]).round(4).tolist() for t, v in sorted(d["tags"].items())
        }
    return result


def _triangulate_tags(
    cams: dict[str, dict[str, Any]],
    detections: dict[str, dict[str, Any]],
    layout: dict[str, Any],
    excluded: set[str],
) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    per_cam: dict[str, dict[str, Any]] = {
        n: {"tags": [], "residuals_px": [], "corners": 0, "dropped_corners": 0} for n in detections
    }
    all_ids = sorted({t for d in detections.values() for t in d["tags"]})
    for tag in all_ids:
        seen_all = {
            n: d["tags"][tag]["corners_px"] for n, d in detections.items() if tag in d["tags"] and n in cams
        }
        seen = {n: uv for n, uv in seen_all.items() if n not in excluded}
        if len(seen) < 2:
            tags[str(tag)] = {
                "group": layout["group_of"][tag],
                "status": "seen_by_<2",
                "cameras": sorted(seen),
                "cameras_excluded": sorted(set(seen_all) - set(seen)),
            }
            continue
        points, rms, dropped_any = [], [], set()
        for k in range(4):
            X, res, dropped = triangulate_robust(cams, {n: uv[k] for n, uv in seen.items()})
            points.append(X)
            rms.extend(res.values())
            dropped_any.update(dropped)
            for n in seen:
                per_cam[n]["corners"] += 1
                per_cam[n]["dropped_corners"] += n in dropped
            for n, v in res.items():
                per_cam[n]["residuals_px"].append(v)
        P = np.array(points)
        edge_mm = float(np.mean(np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1)) * 1000)
        rms_px = float(np.sqrt(np.mean(np.square(rms))))
        tags[str(tag)] = {
            "group": layout["group_of"][tag],
            "status": "ok" if rms_px <= INLIER_PX else "inconsistent",
            "corners_world_m": P.round(7).tolist(),
            "cameras": sorted(set(seen) - dropped_any),
            "dropped_cameras": sorted(dropped_any),
            "cameras_excluded": sorted(set(seen_all) - set(seen)),
            "reprojection_rms_px": rms_px,
            "edge_mm": round(edge_mm, 3),
            "nominal_edge_mm": (layout.get("black_edge_mm") or {}).get(str(tag)),
            # what compare re-triangulates from, with the cameras both captures agree on
            "observations_px": {n: np.asarray(uv).round(4).tolist() for n, uv in sorted(seen_all.items())},
        }
        for n in seen:
            per_cam[n]["tags"].append(tag)
    cameras = {
        n: {
            "tags": sorted(v["tags"]),
            "reprojection_rms_px": float(np.sqrt(np.mean(np.square(v["residuals_px"]))))
            if v["residuals_px"]
            else None,
            "corners": v["corners"],
            "dropped_corners": v["dropped_corners"],
        }
        for n, v in sorted(per_cam.items())
    }
    return {"tags": tags, "cameras": cameras}


def _episode_calibration(episode: Path) -> str | None:
    meta = episode / "meta.json"
    if not meta.exists():
        return None
    world = json.loads(meta.read_text(encoding="utf-8")).get("world_frame") or {}
    return (world.get("last_registration") or {}).get("calibration_id") or world.get("calibration_id")


def _measure(args: argparse.Namespace) -> int:
    episode = args.episode.resolve()
    layout = load_layout(args.layout)
    run = args.calibration_run or _episode_calibration(episode)
    if not run:
        print(f"{episode}: no meta.json naming the calibration it was recorded with; give --calibration-run")
        return 2
    cams, identity = load_cameras(args.calibration_root.resolve(), run)
    ids = set(layout["group_of"])
    detections: dict[str, dict[str, Any]] = {}
    for name in sorted(cams):
        video = episode / f"{name}.mkv"
        if not video.exists():
            continue
        det = detect_video(video, layout["dictionary"], ids, stride=args.stride, max_samples=args.max_samples)
        if det["size"] and tuple(det["size"]) != cams[name]["size"]:
            print(f"{name}: video is {det['size']}, the intrinsics are {cams[name]['size']}")
            return 2
        detections[name] = det
    result = measure_from_detections(cams, detections, layout)
    for name, det in detections.items():
        result["cameras"].setdefault(name, {})["frames_sampled"] = det["sampled"]
        result["cameras"][name]["spread_px"] = {
            str(t): round(v["spread_px"], 3) for t, v in det["tags"].items()
        }
    out = {
        "schema": MEASUREMENT_SCHEMA,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "episode": str(episode),
        "episode_complete": (episode / "meta.json").exists(),
        "calibration": identity,
        "layout_sha256": _sha256(args.layout),
        "dictionary": layout["dictionary"],
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(_measurement_summary(out, layout))
    print(f"wrote {args.out}")
    return 0


def _measurement_summary(m: dict[str, Any], layout: dict[str, Any]) -> str:
    lines = [f"calibration {m['calibration']['extrinsics_run']} ({m['calibration']['world_frame_id']})"]
    for group in layout["groups"]:
        rows = [
            f"{t}:{len(v['cameras'])}cam/{v['reprojection_rms_px']:.2f}px/{v['edge_mm']:.1f}mm"
            + (f" drop {','.join(c[4:] for c in v['dropped_cameras'])}" if v["dropped_cameras"] else "")
            if v["status"] == "ok"
            else f"{t}:{v['status']}"
            for t, v in sorted(m["tags"].items(), key=lambda kv: int(kv[0]))
            if v["group"] == group
        ]
        lines.append(f"  {group:10s} " + ("  ".join(rows) if rows else "no tag seen"))
    for n in m.get("cameras_excluded", []):
        lines.append(f"  excluded {n}: {m['cameras'][n]['why']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ compare ---


def kabsch(P_ref: np.ndarray, P_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R P_ref + t - P_cur|."""
    c_ref, c_cur = P_ref.mean(axis=0), P_cur.mean(axis=0)
    U, _, Vt = np.linalg.svd((P_ref - c_ref).T @ (P_cur - c_cur))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, c_cur - R @ c_ref


def _rms_mm(diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))) * 1000


def _rotvec(R: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def paired_corners(
    cams: dict[str, dict[str, Any]],
    ref: dict[str, Any],
    cur: dict[str, Any],
    tag: int,
    exclude: set[str] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """One tag's corners in both captures, triangulated from the *same* cameras.

    The difference only cancels the cameras' systematic errors if both captures see them through the same
    cameras: a corner from cameras {03, 07, 13} and the same corner from {03, 13} differ by how those two
    camera sets disagree (10-27: tag 42's edge read 74 mm from three cameras and 91 mm from two).
    """
    a, b = ref["tags"].get(str(tag), {}), cur["tags"].get(str(tag), {})
    if a.get("status") != "ok" or b.get("status") != "ok":
        return None
    common = sorted(set(a["cameras"]) & set(b["cameras"]) & set(cams) - set(exclude))
    if len(common) < 2:
        return None
    points = []
    for capture in (a, b):
        obs = {n: np.asarray(capture["observations_px"][n]) for n in common}
        points.append(np.array([triangulate(cams, {n: uv[k] for n, uv in obs.items()})[0] for k in range(4)]))
    if ray_angle_deg(cams, common, points[0].mean(axis=0)) < MIN_RAY_ANGLE_DEG:
        return None
    return points[0], points[1], common


def fit_tags(
    pairs: dict[int, tuple[np.ndarray, np.ndarray, list[str]]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """The tags of ``pairs`` (reference corners, current corners, cameras) that one rigid motion explains, and the
    ones left out because the others predict them badly."""
    common = list(pairs)
    dropped: list[dict[str, Any]] = []
    # dropping one of three leaves two tags, and the rotation about the line through them is then noise (10-27:
    # right cart 40/42/44 lost 42 at 1.1 mm and read 0.14 deg, unmoved); a tag that moved among three shows in the
    # fit residual instead
    while len(common) >= 4:
        # leave one tag out: in the joint fit a tag that moved drags the others along and hides (10 mm on one
        # of four tags leaves it only ~3x the others' residual)
        loo = {}
        for tag in common:
            rest = [x for x in common if x != tag]
            R_k, t_k = kabsch(
                np.concatenate([pairs[x][0] for x in rest]), np.concatenate([pairs[x][1] for x in rest])
            )
            rest_rms = _rms_mm(
                np.concatenate([pairs[x][0] for x in rest]) @ R_k.T
                + t_k
                - np.concatenate([pairs[x][1] for x in rest])
            )
            loo[tag] = (
                float(np.linalg.norm(pairs[tag][0] @ R_k.T + t_k - pairs[tag][1], axis=1).mean() * 1000),
                rest_rms,
            )
        worst = max(loo, key=lambda x: loo[x][0])
        miss, rest_rms = loo[worst]
        if not (miss > TAG_DROP_MM and miss > TAG_DROP_FACTOR * rest_rms):
            break
        dropped.append({"tag": worst, "predicted_by_the_others_mm": round(miss, 3)})
        common.remove(worst)
    return common, dropped


def group_motion(
    cams: dict[str, dict[str, Any]],
    ref: dict[str, Any],
    cur: dict[str, Any],
    ids: list[int],
    reach_m: float,
    exclude: set[str] = frozenset(),
) -> dict[str, Any]:
    """Δ of one group from the tags both captures triangulated, dropping tags that moved on their own."""
    pairs = {t: pc for t in ids if (pc := paired_corners(cams, ref, cur, t, exclude)) is not None}
    # seen in either capture but not usable: not triangulated in both, or only by cameras too close together
    unpaired = [t for t in ids if t not in pairs and (str(t) in ref["tags"] or str(t) in cur["tags"])]
    common, dropped = fit_tags(pairs)
    if len(common) < 2:
        return {
            "status": "insufficient",
            "tags": common,
            "tags_dropped": dropped,
            "tags_unpaired": unpaired,
            "why": f"fewer than 2 tags triangulated by the same 2+ cameras, rays >= {MIN_RAY_ANGLE_DEG:g} deg "
            "apart, in both captures (one tag cannot fix the rotation)",
        }
    P_ref = np.concatenate([pairs[t][0] for t in common])
    P_cur = np.concatenate([pairs[t][1] for t in common])
    R, t = kabsch(P_ref, P_cur)
    res = np.linalg.norm(P_ref @ R.T + t - P_cur, axis=1).reshape(len(common), 4).mean(axis=1) * 1000
    c = P_ref.mean(axis=0)
    omega = _rotvec(R)
    t_c = R @ c + t - c
    rms_mm = float(np.sqrt(np.mean(np.sum((P_ref @ R.T + t - P_cur) ** 2, axis=1)))) * 1000
    n = len(P_ref)
    spread_m = float(np.sqrt(np.mean(np.sum((P_ref - c) ** 2, axis=1))))
    # 1-sigma of the bound from the fit residual: centroid ~ rms/sqrt(n), rotation ~ rms/(sqrt(n) * spread)
    sigma_mm = rms_mm / np.sqrt(n) * (1 + reach_m / max(spread_m, 1e-6))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return {
        "status": "measured",
        "tags": common,
        "tags_dropped": dropped,
        "tags_unpaired": unpaired,
        "centroid_shift_mm": round(float(np.linalg.norm(t_c)) * 1000, 3),
        "centroid_shift_xyz_mm": (t_c * 1000).round(3).tolist(),
        "rotation_deg": round(float(np.degrees(np.linalg.norm(omega))), 4),
        "within_reach_bound_mm": round(
            float(np.linalg.norm(t_c) + np.linalg.norm(omega) * reach_m) * 1000, 3
        ),
        "reach_m": reach_m,
        "bound_sigma_mm": round(float(sigma_mm), 3),
        "fit_rms_mm": round(rms_mm, 3),
        "tag_residual_mm": {str(tg): round(float(r), 3) for tg, r in zip(common, res, strict=True)},
        "tag_spread_m": round(spread_m, 4),
        "tag_cameras": {str(tg): pairs[tg][2] for tg in common},
        "delta_world": {
            "matrix_4x4": T.round(9).tolist(),
            "meaning": "maps the reference capture onto this one: T_world_base_now = Δ · T_world_base_ref",
            "validated_for_correction": False,
        },
    }


def camera_pixels(m: dict[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    """Each camera's corners (4, 2) per tag.  Measurements before the static check kept pixels only for the tags
    they triangulated (``observations_px``); those still work, without the single-camera tags."""
    out = {
        n: {int(t): np.asarray(c, dtype=np.float64) for t, c in v["corners_px"].items()}
        for n, v in (m.get("cameras") or {}).items()
        if "corners_px" in v
    }
    if out:
        return out
    for t, v in (m.get("tags") or {}).items():
        for n, uv in (v.get("observations_px") or {}).items():
            out.setdefault(n, {})[int(t)] = np.asarray(uv, dtype=np.float64)
    return out


def static_check(
    ref_px: dict[str, dict[int, np.ndarray]],
    cur_px: dict[str, dict[int, np.ndarray]],
    ids: list[int],
    limit_px: float,
    cameras: list[str],
) -> dict[str, Any]:
    """Which cameras kept their view of the fixed tags: per camera and tag, the largest corner shift in pixels.

    A camera moved when most of its tags shifted; a tag was bumped when it shifted in every camera that did not.
    ``cameras``: every camera to report on -- one with no fixed tag in both captures is ``unverified``.
    """
    shift = {
        n: {t: float(np.linalg.norm(cur_px[n][t] - ref_px[n][t], axis=1).max()) for t in ids
            if t in ref_px.get(n, {}) and t in cur_px.get(n, {})}
        for n in cameras
    }  # fmt: skip
    over = {n: {t for t, s in v.items() if s > limit_px} for n, v in shift.items()}

    def moved_cameras(bumped: set[int]) -> set[str]:
        return {
            n for n, v in shift.items() if (tags := set(v) - bumped) and 2 * len(over[n] & tags) > len(tags)
        }

    moved = moved_cameras(set())
    steady = [n for n in shift if n not in moved]
    bumped = {
        t for t in ids if (seen := [n for n in steady if t in shift[n]]) and all(t in over[n] for n in seen)
    }
    moved = moved_cameras(bumped)
    per_cam = {}
    for n, v in shift.items():
        tags = set(v) - bumped
        per_cam[n] = {
            "status": "unverified" if not tags else "moved" if n in moved else "ok",
            "shift_px": {str(t): round(s, 3) for t, s in sorted(v.items())},
        }
    ok = sorted(n for n, v in per_cam.items() if v["status"] == "ok")
    moved_list = sorted(moved)
    if not ok and not moved_list:
        status, why = "missing", "no camera saw a fixed tag in both captures"
    elif not ok:
        status, why = "table_or_all_cameras_moved", "every camera that sees the fixed tags saw them shift"
    else:
        status, why = "ok", ""
    return {
        "status": status,
        "why": why,
        "limit_px": limit_px,
        "cameras_ok": ok,
        "cameras_moved": moved_list,
        "cameras_unverified": sorted(n for n, v in per_cam.items() if v["status"] == "unverified"),
        "tags_bumped": sorted(bumped),
        "cameras": per_cam,
    }


def _static_ids(layout: dict[str, Any]) -> list[int]:
    return [t for spec in layout["groups"].values() if spec["kind"] == "static" for t in spec["ids"]]


def compare(
    cams: dict[str, dict[str, Any]], ref: dict[str, Any], cur: dict[str, Any], layout: dict[str, Any]
) -> dict[str, Any]:
    """``cams``: the calibration both captures were measured with (``load_cameras``)."""
    for key in ("extrinsics_sha256", "intrinsics_sha256"):
        if ref["calibration"][key] != cur["calibration"][key]:
            raise ValueError(
                f"the reference was measured with {ref['calibration']['extrinsics_run']}, this capture with "
                f"{cur['calibration']['extrinsics_run']} ({key} differs): camera poses changed, so differences "
                "would be calibration, not carts. Take a new reference after the registration."
            )
    th = layout["thresholds"]
    static = static_check(
        camera_pixels(ref), camera_pixels(cur), _static_ids(layout), float(th["static_px"]), sorted(cams)
    )
    trusted = static["status"] == "ok"
    # the carts are triangulated only by cameras the table vouches for; without that, by all (reported untrusted)
    exclude = set(static["cameras_moved"]) | set(static["cameras_unverified"]) if trusted else set()
    groups: dict[str, Any] = {}
    for name, spec in layout["groups"].items():
        is_static = spec["kind"] == "static"
        reach = float(th["static_reach_m"] if is_static else th["cart_reach_m"])
        g = group_motion(cams, ref, cur, list(spec["ids"]), reach, exclude)
        g.update(kind=spec["kind"], base=spec.get("base"))
        if is_static:
            # the table's own 3D motion: shown, never decides (see the module docstring)
            g["status"] = "info" if g["status"] == "measured" else g["status"]
        else:
            g["limit_mm"] = float(th["cart_mm"])
            if g["status"] == "measured":
                g["status"] = "ok" if g["within_reach_bound_mm"] <= g["limit_mm"] else "moved"
            if not trusted:
                g["status_untrusted"], g["status"] = g["status"], "untrusted"
        groups[name] = g
    carts = [g for g in groups.values() if g["kind"] == "cart"]
    if static["status"] == "table_or_all_cameras_moved":
        verdict = "cameras_or_table_moved"
    elif static["status"] == "missing":
        verdict = "static_reference_missing"
    elif carts and all(g["status"] == "ok" for g in carts):
        verdict = "ok"
    elif any(g["status"] == "moved" for g in carts):
        verdict = "cart_moved"
    else:
        verdict = "insufficient"
    newly_dropped = {
        n: v.get("dropped_corners", 0)
        for n, v in cur["cameras"].items()
        if v.get("dropped_corners", 0) > ref["cameras"].get(n, {}).get("dropped_corners", 0)
    }
    newly_excluded = sorted(set(cur.get("cameras_excluded", [])) - set(ref.get("cameras_excluded", [])))
    return {
        "schema": REPORT_SCHEMA,
        "compared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "reference": {"episode": ref["episode"], "measured_utc": ref["measured_utc"]},
        "current": {"episode": cur["episode"], "measured_utc": cur["measured_utc"]},
        "calibration": cur["calibration"]["extrinsics_run"],
        "verdict": verdict,
        "absolute_replay_allowed": verdict == "ok",
        "thresholds": th,
        "static_check": static,
        "cameras_left_out_of_carts": sorted(exclude),
        "groups": groups,
        "cameras_dropped_more_than_reference": newly_dropped,
        "cameras_excluded": {
            "reference": ref.get("cameras_excluded", []),
            "current": cur.get("cameras_excluded", []),
        },
        "cameras_newly_excluded": newly_excluded,
    }


def _static_summary(s: dict[str, Any]) -> list[str]:
    def cams(names):
        return ",".join(n.removeprefix("cam_") for n in names) or "-"

    lines = [
        f"  table (pixels, limit {s['limit_px']} px): {s['status']}"
        + (f" -- {s['why']}" if s["why"] else "")
        + f"  ok {cams(s['cameras_ok'])}  moved {cams(s['cameras_moved'])}"
        + f"  unverified {cams(s['cameras_unverified'])}"
        + (f"  bumped tags {s['tags_bumped']}" if s["tags_bumped"] else "")
    ]
    for n, v in s["cameras"].items():
        if v["shift_px"]:
            shifts = " ".join(f"{t}:{x:.2f}" for t, x in v["shift_px"].items())
            lines.append(f"    {n} {v['status']:10s} {shifts}")
    return lines


def _report_summary(r: dict[str, Any]) -> str:
    lines = [
        f"verdict: {r['verdict']}  (absolute replay {'allowed' if r['absolute_replay_allowed'] else 'REFUSED'})"
    ]
    lines += _static_summary(r["static_check"])
    if r["cameras_left_out_of_carts"]:
        lines.append(f"  carts triangulated without {r['cameras_left_out_of_carts']}")
    for name, g in r["groups"].items():
        if "within_reach_bound_mm" in g:
            limit = f"limit {g['limit_mm']}, " if "limit_mm" in g else "3D, info only, "
            lines.append(
                f"  {name:10s} {g['status']:9s} bound {g['within_reach_bound_mm']:.2f} mm ({limit}"
                f"±{g['bound_sigma_mm']:.2f})  centroid {g['centroid_shift_mm']:.2f} mm  rot {g['rotation_deg']:.3f}°"
                f"  tags {g['tags']}"
                + (f"  dropped {g['tags_dropped']}" if g["tags_dropped"] else "")
                + (f"  unusable {g['tags_unpaired']}" if g["tags_unpaired"] else "")
            )
        else:
            lines.append(f"  {name:10s} {g['status']}: {g.get('why', '')}")
    if r["static_check"]["cameras_moved"]:
        lines.append("  a camera moved since the reference: its calibrated pose is stale for tracking too")
    if r["cameras_newly_excluded"]:
        lines.append(
            f"  cameras that agreed in the reference but not now (moved?): {r['cameras_newly_excluded']}"
        )
    if r["cameras_dropped_more_than_reference"]:
        lines.append(
            f"  cameras dropped more often than in the reference: {r['cameras_dropped_more_than_reference']}"
        )
    return "\n".join(lines)


def _load_measurement_cameras(
    calibration_root: Path, m: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | str:
    cams, identity = load_cameras(calibration_root.resolve(), m["calibration"]["extrinsics_run"])
    for key in ("extrinsics_sha256", "intrinsics_sha256"):
        if identity[key] != m["calibration"][key]:
            return (
                f"{identity['extrinsics_run']} under {calibration_root} is not the file the measurement was "
                f"made with ({key} differs)"
            )
    return cams, identity


def _compare(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout)
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    cur = json.loads(args.current.read_text(encoding="utf-8"))
    loaded = _load_measurement_cameras(args.calibration_root, ref)
    if isinstance(loaded, str):
        print(loaded)
        return 2
    cams, _ = loaded
    try:
        report = compare(cams, ref, cur, layout)
    except ValueError as exc:
        print(exc)
        return 2
    print(_report_summary(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if report["verdict"] == "ok" else 1


# -------------------------------------------------------------- table datum ---


def datum_cameras(cams: dict[str, dict[str, Any]], names: Any) -> dict[str, dict[str, Any]]:
    """The calibration a datum's pixels were taken with, for the cameras that saw the table: what lets
    ``table_anchor`` triangulate the datum again long after that run is gone from ``outputs/``."""
    return {
        n: {
            "K": cams[n]["K"].round(9).tolist(),
            "D": cams[n]["D"].round(12).tolist(),
            "T_world_camera": cams[n]["T_world_camera"].round(12).tolist(),
        }
        for n in sorted(names)
        if n in cams
    }


def _datum_cameras(datum: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        n: {k: np.asarray(v[k], dtype=np.float64) for k in ("K", "D", "T_world_camera")}
        for n, v in (datum.get("anchor_cameras") or {}).items()
    }


def freeze_datum(
    m: dict[str, Any],
    layout: dict[str, Any],
    source: str,
    cams: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The table tags of one measurement, kept as the table datum: world corners plus each camera's pixels.

    The corners are what the cameras triangulated in this calibration's world, not the printed geometry (the
    triangulated edges read 2-4 % short of the caliper, a detector bias that cancels only in differences).
    ``cams``, the calibration ``m`` was measured with, is kept for the cameras that saw the table
    (``anchor_cameras``): without it the datum cannot anchor a later calibration.
    """
    ids = _static_ids(layout)
    tags = {t: m["tags"][str(t)] for t in ids if str(t) in m["tags"]}
    ok = {t: v for t, v in tags.items() if v["status"] == "ok"}
    bad = {t: v["status"] for t, v in tags.items() if v["status"] == "inconsistent"}
    if bad:
        raise ValueError(f"table tags the cameras disagree on: {bad}; re-capture with the table clear")
    if len(ok) < DATUM_MIN_TAGS:
        raise ValueError(
            f"only {sorted(ok)} triangulated (need {DATUM_MIN_TAGS}): one tag hidden or the calibration stale?"
        )
    P = np.concatenate([np.asarray(v["corners_world_m"]) for v in ok.values()])
    centre = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - centre)
    along = (P - centre) @ Vt[:2].T
    px = camera_pixels(m)
    return {
        "schema": DATUM_SCHEMA,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "world_frame_id": m["calibration"]["world_frame_id"],
        "calibration": m["calibration"],
        "source": {"recording": source, "episode": m["episode"], "measured_utc": m["measured_utc"]},
        "dictionary": m["dictionary"],
        "meaning": (
            "the fixed table tags in this world: corners_world_m for registering a calibration (an independent "
            "datum, world_frame.register_session's anchor), camera_pixels for telling which camera moved "
            "(valid while the camera stays where it was at the capture, whatever the calibration says)"
        ),
        "tags": {
            str(t): {
                "corners_world_m": v["corners_world_m"],
                "cameras": v["cameras"],
                "reprojection_rms_px": round(v["reprojection_rms_px"], 3),
                "edge_mm": v["edge_mm"],
                "nominal_edge_mm": v.get("nominal_edge_mm"),
            }
            for t, v in ok.items()
        },
        "tags_seen_by_one_camera": sorted(t for t, v in tags.items() if v["status"] == "seen_by_<2"),
        "camera_pixels": {
            n: {str(t): v[t].round(4).tolist() for t in ids if t in v} for n, v in sorted(px.items())
        },
        "cameras_without_a_table_tag": sorted(n for n in m["cameras"] if not set(ids) & set(px.get(n, {}))),
        **(
            {"anchor_cameras": datum_cameras(cams, [n for n, v in px.items() if set(ids) & set(v)])}
            if cams
            else {}
        ),
        "plane": {
            "point_world_m": centre.round(6).tolist(),
            "normal_world": Vt[2].round(6).tolist(),
            "extent_m": (along.max(axis=0) - along.min(axis=0)).round(3).tolist(),
            "rms_mm": round(float(np.sqrt(np.mean(((P - centre) @ Vt[2]) ** 2))) * 1000, 3),
        },
    }


def _freeze_datum(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout)
    m = json.loads(args.measurement.read_text(encoding="utf-8"))
    if args.out.exists() and not args.replace:
        print(
            f"{args.out} exists: a datum is what later captures are held against, so replacing it is a decision "
            "(tags re-stuck, table moved). Pass --replace to make it."
        )
        return 2
    cams, identity = load_cameras(args.calibration_root.resolve(), m["calibration"]["extrinsics_run"])
    if any(identity[k] != m["calibration"].get(k) for k in ("extrinsics_sha256", "intrinsics_sha256")):
        print(f"{identity['extrinsics_run']} changed since the measurement; measure again")
        return 2
    try:
        datum = freeze_datum(m, layout, args.source, cams)
    except ValueError as exc:
        print(exc)
        return 2
    if args.out.exists():
        old = json.loads(args.out.read_text(encoding="utf-8"))
        datum["replaces"] = {"frozen_utc": old.get("frozen_utc"), "sha256": _sha256(args.out)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(datum, indent=2) + "\n", encoding="utf-8")
    print(
        f"table datum in {datum['world_frame_id']} ({m['calibration']['extrinsics_run']}): tags "
        f"{sorted(map(int, datum['tags']))}, plane rms {datum['plane']['rms_mm']} mm over "
        f"{datum['plane']['extent_m']} m; no table tag in view: {datum['cameras_without_a_table_tag']}"
    )
    print(f"wrote {args.out}")
    return 0


def check_cameras(datum: dict[str, Any], m: dict[str, Any], limit_px: float) -> dict[str, Any]:
    """Which cameras still see the table where they saw it when the datum was frozen."""
    ref_px = {n: {int(t): np.asarray(c) for t, c in v.items()} for n, v in datum["camera_pixels"].items()}
    ids = sorted({t for v in ref_px.values() for t in v})
    result = static_check(ref_px, camera_pixels(m), ids, limit_px, sorted(set(ref_px) | set(m["cameras"])))
    return {
        "schema": CAMERA_CHECK_SCHEMA,
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "datum": {"frozen_utc": datum["frozen_utc"], "world_frame_id": datum["world_frame_id"]},
        "episode": m["episode"],
        "measured_utc": m["measured_utc"],
        **result,
    }


def _cameras(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout)
    datum = json.loads(args.datum.read_text(encoding="utf-8"))
    m = json.loads(args.measurement.read_text(encoding="utf-8"))
    r = check_cameras(datum, m, float(layout["thresholds"]["static_px"]))
    print(f"cameras against the table datum of {datum['frozen_utc']}:")
    print("\n".join(_static_summary(r)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if r["status"] == "ok" and not r["cameras_moved"] else 1


# ------------------------------------------------------------- table anchor ---


def _frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count


def detect_capture(
    episodes: list[Path], cameras: list[str], dictionary: str, ids: set[int], *, samples: int = ANCHOR_SAMPLES
) -> dict[str, dict[str, Any]]:
    """``detect_video`` spread over each camera's whole video; with several episodes, per camera and tag the one
    that saw it in the most frames."""
    jobs = [(n, ep / f"{n}.mkv") for n in cameras for ep in episodes if (ep / f"{n}.mkv").exists()]

    def run(job: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        name, video = job
        stride = max(1, _frame_count(video) // samples)
        return name, detect_video(video, dictionary, ids, stride=stride, max_samples=samples)

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), 8))) as pool:
        for name, det in pool.map(run, jobs):
            have = out.setdefault(name, {"sampled": 0, "size": det["size"], "tags": {}})
            have["sampled"] += det["sampled"]
            for tag, v in det["tags"].items():
                if v["frames"] > have["tags"].get(tag, {}).get("frames", 0):
                    have["tags"][tag] = v
    return out


def table_anchor(
    datum: dict[str, Any], session_cams: dict[str, dict[str, Any]], detections: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Where a bundle's gauge S sits in the world, from the table tags alone: ``T_world_session``.

    Each table tag is triangulated twice from the same cameras -- the datum's pixels with the datum's calibration
    (world), this capture's pixels with the bundle (S) -- and the corner sets are fitted rigidly.  Same cameras
    because only they make the same detector bias twice (``paired_corners``).  A camera that moved in between is
    no problem: its pixels and its pose move together on each side.  Independent of the camera consensus by
    construction: no camera pose of the reference world enters, only the datum's own capture.

    10-27, the 185829 sweep solved on its own against the datum of 165415: 0.36 mm / 0.13 deg from where the
    camera consensus put the same bundle (0.08-0.17 deg over other tag/camera choices; two solves of the rig
    differ per camera by 0.01-0.17 deg).
    """
    world_cams = _datum_cameras(datum)
    if not world_cams:
        return {"status": "no_datum_calibration", "why": "the datum keeps no anchor_cameras; freeze it again"}
    ref_px = {
        n: {int(t): np.asarray(c, dtype=np.float64) for t, c in v.items()}
        for n, v in datum["camera_pixels"].items()
    }
    cur_px = {n: {int(t): np.asarray(v["corners_px"], dtype=np.float64) for t, v in d["tags"].items()}
              for n, d in detections.items()}  # fmt: skip
    pairs: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    unpaired: list[int] = []
    for tag in sorted({t for v in ref_px.values() for t in v}):
        common = sorted(
            n
            for n in set(world_cams) & set(session_cams)
            if tag in ref_px.get(n, {}) and tag in cur_px.get(n, {})
        )
        if len(common) >= 2:
            P_w = np.array(
                [triangulate(world_cams, {n: ref_px[n][tag][k] for n in common})[0] for k in range(4)]
            )
            if ray_angle_deg(world_cams, common, P_w.mean(axis=0)) >= MIN_RAY_ANGLE_DEG:
                P_s = np.array(
                    [triangulate(session_cams, {n: cur_px[n][tag][k] for n in common})[0] for k in range(4)]
                )
                pairs[tag] = (P_s, P_w, common)
                continue
        if any(tag in v for v in cur_px.values()):
            unpaired.append(tag)
    tags, dropped = fit_tags(pairs)
    head = {"tags": tags, "tags_dropped": dropped, "tags_unpaired": unpaired}
    if len(tags) < DATUM_MIN_TAGS:
        return {
            "status": "insufficient",
            **head,
            "why": f"{len(tags)} table tags triangulated by the same 2+ cameras, rays >= {MIN_RAY_ANGLE_DEG:g} deg "
            f"apart, in both the datum and this capture (need {DATUM_MIN_TAGS})",
        }
    P_s = np.concatenate([pairs[t][0] for t in tags])
    P_w = np.concatenate([pairs[t][1] for t in tags])
    R, t = kabsch(P_s, P_w)
    res = np.linalg.norm(P_s @ R.T + t - P_w, axis=1).reshape(len(tags), 4).mean(axis=1) * 1000
    T_world_session = np.eye(4)
    T_world_session[:3, :3], T_world_session[:3, 3] = R, t
    # The anchor is a frame at the table's centre, so that what world_registration reports as its drift is the
    # drift where the table (and the work) is, not at the bundle's reference camera.
    T_world_anchor = np.eye(4)
    T_world_anchor[:3, 3] = P_w.mean(axis=0)
    centre = P_w.mean(axis=0)
    out = {
        "status": "measured" if res.max() <= ANCHOR_TAG_MM else "inconsistent",
        **head,
        "T_world_session": T_world_session.round(12).tolist(),
        "T_world_anchor": T_world_anchor.round(12).tolist(),
        "T_session_anchor": (np.linalg.inv(T_world_session) @ T_world_anchor).round(12).tolist(),
        "fit_rms_mm": round(_rms_mm(P_s @ R.T + t - P_w), 3),
        "tag_residual_mm": {str(tg): round(float(r), 3) for tg, r in zip(tags, res, strict=True)},
        "tag_cameras": {str(tg): pairs[tg][2] for tg in tags},
        "tag_spread_m": round(float(np.sqrt(np.mean(np.sum((P_w - centre) ** 2, axis=1)))), 4),
    }
    if out["status"] != "measured":
        out["why"] = (
            f"tag {tags[int(np.argmax(res))]} misses the fit by {res.max():.1f} mm (> {ANCHOR_TAG_MM:g}): a table tag "
            "moved, and with this few tags there is no telling which"
        )
    return out


def _anchor(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout)
    capture = args.capture.resolve()
    episodes = (
        [capture]
        if any(capture.glob("cam_*.mkv"))
        else sorted(p for p in (capture / "episodes").glob("episode_*") if p.is_dir())
    )
    if not episodes:
        print(f"{capture}: no cam_XX.mkv and no episodes/episode_* under it")
        return 2
    datum = json.loads(args.datum.read_text(encoding="utf-8"))
    session_cams, bundle = load_bundle_cameras(args.bundle.resolve(), args.intrinsics)
    ids = {int(t) for v in datum["camera_pixels"].values() for t in v}
    detections = detect_capture(
        episodes, sorted(session_cams), datum["dictionary"], ids, samples=args.samples
    )
    anchor = table_anchor(datum, session_cams, detections)
    m = {
        "episode": str(capture),
        "measured_utc": "",
        "cameras": {
            n: {"corners_px": {str(t): v["corners_px"] for t, v in d["tags"].items()}}
            for n, d in detections.items()
        },
    }
    since = check_cameras(datum, m, float(layout["thresholds"]["static_px"]))
    out = {
        "schema": ANCHOR_SCHEMA,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "capture": str(capture),
        "episodes": [p.name for p in episodes],
        "bundle": bundle,
        "datum": {
            "path": str(args.datum),
            "sha256": _sha256(args.datum),
            "frozen_utc": datum["frozen_utc"],
            "world_frame_id": datum["world_frame_id"],
            "extrinsics_run": datum["calibration"]["extrinsics_run"],
        },
        "meaning": (
            "T_world_session: the bundle's gauge placed in the datum's world by the fixed table tags alone; "
            "T_world_anchor / T_session_anchor are world_registration's --anchor (a frame at the table's centre). "
            "camera_pixels: where each camera saw the table tags during this calibration capture -- what episodes "
            "recorded on this calibration are checked against (session)."
        ),
        **anchor,
        "cameras_since_datum": {
            k: since[k]
            for k in ("status", "cameras_ok", "cameras_moved", "cameras_unverified", "tags_bumped")
        }
        | {"shift_px": {n: v["shift_px"] for n, v in since["cameras"].items() if v["shift_px"]}},
        "frames_sampled": {n: d["sampled"] for n, d in sorted(detections.items())},
        "camera_pixels": {
            n: {str(t): np.asarray(v["corners_px"]).round(4).tolist() for t, v in sorted(d["tags"].items())}
            for n, d in sorted(detections.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(describe_anchor(out))
    print(f"wrote {args.out}")
    return 0 if out["status"] == "measured" else 2


def describe_anchor(a: dict[str, Any]) -> str:
    """One line for a log."""
    if a["status"] not in {"measured", "inconsistent"}:
        return f"table anchor {a['status']}: {a.get('why', '')}"
    head = (
        f"table anchor {a['status']}: tags {a['tags']} fit {a['fit_rms_mm']:.2f} mm rms "
        f"(worst {max(a['tag_residual_mm'].values()):.2f} mm)"
    )
    if a.get("tags_dropped"):
        head += f"; dropped {[d['tag'] for d in a['tags_dropped']]}"
    moved = (a.get("cameras_since_datum") or {}).get("cameras_moved") or []
    if moved:
        head += f"; moved since the datum: {', '.join(moved)}"
    return head


# ----------------------------------------------------------- session check ---


def _video_fingerprint(episode: Path) -> dict[str, list[int]]:
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(episode.glob("cam_*.mkv"))}


def detect_episode(
    episode: Path, cameras: list[str], dictionary: str, ids: set[int], *, stride: int, max_samples: int
) -> dict[str, dict[str, Any]]:
    """``detect_video`` on each of ``cameras`` the episode recorded, one thread per camera (decoding and
    detection release the GIL)."""
    videos = [episode / f"{n}.mkv" for n in cameras if (episode / f"{n}.mkv").exists()]
    if not videos:
        return {}
    with ThreadPoolExecutor(max_workers=len(videos)) as pool:
        found = pool.map(
            lambda v: detect_video(v, dictionary, ids, stride=stride, max_samples=max_samples), videos
        )
        return {v.stem: d for v, d in zip(videos, found, strict=True)}


def episode_table_check(
    reference_pixels: dict[str, dict[str, Any]], detections: dict[str, dict[str, Any]], limit_px: float
) -> dict[str, Any]:
    """One episode's cameras against reference table pixels (a datum's or a calibration capture's), restricted to
    the cameras that have any, with one verdict on top."""
    ref_px = {
        n: {int(t): np.asarray(c, dtype=np.float64) for t, c in v.items()}
        for n, v in reference_pixels.items()
        if v
    }
    cur_px = {
        n: {int(t): np.asarray(v["corners_px"], dtype=np.float64) for t, v in d["tags"].items()}
        for n, d in detections.items()
    }
    ids = sorted({t for v in ref_px.values() for t in v})
    r = static_check(ref_px, cur_px, ids, limit_px, sorted(set(ref_px) | set(detections)))
    verdict = (
        "unchecked"
        if r["status"] == "missing"
        else "table_or_all_cameras_moved"
        if r["status"] != "ok"
        else "cameras_moved"
        if r["cameras_moved"]
        else "ok"
    )
    return {
        "verdict": verdict,
        **{
            k: r[k]
            for k in ("status", "why", "cameras_ok", "cameras_moved", "cameras_unverified", "tags_bumped")
        },
        "cameras_not_recorded": sorted(set(ref_px) - set(detections)),
        "cameras": r["cameras"],
    }


def calibration_reference(
    calibration_root: Path | None, run: str | None
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    """The table pixels a calibration was captured with (``table_anchor.json`` in its extrinsics run), if any."""
    if calibration_root is None or not run:
        return None
    path = calibration_root / f"{run}_extrinsics" / ANCHOR_FILE
    if not path.is_file():
        return None
    anchor = json.loads(path.read_text(encoding="utf-8"))
    pixels = anchor.get("camera_pixels") or {}
    if anchor.get("schema") != ANCHOR_SCHEMA or not any(pixels.values()):
        return None
    meta = {
        "kind": "calibration_capture",
        "extrinsics_run": run,
        "path": str(path),
        "sha256": _sha256(path),
        "measured_utc": anchor.get("measured_utc"),
    }
    return meta, pixels


def _summarise_session(episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdicts = {name: e["verdict"] for name, e in episodes.items()}
    worst: dict[str, float] = {}
    for e in episodes.values():
        for n, v in e["cameras"].items():
            shifts = [s for t, s in v["shift_px"].items() if int(t) not in e["tags_bumped"]]
            if shifts:
                worst[n] = max(worst.get(n, 0.0), max(shifts))
    return {
        "verdict": next((v for v in EPISODE_VERDICTS if v in verdicts.values()), "unchecked"),
        "episodes": len(episodes),
        "episodes_checked": sum(v != "unchecked" for v in verdicts.values()),
        "episode_verdicts": verdicts,
        "cameras_moved": sorted({n for e in episodes.values() for n in e["cameras_moved"]}),
        "tags_bumped": sorted({t for e in episodes.values() for t in e["tags_bumped"]}),
        "worst_shift_px": {n: round(s, 3) for n, s in sorted(worst.items())},
        "references": sorted(
            {json.dumps(e["reference"], sort_keys=True) for e in episodes.values() if e["reference"]}
        ),
        "extrinsics_runs_unreferenced": sorted(
            {str(e["extrinsics_run"]) for e in episodes.values() if not e["reference"]}
        ),
    }


def check_session(
    dataset: Path,
    datum_path: Path,
    layout: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    calibration_root: Path | None = None,
    stride: int = SESSION_STRIDE,
    max_samples: int = SESSION_SAMPLES,
) -> dict[str, Any]:
    """Every episode of a recorded dataset against the table pixels of the calibration it was recorded on.

    That is the run's own calibration capture (``<calibration_root>/<run>_extrinsics/table_anchor.json``, which
    the gateway leaves there since 10-28) or, for the datum's run, the datum.  An episode on a run with neither is
    held against the datum and can say no more than ``other_calibration``.

    An episode's detections are kept from ``previous`` (the file this wrote before) while its videos and what it
    was decoded for are unchanged, so re-running on a growing session only reads the new episodes; the verdict
    is worked out again each time, with the meta.json stamp read again.
    """
    datum = json.loads(datum_path.read_text(encoding="utf-8"))
    limit_px = float(layout["thresholds"]["static_px"])
    settings = {
        "datum_sha256": _sha256(datum_path),
        "limit_px": limit_px,
        "stride": stride,
        "max_samples": max_samples,
    }
    datum_run = str(datum["calibration"]["extrinsics_run"]).removesuffix("_extrinsics")
    datum_reference = (
        {
            "kind": "datum",
            "extrinsics_run": datum_run,
            "path": str(datum_path),
            "sha256": settings["datum_sha256"],
        },
        datum["camera_pixels"],
    )
    references: dict[str | None, Any] = {}

    def reference(run: str | None) -> Any:
        if run not in references:
            references[run] = calibration_reference(calibration_root, run) or (
                datum_reference if run == datum_run else None
            )
        return references[run]

    reuse = (previous or {}).get("episodes_detail", {})
    episodes: dict[str, dict[str, Any]] = {}
    for episode in sorted(p for p in (dataset / "episodes").glob("episode_*") if p.is_dir()):
        run = _episode_calibration(episode)
        run = run.removesuffix("_extrinsics") if run else None
        ref = reference(run)
        ref_meta, ref_pixels = ref or (None, datum["camera_pixels"])
        with_pixels = {n: v for n, v in ref_pixels.items() if v}
        decoded = {
            "videos": _video_fingerprint(episode),
            "cameras": sorted(with_pixels),
            "ids": sorted({int(t) for v in with_pixels.values() for t in v}),
            "stride": stride,
            "max_samples": max_samples,
        }
        before = reuse.get(episode.name) or {}
        if before.get("decoded") == decoded and "detections" in before:
            detections = {
                n: {"tags": {int(t): {"corners_px": np.asarray(c)} for t, c in v.items()}}
                for n, v in before["detections"].items()
            }
        else:
            detections = detect_episode(
                episode,
                decoded["cameras"],
                datum["dictionary"],
                set(decoded["ids"]),
                stride=stride,
                max_samples=max_samples,
            )
        check = episode_table_check(with_pixels, detections, limit_px)
        pixels = check["verdict"]
        verdict = "other_calibration" if ref is None and pixels in {"ok", "cameras_moved"} else pixels
        episodes[episode.name] = {
            **check,
            "verdict": verdict,
            "pixel_verdict": pixels,
            "extrinsics_run": run,
            "reference": ref_meta,
            "decoded": decoded,
            "detections": {
                n: {
                    str(t): np.asarray(v["corners_px"]).round(4).tolist()
                    for t, v in sorted(d["tags"].items())
                }
                for n, d in sorted(detections.items())
            },
        }
    runs = sorted({str(e["extrinsics_run"]) for e in episodes.values()})
    return {
        "schema": SESSION_CHECK_SCHEMA,
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "dataset": str(dataset),
        "datum": {
            "path": str(datum_path),
            "frozen_utc": datum["frozen_utc"],
            "world_frame_id": datum["world_frame_id"],
            "extrinsics_run": datum_run,
        },
        "settings": settings,
        "meaning": (
            "whether each calibrated camera still sees the fixed table tags where it saw them when the extrinsics "
            "the episode was recorded on were captured (pixels, no calibration involved: the run's own "
            "table_anchor.json, or the datum for the datum's run); a moved camera's labels are off by what its "
            "extrinsics no longer describe. Cameras with no table tag in view are not covered (datum "
            "cameras_without_a_table_tag)."
        ),
        "cameras_without_a_table_tag": datum.get("cameras_without_a_table_tag", []),
        "extrinsics_runs": runs,
        **_summarise_session(episodes),
        "episodes_detail": episodes,
    }


def describe_session(s: dict[str, Any]) -> str:
    """One line for a log."""
    head = f"{s['verdict']} ({s['episodes_checked']}/{s['episodes']} episodes checked)"
    if s["cameras_moved"]:
        moved = {
            n: [e for e, d in s["episodes_detail"].items() if n in d["cameras_moved"]]
            for n in s["cameras_moved"]
        }
        head += " moved: " + ", ".join(f"{n} in {len(eps)}" for n, eps in moved.items())
    if s["tags_bumped"]:
        head += f" bumped tags {s['tags_bumped']}"
    if s["worst_shift_px"]:
        head += " worst " + " ".join(
            f"{n.removeprefix('cam_')}:{v:.2f}" for n, v in s["worst_shift_px"].items()
        )
        head += f" px (limit {s['settings']['limit_px']})"
    if s["extrinsics_runs_unreferenced"]:
        head += (
            f"; no table pixels for {','.join(s['extrinsics_runs_unreferenced'])} "
            f"(datum on {s['datum']['extrinsics_run']})"
        )
    return head


def _session(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout)
    flagged = 0
    for dataset in args.datasets:
        dataset = dataset.resolve()
        if not (dataset / "episodes").is_dir():
            print(f"{dataset.name}: no episodes/ (not a recorded GMSL2 dataset)")
            continue
        out = dataset / SESSION_CHECK_PATH
        previous = None if args.force or not out.exists() else json.loads(out.read_text(encoding="utf-8"))
        if previous is not None and previous.get("schema") != SESSION_CHECK_SCHEMA:
            previous = None
        s = check_session(
            dataset, args.datum, layout, previous, calibration_root=args.calibration_root.resolve()
        )
        print(f"{dataset.name}: {describe_session(s)}")
        flagged += s["verdict"] in {"cameras_moved", "table_or_all_cameras_moved"}
        if args.write:
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.tmp")
            tmp.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
            tmp.replace(out)
    return 1 if flagged else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="triangulate the cart/table tags of one recorded episode (cv2)")
    m.add_argument("episode", type=Path, help="episodes/episode_XXXXXX dir holding cam_XX.mkv")
    m.add_argument("--calibration-root", type=Path, default=Path("outputs/calibration"))
    m.add_argument(
        "--calibration-run",
        default=None,
        help="e.g. calib_20261014_180431_extrinsics; default: the one in the episode's meta.json",
    )
    m.add_argument("--stride", type=int, default=5, help="use every n-th frame")
    m.add_argument("--max-samples", type=int, default=60)
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="a capture after a re-dock against the reference (cv2)")
    c.add_argument("reference", type=Path, help="measure output taken right after the registration")
    c.add_argument("current", type=Path, help="measure output after the re-dock")
    c.add_argument("--calibration-root", type=Path, default=Path("outputs/calibration"))
    c.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("freeze-datum", help="keep one measurement's table tags as the table datum")
    f.add_argument("measurement", type=Path, help="measure output of a capture with the table clear")
    f.add_argument("--source", required=True, help="the recording it came from, e.g. the Thor dataset name")
    f.add_argument("--out", type=Path, default=DATUM_PATH)
    f.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("outputs/calibration"),
        help="the measurement's calibration is kept in the datum (anchor_cameras)",
    )
    f.add_argument(
        "--replace", action="store_true", help="replace an existing datum (it is recorded as replaced)"
    )
    k = sub.add_parser("cameras", help="which cameras moved since the table datum (pixels only)")
    k.add_argument("measurement", type=Path, help="measure output of any later capture")
    k.add_argument("--datum", type=Path, default=DATUM_PATH)
    k.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("session", help="every episode of recorded datasets against the table datum (cv2)")
    s.add_argument("datasets", type=Path, nargs="+", help="outputs/datasets/<name> dirs holding episodes/")
    s.add_argument("--datum", type=Path, default=DATUM_PATH)
    s.add_argument("--write", action="store_true", help=f"keep the result in <dataset>/{SESSION_CHECK_PATH}")
    s.add_argument("--force", action="store_true", help="re-read every episode, not only the changed ones")
    s.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("outputs/calibration"),
        help=f"where each episode's extrinsics run (and its {ANCHOR_FILE}) is",
    )
    a = sub.add_parser("anchor", help="a calibration's bundle placed in the world by the table tags (cv2)")
    a.add_argument(
        "capture", type=Path, help="the calibration capture: a dir of cam_XX.mkv, or one with episodes/"
    )
    a.add_argument("--bundle", type=Path, required=True, help="the bundle's extrinsics_report.json")
    a.add_argument("--intrinsics", type=Path, default=None, help="instead of the report's intrinsics_source")
    a.add_argument("--datum", type=Path, default=DATUM_PATH)
    a.add_argument("--samples", type=int, default=ANCHOR_SAMPLES, help="frames read per camera")
    a.add_argument("--out", type=Path, required=True, help=f"conventionally <bundle dir>/{ANCHOR_FILE}")
    args = parser.parse_args(argv)
    return {
        "measure": _measure,
        "compare": _compare,
        "freeze-datum": _freeze_datum,
        "cameras": _cameras,
        "session": _session,
        "anchor": _anchor,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
