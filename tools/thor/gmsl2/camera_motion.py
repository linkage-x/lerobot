"""Which cameras moved since their extrinsics, read off the recording itself -- and which to leave out.

``metrology.camera_state`` says *that* a recording's extrinsics no longer fit (corner BA over 1 px) but
not *which* camera stopped fitting, and the table tags (``cart_redock_check.py session``) cover only
the cameras that see one: cam_06/08/14 see none. 2026-11-13 cam_03/13/14 moved 1.3-2.4 px overnight
(one bracket), the other five did not, and every label since carried the three stale poses.

This compares each camera's static background with a **reference recording**: the latest dataset
next to this one, recorded before it, tracked on the same extrinsics with every camera, and judged
``ok`` by the camera state then. Whatever that recording saw, the extrinsics fitted. Per camera:

* the median of ``SAMPLES`` frames spread over an episode -- people, hands and the rig move, the
  room does not, so they drop out;
* SIFT matches against the reference camera's median image, a RANSAC homography (1 px), and the
  median displacement of the inliers. Fixed table pixels moved 0.06-0.26 px between recordings of
  cameras that had not moved and 1.3-2.4 px for the three that had (thor_gmsl2_9ch_v1_20261112_182936
  against 20261113_183417 / 20261116_110051; the table tags agreed where they were seen).

Verdicts: ``moved`` (shift >= ``MOVED_PX``), ``stable``, or ``unverified`` (too few inliers, no video
on either side). The first and the last episode are both checked; the worse verdict counts.

The selection, written as ``excluded_cameras``:

* nothing moved -> leave nothing out: this is every ordinary recording, tracked exactly as before;
* something moved -> track with the ``stable`` cameras only. An ``unverified`` camera is left out
  too: it could not be shown to be where the extrinsics put it;
* fewer than ``MIN_KEPT`` stable cameras -> leave nothing out and say so; the labels stay stale and
  the camera state will report it. Recalibrate.

Measured on the right gripper rig (2026-11-16, three episodes each): leaving out cam_03/13/14 from a
recording where all eight were still valid moved the TCP 0.22 mm median / 0.42 mm p95 and 0.14 deg,
every frame still solved by >= 3 cameras; on the recordings after the move it brought corner BA from
1.44-1.68 px back to 0.88 px (0.85 px before the move).

Run in the opencv_kalibr venv (cv2 with SIFT, numpy, yaml); the tracking runner calls it before the
tracker when asked to (``run_april_cube_tracking_local.sh --auto-exclude-moved-cameras``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "camera_motion/v1"
SIDECAR = "april_cube_tracking_in_robot_base"
OUTPUT = Path("derived") / "camera_motion.json"
#: Median background shift at which a camera counts as moved. The table check's static limit; the
#: cameras that had not moved stayed under 0.26 px, the ones that had were at 1.3 px and more.
MOVED_PX = 0.6
#: RANSAC inliers a camera needs before its shift means anything.
MIN_INLIERS = 80
#: Fewest cameras a selection may keep: four gave >= 3 cameras on every frame of the right rig.
MIN_KEPT = 4
SAMPLES = 25
RANSAC_PX = 1.0
SIFT_FEATURES = 4000
_VERDICT_ORDER = ("moved", "unverified", "stable")
_STAMP = re.compile(r"(\d{8}_\d{6})$")
TRACKER_SCRIPTS = Path("third_party/opencv_kalibr/hikon_cube_tracking_offline")


# ------------------------------------------------------------------ recordings ---


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def episode_dirs(dataset: Path) -> list[Path]:
    root = dataset / "episodes"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))


def recorded_at(dataset: Path) -> datetime | None:
    """When the recording started, for ordering siblings: the name's stamp (every Thor dataset has one,
    in the station's clock), else its first episode's wallclock. One source per comparison would be
    better still; the stamp comes first so that two named datasets never mix clocks."""
    match = _STAMP.search(dataset.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)  # noqa: UP017 -- Python 3.10 in the opencv_kalibr venv
    episodes = episode_dirs(dataset)
    if episodes:
        stamp = _read_json(episodes[0] / "meta.json").get("wallclock_start_utc")
        if isinstance(stamp, str):
            try:
                moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)  # noqa: UP017 -- Python 3.10 in the opencv_kalibr venv
            except ValueError:
                pass
    return None


def camera_state_of(dataset: Path) -> dict[str, Any] | None:
    state = ((_read_json(dataset / "meta" / "info.json").get("derived") or {}).get(SIDECAR) or {}).get(
        "camera_state"
    )
    return state if isinstance(state, dict) else None


def is_reference(dataset: Path, extrinsics_run: str) -> bool:
    """Tracked on ``extrinsics_run`` with every camera, and the extrinsics fitted (camera state ok)."""
    state = camera_state_of(dataset)
    return bool(
        state
        and state.get("verdict") == "ok"
        and str(state.get("extrinsics_run_name") or "") == extrinsics_run
        and not state.get("excluded_cameras")
        and episode_dirs(dataset)
    )


def find_reference(dataset: Path, extrinsics_run: str, candidates: list[Path] | None = None) -> Path | None:
    """The latest reference recording (:func:`is_reference`) started before ``dataset``."""
    when = recorded_at(dataset)
    if when is None:
        return None
    pool = candidates if candidates is not None else [p for p in dataset.parent.iterdir() if p.is_dir()]
    best: tuple[datetime, Path] | None = None
    for other in pool:
        if other.resolve() == dataset.resolve() or not is_reference(other, extrinsics_run):
            continue
        other_when = recorded_at(other)
        if other_when is None or other_when >= when:
            continue
        if best is None or other_when > best[0]:
            best = (other_when, other)
    return None if best is None else best[1]


# ------------------------------------------------------------------ images ---


def median_image(video: Path, samples: int = SAMPLES) -> np.ndarray | None:
    """Grey median of ``samples`` frames spread over the video: what in its view did not move."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for index in np.unique(np.linspace(0, max(total - 1, 0), samples).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if len(frames) < max(3, samples // 2):
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def background_shift(reference: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """Median displacement of the background from ``reference`` to ``current``, in pixels."""
    import cv2

    sift = cv2.SIFT_create(SIFT_FEATURES)
    kr, dr = sift.detectAndCompute(reference, None)
    kc, dc = sift.detectAndCompute(current, None)
    if dr is None or dc is None or len(kr) < 8 or len(kc) < 8:
        return {"inliers": 0}
    pairs = cv2.BFMatcher().knnMatch(dr, dc, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
    if len(good) < 8:
        return {"inliers": 0}
    pr = np.float32([kr[m.queryIdx].pt for m in good])
    pc = np.float32([kc[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pr, pc, cv2.RANSAC, RANSAC_PX)
    if mask is None:
        return {"inliers": 0}
    keep = mask.ravel().astype(bool)
    d = pc[keep] - pr[keep]
    if len(d) == 0:
        return {"inliers": 0}
    return {
        "inliers": int(keep.sum()),
        "shift_px": round(float(np.median(np.linalg.norm(d, axis=1))), 3),
        "shift_vector_px": [round(float(v), 3) for v in np.median(d, axis=0)],
    }


def verdict_of(measure: dict[str, Any], moved_px: float = MOVED_PX, min_inliers: int = MIN_INLIERS) -> str:
    if int(measure.get("inliers") or 0) < min_inliers or measure.get("shift_px") is None:
        return "unverified"
    return "moved" if float(measure["shift_px"]) >= moved_px else "stable"


def worst(verdicts: list[str]) -> str:
    return min(verdicts, key=_VERDICT_ORDER.index) if verdicts else "unverified"


# ------------------------------------------------------------------ selection ---


def select(verdicts: dict[str, str], min_kept: int = MIN_KEPT) -> tuple[list[str], str]:
    """``(excluded cameras, decision)`` from per-camera verdicts (see the module doc)."""
    moved = sorted(c for c, v in verdicts.items() if v == "moved")
    if not moved:
        return [], "none_moved"
    stable = sorted(c for c, v in verdicts.items() if v == "stable")
    if len(stable) < min_kept:
        return [], "too_few_stable"
    return sorted(c for c in verdicts if c not in stable), "excluded"


def extrinsics_cameras(summary_path: Path) -> list[str]:
    summary = _read_json(summary_path)
    cams = (summary.get("joint_solution") or {}).get("cameras") or summary.get("cameras") or {}
    return sorted(str(c) for c in cams) if isinstance(cams, dict) else []


def check(
    dataset: Path,
    extrinsics_run: str,
    cameras: list[str],
    *,
    reference: Path | None = None,
    candidates: list[Path] | None = None,
    moved_px: float = MOVED_PX,
    min_inliers: int = MIN_INLIERS,
    min_kept: int = MIN_KEPT,
    samples: int = SAMPLES,
) -> dict[str, Any]:
    """The camera-motion record of ``dataset`` (what :data:`OUTPUT` holds)."""
    reference = reference or find_reference(dataset, extrinsics_run, candidates)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10 in the opencv_kalibr venv
        "dataset": str(dataset),
        "extrinsics_run": extrinsics_run,
        "thresholds": {"moved_px": moved_px, "min_inliers": min_inliers, "min_kept": min_kept, "samples": samples},
        "reference": None,
        "cameras": {},
        "excluded_cameras": [],
    }
    episodes = episode_dirs(dataset)
    if reference is None or not episodes:
        record["decision"] = "no_reference" if reference is None else "no_episodes"
        record["message"] = (
            f"no reference recording on {extrinsics_run} (tracked with every camera, camera state ok, "
            "recorded before this one): camera motion not checked, every camera used"
            if reference is None
            else "no episodes: camera motion not checked, every camera used"
        )
        return record
    ref_episode = episode_dirs(reference)[-1]
    targets = [episodes[0]] if len(episodes) == 1 else [episodes[0], episodes[-1]]
    record["reference"] = {
        "dataset": str(reference),
        "episode": ref_episode.name,
        "camera_state": camera_state_of(reference),
    }

    def one(camera: str) -> tuple[str, dict[str, Any]]:
        ref_video = ref_episode / f"{camera}.mkv"
        ref_img = median_image(ref_video, samples) if ref_video.is_file() else None
        by_episode: dict[str, Any] = {}
        for episode in targets:
            video = episode / f"{camera}.mkv"
            img = median_image(video, samples) if (ref_img is not None and video.is_file()) else None
            measure = {"inliers": 0} if img is None else background_shift(ref_img, img)
            measure["verdict"] = verdict_of(measure, moved_px, min_inliers)
            by_episode[episode.name] = measure
        verdicts = [m["verdict"] for m in by_episode.values()]
        shifts = [m["shift_px"] for m in by_episode.values() if m.get("shift_px") is not None]
        return camera, {
            "verdict": worst(verdicts),
            "shift_px": max(shifts) if shifts else None,
            "by_episode": by_episode,
        }

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(cameras)))) as pool:
        record["cameras"] = dict(sorted(pool.map(one, cameras)))
    verdicts = {c: v["verdict"] for c, v in record["cameras"].items()}
    excluded, decision = select(verdicts, min_kept)
    record["excluded_cameras"] = excluded
    record["decision"] = decision
    record["moved"] = sorted(c for c, v in verdicts.items() if v == "moved")
    record["unverified"] = sorted(c for c, v in verdicts.items() if v == "unverified")
    record["message"] = describe(record)
    return record


def describe(record: dict[str, Any]) -> str:
    if record.get("decision") in {"no_reference", "no_episodes"}:
        return str(record.get("message") or record["decision"])
    ref = Path(str((record.get("reference") or {}).get("dataset") or "?")).name
    moved = ", ".join(
        f"{c} {record['cameras'][c]['shift_px']:.2f} px" for c in record.get("moved") or []
    )
    if record["decision"] == "none_moved":
        line = f"camera motion against {ref}: no camera moved"
        if record.get("unverified"):
            line += f" ({', '.join(record['unverified'])} unverified)"
        return line + "; every camera used"
    if record["decision"] == "too_few_stable":
        return (
            f"camera motion against {ref}: moved {moved}, but fewer than {record['thresholds']['min_kept']} "
            "cameras verified stable: every camera used, labels stay stale -- recalibrate"
        )
    return (
        f"camera motion against {ref}: moved {moved}; tracking without "
        f"{', '.join(record['excluded_cameras'])}"
    )


# ------------------------------------------------------------------ cli ---


def _tracking_calibration(config: Path, repo_root: Path) -> tuple[str, Path]:
    """(extrinsics run, its summary.json) as the tracker resolves them from ``config``."""
    import yaml

    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    calib = cfg.get("calibration") or {}
    run = str(calib.get("fixed_camera_run_name") or "").strip()
    root_text = str(calib.get("root_dir") or "outputs/calibration")
    root = Path(root_text)
    if not root.is_absolute():
        # resolve_from_scripts: as given from the cwd (the runner's is the repo root), else beside
        # the tracker scripts.
        root = (repo_root / root) if (repo_root / root).exists() else repo_root / TRACKER_SCRIPTS / root
    return run, root / run / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path, help="raw Thor dataset (episodes/episode_*/cam_XX.mkv)")
    parser.add_argument("--config", type=Path, required=True, help="the tracking config the tracker will run")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, default=None, help="instead of the automatic reference")
    parser.add_argument("--write", action="store_true", help=f"keep the record in <dataset>/{OUTPUT}")
    parser.add_argument("--moved-px", type=float, default=MOVED_PX)
    parser.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    parser.add_argument("--min-kept", type=int, default=MIN_KEPT)
    args = parser.parse_args(argv)

    run, summary = _tracking_calibration(args.config, args.repo_root)
    cameras = extrinsics_cameras(summary)
    if not run or not cameras:
        print(f"[WARN] camera motion: no cameras in {summary}; not checked", file=sys.stderr)
        return 1
    record = check(
        args.dataset,
        run,
        cameras,
        reference=args.reference,
        moved_px=args.moved_px,
        min_inliers=args.min_inliers,
        min_kept=args.min_kept,
    )
    for camera, entry in record["cameras"].items():
        shift = "-" if entry["shift_px"] is None else f"{entry['shift_px']:.2f} px"
        print(f"[INFO] camera motion {camera}: {entry['verdict']} ({shift})")
    level = "WARN" if record.get("moved") else "INFO"
    print(f"[{level}] {record['message']}")
    if args.write:
        out = args.dataset / OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] camera motion record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
