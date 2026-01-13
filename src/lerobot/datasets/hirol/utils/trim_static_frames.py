"""
Trim leading static frames from HIROL episodes and keep only specific camera data (default: right).

Usage:
  python -m src.lerobot.datasets.hirol.utils.trim_static_frames \
      -i ~/Data_Collection/1224_duo_unitree_bread_picking_robot_93ep \
      -o ~/Data_Collection/1224_duo_unitree_bread_picking_robot_93ep_trimmed \
      --win 10 --th 0.01 --keep-cam right

What it does
- Per episode, scan the beginning with a sliding window. As soon as we detect
  noticeable end-effector motion (translation jump >= threshold inside the
  window), we drop all frames before the window (i.e. keep from the window's
  left boundary). Conceptually: --------------------[----]++++++++++++++++
  We remove the left side and keep from '[' onward.
- Save a trimmed data.json and copy only referenced images of the kept camera(s)
  (also depths if present). All other cameras (left/mid/head, etc.) are
  removed from JSON and not copied.

Notes
- EE pose source priority: use ee_states['right'] if present; otherwise use
  ee_states['single'] when datasets encode a single EE. If neither exist,
  trimming is skipped for that episode.
- We also re-index the per-step 'idx' field starting from 0 in the trimmed
  output to keep it compact and monotonic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple, Iterable

import glog as log
import numpy as np


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _episode_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith("episode_") and os.path.isdir(os.path.join(root, name)):
            out.append(os.path.join(root, name))
    return out


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to read JSON: {path}: {e}")
        return None


def _save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _extract_ee_positions(frames: List[Dict]) -> Optional[np.ndarray]:
    """Return Nx3 numpy array of EE positions from frames.

    Priority: ee_states['right']['pose'][:3] -> ee_states['single']['pose'][:3].
    Returns None if not available.
    """
    pos_list: List[List[float]] = []
    for st in frames:
        ee = st.get("ee_states", {}) or {}
        pose = None
        if isinstance(ee, dict):
            if "right" in ee and isinstance(ee["right"], dict):
                pose = ee["right"].get("pose", None)
            elif "single" in ee and isinstance(ee["single"], dict):
                pose = ee["single"].get("pose", None)
        if pose is None or not isinstance(pose, (list, tuple)) or len(pose) < 3:
            pos_list.append(None)  # keep alignment; may skip later
        else:
            pos_list.append([float(pose[0]), float(pose[1]), float(pose[2])])
    # If nothing usable, return None
    if all(p is None for p in pos_list):
        return None
    # Replace missing entries by nearest previous valid position to avoid breaking the scan
    last = None
    out: List[List[float]] = []
    for p in pos_list:
        if p is None:
            if last is None:
                # still None at the beginning; skip building positions
                return None
            out.append(last)
        else:
            last = p
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def _find_trim_index(
    positions: np.ndarray,
    window: int = 10,
    move_thresh_m: float = 0.01,
) -> int:
    """Find the trim start index in the given positions array.

    Rule 1: from-start threshold crossing: earliest t with ||pos[t]-pos[0]|| >= th.
    Rule 2: fallback sliding window: first i where max||pos[i:j]-pos[i]|| >= th.

    Returns the index in [0, N). If not found, returns 0.
    """
    n = int(positions.shape[0])
    if n <= 1:
        return 0
    # Rule 1
    d0 = np.linalg.norm(positions - positions[0][None, :], axis=1)
    idxs = np.nonzero(d0 >= move_thresh_m)[0]
    if idxs.size > 0:
        return int(idxs[0])
    # Rule 2
    w = max(2, min(int(window), n))
    for i in range(0, n - 1):
        j_end = min(i + w, n)
        dists = np.linalg.norm(positions[i:j_end] - positions[i][None, :], axis=1)
        if np.max(dists) >= move_thresh_m:
            return int(i)
    return 0


def _match_keep(key: str, keep_patterns: Iterable[str]) -> bool:
    low = key.lower()
    for pat in keep_patterns:
        if pat and pat.lower() in low:
            return True
    return False


def _filter_keep_cameras(step: Dict, keep_cam_patterns: Iterable[str]) -> Dict:
    """Filter a single step to keep only selected camera(s) and right/single ee.

    - colors/depths: keep entries whose key name contains any pattern in
      keep_cam_patterns (case-insensitive). Default pattern is ['right'].
    - ee_states: prefer 'right'; else keep 'single' as-is; drop others.
    - tools: prefer 'right'; else keep 'single' as-is; drop others.
    Other fields are passed through unchanged.
    """
    out = dict(step)
    # colors
    cols = out.get("colors", None)
    if isinstance(cols, dict):
        kept = {k: v for k, v in cols.items() if _match_keep(k, keep_cam_patterns)}
        out["colors"] = kept
    # depths
    deps = out.get("depths", None)
    if isinstance(deps, dict):
        kept = {k: v for k, v in deps.items() if _match_keep(k, keep_cam_patterns)}
        out["depths"] = kept
    # ee_states
    ees = out.get("ee_states", None)
    if isinstance(ees, dict):
        if "right" in ees:
            out["ee_states"] = {"right": ees["right"]}
        elif "single" in ees:
            out["ee_states"] = {"single": ees["single"]}
        else:
            out["ee_states"] = {}
    # joint_states: keep right only when present (best-effort)
    jss = out.get("joint_states", None)
    if isinstance(jss, dict) and len(jss) > 0:
        if "right" in jss:
            out["joint_states"] = {"right": jss["right"]}
        elif "single" in jss:
            out["joint_states"] = {"single": jss["single"]}
        else:
            # keep empty or untouched if structure unknown
            out["joint_states"] = {}
    # actions: keep right or single entries, drop others (left/head)
    acts = out.get("actions", None)
    if isinstance(acts, dict) and len(acts) > 0:
        if "right" in acts:
            out["actions"] = {"right": acts["right"]}
        elif "single" in acts:
            out["actions"] = {"single": acts["single"]}
        else:
            # If actions keyed by multiple arms, drop non-right
            out["actions"] = {}
    # tools: keep right or single entries, drop others (left)
    tools = out.get("tools", None)
    if isinstance(tools, dict) and len(tools) > 0:
        if "right" in tools:
            out["tools"] = {"right": tools["right"]}
        elif "single" in tools:
            out["tools"] = {"single": tools["single"]}
        else:
            out["tools"] = {}
    # tactiles: keep right/single if present
    tacs = out.get("tactiles", None)
    if isinstance(tacs, dict) and len(tacs) > 0:
        if "right" in tacs:
            out["tactiles"] = {"right": tacs["right"]}
        elif "single" in tacs:
            out["tactiles"] = {"single": tacs["single"]}
        else:
            out["tactiles"] = {}
    # imus: keep right/single if present (often single)
    imus = out.get("imus", None)
    if isinstance(imus, dict) and len(imus) > 0:
        if "right" in imus:
            out["imus"] = {"right": imus["right"]}
        elif "single" in imus:
            out["imus"] = {"single": imus["single"]}
        else:
            out["imus"] = {}
    return out


def _collect_image_paths(frames: List[Dict], keep_cam_patterns: Iterable[str]) -> List[Tuple[str, str]]:
    """Return list of (rel_path, type) to copy. type in {"colors","depths"}.

    Include only entries of colors/depths whose key matches any keep_cam_patterns.
    """
    paths: List[Tuple[str, str]] = []
    for st in frames:
        for key in ("colors", "depths"):
            dic = st.get(key, {}) or {}
            for cam_key, meta in dic.items():
                if not _match_keep(cam_key, keep_cam_patterns):
                    continue
                if isinstance(meta, dict):
                    p = meta.get("path", None)
                    if isinstance(p, str) and len(p) > 0:
                        paths.append((p, key))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _copy_selected_images(src_ep_dir: str, dst_ep_dir: str, rel_paths: List[Tuple[str, str]]) -> None:
    for rel, kind in rel_paths:
        src = os.path.join(src_ep_dir, rel)
        dst = os.path.join(dst_ep_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            log.warning(f"Failed to copy {kind} image {src} -> {dst}: {e}")


def _validate_step_images_exist(step: Dict, src_ep_dir: str) -> bool:
    """Return True if all kept color images exist on disk (required), and if
    kept depths exist when present (optional). Frames without any kept color
    image are considered invalid and will be skipped.
    """
    cols = step.get("colors", {}) or {}
    if not isinstance(cols, dict) or len(cols) == 0:
        return False
    for meta in cols.values():
        if not isinstance(meta, dict):
            return False
        p = meta.get("path")
        if not p or not os.path.exists(os.path.join(src_ep_dir, p)):
            return False
    deps = step.get("depths", None)
    if isinstance(deps, dict) and len(deps) > 0:
        for meta in deps.values():
            if not isinstance(meta, dict):
                return False
            p = meta.get("path")
            if not p or not os.path.exists(os.path.join(src_ep_dir, p)):
                return False
    return True


def _trim_one_episode(
    src_ep_dir: str,
    dst_ep_dir: str,
    win: int,
    thresh: float,
    reindex: bool = True,
    keep_cam_patterns: Iterable[str] = ("right",),
    ignore_first: int = 0,
) -> bool:
    src_json = os.path.join(src_ep_dir, "data.json")
    obj = _load_json(src_json)
    if obj is None:
        return False

    frames = obj.get("data", []) or []
    if len(frames) == 0:
        log.info(f"Skip empty episode: {src_ep_dir}")
        return False

    positions = _extract_ee_positions(frames)
    if positions is None:
        log.warning(f"No usable ee pose in {src_ep_dir}; keep all frames but filter to right-only")
        keep_start = max(0, int(ignore_first))
    else:
        ig = max(0, int(ignore_first))
        if ig >= len(positions):
            keep_start = len(positions)
        else:
            sub = positions[ig:]
            keep_start = ig + _find_trim_index(sub, window=win, move_thresh_m=thresh)
        try:
            d0 = np.linalg.norm(positions - positions[0][None, :], axis=1)
            dbg = ", ".join([f"{v:.4f}" for v in d0[: min(8, len(d0))]])
            log.info(
                f"{os.path.basename(src_ep_dir)}: trim_start={keep_start}, ignore_first={ig}, d0[:8]=[{dbg}] (th={thresh}, win={win})"
            )
        except Exception:
            pass

    trimmed: List[Dict] = []
    idx_counter = 0
    for st in frames[keep_start:]:
        filt = _filter_keep_cameras(st, keep_cam_patterns)
        if not _validate_step_images_exist(filt, src_ep_dir):
            # skip frames that lack required files
            continue
        if reindex:
            try:
                filt["idx"] = int(idx_counter)
            except Exception:
                filt["idx"] = int(idx_counter)
        trimmed.append(filt)
        idx_counter += 1

    if len(trimmed) == 0:
        log.info(f"Episode becomes empty after trim: {src_ep_dir}")
        return False

    # Copy only referenced selected-camera images
    img_paths = _collect_image_paths(trimmed, keep_cam_patterns)
    _copy_selected_images(src_ep_dir, dst_ep_dir, img_paths)

    # Write JSON
    new_obj = dict(obj)
    new_obj["data"] = trimmed
    dst_json = os.path.join(dst_ep_dir, "data.json")
    _save_json(new_obj, dst_json)
    return True


def main():
    parser = argparse.ArgumentParser(description="Trim leading static frames in HIROL episodes and keep selected cameras only")
    parser.add_argument("-i", "--input", required=True, help="Input task dir that contains episode_XXXX")
    parser.add_argument("-o", "--output", default=None, help="Output dir; default: <input>_trimmed")
    parser.add_argument("--win", type=int, default=10, help="Sliding window size (frames)")
    parser.add_argument("--th", type=float, default=0.01, help="Movement threshold in meters (translation)")
    parser.add_argument("--keep_idx", action="store_true", help="Keep original per-step idx instead of reindexing from 0")
    parser.add_argument("--keep-cam", dest="keep_cam", default="right",
                        help="Comma-separated substrings of camera keys to keep (e.g., 'right' or 'right,mid')")
    parser.add_argument("--ignore-first", dest="ignore_first", type=int, default=300,
                        help="Ignore first N frames when detecting movement (reduces startup jitter)")
    args = parser.parse_args()

    in_root = _expand(args.input)
    out_root = _expand(args.output) if args.output else f"{in_root}_trimmed"
    if not os.path.isdir(in_root):
        raise SystemExit(f"Input dir not found: {in_root}")
    os.makedirs(out_root, exist_ok=True)

    ep_dirs = _episode_dirs(in_root)
    if len(ep_dirs) == 0:
        log.warning(f"No episodes under: {in_root}")
        return

    n_ok = 0
    keep_cam_patterns = tuple([s.strip() for s in (args.keep_cam or '').split(',') if len(s.strip()) > 0]) or ("right",)
    for ep in ep_dirs:
        ep_name = os.path.basename(ep)
        dst = os.path.join(out_root, ep_name)
        os.makedirs(dst, exist_ok=True)
        try:
            ok = _trim_one_episode(
                ep, dst,
                win=args.win,
                thresh=args.th,
                reindex=(not args.keep_idx),
                keep_cam_patterns=keep_cam_patterns,
                # Ignore first N frames in detection; still keep frames before trim boundary
                # if trimming decides to start after ignore_first.
                # The trimming boundary is computed on positions[ignore_first:].
                ignore_first=args.ignore_first,
            )
            if ok:
                n_ok += 1
        except Exception as e:
            log.error(f"Failed to trim {ep_name}: {e}")

    log.info(f"Done. Episodes processed: {n_ok}/{len(ep_dirs)}. Output: {out_root}")


if __name__ == "__main__":
    main()
