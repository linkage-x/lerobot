"""
Extract right-side observations from HIROL episodes without trimming.

Usage:
  python -m src.lerobot.datasets.hirol.utils.extract_right \
      -i /data/2025/unitree_co-train/insert_tube/0109_duo_unitree_tube_insert_human_282ep \
      -o /data/2025/unitree_co-train/insert_tube/0109_duo_unitree_tube_insert_human_282ep_right \
      --keep-cam right

What it does
- For each episode, keep only selected camera data (default: right) and
  right/single arm fields (ee_states, joint_states, actions, tools, tactiles, imus).
- Copy only referenced images for kept cameras (and depths if present).
- Frames missing required kept color images are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import glog as log


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


def _match_keep(key: str, keep_patterns: Iterable[str]) -> bool:
    low = key.lower()
    for pat in keep_patterns:
        if pat and pat.lower() in low:
            return True
    return False


def _filter_keep_cameras(step: Dict, keep_cam_patterns: Iterable[str]) -> Dict:
    """Filter a single step to keep only selected camera(s) and right/single arm fields."""
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
    # joint_states
    jss = out.get("joint_states", None)
    if isinstance(jss, dict) and len(jss) > 0:
        if "right" in jss:
            out["joint_states"] = {"right": jss["right"]}
        elif "single" in jss:
            out["joint_states"] = {"single": jss["single"]}
        else:
            out["joint_states"] = {}
    # actions
    acts = out.get("actions", None)
    if isinstance(acts, dict) and len(acts) > 0:
        if "right" in acts:
            out["actions"] = {"right": acts["right"]}
        elif "single" in acts:
            out["actions"] = {"single": acts["single"]}
        else:
            out["actions"] = {}
    # tools
    tools = out.get("tools", None)
    if isinstance(tools, dict) and len(tools) > 0:
        if "right" in tools:
            out["tools"] = {"right": tools["right"]}
        elif "single" in tools:
            out["tools"] = {"single": tools["single"]}
        else:
            out["tools"] = {}
    # tactiles
    tacs = out.get("tactiles", None)
    if isinstance(tacs, dict) and len(tacs) > 0:
        if "right" in tacs:
            out["tactiles"] = {"right": tacs["right"]}
        elif "single" in tacs:
            out["tactiles"] = {"single": tacs["single"]}
        else:
            out["tactiles"] = {}
    # imus
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
    """Return list of (rel_path, type) to copy. type in {"colors","depths"}."""
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


def _extract_one_episode(
    src_ep_dir: str,
    dst_ep_dir: str,
    reindex: bool = True,
    keep_cam_patterns: Iterable[str] = ("right",),
) -> bool:
    src_json = os.path.join(src_ep_dir, "data.json")
    obj = _load_json(src_json)
    if obj is None:
        return False

    frames = obj.get("data", []) or []
    if len(frames) == 0:
        log.info(f"Skip empty episode: {src_ep_dir}")
        return False

    extracted: List[Dict] = []
    idx_counter = 0
    for st in frames:
        filt = _filter_keep_cameras(st, keep_cam_patterns)
        if not _validate_step_images_exist(filt, src_ep_dir):
            # Skip frames that lack required files
            continue
        if reindex:
            try:
                filt["idx"] = int(idx_counter)
            except Exception:
                filt["idx"] = int(idx_counter)
        extracted.append(filt)
        idx_counter += 1

    if len(extracted) == 0:
        log.info(f"Episode becomes empty after extraction: {src_ep_dir}")
        return False

    img_paths = _collect_image_paths(extracted, keep_cam_patterns)
    _copy_selected_images(src_ep_dir, dst_ep_dir, img_paths)

    new_obj = dict(obj)
    new_obj["data"] = extracted
    dst_json = os.path.join(dst_ep_dir, "data.json")
    _save_json(new_obj, dst_json)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract right-side observations in HIROL episodes")
    parser.add_argument("-i", "--input", required=True, help="Input task dir that contains episode_XXXX")
    parser.add_argument("-o", "--output", default=None, help="Output dir; default: <input>_right")
    parser.add_argument("--keep_idx", action="store_true", help="Keep original per-step idx instead of reindexing from 0")
    parser.add_argument(
        "--keep-cam",
        dest="keep_cam",
        default="right",
        help="Comma-separated substrings of camera keys to keep (e.g., 'right' or 'right,mid')",
    )
    args = parser.parse_args()

    in_root = _expand(args.input)
    out_root = _expand(args.output) if args.output else f"{in_root}_right"
    if not os.path.isdir(in_root):
        raise SystemExit(f"Input dir not found: {in_root}")
    os.makedirs(out_root, exist_ok=True)

    ep_dirs = _episode_dirs(in_root)
    if len(ep_dirs) == 0:
        log.warning(f"No episodes under: {in_root}")
        return

    n_ok = 0
    keep_cam_patterns = tuple([s.strip() for s in (args.keep_cam or "").split(",") if len(s.strip()) > 0]) or ("right",)
    for ep in ep_dirs:
        ep_name = os.path.basename(ep)
        dst = os.path.join(out_root, ep_name)
        os.makedirs(dst, exist_ok=True)
        try:
            ok = _extract_one_episode(
                ep,
                dst,
                reindex=(not args.keep_idx),
                keep_cam_patterns=keep_cam_patterns,
            )
            if ok:
                n_ok += 1
        except Exception as e:
            log.error(f"Failed to extract {ep_name}: {e}")

    log.info(f"Done. Episodes processed: {n_ok}/{len(ep_dirs)}. Output: {out_root}")


if __name__ == "__main__":
    main()
