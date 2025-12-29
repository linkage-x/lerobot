"""
Convert DAS-Datakit MCAP to HIROL JSON+images episodes.

This tool reads DAS MCAP files via dependencies/das-datakit utils, aligns
sensor streams (EEF pose, gripper, IMU, tactile) to a reference camera timebase,
optionally downsamples by FPS, and writes per-episode folders following the
HIROL format expected by reader.RerunEpisodeReader.

References
- dependencies/das-datakit/README.md (topics and semantics)
- dependencies/lerobot/src/lerobot/datasets/hirol/reader.py (HIROL schema)
- dependencies/lerobot/src/lerobot/datasets/hirol/das2lerobot_converter.md (alignment rules)

CLI
  python das2hirol_converter.py -c path/to/config.yaml

Config (YAML) minimal example
  input:
    mcap_file: ""             # or set task_dir
    task_dir: ""              # dir containing vio_result.json (success_mcap_files)
  output:
    task_dir: "../dataset/data/source"   # where episode_0000 ... are written
  topics:
    ref: "/robot0/sensor/camera0/compressed"
    cameras:
      - topic: "/robot0/sensor/camera0/compressed"
        key: "mid"
      - topic: "/robot0/sensor/camera1/compressed"
        key: "left"
      - topic: "/robot0/sensor/camera2/compressed"
        key: "right"
    eef_pose: "/robot0/vio/eef_pose"
    gripper: "/robot0/sensor/magnetic_encoder"
    imu: "/robot0/sensor/imu"
    tactile_left: "/robot0/sensor/tactile_left"
    tactile_right: "/robot0/sensor/tactile_right"
  fps: 15
  img_new_width: -1   # >0 to resize images proportionally by width
  text:
    desc: "pick and place"
    steps: ""
    goal: ""

Notes
- This exporter focuses on ee/dee-style observation/action downstream. Joint
  state topics are not required and are not synthesized.
- Images are saved as BGR JPEGs (cv2.imwrite). RerunEpisodeReader converts to
  RGB on load, matching existing HIROL data behavior.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import glog as log
import cv2
# Import image utils with a fallback to support running as a script
try:
    from .image_utils import (
        center_crop_and_resize_bgr,
        compute_center_crop_box,
    )
except Exception:
    _cur = os.path.dirname(os.path.abspath(__file__))
    if _cur not in sys.path:
        sys.path.insert(0, _cur)
    from image_utils import (
        center_crop_and_resize_bgr,
        compute_center_crop_box,
    )

# Make local packages importable when run as a script
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo layout:
#   REPO/dependencies/das-datakit
#   REPO/dependencies/lerobot/src/lerobot/datasets/hirol (this file)
# So das-datakit is five levels up from _CUR_DIR then into 'das-datakit'.
_DAS_DK_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../../../../../das-datakit"))
# Repo root is the parent of the 'dependencies' folder that contains das-datakit
_REPO_ROOT = os.path.abspath(os.path.join(_DAS_DK_ROOT, "..", ".."))
if _DAS_DK_ROOT not in sys.path:
    sys.path.insert(0, _DAS_DK_ROOT)

from utils.mcaploader import McapLoader  # noqa: E402
from utils.interpolate import get_inter_data  # noqa: E402


# ------------------------
# Dataclasses for config
# ------------------------

@dataclass
class TopicCamera:
    topic: str
    key: str


@dataclass
class TopicsCfg:
    ref: str
    cameras: List[TopicCamera]
    eef_pose: Optional[str] = None
    gripper: Optional[str] = None
    imu: Optional[str] = None
    tactile_left: Optional[str] = None
    tactile_right: Optional[str] = None


@dataclass
class InputCfg:
    mcap_file: str = ""
    task_dir: str = ""


@dataclass
class OutputCfg:
    task_dir: str  # root directory to place episode_XXXX folders


@dataclass
class ConverterCfg:
    input: InputCfg
    output: OutputCfg
    topics: TopicsCfg
    fps: int = 15
    img_new_width: int = -1
    preprocess: Optional[Dict[str, Any]] = None  # {target_size:[640,480], aspect_ratio:[4,3]}
    text: Optional[Dict[str, Any]] = None  # desc/steps/goal


# ------------------------
# Utilities
# ------------------------

def ns_to_s(ns: int | float) -> float:
    return float(ns) / 1e9


def resize_if_needed(img: np.ndarray, new_width: int) -> np.ndarray:
    if new_width is None or new_width <= 0:
        return img
    h, w = img.shape[:2]
    if w <= 0:
        return img
    new_h = max(1, int(h * float(new_width) / float(w)))
    return cv2.resize(img, (new_width, new_h))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_episode_name(idx: int) -> str:
    return f"episode_{idx:04d}"


def _json_float_or_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


# ------------------------
# Converter core
# ------------------------

class Das2HirolConverter:
    def __init__(self, cfg: ConverterCfg):
        self.cfg = cfg

    # -------- Input discovery --------
    def discover_inputs(self) -> List[str]:
        inputs: List[str] = []
        if self.cfg.input.mcap_file:
            mf = os.path.expanduser(self.cfg.input.mcap_file)
            if not os.path.isfile(mf):
                raise FileNotFoundError(f"mcap_file not found: {mf}")
            inputs.append(mf)
            return inputs
        if self.cfg.input.task_dir:
            td = os.path.expanduser(self.cfg.input.task_dir)
            if not os.path.isdir(td):
                raise FileNotFoundError(f"task_dir not exists: {td}")
            jpath = os.path.join(td, "vio_result.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"vio_result.json not found in task_dir: {td}")
            with open(jpath, "r", encoding="utf-8") as f:
                info = json.load(f)
            files = info.get("success_mcap_files", [])
            files = list(sorted(set(files)))
            if len(files) == 0:
                log.warning("No mcap files found in vio_result.json success_mcap_files")
            inputs.extend(files)
            return inputs
        raise ValueError("Either input.mcap_file or input.task_dir must be provided")

    # -------- Topic loading and interpolation --------
    def _load_topics(self, bag: McapLoader) -> None:
        topics = [self.cfg.topics.ref]
        topics.extend([c.topic for c in self.cfg.topics.cameras])
        if self.cfg.topics.eef_pose:
            topics.append(self.cfg.topics.eef_pose)
        if self.cfg.topics.gripper:
            topics.append(self.cfg.topics.gripper)
        if self.cfg.topics.imu:
            topics.append(self.cfg.topics.imu)
        if self.cfg.topics.tactile_left:
            topics.append(self.cfg.topics.tactile_left)
        if self.cfg.topics.tactile_right:
            topics.append(self.cfg.topics.tactile_right)
        topics = list(dict.fromkeys(topics))
        bag.load_topics(topics, auto_decompress=True, auto_sync=True)
        # Strengthen sync: ensure a time-based relation exists between ref and each camera
        for cam in self.cfg.topics.cameras:
            if cam.topic == self.cfg.topics.ref:
                continue
            try:
                bag.register_sync_relation_with_time(self.cfg.topics.ref, cam.topic, overwrite=True)
            except Exception as e:
                log.warning(f"register_sync_relation_with_time failed for {cam.topic}: {e}")

    def _validate_required_topics(self, bag: McapLoader) -> Tuple[bool, List[str]]:
        present = set(bag.get_all_topic_names())
        required: List[str] = []
        # Always required: ref + listed cameras + eef_pose
        required.append(self.cfg.topics.ref)
        required.extend([c.topic for c in self.cfg.topics.cameras])
        if self.cfg.topics.eef_pose:
            required.append(self.cfg.topics.eef_pose)
        # Conditional requireds
        # Prefer a dedicated 'require' section; fallback to legacy 'observation.include_*' if present
        req = {}
        # Try 'require'
        try:
            req = getattr(self.cfg, 'require') if hasattr(self.cfg, 'require') else {}
        except Exception:
            req = {}
        # Also read raw flags from possible 'observation' block in YAML if present in raw_cfg
        # but ConverterCfg doesn't carry it; we handle only explicit 'require' here.
        include_gripper = bool(req.get('include_gripper', False)) if isinstance(req, dict) else False
        include_imu = bool(req.get('include_imu', False)) if isinstance(req, dict) else False
        include_tactile = bool(req.get('include_tactile', False)) if isinstance(req, dict) else False

        if include_gripper and self.cfg.topics.gripper:
            required.append(self.cfg.topics.gripper)
        if include_imu and self.cfg.topics.imu:
            required.append(self.cfg.topics.imu)
        if include_tactile:
            if self.cfg.topics.tactile_left:
                required.append(self.cfg.topics.tactile_left)
            if self.cfg.topics.tactile_right:
                required.append(self.cfg.topics.tactile_right)
        missing: List[str] = []
        for t in required:
            if t not in present:
                missing.append(t)
                continue
            td = bag.get_topic_data(t)
            if not td:
                missing.append(t)
        return (len(missing) == 0), missing

    def _build_timebase(self, bag: McapLoader) -> Tuple[List[int], List[int], List[float]]:
        ref_topic = self.cfg.topics.ref
        ref_data = bag.get_topic_data(ref_topic)
        if ref_data is None or len(ref_data) == 0:
            raise RuntimeError(f"No data for ref topic: {ref_topic}")
        ref_seq = bag.get_topic_seq_num(ref_topic)
        if len(ref_seq) != len(ref_data):
            ref_seq = [d["data"].header.sequence_num for d in ref_data]
        ref_ts_ns = [d["data"].header.timestamp for d in ref_data]
        ref_ts_s = [ns_to_s(ns) for ns in ref_ts_ns]
        return ref_seq, ref_ts_ns, ref_ts_s

    def _interpolate_streams(
        self, bag: McapLoader, ref_ts_ns: List[int]
    ) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        tp = self.cfg.topics
        present_topics = set(bag.get_all_topic_names())
        # Pose
        if tp.eef_pose:
            if tp.eef_pose in present_topics:
                try:
                    arrays["eef_pose"] = get_inter_data(bag, tp.eef_pose, ref_ts_ns, "pose")
                except Exception as e:
                    log.warning(f"Failed to interpolate eef_pose ({tp.eef_pose}): {e}")
            else:
                log.warning(f"eef_pose topic missing in bag: {tp.eef_pose}")
        # Gripper
        if tp.gripper:
            if tp.gripper in present_topics:
                try:
                    g = get_inter_data(bag, tp.gripper, ref_ts_ns, "linear")
                    arrays["gripper"] = g.reshape((-1, 1)) if g.ndim == 1 else g
                except Exception as e:
                    log.warning(f"Failed to interpolate gripper ({tp.gripper}): {e}")
            else:
                log.warning(f"gripper topic missing in bag: {tp.gripper}")
        # IMU
        if tp.imu:
            if tp.imu in present_topics:
                try:
                    arrays["imu"] = get_inter_data(bag, tp.imu, ref_ts_ns, "linear")
                except Exception as e:
                    log.warning(f"Failed to interpolate imu ({tp.imu}): {e}")
            else:
                log.info(f"imu topic missing; skip: {tp.imu}")
        # Tactile
        if tp.tactile_left:
            if tp.tactile_left in present_topics:
                try:
                    arrays["t_l"] = get_inter_data(bag, tp.tactile_left, ref_ts_ns, "linear")
                except Exception as e:
                    log.warning(f"Failed to interpolate tactile_left ({tp.tactile_left}): {e}")
            else:
                log.info(f"tactile_left topic missing; skip: {tp.tactile_left}")
        if tp.tactile_right:
            if tp.tactile_right in present_topics:
                try:
                    arrays["t_r"] = get_inter_data(bag, tp.tactile_right, ref_ts_ns, "linear")
                except Exception as e:
                    log.warning(f"Failed to interpolate tactile_right ({tp.tactile_right}): {e}")
            else:
                log.info(f"tactile_right topic missing; skip: {tp.tactile_right}")
        return arrays

    # -------- Episode writing (HIROL JSON + images) --------
    def _keep_by_fps(self, ts_s: float, last_s: Optional[float], fps: float) -> bool:
        if last_s is None:
            return True
        dt = ts_s - last_s
        if dt < 0:
            return False
        target_dt = 1.0 / max(float(fps), 1e-6)
        return dt >= target_dt

    def _fetch_images_for_seq(self, bag: McapLoader, seq: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return per-camera dict: {key: {img: np.ndarray, ts_s: float}}.

        Uses the ref topic seq and sync graph to fetch each camera's synchronized
        frame and timestamp. Returns None if any camera fails.
        """
        cam_topics = [c.topic for c in self.cfg.topics.cameras]
        hit = bag.get_topic_data_by_seq_num(self.cfg.topics.ref, seq, sync_topics=cam_topics)
        if hit is None:
            return None
        out: Dict[str, Dict[str, Any]] = {}
        # Always try to include the ref camera image if available
        ref_topic = self.cfg.topics.ref
        ref_cam_entry = next((c for c in self.cfg.topics.cameras if c.topic == ref_topic), None)
        ref_data = hit.get(ref_topic) if isinstance(hit, dict) else None
        if ref_data is not None and ("decode_data" in ref_data) and (ref_data["decode_data"] is not None):
            ref_key = ref_cam_entry.key if ref_cam_entry else "ref"
            ts_s = ns_to_s(ref_data["data"].header.timestamp) if hasattr(ref_data["data"], "header") else None
            out[ref_key] = {"img": ref_data["decode_data"], "ts_s": ts_s}

        # Best-effort add other cameras; do not fail the whole frame if some are missing
        for cam in self.cfg.topics.cameras:
            if cam.topic == ref_topic:
                continue
            data = hit.get(cam.topic, None)
            if data is None or ("decode_data" not in data) or (data["decode_data"] is None):
                continue
            ts_s = ns_to_s(data["data"].header.timestamp) if hasattr(data["data"], "header") else None
            out[cam.key] = {"img": data["decode_data"], "ts_s": ts_s}

        return out if len(out) > 0 else None

    def _write_episode_json(
        self,
        out_dir: str,
        frames: List[Dict[str, Any]],
        image_hw: Tuple[int, int],
    ) -> None:
        info = {
            "version": "1.0.0",
            "date": "",
            "author": "HIROL",
            "image": {"width": int(image_hw[1]), "height": int(image_hw[0]), "fps": int(self.cfg.fps)},
            "depth": {"width": 0, "height": 0, "fps": 0},
            "audio": {"sample_rate": 0, "channels": 0, "format": "", "bits": 0},
            "joint_names": None,
            "tactile_names": None,
            "sim_state": "",
        }
        text = self.cfg.text or {"goal": "", "desc": "", "steps": ""}
        data = frames
        obj = {"info": info, "text": text, "data": data}
        jpath = os.path.join(out_dir, "data.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

    def write_episode(self, bag: McapLoader, out_dir: str) -> Tuple[int, int]:
        """Process one MCAP file and write an episode directory.

        Returns (frames_written, frames_skipped).
        """
        ensure_dir(out_dir)
        ensure_dir(os.path.join(out_dir, "colors"))
        ensure_dir(os.path.join(out_dir, "depths"))  # reserved, not used now
        ensure_dir(os.path.join(out_dir, "audios"))  # reserved, not used now
        ensure_dir(os.path.join(out_dir, "tactiles"))  # reserved

        self._load_topics(bag)
        ok, missing = self._validate_required_topics(bag)
        if not ok:
            log.warning(f"Skip episode {out_dir}: missing required topic(s): {missing}")
            return 0, 0
        ref_seq, ref_ts_ns, ref_ts_s = self._build_timebase(bag)
        arrays = self._interpolate_streams(bag, ref_ts_ns)

        frames_out: List[Dict[str, Any]] = []
        frames_written = 0
        frames_skipped = 0
        last_kept_ts: Optional[float] = None

        image_hw: Optional[Tuple[int, int]] = None

        # preprocess settings
        pp = self.cfg.preprocess or {}
        tgt_size = tuple(pp.get("target_size", [640, 480]))
        tgt_ar = tuple(pp.get("aspect_ratio", [4, 3]))

        # Record kept frame indices in the ref timebase so we can build actions
        # over the downsampled sequence (t -> t+K in kept frames), matching
        # das2lerobot_converter.md L43-L48.
        kept_indices: List[int] = []

        for i, seq in enumerate(ref_seq):
            ts_s = ref_ts_s[i]
            if not self._keep_by_fps(ts_s, last_kept_ts, self.cfg.fps):
                frames_skipped += 1
                continue

            cam_pack = self._fetch_images_for_seq(bag, seq)
            if cam_pack is None:
                frames_skipped += 1
                continue

            # Save images and build colors dict
            colors_entry: Dict[str, Dict[str, Any]] = {}
            saved_any = False
            for cam_key, item in cam_pack.items():
                img: np.ndarray = item["img"]
                ts_cam: Optional[float] = item["ts_s"]
                # Center-crop to target AR then resize to target size
                img = center_crop_and_resize_bgr(img, target_wh=tgt_size, aspect_ratio=tgt_ar)
                # Derive path and write
                rel_name = f"colors/{i:06d}_{cam_key}.jpg"
                abs_path = os.path.join(out_dir, rel_name)
                ok = cv2.imwrite(abs_path, img)
                if not ok:
                    log.warning(f"Failed to write image: {abs_path}")
                    continue
                colors_entry[cam_key] = {"path": rel_name, "time_stamp": _json_float_or_none(ts_cam)}
                saved_any = True
                if image_hw is None:
                    h, w = img.shape[:2]
                    image_hw = (h, w)

            if not saved_any:
                frames_skipped += 1
                continue

            # ee_states (7D pose per step)
            ee_entry: Optional[Dict[str, Any]] = None
            if "eef_pose" in arrays:
                pose = arrays["eef_pose"][i].astype(float).tolist()
                ee_entry = {
                    "single": {
                        "pose": pose,
                        "time_stamp": _json_float_or_none(ts_s),
                    }
                }

            # tools (gripper value)
            tools_entry: Optional[Dict[str, Any]] = None
            if "gripper" in arrays:
                gval = float(arrays["gripper"][i].reshape(-1)[0])
                tools_entry = {
                    "single": {
                        "position": gval,
                        "time_stamp": _json_float_or_none(ts_s),
                    }
                }

            # imu/tactile (optional; store raw vectors if present)
            imus_entry = None
            if "imu" in arrays:
                try:
                    imus_entry = {
                        "single": {
                            "val": arrays["imu"][i].astype(float).tolist(),
                            "time_stamp": _json_float_or_none(ts_s),
                        }
                    }
                except Exception:
                    imus_entry = None
            tactiles_entry = {}
            if "t_l" in arrays:
                tactiles_entry["left"] = arrays["t_l"][i].astype(float).tolist()
            if "t_r" in arrays:
                tactiles_entry["right"] = arrays["t_r"][i].astype(float).tolist()
            if len(tactiles_entry) == 0:
                tactiles_entry = {}

            step_obj: Dict[str, Any] = {
                "idx": int(i),
                "colors": colors_entry,
                "depths": None,
                "joint_states": {},  # Not provided by DAS; leave empty
                "ee_states": ee_entry or {},
                "tactiles": tactiles_entry,
                "imus": imus_entry,
                "audios": None,
                "tools": tools_entry or {},
                # "actions" not needed for state-based downstream; omit for clarity
            }
            frames_out.append(step_obj)
            last_kept_ts = ts_s
            frames_written += 1
            kept_indices.append(i)

        if frames_written == 0:
            log.warning("No frames written for episode; skipping JSON emit")
            return 0, frames_skipped

        # If configured or requested, synthesize state-based actions per frame.
        # Default behavior: action at step t is the observation at kept step t+1.
        try:
            raw = getattr(self.cfg, 'action', None)
        except Exception:
            raw = None
        # Only build actions when ee pose exists; otherwise skip silently.
        has_eef = ("eef_pose" in arrays) and (arrays["eef_pose"].shape[0] > 0)
        if has_eef and frames_written > 0:
            act_type = "ee"
            act_step = 1
            act_ori = "quaternion"
            include_gripper = False
            if isinstance(raw, dict):
                act_type = str(raw.get("type", act_type)).lower()
                act_step = int(raw.get("prediction_step", act_step))
                act_ori = str(raw.get("orientation", act_ori)).lower()
                include_gripper = bool(raw.get("include_gripper", False))

            # Helper: quaternion ops to compute relative pose if needed.
            def _quat_mul(q1: List[float] | Tuple[float, ...], q2: List[float] | Tuple[float, ...]):
                x1, y1, z1, w1 = q1
                x2, y2, z2, w2 = q2
                # Hamilton product (x,y,z,w)
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                return [x, y, z, w]

            def _quat_conj(q: List[float] | Tuple[float, ...]):
                x, y, z, w = q
                return [-x, -y, -z, w]

            def _quat_apply(q: List[float] | Tuple[float, ...], v: List[float] | Tuple[float, ...]):
                # Rotate vector v by unit quaternion q using cross-product form
                x, y, z, w = q
                qv = np.array([x, y, z], dtype=np.float64)
                v = np.array(v, dtype=np.float64)
                t = 2.0 * np.cross(qv, v)
                return (v + w * t + np.cross(qv, t)).tolist()

            def _pose_delta(next_pose: List[float], cur_pose: List[float]) -> List[float]:
                # next - cur in the current frame (position expressed in cur frame)
                p1 = np.array(next_pose[:3], dtype=np.float64)
                p2 = np.array(cur_pose[:3], dtype=np.float64)
                q1 = [float(next_pose[3]), float(next_pose[4]), float(next_pose[5]), float(next_pose[6])]
                q2 = [float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5]), float(cur_pose[6])]
                dq = _quat_mul(_quat_conj(q2), q1)
                dp_world = (p1 - p2).tolist()
                dp_local = _quat_apply(_quat_conj(q2), dp_world)
                return [dp_local[0], dp_local[1], dp_local[2], dq[0], dq[1], dq[2], dq[3]]

            # Build actions for each kept frame, using the next kept frame by default
            for j in range(frames_written):
                # Resolve current/next kept indices into the original ref index space
                cur_ref_idx = kept_indices[j]
                next_j = j + max(1, act_step)
                if next_j >= frames_written:
                    next_ref_idx = kept_indices[-1]  # duplicate last valid action
                else:
                    next_ref_idx = kept_indices[next_j]

                # Populate actions per ee key present in ee_states for this frame
                step_actions: Dict[str, Any] = {}
                ee_states = frames_out[j].get("ee_states", {}) or {}
                ts_next = _json_float_or_none(ref_ts_s[next_ref_idx])
                # allow both single-EE (key 'single') and multi-EE schemas
                for ee_key, state in ee_states.items():
                    # Current and future pose from arrays to avoid JSON float rounding
                    cur_pose = arrays["eef_pose"][cur_ref_idx].astype(float).tolist()
                    next_pose = arrays["eef_pose"][next_ref_idx].astype(float).tolist()
                    if act_type == "dee":
                        act_pose = _pose_delta(next_pose, cur_pose)
                    else:  # "ee" absolute by default
                        act_pose = next_pose
                    # Orientation output: currently only quaternion supported in converter
                    if act_ori != "quaternion":
                        log.warning(
                            f"action.orientation '{act_ori}' not supported in converter; using quaternion"
                        )
                    action_entry: Dict[str, Any] = {
                        "ee": {"pose": act_pose, "time_stamp": ts_next}
                    }
                    if include_gripper and ("gripper" in arrays):
                        g_next = float(np.atleast_1d(arrays["gripper"][next_ref_idx].reshape(-1))[0])
                        action_entry["tool"] = {"position": g_next, "time_stamp": ts_next}
                    step_actions[ee_key] = action_entry
                # If no ee_states, but we still want to write tool-only action
                if (not step_actions) and include_gripper and ("gripper" in arrays):
                    g_next = float(np.atleast_1d(arrays["gripper"][next_ref_idx].reshape(-1))[0])
                    step_actions["single"] = {"tool": {"position": g_next, "time_stamp": ts_next}}
                # Attach actions to the frame
                frames_out[j]["actions"] = step_actions

        # Fallback hw if no image saved (should not happen if frames_written>0)
        if image_hw is None:
            image_hw = (0, 0)
        self._write_episode_json(out_dir, frames_out, image_hw)
        return frames_written, frames_skipped

    # -------- Batch conversion --------
    def convert_all(self) -> str:
        inputs = self.discover_inputs()
        if len(inputs) == 0:
            raise RuntimeError("No MCAP files to convert")

        out_root = os.path.expanduser(self.cfg.output.task_dir)
        ensure_dir(out_root)
        log.info(f"HIROL episodes will be written under: {out_root}")

        for idx, mcap_path in enumerate(inputs):
            ep_name = infer_episode_name(idx)
            ep_dir = os.path.join(out_root, ep_name)
            log.info(f"Processing {idx+1}/{len(inputs)}: {mcap_path} -> {ep_dir}")
            try:
                bag = McapLoader(mcap_path)
                w, s = self.write_episode(bag, ep_dir)
                bag.close()
                log.info(f"Episode {ep_name}: frames_written={w}, frames_skipped={s}")
                if w == 0:
                    # Clean up empty directory to avoid confusing downstream
                    try:
                        for root, dirs, files in os.walk(ep_dir, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                        os.rmdir(ep_dir)
                    except Exception:
                        pass
            except Exception as e:
                log.error(f"Failed to convert {mcap_path}: {e}")
        return out_root


# ------------------------
# Config loader & CLI
# ------------------------

def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_cfg(raw: Dict[str, Any]) -> ConverterCfg:
    # Normalize topics.cameras to dataclasses
    topic_cams = []
    for cam in (raw.get("topics", {}).get("cameras", []) or []):
        topic_cams.append(TopicCamera(topic=str(cam["topic"]), key=str(cam["key"])))
    topics = TopicsCfg(
        ref=str(raw["topics"]["ref"]),
        cameras=topic_cams,
        eef_pose=raw["topics"].get("eef_pose"),
        gripper=raw["topics"].get("gripper"),
        imu=raw["topics"].get("imu"),
        tactile_left=raw["topics"].get("tactile_left"),
        tactile_right=raw["topics"].get("tactile_right"),
    )
    input_cfg = InputCfg(
        mcap_file=str(raw.get("input", {}).get("mcap_file", "")),
        task_dir=str(raw.get("input", {}).get("task_dir", "")),
    )
    # Default output to repo-level dataset/data/source
    default_out_dir = os.path.join(_REPO_ROOT, "dataset", "data", "source")
    out_dir_raw = raw.get("output", {}).get("task_dir", default_out_dir)
    # If user provided a relative path, treat it as relative to repo root
    if not os.path.isabs(out_dir_raw):
        out_dir = os.path.normpath(os.path.join(_REPO_ROOT, out_dir_raw))
    else:
        out_dir = out_dir_raw
    out_cfg = OutputCfg(task_dir=str(out_dir))
    cfg = ConverterCfg(
        input=input_cfg,
        output=out_cfg,
        topics=topics,
        fps=int(raw.get("fps", 15)),
        img_new_width=int(raw.get("img_new_width", -1)),
        preprocess=raw.get("preprocess", None),
        text=raw.get("text", None),
    )
    # Attach optional 'require' dict to cfg for conditional topic requirement
    require = raw.get('require', None)
    if isinstance(require, dict):
        setattr(cfg, 'require', require)
    # Attach optional 'action' configuration so the converter can build actions
    action_cfg = raw.get('action', None)
    if isinstance(action_cfg, dict):
        setattr(cfg, 'action', action_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Convert DAS MCAP to HIROL JSON+images episodes")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    raw_cfg = _load_yaml_cfg(os.path.expanduser(args.config))
    cfg = _parse_cfg(raw_cfg)
    log.info(f"Using cfg: fps={cfg.fps} out={cfg.output.task_dir}")

    conv = Das2HirolConverter(cfg)
    out_root = conv.convert_all()
    log.info(f"Done. Episodes under: {out_root}")


if __name__ == "__main__":
    main()
