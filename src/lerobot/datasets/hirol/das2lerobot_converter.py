"""
Convert DAS-Datakit MCAP episodes to LeRobotDataset ("lerobot format").

This script reads MCAP files using the local das-datakit utils, aligns multi-sensor
streams to a reference camera timeline, downsamples by FPS, and writes episodes
with images, observation.state and action to a LeRobotDataset.

Key references:
- LeRobot writer API: lerobot/datasets/lerobot_dataset.py
- HIROL loader structure: lerobot/datasets/hirol/lerobot_loader.py
- DAS MCAP I/O: dependencies/das-datakit/utils/mcaploader.py, interpolate.py

Usage:
  python das2lerobot_converter.py -c path/to/config.yaml

Config schema (YAML): see das2lerobot_converter.md for detailed description.
"""

from __future__ import annotations

import os
import sys
import math
import json
import copy
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import glog as log
import cv2
from scipy.spatial.transform import Rotation as R

# Ensure local imports for lerobot and das-datakit work when running this script directly.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# Workspace root (repo root) four levels up: hirol -> datasets -> lerobot -> src -> workspace
_WS_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../../../.."))
_LEROBOT_SRC = os.path.abspath(os.path.join(_CUR_DIR, "../../.."))
# Optional old layout support (kept for compatibility)
_DAS_DK_ROOT = os.path.abspath(os.path.join(_WS_ROOT, "dependencies", "das-datakit"))
# Add common roots to sys.path so 'lerobot' and 'das' can be imported when running as a script
for _p in (_LEROBOT_SRC, _WS_ROOT, _DAS_DK_ROOT, "/das"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
# Prefer package-qualified imports (das.utils) so that adding REPO_ROOT to sys.path is enough.
from das.utils.mcaploader import McapLoader  # noqa: E402
from das.utils.interpolate import get_inter_data  # noqa: E402
# Try relative import first; fall back to local file import when run as a script.
try:  # noqa: E402
    from .image_utils import center_crop_and_resize_rgb  # type: ignore
except Exception:  # noqa: E402
    if _CUR_DIR not in sys.path:
        sys.path.insert(0, _CUR_DIR)
    from image_utils import center_crop_and_resize_rgb  # type: ignore  # noqa: E402


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
class ObservationCfg:
    type: str  # q/dq/ee/dee/q_ee/mask/ft (converter focuses on ee/ft)
    orientation: str  # "quaternion" or "euler"
    include_gripper: bool = False
    include_imu: bool = False
    include_tactile: bool = False
    contain_ft: bool = False
    fields: Optional[List[str]] = None  # explicit order, e.g. ["ee", "gripper"]


@dataclass
class ActionCfg:
    type: str  # "ee" or "dee"
    prediction_step: int
    orientation: str  # "quaternion" or "euler"
    include_gripper: bool = False  # optional: include gripper in action vector


@dataclass
class WriterCfg:
    image_writer_threads: int = 1
    image_writer_processes: int = 1
    batch_encoding_size: int = 1
    video_backend: str = "torchcodec"


@dataclass
class IOPaths:
    mcap_file: str = ""
    task_dir: str = ""


@dataclass
class OutputCfg:
    root_path: str
    repo_name: str
    robot_name: str = "fr3"
    # When provided, this path is used as the dataset root directory.
    # Accepts absolute or relative (relative to this script dir).
    task_dir: Optional[str] = None


@dataclass
class ConverterCfg:
    input: IOPaths
    output: OutputCfg
    topics: TopicsCfg
    observation: ObservationCfg
    action: ActionCfg
    fps: int = 15
    img_new_width: int = -1
    # preprocessing: center-crop to target aspect then resize to target size
    preprocess: Optional[Dict[str, Any]] = None  # {target_size: [640,480], aspect_ratio: [4,3]}
    custom_prompt: Optional[str] = None
    writer: WriterCfg = WriterCfg()
    rotation_transform: Optional[Dict[str, List[float]]] = None


# ------------------------
# Utilities
# ------------------------

def ns_to_s(ns: int | float) -> float:
    return float(ns) / 1e9


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx,qy,qz,qw] to euler [roll,pitch,yaw]."""
    return R.from_quat(q).as_euler("xyz", False)


def pose_diff_quat(pose1: np.ndarray, pose2: np.ndarray, translate_in_frame2: bool = True) -> np.ndarray:
    """Compute pose1 - pose2 for 7D quaternion pose.

    - position diff optionally expressed in frame2 (by applying rot2^T).
    - orientation diff as relative quaternion: q = q2^{-1} * q1.
    Returns 7D [dx,dy,dz, qx,qy,qz,qw].
    """
    out = np.zeros(7, dtype=np.float32)
    p1 = np.asarray(pose1[:3], dtype=np.float32)
    p2 = np.asarray(pose2[:3], dtype=np.float32)
    r1 = R.from_quat(np.asarray(pose1[3:], dtype=np.float32))
    r2 = R.from_quat(np.asarray(pose2[3:], dtype=np.float32))
    r_rel = r2.inv() * r1
    dp = p1 - p2
    if translate_in_frame2:
        dp = r2.inv().apply(dp)
    out[:3] = dp
    out[3:] = r_rel.as_quat()
    return out


def resize_if_needed(img: np.ndarray, new_width: int) -> np.ndarray:
    if new_width is None or new_width <= 0:
        return img
    h, w = img.shape[:2]
    if w == 0:
        return img
    new_h = max(1, int(h * float(new_width) / float(w)))
    return cv2.resize(img, (new_width, new_h))


def ensure_hwc_rgb(img_bgr: np.ndarray, new_width: int) -> np.ndarray:
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = resize_if_needed(img, new_width)
    return img


def infer_task_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


# ------------------------
# Converter core
# ------------------------

class Das2LerobotConverter:
    def __init__(self, cfg: ConverterCfg):
        self.cfg = cfg

    # -------- Input discovery --------
    def discover_inputs(self) -> List[str]:
        mcap_file = self.cfg.input.mcap_file or ""
        task_dir = self.cfg.input.task_dir or ""
        inputs: List[str] = []
        if mcap_file:
            if not os.path.isfile(mcap_file):
                raise FileNotFoundError(f"mcap_file not found: {mcap_file}")
            inputs.append(mcap_file)
            return inputs
        if task_dir:
            if not os.path.isdir(task_dir):
                raise FileNotFoundError(f"task_dir not exists: {task_dir}")
            jpath = os.path.join(task_dir, "vio_result.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"vio_result.json not found in task_dir: {task_dir}")
            with open(jpath, "r", encoding="utf-8") as f:
                info = json.load(f)
            files = info.get("success_mcap_files", [])
            files = list(sorted(set(files)))
            if len(files) == 0:
                log.warning("No mcap files found in vio_result.json success_mcap_files")
            inputs.extend(files)
            return inputs
        raise ValueError("Either input.mcap_file or input.task_dir must be provided")

    # -------- MCAP loading & timebase --------
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
        # Deduplicate
        topics = list(dict.fromkeys(topics))
        # Build sync graph and decode images
        bag.load_topics(topics, auto_decompress=True, auto_sync=True)

        # Some bags may miss explicit sync relations (header.inputs) for cameras.
        # Fall back to time-nearest registration if we detect missing links.
        try:
            ref_topic = self.cfg.topics.ref
            ref_seqs = bag.get_topic_seq_num(ref_topic)
            # Probe only a few early seq ids to decide if a camera lacks mapping
            probe_seqs = ref_seqs[: min(3, len(ref_seqs))]
            for cam in self.cfg.topics.cameras:
                needs_register = False
                for s in probe_seqs:
                    rels = bag.sync_graph.get_relations(ref_topic, s)
                    if rels.get(cam.topic, None) is None:
                        needs_register = True
                        break
                if needs_register:
                    try:
                        # Register time-based sync once; do not overwrite existing mappings.
                        bag.register_sync_relation_with_time(ref_topic, cam.topic, overwrite=False)
                    except Exception as e:
                        # Keep going; writer will skip frames that fail later.
                        log.warning(f"Time-based sync registration failed for {cam.topic}: {e}")
        except Exception as e:
            # Don't make topic loading brittle due to sync probing
            log.debug(f"Sync probing failed (non-fatal): {e}")

    def _validate_required_topics(self, bag: McapLoader) -> Tuple[bool, List[str]]:
        """Check all required topics exist and have messages.

        Required: ref, all cameras, eef_pose; plus conditional gripper/imu/tactile by config.
        """
        present = set(bag.get_all_topic_names())
        required: List[str] = []
        required.append(self.cfg.topics.ref)
        required.extend([c.topic for c in self.cfg.topics.cameras])
        if self.cfg.topics.eef_pose:
            required.append(self.cfg.topics.eef_pose)
        # conditional
        if self.cfg.observation.include_gripper or self.cfg.action.include_gripper:
            if self.cfg.topics.gripper:
                required.append(self.cfg.topics.gripper)
        if self.cfg.observation.include_imu and self.cfg.topics.imu:
            required.append(self.cfg.topics.imu)
        if self.cfg.observation.include_tactile:
            if self.cfg.topics.tactile_left:
                required.append(self.cfg.topics.tactile_left)
            if self.cfg.topics.tactile_right:
                required.append(self.cfg.topics.tactile_right)
        missing = []
        for t in required:
            if t not in present:
                missing.append(t)
                continue
            # also ensure non-empty
            td = bag.get_topic_data(t)
            if not td:
                missing.append(t)
        return (len(missing) == 0), missing

    def build_timebase_and_data(self, bag: McapLoader) -> Tuple[List[int], List[float], Dict[str, np.ndarray]]:
        """Return (ref_seq_nums, ref_timestamps_s, arrays_by_name)

        arrays_by_name keys (if available):
          - eef_pose: (N,7)
          - gripper: (N,1)
          - imu: (N,6)
          - tactile_left: (N,D)
          - tactile_right: (N,D)
        """
        self._load_topics(bag)

        ref_topic = self.cfg.topics.ref
        ref_data = bag.get_topic_data(ref_topic)
        if ref_data is None or len(ref_data) == 0:
            raise RuntimeError(f"No data for ref topic: {ref_topic}")

        ref_seq = bag.get_topic_seq_num(ref_topic)
        # Align to list order
        if len(ref_seq) != len(ref_data):
            # Fallback: derive from ref_data headers
            ref_seq = [d["data"].header.sequence_num for d in ref_data]

        ref_ts_s = [ns_to_s(d["data"].header.timestamp) for d in ref_data]

        arrays: Dict[str, np.ndarray] = {}
        topics = self.cfg.topics

        # Interpolate to reference timestamps
        if topics.eef_pose:
            arrays["eef_pose"] = get_inter_data(bag, topics.eef_pose, [d["data"].header.timestamp for d in ref_data], "pose")
        if topics.gripper:
            g = get_inter_data(bag, topics.gripper, [d["data"].header.timestamp for d in ref_data], "linear")
            arrays["gripper"] = g.reshape((-1, 1)) if g.ndim == 1 else g
        if topics.imu and self.cfg.observation.include_imu:
            arrays["imu"] = get_inter_data(bag, topics.imu, [d["data"].header.timestamp for d in ref_data], "linear")
        if topics.tactile_left and self.cfg.observation.include_tactile:
            arrays["t_l"] = get_inter_data(bag, topics.tactile_left, [d["data"].header.timestamp for d in ref_data], "linear")
        if topics.tactile_right and self.cfg.observation.include_tactile:
            arrays["t_r"] = get_inter_data(bag, topics.tactile_right, [d["data"].header.timestamp for d in ref_data], "linear")

        return ref_seq, ref_ts_s, arrays

    # -------- Features inference --------
    def infer_features_from_first(self, bag: McapLoader, ref_seq: List[int], arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Infer LeRobot features and return (features_dict, dims_dict).

        dims_dict contains:
          - obs_dim
          - act_dim
        """
        # Fetch a frame with all cameras synced to infer image shapes. The very first
        # ref frame may lack sync relations for some cameras, so search forward a bit
        # and, if needed, register a time-based sync as a fallback.
        cam_topics = [c.topic for c in self.cfg.topics.cameras]
        hit = None
        hit_seq = None
        # Try a short window of ref seqs to find a complete sync hit
        search_window = min(50, len(ref_seq))
        for k in range(search_window):
            seq_try = ref_seq[k]
            hit_try = bag.get_topic_data_by_seq_num(self.cfg.topics.ref, seq_try, sync_topics=cam_topics)
            if hit_try is None:
                continue
            missing = [ct for ct in cam_topics if (hit_try.get(ct) is None or ("decode_data" not in hit_try.get(ct, {})))]
            if not missing:
                hit = hit_try
                hit_seq = seq_try
                break
        if hit is None:
            # Attempt to register time-based sync and retry once from the first seq
            ref_topic = self.cfg.topics.ref
            for ct in cam_topics:
                try:
                    bag.register_sync_relation_with_time(ref_topic, ct, overwrite=False)
                except Exception:
                    pass
            seq0 = ref_seq[0]
            hit = bag.get_topic_data_by_seq_num(ref_topic, seq0, sync_topics=cam_topics)
            hit_seq = seq0 if hit is not None else None
        if hit is None:
            raise RuntimeError("Failed to fetch a synced frame for camera shapes after fallback")

        features: Dict[str, Any] = {}
        image_dims: List[Tuple[int, int, int]] = []
        # preprocessing cfg
        pp = self.cfg.preprocess or {}
        tgt_size = tuple(pp.get("target_size", [640, 480]))
        tgt_ar = tuple(pp.get("aspect_ratio", [4, 3]))

        for cam in self.cfg.topics.cameras:
            data = hit.get(cam.topic, None)
            if data is None or "decode_data" not in data:
                raise RuntimeError(f"Failed to get synced image for camera {cam.topic}")
            # BGR->RGB and center-crop+resize to configured target
            img_rgb = center_crop_and_resize_rgb(data["decode_data"], target_wh=tgt_size, aspect_ratio=tgt_ar)
            h, w = img_rgb.shape[:2]
            c = img_rgb.shape[2] if img_rgb.ndim == 3 else 1
            features[f"observation.images.{cam.key}"] = dict(
                dtype="video",
                shape=(h, w, c),
                names=["height", "width", "channels"],
            )
            image_dims.append((h, w, c))

        # Observation state dim
        obs_vec0 = self.compose_observation_state_at(0, arrays)
        obs_dim = int(obs_vec0.shape[0])
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (obs_dim,),
            "names": [f"s{i}" for i in range(obs_dim)],
        }

        # Action dim
        act_vec0 = self.compose_action_at(0, arrays, allow_tail_short=False)
        act_dim = int(act_vec0.shape[0]) if act_vec0 is not None else 0
        features["action"] = {
            "dtype": "float32",
            "shape": (act_dim,),
            "names": [f"a{i}" for i in range(act_dim)],
        }

        dims = {"obs_dim": obs_dim, "act_dim": act_dim}
        return features, dims

    # -------- Vector composition --------
    def _ee_vec(self, ee_pose_row: np.ndarray) -> np.ndarray:
        if ee_pose_row is None:
            return np.zeros((0,), dtype=np.float32)
        if self.cfg.observation.orientation == "euler":
            out = np.zeros((6,), dtype=np.float32)
            out[:3] = ee_pose_row[:3]
            out[3:] = quat_to_euler_xyz(ee_pose_row[3:])
            return out
        else:
            return ee_pose_row.astype(np.float32)

    def _ee_vec_for_action(self, ee_pose_row: np.ndarray) -> np.ndarray:
        if self.cfg.action.orientation == "euler":
            out = np.zeros((6,), dtype=np.float32)
            out[:3] = ee_pose_row[:3]
            out[3:] = quat_to_euler_xyz(ee_pose_row[3:])
            return out
        else:
            return ee_pose_row.astype(np.float32)

    def compose_observation_state_at(self, idx: int, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        fields = self.cfg.observation.fields or ["ee"]
        vecs: List[np.ndarray] = []
        for f in fields:
            if f == "ee":
                ee = arrays.get("eef_pose", None)
                if ee is None:
                    raise RuntimeError("Observation field 'ee' requested but eef_pose is missing")
                vecs.append(self._ee_vec(ee[idx]))
            elif f == "gripper":
                g = arrays.get("gripper", None)
                if g is None:
                    raise RuntimeError("Observation field 'gripper' requested but gripper is missing")
                vecs.append(np.asarray(g[idx]).reshape(-1).astype(np.float32))
            elif f == "imu":
                imu = arrays.get("imu", None)
                if imu is None:
                    raise RuntimeError("Observation field 'imu' requested but imu is missing (or include_imu=false)")
                vecs.append(np.asarray(imu[idx]).reshape(-1).astype(np.float32))
            elif f == "tactile_left":
                tl = arrays.get("t_l", None)
                if tl is None:
                    raise RuntimeError("Observation field 'tactile_left' requested but tactile_left is missing (or include_tactile=false)")
                vecs.append(np.asarray(tl[idx]).reshape(-1).astype(np.float32))
            elif f == "tactile_right":
                tr = arrays.get("t_r", None)
                if tr is None:
                    raise RuntimeError("Observation field 'tactile_right' requested but tactile_right is missing (or include_tactile=false)")
                vecs.append(np.asarray(tr[idx]).reshape(-1).astype(np.float32))
            else:
                raise ValueError(f"Unknown observation field: {f}")
        if len(vecs) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(vecs, axis=0).astype(np.float32, copy=False)

    def compose_action_at(self, idx: int, arrays: Dict[str, np.ndarray], allow_tail_short: bool = True) -> Optional[np.ndarray]:
        act_type = (self.cfg.action.type or "").lower()
        K = max(int(self.cfg.action.prediction_step), 0)
        ee = arrays.get("eef_pose", None)
        g = arrays.get("gripper", None)

        # State-based actions need t+K valid index
        if act_type in ("ee", "dee"):
            if ee is None:
                raise RuntimeError("Action requires eef_pose but it's missing")
            t_next = idx + K
            if t_next >= ee.shape[0]:
                if allow_tail_short:
                    return None
                else:
                    # Clamp to last available to infer dimension during features inference.
                    t_next = ee.shape[0] - 1

            if act_type == "ee":
                vec = self._ee_vec_for_action(ee[t_next])
            elif act_type == "dee":
                diff = pose_diff_quat(ee[t_next], ee[idx])
                if self.cfg.action.orientation == "euler":
                    v = np.zeros((6,), dtype=np.float32)
                    v[:3] = diff[:3]
                    v[3:] = quat_to_euler_xyz(diff[3:])
                    vec = v
                else:
                    vec = diff
            else:
                raise ValueError(f"Unsupported action.type: {act_type}")

            # Optional: append gripper action
            if bool(self.cfg.action.include_gripper) and g is not None:
                if act_type == "ee":
                    gv = np.asarray(g[t_next]).reshape(-1).astype(np.float32)
                else:  # dee
                    gv = (np.asarray(g[t_next]) - np.asarray(g[idx])).reshape(-1).astype(np.float32)
                vec = np.concatenate([vec, gv], axis=0)
            return vec

        raise ValueError(f"Unsupported or missing action.type: {act_type}")

    # -------- Episode writing --------
    def _create_writer(self, features: Dict[str, Any]) -> LeRobotDataset:
        repo = self.cfg.output.repo_name
        if not self.cfg.output.task_dir:
            raise ValueError("output.task_dir must be set; fallback to root_path/repo_name is disabled")
        td = os.path.expanduser(self.cfg.output.task_dir)
        dataset_dir = td if os.path.isabs(td) else os.path.join(_CUR_DIR, td)
        if os.path.exists(dataset_dir):
            # 明确拒绝覆盖已有目录，避免隐式回退或污染
            raise FileExistsError(
                f"output.task_dir already exists: {dataset_dir}. Please remove it or choose an empty directory."
            )
        log.info(f"save_path: {dataset_dir}")

        # 先创建数据集但不启动图像写入器（避免在 create() 内部失败导致二次创建目录冲突）
        ds = LeRobotDataset.create(
            root=dataset_dir,
            repo_id=repo,
            robot_type=self.cfg.output.robot_name,
            fps=int(self.cfg.fps),
            features=features,
            image_writer_threads=0,
            image_writer_processes=0,
            batch_encoding_size=int(self.cfg.writer.batch_encoding_size),
            video_backend=self.cfg.writer.video_backend,
        )

        # 按配置尝试启动写入器；若多进程受限则退回到线程（不改变路径）
        iw_threads = max(int(self.cfg.writer.image_writer_threads), 1)
        iw_procs = int(self.cfg.writer.image_writer_processes)
        try:
            if iw_procs or iw_threads:
                ds.start_image_writer(num_processes=iw_procs, num_threads=iw_threads)
        except PermissionError as e:
            log.warning(f"Async image writer with processes={iw_procs} failed: {e}. Falling back to threads-only.")
            # 尝试仅线程写入
            ds.stop_image_writer()
            ds.start_image_writer(num_processes=0, num_threads=iw_threads or 4)

        # 每个 episode 后立刻落盘 meta，避免悬空 writer
        ds.meta.metadata_buffer_size = 1
        return ds

    def _keep_by_fps(self, ts_s: float, last_s: Optional[float], fps: float) -> bool:
        if last_s is None:
            return True
        dt = ts_s - last_s
        if dt < 0:
            # Non-monotonic timestamps; keep it conservatively
            return False
        target_dt = 1.0 / max(float(fps), 1e-6)
        return dt >= target_dt

    def _fetch_images_for_seq(self, bag: McapLoader, seq: int) -> Optional[Dict[str, np.ndarray]]:
        cam_topics = [c.topic for c in self.cfg.topics.cameras]
        hit = bag.get_topic_data_by_seq_num(self.cfg.topics.ref, seq, sync_topics=cam_topics)
        if hit is None:
            return None
        out: Dict[str, np.ndarray] = {}
        pp = self.cfg.preprocess or {}
        tgt_size = tuple(pp.get("target_size", [640, 480]))
        tgt_ar = tuple(pp.get("aspect_ratio", [4, 3]))
        for cam in self.cfg.topics.cameras:
            data = hit.get(cam.topic, None)
            if data is None or "decode_data" not in data:
                return None
            img_rgb = center_crop_and_resize_rgb(data["decode_data"], target_wh=tgt_size, aspect_ratio=tgt_ar)
            out[cam.key] = img_rgb
        return out

    def write_episode(
        self,
        writer: LeRobotDataset,
        bag: McapLoader,
        ref_seq: List[int],
        ref_ts_s: List[float],
        arrays: Dict[str, np.ndarray],
        mcap_path: str,
    ) -> Tuple[int, int]:
        frames_written = 0
        frames_skipped = 0

        # Build kept index list first, then use next kept frame for action (t+K on downsampled timeline).
        kept_indices: List[int] = []
        last_kept_ts: Optional[float] = None
        for i, ts_s in enumerate(ref_ts_s):
            if self._keep_by_fps(ts_s, last_kept_ts, self.cfg.fps):
                kept_indices.append(i)
                last_kept_ts = ts_s

        # Nothing to write
        if len(kept_indices) == 0:
            return 0, len(ref_seq)

        # Iterate kept frames and compose obs/action aligned to kept t+K
        def _compose_action_between(idx_cur: int, idx_next: int) -> Optional[np.ndarray]:
            act_type = (self.cfg.action.type or "").lower()
            ee = arrays.get("eef_pose", None)
            g = arrays.get("gripper", None)
            if act_type in ("ee", "dee"):
                if ee is None:
                    raise RuntimeError("Action requires eef_pose but it's missing")
                if idx_next >= ee.shape[0]:
                    return None
                if act_type == "ee":
                    vec = self._ee_vec_for_action(ee[idx_next])
                else:  # dee
                    diff = pose_diff_quat(ee[idx_next], ee[idx_cur])
                    if self.cfg.action.orientation == "euler":
                        v = np.zeros((6,), dtype=np.float32)
                        v[:3] = diff[:3]
                        v[3:] = quat_to_euler_xyz(diff[3:])
                        vec = v
                    else:
                        vec = diff
                if bool(self.cfg.action.include_gripper) and g is not None:
                    if act_type == "ee":
                        gv = np.asarray(g[idx_next]).reshape(-1).astype(np.float32)
                    else:
                        gv = (np.asarray(g[idx_next]) - np.asarray(g[idx_cur])).reshape(-1).astype(np.float32)
                    vec = np.concatenate([vec, gv], axis=0)
                return vec
            raise ValueError(f"Unsupported or missing action.type: {act_type}")

        # Determine stride on kept timeline (at least 1)
        K = max(int(self.cfg.action.prediction_step), 1)
        for j, i in enumerate(kept_indices):
            # Next kept index: clamp to last kept frame to keep length consistent
            next_j = j + K
            if next_j >= len(kept_indices):
                next_i = kept_indices[-1]
            else:
                next_i = kept_indices[next_j]

            # Fetch images for current kept seq
            imgs = self._fetch_images_for_seq(bag, ref_seq[i])
            if imgs is None or len(imgs) != len(self.cfg.topics.cameras):
                frames_skipped += 1
                continue

            obs_vec = self.compose_observation_state_at(i, arrays)
            act_vec = _compose_action_between(i, next_i)
            if act_vec is None:
                frames_skipped += 1
                continue

            frame: Dict[str, Any] = {}
            for k, img in imgs.items():
                frame[f"observation.images.{k}"] = img
            frame["observation.state"] = obs_vec.astype(np.float32, copy=False)
            frame["action"] = act_vec.astype(np.float32, copy=False)
            frame["task"] = self.cfg.custom_prompt or infer_task_name_from_path(mcap_path)

            try:
                writer.add_frame(frame)
            except Exception as e:
                log.warning(f"add_frame failed at kept_idx={j} (src idx={i}): {e}")
                frames_skipped += 1
                continue

            frames_written += 1

        if frames_written > 0:
            writer.save_episode()
        else:
            # Clean up any temp artifacts if no valid frames
            if getattr(writer, "episode_buffer", None) is not None:
                writer.clear_episode_buffer(delete_images=len(writer.meta.image_keys) > 0)

        return frames_written, frames_skipped

    # -------- Main entry --------
    def convert_all(self) -> LeRobotDataset:
        inputs = self.discover_inputs()
        if len(inputs) == 0:
            raise RuntimeError("No MCAP files to convert")

        # Prepare first episode to infer features
        first_bag = McapLoader(inputs[0])
        # Load topics and validate before any interpolation to avoid crashes
        self._load_topics(first_bag)
        ok, missing = self._validate_required_topics(first_bag)
        if not ok:
            log.warning(f"Skip first episode {inputs[0]}: missing required topic(s): {missing}")
            first_bag.close()
            # find next usable episode for inferring features
            writer = None
            for j in range(1, len(inputs)):
                mp = inputs[j]
                bag = McapLoader(mp)
                try:
                    self._load_topics(bag)
                    ok2, miss2 = self._validate_required_topics(bag)
                    if not ok2:
                        log.warning(f"Skip episode {mp}: missing required topic(s): {miss2}")
                        bag.close()
                        continue
                    ref_seq, ref_ts_s, arrays = self.build_timebase_and_data(bag)
                    features, dims = self.infer_features_from_first(bag, ref_seq, arrays)
                    writer = self._create_writer(features)
                    log.info(f"Features: obs_dim={dims['obs_dim']}, act_dim={dims['act_dim']}")
                    w, s = self.write_episode(writer, bag, ref_seq, ref_ts_s, arrays, mp)
                    log.info(f"Episode from {mp}: frames_written={w}, frames_skipped={s}")
                    bag.close()
                    start_idx = j + 1
                    break
                except Exception as e:
                    log.warning(f"Failed to initialize from {mp}: {e}")
                    bag.close()
                    continue
            if writer is None:
                raise RuntimeError("No valid episodes to initialize dataset features.")
            # Remaining episodes
            for idx in range(start_idx, len(inputs)):
                mp = inputs[idx]
                log.info(f"Processing {idx+1}/{len(inputs)}: {mp}")
                bag = McapLoader(mp)
                ok3, miss3 = self._validate_required_topics(bag)
                if not ok3:
                    log.warning(f"Skip episode {mp}: missing required topic(s): {miss3}")
                    bag.close()
                    continue
                ref_seq, ref_ts_s, arrays = self.build_timebase_and_data(bag)
                w, s = self.write_episode(writer, bag, ref_seq, ref_ts_s, arrays, mp)
                log.info(f"Episode from {mp}: frames_written={w}, frames_skipped={s}")
                bag.close()
            writer.finalize()
            return writer
        # Valid first episode; now do interpolation and infer features
        ref_seq, ref_ts_s, arrays = self.build_timebase_and_data(first_bag)
        features, dims = self.infer_features_from_first(first_bag, ref_seq, arrays)

        writer = self._create_writer(features)
        log.info(f"Features: obs_dim={dims['obs_dim']}, act_dim={dims['act_dim']}")

        # First episode
        log.info(f"Processing 1/{len(inputs)}: {inputs[0]}")
        # Already validated above
        w, s = self.write_episode(writer, first_bag, ref_seq, ref_ts_s, arrays, inputs[0])
        log.info(f"Episode from {inputs[0]}: frames_written={w}, frames_skipped={s}")
        first_bag.close()

        # Remaining episodes
        for idx in range(1, len(inputs)):
            mp = inputs[idx]
            log.info(f"Processing {idx+1}/{len(inputs)}: {mp}")
            bag = McapLoader(mp)
            self._load_topics(bag)
            ok2, miss2 = self._validate_required_topics(bag)
            if not ok2:
                log.warning(f"Skip episode {mp}: missing required topic(s): {miss2}")
                bag.close()
                continue
            ref_seq, ref_ts_s, arrays = self.build_timebase_and_data(bag)
            w, s = self.write_episode(writer, bag, ref_seq, ref_ts_s, arrays, mp)
            log.info(f"Episode from {mp}: frames_written={w}, frames_skipped={s}")
            bag.close()

        writer.finalize()
        return writer


# ------------------------
# Config loader & CLI
# ------------------------

def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_cfg(raw: Dict[str, Any]) -> ConverterCfg:
    # Topics.cameras normalization
    topic_cams = []
    for cam in (raw.get("topics", {}).get("cameras", []) or []):
        topic_cams.append(TopicCamera(topic=cam["topic"], key=cam["key"]))
    topics = TopicsCfg(
        ref=raw["topics"]["ref"],
        cameras=topic_cams,
        eef_pose=raw["topics"].get("eef_pose"),
        gripper=raw["topics"].get("gripper"),
        imu=raw["topics"].get("imu"),
        tactile_left=raw["topics"].get("tactile_left"),
        tactile_right=raw["topics"].get("tactile_right"),
    )

    obs_raw = raw.get("observation", {})
    observation = ObservationCfg(
        type=obs_raw.get("type", "ee"),
        orientation=obs_raw.get("orientation", "quaternion"),
        include_gripper=bool(obs_raw.get("include_gripper", False)),
        include_imu=bool(obs_raw.get("include_imu", False)),
        include_tactile=bool(obs_raw.get("include_tactile", False)),
        contain_ft=bool(obs_raw.get("contain_ft", False)),
        fields=obs_raw.get("fields", ["ee"]),
    )

    act_raw = raw.get("action", {})
    action = ActionCfg(
        type=act_raw.get("type", "dee"),
        prediction_step=int(act_raw.get("prediction_step", 1)),
        orientation=act_raw.get("orientation", "quaternion"),
        include_gripper=bool(act_raw.get("include_gripper", False)),
    )

    writer = WriterCfg(
        image_writer_threads=int(raw.get("writer", {}).get("image_writer_threads", 1)),
        image_writer_processes=int(raw.get("writer", {}).get("image_writer_processes", 1)),
        batch_encoding_size=int(raw.get("writer", {}).get("batch_encoding_size", 1)),
        video_backend=raw.get("writer", {}).get("video_backend", "torchcodec"),
    )

    io_paths = IOPaths(
        mcap_file=raw.get("input", {}).get("mcap_file", ""),
        task_dir=raw.get("input", {}).get("task_dir", ""),
    )

    output = OutputCfg(
        root_path=raw.get("output", {}).get("root_path", "../assets/data"),
        repo_name=raw.get("output", {}).get("repo_name", "das_dataset"),
        robot_name=raw.get("output", {}).get("robot_name", "fr3"),
        task_dir=raw.get("output", {}).get("task_dir"),
    )

    cfg = ConverterCfg(
        input=io_paths,
        output=output,
        topics=topics,
        observation=observation,
        action=action,
        fps=int(raw.get("fps", 15)),
        img_new_width=int(raw.get("img_new_width", -1)),
        preprocess=raw.get("preprocess", None),
        custom_prompt=raw.get("custom_prompt", None),
        writer=writer,
        rotation_transform=raw.get("rotation_transform", None),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Convert DAS MCAP to LeRobotDataset")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    raw_cfg = _load_yaml_cfg(os.path.expanduser(args.config))
    cfg = _parse_cfg(raw_cfg)
    log.info(f"Using cfg repo={cfg.output.repo_name} fps={cfg.fps}")

    converter = Das2LerobotConverter(cfg)
    ds = converter.convert_all()
    log.info(f"Finished conversion. Dataset saved at: {ds.meta.root}")

    # Optional: verify by opening in reader mode
    try:
        if not cfg.output.task_dir:
            raise ValueError("output.task_dir must be set for post-check")
        td = os.path.expanduser(cfg.output.task_dir)
        read_root = td if os.path.isabs(td) else os.path.join(_CUR_DIR, td)
        reader_ds = LeRobotDataset(repo_id=cfg.output.repo_name, root=read_root)
        log.info(f"Loaded dataset back. len={len(reader_ds)} features={getattr(reader_ds, 'features', None)}")
    except Exception as e:
        log.warning(f"Post-check failed to open dataset: {e}")


if __name__ == "__main__":
    main()
