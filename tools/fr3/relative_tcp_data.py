# ruff: noqa: N806  -- T_hist: a stack of transforms
"""Training data for the relative-TCP policy: which exports may be used, their cropped views, the samples.

Three steps, each its own function so the gateway, the trainer and the tests can call them separately:

1. :func:`check_exports` decides whether a set of exports may train one policy. The rules are the ones
   the export already enforces, read back from each info.json, never re-derived: only exports written
   with the QC gate on (``qc_gate.enforced``); one TCP point, the one the arm's FK reaches; label chains
   that agree (:func:`label_provenance.assert_compatible_datasets`, the same check the old trainer ran);
   the chosen hand tracked in every export.
2. :func:`build_view` writes a training view of an export: the parquet is linked, not copied, and only
   the policy's cameras are kept, transcoded once to the crop and size in the contract. Decoding nine
   1080p streams per sample (about a second) is what training would otherwise spend its time on.
   The export itself is never touched.
3. :class:`RelativeTcpDataset` turns a view's frames into what the policy sees (tools/fr3/relative_tcp.py),
   and :func:`compute_stats` gives the normaliser the statistics of those relative quantities, not of the
   absolute poses the dataset stores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tools.fr3 import relative_tcp as rt
from tools.thor.gmsl2 import label_provenance as lp, pose_columns as pc

VIEW_SCHEMA = "fr3_training_view/1"
VIEW_FILE = "training_view.json"
WIDTH_SUFFIX = "box_gripper.distance_m"
WRENCH_PREFIX = "box_six_d_external."
#: ImageNet statistics: the ACT / diffusion backbones are ImageNet-pretrained ResNets.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainingDataError(ValueError):
    """An export cannot be used for this policy; the message says why and what to do."""


# ----------------------------------------------------------------- checking ---


def _read_info(root: Path) -> dict[str, Any]:
    path = Path(root) / "meta" / "info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingDataError(f"{root}: no readable meta/info.json ({exc}); is this an export?") from exc


def _box_columns(state_names: Sequence[str]) -> dict[str, dict[str, int]]:
    """BOX prefix ('' for a single BOX) -> {'width': index, 'fx': index, ...}."""
    out: dict[str, dict[str, int]] = {}
    for i, name in enumerate(state_names):
        if name.endswith(WIDTH_SUFFIX):
            out.setdefault(name[: -len(WIDTH_SUFFIX)], {})["width"] = i
        elif WRENCH_PREFIX in name:
            prefix, _, axis = name.partition(WRENCH_PREFIX)
            if axis in rt.WRENCH_AXES:
                out.setdefault(prefix, {})[axis] = i
    return out


@dataclass
class ExportSummary:
    root: str
    name: str
    episodes: int
    frames: int
    fps: int
    pose_key: str
    cameras: dict[str, list[int]]  # camera name -> [width, height]
    boxes: list[str]  # BOX prefixes with a width column
    has_wrench: dict[str, bool]
    world_frame_id: str
    labels_fingerprint: str
    qc_gate: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportCheck:
    ok: bool
    hand: str
    exports: list[ExportSummary]
    problems: list[str]
    warnings: list[str]
    common_cameras: dict[str, list[int]]
    boxes: list[str]

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise TrainingDataError("; ".join(self.problems))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "hand": self.hand,
            "problems": self.problems,
            "warnings": self.warnings,
            "commonCameras": self.common_cameras,
            "boxes": self.boxes,
            "exports": [vars(e) for e in self.exports],
        }


def summarize_export(root: Path, hand: str) -> tuple[ExportSummary | None, list[str], dict[str, Any]]:
    """One export: its summary (None when unusable) and what is wrong with it."""
    root = Path(root)
    problems: list[str] = []
    info = _read_info(root)
    features = info.get("features") or {}
    name = root.name
    gate = info.get("qc_gate")
    if not isinstance(gate, dict):
        problems.append(
            f"{name} was exported before the QC gate (11-16): nothing says QC passed its episodes; run QC on "
            "its recordings and export again"
        )
    elif gate.get("enforced") is not True:
        problems.append(f"{name} was exported with --ignore-qc; training takes only QC-gated exports")
    link = str((info.get("tcp_frame") or {}).get("link") or "")
    if link != rt.TCP_LINK:
        problems.append(
            f"{name}: its TCP poses are of {link or 'more than one point'!r}, not {rt.TCP_LINK!r} (the point "
            "the arm's FK reaches); export again from sessions tracked with one marker->TCP calibration"
        )
    key = pc.pose_key(features, pc.EE_STATE, hand)
    if key is None:
        tracked = pc.tracked_hands(features)
        problems.append(f"{name} has no tracked {hand} hand (tracked: {', '.join(tracked) or 'none'})")
    state_names = list(((features.get("observation.state") or {}).get("names")) or [])
    boxes = _box_columns(state_names)
    with_width = sorted(prefix for prefix, cols in boxes.items() if "width" in cols)
    if not with_width:
        problems.append(f"{name}: observation.state has no gripper width ({WIDTH_SUFFIX})")
    cameras = {
        k.removeprefix("observation.images."): [int(v["shape"][1]), int(v["shape"][0])]
        for k, v in features.items()
        if k.startswith("observation.images.") and v.get("dtype") in ("video", "image")
    }
    if problems:
        return None, problems, info
    labels = info.get("label_provenance") or {}
    summary = ExportSummary(
        root=str(root.resolve()),
        name=name,
        episodes=int(info.get("total_episodes") or 0),
        frames=int(info.get("total_frames") or 0),
        fps=int(info.get("fps") or 0),
        pose_key=str(key),
        cameras=cameras,
        boxes=with_width,
        has_wrench={p: all(a in boxes[p] for a in rt.WRENCH_AXES) for p in with_width},
        world_frame_id=str((info.get("world_frame") or {}).get("world_frame_id") or ""),
        labels_fingerprint=str(labels.get("fingerprint") or ""),
        qc_gate=gate,
        warnings=[f"{name}: {w}" for w in lp.hand_warnings(labels)],
    )
    if summary.episodes == 0:
        return None, [f"{name} has no episodes (QC let none through?)"], info
    return summary, [], info


def check_exports(roots: Iterable[Path], hand: str) -> ExportCheck:
    roots = [Path(r) for r in roots]
    problems: list[str] = []
    warnings: list[str] = []
    summaries: list[ExportSummary] = []
    infos: list[tuple[str, dict[str, Any]]] = []
    if not roots:
        problems.append("pick at least one export")
    for root in roots:
        try:
            summary, bad, info = summarize_export(root, hand)
        except TrainingDataError as exc:
            problems.append(str(exc))
            continue
        problems += bad
        if summary is not None:
            summaries.append(summary)
            warnings += summary.warnings
            infos.append((summary.name, info))
    if len({s.fps for s in summaries}) > 1:
        problems.append("exports with different frame rates: " + ", ".join(f"{s.name} {s.fps} fps" for s in summaries))
    if len(infos) > 1:
        try:
            lp.assert_compatible_datasets(infos)
        except lp.MixedLabelsError as exc:
            problems.append(f"these exports' label chains differ and cannot train one policy: {exc}")
    common: dict[str, list[int]] = {}
    if summaries:
        names = set.intersection(*(set(s.cameras) for s in summaries))
        for cam in sorted(names):
            sizes = {tuple(s.cameras[cam]) for s in summaries}
            if len(sizes) == 1:
                common[cam] = list(sizes.pop())
            else:
                problems.append(f"{cam} has different resolutions across the exports: {sorted(sizes)}")
    boxes = sorted(set.intersection(*(set(s.boxes) for s in summaries))) if summaries else []
    if summaries and not boxes:
        problems.append("the exports share no BOX with a gripper width")
    return ExportCheck(not problems, hand, summaries, problems, warnings, common, boxes)


# --------------------------------------------------------------------- crops ---


def default_output_size(crop: Sequence[int], long_side: int = 320) -> tuple[int, int]:
    """(width, height) keeping the crop's aspect, long side ``long_side``, both even (H.264)."""
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    scale = long_side / max(w, h)
    return max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)


def camera_input(name: str, source_size: Sequence[int], crop: Sequence[int] | None, long_side: int = 320) -> rt.CameraInput:
    sw, sh = int(source_size[0]), int(source_size[1])
    box = [0, 0, sw, sh] if not crop else [int(v) for v in crop]
    w, h = default_output_size(box, long_side)
    cam = rt.CameraInput(name=name, source_width=sw, source_height=sh, crop=box, width=w, height=h)
    cam.validate()
    return cam


# --------------------------------------------------------------------- views ---


def _export_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in ("meta/info.json", "meta/episodes", "data"):
        path = root / rel
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            st = f.stat()
            digest.update(f"{f.relative_to(root)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    for video in sorted((root / "videos").rglob("*.mp4")):
        st = video.stat()
        digest.update(f"{video.relative_to(root)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]


def view_name(export_root: Path, cameras: Sequence[rt.CameraInput]) -> str:
    spec = json.dumps([vars(c) for c in cameras], sort_keys=True)
    tag = hashlib.sha256(f"{_export_fingerprint(export_root)}|{spec}".encode()).hexdigest()[:10]
    return f"{Path(export_root).name}__view_{tag}"


def _count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
    )
    return int(out.stdout.strip().splitlines()[0])


def transcode(src: Path, dst: Path, camera: rt.CameraInput, fps: int) -> None:
    """Crop and scale one video file, frame for frame. All-intra so random access costs one frame."""
    x0, y0, x1, y1 = camera.crop
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={camera.width}:{camera.height}:flags=area,format=yuv420p"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-g", "1", "-bf", "0",
        "-video_track_timescale", str(int(fps) * 1000), str(dst),
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True)
    n_src, n_dst = _count_frames(src), _count_frames(dst)
    if n_src != n_dst:
        raise TrainingDataError(f"transcoding {src} gave {n_dst} frames, the source has {n_src}")


def build_view(export_root: Path, cameras: Sequence[rt.CameraInput], views_root: Path, log=print) -> Path:
    """The training view of ``export_root`` for ``cameras`` (made once, then reused)."""
    export_root = Path(export_root).resolve()
    views_root = Path(views_root)
    name = view_name(export_root, cameras)
    final = views_root / name
    if (final / VIEW_FILE).is_file():
        log(f"Training view {name}: already built")
        return final
    info = _read_info(export_root)
    fps = int(info["fps"])
    tmp = views_root / f".{name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "meta").mkdir(parents=True)
    for item in (export_root / "meta").iterdir():
        if item.name == "info.json":
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, tmp / "meta" / item.name)
    os.symlink(export_root / "data", tmp / "data")
    features = dict(info["features"])
    wanted = {c.key: c for c in cameras}
    for key in [k for k, v in features.items() if v.get("dtype") in ("video", "image")]:
        if key not in wanted:
            del features[key]
    for key, cam in wanted.items():
        if key not in features:
            raise TrainingDataError(f"{export_root.name} has no {cam.name}")
        feat = dict(features[key])
        feat["shape"] = [cam.height, cam.width, 3]
        feat["info"] = {**(feat.get("info") or {}), "video.height": cam.height, "video.width": cam.width,
                        "video.codec": "h264", "video.pix_fmt": "yuv420p"}
        features[key] = feat
        sources = sorted((export_root / "videos" / key).rglob("*.mp4"))
        if not sources:
            raise TrainingDataError(f"{export_root.name}: no video files for {cam.name}")
        for src in sources:
            log(f"Training view {name}: {cam.name} {src.relative_to(export_root)} -> crop {cam.crop}, {cam.width}x{cam.height}")
            transcode(src, tmp / src.relative_to(export_root), cam, fps)
    view_info = {**info, "features": features}
    view_info["training_view"] = {
        "schema": VIEW_SCHEMA,
        "source": str(export_root),
        "source_fingerprint": _export_fingerprint(export_root),
        "cameras": [vars(c) for c in cameras],
    }
    (tmp / "meta" / "info.json").write_text(json.dumps(view_info, indent=2), encoding="utf-8")
    (tmp / VIEW_FILE).write_text(
        json.dumps({**view_info["training_view"], "created": datetime.now(UTC).isoformat()}, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return final


# ------------------------------------------------------------------- samples ---


@dataclass
class EpisodeArrays:
    """One episode's columns, as the sample builder and the stats need them."""

    poses: np.ndarray  # (n, 7) TCP pose per frame, NaN where tracking failed
    widths: np.ndarray  # (n,)
    wrench: np.ndarray | None  # (n, 6)


def _state_indices(info: dict[str, Any], box: str, with_wrench: bool) -> tuple[int, list[int]]:
    names = list(info["features"]["observation.state"]["names"])
    cols = _box_columns(names).get(box) or {}
    if "width" not in cols:
        raise TrainingDataError(f"BOX {box or '(single)'} has no {WIDTH_SUFFIX}")
    wrench = []
    if with_wrench:
        missing = [a for a in rt.WRENCH_AXES if a not in cols]
        if missing:
            raise TrainingDataError(f"BOX {box or '(single)'} has no external wrench ({', '.join(missing)})")
        wrench = [cols[a] for a in rt.WRENCH_AXES]
    return cols["width"], wrench


def contract_columns(contract: rt.PolicyContract, info: dict[str, Any]) -> dict[str, Any]:
    names = list(info["features"]["observation.state"]["names"])
    box = contract.width_name[: -len(WIDTH_SUFFIX)]
    width_i, wrench_i = _state_indices(info, box, bool(contract.wrench_names))
    if names[width_i] != contract.width_name or [names[i] for i in wrench_i] != list(contract.wrench_names):
        raise TrainingDataError("the dataset's state columns are not the ones the contract names")
    key = pc.pose_key(info["features"], pc.EE_STATE, contract.hand)
    if key is None:
        raise TrainingDataError(f"no tracked {contract.hand} hand")
    return {"pose_key": key, "width": width_i, "wrench": wrench_i}


def load_episode_arrays(root: Path, contract: rt.PolicyContract) -> list[EpisodeArrays]:
    import pyarrow.parquet as pq

    info = _read_info(root)
    cols = contract_columns(contract, info)
    table = pq.read_table(Path(root) / "data", columns=["episode_index", "frame_index", cols["pose_key"], "observation.state"])
    ep = np.asarray(table.column("episode_index").to_pylist())
    fr = np.asarray(table.column("frame_index").to_pylist())
    poses = np.asarray(table.column(cols["pose_key"]).to_pylist(), dtype=np.float64)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    out = []
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(fr[idx])]
        out.append(
            EpisodeArrays(
                poses=poses[idx],
                widths=state[idx, cols["width"]],
                wrench=state[idx][:, cols["wrench"]] if cols["wrench"] else None,
            )
        )
    return out


def _clamped(n: int, t: int, offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray([t + o for o in offsets])
    return np.clip(raw, 0, n - 1), (raw < 0) | (raw > n - 1)


def sample_vectors(
    contract: rt.PolicyContract, arrays: EpisodeArrays, t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(state, actions (horizon, 10), action_is_pad) at frame ``t``, or None when ``t`` cannot be used.

    Offsets past either end of the episode repeat the end frame (as LeRobot pads): history before the
    first frame reads as "not moving", which is what an arm waiting at the start does too; targets after
    the last frame are padded and left out of the loss. A frame whose own pose, width or history was not
    tracked is unusable; a target that was not tracked is padded.
    """
    n = len(arrays.poses)
    hist_i, _ = _clamped(n, t, contract.history_offsets)
    hist = arrays.poses[hist_i]
    if not np.all(np.isfinite(hist)) or not math.isfinite(arrays.widths[t]):
        return None
    wrench = None
    if contract.wrench_names:
        wrench = arrays.wrench[t]
        if not np.all(np.isfinite(wrench)):
            return None
    T_hist = rt.pose7_to_matrix(hist)
    state = rt.state_vector(contract, T_hist, float(arrays.widths[t]), wrench)
    act_i, pad = _clamped(n, t, contract.action_offsets)
    targets = arrays.poses[act_i]
    widths = arrays.widths[act_i]
    bad = ~np.all(np.isfinite(targets), axis=1) | ~np.isfinite(widths)
    pad = pad | bad
    targets = np.where(bad[:, None], hist[-1], targets)
    widths = np.where(bad, arrays.widths[t], widths)
    actions = rt.action_chunk(T_hist[-1], rt.pose7_to_matrix(targets), widths)
    return state, actions, pad


def _vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(0).tolist(),
        "std": np.maximum(values.std(0), 1e-6).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def compute_stats(view_roots: Sequence[Path], contract: rt.PolicyContract) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normaliser statistics of the relative quantities, and a summary of what was usable."""
    states, actions = [], []
    usable = total = padded = 0
    for root in view_roots:
        for arrays in load_episode_arrays(root, contract):
            for t in range(len(arrays.poses)):
                total += 1
                sample = sample_vectors(contract, arrays, t)
                if sample is None:
                    continue
                usable += 1
                state, act, pad = sample
                states.append(state)
                actions.append(act[~pad])
                padded += int(pad.sum())
    if not states:
        raise TrainingDataError("no usable frame: the TCP was never tracked with a width")
    stats: dict[str, Any] = {
        "observation.state": _vector_stats(np.stack(states)),
        "action": _vector_stats(np.concatenate(actions)),
    }
    for cam in contract.cameras:
        stats[cam.key] = {
            "mean": [[[m]] for m in IMAGENET_MEAN],
            "std": [[[s]] for s in IMAGENET_STD],
            "min": [[[0.0]]] * 3,
            "max": [[[1.0]]] * 3,
        }
    summary = {"frames": total, "usable_frames": usable, "padded_targets": padded}
    return stats, summary


class RelativeTcpDataset:
    """Training views -> LeRobot-style samples in the relative-TCP contract (a torch Dataset).

    ``meta`` stands in for LeRobotDatasetMetadata where lerobot_train reads it: ``features`` and
    ``stats`` describe the converted samples, not the stored columns.
    """

    def __init__(
        self,
        view_roots: Sequence[Path],
        contract: rt.PolicyContract,
        stats: dict[str, Any],
        *,
        video_backend: str | None = None,
        image_transforms=None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.contract = contract
        self.view_roots = [Path(r) for r in view_roots]
        self.datasets = []
        self.arrays: list[list[EpisodeArrays]] = []
        self.index: list[tuple[int, int, int, int]] = []  # (dataset, global row, episode, frame)
        from_index, to_index = [], []
        for d, root in enumerate(self.view_roots):
            info = _read_info(root)
            if int(info["fps"]) != contract.fps:
                raise TrainingDataError(f"{root.name} is {info['fps']} fps, the contract says {contract.fps}")
            # Only the camera frames come from LeRobot: one decoded frame per camera per sample. The
            # poses and widths are read once, here, and converted per sample by sample_vectors.
            ds = LeRobotDataset(
                f"local/{root.name}", root=root, video_backend=video_backend, image_transforms=image_transforms
            )
            self.datasets.append(ds)
            episodes = load_episode_arrays(root, contract)
            self.arrays.append(episodes)
            starts = np.asarray(ds.meta.episodes["dataset_from_index"])
            for e, arrays in enumerate(episodes):
                begin = len(self.index)
                for t in range(len(arrays.poses)):
                    if sample_vectors(contract, arrays, t) is not None:
                        self.index.append((d, int(starts[e]) + t, e, t))
                if len(self.index) > begin:
                    from_index.append(begin)
                    to_index.append(len(self.index))
        if not self.index:
            raise TrainingDataError("no usable frame in the training views")
        self.meta = _Meta(contract, stats, fps=contract.fps, from_index=from_index, to_index=to_index)
        self.episodes = None
        self.num_frames = len(self.index)
        self.num_episodes = len(from_index)
        self.root = self.view_roots[0]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        d, row, e, t = self.index[int(i)]
        state, actions, pad = sample_vectors(self.contract, self.arrays[d][e], t)
        item = self.datasets[d][row]
        out = {
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(actions),
            "action_is_pad": torch.from_numpy(pad),
            "index": torch.tensor(i),
            "episode_index": torch.tensor(e),
            "frame_index": torch.tensor(t),
            "task": item.get("task", ""),
        }
        for cam in self.contract.cameras:
            out[cam.key] = item[cam.key]
        return out


class _Meta:
    def __init__(self, contract: rt.PolicyContract, stats: dict[str, Any], *, fps: int, from_index, to_index):
        self.fps = fps
        self.robot_type = "fr3_relative_tcp"
        self.stats = {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for k, d in stats.items()}
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [len(contract.state_names)], "names": contract.state_names},
            "action": {"dtype": "float32", "shape": [len(contract.action_names)], "names": contract.action_names},
        }
        for cam in contract.cameras:
            self.features[cam.key] = {
                "dtype": "video",
                "shape": [cam.height, cam.width, 3],
                "names": ["height", "width", "channels"],
            }
        self.camera_keys = [cam.key for cam in contract.cameras]
        self.video_keys = list(self.camera_keys)
        self.image_keys: list[str] = []
        self.episodes = {"dataset_from_index": from_index, "dataset_to_index": to_index}
        self.total_episodes = len(from_index)
        self.total_frames = to_index[-1] if to_index else 0
