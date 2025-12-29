import json
import random
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:  # pragma: no cover - import hints for type checkers
    LeRobotDataset = Any  # type: ignore


def _episode_name_to_index(name: str, base_index: int = 0) -> int:
    """
    Parse episode name like 'episode_0001' to an integer index.

    Args:
        name: String containing digits (e.g. 'episode_0061').
        base_index: If your episodes are 1-indexed in pair_info, set to 1 to map
            'episode_0001' -> 0. If already 0-indexed, keep 0.
    """
    digits = "".join([c for c in name if c.isdigit()])
    if len(digits) == 0:
        raise ValueError(f"Cannot parse episode index from name: {name}")
    return int(digits) - int(base_index)


@dataclass
class _EpisodeSpan:
    start: int
    end: int  # exclusive

    def __len__(self) -> int:
        return max(0, self.end - self.start)


def _build_episode_spans(ds: LeRobotDataset) -> List[_EpisodeSpan]:
    """
    Create spans [start, end) for each episode for a given LeRobotDataset.

    This helper is careful about the case where `ds` was instantiated with a
    subset of episodes (via the `episodes=` argument). In that situation the
    underlying HF dataset is re-indexed to only contain the requested episodes,
    while `ds.meta.episodes[*]['dataset_from_index']` / `['dataset_to_index']`
    still refer to the original global index space.

    We therefore:
      - Prefer `ds._local_episode_to_range` when available to build spans in the
        local index space of `ds.hf_dataset`, ensuring indices are always
        valid for `ds[idx]`.
      - Fall back to the global metadata ranges for full datasets.
    """
    spans: List[_EpisodeSpan] = []

    # Prefer local episode → range mapping when present (episode-subset case).
    local_map = getattr(ds, "_local_episode_to_range", None)
    if isinstance(local_map, dict) and len(local_map) > 0:
        meta_eps = ds.meta.episodes
        # Build a list indexed by global episode index. Episodes that are not
        # present in `local_map` get a zero-length span and will be skipped
        # when constructing OT pairs.
        for ep_idx, _ in enumerate(meta_eps):
            if ep_idx in local_map:
                start, end = local_map[ep_idx]
                spans.append(_EpisodeSpan(start=int(start), end=int(end)))
            else:
                spans.append(_EpisodeSpan(start=0, end=0))
        return spans

    # Full dataset: use global metadata ranges directly.
    meta_eps = ds.meta.episodes
    for ep in meta_eps:
        spans.append(
            _EpisodeSpan(start=int(ep["dataset_from_index"]), end=int(ep["dataset_to_index"]))
        )
    return spans


def _stack_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a sequence of flat dict samples into a batch, recursing on dict values."""
    if len(items) == 0:
        return {}
    out: Dict[str, Any] = {}
    keys = set().union(*[d.keys() for d in items])
    for k in keys:
        vals = [d[k] for d in items if k in d]
        if isinstance(vals[0], dict):
            out[k] = _stack_dicts(vals)  # type: ignore[arg-type]
        else:
            out[k] = torch.stack([v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in vals], dim=0)
    return out


class LeRobotOTPairDataset(Dataset):
    """
    A lightweight OT pair dataset that reads synchronized frames from two LeRobotDataset
    instances (source and target) according to a DTW-style pair_info JSON.

    The pair_info JSON format is assumed to be similar to:
    {
      "episode_0001": [
        {"demo_name": "episode_0061", "raw_dtw_dist": 5.7, "pairing": {"0": [0, 2, 4], "2": [6], ...}},
        ... more src candidates ...
      ],
      ... more target episodes ...
    }

    Notes:
    - For efficiency, we precompute a list of target indices and for each one the list of
      candidate source indices. __getitem__ randomly picks a source index from the candidates.
    - By default, only 'observation.state' and 'action' are returned. You can extend obs_keys
      to include visual keys such as 'observation.images.left'.
    - If pair_info uses 1-based episode names (episode_0001), set base_index_src/tgt=1.
    - window_size 是给 OT 配对取样加一点时间抖动用的，单位是“帧”。语义与 ot-sim2real 对齐：在从 pair_info 取到一对 (tgt_global, src_global) 后，固定 target 帧，仅对 source 帧在 [-window_size, +window_size] 内做局部随机偏移（并裁剪到合法范围），增强对轻微时序误差的鲁棒性（数据增强）。  
      怎么选  
      - 0：严格按 pair_info 对齐，不做时间抖动。pair_info 很准（同 fps、无跳帧）时，首选 0。
      - 1–2：pair_info 稍有偏差，或 target 的映射是隔帧（如 0,2,4,…）时，给 1 比较稳，2 也可尝试。经验上：
          - 30 fps：1–2 帧 ≈ 33–67ms/帧 的抖动，通常安全
          - 15 fps：1 帧 ≈ 67ms，先用 1
      - ≥3：只在配对时间噪声明显或者想强力增强时使用，注意过大容易引入错配，建议配合更小的 λ_ot 或 batch_ratio。  
      经验起点  
      - 如果你的 pair_info 是按相同 fps、对齐较好：window_size=0
      - 如果像你给的样例 target 索引是 0,2,4… 这种隔帧映射：window_size=1
      - 如果转换/配对存在轻微采样率或对齐误差：window_size=1（必要时再试 2）
    """

    def __init__(
        self,
        ds_src: LeRobotDataset,
        ds_tgt: LeRobotDataset,
        pair_info_path: str | Path,
        obs_keys: Sequence[str] | None = None,
        action_key: str = "action",
        base_index_src: int = 0,
        base_index_tgt: int = 0,
        window_size: int | None = None,
        # Optional sampling controls
        sharpness: float = 0.0,
        no_window: bool = False,
        topk_src_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.ds_src = ds_src
        self.ds_tgt = ds_tgt
        self.obs_keys = list(obs_keys) if obs_keys is not None else ["observation.state"]
        self.action_key = action_key
        self.window = int(window_size) if window_size and window_size > 0 else None
        # Sampling knobs (default preserve legacy behavior)
        self.sharpness = float(sharpness) if sharpness is not None else 0.0
        self.no_window = bool(no_window)
        self.topk_src_episodes = int(topk_src_episodes) if topk_src_episodes not in (None, -1) else None

        self.src_spans = _build_episode_spans(self.ds_src)
        self.tgt_spans = _build_episode_spans(self.ds_tgt)

        with open(pair_info_path, "r") as f:
            pair_info = json.load(f)

        # Build name->episode index mapping using numeric suffix convention.
        # If your dataset ordering differs, adjust base_index_* accordingly.
        def name2idx_src(name: str) -> int:
            return _episode_name_to_index(name, base_index=base_index_src)

        def name2idx_tgt(name: str) -> int:
            return _episode_name_to_index(name, base_index=base_index_tgt)

        # Build sample descriptors: each contains (tgt_global_idx, [src_global_idx ...], [weight ...])
        samples: List[Tuple[int, List[int], List[float]]] = []
        # Keep a reference to the target dataset's local episode map (if any) so
        # that we can ignore pairs for episodes that are not present in `ds_tgt`.
        tgt_local_map: dict[int, tuple[int, int]] | None = getattr(
            self.ds_tgt, "_local_episode_to_range", None
        )

        for tgt_name, src_list in pair_info.items():
            tgt_ep_idx = name2idx_tgt(str(tgt_name))
            if tgt_ep_idx < 0 or tgt_ep_idx >= len(self.tgt_spans):
                # Skip unknown target episode
                continue
            # If the target dataset was instantiated on a subset of episodes,
            # make sure this episode is actually present in the subset.
            if tgt_local_map is not None and tgt_ep_idx not in tgt_local_map:
                continue
            tgt_span = self.tgt_spans[tgt_ep_idx]

            # Optionally keep only top-K closest src episodes for this target episode
            if self.topk_src_episodes is not None and self.topk_src_episodes > 0:
                with_dist = [p for p in src_list if "raw_dtw_dist" in p]
                if len(with_dist) == len(src_list) and len(with_dist) > self.topk_src_episodes:
                    src_iter = sorted(with_dist, key=lambda p: float(p.get("raw_dtw_dist", float("inf"))))[: self.topk_src_episodes]
                else:
                    src_iter = src_list  # distances missing or not enough candidates
            else:
                src_iter = src_list

            # Merge all candidate src episodes for this target episode
            # into a dict: tgt_local_idx -> (List[src_global_idx], List[weight])
            per_tgt_index: Dict[int, Tuple[List[int], List[float]]] = {}
            for pair in src_iter:
                src_name = pair.get("demo_name")
                if src_name is None:
                    continue
                src_ep_idx = name2idx_src(str(src_name))
                if src_ep_idx < 0 or src_ep_idx >= len(self.src_spans):
                    continue
                src_span = self.src_spans[src_ep_idx]
                idx_map: Dict[str, List[int]] = pair.get("pairing", {})  # tgt_local -> list[src_local]
                # Episode-level weight from DTW distance, if available
                dist_val = pair.get("raw_dtw_dist", None)
                if dist_val is None or not isinstance(dist_val, (int, float)):
                    w_ep = 1.0
                else:
                    w_ep = math.exp(-self.sharpness * float(dist_val)) if self.sharpness > 0 else 1.0
                    if not math.isfinite(w_ep):
                        w_ep = 0.0
                    if w_ep == 0.0:
                        w_ep = 1e-12
                for tgt_local_str, src_locals in idx_map.items():
                    try:
                        tgt_local = int(tgt_local_str)
                    except Exception:
                        continue
                    # Bound-check within episode lengths
                    if tgt_local < 0 or tgt_local >= len(tgt_span):
                        continue
                    src_globals: List[int] = []
                    src_weights: List[float] = []
                    for s_local in src_locals:
                        if isinstance(s_local, (list, tuple)):
                            # In case pairing stores [idx] lists
                            if len(s_local) == 0:
                                continue
                            s_local = int(s_local[0])
                        s_local = int(s_local)
                        if 0 <= s_local < len(src_span):
                            g = src_span.start + s_local
                            src_globals.append(g)
                            src_weights.append(w_ep)
                    if len(src_globals) == 0:
                        continue
                    if tgt_local not in per_tgt_index:
                        per_tgt_index[tgt_local] = ([], [])
                    per_tgt_index[tgt_local][0].extend(src_globals)
                    per_tgt_index[tgt_local][1].extend(src_weights)

            # Finalize per-tgt sample list
            for tgt_local, src_candidates in per_tgt_index.items():
                tgt_global = tgt_span.start + int(tgt_local)
                if isinstance(src_candidates, tuple):
                    src_list_final, w_list_final = src_candidates
                else:
                    src_list_final = list(src_candidates)  # type: ignore[list-item]
                    w_list_final = [1.0] * len(src_list_final)
                if not any(w > 0 for w in w_list_final):
                    w_list_final = [1.0] * len(w_list_final)
                samples.append((tgt_global, src_list_final, w_list_final))

        if len(samples) == 0:
            raise ValueError(
                "No valid OT pairs built from pair_info. Check base_index_src/tgt or episode naming."
            )

        self._samples: List[Tuple[int, List[int], List[float]]] = samples

        # Precompute a lookup from global src index to its episode span index
        # for fast no_window sampling.
        try:
            n_src = len(self.ds_src)
        except Exception:
            n_src = max((span.end for span in self.src_spans), default=0)
        self._src_global_to_ep_idx: List[int] = [-1] * int(n_src)
        for ep_i, span in enumerate(self.src_spans):
            for g in range(int(span.start), int(span.end)):
                if 0 <= g < n_src:
                    self._src_global_to_ep_idx[g] = ep_i

    def __len__(self) -> int:
        return len(self._samples)

    def _select_indices(
        self, tgt_global: int, src_candidates: List[int], src_weights: List[float]
    ) -> Tuple[int, int]:
        """Pick concrete indices using window semantics and optional weighted / no-window sampling."""
        tgt_idx = tgt_global  # target is fixed
        # Weighted pick if possible
        if src_weights and any(w > 0 for w in src_weights):
            try:
                src_idx_center = random.choices(src_candidates, weights=src_weights, k=1)[0]
            except Exception:
                src_idx_center = random.choice(src_candidates)
        else:
            src_idx_center = random.choice(src_candidates)

        # no_window variant: ignore aligned index and sample anywhere within that source episode
        if self.no_window:
            ep_idx = -1
            if 0 <= src_idx_center < len(self._src_global_to_ep_idx):
                ep_idx = self._src_global_to_ep_idx[src_idx_center]
            if 0 <= ep_idx < len(self.src_spans):
                span = self.src_spans[ep_idx]
                if len(span) > 0:
                    src_idx = random.randint(int(span.start), int(span.end) - 1)
                    return tgt_idx, src_idx
            # Fallback
            return tgt_idx, src_idx_center

        # window jitter around aligned source index
        if self.window is not None:
            w = int(self.window)
            offset = random.randint(-w, w)
            src_idx = max(0, min(src_idx_center + offset, len(self.ds_src) - 1))
            return tgt_idx, src_idx

        # Default: use aligned index
        return tgt_idx, src_idx_center

    def _filter_obs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Select observation keys to include in OT samples.

        Supports exact keys and simple prefixes:
          - Exact match: include item[k] when k is present.
          - Prefix 'observation.images': include all keys starting with
            'observation.images.' (matches any camera name).
          - Prefix 'observation.image': include all keys starting with
            'observation.image' (for datasets that use singular form).
        """
        out: Dict[str, Any] = {}
        for req in self.obs_keys:
            if req in item:
                out[req] = item[req]
                continue
            # Prefix matching for image collections
            if isinstance(req, str) and (req == "observation.images" or req.startswith("observation.images")):
                for key, val in item.items():
                    if isinstance(key, str) and key.startswith("observation.images."):
                        out[key] = val
            elif isinstance(req, str) and (req == "observation.image" or req.startswith("observation.image")):
                for key, val in item.items():
                    if isinstance(key, str) and key.startswith("observation.image"):
                        out[key] = val
        return out

    def __getitem__(self, i: int) -> Dict[str, Dict[str, Any]]:
        tgt_global, src_candidates, src_weights = self._samples[i]
        tgt_idx, src_idx = self._select_indices(tgt_global, src_candidates, src_weights)

        tgt_item = self.ds_tgt[tgt_idx]
        src_item = self.ds_src[src_idx]

        # Keep structure close to robomimic OT dataset while using LeRobot keys
        tgt = {
            "obs": self._filter_obs(tgt_item),
            "actions": tgt_item.get(self.action_key),
        }
        src = {
            "obs": self._filter_obs(src_item),
            "actions": src_item.get(self.action_key),
        }

        # Ensure tensors
        if not torch.is_tensor(tgt["actions"]):
            tgt["actions"] = torch.as_tensor(tgt["actions"]) if tgt["actions"] is not None else None
        if not torch.is_tensor(src["actions"]):
            src["actions"] = torch.as_tensor(src["actions"]) if src["actions"] is not None else None

        return {"tgt": tgt, "src": src}


def collate_ot_samples(batch: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Custom collation to stack observation dicts and actions for src/tgt."""
    def _collate_side(side: str) -> Dict[str, Any]:
        obs_list = [b[side]["obs"] for b in batch]
        act_list = [b[side]["actions"] for b in batch]
        obs = {}
        obs_keys = set().union(*[o.keys() for o in obs_list])
        for k in obs_keys:
            vals = []
            for o in obs_list:
                if k in o:
                    v = o[k]
                    vals.append(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
            if len(vals) > 0:
                obs[k] = torch.stack(vals, dim=0)
        actions = torch.stack([a if isinstance(a, torch.Tensor) else torch.as_tensor(a) for a in act_list], dim=0)
        return {"obs": obs, "actions": actions}

    return {"src": _collate_side("src"), "tgt": _collate_side("tgt")}
