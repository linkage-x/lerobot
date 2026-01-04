"""
Build DTW-based pair_info.json from HIROL JSON episodes (pollen-robotics/dtw only).

This utility aligns each target episode against a set of source episodes using
Dynamic Time Warping (DTW) on selected HIROL features (e.g., joint states + gripper)
and saves a pair_info.json compatible with OT-style sampling.

Only the pollen-robotics/dtw implementation is supported here to keep the code
minimal. If the dtw package is missing, an informative error is raised.

Example

python -m lerobot.datasets.hirol.build_pair_info \
    --src-dir /data/fr3/1025_insert_tube_fr3_3dmouse_99ep \
    --tgt-dir /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_20eps \
    --label state_joint_gripper \
    --center first \
    --dtw-agg mean \
    --output src/lerobot/datasets/hirol/dtw/it_99to20_pair_info.json

Output JSON format (indices are 0-based):
{
  "episode_0001": [
    {
      "demo_name": "episode_0003",
      # Normalized DTW distance per the paper:
      #   d_bar(x_s, x_t) = d_DTW(x_s, x_t) / max(|x_s|, |x_t|)
      # where |x| is interpreted as the arc length of the trajectory under the
      # same metric as DTW (sum of per-step distances along time).
      "raw_dtw_dist": 0.0012,
      "pairing": { "0": [0], "1": [1, 2] }  # target idx -> list of aligned source indices
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    # pollen-robotics/dtw backend (supports cdist-accelerated distance)
    from dtw.dtw import dtw as pollen_dtw
    from dtw.dtw import accelerated_dtw as pollen_accel_dtw
except Exception as e:  # keep import-time error for a clearer message at runtime
    pollen_dtw = None  # type: ignore
    pollen_accel_dtw = None  # type: ignore
    _pollen_import_error = e
else:
    _pollen_import_error = None


# -------------------------
# IO utilities
# -------------------------

def _natural_key(s: str) -> Tuple:
    # Natural sort key for episode_0001-like names
    m = re.findall(r"\d+|\D+", s)
    return tuple(int(x) if x.isdigit() else x for x in m)


def list_episode_jsons(root: str, pattern: str = "episode_") -> List[Tuple[str, str]]:
    """List (episode_id, json_path) pairs under HIROL-style directory tree.

    Expected structure: <root>/<episode_xxxx>/data.json
    """
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dir not found: {root}")
    for name in os.listdir(root):
        if not name.startswith(pattern):
            continue
        p = os.path.join(root, name, "data.json")
        if os.path.isfile(p):
            out.append((name, p))
    out.sort(key=lambda x: _natural_key(x[0]))
    return out


def load_hirol_episode_features(json_path: str, label: str) -> np.ndarray:
    """Extract a T x D feature sequence from a HIROL episode JSON, following HIROL keys.

    Supported labels:
      - 'state_joint':             data[t].joint_states.single.position (J)
      - 'state_joint_gripper':     concat joint_states.single.position (J) and tools.single.position (1) => (J+1)
      - 'eef_pose':                data[t].ee_states.single.pose => (7): [x,y,z,qw,qx,qy,qz]
      - (also supports 'action_ee', 'action_ee_gripper', 'action_joint', 'action_joint_gripper')

    Returns:
      np.ndarray of shape (T, D), float32
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("data")
    if not isinstance(frames, list):
        raise ValueError(f"Expect data to be a list of frames: {json_path}")

    feats: List[np.ndarray] = []
    for t, fr in enumerate(frames):
        try:
            if label == "state_joint":
                jp = fr["joint_states"]["single"]["position"]
                vec = np.asarray(jp, dtype=np.float32)
            elif label == "state_joint_gripper":
                jp = fr["joint_states"]["single"]["position"]
                grip = fr["tools"]["single"]["position"]
                vec = np.asarray(list(jp) + [grip], dtype=np.float32)
            elif label == "eef_pose":
                pose = fr["ee_states"]["single"]["pose"]
                vec = np.asarray(pose, dtype=np.float32)
            elif label == "action_ee":
                ee = fr["actions"]["single"]["ee"]["pose"]
                vec = np.asarray(ee, dtype=np.float32)
            elif label == "action_ee_gripper":
                ee = fr["actions"]["single"]["ee"]["pose"]
                grip = fr["actions"]["single"]["tool"]["position"]
                vec = np.asarray(list(ee) + [grip], dtype=np.float32)
            elif label == "action_joint":
                jp = fr["actions"]["single"]["joint"]["position"]
                vec = np.asarray(jp, dtype=np.float32)
            elif label == "action_joint_gripper":
                jp = fr["actions"]["single"]["joint"]["position"]
                grip = fr["actions"]["single"]["tool"]["position"]
                vec = np.asarray(list(jp) + [grip], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported label: {label}")
        except KeyError as e:
            raise KeyError(f"Missing key {e} in frame {t} of {json_path}")

        if vec.ndim == 0:
            vec = vec[None]
        feats.append(vec)

    if not feats:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.stack(feats, axis=0)
    return arr.astype(np.float32)


# -------------------------
# Normalization helpers
# -------------------------

def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def compute_norm_stats(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cat = np.concatenate([s for s in seqs if s.size > 0], axis=0)
    mean = cat.mean(axis=0, keepdims=True)
    std = cat.std(axis=0, keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)


def center_sequence(x: np.ndarray, mode: str = "first") -> np.ndarray:
    """Center per-episode to remove constant bias.

    - 'first': subtract the first frame vector x[0]
    - 'mean': subtract the temporal mean vector mean_t x[t]
    - 'none': no centering
    """
    if x is None or x.size == 0:
        return x
    if mode == "first":
        return (x - x[:1]).astype(np.float32)
    if mode == "mean":
        return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)
    return x.astype(np.float32)


# -------------------------
# Main logic (pollen DTW only)
# -------------------------

def _resolve_output_path(output: str, src_dir: str, tgt_dir: str, label: str) -> str:
    """Allow --output to be either a file path or a directory.

    If it's a directory (existing or endswith path sep), auto-name the file:
      - <src>_pair_info.json if src==tgt
      - <src>_to_<tgt>_<label>_pair_info.json otherwise
    """
    if output.endswith(os.sep) or os.path.isdir(output):
        src_name = os.path.basename(os.path.normpath(src_dir))
        tgt_name = os.path.basename(os.path.normpath(tgt_dir))
        if src_name == tgt_name:
            fname = f"{src_name}_pair_info.json"
        else:
            fname = f"{src_name}_to_{tgt_name}_{label}_pair_info.json"
        output = os.path.join(output, fname)
    return output


def _arc_length(seq: np.ndarray, metric: str = "l2") -> float:
    """Compute trajectory arc length |x| under the given metric.

    |x| = sum_t d(x[t], x[t-1]) where d matches the DTW metric.
    """
    if seq is None or len(seq) <= 1:
        return 0.0
    diffs = seq[1:] - seq[:-1]
    if metric == "l1":
        return float(np.sum(np.abs(diffs))) if diffs.ndim == 1 else float(np.sum(np.abs(diffs), axis=1).sum())
    if metric == "linf":
        return float(np.max(np.abs(diffs), axis=1).sum())
    # l2
    return float(np.linalg.norm(diffs, axis=1).sum())


def _per_step_cost_stats(tgt_seq: np.ndarray, src_seq: np.ndarray, p: np.ndarray, q: np.ndarray, metric: str = "l2") -> Tuple[int, float, float, float, float]:
    """Compute per-step path statistics for debugging.

    Returns (path_len, avg, median, min, max).
    """
    if p is None or q is None or len(p) == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    diffs = tgt_seq[p] - src_seq[q]
    if metric == "l1":
        step = np.sum(np.abs(diffs), axis=1)
    elif metric == "linf":
        step = np.max(np.abs(diffs), axis=1)
    else:
        step = np.linalg.norm(diffs, axis=1)
    step = step.astype(np.float64)
    return int(len(step)), float(np.mean(step)), float(np.median(step)), float(np.min(step)), float(np.max(step))


def build_pair_info_from_hirol(
    src_dir: str,
    tgt_dir: str,
    label: str,
    output: str,
    top_k: int | None,
    window: int | None,
    stride: int,
    limit_src: int | None,
    limit_tgt: int | None,
    metric: str = "l2",
    center: str = "first",
    dtw_agg: str = "mean",  # 'mean' or 'sum'
    debug_summary: bool = False,
    debug_topk: int | None = None,
    debug_one: str | None = None,
):
    if pollen_dtw is None or pollen_accel_dtw is None:
        raise ImportError(
            "pollen-robotics/dtw is required. Please install the 'dtw' package to use this tool."
        )

    src_eps = list_episode_jsons(src_dir)
    tgt_eps = list_episode_jsons(tgt_dir)
    if limit_src is not None:
        src_eps = src_eps[:limit_src]
    if limit_tgt is not None:
        tgt_eps = tgt_eps[:limit_tgt]
    if not src_eps or not tgt_eps:
        raise RuntimeError("No episodes found under src or tgt directory.")

    # Load sequences and compute normalization
    src_seqs = [(eid, load_hirol_episode_features(p, label)) for eid, p in src_eps]
    tgt_seqs = [(eid, load_hirol_episode_features(p, label)) for eid, p in tgt_eps]

    # Stride downsample if needed
    if stride > 1:
        src_seqs = [(eid, s[::stride]) for eid, s in src_seqs]
        tgt_seqs = [(eid, s[::stride]) for eid, s in tgt_seqs]

    # Center per-episode to suppress constant offsets between sequences
    center = str(center).lower()
    if center not in ("none", "first", "mean"):
        center = "first"
    if center != "none":
        src_seqs = [(eid, center_sequence(s, mode=center)) for eid, s in src_seqs]
        tgt_seqs = [(eid, center_sequence(s, mode=center)) for eid, s in tgt_seqs]

    # Global z-score after centering
    mean, std = compute_norm_stats([s for _, s in src_seqs] + [s for _, s in tgt_seqs])
    src_seqs = [(eid, zscore(s, mean, std)) for eid, s in src_seqs]
    tgt_seqs = [(eid, zscore(s, mean, std)) for eid, s in tgt_seqs]

    # Map metric to cdist name for accelerated path
    metric_map = {"l2": "euclidean", "l1": "cityblock", "linf": "chebyshev"}
    dist_name = metric_map.get(metric, "euclidean")

    # Optional specific pair debug: parse "tgt_id:src_id"
    debug_tgt = debug_src = None
    if debug_one:
        try:
            parts = debug_one.split(":", 1)
            if len(parts) == 2:
                debug_tgt, debug_src = parts[0], parts[1]
        except Exception:
            debug_tgt = debug_src = None

    # Print run setup summary
    print(f"[DTW] src={len(src_seqs)} eps, tgt={len(tgt_seqs)} eps, label={label}, D={tgt_seqs[0][1].shape[1] if tgt_seqs and tgt_seqs[0][1].ndim==2 else 'N/A'}")
    print(f"[DTW] metric={metric} (cdist={dist_name}), stride={stride}, window={window}, center={center}, dtw_agg={dtw_agg}")

    result: Dict[str, List[Dict]] = {}
    for tgt_id, tgt_seq in tgt_seqs:
        pairs: List[Tuple[str, float, Dict[str, List[int]]]] = []
        # Precompute arc length of tgt once
        xt_len = _arc_length(tgt_seq, metric)
        lt = len(tgt_seq)
        D = tgt_seq.shape[1] if tgt_seq.ndim == 2 else 0
        for src_id, src_seq in src_seqs:
            # Use accelerated DTW when no window is requested; fall back to dtw() to honor window
            if window is None:
                cum_cost, _C, _D1, pq = pollen_accel_dtw(tgt_seq, src_seq, dist=dist_name, warp=1)
                p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)
            else:
                # If a window is specified, ensure w >= |len(x) - len(y)| as required by the library
                lt, ls = len(tgt_seq), len(src_seq)
                w_eff = int(max(window, abs(lt - ls)))

                def _metric(u, v):
                    if metric == "l1":
                        return float(np.sum(np.abs(u - v)))
                    if metric == "linf":
                        return float(np.max(np.abs(u - v)))
                    return float(np.linalg.norm(u - v))

                cum_cost, _C, _D1, pq = pollen_dtw(tgt_seq, src_seq, dist=_metric, warp=1, w=w_eff, s=1.0)
                p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)

            path = list(zip(p.tolist(), q.tolist()))
            if not path:
                continue

            # Convert alignment pairs back to original (pre-stride) indices
            pairing: Dict[int, List[int]] = defaultdict(list)
            for (ti, sj) in path:
                pairing[int(ti * stride)].append(int(sj * stride))

            # Paper-normalized DTW: d_DTW / max(|x_s|, |x_t|), where |x| is arc length.
            # Here, d_DTW can be the mean per-step path cost ('mean') or the total path cost ('sum').
            xs_len = _arc_length(src_seq, metric)
            denom = max(xs_len, xt_len)
            path_len = max(1, int(len(p)))
            cost_val = float(cum_cost) if dtw_agg == "sum" else float(cum_cost) / float(path_len)
            dist_val = float(cost_val / denom) if denom > 1e-12 else float("inf")

            # Per-pair debug: only when explicitly requested for this (tgt,src)
            if debug_tgt is not None and debug_src is not None and tgt_id == debug_tgt and src_id == debug_src:
                path_len, step_avg, step_med, step_min, step_max = _per_step_cost_stats(tgt_seq, src_seq, p, q, metric)
                print((
                    f"[DEBUG one] tgt={tgt_id} (T={lt}, D={D}) src={src_id} (T={len(src_seq)}) "
                    f"cum_cost={cum_cost:.6f} cost_mode={dtw_agg} cost_val={cost_val:.6f} "
                    f"|xs|={xs_len:.6f} |xt|={xt_len:.6f} denom={denom:.6f} ratio={dist_val:.6f} "
                    f"path_len={path_len} step_avg={step_avg:.6f} step_med={step_med:.6f} step_min={step_min:.6f} step_max={step_max:.6f}"
                ))

            pairs.append((src_id, dist_val, {str(k): v for k, v in pairing.items()}))

        # Keep top-k by cost (lower is better)
        pairs.sort(key=lambda x: x[1])
        if top_k is not None and top_k > 0:
            pairs = pairs[:top_k]

        # Per-target summary debug for top few pairs
        if debug_summary and len(pairs) > 0:
            show_k = len(pairs) if not debug_topk or debug_topk <= 0 else min(debug_topk, len(pairs))
            print(f"[DEBUG sum] tgt={tgt_id} (T={lt}, D={D}) |xt|={xt_len:.6f}; top{show_k} pairs by ratio")
            for sid, dist_val, pairing in pairs[:show_k]:
                # recompute DTW path stats for display (run accelerated again with same settings)
                src_seq = next(s for s_id, s in src_seqs if s_id == sid)
                if window is None:
                    cum_cost, _C, _D1, pq = pollen_accel_dtw(tgt_seq, src_seq, dist=dist_name, warp=1)
                    p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)
                else:
                    lt2, ls2 = len(tgt_seq), len(src_seq)
                    w_eff = int(max(window, abs(lt2 - ls2)))
                    def _metric(u, v):
                        if metric == "l1":
                            return float(np.sum(np.abs(u - v)))
                        if metric == "linf":
                            return float(np.max(np.abs(u - v)))
                        return float(np.linalg.norm(u - v))
                    cum_cost, _C, _D1, pq = pollen_dtw(tgt_seq, src_seq, dist=_metric, warp=1, w=w_eff, s=1.0)
                    p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)
                xs_len = _arc_length(src_seq, metric)
                denom = max(xs_len, xt_len)
                path_len2 = max(1, int(len(p)))
                cost_val2 = float(cum_cost) if dtw_agg == "sum" else float(cum_cost) / float(path_len2)
                path_len, step_avg, step_med, step_min, step_max = _per_step_cost_stats(tgt_seq, src_seq, p, q, metric)
                print((
                    f"  - src={sid} (T={len(src_seq)}) cum_cost={cum_cost:.6f} cost_val={cost_val2:.6f} "
                    f"|xs|={xs_len:.6f} denom={denom:.6f} "
                    f"ratio={float(cost_val2/denom) if denom>1e-12 else float('inf'):.6f} (stored={dist_val:.6f}) "
                    f"path_len={path_len} step_avg={step_avg:.6f} step_med={step_med:.6f} step_min={step_min:.6f} step_max={step_max:.6f}"
                ))

        # Emit distance directly as raw_dtw_dist (paper normalization)
        out_list = []
        for sid, dist_val, pairing in pairs:
            out_list.append({"demo_name": sid, "raw_dtw_dist": float(dist_val), "pairing": pairing})

        result[tgt_id] = out_list

    # Save once after all targets are processed
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build DTW pair_info from HIROL JSON episodes (pollen DTW)")
    parser.add_argument("--src-dir", required=True, help="source episodes root (contains episode_xxxx/data.json)")
    parser.add_argument("--tgt-dir", required=True, help="target episodes root (contains episode_xxxx/data.json)")
    parser.add_argument(
        "--label",
        default="state_joint_gripper",
        choices=[
            "state_joint",
            "state_joint_gripper",
            "eef_pose",
            "action_ee",
            "action_ee_gripper",
            "action_joint",
            "action_joint_gripper",
        ],
        help="feature used for DTW",
    )
    parser.add_argument("--output", required=True, help="output json path or directory")
    parser.add_argument("--top-k", type=int, default=None, help="keep top-K best source demos per target")
    parser.add_argument("--window", type=int, default=None, help="Sakoe-Chiba band half-width; smaller = faster")
    parser.add_argument("--stride", type=int, default=1, help="temporal stride to downsample before DTW")
    parser.add_argument("--limit-src", type=int, default=None, help="limit number of source episodes (debug)")
    parser.add_argument("--limit-tgt", type=int, default=None, help="limit number of target episodes (debug)")
    parser.add_argument(
        "--metric",
        choices=["l2", "l1", "linf"],
        default="l2",
        help="distance metric (used by pollen backend)",
    )
    parser.add_argument(
        "--dtw-agg",
        choices=["mean", "sum"],
        default="mean",
        help="aggregate along DTW path for d_DTW: 'mean' (per-step average) or 'sum' (total path cost)",
    )
    parser.add_argument(
        "--center",
        choices=["none", "first", "mean"],
        default="first",
        help=(
            "per-episode centering before z-score: 'first' subtracts the first frame, 'mean' subtracts temporal mean"
        ),
    )
    parser.add_argument("--debug-summary", action="store_true", help="print per-target summary with top pairs")
    parser.add_argument("--debug-topk", type=int, default=3, help="how many top pairs to show per target in debug summary")
    parser.add_argument("--debug-one", type=str, default=None, help="debug a specific pair: '<tgt_id>:<src_id>'")

    args = parser.parse_args()

    out_path = _resolve_output_path(args.output, args.src_dir, args.tgt_dir, args.label)

    build_pair_info_from_hirol(
        src_dir=args.src_dir,
        tgt_dir=args.tgt_dir,
        label=args.label,
        output=out_path,
        top_k=args.top_k,
        window=args.window,
        stride=args.stride,
        limit_src=args.limit_src,
        limit_tgt=args.limit_tgt,
        metric=args.metric,
        center=args.center,
        dtw_agg=args.dtw_agg,
        debug_summary=bool(args.debug_summary),
        debug_topk=args.debug_topk,
        debug_one=args.debug_one,
    )


if __name__ == "__main__":
    main()
