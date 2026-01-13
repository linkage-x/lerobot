#!/usr/bin/env python3
"""
Draw EE x/y trajectories with z encoded as colormap.

Input tree (per episode):
  <root>/episode_xxxx/data.json

The script scans episode_* dirs under --root, parses data.json, extracts
ee_states.<ee_key>.pose[0:3] as (x,y,z), and saves:
  - a single combined PNG overlaying all episodes (default)
  - optionally per-episode PNGs (use --per-episode)

Usage examples:
  python -m lerobot.datasets.hirol.utils.draw_episodes_pts \
    --root /data/2025/unitree_co-train/bread_picking/trimmed/1227_duo_unitree_bread_picking_human_182ep \
    --out  ./outputs/ee_xy_plots \
    --ee-key right \
    --cmap viridis
    --blend mean --quantize 0.0 --combined-name all_episodes_xy_z.png

Notes:
  - If --ee-key is omitted, will try 'right' then 'left', then any available key.
  - Steps without a valid pose are ignored. Episodes with no valid points are skipped.
  - Combined mode blending when multiple points land on the same (x,y):
      * --blend alpha: alpha blending, no de-duplication
      * --blend mean|median|min|max: aggregate z for coincident (x,y)
        Optionally use --quantize <step> to bin (x,y) before aggregating.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Force headless backend for CLI usage
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def find_episode_dirs(root: Path) -> list[Path]:
    """Return episode directories under root, sorted by name."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a dir: {root}")
    eps = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    eps.sort(key=lambda p: p.name)
    return eps


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def choose_ee_key(sample_step: dict, preferred: Optional[str]) -> Optional[str]:
    ee = sample_step.get("ee_states")
    if not isinstance(ee, dict) or len(ee) == 0:
        return None
    keys = list(ee.keys())
    # Try preferred, then common defaults, then any available
    for k in (preferred, "right", "left"):
        if isinstance(k, str) and k in keys:
            return k
    return keys[0]


def extract_xyz(data_steps: Iterable[dict], ee_key: Optional[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """Extract x,y,z arrays from steps. Returns (x,y,z, used_ee_key)."""
    xs, ys, zs = [], [], []
    used_key: Optional[str] = None
    # Try to lock ee_key from the first valid step if not provided
    if ee_key is None:
        for st in data_steps:
            k = choose_ee_key(st, None)
            if k is not None:
                used_key = k
                break
    else:
        used_key = ee_key

    if used_key is None:
        return np.array([]), np.array([]), np.array([]), None

    for st in data_steps:
        ee_states = st.get("ee_states")
        if not isinstance(ee_states, dict):
            continue
        entry = ee_states.get(used_key)
        if not isinstance(entry, dict):
            continue
        pose = entry.get("pose")
        if not isinstance(pose, list) or len(pose) < 3:
            continue
        # pose[0:3] -> x,y,z
        x, y, z = pose[0], pose[1], pose[2]
        # Sanity check numeric
        try:
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
        except (TypeError, ValueError):
            continue

    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([]), used_key
    return np.array(xs), np.array(ys), np.array(zs), used_key


def plot_episode_xy_with_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    dpi: int = 150,
) -> None:
    """Scatter X vs Y, color=Z with colorbar, save to out_path."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    sc = ax.scatter(x, y, c=z, s=6, cmap=cmap, edgecolors="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("z")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def quantize_xy(x: np.ndarray, y: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Quantize x,y onto a grid with cell size 'step'."""
    if step <= 0:
        return x, y
    qx = np.floor(x / step + 0.5) * step
    qy = np.floor(y / step + 0.5) * step
    return qx, qy


def aggregate_same_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray, how: str = "mean", quant_step: float = 0.0):
    """Aggregate duplicate (x,y) by 'how' for z. Optionally quantize before groupby."""
    if x.size == 0:
        return x, y, z
    if quant_step > 0:
        x, y = quantize_xy(x, y, quant_step)
    # group by coordinate pairs
    coords = np.column_stack([x, y])
    # Use structured array for np.unique on rows
    dtype = np.dtype([("x", x.dtype), ("y", y.dtype)])
    view = np.ascontiguousarray(coords).view(dtype)
    uniq, idx, inv, cnt = np.unique(view, return_index=True, return_inverse=True, return_counts=True)
    out_x = np.empty(len(uniq), dtype=x.dtype)
    out_y = np.empty(len(uniq), dtype=y.dtype)
    out_z = np.empty(len(uniq), dtype=z.dtype)
    for i in range(len(uniq)):
        mask = inv == i
        out_x[i] = x[idx[i]]
        out_y[i] = y[idx[i]]
        vals = z[mask]
        if how == "mean":
            out_z[i] = float(np.mean(vals))
        elif how == "median":
            out_z[i] = float(np.median(vals))
        elif how == "min":
            out_z[i] = float(np.min(vals))
        elif how == "max":
            out_z[i] = float(np.max(vals))
        else:  # fallback to mean
            out_z[i] = float(np.mean(vals))
    return out_x, out_y, out_z


def process_one_episode(ep_dir: Path, out_dir: Path, ee_key: Optional[str], cmap: str) -> Optional[Path]:
    data_json = ep_dir / "data.json"
    if not data_json.exists():
        print(f"[WARN] data.json missing: {data_json}")
        return None
    try:
        payload = read_json(data_json)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] JSON error {data_json}: {e}")
        return None

    steps = payload.get("data")
    if not isinstance(steps, list) or len(steps) == 0:
        print(f"[WARN] No steps: {data_json}")
        return None

    x, y, z, used_key = extract_xyz(steps, ee_key)
    if used_key is None or x.size == 0:
        print(f"[WARN] No valid ee pose: {data_json}")
        return None

    title = f"{ep_dir.name}  ee={used_key}  n={x.size}"
    out_name = f"{ep_dir.name}_{used_key}_xy_z.png"
    out_path = out_dir / out_name
    try:
        plot_episode_xy_with_z(x, y, z, out_path, title, cmap=cmap)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Plot failed {ep_dir.name}: {e}")
        return None
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Draw EE x/y with z colormap (combined by default)")
    p.add_argument("--root", required=True, help="Root folder containing episode_* subfolders")
    p.add_argument("--out", default=None, help="Output folder for PNGs (default: <root>/plots)")
    p.add_argument("--ee-key", default=None, help="EE key to use (e.g., right/left). If omitted, auto-detect.")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    p.add_argument("--per-episode", action="store_true", help="Also save per-episode PNGs in addition to the combined figure")
    p.add_argument("--combined-name", default="all_episodes_xy_z.png", help="Filename for the combined figure")
    p.add_argument("--blend", default="mean", choices=["alpha", "mean", "median", "min", "max"], help="Blending/aggregation strategy for coincident (x,y) in combined plot")
    p.add_argument("--quantize", type=float, default=0.0, help="Quantize (x,y) onto a grid of this size before aggregation (0 disables)")
    args = p.parse_args(argv)

    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else (root / "plots")
    ee_key = args.ee_key
    cmap = args.cmap

    try:
        eps = find_episode_dirs(root)
    except FileNotFoundError as e:
        print(str(e))
        return 2

    if len(eps) == 0:
        print(f"No episode_* folders found in: {root}")
        return 1

    saved = 0
    # Per-episode outputs (optional)
    if args.per-episode:
        for ep in eps:
            out_path = process_one_episode(ep, out_dir, ee_key, cmap)
            if out_path is not None:
                saved += 1
                print(f"[OK] Saved {out_path}")

    # Combined overlay (default)
        all_x, all_y, all_z = [], [], []
        used_key_global: Optional[str] = None
        for ep in eps:
            data_json = ep / "data.json"
            try:
                payload = read_json(data_json)
            except Exception:
                continue
            steps = payload.get("data")
            if not isinstance(steps, list) or len(steps) == 0:
                continue
            x, y, z, used_key = extract_xyz(steps, ee_key)
            if used_key_global is None:
                used_key_global = used_key
            if x.size == 0:
                continue
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
        if len(all_x) > 0:
            X = np.concatenate(all_x, axis=0)
            Y = np.concatenate(all_y, axis=0)
            Z = np.concatenate(all_z, axis=0)
            if args.blend == "alpha":
                # Direct scatter; overlapping points naturally overplot. Use low alpha to blend.
                fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
                sc = ax.scatter(X, Y, c=Z, s=4, cmap=cmap, edgecolors="none", alpha=0.3)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
                title = f"All episodes  ee={used_key_global or (ee_key or 'auto')}  n={X.size}"
                ax.set_title(title)
                cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
                cbar.set_label("z")
                fig.tight_layout()
                out_path = out_dir / args.combined_name
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path)
                plt.close(fig)
                print(f"[OK] Saved combined {out_path}")
                saved += 1
            else:
                # Aggregate coincident points by z-statistic
                X2, Y2, Z2 = aggregate_same_xy(X, Y, Z, how=args.blend, quant_step=args.quantize)
                fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
                sc = ax.scatter(X2, Y2, c=Z2, s=6, cmap=cmap, edgecolors="none")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
                title = f"All episodes (agg={args.blend}, q={args.quantize})  ee={used_key_global or (ee_key or 'auto')}  n={X2.size}"
                ax.set_title(title)
                cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
                cbar.set_label("z")
                fig.tight_layout()
                out_path = out_dir / args.combined_name
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path)
                plt.close(fig)
                print(f"[OK] Saved combined {out_path}")
                saved += 1
    print(f"Done. Episodes processed: {len(eps)}, images saved: {saved}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
