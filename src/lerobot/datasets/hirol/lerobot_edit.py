"""
Lightweight utilities to edit HIROL-style episode folders.

Current feature: copy a slice of episodes [i, j] from one dataset root to
another, keeping the original on-disk structure (e.g. episode_0001/...).

Design goals
- Small, dependency-free, and easy to extend with new subcommands.
- Clear separation of concerns: discover -> select -> operate (copy/link).
- Safe by default: never overwrite unless explicitly allowed.

Example
  python -m lerobot.datasets.hirol.lerobot_edit \
    copy-slice \
    --src /data/fr3_lerobot/1107_insert_tube_fr3_3dmouse_contain_ft_279eps \
    --dst /data/fr3_lerobot/1107_it_subset_1_20 \
    --start 1 --end 20

Large datasets: use hard links to save space on the same filesystem
  python -m lerobot.datasets.hirol.lerobot_edit copy-slice \
    --src ... --dst ... --start 1 --end 20 --mode hardlink

Note: hard linking across different filesystems will fail; the tool will
fall back to copying if --fallback-copy is enabled.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# ------------------------------ discovery ------------------------------


_EPISODE_RE = re.compile(r"^episode_(\d+)$")


def _natural_episode_key(name: str) -> Tuple[int, str]:
    """Sort key for names like 'episode_0001'.

    Returns (episode_number, original_name). Unknown formats go last with 0.
    """
    m = _EPISODE_RE.match(name)
    if m:
        return int(m.group(1)), name
    # Fallback: try to extract trailing digits anywhere in the name
    m2 = re.search(r"(\d+)$", name)
    if m2:
        return int(m2.group(1)), name
    return sys.maxsize, name


@dataclass(frozen=True)
class EpisodeEntry:
    idx: int  # 1-based episode index parsed from folder name
    name: str  # folder name
    path: Path  # absolute path


def discover_episodes(root: Path) -> List[EpisodeEntry]:
    """List episode directories under a root.

    Expected structure: <root>/episode_xxxx/...
    Unknown directories are ignored.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a dir: {root}")

    entries: List[EpisodeEntry] = []
    for de in os.scandir(root):
        if not de.is_dir():
            continue
        ep_num, _ = _natural_episode_key(de.name)
        if ep_num is sys.maxsize:
            continue
        entries.append(EpisodeEntry(idx=ep_num, name=de.name, path=Path(de.path)))

    entries.sort(key=lambda e: e.idx)
    return entries


# ------------------------------- actions -------------------------------


CopyFn = Callable[[str, str], None]


def _copytree(src: Path, dst: Path, copy_function: CopyFn, symlinks: bool = False) -> None:
    """Copy a directory tree.

    shutil.copytree accepts a file-level copy_function, so we wrap it to keep
    the call-sites short and to handle dirs_exist_ok pre-3.8 consistently.
    """
    # Python 3.8+ supports dirs_exist_ok; for safety, implement manually
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    shutil.copytree(src, dst, copy_function=copy_function, symlinks=symlinks)


def _hardlink_copy(src: str, dst: str) -> None:
    """Create a hard link for a file; fallback to shutil.copy2 if not supported.

    Note: hard links only work on the same filesystem. This helper is intended
    to be used only when the caller agrees with best-effort behavior.
    """
    os.link(src, dst)


def _symlink_copy(src: str, dst: str) -> None:
    os.symlink(src, dst)


def _file_copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class CopySliceOptions:
    src: Path
    dst: Path
    start: int  # inclusive, 1-based
    end: int  # inclusive, 1-based
    mode: str = "copy"  # one of: copy, hardlink, symlink
    overwrite: bool = False
    fallback_copy: bool = False  # if hardlink/symlink fail, fallback to copy
    dry_run: bool = False

    def validate(self) -> None:
        if self.start <= 0 or self.end <= 0:
            raise ValueError("start and end must be positive (1-based)")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.mode not in ("copy", "hardlink", "symlink"):
            raise ValueError("mode must be one of: copy, hardlink, symlink")


def _select_copy_function(mode: str) -> CopyFn:
    if mode == "copy":
        return _file_copy
    if mode == "hardlink":
        return _hardlink_copy
    if mode == "symlink":
        return _symlink_copy
    raise ValueError(f"unknown mode: {mode}")


def _is_lerobot_v3_dataset(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        import json

        with open(info_path, "r") as f:
            info = json.load(f)
        version = str(info.get("codebase_version", ""))
        return version.startswith("v3")
    except Exception:
        return False


def copy_slice(opts: CopySliceOptions) -> Tuple[int, List[EpisodeEntry]]:
    """Copy episodes [start, end] from src to dst.

    Returns (copied_count, selected_entries)
    """
    opts.validate()

    # Branch: LeRobot v3 dataset (meta/info.json present)
    if _is_lerobot_v3_dataset(opts.src):
        return _copy_slice_lerobot_v3(opts)

    # Branch: HIROL-style folders
    eps = discover_episodes(opts.src)
    if not eps:
        raise RuntimeError(f"no episodes found under: {opts.src}")

    # Clamp the selection to available indices to be forgiving
    min_idx, max_idx = eps[0].idx, eps[-1].idx
    sel_start = max(opts.start, min_idx)
    sel_end = min(opts.end, max_idx)
    selected = [e for e in eps if sel_start <= e.idx <= sel_end]
    if not selected:
        raise RuntimeError(
            f"no episodes in requested range [{opts.start}, {opts.end}] present in source; "
            f"available [{min_idx}, {max_idx}]"
        )

    if opts.dry_run:
        # No filesystem operations, just report
        return 0, selected

    _ensure_dir(opts.dst)

    copy_fn = _select_copy_function(opts.mode)

    copied = 0
    for e in selected:
        dst_ep_dir = opts.dst / e.name
        if dst_ep_dir.exists():
            if not opts.overwrite:
                # Skip existing by default
                continue
            else:
                # Remove the existing directory before copying
                shutil.rmtree(dst_ep_dir)

        try:
            _copytree(e.path, dst_ep_dir, copy_function=copy_fn)
            copied += 1
        except (OSError, shutil.Error) as ex:
            if opts.fallback_copy and opts.mode in ("hardlink", "symlink"):
                # Clean up incomplete destination and retry with regular copy
                if dst_ep_dir.exists():
                    shutil.rmtree(dst_ep_dir, ignore_errors=True)
                _copytree(e.path, dst_ep_dir, copy_function=_file_copy)
                copied += 1
            else:
                # Clean up incomplete destination to avoid leaving partial data
                if dst_ep_dir.exists():
                    shutil.rmtree(dst_ep_dir, ignore_errors=True)
                raise RuntimeError(
                    f"failed to copy {e.path} -> {dst_ep_dir} with mode={opts.mode}: {ex}"
                ) from ex

    return copied, selected


# --------------------------- LeRobot v3 path ---------------------------


def _copy_slice_lerobot_v3(opts: CopySliceOptions) -> Tuple[int, List[EpisodeEntry]]:
    """Create a LeRobot v3 subset dataset with episodes [start, end].

    Uses dataset_tools.delete_episodes under the hood to write a clean subset
    to `opts.dst`, reindexing episodes to start from 0 and copying only needed
    frames/videos (re-encoding videos when necessary).
    """
    # Lazy imports to avoid heavy deps/load for HIROL path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.utils import load_info

    src_root = opts.src
    dst_root = opts.dst

    info = load_info(src_root)
    total = int(info.get("total_episodes", 0))
    if total <= 0:
        raise RuntimeError(f"source LeRobot v3 dataset has no episodes: {src_root}")

    # Convert 1-based [start, end] -> 0-based [s0, e0] inclusive, and clamp
    s0 = max(0, opts.start - 1)
    e0 = min(total - 1, opts.end - 1)
    if e0 < s0:
        raise RuntimeError(
            f"requested range [{opts.start}, {opts.end}] is empty after clamping to [0, {total - 1}]"
        )

    keep = list(range(s0, e0 + 1))

    if opts.dry_run:
        # Return a synthetic EpisodeEntry list for reporting (name=episode_XXXX)
        sel = [
            EpisodeEntry(idx=i + 1, name=f"episode_{i+1:04d}", path=src_root / f"episode_{i+1:04d}")
            for i in keep
        ]
        return 0, sel

    # Instantiate the source dataset from local disk; repo_id can be any label.
    src_ds = LeRobotDataset(repo_id="local", root=src_root)

    # Build delete list as complement
    delete = [i for i in range(total) if i not in keep]

    # Derive a repo_id for the new dataset from destination name
    new_repo_id = dst_root.name or "subset"

    # Create subset by deleting the complement into dst_root
    delete_episodes(dataset=src_ds, episode_indices=delete, output_dir=dst_root, repo_id=new_repo_id)

    # Report synthetic entries copied (as LeRobot v3 has no per-episode dirs)
    sel = [
        EpisodeEntry(idx=i + 1, name=f"episode_{i+1:04d}", path=dst_root)
        for i in keep
    ]
    return len(sel), sel


# ------------------------------- CLI ----------------------------------


def _add_copy_slice_subparser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "copy-slice",
        help="Copy episodes [start, end] from src to dst",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, type=Path, help="source dataset root")
    p.add_argument("--dst", required=True, type=Path, help="destination root")
    p.add_argument("--start", required=True, type=int, help="start episode index (1-based, inclusive)")
    p.add_argument("--end", required=True, type=int, help="end episode index (1-based, inclusive)")
    p.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="file copy strategy",
    )
    p.add_argument("--overwrite", action="store_true", help="overwrite existing episodes in dst")
    p.add_argument(
        "--fallback-copy",
        action="store_true",
        help="on hardlink/symlink failure, fallback to regular copy",
    )
    p.add_argument("--dry-run", action="store_true", help="list episodes to be copied without writing")
    p.set_defaults(_cmd="copy-slice")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIROL dataset editor utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="_cmd", required=True)
    _add_copy_slice_subparser(sub)
    return parser.parse_args(argv)


def _run_from_args(ns: argparse.Namespace) -> int:
    if ns._cmd == "copy-slice":
        opts = CopySliceOptions(
            src=ns.src,
            dst=ns.dst,
            start=ns.start,
            end=ns.end,
            mode=ns.mode,
            overwrite=ns.overwrite,
            fallback_copy=ns.fallback_copy,
            dry_run=ns.dry_run,
        )
        copied, selected = copy_slice(opts)
        if ns.dry_run:
            print(f"[dry-run] would copy {len(selected)} episodes to {ns.dst}:")
            for e in selected:
                print(f"  - {e.name} ({e.idx})")
            return 0
        print(f"copied {copied} / {len(selected)} episodes to {ns.dst}")
        return 0

    raise ValueError(f"unknown command: {ns._cmd}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        ns = _parse_args(argv)
        return _run_from_args(ns)
    except Exception as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
