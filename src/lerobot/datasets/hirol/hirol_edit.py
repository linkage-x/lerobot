"""
HIROL Episode Folder Utilities

Purpose: copy a slice of episodes [i, j] from a HIROL-style dataset directory
into a destination directory, preserving the per-episode subfolder structure
(e.g., episode_0001/, episode_0002/, ...).

Why a separate tool? While `lerobot_edit.py` auto-detects LeRobot v3 datasets
and can fallback to HIROL directories, this dedicated script keeps HIROL-only
operations explicit and minimal for workflows dealing purely with HIROL data.

Example
  python -m lerobot.datasets.hirol.hirol_edit copy-slice \
    --src /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_279eps \
    --dst /data/fr3/1107_subset_1_20 \
    --start 1 --end 20

Space-saving (same filesystem) via hard links
  python -m lerobot.datasets.hirol.hirol_edit copy-slice \
    --src /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_279eps \
    --dst /data/fr3/1107_subset_link_1_20 \
    --start 1 --end 20 --mode hardlink --fallback-copy
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


# ------------------------------ discovery ------------------------------


_EPISODE_RE = re.compile(r"^episode_(\d+)$")


def _natural_episode_key(name: str) -> Tuple[int, str]:
    """Sort key for names like 'episode_0001'.

    Returns (episode_number, original_name). Unknown formats go last.
    """
    m = _EPISODE_RE.match(name)
    if m:
        return int(m.group(1)), name
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
    """Copy a directory tree using a file-level copy function."""
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    shutil.copytree(src, dst, copy_function=copy_function, symlinks=symlinks)


def _hardlink_copy(src: str, dst: str) -> None:
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
    mode: str = "copy"  # copy | hardlink | symlink
    overwrite: bool = False
    fallback_copy: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if self.start <= 0 or self.end <= 0:
            raise ValueError("start and end must be positive (1-based)")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.mode not in ("copy", "hardlink", "symlink"):
            raise ValueError("mode must be: copy | hardlink | symlink")


def _select_copy_function(mode: str) -> CopyFn:
    return {"copy": _file_copy, "hardlink": _hardlink_copy, "symlink": _symlink_copy}[mode]


def copy_slice(opts: CopySliceOptions) -> Tuple[int, List[EpisodeEntry]]:
    """Copy episodes [start, end] from src to dst (HIROL)."""
    opts.validate()
    eps = discover_episodes(opts.src)
    if not eps:
        raise RuntimeError(f"no episodes found under: {opts.src}")

    # Clamp the selection to available indices
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
        return 0, selected

    _ensure_dir(opts.dst)
    copy_fn = _select_copy_function(opts.mode)

    copied = 0
    for e in selected:
        dst_ep_dir = opts.dst / e.name
        if dst_ep_dir.exists():
            if not opts.overwrite:
                continue
            shutil.rmtree(dst_ep_dir)

        try:
            _copytree(e.path, dst_ep_dir, copy_function=copy_fn)
            copied += 1
        except (OSError, shutil.Error) as ex:
            if opts.fallback_copy and opts.mode in ("hardlink", "symlink"):
                if dst_ep_dir.exists():
                    shutil.rmtree(dst_ep_dir, ignore_errors=True)
                _copytree(e.path, dst_ep_dir, copy_function=_file_copy)
                copied += 1
            else:
                if dst_ep_dir.exists():
                    shutil.rmtree(dst_ep_dir, ignore_errors=True)
                raise RuntimeError(
                    f"failed to copy {e.path} -> {dst_ep_dir} with mode={opts.mode}: {ex}"
                ) from ex

    return copied, selected


# -------------------------------- CLI ---------------------------------


def _add_copy_slice_subparser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "copy-slice",
        help="Copy episodes [start, end] from src to dst (HIROL dir)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, type=Path, help="source HIROL directory (contains episode_xxxx)")
    p.add_argument("--dst", required=True, type=Path, help="destination directory")
    p.add_argument("--start", required=True, type=int, help="start episode index (1-based, inclusive)")
    p.add_argument("--end", required=True, type=int, help="end episode index (1-based, inclusive)")
    p.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy", help="file strategy")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing episodes in dst")
    p.add_argument("--fallback-copy", action="store_true", help="fallback to copy if link fails")
    p.add_argument("--dry-run", action="store_true", help="list episodes to be copied without writing")
    p.set_defaults(_cmd="copy-slice")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIROL episode folder editor",
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

