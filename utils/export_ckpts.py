#!/usr/bin/env python3
"""
Export LeRobot dataset meta and pretrained model into a single zip.

Parameters:
  --data_cfg     Data conversion YAML path. Default:
                 src/lerobot/datasets/hirol/config/insert_pinboard.yaml
  --train_cfg    Training JSON path. Default:
                 src/lerobot/scripts/train_config/act.json
  --snapshot     Checkpoint snapshot under 'checkpoints' (e.g. 'last', '000100'). Default: last

Output:
  Creates a temporary directory under /tmp and writes <run_name>.zip inside it,
  where <run_name> is derived from the training config's output_dir.name.
  The zip root contains:
    - meta/ (dataset meta folder)
    - all files and subfolders from pretrained_model/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise


def _resolve_dataset_meta_dir(data_cfg_path: Path) -> tuple[Path, str]:
    """Resolve dataset meta dir from YAML (root_path/repo_name/meta)."""
    with open(data_cfg_path, "r", encoding="utf-8") as f:
        dc = yaml.safe_load(f)

    try:
        root_path = dc["root_path"]
        repo_name = dc["repo_name"]
    except KeyError as e:
        raise KeyError(f"Missing key in data config '{data_cfg_path}': {e}")

    # If root_path is absolute, use it; else resolve relative to hirol dir
    # hirol dir: .../datasets/hirol (parent of 'config')
    hirol_dir = data_cfg_path.parent.parent.resolve()
    rp = Path(str(root_path)).expanduser()
    dataset_root = rp if rp.is_absolute() else (hirol_dir / rp)
    meta_dir = (dataset_root / repo_name / "meta").resolve()

    if not meta_dir.exists():
        raise FileNotFoundError(f"Dataset meta directory not found: {meta_dir}")

    return meta_dir, repo_name


def _load_train_output_dir(train_cfg_path: Path) -> Path:
    """Parse training config (JSON or YAML) for output_dir and resolve a real path."""
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Try JSON first (backward compatible), then fall back to YAML so that
    # we can support both LeRobot JSON configs and VLASH/LeRobot YAML configs.
    try:
        tc = json.loads(raw_text)
    except json.JSONDecodeError:
        tc = yaml.safe_load(raw_text)

    if "output_dir" not in tc:
        raise KeyError(f"'output_dir' missing in training config: {train_cfg_path}")

    raw = tc["output_dir"]
    p = Path(str(raw)).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(train_cfg_path.parent / p)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    tried = "\n".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve output_dir '{raw}'. Tried:\n{tried}"
    )


def _resolve_pretrained_model_dir(output_dir: Path, snapshot: str) -> Path:
    pm_dir = (output_dir / "checkpoints" / snapshot / "pretrained_model").resolve()
    if not pm_dir.exists():
        raise FileNotFoundError(f"Pretrained model directory not found: {pm_dir}")
    return pm_dir


def _copy_pretrained_contents(src_dir: Path, dst_dir: Path) -> None:
    """Copy all files and subfolders from src_dir to dst_dir root."""
    for entry in src_dir.iterdir():
        dst_path = dst_dir / entry.name
        if entry.is_file():
            shutil.copy2(entry, dst_path)
        elif entry.is_dir():
            shutil.copytree(entry, dst_path, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package dataset meta and pretrained model into a zip")
    parser.add_argument(
        "--data_cfg",
        type=str,
        default="src/lerobot/datasets/hirol/config/insert_pinboard.yaml",
        help="Data conversion YAML path",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="src/lerobot/scripts/train_config/act.json",
        help="Training config path (JSON or YAML)",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="last",
        help="Checkpoint snapshot name under 'checkpoints' (e.g. 'last', '000100')",
    )

    args = parser.parse_args()

    data_cfg_path = Path(args.data_cfg).resolve()
    train_cfg_path = Path(args.train_cfg).resolve()

    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg_path}")
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")

    # Resolve inputs
    meta_dir, repo_name = _resolve_dataset_meta_dir(data_cfg_path)
    output_dir = _load_train_output_dir(train_cfg_path)
    pm_dir = _resolve_pretrained_model_dir(output_dir, args.snapshot)

    # Use output_dir.name as the exported run/model name
    run_name = output_dir.name

    # Prepare staging directory in /tmp
    tmp_root = Path(tempfile.mkdtemp(prefix="lerobot_export_", dir="/tmp"))
    staging_dir = tmp_root / run_name
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset meta as folder
    shutil.copytree(meta_dir, staging_dir / "meta", dirs_exist_ok=True)

    # Copy pretrained_model contents into staging root
    _copy_pretrained_contents(pm_dir, staging_dir)

    # Create run_name.zip inside the temp root
    zip_base = tmp_root / run_name
    archive_path = shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)

    print(f"Created zip: {archive_path}")
    print(f"Staging dir: {staging_dir}")


if __name__ == "__main__":
    main()
