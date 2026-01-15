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
    """Resolve dataset meta directory from data config.

    Supports two schemas:
      1) DAS-style:
            output:
              task_dir: "/abs/or/relative/path"
              repo_name: "name"  # optional
         In this case, dataset root = task_dir (relative to hirol dir if not absolute),
         and meta dir = <dataset_root>/meta.

      2) Legacy HIROL-style (top-level keys):
            root_path: "../assets/data" | "/abs/path"
            repo_name: "name"
         In this case, dataset root = root_path/repo_name (root_path relative to hirol dir if not absolute),
         and meta dir = <dataset_root>/meta.
    """
    with open(data_cfg_path, "r", encoding="utf-8") as f:
        dc = yaml.safe_load(f)

    hirol_dir = data_cfg_path.parent.parent.resolve()  # .../datasets/hirol

    # Prefer DAS-style output.task_dir
    out = dc.get("output") if isinstance(dc, dict) else None
    if isinstance(out, dict):
        task_dir = out.get("task_dir")
        if task_dir:
            td = Path(str(task_dir)).expanduser()
            dataset_root = td if td.is_absolute() else (hirol_dir / td)
            meta_dir = (dataset_root / "meta").resolve()
            repo_name = str(out.get("repo_name") or dataset_root.name)
            if not meta_dir.exists():
                raise FileNotFoundError(f"Dataset meta directory not found: {meta_dir}")
            return meta_dir, repo_name
        # Fallback to nested root_path/repo_name if present
        if "root_path" in out and "repo_name" in out:
            rp = Path(str(out["root_path"]))
            rp = rp.expanduser()
            repo_name = str(out["repo_name"])  # must exist
            dataset_root = rp if rp.is_absolute() else (hirol_dir / rp)
            meta_dir = (dataset_root / repo_name / "meta").resolve()
            if not meta_dir.exists():
                raise FileNotFoundError(f"Dataset meta directory not found: {meta_dir}")
            return meta_dir, repo_name

    # Legacy top-level schema
    try:
        root_path = dc["root_path"]
        repo_name = dc["repo_name"]
    except Exception as e:
        raise KeyError(
            f"Unsupported data config schema in '{data_cfg_path}'. Expected either top-level '\n"
            f"'root_path'/'repo_name' or nested 'output.task_dir'. Error: {e}"
        )

    rp = Path(str(root_path)).expanduser()
    dataset_root = rp if rp.is_absolute() else (hirol_dir / rp)
    meta_dir = (dataset_root / repo_name / "meta").resolve()
    if not meta_dir.exists():
        raise FileNotFoundError(f"Dataset meta directory not found: {meta_dir}")
    return meta_dir, str(repo_name)


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
    rerun_candidates = (
        list(train_cfg_path.parent.glob(f"{p.name}_rerun_*")) if not p.is_absolute() else []
    )
    hint = ""
    if rerun_candidates:
        rerun_list = "\n".join(str(c.resolve()) for c in sorted(rerun_candidates))
        hint = f"\nFound rerun dirs:\n{rerun_list}"
    raise FileNotFoundError(
        f"Could not resolve output_dir '{raw}'. Tried:\n{tried}{hint}"
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
        "-d",
        "--data_cfg",
        type=str,
        default="src/lerobot/datasets/hirol/config/insert_pinboard.yaml",
        help="Data conversion YAML path",
    )
    parser.add_argument(
        "-t",
        "--train_cfg",
        type=str,
        default="src/lerobot/scripts/train_config/act.json",
        help="Training config path (JSON or YAML)",
    )
    parser.add_argument(
        "-s",
        "--snapshot",
        type=str,
        default="last",
        help="Checkpoint snapshot name under 'checkpoints' (e.g. 'last', '000100')",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir from training config",
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
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _load_train_output_dir(train_cfg_path)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")
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
