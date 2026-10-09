#!/usr/bin/env python3
"""Resolve the P0 recorder camera set from current MAX96726 locks."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import yaml


def parse_sensor_ids(value: str) -> list[int]:
    values: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sensor_id = int(raw)
        if sensor_id < 0 or sensor_id > 15:
            raise ValueError(f"camera sensor id must be in [0, 15], got {sensor_id}")
        if sensor_id not in values:
            values.append(sensor_id)
    return values


def parse_locked_ids(output: str) -> list[int]:
    for line in output.splitlines():
        if line.startswith("LOCKED_VIDEO_IDS="):
            return parse_sensor_ids(line.split("=", 1)[1])
    raise RuntimeError("MAX96726 lock check did not emit LOCKED_VIDEO_IDS=")


def read_locked_ids(repo_root: Path) -> list[int]:
    script = repo_root / "tools/thor/gmsl2/check_max96726_locks.sh"
    completed = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode not in (0, 1):
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"MAX96726 lock check failed rc={completed.returncode}: {detail}")
    return parse_locked_ids(completed.stdout)


def build_config(template: Path, locked: list[int], excluded: list[int]) -> dict:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"recorder config root must be a mapping: {template}")
    excluded_set = set(excluded)
    selected = [sensor_id for sensor_id in locked if sensor_id not in excluded_set]
    if not selected:
        raise RuntimeError(
            f"no calibration cameras remain after exclusions; locked={locked}, excluded={excluded}"
        )
    cameras = payload.setdefault("sensors", {}).setdefault("cameras", {})
    cameras["detect_all"] = False
    cameras["sensor_ids"] = selected
    payload["p0_calibration_camera_selection"] = {
        "policy": "current_max96726_locks_minus_excluded_sensor_ids",
        "locked_sensor_ids": locked,
        "excluded_sensor_ids": excluded,
        "selected_sensor_ids": selected,
    }
    return payload


def write_yaml_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exclude-sensor-ids", default="1,4,10")
    parser.add_argument(
        "--locked-sensor-ids",
        default=None,
        help="Explicit locks for tests/diagnostics; normally detected from Thor hardware.",
    )
    args = parser.parse_args()

    excluded = parse_sensor_ids(args.exclude_sensor_ids)
    locked = (
        parse_sensor_ids(args.locked_sensor_ids)
        if args.locked_sensor_ids is not None
        else read_locked_ids(args.repo_root.resolve())
    )
    payload = build_config(args.template.resolve(), locked, excluded)
    selected = payload["p0_calibration_camera_selection"]["selected_sensor_ids"]
    write_yaml_atomic(args.output.resolve(), payload)
    print(f"[CAMERAS] locked={locked}")
    print(f"[CAMERAS] excluded={excluded}")
    print(f"[CAMERAS] selected={selected}")
    print(f"[CAMERAS] resolved recorder config: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
