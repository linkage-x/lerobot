#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run FR3 teleoperation recording inside the standard Docker environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess


DEFAULT_SERVICE = "lerobot-user"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("fr3_record_config.yaml")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Docker-based FR3 teleoperation recording.")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Docker compose service to run.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to mount into the container.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose file to use. Defaults to <workspace>/docker/docker-compose.yml.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Host path to a lerobot-record YAML config inside the repository.",
    )
    parser.add_argument("--repo-id", default=None, help="Optional dataset repo id override.")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--task", default=None, help="Optional task description override.")
    parser.add_argument("--control-fps", type=int, default=None, help="Optional robot control-loop FPS override.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Optional episode count override.")
    parser.add_argument("--episode-time-s", type=float, default=None, help="Optional episode duration override.")
    parser.add_argument("--reset-time-s", type=float, default=None, help="Optional reset duration override.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset recording.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    args, extras = parser.parse_known_args(argv)
    return args, extras


def _now() -> datetime:
    return datetime.now()


def _to_container_path(path: Path, workspace: Path) -> str:
    path_str = str(path)
    if path_str.startswith("/lerobot/"):
        return path_str

    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(f"Config path must live inside {resolved_workspace} or already be a /lerobot path.") from exc

    return f"/lerobot/{relative.as_posix()}"


def _to_host_path(path: Path, workspace: Path) -> Path:
    path_str = str(path)
    if path_str.startswith("/lerobot/"):
        return workspace.resolve() / Path(path_str.removeprefix("/lerobot/"))
    return path.resolve()


def _load_config_dataset_defaults(config_path: Path, workspace: Path) -> tuple[str | None, str | None]:
    host_config_path = _to_host_path(config_path, workspace)
    dataset_root = None
    repo_id = None
    current_section = None

    for raw_line in host_config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        is_top_level = raw_line[:1].strip() != "" or not raw_line.startswith(" ")
        if is_top_level and stripped.endswith(":"):
            current_section = stripped[:-1]
            continue
        if current_section != "dataset":
            continue
        if not raw_line.startswith("  "):
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        normalized_value = value.strip().strip("\"'")
        if key == "root" and normalized_value:
            dataset_root = normalized_value
        elif key == "repo_id" and normalized_value:
            repo_id = normalized_value

    return dataset_root, repo_id


def _extras_override_flag(extras: list[str], prefix: str) -> bool:
    return any(extra == prefix or extra.startswith(f"{prefix}=") for extra in extras)


def _build_timestamped_dataset_root(args: argparse.Namespace, extras: list[str], workspace: Path) -> str | None:
    if args.dataset_root is not None or args.resume:
        return None
    if _extras_override_flag(extras, "--dataset.root") or _extras_override_flag(extras, "--resume"):
        return None

    dataset_root, repo_id = _load_config_dataset_defaults(args.config_path, workspace)
    if dataset_root is None:
        dataset_name = repo_id.split("/")[-1] if repo_id else "fr3_recording"
        dataset_root = f"/lerobot/outputs/datasets/{dataset_name}"

    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(dataset_root)
    return str(base_path.parent / f"{base_path.name}_{timestamp}")


def build_docker_command(args: argparse.Namespace, extras: list[str] | None = None) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    config_path = _to_container_path(args.config_path, workspace)
    extra_args = extras or []
    record_args = [
        "cd /lerobot &&",
        "PYTHONPATH=/lerobot/src",
        "/lerobot/.venv/bin/lerobot-record",
        f"--config_path={config_path}",
    ]
    if args.repo_id is not None:
        record_args.append(f"--dataset.repo_id={args.repo_id}")
    dataset_root = args.dataset_root or _build_timestamped_dataset_root(args, extra_args, workspace)
    if dataset_root is not None:
        record_args.append(f"--dataset.root={dataset_root}")
    if args.task is not None:
        record_args.append(f"--dataset.single_task={shlex.quote(args.task)}")
    if args.control_fps is not None:
        record_args.append(f"--control_fps={args.control_fps}")
    if args.num_episodes is not None:
        record_args.append(f"--dataset.num_episodes={args.num_episodes}")
    if args.episode_time_s is not None:
        record_args.append(f"--dataset.episode_time_s={args.episode_time_s}")
    if args.reset_time_s is not None:
        record_args.append(f"--dataset.reset_time_s={args.reset_time_s}")
    if args.resume:
        record_args.append("--resume=true")
    if extra_args:
        record_args.extend(shlex.quote(extra) for extra in extra_args)

    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        args.service,
        "bash",
        "-lc",
        " ".join(record_args),
    ]


def main(argv: list[str] | None = None) -> int:
    args, extras = parse_args(argv)
    command = build_docker_command(args, extras)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
