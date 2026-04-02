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
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import yaml


DEFAULT_SERVICE = "lerobot-user"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("fr3_record_config.yaml")
CONTAINER_WORKSPACE = "/workspace"
LEGACY_CONTAINER_WORKSPACE = "/lerobot"
DEFAULT_RUNTIME = "auto"
_HOST_MVS_PYTHON_PATHS = (
    Path("/opt/MVS/Samples/64/Python"),
    Path("/opt/MVS/Samples/32/Python"),
)
_HOST_MVS_LIBRARY_PATHS = (
    Path("/opt/MVS/lib/64"),
    Path("/opt/MVS/lib"),
)
_DEFAULT_HOST_GEN_CON_SDK_ROOTS = (
    Path("/opt/dependencies/gen_con_sdk_python_release"),
    Path(__file__).resolve().parents[2] / ".." / "HIROLRobotPlatform" / "dependencies" / "gen_con_sdk_python_release",
)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Docker-based FR3 teleoperation recording.")
    parser.add_argument(
        "--runtime",
        choices=("auto", "docker", "host"),
        default=DEFAULT_RUNTIME,
        help=(
            "Runtime to use. 'auto' keeps Docker by default, but switches to host execution when the config "
            "contains Hikrobot GigE cameras that are not reachable from the current rootless Docker setup."
        ),
    )
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
    if path_str.startswith(f"{CONTAINER_WORKSPACE}/"):
        return path_str
    if path_str.startswith(f"{LEGACY_CONTAINER_WORKSPACE}/"):
        return f"{CONTAINER_WORKSPACE}/{path_str.removeprefix(f'{LEGACY_CONTAINER_WORKSPACE}/')}"

    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(
            f"Config path must live inside {resolved_workspace} or already be a {CONTAINER_WORKSPACE} path."
        ) from exc

    return f"{CONTAINER_WORKSPACE}/{relative.as_posix()}"


def _to_host_path(path: Path, workspace: Path) -> Path:
    path_str = str(path)
    if path_str.startswith(f"{CONTAINER_WORKSPACE}/"):
        return workspace.resolve() / Path(path_str.removeprefix(f"{CONTAINER_WORKSPACE}/"))
    if path_str.startswith(f"{LEGACY_CONTAINER_WORKSPACE}/"):
        return workspace.resolve() / Path(path_str.removeprefix(f"{LEGACY_CONTAINER_WORKSPACE}/"))
    return path.resolve()


def _normalize_workspace_path(path_value: str) -> str:
    if path_value.startswith(f"{LEGACY_CONTAINER_WORKSPACE}/"):
        return f"{CONTAINER_WORKSPACE}/{path_value.removeprefix(f'{LEGACY_CONTAINER_WORKSPACE}/')}"
    return path_value


def _container_path_to_host(path_value: str, workspace: Path) -> str:
    normalized = _normalize_workspace_path(path_value)
    if normalized.startswith(f"{CONTAINER_WORKSPACE}/"):
        relative = normalized.removeprefix(f"{CONTAINER_WORKSPACE}/")
        return str(workspace.resolve() / relative)
    return path_value


def _translate_workspace_bound_value_for_host(value: Any, workspace: Path) -> Any:
    if isinstance(value, str):
        return _container_path_to_host(value, workspace)
    if isinstance(value, list):
        return [_translate_workspace_bound_value_for_host(item, workspace) for item in value]
    if isinstance(value, dict):
        return {key: _translate_workspace_bound_value_for_host(item, workspace) for key, item in value.items()}
    return value


def _load_yaml_config(config_path: Path, workspace: Path) -> dict[str, Any]:
    host_config_path = _to_host_path(config_path, workspace)
    payload = yaml.safe_load(host_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping-style YAML config in {host_config_path}, got {type(payload).__name__}.")
    return payload


def _iter_hikrobot_camera_configs(config_payload: dict[str, Any]) -> list[dict[str, Any]]:
    robot_cfg = config_payload.get("robot")
    if not isinstance(robot_cfg, dict):
        return []
    cameras_cfg = robot_cfg.get("cameras")
    if not isinstance(cameras_cfg, dict):
        return []

    hikrobot_cameras: list[dict[str, Any]] = []
    for camera_cfg in cameras_cfg.values():
        if not isinstance(camera_cfg, dict):
            continue
        if str(camera_cfg.get("type", "")).lower() != "hikrobot":
            continue
        hikrobot_cameras.append(camera_cfg)
    return hikrobot_cameras


def determine_runtime(args: argparse.Namespace) -> str:
    if args.runtime != "auto":
        return args.runtime

    config_payload = _load_yaml_config(args.config_path, args.workspace.resolve())
    for camera_cfg in _iter_hikrobot_camera_configs(config_payload):
        if str(camera_cfg.get("transport_layer", "usb")).lower() == "gige":
            return "host"
    return "docker"


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


def _extras_override_value(extras: list[str], prefix: str) -> str | None:
    for idx, extra in enumerate(extras):
        if extra.startswith(f"{prefix}="):
            return extra.split("=", 1)[1]
        if extra == prefix and idx + 1 < len(extras):
            return extras[idx + 1]
    return None


def _build_timestamped_dataset_root(
    args: argparse.Namespace,
    extras: list[str],
    workspace: Path,
    *,
    runtime: str,
) -> str | None:
    if args.dataset_root is not None or args.resume:
        return None
    if _extras_override_flag(extras, "--dataset.root") or _extras_override_flag(extras, "--resume"):
        return None

    dataset_root, repo_id = _load_config_dataset_defaults(args.config_path, workspace)
    if dataset_root is None:
        dataset_name = repo_id.split("/")[-1] if repo_id else "fr3_recording"
        dataset_root = (
            f"{CONTAINER_WORKSPACE}/outputs/datasets/{dataset_name}"
            if runtime == "docker"
            else str(workspace.resolve() / "outputs" / "datasets" / dataset_name)
        )

    dataset_root = _normalize_workspace_path(dataset_root)
    if runtime == "host":
        dataset_root = _container_path_to_host(dataset_root, workspace)

    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(dataset_root)
    return str(base_path.parent / f"{base_path.name}_{timestamp}")


def resolve_dataset_root(
    args: argparse.Namespace,
    extras: list[str] | None = None,
    *,
    runtime: str = "docker",
) -> str | None:
    extra_args = extras or []
    workspace = args.workspace.resolve()
    if args.dataset_root is not None:
        dataset_root = _normalize_workspace_path(args.dataset_root)
        return _container_path_to_host(dataset_root, workspace) if runtime == "host" else dataset_root

    extra_dataset_root = _extras_override_value(extra_args, "--dataset.root")
    if extra_dataset_root is not None:
        dataset_root = _normalize_workspace_path(extra_dataset_root)
        return _container_path_to_host(dataset_root, workspace) if runtime == "host" else dataset_root

    return _build_timestamped_dataset_root(args, extra_args, workspace, runtime=runtime)


def build_docker_command(args: argparse.Namespace, extras: list[str] | None = None) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    config_path = _to_container_path(args.config_path, workspace)
    extra_args = extras or []
    record_args = [
        "cd /workspace &&",
        "PYTHONPATH=/workspace/src",
        "/lerobot/.venv/bin/python",
        "-m",
        "tools.fr3.fr3_record_runtime",
        f"--config_path={config_path}",
    ]
    if args.repo_id is not None:
        record_args.append(f"--dataset.repo_id={args.repo_id}")
    dataset_root = resolve_dataset_root(args, extra_args, runtime="docker")
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


def _host_runtime_config_cache_path(args: argparse.Namespace, workspace: Path) -> Path:
    config_path = _to_host_path(args.config_path, workspace)
    cache_key = hashlib.sha256(str(config_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "lerobot_fr3_record" / f"{config_path.stem}.{cache_key}.host.yaml"


def prepare_host_runtime_config(args: argparse.Namespace) -> Path:
    workspace = args.workspace.resolve()
    config_payload = _load_yaml_config(args.config_path, workspace)
    translated_payload = _translate_workspace_bound_value_for_host(config_payload, workspace)
    cache_path = _host_runtime_config_cache_path(args, workspace)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(yaml.safe_dump(translated_payload, sort_keys=False), encoding="utf-8")
    return cache_path


def build_host_command(
    args: argparse.Namespace,
    extras: list[str] | None = None,
    *,
    config_path: Path | None = None,
) -> list[str]:
    workspace = args.workspace.resolve()
    runtime_config_path = config_path if config_path is not None else _to_host_path(args.config_path, workspace)
    extra_args = extras or []
    host_python = _host_python_executable(workspace)
    record_args = [
        str(host_python),
        "-m",
        "tools.fr3.fr3_record_runtime",
        f"--config_path={runtime_config_path}",
    ]
    if args.repo_id is not None:
        record_args.append(f"--dataset.repo_id={args.repo_id}")
    dataset_root = resolve_dataset_root(args, extra_args, runtime="host")
    if dataset_root is not None:
        record_args.append(f"--dataset.root={dataset_root}")
    if args.task is not None:
        record_args.append(f"--dataset.single_task={args.task}")
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
        record_args.extend(extra_args)
    return record_args


def _host_python_executable(workspace: Path) -> Path:
    candidate = workspace.resolve() / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _prepend_env_paths(env: dict[str, str], key: str, values: list[str]) -> None:
    existing = [entry for entry in env.get(key, "").split(os.pathsep) if entry]
    combined: list[str] = []
    for value in values + existing:
        if not value or value in combined:
            continue
        combined.append(value)
    if combined:
        env[key] = os.pathsep.join(combined)


def _discover_cmeel_prefix_lib(workspace: Path) -> str | None:
    venv_lib = workspace.resolve() / ".venv" / "lib"
    if not venv_lib.exists():
        return None
    matches = sorted(venv_lib.glob("python*/site-packages/cmeel.prefix/lib"))
    if not matches:
        return None
    return str(matches[0])


def _discover_host_gen_con_sdk_root() -> str | None:
    for root in _DEFAULT_HOST_GEN_CON_SDK_ROOTS:
        expanded = root.expanduser()
        if expanded.exists():
            return str(expanded.resolve())
    return None


def build_host_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str((workspace.resolve() / "src"))
    pythonpath_entries = [src_path]
    pythonpath_entries.extend(str(path) for path in _HOST_MVS_PYTHON_PATHS if path.exists())

    gen_con_sdk_root = _discover_host_gen_con_sdk_root()
    if gen_con_sdk_root is not None:
        env.setdefault("GEN_CON_SDK_HOME", gen_con_sdk_root)
        pythonpath_entries.append(str(Path(gen_con_sdk_root).resolve().parent))

    _prepend_env_paths(env, "PYTHONPATH", pythonpath_entries)

    ld_library_entries = [str(path) for path in _HOST_MVS_LIBRARY_PATHS if path.exists()]
    cmeel_prefix_lib = _discover_cmeel_prefix_lib(workspace)
    if cmeel_prefix_lib is not None:
        ld_library_entries.insert(0, cmeel_prefix_lib)
    if Path("/usr/local/lib").exists():
        ld_library_entries.append("/usr/local/lib")
    _prepend_env_paths(env, "LD_LIBRARY_PATH", ld_library_entries)

    if Path("/opt/MVS").exists():
        env.setdefault("HIKROBOT_MVS_HOME", "/opt/MVS")
    if Path("/opt/MVS/lib").exists():
        env.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
    return env


def format_host_command_for_display(command: list[str], *, workspace: Path) -> str:
    env = build_host_env(workspace)
    pythonpath = env["PYTHONPATH"]
    exports = [f"PYTHONPATH={shlex.quote(pythonpath)}"]
    ld_library_path = env.get("LD_LIBRARY_PATH")
    if ld_library_path:
        exports.append(f"LD_LIBRARY_PATH={shlex.quote(ld_library_path)}")
    mvs_home = env.get("HIKROBOT_MVS_HOME")
    if mvs_home:
        exports.append(f"HIKROBOT_MVS_HOME={shlex.quote(mvs_home)}")
    mvs_runenv = env.get("MVCAM_COMMON_RUNENV")
    if mvs_runenv:
        exports.append(f"MVCAM_COMMON_RUNENV={shlex.quote(mvs_runenv)}")
    return f"cd {shlex.quote(str(workspace.resolve()))} && {' '.join(exports)} {shlex.join(command)}"


def build_chown_command(args: argparse.Namespace, dataset_root: str, *, uid: int, gid: int) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file is not None else workspace / "docker" / "docker-compose.yml"
    quoted_dataset_root = shlex.quote(dataset_root)
    runtime_args = [
        "cd /workspace &&",
        f"chown -R {uid}:{gid} {quoted_dataset_root}",
    ]
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
        " ".join(runtime_args),
    ]


def main(argv: list[str] | None = None) -> int:
    args, extras = parse_args(argv)
    runtime = determine_runtime(args)

    if runtime == "host":
        workspace = args.workspace.resolve()
        runtime_config_path = prepare_host_runtime_config(args)
        dataset_root = resolve_dataset_root(args, extras, runtime="host")
        command = build_host_command(args, extras, config_path=runtime_config_path)
        if args.dry_run:
            print(format_host_command_for_display(command, workspace=workspace))
            return 0
        completed = subprocess.run(command, check=False, cwd=workspace, env=build_host_env(workspace))
        return completed.returncode

    dataset_root = resolve_dataset_root(args, extras, runtime="docker")
    command = build_docker_command(args, extras)
    if args.dry_run:
        print(shlex.join(command))
        if dataset_root is not None:
            print(shlex.join(build_chown_command(args, dataset_root, uid=os.getuid(), gid=os.getgid())))
        return 0

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or dataset_root is None:
        return completed.returncode

    ownership_fix = subprocess.run(
        build_chown_command(args, dataset_root, uid=os.getuid(), gid=os.getgid()),
        check=False,
    )
    return ownership_fix.returncode


if __name__ == "__main__":
    raise SystemExit(main())
