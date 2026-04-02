#!/usr/bin/env python

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

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from tools.fr3 import fr3_record


def test_build_docker_command_uses_default_config_inside_workspace(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset:\n"
        "  repo_id: hph/fr3_pick_place_v1\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n"
        "  video: true\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
        ]
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 15, 1, 2)

    command = None
    original_now = fr3_record._now
    fr3_record._now = _FrozenDatetime.now
    try:
        command = fr3_record.build_docker_command(args, extras)
    finally:
        fr3_record._now = original_now
    command_text = " ".join(command)

    assert command[:6] == [
        "docker",
        "compose",
        "-f",
        str((tmp_path / "docker" / "docker-compose.yml").resolve()),
        "run",
        "--rm",
    ]
    assert "lerobot-user" in command
    assert "cd /workspace &&" in command_text
    assert "PYTHONPATH=/workspace/src" in command_text
    assert "/lerobot/.venv/bin/python -m tools.fr3.fr3_record_runtime" in command_text
    assert "--config_path=/workspace/tools/fr3/fr3_record_config.yaml" in command_text
    assert "--dataset.root=/workspace/outputs/datasets/fr3_pick_place_v1_20260313_150102" in command_text


def test_build_docker_command_accepts_common_dataset_overrides():
    args, extras = fr3_record.parse_args(
        [
            "--repo-id",
            "hph/fr3_pick_place_v2",
            "--dataset-root",
            "/workspace/outputs/datasets/fr3_pick_place_v2",
            "--task",
            "Pick and place",
            "--control-fps",
            "200",
            "--num-episodes",
            "12",
            "--episode-time-s",
            "30",
            "--reset-time-s",
            "15",
            "--resume",
            "--video=false",
        ]
    )

    command = fr3_record.build_docker_command(args, extras)
    command_text = " ".join(command)

    assert "--dataset.repo_id=hph/fr3_pick_place_v2" in command_text
    assert "--dataset.root=/workspace/outputs/datasets/fr3_pick_place_v2" in command_text
    assert "--dataset.single_task='Pick and place'" in command_text
    assert "--control_fps=200" in command_text
    assert "--dataset.num_episodes=12" in command_text
    assert "--dataset.episode_time_s=30.0" in command_text
    assert "--dataset.reset_time_s=15.0" in command_text
    assert "--resume=true" in command_text
    assert "--video=false" in command_text


def test_build_docker_command_does_not_timestamp_dataset_root_when_resuming(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset:\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--resume",
        ]
    )

    command = fr3_record.build_docker_command(args, extras)
    command_text = " ".join(command)

    assert "--resume=true" in command_text
    assert "--dataset.root=" not in command_text


def test_build_docker_command_does_not_timestamp_dataset_root_when_extra_override_is_present(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset:\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--dataset.root=/workspace/outputs/datasets/manual_override",
        ]
    )

    command = fr3_record.build_docker_command(args, extras)
    command_text = " ".join(command)

    assert "--dataset.root=/workspace/outputs/datasets/manual_override" in command_text
    assert "fr3_pick_place_v1_" not in command_text


def test_resolve_dataset_root_accepts_split_extra_override():
    args, extras = fr3_record.parse_args(
        [
            "--dataset.root",
            "/workspace/outputs/datasets/manual_override",
        ]
    )

    dataset_root = fr3_record.resolve_dataset_root(args, extras)

    assert dataset_root == "/workspace/outputs/datasets/manual_override"


def test_build_docker_command_rejects_host_config_outside_workspace(tmp_path: Path):
    outside_config = tmp_path.parent / "fr3_record_config.yaml"
    outside_config.write_text("dataset:\n  video: true\n", encoding="utf-8")
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(outside_config),
        ]
    )

    with pytest.raises(ValueError):
        fr3_record.build_docker_command(args, extras)


def test_determine_runtime_switches_to_host_for_hikrobot_gige(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "robot:\n"
        "  type: franka_research3\n"
        "  cameras:\n"
        "    side:\n"
        "      type: hikrobot\n"
        "      serial: DA123\n"
        "      transport_layer: gige\n",
        encoding="utf-8",
    )
    args, _ = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
        ]
    )

    assert fr3_record.determine_runtime(args) == "host"


def test_prepare_host_runtime_config_translates_workspace_bound_paths(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "robot:\n"
        "  type: franka_research3\n"
        "  urdf_path: /lerobot/src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.urdf\n"
        "dataset:\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, _ = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--runtime",
            "host",
        ]
    )

    translated_config_path = fr3_record.prepare_host_runtime_config(args)
    translated = yaml.safe_load(translated_config_path.read_text(encoding="utf-8"))

    assert translated["robot"]["urdf_path"] == str(
        tmp_path / "src" / "lerobot" / "robots" / "franka_research3" / "assets" / "franka_fr3" / "fr3_pika_gripper_ati.urdf"
    )
    assert translated["dataset"]["root"] == str(tmp_path / "outputs" / "datasets" / "fr3_pick_place_v1")


def test_resolve_dataset_root_maps_container_path_for_host_runtime(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset:\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--runtime",
            "host",
        ]
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 15, 1, 2)

    original_now = fr3_record._now
    fr3_record._now = _FrozenDatetime.now
    try:
        dataset_root = fr3_record.resolve_dataset_root(args, extras, runtime="host")
    finally:
        fr3_record._now = original_now

    assert dataset_root == str(tmp_path / "outputs" / "datasets" / "fr3_pick_place_v1_20260313_150102")


def test_build_host_command_uses_runtime_config_and_host_paths(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    host_python = tmp_path / ".venv" / "bin" / "python"
    host_python.parent.mkdir(parents=True)
    host_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    config_path.write_text(
        "dataset:\n"
        "  root: /workspace/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--runtime",
            "host",
        ]
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 15, 1, 2)

    original_now = fr3_record._now
    fr3_record._now = _FrozenDatetime.now
    try:
        runtime_config_path = fr3_record.prepare_host_runtime_config(args)
        command = fr3_record.build_host_command(args, extras, config_path=runtime_config_path)
    finally:
        fr3_record._now = original_now

    command_text = " ".join(command)
    assert command[:3] == [str(tmp_path / ".venv" / "bin" / "python"), "-m", "tools.fr3.fr3_record_runtime"]
    assert f"--config_path={runtime_config_path}" in command_text
    assert f"--dataset.root={tmp_path / 'outputs' / 'datasets' / 'fr3_pick_place_v1_20260313_150102'}" in command_text


def test_build_host_env_adds_mvs_cmeel_and_gencon_paths(tmp_path: Path, monkeypatch):
    (tmp_path / "src").mkdir()
    cmeel_lib = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages" / "cmeel.prefix" / "lib"
    cmeel_lib.mkdir(parents=True)
    mvs64 = tmp_path / "mvs64"
    mvs32 = tmp_path / "mvs32"
    mvslib64 = tmp_path / "mvslib64"
    mvslib = tmp_path / "mvslib"
    gen_con_root = tmp_path / "gen_con_sdk_python_release"
    for path in (mvs64, mvs32, mvslib64, mvslib, gen_con_root):
        path.mkdir(parents=True)

    monkeypatch.setattr(fr3_record, "_HOST_MVS_PYTHON_PATHS", (mvs64, mvs32))
    monkeypatch.setattr(fr3_record, "_HOST_MVS_LIBRARY_PATHS", (mvslib64, mvslib))
    monkeypatch.setattr(fr3_record, "_DEFAULT_HOST_GEN_CON_SDK_ROOTS", (gen_con_root,))
    monkeypatch.setenv("PYTHONPATH", "existing_py")
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing_ld")

    env = fr3_record.build_host_env(tmp_path)

    pythonpath_entries = env["PYTHONPATH"].split(fr3_record.os.pathsep)
    assert pythonpath_entries[:4] == [
        str(tmp_path / "src"),
        str(mvs64),
        str(mvs32),
        str(gen_con_root.parent),
    ]
    assert "existing_py" in pythonpath_entries

    ld_entries = env["LD_LIBRARY_PATH"].split(fr3_record.os.pathsep)
    assert ld_entries[:4] == [
        str(cmeel_lib),
        str(mvslib64),
        str(mvslib),
        "/usr/local/lib",
    ]
    assert "existing_ld" in ld_entries
    assert env["GEN_CON_SDK_HOME"] == str(gen_con_root.resolve())


def test_main_dry_run_prints_command(capsys):
    original_getuid = fr3_record.os.getuid
    original_getgid = fr3_record.os.getgid

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 15, 1, 2)

    original_now = fr3_record._now
    fr3_record._now = _FrozenDatetime.now
    fr3_record.os.getuid = lambda: 1000
    fr3_record.os.getgid = lambda: 1001
    try:
        exit_code = fr3_record.main(["--dry-run"])
    finally:
        fr3_record._now = original_now
        fr3_record.os.getuid = original_getuid
        fr3_record.os.getgid = original_getgid

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "lerobot-user" in captured.out
    assert "--config_path=/workspace/tools/fr3/fr3_record_config.yaml" in captured.out
    assert "--dataset.root=/workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in captured.out
    assert "chown -R 1000:1001 /workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=9)

    monkeypatch.setattr(fr3_record.subprocess, "run", fake_run)

    exit_code = fr3_record.main([])

    assert exit_code == 9
    assert calls


def test_main_runs_ownership_fix_after_success(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(fr3_record.subprocess, "run", fake_run)
    monkeypatch.setattr(fr3_record.os, "getuid", lambda: 1000)
    monkeypatch.setattr(fr3_record.os, "getgid", lambda: 1001)

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 3, 13, 15, 1, 2)

    original_now = fr3_record._now
    fr3_record._now = _FrozenDatetime.now
    try:
        exit_code = fr3_record.main([])
    finally:
        fr3_record._now = original_now

    assert exit_code == 0
    assert len(calls) == 2
    assert "--dataset.root=/workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in " ".join(calls[0])
    assert "chown -R 1000:1001 /workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in " ".join(calls[1])


def test_main_skips_ownership_fix_when_dataset_root_is_unknown_on_resume(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(fr3_record.subprocess, "run", fake_run)

    exit_code = fr3_record.main(["--resume"])

    assert exit_code == 0
    assert len(calls) == 1
