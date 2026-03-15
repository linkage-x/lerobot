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

from tools.fr3 import fr3_record


def test_build_docker_command_uses_default_config_inside_workspace(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset:\n"
        "  repo_id: hph/fr3_pick_place_v1\n"
        "  root: /lerobot/outputs/datasets/fr3_pick_place_v1\n"
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
    assert "cd /lerobot &&" in command_text
    assert "PYTHONPATH=/lerobot/src" in command_text
    assert "/lerobot/.venv/bin/python -m tools.fr3.fr3_record_runtime" in command_text
    assert "--config_path=/lerobot/tools/fr3/fr3_record_config.yaml" in command_text
    assert "--dataset.root=/lerobot/outputs/datasets/fr3_pick_place_v1_20260313_150102" in command_text


def test_build_docker_command_accepts_common_dataset_overrides():
    args, extras = fr3_record.parse_args(
        [
            "--repo-id",
            "hph/fr3_pick_place_v2",
            "--dataset-root",
            "/lerobot/outputs/datasets/fr3_pick_place_v2",
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
    assert "--dataset.root=/lerobot/outputs/datasets/fr3_pick_place_v2" in command_text
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
        "  root: /lerobot/outputs/datasets/fr3_pick_place_v1\n",
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
        "  root: /lerobot/outputs/datasets/fr3_pick_place_v1\n",
        encoding="utf-8",
    )
    args, extras = fr3_record.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--dataset.root=/lerobot/outputs/datasets/manual_override",
        ]
    )

    command = fr3_record.build_docker_command(args, extras)
    command_text = " ".join(command)

    assert "--dataset.root=/lerobot/outputs/datasets/manual_override" in command_text
    assert "fr3_pick_place_v1_" not in command_text


def test_resolve_dataset_root_accepts_split_extra_override():
    args, extras = fr3_record.parse_args(
        [
            "--dataset.root",
            "/lerobot/outputs/datasets/manual_override",
        ]
    )

    dataset_root = fr3_record.resolve_dataset_root(args, extras)

    assert dataset_root == "/lerobot/outputs/datasets/manual_override"


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
    assert "--config_path=/lerobot/tools/fr3/fr3_record_config.yaml" in captured.out
    assert "--dataset.root=/lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in captured.out
    assert "chown -R 1000:1001 /lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in captured.out


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
    assert "--dataset.root=/lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in " ".join(calls[0])
    assert "chown -R 1000:1001 /lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_150102" in " ".join(calls[1])


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
