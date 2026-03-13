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

from tools.fr3 import fr3_hardware_smoke


def test_build_docker_command_contains_expected_mounts_and_envs(tmp_path: Path):
    args = fr3_hardware_smoke.parse_args(
        [
            "--robot-ip",
            "10.0.0.5",
            "--workspace",
            str(tmp_path),
            "--skip-fr3-sdk",
        ]
    )

    command = fr3_hardware_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert command[:5] == ["docker", "run", "--rm", "--network", "host"]
    assert f"{tmp_path.resolve()}:/workspace" in command_text
    assert "FR3_SMOKE_ROBOT_IP=10.0.0.5" in command_text
    assert "FR3_SMOKE_SKIP_FR3_SDK=1" in command_text
    assert "/dev/input:/dev/input" in command_text
    assert "pika_gripper=mocked_skip" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_hardware_smoke.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker run" in captured.out
    assert "FR3_SMOKE_ROBOT_IP=192.168.1.206" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=7)

    monkeypatch.setattr(fr3_hardware_smoke.subprocess, "run", fake_run)

    exit_code = fr3_hardware_smoke.main([])

    assert exit_code == 7
    assert calls
