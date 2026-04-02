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

from tools.fr3 import fr3_move_to_start


def test_build_docker_command_contains_expected_runtime_entrypoint(tmp_path: Path):
    args = fr3_move_to_start.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--robot-ip",
            "10.0.0.5",
            "--service",
            "lerobot-internal",
        ]
    )

    command = fr3_move_to_start.build_docker_command(args)
    command_text = " ".join(command)

    assert command[:6] == [
        "docker",
        "compose",
        "-f",
        str((tmp_path / "docker" / "docker-compose.yml").resolve()),
        "run",
        "--rm",
    ]
    assert "lerobot-internal" in command
    assert "cd /workspace &&" in command_text
    assert "PYTHONPATH=/workspace/src" in command_text
    assert "tools/fr3/fr3_move_to_start_runtime.py" in command_text
    assert "--robot-ip=10.0.0.5" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_move_to_start.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "tools/fr3/fr3_move_to_start_runtime.py" in captured.out
    assert "lerobot-user" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=9)

    monkeypatch.setattr(fr3_move_to_start.subprocess, "run", fake_run)

    exit_code = fr3_move_to_start.main([])

    assert exit_code == 9
    assert calls
