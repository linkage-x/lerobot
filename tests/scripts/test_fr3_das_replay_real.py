#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

from tools.fr3 import fr3_das_replay_real


def test_build_docker_command_defaults_to_hardware_visible_service(tmp_path: Path):
    args = fr3_das_replay_real.parse_args(["--workspace", str(tmp_path)])

    command = fr3_das_replay_real.build_docker_command(args)
    command_text = " ".join(command)

    assert command[:6] == [
        "docker",
        "compose",
        "-f",
        str((tmp_path / "docker" / "docker-compose.yml").resolve()),
        "run",
        "--rm",
    ]
    assert "lerobot-fr3-sim-teleop" in command
    assert "tools/fr3/fr3_das_replay_real_runtime.py" in command_text
    assert "--gripper-port=/dev/ttyUSB0" in command_text
    assert "--gripper-backend=das" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_das_replay_real.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "lerobot-fr3-sim-teleop" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=7)

    monkeypatch.setattr(fr3_das_replay_real.subprocess, "run", fake_run)

    exit_code = fr3_das_replay_real.main([])

    assert exit_code == 7
    assert calls
