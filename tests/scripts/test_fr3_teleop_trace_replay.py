#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

from tools.fr3 import fr3_teleop_trace_replay


def test_build_docker_command_defaults_to_sim_service(tmp_path: Path):
    args = fr3_teleop_trace_replay.parse_args(["--mode", "sim", "--workspace", str(tmp_path)])

    command = fr3_teleop_trace_replay.build_docker_command(args)
    command_text = " ".join(command)

    assert command[:6] == [
        "docker",
        "compose",
        "-f",
        str((tmp_path / "docker" / "docker-compose.yml").resolve()),
        "run",
        "--rm",
    ]
    assert "lerobot-fr3-sim" in command
    assert "--mode=sim" in command_text
    assert "tools/fr3/fr3_teleop_trace_replay_runtime.py" in command_text
    assert "--output=/lerobot/outputs/fr3_traces/sim_trace.json" in command_text


def test_build_docker_command_uses_hardware_service_and_robot_flags(tmp_path: Path):
    args = fr3_teleop_trace_replay.parse_args(
        ["--mode", "hardware", "--workspace", str(tmp_path), "--robot-ip", "192.168.1.206"]
    )

    command = fr3_teleop_trace_replay.build_docker_command(args)
    command_text = " ".join(command)

    assert "lerobot-user" in command
    assert "--mode=hardware" in command_text
    assert "--robot-ip=192.168.1.206" in command_text
    assert "--gripper-port=/dev/ttyUSB80" in command_text


def test_build_docker_command_passes_combined_trace_profile_and_rotation_steps(tmp_path: Path):
    args = fr3_teleop_trace_replay.parse_args(
        [
            "--mode",
            "sim",
            "--workspace",
            str(tmp_path),
            "--trace-profile",
            "combined",
            "--step-wx",
            "0.0003",
        ]
    )

    command = fr3_teleop_trace_replay.build_docker_command(args)
    command_text = " ".join(command)

    assert "--trace-profile=combined" in command_text
    assert "--step-wx=0.0003" in command_text
    assert "--step-wy=0.0002" in command_text
    assert "--step-wz=0.0002" in command_text


def test_build_docker_command_passes_wz_trace_profile(tmp_path: Path):
    args = fr3_teleop_trace_replay.parse_args(
        [
            "--mode",
            "hardware",
            "--workspace",
            str(tmp_path),
            "--trace-profile",
            "wz",
            "--step-wz",
            "0.00025",
        ]
    )

    command = fr3_teleop_trace_replay.build_docker_command(args)
    command_text = " ".join(command)

    assert "--trace-profile=wz" in command_text
    assert "--step-wz=0.00025" in command_text
    assert "--mode=hardware" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_teleop_trace_replay.main(["--mode", "sim", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "lerobot-fr3-sim" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=7)

    monkeypatch.setattr(fr3_teleop_trace_replay.subprocess, "run", fake_run)

    exit_code = fr3_teleop_trace_replay.main(["--mode", "hardware"])

    assert exit_code == 7
    assert calls
