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

from tools.fr3 import fr3_teleop_smoke


def test_build_docker_command_contains_default_smoke_settings(tmp_path: Path):
    args = fr3_teleop_smoke.parse_args(["--workspace", str(tmp_path)])

    command = fr3_teleop_smoke.build_docker_command(args)
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
    assert "--fps=200" in command_text
    assert "--teleop_time_s=30.0" in command_text
    assert "--robot.robot_ip=192.168.1.206" in command_text
    assert f"--robot.gripper_port={fr3_teleop_smoke.DEFAULT_GRIPPER_PORT}" in command_text
    assert "--robot.allow_mock_gripper=true" in command_text
    assert f"--robot.urdf_path={fr3_teleop_smoke.DEFAULT_URDF_PATH}" in command_text
    assert "--teleop.tool_mode=binary" in command_text
    assert "--teleop.enable_rotation=false" in command_text
    assert f"--teleop.translation_scale={fr3_teleop_smoke.DEFAULT_TRANSLATION_SCALE}" in command_text
    assert f"--teleop.rotation_scale={fr3_teleop_smoke.DEFAULT_ROTATION_SCALE}" in command_text
    assert "--teleop.scale_x=" not in command_text
    assert f"--teleop.scale_wx={fr3_teleop_smoke.DEFAULT_SCALE_WX}" in command_text
    assert f"--teleop.scale_wy={fr3_teleop_smoke.DEFAULT_SCALE_WY}" in command_text
    assert "--teleop.scale_wz=" not in command_text
    assert f"--robot.max_target_delta_pos={fr3_teleop_smoke.DEFAULT_TRANSLATION_MAX_TARGET_DELTA_POS}" in command_text
    assert f"--robot.max_target_delta_rot={fr3_teleop_smoke.DEFAULT_TRANSLATION_MAX_TARGET_DELTA_ROT}" in command_text
    assert "cd /lerobot &&" in command_text
    assert "PYTHONPATH=/lerobot/src" in command_text


def test_incremental_mode_adds_incremental_flags():
    args = fr3_teleop_smoke.parse_args(["--tool-mode", "incremental"])

    command = fr3_teleop_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert "--teleop.tool_mode=incremental" in command_text
    assert f"--teleop.incremental_step={fr3_teleop_smoke.DEFAULT_INCREMENTAL_STEP}" in command_text
    assert f"--teleop.move_time={fr3_teleop_smoke.DEFAULT_MOVE_TIME}" in command_text


def test_require_gripper_disables_mock_gripper_fallback():
    args = fr3_teleop_smoke.parse_args(["--require-gripper"])

    command = fr3_teleop_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert "--robot.allow_mock_gripper=false" in command_text


def test_rotation_profile_enables_small_angle_rotation_smoke():
    args = fr3_teleop_smoke.parse_args(["--smoke-profile", "rotation"])

    command = fr3_teleop_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert "--teleop.enable_rotation=true" in command_text
    assert f"--teleop.translation_scale={fr3_teleop_smoke.DEFAULT_ROTATION_ONLY_TRANSLATION_SCALE}" in command_text
    assert f"--teleop.rotation_scale={fr3_teleop_smoke.DEFAULT_ROTATION_SCALE}" in command_text
    assert "--teleop.scale_x=" not in command_text
    assert f"--teleop.scale_wx={fr3_teleop_smoke.DEFAULT_SCALE_WX}" in command_text
    assert f"--teleop.scale_wy={fr3_teleop_smoke.DEFAULT_SCALE_WY}" in command_text
    assert f"--robot.max_target_delta_pos={fr3_teleop_smoke.DEFAULT_ROTATION_MAX_TARGET_DELTA_POS}" in command_text
    assert f"--robot.max_target_delta_rot={fr3_teleop_smoke.DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT}" in command_text


def test_combined_profile_enables_coupled_translation_and_rotation_smoke():
    args = fr3_teleop_smoke.parse_args(["--smoke-profile", "combined"])

    command = fr3_teleop_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert "--teleop.enable_rotation=true" in command_text
    assert f"--teleop.translation_scale={fr3_teleop_smoke.DEFAULT_TRANSLATION_SCALE}" in command_text
    assert f"--teleop.rotation_scale={fr3_teleop_smoke.DEFAULT_ROTATION_SCALE}" in command_text
    assert "--teleop.scale_x=" not in command_text
    assert f"--teleop.scale_wx={fr3_teleop_smoke.DEFAULT_SCALE_WX}" in command_text
    assert f"--teleop.scale_wy={fr3_teleop_smoke.DEFAULT_SCALE_WY}" in command_text
    assert "--teleop.scale_wz=" not in command_text
    assert f"--robot.max_target_delta_pos={fr3_teleop_smoke.DEFAULT_COMBINED_MAX_TARGET_DELTA_POS}" in command_text
    assert f"--robot.max_target_delta_rot={fr3_teleop_smoke.DEFAULT_COMBINED_MAX_TARGET_DELTA_ROT}" in command_text


def test_wz_profile_isolates_yaw_rotation_smoke():
    args = fr3_teleop_smoke.parse_args(["--smoke-profile", "wz"])

    command = fr3_teleop_smoke.build_docker_command(args)
    command_text = " ".join(command)

    assert "--teleop.enable_rotation=true" in command_text
    assert f"--teleop.translation_scale={fr3_teleop_smoke.DEFAULT_WZ_ONLY_TRANSLATION_SCALE}" in command_text
    assert f"--teleop.rotation_scale={fr3_teleop_smoke.DEFAULT_ROTATION_SCALE}" in command_text
    assert f"--teleop.scale_wx={fr3_teleop_smoke.DEFAULT_WZ_ONLY_SCALE_WX}" in command_text
    assert f"--teleop.scale_wy={fr3_teleop_smoke.DEFAULT_WZ_ONLY_SCALE_WY}" in command_text
    assert "--teleop.scale_wz=" not in command_text
    assert f"--robot.max_target_delta_pos={fr3_teleop_smoke.DEFAULT_ROTATION_MAX_TARGET_DELTA_POS}" in command_text
    assert f"--robot.max_target_delta_rot={fr3_teleop_smoke.DEFAULT_ROTATION_MAX_TARGET_DELTA_ROT}" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_teleop_smoke.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "lerobot-user" in captured.out
    assert "--robot.target_frame_name=pika_gripper_ee" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=5)

    monkeypatch.setattr(fr3_teleop_smoke.subprocess, "run", fake_run)

    exit_code = fr3_teleop_smoke.main([])

    assert exit_code == 5
    assert calls
