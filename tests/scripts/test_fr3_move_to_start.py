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
import sys
import types

import pytest

from tools.fr3 import fr3_move_to_start, fr3_move_to_start_runtime


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


def test_build_host_command_uses_workspace_venv_and_runtime_script(tmp_path: Path):
    host_python = tmp_path / ".venv" / "bin" / "python"
    host_python.parent.mkdir(parents=True)
    host_python.write_text("", encoding="utf-8")

    args = fr3_move_to_start.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--runtime",
            "host",
            "--robot-ip",
            "10.0.0.5",
        ]
    )

    command = fr3_move_to_start.build_host_command(args)

    assert command == [
        str(host_python),
        str((tmp_path / "tools" / "fr3" / "fr3_move_to_start_runtime.py").resolve()),
        "--robot-ip=10.0.0.5",
    ]


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_move_to_start.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose" in captured.out
    assert "tools/fr3/fr3_move_to_start_runtime.py" in captured.out
    assert "lerobot-user" in captured.out


def test_main_dry_run_host_prints_host_command(capsys):
    exit_code = fr3_move_to_start.main(["--runtime", "host", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PYTHONPATH=" in captured.out
    assert "tools/fr3/fr3_move_to_start_runtime.py" in captured.out


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


def test_main_host_passes_workspace_env_and_cwd(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(command, check=False, cwd=None, env=None):
        calls.append((command, check, cwd, env))
        return subprocess.CompletedProcess(command, returncode=7)

    monkeypatch.setattr(fr3_move_to_start.subprocess, "run", fake_run)

    exit_code = fr3_move_to_start.main(["--runtime", "host", "--workspace", str(tmp_path)])

    assert exit_code == 7
    assert calls
    _, check, cwd, env = calls[0]
    assert check is False
    assert cwd == tmp_path.resolve()
    assert isinstance(env, dict)


class _FakeState:
    def __init__(self, q):
        self.q = list(q)


class _FakePanda:
    """A robot that lands a fixed residual on each attempt, like the real controller does.

    ``move_to_joint_position`` returns panda_py's own convergence bool, and ``get_state`` returns
    the pose the *last* motion ended at -- the two facts the runtime now reads separately.
    """

    instances: list["_FakePanda"] = []
    residuals: list[list[float]] = []

    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.moves: list[list[float]] = []
        self.stopped = False
        self._q: list[float] = list(fr3_move_to_start_runtime.FR3_PIKA_HOME_JOINTS_RAD)
        type(self).instances.append(self)

    def move_to_joint_position(self, target):
        index = min(len(self.moves), len(type(self).residuals) - 1)
        residual = type(self).residuals[index]
        self.moves.append(list(target))
        self._q = [value + offset for value, offset in zip(target, residual, strict=True)]
        return max(abs(offset) for offset in residual) <= fr3_move_to_start_runtime.MOTION_SUCCESS_THRESHOLD_RAD

    def get_state(self):
        return _FakeState(self._q)

    def stop_controller(self):
        self.stopped = True


@pytest.fixture
def fake_panda(monkeypatch):
    _FakePanda.instances = []
    _FakePanda.residuals = [[0.0] * 7]
    module = types.ModuleType("panda_py")
    module.Panda = _FakePanda
    monkeypatch.setitem(sys.modules, "panda_py", module)
    monkeypatch.setattr(
        fr3_move_to_start_runtime, "check_ping", lambda robot_ip: ("PASS", "fake"), raising=True
    )
    monkeypatch.delenv(fr3_move_to_start_runtime.TOLERANCE_ENV_VAR, raising=False)
    return _FakePanda


# 0.0102 rad on joint 7 is the residual measured on the rig: the softest joint (impedance
# stiffness 50 vs 600) parked just past panda_py's own 0.01 success_threshold. It moves the EE by
# 0.02 mm. The old gate, set to that same 0.01, aborted the rollout on it.
_RIG_WRIST_RESIDUAL = [0.0, 0.006, 0.0, -0.0065, 0.0, 0.005, -0.0102]


def test_the_rig_wrist_residual_no_longer_aborts_homing(fake_panda, capsys):
    fake_panda.residuals = [list(_RIG_WRIST_RESIDUAL)]

    exit_code = fr3_move_to_start_runtime.main(["--robot-ip", "10.0.0.5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fr3_move_to_start=PASS" in captured.out
    # Accepted, but not silently: the log says the controller stopped short of its own threshold.
    assert "fr3_move_to_start=NOTE" in captured.out
    assert len(fake_panda.instances[0].moves) == 1, "a residual inside the gate must not re-move the arm"


def test_a_residual_at_the_controllers_own_threshold_is_not_a_coin_flip(fake_panda):
    """The gate has to sit clear of the number the motion generator stops at, or it fires at random."""
    assert (
        fr3_move_to_start_runtime.DEFAULT_TOLERANCE_RAD
        > 2 * fr3_move_to_start_runtime.MOTION_SUCCESS_THRESHOLD_RAD
    )


def test_joint_errors_are_reported_per_joint_and_name_the_worst_one(fake_panda, capsys):
    fake_panda.residuals = [list(_RIG_WRIST_RESIDUAL)]

    fr3_move_to_start_runtime.main([])

    captured = capsys.readouterr()
    assert "on joint7" in captured.out, "the worst joint has to be named; 'max error' alone is undiagnosable"
    assert "j7=-0.01020" in captured.out
    assert "j4=-0.00650" in captured.out


def test_a_residual_outside_the_gate_gets_one_more_trajectory(fake_panda, capsys):
    fake_panda.residuals = [[0.0] * 6 + [0.09], [0.0] * 6 + [0.004]]

    exit_code = fr3_move_to_start_runtime.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fr3_move_to_start=RETRY" in captured.err
    assert len(fake_panda.instances[0].moves) == 2


def test_a_real_mis_pose_still_fails_after_every_attempt(fake_panda, capsys):
    # Half a radian: the scale of an un-homed arm or the wrong keyframe, which is what the gate is for.
    fake_panda.residuals = [[0.5] + [0.0] * 6]

    with pytest.raises(RuntimeError) as excinfo:
        fr3_move_to_start_runtime.main([])

    assert "after 2 attempt(s)" in str(excinfo.value)
    assert "on joint1" in str(excinfo.value)
    assert "j1=+0.50000" in str(excinfo.value), "the failure has to carry every joint, not just the max"
    assert len(fake_panda.instances[0].moves) == 2
    assert fake_panda.instances[0].stopped, "the controller must be released even on failure"
    assert "fr3_move_to_start=FAIL" in capsys.readouterr().err


def test_attempts_can_be_pinned_to_one(fake_panda):
    fake_panda.residuals = [[0.5] + [0.0] * 6]

    with pytest.raises(RuntimeError) as excinfo:
        fr3_move_to_start_runtime.main(["--attempts", "1"])

    assert "after 1 attempt(s)" in str(excinfo.value)
    assert len(fake_panda.instances[0].moves) == 1


def test_the_rig_can_tighten_the_gate_without_a_code_edit(fake_panda, monkeypatch):
    monkeypatch.setenv(fr3_move_to_start_runtime.TOLERANCE_ENV_VAR, "0.005")
    fake_panda.residuals = [list(_RIG_WRIST_RESIDUAL)]

    with pytest.raises(RuntimeError) as excinfo:
        fr3_move_to_start_runtime.main([])

    assert "tolerance=0.00500" in str(excinfo.value)


def test_an_unparseable_or_nonpositive_override_falls_back_rather_than_disabling_the_gate(monkeypatch):
    for raw in ("", "   ", "not-a-number", "0", "-1"):
        monkeypatch.setenv(fr3_move_to_start_runtime.TOLERANCE_ENV_VAR, raw)
        assert fr3_move_to_start_runtime._env_tolerance_rad() == fr3_move_to_start_runtime.DEFAULT_TOLERANCE_RAD


def test_an_explicit_flag_still_beats_the_environment(monkeypatch):
    monkeypatch.setenv(fr3_move_to_start_runtime.TOLERANCE_ENV_VAR, "0.005")

    assert fr3_move_to_start_runtime.parse_args(["--tolerance-rad", "0.02"]).tolerance_rad == 0.02
