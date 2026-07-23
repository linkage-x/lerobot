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

from tools.fr3 import fr3_record_preflight


def test_extract_expected_hikrobot_gige_cameras_filters_by_type_and_transport():
    config_payload = {
        "robot": {
            "cameras": {
                "ee": {
                    "type": "hikrobot",
                    "serial": "DA001",
                    "transport_layer": "gige",
                },
                "wrist": {
                    "type": "hikrobot",
                    "serial": "USB001",
                    "transport_layer": "usb",
                },
                "side": {
                    "type": "opencv",
                    "device_id": 0,
                },
            }
        }
    }

    cameras = fr3_record_preflight.extract_expected_hikrobot_gige_cameras(config_payload)

    assert cameras == [
        {
            "type": "hikrobot",
            "serial": "DA001",
            "transport_layer": "gige",
            "name": "ee",
        }
    ]


def test_summarize_hikrobot_cameras_matches_expected_serials():
    expected = [
        {"name": "ee", "serial": "DA001", "transport_layer": "gige"},
        {"name": "side", "serial": "DA002", "transport_layer": "gige"},
    ]
    detected = [
        {"serial": "DA002", "current_ip": "192.168.2.11", "net_export": "192.168.2.2"},
        {"serial": "DA001", "current_ip": "192.168.2.10", "net_export": "192.168.2.2"},
    ]

    summary = fr3_record_preflight.summarize_hikrobot_cameras(expected, detected)

    assert summary.ok is True
    assert summary.matched_camera_names == ["ee", "side"]
    assert summary.missing_camera_names == []
    assert summary.unspecified_camera_names == []
    assert summary.suggestion is None


def test_summarize_hikrobot_cameras_reports_missing_and_builds_suggestion():
    expected = [
        {"name": "ee", "serial": "DA001", "transport_layer": "gige"},
        {"name": "side", "serial": None, "transport_layer": "gige"},
    ]
    detected = [
        {"serial": "DA010", "current_ip": "192.168.2.10", "net_export": "192.168.2.2"},
        {"serial": "DA011", "current_ip": "192.168.2.11", "net_export": "192.168.2.2"},
    ]

    summary = fr3_record_preflight.summarize_hikrobot_cameras(expected, detected)

    assert summary.ok is False
    assert summary.missing_camera_names == ["ee"]
    assert summary.unspecified_camera_names == ["side"]
    assert 'serial: "DA010"' in summary.suggestion
    assert 'serial: "DA011"' in summary.suggestion
    assert "    ee:" in summary.suggestion
    assert "    side:" in summary.suggestion


def test_build_hikrobot_camera_config_uses_valid_default_rotation():
    config = fr3_record_preflight._build_hikrobot_camera_config(
        {
            "name": "ee",
            "serial": "DA001",
            "transport_layer": "gige",
            "width": 1280,
            "height": 720,
            "fps": 30,
        }
    )

    assert config.rotation == fr3_record_preflight.Cv2Rotation.NO_ROTATION


def test_build_hikrobot_camera_config_accepts_string_rotation_alias():
    config = fr3_record_preflight._build_hikrobot_camera_config(
        {
            "name": "ee",
            "serial": "DA001",
            "transport_layer": "gige",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "rotation": "no_rotation",
        }
    )

    assert config.rotation == fr3_record_preflight.Cv2Rotation.NO_ROTATION


def test_build_record_command_uses_host_config_path(tmp_path: Path):
    config_path = tmp_path / "tools" / "fr3" / "fr3_record_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("robot:\n  robot_ip: 10.0.0.5\n", encoding="utf-8")
    args = fr3_record_preflight.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config-path",
            str(config_path),
        ]
    )

    command = fr3_record_preflight.build_record_command(args)

    assert "uv run --python .venv/bin/python python tools/fr3/fr3_record.py" in command
    assert str(config_path.resolve()) in command


def test_main_returns_zero_when_all_checks_pass(monkeypatch, capsys):
    args = fr3_record_preflight.parse_args(["--skip-ping", "--skip-arm", "--skip-gripper", "--skip-hikrobot"])

    monkeypatch.setattr(fr3_record_preflight, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(fr3_record_preflight, "ensure_host_env", lambda _args: None)
    monkeypatch.setattr(
        fr3_record_preflight,
        "run_preflight",
        lambda _args: ([fr3_record_preflight.CheckResult("host_runtime_imports", True, "ok")], None),
    )

    exit_code = fr3_record_preflight.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fr3_record_preflight=PASS" in captured.out
    assert "next_command:" in captured.out


def test_skip_host_imports_preserves_device_checks(monkeypatch):
    args = fr3_record_preflight.parse_args(
        ["--skip-host-imports", "--skip-ping", "--skip-arm", "--skip-gripper", "--skip-hikrobot"]
    )
    monkeypatch.setattr(
        fr3_record_preflight,
        "_load_runtime_config",
        lambda *_args: {"robot": {"robot_ip": "192.168.1.99", "gripper_backend": "corenetic"}},
    )
    monkeypatch.setattr(
        fr3_record_preflight,
        "check_host_runtime_imports",
        lambda: (_ for _ in ()).throw(AssertionError("generic imports must be skipped")),
    )

    results, _ = fr3_record_preflight.run_preflight(args)

    assert results == [
        fr3_record_preflight.CheckResult("host_runtime_imports", True, "skipped for replay-only preflight")
    ]


def test_main_returns_nonzero_when_any_check_fails(monkeypatch, capsys):
    args = fr3_record_preflight.parse_args(["--skip-ping", "--skip-arm", "--skip-gripper", "--skip-hikrobot"])

    monkeypatch.setattr(fr3_record_preflight, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(fr3_record_preflight, "ensure_host_env", lambda _args: None)
    monkeypatch.setattr(
        fr3_record_preflight,
        "run_preflight",
        lambda _args: ([fr3_record_preflight.CheckResult("fr3_arm", False, "connect failed")], None),
    )

    exit_code = fr3_record_preflight.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "fr3_record_preflight=FAIL" in captured.out
    assert "fr3_arm" in captured.out
