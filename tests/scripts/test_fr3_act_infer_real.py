#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.franka_research3 import FrankaResearch3Config
from tools.fr3 import fr3_act_infer_real, fr3_act_infer_real_runtime


def test_build_docker_command_defaults_to_infer_service_and_profile(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(["--workspace", str(tmp_path)])

    command = fr3_act_infer_real.build_docker_command(args)
    command_text = " ".join(command)

    assert command[:8] == [
        "docker",
        "compose",
        "--profile",
        "infer",
        "-f",
        str((tmp_path / "docker" / "docker-compose.yml").resolve()),
        "run",
        "--rm",
    ]
    assert "lerobot-infer-fr3-act" in command
    assert "tools/fr3/fr3_act_infer_real_runtime.py" in command_text
    assert "--camera-key-map=ee:left,side:right" in command_text


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_act_infer_real.main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "docker compose --profile infer" in captured.out
    assert "lerobot-infer-fr3-act" in captured.out


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=9)

    monkeypatch.setattr(fr3_act_infer_real.subprocess, "run", fake_run)

    exit_code = fr3_act_infer_real.main([])

    assert exit_code == 9
    assert calls


def test_parse_camera_key_map():
    assert fr3_act_infer_real_runtime.parse_camera_key_map("ee:left,side:right") == {
        "ee": "left",
        "side": "right",
    }


def test_load_camera_configs_uses_record_yaml_defaults():
    camera_configs = fr3_act_infer_real_runtime.load_camera_configs("tools/fr3/fr3_record_config.yaml")

    assert sorted(camera_configs) == ["ee", "front", "side"]
    assert camera_configs["ee"].serial_number_or_name == "315122271876"
    assert camera_configs["side"].fps == 30


def test_build_policy_observation_maps_state_images_and_missing_tactile():
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.tactile.left_clean": PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
    }
    robot_observation = {
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "gripper.pos": 0.4,
        "ee": np.zeros((4, 5, 3), dtype=np.uint8),
        "side": np.ones((4, 5, 3), dtype=np.uint8),
    }

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        robot_observation,
        state_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        input_features=input_features,
        camera_key_map={"ee": "left", "side": "right"},
        state_processor=fr3_act_infer_real_runtime.KeepAbsoluteEEObservation(),
    )

    assert observation["observation.state"].tolist() == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.4]
    assert observation["observation.images.left"].shape == (4, 5, 3)
    assert observation["observation.images.right"].shape == (4, 5, 3)
    assert observation["observation.tactile.left_clean"].shape == (50, 10)
    assert np.allclose(observation["observation.tactile.left_clean"], 0.0)


def test_decode_action_to_robot_command_converts_quat_and_gripper():
    robot_cfg = FrankaResearch3Config(
        robot_ip="192.168.1.208",
        gripper_port="/dev/ttyUSB0",
        gripper_backend="das",
        urdf_path="/tmp/fr3_das.urdf",
    )
    action_tensor = torch.tensor([[0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.103]], dtype=torch.float32)

    command = fr3_act_infer_real_runtime.decode_action_to_robot_command(
        action_tensor,
        action_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        robot_cfg=robot_cfg,
    )

    assert command["ee.x"] == 0.4
    assert command["ee.y"] == 0.1
    assert command["ee.z"] == 0.2
    assert command["ee.wx"] == 0.0
    assert command["ee.wy"] == 0.0
    assert command["ee.wz"] == 0.0
    assert command["gripper.pos"] == 1.0
