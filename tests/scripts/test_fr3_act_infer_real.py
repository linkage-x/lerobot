#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.gmsl2.configuration_gmsl2 import Gmsl2CameraConfig
from lerobot.cameras.hikrobot.configuration_hikrobot import HikrobotCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.franka_research3 import FrankaResearch3Config
from tools.fr3 import fr3_act_infer_real, fr3_act_infer_real_runtime


def test_build_docker_command_defaults_to_infer_service_and_profile(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(['--workspace', str(tmp_path), '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000', '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml'])

    command = fr3_act_infer_real.build_docker_command(args)
    command_text = ' '.join(command)

    assert command[:9] == [
        'docker',
        'compose',
        '--profile',
        'infer',
        '-f',
        str((tmp_path / 'docker' / 'docker-compose.yml').resolve()),
        'run',
        '-T',
        '--rm',
    ]
    assert 'lerobot-infer-fr3-act' in command
    assert 'tools/fr3/fr3_act_infer_real_runtime.py' in command_text
    assert '--checkpoint=/workspace/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000' in command_text
    assert '--camera-config=/workspace/tools/fr3/fr3_act_infer_camera_config.yaml' in command_text
    assert '--camera-key-map' not in command_text


def test_build_docker_command_allocates_a_tty_only_for_interactive_rollouts(tmp_path: Path):
    batch = fr3_act_infer_real.build_docker_command(fr3_act_infer_real.parse_args(['--workspace', str(tmp_path)]))
    interactive = fr3_act_infer_real.build_docker_command(
        fr3_act_infer_real.parse_args(['--workspace', str(tmp_path), '--interactive-rollouts'])
    )
    assert '-T' in batch[: batch.index('--rm')]
    assert '-T' not in interactive[: interactive.index('--rm')]


def test_build_docker_command_forwards_the_arm_and_leaves_the_gripper_to_it(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(['--workspace', str(tmp_path), '--arm', 'right'])
    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))
    assert '--arm=right' in command_text
    assert '--gripper-backend' not in command_text
    legacy = fr3_act_infer_real.parse_args(['--workspace', str(tmp_path)])
    assert '--gripper-backend=das' in ' '.join(fr3_act_infer_real.build_docker_command(legacy))


def test_main_dry_run_prints_command(capsys):
    exit_code = fr3_act_infer_real.main(['--dry-run'])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'docker compose --profile infer' in captured.out
    assert 'lerobot-infer-fr3-act' in captured.out
    assert '--camera-key-map' not in captured.out


def test_build_docker_command_passes_preview_and_safety_flags(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--preview',
            '--max-steps',
            '5',
            '--first-frame-max-pos-delta-mm',
            '20',
            '--first-frame-max-rot-delta-deg',
            '8',
            '--max-step-pos-delta-mm',
            '3',
            '--max-step-rot-delta-deg',
            '2',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--preview' in command_text
    assert '--max-steps=5' in command_text
    assert '--first-frame-max-pos-delta-mm=20.0' in command_text
    assert '--first-frame-max-rot-delta-deg=8.0' in command_text
    assert '--max-step-pos-delta-mm=3.0' in command_text
    assert '--max-step-rot-delta-deg=2.0' in command_text


def test_build_docker_command_passes_tactile_fallback(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--preview',
            '--tactile-fallback',
            'baseline_idle',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--preview' in command_text
    assert '--tactile-fallback=baseline_idle' in command_text


def test_build_docker_command_passes_corenetic_gripper_options(tmp_path: Path):
    urdf_path = tmp_path / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic.urdf'
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text('<robot name="fr3_corenetic"/>', encoding='utf-8')
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--gripper-backend',
            'corenetic',
            '--gripper-max-width-mm',
            '98',
            '--corenetic-bind-ip',
            '192.168.2.45',
            '--corenetic-remote-ip',
            '192.168.2.60',
            '--robot-urdf-path',
            str(urdf_path.relative_to(tmp_path)),
            '--target-frame-name',
            'corenetic_gripper_ee',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--gripper-backend=corenetic' in command_text
    assert '--gripper-max-width-mm=98.0' in command_text
    assert '--corenetic-bind-ip=192.168.2.45' in command_text
    assert '--corenetic-remote-ip=192.168.2.60' in command_text
    assert '--robot-urdf-path=/workspace/src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic.urdf' in command_text
    assert '--target-frame-name=corenetic_gripper_ee' in command_text


def test_build_docker_command_maps_legacy_box_backend_to_corenetic(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--gripper-backend',
            'box',
            '--box-bind-ip',
            '192.168.2.45',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--gripper-backend=corenetic' in command_text
    assert '--corenetic-bind-ip=192.168.2.45' in command_text


def test_build_docker_command_can_disable_default_startup_actions(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--no-move-to-das-start',
            '--no-align-gripper-to-dataset-start',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--no-move-to-das-start' in command_text
    assert '--no-align-gripper-to-dataset-start' in command_text


def test_build_docker_command_passes_robot_init_state(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--robot-init-state',
            'joints=0,0,0,0,0,0,0',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--robot-init-state=joints=0,0,0,0,0,0,0' in command_text


def test_build_docker_command_passes_mujoco_viewer_model(tmp_path: Path):
    model_path = tmp_path / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml'
    model_path.parent.mkdir(parents=True)
    model_path.write_text('<mujoco/>', encoding='utf-8')
    args = fr3_act_infer_real.parse_args(
        [
            '--workspace',
            str(tmp_path),
            '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
            '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
            '--mujoco-viewer',
            '--mujoco-model',
            str(model_path.relative_to(tmp_path)),
            '--mujoco-max-chunk-points',
            '24',
        ]
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--mujoco-viewer' in command_text
    assert '--mujoco-model=/workspace/src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml' in command_text
    assert '--mujoco-max-chunk-points=24' in command_text


def test_resolve_mujoco_model_path_uses_gripper_backend_defaults():
    assert fr3_act_infer_real_runtime.resolve_mujoco_model_path('das', None) == fr3_act_infer_real_runtime._DAS_XML
    assert fr3_act_infer_real_runtime.resolve_mujoco_model_path('pika', None) == fr3_act_infer_real_runtime._PIKA_XML


def test_fr3_config_accepts_corenetic_gripper_backend():
    cfg = FrankaResearch3Config(gripper_backend='corenetic', gripper_max_width_mm=98.0)

    assert cfg.gripper_backend == 'corenetic'
    assert cfg.gripper_max_width_mm == 98.0


def test_main_returns_subprocess_exit_code(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=9)

    monkeypatch.setattr(fr3_act_infer_real.subprocess, 'run', fake_run)

    exit_code = fr3_act_infer_real.main([])

    assert exit_code == 9
    assert calls


def test_move_to_das_start_if_requested_calls_integrated_helper(monkeypatch):
    import sys
    import types

    calls = []

    class FakePanda:
        def __init__(self, robot_ip):
            calls.append(('connect', robot_ip))

        def move_to_joint_position(self, joints):
            calls.append(('move', joints))

    monkeypatch.setitem(sys.modules, 'panda_py', types.SimpleNamespace(Panda=FakePanda))
    monkeypatch.setattr(fr3_act_infer_real_runtime.time, 'sleep', lambda _: None)

    fr3_act_infer_real_runtime.move_to_das_start_if_requested(
        robot_ip='192.168.1.208',
        enabled=True,
    )

    assert calls == [
        ('connect', '192.168.1.208'),
        ('move', fr3_act_infer_real_runtime._DAS_START_JOINTS_RAD.tolist()),
    ]


def test_parse_robot_init_state_shorthand_joints():
    parsed = fr3_act_infer_real_runtime.parse_robot_init_state('joints=0,1,2,3,4,5,6')

    assert parsed is not None
    assert parsed['type'] == 'joints'
    assert parsed['joints_rad'] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert parsed['timeout_s'] == 20.0


def test_parse_robot_init_state_yaml_file(tmp_path: Path):
    init_path = tmp_path / 'init.yaml'
    init_path.write_text(
        """robot_init_state:
  type: ee_xyzquat
  xyzquat: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
  gripper: 0.25
  timeout_s: 3
""",
        encoding='utf-8',
    )

    parsed = fr3_act_infer_real_runtime.parse_robot_init_state(str(init_path))

    assert parsed is not None
    assert parsed['type'] == 'ee_xyzquat'
    assert parsed['xyzquat'] == [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    assert parsed['gripper'] == 0.25
    assert parsed['timeout_s'] == 3.0


def test_extract_required_image_keys():
    input_features = {
        'observation.images.left': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.images.right': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }

    assert fr3_act_infer_real_runtime.extract_required_image_keys(input_features) == ['left', 'right']


def test_validate_camera_keys_accepts_matching_left_right():
    fr3_act_infer_real_runtime.validate_camera_keys(
        required_image_keys=['left', 'right'],
        available_camera_keys=['left', 'right'],
    )


def test_validate_camera_keys_rejects_missing_policy_camera():
    try:
        fr3_act_infer_real_runtime.validate_camera_keys(
            required_image_keys=['left', 'right'],
            available_camera_keys=['left'],
        )
    except ValueError as exc:
        assert 'missing policy-required cameras' in str(exc)
    else:
        raise AssertionError('expected ValueError for missing camera key')


def test_load_camera_configs_uses_infer_yaml_opencv_defaults():
    camera_configs = fr3_act_infer_real_runtime.load_camera_configs('tools/fr3/fr3_act_infer_camera_config.yaml')

    assert sorted(camera_configs) == ['left', 'right']
    assert str(camera_configs['left'].index_or_path) == '/dev/video22'
    assert camera_configs['left'].height == 480
    assert camera_configs['left'].width == 640
    assert camera_configs['left'].fourcc == 'MJPG'
    assert camera_configs['left'].backend == Cv2Backends.V4L2
    assert camera_configs['right'].fps == 30


def test_load_camera_configs_supports_explicit_opencv_backend_and_fourcc(tmp_path: Path):
    config_path = tmp_path / 'camera.yaml'
    config_path.write_text(
        """robot:
  cameras:
    left:
      type: opencv
      device_id: /dev/video22
      image_shape: [480, 640]
      fps: 30
      fourcc: YUYV
      backend: any
""",
        encoding='utf-8',
    )

    camera_configs = fr3_act_infer_real_runtime.load_camera_configs(config_path)

    assert camera_configs['left'].fourcc == 'YUYV'
    assert camera_configs['left'].backend == Cv2Backends.ANY


def test_load_camera_configs_respects_hikrobot_color_mode(tmp_path: Path):
    config_path = tmp_path / 'camera.yaml'
    config_path.write_text(
        """robot:
  cameras:
    wrist:
      type: hikrobot
      serial: "DA9342611"
      image_shape: [480, 640]
      fps: 30
      transport_layer: gige
      color_mode: rgb
""",
        encoding='utf-8',
    )

    camera_configs = fr3_act_infer_real_runtime.load_camera_configs(config_path)

    assert camera_configs['wrist'].color_mode == ColorMode.RGB


def test_load_camera_configs_supports_gmsl2(tmp_path: Path):
    config_path = tmp_path / 'camera.yaml'
    config_path.write_text(
        """robot:
  cameras:
    front:
      type: gmsl2
      sensor_id: 2
      device: /dev/video2
      pipeline: v4l2_bayer
      image_shape: [720, 1280]
      fps: 30
      color_mode: bgr
      rotation: rotate_180
      sync_role: slave
      trig_pin: "0x00020007"
      exposure_us: 5000
      gain: 3
""",
        encoding='utf-8',
    )

    camera_configs = fr3_act_infer_real_runtime.load_camera_configs(config_path)

    assert isinstance(camera_configs['front'], Gmsl2CameraConfig)
    assert camera_configs['front'].sensor_id == 2
    assert camera_configs['front'].resolved_device == '/dev/video2'
    assert camera_configs['front'].pipeline == 'v4l2_bayer'
    assert camera_configs['front'].height == 720
    assert camera_configs['front'].width == 1280
    assert camera_configs['front'].sync_role == 'slave'
    assert camera_configs['front'].trig_pin == 0x00020007
    assert camera_configs['front'].rotation.value == 180


def test_build_policy_observation_maps_state_images_and_tactile_passthrough():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        'observation.images.left': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.images.right': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.tactile.left_clean': PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        'observation.tactile.valid_mask': PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.4,
        'left': np.zeros((4, 5, 3), dtype=np.uint8),
        'right': np.ones((4, 5, 3), dtype=np.uint8),
        'observation.tactile.left_clean': np.full((50, 10), 7.0, dtype=np.float32),
        'observation.tactile.valid_mask': np.ones((50, 10), dtype=np.float32),
    }

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        state_observation,
        state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        input_features=input_features,
    )

    assert np.allclose(observation['observation.state'], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.4])
    # Camera frames are resized (no crop) to the policy's feature shape.
    assert observation['observation.images.left'].shape == (480, 640, 3)
    assert observation['observation.images.right'].shape == (480, 640, 3)
    assert not observation['observation.images.left'].any()
    assert np.all(observation['observation.images.right'] == 1)
    assert np.allclose(observation['observation.tactile.left_clean'], 7.0)
    assert np.allclose(observation['observation.tactile.valid_mask'], 1.0)


def test_build_policy_observation_supports_prev_cmd_state_names():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(16,)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.4,
        'prev_cmd.ee.x': 0.11,
        'prev_cmd.ee.y': 0.22,
        'prev_cmd.ee.z': 0.33,
        'prev_cmd.ee.qx': 0.0,
        'prev_cmd.ee.qy': 0.0,
        'prev_cmd.ee.qz': 0.0,
        'prev_cmd.ee.qw': 1.0,
        'prev_cmd.gripper.pos': 0.5,
    }

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        state_observation,
        state_names=[
            'ee.x',
            'ee.y',
            'ee.z',
            'ee.qx',
            'ee.qy',
            'ee.qz',
            'ee.qw',
            'gripper.pos',
            'prev_cmd.ee.x',
            'prev_cmd.ee.y',
            'prev_cmd.ee.z',
            'prev_cmd.ee.qx',
            'prev_cmd.ee.qy',
            'prev_cmd.ee.qz',
            'prev_cmd.ee.qw',
            'prev_cmd.gripper.pos',
        ],
        input_features=input_features,
    )

    assert np.allclose(
        observation['observation.state'],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.4, 0.11, 0.22, 0.33, 0.0, 0.0, 0.0, 1.0, 0.5],
    )


def test_build_policy_observation_supports_prefixed_training_state_names():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 61.0,
    }

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        state_observation,
        state_names=[
            'observation.state.right.ee.x',
            'observation.state.right.ee.y',
            'observation.state.right.ee.z',
            'observation.state.right.ee.qx',
            'observation.state.right.ee.qy',
            'observation.state.right.ee.qz',
            'observation.state.right.ee.qw',
            'observation.state_raw.handheld_gripper.pika_left.width_mm',
        ],
        input_features=input_features,
    )

    assert np.allclose(observation['observation.state'], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 61.0])


def test_build_policy_observation_normalizes_bgr_hikrobot_frames_to_rgb():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        'observation.images.wrist': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 5)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.4,
        'wrist': np.array([[[255, 0, 0]]], dtype=np.uint8),
    }
    camera_configs = {
        'wrist': HikrobotCameraConfig(
            serial='DA9342611',
            width=1,
            height=1,
            fps=30,
            warmup_s=0,
            transport_layer='gige',
            color_mode='bgr',
        )
    }

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        state_observation,
        state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        input_features=input_features,
        camera_configs=camera_configs,
    )

    assert observation['observation.images.wrist'][0, 0].tolist() == [0, 0, 255]


def test_build_policy_observation_rejects_missing_required_tactile():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        'observation.tactile.left_clean': PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.4,
    }

    try:
        fr3_act_infer_real_runtime.build_policy_observation(
            state_observation,
            state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
            input_features=input_features,
        )
    except KeyError as exc:
        assert 'Missing tactile observation keys' in str(exc)
    else:
        raise AssertionError('expected KeyError for missing tactile observation')


def test_build_tactile_fallback_observation_baseline_idle_uses_assets():
    observation = fr3_act_infer_real_runtime.build_tactile_fallback_observation('baseline_idle')

    assert observation is not None
    assert observation['observation.tactile.left_raw'].shape == (50, 10)
    assert observation['observation.tactile.right_raw'].shape == (50, 10)
    assert observation['observation.tactile.left_clean'].shape == (50, 10)
    assert observation['observation.tactile.right_clean'].shape == (50, 10)
    assert observation['observation.tactile.valid_mask'].shape == (50, 10)
    assert np.allclose(observation['observation.tactile.left_clean'], 0.0)
    assert np.allclose(observation['observation.tactile.right_clean'], 0.0)
    assert float(np.count_nonzero(observation['observation.tactile.valid_mask'])) == 448.0



def test_build_policy_observation_uses_tactile_fallback_for_missing_keys():
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        'observation.tactile.left_clean': PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        'observation.tactile.valid_mask': PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
    }
    state_observation = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.4,
    }
    tactile_fallback = fr3_act_infer_real_runtime.build_tactile_fallback_observation('baseline_idle')

    observation = fr3_act_infer_real_runtime.build_policy_observation(
        state_observation,
        state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        input_features=input_features,
        tactile_fallback_observation=tactile_fallback,
    )

    assert observation['observation.tactile.left_clean'].shape == (50, 10)
    assert np.allclose(observation['observation.tactile.left_clean'], 0.0)
    assert np.allclose(observation['observation.tactile.valid_mask'], tactile_fallback['observation.tactile.valid_mask'])


def test_convert_absolute_observation_from_E_to_I_is_the_identity_without_the_das_extrinsic():
    absolute_observation_e = {
        'ee.x': 0.1811,
        'ee.y': -0.5423,
        'ee.z': 0.2892,
        'ee.qx': 0.7412,
        'ee.qy': -0.0374,
        'ee.qz': 0.6692,
        'ee.qw': 0.0365,
        'gripper.pos': 0.942,
        'left': np.zeros((2, 2, 3), dtype=np.uint8),
    }

    absolute_observation_i = fr3_act_infer_real_runtime.convert_absolute_observation_from_E_to_I(absolute_observation_e)
    # The fixed DAS-rig E->I extrinsic was dropped (7633d8aa): E and I coincide.
    expected_pose_i = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_e)
    actual_pose_i = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_i)

    assert np.allclose(actual_pose_i, expected_pose_i)
    assert absolute_observation_i['gripper.pos'] == absolute_observation_e['gripper.pos']
    assert absolute_observation_i['left'].shape == (2, 2, 3)



def test_convert_base_command_from_I_to_E_roundtrips_absolute_pose():
    absolute_observation_e = {
        'ee.x': 0.1811,
        'ee.y': -0.5423,
        'ee.z': 0.2892,
        'ee.qx': 0.7412,
        'ee.qy': -0.0374,
        'ee.qz': 0.6692,
        'ee.qw': 0.0365,
        'gripper.pos': 0.942,
    }
    absolute_observation_i = fr3_act_infer_real_runtime.convert_absolute_observation_from_E_to_I(absolute_observation_e)
    absolute_pose_i = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_i)
    rotvec_i = fr3_act_infer_real_runtime.Rotation.from_matrix(absolute_pose_i[:3, :3]).as_rotvec()

    base_command_e = fr3_act_infer_real_runtime.convert_base_command_from_I_to_E(
        {
            'ee.x': float(absolute_pose_i[0, 3]),
            'ee.y': float(absolute_pose_i[1, 3]),
            'ee.z': float(absolute_pose_i[2, 3]),
            'ee.wx': float(rotvec_i[0]),
            'ee.wy': float(rotvec_i[1]),
            'ee.wz': float(rotvec_i[2]),
            'gripper.pos': 0.8,
        }
    )

    restored_pose_e = fr3_act_infer_real_runtime._pose_from_position_and_rotvec(
        np.asarray([base_command_e['ee.x'], base_command_e['ee.y'], base_command_e['ee.z']], dtype=np.float64),
        np.asarray([base_command_e['ee.wx'], base_command_e['ee.wy'], base_command_e['ee.wz']], dtype=np.float64),
    )
    expected_pose_e = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_e)

    assert np.allclose(restored_pose_e, expected_pose_e)
    assert base_command_e['gripper.pos'] == 0.8


def test_localize_observation_to_start_frame_zeroes_position_only_and_keeps_absolute_quaternion():
    absolute_observation = {
        'ee.x': 0.149,
        'ee.y': -0.497,
        'ee.z': 0.277,
        'ee.qx': 0.7412,
        'ee.qy': -0.0374,
        'ee.qz': 0.6692,
        'ee.qw': 0.0365,
        'gripper.pos': 0.942,
        'left': np.zeros((2, 2, 3), dtype=np.uint8),
    }
    episode_start_position_xyz = np.asarray([0.149, -0.497, 0.277], dtype=np.float64)

    localized_observation, local_quaternion_xyzw = fr3_act_infer_real_runtime.localize_observation_to_start_frame(
        absolute_observation,
        episode_start_position_xyz,
    )

    assert np.allclose(
        [localized_observation['ee.x'], localized_observation['ee.y'], localized_observation['ee.z']],
        [0.0, 0.0, 0.0],
        atol=1e-9,
    )
    assert np.allclose(local_quaternion_xyzw, [0.7412, -0.0374, 0.6692, 0.0365], atol=1e-9)
    assert np.allclose(
        [localized_observation['ee.qx'], localized_observation['ee.qy'], localized_observation['ee.qz'], localized_observation['ee.qw']],
        [0.7412, -0.0374, 0.6692, 0.0365],
        atol=1e-9,
    )
    assert localized_observation['left'].shape == (2, 2, 3)


def test_convert_local_command_to_base_frame_restores_position_only_and_keeps_absolute_orientation():
    episode_start_position_xyz = np.asarray([0.149, -0.497, 0.277], dtype=np.float64)
    absolute_quaternion_xyzw = np.asarray([0.0, 0.133716, 0.001008, 0.991015], dtype=np.float64)
    absolute_rotvec = fr3_act_infer_real_runtime.Rotation.from_quat(absolute_quaternion_xyzw).as_rotvec()

    base_command = fr3_act_infer_real_runtime.convert_local_command_to_base_frame(
        {
            'ee.x': 0.0,
            'ee.y': 0.0,
            'ee.z': 0.0,
            'ee.wx': float(absolute_rotvec[0]),
            'ee.wy': float(absolute_rotvec[1]),
            'ee.wz': float(absolute_rotvec[2]),
            'gripper.pos': 0.8,
        },
        episode_start_position_xyz,
    )

    assert np.allclose([base_command['ee.x'], base_command['ee.y'], base_command['ee.z']], [0.149, -0.497, 0.277])
    restored_rotation = fr3_act_infer_real_runtime.Rotation.from_rotvec(
        [base_command['ee.wx'], base_command['ee.wy'], base_command['ee.wz']]
    ).as_quat()
    assert np.allclose(restored_rotation, absolute_quaternion_xyzw)
    assert base_command['gripper.pos'] == 0.8


def test_build_hold_command_reuses_current_observation():
    command = fr3_act_infer_real_runtime.build_hold_command(
        {
            'ee.x': 0.1,
            'ee.y': 0.2,
            'ee.z': 0.3,
            'ee.wx': 0.01,
            'ee.wy': 0.02,
            'ee.wz': 0.03,
            'gripper.pos': 0.4,
        }
    )

    assert command == {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.wx': 0.01,
        'ee.wy': 0.02,
        'ee.wz': 0.03,
        'gripper.pos': 0.4,
    }


def test_resolve_dataset_data_file_uses_meta_template(tmp_path: Path):
    dataset_root = tmp_path / 'dataset'
    (dataset_root / 'meta').mkdir(parents=True)
    (dataset_root / 'data' / 'chunk-000').mkdir(parents=True)
    (dataset_root / 'meta' / 'info.json').write_text(
        '{'
        '"features":{"observation.state":{"names":["ee.x","ee.y","ee.z","ee.qx","ee.qy","ee.qz","ee.qw","gripper.pos"]}},'
        '"data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"'
        '}',
        encoding='utf-8',
    )
    data_file = dataset_root / 'data' / 'chunk-000' / 'file-000.parquet'
    data_file.write_bytes(b'PAR1testPAR1')

    resolved = fr3_act_infer_real_runtime._resolve_dataset_data_file(dataset_root, chunk_index=0, file_index=0)

    assert resolved == data_file


def test_extract_dataset_state_contract_indices_ignores_prev_cmd_suffix_fields(tmp_path: Path):
    dataset_root = tmp_path / 'dataset'
    (dataset_root / 'meta').mkdir(parents=True)
    (dataset_root / 'meta' / 'info.json').write_text(
        '{'
        '"features":{"observation.state":{"names":['
        '"ee.x","ee.y","ee.z","ee.qx","ee.qy","ee.qz","ee.qw","gripper.pos",'
        '"prev_cmd.ee.x","prev_cmd.ee.y","prev_cmd.ee.z","prev_cmd.ee.qx","prev_cmd.ee.qy","prev_cmd.ee.qz","prev_cmd.ee.qw","prev_cmd.gripper.pos"'
        ']}}'
        '}',
        encoding='utf-8',
    )

    indices = fr3_act_infer_real_runtime._extract_dataset_state_contract_indices(dataset_root)

    assert indices['ee.x'] == 0
    assert indices['ee.qw'] == 6
    assert indices['gripper.pos'] == 7


def test_denormalize_live_gripper_observation_matches_das_aperture():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.208',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='das',
        urdf_path='/tmp/fr3_das.urdf',
    )

    assert np.isclose(fr3_act_infer_real_runtime.denormalize_live_gripper_observation(1.0, robot_cfg), 0.103)
    assert np.isclose(fr3_act_infer_real_runtime.denormalize_live_gripper_observation(0.5, robot_cfg), 0.0515)


def test_should_reject_first_command_detects_large_first_step():
    reject, position_delta, rotation_delta = fr3_act_infer_real_runtime.should_reject_first_command(
        {
            'ee.x': 0.14,
            'ee.y': 0.0,
            'ee.z': 0.2,
            'ee.wx': 0.0,
            'ee.wy': 0.0,
            'ee.wz': float(np.deg2rad(12.0)),
            'gripper.pos': 0.5,
        },
        {
            'ee.x': 0.1,
            'ee.y': 0.0,
            'ee.z': 0.2,
            'ee.wx': 0.0,
            'ee.wy': 0.0,
            'ee.wz': 0.0,
            'gripper.pos': 0.4,
        },
        max_pos_delta_m=0.03,
        max_rot_delta_rad=float(np.deg2rad(10.0)),
    )

    assert reject is True
    assert np.allclose(position_delta, [0.04, 0.0, 0.0])
    assert np.allclose(rotation_delta, [0.0, 0.0, np.deg2rad(12.0)])


def test_clamp_command_relative_to_current_limits_absolute_pose_step():
    safe_command, position_delta, rotation_delta, clamped = fr3_act_infer_real_runtime.clamp_command_relative_to_current(
        {
            'ee.x': 0.11,
            'ee.y': -0.02,
            'ee.z': 0.21,
            'ee.wx': 0.0,
            'ee.wy': 0.0,
            'ee.wz': float(np.deg2rad(6.0)),
            'gripper.pos': 0.8,
        },
        {
            'ee.x': 0.1,
            'ee.y': 0.0,
            'ee.z': 0.2,
            'ee.wx': 0.0,
            'ee.wy': 0.0,
            'ee.wz': 0.0,
            'gripper.pos': 0.4,
        },
        max_pos_delta_m=0.005,
        max_rot_delta_rad=float(np.deg2rad(3.0)),
    )

    assert clamped is True
    assert np.allclose(position_delta, [0.01, -0.02, 0.01])
    assert np.allclose(rotation_delta, [0.0, 0.0, np.deg2rad(6.0)])
    assert np.allclose([safe_command['ee.x'], safe_command['ee.y'], safe_command['ee.z']], [0.105, -0.005, 0.205])
    assert np.isclose(safe_command['ee.wz'], np.deg2rad(3.0))
    assert safe_command['gripper.pos'] == 0.8


def test_decode_action_to_robot_command_converts_quat_and_gripper():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.208',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='das',
        urdf_path='/tmp/fr3_das.urdf',
    )
    action_tensor = torch.tensor([[0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.103]], dtype=torch.float32)

    command = fr3_act_infer_real_runtime.decode_action_to_robot_command(
        action_tensor,
        action_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        robot_cfg=robot_cfg,
    )

    assert np.isclose(command['ee.x'], 0.4)
    assert np.isclose(command['ee.y'], 0.1)
    assert np.isclose(command['ee.z'], 0.2)
    assert np.isclose(command['ee.wx'], 0.0)
    assert np.isclose(command['ee.wy'], 0.0)
    assert np.isclose(command['ee.wz'], 0.0)
    assert np.isclose(command['gripper.pos'], 1.0)


def test_preview_and_real_alignment_share_step0_policy_observation_state():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.208',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='das',
        urdf_path='/tmp/fr3_das.urdf',
    )
    input_features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    dataset_start_gripper = 0.096
    preview_live_state = {
        'ee.x': 0.1,
        'ee.y': 0.2,
        'ee.z': 0.3,
        'ee.qx': 0.0,
        'ee.qy': 0.0,
        'ee.qz': 0.0,
        'ee.qw': 1.0,
        'gripper.pos': 0.942,
    }
    preview_dataset_state = fr3_act_infer_real_runtime.convert_gripper_observation_to_dataset_units(
        preview_live_state,
        robot_cfg=robot_cfg,
    )
    preview_offset = dataset_start_gripper - float(preview_dataset_state['gripper.pos'])
    preview_policy_state = fr3_act_infer_real_runtime.apply_gripper_observation_offset(
        preview_dataset_state,
        gripper_offset=preview_offset,
    )

    real_live_state = dict(preview_live_state)
    real_live_state['gripper.pos'] = fr3_act_infer_real_runtime.normalize_dataset_gripper(dataset_start_gripper, robot_cfg)
    real_policy_state = fr3_act_infer_real_runtime.convert_gripper_observation_to_dataset_units(
        real_live_state,
        robot_cfg=robot_cfg,
    )

    preview_policy_observation = fr3_act_infer_real_runtime.build_policy_observation(
        preview_policy_state,
        state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        input_features=input_features,
    )
    real_policy_observation = fr3_act_infer_real_runtime.build_policy_observation(
        real_policy_state,
        state_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'],
        input_features=input_features,
    )

    assert np.isclose(float(preview_policy_state['gripper.pos']), dataset_start_gripper)
    assert np.isclose(float(real_policy_state['gripper.pos']), dataset_start_gripper)
    assert np.allclose(
        preview_policy_observation['observation.state'],
        real_policy_observation['observation.state'],
    )


def _info(world: str = '', link: str | None = None) -> dict:
    info: dict = {'world_frame': {'world_frame_id': world}} if world else {}
    if link is not None:
        info['tcp_frame'] = {'link': link}
    return info


def _resolve_alignment(
    requested='auto', policy=None, alignment=None, target='corenetic_gripper_ee', base='fr3_base'
):
    policy = _info() if policy is None else policy
    return fr3_act_infer_real_runtime.resolve_dataset_alignment(
        requested=requested,
        policy_info=policy,
        alignment_info=policy if alignment is None else alignment,
        target_frame_name=target,
        arm_base_world=base,
    )


def test_dataset_alignment_takes_camera_world_data_through_the_world_graph():
    fr3 = _info('fr3_base', 'corenetic_gripper_ee')
    camera_world = _info('world_20260928_063531', 'corenetic_gripper_ee')
    # Camera-world exports (what the GUI writes since 11-10) go through the measured edge by default;
    # the start-pose estimate is only on request.
    assert _resolve_alignment(policy=camera_world) == 'world_graph'
    assert _resolve_alignment(policy=camera_world, base='fr3_base_right') == 'world_graph'
    assert _resolve_alignment('start_pose', policy=camera_world) == 'start_pose'
    # Unstamped recordings have no world to look an edge up from.
    assert _resolve_alignment(policy=_info()) == 'start_pose'
    with pytest.raises(ValueError, match='predates world provenance'):
        _resolve_alignment('world_graph', policy=_info())
    # Data already in this arm's base is still moved to the arm by default.
    assert _resolve_alignment(policy=fr3) == 'start_position'
    assert _resolve_alignment('absolute', policy=fr3) == 'absolute'
    assert _resolve_alignment('start_pose', policy=fr3) == 'start_pose'
    with pytest.raises(ValueError, match='--dataset-alignment world_graph'):
        _resolve_alignment('absolute', policy=camera_world)
    # The left base is not the right one: left-base data on the right arm goes through the graph.
    assert _resolve_alignment(policy=fr3, base='fr3_base_right') == 'world_graph'
    with pytest.raises(ValueError, match="this arm's base fr3_base_right"):
        _resolve_alignment('start_position', policy=fr3, base='fr3_base_right')


def test_dataset_alignment_in_fr3_base_requires_the_ik_target_to_be_the_labelled_link():
    with pytest.raises(ValueError, match='--target-frame-name corenetic_gripper_ee'):
        _resolve_alignment(policy=_info('fr3_base', 'corenetic_gripper_ee'), target='pika_gripper_ee')
    with pytest.raises(ValueError, match='tcp_frame.link'):
        _resolve_alignment(policy=_info('fr3_base', ''))
    # The world-graph mode commands the label orientation as is too.
    with pytest.raises(ValueError, match='--target-frame-name corenetic_socket_ee'):
        _resolve_alignment(policy=_info('world_a', 'corenetic_socket_ee'), target='pika_gripper_ee')


_T_RIGHT_WORLD = [[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]]


def _graph(*edges):
    return {'version': 1, 'nodes': [], 'edges': list(edges)}


def test_world_graph_alignment_uses_the_live_edge_and_names_it():
    edge = {'from_world_frame_id': 'world_a', 'to_world_frame_id': 'fr3_base_right', 'T_to_from': _T_RIGHT_WORLD,
            'method': 'fr3_ball_pivot', 'created_utc': '2026-11-10T03:00:00Z', 'sigma_mm': 3.16}
    transform, route = fr3_act_infer_real_runtime.world_graph_alignment(_graph(edge), 'world_a', 'fr3_base_right')
    assert np.allclose(transform, _T_RIGHT_WORLD)
    assert 'world_a->fr3_base_right' in route and 'solved=2026-11-10T03:00:00Z' in route and 'sigma_mm=3.16' in route
    live = np.eye(4)
    live[:3, 3] = [9.0, 9.0, 9.0]  # wherever the arm starts, the edge decides
    assert np.allclose(
        fr3_act_infer_real_runtime.dataset_alignment_transform('world_graph', live, np.eye(4), world_graph_tf=transform),
        _T_RIGHT_WORLD,
    )


def test_world_graph_alignment_refuses_a_retired_edge_rather_than_estimate():
    edge = {'from_world_frame_id': 'world_a', 'to_world_frame_id': 'fr3_base', 'T_to_from': _T_RIGHT_WORLD,
            'method': 'test', 'created_utc': '2026-10-15T03:08:47Z',
            'retired': {'utc': '2026-11-03T08:00:00Z', 'reason': 'both FR3 bases moved'}}
    with pytest.raises(ValueError, match='retired 2026-11-03T08:00:00Z: both FR3 bases moved'):
        fr3_act_infer_real_runtime.world_graph_alignment(_graph(edge), 'world_a', 'fr3_base')
    with pytest.raises(ValueError, match='--dataset-alignment start_pose'):
        fr3_act_infer_real_runtime.world_graph_alignment(_graph(), 'world_a', 'fr3_base_right')


def test_a_dataset_put_in_a_base_through_a_since_retired_edge_is_stale():
    hop = {'from_world_frame_id': 'world_a', 'to_world_frame_id': 'fr3_base', 'created_utc': '2026-10-15T03:08:47Z',
           'method': 'test', 'traversed_reversed': False}
    info = {'world_frame': {'world_frame_id': 'fr3_base', 'edges': [hop]}}
    edge = {'from_world_frame_id': 'world_a', 'to_world_frame_id': 'fr3_base', 'T_to_from': _T_RIGHT_WORLD,
            'created_utc': '2026-10-15T03:08:47Z'}
    assert fr3_act_infer_real_runtime.stale_base_edges(_graph(edge), info) == []
    retired = {**edge, 'retired': {'utc': '2026-11-03T08:00:00Z', 'reason': 'both FR3 bases moved'}}
    (line,) = fr3_act_infer_real_runtime.stale_base_edges(_graph(retired), info)
    assert 'retired 2026-11-03T08:00:00Z: both FR3 bases moved' in line


def _arm_args(*argv):
    return fr3_act_infer_real_runtime.parse_args(['--checkpoint', '/nonexistent', *argv])


def test_the_arm_is_named_and_its_tool_comes_from_the_registry():
    with pytest.raises(ValueError, match='choose an arm'):
        fr3_act_infer_real_runtime.resolve_arm(_arm_args())
    args = _arm_args('--arm', 'right')
    entry = fr3_act_infer_real_runtime.resolve_arm(args)
    assert entry['arm'] == 'right'
    assert args.robot_ip == '192.168.1.210'
    assert args.gripper_backend == 'corenetic'
    assert args.corenetic_device_id == 3100715645
    assert str(args.robot_urdf_path).endswith('fr3_corenetic_gripper.urdf')
    assert args.target_frame_name == 'corenetic_socket_ee'
    # The IP alone finds the same arm; an IP that is no arm's is refused.
    assert fr3_act_infer_real_runtime.resolve_arm(_arm_args('--robot-ip', '192.168.1.210'))['arm'] == 'right'
    with pytest.raises(ValueError, match='192.168.1.208 is no FR3'):
        fr3_act_infer_real_runtime.resolve_arm(_arm_args('--robot-ip', '192.168.1.208'))
    with pytest.raises(ValueError, match="not the right arm's gripper BOX"):
        fr3_act_infer_real_runtime.resolve_arm(_arm_args('--arm', 'right', '--corenetic-device-id', '3906437858'))


def test_the_das_start_move_is_only_for_the_das_rig_and_never_in_preview():
    args = _arm_args('--arm', 'right')
    fr3_act_infer_real_runtime.resolve_arm(args)
    assert not fr3_act_infer_real_runtime.das_start_applies(args)  # corenetic arm
    das = _arm_args('--arm', 'right', '--gripper-backend', 'das')
    fr3_act_infer_real_runtime.resolve_arm(das)
    assert fr3_act_infer_real_runtime.das_start_applies(das)
    das.preview = True
    assert not fr3_act_infer_real_runtime.das_start_applies(das)


def test_an_arm_whose_gripper_box_is_not_wired_in_is_not_driven_unpinned():
    with pytest.raises(ValueError, match='inference cannot drive'):
        fr3_act_infer_real_runtime.resolve_arm(_arm_args('--arm', 'left'))
    with pytest.raises(ValueError, match='may be the handheld one'):
        fr3_act_infer_real_runtime.resolve_arm(_arm_args('--arm', 'left', '--gripper-backend', 'corenetic'))
    args = _arm_args('--arm', 'left', '--gripper-backend', 'corenetic', '--corenetic-device-id', '123')
    fr3_act_infer_real_runtime.resolve_arm(args)
    assert args.robot_ip == '192.168.11.102'
    assert args.target_frame_name == 'corenetic_socket_ee'  # same tool as the right arm


def test_dataset_alignment_reads_the_training_frame_not_a_re_exported_source():
    # The IL view keeps the frame it was built from; its source has since been re-exported.
    with pytest.raises(ValueError, match='re-exported since the view was built'):
        _resolve_alignment(
            policy=_info('world_20260928_063531', 'corenetic_gripper_ee'),
            alignment=_info('fr3_base', 'corenetic_gripper_ee'),
        )


def test_start_position_alignment_moves_the_demos_without_turning_them():
    rotation = fr3_act_infer_real_runtime.Rotation.from_rotvec([0.0, 0.0, 0.3]).as_matrix()
    contract = np.eye(4)
    contract[:3, 3] = [1.2, 0.7, 0.8]
    live = np.eye(4)
    live[:3, :3] = rotation  # the arm starts turned 17 deg from the demos' mean start
    live[:3, 3] = [0.5, 0.0, 0.4]

    shift = fr3_act_infer_real_runtime.dataset_alignment_transform('start_position', live, contract)
    assert np.allclose(shift[:3, :3], np.eye(3))
    assert np.allclose(shift[:3, 3], [-0.7, -0.7, -0.4])
    assert np.allclose(fr3_act_infer_real_runtime.dataset_alignment_transform('absolute', live, contract), np.eye(4))
    # The legacy estimate turns everything by the start-orientation difference.
    legacy = fr3_act_infer_real_runtime.dataset_alignment_transform('start_pose', live, contract)
    assert np.allclose(legacy[:3, :3], rotation)
    assert np.allclose(legacy @ contract, live)


def _unlock_args(preview: bool):
    import argparse

    return argparse.Namespace(preview=preview)


def _write_unlock(root: Path, **record) -> None:
    import json

    (root / 'outputs').mkdir(parents=True, exist_ok=True)
    (root / fr3_act_infer_real_runtime.REAL_EXECUTION_UNLOCK).write_text(json.dumps(record))


_NOW = __import__('datetime').datetime(2026, 11, 10, 9, 0, tzinfo=__import__('datetime').timezone.utc)
_UNLOCK = {'arm': 'right', 'by': 'maintainer', 'reason': 'L3 trial, checkpoint 060000',
           'created_utc': '2026-11-10T08:00:00Z', 'expires_utc': '2026-11-10T12:00:00Z'}


def test_preview_never_needs_an_unlock(tmp_path: Path):
    assert fr3_act_infer_real_runtime.real_execution_unlock(_unlock_args(True), 'right', repo_root=tmp_path) is None


def test_real_execution_is_locked_without_a_maintainer_unlock(tmp_path: Path):
    with pytest.raises(fr3_act_infer_real_runtime.RealExecutionLockedError, match='--preview'):
        fr3_act_infer_real_runtime.real_execution_unlock(_unlock_args(False), 'right', repo_root=tmp_path, now=_NOW)


def test_a_bounded_unlock_for_this_arm_lets_it_move(tmp_path: Path):
    _write_unlock(tmp_path, **_UNLOCK)
    record = fr3_act_infer_real_runtime.real_execution_unlock(
        _unlock_args(False), 'right', repo_root=tmp_path, now=_NOW
    )
    assert record['by'] == 'maintainer'


@pytest.mark.parametrize(
    ('change', 'arm', 'now_h', 'match'),
    [
        ({}, 'left', 9, "for arm 'right'"),  # an unlock for one arm never drives the other
        ({}, 'right', 13, 'not now'),  # expired
        ({}, 'right', 7, 'not now'),  # not yet
        ({'expires_utc': '2026-11-11T08:00:00Z'}, 'right', 9, 'at most 12 h'),  # standing unlock
        ({'by': ''}, 'right', 9, "'by'"),
        ({'expires_utc': 'tonight'}, 'right', 9, 'expires_utc'),
    ],
)
def test_an_unlock_that_does_not_fit_keeps_it_locked(tmp_path: Path, change, arm, now_h, match):
    _write_unlock(tmp_path, **{**_UNLOCK, **change})
    with pytest.raises(fr3_act_infer_real_runtime.RealExecutionLockedError, match=match):
        fr3_act_infer_real_runtime.real_execution_unlock(
            _unlock_args(False), arm, repo_root=tmp_path, now=_NOW.replace(hour=now_h)
        )


def test_main_refuses_real_execution_before_loading_anything(tmp_path: Path, monkeypatch, capsys):
    import functools

    def no_policy(*_a, **_k):
        raise AssertionError('nothing may be loaded before the lock is checked')

    unlock = fr3_act_infer_real_runtime.real_execution_unlock
    monkeypatch.setattr(fr3_act_infer_real_runtime, 'resolve_arm', lambda args: {'arm': 'right'})
    monkeypatch.setattr(fr3_act_infer_real_runtime, 'real_execution_unlock', functools.partial(unlock, repo_root=tmp_path))
    monkeypatch.setattr(fr3_act_infer_real_runtime, 'resolve_pretrained_model_dir', no_policy)
    assert fr3_act_infer_real_runtime.main(['--arm', 'right']) == 2
    assert '[REFUSED]' in capsys.readouterr().err
