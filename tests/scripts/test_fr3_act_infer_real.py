#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import torch

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.hikrobot.configuration_hikrobot import HikrobotCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.franka_research3 import FrankaResearch3Config
from tools.fr3 import fr3_act_infer_real, fr3_act_infer_real_runtime


def test_build_docker_command_defaults_to_infer_service_and_profile(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(['--workspace', str(tmp_path), '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000', '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml'])

    command = fr3_act_infer_real.build_docker_command(args)
    command_text = ' '.join(command)

    assert command[:8] == [
        'docker',
        'compose',
        '--profile',
        'infer',
        '-f',
        str((tmp_path / 'docker' / 'docker-compose.yml').resolve()),
        'run',
        '--rm',
    ]
    assert 'lerobot-infer-fr3-act' in command
    assert 'tools/fr3/fr3_act_infer_real_runtime.py' in command_text
    assert '--checkpoint=/workspace/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000' in command_text
    assert '--camera-config=/workspace/tools/fr3/fr3_act_infer_camera_config.yaml' in command_text
    assert '--camera-key-map' not in command_text


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
    assert observation['observation.images.left'].shape == (4, 5, 3)
    assert observation['observation.images.right'].shape == (4, 5, 3)
    assert np.allclose(observation['observation.tactile.left_clean'], 7.0)
    assert np.allclose(observation['observation.tactile.valid_mask'], 1.0)


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


def test_convert_absolute_observation_from_E_to_I_matches_fixed_das_extrinsic():
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
    expected_pose_i = (
        fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_e)
        @ fr3_act_infer_real_runtime._T_EI
    )
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
