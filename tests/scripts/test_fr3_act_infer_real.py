#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
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

    # `-T` (no pseudo-TTY) only for non-interactive runs: interactive rollouts read keystrokes
    # from the terminal, so allocating a TTY is exactly what they need and `-T` would break them.
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


def _base_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        '--workspace',
        str(tmp_path),
        '--checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000',
        '--camera-config=/lerobot/tools/fr3/fr3_act_infer_camera_config.yaml',
        *extra,
    ]


def test_build_docker_command_can_disable_default_startup_actions(tmp_path: Path):
    args = fr3_act_infer_real.parse_args(
        _base_args(tmp_path, '--no-align-gripper-to-dataset-start')
    )

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--no-align-gripper-to-dataset-start' in command_text


def test_move_to_das_start_is_off_by_default(tmp_path: Path):
    """Those joint angles belong to the DAS rig, so homing to them offsets every target."""
    args = fr3_act_infer_real.parse_args(_base_args(tmp_path))

    assert args.move_to_das_start is False

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--move-to-das-start' not in command_text


def test_move_to_das_start_opt_in_reaches_the_runtime(tmp_path: Path):
    """The wrapper used to emit only the negative flag. Once the runtime default flipped, that
    turned an explicit opt-in into silence, and silence now means off."""
    args = fr3_act_infer_real.parse_args(_base_args(tmp_path, '--move-to-das-start'))

    assert args.move_to_das_start is True

    command_text = ' '.join(fr3_act_infer_real.build_docker_command(args))

    assert '--move-to-das-start' in command_text
    assert '--no-move-to-das-start' not in command_text


def test_no_move_to_das_start_is_still_accepted(tmp_path: Path):
    """Existing launchers pass it explicitly; it must stay a no-op rather than an error."""
    args = fr3_act_infer_real.parse_args(_base_args(tmp_path, '--no-move-to-das-start'))

    assert args.move_to_das_start is False
    assert '--move-to-das-start' not in ' '.join(fr3_act_infer_real.build_docker_command(args))


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
    # Resized to the checkpoint's declared feature shape (3, 480, 640), not passed through at the
    # camera's own size, and kept HWC -- the 4x5 inputs above are deliberately the wrong size so a
    # regression to pass-through shows up here rather than as a silent shape error inside the policy.
    assert observation['observation.images.left'].shape == (480, 640, 3)
    assert observation['observation.images.right'].shape == (480, 640, 3)
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


def test_convert_absolute_observation_from_E_to_I_is_a_pose_preserving_passthrough():
    """E and I are the same frame on this rig, and the conversion must stay a no-op.

    This used to apply a fixed DAS extrinsic ``_T_EI``. That extrinsic was removed and both
    conversions became pass-throughs, leaving the functions as named seams for a rig that does
    distinguish the two frames. The test now pins the pose is carried through *unchanged* rather
    than that some particular transform is applied: if an extrinsic is ever reintroduced, this
    fails and forces the matching change in ``convert_base_command_from_I_to_E``, which is its
    inverse and is what closes the loop back to the robot.
    """
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

    expected_pose = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_e)
    actual_pose = fr3_act_infer_real_runtime._pose_from_quaternion_observation(absolute_observation_i)

    assert np.allclose(actual_pose, expected_pose)
    assert absolute_observation_i['gripper.pos'] == absolute_observation_e['gripper.pos']
    assert absolute_observation_i['left'].shape == (2, 2, 3)
    # A copy, not the caller's dict: the runtime keeps using the E observation after converting.
    assert absolute_observation_i is not absolute_observation_e



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


def test_normalize_dataset_gripper_uses_feature_name_before_value_heuristic():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.206',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='pika',
        urdf_path='/tmp/fr3_pika.urdf',
        gripper_max_width_mm=90.0,
    )

    assert np.isclose(
        fr3_act_infer_real_runtime.normalize_dataset_gripper(0.08, robot_cfg, feature_name='gripper.pos'),
        0.08,
    )
    assert np.isclose(
        fr3_act_infer_real_runtime.normalize_dataset_gripper(70.0, robot_cfg, feature_name='observation.state_raw.handheld_gripper.pika_left.width_mm'),
        70.0 / 90.0,
    )
    # Legacy fallback stays unchanged when no unit-bearing feature name is available.
    assert np.isclose(fr3_act_infer_real_runtime.normalize_dataset_gripper(0.08, robot_cfg), 0.08 / 0.09)


def test_convert_gripper_observation_to_dataset_units_uses_state_feature_names():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.206',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='pika',
        urdf_path='/tmp/fr3_pika.urdf',
        gripper_max_width_mm=90.0,
    )

    normalized = fr3_act_infer_real_runtime.convert_gripper_observation_to_dataset_units(
        {'gripper.pos': 0.25, 'prev_cmd.gripper.pos': 0.5},
        robot_cfg=robot_cfg,
        state_names=['gripper.pos', 'prev_cmd.gripper.pos'],
    )
    width_mm = fr3_act_infer_real_runtime.convert_gripper_observation_to_dataset_units(
        {'gripper.pos': 0.25},
        robot_cfg=robot_cfg,
        state_names=['observation.state_raw.handheld_gripper.pika_left.width_mm'],
    )

    assert normalized['gripper.pos'] == 0.25
    assert normalized['prev_cmd.gripper.pos'] == 0.5
    assert np.isclose(width_mm['gripper.pos'], 22.5)


def test_decode_action_to_robot_command_treats_named_gripper_pos_as_normalized_for_pika():
    robot_cfg = FrankaResearch3Config(
        robot_ip='192.168.1.206',
        gripper_port='/dev/ttyUSB0',
        gripper_backend='pika',
        urdf_path='/tmp/fr3_pika.urdf',
        gripper_max_width_mm=90.0,
    )
    action_tensor = torch.tensor([[0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.08]], dtype=torch.float32)

    command = fr3_act_infer_real_runtime.decode_action_to_robot_command(
        action_tensor,
        action_names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper.pos'],
        robot_cfg=robot_cfg,
    )

    assert np.isclose(command['gripper.pos'], 0.08)


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


# Measured on the rig and in the training set that this policy was trained on
# (eeframe_fr3_spacemouse_20260813_160401__delta_ee_from_prev_cmd, 20 episodes / 10305 frames):
#
#   policy per-step delta   p50 1.59 mm   p95 2.93 mm   p99 3.57 mm
#   prev_cmd vs measured    p50 5.71 mm   p95 10.65 mm  max 15.92 mm
#
# The second row is servo tracking lag, not policy motion. Judging it by the first row's budget is
# what reported 299 of 299 steps as clamped on a rollout whose arm was in fact tracking *better*
# than the human demonstrations did (median lag 4.18 mm against the demos' 5.71 mm).
_DEMO_STEP_P95_M = 0.00293
_DEMO_LAG_P95_M = 0.01065
_SHIPPED_STEP_LIMIT_M = fr3_act_infer_real_runtime._DEFAULT_MAX_STEP_POS_DELTA_MM / 1000.0
_SHIPPED_LEASH_LIMIT_M = fr3_act_infer_real_runtime._DEFAULT_MAX_LEASH_POS_DELTA_MM / 1000.0


def _observation(*, ee_xyz, prev_cmd_xyz=None, ee_wz=0.0, prev_cmd_wz=0.0):
    """An FR3 observation carrying both the measured pose and the driver's last sent command."""
    observation = {
        'ee.x': ee_xyz[0],
        'ee.y': ee_xyz[1],
        'ee.z': ee_xyz[2],
        'ee.wx': 0.0,
        'ee.wy': 0.0,
        'ee.wz': ee_wz,
        'gripper.pos': 0.4,
    }
    if prev_cmd_xyz is not None:
        observation.update(
            {
                'prev_cmd.ee.x': prev_cmd_xyz[0],
                'prev_cmd.ee.y': prev_cmd_xyz[1],
                'prev_cmd.ee.z': prev_cmd_xyz[2],
                'prev_cmd.ee.wx': 0.0,
                'prev_cmd.ee.wy': 0.0,
                'prev_cmd.ee.wz': prev_cmd_wz,
            }
        )
    return observation


def _command(xyz, *, wz=0.0, gripper=0.8):
    return {
        'ee.x': xyz[0],
        'ee.y': xyz[1],
        'ee.z': xyz[2],
        'ee.wx': 0.0,
        'ee.wy': 0.0,
        'ee.wz': wz,
        'gripper.pos': gripper,
    }


def _guard(command, observation, **overrides):
    kwargs = {
        'max_step_pos_delta_m': _SHIPPED_STEP_LIMIT_M,
        'max_step_rot_delta_rad': float(np.deg2rad(fr3_act_infer_real_runtime._DEFAULT_MAX_STEP_ROT_DELTA_DEG)),
        'max_leash_pos_delta_m': _SHIPPED_LEASH_LIMIT_M,
        'max_leash_rot_delta_rad': float(np.deg2rad(fr3_act_infer_real_runtime._DEFAULT_MAX_LEASH_ROT_DELTA_DEG)),
    }
    kwargs.update(overrides)
    return fr3_act_infer_real_runtime.limit_command_for_safety(command, observation, **kwargs)


def test_tracking_lag_alone_does_not_limit_a_healthy_command():
    """The 299/299 case: the arm trails its command, the policy asks for an ordinary step."""
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.2 + 0.00418))
    # A p50-sized policy step on top of a median rig lag.
    command = _command((0.1, 0.0, 0.2 + 0.00418 + 0.0016))

    safe_command, guard = _guard(command, observation)

    assert guard['status'] == 'pass'
    assert guard['step_limited'] is False
    assert guard['leash_limited'] is False
    # The command reaches the policy's target untouched.
    assert np.isclose(safe_command['ee.z'], command['ee.z'])
    # ...even though the gap from the measured pose is well past the step limit.
    assert np.linalg.norm(guard['position_delta']) > _SHIPPED_STEP_LIMIT_M


def test_the_demonstrated_lag_and_step_both_fit_inside_the_shipped_limits():
    """The gate must admit the data it exists to reproduce.

    p95 lag plus a p95 policy step is an ordinary frame of the training set. If the shipped limits
    reject that, they reject the demonstrations, which is exactly the bug this pair replaced.
    """
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.2 + _DEMO_LAG_P95_M))
    command = _command((0.1, 0.0, 0.2 + _DEMO_LAG_P95_M + _DEMO_STEP_P95_M))

    _, guard = _guard(command, observation)

    assert guard['status'] == 'pass'


def test_the_step_guard_measures_the_policy_delta_against_prev_cmd():
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.21))
    command = _command((0.1, 0.0, 0.212))

    _, guard = _guard(command, observation)

    # 2 mm from prev_cmd, not 12 mm from the measured pose.
    assert np.allclose(guard['step_position_delta'], [0.0, 0.0, 0.002])
    assert np.allclose(guard['position_delta'], [0.0, 0.0, 0.012])
    assert guard['has_prev_cmd_reference'] is True


def test_an_over_long_step_is_shortened_without_being_turned():
    """Per-axis clipping bent a descent sideways; scaling must not."""
    prev_cmd = np.array([0.1, 0.0, 0.2])
    ask = np.array([0.001, 0.0025, -0.02])  # dominated by z, like a reach downward
    observation = _observation(ee_xyz=tuple(prev_cmd), prev_cmd_xyz=tuple(prev_cmd))
    command = _command(tuple(prev_cmd + ask))

    safe_command, guard = _guard(command, observation)

    assert guard['status'] == 'step_limited'
    sent = np.array([safe_command['ee.x'], safe_command['ee.y'], safe_command['ee.z']]) - prev_cmd
    assert np.isclose(np.linalg.norm(sent), _SHIPPED_STEP_LIMIT_M)
    # Same heading, shorter: the cosine between ask and sent is 1.
    assert np.isclose(sent @ ask / (np.linalg.norm(sent) * np.linalg.norm(ask)), 1.0)
    # Per-axis np.clip would have left x and y untouched and only cut z.
    assert not np.isclose(sent[1], ask[1])


def test_a_command_running_away_from_a_stuck_arm_hits_the_leash():
    """The failure the command-vs-measured direction actually exists to catch."""
    # The arm has not moved; the command has been marching away from it.
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.24))
    command = _command((0.1, 0.0, 0.242))

    safe_command, guard = _guard(command, observation)

    assert guard['status'] == 'leash_limited'
    assert guard['leash_limited'] is True
    assert np.isclose(safe_command['ee.z'] - 0.2, _SHIPPED_LEASH_LIMIT_M)


def test_the_leash_is_reported_over_a_step_limit_when_both_fire():
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.24))
    command = _command((0.1, 0.0, 0.26))

    _, guard = _guard(command, observation)

    assert guard['step_limited'] is True
    assert guard['leash_limited'] is True
    # The louder of the two wins the status: it is the one that says stop.
    assert guard['status'] == 'leash_limited'


def test_a_missing_prev_cmd_falls_back_to_the_measured_pose():
    """A robot that reports no prev_cmd must still be guarded, not silently ungated."""
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2))
    command = _command((0.1, 0.0, 0.22))

    safe_command, guard = _guard(command, observation)

    assert guard['has_prev_cmd_reference'] is False
    assert guard['status'] == 'step_limited'
    assert np.isclose(safe_command['ee.z'] - 0.2, _SHIPPED_STEP_LIMIT_M)


def test_rotation_is_scaled_as_a_vector_rather_than_clipped_per_axis():
    observation = _observation(ee_xyz=(0.1, 0.0, 0.2), prev_cmd_xyz=(0.1, 0.0, 0.2))
    command = _command((0.1, 0.0, 0.2), wz=float(np.deg2rad(9.0)))

    safe_command, guard = _guard(command, observation)

    assert guard['status'] == 'step_limited'
    assert np.isclose(
        safe_command['ee.wz'],
        np.deg2rad(fr3_act_infer_real_runtime._DEFAULT_MAX_STEP_ROT_DELTA_DEG),
    )


def test_the_step_limit_stays_below_the_leash():
    """They bound different quantities, and the ordering is what keeps them distinguishable.

    A leash at or under the step limit would fire first on every ordinary step and collapse the
    pair back into the single mixed-reference check this replaced.
    """
    assert _SHIPPED_LEASH_LIMIT_M > 2 * _SHIPPED_STEP_LIMIT_M
    assert _SHIPPED_LEASH_LIMIT_M >= _DEMO_LAG_P95_M


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
