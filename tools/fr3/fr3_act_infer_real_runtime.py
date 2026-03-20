#!/usr/bin/env python3
"""
Minimal FR3 ACT real-robot inference runtime (runs inside Docker).

Execution model:
1. Read the training checkpoint and dataset metadata.
2. Reuse the FR3 camera wiring from tools/fr3/fr3_record_config.yaml.
3. Run low-rate policy inference at the dataset FPS.
4. Convert each absolute EE action to a robot EE command.
5. Hand the command to FrankaResearch3, which performs IK and joint-space OTG
   smoothing before sending high-rate joint targets to the controller.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotObservation
from lerobot.robots.franka_research3 import FrankaResearch3Config
from lerobot.robots.franka_research3.processor_franka_research3 import KeepAbsoluteEEObservation
from lerobot.utils.control_utils import predict_action
from lerobot.utils.rotation import Rotation
from lerobot.utils.robot_utils import precise_sleep

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "outputs/train/2026-03-19/10-48-39_act/checkpoints/060000"
_DEFAULT_CAMERA_CONFIG = _REPO_ROOT / "tools/fr3/fr3_record_config.yaml"
_DEFAULT_ROBOT_IP = "192.168.1.208"
_DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
_DEFAULT_CAMERA_KEY_MAP = "ee:left,side:right"
_DEFAULT_GRIPPER_BACKEND = "das"
_DAS_URDF = _REPO_ROOT / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf"
_DEFAULT_STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
_DEFAULT_ACTION_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR3 ACT real-robot inference inside Docker.")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--camera-config", type=Path, default=_DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--policy-fps", type=float, default=None, help="Optional low-rate policy update FPS override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional inference loop step limit.")
    parser.add_argument("--robot-ip", default=_DEFAULT_ROBOT_IP)
    parser.add_argument("--gripper-port", default=_DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-backend", choices=["pika", "das"], default=_DEFAULT_GRIPPER_BACKEND)
    parser.add_argument(
        "--camera-key-map",
        default=_DEFAULT_CAMERA_KEY_MAP,
        help="Map record-config camera keys to policy image keys, e.g. ee:left,side:right",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override.")
    parser.add_argument("--log-interval", type=int, default=30, help="Step interval for runtime logging.")
    return parser.parse_args(argv)


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def resolve_pretrained_model_dir(checkpoint_path: str | Path) -> Path:
    checkpoint_dir = _resolve_repo_path(checkpoint_path)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    if pretrained_dir.is_dir():
        return pretrained_dir
    if (checkpoint_dir / "config.json").is_file():
        return checkpoint_dir
    raise FileNotFoundError(f"Could not find pretrained_model/config.json under {checkpoint_dir}")


def parse_camera_key_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not value.strip():
        return mapping
    for item in value.split(","):
        source, sep, target = item.partition(":")
        if not sep:
            raise argparse.ArgumentTypeError("camera-key-map entries must use source:target form.")
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise argparse.ArgumentTypeError("camera-key-map entries must use source:target form.")
        mapping[source] = target
    return mapping


def load_train_config(pretrained_dir: Path) -> TrainPipelineConfig:
    return TrainPipelineConfig.from_pretrained(pretrained_dir)


def resolve_dataset_root(pretrained_dir: Path, train_cfg: TrainPipelineConfig, dataset_root: str | None) -> Path:
    root_value = dataset_root or train_cfg.dataset.root
    if root_value is None:
        raise ValueError(
            f"No dataset root resolved from {pretrained_dir / 'train_config.json'}. Pass --dataset-root explicitly."
        )
    return _resolve_repo_path(root_value)


def load_camera_configs(camera_config_path: str | Path) -> dict[str, RealSenseCameraConfig]:
    config_path = _resolve_repo_path(camera_config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    camera_entries = raw.get("robot", {}).get("cameras", {})
    if not camera_entries:
        raise ValueError(f"No robot.cameras entries found in {config_path}")

    camera_configs: dict[str, RealSenseCameraConfig] = {}
    for camera_name, cfg in camera_entries.items():
        camera_type = cfg.get("type")
        if camera_type != "intelrealsense":
            raise ValueError(f"Unsupported camera type '{camera_type}' in {config_path} for {camera_name}")
        camera_configs[camera_name] = RealSenseCameraConfig(
            serial_number_or_name=str(cfg["serial_number_or_name"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=int(cfg["fps"]),
        )
    return camera_configs


def load_dataset_metadata(dataset_root: Path, repo_id: str) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)


def extract_feature_names(feature_entry: dict[str, Any], default_names: list[str]) -> list[str]:
    names = feature_entry.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        for key in ("motors", "dimensions", "axes"):
            if isinstance(names.get(key), list):
                return [str(name) for name in names[key]]
    return list(default_names)


def normalize_dataset_gripper(aperture_m: float, cfg: FrankaResearch3Config) -> float:
    aperture_m = float(max(0.0, aperture_m))
    if cfg.gripper_backend == "das":
        span_m = float(cfg.das_max_distance_m - cfg.das_min_distance_m)
        if span_m <= 0.0:
            return 0.0
        return float(np.clip((aperture_m - cfg.das_min_distance_m) / span_m, 0.0, 1.0))
    max_width_m = float(cfg.gripper_max_width_mm) / 1000.0
    if max_width_m <= 0.0:
        return 0.0
    return float(np.clip(aperture_m / max_width_m, 0.0, 1.0))


def _state_name_to_observation_key(name: str) -> str:
    aliases = {
        "x": "ee.x",
        "y": "ee.y",
        "z": "ee.z",
        "qx": "ee.qx",
        "qy": "ee.qy",
        "qz": "ee.qz",
        "qw": "ee.qw",
        "gripper": "gripper.pos",
    }
    return aliases.get(name, name)


def _action_value(action_map: dict[str, float], *keys: str) -> float:
    for key in keys:
        if key in action_map:
            return float(action_map[key])
    raise KeyError(f"Missing action keys {keys!r} in decoded policy action.")


def build_policy_observation(
    robot_observation: RobotObservation,
    *,
    state_names: list[str],
    input_features: dict[str, Any],
    camera_key_map: dict[str, str],
    state_processor: KeepAbsoluteEEObservation,
) -> dict[str, np.ndarray]:
    processed = state_processor.observation(dict(robot_observation))
    observation: dict[str, np.ndarray] = {
        "observation.state": np.asarray(
            [processed[_state_name_to_observation_key(name)] for name in state_names],
            dtype=np.float32,
        )
    }

    for source_camera_key, target_camera_key in camera_key_map.items():
        if source_camera_key not in robot_observation:
            raise KeyError(f"Camera '{source_camera_key}' missing from robot observation.")
        observation[f"observation.images.{target_camera_key}"] = np.asarray(
            robot_observation[source_camera_key],
            dtype=np.uint8,
        )

    for feature_key, feature in input_features.items():
        if feature_key in observation:
            continue
        if feature.type == FeatureType.STATE:
            observation[feature_key] = np.zeros(tuple(feature.shape), dtype=np.float32)

    return observation


def decode_action_to_robot_command(
    action_tensor: torch.Tensor,
    *,
    action_names: list[str],
    robot_cfg: FrankaResearch3Config,
) -> dict[str, float]:
    action_np = np.asarray(action_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    if action_np.shape != (len(action_names),):
        raise ValueError(f"Expected policy action shape {(len(action_names),)}, got {action_np.shape}")

    action_map = {name: float(action_np[i]) for i, name in enumerate(action_names)}
    quaternion_xyzw = np.asarray(
        [
            _action_value(action_map, "qx", "ee.qx"),
            _action_value(action_map, "qy", "ee.qy"),
            _action_value(action_map, "qz", "ee.qz"),
            _action_value(action_map, "qw", "ee.qw"),
        ],
        dtype=np.float64,
    )
    rotvec_xyz = Rotation.from_quat(quaternion_xyzw).as_rotvec()
    gripper_normalized = normalize_dataset_gripper(
        _action_value(action_map, "gripper", "gripper.pos"),
        robot_cfg,
    )

    return {
        "ee.x": _action_value(action_map, "x", "ee.x"),
        "ee.y": _action_value(action_map, "y", "ee.y"),
        "ee.z": _action_value(action_map, "z", "ee.z"),
        "ee.wx": float(rotvec_xyz[0]),
        "ee.wy": float(rotvec_xyz[1]),
        "ee.wz": float(rotvec_xyz[2]),
        "gripper.pos": gripper_normalized,
    }


def load_policy_stack(
    pretrained_dir: Path,
    *,
    ds_meta: LeRobotDatasetMetadata,
    device: torch.device,
) -> tuple[Any, PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    policy_cfg = load_train_config(pretrained_dir).policy
    if policy_cfg is None:
        raise ValueError(f"No policy config found in {pretrained_dir / 'train_config.json'}")

    policy_cfg.device = str(device)
    policy_cfg.pretrained_path = pretrained_dir
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def run_inference(args: argparse.Namespace) -> int:
    pretrained_dir = resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = load_train_config(pretrained_dir)
    dataset_root = resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    ds_meta = load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    camera_key_map = parse_camera_key_map(args.camera_key_map)
    camera_configs = load_camera_configs(args.camera_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    missing_camera_sources = sorted(set(camera_key_map).difference(camera_configs))
    if missing_camera_sources:
        raise ValueError(
            f"camera-key-map references unknown cameras {missing_camera_sources}; available cameras: {sorted(camera_configs)}"
        )

    policy, preprocessor, postprocessor = load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device)
    policy_fps = float(args.policy_fps or ds_meta.fps)
    if policy_fps <= 0.0:
        raise ValueError("policy-fps must be positive.")
    state_names = extract_feature_names(ds_meta.features["observation.state"], _DEFAULT_STATE_NAMES)
    action_names = extract_feature_names(ds_meta.features["action"], _DEFAULT_ACTION_NAMES)

    robot_cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        workspace_min=(0.1, -0.6, 0.05),
        workspace_max=(0.9, 0.6, 0.8),
        cameras={name: cfg for name, cfg in camera_configs.items()},
    )

    from lerobot.robots.franka_research3 import FrankaResearch3

    robot = FrankaResearch3(robot_cfg)
    state_processor = KeepAbsoluteEEObservation()

    print(f"[INFO] checkpoint={pretrained_dir}")
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] policy_device={device}")
    print(f"[INFO] policy_fps={policy_fps:.3f}")
    print(
        "[INFO] camera_map="
        + ", ".join(f"{source}->{target}" for source, target in camera_key_map.items())
    )
    print(
        "[INFO] joint-space smoothing="
        f"FR3 OTG @ {robot_cfg.otg_control_frequency:.1f}Hz / sender @ {robot_cfg.otg_async_control_frequency:.1f}Hz"
    )

    robot.connect()
    policy.reset()

    try:
        step_idx = 0
        while args.max_steps is None or step_idx < args.max_steps:
            loop_start_t = time.perf_counter()
            robot_observation = robot.get_observation()
            policy_observation = build_policy_observation(
                robot_observation,
                state_names=state_names,
                input_features=policy.config.input_features,
                camera_key_map=camera_key_map,
                state_processor=state_processor,
            )
            action_tensor = predict_action(
                policy_observation,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp),
                robot_type=robot.name,
            )
            robot_command = decode_action_to_robot_command(
                action_tensor,
                action_names=action_names,
                robot_cfg=robot_cfg,
            )
            robot.send_action(robot_command)

            if step_idx % max(args.log_interval, 1) == 0:
                print(
                    "[INFO] step="
                    f"{step_idx} "
                    f"ee=({robot_command['ee.x']:.4f}, {robot_command['ee.y']:.4f}, {robot_command['ee.z']:.4f}) "
                    f"gripper={robot_command['gripper.pos']:.3f}"
                )

            elapsed_s = time.perf_counter() - loop_start_t
            precise_sleep(max(1.0 / policy_fps - elapsed_s, 0.0))
            step_idx += 1
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, stopping inference loop.")
    finally:
        robot.disconnect()

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_inference(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
