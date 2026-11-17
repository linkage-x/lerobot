"""Load a relative-TCP checkpoint and predict with it: shared by training's reload check and Rollout.

A checkpoint is ``.../checkpoints/<step>/pretrained_model`` as lerobot_train writes it, plus
``fr3_policy_contract.json`` and ``fr3_provenance.json`` (train_relative_tcp.py). Without the contract
nothing says what the inputs are, so :func:`load_checkpoint` refuses it.

:func:`observation_sample` builds the policy's input from raw measurements (TCP poses at the contract's
history offsets, gripper width, wrench, camera frames) with the same functions training used on the
dataset (relative_tcp.state_vector, relative_tcp.crop_resize); offline prediction on a recording and
online preview from the recorder both go through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.fr3 import relative_tcp as rt


def predict_chunk(policy, preprocessor, postprocessor, sample: dict[str, Any]) -> np.ndarray:
    """(horizon, 10) unnormalised actions for one sample (tensors without a batch dimension)."""
    import torch

    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
    batch["task"] = [sample.get("task") or ""]
    batch = preprocessor(batch)
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    actions = postprocessor(actions)
    return actions[0].detach().float().cpu().numpy()


@dataclass
class LoadedPolicy:
    path: Path
    contract: rt.PolicyContract
    provenance: dict[str, Any]
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str

    def predict(self, sample: dict[str, Any]) -> np.ndarray:
        return predict_chunk(self.policy, self.preprocessor, self.postprocessor, sample)


def load_checkpoint(path: Path, device: str | None = None) -> LoadedPolicy:
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(path)
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    contract = rt.PolicyContract.read(path)
    prov_path = path / rt.PROVENANCE_FILE
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    expected = {c.key for c in contract.cameras} | {"observation.state"}
    got = set(cfg.input_features)
    if got != expected:
        raise ValueError(f"{path}: the policy's inputs {sorted(got)} are not the contract's {sorted(expected)}")
    return LoadedPolicy(path, contract, provenance, policy, pre, post, device)


def observation_sample(
    contract: rt.PolicyContract,
    history_poses: np.ndarray,
    width: float,
    images: dict[str, np.ndarray],
    wrench: np.ndarray | None = None,
    *,
    already_cropped: bool = False,
    task: str = "",
) -> dict[str, Any]:
    """The policy's input from measurements.

    ``history_poses``: (history + 1, 4, 4) TCP poses at ``contract.history_offsets`` (last = now), in any one
    frame; ``images``: camera name -> HxWx3 uint8 RGB, the full source frame unless ``already_cropped``.
    """
    import torch

    sample: dict[str, Any] = {
        "observation.state": torch.from_numpy(rt.state_vector(contract, history_poses, width, wrench)),
        "task": task,
    }
    for cam in contract.cameras:
        if cam.name not in images:
            raise ValueError(f"no frame from {cam.name}")
        frame = images[cam.name] if already_cropped else rt.crop_resize(images[cam.name], cam)
        if frame.shape != (cam.height, cam.width, 3):
            raise ValueError(f"{cam.name}: frame {frame.shape}, expected {(cam.height, cam.width, 3)}")
        sample[cam.key] = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
    return sample
