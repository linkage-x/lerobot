#!/usr/bin/env python

import json

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


TACTILE_LEFT = "observation.tactile.left_clean"
TACTILE_RIGHT = "observation.tactile.right_clean"
TACTILE_VALID_MASK = "observation.tactile.valid_mask"


def test_act_policy_forward_with_tactile_only_inputs():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            TACTILE_LEFT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_RIGHT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_VALID_MASK: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        use_tactile=True,
        tactile_feature_keys=[TACTILE_LEFT, TACTILE_RIGHT],
        tactile_use_valid_mask=True,
        tactile_valid_mask_feature_key=TACTILE_VALID_MASK,
        tactile_encoder_hidden_channels=[16, 32],
        tactile_encoder_residual_blocks=1,
        tactile_encoder_use_se=True,
        tactile_transformer_layers=1,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=2,
        chunk_size=4,
        n_action_steps=4,
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.train()

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, 8),
        TACTILE_LEFT: torch.randn(batch_size, 50, 10),
        TACTILE_RIGHT: torch.randn(batch_size, 50, 10),
        TACTILE_VALID_MASK: torch.randint(0, 2, (batch_size, 50, 10), dtype=torch.float32),
        ACTION: torch.randn(batch_size, config.chunk_size, 8),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    assert "quat_norm_mean" in loss_dict
    assert "quat_norm_max" in loss_dict
    assert "rot_geodesic_deg" in loss_dict
    assert loss_dict["quat_norm_mean"] > 0.0
    assert loss_dict["quat_norm_max"] > 0.0
    assert loss_dict["rot_geodesic_deg"] >= 0.0
    assert hasattr(policy.model, "encoder_tactile")
    assert hasattr(policy.model, "encoder_tactile_transformer")


def test_act_policy_rejects_missing_tactile_valid_mask_feature():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            TACTILE_LEFT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_RIGHT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        use_tactile=True,
        tactile_feature_keys=[TACTILE_LEFT, TACTILE_RIGHT],
        tactile_use_valid_mask=True,
        tactile_valid_mask_feature_key=TACTILE_VALID_MASK,
    )

    with pytest.raises(ValueError, match="valid mask feature"):
        config.validate_features()


def test_act_config_resolves_mask2ee_indices_from_bare_state_names():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        mask_ee_pose_in_state=True,
        state_feature_names={"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
        device="cpu",
    )

    assert config.masked_robot_state_indices == [0, 1, 2, 3, 4, 5, 6]


def test_act_config_can_mask_only_xyz_components():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        mask_ee_pose_in_state=True,
        mask_ee_state_components=["x", "y", "z"],
        state_feature_names={"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
        device="cpu",
    )

    assert config.masked_robot_state_indices == [0, 1, 2]


def test_act_config_mask2ee_reports_missing_state_features():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        mask_ee_pose_in_state=True,
        state_feature_names=["x", "y", "z", "qx", "qy", "gripper"],
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"Could not resolve required feature names \['ee.qz', 'ee.qw'\]"):
        _ = config.masked_robot_state_indices


def test_act_policy_mask2ee_zeroes_ee_pose_before_model_forward(monkeypatch):
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            TACTILE_LEFT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_RIGHT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_VALID_MASK: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        use_tactile=True,
        tactile_feature_keys=[TACTILE_LEFT, TACTILE_RIGHT],
        tactile_use_valid_mask=True,
        tactile_valid_mask_feature_key=TACTILE_VALID_MASK,
        tactile_encoder_hidden_channels=[16, 32],
        tactile_transformer_layers=1,
        chunk_size=4,
        n_action_steps=4,
        latent_dim=8,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        mask_ee_pose_in_state=True,
        state_feature_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.train()

    captured_state = {}

    def fake_forward(batch):
        captured_state["state"] = batch[OBS_STATE].clone()
        batch_size = batch[OBS_STATE].shape[0]
        actions_hat = torch.zeros(batch_size, config.chunk_size, 8)
        mu = torch.zeros(batch_size, config.latent_dim)
        log_sigma_x2 = torch.zeros(batch_size, config.latent_dim)
        return actions_hat, (mu, log_sigma_x2)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.9]], dtype=torch.float32),
        TACTILE_LEFT: torch.randn(1, 50, 10),
        TACTILE_RIGHT: torch.randn(1, 50, 10),
        TACTILE_VALID_MASK: torch.randint(0, 2, (1, 50, 10), dtype=torch.float32),
        ACTION: torch.randn(1, config.chunk_size, 8),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
    }

    policy.forward(batch)

    assert torch.equal(batch[OBS_STATE], torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.9]], dtype=torch.float32))
    assert torch.equal(
        captured_state["state"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]], dtype=torch.float32),
    )


def test_act_policy_can_mask_only_xyz_before_model_forward(monkeypatch):
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            TACTILE_LEFT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_RIGHT: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
            TACTILE_VALID_MASK: PolicyFeature(type=FeatureType.STATE, shape=(50, 10)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        use_tactile=True,
        tactile_feature_keys=[TACTILE_LEFT, TACTILE_RIGHT],
        tactile_use_valid_mask=True,
        tactile_valid_mask_feature_key=TACTILE_VALID_MASK,
        tactile_encoder_hidden_channels=[16, 32],
        tactile_transformer_layers=1,
        chunk_size=4,
        n_action_steps=4,
        latent_dim=8,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        mask_ee_pose_in_state=True,
        mask_ee_state_components=["x", "y", "z"],
        state_feature_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.train()

    captured_state = {}

    def fake_forward(batch):
        captured_state["state"] = batch[OBS_STATE].clone()
        batch_size = batch[OBS_STATE].shape[0]
        actions_hat = torch.zeros(batch_size, config.chunk_size, 8)
        mu = torch.zeros(batch_size, config.latent_dim)
        log_sigma_x2 = torch.zeros(batch_size, config.latent_dim)
        return actions_hat, (mu, log_sigma_x2)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.9]], dtype=torch.float32),
        TACTILE_LEFT: torch.randn(1, 50, 10),
        TACTILE_RIGHT: torch.randn(1, 50, 10),
        TACTILE_VALID_MASK: torch.randint(0, 2, (1, 50, 10), dtype=torch.float32),
        ACTION: torch.randn(1, config.chunk_size, 8),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
    }

    policy.forward(batch)

    assert torch.equal(
        captured_state["state"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.9]], dtype=torch.float32),
    )


def test_act_policy_forward_renormalizes_predicted_quaternion(monkeypatch):
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        chunk_size=2,
        n_action_steps=2,
        latent_dim=4,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.train()

    def fake_forward(batch):
        batch_size = batch[OBS_STATE].shape[0]
        actions_hat = torch.zeros(batch_size, config.chunk_size, 8)
        actions_hat[..., 3:7] = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        mu = torch.zeros(batch_size, config.latent_dim)
        log_sigma_x2 = torch.zeros(batch_size, config.latent_dim)
        return actions_hat, (mu, log_sigma_x2)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_STATE: torch.zeros(1, 8),
        OBS_ENV_STATE: torch.zeros(1, 4),
        ACTION: torch.zeros(1, config.chunk_size, 8),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
    }
    batch[ACTION][..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    _, loss_dict = policy.forward(batch)

    assert loss_dict["quat_norm_mean"] == pytest.approx(1.0, abs=1e-6)
    assert loss_dict["quat_norm_max"] == pytest.approx(1.0, abs=1e-6)
    assert loss_dict["rot_geodesic_deg"] == pytest.approx(0.0, abs=1e-6)
    assert loss_dict["rotation_geodesic_loss"] == pytest.approx(0.0, abs=1e-6)


def test_act_policy_predict_action_chunk_renormalizes_predicted_quaternion(monkeypatch):
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        chunk_size=2,
        n_action_steps=2,
        latent_dim=4,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.eval()

    def fake_forward(batch):
        batch_size = batch[OBS_STATE].shape[0]
        actions_hat = torch.zeros(batch_size, config.chunk_size, 8)
        actions_hat[..., 3:7] = torch.tensor([0.0, 3.0, 0.0, 4.0], dtype=torch.float32)
        mu = torch.zeros(batch_size, config.latent_dim)
        log_sigma_x2 = torch.zeros(batch_size, config.latent_dim)
        return actions_hat, (mu, log_sigma_x2)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_STATE: torch.zeros(1, 8),
        OBS_ENV_STATE: torch.zeros(1, 4),
    }

    actions = policy.predict_action_chunk(batch)
    quat_norm = torch.linalg.norm(actions[..., 3:7], dim=-1)

    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6)


def test_act_policy_predict_action_chunk_keeps_qoff_space_quaternion(monkeypatch, tmp_path):
    stats_path = tmp_path / "policy_action_chunk_stats.chunk2.json"
    stats_path.write_text(
        json.dumps(
            {
                "method": "quantile_per_offset",
                "chunk_size": 2,
                "lower_quantile": 0.02,
                "upper_quantile": 0.98,
                "action_names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
                "offset_stats": [
                    {
                        "offset": 0,
                        "q02": [-1.0] * 8,
                        "q98": [1.0] * 8,
                    },
                    {
                        "offset": 1,
                        "q02": [-1.0] * 8,
                        "q98": [1.0] * 8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
        chunk_size=2,
        n_action_steps=2,
        latent_dim=4,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        action_chunk_quantile_normalization=True,
        action_chunk_stats_path=str(stats_path),
        state_feature_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        device="cpu",
        use_amp=False,
    )

    policy = ACTPolicy(config)
    policy.eval()

    def fake_forward(batch):
        batch_size = batch[OBS_STATE].shape[0]
        actions_hat = torch.zeros(batch_size, config.chunk_size, 8)
        actions_hat[..., 3:7] = torch.tensor([0.0, 3.0, 0.0, 4.0], dtype=torch.float32)
        mu = torch.zeros(batch_size, config.latent_dim)
        log_sigma_x2 = torch.zeros(batch_size, config.latent_dim)
        return actions_hat, (mu, log_sigma_x2)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_STATE: torch.zeros(1, 8),
        OBS_ENV_STATE: torch.zeros(1, 4),
    }

    actions = policy.predict_action_chunk(batch)
    quat_norm = torch.linalg.norm(actions[..., 3:7], dim=-1)

    assert torch.allclose(quat_norm, torch.full_like(quat_norm, 5.0), atol=1e-6)
