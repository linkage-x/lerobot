#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


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
