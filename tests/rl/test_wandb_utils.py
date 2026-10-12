#!/usr/bin/env python

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.rl.wandb_utils import WandBLogger, get_safe_wandb_artifact_name
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


@pytest.mark.parametrize("length", [127, 128, 129, 300])
def test_artifact_name_length_boundary(length):
    name = "a" * length
    safe_name = get_safe_wandb_artifact_name(name)

    assert len(safe_name) <= 128
    if length <= 128:
        assert safe_name == name
    assert get_safe_wandb_artifact_name(name) == safe_name


def test_artifact_name_preserves_short_name_sanitization():
    assert get_safe_wandb_artifact_name("policy:pi05-dataset:local/fr3-005000") == (
        "policy_pi05-dataset_local_fr3-005000"
    )


def test_long_artifact_names_distinguish_datasets_and_steps():
    prefix = "policy:pi05-seed:1000-dataset:local/" + "fr3_" * 40
    names = [f"{prefix}a-005000", f"{prefix}b-005000", f"{prefix}a-010000"]
    safe_names = [get_safe_wandb_artifact_name(name) for name in names]

    assert len(set(safe_names)) == len(names)
    assert all(len(name) <= 128 and ":" not in name and "/" not in name for name in safe_names)


def test_log_policy_uploads_long_group_name(tmp_path):
    logger = WandBLogger.__new__(WandBLogger)
    logger.cfg = SimpleNamespace(disable_artifact=False)
    logger._group = "policy:pi05-seed:1000-dataset:local/" + "fr3_" * 40
    logger._wandb = Mock()
    checkpoint_dir = tmp_path / "005000"
    model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.safetensors"
    model_file.touch()

    logger.log_policy(checkpoint_dir)

    artifact_name = logger._wandb.Artifact.call_args.args[0]
    assert len(artifact_name) <= 128
    assert artifact_name == get_safe_wandb_artifact_name(f"{logger._group}-005000")
    logger._wandb.Artifact.assert_called_once_with(artifact_name, type="model")
    logger._wandb.Artifact.return_value.add_file.assert_called_once_with(model_file)
    logger._wandb.log_artifact.assert_called_once_with(logger._wandb.Artifact.return_value)


class FakeWandb:
    def __init__(self):
        self.logs = []
        self.metrics = []

    def define_metric(self, key, hidden=False):
        self.metrics.append((key, hidden))

    def log(self, *args, **kwargs):
        self.logs.append((args, kwargs))


def make_logger():
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb = FakeWandb()
    logger._wandb_custom_step_key = None
    return logger


def test_log_dict_expands_numeric_lists_as_scalar_metrics():
    logger = make_logger()

    logger.log_dict({"loss_per_dim": [1.25, 2.5], "loss": 3.75}, step=12)

    logged = [kwargs["data"] for _, kwargs in logger._wandb.logs]
    assert {"train/loss_per_dim/0": 1.25} in logged
    assert {"train/loss_per_dim/1": 2.5} in logged
    assert {"train/loss": 3.75} in logged
    assert all(kwargs["step"] == 12 for _, kwargs in logger._wandb.logs)


def test_log_dict_expands_numeric_lists_with_custom_step_key():
    logger = make_logger()

    logger.log_dict(
        {"Optimization step": 7, "loss_per_dim": (0.5, 0.75)},
        mode="train",
        custom_step_key="Optimization step",
    )

    logged = [args[0] for args, _ in logger._wandb.logs]
    assert {"train/loss_per_dim/0": 0.5, "train/Optimization step": 7} in logged
    assert {"train/loss_per_dim/1": 0.75, "train/Optimization step": 7} in logged


def test_log_dict_still_warns_for_non_scalar_lists(caplog):
    logger = make_logger()

    with caplog.at_level(logging.WARNING):
        logger.log_dict({"nested": [[1, 2]]}, step=1)

    assert 'WandB logging of key "nested" was ignored' in caplog.text
    assert logger._wandb.logs == []
