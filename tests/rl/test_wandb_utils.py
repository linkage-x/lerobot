#!/usr/bin/env python

import logging

from lerobot.rl.wandb_utils import WandBLogger


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
