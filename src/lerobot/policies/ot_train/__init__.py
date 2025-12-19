"""
OT (Optimal Transport) training support for LeRobot.

This package provides:
- A lightweight LeRobot-v3 compatible OT pair dataset that reads from
  LeRobotDataset instances and a DTW pair_info JSON.
- A minimal training helper that mimics the robomimic OT training loop
  while reusing LeRobot's policy factory and pre/post processors.

Note: The actual OT loss is policy-dependent. If your policy implements
`train_on_batch(batch: dict, b_ot: int, ot_params: dict, epoch: int, validate: bool)`
it will be called. Otherwise we fall back to a standard forward() pass on
the BC batch.
"""

from .ot_dataset import LeRobotOTPairDataset  # noqa: F401
from .ot_training import (run_epoch_for_ot_policy, make_ot_dataloader)  # noqa: F401