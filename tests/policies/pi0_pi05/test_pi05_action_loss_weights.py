#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-dimension action loss weighting.

Kept apart from the other pi0.5 tests because this is pure tensor arithmetic: it needs no
checkpoint, no CUDA and no HF token, so it can gate a change that the model-loading tests cannot.
"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import apply_action_loss_weights  # noqa: E402


def test_equal_weights_leave_the_loss_alone():
    """The renormalisation makes all-equal weights a no-op, whatever value they share."""
    losses = torch.rand(2, 3, 5)

    for value in (1.0, 7.0, 0.01):
        out = apply_action_loss_weights(losses, [value] * 5)
        assert torch.allclose(out, losses, atol=1e-6)


def test_weights_redistribute_between_dims_without_changing_the_total():
    """Re-weighting must change how the gradient is divided, not how large it is.

    Without the mean-1 renormalisation every weight change would silently be a learning-rate
    change too, and two runs meant to differ only in emphasis would not be comparable.
    """
    losses = torch.ones(4, 6, 5)

    out = apply_action_loss_weights(losses, [3.0, 1.0, 1.0, 1.0, 1.0])

    # Total budget preserved: the mean over all dims is unchanged.
    assert out.mean().item() == pytest.approx(losses.mean().item(), abs=1e-6)
    # ...but redistributed towards dim 0.
    per_dim = out.mean(dim=[0, 1])
    assert per_dim[0].item() > per_dim[1].item()
    assert per_dim[0].item() / per_dim[1].item() == pytest.approx(3.0, rel=1e-5)


def test_a_zero_weight_removes_a_dimension_from_the_loss():
    """The way to say "this axis is locked, do not spend gradient on it"."""
    losses = torch.ones(2, 3, 5)

    out = apply_action_loss_weights(losses, [1.0, 1.0, 1.0, 1.0, 0.0])

    assert out.mean(dim=[0, 1])[4].item() == pytest.approx(0.0, abs=1e-8)
    assert out.mean(dim=[0, 1])[0].item() > 0.0


def test_length_must_match_the_action_dim():
    # A silently mismatched vector would weight the wrong axes -- the failure would look like a
    # training problem, not a config one.
    losses = torch.ones(2, 3, 5)

    with pytest.raises(ValueError, match="4 entries but the action has 5"):
        apply_action_loss_weights(losses, [1.0, 1.0, 1.0, 1.0])


def test_negative_and_all_zero_weights_are_refused():
    losses = torch.ones(2, 3, 5)

    with pytest.raises(ValueError, match="non-negative"):
        apply_action_loss_weights(losses, [1.0, -1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        apply_action_loss_weights(losses, [0.0] * 5)


def test_weighting_preserves_dtype_and_shape():
    losses = torch.rand(3, 7, 5, dtype=torch.float64)

    out = apply_action_loss_weights(losses, [1.0, 2.0, 3.0, 4.0, 5.0])

    assert out.shape == losses.shape
    assert out.dtype == losses.dtype


def test_the_rigs_intended_split_scales_translation_against_the_gripper():
    """The case this was built for: MEAN_STD equalises scale, weights restore the budget.

    Action order on the FR3 view is (dx, dy, dz, drz, gripper). MEAN_STD gives every dim an
    equal share, which starved the gripper; this puts some back without re-starving translation.
    """
    losses = torch.ones(2, 4, 5)

    out = apply_action_loss_weights(losses, [1.0, 1.0, 1.0, 1.0, 2.0])
    per_dim = out.mean(dim=[0, 1])

    share = (per_dim / per_dim.sum()).tolist()
    assert share[4] == pytest.approx(1 / 3, rel=1e-5)          # gripper
    assert sum(share[:3]) == pytest.approx(0.5, rel=1e-5)      # dx+dy+dz
    assert out.mean().item() == pytest.approx(1.0, abs=1e-6)   # total unchanged
