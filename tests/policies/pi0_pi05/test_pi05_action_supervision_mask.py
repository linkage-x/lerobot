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

"""Which entries of a pi0.5 action chunk carry a label worth regressing on.

Pure tensor arithmetic, like `test_pi05_action_loss_weights.py` and for the same reason: no
checkpoint, no CUDA, no HF token, so it can gate a change the model-loading tests cannot.
"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    _masked_mean,
    action_supervision_mask,
)


def test_no_padding_and_no_intervention_masks_nothing():
    """None, not a tensor of ones: the common case must stay bit-identical to upstream."""
    losses = torch.rand(2, 4, 5)

    assert action_supervision_mask(losses) is None
    assert action_supervision_mask(losses, action_is_pad=None, is_intervention=None) is None
    # A flag with no dimensions listed is also nothing to mask.
    assert (
        action_supervision_mask(
            losses,
            is_intervention=torch.ones(2, 1),
            intervention_unsupervised_dims=[],
        )
        is None
    )
    assert torch.allclose(_masked_mean(losses, None), losses.mean())


def test_padded_steps_leave_the_loss_entirely():
    """A padded step is not a target of zero, it is not a target -- so it leaves the denominator.

    The distinction is the whole point. Zeroing alone would make a batch full of padding a batch
    with a smaller learning rate, which is exactly the bias this removes.
    """
    losses = torch.ones(1, 4, 2)
    losses[:, 2:, :] = 100.0  # the fabricated tail: the last real action, repeated
    pad = torch.tensor([[False, False, True, True]])

    mask = action_supervision_mask(losses, action_is_pad=pad)

    assert mask is not None
    assert mask.shape == losses.shape
    # Mean over the two real steps only -- not (1+1+100+100)/4.
    assert _masked_mean(losses, mask).item() == pytest.approx(1.0)


def test_padding_mask_broadcasts_over_the_action_dimensions():
    """`action_is_pad` is (B, T); the loss is (B, T, D). A step is padded in every dimension."""
    losses = torch.rand(3, 6, 5)
    pad = torch.zeros(3, 6, dtype=torch.bool)
    pad[1, 4:] = True

    mask = action_supervision_mask(losses, action_is_pad=pad)

    assert mask.shape == losses.shape
    assert torch.equal(mask[1, 4:], torch.zeros(2, 5))
    assert torch.equal(mask[0], torch.ones(6, 5))


def test_an_intervention_drops_only_the_dimensions_it_did_not_label():
    """The gripper column of a takeover frame is the policy's own output, not the expert's.

    Everything else about that frame is a real correction and must keep training.
    """
    losses = torch.ones(2, 3, 5)
    flags = torch.tensor([[0.0], [1.0]])  # sample 0 is a demonstration, sample 1 a correction

    mask = action_supervision_mask(
        losses,
        is_intervention=flags,
        intervention_unsupervised_dims=[4],
    )

    assert torch.equal(mask[0], torch.ones(3, 5))  # demonstration: every dimension supervised
    assert torch.equal(mask[1, :, :4], torch.ones(3, 4))  # correction: the driven axes stay
    assert torch.equal(mask[1, :, 4], torch.zeros(3))  # ...the held gripper does not


def test_the_two_masks_compose():
    """A correction span is both: fragmentary (so padded) and partially labelled."""
    losses = torch.ones(1, 4, 3)
    pad = torch.tensor([[False, False, False, True]])

    mask = action_supervision_mask(
        losses,
        action_is_pad=pad,
        is_intervention=torch.ones(1, 1),
        intervention_unsupervised_dims=[2],
    )

    assert torch.equal(mask[0, :3, :2], torch.ones(3, 2))
    assert torch.equal(mask[0, :3, 2], torch.zeros(3))  # unlabelled dimension
    assert torch.equal(mask[0, 3], torch.zeros(3))  # padded step
    assert mask.sum().item() == pytest.approx(6.0)


def test_per_dimension_means_use_each_dimensions_own_count():
    """Masking one dimension must not change what the others report.

    `loss_per_dim` is the metric a run is read by; if dropping the gripper on correction frames
    quietly rescaled the translation rows, every comparison across runs would move with the
    fraction of DAgger data rather than with the model.
    """
    losses = torch.full((4, 5, 3), 2.0)
    flags = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    mask = action_supervision_mask(
        losses, is_intervention=flags, intervention_unsupervised_dims=[2]
    )
    per_dim = _masked_mean(losses, mask, dim=[0, 1])

    assert per_dim.tolist() == pytest.approx([2.0, 2.0, 2.0])
    # Half the samples are interventions, so dim 2 kept half the entries.
    assert mask[..., 2].sum().item() == pytest.approx(2 * 5)
    assert mask[..., 0].sum().item() == pytest.approx(4 * 5)


def test_a_fully_padded_reduction_does_not_divide_by_zero():
    """Cannot happen from the dataset -- delta index 0 is always in-episode -- but a NaN loss is
    an unrecoverable training run, so the clamp is worth the one operation."""
    losses = torch.ones(1, 2, 2)
    mask = torch.zeros_like(losses)

    assert _masked_mean(losses, mask).item() == pytest.approx(0.0)
    assert torch.isfinite(_masked_mean(losses, mask, dim=[0, 1])).all()


def test_a_dimension_outside_the_action_is_refused():
    """A stale index would silently mask nothing; the config is worth failing on."""
    losses = torch.ones(1, 2, 5)

    with pytest.raises(ValueError, match="not a dimension"):
        action_supervision_mask(
            losses, is_intervention=torch.ones(1, 1), intervention_unsupervised_dims=[5]
        )


def test_the_intervention_flag_is_accepted_however_it_is_shaped():
    """The dataset column is (B, 1) float32; something upstream may hand over a bare (B,)."""
    losses = torch.ones(2, 3, 4)

    two_d = action_supervision_mask(
        losses, is_intervention=torch.tensor([[1.0], [0.0]]), intervention_unsupervised_dims=[3]
    )
    one_d = action_supervision_mask(
        losses, is_intervention=torch.tensor([1.0, 0.0]), intervention_unsupervised_dims=[3]
    )

    assert torch.equal(two_d, one_d)
