#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", exc_type=ImportError)

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def test_rtc_processor_accepts_broadcast_tensor_time():
    processor = RTCProcessor(
        RTCConfig(
            enabled=True,
            execution_horizon=3,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    x_t = torch.zeros(1, 5, 2, dtype=torch.float32)
    prev_chunk_left_over = torch.ones(1, 4, 2, dtype=torch.float32)
    time = torch.full((1, 1, 1), 0.5, dtype=torch.float32)

    def original_denoise_step(input_x_t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(input_x_t, 0.25)

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=2,
        time=time,
        original_denoise_step_partial=original_denoise_step,
        execution_horizon=3,
    )

    assert result.shape == x_t.shape
    assert result.dtype == x_t.dtype
    assert torch.isfinite(result).all()


def test_rtc_processor_still_accepts_python_float_time():
    processor = RTCProcessor(
        RTCConfig(
            enabled=True,
            execution_horizon=3,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    x_t = torch.zeros(1, 5, 2, dtype=torch.float32)
    prev_chunk_left_over = torch.ones(1, 4, 2, dtype=torch.float32)

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=2,
        time=0.5,
        original_denoise_step_partial=lambda input_x_t: torch.full_like(input_x_t, 0.25),
        execution_horizon=3,
    )

    assert result.shape == x_t.shape
    assert torch.isfinite(result).all()
