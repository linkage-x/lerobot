from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", exc_type=ImportError)

REPO_ROOT = Path(__file__).resolve().parents[3]
PI_GEMMA_PATH = REPO_ROOT / "src" / "lerobot" / "policies" / "pi_gemma.py"
spec = importlib.util.spec_from_file_location("pi_gemma_under_test", PI_GEMMA_PATH)
assert spec is not None and spec.loader is not None
pi_gemma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi_gemma)


def _install_fake_mask(monkeypatch, fake):
    parameters = inspect.signature(fake).parameters
    monkeypatch.setattr(pi_gemma, "create_causal_mask", fake)
    monkeypatch.setattr(pi_gemma, "_CREATE_CAUSAL_MASK_PARAMETER_NAMES", set(parameters))
    monkeypatch.setattr(
        pi_gemma,
        "_CREATE_CAUSAL_MASK_ACCEPTS_VAR_KEYWORD",
        any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()),
    )


def test_causal_mask_compat_filters_cache_position_when_transformers_does_not_accept_it(monkeypatch):
    calls = {}

    def fake_create_causal_mask(config, inputs_embeds, attention_mask, past_key_values=None, position_ids=None):
        calls.update(
            {
                "config": config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )
        return "mask"

    _install_fake_mask(monkeypatch, fake_create_causal_mask)
    inputs = torch.zeros(1, 2, 3)
    positions = torch.arange(2).unsqueeze(0)

    mask = pi_gemma._create_causal_mask_compat(
        config="cfg",
        inputs_embeds=inputs,
        attention_mask=None,
        cache_position=torch.arange(2),
        past_key_values="cache",
        position_ids=positions,
    )

    assert mask == "mask"
    assert calls["config"] == "cfg"
    assert calls["inputs_embeds"] is inputs
    assert calls["attention_mask"] is None
    assert calls["past_key_values"] == "cache"
    assert calls["position_ids"] is positions


def test_causal_mask_compat_keeps_cache_position_when_transformers_accepts_it(monkeypatch):
    calls = {}

    def fake_create_causal_mask(
        config, inputs_embeds, attention_mask, cache_position=None, past_key_values=None, position_ids=None
    ):
        calls["cache_position"] = cache_position
        return "mask"

    _install_fake_mask(monkeypatch, fake_create_causal_mask)
    cache_position = torch.arange(2)

    mask = pi_gemma._create_causal_mask_compat(
        config="cfg",
        inputs_embeds=torch.zeros(1, 2, 3),
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=None,
    )

    assert mask == "mask"
    assert calls["cache_position"] is cache_position
