#!/usr/bin/env python

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def test_pretrained_pi05_processors_reload_paligemma_by_repo_id(monkeypatch):
    calls = []

    def fake_from_pretrained(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "lerobot.policies.factory.PolicyProcessorPipeline.from_pretrained",
        fake_from_pretrained,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=PI05Config(),
        pretrained_path="/home/tele/Models/pi05_base",
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    assert preprocessor is not postprocessor
    tokenizer_overrides = calls[0]["overrides"]["tokenizer_processor"]
    assert tokenizer_overrides["tokenizer_name"] == "google/paligemma-3b-pt-224"
    assert calls[0]["overrides"]["device_processor"] == {"device": "cuda"}
    assert "tokenizer_processor" not in calls[1]["overrides"]


def test_pretrained_pi0_processors_keep_an_explicit_tokenizer_override(monkeypatch):
    calls = []

    def fake_from_pretrained(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "lerobot.policies.factory.PolicyProcessorPipeline.from_pretrained",
        fake_from_pretrained,
    )

    make_pre_post_processors(
        policy_cfg=PI0Config(),
        pretrained_path="lerobot/pi0_base",
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": "local/test-tokenizer"}},
    )

    tokenizer_overrides = calls[0]["overrides"]["tokenizer_processor"]
    assert tokenizer_overrides["tokenizer_name"] == "local/test-tokenizer"
