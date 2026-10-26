"""The PiGemma classes build what their transformers parents throw away on meta, and end up the same model.

Built for real, those parts were two more fp32 copies of the language model and a pi0.5 load peaked
at ~39 GB anon (16.7 GB after). The reference here is the same class with the meta build switched
off, which is exactly the old construction.
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", exc_type=ImportError)

from transformers import CONFIG_MAPPING  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
PI_GEMMA_PATH = REPO_ROOT / "src" / "lerobot" / "policies" / "pi_gemma.py"
spec = importlib.util.spec_from_file_location("pi_gemma_under_test", PI_GEMMA_PATH)
assert spec is not None and spec.loader is not None
pi_gemma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi_gemma)


def _paligemma_config():
    c = CONFIG_MAPPING["paligemma"]()
    c._vocab_size = 1000  # noqa: SLF001
    c.image_token_index = 999
    t = c.text_config
    t.hidden_size, t.intermediate_size, t.num_attention_heads, t.head_dim = 64, 128, 2, 32
    t.num_hidden_layers, t.num_key_value_heads, t.vocab_size = 2, 1, 1000
    t.use_adarms, t.adarms_cond_dim = False, None
    v = c.vision_config
    v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads = 32, 64, 2, 2
    v.image_size, v.patch_size, v.projection_dim = 28, 14, 64
    return c


def _expert_config():
    return CONFIG_MAPPING["gemma"](
        head_dim=16, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2,
        num_key_value_heads=1, vocab_size=1000, use_adarms=True, adarms_cond_dim=32,
    )  # fmt: skip


BUILDS = {
    "paligemma": lambda: pi_gemma.PaliGemmaForConditionalGenerationWithPiGemma(_paligemma_config()),
    "expert": lambda: pi_gemma.PiGemmaForCausalLM(_expert_config()),
}


def _built_for_real(monkeypatch, build):
    with monkeypatch.context() as m:
        m.setattr(pi_gemma, "_builds_on_meta", lambda namespace, name: contextlib.nullcontext())
        return build()


def _shared(model):
    first: dict[int, str] = {}
    pairs = set()
    for k, t in model.state_dict(keep_vars=True).items():
        if first.setdefault(t.data_ptr(), k) != k:
            pairs.add((first[t.data_ptr()], k))
    return pairs


@pytest.mark.parametrize("which", list(BUILDS))
def test_the_meta_build_makes_the_same_model(monkeypatch, which):
    torch.manual_seed(0)
    new = BUILDS[which]()
    old = _built_for_real(monkeypatch, BUILDS[which])

    sn, so = new.state_dict(), old.state_dict()
    assert list(sn) == list(so)
    assert all(sn[k].shape == so[k].shape and sn[k].dtype == so[k].dtype for k in sn)
    # By name: the PiGemma decoder layer class is made afresh by a factory on every build.
    assert [(n, type(x).__qualname__) for n, x in new.named_modules()] == [
        (n, type(x).__qualname__) for n, x in old.named_modules()
    ]
    assert all(p.device.type == "cpu" for p in (*new.parameters(), *new.buffers()))
    # Buffers no checkpoint restores (rotary frequencies, embed scale, position ids) must be computed alike.
    bn, bo = dict(new.named_buffers()), dict(old.named_buffers())
    assert list(bn) == list(bo)
    non_persistent = [k for k in bn if k not in sn]
    assert non_persistent
    assert all(torch.equal(bn[k], bo[k]) and bn[k].dtype == bo[k].dtype for k in non_persistent)
    # The head was never tied to the embedding the model uses, and checkpoints load both.
    assert _shared(new) == _shared(old) == set()


def test_the_language_model_layers_are_the_pi_gemma_ones():
    model = BUILDS["paligemma"]()
    lm = model.model.language_model
    assert type(lm) is pi_gemma.PiGemmaModel
    assert all(type(layer).__name__ == "_PiGemmaDecoderLayerBase" for layer in lm.layers)
    assert isinstance(lm.norm, pi_gemma.PiGemmaRMSNorm)


def test_the_swapped_global_comes_back_even_when_the_build_fails():
    real = pi_gemma.modeling_gemma.GemmaDecoderLayer
    with pytest.raises(ValueError), pi_gemma._builds_on_meta(pi_gemma.modeling_gemma, "GemmaDecoderLayer"):
        assert pi_gemma.modeling_gemma.GemmaDecoderLayer is not real
        raise ValueError
    assert pi_gemma.modeling_gemma.GemmaDecoderLayer is real


def test_what_is_built_under_it_holds_no_memory():
    with pi_gemma._builds_on_meta(torch.nn, "Linear"):
        layer = torch.nn.Linear(4096, 4096)
    assert layer.weight.is_meta
    assert torch.nn.Linear(2, 2).weight.device.type == "cpu"


def test_a_model_left_with_a_meta_tensor_is_refused():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.device("meta"):
        model.append(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="meta tensors"):
        pi_gemma._assert_nothing_on_meta(model)
