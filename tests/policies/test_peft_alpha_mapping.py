# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""`PeftConfig.alpha` reaches the adapter under the name that method actually uses.

An adapter's strength is alpha / r. Before `alpha` existed on PeftConfig there was no way to
set it, so it always landed on PEFT's fixed default of 8 and raising `r` quietly *weakened*
every update -- r=32 gives a scaling of 0.25. These check the rename that makes it reachable,
without building a policy, since the mapping does not touch `self`.
"""

import pytest

from lerobot.policies.pretrained import PreTrainedPolicy

pytest.importorskip("peft")


def preprocess(overrides, method="LORA"):
    from peft import PeftType

    return PreTrainedPolicy._preprocess_peft_cli_overrides(None, overrides, PeftType[method])


def test_alpha_is_renamed_to_the_key_lora_understands():
    out = preprocess({"r": 32, "alpha": 32})

    assert out == {"r": 32, "lora_alpha": 32}
    assert "alpha" not in out  # LoraConfig would reject it


def test_an_unset_alpha_is_left_out_entirely():
    """A None must not be forwarded: `_build_peft_config` only skips None *after* it would
    have overwritten the policy's own default, so the key has to be absent, not null."""
    out = preprocess({"r": 32, "alpha": None})

    assert "lora_alpha" not in out
    assert "alpha" not in out


def test_alpha_is_refused_rather_than_dropped_for_a_method_that_has_no_such_knob():
    with pytest.raises(ValueError, match="alpha"):
        preprocess({"r": 32, "alpha": 32}, method="MISS")


def test_the_other_renames_still_apply():
    out = preprocess({"full_training_modules": [], "init_type": "gaussian", "method_type": "LORA"})

    assert out == {"modules_to_save": [], "init_lora_weights": "gaussian"}
