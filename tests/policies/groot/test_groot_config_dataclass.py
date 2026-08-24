#!/usr/bin/env python

"""`config_dataclass` keeps `lerobot.policies` importable across transformers majors.

Gr00t is imported eagerly by `lerobot/policies/__init__.py`, so a Gr00t config class that
cannot be constructed is not a Gr00t problem -- it takes down `lerobot.policies.factory`, and
with it every `lerobot_train` run of every policy. These tests pin the one line that decides
whether that happens, because the failure only appears on a transformers upgrade and appears
as an import error thousands of lines away from its cause.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field

import pytest

from lerobot.policies.groot.utils import config_dataclass


class _PlainBase:
    """Stands in for transformers<5 `PretrainedConfig`: not a dataclass, nothing applied yet."""


@dataclasses.dataclass
class _DataclassBase:
    """Stands in for transformers>=5 `PretrainedConfig`: a dataclass whose fields all default."""

    transformers_version: str | None = None
    dtype: str | None = None


def test_the_decorator_still_runs_when_the_base_is_a_plain_class():
    """transformers 4: nothing has processed the class body, so `@dataclass` is still needed."""

    @config_dataclass
    class unprocessed(_PlainBase):  # noqa: N801
        backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
        compute_dtype: str = field(default="float32")

    assert dataclasses.is_dataclass(unprocessed)
    assert {f.name: f.init for f in dataclasses.fields(unprocessed)} == {
        "backbone_cfg": False,
        "compute_dtype": True,
    }


def test_the_decorator_stands_aside_when_the_base_already_applied_it():
    """transformers 5: `PretrainedConfig.__init_subclass__` has already run `@dataclass`.

    Applying it again re-reads the class body with the `field(init=False)` sentinels already
    consumed, so the fields come back as required init arguments sitting after the base's
    defaulted ones -- which is a TypeError at import time, as the second half asserts.
    """

    # The decorator here stands in for the base's __init_subclass__, which is what a real
    # transformers 5 applies, so the test does not depend on which transformers is installed.
    @dataclasses.dataclass
    class already_processed(_DataclassBase):  # noqa: N801
        backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
        compute_dtype: str = field(default="float32")

    assert config_dataclass(already_processed) is already_processed
    assert {f.name for f in dataclasses.fields(already_processed)} >= {
        "backbone_cfg",
        "compute_dtype",
    }

    # And this is the failure being avoided: a second pass over a class the base already
    # processed, which is exactly what `@dataclass` on the class body amounts to under
    # transformers 5. The sentinel is gone by then, so `backbone_cfg` comes back required.
    with pytest.raises(TypeError, match="non-default argument"):
        dataclasses.dataclass(already_processed)


def test_the_real_groot_configs_are_constructible():
    """Whatever transformers is installed, these two classes must import and instantiate."""
    from lerobot.policies.groot.action_head.flow_matching_action_head import (
        FlowmatchingActionHeadConfig,
    )
    from lerobot.policies.groot.groot_n1 import GR00TN15Config

    config = GR00TN15Config(
        backbone_cfg={"tune_llm": False},
        action_head_cfg={},
        action_horizon=16,
        action_dim=32,
    )
    assert config.backbone_cfg == {"tune_llm": False}
    assert config.action_horizon == 16
    # Defaulted on the class, so it is present without being passed.
    assert config.compute_dtype == "float32"

    assert dataclasses.is_dataclass(FlowmatchingActionHeadConfig)


def test_the_policy_package_imports():
    """The reason the above matters: this import is on the path of every training run."""
    import lerobot.policies

    assert lerobot.policies.PI05Config is not None
