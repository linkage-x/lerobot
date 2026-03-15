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

import sys
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@parser.wrap()
def load_train_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


def test_fr3_ee2ee_act_train_config_parses(monkeypatch):
    assert ACTConfig.get_choice_name(ACTConfig) == "act"
    config_path = Path(__file__).resolve().parents[2] / "src/lerobot/configs/franka_research3_ee2ee_act.yaml"
    monkeypatch.setattr(sys, "argv", ["test_train_config", f"--config_path={config_path}"])

    cfg = load_train_config()
    cfg.validate()

    assert cfg.dataset.repo_id == "hph/fr3_pick_place_ee2ee_v1"
    assert cfg.dataset.root is None
    assert cfg.policy.type == "act"
    assert cfg.policy.chunk_size == 50
    assert cfg.policy.n_action_steps == 50
    assert cfg.eval_freq == 0
    assert cfg.wandb.enable is False
