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

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.scripts import lerobot_clean_episodes


def test_parse_episode_indices_accepts_common_forms():
    assert lerobot_clean_episodes.parse_episode_indices(["1", "2", "3"]) == [1, 2, 3]
    assert lerobot_clean_episodes.parse_episode_indices(["1,2,3"]) == [1, 2, 3]
    assert lerobot_clean_episodes.parse_episode_indices(["[1, 2, 3]"]) == [1, 2, 3]


def test_parse_episode_indices_deduplicates_preserving_order():
    assert lerobot_clean_episodes.parse_episode_indices(["3", "1", "3", "2"]) == [3, 1, 2]


@pytest.mark.parametrize("values", [["-1"], ["bad"], ["[]"], ["True"], ["[1.5]"]])
def test_parse_episode_indices_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        lerobot_clean_episodes.parse_episode_indices(values)


def test_main_dry_run_does_not_call_delete_handler(monkeypatch, capsys):
    dataset = _fake_dataset()
    monkeypatch.setattr(lerobot_clean_episodes, "LeRobotDataset", lambda *_args, **_kwargs: dataset)

    def fail_delete_handler(_cfg):
        raise AssertionError("delete handler should not run during dry-run")

    monkeypatch.setattr(lerobot_clean_episodes, "handle_delete_episodes", fail_delete_handler)

    result = lerobot_clean_episodes.main(
        [
            "--repo-id",
            "test/repo",
            "--root",
            "/tmp/test_repo",
            "--episodes",
            "1,3",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Delete episodes: [1, 3]" in captured.out
    assert "Dry run: no files were changed." in captured.out


def test_main_calls_lerobot_edit_dataset_delete_handler(monkeypatch):
    dataset = _fake_dataset()
    captured_configs = []
    monkeypatch.setattr(lerobot_clean_episodes, "LeRobotDataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(lerobot_clean_episodes, "handle_delete_episodes", captured_configs.append)

    result = lerobot_clean_episodes.main(
        [
            "--repo-id",
            "test/repo",
            "--root",
            "/tmp/test_repo",
            "--episodes",
            "1",
            "3",
            "--new-repo-id",
            "test/repo_clean",
            "--yes",
        ]
    )

    assert result == 0
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.repo_id == "test/repo"
    assert cfg.root == "/tmp/test_repo"
    assert cfg.new_repo_id == "test/repo_clean"
    assert cfg.operation.episode_indices == [1, 3]


def test_main_rejects_invalid_episode_before_delete(monkeypatch):
    dataset = _fake_dataset()
    monkeypatch.setattr(lerobot_clean_episodes, "LeRobotDataset", lambda *_args, **_kwargs: dataset)

    with pytest.raises(ValueError, match="Invalid episode indices"):
        lerobot_clean_episodes.main(
            [
                "--repo-id",
                "test/repo",
                "--episodes",
                "9",
                "--yes",
            ]
        )


def _fake_dataset():
    return SimpleNamespace(
        repo_id="test/repo",
        root=Path("/tmp/test_repo"),
        meta=SimpleNamespace(
            total_episodes=4,
            total_frames=40,
            episodes=[
                {"length": 10},
                {"length": 10},
                {"length": 10},
                {"length": 10},
            ],
        ),
    )
