#!/usr/bin/env python

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

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from huggingface_hub import DatasetCard

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from lerobot.datasets.utils import (
    combine_feature_dicts,
    create_lerobot_dataset_card,
    hf_transform_to_torch,
    load_tasks,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES


def test_default_parameters():
    card = create_lerobot_dataset_card()
    assert isinstance(card, DatasetCard)
    assert card.data.tags == ["LeRobot"]
    assert card.data.task_categories == ["robotics"]
    assert card.data.configs == [
        {
            "config_name": "default",
            "data_files": "data/*/*.parquet",
        }
    ]


def test_with_tags():
    tags = ["tag1", "tag2"]
    card = create_lerobot_dataset_card(tags=tags)
    assert card.data.tags == ["LeRobot", "tag1", "tag2"]


def test_calculate_episode_data_index():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    assert torch.equal(episode_data_index["from"], torch.tensor([0, 2, 3]))
    assert torch.equal(episode_data_index["to"], torch.tensor([2, 3, 6]))


def test_merge_simple_vectors():
    g1 = {
        ACTION: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["ee.x", "ee.y"],
        }
    }
    g2 = {
        ACTION: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["ee.y", "ee.z"],
        }
    }

    out = combine_feature_dicts(g1, g2)

    assert ACTION in out
    assert out[ACTION]["dtype"] == "float32"
    # Names merged with preserved order and de-dupuplication
    assert out[ACTION]["names"] == ["ee.x", "ee.y", "ee.z"]
    # Shape correctly recomputed from names length
    assert out[ACTION]["shape"] == (3,)


def test_merge_multiple_groups_order_and_dedup():
    g1 = {ACTION: {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}}
    g2 = {ACTION: {"dtype": "float32", "shape": (2,), "names": ["b", "c"]}}
    g3 = {ACTION: {"dtype": "float32", "shape": (3,), "names": ["a", "c", "d"]}}

    out = combine_feature_dicts(g1, g2, g3)

    assert out[ACTION]["names"] == ["a", "b", "c", "d"]
    assert out[ACTION]["shape"] == (4,)


def test_non_vector_last_wins_for_images():
    # Non-vector (images) with same name should be overwritten by the last image specified
    g1 = {
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    }
    g2 = {
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (3, 720, 1280),
            "names": ["channels", "height", "width"],
        }
    }

    out = combine_feature_dicts(g1, g2)
    assert out[f"{OBS_IMAGES}.front"]["shape"] == (3, 720, 1280)
    assert out[f"{OBS_IMAGES}.front"]["dtype"] == "image"


def test_dtype_mismatch_raises():
    g1 = {ACTION: {"dtype": "float32", "shape": (1,), "names": ["a"]}}
    g2 = {ACTION: {"dtype": "float64", "shape": (1,), "names": ["b"]}}

    with pytest.raises(ValueError, match="dtype mismatch for 'action'"):
        _ = combine_feature_dicts(g1, g2)


def test_non_dict_passthrough_last_wins():
    g1 = {"misc": 123}
    g2 = {"misc": 456}

    out = combine_feature_dicts(g1, g2)
    # For non-dict entries the last one wins
    assert out["misc"] == 456


def test_load_tasks_falls_back_to_legacy_jsonl(tmp_path: Path):
    legacy_tasks_path = tmp_path / "meta" / "tasks.jsonl"
    legacy_tasks_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_tasks = [
        {"task_index": 1, "task": "pick"},
        {"task_index": 0, "task": "approach"},
    ]
    legacy_tasks_path.write_text("".join(json.dumps(item) + "\n" for item in legacy_tasks))

    tasks = load_tasks(tmp_path)

    assert list(tasks.index) == ["pick", "approach"]
    assert tasks.index.name == "task"
    assert tasks["task_index"].tolist() == [1, 0]


def test_metadata_file_path_falls_back_to_six_digit_file_index(tmp_path: Path):
    data_file = tmp_path / "data" / "chunk-000" / "file-000000.parquet"
    video_file = tmp_path / "videos" / "observation.images.left" / "chunk-000" / "file-000000.mp4"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.touch()
    video_file.touch()

    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.root = tmp_path
    meta.info = {
        "codebase_version": "v3.0",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    meta.episodes = [
        {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.left/chunk_index": 0,
            "videos/observation.images.left/file_index": 0,
        }
    ]

    assert meta.get_data_file_path(0) == Path("data/chunk-000/file-000000.parquet")
    assert meta.get_video_file_path(0, "observation.images.left") == Path(
        "videos/observation.images.left/chunk-000/file-000000.mp4"
    )
