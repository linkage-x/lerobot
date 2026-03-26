#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.fr3 import fr3_das_replay_real_runtime
from lerobot.utils.constants import ACTION, OBS_STATE


def test_load_episode_uses_public_dataset_api(tmp_path: Path, empty_lerobot_dataset_factory):
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["ax", "ay"]},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "fr3_runtime_dataset", features=features, use_videos=False)

    for episode_idx in range(2):
        for frame_idx in range(2):
            dataset.add_frame(
                {
                    OBS_STATE: torch.tensor([episode_idx, frame_idx], dtype=torch.float32),
                    ACTION: torch.tensor([episode_idx + 0.5, frame_idx + 1.5], dtype=torch.float32),
                    "task": f"task_{episode_idx}",
                }
            )
        dataset.save_episode()

    dataset.finalize()
    loaded = fr3_das_replay_real_runtime.load_episode(str(dataset.root), 1)

    assert loaded.keys() == {"state", "action", "timestamp"}
    assert loaded["state"].dtype == np.float64
    assert loaded["action"].dtype == np.float64
    assert loaded["timestamp"].dtype == np.float64
    assert np.allclose(loaded["state"], np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64))
    assert np.allclose(loaded["action"], np.array([[1.5, 1.5], [1.5, 2.5]], dtype=np.float64))
    assert np.allclose(loaded["timestamp"], np.array([0.0, 1.0 / 30.0], dtype=np.float64))
