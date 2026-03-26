#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.fr3 import fr3_rank_tactile_hypotheses as rank_hypotheses


def test_load_dataset_profiles_reads_profile_stats_file():
    valid_mask = np.ones((50, 10), dtype=bool)
    profiles = rank_hypotheses._load_dataset_profiles(
        Path(__file__).resolve().parents[2] / "docs/tactile/profile_stats.json",
        valid_mask,
    )

    assert profiles["left"]["frame_count"] == {"mean": 77.0, "std": 1.0}
    assert profiles["right"]["frame_count"] == {"mean": 79.0, "std": 1.0}
    assert profiles["left"]["anisotropy_ratio"]["mean"] > 0.0
    assert profiles["right"]["hot_fraction"]["std"] > 0.0


def test_load_dataset_profiles_falls_back_to_legacy_sequence_payload(tmp_path: Path):
    legacy_payload = {
        "data": [
            {
                "tactiles": {
                    "left": [0.0] * 500,
                    "right": [12.0] * 500,
                }
            },
            {
                "tactiles": {
                    "left": [11.0] * 500,
                    "right": [14.0] * 500,
                }
            },
        ]
    }
    legacy_path = tmp_path / "baseline.json"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    profiles = rank_hypotheses._load_dataset_profiles(legacy_path, np.ones((50, 10), dtype=bool))

    assert profiles["left"]["frame_count"] == {"mean": 1.0, "std": 1.0}
    assert profiles["right"]["frame_count"] == {"mean": 2.0, "std": 1.0}
