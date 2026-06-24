#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.fr3 import fr3_train_il_policy


def test_selected_state_names_supports_feature_name_selector():
    features = {
        "observation.state.right": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"],
        },
        "observation.state_raw": {
            "dtype": "float32",
            "shape": [2],
            "names": [
                "handheld_gripper.pika_right.width_mm",
                "handheld_gripper.pika_left.width_mm",
            ],
        },
    }

    names = fr3_train_il_policy.selected_state_names(
        features,
        [
            "observation.state.right",
            "observation.state_raw:handheld_gripper.pika_left.width_mm",
        ],
    )

    assert names == [
        "observation.state.right.ee.x",
        "observation.state.right.ee.y",
        "observation.state.right.ee.z",
        "observation.state.right.ee.qx",
        "observation.state.right.ee.qy",
        "observation.state.right.ee.qz",
        "observation.state.right.ee.qw",
        "observation.state_raw.handheld_gripper.pika_left.width_mm",
    ]


def test_select_state_matrix_concatenates_full_features_and_selected_dims():
    features = {
        "observation.state.right": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"],
        },
        "observation.state_raw": {
            "dtype": "float32",
            "shape": [2],
            "names": [
                "handheld_gripper.pika_right.width_mm",
                "handheld_gripper.pika_left.width_mm",
            ],
        },
    }
    df = pd.DataFrame(
        {
            "observation.state.right": [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [1.1, 1.2, 1.3, 0.1, 0.2, 0.3, 0.9],
            ],
            "observation.state_raw": [
                [60.0, 61.0],
                [70.0, 71.0],
            ],
        }
    )

    state = fr3_train_il_policy.select_state_matrix(
        df,
        features,
        [
            "observation.state.right",
            "observation.state_raw:handheld_gripper.pika_left.width_mm",
        ],
    )

    assert state.dtype == np.float32
    assert state.shape == (2, 8)
    assert np.allclose(state[:, :7], np.asarray(df["observation.state.right"].tolist(), dtype=np.float32))
    assert np.allclose(state[:, 7], [61.0, 71.0])
