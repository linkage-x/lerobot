#!/usr/bin/env python

import pytest

from lerobot.utils.state_feature_names import (
    flatten_feature_name_paths,
    get_ee_pose_state_indices,
    resolve_feature_name_indices,
)


def test_flatten_feature_name_paths_preserves_nested_metadata_order():
    feature_names = {
        "arm": {
            "pose": ["x", "y", "z"],
            "quat": ["qx", "qy", "qz", "qw"],
        },
        "gripper": ["open"],
    }

    assert flatten_feature_name_paths(feature_names) == [
        "arm/pose/x",
        "arm/pose/y",
        "arm/pose/z",
        "arm/quat/qx",
        "arm/quat/qy",
        "arm/quat/qz",
        "arm/quat/qw",
        "gripper/open",
    ]


def test_get_ee_pose_state_indices_supports_grouped_bare_state_names():
    indices = get_ee_pose_state_indices({"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]})

    assert indices == {
        "ee.x": 0,
        "ee.y": 1,
        "ee.z": 2,
        "ee.qx": 3,
        "ee.qy": 4,
        "ee.qz": 5,
        "ee.qw": 6,
    }


def test_resolve_feature_name_indices_reports_missing_targets_clearly():
    with pytest.raises(ValueError, match=r"Could not resolve required feature names \['ee.qz', 'ee.qw'\]"):
        resolve_feature_name_indices(
            ["x", "y", "z", "qx", "qy"],
            {
                "ee.x": ("ee.x", "x"),
                "ee.y": ("ee.y", "y"),
                "ee.z": ("ee.z", "z"),
                "ee.qx": ("ee.qx", "qx"),
                "ee.qy": ("ee.qy", "qy"),
                "ee.qz": ("ee.qz", "qz"),
                "ee.qw": ("ee.qw", "qw"),
            },
            strict=True,
        )
