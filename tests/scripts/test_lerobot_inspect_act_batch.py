#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch
from argparse import Namespace

from src.lerobot.scripts import lerobot_inspect_act_batch


def test_build_batch_inspection_report_includes_raw_processed_and_contract_fields():
    raw_batch = {
        "observation.state": torch.tensor([[0.3, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0, 0.2]], dtype=torch.float32),
        "action": torch.tensor(
            [[[0.4, -0.1, 0.6, 0.0, 0.0, 0.0, 1.0, 0.4], [0.5, -0.1, 0.6, 0.0, 0.0, 0.0, 1.0, 0.6]]],
            dtype=torch.float32,
        ),
    }
    relative_batch_before_qoff = {
        "observation.state": torch.tensor([[1.1, -0.9, 0.7, 0.3, -0.4, 0.2, 1.5, 0.2]], dtype=torch.float32),
        "action": torch.tensor(
            [[[0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.4], [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.6]]],
            dtype=torch.float32,
        ),
    }
    processed_batch_after_qoff = {
        "observation.state": torch.tensor([[1.1, -0.9, 0.7, 0.3, -0.4, 0.2, 1.5, 0.2]], dtype=torch.float32),
        "action": torch.tensor(
            [[[1.1, 1.1, 1.1, 0.5, 0.5, 0.5, 0.7, 0.4], [1.2, 1.1, 1.1, 0.5, 0.5, 0.5, 0.7, 0.6]]],
            dtype=torch.float32,
        ),
    }
    masked_batch = {
        "observation.state": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=torch.float32),
        "action": torch.tensor(
            [[[0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.4], [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.6]]],
            dtype=torch.float32,
        ),
    }
    policy_cfg = type(
        "PolicyCfg",
        (),
        {
            "type": "act",
            "relative_ee_action": True,
            "mask_ee_pose_in_state": True,
            "chunk_size": 100,
            "n_action_steps": 100,
            "n_obs_steps": 1,
        },
    )()

    report = lerobot_inspect_act_batch.build_batch_inspection_report(
        raw_batch,
        relative_batch_before_qoff,
        processed_batch_after_qoff,
        masked_batch,
        state_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        action_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        sample_index=0,
        dataset_index=123,
        action_step=1,
        max_action_steps=2,
        policy_cfg=policy_cfg,
        preprocessor_steps=["RenameObservationsProcessorStep", "AbsoluteToRelativeEEActionProcessorStep"],
    )

    assert report["contract"]["relative_ee_action"] is True
    assert report["contract"]["mask_ee_pose_in_state"] is True
    assert report["selection"]["dataset_index"] == 123
    assert report["preprocessor_steps"][-1] == "AbsoluteToRelativeEEActionProcessorStep"
    assert report["raw_observation_state"]["x"] == 0.30000001192092896
    assert report["preprocessed_observation_state"]["x"] == 1.100000023841858
    assert report["masked_observation_state"]["x"] == 0.0
    assert report["masked_observation_state"]["gripper"] == 0.20000000298023224
    assert report["raw_absolute_action"]["x"] == 0.5
    assert report["relative_action_before_qoff"]["x"] == 0.20000000298023224
    assert report["processed_action_after_qoff"]["x"] == 1.2000000476837158
    assert report["processed_relative_action"]["x"] == report["processed_action_after_qoff"]["x"]
    assert len(report["raw_absolute_action_head"]) == 1
    assert report["raw_absolute_action_head"][0]["step"] == 1
    assert report["raw_absolute_action_selected_step_head"] == report["raw_absolute_action_head"]
    assert report["relative_action_before_qoff_selected_step_head"] == report["relative_action_before_qoff_head"]
    assert report["processed_action_after_qoff_selected_step_head"] == report["processed_action_after_qoff_head"]
    assert report["processed_relative_action_selected_step_head"] == report["processed_action_after_qoff_head"]
    assert report["relative_action_before_qoff_head"][0]["values"]["qw"] == 1.0
    assert report["processed_action_after_qoff_head"][0]["values"]["qw"] == 0.699999988079071


def test_build_batch_inspection_report_supports_single_step_actions():
    raw_batch = {
        "observation.state": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.2]], dtype=torch.float32),
        "action": torch.tensor([[0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.103]], dtype=torch.float32),
    }
    relative_batch_before_qoff = {
        "observation.state": torch.tensor([[2.0, 1.0, 0.5, -0.2, 0.1, 0.4, 1.3, 0.2]], dtype=torch.float32),
        "action": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.103]], dtype=torch.float32),
    }
    processed_batch_after_qoff = {
        "observation.state": torch.tensor([[2.0, 1.0, 0.5, -0.2, 0.1, 0.4, 1.3, 0.2]], dtype=torch.float32),
        "action": torch.tensor([[1.1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.103]], dtype=torch.float32),
    }
    masked_batch = {
        "observation.state": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=torch.float32),
        "action": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.103]], dtype=torch.float32),
    }
    policy_cfg = type("PolicyCfg", (), {"type": "act"})()

    report = lerobot_inspect_act_batch.build_batch_inspection_report(
        raw_batch,
        relative_batch_before_qoff,
        processed_batch_after_qoff,
        masked_batch,
        state_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        action_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        sample_index=0,
        dataset_index=0,
        action_step=0,
        max_action_steps=3,
        policy_cfg=policy_cfg,
        preprocessor_steps=["NormalizerProcessorStep"],
    )

    assert report["raw_absolute_action"]["gripper"] == 0.10300000011920929
    assert report["preprocessed_observation_state"]["x"] == 2.0
    assert report["masked_observation_state"]["x"] == 0.0
    assert report["relative_action_before_qoff"]["x"] == 0.10000000149011612
    assert report["processed_action_after_qoff"]["x"] == 1.100000023841858
    assert report["raw_absolute_action_selected_step_head"] == report["raw_absolute_action_head"]
    assert report["relative_action_before_qoff_selected_step_head"] == report["relative_action_before_qoff_head"]
    assert report["processed_action_after_qoff_selected_step_head"] == report["processed_action_after_qoff_head"]
    assert report["raw_absolute_action_head"] == [
        {
            "step": 0,
            "values": {
                "x": 0.4000000059604645,
                "y": 0.10000000149011612,
                "z": 0.20000000298023224,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "gripper": 0.10300000011920929,
            },
        }
    ]


def test_run_preprocessor_stages_splits_before_and_after_qoff():
    class SomeStep:
        def __call__(self, transition):
            transition = dict(transition)
            transition["action"] = transition["action"] + 1.0
            return transition

    class ActionChunkQuantileNormalizerProcessorStep:
        def __call__(self, transition):
            transition = dict(transition)
            transition["action"] = transition["action"] * 10.0
            return transition

    raw_batch = {
        "observation.state": torch.tensor([[1.0]], dtype=torch.float32),
        "action": torch.tensor([[2.0]], dtype=torch.float32),
    }

    preprocessor = type(
        "FakePipeline",
        (),
        {
            "steps": [
                SomeStep(),
                ActionChunkQuantileNormalizerProcessorStep(),
            ],
            "to_transition": staticmethod(lambda data: dict(data)),
            "to_output": staticmethod(lambda transition: dict(transition)),
        },
    )()

    before_qoff, after_qoff = lerobot_inspect_act_batch._run_preprocessor_stages(preprocessor, raw_batch)

    assert float(before_qoff["action"][0, 0]) == pytest.approx(3.0)
    assert float(after_qoff["action"][0, 0]) == pytest.approx(30.0)


def test_feature_names_fall_back_to_fr3_xyzquatgripper_defaults():
    assert lerobot_inspect_act_batch._feature_names("observation.state", None, 8) == [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper",
    ]
    assert lerobot_inspect_act_batch._feature_names("action", None, 8) == [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper",
    ]


def test_build_batch_inspection_summary_aggregates_multiple_reports():
    reports = [
        {
            "contract": {"relative_ee_action": True},
            "selection": {"dataset_index": 0, "action_step": 0, "max_action_steps": 2},
            "preprocessor_steps": ["NormalizerProcessorStep"],
            "raw_observation_state": {"x": 1.0, "gripper": 0.2},
            "preprocessed_observation_state": {"x": 2.0, "gripper": 0.3},
            "masked_observation_state": {"x": 0.0, "gripper": 0.3},
            "raw_absolute_action": {"x": 0.4, "qx": 0.1, "qy": 0.0, "qz": 0.0, "qw": 0.99, "gripper": 0.5},
            "relative_action_before_qoff": {"x": 0.1, "qx": 0.01, "qy": 0.02, "qz": 0.03, "qw": 0.999, "gripper": 0.5},
            "processed_action_after_qoff": {"x": 1.1, "qx": 1.01, "qy": 1.02, "qz": 1.03, "qw": 0.7, "gripper": 0.5},
            "processed_relative_action": {"x": 1.1, "qx": 1.01, "qy": 1.02, "qz": 1.03, "qw": 0.7, "gripper": 0.5},
        },
        {
            "contract": {"relative_ee_action": True},
            "selection": {"dataset_index": 5, "action_step": 0, "max_action_steps": 2},
            "preprocessor_steps": ["NormalizerProcessorStep"],
            "raw_observation_state": {"x": -3.0, "gripper": 0.4},
            "preprocessed_observation_state": {"x": -4.0, "gripper": 0.5},
            "masked_observation_state": {"x": 0.0, "gripper": 0.5},
            "raw_absolute_action": {"x": -0.6, "qx": -0.2, "qy": 0.0, "qz": 0.0, "qw": 0.98, "gripper": 0.5},
            "relative_action_before_qoff": {"x": -0.2, "qx": 0.02, "qy": 0.01, "qz": 0.04, "qw": 0.998, "gripper": 0.5},
            "processed_action_after_qoff": {"x": -1.2, "qx": 0.2, "qy": 0.1, "qz": 0.4, "qw": 0.6, "gripper": 0.5},
            "processed_relative_action": {"x": -1.2, "qx": 0.2, "qy": 0.1, "qz": 0.4, "qw": 0.6, "gripper": 0.5},
        },
    ]

    summary = lerobot_inspect_act_batch.build_batch_inspection_summary(reports)

    assert summary["selection"]["sample_count"] == 2
    assert summary["selection"]["dataset_indices"] == [0, 5]
    assert summary["masked_observation_state_always_zero_keys"] == ["x"]
    assert summary["raw_absolute_action_mean_abs"]["x"] == pytest.approx(0.5)
    assert summary["relative_action_before_qoff_max_abs"]["x"] == pytest.approx(0.2)
    assert summary["processed_action_after_qoff_max_abs"]["x"] == pytest.approx(1.2)
    assert summary["processed_relative_action_max_abs"]["x"] == pytest.approx(1.2)
    assert summary["gripper_delta_mean_abs"] == pytest.approx(0.0)
    assert summary["relative_action_before_qoff_rotation_identity_error_mean_abs"]["qw_error"] == pytest.approx(0.0015)
    assert summary["processed_relative_rotation_identity_error_mean_abs"]["qw_error"] == pytest.approx(0.35)


def test_resolve_sample_indices_supports_range_and_csv_modes():
    args = Namespace(
        sample_indices=None,
        sample_index=10,
        start_index=0,
        num_samples=3,
        sample_stride=5,
    )
    assert lerobot_inspect_act_batch._resolve_sample_indices(args, dataset_length=100) == [10, 15, 20]

    args.sample_indices = "1, 4,9"
    assert lerobot_inspect_act_batch._resolve_sample_indices(args, dataset_length=100) == [1, 4, 9]
