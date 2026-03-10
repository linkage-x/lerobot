#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.calibration.fr3_teleop import (
    build_default_trace_profile,
    build_trace_bundle,
    compare_translation_traces,
    load_trace_bundle,
    make_trace_sample,
    save_trace_bundle,
)


def test_default_trace_profile_contains_xyz_translation_segments():
    profile = build_default_trace_profile(fps=60)

    assert profile["name"] == "default_xyz_translation_pulses"
    assert len(profile["actions"]) > 0
    segment_names = [segment["name"] for segment in profile["segments"]]
    assert "x_plus" in segment_names
    assert "x_minus" in segment_names
    assert "y_plus" in segment_names
    assert "z_minus" in segment_names


def test_compare_translation_traces_returns_axis_scale_multiplier():
    profile = build_default_trace_profile(fps=10, move_s=0.2, hold_s=0.1, warmup_s=0.1, settle_s=0.1)
    samples_reference = []
    samples_measured = []
    for sample_index in range(len(profile["actions"]) + 1):
        samples_reference.append(
            make_trace_sample(
                profile_step=sample_index - 1,
                scheduled_time_s=sample_index / 10.0,
                measured_time_s=sample_index / 10.0,
                action=None,
                joint_positions=[0.0] * 7,
                ee_position=[sample_index * 0.01, 0.0, 0.0],
                ee_rotvec=[0.0, 0.0, 0.0],
                gripper=1.0,
            )
        )
        samples_measured.append(
            make_trace_sample(
                profile_step=sample_index - 1,
                scheduled_time_s=sample_index / 10.0,
                measured_time_s=sample_index / 10.0,
                action=None,
                joint_positions=[0.0] * 7,
                ee_position=[sample_index * 0.005, 0.0, 0.0],
                ee_rotvec=[0.0, 0.0, 0.0],
                gripper=1.0,
            )
        )

    reference = build_trace_bundle(mode="sim", profile=profile, samples=samples_reference)
    measured = build_trace_bundle(mode="hardware", profile=profile, samples=samples_measured)
    result = compare_translation_traces(reference, measured)

    assert result["axis_summaries"]["x"]["suggested_scale_multiplier"] == 2.0


def test_compare_translation_traces_detects_direction_mismatch():
    profile = build_default_trace_profile(fps=10, move_s=0.2, hold_s=0.1, warmup_s=0.1, settle_s=0.1)
    samples_reference = []
    samples_measured = []
    for sample_index in range(len(profile["actions"]) + 1):
        samples_reference.append(
            make_trace_sample(
                profile_step=sample_index - 1,
                scheduled_time_s=sample_index / 10.0,
                measured_time_s=sample_index / 10.0,
                action=None,
                joint_positions=[0.0] * 7,
                ee_position=[sample_index * 0.01, 0.0, 0.0],
                ee_rotvec=[0.0, 0.0, 0.0],
                gripper=1.0,
            )
        )
        samples_measured.append(
            make_trace_sample(
                profile_step=sample_index - 1,
                scheduled_time_s=sample_index / 10.0,
                measured_time_s=sample_index / 10.0,
                action=None,
                joint_positions=[0.0] * 7,
                ee_position=[-sample_index * 0.01, 0.0, 0.0],
                ee_rotvec=[0.0, 0.0, 0.0],
                gripper=1.0,
            )
        )

    reference = build_trace_bundle(mode="sim", profile=profile, samples=samples_reference)
    measured = build_trace_bundle(mode="hardware", profile=profile, samples=samples_measured)
    result = compare_translation_traces(reference, measured)

    assert result["axis_summaries"]["x"]["suggested_scale_multiplier"] is None


def test_trace_bundle_round_trip(tmp_path: Path):
    profile = build_default_trace_profile(fps=10)
    bundle = build_trace_bundle(
        mode="sim",
        profile=profile,
        samples=[
            make_trace_sample(
                profile_step=-1,
                scheduled_time_s=0.0,
                measured_time_s=0.0,
                action=None,
                joint_positions=[0.0] * 7,
                ee_position=[0.0, 0.0, 0.0],
                ee_rotvec=[0.0, 0.0, 0.0],
                gripper=1.0,
            )
        ],
    )

    output_path = tmp_path / "trace.json"
    save_trace_bundle(output_path, bundle)
    loaded = load_trace_bundle(output_path)

    assert loaded["metadata"]["mode"] == "sim"
    assert loaded["samples"][0]["profile_step"] == -1
