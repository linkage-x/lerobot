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

"""Contract tests for the rig-independent capture-timestamp measurement."""

from __future__ import annotations

import numpy as np
import pytest

from tools.shared import capture_timestamp_audit as audit

# The FR3 device set, which is what makes the median a bad anchor: with two cameras sitting
# ahead of the arm, the median across four devices lands between a camera and the arm rather
# than on the arm.
NAMES = [
    "fr3.arm.capture_timestamp_s",
    "pika_gripper.capture_timestamp_s",
    "camera.ee.capture_timestamp_s",
    "camera.side.capture_timestamp_s",
]


def _frames(count: int, *, camera_offset_s: float, arm_drift_s: float = 0.0):
    """Frames where both cameras sit a fixed distance ahead of an on-demand arm read."""
    grid = np.arange(count, dtype=np.float64) / 30.0
    arm = grid + arm_drift_s * np.arange(count)
    capture = np.stack(
        [arm, arm + 0.00004, arm - camera_offset_s, arm - camera_offset_s * 0.98], axis=1
    )
    return capture, grid


def test_grid_lag_reference_defaults_to_the_median_across_devices():
    """A rig with no on-demand device has no better anchor; this is the handheld behaviour."""
    capture, grid = _frames(50, camera_offset_s=0.025)

    metrics = audit.compute_frame_metrics(capture, grid)

    expected = np.median(capture, axis=1) - grid
    assert np.allclose(metrics.grid_lag_s, expected)


def test_grid_lag_against_a_named_device_ignores_other_devices_offsets():
    """A camera's honest latency must not be charged to the control loop's cadence.

    This is the failure the split fixed: cameras 25 ms ahead dragged the median to -12 ms and
    reported 13.5 ms of grid lag for a loop whose cadence was exact.
    """
    capture, grid = _frames(50, camera_offset_s=0.025)
    arm_index = audit.resolve_grid_lag_reference_index(NAMES, ("fr3.arm.",))
    assert arm_index == 0

    anchored = audit.compute_frame_metrics(capture, grid, grid_lag_reference_index=arm_index)
    median_based = audit.compute_frame_metrics(capture, grid)

    # The arm sits exactly on the grid here, so the loop shows no lag at all...
    assert np.allclose(anchored.grid_lag_s, 0.0, atol=1e-12)
    # ...while the median reading is dominated by the camera offset.
    assert abs(np.median(median_based.grid_lag_s)) > 0.010


def test_grid_lag_still_reports_real_control_loop_drift():
    """Anchoring must not make the metric blind to the thing it exists to catch."""
    capture, grid = _frames(50, camera_offset_s=0.025, arm_drift_s=0.001)
    arm_index = audit.resolve_grid_lag_reference_index(NAMES, ("fr3.arm.",))

    metrics = audit.compute_frame_metrics(capture, grid, grid_lag_reference_index=arm_index)

    # 1 ms of slip per frame accumulates over the episode and must show up.
    assert metrics.grid_lag_s[-1] == pytest.approx(0.001 * 49, abs=1e-9)


def test_unknown_reference_prefix_falls_back_to_the_median():
    assert audit.resolve_grid_lag_reference_index(NAMES, ("nosuch.",)) is None
    assert audit.resolve_grid_lag_reference_index(NAMES, ()) is None


def test_skew_is_independent_of_the_grid_reference():
    """Skew is a within-frame spread; the anchor cannot change it."""
    capture, grid = _frames(30, camera_offset_s=0.025)

    a = audit.compute_frame_metrics(capture, grid)
    b = audit.compute_frame_metrics(capture, grid, grid_lag_reference_index=0)

    assert np.allclose(a.max_skew_s, b.max_skew_s)


def test_non_finite_frames_are_excluded_rather_than_poisoning_the_statistics():
    capture, grid = _frames(20, camera_offset_s=0.01)
    capture[5, 2] = np.nan

    metrics = audit.compute_frame_metrics(capture, grid, grid_lag_reference_index=0)

    assert not metrics.finite_mask[5]
    assert np.isnan(metrics.max_skew_s[5])
    assert np.isfinite(metrics.max_skew_s[np.arange(20) != 5]).all()


def test_measured_interval_is_elapsed_time_not_the_median_gap():
    """Asymmetric jitter makes the median gap read high; elapsed time is the honest answer."""
    nominal = 1.0 / 30.0
    gaps = np.where(np.arange(299) % 3 == 2, nominal * 0.86, nominal * 1.07)
    centres = np.concatenate([[0.0], np.cumsum(gaps)])
    capture = np.repeat(centres[:, None], 3, axis=1)

    measured = audit.measured_frame_interval_s(capture)

    assert measured == pytest.approx((centres[-1] - centres[0]) / (len(centres) - 1), abs=1e-12)
    assert np.median(gaps) > measured + 0.001
