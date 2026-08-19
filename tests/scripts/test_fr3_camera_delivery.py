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

"""The two camera-delivery diagnostics, pinned on data whose fault is known by construction.

These tools exist to name a root cause, so the property that matters is that each fault shape
produces a *distinguishable* reading: a sensor at half rate must not look like a host that
stalled, or the diagnosis they support is a coin flip.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tools.fr3.fr3_camera_delivery_probe import summarize_bursts, summarize_camera_delivery


def _bench():
    """Import the live bench with pyrealsense2 stubbed; only its pure statistics are tested."""
    if "pyrealsense2" not in sys.modules:
        stub = types.ModuleType("pyrealsense2")
        for attribute in ("stream", "format", "option", "camera_info", "frame_metadata_value"):
            setattr(stub, attribute, types.SimpleNamespace())
        stub.timestamp_domain = types.SimpleNamespace(global_time=0, system_time=1)
        sys.modules["pyrealsense2"] = stub
    from tools.fr3 import fr3_camera_delivery_bench

    return fr3_camera_delivery_bench


def test_a_camera_at_nominal_rate_reuses_nothing_and_reports_its_rate():
    ticks = np.arange(600) / 60.0
    summary = summarize_camera_delivery(ticks - 0.02, ticks, camera_fps=60.0)

    assert summary["reused_fraction"] == pytest.approx(0.0)
    assert summary["effective_fps"] == pytest.approx(60.0, abs=0.1)
    assert summary["gap_histogram_periods"] == {"1x": 599}
    assert summary["stale_vs_arm_ms"]["p50"] == pytest.approx(20.0)


def test_a_sensor_at_half_rate_shows_as_reuse_and_two_period_gaps():
    """The signature of a throttled sensor: every interval at 2x, evenly spread."""
    ticks = np.arange(600) / 60.0
    halved = np.floor(ticks / (2.0 / 60.0)) * (2.0 / 60.0)
    summary = summarize_camera_delivery(halved, ticks, camera_fps=60.0)

    assert summary["reused_fraction"] == pytest.approx(0.5, abs=0.01)
    assert summary["effective_fps"] == pytest.approx(30.0, abs=0.1)
    assert set(summary["gap_histogram_periods"]) == {"2x"}


def test_an_occasional_host_stall_leaves_a_mostly_nominal_histogram_with_a_tail():
    """The signature of a host fault: 1x everywhere, a handful of long gaps -- not 2x throughout."""
    delivered = np.arange(600) / 60.0
    delivered[300:] += 0.100  # one 100 ms hiccup, then back to cadence
    summary = summarize_camera_delivery(delivered, delivered + 0.02, camera_fps=60.0)

    histogram = summary["gap_histogram_periods"]
    assert histogram["1x"] == 598
    assert sum(count for key, count in histogram.items() if key != "1x") == 1
    assert summary["gap_ms"]["max"] > 100.0


def test_bursts_separate_a_steady_fault_from_a_clustered_one():
    steady = np.zeros(1000, dtype=bool)
    steady[::10] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[400:500] = True

    assert summarize_bursts(steady)["longest_burst"] == 1
    assert summarize_bursts(steady)["worst_decile_share"] == pytest.approx(0.1, abs=0.01)
    assert summarize_bursts(clustered)["bursts"] == 1
    assert summarize_bursts(clustered)["longest_burst"] == 100
    assert summarize_bursts(clustered)["worst_decile_share"] == pytest.approx(1.0)


def test_no_over_budget_frames_is_not_a_burst():
    assert summarize_bursts(np.zeros(100, dtype=bool)) == {
        "bursts": 0,
        "longest_burst": 0,
        "worst_decile_share": 0.0,
    }


def test_frame_counter_gaps_are_what_names_a_host_drop():
    """Timestamps alone cannot: a dropped frame and a slow sensor leave the same hole."""
    bench = _bench()
    records = [
        bench.FrameRecord(
            counter=counter,
            frame_number=index,
            device_timestamp_ms=0.0,
            domain_name="global_time",
            exposure_us=8000.0,
            handover_perf_s=index / 60.0 + 0.005,
            acquisition_perf_s=index / 60.0,
        )
        # counters 0,1,2 then a jump to 5: two frames the sensor produced and we never saw
        for index, counter in enumerate([0, 1, 2, 5, 6, 7])
    ]

    summary = bench.summarize_stream(records, nominal_fps=60.0)

    assert summary["dropped_frames"] == 2
    assert summary["drop_events"] == 1
    assert summary["frame_counter_available"] is True
    assert summary["handover_lag_ms"]["p50"] == pytest.approx(5.0)


def test_a_missing_frame_counter_is_reported_rather_than_read_as_zero_drops():
    bench = _bench()
    records = [
        bench.FrameRecord(
            counter=None,
            frame_number=index,
            device_timestamp_ms=0.0,
            domain_name="global_time",
            exposure_us=None,
            handover_perf_s=index / 60.0,
            acquisition_perf_s=index / 60.0,
        )
        for index in range(10)
    ]

    summary = bench.summarize_stream(records, nominal_fps=60.0)

    assert summary["frame_counter_available"] is False
    assert summary["dropped_frames"] == 0


def test_without_frame_counter_a_delivery_gap_is_still_counted_but_not_called_a_sensor_drop():
    """The fallback sees what this process missed, not what the sensor produced. Say which."""
    bench = _bench()
    records = [
        bench.FrameRecord(
            counter=None,
            frame_number=number,
            device_timestamp_ms=0.0,
            domain_name="global_time",
            exposure_us=None,
            handover_perf_s=index / 60.0,
            acquisition_perf_s=index / 60.0,
        )
        for index, number in enumerate([0, 1, 2, 6, 7])
    ]

    summary = bench.summarize_stream(records, nominal_fps=60.0)

    assert summary["dropped_frames"] == 3
    assert summary["frame_counter_available"] is False


def test_poll_summary_measures_the_skew_the_recorder_would_have_recorded():
    bench = _bench()
    samples = [
        bench.PollSample(
            tick_perf_s=index / 60.0,
            selected={"ee": index / 60.0 - 0.020, "side": index / 60.0 - 0.025},
        )
        for index in range(100)
    ]

    summary = bench.summarize_poll(samples, camera_names=["ee", "side"])

    assert summary["cross_camera_skew_ms"]["p50"] == pytest.approx(5.0)
    assert summary["reused_frame_fraction"] == pytest.approx(0.0)
    assert summary["anchor_staleness_ms"]["p50"] == pytest.approx(25.0)


def test_poll_summary_counts_a_tick_that_got_no_new_frame_from_either_camera():
    bench = _bench()
    frozen = {"ee": 0.0, "side": 0.005}
    samples = [bench.PollSample(tick_perf_s=index / 60.0, selected=dict(frozen)) for index in range(10)]

    assert bench.summarize_poll(samples, camera_names=["ee", "side"])["reused_frame_fraction"] == 1.0
