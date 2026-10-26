"""dual_fr3_clock (roadmap D2): the clock fit, the response lag, and the report on fake arms.

The end-to-end runs reuse test_dual_fr3_replay_runtime's two-arm geometry (40 frames at 50 Hz, about a
second each); the fake arms keep libfranka-like clocks with an offset, a drift and network latency.
"""

# ruff: noqa: N806  (T_right_left as in the runtime)
import numpy as np
import pytest

from tests.scripts.test_dual_fr3_replay_runtime import _plans, _run
from tools.fr3 import dual_fr3_clock as clk

PERIOD_MS = 20.0


def _states(n=500, rate_hz=50.0, offset_s=1234.5, drift_ppm=40.0, lat_min_ms=0.3, lat_extra_ms=0.5, seed=1):
    """Host sample times, the robot clock reading them (1 ms floor) and their arrival with latency."""
    rng = np.random.default_rng(seed)
    sample_ns = (5e12 + np.arange(n) * 1e9 / rate_hz + rng.uniform(0, 1e5, n)).astype(np.int64)
    robot_s = np.floor((offset_s + (sample_ns - sample_ns[0]) / 1e9 * (1 + drift_ppm * 1e-6)) * 1e3) / 1e3
    receive_ns = sample_ns + ((lat_min_ms + rng.exponential(lat_extra_ms, n)) * 1e6).astype(np.int64)
    return sample_ns, robot_s, receive_ns


def test_the_fit_finds_drift_and_maps_states_to_their_sample_time_plus_the_fastest_trip() -> None:
    sample_ns, robot_s, receive_ns = _states()
    fit = clk.fit_clock(robot_s, receive_ns)
    assert fit.drift_ppm == pytest.approx(-40.0, abs=fit.drift_resolution_ppm)  # host per robot second
    mapped = fit.to_host_ns(robot_s)
    # sample time + the minimum latency (0.3 ms), within the 1 ms clock floor
    assert np.all(np.abs(mapped - sample_ns - 0.3e6) < 1.1e6)
    assert np.all(receive_ns - mapped >= -1)  # nothing arrives before the envelope
    assert fit.latency_ms["above_min_p50"] < 1.5


def test_least_squares_would_have_put_the_latency_into_the_offset() -> None:
    """Why the lower envelope: a skewed latency (here one arm's queue fills over the run) tilts a mean line."""
    sample_ns, robot_s, receive_ns = _states(lat_extra_ms=0.0)
    backlog = np.linspace(0.0, 5e6, len(receive_ns)).astype(np.int64)  # 0 -> 5 ms of queueing
    receive_ns = receive_ns + backlog * (np.arange(len(receive_ns)) % 2)  # every other state queued
    fit = clk.fit_clock(robot_s, receive_ns)
    b, _ = np.polyfit(robot_s - robot_s[0], (receive_ns - receive_ns[0]) / 1e9, 1)
    assert abs(fit.drift_ppm + 40.0) < 50.0
    assert abs((b - 1) * 1e6 + 40.0) > 200.0  # the mean line reads the backlog as ~250 ppm of drift


def test_the_fit_refuses_too_few_or_frozen_states() -> None:
    with pytest.raises(ValueError, match="needs 10"):
        clk.fit_clock(np.arange(5.0), np.arange(5) * 10**9)
    with pytest.raises(ValueError, match="did not advance"):
        clk.fit_clock(np.full(20, 3.0), np.arange(20) * 10**6)


def test_the_response_lag_overlays_a_delayed_copy() -> None:
    t = np.arange(0.0, 2.0, 0.02)
    p = np.stack([0.1 * np.sin(2 * np.pi * 0.7 * t), 0.05 * t, np.zeros_like(t)], axis=1)
    lag = clk.response_lag_s(t, p, t + 0.0137, p)
    assert lag == pytest.approx(0.0137, abs=3e-4)
    assert clk.response_lag_s(t, np.zeros_like(p), t, p) is None  # no motion, no lag


def test_two_like_arms_on_one_schedule_pass_with_their_clocks_fitted(tmp_path) -> None:
    clocks = {
        "right": {"offset_s": 812.0, "drift_ppm": 25.0, "seed": 1},
        "left": {"offset_s": 3.5, "drift_ppm": -30.0, "latency_ms": [0.4, 0.6], "seed": 2},
    }
    plans, T_right_left, _ = _plans(amplitude=0.3, clocks=clocks)
    report = _run(tmp_path, plans, T_right_left)
    c = report["clock"]
    assert c["status"] == "passed", c
    assert abs(c["start_skew_ms"]) < 3.0  # same controller, same schedule: only jitter
    assert c["drift_ms_over_run"] is not None
    assert c["label"].startswith("estimate")
    assert c["t_lr_error_mm"]["from_skew_p95"] < 2.0
    assert set(c["arms"]) == {"right", "left"} and c["arms"]["left"]["clock"]["n"] == 40
    saved = np.load(tmp_path / "run" / "clock_samples.npz")
    assert saved["q_left"].shape == (40, 7) and len(saved["robot_s_right"]) == 40


def test_an_arm_that_moves_two_frames_late_fails_the_one_period_criterion(tmp_path) -> None:
    plans, T_right_left, _ = _plans(amplitude=0.3, clocks={"left": {"delay_frames": 2}})
    report = _run(tmp_path, plans, T_right_left)
    assert report["stop"] is None  # it still tracks inside its gates; only D2 sees it
    c = report["clock"]
    assert c["status"] == "failed"
    assert c["start_skew_ms"] == pytest.approx(2 * PERIOD_MS, abs=4.0)
    assert c["t_lr_error_mm"]["from_skew_max"] > 5.0
    assert ">= one period" in c["verdict"]


def test_a_run_stopped_at_frame_zero_reports_no_clock(tmp_path) -> None:
    plans, T_right_left, _ = _plans(spacing=0.15)
    report = _run(tmp_path, plans, T_right_left)
    assert report["clock"]["status"].startswith("no robot clock")
