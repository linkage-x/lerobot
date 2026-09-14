"""The four numbers that decide whether scripted motion is shaped like a demonstration.

Every test here is a teacher that would pass a weaker criterion. A constant-speed script matches
the demonstrations' median step exactly, and per-frame jitter matches their dispersion too -- so
the tests are written as "this teacher is wrong, and here is the number that says so".
"""

import math

import numpy as np
import pytest

from tools.fr3.step_profile import (
    DEMO_BAND,
    EpisodeSteps,
    check,
    describe,
    episodes_from_rows,
    profile,
)


def _episode(steps_mm, *, total_mm=300.0):
    """An episode walking in from `total_mm` of remaining travel, taking the given steps."""

    remaining = []
    left = total_mm
    for step in steps_mm:
        remaining.append(max(left, 0.0))
        left -= step
    return EpisodeSteps(stepMm=list(steps_mm), remainingMm=remaining)


def _constant_episodes(n=6, steps=200, mm=2.0):
    return [_episode([mm] * steps, total_mm=steps * mm) for _ in range(n)]


def _jittered_episodes(n=6, steps=200, mm=2.0, cv=0.55, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        drawn = np.clip(rng.normal(mm, mm * cv, steps), 0.01, None)
        out.append(_episode(list(drawn), total_mm=float(drawn.sum())))
    return out


def _human_like_episodes(n=6, steps=200, seed=0):
    """A ramp in and a decelerating tail, with a slowly varying residual: r1 high by construction."""

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        distance = np.linspace(300.0, 0.0, steps)
        # Speed falls with remaining travel, which is the demonstrations' own shape.
        base = np.clip(0.0145 * distance, 0.4, 3.1)
        residual = np.zeros(steps)
        for i in range(1, steps):
            residual[i] = 0.9 * residual[i - 1] + rng.normal(0.0, 0.12)
        drawn = np.clip(base * np.exp(residual), 0.01, None)
        out.append(EpisodeSteps(stepMm=list(drawn), remainingMm=list(distance)))
    return out


def test_a_constant_speed_script_matches_the_median_and_fails_on_everything_else():
    """2.00 mm is the demonstrations' median. A criterion on the median alone passes this."""

    result = profile(_constant_episodes())
    assert DEMO_BAND.p50Mm[0] <= result.p50Mm <= DEMO_BAND.p50Mm[1]
    failures = {name for name, passed, _ in check(result) if not passed}
    assert "cv" in failures
    assert "near/far p50 ratio" in failures


def test_per_frame_jitter_buys_the_dispersion_and_loses_the_autocorrelation():
    """The reason not to add iid noise: it moves the difference, it does not remove it."""

    result = profile(_jittered_episodes())
    passed = {name for name, ok, _ in check(result) if ok}
    assert "cv" in passed, "the jitter was supposed to reproduce the dispersion"
    assert result.r1 < DEMO_BAND.r1Min
    assert "r1" not in passed


def test_a_profile_with_deceleration_and_a_slow_residual_passes_every_criterion():
    result = profile(_human_like_episodes())
    failures = [(name, detail) for name, ok, detail in check(result) if not ok]
    assert not failures, failures


def test_the_step_shrinks_toward_the_target_is_read_off_buckets_and_not_off_the_whole_run():
    result = profile(_human_like_episodes())
    by_label = {bucket["remainingMm"]: bucket for bucket in result.buckets}
    assert by_label[">=200"]["frames"] and by_label["20-50"]["frames"]
    assert by_label["20-50"]["p50Mm"] < by_label[">=200"]["p50Mm"]
    assert math.isclose(
        result.nearFarRatio, by_label["20-50"]["p50Mm"] / by_label[">=200"]["p50Mm"], rel_tol=1e-9
    )


def test_a_reading_that_could_not_be_taken_fails_rather_than_passes():
    """An empty session must not read as a clean bill: nan is "we do not know", not "fine"."""

    result = profile([])
    assert all(not passed for _name, passed, _detail in check(result))
    assert "frames=0" in describe(result)


def _rows(positions, phase="descend_to_peg", episode=0):
    rows = [{"kind": "marker", "marker": "episode_start", "episode": episode, "cycle": episode}]
    for index, (x, y, z) in enumerate(positions):
        rows.append(
            {
                "kind": "frame",
                "t": 100.0 + 0.03 * index,
                "phase": phase,
                "episode": episode,
                "sent_action": {"ee.x": x, "ee.y": y, "ee.z": z, "ee.wx": 0.0, "ee.wy": 0.0,
                                "ee.wz": 0.0, "gripper.pos": 1.0},
            }
        )
    rows.append({"kind": "marker", "marker": "episode_end", "episode": episode, "cycle": episode,
                 "verdict": "held"})
    return rows


def test_a_session_is_read_as_differences_between_successive_commands():
    positions = [(0.30 + 0.002 * i, -0.14, 0.12) for i in range(11)]
    episodes = episodes_from_rows(_rows(positions))
    assert len(episodes) == 1
    # Ten differences from eleven commands: the first frame has no predecessor and is dropped
    # rather than counted as a zero step.
    assert len(episodes[0].stepMm) == 10
    assert all(abs(step - 2.0) < 1e-6 for step in episodes[0].stepMm)


def test_frames_from_a_leg_that_is_not_training_data_are_not_in_the_profile():
    """The fetch and the place are the environment moving; they would flatten every statistic."""

    rows = _rows([(0.30 + 0.002 * i, -0.14, 0.12) for i in range(6)])
    rows += [
        {
            "kind": "frame",
            "t": 200.0 + 0.03 * i,
            "phase": "carry_to_drop",
            "episode": 0,
            "sent_action": {"ee.x": 0.5 + 0.02 * i, "ee.y": 0.0, "ee.z": 0.3, "ee.wx": 0.0,
                            "ee.wy": 0.0, "ee.wz": 0.0, "gripper.pos": 0.0},
        }
        for i in range(10)
    ]
    # The stray leg sits after `episode_end`, which is where the recorder puts it; a leg inside an
    # episode is refused by `auto_collect_dataset`, not silently profiled.
    episodes = episodes_from_rows(rows)
    assert len(episodes) == 1
    assert len(episodes[0].stepMm) == 5


@pytest.mark.parametrize("closing", [True, False])
def test_an_episode_with_one_command_contributes_nothing_rather_than_a_zero(closing):
    rows = [{"kind": "marker", "marker": "episode_start", "episode": 0, "cycle": 0}]
    rows.append(
        {
            "kind": "frame", "t": 100.0, "phase": "descend_to_peg", "episode": 0,
            "sent_action": {"ee.x": 0.3, "ee.y": -0.14, "ee.z": 0.12, "ee.wx": 0.0, "ee.wy": 0.0,
                            "ee.wz": 0.0, "gripper.pos": 1.0},
        }
    )
    if closing:
        rows.append({"kind": "marker", "marker": "episode_end", "episode": 0, "cycle": 0})
    assert episodes_from_rows(rows) == []
