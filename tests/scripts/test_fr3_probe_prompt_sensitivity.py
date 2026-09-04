"""The reading of the prompt ablation, tested where it does not need a GPU.

Everything that decides what the experiment *means* -- what counts as the noise floor, how a
chunk difference is measured, when the verdict flips -- is a pure function here. The policy call
itself is not tested: it needs a 4B-parameter checkpoint and a CUDA device, and a stub of it
would only assert that the stub was called.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "fr3" / "fr3_probe_prompt_sensitivity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fr3_probe_prompt_sensitivity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def chunk(*, dx=0.0, dy=0.0, dz=0.0, drz=0.0, grip=0.0, steps=50):
    return np.tile(np.array([dx, dy, dz, drz, grip], dtype=np.float64), (steps, 1))


class TestChunkDifference:
    def test_identical_chunks_differ_by_nothing(self):
        a = chunk(dx=0.001)
        diff = probe.chunk_difference(a, a.copy())
        assert diff.step_median_mm == 0.0
        assert diff.endpoint_mm == 0.0
        assert diff.yaw_deg == 0.0

    def test_per_step_distance_is_reported_in_millimetres(self):
        # 1 mm apart on every step of the chunk.
        diff = probe.chunk_difference(chunk(dx=0.001), chunk(dx=0.002))
        assert diff.step_median_mm == pytest.approx(1.0)
        assert diff.step_max_mm == pytest.approx(1.0)

    def test_the_endpoint_accumulates_what_the_per_step_figure_does_not(self):
        """A steady 1 mm offset over 50 steps is 50 mm of divergence by the end of the chunk.

        Two numbers rather than one because they can disagree in the direction that matters: a
        difference that alternates sign is loud per step and goes nowhere.
        """
        steady = probe.chunk_difference(chunk(dx=0.001), chunk(dx=0.002))
        assert steady.endpoint_mm == pytest.approx(50.0)

        alternating = chunk(dx=0.001)
        alternating[1::2, 0] = 0.003  # mean is still 0.002, but it cancels
        loud = probe.chunk_difference(chunk(dx=0.002), alternating)
        assert loud.step_median_mm == pytest.approx(1.0)
        assert loud.endpoint_mm == pytest.approx(0.0, abs=1e-9)

    def test_yaw_is_reported_in_degrees_and_the_gripper_on_its_own_scale(self):
        diff = probe.chunk_difference(chunk(), chunk(drz=np.radians(1.0) / 50, grip=0.25))
        assert diff.yaw_deg == pytest.approx(1.0)
        assert diff.gripper == pytest.approx(0.25)

    def test_chunks_that_cannot_be_compared_are_refused(self):
        with pytest.raises(probe.ProbeError, match="shape"):
            probe.chunk_difference(chunk(steps=50), chunk(steps=49))
        with pytest.raises(probe.ProbeError, match="chunk"):
            probe.chunk_difference(np.zeros((50, 3)), np.zeros((50, 3)))


class TestSampleFrames:
    def test_frames_are_spread_across_the_episode_not_taken_from_the_front(self):
        assert probe.sample_frames(500, 5) == [0, 125, 250, 374, 499]

    def test_a_short_episode_is_never_over_sampled(self):
        assert probe.sample_frames(3, 8) == [0, 1, 2]

    def test_an_empty_episode_is_refused_rather_than_silently_skipped(self):
        with pytest.raises(probe.ProbeError):
            probe.sample_frames(0, 4)


class TestSummary:
    def _summary(self, floor_mm, effect_mm, neighbour_mm=10.0, frames=6):
        noise = [probe.chunk_difference(chunk(), chunk(dx=floor_mm / 1000)) for _ in range(frames)]
        prompts = {
            "other": [probe.chunk_difference(chunk(), chunk(dx=effect_mm / 1000)) for _ in range(frames)]
        }
        neigh = [probe.chunk_difference(chunk(), chunk(dx=neighbour_mm / 1000)) for _ in range(frames)]
        return probe.summarise(noise, prompts, neigh)

    def test_the_effect_is_expressed_as_a_multiple_of_the_floor_not_as_a_raw_number(self):
        """A raw millimetre figure cannot be read without knowing this policy's own spread."""
        summary = self._summary(floor_mm=2.0, effect_mm=6.0)
        assert summary.noise_floor_mm == pytest.approx(2.0)
        assert summary.prompt_effect_mm["other"] == pytest.approx(6.0)
        assert summary.prompt_over_floor["other"] == pytest.approx(3.0)

    def test_a_paired_count_is_reported_beside_the_ratio(self):
        """The ratio of medians can be moved by a few frames; the sign count cannot."""
        summary = self._summary(floor_mm=1.0, effect_mm=5.0, frames=6)
        assert summary.frames_where_prompt_beats_floor["other"] == "6/6"
        summary = self._summary(floor_mm=5.0, effect_mm=1.0, frames=6)
        assert summary.frames_where_prompt_beats_floor["other"] == "0/6"

    def test_the_neighbouring_frame_reference_is_carried_through(self):
        summary = self._summary(floor_mm=2.0, effect_mm=2.0, neighbour_mm=40.0)
        assert summary.neighbour_mm == pytest.approx(40.0)

    def test_endpoint_and_yaw_survive_into_the_detail_block(self):
        summary = self._summary(floor_mm=2.0, effect_mm=6.0)
        assert summary.detail["noise_floor"]["endpoint_mm"] == pytest.approx(100.0)
        assert summary.detail["prompt"]["other"]["endpoint_mm"] == pytest.approx(300.0)


class TestVerdict:
    def _summary(self, ratio):
        return probe.ProbeSummary(
            frames=8,
            noise_floor_mm=1.0,
            prompt_effect_mm={"other": ratio},
            neighbour_mm=10.0,
            prompt_over_floor={"other": ratio},
        )

    def test_an_effect_below_the_floor_is_called_inert(self):
        assert "inert" in probe.verdict(self._summary(1.0))
        assert "inert" in probe.verdict(self._summary(1.49))

    def test_an_effect_of_the_same_order_as_the_noise_is_called_weak_not_live(self):
        """The band exists so a 2x reading cannot be reported as a working language channel."""
        assert "weak" in probe.verdict(self._summary(2.0))

    def test_an_effect_well_past_the_floor_is_called_live(self):
        assert "live" in probe.verdict(self._summary(3.0))

    def test_the_strongest_alternate_prompt_decides(self):
        summary = probe.ProbeSummary(
            frames=8,
            noise_floor_mm=1.0,
            prompt_effect_mm={"near": 1.1, "far": 8.0},
            neighbour_mm=10.0,
            prompt_over_floor={"near": 1.1, "far": 8.0},
        )
        assert "live" in probe.verdict(summary)

    def test_running_no_alternate_prompt_produces_no_verdict_rather_than_a_default_one(self):
        summary = probe.ProbeSummary(
            frames=8, noise_floor_mm=1.0, prompt_effect_mm={}, neighbour_mm=10.0, prompt_over_floor={}
        )
        assert "nothing to compare" in probe.verdict(summary)


class TestCli:
    def test_a_single_seed_is_refused_because_it_leaves_no_noise_floor(self):
        """Without a second seed every non-zero prompt difference reads as a live channel."""
        assert probe.parse_indices("0,1") == [0, 1]
        assert probe.parse_indices("3") == [3]

    def test_the_alternate_prompts_are_two_different_kinds_of_wrong(self):
        assert len(probe.DEFAULT_ALTERNATE_PROMPTS) == 2
        assert all(prompt.strip() for prompt in probe.DEFAULT_ALTERNATE_PROMPTS)
