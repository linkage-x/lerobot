# ruff: noqa: N802, N803, N806  -- T, R: the usual names of transforms and rotations
"""Where the demonstrations went (rollout_distribution): the start (A), every target (B), every chunk (C)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import relative_tcp as rt, rollout_distribution as rd

FPS = 60
OFFSETS = [2 * k for k in range(1, 33)]  # 32 steps of 2 frames, as the checkpoints here
Q_DOWN = rt.matrix_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))  # gripper pointing down


def _episode(i: int, rng: np.random.Generator, n: int = 150) -> rd.Episode:
    """2.5 s: still for 0.5 s at a start within ~1 cm of (0, 0, 0.5), then 12 cm along +x, pointing down."""
    start = np.array([0.0, 0.0, 0.5]) + rng.normal(0, 0.005, 3)
    s = np.clip((np.arange(n) - 30) / (n - 30), 0, 1)
    P = start + np.outer(s, [0.12, 0.0, 0.0]) + rng.normal(0, 0.0005, (n, 3))
    Q = np.repeat(Q_DOWN[None], n, axis=0)
    widths = np.where(s < 0.7, 0.085, 0.03) + rng.normal(0, 0.001, n)
    return rd.Episode(i, np.concatenate([P, Q], axis=1), widths)


@pytest.fixture(scope="module")
def dist() -> rd.Distribution:
    rng = np.random.default_rng(0)
    eps = [_episode(i, rng) for i in range(10)]
    held = [_episode(100, rng)]
    return rd.build(eps, fps=FPS, offsets=OFFSETS, frame="world_x", holdout=held)


def _T(p, R=np.diag([1.0, -1.0, -1.0])) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def test_thresholds_come_from_the_demonstrations_with_floors_and_let_held_out_ones_through(dist):
    s = dist.summary
    # the synthetic demonstrations are within millimetres of each other: every threshold is its floor
    assert s["pos_mm"] == rd.FLOOR_POS_M * 1e3 and s["rot_deg"] == rd.FLOOR_ROT_DEG
    assert s["start_pos_mm"] == rd.FLOOR_START_POS_M * 1e3 and s["start_rot_deg"] == rd.FLOOR_START_ROT_DEG
    assert s["holdout"] == {"episodes": 1, "frames_pass": 1.0, "starts_pass": "1/1", "chunks_pass": 1.0}
    assert s["leave_one_out"]["pos_mm_p99"] < 10
    # C: 12 cm over 2 s -> step 32 (1.07 s) moves at most ~6.4 cm; with the margin, below 15 cm
    assert 0.06 < dist.step_m[-1] < 0.15 and np.all(np.diff(dist.step_m) >= 0)


def test_start_check(dist):
    assert dist.start_problem(_T([0.0, 0.0, 0.5]), 0.085) == ""
    assert "mm from where any training episode began" in dist.start_problem(_T([0.0, 0.15, 0.5]), 0.085)
    assert "turned 60°" in dist.start_problem(_T([0.0, 0.0, 0.5], _rot_z(60) @ np.diag([1.0, -1.0, -1.0])), 0.085)
    assert "open the gripper" in dist.start_problem(_T([0.0, 0.0, 0.5]), 0.02)
    assert dist.start_problem(_T([0.0, 0.0, 0.5]), None) == ""  # no width reading: position and orientation only


def test_targets_near_the_tube_pass_and_the_ones_off_it_are_named(dist):
    anchor = _T([0.0, 0.0, 0.5])
    on = np.stack([_T([0.002 * k, 0.0, 0.5]) for k in range(1, 6)])  # the demonstrations: 1 mm per frame
    rel = np.linalg.inv(anchor)[None] @ on
    assert dist.target_problems(on, rel) == [""] * 5
    off = on.copy()
    off[2, :3, 3] += [0.0, 0.0, 0.10]  # 10 cm above the tube, inside any box around it
    why = dist.target_problems(off, np.linalg.inv(anchor)[None] @ off)
    assert why[:2] == ["", ""] and "from any training TCP position" in why[2]
    turned = on.copy()
    turned[1, :3, :3] = _rot_z(90) @ turned[1, :3, :3]
    assert "turned 90° from the training poses there" in dist.target_problems(turned, rel)[1]


def test_a_chunk_that_moves_farther_than_training_did_by_that_step_is_named(dist):
    # inside the tube (so B passes) but the first step already 8 cm from the anchor
    anchor = _T([0.0, 0.0, 0.5])
    T = np.stack([_T([0.08, 0.0, 0.5])])
    why = dist.target_problems(T, np.linalg.inv(anchor)[None] @ T)
    assert "the chunk moves 80 mm by here; training moved at most" in why[0]


def test_save_and_load_round_trip(dist, tmp_path):
    dist.save(tmp_path / "d.npz")
    back = rd.Distribution.load(tmp_path / "d.npz")
    assert back.summary == dist.summary and back.pos_m == dist.pos_m
    np.testing.assert_array_equal(back.P, dist.P)
    np.testing.assert_array_equal(back.step_deg, dist.step_deg)


def _export(root: Path, episodes: int, frames: int) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes, "total_frames": frames}))
    return root


def _contract() -> rt.PolicyContract:
    cam = rt.CameraInput("cam_10", 1920, 1080, [0, 0, 1920, 1080], 32, 18)
    return rt.PolicyContract(hand="right", fps=60, frame_stride=2, horizon=32, history=2, cameras=[cam],
                             width_name="box_gripper.distance_m")


def test_for_checkpoint_builds_from_the_export_it_trained_on_then_reads_its_cache(tmp_path, monkeypatch):
    from tools.fr3 import relative_tcp_data as rtd

    rng = np.random.default_rng(1)
    eps = [_episode(i, rng) for i in range(6)]
    _export(tmp_path / "outputs" / "exports" / "exp_a", 6, 900)
    monkeypatch.setattr(rtd, "load_episode_arrays", lambda root, c: [
        rtd.EpisodeArrays(poses=e.poses, widths=e.widths, wrench=None, episode=e.index) for e in eps])
    prov = {"exports": [{"name": "exp_a", "root": "/elsewhere/exp_a", "episodes": 6, "frames": 900,
                         "world_frame_id": "world_x"}],
            "workspace": {"frame": "world_x"}, "holdout_episodes": [5]}
    d = rd.for_checkpoint(prov, _contract(), tmp_path)
    assert d.summary["episodes"] == [0, 1, 2, 3, 4] and d.summary["holdout"]["episodes"] == 1
    assert len(list((tmp_path / "outputs" / "rollout" / "distribution").glob("*.npz"))) == 1
    monkeypatch.setattr(rtd, "load_episode_arrays", lambda *a: pytest.fail("the cache is read, not rebuilt"))
    assert rd.for_checkpoint(prov, _contract(), tmp_path).summary == d.summary


def test_for_checkpoint_refuses_a_missing_or_different_export(tmp_path):
    prov = {"exports": [{"name": "exp_b", "episodes": 6, "frames": 900}], "workspace": {"frame": "w"}}
    with pytest.raises(rd.DistributionError, match="not on this machine"):
        rd.for_checkpoint(prov, _contract(), tmp_path)
    _export(tmp_path / "outputs" / "exports" / "exp_b", 7, 950)
    with pytest.raises(rd.DistributionError, match="not the same export"):
        rd.for_checkpoint(prov, _contract(), tmp_path)
    with pytest.raises(rd.DistributionError, match="names no training export"):
        rd.for_checkpoint({}, _contract(), tmp_path)
