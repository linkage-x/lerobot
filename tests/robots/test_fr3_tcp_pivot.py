# ruff: noqa: N806  (matrix names: R, K follow the maths)
import numpy as np
import pytest

from lerobot.robots.franka_research3.tcp_pivot import fit_pivot, fit_pivot_joint


def _rot(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _sweep(s, c, n=2000, span_deg=30.0, seed=0):
    rng = np.random.default_rng(seed)
    base = _rot([1, 0, 0], np.pi)  # tool pointing down
    R = []
    for k in range(n):
        u = k / n
        tilt = np.radians(span_deg) * np.sin(2 * np.pi * 3 * u)
        tilt2 = np.radians(span_deg) * np.cos(2 * np.pi * 2 * u)
        spin = np.radians(60) * np.sin(2 * np.pi * u)
        R.append(base @ _rot([1, 0, 0], tilt) @ _rot([0, 1, 0], tilt2) @ _rot([0, 0, 1], spin))
    R = np.asarray(R)
    t = c - np.einsum("nij,j->ni", R, s) + rng.normal(0, 5e-5, (n, 3))
    return R, t


def test_span_is_the_whole_sweep_and_one_axis_alone_reads_as_weak():
    # yaw 0..30° one way only (as recorded in 20261014_140351): the span is 30°, not the ~15° about the mean
    s = np.array([0.0015, -0.0005, 0.006])
    c = np.array([0.645, -0.18, 0.207])
    base = _rot([0, 1, 0], np.pi / 2)  # tool axis level
    R = np.asarray([_rot([0, 0, 1], np.radians(a)) @ base for a in np.linspace(0, 30, 600)])
    t = c - np.einsum("nij,j->ni", R, s) + np.random.default_rng(1).normal(0, 5e-5, (600, 3))
    fit = fit_pivot(R, t)
    assert fit.span_deg == pytest.approx(30, abs=0.5)
    assert fit.weak_singular < 0.01  # turned about one axis only

def test_pivot_recovers_the_socket_offset_in_the_ee_frame():
    s = np.array([-0.0006, -0.0029, -0.0104])
    c = np.array([0.55, 0.10, 0.12])
    R, t = _sweep(s, c)
    fit = fit_pivot(R, t)
    assert np.allclose(fit.s_ee_m, s, atol=1e-4)
    assert np.allclose(fit.c_base_m, c, atol=1e-4)
    assert fit.residual_rms_m < 2e-4
    assert fit.span_deg > 25
    assert np.all(fit.s_sigma_m < 2e-4)


def test_pivot_drops_samples_where_the_insert_lifted_off_the_ball():
    s = np.array([0.0, 0.0, -0.010])
    c = np.array([0.5, 0.0, 0.1])
    R, t = _sweep(s, c)
    t[500:560] += np.array([0.0, 0.0, 0.004])  # lifted 4 mm for a moment
    fit = fit_pivot(R, t)
    assert np.allclose(fit.s_ee_m, s, atol=2e-4)
    assert fit.n_used < fit.n_total


def test_pivot_refuses_a_tool_that_never_turned():
    R = np.repeat(np.eye(3)[None], 50, axis=0)
    t = np.tile([0.5, 0.0, 0.1], (50, 1))
    with pytest.raises(ValueError, match="usable orientations"):
        fit_pivot(R, t)


def _level_sweep(s, c, heading_deg, *, n=1500, tilt_deg=12.0, swing_deg=10.0, noise_m=3e-4, seed=0):
    """As on 10-14: the gripper reaching in level, cup down, turned only a little about the ball."""
    rng = np.random.default_rng(seed)
    base = _rot([0, 0, 1], np.radians(heading_deg)) @ _rot([0, 1, 0], np.pi / 2)  # tool axis level
    R = []
    for k in range(n):
        u = k / n
        pitch = np.radians(tilt_deg) * np.sin(2 * np.pi * 3 * u)
        roll = np.radians(tilt_deg) * np.cos(2 * np.pi * 2 * u)
        swing = np.radians(swing_deg) * np.sin(2 * np.pi * u)
        R.append(_rot([0, 0, 1], swing) @ base @ _rot([1, 0, 0], pitch) @ _rot([0, 0, 1], roll))
    R = np.asarray(R)
    t = c - np.einsum("nij,j->ni", R, s) + rng.normal(0, noise_m, (n, 3))
    return R, t


def test_runs_from_different_headings_fit_one_socket_about_one_ball():
    s = np.array([0.0015, -0.0009, 0.0049])
    c = np.array([0.65, -0.18, 0.21])
    runs = [_level_sweep(s, c, h, seed=i) for i, h in enumerate((0, 70, 140))]
    joint = fit_pivot_joint([R for R, _ in runs], [t for _, t in runs])
    assert np.linalg.norm(joint.s_ee_m - s) < 5e-4
    assert joint.c_base_m.shape == (1, 3) and np.allclose(joint.c_base_m[0], c, atol=5e-4)
    assert np.all(np.linalg.norm(joint.run_centre_offset_m, axis=1) < 3e-4)
    assert joint.s_leave_one_run_out_m.shape == (3, 3)
    assert joint.span_deg > 60  # the headings, 140° apart
    assert np.all(joint.s_sigma_m < 3e-4)
    assert joint.c_sigma_m.shape == (1, 3) and np.all((joint.c_sigma_m > 0) & (joint.c_sigma_m < 3e-4))


def test_a_moved_ball_shows_in_each_runs_own_centre_and_a_centre_per_run_absorbs_it():
    s = np.array([0.0015, -0.0009, 0.0049])
    c = np.array([0.65, -0.18, 0.21])
    moved = c + [-0.0021, 0.0022, 0.0004]  # 10-14, between 11:12 and 11:30
    runs = [_level_sweep(s, c, 0, seed=1), _level_sweep(s, c, 70, seed=2), _level_sweep(s, moved, 140, seed=3)]
    Rs, ts = [R for R, _ in runs], [t for _, t in runs]
    shared = fit_pivot_joint(Rs, ts)
    assert np.linalg.norm(shared.run_centre_offset_m[2]) > 1e-3
    apart = fit_pivot_joint(Rs, ts, shared_centre=False)
    assert apart.c_base_m.shape == (3, 3) and np.allclose(apart.c_base_m[2], moved, atol=5e-4)
    assert apart.residual_rms_m < shared.residual_rms_m
    # a ball moved under one shared centre: dropping that run moves c, which its sigma must say
    assert np.max(shared.c_sigma_m) > 3e-4 and apart.c_sigma_m.shape == (3, 3)
    with pytest.raises(ValueError, match="at least two runs"):
        fit_pivot_joint(Rs[:1], ts[:1])
