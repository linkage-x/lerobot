# ruff: noqa: N806  (matrix names: R, K follow the maths)
import numpy as np
import pytest

from lerobot.robots.franka_research3.tcp_pivot import fit_pivot


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
