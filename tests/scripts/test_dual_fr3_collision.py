"""dual_fr3_collision (roadmap C3/D3): the capsule model two FR3s are planned and gated with."""

import numpy as np
import pytest

from tools.fr3 import dual_fr3_collision as col, fr3_kinematics as kin


def test_the_capsules_were_fitted_to_the_model_in_the_repo() -> None:
    assert col.load().fingerprint == col.model_fingerprint()["sha256"]


def test_each_part_s_capsules_enclose_every_vertex_of_its_meshes() -> None:
    pytest.importorskip("mujoco")
    model = col.load()
    for part, points in col.part_vertices().items():
        idx = [k for k, p in enumerate(model.parts) if p == part]
        assert idx, part
        outside = np.min(
            [col._point_segment_distance(points, model.a[k], model.b[k]) - model.r[k] for k in idx], axis=0
        )
        assert outside.max() <= 1e-9, f"{part}: a vertex {outside.max() * 1e3:.3f} mm outside its capsules"


def test_the_mjcf_body_frames_are_the_dh_frames_the_capsules_hang_on() -> None:
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(col.FR3_MJCF))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(kin.Q_LOWER, kin.Q_UPPER)
        d.qpos[:7] = q
        mujoco.mj_kinematics(m, d)
        F = col.link_frames(kin.NOMINAL, q)[0]
        for i, name in enumerate(col.FRAMES):
            body = m.body(col._FRAME_BODY[name]).id
            assert np.allclose(F[i, :3, 3], d.xpos[body], atol=1e-12)
            assert np.allclose(F[i, :3, :3], d.xmat[body].reshape(3, 3), atol=1e-12)


def test_segment_distance_is_the_closest_approach() -> None:
    rng = np.random.default_rng(1)
    s = np.linspace(0.0, 1.0, 401)
    for _ in range(20):
        p1, q1, p2, q2 = rng.normal(size=(4, 3))
        a = p1 + s[:, None] * (q1 - p1)
        b = p2 + s[:, None] * (q2 - p2)
        brute = np.linalg.norm(a[:, None] - b[None], axis=-1).min()
        exact = float(col.segment_distance(p1, q1, p2, q2))
        assert exact <= brute + 1e-12 and brute - exact < 5e-3
    # parallel segments and points
    assert float(
        col.segment_distance(np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1, 0]), np.array([2.0, 1, 0]))
    ) == pytest.approx(1.0)
    assert float(
        col.segment_distance(np.zeros(3), np.zeros(3), np.array([0, 2.0, 0]), np.array([0, 2.0, 0]))
    ) == pytest.approx(2.0)


def test_two_arms_far_apart_are_apart_and_one_arm_on_the_other_overlaps() -> None:
    model = col.load()
    T = np.eye(4)
    T[1, 3] = 1.5
    far = col.clearance(model, kin.Q_READY, kin.Q_READY, T)
    assert far["arm_arm_m"][0] > 1.0
    T[1, 3] = 0.0
    assert col.clearance(model, kin.Q_READY, kin.Q_READY, T)["arm_arm_m"][0] < 0.0


def test_the_table_keeps_the_arm_and_lets_the_fingers_reach_it() -> None:
    model = col.load()
    table = [
        {
            "name": "table",
            "kind": "halfspace",
            "normal": [0, 0, 1],
            "offset": 0.0,
            "pose": "assumed",
            "size": "assumed",
        }
    ]
    A, B = model.place(kin.NOMINAL, kin.Q_READY)
    gaps = col.obstacle_gaps(table, A, B, model.r, model.table_mask())[0, :, 0]
    for k, part in enumerate(model.parts):
        assert np.isinf(gaps[k]) == (part in col.TABLE_EXEMPT)
    lifted = [dict(table[0], offset=-0.5)]
    assert np.isfinite(col.obstacle_gaps(lifted, A, B, model.r, model.table_mask())).sum() > 0
    assert col.check_obstacles([{"name": "x", "kind": "box", "pose": "measured"}])
