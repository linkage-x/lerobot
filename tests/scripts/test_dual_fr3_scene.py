"""dual_fr3_scene (roadmap C1): two FR3s with their grippers in one MuJoCo scene, placed by T_right_left."""

import numpy as np
import pytest

from tools.fr3 import dual_fr3_collision as col, dual_fr3_scene as sc, fr3_kinematics as kin


def _placed(spacing: float) -> np.ndarray:
    T = np.eye(4)
    T[1, 3] = spacing
    return T


def test_the_fingerprint_follows_the_placement_and_the_obstacles() -> None:
    a = sc.scene_record(_placed(0.8), {"source": "assumed"}, [sc.table_obstacle()])
    b = sc.scene_record(_placed(0.8), {"source": "assumed"}, [sc.table_obstacle()])
    c = sc.scene_record(_placed(0.81), {"source": "assumed"}, [sc.table_obstacle()])
    d = sc.scene_record(_placed(0.8), {"source": "assumed"}, [sc.table_obstacle(0.01)])
    assert a["fingerprint"] == b["fingerprint"]
    assert len({a["fingerprint"], c["fingerprint"], d["fingerprint"]}) == 3
    assert a["model"]["sha256"] == col.load().fingerprint and a["unmodelled"]
    with pytest.raises(ValueError, match="pose must say"):
        sc.scene_record(_placed(0.8), {}, [{"name": "post", "kind": "sphere", "centre": [0, 0, 0], "r": 0.1}])


def test_the_registered_cameras_stand_in_front_of_the_base() -> None:
    cameras, source = sc.camera_obstacles()
    if not cameras:
        pytest.skip(f"no registered cameras here: {source}")
    assert all(ob["pose"] == "measured" and ob["size"] == "assumed" for ob in cameras)
    centres = np.array([ob["centre"] for ob in cameras])
    assert (centres[:, 0] > 0.5).all() and (centres[:, 2] > 0.0).all()
    assert source["edges"]


def test_one_scene_holds_both_arms_where_the_placement_puts_them() -> None:
    pytest.importorskip("mujoco")
    T = _placed(0.6)
    T[:3, :3] = kin._rpy_matrix(0.0, 0.0, 0.3)
    scene = sc.DualScene(sc.scene_record(T, {"source": "assumed"}, [sc.table_obstacle()]))
    rng = np.random.default_rng(0)
    q_r, q_l = rng.uniform(kin.Q_LOWER, kin.Q_UPPER, (2, 7))
    scene.set(q_r, q_l)
    tool = kin.urdf_fixed_transform("corenetic_socket_ee")
    right = scene.data.body("right/corenetic_socket_ee")
    left = scene.data.body("left/corenetic_socket_ee")
    assert np.allclose(right.xpos, (kin.NOMINAL.fk(q_r)[0] @ tool)[:3, 3], atol=1e-9)
    assert np.allclose(left.xpos, (T @ kin.NOMINAL.fk(q_l)[0] @ tool)[:3, 3], atol=1e-9)
    assert scene.env_geoms and len(scene.arm_geoms["right"]) == len(scene.arm_geoms["left"]) > 10


def test_the_capsule_gap_never_exceeds_the_mesh_gap() -> None:
    pytest.importorskip("mujoco")
    T = _placed(0.6)
    scene = sc.DualScene(sc.scene_record(T, {"source": "assumed"}, [sc.table_obstacle()]))
    rng = np.random.default_rng(3)
    q_r = rng.uniform(kin.Q_LOWER, kin.Q_UPPER, (60, 7))
    q_l = rng.uniform(kin.Q_LOWER, kin.Q_UPPER, (60, 7))
    gaps = col.clearance(col.load(), q_r, q_l, T)["arm_arm_m"]
    check = sc.mesh_check(scene, q_r, q_l, gaps, within_m=0.25)
    assert check["frames_checked"] > 5
    assert check["capsule_excess_mm"] is not None and check["capsule_excess_mm"] <= 0.0
