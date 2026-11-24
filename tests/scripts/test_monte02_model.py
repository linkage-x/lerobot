"""monte02_model (roadmap docs/monte02_left_arm_roadmap.html, M1.4): Monte02's MJCF, the session posture and
the capsule model of the left arm against the rest of the robot.

The description is in the private opencv_kalibr submodule; without it (or without MuJoCo) these tests skip.
"""

# ruff: noqa: N806
from __future__ import annotations

import importlib.util
from collections import defaultdict

import numpy as np
import pytest

from tools.fr3 import dual_fr3_collision as fr3col, monte02_model as mm

pytestmark = pytest.mark.skipif(not mm.MJCF.exists(), reason="opencv_kalibr submodule not checked out")
#: Not the default: trunk bent, head turned, right arm moved -- every locked joint off zero.
BENT = mm.Posture(trunk=(0.3, -0.4, 0.5), head=(0.2, -0.6), right_arm=(0.4, 0.3, -0.5, 1.2, 0.3, 0.4, -0.2))


@pytest.fixture(scope="module")
def mujoco():
    return pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def capsules() -> mm.CapsuleModel:
    return mm.load()


def _random_left(n: int, seed: int) -> np.ndarray:
    lo, hi = mm.left_chain().q_limits(margin_deg=0.0)
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 7))


def _set(model, data, mujoco, q_left, posture=None) -> None:
    data.qpos[:] = 0.0
    values = dict(zip(mm.LEFT_ARM_JOINTS, q_left, strict=True)) | (posture.locked() if posture else {})
    for name, value in values.items():
        joint = model.joint(name)  # KeyError when the joint was folded away: the caller passed the wrong posture
        data.qpos[model.jnt_qposadr[joint.id]] = value
    mujoco.mj_kinematics(model, data)


# --- the MJCF -----------------------------------------------------------------------------------------------


def test_the_mjcf_is_what_build_mjcf_makes_and_loads(mujoco) -> None:
    spec = importlib.util.spec_from_file_location("monte02_build_mjcf", mm.DESCRIPTION / "build_mjcf.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.check() == []


def test_the_mjcf_bodies_are_the_urdf_link_frames(mujoco) -> None:
    """Unlocked MJCF with the posture set as joint values, and the posture folded in: both are the URDF."""
    full = mujoco.MjModel.from_xml_path(str(mm.MJCF))
    locked = mm.mujoco_model(BENT)
    assert {full.joint(j).name for j in range(full.njnt)} - {locked.joint(j).name for j in range(locked.njnt)} == set(
        BENT.locked()
    )
    d_full, d_locked = mujoco.MjData(full), mujoco.MjData(locked)
    links = sorted({link for link, _, _ in mm.PARTS.values()} | {mm.TCP_LINK})
    for q in _random_left(5, seed=1):
        _set(full, d_full, mujoco, q, BENT)
        _set(locked, d_locked, mujoco, q)
        for link in links:
            T = mm._chain_to(link, BENT).fk(q[None, : mm._chain_to(link, BENT).dof])[0]
            for model, data in ((full, d_full), (locked, d_locked)):
                body = model.body(link).id
                # 1e-6: the MJCF writes its quaternions with 6-7 digits (0.2 um at the head)
                np.testing.assert_allclose(data.xpos[body], T[:3, 3], atol=1e-6)
                np.testing.assert_allclose(data.xmat[body].reshape(3, 3), T[:3, :3], atol=1e-6)


def test_the_fingerprint_binds_the_posture() -> None:
    assert mm.model_fingerprint() == mm.model_fingerprint(mm.Posture())
    assert mm.model_fingerprint(BENT) != mm.model_fingerprint()
    nudged = mm.Posture(trunk=(0.3, -0.4, 0.5 + 1e-6), head=BENT.head, right_arm=BENT.right_arm)
    assert mm.model_fingerprint(nudged) != mm.model_fingerprint(BENT)


def test_a_posture_outside_the_limits_is_refused(mujoco) -> None:
    with pytest.raises(ValueError, match="outside"):
        mm.mujoco_model(mm.Posture(right_arm=(0.0,) * 7))  # elbow 0 < its 0.087 lower limit
    with pytest.raises(ValueError, match="outside"):
        mm.left_chain(mm.Posture(trunk=(2.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="3 values"):
        mm.Posture(trunk=(0.0, 0.0))


# --- the capsules -------------------------------------------------------------------------------------------


def test_the_capsules_were_fitted_to_the_mjcf_in_the_submodule(capsules) -> None:
    assert capsules.fingerprint == fr3col.model_fingerprint(mm.MJCF)["sha256"]


def test_each_part_s_capsules_enclose_every_vertex_of_its_meshes(mujoco, capsules) -> None:
    vertices = mm.part_vertices()
    vertices["trunk_base"] = vertices["trunk_base"][mm._base_z("trunk_base", vertices["trunk_base"]) > mm.CHASSIS_TOP]
    for part, points in vertices.items():
        idx = [k for k, p in enumerate(capsules.parts) if p == part]
        assert idx, part
        outside = np.min(
            [fr3col._point_segment_distance(points, capsules.a[k], capsules.b[k]) - capsules.r[k] for k in idx], axis=0
        )
        assert outside.max() <= 1e-9, f"{part}: a vertex {outside.max() * 1e3:.3f} mm outside its capsules"


def test_the_chassis_is_below_its_top_or_in_the_column_s_capsules(mujoco, capsules) -> None:
    points = mm.part_vertices(parts=mm.CHASSIS)["chassis"]  # in link_b0_base = the base frame
    column = [k for k, p in enumerate(capsules.parts) if p == "trunk_base"]
    T = mm._chain_to("link_t0_trunk_base", mm.Posture()).fk(np.zeros((1, 0)))[0]
    local = (points - T[:3, 3]) @ T[:3, :3]
    inside = np.min(
        [fr3col._point_segment_distance(local, capsules.a[k], capsules.b[k]) - capsules.r[k] for k in column], axis=0
    )
    assert np.all((points[:, 2] <= mm.CHASSIS_TOP) | (inside <= 1e-9))


def test_every_mesh_contact_of_the_left_arm_is_a_capsule_overlap(mujoco, capsules) -> None:
    """MuJoCo (convex hulls) against the capsules, random arm poses in two postures: wherever the hulls touch,
    the capsules round them overlap -- unless the pair is one the fit excluded."""
    part_of: dict[str, list[str]] = defaultdict(list)
    for part, (_, _, bodies) in (mm.PARTS | mm.CHASSIS).items():
        for body in bodies:
            part_of[body].append(part)
    excluded = {(a, b) for a, b, _ in capsules.excluded}
    names = capsules.gap_names()
    part_by_name = dict(zip(capsules.names, capsules.parts, strict=True)) | {"chassis": "chassis"}
    columns = defaultdict(list)
    for k, (x, y) in enumerate(names):
        columns[(part_by_name[x], part_by_name[y])].append(k)
    left = set(mm.LEFT_INDEX)
    checked = 0
    for posture in (mm.Posture(), BENT):
        model = mm.mujoco_model(posture)
        data = mujoco.MjData(model)
        q = _random_left(1500, seed=2)
        gaps = capsules.pair_gaps(q, posture)
        for n in range(len(q)):
            _set(model, data, mujoco, q[n])
            mujoco.mj_collision(model, data)
            for c in data.contact[: data.ncon]:
                if c.dist >= 0.0:
                    continue
                b1, b2 = (model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
                for p1 in part_of[b1]:
                    for p2 in part_of[b2]:
                        pair = (p1, p2) if p2 in left else (p2, p1)
                        if pair[1] not in left or pair in excluded or pair not in columns:
                            continue
                        alternatives = [pair]
                        if "chassis" in (p1, p2):  # the chassis top or the column's capsules
                            alternatives.append(("trunk_base", pair[1]))
                        ks = [k for alt in alternatives for k in columns.get(alt, [])]
                        assert gaps[n, ks].min() < 0.0, (pair, q[n].round(3).tolist(), c.dist)
                        checked += 1
    assert checked > 50  # the arm does reach the body in random poses


def test_reaching_forward_is_clear_and_through_the_chest_is_not(capsules) -> None:
    forward = [0.9, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0]  # TCP ~0.5 m in front of the shoulder
    across = [1.0, -0.1, -1.2, 2.0, 0.0, 0.0, 0.0]  # the hand folded back into the chest
    out = [0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 0.0]  # hanging, swung out sideways
    c = capsules.clearance(np.array([forward, across, out]))
    assert c["self_m"][0] > 0.05, c
    assert c["self_m"][1] < 0.0 and c["self_pair"][1][0].startswith("chest"), c
    assert c["self_m"][2] > 0.05, c


def test_the_gaps_follow_the_posture_and_the_chassis_top_keeps_the_hand_up(capsules) -> None:
    """Folded forward at the hip, an arm pitched forward reaches down past the chassis top; upright it does not.
    The capsules hang on the locked chain of the posture, not on posture zero."""
    q = np.array([[0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    folded = mm.Posture(trunk=(0.0, 1.2, 0.0))
    upright, bent = capsules.clearance(q), capsules.clearance(q, folded)
    assert mm.left_chain().fk(q)[0, 2, 3] > 0.4 and upright["self_m"][0] > 0.05, upright
    assert mm.left_chain(folded).fk(q)[0, 2, 3] < mm.CHASSIS_TOP
    assert bent["self_m"][0] < 0.0 and bent["self_pair"][0][0] == "chassis", bent


# --- the embodiment description (M1.5) ----------------------------------------------------------------------


def test_the_embodiment_description_is_this_model() -> None:
    """embodiments/monte02_left.json and this module name the same URDF, chain, joints and capsules."""
    from tools.thor.gmsl2 import embodiment as em

    description = em.load(em.EMBODIMENT_DIR / "monte02_left.json", repo_root=mm.REPO)
    arm = description["arms"]["left"]
    assert mm.REPO / arm["kinematics"]["urdf"] == mm.URDF
    assert mm.REPO / arm["collision"] == mm.CAPSULE_FILE
    assert (arm["kinematics"]["base_link"], arm["kinematics"]["tcp_link"]) == (mm.BASE_LINK, mm.TCP_LINK)
    assert tuple(arm["free_joints"]) == mm.LEFT_ARM_JOINTS == mm.left_chain().joint_names
    assert tuple(arm["session_locked_joints"]) == tuple(mm.Posture().locked())
