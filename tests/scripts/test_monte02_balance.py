"""monte02_balance (roadmap docs/monte02_left_arm_roadmap.html, T1-T2): Monte02's centre of mass, its support
polygon, the tip-over margins and the trunk posture whitelist.

The description is in the private opencv_kalibr submodule; without it these tests skip.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.fr3 import monte02_balance as mb, monte02_model as mm

pytestmark = pytest.mark.skipif(not mm.URDF.exists(), reason="opencv_kalibr submodule not checked out")
HANGING = np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
BENT = mm.Posture(trunk=(0.3, -0.4, 0.5), head=(0.2, -0.6), right_arm=(0.4, 0.3, -0.5, 1.2, 0.3, 0.4, -0.2))


def _random_left(n: int, seed: int) -> np.ndarray:
    lo, hi = mm.left_chain().q_limits(margin_deg=0.0)
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 7))


# --- the model ----------------------------------------------------------------------------------------------


def test_the_support_polygon_is_the_drive_axle_in_front_and_the_casters_behind() -> None:
    base = mb.support()
    assert base.floor_z == pytest.approx(-0.0865)  # the drive wheels' radius below their axles
    edges = {name: (n, c) for name, n, c in base.edges()}
    assert sorted(edges) == ["front", "left", "rear", "right"]
    np.testing.assert_allclose(edges["front"][0], [1.0, 0.0], atol=1e-12)
    assert edges["front"][1] == pytest.approx(0.0)  # nothing in front of the drive axle holds it up
    assert edges["rear"][1] == pytest.approx(0.345)
    for corner in base.corners:  # every corner on its two edges, inside the others
        d = np.array([c - n @ corner for _, n, c in base.edges()])
        assert np.sum(np.abs(d) < 1e-12) == 2 and np.all(d > -1e-12)


def test_the_centre_of_mass_is_mujoco_s() -> None:
    """The whole tree walked here against MuJoCo's subtree centre of mass, in two postures. The MJCF gives the
    massless gripper links 1 g (M1.4), hence the 0.2 mm."""
    mujoco = pytest.importorskip("mujoco")
    for posture in (mm.Posture(), BENT):
        model = mm.mujoco_model(posture)
        data = mujoco.MjData(model)
        q = _random_left(5, seed=3)
        ours = mb.MassModel().com(posture, q)
        for n in range(len(q)):
            data.qpos[:] = 0.0
            for name, value in zip(mm.LEFT_ARM_JOINTS, q[n], strict=True):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            np.testing.assert_allclose(data.subtree_com[model.body(mm.BASE_LINK).id], ours[n], atol=2e-4)


def test_a_payload_pulls_the_centre_of_mass_towards_the_hand() -> None:
    model, posture, q = mb.MassModel(), mm.Posture(), _random_left(4, seed=4)
    without, with_kg = model.com(posture, q), model.com(posture, q, (mb.payload(2.0),))
    tcp = mm.left_chain().fk(q)[:, :3, 3]
    np.testing.assert_allclose(with_kg, (model.total * without + 2.0 * tcp) / (model.total + 2.0), atol=1e-12)
    extra = mb.MassModel(extra=(mb.Load("chassis battery", 10.0, mm.BASE_LINK, (-0.2, 0.0, 0.05)),))
    assert extra.total == pytest.approx(model.total + 10.0) and extra.fingerprint != model.fingerprint


def test_margins_and_the_acceleration_that_tips_it() -> None:
    base = mb.support()
    h = 0.6
    com = np.array([[-0.1, 0.0, base.floor_z + h], [0.0, 0.0, base.floor_z + h], [0.05, 0.0, base.floor_z + h]])
    m = mb.margins(com)
    front = m["edges"].index("front")
    np.testing.assert_allclose(m["margin_m"][:, front], [0.1, 0.0, -0.05], atol=1e-12)
    np.testing.assert_allclose(m["accel_mps2"][0, front], mb.G * 0.1 / h)
    assert m["min_edge"] == ["front"] * 3 and m["min_m"][2] < 0.0


def test_bending_forward_with_the_arm_out_tips_it_over_the_drive_axle() -> None:
    """The roadmap's table: upright the centre of mass is ~0.16 m behind the axle; the knee bent 0.5 rad puts it
    past the axle with the arm only hanging."""
    model, load = mb.MassModel(), (mb.payload(1.0),)
    upright = mb.margins(model.com(mm.Posture(), HANGING, load))
    knee = mb.margins(model.com(mm.Posture(trunk=(0.5, 0.0, 0.0)), HANGING, load))
    front = upright["edges"].index("front")
    assert 0.14 < upright["margin_m"][0, front] < 0.18
    assert knee["margin_m"][0, front] < 0.0 and knee["min_edge"] == ["front"]


# --- the worst case -----------------------------------------------------------------------------------------


def test_no_random_pose_comes_nearer_an_edge_than_the_worst_case() -> None:
    posture, load, model = mm.Posture(trunk=(0.0, 0.2, 0.6)), (mb.payload(0.2),), mb.measured()
    w = mb.worst(posture, load)  # the weighed hand by default
    lo, hi = mb._box()
    x = np.random.default_rng(7).uniform(lo, hi, size=(20000, 9))
    m = mb.margins(model.com(posture, x[:, :7], load, head=x[:, 7:]))
    for e, name in enumerate(m["edges"]):
        assert m["margin_m"][:, e].min() >= w[name]["margin_m"] - 1e-6, name
        again = mb.margins(model.com(posture, np.array([w[name]["q_left"]]), load,
                                              head=np.array([w[name]["head"]])))
        assert again["margin_m"][0, e] == pytest.approx(w[name]["margin_m"], abs=1e-9)
        assert np.all(w[name]["q_left"] >= lo[:7]) and np.all(w[name]["q_left"] <= hi[:7])


def test_the_worst_case_under_a_requirement_weighs_the_height() -> None:
    """A deceleration takes more margin the higher the centre of mass: the search minimises margin - k * height,
    and no random pose leaves less."""
    posture, load, model = mm.Posture(), (mb.payload(0.2),), mb.measured()
    req = mb.Requirement(model_error_m=0.01, decel_mps2=1.0, tilt_rad=0.02)
    assert req.margin_m(0.6) == pytest.approx(0.01 + 0.6 * (np.tan(0.02) + 1.0 / mb.G))
    w = mb.worst(posture, load, height_coeff=req.height_coeff)
    lo, hi = mb._box()
    x = np.random.default_rng(8).uniform(lo, hi, size=(20000, 9))
    m = mb.margins(model.com(posture, x[:, :7], load, head=x[:, 7:]))
    for e, name in enumerate(m["edges"]):
        spare = m["margin_m"][:, e] - req.height_coeff * m["height_m"]
        assert spare.min() >= w[name]["spare_m"] - 1e-6, name
        assert w[name]["spare_m"] == pytest.approx(w[name]["margin_m"] - req.height_coeff * w[name]["height_m"])
    with pytest.raises(ValueError, match=">= 0"):
        mb.Requirement(model_error_m=-0.01, decel_mps2=1.0)


def test_leaning_forward_fails_upright_passes() -> None:
    assert mb.verify((0.0, 0.0, 0.0))["passes"]  # screened against SCREENING_MARGIN_M
    hip = mb.verify((0.0, 0.4, 0.0))
    assert not hip["passes"] and hip["worst_edge"] == "front" and hip["basis"] == "screening"
    with pytest.raises(ValueError, match="short at the front edge"):
        mb.build([("upright", (0.0, 0.0, 0.0)), ("hip", (0.0, 0.4, 0.0))])
    with pytest.raises(ValueError, match="outside"):
        mb.verify((2.0, 0.0, 0.0))
    # a hard enough e-stop fails even upright with the arm anywhere: T3 then splits driving off
    hard = mb.verify((0.0, 0.0, 0.0), mb.Requirement(model_error_m=0.01, decel_mps2=3.0))
    assert not hard["passes"] and hard["basis"] == "requirement"


# --- the whitelist ------------------------------------------------------------------------------------------


def test_the_whitelist_in_the_submodule_is_consistent_but_not_in_force() -> None:
    """No measured requirement yet: the entries are screened, and a session is refused whatever its posture."""
    assert mb.check() == []
    data = mb.load_whitelist()
    assert data["mass_model"] == mb.measured().fingerprint != mb.MassModel().fingerprint
    assert data["requirement"] is None and data["payload_kg"] >= mb.PAYLOAD_KG
    upright = mm.Posture(trunk=tuple(data["postures"][0]["trunk"]))
    assert [r for r in mb.refusals(upright) if "not in force" in r]


def test_the_weighed_hand_is_the_urdf_s_plus_the_difference_at_the_tcp() -> None:
    hand = mb.hand_correction()
    assert 0.7 < mb.urdf_hand_kg() < 0.8  # sensor flange, gripper, fingers, camera mount
    assert hand.link == mm.TCP_LINK and hand.mass == pytest.approx(mb.HAND_MEASURED_KG - mb.urdf_hand_kg())
    assert mb.measured().total == pytest.approx(mb.MassModel().total + hand.mass)
    assert mb.hand_correction(0.5).mass == 0.0  # lighter than the URDF's: not subtracted


@pytest.fixture(scope="module")
def in_force(tmp_path_factory) -> object:
    """A whitelist verified against a (made-up) measured requirement."""
    path = tmp_path_factory.mktemp("whitelist") / "trunk_whitelist.json"
    data = mb.build([("upright", (0.0, 0.0, 0.0))], mb.Requirement(model_error_m=0.01, decel_mps2=0.5))
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_session_posture_off_the_whitelist_is_refused(in_force) -> None:
    assert mb.check(in_force) == []
    assert mb.refusals(mm.Posture(), in_force) == []
    assert mb.refusals(mm.Posture(head=(0.3, -1.2)), in_force) == []  # verified over the head's range
    assert any("not in the whitelist" in r for r in mb.refusals(mm.Posture(trunk=(0.0, 0.4, 0.0)), in_force))
    assert any("not in the whitelist" in r for r in mb.refusals(mm.Posture(trunk=(1e-6, 0.0, 0.0)), in_force))
    moved = mm.Posture(right_arm=(0.4, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0))
    assert any("not parked" in r for r in mb.refusals(moved, in_force))


def test_a_changed_mass_model_voids_the_whitelist(in_force, tmp_path) -> None:
    weighed = mb.MassModel(extra=(mb.Load("weighing correction", 3.0, mm.BASE_LINK, (0.05, 0.0, 0.0)),))
    assert any("mass model changed" in p for p in mb.check(in_force, model=weighed))
    assert any("another mass model" in r for r in mb.refusals(mm.Posture(), in_force, model=weighed))
    data = json.loads(in_force.read_text(encoding="utf-8"))
    data["postures"][0]["worst_spare_m"] += 0.01
    edited = tmp_path / "whitelist.json"
    edited.write_text(json.dumps(data), encoding="utf-8")
    assert any("recorded worst spare" in p for p in mb.check(edited))
