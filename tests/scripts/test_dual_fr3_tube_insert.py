"""dual_fr3_tube_insert (roadmap D7): one hand holds a rack, the other puts a tube in, stepped under contact.

The geometry without MuJoCo; then a few single runs on the compiled scene (each a few seconds), picked from the
sweep where the outcome is not marginal. The sweeps and the videos run in the tool itself
(outputs/fr3/dual_fr3_tube_insert/).
"""

# ruff: noqa: N803, N806  (T_ for poses, as in the tool)
import math

import numpy as np
import pytest

from tools.fr3 import dual_fr3_tube_insert as ti

PADS = {"right": (0.020, 0.0023), "left": (0.008, 0.0023)}


def _geometry(case: str = "tube") -> ti.Geometry:
    return ti.geometry(ti.CASES[case], ti.Rack(), ti.Layout(), PADS)


def test_the_planned_tip_is_where_the_right_tcp_is_sent() -> None:
    geo = _geometry()
    tip = geo.T_hole[:3, 3] + [0.0, 0.0, 0.02]
    T = ti.right_tcp_for_tip(geo, tip)
    T_tube = T @ geo.T_tcp_tube
    # the tube hangs down the TCP's z (down): its tip half a length past its middle
    half = ti.CASES["tube"].peg.length_m / 2.0
    assert np.allclose(T_tube[:3, 3] + T_tube[:3, :3] @ [0.0, 0.0, half], tip)
    assert T[2, 2] < -0.999
    # the two hands face away from each other's wrist mounts: turned half round about z
    assert np.allclose(ti.R_HAND["right"][:, :2], -ti.R_HAND["left"][:, :2])


def test_the_rack_offset_is_in_the_holding_hand_and_the_tilt_turns_about_the_pads() -> None:
    geo = _geometry()
    exact = ti.grasp_poses(geo, PADS["right"][1], ti.Error())
    off = ti.grasp_poses(geo, PADS["right"][1], ti.Error(dx_mm=3.0, dy_mm=-1.0))
    T_rack = geo.T_left_tcp @ off["left"]
    assert np.allclose(T_rack[:3, 3] - geo.T_rack[:3, 3], [0.003, -0.001, 0.0])
    assert np.allclose(geo.T_left_tcp @ exact["left"], geo.T_rack)

    tilted = ti.grasp_poses(geo, PADS["right"][1], ti.Error(tilt_deg=2.0, tilt_dir_deg=0.0))["right"]
    pad = np.array([0.0, 0.0, -PADS["right"][1], 1.0])
    # the point between the pads stays put (in the TCP's frame) ...
    def tube_point(T, z):
        return T @ np.array([0.0, 0.0, z, 1.0])
    z_pad = (np.linalg.inv(exact["right"]) @ pad)[2]
    assert np.allclose(tube_point(tilted, z_pad), tube_point(exact["right"], z_pad))
    # ... and the tip swings toward the TCP's +x by the angle over the lever
    tip_z = ti.CASES["tube"].peg.length_m / 2.0
    lever = tip_z - z_pad
    d = tube_point(tilted, tip_z) - tube_point(exact["right"], tip_z)
    assert d[0] == pytest.approx(lever * math.sin(math.radians(2.0)), rel=1e-6)
    assert abs(d[1]) < 1e-12


def test_the_summary_counts_each_outcome() -> None:
    def result(outcome, force, diverged=False):
        return {
            "run": {"case": "vial", "controller": "joint", "error": {"dx_mm": 3.0, "dy_mm": 4.0, "tilt_deg": 0.0}},
            "outcome": outcome, "seated": outcome == "seated", "peak_force_n": force,
            "over_reflex": force > ti.REFLEX_FORCE_N, "final_lateral_force_n": 1.0,
            "time_to_seat_s": 2.0 if outcome == "seated" else None, "diverged": diverged,
        }  # fmt: skip

    (row,) = ti.summarize([result("seated", 3.0), result("missed", 25.0), result("missed", 12.0, True)])
    assert row["offset_mm"] == 5.0 and row["runs"] == 3 and row["seated"] == 1
    assert row["outcomes"] == {"seated": 1, "jammed": 0, "missed": 2, "gave_up": 0}
    assert row["over_reflex"] == 1 and row["diverged"] == 1 and row["time_to_seat_s_median"] == 2.0


# ------------------------------------------------------------------ on the compiled scene ---


@pytest.fixture(scope="module")
def tube_sim():
    pytest.importorskip("mujoco")
    sim = ti.build(ti.CASES["tube"])
    return sim, ti.start_joints(sim)


@pytest.fixture(scope="module")
def vial_sim():
    pytest.importorskip("mujoco")
    sim = ti.build(ti.CASES["vial"])
    return sim, ti.start_joints(sim)


def test_the_start_holds_the_rack_where_planned_and_the_tip_over_the_hole(tube_sim) -> None:
    sim, q = tube_sim
    ti.place(sim, q, ti.Error())
    T_tcp = ti.site_pose(sim, sim.data, sim.site["right_tcp"])
    tip = T_tcp[:3, 3] + T_tcp[:3, :3] @ [0.0, 0.0, sim.geo.tip_m]
    assert np.allclose(tip, sim.geo.T_hole[:3, 3] + [0.0, 0.0, sim.layout.hover_m], atol=1e-5)
    assert np.allclose(ti._object_pose(sim, "rack"), sim.geo.T_rack, atol=1e-5)
    # the Cartesian controller with its EE at the tip holds the tip with exactly its K
    K = ti.equivalent_stiffness(sim, sim.data, ti.CONTROLLERS["cart_tip"])
    assert np.allclose(K, np.diag(ti.INSERT_STIFFNESS))
    assert np.allclose(ti.point_stiffness(K), ti.INSERT_STIFFNESS[:3])
    # centred on the TCP instead, a push on the tip also turns the hand: softer across by r^2 / k_rot
    K_tcp = ti.equivalent_stiffness(sim, sim.data, ti.CONTROLLERS["cart_tcp"])
    k_lat, k_rot = ti.INSERT_STIFFNESS[0], ti.INSERT_STIFFNESS[3]
    across = 1.0 / (1.0 / k_lat + sim.geo.tip_m**2 / k_rot)
    assert np.allclose(ti.point_stiffness(K_tcp), [across, across, ti.INSERT_STIFFNESS[2]], rtol=1e-6)
    # 10-30: K's own diagonal hides that (the same at every point of the hand), and calls the joint
    # controller 5-15 kN/m at the tip where a push meets about 1
    K_joint = ti.equivalent_stiffness(sim, sim.data, ti.CONTROLLERS["joint"])
    assert K_joint[0, 0] > 5000.0 and ti.point_stiffness(K_joint).max() < 1500.0


def test_a_round_bottomed_tube_slides_in_from_3_mm_under_joint_impedance(tube_sim) -> None:
    sim, q = tube_sim
    out = ti.simulate(sim, ti.Run("tube", "joint", error=ti.Error(dx_mm=3.0)), q_start=q)
    assert out["outcome"] == "seated" and not out["diverged"]
    assert out["peak_force_n"] < ti.REFLEX_FORCE_N


def test_a_flat_vial_on_the_top_face_is_missed_not_jammed(vial_sim) -> None:
    sim, q = vial_sim
    out = ti.simulate(sim, ti.Run("vial", "joint", error=ti.Error(dy_mm=-6.0)), q_start=q)
    # it never got past the rim's rounding: 10-30 such runs were called jammed
    assert out["outcome"] == "missed"
    assert out["max_depth_mm"] <= (ti.CASES["vial"].hole.rim_r_m + ti.IN_BORE_M) * 1e3
