"""dual_fr3_compliant_handle (roadmap D6 step 2): the handle's spring is the rubber's, and the budget reads it.

The buffer against the closed forms of a Timoshenko cantilever; the pair against how it must scale and where
it must be soft; the TCP form against the frame it is used in; the sim tray against the drawing. The CAD itself
runs under FreeCAD and is not tested here.
"""

# ruff: noqa: N806  (K, C as in the module)
import math
from dataclasses import replace

import numpy as np
import pytest

from tests.scripts.test_dual_fr3_closed_chain import TOOL, _rigid_pair
from tools.fr3 import (
    dual_fr3_closed_chain as cc,
    dual_fr3_compliant_handle as ch,
    dual_fr3_coop_carry as cp,
)


def test_gents_modulus_rises_with_the_hardness() -> None:
    E = [ch.youngs_modulus_mpa(s) for s in (40, 50, 60, 70)]
    assert sorted(E) == E
    assert 1.4 < E[0] < 2.0 and 2.2 < E[1] < 2.7  # the textbook values for 40A, 50A


def test_a_buffer_is_a_timoshenko_cantilever() -> None:
    b = ch.Buffer(diameter_mm=20.0, height_mm=20.0, shore_a=50.0)
    K = ch.buffer_stiffness(b)
    assert np.allclose(K, K.T) and np.all(np.linalg.eigvalsh(K) > 0)
    E = ch.youngs_modulus_mpa(50.0) * 1e6
    G, d, h = E / 3.0, 0.02, 0.02
    A, second_moment = math.pi * d**2 / 4, math.pi * d**4 / 64
    # pushed sideways with its top free to turn: bending and shear in series
    tip = 1.0 / (h**3 / (3 * E * second_moment) + h / (ch.SHEAR_COEFFICIENT * G * A))
    for u, r in ((0, 4), (1, 3)):
        condensed = K[u, u] - K[u, r] ** 2 / K[r, r]
        assert condensed == pytest.approx(tip, rel=1e-9)
    # pushed along +x the top tips towards +x (about +y), along +y away from it (about -x)
    assert -K[0, 4] / K[4, 4] > 0 and -K[1, 3] / K[3, 3] < 0
    assert K[2, 2] > 3 * K[0, 0]  # stiffer in compression than in shear


def test_the_pair_is_soft_where_the_squeeze_is_and_scales_with_the_rubber() -> None:
    h = ch.Handle()
    K = ch.handle_stiffness(h)
    assert np.allclose(K, K.T) and np.all(np.linalg.eigvalsh(K) > 0)
    assert np.allclose(ch.handle_stiffness(h, 1.3), 1.3 * K)
    c = ch.characteristics(h, 1.4)
    # rolling about the bar is the soft way, tipping along it the stiff one: the tray stays level
    assert c["k_nm_per_rad"]["x"] < c["k_nm_per_rad"]["z"] < c["k_nm_per_rad"]["y"]
    assert c["k_n_per_mm"]["y"] < c["k_n_per_mm"]["x"] < c["k_n_per_mm"]["z"]
    assert c["tip_deg_per_cm_offset"] < 0.2 and c["weight_sag_mm"] < 0.5
    wide = ch.characteristics(replace(h, buffer_pitch_mm=2 * h.buffer_pitch_mm), 1.4)
    assert wide["k_nm_per_rad"]["y"] > 3 * c["k_nm_per_rad"]["y"]  # the pitch's square
    assert 3.0 < ch.turn_stop_deg(h) < 4.5


def test_the_tcp_form_is_the_bar_spring_seen_from_the_tcp() -> None:
    h = ch.Handle()
    C = ch.tcp_compliance(h)
    assert np.allclose(C, C.T)
    C_bar = np.linalg.inv(ch.handle_stiffness(h))
    # a pure moment turns the bar the same, whatever point it is read at; only the axes flip (y, z)
    flip = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(C[3:, 3:], flip @ C_bar[3:, 3:] @ flip)
    # the TCP sits below the bar centre: a force along the bar there pitches the bar more than at the centre
    assert abs(C[0, 4]) != pytest.approx(abs(C_bar[0, 4]))


def test_the_budget_takes_a_designed_grasp() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=20.0)
    held = np.ones(len(q_r), bool)
    eps = np.eye(4)
    eps[:3, 3] = [0.0, 0.01, 0.0]
    m = cc.relative_mismatch(q_r, q_l, held, T_RL, TOOL, T_true=T_RL @ eps)
    chain = cc.ClosedChain(q_r, q_l, m["frames"], m["midpoint"], T_RL, TOOL)
    # an isotropic stiffness and its own matrix are the same grasp
    k = 5000.0
    as_matrix = np.diag([1 / k] * 3 + [1 / (k * cc.GRASP_LEVER_M**2)] * 3)
    assert np.allclose(
        chain.internal_wrench(m["twist"], 1.0, k), chain.internal_wrench(m["twist"], 1.0, as_matrix)
    )
    with pytest.raises(ValueError, match="6, 6"):
        cc.grasp_compliance(chain.T_R, np.eye(3))
    # the handle takes the squeeze down, and gives what the arms do not
    C = ch.tcp_compliance(ch.Handle())
    rigid = np.linalg.norm(chain.internal_wrench(m["twist"])[:, :3], axis=1)
    soft = chain.internal_wrench(m["twist"], 1.0, C)
    assert np.all(np.linalg.norm(soft[:, :3], axis=1) <= rigid + 1e-9)
    give = chain.grasp_give(soft, C)
    assert np.all(give >= 0) and give[..., 0].max() * 1e3 < 10.0
    plan = {"q_right": q_r, "q_left": q_l, "held": held, "T_right_left": T_RL, "tool": TOOL, "fps": 20.0}
    report = cc.budget(plan, samples=20, designs={"tight": {**ch.design(ch.Handle()), "travel_mm": 1e-6}})
    for scenario in report["scenarios"].values():
        (row,) = scenario["designs"]
        assert row["design"] == "tight" and not row["ok"]  # a handle with no travel left is rigid again
        assert row["travel_deg"] == pytest.approx(ch.turn_stop_deg(ch.Handle()))


def test_choose_ranks_what_fits_first() -> None:
    q_r, q_l, T_RL = _rigid_pair(turn_deg=20.0)
    plan = {
        "q_right": q_r,
        "q_left": q_l,
        "held": np.ones(len(q_r), bool),
        "T_right_left": T_RL,
        "tool": TOOL,
        "fps": 20.0,
    }
    handles = [replace(ch.Handle(), buffer=ch.Buffer(shore_a=s)) for s in (40.0, 70.0)]
    report = ch.choose(plan, handles=handles, samples=20)
    assert [r["name"] for r in report["ranked"]] and len(report["ranked"]) == 2
    keys = [(not r["fits"], not r["fits_others"], -r["margin_others"]) for r in report["ranked"]]
    assert keys == sorted(keys)
    for r in report["ranked"]:
        assert set(r["nominal"]) == {"assumed", "measured_b3"} and len(r["spread"]["measured_b3"]) == 3


def test_the_sim_tray_is_the_drawn_one() -> None:
    tray, handle, plate = cp.Tray(), ch.Handle(), ch.TrayPlate()
    assert tray.bar_y == pytest.approx(plate.handle_y_mm / 1e3)
    assert tray.body_half[2] + tray.bar_z == pytest.approx(plate.grip_height_mm / 1e3)
    assert np.allclose(
        2e3 * np.asarray(tray.bar_half), [handle.bar_length_mm, handle.grip_width_mm, handle.bar_height_mm]
    )
    assert np.allclose(
        2e3 * np.asarray(tray.body_half), [plate.length_mm, plate.width_mm, plate.thickness_mm]
    )
    assert 1.2 < ch.tray_mass_kg(plate, handle) < 1.6


def test_meshes_are_coloured_by_part(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ch.mesh_list(tmp_path)
    (tmp_path / "meshes").mkdir()
    for name in ("plate", "right_bar", "left_buffer_0", "right_shoulder_screw_2"):
        (tmp_path / "meshes" / f"{name}.stl").write_bytes(b"")
    colours = {p.stem: c for p, c in ch.mesh_list(tmp_path)}
    assert colours["plate"] == ch.MESH_COLORS["plate"] and colours["right_bar"] == ch.MESH_COLORS["bar"]
    assert (
        colours["left_buffer_0"] == ch.MESH_COLORS["buffer"] and colours["right_shoulder_screw_2"] == ch.STEEL
    )
