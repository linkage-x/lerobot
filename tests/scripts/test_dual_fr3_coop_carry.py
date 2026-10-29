"""dual_fr3_coop_carry (roadmap D6): the scripted tray carry is rigid where the hands hold the tray.

The path only (no MuJoCo, no retarget): the grip opening and pad drop are given. The retarget and the render
run in the tool itself (outputs/fr3/dual_fr3_coop_carry/).
"""

# ruff: noqa: N806  (T_right_left as in the runtime)
import numpy as np

from tools.fr3 import dual_fr3_closed_chain as cc, dual_fr3_coop_carry as cp, fr3_kinematics as kin

OPENING, PAD_DROP = 0.149, 0.0023


def _path(**carry):
    return cp.carry_targets(cp.Carry(**carry), cp.Tray(), OPENING, PAD_DROP)


def test_the_hands_hold_one_rigid_tray_through_the_carry_and_the_turn() -> None:
    path = _path()
    T_RL, held = path["T_right_left"], path["held"]
    segments = cc.held_segments(held)
    assert len(segments) == 1
    T_R = path["targets"]["right"]
    T_L = T_RL @ path["targets"]["left"]
    M = np.linalg.inv(T_R[held]) @ T_L[held]
    # 10-29: interpolating each hand on its own cut the chord of the 22 deg turn, 7 mm between the hands
    assert np.abs(M - M[0]).max() < 1e-9
    a, b = segments[0]
    assert path["opening"][a] == OPENING and path["opening"][a - 1] > OPENING  # held from the fingers seating
    assert np.allclose(path["opening"][held], OPENING) and path["opening"][b] > OPENING  # until they open
    # the tray turns and moves while held
    assert np.ptp(path["tray"][held, 3]) > np.radians(20) and np.ptp(path["tray"][held, 2]) > 0.14


def test_the_hands_are_on_the_bars_and_point_down() -> None:
    path, tray = _path(), cp.Tray()
    held = path["held"]
    k = np.flatnonzero(held)[0]
    x, y, z, yaw = path["tray"][k]
    T_tray = cp.tray_pose(x, y, z, yaw)
    for arm, sign in (("right", -1.0), ("left", 1.0)):
        T = path["targets"][arm][k] if arm == "right" else path["T_right_left"] @ path["targets"][arm][k]
        local = np.linalg.inv(T_tray) @ T
        assert np.allclose(local[:3, 3], [0.0, sign * tray.bar_y, tray.bar_z - PAD_DROP])
        assert T[2, 2] < -0.99  # the TCP's z down


def test_the_carry_starts_and_ends_at_the_ready_pose_in_each_base() -> None:
    path = _path(spacing_m=0.7, start=(0.62, 0.35, 0.0), goal=(0.56, 0.28, 22.0))
    ready = kin.NOMINAL.fk(kin.Q_READY[None])[0] @ kin.urdf_fixed_transform(cp.TOOL_FRAME)
    for arm in ("right", "left"):
        assert np.allclose(path["targets"][arm][-1], ready)
    assert path["T_right_left"][1, 3] == 0.7
    assert len(path["held"]) == round(sum(
        getattr(cp.Carry(), f) for f in ("approach_s", "descend_s", "close_s", "lift_s", "carry_s", "settle_s",
                                         "lower_s", "open_s", "retreat_s", "home_s")
    ) * cp.Carry().fps)  # fmt: skip
