#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
One hand holds a test-tube rack in the air, the other puts a tube into it: with contact, under the real
controllers (roadmap D7).

Unlike the tray carry (``dual_fr3_coop_carry``, drawn only), this is stepped: C2's two arms driven by torque at
1 kHz, as libfranka drives them, each by an emulation of a panda_py controller (``JointPosition``, what D1 runs
today, or ``CartesianImpedance``), with Franka's gravity compensation; the tube and the rack are free bodies
welded to the hands (a weld's give stands in for the grasp's), and the tube touches the rack's hole -- only
there: the arms, the table and the cameras do not collide (C3 keeps the arms apart; ``min_arm_arm_gap_mm`` says
by how much here).

The plan believes the rack is where it was meant to be grasped. It is not: the rack sits ``Error`` off in the
holding hand -- what B3's base-to-base error, a grasp that is a few mm off and the holding arm's sag under the
rack add up to -- and the tube may be tilted in the other hand. What the controller does with that miss is the
question: which one gets the tube in, with how much force, and whether a hole search (a spiral, driven only by
what the robot measures of itself) gets it in when compliance alone cannot.

Not modelled: the fingers' contact (the weld holds), joint friction beyond a Coulomb ``frictionloss`` per
joint, the controllers' Coriolis term, libfranka's torque-rate limits and reflexes (``peak_force_n`` says how
close a run came to Franka's default 20 N contact threshold), and any arm-object collision.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J, K follow the maths)
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import (  # noqa: E402
    dual_fr3_closed_chain as cc,
    dual_fr3_collision as col,
    dual_fr3_coop_carry as cp,
    dual_fr3_scene as sc,
)

SCHEMA = "dual_fr3_tube_insert/v1"
TCP_BODY = "corenetic_gripper_ee"
TCP_SITE = "corenetic_gripper_ee_site"
#: z down, the fingers closing along world x: both hands narrow along y, where the other hand is
R_DOWN_X = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
#: each hand's TCP orientation: R_DOWN_X, the right one turned half round its z so that its wrist's SMR/camera
#: mount points away from the other hand (facing it, it went 38 mm into the left arm's link 7; 62 mm clear now)
R_HAND = {"right": R_DOWN_X @ np.diag([-1.0, -1.0, 1.0]), "left": R_DOWN_X}
TIMESTEP_S = 1e-3
#: libfranka's FR3 torque limits per joint
TORQUE_LIMIT_NM = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
#: Franka's default contact threshold: a run above it would likely have stopped on a reflex
REFLEX_FORCE_N = 20.0
#: the tube's tip within this of the hole's floor is in
SEATED_TOLERANCE_M = 0.002
#: after the hold the fingers open and the arm stays this long: what was short of the floor drops, or is stuck
RELEASE_S = 0.6
#: past the rim's rounding by this is in the bore
IN_BORE_M = 0.001


# ------------------------------------------------------------------ the objects ---


@dataclass(frozen=True)
class Peg:
    """What goes in: a round-bottomed tube (a capsule) or a flat-bottomed vial (a cylinder)."""

    name: str
    radius_m: float
    length_m: float
    round_bottom: bool
    mass_kg: float
    #: the pads close this far below its top
    grip_from_top_m: float = 0.015
    friction: float = 0.3


@dataclass(frozen=True)
class Hole:
    """A blind hole down from the rack's top face, its rim rounded to ``rim_r_m`` (what a chamfer does)."""

    radius_m: float
    depth_m: float
    rim_r_m: float
    segments: int = 32


@dataclass(frozen=True)
class Case:
    peg: Peg
    hole: Hole
    note: str


CASES = {
    "tube": Case(
        Peg("16 x 100 mm round-bottom test tube", 0.008, 0.100, True, 0.012),
        Hole(0.00875, 0.035, 0.001),
        "a 16 mm tube in a 17.5 mm hole: the common rack, 0.75 mm clearance a side",
    ),
    "vial": Case(
        Peg("16 x 60 mm flat-bottom vial", 0.008, 0.060, False, 0.010),
        Hole(0.0083, 0.030, 0.0015),
        "a flat-bottomed vial in a 16.6 mm hole with a 1.5 mm rounded rim: 0.3 mm clearance a side",
    ),
}


@dataclass(frozen=True)
class Rack:
    """A block rack, its frame at the centre of its top face, z up; a fin on top for the holding hand."""

    half: tuple[float, float, float] = (0.03, 0.10, 0.02)
    mass_kg: float = 0.2
    #: the hole the tube goes in (the others are drawn), and the grid they are all on; 155 mm from the fin, so
    #: that the two wrists clear each other (90 mm put them into each other)
    hole_xy: tuple[float, float] = (0.0, -0.07)
    grid_x: tuple[float, ...] = (0.0,)
    grid_y: tuple[float, ...] = (-0.07, -0.045, -0.02, 0.005)
    #: the fin the left pads squeeze: half sizes (thin along x, the fingers' axis), and its centre
    fin_half: tuple[float, float, float] = (0.002, 0.02, 0.015)
    fin_y: float = 0.085


@dataclass(frozen=True)
class Layout:
    """Where the rack is held (its top-face centre in the right base) and the insertion's command path."""

    spacing_m: float = 0.60
    rack_at: tuple[float, float, float] = (0.52, 0.32, 0.28)
    #: the tube's tip over the hole at the start, and the force it is commanded into the hole's floor with
    #: (past the floor by this over the controller's own axial stiffness)
    hover_m: float = 0.02
    press_n: float = 5.0
    speed_m_s: float = 0.02
    #: a guarded touch: this slow from 5 mm over the hole's planned top to 3 mm into it
    touch_speed_m_s: float = 0.005
    touch_zone_m: tuple[float, float] = (0.005, -0.003)
    fps: float = 50.0
    settle_s: float = 0.5
    hold_s: float = 1.0
    #: the arms' Coulomb friction per joint, Nm (D6's 1 Nm: the pivot runs, 10-19)
    joint_friction_nm: float = cc.JOINT_FRICTION_NM


@dataclass(frozen=True)
class Error:
    """What the plan does not know: the rack's offset in the holding hand (rack axes, mm) and the tube's tilt
    in the inserting hand (about a horizontal axis through the pads, at ``tilt_dir_deg`` from x)."""

    dx_mm: float = 0.0
    dy_mm: float = 0.0
    tilt_deg: float = 0.0
    tilt_dir_deg: float = 0.0


# ------------------------------------------------------------------ the controllers ---


@dataclass(frozen=True)
class Controller:
    """An emulation of one panda_py torque controller (src/controllers/*.cpp), as a preset.

    ``joint``: ``JointPosition`` -- tau = K (q_d - q) - D dq. ``cartesian``: ``CartesianImpedance`` --
    tau = J^T (-K e - D J dq) + the nullspace term, K in the base's axes about the EE point, D = 2 zeta sqrt(K)
    (per unit mass, as panda_py does). ``ee`` is that point: the TCP, or the tube's tip (``set_EE`` there: the
    rotation compliance then centres on the tip, a remote centre of compliance).
    """

    name: str
    kind: str
    joint_stiffness: tuple[float, ...] = tuple(cc.JOINT_STIFFNESS)
    joint_damping: tuple[float, ...] = tuple(cc.JOINT_DAMPING)
    cartesian_stiffness: tuple[float, ...] = (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)
    damping_ratio: float = 1.0
    nullspace_stiffness: float = 0.5
    ee: str = "tcp"
    #: a Cartesian controller's stiffness on the way, until the tip is ``soften_above_m`` over the hole
    #: (``set_impedance``, filtered in as panda_py filters it): soft all the way, the joints' friction leaves the
    #: tip mm and degrees off before it reaches the hole (1 Nm against 10 Nm/rad is 6 degrees)
    approach_stiffness: tuple[float, ...] | None = None
    soften_above_m: float = 0.005
    #: the inserting hand looks for the hole (a spiral) when the tube stops on the rack's top
    search: bool = False
    note: str = ""


def _soft(scale: float) -> dict[str, tuple[float, ...]]:
    return {
        "joint_stiffness": tuple(cc.JOINT_STIFFNESS * scale),
        "joint_damping": tuple(cc.JOINT_DAMPING * math.sqrt(scale)),
    }


INSERT_STIFFNESS = (1000.0, 1000.0, 1000.0, 60.0, 60.0, 60.0)
APPROACH_STIFFNESS = (2000.0, 2000.0, 2000.0, 150.0, 150.0, 150.0)
CONTROLLERS = {
    "joint": Controller("joint", "joint", note="panda_py JointPosition defaults: what D1 runs today"),
    "joint_soft": Controller("joint_soft", "joint", **_soft(0.3), note="D6's held softening, x0.3"),
    "cart_default": Controller("cart_default", "cartesian", note="panda_py CartesianImpedance defaults, EE at TCP"),
    "cart_tcp": Controller(
        "cart_tcp", "cartesian", cartesian_stiffness=INSERT_STIFFNESS, note="soft across, firm along, EE at TCP"
    ),
    "cart_tip": Controller(
        "cart_tip", "cartesian", cartesian_stiffness=INSERT_STIFFNESS, ee="tip",
        note="soft across, firm along, EE at the tube's tip (remote centre of compliance)",
    ),  # fmt: skip
    "cart_tip_staged": Controller(
        "cart_tip_staged", "cartesian", cartesian_stiffness=INSERT_STIFFNESS, ee="tip",
        approach_stiffness=APPROACH_STIFFNESS, note="cart_tip, stiff until 5 mm over the hole",
    ),  # fmt: skip
    "cart_tip_search": Controller(
        "cart_tip_search", "cartesian", cartesian_stiffness=INSERT_STIFFNESS, ee="tip",
        approach_stiffness=APPROACH_STIFFNESS, search=True,
        note="cart_tip_staged, and a spiral search when the tube lands on the rack's top",
    ),  # fmt: skip
    "joint_search": Controller("joint_search", "joint", search=True, note="joint, and the same spiral search"),
    # for the holding hand
    "cart_hold": Controller(
        "cart_hold", "cartesian", cartesian_stiffness=(3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0),
        note="a stiff Cartesian hold: the rack not dragged along by the tube rubbing on it",
    ),  # fmt: skip
}
INSERTERS = [name for name in CONTROLLERS if name != "cart_hold"]
HOLDER = "joint"
#: where ``simulate(approach_s=...)`` brings each hand in from, off its start (m, the right base's axes)
APPROACH_FROM = {"right": (0.0, -0.10, 0.12), "left": (0.0, 0.12, 0.10)}


@dataclass(frozen=True)
class Search:
    """Looking for the hole, from what the robot measures of itself, never from where the rack really is: the
    tube's push up its axis as Franka estimates it from the joint torques (``O_F_ext_hat_K``, here the contact's
    own, noise-free, low-passed over ``filter_s``) -- not the tip's lag behind its command, which the joints'
    friction makes 4-8 N in free motion -- and the tip's position by the arm's kinematics.

    Once the tip, between ``near_m`` over the hole's planned top and halfway down it, is pushed by ``contact_n``,
    the descent stops and the hand presses ``press_n`` -- held there by a force loop on the command's height
    (``force_gain``, m/s per N): an offset under a soft, rubbing arm presses anything from 1 to 4 N, and the
    other arm gives under the press -- for ``dwell_s``: a tube on the rim's rounding slides in under it, one on
    the top face does not. Still out after the dwell, the hand turns out a spiral, ``pitch_m`` a turn at
    ``speed_m_s``, up to ``max_r_m``, still pressing, and goes on down once it is in (not stopping to turn when
    the tip drops a little: the holding arm drifts down under the moving press as much): ``entry_m`` down with the push under half the press, for ``settle_s`` -- the drop alone is also the
    holding arm slipping under the moving press (its joints stick and slip too), a bounce comes back up. The pitch is under the vial hole's
    capture width (its rim's rounding and clearance, both sides)."""

    contact_n: float = 3.0
    near_m: float = 0.005
    press_n: float = 4.0
    force_gain: float = 0.002
    dwell_s: float = 0.3
    pitch_m: float = 0.0012
    speed_m_s: float = 0.010
    max_r_m: float = 0.005
    entry_m: float = 0.0015
    settle_s: float = 0.1
    filter_s: float = 0.03


# ------------------------------------------------------------------ the model ---


def _quat(R: np.ndarray) -> np.ndarray:
    return np.asarray(sc._quat_wxyz(R))


def _pose(R: np.ndarray, p: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    return sc._axis_angle(np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis), angle)


@dataclass
class Geometry:
    """The plan's frames, in the right base: the rack, its hole, each hand's TCP, the tube in the right hand."""

    T_rack: np.ndarray
    T_hole: np.ndarray
    T_left_tcp: np.ndarray
    #: the tube's frame in the right TCP's (z down its axis, origin at its middle), and its tip's offset
    T_tcp_tube: np.ndarray
    tip_m: float
    openings: dict[str, float]


def geometry(case: Case, rack: Rack, layout: Layout, pads: dict[str, tuple[float, float]]) -> Geometry:
    """``pads``: per hand, (opening, pad drop) for what it holds (``cp.grip``)."""
    T_rack = _pose(np.eye(3), layout.rack_at)
    T_hole = T_rack @ _pose(np.eye(3), [*rack.hole_xy, 0.0])
    fin = T_rack @ _pose(np.eye(3), [0.0, rack.fin_y, rack.fin_half[2]])
    T_left_tcp = _pose(R_HAND["left"], fin[:3, 3] - [0.0, 0.0, pads["left"][1]])
    # the pads grip_from_top below the tube's top, the TCP pad_drop below the pads (both down the TCP's z)
    peg = case.peg
    centre = -pads["right"][1] - peg.grip_from_top_m + peg.length_m / 2.0
    return Geometry(
        T_rack=T_rack,
        T_hole=T_hole,
        T_left_tcp=T_left_tcp,
        T_tcp_tube=_pose(np.eye(3), [0.0, 0.0, centre]),
        tip_m=centre + peg.length_m / 2.0,
        openings={arm: pads[arm][0] for arm in pads},
    )


def right_tcp_for_tip(geo: Geometry, tip: np.ndarray) -> np.ndarray:
    """The right TCP pose that puts the (planned) tube's tip at ``tip``, the tube vertical."""
    return _pose(R_HAND["right"], np.asarray(tip) + [0.0, 0.0, geo.tip_m])


def grasp_poses(geo: Geometry, pad_drop_right: float, err: Error) -> dict[str, np.ndarray]:
    """Each object's pose in its hand's TCP frame, as it really is: the weld's relpose."""
    T_left_rack = np.linalg.inv(geo.T_left_tcp) @ _pose(np.eye(3), [err.dx_mm * 1e-3, err.dy_mm * 1e-3, 0.0]) @ geo.T_rack
    T_tube = geo.T_tcp_tube
    if err.tilt_deg:
        d = math.radians(err.tilt_dir_deg)
        axis = np.array([-math.sin(d), math.cos(d), 0.0])  # tilts the tip toward (cos d, sin d) of the TCP's x/y
        pad = _pose(np.eye(3), [0.0, 0.0, -pad_drop_right])
        T_tube = pad @ _pose(_axis_rot(axis, math.radians(err.tilt_deg)), np.zeros(3)) @ np.linalg.inv(pad) @ T_tube
    return {"left": T_left_rack, "right": T_tube}


def _hole_geoms(body: Any, hole: Hole, at: tuple[float, float], solref: list[float], friction: float) -> None:
    """A blind hole in ``body`` (frame on the top face, z up): a ring of capsules for the rounded rim, a ring
    of boxes for its wall and for the top face around it, and a floor -- the only part of the rack that
    collides (geom group 5: not drawn, the block is)."""
    import mujoco

    n, r, rim = hole.segments, hole.radius_m, hole.rim_r_m
    t = math.tan(math.pi / n)
    common = {"contype": 2, "conaffinity": 2, "solref": solref, "friction": [friction, 0.005, 0.0001],
              "mass": 0.0, "group": 5}  # fmt: skip
    wall, face_out = 0.003, r + 0.022
    for k in range(n):
        phi = 2.0 * math.pi * k / n
        u = np.array([math.cos(phi), math.sin(phi), 0.0])
        tan = np.array([-math.sin(phi), math.cos(phi), 0.0])
        c = np.array([*at, 0.0])
        rc = r + rim
        a = c + u * rc + tan * rc * t
        b = c + u * rc - tan * rc * t
        a[2] = b[2] = -rim
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rim, 0, 0], fromto=[*a, *b], **common)
        quat = _quat(np.column_stack([u, tan, [0.0, 0.0, 1.0]]))
        h = (hole.depth_m - rim) / 2.0
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[wall, (r + 2 * wall) * t, h],
            pos=(c + u * (r + wall) + [0.0, 0.0, -rim - h]).tolist(), quat=quat.tolist(), **common,
        )  # fmt: skip
        mid, half_r = (rc + face_out) / 2.0, (face_out - rc) / 2.0
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_r, face_out * t, 0.002],
            pos=(c + u * mid + [0.0, 0.0, -0.002]).tolist(), quat=quat.tolist(), **common,
        )  # fmt: skip
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[r + 0.01, r + 0.01, 0.002],
        pos=[*at, -hole.depth_m - 0.002], **common,
    )  # fmt: skip


@dataclass
class Sim:
    """A compiled scene and what the controllers need from it."""

    model: Any
    data: Any
    case: Case
    rack: Rack
    layout: Layout
    geo: Geometry
    pad_drop: dict[str, float]
    qadr: dict[str, list[int]]
    dadr: dict[str, list[int]]
    site: dict[str, int]
    eq: dict[str, int]
    body: dict[str, int]
    gripper_act: dict[str, int]
    T_right_left: np.ndarray = field(default_factory=lambda: np.eye(4))
    #: what ``simulate`` is doing, for a renderer's ``frame_cb``: phase, push (N), the tip's depth (m)
    status: dict[str, Any] = field(default_factory=dict)


def build(case: Case, rack: Rack = Rack(), layout: Layout = Layout()) -> Sim:
    """C2's two arms, stepped: torque-driven (the position actuators removed), gravity-compensated, with
    ``layout``'s joint friction; the tube welded to the right hand, the rack to the left."""
    import mujoco

    T_RL = np.eye(4)
    T_RL[1, 3] = layout.spacing_m
    record = sc.scene_record(T_RL, {"source": "assumed", "note": "D7 tube insertion"}, sc.environment_obstacles())
    kinematic = sc.DualScene(record)
    pads = {
        "right": cp.grip(kinematic, 2.0 * case.peg.radius_m),
        "left": cp.grip(kinematic, 2.0 * rack.fin_half[0]),
    }
    geo = geometry(case, rack, layout, pads)
    spec = sc.build_spec(record)
    spec.option.timestep = TIMESTEP_S
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    for geom in spec.geoms:
        geom.contype, geom.conaffinity = 0, 0
    for body in spec.bodies:
        if body.name.startswith(("right/", "left/")):
            body.gravcomp = 1.0
    for joint in spec.joints:
        if "fr3_joint" in joint.name:
            joint.frictionloss = layout.joint_friction_nm
    for act in list(spec.actuators):
        if "fr3_actuator" in act.name:
            spec.delete(act)
    peg, solref = case.peg, [0.002, 1.0]
    # the planned tip, where a Cartesian controller's EE is set to put its compliance centre
    spec.body("right/" + TCP_BODY).add_site(name="right/tip_ee", pos=[0.0, 0.0, geo.tip_m], size=[0.002, 0, 0])

    tube = spec.worldbody.add_body(name="tube")
    tube.add_freejoint(name="tube_free")
    half = peg.length_m / 2.0
    kind = mujoco.mjtGeom.mjGEOM_CAPSULE if peg.round_bottom else mujoco.mjtGeom.mjGEOM_CYLINDER
    size = [peg.radius_m, half - peg.radius_m, 0] if peg.round_bottom else [peg.radius_m, half, 0]
    tube.add_geom(
        name="tube", type=kind, size=size, mass=peg.mass_kg, rgba=[0.80, 0.90, 1.0, 0.55], contype=2, conaffinity=2,
        solref=solref, friction=[peg.friction, 0.005, 0.0001], condim=4,
    )  # fmt: skip
    liquid = 0.45 * peg.length_m
    tube.add_geom(
        type=kind, size=[peg.radius_m * 0.86, liquid / 2.0 - (peg.radius_m * 0.86 if peg.round_bottom else 0), 0],
        pos=[0.0, 0.0, half - liquid / 2.0 - 0.0015], rgba=[0.86, 0.22, 0.25, 0.9], contype=0, conaffinity=0, mass=0.0,
    )  # fmt: skip
    tube.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[peg.radius_m * 1.12, 0.006, 0], pos=[0.0, 0.0, -half - 0.004],
        rgba=[0.2, 0.45, 0.85, 1.0], contype=0, conaffinity=0, mass=0.0,
    )  # fmt: skip

    body = spec.worldbody.add_body(name="rack")
    body.add_freejoint(name="rack_free")
    hx, hy, hz = rack.half
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], pos=[0.0, 0.0, -hz], mass=rack.mass_kg,
        rgba=[0.96, 0.96, 0.94, 0.5], contype=0, conaffinity=0,
    )  # fmt: skip
    for gx in rack.grid_x:
        for gy in rack.grid_y:
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[case.hole.radius_m, 0.0004, 0], pos=[gx, gy, 0.0002],
                rgba=[0.12, 0.12, 0.14, 1.0], contype=0, conaffinity=0, mass=0.0,
            )  # fmt: skip
    fx, fy, fz = rack.fin_half
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[fx, fy, fz], pos=[0.0, rack.fin_y, fz], rgba=[0.95, 0.55, 0.15, 1.0],
        contype=0, conaffinity=0, mass=0.0,
    )  # fmt: skip
    _hole_geoms(body, case.hole, rack.hole_xy, solref, peg.friction)
    for obj, arm in (("tube", "right"), ("rack", "left")):
        eq = spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD, name=f"grasp_{arm}", name1=f"{arm}/{TCP_BODY}", name2=obj,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
        )  # fmt: skip
        eq.solref = [0.002, 1.0]
        eq.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    model = spec.compile()
    data = mujoco.MjData(model)
    sim = Sim(
        model=model,
        data=data,
        case=case,
        rack=rack,
        layout=layout,
        geo=geo,
        pad_drop={arm: pads[arm][1] for arm in pads},
        qadr={a: [model.joint(f"{p}fr3_joint{i}").qposadr[0] for i in range(1, 8)] for a, p in sc.ARM_PREFIX.items()},
        dadr={a: [model.joint(f"{p}fr3_joint{i}").dofadr[0] for i in range(1, 8)] for a, p in sc.ARM_PREFIX.items()},
        site={
            "right_tcp": model.site("right/" + TCP_SITE).id,
            "left_tcp": model.site("left/" + TCP_SITE).id,
            "right_tip": model.site("right/tip_ee").id,
        },
        eq={arm: model.equality(f"grasp_{arm}").id for arm in ("right", "left")},
        body={"tube": model.body("tube").id, "rack": model.body("rack").id},
        gripper_act={arm: model.actuator(f"{p}corenetic_gripper").id for arm, p in sc.ARM_PREFIX.items()},
        T_right_left=T_RL,
    )
    return sim


# ------------------------------------------------------------------ kinematics on the model ---


def site_pose(sim: Sim, data: Any, site: int) -> np.ndarray:
    return _pose(data.site_xmat[site].reshape(3, 3), data.site_xpos[site])


def site_jacobian(sim: Sim, data: Any, site: int, arm: str) -> np.ndarray:
    import mujoco

    jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(sim.model, data, jp, jr, site)
    return np.vstack([jp, jr])[:, sim.dadr[arm]]


def _orientation_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_d, in world axes."""
    from tools.fr3 import fr3_kinematics as kin

    return kin._rotvec((R_d @ R.T)[None])[0]


def ik(sim: Sim, scratch: Any, arm: str, site: int, T_target: np.ndarray, q0: np.ndarray, iters: int = 60) -> np.ndarray:
    """Damped least squares on the model, from ``q0``, clamped to the joint limits."""
    import mujoco

    m = sim.model
    lo = np.array([m.jnt_range[m.joint(f"{sc.ARM_PREFIX[arm]}fr3_joint{i}").id][0] for i in range(1, 8)])
    hi = np.array([m.jnt_range[m.joint(f"{sc.ARM_PREFIX[arm]}fr3_joint{i}").id][1] for i in range(1, 8)])
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(iters):
        scratch.qpos[sim.qadr[arm]] = q
        mujoco.mj_kinematics(m, scratch)
        mujoco.mj_comPos(m, scratch)
        T = site_pose(sim, scratch, site)
        e = np.r_[T_target[:3, 3] - T[:3, 3], _orientation_error(T[:3, :3], T_target[:3, :3])]
        if np.linalg.norm(e[:3]) < 1e-7 and np.linalg.norm(e[3:]) < 1e-6:
            break
        J = site_jacobian(sim, scratch, site, arm)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q = np.clip(q + dq, lo + 1e-3, hi - 1e-3)
    return q


def equivalent_stiffness(sim: Sim, data: Any, ctrl: Controller, arm: str = "right") -> np.ndarray:
    """6x6 stiffness the controller holds the (planned) tube tip with, world axes about the tip: the joint
    controller's (J K^-1 J^T)^-1 there, the Cartesian one's K moved from its EE point to the tip."""
    site = sim.site["right_tip"] if arm == "right" else sim.site["left_tcp"]
    if ctrl.kind == "joint":
        J = site_jacobian(sim, data, site, arm)
        C = J @ (J.T / np.asarray(ctrl.joint_stiffness)[:, None])
        return np.linalg.inv(C)
    K = np.diag(ctrl.cartesian_stiffness)
    if ctrl.ee == "tip" or arm != "right":
        return K
    # compliance about the TCP moved to the tip: a twist at the TCP moves the tip by v + w x r
    r = data.site_xpos[sim.site["right_tip"]] - data.site_xpos[sim.site["right_tcp"]]
    A = np.eye(6)
    A[:3, 3:] = -cc._skew(r)
    C = A @ np.linalg.inv(K) @ A.T
    return np.linalg.inv(C)


def point_stiffness(K: np.ndarray) -> np.ndarray:
    """What a push on the tip meets, per axis (N/m): the hand free to turn under it, 1 / diag(K^-1). K's own
    diagonal holds the turning still -- the same at every point of the hand, so it reads a TCP-centred spring
    as stiff at the tip as at the TCP, and a joint controller as 5-15 kN/m where a push finds 0.7-1.1 (10-30)."""
    return 1.0 / np.diag(np.linalg.inv(K))[:3]


# ------------------------------------------------------------------ the controllers, emulated ---


class ArmController:
    """One arm's panda_py controller at 1 kHz: set-points from ``command`` filtered as panda_py filters them
    (``filter_coeff`` per tick), torque out into ``qfrc_applied``; gravity is the model's (Franka's own)."""

    def __init__(self, sim: Sim, arm: str, ctrl: Controller, alpha: float, q0: np.ndarray, T_ee0: np.ndarray):
        self.sim, self.arm, self.ctrl, self.alpha = sim, arm, ctrl, alpha
        self.site = sim.site[f"{arm}_tip"] if (arm == "right" and ctrl.ee == "tip") else sim.site[f"{arm}_tcp"]
        self.q_d = q0.copy()
        self.q_target = q0.copy()
        self.p_d, self.R_d = T_ee0[:3, 3].copy(), T_ee0[:3, :3].copy()
        self.p_target, self.R_target = self.p_d.copy(), self.R_d.copy()
        self.q_ns = q0.copy()
        self.set_stiffness(ctrl.approach_stiffness or ctrl.cartesian_stiffness)
        self.K, self.D = self.K_target, self.D_target
        self.Kq = np.asarray(ctrl.joint_stiffness)
        self.Dq = np.asarray(ctrl.joint_damping)
        self.saturated = 0

    def set_stiffness(self, K: tuple[float, ...]) -> None:
        """panda_py's ``set_impedance``: damping recomputed, both filtered in tick by tick."""
        self.K_target = np.diag(np.asarray(K, dtype=np.float64))
        self.D_target = np.diag(self.ctrl.damping_ratio * 2.0 * np.sqrt(np.asarray(K, dtype=np.float64)))

    def command(self, q: np.ndarray, T_ee: np.ndarray) -> None:
        self.q_target = np.asarray(q, dtype=np.float64)
        self.q_ns = self.q_target
        self.p_target, self.R_target = T_ee[:3, 3].copy(), T_ee[:3, :3].copy()

    def torque(self) -> np.ndarray:
        d, a = self.sim.data, self.alpha
        q, dq = d.qpos[self.sim.qadr[self.arm]], d.qvel[self.sim.dadr[self.arm]]
        if self.ctrl.kind == "joint":
            self.q_d = self.q_d + a * (self.q_target - self.q_d)
            tau = self.Kq * (self.q_d - q) - self.Dq * dq
        else:
            self.p_d = self.p_d + a * (self.p_target - self.p_d)
            self.K = self.K + a * (self.K_target - self.K)
            self.D = self.D + a * (self.D_target - self.D)
            w = _orientation_error(self.R_d, self.R_target)
            self.R_d = _axis_rot(w, a * np.linalg.norm(w)) @ self.R_d if np.linalg.norm(w) > 1e-12 else self.R_d
            T = site_pose(self.sim, d, self.site)
            J = site_jacobian(self.sim, d, self.site, self.arm)
            e = np.r_[T[:3, 3] - self.p_d, -_orientation_error(T[:3, :3], self.R_d)]
            tau = J.T @ (-self.K @ e - self.D @ (J @ dq))
            ns = self.ctrl.nullspace_stiffness
            N = np.eye(7) - J.T @ np.linalg.pinv(J.T)
            tau = tau + N @ (ns * (self.q_ns - q) - 2.0 * math.sqrt(ns) * dq)
        clipped = np.clip(tau, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        self.saturated += int(np.any(clipped != tau))
        return clipped


# ------------------------------------------------------------------ one run ---


@dataclass(frozen=True)
class Run:
    case: str = "tube"
    controller: str = "cart_tip"
    holder: str = HOLDER
    error: Error = Error()
    search: Search = Search()
    joint_friction_nm: float = cc.JOINT_FRICTION_NM


def _object_pose(sim: Sim, name: str) -> np.ndarray:
    b = sim.body[name]
    return _pose(sim.data.xmat[b].reshape(3, 3), sim.data.xpos[b])


def _contact_force(sim: Sim) -> np.ndarray:
    """The rack's push on the tube, world axes (N)."""
    import mujoco

    m, d = sim.model, sim.data
    total, f6 = np.zeros(3), np.zeros(6)
    tube = sim.body["tube"]
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if tube not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        f = c.frame.reshape(3, 3).T @ f6[:3]
        total += f if b2 == tube else -f
    return total


def place(sim: Sim, q: dict[str, np.ndarray], err: Error) -> None:
    """Both arms at ``q``, the objects where the (erroneous) grasps hold them, at rest."""
    import mujoco

    m, d = sim.model, sim.data
    mujoco.mj_resetData(m, d)
    for arm in q:
        d.qpos[sim.qadr[arm]] = q[arm]
    rel = grasp_poses(sim.geo, sim.pad_drop["right"], err)
    mujoco.mj_kinematics(m, d)
    for obj, arm in (("tube", "right"), ("rack", "left")):
        m.eq_data[sim.eq[arm], 0:3] = 0.0  # the anchor, in the object's frame
        m.eq_data[sim.eq[arm], 3:6] = rel[arm][:3, 3]
        m.eq_data[sim.eq[arm], 6:10] = _quat(rel[arm][:3, :3])
        m.eq_data[sim.eq[arm], 10] = 1.0
        b = m.body(sim.body[obj])
        T_tcp = _pose(d.xmat[m.body(f"{arm}/{TCP_BODY}").id].reshape(3, 3), d.xpos[m.body(f"{arm}/{TCP_BODY}").id])
        T = T_tcp @ rel[arm]
        adr = m.jnt_qposadr[b.jntadr[0]]
        d.qpos[adr : adr + 3] = T[:3, 3]
        d.qpos[adr + 3 : adr + 7] = _quat(T[:3, :3])
        d.ctrl[sim.gripper_act[arm]] = sim.geo.openings[arm]
    # the fingers start closed on what they hold
    for j in range(m.njnt):
        name = m.joint(j).name
        if "gripper" in name:
            side = "right" if name.startswith("right/") else "left"
            # the drive and every mimic joint at what the actuator settles to: 1443.39 / 1000 per unit ctrl
            d.qpos[m.jnt_qposadr[j]] = 1.44339 * sim.geo.openings[side]
    mujoco.mj_forward(m, d)


def start_joints(sim: Sim) -> dict[str, np.ndarray]:
    """IK for the plan's start: the rack where it is meant to be, the tube's tip ``hover_m`` over the hole."""
    import mujoco

    from tools.fr3 import fr3_kinematics as kin

    scratch = mujoco.MjData(sim.model)
    tip0 = sim.geo.T_hole[:3, 3] + [0.0, 0.0, sim.layout.hover_m]
    seeds = {}
    for arm in ("right", "left"):
        seeds[arm] = kin.Q_READY.copy()
        # the fingers along x; a hand turned half round about its z (R_HAND) the other way round of j7
        flipped = R_HAND[arm][0, 1] < 0
        seeds[arm][6] += math.pi / 2.0 if flipped else -math.pi / 2.0
    return {
        "right": ik(sim, scratch, "right", sim.site["right_tcp"], right_tcp_for_tip(sim.geo, tip0), seeds["right"]),
        "left": ik(sim, scratch, "left", sim.site["left_tcp"], sim.geo.T_left_tcp, seeds["left"]),
    }


def simulate(
    sim: Sim,
    run: Run,
    *,
    q_start: dict[str, np.ndarray] | None = None,
    frame_cb: Any = None,
    frame_every: int = 0,
    release: bool = False,
    approach_s: float = 0.0,
) -> dict[str, Any]:
    """One insertion: settle, descend (searching if the controller does), hold; the measurements.

    ``frame_cb(sim, t)`` every ``frame_every`` ticks (a renderer); ``release`` lets go of the tube after the
    hold, waits ``RELEASE_S`` and lifts the hand 8 cm, ``approach_s`` first brings both hands in from
    ``APPROACH_FROM`` (both for the video); without it the run lets go and waits only. Seated is judged after the
    letting go -- a tube a few mm short (the arm's friction) drops the rest, a wedged one stays -- and times in the
    result are from the start of the descent.
    """
    import mujoco

    m, d, lay, geo = sim.model, sim.data, sim.layout, sim.geo
    hole = sim.case.hole
    ctrl = CONTROLLERS[run.controller]
    q0 = q_start or start_joints(sim)
    scratch = mujoco.MjData(m)
    starts = {
        "right": right_tcp_for_tip(geo, geo.T_hole[:3, 3] + [0.0, 0.0, lay.hover_m]),
        "left": geo.T_left_tcp.copy(),
    }
    q_far = dict(q0)
    if approach_s > 0.0:
        for arm in q_far:
            far = starts[arm].copy()
            far[:3, 3] += APPROACH_FROM[arm]
            q_far[arm] = ik(sim, scratch, arm, sim.site[f"{arm}_tcp"], far, q0[arm])
    place(sim, q_far, run.error)
    alpha = 1.0 - math.exp(-TIMESTEP_S / (cp_filter_tau(lay.fps)))
    ee_site = sim.site["right_tip"] if ctrl.ee == "tip" else sim.site["right_tcp"]
    T_ee0 = site_pose(sim, d, ee_site)
    arms = {
        "right": ArmController(sim, "right", ctrl, alpha, q_far["right"], T_ee0),
        "left": ArmController(sim, "left", CONTROLLERS[run.holder], alpha, q_far["left"], site_pose(sim, d, sim.site["left_tcp"])),
    }
    t0 = approach_s + lay.settle_s  # the descent's start
    K_eq = equivalent_stiffness(sim, d, ctrl)
    k_point = point_stiffness(K_eq)
    kz = float(k_point[2])
    top = geo.T_hole[:3, 3].copy()
    floor_z = top[2] - hole.depth_m
    target_z = floor_z - lay.press_n / kz
    period = int(round(1.0 / (lay.fps * TIMESTEP_S)))
    S = run.search
    phase, t_phase = ("approach" if approach_s > 0.0 else "settle"), 0.0
    tip_cmd = top + [0.0, 0.0, lay.hover_m]
    spiral = {"theta": 0.0, "centre": top[:2].copy(), "z": 0.0, "z0": 0.0, "z_touch": 0.0}
    log: list[tuple[float, ...]] = []
    events: list[tuple[float, str]] = []
    peak_force, peak_raw, force_f, first_contact = 0.0, 0.0, 0.0, None
    q_cmd = q_far["right"].copy()
    q_left = q_far["left"].copy()
    softened = ctrl.approach_stiffness is None
    t_end_hold = t_release = held_depth = None
    F_held = np.zeros(3)
    push_n = 0.0
    calm = 0
    lift = 0.0
    steps_total = int((t0 + (lay.hover_m + hole.depth_m + lay.press_n / kz) / lay.speed_m_s + 18.0 + lay.hold_s) / TIMESTEP_S)
    for step in range(steps_total):
        t = step * TIMESTEP_S
        if step % period == 0:
            T_now = site_pose(sim, d, sim.site["right_tcp"])
            tip_now = T_now[:3, 3] + T_now[:3, :3] @ [0.0, 0.0, geo.tip_m]
            if phase == "approach":
                if t >= approach_s:
                    phase = "settle"
                else:
                    s_ = float(cp._minjerk(t / approach_s))
                    T_left = starts["left"].copy()
                    T_left[:3, 3] += (1.0 - s_) * np.asarray(APPROACH_FROM["left"])
                    q_left = ik(sim, scratch, "left", sim.site["left_tcp"], T_left, q_left, iters=20)
                    arms["left"].command(q_left, T_left)
            if phase == "settle" and t >= t0:
                phase, t_phase = "descend", t
                events.append((t, phase))
            if phase in ("descend", "insert"):
                tip_cmd = tip_cmd.copy()
                slow = top[2] + lay.touch_zone_m[1] < tip_cmd[2] < top[2] + lay.touch_zone_m[0]
                tip_cmd[2] = max(target_z, tip_cmd[2] - (lay.touch_speed_m_s if slow else lay.speed_m_s) / lay.fps)
                touching = push_n > S.contact_n and (
                    top[2] - hole.depth_m / 2.0 < tip_now[2] < top[2] + S.near_m
                )
                if ctrl.search and phase == "descend" and touching:
                    phase, t_phase = "dwell", t
                    spiral.update(
                        theta=0.0, centre=tip_now[:2].copy(), z=tip_now[2] - S.press_n / kz, z0=tip_now[2],
                        z_touch=tip_now[2],
                    )  # fmt: skip
                    tip_cmd = np.array([*spiral["centre"], spiral["z"]])
                    t_end_hold = None
                    events.append((t, phase))
                if tip_cmd[2] <= target_z and t_end_hold is None:
                    t_end_hold = t + lay.hold_s
                    events.append((t, "at_target"))
            elif phase in ("dwell", "search"):
                # the press: the command's height follows the push's error, kept within a cm of where it began
                spiral["z"] -= S.force_gain * (S.press_n - push_n) / lay.fps
                spiral["z"] = float(np.clip(spiral["z"], spiral["z_touch"] - 0.01, spiral["z_touch"] + 0.002))
                tip_cmd = np.array([*tip_cmd[:2], spiral["z"]])
                dropped = spiral["z0"] - tip_now[2]
                calm = calm + 1 if dropped > S.entry_m and push_n < 0.5 * S.press_n else 0
                if calm * period * TIMESTEP_S >= S.settle_s:
                    phase = "insert"
                    tip_cmd = np.array([*tip_cmd[:2], tip_now[2]])
                    events.append((t, phase))
                elif phase == "dwell" and t - t_phase >= S.dwell_s:
                    # where the press has settled (the other arm gives under it too): the spiral's datum
                    phase, spiral["z0"] = "search", tip_now[2]
                    events.append((t, phase))
                elif phase == "search":
                    r = S.pitch_m * spiral["theta"] / (2.0 * math.pi)
                    spiral["theta"] += S.speed_m_s / lay.fps / max(r, 0.5 * S.pitch_m)
                    tip_cmd = np.array([
                        spiral["centre"][0] + r * math.cos(spiral["theta"]),
                        spiral["centre"][1] + r * math.sin(spiral["theta"]),
                        spiral["z"],
                    ])  # fmt: skip
                    if r > S.max_r_m:
                        phase = "gave_up"
                        events.append((t, phase))
            elif phase == "release":
                if t >= t_release + RELEASE_S:
                    if not release:
                        break
                    lift = min(0.08, lift + 0.04 / lay.fps)
                if t >= t_release + RELEASE_S + 2.5:
                    break
            if t_end_hold is not None and t >= t_end_hold and phase != "release":
                phase, t_release = "release", t
                held_depth, F_held = -float(log[-1][3]), np.asarray(log[-1][4:7])
                d.eq_active[sim.eq["right"]] = 0
                d.ctrl[sim.gripper_act["right"]] = 1.0
                events.append((t, phase))
            if phase in ("gave_up",) and not release:
                break
            if not softened and tip_now[2] < top[2] + ctrl.soften_above_m:
                arms["right"].set_stiffness(ctrl.cartesian_stiffness)
                softened = True
                events.append((t, "softened"))
            T_cmd = right_tcp_for_tip(geo, tip_cmd)
            T_cmd[2, 3] += lift
            if phase == "approach":
                T_cmd[:3, 3] += (1.0 - float(cp._minjerk(t / approach_s))) * np.asarray(APPROACH_FROM["right"])
            q_cmd = ik(sim, scratch, "right", sim.site["right_tcp"], T_cmd, q_cmd, iters=20)
            T_ee_cmd = T_cmd.copy()
            if ctrl.ee == "tip":  # its F_T_EE is the planned tip: it does not know the tilt
                T_ee_cmd[:3, 3] = T_cmd[:3, 3] + T_cmd[:3, :3] @ [0.0, 0.0, geo.tip_m]
            arms["right"].command(q_cmd, T_ee_cmd)
        for arm, c in arms.items():
            d.qfrc_applied[sim.dadr[arm]] = c.torque()
        mujoco.mj_step(m, d)
        # MuJoCo resets the state itself when it blows up, so the state stays finite: its warning is the sign
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            events.append((t, "diverged"))
            break
        if step % 5 == 0:
            F = _contact_force(sim)
            fn = float(np.linalg.norm(F))
            if fn > 0.05 and first_contact is None:
                first_contact = t
            # the peak as Franka's estimate would show it (low-passed): single-tick contact impulses are not
            # what trips a reflex
            force_f += (1.0 - math.exp(-5 * TIMESTEP_S / S.filter_s)) * (fn - force_f)
            if phase != "release":
                peak_force, peak_raw = max(peak_force, force_f), max(peak_raw, fn)
            T_rack = _object_pose(sim, "rack")
            T_tube = _object_pose(sim, "tube")
            tip = T_tube[:3, 3] + T_tube[:3, :3] @ [0.0, 0.0, sim.case.peg.length_m / 2.0]
            hole_frame = T_rack @ _pose(np.eye(3), [*sim.rack.hole_xy, 0.0])
            local = np.linalg.inv(hole_frame) @ np.r_[tip, 1.0]
            axis = T_tube[:3, 2]
            tilt = math.degrees(math.acos(min(1.0, abs(float(axis @ hole_frame[:3, 2])))))
            push_n += (1.0 - math.exp(-5 * TIMESTEP_S / S.filter_s)) * (float(F[2]) - push_n)
            log.append((t, *local[:3], *F, tilt, *(T_rack[:3, 3] - geo.T_rack[:3, 3])))
        if frame_cb is not None and frame_every and step % frame_every == 0:
            # the depth only over the hole: a tube let go on the rim falls off the rack
            over = bool(log) and math.hypot(log[-1][1], log[-1][2]) < hole.radius_m
            sim.status.update(phase=phase, force_n=force_f, depth_m=-float(log[-1][3]) if over else 0.0)
            frame_cb(sim, t)
        if not np.all(np.isfinite(d.qpos)):
            events.append((t, "diverged"))
            break
    L = np.asarray(log)
    settled = int(np.searchsorted(L[:, 0], t0))
    deep = np.flatnonzero(-L[:, 3] >= hole.depth_m - SEATED_TOLERANCE_M)
    seat_time = None if not len(deep) else round(float(L[deep[0], 0]) - t0, 3)
    depth = -L[:, 3]
    seated = bool(depth[-1] >= hole.depth_m - SEATED_TOLERANCE_M and np.hypot(L[-1, 1], L[-1, 2]) < hole.radius_m)
    # jammed is in the bore and stuck there; a tip resting on the rim's rounding (it sinks up to rim_r into it)
    # never went in: missed. Read while held and over the hole: a tube let go on the rim falls off the rack, and
    # its fall is no depth (10-30: such runs were called jammed 200-380 mm deep)
    held = L[:, 0] < (t_release if t_release is not None else math.inf)
    over = np.hypot(L[:, 1], L[:, 2]) < hole.radius_m
    in_bore = bool(np.any(held & over & (depth > hole.rim_r_m + IN_BORE_M)))
    outcome = "seated" if seated else ("jammed" if in_bore else ("gave_up" if phase == "gave_up" else "missed"))
    return {
        "run": {
            "case": run.case, "controller": run.controller, "holder": run.holder, "error": asdict(run.error),
            "joint_friction_nm": lay.joint_friction_nm,
        },  # fmt: skip
        "outcome": outcome,
        "seated": seated,
        "peak_force_n": round(peak_force, 2),
        "peak_force_raw_n": round(peak_raw, 2),
        "over_reflex": bool(peak_force > REFLEX_FORCE_N),
        # what the hand still pushed sideways when it let go: the rack and the arms' residual strain
        "final_lateral_force_n": round(float(np.hypot(F_held[0], F_held[1])), 2),
        # None: the tube ended off the hole (let go on the rim, it fell)
        "final_depth_mm": round(float(depth[-1]) * 1e3, 2) if over[-1] else None,
        "max_depth_mm": round(float(depth[held].max(initial=0.0)) * 1e3, 2),
        # the contact blew up (a wedged tube let go of springs out of the soft contact): the log stops there
        "diverged": any(e == "diverged" for _, e in events),
        "held_depth_mm": None if held_depth is None else round(held_depth * 1e3, 2),
        "time_to_seat_s": seat_time,
        "first_contact_s": None if first_contact is None else round(first_contact - t0, 3),
        "max_tilt_deg": round(float(L[:, 7].max()), 2),
        # from where the holding arm had it at the end of the settle: its sag under the rack is the plan's error too
        "rack_pushed_mm": round(float(np.linalg.norm(L[settled:, 8:11] - L[settled, 8:11], axis=1).max()) * 1e3, 2),
        "rack_sag_mm": round(float(-L[settled, 10]) * 1e3, 2),
        "torque_saturated_ticks": {a: c.saturated for a, c in arms.items()},
        "events": [(round(t - t0, 3), e) for t, e in events],
        "stiffness_at_tip": {
            # a push on the tip, the hand free to turn (point_stiffness)
            "lateral_n_m": [round(float(k_point[0])), round(float(k_point[1]))],
            "axial_n_m": round(kz),
            # turning about the tip, the tip held (in the hole)
            "tilt_nm_rad": [round(float(K_eq[3, 3]), 1), round(float(K_eq[4, 4]), 1)],
        },
        "log": L,
    }


def clearance(sim: Sim, q: dict[str, np.ndarray]) -> dict[str, Any]:
    """Between the two arms at ``q``: MuJoCo's hull distance (the real shapes) and the planning capsules' gap
    (C3's model, which encloses them: what ``d_safe`` is checked against)."""
    rec = sc.scene_record(sim.T_right_left, {"source": "assumed"}, sc.environment_obstacles())
    scene = sc.DualScene(rec)
    scene.set(q["right"], q["left"], (sim.geo.openings["right"], sim.geo.openings["left"]))
    d, pair, _ = scene.arm_distance(0.3)
    gap = col.clearance(col.load(), q["right"][None], q["left"][None], sim.T_right_left)
    return {
        "mesh_gap_mm": round(d * 1e3, 1),
        "mesh_pair": list(pair),
        "capsule_gap_mm": round(float(gap["arm_arm_m"][0]) * 1e3, 1),
        "capsule_pair": list(gap["arm_arm_pair"][0]),
    }


def cp_filter_tau(fps: float) -> float:
    """The set-point filter's time constant, as D1 sets it (a third of a command period)."""
    from tools.fr3 import dual_fr3_replay_runtime as rt

    return rt.REAL_FILTER_TAU_PERIODS / fps


# ------------------------------------------------------------------ sweeps ---

_SIMS: dict[tuple[str, float], Sim] = {}
_STARTS: dict[tuple[str, float], dict[str, np.ndarray]] = {}


def _worker(run: Run) -> dict[str, Any]:
    key = (run.case, run.joint_friction_nm)
    if key not in _SIMS:
        _SIMS[key] = build(CASES[run.case], layout=Layout(joint_friction_nm=run.joint_friction_nm))
        _STARTS[key] = start_joints(_SIMS[key])
    out = simulate(_SIMS[key], run, q_start=_STARTS[key])
    out.pop("log")
    return out


def sweep_runs(
    case: str,
    controllers: list[str],
    offsets_mm: list[float],
    directions: int,
    tilts_deg: list[float],
    holder: str = HOLDER,
    joint_friction_nm: float = cc.JOINT_FRICTION_NM,
) -> list[Run]:
    """Each controller against the rack offset ``offsets_mm`` in ``directions`` directions, with and without a
    tilt of the tube (its tip toward the next direction round)."""
    runs = []
    for name in controllers:
        for e in offsets_mm:
            dirs = [0.0] if e == 0 else [360.0 * k / directions for k in range(directions)]
            for a in dirs:
                for tilt in tilts_deg:
                    err = Error(e * math.cos(math.radians(a)), e * math.sin(math.radians(a)), tilt, a + 90.0)
                    runs.append(Run(case, name, holder, err, joint_friction_nm=joint_friction_nm))
    return runs


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per controller, per offset, per tilt: how many seated, the forces."""
    rows: dict[tuple, list] = {}
    for r in results:
        e = r["run"]["error"]
        key = (r["run"]["case"], r["run"]["controller"], round(math.hypot(e["dx_mm"], e["dy_mm"]), 2), e["tilt_deg"])
        rows.setdefault(key, []).append(r)
    out = []
    for (case, name, off, tilt), rs in sorted(rows.items()):
        out.append({
            "case": case,
            "controller": name,
            "offset_mm": off,
            "tilt_deg": tilt,
            "runs": len(rs),
            "seated": sum(r["seated"] for r in rs),
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in ("seated", "jammed", "missed", "gave_up")},
            "diverged": sum(r["diverged"] for r in rs),
            "peak_force_n_max": max(r["peak_force_n"] for r in rs),
            "peak_force_n_median": float(np.median([r["peak_force_n"] for r in rs])),
            "over_reflex": sum(r["over_reflex"] for r in rs),
            "final_lateral_force_n_max": max(r["final_lateral_force_n"] for r in rs),
            "time_to_seat_s_median": (
                float(np.median([r["time_to_seat_s"] for r in rs if r["time_to_seat_s"] is not None]))
                if any(r["time_to_seat_s"] is not None for r in rs) else None
            ),
        })  # fmt: skip
    return out


# ------------------------------------------------------------------ the video ---

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
PHASES_ZH = {
    "approach": "两手送到位", "settle": "稳定", "descend": "下插", "dwell": "触到，按压", "search": "螺旋找孔",
    "insert": "入孔，继续下插", "release": "松手", "gave_up": "找孔失败",
}  # fmt: skip
TITLES_ZH = {
    "joint": "关节阻抗（现在 D1 用的）",
    "joint_soft": "关节阻抗 ×0.3",
    "cart_default": "笛卡尔阻抗（panda_py 默认）",
    "cart_tcp": "笛卡尔阻抗，柔顺中心在 TCP",
    "cart_tip": "笛卡尔阻抗，柔顺中心在管尖",
    "cart_tip_staged": "笛卡尔阻抗：先硬后软，柔顺中心在管尖",
    "cart_tip_search": "笛卡尔阻抗 + 力控螺旋找孔",
    "joint_search": "关节阻抗 + 螺旋找孔",
}


def render_panel(
    case: str, run: Run, video: Path, *, caption: str, approach_s: float = 2.5, fps: float = 30.0,
    size: tuple[int, int] = (960, 720),
) -> dict[str, Any]:
    """One run, both hands brought in, the insertion, letting go, the hand lifted; a close camera on the rack
    with the phase, the push and the depth written on it, and the outcome at the end."""
    import io

    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    sim = build(CASES[case], layout=Layout(joint_friction_nm=run.joint_friction_nm))
    q0 = start_joints(sim)
    width, height = size
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    options = cp._options()
    hole = sim.geo.T_hole[:3, 3]
    big, small = ImageFont.truetype(str(FONT), 30, index=2), ImageFont.truetype(str(FONT), 24, index=2)
    frames: list[tuple[bytes, float]] = []

    def frame(sim: Sim, t: float) -> None:
        s_ = float(cp._minjerk(t / approach_s)) if approach_s else 1.0
        cam = cp._camera(
            hole + [0.0, 0.0, 0.14 - 0.10 * s_], 1.35 - 0.85 * s_, -24.0 + 10.0 * s_,
            200.0 + 8.0 * max(0.0, t - approach_s) / 10.0,
        )  # fmt: skip
        renderer.update_scene(sim.data, camera=cam, scene_option=options)
        image = Image.fromarray(np.asarray(renderer.render(), dtype=np.uint8))
        draw = ImageDraw.Draw(image)
        st = sim.status
        draw.rectangle([0, 0, width, 132], fill=(0, 0, 0))
        draw.text((18, 10), TITLES_ZH.get(run.controller, run.controller), font=big, fill=(255, 255, 255))
        draw.text((18, 54), f"{PHASES_ZH.get(st.get('phase', ''), '')}", font=small, fill=(255, 210, 90))
        draw.text(
            (18, 90), f"接触力 {st.get('force_n', 0.0):4.1f} N    插入深度 {max(0.0, st.get('depth_m', 0.0)) * 1e3:4.1f} mm",
            font=small, fill=(200, 230, 255),
        )  # fmt: skip
        draw.rectangle([0, height - 44, width, height], fill=(0, 0, 0))
        draw.text((18, height - 38), caption, font=small, fill=(220, 220, 220))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        frames.append((buf.getvalue(), t))

    result = simulate(
        sim, run, q_start=q0, frame_cb=frame, frame_every=int(round(1.0 / (fps * TIMESTEP_S))), release=True,
        approach_s=approach_s,
    )  # fmt: skip
    renderer.close()
    t_release = next((t for t, e in result["events"] if e == "release"), None)
    stamp = (
        ("✓ 插到底", (90, 220, 120)) if result["seated"]
        else (f"× 卡在 {result['max_depth_mm']:.0f} mm", (240, 90, 80)) if result["outcome"] == "jammed"
        else ("× 没找到孔" if result["outcome"] == "gave_up" else "× 没进孔", (240, 90, 80))
    )  # fmt: skip
    video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-r", f"{fps:g}",
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(video)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    t0 = approach_s + sim.layout.settle_s
    for data, t in frames:
        if t_release is not None and t - t0 >= t_release + RELEASE_S:
            image = Image.open(io.BytesIO(data))
            ImageDraw.Draw(image).text((width - 300, 54), stamp[0], font=big, fill=stamp[1])
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=92)
            data = buf.getvalue()
        ffmpeg.stdin.write(data)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    result.pop("log")
    return result


def render_pair(case: str, runs: tuple[Run, Run], video: Path, caption: str) -> list[dict[str, Any]]:
    """Two runs side by side, the shorter held on its last frame."""
    parts = [video.with_name(f"{video.stem}_{k}.mp4") for k in range(2)]
    results = [render_panel(case, run, part, caption=caption) for run, part in zip(runs, parts, strict=True)]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(parts[0]), "-i", str(parts[1]),
         "-filter_complex", "[0]tpad=stop_mode=clone:stop_duration=6[a];[1]tpad=stop_mode=clone:stop_duration=6[b];"
         f"[a][b]hstack=2,trim=duration={max(_duration(p) for p in parts) + 1.0:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(video)],
        check=True, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    for p in parts:
        p.unlink()
    return results


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    return float(out.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A tube into a rack the other hand holds, simulated (D7)")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "fr3" / "dual_fr3_tube_insert")
    parser.add_argument("--case", choices=sorted(CASES), default="tube")
    parser.add_argument("--controllers", default=",".join(INSERTERS))
    parser.add_argument("--holder", choices=["joint", "cart_hold"], default=HOLDER)
    parser.add_argument("--friction", type=float, default=Layout.joint_friction_nm, help="per joint, Nm")
    parser.add_argument("--offsets", default="0,1,2,3,4,6", help="rack offset in the hand, mm")
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--tilts", default="0,2", help="tube tilt in the hand, deg")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--video", help="two controllers, A,B: render them side by side instead of the sweep")
    parser.add_argument("--error", default="4.5,0,0,0", help="for --video: dx_mm,dy_mm,tilt_deg,tilt_dir_deg")
    parser.add_argument("--caption", default="", help="for --video: the line under both panels")
    args = parser.parse_args(argv)
    if args.video:
        err = Error(*[float(v) for v in args.error.split(",")])
        runs = tuple(Run(args.case, n, args.holder, err, joint_friction_nm=args.friction) for n in args.video.split(","))
        caption = args.caption or f"托架在手中偏了 {math.hypot(err.dx_mm, err.dy_mm):g} mm，试管倾斜 {err.tilt_deg:g}°（规划不知道）"
        path = args.out / f"tube_insert_{args.case}_{'_vs_'.join(args.video.split(','))}.mp4"
        results = render_pair(args.case, runs, path, caption)
        for r in results:
            print(r["run"]["controller"], r["outcome"], r["peak_force_n"], r["max_depth_mm"])
        print(path)
        return 0
    runs = sweep_runs(
        args.case, args.controllers.split(","), [float(v) for v in args.offsets.split(",")], args.directions,
        [float(v) for v in args.tilts.split(",")], args.holder, args.friction,
    )  # fmt: skip
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_worker, runs, chunksize=1))
    table = summarize(results)
    sim = build(CASES[args.case])
    gaps = clearance(sim, start_joints(sim))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "case": asdict(CASES[args.case]),
        "rack": asdict(Rack()),
        "layout": asdict(Layout(joint_friction_nm=args.friction)),
        "search": asdict(Search()),
        "controllers": {n: asdict(CONTROLLERS[n]) for n in args.controllers.split(",")},
        "holder": asdict(CONTROLLERS[args.holder]),
        "arm_arm_at_start": gaps,
        "summary": table,
        "runs": results,
    }
    path = args.out / f"tube_insert_{args.case}_{args.holder}_f{args.friction:g}.json"
    path.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    for row in table:
        print(
            f"{row['controller']:>16} off {row['offset_mm']:>4} tilt {row['tilt_deg']:>3}: "
            f"{row['seated']}/{row['runs']} seated {row['outcomes']}  F peak max {row['peak_force_n_max']:.1f} N "
            f"median {row['peak_force_n_median']:.1f}  >20N {row['over_reflex']}  diverged {row['diverged']}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
