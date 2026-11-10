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
The compliant handle for a rigid two-hand carry (roadmap D6 step 2): D6's grasp spring, as a part to make.

D6 found no joint stiffness that lets two FR3s hold one 1 kg tray. The internal moment at each hand wants the
arms soft; the tray's sag and the joints' friction want them stiff. What fits is a spring between hand and
object, mostly in rotation. This one goes on the object's side, not the gripper's:

* The gripper closes on a rigid bar with flat faces 35 mm apart. That is the width and the face of the pivot
  insert v2 it already holds (the 10-14 pivot runs), so the fingers, their pads and F8's slip test stay as
  they are.
* The bar sits on two rubber-metal buffers: M6 male-male, off the shelf, and swapped by hardness. They sit on
  a cradle bolted to the tray.
* A shoulder screw in each end of the bar passes through a hole in the cradle's cheek with ``travel_mm`` of
  clearance all round. The handle gives that much and is then rigid; the runtime's internal-force guard
  then stops the arms. If a buffer tears, the tray hangs on the screws.

``handle_stiffness`` models each buffer as a short Timoshenko beam: Gent's modulus from the Shore A
hardness, and a shape factor for compression. It adds the two at the bar centre. ``tcp_compliance`` moves
the result to the TCP, in the TCP's own axes, which is the form ``dual_fr3_closed_chain.budget`` takes as a
design. Rubber off a shop shelf is +-30 % of any formula, so ``choose`` keeps a design only if it fits with
its stiffness at 0.7x and at 1.3x. The made handle is measured before use: hang a weight and read a dial
(the bench check on the drawing).

``--cad DIR`` writes the parts (STEP, STL) and the drawings through FreeCAD
(``dual_fr3_compliant_handle_cad.py``, which runs under ``freecad freecadcmd``).
"""

# ruff: noqa: N802, N803, N806  (matrix names: K, C, R follow the maths)
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import dual_fr3_closed_chain as cc  # noqa: E402

SCHEMA = "dual_fr3_compliant_handle/v1"
#: the TCP's axes in the bar's (x along the bar, y across the grip faces, z up): the coop carry's grasp
R_TCP_IN_BAR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
#: Timoshenko's shear coefficient of a solid circular section (Poisson 0.5)
SHEAR_COEFFICIENT = 0.9
#: the spread of a shop buffer's stiffness about the formula's
RUBBER_TOLERANCE = 0.3
#: rubber-metal buffers sold off the shelf (rubber diameter, height mm) and their usual hardnesses
BUFFER_SIZES = ((15.0, 15.0), (20.0, 15.0), (20.0, 20.0), (25.0, 20.0), (25.0, 25.0))
SHORE_A = (40.0, 50.0, 60.0, 70.0)
DEFAULT_PLAN = REPO / "outputs" / "fr3" / "dual_fr3_coop_carry" / "coop_carry_plan.npz"
DEFAULT_OUT = REPO / "outputs" / "fr3" / "dual_fr3_compliant_handle"


@dataclass(frozen=True)
class Buffer:
    """A rubber-metal buffer, male-male: its rubber body (mm) and hardness."""

    diameter_mm: float = 20.0
    height_mm: float = 20.0
    shore_a: float = 50.0
    thread: str = "M6"
    stud_mm: float = 18.0


@dataclass(frozen=True)
class Handle:
    """One handle, in mm, in the bar frame: origin at the bar centre, x along the bar, y across the faces the
    fingers close on, z up."""

    bar_length_mm: float = 110.0
    #: across the flats the fingers close on: the pivot insert v2's 35 mm
    grip_width_mm: float = 35.0
    bar_height_mm: float = 32.0
    #: the top is sunk this much over the grip zone, under the gripper's inner knuckles
    grip_zone_mm: float = 50.0
    knuckle_recess_mm: float = 2.5
    buffer: Buffer = field(default_factory=Buffer)
    #: the two buffers' centres along the bar
    buffer_pitch_mm: float = 70.0
    #: ISO 7379 shoulder screw in each bar end (shoulder diameter; the thread is M8)
    pin_diameter_mm: float = 10.0
    #: the pins' radial clearance in the cheeks, and the bar ends' gap to them: how far the handle gives
    travel_mm: float = 4.0
    cheek_mm: float = 10.0
    cradle_width_mm: float = 50.0
    deck_mm: float = 8.0
    base_mm: float = 8.0
    #: the TCP below the bar centre: the insert's socket is at EE z -5.06 mm (10-14 laser), the MJCF's pads
    #: 5.6 mm above it
    tcp_below_bar_mm: float = 5.06


@dataclass(frozen=True)
class TrayPlate:
    """The object: an aluminium plate with a handle each side, in mm. ``grip_height_mm`` is the bar centre
    above the table, the coop carry's 105 mm (0.04 m half-height + 0.065 m)."""

    length_mm: float = 200.0  # along x (the bars)
    width_mm: float = 440.0  # along y (hand to hand)
    thickness_mm: float = 3.0
    corner_mm: float = 10.0
    handle_y_mm: float = 195.0
    grip_height_mm: float = 105.0


# ------------------------------------------------------------------ rubber ---


def youngs_modulus_mpa(shore_a: float) -> float:
    """Gent's relation of Shore A hardness to Young's modulus (MPa)."""
    return 0.0981 * (56.0 + 7.62336 * shore_a) / (0.137505 * (254.0 - 2.54 * shore_a))


def buffer_stiffness(buffer: Buffer) -> np.ndarray:
    """(6, 6) the buffer's top plate on its fixed base, at the top plate centre, axes with z along the
    buffer: N/m, N/rad, Nm/rad. A short Timoshenko beam: shear and bending coupled, compression stiffened by
    the bonded ends (shape factor: loaded over free area), torsion G J / h."""
    E = youngs_modulus_mpa(buffer.shore_a) * 1e6
    G = E / 3.0
    d, h = buffer.diameter_mm * 1e-3, buffer.height_mm * 1e-3
    A, second_moment = math.pi * d**2 / 4.0, math.pi * d**4 / 64.0
    shape = d / (4.0 * h)
    phi = 12.0 * E * second_moment / (SHEAR_COEFFICIENT * G * A * h**2)
    b = E * second_moment / ((1.0 + phi) * h**3)
    K = np.zeros((6, 6))
    # (ux, ry): beam along z, rotation about y turns z towards x
    K[np.ix_([0, 4], [0, 4])] = b * np.array([[12.0, -6.0 * h], [-6.0 * h, (4.0 + phi) * h**2]])
    # (uy, rx): rotation about x turns z away from y
    K[np.ix_([1, 3], [1, 3])] = b * np.array([[12.0, 6.0 * h], [6.0 * h, (4.0 + phi) * h**2]])
    K[2, 2] = E * (1.0 + 2.0 * shape**2) * A / h
    K[5, 5] = G * (math.pi * d**4 / 32.0) / h
    return K


def _moved(K: np.ndarray, r: np.ndarray) -> np.ndarray:
    """A stiffness at point r, as seen from the origin (a twist at the origin moves r by ``v - r x w``)."""
    A = np.eye(6)
    A[:3, 3:] = -cc._skew(np.asarray(r, dtype=np.float64))
    return A.T @ K @ A


def handle_stiffness(handle: Handle, factor: float = 1.0) -> np.ndarray:
    """(6, 6) the bar on its two buffers, at the bar centre, in the bar's axes (the cradle fixed); ``factor``
    scales the rubber (its shop spread)."""
    K_one = buffer_stiffness(handle.buffer) * factor
    z = -handle.bar_height_mm / 2.0 * 1e-3
    return sum(
        _moved(K_one, np.array([s * handle.buffer_pitch_mm / 2.0 * 1e-3, 0.0, z])) for s in (-1.0, 1.0)
    )


def tcp_compliance(handle: Handle, factor: float = 1.0) -> np.ndarray:
    """(6, 6) the handle as a grasp compliance at the TCP in the TCP's axes: ``dual_fr3_closed_chain``'s
    form for a designed grasp."""
    C_bar = np.linalg.inv(handle_stiffness(handle, factor))
    C_tcp = cc._at_point(C_bar, np.array([0.0, 0.0, -handle.tcp_below_bar_mm * 1e-3]))
    B = cc._blockdiag(R_TCP_IN_BAR)
    return B.T @ C_tcp @ B


def turn_stop_deg(handle: Handle) -> float:
    """How far the bar turns about its y or z axis before a pin meets its cheek."""
    lever = (handle.bar_length_mm / 2.0 + handle.travel_mm + handle.cheek_mm / 2.0) * 1e-3
    return math.degrees(math.atan2(handle.travel_mm * 1e-3, lever))


def characteristics(handle: Handle, tray_mass_kg: float) -> dict[str, Any]:
    """What a hand feels and what the tray does, for the drawing and the bench check: the bar's stiffness
    along and across itself and vertically (N/mm, the others free), its turn stiffness (Nm/rad) about each
    axis, how far its stops let it turn, and how far the tray tips per cm its centre of mass is off the line
    between the hands (the two handles turn about their y axes)."""
    K = handle_stiffness(handle)
    C = np.linalg.inv(K)
    names = ("x", "y", "z")
    return {
        "k_n_per_mm": {n: round(float(1.0 / C[i, i]) * 1e-3, 1) for i, n in enumerate(names)},
        "k_nm_per_rad": {n: round(float(1.0 / C[3 + i, 3 + i]), 2) for i, n in enumerate(names)},
        "turn_stop_deg": round(turn_stop_deg(handle), 1),
        "tip_deg_per_cm_offset": round(math.degrees(tray_mass_kg * cc.GRAVITY * 0.01 / (2.0 * K[4, 4])), 2),
        "weight_sag_mm": round(float(tray_mass_kg * cc.GRAVITY / 2.0 * C[2, 2]) * 1e3, 3),
    }


# ------------------------------------------------------------------ design ---


def design(handle: Handle, factor: float = 1.0) -> dict[str, Any]:
    """``dual_fr3_closed_chain.budget``'s design entry."""
    return {
        "compliance": tcp_compliance(handle, factor),
        "travel_mm": handle.travel_mm,
        "travel_deg": turn_stop_deg(handle),
    }


def _name(buffer: Buffer) -> str:
    return f"D{buffer.diameter_mm:g}xH{buffer.height_mm:g} {buffer.shore_a:g}A"


def candidates(handle: Handle = Handle()) -> list[Handle]:
    return [
        replace(handle, buffer=replace(handle.buffer, diameter_mm=d, height_mm=h, shore_a=s))
        for d, h in BUFFER_SIZES
        for s in SHORE_A
    ]


def _margin(row: dict[str, Any], budgets: dict[str, float]) -> float:
    """The tightest of a design row's limits: each budget over its use, each travel over its give."""
    return min(
        budgets["force_n"] / max(row["force_n_p95"], 1e-9),
        budgets["torque_nm"] / max(row["moment_nm_p95"], 1e-9),
        budgets["sag_mm"] / max(row["sag_mm"], 1e-9),
        row["travel_mm"] / max(row["give_mm_p95"], 1e-9),
        row["travel_deg"] / max(row["give_deg_p95"], 1e-9),
    )


def choose(
    plan: dict[str, Any],
    *,
    handles: list[Handle] | None = None,
    mass_kg: float = cc.OBJECT_MASS_KG,
    scenario: str = "measured_b3",
    samples: int = cc.SAMPLES,
) -> dict[str, Any]:
    """Each candidate through D6's budget at 0.7x, 1x and 1.3x its formula stiffness. Ranked: those that fit
    at all three under ``scenario`` (the placement the carry is meant for) first; among them, those that fit
    under the other scenarios too (a placement worse than meant), by their worst margin there."""
    handles = handles or candidates()
    spreads = (1.0 - RUBBER_TOLERANCE, 1.0, 1.0 + RUBBER_TOLERANCE)
    designs = {f"{_name(h.buffer)} x{f:g}": design(h, f) for h in handles for f in spreads}
    report = cc.budget(plan, mass_kg=mass_kg, samples=samples, designs=designs)
    budgets = report["model"]["budgets"]
    rows = {sc: {r["design"]: r for r in report["scenarios"][sc]["designs"]} for sc in report["scenarios"]}
    others = [sc for sc in rows if sc != scenario]
    ranked = []
    for h in handles:
        name = _name(h.buffer)
        spread = {sc: [rows[sc][f"{name} x{f:g}"] for f in spreads] for sc in rows}
        ranked.append(
            {
                "buffer": asdict(h.buffer),
                "name": name,
                "fits": all(r["ok"] for r in spread[scenario]),
                "margin": round(min(_margin(r, budgets) for r in spread[scenario]), 2),
                "fits_others": all(r["ok"] for sc in others for r in spread[sc]),
                "margin_others": round(
                    min((_margin(r, budgets) for sc in others for r in spread[sc]), default=math.inf), 2
                ),
                "nominal": {sc: rows[sc][f"{name} x1"] for sc in rows},
                "spread": spread,
            }
        )
    ranked.sort(key=lambda r: (not r["fits"], not r["fits_others"], -r["margin_others"], -r["margin"]))
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario": scenario,
        "mass_kg": mass_kg,
        "rubber_tolerance": RUBBER_TOLERANCE,
        "budgets": budgets,
        "joint_scales_only": {sc: report["scenarios"][sc]["verdict"] for sc in report["scenarios"]},
        "ranked": ranked,
    }


#: g/cm3 and the printed parts' fill, as the CAD weighs them; the off-the-shelf parts per handle (g): two
#: buffers, two shoulder screws, the inserts, nuts and screws
ALUMINIUM, PETG, PRINT_FILL = 2.70, 1.27, 0.6
BOUGHT_PER_HANDLE_G = 2 * 20.0 + 2 * 14.0 + 25.0


def tray_mass_kg(plate: TrayPlate = TrayPlate(), handle: Handle = Handle()) -> float:
    """The tray as drawn (the CAD's ``parts.json`` weighs the same solids): the aluminium plate, and per handle
    the printed bar and cradle at their fill and the bought parts. Holes under a gram are left out."""
    h, t = handle, plate
    bar = (
        h.bar_length_mm * h.grip_width_mm * h.bar_height_mm
        - h.grip_zone_mm * h.grip_width_mm * h.knuckle_recess_mm
    )
    bar -= 2 * math.pi * (4.0**2 * 14.0 + 5.0**2 * 16.0)
    hole_r = h.pin_diameter_mm / 2 + h.travel_mm
    x_in = h.bar_length_mm / 2 + h.travel_mm
    base_top = -t.grip_height_mm + t.thickness_mm + h.base_mm
    cradle = (
        2 * (x_in + h.cheek_mm) * h.cradle_width_mm * h.base_mm
        + 2 * h.cheek_mm * h.cradle_width_mm * (hole_r + 8.0 - base_top)
        + 2 * x_in * h.cradle_width_mm * h.deck_mm
        - 2 * math.pi * hole_r**2 * h.cheek_mm
    )
    plate_mm3 = (t.length_mm * t.width_mm - (4 - math.pi) * t.corner_mm**2) * t.thickness_mm
    grams = plate_mm3 * 1e-3 * ALUMINIUM + 2 * (
        (bar + cradle) * 1e-3 * PETG * PRINT_FILL + BOUGHT_PER_HANDLE_G
    )
    return round(grams / 1e3, 3)


# ------------------------------------------------------------------ cad ---


#: rgba per part, by the start of its mesh's name
MESH_COLORS = {
    "plate": [0.78, 0.80, 0.83, 1.0],
    "bar": [0.96, 0.74, 0.18, 1.0],
    "cradle": [0.32, 0.52, 0.82, 1.0],
    "buffer": [0.12, 0.12, 0.12, 1.0],
}
STEEL = [0.62, 0.62, 0.64, 1.0]


def mesh_list(out: Path) -> list[tuple[Path, list[float]]]:
    """The tray's part meshes the CAD wrote (``out/meshes``, tray frame, mm) with their colours."""
    meshes = sorted((Path(out) / "meshes").glob("*.stl"))
    if not meshes:
        raise FileNotFoundError(f"{out}/meshes: no meshes, run --cad first")
    return [
        (m, next((c for k, c in MESH_COLORS.items() if m.stem.split("_", 1)[-1].startswith(k)), STEEL))
        for m in meshes
    ]


def render_alone(
    out: Path, plate: TrayPlate = TrayPlate(), *, width: int = 1600, height: int = 1000
) -> list[Path]:
    """The tray on a table, and one handle close: PNGs from MuJoCo (a z-buffer, unlike the drawing's
    matplotlib)."""
    import mujoco
    from PIL import Image

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.worldbody.add_light(pos=[0.3, -0.5, 1.2], dir=[-0.3, 0.5, -1.0], diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.6, 0.6, 0.01], rgba=[0.93, 0.92, 0.9, 1.0]
    )
    for k, (path, rgba) in enumerate(mesh_list(out)):
        spec.add_mesh(name=f"m{k}", file=str(path.resolve()), scale=[1e-3] * 3)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", rgba=rgba)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    y, z = plate.handle_y_mm * 1e-3, plate.grip_height_mm * 1e-3
    written = []
    for name, (lookat, distance, elevation, azimuth) in {
        "render_tray": ([0.0, 0.0, 0.05], 0.85, -28.0, 230.0),
        "render_handle": ([0.0, -y, z - 0.03], 0.32, -18.0, 235.0),
    }.items():
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = lookat
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        renderer.update_scene(data, camera=camera)
        Image.fromarray(np.asarray(renderer.render(), dtype=np.uint8)).save(out / f"{name}.png")
        written.append(out / f"{name}.png")
    renderer.close()
    return written


def render_gripped(out: Path, plan: Path) -> list[Path]:
    """Both FR3s holding the tray, at the middle of the coop carry's held segment: the whole scene, and the
    right hand on its handle close."""
    from tools.fr3 import dual_fr3_coop_carry as cp

    with np.load(plan) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    if "tray" not in arrays:
        raise ValueError(f"{plan}: no tray poses (not a dual_fr3_coop_carry plan)")
    segments = cc.held_segments(arrays["held"].astype(bool))
    a, b = segments[0]
    k = (a + b) // 2
    x, y, z, yaw = arrays["tray"][k]
    tray = cp.Tray()
    bar = cp.tray_pose(x, y, z, yaw) @ np.r_[0.0, -tray.bar_y, tray.bar_z, 1.0]
    meshes = mesh_list(out)
    spacing = float(np.asarray(arrays["T_right_left"])[1, 3])
    return [
        cp.still(arrays, tray, k, out / "render_gripped.png", meshes=meshes,
                 view=([0.5, spacing / 2, 0.25], 1.45, -24.0, 205.0)),
        cp.still(arrays, tray, k, out / "render_gripped_closeup.png", meshes=meshes,
                 view=(bar[:3].tolist(), 0.42, -14.0, 215.0)),
    ]  # fmt: skip


def cad(out: Path, handle: Handle, plate: TrayPlate, extra: dict[str, Any], plan: Path | None = None) -> int:
    """The parts and the drawings, through FreeCAD (its own Python; the script reads ``params.json``): the
    geometry first, then the renders from its meshes, then the drawings with the renders in them."""
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "handle": asdict(handle),
        "plate": asdict(plate),
        "out": str(out.resolve()),
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        **extra,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    script = Path(__file__).with_name("dual_fr3_compliant_handle_cad.py")
    for stage in ("geometry", "drawings"):
        done = out / f"cad_{stage}_done.json"
        done.unlink(missing_ok=True)
        env = {
            **os.environ,
            "COMPLIANT_HANDLE_PARAMS": str((out / "params.json").resolve()),
            "COMPLIANT_HANDLE_STAGE": stage,
        }
        proc = subprocess.run(
            ["freecad", "freecadcmd", str(script)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        # freecadcmd swallows a script's exceptions: the script writes its own log and a done marker
        if not done.exists():
            log = out / f"cad_{stage}.log"
            print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
            print(log.read_text(encoding="utf-8") if log.exists() else f"no {log.name}")
            return 1
        if stage == "geometry":
            render_alone(out, plate)
            if plan is not None:
                render_gripped(out, plan)
    return 0


def _summary_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        f"tray {report['mass_kg']} kg, rubber +-{report['rubber_tolerance']:.0%}; meant for [{report['scenario']}]"
    ]
    for r in report["ranked"]:
        cells = []
        for sc, n in r["nominal"].items():
            cells.append(
                f"[{sc}] {n['force_n_p95']:4.1f} N {n['moment_nm_p95']:4.2f} Nm "
                f"give {n['give_mm_p95']:4.2f} mm {n['give_deg_p95']:4.2f} deg"
            )
        verdict = ("fits" if r["fits"] else "-") + ("+" if r["fits_others"] else "")
        lines.append(f"  {r['name']:<14} {verdict:<5} " + "  ".join(cells))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The compliant handle for a two-hand carry (D6): pick and draw"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="a held plan npz")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mass", type=float, default=None, help="tray kg (default: estimated from the parts)"
    )
    parser.add_argument("--samples", type=int, default=cc.SAMPLES)
    parser.add_argument(
        "--pick", default=None, help='draw this candidate (e.g. "D20xH20 40A") instead of the top-ranked one'
    )
    parser.add_argument("--cad", action="store_true", help="also write STEP/STL and the drawings (FreeCAD)")
    args = parser.parse_args(argv)
    plate = TrayPlate()
    mass = args.mass if args.mass is not None else tray_mass_kg(plate)
    report = choose(cc.read_plan(args.plan), mass_kg=mass, samples=args.samples)
    report["plan"] = str(args.plan)
    pool = [r for r in report["ranked"] if r["fits"] and (args.pick is None or r["name"] == args.pick)]
    best = pool[0] if pool else None
    if args.pick is not None and best is None:
        print(f"{args.pick}: not a candidate that fits")
    if best is not None:
        handle = replace(Handle(), buffer=Buffer(**best["buffer"]))
        report["chosen"] = {"name": best["name"], **characteristics(handle, mass)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "handle_choice.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n".join(_summary_lines(report)))
    print(f"chosen: {report.get('chosen')}")
    if args.cad:
        if best is None:
            print("no candidate fits: nothing to draw")
            return 1
        if cad(args.out, handle, plate, {"chosen": report["chosen"], "mass_kg": mass}, args.plan) != 0:
            return 1
        weighed = json.loads((args.out / "parts.json").read_text(encoding="utf-8"))["tray_g"] / 1e3
        if abs(weighed - mass) > 0.03 * mass:
            print(
                f"the CAD weighs the tray {weighed:.3f} kg, not {mass:.3f}: rerun with --mass {weighed:.3f}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
