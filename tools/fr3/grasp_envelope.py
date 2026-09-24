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

"""P0 (card 13): the grasp success envelope -- where, relative to a standing peg, a scripted grasp works.

Why it exists: every threshold the grasp funnel will need -- the XY capture radius that gates the
descent, the relative grasp height it closes at, the offset beyond which the fingers knock the
peg over -- has so far been a guess or a number borrowed from the demonstrations (51.4 mm close
height, card 12). None of them has been measured against the gripper itself. This sweep measures
them with no policy in the loop, so a failure here is the hardware's or the control's, never the
model's.

One trial is one point (dx, dy, dz) relative to the peg:

    above the aim at carry height -> straight down at servo speed, contact-aware
      contact on the way down            -> `contact`, retreat without closing
      reached the aim                    -> close, lift `checkLiftM`, read the width
        width >= heldWidth               -> `held`: lowered straight back down, released
        otherwise                        -> `empty`: opened, retreated
    then a *verify grasp* at the nominal pose (0, 0, verifyDz): the same primitive
      held                               -> the peg is still standing at the spot, and
                                            re-centred along the closing axis for the next trial
      not held                           -> `pegDisturbed`: the trial moved the peg out of reach

Why a held peg goes back by the path it came up: it was lifted straight up from where it stood,
so straight down puts it back where it stood without the loop having to know where that is. The
one thing that does move it is the close itself, which drags an off-centre peg to the middle of
the fingers along their closing axis; the verify grasp drags it back the same way. Along the
pads nothing pushes it, so nothing needs undoing.

What this rig cannot tell apart, stated rather than hidden: without a camera in the loop a peg
that fell over and a peg pushed beyond the verify grasp's reach read the same (`pegDisturbed`).
And `contact` covers both a finger landing on the peg's top and fingertips reaching the table --
the dz column separates those, since only one of them depends on dz alone.

Heights are relative on purpose. `pegRefZ` is the TCP height of the scene reset's own grip on a
standing peg (it places at 0.058), so dz = 0 is "where the reset holds it" and the 51.4 mm
demonstration close is dz = -6.6 mm. With the peg always on the one table this is a fixed world z;
the grasp funnel reads the same field, so a localiser that reports a peg height later replaces
one number, not the semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable

from tools.fr3.grasp_loop import (
    GRASP_LOOP_CHECK_LIFT_M,
    GRASP_LOOP_HELD_WIDTH,
    GRASP_LOOP_PICK_XYZ,
    GRASP_LOOP_TARGET_Z,
    GRASP_LOOP_WIDTH_SETTLE_S,
    read_width,
    wilson_interval,
)
from tools.fr3.scene_reset import (
    SCENE_RESET_LIFT_M,
    SceneResetError,
    SceneResetRequest,
    _check_xyz_in_workspace,
    _hold_where_it_is,
    _move_to_start,
    _observation_xyz_rotvec_gripper,
    _run_step,
    _workspace_bounds,
    execute_scene_reset,
    precise_sleep,
)
from tools.fr3.terminal_servo import TerminalServoRequest, descend_until_refused

# The median of the 38 `resetTarget`s in outputs/rollouts/rollout_log.jsonl (2026-09-22): the
# place a rollout typically has to grasp at, so the envelope is measured where it is used.
GRASP_ENVELOPE_SPOT_XY = (0.4331, -0.1667)
# Coarse sweep defaults. The XY grid spans the 18.8 vs 36.5 mm lateral split card 11 found between
# held and missed rollouts only at its low end on purpose: past 12 mm an open finger is expected to
# land on a 15 mm peg, and the coarse pass is for finding the edge, not for mapping the far side.
GRASP_ENVELOPE_XY_OFFSETS_MM = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)
# The XY grid is swept at the demonstrations' close height, 51.4 mm = pegRefZ - 6.6 mm.
GRASP_ENVELOPE_XY_DZ_MM = -6.0
# The dz column brackets the demonstrations' close (-6.6 mm) and the hands-off rollouts' median
# (71.5 mm = +13.5 mm), and runs past the 76.7-98.5 mm band where 7 of 19 closes caught nothing.
GRASP_ENVELOPE_DZ_OFFSETS_MM = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 14.0, 20.0, 28.0)
GRASP_ENVELOPE_CENTRE_REPEATS = 5
# The lowest TCP z any aim may name. The demonstrations reached 0.046 on the way to the peg; a
# couple of millimetres under that is the floor, and contact detection is the net below it.
GRASP_ENVELOPE_MIN_TCP_Z = 0.044
# The terminal servo's descent speed: the contact test was validated at it, 9/9, zero false stops.
GRASP_ENVELOPE_DESCENT_SPEED_MS = 0.02
# How long a person has to put a lost peg back before the run gives up and parks.
GRASP_ENVELOPE_OPERATOR_WAIT_S = 1800.0
# A verify grasp that comes up empty is asked about again after a restage, but not forever: a spot
# the reset cannot stage a peg at is a staging fault, and the run should say so and stop.
GRASP_ENVELOPE_MAX_RESTAGES = 3
# The held rate an edge has to keep. One miss in five is tolerated; one miss is never an edge.
GRASP_ENVELOPE_EDGE_RATE = 0.8


class GraspEnvelopeHalt(RuntimeError):
    """A condition the sweep must stop on, named for the summary."""


@dataclass(frozen=True)
class EnvelopePoint:
    index: int
    block: str  # "xy" | "z" | "centre" | "extra"
    dxMm: float
    dyMm: float
    dzMm: float
    aimXyz: tuple[float, float, float]


@dataclass(frozen=True)
class GraspEnvelopeRequest:
    spotXy: tuple[float, float] = GRASP_ENVELOPE_SPOT_XY
    # TCP height of the reset's own grip on a standing peg: relative grasp height zero.
    pegRefZ: float = GRASP_LOOP_TARGET_Z
    xyOffsetsMm: tuple[float, ...] = GRASP_ENVELOPE_XY_OFFSETS_MM
    xyDzMm: float = GRASP_ENVELOPE_XY_DZ_MM
    dzOffsetsMm: tuple[float, ...] = GRASP_ENVELOPE_DZ_OFFSETS_MM
    centreRepeats: int = GRASP_ENVELOPE_CENTRE_REPEATS
    # Fine-scan points, typed in after the coarse pass has found the edge: (dx, dy, dz) in mm.
    extraPointsMm: tuple[tuple[float, float, float], ...] = ()
    repeats: int = 1
    seed: int = 0
    # Where the peg comes from: the reset's fixture ("fixture") or already standing at the spot
    # ("spot", which is how every run after the first starts, since each run leaves it there).
    start: str = "fixture"
    pickXyz: tuple[float, float, float] = GRASP_LOOP_PICK_XYZ
    verifyDzMm: float = 0.0
    approachAboveM: float = SCENE_RESET_LIFT_M
    checkLiftM: float = GRASP_LOOP_CHECK_LIFT_M
    heldWidth: float = GRASP_LOOP_HELD_WIDTH
    descentSpeedMs: float = GRASP_ENVELOPE_DESCENT_SPEED_MS
    minTcpZ: float = GRASP_ENVELOPE_MIN_TCP_Z
    maxSeconds: float = 0.0
    operatorWaitS: float = GRASP_ENVELOPE_OPERATOR_WAIT_S
    openGripper: float = 1.0
    closedGripper: float = 0.0
    controlPeriodS: float = 1.0 / 30.0
    requestId: str = ""

    def peg_xyz(self) -> tuple[float, float, float]:
        return (self.spotXy[0], self.spotXy[1], self.pegRefZ)

    def aim(self, dx_mm: float, dy_mm: float, dz_mm: float) -> tuple[float, float, float]:
        return (
            self.spotXy[0] + dx_mm / 1000.0,
            self.spotXy[1] + dy_mm / 1000.0,
            self.pegRefZ + dz_mm / 1000.0,
        )

    def carry_z(self) -> float:
        return self.pegRefZ + self.approachAboveM

    def step_request(self, request_id: str) -> SceneResetRequest:
        """The carrier `_run_step` reads its timeout, tolerance and period from."""

        return SceneResetRequest(
            pickXyz=self.pickXyz,
            targetXyz=self.peg_xyz(),
            openGripper=self.openGripper,
            closedGripper=self.closedGripper,
            controlPeriodS=self.controlPeriodS,
            requestId=request_id,
        )

    def descent_request(self, target_xyz: tuple[float, float, float], request_id: str) -> TerminalServoRequest:
        """The terminal servo's contact-aware descent, pointed at a grasp instead of a hole."""

        return TerminalServoRequest(
            xyz=target_xyz,
            maxSpeedMs=self.descentSpeedMs,
            controlPeriodS=self.controlPeriodS,
            minZ=self.minTcpZ,
            requestId=request_id,
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def parse_offsets_mm(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in str(text).split(",") if part.strip())


def parse_points_mm(text: str) -> tuple[tuple[float, float, float], ...]:
    """`dx,dy,dz; dx,dy,dz; ...` in millimetres."""

    points = []
    for chunk in str(text or "").split(";"):
        if not chunk.strip():
            continue
        parts = [float(part) for part in chunk.split(",") if part.strip()]
        if len(parts) != 3:
            raise SceneResetError(f"extra point {chunk.strip()!r} is not dx,dy,dz.")
        points.append((parts[0], parts[1], parts[2]))
    return tuple(points)


def parse_xy(text: str) -> tuple[float, float]:
    parts = [float(part) for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise SceneResetError(f"{text!r} is not x,y in metres.")
    return (parts[0], parts[1])


def build_envelope_schedule(request: GraspEnvelopeRequest) -> list[EnvelopePoint]:
    """The XY grid at `xyDzMm`, the dz column at the centre, centre repeats, extras -- shuffled.

    Shuffled because the spot drifts the way the hole does: placements, verify grasps and slow
    creep all move the peg a little over a run, and a sweep walked in grid order would read that
    drift as a spatial pattern. Seeded, so a resumed run expands to the same order and can skip by
    index.
    """

    raw: list[tuple[str, float, float, float]] = []
    for _repeat in range(max(1, request.repeats)):
        for dx in request.xyOffsetsMm:
            for dy in request.xyOffsetsMm:
                raw.append(("xy", dx, dy, request.xyDzMm))
        for dz in request.dzOffsetsMm:
            raw.append(("z", 0.0, 0.0, dz))
    for _repeat in range(max(0, request.centreRepeats)):
        raw.append(("centre", 0.0, 0.0, request.xyDzMm))
    for dx, dy, dz in request.extraPointsMm:
        raw.append(("extra", dx, dy, dz))
    random.Random(request.seed).shuffle(raw)
    return [
        EnvelopePoint(
            index=index,
            block=block,
            dxMm=float(dx),
            dyMm=float(dy),
            dzMm=float(dz),
            aimXyz=tuple(round(v, 6) for v in request.aim(dx, dy, dz)),  # type: ignore[arg-type]
        )
        for index, (block, dx, dy, dz) in enumerate(raw)
    ]


def validate_grasp_envelope(
    request: GraspEnvelopeRequest,
    schedule: list[EnvelopePoint],
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Refuse the plan before anything moves: every aim, and the carry point above it, in the fence."""

    if not schedule:
        raise SceneResetError("the schedule is empty: no offsets were given.")
    if request.start not in ("fixture", "spot"):
        raise SceneResetError(f"start must be 'fixture' or 'spot', got {request.start!r}.")
    if not 0.0 < request.heldWidth < 1.0:
        raise SceneResetError("heldWidth is a normalized gripper reading and must be in (0, 1).")
    if request.checkLiftM <= 0.0 or request.descentSpeedMs <= 0.0:
        raise SceneResetError("checkLiftM and descentSpeedMs must be positive.")
    if request.descentSpeedMs > 0.05:
        # The contact test is growth per millimetre of setpoint travel and was validated at 0.02.
        # Past a few times that the arm's answer delay dominates and a peg can go over before the
        # growth has had a window to show.
        raise SceneResetError("descentSpeedMs above 0.05 m/s outruns the contact test; use <= 0.05.")
    low, high = _workspace_bounds(workspace_min, workspace_max)
    verify = request.aim(0.0, 0.0, request.verifyDzMm)
    aims = [point.aimXyz for point in schedule] + [verify]
    lowest = min(aim[2] for aim in aims)
    if lowest < request.minTcpZ - 1e-9:
        raise SceneResetError(
            f"an aim reaches TCP z {lowest:.4f}, below the floor {request.minTcpZ:.4f}; "
            f"raise the lowest dz or lower minTcpZ deliberately."
        )
    for aim in aims:
        if aim[2] >= request.carry_z():
            raise SceneResetError(f"aim z {aim[2]:.4f} is at or above the carry height {request.carry_z():.4f}.")
        _check_xyz_in_workspace(aim, "aim", low, high)
        _check_xyz_in_workspace((aim[0], aim[1], request.carry_z()), "aim_above", low, high)
        _check_xyz_in_workspace((aim[0], aim[1], aim[2] + request.checkLiftM), "aim_lifted", low, high)
    if request.start == "fixture":
        _check_xyz_in_workspace(request.pickXyz, "pickXyz", low, high)
    widest = max(math.hypot(point.dxMm, point.dyMm) for point in schedule)
    return {
        "ok": True,
        "points": len(schedule),
        "widestXyMm": round(widest, 2),
        "lowestTcpZ": round(lowest, 4),
        "highestTcpZ": round(max(aim[2] for aim in aims), 4),
        "blocks": {block: sum(1 for p in schedule if p.block == block) for block in ("xy", "z", "centre", "extra")},
    }


def describe_schedule(request: GraspEnvelopeRequest, schedule: list[EnvelopePoint]) -> str:
    blocks = {block: [p for p in schedule if p.block == block] for block in ("xy", "z", "centre", "extra")}
    lines = [
        f"grasp envelope at spot x={request.spotXy[0]:.4f} y={request.spotXy[1]:.4f}, "
        f"pegRefZ={request.pegRefZ:.4f} (dz=0 is the reset's own grip; the demonstrations close at dz=-6.6)",
        f"start: {'reset from fixture ' + ','.join(f'{v:.4f}' for v in request.pickXyz) if request.start == 'fixture' else 'peg already standing at the spot'}",
        f"xy grid: {len(blocks['xy'])} trials, offsets {','.join(f'{v:g}' for v in request.xyOffsetsMm)} mm at dz={request.xyDzMm:g} mm",
        f"dz column: {len(blocks['z'])} trials at dx=dy=0, dz {','.join(f'{v:g}' for v in request.dzOffsetsMm)} mm "
        f"(TCP z {request.pegRefZ + min(request.dzOffsetsMm, default=0) / 1000:.4f}..{request.pegRefZ + max(request.dzOffsetsMm, default=0) / 1000:.4f})",
        f"centre repeats: {len(blocks['centre'])} at (0,0,{request.xyDzMm:g})",
        f"extra: {len(blocks['extra'])}",
        f"every trial is followed by a verify grasp at (0,0,{request.verifyDzMm:g}); expect ~35 s per trial, "
        f"~{len(schedule) * 35 / 60:.0f} min in all",
        "order is shuffled (seed {}), first ten: {}".format(
            request.seed,
            " ".join(f"#{p.index}({p.dxMm:g},{p.dyMm:g},{p.dzMm:g})" for p in schedule[:10]),
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- motion ---


def _up_and_over(
    robot: Any,
    request: GraspEnvelopeRequest,
    step_request: SceneResetRequest,
    xy: tuple[float, float],
    rotvec: tuple[float, float, float],
) -> None:
    """Straight up to carry height, then across at it, fingers open: nothing moves sideways low."""

    xyz, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    carry = request.carry_z()
    if xyz[2] < carry - 1e-4:
        _run_step(robot, step_request, "retreat_8cm", (xyz[0], xyz[1], carry), rotvec, request.openGripper)
    _run_step(robot, step_request, "go_to_pick_above", (xy[0], xy[1], carry), rotvec, request.openGripper)


def _rounded(xyz: Iterable[float], digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in xyz]


def scripted_grasp(
    robot: Any,
    request: GraspEnvelopeRequest,
    aim: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    *,
    request_id: str,
) -> dict[str, Any]:
    """One grasp at `aim`, graded by the lift, leaving the peg where it found it and the arm above.

    The same primitive serves the sweep and the verify grasp, so the two are graded identically.
    """

    step_request = request.step_request(request_id)
    _up_and_over(robot, request, step_request, (aim[0], aim[1]), rotvec)
    descent = descend_until_refused(
        robot, request.descent_request(aim, request_id), aim, rotvec, request.openGripper
    )
    result: dict[str, Any] = {
        "descentStoppedOn": descent["stoppedOn"],
        "lagMm": round(descent["lagMm"], 2),
        "peakGrowthMm": round(descent["peakGrowthMm"], 2),
    }
    if descent["stoppedOn"] == "contact":
        # Something is under the fingers. Closing now would grade the height the arm was stopped
        # at, not the one that was asked for, so this point's answer is the contact itself.
        stopped = descent["stoppedAtXyz"]
        result.update({"verdict": "contact", "contactZ": round(float(stopped[2]), 5), "closeXyz": None})
        _up_and_over(robot, request, step_request, (aim[0], aim[1]), rotvec)
        return result

    close_xyz, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    # `timeout` is not contact: this arm's static-friction dead-band can leave a free descent a
    # few mm short of the setpoint (4.6-5.7 mm measured on descents, 2026-09-11/22). That residual
    # is constant, and the contact test reads growth, so it never fires -- the descent just ends
    # outside its 2 mm tolerance. The grasp goes ahead where the arm stopped, and how far short
    # that was is kept: the envelope is read on `closeOffsetMm`, the measured pose.
    result["descentShortMm"] = round((float(close_xyz[2]) - aim[2]) * 1000.0, 2)
    # Closed where the arm measurably is, not at the aim: the residual is a few mm of dead-band,
    # and commanding the aim again during the close would push the fingers into whatever the
    # descent stopped short of.
    _run_step(robot, step_request, "close_gripper", close_xyz, rotvec, request.closedGripper)
    width_at_close = read_width(robot, period_s=request.controlPeriodS)
    lifted = (close_xyz[0], close_xyz[1], close_xyz[2] + request.checkLiftM)
    _run_step(robot, step_request, "lift_8cm_after_grasp", lifted, rotvec, request.closedGripper)
    precise_sleep(GRASP_LOOP_WIDTH_SETTLE_S)
    width = read_width(robot, period_s=request.controlPeriodS)
    held = width >= request.heldWidth
    result.update(
        {
            "verdict": "held" if held else "empty",
            "closeXyz": _rounded(close_xyz),
            "widthAtClose": round(width_at_close, 4),
            "widthLifted": round(width, 4),
        }
    )
    if held:
        # Back down the way it came up, contact-aware: a peg that slid in the fingers meets the
        # table early, and that is a stop, not a push.
        put_back = descend_until_refused(
            robot, request.descent_request(close_xyz, request_id), close_xyz, rotvec, request.closedGripper
        )
        result["putBackStoppedOn"] = put_back["stoppedOn"]
    here, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    _run_step(robot, step_request, "open_gripper", here, rotvec, request.openGripper)
    _up_and_over(robot, request, step_request, (here[0], here[1]), rotvec)
    return result


def stage_from_fixture(
    robot: Any, request: GraspEnvelopeRequest, rotvec: tuple[float, float, float], *, request_id: str
) -> None:
    """The ordinary scene reset, fixture to spot, with the arm left above the spot."""

    reset = execute_scene_reset(
        robot,
        replace(
            request.step_request(request_id),
            targetXyz=request.peg_xyz(),
            returnToStart=False,
        ),
    )
    if not reset.get("ok"):
        raise GraspEnvelopeHalt(f"scene_reset_failed: {reset.get('error')}")


# ---------------------------------------------------------------------------- summary ---


def _rings(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], float]) -> list[dict[str, Any]]:
    """Trials grouped by `key` (rounded to 0.01), ascending, with held and disturbed counts."""

    groups: dict[float, dict[str, Any]] = {}
    for row in rows:
        k = round(key(row), 2)
        g = groups.setdefault(k, {"at": k, "n": 0, "held": 0, "disturbed": 0})
        g["n"] += 1
        g["held"] += 1 if row["verdict"] == "held" else 0
        g["disturbed"] += 1 if row.get("pegDisturbed") else 0
    return [groups[k] for k in sorted(groups)]


def summarize_envelope(
    rows: Iterable[dict[str, Any]],
    request: GraspEnvelopeRequest,
    *,
    edge_rate: float = GRASP_ENVELOPE_EDGE_RATE,
) -> dict[str, Any]:
    """Per-point counts and the four numbers card 13 asks for, each on commanded offsets.

    Every edge is a rate, never a single trial. The first real sweep (09-23) held 48 of 49 grid
    points out to 17 mm, and its one miss sat at 4 mm with its neighbours out to 12 mm all held;
    an "every trial inside r was held" rule turned that single outlier into a capture radius of 0.

    xyCaptureRadiusMm   walking the xy rings (trials at the grid's own dz) outward, the largest
                        radius r such that the pooled held rate of every trial at radius <= r is
                        >= edge_rate and ring r itself is at least half held. Misses inside it are
                        listed by offset rather than hidden, and `xyCaptureAtGridEdge` says the
                        grid ran out before the rate did: the true radius is at least this.
    graspDzIntervalMm   the contiguous run of dz values on the centre column, around the xy grid's
                        own dz, whose held rate is >= edge_rate. `graspDzOpen` says which ends are
                        the edge of the column rather than an observed failure.
    contactBelowDzMm    the highest dz on the centre column that stopped on contact.
    knockOverRadiusMm   xy grid only: the smallest ring r from which more than half of the trials at
                        radius >= r left the peg disturbed. A peg tipped by fingers closing on its
                        top (the column, high dz) is a height result, not a radius.

    With one trial per point these are still the edges of a coarse grid; the fine scan sharpens them.
    """

    trials = [r for r in rows if r.get("kind") == "trial"]
    cells: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in trials:
        key = (row["dxMm"], row["dyMm"], row["dzMm"])
        cell = cells.setdefault(
            key, {"dxMm": key[0], "dyMm": key[1], "dzMm": key[2], "n": 0, "held": 0, "empty": 0, "contact": 0, "disturbed": 0}
        )
        cell["n"] += 1
        cell[row["verdict"]] = cell.get(row["verdict"], 0) + 1
        cell["disturbed"] += 1 if row.get("pegDisturbed") else 0

    def radius(row: dict[str, Any]) -> float:
        return math.hypot(row["dxMm"], row["dyMm"])

    xy_rows = [r for r in trials if r.get("dzMm") == request.xyDzMm]
    rings = _rings(xy_rows, radius)
    capture: float | None = None
    pooled_n = pooled_held = 0
    for ring in rings:
        n, held = pooled_n + ring["n"], pooled_held + ring["held"]
        if held < edge_rate * n or ring["held"] < 0.5 * ring["n"]:
            break
        capture, pooled_n, pooled_held = ring["at"], n, held
    inside = [r for r in xy_rows if capture is not None and radius(r) <= capture + 1e-9]
    misses_inside = [[r["dxMm"], r["dyMm"]] for r in inside if r["verdict"] != "held"]

    knock: float | None = None
    for i, ring in enumerate(rings):
        outer = rings[i:]
        n = sum(g["n"] for g in outer)
        if n and sum(g["disturbed"] for g in outer) > 0.5 * n:
            knock = ring["at"]
            break

    column = [r for r in trials if r["dxMm"] == 0.0 and r["dyMm"] == 0.0]
    dz_rings = _rings(column, lambda r: r["dzMm"])
    good = [g["held"] >= edge_rate * g["n"] for g in dz_rings]
    interval: list[float] | None = None
    open_ends = [False, False]
    if any(good):
        dzs = [g["at"] for g in dz_rings]
        anchor = min((i for i, ok in enumerate(good) if ok), key=lambda i: abs(dzs[i] - request.xyDzMm))
        lo = hi = anchor
        while lo > 0 and good[lo - 1]:
            lo -= 1
        while hi < len(dzs) - 1 and good[hi + 1]:
            hi += 1
        interval = [dzs[lo], dzs[hi]]
        open_ends = [lo == 0, hi == len(dzs) - 1]
    contact_dz = [r["dzMm"] for r in column if r["verdict"] == "contact"]

    centre = [r for r in trials if r["dxMm"] == 0.0 and r["dyMm"] == 0.0 and r["dzMm"] == request.xyDzMm]
    centre_held = sum(1 for r in centre if r["verdict"] == "held")
    verifies = [r for r in trials if r.get("verifyVerdict") is not None]
    verify_held = sum(1 for r in verifies if r["verifyVerdict"] == "held")
    counts = {v: sum(1 for r in trials if r["verdict"] == v) for v in ("held", "empty", "contact")}

    def med(values: list[float]) -> float | None:
        return round(statistics.median(values), 4) if values else None

    return {
        "trials": len(trials),
        "counts": counts,
        "disturbed": sum(1 for r in trials if r.get("pegDisturbed")),
        "edgeRate": edge_rate,
        "xyCaptureRadiusMm": capture,
        "xyCaptureAtGridEdge": bool(rings) and capture == rings[-1]["at"],
        "xyCaptureHeld": {
            "n": pooled_n,
            "held": pooled_held,
            "wilson95": [round(v, 3) for v in wilson_interval(pooled_held, pooled_n)] if pooled_n else None,
        },
        "xyMissesInsideCapture": misses_inside,
        "xyRings": [{"radiusMm": g["at"], "n": g["n"], "held": g["held"], "disturbed": g["disturbed"]} for g in rings],
        "graspDzIntervalMm": interval,
        "graspDzOpen": open_ends,
        "dzColumn": [{"dzMm": g["at"], "n": g["n"], "held": g["held"], "disturbed": g["disturbed"]} for g in dz_rings],
        "contactBelowDzMm": max(contact_dz) if contact_dz else None,
        "knockOverRadiusMm": knock,
        "xyDisturbed": sum(g["disturbed"] for g in rings),
        "centre": {
            "n": len(centre),
            "held": centre_held,
            "wilson95": [round(v, 3) for v in wilson_interval(centre_held, len(centre))],
        },
        # Every trial is followed by a nominal grasp, so this is the largest single sample of the
        # "is the staging itself reliable" question the run produces.
        "verify": {
            "n": len(verifies),
            "held": verify_held,
            "wilson95": [round(v, 3) for v in wilson_interval(verify_held, len(verifies))],
            "medianWidth": med([r["verifyWidth"] for r in verifies if r.get("verifyWidth") is not None]),
        },
        "cells": sorted(cells.values(), key=lambda c: (c["dzMm"], c["dxMm"], c["dyMm"])),
    }


def done_indices(rows: Iterable[dict[str, Any]]) -> set[int]:
    return {int(r["index"]) for r in rows if r.get("kind") == "trial" and "index" in r}


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The last line of an interrupted run can be half written; what precedes it stands.
            break
    return rows


# ---------------------------------------------------------------------------- operator ---


class FileOperatorGate:
    """Wait for a person through a file, because an unattended run's stdin is /dev/null.

    The Unattended Runs page answers by creating `continue_path` (its "已放回，继续" button); the
    boundary STOP file answers "nobody is coming". Polled, not watched: this runs for seconds per
    hour, and a poll is the one mechanism that works the same over NFS, ssh and a page reload.
    """

    def __init__(
        self,
        continue_path: Path,
        *,
        stop_requested: Callable[[], bool],
        timeout_s: float,
        poll_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.continue_path = Path(continue_path)
        self.stop_requested = stop_requested
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.sleep = sleep

    def clear(self) -> None:
        self.continue_path.unlink(missing_ok=True)

    def __call__(self, message: str) -> bool:
        self.clear()
        print(f"[ATTENTION] grasp_envelope_needs_operator {message}", flush=True)
        waited = 0.0
        while waited < self.timeout_s:
            if self.continue_path.exists():
                self.clear()
                print("[INFO] grasp_envelope_operator=continued", flush=True)
                return True
            if self.stop_requested():
                break
            self.sleep(self.poll_s)
            waited += self.poll_s
        print("[INFO] grasp_envelope_operator=gone", flush=True)
        return False


# ---------------------------------------------------------------------------- the loop ---


def run_grasp_envelope(
    robot: Any,
    request: GraspEnvelopeRequest,
    schedule: list[EnvelopePoint],
    *,
    on_row: Callable[[dict[str, Any]], None],
    should_stop: Callable[[], bool] = lambda: False,
    wait_for_operator: Callable[[str], bool] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    prior_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Walk `schedule`, one row per point, and end with a summary row. Leaves the peg at the spot.

    The peg's standing at the spot is the loop's invariant: established before the first trial
    (reset from the fixture, or a verify grasp if it is already there), re-established after every
    trial by the verify grasp, and restored by a person when a verify grasp comes up empty.
    """

    xyz, rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    started = clock()
    rows: list[dict[str, Any]] = []
    halted = ""
    spot = request.peg_xyz()
    verify_aim = request.aim(0.0, 0.0, request.verifyDzMm)

    def restage(reason: str, trial_index: int | None) -> bool:
        """Put a peg back at the spot, through a person and the reset. False: nobody came."""

        for attempt in range(GRASP_ENVELOPE_MAX_RESTAGES):
            message = (
                f"{reason}: put the peg back in the pick fixture at "
                f"{request.pickXyz[0]:.4f},{request.pickXyz[1]:.4f},{request.pickXyz[2]:.4f}"
            )
            on_row({"kind": "needs_operator", "trial": trial_index, "message": message, "at": time.time()})
            if wait_for_operator is None or not wait_for_operator(message):
                on_row({"kind": "operator", "answer": "gone", "at": time.time()})
                return False
            on_row({"kind": "operator", "answer": "continued", "at": time.time()})
            stage_from_fixture(robot, request, rotvec, request_id=f"{request.requestId}_restage")
            check = scripted_grasp(robot, request, verify_aim, rotvec, request_id=f"{request.requestId}_verify")
            if check["verdict"] == "held":
                return True
            reason = f"restage {attempt + 1} did not leave a graspable peg at the spot"
        raise GraspEnvelopeHalt("restage_failed")

    try:
        if request.start == "fixture":
            stage_from_fixture(robot, request, rotvec, request_id=f"{request.requestId}_stage")
        first = scripted_grasp(robot, request, verify_aim, rotvec, request_id=f"{request.requestId}_verify")
        on_row({"kind": "staged", "start": request.start, "verify": first})
        if first["verdict"] != "held" and not restage("no graspable peg at the spot before the first trial", None):
            raise GraspEnvelopeHalt("peg_lost")

        for point in schedule:
            if should_stop():
                halted = "stop_requested"
                break
            if request.maxSeconds and clock() - started >= request.maxSeconds:
                halted = "max_seconds"
                break
            trial_started = clock()
            request_id = f"{request.requestId}_{point.index:03d}"
            print(
                f"[INFO] grasp_envelope_trial_start index={point.index} block={point.block} "
                f"d_mm={point.dxMm:g},{point.dyMm:g},{point.dzMm:g}",
                flush=True,
            )
            grasp = scripted_grasp(robot, request, point.aimXyz, rotvec, request_id=request_id)
            verify = scripted_grasp(robot, request, verify_aim, rotvec, request_id=f"{request_id}_verify")
            disturbed = verify["verdict"] != "held"
            close = grasp.get("closeXyz")
            row = {
                "kind": "trial",
                "index": point.index,
                "trialKind": point.block,
                "block": point.block,
                "dxMm": point.dxMm,
                "dyMm": point.dyMm,
                "dzMm": point.dzMm,
                "aimXyz": list(point.aimXyz),
                **grasp,
                # Where the TCP measurably was when the fingers closed, relative to the peg
                # reference -- the commanded offset plus the arm's dead-band, which is what the
                # fingers actually experienced.
                "closeOffsetMm": None
                if close is None
                else [round((close[i] - spot[i]) * 1000.0, 2) for i in range(3)],
                "finalTcpXyz": close,
                "success": grasp["verdict"] == "held",
                "emptyGrasp": grasp["verdict"] == "empty",
                "contactBeforeClose": grasp["verdict"] == "contact",
                "pegDisturbed": disturbed,
                "verifyVerdict": verify["verdict"],
                "verifyWidth": verify.get("widthLifted"),
            }
            restaged = False
            if disturbed:
                if not restage("the verify grasp found no peg at the spot", point.index):
                    row["restaged"] = False
                    row["trialS"] = round(clock() - trial_started, 1)
                    on_row(row)
                    rows.append(row)
                    halted = "peg_lost"
                    break
                restaged = True
            row["restaged"] = restaged
            row["trialS"] = round(clock() - trial_started, 1)
            on_row(row)
            rows.append(row)
            print(
                f"[INFO] grasp_envelope_trial index={point.index} verdict={row['verdict']} "
                f"disturbed={disturbed} width_lifted={row.get('widthLifted')} trial_s={row['trialS']}",
                flush=True,
            )
    except KeyboardInterrupt:
        halted = "interrupted"
    except GraspEnvelopeHalt as exc:
        halted = str(exc)
    except Exception as exc:  # noqa: BLE001 - a motion fault ends the run, and must say so
        halted = f"motion_fault: {type(exc).__name__}: {exc}"

    parked = False
    if halted in ("", "stop_requested", "max_seconds", "peg_lost"):
        # A clean boundary: the fingers are open above the table. Home.
        try:
            _move_to_start(robot)
            parked = True
        except Exception as exc:  # noqa: BLE001 - the summary has to survive a failed park
            print(f"[WARN] grasp_envelope=park_failed details={exc}", flush=True)
    else:
        # Interrupted or faulted mid-motion: the fingers may be holding the peg. Hold still and
        # leave the decision to a person rather than open over whatever is below.
        try:
            _xyz, _rotvec, gripper_now = _observation_xyz_rotvec_gripper(robot)
            _hold_where_it_is(robot, gripper_now)
        except Exception:  # noqa: BLE001
            pass
    # A resumed run is one sweep in two files; its numbers are the sweep's, not the second half's.
    prior_trials = [r for r in prior_rows if r.get("kind") == "trial"]
    summary = {**summarize_envelope([*prior_trials, *rows], request), "trialsThisRun": sum(1 for r in rows if r.get("kind") == "trial")}
    ok = halted in ("", "stop_requested", "max_seconds")
    on_row({"kind": "summary", "ok": ok, "haltedOn": halted or "schedule_complete", "parked": parked, **summary})
    print(
        f"[INFO] grasp_envelope=done halted_on={halted or 'schedule_complete'} trials={summary['trials']} "
        f"capture_mm={summary['xyCaptureRadiusMm']} dz_interval_mm={summary['graspDzIntervalMm']} "
        f"knock_mm={summary['knockOverRadiusMm']}",
        flush=True,
    )
    return {"ok": ok, "haltedOn": halted or "schedule_complete", **summary}
