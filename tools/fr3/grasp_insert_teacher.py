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

"""The grasp loop's insertion teacher: lift, carry, insert and let go, inside the policy's loop.

Why it exists (10-30): the goal is a policy that inserts too, and nothing it was trained on goes
past the close. Arm B's funnel ends at the grasp, and the scripted insertion that follows runs
in the reset's own step loop, where no frame is recorded -- so a whole-trial episode stops ten
frames after the close. This teacher runs the insertion as the funnel runs the grasp: one
command per control step, through the runtime's smoothing and clamp, recorded as an expert.

What it knows that the policy has to read off the wrist camera is where the peg sits in the
fingers. The fingers close along the tool's y axis and centre the peg on it; along the finger
face the peg stays where the close found it, so the tool is off the peg by the close's measured
pose minus the staged peg's, that axis removed (`in_hand_offset_xy`). 362 held trials 10-23..30:
that offset is plain in the pre-insert wrist image (r = 0.92 against one crude feature), and the
scripted insertion, which ignores it, went in for 230/233 grasps within [-4, 0) mm and 7/43 past
+2 mm. The teacher aims the tool at the hole plus the offset, so an off-centre grasp gets a
demonstration of the insertion it needs, not of the one a centred grasp needs.

Phases: LIFT straight up from the close, grading the grasp on the way as `check_grasp` does
(the median width once `checkLiftM` up, against the policy's own close command) and only then
squeezing to the script's close; CARRY across at carry height, turning to the level wrist;
DESCEND fast to `approachDzM` over the seat, then slowly, the setpoint never more than
`contactLagM` below the tool, until the tool stands still; RELEASE if it stands within `seatedM`
of the seat, else stop holding (`blocked`); RETREAT straight up with the fingers open.

No lift-and-retry at the bottom: the funnel's lift labels taught the policy a bimodal bottom
(10-30 unguided A/B), and a blocked insertion is handed back to the loop's scripted search.

Pure: no robot, no clock. The runtime feeds the measured pose and width and sends what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Callable

import numpy as np

from lerobot.utils.rotation import Rotation

LIFT = "LIFT"
CARRY = "CARRY"
DESCEND = "DESCEND"
RELEASE = "RELEASE"
RETREAT = "RETREAT"
DONE = "DONE"

# The funnel's align speed for the lift, and a carry a little faster: the arm trails a 0.05 m/s
# sideways setpoint by up to 9 mm (grasp_funnel._off_path_mm), well inside the runtime's 20 mm leash.
INSERT_TEACHER_LIFT_SPEED_M_S = 0.05
INSERT_TEACHER_CARRY_SPEED_M_S = 0.06
INSERT_TEACHER_DESCEND_SPEED_M_S = 0.05
# The terminal servo's descent speed, over the last `INSERT_TEACHER_APPROACH_DZ_M`.
INSERT_TEACHER_SEAT_SPEED_M_S = 0.02
INSERT_TEACHER_APPROACH_DZ_M = 0.015
# The servo's numbers: a contact lag of 4 mm, a seat within 3 mm (TERMINAL_SERVO_CONTACT_LAG_M,
# TERMINAL_SERVO_SEARCH_SEATED_M), and nothing commanded below its floor.
INSERT_TEACHER_CONTACT_LAG_M = 0.004
INSERT_TEACHER_SEATED_M = 0.003
INSERT_TEACHER_MIN_Z_M = 0.030
# How far below the seat the slow descent's setpoint goes: a seated peg stands on the hole's
# floor, and a setpoint that stopped at the seat would leave the last millimetre to the dead band.
INSERT_TEACHER_PRESS_M = 0.002
INSERT_TEACHER_SETTLE_SPEED_MM_S = 2.0
INSERT_TEACHER_SETTLE_STEPS = 5
# Width samples for the grade, as `read_width` takes them, but read while still rising.
INSERT_TEACHER_WIDTH_SAMPLES = 10
# Half a second open before the retreat: the servo's openSettleS is 0.8 s, which the dataset's
# dead-time rule would trim to 17 frames anyway.
INSERT_TEACHER_OPEN_STEPS = 15
INSERT_TEACHER_RETREAT_M = 0.05


def in_hand_offset_xy(
    close_xyz: tuple[float, float, float],
    peg_xyz: tuple[float, float, float],
    close_rotvec: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Tool minus peg in the base XY plane, once the fingers have centred the peg between them.

    The component along the fingers' closing axis (the tool's y) is removed: the close takes it
    out. Without the wrist's orientation the axis is taken as the base y, which is where the
    level wrist this rig grasps with puts it.
    """

    dx, dy = close_xyz[0] - peg_xyz[0], close_xyz[1] - peg_xyz[1]
    if close_rotvec is None:
        axis = (0.0, 1.0)
    else:
        tool_y = Rotation.from_rotvec(np.asarray(close_rotvec, dtype=np.float64)).as_matrix()[:, 1]
        norm = math.hypot(float(tool_y[0]), float(tool_y[1]))
        axis = (0.0, 1.0) if norm < 1e-6 else (float(tool_y[0]) / norm, float(tool_y[1]) / norm)
    along = dx * axis[0] + dy * axis[1]
    return (dx - along * axis[0], dy - along * axis[1])


def _walk(
    setpoint: tuple[float, float, float], goal: tuple[float, float, float], step_m: float
) -> tuple[tuple[float, float, float], bool]:
    delta = [goal[k] - setpoint[k] for k in range(3)]
    distance = math.sqrt(sum(d * d for d in delta))
    if distance <= step_m:
        return tuple(goal), True  # type: ignore[return-value]
    scale = step_m / distance
    return tuple(setpoint[k] + delta[k] * scale for k in range(3)), False  # type: ignore[return-value]


def _slerp_rotvec(
    start: tuple[float, float, float], end: tuple[float, float, float], fraction: float
) -> tuple[float, float, float]:
    fraction = min(1.0, max(0.0, fraction))
    r0 = Rotation.from_rotvec(np.asarray(start, dtype=np.float64))
    r1 = Rotation.from_rotvec(np.asarray(end, dtype=np.float64))
    turn = (r0.inv() * r1).as_rotvec()
    out = (r0 * Rotation.from_rotvec(turn * fraction)).as_rotvec()
    return (float(out[0]), float(out[1]), float(out[2]))


@dataclass(frozen=True)
class InsertTeacherConfig:
    # Measured at the close, and where the peg was staged: what the in-hand offset is read from.
    closeXyz: tuple[float, float, float]
    closeRotvec: tuple[float, float, float] | None
    pegXyz: tuple[float, float, float]
    # The tool pose that seats a peg gripped centred at `regripZ`: the grasp loop's insertion
    # pose, or its estimate of the hole from earlier insertions (`teacher_hole_xy`).
    holeXyz: tuple[float, float, float]
    regripZ: float
    carryZ: float
    levelRotvec: tuple[float, float, float] | None = None
    # The policy's own close, held until the grade, then the script's (insert_held_peg's reason:
    # a peg pressed in a weaker grip slides up the fingers).
    policyGripper: float = 0.0
    heldGripper: float = 0.0
    openGripper: float = 1.0
    checkLiftM: float = 0.03
    controlPeriodS: float = 1.0 / 30.0
    liftSpeedMS: float = INSERT_TEACHER_LIFT_SPEED_M_S
    carrySpeedMS: float = INSERT_TEACHER_CARRY_SPEED_M_S
    descendSpeedMS: float = INSERT_TEACHER_DESCEND_SPEED_M_S
    seatSpeedMS: float = INSERT_TEACHER_SEAT_SPEED_M_S
    approachDzM: float = INSERT_TEACHER_APPROACH_DZ_M
    contactLagM: float = INSERT_TEACHER_CONTACT_LAG_M
    seatedM: float = INSERT_TEACHER_SEATED_M
    pressM: float = INSERT_TEACHER_PRESS_M
    minZ: float = INSERT_TEACHER_MIN_Z_M
    settleSpeedMmS: float = INSERT_TEACHER_SETTLE_SPEED_MM_S
    settleSteps: int = INSERT_TEACHER_SETTLE_STEPS
    widthSamples: int = INSERT_TEACHER_WIDTH_SAMPLES
    openSteps: int = INSERT_TEACHER_OPEN_STEPS
    retreatM: float = INSERT_TEACHER_RETREAT_M

    @property
    def offsetXy(self) -> tuple[float, float]:
        return in_hand_offset_xy(self.closeXyz, self.pegXyz, self.closeRotvec)

    @property
    def raisedM(self) -> float:
        """How much higher up the peg the fingers closed than the script's grip (insert_held_peg)."""

        return max(0.0, self.closeXyz[2] - self.regripZ)

    @property
    def seatXyz(self) -> tuple[float, float, float]:
        """Where the tool stands with this grasp's peg seated: the hole plus the in-hand offset."""

        ox, oy = self.offsetXy
        return (self.holeXyz[0] + ox, self.holeXyz[1] + oy, self.holeXyz[2] + self.raisedM)


@dataclass
class InsertTeacher:
    """One trial's insertion. `step` once per control step from the close on; `record` at the end."""

    config: InsertTeacherConfig
    # The grade of the grasp: (median width, the policy's gripper command) -> held.
    heldTest: Callable[[float, float], bool]
    state: str = LIFT
    setpoint: tuple[float, float, float] | None = None
    rotvec: tuple[float, float, float] | None = None
    gripper: float | None = None
    previousXyz: tuple[float, float, float] | None = None
    quietSteps: int = 0
    widths: list[float] = field(default_factory=list)
    # None until graded; then the median width and the verdict.
    widthLifted: float | None = None
    held: bool | None = None
    # "empty" (graded empty on the lift), "seated" (let go of in the hole), "blocked" (stood on
    # something above the seat, still held). None while it runs.
    outcome: str | None = None
    carryStartXyz: tuple[float, float, float] | None = None
    carryStartRotvec: tuple[float, float, float] | None = None
    stoppedXyz: tuple[float, float, float] | None = None
    releaseStepCount: int = 0
    stateSteps: dict[str, int] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state != DONE

    @property
    def done(self) -> bool:
        return self.state == DONE

    def _enter(self, state: str, step_idx: int, ee_xyz: tuple[float, float, float], reason: str) -> None:
        self.transitions.append(
            {"step": step_idx, "from": self.state, "to": state, "reason": reason, "eeXyz": [round(v, 5) for v in ee_xyz]}
        )
        self.state = state
        self.quietSteps = 0

    def _settled(self, xyz: tuple[float, float, float]) -> bool:
        cfg = self.config
        speed = math.inf if self.previousXyz is None else math.dist(xyz, self.previousXyz) * 1000.0 / cfg.controlPeriodS
        self.quietSteps = self.quietSteps + 1 if speed < cfg.settleSpeedMmS else 0
        return self.quietSteps >= cfg.settleSteps

    def step(
        self,
        step_idx: int,
        ee_xyz: tuple[float, float, float],
        ee_rotvec: tuple[float, float, float],
        width: float,
    ) -> dict[str, float]:
        """The command for this step, from the measured pose and the measured finger width."""

        cfg = self.config
        ee_xyz = tuple(float(v) for v in ee_xyz)  # type: ignore[assignment]
        if self.setpoint is None:
            # Taken at the measured pose: the arm has stood closing for the handover's settle steps.
            self.setpoint = ee_xyz
            self.rotvec = tuple(float(v) for v in ee_rotvec)  # type: ignore[assignment]
            self.gripper = cfg.policyGripper
        assert self.rotvec is not None and self.gripper is not None
        self.stateSteps[self.state] = self.stateSteps.get(self.state, 0) + 1
        seat = cfg.seatXyz
        carry_z = max(cfg.carryZ, cfg.closeXyz[2] + cfg.checkLiftM)

        if self.state == LIFT:
            self.setpoint, arrived = _walk(
                self.setpoint, (self.setpoint[0], self.setpoint[1], carry_z), cfg.liftSpeedMS * cfg.controlPeriodS
            )
            # Read once the setpoint is `checkLiftM` up, not the tool: the arm parks a few mm short
            # of a setpoint and would never get there if the carry height were that same height.
            if self.held is None and self.setpoint[2] >= cfg.closeXyz[2] + cfg.checkLiftM - 1e-9:
                self.widths.append(float(width))
                if len(self.widths) >= cfg.widthSamples:
                    self.widthLifted = float(statistics.median(self.widths))
                    self.held = bool(self.heldTest(self.widthLifted, cfg.policyGripper))
                    if not self.held:
                        self.outcome = "empty"
                        self._enter(DONE, step_idx, ee_xyz, "graded_empty")
                    else:
                        self.gripper = cfg.heldGripper
            if self.state == LIFT and self.held and arrived:
                self.carryStartXyz = self.setpoint
                self.carryStartRotvec = self.rotvec
                self._enter(CARRY, step_idx, ee_xyz, "at_carry_height")
        elif self.state == CARRY:
            assert self.carryStartXyz is not None and self.carryStartRotvec is not None
            goal = (seat[0], seat[1], carry_z)
            self.setpoint, arrived = _walk(self.setpoint, goal, cfg.carrySpeedMS * cfg.controlPeriodS)
            if cfg.levelRotvec is not None:
                total = math.dist(self.carryStartXyz[:2], goal[:2])
                fraction = 1.0 if total < 1e-6 else 1.0 - math.dist(self.setpoint[:2], goal[:2]) / total
                self.rotvec = _slerp_rotvec(self.carryStartRotvec, cfg.levelRotvec, fraction)
            if arrived and self._settled(ee_xyz):
                self._enter(DESCEND, step_idx, ee_xyz, "over_seat")
        elif self.state == DESCEND:
            bottom = max(cfg.minZ, seat[2] - cfg.pressM)
            slow_from = seat[2] + cfg.approachDzM
            fast = self.setpoint[2] > slow_from + 1e-9
            speed = cfg.descendSpeedMS if fast else cfg.seatSpeedMS
            walked_z = max(slow_from if fast else bottom, self.setpoint[2] - speed * cfg.controlPeriodS)
            # Never further below the tool than the contact lag: whatever it stands on is pressed
            # with the controller's stiffness times this, not with a setpoint running away below.
            next_z = max(walked_z, ee_xyz[2] - cfg.contactLagM)
            self.setpoint = (seat[0], seat[1], next_z)
            settled = self._settled(ee_xyz)
            if settled and (next_z <= bottom + 1e-9 or next_z > walked_z + 1e-9):
                # At the bottom of its travel, or held back by the lag, and the tool has stopped.
                self.stoppedXyz = ee_xyz
                if ee_xyz[2] - seat[2] <= cfg.seatedM:
                    self.gripper = cfg.openGripper
                    self._enter(RELEASE, step_idx, ee_xyz, "seated")
                else:
                    self.outcome = "blocked"
                    # Hold where it stands rather than keep pressing.
                    self.setpoint = ee_xyz
                    self._enter(DONE, step_idx, ee_xyz, "stood_above_seat")
        elif self.state == RELEASE:
            self.releaseStepCount += 1
            if self.releaseStepCount >= cfg.openSteps:
                self._enter(RETREAT, step_idx, ee_xyz, "opened")
        elif self.state == RETREAT:
            assert self.stoppedXyz is not None
            goal = (self.setpoint[0], self.setpoint[1], self.stoppedXyz[2] + cfg.retreatM)
            self.setpoint, arrived = _walk(self.setpoint, goal, cfg.liftSpeedMS * cfg.controlPeriodS)
            if arrived:
                self.outcome = "seated"
                self._enter(DONE, step_idx, ee_xyz, "retreated")

        self.previousXyz = ee_xyz
        return {
            "ee.x": float(self.setpoint[0]),
            "ee.y": float(self.setpoint[1]),
            "ee.z": float(self.setpoint[2]),
            "ee.wx": float(self.rotvec[0]),
            "ee.wy": float(self.rotvec[1]),
            "ee.wz": float(self.rotvec[2]),
            "gripper.pos": float(self.gripper),
        }

    def record(self) -> dict[str, Any]:
        cfg = self.config
        seat = cfg.seatXyz
        return {
            "state": self.state,
            "outcome": self.outcome,
            "held": self.held,
            "widthLifted": None if self.widthLifted is None else round(self.widthLifted, 4),
            "offsetXyMm": [round(v * 1000.0, 1) for v in cfg.offsetXy],
            "holeXyz": [round(v, 5) for v in cfg.holeXyz],
            "seatXyz": [round(v, 5) for v in seat],
            "raisedMm": round(cfg.raisedM * 1000.0, 1),
            "stoppedXyz": None if self.stoppedXyz is None else [round(v, 5) for v in self.stoppedXyz],
            "aboveSeatMm": None if self.stoppedXyz is None else round((self.stoppedXyz[2] - seat[2]) * 1000.0, 1),
            "stateSteps": dict(self.stateSteps),
            "transitions": list(self.transitions),
        }
