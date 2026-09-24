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

"""The GT grasp funnel (roadmap v14 step 2, arm B): the policy approaches, the funnel grasps.

Why it exists: on 09-22 the policy's closures sat a median 31 mm off the peg in XY, and ten of
thirty-four were 20 mm or more too high -- while the P0 sweep of 09-23 grasped 53/54 inside
+-12 mm XY at dz = -6 when the *script* placed the fingers. The grasp itself is not the problem;
getting the fingers into the basin is. So the funnel takes the arm the moment the policy has
brought it near the peg, and puts the fingers where P0 measured the grasp to work:

    SEARCH   the policy drives. Its gripper command is not executed -- the funnel owns the close.
             Ends when the arm comes down to `alignZ` or the policy asks to close, whichever first.
    ALIGN    across to the peg's XY at the height the arm is at (straight up first if it is below
             `alignZ` and outside the capture radius, so nothing moves sideways at peg height).
             Ends when the setpoint has arrived, the measured XY error is inside `captureXyMm`,
             and the arm has stopped.
    DESCEND  straight down to `pegXyz.z + closeDzMm`. Back to ALIGN if the XY error ever leaves
             the capture radius. Ends when the setpoint has arrived and the arm has stopped, not
             when the measured z reaches the target: this arm stops 1.3-2 mm short (P0).
    CLOSE    the fingers close where the arm stands. `GraspHandover` sees the closed command and
             takes the arm off the policy after its settle steps; the lift that grades the grasp
             is the grasp loop's, the same one arm A is graded by.

Thresholds are v14 (3): capture 8 mm (the 12 mm grid measured at ~100%, with 1.5x margin), close
at dz = -6 (the demonstrations' 51.4 mm), settle below 2 mm/s. The peg position is the reset's
own `resetTarget` (`peg_source = "gt"`), which P0's 60/63 verify grasps say is where the peg is.

Every step is recorded against the v13 (7) schema -- what the policy asked for next to what was
executed -- because those pairs are the correction data the roadmap's last phase trains on.

Pure: no robot, no clock. The runtime feeds the observation and the policy's command, and sends
what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

SEARCH = "SEARCH"
ALIGN = "ALIGN"
DESCEND = "DESCEND"
CLOSE = "CLOSE"

# v14 (3). Each is argued there; restated here only as the number.
FUNNEL_CAPTURE_XY_MM = 8.0
FUNNEL_CLOSE_DZ_MM = -6.0
FUNNEL_SETTLE_SPEED_MM_S = 2.0
# Where the funnel takes over, above the peg's grasp height. P0 put the fingertips on the peg's
# top at dz = +20 and clear of it at +28, so +60 leaves 3 cm between the fingers and the peg for
# the sideways move -- and it is below where the policy has done its gross positioning.
FUNNEL_ALIGN_DZ_MM = 60.0
# Setpoint speeds. The lateral one is a third of the reset's 0.15 m/s because this move ends
# above a standing peg; the descent is the terminal servo's 0.02 m/s, the speed this rig already
# descends onto things at.
FUNNEL_ALIGN_SPEED_M_S = 0.05
FUNNEL_DESCEND_SPEED_M_S = 0.02
# Steps the settle test must hold, so one quiet frame between two moves is not a stop.
FUNNEL_SETTLE_STEPS = 5


@dataclass(frozen=True)
class FunnelConfig:
    pegXyz: tuple[float, float, float]
    pegSource: str = "gt"
    captureXyMm: float = FUNNEL_CAPTURE_XY_MM
    closeDzMm: float = FUNNEL_CLOSE_DZ_MM
    alignDzMm: float = FUNNEL_ALIGN_DZ_MM
    settleSpeedMmS: float = FUNNEL_SETTLE_SPEED_MM_S
    settleSteps: int = FUNNEL_SETTLE_STEPS
    alignSpeedMS: float = FUNNEL_ALIGN_SPEED_M_S
    descendSpeedMS: float = FUNNEL_DESCEND_SPEED_M_S
    closedBelow: float = 0.5
    openGripper: float = 1.0
    closedGripper: float = 0.0
    controlPeriodS: float = 1.0 / 30.0

    @property
    def alignZ(self) -> float:
        return self.pegXyz[2] + self.alignDzMm / 1000.0

    @property
    def closeXyz(self) -> tuple[float, float, float]:
        return (self.pegXyz[0], self.pegXyz[1], self.pegXyz[2] + self.closeDzMm / 1000.0)


def _xy_error_mm(xyz: tuple[float, float, float], peg: tuple[float, float, float]) -> float:
    return math.hypot(xyz[0] - peg[0], xyz[1] - peg[1]) * 1000.0


def _walk(
    setpoint: tuple[float, float, float], goal: tuple[float, float, float], step_m: float
) -> tuple[tuple[float, float, float], bool]:
    """The setpoint moved at most `step_m` toward `goal`, and whether it has arrived."""

    delta = [goal[k] - setpoint[k] for k in range(3)]
    distance = math.sqrt(sum(d * d for d in delta))
    if distance <= step_m:
        return tuple(goal), True  # type: ignore[return-value]
    scale = step_m / distance
    return tuple(setpoint[k] + delta[k] * scale for k in range(3)), False  # type: ignore[return-value]


@dataclass
class GraspFunnel:
    """One trial's funnel. `step` once per control step; `trial_record` once at the end."""

    config: FunnelConfig
    state: str = SEARCH
    setpoint: tuple[float, float, float] | None = None
    rotvec: tuple[float, float, float] | None = None
    previousXyz: tuple[float, float, float] | None = None
    quietSteps: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Per-trial readings (v13 (7) second row).
    entryStep: int | None = None
    entryReason: str = ""
    entryXyz: tuple[float, float, float] | None = None
    xyErrorAtEntryMm: float | None = None
    descendStartStep: int | None = None
    xyErrorAtDescendStartMm: float | None = None
    closeStep: int | None = None
    closeXyz: tuple[float, float, float] | None = None
    xyErrorAtCloseMm: float | None = None
    prematureDescend: bool = False
    blockedSteps: int = 0
    realigns: int = 0
    maxResidualMm: float = 0.0

    def _speed_mm_s(self, xyz: tuple[float, float, float]) -> float:
        if self.previousXyz is None:
            return math.inf
        return math.dist(xyz, self.previousXyz) * 1000.0 / self.config.controlPeriodS

    def _settled(self, xyz: tuple[float, float, float]) -> bool:
        if self._speed_mm_s(xyz) < self.config.settleSpeedMmS:
            self.quietSteps += 1
        else:
            self.quietSteps = 0
        return self.quietSteps >= self.config.settleSteps

    def _enter(self, state: str) -> None:
        self.state = state
        self.quietSteps = 0

    def step(
        self,
        step_idx: int,
        ee_xyz: tuple[float, float, float],
        ee_rotvec: tuple[float, float, float],
        policy_command: dict[str, float],
        *,
        policy_gripper_raw: float | None = None,
    ) -> dict[str, float]:
        """The command to execute this step, given the measured pose and the policy's command.

        `policy_command` and the return value are the runtime's absolute command dict
        (`ee.x/y/z`, `ee.wx/wy/wz`, `gripper.pos`) in the frame the observation is in.
        """

        cfg = self.config
        peg = cfg.pegXyz
        xy_err = _xy_error_mm(ee_xyz, peg)
        policy_xyz = (float(policy_command["ee.x"]), float(policy_command["ee.y"]), float(policy_command["ee.z"]))
        policy_gripper = float(policy_command["gripper.pos"])
        transition = ""
        descend_blocked = False

        if self.state == SEARCH:
            policy_closes = policy_gripper < cfg.closedBelow
            reached = ee_xyz[2] <= cfg.alignZ
            if reached or policy_closes:
                transition = "policy_closed" if policy_closes and not reached else "reached_align_z"
                self.entryStep, self.entryReason = step_idx, transition
                self.entryXyz = tuple(ee_xyz)  # type: ignore[assignment]
                self.xyErrorAtEntryMm = round(xy_err, 1)
                # The policy was about to go below the capture zone with the fingers outside it:
                # the reading the funnel's one-line constraint exists for.
                self.prematureDescend = policy_xyz[2] < cfg.alignZ and xy_err > cfg.captureXyMm
                self.setpoint = tuple(ee_xyz)  # type: ignore[assignment]
                # The wrist stays as the policy left it: P0 says the grasp does not need it
                # changed, and turning it above the peg would move the fingertips as well.
                self.rotvec = tuple(ee_rotvec)  # type: ignore[assignment]
                self._enter(ALIGN)

        executed_gripper = cfg.openGripper
        if self.state == SEARCH:
            executed_xyz = policy_xyz
            executed_rotvec = (
                float(policy_command["ee.wx"]),
                float(policy_command["ee.wy"]),
                float(policy_command["ee.wz"]),
            )
        else:
            assert self.setpoint is not None and self.rotvec is not None
            if self.state == ALIGN:
                below = ee_xyz[2] < cfg.alignZ - 0.002
                if xy_err > cfg.captureXyMm and below:
                    # Up first, never sideways at peg height.
                    goal = (self.setpoint[0], self.setpoint[1], cfg.alignZ)
                else:
                    goal = (peg[0], peg[1], self.setpoint[2])
                self.setpoint, arrived = _walk(self.setpoint, goal, cfg.alignSpeedMS * cfg.controlPeriodS)
                settled = self._settled(ee_xyz)
                if arrived and goal[:2] == peg[:2] and xy_err <= cfg.captureXyMm and settled:
                    transition = "aligned"
                    self.descendStartStep = step_idx
                    self.xyErrorAtDescendStartMm = round(xy_err, 1)
                    self._enter(DESCEND)
                elif policy_xyz[2] < ee_xyz[2] - 1e-4:
                    descend_blocked = True
            elif self.state == DESCEND:
                if xy_err > cfg.captureXyMm:
                    transition = "left_capture"
                    self.realigns += 1
                    self._enter(ALIGN)
                else:
                    self.setpoint, arrived = _walk(
                        self.setpoint, cfg.closeXyz, cfg.descendSpeedMS * cfg.controlPeriodS
                    )
                    if arrived and self._settled(ee_xyz):
                        transition = "settled_at_close_height"
                        self.closeStep = step_idx
                        self.closeXyz = tuple(ee_xyz)  # type: ignore[assignment]
                        self.xyErrorAtCloseMm = round(xy_err, 1)
                        self._enter(CLOSE)
            if self.state == CLOSE:
                executed_gripper = cfg.closedGripper
            executed_xyz = self.setpoint
            executed_rotvec = self.rotvec

        if descend_blocked:
            self.blockedSteps += 1
        residual_xy = math.hypot(executed_xyz[0] - policy_xyz[0], executed_xyz[1] - policy_xyz[1]) * 1000.0
        if self.state != SEARCH:
            self.maxResidualMm = max(self.maxResidualMm, residual_xy)
        self.previousXyz = tuple(ee_xyz)  # type: ignore[assignment]

        executed = {
            "ee.x": float(executed_xyz[0]),
            "ee.y": float(executed_xyz[1]),
            "ee.z": float(executed_xyz[2]),
            "ee.wx": float(executed_rotvec[0]),
            "ee.wy": float(executed_rotvec[1]),
            "ee.wz": float(executed_rotvec[2]),
            "gripper.pos": float(executed_gripper),
        }
        self.steps.append(
            {
                "step": step_idx,
                "funnel_state": self.state,
                "transition_reason": transition,
                "policy_raw_action": [round(v, 5) for v in (*policy_xyz, policy_gripper)],
                "executed_action": [round(v, 5) for v in (*executed_xyz, executed_gripper)],
                "servo_residual_xy": round(residual_xy, 2),
                "peg_xy": [round(peg[0], 5), round(peg[1], 5)],
                "peg_source": cfg.pegSource,
                "ee_xyz": [round(v, 5) for v in ee_xyz],
                "xy_error_mm": round(xy_err, 2),
                "policy_gripper_raw": None if policy_gripper_raw is None else round(float(policy_gripper_raw), 4),
                "executed_gripper": executed_gripper,
                "descend_blocked": descend_blocked,
                "grasp_triggered": transition == "settled_at_close_height",
            }
        )
        return executed

    @property
    def active(self) -> bool:
        """True once the funnel, not the policy, is driving the arm."""

        return self.state != SEARCH

    def trial_record(self) -> dict[str, Any]:
        cfg = self.config
        close_dz = None if self.closeXyz is None else round((self.closeXyz[2] - cfg.pegXyz[2]) * 1000.0, 1)
        return {
            "funnelState": self.state,
            "funnelEntryStep": self.entryStep,
            "funnelEntryReason": self.entryReason,
            "funnelEntryXyz": None if self.entryXyz is None else [round(v, 5) for v in self.entryXyz],
            "xyErrorAtEntryMm": self.xyErrorAtEntryMm,
            "descendStartStep": self.descendStartStep,
            "xyErrorAtDescendStartMm": self.xyErrorAtDescendStartMm,
            "funnelCloseStep": self.closeStep,
            "xyErrorAtCloseMm": self.xyErrorAtCloseMm,
            "funnelCloseDzMm": close_dz,
            "prematureDescend": self.prematureDescend,
            "blockedSteps": self.blockedSteps,
            "realigns": self.realigns,
            "maxResidualMm": round(self.maxResidualMm, 1),
            "funnelRotvec": None if self.rotvec is None else [round(v, 5) for v in self.rotvec],
            "pegSource": cfg.pegSource,
        }
