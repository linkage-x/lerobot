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
    DESCEND  straight down to `pegXyz.z + closeDzMm`: at `fastDescendSpeedMS` while the setpoint
             is more than `slowBelowDzMm` above the peg, then at `descendSpeedMS`. Back to ALIGN
             if the XY error ever leaves the capture radius. Ends when the setpoint has arrived
             and the arm has stopped, not when the measured z reaches the target: this arm stops
             1.3-2 mm short (P0).
    CLOSE    the fingers close where the arm stands. `GraspHandover` sees the closed command and
             takes the arm off the policy after its settle steps; the lift that grades the grasp
             is the grasp loop's, the same one arm A is graded by.

Corrective demonstrations (10-20, `residualMm` / `liftDzMm` / `settleBeforeDescend`, all off by
default): checkpoints trained on the funnel above aim once, high, and descend straight -- run
105703 closed exactly as far off the peg as it was at 60 mm, because in 150 recorded trials the
funnel took over at a median 190 mm, removed the whole 52 mm error there and never moved sideways
below. So ALIGN may leave up to `residualMm` of the policy's own error (its direction, never an
invented one) and DESCEND closes it on a straight line that reaches the peg's XY at `centredDzMm`,
above the fingertips' +28 mm clearance; a takeover below `liftDzMm` outside the capture radius
goes up to it first and realigns fully, which is the recovery from a low, off-centre approach.
Below `liftDzMm` a policy that stops (it closes when it stops, copying the settle) is taken over
as well, and so is one the arm cannot follow (`lowContactMm`): the fingers are on something, and
pushing on would tip the peg.

Where the funnel takes over is `alignDzMm` above the peg. At the v14 (3) 60 mm the policy has
already done its aiming: on 09-30 (344 B trials) it arrived there a median 41 mm off the peg,
with its y aim only 0.66 of the peg's (regression to the training mean), and 82% of its lateral
travel had happened above 150 mm. The grasp loop can therefore draw the height per trial, so the
recorded corrections start from every stage of the approach, not only the last 6 cm.

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
# above a standing peg. The descent was the terminal servo's 0.02 m/s until 10-20; the funnel has
# no contact stop for that speed to protect, and checkpoints trained on its trials copy the slow
# last 66 mm, closing at a median step ~470, past arm A's old 450 budget.
FUNNEL_ALIGN_SPEED_M_S = 0.05
# How far below `alignZ` the *measured* tool may sit and still move across. The arm parks up to
# ~2.4 mm short along its direction of travel and does not close that with time (scene_reset,
# 09-11), so "up first" judged against the measurement alone can wait forever: 09-23 trial 9 and
# 09-28 trials 9 and 19 entered ALIGN just under alignZ, the setpoint reached it, the arm stopped
# at alignZ - 2.4 mm, and the funnel held that pose for its whole 360-step budget.
FUNNEL_ALIGN_Z_SLACK_M = 0.006
FUNNEL_DESCEND_SPEED_M_S = 0.03
# Down to `FUNNEL_SLOW_BELOW_DZ_MM` above the peg, from a takeover higher than that: the align
# speed, 1.7 mm a step at 30 Hz against the demonstrations' median 1.6. Below it the slow
# descend speed, so a 60 mm takeover descends at that speed throughout.
FUNNEL_FAST_DESCEND_SPEED_M_S = 0.05
FUNNEL_SLOW_BELOW_DZ_MM = 60.0
# Steps the settle test must hold, so one quiet frame between two moves is not a stop.
FUNNEL_SETTLE_STEPS = 5
# Where a corrective descent must be over the peg's XY: 7 mm above P0's +28 mm fingertip
# clearance, so nothing moves sideways at the peg top.
FUNNEL_CENTRED_DZ_MM = 35.0
# Below `liftDzMm`, how far the arm may stand from the command it was given before the policy is
# taken off it: the fingers are on something. Smoke run 151637 (10-20): the four late trials
# trailed by at most 2.7 mm; in trial 0 a fingertip came down on something at +20 mm, the arm
# stood still in z while the command went on down, slid 22 mm sideways for 35 steps, and the peg
# was knocked over. This threshold would have taken over 6 steps into that contact.
FUNNEL_LOW_CONTACT_MM = 6.0


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
    fastDescendSpeedMS: float = FUNNEL_FAST_DESCEND_SPEED_M_S
    slowBelowDzMm: float = FUNNEL_SLOW_BELOW_DZ_MM
    # Corrective demonstrations (module docstring). 0 / None / True are the v14 (3) funnel.
    residualMm: float = 0.0
    centredDzMm: float = FUNNEL_CENTRED_DZ_MM
    liftDzMm: float | None = None
    lowContactMm: float | None = None
    settleBeforeDescend: bool = True
    closedBelow: float = 0.5
    openGripper: float = 1.0
    closedGripper: float = 0.0
    controlPeriodS: float = 1.0 / 30.0

    @property
    def alignZ(self) -> float:
        return self.pegXyz[2] + self.alignDzMm / 1000.0

    @property
    def slowBelowZ(self) -> float:
        return self.pegXyz[2] + self.slowBelowDzMm / 1000.0

    @property
    def centredZ(self) -> float:
        return self.pegXyz[2] + self.centredDzMm / 1000.0

    @property
    def upZ(self) -> float:
        """The height "up first" goes to: `alignZ`, or `liftDzMm` above the peg when that is higher."""

        if self.liftDzMm is None:
            return self.alignZ
        return max(self.alignZ, self.pegXyz[2] + self.liftDzMm / 1000.0)

    @property
    def closeXyz(self) -> tuple[float, float, float]:
        return (self.pegXyz[0], self.pegXyz[1], self.pegXyz[2] + self.closeDzMm / 1000.0)


def _xy_error_mm(xyz: tuple[float, float, float], peg: tuple[float, float, float]) -> float:
    return math.hypot(xyz[0] - peg[0], xyz[1] - peg[1]) * 1000.0


def _off_path_mm(xyz: tuple[float, float, float], start_xy: tuple[float, float], end_xy: tuple[float, float]) -> float:
    """XY distance from the segment start->end: lag along a moving setpoint is not drift (the
    arm trails a 0.05 m/s sideways setpoint by up to 9 mm, 10-19 ALIGN steps)."""

    dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 <= 0.0 else ((xyz[0] - start_xy[0]) * dx + (xyz[1] - start_xy[1]) * dy) / length2
    t = min(1.0, max(0.0, t))
    return math.hypot(xyz[0] - (start_xy[0] + t * dx), xyz[1] - (start_xy[1] + t * dy)) * 1000.0


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
    previousCommand: tuple[float, float, float] | None = None
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
    # Where ALIGN goes across to: the peg's XY, or `residualMm` short of it along the entry error.
    alignGoalXy: tuple[float, float] | None = None
    residualAtDescendStartMm: float = 0.0
    lifted: bool = False
    # The policy still drove this step, below `liftDzMm`: on a late takeover that is the
    # off-centre descent the recovery undoes, which a whole-trial recording leaves out rather
    # than teach (`dagger_step_role`'s `policy_off_course`).
    lowPolicyStep: bool = False
    lowPolicySteps: int = 0

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
            stalled = contact = False
            if cfg.liftDzMm is not None and ee_xyz[2] < cfg.pegXyz[2] + cfg.liftDzMm / 1000.0:
                stalled = self._settled(ee_xyz)
                contact = (
                    cfg.lowContactMm is not None
                    and self.previousCommand is not None
                    and math.dist(ee_xyz, self.previousCommand) * 1000.0 > cfg.lowContactMm
                )
            else:
                self.quietSteps = 0
            if reached or policy_closes or stalled or contact:
                if reached:
                    transition = "reached_align_z"
                elif policy_closes:
                    transition = "policy_closed"
                else:
                    transition = "contact_low" if contact else "stalled_low"
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
                self.alignGoalXy = self._align_goal_xy(ee_xyz, xy_err)
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
                # Below until the *setpoint* is up, then across once the arm is within the slack
                # of it: the peg top is ~40 mm under alignZ, so a few mm of dead band is no risk.
                up_z = cfg.upZ
                below = self.setpoint[2] < up_z - 1e-9 or ee_xyz[2] < up_z - FUNNEL_ALIGN_Z_SLACK_M
                goal_xy = self.alignGoalXy if self.alignGoalXy is not None else (peg[0], peg[1])
                if xy_err > cfg.captureXyMm and below:
                    # Up first, never sideways at peg height.
                    goal = (self.setpoint[0], self.setpoint[1], up_z)
                else:
                    goal = (goal_xy[0], goal_xy[1], self.setpoint[2])
                self.setpoint, arrived = _walk(self.setpoint, goal, cfg.alignSpeedMS * cfg.controlPeriodS)
                settled = self._settled(ee_xyz)
                goal_err = _xy_error_mm(ee_xyz, (goal_xy[0], goal_xy[1], 0.0))
                if (
                    arrived
                    and goal[:2] == goal_xy
                    and goal_err <= cfg.captureXyMm
                    and (settled or not cfg.settleBeforeDescend)
                ):
                    transition = "aligned"
                    self.descendStartStep = step_idx
                    self.xyErrorAtDescendStartMm = round(xy_err, 1)
                    self.residualAtDescendStartMm = round(_xy_error_mm(self.setpoint, peg), 1)
                    self._enter(DESCEND)
                elif policy_xyz[2] < ee_xyz[2] - 1e-4:
                    descend_blocked = True
            elif self.state == DESCEND:
                # Off the path down: the peg's XY, or on a corrective diagonal the line to it.
                start_xy = self.alignGoalXy if self.alignGoalXy is not None else (peg[0], peg[1])
                if _off_path_mm(ee_xyz, start_xy, (peg[0], peg[1])) > cfg.captureXyMm:
                    transition = "left_capture"
                    self.realigns += 1
                    self.alignGoalXy = (peg[0], peg[1])
                    self._enter(ALIGN)
                else:
                    speed = cfg.fastDescendSpeedMS if self.setpoint[2] > cfg.slowBelowZ + 1e-9 else cfg.descendSpeedMS
                    goal = cfg.closeXyz
                    if self.setpoint[:2] != (peg[0], peg[1]):
                        # The residual closes on a straight line, over the peg by `centredZ`.
                        goal = (peg[0], peg[1], max(cfg.centredZ, cfg.closeXyz[2]))
                    self.setpoint, arrived = _walk(self.setpoint, goal, speed * cfg.controlPeriodS)
                    arrived = arrived and goal == cfg.closeXyz
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

        self.lowPolicyStep = (
            self.state == SEARCH and cfg.liftDzMm is not None and ee_xyz[2] < peg[2] + cfg.liftDzMm / 1000.0
        )
        self.lowPolicySteps += int(self.lowPolicyStep)
        if descend_blocked:
            self.blockedSteps += 1
        residual_xy = math.hypot(executed_xyz[0] - policy_xyz[0], executed_xyz[1] - policy_xyz[1]) * 1000.0
        if self.state != SEARCH:
            self.maxResidualMm = max(self.maxResidualMm, residual_xy)
        self.previousXyz = tuple(ee_xyz)  # type: ignore[assignment]
        self.previousCommand = tuple(executed_xyz)  # type: ignore[assignment]

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

    def _align_goal_xy(self, ee_xyz: tuple[float, float, float], xy_err: float) -> tuple[float, float]:
        """ALIGN's across target: the peg, or up to `residualMm` short of it along the policy's own
        error. Full alignment after a lift (the recovery), and never more residual than the height
        left above `centredZ`, so the diagonal is no flatter than 45 degrees."""

        cfg = self.config
        peg = cfg.pegXyz
        lifting = xy_err > cfg.captureXyMm and ee_xyz[2] < cfg.upZ - FUNNEL_ALIGN_Z_SLACK_M
        self.lifted = lifting and cfg.liftDzMm is not None and cfg.upZ > cfg.alignZ
        across_z = cfg.upZ if xy_err > cfg.captureXyMm else ee_xyz[2]
        residual = 0.0 if lifting else min(cfg.residualMm, xy_err, (across_z - cfg.centredZ) * 1000.0)
        if residual <= 0.0 or xy_err <= 0.0:
            return (peg[0], peg[1])
        scale = residual / xy_err
        return (peg[0] + (ee_xyz[0] - peg[0]) * scale, peg[1] + (ee_xyz[1] - peg[1]) * scale)

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
            "funnelAlignDzMm": round(cfg.alignDzMm, 1),
            "funnelResidualMm": round(cfg.residualMm, 1),
            "residualAtDescendStartMm": self.residualAtDescendStartMm,
            "funnelLiftDzMm": cfg.liftDzMm,
            "lifted": self.lifted,
            "lowPolicySteps": self.lowPolicySteps,
            "pegSource": cfg.pegSource,
        }
