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

"""The grasp-only loop: place the peg, let the policy grasp it, grade the grasp by lifting it.

Why it exists (card 12): the hands-off grasp outcome scatters so widely -- closure height SD
21.5 mm -- that a 20-rollout batch has 19-32% power to see a 6-9 mm shift. The 91-173 trials per
arm that the question needs are 7.6 hours of full rollouts and about an hour of this loop,
because a trial here ends at the grasp: no transport, no insertion, no person grading it.

What makes it unattended is that the grasp grades itself. Checked 2026-09-23 against the 38
hand-graded hands-off rollouts of 09-22: the measured width *once the peg has been lifted* is
bimodal -- held 0.036-0.405, not held 0.006-0.016, 38/38 agreement with the hand grade. The width
right after the close is not: 24/34, because fingers that close on the peg and lose it on the lift
read as a grasp. So the verdict is taken after a scripted lift, never at the close.

The loop keeps track of where the peg is, because that decides how the next trial is staged:

    at_pick   in the fixture at the reset's pick pose  -> the ordinary scene reset ("fixture")
    held      in the fingers after a held verdict       -> set down where it was lifted, re-gripped
                                                           the script's way, carried to the next
                                                           target ("regrip")
    untouched after a miss whose tool path never came  -> the script picks it up where it stands
              low near the peg                             and carries it to the next target
                                                           ("repick"); an empty pick makes it lost
    lost      after any other miss, or a re-grip or     -> a person puts it back in the fixture,
              re-pick that came up empty                   or the run halts if nobody is there

Every staged peg is released from the script's own grip, never the policy's. On 2026-09-23 the
B arm held 8/8 pegs staged from a scripted grip and 2/6 carried on in the policy's: a policy's
grip is high, low or by the edge, the peg slides in it on the carry, and the drop on release
varies with all of that -- it bounced, tipped or walked off the target. Nor is every empty grasp
re-picked: the peg a policy closed on air beside has usually been knocked over, and re-picks came
up empty 9 times in 13, each one costing a reset before the person was asked anyway. Only a miss
whose whole tool path stayed clear of the peg is (GraspHandover.peg_untouched): the pure policy
misses mostly in mid-air, several cm off, and 8 of its 13 misses in the 09-23/09-24 runs were
that kind.

Where the peg stands is the tool position *measured* when the fingers opened, not the commanded
target: a place step converges to within 6 mm, and the funnel aims at the peg.

Only the policy segment is the policy's. Everything else is the reset's own step loop, so the
fence, the reach check and the stall detection are the ones every other motion on this rig has.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any, Callable, Iterable

from tools.fr3.grasp_funnel import FUNNEL_CLOSE_DZ_MM, FunnelConfig, GraspFunnel
from tools.fr3.scene_reset import (
    SCENE_RESET_LIFT_M,
    SceneResetError,
    SceneResetRequest,
    SceneResetStroke,
    _check_xyz_in_workspace,
    _hold_where_it_is,
    _move_to_start,
    _observation_xyz_rotvec_gripper,
    _robot_workspace_bounds,
    _run_step,
    _sample_xy_from_strokes,
    _workspace_bounds,
    execute_scene_reset,
    parse_mask_strokes,
    precise_sleep,
    set_force_trace_path,
    traced_hold,
)

# Midway between the widest empty reading and the narrowest held one, both measured after the
# lift on 09-22 (0.016 and 0.036). terminal_trials uses 0.10 on the driver's note that nothing
# genuine reads between 0.01 and 0.25 -- one hand-graded held rollout read 0.036 while carried,
# so that note does not hold for a policy's grasp, which can take the peg by its edge.
GRASP_LOOP_HELD_WIDTH = 0.025
# ...and the fingers must also stand this far *wider than they were told to close to*. The lift
# holds the policy's own command, and empty fingers stop at that command, not at zero: on
# 2026-09-23 four A-arm closes 31-162 mm off the peg read 0.057 / 0.075 / 0.062 / 0.491 lifted
# against commands of 0.056 / 0.074 / 0.067 / 0.496, and all four were graded held. A peg between
# the fingers reads ~0.31 whatever the command. 0.05 is far from both.
GRASP_LOOP_BLOCKED_MARGIN = 0.05
# Enough to take the peg's whole length off the table, so a peg that is merely being pressed
# down reads as what it is. Not the reset's 8 cm: this lift is the measurement, and a short one
# keeps a dropped peg upright and near where it stood.
GRASP_LOOP_CHECK_LIFT_M = 0.03
# Steps the gripper command must stay closed before the policy is taken off the arm. The trace
# records a two-step 0.4997 blip that was not a grasp (rollout 9 of L4_full48_holdout22_40);
# a third of a second at 30 Hz is well past it and well before the policy lifts on its own.
GRASP_LOOP_CLOSE_SETTLE_STEPS = 10
# The policy's grasp height sets how far the peg hangs below the fingers, so setting a held peg
# down to be re-gripped goes back to the grasp height, where the peg stood on the table. No
# margin: a peg that slid on the lift touches down early and is pressed by the slide, which the
# 6 mm step tolerance absorbs; a margin would be a drop for every peg that did not slide.
GRASP_LOOP_PLACE_MARGIN_M = 0.0
# The script's grip on a peg standing on the table: the height the funnel closes at, which held
# the peg at 0.31 wide in every B trial of 09-23 that reached it on a staged peg.
GRASP_LOOP_REGRIP_DZ_M = FUNNEL_CLOSE_DZ_MM / 1000.0
# How far below touching the table a peg is set down before it is let go. Zero: a peg in the
# script's grip touches at the re-grip height, and the place goes exactly there. A 3 mm press was
# tried on 2026-09-24 and tipped the peg over -- the arm was still sliding sideways when the peg
# bottom caught the table (see GRASP_LOOP_HOVER_M) -- so the knob stays, at zero, under the 6 mm
# step tolerance past which the place step could never converge.
GRASP_LOOP_PLACE_PRESS_M = 0.0
# Held at the place height, still closed, before the fingers open, so the arm has arrived.
GRASP_LOOP_PLACE_DWELL_S = 1.0
# Every descent onto the table -- a place or a pick -- stops this far above it and waits there to
# stop sideways (GRASP_LOOP_HOVER_STILL_S), within GRASP_LOOP_HOVER_TOLERANCE_M in xy, before the
# last stretch. The reset's steps call a
# waypoint reached at 6 mm, and on 09-23 every one of ~70 place and pick descents was called
# reached 5-6 mm short, sideways (x -4, y +3), still closing; the open step after it read 0.3-1.6.
# So the fingers met the table still sliding -- a released peg walked, a pressed one tipped, and
# a closing pick shoved the peg. What the hover is for is that the arm has *stopped*, so it waits
# for stillness, not arrival: the arm parks inside a dead band that does not close with time
# (scene_reset._run_step, 09-11). The first cut, 3 mm in 3-D, timed out on 09-24 parked at 3.2
# mm -- 2.7 mm sideways plus 1.8 mm of sag with the peg in the fingers, unchanged for 20 s.
# 5 mm xy bounds how far off a stopped arm may be and still set down or take the peg.
GRASP_LOOP_HOVER_M = 0.010
GRASP_LOOP_HOVER_TOLERANCE_M = 0.005
GRASP_LOOP_HOVER_STILL_S = 0.3
# A miss leaves the peg standing when the tool never came low near it. "Near" is a horizontal
# radius the open fingers could reach the peg from: fully open they stand about 48 mm apart (a
# 15 mm peg reads 0.31), so a finger's outside is ~30 mm off the tool point, plus the peg's
# 7.5 mm radius. "Low" is below the peg's grasp height plus this clearance -- the peg top.
# Sized on the 09-23/09-24 trajectories: it passes the A misses closed 2-4 cm above and 4-7 cm
# off, and none that went below the grasp height within 3 cm.
GRASP_LOOP_UNTOUCHED_XY_M = 0.040
GRASP_LOOP_UNTOUCHED_Z_M = 0.020
# Held still after the fingers open, so a peg that rocks on release is not dragged by the retreat.
GRASP_LOOP_RELEASE_SETTLE_S = 0.5
GRASP_LOOP_WIDTH_SAMPLES = 10
GRASP_LOOP_WIDTH_SETTLE_S = 0.3
# How long the policy gets to close on the peg. The 09-22 hands-off closures happened at a
# median of ~150 steps; twice that is still a trial, three times is a policy that is lost.
GRASP_LOOP_MAX_POLICY_STEPS = 450
# The reset's own gripper command and table height, so a staged peg is the peg a rollout sees.
GRASP_LOOP_TARGET_Z = 0.058
GRASP_LOOP_PICK_XYZ = (0.3640, -0.1370, 0.0550)
# Radius of the stroke put around each recorded reset target when the mask has to be rebuilt
# from a rollout log: the targets themselves are the distribution, this only fills between them.
GRASP_LOOP_LOG_STROKE_RADIUS_M = 0.010
# Which arms a run grades. A is the pure policy; B is the policy with the GT grasp funnel
# (`grasp_funnel.py`). "AB" interleaves them in randomised pairs, because the rig drifts within a
# session and two arms run one after the other would each carry a different stretch of it.
GRASP_LOOP_ARMS = ("A", "B", "AB")
# Steps the funnel gets once it has the arm: align, descend at 0.02 m/s from +60 to -6 mm, settle,
# close. About 6 s when nothing goes wrong; twice that is a funnel that is stuck.
GRASP_LOOP_FUNNEL_MAX_STEPS = 360


@dataclass(frozen=True)
class GraspLoopRequest:
    pickXyz: tuple[float, float, float] = GRASP_LOOP_PICK_XYZ
    targetZ: float = GRASP_LOOP_TARGET_Z
    strokes: tuple[SceneResetStroke, ...] = field(default_factory=tuple)
    trials: int = 40
    heldWidth: float = GRASP_LOOP_HELD_WIDTH
    blockedMargin: float = GRASP_LOOP_BLOCKED_MARGIN
    checkLiftM: float = GRASP_LOOP_CHECK_LIFT_M
    closeSettleSteps: int = GRASP_LOOP_CLOSE_SETTLE_STEPS
    closedBelow: float = 0.5
    maxPolicySteps: int = GRASP_LOOP_MAX_POLICY_STEPS
    placeMarginM: float = GRASP_LOOP_PLACE_MARGIN_M
    placePressM: float = GRASP_LOOP_PLACE_PRESS_M
    placeDwellS: float = GRASP_LOOP_PLACE_DWELL_S
    releaseSettleS: float = GRASP_LOOP_RELEASE_SETTLE_S
    attended: bool = False
    arms: str = "A"
    funnelMaxSteps: int = GRASP_LOOP_FUNNEL_MAX_STEPS
    seed: int = 0
    openGripper: float = 1.0
    closedGripper: float = 0.0
    controlPeriodS: float = 1.0 / 30.0

    @property
    def regripZ(self) -> float:
        return self.targetZ + GRASP_LOOP_REGRIP_DZ_M

    def step_request(self, request_id: str) -> SceneResetRequest:
        """The carrier `_run_step` reads its timeout, tolerance and period from."""

        return SceneResetRequest(
            pickXyz=self.pickXyz,
            targetXyz=(self.pickXyz[0], self.pickXyz[1], self.targetZ),
            openGripper=self.openGripper,
            closedGripper=self.closedGripper,
            returnToStart=True,
            controlPeriodS=self.controlPeriodS,
            maskStrokes=self.strokes,
            requestId=request_id,
        )


def validate_grasp_loop_request(
    request: GraspLoopRequest,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
) -> None:
    if not request.strokes:
        raise SceneResetError("the target mask is empty; there is nowhere to put the peg.")
    if request.trials <= 0 or request.maxPolicySteps <= 0 or request.closeSettleSteps <= 0:
        raise SceneResetError("trials, maxPolicySteps and closeSettleSteps must be positive.")
    if not 0.0 < request.heldWidth < 1.0:
        raise SceneResetError("heldWidth is a normalized gripper reading and must be in (0, 1).")
    if request.checkLiftM <= 0.0 or request.placeMarginM < 0.0:
        raise SceneResetError("checkLiftM must be positive and placeMarginM non-negative.")
    if not 0.0 <= request.placePressM < 0.006:
        raise SceneResetError("placePressM must be in [0, 6 mm): past the step tolerance the place step never converges.")
    if request.arms not in GRASP_LOOP_ARMS:
        raise SceneResetError(f"arms must be one of {', '.join(GRASP_LOOP_ARMS)}, not {request.arms!r}.")
    if request.funnelMaxSteps <= 0:
        raise SceneResetError("funnelMaxSteps must be positive.")
    low, high = _workspace_bounds(workspace_min, workspace_max)
    _check_xyz_in_workspace(request.pickXyz, "pickXyz", low, high)
    # Every stroke centre, at table and carry height: a mask partly outside the fence is caught by
    # the sampler's retries, one entirely outside it is a mask for some other rig.
    inside = 0
    for stroke in request.strokes:
        try:
            _check_xyz_in_workspace((stroke.x, stroke.y, request.targetZ), "stroke", low, high)
            _check_xyz_in_workspace(
                (stroke.x, stroke.y, request.targetZ + SCENE_RESET_LIFT_M), "stroke", low, high
            )
            inside += 1
        except SceneResetError:
            continue
    if inside == 0:
        raise SceneResetError("no stroke of the target mask is inside the workspace fence.")


def load_mask_strokes(path: Path | str, *, radius_m: float = GRASP_LOOP_LOG_STROKE_RADIUS_M) -> tuple[SceneResetStroke, ...]:
    """Strokes from the gateway's stored mask, or rebuilt from a rollout log's reset targets.

    The second form exists because the mask lives in the page until the gateway stores it, and
    the 09-22 batches that this loop is meant to be compared against were staged from a mask
    that was never written down. Their `resetTarget`s were, so the distribution they were drawn
    from can be rebuilt from the draws themselves.
    """

    path = Path(path)
    if path.suffix == ".jsonl":
        strokes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = row.get("resetTarget") if isinstance(row, dict) else None
            if isinstance(target, list) and len(target) == 3:
                strokes.append(SceneResetStroke(x=float(target[0]), y=float(target[1]), radiusM=radius_m))
        if not strokes:
            raise SceneResetError(f"{path} has no resetTarget to rebuild a mask from.")
        return tuple(strokes)
    return parse_mask_strokes(json.loads(path.read_text(encoding="utf-8")))


def sample_target_xyz(
    request: GraspLoopRequest,
    rng: random.Random,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
) -> tuple[float, float, float]:
    low, high = _workspace_bounds(workspace_min, workspace_max)
    for _attempt in range(1000):
        x, y = _sample_xy_from_strokes(request.strokes, rng)
        candidate = (x, y, request.targetZ)
        try:
            _check_xyz_in_workspace(candidate, "targetXyz", low, high)
            _check_xyz_in_workspace((x, y, request.targetZ + SCENE_RESET_LIFT_M), "targetAboveXyz", low, high)
            if "B" in request.arms:
                # The funnel closes below the reset's own grasp height.
                _check_xyz_in_workspace(
                    (x, y, request.targetZ + FUNNEL_CLOSE_DZ_MM / 1000.0), "funnelCloseXyz", low, high
                )
        except SceneResetError:
            continue
        return candidate
    raise SceneResetError("the target mask does not overlap the workspace fence.")


def arm_for_trial(request: GraspLoopRequest, trial: int) -> str:
    """The arm trial `trial` runs under. "AB" is a coin per pair, seeded, so a resumed run agrees."""

    if request.arms != "AB":
        return request.arms
    pair_first = "A" if random.Random(request.seed * 1_000_003 + trial // 2).random() < 0.5 else "B"
    if trial % 2 == 0:
        return pair_first
    return "B" if pair_first == "A" else "A"


def grasp_handover_due(
    closed_streak: int,
    *,
    commanded_gripper: float,
    closed_below: float,
    settle_steps: int,
) -> tuple[int, bool]:
    """The closed-command streak after this step, and whether the policy is taken off the arm.

    Keyed on the command, as `terminal_servo_arming` and the trace are, but for a different reason
    now that the width no longer drops out: the width is what the verdict reads, and a trigger
    that read it too would be grading the grasp before the lift that makes the reading mean
    anything.
    """

    if commanded_gripper >= closed_below:
        return 0, False
    closed_streak += 1
    return closed_streak, closed_streak >= settle_steps


@dataclass
class GraspHandover:
    """What the policy segment reports back. Filled by the rollout loop, read by this module."""

    closeSettleSteps: int = GRASP_LOOP_CLOSE_SETTLE_STEPS
    closedBelow: float = 0.5
    maxPolicySteps: int = GRASP_LOOP_MAX_POLICY_STEPS
    closedStreak: int = 0
    fired: bool = False
    closeStep: int | None = None
    closeXyz: tuple[float, float, float] | None = None
    handoverStep: int | None = None
    handoverXyz: tuple[float, float, float] | None = None
    commandedGripper: float | None = None
    # Arm B only. When set, the rollout loop executes what it returns instead of the policy's
    # command; the close it makes is then the close this object sees.
    funnel: GraspFunnel | None = None
    funnelMaxSteps: int = GRASP_LOOP_FUNNEL_MAX_STEPS
    # Where the peg stands, when known: the tool's lowest height over it, within the reach of the
    # open fingers, is what says whether a miss can have disturbed it.
    pegXyz: tuple[float, float, float] | None = None
    lowestNearPegM: float | None = None

    def peg_untouched(self) -> bool:
        """True when no step brought the tool below the peg top within reach of the fingers."""

        return self.pegXyz is not None and (
            self.lowestNearPegM is None or self.lowestNearPegM >= GRASP_LOOP_UNTOUCHED_Z_M
        )

    def timed_out(self, step: int) -> bool:
        """Out of steps: the policy's budget until the funnel has the arm, the funnel's after."""

        if self.funnel is not None and self.funnel.active and self.funnel.entryStep is not None:
            return step >= self.funnel.entryStep + self.funnelMaxSteps
        return step >= self.maxPolicySteps

    def observe(self, step: int, xyz: tuple[float, float, float], commanded_gripper: float) -> bool:
        """Feed one policy step; True when the arm should be taken off the policy now."""

        if self.pegXyz is not None and math.dist(xyz[:2], self.pegXyz[:2]) < GRASP_LOOP_UNTOUCHED_XY_M:
            above = float(xyz[2]) - self.pegXyz[2]
            if self.lowestNearPegM is None or above < self.lowestNearPegM:
                self.lowestNearPegM = above
        streak, fire = grasp_handover_due(
            self.closedStreak,
            commanded_gripper=commanded_gripper,
            closed_below=self.closedBelow,
            settle_steps=self.closeSettleSteps,
        )
        if streak == 1:
            # Where the fingers were told to shut, which is the closure height card 10 is about;
            # re-set on every new streak, so a close that was abandoned does not stand in for the
            # one that was held.
            self.closeStep = step
            self.closeXyz = tuple(float(v) for v in xyz)  # type: ignore[assignment]
        self.closedStreak = streak
        if fire:
            self.fired = True
            self.handoverStep = step
            self.handoverXyz = tuple(float(v) for v in xyz)  # type: ignore[assignment]
            self.commandedGripper = float(commanded_gripper)
        return fire


def read_width(robot: Any, samples: int = GRASP_LOOP_WIDTH_SAMPLES, period_s: float = 1.0 / 30.0) -> float:
    """Median of a short burst: one reading is one frame, and a verdict should not be."""

    widths = []
    for _ in range(max(1, samples)):
        _xyz, _rotvec, width = _observation_xyz_rotvec_gripper(robot)
        widths.append(float(width))
        precise_sleep(period_s)
    return float(statistics.median(widths))


def grasp_is_held(width: float, commanded_gripper: float, *, held_width: float, blocked_margin: float) -> bool:
    """Something is between the fingers: they read open, and wider than the command they hold."""

    return width >= held_width and width - commanded_gripper >= blocked_margin


def check_grasp(robot: Any, request: GraspLoopRequest, *, gripper: float, request_id: str) -> dict[str, Any]:
    """Lift straight up by `checkLiftM` holding the policy's own close, and read the fingers.

    The policy's gripper command is kept rather than replaced by the reset's full close: a
    harder squeeze would rescue marginal grasps the policy made, and the loop would then be
    measuring the script.
    """

    step_request = request.step_request(request_id)
    xyz, rotvec, width_at_close = _observation_xyz_rotvec_gripper(robot)
    lifted = (xyz[0], xyz[1], xyz[2] + request.checkLiftM)
    # The reset's name, so `_run_step` lets it move with a clamped peg (see terminal_trials).
    _run_step(robot, step_request, "lift_8cm_after_grasp", lifted, rotvec, gripper)
    precise_sleep(GRASP_LOOP_WIDTH_SETTLE_S)
    width = read_width(robot, period_s=request.controlPeriodS)
    lifted_xyz, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    return {
        "held": grasp_is_held(width, gripper, held_width=request.heldWidth, blocked_margin=request.blockedMargin),
        "widthLifted": round(width, 4),
        "widthOverCommand": round(width - gripper, 4),
        "widthAtClose": round(float(width_at_close), 4),
        "liftedXyz": [round(v, 5) for v in lifted_xyz],
        "rotvec": rotvec,
    }


def _clear_upward(robot: Any, request: GraspLoopRequest, step_request: SceneResetRequest, gripper: float, name: str) -> tuple[float, float, float]:
    """Straight up to carry height before any sideways move, so nothing is swept at table level."""

    xyz, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    carry_z = max(xyz[2], request.targetZ + SCENE_RESET_LIFT_M)
    if carry_z > xyz[2] + 1e-4:
        _run_step(robot, step_request, name, (xyz[0], xyz[1], carry_z), rotvec, gripper)
    return rotvec


def _settled_descent(
    robot: Any,
    step_request: SceneResetRequest,
    name: str,
    xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    gripper: float,
) -> None:
    """Down to `xyz` by way of a hover `GRASP_LOOP_HOVER_M` above it, where the arm has to stop."""

    hover_name = "settle_above_place" if "place" in name else "settle_above_pick"
    hover = (xyz[0], xyz[1], xyz[2] + GRASP_LOOP_HOVER_M)
    _run_step(
        robot, step_request, hover_name, hover, rotvec, gripper,
        tolerance_m=GRASP_LOOP_HOVER_TOLERANCE_M,
        still_window_s=GRASP_LOOP_HOVER_STILL_S,
    )
    _run_step(robot, step_request, name, xyz, rotvec, gripper)


def place_held_peg(
    robot: Any,
    request: GraspLoopRequest,
    target_xyz: tuple[float, float, float],
    *,
    place_z: float,
    gripper: float,
    request_id: str,
) -> tuple[float, float, float]:
    """Carry the peg the fingers already hold to `target_xyz`, set it down, home.

    Answers where the tool measured when the fingers opened, which is where the peg stands.
    """

    step_request = request.step_request(request_id)
    low, high = _robot_workspace_bounds(robot)
    carry_z = max(place_z, request.targetZ) + SCENE_RESET_LIFT_M
    for name, point in (
        ("move_to_place_above", (target_xyz[0], target_xyz[1], carry_z)),
        ("descend_8cm_to_place", (target_xyz[0], target_xyz[1], place_z)),
    ):
        _check_xyz_in_workspace(point, name, low, high)
    rotvec = _clear_upward(robot, request, step_request, gripper, "lift_8cm_after_grasp")
    _run_step(robot, step_request, "move_to_place_above", (target_xyz[0], target_xyz[1], carry_z), rotvec, gripper)
    _settled_descent(robot, step_request, "descend_8cm_to_place", (target_xyz[0], target_xyz[1], place_z), rotvec, gripper)
    released = _open_and_settle(robot, request, step_request, (target_xyz[0], target_xyz[1], place_z), rotvec)
    _run_step(robot, step_request, "retreat_8cm", (target_xyz[0], target_xyz[1], carry_z), rotvec, request.openGripper)
    _move_to_start(robot)
    return released


def _open_and_settle(
    robot: Any,
    request: GraspLoopRequest,
    step_request: SceneResetRequest,
    xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
) -> tuple[float, float, float]:
    traced_hold(robot, step_request.requestId, "dwell_before_open", request.placeDwellS, request.controlPeriodS)
    _run_step(robot, step_request, "open_gripper", xyz, rotvec, request.openGripper)
    traced_hold(robot, step_request.requestId, "settle_after_open", request.releaseSettleS, request.controlPeriodS)
    released, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    return released


def set_down_and_regrip(
    robot: Any,
    request: GraspLoopRequest,
    *,
    place_z: float,
    gripper: float,
    request_id: str,
) -> tuple[float, tuple[float, float, float]]:
    """Put a peg held in the policy's grip down under the lift, and take it again the script's way.

    Straight down from the check lift, so the peg goes back to about where the policy took it;
    home, so the re-grip is at the start orientation rather than whatever the policy's wrist
    was; then the reset's own pick at the measured release point. Answers the lifted width and
    where the fingers opened.
    """

    step_request = request.step_request(request_id)
    xyz, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    low, high = _robot_workspace_bounds(robot)
    _check_xyz_in_workspace((xyz[0], xyz[1], place_z), "descend_8cm_to_place", low, high)
    _settled_descent(robot, step_request, "descend_8cm_to_place", (xyz[0], xyz[1], place_z), rotvec, gripper)
    released = _open_and_settle(robot, request, step_request, (xyz[0], xyz[1], place_z), rotvec)
    _clear_upward(robot, request, step_request, request.openGripper, "retreat_8cm")
    _move_to_start(robot)
    width = verified_pick(robot, request, (released[0], released[1], request.regripZ), request_id=request_id)
    return width, released


def release_and_clear(robot: Any, request: GraspLoopRequest, *, request_id: str) -> None:
    """After an empty or abandoned grasp: open, straight up, home. Nothing is known to be held."""

    step_request = request.step_request(request_id)
    xyz, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    _run_step(robot, step_request, "open_gripper", xyz, rotvec, request.openGripper)
    _clear_upward(robot, request, step_request, request.openGripper, "retreat_8cm")
    _move_to_start(robot)


def verified_pick(
    robot: Any,
    request: GraspLoopRequest,
    at_xyz: tuple[float, float, float],
    *,
    request_id: str,
) -> float:
    """The reset's own pick, at `at_xyz`, answering the width it holds at carry height."""

    step_request = request.step_request(request_id)
    low, high = _robot_workspace_bounds(robot)
    above = (at_xyz[0], at_xyz[1], at_xyz[2] + SCENE_RESET_LIFT_M)
    _check_xyz_in_workspace(above, "regrip_above", low, high)
    _check_xyz_in_workspace(at_xyz, "regrip", low, high)
    rotvec = _clear_upward(robot, request, step_request, request.openGripper, "retreat_8cm")
    _run_step(robot, step_request, "go_to_pick_above", above, rotvec, request.openGripper)
    _settled_descent(robot, step_request, "descend_8cm_to_pick", at_xyz, rotvec, request.openGripper)
    _run_step(robot, step_request, "close_gripper", at_xyz, rotvec, request.closedGripper)
    _run_step(robot, step_request, "lift_8cm_after_grasp", above, rotvec, request.closedGripper)
    precise_sleep(GRASP_LOOP_WIDTH_SETTLE_S)
    return read_width(robot, period_s=request.controlPeriodS)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_grasp_loop(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Held rate over graded trials, and the two covariates card 12 says to look at next.

    A trial the policy never closed on is a failure of the grasp and counts against it; a trial
    stopped by the operator or aborted by a staging fault is not a policy outcome and does not.
    A run of more than one arm is also summarised per arm, which is the only reading of it that
    means anything: the pooled rate of an interleaved run is a rate of neither arm.
    """

    rows = list(rows)
    summary = _summarize_arm(rows)
    arms = sorted({str(r.get("arm", "A")) for r in rows})
    summary["byArm"] = {arm: _summarize_arm([r for r in rows if str(r.get("arm", "A")) == arm]) for arm in arms}
    return summary


def _summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r.get("verdict") in ("held", "empty", "no_close")]
    held = [r for r in graded if r["verdict"] == "held"]
    low, high = wilson_interval(len(held), len(graded))

    def med(values: list[float]) -> float | None:
        return round(statistics.median(values), 1) if values else None

    def by(verdict_set: tuple[str, ...], key: str) -> float | None:
        return med([r[key] for r in graded if r["verdict"] in verdict_set and r.get(key) is not None])

    out = {
        "graded": len(graded),
        "held": len(held),
        "heldRate": round(len(held) / len(graded), 3) if graded else None,
        "wilson95": [round(low, 3), round(high, 3)],
        "noClose": sum(1 for r in graded if r["verdict"] == "no_close"),
        "closeAboveTargetMm": {"held": by(("held",), "closeAboveTargetMm"), "empty": by(("empty",), "closeAboveTargetMm")},
        "lateralMm": {"held": by(("held",), "lateralMm"), "empty": by(("empty",), "lateralMm")},
    }
    funnel = [r for r in graded if r.get("funnelState") is not None]
    if funnel:
        out["funnel"] = {
            "entered": sum(1 for r in funnel if r.get("funnelEntryStep") is not None),
            "prematureDescend": sum(1 for r in funnel if r.get("prematureDescend")),
            "xyErrorAtEntryMm": med([r["xyErrorAtEntryMm"] for r in funnel if r.get("xyErrorAtEntryMm") is not None]),
            "xyErrorAtCloseMm": med([r["xyErrorAtCloseMm"] for r in funnel if r.get("xyErrorAtCloseMm") is not None]),
            "funnelCloseDzMm": med([r["funnelCloseDzMm"] for r in funnel if r.get("funnelCloseDzMm") is not None]),
        }
    return out


def completed_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("kind") == "trial":
            rows.append(row)
    return rows


class GraspLoopControl:
    """The loop's control channel: one word per line on a stream, normally the process's stdin.

    Stdin because the Rollout page already owns it -- the gateway holds every rollout's stdin as a
    pipe and writes control words into it -- so the page's buttons reach this loop the same way
    they reach an interactive rollout, and a person at a terminal can type the same words.

        stop / quit   end the run at the next trial boundary (the gateway's Stop sends `quit` and
                      escalates to a signal after its grace, which is the immediate brake)
        continue / "" the peg is back in the fixture; the loop may go on

    A stream that closes (stdin from /dev/null under setsid) answers any wait with "nobody is
    there", which is what an unattended run needs.
    """

    def __init__(self, stream: Any, *, log: Callable[[str], None] = lambda message: print(message, flush=True)):
        self._stream = stream
        self._log = log
        self._stop = threading.Event()
        self._continue = threading.Event()
        self._closed = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._read, name="grasp-loop-control", daemon=True).start()

    def _read(self) -> None:
        try:
            for raw in self._stream:
                word = str(raw).strip().lower()
                if word in ("stop", "quit"):
                    if not self._stop.is_set():
                        self._log("[INFO] grasp_loop_stop=requested")
                    self._stop.set()
                    self._continue.set()
                elif word in ("continue", ""):
                    self._continue.set()
        except (OSError, ValueError):
            pass
        self._closed.set()
        self._continue.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def wait_for_operator(self, message: str) -> bool:
        self._continue.clear()
        if self._closed.is_set():
            return False
        self._log(f"[ATTENTION] grasp_loop_needs_operator {message}")
        self._continue.wait()
        answered = not self._stop.is_set() and not self._closed.is_set()
        self._log(f"[INFO] grasp_loop_operator={'continued' if answered else 'gone'}")
        return answered


# The rollout runtime hands in one of these: it runs the policy from home with `handover` fed on
# every step, and returns its status. Kept a callback so this module never imports the policy.
PolicyTrial = Callable[[int, GraspHandover], str]
OperatorWait = Callable[[str], bool]


def run_grasp_loop(
    robot: Any,
    request: GraspLoopRequest,
    *,
    run_policy_trial: PolicyTrial,
    out_path: Path,
    stop_requested: Callable[[], bool] = lambda: False,
    wait_for_operator: OperatorWait | None = None,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run `request.trials` graded grasps, appending one JSONL row per trial to `out_path`.

    A resumed run starts from the peg in the fixture, whatever the last run left: the only state
    the file records is outcomes, and the peg's position after an interrupt is not one of them.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_trials(out_path)
    rng = random.Random(request.seed + len(done))
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    peg: str = "at_pick"
    peg_xyz: tuple[float, float, float] | None = None
    held_place_z: float | None = None
    held_gripper: float = request.closedGripper
    halted = ""

    def write(row: dict[str, Any]) -> None:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    force_trace = force_trace_path(out_path)
    set_force_trace_path(force_trace)
    write({"kind": "run_start", "at": time.time(), "request": _request_record(request), "resumedAfter": len(done)})
    log(
        f"[INFO] grasp_loop=start trials={request.trials} resumed_after={len(done)} out={out_path} "
        f"force_trace={force_trace}"
    )
    for trial in range(len(done), request.trials):
        if stop_requested():
            halted = "stop_requested"
            break
        request_id = f"grasp_loop_{trial:03d}_{time.time_ns()}"
        try:
            target = sample_target_xyz(request, rng, workspace_min=workspace_min, workspace_max=workspace_max)
            started = time.perf_counter()
            log(f"[INFO] grasp_loop_trial_start trial={trial} target={target[0]:.4f},{target[1]:.4f}")

            # ---- stage the peg at `target`, arm homed ------------------------------------------
            repicked = False
            if peg == "untouched":
                assert peg_xyz is not None
                width = verified_pick(robot, request, (peg_xyz[0], peg_xyz[1], request.regripZ), request_id=request_id)
                repicked = grasp_is_held(width, request.closedGripper, held_width=request.heldWidth, blocked_margin=request.blockedMargin)
                log(f"[INFO] grasp_loop_repick trial={trial} width={width:.3f} held={repicked}")
                if repicked:
                    peg, held_place_z, held_gripper = "held", request.regripZ - request.placePressM, request.closedGripper
                else:
                    release_and_clear(robot, request, request_id=request_id)
                    peg = "lost"
            if peg == "at_target":
                # A miss that came low near the peg: it is usually lying down by now.
                peg = "lost"
            if peg == "lost":
                if wait_for_operator is None or not wait_for_operator(
                    f"peg not where it was left: put it back in the fixture at pick "
                    f"{request.pickXyz[0]:.4f},{request.pickXyz[1]:.4f},{request.pickXyz[2]:.4f}"
                ):
                    halted = "peg_lost"
                    break
                peg = "at_pick"
            if peg == "held":
                staging = "repick" if repicked else "regrip"
                assert held_place_z is not None
                released = place_held_peg(robot, request, target, place_z=held_place_z, gripper=held_gripper, request_id=request_id)
            else:
                staging = "fixture"
                reset = execute_scene_reset(
                    robot,
                    replace(
                        request.step_request(request_id),
                        pickXyz=request.pickXyz,
                        targetXyz=(target[0], target[1], target[2] - request.placePressM),
                    ),
                    release_dwell_s=request.placeDwellS,
                    release_settle_s=request.releaseSettleS,
                    hover_m=GRASP_LOOP_HOVER_M,
                    hover_tolerance_m=GRASP_LOOP_HOVER_TOLERANCE_M,
                    hover_still_s=GRASP_LOOP_HOVER_STILL_S,
                )
                if not reset.get("ok"):
                    write({"kind": "halt", "trial": trial, "reason": "scene_reset_failed", "error": reset.get("error")})
                    halted = "scene_reset_failed"
                    break
                released = tuple(reset.get("releasedXyz") or target)
            # The peg stands where the fingers opened, at the table height the target names.
            peg, peg_xyz = "at_target", (float(released[0]), float(released[1]), target[2])
            place_offset_mm = math.dist(peg_xyz[:2], target[:2]) * 1000.0
            log(f"[INFO] grasp_loop_staged trial={trial} staging={staging} place_offset_mm={place_offset_mm:.1f}")
            staged_s = time.perf_counter() - started

            # ---- the policy's segment ----------------------------------------------------------
            arm = arm_for_trial(request, trial)
            handover = GraspHandover(
                closeSettleSteps=request.closeSettleSteps,
                closedBelow=request.closedBelow,
                maxPolicySteps=request.maxPolicySteps,
                funnelMaxSteps=request.funnelMaxSteps,
                pegXyz=peg_xyz,
            )
            if arm == "B":
                handover.funnel = GraspFunnel(
                    FunnelConfig(
                        pegXyz=peg_xyz,
                        closedBelow=request.closedBelow,
                        openGripper=request.openGripper,
                        closedGripper=request.closedGripper,
                        controlPeriodS=request.controlPeriodS,
                    )
                )
            log(f"[INFO] grasp_loop_arm trial={trial} arm={arm}")
            status = run_policy_trial(trial, handover)
            row: dict[str, Any] = {
                "kind": "trial",
                "trial": trial,
                "arm": arm,
                "staging": staging,
                "targetXyz": [round(v, 5) for v in target],
                "pegXyz": [round(v, 5) for v in peg_xyz],
                "placeOffsetMm": round(place_offset_mm, 1),
                "policyStatus": status,
                "closeStep": handover.closeStep,
                "closeXyz": None if handover.closeXyz is None else [round(v, 5) for v in handover.closeXyz],
                "handoverStep": handover.handoverStep,
                "commandedGripper": handover.commandedGripper,
                "stagedS": round(staged_s, 1),
            }
            if handover.closeXyz is not None:
                row["closeAboveTargetMm"] = round((handover.closeXyz[2] - target[2]) * 1000.0, 1)
                # From the peg as placed, not the target it was aimed at.
                row["lateralMm"] = round(math.dist(handover.closeXyz[:2], peg_xyz[:2]) * 1000.0, 1)
            row["lowestNearPegMm"] = (
                None if handover.lowestNearPegM is None else round(handover.lowestNearPegM * 1000.0, 1)
            )
            if handover.funnel is not None:
                row.update(handover.funnel.trial_record())
                row["funnelSteps"] = str(_write_funnel_steps(out_path, trial, handover.funnel))

            if handover.fired:
                check = check_grasp(robot, request, gripper=float(handover.commandedGripper), request_id=request_id)
                row.update({k: v for k, v in check.items() if k != "rotvec"})
                row["verdict"] = "held" if check["held"] else "empty"
                if check["held"]:
                    assert handover.closeXyz is not None
                    # Re-gripped now rather than at the next trial's staging, so a run that ends
                    # here parks a peg in the script's grip like any other.
                    width, set_down = set_down_and_regrip(
                        robot,
                        request,
                        place_z=handover.closeXyz[2] + request.placeMarginM,
                        gripper=float(handover.commandedGripper),
                        request_id=request_id,
                    )
                    regripped = grasp_is_held(width, request.closedGripper, held_width=request.heldWidth, blocked_margin=request.blockedMargin)
                    row["regripWidth"] = round(width, 4)
                    log(f"[INFO] grasp_loop_regrip trial={trial} width={width:.3f} held={regripped}")
                    if regripped:
                        peg, peg_xyz = "held", (set_down[0], set_down[1], target[2])
                        held_place_z = request.regripZ - request.placePressM
                        held_gripper = request.closedGripper
                    else:
                        release_and_clear(robot, request, request_id=request_id)
                        peg = "lost"
                else:
                    release_and_clear(robot, request, request_id=request_id)
            else:
                # Ran out of steps without a settled close, or was stopped. Either way nothing is
                # graded as held.
                row["verdict"] = "no_close" if status == "grasp_timeout" else "not_graded"
                release_and_clear(robot, request, request_id=request_id)
            if peg == "at_target":
                # Missed. Standing where it was put if the tool never came low near it; otherwise
                # it may be anywhere, and the next staging hands it to a person.
                row["pegUntouched"] = handover.peg_untouched()
                if row["pegUntouched"]:
                    peg = "untouched"
            row["trialS"] = round(time.perf_counter() - started, 1)
            write(row)
            done.append(row)
            log(
                f"[INFO] grasp_loop_trial trial={trial} arm={arm} verdict={row['verdict']} "
                f"width_lifted={row.get('widthLifted')} close_above_target_mm={row.get('closeAboveTargetMm')} "
                f"lateral_mm={row.get('lateralMm')} trial_s={row['trialS']}"
            )
        except Exception as exc:  # noqa: BLE001 - a motion fault ends the run, it must not end it silently
            # Held still rather than homed: the fingers may have the peg, and a fault is the
            # worst moment to decide that on the loop's behalf.
            try:
                _xyz, _rotvec, gripper_now = _observation_xyz_rotvec_gripper(robot)
                _hold_where_it_is(robot, gripper_now)
            except Exception:  # noqa: BLE001 - the fault being reported is the one worth reading
                pass
            write({"kind": "halt", "trial": trial, "reason": "motion_fault", "error": f"{type(exc).__name__}: {exc}"})
            log(f"[WARN] grasp_loop=halted trial={trial} reason=motion_fault details={exc}")
            halted, peg = "motion_fault", "unknown"
            break
        if status in ("quit", "stopped"):
            halted = status
            break

    if peg == "held" and held_place_z is not None and peg_xyz is not None:
        # Never end a run holding the peg in the air: put it down where it came from.
        place_held_peg(robot, request, peg_xyz, place_z=held_place_z, gripper=held_gripper, request_id="grasp_loop_park")
    set_force_trace_path(None)
    summary = summarize_grasp_loop(done)
    write({"kind": "run_end", "at": time.time(), "halted": halted, "summary": summary})
    log(f"[INFO] grasp_loop=done halted={halted or 'no'} summary={json.dumps(summary, sort_keys=True)}")
    return {"halted": halted, "summary": summary}


def force_trace_path(out_path: Path) -> Path:
    """Beside the row file: one JSONL line per scripted step, with the arm's wrench estimate.

    Read-only instrumentation for deciding whether descents can stop on contact; see
    `scene_reset.set_force_trace_path`. Step lines carry the trial's `requestId`.
    """

    return out_path.with_name(f"{out_path.stem}_force.jsonl")


def _write_funnel_steps(out_path: Path, trial: int, funnel: GraspFunnel) -> Path:
    """Every funnel step of one trial, beside the row file: the correction data, kept whole."""

    path = out_path.with_name(f"{out_path.stem}_funnel") / f"trial_{trial:03d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in funnel.steps:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _request_record(request: GraspLoopRequest) -> dict[str, Any]:
    data = {k: v for k, v in request.__dict__.items() if k != "strokes"}
    data["strokes"] = len(request.strokes)
    return data
