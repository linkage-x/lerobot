"""E6-data: a peg placed where we chose to place it, approached from where we chose to fail.

The one sentence this is built on is already true in this repo and had not been used: **the peg
the reset puts down is at a pose we commanded.** Every other route to automatic data on this rig
runs into "where is the peg", and this one does not have the question. That dissolves the hard
half of E6 with no perception at all, and it is what makes an unattended night possible.

What it does *not* dissolve is the terminal insertion, and the boundary is the whole design.

**The script can teach the approach and the grasp, and cannot teach the insertion.** Knowing where
the peg stands on the table is not knowing where it will sit in the fingers once they close --
E7's frame sweep put that correlation at r=0.13 twelve frames before contact and r=0.53 three
frames after, so the offset that decides an insertion does not exist yet at the moment a script
would have to plan for it. A terminal script therefore has only a fixed pose or a search, and a
search is the one thing that must not be distilled: its next move depends on an internal search
index rather than on the observation, so identical images carry different labels and behaviour
cloning averages eight ring landings onto the ring centre -- the point already measured to miss.
So this module records the approach and the grasp, and stops.

**The value is coverage and correction, not volume.** Grasping is where this rig actually fails:
29 of 31 takeover spans are in the grasp phase and only 1 of 119 end-to-end attempts got through
unassisted. And because the expert here is state-feedback -- E5 drove handoff offsets of
15.8-61.1 mm all the way down to 1.7-2.0 mm -- the arm can be *deliberately displaced* and then
corrected, which manufactures the recovery labels a demonstration never contains because a human
demonstrator does not make those mistakes on purpose.

The displacement is bounded by perception, not by kinematics. Push the arm far enough and the peg
leaves the frame, and then the correct action is no longer determined by the observation, which
reintroduces the multimodal-label problem through a different door. So the envelope belongs to
where the policy actually goes wrong -- the pose error at the moments a human took over -- and not
to what the arm can reach. Phase one displaces in translation only; orientation is the one axis
that can drive the fingertips into the table, and the reset's QC does not check fingertip
clearance yet. `perturbRotDeg` is refused rather than ignored.

**What is recorded and what is not.** A cycle is: place the peg (not recorded -- an environment
operation), displace the arm (not recorded), approach and grasp (recorded, this is the episode),
read the verdict, carry the peg on to the next placement (not recorded). The unrecorded legs emit
markers, because a dataset with unexplained gaps between episodes cannot be told apart from one
that dropped frames.

**And what is sent is what is stored, at a step the deployment will accept.** The recorded legs
walk at a speed derived from a target millimetres-per-step rather than from a speed, because
millimetres per step is what a policy learns; seconds are not in its action space. The reset's own
0.15 m/s is 5.0 mm a step at 30 Hz, which is exactly where `command_guard` clips and above the
99.9th percentile of the demonstrations. `collection_recorder.StepAudit` then checks every command
against that guard and invalidates the episode that breaks it, rather than trusting this paragraph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import random
import time
from typing import Any, Callable, Iterable

from tools.fr3.collection_recorder import (
    DEFAULT_STEP_MM,
    DEPLOYMENT_STEP_LIMIT_MM,
    ControlTap,
    StepAudit,
)
import tools.fr3.scene_reset as scene_reset_module
from tools.fr3.scene_reset import (
    SCENE_RESET_LIFT_M,
    SceneResetError,
    SceneResetRequest,
    SceneResetStroke,
    SceneResetUnreachableError,
    _check_xyz_in_workspace,
    _observation_xyz_rotvec_gripper,
    _reach_probe,
    _robot_workspace_bounds,
    _run_step,
    _sample_xy_from_strokes,
    _workspace_bounds,
)


# The normalized gripper opening below which the fingers are holding nothing, and how far a later
# grasp may sit from the first before the peg is no longer the thing being held. Both are
# `terminal_trials`' numbers and are deliberately the same numbers: two loops on one rig that
# disagree about what "holding the peg" means produce two datasets that cannot be compared.
AUTO_GRASP_FLOOR = 0.10
AUTO_GRASP_TOLERANCE = 0.08

# How far the arm may be displaced before the recorded approach begins, in millimetres.
#
# A placeholder with a job: it is meant to be replaced by the measured pose error at the instants
# a human took over, which is the distribution the policy actually visits. 25 mm is the order of
# the handoff offsets E5 recorded (15.8-61.1 mm) truncated to the half that a wrist camera can
# still see the peg from, and it is written here so that the first run has a number rather than a
# guess made at the console.
AUTO_PERTURB_XY_MM = 25.0
# The band of *absolute tool height* the displaced approaches start in, in metres.
#
# Absolute rather than relative to the peg, because the quantity it is aimed at is absolute: E4
# found z in [0.15, 0.20] empty in the existing data, while 92.2% of the 3846 DAgger takeover
# frames sit below z = 0.12. A band expressed as a height above the peg would move with every
# placement and cover that gap only by accident.
AUTO_PERTURB_Z_RANGE_M = (0.12, 0.22)
# Consecutive empty closes that end the run. One is an outcome and belongs in the data; two in a
# row mean the peg is not where the loop believes it put it, and every later cycle would be aimed
# with a ruler that has already been shown to be wrong.
AUTO_EMPTY_STREAK = 2
# Consecutive grasps whose width disagrees with the first that end the run. Same argument: the
# thing in the fingers has changed and the loop cannot see what it is.
AUTO_CHANGED_STREAK = 2
# How many cycles may fail their step audit before the run stops rather than filling a disk with
# frames the deployment guard would clip.
AUTO_AUDIT_FAILURE_LIMIT = 3


class AutoCollectError(SceneResetError):
    """A collection run that must not be started."""


@dataclass(frozen=True)
class CycleSpec:
    """One cycle, fixed before the arm moves: where the peg goes, and where the arm starts from."""

    index: int
    # Where the reset places the peg. Known by construction, which is the premise of the whole run.
    placeXyz: tuple[float, float, float]
    # Where the recorded approach begins, as an offset from directly above the peg. The zero
    # vector is a legitimate entry and is the nominal approach -- the control condition.
    startOffset: tuple[float, float, float]
    kind: str  # "nominal" | "recovery"

    def start_xyz(self) -> tuple[float, float, float]:
        return (
            self.placeXyz[0] + self.startOffset[0],
            self.placeXyz[1] + self.startOffset[1],
            self.placeXyz[2] + self.startOffset[2],
        )


@dataclass(frozen=True)
class AutoCollectRequest:
    """A whole night: where the peg may go, how far the arm may be displaced, and how fast.

    The mask is the same one the reset panel already draws and persists, so the region this run
    covers is the region a person has already authorised on a map rather than a second set of
    numbers that has to be kept in step with it.
    """

    maskStrokes: tuple[SceneResetStroke, ...]
    # Where the peg is fetched from at the very start, if the arm is not already holding it.
    pickXyz: tuple[float, float, float] | None
    # The table height the peg is placed at, and the height the arm carries it at between cycles.
    placeZ: float
    carryZ: float

    cycles: int = 50
    seed: int = 0
    # The fraction of cycles that begin from a displaced pose. The rest are nominal approaches,
    # which are the control: a dataset made only of corrections has no examples of not needing one.
    recoveryFraction: float = 0.7
    perturbXyMm: float = AUTO_PERTURB_XY_MM
    # Absolute tool heights, not offsets from the peg. See AUTO_PERTURB_Z_RANGE_M.
    perturbZRangeM: tuple[float, float] = AUTO_PERTURB_Z_RANGE_M
    # Phase two. Refused rather than ignored -- see the module docstring.
    perturbRotDeg: float = 0.0

    # What a recorded step may displace the tool by, in millimetres. The speed is derived from it.
    stepMm: float = DEFAULT_STEP_MM
    stepLimitMm: float = DEPLOYMENT_STEP_LIMIT_MM
    # The speed the *unrecorded* legs move at. They are environment operations and are free to use
    # the reset's own limit; only what goes into the dataset has to look like a policy step.
    transferSpeedMs: float | None = None

    liftM: float = SCENE_RESET_LIFT_M
    openGripper: float = 1.0
    closedGripper: float = 0.0
    graspSettleS: float = 0.6
    graspFloor: float = AUTO_GRASP_FLOOR
    graspToleranceM: float = AUTO_GRASP_TOLERANCE
    emptyStreak: int = AUTO_EMPTY_STREAK
    changedStreak: int = AUTO_CHANGED_STREAK
    auditFailureLimit: int = AUTO_AUDIT_FAILURE_LIMIT
    maxSeconds: float = 0.0

    timeoutS: float = 20.0
    toleranceM: float = 0.006
    gripperTolerance: float = 0.08
    controlPeriodS: float = 1.0 / 30.0
    requestId: str = ""

    def recorded_speed_ms(self) -> float:
        """Millimetres per step is the quantity; metres per second is what the walker takes."""

        return (self.stepMm / 1000.0) / self.controlPeriodS

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["maskStrokes"] = [asdict(stroke) for stroke in self.maskStrokes]
        data["recordedSpeedMs"] = self.recorded_speed_ms()
        return data


def _scene_request(request: AutoCollectRequest, request_id: str) -> SceneResetRequest:
    """A `SceneResetRequest` carrying only the fields `_run_step` reads.

    `_run_step` keys its gripper-wait policy and its timing off this object; building one here
    rather than duplicating those fields is what keeps a collection step and a reset step the same
    step. The poses are placeholders: every call passes its own waypoint.
    """

    return SceneResetRequest(
        pickXyz=(0.0, 0.0, 0.0),
        targetXyz=(0.0, 0.0, 0.0),
        liftM=request.liftM,
        openGripper=request.openGripper,
        closedGripper=request.closedGripper,
        timeoutS=request.timeoutS,
        toleranceM=request.toleranceM,
        gripperTolerance=request.gripperTolerance,
        graspSettleS=request.graspSettleS,
        controlPeriodS=request.controlPeriodS,
        requestId=request_id,
    )


def build_collection_schedule(request: AutoCollectRequest) -> tuple[CycleSpec, ...]:
    """The whole night, expanded before the arm moves and written into the log.

    Expanded rather than sampled per cycle for two reasons that both cost nothing here and cannot
    be recovered later. A schedule that exists before the run can be *read* before the run, which
    is the difference between authorising a plan and pressing a button that starts ten thousand
    motions. And a schedule drawn from a seeded generator in one place is reproducible, where
    `scene_reset`'s own `random.SystemRandom()` sampling is deliberately not -- fine for a reset
    nobody will ever need to repeat, wrong for a dataset somebody will.

    The nominal cycles are interleaved rather than blocked at the front, because a night whose
    displaced approaches all happen after midnight confounds the displacement with the fixture's
    creep, in exactly the way `terminal_trials` shuffles its sweep to avoid.
    """

    if request.cycles < 1:
        raise AutoCollectError("cycles must be at least 1.")
    if not request.maskStrokes:
        raise AutoCollectError(
            "maskStrokes is empty: there is nowhere the peg is allowed to be placed, and a run "
            "without a mask is a run whose reachable area was never authorised."
        )
    if not 0.0 <= request.recoveryFraction <= 1.0:
        raise AutoCollectError("recoveryFraction must be between 0 and 1.")
    if request.perturbRotDeg != 0.0:
        raise AutoCollectError(
            "perturbRotDeg is phase two and is refused here rather than ignored. Orientation is "
            "the one axis that can drive the fingertips into the table, and the reset's trajectory "
            "QC checks the tool point against the fence, not the fingertips against the surface. "
            "Add that check before displacing in orientation."
        )
    if request.perturbXyMm < 0.0:
        raise AutoCollectError("perturbXyMm must be non-negative.")
    low_z, high_z = (float(value) for value in request.perturbZRangeM)
    if not (0.0 < low_z <= high_z):
        raise AutoCollectError(f"perturbZRangeM {request.perturbZRangeM} must be a positive ascending band.")
    if low_z <= request.placeZ:
        raise AutoCollectError(
            f"perturbZRangeM starts at {low_z:.3f} m, at or below the table at {request.placeZ:.3f} m. "
            "These are absolute tool heights, not offsets above the peg."
        )
    if request.stepMm <= 0.0:
        raise AutoCollectError("stepMm must be positive.")
    if request.stepMm > request.stepLimitMm:
        raise AutoCollectError(
            f"stepMm {request.stepMm:.2f} is above the deployment step limit "
            f"{request.stepLimitMm:.2f} mm. Every recorded frame would be a command the rollout "
            "guard clips, which is the one thing this run must not produce."
        )

    rng = random.Random(request.seed)
    recovery_count = int(round(request.recoveryFraction * request.cycles))
    kinds = ["recovery"] * recovery_count + ["nominal"] * (request.cycles - recovery_count)
    rng.shuffle(kinds)

    specs: list[CycleSpec] = []
    for index, kind in enumerate(kinds):
        x, y = _sample_xy_from_strokes(request.maskStrokes, rng)
        place = (float(x), float(y), float(request.placeZ))
        if kind == "nominal":
            # Directly above the peg at the carry height: the approach a demonstration begins with.
            offset = (0.0, 0.0, float(request.carryZ) - float(request.placeZ))
        else:
            bearing = rng.uniform(0.0, 2.0 * math.pi)
            # Square-rooted so the draws are uniform over the disc rather than clustered at its
            # centre. A displacement distribution that is dense where the error is small spends
            # the night re-teaching the easy corrections.
            radius = (request.perturbXyMm / 1000.0) * math.sqrt(rng.random())
            height = rng.uniform(low_z, high_z)
            offset = (
                radius * math.cos(bearing),
                radius * math.sin(bearing),
                float(height) - float(request.placeZ),
            )
        specs.append(CycleSpec(index=index, placeXyz=place, startOffset=offset, kind=kind))
    return tuple(specs)


def validate_auto_collection(
    request: AutoCollectRequest,
    schedule: Iterable[CycleSpec],
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    reach_probe: Any = None,
) -> dict[str, Any]:
    """Check every pose the whole night can command, before the first one is sent.

    Run-level rather than cycle-level on purpose, and for the reason `terminal_trials` gives: a
    night that discovers on cycle 90 that one sampled placement is outside the fence has spent the
    night proving the fence is where it always was.
    """

    specs = tuple(schedule)
    if not specs:
        raise AutoCollectError("the schedule is empty.")
    if request.carryZ <= request.placeZ:
        raise AutoCollectError(
            f"carryZ {request.carryZ:.4f} must be above placeZ {request.placeZ:.4f}: the peg is "
            "carried between cycles, not dragged."
        )
    if request.graspFloor <= 0.0 or request.graspToleranceM <= 0.0:
        raise AutoCollectError("graspFloor and graspToleranceM must be positive.")

    low, high = _workspace_bounds(workspace_min, workspace_max)
    checked = 0
    for spec in specs:
        _check_xyz_in_workspace(spec.placeXyz, f"cycle[{spec.index}].placeXyz", low, high)
        _check_xyz_in_workspace(
            (spec.placeXyz[0], spec.placeXyz[1], spec.placeXyz[2] + request.liftM),
            f"cycle[{spec.index}].placeXyz+lift",
            low,
            high,
        )
        start = spec.start_xyz()
        _check_xyz_in_workspace(start, f"cycle[{spec.index}].startXyz", low, high)
        if reach_probe is not None:
            for name, xyz in (("placeXyz", spec.placeXyz), ("startXyz", start)):
                shortfall = reach_probe(xyz)
                if shortfall > 0.0:
                    raise AutoCollectError(
                        f"cycle[{spec.index}].{name} ({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) is "
                        f"{shortfall * 1000.0:.1f} mm outside the arm's own reach at this tool "
                        "orientation. This is the arm, not the fence."
                    )
        checked += 1
    if request.pickXyz is not None:
        _check_xyz_in_workspace(request.pickXyz, "pickXyz", low, high)

    widest = max(
        1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]) for spec in specs
    )
    recoveries = sum(1 for spec in specs if spec.kind == "recovery")
    return {
        "ok": True,
        "cycles": len(specs),
        "recoveryCycles": recoveries,
        "nominalCycles": len(specs) - recoveries,
        "checkedPoses": checked,
        "widestStartOffsetMm": widest,
        "recordedSpeedMs": request.recorded_speed_ms(),
        "stepMm": request.stepMm,
        # How much of the empty band the schedule actually covers. Reported rather than assumed,
        # because a mask whose placements sit high can leave the band as empty as it found it.
        "startZRangeM": [
            min(spec.start_xyz()[2] for spec in specs),
            max(spec.start_xyz()[2] for spec in specs),
        ],
    }


def grasp_verdict(width: float, reference: float | None, request: AutoCollectRequest) -> str:
    """Open fingers, empty fingers, a changed object, or the peg.

    `terminal_trials`' rule with one verdict added, and the addition is not cosmetic. That rule
    separates a clamped peg (about 0.31 normalized) from a hand shut on air (under 0.01) using a
    floor in the empty band between them -- which is correct for a width read *after* a close, and
    silently wrong for one read before. An open hand reads 1.0, sails over the floor and is
    reported as holding the peg. A run started with the fingers open would then place nothing,
    grasp nothing, and record a night of it.
    """

    ceiling = request.openGripper - request.graspToleranceM
    if width > ceiling:
        return "open"
    if width < request.graspFloor:
        return "empty"
    if reference is not None and abs(width - reference) > request.graspToleranceM:
        return "changed"
    return "held"


def describe_schedule(request: AutoCollectRequest, schedule: Iterable[CycleSpec]) -> str:
    """The night as text, for reading before it starts rather than after it has finished."""

    specs = tuple(schedule)
    recoveries = sum(1 for spec in specs if spec.kind == "recovery")
    lines = [
        f"cycles={len(specs)} recovery={recoveries} nominal={len(specs) - recoveries} seed={request.seed}",
        f"step_mm={request.stepMm:.2f} recorded_speed_ms={request.recorded_speed_ms():.4f} "
        f"control_period_s={request.controlPeriodS:.4f} step_limit_mm={request.stepLimitMm:.2f}",
        f"perturb_xy_mm={request.perturbXyMm:.1f} perturb_z_m={request.perturbZRangeM[0]:.3f}"
        f"-{request.perturbZRangeM[1]:.3f} perturb_rot_deg={request.perturbRotDeg:.1f}",
        f"place_z={request.placeZ:.4f} carry_z={request.carryZ:.4f} mask_strokes={len(request.maskStrokes)}",
    ]
    for spec in specs:
        start = spec.start_xyz()
        lines.append(
            f"  [{spec.index:03d}] {spec.kind:<8} place=({spec.placeXyz[0]:+.4f},{spec.placeXyz[1]:+.4f}) "
            f"start=({start[0]:+.4f},{start[1]:+.4f},{start[2]:+.4f}) "
            f"offset_mm={1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]):5.1f}"
        )
    return "\n".join(lines)


@dataclass
class CollectLoopState:
    graspReference: float | None = None
    heldWidth: float = 0.0
    emptyRun: int = 0
    changedRun: int = 0
    auditFailures: int = 0
    completed: int = 0
    held: int = 0
    recorded: int = 0
    invalidated: int = 0


def _leg(
    robot: Any,
    scene: SceneResetRequest,
    name: str,
    xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    gripper: float,
    *,
    tap: ControlTap | None,
    speed_ms: float,
) -> None:
    _run_step(robot, scene, name, xyz, rotvec, gripper, tap=tap, max_speed_ms=speed_ms)


def _place_the_peg(
    robot: Any,
    request: AutoCollectRequest,
    scene: SceneResetRequest,
    spec: CycleSpec,
    rotvec: tuple[float, float, float],
    *,
    tap: ControlTap,
    transfer_ms: float,
) -> None:
    """Put the peg down where the schedule says. Not recorded: this is the environment moving.

    The step names are the reset's because `_run_step` keys its gripper-wait policy on them --
    carrying a clamped peg must not wait for the fingers to reach the closed command, because they
    never will.
    """

    above = (spec.placeXyz[0], spec.placeXyz[1], request.carryZ)
    tap.mark("leg", leg="place", recorded=False, cycle=spec.index, toXyz=list(spec.placeXyz))
    _leg(robot, scene, "move_to_place_above", above, rotvec, request.closedGripper, tap=None, speed_ms=transfer_ms)
    _leg(robot, scene, "descend_8cm_to_place", spec.placeXyz, rotvec, request.closedGripper, tap=None, speed_ms=transfer_ms)
    _leg(robot, scene, "open_gripper", spec.placeXyz, rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)
    _leg(robot, scene, "retreat_after_release", above, rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)


def _displace(
    robot: Any,
    request: AutoCollectRequest,
    scene: SceneResetRequest,
    spec: CycleSpec,
    rotvec: tuple[float, float, float],
    *,
    tap: ControlTap,
    transfer_ms: float,
) -> None:
    """Move the arm to where the recorded approach begins. Not recorded: this is the mistake.

    Recording the displacement would teach the policy to make it. What is worth learning is the
    correction, which is the leg after this one.
    """

    start = spec.start_xyz()
    tap.mark(
        "leg",
        leg="displace",
        recorded=False,
        cycle=spec.index,
        cycleKind=spec.kind,
        toXyz=list(start),
        offsetMm=1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]),
    )
    # Via the carry height rather than straight to the start pose: a diagonal from wherever the
    # retreat left the arm can pass low over the peg it has just set down.
    _leg(robot, scene, "move_above_start", (start[0], start[1], request.carryZ), rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)
    _leg(robot, scene, "move_to_start_pose", start, rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)


def _approach_and_grasp(
    robot: Any,
    request: AutoCollectRequest,
    scene: SceneResetRequest,
    spec: CycleSpec,
    rotvec: tuple[float, float, float],
    *,
    tap: ControlTap,
    speed_ms: float,
) -> float:
    """The episode. Everything here is published to the recorder; nothing else in a cycle is."""

    above = (spec.placeXyz[0], spec.placeXyz[1], spec.placeXyz[2] + request.liftM)
    carry = (spec.placeXyz[0], spec.placeXyz[1], request.carryZ)
    _leg(robot, scene, "approach_above_peg", above, rotvec, request.openGripper, tap=tap, speed_ms=speed_ms)
    _leg(robot, scene, "descend_to_peg", spec.placeXyz, rotvec, request.openGripper, tap=tap, speed_ms=speed_ms)
    _leg(robot, scene, "close_gripper", spec.placeXyz, rotvec, request.closedGripper, tap=tap, speed_ms=speed_ms)
    _, _, width = _observation_xyz_rotvec_gripper(robot)
    _leg(robot, scene, "lift_8cm_after_grasp", carry, rotvec, request.closedGripper, tap=tap, speed_ms=speed_ms)
    return float(width)


def _park(robot: Any, request: AutoCollectRequest, scene: SceneResetRequest, transfer_ms: float) -> None:
    """Leave the arm at carry height holding whatever it holds.

    Deliberately not "open the fingers and go home", for `terminal_trials`' reason: a run that
    stopped because it could not read its own state is the worst moment to drop a peg on a fixture
    nobody is watching.
    """

    xyz, rotvec, gripper = _observation_xyz_rotvec_gripper(robot)
    if xyz[2] >= request.carryZ:
        return
    _run_step(
        robot,
        scene,
        "lift_8cm_after_grasp",
        (xyz[0], xyz[1], request.carryZ),
        rotvec,
        gripper,
        max_speed_ms=transfer_ms,
    )


class _Halt(Exception):
    """Internal: leave the loop through the same exit as every other stop."""


def run_auto_collection(
    robot: Any,
    request: AutoCollectRequest,
    schedule: Iterable[CycleSpec] | None = None,
    *,
    tap: ControlTap,
    should_stop: Callable[[], bool] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the night and answer with what happened and why it stopped.

    The loop holds one invariant at the top of every cycle: **the arm is holding the peg at carry
    height.** Everything is derived from it. Placing the peg is how a cycle begins rather than how
    it ends, so the peg's pose is commanded rather than searched for; and restoring the invariant
    is exactly what the recorded grasp does, so the thing being measured and the thing that closes
    the loop are the same event. A cycle that ends without the invariant cannot continue, which is
    why an empty close is a stop condition rather than a row.
    """

    specs = tuple(schedule) if schedule is not None else build_collection_schedule(request)
    _, preflight_rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    # Run-level QC lives here rather than in the caller so no entry point can skip it.
    validate_auto_collection(
        request,
        specs,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        reach_probe=_reach_probe(robot, preflight_rotvec),
    )

    scene = _scene_request(request, request.requestId or "auto_collect")
    # Read off the module rather than imported by name: the reset's speed limit is a module
    # global that tests and callers legitimately change, and a name bound at import time would
    # freeze whichever value happened to be current when this file was first loaded.
    transfer_ms = (
        float(request.transferSpeedMs)
        if request.transferSpeedMs
        else scene_reset_module.SCENE_RESET_MAX_SPEED_MS
    )
    recorded_ms = request.recorded_speed_ms()

    state = CollectLoopState()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    halted = ""

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    try:
        _, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
        # Fetch the peg once, to establish the invariant the rest of the run assumes.
        if request.pickXyz is not None:
            pick = tuple(float(value) for value in request.pickXyz)
            tap.mark("leg", leg="fetch", recorded=False, cycle=-1, toXyz=list(pick))
            _leg(robot, scene, "move_to_pick_above", (pick[0], pick[1], request.carryZ), rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)
            _leg(robot, scene, "descend_8cm_to_pick", pick, rotvec, request.openGripper, tap=None, speed_ms=transfer_ms)
            _leg(robot, scene, "close_gripper", pick, rotvec, request.closedGripper, tap=None, speed_ms=transfer_ms)
            _, _, width = _observation_xyz_rotvec_gripper(robot)
            _leg(robot, scene, "lift_8cm_after_grasp", (pick[0], pick[1], request.carryZ), rotvec, request.closedGripper, tap=None, speed_ms=transfer_ms)
        else:
            _, _, width = _observation_xyz_rotvec_gripper(robot)
        verdict = grasp_verdict(width, None, request)
        state.graspReference = width
        state.heldWidth = width
        emit({"kind": "grasp", "stage": "start", "widthNormalized": width, "graspVerdict": verdict})
        if verdict != "held":
            halted = "grasp_lost_at_start"
            raise _Halt

        for spec in specs:
            if request.maxSeconds > 0.0 and time.perf_counter() - started >= request.maxSeconds:
                halted = "time_budget"
                break
            # The gentle half of the brake, checked *here* rather than inside the walk, and that
            # placement is the whole point of having two. A cycle interrupted halfway leaves the
            # peg on the table and the arm somewhere the next cycle does not expect -- the loop's
            # invariant broken by the person trying to be careful. Stopping at a boundary ends the
            # night holding the peg at carry height, which is the state the next run starts from.
            #
            # The hard half is not here and does not need to be: SIGINT already unwinds through
            # the same exit, parks, and writes the summary.
            if should_stop is not None and should_stop():
                halted = "stop_requested"
                break

            _, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
            _place_the_peg(robot, request, scene, spec, rotvec, tap=tap, transfer_ms=transfer_ms)
            _displace(robot, request, scene, spec, rotvec, tap=tap, transfer_ms=transfer_ms)

            # Everything between these two marks is one episode, and the episode number is stamped
            # on the frames as they are published rather than recovered from timestamps later.
            episode = spec.index
            # One call: the tap resets its own audit, so the episode the frames are stamped with
            # and the episode the audit is judging can never be two different episodes.
            tap.begin_episode(episode)
            tap.mark(
                "episode_start",
                cycle=spec.index,
                cycleKind=spec.kind,
                placeXyz=list(spec.placeXyz),
                startXyz=list(spec.start_xyz()),
                offsetMm=1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]),
            )
            width = _approach_and_grasp(robot, request, scene, spec, rotvec, tap=tap, speed_ms=recorded_ms)
            verdict = grasp_verdict(width, state.graspReference, request)
            audit = tap.audit
            audit_ok = audit.episode_is_valid if audit is not None else True
            tap.mark(
                "episode_end",
                cycle=spec.index,
                verdict=verdict,
                auditOk=audit_ok,
                widthNormalized=width,
            )
            tap.begin_episode(-1)

            state.completed += 1
            state.heldWidth = width
            if verdict == "held":
                state.held += 1
            if audit_ok:
                state.recorded += 1
            else:
                state.invalidated += 1
                state.auditFailures += 1

            row = {
                "kind": "cycle",
                "index": spec.index,
                "cycleKind": spec.kind,
                "placeXyz": list(spec.placeXyz),
                "startXyz": list(spec.start_xyz()),
                "offsetMm": 1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]),
                "verdict": verdict,
                "widthNormalized": width,
                "referenceWidth": state.graspReference,
                "auditOk": audit_ok,
                "elapsedS": time.perf_counter() - started,
            }
            emit(row)

            state.emptyRun = state.emptyRun + 1 if verdict == "empty" else 0
            state.changedRun = state.changedRun + 1 if verdict == "changed" else 0
            if state.emptyRun >= request.emptyStreak:
                halted = "empty_streak"
                break
            if state.changedRun >= request.changedStreak:
                halted = "changed_streak"
                break
            if state.auditFailures >= request.auditFailureLimit:
                halted = "audit_failures"
                break
            if verdict != "held":
                # The invariant is broken: the fingers do not hold the peg, and the next cycle
                # would place nothing. One unrecorded recovery attempt from directly above the peg
                # is allowed, because the peg is still standing where it was set down and the
                # nominal grasp is the one that has already been shown to work.
                tap.mark("leg", leg="recover", recorded=False, cycle=spec.index)
                _, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
                nominal = replace(spec, startOffset=(0.0, 0.0, request.carryZ - request.placeZ))
                _displace(robot, request, scene, nominal, rotvec, tap=tap, transfer_ms=transfer_ms)
                width = _approach_and_grasp(
                    robot, request, scene, nominal, rotvec, tap=None, speed_ms=transfer_ms
                )
                recovered = grasp_verdict(width, state.graspReference, request)
                emit(
                    {
                        "kind": "recover",
                        "index": spec.index,
                        "widthNormalized": width,
                        "graspVerdict": recovered,
                    }
                )
                if recovered != "held":
                    halted = f"recovery_{recovered}"
                    break
                state.heldWidth = width
        else:
            halted = "schedule_complete"
    except _Halt:
        pass
    except KeyboardInterrupt:
        halted = "interrupted"
    except SceneResetUnreachableError as exc:
        halted = f"unreachable: {exc}"
    except SceneResetError as exc:
        halted = f"step_failed: {exc}"
    except (TimeoutError, RuntimeError) as exc:
        halted = f"step_failed: {exc}"

    try:
        _park(robot, request, scene, transfer_ms)
        parked = True
    except Exception as exc:  # noqa: BLE001 - the summary has to survive a failed park
        parked = False
        print(f"[WARN] auto_collect=park_failed details={exc}", flush=True)

    tap.mark("run_end", haltedOn=halted or "schedule_complete", cycles=state.completed)
    summary = {
        "kind": "summary",
        # A requested stop is a success. The night did what it was told, and logging the
        # operator's own brake as a failure would make every deliberate stop look like a fault.
        "ok": halted in {"schedule_complete", "time_budget", "stop_requested"},
        "haltedOn": halted or "schedule_complete",
        "parked": parked,
        "scheduled": len(specs),
        "cycles": state.completed,
        "held": state.held,
        "recordedEpisodes": state.recorded,
        "invalidatedEpisodes": state.invalidated,
        "graspReference": state.graspReference,
        "heldWidth": state.heldWidth,
        "elapsedS": time.perf_counter() - started,
        "audit": tap.audit.summary() if tap.audit is not None else None,
        "tap": tap.status(),
    }
    if on_row is not None:
        on_row(summary)
    summary["rows"] = rows
    return summary
