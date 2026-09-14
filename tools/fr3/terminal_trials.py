"""E6-lite: run the terminal experiment by itself, a hundred times, with nobody in the room.

Every terminal reading this project has rests on n = 8 or 9. The 62% that refuted the eighth
card's prediction carries a 95% interval of 24-91%, and the 4.2 mm capture radius inverted out
of it carries 2.5-8.8 mm. That interval is the problem, because the search added on 2026-09-10
is *designed against that number*: a 7 mm ring of eight landings covers a 9.7 mm disc if the
capture radius is 4.2 mm, and leaves a dead annulus between 2.5 and 4.5 mm if it is really the
nominal clearance. The pattern that is about to be trusted was sized by a figure whose interval
still admits the value that breaks it.

So this module runs the terminal experiment as a closed cycle -- descend, read the verdict off
the log, pick the peg back up, do it again -- and the point of the cycle is n.

Three things make it possible today, and all three are results rather than assumptions:

  * The verdict needs no camera and no person. `above_target_mm` separated seated (1.1-2.5 mm)
    from standing on the face (4.9-5.6 mm) with nothing in between, and `settle_mm` separates
    both from a peg sliding in the jaws. That is the whole scoring function.
  * The peg is always somewhere known. It is either in the hole or standing where the fingers
    let go of it, and both of those are the pose the servo already recorded on its way out. The
    hard half of E6 -- finding a peg the policy dropped somewhere on the table -- does not arise
    in a terminal cycle at all.
  * Nothing here needs the policy. The servo drives from wherever the arm is, and E5 measured
    that the approach contributes nothing to terminal XY (handoff offsets of 15.8-61.1 mm all
    converge to 1.7-2.0 mm before the descent starts). A trial loop that skips the policy is not
    a simplification of the experiment; it is the same experiment with its no-op removed.

What this deliberately does *not* do is randomise the handoff pose, which is how E6-lite was
first written down. E5 measured that variable to have no effect -- the servo erases it -- so a
hundred trials that vary only the handoff are a hundred repeats of one condition, not a curve.
The quantity that decides the outcome is where the peg tip is relative to the hole, and with no
perception on this rig the only way to *know* that number is to command it: aim the servo at the
hole plus a chosen offset and record whether it went in. That turns the run's output from one
more success rate into p(seat | offset), which is the capture radius itself, measured rather
than inverted out of a hit rate.

Commanding the offset only works if the peg sits in the jaws where the tool thinks it does, and
that is what the reference trials are for. A trial aimed at zero offset with the search switched
on will find the hole whatever the peg's seating bias is; re-gripping the peg while it stands in
the hole then re-zeroes that bias against the hole itself, and the tool pose at which it was
re-gripped is a *measurement* of where the hole now is. Fixture creep -- the reason the eighth
card says a fixed pose expires -- is therefore not a threat to a run of this shape but one of
its readings. The reference is re-read from physical contact every few trials, never assumed.

The loop stops rather than continuing on anything it cannot interpret: fingers that close on
nothing, a reference trial that cannot find the hole, two slips in a row. Sixty rows that mean
something beat two hundred that were taken after the peg fell over at 2 a.m.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import random
import time
from typing import Any, Callable, Iterable

from tools.fr3.scene_reset import (
    SCENE_RESET_LIFT_M,
    SceneResetError,
    _check_xyz_in_workspace,
    _observation_xyz_rotvec_gripper,
    _reach_probe,
    _robot_workspace_bounds,
    _run_step,
    _workspace_bounds,
)
from tools.fr3.terminal_servo import (
    TERMINAL_SERVO_SEARCH_RING_M,
    TerminalServoRequest,
    execute_terminal_servo,
    terminal_servo_search_offsets,
    validate_terminal_servo_trajectory,
)


# The normalized opening below which the fingers are holding nothing. Sourced, not chosen: the
# Pika reads about 0.31 with this peg clamped and under 0.01 with the hand shut on air, and the
# driver's own note records that no genuine reading on this hardware falls between 0.01 and 0.25.
# Anywhere in that empty band separates the two; the middle of it is the least arbitrary place.
# How far the tool axis may lean off the table's normal before a run means something else.
# Measured 2026-09-11: the arm was left at 15.1 deg by an earlier session and the servo inherited
# it, because the run reads its orientation from wherever the arm happens to be. A rigid peg
# leaning by theta wedges in a hole of single-side clearance c at a depth of about 2c/tan(theta):
# at 15.1 deg and 2.5 mm that is 18.5 mm, so the peg stops short and every verdict becomes a
# measurement of where it jammed. At 2 deg it is 143 mm, past anything this fixture can ask for.
# The home keyframe is level to 0.05 deg, so the remedy is to home rather than to trim a number.
TERMINAL_TRIAL_MAX_TILT_DEG = 2.0

TERMINAL_TRIAL_GRASP_FLOOR = 0.10
# How far a later grasp may sit from the first one of the same run before the loop stops. The
# first grasp is the reference because the absolute number belongs to the gripper's calibration
# and the run should not have to be told it. 0.08 normalized is about 7 mm of jaw opening on a
# 90 mm gripper -- far wider than the same peg re-gripped at the same place, far narrower than
# the 0.31 that separates holding the peg from holding nothing.
TERMINAL_TRIAL_GRASP_TOLERANCE = 0.08
# The low edge of the measured standing-on-the-face cluster, in millimetres. Together with the
# servo's own seated threshold this leaves a band that the nine descents of 2026-09-10 never
# produced; a stop that lands in it is reported as ambiguous rather than assigned to whichever
# side is nearer, because with n = 9 behind the two clusters an unoccupied gap is not yet
# evidence that nothing lives there.
TERMINAL_TRIAL_STANDING_MM = 4.9
# How far a reference trial may move the hole estimate in one step, in metres. Derived from the
# search geometry rather than picked: a search that seats the peg cannot have found the hole
# further than the ring radius plus the capture radius from where it was aimed, 7 + 4.2 mm, so a
# correction past this is not a hole the search found. It is a misread verdict, a peg that moved
# in the jaws, or a fixture that was knocked -- none of which should be silently adopted as the
# new target by a loop with nobody watching it.
TERMINAL_TRIAL_MAX_REFERENCE_STEP_M = 0.012
# Consecutive slips that end the run. One slip is an outcome and belongs in the data. Two in a
# row mean the peg is no longer where the last reference trial established it in the fingers, and
# from then on every offset is commanded with a ruler that changed length between the marks.
TERMINAL_TRIAL_SLIP_STREAK = 2
# How many times a reference trial may fail to find the hole before the run stops. The search
# covers 9.7 mm; failing that twice is not an unlucky offset, it is a rig that has moved.
TERMINAL_TRIAL_REFERENCE_ATTEMPTS = 2


class TerminalTrialError(SceneResetError):
    """A trial loop that must not be started."""


@dataclass(frozen=True)
class TrialSpec:
    """One commanded trial: where to aim, relative to the current hole estimate."""

    index: int
    # "reference" re-reads the hole from a seating event; "offset" is a datum on the curve.
    kind: str
    offsetMm: float
    bearingDeg: float
    searchRingM: float

    def offset_m(self) -> tuple[float, float]:
        radians = math.radians(self.bearingDeg)
        radius = self.offsetMm / 1000.0
        return (radius * math.cos(radians), radius * math.sin(radians))


@dataclass(frozen=True)
class TerminalTrialsRequest:
    """A whole run: the servo it is made of, the offsets to sweep, and where the peg starts.

    `servo` is a real `TerminalServoRequest` and every trial is a `replace()` of it, so a run is
    configured in exactly one place and a sweep with `searchRingM = 0` is the same descent E5
    measured, one variable at a time.
    """

    servo: TerminalServoRequest
    # The offsets to sweep, in millimetres of commanded lateral error. Zero is a legitimate entry
    # and is the search-off control at the centre.
    offsetsMm: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0)
    repeats: int = 4
    # A reference trial every this many offset trials, and always before the first.
    controlEvery: int = 8
    seed: int = 0
    # The ring used by the *offset* trials. Zero measures the bare capture radius, which is what
    # the pattern was sized against; non-zero measures the search on top of it.
    searchRingM: float = 0.0
    # The ring used by the reference trials, which have to find the hole rather than measure it.
    referenceRingM: float = TERMINAL_SERVO_SEARCH_RING_M
    # Where the peg is at the start, if the loop is to fetch it. None means it is already held.
    pickXyz: tuple[float, float, float] | None = None
    liftM: float = SCENE_RESET_LIFT_M
    openGripper: float = 1.0
    closedGripper: float = 0.0
    graspSettleS: float = 0.6
    graspFloor: float = TERMINAL_TRIAL_GRASP_FLOOR
    maxTiltDeg: float = TERMINAL_TRIAL_MAX_TILT_DEG
    # How many times to descend and close before calling a grasp lost. 1 is what this loop did
    # before: one miss ended the night. More than one recovers a finger that closed just off the
    # peg; nothing here recovers a peg that is no longer where it was released, which is a
    # different failure and stays a halt.
    graspAttempts: int = 1
    # Close the fingers again where the peg was released, before retreating, so it is never left
    # standing. False keeps the older cycle (retreat with the hand open, descend again next
    # trial), which is the one that loses a peg every time a trial does not seat.
    regripInPlace: bool = False
    # Keep hold of a peg that did not seat instead of standing it on the face. See
    # `TerminalServoRequest.releaseOnlyWhenSeated`: this is the only one of the three fixes that
    # addresses where the peg is actually lost.
    releaseOnlyWhenSeated: bool = False
    # Close this far below the pose the peg was released at. See
    # `TerminalServoRequest.regripDropM`: a released peg falls, and the fingers have to follow it.
    regripDropM: float = 0.0
    graspToleranceM: float = TERMINAL_TRIAL_GRASP_TOLERANCE
    standingMm: float = TERMINAL_TRIAL_STANDING_MM
    maxReferenceStepM: float = TERMINAL_TRIAL_MAX_REFERENCE_STEP_M
    slipStreak: int = TERMINAL_TRIAL_SLIP_STREAK
    referenceAttempts: int = TERMINAL_TRIAL_REFERENCE_ATTEMPTS
    maxSeconds: float = 0.0
    # Step timing for the fetch and re-grip legs. These are the reset's own numbers; the descent
    # onto the fixture keeps the servo's, which is stricter and lives on `servo`.
    timeoutS: float = 20.0
    toleranceM: float = 0.006
    gripperTolerance: float = 0.08
    controlPeriodS: float = 1.0 / 30.0
    requestId: str = ""

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["servo"] = self.servo.payload()
        return data


def build_trial_schedule(request: TerminalTrialsRequest) -> tuple[TrialSpec, ...]:
    """The order the trials run in, fixed before the arm moves and written into the log.

    Two properties are worth more than they cost. The sweep is *shuffled*, because a monotone
    0-2-4-6-8 sweep confounds the offset axis with anything that drifts over the run -- the
    fixture creeps, and a run whose largest offsets all happen last cannot tell a capture radius
    from an afternoon of creep. And each offset is placed on a *random bearing* rather than a
    fixed axis, because the peg sits off-centre in the jaws mostly along one axis and the hole is
    round: a sweep down one bearing measures the capture radius along that bearing only, and
    anisotropy would be invisible instead of testable.
    """

    if request.repeats < 1:
        raise TerminalTrialError("repeats must be at least 1.")
    if request.controlEvery < 1:
        raise TerminalTrialError("controlEvery must be at least 1.")
    if not request.offsetsMm:
        raise TerminalTrialError("offsetsMm is empty: there is nothing to sweep.")
    for offset in request.offsetsMm:
        if not math.isfinite(offset) or offset < 0.0:
            raise TerminalTrialError(f"offset {offset} must be a finite non-negative millimetre value.")

    rng = random.Random(request.seed)
    sweep = [float(offset) for offset in request.offsetsMm for _ in range(request.repeats)]
    rng.shuffle(sweep)

    specs: list[TrialSpec] = []
    since_reference = request.controlEvery
    for offset in sweep:
        if since_reference >= request.controlEvery:
            specs.append(
                TrialSpec(
                    index=len(specs),
                    kind="reference",
                    offsetMm=0.0,
                    bearingDeg=0.0,
                    searchRingM=request.referenceRingM,
                )
            )
            since_reference = 0
        specs.append(
            TrialSpec(
                index=len(specs),
                kind="offset",
                offsetMm=offset,
                # Drawn here rather than at run time so the schedule in the log is the schedule
                # that ran, and a rerun with the same seed lands on the same bearings.
                bearingDeg=rng.uniform(0.0, 360.0),
                searchRingM=request.searchRingM,
            )
        )
        since_reference += 1
    return tuple(specs)


def classify_stop(
    above_target_mm: float,
    settle_mm: float,
    *,
    seated_mm: float,
    slip_mm: float,
    standing_mm: float,
) -> str:
    """Turn one descent's two numbers into a verdict, in the order the search reads them.

    Creep before depth, which is the whole lesson of 2026-09-10: two descents reached the seated
    depth and were misses, because the peg had slid up between the fingers while the arm leaned.
    Depth alone cannot tell those apart -- 0.7 and 1.4 mm above target, squarely inside the
    seated cluster -- and the column that can is how long the last millimetres took.
    """

    if settle_mm >= slip_mm:
        return "slip"
    if above_target_mm <= seated_mm:
        return "seated"
    if above_target_mm >= standing_mm:
        return "standing"
    return "ambiguous"


def tool_axis_tilt_deg(rotvec: tuple[float, float, float]) -> float:
    """Angle between the tool's own z axis and the table normal, in degrees.

    Rodrigues applied to e3 only, so this module keeps needing nothing but `math`: for a rotation
    vector of angle t about unit axis k, R @ e3 is
    `e3 cos t + (k x e3) sin t + k (k . e3)(1 - cos t)`.
    """

    angle = math.sqrt(sum(value * value for value in rotvec))
    if angle <= 0.0:
        return 0.0
    kx, ky, kz = (value / angle for value in rotvec)
    sin_t, cos_t = math.sin(angle), math.cos(angle)
    axis_z = (
        ky * sin_t + kx * kz * (1.0 - cos_t),
        -kx * sin_t + ky * kz * (1.0 - cos_t),
        cos_t + kz * kz * (1.0 - cos_t),
    )
    # Sign-free: a tool pointing down is as level as one pointing up, and this loop's tool points
    # down. What is being asked is how far the axis leans, not which way along it the peg sits.
    return math.degrees(math.acos(min(1.0, abs(axis_z[2]))))


def assert_tool_is_level(rotvec: tuple[float, float, float], request: TerminalTrialsRequest) -> float:
    """Refuse a run whose tool is leaning, rather than let it measure a jam depth.

    This is a refusal and not a correction because correcting it means moving the wrist, and the
    pose that is already known to be level is the home keyframe -- so the remedy names it.
    """

    tilt = tool_axis_tilt_deg(rotvec)
    if request.maxTiltDeg > 0.0 and tilt > request.maxTiltDeg:
        raise TerminalTrialError(
            f"tool leans {tilt:.2f} deg off the table normal, over the {request.maxTiltDeg:.2f} deg "
            f"limit: a peg at this angle wedges at about "
            f"{5.0 / math.tan(math.radians(tilt)):.0f} mm of engagement and every verdict becomes "
            f"a jam depth. Home the arm first (the home keyframe is level), or pass "
            f"--max-tilt-deg 0 to measure deliberately."
        )
    return tilt


def _grasp_verdict(width: float, reference: float | None, request: TerminalTrialsRequest) -> str:
    if width < request.graspFloor:
        return "empty"
    if reference is not None and abs(width - reference) > request.graspToleranceM:
        return "changed"
    return "held"


def _grasp_until_held(
    robot: Any,
    request: TerminalTrialsRequest,
    at_xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    reference: float | None,
) -> tuple[float, str, int]:
    """Descend, close and lift until the fingers hold something, up to `graspAttempts` times.

    A retry answers one failure and not the other. Fingers that closed a millimetre off the peg
    will often take it on the second try, because the descent re-approaches from above and the
    peg has not moved. A peg that fell over is somewhere this pose no longer describes, and no
    number of retries at the same pose finds it -- that one has to stay a halt, because the
    alternative is a loop that keeps closing on empty air all night and calls it a run.
    """

    attempts = max(1, int(request.graspAttempts))
    width = 0.0
    verdict = "empty"
    for attempt in range(1, attempts + 1):
        width = _close_and_lift(robot, request, at_xyz, rotvec)
        verdict = _grasp_verdict(width, reference, request)
        if verdict == "held":
            return width, verdict, attempt
    return width, verdict, attempts


def _hold_z(request: TerminalTrialsRequest) -> float:
    """The height the peg is carried at between trials: the servo's own retreat."""

    return request.servo.xyz[2] + request.servo.retreatM


def _close_and_lift(
    robot: Any,
    request: TerminalTrialsRequest,
    at_xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
) -> float:
    """Descend onto the peg where it is, close, lift, and answer what the fingers measured.

    The step names are the reset's on purpose rather than by copying: `_run_step` keys its
    gripper-wait policy on them, so `close_gripper` gets the grasp-settle wait that a clamped
    peg needs (the measured opening becomes the peg's thickness and never reaches the closed
    command) and `lift_8cm_after_grasp` is allowed to move while the fingers hold something.
    """

    _run_step(robot, request, "descend_8cm_to_pick", at_xyz, rotvec, request.openGripper)
    _run_step(robot, request, "close_gripper", at_xyz, rotvec, request.closedGripper)
    _, _, width = _observation_xyz_rotvec_gripper(robot)
    _run_step(
        robot,
        request,
        "lift_8cm_after_grasp",
        (at_xyz[0], at_xyz[1], _hold_z(request)),
        rotvec,
        request.closedGripper,
    )
    return float(width)


def _park(robot: Any, request: TerminalTrialsRequest) -> None:
    """Leave the arm at carrying height wherever it is, still holding whatever it holds.

    Deliberately not "open the fingers and go home". A run that stopped because it could not
    read its own state is the worst moment to drop a peg on a fixture nobody is watching.
    """

    xyz, rotvec, gripper = _observation_xyz_rotvec_gripper(robot)
    hold_z = _hold_z(request)
    if xyz[2] >= hold_z:
        return
    _run_step(robot, request, "lift_8cm_after_grasp", (xyz[0], xyz[1], hold_z), rotvec, gripper)


def validate_terminal_trials(
    request: TerminalTrialsRequest,
    schedule: Iterable[TrialSpec],
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    reach_probe: Any = None,
) -> dict[str, Any]:
    """Check every pose the whole run may command, before the first one is sent.

    A single trial is validated again by `execute_terminal_servo` when it runs, which is the
    check that protects the arm. This one protects the *run*: an unattended sweep that discovers
    its widest offset is outside the fence after ninety trials has spent the night proving that
    the fence is where it always was. Every offset in the schedule is walked here, at the hole
    estimate it will be aimed from -- which is the estimate at the start, so the margin left for
    the reference trials to move it is reported rather than assumed.
    """

    specs = tuple(schedule)
    if not specs:
        raise TerminalTrialError("the schedule is empty.")
    if request.graspFloor <= 0.0 or request.graspToleranceM <= 0.0:
        raise TerminalTrialError("graspFloor and graspToleranceM must be positive.")
    if request.standingMm <= request.servo.searchSeatedM * 1000.0:
        raise TerminalTrialError(
            f"standingMm {request.standingMm:.1f} must be above the seated threshold "
            f"{request.servo.searchSeatedM * 1000.0:.1f} mm, or no stop can be ambiguous and the "
            "two clusters are being asserted to touch."
        )
    if request.maxReferenceStepM <= 0.0:
        raise TerminalTrialError("maxReferenceStepM must be positive.")
    if request.referenceRingM <= 0.0:
        raise TerminalTrialError(
            "referenceRingM must be positive: a reference trial has to be able to find the hole "
            "when the peg's seating bias is unknown, and that is what the search is for."
        )

    low, high = _workspace_bounds(workspace_min, workspace_max)
    checked = 0
    seen: set[tuple[float, float, float]] = set()
    for spec in specs:
        dx, dy = spec.offset_m()
        servo = _servo_for(request, spec, request.servo.xyz)
        validate_terminal_servo_trajectory(
            servo,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            reach_probe=reach_probe,
        )
        checked += 1
        seen.add((round(dx, 6), round(dy, 6), spec.searchRingM))
    # The fetch leg is not part of any servo request, so it is checked here or nowhere.
    if request.pickXyz is not None:
        _check_xyz_in_workspace(request.pickXyz, "pickXyz", low, high)
        _check_xyz_in_workspace(
            (request.pickXyz[0], request.pickXyz[1], request.pickXyz[2] + request.liftM),
            "pickXyz+lift",
            low,
            high,
        )
    widest = max(spec.offsetMm for spec in specs)
    reach_mm = widest + 1000.0 * max(
        (spec.searchRingM for spec in specs), default=0.0
    )
    return {
        "ok": True,
        "trials": len(specs),
        "distinctAims": len(seen),
        "widestOffsetMm": widest,
        # How far from the starting estimate the run can reach at all. A reference trial that
        # wanted to move the estimate further than the fence allows would fail its own QC at that
        # point, and this is the number that says how much room it has before it does.
        "widestCommandedMm": reach_mm,
    }


def _servo_for(
    request: TerminalTrialsRequest,
    spec: TrialSpec,
    reference_xyz: tuple[float, float, float],
) -> TerminalServoRequest:
    dx, dy = spec.offset_m()
    return replace(
        request.servo,
        xyz=(reference_xyz[0] + dx, reference_xyz[1] + dy, reference_xyz[2]),
        searchRingM=spec.searchRingM,
        regripGripper=request.closedGripper if request.regripInPlace else None,
        releaseOnlyWhenSeated=request.releaseOnlyWhenSeated,
        regripDropM=request.regripDropM,
        graspSettleS=request.graspSettleS,
        requestId=f"{request.requestId or 'terminal_trials'}#{spec.index:03d}",
    )


@dataclass
class TrialLoopState:
    referenceXyz: tuple[float, float, float]
    graspReference: float | None = None
    heldWidth: float = 0.0
    slipRun: int = 0
    referenceFailures: int = 0
    seated: int = 0
    completed: int = 0
    referenceUpdates: list[dict[str, Any]] = field(default_factory=list)


def run_terminal_trials(
    robot: Any,
    request: TerminalTrialsRequest,
    schedule: Iterable[TrialSpec] | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    reference_xyz: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Run the whole sweep and answer with what happened and why it stopped.

    The cycle is: aim at the current hole estimate plus this trial's offset, descend, read the
    verdict off the descent, pick the peg back up from the pose the servo let go at, repeat. That
    last step is what makes the loop closed and it needs no perception, because the servo already
    reports where it opened the fingers -- the peg is standing at that XY whether it went into
    the hole or stopped on the face, and returning the tool there is returning it to a pose the
    fingers occupied a second earlier.

    Re-gripping in place is also what keeps the commanded offset meaningful. The peg comes back
    into the jaws at the pose it was released from, so the seating bias the next trial inherits
    is the one the last reference trial established against the hole itself, not one accumulated
    over the night.
    """

    specs = tuple(schedule) if schedule is not None else build_trial_schedule(request)
    _, preflight_rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    # Before the fence check, because a level tool is a precondition of the experiment and not a
    # property of the poses it will visit: a leaning run passes every reach and fence test and
    # still measures the wrong thing.
    preflight_tilt_deg = assert_tool_is_level(preflight_rotvec, request)
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    # Run-level QC lives here rather than in the caller so that no entry point can skip it. It
    # raises, and an unattended run that cannot reach its widest offset should never have been
    # started rather than discovered halfway through.
    validate_terminal_trials(
        request,
        specs,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        reach_probe=_reach_probe(robot, preflight_rotvec),
    )
    # A resumed run inherits the hole estimate its earlier half paid for. Starting again from
    # the command line's nominal pose would throw away every reference trial already run and
    # silently aim the rest of the schedule somewhere else.
    state = TrialLoopState(
        referenceXyz=tuple(
            float(value)
            for value in (reference_xyz if reference_xyz is not None else request.servo.xyz)
        )
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    halted = ""

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    try:
        _, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
        start_attempts = 1
        if request.pickXyz is not None:
            width, verdict, start_attempts = _grasp_until_held(
                robot, request, tuple(request.pickXyz), rotvec, None
            )
        else:
            _, _, width = _observation_xyz_rotvec_gripper(robot)
            verdict = _grasp_verdict(width, None, request)
        state.graspReference = width
        state.heldWidth = width
        emit(
            {
                "kind": "grasp",
                "stage": "start",
                "widthNormalized": width,
                "graspVerdict": verdict,
                "attempts": start_attempts,
                "pickXyz": list(request.pickXyz) if request.pickXyz is not None else None,
            }
        )
        if verdict != "held":
            halted = "grasp_lost_at_start"
            raise _Halt

        # A list with a cursor rather than a plain iteration, because a reference trial that
        # could not find the hole has to be retried *before* anything is aimed at the estimate it
        # failed to confirm. Running the next offset first would spend a datum on a target the
        # loop already has reason to doubt.
        pending = list(specs)
        cursor = 0
        while cursor < len(pending):
            spec = pending[cursor]
            cursor += 1
            if request.maxSeconds > 0.0 and time.perf_counter() - started >= request.maxSeconds:
                halted = "time_budget"
                break
            # Checked between trials, not inside a descent. A run stopped mid-descent leaves the
            # peg somewhere the next trial cannot re-grip it from, and the whole point of the loop
            # closing is that it ends every trial holding the peg. SIGINT remains the hard stop
            # for when that trade is the wrong way round.
            if should_stop is not None and should_stop():
                halted = "stop_requested"
                break

            _, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
            servo = _servo_for(request, spec, state.referenceXyz)
            result = execute_terminal_servo(robot, servo)
            if not result.get("ok"):
                emit(
                    {
                        "kind": "trial",
                        "index": spec.index,
                        "trialKind": spec.kind,
                        "ok": False,
                        "error": result.get("error"),
                        # The offset this trial *was* is known here and nowhere later. Without it
                        # a night's failures cannot be asked the one question worth asking of
                        # them -- whether they cluster at an offset -- and the by-offset summary
                        # reads the field whether the trial finished or not.
                        "offsetMm": spec.offsetMm,
                        "bearingDeg": spec.bearingDeg,
                        "searchRingM": spec.searchRingM,
                        "referenceXyz": list(state.referenceXyz),
                        "aimXyz": list(servo.xyz),
                    }
                )
                halted = "servo_failed"
                break

            above_target_mm = float(result["seatedDepthErrorMm"])
            settle_mm = float(result["settleMm"])
            verdict = classify_stop(
                above_target_mm,
                settle_mm,
                seated_mm=servo.searchSeatedM * 1000.0,
                slip_mm=servo.searchSlipM * 1000.0,
                standing_mm=request.standingMm,
            )
            release_xyz = (
                float(result["searchLandingXyz"][0]),
                float(result["searchLandingXyz"][1]),
                float(result["stoppedAtXyz"][2]),
            )
            row: dict[str, Any] = {
                "kind": "trial",
                "index": spec.index,
                "trialKind": spec.kind,
                "ok": True,
                "offsetMm": spec.offsetMm,
                "bearingDeg": spec.bearingDeg,
                "searchRingM": spec.searchRingM,
                "referenceXyz": list(state.referenceXyz),
                "aimXyz": list(servo.xyz),
                "verdict": verdict,
                "aboveTargetMm": above_target_mm,
                "settleMm": settle_mm,
                "settleSeconds": float(result["settleSeconds"]),
                "lateralErrorMm": float(result["lateralErrorMm"]),
                "lagMm": float(result["lagMm"]),
                "peakGrowthMm": float(result["peakGrowthMm"]),
                "descentMm": float(result["descentMm"]),
                "descentSeconds": float(result["descentSeconds"]),
                "stoppedOn": result["stoppedOn"],
                "stoppedAtXyz": [float(value) for value in result["stoppedAtXyz"]],
                "searchStoppedOn": result["searchStoppedOn"],
                "searchIndex": int(result["searchIndex"]),
                "searchLandings": int(result["searchLandings"]),
                "searchAttempts": result.get("searchAttempts", []),
                "releaseXyz": list(release_xyz),
                "handoffXyz": [float(value) for value in result["handoffXyz"]],
                "elapsedS": time.perf_counter() - started,
            }

            state.completed += 1
            if verdict == "seated":
                state.seated += 1
            state.slipRun = state.slipRun + 1 if verdict == "slip" else 0

            # A reference trial exists to re-read the hole, so its seating event is where the
            # estimate comes from. The tool is held laterally by the seated peg, so where it
            # actually stopped is a measurement of the hole plus whatever bias the peg carries
            # in the fingers -- and since the peg is about to be re-gripped at exactly that
            # pose, that sum is precisely the pose the next trial should aim at.
            if spec.kind == "reference":
                if verdict == "seated":
                    state.referenceFailures = 0
                    # z as well as xy, for the same reason the comment above gives for xy:
                    # a seated reference stopped where the hole is, and carrying the typed-in z
                    # forward instead left every `aboveTargetMm` in the run offset by whatever
                    # that number was wrong by. Measured 2026-09-11: seven seated trials read
                    # 1.50-2.03 mm "above target" with a spread of half a millimetre -- a
                    # constant, i.e. the estimate, not the peg. With a 3 mm seated threshold
                    # that left one millimetre of margin on a two millimetre bias.
                    proposed = (
                        float(result["stoppedAtXyz"][0]),
                        float(result["stoppedAtXyz"][1]),
                        float(result["stoppedAtXyz"][2]),
                    )
                    step_m = math.hypot(
                        proposed[0] - state.referenceXyz[0], proposed[1] - state.referenceXyz[1]
                    )
                    row["referenceStepMm"] = 1000.0 * step_m
                    # Reported and bounded separately: the lateral step is the hole moving, which
                    # is the question `hole_stability` asks and what `maxReferenceStepM` was
                    # sized for. A vertical step is the estimate being corrected, a different
                    # thing that would corrupt that statistic if it were folded into the same
                    # number -- but it gets the same bound, because a large one is not a
                    # correction either.
                    step_z_m = abs(proposed[2] - state.referenceXyz[2])
                    row["referenceStepZMm"] = 1000.0 * step_z_m
                    if step_z_m > request.maxReferenceStepM:
                        row["referenceAccepted"] = False
                        emit(row)
                        halted = "reference_step_too_large"
                        break
                    if step_m > request.maxReferenceStepM:
                        row["referenceAccepted"] = False
                        emit(row)
                        halted = "reference_step_too_large"
                        break
                    row["referenceAccepted"] = True
                    state.referenceUpdates.append(
                        {
                            "index": spec.index,
                            "fromXyz": list(state.referenceXyz),
                            "toXyz": list(proposed),
                            "stepMm": 1000.0 * step_m,
                            "elapsedS": row["elapsedS"],
                        }
                    )
                    state.referenceXyz = proposed
                else:
                    state.referenceFailures += 1
                    row["referenceAccepted"] = False
                    row["referenceFailures"] = state.referenceFailures

            emit(row)

            if spec.kind == "reference" and verdict != "seated":
                if state.referenceFailures >= request.referenceAttempts:
                    halted = "reference_lost"
                    break
                pending.insert(cursor, spec)
            if state.slipRun >= request.slipStreak:
                halted = "slip_streak"
                break

            if not result.get("released", True):
                # The servo never let go, so the peg is where it always was: in the fingers.
                _, _, width = _observation_xyz_rotvec_gripper(robot)
                grasp = _grasp_verdict(width, state.graspReference, request)
                attempts = 0
            elif request.regripInPlace:
                # The servo closed on the peg at the pose it released it, so there is nothing to
                # descend to: the peg is already in the fingers and re-approaching would be a
                # second grasp of something being held.
                _, _, width = _observation_xyz_rotvec_gripper(robot)
                grasp = _grasp_verdict(width, state.graspReference, request)
                attempts = 1
            else:
                # Same reason as the servo's own re-grip: the peg fell when the fingers
                # opened, so the fingers have to go after it.
                fetch_xyz = (release_xyz[0], release_xyz[1],
                             max(release_xyz[2] - request.regripDropM, request.servo.minZ))
                width, grasp, attempts = _grasp_until_held(robot, request, fetch_xyz, rotvec,
                                                           state.graspReference)
            state.heldWidth = width
            emit(
                {
                    "kind": "grasp",
                    "stage": "regrip",
                    "index": spec.index,
                    "widthNormalized": width,
                    "referenceWidth": state.graspReference,
                    "graspVerdict": grasp,
                    "attempts": attempts,
                    "atXyz": list(release_xyz),
                }
            )
            if grasp != "held":
                halted = f"grasp_{grasp}"
                break
        else:
            halted = "schedule_complete"
    except _Halt:
        pass
    except KeyboardInterrupt:
        halted = "interrupted"
    except SceneResetError as exc:
        halted = f"step_failed: {exc}"
    except (TimeoutError, RuntimeError) as exc:
        halted = f"step_failed: {exc}"

    try:
        _park(robot, request)
        parked = True
    except Exception as exc:  # noqa: BLE001 - the summary has to survive a failed park
        parked = False
        print(f"[WARN] terminal_trials=park_failed details={exc}", flush=True)

    trials = [row for row in rows if row.get("kind") == "trial" and row.get("ok")]
    summary = {
        "kind": "summary",
        # A requested stop did what it was told; logging it as a failure would make every
        # deliberate overnight stop look like a fault.
        "ok": halted in {"schedule_complete", "time_budget", "stop_requested"},
        "haltedOn": halted or "schedule_complete",
        "parked": parked,
        "trials": len(trials),
        "seated": state.seated,
        "scheduled": len(specs),
        # Recorded, not merely checked. Two runs that disagree are worth being able to ask this
        # of afterwards, and it is not recoverable from anything else in the file.
        "toolTiltDeg": preflight_tilt_deg,
        "referenceXyz": list(state.referenceXyz),
        "referenceUpdates": state.referenceUpdates,
        "graspReference": state.graspReference,
        "heldWidth": state.heldWidth,
        "elapsedS": time.perf_counter() - started,
        "byOffsetMm": summarize_by_offset(trials),
    }
    if on_row is not None:
        on_row(summary)
    summary["rows"] = rows
    return summary


class _Halt(Exception):
    """Internal: leave the loop through the same exit as every other stop."""


def summarize_by_offset(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """p(seat | offset), which is what the run was for.

    Reference trials are excluded: they carry the search and are aimed at zero by construction,
    so counting them would fold the thing being measured into its own control. Slips and
    ambiguous stops are counted separately rather than as misses -- a peg that moved in the jaws
    says nothing about whether that offset is inside the capture radius.
    """

    buckets: dict[float, dict[str, Any]] = {}
    for row in rows:
        if row.get("trialKind") != "offset" or "offsetMm" not in row:
            continue
        bucket = buckets.setdefault(
            float(row["offsetMm"]),
            {"offsetMm": float(row["offsetMm"]), "n": 0, "seated": 0, "standing": 0, "slip": 0,
             "ambiguous": 0, "failed": 0},
        )
        verdict = row.get("verdict")
        if verdict is None:
            # A trial that stopped before it could be classified measured nothing. Counting it in
            # `n` would put it in the denominator of a seated fraction it never got to answer;
            # dropping it silently would hide that the offset is short of the repeats it was
            # scheduled for. It gets its own column.
            bucket["failed"] += 1
            continue
        bucket["n"] += 1
        bucket[verdict] = bucket.get(verdict, 0) + 1
    for bucket in buckets.values():
        decided = bucket["seated"] + bucket["standing"]
        bucket["decided"] = decided
        bucket["seatedFraction"] = bucket["seated"] / decided if decided else None
    return [buckets[key] for key in sorted(buckets)]


def resume_state(rows: Iterable[dict[str, Any]]) -> tuple[set[int], tuple[float, float, float] | None]:
    """What an interrupted run already answered: which trials are done, and where the hole is.

    A trial counts as done only if it produced a verdict. One that failed part way answered
    nothing and is worth running again -- and it is exactly the kind that stopped the run, so
    dropping it would quietly shrink the schedule every time a night was resumed.

    The hole estimate is read from the last row that carried one, so a resumed run keeps the
    reference its earlier half established instead of starting over from the nominal pose.
    """

    done: set[int] = set()
    reference: tuple[float, float, float] | None = None
    for row in rows:
        kind = row.get("kind")
        if kind == "trial":
            if row.get("ok") and row.get("verdict"):
                done.add(int(row["index"]))
            if row.get("referenceAccepted") and row.get("referenceXyz"):
                reference = tuple(float(v) for v in row["referenceXyz"])
        elif kind == "summary" and row.get("referenceXyz"):
            reference = tuple(float(v) for v in row["referenceXyz"])
    return done, reference


def describe_schedule(request: TerminalTrialsRequest, schedule: Iterable[TrialSpec]) -> str:
    """The run as text, for reading before it is started rather than after it has finished."""

    specs = tuple(schedule)
    offsets = sorted({spec.offsetMm for spec in specs if spec.kind == "offset"})
    references = sum(1 for spec in specs if spec.kind == "reference")
    landings = len(terminal_servo_search_offsets(replace(request.servo, searchRingM=request.searchRingM)))
    lines = [
        f"trials={len(specs)} offset_trials={len(specs) - references} reference_trials={references}",
        f"offsets_mm={','.join(f'{value:g}' for value in offsets)} repeats={request.repeats} seed={request.seed}",
        f"aim={request.servo.xyz[0]:+.4f},{request.servo.xyz[1]:+.4f},{request.servo.xyz[2]:+.4f} "
        f"handoff_z={request.servo.handoffZ:.4f}",
        f"offset_search_ring_m={request.searchRingM:.4f} landings_per_offset_trial={landings} "
        f"reference_search_ring_m={request.referenceRingM:.4f}",
        f"control_every={request.controlEvery} slip_streak={request.slipStreak} "
        f"max_reference_step_mm={1000.0 * request.maxReferenceStepM:.1f}",
    ]
    for spec in specs:
        lines.append(
            f"  [{spec.index:03d}] {spec.kind:<9} offset_mm={spec.offsetMm:5.1f} "
            f"bearing_deg={spec.bearingDeg:6.1f} ring_m={spec.searchRingM:.4f}"
        )
    return "\n".join(lines)
