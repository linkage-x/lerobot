"""E5's terminal controller: hand the last centimetres to a fixed pose and see what happens.

The policy is left to fly the approach, and at a chosen height the arm is taken off it and
driven to one absolute pose read off the demonstrations. That is the whole experiment. It is
worth running because its prediction is already written down and it is cheap to refute: the 34
demonstrations that released at the seated depth scatter a median 3.5 mm and a p90 8.1 mm about
their own mean, 74% of them beyond the 2.5 mm radial clearance, so driving the tool to that mean
should mostly miss. If it does not miss -- if a fixed pose inserts reliably -- then tool pose
does determine where the peg tip is, and the terminal problem is a controller rather than a
perception one.

Two things here are not in `scene_reset`, and they are why this is a module rather than another
`PoseProbeRequest`:

  * The descent is the only motion on this rig that drives a gripped peg down onto the fixture
    on purpose. A probe descends to a point in free space; this one expects to be stopped by
    something. So the descent watches how far the tool is above its own setpoint and gives up
    when the arm stays held up, instead of leaning on the fixture until the step times out.

  * That stopping height is the measurement. A peg standing on the fixture's face and a peg
    seated in the hole are the same XY and different z, so the height the descent is refused at
    is the insertion verdict -- reported, not thresholded, because the separation between the two
    belongs in the analysis rather than hard-coded here.

2026-09-10 added the search, and the reason is that the experiment above came back the other
way. A fixed pose inserted 5 of 8, not the predicted 24%, which puts the capture radius at
4.2 mm rather than the 2.5 mm nominal clearance and turns what is left into a covering problem
rather than a perception one: the tip lands 6-8 mm off the hole and nothing on this arm knows
it. `search_for_seat` is E7 route C -- try the nominal pose, and on a miss lift, step to the
next landing on a ring, descend again. It is off by default, because that 5 of 8 is the control
arm every reading of the search is compared against.

Everything else -- the speed clamp, the workspace checks, the step loop -- is imported from
`scene_reset` rather than copied. A second way of walking the arm to a typed-in coordinate is a
second set of safety checks to keep in step, and the one that gets skipped is always the one
nobody thought of as a real motion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Iterable

from tools.fr3.scene_reset import (
    ControlLoopDiedError,
    SceneResetError,
    _ForceTrace,
    _check_reach_along_path,
    _check_xyz_in_workspace,
    _distance,
    _iter_segment_samples,
    _observation_xyz_rotvec_gripper,
    _reach_probe,
    _robot_workspace_bounds,
    _run_step,
    _send_absolute,
    _step_toward,
    _workspace_bounds,
    control_loop_alive,
    precise_sleep,
    read_fz,
)

# How fast the commanded tool point may travel, in m/s. A third of the reset's, because the
# reset's rate limit exists to keep a transfer from crossing the table at a metre a second and
# this one exists to keep a 15 mm peg from being driven into a fixture face.
TERMINAL_SERVO_MAX_SPEED_MS = 0.02
# How much *further* behind its own setpoint the tool has to fall before the descent calls it
# contact, in metres. Growth, not absolute lag: the arm answers a streamed setpoint about three
# control periods late, so a free descent runs a constant speed-times-delay behind it, and a
# threshold on the absolute figure fires on the descent itself. It did. All eight runs of
# 2026-09-10 stopped on "contact" about 15 mm into a 68 mm descent with `held_up_mm` sitting at
# 5.2-5.9 the whole time, which is an arm moving three steps late, not an arm blocked: a blocked
# tool falls a further full step behind every step, so its lag would have passed 19 mm before
# the hold below elapsed. A constant lag has zero growth, at any speed, with any delay.
TERMINAL_SERVO_CONTACT_LAG_M = 0.004
# For how long, in seconds.
TERMINAL_SERVO_CONTACT_HOLD_S = 0.3
# The commanded descent that growth is measured across, in metres. A distance rather than a time
# so it means the same thing at any speed: across this much setpoint travel a blocked tool falls
# the whole window behind, and a tool that is merely late falls no further behind at all.
TERMINAL_SERVO_CONTACT_WINDOW_M = 0.006
# How much of the descent runs before the growth test is armed, in metres, measured to the far
# end of the window rather than the near one. The setpoint starts from rest, so the lag climbs
# from zero to its steady value over the first few millimetres, and that climb is growth: the
# window has to be clear of it entirely. Costs a blind 8 + 6 mm below the handoff height, which
# is chosen to have nothing under it -- and `contactStallM` below is not blinded at all.
TERMINAL_SERVO_CONTACT_WARMUP_M = 0.008
# The absolute lag that is contact whatever the growth test says, in metres, with no hold and no
# warm-up: several times the free-descent lag at the speed above. It backstops the two cases the
# growth test can miss, a tool already against the fixture when the descent starts and one that
# gives way slowly enough to stay under the growth threshold.
TERMINAL_SERVO_CONTACT_STALL_M = 0.015
# How long the descent may keep commanding the target after the setpoint has arrived there, in
# seconds, before it gives up and reports where the tool actually is. Measured, not guessed: on
# 2026-09-10 the three descents that met the depth met it 0.1 s after the setpoint parked, so
# this is five times what a free arm needs. What it replaces is the full `timeoutS`, which left
# the arm leaning on the peg for twenty seconds -- the operator wrote that down twice -- and
# leaning is not passive here. The setpoint parks *below* whatever stopped the peg, so the
# position error, and with it the force, is held until the softest thing in the chain gives. On
# this rig that is the grip: the two runs graded as misses took 8.1 s and 18.1 s to close their
# last millimetres, which is a peg sliding up between the fingers, not an arm settling.
TERMINAL_SERVO_SETTLE_S = 0.5
# E7-C. How far off the nominal pose the search ring sits, in metres, and how many landings on
# it. Zero disables the search entirely and the module behaves exactly as E5 measured it, which
# is the point: the search is one variable added to a control arm that has a number on it.
# Nine landings: the nominal pose and eight on the ring. The count is set by the covering, not
# picked -- a tip offset r on the worst bearing (halfway between two ring points) is
# sqrt(R^2 + r^2 - 2Rr cos(pi/n)) from the nearest landing, and that has to stay inside the
# 4.2 mm capture radius measured on 2026-09-10. At R = 7 mm, six landings cover only to 8.4 mm
# and eight cover to 9.7 mm, against a demonstration scatter whose p90 is 8.7 mm. Eight also
# keeps the *inner* gap honest: the ring's coverage starts at 3.2 mm and the nominal landing
# reaches 4.2 mm, so there is no annulus between them that nothing can reach -- which is what
# widening the ring instead of adding landings would have opened up.
TERMINAL_SERVO_SEARCH_RING_M = 0.007
TERMINAL_SERVO_SEARCH_POINTS = 8
# How far the arm lifts between landings, in metres. Not a clearance figure -- 8 mm would clear
# a peg standing on the fixture face -- but a run-up: the growth test below arms only after
# `contactWarmupM + contactWindowM` of commanded descent, so a lift shorter than that leaves
# every landing after the first with no contact detection at all, stopping instead on the settle
# timeout with the arm leaning on the peg. Validated, not just documented.
TERMINAL_SERVO_SEARCH_LIFT_M = 0.020
# How close to the target depth counts as in the hole, in metres, and so ends the search. The
# two clusters on 2026-09-10 do not overlap: seated stopped 1.1-2.5 mm above the target, standing
# on the face 4.9-5.6 mm. This sits between them.
TERMINAL_SERVO_SEARCH_SEATED_M = 0.003
# Post-arrival creep that ends the search as a slip rather than a miss, in metres. Once the peg
# has moved in the jaws every landing computed from the tool pose is aimed at the wrong place,
# so continuing the search is searching with a ruler that changed length. The two runs graded as
# slips crept for 8.1 s and 18.1 s; a seated one closed in 0.1 s.
TERMINAL_SERVO_SEARCH_SLIP_M = 0.003
# How far above the target the arm lifts once it has let go, in metres. Enough to clear a
# standing peg before the waiting loop homes the arm across the table.
TERMINAL_SERVO_RETREAT_M = 0.08
# The lowest z a target may name. The seated release is 0.052; anything materially below that is
# a typo or a stale calibration, and this motion has no force control to survive one.
TERMINAL_SERVO_MIN_Z_M = 0.030


def classify_terminal_servo_descent(
    *,
    above_target_mm: float,
    settle_mm: float,
    seated_mm: float = 1000.0 * TERMINAL_SERVO_SEARCH_SEATED_M,
    slip_mm: float = 1000.0 * TERMINAL_SERVO_SEARCH_SLIP_M,
    standing_mm: float | None = None,
) -> str:
    """What one descent says happened to the peg: ``slip``, ``seated`` or ``standing``.

    The rule the search loop already applied and `terminal_trials.classify_stop` had a second
    copy of, in the lowest module the three callers share. Two copies of a threshold are two
    thresholds, and these two would have drifted the first time either was retuned.

    ``standing_mm`` opens a fourth answer. The search does not want one -- it is deciding
    whether to fly to the next landing, and "not seated" is the whole of what it needs -- so it
    passes nothing and everything above ``seated_mm`` reads as standing. An offset sweep does
    want one: its job is to find where the boundary is, and a descent that stopped between the
    two clusters is evidence that the boundary is there rather than evidence of either side, so
    it names that band ``ambiguous`` instead of rounding it into a verdict.

    Creep is tested before depth, and that order is the whole lesson of 2026-09-10: two runs
    reached the seated depth and were graded misses because the peg had slid up between the
    fingers while the arm leaned, so the tool arrived where the hole is and the peg did not.

    The 2026-09-10 card lists *four* signatures; this returns three verdicts. Its first two --
    "converged inside 0.1 s" and "never converged but stopped shallow and laterally held" --
    were both insertions and both graded success, so they are one verdict here and a reader who
    wants them apart has ``settleSeconds``. What the card cannot be ported from directly is its
    time axis: it measured slips at 8.1 s and 18.1 s against a settle window that was still the
    full 20 s timeout, and that window is now ``TERMINAL_SERVO_SETTLE_S`` = 0.5 s. A slip today
    cannot show up as a long settle because nothing is allowed to settle that long -- it shows
    up as distance travelled after the setpoint parked, which is what ``settleMm`` measures and
    why the threshold here is a length and not a duration.

    Provisional, and it should be read as such: the four signatures were fitted on nine descents
    and matched the operator's grade 9/9, which is agreement on the set they were drawn from,
    not validation. The reason to write them down is that the next batch then *tests* this
    function instead of re-deriving it -- which is also why the GUI keeps the operator's grade in
    its own column rather than letting this one stand in for it.
    """

    # Millimetres on both sides of both comparisons. The thresholds are declared in metres and
    # scaled here rather than the measurements being divided back down, because scaling both
    # sides by the same factor cannot change which way a comparison goes and a round trip
    # through the other unit can: 1000 * 0.0042 / 1000 is not 0.0042.
    if settle_mm >= slip_mm:
        return "slip"
    if above_target_mm <= seated_mm:
        return "seated"
    if standing_mm is None or above_target_mm >= standing_mm:
        return "standing"
    return "ambiguous"


class TerminalServoError(SceneResetError):
    """A terminal-servo request that must not be sent to the arm."""


@dataclass(frozen=True)
class TerminalServoRequest:
    # Where the demonstrations let go. Not a default: the pose is an experimental parameter and
    # belongs on the command line, where it is visible in the log of the run that used it.
    xyz: tuple[float, float, float]
    # The z at which the policy is taken off the task. Below the height band E4 is about and
    # above the seated depth, so the servo owns the descent and nothing else.
    handoffZ: float = 0.12
    # What the gripper is opened to once the descent has stopped. The reset's own release value,
    # in the robot's normalization, deliberately not the demonstrations' median release of 0.865:
    # that figure is read off the dataset's gripper column, which goes through
    # `denormalize_live_gripper_observation` to get here, and a release that lands a few percent
    # short is a peg still pinched at the moment the arm retreats.
    openGripper: float = 1.0
    openSettleS: float = 0.8
    maxSpeedMs: float = TERMINAL_SERVO_MAX_SPEED_MS
    contactLagM: float = TERMINAL_SERVO_CONTACT_LAG_M
    contactHoldS: float = TERMINAL_SERVO_CONTACT_HOLD_S
    contactWindowM: float = TERMINAL_SERVO_CONTACT_WINDOW_M
    contactWarmupM: float = TERMINAL_SERVO_CONTACT_WARMUP_M
    contactStallM: float = TERMINAL_SERVO_CONTACT_STALL_M
    settleS: float = TERMINAL_SERVO_SETTLE_S
    # Off by default. E5's 5/8 is the control arm this is measured against, and a search that
    # arrives switched on turns every later run into a different experiment than that one.
    searchRingM: float = 0.0
    searchPoints: int = TERMINAL_SERVO_SEARCH_POINTS
    searchLiftM: float = TERMINAL_SERVO_SEARCH_LIFT_M
    searchSeatedM: float = TERMINAL_SERVO_SEARCH_SEATED_M
    searchSlipM: float = TERMINAL_SERVO_SEARCH_SLIP_M
    minZ: float = TERMINAL_SERVO_MIN_Z_M
    retreatM: float = TERMINAL_SERVO_RETREAT_M
    timeoutS: float = 20.0
    toleranceM: float = 0.002
    # What the *positioning* steps have to reach. None keeps `toleranceM`, which is where this
    # started and what the descent's own "reached target z" test keeps using. They were split
    # because this arm's residual is a direction-dependent dead-band of about 2 mm, so 2.0 mm
    # aborts runs on steps that do not need it -- while loosening the shared number would let a
    # descent call `target` 4 mm high and reclassify a seated peg (`searchSeatedM` is 3 mm).
    stepToleranceM: float | None = None
    # Close the fingers again on the peg, at the pose it was just released at, before retreating.
    # None retreats with the hand open, which is what this did first and which leaves the peg
    # standing alone for the retreat plus the next descent. Measured 2026-09-11: a peg left
    # standing off-centre -- which is every trial that does not seat, i.e. the half of the sweep
    # the sweep exists to produce -- falls over, and a fallen peg ends the run. Nothing measured
    # changes: the verdict is read from the descent (`aboveTargetMm`, `settleMm`), both of which
    # are already decided before the fingers open, so the release still happens and still means
    # what it meant.
    regripGripper: float | None = None
    # Let go only when the peg is in the hole. Measured 2026-09-11, three runs in a row: a peg
    # that stopped on the face stands on the rim off-centre, and it falls at the *instant* the
    # fingers open -- a re-close at the same pose, milliseconds later, read 0.076 and caught
    # nothing. So closing the gap between release and re-grip cannot help; there is no gap.
    # Nothing measured is lost by not releasing: the verdict is read off the descent
    # (`aboveTargetMm`, `settleMm`), both decided before the fingers move.
    releaseOnlyWhenSeated: bool = False
    # How long a re-close may take to settle before the step gives up on the fingers reaching the
    # closed command. Needed only because `regripGripper` makes this request perform a grasp: a
    # clamped peg reads its own thickness and never reaches the command, so the step has to wait
    # on time rather than on the reading. Mirrors `TerminalTrialsRequest.graspSettleS`, which
    # passes its own value down so the two cannot drift.
    graspSettleS: float = 0.6
    # How far *below* the release height to close, when re-gripping in place. A peg let go of at
    # the height the fingers were holding it at can only go one way: down. Measured 2026-09-11 --
    # a trial that seated cleanly, released, and then re-gripped at the same z came up empty, and
    # the re-grips in the one table that did finish needed 1, 2 and 3 attempts at random, which
    # is what "sometimes it fell further than the fingers are deep" looks like. 0 keeps the old
    # behaviour. Clamped at `minZ` so this can never drive the fingers into the fixture.
    regripDropM: float = 0.0
    # What the fingers are told while the peg is carried and pressed. None keeps the measured
    # width, which is what this did first -- and a position-controlled gripper told to stay where
    # it is squeezes nothing. 09-28 trial 021: a landing on the face pressed to -18 N and the peg
    # slid ~20 mm up through the fingers in 0.2 s, the tool reached the seated depth, and the
    # verdict said "seated" (the operator: out). The next trial wedged, and the park -- still
    # re-commanding a measured 0.332 -- let the peg go. The scripted grasps close to 0.0.
    holdGripper: float | None = None
    # How hard, in N below the descent's first reading, the descent may press before its setpoint
    # is moved up to the tool and held there. See `descend_until_refused`. None never caps; a
    # stretch of all-zero wrench readings leaves it uncapped too, as before.
    pressCapN: float | None = 16.0
    gripperTolerance: float = 0.08
    controlPeriodS: float = 1.0 / 30.0
    requestId: str = ""

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def parse_terminal_servo_pose(text: str) -> tuple[float, float, float]:
    """Read `x,y,z` off the command line, in metres."""

    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise TerminalServoError(
            f"terminal servo pose must be three comma-separated metres, got {text!r}."
        )
    try:
        xyz = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise TerminalServoError(f"terminal servo pose {text!r} is not three numbers: {exc}") from exc
    for index, value in enumerate(xyz):
        if not math.isfinite(value):
            raise TerminalServoError(f"terminal servo pose component {index} must be finite.")
    return xyz  # type: ignore[return-value]


def terminal_servo_arming(
    armed: bool,
    *,
    commanded_gripper: float,
    observed_z: float,
    handoff_z: float,
    closed_below: float = 0.5,
    held: bool = True,
) -> tuple[bool, bool]:
    """Whether the handoff is armed after this observation, and whether it fires on it.

    A rollout crosses `handoff_z` downward twice: once reaching for the peg on the table at
    z = 0.046, and once carrying it to the fixture. Only the second is the descent E5 is about,
    and height alone cannot tell them apart, so the state machine is: holding the peg arms
    nothing by itself; being above the handoff height *while holding it* arms the trigger; and
    the next crossing downward fires it. Letting go disarms, so a rollout that drops the peg and
    re-grasps has to climb back over the height before it can hand over again.

    `commanded_gripper` is the command rather than the measured width, matching
    `RolloutGeometryTrace`: the observed width read 0 on 47% of frames in that dataset, so a
    test keyed on it alone fires on signal dropouts instead of on grasps. But a closed command
    is not a peg: fingers that closed on air hold the same command (D3), and the servo would
    then carry an empty gripper into the fixture. So `held` -- the caller's reading of the
    measured width over a short window, possible since the 09-07 dropout repair -- must agree,
    and an empty gripper disarms like an open one.
    """

    if commanded_gripper >= closed_below or not held:
        return False, False
    if observed_z > handoff_z:
        return True, False
    return armed, armed


def terminal_servo_waypoints(
    request: TerminalServoRequest,
    current_xyz: tuple[float, float, float],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Across at the height the policy left off, then straight down, then straight back up.

    The lateral move happens before the descent so the peg never travels sideways at fixture
    height, which is the one path that can shear a standing peg off the table. It keeps the
    handoff height rather than the current one so the same three segments are checked whether
    the policy handed over exactly at the threshold or a few millimetres below it.
    """

    x, y, z = request.xyz
    across = max(float(current_xyz[2]), request.handoffZ)
    return (
        ("align_above_target", (x, y, across)),
        ("descend_to_target", (x, y, z)),
        ("retreat_after_release", (x, y, z + request.retreatM)),
    )


def terminal_servo_search_offsets(
    request: TerminalServoRequest,
) -> tuple[tuple[float, float], ...]:
    """Where the search may put the peg down, as XY offsets from the nominal pose.

    The nominal pose comes first and always: the search is an addition to E5's descent, not a
    replacement for it, so a run in which the fixed pose would have worked spends no extra time
    and lands in exactly the place the control arm landed. A ring after it, rather than a
    spiral, because the thing being covered is a disc of possible peg-tip offsets and a ring of
    six plus the centre covers one to about 11 mm with no landing wasted on the middle of an
    already-covered patch. A spiral's extra landings buy resolution the capture radius does not
    need -- anything within 4.2 mm of a landing goes in on its own.
    """

    if request.searchRingM <= 0.0 or request.searchPoints <= 0:
        return ((0.0, 0.0),)
    step = 2.0 * math.pi / float(request.searchPoints)
    ring = tuple(
        (request.searchRingM * math.cos(index * step), request.searchRingM * math.sin(index * step))
        for index in range(int(request.searchPoints))
    )
    return ((0.0, 0.0),) + ring


def terminal_servo_search_path(
    request: TerminalServoRequest,
    current_xyz: tuple[float, float, float],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Every point the run may visit, for the QC to check before the arm moves.

    The search's own geometry is not known until it runs -- each landing is lifted from wherever
    the previous descent stopped -- so what is checked is the envelope that contains it: each
    landing at the target depth, and each landing at the highest the transfer can happen, which
    is a stop at the handoff height plus the lift. Both bounds are named points; the workspace
    test is a box, so nothing between two accepted corners can fall outside it, and the reach
    probe walks the same list.
    """

    points = list(terminal_servo_waypoints(request, current_xyz))
    offsets = terminal_servo_search_offsets(request)
    if len(offsets) == 1:
        return tuple(points)
    x, y, z = request.xyz
    transfer_z = request.handoffZ + request.searchLiftM
    for index, (dx, dy) in enumerate(offsets[1:], start=1):
        points.append((f"search_transfer[{index}]", (x + dx, y + dy, transfer_z)))
        points.append((f"search_descend[{index}]", (x + dx, y + dy, z)))
    return tuple(points)


def validate_terminal_servo_trajectory(
    request: TerminalServoRequest,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    current_xyz: tuple[float, float, float] | None = None,
    reach_probe: Any = None,
) -> dict[str, Any]:
    """Deterministic QC, before any waypoint is sent to the arm."""

    if request.timeoutS <= 0.0 or request.toleranceM <= 0.0 or request.controlPeriodS <= 0.0:
        raise TerminalServoError("timeoutS, toleranceM and controlPeriodS must be positive.")
    if request.stepToleranceM is not None and request.stepToleranceM <= 0.0:
        raise TerminalServoError("stepToleranceM must be positive when given.")
    if request.regripDropM < 0.0:
        raise TerminalServoError("regripDropM cannot be negative: a released peg falls, it does not rise.")
    if request.maxSpeedMs <= 0.0 or request.maxSpeedMs > TERMINAL_SERVO_MAX_SPEED_MS:
        raise TerminalServoError(
            f"maxSpeedMs must be in (0, {TERMINAL_SERVO_MAX_SPEED_MS}]: this motion drives a "
            "gripped peg onto the fixture and has no force control."
        )
    if request.contactLagM <= 0.0 or request.contactHoldS < 0.0:
        raise TerminalServoError("contactLagM must be positive and contactHoldS non-negative.")
    if request.settleS <= 0.0:
        raise TerminalServoError("settleS must be positive.")
    if request.contactWindowM <= 0.0 or request.contactWarmupM < 0.0:
        raise TerminalServoError("contactWindowM must be positive and contactWarmupM non-negative.")
    if request.contactStallM <= request.contactLagM:
        raise TerminalServoError(
            f"contactStallM {request.contactStallM:.4f} must be above contactLagM "
            f"{request.contactLagM:.4f}, or the backstop fires before the test it backs up."
        )
    if request.searchRingM < 0.0:
        raise TerminalServoError("searchRingM must be non-negative; zero disables the search.")
    if request.searchRingM > 0.0:
        if request.searchPoints < 3:
            raise TerminalServoError(
                f"searchPoints {request.searchPoints} is too few for a ring: under three the "
                "landings leave a gap wider than the ring itself in the middle of the disc."
            )
        if request.searchSeatedM <= 0.0 or request.searchSlipM <= 0.0:
            raise TerminalServoError("searchSeatedM and searchSlipM must be positive.")
        run_up_m = request.contactWarmupM + request.contactWindowM
        if request.searchLiftM < run_up_m:
            raise TerminalServoError(
                f"searchLiftM {request.searchLiftM:.4f} is below the {run_up_m:.4f} the growth "
                "test needs to arm, so every landing after the first would descend with no "
                "contact detection and stop by leaning on the peg until settleS."
            )
    if not 0.0 <= request.openGripper <= 1.0:
        raise TerminalServoError("openGripper must be normalized in [0, 1].")
    if request.retreatM <= 0.0:
        raise TerminalServoError("retreatM must be positive: the arm has to clear the peg it left.")
    for index, value in enumerate(request.xyz):
        if not math.isfinite(value):
            raise TerminalServoError(f"xyz[{index}] must be finite.")
    if request.xyz[2] < request.minZ:
        raise TerminalServoError(
            f"target z {request.xyz[2]:.4f} is below the {request.minZ:.3f} floor. The seated "
            "release is 0.052; a target under the floor is a typo or a stale calibration."
        )
    if not math.isfinite(request.handoffZ) or request.handoffZ <= request.xyz[2]:
        raise TerminalServoError(
            f"handoffZ {request.handoffZ:.4f} must be finite and above the target z "
            f"{request.xyz[2]:.4f}, or there is no descent for the servo to own."
        )

    low, high = _workspace_bounds(workspace_min, workspace_max)
    points: list[tuple[str, tuple[float, float, float]]] = list(
        terminal_servo_search_path(
            request, current_xyz or (request.xyz[0], request.xyz[1], request.handoffZ)
        )
    )
    if current_xyz is not None:
        points.insert(0, ("current", current_xyz))
    for name, xyz in points:
        _check_xyz_in_workspace(xyz, name, low, high)
    for (start_name, start), (end_name, end) in zip(points, points[1:], strict=False):
        for sample_index, sample in enumerate(_iter_segment_samples(start, end)):
            _check_xyz_in_workspace(sample, f"segment:{start_name}->{end_name}[{sample_index}]", low, high)
    reach_checked = 0
    if reach_probe is not None:
        reach_checked = _check_reach_along_path(points, reach_probe, request.toleranceM)
    return {"ok": True, "waypoints": len(points), "reachCheckedPoints": reach_checked}


def descend_until_refused(
    robot: Any,
    request: TerminalServoRequest,
    target_xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    gripper: float,
) -> dict[str, Any]:
    """Walk the setpoint down to `target_xyz` and report what stopped it.

    Three ways to stop, and all three are results rather than errors. `target` means the tool
    reached the seated depth with nothing in the way, which for a peg that is supposed to be in
    a hole means it went in -- or that it missed and the fixture is not where the target says.
    `contact` means the arm stopped following the setpoint down, and the height it stopped at is
    the reading. `timeout` means neither, and is reported so it is never silently read as a miss.

    What counts as contact is how much *further* behind its setpoint the tool falls across
    `contactWindowM` of commanded travel, not how far behind it is. The absolute lag is mostly
    the arm's answer delay: it scales with speed, it is there before the peg is anywhere near
    the fixture, and thresholding it stops the descent on the descent. The growth is zero while
    the tool is moving at all and one whole step per step once it is not. `lagMm` reports the
    delay itself so it stays visible in the log rather than being inferred from a stop.
    """

    commanded, current_rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    start_z = float(commanded[2])
    max_step_m = request.maxSpeedMs * request.controlPeriodS
    started = time.perf_counter()
    deadline = started + request.timeoutS + _distance(commanded, target_xyz) / request.maxSpeedMs
    held_up_since: float | None = None
    arrived_at: float | None = None
    arrived_z: float | None = None
    current_xyz = commanded
    held_up_m = 0.0
    growth_m = 0.0
    peak_growth_m = 0.0
    lags: list[float] = []
    # (commanded travel so far, the lag then), oldest first, trimmed to just span the window.
    trail: deque[tuple[float, float]] = deque()
    # Read-only, and nothing below decides on it. Recorded because the position verdict has
    # called a peg standing on the rim "seated" (09-11: four re-grips came up empty after one,
    # the peg having fallen), and the force is the other candidate for telling the two apart.
    # The tare is the first reading of the descent, taken while the peg is still in free air.
    trace = _ForceTrace(robot, request.requestId, "terminal_descent")
    fz_tare = read_fz(robot)
    fz_peak_drop = 0.0

    def stopped(reason: str) -> dict[str, Any]:
        ranked = sorted(lags)
        # How long the tool went on descending after the setpoint stopped, and how far. A free
        # arm closes its answer delay in a tenth of a second; anything longer is the tool being
        # let down by something rather than arriving. This is the only column that sees a peg
        # sliding in the jaws: while the setpoint is still moving, a sliding peg and a seated
        # one are the same kinematics, because it is the peg that gives way and not the arm.
        settle_s = 0.0 if arrived_at is None else time.perf_counter() - arrived_at
        settle_mm = 0.0 if arrived_z is None else 1000.0 * (arrived_z - float(current_xyz[2]))
        trace.flush(reason)
        fz_end = read_fz(robot)
        return {
            # Change from the tare, N; negative is the tool pushed on. None without a reading.
            "fzTareN": fz_tare,
            "dfzEndN": None if fz_tare is None or fz_end is None else fz_end - fz_tare,
            "dfzPeakN": None if fz_tare is None else fz_peak_drop,
            "pressCapped": unload_z is not None,
            "stoppedOn": reason,
            "stoppedAtXyz": list(current_xyz),
            # Measured, under whatever the press is doing to the wrist: what a peg that will not
            # go home is leaning at, as opposed to what it was told to lean at.
            "stoppedAtRotvec": [float(value) for value in current_rotvec],
            "heldUpMm": 1000.0 * held_up_m,
            "heldUpGrowthMm": 1000.0 * growth_m,
            "peakGrowthMm": 1000.0 * peak_growth_m,
            "lagMm": 1000.0 * (ranked[len(ranked) // 2] if ranked else 0.0),
            "descentMm": 1000.0 * (start_z - float(current_xyz[2])),
            "descentSeconds": time.perf_counter() - started,
            "settleSeconds": settle_s,
            "settleMm": settle_mm,
        }

    # Where the setpoint is held once the press passes `pressCapN`: at the tool, not below it.
    # 09-28 (fixed hole): the peg bottomed 1.8-2.7 mm above the target and the descent kept
    # commanding the target through its 0.5 s settle -- seats read -8 to -14 N, but trials 12-14
    # reached -19 to -25 N, the tool kicked 3 mm sideways, and trial 14 tripped cartesian_reflex.
    # A cap rather than always letting off at arrival, because the settle reads a peg sliding in
    # the jaws by pressing on it (09-10), and below the cap that reading is kept.
    unload_z: float | None = None

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        commanded = _step_toward(commanded, target_xyz, max_step_m)
        sent = commanded if unload_z is None else (commanded[0], commanded[1], max(commanded[2], unload_z))
        _send_absolute(robot, sent, rotvec, gripper)
        current_xyz, current_rotvec, _current_gripper = _observation_xyz_rotvec_gripper(robot)
        if not control_loop_alive(robot):
            trace.flush("control_loop_died")
            raise ControlLoopDiedError(
                f"terminal descent: the FR3 control loop has ended (a reflex); the arm stopped at "
                f"({current_xyz[0]:+.4f}, {current_xyz[1]:+.4f}, {current_xyz[2]:+.4f})."
            )
        trace.sample(float(sent[2]), current_xyz)
        if fz_tare is None:
            fz_tare = read_fz(robot)
        else:
            fz_now = read_fz(robot)
            if fz_now is not None:
                fz_peak_drop = min(fz_peak_drop, fz_now - fz_tare)
                if (
                    unload_z is None
                    and request.pressCapN is not None
                    and fz_now - fz_tare <= -request.pressCapN
                ):
                    unload_z = float(current_xyz[2])
                    print(
                        f"[INFO] terminal_servo_press_capped request_id={request.requestId} "
                        f"z={unload_z:.4f} dfz_n={fz_now - fz_tare:+.1f}",
                        flush=True,
                    )
        # Positive means the tool is sitting above where it was told to be. The sign matters: an
        # arm that overshoots downward is not in contact, and reading |error| here would stop the
        # descent on its own tracking.
        held_up_m = float(current_xyz[2] - commanded[2])
        travelled_m = start_z - float(commanded[2])
        lags.append(held_up_m)
        trail.append((travelled_m, held_up_m))
        while len(trail) > 1 and travelled_m - trail[1][0] >= request.contactWindowM:
            trail.popleft()
        growth_m = held_up_m - trail[0][1]
        peak_growth_m = max(peak_growth_m, growth_m)
        # No hold and no warm-up on the backstop: an arm this far behind is not late, and the
        # two cases it exists for both begin before the growth test can see them.
        if held_up_m >= request.contactStallM:
            return stopped("contact")
        # The whole window has to clear the warm-up, not just its near end: a window with one
        # foot in the ramp reads the ramp's climb as growth.
        refused = trail[0][0] >= request.contactWarmupM and growth_m >= request.contactLagM
        if not refused:
            held_up_since = None
        elif held_up_since is None:
            held_up_since = now
        elif now - held_up_since >= request.contactHoldS:
            return stopped("contact")
        # Depth, not distance. The lateral leg ran before the descent and converged; what is
        # left of it is a reading, reported as `lateralErrorMm`, not a reason to keep pressing.
        # A seated peg holds the tool laterally -- that is what being in a hole means -- so a
        # 3-D tolerance turns the successful insertions into timeouts. It did: on 2026-09-10
        # five descents reached the seated depth and four of them were still commanding it
        # twenty seconds later, two with the operator watching and writing down that the arm
        # would not let go.
        if commanded == target_xyz:
            if arrived_at is None:
                arrived_at = now
                arrived_z = float(current_xyz[2])
            if abs(float(current_xyz[2]) - target_xyz[2]) <= request.toleranceM:
                return stopped("target")
            if now - arrived_at >= request.settleS:
                return stopped("timeout")
        precise_sleep(request.controlPeriodS)
    return stopped("timeout")


def search_for_seat(
    robot: Any,
    request: TerminalServoRequest,
    rotvec: tuple[float, float, float],
    gripper: float,
) -> dict[str, Any]:
    """Descend at the nominal pose; if the peg did not go in, lift, step sideways, try again.

    This is E7 route C, and what makes it cheap rather than a research project is that both
    things it needs already exist and were measured on 2026-09-10. The verdict per landing is
    `above_target_mm`, whose two clusters -- 1.1-2.5 mm seated, 4.9-5.6 mm standing on the face
    -- do not overlap, so no force channel is wanted for the decision even though this rig has
    none. And the descent already knows how to be stopped by something rather than lean on it.

    Lift, then across, then down, rather than dragging the peg over the face: the two runs that
    crept for 8.1 s and 18.1 s showed the peg moving in the jaws under a held force, and a
    sliding search would apply that force sideways for the whole pattern. Each landing therefore
    starts from a fresh descent with the peg where the previous landing left it in the fingers.
    That is also why creep ends the search rather than counting as one more miss: after a slip
    the tool pose no longer says where the peg tip is, so every remaining landing is aimed with
    a ruler that changed length between the marks.

    The lift and the traverse run at the reset's speed, not this module's. The slow cap exists
    for the one leg that drives a gripped peg down at a fixture; these two climb and then cross
    a lifted plane, which is what `align_above_target` already does.
    """

    x, y, z = request.xyz
    offsets = terminal_servo_search_offsets(request)
    attempts: list[dict[str, Any]] = []
    descent: dict[str, Any] = {}
    landing = (x, y, z)
    previous = landing
    verdict = "exhausted"
    # Initialised for the same reason `descent` is: the loop below always runs at least once
    # (`terminal_servo_search_offsets` never returns empty), and a reader should not have to
    # prove that to know what the return statement refers to.
    reading = "standing"
    index = 0
    for index, (dx, dy) in enumerate(offsets):
        landing = (x + dx, y + dy, z)
        if index:
            lifted_z = float(descent["stoppedAtXyz"][2]) + request.searchLiftM
            _run_step(robot, request, f"search_lift[{index}]", (previous[0], previous[1], lifted_z),
                      rotvec, gripper, tolerance_m=request.stepToleranceM)
            _run_step(robot, request, f"search_transfer[{index}]", (landing[0], landing[1], lifted_z),
                      rotvec, gripper, tolerance_m=request.stepToleranceM)
        descent = descend_until_refused(robot, request, landing, rotvec, gripper)
        above_target_m = float(descent["stoppedAtXyz"][2]) - z
        reading = classify_terminal_servo_descent(
            above_target_mm=1000.0 * above_target_m,
            settle_mm=descent["settleMm"],
            seated_mm=1000.0 * request.searchSeatedM,
            slip_mm=1000.0 * request.searchSlipM,
        )
        attempts.append(
            {
                "index": index,
                "offsetMm": 1000.0 * math.hypot(dx, dy),
                "landingXyz": list(landing),
                "stoppedOn": descent["stoppedOn"],
                "aboveTargetMm": 1000.0 * above_target_m,
                "settleSeconds": descent["settleSeconds"],
                "settleMm": descent["settleMm"],
                "dfzPeakN": descent["dfzPeakN"],
                "dfzEndN": descent["dfzEndN"],
                "verdict": reading,
            }
        )
        if len(offsets) > 1:
            print(
                f"[INFO] terminal_servo_search=landing request_id={request.requestId} "
                f"index={index}/{len(offsets) - 1} offset_mm={1000.0 * math.hypot(dx, dy):.1f} "
                f"stopped_on={descent['stoppedOn']} above_target_mm={1000.0 * above_target_m:+.1f} "
                f"settle_s={descent['settleSeconds']:.2f} settle_mm={descent['settleMm']:+.1f}",
                flush=True,
            )
        # Creep is read before depth, and the order is the whole lesson of 2026-09-10. Two runs
        # reached the seated depth and were graded misses: the peg had slid up between the
        # fingers while the arm leaned, so the tool arrived where the hole is and the peg did
        # not. Depth alone cannot tell those from an insertion -- 0.7 and 1.4 mm above target,
        # squarely in the seated cluster -- and the only column that can is how long the last
        # millimetres took, 8.1 s and 18.1 s against 0.1 s for a real one. Both tests live in
        # `classify_terminal_servo_descent`, which keeps that order.
        if reading in ("slip", "seated"):
            verdict = reading
            break
        previous = landing
    return {
        **descent,
        # What the last descent says happened to the peg, as opposed to why the *search* stopped:
        # with the ring off (E5) the loop runs once and reports `exhausted`, which is a true
        # statement about the search and says nothing about whether the peg went in. A reader
        # asking "did it seat" should not have to know that `exhausted` means `standing` here
        # and "ran out of landings" there.
        "verdict": reading,
        "searchStoppedOn": verdict,
        "searchIndex": index,
        "searchLandings": len(offsets),
        "searchOffsetMm": 1000.0 * math.hypot(landing[0] - x, landing[1] - y),
        "searchLandingXyz": list(landing),
        "searchAttempts": attempts,
    }


def execute_terminal_servo(
    robot: Any,
    request: TerminalServoRequest,
    *,
    release_gate: Callable[[dict[str, Any]], bool] | None = None,
    commanded_rotvec: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Take the arm off the policy and drive the last centimetres to the fixed pose.

    `release_gate`, when given, decides whether the fingers open, in place of
    `releaseOnlyWhenSeated`: it is handed the descent (with `seatedDepthErrorMm`) while the peg
    is still held, so a person can look before anything is let go of. True releases (and
    re-grips in place if asked to), False keeps hold, and "open" releases without re-gripping.

    Only the descent is slowed to this module's speed. The lateral leg runs at the reset's
    0.15 m/s because it happens at the handoff height with nothing under it, and so does the
    retreat because it climbs. The slow limit exists for the one leg that drives a gripped peg
    at a fixture, and applying it to the other two would only make each run a few seconds longer.

    The wrist orientation is whatever the policy was holding when it handed over, and is
    reported rather than commanded. E5 asks whether tool *position* determines the peg tip;
    rotating the wrist to the demonstrations' median on the way down would change the peg's
    offset at the same time as the position and make the answer unreadable. The angle to the
    demonstrations is logged so a run can be checked afterwards for having handed over at an
    orientation the demonstrations never used.

    `commanded_rotvec` replaces that reading as the orientation sent, for a caller that holds
    one fixed across many calls. The reading is the *measured* pose, and the joint PD sags
    under the wrist's unmodelled load, so a loop that hands each call back the last one's
    reading re-commands the sag and adds another on top (09-28: four runs whose landing side
    force grew every trial until trial 11-13 jammed). `handoffRotvec` stays the reading.
    """

    current_xyz, measured_rotvec, gripper = _observation_xyz_rotvec_gripper(robot)
    rotvec = measured_rotvec if commanded_rotvec is None else tuple(float(v) for v in commanded_rotvec)
    if request.holdGripper is not None:
        gripper = float(request.holdGripper)
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    try:
        qc = validate_terminal_servo_trajectory(
            request,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            current_xyz=current_xyz,
            reach_probe=_reach_probe(robot, rotvec),
        )
    except SceneResetError as exc:
        print(
            f"[WARN] terminal_servo=failed request_id={request.requestId} "
            f"details=trajectory_qc_failed: {exc}",
            flush=True,
        )
        return {"ok": False, "error": f"trajectory_qc_failed: {exc}", "request": request.payload()}

    print(
        f"[INFO] terminal_servo=start request_id={request.requestId} "
        f"handoff_xyz={current_xyz[0]:+.4f},{current_xyz[1]:+.4f},{current_xyz[2]:+.4f} "
        f"target_xyz={request.xyz[0]:+.4f},{request.xyz[1]:+.4f},{request.xyz[2]:+.4f} "
        f"trajectory_qc=passed waypoints={qc['waypoints']} reach_checked={qc['reachCheckedPoints']}",
        flush=True,
    )
    waypoints = dict(terminal_servo_waypoints(request, current_xyz))
    try:
        _run_step(robot, request, "align_above_target", waypoints["align_above_target"], rotvec, gripper,
                  tolerance_m=request.stepToleranceM)
        descent = search_for_seat(robot, request, rotvec, gripper)
        stopped_at = tuple(float(value) for value in descent["stoppedAtXyz"])
        landing = tuple(float(value) for value in descent["searchLandingXyz"])
        # Let go where the descent stopped, not at the target: on a contact stop the target is a
        # height the arm could not reach, and re-commanding it while opening the fingers would
        # lean on the fixture through the one moment the peg is no longer clamped. The XY is the
        # tool's too, not the landing's. A seated peg holds the tool ~1.2 mm off the landing, so
        # commanding the landing jumped the open fingers 1.2 mm sideways and the re-grip closed
        # off-centre on a peg whose foot the hole holds -- a little crooked each time. 09-28,
        # three runs: the seated press grew -8 to -15 N over eleven re-grips and the twelfth could
        # not be pushed home (the operator saw the peg crooked in the fingers). This alone did not
        # cure that: the 17:17 run released here and jammed at trial 13 all the same, its side
        # force climbing from the first trial -- see `anchor_rotvec` in terminal_trials.
        release_xyz = (stopped_at[0], stopped_at[1], stopped_at[2])
        seated = descent["searchStoppedOn"] == "seated"
        if release_gate is not None:
            decision = release_gate(
                {**descent, "seatedDepthErrorMm": 1000.0 * (stopped_at[2] - request.xyz[2])}
            )
            # "open": let go and leave the peg for a person to set right -- no re-grip in place.
            released = decision is True or decision == "open"
            regrip = decision is True
        else:
            released = seated or not request.releaseOnlyWhenSeated
            regrip = released
        retreat_gripper = request.openGripper
        if released:
            _send_absolute(robot, release_xyz, rotvec, request.openGripper)
            precise_sleep(request.openSettleS)
        else:
            # Still holding it. The trial is over either way -- what the peg does next is not a
            # measurement, it is the next trial's starting condition.
            retreat_gripper = gripper
        if regrip and request.regripGripper is not None:
            regrip_xyz = (
                release_xyz[0],
                release_xyz[1],
                max(release_xyz[2] - request.regripDropM, request.minZ),
            )
            _run_step(robot, request, "regrip_after_release", regrip_xyz, rotvec,
                      float(request.regripGripper), tolerance_m=request.stepToleranceM)
            retreat_gripper = float(request.regripGripper)
        _run_step(
            robot,
            request,
            "retreat_after_release",
            # Straight up from where it let go: the peg may still be in the hole.
            (release_xyz[0], release_xyz[1], waypoints["retreat_after_release"][2]),
            rotvec,
            retreat_gripper,
            tolerance_m=request.stepToleranceM,
        )
        result = {
            "ok": True,
            "request": request.payload(),
            "trajectoryQc": qc,
            "handoffXyz": list(current_xyz),
            "handoffRotvec": list(measured_rotvec),
            "commandedRotvec": list(rotvec),
            "lateralErrorMm": 1000.0
            * math.hypot(stopped_at[0] - request.xyz[0], stopped_at[1] - request.xyz[1]),
            "seatedDepthErrorMm": 1000.0 * (stopped_at[2] - request.xyz[2]),
            "regripped": regrip and request.regripGripper is not None,
            "released": released,
            **descent,
        }
        print(
            f"[INFO] terminal_servo=done request_id={request.requestId} "
            # The conclusion first, because it is what a reader wants and what the page records;
            # the columns after it are how it was reached.
            f"verdict={result['verdict']} "
            f"stopped_on={result['stoppedOn']} "
            f"stopped_z={stopped_at[2]:.4f} above_target_mm={result['seatedDepthErrorMm']:+.1f} "
            f"lateral_mm={result['lateralErrorMm']:.1f} held_up_mm={result['heldUpMm']:+.1f} "
            f"growth_mm={result['heldUpGrowthMm']:+.1f} peak_growth_mm={result['peakGrowthMm']:+.1f} "
            f"lag_mm={result['lagMm']:+.1f} descent_mm={result['descentMm']:.1f} "
            f"descent_s={result['descentSeconds']:.1f} settle_s={result['settleSeconds']:.2f} "
            f"settle_mm={result['settleMm']:+.1f} search={result['searchStoppedOn']} "
            f"search_index={result['searchIndex']}/{result['searchLandings'] - 1} "
            f"search_offset_mm={result['searchOffsetMm']:.1f}",
            flush=True,
        )
        return result
    except ControlLoopDiedError as exc:
        # Nothing more can be sent, and the caller must not try: no grade, no release, no park.
        print(f"[WARN] terminal_servo=control_loop_died request_id={request.requestId} details={exc}", flush=True)
        return {"ok": False, "error": str(exc), "controlLoopDied": True, "request": request.payload()}
    except Exception as exc:  # noqa: BLE001 - the caller reports this without killing the session
        print(f"[WARN] terminal_servo=failed request_id={request.requestId} details={exc}", flush=True)
        return {"ok": False, "error": str(exc), "request": request.payload()}
