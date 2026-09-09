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

Everything else -- the speed clamp, the workspace checks, the step loop -- is imported from
`scene_reset` rather than copied. A second way of walking the arm to a typed-in coordinate is a
second set of safety checks to keep in step, and the one that gets skipped is always the one
nobody thought of as a real motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable

from tools.fr3.scene_reset import (
    SceneResetError,
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
    precise_sleep,
)

# How fast the commanded tool point may travel, in m/s. A third of the reset's, because the
# reset's rate limit exists to keep a transfer from crossing the table at a metre a second and
# this one exists to keep a 15 mm peg from being driven into a fixture face.
TERMINAL_SERVO_MAX_SPEED_MS = 0.05
# How far the tool may sit above its own setpoint before the descent calls it contact, in metres.
# At the speed above and a 30 Hz control period the setpoint moves 1.7 mm a step, so a servo one
# full step behind reads as 1.7 mm of lag with nothing touching anything.
TERMINAL_SERVO_CONTACT_LAG_M = 0.004
# For how long, in seconds. Acceleration at the top of the descent produces the same reading for
# a few steps.
TERMINAL_SERVO_CONTACT_HOLD_S = 0.3
# How far above the target the arm lifts once it has let go, in metres. Enough to clear a
# standing peg before the waiting loop homes the arm across the table.
TERMINAL_SERVO_RETREAT_M = 0.08
# The lowest z a target may name. The seated release is 0.052; anything materially below that is
# a typo or a stale calibration, and this motion has no force control to survive one.
TERMINAL_SERVO_MIN_Z_M = 0.030


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
    minZ: float = TERMINAL_SERVO_MIN_Z_M
    retreatM: float = TERMINAL_SERVO_RETREAT_M
    timeoutS: float = 20.0
    toleranceM: float = 0.002
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
) -> tuple[bool, bool]:
    """Whether the handoff is armed after this observation, and whether it fires on it.

    A rollout crosses `handoff_z` downward twice: once reaching for the peg on the table at
    z = 0.046, and once carrying it to the fixture. Only the second is the descent E5 is about,
    and height alone cannot tell them apart, so the state machine is: holding the peg arms
    nothing by itself; being above the handoff height *while holding it* arms the trigger; and
    the next crossing downward fires it. Letting go disarms, so a rollout that drops the peg and
    re-grasps has to climb back over the height before it can hand over again.

    `commanded_gripper` is the command rather than the measured width, matching
    `RolloutGeometryTrace`: the observed width reads 0 on 47% of frames in this dataset, so a
    test keyed on it fires on signal dropouts instead of on grasps.
    """

    if commanded_gripper >= closed_below:
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
    if request.maxSpeedMs <= 0.0 or request.maxSpeedMs > TERMINAL_SERVO_MAX_SPEED_MS:
        raise TerminalServoError(
            f"maxSpeedMs must be in (0, {TERMINAL_SERVO_MAX_SPEED_MS}]: this motion drives a "
            "gripped peg onto the fixture and has no force control."
        )
    if request.contactLagM <= 0.0 or request.contactHoldS < 0.0:
        raise TerminalServoError("contactLagM must be positive and contactHoldS non-negative.")
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
        terminal_servo_waypoints(request, current_xyz or (request.xyz[0], request.xyz[1], request.handoffZ))
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
    `contact` means the arm stayed above its own setpoint, and the height it stayed at is the
    reading. `timeout` means neither, and is reported so it is never silently read as a miss.
    """

    commanded, _rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    max_step_m = request.maxSpeedMs * request.controlPeriodS
    deadline = (
        time.perf_counter()
        + request.timeoutS
        + _distance(commanded, target_xyz) / request.maxSpeedMs
    )
    held_up_since: float | None = None
    current_xyz = commanded
    held_up_m = 0.0
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        commanded = _step_toward(commanded, target_xyz, max_step_m)
        _send_absolute(robot, commanded, rotvec, gripper)
        current_xyz, _current_rotvec, _current_gripper = _observation_xyz_rotvec_gripper(robot)
        # Positive means the tool is sitting above where it was told to be, which on a descent
        # is something holding it up. The sign matters: an arm that overshoots downward is not
        # in contact, and reading |error| here would stop the descent on its own tracking.
        held_up_m = float(current_xyz[2] - commanded[2])
        if held_up_m < request.contactLagM:
            held_up_since = None
        elif held_up_since is None:
            held_up_since = now
        elif now - held_up_since >= request.contactHoldS:
            return {
                "stoppedOn": "contact",
                "stoppedAtXyz": list(current_xyz),
                "heldUpMm": 1000.0 * held_up_m,
            }
        if commanded == target_xyz and _distance(current_xyz, target_xyz) <= request.toleranceM:
            return {
                "stoppedOn": "target",
                "stoppedAtXyz": list(current_xyz),
                "heldUpMm": 1000.0 * held_up_m,
            }
        precise_sleep(request.controlPeriodS)
    return {
        "stoppedOn": "timeout",
        "stoppedAtXyz": list(current_xyz),
        "heldUpMm": 1000.0 * held_up_m,
    }


def execute_terminal_servo(robot: Any, request: TerminalServoRequest) -> dict[str, Any]:
    """Take the arm off the policy and drive the last centimetres to the fixed pose.

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
    """

    current_xyz, rotvec, gripper = _observation_xyz_rotvec_gripper(robot)
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
        _run_step(robot, request, "align_above_target", waypoints["align_above_target"], rotvec, gripper)
        descent = descend_until_refused(robot, request, waypoints["descend_to_target"], rotvec, gripper)
        stopped_at = tuple(float(value) for value in descent["stoppedAtXyz"])
        # Let go where the descent stopped, not at the target: on a contact stop the target is a
        # height the arm could not reach, and re-commanding it while opening the fingers would
        # lean on the fixture through the one moment the peg is no longer clamped.
        release_xyz = (waypoints["descend_to_target"][0], waypoints["descend_to_target"][1], stopped_at[2])
        _send_absolute(robot, release_xyz, rotvec, request.openGripper)
        precise_sleep(request.openSettleS)
        _run_step(
            robot,
            request,
            "retreat_after_release",
            waypoints["retreat_after_release"],
            rotvec,
            request.openGripper,
        )
        result = {
            "ok": True,
            "request": request.payload(),
            "trajectoryQc": qc,
            "handoffXyz": list(current_xyz),
            "handoffRotvec": list(rotvec),
            "lateralErrorMm": 1000.0
            * math.hypot(stopped_at[0] - request.xyz[0], stopped_at[1] - request.xyz[1]),
            "seatedDepthErrorMm": 1000.0 * (stopped_at[2] - request.xyz[2]),
            **descent,
        }
        print(
            f"[INFO] terminal_servo=done request_id={request.requestId} "
            f"stopped_on={result['stoppedOn']} "
            f"stopped_z={stopped_at[2]:.4f} above_target_mm={result['seatedDepthErrorMm']:+.1f} "
            f"lateral_mm={result['lateralErrorMm']:.1f} held_up_mm={result['heldUpMm']:+.1f}",
            flush=True,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - the caller reports this without killing the session
        print(f"[WARN] terminal_servo=failed request_id={request.requestId} details={exc}", flush=True)
        return {"ok": False, "error": str(exc), "request": request.payload()}
