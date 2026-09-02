"""Scene reset plan and executor for the FR3 peg task.

A reset is an environment operation, not a policy rollout: pick the peg from a fixed staging
pose, lift it clear by the fixed reset clearance, place it at a sampled task target, and
optionally return the arm to its start pose. The web UI and gateway decide which target is
sampled; this module owns the robot-side invariants and refuses to execute a trajectory that
fails the reset QC checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import json
import math
import random
import time
from typing import Any, Iterable

try:  # pragma: no cover - tests run without the full lerobot import tree in some environments
    from lerobot.utils.robot_utils import precise_sleep
except Exception:  # noqa: BLE001
    def precise_sleep(seconds: float) -> None:
        time.sleep(seconds)


SCENE_RESET_LIFT_M = 0.08
# How fast the commanded tool point may travel between waypoints, in m/s.
#
# A step used to send its waypoint as one absolute pose and leave the arm to find its own way
# there. Between the post-grasp lift and the place approach that is a single 514 mm command, and
# the OTG underneath is configured at the FR3's rated joint maxima, so the tool crossed it at
# roughly a metre a second with the peg in the gripper. The teleop path never does this: the
# driver clamps a relative command to `max_target_delta_pos`, 1 mm a step on this rig. But the
# absolute-pose branch of `send_action` has no equivalent clamp, and a reset only ever uses that
# branch, so the limit has to live here.
#
# Walking the setpoint also makes the straight line the QC samples the line the arm travels.
# Sampling a segment the executor never commanded was checking a path nothing followed.
SCENE_RESET_MAX_SPEED_MS = 0.15
# How long the arm may be behind its own command before a step gives up, in seconds.
#
# `FrankaResearch3.reach_stall_error_m` is non-zero only when IK could not realise the pose it
# was handed -- it compares two poses from the kinematics model, so servo lag and a slow gripper
# do not show up in it. Under a rate-limited setpoint the command is never more than a few mm
# ahead of the arm, so a reading that stays non-zero means the waypoint itself is out of reach
# and waiting cannot help. Without this, a reset spent its whole 20 s timeout leaning on a joint
# limit with the peg gripped.
SCENE_RESET_REACH_STALL_S = 0.5
# What fraction of the arm's rated joint speed a reset homes at.
#
# `move_to_start` is a joint-space move at the OTG's configured ceilings, which are the FR3's
# rated maxima -- the same ceilings that turned a single large absolute command into a metre a
# second across the table. Rate-limiting the reset's Cartesian steps and then homing flat out
# would leave the fastest motion of a reset in the one step nobody was watching. A failed reset
# reaches homing from states the success path never produces: mid-transfer, sometimes at the edge
# of the arm's reach, usually still holding the peg.
SCENE_RESET_HOMING_SPEED_SCALE = 0.25
_SCENE_RESET_EPS = 1e-6
_TRAJECTORY_QC_SAMPLE_STEP_M = 0.01


class SceneResetError(ValueError):
    """A reset request that is invalid before a robot should move."""


class SceneResetUnreachableError(RuntimeError):
    """A waypoint IK cannot realise, found with the arm already part-way through the reset.

    Deliberately not a `SceneResetError`: that one means the request was wrong before anything
    moved and nothing needs recovering. This one is only knowable from the arm standing at the
    edge of its own reach, so it leaves a trajectory half-executed and something holding the
    peg. The caller's recovery differs accordingly -- see `_abort_scene_reset`.
    """


@dataclass(frozen=True)
class SceneResetStroke:
    x: float
    y: float
    radiusM: float


@dataclass(frozen=True)
class SceneResetWaypoint:
    name: str
    xyz: tuple[float, float, float]
    gripper: float


@dataclass(frozen=True)
class SceneResetRequest:
    pickXyz: tuple[float, float, float]
    targetXyz: tuple[float, float, float]
    liftM: float = SCENE_RESET_LIFT_M
    approachClearanceM: float = SCENE_RESET_LIFT_M
    openGripper: float = 1.0
    closedGripper: float = 0.0
    returnToStart: bool = True
    timeoutS: float = 20.0
    toleranceM: float = 0.006
    gripperTolerance: float = 0.08
    graspSettleS: float = 0.6
    controlPeriodS: float = 1.0 / 30.0
    maskStrokes: tuple[SceneResetStroke, ...] = field(default_factory=tuple)
    requestId: str = ""

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["maskStrokes"] = [asdict(stroke) for stroke in self.maskStrokes]
        return data


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SceneResetError(f"{field_name} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneResetError(f"{field_name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise SceneResetError(f"{field_name} must be finite.")
    return parsed


def _xyz(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SceneResetError(f"{field_name} must be [x, y, z].")
    return tuple(_finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _workspace_bounds(
    workspace_min: Iterable[float] | None,
    workspace_max: Iterable[float] | None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    if workspace_min is None or workspace_max is None:
        return None, None
    low = tuple(float(value) for value in workspace_min)
    high = tuple(float(value) for value in workspace_max)
    if len(low) != 3 or len(high) != 3:
        raise SceneResetError("workspace bounds must be 3D.")
    return low, high


def _check_xyz_in_workspace(
    xyz: tuple[float, float, float],
    field_name: str,
    workspace_min: tuple[float, float, float] | None,
    workspace_max: tuple[float, float, float] | None,
) -> None:
    if workspace_min is None or workspace_max is None:
        return
    for axis, value, low, high in zip("xyz", xyz, workspace_min, workspace_max, strict=True):
        if value < low or value > high:
            raise SceneResetError(
                f"{field_name}.{axis}={value:g} is outside the robot workspace [{low:g}, {high:g}]."
            )


def _parse_strokes(raw: Any) -> tuple[SceneResetStroke, ...]:
    raw_strokes = raw
    if isinstance(raw, dict):
        raw_strokes = raw.get("strokes")
    if not isinstance(raw_strokes, list) or not raw_strokes:
        raise SceneResetError("Draw at least one target-region brush stroke before reset.")
    strokes: list[SceneResetStroke] = []
    for index, item in enumerate(raw_strokes):
        if not isinstance(item, dict):
            raise SceneResetError(f"mask.strokes[{index}] must be an object.")
        x = _finite_float(item.get("x"), f"mask.strokes[{index}].x")
        y = _finite_float(item.get("y"), f"mask.strokes[{index}].y")
        radius = _finite_float(
            item.get("radiusM", item.get("radius", item.get("r"))),
            f"mask.strokes[{index}].radiusM",
        )
        if radius <= 0.0 or radius > 0.25:
            raise SceneResetError(f"mask.strokes[{index}].radiusM must be in (0, 0.25].")
        strokes.append(SceneResetStroke(x=x, y=y, radiusM=radius))
    if len(strokes) > 512:
        raise SceneResetError("mask carries too many brush strokes; clear it and draw a smaller region.")
    return tuple(strokes)


def parse_mask_strokes(raw: Any) -> tuple[SceneResetStroke, ...]:
    """The same mask validation a reset request gets, but with "no region" allowed.

    A reset with no target region has nowhere to put the peg, so the request parser refuses an
    empty one. A *stored* mask is a different question: an operator who cleared the region has
    said something, and the next page load has to hear "no region" rather than be handed back
    the region they just deleted.
    """

    raw_strokes = raw.get("strokes") if isinstance(raw, dict) else raw
    if isinstance(raw_strokes, list) and not raw_strokes:
        return ()
    return _parse_strokes(raw)


def _sample_xy_from_strokes(strokes: tuple[SceneResetStroke, ...], rng: random.Random) -> tuple[float, float]:
    weights = [stroke.radiusM * stroke.radiusM for stroke in strokes]
    stroke = rng.choices(strokes, weights=weights, k=1)[0]
    radius = stroke.radiusM * math.sqrt(rng.random())
    theta = rng.random() * math.tau
    return stroke.x + radius * math.cos(theta), stroke.y + radius * math.sin(theta)


def _require_fixed_lift(lift_m: float, field_name: str) -> None:
    if abs(lift_m - SCENE_RESET_LIFT_M) > _SCENE_RESET_EPS:
        raise SceneResetError(f"{field_name} must be exactly 0.08 m; scene reset lifts and descends 8 cm.")


def _validate_common_request_fields(request: SceneResetRequest) -> None:
    _require_fixed_lift(request.liftM, "liftM")
    _require_fixed_lift(request.approachClearanceM, "approachClearanceM")
    if request.timeoutS <= 0.0 or request.toleranceM <= 0.0 or request.controlPeriodS <= 0.0:
        raise SceneResetError("timeoutS, toleranceM and controlPeriodS must be positive.")
    if request.gripperTolerance < 0.0 or request.graspSettleS < 0.0:
        raise SceneResetError("gripperTolerance and graspSettleS must be non-negative.")
    if request.openGripper < 0.0 or request.openGripper > 1.0 or request.closedGripper < 0.0 or request.closedGripper > 1.0:
        raise SceneResetError("openGripper and closedGripper must be normalized in [0, 1].")
    for name, xyz in (("pickXyz", request.pickXyz), ("targetXyz", request.targetXyz)):
        for index, value in enumerate(xyz):
            if not math.isfinite(value):
                raise SceneResetError(f"{name}[{index}] must be finite.")


def build_scene_reset_waypoints(request: SceneResetRequest) -> tuple[SceneResetWaypoint, ...]:
    pick = request.pickXyz
    target = request.targetXyz
    pick_above = (pick[0], pick[1], pick[2] + SCENE_RESET_LIFT_M)
    target_above = (target[0], target[1], target[2] + SCENE_RESET_LIFT_M)
    return (
        SceneResetWaypoint("go_to_pick_above", pick_above, request.openGripper),
        SceneResetWaypoint("descend_8cm_to_pick", pick, request.openGripper),
        SceneResetWaypoint("close_gripper", pick, request.closedGripper),
        SceneResetWaypoint("lift_8cm_after_grasp", pick_above, request.closedGripper),
        SceneResetWaypoint("move_to_place_above", target_above, request.closedGripper),
        SceneResetWaypoint("descend_8cm_to_place", target, request.closedGripper),
        SceneResetWaypoint("open_gripper", target, request.openGripper),
        SceneResetWaypoint("retreat_8cm", target_above, request.openGripper),
    )


def _iter_segment_samples(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    step_m: float = _TRAJECTORY_QC_SAMPLE_STEP_M,
):
    distance = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, end, strict=True)))
    steps = max(1, int(math.ceil(distance / step_m)))
    for index in range(steps + 1):
        alpha = index / steps
        yield tuple(a + (b - a) * alpha for a, b in zip(start, end, strict=True))  # type: ignore[misc]


def _iter_reach_path(
    points: list[tuple[str, tuple[float, float, float]]],
) -> Iterable[tuple[str, tuple[float, float, float]]]:
    """The QC's sampled path as one ordered walk, without the seam between segments repeated.

    The workspace check can afford to sample each segment independently -- a box test has no
    memory. The reach check cannot: it feeds each solution back in as the next seed, so a point
    solved twice would quietly advance the walk by nothing while costing a solve.
    """

    for index, ((start_name, start), (end_name, end)) in enumerate(zip(points, points[1:], strict=False)):
        for sample_index, sample in enumerate(_iter_segment_samples(start, end)):
            if sample_index == 0 and index > 0:
                continue
            yield f"{start_name}->{end_name}[{sample_index}]", sample


def _check_reach_along_path(
    points: list[tuple[str, tuple[float, float, float]]],
    reach_probe: Any,
    tolerance_m: float,
) -> int:
    """Refuse a path the arm's own IK cannot follow, and say where it gives out.

    The workspace fence is an axis-aligned box; the set of tool points reachable at a fixed tool
    orientation is not, and the box is the larger of the two near the top. A place height that
    passes the fence by 7 cm can still be 6 mm past the last configuration IK can find, and the
    solver reports that by returning the joints it already had -- no exception, no flag. Asking it
    here, on the model, is the difference between refusing a request and discovering it with the
    peg 60 cm up.
    """

    labelled = list(_iter_reach_path(points))
    checked = 0
    # zip is what makes this lazy: a probe that yields per point stops being pulled the moment
    # this raises, so an unreachable target does not pay for the rest of the trajectory.
    for (label, xyz), error_m in zip(labelled, reach_probe([xyz for _label, xyz in labelled]), strict=False):
        checked += 1
        if error_m > tolerance_m:
            raise SceneResetError(
                f"trajectory QC failed: {label} at "
                f"({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) is {error_m * 1000.0:.1f} mm past what "
                f"the arm can reach at this tool orientation, against a tolerance of "
                f"{tolerance_m * 1000.0:.1f} mm. This is the arm's own reach, not the workspace "
                f"fence -- lower the place height, or move the target region closer in."
            )
    return checked


def _reach_probe(robot: Any, rotvec: tuple[float, float, float]) -> Any:
    """The robot's own IK as a QC callable, or None for a robot that cannot answer.

    Read through `getattr` for the reason `_reach_stall_error_m` is: the reset also runs against
    the MuJoCo twin and the test fakes, and an arm that cannot check its own reach must not be
    treated as an arm whose reach is exhausted.
    """

    iter_reach_errors_m = getattr(robot, "iter_reach_errors_m", None)
    if not callable(iter_reach_errors_m):
        return None
    return lambda tool_points: iter_reach_errors_m(tool_points, rotvec)


def validate_scene_reset_trajectory(
    request: SceneResetRequest,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    current_xyz: tuple[float, float, float] | None = None,
    reach_probe: Any = None,
) -> dict[str, Any]:
    """Run deterministic reset QC before any waypoint is sent to the robot.

    `reach_probe` is optional because this also runs gateway-side, where the request is checked
    before any robot exists to ask. Where it is supplied, it is the last check and the slowest:
    everything geometry can reject on its own is already gone by then.
    """

    _validate_common_request_fields(request)
    low, high = _workspace_bounds(workspace_min, workspace_max)
    waypoints = build_scene_reset_waypoints(request)
    by_name = {waypoint.name: waypoint for waypoint in waypoints}

    if abs(by_name["lift_8cm_after_grasp"].xyz[2] - request.pickXyz[2] - SCENE_RESET_LIFT_M) > _SCENE_RESET_EPS:
        raise SceneResetError("trajectory QC failed: post-grasp lift is not 8 cm.")
    if abs(by_name["move_to_place_above"].xyz[2] - request.targetXyz[2] - SCENE_RESET_LIFT_M) > _SCENE_RESET_EPS:
        raise SceneResetError("trajectory QC failed: place approach is not 8 cm above target.")
    if by_name["lift_8cm_after_grasp"].xyz[:2] != request.pickXyz[:2]:
        raise SceneResetError("trajectory QC failed: lift after grasp must be vertical over pick.")
    if by_name["descend_8cm_to_place"].xyz[:2] != request.targetXyz[:2]:
        raise SceneResetError("trajectory QC failed: descend to place must be vertical over target.")

    points: list[tuple[str, tuple[float, float, float]]] = [(waypoint.name, waypoint.xyz) for waypoint in waypoints]
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

    return {
        "ok": True,
        "waypoints": len(waypoints),
        "liftM": SCENE_RESET_LIFT_M,
        "sampleStepM": _TRAJECTORY_QC_SAMPLE_STEP_M,
        "reachCheckedPoints": reach_checked,
    }


def sanitize_scene_reset_request(
    raw: Any,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    rng: random.Random | None = None,
) -> SceneResetRequest:
    """Validate a UI reset request and sample the concrete place target."""

    if not isinstance(raw, dict):
        raise SceneResetError("scene reset payload must be a JSON object.")
    low, high = _workspace_bounds(workspace_min, workspace_max)
    pick_xyz = _xyz(raw.get("pickXyz"), "pickXyz")
    lift_m = _finite_float(raw.get("liftM", SCENE_RESET_LIFT_M), "liftM")
    _require_fixed_lift(lift_m, "liftM")
    approach = _finite_float(raw.get("approachClearanceM", lift_m), "approachClearanceM")
    _require_fixed_lift(approach, "approachClearanceM")
    target_z = _finite_float(raw.get("targetZ", raw.get("placeZ")), "targetZ")
    strokes = _parse_strokes(raw.get("mask", raw.get("maskStrokes")))

    sampler = rng or random.SystemRandom()
    target_xy: tuple[float, float] | None = None
    last_error = ""
    for _attempt in range(1000):
        x, y = _sample_xy_from_strokes(strokes, sampler)
        candidate = (x, y, target_z)
        try:
            _check_xyz_in_workspace(candidate, "targetXyz", low, high)
            _check_xyz_in_workspace((x, y, target_z + SCENE_RESET_LIFT_M), "targetAboveXyz", low, high)
        except SceneResetError as exc:
            last_error = str(exc)
            continue
        target_xy = (x, y)
        break
    if target_xy is None:
        detail = f" Last rejected sample: {last_error}" if last_error else ""
        raise SceneResetError("The painted target region does not overlap the robot workspace." + detail)

    request = SceneResetRequest(
        pickXyz=pick_xyz,
        targetXyz=(target_xy[0], target_xy[1], target_z),
        liftM=SCENE_RESET_LIFT_M,
        approachClearanceM=SCENE_RESET_LIFT_M,
        openGripper=_finite_float(raw.get("openGripper", 1.0), "openGripper"),
        closedGripper=_finite_float(raw.get("closedGripper", 0.0), "closedGripper"),
        returnToStart=bool(raw.get("returnToStart", True)),
        timeoutS=_finite_float(raw.get("timeoutS", 20.0), "timeoutS"),
        toleranceM=_finite_float(raw.get("toleranceM", 0.006), "toleranceM"),
        gripperTolerance=_finite_float(raw.get("gripperTolerance", 0.08), "gripperTolerance"),
        graspSettleS=_finite_float(raw.get("graspSettleS", 0.6), "graspSettleS"),
        controlPeriodS=_finite_float(raw.get("controlPeriodS", 1.0 / 30.0), "controlPeriodS"),
        maskStrokes=strokes,
        requestId=str(raw.get("requestId") or f"scene_reset_{time.time_ns()}"),
    )
    validate_scene_reset_trajectory(request, workspace_min=low, workspace_max=high)
    return request


def scene_reset_command(request: SceneResetRequest) -> str:
    return "scene_reset " + json.dumps(request.payload(), sort_keys=True, separators=(",", ":"))


def parse_scene_reset_command(command: str) -> SceneResetRequest | None:
    prefix = "scene_reset "
    if not command.startswith(prefix):
        return None
    try:
        payload = json.loads(command[len(prefix):].strip())
    except json.JSONDecodeError as exc:
        raise SceneResetError(f"Malformed scene_reset payload: {exc}") from exc
    return scene_reset_request_from_payload(payload)


def scene_reset_request_from_payload(payload: Any) -> SceneResetRequest:
    if not isinstance(payload, dict):
        raise SceneResetError("scene_reset payload must be an object.")
    lift_m = _finite_float(payload.get("liftM", SCENE_RESET_LIFT_M), "liftM")
    approach = _finite_float(payload.get("approachClearanceM", lift_m), "approachClearanceM")
    _require_fixed_lift(lift_m, "liftM")
    _require_fixed_lift(approach, "approachClearanceM")
    request = SceneResetRequest(
        pickXyz=_xyz(payload.get("pickXyz"), "pickXyz"),
        targetXyz=_xyz(payload.get("targetXyz"), "targetXyz"),
        liftM=SCENE_RESET_LIFT_M,
        approachClearanceM=SCENE_RESET_LIFT_M,
        openGripper=_finite_float(payload.get("openGripper", 1.0), "openGripper"),
        closedGripper=_finite_float(payload.get("closedGripper", 0.0), "closedGripper"),
        returnToStart=bool(payload.get("returnToStart", True)),
        timeoutS=_finite_float(payload.get("timeoutS", 20.0), "timeoutS"),
        toleranceM=_finite_float(payload.get("toleranceM", 0.006), "toleranceM"),
        gripperTolerance=_finite_float(payload.get("gripperTolerance", 0.08), "gripperTolerance"),
        graspSettleS=_finite_float(payload.get("graspSettleS", 0.6), "graspSettleS"),
        controlPeriodS=_finite_float(payload.get("controlPeriodS", 1.0 / 30.0), "controlPeriodS"),
        maskStrokes=_parse_strokes({"strokes": payload.get("maskStrokes", [])}),
        requestId=str(payload.get("requestId") or ""),
    )
    validate_scene_reset_trajectory(request)
    return request


def _robot_workspace_bounds(robot: Any) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    config = getattr(robot, "config", None)
    return _workspace_bounds(
        getattr(config, "workspace_min", None),
        getattr(config, "workspace_max", None),
    )


def _observation_xyz_rotvec_gripper(robot: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    observation = robot.get_observation(include_cameras=False)
    xyz = tuple(float(observation[key]) for key in ("ee.x", "ee.y", "ee.z"))
    rotvec = tuple(float(observation[key]) for key in ("ee.wx", "ee.wy", "ee.wz"))
    gripper = float(observation["gripper.pos"])
    return xyz, rotvec, gripper


def _send_absolute(robot: Any, xyz: tuple[float, float, float], rotvec: tuple[float, float, float], gripper: float) -> None:
    robot.send_action(
        {
            "ee.x": float(xyz[0]),
            "ee.y": float(xyz[1]),
            "ee.z": float(xyz[2]),
            "ee.wx": float(rotvec[0]),
            "ee.wy": float(rotvec[1]),
            "ee.wz": float(rotvec[2]),
            "gripper.pos": float(gripper),
        }
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)))


def _step_toward(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    max_step_m: float,
) -> tuple[float, float, float]:
    """The next setpoint along the straight line to `goal`, at most `max_step_m` from `start`."""

    delta = tuple(g - s for s, g in zip(start, goal, strict=True))
    distance = math.sqrt(sum(component * component for component in delta))
    if distance <= max_step_m or distance <= 0.0:
        return goal
    scale = max_step_m / distance
    return tuple(s + component * scale for s, component in zip(start, delta, strict=True))  # type: ignore[return-value]


def _reach_stall_error_m(robot: Any) -> float:
    """How far IK is behind the commanded pose, for a robot that can say. 0.0 for one that cannot.

    Read through `getattr` because the reset also runs against the MuJoCo twin and the test
    fakes, and an arm that cannot report a stall is not the same as one that is stalling.
    """

    try:
        return max(0.0, float(getattr(robot, "reach_stall_error_m", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _scene_reset_waits_for_gripper_position(name: str) -> bool:
    # Once the peg is clamped, the measured opening is the peg thickness, not the closed command.
    # Waiting for `closedGripper` would block the lift and carry steps forever on a successful grasp.
    return name not in {
        "close_gripper",
        "lift_8cm_after_grasp",
        "move_to_place_above",
        "descend_8cm_to_place",
    }


def _run_step(
    robot: Any,
    request: SceneResetRequest,
    name: str,
    xyz: tuple[float, float, float],
    rotvec: tuple[float, float, float],
    gripper: float,
) -> None:
    print(
        f"[INFO] scene_reset_step=start request_id={request.requestId} name={name} "
        f"xyz={xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f} gripper={gripper:.3f}",
        flush=True,
    )
    # Where the setpoint starts walking from. The previous step converged to within toleranceM
    # of its own waypoint, so this is that waypoint up to a few mm -- close enough that the
    # walked line and the line the QC sampled are the same line.
    commanded, _start_rotvec, _start_gripper = _observation_xyz_rotvec_gripper(robot)
    max_step_m = SCENE_RESET_MAX_SPEED_MS * request.controlPeriodS
    # timeoutS keeps its meaning of "how long this step may take to settle"; the walk to the
    # waypoint is granted on top of it. Folding travel into the timeout would turn slowing the
    # setpoint down into a timeout on waypoints that are perfectly reachable.
    deadline = (
        time.perf_counter()
        + request.timeoutS
        + _distance(commanded, xyz) / SCENE_RESET_MAX_SPEED_MS
    )
    last_error = ""
    position_reached_at: float | None = None
    stalled_since: float | None = None
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        commanded = _step_toward(commanded, xyz, max_step_m)
        _send_absolute(robot, commanded, rotvec, gripper)
        current_xyz, _current_rotvec, current_gripper = _observation_xyz_rotvec_gripper(robot)
        stall_m = _reach_stall_error_m(robot)
        if stall_m <= 0.0:
            stalled_since = None
        elif stalled_since is None:
            stalled_since = now
        elif now - stalled_since >= SCENE_RESET_REACH_STALL_S:
            raise SceneResetUnreachableError(
                f"scene reset step {name} asks for a tool point the arm cannot reach: IK has "
                f"been {stall_m * 1000.0:.1f} mm short of the commanded pose for "
                f"{now - stalled_since:.1f}s on the way to "
                f"({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}). This is the arm's own reach at "
                f"this tool orientation, not the workspace fence."
            )
        pos_error = _distance(current_xyz, xyz)
        gripper_error = abs(current_gripper - gripper)
        last_error = f"pos_err_mm={pos_error * 1000.0:.1f} gripper_err={gripper_error:.3f}"
        # The walked setpoint has to land on the waypoint before the step can be done, not merely
        # get within toleranceM of it. Otherwise a step converges on the last interpolated point
        # and the arm is left a few mm short of a coordinate that was asked for exactly -- which
        # for the pose probe, whose whole job is putting the tool at a known base coordinate for
        # the camera to be solved against, is calibration error rather than tracking error.
        pos_ok = commanded == xyz and pos_error <= request.toleranceM
        gripper_ok = gripper_error <= request.gripperTolerance
        if pos_ok and position_reached_at is None:
            position_reached_at = now
        elif not pos_ok:
            position_reached_at = None
        waits_for_gripper = _scene_reset_waits_for_gripper_position(name)
        if waits_for_gripper:
            gripper_wait_done = gripper_ok
            gripper_wait = "position"
        elif name == "close_gripper":
            gripper_wait_done = gripper_ok or (
                position_reached_at is not None
                and now - position_reached_at >= request.graspSettleS
            )
            gripper_wait = "position" if gripper_ok else "grasp_settle"
        else:
            gripper_wait_done = True
            gripper_wait = "carried_object"
        if pos_ok and gripper_wait_done:
            print(
                f"[INFO] scene_reset_step=done request_id={request.requestId} name={name} "
                f"{last_error} gripper_wait={gripper_wait}",
                flush=True,
            )
            return
        precise_sleep(request.controlPeriodS)
    raise TimeoutError(f"scene reset step {name} timed out: {last_error}")


def _move_to_start(robot: Any) -> None:
    """Home the arm, slowly on an arm that can be told to.

    The signature is inspected rather than the call being retried on TypeError, so a TypeError
    raised from inside a homing move is never mistaken for an older signature and silently
    re-run at full speed. The MuJoCo twin and the test fakes take no argument and are homed as
    they always were.
    """

    try:
        accepts_speed_scale = "speed_scale" in inspect.signature(robot.move_to_start).parameters
    except (TypeError, ValueError):
        accepts_speed_scale = False
    if accepts_speed_scale:
        robot.move_to_start(speed_scale=SCENE_RESET_HOMING_SPEED_SCALE)
    else:
        robot.move_to_start()


def _hold_where_it_is(robot: Any, gripper: float) -> None:
    """Stop asking for the point the arm could not reach.

    A step that fails leaves the driver's OTG target at the IK solution for a pose the arm never
    realised, and nothing in the loop clears it: the arm goes on leaning into its own limit for
    as long as the process lives, which is how a failed reset ended with the tool held 62 cm up,
    straining, until the runtime was killed. Commanding the pose it is measurably at replaces
    that target with one it is already holding.

    The gripper keeps whatever it was last told. A reset that fails carrying the peg should not
    also drop it from wherever it stopped.
    """

    xyz, rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    _send_absolute(robot, xyz, rotvec, gripper)


def _abort_scene_reset(robot: Any, request: SceneResetRequest, *, gripper: float) -> dict[str, Any]:
    """Bring a failed reset to rest, and home the arm if the request asked for homing.

    Every step here is wrapped, because recovery runs while an error is already being reported
    and the operator needs to read that error rather than one raised trying to tidy up after it.
    """

    returned_to_start = False
    try:
        _hold_where_it_is(robot, gripper)
        print(f"[INFO] scene_reset_abort=holding request_id={request.requestId}", flush=True)
    except Exception as exc:  # noqa: BLE001 - recovery must not replace the failure being reported
        print(
            f"[WARN] scene_reset_abort=hold_failed request_id={request.requestId} details={exc}",
            flush=True,
        )
    if request.returnToStart:
        try:
            _move_to_start(robot)
            returned_to_start = True
            print(
                f"[INFO] scene_reset_abort=returned_to_start request_id={request.requestId}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - same reason as the hold above
            print(
                f"[WARN] scene_reset_abort=return_to_start_failed "
                f"request_id={request.requestId} details={exc}",
                flush=True,
            )
    return {"returnedToStart": returned_to_start}


def execute_scene_reset(robot: Any, request: SceneResetRequest) -> dict[str, Any]:
    """Execute the fixed-pick/random-place reset on an already connected FR3 robot."""

    current_xyz, rotvec, _ = _observation_xyz_rotvec_gripper(robot)
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    try:
        qc = validate_scene_reset_trajectory(
            request,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            current_xyz=current_xyz,
            # The tool orientation the whole reset is executed at, so the reach check asks about
            # the poses that will actually be commanded rather than about bare points.
            reach_probe=_reach_probe(robot, rotvec),
        )
    except SceneResetError as exc:
        print(f"[WARN] scene_reset=failed request_id={request.requestId} details=trajectory_qc_failed: {exc}", flush=True)
        return {"ok": False, "error": f"trajectory_qc_failed: {exc}", "request": request.payload()}

    print(
        f"[INFO] scene_reset=start request_id={request.requestId} "
        f"pick_xyz={request.pickXyz[0]:+.4f},{request.pickXyz[1]:+.4f},{request.pickXyz[2]:+.4f} "
        f"target_xyz={request.targetXyz[0]:+.4f},{request.targetXyz[1]:+.4f},{request.targetXyz[2]:+.4f} "
        f"lift_m={SCENE_RESET_LIFT_M:.3f} trajectory_qc=passed waypoints={qc['waypoints']} "
        f"reach_checked={qc['reachCheckedPoints']}",
        flush=True,
    )
    # What the gripper was last told, so an abort can hold the arm still without deciding on its
    # own whether the peg is dropped. Seeded open: nothing has been commanded yet at this point.
    commanded_gripper = request.openGripper
    try:
        for waypoint in build_scene_reset_waypoints(request):
            commanded_gripper = waypoint.gripper
            _run_step(robot, request, waypoint.name, waypoint.xyz, rotvec, waypoint.gripper)
        if request.returnToStart:
            print(f"[INFO] scene_reset_step=start request_id={request.requestId} name=return_to_start", flush=True)
            _move_to_start(robot)
            print(f"[INFO] scene_reset_step=done request_id={request.requestId} name=return_to_start", flush=True)
        print(f"[INFO] scene_reset=done request_id={request.requestId}", flush=True)
        return {
            "ok": True,
            "request": request.payload(),
            "trajectoryQc": qc,
            "returnedToStart": bool(request.returnToStart),
        }
    except Exception as exc:  # noqa: BLE001 - the caller reports this without killing the gateway
        print(f"[WARN] scene_reset=failed request_id={request.requestId} details={exc}", flush=True)
        recovery = _abort_scene_reset(robot, request, gripper=commanded_gripper)
        return {
            "ok": False,
            "error": str(exc),
            "request": request.payload(),
            "trajectoryQc": qc,
            **recovery,
        }


# ------------------------------------------------------------------------------ pose probe ---
# Probing is the reset's little sibling: one commanded point, no object, nothing sampled. It
# exists so the camera can be tied to the base frame without a calibration board -- the robot
# puts its own tool at a base coordinate the gateway chose, a still is taken there, and an
# operator clicks the tool in it. Four of those pairs are a plane homography (see
# tools/data_collection_gui/table_plane.py).
#
# It reuses the reset's clearance, QC and step loop deliberately. A second way of moving the
# arm to a typed-in coordinate is a second set of workspace checks to keep in step, and the
# one that gets skipped is always the one nobody thought of as a real motion.
POSE_PROBE_CLEARANCE_M = SCENE_RESET_LIFT_M


@dataclass(frozen=True)
class PoseProbeRequest:
    xyz: tuple[float, float, float]
    # Closed, so the tool is a single point in the image instead of two fingers with a gap
    # between them that every operator would bisect slightly differently.
    gripper: float = 0.0
    # Time held at the point before the still is taken. The camera pipeline is several frames
    # deep, so a frame grabbed the instant the controller reports convergence can still show
    # the arm arriving.
    dwellS: float = 0.6
    timeoutS: float = 20.0
    toleranceM: float = 0.006
    gripperTolerance: float = 0.08
    controlPeriodS: float = 1.0 / 30.0
    requestId: str = ""

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def build_pose_probe_waypoints(request: PoseProbeRequest) -> tuple[SceneResetWaypoint, ...]:
    """Down from above, then straight back up.

    The retreat is a waypoint rather than something the caller does afterwards so that the QC
    below covers it: the arm leaves the probe point at the same clearance it arrived at, and
    never travels laterally at table height.
    """

    x, y, z = request.xyz
    above = (x, y, z + POSE_PROBE_CLEARANCE_M)
    return (
        SceneResetWaypoint("approach_above_probe", above, request.gripper),
        SceneResetWaypoint("descend_8cm_to_probe", (x, y, z), request.gripper),
        SceneResetWaypoint("retreat_8cm_from_probe", above, request.gripper),
    )


def validate_pose_probe_trajectory(
    request: PoseProbeRequest,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
    current_xyz: tuple[float, float, float] | None = None,
    reach_probe: Any = None,
) -> dict[str, Any]:
    """Run deterministic probe QC before any waypoint is sent to the robot.

    `reach_probe` as in the reset's QC: optional, last, and the only check that needs a robot.
    """

    if request.timeoutS <= 0.0 or request.toleranceM <= 0.0 or request.controlPeriodS <= 0.0:
        raise SceneResetError("timeoutS, toleranceM and controlPeriodS must be positive.")
    if request.dwellS < 0.0 or request.dwellS > 5.0:
        raise SceneResetError("dwellS must be in [0, 5] seconds.")
    if request.gripperTolerance < 0.0:
        raise SceneResetError("gripperTolerance must be non-negative.")
    if request.gripper < 0.0 or request.gripper > 1.0:
        raise SceneResetError("gripper must be normalized in [0, 1].")
    for index, value in enumerate(request.xyz):
        if not math.isfinite(value):
            raise SceneResetError(f"xyz[{index}] must be finite.")

    low, high = _workspace_bounds(workspace_min, workspace_max)
    waypoints = build_pose_probe_waypoints(request)
    points: list[tuple[str, tuple[float, float, float]]] = [
        (waypoint.name, waypoint.xyz) for waypoint in waypoints
    ]
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
    return {
        "ok": True,
        "waypoints": len(waypoints),
        "clearanceM": POSE_PROBE_CLEARANCE_M,
        "reachCheckedPoints": reach_checked,
    }


def sanitize_pose_probe_request(
    raw: Any,
    *,
    workspace_min: Iterable[float] | None = None,
    workspace_max: Iterable[float] | None = None,
) -> PoseProbeRequest:
    """Validate a UI probe request before the arm is asked to visit the point."""

    if not isinstance(raw, dict):
        raise SceneResetError("pose probe payload must be a JSON object.")
    request = PoseProbeRequest(
        xyz=_xyz(raw.get("xyz"), "xyz"),
        gripper=_finite_float(raw.get("gripper", 0.0), "gripper"),
        dwellS=_finite_float(raw.get("dwellS", 0.6), "dwellS"),
        timeoutS=_finite_float(raw.get("timeoutS", 20.0), "timeoutS"),
        toleranceM=_finite_float(raw.get("toleranceM", 0.006), "toleranceM"),
        gripperTolerance=_finite_float(raw.get("gripperTolerance", 0.08), "gripperTolerance"),
        controlPeriodS=_finite_float(raw.get("controlPeriodS", 1.0 / 30.0), "controlPeriodS"),
        requestId=str(raw.get("requestId") or f"pose_probe_{time.time_ns()}"),
    )
    validate_pose_probe_trajectory(request, workspace_min=workspace_min, workspace_max=workspace_max)
    return request


def pose_probe_command(request: PoseProbeRequest) -> str:
    return "probe_pose " + json.dumps(request.payload(), sort_keys=True, separators=(",", ":"))


def pose_probe_request_from_payload(payload: Any) -> PoseProbeRequest:
    if not isinstance(payload, dict):
        raise SceneResetError("probe_pose payload must be an object.")
    request = PoseProbeRequest(
        xyz=_xyz(payload.get("xyz"), "xyz"),
        gripper=_finite_float(payload.get("gripper", 0.0), "gripper"),
        dwellS=_finite_float(payload.get("dwellS", 0.6), "dwellS"),
        timeoutS=_finite_float(payload.get("timeoutS", 20.0), "timeoutS"),
        toleranceM=_finite_float(payload.get("toleranceM", 0.006), "toleranceM"),
        gripperTolerance=_finite_float(payload.get("gripperTolerance", 0.08), "gripperTolerance"),
        controlPeriodS=_finite_float(payload.get("controlPeriodS", 1.0 / 30.0), "controlPeriodS"),
        requestId=str(payload.get("requestId") or ""),
    )
    validate_pose_probe_trajectory(request)
    return request


def execute_pose_probe(robot: Any, request: PoseProbeRequest, *, on_arrival: Any = None) -> dict[str, Any]:
    """Put the tool at one base coordinate, hold, call back, and retreat.

    `on_arrival` runs while the arm is standing at the point, and its job is to write the
    still that will be clicked. It runs inside the probe rather than after it because the
    caller's next act is to hand the arm back to the waiting loop, which on this runtime homes
    it -- a snapshot taken there would show an empty table and a point somewhere else.
    """

    current_xyz, rotvec, _gripper = _observation_xyz_rotvec_gripper(robot)
    workspace_min, workspace_max = _robot_workspace_bounds(robot)
    try:
        qc = validate_pose_probe_trajectory(
            request,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            current_xyz=current_xyz,
            reach_probe=_reach_probe(robot, rotvec),
        )
    except SceneResetError as exc:
        print(f"[WARN] pose_probe=failed request_id={request.requestId} details=trajectory_qc_failed: {exc}", flush=True)
        return {"ok": False, "error": f"trajectory_qc_failed: {exc}", "request": request.payload()}

    print(
        f"[INFO] pose_probe=start request_id={request.requestId} "
        f"xyz={request.xyz[0]:+.4f},{request.xyz[1]:+.4f},{request.xyz[2]:+.4f} "
        f"clearance_m={POSE_PROBE_CLEARANCE_M:.3f} trajectory_qc=passed waypoints={qc['waypoints']} "
        f"reach_checked={qc['reachCheckedPoints']}",
        flush=True,
    )
    try:
        for waypoint in build_pose_probe_waypoints(request):
            _run_step(robot, request, waypoint.name, waypoint.xyz, rotvec, waypoint.gripper)
            if waypoint.name == "descend_8cm_to_probe":
                precise_sleep(request.dwellS)
                if callable(on_arrival):
                    on_arrival()
        print(f"[INFO] pose_probe=done request_id={request.requestId}", flush=True)
        return {"ok": True, "request": request.payload(), "trajectoryQc": qc}
    except Exception as exc:  # noqa: BLE001 - the caller reports this without killing the session
        print(f"[WARN] pose_probe=failed request_id={request.requestId} details={exc}", flush=True)
        # Same reasoning as the reset's abort: a probe that runs out of reach leaves the OTG
        # target on a pose the arm never made, and the waiting loop it returns to does not
        # command the arm at all. There is nothing to home here -- the probe never claimed the
        # start pose -- so the hold is the whole recovery.
        try:
            _hold_where_it_is(robot, request.gripper)
            print(f"[INFO] pose_probe_abort=holding request_id={request.requestId}", flush=True)
        except Exception as hold_exc:  # noqa: BLE001 - must not replace the failure above
            print(
                f"[WARN] pose_probe_abort=hold_failed request_id={request.requestId} "
                f"details={hold_exc}",
                flush=True,
            )
        return {"ok": False, "error": str(exc), "request": request.payload(), "trajectoryQc": qc}
