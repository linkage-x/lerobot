import math
import random
import time

import pytest

import tools.fr3.scene_reset as scene_reset
from tools.fr3.scene_reset import (
    SceneResetError,
    build_pose_probe_waypoints,
    execute_pose_probe,
    execute_scene_reset,
    sanitize_pose_probe_request,
    sanitize_scene_reset_request,
    validate_scene_reset_trajectory,
)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    """Let the suite run at wall-clock speed rather than the arm's.

    `_run_step` walks its setpoint at SCENE_RESET_MAX_SPEED_MS, so a reset costs
    (total travel / speed) seconds however short controlPeriodS is -- which is the point on
    hardware and pure waiting here. The rate limit itself is covered below by tests that set
    their own speed and assert against that.
    """

    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)


class FakeRobot:
    def __init__(self):
        self.xyz = (0.30, 0.0, 0.20)
        self.rotvec = (0.0, 0.0, 0.0)
        self.gripper = 1.0
        self.actions = []
        self.move_to_start_calls = 0

    def get_observation(self, *, include_cameras=False):
        return {
            "ee.x": self.xyz[0],
            "ee.y": self.xyz[1],
            "ee.z": self.xyz[2],
            "ee.wx": self.rotvec[0],
            "ee.wy": self.rotvec[1],
            "ee.wz": self.rotvec[2],
            "gripper.pos": self.gripper,
        }

    def send_action(self, action):
        self.actions.append(dict(action))
        self.xyz = (action["ee.x"], action["ee.y"], action["ee.z"])
        self.rotvec = (action["ee.wx"], action["ee.wy"], action["ee.wz"])
        self.gripper = action["gripper.pos"]
        return dict(action)

    def move_to_start(self):
        self.move_to_start_calls += 1


class FakePegBlockedRobot(FakeRobot):
    def send_action(self, action):
        self.actions.append(dict(action))
        self.xyz = (action["ee.x"], action["ee.y"], action["ee.z"])
        self.rotvec = (action["ee.wx"], action["ee.wy"], action["ee.wz"])
        commanded_gripper = float(action["gripper.pos"])
        self.gripper = 0.25 if commanded_gripper <= 0.0 else commanded_gripper
        return dict(action)


class FakeStalledRobot(FakeRobot):
    """An arm that runs out of reach at a ceiling: it stops climbing and reports how far short.

    Mirrors the real driver rather than inventing a failure. FrankaResearch3 clips its IK step
    to the URDF joint limits and returns the joints it already had -- no exception, no error
    state, an arm that simply stops moving in one direction -- and publishes the shortfall on
    `reach_stall_error_m`, zeroed below its own 5 mm tolerance.
    """

    def __init__(self, ceiling_z):
        super().__init__()
        self.ceiling_z = float(ceiling_z)
        self.reach_stall_error_m = 0.0

    def send_action(self, action):
        commanded_z = float(action["ee.z"])
        realised_z = min(commanded_z, self.ceiling_z)
        shortfall = commanded_z - realised_z
        self.reach_stall_error_m = shortfall if shortfall > 0.005 else 0.0
        return super().send_action({**action, "ee.z": realised_z})


class FakeReachCheckedRobot(FakeRobot):
    """An arm that can say, on the model, how far short IK falls of a tool point.

    Same shape as FrankaResearch3.iter_reach_errors_m: the points are taken eagerly, the solving
    is lazy, and a point above the arm's reach reports the shortfall rather than raising. The
    ceiling stands in for the real thing -- a tool orientation the last configuration cannot hold
    -- because what matters downstream is only that the arm answers with a distance.
    """

    def __init__(self, ceiling_z=math.inf):
        super().__init__()
        self.ceiling_z = float(ceiling_z)
        self.reach_points_asked = []
        self.reach_points_solved = 0

    def iter_reach_errors_m(self, tool_points, rotvec):
        del rotvec
        tool_points = list(tool_points)
        self.reach_points_asked.append(tool_points)
        return self._iter_reach_errors_m(tool_points)

    def _iter_reach_errors_m(self, tool_points):
        for point in tool_points:
            self.reach_points_solved += 1
            yield max(0.0, float(point[2]) - self.ceiling_z)


class FakeSlowHomingRobot(FakeRobot):
    """An arm whose move_to_start takes a speed scale, as FrankaResearch3's does."""

    def __init__(self):
        super().__init__()
        self.move_to_start_speed_scales = []

    def move_to_start(self, speed_scale=1.0):
        self.move_to_start_calls += 1
        self.move_to_start_speed_scales.append(float(speed_scale))


class FakeStalledSlowHomingRobot(FakeStalledRobot):
    """The stalling arm, homing on the newer signature."""

    def __init__(self, ceiling_z):
        super().__init__(ceiling_z)
        self.move_to_start_speed_scales = []

    def move_to_start(self, speed_scale=1.0):
        self.move_to_start_calls += 1
        self.move_to_start_speed_scales.append(float(speed_scale))


def request_payload(**overrides):
    payload = {
        "pickXyz": [0.40, 0.00, 0.035],
        "targetZ": 0.035,
        "liftM": 0.08,
        "mask": {"strokes": [{"x": 0.45, "y": -0.05, "radiusM": 0.01}]},
    }
    payload.update(overrides)
    return payload


def test_scene_reset_samples_target_from_painted_mask_and_keeps_z_fixed():
    request = sanitize_scene_reset_request(
        request_payload(),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    assert request.targetXyz[2] == pytest.approx(0.035)
    assert abs(request.targetXyz[0] - 0.45) <= 0.01
    assert abs(request.targetXyz[1] + 0.05) <= 0.01


def test_scene_reset_refuses_non_fixed_eight_centimetre_lift():
    with pytest.raises(SceneResetError, match="8 cm"):
        sanitize_scene_reset_request(request_payload(liftM=0.12))


def test_scene_reset_executes_lift_before_horizontal_transfer():
    request = sanitize_scene_reset_request(request_payload(), rng=random.Random(1))
    robot = FakeRobot()

    result = execute_scene_reset(robot, request)

    assert result["ok"] is True
    xyzs = [(action["ee.x"], action["ee.y"], action["ee.z"]) for action in robot.actions]
    assert request.pickXyz in xyzs
    lifted = (request.pickXyz[0], request.pickXyz[1], request.pickXyz[2] + 0.08)
    target_above = (request.targetXyz[0], request.targetXyz[1], request.targetXyz[2] + 0.08)
    assert lifted in xyzs
    assert target_above in xyzs
    assert xyzs.index(lifted) < xyzs.index(target_above)
    assert robot.move_to_start_calls == 1


def test_scene_reset_lifts_after_gripper_feedback_stops_on_peg():
    request = sanitize_scene_reset_request(
        request_payload(gripperTolerance=0.01, graspSettleS=0.0),
        rng=random.Random(1),
    )
    robot = FakePegBlockedRobot()

    result = execute_scene_reset(robot, request)

    assert result["ok"] is True
    xyzs = [(action["ee.x"], action["ee.y"], action["ee.z"]) for action in robot.actions]
    lifted = (request.pickXyz[0], request.pickXyz[1], request.pickXyz[2] + 0.08)
    target_above = (request.targetXyz[0], request.targetXyz[1], request.targetXyz[2] + 0.08)
    assert lifted in xyzs
    assert target_above in xyzs
    assert xyzs.index(lifted) < xyzs.index(target_above)


def test_scene_reset_trajectory_qc_rejects_workspace_escape():
    request = sanitize_scene_reset_request(request_payload(), rng=random.Random(1))

    with pytest.raises(SceneResetError, match="outside the robot workspace"):
        validate_scene_reset_trajectory(
            request,
            workspace_min=(0.18, -0.45, 0.0),
            workspace_max=(0.70, 0.45, 0.10),
        )


def test_a_pose_probe_descends_from_above_and_backs_off_the_same_way():
    request = sanitize_pose_probe_request({"xyz": [0.45, 0.05, 0.035]})

    names = [waypoint.name for waypoint in build_pose_probe_waypoints(request)]
    heights = [waypoint.xyz[2] for waypoint in build_pose_probe_waypoints(request)]

    assert names == ["approach_above_probe", "descend_8cm_to_probe", "retreat_8cm_from_probe"]
    assert heights == [0.115, 0.035, 0.115]
    # The tool never travels sideways at table height, which is the whole reason the approach
    # and the retreat are waypoints rather than something the caller does around this.
    assert all(
        waypoint.xyz[:2] == (0.45, 0.05) for waypoint in build_pose_probe_waypoints(request)
    )


def test_the_probe_still_is_taken_while_the_arm_is_standing_at_the_point():
    """The snapshot has to happen between the descent and the retreat.

    Taken any later it shows an empty table -- the waiting loop homes the arm the moment the
    probe returns -- and the click on it would be recorded against a coordinate the tool was
    nowhere near.
    """

    request = sanitize_pose_probe_request({"xyz": [0.45, 0.0, 0.035], "dwellS": 0.0})
    robot = FakeRobot()
    seen: list[tuple[float, float, float]] = []

    result = execute_pose_probe(robot, request, on_arrival=lambda: seen.append(robot.xyz))

    assert result["ok"] is True
    assert seen == [(0.45, 0.0, 0.035)]
    assert robot.xyz == (0.45, 0.0, 0.115)


def test_a_probe_outside_the_workspace_is_refused_before_the_arm_moves():
    with pytest.raises(SceneResetError, match="outside the robot workspace"):
        sanitize_pose_probe_request(
            {"xyz": [0.90, 0.0, 0.035]},
            workspace_min=(0.18, -0.45, 0.0),
            workspace_max=(0.70, 0.45, 0.70),
        )


def test_a_probe_whose_approach_leaves_the_workspace_is_refused_too():
    # The point itself is legal and the 8 cm above it is not, which is exactly the case a check
    # on the commanded coordinate alone would wave through.
    with pytest.raises(SceneResetError, match="outside the robot workspace"):
        sanitize_pose_probe_request(
            {"xyz": [0.45, 0.0, 0.035]},
            workspace_min=(0.18, -0.45, 0.0),
            workspace_max=(0.70, 0.45, 0.10),
        )


def test_a_far_waypoint_is_walked_to_rather_than_commanded_as_one_jump(monkeypatch):
    """The place approach used to arrive as a single 514 mm absolute pose.

    `send_action` clamps a relative command to `max_target_delta_pos` -- 1 mm a step on this rig
    -- but its absolute branch, the only one a reset uses, clamps nothing, and the OTG beneath
    it is configured at the FR3's rated joint maxima. A jump that size was crossed at roughly a
    metre a second with the peg in the gripper.
    """

    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 0.5)
    request = sanitize_scene_reset_request(
        request_payload(controlPeriodS=0.01), rng=random.Random(1)
    )
    robot = FakeRobot()

    assert execute_scene_reset(robot, request)["ok"] is True

    xyzs = [(action["ee.x"], action["ee.y"], action["ee.z"]) for action in robot.actions]
    steps = [math.dist(before, after) for before, after in zip(xyzs, xyzs[1:])]
    assert steps, "the reset commanded no motion at all"
    assert max(steps) <= 0.5 * request.controlPeriodS + 1e-9
    # The limit is on how fast the setpoint travels, not on where it is allowed to end up.
    target_above = (request.targetXyz[0], request.targetXyz[1], request.targetXyz[2] + 0.08)
    assert target_above in xyzs


def test_a_waypoint_past_the_arms_reach_fails_in_seconds_not_at_the_timeout():
    """A stalled step used to spend its whole 20 s leaning on a joint limit before giving up."""

    request = sanitize_scene_reset_request(
        request_payload(targetZ=0.30, timeoutS=20.0), rng=random.Random(1)
    )
    robot = FakeStalledRobot(ceiling_z=0.20)

    started = time.perf_counter()
    result = execute_scene_reset(robot, request)
    elapsed = time.perf_counter() - started

    assert result["ok"] is False
    assert "cannot reach" in result["error"]
    assert elapsed < request.timeoutS


def test_a_failed_reset_stops_pushing_and_homes_instead_of_holding_the_strain():
    request = sanitize_scene_reset_request(request_payload(targetZ=0.30), rng=random.Random(1))
    robot = FakeStalledRobot(ceiling_z=0.20)

    result = execute_scene_reset(robot, request)

    assert result["ok"] is False
    # The last command is the pose the arm is measurably at, so the driver's OTG target stops
    # being one it never reached. Before this the arm kept leaning until the process was killed.
    last = robot.actions[-1]
    assert (last["ee.x"], last["ee.y"], last["ee.z"]) == pytest.approx(robot.xyz)
    # Still holding whatever it was carrying: an abort must not also drop the peg mid-air.
    assert last["gripper.pos"] == request.closedGripper
    assert robot.move_to_start_calls == 1
    assert result["returnedToStart"] is True


def test_a_reset_that_never_reaches_its_waypoint_reports_it_did_not_home():
    """`returnedToStart` is what the runtime sets `arm_at_start` from, so it has to be measured."""

    request = sanitize_scene_reset_request(
        request_payload(targetZ=0.30, returnToStart=False), rng=random.Random(1)
    )
    robot = FakeStalledRobot(ceiling_z=0.20)

    result = execute_scene_reset(robot, request)

    assert result["ok"] is False
    assert result["returnedToStart"] is False
    assert robot.move_to_start_calls == 0


def test_a_target_the_arm_cannot_reach_is_refused_before_anything_moves():
    """The whole point of asking IK during QC: no motion at all, not a motion that stops badly.

    The stall detector in `_run_step` catches this too, but only from the arm standing at the
    edge of its own reach, half-way through a transfer with the peg gripped. Here the same fact
    is known while the arm is still parked, so the reset is simply declined.
    """

    robot = FakeReachCheckedRobot(ceiling_z=0.50)
    request = sanitize_scene_reset_request(
        request_payload(targetZ=0.55),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    result = execute_scene_reset(robot, request)

    assert result["ok"] is False
    assert "reach" in result["error"]
    # Not the fence: the request passed that, which is exactly why this check had to exist.
    assert "workspace fence" in result["error"]
    assert robot.actions == []
    assert robot.move_to_start_calls == 0


def test_the_reach_check_stops_at_the_first_point_it_cannot_reach():
    """A refused trajectory must not cost a full solve for every point after the bad one.

    Each unreachable point costs the real solver its whole iteration budget, and the QC samples
    the path every centimetre, so an eager check would spend that budget a hundred times over
    to learn what the first failure already said.
    """

    robot = FakeReachCheckedRobot(ceiling_z=0.50)
    request = sanitize_scene_reset_request(
        request_payload(targetZ=0.55),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    execute_scene_reset(robot, request)

    assert robot.reach_points_solved < len(robot.reach_points_asked[0])


def test_a_reachable_reset_reports_how_much_of_its_path_the_reach_check_covered():
    robot = FakeReachCheckedRobot()
    request = sanitize_scene_reset_request(
        request_payload(),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    result = execute_scene_reset(robot, request)

    assert result["ok"] is True
    # Every sampled point of the walked path, not just the eight waypoints: reach gives out
    # between waypoints as readily as at one.
    checked = result["trajectoryQc"]["reachCheckedPoints"]
    assert checked == robot.reach_points_solved
    assert checked > len(scene_reset.build_scene_reset_waypoints(request))


def test_a_robot_that_cannot_check_its_own_reach_still_resets():
    """The MuJoCo twin and older backends answer nothing. That is not the same as answering badly."""

    robot = FakeRobot()
    request = sanitize_scene_reset_request(
        request_payload(),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    result = execute_scene_reset(robot, request)

    assert result["ok"] is True
    assert result["trajectoryQc"]["reachCheckedPoints"] == 0


def test_a_reset_homes_at_a_fraction_of_the_arms_rated_speed():
    """Homing is a joint-space move at the OTG's rated ceilings -- the reset's fastest motion.

    Rate-limiting the Cartesian steps and leaving this one flat out would move the violence of a
    reset into the step nobody was watching.
    """

    robot = FakeSlowHomingRobot()
    request = sanitize_scene_reset_request(
        request_payload(),
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
        rng=random.Random(7),
    )

    result = execute_scene_reset(robot, request)

    assert result["ok"] is True
    assert robot.move_to_start_speed_scales == [scene_reset.SCENE_RESET_HOMING_SPEED_SCALE]


def test_a_failed_reset_homes_slowly_too():
    """The abort path needs this more than the success path does.

    It reaches homing from states the success path never produces: half-way through a transfer,
    at the edge of the arm's reach, still holding the peg.
    """

    request = sanitize_scene_reset_request(request_payload(targetZ=0.30), rng=random.Random(1))
    robot = FakeStalledSlowHomingRobot(ceiling_z=0.20)

    result = execute_scene_reset(robot, request)

    assert result["ok"] is False
    assert result["returnedToStart"] is True
    assert robot.move_to_start_speed_scales == [scene_reset.SCENE_RESET_HOMING_SPEED_SCALE]


def test_a_pose_probe_past_the_arms_reach_is_refused_before_anything_moves():
    robot = FakeReachCheckedRobot(ceiling_z=0.30)
    request = sanitize_pose_probe_request(
        {"xyz": [0.45, 0.0, 0.40]},
        workspace_min=(0.18, -0.45, 0.0),
        workspace_max=(0.70, 0.45, 0.70),
    )

    result = execute_pose_probe(robot, request)

    assert result["ok"] is False
    assert "reach" in result["error"]
    assert robot.actions == []
