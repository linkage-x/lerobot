import random

import pytest

from tools.fr3.scene_reset import (
    SceneResetError,
    execute_scene_reset,
    sanitize_scene_reset_request,
    validate_scene_reset_trajectory,
)


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
