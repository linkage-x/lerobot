#!/usr/bin/env python

from __future__ import annotations

import threading
import numpy as np
import pytest

from lerobot.envs.fr3_mujoco import FR3MujocoEnv

from tools.fr3 import fr3_mujoco_record


def test_get_env_observation_includes_camera_images():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
        "camera_obs": {
            "third_person": np.zeros((480, 640, 3), dtype=np.uint8),
            "side": np.ones((480, 640, 3), dtype=np.uint8),
            "wrist": np.full((480, 640, 3), 2, dtype=np.uint8),
        },
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=1.0)

    assert observation["third_person"].shape == (480, 640, 3)
    assert observation["side"][0, 0, 0] == 1
    assert observation["wrist"][0, 0, 0] == 2


def test_get_env_observation_extracts_ee_pose_and_joints():
    ee_pose = np.eye(4, dtype=np.float64)
    ee_pose[0, 3] = 0.5
    ee_pose[1, 3] = 0.3
    ee_pose[2, 3] = 0.2
    info = {
        "ee_pose": ee_pose,
        "joint_positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64),
        "camera_obs": {},
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=0.5)

    assert observation["ee.x"] == 0.5
    assert observation["ee.y"] == 0.3
    assert observation["ee.z"] == 0.2
    assert observation["gripper.pos"] == 0.5
    for i in range(1, 8):
        assert np.isclose(observation[f"joint_{i}.pos"], float(i) * 0.1)


def test_get_env_observation_without_cameras():
    info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
    }

    observation = fr3_mujoco_record._get_env_observation(info, gripper_pos=0.0)

    for camera_name in fr3_mujoco_record._SIM_CAMERA_NAMES:
        assert camera_name not in observation
    assert observation["gripper.pos"] == 0.0


def test_complete_robot_observation_adds_prev_cmd_quats():
    observation = {
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
    }

    completed = fr3_mujoco_record._complete_robot_observation(observation)

    assert completed["prev_cmd.ee.qx"] == 0.0
    assert completed["prev_cmd.ee.qy"] == 0.0
    assert completed["prev_cmd.ee.qz"] == 0.0
    assert completed["prev_cmd.ee.qw"] == 1.0


def test_complete_robot_observation_preserves_existing_prev_cmd():
    observation = {
        "ee.qx": 0.1,
        "ee.qy": 0.2,
        "ee.qz": 0.3,
        "ee.qw": 0.4,
        "prev_cmd.ee.qx": 999.0,
    }

    completed = fr3_mujoco_record._complete_robot_observation(observation)

    assert completed["prev_cmd.ee.qx"] == 999.0
    assert completed["prev_cmd.ee.qy"] == 0.2
    assert completed["prev_cmd.ee.qz"] == 0.3
    assert completed["prev_cmd.ee.qw"] == 0.4


def test_build_robot_observation_features_includes_joints():
    features = fr3_mujoco_record._build_robot_observation_features(include_cameras=False)

    assert features["ee.x"] is float
    assert features["ee.y"] is float
    assert features["ee.z"] is float
    assert features["ee.wx"] is float
    assert features["ee.wy"] is float
    assert features["ee.wz"] is float
    assert features["gripper.pos"] is float
    for i in range(1, 8):
        assert features[f"joint_{i}.pos"] is float
    for camera_name in fr3_mujoco_record._SIM_CAMERA_NAMES:
        assert camera_name not in features


def test_build_robot_observation_features_with_cameras():
    features = fr3_mujoco_record._build_robot_observation_features(
        include_cameras=True, camera_shape=(480, 640, 3)
    )

    for camera_name in fr3_mujoco_record._SIM_CAMERA_NAMES:
        assert features[camera_name] == (480, 640, 3)
    assert features["gripper.pos"] is float


def test_sample_disk_xy_stays_within_requested_radius():
    center_xy = np.array([0.48, 0.0], dtype=np.float64)
    rng = np.random.default_rng(0)

    for _ in range(128):
        sample_xy = fr3_mujoco_record._sample_disk_xy(center_xy, 0.10, rng)
        assert np.linalg.norm(sample_xy - center_xy) <= 0.10 + 1e-9


def test_reset_workspace_object_for_episode_randomizes_pose_within_disk():
    env = FR3MujocoEnv()
    rng = np.random.default_rng(0)
    try:
        env.reset()
        sampled_pos = fr3_mujoco_record._reset_workspace_object_for_episode(env, rng)

        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, "workspace_object_body")
        joint_id = int(env.model.body_jntadr[body_id])
        qpos_adr = int(env.model.jnt_qposadr[joint_id])
        initial_pos = np.asarray(env.model.body_pos[body_id], dtype=np.float64)

        assert np.linalg.norm(sampled_pos[:2] - initial_pos[:2]) <= fr3_mujoco_record._WORKSPACE_OBJECT_RANDOM_RADIUS_M + 1e-9
        assert sampled_pos[2] == pytest.approx(initial_pos[2])
        np.testing.assert_allclose(env.data.qpos[qpos_adr : qpos_adr + 3], sampled_pos, atol=1e-9)
        np.testing.assert_allclose(
            env.data.qpos[qpos_adr + 3 : qpos_adr + 7],
            np.asarray(env.model.body_quat[body_id], dtype=np.float64),
            atol=1e-9,
        )
    finally:
        env.close()


def test_wait_for_teleop_idle_uses_three_consecutive_samples(monkeypatch):
    calls: list[int] = []

    class FakeTeleop:
        def wait_until_idle(self, *, consecutive_samples: int):
            calls.append(consecutive_samples)
            return True

    messages: list[tuple[str, bool]] = []
    monkeypatch.setattr(fr3_mujoco_record, "log_say", lambda text, play_sounds: messages.append((text, play_sounds)))

    fr3_mujoco_record._wait_for_teleop_idle(FakeTeleop(), play_sounds=False)

    assert calls == [3]
    assert messages == [("Release teleop input", False)]


def test_wait_for_teleop_idle_raises_when_input_does_not_settle(monkeypatch):
    class FakeTeleop:
        def wait_until_idle(self, *, consecutive_samples: int):
            assert consecutive_samples == 3
            return False

    monkeypatch.setattr(fr3_mujoco_record, "log_say", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="did not return to idle"):
        fr3_mujoco_record._wait_for_teleop_idle(FakeTeleop(), play_sounds=False)


def test_build_teleop_features_returns_expected_keys():
    features = fr3_mujoco_record._build_teleop_features()

    assert features["enabled"] is bool
    assert features["target_x"] is float
    assert features["target_y"] is float
    assert features["target_z"] is float
    assert features["target_wx"] is float
    assert features["target_wy"] is float
    assert features["target_wz"] is float
    assert features["gripper"] is float
    assert len(features) == 8


def test_run_episode_control_loop_publishes_pre_step_observation_and_action():
    initial_info = {
        "ee_pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(7, dtype=np.float64),
        "gripper_command": 0.25,
    }
    next_info = {
        "ee_pose": np.eye(4, dtype=np.float64) * 2.0,
        "joint_positions": np.ones(7, dtype=np.float64),
        "gripper_command": 0.75,
    }
    action = {
        "enabled": True,
        "target_x": 0.1,
        "target_y": 0.2,
        "target_z": 0.3,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.75,
    }

    class FakeTeleop:
        def get_action(self):
            return dict(action)

    class FakeEnv:
        def __init__(self):
            self.calls: list[tuple[dict, float, bool, bool]] = []

        def step_teleop_action(
            self,
            action_arg,
            *,
            control_period_s: float,
            include_camera_obs_in_observation: bool,
            include_camera_obs_in_info: bool,
        ):
            self.calls.append(
                (
                    dict(action_arg),
                    control_period_s,
                    include_camera_obs_in_observation,
                    include_camera_obs_in_info,
                )
            )
            return None, None, True, False, dict(next_info)

    fake_env = FakeEnv()
    shared_state = fr3_mujoco_record._LatestEpisodeControlState(
        sample_info=initial_info,
        viewer_info=initial_info,
        action=fr3_mujoco_record._zero_teleop_action(0.25),
        sample_gripper=0.25,
    )
    stop_event = threading.Event()

    fr3_mujoco_record._run_episode_control_loop(
        env=fake_env,
        teleop=FakeTeleop(),
        fps=200,
        initial_info=initial_info,
        initial_gripper=0.25,
        shared_state=shared_state,
        stop_event=stop_event,
    )

    sample_info, viewer_info, published_action, sample_gripper, loop_steps, terminated, truncated, error = (
        shared_state.snapshot()
    )

    assert fake_env.calls == [(action, 1.0 / 200.0, False, False)]
    assert sample_info == initial_info
    assert viewer_info == next_info
    assert published_action == action
    assert sample_gripper == pytest.approx(0.25)
    assert loop_steps == 1
    assert terminated is True
    assert truncated is False
    assert error is None
    assert stop_event.is_set()
