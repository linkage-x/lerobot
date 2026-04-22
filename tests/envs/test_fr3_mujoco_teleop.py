#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pytest

from lerobot.envs.fr3_mujoco import FR3MujocoEnv, FR3MujocoEnvConfig
from lerobot.envs.fr3_mujoco_teleop import MarkerStyle, marker_geoms_from_info, run_sim_teleop_loop


class FakeTeleop:
    def __init__(self, actions):
        self._actions = list(actions)
        self._index = 0

    def get_action(self):
        if self._index >= len(self._actions):
            return {"enabled": False}
        action = self._actions[self._index]
        self._index += 1
        return action


class SyncingFakeTeleop(FakeTeleop):
    def __init__(self, actions):
        super().__init__(actions)
        self.synced_gripper_commands: list[float] = []

    def sync_gripper_baseline(self, normalized_command: float) -> float:
        value = float(normalized_command)
        self.synced_gripper_commands.append(value)
        return value


class FakeViewer:
    def __init__(self):
        self.user_scn = type("Scene", (), {"maxgeom": 16, "ngeom": 0, "geoms": [type("Geom", (), {"rgba": np.zeros(4, dtype=np.float32)})() for _ in range(16)]})()
        self.sync_calls = 0
        self.running = True

    def is_running(self) -> bool:
        return self.running

    class _Lock:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    def lock(self):
        return self._Lock()

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.running = False


class FakeMujoco:
    class mjtGeom:
        mjGEOM_SPHERE = 0
        mjGEOM_CAPSULE = 1

    class Renderer:
        def __init__(self, model, height, width):
            del model, height, width

        def update_scene(self, data, camera=None):
            del data, camera

        def render(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            return None

    class _FakeMjData(dict):
        pass

    def MjData(self, model):
        del model
        return self._FakeMjData()

    def mjv_initGeom(self, geom, type, size, pos, mat, rgba):
        del type, size, pos, mat
        geom.rgba[:] = rgba

    def mjv_connector(self, geom, geom_type, radius, start, end):
        del geom, geom_type, radius, start, end


def test_marker_geoms_include_target_tcp_spheres_and_axes():
    env = FR3MujocoEnv()
    try:
        _, info = env.reset()
        geoms = marker_geoms_from_info(info, MarkerStyle())
        assert len(geoms) == 8
        assert [geom["kind"] for geom in geoms[:2]] == ["sphere", "sphere"]
        assert [geom["name"] for geom in geoms[:2]] == ["target", "TCP"]
        assert all(geom["kind"] == "connector" for geom in geoms[2:])
    finally:
        env.close()


def test_marker_axes_use_configured_axis_length():
    env = FR3MujocoEnv()
    try:
        _, info = env.reset()
        style = MarkerStyle(axis_length=0.08)
        geoms = marker_geoms_from_info(info, style)
        connector = geoms[2]
        measured = np.linalg.norm(np.asarray(connector["end"]) - np.asarray(connector["start"]))
        assert abs(measured - style.axis_length) < 1e-9
    finally:
        env.close()


def test_run_sim_teleop_loop_steps_env_without_viewer():
    env = FR3MujocoEnv(
        FR3MujocoEnvConfig(
            max_episode_steps=20,
            otg_max_velocity=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
            otg_max_acceleration=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
            otg_max_jerk=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )
    teleop = FakeTeleop(
        [
            {"enabled": True, "target_x": 0.002, "target_y": 0.0, "target_z": 0.0},
            {"enabled": False},
            {"enabled": False},
        ]
    )
    try:
        info = run_sim_teleop_loop(env=env, teleop=teleop, fps=200, viewer=None, max_steps=3)
        assert info["loop_steps"] == 3
        assert info["target_marker_name"] == "target"
        assert info["tcp_marker_name"] == "TCP"
        assert info["otg_enabled"] is True
        assert np.linalg.norm(info["target_pose"][:3, 3] - info["tcp_pose"][:3, 3]) > 1e-5
    finally:
        env.close()


def test_run_sim_teleop_loop_syncs_gripper_baseline_from_env_reset():
    env = FR3MujocoEnv(FR3MujocoEnvConfig(max_episode_steps=5))
    teleop = SyncingFakeTeleop([{"enabled": False}])
    try:
        info = run_sim_teleop_loop(env=env, teleop=teleop, fps=200, viewer=None, max_steps=1)
        assert info["loop_steps"] == 1
        assert teleop.synced_gripper_commands == [pytest.approx(1.0)]
    finally:
        env.close()


def test_run_sim_teleop_loop_copies_visual_state_into_separate_viewer_data():
    class FakeEnv:
        def __init__(self):
            self._mujoco = FakeMujoco()
            self.copy_calls = 0
            self.info = {
                "target_pose": np.eye(4, dtype=np.float64),
                "tcp_pose": np.eye(4, dtype=np.float64),
                "target_marker_name": "target",
                "tcp_marker_name": "TCP",
                "target_site_name": "target_site",
                "tcp_site_name": "tcp_site",
                "camera_names": (),
                "gripper_command": 1.0,
            }

        def reset(self, **kwargs):
            del kwargs
            return {}, dict(self.info)

        def copy_visual_state(self, target_data):
            target_data["copied"] = True
            self.copy_calls += 1

        def step_teleop_action(self, action, control_period_s=None):
            del action, control_period_s
            return {}, 0.0, False, True, dict(self.info)

    env = FakeEnv()
    teleop = SyncingFakeTeleop([{"enabled": False}])
    viewer = FakeViewer()
    viewer_data: dict[str, object] = {}

    info = run_sim_teleop_loop(
        env=env,
        teleop=teleop,
        fps=200,
        viewer=viewer,
        viewer_data=viewer_data,
        max_steps=1,
    )

    assert info["loop_steps"] == 1
    assert viewer_data["copied"] is True
    assert env.copy_calls >= 2
    assert viewer.sync_calls >= 2


def test_run_sim_teleop_loop_skips_step_camera_obs_when_camera_stream_enabled(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self._mujoco = FakeMujoco()
            self.step_kwargs = None
            self.reset_kwargs = None
            self.copy_calls = 0
            self.model = object()
            self.cfg = type(
                "Cfg",
                (),
                {
                    "camera_names": ("third_person", "side", "wrist"),
                    "camera_name_mapping": {"third_person": "third_person", "side": "side", "wrist": "wrist"},
                },
            )()
            self.info = {
                "target_pose": np.eye(4, dtype=np.float64),
                "tcp_pose": np.eye(4, dtype=np.float64),
                "target_marker_name": "target",
                "tcp_marker_name": "TCP",
                "target_site_name": "target_site",
                "tcp_site_name": "tcp_site",
                "camera_names": ("third_person", "side", "wrist"),
                "gripper_command": 1.0,
            }

        def reset(self, **kwargs):
            self.reset_kwargs = kwargs
            return {}, dict(self.info)

        def step_teleop_action(self, action, control_period_s=None, **kwargs):
            del action, control_period_s
            self.step_kwargs = kwargs
            return {}, 0.0, False, True, dict(self.info)

        def copy_visual_state(self, target_data):
            target_data["copied"] = True
            self.copy_calls += 1

    class FakeCameraServer:
        def shutdown(self):
            return None

    class FakeLatestFrame:
        def __init__(self):
            self.value = None

        def set(self, jpeg_bytes):
            self.value = jpeg_bytes

        def get(self):
            return self.value

    def fake_start_camera_stream_outputs(**kwargs):
        del kwargs
        return FakeCameraServer(), FakeLatestFrame(), None

    monkeypatch.setattr(
        "lerobot.envs.fr3_mujoco_teleop._start_camera_stream_outputs",
        fake_start_camera_stream_outputs,
    )

    env = FakeEnv()
    teleop = SyncingFakeTeleop([{"enabled": False}])
    info = run_sim_teleop_loop(
        env=env,
        teleop=teleop,
        fps=200,
        viewer=None,
        max_steps=1,
        render_cameras=True,
        camera_width=8,
        camera_height=8,
        camera_fps=30,
    )

    assert info["loop_steps"] == 1
    assert env.reset_kwargs == {
        "include_camera_obs_in_observation": False,
        "include_camera_obs_in_info": False,
    }
    assert env.step_kwargs == {
        "include_camera_obs_in_observation": False,
        "include_camera_obs_in_info": False,
    }
    assert env.copy_calls >= 1
