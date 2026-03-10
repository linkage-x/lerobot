#!/usr/bin/env python

from __future__ import annotations

import numpy as np

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
