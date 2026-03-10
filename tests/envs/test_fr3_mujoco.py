#!/usr/bin/env python

from pathlib import Path

from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig


def test_default_fr3_mujoco_urdf_path_exists():
    cfg = FR3MujocoEnvConfig()
    assert Path(cfg.urdf_path).is_file()


def test_local_envhub_wrapper_exists():
    wrapper_path = Path("sim/fr3_mujoco_env/env.py")
    assert wrapper_path.is_file()
