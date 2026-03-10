# FR3 MuJoCo Local Environment

This directory hosts the local, EnvHub-compatible wrapper for the FR3 MuJoCo
simulation entrypoint.

Current intent:

- prototype the FR3 MuJoCo environment locally inside the `lerobot` repository
- keep the public entrypoint shape compatible with future EnvHub extraction
- avoid mixing hardware discovery and real-robot orchestration into the core
  environment package

The pure environment implementation lives in:

- `src/lerobot/envs/fr3_mujoco.py`
- `src/lerobot/envs/fr3_mujoco_teleop.py`

Interactive local teleoperation entrypoint:

- `scripts/fr3_mujoco_teleop.py`

Recommended Docker service for local viewer + SpaceMouse teleop:

- `lerobot-fr3-sim-teleop`

On X11 desktops, the interactive viewer path currently expects:

- `xhost +si:localuser:root`

This directory stays intentionally small so it can later be extracted into a
standalone Hub environment repository with minimal reshaping.
