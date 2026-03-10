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

This directory stays intentionally small so it can later be extracted into a
standalone Hub environment repository with minimal reshaping.
