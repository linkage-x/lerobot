# FR3 MuJoCo Simulation TODO

## Scope

This document captures the current execution plan for the FR3 teleoperation
simulation track discussed in the latest architecture review.

The immediate goal is not a full digital twin. The goal is a local MuJoCo
simulation gate that can catch frame, IK, OTG, and teleop-target issues before
the next real-hardware smoke test.

## Current Decisions

1. Use MuJoCo as the next simulation backend for FR3 teleoperation bring-up.
2. Keep the simulation runtime as a dedicated Docker service/profile, but run it
   in the GPU-capable Docker environment and reuse the existing LeRobot
   dependency stack.
3. Reuse the same FR3 assets and kinematic contract used by hardware bring-up:
   `fr3_pika_gripper_ati.urdf`, `pika_gripper_ee`, FR3 joint names, and the
   current IK/OTG configuration path.
4. Keep EnvHub as a later packaging target for the pure simulation layer, not
   as the first implementation step for the full FR3 teleop workflow.

## Priority Order

### P0: Foundation

- [x] Record the architecture and implementation plan.
- [x] Add a dedicated `lerobot-fr3-sim` Docker compose service/profile.
- [x] Make MuJoCo an explicit FR3 sim dependency instead of relying on
      transitive installs.
- [x] Create a local, EnvHub-compatible FR3 MuJoCo environment entrypoint.

### P1: Minimum Local Sim Gate

- [x] Load the local FR3 MuJoCo model with the imported FR3/Pika assets.
- [x] Expose a Gym-compatible environment with deterministic reset/step.
- [x] Return joint-state observations and same-frame EE pose observations.
- [x] Add a local smoke entrypoint that validates model load and one-step
      rollout.
- [x] Validate the GPU runtime path by creating a MuJoCo EGL context inside the
      dedicated sim container.

### P2: Teleop Path Validation

- [x] Add target/TCP visualization markers consistent with the HIROL MuJoCo
      workflow.
- [x] Bridge the current teleop target semantics into the local sim path.
- [x] Validate `enabled` transitions, workspace clipping, and first-frame hold
      behavior without touching hardware.
- [ ] Add regression checks for FK/IK consistency and target-frame alignment.

### P3: Future Extraction

- [ ] Split the pure environment layer into a standalone EnvHub-ready repo
      layout.
- [ ] Keep local hardware discovery, SpaceMouse orchestration, and real-robot
      smoke scripts outside the EnvHub core package.
- [ ] Publish only after the local MuJoCo gate is stable and reproducible.

## Container Recommendation

Use a dedicated simulation container/service instead of folding simulation into
the default FR3 hardware runtime service.

Why:

- the sim gate should remain runnable without FR3, serial, USB, or privileged
  hardware access
- MuJoCo for this FR3 sim track should assume a GPU-capable runtime and use the
  GPU-oriented container profile by default
- the code path should still reuse the same Python environment and FR3 assets
- the resulting service can later serve local CI and future EnvHub packaging

Implementation rule:

- separate service/profile
- shared LeRobot Docker base, but GPU-oriented for MuJoCo execution
- shared repository volume
- no hard dependency on real hardware mounts

## Immediate Next Steps

1. Add target/TCP visualization markers consistent with the HIROL MuJoCo
   workflow.
2. Bridge the current teleop target semantics into the local sim path.
3. Add regression checks for FK/IK consistency, `enabled` transitions, and
   first-frame hold behavior.

## Latest Validation

Validated on March 10, 2026 in the dedicated `lerobot-fr3-sim` GPU container:

- `nvidia-smi -L` detected `NVIDIA GeForce RTX 4090 D`
- `MUJOCO_GL=egl` successfully created a `mujoco.GLContext`
- `scripts/fr3_mujoco_env_smoke.py` loaded the FR3/Pika model and completed
  reset plus three zero-action steps
- the same smoke entrypoint completed one relative-target teleop probe and
  returned aligned target/TCP marker poses

Current non-blocking warnings:

- the imported URDF reports expected self-collision pairs in the neutral pose
- MuJoCo warns about duplicate `fr3_link0_visual` geom names during URDF import
