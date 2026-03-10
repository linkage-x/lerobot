# Franka Research 3 Docker Runtime Notes

## Scope

This note captures the current Docker runtime conclusions for the
`franka_research3` minimal integration in this repository.

It focuses on:

- `docker/Dockerfile.user`
- `docker/Dockerfile.internal`
- `docker/docker-compose.yml`

It does not redefine the robot and teleoperator contracts already documented in
`docs/franka_research3_minimal_integration.md`.

## Current Decision

For the FR3 + SpaceMouse minimal bring-up, the repository does not need to
introduce a full conda-based Python environment.

The current preferred direction is:

- keep `lerobot` itself installed with `uv` in the container
- only install the FR3 runtime dependencies that are actually needed now
- avoid broad environment changes motivated by unrelated stacks

## Dependency Assessment

### What FR3 Actually Uses

The current `franka_research3` implementation depends on:

- `placo` via `src/lerobot/model/kinematics.py`
- `panda_py` for FR3 arm control
- `pika.gripper` for the Pika gripper
- `pyspacemouse` for the SpaceMouse teleoperator
- `libfranka` as the native backend required by `panda_py`

### What FR3 Does Not Currently Depend On Directly

The current FR3 minimal integration does not directly depend on:

- `pinocchio` Python APIs in application code
- `casadi`
- the older HIROL / open-sop inference stack

Those dependencies matter for other stacks, but they are not the main runtime
path for the current FR3 teleoperation integration.

## Important Build Finding

The first real Docker build exposed an important native dependency detail:

- `libfranka` version `0.15.0` requires `pinocchioConfig.cmake` at CMake
  configure time

This means `libfranka` cannot simply be built first in a blank image layer.

The current Dockerfiles were therefore reordered to:

1. create the Python environment
2. install `lerobot` Python dependencies with `uv pip install ".[all]"`
3. reuse the `cmeel.prefix` tree created by the installed robotics packages as a
   `CMAKE_PREFIX_PATH`
4. build and install `libfranka`
5. install `panda_py`, `agx-pypika`, and `pyspacemouse`
6. run import smoke checks during image build

This ordering is currently the most defensible path inside the existing
repository setup.

## Dockerfile Changes Landed

Both Dockerfiles now include:

- extra system packages needed for FR3 native builds
- `libfranka` source build
- `panda_py` installation from the specified git repository
- `agx-pypika` installation for `pika.gripper`
- `pyspacemouse` installation
- build-time import checks for:
  - `placo`
  - `panda_py`
  - `pika.gripper`
  - `pyspacemouse`

Updated files:

- `docker/Dockerfile.user`
- `docker/Dockerfile.internal`

## Compose Status

`docker/docker-compose.yml` now provides:

- `lerobot-user` as the default container profile for local FR3 bring-up
- `lerobot-internal` as the GPU-oriented variant
- host networking
- privileged mode
- X11 mount
- USB, input, and dbus mounts
- Hugging Face / Torch / Triton cache mounts

This is intended to match the hardware-facing requirements of FR3 + SpaceMouse
bring-up while still using the repository's own Dockerfile structure.

## Current Status

The Docker dependency chain has been implemented, but not fully validated end to
end yet.

What is known:

- the previous image lacked `panda_py`, `pika.gripper`, and `pyspacemouse`
- a real Docker build reproduced a native build failure in `libfranka`
- that failure identified a required CMake dependency on `pinocchio`
- the Dockerfiles were updated to address that failure by reordering the build

What is not yet confirmed:

- that `docker compose build lerobot-user` finishes successfully after the new
  ordering
- that `panda_py` imports and links successfully against the built `libfranka`
- that `from pika.gripper import Gripper` resolves correctly in the built image
- that `pyspacemouse.list_devices()` sees the physical device inside the
  container

## Practical Conclusion

At this stage:

- there is no evidence that FR3 minimal integration requires a full conda
  environment
- there is evidence that FR3 runtime requires careful native dependency
  ordering
- the correct immediate focus is completing Docker validation of the current
  `uv`-based approach before introducing another environment manager

## Next Validation Target

The next concrete milestone is:

- get `docker compose -f docker/docker-compose.yml build lerobot-user` to
  complete successfully
- then run a container smoke check for:
  - `import placo`
  - `import panda_py`
  - `from pika.gripper import Gripper`
  - `import pyspacemouse`
  - device visibility under `/dev/bus/usb`, `/dev/input`, and serial nodes
