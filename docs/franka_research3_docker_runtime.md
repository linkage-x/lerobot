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

The repository's `.[all]` extra does not declare `pin` directly, but the real
Docker build showed that `pin` and the related `cmeel` package tree are pulled
in transitively. That matters because the wheel install provides a usable
`pinocchioConfig.cmake` under the virtualenv's `cmeel.prefix` directory.

An attempted switch to upstream source installation of `pinocchio` failed during
configuration because it also required additional CMake packages such as
`urdfdom_headers`. For the current repository layout, that path adds complexity
without solving a real gap.

The current Dockerfiles therefore keep `uv pip install ".[all]"` first and then
build `libfranka` against the `cmeel.prefix` tree created by the Python
dependencies.

The current ordering is:

1. create the Python environment
2. install `lerobot` Python dependencies with `uv pip install ".[all]"`
3. verify that `pinocchioConfig.cmake` exists in the virtualenv `cmeel.prefix`
4. build and install `libfranka`
5. install `panda_py`, `agx-pypika`, and `pyspacemouse`
6. run import smoke checks during image build

This ordering is currently the most defensible path inside the existing
repository setup.

## Dockerfile Changes Landed

Both Dockerfiles now include:

- extra system packages needed for FR3 native builds
- extra URDF DOM development packages so `pinocchio`'s CMake config can resolve
  `urdfdom_headers`
- `libfranka` source build
- `panda_py` installation from the specified git repository
- `agx-pypika` installation for `pika.gripper`
- `pyspacemouse` installation
- `easyhid` installation required by `pyspacemouse`
- dynamic linker registration for the virtualenv `cmeel.prefix/lib` directory
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

The `lerobot-user` image now builds successfully with the FR3 dependency chain.

What is known:

- the previous image lacked `panda_py`, `pika.gripper`, and `pyspacemouse`
- a real Docker build reproduced a native build failure in `libfranka`
- that failure identified a required CMake dependency on `pinocchio`
- a later Docker build showed that `pin` is pulled transitively by `.[all]`
- an attempted upstream `pinocchio` source build failed on missing
  `urdfdom_headers` CMake packages
- the Dockerfiles were updated to consume the virtualenv `cmeel.prefix` tree
  directly for `libfranka` and `panda_py`
- `libfranka` CMake configuration must also be pointed at the venv Python
  executable so `eigenpy` can detect the installed `numpy`
- `libfranka` must be configured with the system CMake binary rather than the
  newer venv `cmake` package to avoid `common/` compatibility failures
- the fetched `fmt` dependency from the `libfranka` build must also be installed
  to `/usr/local` so downstream `panda_py` can satisfy `FrankaConfig.cmake`
- the runtime linker also needs the virtualenv `cmeel.prefix/lib` directory so
  `panda_py` can resolve `libpinocchio_parsers.so`
- `pyspacemouse` additionally requires `easyhid` at import time
- `docker compose -f docker/docker-compose.yml build lerobot-user` now finishes
  successfully
- the image build-time smoke checks now pass for:
  - `import placo`
  - `import panda_py`
  - `from pika.gripper import Gripper`
  - `import pyspacemouse`

What is not yet confirmed:

- that `docker compose -f docker/docker-compose.yml build lerobot-internal`
  also succeeds with the same dependency chain
- that `pyspacemouse.list_devices()` sees the physical device inside the
  running container
- that FR3 arm and Pika serial hardware are reachable from the running
  container on the target machine

## Practical Conclusion

At this stage:

- there is no evidence that FR3 minimal integration requires a full conda
  environment
- there is no current need to source-build `pinocchio` from upstream for the
  FR3 minimal container path
- there is evidence that FR3 runtime requires careful native dependency
  ordering
- the current `uv`-based approach is sufficient for `lerobot-user`
- the next focus should move from build-time dependency resolution to runtime
  device validation

## Next Validation Target

The next concrete milestone is:

- build and validate `lerobot-internal`
- then run a container runtime smoke check for:
  - `pyspacemouse.list_devices()`
  - device visibility under `/dev/bus/usb`, `/dev/input`, and serial nodes
  - FR3 and gripper connectivity from inside the compose environment
