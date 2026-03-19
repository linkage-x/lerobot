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
- `ruckig` for joint-space OTG smoothing after IK
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
5. install `panda_py`, `pika_sdk`, and `pyspacemouse`
6. install `ruckig`
7. run import smoke checks during image build

This ordering is currently the most defensible path inside the existing
repository setup.

## Dockerfile Changes Landed

Both Dockerfiles now include:

- extra system packages needed for FR3 native builds
- extra URDF DOM development packages so `pinocchio`'s CMake config can resolve
  `urdfdom_headers`
- `libfranka` source build
- `panda_py` installation from the specified git repository
- `pika_sdk` installation from `git@github.com:linkage-x/pika_sdk.git` for `pika.gripper`
- `ruckig` installation for FR3 joint OTG
- `pyspacemouse` installation
- `easyhid` installation required by `pyspacemouse`
- dynamic linker registration for the virtualenv `cmeel.prefix/lib` directory
- build-time import checks for:
  - `placo`
  - `panda_py`
  - `pika.gripper`
  - `ruckig`
  - `pyspacemouse`

Updated files:

- `docker/Dockerfile.user`
- `docker/Dockerfile.internal`

## Mirror Selection

On March 18, 2026, the current host measured the following mirror latency with
`curl -w '%{time_connect} %{time_starttransfer} %{time_total}'` before changing
the Dockerfiles:

- Debian apt:
  - USTC: `0.000990 0.050702 0.082206`
  - Aliyun: `0.000948 0.164176 0.230495`
  - TUNA: `0.000888 0.261199 0.510412`
- Debian security:
  - USTC: `0.000697 0.055855 0.074384`
  - Aliyun: `0.000818 0.346434 0.370837`
  - TUNA: `0.000922 0.260956 0.390333`
- Ubuntu apt:
  - Aliyun: `0.000772 0.360750 0.417862`
  - USTC: `0.000780 0.455346 0.592105`
  - TUNA: `0.000714 0.273451 0.693071`
- PyPI simple index:
  - Aliyun: `0.000925 0.105898 0.147795`
  - USTC: `0.001545 0.258952 0.282000`
  - TUNA: `0.000819 0.516508 0.677986`
- deadsnakes PPA:
  - USTC Launchpad proxy: `0.000870 2.560676 2.687783`
  - Launchpad direct: `0.000812 2.711509 2.739291`

The Dockerfiles now default to:

- `docker/Dockerfile.user`
  - Debian apt: `https://mirrors.ustc.edu.cn`
  - Debian security: `https://mirrors.ustc.edu.cn`
- `docker/Dockerfile.internal`
  - Ubuntu apt: `https://mirrors.aliyun.com`
  - Ubuntu security: `https://mirrors.aliyun.com`
  - deadsnakes PPA proxy: `https://launchpad.proxy.ustclug.org/deadsnakes/ppa/ubuntu`
- both Dockerfiles
  - PyPI / `uv pip`: `https://mirrors.aliyun.com/pypi/simple/`

These defaults remain overrideable through Docker build args so the target
machine can switch mirrors without editing the Dockerfiles.

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

On March 18, 2026, the DAS gripper serial mapping on the current host was
confirmed as:

- DAS gripper serial node: `/dev/ttyUSB0`

The repository's FR3 hardware-facing defaults have been aligned to that serial
node so the common replay / smoke / teleop entrypoints no longer assume the
older placeholder path.

For the DAS-controller path, the Docker images now also clone
`genrobot-ai/gen_con_sdk_python_release` into
`/opt/dependencies/gen_con_sdk_python_release`, install its Python
requirements, and expose that root via `GEN_CON_SDK_HOME`.

Important: the SDK's `config/99-usb-serial.rules` udev template still belongs
on the host machine, not inside the container. The compose services mount the
host `/dev`, so stable DAS serial naming must be configured on the host first.

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
  - `import ruckig`
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

## SSH Dataset Visualization

The default FR3 recording wrapper now writes an `ee2ee`-style dataset:

- `observation.state`: absolute EE pose + absolute gripper position
- `action`: absolute EE target pose + absolute gripper target
- camera streams remain under `observation.images.*`

This means the default FR3 training path follows the existing LeRobot pattern:
pick the final learning representation at record time instead of recording a
multi-semantic canonical schema first.

For dataset inspection over SSH, the current preferred path is not to rely on
the remote machine's X display.

The working path is:

1. run `lerobot_dataset_viz` on the target machine in distant mode
2. connect from the local machine with a local `rerun` viewer

Remote command:

```bash
PYTHONPATH=src .venv-codex/bin/python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id hph/fr3_pick_place_ee2ee_v1 \
  --root /home/hph/Code/lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_153947 \
  --episode-index 0 \
  --mode distant \
  --grpc-port 9876
```

Local command:

```bash
rerun --connect rerun+http://192.168.1.200:9876/proxy
```

While the distant-mode script is running on the remote machine:

- press `n` in that terminal to switch to the next episode
- press `q` to quit
- once the last episode is reached, further `n` presses are ignored

If you want to drive episode switching from the local machine instead of the
remote SSH terminal, start `lerobot_dataset_viz` with a control port and then
run the companion client locally:

Remote command:

```bash
PYTHONPATH=src .venv-codex/bin/python src/lerobot/scripts/lerobot_dataset_viz.py \
  --repo-id hph/fr3_pick_place_ee2ee_v1 \
  --root outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_125942 \
  --episode-index 0 \
  --mode distant \
  --web-port 9090 \
  --grpc-port 19876 \
  --control-port 19999
```

Local command:

```bash
uv run src/lerobot/scripts/dataset_viz_client.py     --host 192.168.1.200     --control-port 19999     --rerun-connect-url rerun+http://192.168.1.200:19876/proxy
```

In this mode:

- press `n` in the local client terminal to switch to the next episode
- press `q` in the local client terminal to stop the remote supervisor cleanly
- if the dataset is already at the last episode, further `n` commands are ignored

Implementation note:
episode switching is handled by terminating the current episode worker process
and starting a new one for the next episode. This is the current defensive path
to keep video decoder and dataset memory usage from accumulating across
switches.

This is currently more reliable than exporting `DISPLAY` over SSH and trying to
spawn a remote viewer process. In the observed setup, the remote viewer failed
because the SSH shell did not have valid X authorization, while distant-mode
streaming worked.

## Current Best Practice

For FR3 hardware recording and inspection, the current recommended path is:

1. run recordings through `tools/fr3/fr3_record.py`
2. let the wrapper choose or forward the dataset root
3. let the wrapper normalize output ownership back to the host user after a
   successful Docker recording
4. inspect the dataset through `lerobot_dataset_viz --mode distant` and a local
   `rerun` viewer
5. when stepping through multiple episodes remotely, prefer `--control-port`
   plus local `dataset_viz_client.py`; if that is not convenient, fall back to
   remote terminal `n` / `q`

This is the current best practice because it preserves the existing Docker
hardware setup while also avoiding the two most common operator failures seen in
this round:

- root-owned dataset files that break downstream video decoding
- fragile remote GUI/X11 viewer workflows over SSH

### Dataset Ownership Note

If the dataset was recorded through `sudo` or a root-owned Docker invocation,
the recorded mp4 files may end up unreadable by the normal user, for example as
`root:root` with mode `600`.

The FR3 recording wrapper now treats host-readable ownership as part of the
normal success path and runs a container-side `chown -R <host_uid>:<host_gid>
<dataset_root>` after a successful recording.

If a dataset was still produced outside that wrapper, fix ownership before
running visualization:

```bash
sudo chown -R hph:hph /home/hph/Code/lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260313_153947
```
