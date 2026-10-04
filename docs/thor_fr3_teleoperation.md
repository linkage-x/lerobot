# Thor FR3 teleoperation through the BOX UI

This workflow extends the BOX platform at commit
`8ffb5bf6e1c391ea771207f628601cb753127f73`. Run `bash run/deploy.sh` on the
development host and operate the existing **Live Record** page. The host owns
the USB SpaceMouse and FR3 FCI connection. Thor owns the selected Sengyun
cameras, BOX tactile/force/gripper sensors, gripper actuation and recording. The default deployment contacts
only `nvidia@192.168.111.122`; no second workstation SSH login is required.

## Deploy and operate

From the development host repository root:

```bash
bash run/deploy.sh
```

Open the frontend URL printed by the script, normally `http://localhost:5173/`.
The gateway runs on Thor at `http://192.168.111.122:8765`. Deployment syncs code
and existing calibration inputs, restarts the gateway/owned recorder, and starts
the host frontend and idle FR3 host service. It also establishes a host-initiated
SSH reverse tunnel; Thor never SSHes to the host. It neither installs FR3
dependencies nor opens FCI. If host setup fails, camera/BOX UI remains usable;
F reports the unavailable host link.

Use the existing task selection and Live Record page:

| Control | Result |
| --- | --- |
| **C — Connect** | Connect the Sengyun cameras checked on Live Record and, if checked, the BOX board. FR3 remains inactive. |
| **Select all except FR3 and laser** | Check every detected camera and the BOX board, and turn off the laser tracker choice. |
| **F — Start FR3** | After C completes, check the native runtime, connect FR3, execute `move_to_start`, then activate SpaceMouse control. |
| **E — Start Episode** | Begin a 30-second recording while FR3 teleoperation is running. |
| **S — Save** | End and save early, then return FR3 to its start pose. |
| **D — Discard** | End and discard, then return FR3 to its start pose. |
| **Stop FR3** | Stop robot control; discard an active episode. Cameras/BOX stay connected. |
| **Esc — Exit** | End the connected recording session and stop FR3 control. |

Existing shortcuts and controls retain their BOX behavior. Shortcuts do not fire
while typing in an input, select, or text editor. A disabled button also disables
its shortcut. F is unavailable while connecting, moving to start, already
running, recording, or resolving an episode.

When an episode reaches 30 seconds, the recorder stops and saves it automatically.
Save, discard, and automatic completion request a fresh return to start **after**
recording stops. The return is excluded from the episode data. Release the
SpaceMouse; the UI shows the move in progress and E is unavailable until the
robot is back at start and the puck is neutral. A robot or link fault during
the return stops teleoperation and requires the usual Desk recovery and F retry.
Calibration captures may specify their own episode length.

Before C, choose individual camera IDs on **Live Record**. At least one camera
must be selected because the Thor episode recorder uses camera frames as its
timeline. The BOX board is one SDK connection for gripper, force, touch, IMU and
trigger; its streams cannot be connected independently from this UI. Unchecking
BOX allows camera-only capture and disables F because FR3 gripper control needs
the BOX. The selection is fixed for a connected session; disconnect to change it.
FR3 and SpaceMouse are activated only by F, and the laser tracker has its own
separate checkbox. The top-bar and Dashboard connection links open Live Record
so the choice is visible before C.

To compare camera load during commissioning, select two known connected IDs
(for example `cam_06` and `cam_07` on the current Thor wiring), leave BOX checked
and laser off, then press C and F with the SpaceMouse released. Compare the FR3
FCI success rate with the full-camera session. FCI fault checks retain the same
threshold for both runs; a two-camera success only shows that camera load
contributes, and a repeated 0.99 fault points to host scheduling or FCI network
timing that still needs investigation.

On 2026-10-04, a neutral test with only `cam_06`, `cam_07` and BOX connected
still stopped after about three seconds of active control when the FCI rate
reached 0.99 (configured limit 0.995). Reducing camera count alone therefore
did not resolve this Thor's current FCI timing fault.

F starts physical motion. Release the SpaceMouse before pressing F and keep the
robot's user stop accessible. Wait for **running** before using the puck or E.
SpaceMouse axes use the existing `[-raw_y, raw_x, raw_z]` translation mapping;
buttons adjust the BOX gripper's opening. The profile now separates translation
and rotation: any translation suppresses rotational input until all six axes
return to neutral. Release the puck fully before a deliberate rotation-only
gesture. This prevents incidental tilt and the release tail from rotating the
EE during an X/Y/Z translation. `separate_translation_rotation: false` restores
simultaneous 6-DOF control; `enable_rotation: false` disables rotation entirely.
Rotation deadzones are 0.12. The FR3 profile doubles the SpaceMouse gains to
0.00123 translation and 0.000648 rotation per input update, and raises the
translation command cap to 0.002 m per axis so larger inputs are not clipped.
The gripper button step is unchanged. `robot.ik_orientation_weight: 1.0` strengthens
the IK orientation target. These hold the commanded orientation; physical
tracking error still depends on the controller and robot load. The UI displays measured joints,
velocity, measured/external joint torque, TCP pose and estimated external wrench.

This profile requires recent HID reports for active puck motion. Cached motion
older than 200 ms sends zero arm motion and holds the gripper until a fresh HID
report arrives. If a held button produces only its initial press report,
gripper increments freeze after that window; release and press again to continue.
Neutral input may stop sending reports while remaining valid.

Esc and recorder stdin EOF stop SpaceMouse input immediately, including while
camera save/discard processing is busy. Video finalization and session teardown
may continue after robot input has stopped. The native runtime's ownership lock
refuses overlapping FR3 workers for the same robot under the same login; it does
not replace checking that another FCI application is disconnected.

On a robot error, the UI displays the reason, stops teleoperation, and discards
the active episode. Clear the reported condition using the robot's physical
controls/Desk, release the SpaceMouse, and press F again. Every F attempt returns
to start before enabling teleoperation. The program does not automatically
recover errors or resume motion.

The original camera/BOX-only profile remains available:

```bash
bash run/deploy.sh --box-only
bash run/deploy.sh --sync-only
bash run/deploy.sh --no-frontend
```

`--box-only` uses `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml`; the default uses
`tools/thor/gmsl2/thor_fr3_teleop.yaml`. The latter preserves the original camera,
BOX, trigger, laser tracker, dataset and calibration settings and adds FR3
configuration. Calibration pages, camera previews, task management, replay,
trajectory processing, dataset handling and sensor diagnostics remain in the
BOX interface.

Use **Stop FR3** before taking calibration captures. Cameras and BOX remain
connected for the existing calibration workflows without active robot control.

## Host controller and approximate synchronization

The default `fr3_teleop.execution_host: host` splits ownership as follows:

```text
Host: SpaceMouse -> 100 Hz input/IK targets -> native 1 kHz FCI -> FR3
                |                     |
                + gripper targets     + joint/TCP/torque telemetry
                           authenticated SSH link (50 Hz)
                                      |
Thor: cameras + BOX sensors <- recorder/coordinator -> BOX gripper SDK
```

The native FCI worker remains a separate **host** process with same-machine
IPC. Camera encoding, BOX polling and network serialization stay outside it.
Host/Thor controller config digests must match. The FCI quality target is
`0.995`, with the bounded warning policy described below. Workspace/joint/delta
limits, native watchdog and no-automatic-recovery patch remain active. Moving machines reduces shared CPU load; it does not prove that
a non-RT host meets every 1 ms FCI deadline.

F opens a fresh authenticated session, qualifies eight clock probes, then starts
the host SpaceMouse/native worker. Gripper targets use the existing bounded
incremental button behavior, max 15 commands/second and a 0.5 mm deadband.
Thor validates range, sequence and age, calls the BOX SDK, and returns its real
acknowledgement. The recorded gripper target is the last acknowledged command.
A lost or stale link disables further commands and stops the host worker; Thor
restores BOX collection mode. A retry always requires a fresh F session. No
commands are queued for replay across reconnects.

Each 50 Hz request/response carries four monotonic timestamps: Thor send,
host receive, host send, Thor receive. The minimum-RTT estimate from the last
five seconds maps host state time into Thor's clock domain. Every raw sidecar
sample retains `host_sample_monotonic_s`, `host_receiver_monotonic_s`,
`sample_monotonic_s` (Thor estimate), `receiver_monotonic_s` (actual Thor arrival),
`clock_host_minus_thor_s`, `clock_rtt_s`, `clock_uncertainty_s` and
`clock_sync_valid`. `spacemouse_action` includes both mapped and original input
timestamps. `gripper_ack` retains the command sequence, requested opening,
host send time, estimated Thor send time and actual Thor SDK acknowledgement
time (`thor_applied_s`); it marks SDK acceptance, not physical completion of the
gripper travel. This is software synchronization, not camera hardware triggering or
PTP. Uncertainty includes half RTT plus a 200 ppm allowance for estimate age;
network asymmetry remains a source of error.

The host/Thor data-link lease is 400 ms. An isolated round trip over 100 ms is
excluded from the clock fit when a recent good probe exists; a missing good
probe for five seconds or uncertainty over 10 ms ends the session. Telemetry
older than 200 ms is omitted and logged as delayed, never written with a fresh
timestamp. If no fresh host state arrives within the 400 ms lease, teleoperation
stops. The dataset's existing 25 ms nearest-camera-sample budget includes clock
uncertainty; samples outside that budget remain invalid. The host native FCI
input watchdog remains 200 ms, and its 1 kHz control loop stays local to the host.

## One-time host setup

Plug SpaceMouse into **corenetic**, keep Thor and host network connections, and
ensure the host routes directly to `192.168.11.102`. On this machine `eno2`
uses `192.168.11.44`. Its kernel is `6.8.0-138-generic` (PREEMPT_DYNAMIC, not RT);
the profile retains the explicitly selected `realtime_mode: ignore`.

Use a host-native x86_64 wheel built with the same inspected patch below,
against the installed FR3-compatible libfranka version. Do not install Thor's
aarch64 wheel on the host. The wheel built here is
`outputs/wheels/panda_python-0.8.1-cp312-cp312-linux_x86_64.whl`, against
libfranka 0.15.0. The existing setup helper also works on the host:

```bash
bash run/setup_thor_fr3.sh --install-python \
  --panda-wheel outputs/wheels/panda_python-0.8.1-cp312-cp312-linux_x86_64.whl
bash run/setup_thor_fr3.sh --check
THOR_PYTHON=.venv-fr3/bin/python bash run/setup_thor_spacemouse.sh --check
# Explicit scheduling permission setup; no robot connection or motion:
bash run/setup_host_fr3_permissions.sh --install
bash run/deploy.sh
```

The host `.venv-fr3` was provisioned with Python 3.12, pyspacemouse 2.1.0,
Placo and Ruckig, and the patched wheel passed preflight without opening FCI.
The installed system libfranka needs fmt ABI 9; its existing local
`anaconda3/lib/libfmt.so.9.1.0` was copied into `.venv-fr3/lib/libfmt.so.9`.
The launcher includes that directory and the environment's `cmeel.prefix/lib`
in the native library search path. Recreating the environment may require
restoring this ABI-matching library (never substitute a different soname).
The SpaceMouse Compact `256f:c635` has a host udev rule for plugdev access.

The profile sets `host_performance_governor: true`; deployment reapplies the
host CPU `performance` governor with noninteractive sudo and reports a warning
if it cannot. This increases host power use; set the option false to manage CPU
policy yourself. `--performance` on the scheduling helper applies it manually.
The pre-test governor snapshot was saved in
`/tmp/lerobot-host-fr3-governors-before.json` (all 20 policies were `powersave`).
The native worker limits OpenMP/BLAS pools to one thread for small 7-DOF IK.

The scheduling helper permits this account to request FIFO priority up to 99
in future logins. It writes an opt-in marker under `outputs/secrets`; for an
older desktop login, deployment uses `sudo -n prlimit` only on the newly spawned
host service so its native children inherit that ceiling. If sudo is unavailable,
log out/in before F. Python sensor/network threads themselves remain normal
scheduled threads.

`deploy.sh` starts the host service bound to `127.0.0.1:18766` and forwards
Thor's loopback port through the existing SSH credentials. A random token in
`outputs/secrets/fr3_host.token` is copied over SSH with mode 0600; never commit
or share that directory. No new public listener or host SSH password is needed.
The service and tunnel remain running when the frontend exits or `--no-frontend`
is used; neither starts motion by itself. Stop FR3/exit the session before
redeploying. A busy host service refuses the idle-only restart.

## Test the split workflow

1. Run the two component checks above. The SpaceMouse check should print the
   device name, six neutral axes and released buttons. The runtime check should
   pass without FCI connection. Check host route with `ip route get 192.168.11.102`.
2. Run `bash run/deploy.sh`. For a no-motion link check, run on Thor:
   `cd ~/lerobot && .venv/bin/python -m tools.thor.fr3_remote`.
   This performs 100 clock probes and cannot start the controller. On Live
   Record select two cameras plus BOX and press
   C. The host service remains idle, with no native FR3 worker or open SpaceMouse.
3. Release the puck and press F. Check host `outputs/logs/fr3_teleop/host.log` and
   `native_*.log` if startup fails. Only a reported robot fault requires Desk
   recovery. Link/runtime/clock errors need their own fix.
4. After running, move the puck gently and test both gripper buttons. Verify
   actual opening updates on Thor. Press E, record, then S; repeat with D.
   Compare FCI success with two cameras and with your full desired selection.
5. Inspect the saved `fr3_state.jsonl`: finite offset/RTT/uncertainty, valid mapped
   times, real torque/pose values and acknowledged gripper targets. Camera-aligned
   `fr3.valid` and `fr3.action_valid` must reflect the skew budget.
6. With the robot stationary and the operator at the stop control, test a
   deliberate link interruption. Both sides must stop, the UI must show an error,
   and reconnecting alone must not resume motion. Restore the link via deployment
   and explicitly press F only after the area is ready.

No-hardware regression suite (socketpairs and fake BOX/arm):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/scripts/test_thor_fr3_link.py tests/scripts/test_thor_fr3_session.py \
  tests/scripts/test_thor_fr3_control_worker.py tests/scripts/test_thor_fr3_data.py \
  tests/scripts/test_thor_fr3_deploy.py
```

Validation on 2026-10-04: host SpaceMouse open/read and native/kinematics preflight
passed; a read-only host FCI connection returned `kIdle`, no current errors, and
valid joint positions. The deployed SSH link completed 100 probes: median RTT
0.618 ms, maximum 0.871 ms, estimated uncertainty 0.549 ms. These measurements
were taken while idle, not during camera recording. With camera 6/7 and BOX
connected (no episode recording), another 100 probes measured median 0.625 ms,
maximum 0.899 ms and estimated uncertainty 0.442 ms. The test session was then
cleanly disconnected. Sustained host-driven motion
and loaded recording stability still require the operator F trial above.

## FCI quality policy and 2026-10-04 follow-up validation

SpaceMouse reports are user input; this profile polls them at 100 Hz and sends
telemetry over the host/Thor link at 50 Hz. Regardless of input frequency,
libfranka must keep its native FCI loop at 1 kHz. A slower mouse does not relax
robot communication deadlines. `control_command_success_rate` reports the last
100 native commands: 0.98 means two missed commands in that window, not a
20 Hz SpaceMouse issue. See [libfranka's state definition](https://raw.githubusercontent.com/frankarobotics/libfranka/0.15.0/include/franka/robot_state.h)
and [Franka timing guidance](https://frankarobotics.github.io/docs/troubleshooting.html).

The previous application guard stopped on every sample below 0.995. The host
profile now keeps 0.995 as the target and startup qualification threshold, but
allows a **warning for up to 0.2 seconds** at rates from 0.98 up to that target.
Below 0.98 stops immediately; continuous below-target readings for 0.2 seconds
also stop. These are application tuning choices, not manufacturer-certified
safety thresholds. Native reflexes, communication constraints, local state/input
watchdogs and joint/workspace bounds remain active. Do not lower this setting
to mask a native `communication_constraints_violation` fault. Omitting
`success_rate_grace_s` preserves the legacy immediate-stop policy.

Live Record shows a warning during a dip and retains the lowest observed rate
for the session. Sidecars retain `fci_quality_warning`, `fci_below_target_s`,
`fci_quality_dips`, and `fci_min_observed_success_rate`, so short dips are not
hidden by a later recovery to 1.0.

Investigation found the host service still had an rtprio ceiling of 0, CPU
policies were `powersave`, and both FR3 and Thor traffic use `eno2`. FIFO ceiling
99 is now installed for corenetic and inherited by new native workers;
performance governors are configured as above. The host profile now requires
FIFO permission even with `realtime_mode: ignore`; if that permission is lost,
F refuses before connecting or homing. `--check` from an old desktop terminal
with `ulimit -r = 0` also refuses: use a fresh login after permission setup.
There were no NIC hardware
error counters indicating cable corruption. Shared-NIC contention remains a
possible source of jitter: a dedicated direct FR3 cable/NIC is preferable.
`enp7s0` exists but currently has no link; no IPs or routes were changed.

The user's three-camera recording contained 1,321 telemetry samples with FCI
rate 1.0 throughout. Of 1,043 samples commanding translation, 981 also commanded
rotation. This established input cross-coupling as a major contributor to the
reported unwanted rotation. With the new IK weight, an offline URDF check over
±5 cm X/Y targets had maximum orientation residual below 0.0001 degrees and
position residual below 0.023 mm. That checks the kinematic target, not physical
servo tracking.

With all 11 cameras connected and their previews fetched at up to 5 Hz each,
a user-approved 15-second native neutral hold passed: 750 telemetry samples,
minimum/mean FCI rate 1.0, no below-target samples, and confirmed FIFO priority
99 native threads. The worker exited cleanly. Result and samples:
`outputs/diagnostics/fr3_host_20261004/neutral_all_cameras.json`.
This was a zero-input hold with preview traffic, not a moving-arm or full video
recording endurance test. Repeat an operator motion/recording trial for final
operational acceptance; the non-RT kernel remains a limitation.

The explicit repeatable neutral test performs physical homing, then forces all
Cartesian increments to zero; it never opens SpaceMouse or actuates BOX:

```bash
# Run from a login with ulimit -r = 99. Stop other FR3 control first.
PYTHONPATH=src:. .venv-fr3/bin/python -m tools.thor.fr3_neutral_check \
  --confirm-motion --seconds 15 \
  --output outputs/diagnostics/fr3_neutral_repeat.json
```

## Legacy direct-Thor setup and commissioning history

The material below describes `execution_host: thor` and earlier direct-Thor
tests. The default is now `execution_host: host`; use the host setup above.
The native safety limits and controller implementation are shared.

Deployment and C use Thor's existing `.venv`. F launches the arm in the separate
`.venv-fr3` interpreter configured by `fr3_teleop.runtime_python`. An unavailable
FR3 environment produces a UI error on F and leaves camera/BOX connection usable.

The checked Thor has kernel `6.8.12-tegra` with ordinary PREEMPT and no
`/sys/kernel/realtime`. The supplied profile explicitly selects
`fr3_teleop.realtime_mode: ignore`, matching the working native replay.
The robot address is `192.168.11.102`, reached through `enP2p1s0` from
`192.168.11.100`; the previous `192.168.1.206` address did not respond.
Direct SpaceMouse enumeration/open/read was verified on aarch64 using
`pyspacemouse==2.1.0`. Thor's current collection account has USB access and the
required packages. The live F test below covers homing and a short neutral-input
hold; sustained recording and user-driven motion still need operational testing.

### SpaceMouse USB access

Run these commands **on Thor**, after code has been synced:

```bash
cd ~/lerobot
bash run/setup_thor_spacemouse.sh
# Replug the SpaceMouse, then log out and back in for plugdev membership.
bash run/setup_thor_spacemouse.sh --check
```

The explicit setup installs hidapi libraries and the input dependencies in
`.venv`, adds udev/group access, and leaves FR3 untouched. The check lists the
device, opens it and prints axes/buttons using the owning device API in the
pinned `pyspacemouse==2.1.0`. Use `THOR_PYTHON=/path/to/python` if the
collection environment is elsewhere. Restart the gateway after logging back in.

### Real-time kernel and native FR3 runtime

FR3 control uses libfranka's native 1 kHz loop. The runtime supports two explicit
values of `fr3_teleop.realtime_mode`:

- `ignore`: the supplied Thor profile uses native `RealtimeConfig.kIgnore` and
  skips the host PREEMPT_RT/SCHED_FIFO gate, as requested for the replay-compatible
  setup. This does not establish a real-time timing guarantee. Joint bounds,
  command/state watchdogs, native robot errors and the FCI success-rate check
  remain enforced.
- `enforce`: require PREEMPT_RT and SCHED_FIFO permission, and pass native
  `RealtimeConfig.kEnforce`. This is the worker default when the setting is absent.
  Unknown values are rejected before connecting.

[libfranka's scheduling option](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html)
controls whether unavailable real-time scheduling raises an exception.
Even in `ignore` mode, libfranka attempts to give its control thread FIFO
priority. A login with `ulimit -r` equal to `0` prevents that attempt from
succeeding. The kernel-name bypass therefore does not guarantee that FCI timing
will meet `fr3_teleop.min_success_rate` under camera load. Scheduling permissions
are a separate host setting; changing the YAML does not grant them.
With operator approval, this Thor now has `/etc/security/limits.d/90-lerobot-fr3.conf`
containing `nvidia - rtprio 99`. New logins inherit that limit. The active
gateway/recorder limits were also updated so their next FR3 worker inherited it
without restarting sensor capture. Verify with `ulimit -r` in a new Thor SSH
login (expected `99`); the live test confirmed FIFO priority 99 in the worker.

After homing, the worker holds its current native joint target while it waits
for at least 100 ms of advancing active-controller state and a success rate
at or above `0.995`. This allows the native last-100-command metric to initialize.
SpaceMouse increments stay disabled until this check passes; startup aborts
after one second if it cannot qualify. During teleoperation the existing
per-sample success-rate limit, state freshness checks and input watchdogs apply.
Franka recommends a direct Ethernet connection to **Control's** LAN port and
validating timing under the intended load.
[Franka requirements and troubleshooting](https://frankarobotics.github.io/docs/troubleshooting.html)

NVIDIA documents a Thor RT kernel installation/build path. Choose the instructions
for the **installed Jetson Linux release** and verify that the Sengyun camera
drivers, device-tree overlays and NVIDIA modules work with that RT kernel before
commissioning FR3. Kernel/driver installation and reboot are a separate machine
maintenance step; deployment and these setup scripts do not change the kernel.
[NVIDIA Thor RT kernel instructions](https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/Kernel/RealTimeKernel.html)

For `realtime_mode: enforce`, after reboot check:

```bash
uname -a
cat /sys/kernel/realtime       # must print 1
ulimit -r                     # inspect realtime priority limits; preflight tests maximum FIFO priority
ip route get 192.168.11.102
```

Configure realtime limits for the login running the gateway according to the
Franka setup documentation, then start a fresh login/session. A kernel whose
name only contains `PREEMPT` does not satisfy the worker's `enforce` check.

Build/provide an **aarch64 panda-py wheel for Python 3.12** whose linked libfranka
version matches FR3's Desk system version. Generic panda-python installation
cannot be assumed suitable: upstream documents a default libfranka 0.9.2 build
for FER and FR3 requires a newer compatible build; its standard wheel build
configuration targets x86_64. Use a native aarch64 source build or a verified
matching wheel. The repository's existing `tools/fr3/setup_host_env.sh` shows
libfranka/panda-py source-build steps, but its machine/environment defaults must
be adapted deliberately for Thor; it is not invoked by deployment.
[panda-py installation](https://github.com/JeanElsner/panda-py),
[wheel architecture](https://github.com/JeanElsner/panda-py/blob/main/pyproject.toml),
[Franka version compatibility](https://frankarobotics.github.io/docs/compatibility.html)

The upstream native Panda controller can call automatic error recovery when
starting a controller or moving to start. This workflow requires a patched
native wheel that throws on that condition and exports
`panda_py._core.FR3_NO_AUTOMATIC_ERROR_RECOVERY = True`. The worker checks the
compiled extension's capability before connecting. A normal upstream wheel
fails preflight/F; setting a Python variable is not a substitute for the patch.

Prepare a local checkout of the repository's inspected panda-py revision:

```bash
git clone https://github.com/linkage-x/panda-py.git /tmp/panda-py-thor
git -C /tmp/panda-py-thor checkout 47c304f9b8147ae7582dcfe8a97af554c363021d
python3 run/patch_thor_panda_py.py /tmp/panda-py-thor
python3 run/patch_thor_panda_py.py --check /tmp/panda-py-thor
```

The helper accepts only the inspected `Panda::recover` body, replaces its native
recovery call with an error, verifies no additional native recovery calls remain,
releases Python's GIL while stopping the native controller, then adds the compiled
capability. Shutdown also copies native state through Panda's mutex before using
it. The helper refuses unknown source changes. Build the
patched wheel **on Thor** against the installed compatible libfranka development
files. For a libfranka installation under `/usr/local`, for example:

```bash
# Existing native build environment with Python 3.12, uv, CMake and libfranka.
CMAKE_PREFIX_PATH=/usr/local LD_LIBRARY_PATH=/usr/local/lib \
  uv build --wheel --python 3.12 /tmp/panda-py-thor
```

Include your Placo/Pinocchio `cmeel.prefix` in these paths if that libfranka build
uses its dependencies. Inspect the generated wheel's architecture/Python tag and
install that wheel below. Source-build prerequisites and the exact libfranka
version depend on the installed Desk/BSP versions and cannot be inferred from
the robot IP address.

With a matching wheel available on Thor, provision the separate environment:

```bash
cd ~/lerobot
bash run/setup_thor_fr3.sh --install-system-deps --install-python \
  --panda-wheel /absolute/path/to/panda_python-compatible-aarch64.whl
```

This explicit command installs native prerequisites, syncs LeRobot's declared
core dependencies with kinematics/SpaceMouse, adds Ruckig and the supplied
panda-py wheel, then runs the local preflight. It does not open FCI or move the
robot. The FR3 adapter imports LeRobot's processor stack, so this environment is
larger than the camera/BOX environment. Placo/Pinocchio and panda-py may require
native aarch64 builds; resolve build/import errors before F. A successful package
installation alone does not establish compatible firmware or realtime behavior.

If the compatible runtime is already installed, run only:

```bash
bash run/setup_thor_fr3.sh --check
# Equivalent worker check; no robot connection:
PYTHONPATH=src:. .venv-fr3/bin/python -m tools.thor.fr3_control_worker \
  --check --config-path tools/thor/gmsl2/thor_fr3_teleop.yaml
```

`--venv /path/to/environment` selects another environment for setup/check; update
`fr3_teleop.runtime_python` in the YAML to its Python executable before deployment.

The 2026-10-04 Thor setup built the inspected source revision with
`run/patch_thor_panda_py.py` against installed libfranka 0.15.0. The resulting
wheel is stored on Thor at
`outputs/wheels/panda_python-0.8.1-cp312-cp312-linux_aarch64.whl` and installed in
`.venv-fr3`. Its compiled no-automatic-recovery capability and the full
`bash run/setup_thor_fr3.sh --check` preflight passed in `ignore` mode.
The replaced package was backed up under `outputs/runtime_backups/fr3_20261004`.
Live connection to `192.168.11.102` returned idle mode, no current robot errors,
and valid joint/torque/pose data. With 11 cameras and BOX sensors connected, the
final F test moved to start, entered SpaceMouse control with the puck released
for five seconds, then stopped cleanly. All 44 sampled UI telemetry packets
reported `control_command_success_rate=1.0`; the configured limit stayed `0.995`.
The final direct state read showed `kIdle` and no current errors. The test report
is `outputs/logs/fr3_teleop/live_check_ready_20261004.json` on Thor and the host.
No episode was recorded in this test. The initial startup-rate failures were
resolved with the approved FIFO permissions and startup qualification above.

## Configuration

All connected Sengyun cameras use the original generic `cam_00`, `cam_01`, etc.
`sensors.cameras.detect_all: true` discovers locked ports and probes them at C.
A newly added end-effector camera participates in the same capture, preview,
synchronization and export as other cameras. No wrist role or special ID is
required in this round. Reconnect after changing camera cabling/topology.
Adding a camera on an unused generic port works through that detection path.
Replacing a camera on a previously calibrated port can trigger the retained BOX
camera-identity gate; update its expected identity and repeat the relevant
calibration before using results that depend on that calibration.

The FR3 extension has these principal settings:

| Setting | Default / meaning |
| --- | --- |
| `robot.robot_ip` | `192.168.11.102`, the working replay address, contacted directly from Thor after F |
| `robot.urdf_path` / `target_frame_name` | FR3 + Corenetic gripper model / `corenetic_gripper_ee` |
| `fr3_teleop.runtime_python` | `.venv-fr3/bin/python`, relative to the deployed repository |
| `fr3_teleop.realtime_mode` | `ignore` in this Thor profile; `enforce` requires PREEMPT_RT and SCHED_FIFO |
| `fr3_teleop.control_hz` | 100 Hz Python target updates and native state reads; native FCI remains 1 kHz |
| `fr3_teleop.command_timeout_s` | 0.2 s parent command watchdog |
| `fr3_teleop.max_state_age_s` | 0.1 s default maximum age of the native telemetry stream |
| `fr3_teleop.startup_timeout_s` | 60 s to connect, move to start and become ready |
| `fr3_teleop.min_success_rate` | 0.995 startup/quality target; host profile allows a 0.2 s warning at rates ≥0.98 |
| `fr3_teleop.gripper_box_id` | Blank accepts exactly one BOX; specify the real ID for multiple BOXes |
| `fr3_teleop.gripper_max_width_m` | 0.09 m |

The BOX remains owned by the existing recorder. The arm worker uses a mock
gripper adapter internally to avoid opening a second BOX session; actual gripper
commands and measured opening come from the recorder's existing BOX client.

The pinned Panda controller has virtual joint walls originally defined for FER.
This configuration uses a conservative FR3/common joint envelope clear of those
walls' damping zones. The worker checks the native wall constants, initial joint
position and each target against the configured limits. If F reports an
out-of-range initial pose, reposition the robot using Desk into the documented
envelope before retrying; F does not drive through the boundary to recover it.
Only reduce this envelope for the current pinned native build:

| Joint | Minimum (rad) | Maximum (rad) |
| --- | --- | --- |
| 1 | -2.64 | 2.64 |
| 2 | -1.57 | 1.57 |
| 3 | -2.70 | 2.70 |
| 4 | -2.84 | -0.27 |
| 5 | -2.70 | 2.70 |
| 6 | 0.60 | 3.65 |
| 7 | -2.70 | 2.70 |

Any reduced joint envelope must still contain the compiled native move-to-start
joint configuration; preflight checks this before FCI. Per-joint OTG dynamics
must be finite and positive and can only be reduced from the initial caps:
0.5 rad/s velocity, 1.0 rad/s² acceleration and 1000 rad/s³ jerk.

## Component tests and commissioning

Before the first supervised F test, configure Desk's tool/end-effector and load
parameters and the URDF/TCP to match the mounted Corenetic gripper, added camera
and other attachments. Check the move-to-start path with those attachments in
place. The software's conservative limits do not establish payload or tool
calibration for a changed physical setup.

| Test | Procedure | Expected result |
| --- | --- | --- |
| Deploy / UI | `bash run/deploy.sh` on the host | Only Thor SSH; original BOX Live Record interface, F added |
| Cameras / BOX | Press C before provisioning FR3 | Existing sensor connection and generic camera previews work; no robot motion |
| SpaceMouse | `bash run/setup_thor_spacemouse.sh --check` on Thor | Device opens and supplies axes/buttons without FR3 |
| Native readiness | `bash run/setup_thor_fr3.sh --check` on Thor | Selected scheduling policy, configuration, patched native capability and imports pass without FCI |
| F failure isolation | With a missing FR3 runtime, press F after C | Specific UI error; cameras/BOX remain connected; F can be retried |
| Motion / input | Commission the robot, enable FCI in Desk, then press F | Move to start completes before SpaceMouse changes targets |
| Episode controls | After F is running: E then S; E then D; E then wait 30 seconds | Each episode ends before FR3 returns to start; E remains disabled during the return |
| Fault / retry | During a supervised test stop the robot, then clear the fault and press F | Alert, stopped motion, active episode discarded; fresh move-to-start on F |
| Shutdown | Stop FR3, Esc, and deployment restart | Owned worker releases robot control; a new Connect/F does not compete with an orphan |

Before motion commissioning, validate FCI using the compatible libfranka build's
`communication_test` and repeat under the actual camera preview/recording/BOX
load. **This example moves the robot** before measuring communication; follow
its prompt and operate it as a supervised motion test.
[communication_test source](https://github.com/frankarobotics/libfranka/blob/main/examples/communication_test.cpp)

The 1 kHz controller runs in native code in its own process. Camera acquisition,
BOX I/O, UI, SpaceMouse sampling and file writes remain outside that callback.
Only the latest target is exchanged over local IPC; old targets are not replayed.
If a SpaceMouse HID report stops advancing, the arm receives zero motion and the
gripper holds its last command until a fresh report arrives. A released mouse
may legitimately send no reports; a brief report gap does not require pressing F.
Loss of the parent, stale commands/state, native controller errors or a low
success rate stop control and require F again. These guards detect failures;
they cannot guarantee that FCI communication constraints remain satisfied under
unmeasured system load. Native scheduling/CPU/IRQ tuning and loaded testing are
still required on this specific Thor/camera configuration.

## Recorded data

Videos are saved **on Thor**, even when SpaceMouse and FR3 control run on the
host. Raw camera files are under
`/home/nvidia/lerobot/outputs/datasets/<dataset>/episodes/episode_000000/cam_XX.mkv`.
The raw capture does not require a top-level `videos/` directory; that layout is
used by video export. The host's `outputs/datasets` is not automatically mirrored.
In the UI, select the dataset and episode in **Episode Replay**, then use the
**Replay Inspector** camera tiles and Play. The first video request creates an
H.264 `.mp4` playback cache beside each original H.265 `.mkv`; allow time for
that conversion. The original recordings are retained.

For `thor_gmsl2_3ch_v1_20261004_164412`, episode 0 was verified to contain
`cam_03.mkv`, `cam_07.mkv` and `cam_13.mkv`: each has 1,191 readable frames,
1920×1080 at 60 fps, duration 19.85 seconds, and approximately 49.9 MB.

Existing per-camera MKVs, synchronized Argus timestamps, BOX force/tactile/gripper
samples and episode metadata remain intact. FR3 recordings also include
`fr3_state.jsonl` with measured joint position/velocity, measured joint torque,
filtered external joint torque, configured end-effector transform, estimated external wrench,
task TCP, command target and timing/communication information. Recording begins
only after F reports running; errors discard incomplete episodes.

The live LeRobot v3 writer and offline export preserve aligned FR3 measurements,
pose/gripper actions and validity/timestamp fields alongside the original camera
and BOX features. Measured TCP and the commanded target are different fields;
the native `O_T_EE` transform is stored column-major in libfranka's convention.
Samples outside the alignment tolerance are marked invalid rather than invented.
The FR3-to-camera alignment tolerance defaults to 25 ms; validity fields identify
frames without a sufficiently close measured robot sample.
The dataset root, task overlays, calibration inputs and export UI keep their
original BOX defaults.

## Software checks without hardware

From the development checkout with its Python test environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HOME=/tmp/lerobot-fr3-test-hf \
  HF_DATASETS_CACHE=/tmp/lerobot-fr3-test-hf/datasets PYTHONPATH=src:. \
  .venv/bin/python -m pytest -q tests/scripts/test_thor_fr3_*.py \
  tests/teleoperators/test_spacemouse.py tests/robots/test_franka_research3.py

cd tools/data_collection_gui/frontend
npm test
npm run build
```

These checks use fake robot/USB devices, temporary files and local IPC. They
verify startup order, input/state watchdogs, fault/discard/retry, command bounds,
ownership, recording/export and deployment routing. They do not certify physical
motion or loaded FCI timing; use the commissioning procedures above for that.
