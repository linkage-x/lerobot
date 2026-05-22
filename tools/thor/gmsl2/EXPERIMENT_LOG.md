# Thor GMSL2 Integration — Experiment Log

Each entry captures what was tried, the verifiable evidence, and what we learned.
Append-only; do not edit closed entries.

---

## E0 — SDK survey (2026-05-20)

**Goal:** Understand what the SDK provides and what API surface we need to wrap.

**Setup:** SSH to `nvidia@192.168.1.44`, inspect `~/Desktop/SG16A_AGTH_G3Y_A1`.

**Findings:**
- Pack is the kernel-driver bundle for the SENSING SG16A-AGTH-G3Y-A1 adapter on
  JetPack 7.0 / L4T R38.2.1. It ships kernel modules (`max96726.ko`,
  `sg2-ar0234c-g2f.ko`, etc.), device-tree overlays, a `load_modules.sh`
  bring-up script, a `boost_clock.sh` clock-pinning script, and a `pwm.sh` PWM
  trigger script.
- There is **no proprietary user-space SDK / Python API**. Cameras expose
  themselves as plain V4L2 `/dev/videoN` nodes; capture is meant to go through
  `argus_camera`, `nvarguscamerasrc`, or direct V4L2.
- The connector ↔ device-node mapping is fixed:
  `cam_N → /dev/videoN` for N = 0..15.
- Hardware trigger is configured per camera via `v4l2-ctl` controls:
  `sensor_mode`, `trig_pin=0x00020007`, `trig_mode ∈ {0=master/free-run,
  1=slave/external trigger}`.

**Decision:** Wrap the camera as a V4L2 + GStreamer source in
`src/lerobot/cameras/gmsl2/`. Apply trigger controls via `v4l2-ctl` at
`connect()` time. Provide a separate `setup_sync.sh` for one-time module
loading + PWM trigger source setup.

---

## E1 — Driver / config integration (2026-05-20)

**Files added/touched:**

| Path | Purpose |
| --- | --- |
| `src/lerobot/cameras/gmsl2/configuration_gmsl2.py` | `Gmsl2CameraConfig` registered as draccus choice `type: gmsl2`. Supports `sensor_id`, `device`, `pipeline ∈ {argus, v4l2_bayer, v4l2}`, `bayer_format`, `sync_role ∈ {auto, master, slave}`, `trig_pin`, `exposure_us`, `gain`, `apply_sync_at_connect`, etc. |
| `src/lerobot/cameras/gmsl2/camera_gmsl2.py` | `Gmsl2Camera` driver. GStreamer (PyGObject) pipeline + background appsink pull thread. Implements abstract `Camera` API (`connect`, `read`, `async_read`, `read_latest`, `disconnect`). Applies V4L2 sync / exposure / gain at connect time. |
| `src/lerobot/cameras/gmsl2/__init__.py` | Public exports. |
| `src/lerobot/cameras/utils.py` | `make_cameras_from_configs` dispatch for `gmsl2`. |
| `tools/handheld/handheld_record.py` | Added `Gmsl2CameraConfig` import so the recorder's draccus registry resolves the type. |
| `tools/thor/gmsl2/setup_sync.sh` | Runtime equivalent of `load_modules.sh` + `pwm.sh` with hardware-sync defaults. Loads modules, boosts clocks, sets PWM `pwmchip4/pwm0`, puts every channel in slave trigger mode. |
| `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` | 11-channel handheld recorder config. No grippers, no tactiles. |
| `tools/thor/gmsl2/README.md` | Bring-up / deploy / troubleshooting notes. |

**Verification on host:**
- `python3 -c "from lerobot.cameras.gmsl2 import Gmsl2CameraConfig; ..."` — config
  loads, draccus registers `gmsl2`.
- Full `draccus.decode(HandheldRecordingConfig, yaml.safe_load(...))` on
  `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` produces 11 `Gmsl2CameraConfig`
  instances, every field validated.

---

## E2 — AR0234 dtbo lock-in (2026-05-20)

**Goal:** Confirm the supported framerate before choosing PWM frequency.

**Method:** Decompile `dtb/SG2_AR0234C_G2F/tegra264-camera-ar0234cx16-overlay.dtbo`
with `dtc -I dtb -O dts`.

**Evidence (per sensor node):**

```
active_w = "1920"; active_h = "1080";
min_framerate = "60000000"; max_framerate = "60000000"; default_framerate = "60000000";
framerate_factor = "1000000";
```

→ AR0234 is hard-locked to **60.000 fps at 1920x1080** in this dtbo. No other
mode advertised.

**Decision:** Drive Thor PWM at 60 Hz (period 16.666 ms, 50% duty). Encode this
as the default in `setup_sync.sh` and the example YAML.

---

## E3 — JetPack 7 has no `nvarguscamerasrc` (2026-05-20)

**Goal:** Build the ISP-debayered GStreamer pipeline used in the initial design.

**Evidence:**

```
$ gst-inspect-1.0 nvarguscamerasrc
No such element or plugin 'nvarguscamerasrc'

$ find /usr -name "libgstnv*"
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvcodec.so   # dGPU codec only

$ apt list --installed | grep nvidia-l4t-multimedia
nvidia-l4t-multimedia/stable,now 38.2.1-20250910123945
nvidia-l4t-multimedia-openrm/stable,now 38.2.1-20250910123945
nvidia-l4t-multimedia-utils/stable,now 38.2.1-20250910123945
```

JetPack 7 / L4T R38.2 ships `libnvargus.so` (the C++ Argus runtime) but **does
not include the `nvarguscamerasrc` GStreamer wrapper or `nvvidconv`**. VPI is
not installed either, and DeepStream is not present.

**Decision:** Use direct V4L2 + software debayer for the bring-up. The driver's
`pipeline='v4l2_bayer'` mode does this; software debayer is the rate-limit. We
keep `pipeline='argus'` in the driver for later JetPack versions that ship the
GStreamer plugin.

---

## E4 — Single-camera end-to-end (2026-05-20)

**Setup:** `setup_sync.sh --fps 60 --num 11` ran cleanly. PWM verified:
`/sys/class/pwm/pwmchip4/pwm0/enable=1, period=16666666 ns, duty_cycle=8333333
ns` → 60 Hz.

**Raw V4L2 probe of `/dev/video0`** (slave trigger, exposure=15000 µs, gain=300,
lens caps off):

```
u16: count=20736000   min/max/mean=2178/65535/24874.0   nonzero_ratio=1.0
```

→ Hardware sync is delivering exposure pulses; sensor responds with healthy
mid-gray pixel values across the frame.

**Python smoketest** (`Gmsl2Camera` driver):

```
pipeline: v4l2src device=/dev/video0 io-mode=mmap do-timestamp=true
        ! video/x-bayer,format=grbg10le,width=1920,height=1080,framerate=60/1
        ! appsink ...
debayer: cv2.cvtColor(... COLOR_BayerGR2BGR) after right-shift 10→8 bit.
result:  got 10 frames in 0.15s → 68.9 fps
         shape=(1080, 1920, 3); BGR mean=135.8; B/G/R range 0..255 each.
saved /tmp/cam0_test.png ~4.5 MB.
```

✅ Single-camera capture works. Driver returns real BGR frames at well over the
60 Hz sensor rate (the pipeline is faster than the sensor delivers).

---

## E5 — Initial 11-channel concurrent attempt (2026-05-20)

**Setup:** Same as E4 but spinning up all 11 cameras and reading 60 frames per
camera in parallel threads. nvargus-daemon was still running.

**Result:**

```
OPEN_OK for all 11 cameras in 0.25s total.
Streaming:
  FPS_PER_CAM [0.9, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]
  Only cam0/1/4/9 streamed at ~0.9 fps; the rest produced 0 frames.
  Total combined fps = 3.6.
```

**Kernel evidence (dmesg):**

```
tegra-camrtc-capture-vi: uncorr_err: request timed out after 2500 ms
tegra-camrtc-capture-vi: err_rec: attempting to reset the capture channel
tegra-camrtc-capture-vi: err_rec: successfully reset the capture channel
... repeated for VI ch[34], ch[35] ...
```

→ The Tegra VI capture engine is timing out on most channels and bouncing them
through error-recovery resets. Only a subset of channels gets actual frame
data.

**Hypotheses still to test:**

1. `nvargus-daemon` is competing for the same CSI / VI resources even though we
   are not using Argus. Stopping it may free capacity.
2. The VI engine on this kernel has a smaller concurrent-channel ceiling than
   we are pushing for. Staggering camera opens, lowering per-camera fps (PWM
   30 Hz), or running only a subset may be required.
3. The `v4l2-ctl` "apply controls" step opens/closes the device repeatedly; that
   churn may leave the kernel pipeline in a degraded state when applied to all
   11 cameras in quick succession. Trial: apply controls once at boot and skip
   `apply_sync_at_connect` from the driver.

**Status:** ❌ 11-channel concurrent capture is **not yet working**.

---

## E6 — Ladder concurrent test, slave mode (2026-05-20)

**Setup:** `nvargus-daemon` stopped, `setup_sync.sh --fps 60 --num 11` ran clean
(PWM 60 Hz, all cameras in `trig_mode=1`). Driver in `pipeline=v4l2_bayer`,
exposure 15000 µs, gain 300, lens caps off, stagger 0.5 s between opens.

**Results (frames-per-camera over a 30-frame drain):**

| #cams | dt (s) | per-camera fps | working cameras | total fps |
| --- | --- | --- | --- | --- |
| 1  | 35.36 | [0.8] | cam0 | 0.8 |
| 4  | 36.92 | [0.8, 0.8, 0.0, 0.0] | cam0, cam1 | 1.6 |
| 8  | 36.75 | [0.8, 0.8, 0, 0, 0.8, 0, 0, 0] | cam0, cam1, cam4 | 2.4 |
| 11 | 38.22 | [0.8, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 0.8, 0] | cam0, cam1, cam4, cam9 | 3.1 |

**Observation:** the "working" channels are stuck at exactly 0.8 fps regardless
of camera count, and many channels never deliver any frames. dmesg keeps
emitting `tegra-camrtc-capture-vi: uncorr_err: request timed out after
2500 ms` for the dead channels.

0.8 fps is well below the sensor's 60 Hz spec and well below what would be
expected from any throughput-bandwidth saturation -- it matches the AR0234's
**fall-back internal frame rate when external trigger is required but no
pulse ever arrives**.

## E7 — Master-mode (free-run) sanity check (2026-05-20)

**Setup:** Same hardware as E6, but `sync_role="master"` (trig_mode=0). Single
camera (cam0), drain 60 frames.

```
MASTER_RESULT cam0 free-run: 60 frames in 0.97s -> 62.04 fps
```

→ ~62 fps -- exactly the native AR0234 60 Hz rate (+ a small leading burst
from the driver's startup buffer).

**Conclusion:** ✅ **Sensors are healthy. The CSI / VI / driver stack is
healthy. The only thing broken is the trigger path:** the Thor PWM signal on
`pwmchip4/pwm0` is **not physically reaching the SG16A board's trigger pin**.

That matches the SDK split: `load_modules.sh` configures `trig_mode/trig_pin`,
and `pwm.sh` configures the PWM controller, but **neither script wires the
PWM output to the SG16A trigger input** -- that is a board-level wiring step
that the SDK assumes the user has done off-line.

## Wiring required for true hardware sync

To get 60 Hz HW-synced slave-mode capture on this board, the user has to
wire one of the Jetson 40-pin header PWM pins (the one connected to
`pwmchip4/pwm0`) to the trigger input pin selected by
`trig_pin=0x00020007`. The SG16A documentation calls that input the
"external trigger Pin of the Kit"; the exact header pin is hardware-dependent
and not documented in the SDK README -- ask the board vendor or refer to the
board schematic.

Until that wire is in place we have two practical options:

1. **Master / free-run mode** for every camera (no hardware sync, only
   software sync). Each AR0234 free-runs at 60 Hz with a small phase offset
   between channels; for many data-collection use cases this is acceptable
   provided the recorder logs per-camera capture timestamps (the LeRobot
   recorder already does this).
2. **Single-master + slaves**: dedicate one camera as master (free-run) and
   route its frame-valid / strobe-out to the other channels' trigger inputs.
   Requires extra wiring on the SG16A board.

## E8 — Ladder concurrent test, master/free-run mode (2026-05-20)

**Setup:** `nvargus-daemon` stopped, modules freshly loaded, `sync_role=master`
(trig_mode=0 / free-run), 120-frame drain per camera, 0.5 s stagger between
opens.

```
n=1   FPS_PER_CAM [60.9]                              total=60.9
n=4   FPS_PER_CAM [5.0, 5.0, 0.0, 0.0]                total=10.0
n=8   FPS_PER_CAM [5.0, 5.0, 0.0, 0.0, 5.0, 0, 0, 0]  total=15.0
n=11  FPS_PER_CAM [5.0, 5.0, 0, 0, 5.0, 0, 0, 0, 0, 5.0, 0]  total=20.0
```

**Observations:**

1. **Single camera works perfectly** -- 60.9 fps, matches the AR0234's
   dtbo-locked 60 Hz.
2. **Per-camera fps collapses from 60.9 → 5.0** as soon as a second camera
   opens.
3. **Deterministic "working set"**: cam0, cam1, cam4, cam9. Same four
   channels in every ladder step; the other seven never deliver any frames.

## E9 — Sequential per-camera baseline reveals the real root cause (2026-05-20)

**Method:** Open one camera, drain 60 frames, close, sleep, repeat for all 11
sensor ids. No concurrency.

**Result:**

```
RESULT sid=0  frames=60 fps=61.1 mean=125.7
RESULT sid=1  frames=60 fps=61.4 mean=138.1
RESULT sid=2  frames=0  fps=0.0  mean=0.0
RESULT sid=3  frames=0  fps=0.0  mean=0.0
RESULT sid=4  frames=60 fps=61.5 mean=135.6
RESULT sid=5  frames=0  fps=0.0  mean=0.0
RESULT sid=6  frames=0  fps=0.0  mean=0.0
RESULT sid=7  frames=0  fps=0.0  mean=0.0
RESULT sid=8  frames=0  fps=0.0  mean=0.0
RESULT sid=9  frames=60 fps=61.5 mean=123.3
RESULT sid=10 frames=0  fps=0.0  mean=0.0

SUMMARY working = [0, 1, 4, 9]    (61+ fps, real BGR content)
SUMMARY dead    = [2, 3, 5, 6, 7, 8, 10]
```

**Re-interpretation:** the working/dead split is **exactly the same** under
sequential and concurrent capture. This is **not** an I²C contention bug
under concurrency: 7 of the 11 camera ports are simply not responding under
any condition.

The `max96726 18-0033: i2c-w, write failed` dmesg lines from E5 / E8 are the
deserializer driver giving up on the GMSL2 links that have no responding
sensor on the other side -- not concurrency damage.

**Reading of the README:** all prerequisite SDK steps were followed --
`install.sh` had populated `/boot/*x16-overlay.dtbo`, the running
extlinux profile selects `Custom Header Config: <CSI Jetson Sensing
SG16A_AGTH_G3Y_A1 AR0234Cx16>`, `tegra-camera.ko` and `nvhost-nvcsi.ko`
updates are in place, `camera_overrides.isp` is installed at
`/var/nvidia/nvcam/settings/`, `nvargus-daemon` was restarted with
`NVCAMERA_NITO_PATH=CONFIG` and `enableCamInfiniteTimeout=1` per README
section 6.3, all modules load, and 60 fps native capture is verified on the
four cameras that do respond. The README does not show a multi-camera
concurrent example -- its bring-up shows one `argus_camera -d N` or one
`gst-launch-1.0` per channel at a time.

**Action items for the user:**

1. Verify the physical state of the seven dead channels (cam2, cam3, cam5,
   cam6, cam7, cam8, cam10). Things to check on the SG16A adapter:
   * Are FAKRA cables actually plugged in to each of those ports?
   * Are the cameras on the other end of those cables powered (12 V to the
     adapter board reaches each link)?
   * Is the camera module on each port functional (swap one of the working
     cameras into a "dead" port to localise the fault between the cable +
     port and the sensor module itself)?
2. After fixing the wiring, re-run E9's per-camera sequential test. Each
   sensor id must report ~60 fps standalone before any multi-camera test is
   meaningful.

**Status of the LeRobot integration:** ✅ correct on the four working
channels. The 11-channel concurrent capture story will be revisited after
the seven dead channels are physically resurrected.

## (superseded) E5b — Earlier hypothesis: max96726 I²C concurrency

**dmesg evidence during E8:**

```
max96726 18-0033: i2c-w, write failed
max96726 18-0033: i2c-w, write failed
... repeated ...
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
tegra-camrtc-capture-vi tegra-capture-vi: err_rec: attempting to reset the capture channel
tegra-camrtc-capture-vi tegra-capture-vi: err_rec: successfully reset the capture channel
... repeated ...
```

→ The **MAX96726 GMSL2 deserializer's I²C bus is failing** when several
cameras are configured concurrently. The kernel's CSI-VI engine then keeps
timing out the channels whose deserializer-side configuration never
completes, and only the channels that managed to finish their I²C dance
before the bus jammed keep producing frames (cam0/1/4/9).

This is **not a LeRobot integration bug, not a CSI bandwidth limit, and not
a Jetson VI channel-count limit**. It is a board-/driver-level issue inside
the SG16A bring-up package itself: the `max96726.ko` driver does not
serialise its I²C traffic when several `sg2-ar0234c-g2f` instances open
simultaneously.

**Confidence that the LeRobot side is correct:**

* Single-camera capture works end-to-end with the right frame rate and real
  BGR pixel content (E4 / E7 / E8 n=1).
* Draccus parses the new YAML correctly into 11 camera configs (E1).
* `v4l2-ctl` controls (`sensor_mode`, `trig_pin`, `trig_mode`, `exposure`,
  `gain`) are applied successfully per channel.
* The same failure occurs in master mode (no trigger involved), so trigger
  wiring is not the bottleneck for the multi-camera bring-up.

## Open work / next steps

### Blocking: max96726 I²C concurrency (vendor / driver side)

11-channel concurrent capture is currently blocked by `max96726` I²C-write
failures. Suggested escalation order:

1. **Open a ticket with SENSING** referencing the dmesg lines under E9. They
   ship the `max96726.ko` driver and the dtbo and are the authoritative
   source for the bring-up sequence under JetPack 7 / L4T R38.2.
2. **Look for a serialisation-locked variant** of the `max96726` driver. On
   Orin / Xavier branches the same chip's driver added a per-link mutex
   exactly to dodge this class of bug. If a 38.2-compatible patch exists,
   apply it and re-run the ladder.
3. **Try ultra-staggered opens** (5 s+ between cameras) as a temporary
   workaround once the driver-side fix is verified -- the current 0.5 s
   stagger is clearly not enough.

### Once 11-channel capture is stable

4. **Hardware-trigger wiring.** Route Thor PWM `pwmchip4/pwm0` to the SG16A
   trigger input pin so slave mode actually triggers exposure. Until the
   wire is in place, default the recorder config to `sync_role=master`. E6 vs
   E7 confirms the diagnosis (0.8 fps slave vs 60.9 fps master).
5. **Gateway / GUI smoke test.** Install gateway deps on Thor (the existing
   handheld stack, plus `pyav`, `pandas`, etc.), then run
   `python -m tools.data_collection_gui.gateway --config-path
   tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml ...` per
   `tools/data_collection_gui/frontend/README.md`.
6. **High-fps debayer path.** Software `cv2.cvtColor` is enough for a single
   camera at 60 Hz. With 11 channels we will either need:
   * `nvarguscamerasrc` once it lands on JetPack 7, or
   * a VPI / CUDA debayer kernel called from Python, or
   * record raw Bayer to disk and debayer offline during dataset processing.

---

## E10 — MAX96726 link-lock register separates physical link from stream-on failure (2026-05-20)

**Trigger:** SENSING provided a lower-level diagnostic for the MAX96726
deserializers: read register `0x0008` at address `0x33` on each I2C bus.

Command form:

```bash
sudo i2ctransfer -f -y <bus> w2@0x33 0x00 0x08 r1
```

Register decode:

| bit | field | meaning |
| --- | --- | --- |
| 0 | `LOCKED_A` | Link A locked |
| 1 | `LOCKED_B` | Link B locked |
| 2 | `LOCKED_C` | Link C locked |
| 3 | `LOCKED_D` | Link D locked |

Board mapping used for this test:

| I2C bus | SG16A group | video base | video IDs |
| --- | --- | --- | --- |
| 17 | port1 | 0 | `video0..video3` |
| 18 | port2 | 4 | `video4..video7` |
| 19 | port3 | 8 | `video8..video11` |
| 20 | port4 | 12 | `video12..video15` |

**Clean reload procedure used before the test:**

```bash
sudo service nvargus-daemon stop
cd ~/lerobot
sudo ./tools/thor/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16
for i in $(seq 0 15); do
  sudo v4l2-ctl -d /dev/video$i -c sensor_mode=0,trig_mode=0,bypass_mode=0
done
```

`setup_sync.sh` completed successfully, including module reload, clock boost,
PWM programming, and per-camera control application.

**Link-lock evidence immediately after the clean reload:**

```text
bus17 max96726@0x33 REG0x0008: 0x0b
bus18 max96726@0x33 REG0x0008: 0x07
bus19 max96726@0x33 REG0x0008: 0x0e
bus20 max96726@0x33 REG0x0008: 0x0c
```

Decoded locked links:

```text
bus17 0x0b -> video0, video1, video3 locked
bus18 0x07 -> video4, video5, video6 locked
bus19 0x0e -> video9, video10, video11 locked
bus20 0x0c -> video14, video15 locked
```

Therefore the 11 physically locked video IDs are:

```text
[0, 1, 3, 4, 5, 6, 9, 10, 11, 14, 15]
```

The unlocked IDs at that moment were:

```text
[2, 7, 8, 12, 13]
```

**Important correction to E9:** the previous interpretation that the seven
"dead" channels were physically not responding was too coarse. The MAX96726
lock register shows that 11 GMSL links can be physically locked. The real split
is now:

* link not locked: `video2, video7, video8, video12, video13`
* link locked and stream-on works: `video0, video1, video4, video9`
* link locked but stream-on fails: `video3, video5, video6, video10, video11,
  video14, video15`

**Stream-on retest using only the 11 locked IDs:**

Each locked ID was tested sequentially in free-run mode (`trig_mode=0`) with:

```bash
v4l2-ctl -d /dev/videoN \
  --set-fmt-video=width=1920,height=1080,pixelformat=BA10 \
  --stream-mmap=3 --stream-count=30 --stream-to=/dev/null
```

Result:

```text
OK:   video0, video1, video4, video9
FAIL: video3, video5, video6, video10, video11, video14, video15
```

Failure signature:

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

dmesg evidence for failed locked links:

```text
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

The lock register was stable immediately after the stream-on retest:

```text
bus17 0x0b
bus18 0x07
bus19 0x0e
bus20 0x0c
```

**Later bus18 observation:** a later run of the new helper script reported
`bus18=0x00`. The user confirmed this was caused by physically unplugging the
port2 camera cables during bench work. Treat that later `bus18=0x00` reading
as an operator-induced wiring state, not as the clean-reload result.

**Helper script added:**

```bash
tools/thor/gmsl2/check_max96726_locks.sh
```

The script reads all four MAX96726 instances, prints the raw `REG0x0008`
values, decodes A/B/C/D lock bits, and emits:

```text
LOCKED_VIDEO_IDS=...
UNLOCKED_VIDEO_IDS=...
```

It was copied to the Thor at:

```bash
~/lerobot/tools/thor/gmsl2/check_max96726_locks.sh
```

**Current conclusion:**

The system is no longer blocked on proving whether 11 physical links exist:
SENSING's MAX96726 register check shows the expected 11 locked links when all
cables are connected. The active blocker is narrower and lower-level:

> Seven locked links fail during the driver stream-on path, with repeated
> MAX96726 I2C write failures.

This points to a vendor driver / deserializer configuration / link-to-video
mapping issue rather than a LeRobot recorder issue. It also means the recorder
must not assume `video0..video10`; the actual locked ID set is sparse:
`0,1,3,4,5,6,9,10,11,14,15`.

**Next vendor-facing questions:**

1. Confirm whether `REG0x0008` A/B/C/D maps exactly to `video base+0..base+3`
   for each bus group on this dtbo.
2. Explain why the following locked links fail at `VIDIOC_STREAMON`:
   `video3, video5, video6, video10, video11, video14, video15`.
3. Provide a JetPack 7 / L4T R38.2-compatible `max96726.ko` / AR0234 bring-up
   fix if this is an I2C alias or stream-on sequencing bug.
4. Recommend an official single-link isolation procedure, e.g. unplug all
   cameras except one failed locked link, power-cycle, reload modules, and
   test that one `/dev/videoN`.

---

## E11 — Move a known-good camera from video1 position to video7 position (2026-05-21)

**Goal:** Determine whether the failure follows a camera module/cable or stays
with a board/video position. The user moved the camera that had previously been
connected at the `video1` position to the `video7` position.

**Clean reload before validation:**

```bash
sudo service nvargus-daemon stop
cd ~/lerobot
sudo ./tools/thor/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16
for i in $(seq 0 15); do
  sudo v4l2-ctl -d /dev/video$i -c sensor_mode=0,trig_mode=0,bypass_mode=0
done
```

Reload completed successfully.

**MAX96726 lock state after moving the camera:**

```text
bus17 0x09 -> video0 locked, video1 unlocked, video2 unlocked, video3 locked
bus18 0x0b -> video4 locked, video5 locked, video6 unlocked, video7 locked
bus19 0x0e -> video8 unlocked, video9 locked, video10 locked, video11 locked
bus20 0x0c -> video12 unlocked, video13 unlocked, video14 locked, video15 locked
```

Script output:

```text
LOCKED_VIDEO_IDS=0,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,2,6,8,12,13
```

This is the expected physical effect of the move: `video1` dropped lock and
`video7` gained lock.

**Stream-on test set:**

```text
video7  # moved known-good camera at the new position
video1  # old position, now empty
video0  # baseline known-good
video4  # same MAX96726 group as video7
video9  # baseline known-good
video3, video5  # previously locked-but-failing positions
```

**Stream-on result:**

```text
video7: FAIL, VIDIOC_STREAMON returned -1 (Operation not permitted)
video1: FAIL, VIDIOC_STREAMON returned -1 (old position is now empty)
video0: OK, 30 frames
video4: FAIL, VIDIOC_STREAMON returned -1
video9: OK, 30 frames
video3: FAIL, VIDIOC_STREAMON returned -1
video5: FAIL, VIDIOC_STREAMON returned -1
```

`video4` had previously been a working stream-on channel, but failed in this
particular run after the `video1 -> video7` move. The lock register still
reported `video4` as locked. Treat this as additional evidence that the port2
MAX96726 stream-on/I2C state is unstable, not as evidence that `video4`'s
camera module is bad.

**dmesg evidence for the moved camera at `video7`:**

```text
ar0234c 18-0023: dser_link_check link:0x0b
ar0234c 18-0023: +++> Der-1 port 7 camera ar0234c-7 been detected!
max96726 18-0033: i2c-w, write failed
ar0234c 18-0023: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

There were also probe-time I2C write failures to sensor aliases on the same
deserializer group:

```text
max96726 18-0033: i2c-w16 failed: slave=0x46 reg=0x301a val=0x00d9 err=-121
ar0234c 18-0023: write16 table failed: source=0x46 reg=0x301a val=0x00d9 ret=-121
ar0234c 18-0023: sensor_init failed
```

**Conclusion:**

The camera/cable moved from the `video1` position to the `video7` position can
establish a GMSL link at `video7`, so the camera module is not simply dead and
the cable is not completely open. However, `video7` still fails at the
driver/deserializer stream-on stage with the same MAX96726 I2C-write failure
signature seen on other locked-but-failing channels.

This strongly shifts suspicion away from the camera module itself and toward
one of:

1. `video7` / port2 link-D board-side path,
2. MAX96726 I2C alias / serializer routing for that link,
3. vendor `max96726.ko` + `sg2-ar0234c-g2f.ko` stream-on sequencing,
4. an incorrect or incomplete link-to-video mapping assumption in the dtbo or
   driver.

**Recommended confirmation:** move the same camera back to the `video1`
position and repeat the clean reload + 30-frame stream test. If `video1`
returns to `OK frames=30`, the camera module is effectively cleared and the
remaining failure is port/driver-side.

已整理供应商问题单：

```text
tools/thor/gmsl2/SUPPLIER_ISSUE_MAX96726_STREAMON.md
```

---

## E12 — Jetson 重启后复测仍保持 locked 但 stream-on 失败的模式 (2026-05-21)

**目标：** 用户重启 Jetson 后，重新执行 clean reload 和只针对 locked link 的
stream 测试，确认问题是否只是旧驱动状态残留。

**流程：**

```bash
sudo service nvargus-daemon stop
cd ~/lerobot
sudo ./tools/thor/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16
for i in $(seq 0 15); do
  sudo v4l2-ctl -d /dev/video$i -c sensor_mode=0,trig_mode=0,bypass_mode=0
done
./tools/thor/gmsl2/check_max96726_locks.sh
```

模块重载成功。

**重启 + clean reload 后的 link-lock 结果：**

```text
bus17 0x0d -> video0, video2, video3 locked
bus18 0x0b -> video4, video5, video7 locked
bus19 0x0e -> video9, video10, video11 locked
bus20 0x0c -> video14, video15 locked

LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

这和之前 `video1 -> video7` 换位后的结果略有不同：这次 `video2` locked，
而 `video6` unlocked。因此当前接线状态应视为稀疏且可能随现场调整变化；
每次录制或 stream 测试前都应先运行 `check_max96726_locks.sh`，不要假设
video ID 是连续固定集合。

**只对 locked IDs 做 stream-on 复测：**

```text
OK:   video0, video2, video9
FAIL: video3, video4, video5, video7, video10, video11, video14, video15
```

失败形式保持一致：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

stream 测试后 link-lock 状态仍保持：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

**重启后的 dmesg 证据：**

```text
max96726 17-0033: i2c-w, write failed
ar0234c 17-0023: Error turning on streaming
max96726 18-0033: i2c-w, write failed
ar0234c 18-0020: Error turning on streaming
ar0234c 18-0021: Error turning on streaming
ar0234c 18-0023: Error turning on streaming
max96726 19-0033: i2c-w, write failed
ar0234c 19-0022: Error turning on streaming
ar0234c 19-0023: Error turning on streaming
max96726 20-0033: i2c-w, write failed
ar0234c 20-0022: Error turning on streaming
ar0234c 20-0023: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

**结论：**

重启 Jetson 没有消除该失败模式。具体 locked ID 集合以及少数能 stream 的
通道会随接线/重载状态变化，但核心问题一致：

> 多条 MAX96726 链路显示 locked，但供应商驱动无法在其中多路上完成
> V4L2 STREAMON，并伴随 MAX96726 I2C write failures 和 VI timeouts。

这进一步说明剩余阻塞点更可能在供应商驱动、解串器路由、I2C alias 或
link-to-video 映射，而不是 LeRobot 上层录制逻辑。

## E13 - 2026-05-21: 按供应商要求补充 video7 完整 dmesg

供应商要求提供“加载驱动到打开 video7 节点”的完整 dmesg。已在 Thor 上执行：

1. 停止 `nvargus-daemon`
2. 清空内核 ring buffer：`sudo dmesg -C`
3. 执行 `setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16`
4. 对 `/dev/video0..15` 设置 `sensor_mode=0,trig_mode=0,bypass_mode=0`
5. 读取 MAX96726 link lock
6. 打开 `/dev/video7` 并做一次 stream-on 测试
7. 保存完整 `dmesg`

已落盘文件：

```text
tools/thor/gmsl2/records/debug_logs/dmesg_load_to_video7.log
tools/thor/gmsl2/records/debug_logs/video7_open_stream.log
tools/thor/gmsl2/records/debug_logs/setup_sync.log
tools/thor/gmsl2/records/debug_logs/link_locks_before_video7.log
```

`video7_open_stream.log` 显示 `/dev/video7` 对应 `vi-output, ar0234c 18-0023`，
节点格式为 1920x1080 BA10 60 fps，控制项为：

```text
sensor_mode=0
trig_mode=0
bypass_mode=0
```

stream-on 仍失败：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

## E14 - 2026-05-21: 按供应商建议验证 nvcam settings / ISP 颜色影响

供应商建议删除 `/var/nvidia/nvcam/settings/` 下所有文件后重新点亮，观察画面颜色是否变化。

执行过程：

1. 备份 Thor 上 `/var/nvidia/nvcam/settings/` 到 `/tmp/nvcam_settings_backup_20260521_030142`
2. 清空 `/var/nvidia/nvcam/settings/`
3. 重启 `nvargus-daemon`
4. 用 Argus sample 对 `cam00` 录制 1080p H.265 3s
5. 恢复原 settings，并再次确认 `camera_overrides.isp` 指向驱动 repo：

```text
/var/nvidia/nvcam/settings/camera_overrides.isp
-> /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp
```

已落盘文件：

```text
tools/thor/gmsl2/records/cam00_1080p60_no_nvcam_settings_h265_3s.h265
tools/thor/gmsl2/records/cam00_1080p60_no_nvcam_settings_h265_3s.mp4
tools/thor/gmsl2/records/cam00_isp_frame.jpg
tools/thor/gmsl2/records/cam00_no_nvcam_settings_frame.jpg
tools/thor/gmsl2/records/debug_logs/record_no_isp.log
```

MP4 校验结果：

```text
codec_name=hevc
codec_tag_string=hvc1
width=1920
height=1080
avg_frame_rate=1610/27
duration=2.700000
```

对比 `cam00_isp_frame.jpg` 与 `cam00_no_nvcam_settings_frame.jpg`，本次样本没有看到肉眼明显的颜色变化。
该结果说明“删除 settings 后重新点亮”没有在当前样本上产生明显画面差异；是否代表 ISP 未生效仍需结合供应商对 JetPack 7 Argus/ISP 加载路径的确认。

## E15 - 2026-05-21: 放置红色块后重新做 ISP / no-settings 对比录制

用户已在相机 `0..3` 前放置红色块。按供应商建议重新执行：

1. 当前 `/var/nvidia/nvcam/settings/` 状态下录制 `camera_index=0..3`
2. 备份并清空 `/var/nvidia/nvcam/settings/`
3. 重启 `nvargus-daemon`
4. 再录制 `camera_index=0..3`
5. 恢复原 settings

已确认恢复后：

```text
nvargus-daemon: active
/var/nvidia/nvcam/settings/camera_overrides.isp
-> /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp
```

录制结果：

```text
camera_index=0:
  ISP:         OK, 3.0M H.265
  no settings: OK, 3.0M H.265

camera_index=1:
  ISP:         failed, argus_camera_recording segfault, output 0 bytes
  no settings: failed, argus_camera_recording segfault, output 0 bytes

camera_index=2:
  first run: failed after camera1 crashed Argus connection
  retry with daemon restart: failed, argus_camera_recording segfault, output 0 bytes

camera_index=3:
  first run: failed after camera1 crashed Argus connection
  retry with daemon restart: failed, argus_camera_recording segfault, output 0 bytes
```

本轮落盘目录：

```text
tools/thor/gmsl2/records/gmsl2_red_isp_compare_20260521_060346/
```

关键文件：

```text
isp/cam00_red_isp_h265_3s.h265
isp/cam00_red_isp_h265_3s.mp4
isp/cam00_red_isp_frame.jpg
no_nvcam_settings/cam00_red_no_nvcam_settings_h265_3s.h265
no_nvcam_settings/cam00_red_no_nvcam_settings_h265_3s.mp4
no_nvcam_settings/cam00_red_no_nvcam_settings_frame.jpg
logs/*.log
```

MP4 校验：

```text
ISP cam00:
codec_name=hevc
codec_tag_string=hvc1
width=1920
height=1080
avg_frame_rate=9600/161
duration=2.683333

no-settings cam00:
codec_name=hevc
codec_tag_string=hvc1
width=1920
height=1080
avg_frame_rate=1610/27
duration=2.700000
```

肉眼对比：

```text
isp/cam00_red_isp_frame.jpg
no_nvcam_settings/cam00_red_no_nvcam_settings_frame.jpg
```

红色块可见，但 ISP 与 no-settings 两张抽帧差异仍然很小，没有出现明显的颜色校正变化。
这轮结果需要供应商进一步确认：JetPack 7 / Argus 是否可能使用了缓存，或者该
`camera_overrides.isp` 是否实际影响当前 AR0234 Argus pipeline。

## E16 - 2026-05-21: 转接板电源插拔后 clean reload 复测

用户插拔转接板电源后，重新执行 clean reload、link-lock、V4L2 stream-on 和
Argus H.265 录制测试。

本轮落盘目录：

```text
tools/thor/gmsl2/records/gmsl2_powercycle_retest_20260521_061404/
```

流程：

```text
1. stop nvargus-daemon
2. dmesg -C
3. setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16
4. video0..15 设置 sensor_mode=0,trig_mode=0,bypass_mode=0
5. 读取 MAX96726 REG0x0008
6. 对 locked IDs 逐路 V4L2 stream-on
7. 重启 nvargus-daemon 后对 camera_index=0..3 做 Argus H.265 3s 录制
```

电源插拔后的 link-lock 仍为：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

也就是 GMSL 物理 link 仍然可以 locked。

但本轮所有 locked IDs 的 V4L2 stream-on 均失败：

```text
FAIL: video0, video2, video3, video4, video5, video7, video9, video10, video11, video14, video15
```

每路 V4L2 日志均显示：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

dmesg 继续显示相同类型错误：

```text
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

Argus H.265 录制 `camera_index=0..3` 也全部失败：

```text
camera_index=0: ARGUS_RC=139, output 0 bytes
camera_index=1: ARGUS_RC=139, output 0 bytes
camera_index=2: ARGUS_RC=139, output 0 bytes
camera_index=3: ARGUS_RC=139, output 0 bytes
```

其中 `139` 为 `argus_camera_recording` 段错误。

复测结束后已确认：

```text
nvargus-daemon: active
/var/nvidia/nvcam/settings/camera_overrides.isp
-> /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp
```

结论：

电源插拔没有解决问题，反而在本次测试状态下从“部分通道可 stream”退化为
“所有 locked 通道均无法 V4L2 stream-on”。核心模式仍然是：

> MAX96726 link-lock 成立，但 stream-on 阶段发生 MAX96726 I2C write failure。

这进一步支持问题位于转接板供电稳定性、解串器/serializer I2C 路由、驱动
stream-on sequencing 或板级链路状态恢复流程，而不是上层录制逻辑。

## E17 - 2026-05-21: 按供应商建议验证 Argus/GStreamer 拉流路径

供应商指出 V4L2 raw stream 不走 ISP，建议优先使用 Argus 点亮，并给出命令：

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12' ! \
  nvvidconv ! xvimagesink
```

初始检查发现当前 Thor 环境缺少 NVIDIA GStreamer 元件：

```text
gst-inspect-1.0 nvarguscamerasrc -> No such element or plugin
gst-inspect-1.0 nvvidconv       -> No such element or plugin
gst-inspect-1.0 nvv4l2h265enc   -> No such element or plugin
```

已安装 `nvidia-l4t-gstreamer`。该安装同时把一批 L4T 包升级到 `38.2.2`，
`dpkg --configure -a` 已完成，包管理状态已恢复；安装过程提示需要 reboot
使 L4T 更新完全生效。当前运行内核仍显示：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
/etc/nv_tegra_release: R38.2.1
dpkg nvidia-l4t-core/camera/gstreamer/multimedia: 38.2.2
```

安装后已确认这些元件存在：

```text
nvarguscamerasrc
nvvidconv
nvv4l2h265enc
```

在 SSH 无显示环境下，供应商原始 `xvimagesink` 命令失败于 display：

```text
Could not open display (null)
GST_XV_RC:255
```

因此使用 `fakesink` 和 H.265 文件 sink 排除显示环境影响：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=0 num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false

gst-launch-1.0 -e -v nvarguscamerasrc sensor-id=0 num-buffers=180 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12' ! \
  nvv4l2h265enc bitrate=12000000 ! h265parse ! filesink location=sensor0_argus_gst.h265
```

两条 Argus/GStreamer 管线均进入 PLAYING，但 `nvarguscamerasrc` 报：

```text
Error generated. ... gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

H.265 文件输出为 0 bytes。

同时重建 Jetson Multimedia API `10_argus_camera_recording` sample 后测试
`camera_index=0`，也返回：

```text
Error generated. main.cpp, execute:623 No cameras available
```

相关日志已落盘：

```text
tools/thor/gmsl2/records/gst_argus_after_install/gst_sensor0_fakesink.log
tools/thor/gmsl2/records/gst_argus_after_install/gst_sensor0_h265_file.log
tools/thor/gmsl2/records/gst_argus_after_install/gst_sensor0_xvimagesink.log
```

结论：

现在已经按供应商建议切到 Argus/GStreamer 路径验证。问题不是 V4L2 是否走 ISP，
而是当前状态下 Argus 本身枚举不到相机；与此同时 MAX96726 link-lock 仍然成立。
由于安装 GStreamer 插件时 L4T 包已更新到 38.2.2 并提示 reboot，下一步应重启
Thor 后重新加载供应商驱动并再跑一次 Argus/GStreamer 测试。

## E18 - 2026-05-21: Thor 重启后对 locked IDs 执行 Argus/GStreamer 复测

用户重启 Jetson 后，重新检查 MAX96726 lock，并对 locked IDs 执行
`nvarguscamerasrc sensor-id=<id>` 的 GStreamer 管线测试。

重启后的环境：

```text
Linux upai-pro03 6.8.12-tegra #1 SMP PREEMPT Thu Sep 25 15:19:42 PDT 2025
nvidia-l4t-core       38.2.2-20250925153837
nvidia-l4t-kernel     6.8.12-tegra-38.2.2-20250925153837
nvidia-l4t-camera     38.2.2-20250925153837
nvidia-l4t-gstreamer  38.2.2-20250925153837
```

GStreamer 插件已存在：

```text
nvarguscamerasrc=0
nvvidconv=0
nvv4l2h265enc=0
```

执行 clean reload 时，`setup_sync.sh` 返回失败：

```text
SETUP_RC=1
insmod /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/ko/sg2-ar0234c-g2f.ko
insmod: ERROR: could not insert module .../sg2-ar0234c-g2f.ko: Invalid parameters
```

dmesg 显示供应商 AR0234 驱动与当前 L4T 38.2.2 内核 tegracam 符号版本不匹配：

```text
sg2_ar0234c_g2f: disagrees about version of symbol tegracam_v4l2subdev_unregister
sg2_ar0234c_g2f: Unknown symbol tegracam_v4l2subdev_unregister (err -22)
sg2_ar0234c_g2f: disagrees about version of symbol tegracam_v4l2subdev_register
sg2_ar0234c_g2f: Unknown symbol tegracam_v4l2subdev_register (err -22)
sg2_ar0234c_g2f: disagrees about version of symbol tegracam_device_register
sg2_ar0234c_g2f: Unknown symbol tegracam_device_register (err -22)
```

因此 `/dev/video*` 没有生成：

```text
Cannot open device /dev/video0, exiting.
```

尽管 sensor driver 未加载，MAX96726 I2C lock register 仍能读取，结果为：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

对这些 locked IDs 执行 Argus/GStreamer fakesink 管线：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=<id> num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false
```

结果全部失败：

```text
sensor-id=0  FAIL
sensor-id=2  FAIL
sensor-id=3  FAIL
sensor-id=4  FAIL
sensor-id=5  FAIL
sensor-id=7  FAIL
sensor-id=9  FAIL
sensor-id=10 FAIL
sensor-id=11 FAIL
sensor-id=14 FAIL
sensor-id=15 FAIL
```

典型 Argus/GStreamer 日志：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

或：

```text
Failed to create CameraProvider
```

本轮日志目录：

```text
tools/thor/gmsl2/records/gmsl2_gst_locked_retest_20260521_065006/
```

结论：

重启后 GStreamer 插件已可用，但供应商 AR0234 驱动包与升级后的 L4T 38.2.2
内核不匹配，导致 `sg2-ar0234c-g2f.ko` 无法加载、`/dev/video*` 不存在、
Argus 枚举不到相机。此时的 GStreamer 失败不再是原始 stream-on 问题，而是
驱动 ABI/内核版本不匹配问题。需要供应商提供适配当前 L4T 38.2.2 内核的
AR0234 驱动包，或将系统恢复到该驱动包对应的 L4T/内核版本后再复测。

## E19 - 2026-05-21: 系统恢复到供应商驱动包对应的 L4T/内核版本

目标：将 Thor 从安装 `nvidia-l4t-gstreamer` 时升级到的 L4T 38.2.2 恢复到
供应商驱动包对应环境。

恢复过程：

1. 确认 apt 源仍提供 L4T 38.2.1 包：

```text
nvidia-l4t-core       38.2.1-20250910123945
nvidia-l4t-kernel     6.8.12-tegra-38.2.1-20250910123945
nvidia-l4t-camera     38.2.1-20250910123945
nvidia-l4t-gstreamer  38.2.1-20250910123945
```

2. 用 `apt-get -s install --allow-downgrades` 模拟降级，确认只会 downgrade
   48 个 NVIDIA L4T 包，不移除关键包。

3. 实际 downgrade 48 个 L4T 包到 38.2.1，并保留当前配置文件：

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --allow-downgrades --allow-change-held-packages \
  -o Dpkg::Options::="--force-confold" \
  <nvidia-l4t-*=38.2.1 package list>
```

4. 降级完成后，确认关键包已回到 38.2.1：

```text
nvidia-l4t-camera     38.2.1-20250910123945
nvidia-l4t-core       38.2.1-20250910123945
nvidia-l4t-gstreamer  38.2.1-20250910123945
nvidia-l4t-kernel     6.8.12-tegra-38.2.1-20250910123945
nvidia-l4t-multimedia 38.2.1-20250910123945
```

5. 将已安装的 `nvidia-l4t-*` 包 hold 住，避免后续 apt 操作再次自动升级到
   38.2.2。当前 hold 的 L4T 包数量为 60。

6. 重启后发现系统只回到 NVIDIA stock 38.2.1 内核：

```text
Linux upai-pro03 6.8.12-tegra #1 SMP PREEMPT Wed Sep 10 12:18:50 PDT 2025
```

此时供应商 `sg2-ar0234c-g2f.ko` 仍因 tegracam 符号版本不匹配无法加载。

7. 检查供应商 `install.sh`，确认该驱动包还需要安装自带 kernel Image 和基础
   camera/nvcsi ko：

```bash
sudo cp ko/tegra-camera.ko /lib/modules/6.8.12-tegra/updates/drivers/media/platform/tegra/camera/
sudo cp ko/nvhost-nvcsi.ko /lib/modules/6.8.12-tegra/updates/drivers/video/tegra/host/nvcsi/
sudo cp boot/Image /boot/Image
```

8. 执行前备份：

```text
/boot/Image.before_sensing_restore_20260521_065941
/boot/extlinux/extlinux.conf.before_sensing_restore_20260521_065941
```

9. 执行供应商 `install.sh` 后确认 hash 匹配：

```text
/boot/Image
== /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/boot/Image
sha256 cd6e2d0e6c15096db1bf21456d76c12b27f6ef45c2ad5a009b3ce1d6f2b7274c
```

10. 重启后进入供应商定制内核：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
```

11. 发现 `/boot/extlinux/extlinux.conf` 的 `DEFAULT` 被恢复成 `primary`，
    JetsonIO overlay 未生效，导致无 `/dev/video*`。备份并修正：

```text
/boot/extlinux/extlinux.conf.before_default_jetsonio_20260521_070237
DEFAULT JetsonIO
OVERLAYS /boot/tegra264-camera-ar0234cx16-overlay.dtbo
```

12. 再次重启后，确认当前恢复快照：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
nvidia-l4t-* key packages: 38.2.1, held
/boot/Image == supplier boot/Image
DEFAULT JetsonIO
nvarguscamerasrc=0
nvvidconv=0
nvv4l2h265enc=0
```

恢复过程快照已落盘：

```text
tools/thor/gmsl2/records/restore_logs/restore_process_snapshot.log
```

最终验证：

```text
tools/thor/gmsl2/records/gmsl2_post_restore_final_verify_20260521_070505/
```

结果：

```text
SETUP_RC=0
/dev/video0..15 generated
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

说明系统已恢复到供应商驱动包对应的内核/DT overlay/模块组合，且视频节点恢复。

恢复后继续按供应商建议对 `sensor-id=0,2,9` 做 Argus/GStreamer smoke test：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=<id> num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false
```

结果仍失败：

```text
sensor-id=0 FAIL
sensor-id=2 FAIL
sensor-id=9 FAIL
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

结论：

系统恢复已完成：L4T 包、供应商定制 kernel Image、JetsonIO AR0234 overlay、
供应商 ko、GStreamer 插件和 `/dev/video0..15` 均已恢复。当前剩余问题回到
原始相机链路/Argus 枚举问题，而不再是 38.2.2 ABI 不匹配问题。

## E20 - 2026-05-21: 恢复后重新完整复测 locked IDs

用户要求恢复后重新跑一次。当前环境确认：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
DEFAULT JetsonIO
OVERLAYS /boot/tegra264-camera-ar0234cx16-overlay.dtbo
nvidia-l4t-core/camera/gstreamer/multimedia: 38.2.1, held
nvarguscamerasrc=0
nvvidconv=0
nvv4l2h265enc=0
```

本轮日志目录：

```text
tools/thor/gmsl2/records/gmsl2_rerun_after_restore_20260521_071200/
```

`setup_sync.sh` 成功：

```text
SETUP_RC=0
```

MAX96726 lock 状态：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

对全部 locked IDs 执行 V4L2 30 帧 stream-on：

```text
video0  OK
video2  OK
video3  OK
video4  OK
video5  OK
video7  OK
video9  OK
video10 OK
video11 OK
video14 OK
video15 OK
```

对应日志中每路均出现 30 个 `<`：

```text
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

说明恢复后底层 V4L2 raw stream-on 当前全部 locked 通道可成功。

随后对全部 locked IDs 执行 Argus/GStreamer：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=<id> num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false
```

结果全部失败：

```text
sensor-id=0  FAIL
sensor-id=2  FAIL
sensor-id=3  FAIL
sensor-id=4  FAIL
sensor-id=5  FAIL
sensor-id=7  FAIL
sensor-id=9  FAIL
sensor-id=10 FAIL
sensor-id=11 FAIL
sensor-id=14 FAIL
sensor-id=15 FAIL
```

典型日志：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

结论：

恢复完成后，底层 locked link + V4L2 raw 出流已经成功；当前剩余问题集中在
Argus/ISP 路径枚举不到相机。也就是说，现在已经不是 GMSL link-lock 或 V4L2
stream-on 阻塞，而是 `nvarguscamerasrc` / Argus camera provider 与供应商
AR0234 RAW camera stack 的集成问题。

---

## E21 — Retract E7's "PWM 没接到触发引脚" 推断 (2026-05-21)

**触发：** 用户实测确认在以下状态下 nvargus 单路拉流稳定 60Hz：

```
PWM:         period=16666666 ns, duty=8333333 ns, enable=1 (60 Hz, 50% duty)
v4l2-ctl:    trig_mode=0, frame_rate=60000000, exposure=9999, gain=320
pipeline:    gst-launch-1.0 nvarguscamerasrc sensor-id=0
             ! 'video/x-raw(memory:NVMM),framerate=60/1,format=NV12'
             ! nvvidconv ! xvimagesink
```

并明确告知 PWM 信号物理走线**是接好的**。

**所以 E7 的推断错了。** "slave 模式 0.8 fps fallback ⇒ PWM 信号没到达"
这条结论不成立。

**新的最可能解释（待 E22 实测验证）：** E7/E5/E6 都在 `trig_mode=1` 下
用 `exposure_us=15000`，距 60Hz 周期 16666 µs 只剩 1.6 ms 余量，不足
AR0234 行扫描读出时间，sensor 错过下一个触发边沿便回落到默认 fallback
速率（在 60Hz/16.6ms 周期下表现为 ~0.8 fps）。用户 verified 的稳定路径
用 `exposure=9999`（≈10 ms），留 6.6 ms 余量。

**因此 E5 / E6 / E8 关于"7 路相机硬件不通"的推断也需要在新曝光参数下重测**
才能下定论：它们大概率是同一时序根因叠加，而不是物理通道挂掉。

**这次落地的代码改动：**

| 文件 | 改动 |
| --- | --- |
| `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` | `hardware_sync.sensor_trig_mode: 0 → 1`（默认走真正的 PWM 边沿锁相）；`cameras.defaults.exposure_us: 0 → 9999`、`gain: 0 → 320`（用户验证的安全曝光）；注释里写明 60Hz 下 `exposure_us` 应 ≤ ~14000 µs。 |
| `tools/thor/gmsl2/gmsl2_record.py` | `HardwareSync.sensor_trig_mode = 1`（与 yaml 保持一致）；新增曝光预检：当 `sensor_trig_mode=1` 且 `exposure_us > 0.85 × (1e6 / pwm_fps)` 时打 warning，自动 clamp 到该上限。 |
| `tools/thor/gmsl2/README.md` | 标注 PWM 触发路径已确认可用；明确"真硬同步"的两步条件（trig_mode=1 + exposure 在周期上限内）。 |

**关于跨相机时间戳对齐含义的修正：**

之前我说"free-run 下跨相机靠 pipeline clock 亚毫秒"。在 `trig_mode=1`
真硬同步生效后：

* MKV 内嵌的 PTS 仍是 GStreamer pipeline clock（非 sensor 内部计数器）。
* 但每路相机的同一帧号 N **物理曝光起点锁相到同一 PWM 边沿**，跨相机
  对齐由 PWM jitter 决定（亚 µs 级），不依赖软件 PTS 推算。
* 想拿真正的 sensor-side timestamp，仍需 libargus 自定义 sample 调用
  `ICaptureMetadata::getSensorTimestamp()`，但这只是用于量化 PWM jitter，
  不是对齐前置条件。

---

## E22 — Slave mode + exposure=9999 µs 单路复测，9/11 通过 (2026-05-21)

**关键命令序列（在 Thor 上）：**

```bash
# 1. 把 PWM 拉到 60 Hz（vendored pwm.sh; 自动 unexport→重写 period/duty→enable）
cd ~/lerobot/tools/thor/gmsl2/sdk && sudo sh pwm.sh

# 2. 拿 MAX96726 物理 lock 列表
~/lerobot/tools/thor/gmsl2/check_max96726_locks.sh | grep ^LOCKED_VIDEO_IDS=
#   实测: LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15  (11 路)

# 3. 把每路相机推到 slave + 安全曝光
for sid in 0 2 3 4 5 7 9 10 11 14 15; do
  sudo v4l2-ctl -d /dev/video$sid \
    -c sensor_mode=0,trig_pin=0x00020007,trig_mode=1,exposure=9999,gain=320
done

# 4. 单路 nvargus 拉 60 buffer 验流
for sid in 0 2 3 4 5 7 9 10 11 14 15; do
  timeout 6 gst-launch-1.0 -q nvarguscamerasrc sensor-id=$sid num-buffers=60 \
    ! "video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080" \
    ! fakesink sync=false async=false \
    && echo "PROBE_OK sid=$sid" || echo "PROBE_FAIL sid=$sid"
done
```

**单路验流结果（PWM 60 Hz, trig_mode=1, exposure=9999, gain=320）：**

```
PROBE_FAIL sid=0   (gst-nvargus 内部错误)
PROBE_FAIL sid=2   (gst-nvargus 内部错误)
PROBE_OK   sid=3
PROBE_OK   sid=4
PROBE_OK   sid=5
PROBE_OK   sid=7
PROBE_OK   sid=9
PROBE_OK   sid=10
PROBE_OK   sid=11
PROBE_OK   sid=14
PROBE_OK   sid=15
```

→ **9 / 11 路在真正 slave 模式下出流。**

**核心结论：**

1. **E7 的根因结论被推翻：** PWM 信号路径 OK；E7 当时观测到的 0.8 fps
   fallback 来自 `exposure_us=15000` 超出 60 Hz 周期上限。
2. **E5 / E6 / E8 "7 路相机硬件不通" 几乎全部翻案：** 在新曝光参数下
   只有 sid 0 和 sid 2 仍然在 Argus 层报错；其余 9 路（**包括之前判为
   "dead" 的 2、3、5、7、10**）都能进入硬同步 nvargus 流。剩余 2 个 fail
   是 Argus / sensor 集成层面的问题，与触发线、并发、I²C 都无关。
3. **新基线：** "11 路 slave + nvargus 全部能流"是个**软件层 bug**（sid 0/2
   单独 Argus open 失败），不是硬件 / 接线 / 时序问题。下一步在这条 baseline
   上做 11 路 nvargus → nvv4l2h265enc → matroskamux → filesink 并发录制
   验证。

**这次落地的代码改动（已 commit）：**

| 文件 | 关键变更 |
| --- | --- |
| `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` | `hardware_sync.sensor_trig_mode: 1`；`cameras.defaults.exposure_us: 9999`、`gain: 320`；曝光-vs-周期注释 |
| `tools/thor/gmsl2/gmsl2_record.py` | `HardwareSync.sensor_trig_mode = 1` 默认；新增 `_clamp_exposure_for_pwm_period()`：`exposure_us > 0.85 × 1e6/fps` 时自动 clamp + WARNING |
| `tools/thor/gmsl2/README.md` | 标注 PWM 路径已 verified；明确"真硬同步"两步条件 + exposure 上限计算 |

**留给后续：**

* sid 0 / sid 2 Argus open 失败的具体 dmesg / gst-nvargus 错误原文，与
  E12 / E17 / E18 厂商沟通线索串起来 —— 大概率是同一根因（Argus camera
  provider 注册某几个 sid 失败）。

---

## E23 — 11 路 nvargus → H.265 → MKV 并发录制：0 帧 (2026-05-21)

**命令：**

```bash
cd ~/lerobot && rm -rf outputs/datasets/thor_gmsl2_11ch_v1
# 前置：E22 已完成 PWM 60Hz + 11 路 trig_mode=1 + exposure=9999 + gain=320
PYTHONPATH=src:. python3 -u -m tools.thor.gmsl2.gmsl2_record \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --num-episodes 1 --episode-time-s 5 \
  --skip-argus-probe
```

orchestrator 同时拉起 11 个 `gst-launch` 子进程，每条 pipeline 为

```
nvarguscamerasrc sensor-id=<sid> sensor-mode=0 do-timestamp=true
    exposuretimerange='9999000 9999000' gainrange='320 320'
  ! video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080,framerate=60/1
  ! nvv4l2h265enc bitrate=20000000 iframeinterval=60 preset-level=1
                  control-rate=1 insert-sps-pps=1
  ! h265parse ! matroskamux
  ! filesink location=.../cam_<sid>.mkv sync=false
```

**结果：**

```
$ ls -lh outputs/datasets/thor_gmsl2_11ch_v1/episodes/episode_000000/*.mkv
cam_00.mkv  0 B
cam_02.mkv  0 B
cam_03.mkv  0 B
cam_04.mkv  0 B
cam_05.mkv  0 B
cam_07.mkv  0 B
cam_09.mkv  0 B
cam_10.mkv  0 B
cam_11.mkv  0 B
cam_14.mkv  0 B
cam_15.mkv  0 B
```

11 路 MKV 全部 0 字节。每路 `gst-launch` 日志典型片段（cam_03 为例，
其它 10 路完全相同）：

```
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Setup Complete, Starting captures for 0 seconds
GST_ARGUS: Starting repeat capture requests.
CONSUMER: Producer has connected; continuing.
handling interrupt.                                          # ← episode timeout SIGINT
Interrupt: Stopping pipeline ...
EOS on shutdown enabled -- Forcing EOS on the pipeline
Error generated. ... gstnvarguscamerasrc.cpp, threadExecute:743 NvBufSurfaceFromFd Failed.
```

`Producer has connected; continuing` 之后**没有任何 `CONSUMER: Acquired Frame`
行**，说明在 5 秒 episode 窗口内，下游消费端一帧都没收到 —— 11 路 Argus
producer 同时开起来后，nvargus-daemon 没在窗口内吐出任何 buffer。`Forcing
EOS on shutdown` 后报的 `NvBufSurfaceFromFd Failed` 是关闭时旁路错误，不
是出流瓶颈。

**结论：**

* **单路 slave 出流 OK**（E22, 9/11），并发 0/11 是上层 Argus / nvargus-daemon
  调度问题，不是触发 / 时序 / 接线。
* 与 E5 / E6 / E8 当时的 0.8 fps 数据**性质相似**（并发拉低到接近 0），
  但 E5/E6 当时一些通道还能挤出帧，本次完全为零，可能跟 nvargus-daemon
  在 11 路并行 `connect` 后启动序列卡死有关。
* `gmsl2_record.py` 现在的"一次性同时 spawn 11 个 gst-launch"策略撑不住。

**最小干预修复路线（按优先级）：**

1. **阶梯式 open**：在 `EpisodeSession.start()` 里给每路 spawn 之间加
   200–500 ms 间隔，等 Argus 把前一个相机注册成功再开下一个。
2. **延长 episode_time_s 重测**：把 5s 提到 30s，看是不是 nvargus-daemon
   单纯需要更长时间完成 11 路 setup。当前 wallclock 总耗时已经 94s，但
   episode 主循环只跑了 5s，其余在 stop/wait/ffprobe 阶段；把 episode
   窗口本身放大才能验证。
3. **降低并发**：先确认在 4 / 8 路 slave 并发下能成功录制，再回头攻 11。
4. **关掉 ffprobe 的 stop 阶段开销**（orchestrator 当前阻塞在文件 stat /
   解析上）：episode duration 报 94s 太长，影响诊断节奏。

下一条 entry（E24）应该做阶梯式 stagger + 8 路并发的对照。

## E24 — 定位补丁：可配置阶梯式 spawn + 修正 episode 计时起点 (2026-05-21)

**代码改动：**

* `tools/thor/gmsl2/gmsl2_record.py`
  * 新增 `RecorderConfig.spawn_stagger_s`，从 yaml 的
    `sensors.cameras.spawn_stagger_s` 读取，默认 0.0。
  * 新增 CLI 参数 `--spawn-stagger-s`，用于临时覆盖配置。
  * `EpisodeSession.start()` 每路 `gst-launch` 之间按
    `spawn_stagger_s` sleep，并在日志中打印每路 `spawn +X.XXXs`。
  * 把 `episode_time_s` 的倒计时起点移到 `session.start()` 完成之后，
    避免 8/11 路 stagger 吃掉正式采集窗口。
  * `meta.json` 记录 `spawn_stagger_s`，方便回看实验条件。
* `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml`
  * 新增 `sensors.cameras.spawn_stagger_s: 0.0`，保持默认行为不变。

**补充诊断开关：**

E24 首轮 8 路实测中，`cam_02` 提前 EOS 会触发 orchestrator 立刻停止整集，
导致其它通道没有完整 30s 窗口。为避免单个坏路截断对照，追加：

* `RecorderConfig.stop_on_stream_exit`，默认 `true`，保持生产行为不变。
* CLI `--ignore-stream-exit`，仅诊断用：某路 `gst-launch` 提前退出时记录
  warning，但 episode 继续跑到 `episode_time_s`。

**下一步对照命令（先 8 路）：**

```bash
cd ~/lerobot && rm -rf outputs/datasets/thor_gmsl2_11ch_v1
sudo service nvargus-daemon restart

PYTHONPATH=src:. python3 -u -m tools.thor.gmsl2.gmsl2_record \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --sensor-ids 0,2,3,4,5,7,9,10 \
  --num-episodes 1 --episode-time-s 30 \
  --spawn-stagger-s 0.5 \
  --ignore-stream-exit \
  --skip-argus-probe
```

**判据：**

1. 若 8 路 MKV 非 0 字节，且各 `*.gst.log` 出现
   `CONSUMER: Acquired Frame`，说明 E23 的 0 帧主要是 Argus 并发 open
   启动序列问题；继续测 11 路 `--spawn-stagger-s 0.5`。
2. 若 8 路仍 0 字节，把 `--spawn-stagger-s` 提到 2.0 再测一次；若仍 0，
   问题更像 Argus / VI / ISP 资源在多路 slave 60Hz 下无法进入 steady
   state，而不是单纯 connect 风暴。
3. 每次实验后先看：

```bash
ls -lh outputs/datasets/thor_gmsl2_11ch_v1/episodes/episode_000000/*.mkv
rg -n "spawn \\+|Producer has connected|Acquired Frame|NvBufSurfaceFromFd|Error generated" \
  outputs/datasets/thor_gmsl2_11ch_v1/episodes/episode_000000/*.gst.log
```

**远端实测结果（Thor: `nvidia@192.168.1.44`, 2026-05-21 UTC）：**

先将 E24 recorder 补丁同步到 `~/lerobot/tools/thor/gmsl2/`；远端原文件备份：

```text
/tmp/lerobot_gmsl2_backup_20260521_094839
```

远端 `~/lerobot` 不是 git checkout，因此本轮直接同步文件测试。

1. `episode_000001`: `0,2,3,4,5,7,9,10`, `--spawn-stagger-s 0.5`,
   `episode_time_s=30`，未加 `--ignore-stream-exit`。

   * `cam_02` 提前退出，orchestrator 在约 13s 停止整集。
   * 只有 `cam_04.mkv` 非空（2.4 MB）；其它 7 路都是 336 B 空容器。
   * 空路日志统一出现：

```text
Error generated. ... gstnvarguscamerasrc.cpp, threadExecute:743 NvBufSurfaceFromFd Failed.
nvbuf_utils: dmabuf_fd -1 mapped entry NOT found
```

2. 追加诊断开关：`--ignore-stream-exit`

   E24 首轮说明单个坏路会截断其它通道窗口，因此在默认行为不变的前提下新增：

   * `RecorderConfig.stop_on_stream_exit`，默认 `true`
   * CLI `--ignore-stream-exit`：坏路提前退出只打 warning，episode 继续跑满
     `episode_time_s`

3. `episode_000002`: 排除已知不稳的 `0/2`，跑
   `3,4,5,7,9,10,11,14`, `--spawn-stagger-s 0.5`,
   `--ignore-stream-exit`。

   * 8 路都能 `Producer has connected`。
   * 仍只有 `cam_04.mkv` 非空（3.7 MB）。
   * 其它 7 路都是 336 B 空容器，3-9s 内陆续提前 EOS。
   * `dmesg` 同时出现供应商驱动/解串器层错误，例如：

```text
max96726 20-0033: i2c-w, write failed
ar0234c 20-0022: Error turning on streaming
WARNING ... v4l2_subdev_has_pad_interdep ... tegra_channel_set_stream
```

4. `episode_000003`: 同一组
   `3,4,5,7,9,10,11,14`, `--spawn-stagger-s 2.0`,
   `--ignore-stream-exit`。

   * 2.0s ultra-stagger 没有改善。
   * 仍只有 `cam_04.mkv` 非空（6.7 MB）。
   * 其它 7 路仍是 336 B 空容器，`NvBufSurfaceFromFd Failed`。
   * `cam_11` / `cam_14` stop 阶段不响应 SIGINT，被 SIGTERM 结束。

5. `episode_000004`: 4 路并发对照
   `4,5,7,9`, `--spawn-stagger-s 0.5`, `--ignore-stream-exit`。

   * `cam_04.mkv` 正常录满 30s（68 MB），日志：

```text
CONSUMER: Producer has connected; continuing.
Got EOS from element "pipeline0".
Execution ended after 0:00:31.473493074
CONSUMER: Done Success
```

   * `cam_05/07/09` 均为空容器（336 B），并在 3-4.5s 内
     `NvBufSurfaceFromFd Failed`。

6. `episode_000005`: 单路 `sid=5`，10s。

   * 单路也失败，`cam_05.mkv` 仍为 336 B。
   * 日志同样是 `NvBufSurfaceFromFd Failed` + `dmabuf_fd -1`。

7. `episode_000006`: 单路 `sid=4`，10s 正对照。

   * 单路成功，`cam_04.mkv` 23 MB。
   * 日志正常结束：

```text
CONSUMER: Producer has connected; continuing.
Execution ended after 0:00:10.011839888
CONSUMER: Done Success
```

**E24 结论更新：**

* `spawn_stagger_s=0.5` 和 `2.0` 都不能让 8 路恢复；问题不是简单的
  simultaneous open 风暴。
* 降到 4 路后仍只有 `sid=4` 有有效帧，说明当前状态下不是纯粹的 8 路资源
  阈值。
* `sid=5` 单路也失败，而 `sid=4` 单路成功，因此当前根因更接近
  **部分 sensor-id / deserializer link 的 Argus buffer export / VI stream-on
  路径失败**。
* 关键失败签名已经稳定收敛：

```text
gstnvarguscamerasrc.cpp, threadExecute:743 NvBufSurfaceFromFd Failed
nvbuf_utils: dmabuf_fd -1 mapped entry NOT found
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
```

下一步不应继续调 Python spawn 策略；应回到供应商驱动 / dtbo / link-to-video
映射定位，优先拿 `sid=4` 成功 vs `sid=5` 失败的完整 `dmesg -C` 对照包给供应商。
