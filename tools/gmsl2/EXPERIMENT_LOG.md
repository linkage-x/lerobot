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
| `tools/gmsl2/setup_sync.sh` | Runtime equivalent of `load_modules.sh` + `pwm.sh` with hardware-sync defaults. Loads modules, boosts clocks, sets PWM `pwmchip4/pwm0`, puts every channel in slave trigger mode. |
| `tools/gmsl2/thor_gmsl2_11ch_example.yaml` | 11-channel handheld recorder config. No grippers, no tactiles. |
| `tools/gmsl2/README.md` | Bring-up / deploy / troubleshooting notes. |

**Verification on host:**
- `python3 -c "from lerobot.cameras.gmsl2 import Gmsl2CameraConfig; ..."` — config
  loads, draccus registers `gmsl2`.
- Full `draccus.decode(HandheldRecordingConfig, yaml.safe_load(...))` on
  `tools/gmsl2/thor_gmsl2_11ch_example.yaml` produces 11 `Gmsl2CameraConfig`
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
   tools/gmsl2/thor_gmsl2_11ch_example.yaml ...` per
   `tools/data_collection_gui/frontend/README.md`.
6. **High-fps debayer path.** Software `cv2.cvtColor` is enough for a single
   camera at 60 Hz. With 11 channels we will either need:
   * `nvarguscamerasrc` once it lands on JetPack 7, or
   * a VPI / CUDA debayer kernel called from Python, or
   * record raw Bayer to disk and debayer offline during dataset processing.
