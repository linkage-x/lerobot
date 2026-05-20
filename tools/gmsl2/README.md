# Thor GMSL2 (SG16A-AGTH-G3Y-A1) 11-channel capture

This folder hosts the helpers needed to run the LeRobot data-collection stack on a
Jetson Thor wired to a SENSING SG16A-AGTH-G3Y-A1 GMSL adapter. The current
configuration targets eleven SG2-AR0234C-G2F GMSL2 cameras with hardware-synchronous
trigger driven by the Jetson's own PWM.

## What is here

| File | Purpose |
| --- | --- |
| `setup_sync.sh` | Replacement for the SDK's `load_modules.sh` + `pwm.sh`. Loads the GMSL2 + AR0234 kernel modules, boosts VI/ISP/NVCSI/EMC clocks, programs `pwmchip4/pwm0` to the chosen sync frequency, and sets every channel to slave-trigger via `v4l2-ctl`. |
| `thor_gmsl2_11ch_example.yaml` | Recorder config with 11 GMSL2 cameras (no grippers / tactiles). Used by `tools/handheld/handheld_record.py` and by `tools/data_collection_gui/gateway.py`. |

The actual Python driver lives at `src/lerobot/cameras/gmsl2/`. Registered draccus
key: `type: gmsl2`.

## Hardware sync, in one paragraph

The board exposes a trigger pin (`trig_pin=0x00020007` in the SDK). When a sensor
is put into `trig_mode=1`, its frame start is gated by a falling/rising edge on
that pin. By feeding the pin with the Jetson PWM (`pwmchip4/pwm0`) we get every
channel locked to the same frame clock -- no master-camera daisy-chain required.

`setup_sync.sh` programs the PWM to `1/<fps>` seconds with 50% duty (60 Hz by
default -- matches the AR0234 dtbo, which hard-codes
`min_framerate=max_framerate=60000000` with `framerate_factor=1000000`).

## Bring-up

The driver and config are pushed from the host to the Thor over rsync (see
"Deploying to Thor" below). On the Thor:

```bash
# 1. One-time per boot: load modules, boost clocks, program PWM, put cameras
#    into slave trigger mode.
sudo ./tools/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 11

# 2. (Optional) start the Argus daemon explicitly. If you skip this, GStreamer
#    will start it on demand.
sudo service nvargus-daemon restart

# 3. Sanity check a single channel.
GST_DEBUG=2 gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 sensor-mode=0 \
  ! 'video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080,framerate=60/1' \
  ! nvvidconv ! 'video/x-raw,format=BGRx' ! fakesink -v
```

If step 3 prints a stable `current-fps:` close to 60 you are good to go.

## Running the recorder / GUI on Thor

```bash
# Headless recorder (no GUI):
cd ~/lerobot
PYTHONPATH=src:. python -m tools.handheld.handheld_record \
  --config-path tools/gmsl2/thor_gmsl2_11ch_example.yaml

# GUI gateway:
PYTHONPATH=src:. python -m tools.data_collection_gui.gateway \
  --config-path tools/gmsl2/thor_gmsl2_11ch_example.yaml \
  --datasets-root outputs/datasets \
  --port 8765

# Vite frontend (run on the Thor or proxy from a workstation):
cd tools/data_collection_gui/frontend && npm install && npm run dev
```

See `tools/data_collection_gui/frontend/README.md` for the full GUI walkthrough.

## Deploying to Thor

The repo lives on the host; the Thor needs the same tree. Recommended:

```bash
# from the host repo root
rsync -avh --delete \
  --exclude '.git/' --exclude 'outputs/' --exclude 'node_modules/' \
  --exclude '__pycache__/' --exclude '.venv/' --exclude 'dist/' \
  ./ nvidia@192.168.1.44:~/lerobot/
```

Then on the Thor make sure the Python deps are installed once:

```bash
sudo apt install -y \
  python3-gi python3-gst-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly v4l-utils \
  nvidia-l4t-jetson-multimedia-api
```

LeRobot itself can be installed with `uv pip install -e .` inside a Python 3.12
venv (Thor's stock python3 works).

## Troubleshooting

* `nvarguscamerasrc` plugin not found -> install
  `nvidia-l4t-jetson-multimedia-api`, then restart the shell so the Argus
  GStreamer plugin is picked up by `gst-inspect-1.0`.
* `v4l2-ctl` rejects `sensor_mode=0,trig_pin=...,trig_mode=...` -> the AR0234
  kernel module isn't loaded; re-run `setup_sync.sh`.
* All cameras free-run at slightly different rates -> `--master-id` is set or
  PWM is off. Re-run `setup_sync.sh` without `--master-id`, then verify
  `cat /sys/class/pwm/pwmchip4/pwm0/enable` prints `1`.
* `connect()` warmup never produces a frame -> check the bus log line emitted by
  the driver (`GStreamer ERROR:`) for an Argus daemon error and inspect
  `/var/log/syslog` for CSI / TPG complaints.
