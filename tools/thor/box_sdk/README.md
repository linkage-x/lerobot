# tools/thor/box_sdk

Vendored ARM (`aarch64`, JetPack 6+) release bundle of the BOX 采集板 SDK
plus a thin LeRobot-side wrapper. Used by the GMSL2 data-collection rig: the
Thor box hosts the 11 GMSL2 cameras (see `../gmsl2/`) and the BOX MCU board
that aggregates the gripper, IMU, trigger, 6D force, and two Paxini touch
sensors over UDP/15000.

## Layout

| Path | Source |
| --- | --- |
| `python/box_collection_sdk-0.1.0-py3-none-any.whl` | vendor delivery (ARM `release_bundle_v2_arm.tar.gz`) |
| `lib/lib*.so*` | vendor delivery — runtime libs loaded via `setup_env.sh` |
| `share/monte_gripper.urdf` | vendor delivery — referenced by `BOX_SDK_URDF` |
| `setup_env.sh` | vendor delivery — exports `LD_LIBRARY_PATH`/`BOX_SDK_URDF` |
| `demo.py` | vendor delivery — minimal start/set_mode/read loop |
| `README_DEPLOY.md` | vendor delivery — original install notes |
| `box_client.py` | LeRobot wrapper: dataclass config, `SensorCache` decoder, polling thread |

The wheel ships its own copy of `libbox_controller.so` (and binds `box_sdk`
to it via `importlib.resources`). The extra `lib/*.so` files in this folder
are the run-time dependencies (pinocchio + URDF parser + gripper kinematics)
that `libbox_controller.so` `dlopen`s -- they need to be on
`LD_LIBRARY_PATH`, which is exactly what `setup_env.sh` does.

## One-time host preparation

Per `BOX 采集板 SDK 需求整理.docx §4`:

```bash
sudo apt update
sudo apt install -y libeigen3-dev liburdfdom-dev
bash tools/thor/box_sdk/install_compat_links.sh
```

`liburdfdom-dev` pulls in `liburdfdom` and `tinyxml2`, the libraries that
`libgripper_kinematics.so` and `libpinocchio_parsers.so` `dlopen` at runtime.
`install_compat_links.sh` then drops two compat symlinks into
`tools/thor/box_sdk/lib/` so the SDK's hard-coded sonames
(`libtinyxml2.so.9`, `liburdfdom_model.so.3.0`) resolve to the JetPack 6
(`.so.10`, `.so.4.0`) system libraries. The script is idempotent and
detects aarch64 vs x86_64 automatically.

Without these, the wheel imports fine but `Box()` construction fails with
`OSError: libtinyxml2.so.9: cannot open shared object file` or similar.

## Per-session environment

```bash
source tools/thor/box_sdk/setup_env.sh
python3 -m pip install --force-reinstall tools/thor/box_sdk/python/box_collection_sdk-*.whl
```

The script prepends `tools/thor/box_sdk/lib/` (and ROS humble, if present)
to `LD_LIBRARY_PATH`, and exports `BOX_SDK_URDF=<this dir>/share/monte_gripper.urdf`.

## Smoke test

```bash
source tools/thor/box_sdk/setup_env.sh
python3 tools/thor/box_sdk/demo.py 5000 5000 192.168.2.60
```

(args: local bind port / remote port / remote IP).

## LeRobot integration

The GUI gateway and recorder use `box_client.BoxClient` rather than the
upstream wheel directly. It:

* normalizes the YAML config block (`box_collection:` in
  `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml`),
* runs the SDK poll loop in a background thread,
* decodes `SensorCache` (a `ctypes.Structure`) into JSON-friendly dicts
  so the gateway can pass it through HTTP and the recorder can write it
  into the dataset metadata,
* auto-detects which of the optional sensors are actually publishing (the
  rig may have 0–2 Paxini pads attached at any time),
* degrades gracefully on dev hosts where the wheel is not installed.

See `box_client.py` for the API. The dataclass / decoder layer is unit-tested
in `tests/scripts/test_thor_box_client.py` using a fake `box_sdk` stub, so
the wheel does not need to be installed to run the test suite.

## API surface from the vendor

The wheel itself is opaque -- runtime symbols extracted from
`libbox_controller.so`:

```
box_create / box_destroy / box_start / box_stop
box_get_box_sensor_data           # fills a SensorCache snapshot
box_get_mode / box_set_mode       # 0 = collection (trigger), 1 = control
box_get_gripper_pos               # blocking read of current grip in cm
box_set_clamp_pos                 # commanded clamp opening in meters
box_set_trigger_zero              # operator-zero the trigger travel
box_set_packet_observer           # raw decoded packet callback
error_message_c                   # rc -> human string
```

UDP ports are fixed to 15000 in both directions (`BOX 采集板 SDK 需求整理 §5.1`).
