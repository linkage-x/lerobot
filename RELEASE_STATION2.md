# Station 2 release (thor2 handover)

This tree is a release cut for thor2 (hostname `upai-thor2`), which is moving to a second lab.
It records the 4 GMSL2 cameras with hardware sync and the BOX 采集板 sensors through the
data-collection GUI. Deploy it as a maintainer: see "Deploying" below.

## What is not in this release

The originating lab's calibration and robot-arm configuration were removed. They describe
that lab's cameras, table, carts, BOXes and FR3 arms and are wrong on any other rig:

| Removed | What it was |
| --- | --- |
| `tools/thor/gmsl2/world/*.json` | world frame, registration edges, table datum, cart references |
| `tools/thor/gmsl2/camera_intrinsics_registry.json`, `camera_identity_expected.json`, `camera_serial_map.yaml` | per-serial camera intrinsics and port map of the other lab's cameras |
| `tools/thor/gmsl2/box_time_delays.json`, `cart_tags.json` | BOX timing calibration, cart AprilTag layout |
| `tools/thor/gmsl2/embodiments/*.json`, `tools/fr3/fr3_arms.json`, `tools/fr3/desk_templates/`, `tools/fr3/replay_cube_pose_in_robot_base.thor.right.yaml` | FR3 arm registry, embodiments, Desk payload, replay config |
| `third_party/opencv_kalibr`: marker→TCP calibrations, `*_tracking_in_robot_base_thor.yaml`, FR3 calibration/replay/eval configs with arm addresses, the RealSense camera-in-base result under `realsense_base_extrinsics_calibration/results/` | tool and tracking calibration, camera extrinsics, arm configs |
| `docs/rig0818_vs_hybrid_carrier_v1/` | a simulation study built on the originating lab's camera poses and trajectories |

`tools/thor/box_devices.json` is kept as an empty registry: add this station's BOXes there once
they are characterised. Recording does not need any of the removed files; the recorder and the
GUI warn that there is no world frame / production calibration and carry on. EE-trajectory
generation, marker tracking, extrinsics and arm replay need this station's own calibration first.

## Station files

- `run/targets/thor2.json`: the deploy target. Set `address` to thor2's wired address on your
  network (it is `CHANGE-ME`). `hostname` is checked over ssh before every deploy.
- `tools/thor/gmsl2/thor_gmsl2_station2_4ch.yaml`: recorder config. Cameras are auto-detected
  from the locked MAX96726 links (sensor ids 0/5/10/14 today), BOXes are discovered on the
  subnet (`bind_ip: 0.0.0.0`), laser tracker off, datasets under
  `outputs/datasets/thor2_gmsl2_box_v1`.

## Deploying

From a clean checkout of this tree on a development machine (git, python3, Node.js LTS, ssh
key login to thor2):

```bash
python3 run/targets.py list                 # validates run/targets/*.json
git status                                  # must be clean: commit target/config edits first
bash run/deploy.sh thor2 --maintainer       # rsync to ~/lerobot, restart the gateway, start the GUI
```

One-time on thor2 after the first sync (details in `tools/thor/DEPLOYMENT.md` §1-§3):

```bash
bash ~/lerobot/tools/thor/box_sdk/install_compat_links.sh
python3 -m pip install --user --break-system-packages --force-reinstall \
  ~/lerobot/tools/thor/box_sdk/python/box_collection_sdk-0.1.0-py3-none-any.whl
```

After every boot, load the camera drivers with the vendor pack (the copy of the AR0234 driver in
`tools/thor/gmsl2/sdk/ko/` is the unpatched original and fails at stream-on):

```bash
sudo ~/lerobot/tools/thor/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16
```

BOX network: firmware v4.1+ takes a DHCP address; plug the BOXes into the same LAN as thor2's
wired port. Only BOXes with the old fixed-address firmware need `"box_net_addr": "192.168.2.45/24"`
in the target file (`tools/thor/DEPLOYMENT.md` §6 describes that old setup).

## Tests that fail in this release, by design

Run as in `AGENTS.md` §5. With the arm registry and calibration removed, these fail and are
expected to:

- `tests/scripts/test_run_targets.py`: three tests that check the originating lab's `thor1` target.
- `tests/scripts/test_data_collection_gui_gateway.py`: 37 tests of FR3 pivot / replay / gripper
  teleop / marker→TCP / world-edge features, which need `fr3_arms.json` or the removed tracking
  configs. Its arm-registry fixture falls back to an empty registry so the other tests still run
  (2026-11-12: 459 passed, 40 failed across the gateway, BOX client and targets test files).
- Tests under `third_party/opencv_kalibr/metrology/tests/` that read the removed tracking configs.

`tests/scripts/test_thor_box_client.py` and the rest of the gateway tests pass.
