# Vendored SENSING SG16A_AGTH_G3Y_A1 SDK assets

The subset of the SENSING SG16A-AGTH-G3Y-A1 driver pack that we need on the
Jetson, copied into this repository so the recorder has zero dependence on
`~/Desktop/SG16A_AGTH_G3Y_A1/` being present.

Only the files relevant to the **SG2-AR0234C-G2F GMSL2 camera** path are
checked in. Modules for the other supported sensor families (IMX715 / 735 /
577 / ISX028) are intentionally omitted -- pull them from the upstream pack
if you ever wire those in.

## What is here

| File | Purpose | Size |
| --- | --- | --- |
| `pwm.sh` | Arms `pwmchip4/pwm0` at 60 Hz (50% duty). Replaces the SDK's pwm.sh, which ships at 25 Hz. Set `FPS=N sudo -E sh pwm.sh` to override. | ~1 KB |
| `boost_clock.sh` | Locks VI / ISP / NVCSI / EMC clocks to max. Verbatim copy from SDK. | ~700 B |
| `camera_overrides.isp` | NVIDIA Argus ISP override; verbatim copy from SDK. Symlinked into `/var/nvidia/nvcam/settings/` by `tools/thor/gmsl2/setup_sync.sh`. | 109 KB |
| `ko/max96726.ko` | GMSL2 deserializer kernel module (handles the 4 deserializers on the SG16A board). | 67 KB |
| `ko/pwm-gpio.ko` | Tegra PWM GPIO driver used by `pwm.sh`. | 45 KB |
| `ko/sg2-ar0234c-g2f.ko` | AR0234 sensor + V4L2 driver. | 90 KB |
| `dtb/tegra264-camera-ar0234cx16-overlay.dtbo` | Device-tree overlay that wires 16 AR0234 channels under the SG16A profile. | 53 KB |

Kernel-version-tied modules that the SDK's `install.sh` deploys system-wide
(`tegra-camera.ko`, `nvhost-nvcsi.ko`) are **not** vendored: they ship at
~1.4 MB combined and are tightly coupled to the running kernel. Run the
SDK's `install.sh` once per kernel; subsequent boots only need the assets
listed above.

## Kernel / JetPack provenance

These files were captured from the SENSING SG16A_AGTH_G3Y_A1 driver pack on
2026-05-21 on `nvidia@192.168.1.44` (JetPack 7.0 / L4T R38.2.1, kernel
`6.8.12-tegra`). Re-vendor when the SDK is refreshed or the kernel changes.

The `.ko` files are aarch64 ELF objects and only load on a JP7 / L4T R38.2
Thor. They will not work on Orin, Xavier, or any kernel other than
`6.8.12-tegra`. The dtbo is similarly tied to the SG16A_AGTH_G3Y_A1 carrier
board.

## How the rest of the repo uses these

* `tools/thor/gmsl2/setup_sync.sh` — cold-boot helper. Unloads stale modules,
  insmods the three `.ko` files from `ko/`, runs `boost_clock.sh`, then
  programs PWM via `pwm.sh`, and finally sets `trig_mode=1` on every
  `/dev/videoN`.
* `tools/thor/gmsl2/sdk/pwm.sh` — invoked at every session start by
  `tools/thor/gmsl2/gmsl2_record.py` when `hardware_sync.enabled: true`. This is
  the only step required between fresh boots once `setup_sync.sh` has been
  run once.
* `tools/thor/gmsl2/gmsl2_record.py` — default `hardware_sync.sdk_dir` is
  `tools/thor/gmsl2/sdk/` so no external paths are referenced.
