# Thor depth experiment: FFS or LingBot-Depth

The selector is `run_thor_depth_experiment.py`. `ffs` invokes the existing
calibrated Fast-FoundationStereo workflow. `lingbot-depth` processes `cam_03`
and `cam_13` separately. By default it uses each camera's FFS depth map as its
input prior. FFS creates that prior from the stereo pair; LingBot-Depth then
refines each rectified view separately. The selector does not fuse the two
cameras' tracking results.

## Upstream dependency

Initialize the [Robbyant LingBot-Depth submodule](https://github.com/Robbyant/lingbot-depth)
at `third_party/lingbot-depth` (or pass `--lingbot-repo` for another checkout).
Install it in the interpreter named by `--track-python`:

```bash
git submodule update --init third_party/lingbot-depth
/home/dante/miniconda3/envs/sam3/bin/python -m pip install -e third_party/lingbot-depth
```

The v0.5 checkpoint is downloaded by the upstream loader on first inference,
unless `--lingbot-model` names a local checkpoint. Use a CUDA GPU. The model
weights, checkout, and generated results are separate from this experiment's
source files.
For one-frame-at-a-time inference, the runner uses PyTorch's scaled dot product
attention when xFormers is unavailable. No xFormers installation is required.

## Run one episode

From the LeRobot repository root:

```bash
python tools/track4world/run_thor_depth_experiment.py \
  --depth-backend lingbot-depth --lingbot-input ffs \
  --dataset-root outputs/datasets/thor_gmsl2_10ch_v1_20261006_171735 \
  --episodes 1 --check-only

python tools/track4world/run_thor_depth_experiment.py \
  --depth-backend lingbot-depth --lingbot-input ffs \
  --dataset-root outputs/datasets/thor_gmsl2_10ch_v1_20261006_171735 \
  --episodes 1 --prepare-only
```

Remove `--prepare-only` to run the existing SAM3 and Track4World stages. The
runner invokes those stages separately for each camera; outputs appear under
`third_party/Track4World/results/<dataset>_lingbot_depth/episode_000001/`.
The runner reuses existing FFS geometry for the episode when present. If it is
missing, it first runs the existing FFS preparation.
Each camera has its own refined `depth_m.npy`, input `depth_input_m.npy`, `ensemble_points.npy`,
`ensemble_valid.npy`, and `manifest.json`. The per-camera reports are
`sam3_selected_cam_03_index.html` and `sam3_selected_cam_13_index.html`.
Each camera also has a `cam_03_depth_previews/` or `cam_13_depth_previews/`
folder containing sampled RGB/input depth/refined depth comparison images.
The internal scene folder retains a `stereo_` prefix because the existing
Track4World reader selects Thor scenes by that folder pattern. The LingBot
stage itself does not perform stereo estimation; its default input is the FFS
stereo result.

To run FFS through the same selector, use `--depth-backend ffs`. FFS still
uses the calibrated stereo pair to estimate depth, then the selector produces
separate `sam3_selected_ffs_cam_03` and `sam3_selected_ffs_cam_13` tracking
results without a fused result. The original FFS entry point remains available
for its existing workflow.

## Input depth and scale

The [upstream README](https://github.com/Robbyant/lingbot-depth#quick-start)
shows `depth_in=None`, but the [released model implementation](https://github.com/Robbyant/lingbot-depth/blob/main/mdm/model/v2.py)
requires a depth tensor. The default `--lingbot-input ffs` converts each
camera's FFS robot-base point map back to rectified camera Z, preserving pixel
alignment with its FFS RGB video. The refined depth still needs validation
against a measured reference.

Use `--lingbot-input zero` for an **experimental RGB-only trial**. This code
path has no upstream demonstration or established metric accuracy. Treat its
`depth_m.npy` values and robot-base points as provisional, even when plausible.

To use separately captured depth sources, supply *independent* initial depth
maps for each camera. The directory must contain one float32 `.npy` array per
camera and episode, in meters, aligned with that camera's 8 fps, 576×320 RGB
clip **before undistortion**:

```text
<input-depth-dir>/episode_000001/cam_03.npy  # [frames, 320, 576]
<input-depth-dir>/episode_000001/cam_13.npy  # [frames, 320, 576]
```

Then pass `--lingbot-input npy --input-depth-dir <input-depth-dir>`. Invalid
depth can be zero or NaN. The runner undistorts each depth map with its own
RGB camera calibration. Check predicted scale against measured references
before using its 3D tracks as metric measurements.
