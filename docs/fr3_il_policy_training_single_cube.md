# FR3 IL policy training

This repo can train ACT and Diffusion Policy through the standard
`lerobot_train` pipeline. For `dataset_test/single_cube2_20260429_165325`,
use the helper below instead of training directly on the raw dataset:

For real-hardware inference after training, see
`docs/fr3_il_policy_inference_single_cube.md`.

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras observation.images.cam_1,observation.images.cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 1000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

The script first creates a derived LeRobot dataset view under
`outputs/datasets/<job_name>`, then writes a generated train config and launches
`lerobot.scripts.lerobot_train`. The source dataset is not modified.

`--dataset-root` supports two input layouts:

- Single dataset root: the path itself contains `meta/info.json`, `data/`,
  `videos/`, and `meta/episodes/`. Example:
  `dataset_test/water_pouring/water_20260514_162838`.
- Multi-dataset parent root: the path does not contain `meta/info.json`, but
  its direct child directories are separate LeRobot datasets. Example:
  `dataset_test/water_pouring`, which contains
  `water_20260514_162838/` and `water_20260514_164925/`.

For the multi-dataset layout, the helper sorts child dataset directories by
name, validates that the selected cameras, state keys, action key or derived
action file, and appended action selectors are compatible, then writes one
training view. The generated view reindexes global `episode_index`, `index`,
and `task_index`, merges `meta/tasks.parquet` and `meta/episodes`, and maps
each source dataset's data/video files to distinct `file-xxx` ids so LeRobot's
loader can read all episodes as one dataset.

Generated files:

- `outputs/datasets/<job_name>/train_config.generated.json`: exact LeRobot
  training config passed to `lerobot_train`.
- `outputs/datasets/<job_name>/inference_config.generated.yaml`: deployment
  config that points back to the matching dataset view, expected checkpoint,
  selected camera keys, image resize shape, low-dimensional observation keys,
  action layout, gripper action append, and conservative runtime safety
  defaults.
- `outputs/datasets/<job_name>/meta/il_view_manifest.json`: dataset-view
  manifest with the source dataset roots and observation/action contract.

Use the generated inference YAML as the default entrypoint for real-hardware
deployment. It avoids manually repeating training choices during inference:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --preview \
  --max-steps 20
```

CLI arguments override the YAML, so you can keep the training-aligned defaults
while changing hardware details such as `--robot-ip`, `--gripper-port`, or
`--camera-config`.

## Dataset contract

The sample dataset has:

- cameras: `observation.images.cam_0` ... `cam_7`,
  `observation.images.pika_left_opencv`,
  `observation.images.pika_right_opencv`,
  `observation.images.pika_left_realsense`,
  `observation.images.pika_right_realsense`
- low-dimensional observations:
  `observation.state` with 7D EE pose
  `[x, y, z, qx, qy, qz, qw]`
- extra low-dimensional observations:
  `observation.state_raw` with 2D handheld gripper widths
- action:
  `action` with 7D target EE pose. The helper defaults to appending
  `observation.state_raw:handheld_gripper.pika_left.width_mm` as the final
  gripper action dimension, so the default policy output is 8D:
  `[ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, pika_left.width_mm]`.

ACT and Diffusion Policy in this codebase consume fixed keys:
`observation.state`, selected `observation.images.*`, and `action`. If you pass
multiple `--state-keys`, the helper concatenates them into a new
`observation.state` in the dataset view. A state key may also select one
dimension with `feature_key:name` or `feature_key:index`, for example
`observation.state_raw:handheld_gripper.pika_left.width_mm`. If you pass
`--action-append-selectors`, the helper appends those selected dimensions to the
training `action`.

## Argument Reference

Dataset and output arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dataset-root` | `dataset_test/single_cube2_20260429_165325` | Source LeRobot dataset root, or a parent directory whose direct children are separate LeRobot dataset roots. The script does not modify source datasets. |
| `--repo-id` | `single_cube2_il_view` | Repo id stored in the generated dataset view metadata. It is local-only unless you later upload it. |
| `--view-root` | `outputs/datasets/<job_name>` | Output path for the derived training dataset view. |
| `--output-dir` | `outputs/train/<job_name>` | Training output directory. Checkpoints and offline WandB logs are written here. |
| `--job-name` | auto-generated | Name used for the training run and default output paths. |
| `--overwrite-view` | off | Delete and recreate `--view-root` if it already exists. Use only when you intentionally want to replace the generated view. |
| `--copy-videos` | off | Copy selected videos into the view. By default videos are symlinked to avoid duplicating large files. |
| `--prepare-only` | off | Build the dataset view and train config, then stop before training. Useful for inspection. |
| `--smoke` | off | Load the generated view with the policy's temporal indexing and print tensor shapes. |
| `--resume` | off | Resume LeRobot training from an existing checkpoint instead of starting from scratch. The helper keeps the existing view unless `--overwrite-view` is also set. |
| `--resume-checkpoint` | `--output-dir/checkpoints/last` | Checkpoint directory to resume from. May point to `checkpoints/last`, a numbered checkpoint such as `checkpoints/030000`, or its `pretrained_model` subdirectory. |

Observation arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--policy` | `act` | Policy family. Supported values are `act` and `diffusion`. |
| `--cameras` | `observation.images.cam_1,observation.images.cam_3` | Comma-separated camera features to keep as visual observations. You may write `cam_1,cam_3` as shorthand for `observation.images.cam_1,observation.images.cam_3`. |
| `--state-keys` | `observation.state` | Comma-separated low-dimensional observation keys to concatenate into the generated `observation.state`. Each item may be a full feature key or a selector `feature_key:name` / `feature_key:index`. Use `none` for image-only ACT. |
| `--image-resize-shape` | unset | Optional shared image resize as `H,W`, for example `360,640`. When set, the generated view metadata uses that camera shape, the training dataloader resizes decoded video frames to it, and the real-robot runtime resizes live camera frames to the same policy shape. |
| `--use-imagenet-stats` | true | Let LeRobot use ImageNet visual normalization stats for selected cameras. Disable with `--no-use-imagenet-stats`. |
| `--video-backend` | `pyav` | Video decoder backend used by the LeRobot dataset loader. |

Action arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--action-key` | `action` | Base action feature from the source parquet. For this dataset it is 7D next-frame EE pose. |
| `--action-npy` | none | Optional `.npy` file to use as the base action instead of `--action-key`. Relative paths are resolved under each source dataset root. Absolute paths are allowed only for single-dataset input, because they are ambiguous for multi-dataset input. |
| `--use-derived-action` | off | Shortcut for `derived/hikon_cube_tracking_in_robot_base/action.npy` under each source dataset root. |
| `--action-append-selectors` | `observation.state_raw:handheld_gripper.pika_left.width_mm` | Extra feature dimensions appended to the generated `action`. Selector format is `feature_key:name` or `feature_key:index`. |
| `--action-append-names` | `gripper` | Names assigned to appended action dimensions. The default `gripper` is recognized by the real-robot inference runtime. |
| `--action-append-shift` | `1` | Temporal shift for appended action dimensions within each episode. `1` means use the next frame, matching this dataset's EE `action[t] == state[t+1]`; `0` means current frame. |

ACT arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--act-chunk-size` | `30` | Number of future actions supervised and predicted per ACT query. |
| `--act-n-action-steps` | `30` | Number of predicted chunk actions consumed before the next policy query. Must be <= `--act-chunk-size`. |
| `--act-lr` | `1e-5` | ACT optimizer learning rate. |
| `--act-lr-backbone` | `1e-5` | Vision backbone learning rate saved into the ACT config. |
| `--act-pretrained-backbone-weights` | `ResNet18_Weights.IMAGENET1K_V1` | Torchvision ResNet18 initialization for ACT image encoder. |

Diffusion Policy arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dp-n-obs-steps` | `2` | Number of observation timesteps loaded for each policy sample. |
| `--dp-horizon` | `16` | Action sequence horizon used by the diffusion model. Must be compatible with the model downsampling factor. |
| `--dp-n-action-steps` | `8` | Number of generated actions used per inference query. |
| `--dp-resize-shape` | `224,224` | DP-internal image resize as `H,W`. Prefer `--image-resize-shape` for real-robot work because it is shared by training metadata, dataloader, and inference. If `--image-resize-shape` is set, this DP-internal resize is disabled to avoid double resizing. |
| `--dp-lr` | `1e-4` | Diffusion Policy optimizer learning rate. |

Training arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | Torch device. Use `cuda`, `cpu`, or `auto`. |
| `--use-amp` | off | Enable policy AMP flag. |
| `--batch-size` | `4` | Dataloader batch size per process. |
| `--num-workers` | `4` | Dataloader worker count. Use `0` for debugging. |
| `--steps` | `1000` | Number of optimizer updates. |
| `--log-freq` | `20` | Console and scalar WandB logging interval in training steps. |
| `--save-freq` | `500` | Checkpoint save interval. A `checkpoints/last` symlink is updated on save. |
| `--seed` | `1000` | Random seed for training and image-log step sampling. |
| `--tolerance-s` | `1e-3` | Timestamp tolerance for loading temporally indexed video frames. |
| `--lr-scheduler` | `none` | Optional LR scheduler. Supported values are `none` and `cosine_decay_with_warmup`. Intended for new training runs; resume keeps the scheduler saved in the checkpoint config. |
| `--lr-warmup-steps` | `0` | Warmup length for `cosine_decay_with_warmup`. LR ramps up to the policy optimizer LR during these optimizer updates. |
| `--lr-decay-steps` | `--steps` | Decay horizon for `cosine_decay_with_warmup`. This is the total step index over which LR decays, not an interval. |
| `--lr-decay-final-lr` | `0.1 * policy_lr` | Final LR for `cosine_decay_with_warmup`. For ACT, `policy_lr` is `--act-lr`; for DP, it is `--dp-lr`. |

WandB arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--wandb` | off | Enable WandB logging. |
| `--wandb-mode` | unset | WandB mode: `online`, `offline`, or `disabled`. Use `offline` for local logging and later sync. |
| `--wandb-project` | `lerobot` | WandB project name. |
| `--wandb-entity` | unset | Optional WandB user or team entity. |
| `--wandb-log-images-n-steps` | `0` | Number of random training steps at which to log raw observation images. `0` disables image logging. |
| `--wandb-log-images-n-samples` | `2` | Number of batch samples to log per selected image-log step. |

## Recommended first runs

Prepare and train on a parent directory containing multiple dataset roots:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/water_pouring \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

Use one specific dataset under that parent when you do not want to merge:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/water_pouring/water_20260514_162838 \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

For GMSL2 data, training does not need a different policy path. Use the camera
feature suffixes recorded in the dataset, for example:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/<gmsl2_collection> \
  --policy act \
  --cameras gmsl2_front,gmsl2_wrist \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:corenetic_gripper.distance_m \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

The important rule is that the infer camera config later must contain matching
keys under `robot.cameras`, here `gmsl2_front` and `gmsl2_wrist`. If an older
dataset still stores the vendor BOX SDK sensor key as `box_gripper.distance_m`,
use that exact selector instead; the infer runtime recognizes both names.

ACT smoke/overfit:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --act-chunk-size 30 \
  --act-n-action-steps 30 \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

Diffusion Policy smoke/overfit:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy diffusion \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --dp-n-obs-steps 2 \
  --dp-horizon 16 \
  --dp-n-action-steps 8 \
  --steps 2000 \
  --batch-size 8 \
  --device cuda \
  --smoke \
  --overwrite-view
```

Use the derived tracking action instead of the parquet `action` column:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --use-derived-action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --overwrite-view
```

Include only one Pika gripper width as proprioception:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state,observation.state_raw:handheld_gripper.pika_left.width_mm \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --overwrite-view
```

That produces an 8D `observation.state`:
`[ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, observation.state_raw.handheld_gripper.pika_left.width_mm]`.

For your current right-arm/right-camera dataset, the equivalent command is:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/pick_place_v1_20260529_152103 \
  --policy act \
  --cameras pika_right_opencv,pika_right_realsense \
  --state-keys observation.state.right,observation.state_raw:handheld_gripper.pika_left.width_mm \
  --image-resize-shape 360,640 \
  --action-key action.right \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --job-name pick_place_act_pika_right_opencv_realsense_proprio_gripper1d \
  --act-chunk-size 50 \
  --act-n-action-steps 50 \
  --steps 60000 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda \
  --wandb \
  --wandb-mode offline \
  --wandb-project box-act \
  --wandb-log-images-n-steps 20 \
  --wandb-log-images-n-samples 2 \
  --lr-scheduler cosine_decay_with_warmup \
  --lr-warmup-steps 1000 \
  --lr-decay-steps 60000 \
  --lr-decay-final-lr 1e-6 \
  --act-lr 1e-5 \
  --overwrite-view
```

Image-only ACT, with no low-dimensional observation state:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys none \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --smoke \
  --overwrite-view
```

This removes `observation.state` from the policy inputs while keeping the 8D
action target `[ee pose, gripper]`. It is currently supported for ACT only. The
current Diffusion Policy implementation expects `observation.state`, so do not
use `--state-keys none` with `--policy diffusion` without changing the DP model
path.

Resume training:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/updated/single_cube2_20260429_165325 \
  --policy act \
  --cameras cam_1,cam_2,pika_right_opencv \
  --state-keys none \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --job-name single_cube2_updated_act_cam1_cam2_pika_right_imgonly \
  --output-dir outputs/train/single_cube2_updated_act_cam1_cam2_pika_right_imgonly \
  --resume \
  --resume-checkpoint outputs/train/single_cube2_updated_act_cam1_cam2_pika_right_imgonly/checkpoints/last \
  --steps 50000
```

`--steps` is the new total target step count, not the number of additional
steps. For example, if the checkpoint is at step 30000, `--steps 50000`
continues for 20000 more optimizer updates. If you omit `--steps`, LeRobot uses
the total step count saved in the checkpoint config; if the checkpoint is
already at that count, no more training happens.

Resume loads the checkpoint's saved training config, policy weights, optimizer
state, scheduler state if present, RNG state, and WandB run id. CLI observation
or model arguments are not used to change the resumed model contract; change
them only when starting a new run. Use `--overwrite-view` together with
`--resume` only when you intentionally need to rebuild the same dataset view,
for example after a helper-side data repair.

For large image batches, DataLoader workers can fail with shared-memory errors
such as `unable to allocate shared memory(shm)`. On resume you may override
runtime dataloader/logging fields without changing the policy contract:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --job-name single_cube2_updated_act_cam1_cam2_pika_right_imgonly \
  --resume \
  --steps 60000 \
  --num-workers 0
```

If this is too slow, try `--num-workers 1` or `--num-workers 2`. If it still
fails, reduce `--batch-size`, for example `--batch-size 16`.

ACT with cosine LR decay from the beginning:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --dataset-root dataset_test/updated/single_cube2_20260429_165325 \
  --policy act \
  --cameras cam_1,cam_2,pika_right_opencv \
  --state-keys none \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --job-name single_cube2_updated_act_cam1_cam2_pika_right_imgonly_decay \
  --act-chunk-size 50 \
  --act-n-action-steps 50 \
  --act-lr 1e-5 \
  --steps 60000 \
  --lr-scheduler cosine_decay_with_warmup \
  --lr-warmup-steps 1000 \
  --lr-decay-steps 60000 \
  --lr-decay-final-lr 1e-6 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --wandb \
  --wandb-mode online \
  --wandb-project box-act \
  --overwrite-view
```

Do not use these LR scheduler flags to change the schedule of an existing
constant-LR checkpoint during `--resume`. That requires a scheduler-state
migration and is intentionally not implemented in the helper.

Offline WandB with raw observation-image snapshots:

```bash
uv run python tools/fr3/fr3_train_il_policy.py \
  --policy act \
  --cameras cam_1,cam_3 \
  --state-keys observation.state \
  --image-resize-shape 360,640 \
  --action-key action \
  --action-append-selectors observation.state_raw:handheld_gripper.pika_left.width_mm \
  --steps 2000 \
  --batch-size 4 \
  --device cuda \
  --wandb \
  --wandb-mode offline \
  --wandb-project fr3-il \
  --wandb-log-images-n-steps 8 \
  --wandb-log-images-n-samples 2 \
  --overwrite-view
```

The image logger samples random training steps and logs every selected camera in
the raw dataloader batch. For ACT, each camera entry is `[B, C, H, W]`; for
Diffusion Policy it is `[B, T, C, H, W]`, so the log includes each observation
history step.

## Upload Offline WandB Logs

When `--wandb --wandb-mode offline` is used, training does not upload logs during
the run. Instead, WandB writes an offline run directory under the training output
directory:

```text
outputs/train/<job_name>/wandb/offline-run-*
```

After training finishes, upload the offline run to the WandB website:

```bash
uv run wandb login

uv run wandb sync outputs/train/<job_name>/wandb/offline-run-*
```

If the run was launched with:

```bash
--wandb-project fr3-il
```

then it will appear in the `fr3-il` project on the WandB website. If you need to
upload into a specific entity/team, include it during training:

```bash
--wandb-entity <your_wandb_entity_or_team>
```

Useful local checks before syncing:

```bash
ls outputs/train/<job_name>/wandb
find outputs/train/<job_name>/wandb -maxdepth 1 -type d -name 'offline-run-*'
```

To sync exactly one run, pass that directory explicitly:

```bash
uv run wandb sync outputs/train/<job_name>/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
```

After `wandb sync` finishes, the terminal prints the WandB run URL. Open that URL
to inspect scalar losses, learning rate, gradient norm, and the sampled
`train/observation_images/*` panels logged from the selected cameras.

## Verified smoke results

The helper has been smoke-tested on
`dataset_test/single_cube2_20260429_165325` with `cam_1,cam_3` and the default
left-gripper action append:

- ACT view: `observation.state` is `(7,)`, `action` is `(30, 8)`, and each
  selected camera is `(3, 720, 1280)`.
- ACT resized view with `--image-resize-shape 360,640`: each selected camera is
  `(3, 360, 640)`.
- Diffusion Policy view: `observation.state` is `(2, 7)`, `action` is
  `(16, 8)`, and each selected camera is `(2, 3, 720, 1280)`.
- Diffusion Policy resized view with `--image-resize-shape 360,640`: each
  selected camera is `(2, 3, 360, 640)`, and `policy.resize_shape` is disabled
  to avoid double resizing.
- The generated action feature names are
  `[ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, gripper]`.
- Image-only ACT view: `observation.state` is absent, `action` is `(30, 8)`,
  and each selected camera is `(3, 720, 1280)`.
- Updated single-cube image-only ACT with `cam_1,cam_2,pika_right_opencv`,
  `--act-chunk-size 50`, and `--image-resize-shape 360,640`: the helper repairs
  91 non-finite source action values, checks 2250 episode-tail temporal action
  queries, runs an ACT forward smoke, and a 3-step `batch_size=32`,
  `num_workers=4` train check passes.
- The generated `meta/il_view_manifest.json` records the selected cameras,
  state keys, action append selectors, action append shift, state dimension,
  action dimension, and source dataset roots.
- Multi-dataset preparation has been checked on `dataset_test/water_pouring`
  with `cam_1,cam_3`: the generated view contains 54 episodes and 17201 frames,
  maps the second source dataset to `data/chunk-000/file-001.parquet` and
  `videos/<camera>/chunk-000/file-001.mp4`, and `LeRobotDataset` can read
  samples from both the first source dataset and the second source dataset
  boundary at global index 9088.
- Multi-dataset `--use-derived-action` has been checked with the same
  `dataset_test/water_pouring` parent root. The helper resolves
  `derived/hikon_cube_tracking_in_robot_base/action.npy` under each child
  dataset root before concatenating the generated view.

## Practical notes

- Start with two stable third-person cameras. Add wrist/RealSense views only
  after loss and dataloader speed look sane.
- Prefer setting `--image-resize-shape` for real-robot policies. Raw capture can
  stay high-resolution, but the training view and inference runtime should share
  a smaller policy resolution such as `360,640` or `224,224`.
- `--state-keys none` means no low-dimensional observation is fed to the policy.
  The generated view still records `source_dataset_root` for single-dataset
  input and `source_dataset_roots` for both single- and multi-dataset input in
  `meta/il_view_manifest.json`.
- When `--dataset-root` points to a multi-dataset parent, only direct children
  that contain `meta/info.json` are included. Nested grandchildren are not
  searched. Keep child directory names stable, because they define merge order.
- Multi-dataset input requires all selected features to be schema-compatible
  across child datasets. If a selected camera, state key, action key, or action
  append selector is missing or has a mismatched dimension, preparation fails
  before writing the training view.
- For multi-dataset `--action-npy`, pass a relative path that exists under each
  child dataset root, or use `--use-derived-action`. Absolute `--action-npy` is
  intentionally rejected for multi-dataset input because one absolute file
  cannot unambiguously label multiple source datasets.
- `--action-append-shift` defaults to `1`, because this dataset's EE `action[t]`
  equals the next frame's EE pose. The appended gripper width therefore also
  uses the next frame by default. Use `--action-append-shift 0` if you want the
  current frame width instead.
- The helper repairs non-finite values in the generated `action` by
  per-episode forward-fill then backfill, independently for each action
  dimension. This handles source rows whose EE action is `nan` while preserving
  valid gripper widths. If an episode has no valid value for an action
  dimension, preparation fails instead of silently writing an unusable view.
- `--smoke` now checks more than tensor shapes for ACT: it scans episode-tail
  temporal action chunks and runs one ACT `forward` pass, so image-only ACT and
  chunk-boundary issues are caught before a long training run.
- `--action-append-names` defaults to `gripper`, which is the name expected by
  the current real-robot inference runtime. If you append a different action
  dimension, either give it a runtime-recognized name or update the runtime
  decoder.
- `--dp-resize-shape` is still available for DP-only experiments, but it is less
  useful for deployment because ACT and the real-robot runtime do not read it.
  Use `--image-resize-shape` when you want training and inference to share the
  same image size contract.
- For this dataset size, first verify overfit behavior with 1k-2k steps before
  scaling to 50k-100k steps.
- Offline action loss is only a sanity metric. A useful policy still needs a
  real-robot replay/evaluation pass with the same observation and action
  contract.
