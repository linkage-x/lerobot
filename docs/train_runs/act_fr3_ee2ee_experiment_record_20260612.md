# ACT FR3 ee2ee comparison experiment - 2026-06-12

## Purpose

Train ACT on the same FR3 pick-place ee2ee dataset used for pi05+LoRA, then compare offline
first-action and action-chunk quality.

This run is specifically non-DAS:

- No DAS tactile inputs.
- No `das_start`.
- Current robot start is the FR3/Pika default start used by the dataset.
- Dataset action frame is absolute EE in `target_ee`/`pika_gripper_ee` convention.

## Dataset

- Path: `outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011`
- Episodes: 20
- Frames: 8400
- FPS: 30
- State: 16D
  - `ee.x/y/z`
  - `prev_cmd.ee.x/y/z`
  - `ee.qx/qy/qz/qw`
  - `prev_cmd.ee.qx/qy/qz/qw`
  - `gripper.pos`
  - `prev_cmd.gripper.pos`
- Action: 8D
  - `ee.x/y/z`
  - `ee.qx/qy/qz/qw`
  - `gripper.pos`
- Images:
  - `observation.images.ee`
  - `observation.images.side`
  - `observation.images.front`

## Current robot state confirmation

Read-only joint state was checked with:

```bash
cd /home/hph/Code/lerobot
python3 tools/fr3/fr3_print_joints.py --robot-ip=192.168.1.206
```

Observed:

```text
Joint angles (rad): [0.002372, -0.771331, -0.001412, -2.372943, 0.002452, 1.572661, 0.794594]
Joint angles (deg): [0.136, -44.194, -0.081, -135.96, 0.14, 90.107, 45.527]
```

This is the default FR3/Pika start region, not a DAS-specific start.

## Training script

Script:

```bash
scripts/train_act_fr3_ee2ee_20260612.sh
```

Default settings:

- Policy: ACT
- Steps: 60000
- Batch size: 8
- Chunk size: 100
- Action steps: 100
- Backbone: ResNet18
- Backbone weights: `ResNet18_Weights.IMAGENET1K_V1`
- LR: `1e-5`
- Backbone LR: `1e-5`
- Weight decay: `1e-4`
- Save frequency: 5000
- Eval frequency: 0
- WandB: disabled

The ResNet18 checkpoint exists locally at:

```text
/home/hph/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

so training should not require external network access.

## Smoke command

```bash
cd /home/hph/Code/lerobot
RUN_ID=act_smoke_$(date +%Y%m%d_%H%M%S) \
STEPS=2 \
SAVE_CHECKPOINT=false \
LOG_FREQ=1 \
bash scripts/train_act_fr3_ee2ee_20260612.sh
```

## Full training command

```bash
cd /home/hph/Code/lerobot
RUN_ID=20260612_1032_default_start \
STEPS=60000 \
SAVE_FREQ=5000 \
BATCH_SIZE=8 \
bash scripts/train_act_fr3_ee2ee_20260612.sh
```

Expected output:

```text
outputs/train/act_fr3_ee2ee_20260612_1032_default_start
scripts/train_logs/act_fr3_ee2ee_20260612_1032_default_start.log
```

## Offline evaluation plan

Use the same offline action quality evaluator:

```bash
cd /home/hph/Code/lerobot
docker run --rm --gpus all --ipc=host --network host --user root \
  -v /home/hph/Code/lerobot:/workspace \
  -v /home/hph/.cache/huggingface:/root/.cache/huggingface \
  -v /home/hph/.cache/torch:/root/.cache/torch \
  -v /home/hph/.cache/triton:/root/.cache/triton \
  -v /home/tele/Models:/models:ro \
  -v /home/tele/Models:/data/model:ro \
  -w /workspace \
  -e PYTHONPATH=/workspace/src:/workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e WANDB_MODE=disabled \
  lerobot-user:local \
  python scripts/eval_pi05_lora_fr3_offline.py \
    --checkpoint outputs/train/act_fr3_ee2ee_20260612_1032_default_start/checkpoints/<step> \
    --start-frames-only \
    --max-samples 20
```

This evaluator is policy-agnostic for policies that expose `predict_action_chunk`.
ACT satisfies that requirement.

Pass criteria before any real preview:

- `first_current_pos_mm max < 30 mm`; target `< 5-10 mm`.
- `pred_gt_pos_mm p95 < 40 mm`; target `< 10-20 mm`.
- `chunk_max_step_pos_mm` should ideally be below the runtime continuity threshold around `5 mm`.

## Notes

- Do not run any real execution command from this experiment until offline start-frame gates pass.
- Real preview, when reached, should use `tools/fr3/fr3_infer_real.py`, `--gripper-backend=pika`,
  `--no-move-to-das-start`, and `--preview`.

## Actual run update - 2026-06-12 10:36 CST

Smoke result:

```text
Run: act_fr3_ee2ee_act_smoke_20260612_103551
Status: PASS
step:1 loss:79.617
step:2 loss:69.953
```

The smoke run verified dataset loading, ACT policy construction, ResNet18 cached backbone weights,
and optimizer/training loop execution.

Formal run launched:

```text
RUN_ID=20260612_1036_default_start
Output: outputs/train/act_fr3_ee2ee_20260612_1036_default_start
Log: scripts/train_logs/act_fr3_ee2ee_20260612_1036_default_start.log
Launch log: scripts/train_logs/act_fr3_ee2ee_20260612_1036_default_start.launch.log
Remote launch PID: 1006964
Training python PID observed: 1007048
GPU memory during early training: about 7.2 GB / 24 GB
Throughput during early training: about 9.4 step/s
```

Early training log:

```text
step:100 loss:9.753
step:200 loss:3.741
step:400 loss:2.789
step:500 loss:2.544
step:600 loss:2.341
```


## 2026-06-12 ACT 060000 real-input preflight

Checkpoint: `outputs/train/act_fr3_ee2ee_20260612_1036_default_start/checkpoints/060000`

Offline eval final result from watcher:

- `first_current_pos_mm mean=13.11 p95=15.75 max=16.72`
- `first_current_rot_deg mean=0.12 p95=0.21 max=0.59`
- `chunk_max_step_pos_mm mean=5.54 p95=7.87 max=8.03`
- `[PASS] offline action quality gates passed`

Real-input preview-only preflight notes:

- Initial attempts failed before policy output because `fr3_infer_real_runtime.py` defaults to robot IP `192.168.1.208`; the actual FR3 is `192.168.1.206`. Retried with explicit `--robot-ip 192.168.1.206`.
- Host/docker read-only joint check passed at current default FR3 start:
  `[0.002376, -0.771332, -0.001417, -2.372946, 0.002449, 1.572662, 0.794596]` rad.
- Strict real-input preflight with 5mm chunk step threshold failed:
  `action[90] is discontinuous relative to previous target; pos_delta_mm=(2.25, -4.06, -5.21)`, vector magnitude about 7mm.
- Relaxed chunk preflight with `--preflight-max-step-pos-delta-mm 10` passed all 100 actions:
  `first_pos_delta_mm=(3.58, -5.25, -5.47)`, `max_step_pos_delta_mm=(4.55, 5.25, 5.47)`, `max_step_gripper_delta=0.006`.
- Runtime preview for first 3 steps kept the 5mm execution clamp and did not send actions:
  - step0: `status=clamped`, raw EE `(0.3047, -0.0032, 0.3162)`, safe EE `(0.3047, -0.0030, 0.3166)`, pos delta `(3.58, -5.25, -5.47)mm`
  - step1: `status=clamped`, raw EE `(0.3061, -0.0033, 0.3150)`, safe EE `(0.3059, -0.0030, 0.3159)`, pos delta `(5.12, -5.34, -5.96)mm`
  - step2: `status=clamped`, raw EE `(0.3068, -0.0038, 0.3140)`, safe EE `(0.3059, -0.0030, 0.3159)`, pos delta `(5.90, -5.78, -6.94)mm`
- Live EE start aligns well to dataset start: nearest start position error `0.06mm`, median `0.11mm`, p95 `0.29mm`; nearest rotation error `0.07deg`.
- Live gripper is not aligned: live `0.089`, dataset mean `0.994`, delta `0.905`. Preview used virtual gripper correction only. Before any real execution, physically align/open gripper to the dataset-like start.

Conclusion: ACT 060000 is much better than pi05 old LoRA and passes offline gates. Real-input strict 5mm preflight does not fully pass because of a late-chunk ~7mm step; relaxed 10mm chunk continuity passes. Do not execute on hardware yet without gripper alignment and an explicit decision on whether the 5mm strict chunk threshold should remain a blocker or whether runtime 5mm clamping is acceptable for a low-speed first trial.


## 2026-06-12 Pika gripper normalization root cause

The earlier real-input preflight warning `live_gripper=0.089 dataset_gripper_mean=0.994` was a unit-conversion bug, not a physical gripper misalignment.

Evidence:

- The FR3 Pika dataset stores gripper values as normalized width in `[0, 1]`:
  - start `action[7] = 1.0`
  - start `observation.state gripper.pos ~= 0.994`
  - full dataset `state_g min/mean/max = 0.0 / 0.782 / 0.995`
- `PikaGripperHardwareDriver.get_position()` already returns normalized width: `width_mm / max_width_mm`.
- `tools/fr3/fr3_act_infer_real_runtime.py` incorrectly treated Pika live observations as normalized values that needed conversion back to meters, returning `0.994 * 0.09 ~= 0.089m`.
- The runtime then compared that meter value against the dataset's normalized `0.994`, producing the false `delta_to_mean=0.905` warning.
- The same bug affected policy gripper commands: for Pika, decoded dataset actions were treated as meters and divided by `0.09`, so any action above `0.09` would saturate to `1.0`.

Fix:

- For Pika, keep dataset gripper values and live gripper observations in normalized `[0, 1]` units.
- Keep DAS-specific meter conversion only for `gripper_backend == 'das'`.
- Restored runtime compatibility for `--dataset-frame`, Pika URDF/target frame, and action-chunk preflight helper after the local/remote file mismatch was found.

Validation after fix:

```text
pika_norm 0.994
pika_live 0.994
pika_action05 0.5
das_norm 1.0
das_live 0.1
```

Real-input preview-only strict preflight after fix:

- log: `outputs/preflight/act_fr3_ee2ee_20260612_1036_default_start_060000/preflight_real_preview_gripperfix_strict_20260612_142212.log`
- `live_gripper=0.994 dataset_gripper_mean=0.994 delta_to_mean=0.000`
- no gripper alignment warning
- `[PREFLIGHT] action_chunk=pass checked_actions=100/100`
- `max_step_gripper_delta=0.002`
- first action EE delta remains nonzero and is runtime-clamped: `pos_delta_mm=(6.29, -7.72, -7.16)`; this is separate from gripper normalization.
