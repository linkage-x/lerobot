# pi05 LoRA FR3 experiment record - 2026-06-11 to 2026-06-12

## Context

- Host: `hph@192.168.100.155`
- Repo: `/home/hph/Code/lerobot`
- GPU: NVIDIA RTX 4090 24 GB
- Dataset: `outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011`
- Dataset size: 20 episodes, 8400 frames, 30 FPS
- Policy: `pi05`
- Base model: `/home/tele/Models/pi05_base`
- Fine-tuning: PEFT LoRA
- Robot/action convention: absolute end-effector action, target frame `pika_gripper_ee`, dataset frame `target_ee`
- Collection mode: 3D mouse teleoperation

The dataset start pose is intentionally consistent across episodes. At each episode start,
the first ground-truth action is effectively a hold command at the current EE pose.

## Source and runtime fixes made before evaluation

1. `src/lerobot/policies/pi05/modeling_pi05.py`
   - Added a key remap from `.vision_tower.vision_model.` to `.vision_tower.`.
   - Reason: local `/home/tele/Models/pi05_base/model.safetensors` has PaliGemma vision-tower keys with the extra `vision_model` segment, while the current code expects keys without it.
   - Verification: model load prints `Remapped 812 state dict keys` and `All keys loaded successfully!`.

2. `src/lerobot/policies/pi_gemma.py`
   - Added compatibility fallback for `create_causal_mask(..., cache_position=...)`.
   - Reason: the installed transformers version in the Docker image has a `create_causal_mask` signature without `cache_position`.

3. `tools/fr3/fr3_infer_real.py` and `tools/fr3/fr3_infer_real_runtime.py`
   - Added generic FR3 inference entrypoint for policies that expose `predict_action_chunk`, including ACT and pi05+LoRA.
   - Kept `tools/fr3/fr3_act_infer_real.py` for backward compatibility.

4. `tools/fr3/fr3_act_infer_real_runtime.py`
   - Added real-input preflight before any action is sent to the robot.
   - Preflight uses real robot/camera observation, calls `policy.predict_action_chunk`, decodes the first action chunk, and rejects:
     - non-finite actions,
     - first target too far from current EE pose,
     - discontinuous adjacent targets inside the action chunk.
   - Added Pika/default target frame support:
     - Pika URDF: `fr3_pika_gripper_ati.urdf`
     - target frame: `pika_gripper_ee`
     - dataset frame: `target_ee`
   - Added task text handling from dataset `meta/tasks.parquet`.

## Initial training run: 3000 steps

Command family:

```bash
docker run --rm --gpus all --ipc=host --network host --user root \
  -v /home/hph/Code/lerobot:/workspace \
  -v /home/tele/Models:/models:ro \
  -v /home/tele/Models:/data/model:ro \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e WANDB_MODE=disabled \
  lerobot-user:local \
  python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=hph/fr3_pick_place_ee2ee_v1 \
    --dataset.root=/workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011 \
    --policy.type=pi05 \
    --policy.pretrained_path=/models/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r=16 \
    --batch_size=1 \
    --steps=3000 \
    --save_freq=500 \
    --eval_freq=0 \
    --num_workers=2 \
    --wandb.enable=false
```

Output:

- Run: `outputs/train/pi05_lora_fr3_20260611_143011_20260611_150930`
- Final checkpoint: `checkpoints/003000/pretrained_model`
- Log: `scripts/train_logs/pi05_lora_train_20260611_150930.log`
- Result: training completed successfully.

Important observation:

- The 3000-step run covered only about 0.36 dataset epochs with `batch_size=1`.
- Training loss dropped into roughly `0.5-1.0` range near the end, but this did not imply safe actions.

## Real-input preflight result for 003000

Command:

```bash
python3 tools/fr3/fr3_infer_real.py \
  --checkpoint outputs/train/pi05_lora_fr3_20260611_143011_20260611_150930/checkpoints/003000 \
  --gripper-backend=pika \
  --robot-ip=192.168.1.206 \
  --preview \
  --max-steps=1 \
  --no-move-to-das-start \
  --preflight-max-actions=50
```

Result:

```text
RuntimeError: Preflight failed: first action target is too far from current EE pose;
pos_delta_mm=(195.52, -24.39, -168.88)
rot_delta_deg=(-0.05, -0.67, -1.16)
```

Interpretation:

- The command was run in preview mode, so no action was sent.
- The robot start pose matched the dataset start pose closely.
- Therefore the failure was not explained by the robot being in the wrong joint/EE start state.

## Offline evaluation added

Files added:

- `scripts/train_pi05_lora_fr3_long_20260611.sh`
- `scripts/eval_pi05_lora_fr3_offline.py`
- `docs/train_runs/pi05_lora_fr3_retrain_eval_20260611.md`

Offline evaluation checks the checkpoint against dataset frames without touching hardware.
For start-frame evaluation, every episode frame 0 should predict an action close to the
current/ground-truth EE pose.

Main metrics:

- `first_current_pos_mm`: distance from predicted first action to current observation EE position.
- `pred_gt_pos_mm`: distance from predicted first action to dataset ground-truth action.
- `first_current_rot_deg`: rotation error from predicted first action to current observation.
- `chunk_max_step_pos_mm`: largest adjacent position jump inside the predicted chunk.

Acceptance gates before real preview:

- `first_current_pos_mm max < 30 mm`; target is `< 5-10 mm`.
- `pred_gt_pos_mm p95 < 40 mm`; target is `< 10-20 mm`.
- `chunk_max_step_pos_mm` should be compatible with runtime preflight; target is usually `< 5 mm` at policy rate.

## Offline result for 003000

The failed 003000 checkpoint was evaluated on dataset start frames and individual start samples.

Representative result for dataset index 0:

```text
current_xyz = [0.305559, 0.001911, 0.332866]
gt_xyz      = [0.305559, 0.001911, 0.332866]
pred_xyz    = [0.525598, -0.047627, 0.164754]
pred-current mm = [220.04, -49.54, -168.11]
```

Another scripted evaluation reported:

```text
first_current_mm = 282.34
pred_gt_mm       = 282.34
chunk_max_step_pos_mm = 20.66
```

Conclusion:

- The 003000 checkpoint collapsed toward a middle/mean region of the action distribution.
- The issue reproduced on training data itself, so it was not caused by live camera input or robot start alignment.

## Long training run: 60000 steps

Command:

```bash
cd /home/hph/Code/lerobot
STEPS=60000 LORA_R=64 LR=5e-5 SAVE_FREQ=5000 bash scripts/train_pi05_lora_fr3_long_20260611.sh
```

Run:

- `outputs/train/pi05_lora_fr3_long_20260611_162839`
- Checkpoint evaluated: `checkpoints/060000/pretrained_model`

This run used:

- LoRA rank: 64
- Steps: 60000
- Learning rate: `5e-5`
- Batch size: 1
- Save frequency: 5000

## Offline result for 060000

User ran:

```bash
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
    --checkpoint outputs/train/pi05_lora_fr3_long_20260611_162839/checkpoints/060000 \
    --start-frames-only \
    --max-samples 20 \
  2>&1 | tee outputs/eval/pi05_lora_fr3_long_20260611_162839/eval_060000_start20.log
```

Summary:

```text
[SUMMARY] first_current_pos_mm: mean=39.91 p50=37.58 p95=69.55 max=82.41
[SUMMARY] first_current_rot_deg: mean=0.47 p50=0.41 p95=0.98 max=1.38
[SUMMARY] pred_gt_pos_mm: mean=39.91 p50=37.58 p95=69.55 max=82.41
[SUMMARY] pred_gt_rot_deg: mean=0.47 p50=0.41 p95=0.98 max=1.38
[SUMMARY] chunk_max_step_pos_mm: mean=10.66 p50=10.58 p95=13.34 max=14.83
[FAIL] first_current_pos max exceeds 30.0 mm
[FAIL] pred_gt_pos p95 exceeds 40.0 mm
```

Interpretation:

- 060000 improved substantially over 003000.
  - 003000 first-action error was roughly 150-280 mm.
  - 060000 first-action mean error is about 40 mm.
- However, 060000 still fails the safety gate.
  - Max first-action error is 82.41 mm.
  - p95 first-action error is 69.55 mm.
  - Chunk step jumps average about 10.66 mm, above the usual 5 mm runtime continuity target.
- The checkpoint must not be used for real execution.

## Training while running eval

Running offline eval during training does not intentionally stop or mutate the training process.
It loads a saved checkpoint in a separate Docker container and reads dataset/model files.

Risks:

- GPU memory contention:
  - Training was observed using about 10 GB of the RTX 4090.
  - pi05 eval also loads the base model and LoRA adapter.
  - If combined memory exceeds available VRAM, one of the processes can fail with CUDA OOM.
- GPU compute contention:
  - Training steps may slow down during eval.
- Checkpoint race:
  - Do not evaluate a checkpoint while it is still being written.
  - Evaluate only after the checkpoint directory contains a complete `pretrained_model` with adapter/config/processor files.

Practical rule:

- It is acceptable to evaluate completed checkpoints while training continues if VRAM headroom is enough.
- If training is critical, wait until a save finishes, run eval with small `--max-samples`, and watch `nvidia-smi`.
- A failed eval container normally does not kill training, but a severe GPU OOM can destabilize either process. Avoid running eval if training already uses most VRAM.

## Current diagnosis

The current evidence points away from these causes:

- Wrong robot start pose: rejected, because offline dataset start frames reproduce the issue.
- Live camera mismatch only: rejected, because offline dataset images reproduce the issue.
- Missing base model keys: unlikely after the vision-tower key remap; model load reports all keys loaded.
- No LoRA training at all: unlikely, because 060000 improved strongly compared with 003000.

Most likely causes:

1. Objective/target mismatch for start-frame behavior.
   - For 3D mouse absolute-EE data, the first action at episode start is a hold action.
   - pi05 flow matching may be learning the broader future chunk/action distribution but not sufficiently anchoring the first action to the current state.
   - This explains why loss improves while the first action still drifts tens of millimeters.

2. Loss is not aligned with the hardware safety metric.
   - The train loss is a flow/diffusion-style objective over normalized action chunks.
   - A checkpoint can have acceptable loss but still be unsafe at the first decoded action.
   - Therefore checkpoint selection must use offline action-quality metrics, not loss alone.

3. Absolute action representation may be harder for pi05 than delta action representation.
   - The policy predicts absolute EE pose directly.
   - Small normalized errors in absolute pose can decode into large real-space first-action offsets.
   - A delta/residual action target or explicit first-action anchoring may be needed.

4. Start-frame samples are sparse relative to all action-chunk targets.
   - There are only 20 episode-start frames.
   - The model can reduce global loss without perfectly fitting these safety-critical start states.

5. Chunk continuity is still not good enough.
   - 060000 has `chunk_max_step_pos_mm mean=10.66`, which is too jumpy for direct real-time execution.
   - Even if first action improved, chunk continuity remains a separate blocker.

## Recommended next experiments

1. Evaluate intermediate checkpoints from the 60000-step run.
   - Check `005000`, `010000`, ..., `060000`.
   - Determine whether 060000 is the best checkpoint or whether an earlier checkpoint had lower first-action drift.

2. Add a start-frame/action-chunk diagnostic table.
   - For every checkpoint, record:
     - `first_current_pos_mm mean/p95/max`
     - `pred_gt_pos_mm mean/p95/max`
     - `chunk_max_step_pos_mm mean/p95/max`
     - train loss near that checkpoint

3. If all checkpoints fail similarly, stop increasing steps.
   - More training alone is unlikely to solve the first-action safety issue after 60000 steps.

4. Test a delta/residual action formulation or first-action anchoring.
   - Candidate target: action as delta from current EE pose for the policy, then reconstruct absolute command at runtime.
   - Candidate runtime guard: force the first command to current pose and blend into the predicted chunk only after continuity passes. This is a guard, not a substitute for a correct policy.

5. Compare with ACT on the same dataset and evaluation script.
   - If ACT learns the start-frame hold behavior while pi05 does not, the issue is likely pi05 objective/action representation rather than dataset quality.

6. Do not run real execution yet.
   - The next allowed hardware step is preview-only real-input preflight after offline gates pass.


本项目当前 pi05 默认 LoRA target 只覆盖 gemma_expert 的 q/v 和部分投影层，而且投影层也只是 LoRA 形式；openpi 官方 LoRA 语义是 LLM/action-expert 的 attn+ffn
  上 LoRA，同时非 LLM 的 action/time projection 仍可全量训练。我们应该修这个默认 PEFT 目标，再重跑 pi05+lora 对比。

## 2026-06-12 OpenPI-style pi0.5 LoRA review

User requested re-checking pi0.5+LoRA against `git@github.com:Physical-Intelligence/openpi.git`, with LeRobot v2.1 loading adapted to this repo's LeRobot v3 dataset format.

### OpenPI findings

- OpenPI's LeRobot loader imports the old API: `lerobot.common.datasets.lerobot_dataset.LeRobotDatasetMetadata/LeRobotDataset`.
- This repo uses LeRobot v3: `lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata/LeRobotDataset`, with an explicit local `root` for the FR3 dataset.
- The FR3 dataset is LeRobot v3, 8400 frames at 30fps:
  - `observation.state`: 16 dims
  - `action`: 8 dims
  - images: `observation.images.ee`, `observation.images.side`, `observation.images.front`
  - task: `Pick and place`
- OpenPI's pi0.5 model expects state/action padded to 32 dims and three image slots. The natural FR3 mapping is:
  - `base_0_rgb <- observation.images.front`
  - `left_wrist_0_rgb <- observation.images.ee`
  - `right_wrist_0_rgb <- observation.images.side`
- Directly running the upstream openpi repo inside the current `lerobot-user:local` image is not currently viable: the image lacks `jax`, `flax`, `tyro`, `optax`, and `sentencepiece`; the target machine is assumed offline.

### Root cause found in current pi05 LoRA training

The previous run used PEFT, but not the same effective LoRA recipe as OpenPI:

- Old `adapter_config.json` at `outputs/train/pi05_lora_fr3_long_20260611_162839/checkpoints/060000/pretrained_model`:
  - `r=64`
  - `lora_alpha=8` because our CLI did not expose `lora_alpha`
  - `target_modules=(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))`
  - `modules_to_save=[]`
- Problems:
  - `alpha/r = 8/64 = 0.125`, much weaker than OpenPI's built-in LoRA variants, where alpha equals rank.
  - Only action expert q/v attention got LoRA; OpenPI's LoRA variants cover attention and FFN blocks.
  - PaliGemma language model LoRA was not included.
  - pi0.5 uses `time_mlp_in/out`, but old target listed `action_time_mlp_in/out` from pi0, so the time MLP was not adapted.
  - OpenPI's freeze filter keeps non-LLM action/time projections trainable; our generic PEFT wrapper froze everything and did not save those projections fully.

This is now the leading explanation for why 60000 steps improved from catastrophic drift to ~40mm but did not pass preflight.

### Code changes made

- `src/lerobot/configs/default.py`
  - Added PEFT CLI fields `lora_alpha` and `lora_dropout`.
- `src/lerobot/policies/pi05/modeling_pi05.py`
  - Changed pi05 default PEFT targets to OpenPI-style:
    - LoRA on PaliGemma language model and action expert attention q/k/v/o projections.
    - LoRA on PaliGemma language model and action expert FFN gate/up/down projections.
    - Full train/save for `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` via `modules_to_save`.
- `scripts/train_pi05_lora_fr3_openpi_style_20260612.sh`
  - New training script using the same FR3 LeRobot v3 dataset and local `/home/tele/Models/pi05_base`.
  - Defaults: `STEPS=60000`, `LORA_R=64`, `LORA_ALPHA=${LORA_R}`, `LR=5e-5`, `SAVE_FREQ=5000`.

Remote syntax/config checks passed in docker:

```bash
python -m py_compile src/lerobot/configs/default.py src/lerobot/policies/pi05/modeling_pi05.py
python -c "from lerobot.configs.default import PeftConfig; print(PeftConfig(r=64,lora_alpha=64,lora_dropout=0.0))"
```

### Training command to run after ACT frees the GPU

Do not launch this while the ACT comparison job is still using the 4090 unless we intentionally pause/stop ACT.

```bash
cd /home/hph/Code/lerobot
RUN_ID=20260612_openpi_style_fr3_default_start \
STEPS=60000 LORA_R=64 LORA_ALPHA=64 LR=5e-5 SAVE_FREQ=5000 LOG_FREQ=20 \
bash scripts/train_pi05_lora_fr3_openpi_style_20260612.sh
```

Expected early validation after launch:

- Log should show PEFT wrapping and a much larger `num_learnable_params` than the previous `5,148,672`.
- The produced `adapter_config.json` should show:
  - `r=64`
  - `lora_alpha=64`
  - `modules_to_save` containing `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`
  - target regex covering `paligemma.model.language_model` and `gemma_expert.model` layers.

### Eval command for each checkpoint

```bash
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
    --checkpoint outputs/train/pi05_lora_fr3_openpi_style_20260612_openpi_style_fr3_default_start/checkpoints/005000 \
    --start-frames-only \
    --max-samples 20
```

Offline gates remain unchanged before any real robot movement:

- `first_current_pos_mm max <= 30mm`
- `pred_gt_pos_mm p95 <= 40mm`
- chunk continuity should trend down; target `chunk_max_step_pos_mm p95 <= 10mm` before preview-only real-input preflight.

### ACT comparison status at 2026-06-12 11:05 CST

ACT training is still running and improving on the same dataset/default FR3 start:

- 005000: `first_current_pos_mm mean=50.38 p95=65.90 max=68.54`, failed.
- 010000: `first_current_pos_mm mean=43.60 p95=50.18 max=52.05`, failed.
- 015000: `first_current_pos_mm mean=32.10 p95=36.05 max=38.41`, only failed `first_current_pos max > 30mm`.

This trend supports that the dataset and default-start framing are learnable; it does not support continuing the old pi05 LoRA recipe unchanged.
