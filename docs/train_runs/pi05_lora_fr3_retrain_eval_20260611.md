# pi05 LoRA FR3 retraining and evaluation plan - 2026-06-11

## Why retrain

The checkpoint from `outputs/train/pi05_lora_fr3_20260611_143011_20260611_150930/checkpoints/003000`
is not safe for real execution. Real-input preflight rejected the first action by about
`(+195.5, -24.4, -168.9) mm`. An offline check on dataset start frames reproduced the same failure:
the ground-truth first action stays at the current end-effector pose, while the model predicts a pose
near the middle of the dataset action distribution.

The 3000-step run only covered about 0.36 dataset epochs with `batch_size=1`, so it is too weak as a
final real-robot checkpoint.

## Training command

Run this on `hph@192.168.100.155`:

```bash
cd /home/hph/Code/lerobot
bash scripts/train_pi05_lora_fr3_long_20260611.sh
```

Default settings in the script:

- Dataset: `/home/hph/Code/lerobot/outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011`
- Base model: `/home/tele/Models/pi05_base`, mounted as both `/models` and `/data/model`
- Docker image: `lerobot-user:local`
- GPU: one RTX 4090
- Steps: `30000`
- Batch size: `1`
- LoRA rank: `32`
- Learning rate: `5e-5`
- Save frequency: every `2500` steps
- Output: `outputs/train/pi05_lora_fr3_long_<timestamp>`
- Log: `scripts/train_logs/pi05_lora_fr3_long_<timestamp>.log`

Override example:

```bash
cd /home/hph/Code/lerobot
STEPS=60000 LORA_R=64 LR=5e-5 SAVE_FREQ=5000 bash scripts/train_pi05_lora_fr3_long_20260611.sh
```

Do not pick the final checkpoint automatically. Choose the checkpoint by offline action metrics first.

## Offline quality evaluation

Evaluate a checkpoint inside the same docker image:

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
    --checkpoint outputs/train/<run>/checkpoints/<step> \
    --start-frames-only \
    --max-samples 20
```

For the current failed checkpoint, this check reports first-action start-frame jumps around
150-230 mm. A candidate checkpoint should pass these gates before any real robot run:

- On dataset start frames, `first_current_pos_mm` max should be below `30 mm`; target is below `5-10 mm`.
- On dataset start frames, `first_current_rot_deg` max should be below `10 deg`; target is below `2 deg`.
- `pred_gt_pos_mm` p95 should be below `40 mm`; target is below `10-20 mm`.
- `chunk_max_step_pos_mm` should be compatible with runtime preflight, normally below `5 mm` if the chunk is meant to be sent at policy rate.
- The first predicted action must not collapse to the dataset mean position.

Training loss is only a weak signal for pi05 flow matching. Use loss to detect obvious failure, but use
offline action metrics to select checkpoints.

## Real-input preflight gate

Only after offline gates pass, run real-input preview:

```bash
cd /home/hph/Code/lerobot
python3 tools/fr3/fr3_infer_real.py \
  --checkpoint outputs/train/<run>/checkpoints/<step> \
  --gripper-backend=pika \
  --robot-ip=192.168.1.206 \
  --preview \
  --max-steps=1 \
  --no-move-to-das-start \
  --preflight-max-actions=50
```

This must print `[PREFLIGHT] action_chunk=pass`. If it fails the first-action or continuity gate, do
not execute the policy on hardware.

## How to check whether LoRA is effective

1. Confirm adapter files exist:

```bash
find outputs/train/<run>/checkpoints/<step>/pretrained_model -maxdepth 1 -type f \
  \( -name 'adapter_config.json' -o -name 'adapter_model.safetensors' \) -ls
```

2. Confirm adapter tensors are non-trivial:

```bash
cd /home/hph/Code/lerobot
docker run --rm --ipc=host --network host --user root \
  -v /home/hph/Code/lerobot:/workspace \
  -w /workspace lerobot-user:local \
  python - <<'PY'
from pathlib import Path
from safetensors.torch import load_file

ckpt = Path("outputs/train/<run>/checkpoints/<step>/pretrained_model/adapter_model.safetensors")
sd = load_file(str(ckpt))
total_params = sum(t.numel() for t in sd.values())
total_abs = sum(float(t.float().abs().sum()) for t in sd.values())
total_l2 = sum(float((t.float() ** 2).sum()) for t in sd.values()) ** 0.5
print("adapter_tensors", len(sd))
print("adapter_params", total_params)
print("adapter_abs_sum", total_abs)
print("adapter_l2", total_l2)
for name, tensor in sorted(sd.items())[:10]:
    print(name, tuple(tensor.shape), float(tensor.float().norm()))
PY
```

3. Confirm the loaded policy is a PEFT model and contains LoRA modules:

```bash
cd /home/hph/Code/lerobot
docker run --rm --gpus all --ipc=host --network host --user root \
  -v /home/hph/Code/lerobot:/workspace \
  -v /home/tele/Models:/models:ro \
  -v /home/tele/Models:/data/model:ro \
  -w /workspace \
  -e PYTHONPATH=/workspace/src:/workspace \
  lerobot-user:local \
  python - <<'PY'
from pathlib import Path
import torch
from tools.fr3.fr3_act_infer_real_runtime import (
    load_dataset_metadata,
    load_policy_stack,
    load_train_config,
    resolve_dataset_root,
    resolve_pretrained_model_dir,
)

ckpt = resolve_pretrained_model_dir("outputs/train/<run>/checkpoints/<step>")
cfg = load_train_config(ckpt)
root = resolve_dataset_root(ckpt, cfg, None)
meta = load_dataset_metadata(root, cfg.dataset.repo_id)
policy, _, _ = load_policy_stack(ckpt, ds_meta=meta, device=torch.device("cuda"))
lora_names = [name for name, _ in policy.named_parameters() if "lora_" in name]
print("policy_type", type(policy))
print("num_lora_params", len(lora_names))
print("first_lora_params", lora_names[:20])
PY
```

4. Compare checkpoints by offline metrics. LoRA is only useful if later checkpoints improve
`first_current_pos_mm` and `pred_gt_pos_mm` over early checkpoints. A nonzero adapter file is necessary
but not sufficient.

5. If adapter norms are nonzero but offline action metrics stay near the dataset mean, increase training
duration first. If long runs still fail, raise LoRA capacity (`LORA_R=64`) or review whether the pi05
training objective is using the expected action chunk target for absolute EE actions.
