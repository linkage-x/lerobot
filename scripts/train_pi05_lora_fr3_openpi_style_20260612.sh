#!/usr/bin/env bash
set -euo pipefail

# OpenPI-style PI0.5 LoRA training for the FR3 LeRobot v3 dataset.
# Run from /home/hph/Code/lerobot on hph@192.168.100.155.
# This uses the local pi05_base under /home/tele/Models and does not download weights.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${REPO_ROOT:-/home/hph/Code/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011}"
MODELS_ROOT="${MODELS_ROOT:-/home/tele/Models}"
IMAGE="${IMAGE:-lerobot-user:local}"

STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_R}}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-60000}"

JOB_NAME="${JOB_NAME:-pi05_lora_fr3_openpi_style_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/scripts/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_DIR}")"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -f "${MODELS_ROOT}/pi05_base/model.safetensors" ]]; then
  echo "[INFO] pi05_base host_check=readable"
else
  echo "[WARN] pi05_base is not readable from host user at ${MODELS_ROOT}/pi05_base/model.safetensors; checking from Docker mount..."
  if ! docker run --rm --ipc=host --network host --user root     -v "${MODELS_ROOT}:/models:ro"     "${IMAGE}" bash -lc "test -f /models/pi05_base/model.safetensors"; then
    echo "[ERROR] pi05 base checkpoint not found or not readable inside Docker: /models/pi05_base/model.safetensors mounted from ${MODELS_ROOT}" >&2
    exit 1
  fi
  echo "[INFO] pi05_base docker_check=readable"
fi

echo "[INFO] job=${JOB_NAME}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] log_file=${LOG_FILE}"
echo "[INFO] dataset_root=${DATASET_ROOT}"
echo "[INFO] steps=${STEPS} batch_size=${BATCH_SIZE} lora_r=${LORA_R} lora_alpha=${LORA_ALPHA} lr=${LR}"
echo "[INFO] openpi_style=llm_and_action_expert_attn_ffn_lora plus full action/time projections"

docker run --rm --gpus all --ipc=host --network host --user root \
  -v "${REPO_ROOT}:/workspace" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/triton:/root/.cache/triton" \
  -v "${MODELS_ROOT}:/models:ro" \
  -v "${MODELS_ROOT}:/data/model:ro" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e WANDB_MODE=disabled \
  "${IMAGE}" bash -lc "
    set -euo pipefail
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
      --policy.optimizer_lr=${LR} \
      --policy.scheduler_warmup_steps=${WARMUP_STEPS} \
      --policy.scheduler_decay_steps=${DECAY_STEPS} \
      --peft.method_type=LORA \
      --peft.r=${LORA_R} \
      --peft.lora_alpha=${LORA_ALPHA} \
      --peft.lora_dropout=${LORA_DROPOUT} \
      --batch_size=${BATCH_SIZE} \
      --steps=${STEPS} \
      --save_freq=${SAVE_FREQ} \
      --eval_freq=0 \
      --log_freq=${LOG_FREQ} \
      --num_workers=${NUM_WORKERS} \
      --wandb.enable=false \
      --output_dir=/workspace/outputs/train/${JOB_NAME} \
      --job_name=${JOB_NAME}
  " 2>&1 | tee "${LOG_FILE}"

echo "[INFO] done"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] log_file=${LOG_FILE}"
