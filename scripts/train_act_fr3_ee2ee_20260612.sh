#!/usr/bin/env bash
set -euo pipefail

# ACT comparison run for FR3 ee2ee data.
# Run on hph@192.168.100.155 from /home/hph/Code/lerobot.
# This is the non-DAS setup: no tactile inputs, no das_start move, Pika/default FR3 start data.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${REPO_ROOT:-/home/hph/Code/lerobot}"
IMAGE="${IMAGE:-lerobot-user:local}"

STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
LR="${LR:-1e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

JOB_NAME="${JOB_NAME:-act_fr3_ee2ee_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/scripts/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_DIR}")"

echo "[INFO] job=${JOB_NAME}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] log_file=${LOG_FILE}"
echo "[INFO] dataset=${REPO_ROOT}/outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011"
echo "[INFO] steps=${STEPS} batch_size=${BATCH_SIZE} chunk_size=${CHUNK_SIZE} n_action_steps=${N_ACTION_STEPS} lr=${LR}"

docker run --rm --gpus all --ipc=host --network host --user root \
  -v "${REPO_ROOT}:/workspace" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/triton:/root/.cache/triton" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e WANDB_MODE=disabled \
  "${IMAGE}" bash -lc "
    set -euo pipefail
    python -m lerobot.scripts.lerobot_train \
      --dataset.repo_id=hph/fr3_pick_place_ee2ee_v1 \
      --dataset.root=/workspace/outputs/datasets/fr3_pick_place_ee2ee_v1_20260611_143011 \
      --policy.type=act \
      --policy.device=cuda \
      --policy.use_amp=false \
      --policy.push_to_hub=false \
      --policy.chunk_size=${CHUNK_SIZE} \
      --policy.n_action_steps=${N_ACTION_STEPS} \
      --policy.vision_backbone=resnet18 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --policy.optimizer_lr=${LR} \
      --policy.optimizer_lr_backbone=${LR_BACKBONE} \
      --policy.optimizer_weight_decay=${WEIGHT_DECAY} \
      --batch_size=${BATCH_SIZE} \
      --steps=${STEPS} \
      --save_checkpoint=${SAVE_CHECKPOINT} \
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
