#!/usr/bin/env bash
set -euo pipefail
log_dir=logs/experiments/7jfgkukz/current
mkdir -p "$log_dir"
start_job() {
  local gpu=$1; shift
  local cfg=$1; shift
  local tag=$1; shift
  local log="$log_dir/${tag}.log"
  echo "Launching $tag on GPU $gpu → log: $log"
  nohup setsid bash -lc "CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --num_processes 1 src/lerobot/scripts/lerobot_train.py --config_path=${cfg}" > "$log" 2>&1 &
  local pid=$!
  echo $pid > "$log_dir/${tag}.pid"
  local pgid
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  echo $pgid > "$log_dir/${tag}.pgid"
}
start_job 3 src/lerobot/scripts/train_config/experiments/7jfgkukz/act_expV1_vae_noaug.json expV1_vae_noaug
start_job 4 src/lerobot/scripts/train_config/experiments/7jfgkukz/act_expV2_vae_rn50_frozen.json expV2_vae_rn50_frozen
start_job 5 src/lerobot/scripts/train_config/experiments/7jfgkukz/act_expV3_vae_prenorm_drop01.json expV3_vae_prenorm_drop01
start_job 6 src/lerobot/scripts/train_config/experiments/7jfgkukz/act_expV4_vae_sched_coswarm.json expV4_vae_sched_coswarm
start_job 7 src/lerobot/scripts/train_config/experiments/7jfgkukz/act_expV5_vae_chunk60.json expV5_vae_chunk60
