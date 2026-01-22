#!/usr/bin/env bash
set -euo pipefail
log_dir=logs/experiments/bp_0121_68ep_ee2ee/current
mkdir -p "$log_dir"
start_job() {
  local gpu=$1; shift
  local cfg=$1; shift
  local tag=$1; shift
  local log="$log_dir/${tag}.log"
  echo "Launching $tag on GPU $gpu -> log: $log"
  nohup setsid bash -lc "CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --num_processes 1 src/lerobot/scripts/lerobot_train.py --config_path=${cfg}" > "$log" 2>&1 &
  local pid=$!
  echo $pid > "$log_dir/${tag}.pid"
  local pgid
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  echo $pgid > "$log_dir/${tag}.pgid"
}
# start_job 2 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp1_coswarm_lr2e5.json exp1_coswarm_lr2e5
start_job 3 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp2_lr5e5.json exp2_lr5e5
start_job 4 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp3_lr1e5.json exp3_lr1e5
start_job 5 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp4_reg_wd2e4_drop03.json exp4_reg_wd2e4_drop03
start_job 6 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp5_noaug.json exp5_noaug
start_job 7 src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee/act_bp_0121_68ep_exp6_chunk40.json exp6_chunk40
