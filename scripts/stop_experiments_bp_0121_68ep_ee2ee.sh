#!/usr/bin/env bash
set -euo pipefail
log_dir=logs/experiments/bp_0121_68ep_ee2ee/current
kill_tag(){
  local tag=$1
  local pid_file="$log_dir/${tag}.pid"
  local pgid_file="$log_dir/${tag}.pgid"
  if [[ -f "$pgid_file" ]]; then
    pgid=$(cat "$pgid_file")
    if [[ -n "$pgid" ]]; then
      echo "Stopping $tag by PGID $pgid" && kill -TERM -"$pgid" 2>/dev/null || true
    fi
  fi
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $tag (PID $pid)" && kill "$pid" 2>/dev/null || true
    fi
  fi
}
force_kill_by_cfg(){
  local cfg=$1
  mapfile -t pids < <(pgrep -f "src/lerobot/scripts/lerobot_train.py --config_path=${cfg}" || true)
  for pid in "${pids[@]:-}"; do
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "Force killing cfg=$cfg pid=$pid pgid=$pgid"
    kill -TERM "$pid" 2>/dev/null || true
    [[ -n "$pgid" ]] && kill -TERM -"$pgid" 2>/dev/null || true
  done
}
for tag in exp1_coswarm_lr2e5 exp2_lr5e5 exp3_lr1e5 exp4_reg_wd2e4_drop03 exp5_noaug exp6_chunk40; do
  kill_tag "$tag"
done
sleep 2
cfg_dir=src/lerobot/scripts/train_config/experiments/bp_0121_68ep_ee2ee
for name in act_bp_0121_68ep_exp1_coswarm_lr2e5.json act_bp_0121_68ep_exp2_lr5e5.json act_bp_0121_68ep_exp3_lr1e5.json act_bp_0121_68ep_exp4_reg_wd2e4_drop03.json act_bp_0121_68ep_exp5_noaug.json act_bp_0121_68ep_exp6_chunk40.json; do
  force_kill_by_cfg "$cfg_dir/$name"
done
sleep 1
# Hard kill leftovers
for name in act_bp_0121_68ep_exp1_coswarm_lr2e5.json act_bp_0121_68ep_exp2_lr5e5.json act_bp_0121_68ep_exp3_lr1e5.json act_bp_0121_68ep_exp4_reg_wd2e4_drop03.json act_bp_0121_68ep_exp5_noaug.json act_bp_0121_68ep_exp6_chunk40.json; do
  mapfile -t pids < <(pgrep -f "src/lerobot/scripts/lerobot_train.py --config_path=${cfg_dir}/${name}" || true)
  for pid in "${pids[@]:-}"; do
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "KILL -9 cfg=$name pid=$pid pgid=$pgid"
    kill -KILL "$pid" 2>/dev/null || true
    [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || true
  done

done
