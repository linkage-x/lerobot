#!/usr/bin/env bash
set -euo pipefail
log_dir=logs/experiments/7jfgkukz/current
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
for tag in expV1_vae_noaug expV2_vae_rn50_frozen expV3_vae_prenorm_drop01 expV4_vae_sched_coswarm expV5_vae_chunk60; do
  kill_tag "$tag"
done
sleep 2
cfg_dir=src/lerobot/scripts/train_config/experiments/7jfgkukz
for name in act_expV1_vae_noaug.json act_expV2_vae_rn50_frozen.json act_expV3_vae_prenorm_drop01.json act_expV4_vae_sched_coswarm.json act_expV5_vae_chunk60.json; do
  force_kill_by_cfg "$cfg_dir/$name"
done
sleep 1
# Hard kill leftovers
for name in act_expV1_vae_noaug.json act_expV2_vae_rn50_frozen.json act_expV3_vae_prenorm_drop01.json act_expV4_vae_sched_coswarm.json act_expV5_vae_chunk60.json; do
  mapfile -t pids < <(pgrep -f "src/lerobot/scripts/lerobot_train.py --config_path=${cfg_dir}/${name}" || true)
  for pid in "${pids[@]:-}"; do
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "KILL -9 cfg=$name pid=$pid pgid=$pgid"
    kill -KILL "$pid" 2>/dev/null || true
    [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || true
  done

done
