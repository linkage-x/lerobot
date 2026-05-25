#!/usr/bin/env bash
set -euo pipefail

cd /home/corenetic/Code/zyx/lerobot

mode="${1:-preview}"

common_args=(
  tools/fr3/fr3_act_infer_real.py
  --inference-config outputs/datasets/pick_place_act_cam2_cam3_pika_right_imgonly/inference_config.generated.yaml
  --checkpoint outputs/train/pick_place_act_cam2_cam3_pika_right_imgonly/checkpoints/060000
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml
  --gripper-backend pika
  --gripper-port /dev/ttyUSB0
  --robot-ip 192.168.11.102
  --robot-init-state ee_xyzquat=0.584972 -0.198668 0.280659 0.959580 0.006716 0.281263 -0.007254
  --first-frame-max-pos-delta-mm 20
  --first-frame-max-rot-delta-deg 8
  --max-step-pos-delta-mm 3
  --max-step-rot-delta-deg 2
  --camera-preview-window
)

case "$mode" in
  preview)
    exec python3 "${common_args[@]}" --preview --max-steps 20
    ;;
  real)
    exec python3 "${common_args[@]}" --max-steps 300
    ;;
  *)
    echo "Usage: $0 [preview|real]" >&2
    exit 2
    ;;
esac
