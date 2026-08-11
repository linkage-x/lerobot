#!/usr/bin/env bash
# Incrementally replace repository files on a deployment target.
#
# Usage:
#   bash run/sync_to_target.sh [thor|workstation] [rsync options...]
set -euo pipefail

target="${1:-thor}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$target" in
  thor)
    remote="nvidia@192.168.111.122"
    remote_dir="/home/nvidia/lerobot"
    ;;
  workstation)
    remote="hph@192.168.100.155"
    remote_dir="/home/hph/Code/lerobot"
    ;;
  *)
    echo "ERROR: unknown deployment target '$target' (expected thor or workstation)" >&2
    exit 2
    ;;
esac

local_dir="$(cd "$(dirname "$0")/.." && pwd)/"

echo "==> Preparing ${remote}:${remote_dir}"
ssh -o ConnectTimeout=5 "$remote" "mkdir -p '$remote_dir'"

echo "==> Incrementally replacing files on ${target}..."
rsync -avz --itemize-changes --delete-delay \
  --exclude='.git/' \
  --exclude='node_modules/' \
  --exclude='dist/' \
  --exclude='.pytest_cache/' \
  --exclude='.tmp-*' \
  --exclude='*.tsbuildinfo' \
  --exclude='MUJOCO_LOG.TXT' \
  --exclude='.venv' \
  --exclude='.venv/' \
  --exclude='.venv-fr3' \
  --exclude='.venv-fr3/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.eggs/' \
  --exclude='*.egg-info/' \
  --exclude='outputs/' \
  --exclude='notes/' \
  --exclude='.claude/' \
  --exclude='core' \
  --exclude='run/run_gateway.sh' \
  --exclude='run/run_vite.sh' \
  --exclude='run/restart_gateway.sh' \
  --exclude='run/logs/' \
  "$@" \
  "$local_dir" "${remote}:${remote_dir}/"
