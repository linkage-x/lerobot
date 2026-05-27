#!/usr/bin/env bash
# Sync local repo to Thor (nvidia@192.168.111.122:~/lerobot).
# Usage:
#   bash run/sync_to_thor.sh            # full sync
#   bash run/sync_to_thor.sh --dry-run  # preview only
set -euo pipefail

THOR="nvidia@192.168.111.122"
REMOTE_DIR="~/lerobot"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"

rsync -avz --delete \
  --exclude='.git/' \
  --exclude='node_modules/' \
  --exclude='dist/' \
  --exclude='.venv/' \
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
  "$LOCAL_DIR" "${THOR}:${REMOTE_DIR}"

echo "✓ synced to ${THOR}:${REMOTE_DIR}"
