#!/usr/bin/env bash
# Invoking this launcher authorizes one selected episode's staged test sequence.
# No piped confirmations, mode switching, failure recovery, or loop/retry.
set -euo pipefail
if [[ "$#" != 1 || ! "$1" =~ ^[01]$ ]]; then
  echo "用法: bash run_replay_once.sh 0  （或 1）；调用即授权启动真机流程。" >&2
  exit 2
fi
STAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "即将检查并启动 Episode $1：无需再输入 RUN；请确保现场已清空、急停在手边。"
exec bash "$STAGE_DIR/run_replay.sh" "$1" full-replay --execute
