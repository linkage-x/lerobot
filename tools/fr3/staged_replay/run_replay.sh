#!/usr/bin/env bash
set -euo pipefail
STAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$STAGE_DIR/python:$STAGE_DIR"
export LD_LIBRARY_PATH="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"
cd "$STAGE_DIR"
exec /home/nvidia/Code/infer/.venv-fr3/bin/python "$STAGE_DIR/operator_replay.py" "$@"
