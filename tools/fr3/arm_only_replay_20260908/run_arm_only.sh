#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/python:$ROOT:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"
cd "$ROOT"
# No SDK, networking changes, sudo, RT tuning or error recovery.
exec /home/nvidia/Code/infer/.venv-fr3/bin/python "$ROOT/arm_runtime.py" "$@"
