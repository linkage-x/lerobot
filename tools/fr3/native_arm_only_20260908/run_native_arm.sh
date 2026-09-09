#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/python:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"
exec /home/nvidia/Code/infer/.venv-fr3/bin/python "$ROOT/native_arm.py" "$@"
