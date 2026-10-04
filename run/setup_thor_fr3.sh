#!/usr/bin/env bash
# Run on the FR3 control computer (host or Thor). Default/--check validates the local FR3 runtime without FCI.
# Provision explicitly with --install-system-deps and/or --install-python.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_path="tools/thor/gmsl2/thor_fr3_teleop.yaml"
venv_path="${VENV_PATH:-.venv-fr3}"
python_version="${PYTHON_VERSION:-3.12}"
panda_wheel=""
install_system_deps=false
install_python=false

usage() {
  cat <<'HELP'
Usage: bash run/setup_thor_fr3.sh [options]
  --check                 Validate dependencies, configuration and RT permission; no FCI (default)
  --install-system-deps   Explicitly install native build/runtime packages
  --install-python        Explicitly create/sync the separate FR3 Python environment
  --panda-wheel PATH      Patched panda-py wheel matching this CPU/Python (required with --install-python)
  --venv PATH             FR3 environment, default .venv-fr3 (also set fr3_teleop.runtime_python)
  --config-path PATH      Thor profile YAML
  --help                  Show this help
No kernel, robot firmware, robot recovery, motion, or remote SSH changes are made.
HELP
}

while (($#)); do
  case "$1" in
    --check) shift ;;
    --install-system-deps) install_system_deps=true; shift ;;
    --install-python) install_python=true; shift ;;
    --panda-wheel|--venv|--config-path)
      option="$1"
      if (($# < 2)); then echo "ERROR: $option requires a value" >&2; exit 2; fi
      case "$option" in
        --panda-wheel) panda_wheel="$2" ;;
        --venv) venv_path="$2" ;;
        --config-path) config_path="$2" ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if $install_python; then
  if [[ ! -f "$panda_wheel" || "$panda_wheel" != *.whl ]]; then
    echo "ERROR: --install-python requires --panda-wheel /path/to/compatible.whl" >&2
    echo "Patch panda-py and build against the libfranka version compatible with FR3's Desk system version." >&2
    echo "See docs/thor_fr3_teleoperation.md; generic PyPI panda-python is not selected automatically." >&2
    exit 2
  fi
  if [[ -n "${UV_BIN:-}" ]]; then
    uv_bin="$UV_BIN"
  elif command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
  else
    echo "ERROR: uv is required for --install-python; install uv on Thor first." >&2
    exit 1
  fi
fi

if $install_system_deps; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config libeigen3-dev libpoco-dev \
    liburdfdom-dev libhidapi-dev libhidapi-hidraw0 libhidapi-libusb0 libusb-1.0-0
fi

if $install_python; then
  # Core LeRobot imports used by the existing FR3 adapter include processors;
  # use its declared dependencies instead of guessing a partial runtime stack.
  # Camera/BOX ownership remains in the independent collection environment.
  UV_PROJECT_ENVIRONMENT="$venv_path" "$uv_bin" sync \
    --python "$python_version" --extra kinematics --extra spacemouse --no-dev
  "$uv_bin" pip install --python "$venv_path/bin/python" \
    'pyspacemouse==2.1.0' 'ruckig>=0.15.0,<0.16.0' 'scipy>=1.14,<2' 'websockets>=11.0' requests
  "$uv_bin" pip install --python "$venv_path/bin/python" --no-deps "$panda_wheel"
fi

runtime_python="$venv_path/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  echo "ERROR: FR3 runtime is missing: $repo_root/$runtime_python" >&2
  echo "C and camera/BOX collection remain available through bash run/deploy.sh." >&2
  echo "Provision this separate environment explicitly; see docs/thor_fr3_teleoperation.md." >&2
  exit 1
fi

# Placo/Pinocchio wheels put native dependencies in cmeel.prefix rather than a
# global library path. The child launched by F uses the same runtime directory.
cmeel_lib="$("$runtime_python" - <<'PY'
from pathlib import Path
import site
for directory in site.getsitepackages():
    candidate = Path(directory) / 'cmeel.prefix' / 'lib'
    if candidate.is_dir():
        print(candidate)
        break
PY
)"
if [[ -n "$cmeel_lib" ]]; then
  export LD_LIBRARY_PATH="$cmeel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
venv_absolute="$(cd "$venv_path" && pwd)"
export LD_LIBRARY_PATH="$venv_absolute/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Reject an unpatched wheel even when this machine's RT kernel still needs
# commissioning. Importing the extension does not instantiate a robot.
if ! "$runtime_python" - <<'PY'
import sys
try:
    from panda_py import _core
    if getattr(_core, 'FR3_NO_AUTOMATIC_ERROR_RECOVERY', None) is not True:
        raise RuntimeError('panda-py native extension lacks FR3_NO_AUTOMATIC_ERROR_RECOVERY=True')
except Exception as exc:
    print(f'ERROR: patched panda-py capability check failed: {exc}', file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "Use run/patch_thor_panda_py.py on the inspected source and rebuild/install its host-native wheel." >&2
  exit 1
fi

echo "==> Checking local FR3 runtime (no robot connection or motion)..."
exec env PYTHONPATH=src:. "$runtime_python" -m tools.thor.fr3_control_worker \
  --check --config-path "$config_path"
