#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
BUILD_LIBFRANKA="${BUILD_LIBFRANKA:-1}"
WITH_PIKA_SDK="${WITH_PIKA_SDK:-0}"
WITH_GEN_CON_SDK="${WITH_GEN_CON_SDK:-0}"

LIBFRANKA_REF="${LIBFRANKA_REF:-0.15.0}"
LIBFRANKA_REPO="${LIBFRANKA_REPO:-https://github.com/frankarobotics/libfranka.git}"
PANDA_PY_REF="${PANDA_PY_REF:-main}"
PANDA_PY_REPO="${PANDA_PY_REPO:-https://github.com/linkage-x/panda-py.git}"
PIKA_SDK_REPO="${PIKA_SDK_REPO:-https://github.com/linkage-x/pika_sdk.git}"
GEN_CON_SDK_REPO="${GEN_CON_SDK_REPO:-https://github.com/genrobot-ai/gen_con_sdk_python_release.git}"

DEFAULT_HIROL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/HIROLRobotPlatform"
PANDA_PY_SRC="${PANDA_PY_SRC:-${DEFAULT_HIROL_ROOT}/dependencies/panda-py}"
PIKA_SDK_SRC="${PIKA_SDK_SRC:-${DEFAULT_HIROL_ROOT}/dependencies/pika_sdk}"
GEN_CON_SDK_SRC="${GEN_CON_SDK_SRC:-${DEFAULT_HIROL_ROOT}/dependencies/gen_con_sdk_python_release}"

APT_PACKAGES=(
  build-essential
  git
  curl
  cmake
  pkg-config
  ninja-build
  libglib2.0-0
  libegl1-mesa-dev
  ffmpeg
  libusb-1.0-0-dev
  speech-dispatcher
  libgeos-dev
  portaudio19-dev
  libeigen3-dev
  libpoco-dev
  libhidapi-dev
  libudev-dev
  libx11-dev
  freeglut3-dev
  libglu1-mesa-dev
  libssl-dev
  libboost-filesystem-dev
  libboost-system-dev
  libboost-test-dev
  libboost-serialization-dev
  liburdfdom-dev
)

log() {
  printf '[setup_host_env] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

pick_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

pick_existing_python_project() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
      if [[ -f "${candidate}/pyproject.toml" || -f "${candidate}/setup.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

install_system_deps() {
  log "Installing host system dependencies via apt"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
}

ensure_venv() {
  mkdir -p "${UV_CACHE_DIR}"
  if [[ ! -x .venv/bin/python ]]; then
    log "Creating .venv with ${PYTHON_BIN}"
    "${UV_BIN}" venv --python "${PYTHON_BIN}" .venv
  fi
}

sync_python_deps() {
  log "Syncing lerobot Python dependencies with uv"
  "${UV_BIN}" sync --python .venv/bin/python --extra kinematics
  log "Installing FR3 host runtime Python extras"
  "${UV_BIN}" pip install --python .venv/bin/python --no-cache cffi
  "${UV_BIN}" pip install --python .venv/bin/python --no-deps easyhid pyspacemouse "ruckig>=0.15,<0.16"
}

discover_cmeel_prefix() {
  find "${REPO_ROOT}/.venv/lib" -path '*/site-packages/cmeel.prefix' -type d | head -n 1
}

build_libfranka_if_needed() {
  if [[ "${BUILD_LIBFRANKA}" != "1" ]]; then
    return 0
  fi
  if [[ -f /usr/local/lib/libfranka.so ]]; then
    log "libfranka already present at /usr/local/lib/libfranka.so"
    return 0
  fi

  local cmeel_prefix cmake_bin tmpdir
  cmeel_prefix="$(discover_cmeel_prefix)"
  if [[ -z "${cmeel_prefix}" ]]; then
    printf 'Could not find cmeel.prefix after uv sync. Aborting libfranka build.\n' >&2
    exit 1
  fi

  cmake_bin="/usr/bin/cmake"
  if [[ ! -x "${cmake_bin}" ]]; then
    cmake_bin="$(command -v cmake)"
  fi

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  log "Cloning libfranka ${LIBFRANKA_REF}"
  git clone --recursive --depth 1 --branch "${LIBFRANKA_REF}" "${LIBFRANKA_REPO}" "${tmpdir}/libfranka"

  log "Configuring libfranka"
  "${cmake_bin}" -S "${tmpdir}/libfranka" -B "${tmpdir}/libfranka/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPython3_EXECUTABLE="${REPO_ROOT}/.venv/bin/python" \
    -DPython_EXECUTABLE="${REPO_ROOT}/.venv/bin/python" \
    -DCMAKE_PREFIX_PATH="${cmeel_prefix};/usr/local"

  log "Building libfranka"
  "${cmake_bin}" --build "${tmpdir}/libfranka/build" -j"$(nproc)"

  log "Installing libfranka to /usr/local"
  if ! sudo "${cmake_bin}" --install "${tmpdir}/libfranka/build"; then
    if [[ -f /usr/local/lib/libfranka.so ]]; then
      log "libfranka installed, ignoring trailing CMake install failure from bundled dependencies"
    else
      printf 'libfranka install failed before /usr/local/lib/libfranka.so became available.\n' >&2
      exit 1
    fi
  fi
  if [[ -d "${tmpdir}/libfranka/build/_deps/fmt-build" ]]; then
    sudo "${cmake_bin}" --install "${tmpdir}/libfranka/build/_deps/fmt-build" --prefix /usr/local
  fi
  sudo ldconfig
}

install_panda_py() {
  local panda_py_source cmeel_prefix
  cmeel_prefix="$(discover_cmeel_prefix)"
  if [[ -z "${cmeel_prefix}" ]]; then
    printf 'Could not find cmeel.prefix before installing panda_py.\n' >&2
    exit 1
  fi

  if CMAKE_PREFIX_PATH="${cmeel_prefix};/usr/local" \
     LD_LIBRARY_PATH="${cmeel_prefix}/lib:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
     .venv/bin/python - <<'PY' >/dev/null 2>&1
from panda_py import libfranka
assert libfranka.Gripper is not None
PY
  then
    log "panda_py with libfranka bindings already available in .venv"
    return 0
  fi

  panda_py_source="$(pick_existing_python_project "${PANDA_PY_SRC}")" || true
  if [[ -z "${panda_py_source}" ]]; then
    panda_py_source="git+${PANDA_PY_REPO}@${PANDA_PY_REF}"
  fi

  log "Installing panda_py from ${panda_py_source}"
  CMAKE_PREFIX_PATH="${cmeel_prefix};/usr/local" \
  LD_LIBRARY_PATH="${cmeel_prefix}/lib:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${UV_BIN}" pip install --python .venv/bin/python --no-cache "${panda_py_source}"
}

install_optional_sdk() {
  local flag="$1"
  local label="$2"
  local source_path="$3"
  local repo_url="$4"
  local target_path="$5"

  if [[ "${flag}" != "1" ]]; then
    return 0
  fi

  local resolved_source
  resolved_source="$(pick_existing_path "${source_path}")" || true
  if [[ -z "${resolved_source}" ]]; then
    resolved_source="${target_path}"
    if ! mkdir -p "${resolved_source}" 2>/dev/null; then
      resolved_source="${REPO_ROOT}/.host_deps/$(basename "${target_path}")"
      mkdir -p "$(dirname "${resolved_source}")"
    fi
    log "Cloning ${label} into ${resolved_source}"
    git clone --depth 1 "${repo_url}" "${resolved_source}"
  fi

  if [[ "${label}" == "pika_sdk" ]]; then
    log "Installing ${label} from ${resolved_source}"
    "${UV_BIN}" pip install --python .venv/bin/python --no-cache --no-deps "${resolved_source}"
  else
    log "Installing ${label} requirements from ${resolved_source}"
    "${UV_BIN}" pip install --python .venv/bin/python --no-cache -r "${resolved_source}/requirements.txt"
  fi
}

run_import_smoke() {
  local cmeel_prefix pythonpath_entries ld_entries
  cmeel_prefix="$(discover_cmeel_prefix)"
  pythonpath_entries=("${REPO_ROOT}/src" "/opt/MVS/Samples/64/Python" "/opt/MVS/Samples/32/Python")
  if [[ -n "${GEN_CON_SDK_HOME:-}" ]]; then
    pythonpath_entries+=("$(dirname "${GEN_CON_SDK_HOME}")")
  fi
  ld_entries=()
  if [[ -n "${cmeel_prefix}" ]]; then
    ld_entries+=("${cmeel_prefix}/lib")
  fi
  ld_entries+=("/opt/MVS/lib/64" "/opt/MVS/lib" "/usr/local/lib")

  log "Running host import smoke checks"
  PYTHONPATH="$(IFS=:; echo "${pythonpath_entries[*]}")" \
  LD_LIBRARY_PATH="$(IFS=:; echo "${ld_entries[*]}")" \
  HIKROBOT_MVS_HOME="/opt/MVS" \
  MVCAM_COMMON_RUNENV="/opt/MVS/lib" \
  .venv/bin/python - <<'PY'
import importlib
modules = [
    "placo",
    "panda_py",
    "ruckig",
    "pyspacemouse",
    "easyhid",
]
for name in modules:
    importlib.import_module(name)
    print(f"{name}=ok")
from MvImport import MvCameraControl_class as mvs
print(f"mvs=ok {mvs.__file__}")
PY
}

print_next_steps() {
  local cmeel_prefix
  cmeel_prefix="$(discover_cmeel_prefix)"
  cat <<EOF

Host environment prepared.

Use these environment variables when running host-side FR3 commands manually:

export PYTHONPATH="${REPO_ROOT}/src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python"
export LD_LIBRARY_PATH="${cmeel_prefix:+${cmeel_prefix}/lib:}/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib"
export HIKROBOT_MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

Then run:
  uv run --python .venv/bin/python python tools/fr3/fr3_record_preflight.py --config-path tools/fr3/fr3_record_hikrobot_example.yaml
  uv run --python .venv/bin/python python tools/fr3/fr3_record.py --config-path tools/fr3/fr3_record_hikrobot_example.yaml
EOF
}

require_cmd "${UV_BIN}"
require_cmd git
require_cmd sudo

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  install_system_deps
fi

ensure_venv
sync_python_deps
build_libfranka_if_needed
install_panda_py

install_optional_sdk "${WITH_PIKA_SDK}" "pika_sdk" "${PIKA_SDK_SRC}" "${PIKA_SDK_REPO}" "/opt/dependencies/pika_sdk"

if [[ "${WITH_GEN_CON_SDK}" == "1" ]]; then
  GEN_CON_SDK_HOME="$(pick_existing_path "${GEN_CON_SDK_SRC}" "/opt/dependencies/gen_con_sdk_python_release" || true)"
  if [[ -z "${GEN_CON_SDK_HOME}" ]]; then
    GEN_CON_SDK_HOME="/opt/dependencies/gen_con_sdk_python_release"
  fi
  export GEN_CON_SDK_HOME
  install_optional_sdk "1" "gen_con_sdk_python_release" "${GEN_CON_SDK_HOME}" "${GEN_CON_SDK_REPO}" "/opt/dependencies/gen_con_sdk_python_release"
fi

run_import_smoke
print_next_steps
