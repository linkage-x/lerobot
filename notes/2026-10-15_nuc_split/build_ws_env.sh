#!/usr/bin/env bash
# New FR3 workstation env: base extras -> libfranka 0.15.0 (against the venv's cmeel pinocchio) -> full setup.
set -euo pipefail
cd /home/hph/Code/lerobot
export PATH="$HOME/.local/bin:$PATH"
echo "==> [1/4] apt deps for libfranka"
sudo apt-get install -y --no-install-recommends git pkg-config libssl-dev libboost-filesystem-dev \
  libboost-system-dev libboost-serialization-dev libfmt-dev >/dev/null
echo "==> [2/4] uv sync without fr3-host (pulls cmeel pinocchio via kinematics)"
UV_PROJECT_ENVIRONMENT=.venv-fr3 uv sync --python 3.12 --extra fr3-workstation-teleop --extra fr3-train --extra kinematics --no-dev
cmeel_prefix="$(find .venv-fr3/lib -path '*/site-packages/cmeel.prefix' -type d | head -n 1)"
echo "cmeel_prefix=$cmeel_prefix"
[[ -n "$cmeel_prefix" ]]
cmeel_prefix="$(realpath "$cmeel_prefix")"
echo "==> [3/4] libfranka 0.15.0"
if [[ ! -f /usr/local/lib/libfranka.so ]]; then
  tmpdir="$(mktemp -d)"
  git clone --recursive --depth 1 --branch 0.15.0 https://github.com/frankarobotics/libfranka.git "$tmpdir/libfranka"
  cmake -S "$tmpdir/libfranka" -B "$tmpdir/libfranka/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DPython3_EXECUTABLE="$PWD/.venv-fr3/bin/python" \
    -DPython_EXECUTABLE="$PWD/.venv-fr3/bin/python" -DCMAKE_PREFIX_PATH="${cmeel_prefix};/usr/local"
  cmake --build "$tmpdir/libfranka/build" -j"$(nproc)"
  sudo cmake --install "$tmpdir/libfranka/build" || [[ -f /usr/local/lib/libfranka.so ]]
  if [[ -d "$tmpdir/libfranka/build/_deps/fmt-build" ]]; then
    sudo cmake --install "$tmpdir/libfranka/build/_deps/fmt-build" --prefix /usr/local
  fi
  sudo ldconfig
fi
ls -la /usr/local/lib/libfranka.so*
echo "==> [4/4] full workstation setup (adds fr3-host: panda-python, pika)"
export CMAKE_PREFIX_PATH="${cmeel_prefix};/usr/local"
export LD_LIBRARY_PATH="${cmeel_prefix}/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash tools/fr3/setup_workstation_teleop_env.sh
echo "BUILD_WS_ENV_DONE"
