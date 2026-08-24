#!/usr/bin/env bash
# Build or extend a machine's uv-managed environment, non-interactively.
#
# This is what the Training page's install button runs, so it is written for a caller with no
# terminal: nothing here prompts, and nothing here uses sudo. A sudo prompt with stdin on
# /dev/null does not fail, it hangs -- and the operator would be looking at a modal that never
# finishes. System packages and device groups therefore stay where they were:
# setup_workstation_teleop_env.sh, run by a person at a shell.
#
# uv is the only way this installs anything. The environment it manages is a uv project
# environment, and mixing a `pip install` into one leaves packages uv's next sync does not know
# about and will happily resolve around.
#
# Two rules keep the button from breaking the machine it runs on:
#
#   1. `--inexact`. A plain `uv sync` makes the environment *exactly* the requested extras,
#      which on this workstation would uninstall mujoco, pyrealsense2 and pyspacemouse -- the
#      recorder's own dependencies -- as a side effect of installing transformers. `--inexact`
#      adds without removing.
#   2. The caller passes every extra the machine needs, not only the one it is short of. On the
#      recording workstation gateway.py hands the recorder `.venv-fr3/bin/python` the moment
#      that path exists, so an environment built here holding only the training extra would
#      stop recording on the next Connect. The gateway adds that machine's baseline extras to
#      whatever the page asked for; this script installs the union it is given.
#
# Zero extras is a valid request: torch, accelerate and wandb are base dependencies, so
# `uv sync` with no extra at all is exactly what a machine missing torch needs.
#
# Usage:  install_training_deps.sh [<extra> ...]
# Env:    VENV_PATH       target environment, relative to the repo root (default .venv-fr3)
#         PYTHON_VERSION  interpreter to build a *new* environment with (default 3.12)
#         UV_BIN          uv to use, when it is not on PATH

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

venv_path="${VENV_PATH:-.venv-fr3}"
python_version="${PYTHON_VERSION:-3.12}"

# Extras whose wheels are not wheels: Ruckig is built from source against the build dependency
# pinned in `[tool.uv.extra-build-dependencies]`, and uv only reads that table from 0.10.0. An
# older uv resolves the same extras and then fails in the build, several minutes in, saying
# nothing about why -- so the floor is raised up front when the plan includes one of these.
build_dep_extras=" fr3-host fr3_teleop "

extras=()
uv_min="0.5.0"
uv_min_reason="the --inexact flag"
for extra in "$@"; do
  if [[ ! "$extra" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$ ]]; then
    echo "ERROR: '$extra' is not a pyproject extra name." >&2
    exit 2
  fi
  extras+=(--extra "$extra")
  if [[ "$build_dep_extras" == *" $extra "* ]]; then
    uv_min="0.10.0"
    uv_min_reason="the Ruckig build dependencies '$extra' pulls in"
  fi
done

if [[ -n "${UV_BIN:-}" ]]; then
  uv_bin="$UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
else
  echo "ERROR: uv is not installed on this machine, and this project's environment is" >&2
  echo "       managed with uv. Install it (https://docs.astral.sh/uv/) and run this again." >&2
  exit 3
fi

# Note this floor is usually below the 0.10.0 setup_workstation_teleop_env.sh asks for. That
# script always installs the teleop extras, so it always needs the higher one; a training-only
# sync does not, and holding the button to 0.10.0 anyway would turn it off on a machine where
# it works.
uv_version="$("$uv_bin" --version | awk '{print $2}')"
if [[ "$(printf '%s\n' "$uv_min" "$uv_version" | sort -V | head -n1)" != "$uv_min" ]]; then
  echo "ERROR: uv>=$uv_min is required for $uv_min_reason (found $uv_version)." >&2
  echo "       Run: $uv_bin self update" >&2
  exit 3
fi

# `--python` only when there is nothing there yet. Passing it at an existing environment built
# on another minor version does not adopt that version, it *replaces* the environment -- which
# on the workstation would silently discard the recorder's install.
sync_args=(--inexact --no-dev --no-progress)
if [[ -x "$venv_path/bin/python" ]]; then
  mode="extending $venv_path ($("$venv_path/bin/python" --version 2>&1))"
else
  mode="building $venv_path from scratch with Python $python_version -- this downloads several GB"
  sync_args+=(--python "$python_version")
fi

echo "==> Repo:        $repo_root"
echo "==> Environment: $mode"
echo "==> uv:          $uv_bin $uv_version"
echo "==> Extras:      ${*:-(none -- base dependencies only, which is where torch lives)}"
echo "==> Running:     UV_PROJECT_ENVIRONMENT=$venv_path uv sync ${sync_args[*]} ${extras[*]}"
echo

# NO_COLOR and --no-progress: this output is read back through a log file and shown in a
# browser, where an escape sequence is a line of noise rather than a moving bar.
NO_COLOR=1 UV_PROJECT_ENVIRONMENT="$venv_path" "$uv_bin" sync "${sync_args[@]}" "${extras[@]}"

echo
echo "==> Done. Re-probing the machine will now report what changed."
