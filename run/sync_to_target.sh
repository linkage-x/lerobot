#!/usr/bin/env bash
# Incrementally replace repository files on a deployment target.
#
# Usage:
#   bash run/sync_to_target.sh [thor|workstation] [rsync options...]
#
# Reports -- loudly, but without blocking -- when the target already has a NEWER
# sidecar schema than this working tree. Mirroring an old checkout onto a
# deployment target is a silent, unreported data-loss path (see the gate below);
# what this check buys is that it stops being silent. Pass --refuse-downgrade to
# turn the warning into a hard failure.
set -euo pipefail

target="${1:-thor}"
if [[ $# -gt 0 ]]; then
  shift
fi

# The default is to proceed. This script is also how a broken target gets fixed,
# and a gate that stands between an operator and the fix gets worked around
# rather than heeded -- at which point it protects nothing and the warning is
# gone too. So the check keeps its volume and loses its veto: the failure mode it
# exists to prevent is the *silent* downgrade, not the downgrade.
# --refuse-downgrade restores the veto (for CI, or a deploy nobody is watching).
# --allow-downgrade is accepted and ignored; it is now what happens anyway.
refuse_downgrade=0
rsync_args=()
for arg in "$@"; do
  case "$arg" in
    --refuse-downgrade) refuse_downgrade=1 ;;
    --allow-downgrade) ;;  # the default now; swallowed so it never reaches rsync
    *) rsync_args+=("$arg") ;;
  esac
done

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

# --- Version monotonicity gate (roadmap P0-Now (7)) --------------------------
#
# On 2026-08-19 16:03 an rsync from an old checkout reverted Thor's
# third_party/opencv_kalibr (tracker back 3532 lines, 8 CLI modules gone). The
# damage was not that one run: it was that nothing anywhere reported it. A Thor
# rolled back to an older tracker keeps recording, and writes untagged v1
# sidecars over v2 ones without a single error on either side, so the loss shows
# up weeks later as a dataset that cannot be aggregated and cannot be re-derived.
#
# SIDECAR_SCHEMA_VERSION is the proxy for "which tracker is over there". It is a
# single integer that only ever goes up, it lives in the tree this script
# mirrors, and it is exactly the contract the downstream aggregate gate reads.
#
# Note what this check is and is not. It makes the downgrade *visible* at the
# moment it happens; it does not prevent it, and by design it never stands
# between an operator and a target that needs fixing. It is also only half the
# exposure: it governs who may write to the target, not the older code already
# sitting there being re-run.
schema_file="third_party/opencv_kalibr/metrology/sidecar_schema.py"
extract_schema_version="grep -E '^SIDECAR_SCHEMA_VERSION[[:space:]]*=' '$remote_dir/$schema_file' 2>/dev/null | head -n 1 | sed -E 's/.*=[[:space:]]*([0-9]+).*/\\1/'"

local_schema_version="$(grep -E '^SIDECAR_SCHEMA_VERSION[[:space:]]*=' "$local_dir$schema_file" 2>/dev/null | head -n 1 | sed -E 's/.*=[[:space:]]*([0-9]+).*/\1/' || true)"
if [[ -z "$local_schema_version" ]]; then
  echo "ERROR: cannot read SIDECAR_SCHEMA_VERSION from $local_dir$schema_file" >&2
  echo "       This working tree is not a source this script can safely mirror from." >&2
  exit 3
fi

echo "==> Preparing ${remote}:${remote_dir}"
remote_probe="$(ssh -o ConnectTimeout=5 "$remote" "mkdir -p '$remote_dir'; $extract_schema_version")"
remote_schema_version="$(printf '%s' "$remote_probe" | tr -d '[:space:]')"

if [[ -z "$remote_schema_version" ]]; then
  echo "    sidecar schema: local v${local_schema_version}, target has no ${schema_file} (first deploy)"
elif ! [[ "$remote_schema_version" =~ ^[0-9]+$ ]]; then
  echo "WARNING: ${target} has ${schema_file} but its SIDECAR_SCHEMA_VERSION is unreadable" >&2
  echo "         (got '${remote_schema_version}'), so this sync cannot tell whether it is a" >&2
  echo "         downgrade. Inspect the target if that is a surprise." >&2
  if [[ "$refuse_downgrade" == "1" ]]; then
    echo "         --refuse-downgrade given; refusing." >&2
    exit 4
  fi
elif (( remote_schema_version > local_schema_version )); then
  echo "WARNING: this DOWNGRADES ${target}." >&2
  echo "         ${target} has sidecar schema v${remote_schema_version}; this working tree is v${local_schema_version}." >&2
  echo "         This is the shape of the 2026-08-19 accident: the target keeps recording and" >&2
  echo "         writes older sidecars over newer ones with no error on either side. The" >&2
  echo "         difference now is that you were told." >&2
  if [[ "$refuse_downgrade" == "1" ]]; then
    echo "         --refuse-downgrade given; refusing." >&2
    exit 5
  fi
  echo "         Proceeding (default). If it was not deliberate: pull/rebase this checkout" >&2
  echo "         and deploy again, which restores the target to the newer tree." >&2
else
  echo "    sidecar schema: local v${local_schema_version} >= target v${remote_schema_version}, ok"
fi

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
  ${rsync_args[@]+"${rsync_args[@]}"} \
  "$local_dir" "${remote}:${remote_dir}/"
