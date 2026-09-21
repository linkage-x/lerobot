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
# turn every one of these warnings -- including "could not tell" -- into a hard
# failure.
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
  # Almost always an uninitialised submodule, not a corrupt checkout: the file
  # lives in third_party/opencv_kalibr, and a colleague who has never run
  # `git submodule update --init` has an empty directory there. Refusing would
  # take the deploy path away from exactly the person least equipped to work out
  # why -- and deploying from such a tree is not the 2026-08-19 accident, which
  # was an *older* tracker overwriting a newer one, not a missing one. So this
  # says what it could not check and carries on, and --refuse-downgrade (CI, or
  # an unattended deploy) still turns the unknown into a refusal.
  echo "WARNING: cannot read SIDECAR_SCHEMA_VERSION from $local_dir$schema_file" >&2
  echo "         (most likely third_party/opencv_kalibr is not checked out: run" >&2
  echo "         'git submodule update --init --recursive'). This sync therefore cannot" >&2
  echo "         tell whether it downgrades ${target}." >&2
  if [[ "$refuse_downgrade" == "1" ]]; then
    echo "         --refuse-downgrade given; refusing." >&2
    exit 3
  fi
fi

# --- Calibration pointer drift ----------------------------------------------
#
# The two keys below are what production reads to decide which calibration run
# it loads, and the GUI's promotion step rewrites them *on the target*. This
# script mirrors the same file from here, so a deploy from a checkout that never
# saw that promotion puts the old run name back -- silently, and with exactly
# the consequence the promotion existed to prevent (2026-08-20: production kept
# loading a calibration that was missing a moved camera, for seven days).
#
# Warning only, never a refusal: changing these keys here and deploying is also
# the legitimate way to move production, and this script cannot tell the two
# apart. What it can do is name both values so the operator can.
tracking_config="third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml"
pointer_keys='^[[:space:]]*(intrinsics_run_name|fixed_camera_run_name)[[:space:]]*:'
extract_pointers="grep -hE '$pointer_keys' '$remote_dir/$tracking_config' 2>/dev/null"

# Normalisation lives here rather than in the remote command so the ssh payload
# stays a bare grep: comments off, whitespace and quotes out, sorted so the two
# sides compare as sets rather than as file order.
normalise_pointers() {
  # [[:blank:]] rather than [[:space:]]: the newlines are the record separator.
  sed -E 's/#.*$//; s/[[:blank:]]//g; s/["'"'"']//g' | grep -v '^$' | sort
}

echo "==> Preparing ${remote}:${remote_dir}"
# The trailing `true` is load-bearing: the probe ends in a bare grep, and on a
# target that has neither file yet that grep exits 1 -- which would make ssh
# fail, and `set -e` abort the deploy before it had done anything.
remote_probe="$(ssh -o ConnectTimeout=5 "$remote" "mkdir -p '$remote_dir'; $extract_schema_version; echo '---POINTERS---'; $extract_pointers; true")"
# `|| true` on both: with `set -o pipefail`, a grep that matches nothing (a
# target with neither file yet -- a first deploy) is a failed assignment, and
# `set -e` would turn that into an aborted deploy.
remote_schema_version="$(printf '%s\n' "$remote_probe" | sed -n '1,/^---POINTERS---$/p' | grep -v '^---POINTERS---$' | tr -d '[:space:]' || true)"
remote_pointers="$(printf '%s\n' "$remote_probe" | sed -n '/^---POINTERS---$/,$p' | grep -v '^---POINTERS---$' | normalise_pointers || true)"
local_pointers="$(grep -hE "$pointer_keys" "$local_dir$tracking_config" 2>/dev/null | normalise_pointers || true)"

if [[ -z "$local_schema_version" ]]; then
  echo "    sidecar schema: local unknown (see the warning above); target reports '${remote_schema_version:-none}'"
elif [[ -z "$remote_schema_version" ]]; then
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

if [[ -n "$remote_pointers" && -n "$local_pointers" && "$remote_pointers" != "$local_pointers" ]]; then
  echo "WARNING: ${target} points production at a different calibration than this tree does," >&2
  echo "         and this sync overwrites the target's ${tracking_config##*/}:" >&2
  echo "$remote_pointers" | sed "s/^/           ${target} has: /" >&2
  echo "$local_pointers" | sed 's/^/           this tree: /' >&2
  echo "         If the target's value came from a GUI promotion, this reverts it. Promote" >&2
  echo "         again after the deploy, or bring the value into this checkout first." >&2
elif [[ -z "$remote_pointers" && -n "$local_pointers" ]]; then
  echo "    calibration pointers: target has none yet; this sync installs this tree's"
elif [[ -z "$local_pointers" ]]; then
  echo "    calibration pointers: not declared in this tree, nothing to compare"
else
  echo "    calibration pointers: match"
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
