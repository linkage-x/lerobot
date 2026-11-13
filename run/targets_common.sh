# Deployment targets for the run/ scripts. Source it, do not run it.
#
# Machines live in run/targets/<name>.json (one file per machine, committed); how you reach one
# from your own computer can be overridden in the git-ignored run/targets.local.json. Nothing in
# run/ may hard-code an address: add or change a machine there. See run/targets/README.md.

_targets_run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# resolve_target NAME_OR_ALIAS -> sets TARGET_NAME TARGET_STATION TARGET_SSH TARGET_ADDRESS
# TARGET_HOSTNAME TARGET_DIR TARGET_PROFILE TARGET_CONFIG TARGET_PORT TARGET_GATEWAY_URL
# TARGET_SANDBOXES TARGET_CALIBRATION_IN_TREE TARGET_BOX_NET_ADDR TARGET_OVERRIDDEN; exits 2 for an unknown target.
resolve_target() {
  local assignments
  assignments="$(python3 "$_targets_run_dir/targets.py" shell "$1")" || exit 2
  eval "$assignments"
}

# require_target_name WORD -> exits 2 listing the targets unless WORD names one. There is no default
# target: with more than one machine, a forgotten name must not quietly restart thor1's gateway.
require_target_name() {
  if [[ -z "${1:-}" || "${1:-}" == -* ]]; then
    echo "ERROR: name the machine; there is no default target. Known targets (run/targets/):" >&2
    python3 "$_targets_run_dir/targets.py" list >&2 || true
    exit 2
  fi
}

target_ssh() {
  ssh -o ConnectTimeout=5 "$TARGET_SSH" "$@"
}

# require_target_identity -> exits 1 unless the machine behind TARGET_SSH really is TARGET_NAME.
# Run it before anything writes to or stops something on the target: an address that DHCP gave to
# another machine, a cloned image, or a wrong ssh alias must stop here, not after the rsync.
require_target_identity() {
  local actual
  if ! actual="$(target_ssh hostname 2>/dev/null)"; then
    echo "ERROR: cannot reach ${TARGET_NAME} over ssh (${TARGET_SSH})." >&2
    echo "       From another LAN, point run/targets.local.json at an ~/.ssh/config alias (see run/targets/README.md)." >&2
    exit 1
  fi
  actual="${actual//[$'\r\n']/}"
  if [[ "$actual" != "$TARGET_HOSTNAME" ]]; then
    echo "ERROR: ${TARGET_SSH} is the machine '${actual}', but ${TARGET_NAME} is '${TARGET_HOSTNAME}' (run/targets/${TARGET_NAME}.json)." >&2
    echo "       Nothing was changed. Either the address now belongs to another machine (DHCP, a clone)," >&2
    echo "       or the hostname in the target file is stale; fix whichever is wrong before deploying." >&2
    exit 1
  fi
  echo "==> target ${TARGET_NAME}${TARGET_STATION:+ (${TARGET_STATION})}: ${TARGET_SSH} is ${actual}${TARGET_OVERRIDDEN:+, local override: ${TARGET_OVERRIDDEN}}"
}
