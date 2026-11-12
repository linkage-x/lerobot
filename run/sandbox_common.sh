# Shared by run/deploy.sh, run/sync_to_target.sh and run/sandbox.sh. Source it, do not run it.
#
# Personal sandboxes: one directory per person on Thor, each with its own gateway port, so a
# branch can be tried on the real machine and the real data without touching production
# (the target's repo_dir, port 8765, which the students' GUI and every recording use).
# Every Thor target (run/targets/*.json with "sandboxes": true) has its own sandboxes; production
# there is the target's repo_dir.

# shellcheck source=run/targets_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/targets_common.sh"

THOR_SANDBOX_BASE="/home/nvidia/sandboxes"
SANDBOX_PORT_MIN=8770
SANDBOX_PORT_MAX=8799

# validate_sandbox_name NAME -> exits 2 with a reason when NAME is not usable.
validate_sandbox_name() {
  local name="$1"
  if [[ ! "$name" =~ ^[a-z][a-z0-9_-]{1,23}$ ]]; then
    echo "ERROR: sandbox name '$name' must be 2-24 characters: a lowercase letter, then lowercase letters, digits, - or _ (use your own name, e.g. zhangsan)" >&2
    exit 2
  fi
  case "$name" in
    lerobot|production|prod|main|box|thor|workstation)
      echo "ERROR: '$name' is reserved; name the sandbox after yourself" >&2
      exit 2
      ;;
  esac
}

sandbox_dir() {
  printf '%s/%s' "$THOR_SANDBOX_BASE" "$1"
}

# The commit-and-provenance stamp written next to the deployed tree as DEPLOYED.json. The gateway
# shows it in the GUI, so anyone can see which version (and whose) a gateway is running.
# deployed_release_json REPO_DIR SANDBOX_NAME_OR_EMPTY ON_BOX(true|false|null) MAINTAINER(true|false)
deployed_release_json() {
  local repo="$1" sandbox="$2" on_box="$3" maintainer="$4"
  local commit branch dirty who
  commit="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git -C "$repo" status --porcelain 2>/dev/null)" ]]; then dirty=true; else dirty=false; fi
  who="$(git -C "$repo" config user.name 2>/dev/null || true) ($(id -un)@$(hostname))"
  python3 - "$commit" "$branch" "$dirty" "$who" "$sandbox" "$on_box" "$maintainer" "${TARGET_NAME:-}" "${TARGET_STATION:-}" <<'PY'
import json, sys, datetime
commit, branch, dirty, who, sandbox, on_box, maintainer, target, station = sys.argv[1:]
flag = {"true": True, "false": False}
print(json.dumps({
    "commit": commit,
    "branch": branch,
    "dirty": flag[dirty],
    "onBox": flag.get(on_box),
    "maintainerOverride": flag[maintainer],
    "sandbox": sandbox or None,
    "target": target or None,
    "station": station or None,
    "by": who.strip(),
    "deployedUtc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, ensure_ascii=False, indent=2))
PY
}
