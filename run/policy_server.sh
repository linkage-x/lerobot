#!/usr/bin/env bash
# Start, stop or inspect the policy server on this (GPU) machine, from this checkout.
#
# Usage:
#   bash run/policy_server.sh start   [--port 8790] [server flags...]   # no-op if already running
#   bash run/policy_server.sh restart [--port 8790] [server flags...]   # picks up new code
#   bash run/policy_server.sh ensure  [--port 8790] [server flags...]   # restart only if its code changed
#   bash run/policy_server.sh stop
#   bash run/policy_server.sh status
#
# The server loads whatever checkpoint a robot host asks for (see tools/fr3/fr3_policy_server.py),
# so it takes no checkpoint here. `bash run/deploy.sh nuc` runs `ensure --warm-last` on the policy
# host as part of a deploy: a server whose imported code is unchanged keeps running with whatever
# it holds, and one that is restarted loads the last policy again in the background.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

action="${1:-status}"
if [[ $# -gt 0 ]]; then
  shift
fi
port=8790
server_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port="$2"; server_args+=(--port "$2"); shift 2 ;;
    --port=*) port="${1#--port=}"; server_args+=("$1"); shift ;;
    *) server_args+=("$1"); shift ;;
  esac
done

log_path="run/logs/policy_server.log"
# Written by the server: every repo .py file it has imported, hashed when first seen.
code_manifest="run/logs/policy_server_code.json"

# The interpreter's argv, not a substring of the command line: `pgrep -f fr3_policy_server`
# also matches the shell running this script, an editor, or an ssh command naming the file.
server_pids() {
  python3 - <<'PY'
import os

SCRIPT = "tools/fr3/fr3_policy_server.py"
for name in os.listdir("/proc"):
    if not name.isdigit() or int(name) == os.getpid():
        continue
    try:
        args = [a.decode("utf-8", "ignore") for a in open(f"/proc/{name}/cmdline", "rb").read().split(b"\0") if a]
    except OSError:
        continue
    if args and os.path.basename(args[0]).startswith("python") and any(a.endswith(SCRIPT) for a in args[1:]):
        print(name)
PY
}

# Exit 0 when the server with pid $1 runs the code now on disk. Says what changed otherwise.
code_current() {
  python3 - "$1" "$code_manifest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pid, manifest_path = sys.argv[1], Path(sys.argv[2])
try:
    manifest = json.loads(manifest_path.read_text())
except (OSError, ValueError):
    print("no code record from the running server")
    sys.exit(1)
if str(manifest.get("pid")) != pid:
    print("the code record is from another server process")
    sys.exit(1)
changed = []
for rel, digest in manifest.get("files", {}).items():
    try:
        now = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    except OSError:
        now = None
    if now != digest:
        changed.append(rel)
if changed:
    more = f" (+{len(changed) - 3} more)" if len(changed) > 3 else ""
    print("changed since it started: " + ", ".join(sorted(changed)[:3]) + more)
    sys.exit(1)
PY
}

listening() {
  ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .
}

stop_server() {
  local pids
  pids="$(server_pids || true)"
  if [[ -z "$pids" ]]; then
    echo "policy server not running"
    return 0
  fi
  echo "$pids" | xargs -r kill 2>/dev/null || true
  for _ in $(seq 1 20); do
    [[ -z "$(server_pids || true)" ]] && break
    sleep 0.25
  done
  pids="$(server_pids || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
  echo "policy server stopped"
}

start_server() {
  local pids
  pids="$(server_pids || true)"
  if [[ -n "$pids" ]]; then
    echo "policy server already running (pid $(echo "$pids" | head -1)); use restart to pick up new code"
    return 0
  fi
  if [[ ! -x .venv-fr3/bin/python ]]; then
    echo "ERROR: no .venv-fr3 in ${repo_root}" >&2
    exit 1
  fi
  if listening; then
    echo "ERROR: port ${port} is already taken by something that is not the policy server:" >&2
    ss -ltnpH "sport = :${port}" >&2 || true
    exit 1
  fi
  mkdir -p run/logs
  echo "==== $(date -Iseconds) start ${server_args[*]:-}" >>"$log_path"
  # Offline hub: pi05's preprocessor builds the paligemma tokenizer, and without HF_HUB_OFFLINE the
  # hub tries huggingface.co before the local cache -- from here that connect hangs, inside the
  # hello, while the robot host waits. Set FR3_HF_HUB_OFFLINE=0 to fetch a tokenizer not cached yet.
  setsid env PYTHONPATH=src:. PYTHONUNBUFFERED=1 HF_HUB_OFFLINE="${FR3_HF_HUB_OFFLINE-1}" \
    .venv-fr3/bin/python tools/fr3/fr3_policy_server.py "${server_args[@]}" \
    </dev/null >>"$log_path" 2>&1 &
  disown
  # Binding comes before anything slow (torch import aside), so a server that is going to come up
  # is listening within seconds; one that is not has already said why in the log.
  for _ in $(seq 1 60); do
    listening && break
    [[ -z "$(server_pids || true)" ]] && break
    sleep 0.5
  done
  if ! listening; then
    echo "ERROR: policy server did not come up on port ${port}" >&2
    tail -20 "$log_path" >&2
    exit 1
  fi
  echo "policy server listening on :${port} (pid $(server_pids | head -1)), log ${repo_root}/${log_path}"
}

case "$action" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  ensure)
    pids="$(server_pids || true)"
    pid="$(echo "$pids" | head -1)"
    if [[ -n "$pid" ]] && listening && why="$(code_current "$pid")"; then
      echo "policy server (pid ${pid}) already runs this code; kept, with whatever policy it holds"
    else
      [[ -n "$pid" ]] && echo "restarting policy server: ${why:-not listening on :${port}}"
      stop_server
      start_server
    fi
    ;;
  status)
    pids="$(server_pids || true)"
    if [[ -n "$pids" ]]; then
      echo "policy server running (pid $(echo "$pids" | head -1)); listening on :${port}: $(listening && echo yes || echo no)"
      tail -3 "$log_path" 2>/dev/null || true
    else
      echo "policy server not running"
    fi
    ;;
  *)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
