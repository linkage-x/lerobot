#!/usr/bin/env bash
# Stop the data-collection GUI gateway on Thor without starting a replacement.
#
# Usage:
#   bash run/kill_thor_gateway.sh          # send TERM, report any survivors
#   bash run/kill_thor_gateway.sh --force  # send TERM, then KILL survivors
set -euo pipefail

THOR="nvidia@192.168.111.122"
force=false
for arg in "$@"; do
  case "$arg" in
    --force) force=true ;;
    -h|--help)
      cat <<'EOF'
Stop the data-collection GUI gateway on Thor without starting a replacement.

Usage:
  bash run/kill_thor_gateway.sh          # send TERM, report any survivors
  bash run/kill_thor_gateway.sh --force  # send TERM, then KILL survivors
EOF
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

ssh -o ConnectTimeout=5 "$THOR" "FORCE=$force bash -s" <<'REMOTE'
set -euo pipefail

_gateway_pids() {
  python3 - <<'PY'
import os

MODULE = 'tools.data_collection_gui.gateway'
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    try:
        raw = open(f'/proc/{name}/cmdline', 'rb').read().split(b'\0')
    except OSError:
        continue
    args = [x.decode('utf-8', 'ignore') for x in raw if x]
    # Match on the interpreter + `-m <module>` pair rather than on argv[0]
    # being exactly python3: other branches launch the gateway through
    # .venv/bin/python (a symlink to python3) and may pass flags before -m.
    # A plain substring test would instead match greps, editors and pkill.
    if not args or not os.path.basename(args[0]).startswith('python'):
        continue
    if any(flag == '-m' and mod == MODULE for flag, mod in zip(args, args[1:])):
        print(name)
PY
}

pids="$(_gateway_pids || true)"
if [[ -z "$pids" ]]; then
  echo "No Thor gateway process is running."
  exit 0
fi

echo "Stopping Thor gateway pid(s): $(echo "$pids" | tr '\n' ' ')"
echo "$pids" | xargs -r kill 2>/dev/null || true
sleep 1

left="$(_gateway_pids || true)"
if [[ -z "$left" ]]; then
  echo "Thor gateway stopped."
  exit 0
fi

if [[ "${FORCE:-false}" == "true" ]]; then
  echo "Force killing remaining Thor gateway pid(s): $(echo "$left" | tr '\n' ' ')"
  echo "$left" | xargs -r kill -9 2>/dev/null || true
  sleep 1
  left="$(_gateway_pids || true)"
fi

if [[ -n "$left" ]]; then
  echo "ERROR: Thor gateway still running: $(echo "$left" | tr '\n' ' ')" >&2
  exit 1
fi

echo "Thor gateway stopped."
REMOTE
