#!/usr/bin/env bash
# Run the GPU services agent (tools/fr3/gpu_services_agent.py) on this policy host as a user unit.
#
# Usage:
#   bash run/gpu_services_agent.sh start    # no-op if running
#   bash run/gpu_services_agent.sh restart  # picks up new code or a new outputs/gpu_services.json
#   bash run/gpu_services_agent.sh stop
#   bash run/gpu_services_agent.sh status
#
# The agent lets the Rollout page start, stop and switch the services on this machine's GPU (the
# policy server, and whatever outputs/gpu_services.json lists). Restarting it never stops one of
# them: each runs in a unit of its own. `bash run/deploy.sh nuc` restarts it on the policy host.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
unit="fr3-gpu-services-agent"
log_path="${repo_root}/run/logs/gpu_services_agent.out"

running() {
  [[ "$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)" == "active" ]]
}

start_agent() {
  if running; then
    echo "gpu services agent already running ($(systemctl --user show -p MainPID --value "${unit}.service"))"
    return 0
  fi
  if [[ ! -x .venv-fr3/bin/python ]]; then
    echo "ERROR: no .venv-fr3 in ${repo_root}" >&2
    exit 1
  fi
  mkdir -p run/logs
  systemctl --user reset-failed "${unit}.service" 2>/dev/null || true
  systemd-run --user --quiet --collect --unit="$unit" --working-directory="$repo_root" \
    -p Restart=on-failure -p RestartSec=5 -p MemoryMax=2G \
    -p "StandardOutput=append:${log_path}" -p "StandardError=append:${log_path}" \
    --setenv PYTHONPATH=src:. --setenv PYTHONUNBUFFERED=1 --setenv HF_HUB_OFFLINE=1 \
    .venv-fr3/bin/python tools/fr3/gpu_services_agent.py
  for _ in $(seq 1 20); do
    ss -ltnH "sport = :8792" 2>/dev/null | grep -q . && break
    sleep 0.25
  done
  if ! running; then
    echo "ERROR: gpu services agent did not stay up" >&2
    tail -20 "$log_path" >&2 || true
    exit 1
  fi
  echo "gpu services agent running (unit ${unit}), log ${log_path}"
}

stop_agent() {
  systemctl --user stop "${unit}.service" 2>/dev/null || true
  echo "gpu services agent stopped"
}

case "${1:-status}" in
  start) start_agent ;;
  stop) stop_agent ;;
  restart) stop_agent; start_agent ;;
  status)
    if running; then
      echo "gpu services agent running ($(systemctl --user show -p MainPID --value "${unit}.service"))"
    else
      echo "gpu services agent not running"
    fi
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
