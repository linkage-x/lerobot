#!/usr/bin/env bash
# Run a long python job (training, an offline probe) on this GPU machine as its own systemd
# user unit, so it neither takes down nor is taken down by the terminal it was started from.
#
# Usage:
#   bash run/user_job.sh start <name> [--probe] [--need-gb N] [--log PATH] [--cwd DIR] -- <cmd...>
#   bash run/user_job.sh status <name>
#   bash run/user_job.sh stop <name>
#   bash run/user_job.sh peak <name>        # peak memory of the unit so far
#
# Why not `setsid nohup`: setsid leaves the job in the cgroup of the terminal's app scope. On
# 2026-10-22 10:49 a pi05 probe (35 GB) next to a trainer (~22 GB) ran Dante's 62 GB out; the
# kernel killed only the probe, but systemd's OOMPolicy=stop then stopped the whole VSCode scope,
# and the trainer, the agent session and every terminal in it went with it.
#
# --probe marks a side job: it gets MemoryMax = available RAM minus a reserve, so when it runs
# out it is killed inside its own cgroup and nothing else is touched, plus OOMScoreAdjust=1000.
# A pi05 load peaked at ~39 GB anon until 10-26 (pi_gemma built the language model three times
# over in fp32); it now peaks at ~16 GB, a whole holdout probe included, and the safetensors are
# read through mmap (page cache, which reclaim under MemoryMax drops first). It is refused when
# less than --need-gb (default 26, so the cap is >= 20 GB) is available. A job without --probe
# (a trainer) is uncapped and refused below --need-gb (default 28).
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
reserve_gb=6

usage() { sed -n '5,9p' "$0" >&2; exit 2; }

action="${1:-}"; name="${2:-}"
[[ -n "$action" && -n "$name" ]] || usage
shift 2
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name: $name" >&2; exit 2; }
unit="job-$name"

avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }

case "$action" in
  status)
    systemctl --user status "$unit" --no-pager -n 5 || true
    exit 0 ;;
  stop)
    systemctl --user stop "$unit"
    exit 0 ;;
  peak)
    systemctl --user show "$unit" -p MemoryPeak -p MemoryCurrent -p ActiveState -p Result
    exit 0 ;;
  start) ;;
  *) usage ;;
esac

probe=0; need_gb=""; log=""; cwd="$repo_root"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) probe=1; shift ;;
    --need-gb) need_gb="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    --cwd) cwd="$(cd "$2" && pwd)"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[[ $# -gt 0 ]] || usage
[[ -n "$need_gb" ]] || need_gb=$(( probe ? 26 : 28 ))
[[ -n "$log" ]] || log="$repo_root/run/logs/job_$name.log"
log="$(realpath -m "$log")"   # systemd wants an absolute path
mkdir -p "$(dirname "$log")"

if systemctl --user is-active -q "$unit"; then
  echo "$unit is already running; stop it first or pick another name" >&2
  exit 1
fi
systemctl --user reset-failed "$unit" 2>/dev/null || true
# A unit that has just exited lingers for a moment before --collect unloads it; starting the same
# name then fails with "was already loaded".
for _ in $(seq 20); do
  [[ "$(systemctl --user show "$unit" -p LoadState --value)" == not-found ]] && break
  sleep 0.5
done

avail=$(avail_gb)
if (( avail < need_gb )); then
  echo "refused: ${avail} GB available, $name needs ${need_gb} GB. Largest users:" >&2
  ps -eo pid,rss,etime,args --sort=-rss | head -6 | cut -c1-160 >&2
  exit 1
fi

props=(-p "StandardOutput=file:$log" -p "StandardError=file:$log")
if (( probe )); then
  props+=(-p "MemoryMax=$(( avail - reserve_gb ))G" -p OOMScoreAdjust=1000)
fi

systemd-run --user --unit="$unit" --working-directory="$cwd" --collect "${props[@]}" \
  -E HOME="$HOME" -E PATH="$PATH" -E PYTHONPATH="$cwd/src:$cwd" -E PYTHONUNBUFFERED=1 \
  -E HF_HUB_OFFLINE="${HF_HUB_OFFLINE-1}" -E WANDB_MODE="${WANDB_MODE-disabled}" \
  "$@"
echo "started $unit (log $log; ${avail} GB available$( (( probe )) && echo ", capped at $(( avail - reserve_gb )) GB"))"
