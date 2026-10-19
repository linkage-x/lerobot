#!/usr/bin/env bash
# cyclictest matrix on the NUC: governor x background load x C-state policy. Moves nothing.
#
#   bash rt_bench.sh OUT_DIR [DURATION_S] [CASES...]
#
# A case is GOVERNOR:LOAD:CSTATE -- GOVERNOR powersave|performance, LOAD idle|stress, CSTATE
# free|held. "free" passes --laptop so cyclictest does NOT hold /dev/cpu_dma_latency at 0: that
# is what libfranka sees (it holds no PM QoS), and on this box the ACPI C2/C3 exit latencies are
# 253/1048 us. "held" is cyclictest's default and shows what pinning C-states would buy.
# The stress load is confined to the housekeeping cores (0-3,8-11), like the gateway, cameras and
# policy client would be. The governor is restored on exit.
set -euo pipefail

out_dir="$1"
duration="${2:-60}"
shift 2 || shift $#
cases=("$@")
if ((${#cases[@]} == 0)); then
  cases=(
    powersave:idle:free powersave:stress:free
    performance:idle:free performance:stress:free
    powersave:stress:held performance:stress:held
  )
fi
mkdir -p "$out_dir"

original_governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
restore() { sudo cpupower -c all frequency-set -g "$original_governor" >/dev/null; }
trap restore EXIT

for case in "${cases[@]}"; do
  IFS=: read -r governor load cstate <<<"$case"
  sudo cpupower -c all frequency-set -g "$governor" >/dev/null
  sleep 2
  stress_pid=""
  if [[ "$load" == "stress" ]]; then
    (cd "$out_dir" && exec taskset -c 0-3,8-11 stress-ng --cpu 8 --vm 2 --vm-bytes 512M --io 2 \
      --timeout "$((duration + 5))s" --quiet) &
    stress_pid=$!
    sleep 2
  fi
  extra=()
  [[ "$cstate" == "free" ]] && extra+=(--laptop)
  echo "==> $case ($(date +%T))"
  sudo cyclictest -m -p 90 -i 1000 -D "$duration" -a 0-15 -t 16 -h 1200 -q "${extra[@]}" \
    --json="$out_dir/$governor-$load-$cstate.json" >/dev/null
  if [[ -n "$stress_pid" ]]; then
    wait "$stress_pid" 2>/dev/null || true
  fi
done
echo "==> done, governor back to $original_governor"
