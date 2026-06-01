#!/usr/bin/env bash
# Recover Thor SG16A/AR0234 Argus capture without rebooting the Jetson.
#
# This is intentionally a stronger reset than just restarting nvargus-daemon:
# it stops stale recorder/GStreamer processes, stops Argus, reloads the camera
# kernel modules via setup_sync.sh, restarts Argus, then probes each requested
# sensor-id with nvarguscamerasrc.
#
# Typical use on Thor, from the lerobot repo root:
#   ./tools/thor/gmsl2/recover_argus.sh
#
# If your SDK is not vendored under tools/thor/gmsl2/sdk:
#   ./tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SETUP_SYNC="${SCRIPT_DIR}/setup_sync.sh"
LOCK_CHECK="${SCRIPT_DIR}/check_max96726_locks.sh"

SDK_DIR="${SCRIPT_DIR}/sdk"
FPS=60
NUM=16
SIDS=""
PROBE_TIMEOUT_S=12
PROBE_BUFFERS=60
WIDTH=1920
HEIGHT=1080
TRIG_PIN="0x00020007"
SENSOR_MODE=0
EXPOSURE_US=0
GAIN=0
SKIP_KILL=0
SKIP_SETUP=0
SKIP_PROBE=0
RETRY_FAILED=1

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --sdk DIR              SDK dir for setup_sync.sh (default: ${SDK_DIR})
  --fps N                PWM/probe framerate (default: ${FPS})
  --num N                Number of /dev/videoN entries configured by setup_sync.sh (default: ${NUM})
  --sids CSV             Sensor IDs to probe. Default: locked IDs from check_max96726_locks.sh,
                         falling back to 0,2,3,4,5,7,9,10,11,14,15.
  --probe-timeout SEC    Timeout per nvarguscamerasrc probe (default: ${PROBE_TIMEOUT_S})
  --probe-buffers N      num-buffers per probe (default: ${PROBE_BUFFERS})
  --width N              Probe width (default: ${WIDTH})
  --height N             Probe height (default: ${HEIGHT})
  --trig-pin HEX         Trigger pin passed to setup_sync.sh (default: ${TRIG_PIN})
  --sensor-mode N        Sensor mode passed to setup_sync.sh/probe (default: ${SENSOR_MODE})
  --exposure-us N        Exposure control applied before probe; 0 disables it (default: ${EXPOSURE_US})
  --gain N               Gain control applied before probe; 0 disables it (default: ${GAIN})
  --skip-kill            Do not terminate stale recorder/gst-launch processes.
  --skip-setup-sync      Do not reload camera modules or reapply PWM/v4l2 controls.
  --skip-probe           Do not run per-sid nvarguscamerasrc probes.
  --no-retry-failed      Do not restart Argus and retry failed probe IDs once.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk) SDK_DIR="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --num) NUM="$2"; shift 2 ;;
    --sids) SIDS="$2"; shift 2 ;;
    --probe-timeout) PROBE_TIMEOUT_S="$2"; shift 2 ;;
    --probe-buffers) PROBE_BUFFERS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --trig-pin) TRIG_PIN="$2"; shift 2 ;;
    --sensor-mode) SENSOR_MODE="$2"; shift 2 ;;
    --exposure-us) EXPOSURE_US="$2"; shift 2 ;;
    --gain) GAIN="$2"; shift 2 ;;
    --skip-kill) SKIP_KILL=1; shift ;;
    --skip-setup-sync) SKIP_SETUP=1; shift ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    --no-retry-failed) RETRY_FAILED=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "${SETUP_SYNC}" ]]; then
  echo "ERROR: missing executable setup_sync.sh at ${SETUP_SYNC}" >&2
  exit 1
fi
if [[ ! -x "${LOCK_CHECK}" ]]; then
  echo "ERROR: missing executable check_max96726_locks.sh at ${LOCK_CHECK}" >&2
  exit 1
fi

run_sudo() {
  if [[ ${EUID} -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

split_csv() {
  local csv="$1"
  csv="${csv//,/ }"
  printf '%s\n' ${csv}
}

print_step() {
  echo
  echo "==> $*"
}

if [[ ${SKIP_KILL} -eq 0 ]]; then
  print_step "Stopping stale recorder/GStreamer processes"
  # These processes own Argus sessions and can keep nvargus-daemon wedged after
  # a failed connect. Ignore misses so the script is idempotent.
  pkill -TERM -f 'python.*tools\.thor\.gmsl2\.thor_record' 2>/dev/null || true
  pkill -TERM -f 'python.*tools\.data_collection_gui\.gateway' 2>/dev/null || true
  pkill -TERM -x gst-launch-1.0 2>/dev/null || true
  sleep 1
  pkill -KILL -f 'python.*tools\.thor\.gmsl2\.thor_record' 2>/dev/null || true
  pkill -KILL -x gst-launch-1.0 2>/dev/null || true
else
  print_step "Skipping stale process cleanup"
fi

print_step "Stopping nvargus-daemon"
run_sudo service nvargus-daemon stop || true
sleep 1

if [[ ${SKIP_SETUP} -eq 0 ]]; then
  print_step "Reloading camera modules and reapplying hardware sync"
  run_sudo "${SETUP_SYNC}" \
    --sdk "${SDK_DIR}" \
    --fps "${FPS}" \
    --num "${NUM}" \
    --trig-pin "${TRIG_PIN}" \
    --sensor-mode "${SENSOR_MODE}"
else
  print_step "Skipping setup_sync.sh"
fi

print_step "Starting nvargus-daemon"
run_sudo service nvargus-daemon start || run_sudo service nvargus-daemon restart
sleep 2
run_sudo service nvargus-daemon status --no-pager || true

if [[ -z "${SIDS}" && ${SKIP_PROBE} -eq 0 ]]; then
  print_step "Detecting locked MAX96726 video IDs"
  lock_output="$("${LOCK_CHECK}" || true)"
  echo "${lock_output}"
  SIDS="$(printf '%s\n' "${lock_output}" | awk -F= '/^LOCKED_VIDEO_IDS=/{print $2}' | tail -n1)"
  if [[ -z "${SIDS}" ]]; then
    echo "WARNING: lock check did not return locked IDs; using known 11-camera baseline" >&2
    SIDS="0,2,3,4,5,7,9,10,11,14,15"
  fi
fi

if [[ ${SKIP_PROBE} -eq 1 ]]; then
  print_step "Skipping probes"
  echo "Recovery actions completed. Reconnect the GUI/recorder and watch nvargus logs if capture still fails."
  exit 0
fi

apply_controls() {
  local csv="$1"
  print_step "Applying per-camera controls to sensor IDs: ${csv}"
  for sid in $(split_csv "${csv}"); do
    [[ -z "${sid}" ]] && continue
    ctrls="sensor_mode=${SENSOR_MODE},trig_pin=${TRIG_PIN},trig_mode=1"
    if [[ "${EXPOSURE_US}" != "0" ]]; then
      ctrls="${ctrls},exposure=${EXPOSURE_US}"
    fi
    if [[ "${GAIN}" != "0" ]]; then
      ctrls="${ctrls},gain=${GAIN}"
    fi
    echo "-- v4l2-ctl /dev/video${sid} -c ${ctrls}"
    run_sudo v4l2-ctl -d "/dev/video${sid}" -c "${ctrls}" || true
  done
  sleep 1
}

probe_sids() {
  local csv="$1"
  current_ok=()
  current_fail=()
  print_step "Probing nvarguscamerasrc sensor IDs: ${csv}"
  mkdir -p /tmp/thor_argus_recover
  for sid in $(split_csv "${csv}"); do
    [[ -z "${sid}" ]] && continue
    log="/tmp/thor_argus_recover/sid_${sid}.log"
    echo "-- probe sid=${sid} (log: ${log})"
    if timeout "${PROBE_TIMEOUT_S}" gst-launch-1.0 -q \
        nvarguscamerasrc sensor-id="${sid}" sensor-mode="${SENSOR_MODE}" num-buffers="${PROBE_BUFFERS}" \
        ! "video/x-raw(memory:NVMM),format=NV12,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" \
        ! fakesink sync=false async=false >"${log}" 2>&1; then
      echo "PROBE_OK sid=${sid}"
      current_ok+=("${sid}")
    else
      rc=$?
      echo "PROBE_FAIL sid=${sid} rc=${rc}"
      tail -n 20 "${log}" || true
      current_fail+=("${sid}")
    fi
  done
}

join_csv() {
  local IFS=,
  echo "$*"
}

apply_controls "${SIDS}"
probe_sids "${SIDS}"
ok=("${current_ok[@]}")
fail=("${current_fail[@]}")

if [[ ${#fail[@]} -gt 0 && ${RETRY_FAILED} -eq 1 ]]; then
  retry_csv="$(join_csv "${fail[@]}")"
  print_step "Retrying failed sensor IDs after nvargus-daemon restart: ${retry_csv}"
  run_sudo service nvargus-daemon restart
  sleep 2
  apply_controls "${retry_csv}"
  probe_sids "${retry_csv}"

  retry_ok_csv="$(join_csv "${current_ok[@]}")"
  remaining_fail=("${current_fail[@]}")
  merged_ok=()
  for sid in "${ok[@]}" "${current_ok[@]}"; do
    [[ -z "${sid}" ]] && continue
    merged_ok+=("${sid}")
  done
  ok=("${merged_ok[@]}")
  fail=("${remaining_fail[@]}")
fi

echo
printf 'RECOVER_OK_SIDS='
(IFS=,; echo "${ok[*]}")
printf 'RECOVER_FAIL_SIDS='
(IFS=,; echo "${fail[*]}")
echo "Probe logs: /tmp/thor_argus_recover/sid_<sid>.log"

if [[ ${#fail[@]} -eq 0 ]]; then
  echo "Recovery succeeded: all probed sensors opened through Argus."
  exit 0
fi

cat <<'EOF'
Recovery completed but some sensors still fail. Check recent kernel signatures:
  dmesg -T | tail -n 200 | egrep 'NvBufSurfaceFromFd|dmabuf_fd|max96726|ar0234|streaming|vi|nvcsi'

If failures include NvBufSurfaceFromFd/dmabuf_fd -1 plus max96726/ar0234 stream-on
errors, this is below Python/GStreamer orchestration. Capture dmesg and the
/tmp/thor_argus_recover logs for supplier/driver debugging.
EOF
exit 1
