#!/usr/bin/env bash
set -euo pipefail

# Local-on-Thor entry point for guided/manual and automatic P0 single-AprilTag
# calibration capture workflows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEACHING_SCRIPT="third_party/opencv_kalibr/fr3_calibration/teaching_pose_recorder.py"
TEACHING_CONFIG="third_party/opencv_kalibr/fr3_calibration/host/teaching_pose_recorder.host.yaml"
CAPTURE_SCRIPT="third_party/opencv_kalibr/fr3_calibration/batch_execute_pose_and_capture_thor_gmsl2.py"
CAPTURE_CONFIG="third_party/opencv_kalibr/fr3_calibration/host/execute_pose_and_capture_thor_gmsl2_apriltag.host.yaml"
RECORDER_CONFIG="third_party/opencv_kalibr/fr3_calibration/host/thor_gmsl2_calibration.yaml"
CAMERA_CONFIG_HELPER="tools/thor/prepare_p0_calibration_recorder_config.py"
GUIDED_SCRIPT="tools/thor/p0_two_marker_calibration.py"
FR3_URDF="src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
CURRENT_INTRINSICS="outputs/calibration/calib_20260929_163033_intrinsics/summary.json"

usage() {
  cat <<'EOF'
Run this script in a terminal on Thor itself.

Usage:
  bash tools/thor/run_p0_two_marker_calibration_local.sh guided \
    [--dataset-root ROOT ...] --execute --confirmation P0_TWO_MARKER_TEACHING \
    [-- guided calibration overrides]

  bash tools/thor/run_p0_two_marker_calibration_local.sh teaching \
    --key NAME --execute --confirmation P0_TWO_MARKER_TEACHING [-- recorder overrides]

  bash tools/thor/run_p0_two_marker_calibration_local.sh capture \
    --key NAME [--key NAME ...] [--input-json PATH] [--max-records N|all] \
    --execute --confirmation P0_TWO_MARKER_AUTOMATIC_CAPTURE [-- capture overrides]

  bash tools/thor/run_p0_two_marker_calibration_local.sh calibrate \
    --dataset-root ROOT [--dataset-root ROOT ...] [-- solver overrides]

  bash tools/thor/run_p0_two_marker_calibration_local.sh solve \
    --solve-run PATH|latest [--activate]

Modes:
  guided    Live multi-camera AprilTag UI plus zero-stiffness FR3 teaching.
            Enter commits images and measured robot state; q solves when all
            cameras reach the target. --dataset-root imports existing automatic
            captures first, so the UI displays accumulated valid/remaining counts.

  teaching  Put FR3 in teaching mode. Press r to save a robot pose and q to
            save/quit. No cameras are opened. Poses are written to
            outputs/datasets/NAME/teaching_pose_records.json.

  capture   Automatically execute the saved poses and record the current Thor
            GMSL2 camera set. One --key uses the teaching JSON above; --key may
            be repeated to combine several existing teaching trajectories.

  calibrate Offline import and robust solve of one or more same-layout automatic
            capture datasets. Repeating --dataset-root explicitly enables merge.

  solve     Re-solve an existing guided/manual session without cameras or FR3.
            Add --activate only after reviewing a passing candidate.

Wrapper options:
  --key NAME             Dataset/trajectory key. Required; repeatable in capture mode.
  --input-json PATH      Capture-mode pose source. Accepts a standalone
                         calibration manual_run_*/captures.json directly.
  --exclude-camera-ids   Comma-separated numeric IDs removed from the current
                         MAX96726 locked set (default: 1,4,10).
  --max-records N|all    Capture at most N saved poses (default: 30).
  --merged-root PATH     Capture-mode merged dataset output path.
  --dataset-root PATH    Existing automatic capture dataset. Repeat to combine
                         capture rounds in guided/calibrate mode.
  --execute              Required hardware authorization except calibrate mode.
  --confirmation TOKEN   Mode-specific confirmation token shown above.
  --dry-run              Print the resolved command without touching hardware.
  -h, --help             Show this help.

Arguments after --, and unrecognized arguments, are forwarded to the selected
Python tool. Examples include --robot.robot_ip=... and controller overrides.

Environment:
  PYTHON_BIN  Python containing panda_py/OpenCV/LeRobot. On Thor the launcher
              prefers /home/nvidia/Code/infer/.venv-fr3/bin/python3.
  P0_LOCKED_CAMERA_IDS
              Override hardware lock discovery for diagnostics/tests only.
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

print_command() {
  printf 'Resolved command:'
  printf ' %q' "$@"
  printf '\n'
}

has_forwarded_prefix() {
  local prefix="$1"
  local arg
  for arg in "${forwarded_args[@]}"; do
    if [[ "${arg}" == "${prefix}" || "${arg}" == "${prefix}="* ]]; then
      return 0
    fi
  done
  return 1
}

[[ $# -gt 0 ]] || {
  usage
  exit 2
}

mode="$1"
case "${mode}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  guided|teaching|capture|calibrate|solve) ;;
  *)
    die "unknown mode '${mode}'; expected guided, teaching, capture, calibrate, or solve"
    ;;
esac
shift

keys=()
dataset_roots=()
max_records="30"
merged_root=""
input_json=""
exclude_camera_ids="1,4,10"
execute=false
confirmation=""
dry_run=false
forwarded_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)
      [[ $# -ge 2 ]] || die "--key requires a value"
      keys+=("$2")
      shift 2
      ;;
    --key=*)
      keys+=("${1#*=}")
      shift
      ;;
    --max-records)
      [[ $# -ge 2 ]] || die "--max-records requires a value"
      max_records="$2"
      shift 2
      ;;
    --dataset-root)
      [[ $# -ge 2 ]] || die "--dataset-root requires a value"
      dataset_roots+=("$2")
      shift 2
      ;;
    --dataset-root=*)
      dataset_roots+=("${1#*=}")
      shift
      ;;
    --max-records=*)
      max_records="${1#*=}"
      shift
      ;;
    --merged-root)
      [[ $# -ge 2 ]] || die "--merged-root requires a value"
      merged_root="$2"
      shift 2
      ;;
    --merged-root=*)
      merged_root="${1#*=}"
      shift
      ;;
    --input-json)
      [[ $# -ge 2 ]] || die "--input-json requires a value"
      input_json="$2"
      shift 2
      ;;
    --input-json=*)
      input_json="${1#*=}"
      shift
      ;;
    --exclude-camera-ids)
      [[ $# -ge 2 ]] || die "--exclude-camera-ids requires a value"
      exclude_camera_ids="$2"
      shift 2
      ;;
    --exclude-camera-ids=*)
      exclude_camera_ids="${1#*=}"
      shift
      ;;
    --execute)
      execute=true
      shift
      ;;
    --confirmation)
      [[ $# -ge 2 ]] || die "--confirmation requires a value"
      confirmation="$2"
      shift 2
      ;;
    --confirmation=*)
      confirmation="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      forwarded_args+=("$@")
      break
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${mode}" == "teaching" || "${mode}" == "capture" ]]; then
  [[ ${#keys[@]} -gt 0 ]] || die "${mode} mode requires --key NAME"
  for key in "${keys[@]}"; do
    [[ -n "${key}" ]] || die "--key must not be empty"
    [[ "${key}" != */* && "${key}" != *..* ]] || die "--key must be a simple dataset name, got '${key}'"
  done
elif [[ ${#keys[@]} -gt 0 ]]; then
  die "--key is only valid in teaching/capture mode"
fi

if [[ "${mode}" == "calibrate" && ${#dataset_roots[@]} -eq 0 ]]; then
  die "calibrate mode requires at least one --dataset-root PATH"
fi
if [[ "${mode}" != "guided" && "${mode}" != "calibrate" && ${#dataset_roots[@]} -gt 0 ]]; then
  die "--dataset-root is only valid in guided/calibrate mode"
fi
if [[ "${mode}" == "solve" ]] && ! has_forwarded_prefix "--solve-run"; then
  die "solve mode requires --solve-run PATH|latest"
fi

if [[ "${mode}" == "teaching" && ${#keys[@]} -ne 1 ]]; then
  die "teaching mode accepts exactly one --key"
fi
if [[ "${mode}" != "capture" && -n "${input_json}" ]]; then
  die "--input-json is only valid in capture mode"
fi
if [[ -n "${input_json}" && ${#keys[@]} -ne 1 ]]; then
  die "capture with --input-json accepts exactly one --key"
fi

if [[ "${mode}" == "teaching" || "${mode}" == "guided" ]]; then
  required_confirmation="P0_TWO_MARKER_TEACHING"
elif [[ "${mode}" == "capture" ]]; then
  required_confirmation="P0_TWO_MARKER_AUTOMATIC_CAPTURE"
  if [[ "${max_records}" != "all" && ! "${max_records}" =~ ^[1-9][0-9]*$ ]]; then
    die "--max-records must be a positive integer or 'all'"
  fi
else
  required_confirmation=""
fi

if [[ "${mode}" == "guided" || "${mode}" == "teaching" || "${mode}" == "capture" ]] && ! ${dry_run}; then
  ${execute} || die "hardware run requires --execute"
  [[ "${confirmation}" == "${required_confirmation}" ]] || \
    die "hardware run requires --confirmation ${required_confirmation}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x /home/nvidia/Code/infer/.venv-fr3/bin/python3 ]]; then
  python_bin="/home/nvidia/Code/infer/.venv-fr3/bin/python3"
elif [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  python_bin="${REPO_ROOT}/.venv/bin/python3"
elif [[ -x "${REPO_ROOT}/third_party/opencv_kalibr/.venv/bin/python3" ]]; then
  python_bin="${REPO_ROOT}/third_party/opencv_kalibr/.venv/bin/python3"
else
  die "no suitable Python found; set PYTHON_BIN to the Thor FR3 Python"
fi
[[ -x "${python_bin}" ]] || die "PYTHON_BIN is not executable: ${python_bin}"

for required_path in \
  "${REPO_ROOT}/${TEACHING_SCRIPT}" \
  "${REPO_ROOT}/${TEACHING_CONFIG}" \
  "${REPO_ROOT}/${CAPTURE_SCRIPT}" \
  "${REPO_ROOT}/${CAPTURE_CONFIG}" \
  "${REPO_ROOT}/${RECORDER_CONFIG}" \
  "${REPO_ROOT}/${CAMERA_CONFIG_HELPER}" \
  "${REPO_ROOT}/${GUIDED_SCRIPT}" \
  "${REPO_ROOT}/${FR3_URDF}"; do
  [[ -e "${required_path}" ]] || die "required current-repo path is missing: ${required_path}"
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cmeel_entries=()
for candidate in \
  "${REPO_ROOT}/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib" \
  "/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib" \
  "${REPO_ROOT}/.venv/lib/python3.12/site-packages/cmeel.prefix/lib"; do
  [[ -d "${candidate}" ]] && cmeel_entries+=("${candidate}")
done
cmeel_entries+=("/usr/local/lib")
joined_cmeel="$(IFS=:; echo "${cmeel_entries[*]}")"
export LD_LIBRARY_PATH="${joined_cmeel}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"

temporary_recorder_config=""
cleanup() {
  if [[ -n "${temporary_recorder_config}" && -f "${temporary_recorder_config}" ]]; then
    rm -f -- "${temporary_recorder_config}"
  fi
}
trap cleanup EXIT

if [[ "${mode}" == "guided" || "${mode}" == "calibrate" || "${mode}" == "solve" ]]; then
  if [[ "${mode}" == "guided" && -z "${DISPLAY:-}" ]]; then
    if [[ -S /tmp/.X11-unix/X0 ]]; then
      export DISPLAY=:0
    elif ! ${dry_run}; then
      die "guided mode requires the Thor desktop DISPLAY; /tmp/.X11-unix/X0 is missing"
    fi
  fi
  command=(
    "${python_bin}"
    "${GUIDED_SCRIPT}"
  )
  if ! has_forwarded_prefix "--existing"; then
    command+=("--existing" "recalibrate")
  fi
  if ! has_forwarded_prefix "--intrinsics-summary"; then
    command+=("--intrinsics-summary" "${CURRENT_INTRINSICS}")
  fi
  if ! has_forwarded_prefix "--urdf"; then
    command+=("--urdf" "${REPO_ROOT}/${FR3_URDF}")
  fi
  if ! has_forwarded_prefix "--recorder-config"; then
    command+=("--recorder-config" "${REPO_ROOT}/${RECORDER_CONFIG}")
  fi
  for dataset_root in "${dataset_roots[@]}"; do
    command+=("--merge-dataset" "${dataset_root}")
  done
  if [[ "${mode}" == "guided" ]]; then
    IFS=',' read -r -a excluded_ids <<< "${exclude_camera_ids}"
    for sensor_id in "${excluded_ids[@]}"; do
      [[ "${sensor_id}" =~ ^[0-9]+$ ]] || die "invalid camera id in --exclude-camera-ids: ${sensor_id}"
      command+=("--exclude-camera" "cam_$(printf '%02d' "$((10#${sensor_id}))")")
    done
    command+=("--execute" "--confirmation" "${confirmation}")
    echo "[MODE] guided multi-camera teaching capture with accumulated per-camera counts"
    echo "[SAFETY] FR3 enters zero-stiffness teaching; support the arm and keep the E-stop ready."
  elif [[ "${mode}" == "calibrate" ]]; then
    command+=("--solve-merged-datasets")
    echo "[MODE] offline robust calibration from explicitly selected same-layout datasets"
  else
    echo "[MODE] offline solve of an existing guided/manual capture session"
  fi
  command+=("${forwarded_args[@]}")
elif [[ "${mode}" == "teaching" ]]; then
  key="${keys[0]}"
  output_json="outputs/datasets/${key}/teaching_pose_records.json"
  command=(
    "${python_bin}"
    "${TEACHING_SCRIPT}"
    "--config_path=${TEACHING_CONFIG}"
    "--robot.urdf_path=${REPO_ROOT}/${FR3_URDF}"
  )
  if ! has_forwarded_prefix "--output.json_path"; then
    command+=("--output.json_path=${output_json}")
  fi
  command+=("${forwarded_args[@]}")
  echo "[MODE] teaching pose capture; cameras remain closed"
  echo "[OUTPUT] ${REPO_ROOT}/${output_json}"
elif [[ "${mode}" == "capture" ]]; then
  # NVIDIA Argus must use headless EGL. A local desktop DISPLAY or an SSH X11
  # DISPLAY makes EGL select X/DRI3 and causes NvBufSurfaceMapEglImage to fail
  # for every camera in sequence.
  unset DISPLAY WAYLAND_DISPLAY
  echo "[ENV] DISPLAY/WAYLAND_DISPLAY cleared for headless Argus EGL"

  temporary_recorder_config="$(mktemp --tmpdir p0_calibration_recorder.XXXXXX.yaml)"
  camera_config_command=(
    "${python_bin}"
    "${CAMERA_CONFIG_HELPER}"
    "--template" "${RECORDER_CONFIG}"
    "--output" "${temporary_recorder_config}"
    "--repo-root" "${REPO_ROOT}"
    "--exclude-sensor-ids" "${exclude_camera_ids}"
  )
  if [[ -n "${P0_LOCKED_CAMERA_IDS:-}" ]]; then
    camera_config_command+=("--locked-sensor-ids" "${P0_LOCKED_CAMERA_IDS}")
  fi
  "${camera_config_command[@]}"

  if [[ -z "${merged_root}" ]]; then
    if [[ ${#keys[@]} -eq 1 ]]; then
      merged_root="outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_${keys[0]}_merged"
    else
      merged_root="outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_selected_merged"
    fi
  fi
  command=(
    "${python_bin}"
    "${CAPTURE_SCRIPT}"
    "--config" "${CAPTURE_CONFIG}"
    "--python-exec" "${python_bin}"
    "--merged-root" "${merged_root}"
    "--overwrite-merged"
  )
  if ! has_forwarded_prefix "--robot.urdf_path"; then
    command+=("--robot.urdf_path=${REPO_ROOT}/${FR3_URDF}")
  fi
  if ! has_forwarded_prefix "--thor.recorder_config_path"; then
    command+=("--thor.recorder_config_path=${temporary_recorder_config}")
  fi
  if [[ -n "${input_json}" ]]; then
    input_json_path="${input_json}"
    if [[ "${input_json_path}" != /* ]]; then
      input_json_path="${REPO_ROOT}/${input_json_path}"
    fi
    if ! ${dry_run} && [[ ! -f "${input_json_path}" ]]; then
      die "capture input JSON is missing: ${input_json_path}"
    fi
    command+=("--input.json_path=${input_json_path}")
    echo "[INPUT] ${input_json_path}"
  fi
  for key in "${keys[@]}"; do
    pose_json="outputs/datasets/${key}/teaching_pose_records.json"
    if [[ -z "${input_json}" ]] && ! ${dry_run} && [[ ! -f "${pose_json}" ]]; then
      die "teaching pose file is missing for key '${key}': ${REPO_ROOT}/${pose_json}"
    fi
    command+=("--key" "${key}")
  done
  if [[ "${max_records}" != "all" ]] && ! has_forwarded_prefix "--execution.max_records"; then
    command+=("--execution.max_records=${max_records}")
  fi
  command+=("${forwarded_args[@]}")
  echo "[MODE] automatic FR3 pose execution + Thor GMSL2 camera recording"
  echo "[SAFETY] Clear the full robot workspace, keep the E-stop ready, and supervise every pose."
  echo "[OUTPUT] ${REPO_ROOT}/${merged_root}"
fi

print_command "${command[@]}"
if ${dry_run}; then
  exit 0
fi
"${command[@]}"
