#!/usr/bin/env bash
set -euo pipefail

cd /home/corenetic/Code/zyx/lerobot

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_args=("$@")
gripper_close_below="${FR3_GRIPPER_CLOSE_BELOW-70}"
gripper_change_delay_s="${FR3_GRIPPER_CHANGE_DELAY_S-0.25}"
gripper_change_min_delta="${FR3_GRIPPER_CHANGE_MIN_DELTA-0.08}"
gripper_change_settle_tolerance="${FR3_GRIPPER_CHANGE_SETTLE_TOLERANCE-0.12}"
gripper_change_settle_timeout_s="${FR3_GRIPPER_CHANGE_SETTLE_TIMEOUT_S-1.5}"
policy_n_action_steps="${FR3_POLICY_N_ACTION_STEPS-50}"
act_temporal_ensemble_coeff="${FR3_ACT_TEMPORAL_ENSEMBLE_COEFF-0.01}"
act_temporal_action_offset="${FR3_ACT_TEMPORAL_ACTION_OFFSET-0}"
act_temporal_stuck_max_offset="${FR3_ACT_TEMPORAL_STUCK_MAX_OFFSET-}"
act_temporal_stuck_offset_step="${FR3_ACT_TEMPORAL_STUCK_OFFSET_STEP-2}"
act_temporal_stuck_steps="${FR3_ACT_TEMPORAL_STUCK_STEPS-12}"
act_temporal_stuck_pos_delta_mm="${FR3_ACT_TEMPORAL_STUCK_POS_DELTA_MM-3}"
act_temporal_stuck_closed_gripper_max="${FR3_ACT_TEMPORAL_STUCK_CLOSED_GRIPPER_MAX-0.05}"
command_ema_alpha="${FR3_COMMAND_EMA_ALPHA-0.35}"
place_assist_offset_base_xyz="${FR3_PLACE_ASSIST_OFFSET_BASE_XYZ-}"
place_assist_stuck_steps="${FR3_PLACE_ASSIST_STUCK_STEPS-20}"
place_assist_stuck_pos_delta_mm="${FR3_PLACE_ASSIST_STUCK_POS_DELTA_MM-3}"
place_assist_ramp_step_mm="${FR3_PLACE_ASSIST_RAMP_STEP_MM-1.5}"
place_assist_closed_gripper_max="${FR3_PLACE_ASSIST_CLOSED_GRIPPER_MAX-0.05}"
controller_stiffness="${FR3_CONTROLLER_STIFFNESS-600,600,600,600,280,180,70}"
controller_damping="${FR3_CONTROLLER_DAMPING-50,50,50,50,20,15,10}"
controller_filter_coeff="${FR3_CONTROLLER_FILTER_COEFF-}"
first_frame_max_pos_delta_mm="${FR3_FIRST_FRAME_MAX_POS_DELTA_MM-20}"
first_frame_max_rot_delta_deg="${FR3_FIRST_FRAME_MAX_ROT_DELTA_DEG-8}"
max_step_pos_delta_mm="${FR3_MAX_STEP_POS_DELTA_MM-3}"
max_step_rot_delta_deg="${FR3_MAX_STEP_ROT_DELTA_DEG-2}"
checkpoint="${FR3_INFER_CHECKPOINT-outputs/train/pick_place_act_cam2_cam3_pika_right_imgonly/checkpoints/060000}"
dataset_root="${FR3_INFER_DATASET_ROOT-}"
camera_config="${FR3_INFER_CAMERA_CONFIG-tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml}"
gripper_backend="${FR3_GRIPPER_BACKEND-pika}"
gripper_port="${FR3_GRIPPER_PORT-/dev/ttyUSB0}"
gripper_max_width_mm="${FR3_GRIPPER_MAX_WIDTH_MM-90}"
if [[ "${gripper_backend}" == "box" ]]; then
  echo "[WARN] FR3_GRIPPER_BACKEND=box is deprecated; use FR3_GRIPPER_BACKEND=corenetic." >&2
  gripper_backend="corenetic"
fi
corenetic_bind_ip="${FR3_CORENETIC_BIND_IP-${FR3_BOX_BIND_IP-0.0.0.0}}"
corenetic_bind_port="${FR3_CORENETIC_BIND_PORT-${FR3_BOX_BIND_PORT-15000}}"
corenetic_remote_ip="${FR3_CORENETIC_REMOTE_IP-${FR3_BOX_REMOTE_IP-192.168.2.60}}"
corenetic_remote_port="${FR3_CORENETIC_REMOTE_PORT-${FR3_BOX_REMOTE_PORT-15000}}"
corenetic_sdk_dir="${FR3_CORENETIC_SDK_DIR-${FR3_BOX_SDK_DIR-tools/thor/box_sdk}}"
corenetic_connect_timeout_s="${FR3_CORENETIC_CONNECT_TIMEOUT_S-${FR3_BOX_CONNECT_TIMEOUT_S-3.0}}"
corenetic_poll_interval_s="${FR3_CORENETIC_POLL_INTERVAL_S-${FR3_BOX_POLL_INTERVAL_S-0.01}}"
corenetic_stale_threshold_s="${FR3_CORENETIC_STALE_THRESHOLD_S-${FR3_BOX_STALE_THRESHOLD_S-1.0}}"
robot_urdf_path="${FR3_ROBOT_URDF_PATH-}"
target_frame_name="${FR3_TARGET_FRAME_NAME-}"

select_python() {
  if [[ -n "${FR3_HOST_PYTHON:-}" ]]; then
    printf '%s\n' "${FR3_HOST_PYTHON}"
    return
  fi

  if [[ -x ".venv-fr3/bin/python" ]]; then
    printf '%s\n' ".venv-fr3/bin/python"
    return
  fi

  if [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
    return
  fi

  printf 'Could not find a repo-local host Python. Run: UV_PROJECT_ENVIRONMENT=.venv-fr3 uv sync --extra fr3-host --extra cv2-gui\n' >&2
  exit 2
}

FR3_HOST_PYTHON="$(select_python)"
venv_root="$(cd "$(dirname "${FR3_HOST_PYTHON}")/.." && pwd)"
cmeel_prefix="$(find "${venv_root}/lib" -path '*/site-packages/cmeel.prefix' -type d | head -n 1 || true)"

export PYTHONPATH="$PWD/src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python${PYTHONPATH:+:$PYTHONPATH}"
export HIKROBOT_MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH="${cmeel_prefix:+${cmeel_prefix}/lib:}/usr/local/lib:/opt/MVS/lib/64:/opt/MVS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${gripper_backend}" == "corenetic" ]]; then
  corenetic_sdk_abs="${corenetic_sdk_dir}"
  if [[ "${corenetic_sdk_abs}" != /* ]]; then
    corenetic_sdk_abs="${PWD}/${corenetic_sdk_abs}"
  fi
  export BOX_SDK_URDF="${BOX_SDK_URDF:-${corenetic_sdk_abs}/share/monte_gripper.urdf}"
  export LD_LIBRARY_PATH="${corenetic_sdk_abs}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

common_args=(
  tools/fr3/fr3_act_infer_real_runtime.py
  --checkpoint "${checkpoint}"
  --camera-config "${camera_config}"
  --gripper-backend "${gripper_backend}"
  --gripper-port "${gripper_port}"
  --gripper-max-width-mm "${gripper_max_width_mm}"
  --robot-ip 192.168.11.102
  --first-frame-max-pos-delta-mm "${first_frame_max_pos_delta_mm}"
  --first-frame-max-rot-delta-deg "${first_frame_max_rot_delta_deg}"
  --max-step-pos-delta-mm "${max_step_pos_delta_mm}"
  --max-step-rot-delta-deg "${max_step_rot_delta_deg}"
)

if [[ "${gripper_backend}" == "corenetic" ]]; then
  common_args+=(--corenetic-bind-ip "${corenetic_bind_ip}")
  common_args+=(--corenetic-bind-port "${corenetic_bind_port}")
  common_args+=(--corenetic-remote-ip "${corenetic_remote_ip}")
  common_args+=(--corenetic-remote-port "${corenetic_remote_port}")
  common_args+=(--corenetic-sdk-dir "${corenetic_sdk_dir}")
  common_args+=(--corenetic-connect-timeout-s "${corenetic_connect_timeout_s}")
  common_args+=(--corenetic-poll-interval-s "${corenetic_poll_interval_s}")
  common_args+=(--corenetic-stale-threshold-s "${corenetic_stale_threshold_s}")
fi
if [[ -n "${robot_urdf_path}" ]]; then
  common_args+=(--robot-urdf-path "${robot_urdf_path}")
fi
if [[ -n "${target_frame_name}" ]]; then
  common_args+=(--target-frame-name "${target_frame_name}")
fi
if [[ -n "${dataset_root}" ]]; then
  common_args+=(--dataset-root "${dataset_root}")
fi
if [[ -n "${gripper_close_below}" ]]; then
  common_args+=(--gripper-close-below "${gripper_close_below}")
fi
if [[ -n "${gripper_change_delay_s}" ]]; then
  common_args+=(--gripper-change-delay-s "${gripper_change_delay_s}")
fi
if [[ -n "${gripper_change_min_delta}" ]]; then
  common_args+=(--gripper-change-min-delta "${gripper_change_min_delta}")
fi
if [[ -n "${gripper_change_settle_tolerance}" ]]; then
  common_args+=(--gripper-change-settle-tolerance "${gripper_change_settle_tolerance}")
fi
if [[ -n "${gripper_change_settle_timeout_s}" ]]; then
  common_args+=(--gripper-change-settle-timeout-s "${gripper_change_settle_timeout_s}")
fi
if [[ -n "${policy_n_action_steps}" ]]; then
  common_args+=(--policy-n-action-steps "${policy_n_action_steps}")
fi
if [[ -n "${act_temporal_ensemble_coeff}" ]]; then
  common_args+=(--act-temporal-ensemble-coeff "${act_temporal_ensemble_coeff}")
fi
if [[ -n "${act_temporal_action_offset}" ]]; then
  common_args+=(--act-temporal-action-offset "${act_temporal_action_offset}")
fi
if [[ -n "${act_temporal_stuck_max_offset}" ]]; then
  common_args+=(--act-temporal-stuck-max-offset "${act_temporal_stuck_max_offset}")
  common_args+=(--act-temporal-stuck-offset-step "${act_temporal_stuck_offset_step}")
  common_args+=(--act-temporal-stuck-steps "${act_temporal_stuck_steps}")
  common_args+=(--act-temporal-stuck-pos-delta-mm "${act_temporal_stuck_pos_delta_mm}")
  common_args+=(--act-temporal-stuck-closed-gripper-max "${act_temporal_stuck_closed_gripper_max}")
fi
if [[ -n "${command_ema_alpha}" ]]; then
  common_args+=(--command-ema-alpha "${command_ema_alpha}")
fi
if [[ -n "${place_assist_offset_base_xyz}" ]]; then
  common_args+=(--place-assist-offset-base-xyz "${place_assist_offset_base_xyz}")
  common_args+=(--place-assist-stuck-steps "${place_assist_stuck_steps}")
  common_args+=(--place-assist-stuck-pos-delta-mm "${place_assist_stuck_pos_delta_mm}")
  common_args+=(--place-assist-ramp-step-mm "${place_assist_ramp_step_mm}")
  common_args+=(--place-assist-closed-gripper-max "${place_assist_closed_gripper_max}")
fi
if [[ -n "${controller_stiffness}" ]]; then
  common_args+=(--controller-stiffness "${controller_stiffness}")
fi
if [[ -n "${controller_damping}" ]]; then
  common_args+=(--controller-damping "${controller_damping}")
fi
if [[ -n "${controller_filter_coeff}" ]]; then
  common_args+=(--controller-filter-coeff "${controller_filter_coeff}")
fi

init_state_args=(
  --robot-init-state ee_xyzquat=0.584972,-0.198668,0.270659,0.959580,0.006716,0.281263,-0.007254
)

case "$mode" in
  smoke)
    echo "[INFO] host_python=${FR3_HOST_PYTHON}"
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --preview \
      --max-steps 1 \
      --no-move-to-das-start \
      --no-align-gripper-to-dataset-start \
      "${extra_args[@]}"
    ;;
  preview)
    echo "[INFO] host_python=${FR3_HOST_PYTHON}"
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" "${init_state_args[@]}" \
      --preview \
      --max-steps 20 \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  real)
    echo "[INFO] host_python=${FR3_HOST_PYTHON}"
    echo "[INFO] interactive_real_mode=enabled"
    echo "[INFO] startup: move to robot init state, then wait for keyboard input."
    echo "[INFO] keys: s=start rollout, x=stop current rollout, q=quit program."
    echo "[INFO] policy_n_action_steps=${policy_n_action_steps:-checkpoint_default}; default 1 means replan every step."
    echo "[INFO] gripper_change_delay_s=${gripper_change_delay_s:-disabled}; min_delta=${gripper_change_min_delta:-default}, settle_tol=${gripper_change_settle_tolerance:-default}, settle_timeout=${gripper_change_settle_timeout_s:-default}; normalized [0,1]."
    echo "[INFO] act_temporal_ensemble_coeff=${act_temporal_ensemble_coeff:-disabled}; enabled forces ACT n_action_steps=1; positive favors old chunks, negative favors new chunks."
    echo "[INFO] act_temporal_action_offset=${act_temporal_action_offset:-0}; 0 sends immediate action, larger values send farther into the ensembled chunk."
    echo "[INFO] command_ema_alpha=${command_ema_alpha:-disabled}; use with action queue to smooth commands without ACT target sticking."
    echo "[INFO] controller_gains=stiffness:${controller_stiffness:-default} damping:${controller_damping:-default} filter:${controller_filter_coeff:-default}."
    echo "[INFO] safety: first_frame<${first_frame_max_pos_delta_mm}mm/${first_frame_max_rot_delta_deg}deg, per_step<${max_step_pos_delta_mm}mm/${max_step_rot_delta_deg}deg."
    echo "[INFO] camera_preview_window=enabled; click/focus the terminal before pressing rollout keys."
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" "${init_state_args[@]}" \
      --interactive-rollouts \
      --rollout-start-key s \
      --rollout-stop-key x \
      --rollout-quit-key q \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  real_debug)
    echo "[INFO] host_python=${FR3_HOST_PYTHON}"
    echo "[INFO] interactive_real_debug_mode=enabled"
    echo "[INFO] startup: move to robot init state, then wait for keyboard input."
    echo "[INFO] keys: s=start rollout, x=stop current rollout, q=quit program."
    echo "[INFO] debug windows: OpenCV camera preview + MuJoCo current/target/action-chunk viewer."
    echo "[INFO] MuJoCo colors: orange=current EE, green=raw policy target EE, yellow=safe sent EE, blue-to-pink=policy action chunk."
    echo "[INFO] Terminal log shows raw_ee, safe_ee, clamp/hold status, and gripper at every step."
    echo "[INFO] policy_n_action_steps=${policy_n_action_steps:-checkpoint_default}; default 1 means replan every step."
    echo "[INFO] gripper_change_delay_s=${gripper_change_delay_s:-disabled}; min_delta=${gripper_change_min_delta:-default}, settle_tol=${gripper_change_settle_tolerance:-default}, settle_timeout=${gripper_change_settle_timeout_s:-default}; normalized [0,1]."
    echo "[INFO] act_temporal_ensemble_coeff=${act_temporal_ensemble_coeff:-disabled}; enabled forces ACT n_action_steps=1; positive favors old chunks, negative favors new chunks."
    echo "[INFO] act_temporal_action_offset=${act_temporal_action_offset:-0}; 0 sends immediate action, larger values send farther into the ensembled chunk."
    echo "[INFO] command_ema_alpha=${command_ema_alpha:-disabled}; use with action queue to smooth commands without ACT target sticking."
    echo "[INFO] controller_gains=stiffness:${controller_stiffness:-default} damping:${controller_damping:-default} filter:${controller_filter_coeff:-default}."
    echo "[INFO] safety: first_frame<${first_frame_max_pos_delta_mm}mm/${first_frame_max_rot_delta_deg}deg, per_step<${max_step_pos_delta_mm}mm/${max_step_rot_delta_deg}deg."
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" "${init_state_args[@]}" \
      --interactive-rollouts \
      --rollout-start-key s \
      --rollout-stop-key x \
      --rollout-quit-key q \
      --camera-preview-window \
      --mujoco-viewer \
      --log-interval 1 \
      "${extra_args[@]}"
    ;;
  real_once)
    echo "[INFO] host_python=${FR3_HOST_PYTHON}"
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" "${init_state_args[@]}" \
      --max-steps "${FR3_INFER_MAX_STEPS:-300}" \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  env)
    echo "FR3_HOST_PYTHON=${FR3_HOST_PYTHON}"
    echo "FR3_INFER_CHECKPOINT=${checkpoint}"
    echo "FR3_INFER_DATASET_ROOT=${dataset_root}"
    echo "FR3_INFER_CAMERA_CONFIG=${camera_config}"
    echo "FR3_GRIPPER_BACKEND=${gripper_backend}"
    echo "FR3_GRIPPER_PORT=${gripper_port}"
    echo "FR3_GRIPPER_MAX_WIDTH_MM=${gripper_max_width_mm}"
    echo "FR3_CORENETIC_BIND_IP=${corenetic_bind_ip}"
    echo "FR3_CORENETIC_REMOTE_IP=${corenetic_remote_ip}"
    echo "FR3_CORENETIC_SDK_DIR=${corenetic_sdk_dir}"
    echo "FR3_ROBOT_URDF_PATH=${robot_urdf_path}"
    echo "FR3_TARGET_FRAME_NAME=${target_frame_name}"
    echo "BOX_SDK_URDF=${BOX_SDK_URDF:-}"
    echo "PYTHONPATH=${PYTHONPATH}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    ;;
  *)
    echo "Usage: $0 [smoke|preview|real|real_debug|real_once|env]" >&2
    exit 2
    ;;
esac
