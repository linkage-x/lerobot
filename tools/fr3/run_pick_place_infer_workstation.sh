#!/usr/bin/env bash
# ACT rollout on the FR3 teleoperation workstation (the rig `bash run/deploy.sh workstation`
# records with). Workstation counterpart of run_pick_place_infer_host.sh, which drives the other
# FR3 rig (Hikrobot cameras, DAS/Corenetic gripper, 192.168.11.102) and cannot be pointed here by
# environment alone -- three of its settings mean different things on this rig, see below.
#
# Usage:
#   bash tools/fr3/run_pick_place_infer_workstation.sh env                    # print resolved settings
#   bash tools/fr3/run_pick_place_infer_workstation.sh home                   # move the arm to its start pose
#   bash tools/fr3/run_pick_place_infer_workstation.sh smoke                  # one step, no motion
#   bash tools/fr3/run_pick_place_infer_workstation.sh preview                # 20 steps, no motion, camera window
#   bash tools/fr3/run_pick_place_infer_workstation.sh real                   # interactive rollouts (s/x/q)
#   bash tools/fr3/run_pick_place_infer_workstation.sh real_debug             # + MuJoCo target viewer
#   bash tools/fr3/run_pick_place_infer_workstation.sh real_once              # one bounded rollout
#
# Point it at a checkpoint:
#   FR3_INFER_CHECKPOINT=outputs/train/<job>/checkpoints/last \
#     bash tools/fr3/run_pick_place_infer_workstation.sh smoke
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_args=("$@")

# --- What this rig is --------------------------------------------------------
# Defaults mirror tools/fr3/fr3_record_config.yaml, because a rollout has to meet the hardware
# the data came off. Anything that disagrees with the record config is a silent distribution
# shift, not a preference.
robot_ip="${FR3_ROBOT_IP-192.168.1.206}"
gripper_backend="${FR3_GRIPPER_BACKEND-pika}"
gripper_port="${FR3_GRIPPER_PORT-/dev/serial/by-path/pci-0000:00:14.0-usb-0:9.1.4:1.0-port0}"
gripper_max_width_mm="${FR3_GRIPPER_MAX_WIDTH_MM-90}"
camera_config="${FR3_INFER_CAMERA_CONFIG-tools/fr3/fr3_il_infer_realsense_camera_config.yaml}"

# The IK tool frame, and the reason this script exists rather than a few env vars on the host one.
# fr3_act_infer_real_runtime.py defaults a Pika gripper to `pika_gripper_ee`, but the workstation
# records against `pika_task_tcp` (fr3_record_config.yaml) -- two fixed frames on the same URDF
# roughly 0.4 m apart. Left at the default, every recorded pose would be interpreted against the
# wrong frame: the rollout would run, track its targets, and be wrong by that offset everywhere.
robot_urdf_path="${FR3_ROBOT_URDF_PATH-src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf}"
target_frame_name="${FR3_TARGET_FRAME_NAME-pika_task_tcp}"

# Gripper units. On this rig `gripper.pos` is a normalized 0..1 opening in both the dataset and
# the robot command; on the Hikrobot rig unit-bearing feature names such as `*.width_mm` carry the
# millimetre contract. fr3_act_infer_real_runtime.py now decodes gripper units from the dataset
# feature name before falling back to the legacy value heuristic, so the old 0.08 -> 80 mm
# ambiguity no longer needs a default threshold here.
#
# Leave disabled unless you intentionally want a task-specific binary close guard. If enabled, it
# clamps raw policy gripper values below the threshold to fully closed before unit conversion.
gripper_close_below="${FR3_GRIPPER_CLOSE_BELOW-}"

checkpoint="${FR3_INFER_CHECKPOINT-outputs/train/fr3_spacemouse__delta_ee_from_prev_cmd/checkpoints/last}"
# Empty means "the dataset the checkpoint was trained on", read out of its own train_config.json.
# For a view built by the Training View page that is the view root, which is what the action
# contract and the start-pose reference have to come from.
dataset_root="${FR3_INFER_DATASET_ROOT-}"

# Safety envelope. Same numbers fr3_train_il_policy.py writes into the generated inference config.
first_frame_max_pos_delta_mm="${FR3_FIRST_FRAME_MAX_POS_DELTA_MM-20}"
first_frame_max_rot_delta_deg="${FR3_FIRST_FRAME_MAX_ROT_DELTA_DEG-8}"
max_step_pos_delta_mm="${FR3_MAX_STEP_POS_DELTA_MM-3}"
max_step_rot_delta_deg="${FR3_MAX_STEP_ROT_DELTA_DEG-2}"

# Deliberately unset by default. The host script ships values tuned against the other rig's arm,
# tool and task (n_action_steps=50, ensemble 0.01, EMA 0.35, its own controller gains); importing
# them here would be borrowing a tuning nobody measured on this hardware. Unset means the
# checkpoint's and the driver's own defaults, which is the honest baseline to tune away from.
policy_n_action_steps="${FR3_POLICY_N_ACTION_STEPS-}"
act_temporal_ensemble_coeff="${FR3_ACT_TEMPORAL_ENSEMBLE_COEFF-}"
command_ema_alpha="${FR3_COMMAND_EMA_ALPHA-}"
controller_stiffness="${FR3_CONTROLLER_STIFFNESS-}"
controller_damping="${FR3_CONTROLLER_DAMPING-}"
gripper_change_delay_s="${FR3_GRIPPER_CHANGE_DELAY_S-}"
gripper_change_min_delta="${FR3_GRIPPER_CHANGE_MIN_DELTA-}"

# Startup pose. Never the runtime's --move-to-das-start: that moves to a hardcoded joint
# configuration belonging to the DAS rig. The dataset frame is anchored to wherever this arm
# starts (T_B_Ws is solved from the first observation against the dataset's start pose), so the
# start pose is what places the whole trajectory in the workspace -- it has to be the pose the
# episodes were recorded from, which on this rig is the XML `home` keyframe the recorder homes to
# between episodes.
move_to_start="${FR3_MOVE_TO_START-1}"
robot_init_state="${FR3_INFER_ROBOT_INIT_STATE-}"

select_python() {
  if [[ -n "${FR3_HOST_PYTHON:-}" ]]; then
    printf '%s\n' "${FR3_HOST_PYTHON}"
    return
  fi
  if [[ -x ".venv-fr3/bin/python" ]]; then
    printf '%s\n' ".venv-fr3/bin/python"
    return
  fi
  echo "ERROR: no .venv-fr3 in ${repo_root}. Run: bash tools/fr3/setup_workstation_teleop_env.sh" >&2
  exit 2
}

FR3_HOST_PYTHON="$(select_python)"
venv_root="$(cd "$(dirname "${FR3_HOST_PYTHON}")/.." && pwd)"
# placo/pinocchio ship their shared objects under the cmeel prefix; the IK import fails without it.
cmeel_prefix="$(find "${venv_root}/lib" -path '*/site-packages/cmeel.prefix' -type d | head -n 1 || true)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${cmeel_prefix:+${cmeel_prefix}/lib:}/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

common_args=(
  tools/fr3/fr3_act_infer_real_runtime.py
  --checkpoint "${checkpoint}"
  --camera-config "${camera_config}"
  --robot-ip "${robot_ip}"
  --gripper-backend "${gripper_backend}"
  --gripper-port "${gripper_port}"
  --gripper-max-width-mm "${gripper_max_width_mm}"
  --robot-urdf-path "${robot_urdf_path}"
  --target-frame-name "${target_frame_name}"
  --first-frame-max-pos-delta-mm "${first_frame_max_pos_delta_mm}"
  --first-frame-max-rot-delta-deg "${first_frame_max_rot_delta_deg}"
  --max-step-pos-delta-mm "${max_step_pos_delta_mm}"
  --max-step-rot-delta-deg "${max_step_rot_delta_deg}"
  # This rig homes with `home`, not with the DAS rig's joint configuration.
  --no-move-to-das-start
)

if [[ -n "${gripper_close_below}" ]]; then
  common_args+=(--gripper-close-below "${gripper_close_below}")
fi
if [[ -n "${dataset_root}" ]]; then
  common_args+=(--dataset-root "${dataset_root}")
fi
if [[ -n "${robot_init_state}" ]]; then
  common_args+=(--robot-init-state "${robot_init_state}")
fi
if [[ -n "${policy_n_action_steps}" ]]; then
  common_args+=(--policy-n-action-steps "${policy_n_action_steps}")
fi
if [[ -n "${act_temporal_ensemble_coeff}" ]]; then
  common_args+=(--act-temporal-ensemble-coeff "${act_temporal_ensemble_coeff}")
fi
if [[ -n "${command_ema_alpha}" ]]; then
  common_args+=(--command-ema-alpha "${command_ema_alpha}")
fi
if [[ -n "${controller_stiffness}" ]]; then
  common_args+=(--controller-stiffness "${controller_stiffness}")
fi
if [[ -n "${controller_damping}" ]]; then
  common_args+=(--controller-damping "${controller_damping}")
fi
if [[ -n "${gripper_change_delay_s}" ]]; then
  common_args+=(--gripper-change-delay-s "${gripper_change_delay_s}")
fi
if [[ -n "${gripper_change_min_delta}" ]]; then
  common_args+=(--gripper-change-min-delta "${gripper_change_min_delta}")
fi

home_the_arm() {
  echo "[INFO] moving FR3 ${robot_ip} to fr3_pika_gripper.xml home keyframe (FR3_MOVE_TO_START=0 to skip)"
  "${FR3_HOST_PYTHON}" tools/fr3/fr3_move_to_start_runtime.py --robot-ip "${robot_ip}"
}

announce() {
  echo "[INFO] host_python=${FR3_HOST_PYTHON}"
  echo "[INFO] checkpoint=${checkpoint}"
  echo "[INFO] dataset_root=${dataset_root:-<from checkpoint train_config.json>}"
  echo "[INFO] cameras=${camera_config} (keys must match the checkpoint's observation.images.*)"
  echo "[INFO] tool_frame=${target_frame_name} urdf=${robot_urdf_path}"
  echo "[INFO] gripper=${gripper_backend}@${gripper_port} max_width=${gripper_max_width_mm}mm close_below=${gripper_close_below:-<disabled>} (normalized 0..1)"
  echo "[INFO] safety: first_frame<${first_frame_max_pos_delta_mm}mm/${first_frame_max_rot_delta_deg}deg, per_step<${max_step_pos_delta_mm}mm/${max_step_rot_delta_deg}deg"
}

case "$mode" in
  env)
    announce
    echo "FR3_MOVE_TO_START=${move_to_start}"
    echo "FR3_POLICY_N_ACTION_STEPS=${policy_n_action_steps:-<checkpoint default>}"
    echo "FR3_ACT_TEMPORAL_ENSEMBLE_COEFF=${act_temporal_ensemble_coeff:-<disabled>}"
    echo "FR3_COMMAND_EMA_ALPHA=${command_ema_alpha:-<disabled>}"
    echo "FR3_CONTROLLER_STIFFNESS=${controller_stiffness:-<driver default>}"
    echo "FR3_CONTROLLER_DAMPING=${controller_damping:-<driver default>}"
    echo "PYTHONPATH=${PYTHONPATH}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    ;;
  home)
    home_the_arm
    ;;
  smoke)
    # No motion and no homing: this only proves the checkpoint loads, the two RealSense units open
    # under the names the policy asks for, and one forward pass produces a decodable action.
    announce
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --preview \
      --max-steps 1 \
      --no-align-gripper-to-dataset-start \
      "${extra_args[@]}"
    ;;
  preview)
    announce
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --preview \
      --max-steps 20 \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  real)
    announce
    echo "[INFO] interactive_real_mode=enabled; keys: s=start rollout, x=stop, q=quit."
    echo "[INFO] focus the terminal (not the preview window) before pressing a key."
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --interactive-rollouts \
      --rollout-start-key s \
      --rollout-stop-key x \
      --rollout-quit-key q \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  real_debug)
    announce
    echo "[INFO] interactive_real_debug_mode=enabled; keys: s=start rollout, x=stop, q=quit."
    echo "[INFO] MuJoCo colors: orange=current EE, green=raw policy target, yellow=safe sent EE, blue-to-pink=action chunk."
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
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
    announce
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --max-steps "${FR3_INFER_MAX_STEPS:-300}" \
      --camera-preview-window \
      "${extra_args[@]}"
    ;;
  *)
    echo "Usage: $0 [env|home|smoke|preview|real|real_debug|real_once]" >&2
    exit 2
    ;;
esac
