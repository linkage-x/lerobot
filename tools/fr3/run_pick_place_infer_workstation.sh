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
#   bash tools/fr3/run_pick_place_infer_workstation.sh dagger_sim             # MuJoCo takeover rehearsal
#   Each rehearsal drops outputs/dagger_sim/dryrun_<timestamp>.json (FR3_DAGGER_SIM_REPORT to
#   place it elsewhere): expert spans and the handback gap in mm, the number to read before the
#   takeover is allowed near the real arm.
#
# Take the arm over mid-rollout and keep the corrections as training data:
#   FR3_DAGGER_TAKEOVER=1 FR3_DAGGER_DATASET_ROOT=<dir> \
#     bash tools/fr3/run_pick_place_infer_workstation.sh real
#   Move the SpaceMouse to take the arm; let go and the policy resumes. Each correction
#   becomes one episode flagged is_intervention. Without the dataset root the takeover
#   still steers the arm, but nothing is written.
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
# shift, not a preference. `target_frame_name` below follows the checkpoint's dataset, which is the
# same thing as today's record config until someone trains on pre-switch episodes -- see the note
# there.
robot_ip="${FR3_ROBOT_IP-192.168.1.206}"
gripper_backend="${FR3_GRIPPER_BACKEND-pika}"
gripper_port="${FR3_GRIPPER_PORT-/dev/serial/by-path/pci-0000:00:14.0-usb-0:9.1.4:1.0-port0}"
gripper_max_width_mm="${FR3_GRIPPER_MAX_WIDTH_MM-90}"
camera_config="${FR3_INFER_CAMERA_CONFIG-tools/fr3/fr3_il_infer_realsense_camera_config.yaml}"
# The record config, named here because this file is the layer that knows which rig it started.
# The runtime reads robot.workspace_min/max out of it: the driver clips every commanded pose to
# that box and reports the clipped pose back, so a rollout running a box of its own stops short of
# where the demonstrations went and says nothing. It did -- the runtime's own copy stood at
# z >= 0.05 against this file's z >= 0, and the recorded frames reach z = 0.028.
record_config="${FR3_RECORD_CONFIG-tools/fr3/fr3_record_config.yaml}"

# The IK tool frame, and the reason this script exists rather than a few env vars on the host one.
# The two Pika frames are fixed on the same URDF and 410.85 mm apart, so naming the wrong one does
# not fail: the rollout runs, tracks its targets, and is wrong by that offset everywhere.
#
# The frame must match the dataset the *checkpoint* was trained on, which is normally the same as
# the record config -- and is, here. This default tracked `pika_task_tcp` while the recorder did;
# when the recorder switched, nothing had been trained yet (no outputs/train on either machine), so
# there was no pre-switch checkpoint for the old value to protect and it moved with it.
#
# Export FR3_TARGET_FRAME_NAME=pika_task_tcp if you ever roll out a checkpoint trained on
# pre-switch episodes -- the datasets recorded before the switch are still anchored there. Check the
# checkpoint's dataset before the run, not after it.
robot_urdf_path="${FR3_ROBOT_URDF_PATH-src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf}"
target_frame_name="${FR3_TARGET_FRAME_NAME-pika_gripper_ee}"

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
task_prompt="${FR3_TASK_PROMPT-}"

# Safety envelope. Same numbers fr3_train_il_policy.py writes into the generated inference config.
first_frame_max_pos_delta_mm="${FR3_FIRST_FRAME_MAX_POS_DELTA_MM-20}"
first_frame_max_rot_delta_deg="${FR3_FIRST_FRAME_MAX_ROT_DELTA_DEG-8}"
# 5 mm of policy step, measured against prev_cmd, admits 99.90% of the recorded demo frames.
# This used to be 3 mm and was compared against the *measured* pose instead, which folded servo
# tracking lag into the same budget and clamped every step of a healthy rollout.
max_step_pos_delta_mm="${FR3_MAX_STEP_POS_DELTA_MM-5}"
max_step_rot_delta_deg="${FR3_MAX_STEP_ROT_DELTA_DEG-2}"
# The command-vs-measured leash. Sized from the recorded lag (p95 10.65 mm, max 15.92 mm), so it
# only fires when the arm has genuinely stopped following.
max_leash_pos_delta_mm="${FR3_MAX_LEASH_POS_DELTA_MM-20}"
max_leash_rot_delta_deg="${FR3_MAX_LEASH_ROT_DELTA_DEG-8}"

# Deliberately unset by default. The host script ships values tuned against the other rig's arm,
# tool and task (n_action_steps=50, ensemble 0.01, EMA 0.35, its own controller gains); importing
# them here would be borrowing a tuning nobody measured on this hardware. Unset means the
# checkpoint's and the driver's own defaults, which is the honest baseline to tune away from.
policy_n_action_steps="${FR3_POLICY_N_ACTION_STEPS-}"
act_temporal_ensemble_coeff="${FR3_ACT_TEMPORAL_ENSEMBLE_COEFF-}"
rtc_mode="${FR3_RTC_MODE-auto}"
rtc_execution_horizon="${FR3_RTC_EXECUTION_HORIZON-16}"
rtc_max_guidance_weight="${FR3_RTC_MAX_GUIDANCE_WEIGHT-10}"
rtc_prefix_attention_schedule="${FR3_RTC_PREFIX_ATTENTION_SCHEDULE-EXP}"
rtc_replan_queue_size="${FR3_RTC_REPLAN_QUEUE_SIZE-25}"
rtc_inference_delay_steps="${FR3_RTC_INFERENCE_DELAY_STEPS-}"
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

# pi0.5 asks the Hub for the PaliGemma tokenizer (`google/paligemma-3b-pt-224`) at every startup.
# The policy weights come off a local checkpoint, but that one tokenizer does not, so a rollout on
# a rig with no route to huggingface.co stalls in retry backoff and then fails to build its
# processor -- with the arm powered and the cameras open.
#
# The default cache under ~/.cache/huggingface/hub is root-owned on this workstation (a container
# created it), so this points at one the operator can write. `unset` is not "use whatever is
# cached": without HF_HUB_OFFLINE the hub still round-trips to the network first and only falls
# back to the cache after five backoffs, which is a minute of dead time before an arm starts
# moving. Offline makes a missing asset a fast, legible failure instead.
#
# To refresh it, copy the snapshot from a machine that does have network:
#   rsync -aL ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/ \
#     <rig>:${FR3_HF_HOME:-$HOME/hf_cache}/hub/models--google--paligemma-3b-pt-224/
# Set FR3_HF_HUB_OFFLINE=0 to let a networked rig fetch what it is missing.
hf_home="${FR3_HF_HOME-$HOME/hf_cache}"
if [[ -d "${hf_home}" ]]; then
  export HF_HOME="${hf_home}"
  export HF_HUB_OFFLINE="${FR3_HF_HUB_OFFLINE-1}"
else
  echo "WARN: no HF cache at ${hf_home}; pi0.5 will fetch its tokenizer from huggingface.co" >&2
fi

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
  --record-config "${record_config}"
  --first-frame-max-pos-delta-mm "${first_frame_max_pos_delta_mm}"
  --first-frame-max-rot-delta-deg "${first_frame_max_rot_delta_deg}"
  --max-step-pos-delta-mm "${max_step_pos_delta_mm}"
  --max-step-rot-delta-deg "${max_step_rot_delta_deg}"
  --max-leash-pos-delta-mm "${max_leash_pos_delta_mm}"
  --max-leash-rot-delta-deg "${max_leash_rot_delta_deg}"
  # This rig homes with `home`, not with the DAS rig's joint configuration.
  --no-move-to-das-start
)

if [[ -n "${gripper_close_below}" ]]; then
  common_args+=(--gripper-close-below "${gripper_close_below}")
fi
if [[ -n "${dataset_root}" ]]; then
  common_args+=(--dataset-root "${dataset_root}")
fi
if [[ -n "${task_prompt}" ]]; then
  common_args+=(--task-prompt "${task_prompt}")
fi
if [[ -n "${robot_init_state}" ]]; then
  common_args+=(--robot-init-state "${robot_init_state}")
fi
if [[ -n "${policy_n_action_steps}" ]]; then
  common_args+=(--policy-n-action-steps "${policy_n_action_steps}")
fi

# DAgger takeover, for the interactive modes only: the runtime refuses --dagger-takeover
# without --interactive-rollouts, because a device connected with nobody at the rig is worse
# than one that was never opened. Kept out of common_args for exactly that reason.
dagger_args=()
if [[ "${FR3_DAGGER_TAKEOVER-0}" == "1" ]]; then
  dagger_args+=(--dagger-takeover)
  if [[ -n "${FR3_DAGGER_DATASET_ROOT-}" ]]; then
    dagger_args+=(--dagger-dataset-root "${FR3_DAGGER_DATASET_ROOT}")
  else
    # Loud, because this is the difference between corrections that train something and
    # corrections that only steer the arm for the length of one rollout.
    echo "[WARN] FR3_DAGGER_TAKEOVER=1 without FR3_DAGGER_DATASET_ROOT: takeovers will steer the arm but write no training data." >&2
  fi
  if [[ -n "${FR3_DAGGER_RELEASE_AFTER_S-}" ]]; then
    dagger_args+=(--dagger-takeover-release-after-s "${FR3_DAGGER_RELEASE_AFTER_S}")
  fi
fi
if [[ -n "${act_temporal_ensemble_coeff}" ]]; then
  common_args+=(--act-temporal-ensemble-coeff "${act_temporal_ensemble_coeff}")
fi
case "${rtc_mode}" in
  auto|AUTO) common_args+=(--rtc-auto) ;;
  enabled|enable|on|true|1|ENABLED|ENABLE|ON|TRUE) common_args+=(--rtc) ;;
  disabled|disable|off|false|0|DISABLED|DISABLE|OFF|FALSE) common_args+=(--no-rtc) ;;
  *) echo "ERROR: FR3_RTC_MODE must be auto, enabled, or disabled; got '${rtc_mode}'" >&2; exit 2 ;;
esac
if [[ -n "${rtc_execution_horizon}" ]]; then
  common_args+=(--rtc-execution-horizon "${rtc_execution_horizon}")
fi
if [[ -n "${rtc_max_guidance_weight}" ]]; then
  common_args+=(--rtc-max-guidance-weight "${rtc_max_guidance_weight}")
fi
if [[ -n "${rtc_prefix_attention_schedule}" ]]; then
  common_args+=(--rtc-prefix-attention-schedule "${rtc_prefix_attention_schedule}")
fi
if [[ -n "${rtc_replan_queue_size}" ]]; then
  common_args+=(--rtc-replan-queue-size "${rtc_replan_queue_size}")
fi
if [[ -n "${rtc_inference_delay_steps}" ]]; then
  common_args+=(--rtc-inference-delay-steps "${rtc_inference_delay_steps}")
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
  echo "[INFO] task_prompt=${task_prompt:-<auto from single dataset task>}"
  echo "[INFO] cameras=${camera_config} (keys must match the checkpoint's observation.images.*)"
  echo "[INFO] tool_frame=${target_frame_name} urdf=${robot_urdf_path}"
  echo "[INFO] workspace_fence=${record_config} (robot.workspace_min/max; the box the driver clips to)"
  echo "[INFO] gripper=${gripper_backend}@${gripper_port} max_width=${gripper_max_width_mm}mm close_below=${gripper_close_below:-<disabled>} (normalized 0..1)"
  echo "[INFO] safety: first_frame<${first_frame_max_pos_delta_mm}mm/${first_frame_max_rot_delta_deg}deg, per_step<${max_step_pos_delta_mm}mm/${max_step_rot_delta_deg}deg (vs prev_cmd), leash<${max_leash_pos_delta_mm}mm/${max_leash_rot_delta_deg}deg (vs measured)"
  echo "[INFO] rtc: mode=${rtc_mode} horizon=${rtc_execution_horizon:-<runtime default>} guidance=${rtc_max_guidance_weight:-<runtime default>} schedule=${rtc_prefix_attention_schedule:-<runtime default>} replan_q=${rtc_replan_queue_size:-<runtime default>} delay=${rtc_inference_delay_steps:-auto}"
}

case "$mode" in
  env)
    announce
    echo "FR3_MOVE_TO_START=${move_to_start}"
    echo "FR3_TASK_PROMPT=${task_prompt:-<auto>}"
    echo "FR3_POLICY_N_ACTION_STEPS=${policy_n_action_steps:-<checkpoint default>}"
    echo "FR3_ACT_TEMPORAL_ENSEMBLE_COEFF=${act_temporal_ensemble_coeff:-<disabled>}"
    echo "FR3_RTC_MODE=${rtc_mode}"
    echo "FR3_RTC_EXECUTION_HORIZON=${rtc_execution_horizon:-<runtime default>}"
    echo "FR3_RTC_MAX_GUIDANCE_WEIGHT=${rtc_max_guidance_weight:-<runtime default>}"
    echo "FR3_RTC_PREFIX_ATTENTION_SCHEDULE=${rtc_prefix_attention_schedule:-<runtime default>}"
    echo "FR3_RTC_REPLAN_QUEUE_SIZE=${rtc_replan_queue_size:-<runtime default>}"
    echo "FR3_RTC_INFERENCE_DELAY_STEPS=${rtc_inference_delay_steps:-<auto>}"
    echo "FR3_COMMAND_EMA_ALPHA=${command_ema_alpha:-<disabled>}"
    echo "FR3_CONTROLLER_STIFFNESS=${controller_stiffness:-<driver default>}"
    echo "FR3_CONTROLLER_DAMPING=${controller_damping:-<driver default>}"
    echo "HF_HOME=${HF_HOME:-<unset, tokenizer will be fetched from huggingface.co>}"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
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
    echo "[INFO] interactive_real_mode=enabled; keys: s=start rollout, x=stop, h=move to start, q=quit."
    echo "[INFO] focus the terminal (not the preview window) before pressing a key."
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --interactive-rollouts \
      --rollout-start-key s \
      --rollout-stop-key x \
      --rollout-home-key h \
      --rollout-quit-key q \
      --camera-preview-window \
      "${dagger_args[@]}" \
      "${extra_args[@]}"
    ;;
  real_debug)
    announce
    echo "[INFO] interactive_real_debug_mode=enabled; keys: s=start rollout, x=stop, h=move to start, q=quit."
    echo "[INFO] MuJoCo colors: orange=current EE, green=raw policy target, yellow=safe sent EE, blue-to-pink=action chunk."
    if [[ "${move_to_start}" == "1" ]]; then home_the_arm; fi
    exec "${FR3_HOST_PYTHON}" "${common_args[@]}" \
      --interactive-rollouts \
      --rollout-start-key s \
      --rollout-stop-key x \
      --rollout-home-key h \
      --rollout-quit-key q \
      --camera-preview-window \
      --mujoco-viewer \
      --log-interval 1 \
      "${dagger_args[@]}" \
      "${extra_args[@]}"
    ;;
  dagger_sim)
    # The MuJoCo rehearsal of the takeover. No checkpoint, no cameras, no real arm: a recorded
    # episode stands in for the policy so that what gets rehearsed is the handoff itself. The
    # clamp limits are handed over from this file so the rehearsal bounds motion by exactly the
    # numbers the real rollout will bound it by -- see tools/fr3/dagger_sim_dryrun.py.
    dagger_dataset="${FR3_DAGGER_SIM_DATASET-${dataset_root}}"
    if [[ -z "${dagger_dataset}" ]]; then
      echo "[ERROR] dagger_sim needs a dataset of demonstrations: set FR3_DAGGER_SIM_DATASET (or FR3_INFER_DATASET_ROOT)." >&2
      exit 2
    fi
    # Every rehearsal writes its report, because the handback gaps are the whole point of running
    # one and a terminal that scrolls away is not a place to keep a measurement. Timestamped, so a
    # second rehearsal cannot quietly overwrite the first -- comparing two of them is how anyone
    # sees whether a clamp change helped.
    dagger_report="${FR3_DAGGER_SIM_REPORT-outputs/dagger_sim/dryrun_$(date +%Y%m%d_%H%M%S).json}"
    echo "[INFO] dagger_sim_mode=enabled dataset=${dagger_dataset} episode=${FR3_DAGGER_SIM_EPISODE-0}"
    echo "[INFO] keys: s=start, t=take over / hand back, x=stop, q=quit. The arm is simulated."
    echo "[INFO] dagger_sim_report=${dagger_report}"
    # The inference flags the browser appends (camera preview, JPEG directory) mean nothing to a
    # run with no cameras, so they are deliberately not forwarded.
    exec "${FR3_HOST_PYTHON}" tools/fr3/dagger_sim_dryrun.py \
      --dataset "${dagger_dataset}" \
      --episode "${FR3_DAGGER_SIM_EPISODE-0}" \
      --config-path "${record_config}" \
      --output "${dagger_report}" \
      --max-step-pos-delta-mm "${max_step_pos_delta_mm}" \
      --max-step-rot-delta-deg "${max_step_rot_delta_deg}" \
      --max-leash-pos-delta-mm "${max_leash_pos_delta_mm}" \
      --max-leash-rot-delta-deg "${max_leash_rot_delta_deg}" \
      --live-frame-interval 1
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
    echo "Usage: $0 [env|home|smoke|preview|real|real_debug|real_once|dagger_sim]" >&2
    exit 2
    ;;
esac
