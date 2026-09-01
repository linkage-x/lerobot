#!/usr/bin/env python3

"""Rehearse the DAgger handoff in MuJoCo, before it is allowed near the real arm.

The takeover changes the inference control loop while that loop is moving a robot. Two things
can go wrong there that no unit test reaches, because both are about *timing* rather than about
values: the arm can jolt at the instant control changes hands, and the operator can find that
the gripper is not theirs at the moment they need it. This script puts both under a simulated
arm, at the real frame rate, with the real SpaceMouse in the operator's hand.

**Why a recorded episode stands in for the policy.** What is being rehearsed is the handoff, not
the policy: two action sources, one clamp path, one reference pose. A demonstration replayed
frame by frame is an action source with exactly the properties that matter here -- it produces a
command every frame, it is reproducible, and it is *wrong* about where the arm is the moment the
operator moves it, which is precisely the condition the handback has to survive. Running the
checkpoint instead would need the sim to render both cameras well enough for the policy to be
believed, which rehearses camera rendering, not the handoff.

**How the operator takes over.** By moving the SpaceMouse. There is no engage key to find at
the moment something is going wrong, and nothing is left latched when the rollout ends: the arm
goes back to the demo stream once the device has been quiet for ``--takeover-release-after-s``.
So a rehearsal is a sequence of short corrections, and each one prints its own handback gap.

**What happens when the operator takes over.** The demo stream stops advancing. It is open loop:
it would otherwise keep marching through poses derived from a trajectory the arm is no longer
on, and hand back a target metres from where the operator left it. Frozen, the handback is a
single command back to the pose the demo was paused at, and it goes through the same step guard
as everything else -- so the arm walks back at the clamp's rate instead of lunging. That walk is
worth watching: it is the same reconciliation a real rollout performs when the policy resumes,
and ``handback_gap_mm`` on the release line reports how far it had to come.

Run it with the browser open on the Rollout page: ``--live-frame-interval 1`` publishes the arm
state every frame, and the page draws it live rather than after the fact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.utils.rotation import Rotation  # noqa: E402

from tools.fr3.command_guard import (  # noqa: E402
    limit_command_for_safety,
    observation_with_prev_cmd,
    smooth_robot_command_ema,
)
from tools.fr3.dagger_takeover import ExpertTakeover, expert_spans, motion_gain_for  # noqa: E402
from tools.fr3.interactive_control import InteractiveRolloutKeyboard  # noqa: E402
from tools.fr3.live_frames import LiveFrameEmitter  # noqa: E402

# Imported from the replay runtime rather than reimplemented: the episode's action stream must be
# reconstructed the same way the replay validation reconstructs it, or the rehearsal would be
# driving a trajectory that no other tool in this repo agrees with.
from tools.fr3.fr3_gui_replay_runtime import (  # noqa: E402
    _column,
    _replay_robot_config,
    _settle_to_start_pose,
    emit,
    load_episode_actions,
    reconstruct_absolute_pose_stream,
)

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")


def _command_from_pose(position: np.ndarray, quaternion: np.ndarray, gripper: float) -> dict[str, float]:
    rotvec = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64)).as_rotvec()
    return {
        "ee.x": float(position[0]),
        "ee.y": float(position[1]),
        "ee.z": float(position[2]),
        "ee.wx": float(rotvec[0]),
        "ee.wy": float(rotvec[1]),
        "ee.wz": float(rotvec[2]),
        "gripper.pos": float(gripper),
    }


def _joint_positions(observation: dict[str, Any]) -> list[float]:
    return [
        float(observation[f"joint_{index}.pos"])
        for index in range(1, 8)
        if f"joint_{index}.pos" in observation
    ]


def _position_of(command: dict[str, float]) -> np.ndarray:
    return np.asarray([command[key] for key in EE_POSITION_KEYS], dtype=np.float64)


def build_expert_takeover(args: argparse.Namespace, *, step_period_s: float) -> ExpertTakeover | None:
    """The same device the recorder uses, or nothing at all.

    Imported inside the function so the rehearsal still starts on a machine with no HID library
    -- with no device there is no takeover to rehearse, but the replayed trajectory and the live
    stream are still worth having while setting the rest up.
    """
    if args.no_spacemouse:
        return None
    try:
        from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
    except Exception as exc:  # noqa: BLE001 - reported, not fatal; see docstring
        emit(f"WARN: SpaceMouse unavailable ({exc}); running without takeover")
        return None

    overrides: dict[str, Any] = {"device_id": int(args.spacemouse_device_id)}
    if args.translation_scale is not None:
        overrides["translation_scale"] = float(args.translation_scale)
    if args.rotation_scale is not None:
        overrides["rotation_scale"] = float(args.rotation_scale)
    try:
        teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(**overrides))
        teleop.connect()
    except Exception as exc:  # noqa: BLE001
        emit(f"WARN: SpaceMouse did not connect ({exc}); running without takeover")
        return None
    motion_gain = motion_gain_for(tick_hz=float(teleop.config.frequency), step_period_s=step_period_s)
    emit(
        f"dagger_takeover=ready device_id={overrides['device_id']} "
        f"translation_scale={teleop.config.translation_scale:.6f} "
        f"rotation_scale={teleop.config.rotation_scale:.6f} "
        f"release_after_s={float(args.takeover_release_after_s):.2f} "
        f"motion_gain={motion_gain:.2f} "
        f"full_deflection_mm_per_step={teleop.config.translation_scale * motion_gain * 1000.0:.1f}"
    )
    return ExpertTakeover(
        teleop,
        release_after_s=float(args.takeover_release_after_s),
        motion_gain=motion_gain,
    )


def run_dryrun(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset.resolve()
    episode_data = load_episode_actions(dataset_root, args.episode)
    actions = episode_data["actions"]
    action_names = episode_data["action_names"]
    total_frames = int(actions.shape[0])
    action_positions, action_quaternions, action_source = reconstruct_absolute_pose_stream(
        action_names=action_names,
        actions=actions,
        observation_names=episode_data["observation_names"],
        observations=episode_data["observations"],
    )
    action_gripper = _column(action_names, actions, "gripper.pos")
    emit(f"Action source: {action_source}")

    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    with open(args.config_path) as config_file:
        record_cfg = draccus.load(RecordConfig, config_file)
    fps = int(args.fps or episode_data["fps"] or 30)
    robot_cfg = _replay_robot_config(record_cfg, args, fps)
    robot = make_robot_from_config(robot_cfg)

    live_frames = LiveFrameEmitter(interval=int(args.live_frame_interval))
    keyboard = InteractiveRolloutKeyboard(
        start_key=args.start_key,
        stop_key=args.stop_key,
        home_key=args.home_key,
        quit_key=args.quit_key,
        takeover_key=args.takeover_key,
    )
    takeover = build_expert_takeover(args, step_period_s=1.0 / max(fps, 1))
    if takeover is None:
        emit("No takeover device: this run rehearses the trajectory and the live stream only.")

    frame_period_s = 1.0 / max(fps, 1)
    sources: list[str] = []
    handback_gaps_mm: list[float] = []
    expert_steps = 0
    demo_index = 0
    step_index = 0
    previous_sent_command: dict[str, float] | None = None
    previous_smoothed_command: dict[str, float] | None = None
    previously_engaged = False

    robot.connect()
    keyboard.start()
    try:
        emit(
            f"fr3_dagger_sim_dryrun dataset={dataset_root.name} episode={args.episode} "
            f"frames={total_frames} fps={fps}"
        )
        _settle_to_start_pose(
            robot,
            target_position=action_positions[0],
            target_quaternion=action_quaternions[0],
            target_gripper=float(action_gripper[0]) if action_gripper is not None else 1.0,
            settle_steps=args.settle_steps,
            settle_tolerance_mm=args.settle_tolerance_mm,
            settle_period_s=0.0,
        )
        # `wait_for_command` prints the same `interactive_waiting_for_start` marker the
        # inference runtime prints, so the page's Start button, its rollout index and its
        # takeover control all work against this process unchanged. Home is answered rather
        # than ignored: the operator who presses it after a takeover means "put it back", and
        # an unanswered control is worse than an absent one.
        while True:
            requested = keyboard.wait_for_command(arm_at_start=True)
            if requested == "start":
                break
            if requested == "quit" or keyboard.quit_requested.is_set():
                return {"status": "quit", "steps": 0}
            _settle_to_start_pose(
                robot,
                target_position=action_positions[0],
                target_quaternion=action_quaternions[0],
                target_gripper=float(action_gripper[0]) if action_gripper is not None else 1.0,
                settle_steps=args.settle_steps,
                settle_tolerance_mm=args.settle_tolerance_mm,
                settle_period_s=0.0,
            )
            print("[INFO] interactive_homing=done")
        print("[INFO] interactive_rollout_start index=1")

        observation = robot.get_observation(include_cameras=False)
        while demo_index < total_frames and not keyboard.should_stop_rollout():
            loop_start_s = time.perf_counter()
            policy_command = _command_from_pose(
                action_positions[demo_index],
                action_quaternions[demo_index],
                float(action_gripper[demo_index]) if action_gripper is not None else 1.0,
            )
            guard_observation = observation_with_prev_cmd(observation, previous_sent_command)

            if takeover is None:
                robot_command, takeover_debug = policy_command, {"source": "policy"}
            else:
                # `latched` is the manual override, not the ordinary way in: moving the device
                # is. The key stays bound for an operator who wants the arm held still.
                robot_command, takeover_debug = takeover.command(
                    latched=keyboard.takeover_is_engaged(),
                    policy_command=policy_command,
                    previous_sent_command=previous_sent_command,
                    robot_observation=guard_observation,
                )
            command_source = str(takeover_debug["source"])
            if previously_engaged and command_source == "policy":
                # The distance the clamp is about to walk back. The number the rehearsal exists
                # to produce: it is the size of the discontinuity a real handback would have to
                # absorb, measured rather than assumed.
                gap_mm = float(np.linalg.norm(_position_of(policy_command) - _position_of(previous_sent_command)) * 1e3)
                handback_gaps_mm.append(gap_mm)
                emit(f"dagger_handback step={step_index} handback_gap_mm={gap_mm:.1f}")
            previously_engaged = command_source == "expert"

            robot_command = smooth_robot_command_ema(
                robot_command, previous_smoothed_command, alpha=args.command_ema_alpha
            )
            previous_smoothed_command = dict(robot_command)
            command_to_send, guard = limit_command_for_safety(
                robot_command,
                guard_observation,
                max_step_pos_delta_m=float(args.max_step_pos_delta_mm) / 1000.0,
                max_step_rot_delta_rad=float(np.deg2rad(args.max_step_rot_delta_deg)),
                max_leash_pos_delta_m=float(args.max_leash_pos_delta_mm) / 1000.0,
                max_leash_rot_delta_rad=float(np.deg2rad(args.max_leash_rot_delta_deg)),
            )
            robot.send_action(command_to_send)
            previous_sent_command = dict(command_to_send)
            observation = robot.get_observation(include_cameras=False)

            sources.append(command_source)
            if command_source == "expert":
                expert_steps += 1
            else:
                # Frozen while the operator drives -- see the module docstring.
                demo_index += 1

            live_frames.emit_step(
                step_index,
                joints_rad=lambda: _joint_positions(observation),
                gripper=lambda: float(observation.get("gripper.pos", command_to_send["gripper.pos"])),
                source=command_source,
                status=str(guard["status"]),
                rollout_index=1,
                target_position_m=_position_of(command_to_send),
                actual_position_m=[float(observation[key]) for key in EE_POSITION_KEYS],
            )
            if args.log_interval > 0 and step_index % args.log_interval == 0:
                emit(
                    f"step={step_index} demo_frame={demo_index}/{total_frames} "
                    f"source={command_source} status={guard['status']} "
                    f"takeover={takeover_debug.get('status', '')} "
                    f"step_mm={takeover_debug.get('step_mm', 0.0):.1f} "
                    f"gripper_cmd={command_to_send['gripper.pos']:.3f}"
                )
            step_index += 1

            remaining_s = frame_period_s - (time.perf_counter() - loop_start_s)
            if remaining_s > 0:
                time.sleep(remaining_s)
    finally:
        keyboard.close()
        if takeover is not None:
            takeover.close()
        if robot.is_connected:
            robot.disconnect()

    spans = expert_spans(sources)
    result = {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "episode": int(args.episode),
        "fps": fps,
        "steps": step_index,
        "demo_frames_consumed": demo_index,
        "demo_frames_total": total_frames,
        "expert_steps": expert_steps,
        "expert_spans": spans,
        # The clamp the run was bounded by, recorded rather than assumed: a handback gap only
        # means something next to the limits that produced it, and the launcher passes different
        # ones than this script's own defaults.
        "limits": {
            "max_step_pos_delta_mm": float(args.max_step_pos_delta_mm),
            "max_step_rot_delta_deg": float(args.max_step_rot_delta_deg),
            "max_leash_pos_delta_mm": float(args.max_leash_pos_delta_mm),
            "max_leash_rot_delta_deg": float(args.max_leash_rot_delta_deg),
            "command_ema_alpha": (
                None if args.command_ema_alpha is None else float(args.command_ema_alpha)
            ),
        },
        "handback_gaps_mm": handback_gaps_mm,
        "max_handback_gap_mm": max(handback_gaps_mm) if handback_gaps_mm else 0.0,
        "status": "completed" if demo_index >= total_frames else "stopped",
    }
    # Same field names the rollout end marker uses, so a rehearsal and a rollout read alike.
    print(
        f"[INFO] interactive_rollout_end index=1 status={result['status']} "
        f"samples={step_index}"
        + (
            ""
            if not spans
            else (
                f" intervened=1 expert_steps={expert_steps} "
                + "expert_spans=" + ";".join(f"{first}-{last}" for first, last in spans)
                + f" max_handback_gap_mm={result['max_handback_gap_mm']:.1f}"
            )
        )
    )
    print("[INFO] interactive_rollouts=stopped")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(f"dagger_sim_dryrun_report={args.output}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rehearse the DAgger takeover against a simulated FR3.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--config-path", dest="config_path", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--settle-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--ik-orientation-weight", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--live-frame-interval",
        type=int,
        default=1,
        help="Publish one live_frame= line every N steps for the browser. 0 disables it.",
    )
    # The rehearsal is only worth trusting if these match what the rollout runs with. Defaults
    # are the launcher's, and any override belongs in both places at once.
    parser.add_argument("--command-ema-alpha", type=float, default=None)
    parser.add_argument("--max-step-pos-delta-mm", type=float, default=6.0)
    parser.add_argument("--max-step-rot-delta-deg", type=float, default=3.0)
    parser.add_argument("--max-leash-pos-delta-mm", type=float, default=60.0)
    parser.add_argument("--max-leash-rot-delta-deg", type=float, default=25.0)
    parser.add_argument("--start-key", default="s")
    parser.add_argument("--stop-key", default="x")
    parser.add_argument("--home-key", default="h")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--takeover-key", default="t")
    parser.add_argument(
        "--takeover-release-after-s",
        type=float,
        default=1.0,
        help=(
            "Hand the arm back to the demo stream once the SpaceMouse has been quiet this long. "
            "0 disables automatic takeover, leaving only the takeover key."
        ),
    )
    parser.add_argument("--spacemouse-device-id", type=int, default=0)
    parser.add_argument("--translation-scale", type=float, default=None)
    parser.add_argument("--rotation-scale", type=float, default=None)
    parser.add_argument(
        "--no-spacemouse",
        action="store_true",
        help="Run the trajectory and the live stream with no takeover device attached.",
    )
    args = parser.parse_args(argv)
    # `_replay_robot_config` reads this to choose the simulated arm; there is no real backend
    # here, because rehearsing a handoff on hardware is the thing this exists to postpone.
    args.backend = "sim"
    args.robot_ip = ""
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_dryrun(args)
    return 0 if result.get("status") in ("completed", "stopped", "quit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
