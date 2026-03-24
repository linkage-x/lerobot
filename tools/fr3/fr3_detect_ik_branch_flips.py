#!/usr/bin/env python3
"""
全 episode 离线检测 FR3 replay 中的 IK branch-flip 候选事件。

输出三类候选:
- cross_stream_divergence: state/action EE gap 较小，但 joint target 差异极大
- state_stream_jump: state 连续帧 EE step 较小，但 state joint target 步长极大
- action_stream_jump: action 连续帧 EE step 较小，但 action joint target 步长极大
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fr3_das_replay_runtime import (
    _DAS_URDF,
    _IK_SEED_JOINTS_RAD,
    _JOINT_NAMES,
    _RESET_POSE_B_XYZQUAT,
    build_T_B_Ws,
    load_episode,
    pose_from_xyzquat,
    rotation_angle_error_deg,
    ws_to_base,
)


DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive float.") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline FR3 IK branch-flip detector")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--cross-ee-gap-mm-max", type=positive_float, default=25.0)
    parser.add_argument("--cross-ee-gap-deg-max", type=positive_float, default=2.0)
    parser.add_argument("--cross-joint-gap-wrap-max-min", type=positive_float, default=120.0)
    parser.add_argument("--cross-joint-gap-wrap-l2-min", type=positive_float, default=180.0)
    parser.add_argument("--stream-ee-step-mm-max", type=positive_float, default=25.0)
    parser.add_argument("--stream-ee-step-deg-max", type=positive_float, default=2.0)
    parser.add_argument("--stream-joint-step-wrap-max-min", type=positive_float, default=120.0)
    parser.add_argument("--stream-joint-step-wrap-l2-min", type=positive_float, default=180.0)
    return parser.parse_args(argv)


def joint_values_to_deg(values_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.asarray(values_rad, dtype=np.float64))


def wrap_angle_deg(values_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(values_deg, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def format_joint_vector_deg(values_deg: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.2f}" for value in values_deg.tolist()) + "]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_joint_target_sequences(
    dataset_path: str,
    episode_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

    ep = load_episode(dataset_path, episode_idx)
    states = ep["state"]
    actions = ep["action"]
    timestamps = ep["timestamp"]

    kin = PlacoKinematicsDriver(
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        joint_names=_JOINT_NAMES,
    )

    T_B_E_reset = pose_from_xyzquat(_RESET_POSE_B_XYZQUAT)
    T_B_Ws = build_T_B_Ws(T_B_E_reset, pose_from_xyzquat(states[0]))

    state_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in states], dtype=np.float64)
    action_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in actions], dtype=np.float64)

    state_joint_targets: list[np.ndarray] = []
    action_joint_targets: list[np.ndarray] = []
    prev_state_joints = _IK_SEED_JOINTS_RAD.copy()
    prev_action_joints = _IK_SEED_JOINTS_RAD.copy()

    for frame_idx in range(len(states)):
        state_joint_target = np.asarray(kin.inverse_kinematics(prev_state_joints, state_poses[frame_idx]), dtype=np.float64)
        action_joint_target = np.asarray(kin.inverse_kinematics(prev_action_joints, action_poses[frame_idx]), dtype=np.float64)
        state_joint_targets.append(state_joint_target.copy())
        action_joint_targets.append(action_joint_target.copy())
        prev_state_joints = state_joint_target.copy()
        prev_action_joints = action_joint_target.copy()

    return states, actions, timestamps, np.asarray(state_poses), np.asarray(action_poses), np.asarray(state_joint_targets), np.asarray(action_joint_targets)


def detect_branch_flips(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, int]]:
    dataset_path = str((Path.cwd() / args.dataset).resolve()) if not Path(args.dataset).is_absolute() else args.dataset
    (
        _states,
        _actions,
        timestamps,
        state_poses,
        action_poses,
        state_joint_targets_rad,
        action_joint_targets_rad,
    ) = build_joint_target_sequences(dataset_path, args.episode)

    state_joint_deg = joint_values_to_deg(state_joint_targets_rad)
    action_joint_deg = joint_values_to_deg(action_joint_targets_rad)
    n_frames = len(timestamps)
    events: list[dict[str, object]] = []
    counts = {
        "cross_stream_divergence": 0,
        "state_stream_jump": 0,
        "action_stream_jump": 0,
    }

    for fi in range(1, n_frames):
        prev_fi = fi - 1
        state_pos = state_poses[fi][:3, 3]
        action_pos = action_poses[fi][:3, 3]
        prev_state_pos = state_poses[prev_fi][:3, 3]
        prev_action_pos = action_poses[prev_fi][:3, 3]
        state_rot = state_poses[fi][:3, :3]
        action_rot = action_poses[fi][:3, :3]
        prev_state_rot = state_poses[prev_fi][:3, :3]
        prev_action_rot = action_poses[prev_fi][:3, :3]

        ee_gap_mm = float(np.linalg.norm(state_pos - action_pos) * 1000.0)
        ee_gap_deg = rotation_angle_error_deg(state_rot, action_rot)
        state_step_mm = float(np.linalg.norm(state_pos - prev_state_pos) * 1000.0)
        action_step_mm = float(np.linalg.norm(action_pos - prev_action_pos) * 1000.0)
        state_step_deg = rotation_angle_error_deg(prev_state_rot, state_rot)
        action_step_deg = rotation_angle_error_deg(prev_action_rot, action_rot)

        cross_delta_raw_deg = action_joint_deg[fi] - state_joint_deg[fi]
        cross_delta_wrap_deg = wrap_angle_deg(cross_delta_raw_deg)
        cross_wrap_l2 = float(np.linalg.norm(cross_delta_wrap_deg))
        cross_wrap_max = float(np.max(np.abs(cross_delta_wrap_deg)))
        cross_raw_max = float(np.max(np.abs(cross_delta_raw_deg)))

        state_step_joint_wrap_deg = wrap_angle_deg(state_joint_deg[fi] - state_joint_deg[prev_fi])
        action_step_joint_wrap_deg = wrap_angle_deg(action_joint_deg[fi] - action_joint_deg[prev_fi])
        state_step_joint_l2 = float(np.linalg.norm(state_step_joint_wrap_deg))
        state_step_joint_max = float(np.max(np.abs(state_step_joint_wrap_deg)))
        action_step_joint_l2 = float(np.linalg.norm(action_step_joint_wrap_deg))
        action_step_joint_max = float(np.max(np.abs(action_step_joint_wrap_deg)))

        if (
            ee_gap_mm <= args.cross_ee_gap_mm_max
            and ee_gap_deg <= args.cross_ee_gap_deg_max
            and (
                cross_wrap_max >= args.cross_joint_gap_wrap_max_min
                or cross_wrap_l2 >= args.cross_joint_gap_wrap_l2_min
            )
        ):
            counts["cross_stream_divergence"] += 1
            events.append(
                {
                    "event_type": "cross_stream_divergence",
                    "frame": fi,
                    "timestamp_s": float(timestamps[fi]),
                    "dt_prev_ms": float((timestamps[fi] - timestamps[prev_fi]) * 1000.0),
                    "ee_gap_mm": ee_gap_mm,
                    "ee_gap_deg": ee_gap_deg,
                    "state_step_mm": state_step_mm,
                    "action_step_mm": action_step_mm,
                    "state_step_deg": state_step_deg,
                    "action_step_deg": action_step_deg,
                    "joint_wrap_l2_deg": cross_wrap_l2,
                    "joint_wrap_max_deg": cross_wrap_max,
                    "joint_raw_max_deg": cross_raw_max,
                    "joint_delta_wrapped_deg": cross_delta_wrap_deg,
                }
            )

        if (
            state_step_mm <= args.stream_ee_step_mm_max
            and state_step_deg <= args.stream_ee_step_deg_max
            and (
                state_step_joint_max >= args.stream_joint_step_wrap_max_min
                or state_step_joint_l2 >= args.stream_joint_step_wrap_l2_min
            )
        ):
            counts["state_stream_jump"] += 1
            events.append(
                {
                    "event_type": "state_stream_jump",
                    "frame": fi,
                    "timestamp_s": float(timestamps[fi]),
                    "dt_prev_ms": float((timestamps[fi] - timestamps[prev_fi]) * 1000.0),
                    "ee_gap_mm": ee_gap_mm,
                    "ee_gap_deg": ee_gap_deg,
                    "state_step_mm": state_step_mm,
                    "action_step_mm": action_step_mm,
                    "state_step_deg": state_step_deg,
                    "action_step_deg": action_step_deg,
                    "joint_wrap_l2_deg": state_step_joint_l2,
                    "joint_wrap_max_deg": state_step_joint_max,
                    "joint_raw_max_deg": state_step_joint_max,
                    "joint_delta_wrapped_deg": state_step_joint_wrap_deg,
                }
            )

        if (
            action_step_mm <= args.stream_ee_step_mm_max
            and action_step_deg <= args.stream_ee_step_deg_max
            and (
                action_step_joint_max >= args.stream_joint_step_wrap_max_min
                or action_step_joint_l2 >= args.stream_joint_step_wrap_l2_min
            )
        ):
            counts["action_stream_jump"] += 1
            events.append(
                {
                    "event_type": "action_stream_jump",
                    "frame": fi,
                    "timestamp_s": float(timestamps[fi]),
                    "dt_prev_ms": float((timestamps[fi] - timestamps[prev_fi]) * 1000.0),
                    "ee_gap_mm": ee_gap_mm,
                    "ee_gap_deg": ee_gap_deg,
                    "state_step_mm": state_step_mm,
                    "action_step_mm": action_step_mm,
                    "state_step_deg": state_step_deg,
                    "action_step_deg": action_step_deg,
                    "joint_wrap_l2_deg": action_step_joint_l2,
                    "joint_wrap_max_deg": action_step_joint_max,
                    "joint_raw_max_deg": action_step_joint_max,
                    "joint_delta_wrapped_deg": action_step_joint_wrap_deg,
                }
            )

    events.sort(key=lambda item: (item["event_type"], item["frame"]))
    return events, counts


def write_csv(events: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "event_type",
        "frame",
        "timestamp_s",
        "dt_prev_ms",
        "ee_gap_mm",
        "ee_gap_deg",
        "state_step_mm",
        "action_step_mm",
        "state_step_deg",
        "action_step_deg",
        "joint_wrap_l2_deg",
        "joint_wrap_max_deg",
        "joint_raw_max_deg",
    ]
    for joint_idx in range(1, 8):
        header.append(f"joint_delta_wrapped_{joint_idx}_deg")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for event in events:
            row = {key: event[key] for key in header if key in event}
            for joint_idx, value in enumerate(event["joint_delta_wrapped_deg"], start=1):
                row[f"joint_delta_wrapped_{joint_idx}_deg"] = f"{float(value):.6f}"
            writer.writerow(row)


def write_markdown(
    events: list[dict[str, object]],
    counts: dict[str, int],
    output_path: Path,
    dataset_path: str,
    episode_idx: int,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_headers = ["type", "count"]
    summary_rows = [[event_type, str(count)] for event_type, count in counts.items()]

    top_headers = [
        "type",
        "frame",
        "ts(s)",
        "ee_gap(mm)",
        "ee_gap(deg)",
        "state_step(mm)",
        "action_step(mm)",
        "joint_wrap_l2(deg)",
        "joint_wrap_max(deg)",
    ]
    top_rows = [
        [
            str(event["event_type"]),
            str(int(event["frame"])),
            f"{float(event['timestamp_s']):.3f}",
            f"{float(event['ee_gap_mm']):.2f}",
            f"{float(event['ee_gap_deg']):.2f}",
            f"{float(event['state_step_mm']):.2f}",
            f"{float(event['action_step_mm']):.2f}",
            f"{float(event['joint_wrap_l2_deg']):.2f}",
            f"{float(event['joint_wrap_max_deg']):.2f}",
        ]
        for event in events
    ]

    content = [
        "# FR3 IK Branch-Flip Detection",
        "",
        f"- dataset: `{dataset_path}`",
        f"- episode: `{episode_idx}`",
        "",
        "## Thresholds",
        "",
        f"- cross-stream: `ee_gap_mm <= {args.cross_ee_gap_mm_max}` and `ee_gap_deg <= {args.cross_ee_gap_deg_max}` and (`joint_wrap_max >= {args.cross_joint_gap_wrap_max_min}` or `joint_wrap_l2 >= {args.cross_joint_gap_wrap_l2_min}`)",
        f"- stream jump: `ee_step_mm <= {args.stream_ee_step_mm_max}` and `ee_step_deg <= {args.stream_ee_step_deg_max}` and (`joint_wrap_max >= {args.stream_joint_step_wrap_max_min}` or `joint_wrap_l2 >= {args.stream_joint_step_wrap_l2_min}`)",
        "",
        "## Event Counts",
        "",
        markdown_table(summary_headers, summary_rows),
        "",
        "## Detected Events",
        "",
        markdown_table(top_headers, top_rows) if top_rows else "_No events detected._",
        "",
        "## Notes",
        "",
        "- `cross_stream_divergence` means `state` and `action` are close in EE space but far apart in IK joint space.",
        "- `state_stream_jump` and `action_stream_jump` mean a single stream changed IK solution abruptly between adjacent frames despite a small EE step.",
        "- `joint_wrap_*` metrics use shortest-path angle differences in `[-180, 180]`.",
    ]
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = str((Path.cwd() / args.dataset).resolve()) if not Path(args.dataset).is_absolute() else args.dataset
    output_dir = args.output_dir.resolve()
    events, counts = detect_branch_flips(args)

    stem = f"fr3_ik_branch_flips_ep{args.episode:03d}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    write_csv(events, csv_path)
    write_markdown(events, counts, md_path, dataset_path, args.episode, args)

    print(f"[INFO] 已导出 CSV: {csv_path}")
    print(f"[INFO] 已导出 Markdown: {md_path}")
    print("[INFO] 事件计数：")
    for event_type, count in counts.items():
        print(f"  {event_type}: {count}")
    print("[INFO] Top events:")
    for event in events[:20]:
        print(
            f"  {event['event_type']:24s} frame={int(event['frame']):4d}  "
            f"ee_gap={float(event['ee_gap_mm']):6.2f}mm/{float(event['ee_gap_deg']):5.2f}deg  "
            f"joint_wrap={float(event['joint_wrap_l2_deg']):6.2f}deg  "
            f"joint_wrap_max={float(event['joint_wrap_max_deg']):6.2f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
