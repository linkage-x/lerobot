#!/usr/bin/env python3
"""
离线分析 FR3 DAS replay 峰值帧的 state/action/IK joint target 差异。

示例：
    PYTHONPATH=/home/hph/Code/lerobot-replay/src \
    python3 tools/fr3/fr3_analyze_peak_segments.py \
        --episode 0 \
        --dataset outputs/datasets/lerobotv3_0310_100ep \
        --frames 160,162,163,240,264,265
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
DEFAULT_FRAMES = (160, 162, 163, 240, 264, 265)


def parse_frame_list(value: str) -> list[int]:
    try:
        frames = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer frame indices.") from exc
    if not frames:
        raise argparse.ArgumentTypeError("Expected at least one frame index.")
    if len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("Frame indices must be unique.")
    return frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline FR3 DAS peak-frame state/action/joint-target analysis")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--frames",
        type=parse_frame_list,
        default=list(DEFAULT_FRAMES),
        help=f"Comma-separated frame indices (default: {','.join(str(frame) for frame in DEFAULT_FRAMES)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory for generated CSV/Markdown tables.",
    )
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


def analyze_episode(
    dataset_path: str,
    episode_idx: int,
    frames: list[int],
) -> list[dict[str, object]]:
    from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

    ep = load_episode(dataset_path, episode_idx)
    states = ep["state"]
    actions = ep["action"]
    timestamps = ep["timestamp"]
    n_frames = len(states)

    bad_frames = [frame for frame in frames if frame < 0 or frame >= n_frames]
    if bad_frames:
        raise ValueError(f"Frames out of range for episode {episode_idx}: {bad_frames} (n_frames={n_frames})")

    kin = PlacoKinematicsDriver(
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        joint_names=_JOINT_NAMES,
    )

    T_B_E_reset = pose_from_xyzquat(_RESET_POSE_B_XYZQUAT)
    T_B_Ws = build_T_B_Ws(T_B_E_reset, pose_from_xyzquat(states[0]))

    state_joint_targets_rad: list[np.ndarray] = []
    action_joint_targets_rad: list[np.ndarray] = []
    prev_state_joints = _IK_SEED_JOINTS_RAD.copy()
    prev_action_joints = _IK_SEED_JOINTS_RAD.copy()

    for fi in range(n_frames):
        state_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(states[fi]))
        action_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(actions[fi]))
        state_joint_target = np.asarray(kin.inverse_kinematics(prev_state_joints, state_pose), dtype=np.float64)
        action_joint_target = np.asarray(kin.inverse_kinematics(prev_action_joints, action_pose), dtype=np.float64)
        state_joint_targets_rad.append(state_joint_target.copy())
        action_joint_targets_rad.append(action_joint_target.copy())
        prev_state_joints = state_joint_target.copy()
        prev_action_joints = action_joint_target.copy()

    results: list[dict[str, object]] = []
    for fi in frames:
        prev_fi = max(fi - 1, 0)
        state_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(states[fi]))
        action_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(actions[fi]))
        prev_state_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(states[prev_fi]))
        prev_action_pose = ws_to_base(T_B_Ws, pose_from_xyzquat(actions[prev_fi]))

        state_pos = state_pose[:3, 3]
        action_pos = action_pose[:3, 3]
        state_rot = state_pose[:3, :3]
        action_rot = action_pose[:3, :3]
        prev_state_pos = prev_state_pose[:3, 3]
        prev_action_pos = prev_action_pose[:3, 3]
        prev_state_rot = prev_state_pose[:3, :3]
        prev_action_rot = prev_action_pose[:3, :3]

        state_joint_deg = joint_values_to_deg(state_joint_targets_rad[fi])
        action_joint_deg = joint_values_to_deg(action_joint_targets_rad[fi])
        raw_joint_delta_deg = action_joint_deg - state_joint_deg
        wrapped_joint_delta_deg = wrap_angle_deg(raw_joint_delta_deg)
        state_joint_step_deg = wrap_angle_deg(state_joint_deg - joint_values_to_deg(state_joint_targets_rad[prev_fi]))
        action_joint_step_deg = wrap_angle_deg(action_joint_deg - joint_values_to_deg(action_joint_targets_rad[prev_fi]))

        results.append(
            {
                "frame": fi,
                "timestamp_s": float(timestamps[fi]),
                "dt_prev_ms": float(max(timestamps[fi] - timestamps[prev_fi], 0.0) * 1000.0),
                "state_z_m": float(state_pos[2]),
                "action_z_m": float(action_pos[2]),
                "state_action_pos_gap_mm": float(np.linalg.norm(state_pos - action_pos) * 1000.0),
                "state_action_rot_gap_deg": rotation_angle_error_deg(state_rot, action_rot),
                "state_step_mm": float(np.linalg.norm(state_pos - prev_state_pos) * 1000.0),
                "action_step_mm": float(np.linalg.norm(action_pos - prev_action_pos) * 1000.0),
                "state_step_rot_deg": rotation_angle_error_deg(prev_state_rot, state_rot),
                "action_step_rot_deg": rotation_angle_error_deg(prev_action_rot, action_rot),
                "joint_target_gap_raw_l2_deg": float(np.linalg.norm(raw_joint_delta_deg)),
                "joint_target_gap_raw_max_abs_deg": float(np.max(np.abs(raw_joint_delta_deg))),
                "joint_target_gap_wrapped_l2_deg": float(np.linalg.norm(wrapped_joint_delta_deg)),
                "joint_target_gap_wrapped_max_abs_deg": float(np.max(np.abs(wrapped_joint_delta_deg))),
                "state_joint_step_l2_deg": float(np.linalg.norm(state_joint_step_deg)),
                "action_joint_step_l2_deg": float(np.linalg.norm(action_joint_step_deg)),
                "state_joint_targets_deg": state_joint_deg,
                "action_joint_targets_deg": action_joint_deg,
                "joint_target_delta_raw_deg": raw_joint_delta_deg,
                "joint_target_delta_wrapped_deg": wrapped_joint_delta_deg,
            }
        )

    return results


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "frame",
        "timestamp_s",
        "dt_prev_ms",
        "state_z_m",
        "action_z_m",
        "state_action_pos_gap_mm",
        "state_action_rot_gap_deg",
        "state_step_mm",
        "action_step_mm",
        "state_step_rot_deg",
        "action_step_rot_deg",
        "joint_target_gap_raw_l2_deg",
        "joint_target_gap_raw_max_abs_deg",
        "joint_target_gap_wrapped_l2_deg",
        "joint_target_gap_wrapped_max_abs_deg",
        "state_joint_step_l2_deg",
        "action_joint_step_l2_deg",
    ]
    for prefix in ("state_joint", "action_joint", "delta_raw_joint", "delta_wrapped_joint"):
        for joint_idx in range(1, 8):
            header.append(f"{prefix}_{joint_idx}_deg")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            flat_row = {key: row[key] for key in header if key in row}
            for joint_idx, value in enumerate(row["state_joint_targets_deg"], start=1):
                flat_row[f"state_joint_{joint_idx}_deg"] = f"{float(value):.6f}"
            for joint_idx, value in enumerate(row["action_joint_targets_deg"], start=1):
                flat_row[f"action_joint_{joint_idx}_deg"] = f"{float(value):.6f}"
            for joint_idx, value in enumerate(row["joint_target_delta_raw_deg"], start=1):
                flat_row[f"delta_raw_joint_{joint_idx}_deg"] = f"{float(value):.6f}"
            for joint_idx, value in enumerate(row["joint_target_delta_wrapped_deg"], start=1):
                flat_row[f"delta_wrapped_joint_{joint_idx}_deg"] = f"{float(value):.6f}"
            writer.writerow(flat_row)


def write_markdown(rows: list[dict[str, object]], output_path: Path, dataset_path: str, episode_idx: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_headers = [
        "frame",
        "ts(s)",
        "dt_prev(ms)",
        "ee_gap(mm)",
        "ee_gap(deg)",
        "state_step(mm)",
        "action_step(mm)",
        "joint_gap_wrap_l2(deg)",
        "joint_gap_wrap_max(deg)",
        "joint_gap_raw_max(deg)",
    ]
    summary_rows = [
        [
            str(int(row["frame"])),
            f"{float(row['timestamp_s']):.3f}",
            f"{float(row['dt_prev_ms']):.2f}",
            f"{float(row['state_action_pos_gap_mm']):.2f}",
            f"{float(row['state_action_rot_gap_deg']):.2f}",
            f"{float(row['state_step_mm']):.2f}",
            f"{float(row['action_step_mm']):.2f}",
            f"{float(row['joint_target_gap_wrapped_l2_deg']):.2f}",
            f"{float(row['joint_target_gap_wrapped_max_abs_deg']):.2f}",
            f"{float(row['joint_target_gap_raw_max_abs_deg']):.2f}",
        ]
        for row in rows
    ]

    joint_headers = ["frame", "state joints (deg)", "action joints (deg)", "delta wrapped (deg)"]
    joint_rows = [
        [
            str(int(row["frame"])),
            format_joint_vector_deg(np.asarray(row["state_joint_targets_deg"], dtype=np.float64)),
            format_joint_vector_deg(np.asarray(row["action_joint_targets_deg"], dtype=np.float64)),
            format_joint_vector_deg(np.asarray(row["joint_target_delta_wrapped_deg"], dtype=np.float64)),
        ]
        for row in rows
    ]

    content = [
        f"# FR3 Peak Segment Analysis",
        "",
        f"- dataset: `{dataset_path}`",
        f"- episode: `{episode_idx}`",
        f"- frames: `{','.join(str(int(row['frame'])) for row in rows)}`",
        "",
        "## Summary",
        "",
        markdown_table(summary_headers, summary_rows),
        "",
        "## Joint Targets",
        "",
        markdown_table(joint_headers, joint_rows),
        "",
        "## Notes",
        "",
        "- `state/action` 的 EE pose 差异使用同一套 `T(B,W_s)` 与 TCP 外参计算。",
        "- `state joints` 与 `action joints` 分别用独立的顺序 IK 流计算，二者都从 DAS 起始关节角开始滚动。",
        "- `delta wrapped (deg)` 是 `action joints - state joints` 映射到 `[-180, 180]` 的最短角差。",
        "- `joint_gap_raw_max(deg)` 保留未 wrap 的最大关节差，用来暴露 IK 分支翻转或 360° 周期跳变。",
    ]
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = str((Path.cwd() / args.dataset).resolve()) if not Path(args.dataset).is_absolute() else args.dataset
    output_dir = args.output_dir.resolve()
    rows = analyze_episode(dataset_path=dataset_path, episode_idx=args.episode, frames=args.frames)

    stem = f"fr3_peak_analysis_ep{args.episode:03d}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, dataset_path, args.episode)

    print(f"[INFO] 已导出 CSV: {csv_path}")
    print(f"[INFO] 已导出 Markdown: {md_path}")
    print("[INFO] 摘要：")
    for row in rows:
        print(
            f"  frame={int(row['frame']):4d}  "
            f"ee_gap={float(row['state_action_pos_gap_mm']):6.2f}mm/{float(row['state_action_rot_gap_deg']):5.2f}deg  "
            f"joint_gap_wrap={float(row['joint_target_gap_wrapped_l2_deg']):5.2f}deg  "
            f"joint_gap_wrap_max={float(row['joint_target_gap_wrapped_max_abs_deg']):5.2f}deg  "
            f"joint_gap_raw_max={float(row['joint_target_gap_raw_max_abs_deg']):5.2f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
