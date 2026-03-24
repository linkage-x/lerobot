#!/usr/bin/env python3
"""
统一验证 FR3 joint-target streams 的离线仿真 replay 误差。

输入一份 joint target CSV（通常来自 fr3_generate_branch_consistent_targets.py），
对 `naive_joint`、`bc_joint`、`state_ref_joint` 三条流统一做：

- FK vs action pose 误差
- FK vs state pose 误差
- joint step 连续性统计
- 坏帧计数与峰值帧表
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fr3_das_replay_real_runtime import (
    _DAS_URDF,
    _IK_SEED_JOINTS_RAD,
    _JOINT_NAMES,
    _RESET_POSE_B_XYZQUAT,
    _T_IE,
    load_episode,
    pose_from_xyzquat,
    rotation_angle_error_deg,
    se3_inv,
    ws_to_base,
)
from fr3_generate_branch_consistent_targets import parse_xyzquat


DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"
DEFAULT_OUTPUT_DIR = Path("outputs/analysis")
DEFAULT_STREAMS = ("naive_joint", "bc_joint", "state_ref_joint")
DEFAULT_WARMUP = 30
DEFAULT_PEAK_ROWS = 8


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive float.") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_streams(value: str) -> list[str]:
    streams = [part.strip() for part in value.split(",") if part.strip()]
    if not streams:
        raise argparse.ArgumentTypeError("Expected at least one stream prefix.")
    return streams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FR3 joint-target replay streams offline")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--joint-targets-csv",
        default=None,
        help="Joint target CSV. Defaults to outputs/analysis/fr3_branch_consistent_targets_epXXX.csv.",
    )
    parser.add_argument("--streams", type=parse_streams, default=list(DEFAULT_STREAMS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--peak-rows", type=positive_int, default=DEFAULT_PEAK_ROWS)
    parser.add_argument("--bad-pos-mm", type=positive_float, default=20.0)
    parser.add_argument("--bad-rot-deg", type=positive_float, default=5.0)
    parser.add_argument("--bad-joint-step-max-deg", type=positive_float, default=45.0)
    parser.add_argument(
        "--start-pose-b-xyzquat",
        type=parse_xyzquat,
        default=None,
        help="Measured FR3 EE start pose in base frame: x,y,z,qx,qy,qz,qw. "
        "If omitted, fall back to the static reset pose.",
    )
    return parser.parse_args(argv)


def resolve_dataset_path(dataset: str) -> str:
    path = Path(dataset)
    return str(path if path.is_absolute() else (Path.cwd() / path).resolve())


def resolve_joint_targets_csv(args: argparse.Namespace) -> Path:
    if args.joint_targets_csv is not None:
        path = Path(args.joint_targets_csv)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return (Path.cwd() / "outputs" / "analysis" / f"fr3_branch_consistent_targets_ep{args.episode:03d}.csv").resolve()


def wrap_angle_deg(values_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(values_deg, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def summarize(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.max()), float(np.percentile(arr, 95))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_joint_sequences(csv_path: Path, streams: list[str], n_frames: int) -> dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Joint target CSV not found: {csv_path}")

    expected_columns = {
        stream: [f"{stream}_{joint_idx}_deg" for joint_idx in range(1, 8)] for stream in streams
    }
    values_by_stream: dict[str, list[np.ndarray]] = {stream: [] for stream in streams}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for stream in streams for col in expected_columns[stream] if col not in fieldnames]
        if missing:
            raise ValueError(f"Missing joint columns in {csv_path}: {missing}")

        for expected_frame, row in enumerate(reader):
            frame_text = row.get("frame")
            if frame_text is not None and int(frame_text) != expected_frame:
                raise ValueError(
                    f"Unexpected frame index in {csv_path}: expected {expected_frame}, got {frame_text}"
                )
            for stream in streams:
                values_by_stream[stream].append(
                    np.asarray([float(row[col]) for col in expected_columns[stream]], dtype=np.float64)
                )

    result: dict[str, np.ndarray] = {}
    for stream, rows in values_by_stream.items():
        arr_deg = np.asarray(rows, dtype=np.float64)
        if arr_deg.shape != (n_frames, 7):
            raise ValueError(f"Expected shape {(n_frames, 7)} for {stream}, got {arr_deg.shape}")
        result[stream] = np.deg2rad(arr_deg)
    return result


@dataclass
class StreamResult:
    stream: str
    pos_action_mm: np.ndarray
    rot_action_deg: np.ndarray
    pos_state_mm: np.ndarray
    rot_state_deg: np.ndarray
    joint_step_l2_deg: np.ndarray
    joint_step_max_deg: np.ndarray
    bad_action_frames: int
    bad_state_frames: int
    bad_joint_step_frames: int
    peak_indices: np.ndarray


def evaluate_streams(
    *,
    state_poses: np.ndarray,
    action_poses: np.ndarray,
    joint_streams_rad: dict[str, np.ndarray],
    peak_rows: int,
    bad_pos_mm: float,
    bad_rot_deg: float,
    bad_joint_step_max_deg: float,
) -> list[StreamResult]:
    from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

    kin = PlacoKinematicsDriver(
        urdf_path=str(_DAS_URDF),
        target_frame_name="das_gripper_ee",
        joint_names=_JOINT_NAMES,
    )

    results: list[StreamResult] = []
    for stream, joints_rad in joint_streams_rad.items():
        pos_action_mm: list[float] = []
        rot_action_deg: list[float] = []
        pos_state_mm: list[float] = []
        rot_state_deg: list[float] = []
        joint_step_l2_deg: list[float] = []
        joint_step_max_deg: list[float] = []

        prev_joint_deg = None
        for fi in range(len(joints_rad)):
            q_rad = np.asarray(joints_rad[fi], dtype=np.float64)
            q_deg = np.rad2deg(q_rad)
            fk_pose = np.asarray(kin.forward_kinematics(q_rad), dtype=np.float64)

            action_pose = action_poses[fi]
            state_pose = state_poses[fi]
            pos_action_mm.append(float(np.linalg.norm(fk_pose[:3, 3] - action_pose[:3, 3]) * 1000.0))
            rot_action_deg.append(rotation_angle_error_deg(fk_pose[:3, :3], action_pose[:3, :3]))
            pos_state_mm.append(float(np.linalg.norm(fk_pose[:3, 3] - state_pose[:3, 3]) * 1000.0))
            rot_state_deg.append(rotation_angle_error_deg(fk_pose[:3, :3], state_pose[:3, :3]))

            if prev_joint_deg is None:
                joint_step_l2_deg.append(0.0)
                joint_step_max_deg.append(0.0)
            else:
                step_deg = wrap_angle_deg(q_deg - prev_joint_deg)
                joint_step_l2_deg.append(float(np.linalg.norm(step_deg)))
                joint_step_max_deg.append(float(np.max(np.abs(step_deg))))
            prev_joint_deg = q_deg.copy()

        pos_action_mm_arr = np.asarray(pos_action_mm, dtype=np.float64)
        rot_action_deg_arr = np.asarray(rot_action_deg, dtype=np.float64)
        pos_state_mm_arr = np.asarray(pos_state_mm, dtype=np.float64)
        rot_state_deg_arr = np.asarray(rot_state_deg, dtype=np.float64)
        joint_step_l2_deg_arr = np.asarray(joint_step_l2_deg, dtype=np.float64)
        joint_step_max_deg_arr = np.asarray(joint_step_max_deg, dtype=np.float64)
        peak_score = np.maximum(pos_action_mm_arr, pos_state_mm_arr)
        peak_indices = np.argsort(peak_score)[-peak_rows:][::-1]

        results.append(
            StreamResult(
                stream=stream,
                pos_action_mm=pos_action_mm_arr,
                rot_action_deg=rot_action_deg_arr,
                pos_state_mm=pos_state_mm_arr,
                rot_state_deg=rot_state_deg_arr,
                joint_step_l2_deg=joint_step_l2_deg_arr,
                joint_step_max_deg=joint_step_max_deg_arr,
                bad_action_frames=int(np.sum((pos_action_mm_arr > bad_pos_mm) | (rot_action_deg_arr > bad_rot_deg))),
                bad_state_frames=int(np.sum((pos_state_mm_arr > bad_pos_mm) | (rot_state_deg_arr > bad_rot_deg))),
                bad_joint_step_frames=int(np.sum(joint_step_max_deg_arr > bad_joint_step_max_deg)),
                peak_indices=peak_indices,
            )
        )

    return results


def write_summary_csv(
    results: list[StreamResult],
    output_path: Path,
    *,
    warmup_frames: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "stream",
        "pos_action_mean_mm",
        "pos_action_max_mm",
        "pos_action_p95_mm",
        "rot_action_mean_deg",
        "rot_action_max_deg",
        "rot_action_p95_deg",
        "pos_state_mean_mm",
        "pos_state_max_mm",
        "pos_state_p95_mm",
        "rot_state_mean_deg",
        "rot_state_max_deg",
        "rot_state_p95_deg",
        "joint_step_l2_mean_deg",
        "joint_step_l2_max_deg",
        "joint_step_l2_p95_deg",
        "joint_step_max_mean_deg",
        "joint_step_max_deg",
        "joint_step_max_p95_deg",
        "bad_action_frames",
        "bad_state_frames",
        "bad_joint_step_frames",
        "stable_pos_action_p95_mm",
        "stable_pos_state_p95_mm",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for result in results:
            pa_mean, pa_max, pa_p95 = summarize(result.pos_action_mm)
            ra_mean, ra_max, ra_p95 = summarize(result.rot_action_deg)
            ps_mean, ps_max, ps_p95 = summarize(result.pos_state_mm)
            rs_mean, rs_max, rs_p95 = summarize(result.rot_state_deg)
            jl2_mean, jl2_max, jl2_p95 = summarize(result.joint_step_l2_deg)
            jmax_mean, jmax_max, jmax_p95 = summarize(result.joint_step_max_deg)
            stable_slice = slice(min(warmup_frames, len(result.pos_action_mm)), None)
            stable_pos_action = result.pos_action_mm[stable_slice]
            stable_pos_state = result.pos_state_mm[stable_slice]
            writer.writerow(
                {
                    "stream": result.stream,
                    "pos_action_mean_mm": f"{pa_mean:.6f}",
                    "pos_action_max_mm": f"{pa_max:.6f}",
                    "pos_action_p95_mm": f"{pa_p95:.6f}",
                    "rot_action_mean_deg": f"{ra_mean:.6f}",
                    "rot_action_max_deg": f"{ra_max:.6f}",
                    "rot_action_p95_deg": f"{ra_p95:.6f}",
                    "pos_state_mean_mm": f"{ps_mean:.6f}",
                    "pos_state_max_mm": f"{ps_max:.6f}",
                    "pos_state_p95_mm": f"{ps_p95:.6f}",
                    "rot_state_mean_deg": f"{rs_mean:.6f}",
                    "rot_state_max_deg": f"{rs_max:.6f}",
                    "rot_state_p95_deg": f"{rs_p95:.6f}",
                    "joint_step_l2_mean_deg": f"{jl2_mean:.6f}",
                    "joint_step_l2_max_deg": f"{jl2_max:.6f}",
                    "joint_step_l2_p95_deg": f"{jl2_p95:.6f}",
                    "joint_step_max_mean_deg": f"{jmax_mean:.6f}",
                    "joint_step_max_deg": f"{jmax_max:.6f}",
                    "joint_step_max_p95_deg": f"{jmax_p95:.6f}",
                    "bad_action_frames": result.bad_action_frames,
                    "bad_state_frames": result.bad_state_frames,
                    "bad_joint_step_frames": result.bad_joint_step_frames,
                    "stable_pos_action_p95_mm": (
                        f"{float(np.percentile(stable_pos_action, 95)):.6f}" if len(stable_pos_action) else ""
                    ),
                    "stable_pos_state_p95_mm": (
                        f"{float(np.percentile(stable_pos_state, 95)):.6f}" if len(stable_pos_state) else ""
                    ),
                }
            )


def write_details_csv(
    results: list[StreamResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "stream",
        "frame",
        "pos_action_mm",
        "rot_action_deg",
        "pos_state_mm",
        "rot_state_deg",
        "joint_step_l2_deg",
        "joint_step_max_deg",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for result in results:
            for fi in range(len(result.pos_action_mm)):
                writer.writerow(
                    {
                        "stream": result.stream,
                        "frame": fi,
                        "pos_action_mm": f"{result.pos_action_mm[fi]:.6f}",
                        "rot_action_deg": f"{result.rot_action_deg[fi]:.6f}",
                        "pos_state_mm": f"{result.pos_state_mm[fi]:.6f}",
                        "rot_state_deg": f"{result.rot_state_deg[fi]:.6f}",
                        "joint_step_l2_deg": f"{result.joint_step_l2_deg[fi]:.6f}",
                        "joint_step_max_deg": f"{result.joint_step_max_deg[fi]:.6f}",
                    }
                )


def write_markdown(
    results: list[StreamResult],
    output_path: Path,
    *,
    args: argparse.Namespace,
    csv_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[str]] = []
    for result in results:
        pa_mean, _, pa_p95 = summarize(result.pos_action_mm)
        ps_mean, _, ps_p95 = summarize(result.pos_state_mm)
        jl2_mean, _, jl2_p95 = summarize(result.joint_step_l2_deg)
        jmax_mean, jmax_max, jmax_p95 = summarize(result.joint_step_max_deg)
        summary_rows.append(
            [
                result.stream,
                f"{pa_mean:.2f} / {pa_p95:.2f}",
                f"{ps_mean:.2f} / {ps_p95:.2f}",
                f"{jl2_mean:.2f} / {jl2_p95:.2f}",
                f"{jmax_mean:.2f} / {jmax_p95:.2f} / {jmax_max:.2f}",
                str(result.bad_action_frames),
                str(result.bad_state_frames),
                str(result.bad_joint_step_frames),
            ]
        )

    peak_sections: list[str] = []
    for result in results:
        rows: list[list[str]] = []
        for idx in result.peak_indices[: args.peak_rows]:
            rows.append(
                [
                    str(int(idx)),
                    f"{result.pos_action_mm[idx]:.2f}",
                    f"{result.rot_action_deg[idx]:.2f}",
                    f"{result.pos_state_mm[idx]:.2f}",
                    f"{result.rot_state_deg[idx]:.2f}",
                    f"{result.joint_step_max_deg[idx]:.2f}",
                ]
            )
        peak_sections.extend(
            [
                f"### {result.stream}",
                "",
                markdown_table(
                    ["frame", "pos_vs_action(mm)", "rot_vs_action(deg)", "pos_vs_state(mm)", "rot_vs_state(deg)", "joint_step_max(deg)"],
                    rows,
                ),
                "",
            ]
        )

    start_pose = _RESET_POSE_B_XYZQUAT if args.start_pose_b_xyzquat is None else np.asarray(args.start_pose_b_xyzquat)
    content = [
        "# FR3 Sim Replay Joint-Target Validation",
        "",
        f"- dataset: `{resolve_dataset_path(args.dataset)}`",
        f"- joint_targets_csv: `{csv_path}`",
        f"- episode: `{args.episode}`",
        f"- streams: `{args.streams}`",
        f"- start_pose_b_xyzquat: `{np.round(start_pose, 6).tolist()}`",
        f"- bad thresholds: `pos>{args.bad_pos_mm}mm`, `rot>{args.bad_rot_deg}deg`, `joint_step_max>{args.bad_joint_step_max_deg}deg`",
        "",
        "## Summary",
        "",
        markdown_table(
            [
                "stream",
                "pos_vs_action mean/p95",
                "pos_vs_state mean/p95",
                "joint_step_l2 mean/p95",
                "joint_step_max mean/p95/max",
                "bad_action",
                "bad_state",
                "bad_joint_step",
            ],
            summary_rows,
        ),
        "",
        "## Peak Frames",
        "",
        *peak_sections,
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_dataset_path(args.dataset)
    csv_path = resolve_joint_targets_csv(args)
    ep = load_episode(dataset_path, args.episode)
    states = ep["state"]
    actions = ep["action"]
    n_frames = len(states)

    start_pose = _RESET_POSE_B_XYZQUAT if args.start_pose_b_xyzquat is None else np.asarray(args.start_pose_b_xyzquat, dtype=np.float64)
    T_B_Ws = pose_from_xyzquat(start_pose) @ se3_inv(_T_IE) @ se3_inv(pose_from_xyzquat(states[0]))
    state_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in states], dtype=np.float64)
    action_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in actions], dtype=np.float64)

    joint_streams_rad = load_joint_sequences(csv_path, args.streams, n_frames)
    results = evaluate_streams(
        state_poses=state_poses,
        action_poses=action_poses,
        joint_streams_rad=joint_streams_rad,
        peak_rows=args.peak_rows,
        bad_pos_mm=args.bad_pos_mm,
        bad_rot_deg=args.bad_rot_deg,
        bad_joint_step_max_deg=args.bad_joint_step_max_deg,
    )

    stem = f"fr3_sim_replay_joint_validation_ep{args.episode:03d}"
    summary_csv = args.output_dir.resolve() / f"{stem}_summary.csv"
    details_csv = args.output_dir.resolve() / f"{stem}_details.csv"
    md_path = args.output_dir.resolve() / f"{stem}.md"
    write_summary_csv(results, summary_csv, warmup_frames=args.warmup_frames)
    write_details_csv(results, details_csv)
    write_markdown(results, md_path, args=args, csv_path=csv_path)

    print(f"[INFO] 已导出 Summary CSV: {summary_csv}")
    print(f"[INFO] 已导出 Details CSV: {details_csv}")
    print(f"[INFO] 已导出 Markdown: {md_path}")
    print("[INFO] Summary:")
    for result in results:
        pa_mean, _, pa_p95 = summarize(result.pos_action_mm)
        ps_mean, _, ps_p95 = summarize(result.pos_state_mm)
        jmax_mean, jmax_max, jmax_p95 = summarize(result.joint_step_max_deg)
        print(
            f"  {result.stream}: "
            f"pos_vs_action mean/p95={pa_mean:.2f}/{pa_p95:.2f}mm  "
            f"pos_vs_state mean/p95={ps_mean:.2f}/{ps_p95:.2f}mm  "
            f"joint_step_max mean/p95/max={jmax_mean:.2f}/{jmax_p95:.2f}/{jmax_max:.2f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
