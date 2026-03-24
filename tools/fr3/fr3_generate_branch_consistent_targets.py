#!/usr/bin/env python3
"""
离线生成 branch-consistent 的 FR3 replay joint targets。

目标：
- 保留 action EE 目标意图
- 尽量贴近 state 参考分支
- 避免小 EE 差异映射成大 joint-branch 跳变
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fr3_das_replay_runtime import (
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


DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive float.") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_xyzquat(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("Expected 7 comma-separated floats: x,y,z,qx,qy,qz,qw")
    try:
        return np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected 7 comma-separated floats: x,y,z,qx,qy,qz,qw") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate branch-consistent FR3 replay joint targets")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--start-pose-b-xyzquat",
        type=parse_xyzquat,
        default=None,
        help="Measured FR3 EE start pose in base frame: x,y,z,qx,qy,qz,qw. "
        "If omitted, fall back to the static reset pose.",
    )
    parser.add_argument("--w-pose-mm", type=positive_float, default=1.0)
    parser.add_argument("--w-pose-deg", type=positive_float, default=5.0)
    parser.add_argument("--w-prev-l2", type=positive_float, default=2.0)
    parser.add_argument("--w-ref-l2", type=positive_float, default=1.5)
    parser.add_argument("--hard-prev-max-deg", type=positive_float, default=45.0)
    parser.add_argument("--hard-ref-max-deg", type=positive_float, default=60.0)
    parser.add_argument("--w-prev-max-excess", type=positive_float, default=20.0)
    parser.add_argument("--w-ref-max-excess", type=positive_float, default=20.0)
    parser.add_argument("--pose-score-margin", type=positive_float, default=100.0)
    parser.add_argument("--divergence-wrap-max-deg", type=positive_float, default=120.0)
    parser.add_argument("--divergence-wrap-l2-deg", type=positive_float, default=180.0)
    return parser.parse_args(argv)


def joint_values_to_deg(values_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.asarray(values_rad, dtype=np.float64))


def wrap_angle_deg(values_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(values_deg, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


@dataclass
class Candidate:
    mode: str
    q_deg: np.ndarray
    pose_pos_err_mm: float
    pose_rot_err_deg: float
    pose_score: float
    prev_l2_deg: float
    prev_max_deg: float
    ref_l2_deg: float
    ref_max_deg: float
    branch_score: float
    cost: float
    feasible: bool
def build_pose_and_joint_sequences(
    dataset_path: str,
    episode_idx: int,
    *,
    start_pose_b_xyzquat: np.ndarray | None = None,
):
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

    T_B_E_start = pose_from_xyzquat(
        _RESET_POSE_B_XYZQUAT if start_pose_b_xyzquat is None else np.asarray(start_pose_b_xyzquat, dtype=np.float64)
    )
    T_B_Ws = T_B_E_start @ se3_inv(_T_IE) @ se3_inv(pose_from_xyzquat(states[0]))
    state_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in states], dtype=np.float64)
    action_poses = np.asarray([ws_to_base(T_B_Ws, pose_from_xyzquat(frame)) for frame in actions], dtype=np.float64)

    state_joint_targets_rad: list[np.ndarray] = []
    action_joint_naive_rad: list[np.ndarray] = []
    prev_state = _IK_SEED_JOINTS_RAD.copy()
    prev_action = _IK_SEED_JOINTS_RAD.copy()

    for fi in range(len(states)):
        q_state = np.asarray(kin.inverse_kinematics(prev_state, state_poses[fi]), dtype=np.float64)
        q_action = np.asarray(kin.inverse_kinematics(prev_action, action_poses[fi]), dtype=np.float64)
        state_joint_targets_rad.append(q_state.copy())
        action_joint_naive_rad.append(q_action.copy())
        prev_state = q_state.copy()
        prev_action = q_action.copy()

    return (
        timestamps,
        state_poses,
        action_poses,
        np.asarray(state_joint_targets_rad, dtype=np.float64),
        np.asarray(action_joint_naive_rad, dtype=np.float64),
        kin,
    )


def evaluate_candidate(
    *,
    mode: str,
    q_deg: np.ndarray,
    action_pose: np.ndarray,
    ref_deg: np.ndarray,
    prev_deg: np.ndarray,
    kin,
    args: argparse.Namespace,
) -> Candidate:
    fk_pose = np.asarray(kin.forward_kinematics(np.deg2rad(q_deg)), dtype=np.float64)
    pose_pos_err_mm = float(np.linalg.norm(fk_pose[:3, 3] - action_pose[:3, 3]) * 1000.0)
    pose_rot_err_deg = rotation_angle_error_deg(fk_pose[:3, :3], action_pose[:3, :3])
    prev_delta = wrap_angle_deg(q_deg - prev_deg)
    ref_delta = wrap_angle_deg(q_deg - ref_deg)
    prev_l2_deg = float(np.linalg.norm(prev_delta))
    prev_max_deg = float(np.max(np.abs(prev_delta)))
    ref_l2_deg = float(np.linalg.norm(ref_delta))
    ref_max_deg = float(np.max(np.abs(ref_delta)))
    feasible = prev_max_deg <= args.hard_prev_max_deg and ref_max_deg <= args.hard_ref_max_deg
    prev_max_excess_deg = max(0.0, prev_max_deg - args.hard_prev_max_deg)
    ref_max_excess_deg = max(0.0, ref_max_deg - args.hard_ref_max_deg)
    pose_score = args.w_pose_mm * pose_pos_err_mm + args.w_pose_deg * pose_rot_err_deg
    branch_score = (
        args.w_prev_l2 * prev_l2_deg
        + args.w_ref_l2 * ref_l2_deg
        + args.w_prev_max_excess * prev_max_excess_deg
        + args.w_ref_max_excess * ref_max_excess_deg
    )
    cost = pose_score + branch_score
    return Candidate(
        mode=mode,
        q_deg=q_deg,
        pose_pos_err_mm=pose_pos_err_mm,
        pose_rot_err_deg=pose_rot_err_deg,
        pose_score=pose_score,
        prev_l2_deg=prev_l2_deg,
        prev_max_deg=prev_max_deg,
        ref_l2_deg=ref_l2_deg,
        ref_max_deg=ref_max_deg,
        branch_score=branch_score,
        cost=cost,
        feasible=feasible,
    )


def count_divergent_frames(
    reference_deg: np.ndarray,
    candidate_deg: np.ndarray,
    wrap_max_threshold_deg: float,
    wrap_l2_threshold_deg: float,
) -> int:
    count = 0
    for fi in range(len(reference_deg)):
        delta = wrap_angle_deg(candidate_deg[fi] - reference_deg[fi])
        if np.max(np.abs(delta)) >= wrap_max_threshold_deg or np.linalg.norm(delta) >= wrap_l2_threshold_deg:
            count += 1
    return count


def generate_branch_consistent_targets(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_path = str((Path.cwd() / args.dataset).resolve()) if not Path(args.dataset).is_absolute() else args.dataset
    timestamps, state_poses, action_poses, state_joint_rad, action_naive_rad, kin = build_pose_and_joint_sequences(
        dataset_path,
        args.episode,
        start_pose_b_xyzquat=args.start_pose_b_xyzquat,
    )

    state_joint_deg = joint_values_to_deg(state_joint_rad)
    action_naive_deg = joint_values_to_deg(action_naive_rad)
    branch_consistent_deg: list[np.ndarray] = []
    chosen_modes: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    prev_bc_deg = joint_values_to_deg(_IK_SEED_JOINTS_RAD)

    for fi in range(len(timestamps)):
        ref_deg = state_joint_deg[fi]
        prev_ref_deg = state_joint_deg[fi - 1] if fi > 0 else ref_deg
        action_pose = action_poses[fi]

        candidates: list[Candidate] = []
        seen_keys: set[tuple[float, ...]] = set()

        def add_candidate(mode: str, q_deg: np.ndarray) -> None:
            key = tuple(np.round(q_deg, 6).tolist())
            if key in seen_keys:
                return
            seen_keys.add(key)
            candidates.append(
                evaluate_candidate(
                    mode=mode,
                    q_deg=q_deg,
                    action_pose=action_pose,
                    ref_deg=ref_deg,
                    prev_deg=prev_bc_deg,
                    kin=kin,
                    args=args,
                )
            )

        direct_candidate_specs = [
            ("naive_direct", action_naive_deg[fi]),
            ("state_ref_direct", ref_deg.copy()),
        ]
        for mode, q_deg in direct_candidate_specs:
            add_candidate(mode, np.asarray(q_deg, dtype=np.float64).copy())

        seed_candidate_specs = [
            ("ik_prev_bc", prev_bc_deg),
            ("ik_state_ref", ref_deg),
            ("ik_prev_state_ref", prev_ref_deg),
        ]
        for mode, seed_deg in seed_candidate_specs:
            q_rad = np.asarray(kin.inverse_kinematics(np.deg2rad(seed_deg), action_pose), dtype=np.float64)
            add_candidate(mode, joint_values_to_deg(q_rad))

        best_pose_score = min(candidate.pose_score for candidate in candidates)
        pose_eligible = [
            candidate for candidate in candidates if candidate.pose_score <= best_pose_score + args.pose_score_margin
        ]
        feasible_pose_eligible = [candidate for candidate in pose_eligible if candidate.feasible]
        selection_pool = feasible_pose_eligible or pose_eligible
        chosen = min(selection_pool, key=lambda candidate: (candidate.branch_score, candidate.pose_score, candidate.cost))
        chosen_modes[chosen.mode] += 1
        branch_consistent_deg.append(chosen.q_deg.copy())
        prev_bc_deg = chosen.q_deg.copy()

        naive_delta = wrap_angle_deg(action_naive_deg[fi] - ref_deg)
        bc_delta = wrap_angle_deg(chosen.q_deg - ref_deg)
        rows.append(
            {
                "frame": fi,
                "timestamp_s": float(timestamps[fi]),
                "mode": chosen.mode,
                "chosen_feasible": chosen.feasible,
                "chosen_cost": chosen.cost,
                "chosen_pose_score": chosen.pose_score,
                "chosen_branch_score": chosen.branch_score,
                "best_pose_score": best_pose_score,
                "chosen_pose_score_gap": chosen.pose_score - best_pose_score,
                "chosen_pose_err_mm": chosen.pose_pos_err_mm,
                "chosen_pose_err_deg": chosen.pose_rot_err_deg,
                "chosen_prev_l2_deg": chosen.prev_l2_deg,
                "chosen_prev_max_deg": chosen.prev_max_deg,
                "chosen_ref_l2_deg": chosen.ref_l2_deg,
                "chosen_ref_max_deg": chosen.ref_max_deg,
                "naive_ref_l2_deg": float(np.linalg.norm(naive_delta)),
                "naive_ref_max_deg": float(np.max(np.abs(naive_delta))),
                "bc_ref_l2_deg": float(np.linalg.norm(bc_delta)),
                "bc_ref_max_deg": float(np.max(np.abs(bc_delta))),
                "naive_joint_deg": action_naive_deg[fi],
                "bc_joint_deg": chosen.q_deg.copy(),
                "state_ref_joint_deg": ref_deg.copy(),
            }
        )

    bc_joint_deg = np.asarray(branch_consistent_deg, dtype=np.float64)
    summary = {
        "dataset_path": dataset_path,
        "episode": args.episode,
        "n_frames": len(rows),
        "start_pose_b_xyzquat": (
            _RESET_POSE_B_XYZQUAT.copy() if args.start_pose_b_xyzquat is None else np.asarray(args.start_pose_b_xyzquat, dtype=np.float64)
        ),
        "chosen_modes": chosen_modes,
        "naive_divergent_frames": count_divergent_frames(
            state_joint_deg,
            action_naive_deg,
            args.divergence_wrap_max_deg,
            args.divergence_wrap_l2_deg,
        ),
        "bc_divergent_frames": count_divergent_frames(
            state_joint_deg,
            bc_joint_deg,
            args.divergence_wrap_max_deg,
            args.divergence_wrap_l2_deg,
        ),
        "naive_ref_l2_mean_deg": float(np.mean([row["naive_ref_l2_deg"] for row in rows])),
        "naive_ref_l2_p95_deg": float(np.percentile([row["naive_ref_l2_deg"] for row in rows], 95)),
        "bc_ref_l2_mean_deg": float(np.mean([row["bc_ref_l2_deg"] for row in rows])),
        "bc_ref_l2_p95_deg": float(np.percentile([row["bc_ref_l2_deg"] for row in rows], 95)),
        "chosen_pose_err_mm_mean": float(np.mean([row["chosen_pose_err_mm"] for row in rows])),
        "chosen_pose_err_mm_p95": float(np.percentile([row["chosen_pose_err_mm"] for row in rows], 95)),
        "chosen_pose_err_deg_mean": float(np.mean([row["chosen_pose_err_deg"] for row in rows])),
        "chosen_pose_err_deg_p95": float(np.percentile([row["chosen_pose_err_deg"] for row in rows], 95)),
        "chosen_pose_score_gap_mean": float(np.mean([row["chosen_pose_score_gap"] for row in rows])),
        "chosen_pose_score_gap_p95": float(np.percentile([row["chosen_pose_score_gap"] for row in rows], 95)),
    }
    return rows, summary


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "frame",
        "timestamp_s",
        "mode",
        "chosen_feasible",
        "chosen_cost",
        "chosen_pose_score",
        "chosen_branch_score",
        "best_pose_score",
        "chosen_pose_score_gap",
        "chosen_pose_err_mm",
        "chosen_pose_err_deg",
        "chosen_prev_l2_deg",
        "chosen_prev_max_deg",
        "chosen_ref_l2_deg",
        "chosen_ref_max_deg",
        "naive_ref_l2_deg",
        "naive_ref_max_deg",
        "bc_ref_l2_deg",
        "bc_ref_max_deg",
    ]
    for prefix in ("state_ref_joint", "naive_joint", "bc_joint"):
        for joint_idx in range(1, 8):
            header.append(f"{prefix}_{joint_idx}_deg")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in header if key in row}
            for prefix, values_key in (
                ("state_ref_joint", "state_ref_joint_deg"),
                ("naive_joint", "naive_joint_deg"),
                ("bc_joint", "bc_joint_deg"),
            ):
                for joint_idx, value in enumerate(row[values_key], start=1):
                    flat[f"{prefix}_{joint_idx}_deg"] = f"{float(value):.6f}"
            writer.writerow(flat)


def write_markdown(rows: list[dict[str, object]], summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode_rows = [[mode, str(count)] for mode, count in summary["chosen_modes"].most_common()]
    sample_frames = [160, 162, 163, 240, 264, 265]
    sample_rows = []
    by_frame = {int(row["frame"]): row for row in rows}
    for frame in sample_frames:
        row = by_frame.get(frame)
        if row is None:
            continue
        sample_rows.append(
            [
                str(frame),
                str(row["mode"]),
                f"{float(row['naive_ref_max_deg']):.2f}",
                f"{float(row['bc_ref_max_deg']):.2f}",
                f"{float(row['chosen_pose_err_mm']):.2f}",
                f"{float(row['chosen_pose_err_deg']):.2f}",
                f"{float(row['chosen_pose_score_gap']):.2f}",
            ]
        )

    content = [
        "# FR3 Branch-Consistent Target Generation",
        "",
        f"- dataset: `{summary['dataset_path']}`",
        f"- episode: `{summary['episode']}`",
        f"- frames: `{summary['n_frames']}`",
        f"- start_pose_b_xyzquat: `{np.round(summary['start_pose_b_xyzquat'], 6).tolist()}`",
        "",
        "## Summary",
        "",
        f"- naive divergent frames: `{summary['naive_divergent_frames']}`",
        f"- branch-consistent divergent frames: `{summary['bc_divergent_frames']}`",
        f"- naive ref L2 mean / p95: `{summary['naive_ref_l2_mean_deg']:.2f} / {summary['naive_ref_l2_p95_deg']:.2f} deg`",
        f"- branch-consistent ref L2 mean / p95: `{summary['bc_ref_l2_mean_deg']:.2f} / {summary['bc_ref_l2_p95_deg']:.2f} deg`",
        f"- chosen action pose err mean / p95: `{summary['chosen_pose_err_mm_mean']:.2f} / {summary['chosen_pose_err_mm_p95']:.2f} mm`",
        f"- chosen action rot err mean / p95: `{summary['chosen_pose_err_deg_mean']:.2f} / {summary['chosen_pose_err_deg_p95']:.2f} deg`",
        f"- chosen pose-score gap to best mean / p95: `{summary['chosen_pose_score_gap_mean']:.2f} / {summary['chosen_pose_score_gap_p95']:.2f}`",
        "",
        "## Chosen Modes",
        "",
        markdown_table(["mode", "count"], mode_rows),
        "",
        "## Sample Frames",
        "",
        markdown_table(
            [
                "frame",
                "mode",
                "naive_ref_max(deg)",
                "bc_ref_max(deg)",
                "bc_pose_err(mm)",
                "bc_pose_err(deg)",
                "pose_gap_to_best",
            ],
            sample_rows,
        ),
        "",
        "## Notes",
        "",
        "- `state_ref` is the sequential IK solution of `state[t]` and defines the reference branch.",
        "- `naive_joint` is the sequential IK solution of `action[t]` seeded only by the previous action solution.",
        "- `bc_joint` is selected in two stages: keep only candidates within `pose_score_margin` of the best pose score, then choose the most branch-consistent one.",
        "- `state_ref_direct` remains a fallback candidate, but it can only win when its pose score stays close to the best available candidate.",
    ]
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary = generate_branch_consistent_targets(args)
    stem = f"fr3_branch_consistent_targets_ep{args.episode:03d}"
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    write_csv(rows, csv_path)
    write_markdown(rows, summary, md_path)

    print(f"[INFO] 已导出 CSV: {csv_path}")
    print(f"[INFO] 已导出 Markdown: {md_path}")
    print("[INFO] Summary:")
    print(f"  naive divergent frames: {summary['naive_divergent_frames']}")
    print(f"  branch-consistent divergent frames: {summary['bc_divergent_frames']}")
    print(f"  naive ref L2 mean/p95: {summary['naive_ref_l2_mean_deg']:.2f}/{summary['naive_ref_l2_p95_deg']:.2f} deg")
    print(f"  bc ref L2 mean/p95: {summary['bc_ref_l2_mean_deg']:.2f}/{summary['bc_ref_l2_p95_deg']:.2f} deg")
    print(f"  chosen pose err mean/p95: {summary['chosen_pose_err_mm_mean']:.2f}/{summary['chosen_pose_err_mm_p95']:.2f} mm")
    print(
        f"  chosen pose-score gap mean/p95: "
        f"{summary['chosen_pose_score_gap_mean']:.2f}/{summary['chosen_pose_score_gap_p95']:.2f}"
    )
    print("[INFO] Chosen modes:")
    for mode, count in summary["chosen_modes"].most_common():
        print(f"  {mode}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
