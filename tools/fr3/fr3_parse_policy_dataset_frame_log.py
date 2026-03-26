#!/usr/bin/env python3
"""Parse FR3 policy-vs-dataset frame comparison logs into CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


CHECK_RE = re.compile(
    r"^\[CHECK\]\s+episode=\s*(?P<episode>\d+)\s+"
    r"frame=\s*(?P<frame>\d+)\s+"
    r"pos_err_mm=(?P<pos_err_mm>[-+0-9.eE]+)\s+"
    r"rot_err_deg=(?P<rot_err_deg>[-+0-9.eE]+)\s+"
    r"grip_err_mm=(?P<grip_err_mm>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class PhaseLabel:
    episode: int
    frame_start: int
    frame_end: int
    phase: str


@dataclass(frozen=True)
class ErrorRow:
    run_id: str
    checkpoint: str
    episode: int
    frame: int
    phase: str
    is_first_frame: bool
    pos_err_mm: float
    rot_err_deg: float
    grip_err_mm: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse FR3 policy-vs-dataset frame logs into CSV artifacts.")
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        required=True,
        help="Path to a raw log file from tools/fr3/fr3_check_policy_dataset_frame.py. Repeat for multiple logs.",
    )
    parser.add_argument("--phase-labels", type=Path, default=None, help="Optional CSV with episode/frame ranges to phase labels.")
    parser.add_argument("--run-id", required=True, help="Run identifier to stamp into the CSV output.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path string to stamp into the CSV output.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output path for per_frame_action_error.csv.")
    parser.add_argument("--topk-out", type=Path, default=None, help="Optional output path for topk_worst_frames.csv.")
    parser.add_argument("--topk", type=int, default=10, help="Number of rows to write when --topk-out is provided.")
    return parser.parse_args(argv)


def load_phase_labels(path: Path | None) -> dict[int, list[PhaseLabel]]:
    if path is None:
        return {}

    labels: dict[int, list[PhaseLabel]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode", "frame_start", "frame_end", "phase"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Phase label CSV missing required columns {sorted(required)}: {path}")
        for raw_row in reader:
            label = PhaseLabel(
                episode=int(raw_row["episode"]),
                frame_start=int(raw_row["frame_start"]),
                frame_end=int(raw_row["frame_end"]),
                phase=str(raw_row["phase"]).strip(),
            )
            labels.setdefault(label.episode, []).append(label)

    for episode_labels in labels.values():
        episode_labels.sort(key=lambda item: (item.frame_start, item.frame_end))
    return labels


def resolve_phase(episode: int, frame: int, labels: dict[int, list[PhaseLabel]]) -> str:
    for label in labels.get(episode, []):
        if label.frame_start <= frame <= label.frame_end:
            return label.phase
    return "unknown"


def parse_log_rows(log_path: Path, *, run_id: str, checkpoint: str, labels: dict[int, list[PhaseLabel]]) -> list[ErrorRow]:
    rows: list[ErrorRow] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = CHECK_RE.match(line.strip())
            if match is None:
                continue
            episode = int(match.group("episode"))
            frame = int(match.group("frame"))
            rows.append(
                ErrorRow(
                    run_id=run_id,
                    checkpoint=checkpoint,
                    episode=episode,
                    frame=frame,
                    phase=resolve_phase(episode, frame, labels),
                    is_first_frame=(frame == 0),
                    pos_err_mm=float(match.group("pos_err_mm")),
                    rot_err_deg=float(match.group("rot_err_deg")),
                    grip_err_mm=float(match.group("grip_err_mm")),
                )
            )
    return rows


def collect_log_rows(
    log_paths: list[Path],
    *,
    run_id: str,
    checkpoint: str,
    labels: dict[int, list[PhaseLabel]],
) -> list[ErrorRow]:
    rows: list[ErrorRow] = []
    for log_path in log_paths:
        rows.extend(parse_log_rows(log_path, run_id=run_id, checkpoint=checkpoint, labels=labels))
    rows.sort(key=lambda row: (row.episode, row.frame, row.phase))
    return rows


def write_per_frame_csv(path: Path, rows: list[ErrorRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "checkpoint",
                "episode",
                "frame",
                "phase",
                "is_first_frame",
                "pos_err_mm",
                "rot_err_deg",
                "grip_err_mm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "checkpoint": row.checkpoint,
                    "episode": row.episode,
                    "frame": row.frame,
                    "phase": row.phase,
                    "is_first_frame": "true" if row.is_first_frame else "false",
                    "pos_err_mm": f"{row.pos_err_mm:.6f}",
                    "rot_err_deg": f"{row.rot_err_deg:.6f}",
                    "grip_err_mm": f"{row.grip_err_mm:.6f}",
                }
            )


def _normalized_metric_scores(rows: list[ErrorRow]) -> tuple[float, float, float]:
    max_pos = max((row.pos_err_mm for row in rows), default=0.0)
    max_rot = max((row.rot_err_deg for row in rows), default=0.0)
    max_grip = max((row.grip_err_mm for row in rows), default=0.0)
    return max_pos, max_rot, max_grip


def _topk_entry(row: ErrorRow, *, max_pos: float, max_rot: float, max_grip: float) -> dict[str, str]:
    scores = {
        "pos": row.pos_err_mm / max_pos if max_pos > 0.0 else 0.0,
        "rot": row.rot_err_deg / max_rot if max_rot > 0.0 else 0.0,
        "grip": row.grip_err_mm / max_grip if max_grip > 0.0 else 0.0,
    }
    aggregate_score = max(scores.values(), default=0.0)
    dominant_metrics = [name for name, score in scores.items() if abs(score - aggregate_score) <= 1e-12]
    dominant_metric = dominant_metrics[0] if len(dominant_metrics) == 1 else "mixed"
    return {
        "run_id": row.run_id,
        "episode": str(row.episode),
        "frame": str(row.frame),
        "phase": row.phase,
        "pos_err_mm": f"{row.pos_err_mm:.6f}",
        "rot_err_deg": f"{row.rot_err_deg:.6f}",
        "grip_err_mm": f"{row.grip_err_mm:.6f}",
        "dominant_metric": dominant_metric,
        "_aggregate_score": f"{aggregate_score:.12f}",
    }


def write_topk_csv(path: Path, rows: list[ErrorRow], *, topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_pos, max_rot, max_grip = _normalized_metric_scores(rows)
    ranked = [
        _topk_entry(row, max_pos=max_pos, max_rot=max_rot, max_grip=max_grip)
        for row in rows
    ]
    ranked.sort(
        key=lambda item: (
            float(item["_aggregate_score"]),
            float(item["pos_err_mm"]),
            float(item["rot_err_deg"]),
            float(item["grip_err_mm"]),
        ),
        reverse=True,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "episode",
                "frame",
                "phase",
                "pos_err_mm",
                "rot_err_deg",
                "grip_err_mm",
                "dominant_metric",
            ],
        )
        writer.writeheader()
        for row in ranked[: max(topk, 0)]:
            row = dict(row)
            row.pop("_aggregate_score", None)
            writer.writerow(row)


def build_csv_artifacts(
    *,
    log_paths: list[Path],
    phase_labels_path: Path | None,
    run_id: str,
    checkpoint: str,
    output_csv: Path,
    topk_out: Path | None = None,
    topk: int = 10,
) -> list[ErrorRow]:
    labels = load_phase_labels(phase_labels_path)
    rows = collect_log_rows(log_paths, run_id=run_id, checkpoint=checkpoint, labels=labels)
    if not rows:
        raise ValueError("No [CHECK] rows parsed from the provided log files.")

    write_per_frame_csv(output_csv, rows)
    if topk_out is not None:
        write_topk_csv(topk_out, rows, topk=topk)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_csv_artifacts(
        log_paths=[Path(log_arg) for log_arg in args.logs],
        phase_labels_path=args.phase_labels,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        output_csv=args.output_csv,
        topk_out=args.topk_out,
        topk=args.topk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
