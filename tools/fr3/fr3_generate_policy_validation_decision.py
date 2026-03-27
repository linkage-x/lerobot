#!/usr/bin/env python3
"""Generate a decision.md skeleton from FR3 per-frame action error CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tools.fr3.fr3_parse_policy_dataset_frame_log import ErrorRow, PhaseLabel, load_phase_labels
except ImportError:  # pragma: no cover - fallback for direct script execution
    from fr3_parse_policy_dataset_frame_log import ErrorRow, PhaseLabel, load_phase_labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FR3 decision.md scaffolding from per-frame action error CSV.")
    parser.add_argument("--per-frame-csv", type=Path, required=True, help="Path to per_frame_action_error.csv.")
    parser.add_argument("--phase-labels", type=Path, default=None, help="Optional phase_labels.csv used for coverage summaries.")
    parser.add_argument("--output-md", type=Path, required=True, help="Output path for decision.md.")
    parser.add_argument("--dataset", default="TODO", help="Dataset identifier or path for the decision header.")
    parser.add_argument("--decode-gate", default="TODO", help="Human-entered Gate 0 status string.")
    parser.add_argument("--dataset-fed-result", default="TODO", help="Human-entered Gate 1 status string.")
    parser.add_argument("--dominant-blocker", default="TODO", help="Human-entered current blocker hypothesis.")
    parser.add_argument("--next-action", default="TODO", help="Human-entered next action.")
    return parser.parse_args(argv)


def read_per_frame_rows(path: Path) -> list[ErrorRow]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "run_id",
            "checkpoint",
            "episode",
            "frame",
            "phase",
            "is_first_frame",
            "pos_err_mm",
            "rot_err_deg",
            "grip_err_mm",
        }
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"per-frame CSV missing required columns {sorted(required)}: {path}")

        rows: list[ErrorRow] = []
        for raw_row in reader:
            rows.append(
                ErrorRow(
                    run_id=str(raw_row["run_id"]).strip(),
                    checkpoint=str(raw_row["checkpoint"]).strip(),
                    episode=int(raw_row["episode"]),
                    frame=int(raw_row["frame"]),
                    phase=str(raw_row["phase"]).strip() or "unknown",
                    is_first_frame=str(raw_row["is_first_frame"]).strip().lower() == "true",
                    pos_err_mm=float(raw_row["pos_err_mm"]),
                    rot_err_deg=float(raw_row["rot_err_deg"]),
                    grip_err_mm=float(raw_row["grip_err_mm"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in per-frame CSV: {path}")
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unique_or_join(values: list[str]) -> str:
    deduped = [value for value in dict.fromkeys(values) if value]
    if not deduped:
        return ""
    if len(deduped) == 1:
        return deduped[0]
    return ", ".join(deduped)


def _normalize_metric(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= 0.0:
        return [0.0 for _ in values]
    return [value / max_value for value in values]


def _row_severity(rows: list[ErrorRow]) -> dict[tuple[int, int], float]:
    pos_values = [row.pos_err_mm for row in rows]
    rot_values = [row.rot_err_deg for row in rows]
    grip_values = [row.grip_err_mm for row in rows]
    pos_norm = _normalize_metric(pos_values)
    rot_norm = _normalize_metric(rot_values)
    grip_norm = _normalize_metric(grip_values)
    severity: dict[tuple[int, int], float] = {}
    for row, pos_score, rot_score, grip_score in zip(rows, pos_norm, rot_norm, grip_norm, strict=True):
        severity[(row.episode, row.frame)] = max(pos_score, rot_score, grip_score)
    return severity


def _dominant_metric(rows: list[ErrorRow]) -> str:
    pos_p95 = _percentile([row.pos_err_mm for row in rows], 0.95)
    rot_p95 = _percentile([row.rot_err_deg for row in rows], 0.95)
    grip_p95 = _percentile([row.grip_err_mm for row in rows], 0.95)
    metrics = {"pos": pos_p95, "rot": rot_p95, "grip": grip_p95}
    max_value = max(metrics.values(), default=0.0)
    if max_value <= 0.0:
        return "unknown"
    winners = [name for name, value in metrics.items() if abs(value - max_value) <= 1e-12]
    return winners[0] if len(winners) == 1 else "mixed"


def _phase_summaries(rows: list[ErrorRow]) -> list[dict[str, str | float]]:
    phase_rows: dict[str, list[ErrorRow]] = defaultdict(list)
    for row in rows:
        phase_rows[row.phase].append(row)

    pos_phase_p95 = {phase: _percentile([row.pos_err_mm for row in items], 0.95) for phase, items in phase_rows.items()}
    rot_phase_p95 = {phase: _percentile([row.rot_err_deg for row in items], 0.95) for phase, items in phase_rows.items()}
    grip_phase_p95 = {phase: _percentile([row.grip_err_mm for row in items], 0.95) for phase, items in phase_rows.items()}
    pos_max = max(pos_phase_p95.values(), default=0.0)
    rot_max = max(rot_phase_p95.values(), default=0.0)
    grip_max = max(grip_phase_p95.values(), default=0.0)

    summaries: list[dict[str, str | float]] = []
    for phase, items in phase_rows.items():
        pos_p95 = pos_phase_p95[phase]
        rot_p95 = rot_phase_p95[phase]
        grip_p95 = grip_phase_p95[phase]
        aggregate = max(
            pos_p95 / pos_max if pos_max > 0.0 else 0.0,
            rot_p95 / rot_max if rot_max > 0.0 else 0.0,
            grip_p95 / grip_max if grip_max > 0.0 else 0.0,
        )
        summaries.append(
            {
                "phase": phase,
                "count": len(items),
                "pos_mean": _mean([row.pos_err_mm for row in items]),
                "rot_mean": _mean([row.rot_err_deg for row in items]),
                "grip_mean": _mean([row.grip_err_mm for row in items]),
                "pos_p95": pos_p95,
                "rot_p95": rot_p95,
                "grip_p95": grip_p95,
                "aggregate": aggregate,
            }
        )
    summaries.sort(key=lambda item: (float(item["aggregate"]), float(item["grip_p95"]), float(item["pos_p95"])), reverse=True)
    return summaries


def _earliest_divergence_candidate(rows: list[ErrorRow], phase_summaries: list[dict[str, str | float]]) -> str:
    if not rows:
        return "TODO"
    severity = _row_severity(rows)
    if not severity:
        return "TODO"

    worst_phase = str(phase_summaries[0]["phase"]) if phase_summaries else "unknown"
    phase_rows = [row for row in rows if row.phase == worst_phase]
    if not phase_rows:
        phase_rows = rows
    candidate = min(
        phase_rows,
        key=lambda row: (
            row.frame,
            row.episode,
            -severity[(row.episode, row.frame)],
        ),
    )
    return (
        f"Candidate from current CSV: episode={candidate.episode} "
        f"frame={candidate.frame} phase={candidate.phase} "
        f"(pos={candidate.pos_err_mm:.3f} mm, rot={candidate.rot_err_deg:.3f} deg, grip={candidate.grip_err_mm:.3f} mm)"
    )


def _phase_coverage_summary(rows: list[ErrorRow], phase_labels: dict[int, list[PhaseLabel]]) -> list[str]:
    if not phase_labels:
        unknown_count = sum(1 for row in rows if row.phase == "unknown")
        return [f"- phase_labels: not provided; unknown_phase_rows={unknown_count}"]

    observed_by_episode: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        observed_by_episode[row.episode].add(row.phase)

    lines: list[str] = []
    for episode in sorted(phase_labels):
        expected = [label.phase for label in phase_labels[episode]]
        observed = sorted(observed_by_episode.get(episode, set()))
        missing = [phase for phase in expected if phase not in observed]
        lines.append(
            f"- episode {episode}: observed_phases={', '.join(observed) if observed else 'none'}; "
            f"missing_expected={', '.join(missing) if missing else 'none'}"
        )
    return lines


def _first_frame_summary(rows: list[ErrorRow]) -> str:
    first_rows = [row for row in rows if row.is_first_frame]
    if not first_rows:
        return "No frame-0 rows present in the CSV."
    return (
        f"first_frame_count={len(first_rows)}; "
        f"mean_pos_err_mm={_mean([row.pos_err_mm for row in first_rows]):.3f}; "
        f"mean_rot_err_deg={_mean([row.rot_err_deg for row in first_rows]):.3f}; "
        f"mean_grip_err_mm={_mean([row.grip_err_mm for row in first_rows]):.3f}"
    )


def build_markdown(
    *,
    rows: list[ErrorRow],
    phase_labels: dict[int, list[PhaseLabel]],
    dataset: str,
    decode_gate: str,
    dataset_fed_result: str,
    dominant_blocker: str,
    next_action: str,
) -> str:
    run_id = _unique_or_join([row.run_id for row in rows]) or "TODO"
    checkpoint = _unique_or_join([row.checkpoint for row in rows]) or "TODO"
    sampled_episodes = ", ".join(str(episode) for episode in sorted({row.episode for row in rows}))
    phase_summaries = _phase_summaries(rows)
    worst_phase = str(phase_summaries[0]["phase"]) if phase_summaries else "TODO"
    dominant_metric = _dominant_metric(rows)
    earliest_divergence = _earliest_divergence_candidate(rows, phase_summaries)
    first_frame_summary = _first_frame_summary(rows)
    coverage_lines = _phase_coverage_summary(rows, phase_labels)

    phase_lines: list[str] = []
    for summary in phase_summaries[: min(len(phase_summaries), 5)]:
        phase_lines.append(
            "- "
            f"{summary['phase']}: count={int(summary['count'])}, "
            f"pos_mean={float(summary['pos_mean']):.3f} mm, "
            f"rot_mean={float(summary['rot_mean']):.3f} deg, "
            f"grip_mean={float(summary['grip_mean']):.3f} mm, "
            f"pos_p95={float(summary['pos_p95']):.3f}, "
            f"rot_p95={float(summary['rot_p95']):.3f}, "
            f"grip_p95={float(summary['grip_p95']):.3f}"
        )

    phase_counter = Counter(row.phase for row in rows)
    phase_distribution = ", ".join(f"{phase}={count}" for phase, count in sorted(phase_counter.items()))

    markdown = [
        "# FR3 Policy Validation Decision",
        "",
        f"- run_id: {run_id}",
        f"- checkpoint: {checkpoint}",
        f"- dataset: {dataset}",
        f"- sampled_episodes: {sampled_episodes}",
        f"- decode_gate: {decode_gate}",
        f"- dataset_fed_result: {dataset_fed_result}",
        f"- earliest_divergence: {earliest_divergence}",
        f"- worst_phase: Candidate from current CSV: {worst_phase}",
        f"- dominant_metric: Candidate from current CSV: {dominant_metric}",
        f"- dominant_blocker: {dominant_blocker}",
        f"- next_action: {next_action}",
        "",
        "## Auto-Filled Summary",
        "",
        f"- rows_parsed: {len(rows)}",
        f"- phase_distribution: {phase_distribution}",
        f"- {first_frame_summary}",
        "",
        "## Phase Coverage",
        "",
        *coverage_lines,
        "",
        "## Phase Candidates",
        "",
        *(phase_lines if phase_lines else ["- No phase summaries available."]),
        "",
        "## Manual Review Prompts",
        "",
        "- Confirm Gate 0 from the decode logs before trusting any dataset-fed interpretation.",
        "- Check whether the earliest candidate failure is a real divergence or only a sparse-sampling artifact.",
        "- Confirm whether `close_gripper` and `lift` transitions remain present, not just low average pose error.",
        "- Replace candidate fields above once the human review is complete.",
        "",
    ]
    return "\n".join(markdown)


def write_decision_markdown(
    *,
    per_frame_csv: Path,
    phase_labels_path: Path | None,
    output_md: Path,
    dataset: str,
    decode_gate: str,
    dataset_fed_result: str,
    dominant_blocker: str,
    next_action: str,
) -> str:
    rows = read_per_frame_rows(per_frame_csv)
    phase_labels = load_phase_labels(phase_labels_path)
    output = build_markdown(
        rows=rows,
        phase_labels=phase_labels,
        dataset=dataset,
        decode_gate=decode_gate,
        dataset_fed_result=dataset_fed_result,
        dominant_blocker=dominant_blocker,
        next_action=next_action,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(output, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    write_decision_markdown(
        per_frame_csv=args.per_frame_csv,
        phase_labels_path=args.phase_labels,
        output_md=args.output_md,
        dataset=args.dataset,
        decode_gate=args.decode_gate,
        dataset_fed_result=args.dataset_fed_result,
        dominant_blocker=args.dominant_blocker,
        next_action=args.next_action,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
