#!/usr/bin/env python3
"""Build FR3 policy validation CSV and decision artifacts from raw logs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.fr3 import fr3_generate_policy_validation_decision as decision_builder
    from tools.fr3 import fr3_parse_policy_dataset_frame_log as log_parser
except ImportError:  # pragma: no cover - fallback for direct script execution
    import fr3_generate_policy_validation_decision as decision_builder
    import fr3_parse_policy_dataset_frame_log as log_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FR3 policy validation report artifacts from raw logs.")
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        required=True,
        help="Path to a raw log file from tools/fr3/fr3_check_policy_dataset_frame.py. Repeat for multiple logs.",
    )
    parser.add_argument("--phase-labels", type=Path, default=None, help="Optional phase_labels.csv for phase mapping and coverage.")
    parser.add_argument("--run-id", required=True, help="Run identifier to stamp into generated artifacts.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path string to stamp into generated artifacts.")
    parser.add_argument("--dataset", default="TODO", help="Dataset identifier or path for decision.md.")
    parser.add_argument("--per-frame-csv", type=Path, required=True, help="Output path for per_frame_action_error.csv.")
    parser.add_argument("--topk-out", type=Path, default=None, help="Optional output path for topk_worst_frames.csv.")
    parser.add_argument("--topk", type=int, default=10, help="Number of rows to write when --topk-out is provided.")
    parser.add_argument("--output-md", type=Path, required=True, help="Output path for decision.md.")
    parser.add_argument("--decode-gate", default="TODO", help="Human-entered Gate 0 status string.")
    parser.add_argument("--dataset-fed-result", default="TODO", help="Human-entered Gate 1 status string.")
    parser.add_argument("--dominant-blocker", default="TODO", help="Human-entered current blocker hypothesis.")
    parser.add_argument("--next-action", default="TODO", help="Human-entered next action.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_parser.build_csv_artifacts(
        log_paths=[Path(log_arg) for log_arg in args.logs],
        phase_labels_path=args.phase_labels,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        output_csv=args.per_frame_csv,
        topk_out=args.topk_out,
        topk=args.topk,
    )
    decision_builder.write_decision_markdown(
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
