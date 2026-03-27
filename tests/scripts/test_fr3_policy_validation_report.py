#!/usr/bin/env python

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

from tools.fr3 import (
    fr3_build_policy_validation_report,
    fr3_generate_policy_validation_decision,
    fr3_parse_policy_dataset_frame_log,
)


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_phase_labels(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "frame_start", "frame_end", "phase"])
        writer.writeheader()
        writer.writerow({"episode": 0, "frame_start": 0, "frame_end": 1, "phase": "approach"})
        writer.writerow({"episode": 0, "frame_start": 2, "frame_end": 3, "phase": "close_gripper"})
        writer.writerow({"episode": 1, "frame_start": 0, "frame_end": 0, "phase": "lift"})
    return path


def test_collect_log_rows_and_build_csv_artifacts(tmp_path: Path):
    log_path = _write_text(
        tmp_path / "run.log",
        "\n".join(
            [
                "[INFO] ignore this line",
                "[CHECK] episode=0 frame=0 pos_err_mm=1.0 rot_err_deg=2.0 grip_err_mm=3.0",
                "[CHECK] episode=0 frame=2 pos_err_mm=4.0 rot_err_deg=1.0 grip_err_mm=0.5",
                "[CHECK] episode=1 frame=0 pos_err_mm=0.2 rot_err_deg=0.1 grip_err_mm=9.0",
                "",
            ]
        ),
    )
    phase_labels_path = _write_phase_labels(tmp_path / "phase_labels.csv")
    output_csv = tmp_path / "per_frame_action_error.csv"
    topk_csv = tmp_path / "topk_worst_frames.csv"

    rows = fr3_parse_policy_dataset_frame_log.build_csv_artifacts(
        log_paths=[log_path],
        phase_labels_path=phase_labels_path,
        run_id="run-42",
        checkpoint="ckpt.pt",
        output_csv=output_csv,
        topk_out=topk_csv,
        topk=2,
    )

    assert [(row.episode, row.frame, row.phase) for row in rows] == [
        (0, 0, "approach"),
        (0, 2, "close_gripper"),
        (1, 0, "lift"),
    ]

    per_frame_text = output_csv.read_text(encoding="utf-8")
    assert "run-42,ckpt.pt,0,0,approach,true,1.000000,2.000000,3.000000" in per_frame_text
    assert "run-42,ckpt.pt,1,0,lift,true,0.200000,0.100000,9.000000" in per_frame_text

    topk_text = topk_csv.read_text(encoding="utf-8")
    assert "dominant_metric" in topk_text
    assert "close_gripper" in topk_text
    assert "approach" in topk_text


def test_write_decision_markdown_uses_public_readers(tmp_path: Path):
    per_frame_csv = tmp_path / "per_frame_action_error.csv"
    per_frame_csv.write_text(
        "\n".join(
            [
                "run_id,checkpoint,episode,frame,phase,is_first_frame,pos_err_mm,rot_err_deg,grip_err_mm",
                "run-42,ckpt.pt,0,0,approach,true,1.000000,2.000000,3.000000",
                "run-42,ckpt.pt,0,2,close_gripper,false,4.000000,1.000000,0.500000",
                "run-42,ckpt.pt,1,0,lift,true,0.200000,0.100000,9.000000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    phase_labels_path = _write_phase_labels(tmp_path / "phase_labels.csv")
    output_md = tmp_path / "decision.md"

    markdown = fr3_generate_policy_validation_decision.write_decision_markdown(
        per_frame_csv=per_frame_csv,
        phase_labels_path=phase_labels_path,
        output_md=output_md,
        dataset="demo-dataset",
        decode_gate="pass",
        dataset_fed_result="fail",
        dominant_blocker="gripper timing",
        next_action="inspect frame 0",
    )

    assert output_md.read_text(encoding="utf-8") == markdown
    assert "- dataset: demo-dataset" in markdown
    assert "- decode_gate: pass" in markdown
    assert "- dataset_fed_result: fail" in markdown
    assert "phase_distribution: approach=1, close_gripper=1, lift=1" in markdown
    assert "episode 0: observed_phases=approach, close_gripper; missing_expected=none" in markdown


def test_build_policy_validation_report_end_to_end(tmp_path: Path):
    log_path = _write_text(
        tmp_path / "run.log",
        "\n".join(
            [
                "[CHECK] episode=0 frame=0 pos_err_mm=1.0 rot_err_deg=2.0 grip_err_mm=3.0",
                "[CHECK] episode=0 frame=2 pos_err_mm=4.0 rot_err_deg=1.0 grip_err_mm=0.5",
                "[CHECK] episode=1 frame=0 pos_err_mm=0.2 rot_err_deg=0.1 grip_err_mm=9.0",
                "",
            ]
        ),
    )
    phase_labels_path = _write_phase_labels(tmp_path / "phase_labels.csv")
    per_frame_csv = tmp_path / "per_frame_action_error.csv"
    topk_csv = tmp_path / "topk_worst_frames.csv"
    output_md = tmp_path / "decision.md"

    exit_code = fr3_build_policy_validation_report.main(
        [
            "--log",
            str(log_path),
            "--phase-labels",
            str(phase_labels_path),
            "--run-id",
            "run-42",
            "--checkpoint",
            "ckpt.pt",
            "--dataset",
            "demo-dataset",
            "--per-frame-csv",
            str(per_frame_csv),
            "--topk-out",
            str(topk_csv),
            "--topk",
            "2",
            "--output-md",
            str(output_md),
            "--decode-gate",
            "pass",
            "--dataset-fed-result",
            "fail",
            "--dominant-blocker",
            "gripper timing",
            "--next-action",
            "inspect frame 0",
        ]
    )

    assert exit_code == 0
    assert per_frame_csv.exists()
    assert topk_csv.exists()
    assert output_md.exists()
    assert "run-42,ckpt.pt,0,2,close_gripper,false,4.000000,1.000000,0.500000" in per_frame_csv.read_text(
        encoding="utf-8"
    )
    markdown = output_md.read_text(encoding="utf-8")
    assert "- dominant_blocker: gripper timing" in markdown
    assert "- next_action: inspect frame 0" in markdown
    assert "## Phase Candidates" in markdown


def test_generate_policy_validation_decision_supports_direct_script_execution():
    script_path = Path(__file__).resolve().parents[2] / "tools/fr3/fr3_generate_policy_validation_decision.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=script_path.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate FR3 decision.md scaffolding" in result.stdout


def test_build_policy_validation_report_supports_direct_script_execution():
    script_path = Path(__file__).resolve().parents[2] / "tools/fr3/fr3_build_policy_validation_report.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=script_path.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Build FR3 policy validation report artifacts" in result.stdout
