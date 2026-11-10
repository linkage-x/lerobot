"""bimanual_qc (roadmap A1): what a two-arm replay needs from the two hands' tracks together."""

import csv
import math
from pathlib import Path

import pytest

from tools.thor.gmsl2 import bimanual_qc as bq

FPS = 60.0


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(frame: int, p, *, yaw: float = 0.0, ok: bool = True, cams: int = 5, cset: str = "06,07,08,09,13",
         changed: bool = False, episode: int = 0) -> dict:
    return {
        "schema_version": 3,
        "episode_index": episode,
        "frame_index": frame,
        "timestamp_s": frame / FPS,
        "state_x_m": p[0] if ok else "nan",
        "state_y_m": p[1] if ok else "nan",
        "state_z_m": p[2] if ok else "nan",
        "state_qx": 0.0,
        "state_qy": 0.0,
        "state_qz": math.sin(yaw / 2),
        "state_qw": math.cos(yaw / 2),
        "solve_ok": ok,
        "num_cameras": cams if ok else 0,
        "camera_set": cset if ok else "",
        "camera_set_changed": changed,
    }


def _session(
    tmp_path: Path,
    *,
    n: int = 600,
    right_step_at: int | None = None,
    left_gap=None,
    few_cams=None,
    left_y: float = 0.3,
    left_yaw=lambda t: 0.3 * math.sin(t),
) -> Path:
    left, right = [], []
    for f in range(n):
        t = f / FPS
        pl = (0.4 + 0.1 * math.sin(t), left_y, 0.2 + 0.05 * math.cos(t))
        pr = (0.4 + 0.1 * math.sin(t) + 0.02 * math.sin(2 * t), 0.0, 0.2)
        if right_step_at is not None and f >= right_step_at:
            pr = (pr[0], pr[1] + 0.012, pr[2])
        lost = left_gap is not None and left_gap[0] <= f < left_gap[1]
        cams = 2 if few_cams is not None and few_cams[0] <= f < few_cams[1] else 5
        left.append(_row(f, pl, yaw=left_yaw(t), ok=not lost, cams=cams))
        right.append(_row(f, pr, changed=right_step_at == f, cset="06,07,08" if right_step_at and f >= right_step_at else "06,07,08,09,13"))
    session = tmp_path / "session_a"
    _write(session / "derived" / bq.SIDECAR / "state_action.left.csv", left)
    _write(session / "derived" / bq.SIDECAR / "state_action.right.csv", right)
    return session


def test_relative_position_is_the_right_tcp_in_the_left_frame() -> None:
    left = {"p": [1.0, 0.0, 0.0], "q": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]}  # yaw 90 deg
    right = {"p": [1.0, 2.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
    assert bq.relative_position(left, right) == pytest.approx([2.0, 0.0, 0.0], abs=1e-12)


def test_a_clean_session_passes_with_one_usable_segment(tmp_path: Path) -> None:
    report = bq.session_qc(_session(tmp_path))
    row = report["episodes"]["0"]
    assert row["bimanual_pass"] and row["tlr_steps"] == 0
    assert row["both_seen"] == row["frames"] == 600
    assert row["tlr_jitter_mm"] < 0.05  # noiseless tracks: only the motion's own curvature
    assert row["usable_segments"] == [{"start_frame": 0, "end_frame": 599, "duration_s": 10.0}]


def test_a_camera_set_step_in_one_hand_is_a_t_lr_step_and_splits_the_segment(tmp_path: Path) -> None:
    row = bq.session_qc(_session(tmp_path, right_step_at=300))["episodes"]["0"]
    assert row["tlr_steps"] == 1 and row["tlr_steps_on_camera_set_change"] == 1
    assert 299 <= row["tlr_step_frames"][0] <= 300
    assert len(row["usable_segments"]) == 2


def test_the_left_hand_turning_is_not_a_step(tmp_path: Path) -> None:
    # 10-26, pseudo_bimanual pairs: read in the left TCP's frame, a wrist flick (0.5 rad at 8 rad/s, so
    # 32 rad/s^2) across 0.8 m between the hands is a 7 mm second difference -- every pair failed.
    row = bq.session_qc(_session(tmp_path, left_y=0.8, left_yaw=lambda t: 0.5 * math.sin(8 * t)))["episodes"]["0"]
    assert row["tlr_steps"] == 0 and row["bimanual_pass"]


def test_a_one_hand_loss_longer_than_allowed_fails(tmp_path: Path) -> None:
    row = bq.session_qc(_session(tmp_path, left_gap=(200, 260)))["episodes"]["0"]
    assert not row["bimanual_pass"]
    assert row["one_hand_losses"] == 1 and row["one_hand_losses_over_max"] == 1
    assert row["right_only"] == 60
    assert any("one hand lost" in r for r in row["reasons"])


def test_frames_with_too_few_cameras_are_stratified_and_not_usable(tmp_path: Path) -> None:
    row = bq.session_qc(_session(tmp_path, few_cams=(0, 200)))["episodes"]["0"]
    assert row["camera_strata"] == {"L2-3 R>=4": 200, "L>=4 R>=4": 400}
    assert row["usable_segments"][0]["start_frame"] == 200
    assert not row["bimanual_pass"]  # 67 % < 80 % both seen


def test_a_v1_sidecar_or_a_missing_hand_is_reported_not_guessed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    (session / "derived" / bq.SIDECAR / "state_action.left.csv").unlink()
    assert "left" in bq.session_qc(session)["error"]
    rows = [{"episode_index": 0, "frame_index": 0, "state_x_m": 0}]
    _write(session / "derived" / bq.SIDECAR / "state_action.left.csv", rows)
    with pytest.raises(ValueError, match="v1 sidecar"):
        bq.session_qc(session)
