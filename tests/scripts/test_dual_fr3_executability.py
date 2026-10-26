"""dual_fr3_executability (roadmap C4): each episode's verdict in the dataset, and a replay that streams it."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.scripts.test_dual_fr3_replay_modes import (
    EXPORT_EDGE,
    TOOL,
    description,
    graph,
    hands_in_right_base,
    write_session,
)
from tools.fr3 import (
    dual_fr3_executability as ex,
    dual_fr3_replay_runtime as rt,
    dual_fr3_scene as sc,
    fr3_kinematics as kin,
)
from tools.thor.gmsl2 import bimanual_qc as bq


def _digest(session: Path) -> dict[str, str]:
    files = [session / "meta" / "info.json", *sorted((session / "derived" / bq.SIDECAR).iterdir())]
    return {str(p.relative_to(session)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _evaluate(session: Path, mode: str = "absolute", g: dict | None = None) -> dict:
    return ex.evaluate_session(
        session,
        description=description(),
        graph=g or graph(),
        mode=mode,
        obstacles=[sc.table_obstacle()],
        obstacle_source={"test": "table only"},
        mesh=False,
    )


def test_every_episode_gets_a_verdict_and_the_labels_stay_as_they_were(tmp_path: Path) -> None:
    session = write_session(tmp_path, "s", [hands_in_right_base(40), hands_in_right_base(36, phase=1.0)])
    before = _digest(session)
    out = _evaluate(session)
    assert _digest(session) == before  # info.json and the sidecars untouched
    block = out["modes"]["absolute"]
    assert block["summary"] == {"episodes": 2, "executable": 2}
    row = block["episodes"]["0"]
    assert row["dual_fr3_executable"] and row["reasons"] == []
    assert row["retarget_change_position_mm"]["right"]["max"] < 0.5 and row["min_arm_arm_gap_mm"] > 100
    assert row["mode"]["placement"]["sigma_mm"] == pytest.approx(1.0) and not row["placement_assumed"]
    assert row["d_safe_mm"] == pytest.approx(max(3.0 + 2 * 7.8, 20.0))
    report = json.loads((session / row["report"]).read_text())
    assert report["scene_fingerprint"] == row["scene_fingerprint"] and report["frames"] == list(range(40))
    assert ex.read_verdicts(session) == out


def test_a_refused_mode_is_a_verdict_too_and_modes_are_kept_apart(tmp_path: Path) -> None:
    session = write_session(tmp_path, "s", [hands_in_right_base(30)])
    moved = copy.deepcopy(EXPORT_EDGE)
    moved["created_utc"] = "2026-11-01T00:00:00Z"
    _evaluate(session, "relative", graph(export_edge=moved))
    out = _evaluate(session, "absolute", graph(export_edge=moved))
    assert out["modes"]["relative"]["episodes"]["0"]["dual_fr3_executable"]
    row = out["modes"]["absolute"]["episodes"]["0"]
    assert not row["dual_fr3_executable"] and "re-export" in row["reasons"][0]


def test_the_replay_streams_the_reference_path_and_keeps_three_time_axes(tmp_path: Path) -> None:
    session = write_session(tmp_path, "s", [hands_in_right_base(40)])
    _evaluate(session)
    reference = ex.plans_from_retarget(session, 0, "absolute")
    assert reference["fps"] == ex.DEFAULT_REPLAY_FPS
    labels = hands_in_right_base(40)
    # 60 fps labels at 20 Hz, three times slower: one command per label
    assert len(reference["targets"]["right"]) == 40
    assert np.abs(reference["targets"]["right"][:, :3, 3] - labels["right"][:, :3, 3]).max() < 5e-4
    assert np.allclose(reference["targets"]["left"], kin.NOMINAL.fk(reference["q"]["left"]) @ TOOL)

    plans, T_right_left, fps, timelines, record = rt.plans_from_retarget(session, 0, "absolute")
    assert plans[0].joints is not None and record["scene_fingerprint"]
    report = rt.run(
        plans, fps=50.0, T_right_left=T_right_left, run_dir=tmp_path / "run",
        settings=rt.Settings(d_safe_m=0.02), go=lambda: True, start_lead_s=0.1, timelines=timelines,
    )  # fmt: skip
    assert report["stop"] is None and all(r["outcome"] == "completed" for r in report["arms"].values())
    saved = np.load(tmp_path / "run" / "timelines.npz")
    assert {"t_raw", "t_ref", "t_executed_right", "t_executed_left"} <= set(saved.files)
    assert len(saved["t_executed_right"]) == len(saved["t_ref"]) == 40
    assert saved["t_raw"][-1] == pytest.approx(39 / 60.0)


def test_a_replay_of_an_episode_c4_rejected_is_refused(tmp_path: Path) -> None:
    session = write_session(tmp_path, "s", [hands_in_right_base(30)])
    _evaluate(session)
    path, _ = ex.retarget_paths(session, 0, "absolute")
    report = json.loads(path.read_text())
    report.update(executable=False, reasons=["arms 10 mm apart"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not executable: arms 10 mm apart"):
        rt.plans_from_retarget(session, 0, "absolute")


def test_the_operator_s_abort_stops_both_arms_together(tmp_path: Path) -> None:
    """D4: the GUI aborts a running two-arm replay with SIGTERM; it is one stop for both, like a fault."""
    import os
    import signal
    import threading

    session = write_session(tmp_path, "s", [hands_in_right_base(40)])
    _evaluate(session)
    plans, T_right_left, _, timelines, _ = rt.plans_from_retarget(session, 0, "absolute")

    def go() -> bool:
        threading.Timer(0.6, os.kill, (os.getpid(), signal.SIGTERM)).start()
        return True

    before = signal.getsignal(signal.SIGTERM)
    report = rt.run(
        plans, fps=20.0, T_right_left=T_right_left, run_dir=tmp_path / "run",
        settings=rt.Settings(d_safe_m=0.02), go=go, start_lead_s=0.1, timelines=timelines,
    )  # fmt: skip
    assert signal.getsignal(signal.SIGTERM) is before  # restored
    assert report["stop"]["events"][0]["reason"] == "operator abort (SIGTERM)"
    assert report["stopped_together"]
    assert all(r["outcome"] != "completed" for r in report["arms"].values())


def test_the_real_backend_says_what_it_still_waits_for() -> None:
    from tools.fr3 import fr3_arms

    registry = fr3_arms.load()
    assert rt.real_backend_refusal(registry).startswith("the left arm is not in service: B1")
    both = copy.deepcopy(registry)
    both["arms"]["left"].update(in_service=True, robot_ip="192.168.11.103")
    assert "real-arm driver is not written yet" in rt.real_backend_refusal(both)
