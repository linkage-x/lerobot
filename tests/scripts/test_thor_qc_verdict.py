from __future__ import annotations

import json
import os
from pathlib import Path

from tools.thor.gmsl2 import qc_verdict as qcv

SIDECAR = Path("derived") / qcv.TRAJ_SIDECAR_NAME


def _session(root: Path, episodes=(0, 1)) -> Path:
    session = root / "pick_20261116_101010"
    (session / "meta").mkdir(parents=True)
    (session / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (session / "data" / "chunk-000").mkdir(parents=True)
    (session / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"rows")
    for ep in episodes:
        ep_dir = session / "episodes" / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True)
        (ep_dir / "cam_00.mkv").write_bytes(b"video" * (ep + 1))
        (ep_dir / "meta.json").write_text("{}", encoding="utf-8")
    (session / SIDECAR).mkdir(parents=True)
    (session / SIDECAR / "state_action.right.csv").write_text("t,x\n", encoding="utf-8")
    return session


def _qc(session: Path, status: str, **extra) -> dict:
    return {"status": status, "summary": status, "data_fingerprint": qcv.data_fingerprint(session), **extra}


def _touch_later(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_what_qc_ik_and_replay_write_afterwards_leaves_the_fingerprint_alone(tmp_path):
    session = _session(tmp_path)
    before = qcv.data_fingerprint(session)

    (session / "meta" / "processing.json").write_text("{}", encoding="utf-8")
    (session / "meta" / "gui_replay_validations.json").write_text("{}", encoding="utf-8")
    (session / SIDECAR / "by_arm").mkdir()
    (session / SIDECAR / "by_arm" / "state_action.right.csv").write_text("x", encoding="utf-8")
    (session / SIDECAR / "ik_qc" / "right").mkdir(parents=True)
    (session / SIDECAR / "ik_qc" / "right" / "report.json").write_text("{}", encoding="utf-8")
    (session / "derived" / "table_check.json").write_text("{}", encoding="utf-8")

    assert qcv.data_fingerprint(session) == before


def test_rewritten_renumbered_or_retracked_data_changes_the_fingerprint(tmp_path):
    session = _session(tmp_path, episodes=(0, 1, 2))
    seen = {qcv.data_fingerprint(session)}

    _touch_later(session / "data" / "chunk-000" / "file-000.parquet")  # BOX re-pairing rewrote rows
    seen.add(qcv.data_fingerprint(session))
    (session / "episodes" / "episode_000000").rename(session / "episodes" / "gone")  # an episode deleted
    seen.add(qcv.data_fingerprint(session))
    (session / "episodes" / "episode_000001").rename(session / "episodes" / "episode_000000")  # renumbered
    seen.add(qcv.data_fingerprint(session))
    (session / SIDECAR / "state_action.right.csv").write_text("t,x\n1,2\n", encoding="utf-8")  # re-tracked
    seen.add(qcv.data_fingerprint(session))

    assert len(seen) == 5


def test_a_verdict_names_what_the_export_may_take(tmp_path):
    session = _session(tmp_path)

    assert qcv.verdict_from_qc(session, None).state == qcv.NOT_RUN
    assert qcv.verdict_from_qc(session, _qc(session, "pass")).state == qcv.PASS
    assert qcv.verdict_from_qc(session, _qc(session, "warn")).state == qcv.PASS
    assert qcv.verdict_from_qc(session, _qc(session, "fail")).state == qcv.FAIL

    partial = qcv.verdict_from_qc(
        session, _qc(session, "partial", excluded_episodes=[{"episode": 1, "checks": ["quat_norm"]}])
    )
    assert partial.state == qcv.PARTIAL
    assert partial.allows(0) and not partial.allows(1)
    assert partial.excluded == {1: "QC: quat_norm"}

    # "partial" that names no episode cannot say what to keep
    assert qcv.verdict_from_qc(session, _qc(session, "partial")).state == qcv.FAIL


def test_a_qc_result_without_a_fingerprint_or_for_other_data_is_stale(tmp_path):
    session = _session(tmp_path)
    legacy = {"status": "pass", "summary": "old"}
    assert qcv.verdict_from_qc(session, legacy).state == qcv.STALE

    qc = _qc(session, "pass")
    (session / "episodes" / "episode_000001" / "cam_00.mkv").unlink()
    verdict = qcv.verdict_from_qc(session, qc)
    assert verdict.state == qcv.STALE
    assert not verdict.exportable and not verdict.allows(0)


def test_read_verdict_uses_the_active_versions_qc(tmp_path):
    session = _session(tmp_path)
    meta = {
        "active_version": "v2",
        "versions": {"v1": {"qc": _qc(session, "pass")}, "v2": {"qc": None}},
    }
    (session / "meta" / "processing.json").write_text(json.dumps(meta), encoding="utf-8")
    assert qcv.read_verdict(session).state == qcv.NOT_RUN

    meta["versions"]["v2"]["qc"] = _qc(session, "pass")
    (session / "meta" / "processing.json").write_text(json.dumps(meta), encoding="utf-8")
    assert qcv.read_verdict(session).state == qcv.PASS
