"""The gateway serves an episode's two-arm verdicts (roadmap C4) read-only, as the replay page shows them."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.data_collection_gui import gateway


def _state(tmp_path: Path, verdicts: dict | None) -> gateway.GatewayState:
    dataset_root = tmp_path / "session"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    if verdicts is not None:
        (dataset_root / gateway.DUAL_FR3_EXECUTABLE_FILE).write_text(json.dumps(verdicts))
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
    )
    state.selected_replay_root = dataset_root
    return state


VERDICTS = {
    "schema": "dual_fr3_executable/v1",
    "modes": {
        "absolute": {
            "created_utc": "2026-10-26T04:00:00Z",
            "embodiment": "fr3_dual",
            "summary": {"episodes": 2, "executable": 1},
            "episodes": {
                "0": {"dual_fr3_executable": True, "reasons": []},
                "1": {"dual_fr3_executable": False, "reasons": ["arms 10 mm apart"]},
            },
        }
    },
}


def test_an_episode_s_verdicts_are_served_per_mode(tmp_path: Path) -> None:
    state = _state(tmp_path, VERDICTS)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=1", timeout=5) as response:
            payload = json.loads(response.read())
        absolute = payload["modes"]["absolute"]
        assert payload["episode"] == 1 and absolute["summary"] == {"episodes": 2, "executable": 1}
        assert absolute["episode"] == {"dual_fr3_executable": False, "reasons": ["arms 10 mm apart"]}
        with urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=5", timeout=5) as response:
            assert json.loads(response.read())["modes"]["absolute"]["episode"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_a_dataset_without_verdicts_is_a_404_that_says_what_to_run(tmp_path: Path) -> None:
    state = _state(tmp_path, None)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=0", timeout=5)
        assert missing.value.code == 404
        assert "dual_fr3_executability.py" in json.loads(missing.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


# --- the operator's half (roadmap D4) -------------------------------------------------------------


def _replay_state(tmp_path: Path, monkeypatch, frames: int = 30):
    import os
    import sys

    from tests.scripts.test_dual_fr3_executability import _evaluate
    from tests.scripts.test_dual_fr3_replay_modes import hands_in_right_base, write_session

    session = write_session(tmp_path, "s", [hands_in_right_base(frames)])
    _evaluate(session)
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
    )
    state.selected_replay_root = session
    (tmp_path / "tools" / "fr3").mkdir(parents=True)
    # the real runtime, run from the repo: the tmp repo only holds the outputs
    (tmp_path / gateway._DUAL_REPLAY_SCRIPT).symlink_to(Path.cwd() / gateway._DUAL_REPLAY_SCRIPT)
    env = dict(os.environ, PYTHONPATH=f"{Path.cwd() / 'src'}{os.pathsep}{Path.cwd()}")
    monkeypatch.setattr(gateway, "_mujoco_replay_python", lambda _s: Path(sys.executable))
    monkeypatch.setattr(gateway, "_real_preflight_env", lambda _s: env)
    return state


def _await_job(state, done, timeout_s=60.0):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = gateway._dual_replay_job_view(state)
        if done(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"two-arm replay stuck in {gateway._dual_replay_job_view(state)}")


def test_connect_waits_at_frame_0_and_execute_runs_both_arms(tmp_path: Path, monkeypatch) -> None:
    state = _replay_state(tmp_path, monkeypatch)
    assert "Connect first" in gateway._dual_replay_execute(state, {})["error"]
    started = gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})
    assert started["ok"], started["error"]
    assert started["job"]["edges"]["right"] == [] and started["job"]["edges"]["left"]
    assert "already running" in gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})["error"]
    _await_job(state, lambda j: j.get("state") == "at_start")
    assert gateway._dual_replay_execute(state, {})["ok"]
    job = _await_job(state, lambda j: j.get("state") in ("finished", "stopped", "error"))
    assert job["state"] == "finished", job
    report = job["report"]
    assert report["backend"] == "fake" and report["stop"] == []
    assert {arm: row["outcome"] for arm, row in report["arms"].items()} == {"right": "completed", "left": "completed"}
    assert Path(report["path"]).is_file()


def test_abort_at_frame_0_starts_nothing(tmp_path: Path, monkeypatch) -> None:
    state = _replay_state(tmp_path, monkeypatch)
    assert gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})["ok"]
    _await_job(state, lambda j: j.get("state") == "at_start")
    gateway._dual_replay_abort(state, {})
    job = _await_job(state, lambda j: j.get("state") in ("finished", "stopped", "error"))
    assert job["state"] == "stopped" and job["report"]["stop"] == ["operator did not say go"]
    assert all(row["framesCommanded"] == 0 for row in job["report"]["arms"].values())


def test_connect_refuses_what_c4_did_not_clear_and_the_real_arms_until_they_exist(tmp_path: Path, monkeypatch) -> None:
    state = _replay_state(tmp_path, monkeypatch)
    assert "has no relative verdict" in gateway._dual_replay_connect(state, {"mode": "relative", "backend": "fake"})["error"]
    assert "backend must be" in gateway._dual_replay_connect(state, {"mode": "absolute"})["error"]
    path = state.selected_replay_root / gateway.DUAL_FR3_EXECUTABLE_FILE
    verdicts = json.loads(path.read_text())
    verdicts["modes"]["absolute"]["episodes"]["0"].update(dual_fr3_executable=False, reasons=["arms 10 mm apart"])
    path.write_text(json.dumps(verdicts))
    refused = gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})
    assert "not executable in absolute: arms 10 mm apart" in refused["error"]
    verdicts["modes"]["absolute"]["episodes"]["0"].update(dual_fr3_executable=True, reasons=[])
    path.write_text(json.dumps(verdicts))
    assert gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "real"})["ok"]
    job = _await_job(state, lambda j: j.get("state") in ("finished", "stopped", "error"))
    assert job["state"] == "error" and job["error"].startswith("the left arm is not in service: B1")
