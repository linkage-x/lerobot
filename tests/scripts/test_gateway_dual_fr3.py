"""The gateway serves an episode's two-arm verdicts (roadmap C4) read-only, as the replay page shows them."""

from __future__ import annotations

import json
import os
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
    from tests.scripts.test_dual_fr3_replay_modes import hands_in_left_base, write_session

    session = write_session(tmp_path, "s", [hands_in_left_base(frames)])
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
    assert started["job"]["edges"]["left"] == [] and started["job"]["edges"]["right"]  # left in fr3_base
    assert "already running" in gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})["error"]
    _await_job(state, lambda j: j.get("state") == "at_start")
    assert gateway._dual_replay_execute(state, {})["ok"]
    job = _await_job(state, lambda j: j.get("state") in ("finished", "stopped", "error"))
    assert job["state"] == "finished", job
    report = job["report"]
    assert report["backend"] == "fake" and report["stop"] == []
    assert {arm: row["outcome"] for arm, row in report["arms"].items()} == {"right": "completed", "left": "completed"}
    assert Path(report["path"]).is_file()
    assert report["clock"]["status"] in ("passed", "failed") and "one period" in report["clock"]["verdict"]  # D2


def test_abort_at_frame_0_starts_nothing(tmp_path: Path, monkeypatch) -> None:
    state = _replay_state(tmp_path, monkeypatch)
    assert gateway._dual_replay_connect(state, {"mode": "absolute", "backend": "fake"})["ok"]
    _await_job(state, lambda j: j.get("state") == "at_start")
    gateway._dual_replay_abort(state, {})
    job = _await_job(state, lambda j: j.get("state") in ("finished", "stopped", "error"))
    assert job["state"] == "stopped" and job["report"]["stop"] == ["operator did not say go"]
    assert all(row["framesCommanded"] == 0 for row in job["report"]["arms"].values())


def test_connect_refuses_what_c4_did_not_clear_and_the_real_arms_without_panda_py(tmp_path: Path, monkeypatch) -> None:
    state = _replay_state(tmp_path, monkeypatch)
    # both arms are registered since 10-26: a panda_py that will not import keeps the runtime off the network
    blocked = tmp_path / "no_panda_py"
    (blocked / "panda_py").mkdir(parents=True)
    (blocked / "panda_py" / "__init__.py").write_text("raise ImportError('not in tests')\n")
    env = gateway._real_preflight_env(state)
    env = {**env, "PYTHONPATH": f"{blocked}{os.pathsep}{env['PYTHONPATH']}"}
    monkeypatch.setattr(gateway, "_real_preflight_env", lambda _s: env)
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
    assert job["state"] == "error" and "panda_py does not import" in job["error"], job


# ------------------------------------------------------------------ both arms home ---


def _home_state(tmp_path: Path, monkeypatch, right_away: str = "#q=1:0.5"):
    """The real runtime and the real registry, on the stand-in panda_py: a ``panda_py`` shim first on the
    path, which puts the right arm ``right_away`` (fake_panda_py's host suffix) from its start."""
    import sys

    shim = tmp_path / "shim" / "panda_py"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text(
        "from tests.scripts import fake_panda_py as _fake\n"
        "class Panda(_fake.Panda):\n"
        "    def __init__(self, hostname, name='panda'):\n"
        f"        super().__init__(hostname + ({right_away!r} if hostname == '192.168.1.210' else ''), name)\n"
    )
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
    )
    (tmp_path / "tools" / "fr3").mkdir(parents=True)
    (tmp_path / gateway._DUAL_HOME_SCRIPT).symlink_to(Path.cwd() / gateway._DUAL_HOME_SCRIPT)
    # the registry and the replay configs it names, for the gateway's own arm lookups
    for name in ("tools/fr3/fr3_arms.json", "tools/fr3/replay_cube_pose_in_robot_base.thor.right.yaml",
                 "third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.thor.yaml"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).symlink_to(Path.cwd() / name)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(shim.parent), str(Path.cwd() / "src"), str(Path.cwd())]))
    monkeypatch.setattr(gateway, "_mujoco_replay_python", lambda _s: Path(sys.executable))
    monkeypatch.setattr(gateway, "_real_preflight_env", lambda _s: env)
    return state


def _await_home(state, done, timeout_s=60.0):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = gateway._dual_home_payload(state)["job"]
        if done(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"both-arm homing stuck in {gateway._dual_home_payload(state)['job']}")


def test_check_shows_the_plan_and_execute_brings_the_arm_home(tmp_path: Path, monkeypatch) -> None:
    state = _home_state(tmp_path, monkeypatch)
    assert "先 Check" in gateway._dual_home_execute(state, {})["error"]
    assert gateway._dual_home_check(state, {})["ok"]
    assert "已在进行" in gateway._dual_home_check(state, {})["error"]
    # the single-arm button and a real two-arm replay keep off the arms meanwhile
    assert "双臂归位正在进行" in gateway._fr3_move_to_start(state, {"arm": "left"})["error"]
    job = _await_home(state, lambda j: j.get("state") != "checking")
    assert job["state"] == "ready", job
    plan = job["plan"]
    assert plan["stages"] == [["right"]] and plan["chosen"] == "together"
    assert plan["placement"]["T_right_left"][1][3] == 0.66 and plan["gate_mm"] == 60.0
    assert gateway._dual_home_execute(state, {})["ok"]
    job = _await_home(state, lambda j: j.get("state") not in ("ready", "moving"))
    assert job["state"] == "done" and job["result"]["ok"], job


def test_cancel_moves_nothing_and_arms_at_home_need_no_answer(tmp_path: Path, monkeypatch) -> None:
    state = _home_state(tmp_path, monkeypatch)
    assert gateway._dual_home_check(state, {})["ok"]
    _await_home(state, lambda j: j.get("state") == "ready")
    gateway._dual_home_cancel(state, {})
    job = _await_home(state, lambda j: j.get("state") not in ("ready", "checking"))
    assert job["state"] == "cancelled" and job["result"] is None
    home = _home_state(tmp_path / "home", monkeypatch, right_away="")
    assert gateway._dual_home_check(home, {})["ok"]
    job = _await_home(home, lambda j: j.get("state") != "checking")
    assert job["state"] == "done" and job["plan"]["stages"] == [] and job["result"] is None


def test_check_refuses_while_another_fr3_job_holds_an_arm(tmp_path: Path, monkeypatch) -> None:
    state = _home_state(tmp_path, monkeypatch)
    with gateway._FR3_MOVE_TO_START_LOCK:
        assert "单臂 move to start" in gateway._dual_home_check(state, {})["error"]
    state.fr3_pivot_job = {"state": "collecting"}
    assert "pivot" in gateway._dual_home_check(state, {})["error"]


# --- a pseudo_bimanual pair set on the replay page ------------------------------------------------


def test_a_pseudo_bimanual_set_is_listed_and_selected_for_the_two_arm_panels_only(tmp_path: Path) -> None:
    # no parquet, no videos: the pairs and their two-arm verdicts are all it holds
    datasets = tmp_path / "outputs" / "datasets"
    pairs = datasets / "pseudo_bimanual_a5_s066"
    (pairs / "meta").mkdir(parents=True)
    info = {"fps": 60.0, "pseudo_bimanual": {"pairs": [{"episode": 1, "frames": 300}, {"episode": 0, "frames": 706}]}}
    (pairs / "meta" / "info.json").write_text(json.dumps(info))
    (pairs / gateway.DUAL_FR3_EXECUTABLE_FILE).write_text(json.dumps(VERDICTS))
    (datasets / "unfinished" / "meta").mkdir(parents=True)
    (datasets / "unfinished" / "meta" / "info.json").write_text("{}")
    state = gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
        datasets_root=datasets,
    )
    (item,) = gateway._recorded_dataset_items(state)
    assert item["name"] == pairs.name and item["datasetKind"] == "pseudo_bimanual"
    assert (item["totalEpisodes"], item["totalFrames"], item["dataStatus"]) == (2, 1006, "loaded")
    assert not item["isLatest"]  # never "the capture that just happened"

    gateway._select_replay_dataset(state, str(pairs))
    assert state.replay.episodeOptions == [0, 1] and state.replay.datasetKind == "pseudo_bimanual"
    gateway._select_replay_episode(state, "1")
    assert gateway._dual_fr3_executability_payload(pairs, 1)["modes"]["absolute"]["episode"]["reasons"]
    gateway._snapshot(state)  # the page's poll holds up on a set with no frames of its own
    with pytest.raises(RuntimeError, match="two arms only"):
        gateway._require_mujoco_validation(state)
