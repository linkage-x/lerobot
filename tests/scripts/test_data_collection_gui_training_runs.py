from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.data_collection_gui import gateway, training_runs as tr

STATE_NAMES = ["box_gripper.distance_m", "box_trigger.travel_pct"] + [
    f"box_six_d_external.{a}" for a in ("fx", "fy", "fz", "mx", "my", "mz")
]


def _export(root: Path, name: str, *, gated: bool = True, link: str = "corenetic_gripper_ee") -> Path:
    path = root / "exports" / name
    (path / "meta").mkdir(parents=True)
    info = {
        "fps": 60,
        "total_episodes": 2,
        "total_frames": 600,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
            "observation.ee_pose.right.world": {"dtype": "float32", "shape": [7]},
            "observation.images.cam_10": {"dtype": "video", "shape": [1080, 1920, 3]},
            "observation.images.cam_06": {"dtype": "video", "shape": [1080, 1920, 3]},
        },
        "tcp_frame": {"link": link},
        "world_frame": {"world_frame_id": "world_x"},
        "label_provenance": {"fingerprint": "abc", "key": {"world_frame_id": "world_x"}, "hands": {}},
    }
    if gated:
        info["qc_gate"] = {"enforced": True}
    (path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return path


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid


def _manager(tmp_path: Path, **kw) -> tr.TrainingManager:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return tr.TrainingManager(repo, exports_root=tmp_path / "exports", datasets_root=tmp_path / "datasets", **kw)


def test_suggest_crop_boxes_the_rig_with_margin_at_4_by_3_inside_the_image():
    pts = [(300.0, 400.0), (900.0, 400.0), (300.0, 700.0), (900.0, 700.0)] * 50
    x0, y0, x1, y1 = tr.suggest_crop(pts, (1920, 1080))
    assert x0 <= 300 and x1 >= 900 and y0 <= 400 and y1 >= 700
    assert (x1 - x0) / (y1 - y0) == pytest.approx(4 / 3, rel=0.01)
    assert x0 >= 0 and x1 <= 1920 and y0 >= 0 and y1 <= 1080
    assert tr.suggest_crop([], (1920, 1080)) == [0, 0, 1920, 1080]


def test_exports_list_says_which_can_train_and_why_not(tmp_path):
    _export(tmp_path, "good")
    _export(tmp_path, "old", gated=False)
    _export(tmp_path, "other_tcp", link="panda_hand")
    rows = {row["name"]: row for row in _manager(tmp_path).exports()}
    assert rows["good"]["usable"] and rows["good"]["qcGated"] and rows["good"]["hands"] == ["right"]
    assert not rows["old"]["usable"] and "before the QC gate" in rows["old"]["problems"][0]
    assert not rows["other_tcp"]["usable"] and "panda_hand" in rows["other_tcp"]["problems"][0]


def test_start_refuses_bad_exports_and_local_training_while_the_recorder_is_connected(tmp_path):
    _export(tmp_path, "good")
    _export(tmp_path, "old", gated=False)
    launched = []
    manager = _manager(tmp_path, recorder_busy=lambda: True,
                       popen=lambda cmd, **kw: launched.append(cmd) or _FakeProc(999999))
    params = {"exports": ["old"], "hand": "right", "cameras": [{"name": "cam_10"}], "hostId": "local"}
    with pytest.raises(tr.TrainingError, match="QC gate"):
        manager.start(params)
    with pytest.raises(tr.TrainingError, match="recorder is connected"):
        manager.start({**params, "exports": ["good"]})
    assert launched == []

    sandbox = _manager(tmp_path, sandbox="hph")
    with pytest.raises(tr.TrainingError, match="sandbox"):
        sandbox.start({**params, "exports": ["good"]})


def test_a_local_run_follows_its_log_and_finishes_with_the_reload_check(tmp_path):
    _export(tmp_path, "good")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL)
    launched = []
    manager = _manager(tmp_path, popen=lambda cmd, **kw: launched.append((cmd, kw)) or _FakeProc(sleeper.pid))
    try:
        rec = manager.start({
            "exports": ["good"], "hand": "right", "hostId": "local", "jobName": "run1", "steps": 100,
            "cameras": [{"name": "cam_10"}, {"name": "cam_06", "crop": [140, 220, 1180, 1000]}],
        })
        for _ in range(100):
            if rec.state == "running":
                break
            threading.Event().wait(0.02)
        cmd, kw = launched[0]
        assert cmd[1] == tr.TRAIN_SCRIPT and kw["start_new_session"] and kw["stdin"] is subprocess.DEVNULL
        assert "cam_06:140,220,1180,1000" in cmd and "--hand" in cmd
        # The trainer reports through its log; the job name is in the sleeper's cmdline only in real runs,
        # so the pid check reads "not this job" and the run is finished from the log.
        with open(rec.logPath, "a", encoding="utf-8") as log:
            log.write('FR3_PROGRESS {"event": "train", "steps": 50, "loss": 0.5}\n')
            log.write('FR3_PROGRESS {"event": "reload_check", "ok": true, "max_abs_diff": 0.0}\n')
            log.write('FR3_PROGRESS {"event": "checkpoint", "step": 100, "path": "/x"}\n')
            log.write('FR3_PROGRESS {"event": "done"}\n')
        status = manager.status()
        run = status["runs"][0]
        assert run["state"] == "done" and run["step"] == 100 and run["loss"] == 0.5
        assert "reloads and predicts identically" in run["message"]
        assert not any('"event": "train"' in line for line in manager.log_tail("run1"))
        # survives a gateway restart
        again = _manager(tmp_path)
        assert again.runs["run1"].state == "done"
        with pytest.raises(tr.TrainingError, match="exists"):
            manager.start({"exports": ["good"], "hand": "right", "jobName": "run1", "cameras": [{"name": "cam_10"}]})
    finally:
        sleeper.kill()
        sleeper.wait()


def test_remote_host_and_directory_cannot_carry_shell_text(tmp_path):
    manager = _manager(tmp_path)
    with pytest.raises(tr.TrainingError):
        manager.add_host("x", "hph@host; rm -rf /", "/home/hph/train", "")
    with pytest.raises(tr.TrainingError):
        manager.add_host("x", "hph@host", "/home/hph/train $(id)", "")
    with pytest.raises(tr.TrainingError):
        manager.add_host("x", "hph@host", "/home", "")
    host = manager.add_host("5090", "hph@10.0.0.2", "/home/hph/lerobot_train/", "")
    assert host.repoDir == "/home/hph/lerobot_train" and host.python == ".venv/bin/python"
    assert [h.id for h in manager.hosts()] == ["local", host.id]


def test_training_endpoints_answer_over_http_outside_the_gateway_lock(tmp_path):
    _export(tmp_path, "good")
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    state.training = _manager(tmp_path)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with state.lock:  # a held gateway lock must not block the training page
            with urllib.request.urlopen(f"{base}/api/training/exports", timeout=5) as answered:
                assert [e["name"] for e in json.loads(answered.read())["exports"]] == ["good"]
            request = urllib.request.Request(
                f"{base}/api/training/check", method="POST",
                data=json.dumps({"exports": ["good"], "hand": "left"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as answered:
                body = json.loads(answered.read())
            check = body["check"]
            assert body["ok"] is True and not check["ok"] and "no tracked left hand" in check["problems"][0]
        request = urllib.request.Request(f"{base}/api/training/stop", method="POST", data=b'{"job": "nope"}',
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_connect_is_refused_on_a_thor_while_a_local_run_trains(tmp_path, monkeypatch):
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "local_run_active", lambda: True)
    state.training = manager
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/handheld/record/connect", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 409 and b"training run" in err.value.read()
    finally:
        server.shutdown()
        server.server_close()
