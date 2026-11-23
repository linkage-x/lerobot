# ruff: noqa: N806  -- T: a stack of transforms
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tools.data_collection_gui import gateway, rollout_runs as rr
from tools.fr3 import relative_tcp as rt, rollout_relative_tcp as ro


def _contract(wrench: bool = False) -> rt.PolicyContract:
    cam = rt.CameraInput("cam_10", 1920, 1080, [0, 0, 1920, 1080], 320, 180)
    return rt.PolicyContract(
        hand="right", fps=60, frame_stride=2, horizon=4, history=2, cameras=[cam],
        width_name="box_gripper.distance_m",
        wrench_names=[f"box_six_d_external.{a}" for a in rt.WRENCH_AXES] if wrench else [],
    )


def _live(contract: rt.PolicyContract, box_id: str = "3100715645") -> ro.LiveInputs:
    live = ro.LiveInputs.__new__(ro.LiveInputs)  # no frame bus in a test
    live.contract, live.box_id, live.gateway = contract, box_id, "http://x"
    return live


def test_live_width_and_wrench_come_from_the_arms_box_in_the_recorders_layout():
    snapshot = {
        "sensors": {
            "box3906437858/box_gripper": {"distance_m": 0.07},  # the hand-held BOX: not this arm's
            "box3100715645/box_gripper": {"distance_m": 0.031},
            "box3100715645/box_six_d_force": {"fxyz_mxyz": [9] * 6, "fxyz_mxyz_external": [0.5, 0, -1.0, 0, 0, 0.01]},
        },
        "status": {},
    }
    width, wrench, _ = _live(_contract(wrench=True)).box_from_snapshot(snapshot)
    assert width == pytest.approx(0.031)
    np.testing.assert_allclose(wrench, [0.5, 0, -1.0, 0, 0, 0.01])

    with pytest.raises(RuntimeError, match="not in the live data"):
        _live(_contract(), box_id="42").box_from_snapshot(snapshot)
    no_external = {"sensors": {"box3100715645/box_gripper": {"distance_m": 0.03},
                               "box3100715645/box_six_d_force": {"external_note": "这次连接后还没做动态校准"}}}
    with pytest.raises(RuntimeError, match="还没做动态校准"):
        _live(_contract(wrench=True)).box_from_snapshot(no_external)
    single = {"sensors": {"box_gripper": {"distance_m": 0.02}}, "status": {"device_id": "3906437858"}}
    with pytest.raises(RuntimeError, match="not the arm's gripper BOX"):
        _live(_contract()).box_from_snapshot(single)


def test_a_gripper_box_that_dropped_out_is_not_read_from_its_last_values():
    """11-19: an unplugged gripper BOX keeps its last values in BOX_LIVE, which still comes at 10 Hz."""

    def snap(gripper_age, six_d_age=0.01, seen=True):
        row = lambda age: {"seen": seen, "fresh": age <= 1.0, "last_seen_age_s": age}  # noqa: E731
        return {
            "sensors": {"box3100715645/box_gripper": {"distance_m": 0.031},
                        "box3100715645/box_six_d_force": {"fxyz_mxyz_external": [0.5, 0, -1.0, 0, 0, 0.01]},
                        "box3906437858/box_gripper": {"distance_m": 0.07}},
            "status": {"sensor_status": {"box3100715645/box_gripper": row(gripper_age),
                                         "box3100715645/box_six_d_force": row(six_d_age),
                                         "box3906437858/box_gripper": row(0.0)}},
        }

    width, _, meta = _live(_contract(wrench=True)).box_from_snapshot(snap(0.02))
    assert width == pytest.approx(0.031) and meta["dataAgeS"] == pytest.approx(0.02)
    with pytest.raises(RuntimeError, match=r"3100715645 is not streaming \(box3100715645/box_gripper: last sample 4.0 s"):
        _live(_contract()).box_from_snapshot(snap(4.0))
    with pytest.raises(RuntimeError, match="box_six_d_force: last sample 0.8 s"):
        _live(_contract(wrench=True)).box_from_snapshot(snap(0.02, six_d_age=0.8))
    _live(_contract()).box_from_snapshot(snap(0.02, six_d_age=0.8))  # a policy without the wrench does not need it
    with pytest.raises(RuntimeError, match="sent nothing since Connect"):
        _live(_contract()).box_from_snapshot(snap(0.0, seen=False))


def test_chunk_errors_measure_position_rotation_width_and_skip_padding():
    T = np.repeat(np.eye(4)[None], 3, axis=0)
    gt = np.concatenate([rt.encode(T), np.full((3, 1), 0.04)], axis=1)
    pred = gt.copy()
    pred[:, 0] += [0.001, 0.002, 0.003]
    pred[:, rt.POSE_DIM] += 0.001
    e = ro.chunk_errors(pred, gt, np.array([False, False, True]))
    np.testing.assert_allclose(e["pos_mm"][:2], [1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(e["width_mm"][:2], [1.0, 1.0], atol=1e-6)
    assert np.isnan(e["pos_mm"][2]) and e["rot_deg"][0] == pytest.approx(0.0, abs=1e-6)


def test_the_plan_check_uses_the_point_the_labels_are_of():
    """The arm's FK goes to corenetic_socket_ee: the labels' point, with corenetic_gripper_ee's axes."""
    from tools.fr3 import fr3_kinematics as kin

    a = kin.urdf_fixed_transform(rt.TCP_LINK)
    b = kin.urdf_fixed_transform(rt.ARM_FK_FRAME)
    np.testing.assert_allclose(a[:3, :3], b[:3, :3], atol=1e-12)
    assert np.linalg.norm(b[:3, 3] - a[:3, 3]) == pytest.approx(0.0335, abs=0.001)


def _manager(tmp_path: Path, **kw) -> rr.RolloutManager:
    repo = tmp_path / "repo"
    model = repo / "outputs" / "training" / "runs" / "run1" / "lerobot" / "checkpoints" / "000100" / "pretrained_model"
    model.mkdir(parents=True)
    _contract().write(model)
    (model / "fr3_provenance.json").write_text(json.dumps({"exports": [{"name": "e1"}], "reload_check": {"ok": True}}))
    (tmp_path / "exports" / "e1" / "meta").mkdir(parents=True)
    (tmp_path / "exports" / "e1" / "meta" / "info.json").write_text("{}")
    return rr.RolloutManager(repo, exports_root=tmp_path / "exports", python=sys.executable, **kw)


def test_the_preview_plan_check_follows_the_arm_when_a_person_moves_it():
    now = [100.0]
    reads = iter([np.full(7, 0.2), RuntimeError("connection refused")])

    def read():
        r = next(reads)
        if isinstance(r, Exception):
            raise r
        return r

    watcher = ro.JointWatcher(read, np.zeros(7), clock=lambda: now[0])  # never started: polled by hand
    q, info = ro.joints_for_check(watcher, None)
    assert np.allclose(q, 0.0) and info["age_s"] == 0.0 and info["source"] == "read"
    now[0] += 1.0
    watcher.poll()  # the arm was moved: the next check judges the new pose
    q, info = ro.joints_for_check(watcher, None)
    assert np.allclose(q, 0.2) and info["age_s"] == 0.0 and info["error"] == ""
    now[0] += 1.0
    watcher.poll()  # a failed read keeps the last joints while they are recent, and says why
    q, info = ro.joints_for_check(watcher, None)
    assert np.allclose(q, 0.2) and "connection refused" in info["error"]
    now[0] += ro.JOINT_STALE_S
    q, info = ro.joints_for_check(watcher, None)  # too old to vouch for the pose the arm stands in
    assert q is None and info["age_s"] > ro.JOINT_STALE_S
    q, info = ro.joints_for_check(None, np.ones(7))  # joints given on the command line stay as given
    assert np.allclose(q, 1.0) and info["source"] == "given"
    assert ro.joints_for_check(None, None) == (None, None)


def test_the_joint_watcher_reads_until_stopped():
    seen = threading.Event()
    calls = []

    def read():
        calls.append(1)
        seen.set()
        return np.full(7, len(calls), dtype=float)

    watcher = ro.JointWatcher(read, np.zeros(7), refresh_s=0.01).start()
    assert seen.wait(2.0)
    watcher.stop()
    n = len(calls)
    assert n >= 1 and not watcher._thread.is_alive()
    assert watcher.latest()[0][0] == n


def test_reading_the_joints_does_not_ask_the_arm_for_its_model(monkeypatch, capsys):
    """11-19: Panda() downloads the robot model on every connect; read about once a second, the arm refused
    that command for 20-30 s and the preview had no joints for its plan check."""
    import types

    from tools.fr3 import fr3_print_joints_runtime as pj

    calls = []

    class Robot:
        def __init__(self, ip):
            calls.append(("connect", ip))

        def read_once(self):
            calls.append(("read_once",))
            return types.SimpleNamespace(q=[0.1, -0.2, 0.3, -2.0, 0.0, 1.5, 0.7])

        def load_model(self):
            raise AssertionError("the joints need no robot model")

    fake = types.ModuleType("panda_py")
    fake.libfranka = types.SimpleNamespace(Robot=Robot)
    fake.Panda = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Panda() loads the robot model"))
    monkeypatch.setitem(sys.modules, "panda_py", fake)
    assert pj.main(["--robot-ip", "10.0.0.2"]) == 0
    assert calls == [("connect", "10.0.0.2"), ("read_once",)]
    assert "Joint angles (rad): [0.1, -0.2, 0.3, -2.0, 0.0, 1.5, 0.7]" in capsys.readouterr().out


def test_preview_is_refused_without_the_recorder_in_a_sandbox_or_off_thor(tmp_path):
    manager = _manager(tmp_path)
    ckpt = manager.checkpoints()[0]
    assert ckpt["run"] == "run1" and ckpt["step"] == 100 and ckpt["exports"] == ["e1"]
    with pytest.raises(rr.RolloutError, match="connect the recorder"):
        manager.start_preview(ckpt["path"], "right", 2.0, False)
    sandbox = _manager(tmp_path / "s", sandbox="hph", recorder_busy=lambda: True)
    with pytest.raises(rr.RolloutError, match="sandbox"):
        sandbox.start_preview(sandbox.checkpoints()[0]["path"], "right", 2, False)
    workstation = _manager(tmp_path / "w", profile="workstation", recorder_busy=lambda: True)
    with pytest.raises(rr.RolloutError, match="not a Thor"):
        workstation.start_preview(workstation.checkpoints()[0]["path"], "right", 2, False)
    with pytest.raises(rr.RolloutError, match="not a relative-TCP checkpoint"):
        manager.start_offline(str(tmp_path), "e1", [], 15)
    assert [s["status"] for s in manager.status()["stages"]][-1] == "available"


def test_one_job_at_a_time_and_its_end_is_read_from_its_log(tmp_path):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL)

    class Proc:
        pid = sleeper.pid

    manager = _manager(tmp_path, popen=lambda *a, **k: Proc())
    try:
        job = manager.start_offline(manager.checkpoints()[0]["path"], "e1", [0, 2], 30)
        # the sleeper's cmdline does not name the script, so the job reads as exited and finishes from its log
        Path(job.logPath).write_text('FR3_ROLLOUT {"event": "done", "frames": 3}\n', encoding="utf-8")
        status = manager.status()
        assert status["jobs"][0]["state"] == "done"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_rollout_endpoints_answer_outside_the_gateway_lock(tmp_path):
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    state.rollout = _manager(tmp_path)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with state.lock, urllib.request.urlopen(f"{base}/api/rollout/checkpoints", timeout=5) as answered:
            assert json.loads(answered.read())["checkpoints"][0]["run"] == "run1"
        request = urllib.request.Request(
            f"{base}/api/rollout/preview", method="POST", data=json.dumps({"checkpoint": "x", "arm": "right"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def _replayed_preview(manager: rr.RolloutManager, ok: bool = True) -> None:
    record = manager.root / "records" / "preview_1"
    record.mkdir(parents=True)
    (record / "record.json").write_text(json.dumps({
        "kind": "preview", "checkpoint": manager.checkpoints()[0]["path"],
        "replay_check": {"ok": ok, "max_abs_diff": 0.0, "steps": 3},
    }))


def test_execute_is_refused_unless_every_condition_holds(tmp_path):
    spawned = []

    class Proc:
        pid = 2**22 + 7  # a real integer, never a MagicMock (os.killpg)

    holder = {"who": ""}
    manager = _manager(tmp_path, recorder_busy=lambda: True, fci_holder=lambda: holder["who"],
                       popen=lambda cmd, **k: spawned.append(cmd) or Proc())
    ckpt = manager.checkpoints()[0]["path"]
    with pytest.raises(rr.RolloutError, match="e-stop"):
        manager.start_execute(ckpt, "right", 1, attended=False)
    with pytest.raises(rr.RolloutError, match="Replay check"):
        manager.start_execute(ckpt, "right", 1, attended=True)
    _replayed_preview(manager)
    assert "unlock" not in manager.status()  # 11-18: no maintainer unlock file; the page's dialog confirms
    with pytest.raises(rr.RolloutError, match="stage"):
        manager.start_execute(ckpt, "right", 4, attended=True)
    holder["who"] = "真机 replay 正在运行"
    with pytest.raises(rr.RolloutError, match="holds the arm's FCI"):
        manager.start_execute(ckpt, "right", 1, attended=True)
    holder["who"] = ""
    for kw in ({"sandbox": "hph"}, {"profile": "workstation"}):
        other = _manager(tmp_path / next(iter(kw)), recorder_busy=lambda: True, **kw)
        _replayed_preview(other)
        with pytest.raises(rr.RolloutError, match="sandbox|not a Thor"):
            other.start_execute(other.checkpoints()[0]["path"], "right", 1, attended=True)
    no_recorder = _manager(tmp_path / "nr")
    _replayed_preview(no_recorder)
    with pytest.raises(rr.RolloutError, match="connect the recorder"):
        no_recorder.start_execute(no_recorder.checkpoints()[0]["path"], "right", 1, attended=True)

    job = manager.start_execute(ckpt, "right", 2, attended=True)
    assert spawned[0][1:3] == [rr.SCRIPT, "execute"] and spawned[0][spawned[0].index("--stage") + 1] == "2"
    assert "--attended" in spawned[0]  # the script refuses without it
    assert not (manager.repo_root / "outputs" / "fr3_infer_real_unlock.json").exists()
    assert job.kind == "execute" and job.params["attended"] is True


def test_the_move_to_a_training_start_is_refused_unless_attended_and_free_and_holds_the_fci(tmp_path):
    spawned = []

    class Proc:
        pid = 2**22 + 11  # a real integer, never a MagicMock (os.killpg)

    holder = {"who": ""}
    manager = _manager(tmp_path, fci_holder=lambda: holder["who"], popen=lambda cmd, **k: spawned.append(cmd) or Proc())
    ckpt = manager.checkpoints()[0]["path"]
    with pytest.raises(rr.RolloutError, match="e-stop"):
        manager.start_to_start(ckpt, "right", attended=False)
    holder["who"] = "真机 replay 正在运行"
    with pytest.raises(rr.RolloutError, match="holds the arm's FCI"):
        manager.start_to_start(ckpt, "right", attended=True)
    holder["who"] = ""
    for kw in ({"sandbox": "hph"}, {"profile": "workstation"}):
        other = _manager(tmp_path / next(iter(kw)), **kw)
        with pytest.raises(rr.RolloutError, match="sandbox|not a Thor"):
            other.start_to_start(other.checkpoints()[0]["path"], "right", attended=True)
    job = manager.start_to_start(ckpt, "right", attended=True)  # no recorder, no replayed preview needed
    assert spawned[0][1:3] == [rr.SCRIPT, "to-start"] and "--attended" in spawned[0]
    assert job.kind == "tostart" and job.params == {"arm": "right", "attended": True}
    assert gateway._is_hardware_request("POST", "/api/rollout/to-start")  # a dev gateway needs the lease for it


def test_jobs_find_the_venvs_cmeel_libraries_for_panda_py(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    cmeel = venv / "lib" / "python3.12" / "site-packages" / "cmeel.prefix" / "lib"
    cmeel.mkdir(parents=True)
    env = rr.job_env(str(venv / "bin" / "python"), {"LD_LIBRARY_PATH": "/opt/x", "HOME": "/h"})
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(cmeel), "/opt/x"]
    assert env["PYTHONPATH"] == "src:." and env["HOME"] == "/h"
    assert "LD_LIBRARY_PATH" not in rr.job_env(str(tmp_path / "none" / "bin" / "python"), {})

    seen = {}

    class Proc:
        pid = 2**22 + 9  # a real integer, never a MagicMock (os.killpg)

    manager = _manager(tmp_path / "m", recorder_busy=lambda: True,
                       popen=lambda cmd, **k: seen.update(k) or Proc())
    manager.python = str(venv / "bin" / "python")
    _replayed_preview(manager)
    manager.start_execute(manager.checkpoints()[0]["path"], "right", 1, attended=True)
    assert str(cmeel) in seen["env"]["LD_LIBRARY_PATH"] and seen["stdin"] is subprocess.DEVNULL


def test_a_running_execution_holds_the_fci_for_the_other_arm_jobs(tmp_path):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", rr.SCRIPT], stdin=subprocess.DEVNULL)

    class Proc:
        pid = sleeper.pid

    manager = _manager(tmp_path, recorder_busy=lambda: True, popen=lambda *a, **k: Proc())
    _replayed_preview(manager)
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    state.rollout = manager
    try:
        assert gateway._fr3_pivot_fci_holder(state) == ""
        manager.start_execute(manager.checkpoints()[0]["path"], "right", 1, attended=True)
        assert "Rollout 执行" in gateway._fr3_pivot_fci_holder(state)
        with pytest.raises(rr.RolloutError, match="still running"):
            manager.start_preview(manager.checkpoints()[0]["path"], "right", 2.0, False)
    finally:
        sleeper.kill()
        sleeper.wait()
    assert gateway._fr3_pivot_fci_holder(state) == ""


def test_execute_endpoint_answers_outside_the_gateway_lock(tmp_path):
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    state.rollout = _manager(tmp_path)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with state.lock:
            request = urllib.request.Request(
                f"{base}/api/rollout/execute", method="POST",
                data=json.dumps({"checkpoint": state.rollout.checkpoints()[0]["path"], "arm": "right", "stage": 1,
                                 "attended": "yes"}).encode(),  # only a literal true confirms
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 400 and "e-stop" in json.loads(err.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_to_start_endpoint_answers_outside_the_gateway_lock_and_wants_a_literal_confirmation(tmp_path):
    state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
    state.rollout = _manager(tmp_path)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with state.lock:
            request = urllib.request.Request(
                f"{base}/api/rollout/to-start", method="POST",
                data=json.dumps({"checkpoint": state.rollout.checkpoints()[0]["path"], "arm": "right",
                                 "attended": 1}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 400 and "e-stop" in json.loads(err.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_connected_for_inference_the_jobs_read_the_main_bus_if_it_has_the_checkpoints_cameras(tmp_path):
    spawned = []

    class Proc:
        pid = 2**22 + 9  # a real integer, never a MagicMock (os.killpg)

    cameras = {"open": ["cam_06"]}
    manager = _manager(tmp_path, recorder_busy=lambda: True, recorder_cameras=lambda: cameras["open"],
                       popen=lambda cmd, **k: spawned.append(cmd) or Proc())
    ckpt = manager.checkpoints()[0]["path"]  # reads cam_10
    with pytest.raises(rr.RolloutError, match="connected for inference with cam_06 only; this checkpoint reads cam_10"):
        manager.start_preview(ckpt, "right", 2.0, False)
    cameras["open"] = ["cam_10"]
    assert manager.status()["recorderCameras"] == ["cam_10"]
    with pytest.raises(rr.RolloutError, match=r"0.2..10 Hz"):
        manager.start_preview(ckpt, "right", 12.0, False)
    job = manager.start_preview(ckpt, "right", 8.0, False)
    cmd = spawned[-1]
    assert cmd[cmd.index("--frame-bus") + 1] == rr.INFERENCE_FRAME_BUS and job.params["frameBus"].startswith("inference bus")
    manager.jobs[job.id].state = "done"

    _replayed_preview(manager)
    job = manager.start_execute(ckpt, "right", 1, attended=True)
    assert spawned[-1][spawned[-1].index("--frame-bus") + 1] == rr.INFERENCE_FRAME_BUS
    manager.jobs[job.id].state = "done"

    # a full session: the preview bus, the script's default, and its 5 Hz
    cameras["open"] = []
    with pytest.raises(rr.RolloutError, match=r"0.2..5 Hz"):
        manager.start_preview(ckpt, "right", 8.0, False)
    job = manager.start_preview(ckpt, "right", 2.0, False)
    assert "--frame-bus" not in spawned[-1] and job.params["frameBus"].startswith("preview bus")


def test_the_inference_bus_is_the_same_on_both_sides_and_tightens_the_frame_age():
    assert Path(rr.INFERENCE_FRAME_BUS) == ro.INFERENCE_FRAME_BUS
    assert rr.INFERENCE_FRAME_EVERY_N == ro.INFERENCE_FRAME_EVERY_N == 2
    assert ro.frame_age_limit_s(ro.INFERENCE_FRAME_BUS) == pytest.approx(0.15)
    assert ro.frame_age_limit_s(ro.PREVIEW_FRAME_BUS) == pytest.approx(0.40)
    assert not ro.LiveInputs(_contract(), gateway="http://x", box_id="1", frame_bus=ro.INFERENCE_FRAME_BUS).demand_previews
    assert ro.LiveInputs(_contract(), gateway="http://x", box_id="1").demand_previews


def test_live_jobs_wait_until_the_recorder_has_brought_every_sensor_up(tmp_path):
    spawned = []

    class Proc:
        pid = 2**22 + 11  # a real integer, never a MagicMock (os.killpg)

    rec = {"state": "connecting", "sensors": [{"id": "cam_10", "kind": "camera", "state": "running"},
                                              {"id": "box/gripper", "kind": "box_collection", "state": "warning"}]}
    manager = _manager(tmp_path, recorder_busy=lambda: True, recorder_cameras=lambda: ["cam_10"],
                       recorder_sensors=lambda: rec, popen=lambda cmd, **k: spawned.append(cmd) or Proc())
    ckpt = manager.checkpoints()[0]["path"]
    readiness = manager.status()["recorder"]
    assert readiness["pending"] and not readiness["ready"] and not readiness["failed"]
    assert not readiness["stuck"]  # no elapsed time reported: never called stuck
    rec["elapsedS"] = rr.CONNECT_STUCK_S - 1
    assert not manager.status()["recorder"]["stuck"]
    rec["elapsedS"] = rr.CONNECT_STUCK_S + 1  # hung without failing: the page may cancel it
    assert manager.status()["recorder"]["stuck"] and manager.status()["recorder"]["pending"]
    with pytest.raises(rr.RolloutError, match=r"still connecting \(not ready yet: box/gripper\)"):
        manager.start_preview(ckpt, "right", 2.0, False)
    assert not spawned
    rec["sensors"][1]["state"] = "error"  # a failed connect is not pending: Disconnect may tear it down
    assert manager.status()["recorder"]["failed"] and not manager.status()["recorder"]["pending"]
    rec["state"], rec["sensors"][1]["state"] = "armed", "running"
    assert manager.status()["recorder"]["ready"]
    manager.start_preview(ckpt, "right", 2.0, False)
    assert spawned


#: 11-19 18:08 (execute_20261119_180851): the second chunk's anchor, from which the policy's descent took j2
#: through 0. The synthetic path below drives j2 from there to +0.25.
_Q_SINGULAR_ANCHOR = np.array([0.736, -0.443, -0.946, -2.473, -0.618, 3.065, 1.056])


def _through_the_shoulder_singularity():
    from tools.fr3 import fr3_kinematics as kin

    qa = _Q_SINGULAR_ANCHOR
    qb = qa.copy()
    qb[1] = 0.25
    F = kin.NOMINAL.fk(qa + np.linspace(0, 1, 33)[1:, None] * (qb - qa))
    q, ok = kin.ik_retry(kin.NOMINAL, F, qa, early_exit=True, rest=qa)
    return kin, qa, F, q, ok


def test_a_chunk_toward_a_singularity_is_steered_off_it_on_the_same_tcp_path():
    from tools.fr3 import rollout_execute as rx

    kin, qa, F, q, ok = _through_the_shoulder_singularity()
    assert ok.all()
    q_s, ok_s, sig = ro.steer_off_singularity(F, q, ok, qa)
    assert ok_s.all()
    np.testing.assert_allclose(sig, ro.sigma_min(q_s))
    assert np.abs(kin.NOMINAL.fk(q_s)[:, :3, 3] - F[:, :3, 3]).max() < 1e-5  # the policy's path, only the posture differs
    assert sig.min() > ro.sigma_min(q).min() + 0.01
    assert sig[-1] > ro.sigma_min(q)[-1] + 0.05
    Q = np.concatenate([qa[None], q_s])
    # the null-space motion is no re-posture and no turn of its own: every step stays executable
    assert rx.reposture(Q).max() <= rx.Limits().reposture_rad
    T = kin.NOMINAL.fk(Q)
    steps_ok, why = rx.assess(T, Q, ok_s, [""] * len(ok_s), rx.Limits(step_mm=40.0))
    assert steps_ok.all(), next(w for w in why if w)


def test_a_chunk_clear_of_singularities_is_left_as_planned():
    from tools.fr3 import fr3_kinematics as kin

    qa = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # the ready posture: sigma_min ~0.2
    assert ro.sigma_min(qa[None])[0] > ro.STEER_SIGMA
    F = kin.NOMINAL.fk(qa + np.linspace(0, 0.05, 8)[:, None] * np.array([1, 0, 0, 0, 0, 0, 0]))
    q, ok = kin.ik_retry(kin.NOMINAL, F, qa, early_exit=True, rest=qa)
    q_s, ok_s, _ = ro.steer_off_singularity(F, q, ok, qa)
    assert ok_s.all()
    np.testing.assert_array_equal(q_s, q)


def test_a_step_the_steering_cannot_reach_ends_the_chunk_there():
    kin, qa, F, q, ok = _through_the_shoulder_singularity()
    F = F.copy()
    F[20, :3, 3] += [2.0, 0.0, 0.0]  # 2 m away: out of reach
    ok[20] = False
    _, ok_s, _ = ro.steer_off_singularity(F, q, ok, qa)
    assert ok_s[:20].all() and not ok_s[20:].any()


def test_a_move_to_a_training_start_is_not_replayed(tmp_path):
    # 11-23: Replay check on a to-start record died on KeyError: 'contract' (no policy ran in it)
    spawned = []

    class Proc:
        pid = 2**22 + 13  # a real integer, never a MagicMock (os.killpg)

    manager = _manager(tmp_path, popen=lambda cmd, **k: spawned.append(cmd) or Proc())
    record = manager.root / "records" / "tostart_1"
    record.mkdir(parents=True)
    (record / "record.json").write_text(json.dumps({"kind": "to_start", "checkpoint": manager.checkpoints()[0]["path"]}))
    with pytest.raises(rr.RolloutError, match="nothing to replay"):
        manager.start_replay(str(record))
    assert not spawned
    with pytest.raises(RuntimeError, match="nothing to replay"):
        ro.replay(record)
    _replayed_preview(manager)
    manager.start_replay(str(manager.root / "records" / "preview_1"))
    assert spawned and spawned[0][1:3] == [rr.SCRIPT, "replay"]


def test_records_are_listed_newest_first_whatever_their_kind(tmp_path):
    manager = _manager(tmp_path)
    for name, kind, created in (("tostart_a", "to_start", "2026-11-23T08:00:00+00:00"),
                                ("preview_b", "preview", "2026-11-23T08:30:00+00:00"),
                                ("tostart_c", "to_start", "2026-11-23T08:40:00+00:00")):
        (manager.root / "records" / name).mkdir(parents=True)
        (manager.root / "records" / name / "record.json").write_text(json.dumps({"kind": kind, "created": created}))
    assert [Path(r["path"]).name for r in manager.results()] == ["tostart_c", "preview_b", "tostart_a"]
