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


def test_suggest_crop_boxes_the_rig_with_margin_square_inside_the_image():
    pts = [(300.0, 400.0), (900.0, 400.0), (300.0, 700.0), (900.0, 700.0)] * 50
    x0, y0, x1, y1 = tr.suggest_crop(pts, (1920, 1080))
    assert x0 <= 300 and x1 >= 900 and y0 <= 400 and y1 >= 700
    assert (x1 - x0) / (y1 - y0) == pytest.approx(1.0, rel=0.01)
    assert x0 >= 0 and x1 <= 1920 and y0 >= 0 and y1 <= 1080
    x0, y0, x1, y1 = tr.suggest_crop(pts, (1920, 1080), aspect=4 / 3)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(4 / 3, rel=0.01)
    assert tr.suggest_crop([], (1920, 1080)) == [0, 0, 1920, 1080]


def test_suggest_crop_keeps_what_the_fingertip_handled_below_the_rig():
    rig = [(500.0, 350.0), (900.0, 350.0), (500.0, 700.0), (900.0, 700.0)] * 50
    # the fingertip reaches down to the table, an object of 100 px radius around it
    targets = [(700.0, v, 100.0) for v in range(500, 960, 10)]
    old = tr.suggest_crop(rig, (1920, 1080), margin=0.2, aspect=4 / 3)
    assert tr.crop_target_coverage(old, targets, size=(1920, 1080))["inside"] < 0.5
    crop = tr.suggest_crop(rig, (1920, 1080), targets=targets)
    assert crop[2] - crop[0] == pytest.approx(crop[3] - crop[1], abs=2)
    assert tr.crop_target_coverage(crop, targets, size=(1920, 1080))["inside"] == 1.0


def test_suggest_crop_gives_up_the_square_never_the_targets():
    targets = [(u, 540.0, 50.0) for u in range(200, 1700, 20)]  # 1600 px wide, the image is 1080 high
    crop = tr.suggest_crop([], (1920, 1080), targets=targets)
    assert crop[2] - crop[0] > 1500 and crop[3] - crop[1] <= 1080
    cov = tr.crop_target_coverage(crop, targets, [0] * len(targets), size=(1920, 1080))
    assert cov["inside"] >= 0.98 and cov["worstEpisode"] == 0


def test_crop_coverage_does_not_blame_the_crop_for_what_is_outside_the_image():
    target = [(500.0, 1050.0, 100.0)]  # half of it below the image
    assert tr.crop_target_coverage([0, 0, 1920, 1080], target)["inside"] == 0.0
    assert tr.crop_target_coverage([0, 0, 1920, 1080], target, size=(1920, 1080))["inside"] == 1.0


@pytest.mark.parametrize("model", ["opencv_fisheye", "opencv"])
def test_projection_matches_opencv(model):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    kmat = np.array([[993.0, 0.0, 966.0], [0.0, 993.2, 561.4], [0.0, 0.0, 1.0]])
    pts = np.random.default_rng(0).uniform([-0.6, -0.4, 0.4], [0.6, 0.4, 1.6], (500, 3))
    if model == "opencv_fisheye":
        dist = np.array([-0.068, -0.021, 0.024, -0.011])
        ref = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), kmat, dist.reshape(4, 1))[0]
    else:
        dist = np.array([-0.1, 0.02, 0.001, -0.002, 0.003])
        ref = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), kmat, dist)[0]
    assert np.abs(tr.project_camera_points(pts, kmat, dist, model) - ref.reshape(-1, 2)).max() < 1e-6


def test_crop_suggestion_projects_the_fingertip_with_the_labels_intrinsics(tmp_path):
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "exports" / "e1"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pose = {"dtype": "float32", "shape": [7]}
    info = {
        "fps": 60,
        "features": {
            "observation.images.cam_06": {"dtype": "video", "shape": [1080, 1920, 3]},
            "observation.cube_pose.right.camera.cam_06": pose,
            "observation.cube_pose.right.world": pose,
            "observation.ee_pose.right.world": pose,
        },
        "label_provenance": {"key": {"intrinsics": ["calib_x_intrinsics"]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "export_sources.json").write_text(json.dumps({"episodes": []}))
    run = tmp_path / "repo" / "outputs" / "calibration" / "calib_x_intrinsics"
    (run / "converted" / "cam_06_S1").mkdir(parents=True)
    kmat = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    (run / "converted" / "cam_06_S1" / "intrinsics_producer.json").write_text(
        json.dumps({"camera_matrix": kmat, "dist_coeffs": [[0.0, 0.0, 0.0, 0.0]], "model": "opencv_fisheye"}))
    # written on another machine: the absolute path does not exist here
    (run / "summary.json").write_text(json.dumps({"cameras": [{
        "camera_name": "cam_06", "intrinsics_json": "/elsewhere/converted/cam_06_S1/intrinsics_producer.json"}]}))
    n = 51
    # world = camera; the rig 0.2 m above the fingertip, which moves along x at 1 m depth
    tcp = [[x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for x in np.linspace(-0.2, 0.2, n)]
    rig = [[x, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0] for x in np.linspace(-0.2, 0.2, n)]
    table = pa.table({
        "episode_index": [0] * n,
        "observation.cube_pose.right.camera.cam_06": rig,
        "observation.cube_pose.right.world": rig,
        "observation.ee_pose.right.world": tcp,
    })
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    m = _manager(tmp_path)
    targets, eps = m._fingertip_targets(root, info, "cam_06", "right")
    assert len(targets) == n and set(eps) == {0}
    u, v, r = targets[n // 2]
    assert (u, v) == pytest.approx((960.0, 540.0), abs=1e-6) and r == pytest.approx(50.0)
    result = m.crop_suggestion("e1", "cam_06", "right")
    assert result["targets"] == n and result["coverage"]["inside"] == 1.0
    x0, y0, x1, y1 = result["crop"]
    assert x1 - x0 == pytest.approx(y1 - y0, abs=2)
    assert "fingertip" in result["note"]


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
        assert cmd[:2] == ["bash", "-c"] and kw["start_new_session"] and kw["stdin"] is subprocess.DEVNULL
        assert tr.TRAIN_SCRIPT in cmd[2] and "cam_06:140,220,1180,1000" in cmd[2] and "--hand" in cmd[2]
        assert "--policy act" in cmd[2] and "accelerate" not in cmd[2] and "CUDA_VISIBLE_DEVICES" not in cmd[2]
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


def test_registry_hosts_are_listed_with_their_port_and_cannot_be_removed_from_the_page(tmp_path):
    manager = _manager(tmp_path)
    hosts_dir = manager.repo_root / tr.HOSTS_DIR
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "a100.json").write_text(json.dumps({
        "label": "a100", "user": "hph", "address": "10.0.0.9", "port": 2200, "hostname": "A100",
        "repo_dir": "/data/hph/fr3_train", "python": ".venv/bin/python", "pythonpath_prefix": "/home/hph/shim",
    }), encoding="utf-8")
    (hosts_dir / "bad.json").write_text(json.dumps({"user": "hph", "address": "x; id", "repo_dir": "/data/x"}))
    hosts = {h.id: h for h in manager.hosts()}
    assert set(hosts) == {"local", "a100"}
    assert hosts["a100"].port == 2200 and hosts["a100"].source == "registry"
    assert manager._pythonpath(hosts["a100"]) == "/home/hph/shim:src:."
    with pytest.raises(tr.TrainingError, match="training_hosts"):
        manager.remove_host("a100")
    page = manager.add_host("4060", "hph@10.0.0.3:2222", "/home/hph/fr3_train", "")
    assert page.port == 2222 and {h.id for h in manager.hosts()} == {"local", "a100", page.id}


def test_a_git_ignored_local_registry_file_keeps_its_plain_id(tmp_path):
    # A machine whose address must not be published (this repository is public) lives in
    # <id>.local.json, which .gitignore keeps out of git; runs started on it still say hostId <id>.
    manager = _manager(tmp_path)
    hosts_dir = manager.repo_root / tr.HOSTS_DIR
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "dc-a100.local.json").write_text(json.dumps({
        "label": "dc", "user": "hph", "address": "10.0.0.9", "port": 2200, "repo_dir": "/data/hph/fr3_train",
    }), encoding="utf-8")
    hosts = {h.id: h for h in manager.hosts()}
    assert hosts["dc-a100"].source == "registry" and hosts["dc-a100"].port == 2200
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "run/training_hosts/dc-a100.local.json"],
        cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
    )
    assert ignored.returncode == 0, ".gitignore must keep run/training_hosts/*.local.json out of git"


def test_several_gpus_prepare_once_then_train_under_accelerate(tmp_path):
    manager = _manager(tmp_path)
    host = tr.TrainingHost("a100", "a100", "remote", "hph@10.0.0.9", "/data/fr3_train", ".venv/bin/python",
                           pythonpathPrefix="/home/hph/shim")
    one = manager._shell(host, [tr.TRAIN_SCRIPT, "--job-name", "j1"], "j1", [3])
    assert "CUDA_VISIBLE_DEVICES=3" in one and "accelerate" not in one and "--prepare-only" not in one
    assert "PYTHONPATH=/home/hph/shim:src:." in one
    many = manager._shell(host, [tr.TRAIN_SCRIPT, "--job-name", "j1"], "j1", [0, 1, 2, 3])
    prepare, _, launch = many.partition("&&")
    assert "--prepare-only" in prepare and "accelerate" not in prepare
    assert "accelerate.commands.launch --num_processes 4" in launch
    assert "--start outputs/training/runs/j1" in launch and "CUDA_VISIBLE_DEVICES=0,1,2,3" in many
    resume = manager._shell(host, [tr.TRAIN_SCRIPT, "--resume", "outputs/training/runs/j1"], "j1", [0, 1])
    assert "--prepare-only" not in resume and "--num_processes 2" in resume and "--resume outputs/training/runs/j1" in resume


def test_pi05_needs_a_task_and_passes_its_settings(tmp_path):
    _export(tmp_path, "good")
    manager = _manager(tmp_path)
    host = manager.host("local")
    base = {"exports": ["good"], "hand": "right", "cameras": [{"name": "cam_10"}], "policy": "pi05"}
    with pytest.raises(tr.TrainingError, match="task"):
        manager._train_args(base, ["good"], "j", host)
    args = manager._train_args({**base, "task": "  pick   up the part ", "loraR": 16, "holdoutEpisodes": [4, 1]},
                               ["good"], "j", host)
    assert args[args.index("--policy") + 1] == "pi05" and args[args.index("--task") + 1] == "pick up the part"
    assert args[args.index("--lora-r") + 1] == "16" and args[args.index("--holdout-episodes") + 1] == "1,4"
    with pytest.raises(tr.TrainingError, match="policy"):
        manager._train_args({**base, "policy": "diffusion"}, ["good"], "j", host)
    assert "--history-dropout" not in args
    args = manager._train_args({**base, "task": "t", "historyDropout": 0.3}, ["good"], "j", host)
    assert args[args.index("--history-dropout") + 1] == "0.3"
    for bad in (1, -0.1, "x"):
        with pytest.raises(tr.TrainingError, match="history dropout"):
            manager._train_args({**base, "task": "t", "historyDropout": bad}, ["good"], "j", host)


def test_gpu_status_lists_every_gpu_with_whose_processes_run_on_it():
    text = (
        "0, GPU-a, NVIDIA A100-SXM4-80GB, 1200, 81920, 35\n"
        "1, GPU-b, NVIDIA A100-SXM4-80GB, 0, 81920, 0\n"
        "--apps--\n"
        "GPU-a, 4242, 1180\n"
        "--ps--\n"
        "  4242 hph      python tools/fr3/train_relative_tcp.py --job-name pi05_v1\n"
        "  1 root /sbin/init\n"
    )
    gpus = tr.parse_gpu_status(text)
    assert [g["index"] for g in gpus] == [0, 1] and gpus[0]["utilPct"] == 35 and gpus[1]["processes"] == []
    proc = gpus[0]["processes"][0]
    assert proc["pid"] == 4242 and proc["user"] == "hph" and "pi05_v1" in proc["command"] and proc["memMiB"] == 1180


def test_remote_poll_and_stop_work_where_dev_null_is_not_writable(tmp_path, monkeypatch):
    """On a100-19 /dev/null is a read-only file: a `2>/dev/null` made a live trainer look dead and Stop a no-op."""
    manager = _manager(tmp_path)
    hosts_dir = manager.repo_root / tr.HOSTS_DIR
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "a100.json").write_text(json.dumps({
        "label": "a100", "user": "hph", "address": "10.0.0.9", "hostname": "A100", "repo_dir": "/data/fr3_train",
        "python": ".venv/bin/python",
    }), encoding="utf-8")
    commands: list[str] = []

    def ssh(host, command, timeout=30.0):  # the remote shell, run here
        commands.append(command)
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)

    monkeypatch.setattr(manager, "_ssh", ssh)
    monkeypatch.setattr(manager, "_run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", ""))
    trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "remote_job"],
                               start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        assert isinstance(trainer.pid, int) and trainer.pid > 1
        log = tmp_path / "remote_job.log"
        log.write_text("", encoding="utf-8")
        rec = tr.RunRecord("remote_job", "a100", "a100", "remote", state="running", pid=trainer.pid, logPath=str(log))
        manager.runs[rec.job] = rec
        assert manager._poll_remote(rec) is False and rec.state == "running"
        assert "stopped" in manager.stop("remote_job")
        assert trainer.wait(timeout=10) is not None
    finally:
        if trainer.poll() is None:
            trainer.kill()
    assert commands and not any("/dev/null" in c for c in commands)


def test_pi05_training_is_refused_on_a_gpu_too_small_for_it(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    host = manager.host("local")
    small = {"state": "ok", "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4060", "memUsedMiB": 0,
                                      "memTotalMiB": 8188, "utilPct": 0, "processes": []}]}
    monkeypatch.setattr(manager, "gpu_status", lambda host_id, wait=False: small)
    manager._gpus[host.id] = small
    with pytest.raises(tr.TrainingError, match=r"pi05 cannot run on .*RTX 4060.*8\.8 GiB"):
        manager._gpus_param({"policy": "pi05", "gpus": [0]}, host)
    assert manager._gpus_param({"policy": "act", "gpus": [0]}, host) == [0]
