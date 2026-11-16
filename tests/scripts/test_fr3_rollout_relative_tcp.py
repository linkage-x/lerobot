# ruff: noqa: N806  -- T: a stack of transforms
from __future__ import annotations

import json
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
    assert [s["status"] for s in manager.status()["stages"]][-1] == "not delivered"


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
