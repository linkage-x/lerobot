"""Calibration page: FR3 end-effector load and BOX 6D fit jobs (tools/data_collection_gui/fr3_payload_calibration.py)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tools.data_collection_gui import fr3_payload_calibration as pc

REPO = Path(__file__).resolve().parents[2]


def _host(tmp_path: Path, **kw) -> pc.Host:
    logs = []
    host = pc.Host(
        repo_root=tmp_path,
        python=lambda: Path(sys.executable),
        env=lambda: {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
        resolve_arm=kw.get(
            "resolve_arm",
            lambda arm: {"arm": arm or "right", "robot_ip": "192.168.1.210", "corenetic_device_id": 3100715645},
        ),
        arms=lambda: [{"arm": "right"}],
        fci_holder=kw.get("fci_holder", lambda: ""),
        box_port_holder=kw.get("box_port_holder", lambda: ""),
        log=lambda level, msg: logs.append((level, msg)),
    )
    return host


def _wait(cal: pc.PayloadCalibration, timeout=20.0) -> dict:
    deadline = time.monotonic() + timeout
    while cal.running() and time.monotonic() < deadline:
        time.sleep(0.05)
    return cal.payload()["job"]


def _fake_tool(cal: pc.PayloadCalibration, script_lines: list[str], code: int = 0):
    """Replace the subprocess with a python one-liner printing these lines."""
    body = "; ".join(f"print({line!r}, flush=True)" for line in script_lines) + f"; raise SystemExit({code})"
    cal._tool = lambda script, *args: [sys.executable, "-c", body]  # type: ignore[method-assign]


def test_collect_is_refused_until_a_check_with_the_table_passed(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    body = {"arm": "right", "poses": "payload+six_d", "toolConfig": "gripper", "sixD": True}
    assert "碰撞检查" in cal.collect(body)["error"]

    _fake_tool(cal, ["table (edge ...): lowest 260 mm", "check-poses=FAIL (...)"], code=1)
    cal.check({"arm": "right", "poses": "payload+six_d"})
    job = _wait(cal)
    assert job["state"] == "error" and not cal.checks["right|payload+six_d"]["pass"]
    assert "碰撞检查" in cal.collect(body)["error"]

    _fake_tool(cal, ["table (edge x > 0.200 m ...): 120 mm", "check-poses=PASS (...)"])
    cal.check({"arm": "right", "poses": "payload+six_d", "tableEdgeM": 0.22, "tableTopM": 0.12})
    job = _wait(cal)
    check = cal.checks["right|payload+six_d"]
    assert job["state"] == "done" and check["pass"] and check["table"]["edgeM"] == 0.22
    assert check["summary"] == ["table (edge x > 0.200 m ...): 120 mm"]
    # a pass for another pose set does not open this one
    assert "碰撞检查" in cal.collect({**body, "poses": "payload"})["error"]
    # an old pass does not either
    check["atUnixS"] -= pc.CHECK_VALID_S + 1
    assert "超过" in cal.collect(body)["error"]


def test_collect_refuses_while_someone_else_drives_the_arm_or_holds_the_box(tmp_path):
    holder = {"fci": "", "box": ""}
    cal = pc.PayloadCalibration(_host(tmp_path, fci_holder=lambda: holder["fci"], box_port_holder=lambda: holder["box"]))
    cal.checks["right|six_d"] = {"pass": True, "atUnixS": time.time()}
    body = {"arm": "right", "poses": "six_d", "toolConfig": "gripper", "sixD": True}
    holder["fci"] = "真机 replay 正在运行"
    assert "FCI" in cal.collect(body)["error"]
    holder["fci"], holder["box"] = "", "录制器已连接，它占着 BOX 端口"
    assert "BOX 端口" in cal.collect(body)["error"]
    # without 6D logging the BOX port does not matter
    _fake_tool(cal, ["  pose  0/13 up    : n=1", "  pose  0/13 down  : n=1", "  pose  1/13 up    : n=1"])
    payload = cal.collect({**body, "sixD": False})
    assert payload["ok"], payload["error"]
    assert cal.fci_user()  # its own collect now holds the FCI ...
    assert cal.box_port_user() == ""  # ... but not the BOX port
    job = _wait(cal)
    assert job["progress"] == {"done": 3, "total": 26}
    assert job["state"] == "error"  # the fake wrote no run file


def test_unknown_configs_and_pose_sets_are_refused(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    assert "末端配置" in cal.collect({"arm": "right", "poses": "six_d", "toolConfig": ""})["error"]
    assert "姿态集" in cal.collect({"arm": "right", "poses": "all", "toolConfig": "gripper"})["error"]
    assert "范围" in cal.check({"arm": "right", "poses": "six_d", "tableEdgeM": 3})["error"]


def _faces_log(path: Path, faces=("x+", "x-", "y+", "y-", "z+", "z-")) -> None:
    rng = np.random.default_rng(0)
    k = np.vstack([-0.8 * 9.80665 * np.eye(3), 0.05 * rng.standard_normal((3, 3))])
    rows = []
    for face in faces:
        axis, sign = "xyz".index(face[0]), 1.0 if face[1] == "+" else -1.0
        for _ in range(40):
            v = rng.normal(0, 0.08, 3)
            v[axis] = sign
            v /= np.linalg.norm(v)
            rows.append({"t": 0, "k": "box_six_d_force", "f": list(k @ v), "acc": list(v), "gyr": [0, 0, 0]})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_a_capture_is_fitted_and_registered_with_its_mount_time(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    cal._tool = lambda script, *args: [sys.executable, str(REPO / script), *args]  # type: ignore[method-assign]
    reg = tmp_path / pc.REGISTRY
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"schema": "box_devices/v1", "devices": {"3906437858": {"role": "handheld", "hand": "right", "six_d_payload": []}}}))
    d = tmp_path / pc.CAPTURE_SUBDIR / "20261109_100000_box3906437858"
    d.mkdir(parents=True)
    (d / "capture.json").write_text(json.dumps({"device_id": 3906437858, "end_effector": "", "started_unix": 1.8e9}))
    _faces_log(d / "free_air.log")
    payload = cal.refit({"source": "capture:20261109_100000_box3906437858"})
    cap = payload["captures"][0]
    assert cap["fit"]["meetsTarget"] and cap["fit"]["missingAttitudes"] == []

    src = "capture:20261109_100000_box3906437858"
    assert "末端配置" in cal.register({"source": src, "validFromUnix": 1.79e9})["error"]
    assert "装上时间" in cal.register({"source": src, "endEffector": "gripper + rig v9b"})["error"]
    out = cal.register({"source": src, "endEffector": "gripper + rig v9b", "validFromUnix": 1.7935e9})
    assert out["ok"], out.get("output")
    box = next(b for b in out["boxes"] if b["deviceId"] == "3906437858")
    assert box["records"][0]["endEffector"] == "gripper + rig v9b" and box["records"][0]["fitted"]
    assert out["captures"][0]["registered"]["valid_from_unix"] == 1793500000


def test_a_fit_with_missing_faces_is_shown_as_refused(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    cal._tool = lambda script, *args: [sys.executable, str(REPO / script), *args]  # type: ignore[method-assign]
    d = tmp_path / pc.CAPTURE_SUBDIR / "20261109_100000_box7"
    d.mkdir(parents=True)
    (d / "capture.json").write_text(json.dumps({"device_id": 7}))
    _faces_log(d / "free_air.log", faces=("x+", "x-"))
    cap = cal.refit({"source": "capture:20261109_100000_box7"})["captures"][0]
    assert "attitudes" in cap["fit"]["error"]
    assert "拟合被拒" in cal.register({"source": "capture:20261109_100000_box7", "endEffector": "x", "validFromUnix": 1.79e9})["error"]


def test_capture_counts_only_still_samples_after_the_pick_up():
    from tools.thor.gmsl2 import box_six_d_capture as cap

    c = cap.Counter(need=2)
    c.add([0, 0, 1.0], [0, 0, 0])  # resting before the pick-up: not counted
    assert c.faces["z+"] == 0 and not c.picked_up
    c.add([0, 0, 1.0], [40, 0, 0])  # the pick-up (moving: not counted itself)
    c.add([0.7, 0, 0.7], [0, 0, 0])  # between faces
    c.add([0, -1.0, 0], [0, 0, 0])
    c.add([0, -1.0, 0], [0, 20, 0])  # turning
    assert c.picked_up and c.faces["y-"] == 1 and sum(c.faces.values()) == 1 and not c.done


def test_desk_tcp_choices_become_desk_config_arguments(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    assert cal._tcp_args({}) == ([], "")  # the new tips' estimate is desk-config's default
    assert cal._tcp_args({"tcp": {"source": "template"}}) == (["--tcp-template"], "")
    assert cal._tcp_args({"tcp": {"source": "manual", "zM": 0.19}}) == (["--tcp-z-m", "0.19"], "")
    assert cal._tcp_args({"tcpZM": 0.18}) == (["--tcp-z-m", "0.18"], "")  # the older body
    assert cal._tcp_args({"tcp": {"source": "manual", "zM": 0.9}})[1]
    assert cal._tcp_args({"tcp": {"source": "pivot", "pivot": "../../etc/passwd"}})[1]
    pivots = tmp_path / "outputs" / "fr3_tcp_pivot"
    (pivots / "joint").mkdir(parents=True)
    fit = {"frame": "corenetic_gripper_ee", "robot_ip": "172.16.1.210", "fk": {"s_ee_mm": [0.4, -0.2, 18.1], "s_sigma_mm": [0.1, 0.1, 0.2], "residual_rms_mm": 0.6}}
    (pivots / "20261110_101010_pivot.fit.json").write_text(json.dumps(fit))
    (pivots / "joint" / "20261110_111111_joint.json").write_text(json.dumps(fit))
    (pivots / "20261110_101010_pivot.json").write_text("{}")  # the run itself, not a fit
    names = [f["name"] for f in cal.payload()["pivotFits"]]
    assert names == ["joint/20261110_111111_joint.json", "20261110_101010_pivot.fit.json"]
    args, refusal = cal._tcp_args({"tcp": {"source": "pivot", "pivot": "joint/20261110_111111_joint.json"}})
    assert not refusal and args[0] == "--tcp-from-pivot" and args[1].endswith("20261110_111111_joint.json")


def test_a_step_warning_lands_in_the_job_warnings_and_the_job_goes_on(tmp_path):
    cal = pc.PayloadCalibration(_host(tmp_path))
    cal.checks["right|six_d"] = {"pass": True, "atUnixS": time.time()}
    warn = "BOX 才上电 1 分钟；6D 拟合要求预热 30 分钟，读数会随温度漂，拟合可能不达标（照常继续）"
    _fake_tool(cal, ["BOX 3100715645: 6D + IMU of every hold -> x.six_d.log", f"WARNING: {warn}", "  pose  0/13 up    : n=1"])
    payload = cal.collect({"arm": "right", "poses": "six_d", "toolConfig": "gripper", "sixD": True})
    assert payload["ok"], payload["error"]
    job = _wait(cal)
    assert job["warnings"] == [warn]
    assert job["progress"]["done"] == 1  # it kept going after the warning


def test_the_fit_summary_carries_the_drift_and_the_warnings(tmp_path):
    from tools.thor.gmsl2 import box_six_d as bsd

    rng = np.random.default_rng(0)
    k = np.vstack([-0.8 * 9.80665 * np.eye(3), 0.05 * rng.standard_normal((3, 3))])
    d = tmp_path / pc.CAPTURE_SUBDIR / "20261109_100000_box7"
    d.mkdir(parents=True)
    (d / "capture.json").write_text(json.dumps({"device_id": 7}))
    rows, t = [], 1.79e9
    for face in ("x+", "x-", "y+", "y-", "z+", "z-", "x+"):
        axis, sign = "xyz".index(face[0]), 1.0 if face[1] == "+" else -1.0
        v = np.zeros(3)
        v[axis] = sign
        for _ in range(300):
            f = k @ v
            f[0] += 0.004 * (t - 1.79e9)  # Fx drifts 0.004 N/s
            rows.append({"t": t, "k": "box_six_d_force", "ts": int((t - 1.79e9 + 60) * 1e6), "f": list(f), "acc": list(v), "gyr": [0, 0, 0]})
            t += 0.01
        t += 40
    (d / "free_air.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cal = pc.PayloadCalibration(_host(tmp_path))
    cal._tool = lambda script, *args: [sys.executable, str(REPO / script), *args]  # type: ignore[method-assign]
    fit = cal.refit({"source": "capture:20261109_100000_box7"})["captures"][0]["fit"]
    assert fit["uptimeStartS"] == 60.0
    assert fit["drift"]["pairs"] == 1 and fit["drift"]["axis"] == "Fx" and fit["drift"]["changeN"] > bsd.DRIFT_WARN_N
    assert len(fit["warnings"]) == 2
    assert any(line.startswith("WARNING: ") and "Fx" in line for line in cal.job.get("log", []))
