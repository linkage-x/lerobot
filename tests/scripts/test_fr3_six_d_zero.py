"""Calibration page: the arm-held 6D zero (tools/data_collection_gui/fr3_six_d_zero.py)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tools.data_collection_gui import fr3_payload_calibration as pc, fr3_six_d_zero as z
from tools.thor.gmsl2 import box_six_d

REPO = Path(__file__).resolve().parents[2]
DEVICE = 3100715645
BOX_ID = f"box{DEVICE}"
K = [[0.1, -9.0, 0.2], [8.8, 0.1, -0.3], [0.2, 0.3, 9.1], [0.01, 0.2, 0.0], [-0.2, 0.01, 0.02], [0.0, 0.0, 0.01]]
ACC = {
    "start": (0.0, 0.02, -1.0),
    "six_d_zero": (-0.99, 0.01, -0.06),
    "check_xplus": (0.03, -1.0, 0.01),
    "check_yminus": (0.02, 0.99, 0.05),
}


def test_the_constants_are_box_six_d_s():
    assert z.FIXTURE_ACC_G == box_six_d.FIXTURE_ACC_G
    assert z.FIXTURE_ACC_TOL_G == box_six_d.FIXTURE_ACC_TOL_G
    assert z.ORIGIN_RAW_LIMITS == box_six_d.ORIGIN_RAW_LIMITS
    assert z.WARMUP_S == box_six_d.WARMUP_S


def test_the_box_id_is_exact_or_nothing():
    roster = [{"device_id": DEVICE, "box_id": BOX_ID}, {"device_id": 3906437858, "box_id": "box3906437858"}]
    assert z.box_id_for(roster, DEVICE) == BOX_ID
    # not connected: never fall back to "" (BoxPool would zero every BOX, the handheld too)
    assert z.box_id_for(roster[1:], DEVICE) is None
    assert z.box_id_for([{"device_id": DEVICE, "box_id": ""}], DEVICE) == ""
    assert z.box_id_for([{"device_id": DEVICE, "box_id": ""}, roster[1]], DEVICE) is None


def test_external_and_k_choice_match_box_six_d():
    rng = np.random.default_rng(0)
    f, f0 = rng.normal(size=6).tolist(), rng.normal(size=6).tolist()
    acc, acc0 = rng.normal(size=3).tolist(), rng.normal(size=3).tolist()
    registry = {
        "schema": box_six_d.SCHEMA,
        "devices": {
            str(DEVICE): {
                "six_d_payload": [
                    {"valid_from_unix": 100, "model": box_six_d.MODEL, "K": K, "end_effector": "a"},
                    {"valid_from_unix": 200, "model": box_six_d.MODEL, "K": None, "end_effector": "b"},
                ]
            }
        },
    }
    k, rec = z.payload_k(registry, DEVICE, 150)
    model = box_six_d.payload_model(registry, DEVICE, 150)
    assert k == K and rec["end_effector"] == "a"
    bias = box_six_d.zero_bias(model, {"kind": "software", "fxyz_mxyz": f0, "acc_g": acc0})
    np.testing.assert_allclose(z.external(f, acc, f0, acc0, k), box_six_d.external_wrench(f, acc, model, bias))
    # the end effector changed and nobody re-fitted: no K, as box_six_d says
    assert z.payload_k(registry, DEVICE, 250)[0] is None and box_six_d.payload_model(registry, DEVICE, 250) is None


class FakeRig:
    """The arm (waypoints), the BOX (raw = K acc + offset, f = raw + the SDK's shift) and the recorder."""

    def __init__(self, zero_acc=ACC["six_d_zero"], k=K):
        self.at = "start"
        self.moves: list[str] = []
        self.k = np.asarray(k)
        self.zero_acc = zero_acc
        self.raw_offset = np.array([0.8, -0.4, 1.5, 0.02, -0.03, 0.01])
        self.shift = np.zeros(6)
        self.boot = time.time() - 4000.0
        self.pending_offset = None
        self.cali_log: list[dict] = []
        self.cali_running = False
        self.triggers: list[tuple[bool, str]] = []
        self.dist = 0.0346

    def acc(self):
        return np.asarray(self.zero_acc if self.at == "six_d_zero" else ACC[self.at])

    def preview(self):
        acc = self.acc()
        raw = self.k @ acc + self.raw_offset
        now = time.time()
        return {
            "active": True,
            "updatedAt": now,
            "sensors": {
                f"{BOX_ID}/box_six_d_force": {
                    "fxyz_mxyz": (raw + self.shift).tolist(), "fxyz_mxyz_raw": raw.tolist(),
                    "timestamp": int((now - self.boot) * 1e6),
                },
                f"{BOX_ID}/box_imu": {"acc_g": acc.tolist(), "gyr_deg_s": [0.1, -0.2, 0.05]},
                f"{BOX_ID}/box_gripper": {"distance_m": self.dist},
            },
        }

    def trigger(self, origin, box_id):
        self.triggers.append((origin, box_id))
        now = time.time()
        self.cali_log.append({"ts": now, "line": "command sent", "done": False})
        if origin:
            self.pending_offset = -(self.k @ self.acc())  # raw reads 0 at this attitude after the re-power

            def repower():  # the operator, a moment later
                time.sleep(0.2)
                self.raw_offset = self.pending_offset
                self.boot = time.time()

            threading.Thread(target=repower, daemon=True).start()
        else:
            target = np.array([t for t, _ in z.DYNAMIC_ACCEPT])
            self.shift = target - (self.k @ self.acc() + self.raw_offset)
        self.cali_log.append({"ts": time.time(), "line": "ok", "done": True})
        return {"ok": True}

    def run_step(self, command):
        to = command[command.index("--to") + 1]
        self.moves.append(to)
        self.at = to
        return 0


def _fast(monkeypatch):
    for name, value in {
        "SETTLE_S": 0.0, "STILL_WINDOW_S": 0.15, "POLL_S": 0.01, "REPOWER_WINDOW_S": 0.1,
        "REPOWER_MIN_SAMPLES": 3, "REPOWER_MIN_UPTIME_S": 0.05, "CALI_TIMEOUT_S": 5.0,
    }.items():
        monkeypatch.setattr(z, name, value)


def _cal(tmp_path, rig: FakeRig, *, registered=True, connected=True):
    if registered:
        reg = tmp_path / pc.REGISTRY
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps({"schema": box_six_d.SCHEMA, "devices": {str(DEVICE): {"six_d_payload": [
            {"valid_from_unix": 1.7e9, "valid_from": "x", "model": box_six_d.MODEL, "K": K, "end_effector": "gripper"}
        ]}}}))
    host = pc.Host(
        repo_root=tmp_path,
        python=lambda: Path(sys.executable),
        env=lambda: {},
        resolve_arm=lambda arm: {"arm": "right", "robot_ip": "192.168.1.210", "corenetic_device_id": DEVICE},
        arms=lambda: [],
        fci_holder=lambda: "",
        box_port_holder=lambda: "",
        log=lambda level, msg: None,
        recorder_box_refusal=lambda: "" if connected else "录制器没有连接；先 Connect",
        box_preview=rig.preview,
        box_roster=lambda: [{"device_id": DEVICE, "box_id": BOX_ID}, {"device_id": 3906437858, "box_id": "box3906437858"}],
        trigger_six_d_cali=rig.trigger,
        six_d_cali_log=lambda: (rig.cali_running, [dict(e) for e in rig.cali_log]),
    )
    cal = pc.PayloadCalibration(host)
    cal._run_step = rig.run_step  # type: ignore[method-assign]
    cal.checks["right|six_d_zero"] = {"pass": True, "atUnixS": time.time()}
    return cal


def _wait(cal, timeout=20.0):
    deadline = time.monotonic() + timeout
    while cal.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    return cal.payload()["job"]


def test_a_dynamic_zero_goes_up_zeros_comes_back_and_rechecks_on_two_sides(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig()
    cal = _cal(tmp_path, rig)
    res = cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})
    assert res["ok"], res["error"]
    assert "FCI" not in cal.fci_user() and "零点" in cal.fci_user()
    job = _wait(cal)
    assert job["state"] == "done", (job["error"], job["log"][-8:])
    assert rig.moves == ["six_d_zero", "start", "check_xplus", "check_yminus", "start"]
    assert rig.triggers == [(False, BOX_ID)]  # this BOX only, never the whole fleet
    assert [c["id"] for c in job["checks"]] == ["attitude", "dynamic", "check_xplus", "check_yminus"]
    assert all(c["pass"] for c in job["checks"]) and job["pass"]
    report = json.loads((tmp_path / z.REPORT_SUBDIR / job["report"]).read_text())
    assert report["pass"] and report["mode"] == "dynamic" and report["k_record"]["end_effector"] == "gripper"
    np.testing.assert_allclose(report["zero"]["acc_g"], ACC["six_d_zero"])
    assert cal.payload()["zeroReports"][0]["pass"]


def test_an_origin_waits_for_the_re_power_and_checks_raw_before_the_dynamic_zero(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig()
    cal = _cal(tmp_path, rig)
    assert cal.zero_start({"arm": "right", "mode": "origin", "confirmed": True})["ok"]
    job = _wait(cal)
    assert job["state"] == "done", (job["error"], job["log"][-8:])
    assert rig.triggers == [(True, BOX_ID), (False, BOX_ID)]
    assert [c["id"] for c in job["checks"]][:3] == ["attitude", "origin_raw", "dynamic"]
    assert job["pass"]


def test_a_wrong_attitude_zeros_nothing_and_goes_back(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig(zero_acc=(-0.7, 0.6, -0.05))  # the BOX sits rotated in the gripper: not the fixture's attitude
    cal = _cal(tmp_path, rig)
    assert cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})["ok"]
    job = _wait(cal)
    assert job["state"] == "error" and "姿态" in job["error"]
    assert rig.triggers == [] and rig.moves == ["six_d_zero", "start"]


def test_a_recheck_off_by_more_than_half_a_newton_fails_the_zero(tmp_path, monkeypatch):
    _fast(monkeypatch)
    real = np.asarray(K) * 1.0
    real[2, 1] += 1.2  # the registered K is not this payload's (e.g. something changed on the gripper)
    rig = FakeRig(k=real)
    cal = _cal(tmp_path, rig)
    assert cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})["ok"]
    job = _wait(cal)
    assert job["state"] == "error" and not job["pass"]
    assert rig.moves[-1] == "start"
    failed = [c["id"] for c in job["checks"] if not c["pass"]]
    assert failed and set(failed) <= {"check_xplus", "check_yminus"}


def test_without_a_registered_fit_the_rechecks_are_skipped_with_a_warning(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig()
    cal = _cal(tmp_path, rig, registered=False)
    assert cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})["ok"]
    job = _wait(cal)
    assert job["state"] == "done" and rig.moves == ["six_d_zero", "start"]
    assert any("跳过" in w for w in job["warnings"])


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"confirmed": False}, "清单"),
        ({"mode": "both"}, "校零方式"),
        ({"check": None}, "零点路线"),
        ({"connected": False}, "Connect"),
    ],
)
def test_the_zero_is_refused_before_anything_moves(tmp_path, change, expected):
    rig = FakeRig()
    cal = _cal(tmp_path, rig, connected=change.get("connected", True))
    if "check" in change:
        cal.checks.clear()
    body = {"arm": "right", "mode": change.get("mode", "dynamic"), "confirmed": change.get("confirmed", True)}
    res = cal.zero_start(body)
    assert not res["ok"] and expected in res["error"]
    assert rig.moves == [] and rig.triggers == []


def test_an_origin_on_a_cold_box_warns_but_still_runs(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig()
    rig.boot = time.time() - 23 * 60
    cal = _cal(tmp_path, rig)
    res = cal.zero_start({"arm": "right", "mode": "origin", "confirmed": True})
    assert res["ok"], res["error"]
    job = _wait(cal)
    assert job["state"] == "done", (job["error"], job["log"][-8:])
    assert rig.triggers == [(True, BOX_ID), (False, BOX_ID)]
    assert any("才上电 23 分钟" in w and "origin 要先预热 30 分钟" in w for w in job["warnings"])
    report = json.loads((tmp_path / z.REPORT_SUBDIR / job["report"]).read_text())
    assert any("预热" in w for w in report["warnings"])


def test_stop_between_moves_ends_the_job_where_the_arm_is(tmp_path, monkeypatch):
    _fast(monkeypatch)
    monkeypatch.setattr(z, "SETTLE_S", 2.0)
    rig = FakeRig()
    cal = _cal(tmp_path, rig)
    assert cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})["ok"]
    time.sleep(0.3)
    cal.stop()
    job = _wait(cal)
    assert job["state"] == "stopped" and rig.moves == ["six_d_zero"] and rig.triggers == []


def test_the_zero_route_is_a_check_pose_set(tmp_path):
    cal = pc.PayloadCalibration(
        pc.Host(
            repo_root=tmp_path, python=lambda: Path(sys.executable), env=lambda: {},
            resolve_arm=lambda arm: {"arm": "right", "robot_ip": "x"}, arms=list, fci_holder=lambda: "",
            box_port_holder=lambda: "", log=lambda *a: None,
        )
    )
    cal._tool = lambda script, *args: [sys.executable, "-c", f"print({list(args)!r})"]  # type: ignore[method-assign]
    assert cal.check({"arm": "right", "poses": "six_d_zero"})["ok"]
    _wait(cal)
    assert "'--poses', 'six_d_zero'" in cal.payload()["job"]["log"][0]


def test_a_failed_move_is_not_followed_by_another_one(tmp_path, monkeypatch):
    _fast(monkeypatch)
    rig = FakeRig()
    cal = _cal(tmp_path, rig)

    def reflex(command):  # the move up stops on a reflex
        rig.moves.append(command[command.index("--to") + 1])
        return 1

    cal._run_step = reflex  # type: ignore[method-assign]
    assert cal.zero_start({"arm": "right", "mode": "dynamic", "confirmed": True})["ok"]
    job = _wait(cal)
    assert job["state"] == "error" and "手动" in job["error"]
    assert rig.moves == ["six_d_zero"] and rig.triggers == []


def test_the_box_tab_still_finds_this_mornings_origin_after_many_dynamic_zeros(tmp_path):
    cal = _cal(tmp_path, FakeRig())
    root = tmp_path / z.REPORT_SUBDIR
    root.mkdir(parents=True)

    def report(created: str, mode: str, passed: bool, arm: str = "right", device: int = DEVICE) -> None:
        stamp = created.replace("-", "").replace(":", "").replace("T", "_")
        rep = {"created": created, "arm": arm, "device_id": device, "mode": mode, "pass": passed, "state": "done"}
        (root / f"{stamp}_{arm}_box{device}.json").write_text(json.dumps(rep))

    report("2026-11-10T08:00:00", "origin", True)
    report("2026-11-10T08:10:00", "origin", False)
    for i in range(pc.LIST_LIMIT + 3):
        report(f"2026-11-10T{9 + i // 2:02d}:{30 * (i % 2):02d}:00", "dynamic", True)
    report("2026-11-10T09:15:00", "dynamic", True, arm="left", device=1234)
    report("2026-11-10T17:00:00", "dynamic", False)

    payload = cal.payload()
    assert all(r["mode"] == "dynamic" for r in payload["zeroReports"])  # the origin fell off the recent list
    latest = {(r["arm"], r["deviceId"], r["mode"]): r["created"] for r in payload["zeroLatest"]}
    assert latest == {
        ("right", DEVICE, "origin"): "2026-11-10T08:00:00",  # the failed 08:10 one does not count
        ("right", DEVICE, "dynamic"): "2026-11-10T16:00:00",  # nor the failed 17:00 one
        ("left", 1234, "dynamic"): "2026-11-10T09:15:00",
    }
