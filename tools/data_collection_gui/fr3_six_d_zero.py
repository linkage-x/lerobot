"""Calibration page: the 6D zero of an FR3 gripper BOX, with the arm as the fixture (fingertips up).

The vendor procedure (数采Box传感器标定与操作说明 v1.1) zeros the 6D sensor upright on its fixture,
fingertips up, never held by hand. On the arm the robot holds that attitude instead. The job:

  1. moves start -> six_d_zero (``move-waypoint``; only j2/j4/j6, the route check-poses checked)
  2. waits until the BOX is still, then checks the attitude against the fixture's IMU acc
  3. origin mode: ``cali_6d_force_sensor_origin``, waits for the operator to re-power the BOX (the arm
     keeps holding), reads raw past the boot and checks it (±0.5 N / ±0.01 N·m)
  4. dynamic zero (``cali_6d_force_sensor``) and its acceptance (Fz −7.784 ±0.5 N, Mx −0.168 ±0.01 N·m),
     then reads f0 / acc0 at the zero pose
  5. back to start, then the two re-check poses: external = (f − f0) − K·(acc − acc0) with the BOX's
     registered 6D fit, every force axis within 0.5 N
  6. back to start, report in outputs/fr3_six_d_zero/

The zeros go through the recorder (it owns the BOX port, and the software zero lives in its SDK
process: a zero made by another process would be gone at Connect). So the recorder must be
connected, and the job reads the BOX from the recorder's BOX_LIVE preview (10 Hz).

The gateway's python has no numpy: the arithmetic here is plain Python on 6x3 matrices.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools.data_collection_gui.fr3_payload_calibration import PayloadCalibration

REPORT_SUBDIR = Path("outputs") / "fr3_six_d_zero"
MODES = ("dynamic", "origin")

# box_six_d.py's constants (that module needs numpy); a test keeps the two equal.
FIXTURE_ACC_G = (-1.0, 0.0, -0.05)
FIXTURE_ACC_TOL_G = 0.15
ORIGIN_RAW_LIMITS = (0.5, 0.5, 0.5, 0.01, 0.01, 0.01)
WARMUP_S = 30 * 60
# The vendor's acceptance after a dynamic zero (the calibration page's DYNAMIC_FORCE_LIMITS): the SDK's
# own payload model at the fixture attitude, (target, tolerance) per channel.
DYNAMIC_ACCEPT = ((0.0, 0.5), (0.0, 0.5), (-7.784, 0.5), (-0.168, 0.01), (0.0, 0.01), (0.0, 0.01))
RECHECK_FORCE_N = 0.5
CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

# The arm stops at the end of a move; the BOX then reads still when its gyro and force are quiet over
# a window. Tries are separate windows, so a bump does not fail the job at once.
SETTLE_S = 3.0
STILL_WINDOW_S = 2.0
STILL_GYR_DEG_S = 2.0
STILL_FORCE_SPAN_N = 0.3
STILL_TRIES = 5
STALE_S = 1.5
POLL_S = 0.1
CALI_TIMEOUT_S = 45.0
# Origin re-check (the panel's ORIGIN_RECHECK): raw once the BOX is back past 25 s of uptime, over 3 s.
REPOWER_MIN_UPTIME_S = 25.0
REPOWER_WINDOW_S = 3.0
REPOWER_MIN_SAMPLES = 5
REPOWER_TIMEOUT_S = 300.0
REBOOT_SLACK_S = 2.0
# A re-power can open the jaw (10-13); the zero belongs to the opening K was fitted at.
GRIPPER_DRIFT_M = 0.002


# ---------------------------------------------------------------- pure helpers
def box_id_for(roster: list[dict[str, Any]], device_id: int) -> str | None:
    """The recorder's box_id (sensor prefix, calibration target) of ``device_id``, or None if not connected.

    Exact or nothing: BoxPool zeros EVERY BOX when the box_id matches none, and a handheld zeroed off its
    fixture is ruined. "" (the single-BOX passthrough) only when this BOX is the one connected.
    """
    found = [e for e in roster if str(e.get("device_id")) == str(device_id)]
    if len(found) != 1:
        return None
    box_id = str(found[0].get("box_id") or "")
    if not box_id and len(roster) != 1:
        return None
    return box_id


def sample(preview: dict[str, Any], box_id: str, now_s: float) -> dict[str, Any] | None:
    """One BOX reading from the recorder's preview, or None when it is stale or incomplete."""
    updated = float(preview.get("updatedAt") or 0.0)
    if not preview.get("active") or now_s - updated > STALE_S:
        return None
    sensors = preview.get("sensors") or {}
    prefix = f"{box_id}/" if box_id else ""
    six_d = sensors.get(prefix + "box_six_d_force") or {}
    imu = sensors.get(prefix + "box_imu") or {}
    grip = sensors.get(prefix + "box_gripper") or {}
    f, acc = six_d.get("fxyz_mxyz"), imu.get("acc_g")
    if not f or len(f) != 6 or not acc or len(acc) != 3:
        return None
    return {
        "us": six_d.get("timestamp"),
        "f": [float(v) for v in f],
        "raw": [float(v) for v in six_d["fxyz_mxyz_raw"]] if six_d.get("fxyz_mxyz_raw") else None,
        "acc": [float(v) for v in acc],
        "gyr": [float(v) for v in imu.get("gyr_deg_s") or (0.0, 0.0, 0.0)],
        "dist": grip.get("distance_m"),
    }


def median_vec(rows: list[list[float]]) -> list[float]:
    return [statistics.median(col) for col in zip(*rows, strict=True)]


def still(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Whether a window of readings is at rest: gyro quiet and the force not moving."""
    if len(rows) < 5:
        return False, f"只有 {len(rows)} 个样本"
    gyr = max(max(abs(v) for v in r["gyr"]) for r in rows)
    span = max(max(r["f"][i] for r in rows) - min(r["f"][i] for r in rows) for i in range(3))
    ok = gyr < STILL_GYR_DEG_S and span < STILL_FORCE_SPAN_N
    return ok, f"gyr 最大 {gyr:.2f}°/s，力波动 {span:.2f} N"


def attitude_check(acc: list[float]) -> tuple[bool, float]:
    dist = sum((a - b) ** 2 for a, b in zip(acc, FIXTURE_ACC_G, strict=True)) ** 0.5
    return dist <= FIXTURE_ACC_TOL_G, dist


def within(values: list[float], limits) -> bool:
    return all(abs(v) <= lim for v, lim in zip(values, limits, strict=True))


def dynamic_check(f: list[float]) -> bool:
    return all(abs(v - t) <= tol for v, (t, tol) in zip(f, DYNAMIC_ACCEPT, strict=True))


def payload_k(registry: dict[str, Any], device_id: int, at_unix: float) -> tuple[list[list[float]] | None, dict]:
    """box_six_d.payload_model's choice: the latest record not after ``at_unix``, if it has a K."""
    entry = (registry.get("devices") or {}).get(str(int(device_id))) or {}
    eligible = [p for p in entry.get("six_d_payload") or [] if float(p.get("valid_from_unix") or 0) <= at_unix]
    if not eligible:
        return None, {}
    p = max(eligible, key=lambda p: float(p["valid_from_unix"]))
    k = p.get("K")
    if p.get("model") != "affine_acc/v1" or not k or len(k) != 6 or any(len(row) != 3 for row in k):
        return None, p
    return [[float(v) for v in row] for row in k], p


def external(f, acc, f0, acc0, k) -> list[float]:
    """(f − f0) − K·(acc − acc0): the external wrench under a software zero taken at (f0, acc0)."""
    da = [a - b for a, b in zip(acc, acc0, strict=True)]
    return [f[i] - f0[i] - sum(k[i][j] * da[j] for j in range(3)) for i in range(6)]


def rebooted(us: float, baseline_us: float, baseline_wall_s: float, now_s: float) -> bool:
    """A BOX clock behind where an unbroken one would be: it restarted since the baseline."""
    return us < baseline_us + (now_s - baseline_wall_s) * 1e6 - REBOOT_SLACK_S * 1e6


def fmt(v: list[float]) -> str:
    return ", ".join(f"{c} {x:+.3f}" for c, x in zip(CHANNELS, v, strict=False))


def report_summary(path: Path) -> dict[str, Any] | None:
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rep, dict):
        return None
    return {
        "name": path.name,
        "created": rep.get("created", ""),
        "arm": rep.get("arm", ""),
        "deviceId": rep.get("device_id"),
        "mode": rep.get("mode", ""),
        "pass": bool(rep.get("pass")),
        "state": rep.get("state", ""),
        "checks": rep.get("checks") or [],
        "warnings": rep.get("warnings") or [],
    }


# ---------------------------------------------------------------- the job
class JobStoppedError(Exception):
    pass


class StepFailedError(Exception):
    pass


class MoveFailedError(StepFailedError):
    """A move itself failed (a reflex, a refusal): the job does not try another one on its own."""


class ZeroJob:
    """One run, on its own thread. ``cal`` holds the job dict, the log, the lock and the subprocess."""

    def __init__(self, cal: PayloadCalibration, ctx: dict[str, Any]) -> None:
        self.cal = cal
        self.ctx = ctx
        self.report: dict[str, Any] = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "arm": ctx["arm"],
            "robot_ip": ctx["robot_ip"],
            "device_id": ctx["device_id"],
            "box_id": ctx["box_id"],
            "mode": ctx["mode"],
            "k_record": {k: ctx["k_record"].get(k) for k in ("valid_from", "valid_from_unix", "end_effector")}
            if ctx.get("k_record")
            else None,
            "checks": [],
            "warnings": [],
        }
        self.at = "start"

    # -- plumbing
    def log(self, line: str) -> None:
        self.cal._append_log(line)

    def phase(self, text: str, prompt: str = "") -> None:
        with self.cal.lock:
            self.cal.job["phase"] = text
            self.cal.job["prompt"] = prompt
        self.log(f"== {text}")

    def check(self, cid: str, label: str, ok: bool, detail: str) -> bool:
        item = {"id": cid, "label": label, "pass": bool(ok), "detail": detail}
        self.report["checks"].append(item)
        with self.cal.lock:
            self.cal.job.setdefault("checks", []).append(item)
        self.log(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
        return ok

    def warn(self, text: str) -> None:
        self.report["warnings"].append(text)
        with self.cal.lock:
            self.cal.job.setdefault("warnings", []).append(text)
        self.log(f"WARN {text}")

    def stop_check(self) -> None:
        if self.cal._stop_requested():
            raise JobStoppedError

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.stop_check()
            time.sleep(POLL_S)

    def read(self) -> dict[str, Any] | None:
        return sample(self.cal.host.box_preview() or {}, self.ctx["box_id"], time.time())

    def collect(self, seconds: float) -> list[dict[str, Any]]:
        """Distinct readings (by the BOX clock) over ``seconds``; StepFailedError if the preview goes stale."""
        rows: list[dict[str, Any]] = []
        last_us = None
        stale_since = None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.stop_check()
            r = self.read()
            if r is None:
                stale_since = stale_since or time.monotonic()
                if time.monotonic() - stale_since > 3.0:
                    raise StepFailedError("录制器的 BOX 实时数据中断（Disconnect 了？BOX 掉电？）")
            else:
                stale_since = None
                if r["us"] is None or r["us"] != last_us:
                    rows.append(r)
                    last_us = r["us"]
            time.sleep(POLL_S)
        return rows

    def at_rest(self, where: str) -> list[dict[str, Any]]:
        self.wait(SETTLE_S)
        detail = ""
        for attempt in range(STILL_TRIES):
            rows = self.collect(STILL_WINDOW_S)
            ok, detail = still(rows)
            if ok:
                return rows
            self.log(f"  {where}: 还没静止（{detail}），再等（{attempt + 1}/{STILL_TRIES}）")
        raise StepFailedError(f"{where}: BOX 一直没有静止（{detail}）；别碰夹爪和线缆")

    def move(self, to: str) -> None:
        self.stop_check()
        command = self.cal._tool(
            self.ctx["script"], "move-waypoint", "--robot-ip", self.ctx["robot_ip"], "--to", to
        )
        if self.ctx["arm"]:
            command[3:3] = ["--arm", self.ctx["arm"]]
        code = self.cal._run_step(command)
        self.stop_check()
        if code != 0:
            raise MoveFailedError(f"机械臂移到 {to} 失败（exit {code}），看日志；确认原因后手动 move to start")
        self.at = to

    def cali(self, origin: bool) -> None:
        name = "origin" if origin else "动态校零"
        sent_at = time.time()
        res = self.cal.host.trigger_six_d_cali(origin, self.ctx["box_id"])
        if not res.get("ok"):
            raise StepFailedError(f"{name} 命令没发出去：{res.get('error')}")
        logged = 0
        end = time.monotonic() + CALI_TIMEOUT_S
        while time.monotonic() < end:
            self.stop_check()
            running, entries = self.cal.host.six_d_cali_log()
            # the log is a bounded buffer shared with the BOX panel: this command's lines by their time
            fresh = [e for e in entries if float(e.get("ts") or 0) >= sent_at]
            for e in fresh[logged:]:
                self.log(f"  录制器: {e.get('line')}")
            logged = max(logged, len(fresh))
            done = [e for e in fresh if e.get("done")]
            if done and not running:
                if str(done[-1].get("line")).strip() != "ok":
                    raise StepFailedError(f"{name} 失败（录制器 CALI_DONE {done[-1].get('line')}）")
                return
            time.sleep(0.2)
        raise StepFailedError(f"{name} {CALI_TIMEOUT_S:.0f} s 没有结果")

    # -- steps
    def run(self) -> None:
        state, error = "done", ""
        try:
            self.sequence()
        except JobStoppedError:
            state = "stopped"
            error = f"已停止；机械臂停在 {self.at} 附近，确认安全后用 move to start 回位"
        except StepFailedError as exc:
            state, error = "error", str(exc)
            self.log(f"ERROR {exc}")
            if self.at != "start" and not isinstance(exc, MoveFailedError) and not self.cal._stop_requested():
                try:
                    self.phase("出错后回到起点")
                    self.move("start")
                except (StepFailedError, JobStoppedError) as exc2:
                    error += f"；回起点也失败：{exc2}"
        except Exception as exc:  # noqa: BLE001 -- the job must end with a state
            state, error = "error", f"{type(exc).__name__}: {exc}"
        passed = state == "done" and all(c["pass"] for c in self.report["checks"])
        self.report.update({"state": state, "error": error, "pass": passed})
        with self.cal.lock:
            self.report["log"] = list(self.cal.job.get("log") or [])
        path = self.write_report()
        with self.cal.lock:
            self.cal.job["report"] = path.name
            self.cal.job["pass"] = passed
            self.cal.job["prompt"] = ""
        if state == "done" and not passed:
            state, error = "error", "零点流程走完了，但有检查没通过（见红项）"
        self.cal._finish(state, error)
        self.cal.host.log("info" if passed else "warn", f"FR3 6D zero ({self.ctx['arm']}): {state}, pass={passed}")

    def write_report(self) -> Path:
        out = self.cal.root / REPORT_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        stamp = self.report["created"].replace("-", "").replace(":", "").replace("T", "_")
        path = out / f"{stamp}_{self.ctx['arm'] or 'fr3'}_box{self.ctx['device_id']}.json"
        path.write_text(json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def sequence(self) -> None:
        first = self.read()
        if first is None:
            raise StepFailedError("读不到这台 BOX 的实时数据")
        uptime = (first["us"] or 0) / 1e6
        self.report["uptime_at_start_s"] = round(uptime, 1)
        if uptime < WARMUP_S:
            need = "origin 要先预热 30 分钟" if self.ctx["mode"] == "origin" else "要求预热 30 分钟"
            self.warn(f"BOX 才上电 {uptime / 60:.0f} 分钟；{need}，零点可能随温度漂（照常继续）")
        dist0 = first["dist"]

        self.phase("① 移到「指尖朝上」零点姿态（只动 j2/j4/j6）")
        self.move("six_d_zero")

        self.phase("② 等静止、核对姿态")
        rows = self.at_rest("零点姿态")
        acc = median_vec([r["acc"] for r in rows])
        ok, dist = attitude_check(acc)
        self.report["acc_at_zero_pose_g"] = acc
        if not self.check(
            "attitude", "姿态（相当于工装）", ok,
            f"IMU acc = ({', '.join(f'{a:+.3f}' for a in acc)}) g，工装 {FIXTURE_ACC_G}，偏差 {dist:.3f} g（≤{FIXTURE_ACC_TOL_G}）",
        ):
            raise StepFailedError("姿态与工装不符，没有校零")

        if self.ctx["mode"] == "origin":
            self.origin_step()

        self.phase("④ 动态校零（cali_6d_force_sensor）")
        self.cali(origin=False)
        rows = self.at_rest("校零后")
        f0 = median_vec([r["f"] for r in rows])
        acc0 = median_vec([r["acc"] for r in rows])
        self.report["zero"] = {"fxyz_mxyz": f0, "acc_g": acc0, "wall_s": round(time.time(), 3)}
        self.check(
            "dynamic", "动态校零判定", dynamic_check(f0),
            f"{fmt(f0)}（要求 Fz −7.784±0.5 N、Mx −0.168±0.01 N·m，其余 ±0.5 N / ±0.01 N·m）",
        )
        dist_zero = rows[-1]["dist"]
        if dist0 is not None and dist_zero is not None and abs(dist_zero - dist0) > GRIPPER_DRIFT_M:
            self.warn(
                f"夹爪开度从 {dist0 * 1000:.1f} mm 变成 {dist_zero * 1000:.1f} mm（重新上电会张开夹爪）；"
                "零点要在 K 拟合时的开度下做"
            )
        self.report["gripper_distance_m"] = {"start": dist0, "zero": dist_zero}

        self.phase("⑤ 回到起点")
        self.move("start")

        k = self.ctx.get("k")
        if k is None:
            self.warn("这台 BOX 没有登记 6D 拟合（box_devices.json），跳过两个姿态的复核；先做负载+6D 采集并登记")
        else:
            for name, label in (("check_xplus", "复核 1/2（x+ 朝上）"), ("check_yminus", "复核 2/2（y− 朝上）")):
                self.phase(f"⑥ {label}")
                self.move(name)
                rows = self.at_rest(label)
                f = median_vec([r["f"] for r in rows])
                acc = median_vec([r["acc"] for r in rows])
                ext = external(f, acc, f0, acc0, k)
                self.report.setdefault("rechecks", {})[name] = {"f": f, "acc_g": acc, "external": ext}
                self.check(
                    name, label, within(ext[:3], (RECHECK_FORCE_N,) * 3),
                    f"外力 {fmt(ext)}（力每轴 ≤{RECHECK_FORCE_N} N）",
                )
            self.phase("⑦ 回到起点")
            self.move("start")
        self.phase("完成")

    def origin_step(self) -> None:
        self.phase("③ origin（cali_6d_force_sensor_origin）")
        self.cali(origin=True)
        base = self.read()
        if base is None or base["us"] is None:
            raise StepFailedError("origin 后读不到 BOX 时钟")
        base_us, base_wall = float(base["us"]), time.time()
        self.phase(
            "③ 等 BOX 重新上电",
            "现在给这台 BOX 断电再上电（机械臂会一直停在这里）。上电后等 25 s 读 raw，不要碰夹爪。",
        )
        samples: list[list[float]] = []
        first_ms = None
        last_us = None
        deadline = time.monotonic() + REPOWER_TIMEOUT_S
        is_back = False
        while time.monotonic() < deadline:
            self.stop_check()
            r = self.read()
            if r is not None and r["us"] is not None and r["us"] != last_us:
                last_us = r["us"]
                now = time.time()
                if not is_back and rebooted(float(r["us"]), base_us, base_wall, now):
                    is_back = True
                    self.log(f"  BOX 已重新上电（MCU 时钟 {r['us'] / 1e6:.1f} s）")
                if is_back and r["us"] / 1e6 >= REPOWER_MIN_UPTIME_S and r["raw"]:
                    if first_ms is None:
                        self.phase("③ 上电后读 raw")
                        first_ms = time.monotonic()
                    samples.append(r["raw"])
                    if len(samples) >= REPOWER_MIN_SAMPLES and time.monotonic() - first_ms >= REPOWER_WINDOW_S:
                        break
            time.sleep(POLL_S)
        else:
            raise StepFailedError(
                "等 BOX 重新上电超时" if not is_back else "BOX 上电了，但没读到足够的 raw 样本"
            )
        raw = median_vec(samples)
        self.report["origin_recheck_raw"] = raw
        if not self.check(
            "origin_raw", "origin 上电后 raw", within(raw, ORIGIN_RAW_LIMITS),
            f"{fmt(raw)}（要求 ±0.5 N / ±0.01 N·m）",
        ):
            raise StepFailedError("origin 复核没过；按手册重做 origin（BOX 留在臂上）")
