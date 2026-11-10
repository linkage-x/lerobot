"""Calibration page: the FR3 end-effector load (Desk) and the BOX 6D payload fit (K).

The gateway owns the HTTP routes and hands this module what it needs from its state through
:class:`Host`; everything else (the job, the files, the gates) lives here, so the gateway file does
not grow by another few hundred lines and this part can be tested without it.

One job at a time, whatever its kind: a payload ``collect`` holds the arm's FCI and (with 6D logging)
the BOX port, a hand-held capture holds the BOX port, the arm-held 6D zero (fr3_six_d_zero.py) holds
the FCI and zeros through the recorder, and a ``check-poses`` is the gate for a collect or a zero.

Tools run as subprocesses in the FR3 venv (the gateway's python has no numpy):
  tools/fr3/fr3_payload_identification_runtime.py  check-poses / collect / desk-config
  tools/thor/gmsl2/box_six_d_capture.py              hand-held free-air capture
  tools/thor/gmsl2/box_six_d.py                      fit / register

Files:
  outputs/fr3_payload/<stamp>_fr3_<config>.json         a collect run (+ .six_d.log, .six_d.fit.json)
  outputs/fr3_payload/desk/<stamp>/                      desk-config output + report.json
  outputs/calibration/box_six_d/<stamp>_box<id>/        hand-held capture: free_air.log, status.json,
                                                         capture.json, fit.json
  outputs/fr3_six_d_zero/<stamp>_<arm>_box<id>.json      an arm-held 6D zero: checks, f0/acc0, log
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from tools.data_collection_gui import fr3_six_d_zero as six_d_zero

PAYLOAD_SCRIPT = Path("tools") / "fr3" / "fr3_payload_identification_runtime.py"
CAPTURE_SCRIPT = Path("tools") / "thor" / "gmsl2" / "box_six_d_capture.py"
SIX_D_SCRIPT = Path("tools") / "thor" / "gmsl2" / "box_six_d.py"
REGISTRY = Path("tools") / "thor" / "box_devices.json"
RUNS_SUBDIR = Path("outputs") / "fr3_payload"
DESK_SUBDIR = RUNS_SUBDIR / "desk"
CAPTURE_SUBDIR = Path("outputs") / "calibration" / "box_six_d"

TOOL_CONFIGS = (
    {"id": "gripper", "label": "仅夹爪（无相机、无 SMR 靶座）"},
    {"id": "gripper_cam", "label": "夹爪 + 相机/SMR 支架（SMR 取下）"},
    {"id": "gripper_cam_smr", "label": "夹爪 + 相机/SMR 支架 + SMR"},
)
TOOL_CONFIG_IDS = tuple(c["id"] for c in TOOL_CONFIGS)
POSE_SETS = (
    {"id": "payload+six_d", "label": "负载 + 6D（45 个姿态，约 12 分钟）", "poses": 45},
    {"id": "payload", "label": "仅负载（32 个姿态，约 8 分钟）", "poses": 32},
    {"id": "six_d", "label": "仅 6D 六面（13 个姿态，约 4 分钟）", "poses": 13},
)
POSE_SET_IDS = tuple(p["id"] for p in POSE_SETS)
# check-poses only: the route of the arm-held 6D zero (start -> fingertips up -> start -> two re-checks)
ZERO_POSES = "six_d_zero"
CHECK_POSE_IDS = (*POSE_SET_IDS, ZERO_POSES)
# Table in front of the base, base frame: rough 11-06 image estimate of the right arm at the table
# (edge ~0.24 m out, top ~0.12 m up), taken conservatively. The operator can change it.
DEFAULT_TABLE = {"edgeM": 0.20, "topM": 0.13, "marginMm": 50.0}
# A collect needs a passing check of the same pose set and table, this recent.
CHECK_VALID_S = 12 * 3600
LOG_LINES = 200
#: A line a step prints with this prefix becomes one of the job's warnings (shown on the page, not fatal).
WARNING_PREFIX = "WARNING: "
LIST_LIMIT = 12
#: How far back _zero_latest looks for a passing zero (about four days of 30-min dynamic zeros).
ZERO_LATEST_SCAN = 200
CAPTURE_DURATION_RANGE_S = (30.0, 900.0)
_RUN_RE = re.compile(r"^\d{8}_\d{6}_fr3_[a-z_]+\.json$")
# FR3 TCP pivots (fr3_tcp_pivot.py fit / fit-joint): a measured Desk TCP for desk-config.
PIVOT_SUBDIR = Path("outputs") / "fr3_tcp_pivot"
_PIVOT_FIT_RE = re.compile(r"^(\d{8}_\d{6}_pivot\.fit\.json|joint/\d{8}_\d{6}_joint\.json)$")
# Default Desk TCP z of the 44 mm tips (fr3_payload_identification_runtime.NEW_TIP_TCP_Z_M; estimated, not measured on the arm).
TCP_ESTIMATE_Z_M = 0.19016
TCP_TEMPLATE_Z_M = 0.17199
TCP_SOURCES = ("estimate", "pivot", "manual", "template")
_STAMP_RE = re.compile(r"^\d{8}_\d{6}")
_POSE_RE = re.compile(r"pose\s+(\d+)/(\d+)\s+(up|down|single)")


@dataclass
class Host:
    """What the module needs from the gateway."""

    repo_root: Path
    python: Callable[[], Path]
    env: Callable[[], dict[str, str]]
    resolve_arm: Callable[[str], dict[str, Any]]  # fr3_arms entry + "arm" + "robot_ip"; raises ValueError
    arms: Callable[[], list[dict[str, Any]]]
    fci_holder: Callable[[], str]  # who else drives the FR3 now, or ""
    box_port_holder: Callable[[], str]  # who else holds the BOX port (:15000) now, or ""
    log: Callable[[str, str], None]
    # the recorder, for the arm-held 6D zero (it owns the BOX and the software zero)
    recorder_box_refusal: Callable[[], str] = lambda: "网关没有接录制器"  # "" when a GMSL2 recorder is connected
    box_preview: Callable[[], dict[str, Any]] = dict  # the recorder's BOX_LIVE snapshot + updatedAt/active
    box_roster: Callable[[], list[dict[str, Any]]] = list  # BOX_DEVICES_JSON: device_id / box_id per BOX
    trigger_six_d_cali: Callable[[bool, str], dict[str, Any]] = lambda origin, box_id: {"ok": False, "error": "no recorder"}
    six_d_cali_log: Callable[[], tuple[bool, list[dict[str, Any]]]] = lambda: (False, [])


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stamp_unix(name: str) -> float | None:
    m = _STAMP_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def six_d_fit_summary(fit: Any) -> dict[str, Any] | None:
    """What the page shows of a box_six_d fit (or its refusal)."""
    if not isinstance(fit, dict):
        return None
    if fit.get("error"):
        return {"error": str(fit["error"])}
    rigid = fit.get("rigid_reading") or {}
    return {
        "meetsTarget": bool(fit.get("meets_target")),
        "heldoutWorstForceN": fit.get("heldout_worst_force_N"),
        "heldoutWorstMomentNm": fit.get("heldout_worst_moment_Nm"),
        "nStatic": fit.get("n_static"),
        "attitudes": fit.get("attitudes") or {},
        "missingAttitudes": fit.get("missing_attitudes") or [],
        "massKg": rigid.get("mass_kg"),
        "axisGainKg": rigid.get("axis_gain_kg"),
        "comInFtM": rigid.get("com_in_ft_m"),
        "K": fit.get("K"),
        "uptimeStartS": (fit.get("box_uptime_s") or {}).get("start_s"),
        "drift": _drift_summary(fit.get("drift_same_attitude")),
        "warnings": [str(w) for w in fit.get("warnings") or []],
    }


def _drift_summary(drift: Any) -> dict[str, Any] | None:
    """box_six_d's same-attitude drift, camelCased; None for a fit made before the check existed."""
    if not isinstance(drift, dict):
        return None
    return {
        "pairs": int(drift.get("pairs") or 0),
        "worstForceN": drift.get("worst_force_N"),
        "axis": drift.get("axis"),
        "changeN": drift.get("change_N"),
        "fromS": drift.get("from_s"),
        "toS": drift.get("to_s"),
        "maxAbsByAxisN": drift.get("max_abs_by_axis_N"),
    }


def run_summary(path: Path) -> dict[str, Any] | None:
    run = _read_json(path)
    if not isinstance(run, dict):
        return None
    poses = run.get("poses") or []
    config = run.get("tool_config") or ("gripper_cam_smr" if run.get("smr_present", True) else "gripper_cam")
    six_d = run.get("six_d") or None
    fit_path = path.with_name(path.name.removesuffix(".json") + ".six_d.fit.json")
    return {
        "name": path.name,
        "created": run.get("created", ""),
        "arm": run.get("arm") or "",
        "robotIp": run.get("robot_ip") or "",
        "toolConfig": config,
        "posesSet": run.get("poses_set") or "payload",
        "complete": bool(run.get("complete", True)),
        "nPoses": len(poses),
        "twoWay": bool(poses) and "tau_J_up" in poses[0],
        "note": run.get("note") or "",
        "deskAtStart": (run.get("desk_load_at_start") or {}).get("m_ee"),
        "sixD": (
            {
                "deviceId": six_d.get("device_id"),
                "log": six_d.get("log"),
                "fit": six_d_fit_summary(_read_json(fit_path)) if fit_path.is_file() else None,
            }
            if six_d
            else None
        ),
    }


def desk_summary(directory: Path) -> dict[str, Any] | None:
    report = _read_json(directory / "report.json")
    if not isinstance(report, dict):
        return None
    configs = []
    for config, entry in (report.get("configs") or {}).items():
        configs.append(
            {
                "toolConfig": config,
                "deskName": entry.get("desk_name"),
                "massKg": entry.get("mass_kg"),
                "comFlangeM": entry.get("com_flange_m"),
                "massStdG": entry.get("mass_std_g"),
                "predictedForceMeanN": entry.get("predicted_force_mean_n"),
                "predictedForceMaxN": entry.get("predicted_force_max_n"),
                "references": entry.get("references") or {},
                "cad": entry.get("cad") or {},
                "deskAtStart": entry.get("desk_at_start") or [],
                "runs": entry.get("runs") or [],
                "nPoses": entry.get("n_poses"),
                "deskJson": Path(entry.get("desk_json") or "").name,
                "tcp": entry.get("tcp"),
            }
        )
    return {"name": directory.name, "tcpSource": report.get("tcp_source"), "configs": configs}


def capture_summary(directory: Path) -> dict[str, Any] | None:
    meta = _read_json(directory / "capture.json")
    if not isinstance(meta, dict):
        return None
    status = _read_json(directory / "status.json") or {}
    return {
        "name": directory.name,
        "deviceId": meta.get("device_id"),
        "endEffector": meta.get("end_effector") or "",
        "startedUnix": meta.get("started_unix"),
        "state": status.get("state") or "",
        "faces": status.get("faces") or {},
        "rows": status.get("rows") or 0,
        "need": status.get("need"),
        "fit": six_d_fit_summary(_read_json(directory / "fit.json")) if (directory / "fit.json").is_file() else None,
        "registered": meta.get("registered"),
    }


def registry_summary(path: Path) -> list[dict[str, Any]]:
    reg = _read_json(path) or {}
    out = []
    for did, entry in (reg.get("devices") or {}).items():
        records = []
        for r in entry.get("six_d_payload") or []:
            fit = r.get("fit") or {}
            records.append(
                {
                    "validFromUnix": r.get("valid_from_unix"),
                    "validFrom": r.get("valid_from"),
                    "endEffector": r.get("end_effector"),
                    "fitted": r.get("K") is not None,
                    "status": r.get("status") or "",
                    "heldoutWorstForceN": fit.get("heldout_worst_force_N"),
                }
            )
        out.append({"deviceId": did, "role": entry.get("role"), "hand": entry.get("hand"), "records": records})
    return out


class PayloadCalibration:
    def __init__(self, host: Host) -> None:
        self.host = host
        self.lock = Lock()
        self.process: subprocess.Popen[str] | None = None
        self.job: dict[str, Any] = {}
        # the last check-poses verdicts, by (arm, pose set): what a collect is gated on
        self.checks: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- paths
    @property
    def root(self) -> Path:
        return self.host.repo_root

    def _runs_dir(self) -> Path:
        return self.root / RUNS_SUBDIR

    def _capture_dir(self) -> Path:
        return self.root / CAPTURE_SUBDIR

    def _desk_dir(self) -> Path:
        return self.root / DESK_SUBDIR

    def _tool(self, script: Path, *args: str) -> list[str]:
        return [str(self.host.python()), str(self.root / script), *args]

    # ---------------------------------------------------------------- views
    def running(self) -> bool:
        with self.lock:
            return self.job.get("state") in ("running", "fitting")

    def box_port_user(self) -> str:
        """"" unless this module's job holds the BOX port (for the gateway's Connect / replay refusals)."""
        with self.lock:
            job = dict(self.job)
        if job.get("state") != "running":
            return ""
        if job.get("kind") == "six_d_capture":
            return "6D 手持采集正在运行，它占着 BOX 端口（:15000）；先在标定页停止"
        if job.get("kind") == "collect" and job.get("sixD"):
            return "FR3 负载辨识正在运行（带 6D 记录），它占着 BOX 端口（:15000）"
        return ""

    def fci_user(self) -> str:
        with self.lock:
            job = dict(self.job)
        if job.get("state") == "running" and job.get("kind") == "collect":
            return "FR3 负载辨识正在运行，它占着 FCI"
        if job.get("state") == "running" and job.get("kind") == "six_d_zero":
            return "FR3 6D 零点任务正在运行，它在动机械臂"
        return ""

    def _job_view(self) -> dict[str, Any]:
        with self.lock:
            job = dict(self.job)
            job["log"] = list(self.job.get("log") or [])
        if job.get("kind") == "six_d_capture" and job.get("dir"):
            job["capture"] = _read_json(Path(job["dir"]) / "status.json")
        for key in ("checks", "warnings"):
            if key in job:
                job[key] = list(job[key])
        return job

    def _zero_reports(self) -> list[dict[str, Any]]:
        root = self.root / six_d_zero.REPORT_SUBDIR
        if not root.is_dir():
            return []
        out = []
        for path in sorted(root.glob("*.json"), reverse=True)[:LIST_LIMIT]:
            s = six_d_zero.report_summary(path)
            if s:
                out.append(s)
        return out

    def _zero_latest(self) -> list[dict[str, Any]]:
        """The newest passing report per (arm, BOX, mode), over the last ZERO_LATEST_SCAN reports.

        ``zeroReports`` is the newest LIST_LIMIT of any outcome, both arms mixed: at one dynamic zero per
        30 min this morning's origin falls off it by the afternoon.  The BOX tab dates each arm BOX's
        zeros from this list instead.
        """
        root = self.root / six_d_zero.REPORT_SUBDIR
        if not root.is_dir():
            return []
        latest: dict[tuple[str, str, str], dict[str, Any]] = {}
        for path in sorted(root.glob("*.json"), reverse=True)[:ZERO_LATEST_SCAN]:
            s = six_d_zero.report_summary(path)
            if not s or not s["pass"]:
                continue
            latest.setdefault((str(s["arm"]), str(s["deviceId"]), str(s["mode"])), s)
        return list(latest.values())

    def payload(self) -> dict[str, Any]:
        runs_dir, cap_dir, desk_dir = self._runs_dir(), self._capture_dir(), self._desk_dir()
        runs = []
        if runs_dir.is_dir():
            for p in sorted((p for p in runs_dir.glob("*.json") if _RUN_RE.match(p.name)), reverse=True)[:LIST_LIMIT]:
                s = run_summary(p)
                if s:
                    runs.append(s)
        desks = []
        if desk_dir.is_dir():
            for d in sorted((d for d in desk_dir.iterdir() if d.is_dir()), reverse=True)[:LIST_LIMIT]:
                s = desk_summary(d)
                if s:
                    desks.append(s)
        captures = []
        if cap_dir.is_dir():
            for d in sorted((d for d in cap_dir.iterdir() if d.is_dir()), reverse=True)[:LIST_LIMIT]:
                s = capture_summary(d)
                if s:
                    captures.append(s)
        try:
            arms = self.host.arms()
        except (OSError, ValueError):
            arms = []
        with self.lock:
            checks = {k: dict(v) for k, v in self.checks.items()}
        return {
            "ok": True,
            "error": "",
            "job": self._job_view(),
            "arms": arms,
            "toolConfigs": list(TOOL_CONFIGS),
            "poseSets": list(POSE_SETS),
            "table": dict(DEFAULT_TABLE),
            "checks": checks,
            "checkValidS": CHECK_VALID_S,
            "runs": runs,
            "desks": desks,
            "captures": captures,
            "boxes": registry_summary(self.root / REGISTRY),
            "registryPath": str(REGISTRY),
            "runsRoot": str(RUNS_SUBDIR),
            "captureRoot": str(CAPTURE_SUBDIR),
            "captureDurationRangeS": list(CAPTURE_DURATION_RANGE_S),
            "pivotFits": self._pivot_fits(),
            "tcpEstimateZM": TCP_ESTIMATE_Z_M,
            "tcpTemplateZM": TCP_TEMPLATE_Z_M,
            "zeroReports": self._zero_reports(),
            "zeroLatest": self._zero_latest(),
            "zeroPoses": ZERO_POSES,
            "zeroReportRoot": str(six_d_zero.REPORT_SUBDIR),
        }

    def refusal(self, message: str) -> dict[str, Any]:
        return {**self.payload(), "ok": False, "error": message}

    # ---------------------------------------------------------------- job plumbing
    def _append_log(self, line: str) -> None:
        with self.lock:
            log = self.job.setdefault("log", [])
            log.append(line)
            del log[:-LOG_LINES]
            if line.startswith(WARNING_PREFIX):  # a step's reminder (e.g. a cold BOX): shown, not fatal
                warnings = self.job.setdefault("warnings", [])
                text = line[len(WARNING_PREFIX) :].strip()
                if text and text not in warnings:
                    warnings.append(text)
            m = _POSE_RE.search(line)
            if m:
                index, total = int(m.group(1)), int(m.group(2))
                done = self.job.setdefault("progress", {"done": 0, "total": 0})
                # two measurements per pose: up, then down (or one: single)
                step = 2 * index + (1 if m.group(3) in ("down",) else 0) + 1
                done["total"] = total * (1 if m.group(3) == "single" else 2)
                done["done"] = max(done["done"], index + 1 if m.group(3) == "single" else step)

    def _spawn(self, kind: str, command: list[str], info: dict[str, Any], on_exit: Callable[[int], None]) -> str:
        """Start ``command`` as the job; the caller does not hold ``self.lock``. Returns a refusal or ""."""
        with self.lock:
            if self.job.get("state") in ("running", "fitting"):
                return f"已有任务在进行（{self.job.get('label') or self.job.get('kind')}），等它结束或先停止"
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.root),
                    env=self.host.env(),
                    stdin=subprocess.DEVNULL,  # an inherited ssh stdin turns non-blocking (11-06)
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,  # Stop signals this group, never the gateway
                )
            except OSError as exc:
                return str(exc)
            self.process = process
            self.job = {
                **info,
                "kind": kind,
                "state": "running",
                "startedUnixS": time.time(),
                "pid": process.pid,
                "log": [],
                "error": "",
                "command": " ".join(command[1:]),
            }
        Thread(target=self._watch, args=(process, on_exit), daemon=True, name=f"fr3-payload-{kind}").start()
        return ""

    def _watch(self, process: subprocess.Popen[str], on_exit: Callable[[int], None]) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line and not line.startswith("Testing with DEVICE"):
                self._append_log(line)
        code = process.wait()
        with self.lock:
            if self.process is process:
                self.process = None
            self.job["returncode"] = code
        try:
            on_exit(code)
        except Exception as exc:  # noqa: BLE001 -- a post-step failure must still end the job
            self._finish("error", f"{type(exc).__name__}: {exc}")

    def _finish(self, state: str, error: str = "") -> None:
        with self.lock:
            if self.job.get("stopRequested") and state == "error":
                state = "stopped"
            self.job["state"] = state
            self.job["endedUnixS"] = time.time()
            if error:
                self.job["error"] = error

    def _run_sync(self, command: list[str], timeout_s: float) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.root),
                env=self.host.env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return -1, str(exc)
        output = "\n".join(
            line for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines() if not line.startswith("Testing with DEVICE")
        ).strip()
        return proc.returncode, output

    def _stop_requested(self) -> bool:
        with self.lock:
            return bool(self.job.get("stopRequested"))

    def _run_step(self, command: list[str]) -> int:
        """One subprocess of a multi-step job (the zero job's moves): output into the log, Stop kills it."""
        with self.lock:
            if self.job.get("stopRequested"):
                return -1
            try:
                process = subprocess.Popen(
                    command, cwd=str(self.root), env=self.host.env(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
            except OSError as exc:
                self.job.setdefault("log", []).append(f"cannot start {command[1]}: {exc}")
                return -1
            self.process = process
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line and not line.startswith("Testing with DEVICE"):
                self._append_log(line)
        code = process.wait()
        with self.lock:
            if self.process is process:
                self.process = None
        return code

    def stop(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            kind = self.job.get("kind")
            stepping = kind == "six_d_zero" and self.job.get("state") == "running"
            if not stepping and (process is None or process.poll() is not None):
                return {**self.payload(), "ok": False, "error": "没有正在运行的任务"}
            self.job["stopRequested"] = True
        if process is None or process.poll() is not None:  # the zero job between moves: its thread sees the flag
            self.host.log("warn", "calibration job six_d_zero stopped from the page")
            return self.payload()
        # SIGINT first: collect's finally stops the controller and closes the BOX; a capture
        # writes its last status. SIGTERM if it does not go.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT if kind != "six_d_capture" else signal.SIGTERM)
        deadline = time.monotonic() + 8.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        self.host.log("warn", f"calibration job {kind} stopped from the page")
        return self.payload()

    # ---------------------------------------------------------------- check-poses
    @staticmethod
    def _check_key(arm: str, poses: str) -> str:
        return f"{arm or '-'}|{poses}"

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        arm = str(body.get("arm") or "")
        poses = str(body.get("poses") or POSE_SET_IDS[0])
        if poses not in CHECK_POSE_IDS:
            return self.refusal(f"未知姿态集 {poses}")
        try:
            edge = float(body.get("tableEdgeM", DEFAULT_TABLE["edgeM"]))
            top = float(body.get("tableTopM", DEFAULT_TABLE["topM"]))
            margin = float(body.get("tableMarginMm", DEFAULT_TABLE["marginMm"]))
        except (TypeError, ValueError):
            return self.refusal("桌面参数要是数字")
        if not (0.0 < edge < 1.5 and -0.5 < top < 1.0 and 10.0 <= margin <= 300.0):
            return self.refusal("桌面参数超出范围（桌沿 0–1.5 m、桌面 −0.5–1.0 m、余量 10–300 mm）")
        if arm:
            try:
                self.host.resolve_arm(arm)
            except ValueError as exc:
                return self.refusal(str(exc))
        table = {"edgeM": edge, "topM": top, "marginMm": margin}
        command = self._tool(
            PAYLOAD_SCRIPT, "check-poses", "--poses", poses, "--spacing-mm", "5",
            "--table-edge-m", f"{edge:g}", "--table-top-m", f"{top:g}", "--table-margin-mm", f"{margin:g}",
        )
        key = self._check_key(arm, poses)

        def done(code: int) -> None:
            with self.lock:
                log = list(self.job.get("log") or [])
            verdict = next((line for line in reversed(log) if line.startswith("check-poses=")), "")
            passed = code == 0 and verdict.startswith("check-poses=PASS")
            summary = [line for line in log if line.startswith(("min tool-arm", "lowest tool", "table ("))]
            with self.lock:
                self.checks[key] = {
                    "arm": arm, "poses": poses, "pass": passed, "atUnixS": time.time(), "table": table, "summary": summary,
                }
            self._finish("done" if passed else "error", "" if passed else (verdict or f"check-poses exit {code}"))

        refusal = self._spawn("check", command, {"label": "碰撞检查", "arm": arm, "poses": poses, "table": table}, done)
        return self.refusal(refusal) if refusal else self.payload()

    # ---------------------------------------------------------------- collect
    def collect(self, body: dict[str, Any]) -> dict[str, Any]:
        arm_name = str(body.get("arm") or "")
        poses = str(body.get("poses") or POSE_SET_IDS[0])
        config = str(body.get("toolConfig") or "")
        six_d = bool(body.get("sixD"))
        note = " ".join(str(body.get("note") or "").split())[:120]
        if poses not in POSE_SET_IDS:
            return self.refusal(f"未知姿态集 {poses}")
        if config not in TOOL_CONFIG_IDS:
            return self.refusal("先选末端配置（装了什么）")
        try:
            target = self.host.resolve_arm(arm_name)
        except ValueError as exc:
            return self.refusal(str(exc))
        check = self.checks.get(self._check_key(arm_name, poses))
        if not check or not check.get("pass"):
            return self.refusal("先对这条臂、这组姿态跑一次碰撞检查（含桌面），通过后才能采集")
        if time.time() - float(check.get("atUnixS") or 0) > CHECK_VALID_S:
            return self.refusal(f"碰撞检查已超过 {CHECK_VALID_S // 3600} 小时，重新检查一次")
        holder = self.host.fci_holder()
        if holder:
            return self.refusal(f"{holder}，它占着 FR3 的 FCI；先停掉再采集")
        device_id = target.get("corenetic_device_id")
        if six_d:
            if device_id is None:
                return self.refusal(f"{target.get('arm')} 臂没有登记夹爪 BOX（fr3_arms.json corenetic_device_id），无法记录 6D")
            holder = self.host.box_port_holder()
            if holder:
                return self.refusal(holder)
        self._runs_dir().mkdir(parents=True, exist_ok=True)
        suffix = {"gripper_cam_smr": "smr_in", "gripper_cam": "smr_out", "gripper": "gripper"}[config]
        run = self._runs_dir() / f"{datetime.now():%Y%m%d_%H%M%S}_fr3_{suffix}.json"
        command = self._tool(
            PAYLOAD_SCRIPT, "collect", "--robot-ip", str(target["robot_ip"]), "--tool-config", config,
            "--poses", poses, "--note", note, "--out", str(run),
            "--yes",  # the page's confirmation replaces the terminal prompt
        )
        if target.get("arm"):
            command[3:3] = ["--arm", str(target["arm"])]
        if six_d:
            command += ["--six-d-device-id", str(device_id)]

        def done(code: int) -> None:
            if code != 0 or not run.is_file():
                self._finish("error", f"collect exit {code}")
                return
            if six_d:
                with self.lock:
                    self.job["state"] = "fitting"
                self._fit_six_d_log(
                    run.with_name(run.name.removesuffix(".json") + ".six_d.log"),
                    run.with_name(run.name.removesuffix(".json") + ".six_d.fit.json"),
                )
            self._finish("done")
            self.host.log("info", f"FR3 payload run {run.name} complete")

        info = {
            "label": "负载辨识采集", "arm": target.get("arm") or "", "robotIp": target["robot_ip"], "poses": poses,
            "toolConfig": config, "sixD": six_d, "deviceId": device_id if six_d else None, "run": run.name,
            "progress": {"done": 0, "total": 0},
        }
        refusal = self._spawn("collect", command, info, done)
        if refusal:
            return self.refusal(refusal)
        self.host.log("warn", f"FR3 payload collect on {target['robot_ip']} ({poses}, {config}) -> {run.name}")
        return self.payload()

    # ---------------------------------------------------------------- arm-held 6D zero
    def zero_start(self, body: dict[str, Any]) -> dict[str, Any]:
        arm_name = str(body.get("arm") or "")
        mode = str(body.get("mode") or "dynamic")
        if mode not in six_d_zero.MODES:
            return self.refusal(f"未知校零方式 {mode}")
        if not body.get("confirmed"):
            return self.refusal("先在页面上逐项确认准备清单")
        try:
            target = self.host.resolve_arm(arm_name)
        except ValueError as exc:
            return self.refusal(str(exc))
        device_id = target.get("corenetic_device_id")
        if device_id is None:
            return self.refusal(f"{target.get('arm')} 臂没有登记夹爪 BOX（fr3_arms.json corenetic_device_id）")
        check = self.checks.get(self._check_key(arm_name, ZERO_POSES))
        if not check or not check.get("pass"):
            return self.refusal("先对这条臂跑一次「零点路线」碰撞检查（含桌面），通过后才能校零")
        if time.time() - float(check.get("atUnixS") or 0) > CHECK_VALID_S:
            return self.refusal(f"零点路线的碰撞检查已超过 {CHECK_VALID_S // 3600} 小时，重新检查一次")
        holder = self.host.fci_holder()
        if holder:
            return self.refusal(f"{holder}，它占着 FR3 的 FCI；先停掉")
        refusal = self.host.recorder_box_refusal()
        if refusal:
            return self.refusal(f"{refusal}（零点要经录制器做：软件零点存在录制器进程里）")
        box_id = six_d_zero.box_id_for(self.host.box_roster() or [], int(device_id))
        if box_id is None:
            return self.refusal(f"录制器没有连上 BOX {device_id}（或认不准它的 box_id）；Connect 时它要在线")
        first = six_d_zero.sample(self.host.box_preview() or {}, box_id, time.time())
        if first is None:
            return self.refusal(f"读不到 BOX {device_id} 的实时 6D/IMU 数据")
        # a cold BOX is not refused: the job warns (report + panel) and goes on
        running, _ = self.host.six_d_cali_log()
        if running:
            return self.refusal("BOX 页签的 6D 校准正在进行，等它结束")
        k, k_record = None, {}
        registry = _read_json(self.root / REGISTRY)
        if isinstance(registry, dict):
            k, k_record = six_d_zero.payload_k(registry, int(device_id), time.time())
        ctx = {
            "arm": str(target.get("arm") or ""), "robot_ip": str(target["robot_ip"]), "device_id": int(device_id),
            "box_id": box_id, "mode": mode, "k": k, "k_record": k_record, "script": PAYLOAD_SCRIPT,
        }
        with self.lock:
            if self.job.get("state") in ("running", "fitting"):
                return self.refusal(f"已有任务在进行（{self.job.get('label') or self.job.get('kind')}），等它结束或先停止")
            self.job = {
                "kind": "six_d_zero", "label": "6D 零点（臂上，指尖朝上）", "state": "running",
                "startedUnixS": time.time(), "log": [], "error": "", "arm": ctx["arm"], "robotIp": ctx["robot_ip"],
                "deviceId": int(device_id), "mode": mode, "phase": "准备", "prompt": "", "checks": [], "warnings": [],
                "recheck": k is not None,
            }
        job = six_d_zero.ZeroJob(self, ctx)
        Thread(target=job.run, daemon=True, name="fr3-six-d-zero").start()
        self.host.log("warn", f"FR3 6D zero on {ctx['robot_ip']} (BOX {device_id}, {mode}) started from the page")
        return self.payload()

    # ---------------------------------------------------------------- desk-config
    def desk(self, body: dict[str, Any]) -> dict[str, Any]:
        names = body.get("runs") or []
        if not isinstance(names, list) or not names or len(names) > 8:
            return self.refusal("选 1–8 条采集")
        paths = []
        for name in names:
            name = str(name)
            if not _RUN_RE.match(name) or not (self._runs_dir() / name).is_file():
                return self.refusal(f"没有这条采集：{name}")
            paths.append(self._runs_dir() / name)
        desk_names = body.get("names") or {}
        if not isinstance(desk_names, dict):
            return self.refusal("names 要是 {配置: Desk 名}")
        out_dir = self._desk_dir() / f"{datetime.now():%Y%m%d_%H%M%S}"
        command = self._tool(PAYLOAD_SCRIPT, "desk-config", *map(str, paths), "--out-dir", str(out_dir),
                             "--report", str(out_dir / "report.json"))
        for config, name in desk_names.items():
            name = " ".join(str(name).split())[:60]
            if config in TOOL_CONFIG_IDS and name:
                command += ["--name", f"{config}={name}"]
        tcp_args, refusal = self._tcp_args(body)
        if refusal:
            return self.refusal(refusal)
        command += tcp_args
        code, output = self._run_sync(command, 300)
        if code != 0 or not (out_dir / "report.json").is_file():
            return {**self.refusal(output.splitlines()[-1] if output else f"desk-config exit {code}"), "output": output}
        self.host.log("info", f"Desk load configs written to {out_dir}")
        return {**self.payload(), "deskWritten": out_dir.name, "output": output}

    def _pivot_fits(self) -> list[dict[str, Any]]:
        """FR3 pivot fits, newest first: what each says the socket centre is, for the Desk TCP."""
        root = self.root / PIVOT_SUBDIR
        if not root.is_dir():
            return []
        paths = sorted(root.glob("*_pivot.fit.json"), reverse=True)[:LIST_LIMIT]
        paths += sorted((root / "joint").glob("*_joint.json"), reverse=True)[:LIST_LIMIT]
        out = []
        for path in paths:
            name = path.relative_to(root).as_posix()
            if not _PIVOT_FIT_RE.match(name):
                continue
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            fk = rep.get("fk") or {}
            if "s_ee_mm" not in fk:
                continue
            out.append(
                {
                    "name": name,
                    "joint": name.startswith("joint/"),
                    "created": str(rep.get("created") or name[:15]),
                    "robotIp": str(rep.get("robot_ip") or ""),
                    "note": str(rep.get("note") or ""),
                    "frame": str(rep.get("frame") or ""),
                    "sEeMm": fk.get("s_ee_mm"),
                    "sigmaMm": fk.get("s_sigma_mm"),
                    "residualRmsMm": fk.get("residual_rms_mm"),
                }
            )
        return sorted(out, key=lambda f: f["name"].rsplit("/", 1)[-1], reverse=True)

    def _tcp_args(self, body: dict[str, Any]) -> tuple[list[str], str]:
        """desk-config TCP arguments for the request, or a refusal message."""
        tcp = body.get("tcp") if isinstance(body.get("tcp"), dict) else {}
        source = str(tcp.get("source") or "")
        if not source and body.get("tcpZM") not in (None, ""):  # the first panel sent only a z
            source, tcp = "manual", {"zM": body.get("tcpZM")}
        source = source or "estimate"
        if source not in TCP_SOURCES:
            return [], f"未知的 TCP 来源：{source}"
        if source == "estimate":
            return [], ""
        if source == "template":
            return ["--tcp-template"], ""
        if source == "manual":
            try:
                z = float(tcp.get("zM"))
            except (TypeError, ValueError):
                return [], "TCP z 要是数字（米）"
            if not 0.05 <= z <= 0.35:
                return [], "TCP z 要在 0.05–0.35 m"
            return ["--tcp-z-m", f"{z:g}"], ""
        name = str(tcp.get("pivot") or "")
        path = self.root / PIVOT_SUBDIR / name
        if not _PIVOT_FIT_RE.match(name) or not path.is_file():
            return [], f"没有这个 pivot 拟合：{name}"
        return ["--tcp-from-pivot", str(path)], ""

    def desk_file(self, desk: str, name: str) -> Path | None:
        """A desk-config JSON to download (``desk/<stamp>/<file>.json``), or None."""
        if not re.fullmatch(r"\d{8}_\d{6}", desk) or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.json", name):
            return None
        path = self._desk_dir() / desk / name
        return path if path.is_file() else None

    # ---------------------------------------------------------------- 6D: hand-held capture
    def capture_start(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            device_id = int(str(body.get("deviceId") or "").strip())
            duration = float(body.get("durationS") or 300.0)
        except (TypeError, ValueError):
            return self.refusal("BOX device_id 要是整数")
        lo, hi = CAPTURE_DURATION_RANGE_S
        if not lo <= duration <= hi:
            return self.refusal(f"时长要在 {lo:.0f}–{hi:.0f} s")
        end_effector = " ".join(str(body.get("endEffector") or "").split())[:160]
        holder = self.host.box_port_holder()
        if holder:
            return self.refusal(holder)
        directory = self._capture_dir() / f"{datetime.now():%Y%m%d_%H%M%S}_box{device_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "capture.json").write_text(
            json.dumps(
                {"device_id": device_id, "end_effector": end_effector, "started_unix": round(time.time(), 3),
                 "mode": "hand-held free air"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = self._tool(CAPTURE_SCRIPT, "--device-id", str(device_id), "--out", str(directory),
                             "--duration-s", f"{duration:g}")

        def done(code: int) -> None:
            if code != 0 or not (directory / "free_air.log").is_file():
                status = _read_json(directory / "status.json") or {}
                self._finish("error", status.get("error") or f"capture exit {code}")
                return
            with self.lock:
                self.job["state"] = "fitting"
            self._fit_six_d_log(directory / "free_air.log", directory / "fit.json")
            self._finish("done")

        refusal = self._spawn(
            "six_d_capture", command,
            {"label": "6D 手持采集", "deviceId": device_id, "dir": str(directory), "name": directory.name,
             "durationS": duration},
            done,
        )
        return self.refusal(refusal) if refusal else self.payload()

    def _fit_six_d_log(self, log: Path, out: Path) -> dict[str, Any] | None:
        code, output = self._run_sync(self._tool(SIX_D_SCRIPT, "fit", str(log), "--json", str(out)), 120)
        fit = _read_json(out)
        summary = six_d_fit_summary(fit)
        if summary is None:
            self._append_log(f"6D fit failed: {output.splitlines()[-1] if output else code}")
        elif summary.get("error"):
            self._append_log(f"6D fit refused: {summary['error']}")
        else:
            for w in summary.get("warnings") or []:
                self._append_log(f"{WARNING_PREFIX}{w}")
            self._append_log(
                f"6D fit: held-out worst {summary['heldoutWorstForceN']} N / {summary['heldoutWorstMomentNm']} N*m "
                f"over {summary['attitudes']} -> {'meets' if summary['meetsTarget'] else 'MISSES'} the 0.3 N target"
            )
        return summary

    def _six_d_source(self, source: str) -> tuple[Path, Path, dict[str, Any]] | None:
        """``capture:<dir>`` or ``run:<file>`` -> (log, fit path, meta)."""
        kind, _, name = source.partition(":")
        if kind == "capture" and re.fullmatch(r"\d{8}_\d{6}_box\d+", name):
            d = self._capture_dir() / name
            meta = _read_json(d / "capture.json")
            if isinstance(meta, dict):
                return d / "free_air.log", d / "fit.json", meta
        if kind == "run" and _RUN_RE.match(name):
            path = self._runs_dir() / name
            run = _read_json(path)
            if isinstance(run, dict) and run.get("six_d"):
                base = path.name.removesuffix(".json")
                meta = {
                    "device_id": run["six_d"].get("device_id"),
                    "end_effector": "",
                    "started_unix": _stamp_unix(name),
                    "mode": f"held by the FR3 ({run.get('arm') or run.get('robot_ip')}), {run.get('poses_set')} poses",
                    "tool_config": run.get("tool_config"),
                }
                return path.with_name(base + ".six_d.log"), path.with_name(base + ".six_d.fit.json"), meta
        return None

    def refit(self, body: dict[str, Any]) -> dict[str, Any]:
        src = self._six_d_source(str(body.get("source") or ""))
        if src is None or not src[0].is_file():
            return self.refusal("没有这份 6D 记录")
        if self.running():
            return self.refusal("有任务在进行，等它结束")
        self._fit_six_d_log(src[0], src[1])
        return self.payload()

    def register(self, body: dict[str, Any]) -> dict[str, Any]:
        source = str(body.get("source") or "")
        src = self._six_d_source(source)
        if src is None or not src[1].is_file():
            return self.refusal("这份 6D 记录还没有拟合结果")
        log, fit_path, meta = src
        summary = six_d_fit_summary(_read_json(fit_path)) or {}
        if summary.get("error"):
            return self.refusal(f"拟合被拒：{summary['error']}")
        end_effector = " ".join(str(body.get("endEffector") or meta.get("end_effector") or "").split())[:160]
        if not end_effector:
            return self.refusal("写明 6D 传感器这次带的是什么（末端配置）")
        try:
            valid_from = float(body.get("validFromUnix"))
        except (TypeError, ValueError):
            return self.refusal("填这个末端配置的装上时间（不是拟合时间）")
        if valid_from > time.time() + 60 or valid_from < 1.7e9:
            return self.refusal("装上时间不对（未来或太早）")
        device_id = str(body.get("deviceId") or meta.get("device_id") or "")
        status = " ".join(str(body.get("status") or "").split())[:300] or f"fitted from the calibration page ({meta.get('mode')})"
        command = self._tool(
            SIX_D_SCRIPT, "register", str(fit_path), "--device-id", device_id, "--end-effector", end_effector,
            "--valid-from-unix", f"{valid_from:.0f}", "--status", status,
            "--source", f"{log.relative_to(self.root)} ({meta.get('mode')})", "--registry", str(self.root / REGISTRY),
        )
        if body.get("allowUnmet"):
            command.append("--allow-unmet")
        code, output = self._run_sync(command, 60)
        if code != 0:
            return {**self.refusal(output.splitlines()[-1] if output else f"register exit {code}"), "output": output}
        if source.startswith("capture:"):
            cap = fit_path.parent / "capture.json"
            data = _read_json(cap) or {}
            data["registered"] = {"at_unix": round(time.time(), 3), "valid_from_unix": int(valid_from)}
            cap.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.host.log("warn", f"BOX {device_id}: 6D payload record written to {REGISTRY} (copy it back to the repo)")
        return {**self.payload(), "registered": device_id, "output": output}
