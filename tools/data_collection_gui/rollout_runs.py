# ruff: noqa: N815  -- record fields are the JSON keys the Rollout page reads
"""The Rollout page's backend: checkpoints, offline prediction, online preview, their records.

Kept out of gateway.py and outside ``state.lock``, like training_runs. Jobs are detached processes
(tools/fr3/rollout_relative_tcp.py) followed through their log; one at a time, because the online
preview shares this machine's GPU and the recorder's frame bus with the offline ones.

Stages (rollout_relative_tcp.STAGES): offline prediction, online preview and the execution plan check are
delivered; execution is not, and nothing here moves an arm. The preview needs the recorder connected
(its cameras and BOX are the inputs) and is refused in a sandbox, which has no hardware.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT = "tools/fr3/rollout_relative_tcp.py"
PROGRESS_PREFIX = "FR3_ROLLOUT "
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")
STAGES = (
    {"id": "offline", "label": "Offline prediction on recordings", "status": "available"},
    {"id": "preview", "label": "Online preview (live cameras + BOX, arm still)", "status": "available"},
    {"id": "plan", "label": "Execution plan check: IK, training workspace, collision (no motion)", "status": "available"},
    {"id": "execute", "label": "Execution on the arm", "status": "not delivered"},
)


class RolloutError(RuntimeError):
    """Something the operator can fix; the gateway answers 4xx."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RolloutJob:
    id: str
    kind: str  # offline | preview | replay
    checkpoint: str
    state: str = "running"  # running | done | failed | stopped
    message: str = ""
    pid: int | None = None
    startedAt: str = ""
    finishedAt: str = ""
    logPath: str = ""
    out: str = ""  # result json (offline) or record dir (preview / replay)
    params: dict[str, Any] = field(default_factory=dict)
    last: dict[str, Any] = field(default_factory=dict)
    logOffset: int = 0


class RolloutManager:
    def __init__(
        self,
        repo_root: Path,
        *,
        exports_root: Path,
        profile: str = "thor",
        sandbox: str | None = None,
        recorder_busy: Callable[[], bool] = lambda: False,
        gateway_url: str = "http://127.0.0.1:8765",
        python: str = "",
        popen=subprocess.Popen,
    ):
        self.repo_root = Path(repo_root)
        self.exports_root = Path(exports_root)
        self.profile = profile
        self.sandbox = sandbox
        self.recorder_busy = recorder_busy
        self.gateway_url = gateway_url
        self._popen = popen
        self.python = python or next(
            (str(self.repo_root / rel) for rel in (".venv-fr3/bin/python", ".venv/bin/python") if (self.repo_root / rel).exists()),
            "python3",
        )
        self.root = self.repo_root / "outputs" / "rollout"
        self.lock = threading.Lock()
        self.jobs: dict[str, RolloutJob] = {}
        try:
            for item in json.loads((self.root / "jobs.json").read_text(encoding="utf-8")):
                job = RolloutJob(**item)
                self.jobs[job.id] = job
        except (OSError, ValueError, TypeError):
            pass

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "jobs.json.tmp"
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs.values()], indent=2), encoding="utf-8")
        tmp.replace(self.root / "jobs.json")

    # -------------------------------------------------------- checkpoints ---
    def checkpoints(self) -> list[dict[str, Any]]:
        out = []
        runs = self.repo_root / "outputs" / "training" / "runs"
        for contract_path in sorted(runs.glob("*/lerobot/checkpoints/*/pretrained_model/fr3_policy_contract.json")):
            model = contract_path.parent
            if model.parent.name == "last":
                continue
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            prov: dict[str, Any] = {}
            with contextlib.suppress(OSError, ValueError):
                prov = json.loads((model / "fr3_provenance.json").read_text(encoding="utf-8"))
            out.append({
                "path": str(model),
                "run": model.parents[3].name,
                "step": int(model.parent.name) if model.parent.name.isdigit() else model.parent.name,
                "hand": contract.get("hand"),
                "cameras": [c.get("name") for c in contract.get("cameras") or []],
                "wrench": bool(contract.get("wrench_names")),
                "exports": [e.get("name") for e in prov.get("exports") or []],
                "reloadCheck": prov.get("reload_check"),
                "hasWorkspace": bool(prov.get("workspace")),
                "saved": prov.get("saved", ""),
            })
        return sorted(out, key=lambda c: (c["run"], c["step"] if isinstance(c["step"], int) else 0), reverse=True)

    def _checkpoint(self, path: str) -> Path:
        model = Path(path).resolve()
        runs = (self.repo_root / "outputs" / "training" / "runs").resolve()
        if runs not in model.parents or not (model / "fr3_policy_contract.json").is_file():
            raise RolloutError(f"not a relative-TCP checkpoint on this machine: {path}")
        return model

    def _export(self, name: str) -> Path:
        root = (self.exports_root / name).resolve()
        if root.parent != self.exports_root.resolve() or not (root / "meta" / "info.json").is_file():
            raise RolloutError(f"no export {name!r}")
        return root

    # --------------------------------------------------------------- jobs ---
    def _active(self) -> RolloutJob | None:
        self.refresh()
        return next((j for j in self.jobs.values() if j.state == "running"), None)

    def _spawn(self, kind: str, checkpoint: str, args: list[str], out: str, params: dict[str, Any]) -> RolloutJob:
        busy = self._active()
        if busy is not None:
            raise RolloutError(f"{busy.kind} {busy.id} is still running; stop it first")
        job_id = f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"
        log = self.root / "logs" / f"{job_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as handle:
            proc = self._popen(
                [self.python, SCRIPT, *args],
                cwd=self.repo_root,
                env={**os.environ, "PYTHONPATH": "src:.", "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
        job = RolloutJob(id=job_id, kind=kind, checkpoint=checkpoint, pid=int(proc.pid), startedAt=_now(),
                         logPath=str(log), out=out, params=params)
        with self.lock:
            self.jobs[job_id] = job
            self._save()
        return job

    def start_offline(self, checkpoint: str, export: str, episodes: list[int], stride: int) -> RolloutJob:
        model = self._checkpoint(checkpoint)
        root = self._export(export)
        if not 1 <= int(stride) <= 600:
            raise RolloutError("stride: 1..600 frames")
        out = self.root / "results" / f"offline_{datetime.now():%Y%m%d_%H%M%S}.json"
        args = ["offline", "--checkpoint", str(model), "--export", str(root), "--stride", str(int(stride)), "--out", str(out)]
        for ep in episodes:
            args += ["--episode", str(int(ep))]
        return self._spawn("offline", str(model), args, str(out), {"export": export, "episodes": episodes, "stride": stride})

    def start_preview(self, checkpoint: str, arm: str, rate: float, read_joints: bool) -> RolloutJob:
        model = self._checkpoint(checkpoint)
        if self.sandbox:
            raise RolloutError("a sandbox has no cameras or BOXes: the online preview runs on the station's own gateway")
        if self.profile != "thor":
            raise RolloutError("the online preview reads the Thor recorder's cameras; this gateway is not a Thor")
        if not self.recorder_busy():
            raise RolloutError("connect the recorder first (Live Record): its cameras and BOX are the policy's inputs")
        if arm not in ("right", "left"):
            raise RolloutError("pick the arm")
        if not 0.2 <= float(rate) <= 5.0:
            raise RolloutError("rate: 0.2..5 Hz (the preview frame bus publishes about 5 Hz)")
        record = self.root / "records" / f"preview_{datetime.now():%Y%m%d_%H%M%S}"
        args = ["preview", "--checkpoint", str(model), "--record", str(record), "--arm", arm,
                "--gateway", self.gateway_url, "--rate", str(float(rate))]
        if read_joints:
            args.append("--read-joints")
        return self._spawn("preview", str(model), args, str(record), {"arm": arm, "rate": rate, "readJoints": read_joints})

    def start_replay(self, record: str) -> RolloutJob:
        path = Path(record).resolve()
        if (self.root / "records").resolve() not in path.parents or not (path / "record.json").is_file():
            raise RolloutError(f"no preview record {record!r}")
        meta = json.loads((path / "record.json").read_text(encoding="utf-8"))
        return self._spawn("replay", str(meta.get("checkpoint", "")), ["replay", "--record", str(path)], str(path), {})

    def stop(self, job_id: str) -> str:
        job = self.jobs.get(job_id)
        if job is None or job.state != "running" or job.pid is None:
            raise RolloutError(f"{job_id} is not running")
        if _pid_runs(job.pid, SCRIPT):
            os.killpg(int(job.pid), signal.SIGTERM)
        job.state = "stopped"
        job.finishedAt = _now()
        job.message = "stopped by the operator"
        with self.lock:
            self._save()
        return job.message

    # ------------------------------------------------------------- status ---
    def refresh(self) -> None:
        changed = False
        for job in self.jobs.values():
            if job.state != "running":
                continue
            changed |= self._read_log(job)
            if job.pid and not _pid_runs(job.pid, SCRIPT):
                self._read_log(job)
                if job.last.get("event") == "done":
                    job.state = "done"
                    job.message = job.message or "done"
                else:
                    job.state = "stopped" if job.kind == "preview" and not job.last.get("error") else "failed"
                    job.message = job.last.get("error") or job.message or "exited; see the log"
                job.finishedAt = _now()
                changed = True
        if changed:
            with self.lock:
                self._save()

    def _read_log(self, job: RolloutJob) -> bool:
        try:
            with open(job.logPath, "rb") as handle:
                handle.seek(job.logOffset)
                data = handle.read()
        except OSError:
            return False
        cut = data.rfind(b"\n")
        if cut < 0:
            return False
        job.logOffset += cut + 1
        for line in data[: cut + 1].decode("utf-8", "replace").splitlines():
            if not line.startswith(PROGRESS_PREFIX):
                continue
            try:
                event = json.loads(line[len(PROGRESS_PREFIX):])
            except ValueError:
                continue
            job.last = event
            if event.get("event") == "error":
                job.message = str(event.get("error"))
            elif event.get("event") == "wait":
                job.message = f"waiting: {event.get('error')}"
            elif event.get("event") == "preview":
                job.message = f"step {event.get('step')}, {event.get('latency_ms', 0):.0f} ms per prediction"
            elif event.get("event") == "done":
                job.message = "done"
        return True

    def status(self) -> dict[str, Any]:
        self.refresh()
        with self.lock:
            jobs = [asdict(j) for j in sorted(self.jobs.values(), key=lambda j: j.startedAt, reverse=True)]
        for job in jobs:
            job.pop("logOffset", None)
        return {
            "stages": list(STAGES),
            "jobs": jobs,
            "profile": self.profile,
            "sandbox": self.sandbox,
            "recorderConnected": bool(self.recorder_busy()),
        }

    def log_tail(self, job_id: str, lines: int = 120) -> list[str]:
        job = self.jobs.get(job_id)
        if job is None:
            raise RolloutError(f"no job {job_id!r}")
        try:
            text = Path(job.logPath).read_bytes()[-120_000:].decode("utf-8", "replace")
        except OSError:
            return []
        return [ln.split("\r")[-1] for ln in text.splitlines()][-lines:]

    def results(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((self.root / "results").glob("*.json"), reverse=True):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({
                "path": str(path), "kind": r.get("kind"), "created": r.get("created"), "checkpoint": r.get("checkpoint"),
                "export": Path(str(r.get("export", ""))).name, "frames": r.get("frames"),
                "trainedOnThisExport": r.get("trained_on_this_export"),
                "posMm": (r.get("error") or {}).get("pos_mm", {}).get("mean"),
                "rotDeg": (r.get("error") or {}).get("rot_deg", {}).get("mean"),
                "stillMm": (r.get("not_moving_baseline_pos_mm") or {}).get("mean"),
                "viewVsOnlineMm": (r.get("view_vs_online_path_mm") or {}).get("max"),
            })
        for path in sorted((self.root / "records").glob("*/record.json"), reverse=True):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({
                "path": str(path.parent), "kind": "preview", "created": r.get("created"), "checkpoint": r.get("checkpoint"),
                "arm": r.get("arm"), "steps": r.get("steps"), "replayCheck": r.get("replay_check"),
                "planProblem": r.get("plan_problem"),
            })
        return out

    def result(self, path: str) -> dict[str, Any]:
        p = Path(path).resolve()
        if (self.root / "results").resolve() not in p.parents:
            raise RolloutError("not a rollout result")
        return json.loads(p.read_text(encoding="utf-8"))

    def preview_latest(self, record: str) -> dict[str, Any]:
        p = Path(record).resolve()
        if (self.root / "records").resolve() not in p.parents:
            raise RolloutError("not a preview record")
        try:
            return json.loads((p / "latest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def preview_image(self, record: str) -> bytes:
        p = Path(record).resolve()
        if (self.root / "records").resolve() not in p.parents:
            raise RolloutError("not a preview record")
        return (p / "latest.jpg").read_bytes()


def _pid_runs(pid: int, needle: str) -> bool:
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return needle in cmdline
