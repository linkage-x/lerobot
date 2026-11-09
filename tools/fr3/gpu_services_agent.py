#!/usr/bin/env python
"""Start, stop and watch the GPU services of a policy host (Dante's 5090), for the GUI on the robot host.

The card is shared by services that do not fit beside each other comfortably: this repo's policy
server (`run/policy_server.sh`, ~11 GB once a pi05 is loaded) and servers from other checkouts,
such as the WAM VLA-JEPA server the NUC reaches through the R4 bridge. Switching between them was
a shell session on the GPU host; this agent lets the Rollout page do it.

What the agent may run is fixed on the GPU host, in `outputs/gpu_services.json` (git-ignored, per
machine). A client names a service and an action -- start, stop, or switch to it (stop the others,
then start it) -- and never sends a command:

    {"allowFrom": ["127.0.0.1", "10.77.0.156"],
     "services": [{
        "id": "wam_jepa",
        "label": "WAM VLA-JEPA J1 s0 (Zenoh :8791)",
        "unitGlob": "wam-p3-jepa-serve-*",          # the systemd user unit(s) the service runs as
        "cwd": "/home/hph/Codes/lerobot-main-1029",
        "env": {"UV_CACHE_DIR": "/tmp/codex-wam-uv"},
        "start": ["/home/hph/.local/bin/uv", "run", "...", "serve_jepa", "..."],
        "port": 8791,                                # listening = ready; optional
        "gpuGb": 12                                  # free memory a start needs
     }]}

The policy server is built in (id `policy_server`). A service is found by its unit (any unit
matching `unitGlob`, so one started by hand counts too), and stopped with `systemctl --user stop`.
A start runs the configured argv in a transient unit of its own (`fr3-gpu-launch-<id>-<time>`),
so restarting this agent never takes a service down with it.

Refused here, whatever the client: stopping the policy server while a rollout holds its policy,
starting anything while a training run is on this machine, starting without the free memory the
service needs (a switch frees it first). A policy server stopped here is parked
(`run/logs/policy_server_parked`), so a deploy's `policy_server.sh ensure` does not start it again
beside the service that now has the card. The robot host's gateway adds its own refusal while a
rollout runs there. Every action is logged to `run/logs/gpu_services.log`.

    PYTHONPATH=src:. .venv-fr3/bin/python tools/fr3/gpu_services_agent.py [--port 8792]

`run/gpu_services_agent.sh` runs it as the user unit `fr3-gpu-services-agent`. Stdlib only, apart
from the policy server's status probe (torch's serialiser), imported when first needed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8792
CONFIG_PATH = Path("outputs/gpu_services.json")
LOG_PATH = Path("run/logs/gpu_services.log")
POLICY_SERVER = "policy_server"
POLICY_SCRIPT = "tools/fr3/fr3_policy_server.py"
POLICY_PORT = 8790
# pi05 resident plus its first forwards (11.7 GB measured on the 5090, 11-09).
POLICY_GPU_GB = 11.0
# tools/fr3/fr3_policy_server.py's TRAINING_MODULES / TRAINING_SCRIPTS, restated so the agent does
# not import torch and lerobot to read /proc; a test holds them equal.
TRAINING_MODULES = frozenset({"lerobot.scripts.lerobot_train"})
TRAINING_SCRIPTS = frozenset({"lerobot_train.py", "lerobot-train", "fr3_train_il_policy.py"})
ACTIONS = ("start", "stop", "switch")
_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_GLOB_RE = re.compile(r"^[A-Za-z0-9_.@*-]{1,120}$")
# Memory of a stopped service is gone within seconds; a switch waits this long for it.
FREE_WAIT_S = 30.0
# tools.fr3.policy_remote.WARMUP_HOLDER: the server's own startup load, not a rollout; a test holds them equal.
WARMUP_HOLDER = "warmup"


class ServiceError(Exception):
    """A request refused for a reason the operator can act on (HTTP 409)."""


@dataclass
class Service:
    id: str
    label: str
    unit_glob: str = ""
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    start: list[str] = field(default_factory=list)
    port: int | None = None
    gpu_gb: float = 0.0
    start_timeout_s: float = 120.0


@dataclass
class Config:
    allow_from: list[str]
    services: list[Service]

    def service(self, service_id: str) -> Service:
        for svc in self.services:
            if svc.id == service_id:
                return svc
        raise ServiceError(f"unknown service {service_id!r}; this host has: {', '.join(s.id for s in self.services)}")


def policy_service() -> Service:
    return Service(
        id=POLICY_SERVER,
        label=f"FR3 policy server (:{POLICY_PORT})",
        port=POLICY_PORT,
        gpu_gb=POLICY_GPU_GB,
    )


def parse_config(raw: dict[str, Any]) -> Config:
    """The configured services after the built-in policy server; anything malformed raises ValueError."""
    allow = [str(a) for a in raw.get("allowFrom", ["127.0.0.1"])]
    services = [policy_service()]
    for entry in raw.get("services", []):
        sid = str(entry.get("id", ""))
        if not _ID_RE.match(sid) or sid == POLICY_SERVER or any(s.id == sid for s in services):
            raise ValueError(f"service id {sid!r}: lower-case letters, digits and _ only, unique, not {POLICY_SERVER}")
        glob = str(entry.get("unitGlob", ""))
        if not _GLOB_RE.match(glob):
            raise ValueError(f"{sid}: unitGlob {glob!r} is required (systemd unit name pattern)")
        start = entry.get("start")
        if not isinstance(start, list) or not start or not all(isinstance(a, str) and a for a in start):
            raise ValueError(f"{sid}: start must be a non-empty argv list of strings")
        if not os.path.isabs(start[0]):
            raise ValueError(f"{sid}: start[0] must be an absolute path (a user unit has a bare PATH)")
        cwd = str(entry.get("cwd", ""))
        if not os.path.isabs(cwd):
            raise ValueError(f"{sid}: cwd must be an absolute path")
        env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
        port = entry.get("port")
        services.append(
            Service(
                id=sid,
                label=str(entry.get("label") or sid),
                unit_glob=glob,
                cwd=cwd,
                env=env,
                start=list(start),
                port=int(port) if port is not None else None,
                gpu_gb=float(entry.get("gpuGb", 0.0)),
                start_timeout_s=float(entry.get("startTimeoutS", 120.0)),
            )
        )
    return Config(allow_from=allow, services=services)


def load_config(path: Path) -> Config:
    if not path.exists():
        return parse_config({})
    return parse_config(json.loads(path.read_text()))


# --------------------------------------------------------------------------- the machine ---


def _run(args: list[str], timeout: float = 15.0, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)


def _cmdline(pid: int, proc_root: Path = Path("/proc")) -> list[str]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "ignore") for a in raw.split(b"\0") if a]


def training_processes(proc_root: Path = Path("/proc")) -> list[str]:
    """As `fr3_policy_server.training_processes`: one 'pid N (what)' per kind of training running."""
    runs: dict[str, int] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        args = _cmdline(int(entry.name), proc_root)
        if not args or not os.path.basename(args[0]).startswith("python"):
            continue
        what = next((a for a in args[1:] if a in TRAINING_MODULES or os.path.basename(a) in TRAINING_SCRIPTS), None)
        if what is not None:
            what = os.path.basename(what)
            runs[what] = min(runs.get(what, int(entry.name)), int(entry.name))
    return sorted(f"pid {pid} ({what})" for what, pid in runs.items())


class Machine:
    """Everything the agent reads from or does to the host; a test replaces it."""

    def __init__(self, repo_root: Path = REPO_ROOT, proc_root: Path = Path("/proc")):
        self.repo_root = repo_root
        self.proc_root = proc_root

    def gpu(self) -> dict[str, Any]:
        """The first card (the agent's host has one): totals and each compute process's memory."""
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"])
        name, total, used = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        apps = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        procs = []
        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                procs.append({"pid": int(parts[0]), "mb": int(parts[1])})
        return {"name": name, "total_mb": int(total), "used_mb": int(used), "procs": procs}

    def units(self, glob: str) -> list[dict[str, Any]]:
        """Loaded user units matching `glob` that are not inactive."""
        out = _run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "--all", f"{glob}.service"])
        units = []
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4 or not fnmatch.fnmatch(parts[0], f"{glob}.service"):
                continue
            unit, active = parts[0], parts[2]
            if active in ("inactive", "failed"):
                continue
            show = _run(["systemctl", "--user", "show", unit, "-p", "MainPID", "-p", "ActiveEnterTimestampMonotonic", "-p", "ActiveState"])
            props = dict(line.split("=", 1) for line in show.stdout.splitlines() if "=" in line)
            since_us = int(props.get("ActiveEnterTimestampMonotonic") or 0)
            up_s = time.clock_gettime(time.CLOCK_MONOTONIC) - since_us / 1e6 if since_us else None
            units.append({"unit": unit, "state": props.get("ActiveState", active), "main_pid": int(props.get("MainPID") or 0), "up_s": up_s})
        return units

    def unit_of(self, pid: int) -> str:
        """The systemd unit (service or scope) pid `pid` runs in, '' if none is readable."""
        try:
            text = (self.proc_root / str(pid) / "cgroup").read_text()
        except OSError:
            return ""
        for line in text.splitlines():
            path = line.rsplit(":", 1)[-1]
            for part in reversed(path.split("/")):
                if part.endswith((".service", ".scope")):
                    return part
        return ""

    def cmdline(self, pid: int) -> list[str]:
        return _cmdline(pid, self.proc_root)

    def policy_pids(self) -> list[int]:
        pids = []
        for entry in self.proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            args = _cmdline(int(entry.name), self.proc_root)
            if args and os.path.basename(args[0]).startswith("python") and any(a.endswith(POLICY_SCRIPT) for a in args[1:]):
                pids.append(int(entry.name))
        return sorted(pids)

    def process_up_s(self, pid: int) -> float | None:
        try:
            start_ticks = int((self.proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
            uptime = float((self.proc_root / "uptime").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")

    def listening(self, port: int) -> bool:
        out = _run(["ss", "-ltnH", f"sport = :{port}"])
        return bool(out.stdout.strip())

    def policy_status(self) -> dict[str, Any] | None:
        try:
            from tools.fr3.policy_remote import query_policy_server

            return query_policy_server(f"127.0.0.1:{POLICY_PORT}", "status", timeout_s=2.0).get("status") or {}
        except Exception:
            return None

    def training(self) -> list[str]:
        return training_processes(self.proc_root)

    # ---- actions

    def start_policy_server(self) -> str:
        # A scope of its own: the server outlives the script that starts it, and this agent.
        unit = f"fr3-policy-server-{int(time.time())}"
        out = _run(
            ["systemd-run", "--user", "--scope", "--collect", "--quiet", f"--unit={unit}",
             "bash", "run/policy_server.sh", "start", "--port", str(POLICY_PORT), "--warm-last"],
            timeout=90.0,
            cwd=self.repo_root,
        )
        if out.returncode != 0:
            raise ServiceError(f"policy server did not start: {(out.stderr or out.stdout).strip()[-600:]}")
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "started"

    def stop_policy_server(self, why: str) -> str:
        # Parked first: a deploy's `policy_server.sh ensure` then leaves it stopped instead of
        # starting it (and warming a policy) beside the service the card was given to.
        parked = self.repo_root / "run/logs/policy_server_parked"
        parked.parent.mkdir(parents=True, exist_ok=True)
        parked.write_text(f"{datetime.now().isoformat(timespec='seconds')} {why}")
        out = _run(["bash", "run/policy_server.sh", "stop"], timeout=30.0, cwd=self.repo_root)
        if out.returncode != 0:
            raise ServiceError(f"policy server stop failed: {(out.stderr or out.stdout).strip()[-600:]}")
        return out.stdout.strip() or "stopped"

    def launch(self, svc: Service, log_path: Path) -> str:
        log_path = (self.repo_root / log_path).resolve()  # systemd wants an absolute path
        unit = f"fr3-gpu-launch-{svc.id.replace('_', '-')}-{int(time.time())}"
        args = ["systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}", f"--working-directory={svc.cwd}",
                "-p", f"StandardOutput=append:{log_path}", "-p", f"StandardError=append:{log_path}"]
        for key, value in svc.env.items():
            args += ["--setenv", f"{key}={value}"]
        out = _run(args + ["--", *svc.start], timeout=30.0)
        if out.returncode != 0:
            raise ServiceError(f"{svc.id}: launch failed: {(out.stderr or out.stdout).strip()[-600:]}")
        return unit

    def stop_units(self, units: list[str]) -> None:
        if units:
            _run(["systemctl", "--user", "stop", *units], timeout=60.0)


# ------------------------------------------------------------------------------ the agent ---


def _ago(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds, 1)


def _checkpoint_label(spec: dict[str, Any] | None) -> str:
    path = str((spec or {}).get("checkpoint") or "")
    match = re.search(r"train/([^/]+)/checkpoints/([^/]+)", path)
    return f"{match.group(1)}/{match.group(2)}" if match else path


class Agent:
    def __init__(self, config: Config, machine: Machine, log_path: Path, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.machine = machine
        self.log_path = log_path
        self.clock = clock
        self.sleep = sleep
        self._op_lock = threading.RLock()  # request() plans (and so snapshots) under it
        self._op: dict[str, Any] | None = None

    def log(self, line: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")

    # ---- reading

    def services_state(self, gpu: dict[str, Any]) -> tuple[list[dict[str, Any]], set[int]]:
        procs = {p["pid"]: p["mb"] for p in gpu["procs"]}
        claimed: set[int] = set()
        out = []
        for svc in self.config.services:
            row: dict[str, Any] = {"id": svc.id, "label": svc.label, "port": svc.port, "gpuGb": svc.gpu_gb}
            if svc.id == POLICY_SERVER:
                pids = self.machine.policy_pids()
                gpu_pids = [p for p in pids if p in procs]
                listening = bool(pids) and self.machine.listening(POLICY_PORT)
                status = self.machine.policy_status() if listening else None
                loaded = (status or {}).get("loaded")
                row.update(
                    state="stopped" if not pids else ("running" if listening else "starting"),
                    pids=pids,
                    upS=_ago(self.machine.process_up_s(pids[0])) if pids else None,
                    loaded=_checkpoint_label(loaded.get("spec")) if loaded else None,
                    loading=bool((status or {}).get("loading")),
                    holder=(status or {}).get("holder"),
                )
            else:
                units = self.machine.units(svc.unit_glob)
                gpu_pids = [p for p in procs if fnmatch.fnmatch(self.machine.unit_of(p), f"{svc.unit_glob}.service")]
                if not units:
                    state = "stopped"
                elif any(u["state"] == "deactivating" for u in units):
                    state = "stopping"
                elif svc.port is not None and not self.machine.listening(svc.port):
                    state = "starting"
                else:
                    state = "running"
                row.update(
                    state=state,
                    units=[u["unit"] for u in units],
                    pids=[u["main_pid"] for u in units if u["main_pid"]],
                    upS=_ago(units[0]["up_s"]) if units else None,
                )
            row["gpuMb"] = sum(procs[p] for p in gpu_pids)
            claimed.update(gpu_pids)
            out.append(row)
        return out, claimed

    def snapshot(self) -> dict[str, Any]:
        gpu = self.machine.gpu()
        services, claimed = self.services_state(gpu)
        others = []
        for proc in gpu["procs"]:
            if proc["pid"] in claimed:
                continue
            args = self.machine.cmdline(proc["pid"])
            others.append({"pid": proc["pid"], "mb": proc["mb"], "what": " ".join(args)[:160] or "?", "unit": self.machine.unit_of(proc["pid"])})
        with self._op_lock:
            op = dict(self._op) if self._op else None
        return {
            "ok": True,
            "host": socket.gethostname(),
            "gpu": {k: gpu[k] for k in ("name", "total_mb", "used_mb")},
            "services": services,
            "others": others,
            "training": self.machine.training(),
            "op": op,
        }

    # ---- acting

    def _free_mb(self) -> int:
        gpu = self.machine.gpu()
        return gpu["total_mb"] - gpu["used_mb"]

    def plan(self, action: str, service_id: str) -> list[tuple[str, Service]]:
        """The steps `action` on `service_id` takes, checked now; raises ServiceError when refused."""
        if action not in ACTIONS:
            raise ServiceError(f"unknown action {action!r}; one of {', '.join(ACTIONS)}")
        target = self.config.service(service_id)
        snap = self.snapshot()
        rows = {row["id"]: row for row in snap["services"]}
        running = [s for s in self.config.services if rows[s.id]["state"] != "stopped"]

        def check_stop(svc: Service) -> None:
            holder = rows[svc.id].get("holder")
            if svc.id == POLICY_SERVER and holder and holder != WARMUP_HOLDER:
                raise ServiceError(f"the policy server is serving {rows[svc.id]['holder']}; stop that rollout first")

        if action == "stop":
            if rows[target.id]["state"] == "stopped":
                raise ServiceError(f"{target.id} is not running")
            check_stop(target)
            return [("stop", target)]
        if rows[target.id]["state"] != "stopped":
            if action == "start":
                raise ServiceError(f"{target.id} is already {rows[target.id]['state']}")
        if snap["training"]:
            raise ServiceError(f"a training run has this card ({', '.join(snap['training'])}); start nothing beside it")
        steps: list[tuple[str, Service]] = []
        if action == "switch":
            for svc in running:
                if svc.id != target.id:
                    check_stop(svc)
                    steps.append(("stop", svc))
            if rows[target.id]["state"] == "stopped":
                steps.append(("start", target))
            if not steps:
                raise ServiceError(f"{target.id} already runs alone")
            return steps
        free_mb = snap["gpu"]["total_mb"] - snap["gpu"]["used_mb"]
        if free_mb < target.gpu_gb * 1024:
            busy = ", ".join(f"{r['id']} {r['gpuMb'] / 1024:.1f} GB" for r in snap["services"] if r["gpuMb"]) or "other processes"
            raise ServiceError(
                f"{target.id} needs {target.gpu_gb:.0f} GB, {free_mb / 1024:.1f} GB free ({busy}); use switch to stop the others first"
            )
        return [("start", target)]

    def request(self, action: str, service_id: str, client: str) -> dict[str, Any]:
        """Check and begin `action`; it runs in the background and `snapshot()['op']` follows it."""
        with self._op_lock:
            if self._op and self._op["state"] == "running":
                raise ServiceError(f"busy: {self._op['action']} {self._op['service']} is still running")
            steps = self.plan(action, service_id)
            self._op = {
                "action": action,
                "service": service_id,
                "state": "running",
                "message": "; ".join(f"{verb} {svc.id}" for verb, svc in steps),
                "startedAt": self.clock(),
                "finishedAt": None,
            }
            op = dict(self._op)
        self.log(f"{client} {action} {service_id}: {op['message']}")
        threading.Thread(target=self._execute, args=(steps,), name="gpu-op", daemon=True).start()
        return op

    def _finish(self, state: str, message: str) -> None:
        with self._op_lock:
            assert self._op is not None
            self._op.update(state=state, message=message, finishedAt=self.clock())
        self.log(f"  -> {state}: {message}")

    def _execute(self, steps: list[tuple[str, Service]]) -> None:
        notes = []
        try:
            for verb, svc in steps:
                notes.append(self._stop(svc) if verb == "stop" else self._start(svc))
        except ServiceError as exc:
            self._finish("failed", "; ".join([*notes, str(exc)]))
            return
        except Exception as exc:  # an agent bug must still end the op, or every later request is "busy"
            self._finish("failed", "; ".join([*notes, f"{type(exc).__name__}: {exc}"]))
            return
        self._finish("done", "; ".join(notes))

    def _refusal(self) -> str:
        """The reason a preflight wrapper (WAM's resource_guard) printed for refusing, if it did."""
        try:
            tail = self.log_path.read_text()[-6000:]
        except OSError:
            return ""
        found = re.findall(r'"refused":\s*\[([^\]]*)\]', tail)
        reasons = re.findall(r'"([^"]+)"', found[-1]) if found else []
        return f" (refused: {'; '.join(reasons)})" if reasons else ""

    def _stop(self, svc: Service) -> str:
        before = self._free_mb()
        if svc.id == POLICY_SERVER:
            with self._op_lock:
                why = f"{self._op['action']} {self._op['service']} from the GPU services panel" if self._op else "GPU services panel"
            self.machine.stop_policy_server(why)
        else:
            launchers = self.machine.units(f"fr3-gpu-launch-{svc.id.replace('_', '-')}-*")
            self.machine.stop_units([u["unit"] for u in self.machine.units(svc.unit_glob)] + [u["unit"] for u in launchers])
            deadline = self.clock() + FREE_WAIT_S
            while self.machine.units(svc.unit_glob):
                if self.clock() > deadline:
                    raise ServiceError(f"{svc.id}: its unit is still up {FREE_WAIT_S:.0f} s after the stop")
                self.sleep(0.5)
        # The driver releases the memory a moment after the process exits.
        deadline = self.clock() + FREE_WAIT_S
        while any(row["id"] == svc.id and row["gpuMb"] for row in self.services_state(self.machine.gpu())[0]):
            if self.clock() > deadline:
                raise ServiceError(f"{svc.id}: still holds GPU memory {FREE_WAIT_S:.0f} s after the stop")
            self.sleep(0.5)
        return f"stopped {svc.id} (+{(self._free_mb() - before) / 1024:.1f} GB free)"

    def _start(self, svc: Service) -> str:
        free_mb = self._free_mb()
        if free_mb < svc.gpu_gb * 1024:
            raise ServiceError(f"{svc.id} needs {svc.gpu_gb:.0f} GB, only {free_mb / 1024:.1f} GB free")
        if svc.id == POLICY_SERVER:
            return f"policy server: {self.machine.start_policy_server()}"
        unit = self.machine.launch(svc, self.log_path)
        t0 = self.clock()
        while not self.machine.units(svc.unit_glob):
            # The launcher (a preflight wrapper, say) lives as long as the service; gone means it gave up.
            if self.clock() - t0 > 5.0 and not self.machine.units(unit):
                raise ServiceError(f"{svc.id}: launcher {unit} exited before the service came up{self._refusal()}; see {self.log_path}")
            if self.clock() - t0 > svc.start_timeout_s:
                raise ServiceError(f"{svc.id}: no unit matching {svc.unit_glob} after {svc.start_timeout_s:.0f} s; see {self.log_path}")
            self.sleep(1.0)
        return f"started {svc.id} (launcher {unit})"


# ------------------------------------------------------------------------------ http ---


def make_handler(agent: Agent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "fr3-gpu-services/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # the access log would be one line per poll
            pass

        def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self) -> bool:
            if self.client_address[0] in agent.config.allow_from:
                return True
            self._reply(HTTPStatus.FORBIDDEN, {"ok": False, "error": f"{self.client_address[0]} is not in allowFrom"})
            return False

        def do_GET(self) -> None:
            if not self._allowed():
                return
            if self.path.split("?")[0] != "/status":
                self._reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown route"})
                return
            try:
                self._reply(HTTPStatus.OK, agent.snapshot())
            except Exception as exc:
                self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            if not self._allowed():
                return
            if self.path != "/action":
                self._reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown route"})
                return
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 4096)
                body = json.loads(self.rfile.read(length) or b"{}")
                op = agent.request(str(body.get("action", "")), str(body.get("service", "")), self.client_address[0])
            except ServiceError as exc:
                self._reply(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
                return
            except (ValueError, TypeError) as exc:
                self._reply(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._reply(HTTPStatus.ACCEPTED, {"ok": True, "op": op})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    agent = Agent(config, Machine(), LOG_PATH)
    agent.log(f"agent up on :{args.port}; services {', '.join(s.id for s in config.services)}; allowFrom {', '.join(config.allow_from)}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent))
    print(f"[INFO] gpu_services_agent listening={args.host}:{args.port} services={[s.id for s in config.services]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
