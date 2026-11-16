# ruff: noqa: N815  -- record fields are the JSON keys the Training page reads
"""The Training page's backend: where a run trains, starting / stopping / resuming it, following it.

Kept out of gateway.py. Nothing here takes ``state.lock``: the gateway calls in from its handlers, and a
run outlives any one request (and any one gateway: runs are detached processes, and their state is
re-read from their log and pid on every status call).

A run trains with tools/fr3/train_relative_tcp.py, either here or on a remote machine over ssh:

* **here** ("local"): the gateway's own tree and outputs/exports. On a Thor the GPU also serves the
  cameras, so a local run is refused while the recorder is connected, and the gateway refuses Connect
  while a local run trains (:meth:`TrainingManager.local_run_active`).
* **remote**: ``user@host`` and a directory used only for training there. Before each start the code this
  gateway runs is copied into it (rsync --delete; the directory is marked on first use and a directory
  that is not marked is never written into) and so are the chosen exports. Checkpoints stay there until
  **Fetch** copies the run back (without optimiser state) so Rollout here can load it.

Every run has a record in outputs/training/gui_runs.json and a log in outputs/training/logs/<job>.log.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRAIN_SCRIPT = "tools/fr3/train_relative_tcp.py"
PROGRESS_PREFIX = "FR3_PROGRESS "
SSH_OPTS = ("-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
HOST_MARKER = ".lerobot_training_host"
#: What a remote training directory needs of this tree. Not outputs/ (its own runs live there), not
#: environments, not git.
SYNC_EXCLUDES = (
    "/outputs/", "/.git/", ".venv*/", "node_modules/", "__pycache__/", "/third_party/",
    "/tools/data_collection_gui/frontend/", "/run/logs/", "*.mkv", "*.mp4",
    # Excluded so --delete keeps it: it is what says the directory belongs to this page.
    f"/{HOST_MARKER}",
)
LOCAL_PYTHON_CANDIDATES = (".venv-fr3/bin/python", ".venv/bin/python")
ACTIVE_STATES = ("syncing", "starting", "running")
PROBE_SNIPPET = (
    "import json,shutil,sys\n"
    "out={'python':sys.version.split()[0],'ffmpeg':bool(shutil.which('ffmpeg'))}\n"
    "try:\n import torch;out['torch']=torch.__version__;out['cuda']=torch.cuda.is_available()\n"
    " out['gpu']=torch.cuda.get_device_name(0) if out['cuda'] else ''\n"
    " out['gpu_mem_gb']=round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if out['cuda'] else 0\n"
    "except Exception as e: out['torch_error']=repr(e)\n"
    "for m in ('lerobot','torchcodec','av'):\n"
    " try: __import__(m);out[m]=True\n except Exception as e: out[m]=False;out[m+'_error']=repr(e)[:200]\n"
    "print('PROBE '+json.dumps(out))\n"
)


class TrainingError(RuntimeError):
    """Something the operator can fix; the gateway answers 4xx with this message."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class TrainingHost:
    id: str
    label: str
    kind: str  # local | remote
    sshTarget: str = ""
    repoDir: str = ""
    python: str = ""


@dataclass
class RunRecord:
    job: str
    hostId: str
    hostLabel: str
    kind: str
    state: str = "starting"  # syncing | starting | running | done | failed | stopped | lost
    message: str = ""
    pid: int | None = None
    logPath: str = ""
    runDir: str = ""  # on the training host
    localRunDir: str = ""  # here, once fetched (or always, for local runs)
    startedAt: str = ""
    finishedAt: str = ""
    step: int = 0
    totalSteps: int = 0
    loss: float | None = None
    lastCheckpoint: str = ""
    reloadCheck: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    exports: list[str] = field(default_factory=list)
    resumes: list[str] = field(default_factory=list)
    fetchedAt: str = ""
    logOffset: int = 0


def validate_ssh_target(target: str) -> str:
    target = (target or "").strip()
    if not SSH_TARGET_RE.match(target):
        raise TrainingError(f"a remote host is user@host (got {target!r}); it builds ssh commands, so nothing else")
    return target


def validate_remote_dir(path: str) -> str:
    path = (path or "").strip().rstrip("/")
    if not path.startswith("/") or any(ch in path for ch in "'\"\\$`\n ;&|"):
        raise TrainingError(f"remote directory must be an absolute path without shell characters (got {path!r})")
    if path in ("/", "/home", "/root", "/tmp"):
        raise TrainingError(f"{path} cannot be a training directory")
    return path


def suggest_crop(corner_points: list[tuple[float, float]], size: tuple[int, int], *, margin: float = 0.2,
                 aspect: float = 4 / 3) -> list[int]:
    """A crop around where the gripper rig was seen: 0.5-99.5 percentile box, ``margin`` on each side, then
    widened to ``aspect`` and clipped to the image. A starting point for the operator, who checks the preview.
    """
    import numpy as np

    w, h = size
    if not corner_points:
        return [0, 0, w, h]
    pts = np.asarray(corner_points, dtype=float)
    x0, x1 = np.percentile(pts[:, 0], [0.5, 99.5])
    y0, y1 = np.percentile(pts[:, 1], [0.5, 99.5])
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - margin * bw, x1 + margin * bw
    y0, y1 = y0 - margin * bh, y1 + margin * bh
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0, y1 - y0
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    bw, bh = min(bw, w), min(bh, h)
    x0 = min(max(0.0, cx - bw / 2), w - bw)
    y0 = min(max(0.0, cy - bh / 2), h - bh)
    box = [int(round(x0)), int(round(y0)), int(round(x0 + bw)), int(round(y0 + bh))]
    box[0] -= box[0] % 2
    box[1] -= box[1] % 2
    return box


class TrainingManager:
    def __init__(
        self,
        repo_root: Path,
        *,
        exports_root: Path,
        datasets_root: Path,
        profile: str = "thor",
        sandbox: str | None = None,
        recorder_busy: Callable[[], bool] = lambda: False,
        popen=subprocess.Popen,
        run=subprocess.run,
    ):
        self.repo_root = Path(repo_root)
        self.exports_root = Path(exports_root)
        self.datasets_root = Path(datasets_root)
        self.profile = profile
        self.sandbox = sandbox
        self.recorder_busy = recorder_busy
        self._popen = popen
        self._run = run
        self.root = self.repo_root / "outputs" / "training"
        self.logs = self.root / "logs"
        self.lock = threading.Lock()
        self.runs: dict[str, RunRecord] = {}
        self._remote_polled: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------ storage ---
    @property
    def _runs_file(self) -> Path:
        return self.root / "gui_runs.json"

    @property
    def _hosts_file(self) -> Path:
        return self.root / "hosts.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self._runs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        for item in raw if isinstance(raw, list) else []:
            try:
                rec = RunRecord(**item)
            except TypeError:
                continue
            self.runs[rec.job] = rec

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._runs_file.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(r) for r in self.runs.values()], indent=2), encoding="utf-8")
        tmp.replace(self._runs_file)

    # -------------------------------------------------------------- hosts ---
    def _local_python(self) -> str:
        for rel in LOCAL_PYTHON_CANDIDATES:
            if (self.repo_root / rel).exists():
                return str(self.repo_root / rel)
        return "python3"

    def hosts(self) -> list[TrainingHost]:
        label = "This machine" + (" (Thor: refused while the recorder is connected)" if self.profile == "thor" else "")
        hosts = [TrainingHost("local", label, "local", python=self._local_python())]
        try:
            raw = json.loads(self._hosts_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        for item in raw if isinstance(raw, list) else []:
            try:
                hosts.append(TrainingHost(**item))
            except TypeError:
                continue
        return hosts

    def host(self, host_id: str) -> TrainingHost:
        for host in self.hosts():
            if host.id == host_id:
                return host
        raise TrainingError(f"no training host {host_id!r}")

    def add_host(self, label: str, ssh_target: str, repo_dir: str, python: str) -> TrainingHost:
        target = validate_ssh_target(ssh_target)
        repo_dir = validate_remote_dir(repo_dir)
        python = (python or "").strip() or ".venv/bin/python"
        if any(ch in python for ch in "'\"\\$`\n ;&|"):
            raise TrainingError("python path must not contain shell characters")
        host = TrainingHost(
            id=re.sub(r"[^A-Za-z0-9]+", "-", f"{target}-{repo_dir}").strip("-").lower(),
            label=(label or target).strip()[:60],
            kind="remote",
            sshTarget=target,
            repoDir=repo_dir,
            python=python,
        )
        remotes = [h for h in self.hosts() if h.kind == "remote" and h.id != host.id] + [host]
        self.root.mkdir(parents=True, exist_ok=True)
        self._hosts_file.write_text(json.dumps([asdict(h) for h in remotes], indent=2), encoding="utf-8")
        return host

    def remove_host(self, host_id: str) -> None:
        remotes = [h for h in self.hosts() if h.kind == "remote" and h.id != host_id]
        self._hosts_file.write_text(json.dumps([asdict(h) for h in remotes], indent=2), encoding="utf-8")

    def _ssh(self, host: TrainingHost, command: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        try:
            return self._run(
                ["ssh", *SSH_OPTS, host.sshTarget, command],
                capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrainingError(f"{host.sshTarget} did not answer within {timeout:.0f} s") from exc

    def probe(self, host_id: str) -> dict[str, Any]:
        host = self.host(host_id)
        if host.kind == "local":
            cmd = [host.python, "-c", PROBE_SNIPPET]
            result = self._run(cmd, capture_output=True, text=True, timeout=120, cwd=self.repo_root,
                               env={**os.environ, "PYTHONPATH": "src:."}, stdin=subprocess.DEVNULL)
        else:
            result = self._ssh(
                host,
                f"mkdir -p {shlex.quote(host.repoDir)} && cd {shlex.quote(host.repoDir)} && "
                f"PYTHONPATH=src:. {shlex.quote(host.python)} -c {shlex.quote(PROBE_SNIPPET)}",
                timeout=120,
            )
        line = next((ln for ln in (result.stdout or "").splitlines() if ln.startswith("PROBE ")), "")
        if not line:
            detail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            return {"ok": False, "host": host.id, "error": " / ".join(detail) or f"exit {result.returncode}"}
        data = json.loads(line[6:])
        ready = bool(data.get("cuda")) and data.get("lerobot") and data.get("ffmpeg") and (data.get("torchcodec") or data.get("av"))
        return {"ok": True, "host": host.id, "ready": bool(ready), **data}

    # ------------------------------------------------------------ exports ---
    def exports(self) -> list[dict[str, Any]]:
        from tools.fr3 import relative_tcp_data as rd
        from tools.thor.gmsl2 import pose_columns as pc

        out = []
        if not self.exports_root.is_dir():
            return out
        for root in sorted(self.exports_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            info_path = root / "meta" / "info.json"
            if not info_path.is_file():
                continue
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            hands = pc.tracked_hands(info.get("features") or {})
            problems = []
            for hand in hands or ["right"]:
                _, bad, _ = rd.summarize_export(root, hand)
                problems += bad
            gate = info.get("qc_gate")
            out.append({
                "name": root.name,
                "path": str(root),
                "episodes": int(info.get("total_episodes") or 0),
                "frames": int(info.get("total_frames") or 0),
                "fps": int(info.get("fps") or 0),
                "hands": hands,
                "cameras": sorted(k.removeprefix("observation.images.") for k in (info.get("features") or {})
                                  if k.startswith("observation.images.")),
                "qcGated": isinstance(gate, dict) and gate.get("enforced") is True,
                "worldFrameId": str((info.get("world_frame") or {}).get("world_frame_id") or ""),
                "labelsFingerprint": str((info.get("label_provenance") or {}).get("fingerprint") or ""),
                "usable": not problems,
                "problems": sorted(set(problems)),
                "modified": datetime.fromtimestamp(info_path.stat().st_mtime, UTC).isoformat(timespec="seconds"),
            })
        return out

    def _export_root(self, name: str) -> Path:
        root = (self.exports_root / name).resolve()
        if root.parent != self.exports_root.resolve() or not (root / "meta" / "info.json").is_file():
            raise TrainingError(f"no export {name!r}")
        return root

    def check(self, names: list[str], hand: str) -> dict[str, Any]:
        from tools.fr3 import relative_tcp_data as rd

        return rd.check_exports([self._export_root(n) for n in names], hand).to_dict()

    def camera_frame(self, export: str, camera: str, crop: list[int] | None, at: float = 0.5) -> bytes:
        """A JPEG of one frame of ``camera`` in ``export`` with ``crop`` drawn on it (crop preview)."""
        if not re.match(r"^cam_\d\d$", camera):
            raise TrainingError("camera is cam_NN")
        root = self._export_root(export)
        videos = sorted((root / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
        if not videos:
            raise TrainingError(f"{export} has no {camera}")
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        duration = int(info.get("total_frames") or 1) / float(info.get("fps") or 30)
        vf = []
        if crop:
            x0, y0, x1, y1 = (int(v) for v in crop)
            vf.append(f"drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:color=lime@0.9:t=6")
        vf.append("scale=960:-2")
        result = self._run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, min(at, 0.99)) * duration:.2f}", "-i", str(videos[0]),
             "-frames:v", "1", "-vf", ",".join(vf), "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
            capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0 or not result.stdout:
            raise TrainingError(f"could not read a frame of {camera}: {(result.stderr or b'').decode()[-200:]}")
        return result.stdout

    def crop_suggestion(self, export: str, camera: str, hand: str) -> dict[str, Any]:
        """Where the gripper rig was seen by ``camera`` in the recordings behind ``export``."""
        root = self._export_root(export)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        feat = (info.get("features") or {}).get(f"observation.images.{camera}")
        if not feat:
            raise TrainingError(f"{export} has no {camera}")
        size = (int(feat["shape"][1]), int(feat["shape"][0]))
        if camera == "cam_10":
            return {"crop": [0, 0, *size], "points": 0, "note": "cam_10 rides on the gripper: whole frame"}
        sources = json.loads((root / "meta" / "export_sources.json").read_text(encoding="utf-8"))
        wanted: dict[str, set[int]] = {}
        for ep in sources.get("episodes") or []:
            try:
                wanted.setdefault(str(ep["session"]), set()).add(int(str(ep["source_episode"]).split("_")[-1]))
            except (KeyError, ValueError):
                continue
        csv.field_size_limit(1 << 30)
        points: list[tuple[float, float]] = []
        missing = []
        for session, episodes in wanted.items():
            path = (self.datasets_root / session / "derived" / "april_cube_tracking_in_robot_base"
                    / f"cube_pose.{hand}.{camera}.csv")
            if not path.is_file():
                missing.append(session)
                continue
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("cube_detected") != "1" or int(row.get("episode_index") or -1) not in episodes:
                        continue
                    for marker in json.loads(row.get("detected_marker_corners_px") or "[]"):
                        points += [(float(x), float(y)) for x, y in marker.get("points") or []]
        note = f"{len(points)} rig corners from {len(wanted) - len(missing)} recording(s)"
        if missing:
            note += f"; not on this machine: {', '.join(missing)}"
        if not points:
            note += "; no rig seen by this camera, so the whole frame"
        return {"crop": suggest_crop(points, size), "points": len(points), "size": list(size), "note": note}

    # --------------------------------------------------------------- runs ---
    def local_run_active(self) -> bool:
        self.refresh()
        return any(r.kind == "local" and r.state in ACTIVE_STATES for r in self.runs.values())

    def _check_local_allowed(self) -> None:
        if self.profile == "thor" and self.sandbox:
            raise TrainingError(
                "a sandbox does not train on this Thor: its GPU serves production's cameras. Add a remote host."
            )
        if self.profile == "thor" and self.recorder_busy():
            raise TrainingError(
                "the recorder is connected: on a Thor the GPU also serves the cameras. Disconnect it first, or "
                "train on a remote host."
            )

    def start(self, params: dict[str, Any]) -> RunRecord:
        job = str(params.get("jobName") or "").strip() or f"{params.get('policy', 'act')}_{params.get('hand', 'x')}_{datetime.now():%Y%m%d_%H%M%S}"
        if not JOB_RE.match(job):
            raise TrainingError("run name: letters, digits, . _ - (up to 80)")
        with self.lock:
            if job in self.runs:
                raise TrainingError(f"a run named {job} exists; pick another name or resume it")
        host = self.host(str(params.get("hostId") or "local"))
        exports = [str(n) for n in params.get("exports") or []]
        roots = [self._export_root(n) for n in exports]
        check = self.check(exports, str(params.get("hand") or ""))
        if not check["ok"]:
            raise TrainingError("; ".join(check["problems"]))
        if host.kind == "local":
            self._check_local_allowed()
        args = self._train_args(params, exports if host.kind == "remote" else [str(r) for r in roots], job, host)
        rec = RunRecord(job=job, hostId=host.id, hostLabel=host.label, kind=host.kind,
                        state="syncing" if host.kind == "remote" else "starting",
                        startedAt=_now(), params=params, exports=exports,
                        totalSteps=int(params.get("steps") or 20000),
                        logPath=str(self.logs / f"{job}.log"))
        with self.lock:
            self.runs[job] = rec
            self._save()
        threading.Thread(target=self._launch, args=(rec, host, roots, args), daemon=True).start()
        return rec

    def _train_args(self, params: dict[str, Any], exports: list[str], job: str, host: TrainingHost) -> list[str]:
        def num(key: str, default: int, lo: int, hi: int) -> int:
            try:
                value = int(params.get(key, default))
            except (TypeError, ValueError):
                raise TrainingError(f"{key} must be a whole number") from None
            if not lo <= value <= hi:
                raise TrainingError(f"{key} must be between {lo} and {hi}")
            return value

        hand = str(params.get("hand") or "")
        if hand not in ("right", "left"):
            raise TrainingError("pick the hand (right / left)")
        args = [TRAIN_SCRIPT, "--hand", hand, "--job-name", job, "--policy", "act"]
        for name in exports:
            args += ["--export", f"outputs/exports/{name}" if host.kind == "remote" else name]
        cameras = params.get("cameras") or []
        if not cameras:
            raise TrainingError("pick the cameras")
        for cam in cameras:
            name = str(cam.get("name") or "")
            if not re.match(r"^cam_\d\d$", name):
                raise TrainingError(f"camera {name!r}")
            crop = cam.get("crop")
            if crop:
                crop = [int(v) for v in crop]
                if len(crop) != 4:
                    raise TrainingError(f"{name}: a crop is x0,y0,x1,y1")
                args += ["--camera", f"{name}:{','.join(str(v) for v in crop)}"]
            else:
                args += ["--camera", name]
        args += [
            "--steps", str(num("steps", 20000, 10, 2_000_000)),
            "--batch-size", str(num("batchSize", 32, 1, 512)),
            "--horizon", str(num("horizon", 32, 1, 256)),
            "--n-action-steps", str(num("nActionSteps", 16, 1, 256)),
            "--frame-stride", str(num("frameStride", 2, 1, 30)),
            "--history", str(num("history", 2, 0, 16)),
            "--image-long-side", str(num("imageLongSide", 320, 64, 1024)),
            "--save-freq", str(num("saveFreq", 2000, 10, 1_000_000)),
            "--num-workers", str(num("numWorkers", 8, 0, 32)),
            "--log-freq", "50",
        ]
        if params.get("wrench"):
            args.append("--wrench")
        if params.get("imageAug"):
            args.append("--image-aug")
        if params.get("box"):
            box = str(params["box"])
            if not re.match(r"^[A-Za-z0-9_./-]*$", box):
                raise TrainingError("BOX prefix")
            args += ["--box", box]
        return args

    def _spawn_local(self, rec: RunRecord, host: TrainingHost, args: list[str]) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        log = open(rec.logPath, "ab")  # noqa: SIM115 -- handed to the child
        try:
            proc = self._popen(
                [host.python, *args],
                cwd=self.repo_root,
                env={**os.environ, "PYTHONPATH": "src:.", "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        rec.pid = int(proc.pid)
        rec.runDir = str(self.root / "runs" / rec.job)
        rec.localRunDir = rec.runDir

    def _remote_command(self, host: TrainingHost, args: list[str], job: str) -> str:
        q = shlex.quote
        log = f"outputs/training/logs/{job}.log"
        pidfile = f"outputs/training/pids/{job}.pid"
        return (
            f"cd {q(host.repoDir)} && mkdir -p outputs/training/logs outputs/training/pids && "
            f"(PYTHONPATH=src:. PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 setsid {q(host.python)} "
            + " ".join(q(a) for a in args)
            + f" >> {q(log)} 2>&1 < /dev/null & echo $! > {q(pidfile)}; cat {q(pidfile)})"
        )

    def _sync_remote(self, host: TrainingHost, roots: list[Path], rec: RunRecord) -> None:
        q = shlex.quote
        check = self._ssh(
            host,
            f"mkdir -p {q(host.repoDir)} && cd {q(host.repoDir)} && "
            f"if [ -e {HOST_MARKER} ] || [ -z \"$(ls -A)\" ]; then touch {HOST_MARKER} && mkdir -p outputs/exports && echo ok; else echo foreign; fi",
        )
        if (check.stdout or "").strip() != "ok":
            raise TrainingError(
                f"{host.sshTarget}:{host.repoDir} is not empty and was not set up by this page; the code sync "
                "would delete files there. Use an empty directory for training."
            )
        excludes = [arg for pattern in SYNC_EXCLUDES for arg in ("--exclude", pattern)]
        self._append_log(rec, f"Syncing code to {host.sshTarget}:{host.repoDir}")
        result = self._run(
            ["rsync", "-a", "--delete", *excludes, "-e", "ssh " + " ".join(SSH_OPTS),
             f"{self.repo_root}/", f"{host.sshTarget}:{host.repoDir}/"],
            capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise TrainingError(f"code sync failed: {(result.stderr or '').strip()[-300:]}")
        for root in roots:
            self._append_log(rec, f"Copying export {root.name}")
            result = self._run(
                ["rsync", "-a", "--delete", "-e", "ssh " + " ".join(SSH_OPTS), f"{root}/",
                 f"{host.sshTarget}:{host.repoDir}/outputs/exports/{root.name}/"],
                capture_output=True, text=True, timeout=7200, stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise TrainingError(f"copying {root.name} failed: {(result.stderr or '').strip()[-300:]}")

    def _append_log(self, rec: RunRecord, line: str) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(rec.logPath, "a", encoding="utf-8") as handle:
            handle.write(f"[gateway {_now()}] {line}\n")

    def _launch(self, rec: RunRecord, host: TrainingHost, roots: list[Path], args: list[str]) -> None:
        try:
            if host.kind == "local":
                self._spawn_local(rec, host, args)
            else:
                self._sync_remote(host, roots, rec)
                result = self._ssh(host, self._remote_command(host, args, rec.job))
                pid = (result.stdout or "").strip().splitlines()[-1:] or [""]
                if result.returncode != 0 or not pid[0].isdigit():
                    raise TrainingError(f"could not start on {host.sshTarget}: {(result.stderr or '').strip()[-300:]}")
                rec.pid = int(pid[0])
                rec.runDir = f"{host.repoDir}/outputs/training/runs/{rec.job}"
            rec.state = "running"
            rec.message = f"Training on {host.label}"
            self._append_log(rec, f"Started (pid {rec.pid}) on {host.label}: {' '.join(args)}")
        except Exception as exc:  # noqa: BLE001 -- reported on the page
            rec.state = "failed"
            rec.message = str(exc)
            rec.finishedAt = _now()
            self._append_log(rec, f"ERROR: {exc}")
        with self.lock:
            self._save()

    def resume(self, job: str, steps: int | None = None) -> RunRecord:
        rec = self._get(job)
        self.refresh()
        if rec.state in ACTIVE_STATES:
            raise TrainingError(f"{job} is still {rec.state}")
        host = self.host(rec.hostId)
        if host.kind == "local":
            self._check_local_allowed()
        args = [TRAIN_SCRIPT, "--resume", f"outputs/training/runs/{job}"]
        if steps is not None:
            if not 10 <= int(steps) <= 2_000_000:
                raise TrainingError("steps must be between 10 and 2000000")
            if int(steps) <= rec.step:
                raise TrainingError(f"{job} is at step {rec.step}; raise the total steps above that")
            args += ["--steps", str(int(steps))]
            rec.totalSteps = int(steps)
        rec.state = "starting"
        rec.finishedAt = ""
        rec.message = "Resuming"
        rec.resumes.append(_now())
        threading.Thread(target=self._launch, args=(rec, host, [], args), daemon=True).start()
        with self.lock:
            self._save()
        return rec

    def stop(self, job: str) -> str:
        rec = self._get(job)
        if rec.pid is None or rec.state not in ACTIVE_STATES:
            raise TrainingError(f"{job} is not running")
        host = self.host(rec.hostId)
        if host.kind == "local":
            if not _pid_runs_job(rec.pid, job):
                message = "the trainer had already exited"
            else:
                os.killpg(int(rec.pid), signal.SIGTERM)
                message = f"stopped pid {rec.pid}"
        else:
            q = shlex.quote
            result = self._ssh(
                host,
                f"pid={int(rec.pid)}; if tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -qF -- {q(job)}; "
                "then kill -TERM -- -$pid && echo stopped; else echo gone; fi",
            )
            message = (result.stdout or "").strip() or (result.stderr or "").strip()
        rec.state = "stopped"
        rec.finishedAt = _now()
        rec.message = f"Stopped by the operator ({message}); Resume continues from the last checkpoint"
        self._append_log(rec, rec.message)
        with self.lock:
            self._save()
        return rec.message

    def fetch(self, job: str) -> RunRecord:
        """Copy a remote run's checkpoints (models only, no optimiser state) back here."""
        rec = self._get(job)
        host = self.host(rec.hostId)
        if host.kind == "local":
            raise TrainingError("a local run is already here")
        local = self.root / "runs" / job
        local.mkdir(parents=True, exist_ok=True)
        result = self._run(
            ["rsync", "-a", "--exclude", "training_state/", "-e", "ssh " + " ".join(SSH_OPTS),
             f"{host.sshTarget}:{rec.runDir}/", f"{local}/"],
            capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise TrainingError(f"fetch failed: {(result.stderr or '').strip()[-300:]}")
        rec.localRunDir = str(local)
        rec.fetchedAt = _now()
        with self.lock:
            self._save()
        return rec

    def delete_record(self, job: str) -> None:
        rec = self._get(job)
        if rec.state in ACTIVE_STATES:
            raise TrainingError("stop the run first")
        with self.lock:
            del self.runs[job]
            self._save()

    def _get(self, job: str) -> RunRecord:
        with self.lock:
            rec = self.runs.get(job)
        if rec is None:
            raise TrainingError(f"no run {job!r}")
        return rec

    # ------------------------------------------------------------- status ---
    def refresh(self) -> None:
        changed = False
        for rec in list(self.runs.values()):
            if rec.state not in ACTIVE_STATES:
                continue
            if rec.kind == "remote" and rec.state == "running":
                changed |= self._poll_remote(rec)
            changed |= self._read_log(rec)
            if rec.state == "running" and rec.kind == "local" and rec.pid and not _pid_runs_job(rec.pid, rec.job):
                self._read_log(rec)
                self._finish(rec)
                changed = True
        if changed:
            with self.lock:
                self._save()

    def _poll_remote(self, rec: RunRecord) -> bool:
        if time.monotonic() - self._remote_polled.get(rec.job, 0.0) < 5.0:
            return False
        self._remote_polled[rec.job] = time.monotonic()
        try:
            host = self.host(rec.hostId)
        except TrainingError:
            return False
        q = shlex.quote
        self._run(
            ["rsync", "-a", "-e", "ssh " + " ".join(SSH_OPTS),
             f"{host.sshTarget}:{host.repoDir}/outputs/training/logs/{rec.job}.log", rec.logPath],
            capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
        )
        try:
            alive = self._ssh(
                host, f"tr '\\0' ' ' < /proc/{int(rec.pid or 0)}/cmdline 2>/dev/null | grep -qF -- {q(rec.job)} && echo alive",
                timeout=15,
            )
        except TrainingError:
            return False
        if (alive.stdout or "").strip() != "alive":
            self._read_log(rec)
            self._finish(rec)
            return True
        return False

    def _finish(self, rec: RunRecord) -> None:
        if rec.state != "running":
            return
        if rec.message == "done":
            rec.state = "done"
            check = rec.reloadCheck or {}
            rec.message = "Finished" + (
                " — the last checkpoint reloads and predicts identically" if check.get("ok")
                else " — RELOAD CHECK FAILED, do not use this checkpoint" if check else ""
            )
        else:
            rec.state = "failed"
            rec.message = "The trainer exited before finishing; see the log (Resume continues from the last checkpoint)"
        rec.finishedAt = _now()

    def _read_log(self, rec: RunRecord) -> bool:
        try:
            with open(rec.logPath, "rb") as handle:
                handle.seek(rec.logOffset)
                data = handle.read()
        except OSError:
            return False
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            return False
        rec.logOffset += last_nl + 1
        for raw in data[: last_nl + 1].decode("utf-8", "replace").splitlines():
            line = raw.split("\r")[-1]
            idx = line.find(PROGRESS_PREFIX)
            if idx < 0:
                continue
            try:
                event = json.loads(line[idx + len(PROGRESS_PREFIX):])
            except ValueError:
                continue
            kind = event.get("event")
            if kind == "train":
                rec.step = int(event.get("steps") or rec.step)
                rec.loss = event.get("loss")
            elif kind == "checkpoint":
                rec.step = max(rec.step, int(event.get("step") or 0))
                rec.lastCheckpoint = str(event.get("path") or "")
            elif kind == "reload_check":
                rec.reloadCheck = {k: v for k, v in event.items() if k != "event"}
            elif kind == "done":
                rec.message = "done"
            elif kind == "refused":
                rec.message = f"refused: {event.get('error')}"
        return True

    def log_tail(self, job: str, lines: int = 200) -> list[str]:
        rec = self._get(job)
        try:
            with open(rec.logPath, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 200_000))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            return []
        out = []
        for raw in text.splitlines():
            line = raw.split("\r")[-1]
            if PROGRESS_PREFIX in line and '"event": "train"' in line:
                continue
            out.append(line)
        return out[-lines:]

    def checkpoints(self, job: str) -> list[dict[str, Any]]:
        """Checkpoints of a run that are here (local runs, or remote runs after Fetch)."""
        rec = self._get(job)
        base = Path(rec.localRunDir) if rec.localRunDir else None
        if base is None or not (base / "lerobot" / "checkpoints").is_dir():
            return []
        out = []
        for ckpt in sorted((base / "lerobot" / "checkpoints").iterdir()):
            model = ckpt / "pretrained_model"
            if ckpt.name == "last" or not model.is_dir():
                continue
            prov: dict[str, Any] = {}
            with contextlib.suppress(OSError, ValueError):
                prov = json.loads((model / "fr3_provenance.json").read_text(encoding="utf-8"))
            out.append({
                "step": int(ckpt.name) if ckpt.name.isdigit() else ckpt.name,
                "path": str(model),
                "hasContract": (model / "fr3_policy_contract.json").is_file(),
                "reloadCheck": prov.get("reload_check"),
                "exports": [e.get("name") for e in prov.get("exports") or []],
                "code": (prov.get("code") or {}).get("git") or (prov.get("code") or {}).get("release"),
            })
        return out

    def status(self) -> dict[str, Any]:
        self.refresh()
        with self.lock:
            runs = [asdict(r) for r in sorted(self.runs.values(), key=lambda r: r.startedAt, reverse=True)]
        for run in runs:
            run.pop("logOffset", None)
        return {
            "hosts": [asdict(h) for h in self.hosts()],
            "runs": runs,
            "profile": self.profile,
            "sandbox": self.sandbox,
            "recorderConnected": bool(self.recorder_busy()),
        }


def _pid_runs_job(pid: int, job: str) -> bool:
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return job in cmdline
