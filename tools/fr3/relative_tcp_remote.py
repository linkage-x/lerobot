#!/usr/bin/env python
"""A relative-TCP policy whose forward pass runs on another machine's GPU (pi0.5 is not run on Thor).

The robot host (Thor) keeps a *stub* of the checkpoint: everything but the weights (contract, provenance,
configs, normaliser statistics) plus ``fr3_remote_policy.json``, which names the policy server and the
fingerprint of the weights the stub was made from. :func:`relative_tcp_policy.load_checkpoint` on a stub
returns a :class:`RemotePolicy`; preview, replay and execute then work as with a local checkpoint, and the
server must hold the very weights the stub names (checked at load and on every answer).

The server (on the GPU machine, e.g. the 5090 workstation)::

    PYTHONPATH=src:. .venv/bin/python tools/fr3/relative_tcp_remote.py serve \\
        --checkpoint outputs/training/runs/<run>/lerobot/checkpoints/<step>/pretrained_model \\
        --allow-from 192.168.111.122

    # the stub, copied to the robot host's outputs/training/runs/<run>/lerobot/checkpoints/<step>/pretrained_model
    ... relative_tcp_remote.py stub --checkpoint <the same> --server http://<gpu-host>:8795 --out <dir>
    # from the robot host: status and the round trip's latency, on a synthetic input (no hardware)
    ... relative_tcp_remote.py ping --checkpoint <stub> [--n 50]

Wire format: an ``.npz`` (``np.load(allow_pickle=False)``, so nothing in a request can run code): the state,
each camera's cropped frame as uint8 HxWx3, the task and the fingerprint; the answer is an ``.npz`` with the
actions. The frames go as uint8 because the policy's float input is exactly ``uint8 / 255``: the server
rebuilds the same tensor bit for bit (relative_tcp_policy.observation_sample), so a preview record replays
identically through the server.

The GPU is the server's alone (user, 2026-11-20): while any other compute process holds the card, the
server answers 409 instead of predicting, and an execution stops with that reason. ``--allow-shared-gpu``
turns this off for benchmarks; an execution refuses a server started with it.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fr3 import relative_tcp as rt, relative_tcp_policy as rtp

REMOTE_FILE = "fr3_remote_policy.json"
REMOTE_SCHEMA = "fr3_remote_policy/1"
DEFAULT_PORT = 8795
#: The prediction time execution allows a remote policy (user, 2026-11-20: 0.30 s). pi0.5 on the 5090 is
#: ~173 ms per chunk (11-17); the network adds ~1 ms (Thor1 -> workstation ping 0.2 ms).
REMOTE_PREDICT_S = 0.30
#: The weights: left out of a stub, hashed into the fingerprint (with the configs that say how to load them).
WEIGHT_FILES = ("model.safetensors", "adapter_model.safetensors")
FINGERPRINT_FILES = WEIGHT_FILES + ("config.json", "adapter_config.json", rt.CONTRACT_FILE)
MAX_REQUEST_BYTES = 32 << 20
CLIENT_TIMEOUT_S = 2.0
GPU_POLL_S = 1.0


class RemotePolicyError(RuntimeError):
    """The server refused or could not be reached; the message says why."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fingerprint(model: Path) -> str:
    """Of the weights and the configs that load them (names and bytes); 16 hex digits."""
    model = Path(model)
    h = hashlib.sha256()
    found = False
    for name in FINGERPRINT_FILES:
        path = model / name
        if not path.is_file():
            continue
        found = found or name in WEIGHT_FILES
        h.update(name.encode() + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
    if not found:
        raise FileNotFoundError(f"{model}: no weights ({' or '.join(WEIGHT_FILES)}) to fingerprint")
    return h.hexdigest()[:16]


def make_stub(model: Path, out: Path, server: str) -> dict[str, Any]:
    """Copy ``model`` without its weights into ``out``, naming ``server``; returns the remote record."""
    model, out = Path(model), Path(out)
    if (model / "pretrained_model").is_dir():
        model = model / "pretrained_model"
    if not (model / rt.CONTRACT_FILE).is_file():
        raise FileNotFoundError(f"{model}: not a relative-TCP checkpoint (no {rt.CONTRACT_FILE})")
    if (model / REMOTE_FILE).is_file():
        raise ValueError(f"{model} is a stub already")
    cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(model.iterdir()):
        if path.is_file() and path.name not in WEIGHT_FILES:
            shutil.copy2(path, out / path.name)
    record = {
        "schema": REMOTE_SCHEMA,
        "server": server.rstrip("/"),
        "fingerprint": fingerprint(model),
        "policy_type": cfg.get("type"),
        "n_action_steps": cfg.get("n_action_steps"),
        "chunk_size": cfg.get("chunk_size"),
        "source": str(model.resolve()),
        "created": _now(),
    }
    (out / REMOTE_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# ----------------------------------------------------------------- wire ---


def _npz(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _unpack(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def encode_sample(contract: rt.PolicyContract, sample: dict[str, Any], fp: str) -> bytes:
    """relative_tcp_policy.observation_sample's dict -> request body (frames back to the uint8 they were)."""
    import torch

    arrays: dict[str, np.ndarray] = {
        "state": sample["observation.state"].detach().cpu().numpy().astype(np.float32),
        "task": np.array(str(sample.get("task") or "")),
        "fingerprint": np.array(fp),
    }
    for cam in contract.cameras:
        t = sample[cam.key]
        img = torch.round(t.detach().cpu() * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        arrays[f"image.{cam.name}"] = np.ascontiguousarray(img.numpy())
    if sample.get(rtp.RTC_PREFIX_KEY) is not None:
        arrays["rtc_prefix"] = np.asarray(sample[rtp.RTC_PREFIX_KEY], dtype=np.float32)
        arrays["rtc_delay_horizon"] = np.array([sample[rtp.RTC_DELAY_KEY], sample[rtp.RTC_HORIZON_KEY]], np.int64)
    return _npz(**arrays)


def decode_sample(contract: rt.PolicyContract, arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], str]:
    """Request arrays -> (the sample observation_sample would have built, the client's fingerprint)."""
    import torch

    state = arrays.get("state")
    if state is None or state.dtype != np.float32 or state.ndim != 1 or len(state) != len(contract.state_names):
        raise ValueError(f"state: float32 of {len(contract.state_names)} expected")
    sample: dict[str, Any] = {"observation.state": torch.from_numpy(state.copy()),
                              "task": str(arrays.get("task", np.array(""))) or contract.task}
    for cam in contract.cameras:
        img = arrays.get(f"image.{cam.name}")
        if img is None or img.dtype != np.uint8 or img.shape != (cam.height, cam.width, 3):
            raise ValueError(f"image.{cam.name}: uint8 {(cam.height, cam.width, 3)} expected")
        sample[cam.key] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    if "rtc_prefix" in arrays:
        dh = arrays.get("rtc_delay_horizon")
        if dh is None or dh.shape != (2,) or arrays["rtc_prefix"].dtype != np.float32:
            raise ValueError("rtc_prefix: float32 with rtc_delay_horizon (2,) expected")
        sample = rtp.with_rtc_prefix(sample, arrays["rtc_prefix"], delay=int(dh[0]), horizon=int(dh[1]))
    return sample, str(arrays.get("fingerprint", np.array("")))


# --------------------------------------------------------------- client ---


class _Connection:
    """One kept-alive HTTP connection to the server (reopened once when it has gone away).

    Not urllib: its opener builds an SSL context per call (~24 ms on Thor, 11-20) and closes the
    connection after each request."""

    def __init__(self, server: str, timeout: float = CLIENT_TIMEOUT_S):
        parts = urllib.parse.urlsplit(server)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"policy server {server!r}: http://<host>:<port> expected")
        self.server, self.host, self.port, self.timeout = server, parts.hostname, parts.port or 80, timeout
        self.conn: http.client.HTTPConnection | None = None
        self.lock = threading.Lock()

    def request(self, path: str, data: bytes | None = None) -> tuple[int, bytes]:
        with self.lock:
            for attempt in (0, 1):
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                try:
                    headers = {"Content-Type": "application/octet-stream"} if data is not None else {}
                    self.conn.request("POST" if data is not None else "GET", path, body=data, headers=headers)
                    resp = self.conn.getresponse()
                    return resp.status, resp.read()
                except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError) as exc:
                    self.close_locked()
                    if attempt:
                        raise RemotePolicyError(f"the policy server {self.server} closed the connection: {exc}") from exc
                except (OSError, http.client.HTTPException) as exc:  # refused, unreachable, timed out
                    self.close_locked()
                    raise RemotePolicyError(f"the policy server {self.server} cannot be reached: {exc}") from exc
        raise AssertionError("unreachable")

    def close_locked(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def _error_text(body: bytes) -> str:
    try:
        return str(json.loads(body.decode("utf-8")).get("error") or body[:200])
    except (ValueError, UnicodeDecodeError, AttributeError):
        return repr(body[:200])


@dataclass
class RemotePolicy:
    """Stands in for relative_tcp_policy.LoadedPolicy: same ``contract``/``provenance``/``predict``."""

    path: Path
    contract: rt.PolicyContract
    provenance: dict[str, Any]
    remote: dict[str, Any]
    policy: Any = None  # .config.n_action_steps / .config.type, as relative_tcp_policy._config reads them
    preprocessor: Any = None
    postprocessor: Any = None
    device: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    last_server_s: float = 0.0
    rtc: bool = False  # the server's policy takes an RTC prefix (relative_tcp_policy.LoadedPolicy.rtc)
    _conn: _Connection | None = None

    @property
    def server(self) -> str:
        return str(self.remote["server"])

    def _http(self, path: str, data: bytes | None = None) -> tuple[int, bytes]:
        if self._conn is None:
            self._conn = _Connection(self.server)
        return self._conn.request(path, data)

    def hello(self) -> dict[str, Any]:
        """The server's status, checked: it holds this stub's weights. Raises RemotePolicyError."""
        code, body = self._http("/status")
        if code != 200:
            raise RemotePolicyError(f"the policy server {self.server}: HTTP {code}: {_error_text(body)}")
        status = json.loads(body.decode("utf-8"))
        if status.get("fingerprint") != self.remote["fingerprint"]:
            raise RemotePolicyError(
                f"the policy server {self.server} holds {status.get('checkpoint')} (weights {status.get('fingerprint')}), "
                f"not the weights this stub was made from ({self.remote['fingerprint']}, {self.remote.get('source')})")
        self.status = status
        self.rtc = bool(status.get("rtc"))
        return status

    def predict(self, sample: dict[str, Any]) -> np.ndarray:
        body = encode_sample(self.contract, sample, self.remote["fingerprint"])
        code, answer = self._http("/predict", body)
        if code != 200:
            raise RemotePolicyError(f"the policy server {self.server}: {_error_text(answer)}")
        arrays = _unpack(answer)
        if str(arrays.get("fingerprint", np.array(""))) != self.remote["fingerprint"]:
            raise RemotePolicyError(f"the policy server {self.server} answered with other weights")
        self.last_server_s = float(arrays.get("server_s", np.float64(0.0)))
        return arrays["actions"]


def is_stub(model: Path) -> bool:
    return (Path(model) / REMOTE_FILE).is_file()


def load_remote(model: Path) -> RemotePolicy:
    """A stub -> RemotePolicy, after checking the server holds its weights."""
    model = Path(model)
    remote = json.loads((model / REMOTE_FILE).read_text(encoding="utf-8"))
    if remote.get("schema") != REMOTE_SCHEMA:
        raise ValueError(f"{model / REMOTE_FILE}: schema {remote.get('schema')!r}, expected {REMOTE_SCHEMA}")
    contract = rt.PolicyContract.read(model)
    prov_path = model / rt.PROVENANCE_FILE
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    config = SimpleNamespace(type=remote.get("policy_type"), n_action_steps=remote.get("n_action_steps"),
                             chunk_size=remote.get("chunk_size"))
    policy = RemotePolicy(model, contract, provenance, remote, policy=SimpleNamespace(config=config),
                          device=f"remote {remote['server']}")
    policy.hello()
    return policy


# --------------------------------------------------------------- server ---


def other_gpu_processes(own_pid: int, run=subprocess.run) -> list[dict[str, Any]] | None:
    """Compute processes on the GPU(s) other than ``own_pid``; None when nvidia-smi cannot say."""
    try:
        out = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        if int(parts[0]) == own_pid:
            continue
        rows.append({"pid": int(parts[0]), "name": parts[1], "mib": int(parts[2]) if parts[2].isdigit() else None})
    return rows


class PolicyServer:
    """One loaded checkpoint; predictions one at a time (one GPU)."""

    def __init__(self, loaded: Any, fp: str, *, allow_from: list[str], exclusive: bool = True,
                 gpu_probe=other_gpu_processes):
        self.loaded = loaded
        self.fp = fp
        self.allow = [ipaddress.ip_network(a, strict=False) for a in allow_from]
        self.exclusive = exclusive
        self.gpu_probe = gpu_probe
        self.lock = threading.Lock()
        self.others: list[dict[str, Any]] | None = []
        self.others_at = 0.0
        self.started = _now()
        self.count = 0
        self.refused = 0
        self.times: list[float] = []

    def allowed(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in self.allow)

    def poll_gpu(self) -> None:
        self.others = self.gpu_probe(os.getpid())
        self.others_at = time.monotonic()

    def gpu_problem(self) -> str:
        """Why the GPU is not this server's alone right now ("" = it is, or exclusivity is off)."""
        if not self.exclusive:
            return ""
        if time.monotonic() - self.others_at > 5 * GPU_POLL_S:
            return "the GPU's other users are not known (nvidia-smi has not answered for 5 s)"
        if self.others is None:
            return "nvidia-smi cannot list the GPU's processes, so the GPU is not known to be this server's alone"
        if self.others:
            who = ", ".join(f"pid {o['pid']} {o['name']} ({o['mib']} MiB)" for o in self.others)
            return f"the GPU is shared with {who}: stop it, the policy server's timing is only known on an idle GPU"
        return ""

    def status(self) -> dict[str, Any]:
        t = sorted(self.times[-200:])
        return {
            "schema": REMOTE_SCHEMA,
            "checkpoint": str(self.loaded.path),
            "fingerprint": self.fp,
            "contract": self.loaded.contract.to_dict(),
            "device": str(self.loaded.device),
            "rtc": bool(getattr(self.loaded, "rtc", False)),
            "exclusive": self.exclusive,
            "gpu_problem": self.gpu_problem(),
            "others": self.others,
            "started": self.started,
            "predictions": self.count,
            "refused": self.refused,
            "server_ms_p50": round(t[len(t) // 2] * 1e3, 1) if t else None,
            "server_ms_max": round(t[-1] * 1e3, 1) if t else None,
        }

    def predict(self, body: bytes) -> tuple[int, bytes]:
        problem = self.gpu_problem()
        if problem:
            self.refused += 1
            return 409, json.dumps({"error": problem}).encode()
        try:
            sample, fp = decode_sample(self.loaded.contract, _unpack(body))
        except (ValueError, OSError, KeyError) as exc:
            return 400, json.dumps({"error": f"bad request: {exc}"}).encode()
        if fp != self.fp:
            return 409, json.dumps({"error": f"this server holds weights {self.fp}, the request names {fp}"}).encode()
        with self.lock:
            t0 = time.perf_counter()
            actions = np.asarray(self.loaded.predict(sample))
            dt = time.perf_counter() - t0
        self.count += 1
        self.times = self.times[-999:] + [dt]
        return 200, _npz(actions=actions, fingerprint=np.array(self.fp), server_s=np.float64(dt))


def make_handler(server: PolicyServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # headers and body leave in separate writes: with Nagle on, the body waits for the client's delayed
        # ACK (~40 ms, measured 11-20: 87 ms round trip around a 23 ms prediction)
        disable_nagle_algorithm = True

        def log_message(self, fmt: str, *args: Any) -> None:  # one line per refusal, not per prediction
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _gate(self) -> bool:
            if server.allowed(self.client_address[0]):
                return True
            print(f"[{_now()}] refused {self.client_address[0]}: not in --allow-from", flush=True)
            self._json(403, {"error": f"{self.client_address[0]} is not allowed (--allow-from)"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._gate():
                return
            if self.path == "/status":
                self._json(200, server.status())
            else:
                self._json(404, {"error": f"unknown {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._gate():
                return
            if self.path != "/predict":
                self._json(404, {"error": f"unknown {self.path}"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_REQUEST_BYTES:
                self._json(413, {"error": f"request of {n} bytes (1..{MAX_REQUEST_BYTES})"})
                return
            code, body = server.predict(self.rfile.read(n))
            if code != 200:
                print(f"[{_now()}] {code}: {_error_text(body)}", flush=True)
                self._send(code, body, "application/json")
            else:
                self._send(code, body, "application/octet-stream")

    return Handler


def serve(checkpoint: Path, *, host: str, port: int, allow_from: list[str], exclusive: bool,
          device: str | None = None) -> None:
    if is_stub(checkpoint):
        raise ValueError(f"{checkpoint} is a stub: serve the checkpoint with its weights")
    fp = fingerprint(Path(checkpoint) / "pretrained_model" if (Path(checkpoint) / "pretrained_model").is_dir() else checkpoint)
    loaded = rtp.load_checkpoint(Path(checkpoint), device=device)
    state = PolicyServer(loaded, fp, allow_from=allow_from, exclusive=exclusive)
    state.poll_gpu()

    def watch() -> None:
        while True:
            time.sleep(GPU_POLL_S)
            state.poll_gpu()

    threading.Thread(target=watch, daemon=True, name="gpu-watch").start()
    # warm up (the first CUDA calls are slow) on a synthetic input
    sample = synthetic_sample(loaded.contract)
    for _ in range(3):
        actions = loaded.predict(sample)
    if getattr(loaded, "rtc", False):  # the guided path is another path (autograd per step): warm it too
        guided = rtp.with_rtc_prefix(sample, actions[8:], delay=8, horizon=16)
        for _ in range(2):
            loaded.predict(guided)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"[{_now()}] serving {loaded.path} (weights {fp}) on {host}:{port} for {', '.join(allow_from)}; "
          f"GPU exclusive: {exclusive}; other GPU processes: {state.others}", flush=True)
    httpd.serve_forever()


def synthetic_sample(contract: rt.PolicyContract) -> dict[str, Any]:
    """A grey frame per camera, the arm still, the gripper half open: for warm-ups and latency (no hardware)."""
    images = {c.name: np.full((c.height, c.width, 3), 128, np.uint8) for c in contract.cameras}
    still = np.repeat(np.eye(4)[None], contract.history + 1, axis=0)
    wrench = np.zeros(len(contract.wrench_names)) if contract.wrench_names else None
    return rtp.observation_sample(contract, still, 0.04, images, wrench, already_cropped=True)


def ping(stub: Path, n: int) -> dict[str, Any]:
    policy = load_remote(stub)
    sample = synthetic_sample(policy.contract)
    rtt, srv = [], []
    for _ in range(max(1, n)):
        t0 = time.perf_counter()
        policy.predict(sample)
        rtt.append(time.perf_counter() - t0)
        srv.append(policy.last_server_s)
    q = lambda v, p: round(float(np.percentile(v, p)) * 1e3, 1)  # noqa: E731
    return {"server": policy.server, "checkpoint": policy.status.get("checkpoint"), "n": len(rtt),
            "round_trip_ms": {"p50": q(rtt, 50), "p95": q(rtt, 95), "max": q(rtt, 100)},
            "server_ms": {"p50": q(srv, 50), "p95": q(srv, 95), "max": q(srv, 100)},
            "limit_ms": REMOTE_PREDICT_S * 1e3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--allow-from", action="append", default=[],
                   help="a client address or network allowed to call (repeat); 127.0.0.1 always is")
    s.add_argument("--allow-shared-gpu", action="store_true",
                   help="predict while other processes use the GPU (benchmarks only: an execution refuses it)")
    s.add_argument("--device", default=None)
    st = sub.add_parser("stub")
    st.add_argument("--checkpoint", type=Path, required=True)
    st.add_argument("--server", required=True, help="http://<gpu-host>:<port>")
    st.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ping")
    p.add_argument("--checkpoint", type=Path, required=True, help="the stub")
    p.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        serve(args.checkpoint, host=args.host, port=args.port, allow_from=["127.0.0.1", *args.allow_from],
              exclusive=not args.allow_shared_gpu, device=args.device)
    elif args.cmd == "stub":
        print(json.dumps(make_stub(args.checkpoint, args.out, args.server), indent=2))
    else:
        print(json.dumps(ping(args.checkpoint, args.n), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
