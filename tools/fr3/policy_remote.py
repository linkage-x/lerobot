"""Run the policy's forward pass on another machine: the wire protocol, the client, the serve loop.

The robot host (the NUC) keeps everything that touches hardware and everything the 30 Hz loop
decides: cameras, the arm, the RTC action queue, the leash, recording. Only the forward pass
crosses the wire, at the one place it is already isolated -- `predict_action_chunk_for_rollout`.
The server runs that same function against its GPU-resident policy, so the chunk that comes back
is the chunk a single-machine rollout would have computed from the same inputs.

The server is a standing process on the GPU machine that loads whatever checkpoint a client
names in its hello (and keeps it until a different one is asked for, or until it has sat idle
long enough to be in the way of training). Nobody starts it per checkpoint, so nobody has to
start it with flags that match the robot host's; the fingerprint checks that both sides derived
the same config from the same inputs.

Without `--policy-server` nothing here is called and the runtime is what it always was.

Frames are torch-serialised and read back with ``weights_only=True``: tensors, numbers, strings,
lists and dicts, nothing executable. Pickle would let any host that can reach the port run code
on the GPU machine; the firewall is the first line, this is the second.
"""

from __future__ import annotations

import io
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

DEFAULT_POLICY_SERVER_PORT = 8790
# 2: the hello carries a load spec and the server loads on demand; status/unload need no hello.
PROTOCOL_VERSION = 2

_LENGTH = struct.Struct('>Q')
# A request carries two cropped camera frames (~1 MB as uint8); a reply carries one action chunk.
# Anything far past this is not a message from this protocol.
_MAX_FRAME_BYTES = 256 * 1024 * 1024


class PolicyServerError(RuntimeError):
    """The server refused or failed a request; the message is the server's own."""


def parse_policy_server_address(value: str) -> tuple[str, int]:
    host, sep, port = str(value).strip().rpartition(':')
    if not sep:
        return str(value).strip(), DEFAULT_POLICY_SERVER_PORT
    if not host:
        raise ValueError(f'--policy-server expects host[:port], got {value!r}.')
    return host, int(port)


def _encode(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _decode(payload: bytes) -> Any:
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(size - len(chunks), 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError('policy server connection closed mid-message')
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = _encode(obj)
    sock.sendall(_LENGTH.pack(len(payload)) + payload)


def recv_frame(sock: socket.socket) -> bytes:
    (size,) = _LENGTH.unpack(_recv_exact(sock, _LENGTH.size))
    if size > _MAX_FRAME_BYTES:
        raise ConnectionError(f'policy server frame of {size} bytes exceeds the {_MAX_FRAME_BYTES} byte limit')
    return _recv_exact(sock, size)


def recv_message(sock: socket.socket) -> Any:
    return _decode(recv_frame(sock))


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """numpy arrays become tensors (the only array type a weights-only load accepts), marked so the
    far side can turn them back; the observation builder hands out numpy and the predictor expects it."""
    tensors: dict[str, torch.Tensor] = {}
    numpy_keys: list[str] = []
    other: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            tensors[key] = torch.from_numpy(np.ascontiguousarray(value))
            numpy_keys.append(key)
        elif torch.is_tensor(value):
            tensors[key] = value.detach().cpu()
        else:
            other[key] = value
    return {'tensors': tensors, 'numpy_keys': numpy_keys, 'other': other}


def decode_observation(encoded: dict[str, Any]) -> dict[str, Any]:
    numpy_keys = set(encoded['numpy_keys'])
    observation: dict[str, Any] = dict(encoded['other'])
    for key, value in encoded['tensors'].items():
        observation[key] = value.numpy() if key in numpy_keys else value
    return observation




def outputs_relative(path: str | Path) -> str | None:
    """`path` from its `outputs/` component on, or None when it has none.

    This is how the two machines name the same checkpoint or view: they keep the repo at
    different absolute paths, but both keep training artefacts under `outputs/`.
    """
    parts = Path(path).resolve().parts
    return '/'.join(parts[parts.index('outputs'):]) if 'outputs' in parts else None


def resolve_outputs_path(repo_root: Path, relative: str) -> Path:
    """The server's copy of a path a client named with `outputs_relative`.

    Checked lexically rather than by resolving: `outputs/train` is often a symlink onto a bigger
    disk, and what is being refused is a client naming something outside `outputs/` at all.
    """
    parts = Path(str(relative)).parts
    if not parts or parts[0] != 'outputs' or Path(str(relative)).is_absolute() or '..' in parts:
        raise ValueError(f'expected a path under outputs/, got {relative!r}')
    return Path(repo_root) / Path(*parts)


def policy_fingerprint(policy_cfg: Any, pretrained_dir: Path) -> dict[str, Any]:
    """What both ends must agree on for a remote chunk to mean the same as a local one.

    The checkpoint is named by its path from `outputs/` on (see `outputs_relative`); everything
    else is the loaded config, after the same runtime overrides on both sides.
    """
    checkpoint = outputs_relative(pretrained_dir) or '/'.join(Path(pretrained_dir).resolve().parts[-4:])
    rtc = getattr(policy_cfg, 'rtc_config', None)
    schedule = getattr(rtc, 'prefix_attention_schedule', None)
    return {
        'checkpoint': checkpoint,
        'policy_type': str(getattr(policy_cfg, 'type', '')),
        'chunk_size': int(getattr(policy_cfg, 'chunk_size', 0) or 0),
        'n_action_steps': int(getattr(policy_cfg, 'n_action_steps', 0) or 0),
        'use_amp': bool(getattr(policy_cfg, 'use_amp', False)),
        'rtc_enabled': bool(rtc is not None and getattr(rtc, 'enabled', False)),
        'rtc_execution_horizon': int(getattr(rtc, 'execution_horizon', 0) or 0) if rtc is not None else 0,
        'rtc_max_guidance_weight': float(getattr(rtc, 'max_guidance_weight', 0.0) or 0.0) if rtc is not None else 0.0,
        'rtc_prefix_attention_schedule': str(getattr(schedule, 'value', schedule)),
    }


# The runtime overrides that shape the loaded policy, i.e. the keyword arguments
# `prepare_rollout_policy_config` takes. A load spec carries exactly these, plus where to find
# the checkpoint and its view.
LOAD_SPEC_OVERRIDES = (
    'n_action_steps_override',
    'act_temporal_ensemble_coeff',
    'rtc_mode',
    'rtc_execution_horizon',
    'rtc_max_guidance_weight',
    'rtc_prefix_attention_schedule',
)


def policy_load_spec(pretrained_dir: Path, *, dataset_root: Path | None, **config_overrides: Any) -> dict[str, Any]:
    """What a client asks the server to load: the same inputs its own config was derived from.

    The server loads whatever the client names, so nobody has to start it with flags that match
    the robot host's -- the fingerprint then checks that the two derivations agreed.
    """
    checkpoint = outputs_relative(pretrained_dir)
    if checkpoint is None:
        raise ValueError(
            f'{pretrained_dir} is not under an outputs/ directory, so a policy server cannot name it. '
            'Keep checkpoints under outputs/train/.'
        )
    unknown = set(config_overrides) - set(LOAD_SPEC_OVERRIDES)
    if unknown:
        raise ValueError(f'not policy overrides: {sorted(unknown)}')
    spec: dict[str, Any] = {
        'checkpoint': checkpoint,
        'dataset_root': outputs_relative(dataset_root) if dataset_root is not None else None,
    }
    # Only what the caller gave: an omitted override means the loader's default on both machines,
    # and spelling it out as None here would turn "default" into "None" on the server.
    for key in LOAD_SPEC_OVERRIDES:
        if key in config_overrides:
            value = config_overrides[key]
            spec[key] = value.value if hasattr(value, 'value') else value
    return spec


class ServerSideProcessor:
    """Where the robot host's pre/post processor would be: they run on the server, next to the policy.

    Building them here would also mean building pi05's tokenizer, which fetches from the HF hub --
    a dependency the robot host has no use for and, on the NUC, no route to. `reset()` is the one
    thing the rollout does to them outside the forward pass; anything else is a code path that
    the remote split does not cover, and it fails here rather than computing on the wrong side.
    """

    def __init__(self, role: str) -> None:
        self.role = role

    def reset(self) -> None:
        return None

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f'the {self.role} runs on the policy server; this path is not supported with --policy-server')


# The predictor's own arguments that describe *where* it runs. The client cannot send them -- they
# name objects on its side -- and the server substitutes its own.
_LOCAL_ONLY_PREDICT_KWARGS = ('policy', 'device', 'preprocessor', 'postprocessor', 'use_amp')


def _connect(address: str, timeout_s: float) -> socket.socket:
    host, port = parse_policy_server_address(address)
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def query_policy_server(address: str, op: str = 'status', *, timeout_s: float = 2.0) -> dict[str, Any]:
    """One request that needs no hello -- `status` or `unload` -- on a connection of its own.

    This is how the gateway reads a server without becoming its client: a status probe must not
    take the policy away from a rollout that is using it.
    """
    if op not in ('status', 'unload'):
        raise ValueError(f'{op!r} needs a session; use RemotePolicyClient')
    with _connect(address, timeout_s) as sock:
        sock.settimeout(timeout_s)
        send_message(sock, {'op': op})
        reply = recv_message(sock)
    if not isinstance(reply, dict) or not reply.get('ok', False):
        error = reply.get('error') if isinstance(reply, dict) else repr(reply)
        raise PolicyServerError(f'policy server {address}: {error}')
    return reply


class RemotePolicyClient:
    """Stands where the policy object stood in the rollout.

    `config` is the checkpoint's config loaded locally (no weights), so everything the runtime
    reads off `policy.config` -- input features, chunk size, the RTC config -- is answered without
    the network. `reset()` and chunk prediction are requests. Calls are serialised: the async RTC
    planner thread and the main thread share one connection.

    The hello waits up to `load_timeout_s`, because it is where the server loads the checkpoint
    when it does not already hold it -- about a minute for pi05.
    """

    def __init__(
        self,
        address: str,
        *,
        config: Any,
        fingerprint: dict[str, Any],
        load_spec: dict[str, Any],
        connect_timeout_s: float = 5.0,
        load_timeout_s: float = 600.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.address = address
        self.config = config
        self.fingerprint = dict(fingerprint)
        self.load_spec = dict(load_spec)
        parse_policy_server_address(address)
        self._connect_timeout_s = float(connect_timeout_s)
        self._load_timeout_s = float(load_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.server_info: dict[str, Any] = {}
        self.last_server_latency_s = 0.0

    def connect(self) -> dict[str, Any]:
        with self._lock:
            if self._sock is not None:
                return self.server_info
            self._sock = _connect(self.address, self._connect_timeout_s)
            try:
                self._sock.settimeout(self._load_timeout_s)
                self.server_info = self._request_locked(
                    {'op': 'hello', 'protocol': PROTOCOL_VERSION, 'load': self.load_spec, 'fingerprint': self.fingerprint}
                )
                self._sock.settimeout(self._request_timeout_s)
            except BaseException:
                self._close_locked()
                raise
            return self.server_info

    def _request_locked(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise ConnectionError(f'not connected to policy server {self.address}')
        try:
            send_message(self._sock, message)
            reply = recv_message(self._sock)
        except BaseException:
            # A half-read reply leaves the stream unsynchronised; nothing after it can be trusted.
            self._close_locked()
            raise
        if not isinstance(reply, dict) or not reply.get('ok', False):
            error = reply.get('error') if isinstance(reply, dict) else repr(reply)
            raise PolicyServerError(f'policy server {self.address}: {error}')
        return reply

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._request_locked(message)

    def reset(self) -> None:
        self._request({'op': 'reset'})

    def predict_action_chunk_for_rollout(
        self, observation: dict[str, Any], **predict_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {k: v for k, v in predict_kwargs.items() if k not in _LOCAL_ONLY_PREDICT_KWARGS}
        prev = kwargs.get('prev_chunk_left_over')
        if prev is not None:
            kwargs['prev_chunk_left_over'] = prev.detach().cpu()
        if kwargs.get('lateral_action_indices') is not None:
            kwargs['lateral_action_indices'] = list(kwargs['lateral_action_indices'])
        reply = self._request({'op': 'predict', 'observation': encode_observation(observation), 'kwargs': kwargs})
        self.last_server_latency_s = float(reply.get('server_latency_s', 0.0))
        return reply['original_actions'], reply['processed_actions']

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()


# ------------------------------------------------------------------ server side ---


@dataclass
class LoadedPolicy:
    """One loaded checkpoint, as the serve loop uses it. `release` gives its memory back."""

    spec: dict[str, Any]
    fingerprint: dict[str, Any]
    device: torch.device
    use_amp: bool
    predict: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    reset: Callable[[], None]
    release: Callable[[], None] = lambda: None


class PolicyBusy(RuntimeError):
    """Another client holds the policy."""


# The holder name of a load nobody asked for yet (`PolicyHost.warm`). Peers are named host:port.
WARMUP_HOLDER = 'warmup'


class PolicyHost:
    """The one policy a GPU machine holds: loaded when a client asks for it, dropped when idle.

    One holder at a time, because a second rollout would interleave its RTC prefixes with the
    first one's -- and loading happens only inside a hold, so nothing can swap the weights out
    from under a running rollout. Status and unload are answered from any connection, so a
    gateway can look at the server without becoming its client.

    The idle unload is for the machine's other job: the same card trains, and 9.5 GB of resident
    pi05 is what a training run started an hour after the last rollout would run out of.
    """

    def __init__(
        self,
        loader: Callable[[dict[str, Any]], LoadedPolicy],
        *,
        idle_unload_s: float | None = None,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loader = loader
        self.idle_unload_s = idle_unload_s
        self._log = log
        self._clock = clock
        self._lock = threading.Lock()
        self._released = threading.Condition(self._lock)
        self._loaded: LoadedPolicy | None = None
        self._loading: dict[str, Any] | None = None
        self._holder: str | None = None
        self._idle_since = clock()

    def acquire(self, spec: dict[str, Any], holder: str) -> LoadedPolicy:
        with self._lock:
            # A warm-up is not a rollout: whoever arrives during one waits for it -- and gets its
            # load for free when the specs match -- rather than being told the policy is in use.
            self._released.wait_for(lambda: self._holder != WARMUP_HOLDER)
            if self._holder is not None:
                raise PolicyBusy(f'the policy is in use by {self._holder}; one rollout at a time')
            self._holder = holder
            current = self._loaded
        try:
            if current is not None and current.spec == spec:
                return current
            with self._lock:
                self._loaded = None
                self._loading = dict(spec)
            if current is not None:
                self._log(f'[INFO] policy_server unloading checkpoint={current.spec.get("checkpoint")} for a different spec')
                current.release()
            self._log(f'[INFO] policy_server loading spec={spec}')
            started = self._clock()
            try:
                loaded = self._loader(spec)
            finally:
                with self._lock:
                    self._loading = None
            with self._lock:
                self._loaded = loaded
            self._log(f'[INFO] policy_server loaded checkpoint={spec.get("checkpoint")} in {self._clock() - started:.1f}s')
            return loaded
        except BaseException:
            self.release(holder)
            raise

    def release(self, holder: str) -> None:
        """End `holder`'s hold. A fresh client is a fresh rollout, so whatever the last one left in
        the policy (its RTC prefix, its action queue) is cleared here, not at the next hello."""
        with self._lock:
            if self._holder != holder:
                return
            loaded = self._loaded
        try:
            if loaded is not None:
                loaded.reset()
        except Exception as exc:  # a reset that fails must not keep the policy held forever
            self._log(f'[WARN] policy_server reset_failed {type(exc).__name__}: {exc}')
        finally:
            with self._lock:
                self._holder = None
                self._idle_since = self._clock()
                self._released.notify_all()

    def warm(self, spec: dict[str, Any]) -> LoadedPolicy | None:
        """Load `spec` before any rollout asks for it, so the first one does not wait the minute.

        Returns None when a rollout got the policy first; that rollout loads what it needs. A
        failed load raises, as it would in a hello, and leaves the server free.
        """
        try:
            loaded = self.acquire(spec, WARMUP_HOLDER)
        except PolicyBusy:
            return None
        self.release(WARMUP_HOLDER)
        return loaded

    def unload(self) -> bool:
        with self._lock:
            if self._holder is not None:
                raise PolicyBusy(f'the policy is in use by {self._holder}; stop that rollout first')
            loaded, self._loaded = self._loaded, None
        if loaded is None:
            return False
        loaded.release()
        self._log(f'[INFO] policy_server unloaded checkpoint={loaded.spec.get("checkpoint")}')
        return True

    def unload_if_idle(self) -> bool:
        if self.idle_unload_s is None:
            return False
        with self._lock:
            idle = self._holder is None and self._loaded is not None and self._clock() - self._idle_since >= self.idle_unload_s
        return self.unload() if idle else False

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded, loading, holder = self._loaded, self._loading, self._holder
            idle_s = None if holder is not None else self._clock() - self._idle_since
        status: dict[str, Any] = {
            'loaded': None if loaded is None else {'spec': loaded.spec, 'fingerprint': loaded.fingerprint, 'device': str(loaded.device)},
            'loading': loading,
            'holder': holder,
            'idle_s': idle_s,
            'idle_unload_s': self.idle_unload_s,
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            status['gpu'] = {
                'name': torch.cuda.get_device_name(),
                'free_mb': int(free // 2**20),
                'total_mb': int(total // 2**20),
                'allocated_mb': int(torch.cuda.memory_allocated() // 2**20),
            }
        return status


def _normalise_predict_kwargs(kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    kwargs = dict(kwargs)
    prev = kwargs.get('prev_chunk_left_over')
    if prev is not None:
        # RTC only pads (and thereby moves) a prefix shorter than the chunk; a full-length one is
        # used as-is against GPU tensors.
        kwargs['prev_chunk_left_over'] = prev.to(device)
    if kwargs.get('lateral_action_indices') is not None:
        kwargs['lateral_action_indices'] = tuple(int(i) for i in kwargs['lateral_action_indices'])
    return kwargs


def handle_request(message: Any, *, host: PolicyHost, session: dict[str, Any]) -> dict[str, Any]:
    """One request in, one reply out. `session['policy']` gates the policy behind a matching hello."""
    if not isinstance(message, dict):
        return {'ok': False, 'error': f'malformed request {type(message).__name__}'}
    op = message.get('op')
    if op == 'status':
        return {'ok': True, 'protocol': PROTOCOL_VERSION, 'status': host.status()}
    if op == 'unload':
        if session.get('policy') is not None:
            return {'ok': False, 'error': 'this connection holds the policy; close it instead'}
        return {'ok': True, 'unloaded': host.unload()}
    if op == 'hello':
        if message.get('protocol') != PROTOCOL_VERSION:
            return {'ok': False, 'error': f'protocol {message.get("protocol")!r} != server {PROTOCOL_VERSION}; deploy the same code to both machines'}
        if session.get('policy') is not None:
            return {'ok': False, 'error': 'hello sent twice on one connection'}
        spec = message.get('load')
        if not isinstance(spec, dict) or 'checkpoint' not in spec:
            return {'ok': False, 'error': 'hello without a load spec'}
        loaded = host.acquire(spec, session['peer'])
        client_fp = message.get('fingerprint') or {}
        mismatched = {
            key: {'client': client_fp.get(key), 'server': loaded.fingerprint.get(key)}
            for key in sorted(set(client_fp) | set(loaded.fingerprint))
            if client_fp.get(key) != loaded.fingerprint.get(key)
        }
        if mismatched:
            host.release(session['peer'])
            return {'ok': False, 'error': f'policy mismatch, the two machines derived different configs: {mismatched}'}
        session['policy'] = loaded
        return {'ok': True, 'fingerprint': loaded.fingerprint, 'device': str(loaded.device)}
    loaded = session.get('policy')
    if loaded is None:
        return {'ok': False, 'error': f'{op!r} before a matching hello'}
    if op == 'reset':
        loaded.reset()
        return {'ok': True}
    if op == 'predict':
        started = time.perf_counter()
        original_actions, processed_actions = loaded.predict(
            decode_observation(message['observation']),
            device=loaded.device,
            use_amp=loaded.use_amp,
            **_normalise_predict_kwargs(message['kwargs'], loaded.device),
        )
        return {
            'ok': True,
            'original_actions': original_actions.detach().cpu(),
            'processed_actions': processed_actions.detach().cpu(),
            'server_latency_s': time.perf_counter() - started,
        }
    return {'ok': False, 'error': f'unknown op {op!r}'}


def serve_connection(
    conn: socket.socket, *, host: PolicyHost, peer: str, log: Callable[[str], None] = print
) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    session: dict[str, Any] = {'peer': peer, 'policy': None}
    try:
        while True:
            try:
                frame = recv_frame(conn)
            except (ConnectionError, OSError):
                return
            message: Any = None
            try:
                # Decoded inside the guard: a frame the weights-only loader refuses is answered, and
                # the stream stays in step because the whole frame has already been read.
                message = _decode(frame)
                reply = handle_request(message, host=host, session=session)
            except Exception as exc:  # the client sees it as PolicyServerError; the server keeps serving
                reply = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            op = message.get('op') if isinstance(message, dict) else None
            if not reply.get('ok'):
                log(f'[WARN] policy_server request_failed peer={peer} op={op} {reply["error"]}')
            elif op == 'predict':
                log(f'[INFO] policy_server predict server_latency_s={reply["server_latency_s"]:.3f}')
            try:
                send_message(conn, reply)
            except OSError:
                return
    finally:
        if session['policy'] is not None:
            host.release(peer)
