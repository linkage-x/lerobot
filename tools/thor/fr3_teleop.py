"""FR3 lifecycle owned by the existing Thor camera/BOX recorder.

C constructs this coordinator without opening USB or FCI. F creates a fresh
native worker on the same host. Cameras and BOX never enter that worker.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import select
import socket
import subprocess
import threading
import time

from tools.thor.fr3_ipc import JsonChannel


def gripper_client(pool, box_id: str):
    clients = list(pool._clients)
    if box_id:
        matching = [client for name, client in clients if name == box_id]
        if len(matching) != 1:
            raise RuntimeError(f"Configured FR3 gripper BOX {box_id!r} is not connected")
        return matching[0]
    if len(clients) != 1:
        raise RuntimeError("Set fr3_teleop.gripper_box_id when multiple BOX devices are connected")
    return clients[0][1]


def measured_opening(client, max_width_m: float) -> float:
    sample = client.read()
    sensor = (sample.get("sensors") or {}).get("box_gripper") or {}
    status = ((sample.get("status") or {}).get("sensor_status") or {}).get("box_gripper") or {}
    value = sensor.get("distance_m")
    if status.get("fresh") is not True or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError("BOX gripper opening is unavailable or stale; check the sensor stream")
    if not 0 <= value <= max_width_m + 0.005:
        raise RuntimeError("BOX gripper opening is outside its configured physical range")
    return max(0.0, min(float(value), max_width_m))


def make_spacemouse(config: dict, opening: float, repo_root: Path):
    from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

    raw = dict(config.get("teleop") or {})
    raw.pop("type", None)
    raw["initial_gripper"] = opening
    raw["calibration_dir"] = repo_root / "outputs" / "calibration" / "spacemouse"
    device = SpaceMouseTeleop(SpaceMouseTeleopConfig(**raw))
    device.connect()
    return device


def native_environment(repo_root: Path, runtime_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    # Small 7-DOF IK operations do not benefit from competing BLAS/OpenMP pools.
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root / "src"), str(repo_root), env.get("PYTHONPATH", "")))
    libraries = sorted((runtime_python.parent.parent / "lib").glob("python*/site-packages/cmeel.prefix/lib"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(runtime_python.parent.parent / "lib"),
                                             *(str(path) for path in libraries),
                                             env.get("LD_LIBRARY_PATH", "")])
    return env


def pending_worker_error(channel: JsonChannel) -> str:
    """Preserve a queued native error when the peer closes immediately after it."""
    for _ in range(16):
        if b"\n" not in channel.buffer and not select.select([channel.sock], [], [], 0)[0]:
            break
        try:
            packet = channel.receive()
        except (EOFError, OSError, ValueError):
            break
        if packet.get("state") == "error":
            return str(packet.get("message") or "FR3 controller error")
    return ""


class ThorFr3Session:
    def __init__(self, config: dict, config_path: Path, repo_root: Path, box, *, emit=print):
        self.config = config
        self.settings = config["fr3_teleop"]
        self.config_path = config_path.resolve()
        self.repo_root = repo_root.resolve()
        self.box = box
        self.emit = emit
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.process = None
        self.state = "idle"
        self.error = ""
        self.telemetry = {}
        self.pid = None
        self.recording = False
        self.episode_interrupted = False
        self.history = deque(maxlen=200)
        self.samples = []
        self.last_action = {}
        self.publish("idle", "Sensors connected. Press F to move FR3 to start and enable SpaceMouse")

    def publish(self, state: str, message: str) -> None:
        with self.lock:
            self.state = state
            payload = {"enabled": True, "state": state, "message": message,
                       "telemetry": self.telemetry if state == "running" else {}, "pid": self.pid}
        self.emit("FR3_LIVE " + json.dumps(payload, separators=(",", ":"), allow_nan=False))

    def request_start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            if self.process is not None and self.process.poll() is None:
                self.publish("error", "The previous FR3 worker is still stopping. Exit and restart the session before retrying F")
                return
            self.stop.clear()
            self.error = ""
            self.telemetry = {}
            self.history.clear()
            self.publish("starting", "Preparing FR3 and SpaceMouse; keep the puck released")
            self.thread = threading.Thread(target=self._run, name="thor-fr3-coordinator", daemon=True)
            self.thread.start()

    def request_stop(self) -> None:
        with self.lock:
            if self.recording:
                self.episode_interrupted = True
            self.stop.set()
            if self.thread is not None and self.thread.is_alive():
                self.publish("stopping", "Stopping FR3; cameras and BOX remain connected")

    @property
    def running(self) -> bool:
        with self.lock:
            return self.state == "running" and not self.stop.is_set() and not self.error

    def start_recording(self) -> None:
        with self.lock:
            if not self.running:
                raise RuntimeError("Press F and wait for FR3 teleoperation before recording")
            self.samples = list(self.history)
            self.recording = True
            self.episode_interrupted = False

    def stop_recording(self) -> tuple[list[dict], bool]:
        with self.lock:
            self.recording = False
            samples, self.samples = self.samples, []
            return samples, self.episode_interrupted

    def _sample(self, state: dict, opening: float, gripper: float) -> None:
        sample = {**state, "receiver_monotonic_s": time.monotonic(),
                  "gripper_measured_m": opening, "gripper_command": gripper,
                  "spacemouse_action": self.last_action}
        with self.lock:
            self.telemetry = sample
            self.history.append(sample)
            if self.recording:
                if len(self.samples) >= int(self.settings.get("max_episode_samples", 240000)):
                    raise RuntimeError("FR3 episode sample limit exceeded")
                self.samples.append(sample)

    def _spawn_worker(self):
        raw = Path(self.settings.get("runtime_python", ".venv-fr3/bin/python"))
        python = raw if raw.is_absolute() else self.repo_root / raw
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RuntimeError(f"FR3 runtime is missing: {python}. Run run/setup_thor_fr3.sh on the controller computer")
        parent, child = socket.socketpair()
        channel = JsonChannel(parent)
        log_dir = self.repo_root / "outputs" / "logs" / "fr3_teleop"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"native_{time.time_ns()}.log").open("wb")
        try:
            process = subprocess.Popen(
                [str(python), "-m", "tools.thor.fr3_control_worker", "--config-path", str(self.config_path),
                 "--ipc-fd", str(child.fileno())],
                cwd=self.repo_root, env=native_environment(self.repo_root, python),
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            channel.close()
            raise
        finally:
            child.close()
            log.close()
        self.process = process
        self.pid = process.pid
        return process, channel

    def _run(self) -> None:
        process = channel = device = client = None
        gripper_mode = False
        failure = ""
        try:
            width = float(self.settings.get("gripper_max_width_m", 0.09))
            if not math.isfinite(width) or width <= 0:
                raise ValueError("gripper_max_width_m must be finite and positive")
            control_hz = float(self.settings.get("control_hz", 200))
            if not math.isfinite(control_hz) or not 1 < control_hz <= 500:
                raise ValueError("fr3_teleop.control_hz must be finite and in (1, 500]")
            client = gripper_client(self.box, str(self.settings.get("gripper_box_id") or ""))
            opening = measured_opening(client, width)
            device = make_spacemouse(self.config, opening / width, self.repo_root)
            if self.stop.is_set():
                return
            process, channel = self._spawn_worker()
            interval = 1 / control_hz
            startup_deadline = time.monotonic() + float(self.settings.get("startup_timeout_s", 60))
            ready = False
            last_state_s = last_publish_s = last_gripper_s = 0.0
            last_heartbeat_s = 0.0
            last_position = None
            last_native = None
            gripper = opening / width
            while not self.stop.is_set():
                started = time.monotonic()
                if not ready and started > startup_deadline:
                    raise TimeoutError("FR3 startup timed out; check FCI, robot mode, and native worker log")
                # The native worker sends low-rate telemetry, never pixels or
                # BOX packets. Bound draining so input delivery cannot starve.
                for _ in range(16):
                    if b"\n" not in channel.buffer and not select.select([channel.sock], [], [], 0)[0]:
                        break
                    try:
                        packet = channel.receive()
                    except socket.timeout:
                        break
                    state = packet.get("state")
                    if state == "error":
                        raise RuntimeError(packet.get("message") or "FR3 controller error")
                    if state == "stopped":
                        raise RuntimeError(packet.get("message") or "FR3 worker stopped")
                    if state in ("starting", "moving_to_start") and not ready:
                        self.publish(state, str(packet.get("message") or "Preparing FR3"))
                    if state == "running":
                        telemetry = packet.get("telemetry") or {}
                        if not telemetry:
                            continue
                        last_state_s = time.monotonic()
                        first_sample = not ready
                        if not ready:
                            # Homing completed. Seed the real BOX gripper from
                            # its measured opening, then enable control once.
                            opening = measured_opening(client, width)
                            gripper = opening / width
                            device.sync_gripper_baseline(gripper)
                            if client.set_clamp_pos(opening) != 0:
                                raise RuntimeError("BOX rejected the initial gripper hold")
                            gripper_mode = True
                            if client.set_mode(1) != 0 or client.set_clamp_pos(opening) != 0:
                                raise RuntimeError("BOX could not enter gripper control mode")
                            ready = True
                            last_position = opening
                        source_s = telemetry.get("sample_monotonic_s")
                        if source_s != last_native:
                            # Record the target accepted by BOX's rate-limited
                            # command path, rather than a pending mouse value.
                            self._sample(telemetry, opening, last_position / width)
                            last_native = source_s
                        if first_sample:
                            self.publish("running", "FR3 is at start. SpaceMouse control is active; press E to record")
                if process.poll() is not None:
                    raise RuntimeError(f"FR3 worker exited ({process.returncode}); inspect outputs/logs/fr3_teleop")
                if ready:
                    if time.monotonic() - last_state_s > float(self.settings.get("max_state_age_s", 0.1)):
                        raise RuntimeError("FR3 telemetry stopped arriving")
                    opening = measured_opening(client, width)
                    action = device.get_action()
                    self.last_action = {**action, "sample_monotonic_s": time.monotonic()}
                    gripper = float(action["gripper"])
                    if not math.isfinite(gripper) or not 0 <= gripper <= 1:
                        raise ValueError("SpaceMouse gripper command is invalid")
                    channel.send({"op": "action", "action": action, "sent_monotonic_s": time.monotonic()})
                    position = gripper * width
                    if started - last_gripper_s >= 1 / 15 and (last_position is None or abs(position - last_position) >= 0.0005):
                        if client.set_clamp_pos(position) != 0:
                            raise RuntimeError("BOX rejected the gripper command")
                        last_gripper_s, last_position = started, position
                else:
                    # Native imports may briefly hold their process's GIL.
                    # Startup needs a liveness signal, not a 200 Hz backlog.
                    if started - last_heartbeat_s >= 0.05:
                        channel.send({"op": "heartbeat", "sent_monotonic_s": time.monotonic()})
                        last_heartbeat_s = started
                if ready and started - last_publish_s >= 0.1:
                    self.publish("running", "SpaceMouse control active; E records, S saves, D discards")
                    last_publish_s = started
                self.stop.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            failure = pending_worker_error(channel) if channel is not None else ""
            failure = failure or str(exc)
            with self.lock:
                self.error = failure
                if self.recording:
                    self.episode_interrupted = True
            self.publish("stopping", f"{failure}; stopping FR3 before allowing a retry")
        finally:
            self.stop.set()
            if channel is not None:
                try:
                    channel.send({"op": "stop"})
                except Exception:
                    pass
                channel.close()
            if gripper_mode and client is not None:
                try:
                    self._restore_gripper_mode(client)
                except Exception as exc:
                    failure = failure or f"Could not restore BOX collection mode: {exc}"
            if process is not None:
                try:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                except Exception as exc:
                    failure = failure or f"Could not stop the native FR3 worker: {exc}"
            if device is not None:
                try:
                    device.disconnect()
                except Exception:
                    pass
            with self.lock:
                still_alive = process is not None and process.poll() is None
                self.process = process if still_alive else None
                self.pid = process.pid if still_alive else None
                if still_alive:
                    failure = f"{failure or 'FR3 shutdown failed'}; native worker {process.pid} is still alive"
                self.telemetry = {}
                # Cleanup is complete before exposing F again. A new request
                # must not be lost because this final status emitter is alive.
                self.thread = None
                if failure:
                    self.error = failure
                    if self.recording:
                        self.episode_interrupted = True
                    self.publish("error", f"{failure.rstrip('. ')}. Fix the reported problem; "
                                 "clear a robot fault in Desk only if one is reported, then press F to retry")
                else:
                    self.publish("idle", "FR3 stopped. Sensors remain connected; press F to move to start again")

    def _restore_gripper_mode(self, client) -> None:
        if client.set_mode(0) != 0:
            raise RuntimeError("BOX rejected collection mode")

    def close(self) -> None:
        self.request_stop()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=8)
            if thread.is_alive():
                raise RuntimeError("FR3 coordinator did not stop; inspect the native worker")
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError(f"Native FR3 worker {self.process.pid} has not exited; a second controller will be refused")


def write_fr3_samples(ep_dir: Path, samples: list[dict]) -> Path:
    path = ep_dir / "fr3_state.jsonl"
    with path.open("w") as output:
        for sample in samples:
            output.write(json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n")
    return path
