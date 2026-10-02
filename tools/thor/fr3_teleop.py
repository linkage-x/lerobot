"""FR3 coordinator owned by the Thor recorder's existing camera/BOX session."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import socket
import threading
import time

from tools.fr3.box_teleop_protocol import (
    BridgeClient, JsonConnection, Lease, authenticate, neutral_action, read_token, validate_action,
)


def gripper_client(box, box_id: str):
    # BoxPool exposes collection operations publicly; its per-device client
    # handles already own the SDK socket. Keep this adaptation in one place.
    targets = [(bid, client) for bid, client in box._clients if not box_id or bid == box_id]
    if len(targets) != 1:
        raise RuntimeError("FR3 gripper requires exactly one BOX; configure gripper_box_id for multi-box rigs")
    return targets[0][1]


def make_spacemouse(config: dict, initial_gripper: float):
    from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
    raw = dict(config)
    raw.pop("type", None)
    raw["initial_gripper"] = initial_gripper
    device = SpaceMouseTeleop(SpaceMouseTeleopConfig(**raw))
    device.connect()
    return device


class ThorFr3Session:
    def __init__(self, config: dict, box, emit=print):
        self.settings = config["fr3_teleop"]
        self.teleop_config = config["teleop"]
        self.box = box
        self.emit = emit
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active = False
        self.error = ""
        self.state = {}
        self.device = None
        self.remote_action = None
        self.remote_at_s = 0.0
        self.remote_connected = False
        self.input_epoch = 0
        self.gripper = 1.0
        self.measured_gripper_m = None
        self.gripper_client = None
        self.gripper_mode = False
        self.last_gripper_at_s = 0.0
        self.last_gripper_m = None
        self.thread = None
        self.input_thread = None
        self.input_listener = None
        self.history = deque(maxlen=200)
        self.recording = None
        self.max_samples = int(self.settings.get("max_episode_samples", 240000))
        self.token = read_token(self.settings["token_file"])
        self.bridge = BridgeClient(self.settings["arm_host"], int(self.settings["arm_port"]), self.token,
                                   float(self.settings.get("command_timeout_s", 0.15)))

    def start(self) -> None:
        self.bridge.connect()
        self.thread = threading.Thread(target=self._run, name="thor-fr3-coordinator", daemon=True)
        self.thread.start()
        if self.settings.get("input_source", "thor") == "host":
            self.input_listener = socket.socket()
            self.input_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.input_listener.bind((self.settings.get("input_bind", "0.0.0.0"),
                                      int(self.settings.get("input_port", 18771))))
            self.input_listener.listen(1)
            self.input_listener.settimeout(0.2)
            self.input_thread = threading.Thread(target=self._serve_input, name="thor-spacemouse-input", daemon=True)
            self.input_thread.start()

    def _read_gripper(self) -> float:
        client = gripper_client(self.box, str(self.settings.get("gripper_box_id", "")))
        snap = client.read()
        data = snap.get("sensors", {}).get("box_gripper", {})
        status = snap.get("status", {}).get("sensor_status", {}).get("box_gripper", {})
        distance = data.get("distance_m")
        if distance is None or not math.isfinite(distance) or not status.get("fresh", False):
            raise RuntimeError("BOX gripper opening is unavailable or stale")
        self.gripper_client = client
        self.measured_gripper_m = float(distance)
        return float(distance)

    def set_active(self, active: bool) -> None:
        with self.lock:
            if not active:
                self.active = False
                if self.device is not None:
                    self.device.disconnect()
                    self.device = None
                self._release_gripper()
                self._emit_status()
                return
            if self.error or self.stop.is_set() or not self.state:
                raise RuntimeError(self.error or "FR3 state is not ready; reconnect devices")
            if self.active:
                return
            distance = self._read_gripper()
            self.gripper = min(1.0, max(0.0, distance / float(self.settings.get("gripper_max_width_m", 0.09))))
            if self.settings.get("input_source", "thor") == "thor":
                self.device = make_spacemouse(self.teleop_config, self.gripper)
            elif not self.remote_connected:
                raise RuntimeError("Start the host SpaceMouse sender before starting teleoperation")
            try:
                # Seed the actual opening before and immediately after the firmware mode switch.
                self.gripper_client.set_clamp_pos(distance)
                if self.gripper_client.set_mode(1) != 0:
                    raise RuntimeError("BOX could not enter gripper control mode")
                self.gripper_mode = True
                if self.gripper_client.set_clamp_pos(distance) != 0:
                    raise RuntimeError("BOX could not hold the measured gripper opening")
                self.last_gripper_m = distance
                self.last_gripper_at_s = time.monotonic()
                self.active = True
                self.input_epoch += 1
                self.remote_action = None
                self.remote_at_s = time.monotonic()
            except Exception:
                self.set_active(False)
                raise
            self._emit_status()

    def _release_gripper(self) -> None:
        if self.gripper_mode and self.gripper_client is not None:
            try:
                distance = self._read_gripper()
                self.gripper_client.set_clamp_pos(distance)
            finally:
                self.gripper_client.set_mode(0)
                self.gripper_mode = False

    def _emit_status(self) -> None:
        payload = {"state": "error" if self.error else "running" if self.active else "idle",
                   "backend": "real", "realRobotReady": bool(self.state) and not self.error,
                   "message": self.error or ("FR3 teleoperation active" if self.active else "FR3 telemetry connected; motion disabled"),
                   "telemetry": self.state, "inputSource": self.settings.get("input_source", "thor")}
        self.emit("FR3_LIVE " + json.dumps(payload, allow_nan=False, separators=(",", ":")))

    def _run(self) -> None:
        interval = 1.0 / float(self.settings.get("control_hz", 200))
        last_emit_s = 0.0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    if self.error:
                        raise RuntimeError(self.error)
                    active = self.active
                    action = neutral_action(self.gripper)
                    if active:
                        if self.device is not None:
                            action = self.device.get_action()
                        else:
                            if started - self.remote_at_s > float(self.settings.get("command_timeout_s", 0.15)):
                                raise TimeoutError("Host SpaceMouse input watchdog expired")
                            action = self.remote_action or action
                            self.remote_action = None  # consume each host delta once
                        action = validate_action(action)
                    try:
                        self._read_gripper()
                    except RuntimeError:
                        self.measured_gripper_m = None
                        if active:
                            raise  # freshness gate during control, not only at startup
                    reply = self.bridge.exchange(action, active=active)
                    state = reply["state"]
                    source_s = float(state["sample_monotonic_s"])
                    mapped_s = source_s + reply["clock_offset_s"]
                    age_s = reply["receiver_monotonic_s"] - mapped_s
                    uncertainty_s = reply["clock_uncertainty_s"]
                    if uncertainty_s > float(self.settings.get("max_clock_uncertainty_ms", 20)) / 1000:
                        raise RuntimeError("FR3 bridge clock uncertainty exceeds limit")
                    if age_s + uncertainty_s > float(self.settings.get("max_state_age_ms", 100)) / 1000:
                        raise RuntimeError("FR3 bridge state is stale")
                    self.gripper = action["gripper"]
                    now = time.monotonic()
                    if active and now - self.last_gripper_at_s >= 1 / 15:
                        target = self.gripper * float(self.settings.get("gripper_max_width_m", 0.09))
                        if abs(target - self.last_gripper_m) >= 0.0005:
                            if self.gripper_client.set_clamp_pos(target) != 0:
                                raise RuntimeError("BOX gripper command failed")
                            self.last_gripper_m = target
                        self.last_gripper_at_s = now
                    sample = {**state, "thor_sample_monotonic_s": mapped_s,
                              "receiver_monotonic_s": reply["receiver_monotonic_s"],
                              "clock_uncertainty_s": uncertainty_s, "round_trip_ms": reply["round_trip_ms"],
                              "gripper_command": self.gripper, "teleop_active": active,
                              "gripper_measured_m": self.measured_gripper_m,
                              "input_action": action}
                    self.state = sample
                    self.history.append(sample)
                    if self.recording is not None:
                        if len(self.recording) >= self.max_samples:
                            raise RuntimeError("FR3 episode sample limit exceeded; save shorter episodes")
                        self.recording.append(sample)
                    if now - last_emit_s >= 0.1:
                        self._emit_status()
                        last_emit_s = now
                self.stop.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                try:
                    self.set_active(False)
                except Exception:
                    pass
                self._emit_status()
            self.stop.set()
        finally:
            self.bridge.close()  # loss of owner stops the workstation controller

    def _serve_input(self) -> None:
        while not self.stop.is_set():
            try:
                sock, _ = self.input_listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            connection = JsonConnection(sock, 2.0)
            try:
                authenticate(connection.receive(), self.token)
                lease = Lease(float(self.settings.get("command_timeout_s", 0.15)))
                connection.timeout_s = lease.timeout_s
                with self.lock:
                    self.remote_connected = True
                connection.send({"lease": lease.issue(), "gripper": self.gripper, "active": self.active,
                                 "epoch": self.input_epoch})
                while not self.stop.is_set():
                    packet = connection.receive()
                    received_s = time.monotonic()
                    lease.accept(packet)
                    action = validate_action(packet.get("action"))
                    with self.lock:
                        self.remote_action = action if packet.get("epoch") == self.input_epoch and self.active else None
                        self.remote_at_s = time.monotonic()
                    connection.send({"sequence": packet["sequence"], "lease": lease.issue(),
                                     "gripper": self.gripper, "active": self.active,
                                     "epoch": self.input_epoch,
                                     "server_received_s": received_s, "server_sent_s": time.monotonic()})
            except Exception as exc:
                self.emit(f"Host SpaceMouse input disconnected: {exc}")
            finally:
                sock.close()
                with self.lock:
                    self.remote_connected = False
                    self.remote_action = None
                    if self.active:
                        self.error = "Host SpaceMouse disconnected; reconnect devices"
                        self.set_active(False)

    def start_recording(self) -> None:
        with self.lock:
            self.recording = list(self.history)

    def stop_recording(self) -> list[dict]:
        with self.lock:
            samples = self.recording or []
            self.recording = None
            return samples

    def close(self) -> None:
        self.stop.set()
        if self.input_listener is not None:
            self.input_listener.close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.input_thread is not None:
            self.input_thread.join(timeout=1.0)
        self.bridge.close()
        self.set_active(False)


def write_fr3_samples(episode_dir: Path, samples: list[dict], t0_mono_s: float) -> dict:
    path = episode_dir / "fr3_state.jsonl"
    with path.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps({**sample, "t_relative_s": sample["thor_sample_monotonic_s"] - t0_mono_s},
                                    allow_nan=False, separators=(",", ":")) + "\n")
    return {"enabled": True, "samples": len(samples), "state_file": path.name,
            "clock": "workstation CLOCK_MONOTONIC mapped to Thor with request/response midpoint",
            "t0_monotonic_s": t0_mono_s,
            "max_clock_uncertainty_ms": max((s["clock_uncertainty_s"] * 1000 for s in samples), default=None)}
