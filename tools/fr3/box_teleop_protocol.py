"""Bounded, lease-based transport for high-level teleop (never FCI packets)."""
from __future__ import annotations

import hmac
import json
import math
import os
from pathlib import Path
import secrets
import socket
import time

VERSION = 1
MAX_PACKET = 32768
AXES = ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz")


def read_token(path: str) -> str:
    token = Path(os.environ.get("FR3_BRIDGE_TOKEN_FILE", path)).read_text().strip()
    if len(token) < 32:
        raise ValueError("FR3 bridge token must contain at least 32 characters")
    return token


class JsonConnection:
    def __init__(self, sock: socket.socket, timeout_s: float):
        self.sock = sock
        self.timeout_s = timeout_s
        self.buffer = bytearray()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, packet: dict) -> None:
        data = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(data) > MAX_PACKET:
            raise ValueError("FR3 packet is too large")
        self.sock.settimeout(self.timeout_s)
        self.sock.sendall(data)

    def receive(self) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("FR3 packet deadline expired")
            self.sock.settimeout(remaining)
            chunk = self.sock.recv(min(4096, MAX_PACKET - len(self.buffer)))
            if not chunk:
                raise ConnectionError("FR3 peer disconnected")
            self.buffer.extend(chunk)
            if len(self.buffer) >= MAX_PACKET:
                raise ValueError("FR3 packet is too large")
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        packet = json.loads(line)
        if not isinstance(packet, dict):
            raise ValueError("FR3 packet must be an object")
        return packet


def authenticate(packet: dict, token: str) -> None:
    if packet.get("version") != VERSION or not isinstance(packet.get("token"), str):
        raise ValueError("Invalid FR3 bridge handshake")
    if not hmac.compare_digest(packet["token"], token):
        raise ValueError("FR3 bridge authentication failed")


def validate_action(action: dict, max_translation: float = 0.002, max_rotation: float = 0.02) -> dict:
    if not isinstance(action, dict) or type(action.get("enabled")) is not bool:
        raise ValueError("Invalid FR3 action")
    result = {"enabled": action["enabled"]}
    for index, key in enumerate(AXES + ("gripper",)):
        value = action.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid FR3 action field {key}")
        limit = max_translation if index < 3 else max_rotation
        if key == "gripper":
            if not 0 <= value <= 1:
                raise ValueError("gripper must be in [0, 1]")
        elif abs(value) > limit:
            raise ValueError(f"FR3 action exceeds {key} limit")
        result[key] = float(value)
    return result


def neutral_action(gripper: float = 1.0) -> dict:
    return {"enabled": False, **dict.fromkeys(AXES, 0.0), "gripper": gripper}


class Lease:
    """A delayed or duplicated packet cannot acquire a fresh motion lease."""
    def __init__(self, timeout_s: float):
        if not 0.02 <= timeout_s <= 0.5:
            raise ValueError("command_timeout_s must be in [0.02, 0.5]")
        self.timeout_s = timeout_s
        self.sequence = -1
        self.nonce = ""
        self.deadline_s = 0.0

    def issue(self) -> str:
        self.nonce = secrets.token_hex(16)
        self.deadline_s = time.monotonic() + self.timeout_s
        return self.nonce

    def accept(self, packet: dict) -> None:
        sequence = packet.get("sequence")
        if (type(sequence) is not int or sequence <= self.sequence
                or packet.get("lease") != self.nonce or time.monotonic() > self.deadline_s):
            raise ValueError("Expired, duplicate, or invalid FR3 command lease")
        self.sequence = sequence
        self.nonce = ""  # one use; only a response grants the next lease


class BridgeClient:
    def __init__(self, host: str, port: int, token: str, timeout_s: float = 0.15):
        self.host, self.port, self.token = host, port, token
        self.timeout_s = timeout_s
        self.sock = None
        self.connection = None
        self.lease = ""
        self.sequence = 0

    def connect(self) -> dict:
        self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self.connection = JsonConnection(self.sock, 30.0)
        self.connection.send({"version": VERSION, "token": self.token})
        reply = self.connection.receive()
        if reply.get("error"):
            self.close()
            raise RuntimeError(reply["error"])
        self.lease = reply["lease"]
        self.connection.timeout_s = self.timeout_s
        return reply

    def exchange(self, action: dict, *, active: bool, **fields) -> dict:
        if self.connection is None:
            raise RuntimeError("FR3 bridge is disconnected")
        sent_s = time.monotonic()
        self.connection.send({"sequence": self.sequence, "lease": self.lease,
                              "action": action, "active": active, **fields})
        reply = self.connection.receive()
        received_s = time.monotonic()
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        if reply.get("sequence") != self.sequence:
            raise ValueError("FR3 response sequence mismatch")
        self.sequence += 1
        self.lease = reply["lease"]
        t2, t3 = reply["server_received_s"], reply["server_sent_s"]
        uncertainty = max(0.0, (received_s - sent_s - (t3 - t2)) / 2)
        offset = (sent_s + received_s - t2 - t3) / 2
        reply.update(receiver_monotonic_s=received_s, clock_offset_s=offset,
                     clock_uncertainty_s=uncertainty, round_trip_ms=(received_s - sent_s) * 1000)
        return reply

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        self.sock = self.connection = None
