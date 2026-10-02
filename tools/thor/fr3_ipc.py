"""Bounded, local-only messages between the Thor recorder and arm worker."""
from __future__ import annotations

import json
import socket
import threading

MAX_PACKET_BYTES = 32768


class JsonChannel:
    def __init__(self, sock: socket.socket, timeout_s: float = 0.05):
        self.sock = sock
        self.sock.settimeout(timeout_s)
        self.buffer = bytearray()
        self.send_lock = threading.Lock()

    def send(self, packet: dict) -> None:
        if not isinstance(packet, dict):
            raise ValueError("FR3 IPC requires a JSON object")
        data = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(data) > MAX_PACKET_BYTES:
            raise ValueError("FR3 IPC packet exceeds size limit")
        with self.send_lock:
            self.sock.sendall(data)

    def receive(self) -> dict:
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise EOFError("FR3 IPC peer disconnected")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_PACKET_BYTES and b"\n" not in self.buffer:
                raise ValueError("FR3 IPC packet exceeds size limit")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        if len(line) + 1 > MAX_PACKET_BYTES:
            raise ValueError("FR3 IPC packet exceeds size limit")
        packet = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(packet, dict):
            raise ValueError("FR3 IPC requires a JSON object")
        return packet

    def close(self) -> None:
        self.sock.close()
