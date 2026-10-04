"""Host SpaceMouse/FR3 service, accessed only over the deploy SSH tunnel.

Starting this service never opens USB or FCI. A fresh authenticated F session,
clock qualification, and live Thor BOX measurements are required for motion.
"""
from __future__ import annotations

import argparse
import hmac
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from tools.thor.fr3_control_worker import load_config
from tools.thor.fr3_ipc import JsonChannel
from tools.thor.fr3_link import PROTOCOL, TOKEN_PATH, LINK_TIMEOUT_S, config_digest, finite, RemoteBox
from tools.thor.fr3_teleop import ThorFr3Session


class HostSession(ThorFr3Session):
    def _sample(self, state, opening, gripper):
        box = self.box._clients[0][1]
        with box.condition:
            state = {**state, "gripper_ack": box.last_ack}
        super()._sample(state, opening, gripper)

    def _restore_gripper_mode(self, client):
        # Thor owns BOX teardown, including when the link is already gone.
        # Never fabricate an acknowledgement or wait for network on shutdown.
        pass

    def publish(self, state, message):
        with self.lock:
            changed = state != self.state or message != getattr(self, "message", "")
            self.state = state
            self.message = message
        if changed:
            print(f"FR3 host {state}: {message}", flush=True)


def serve_session(channel, hello, config, config_path, root, session_factory=HostSession):
    if hello.get("digest") != config_digest(config):
        raise ValueError("Host/Thor controller configurations differ; rerun deploy.sh")
    channel.send({"protocol": PROTOCOL, "state": "linked"})
    box = RemoteBox()
    pool = SimpleNamespace(_clients=[(str(config["fr3_teleop"].get("gripper_box_id") or "box"), box)])
    session = session_factory(config, config_path, root, pool)
    seq = 0
    previous_tx = None
    started = False
    orderly_stop = False
    try:
        while True:
            request = channel.receive()
            rx = time.monotonic()
            if request.get("seq") != seq or request.get("op") not in ("probe", "start", "tick", "stop"):
                raise ValueError("Invalid FR3 host session sequence or operation")
            if previous_tx is not None:
                if request.get("echo_host_s") != previous_tx or rx - previous_tx > LINK_TIMEOUT_S:
                    raise RuntimeError("Thor link lease expired; press F for a fresh session")
            if request["op"] == "stop":
                orderly_stop = True
                break
            opening = finite(request["opening_m"])
            width = float(config["fr3_teleop"].get("gripper_max_width_m", .09))
            if not 0 <= opening <= width:
                raise ValueError("Thor opening outside configured range")
            box.update(opening, request.get("ack"))
            if request["op"] == "start":
                if started or seq < 8:
                    raise ValueError("FR3 start requires a fresh qualified clock session")
                started = True
                session.request_start()
            elif request["op"] == "tick" and not started:
                raise ValueError("F start was not requested")
            with session.lock, box.condition:
                response = {"seq": seq, "host_rx_s": rx, "host_tx_s": time.monotonic(),
                            "state": session.state, "message": session.message,
                            "telemetry": dict(session.telemetry), "gripper_rpc": box.pending}
            previous_tx = response["host_tx_s"]
            channel.send(response)
            seq += 1
    finally:
        box.fail("Thor link closed; gripper commands disabled")
        session.close()  # Native ownership is released before accepting another F.
        if orderly_stop:
            channel.send({"state": "stopped"})


def serve(config_path, root, token_path, port):
    # Import input dependencies before accepting a leased connection. Imports
    # may briefly hold the GIL; no HID handle is opened by these imports.
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop  # noqa: F401

    config = load_config(str(config_path))
    token = token_path.read_text().strip()
    if len(token) < 32:
        raise ValueError("FR3 host token must contain at least 32 characters")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        print(f"FR3 host ready on loopback:{port}; USB/FCI idle until F", flush=True)
        while True:
            sock, _ = listener.accept()
            channel = JsonChannel(sock, timeout_s=LINK_TIMEOUT_S)
            try:
                hello = channel.receive()
                if hello.get("protocol") != PROTOCOL or not hmac.compare_digest(str(hello.get("token", "")), token):
                    raise ValueError("FR3 host authentication failed")
                if hello.get("op") in ("health", "shutdown"):
                    channel.send({"state": "idle", "digest": config_digest(config)})
                    if hello["op"] == "shutdown":
                        return
                elif hello.get("op") == "connect":
                    serve_session(channel, hello, config, config_path, root)
                else:
                    raise ValueError("Unknown FR3 host operation")
            except Exception as exc:
                print(f"FR3 host session ended: {exc}", flush=True)
                try:
                    channel.send({"state": "error", "message": str(exc)})
                except (OSError, ValueError):
                    pass
            finally:
                channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--shutdown-if-idle", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.health or args.shutdown or args.shutdown_if_idle:
        try:
            sock = socket.create_connection(("127.0.0.1", args.port), timeout=1)
        except ConnectionRefusedError:
            if args.shutdown_if_idle:
                return
            raise
        channel = JsonChannel(sock, timeout_s=1)
        try:
            channel.send({"protocol": PROTOCOL, "token": (root / TOKEN_PATH).read_text().strip(),
                          "op": "shutdown" if args.shutdown or args.shutdown_if_idle else "health"})
            result = channel.receive()
            if result.get("state") != "idle":
                raise RuntimeError(result)
            print(json.dumps(result))
        finally:
            channel.close()
    else:
        serve(args.config_path.resolve(), root, root / TOKEN_PATH, args.port)


if __name__ == "__main__":
    main()
