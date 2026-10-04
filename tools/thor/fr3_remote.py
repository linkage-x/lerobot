"""Thor recording session backed by the host controller; BOX stays on Thor."""
from __future__ import annotations

import socket
import time

from tools.thor.fr3_ipc import JsonChannel
from tools.thor.fr3_link import ClockMap, LINK_TIMEOUT_S, PROTOCOL, TOKEN_PATH, config_digest, finite
from tools.thor.fr3_teleop import ThorFr3Session, gripper_client, measured_opening


class GripperCommands:
    """Strictly ordered, fresh, once-only BOX operations for a single F session."""
    def __init__(self, client, width):
        self.client, self.width = client, width
        self.serial = 0
        self.control_mode = False

    def apply(self, rpc, clock, now):
        if rpc is None:
            return None
        if type(rpc.get("id")) is not int or rpc["id"] != self.serial + 1:
            raise ValueError("Duplicate/out-of-order host gripper command")
        offset, uncertainty, _ = clock.estimate(now)
        age = now - (finite(rpc["host_sent_s"]) - offset)
        if age < -uncertainty or age + uncertainty > LINK_TIMEOUT_S:
            raise RuntimeError("Stale host gripper command rejected")
        value = finite(rpc["value"])
        kind = rpc.get("kind")
        if kind == "position" and 0 <= value <= self.width:
            result = self.client.set_clamp_pos(value)
        elif kind == "mode" and value in (0, 1):
            # Set before the SDK call so failure still attempts cleanup.
            self.control_mode = bool(value)
            result = self.client.set_mode(int(value))
        else:
            raise ValueError("Invalid host gripper operation or bounds")
        self.serial = rpc["id"]
        return {"id": self.serial, "result": result, "thor_applied_s": time.monotonic()}

    def close(self):
        if self.control_mode:
            if self.client.set_mode(0) != 0:
                raise RuntimeError("BOX rejected collection mode during host-link shutdown")
            self.control_mode = False


class RemoteFr3Session(ThorFr3Session):
    def _sample(self, state, opening, gripper):
        # Keep the host input snapshot instead of overwriting it with an empty
        # local SpaceMouse state. Receiver timestamp is always Thor monotonic.
        self.last_action = state.get("spacemouse_action", {})
        super()._sample(state, opening, gripper)

    def _run(self):
        channel = commands = None
        seq, previous_tx = 0, None
        failure = ""
        try:
            width = finite(self.settings.get("gripper_max_width_m", .09))
            if width <= 0:
                raise ValueError("Invalid gripper width")
            client = gripper_client(self.box, str(self.settings.get("gripper_box_id") or ""))
            commands = GripperCommands(client, width)
            # This endpoint is intentionally fixed to loopback: deploy owns the
            # SSH tunnel and no unauthenticated robot service is exposed to LAN.
            channel = JsonChannel(socket.create_connection(("127.0.0.1", 18766), timeout=1),
                                  timeout_s=LINK_TIMEOUT_S)
            channel.send({"op": "connect", "protocol": PROTOCOL,
                          "token": (self.repo_root / TOKEN_PATH).read_text().strip(),
                          "digest": config_digest(self.config)})
            hello = channel.receive()
            if hello.get("state") != "linked" or hello.get("protocol") != PROTOCOL:
                raise RuntimeError(hello.get("message") or "FR3 host handshake failed")
            clock = ClockMap()
            ack, last_sample = None, None
            last_publish = 0.
            deadline = time.monotonic() + float(self.settings.get("startup_timeout_s", 60))
            ready = False
            home_sent = home_started = False
            while not self.stop.is_set():
                tick = time.monotonic()
                opening = measured_opening(client, width)
                op = "probe" if seq < 8 else ("start" if seq == 8 else "tick")
                if ready and self.home_requested.is_set() and not home_sent:
                    op = "home"
                    home_sent = True
                t0 = time.monotonic()
                channel.send({"op": op, "seq": seq, "echo_host_s": previous_tx,
                              "opening_m": opening, "ack": ack})
                response = channel.receive()
                t3 = time.monotonic()
                if response.get("state") == "error":
                    raise RuntimeError(response.get("message") or "FR3 host error")
                if response.get("seq") != seq:
                    raise ValueError("Out-of-order FR3 host reply")
                clock.update(t0, response["host_rx_s"], response["host_tx_s"], t3)
                previous_tx = response["host_tx_s"]
                state = response.get("state")
                if seq >= 8 and state == "idle":
                    raise RuntimeError("Host controller stopped; press F for a new session")
                if state == "moving_to_start" and home_sent:
                    home_started = True
                if state == "running" and self.home_requested.is_set():
                    if not home_started:
                        state = "moving_to_start"  # An older running reply cannot complete this return.
                    else:
                        self.home_requested.clear()
                        self.history.clear()
                        home_sent = home_started = False
                telemetry = response.get("telemetry") or {}
                if state == "running":
                    if not telemetry:
                        raise RuntimeError("Host reported running without FR3 telemetry")
                    sample = clock.translate(telemetry, t3)
                    if sample["host_sample_monotonic_s"] != last_sample:
                        self._sample(sample, opening, sample["gripper_command"])
                        last_sample = sample["host_sample_monotonic_s"]
                    ready = True
                if not ready and t3 > deadline:
                    raise TimeoutError("Host FR3 startup timed out")
                ack = commands.apply(response.get("gripper_rpc"), clock, time.monotonic())
                if seq >= 8 and (state != self.state or t3 - last_publish >= .1):
                    self.publish(state, "Host: " + str(response.get("message", "")))
                    last_publish = t3
                seq += 1
                self.stop.wait(max(0., .02 - (time.monotonic() - tick)))
        except Exception as exc:
            failure = str(exc)
        finally:
            self.stop.set()
            if commands is not None:
                try:
                    commands.close()
                except Exception as exc:
                    failure = failure or str(exc)
            if channel is not None:
                try:
                    channel.send({"op": "stop", "seq": seq, "echo_host_s": previous_tx})
                    # Wait for native ownership release before exposing F again.
                    # If an error packet is pending, retain the original reason.
                    channel.sock.settimeout(6)
                    reply = channel.receive()
                    if reply.get("state") != "stopped" and not failure:
                        failure = "Host FR3 shutdown was not confirmed; wait for the host worker to exit before F"
                except Exception as exc:
                    failure = failure or f"Host FR3 shutdown acknowledgement unavailable: {exc}"
                channel.close()
            with self.lock:
                self.thread = None
                self.error = failure
                if failure:
                    if self.recording:
                        self.episode_interrupted = True
                    self.publish("error", f"{failure}. Fix the host/link problem; clear Desk only if a robot fault "
                                 "is reported. Release SpaceMouse, then press F to retry")
                else:
                    self.publish("idle", "Host FR3 stopped. Sensors remain connected; press F to restart")


def check_link(root, config, count=100):
    """Measure the real tunnel without opening BOX, SpaceMouse or FCI."""
    from statistics import median
    channel = JsonChannel(socket.create_connection(("127.0.0.1", 18766), timeout=1),
                          timeout_s=LINK_TIMEOUT_S)
    clock, previous_tx, results = ClockMap(), None, []
    try:
        channel.send({"op": "connect", "protocol": PROTOCOL,
                      "token": (root / TOKEN_PATH).read_text().strip(), "digest": config_digest(config)})
        hello = channel.receive()
        if hello.get("state") != "linked":
            raise RuntimeError(hello.get("message") or "Host handshake failed")
        for seq in range(count):
            t0 = time.monotonic()
            # Probe-only sessions cannot start a controller. Zero here is a
            # synthetic diagnostic opening, never used for a gripper command.
            channel.send({"op": "probe", "seq": seq, "echo_host_s": previous_tx, "opening_m": 0.})
            reply = channel.receive()
            t3 = time.monotonic()
            if reply.get("seq") != seq or reply.get("state") != "idle" or reply.get("gripper_rpc"):
                raise RuntimeError("Unexpected active controller during probe-only diagnostics")
            offset, uncertainty, _ = clock.update(t0, reply["host_rx_s"], reply["host_tx_s"], t3)
            previous_tx = reply["host_tx_s"]
            results.append(t3 - t0)
            time.sleep(max(0., .02 - (time.monotonic() - t0)))
        channel.send({"op": "stop", "seq": count, "echo_host_s": previous_tx})
        if channel.receive().get("state") != "stopped":
            raise RuntimeError("Probe session did not close cleanly")
        return {"probes": count, "rtt_median_ms": median(results) * 1000,
                "rtt_max_ms": max(results) * 1000, "host_minus_thor_s": offset,
                "estimated_uncertainty_ms": uncertainty * 1000, "motion_started": False}
    finally:
        channel.close()


if __name__ == "__main__":
    import json
    from pathlib import Path
    from tools.thor.fr3_control_worker import load_config
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(check_link(root, load_config(str(root / "tools/thor/gmsl2/thor_fr3_teleop.yaml"))), indent=2))
