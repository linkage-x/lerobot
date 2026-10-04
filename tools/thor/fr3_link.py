"""Host/Thor protocol: bounded stop-and-wait traffic and monotonic clock mapping.

Transport is loopback TCP through a host-initiated SSH reverse tunnel. A separate
random token prevents other local users of either computer from starting an arm.
No FCI command is ever relayed over this connection.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import time

PROTOCOL = 1
LINK_TIMEOUT_S = 0.4
MAX_RTT_S = 0.1
MAX_UNCERTAINTY_S = 0.01
MAX_TELEMETRY_AGE_S = 0.2
TOKEN_PATH = "outputs/secrets/fr3_host.token"


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Non-finite FR3 link number")
    return float(value)


def config_digest(config):
    # Only controller settings: camera selection and Thor sensor paths may differ.
    settings = dict(config["fr3_teleop"])
    for key in ("execution_host", "runtime_python"):
        settings.pop(key, None)
    data = {"robot": config["robot"], "teleop": config["teleop"], "fr3_teleop": settings}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ClockMap:
    """NTP four-timestamp estimate, lowest RTT in a recent 5-second window.

    Offset is host minus Thor. Uncertainty includes half RTT and a conservative
    200 ppm drift allowance. This is approximate synchronization, not PTP.
    """
    def __init__(self):
        self.probes = deque(maxlen=300)

    def update(self, t0, t1, t2, t3):
        t0, t1, t2, t3 = map(finite, (t0, t1, t2, t3))
        rtt = (t3 - t0) - (t2 - t1)
        if t3 < t0 or t2 < t1 or rtt < -1e-6 or t3 - t0 > LINK_TIMEOUT_S:
            raise RuntimeError("FR3 host link lease expired or has invalid timestamps")
        if t3 - t0 > MAX_RTT_S:
            # One delayed SSH scheduling turn is not a valid clock probe. Use
            # an earlier low-RTT observation instead of poisoning the clock
            # fit or tearing down a healthy native controller.
            return self.estimate(t3)
        self.probes.append((t3, max(0., rtt), ((t1 - t0) + (t2 - t3)) / 2))
        while self.probes and t3 - self.probes[0][0] > 5:
            self.probes.popleft()
        return self.estimate(t3)

    def estimate(self, now):
        recent = [p for p in self.probes if 0 <= now - p[0] <= 5]
        if not recent:
            raise RuntimeError("FR3 host clock synchronization expired")
        stamp, rtt, offset = min(recent, key=lambda p: p[1])
        uncertainty = rtt / 2 + (now - stamp) * 0.0002
        if uncertainty > MAX_UNCERTAINTY_S:
            raise RuntimeError("FR3 host clock uncertainty exceeds 10 ms")
        return offset, uncertainty, rtt

    def translate(self, sample, now, *, drop_stale=False):
        offset, uncertainty, rtt = self.estimate(now)
        source = finite(sample["sample_monotonic_s"])
        mapped = source - offset
        # Allow only the stated estimation uncertainty, never silently clamp.
        if mapped > now + uncertainty:
            raise RuntimeError("FR3 host telemetry is stale or future-dated")
        if now - mapped + uncertainty > MAX_TELEMETRY_AGE_S:
            if drop_stale:
                return None
            raise RuntimeError("FR3 host telemetry is stale or future-dated")
        result = {**sample, "host_sample_monotonic_s": source,
                  "host_receiver_monotonic_s": sample.get("receiver_monotonic_s"),
                  "sample_monotonic_s": mapped, "clock_host_minus_thor_s": offset,
                  "clock_uncertainty_s": uncertainty, "clock_rtt_s": rtt,
                  "clock_sync_valid": True, "receiver_monotonic_s": now}
        if sample.get("spacemouse_action"):
            action = dict(sample["spacemouse_action"])
            action["host_sample_monotonic_s"] = action["sample_monotonic_s"]
            action["sample_monotonic_s"] -= offset
            result["spacemouse_action"] = action
        if sample.get("gripper_ack"):
            result["gripper_ack"] = {
                **sample["gripper_ack"],
                "thor_sent_estimate_s": finite(sample["gripper_ack"]["host_sent_s"]) - offset,
            }
        return result


class RemoteBox:
    """Host-side BOX proxy. Commands require a real Thor SDK acknowledgement.

    Only the session thread calls commands. The link thread supplies fresh
    measured opening and acknowledgements. No motion is replayed after timeout.
    """
    def __init__(self):
        import threading
        self.condition = threading.Condition()
        self.updated = 0.
        self.opening = None
        self.pending = None
        self.result = None
        self.serial = 0
        self.failure = ""
        self.last_ack = None

    def fail(self, reason):
        with self.condition:
            self.failure = str(reason)
            self.condition.notify_all()

    def update(self, opening, ack):
        opening = finite(opening)
        with self.condition:
            self.opening, self.updated = opening, time.monotonic()
            if ack is not None:
                if self.pending is None or ack.get("id") != self.pending["id"]:
                    raise ValueError("Unexpected BOX acknowledgement")
                self.result = ack["result"]
                self.last_ack = {**self.pending, **ack}
                self.pending = None
            self.condition.notify_all()

    def read(self):
        with self.condition:
            if self.failure or time.monotonic() - self.updated > LINK_TIMEOUT_S:
                raise RuntimeError(self.failure or "Thor BOX/link heartbeat is stale")
            return {"sensors": {"box_gripper": {"distance_m": self.opening}},
                    "status": {"sensor_status": {"box_gripper": {"fresh": True}}}}

    def command(self, kind, value):
        with self.condition:
            self.read()
            self.serial += 1
            self.pending = {"id": self.serial, "kind": kind, "value": value,
                            "host_sent_s": time.monotonic()}
            self.result = None
            deadline = time.monotonic() + LINK_TIMEOUT_S
            while self.result is None and not self.failure:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.failure = "Thor did not acknowledge the gripper command within 200 ms"
                    break
                self.condition.wait(remaining)
            if self.failure:
                raise RuntimeError(self.failure)
            return self.result

    def set_clamp_pos(self, value):
        return self.command("position", value)

    def set_mode(self, value):
        return self.command("mode", value)
