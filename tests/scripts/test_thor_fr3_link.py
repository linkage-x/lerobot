"""Distributed lifecycle and timing tests, without robot/USB/BOX hardware."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.thor.fr3_data import align_fr3_samples
from tools.thor.fr3_host import serve_session
from tools.thor.fr3_ipc import JsonChannel
from tools.thor.fr3_link import ClockMap, RemoteBox, config_digest
from tools.thor.fr3_remote import GripperCommands, RemoteFr3Session
from tools.thor.fr3_teleop import ThorFr3Session


def test_clock_mapping_handles_unrelated_monotonic_origins_and_uncertainty():
    clock = ClockMap()
    offset, uncertainty, rtt = clock.update(100, 501.001, 501.002, 100.003)
    assert offset == pytest.approx(401)
    assert uncertainty == pytest.approx(.001)
    assert rtt == pytest.approx(.002)
    sample = clock.translate({"sample_monotonic_s": 500.999,
                              "spacemouse_action": {"sample_monotonic_s": 500.998},
                              "gripper_ack": {"host_sent_s": 500.990, "thor_applied_s": 99.995}}, 100.003)
    assert sample["sample_monotonic_s"] == pytest.approx(99.999)
    assert sample["host_sample_monotonic_s"] == 500.999
    assert sample["spacemouse_action"]["sample_monotonic_s"] == pytest.approx(99.998)
    assert sample["clock_sync_valid"]
    assert sample["gripper_ack"]["thor_sent_estimate_s"] == pytest.approx(99.990)
    assert sample["gripper_ack"]["thor_applied_s"] == 99.995
    with pytest.raises(RuntimeError, match="expired"):
        clock.estimate(106)
    with pytest.raises(RuntimeError, match="stale"):
        clock.translate({"sample_monotonic_s": 490}, 100.004)


def test_one_delayed_link_turn_reuses_a_good_clock_probe_and_drops_stale_data():
    clock = ClockMap()
    baseline = clock.update(100, 501.001, 501.002, 100.003)
    # A 279 ms scheduler pause is within the bounded link lease, but its
    # timestamps are unsuitable for synchronizing recorded robot samples.
    offset, uncertainty, rtt = clock.update(100.01, 501.011, 501.012, 100.29)
    assert offset == pytest.approx(baseline[0])
    assert rtt == pytest.approx(baseline[2])
    assert uncertainty > baseline[1]
    assert len(clock.probes) == 1
    assert clock.translate({"sample_monotonic_s": 501.01}, 100.29, drop_stale=True) is None
    with pytest.raises(RuntimeError, match="future-dated"):
        clock.translate({"sample_monotonic_s": 502.0}, 100.29, drop_stale=True)


@pytest.mark.parametrize("times", [(1, 2, 2, 1.11), (1, 2, 1.9, 1.001), (1, 2, 2, 1.03)])
def test_clock_rejects_delayed_or_invalid_probes(times):
    with pytest.raises(RuntimeError):
        ClockMap().update(*times)


def test_camera_alignment_charges_clock_uncertainty_against_skew_budget():
    sample = {"sample_monotonic_s": 100., "receiver_monotonic_s": 100.001,
              "clock_sync_valid": True, "clock_uncertainty_s": .009,
              "commanded_ee": [0.] * 6, "gripper_command": .5}
    rows = align_fr3_samples([sample], [.01, .02], 100., n_frames=2)
    assert rows[0]["fr3.action_valid"] == [1.]
    assert rows[1]["fr3.action_valid"] == [0.]
    sample["clock_sync_valid"] = False
    assert align_fr3_samples([sample], [0], 100., n_frames=1)[0]["fr3.action_valid"] == [0.]


class Box:
    def __init__(self):
        self.calls = []

    def read(self):
        return {"sensors": {"box_gripper": {"distance_m": .045}},
                "status": {"sensor_status": {"box_gripper": {"fresh": True}}}}

    def set_clamp_pos(self, value):
        self.calls.append(("position", value))
        return 0

    def set_mode(self, value):
        self.calls.append(("mode", value))
        return 0


def test_gripper_rejects_replay_stale_and_out_of_bounds_commands():
    box = Box()
    commands = GripperCommands(box, .09)
    clock = ClockMap()
    clock.update(100, 500.001, 500.001, 100.002)
    command = {"id": 1, "kind": "position", "value": .06, "host_sent_s": 500.001}
    assert commands.apply(command, clock, 100.003)["result"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        commands.apply(command, clock, 100.004)
    with pytest.raises(RuntimeError, match="Stale"):
        commands.apply({**command, "id": 2, "host_sent_s": 499}, clock, 100.005)
    with pytest.raises(ValueError, match="bounds"):
        commands.apply({**command, "id": 2, "value": 1}, clock, 100.006)
    assert box.calls == [("position", .06)]


def test_remote_box_requires_real_ack_and_fails_closed():
    box = RemoteBox()
    box.update(.045, None)
    result = []
    worker = threading.Thread(target=lambda: result.append(box.set_clamp_pos(.06)))
    worker.start()
    deadline = time.monotonic() + 1
    while box.pending is None and time.monotonic() < deadline:
        time.sleep(.001)
    assert result == []
    box.update(.045, {"id": box.pending["id"], "result": 0})
    worker.join(1)
    assert result == [0]
    assert box.last_ack["value"] == .06 and box.last_ack["result"] == 0
    box.fail("link lost")
    with pytest.raises(RuntimeError, match="link lost"):
        box.set_clamp_pos(.02)


def test_remote_box_timeout_never_reports_unacknowledged_success():
    box = RemoteBox()
    box.update(.045, None)
    with pytest.raises(RuntimeError, match="acknowledge"):
        box.set_mode(1)
    with pytest.raises(RuntimeError):
        box.read()


class FakeHostSession(ThorFr3Session):
    """Exercise the real cross-machine session but replace only arm/USB work."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.message = "idle"
        self.commands_done = threading.Event()
        self.pause_samples = threading.Event()
        self.closed = False
        super().__init__(*args, **kwargs)
        self.instances.append(self)

    def publish(self, state, message):
        with self.lock:
            self.state, self.message = state, message

    def _run(self):
        client = self.box._clients[0][1]
        try:
            assert client.set_clamp_pos(.045) == 0
            assert client.set_mode(1) == 0
            assert client.set_clamp_pos(.06) == 0
            self.commands_done.set()
            while not self.stop.is_set():
                if self.home_requested.is_set():
                    self.publish("moving_to_start", "fake returning")
                    self.stop.wait(.04)
                    self.home_requested.clear()
                    self.history.clear()
                client.read()
                if not self.pause_samples.is_set():
                    self._sample({"sample_monotonic_s": time.monotonic(), "q": [1.] * 7,
                                  "control_command_success_rate": 1.}, .045, .06 / .09)
                self.publish("running", "fake ready")
                self.stop.wait(.01)
        except RuntimeError:
            # The fake arm observes the same BOX lease failure as the real
            # session; its finally block stands in for native shutdown.
            self.stop.set()
        finally:
            self.closed = True


@pytest.mark.parametrize("disconnect", [False, True])
def test_f_remote_lifecycle_record_gripper_stop_and_fresh_retry(tmp_path, monkeypatch, disconnect):
    config = {"fr3_teleop": {"gripper_max_width_m": .09}, "robot": {}, "teleop": {}}
    token = tmp_path / "outputs/secrets/fr3_host.token"
    token.parent.mkdir(parents=True)
    token.write_text("test-token")
    client = Box()
    errors = []
    hosts = []
    sockets = []

    def connect(*args, **kwargs):
        thor_sock, host_sock = socket.socketpair()
        sockets.append(thor_sock)
        def serve():
            channel = JsonChannel(host_sock, .2)
            try:
                hello = channel.receive()
                serve_session(channel, hello, config, Path("fake.yaml"), tmp_path, FakeHostSession)
            except (EOFError, OSError):
                pass
            except Exception as exc:
                errors.append(exc)
            finally:
                channel.close()
        host = threading.Thread(target=serve)
        host.start()
        hosts.append(host)
        return thor_sock

    monkeypatch.setattr(socket, "create_connection", connect)
    session = RemoteFr3Session(config, tmp_path / "fake.yaml", tmp_path,
                               SimpleNamespace(_clients=[("box", client)]), emit=lambda _: None)
    assert session.state == "idle" and not hosts  # C opens neither host nor USB/FCI.
    for _ in range(2):
        session.request_start()
        deadline = time.monotonic() + 3
        while not session.running and time.monotonic() < deadline:
            time.sleep(.005)
        assert session.running, session.error
        session.start_recording()
        time.sleep(.06)
        samples, interrupted = session.stop_recording()
        assert samples and not interrupted
        assert samples[-1]["clock_sync_valid"]
        assert samples[-1]["gripper_command"] == pytest.approx(2 / 3)
        assert session.request_home()
        assert not session.running
        deadline = time.monotonic() + 3
        while not session.running and time.monotonic() < deadline:
            time.sleep(.005)
        assert session.running, session.error
        if disconnect:
            sockets[-1].shutdown(socket.SHUT_RDWR)
            deadline = time.monotonic() + 2
            while session.thread is not None and time.monotonic() < deadline:
                time.sleep(.005)
            assert session.state == "error"
        session.close()
        hosts[-1].join(2)
        assert not hosts[-1].is_alive()
        assert FakeHostSession.instances[-1].closed
        assert client.calls[-1] == ("mode", 0)
    assert errors == []


def test_short_cached_host_telemetry_gap_does_not_abort_the_link(tmp_path, monkeypatch):
    config = {"fr3_teleop": {"gripper_max_width_m": .09}, "robot": {}, "teleop": {}}
    token = tmp_path / "outputs/secrets/fr3_host.token"
    token.parent.mkdir(parents=True)
    token.write_text("test-token")
    hosts = []
    notices = []

    def connect(*_args, **_kwargs):
        thor_sock, host_sock = socket.socketpair()
        def serve():
            channel = JsonChannel(host_sock, .4)
            try:
                serve_session(channel, channel.receive(), config, Path("fake.yaml"), tmp_path, FakeHostSession)
            finally:
                channel.close()
        thread = threading.Thread(target=serve)
        thread.start()
        hosts.append(thread)
        return thor_sock

    monkeypatch.setattr(socket, "create_connection", connect)
    session = RemoteFr3Session(config, tmp_path / "fake.yaml", tmp_path,
                               SimpleNamespace(_clients=[("box", Box())]), emit=notices.append)
    try:
        session.request_start()
        deadline = time.monotonic() + 3
        while not session.running and time.monotonic() < deadline:
            time.sleep(.005)
        assert session.running, session.error
        fake = FakeHostSession.instances[-1]
        fake.pause_samples.set()
        time.sleep(.26)
        assert session.running, session.error
        initial = session.history[-1]["host_sample_monotonic_s"]
        fake.pause_samples.clear()
        deadline = time.monotonic() + 1
        while session.history[-1]["host_sample_monotonic_s"] == initial and time.monotonic() < deadline:
            time.sleep(.005)
        assert session.history[-1]["host_sample_monotonic_s"] > initial
        assert any("telemetry delayed" in item for item in notices)
        assert any("telemetry recovered" in item for item in notices)
    finally:
        session.close()
        hosts[-1].join(2)


def test_host_rejects_config_mismatch_before_session_creation(tmp_path):
    config = {"fr3_teleop": {}, "robot": {}, "teleop": {}}
    before = len(FakeHostSession.instances)
    with pytest.raises(ValueError, match="configurations differ"):
        serve_session(None, {"digest": "wrong"}, config, tmp_path / "fake", tmp_path, FakeHostSession)
    assert len(FakeHostSession.instances) == before


def test_host_link_loss_stops_session_and_never_starts_on_probe(tmp_path):
    config = {"fr3_teleop": {}, "robot": {}, "teleop": {}}
    parent, child = socket.socketpair()
    channel = JsonChannel(parent, .5)
    errors = []

    def serve():
        host = JsonChannel(child, .05)
        try:
            serve_session(host, {"digest": config_digest(config)}, config,
                          tmp_path / "fake", tmp_path, FakeHostSession)
        except (EOFError, OSError) as exc:
            errors.append(exc)
        finally:
            host.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert channel.receive()["state"] == "linked"
    channel.send({"op": "probe", "seq": 0, "opening_m": .045})
    response = channel.receive()
    assert response["state"] == "idle"
    instance = FakeHostSession.instances[-1]
    assert instance.thread is None
    # Abrupt socket loss, no stop packet: service still requests stop.
    channel.close()
    thread.join(1)
    assert not thread.is_alive()
    assert instance.stop.is_set()
    assert errors


def test_host_expired_echo_cannot_renew_lease(tmp_path):
    config = {"fr3_teleop": {}, "robot": {}, "teleop": {}}
    class FakeChannel:
        def __init__(self):
            self.responses = []
            self.reads = 0
        def send(self, value):
            self.responses.append(value)
        def receive(self):
            self.reads += 1
            return {"op": "probe", "seq": self.reads - 1,
                    "opening_m": .045, "echo_host_s": -1}
    channel = FakeChannel()
    with pytest.raises(RuntimeError, match="lease expired"):
        serve_session(channel, {"digest": config_digest(config)}, config,
                      tmp_path / "fake", tmp_path, FakeHostSession)
    assert FakeHostSession.instances[-1].stop.is_set()
