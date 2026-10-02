"""Exercise C/F/E/S/error/retry with a local fake worker; no robot or USB opens."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools.thor import fr3_teleop as teleop
from tools.thor.fr3_ipc import JsonChannel, MAX_PACKET_BYTES


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("fake FR3 lifecycle timed out")
        time.sleep(0.005)


class Box:
    def __init__(self):
        self.calls = []
        self.command_history = []
        self.fresh = True

    def read(self):
        return {"sensors": {"box_gripper": {"distance_m": 0.045}},
                "status": {"sensor_status": {"box_gripper": {"fresh": self.fresh}}}}

    def set_clamp_pos(self, position):
        self.calls.append(("position", position))
        self.command_history.append((time.monotonic(), position))
        return 0

    def set_mode(self, mode):
        self.calls.append(("mode", mode))
        return 0


class Mouse:
    def __init__(self):
        self.closed = False
        self.baseline = None

    def sync_gripper_baseline(self, value):
        self.baseline = value

    def get_action(self):
        return {"enabled": False, "target_x": 0, "target_y": 0, "target_z": 0,
                "target_wx": 0, "target_wy": 0, "target_wz": 0, "gripper": self.baseline}

    def disconnect(self):
        self.closed = True


class LocalWorker:
    """Use the actual IPC framing and socket loss semantics, with fake motion."""
    def __init__(self):
        parent, child = socket.socketpair()
        self.channel = JsonChannel(parent)
        self.native = JsonChannel(child)
        self.pid = 123
        self.returncode = None
        self.error = threading.Event()
        self.commands = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            self.native.receive()  # wait for the recorder heartbeat
            for state in ("starting", "moving_to_start"):
                self.native.send({"state": state, "message": state})
            while True:
                if self.error.is_set():
                    self.native.send({"state": "error", "message": "native communication reflex"})
                    self.returncode = 1
                    return
                self.native.send({"state": "running", "telemetry": {
                    "q": [0.1] * 7, "sample_monotonic_s": time.monotonic(),
                    "control_command_success_rate": 1.0}})
                packet = self.native.receive()
                self.commands.append(packet)
                if packet["op"] == "stop":
                    return
                time.sleep(0.005)
        except (EOFError, OSError):
            pass
        finally:
            self.native.close()
            if self.returncode is None:
                self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise subprocess.TimeoutExpired("fake worker", timeout)
        return self.returncode

    def terminate(self):
        self.native.close()

    kill = terminate


@pytest.fixture
def session(tmp_path, monkeypatch):
    client = Box()
    pool = SimpleNamespace(_clients=[("box-a", client)])
    output = []
    config = {"fr3_teleop": {"control_hz": 200, "gripper_max_width_m": 0.09}}
    value = teleop.ThorFr3Session(config, tmp_path / "config.yaml", tmp_path, pool, emit=output.append)
    mice, workers = [], []

    def mouse(*_):
        device = Mouse()
        mice.append(device)
        return device

    def worker():
        process = LocalWorker()
        workers.append(process)
        value.process, value.pid = process, process.pid
        return process, process.channel

    monkeypatch.setattr(teleop, "make_spacemouse", mouse)
    monkeypatch.setattr(value, "_spawn_worker", worker)
    yield value, client, mice, workers, output
    value.close()


def test_c_is_sensor_only_f_starts_e_s_preserve_teleop_and_fault_allows_f_retry(session):
    value, box, mice, workers, output = session
    assert value.state == "idle" and not mice and not workers and not box.calls
    with pytest.raises(RuntimeError, match="Press F"):
        value.start_recording()
    value.request_start()
    wait_for(lambda: value.running and len(value.history) > 1)
    assert box.calls[:3] == [("position", 0.045), ("mode", 1), ("position", 0.045)]
    assert mice[0].baseline == pytest.approx(0.5)
    value.start_recording()
    wait_for(lambda: len(value.samples) > 3)
    samples, interrupted = value.stop_recording()
    assert samples and not interrupted and value.running
    assert len(workers) == 1 and not mice[0].closed
    value.start_recording()
    workers[0].error.set()
    wait_for(lambda: value.state == "error")
    assert "native communication reflex" in value.error
    assert value.stop_recording()[1] is True
    assert box.calls[-1] == ("mode", 0) and mice[0].closed
    assert not value.telemetry
    value.request_start()
    wait_for(lambda: value.running and len(value.history) > 1)
    assert len(workers) == 2 and value.error == ""
    value.close()
    assert value.state == "idle" and mice[1].closed
    assert any(json.loads(line.removeprefix("FR3_LIVE "))["state"] == "moving_to_start" for line in output)


def test_stale_box_cancels_f_before_native_start(session):
    value, box, mice, workers, _ = session
    box.fresh = False
    value.request_start()
    wait_for(lambda: value.state == "error")
    assert "stale" in value.error and not mice and not workers


def test_lost_box_stream_stops_arm_and_interrupts_episode(session):
    value, box, mice, _, _ = session
    value.request_start()
    wait_for(lambda: value.running)
    value.start_recording()
    box.fresh = False
    wait_for(lambda: value.state == "error")
    assert value.stop_recording()[1] and mice[0].closed
    assert box.calls[-1] == ("mode", 0)


def test_recorded_gripper_target_is_the_command_box_accepted(session):
    value, box, mice, _, _ = session
    value.request_start()
    wait_for(lambda: value.running)
    original = mice[0].get_action
    ticks = [0]
    def changing_command():
        ticks[0] += 1
        return {**original(), "gripper": 0.25 if ticks[0] % 2 else 0.75}
    mice[0].get_action = changing_command
    value.start_recording()
    wait_for(lambda: ticks[0] >= 20)
    samples, interrupted = value.stop_recording()
    assert samples and not interrupted
    assert len(box.command_history) < ticks[0]  # BOX commands are rate limited
    for sample in samples:
        accepted = [position for received, position in box.command_history
                    if received <= sample["receiver_monotonic_s"]]
        assert sample["gripper_command"] == pytest.approx(accepted[-1] / 0.09)


def test_stop_during_episode_marks_discard_and_keeps_collection_pool(session):
    value, box, _, workers, _ = session
    value.request_start()
    wait_for(lambda: value.running)
    value.start_recording()
    value.close()
    assert value.stop_recording()[1]
    assert value.box._clients[0][1] is box
    assert workers[0].poll() == 0


def test_measured_opening_selects_existing_box_and_rejects_ambiguous_fleet():
    a, b = Box(), Box()
    pool = SimpleNamespace(_clients=[("a", a), ("b", b)])
    assert teleop.gripper_client(pool, "b") is b
    with pytest.raises(RuntimeError, match="multiple"):
        teleop.gripper_client(pool, "")
    with pytest.raises(RuntimeError, match="not connected"):
        teleop.gripper_client(pool, "unknown")


def test_ipc_handles_partial_coalesced_packets_and_eof():
    a, b = socket.socketpair()
    channel = JsonChannel(a, timeout_s=0.01)
    try:
        b.sendall(b'{"state":')
        with pytest.raises(socket.timeout):
            channel.receive()
        b.sendall(b'"running"}\n{"state":"error"}\n')
        assert channel.receive() == {"state": "running"}
        assert channel.receive() == {"state": "error"}
        b.close()
        with pytest.raises(EOFError):
            channel.receive()
    finally:
        channel.close()
        b.close()


@pytest.mark.parametrize("data", [b'[]\n', b'{"value":NaN}\n', b'x' * MAX_PACKET_BYTES + b'\n'], ids=["array", "nan", "oversize"])
def test_ipc_rejects_nonobjects_nonfinite_and_oversize(data):
    a, b = socket.socketpair()
    channel = JsonChannel(a)
    try:
        b.sendall(data)
        with pytest.raises(ValueError):
            channel.receive()
        with pytest.raises(ValueError):
            channel.send({"value": float("nan")})
    finally:
        channel.close()
        b.close()


def test_light_spacemouse_import_does_not_pull_training_or_motor_dependencies(tmp_path):
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'accelerate', 'datasets', 'serial', 'deepdiff'}:
            raise AssertionError('unexpected collection dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from pathlib import Path
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
device = SpaceMouseTeleop(SpaceMouseTeleopConfig(calibration_dir=Path(sys.argv[1])))
assert not device.is_connected
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=15)


def test_native_queued_error_survives_immediate_worker_exit():
    a, b = socket.socketpair()
    channel = JsonChannel(a)
    b.sendall(b'{"state":"starting"}\n{"state":"error","message":"PREEMPT_RT required"}\n')
    b.close()
    try:
        assert teleop.pending_worker_error(channel) == "PREEMPT_RT required"
    finally:
        channel.close()


def test_unconfirmed_worker_exit_cannot_spawn_another_controller(tmp_path):
    output = []
    value = teleop.ThorFr3Session({"fr3_teleop": {}}, tmp_path / "config", tmp_path, None, emit=output.append)
    value.process = SimpleNamespace(pid=42, poll=lambda: None)
    value.pid = 42
    value.request_start()
    assert value.state == "error" and value.thread is None and value.pid == 42
    assert "previous FR3 worker" in value.error or "previous FR3 worker" in output[-1]
    with pytest.raises(RuntimeError, match="second controller"):
        value.close()
