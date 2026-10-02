from __future__ import annotations

import io
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools.fr3.box_arm_server import ArmSession, serve_connection, validate_state
from tools.fr3.box_teleop_data import aligned_fr3_rows
from tools.fr3.box_teleop_protocol import BridgeClient, JsonConnection, Lease, neutral_action, validate_action
from tools.thor.fr3_teleop import ThorFr3Session, gripper_client
from tools.thor.gmsl2.thor_lerobot_v3 import Lr3Writer


def sample(at_s=100.0):
    return {"q": [0.1] * 7, "dq": [0.2] * 7, "tau_J": [1.2] * 7,
            "tau_ext_hat_filtered": [0.3] * 7, "O_T_EE": np.eye(4).flatten().tolist(),
            "O_F_ext_hat_K": [2.0] * 6, "measured_tcp": [0.4, 0, 0.3, 0, 0, 0],
            "sample_monotonic_s": at_s, "thor_sample_monotonic_s": at_s,
            "robot_time_s": at_s, "receiver_monotonic_s": at_s + 0.003,
            "clock_uncertainty_s": 0.002, "control_command_success_rate": 1.0,
            "robot_mode": "kMove", "commanded_ee": [0.4, 0, 0.3, 0, 0, 0], "gripper_command": 0.5}


class FakeRobot:
    def __init__(self):
        self.is_connected = False
        self.disconnected = threading.Event()
        self.actions = []
        self._last_command_pose = np.eye(4)
        self._arm = SimpleNamespace(get_telemetry=lambda: sample(time.monotonic()))

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False
        self.disconnected.set()

    def send_action(self, action):
        self.actions.append(dict(action))

    def _compute_ee_pose(self, _q):
        return np.eye(4)


@pytest.mark.parametrize("field,value", [("target_x", float("nan")), ("target_y", 0.01),
                                         ("target_wx", float("inf")), ("gripper", -0.1), ("enabled", 1)])
def test_reject_invalid_motion(field, value):
    action = neutral_action()
    action[field] = value
    with pytest.raises(ValueError):
        validate_action(action)


def test_lease_rejects_replay_and_expired_packet(monkeypatch):
    lease = Lease(0.15)
    packet = {"sequence": 1, "lease": lease.issue()}
    lease.accept(packet)
    with pytest.raises(ValueError):
        lease.accept(packet)
    packet = {"sequence": 2, "lease": lease.issue()}
    monkeypatch.setattr(time, "monotonic", lambda: lease.deadline_s + 0.001)
    with pytest.raises(ValueError):
        lease.accept(packet)


def test_bad_auth_does_not_construct_robot():
    from tools.fr3.box_teleop_protocol import authenticate
    with pytest.raises(ValueError, match="authentication"):
        authenticate({"version": 1, "token": "wrong"}, "x" * 64)


def test_bounded_packets_reject_non_objects_and_oversized_input():
    class Sock:
        def __init__(self, data):
            self.data = data
        def setsockopt(self, *_):
            pass
        def settimeout(self, *_):
            pass
        def recv(self, size):
            result, self.data = self.data[:size], self.data[size:]
            return result
    with pytest.raises(ValueError, match="object"):
        JsonConnection(Sock(b"[]\n"), 0.15).receive()
    with pytest.raises(ValueError, match="large"):
        JsonConnection(Sock(b"x" * 32768), 0.15).receive()


def test_native_clock_is_required():
    state = sample()
    del state["robot_time_s"]
    with pytest.raises(ValueError, match="robot_time_s"):
        validate_state(state)


def test_native_driver_enforces_rt_and_copies_coherent_state(monkeypatch):
    from tools.fr3.box_arm_server import validate_state
    from lerobot.robots.franka_research3.backends import PandaPyArmDriver
    construction = []
    state = SimpleNamespace(**{key: value for key, value in sample().items()
                               if key in ("q", "dq", "tau_J", "tau_ext_hat_filtered", "O_T_EE",
                                          "O_F_ext_hat_K", "control_command_success_rate", "robot_mode")},
                            time=SimpleNamespace(to_sec=lambda: 123.0))
    class Panda:
        def __init__(self, ip, **kwargs):
            construction.append((ip, kwargs))
        def get_state(self):
            return state
        def raise_error(self):
            pass
    monkeypatch.setitem(sys.modules, "panda_py", SimpleNamespace(Panda=Panda, controllers=SimpleNamespace(),
                         libfranka=SimpleNamespace(RealtimeConfig=SimpleNamespace(kEnforce="ENFORCE"))))
    driver = PandaPyArmDriver("192.168.1.206", realtime_enforce=True, start_controller_on_connect=False)
    monkeypatch.setattr(driver, "_start_state_reader", lambda: None)
    driver.connect()
    telemetry = driver.get_telemetry()
    assert construction == [("192.168.1.206", {"realtime_config": "ENFORCE"})]
    assert telemetry["robot_time_s"] == 123
    validate_state(telemetry)
    telemetry["tau_J"][0] = 99
    assert driver.get_telemetry()["tau_J"][0] == 1.2


def test_watchdog_stops_worker_and_does_not_replay_delta():
    robot = FakeRobot()
    session = ArmSession(robot, {"command_timeout_s": 0.05})
    session.start()
    action = {**neutral_action(), "enabled": True, "target_x": 0.001}
    session.submit(action, True)
    assert session.stop.wait(1)
    session.close()
    assert "watchdog" in session.error
    assert robot.actions == [action]
    assert robot.disconnected.is_set()


def test_frozen_native_clock_is_a_fault():
    robot = FakeRobot()
    at_s = time.monotonic()
    robot._arm.get_telemetry = lambda: {**sample(time.monotonic()), "robot_time_s": at_s}
    session = ArmSession(robot, {"command_timeout_s": 0.05})
    session.start()
    deadline = time.monotonic() + 0.5
    while not session.stop.is_set() and time.monotonic() < deadline:
        session.submit(neutral_action(), False)
        time.sleep(0.005)
    session.close()
    assert "clock stopped" in session.error


def test_localhost_bridge_auth_telemetry_and_disconnect():
    token = "x" * 64
    robot = FakeRobot()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        def serve():
            peer, _ = listener.accept()
            serve_connection(peer, token, {"fr3_teleop": {"command_timeout_s": 0.15}}, lambda _: robot)
        thread = threading.Thread(target=serve)
        thread.start()
        client = BridgeClient("127.0.0.1", port, token)
        try:
            client.connect()
            reply = client.exchange(neutral_action(), active=False)
            assert reply["state"]["tau_J"] == [1.2] * 7
            assert 0 <= reply["clock_uncertainty_s"] < 0.02
        finally:
            client.close()
            thread.join(timeout=1)
        assert robot.disconnected.is_set()
        assert not thread.is_alive()


def test_alignment_uses_capture_time_and_uncertainty():
    rows = aligned_fr3_rows([sample()], [0, 0.024, None, 0.5], t0_mono_s=100)
    assert rows[0]["fr3.valid"] == [1]
    assert rows[0]["observation.fr3.tau_J"] == [1.2] * 7
    assert rows[0]["action"][-1] == 0.5
    assert [r["fr3.valid"] for r in rows[1:]] == [[0], [0], [0]]
    assert rows[1]["observation.fr3.q"] == [0] * 7


def test_writer_preserves_box_fields_and_adds_fr3_across_episodes(tmp_path):
    import pyarrow.parquet as pq
    roles = {"camera": "cam_15", "cameras": {"cam_15": {"role": "wrist", "serial": "WRIST_MODULE"}}}
    writer = Lr3Writer(tmp_path, repo_id="local/test", task="test", fps=60, fr3_enabled=True, camera_roles=roles)
    for episode in range(2):
        writer.append_episode(episode_index=episode, snapshots=[{"t_relative_s": 0, "sensors": {}}],
                              duration_s=1 / 60, frame_times_s=[0], fr3_samples=[sample()], t0_mono_s=100)
    writer.finalize()
    table = pq.read_table(writer.data_path)
    assert table.num_rows == 2
    assert "box.timestamps" in table.column_names
    assert "observation.state" in table.column_names
    assert table["fr3.valid"].to_pylist() == [[1], [1]]
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["camera_roles"] == roles
    assert info["features"]["observation.fr3.tau_J"]["shape"] == [7]
    assert info["features"]["action"]["names"][-1] == "gripper.pos"


def test_export_preserves_fr3_action_and_torque(tmp_path):
    import pyarrow.parquet as pq
    from tools.fr3.box_teleop_data import FR3_FEATURE_NAMES
    from tools.thor.gmsl2.export_v3 import _V3Writer
    values = aligned_fr3_rows([sample()], [0], t0_mono_s=100)[0]
    writer = _V3Writer(tmp_path, repo_id="local/test", task="test", fps=60, height=1080, width=1920,
                       video_keys=[], state_width=28, state_names=None, fr3_enabled=True)
    writer.append_episode(episode_index=0, n_frames=1, state_rows=[[0] * 28], action_rows=[[9] * 28],
                          video_files={}, fr3_rows={key: [values[key]] for key in FR3_FEATURE_NAMES})
    writer.finalize()
    table = pq.read_table(writer.data_path)
    assert table["action"].to_pylist() == [[pytest.approx(0.4), 0, pytest.approx(0.3), 0, 0, 0, 0.5]]
    assert table["observation.fr3.tau_J"].to_pylist()[0] == pytest.approx([1.2] * 7)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [7]


def test_pool_never_falls_back_to_wrong_gripper():
    box = SimpleNamespace(_clients=[("box0", object()), ("box1", object())])
    with pytest.raises(RuntimeError):
        gripper_client(box, "")
    with pytest.raises(RuntimeError):
        gripper_client(box, "missing")
    assert gripper_client(box, "box1") is box._clients[1][1]


def test_gripper_start_seeds_measured_opening_and_stop_restores_mode(tmp_path, monkeypatch):
    import tools.thor.fr3_teleop as coordinator
    path = tmp_path / "token"
    path.write_text("x" * 64)
    calls = []
    client = SimpleNamespace(
        read=lambda: {"sensors": {"box_gripper": {"distance_m": 0.045}},
                      "status": {"sensor_status": {"box_gripper": {"fresh": True}}}},
        set_mode=lambda v: calls.append(("mode", v)) or 0,
        set_clamp_pos=lambda v: calls.append(("position", v)) or 0,
    )
    cfg = {"fr3_teleop": {"arm_host": "127.0.0.1", "arm_port": 18770, "token_file": str(path)}, "teleop": {}}
    session = ThorFr3Session(cfg, SimpleNamespace(_clients=[("", client)]), emit=lambda _: None)
    session.state = sample()
    device = SimpleNamespace(disconnect=lambda: calls.append(("disconnect", None)))
    monkeypatch.setattr(coordinator, "make_spacemouse", lambda cfg, grip: device if grip == 0.5 else None)
    session.set_active(True)
    assert calls[:3] == [("position", 0.045), ("mode", 1), ("position", 0.045)]
    assert session.gripper == 0.5
    session.set_active(False)
    assert calls[-1] == ("mode", 0)
    assert not session.active


def test_thor_gateway_teleop_uses_existing_recorder(monkeypatch):
    from tools.data_collection_gui import gateway
    root = Path(__file__).resolve().parents[2]
    state = gateway.make_state(root, Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    assert "fr3_bridge" in gateway._deployment_payload(state)["capabilities"]
    process = SimpleNamespace(poll=lambda: None)
    state.process = process
    gateway._apply_recorder_output(state, 'FR3_LIVE ' + json.dumps({"state": "idle", "realRobotReady": True,
                                                                 "telemetry": sample()}))
    commands = []
    monkeypatch.setattr(gateway, "_write_recorder_stdin", lambda p, command: commands.append(command))
    gateway._start_fr3_real_teleop(state)
    gateway._stop_fr3_teleop(state)
    assert commands == ["fr3_start\n", "fr3_stop\n"]
    assert state.teleop_process is None
    with pytest.raises(RuntimeError, match="dedicated RT workstation"):
        gateway._start_real_replay(state)


def test_stdin_teleop_commands_are_out_of_band(monkeypatch):
    from tools.thor.gmsl2 import thor_record
    monkeypatch.setattr("sys.stdin", io.StringIO("fr3_start\nfr3_stop\nquit\n"))
    queue = []
    commands = []
    thor_record._read_stdin_loop(queue, threading.Event(), on_fr3_teleop=commands.append)
    assert commands == [True, False]
    assert [cmd.kind for cmd in queue] == ["quit"]


def test_shipped_config_has_existing_tcp_and_valid_robot_settings():
    from tools.fr3.check_box_teleop import main
    assert main(["config"]) == 0


def test_preflight_checks_native_libraries_without_connecting(monkeypatch):
    from tools.fr3 import box_arm_server
    calls = []
    monkeypatch.setattr(box_arm_server, "realtime_preflight", lambda: calls.append("rt"))
    monkeypatch.setattr(box_arm_server.importlib, "import_module", lambda name: calls.append(name))
    robot = SimpleNamespace(_make_kinematics_driver=lambda: calls.append("model"))
    monkeypatch.setattr(box_arm_server, "make_robot", lambda _: robot)
    assert box_arm_server.main(["--check"]) == 0
    assert calls == ["rt", "panda_py", "ruckig", "model"]


def test_spacemouse_can_import_without_training_or_motor_dependencies():
    import subprocess
    result = subprocess.run([sys.executable, "-c", """
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
class BlockTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'accelerate', 'datasets', 'serial', 'deepdiff'):
            raise ImportError('unexpected training/motor dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
with TemporaryDirectory(prefix='lerobot-space-import-') as directory:
    device = SpaceMouseTeleop(SpaceMouseTeleopConfig(calibration_dir=Path(directory)))
    assert not device.is_connected
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
