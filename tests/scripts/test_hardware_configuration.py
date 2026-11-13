from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lerobot.hardware import HardwareModel, catalog, catalog as list_models, get_model, register_model
from lerobot.hardware.catalog import model_fields
from tools.data_collection_gui import gateway, hardware_config as hardware

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def base():
    return yaml.safe_load((REPO / "tools/fr3/fr3_record_config.yaml").read_text())


@pytest.fixture
def document(base):
    document = hardware.from_runtime(base)
    document["mounting"]["urdf_path"] = str(
        REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"
    )
    return document


def test_catalog_exposes_configurable_models_without_connecting_hardware(monkeypatch):
    from lerobot.robots.franka_research3 import FrankaResearch3

    monkeypatch.setattr(
        FrankaResearch3, "connect", lambda *a, **k: pytest.fail("configuration must not connect")
    )
    models = {(m["kind"], m["type"]): m for m in catalog()}
    assert models["robot", "franka_research3"]["available"]
    assert models["robot", "so101_follower"]["available"]
    assert models["end_effector", "franka_hand"]["compatibleRobots"] == ["franka_research3"]
    assert {f["name"] for f in models["camera", "intelrealsense"]["fields"]} >= {
        "serial_number_or_name",
        "width",
        "height",
        "fps",
    }


def test_existing_yaml_round_trip_preserves_mounting_and_camera_bindings(base, document):
    runtime, _ = hardware.compile_setup(document, base, REPO)
    assert document["endEffector"]["type"] == "pika"
    assert runtime["robot"]["target_frame_name"] == base["robot"]["target_frame_name"]
    assert runtime["robot"]["workspace_min"] == base["robot"]["workspace_min"]
    for name, camera in base["robot"]["cameras"].items():
        assert all(runtime["robot"]["cameras"][name][key] == value for key, value in camera.items())
    assert runtime["robot"]["allow_mock_gripper"] is False
    assert "recorder" not in runtime


def test_moving_setup_changes_connection_only(base, document):
    before, _ = hardware.compile_setup(document, base, REPO)
    document["robot"]["params"]["robot_ip"] = "192.168.20.42"
    document["endEffector"]["params"]["port"] = "/dev/serial/by-id/new-gripper"
    after, _ = hardware.compile_setup(document, base, REPO)
    assert after["robot"]["robot_ip"] == "192.168.20.42"
    assert after["robot"]["gripper_port"] == "/dev/serial/by-id/new-gripper"
    for key in (
        "urdf_path",
        "target_frame_name",
        "start_joint_positions",
        "workspace_min",
        "workspace_max",
        "cameras",
    ):
        assert before["robot"][key] == after["robot"][key]


def test_switching_to_franka_hand_uses_its_own_model_and_tcp(base, document):
    hand = get_model("end_effector", "franka_hand")
    document["endEffector"] = {"type": hand.type, "params": {}}
    document["mounting"] = copy.deepcopy(hand.mounting)
    document["teleop"]["params"]["tool_mode"] = "binary"
    runtime, warnings = hardware.compile_setup(document, base, REPO)
    assert runtime["robot"]["gripper_backend"] == "franka_hand"
    assert runtime["robot"]["target_frame_name"] == "fr3_hand_tcp"
    assert runtime["robot"]["urdf_path"].endswith("fr3_franka_hand.urdf")
    assert any("binary" in warning for warning in warnings)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["cameras"].append(copy.deepcopy(d["cameras"][0])), "unique"),
        (lambda d: d["cameras"].append({**copy.deepcopy(d["cameras"][0]), "name": "other"}), "already bound"),
        (lambda d: d["cameras"][0]["params"].update(width=0), "positive"),
        (lambda d: d["cameras"][0]["params"].update(fps="60"), "number"),
        (lambda d: d["robot"]["params"].update(robot_ip=""), "IP is required"),
        (lambda d: d["mounting"].update(target_frame_name="missing_tcp"), "TCP"),
        (lambda d: d["robot"]["params"].update(unrecognized_option=True), "unknown parameters"),
        (lambda d: d.update(id="../../escape"), "Configuration id"),
        (lambda d: d["cameras"][0].update(name="../../escape"), "Camera role"),
    ],
)
def test_invalid_setups_are_rejected_before_runtime_creation(base, document, mutation, message):
    mutation(document)
    with pytest.raises(ValueError, match=message):
        hardware.compile_setup(document, base, REPO)


def test_native_joint_robot_does_not_inherit_fr3_commands(base, document):
    document.update(
        robot={"type": "so101_follower", "params": {"port": "/dev/so101", "id": "my-arm"}},
        teleop={"type": "so101_leader", "params": {"port": "/dev/leader", "id": "my-leader"}},
        endEffector=None,
        mounting={},
    )
    runtime, warnings = hardware.compile_setup(document, base, REPO)
    assert runtime["robot"]["type"] == "so101_follower"
    assert runtime["robot"]["port"] == "/dev/so101"
    assert "robot_ip" not in runtime["robot"]
    assert "gripper_backend" not in runtime["robot"]
    assert runtime["auto_move_to_start_after_episode"] is False
    assert runtime["move_to_start_after_last_episode"] is False
    assert any("Native-schema" in warning for warning in warnings)


def test_saved_draft_does_not_change_applied_revision(tmp_path, base, document):
    root = tmp_path / "hardware"
    runtime, warnings = hardware.compile_setup(document, base, REPO)
    first = hardware.save_setup(root, document, runtime, warnings, None)
    assert hardware.active_setup(root) is None
    runtime_path = hardware.activate_setup(root, first)
    document = copy.deepcopy(document)
    document["label"] = "Unapplied draft"
    document["robot"]["params"]["robot_ip"] = "10.0.0.42"
    changed, warnings = hardware.compile_setup(document, base, REPO)
    second = hardware.save_setup(root, document, changed, warnings, first["revision"])
    assert hardware.active_setup(root)["revision"] == first["revision"]
    assert yaml.safe_load(runtime_path.read_text())["robot"]["robot_ip"] != "10.0.0.42"
    assert hardware.read_setup(root, document["id"])["revision"] == second["revision"]
    with pytest.raises(ValueError, match="another browser"):
        hardware.save_setup(root, document, changed, warnings, first["revision"])


def _state(tmp_path, base):
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(base))
    return gateway.make_state(tmp_path, source, profile="workstation")


def test_apply_and_restart_restore_selection_without_overwriting_source(tmp_path, base, document):
    state = _state(tmp_path, base)
    source = state.config_path
    original = source.read_bytes()
    runtime, warnings = hardware.compile_setup(document, base, REPO)
    saved = hardware.save_setup(gateway._hardware_store(state), document, runtime, warnings, None)
    gateway._apply_hardware_config(state, saved)
    assert source.read_bytes() == original
    assert state.config_path != source
    assert state.teleop.robotModel == "franka_research3"
    assert state.teleop.inputDevice == "spacemouse"
    assert state.devices[0]["kind"] == "camera"
    restored = gateway.make_state(tmp_path, source, profile="workstation")
    assert restored.hardware_active["revision"] == saved["revision"]
    assert restored.config["robot"]["robot_ip"] == runtime["robot"]["robot_ip"]
    gateway._apply_hardware_config(restored, None)
    assert restored.config_path == source
    assert hardware.active_setup(gateway._hardware_store(restored)) is None


@pytest.mark.parametrize(
    "attribute", ["process", "teleop_process", "replay_process", "rollout_process", "training_process"]
)
def test_running_sessions_block_configuration_application(tmp_path, base, document, attribute):
    state = _state(tmp_path, base)
    runtime, warnings = hardware.compile_setup(document, base, REPO)
    saved = hardware.save_setup(gateway._hardware_store(state), document, runtime, warnings, None)
    setattr(state, attribute, SimpleNamespace(poll=lambda: None))
    with pytest.raises(ValueError, match="Stop"):
        gateway._apply_hardware_config(state, saved)
    assert hardware.active_setup(gateway._hardware_store(state)) is None
    assert state.config == base


def _request(state, path, body=None):
    handler = gateway.DataCollectionGuiHandler.__new__(gateway.DataCollectionGuiHandler)
    handler.server = SimpleNamespace(state=state)
    handler.path = path
    encoded = json.dumps(body).encode() if body is not None else b""
    handler.headers = {"Content-Length": str(len(encoded))}
    handler.rfile = io.BytesIO(encoded)
    handler.wfile = io.BytesIO()
    status = []
    handler.send_response = lambda code: status.append(code)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    if body is None:
        handler.do_GET()
    else:
        handler.do_POST()
    return status[-1], json.loads(handler.wfile.getvalue())


def test_browser_api_save_apply_read_and_stale_revision(tmp_path, base, document):
    state = _state(tmp_path, base)
    # The temporary test workspace shares model assets but never connects to hardware.
    document["mounting"]["urdf_path"] = str(
        REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"
    )
    code, listing = _request(state, "/api/hardware")
    assert code == 200 and listing["models"]
    code, preview = _request(state, "/api/hardware/validate", {"document": document})
    assert code == 200 and "robot_ip" in preview["runtimeYaml"]
    assert not gateway._hardware_store(state).exists()
    code, saved = _request(state, "/api/hardware/save", {"document": document})
    assert code == 200
    code, applied = _request(
        state, "/api/hardware/activate", {"id": document["id"], "revision": saved["revision"]}
    )
    assert code == 200 and applied["activeId"] == document["id"]
    code, stale = _request(state, "/api/hardware/activate", {"id": document["id"], "revision": "old"})
    assert code == 400 and stale["ok"] is False
    code, loaded = _request(state, f"/api/hardware/setup?id={document['id']}")
    assert code == 200 and loaded["document"] == document


def test_franka_hand_is_shown_as_network_gripper():
    devices = gateway._device_statuses(
        {"robot": {"type": "franka_research3", "robot_ip": "10.0.0.8", "gripper_backend": "franka_hand"}}
    )
    hand = next(d for d in devices if d["kind"] == "gripper")
    assert hand["id"] == "franka_hand"
    assert hand["label"] == "Franka Hand"
    assert hand["detail"] == "10.0.0.8"
    assert not any(d["id"] == "pika" for d in devices)


def test_native_pipeline_preserves_joint_actions_and_features():
    from lerobot.hardware.recording import native_recording_features, native_recording_processors

    config = SimpleNamespace(robot=SimpleNamespace(type="so101_follower"))
    robot = SimpleNamespace(
        action_features={"joint.pos": float, "gripper.pos": float},
        observation_features={"joint.pos": float, "gripper.pos": float},
    )
    teleop = SimpleNamespace(action_features=dict(robot.action_features))
    processors = native_recording_processors(config, robot, teleop)
    action = {"joint.pos": 32.0, "gripper.pos": 0.4}
    assert processors[0]((action, {})) == action
    features = native_recording_features(robot, teleop, processors, use_videos=False)
    assert features["action"]["names"] == ["joint.pos", "gripper.pos"]
    assert features["observation.state"]["names"] == ["joint.pos", "gripper.pos"]
    assert "ee.x" not in features["action"]["names"]


def test_native_pipeline_rejects_an_unmapped_action_schema():
    from lerobot.hardware.recording import native_recording_processors

    config = SimpleNamespace(robot=SimpleNamespace(type="so101_follower"))
    with pytest.raises(ValueError, match="schemas differ"):
        native_recording_processors(
            config,
            SimpleNamespace(action_features={"joint.pos": float}),
            SimpleNamespace(action_features={"target_x": float}),
        )


@pytest.mark.parametrize("model", ["so100_follower", "so101_follower", "so100_leader", "so101_leader"])
def test_selected_so_model_survives_decode_and_encode(model):
    import draccus

    from lerobot.robots import RobotConfig
    from lerobot.teleoperators import TeleoperatorConfig

    catalog()
    config_class = RobotConfig if model.endswith("follower") else TeleoperatorConfig
    decoded = draccus.decode(config_class, {"type": model, "port": "/dev/test"})
    assert decoded.type == model
    assert draccus.encode(decoded)["type"] == model


def test_another_robot_can_assemble_its_native_end_effector_config(monkeypatch, base, document):
    import importlib
    from dataclasses import field

    registry = importlib.import_module("lerobot.hardware.catalog")
    monkeypatch.setattr(registry, "_MODELS", dict(registry._MODELS))

    @dataclass
    class NativeConfig:
        address: str
        native_tool: dict[str, str] = field(default_factory=dict)
        cameras: dict = field(default_factory=dict)

    @dataclass
    class ToolConfig:
        port: str

    def assemble(config):
        return {
            "type": "test_native_robot",
            "address": config["address"],
            "native_tool": {"type": config["gripper_backend"], "port": config["tool_connection"]["port"]},
            "cameras": config["cameras"],
        }

    register_model(
        HardwareModel(
            "robot",
            "test_native_robot",
            "Native robot",
            NativeConfig,
            exclude=("cameras",),
            assemble=assemble,
        )
    )
    register_model(
        HardwareModel(
            "end_effector",
            "test_native_tool",
            "Native tool",
            ToolConfig,
            compatible_robots=("test_native_robot",),
            robot_fields={"port": "tool_connection.port"},
        )
    )
    document.update(
        robot={"type": "test_native_robot", "params": {"address": "10.0.0.8"}},
        endEffector={"type": "test_native_tool", "params": {"port": "/dev/native-tool"}},
        mounting={},
    )
    runtime, _ = hardware.compile_setup(document, base, REPO)
    assert runtime["robot"]["native_tool"] == {"type": "test_native_tool", "port": "/dev/native-tool"}
    assert "gripper_backend" not in runtime["robot"]


def test_plugin_adds_form_and_executable_end_effector_adapter(monkeypatch, base, document):
    import importlib

    registry = importlib.import_module("lerobot.hardware.catalog")
    monkeypatch.setattr(registry, "_MODELS", dict(registry._MODELS))

    @dataclass
    class Config:
        port: str
        speed: float = 0.1

    class Driver:
        def __init__(self, config):
            self.config = config

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_position(self):
            return 0.5

        def set_position(self, position):
            pass

    model = HardwareModel(
        "end_effector",
        "test_gripper",
        "Test gripper",
        Config,
        primary=("port",),
        compatible_robots=("franka_research3",),
        robot_fields={"port": "gripper_config.port", "speed": "gripper_config.speed"},
        mounting=get_model("end_effector", "pika").mounting,
        end_effector_factory=Driver,
    )
    register_model(model)
    assert any(m["type"] == "test_gripper" for m in list_models())
    assert model_fields(model)[0]["name"] == "port"
    document.update(endEffector={"type": "test_gripper", "params": {"port": "/dev/custom"}})
    runtime, _ = hardware.compile_setup(document, base, REPO)
    from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config

    robot_config = FrankaResearch3Config(
        **{k: v for k, v in runtime["robot"].items() if k not in {"type", "cameras"}}
    )
    driver = FrankaResearch3._make_gripper_driver(SimpleNamespace(config=robot_config))
    assert isinstance(driver, Driver)
    assert driver.config.port == "/dev/custom"
    assert driver.get_position() == 0.5
