from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.data_collection_gui import gateway
from tools.thor.gmsl2.camera_roles import require_wrist_camera, resolve_camera_roles, wrist_camera_selector


def config(**selector):
    return {"fr3_teleop": {"enabled": True, "wrist_camera": selector}}


def cameras(*ids):
    return [{"name": f"cam_{sid:02d}", "sensor_id": sid} for sid in ids]


@pytest.mark.parametrize("selector", [{"sensor_id": -1}, {"sensor_id": 16}, {"sensor_id": True},
                                      {"sensor_id": "15"}, {"serial": 123}, []])
def test_invalid_selectors_are_rejected(selector):
    with pytest.raises(ValueError, match="wrist_camera"):
        wrist_camera_selector({"fr3_teleop": {"wrist_camera": selector}})


def test_serial_follows_physical_module_to_a_different_port():
    roles = resolve_camera_roles(config(serial="WRIST", sensor_id=6), cameras(6, 15),
                                 {"cam_06": {"serial": "SCENE"}, "cam_15": {"serial": "WRIST"}})
    assert roles["camera"] == "cam_15"
    assert roles["cameras"]["cam_15"]["role"] == "wrist"
    assert roles["cameras"]["cam_06"]["role"] == "unassigned"
    require_wrist_camera(roles)


@pytest.mark.parametrize("identity,state", [({}, "missing"),
    ({"cam_06": {"serial": "OTHER"}, "cam_15": {"serial": "OTHER2"}}, "missing"),
    ({"cam_06": {"serial": "WRIST"}, "cam_15": {"serial": "WRIST"}}, "ambiguous")])
def test_serial_never_falls_back_to_wrong_or_ambiguous_camera(identity, state):
    roles = resolve_camera_roles(config(serial="WRIST", sensor_id=6), cameras(6, 15), identity)
    assert roles["state"] == state
    assert roles["camera"] is None
    with pytest.raises(RuntimeError):
        require_wrist_camera(roles)


def test_explicit_port_works_without_eeprom_and_with_custom_stream_prefix():
    roles = resolve_camera_roles(config(sensor_id=13), [{"name": "sengyun_13", "sensor_id": 13}], {})
    assert roles["camera"] == "sengyun_13"
    require_wrist_camera(roles)
    roles = resolve_camera_roles(config(serial="WRIST"), [{"name": "sengyun_13", "sensor_id": 13}],
                                 {"cam_13": {"serial": "WRIST"}})
    assert roles["camera"] == "sengyun_13"


def test_future_camera_can_stay_unconfigured_until_installation():
    roles = resolve_camera_roles(config(), cameras(6), {})
    assert roles["state"] == "unconfigured"
    require_wrist_camera(roles)


def test_gateway_discovers_wrist_plugged_after_startup_and_uses_current_id(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(gateway, "_detect_locked_sids", lambda _: [6])
    state = gateway.make_state(root, Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    state.config["fr3_teleop"]["wrist_camera"] = {"serial": "WRIST"}
    roles = resolve_camera_roles(state.config, cameras(6, 15),
                                 {"cam_06": {"serial": "SCENE"}, "cam_15": {"serial": "WRIST"}})
    gateway._apply_recorder_output(state, "CAMERA_ROLES " + json.dumps(roles))
    assert state.teleop.wristCamera["camera"] == "cam_15"
    assert [view["deviceId"] for view in state.teleop.cameraViews] == ["cam_06", "cam_15"]
    wrist = next(d for d in state.devices if d["id"] == "cam_15")
    assert wrist["label"] == "FR3 wrist · cam_15"
    assert wrist["config"]["camera_role"]["serial"] == "WRIST"
    # A new connection must not inherit the preceding module's identity.
    gateway._reset_thor_camera_roles(state)
    assert state.teleop.wristCamera["state"] == "pending"
    with pytest.raises(RuntimeError):
        require_wrist_camera(state.teleop.wristCamera)


def test_visual_trajectory_refuses_static_extrinsics_for_moving_wrist(tmp_path):
    root = tmp_path / "repo"
    runner = root / gateway.DEFAULT_EE_TRAJECTORY_RUNNER
    tracker_config = root / gateway.DEFAULT_EE_TRAJECTORY_CONFIG
    runner.parent.mkdir(parents=True)
    tracker_config.parent.mkdir(parents=True)
    runner.write_text("#!/usr/bin/env bash\n")
    tracker_config.write_text("calibration:\n  root_dir: outputs/calibration\n  fixed_camera_run_name: fixed\n")
    summary = root / "outputs/calibration/fixed/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"joint_solution": {"cameras": {"cam_15": {}, "cam_06": {}}}}))
    dataset = root / "outputs/datasets/teleop"
    episode = dataset / "episodes/episode_000000"
    episode.mkdir(parents=True)
    (episode / "cam_06.mkv").touch()
    (episode / "cam_15.mkv").touch()
    roles = resolve_camera_roles(config(sensor_id=15), cameras(6, 15), {})
    (episode / "meta.json").write_text(json.dumps({"camera_roles": roles}))
    state = gateway.GatewayState(repo_root=root, config_path=root / "config.yaml", config={},
                                 recording=gateway.RecordingStatus(), replay=gateway.ReplayStatus())
    assert gateway._marker_tcp_tracking_cameras(state, dataset, [0]) == ["cam_06"]
    with pytest.raises(ValueError, match="fixed base extrinsics"):
        gateway._ee_trajectory_command(state, dataset)
    summary.write_text(json.dumps({"joint_solution": {"cameras": {"cam_06": {}}}}))
    command = gateway._ee_trajectory_command(state, dataset)
    assert command[command.index("--dataset-root") + 1] == str(dataset)
