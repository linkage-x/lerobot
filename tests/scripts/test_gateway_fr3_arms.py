"""The gateway drives the FR3 a request names, never a default one (roadmap B2)."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from tools.data_collection_gui import gateway


def _state(tmp_path: Path, profile: str = "thor") -> gateway.GatewayState:
    dataset_root = tmp_path / "session"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={"replay": {"real_preflight_enabled": True}},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
        profile=profile,
    )
    state.selected_replay_root = dataset_root
    return state


def test_the_arms_are_listed_and_required_on_thor_only(tmp_path: Path) -> None:
    state = _state(tmp_path)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/robot/arms", timeout=5) as r:
            payload = json.loads(r.read())
    finally:
        server.shutdown()
        server.server_close()
    assert payload["ok"] and payload["required"]
    right, left = payload["arms"]
    # 10-26 (B1): the registered arm is the left one; the new one on its right has a bare flange
    assert right == {**right, "arm": "right", "inService": True, "robotIp": "192.168.1.210", "gripperBox": None}
    assert left == {**left, "arm": "left", "inService": True, "robotIp": "192.168.11.102", "gripperBox": 596523097}
    assert not gateway._robot_arms_payload(_state(tmp_path / "ws", "workstation"))["required"]


def test_a_preflight_names_its_arm_and_both_checks_every_arm_in_service(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    with pytest.raises(ValueError, match="there is no default arm"):
        gateway._preflight_replay(state)
    checked = []
    monkeypatch.setattr(gateway, "_run_real_preflight", lambda _s, targets: checked.extend(t["arm"] for t in targets))
    gateway._preflight_replay(state, "both")
    assert checked == ["right", "left"]
    gateway._preflight_replay(state, "", "192.168.11.102")
    assert checked[-1] == "left" and state.replay.realArm == "left"


def test_the_preflight_runs_against_the_arm_s_own_config(tmp_path: Path) -> None:
    command = gateway._real_preflight_command(_state(tmp_path), "192.168.11.103", "some/left.thor.yaml")
    assert f"--config-path={Path.cwd() / 'some/left.thor.yaml'}" in command
    assert "--robot-ip=192.168.11.103" in command


def test_single_arm_replay_is_only_for_the_arm_whose_base_the_dataset_is_in(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(gateway, "_require_mujoco_validation", lambda *a, **k: state.selected_replay_root)
    monkeypatch.setattr(gateway, "_read_sidecar_cube_poses", lambda *a: {"left": [object()], "right": [object()]})
    monkeypatch.setattr(gateway, "_trajectory_tcp_link", lambda *a: "")
    state.replay.mujocoValidation = {"cubeMode": "right"}
    with pytest.raises(ValueError, match="right arm commands in fr3_base_right, the dataset is in fr3_base"):
        gateway._start_real_replay(state, "right", arm="right")


def test_every_arm_s_gripper_box_is_the_arm_s_not_the_operator_s(tmp_path: Path) -> None:
    assert "box596523097" in gateway._fr3_arm_box_ids(Path.cwd())
