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
    # 10-26 (B1): the registered arm is the left one; 11-02 the gripper moved to the right one
    assert right == {**right, "arm": "right", "inService": True, "robotIp": "192.168.1.210", "gripperBox": 3100715645}
    assert left == {**left, "arm": "left", "inService": True, "robotIp": "192.168.11.102", "gripperBox": None}
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


def test_a_recording_from_before_world_stamps_replays_only_on_the_arm_in_fr3_base(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(gateway, "_require_mujoco_validation", lambda *a, **k: state.selected_replay_root)
    monkeypatch.setattr(gateway, "_read_sidecar_cube_poses", lambda *a: {"left": [object()], "right": [object()]})
    monkeypatch.setattr(gateway, "_trajectory_tcp_link", lambda *a: "")
    state.replay.mujocoValidation = {"cubeMode": "right"}
    # no world stamps: recorded in the robot base itself, fr3_base (the left arm's), and in no other
    with pytest.raises(RuntimeError, match="predates world stamps: its poses are in fr3_base as recorded, and no edge takes them to fr3_base_right"):
        gateway._start_real_replay(state, "right", arm="right")


def test_every_arm_s_gripper_box_is_the_arm_s_not_the_operator_s(tmp_path: Path) -> None:
    assert "box3100715645" in gateway._fr3_arm_box_ids(Path.cwd())


# --- Device Manager gripper close/open button ---------------------------------


class _AnsweringStdin:
    """The recorder's stdin: answers each arm_gripper line the way thor_record does, on its own thread.

    The answer is applied as the recorder-output consumer applies it -- under ``state.lock`` -- so the
    POST, which holds that lock, gets it only if its wait really releases the lock.
    """

    def __init__(self, state: gateway.GatewayState, *, ok: bool = True, answer: bool = True) -> None:
        self.state, self.ok, self.answer, self.writes = state, ok, answer, []

    def write(self, text: str) -> None:
        self.writes.append(text)
        if not self.answer:
            return
        _, device_id, width, request = text.strip().split(":")
        reply = {
            "request": request,
            "deviceId": int(device_id),
            "targetM": float(width),
            "ok": self.ok,
            "error": "" if self.ok else "set_mode(1) 失败 rc=-1",
            "modeBefore": 1,
            "widthBeforeM": 0.0887,
        }

        def apply() -> None:
            with self.state.lock:
                gateway._apply_recorder_output(self.state, "ARMGRIPPER_DONE " + json.dumps(reply))
                self.state.recorder_output_applied.notify_all()

        threading.Thread(target=apply, daemon=True).start()

    def flush(self) -> None:
        pass


class _Recorder:
    pid = 4321

    def __init__(self, stdin: _AnsweringStdin) -> None:
        self.stdin = stdin

    def poll(self):
        return None


def _gripper_state(tmp_path: Path, monkeypatch, **stdin_kwargs) -> tuple[gateway.GatewayState, _AnsweringStdin]:
    state = _state(tmp_path)
    stdin = _AnsweringStdin(state, **stdin_kwargs)
    state.process = _Recorder(stdin)
    monkeypatch.setattr(gateway, "_state_is_gmsl2", lambda _state: True)
    monkeypatch.setattr(gateway, "_gripper_teleop_root", lambda _state: tmp_path / "teleop")
    return state, stdin


def _post(server, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as answered:
        return json.loads(answered.read())


def test_the_gripper_button_closes_then_opens_through_the_recorder(tmp_path: Path, monkeypatch) -> None:
    state, stdin = _gripper_state(tmp_path, monkeypatch)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        closed = _post(server, "/api/fr3/arm-gripper", {"arm": "right", "action": "close"})
        hold_after_close = json.loads((tmp_path / "teleop" / "hold.json").read_text())
        opened = _post(server, "/api/fr3/arm-gripper", {"arm": "right", "action": "open"})
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/fr3/arm-gripper", timeout=5) as r:
            listed = json.loads(r.read())
    finally:
        server.shutdown()
        server.server_close()
    # close = the right arm's replay grasp width (0 m), open = just short of the jaw's 88.7 mm
    assert stdin.writes == ["arm_gripper:3100715645:0.0000:g1\n", "arm_gripper:3100715645:0.0880:g2\n"]
    assert closed["ok"] and closed["error"] == ""
    assert hold_after_close == {**hold_after_close, "device_id": 3100715645, "target_m": 0.0, "mode": 1}
    assert opened["ok"]
    assert json.loads((tmp_path / "teleop" / "hold.json").read_text())["target_m"] == 0.088
    (right,) = listed["arms"]  # the left arm has no gripper BOX wired in
    assert right["arm"] == "right" and right["deviceId"] == 3100715645
    assert right["last"]["action"] == "open" and right["last"]["ok"] is True
    assert listed["refusal"] == ""
    assert state.lock.acquire(timeout=1)
    state.lock.release()


def test_the_gripper_button_reports_what_the_recorder_answered(tmp_path: Path, monkeypatch) -> None:
    state, _ = _gripper_state(tmp_path, monkeypatch, ok=False)
    with state.lock:
        result = gateway._arm_gripper_command(state, {"arm": "right", "action": "close"})
    assert not result["ok"] and "set_mode(1)" in result["error"]
    assert not (tmp_path / "teleop" / "hold.json").exists()  # a failed command is no new hold target


def test_a_recorder_that_never_answers_is_reported_not_waited_on(tmp_path: Path, monkeypatch) -> None:
    state, stdin = _gripper_state(tmp_path, monkeypatch, answer=False)
    monkeypatch.setattr(gateway, "_ARM_GRIPPER_REPLY_TIMEOUT_S", 0.2)
    with state.lock:
        result = gateway._arm_gripper_command(state, {"arm": "right", "action": "open"})
    assert len(stdin.writes) == 1
    assert not result["ok"] and "没有回话" in result["error"]
    assert result["arms"][0]["last"]["ok"] is False


def test_the_gripper_button_refuses_without_a_recorder_a_gripper_or_a_clear_action(tmp_path: Path, monkeypatch) -> None:
    state, stdin = _gripper_state(tmp_path, monkeypatch)
    with state.lock:
        toggled = gateway._arm_gripper_command(state, {"arm": "right", "action": "toggle"})
        unnamed = gateway._arm_gripper_command(state, {"action": "close"})
        bare = gateway._arm_gripper_command(state, {"arm": "left", "action": "close"})
        state.process = None
        offline = gateway._arm_gripper_command(state, {"arm": "right", "action": "close"})
    assert "close 或 open" in toggled["error"]
    assert "no default arm" in unnamed["error"]
    assert "没有接入的夹爪 BOX" in bare["error"]
    assert "先 Connect" in offline["error"] and "先 Connect" in offline["refusal"]
    assert stdin.writes == []


def test_the_live_width_is_the_arm_box_s_and_only_while_fresh(tmp_path: Path) -> None:
    import time

    state = _state(tmp_path)
    jaw = {"distance_m": 0.0476}
    state.device_preview["box"] = {"active": True, "updatedAt": time.time(), "sensors": {"box3100715645/box_gripper": jaw, "box3906437858/box_gripper": {"distance_m": 0.08}}}
    assert gateway._arm_gripper_live_width_mm(state, 3100715645) == 47.6
    state.device_preview["box"] = {"active": True, "updatedAt": time.time(), "sensors": {"box_gripper": jaw}, "status": {"device_id": 3906437858}}
    assert gateway._arm_gripper_live_width_mm(state, 3100715645) is None  # one BOX, but the handheld
    state.device_preview["box"]["status"] = {"device_id": 3100715645}
    assert gateway._arm_gripper_live_width_mm(state, 3100715645) == 47.6
    state.device_preview["box"]["updatedAt"] = time.time() - 10
    assert gateway._arm_gripper_live_width_mm(state, 3100715645) is None
