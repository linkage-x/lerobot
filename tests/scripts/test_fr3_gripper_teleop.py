"""fr3_gripper_teleop: the FR3's gripper BOX follows the handheld BOX, and keeps its grip on exit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.fr3 import fr3_gripper_teleop as gt


class FakeBox:
    def __init__(self, width_m: float, *, mode: int = 0, trigger_pct: float | None = None):
        self.width_m = width_m
        self.trigger_pct = trigger_pct
        self.mode = mode
        self.ts = 1_000_000
        self.clamp: list[float] = []
        self.modes: list[int] = []

    def read(self):
        self.ts += 5_000
        sensors = {"box_gripper": {"timestamp": self.ts, "distance_m": self.width_m}}
        if self.trigger_pct is not None:
            sensors["box_trigger"] = {"timestamp": self.ts, "travel_pct": self.trigger_pct}
        return {"sensors": sensors}

    def get_mode(self, timeout_ms: int = 100):
        return 0, self.mode

    def set_mode(self, mode: int) -> int:
        self.mode = mode
        self.modes.append(mode)
        return 0

    def set_clamp_pos(self, pos_m: float) -> int:
        self.clamp.append(pos_m)
        return 0


def _teleop(arm: FakeBox, hand: FakeBox) -> gt.GripperTeleop:
    return gt.GripperTeleop(arm, hand, gt.TeleopConfig(command_rate_hz=1000.0))


def test_it_takes_control_where_the_arm_is_and_follows_only_after_the_handheld_matches():
    arm, hand = FakeBox(0.060), FakeBox(0.090, trigger_pct=0.0)
    t = _teleop(arm, hand)
    assert "保持当前开度 60.0 mm" in t.take_control(None, 0.0)
    assert arm.modes == [1] and arm.clamp == [0.060, 0.060]

    # An open handheld must not open the arm: soft takeover.
    assert t.step(1.0)["state"] == "waiting"
    assert arm.clamp == [0.060, 0.060]

    hand.width_m = 0.058
    snap = t.step(2.0)
    assert snap["state"] == "following" and arm.clamp[-1] == pytest.approx(0.058)

    hand.width_m = 0.040
    t.step(3.0)
    assert arm.clamp[-1] == pytest.approx(0.040)
    hand.width_m = 0.0402  # inside the deadband: nothing sent
    n = len(arm.clamp)
    t.step(4.0)
    assert len(arm.clamp) == n


def test_a_pressed_trigger_shuts_the_arm_so_it_grips_what_the_handheld_could_not_close_on():
    arm, hand = FakeBox(0.036), FakeBox(0.036, trigger_pct=20.0)
    t = _teleop(arm, hand)
    t.take_control(None, 0.0)
    t.step(1.0)
    assert t.state == "following"
    hand.trigger_pct = 96.0  # jaw stopped on the insert, trigger pressed home
    snap = t.step(2.0)
    assert snap["commandMm"] == 0.0 and arm.clamp[-1] == 0.0


def test_re_entering_keeps_the_squeeze_instead_of_re_sending_the_measured_width():
    # Last exit left mode 1 with target 0 on an insert 36 mm wide.
    arm, hand = FakeBox(0.036, mode=1), FakeBox(0.090, trigger_pct=0.0)
    t = _teleop(arm, hand)
    assert "沿用上次的目标 0.0 mm" in t.take_control(0.0, 0.0)
    assert arm.clamp == [] and arm.modes == []
    t.step(1.0)
    assert t.state == "waiting" and arm.clamp == []  # open handheld: the insert stays held
    hand.trigger_pct = 95.0
    t.step(2.0)
    assert t.state == "following"


def test_a_saved_target_is_ignored_once_the_box_is_back_in_mode_0():
    arm, hand = FakeBox(0.090, mode=0), FakeBox(0.090)
    t = _teleop(arm, hand)
    assert "之前是模式 0" in t.take_control(0.0, 0.0)
    assert t.target_m == pytest.approx(0.090)


def test_a_stale_handheld_holds_and_a_rebooted_arm_box_is_taken_back():
    arm, hand = FakeBox(0.050), FakeBox(0.050)
    t = _teleop(arm, hand)
    t.take_control(None, 0.0)
    t.step(1.0)
    hand.width_m = None  # no jaw reading
    n = len(arm.clamp)
    assert t.step(2.0)["handheldFresh"] is False and len(arm.clamp) == n

    arm.mode, arm.ts = 0, 0  # boots in mode 0, clock from zero
    t.step(3.0)
    assert arm.mode == 1 and arm.clamp[-1] == pytest.approx(0.050)
    assert any("重启" in e for e in t.events)


def test_handheld_pick_and_hold_file(tmp_path):
    dev = lambda i, caps=("box_gripper",): SimpleNamespace(device_id=i, expected_devices=list(caps))  # noqa: E731
    assert gt.pick_handheld([dev(5), dev(9)], 9, None) == 5
    with pytest.raises(RuntimeError, match="恰好一台"):
        gt.pick_handheld([dev(5), dev(6), dev(9)], 9, None)
    assert gt.pick_handheld([dev(5), dev(6), dev(9)], 9, 6) == 6

    hold = tmp_path / "hold.json"
    gt.write_hold(hold, 9, 0.0, 1)
    assert json.loads(hold.read_text())["mode"] == 1
    assert gt.read_hold(hold, 9) == 0.0
    assert gt.read_hold(hold, 5) is None
    assert gt.read_hold(tmp_path / "missing.json", 9) is None


def test_it_never_puts_the_arm_box_back_in_mode_0_on_exit(monkeypatch, tmp_path, capsys):
    arm, hand = FakeBox(0.036), FakeBox(0.036, trigger_pct=20.0)
    stopped = []

    class Pool:
        def discovered(self):
            return [SimpleNamespace(device_id=9, expected_devices=["box_gripper"]),
                    SimpleNamespace(device_id=5, expected_devices=["box_gripper"])]

        def client_for_device(self, i):
            return {9: arm, 5: hand}[i]

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(gt, "open_pool", lambda arm_id: Pool())
    ticks = iter(range(30))

    def step(self, now=None, _orig=gt.GripperTeleop.step):
        tick = next(ticks)
        if tick == 10:  # matched, following: now press the trigger home on the insert
            hand.trigger_pct = 99.0
        if tick == 20:
            gt.signal.raise_signal(gt.signal.SIGINT)
        return _orig(self, now)

    monkeypatch.setattr(gt.GripperTeleop, "step", step)
    hold = tmp_path / "hold.json"
    handlers = {sig: gt.signal.getsignal(sig) for sig in (gt.signal.SIGINT, gt.signal.SIGTERM)}
    try:
        assert gt.main(["--arm-device-id", "9", "--hold-file", str(hold), "--loop-hz", "1000", "--status-hz", "1000"]) == 0
    finally:
        for sig, handler in handlers.items():
            gt.signal.signal(sig, handler)

    assert stopped and 0 not in arm.modes and arm.mode == 1
    assert arm.clamp[-1] == 0.0  # the pressed trigger squeezed it shut
    out = capsys.readouterr().out
    assert "不卸力" in out
    last = [json.loads(line[7:]) for line in out.splitlines() if line.startswith("STATUS ")][-1]
    assert last["state"] == "exited" and last["armMode"] == 1 and last["targetMm"] == 0.0
    assert json.loads(hold.read_text())["target_m"] == 0.0
