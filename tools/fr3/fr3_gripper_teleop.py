#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Drive the FR3's gripper from the handheld BOX: the arm's jaw follows the handheld's jaw.

Both grippers are BOXes on the same UDP network (one shared socket on :15000), so this
never touches the FR3's FCI: teaching mode, a pivot collect, or nothing at all may be
running on the arm meanwhile. The command is the replay's rule -- the handheld's jaw
opening in metres, or shut (``--grasp-closed-width-m``) once its trigger is pressed past
``--grasp-trigger-close-pct``, so a squeeze on an object the handheld cannot close on
still grips with the arm's jaw.

Entering:
  - The arm's BOX is put in control mode (1) holding where it is -- or, if it already is
    in control mode and this script left it so (``--hold-file``), holding that target.
    The pool never set_mode()s it otherwise (``keep_mode_device_ids``).
  - It only starts following once the handheld's command comes within
    ``--engage-tolerance-m`` of what the arm holds (a soft takeover): an open handheld
    must not open a jaw that is holding something.

Exiting (SIGINT/SIGTERM): the arm's BOX stays in control mode at the last target, so a
grip keeps its force. Mode 0 would hand the jaw back to the arm BOX's own trigger and
open it. The target is written to ``--hold-file`` for the next entry.

Status goes to stdout as ``STATUS {json}`` lines (the GUI's panel reads them).
On Thor, from ~/lerobot:  PYTHONPATH=src ~/Code/infer/.venv-fr3/bin/python \
  tools/fr3/fr3_gripper_teleop.py --arm-device-id 596523097
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REPO = Path(__file__).resolve().parents[2]

CONTROL_MODE = 1


class BoxView(Protocol):
    def read(self) -> dict[str, Any]: ...

    def get_mode(self, timeout_ms: int = 100) -> tuple[int, int]: ...

    def set_mode(self, mode: int) -> int: ...

    def set_clamp_pos(self, pos_m: float) -> int: ...


def _sensor(sample: dict[str, Any], sid: str) -> dict[str, Any]:
    data = (sample.get("sensors") or {}).get(sid)
    return data if isinstance(data, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


@dataclass
class TeleopConfig:
    grasp_trigger_close_pct: float | None = 90.0
    grasp_closed_width_m: float = 0.0
    max_width_m: float = 0.09
    engage_tolerance_m: float = 0.005
    command_rate_hz: float = 20.0
    deadband_m: float = 0.0005
    handheld_stale_s: float = 0.3


@dataclass
class GripperTeleop:
    """The follow loop, one ``step`` per tick; owns no sockets so it can be driven by fakes."""

    arm: BoxView
    handheld: BoxView
    cfg: TeleopConfig = field(default_factory=TeleopConfig)
    state: str = "init"  # init -> waiting (soft takeover) -> following
    target_m: float | None = None  # what the arm's BOX was last told to hold
    arm_reboots: int = 0
    _last_send_s: float = -1e9
    _arm_ts: int | None = None
    _hand_ts: int | None = None
    _hand_seen_s: float = -1e9
    events: list[str] = field(default_factory=list)

    def handheld_command(self) -> tuple[float | None, float | None, float | None, bool]:
        """(command m, jaw m, trigger %, fresh) from the handheld's latest sample."""
        sample = self.handheld.read()
        jaw = _sensor(sample, "box_gripper")
        width = _finite(jaw.get("distance_m"))
        trigger = _finite(_sensor(sample, "box_trigger").get("travel_pct"))
        ts = jaw.get("timestamp")
        now = time.monotonic()
        if ts is not None and ts != self._hand_ts:
            self._hand_ts = ts
            self._hand_seen_s = now
        fresh = width is not None and now - self._hand_seen_s <= self.cfg.handheld_stale_s
        if width is None:
            return None, None, trigger, False
        command = min(max(width, 0.0), self.cfg.max_width_m)
        if self.cfg.grasp_trigger_close_pct is not None and trigger is not None and trigger >= self.cfg.grasp_trigger_close_pct:
            command = self.cfg.grasp_closed_width_m
        return command, width, trigger, fresh

    def arm_width(self) -> float | None:
        jaw = _sensor(self.arm.read(), "box_gripper")
        return _finite(jaw.get("distance_m"))

    def _send(self, target_m: float, now: float) -> None:
        rc = self.arm.set_clamp_pos(target_m)
        if rc != 0:
            raise RuntimeError(f"FR3 gripper set_clamp_pos({target_m:.4f}) failed rc={rc}")
        self.target_m = target_m
        self._last_send_s = now

    def take_control(self, kept_target_m: float | None, now: float | None = None) -> str:
        """Control mode, holding what the arm holds. Returns how the hold target was chosen."""
        now = time.monotonic() if now is None else now
        rc, mode = 1, -1
        for _ in range(3):
            rc, mode = self.arm.get_mode()
            if rc == 0:
                break
        if rc == 0 and mode == CONTROL_MODE and kept_target_m is not None:
            # Still holding what the last exit left: re-sending the measured width would
            # turn a squeeze on an object into a mere touch.
            self.target_m = kept_target_m
            self.state = "waiting"
            return f"FR3 夹爪仍在控制模式，沿用上次的目标 {kept_target_m * 1e3:.1f} mm"
        width = self.arm_width()
        if width is None:
            raise RuntimeError("FR3 夹爪 BOX 没有开度读数（box_gripper.distance_m）")
        # Preload, then control mode, then hold: firmware that latches the target early
        # must not see a stale or zero one (same order as CoreneticGripperHardwareDriver).
        self.arm.set_clamp_pos(width)
        rc_mode = self.arm.set_mode(CONTROL_MODE)
        if rc_mode != 0:
            raise RuntimeError(f"FR3 夹爪 set_mode(1) 失败 rc={rc_mode}")
        self._send(width, now)
        self.state = "waiting"
        why = "读不到模式" if rc != 0 else f"之前是模式 {mode}"
        return f"FR3 夹爪进入控制模式（{why}），保持当前开度 {width * 1e3:.1f} mm"

    def _check_arm_reboot(self, now: float) -> None:
        """A restarted BOX boots in mode 0 (jaw follows its own trigger): take it back."""
        ts = _sensor(self.arm.read(), "box_gripper").get("timestamp")
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            return
        previous, self._arm_ts = self._arm_ts, ts
        if previous is None or ts >= previous - 1_000_000 or self.target_m is None:
            return
        self.arm_reboots += 1
        self.events.append(f"FR3 夹爪 BOX 重启过（第 {self.arm_reboots} 次），已重新进入控制模式并发送 {self.target_m * 1e3:.1f} mm")
        if self.arm.set_mode(CONTROL_MODE) != 0:
            raise RuntimeError("FR3 夹爪 BOX 重启后 set_mode(1) 失败")
        self._send(self.target_m, now)

    def flush(self) -> None:
        """Send the handheld's latest command now, past the rate limit: the one exit must not drop."""
        if self.state != "following":
            return
        command, _, _, fresh = self.handheld_command()
        if fresh and command is not None and (self.target_m is None or abs(command - self.target_m) >= self.cfg.deadband_m):
            self._send(command, time.monotonic())

    def step(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        self._check_arm_reboot(now)
        command, width, trigger, fresh = self.handheld_command()
        if (
            self.state == "waiting"
            and fresh
            and command is not None
            and self.target_m is not None
            and abs(command - self.target_m) <= self.cfg.engage_tolerance_m
        ):
            self.state = "following"
            self.events.append("手持夹爪与 FR3 夹爪对上了，开始跟随")
        if (
            self.state == "following"
            and fresh
            and command is not None
            and (self.target_m is None or abs(command - self.target_m) >= self.cfg.deadband_m)
            and now - self._last_send_s >= 1.0 / self.cfg.command_rate_hz
        ):
            self._send(command, now)
        return {
            "state": self.state,
            "handheldFresh": fresh,
            "handheldWidthMm": None if width is None else round(width * 1e3, 1),
            "triggerPct": None if trigger is None else round(trigger, 1),
            "commandMm": None if command is None else round(command * 1e3, 1),
            "targetMm": None if self.target_m is None else round(self.target_m * 1e3, 1),
            "armWidthMm": None if (w := self.arm_width()) is None else round(w * 1e3, 1),
            "armReboots": self.arm_reboots,
        }


def read_hold(path: Path, arm_device_id: int) -> float | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if int(raw.get("device_id", -1)) != arm_device_id:
        return None
    return _finite(raw.get("target_m"))


def write_hold(path: Path, arm_device_id: int, target_m: float | None, mode: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "device_id": arm_device_id,
                "target_m": target_m,
                "mode": mode,
                "written": dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )


def status(payload: dict[str, Any]) -> None:
    print("STATUS " + json.dumps(payload, ensure_ascii=False), flush=True)


def say(text: str) -> None:
    print(text, flush=True)


def _await_sample(view: BoxView, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _finite(_sensor(view.read(), "box_gripper").get("distance_m")) is not None:
            return True
        time.sleep(0.02)
    return False


def open_pool(arm_device_id: int):
    """One shared Box over every BOX on the network; the arm's mode left alone."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.thor.box_sdk import box_client as bc

    template = bc.BoxClientConfig(poll_interval_s=0.005, stale_threshold_s=0.5)
    pool = bc.BoxPool(bc.BoxFleetConfig(enabled=True, boxes=[template], keep_mode_device_ids=[arm_device_id]))
    if not pool.start():
        raise RuntimeError("BOX 客户端没起来：box_sdk 不可用，或 :15000 被占用（录制器 / 真机 replay 是否连着？）")
    return pool


def pick_handheld(devices: list[Any], arm_device_id: int, wanted: int | None) -> int:
    ids = [int(d.device_id) for d in devices]
    if wanted is not None:
        if wanted not in ids:
            raise RuntimeError(f"没发现手持 BOX {wanted}；在线的：{ids}")
        return wanted
    others = [int(d.device_id) for d in devices if int(d.device_id) != arm_device_id and "box_gripper" in d.expected_devices]
    if len(others) != 1:
        raise RuntimeError(f"手持 BOX 要恰好一台（除 FR3 的 {arm_device_id} 外带夹爪的有 {others}）；用 --handheld-device-id 指定")
    return others[0]


def run(args: argparse.Namespace) -> int:
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    cfg = TeleopConfig(
        grasp_trigger_close_pct=args.grasp_trigger_close_pct if args.grasp_trigger_close_pct > 0 else None,
        grasp_closed_width_m=args.grasp_closed_width_m,
        max_width_m=args.max_width_m,
        engage_tolerance_m=args.engage_tolerance_mm / 1e3,
        command_rate_hz=args.rate_hz,
    )
    hold_file = Path(args.hold_file)
    pool = open_pool(args.arm_device_id)
    teleop: GripperTeleop | None = None
    try:
        devices = pool.discovered()
        if args.arm_device_id not in [int(d.device_id) for d in devices]:
            raise RuntimeError(f"没发现 FR3 夹爪 BOX {args.arm_device_id}；在线的：{[int(d.device_id) for d in devices]}")
        hand_id = pick_handheld(devices, args.arm_device_id, args.handheld_device_id)
        arm, hand = pool.client_for_device(args.arm_device_id), pool.client_for_device(hand_id)
        if arm is None or hand is None:
            raise RuntimeError("BOX 视图没建起来")
        for name, view in (("FR3 夹爪", arm), ("手持", hand)):
            if not _await_sample(view, 3.0):
                raise RuntimeError(f"{name} BOX 3 s 内没有开度数据")
        say(f"FR3 夹爪 BOX {args.arm_device_id}，手持 BOX {hand_id}")
        teleop = GripperTeleop(arm, hand, cfg)
        say(teleop.take_control(read_hold(hold_file, args.arm_device_id)))
        write_hold(hold_file, args.arm_device_id, teleop.target_m, CONTROL_MODE)
        say(f"把手持夹爪调到与 FR3 夹爪相差 {args.engage_tolerance_mm:g} mm 以内后开始跟随")
        period = 1.0 / args.loop_hz
        last_status = 0.0
        parent = os.getppid()
        while not stop["flag"]:
            if os.getppid() != parent:  # the gateway went away: nobody can press Exit any more
                say("父进程退出了，按退出遥控处理")
                break
            t0 = time.monotonic()
            snap = teleop.step(t0)
            while teleop.events:
                say(teleop.events.pop(0))
            if t0 - last_status >= 1.0 / args.status_hz:
                status(snap)
                last_status = t0
                write_hold(hold_file, args.arm_device_id, teleop.target_m, CONTROL_MODE)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        mode = None
        if teleop is not None and teleop.target_m is not None:
            # Never set_mode(0): that hands the jaw to the arm BOX's own trigger and opens it.
            arm = pool.client_for_device(args.arm_device_id)
            with contextlib.suppress(Exception):
                teleop.flush()
            with contextlib.suppress(Exception):
                rc, m = arm.get_mode()
                mode = m if rc == 0 else None
            write_hold(hold_file, args.arm_device_id, teleop.target_m, mode)
            say(f"退出遥控：FR3 夹爪保持 {teleop.target_m * 1e3:.1f} mm（模式 {mode if mode is not None else '读不到'}），不卸力")
            width = teleop.arm_width()
            status(
                {
                    "state": "exited",
                    "targetMm": round(teleop.target_m * 1e3, 1),
                    "armWidthMm": None if width is None else round(width * 1e3, 1),
                    "armMode": mode,
                    "armReboots": teleop.arm_reboots,
                }
            )
        pool.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-device-id", type=int, required=True, help="device_id of the BOX on the FR3")
    p.add_argument("--handheld-device-id", type=int, default=None, help="default: the one other BOX with a gripper")
    p.add_argument("--grasp-trigger-close-pct", type=float, default=90.0, help="<= 0 disables")
    p.add_argument("--grasp-closed-width-m", type=float, default=0.0)
    p.add_argument("--max-width-m", type=float, default=0.09)
    p.add_argument("--engage-tolerance-mm", type=float, default=5.0)
    p.add_argument("--rate-hz", type=float, default=20.0, help="max clamp command rate")
    p.add_argument("--loop-hz", type=float, default=100.0)
    p.add_argument("--status-hz", type=float, default=5.0)
    p.add_argument("--hold-file", default=str(REPO / "outputs" / "fr3_gripper_teleop" / "hold.json"))
    args = p.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        say(f"ERROR {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
