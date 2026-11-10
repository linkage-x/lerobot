"""Log one BOX's 6D force + IMU while it is turned through the six faces by hand (free air).

Writes ``free_air.log`` (rows in ``box_six_d.py fit`` format) and ``status.json`` (rewritten twice a
second: how many quasi-static samples each face has so far) into ``--out``, until ``--duration-s``
or SIGTERM. The calibration page runs it and shows the counts; ``box_six_d.py fit`` then fits K.

It opens its own BOX client on :15000, like fr3_gripper_teleop.py, so the recorder must be
disconnected. Nothing is sent to the BOX (its mode is kept).

Counting starts at the pick-up (the first turn faster than 30 deg/s): whatever the BOX rests on
before that is not free air. A sample counts when it is quasi-static by the fit's own test
(box_six_d.static_mask) and one axis is clearly up (|acc| on it > 0.85 g).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.thor.gmsl2 import box_six_d  # noqa: E402

FACES = ("x+", "x-", "y+", "y-", "z+", "z-")
PICKUP_GYR_DEG_S = 30.0
FACE_AXIS_G = 0.85
DEFAULT_NEED = 30


def face_of(acc: list[float], gyr: list[float]) -> str | None:
    """The face a sample counts for, or None (moving, or between faces)."""
    norm = math.sqrt(sum(a * a for a in acc))
    spin = math.sqrt(sum(g * g for g in gyr))
    if abs(norm - 1.0) >= box_six_d.STATIC_ACC_TOL_G or spin >= box_six_d.STATIC_GYR_DEG_S:
        return None
    i = max(range(3), key=lambda k: abs(acc[k]))
    if abs(acc[i]) <= FACE_AXIS_G:
        return None
    return "xyz"[i] + ("+" if acc[i] > 0 else "-")


class Counter:
    def __init__(self, need: int = DEFAULT_NEED) -> None:
        self.need = need
        self.faces = dict.fromkeys(FACES, 0)
        self.picked_up = False
        self.rows = 0

    def add(self, acc: list[float], gyr: list[float]) -> None:
        self.rows += 1
        if not self.picked_up and math.sqrt(sum(g * g for g in gyr)) > PICKUP_GYR_DEG_S:
            self.picked_up = True
        if self.picked_up:
            face = face_of(acc, gyr)
            if face:
                self.faces[face] += 1

    @property
    def done(self) -> bool:
        return all(n >= self.need for n in self.faces.values())


def _open(device_id: int):
    from tools.thor.box_sdk import box_client as bc

    template = bc.BoxClientConfig(poll_interval_s=0.002, stale_threshold_s=0.5)
    pool = bc.BoxPool(bc.BoxFleetConfig(enabled=True, boxes=[template], keep_mode_device_ids=[device_id]))
    if not pool.start():
        raise RuntimeError("BOX 客户端没起来：box_sdk 不可用，或 :15000 被占用（录制器是否还连着？先 Disconnect）")
    ids = [int(d.device_id) for d in pool.discovered()]
    if device_id not in ids:
        pool.stop()
        raise RuntimeError(f"没发现 BOX {device_id}；在线的：{ids}")
    return pool, pool.client_for_device(device_id)


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    counter = Counter(args.need)
    status = {"state": "starting", "device_id": args.device_id, "need": args.need, "error": ""}

    def write_status(**kw) -> None:
        status.update(kw)
        status.update(
            {"faces": counter.faces, "rows": counter.rows, "picked_up": counter.picked_up, "done": counter.done,
             "updated_unix": round(time.time(), 3)}
        )
        tmp = out / "status.json.tmp"
        tmp.write_text(json.dumps(status, ensure_ascii=False))
        tmp.replace(out / "status.json")

    try:
        pool, client = _open(int(args.device_id))
    except RuntimeError as exc:
        write_status(state="error", error=str(exc))
        print(f"ERROR: {exc}", flush=True)
        return 2
    write_status(state="live", started_unix=round(time.time(), 3))
    print(f"BOX {args.device_id}: logging to {out / 'free_air.log'}", flush=True)
    last_ts, last_status = None, 0.0
    end = time.monotonic() + args.duration_s
    try:
        with (out / "free_air.log").open("a", buffering=1) as log:
            while not stop["flag"] and time.monotonic() < end:
                sensors = (client.read() or {}).get("sensors") or {}
                f, imu = sensors.get("box_six_d_force") or {}, sensors.get("box_imu") or {}
                if f.get("fxyz_mxyz") and imu.get("acc_g") and f.get("timestamp") != last_ts:
                    last_ts = f.get("timestamp")
                    acc, gyr = imu["acc_g"], imu.get("gyr_deg_s") or [0.0, 0.0, 0.0]
                    grip = sensors.get("box_gripper") or {}
                    log.write(
                        json.dumps(
                            {"t": round(time.time(), 4), "k": "box_six_d_force", "ts": last_ts, "f": f["fxyz_mxyz"],
                             "raw": f.get("fxyz_mxyz_raw"), "acc": acc, "gyr": gyr, "dist": grip.get("distance_m")}
                        )
                        + "\n"
                    )
                    counter.add(acc, gyr)
                    status["opening_mm"] = None if grip.get("distance_m") is None else round(grip["distance_m"] * 1000, 1)
                if time.monotonic() - last_status > 0.5:
                    last_status = time.monotonic()
                    write_status()
                time.sleep(0.005)
    finally:
        pool.stop()
    write_status(state="stopped" if stop["flag"] else "timeout", stopped_unix=round(time.time(), 3))
    print(f"stopped: {counter.rows} rows, faces {counter.faces}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--device-id", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=300.0)
    p.add_argument("--need", type=int, default=DEFAULT_NEED, help="quasi-static samples per face to call it done")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
