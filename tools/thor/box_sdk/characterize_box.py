#!/usr/bin/env python3
"""Read-only BOX collection-board bring-up smoke test.

Exercises the *real* recording path
(:meth:`BoxClient.start_recording` -> 500 Hz MCU-timestamp dedup ->
:meth:`BoxClient.stop_recording`) and characterizes every sensor that checks
in, so a freshly-wired handheld device can be validated in ~10 s without
launching the camera rig.

It NEVER commands gripper motion -- the only command sent is the benign
``set_mode(startup_mode)`` that ``BoxClient.start`` already issues for normal
collection. For each sensor it reports observed rate, MCU-timestamp
monotonicity, and whether the L3b MCU->host clock calibration
(:func:`thor_lerobot_v3.calibrate_mcu_clock`, the same code the writer uses)
would ENGAGE or fall back -- see ``tools/thor/ts_sync.md`` §5.2/§5.3.

Preconditions (see ts_sync.md / DEPLOYMENT.md):
  * host owns ``192.168.2.45`` (the box streams telemetry there) -- otherwise
    ``start()`` succeeds but no samples arrive;
  * box reachable at ``192.168.2.60:15000``.

Usage (from repo root, with the box_sdk wheel installed)::

    python tools/thor/box_sdk/characterize_box.py [duration_s]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a plain script (python tools/thor/box_sdk/characterize_box.py)
# as well as a module: resolve the repo root from this file's location.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.box_sdk import box_client as bc  # noqa: E402
from tools.thor.gmsl2 import thor_lerobot_v3 as lr3  # noqa: E402

# calibrate_sensor_samples fallback thresholds (ts_sync.md §5.3).
_MIN_SAMPLES = 10
_MAX_RES_STD_S = 0.05


def _fmt_row(sid: str, n: int, hz, mono, nonzero, engage, slope, res_ms) -> str:
    return (f"{sid:<16}{n:>6}{hz:>9}{str(mono):>14}{str(nonzero):>11}"
            f"{str(engage):>9}{slope:>14}{res_ms:>9}")


def main() -> int:
    duration_s = float(sys.argv[1]) if len(sys.argv) >= 2 else 10.0

    if not bc.available():
        print("FATAL: box_sdk wheel not importable on this host "
              "(install tools/thor/box_sdk/python/*.whl on the Jetson).")
        return 2

    cfg = bc.BoxClientConfig()  # defaults: 0.0.0.0/.2.60 :15000, startup_mode=0
    client = bc.BoxClient(cfg)
    if not client.start():
        print("FATAL: BoxClient.start() returned False -- box unreachable or "
              "SDK init failed. Check the host owns 192.168.2.45 and the box "
              "is powered at 192.168.2.60.")
        return 2

    print(f"started; warming up 1s, then recording {duration_s:.0f}s at "
          f"{cfg.record_poll_interval_s*1000:.0f}ms poll (read-only)...")
    time.sleep(1.0)
    t0 = time.time()
    client.start_recording(t0)
    time.sleep(duration_s)
    samples = client.stop_recording()
    dur = time.time() - t0
    client.stop()

    print(f"\n=== characterization over {dur:.2f}s ===")
    print(_fmt_row("sensor", "n", "Hz", "ts_mono", "ts_nonzero",
                   "calib?", "slope(s/tick)", "res_ms"))
    any_engaged = False
    seen = 0
    for sid in bc.KNOWN_SENSOR_IDS:
        slist = samples.get(sid, [])
        n = len(slist)
        if n == 0:
            print(_fmt_row(sid, 0, "-", "-", "-", "-", "-", "-"))
            continue
        seen += 1
        mcu = [s.mcu_timestamp for s in slist]
        wall = [s.wall_time_s for s in slist]
        hz = n / dur
        mono = all(b >= a for a, b in zip(mcu, mcu[1:]))
        nonzero = any(mcu)
        # Use the REAL calibration primitive the writer relies on.
        slope, _intercept, res_std = lr3.calibrate_mcu_clock(mcu, wall)
        engage = (n >= _MIN_SAMPLES and nonzero
                  and res_std < _MAX_RES_STD_S and slope != 0.0)
        any_engaged = any_engaged or engage
        print(_fmt_row(sid, n, f"{hz:.1f}", mono, nonzero, engage,
                       f"{slope:.3e}", f"{res_std*1000:.2f}"))

    print("\n=== decoded value sanity (first sample per present sensor) ===")
    for sid in bc.KNOWN_SENSOR_IDS:
        slist = samples.get(sid, [])
        if not slist:
            continue
        first = slist[0].data
        parts = []
        for k, v in first.items():
            if isinstance(v, list):
                parts.append(f"{k}=[{len(v)}]")
            else:
                parts.append(f"{k}={v}")
        print(f"  {sid}: " + "  ".join(parts))

    ok = seen > 0 and any_engaged
    print(f"\n=== SMOKE TEST: {'PASS' if ok else 'FAIL'} "
          f"({seen}/{len(bc.KNOWN_SENSOR_IDS)} sensors present, "
          f"calibration {'engaged' if any_engaged else 'did NOT engage'}) ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
