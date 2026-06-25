"""Smoke-test the BOX gripper by cycling it closed/open a few times.

Uses the self-contained wheel (``Box()`` with no ``so_path``, same as the
production ``BoxClient``) -- the local ``lib/`` is a trimmed set that would
need ``LD_LIBRARY_PATH``/compat symlinks to ``dlopen`` libtinyxml2 etc.
Requires ``BOX_SDK_URDF`` from ``setup_env.sh`` and the host holding
``192.168.2.45`` so the BOX's UDP telemetry has a listener.

Sequence:
  set_mode(1)  -> control mode
  for N cycles: set_clamp_pos(close) , dwell , set_clamp_pos(open) , dwell
  set_mode(0)  -> back to collection/trigger mode

Clamp position is the commanded opening in METERS (box_set_clamp_pos).
On this rig fully-open reads ~0.095 m and demo.py uses 0.004 m as "closed".

Usage:
    source setup_env.sh
    python gripper_cycle_test.py [cycles] [close_m] [open_m] [dwell_s]

Defaults: cycles=3  close=0.004  open=0.090  dwell=1.5
"""

import sys
import time

from box_sdk import Box


def _read_distance(box) -> float | None:
    """Best-effort read-back of the current gripper opening (meters)."""
    rc, snap = box.get_sensor_cache()
    if rc == 0 and snap.valid:
        return float(snap.data.gripper_data.distance)
    return None


def main() -> int:
    cycles = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
    close_m = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.004
    open_m = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.090
    dwell_s = float(sys.argv[4]) if len(sys.argv) >= 5 else 1.5

    box = Box()  # bundled self-contained lib (no so_path) -- see module docstring
    box.start(bind_ip="0.0.0.0", bind_port=15000,
              remote_ip="192.168.2.60", remote_port=15000)

    rc = box.set_mode(1)  # control mode
    print("set_mode(1):", rc, box.err_str(rc))
    if rc != 0:
        print("WARNING: could not enter control mode; gripper may not move")
    time.sleep(1.0)

    try:
        for i in range(1, cycles + 1):
            for label, pos in (("CLOSE", close_m), ("OPEN", open_m)):
                rc = box.set_clamp_pos(pos)
                print(f"[cycle {i}/{cycles}] {label:5s} set_clamp_pos({pos:.3f}): "
                      f"{rc} {box.err_str(rc)}")
                time.sleep(dwell_s)
                dist = _read_distance(box)
                if dist is not None:
                    print(f"           -> measured distance: {dist:.4f} m")
                else:
                    print("           -> no valid sensor read-back")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        rc = box.set_mode(0)  # back to collection/trigger mode
        print("set_mode(0):", rc, box.err_str(rc))
        box.stop()
        box.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
