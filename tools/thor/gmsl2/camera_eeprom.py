#!/usr/bin/env python3
"""Which physical camera is on which GMSL2 port: read it from the module's EEPROM.

The sensor-id Argus hands out (``cam_%02u``) is the deserializer link the cable
is plugged into, not the camera. Swap two cables and the ids swap with them --
and every per-camera constant keyed on the id (intrinsics, extrinsics, the
serial in ``camera_serial_map.yaml``) silently lands on the wrong camera.
2026-09-28 found that this had already happened at the 09-23 remount, and that
the hand-written serial map had never matched the hardware.

Each SG2-AR0234C-G2F module carries an 8 KB EEPROM (0x50 on the module,
translated per link to ``0x60 + link`` on its deserializer's bus). It holds the
factory serial number and a factory intrinsic calibration. Layout, read off
four modules (all agree):

* ``0x000``  u16 0x001a, u16 width, u16 height, ...
* ``0x060``  u16 width, u16 height, u8 model (1 = OpenCV rational), then 12
  little-endian float64: ``fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6``
* ``0x120``  ASCII serial, e.g. ``H120K-I05130057``

Reading is ``i2ctransfer`` with a two-byte address pointer and no data -- the
EEPROM is never written. Do it with the cameras idle; a link the driver did not
bring up at boot does not answer (reported as such, not guessed).

    sudo -n true && python3 tools/thor/gmsl2/camera_eeprom.py \
        --intrinsics outputs/calibration/calib_20260923_cam13refit_intrinsics
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

FIRST_BUS = 17  # i2c-12 mux channels 0..3, one per MAX96726
LINKS_PER_DESERIALIZER = 4
EEPROM_BASE_ADDR = 0x60
READ_BYTES = 0x140  # through the serial number

CALIB_OFFSET = 0x60
SERIAL_OFFSET = 0x120
SERIAL_MAX = 32
SERIAL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9-]{5,}")
DIST_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")

Reader = Callable[[int, int, int, int], bytes | None]
"""``(bus, addr, offset, length) -> bytes`` or None when the device does not answer."""


def eeprom_location(sid: int) -> tuple[int, int]:
    """``(i2c bus, 7-bit address)`` of sensor-id ``sid``'s module EEPROM."""
    return FIRST_BUS + sid // LINKS_PER_DESERIALIZER, EEPROM_BASE_ADDR + sid % LINKS_PER_DESERIALIZER


def i2ctransfer_reader(bus: int, addr: int, offset: int, length: int) -> bytes | None:
    out = bytearray()
    for start in range(offset, offset + length, 32):
        n = min(32, offset + length - start)
        cmd = ["sudo", "-n", "i2ctransfer", "-y", str(bus),
               f"w2@0x{addr:02x}", f"0x{start >> 8:02x}", f"0x{start & 0xFF:02x}", f"r{n}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return None
        out.extend(int(tok, 16) for tok in res.stdout.split())
    return bytes(out)


@dataclass
class ModuleIdentity:
    sid: int
    bus: int
    addr: int
    answered: bool
    serial: str | None = None
    width: int | None = None
    height: int | None = None
    model: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist: dict[str, float] = field(default_factory=dict)
    problem: str | None = None

    @property
    def camera_name(self) -> str:
        return f"cam_{self.sid:02d}"


def parse_eeprom(sid: int, raw: bytes | None) -> ModuleIdentity:
    bus, addr = eeprom_location(sid)
    ident = ModuleIdentity(sid=sid, bus=bus, addr=addr, answered=raw is not None)
    if raw is None:
        ident.problem = "EEPROM did not answer (no camera, link down, or not brought up at boot)"
        return ident
    if len(raw) < SERIAL_OFFSET + 16:
        ident.problem = f"short read ({len(raw)} bytes)"
        return ident
    text = raw[SERIAL_OFFSET:SERIAL_OFFSET + SERIAL_MAX].split(b"\xff")[0].split(b"\x00")[0]
    match = SERIAL_PATTERN.match(text.decode("ascii", "replace"))
    ident.serial = match.group(0) if match else None
    w, h, model = struct.unpack_from("<HHB", raw, CALIB_OFFSET)
    vals = struct.unpack_from("<12d", raw, CALIB_OFFSET + 5)
    if model != 1 or not all(math.isfinite(v) for v in vals) or not (100.0 < vals[0] < 10000.0):
        ident.problem = f"no factory calibration recognised (model byte {model})"
    else:
        ident.width, ident.height, ident.model = w, h, model
        ident.fx, ident.fy, ident.cx, ident.cy = vals[:4]
        ident.dist = dict(zip(DIST_NAMES, vals[4:], strict=True))
    if ident.serial is None:
        ident.problem = (ident.problem + "; " if ident.problem else "") + "no serial number at 0x120"
    return ident


def read_modules(sids: Sequence[int], reader: Reader = i2ctransfer_reader) -> list[ModuleIdentity]:
    out = []
    for sid in sids:
        bus, addr = eeprom_location(sid)
        out.append(parse_eeprom(sid, reader(bus, addr, 0, READ_BYTES)))
    return out


# --------------------------------------------------------------------------- #
# who is it: factory intrinsics against calibrated ones                         #
# --------------------------------------------------------------------------- #
def load_calibrated(intrinsics_dir: Path) -> dict[str, tuple[float, float, float]]:
    """``{camera_name: (fx, cx, cy)}`` from ``converted/<cam>_<serial>/intrinsics_producer.json``."""
    out: dict[str, tuple[float, float, float]] = {}
    root = intrinsics_dir / "converted" if (intrinsics_dir / "converted").is_dir() else intrinsics_dir
    for path in sorted(root.glob("*/intrinsics_producer.json")):
        data = json.loads(path.read_text())
        k_mat = data.get("camera_matrix") or data.get("K")
        if not k_mat:
            continue
        name = re.match(r"(cam_\d+)", path.parent.name)
        if name:
            out[name.group(1)] = (float(k_mat[0][0]), float(k_mat[0][2]), float(k_mat[1][2]))
    return out


def match_calibrated(
    ident: ModuleIdentity, calibrated: dict[str, tuple[float, float, float]]
) -> dict[str, object] | None:
    """Nearest calibrated camera by principal point and focal length (px).

    Factory and our own calibrations of the same module agree to 1-4 px in the
    principal point; different modules on this rig are >= 12 px apart. The
    margin to the runner-up is reported so a close call is visible.
    """
    if ident.cx is None or not calibrated:
        return None
    dists = sorted(
        (math.hypot(ident.cx - cx, ident.cy - cy, 0.5 * (ident.fx - fx)), name)
        for name, (fx, cx, cy) in calibrated.items()
    )
    best_d, best = dists[0]
    runner = dists[1][0] if len(dists) > 1 else float("inf")
    return {"camera": best, "distance_px": best_d, "runner_up_px": runner,
            "same_port": best == ident.camera_name}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sids", default="0-15", help="e.g. 0-15 or 4,6,8")
    ap.add_argument("--intrinsics", type=Path, default=None,
                    help="calibration intrinsics dir to fingerprint against (converted/<cam>_*/)")
    ap.add_argument("--json", type=Path, default=None, help="write the result here")
    ap.add_argument("--match-px", type=float, default=8.0,
                    help="a fingerprint farther than this matches nothing")
    args = ap.parse_args(argv)

    sids: list[int] = []
    for part in args.sids.split(","):
        a, _, b = part.partition("-")
        sids.extend(range(int(a), int(b or a) + 1))
    modules = read_modules(sids)
    calibrated = load_calibrated(args.intrinsics) if args.intrinsics else {}

    rows = []
    for m in modules:
        row = asdict(m)
        row["camera_name"] = m.camera_name
        match = match_calibrated(m, calibrated)
        if match is not None and match["distance_px"] > args.match_px:
            match = {**match, "camera": None, "same_port": False}
        row["calibrated_match"] = match
        rows.append(row)
        if not m.answered:
            continue
        fp = "—" if m.cx is None else f"fx {m.fx:7.1f} cx {m.cx:6.1f} cy {m.cy:6.1f}"
        verdict = ""
        if match is not None:
            if match["camera"] is None:
                verdict = f"no calibrated camera within {args.match_px:g} px"
            elif match["same_port"]:
                verdict = f"= calibrated {match['camera']} ({match['distance_px']:.1f} px)"
            else:
                verdict = (f"!! calibrated as {match['camera']} ({match['distance_px']:.1f} px, "
                           f"next {match['runner_up_px']:.1f}) -- this port's constants are another camera's")
        print(f"{m.camera_name}  {m.serial or '?':<18} {fp}  {verdict}".rstrip())
    silent = [m.camera_name for m in modules if not m.answered]
    if silent:
        print(f"no answer: {', '.join(silent)}")
    if args.json:
        args.json.write_text(json.dumps({"modules": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
