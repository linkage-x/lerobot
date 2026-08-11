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

"""Verify Pika gripper read and write against the same config the FR3 teleop uses.

The Device Manager only echoes the gripper config, and the teleop runtime prints
nothing about the gripper, so a gripper that never moves gives no signal anywhere.
This runs the two halves separately -- readback first, then commanded motion -- so a
stuck sensor is distinguishable from a rejected command.

The arm is never touched; only the gripper serial link is opened.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import yaml

from lerobot.robots.franka_research3.backends import PikaGripperHardwareDriver

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "tools" / "fr3" / "fr3_record_config.yaml"
DEFAULT_TARGETS = (1.0, 0.5, 0.0, 1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Pika gripper read and write.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="lerobot-record YAML whose robot section supplies the gripper defaults.",
    )
    parser.add_argument("--port", default=None, help="Override the gripper serial port.")
    parser.add_argument("--max-width-mm", type=float, default=None, help="Override gripper_max_width_mm.")
    parser.add_argument(
        "--targets",
        default=",".join(str(value) for value in DEFAULT_TARGETS),
        help="Comma-separated normalized targets to command, in order (0.0 closed, 1.0 open).",
    )
    parser.add_argument("--read-samples", type=int, default=40, help="Readback samples for the read check.")
    parser.add_argument("--read-interval-s", type=float, default=0.025, help="Delay between read samples.")
    parser.add_argument("--settle-s", type=float, default=1.5, help="Settle time after each commanded target.")
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=8.0,
        help="Allowed |readback - target| for a commanded target to count as reached.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Only run the readback check; never command the gripper to move.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Listen on every serial port and report which one streams Pika gripper frames, then exit.",
    )
    parser.add_argument("--scan-seconds", type=float, default=1.5, help="Listen time per port when scanning.")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print the raw bytes arriving on the port and exit, without opening it through the SDK.",
    )
    parser.add_argument("--dump-seconds", type=float, default=2.0, help="Listen time for --dump.")
    return parser.parse_args(argv)


def holders_of(port: str) -> list[str]:
    """Processes with the port open.

    Two readers on one serial port each get a fragment of every frame, so the JSON
    never assembles -- the same failure shape as a wrong device. Worth ruling out first.
    """
    node = str(Path(port).resolve())
    holders: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return holders
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            targets = list(fd_dir.iterdir())
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for fd in targets:
            try:
                if str(fd.resolve()) != node:
                    continue
            except (PermissionError, FileNotFoundError, OSError):
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
            except (PermissionError, FileNotFoundError, OSError):
                cmdline = "<unreadable>"
            holders.append(f"pid {entry.name}: {cmdline or '<unknown>'}")
            break
    return holders


EMPTY_MOTOR_HINT = (
    "The gripper MCU is streaming well-formed frames, but the `motor` and `motorstatus` "
    "objects are empty -- it has no data from the motor driver to put in them. The USB "
    "serial converter is bus-powered, so the link comes up and bytes flow even when the "
    "motor itself has none: check the gripper's separate motor power supply and the cable "
    "between the MCU and the motor driver. No port or baud-rate change will fix this."
)


def diagnose_stream(text: str) -> tuple[str, str]:
    """Classify a captured stream into (verdict, explanation)."""
    if not text:
        return "silent", "Nothing arrived: the gripper is not streaming on this port."
    if '"motor":{' in text:
        return "healthy", "Frames carry a populated motor object."
    if '"motor"' in text:
        return "empty-motor", EMPTY_MOTOR_HINT
    if '"AS5047"' in text:
        return "sense", "This is a Pika Sense, not the gripper."
    return "foreign", "Traffic present but no Pika frame markers: wrong device or wrong baud rate."


def dump_port(port: str, *, seconds: float, baudrate: int = 460800) -> None:
    """Show what is actually on the wire, so frame damage is visible rather than inferred."""
    import serial

    chunks: list[bytes] = []
    with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as link:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            chunk = link.read(4096)
            if chunk:
                chunks.append(chunk)
    raw = b"".join(chunks)
    print(f"read {len(raw)} bytes in {seconds:.1f}s")
    text = raw.decode("utf-8", errors="replace") if raw else ""
    verdict, explanation = diagnose_stream(text)
    print(f"DIAGNOSIS: {verdict} -- {explanation}")
    if not raw:
        return

    frames = text.split("{")
    print(f"'{{' count={text.count('{')}  '}}' count={text.count('}')}  '\"motor\"' count={text.count(chr(34) + 'motor' + chr(34))}")
    non_ascii = sum(1 for byte in raw if byte > 0x7F)
    print(f"non-ASCII bytes={non_ascii} ({100.0 * non_ascii / len(raw):.1f}%)  -- a healthy stream is pure ASCII")
    print("\nfirst 3 candidate frames (repr, truncated to 240 chars each):")
    shown = 0
    for frame in frames[1:]:
        if not frame.strip():
            continue
        print(f"  {{{frame[:240]!r}")
        shown += 1
        if shown >= 3:
            break


DEV_ROOT = Path("/dev")
SERIAL_BY_ID = Path("/dev/serial/by-id")
SERIAL_BY_PATH = Path("/dev/serial/by-path")


def candidate_ports() -> list[str]:
    """Every serial device node, de-duplicated across by-id/by-path aliases."""
    tty = sorted(str(path) for path in DEV_ROOT.glob("ttyUSB*"))
    by_id = sorted(str(path) for path in SERIAL_BY_ID.glob("*")) if SERIAL_BY_ID.is_dir() else []
    seen: list[str] = []
    resolved_seen: set[str] = set()
    for port in [*tty, *by_id]:
        resolved = str(Path(port).resolve())
        if resolved in resolved_seen:
            continue
        resolved_seen.add(resolved)
        seen.append(port)
    return seen


def stable_aliases(port: str) -> dict[str, str | None]:
    """Find persistent /dev/serial aliases for a port.

    A CH340 adapter reports no USB serial number, so its by-id name
    (`usb-1a86_USB_Serial-if00-port0`) is not unique -- plug in two of them and the
    name resolves to whichever enumerated first. by-path is keyed on the physical USB
    topology instead, so it stays put as long as the cable stays in the same socket.
    """
    target = Path(port).resolve()
    aliases: dict[str, str | None] = {"by_path": None, "by_id": None}
    for kind, root in (("by_path", SERIAL_BY_PATH), ("by_id", SERIAL_BY_ID)):
        if not root.is_dir():
            continue
        matches = [str(entry) for entry in sorted(root.glob("*")) if entry.resolve() == target]
        if matches:
            aliases[kind] = matches[0]
        if kind == "by_id" and len(matches) > 1:
            aliases["by_id_ambiguous"] = "yes"
    return aliases


def scan_ports(ports: list[str], *, seconds: float, baudrate: int = 460800) -> list[dict[str, Any]]:
    """Listen passively on each port and look for the gripper's telemetry signature.

    Read-only on purpose: the ports being probed may belong to other hardware, so
    nothing is ever written to them.
    """
    import serial  # imported lazily so --scan is the only path needing pyserial

    results: list[dict[str, Any]] = []
    for port in ports:
        row: dict[str, Any] = {"port": port, "bytes": 0, "gripper_frames": False, "error": None}
        try:
            with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as link:
                buffer = ""
                deadline = time.perf_counter() + seconds
                while time.perf_counter() < deadline:
                    chunk = link.read(4096)
                    if chunk:
                        row["bytes"] += len(chunk)
                        buffer += chunk.decode("utf-8", errors="ignore")
                        buffer = buffer[-8192:]
                row["gripper_frames"] = '"motor"' in buffer and '"motorstatus"' in buffer
                row["sense_frames"] = '"AS5047"' in buffer
        except Exception as exc:  # noqa: BLE001 - reported per port
            row["error"] = repr(exc)
        results.append(row)
    return results


def load_gripper_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    robot = payload.get("robot") if isinstance(payload.get("robot"), dict) else {}
    return {
        "port": robot.get("gripper_port"),
        "backend": robot.get("gripper_backend", "pika"),
        "max_width_mm": robot.get("gripper_max_width_mm", 90.0),
        "command_rate_limit_hz": robot.get("gripper_command_rate_limit_hz", 15.0),
        "command_deadband_mm": robot.get("gripper_command_deadband_mm", 0.5),
    }


def read_check(driver, *, samples: int, interval_s: float) -> dict[str, Any]:
    """Sample the gripper readback without commanding anything."""
    widths_mm: list[float] = []
    for _ in range(max(samples, 1)):
        widths_mm.append(float(driver.get_width_mm()))
        if interval_s > 0.0:
            time.sleep(interval_s)
    return {
        "samples": len(widths_mm),
        "min_mm": min(widths_mm),
        "max_mm": max(widths_mm),
        "mean_mm": statistics.fmean(widths_mm),
        "spread_mm": max(widths_mm) - min(widths_mm),
    }


def write_check(
    driver,
    *,
    targets: list[float],
    settle_s: float,
    max_width_mm: float,
    read_settle_samples: int = 5,
) -> list[dict[str, Any]]:
    """Command each target and record where the gripper actually ended up."""
    results: list[dict[str, Any]] = []
    for target in targets:
        target_mm = float(target) * max_width_mm
        driver.set_position(float(target))
        # The driver defers a command that arrives inside its rate-limit window until the
        # next call, so nudge it once more before sleeping to flush any pending write.
        driver.set_position(float(target))
        if settle_s > 0.0:
            time.sleep(settle_s)
        widths_mm = []
        for _ in range(max(read_settle_samples, 1)):
            widths_mm.append(float(driver.get_width_mm()))
            time.sleep(0.02)
        measured_mm = statistics.fmean(widths_mm)
        results.append(
            {
                "target": float(target),
                "target_mm": target_mm,
                "measured_mm": measured_mm,
                "error_mm": abs(measured_mm - target_mm),
            }
        )
    return results


def evaluate(
    read_result: dict[str, Any],
    write_results: list[dict[str, Any]],
    *,
    tolerance_mm: float,
    max_width_mm: float,
    has_telemetry: bool = True,
    stream_verdict: str | None = None,
) -> tuple[bool, list[str]]:
    """Turn the raw measurements into a pass/fail verdict with actionable notes."""
    notes: list[str] = []
    ok = True

    if not has_telemetry:
        # Without telemetry the write results are meaningless: readback is the SDK's
        # initial 0.0, not a measurement. Report the cause instead of the symptom.
        if stream_verdict == "empty-motor":
            notes.append(f"MOTOR: {EMPTY_MOTOR_HINT}")
        else:
            notes.append(
                "LINK: no Pika gripper frame was ever parsed off this port, so every readback is "
                "the SDK default (0.0 mm) and every command went nowhere. Run --dump to see what "
                "is actually on the wire."
            )
        return False, notes

    if read_result["max_mm"] > max_width_mm:
        notes.append(
            f"READ: raw width peaks at {read_result['max_mm']:.1f} mm but max_width_mm={max_width_mm:.1f}; "
            "normalized readback saturates at 1.0 and gripper.pos loses range."
        )
        ok = False

    if not write_results:
        return ok, notes

    measured_span_mm = max(row["measured_mm"] for row in write_results) - min(
        row["measured_mm"] for row in write_results
    )
    commanded_span_mm = max(row["target_mm"] for row in write_results) - min(
        row["target_mm"] for row in write_results
    )
    if commanded_span_mm > 0.0 and measured_span_mm < tolerance_mm:
        notes.append(
            f"WRITE: commanded a {commanded_span_mm:.1f} mm span but readback only moved "
            f"{measured_span_mm:.1f} mm -- the gripper is not following commands."
        )
        ok = False

    missed = [row for row in write_results if row["error_mm"] > tolerance_mm]
    if missed:
        detail = ", ".join(
            f"{row['target']:.2f}->{row['measured_mm']:.1f}mm (want {row['target_mm']:.1f}mm)" for row in missed
        )
        notes.append(f"WRITE: {len(missed)}/{len(write_results)} targets outside {tolerance_mm:.1f} mm: {detail}")
        ok = False

    return ok, notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.scan:
        ports = candidate_ports()
        if not ports:
            print("No serial ports found under /dev/serial/by-id or /dev/ttyUSB*.")
            return 1
        print(f"Listening {args.scan_seconds:.1f}s on each of {len(ports)} port(s) at 460800 baud (read-only) ...")
        rows = scan_ports(ports, seconds=float(args.scan_seconds))
        matches = [row for row in rows if row["gripper_frames"]]
        for row in rows:
            if row["error"]:
                verdict = f"error {row['error']}"
            elif row["gripper_frames"]:
                verdict = "PIKA GRIPPER"
            elif row.get("sense_frames"):
                verdict = "pika sense (not the gripper)"
            elif row["bytes"]:
                verdict = "traffic, but not gripper frames"
            else:
                verdict = "silent"
            print(f"  {row['port']}  bytes={row['bytes']:>6}  {verdict}")
        if matches:
            found = matches[0]["port"]
            aliases = stable_aliases(found)
            print(f"\nPika gripper is on {found} ({Path(found).resolve()}).")
            if aliases["by_path"]:
                print("Set robot.gripper_port to this by-path alias -- it survives replugs into the")
                print("same USB socket, unlike the CH340's non-unique by-id name:")
                print(f"  gripper_port: {aliases['by_path']}")
            else:
                print("No /dev/serial/by-path alias exists for it; the bare node below is not stable")
                print("across replugs, so re-run --scan after any reconnection:")
                print(f"  gripper_port: {found}")
            if aliases["by_id"]:
                print(f"(by-id alias, not recommended when several CH340 adapters are attached: {aliases['by_id']})")
            return 0
        print("\nNo port streamed Pika gripper frames. Check that the gripper is powered and cabled.")
        return 1

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    gripper_config = load_gripper_config(args.config)

    port = args.port or gripper_config["port"]
    if not port:
        print(f"No gripper_port in {args.config}; pass --port explicitly.", file=sys.stderr)
        return 2
    if gripper_config["backend"] != "pika":
        print(
            f"Config selects gripper_backend={gripper_config['backend']!r}; this tool only verifies the pika backend.",
            file=sys.stderr,
        )
        return 2
    max_width_mm = float(args.max_width_mm or gripper_config["max_width_mm"])

    targets = [float(value) for value in str(args.targets).split(",") if value.strip()]
    out_of_range = [value for value in targets if not 0.0 <= value <= 1.0]
    if out_of_range:
        print(f"Targets must be within [0, 1]; got {out_of_range}", file=sys.stderr)
        return 2

    driver = PikaGripperHardwareDriver(
        serial_port=str(port),
        max_width_mm=max_width_mm,
        command_rate_limit_hz=gripper_config["command_rate_limit_hz"],
        command_deadband_mm=gripper_config["command_deadband_mm"],
    )

    print(f"config={args.config}")
    print(f"port={port}  max_width_mm={max_width_mm:.1f}")

    holders = holders_of(str(port))
    if holders:
        print("WARNING: this port is already open by another process; two readers split every")
        print("frame between them, so neither ever assembles a complete one. Stop it first:")
        for holder in holders:
            print(f"  {holder}")

    if args.dump:
        dump_port(str(port), seconds=float(args.dump_seconds))
        return 0

    print("Connecting ...")
    # Connect without the telemetry gate so this tool can report a dead link itself
    # rather than dying inside connect() with no measurements to show.
    driver.telemetry_timeout_s = 0.0
    driver.connect()
    stream_verdict: str | None = None
    try:
        has_telemetry = driver.has_telemetry()
        print(f"LINK  : telemetry={'yes' if has_telemetry else 'NO'}")
        if not has_telemetry:
            # Look at the wire directly to tell a dead/wrong port apart from a live MCU
            # that has nothing to report.
            latest = getattr(getattr(driver._gripper, "serial_comm", None), "buffer", "") or ""
            stream_verdict, _explanation = diagnose_stream(str(latest))
        read_result = read_check(
            driver,
            samples=int(args.read_samples),
            interval_s=float(args.read_interval_s),
        )
        print(
            "READ  : "
            f"samples={read_result['samples']} "
            f"min={read_result['min_mm']:.2f}mm max={read_result['max_mm']:.2f}mm "
            f"mean={read_result['mean_mm']:.2f}mm spread={read_result['spread_mm']:.2f}mm "
            f"normalized={read_result['mean_mm'] / max_width_mm:.3f}"
        )

        write_results: list[dict[str, Any]] = []
        if args.read_only:
            print("WRITE : skipped (--read-only)")
        elif not has_telemetry:
            print("WRITE : skipped (no telemetry -- commands cannot be verified)")
        else:
            print(f"WRITE : commanding {targets} (the gripper will move)")
            write_results = write_check(
                driver,
                targets=targets,
                settle_s=float(args.settle_s),
                max_width_mm=max_width_mm,
            )
            for row in write_results:
                print(
                    f"        target={row['target']:.2f} ({row['target_mm']:.1f}mm) "
                    f"measured={row['measured_mm']:.1f}mm error={row['error_mm']:.1f}mm"
                )
    finally:
        driver.disconnect()
        print("Disconnected.")

    ok, notes = evaluate(
        read_result,
        write_results,
        tolerance_mm=float(args.tolerance_mm),
        max_width_mm=max_width_mm,
        has_telemetry=has_telemetry,
        stream_verdict=stream_verdict,
    )
    for note in notes:
        print(f"  ! {note}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
