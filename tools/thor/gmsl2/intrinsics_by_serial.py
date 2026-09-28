"""Intrinsics follow the lens, not the port: assemble a run by module serial.

A lens is calibrated once; the cable it hangs on changes whenever the rig is
rebuilt. Keyed on the port (``cam_%02u``), the lens constants silently land on
whichever module is plugged in there -- 09-23 and again 09-28, found only by
reading the module EEPROMs. So a calibration capture is solved against
intrinsics looked up by the serial each port reported at Connect (recorded in
every episode's ``meta.json`` as ``camera_identity``), and re-labelled with the
port that serial is on now.

Where a serial's intrinsics come from, in order:

1. the newest intrinsics run under ``outputs/calibration`` whose summary says
   its serials were read from the EEPROM (``serial_source: eeprom``). Every run
   exported through this path is one, so the table maintains itself;
2. ``camera_intrinsics_registry.json`` beside this file, for runs exported
   before 09-28. Their directory names carry serials from the hand-written
   ``camera_serial_map.yaml``, which never matched the hardware, so each camera
   in them was tied to its module by fingerprint: the factory intrinsics in the
   EEPROM against the calibrated ones (1-4 px apart for the same module, >= 10 px
   for different ones). ``camera_eeprom.py --register-intrinsics`` writes it.

The producer file's numbers are copied untouched; only its camera name and
serial are rewritten, and where it came from is recorded with its sha256.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).with_name("camera_intrinsics_registry.json")
SERIAL_SOURCE_EEPROM = "eeprom"


@dataclass(frozen=True)
class IntrinsicsSource:
    serial: str
    run: str  # run directory name
    camera: str  # the camera name the lens was calibrated under in that run
    producer: Path  # absolute path of its intrinsics_producer.json
    how: str  # "eeprom_run" or "registry"


def capture_port_serials(episodes_dir: Path) -> dict[str, str]:
    """``{cam_NN: serial}`` read at Connect, from every episode of a capture.

    Empty when the capture predates identity recording. A port that reported
    two different serials inside one capture had a camera swapped mid-capture,
    which makes the capture itself unusable, so that raises.
    """
    seen: dict[str, set[str]] = {}
    for meta_path in sorted(Path(episodes_dir).glob("episode_*/meta.json")):
        try:
            identity = json.loads(meta_path.read_text(encoding="utf-8")).get("camera_identity") or {}
        except (OSError, ValueError):
            continue
        for camera, entry in identity.items():
            serial = entry.get("serial") if isinstance(entry, dict) else None
            if serial:
                seen.setdefault(str(camera), set()).add(str(serial).upper())
    changed = {cam: sorted(s) for cam, s in seen.items() if len(s) > 1}
    if changed:
        detail = "；".join(f"{cam}: {', '.join(s)}" for cam, s in sorted(changed.items()))
        raise ValueError(f"采集过程中有端口换过相机（{detail}），这段采集不能用来标定")
    return {cam: next(iter(s)) for cam, s in sorted(seen.items())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def eeprom_run_sources(calib_root: Path) -> dict[str, IntrinsicsSource]:
    """Serial -> producer, from runs that were exported with EEPROM serials. Newest wins."""
    runs = []
    for summary_path in Path(calib_root).glob("*/summary.json"):
        summary = _load_json(summary_path)
        if not isinstance(summary, dict) or summary.get("serial_source") != SERIAL_SOURCE_EEPROM:
            continue
        if not isinstance(summary.get("cameras"), list):
            continue
        runs.append((str(summary.get("timestamp_utc") or ""), summary_path.parent, summary))
    out: dict[str, IntrinsicsSource] = {}
    for _, run_dir, summary in sorted(runs, key=lambda r: r[0]):
        for row in summary["cameras"]:
            serial = str(row.get("camera_serial") or "").upper()
            producer = Path(str(row.get("intrinsics_json") or ""))
            if not producer.is_absolute():
                producer = run_dir / "converted" / producer.parent.name / producer.name
            if not serial or row.get("status") != "ok" or not producer.is_file():
                continue
            out[serial] = IntrinsicsSource(serial, run_dir.name, str(row.get("camera_name") or ""), producer, "eeprom_run")
    return out


def registry_sources(calib_root: Path, registry: Path = REGISTRY) -> dict[str, IntrinsicsSource]:
    data = _load_json(registry)
    if not isinstance(data, dict):
        return {}
    out: dict[str, IntrinsicsSource] = {}
    for serial, entry in (data.get("serials") or {}).items():
        producer = Path(calib_root) / entry["run"] / entry["producer"]
        if producer.is_file():
            out[serial.upper()] = IntrinsicsSource(serial.upper(), entry["run"], entry["camera"], producer, "registry")
    return out


def resolve_sources(
    serials: list[str], calib_root: Path, registry: Path = REGISTRY
) -> tuple[dict[str, IntrinsicsSource], list[str]]:
    """Where each serial's intrinsics come from, and the serials nothing covers."""
    found = {**registry_sources(calib_root, registry), **eeprom_run_sources(calib_root)}
    sources = {s: found[s] for s in serials if s in found}
    return sources, sorted(s for s in serials if s not in found)


def stage_run(
    port_serials: dict[str, str],
    sources: dict[str, IntrinsicsSource],
    out_dir: Path,
) -> dict[str, Any]:
    """Write an intrinsics run for the ports as they are now, one lens per serial.

    Laid out like an exported run (``converted/<cam>_<serial>/``, ``summary.json``)
    so the bundle adjustment, the exporter's carry-forward and the production
    tracker all read it unchanged. Ports whose serial has no source are left out;
    the caller decides whether that is acceptable.
    """
    out_dir = Path(out_dir)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    models: set[str] = set()
    for camera, serial in sorted(port_serials.items()):
        source = sources.get(serial)
        if source is None:
            continue
        payload = json.loads(source.producer.read_text(encoding="utf-8"))
        models.add(str(payload.get("model") or ""))
        payload["camera_name"] = camera
        payload["camera_serial"] = serial
        payload["intrinsics_origin"] = {
            "run": source.run,
            "camera": source.camera,
            "file_sha256": _sha256(source.producer),
            "matched_by": source.how,
        }
        cam_dir = out_dir / "converted" / f"{camera}_{serial}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        target = cam_dir / "intrinsics_producer.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "camera_name": camera,
                "camera_serial": serial,
                "enabled": True,
                # "ok" is what every reader of a run loads; that nothing was
                # re-measured is said by calibration_status, as for carry-forward
                "status": "ok",
                "calibration_status": "by_serial",
                "source": "by_serial",
                "model": payload.get("model"),
                "source_run": source.run,
                "source_camera": source.camera,
                "source_sha256": _sha256(source.producer),
                "intrinsics_json": str(target),
            }
        )
    if len(models) > 1:
        raise ValueError(f"按序列号取到的内参模型不一致：{sorted(models)}，同一个 run 只能有一种模型")
    summary = {
        "timestamp_utc": stamp,
        "config": "intrinsics_by_serial",
        "source": "by_serial",
        "camera_model": models.pop() if models else "",
        "serial_source": SERIAL_SOURCE_EEPROM,
        "port_serials": dict(sorted(port_serials.items())),
        "counts": {"total": len(rows), "ok": len(rows), "by_serial": len(rows), "failed": 0, "skipped": 0},
        "cameras": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_serial_map(port_serials: dict[str, str], path: Path) -> Path:
    """The map the exporter names cameras with -- the true one, from the capture."""
    lines = ["# Written from the capture's EEPROM reads; not hand-edited.", "camera_serial_by_name:"]
    lines += [f"  {cam}: {serial}" for cam, serial in sorted(port_serials.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def describe(port_serials: dict[str, str], sources: dict[str, IntrinsicsSource]) -> list[str]:
    """One line per port, for the solve log: where its lens came from, and whether it moved."""
    lines = []
    for camera, serial in sorted(port_serials.items()):
        source = sources.get(serial)
        if source is None:
            lines.append(f"{camera} {serial}: 没有已标定的内参")
            continue
        moved = "" if source.camera == camera else f"（标定时在 {source.camera}）"
        lines.append(f"{camera} {serial}: 内参取自 {source.run}/{source.camera}{moved}")
    return lines
