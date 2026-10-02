"""Resolve an FR3 wrist camera without treating its cable port as its identity."""

from __future__ import annotations

from typing import Any


def wrist_camera_selector(config: dict[str, Any]) -> dict[str, Any]:
    raw = (config.get("fr3_teleop") or {}).get("wrist_camera")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("fr3_teleop.wrist_camera must be a mapping")
    serial = raw.get("serial", "")
    sid = raw.get("sensor_id")
    if not isinstance(serial, str):
        raise ValueError("wrist_camera.serial must be a string")
    if sid is not None and (type(sid) is not int or not 0 <= sid <= 15):
        raise ValueError("wrist_camera.sensor_id must be null or an integer in 0..15")
    return {"serial": serial.strip(), "sensor_id": sid}


def resolve_camera_roles(
    config: dict[str, Any],
    cameras: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Use the EEPROM serial when supplied; never fall back to another module.

    ``cameras`` is the active roster, with name/sensor_id per stream. A serial
    selector follows a module to a different port. With no serial available a
    sensor-id selector is explicit, but only identifies the cable port.
    """
    selector = wrist_camera_selector(config)
    roles: dict[str, Any] = {}
    for camera in cameras:
        name = str(camera["name"])
        sid = camera.get("sensor_id")
        # EEPROM summaries use cam_NN even when the recorder has another prefix.
        physical = identity.get(name) or identity.get(f"cam_{sid:02d}" if type(sid) is int else "") or {}
        roles[name] = {
            "role": "unassigned",
            "sensor_id": sid,
            "serial": physical.get("serial"),
        }
    serial, sid = selector["serial"], selector["sensor_id"]
    selected = [name for name, role in roles.items()
                if (str(role["serial"] or "").strip() == serial if serial else role["sensor_id"] == sid)]
    if not serial and sid is None:
        state, message = "unconfigured", "Set the wrist camera serial or sensor ID after installation"
        selected = []
    elif len(selected) == 1:
        state, message = "resolved", f"FR3 wrist camera: {selected[0]}"
        roles[selected[0]]["role"] = "wrist"
        roles[selected[0]]["mount"] = "fr3_end_effector"
        roles[selected[0]]["extrinsics_mode"] = "eye_in_hand_uncalibrated"
    elif not selected:
        state, message = "missing", "Configured wrist camera is not in the active camera roster"
    else:
        state, message = "ambiguous", "Several active cameras match the wrist selector"
    return {"selector": selector, "state": state, "camera": selected[0] if state == "resolved" else None,
            "message": message, "cameras": roles}


def require_wrist_camera(roles: dict[str, Any]) -> None:
    if roles.get("state") not in ("unconfigured", "resolved"):
        raise RuntimeError(str(roles.get("message") or "Wrist camera is unresolved"))


def wrist_camera_names(meta: dict[str, Any]) -> set[str]:
    """Moving streams must not be used with a fixed base-to-camera transform."""
    roles = (meta.get("camera_roles") or {}).get("cameras") or {}
    return {name for name, role in roles.items() if role.get("role") == "wrist"}
