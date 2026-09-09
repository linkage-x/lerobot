#!/usr/bin/env python3
"""Offline preflight of contact IK. NO robot/gripper connection or movement.

This is not a hardware launcher. --execute and --start-only fail closed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, .5445, -3.0159])
UPPER = np.array([2.7437, 1.7837, 2.9007, -.1518, 2.8065, 4.5169, 3.0159])
WALL_BAND = np.array([.24, .18, .18, .18, .0698, .0698, .0698])
PATCH_ID = "fr3-5.8.1-protocol9-20260907-candidate"
SETTINGS = dict(speed_factor=.01, max_deviation_rad=0., planning_timeout_s=3.,
                chunk_size=25, sample_period_s=.001)
BLOCKERS = [
    "Right BOX 1819152274 cube-to-TCP calibration was copied, not independently validated.",
    "Table is +110 mm above mounting surface, but footprint and obstacles are not measured.",
    "Current-state to first-frame transit has not been collision validated.",
    "Continuous collision/Cartesian tracking and gripper time synchronization are not certified.",
    "The isolated native candidate has offline tests only, no physical tracking validation.",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_contact_ik(path):
    """Reject bad input; never silently drop, reorder or clamp samples."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = JOINTS + ["episode_index", "frame_index", "timestamp_s",
                            "gripper_width_m", "tcp_retreat_m", "ik_ok"]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Missing contact-IK columns")
        episodes = {}
        last_episode = -1
        for row_number, row in enumerate(reader, 2):
            ep, frame = int(row["episode_index"]), int(row["frame_index"])
            if ep < 0 or frame < 0 or ep < last_episode:
                raise ValueError(f"Invalid episode/frame order at row {row_number}")
            last_episode = ep
            q = np.array([float(row[name]) for name in JOINTS])
            t, width, retreat = [float(row[name]) for name in
                                  ("timestamp_s", "gripper_width_m", "tcp_retreat_m")]
            if not np.isfinite(np.r_[q, t, width, retreat]).all() or t < 0:
                raise ValueError(f"Nonfinite/invalid sample at row {row_number}")
            if row["ik_ok"].strip().lower() not in ("true", "1"):
                raise ValueError(f"IK failure at row {row_number}; dropping frames is forbidden")
            if (q < LOWER).any() or (q > UPPER).any():
                raise ValueError(f"Joint bounds violated at row {row_number}")
            d = width * 1000
            radicand = 49.699345 ** 2 - 5.474953 * d - d ** 2 / 4
            if width < 0 or radicand < 0:
                raise ValueError(f"Opening outside V2 formula domain at row {row_number}")
            predicted = (49.699345 - np.sqrt(radicand)) / 1000
            if abs(predicted - retreat) > 1e-8:
                raise ValueError(f"V2 retreat/width mismatch at row {row_number}")
            seq = episodes.setdefault(ep, [])
            if (not seq and frame != 0) or (seq and (frame != seq[-1]["frame_index"] + 1 or t <= seq[-1]["timestamp_s"])):
                raise ValueError(f"Missing/duplicate/unordered sample at row {row_number}")
            seq.append(dict(frame_index=frame, timestamp_s=t, q=q, width_m=width,
                            retreat_m=retreat, original=row))
    if not episodes or any(len(seq) < 2 for seq in episodes.values()):
        raise ValueError("Each episode must contain at least two valid samples")
    return episodes


def require_candidate(core):
    if getattr(core, "_FR3_LIMITS_PATCH", None) != PATCH_ID or getattr(core, "_FR3_CONFIGURED_NATIVE_API", None) != 1:
        raise RuntimeError("Isolated guarded FR3 candidate is not loaded")
    limits = core.joint_limits_for_server_version(9)
    for name, expected in (("lower", LOWER), ("upper", UPPER)):
        if not np.array_equal(np.asarray(limits[name]), expected):
            raise RuntimeError("Candidate limits differ from verified FR3 5.8.1 limits")
    for unknown in (0, 5, 8, 10, 65535):
        try:
            core.joint_limits_for_server_version(unknown)
        except RuntimeError:
            continue
        raise RuntimeError("Unknown robot version was accepted")


def preflight(csv_path, urdf_path, core):
    require_candidate(core)
    episodes = read_contact_ik(csv_path)
    tree = ET.parse(urdf_path)
    for i, name in enumerate(JOINTS):
        lim = tree.find(f"./joint[@name='{name}']/limit")
        if lim is None or float(lim.attrib["lower"]) != LOWER[i] or float(lim.attrib["upper"]) != UPPER[i]:
            raise ValueError(f"URDF limit mismatch: {name}")
    report = dict(input_sha256=sha256(csv_path), urdf_sha256=sha256(urdf_path),
                  core_sha256=sha256(core.__file__), core_path=core.__file__,
                  script_sha256=sha256(__file__), settings=SETTINGS,
                  hardware_ready=False, controller_started=False, gripper_commands_sent=0,
                  blockers=BLOCKERS, episodes=[], limitations=[
                      "Positions sampled every <=1 ms; not a continuous collision proof.",
                      "Zero corner blending preserves the joint polyline, not exact Cartesian interpolation or recorded timing.",
                      "Chunk stops may have acceleration/jerk discontinuities; dynamics are not certified.",
                      "The actual start state is not included in this offline path."])
    for ep, seq in episodes.items():
        q = np.array([f["q"] for f in seq])
        if np.min(np.minimum(q - LOWER, UPPER - q) - WALL_BAND) <= 0:
            raise ValueError(f"Episode {ep} enters a virtual-wall band")
        chunks = []
        for start in range(0, len(q) - 1, SETTINGS["chunk_size"] - 1):
            waypoints = q[start:start + SETTINGS["chunk_size"]]
            trajectory = core.JointTrajectory(waypoints.tolist(), SETTINGS["speed_factor"],
                                               SETTINGS["max_deviation_rad"], SETTINGS["planning_timeout_s"])
            duration = float(trajectory.get_duration())
            if not np.isfinite(duration) or not 0 < duration < 120:
                raise ValueError("Invalid native planner duration")
            times = np.linspace(0, duration, int(np.ceil(duration / SETTINGS["sample_period_s"])) + 1)
            planned = np.array([trajectory.get_joint_positions(float(t)) for t in times])
            margins = np.minimum(planned - LOWER, UPPER - planned)
            if not np.isfinite(planned).all() or np.min(margins - WALL_BAND) <= 0:
                raise ValueError("Native planned samples enter joint wall bands")
            if not np.allclose(planned[0], waypoints[0], atol=1e-7, rtol=0) or not np.allclose(planned[-1], waypoints[-1], atol=1e-7, rtol=0):
                raise ValueError("Native planner endpoint mismatch")
            chunks.append(dict(start=start, end=start + len(waypoints) - 1, duration_s=duration,
                               samples=len(times), minimum_limit_margin_rad=float(margins.min())))
        result = dict(episode=ep, frames=len(seq), offline_position_and_planning_pass=True,
                      planned_duration_s=sum(c["duration_s"] for c in chunks), chunks=chunks)
        report["episodes"].append(result)
        print(f"Episode {ep}: {len(seq)} frames / {len(chunks)} chunks, offline position/planning PASS", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    args = parser.parse_args(argv)
    # Must precede SDK loading, robot state reads and input file access.
    if args.execute or args.start_only:
        parser.exit(9, "BLOCKED: offline-only candidate. " + " ".join(BLOCKERS) + "\n")
    if any(x is None for x in (args.csv, args.urdf, args.output)):
        parser.error("--csv, --urdf and --output are required for offline preflight")
    if args.output.exists():
        raise FileExistsError(f"Preserve earlier report: {args.output}")
    from panda_py import _core  # Import only; never instantiate Panda/Robot.
    report = preflight(args.csv, args.urdf, _core)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(f"Report: {args.output}; hardware replay remains BLOCKED")


if __name__ == "__main__":
    main()
