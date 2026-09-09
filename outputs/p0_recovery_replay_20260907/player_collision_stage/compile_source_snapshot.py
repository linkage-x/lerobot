#!/usr/bin/env python3
"""Compile and inspect timestamped FR3 replay plans. Hardware execution is blocked.

Recorded joint/width knots are interpolated, never dropped, offset or clamped.
The continuous interpolation is NEW and needs its own geometry verification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.spatial.transform import Rotation

from replay_ik_trajectory_guarded import JOINTS, LOWER, UPPER, WALL_BAND, read_contact_ik, sha256

SCHEMA = "fr3-timed-offline-candidate-v1"
VELOCITY = .12
ACCELERATION = .4
JERK = 5.
WIDTH_SPEED = .01
WIDTH_MAX = .0887398049235344


def transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def urdf_chain(path):
    tree = ET.parse(path)
    by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
    chain, link = [], "corenetic_gripper_ee"
    while link != "base":
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    origins, axes, names, pending = [], [], [], np.eye(4)
    for joint in reversed(chain):
        pending = pending @ transform(joint.find("origin"))
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        if name not in JOINTS or joint.get("type") != "revolute":
            raise ValueError("Unexpected moving joint on the configured closed-TCP chain")
        idx = JOINTS.index(name)
        lim = joint.find("limit")
        if float(lim.get("lower")) != LOWER[idx] or float(lim.get("upper")) != UPPER[idx]:
            raise ValueError("URDF/FR3 software bound mismatch")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not np.isclose(np.linalg.norm(axis), 1.):
            raise ValueError("Invalid URDF axis")
        names.append(name)
        origins.append(pending.tolist())
        axes.append(axis.tolist())
        pending = np.eye(4)
    if names != JOINTS:
        raise ValueError("Wrong arm joint order")
    return dict(base_frame="base", tcp_frame="corenetic_gripper_ee", joint_names=names,
                origins=origins, axes=axes, tail=pending.tolist())


def fk(chain, q):
    pose = np.eye(4)
    for origin, axis, angle in zip(chain["origins"], chain["axes"], q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()
        pose = pose @ origin @ rot
    return pose @ chain["tail"]


def extrema(coeff, derivative=0, duration=1.):
    """All real stationary points of this low-degree scalar polynomial, plus ends."""
    c = np.polyder(coeff, derivative) / duration ** derivative
    roots = np.roots(np.trim_zeros(np.polyder(c), "f"))
    u = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1]
    values = np.polyval(c, u)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite polynomial extremum")
    return float(np.min(values)), float(np.max(values))


def bounds(times, coeff):
    lo, hi = np.full(8, np.inf), np.full(8, -np.inf)
    peaks = np.zeros((3, 8))
    for dt, segment in zip(np.diff(times), coeff, strict=True):
        for joint in range(8):
            a, b = extrema(segment[:, joint])
            lo[joint], hi[joint] = min(lo[joint], a), max(hi[joint], b)
            for order in (1, 2, 3):
                a, b = extrema(segment[:, joint], order, dt)
                peaks[order - 1, joint] = max(peaks[order - 1, joint], abs(a), abs(b))
    return lo, hi, peaks


def evaluate(episode, time_s, derivative=0):
    t = np.asarray(episode["times_s"])
    c = np.asarray(episode["coeff_descending_unit_interval"])
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Invalid controller time")
    if time_s > t[-1] and derivative:
        return np.zeros(8)
    phase = min(time_s, t[-1])
    idx = min(int(np.searchsorted(t, phase, side="right") - 1), len(c) - 1)
    dt = t[idx + 1] - t[idx]
    polynomial = c[idx]
    for _ in range(derivative):
        polynomial = polynomial[:-1] * np.arange(len(polynomial) - 1, 0, -1)[:, None]
    return np.polyval(polynomial, (phase - t[idx]) / dt) / dt ** derivative


def compile_episode(ep, seq, minimum_scale):
    raw_t = np.array([s["timestamp_s"] for s in seq])
    times = raw_t - raw_t[0]
    q = np.array([s["q"] for s in seq])
    widths = np.array([s["width_m"] for s in seq])
    if len(q) < 6 or widths.max() > WIDTH_MAX + 1e-12:
        raise ValueError("Too few knots or width outside measured V2 actuator range")
    spline = make_interp_spline(times, q, k=5,
                               bc_type=([(1, np.zeros(7)), (2, np.zeros(7))],
                                        [(1, np.zeros(7)), (2, np.zeros(7))]))
    for knot in np.unique(spline.t):
        if np.min(abs(times - knot)) > 1e-10:
            raise ValueError("Spline has an unexported internal breakpoint")
    grip = PchipInterpolator(times, widths)
    coeff = np.zeros((len(q) - 1, 6, 8))
    for i, dt in enumerate(np.diff(times)):
        for degree in range(6):
            coeff[i, 5 - degree, :7] = spline(times[i], nu=degree) * dt ** degree / math.factorial(degree)
        for degree in range(4):
            coeff[i, 5 - degree, 7] = grip.c[3 - degree, i] * dt ** degree
    lo, hi, raw_peaks = bounds(times, coeff)
    if (lo[:7] <= LOWER + WALL_BAND).any() or (hi[:7] >= UPPER - WALL_BAND).any():
        raise ValueError("Interpolated path enters an FR3 virtual-wall band")
    if lo[7] < -1e-12 or hi[7] > WIDTH_MAX + 1e-12:
        raise ValueError("Interpolated opening is out of range")
    required = max(raw_peaks[0, :7].max() / VELOCITY,
                   np.sqrt(raw_peaks[1, :7].max() / ACCELERATION),
                   np.cbrt(raw_peaks[2, :7].max() / JERK), raw_peaks[0, 7] / WIDTH_SPEED)
    # Uniform clock stretch changes neither the geometric path nor the knots.
    scale = max(float(minimum_scale), float(required) * 1.01)
    result = dict(episode=ep, frames=len(q), times_s=(times * scale).tolist(),
                  original_times_s=raw_t.tolist(), time_scale=scale, duration_s=float(times[-1] * scale),
                  coeff_descending_unit_interval=coeff.tolist())
    evaluator = dict(result, times_s=times * scale, coeff_descending_unit_interval=coeff)
    values = np.array([evaluate(evaluator, t) for t in result["times_s"]])
    knot_error = np.max(abs(values - np.c_[q, widths]))
    if knot_error > 1e-10:
        raise ValueError("Exported polynomial does not retain every input knot")
    peak = raw_peaks / np.array([scale, scale ** 2, scale ** 3])[:, None]
    result["validation"] = dict(maximum_knot_error=float(knot_error),
        minimum_joint_rad=lo[:7].tolist(), maximum_joint_rad=hi[:7].tolist(),
        minimum_wall_clearance_rad=float(np.minimum(lo[:7] - LOWER - WALL_BAND, UPPER - WALL_BAND - hi[:7]).min()),
        maximum_joint_velocity_rad_s=float(peak[0, :7].max()),
        maximum_joint_acceleration_rad_s2=float(peak[1, :7].max()),
        maximum_joint_jerk_rad_s3=float(peak[2, :7].max()),
        maximum_width_velocity_m_s=float(peak[0, 7]), width_range_m=[float(lo[7]), float(hi[7])],
        endpoint_velocity_rad_s=float(max(abs(evaluate(result, 0., 1)[:7]).max(), abs(evaluate(result, result["duration_s"], 1)[:7]).max())),
        endpoint_acceleration_rad_s2=float(max(abs(evaluate(result, 0., 2)[:7]).max(), abs(evaluate(result, result["duration_s"], 2)[:7]).max())))
    return result


class GripperClock:
    """Pure scheduler: return a metric-width command on the ARM's elapsed clock.

    A transport must acknowledge commands, obtain fresh measurements and feed a
    native watchdog. This class does not talk to devices or imply synchrony.
    """
    def __init__(self, episode, rate_hz=15., deadband_m=.0005):
        if not np.isfinite([rate_hz, deadband_m]).all() or rate_hz <= 0 or deadband_m < 0:
            raise ValueError("Invalid scheduler parameters")
        self.episode, self.period, self.deadband = episode, 1. / rate_hz, deadband_m
        self.previous_phase = -1.
        self.last_sent_phase = -np.inf
        self.last_width = None
        self.final_sent = False

    def command(self, arm_elapsed_s):
        if not np.isfinite(arm_elapsed_s) or arm_elapsed_s < self.previous_phase or arm_elapsed_s < 0:
            raise ValueError("Arm clock is invalid or went backwards")
        self.previous_phase = arm_elapsed_s
        width = float(evaluate(self.episode, arm_elapsed_s)[7])
        final = arm_elapsed_s >= self.episode["duration_s"]
        if self.final_sent or arm_elapsed_s - self.last_sent_phase < self.period - 1e-12:
            return None
        if self.last_width is not None and abs(width - self.last_width) < self.deadband and not final:
            return None
        self.last_sent_phase, self.last_width, self.final_sent = arm_elapsed_s, width, final
        return width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-scale", type=float, default=8.)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    args = parser.parse_args()
    if args.execute or args.start_only:
        parser.exit(9, "BLOCKED: offline candidate; no robot/gripper connection is implemented.\n")
    if args.csv is None or args.urdf is None or args.output is None:
        parser.error("--csv, --urdf, --output required")
    if not np.isfinite(args.minimum_scale) or args.minimum_scale < 8:
        parser.error("Minimum time scale must be >= 8")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = dict(schema=SCHEMA, input_sha256=sha256(args.csv), urdf_sha256=sha256(args.urdf),
                script_sha256=sha256(__file__), hardware_ready=False, hardware_commands_sent=0,
                chain=urdf_chain(args.urdf),
                limits=dict(joint_velocity_rad_s=VELOCITY, joint_acceleration_rad_s2=ACCELERATION,
                            joint_jerk_rad_s3=JERK, width_speed_m_s=WIDTH_SPEED),
                limitations=["Preserves recorded knots, not exact continuous Cartesian interpolation.",
                             "Quintic joint interpolation requires new between-knot collision checks.",
                             "PCHIP width has no calibrated gripper actuator dynamics model.",
                             "No measured tabletop footprint, obstacles, start transit or hardware validation.",
                             "No original robot data, controller configuration or installed SDK was changed."], episodes=[])
    for ep, seq in read_contact_ik(args.csv).items():
        result = compile_episode(ep, seq, args.minimum_scale)
        plan["episodes"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("times_s", "original_times_s", "coeff_descending_unit_interval")}), flush=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
