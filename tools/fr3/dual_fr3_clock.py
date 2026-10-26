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
Two FR3s' clocks against the host's: start skew, drift over a run, and what they cost T_LR (roadmap D2).

Both arms are commanded on one host schedule (``dual_fr3_replay_runtime``: frame k at ``t0 + k * period`` on
CLOCK_MONOTONIC), so the commands leave together; whether the arms *move* together is what this measures.
Each arm's state carries its own clock (libfranka's ``RobotState.time``, 1 ms, since the controller booted),
and each state is stamped again on the host when it arrives. From those pairs:

* **clock fit** -- host = a + b * robot + latency, latency >= 0. A least-squares line would put the mean
  latency into ``a`` and let its jitter tilt ``b``; the line under every point that is closest to them on
  average (the LP of Moon, Skelly & Towsley 1999: the lower convex hull's edge over the mean robot time) does
  not. What is left above it is each state's latency beyond the fastest one. The fastest one itself (a
  one-way trip) cannot be seen without a round trip, so a mapped robot time is the state's sample time *plus*
  that arm's minimum latency; the two arms' minima may differ by up to the larger of them, reported as
  ``min_latency_ms`` and part of the skew's stated bound.
* **response lag** -- each arm's measured TCP (FK of its joints, on the fitted host time) against its
  commanded TCP (on the host send time): the delay that best overlays them. It contains the controller's
  tracking lag, which is the same for two like arms; the *difference* between the arms is the execution skew.
  Measured over the first and the last half of the run, its change is the drift.
* **T_LR cost** -- a skew Δt between the hands moves T_LR by about v·|Δt| (v the faster hand's speed); for a
  readback that pairs the two arms' states by arrival time instead of the fitted clock, the latency jitter
  plays Δt. Both are *estimates* (the roadmap's red line: the laser locks one target at a time and cannot
  certify a dynamic T_LR).

Exit criterion (D2): start skew under one period, drift stated.
"""

# ruff: noqa: N803, N806  (matrix names: T follows the maths)
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tools.fr3 import fr3_kinematics as kin

#: below this much commanded TCP travel a lag is not observable
MIN_MOTION_M = 0.005
#: fewer state samples than this, no fit
MIN_SAMPLES = 10
#: libfranka's RobotState.time resolution
ROBOT_CLOCK_RESOLUTION_S = 0.001


@dataclass(frozen=True)
class ClockFit:
    """host_ns = host0_ns + (offset_s + rate * (robot_s - robot0_s)) * 1e9, the lower envelope of the states."""

    robot0_s: float
    host0_ns: int
    offset_s: float
    rate: float
    n: int
    span_s: float
    latency_ms: dict[str, float]

    @property
    def drift_ppm(self) -> float:
        return (self.rate - 1.0) * 1e6

    @property
    def drift_resolution_ppm(self) -> float:
        """What the robot clock's resolution leaves undetermined in the rate over this span."""
        return ROBOT_CLOCK_RESOLUTION_S / self.span_s * 1e6 if self.span_s > 0 else float("inf")

    def to_host_ns(self, robot_s: np.ndarray) -> np.ndarray:
        r = np.asarray(robot_s, dtype=np.float64) - self.robot0_s
        return self.host0_ns + (self.offset_s + self.rate * r) * 1e9

    def summary(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            drift_ppm=round(self.drift_ppm, 2),
            drift_resolution_ppm=round(self.drift_resolution_ppm, 2),
            span_s=round(self.span_s, 3),
        )
        return row


def _lower_hull(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) <= 0.0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def fit_clock(robot_s: np.ndarray, receive_ns: np.ndarray) -> ClockFit:
    """The lower-envelope line of (robot time, arrival time); see the module docstring."""
    robot_s = np.asarray(robot_s, dtype=np.float64)
    receive_ns = np.asarray(receive_ns, dtype=np.int64)
    if len(robot_s) != len(receive_ns):
        raise ValueError("robot_s and receive_ns differ in length")
    if len(robot_s) < MIN_SAMPLES:
        raise ValueError(f"{len(robot_s)} state samples; a clock fit needs {MIN_SAMPLES}")
    order = np.argsort(robot_s, kind="stable")
    robot0, host0 = float(robot_s[order[0]]), int(receive_ns[order[0]])
    x = robot_s[order] - robot0
    y = (receive_ns[order] - host0) / 1e9
    # equal robot times (a repeated state): only the earliest arrival can be on the envelope
    keep = np.ones(len(x), dtype=bool)
    keep[1:] = np.diff(x) > 0
    for i in np.flatnonzero(~keep):
        j = i - 1
        while not keep[j]:
            j -= 1
        y[j] = min(y[j], y[i])
    x, y = x[keep], y[keep]
    if x[-1] - x[0] <= 0:
        raise ValueError("the robot clock did not advance")
    hull = _lower_hull(x, y)
    xm = float(np.mean(robot_s - robot0))
    seg = 0
    while seg < len(hull) - 2 and x[hull[seg + 1]] < xm:
        seg += 1
    i, j = hull[seg], hull[seg + 1]
    rate = (y[j] - y[i]) / (x[j] - x[i])
    offset = y[i] - rate * x[i]
    latency = ((receive_ns - host0) / 1e9 - (offset + rate * (robot_s - robot0))) * 1e3
    latency = np.maximum(latency, 0.0)  # the envelope's own points, to rounding
    return ClockFit(
        robot0_s=robot0,
        host0_ns=host0,
        offset_s=float(offset),
        rate=float(rate),
        n=len(robot_s),
        span_s=float(x[-1] - x[0]),
        latency_ms={
            "above_min_p50": round(float(np.percentile(latency, 50)), 3),
            "above_min_p95": round(float(np.percentile(latency, 95)), 3),
            "above_min_max": round(float(latency.max()), 3),
        },
    )


def response_lag_s(
    t_cmd_s: np.ndarray,
    p_cmd: np.ndarray,
    t_meas_s: np.ndarray,
    p_meas: np.ndarray,
    *,
    max_lag_s: float = 0.25,
    step_s: float = 0.0002,
) -> float | None:
    """The delay d that best overlays p_meas(t) on p_cmd(t - d); None when the command barely moves."""
    t_cmd_s, t_meas_s = np.asarray(t_cmd_s, dtype=np.float64), np.asarray(t_meas_s, dtype=np.float64)
    p_cmd, p_meas = np.asarray(p_cmd, dtype=np.float64), np.asarray(p_meas, dtype=np.float64)
    if len(t_cmd_s) < 3 or len(t_meas_s) < 3:
        return None
    if np.ptp(p_cmd, axis=0).max() < MIN_MOTION_M:
        return None
    lags = np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s)
    cost = np.full(len(lags), np.inf)
    for n, d in enumerate(lags):
        t = t_meas_s - d
        inside = (t >= t_cmd_s[0]) & (t <= t_cmd_s[-1])
        if inside.sum() < max(3, len(t_meas_s) // 2):
            continue
        ref = np.stack([np.interp(t[inside], t_cmd_s, p_cmd[:, a]) for a in range(3)], axis=1)
        cost[n] = float(np.mean(np.sum((p_meas[inside] - ref) ** 2, axis=1)))
    best = int(np.argmin(cost))
    if not np.isfinite(cost[best]):
        return None
    if 0 < best < len(lags) - 1 and np.isfinite(cost[best - 1]) and np.isfinite(cost[best + 1]):
        c0, c1, c2 = cost[best - 1], cost[best], cost[best + 1]
        denom = c0 - 2 * c1 + c2
        if denom > 0:
            return float(lags[best] + 0.5 * step_s * (c0 - c2) / denom)
    return float(lags[best])


def _arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's samples on the host clock: command times/positions, measured times/positions, the fit."""
    cmd_ns = np.asarray(arm["command_ns"], dtype=np.int64)
    robot_s = np.asarray(arm["robot_s"], dtype=np.float64)
    receive_ns = np.asarray(arm["receive_ns"], dtype=np.int64)
    q = np.asarray(arm["q"], dtype=np.float64).reshape(-1, 7)
    fit = fit_clock(robot_s, receive_ns)
    tool = np.asarray(arm["tool"], dtype=np.float64)
    p_meas = (kin.NOMINAL.fk(q) @ tool)[:, :3, 3]
    p_cmd = np.asarray(arm["targets"], dtype=np.float64)[: len(cmd_ns), :3, 3]
    return {
        "fit": fit,
        "cmd_ns": cmd_ns,
        "p_cmd": p_cmd,
        "meas_ns": fit.to_host_ns(robot_s),
        "p_meas": p_meas,
        "latency_s": (receive_ns - fit.to_host_ns(robot_s)) / 1e9,
    }


def _lag(a: dict[str, Any], t_ref_ns: int, lo_ns: float, hi_ns: float) -> float | None:
    c = (a["cmd_ns"] >= lo_ns) & (a["cmd_ns"] <= hi_ns)
    m = (a["meas_ns"] >= lo_ns) & (a["meas_ns"] <= hi_ns)
    return response_lag_s(
        (a["cmd_ns"][c] - t_ref_ns) / 1e9, a["p_cmd"][c], (a["meas_ns"][m] - t_ref_ns) / 1e9, a["p_meas"][m]
    )


def analyse(
    arms: Mapping[str, Mapping[str, Any]], *, period_ns: int, T_right_left: np.ndarray
) -> dict[str, Any]:
    """D2's report for one dual run. ``arms`` = {"right": ..., "left": ...}, each with ``command_ns`` (host send
    times), ``targets`` (n,4,4 TCP in its base), ``tool``, and per measured state ``robot_s``, ``receive_ns``,
    ``q`` -- one state per command, read after it (``dual_fr3_replay_runtime.arm_main``)."""
    period_ms = period_ns / 1e6
    out: dict[str, Any] = {
        "label": "estimate: FK readback on fitted robot clocks; the laser cannot certify a dynamic T_LR",
        "period_ms": period_ms,
    }
    missing = [n for n in ("right", "left") if len(arms.get(n, {}).get("robot_s", [])) == 0]
    if missing:
        out["status"] = f"no robot clock from {', '.join(missing)}"
        return out
    short = [n for n in ("right", "left") if len(arms[n]["robot_s"]) < MIN_SAMPLES]
    if short:
        out["status"] = f"too few states from {', '.join(short)} (need {MIN_SAMPLES})"
        return out
    r, l = _arm(arms["right"]), _arm(arms["left"])  # noqa: E741
    t_ref = int(min(r["cmd_ns"][0], l["cmd_ns"][0]))
    end = float(min(r["cmd_ns"][-1], l["cmd_ns"][-1]))
    mid = (t_ref + end) / 2.0
    lag = {
        name: {
            "run": _lag(a, t_ref, t_ref, end + period_ns),
            "first_half": _lag(a, t_ref, t_ref, mid + period_ns),
            "last_half": _lag(a, t_ref, mid - period_ns, end + period_ns),
        }
        for name, a in (("right", r), ("left", l))
    }
    n_common = min(len(r["cmd_ns"]), len(l["cmd_ns"]))
    send_skew_ms = float(np.median((l["cmd_ns"][:n_common] - r["cmd_ns"][:n_common]) / 1e6))
    out["arms"] = {
        name: {
            "clock": a["fit"].summary(),
            "lag_ms": {k: None if v is None else round(v * 1e3, 3) for k, v in lag[name].items()},
        }
        for name, a in (("right", r), ("left", l))
    }
    out["send_skew_ms"] = round(send_skew_ms, 3)
    # robot timestamps paired with their offsets only (no rate) drift apart this much over the run
    span_s = (end - t_ref) / 1e9
    out["robot_clocks_diverge_ms"] = round(
        abs(r["fit"].drift_ppm - l["fit"].drift_ppm) * 1e-6 * span_s * 1e3, 4
    )
    # what the mapping cannot see: each arm's fastest one-way trip
    out["skew_bound_note"] = (
        "skew excludes the arms' unequal minimum one-way latency (unobservable without a round trip)"
    )
    if lag["right"]["run"] is None or lag["left"]["run"] is None:
        out["status"] = "the commanded TCP barely moves; no lag to measure"
        return out
    skew_ms = send_skew_ms + (lag["left"]["run"] - lag["right"]["run"]) * 1e3
    halves = [lag[n][h] for n in ("right", "left") for h in ("first_half", "last_half")]
    drift_ms = None
    if all(h is not None for h in halves):
        first = (lag["left"]["first_half"] - lag["right"]["first_half"]) * 1e3
        last = (lag["left"]["last_half"] - lag["right"]["last_half"]) * 1e3
        drift_ms = last - first
    out["start_skew_ms"] = round(skew_ms, 3)
    out["drift_ms_over_run"] = None if drift_ms is None else round(drift_ms, 3)

    # T_LR cost, on the frames both arms measured: an arm that faulted sent its last command unmeasured
    n_meas = min(len(r["meas_ns"]), len(l["meas_ns"]))
    p_r = r["p_meas"][:n_meas]
    T = np.asarray(T_right_left, dtype=np.float64)
    p_l = l["p_meas"][:n_meas] @ T[:3, :3].T + T[:3, 3]
    t_r = (r["meas_ns"][:n_meas] - t_ref) / 1e9
    t_l = (l["meas_ns"][:n_meas] - t_ref) / 1e9
    speed = np.maximum(
        np.linalg.norm(np.gradient(p_r, t_r, axis=0), axis=1),
        np.linalg.norm(np.gradient(p_l, t_l, axis=0), axis=1),
    )
    frac = np.clip((t_r - t_r[0]) / max(t_r[-1] - t_r[0], 1e-9), 0.0, 1.0)
    dt_s = (skew_ms + (0.0 if drift_ms is None else drift_ms * (frac - 0.5))) / 1e3
    skew_mm = speed * np.abs(dt_s) * 1e3
    pairing_mm = speed * np.abs(l["latency_s"][:n_meas] - r["latency_s"][:n_meas]) * 1e3
    out["t_lr_error_mm"] = {
        "from_skew_p95": round(float(np.percentile(skew_mm, 95)), 3),
        "from_skew_max": round(float(skew_mm.max()), 3),
        "pairing_by_arrival_p95": round(float(np.percentile(pairing_mm, 95)), 3),
        "hand_speed_p95_mm_s": round(float(np.percentile(speed, 95)) * 1e3, 1),
    }
    passed = abs(skew_ms) < period_ms
    out["status"] = "passed" if passed else "failed"
    out["verdict"] = (
        f"start skew {skew_ms:+.2f} ms {'<' if passed else '>='} one period ({period_ms:.2f} ms); "
        + ("drift not measurable" if drift_ms is None else f"drift {drift_ms:+.2f} ms over {span_s:.1f} s")
    )
    return out
