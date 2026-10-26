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
Two FR3s replaying one recording together: one process per arm, a coordinator, one stop for both.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, D1 (runtime) and D3 (separation gate), written against
fake arms first. The single-arm replay (``replay_cube_pose_in_robot_base.py``) streams Cartesian targets at
the replay rate, holds at frame 0 and starts on "go"; this keeps those semantics for two arms and adds what
two arms need:

* **one start** -- each arm moves to its frame 0 on its own and reports AT_START; on one "go" the
  coordinator writes a start time ``t0`` on CLOCK_MONOTONIC a little ahead, and both arms put frame k's
  command at ``t0 + k * period`` on that same clock. No message is in the timing path, so the two start
  within the jitter of the sleep, and they cannot drift apart: one clock, one schedule.
* **one stop** -- a stop word in shared memory. An arm reads it *immediately before every command*; an arm
  that faults (an exception from its driver, its own tracking-error gate, a lost coordinator) writes it
  first, before anything else. So the other arm sends at most the one command it may already have been
  past the check for: stopped within one period. The coordinator writes it for what an arm cannot report
  itself -- its process died, or it hangs (no heartbeat for ``stale_periods``) -- and for the gates.
* **separation gate (D3)** -- each arm publishes its measured joints every frame; the coordinator puts both
  arms in the right base (``T_right_left``) and stops both when two capsules come within the gate distance.
  The capsules are C3's (``dual_fr3_collision``, fitted round the MJCF meshes), so the gate checks what the
  retarget planned against; the distance is the placement's uncertainty alone (``derive_gate``), the arms'
  tracking error being in the measured joints already.
* **coordinator lost** -- the coordinator beats a heartbeat too; an arm that sees it stale stops itself.

Shared state is a small memory-mapped file (``ArmSync``): stdlib, attachable from any process that knows the
path, which is what the real arms need (each is its own process with its own 1 kHz libfranka loop).

Two drivers have the same calls -- ``move_to_start``, ``observe``, ``stamp``, ``command``, ``hold``. The fake
(``FakeFr3Driver``) is what the coordination, the gates and D2 are tested on, faults included. The real one
(``RealFr3Driver``, D1) is panda_py's joint position control: it takes only C3's joint paths (``--retarget``),
starts at the plan's joints, and is refused until both arms are in service in fr3_arms.json (B1/B2).
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import argparse
import contextlib
import json
import math
import mmap
import multiprocessing as mp
import os
import signal
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import (  # noqa: E402
    dual_fr3_clock as clock,
    dual_fr3_collision as col,
    fr3_arms,
    fr3_kinematics as kin,
)

SCHEMA = "dual_fr3_replay/v1"
ARMS = ("right", "left")

# ------------------------------------------------------------------ shared state ---

_MAGIC = 0x46523344  # "FR3D"
_HEADER_SLOTS = 8
_ARM_SLOTS = 16
_SLOT = struct.Struct("<q")
_DSLOT = struct.Struct("<d")

# arm states
INIT, AT_START, RUNNING, DONE, FAULTED = 0, 1, 2, 3, 4
STATE_NAMES = {INIT: "init", AT_START: "at_start", RUNNING: "running", DONE: "done", FAULTED: "faulted"}
# who stopped: an arm's index + 1, or one of these
STOP_BY_COORDINATOR = -1
STOP_BY_GATE = -2


def now_ns() -> int:
    return time.monotonic_ns()


class ArmSync:
    """The shared word block: start time, stop word, and per arm its state, heartbeat, commands and joints.

    Header: magic, n_arms, period_ns, t0_ns, stop_ns, stop_by, coordinator heartbeat, spare.
    Per arm: state, heartbeat_ns, frame, last_command_ns, commands, joints_ns, joints[7], spare.
    Every slot is an aligned 8-byte word written whole, and the stop word only ever goes from 0 to a time,
    so a reader never mistakes a half-written stop for no stop.
    """

    def __init__(self, path: Path | str, *, create: bool = False, n_arms: int = 2, period_ns: int = 0):
        self.path = Path(path)
        size = 8 * (_HEADER_SLOTS + _ARM_SLOTS * n_arms)
        if create:
            with self.path.open("wb") as handle:
                handle.write(b"\0" * size)
        self._file = self.path.open("r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0)
        if create:
            self._put(0, _MAGIC)
            self._put(1, n_arms)
            self._put(2, int(period_ns))
        elif self._get(0) != _MAGIC:
            raise ValueError(f"{self.path} is not an ArmSync block")
        self.n_arms = self._get(1)
        self.period_ns = self._get(2)

    def close(self) -> None:
        self._mm.close()
        self._file.close()

    def _get(self, slot: int) -> int:
        return _SLOT.unpack_from(self._mm, 8 * slot)[0]

    def _put(self, slot: int, value: int) -> None:
        _SLOT.pack_into(self._mm, 8 * slot, int(value))

    def _arm(self, arm: int, offset: int) -> int:
        return _HEADER_SLOTS + _ARM_SLOTS * arm + offset

    # start and stop
    @property
    def t0_ns(self) -> int:
        return self._get(3)

    def set_t0(self, t0_ns: int) -> None:
        self._put(3, t0_ns)

    @property
    def stop_ns(self) -> int:
        return self._get(4)

    @property
    def stop_by(self) -> int:
        return self._get(5)

    def stopped(self) -> bool:
        return self._get(4) != 0

    def request_stop(self, by: int) -> bool:
        """Write the stop word unless it is already set; True when this call set it."""
        if self._get(4):
            return False
        self._put(5, by)
        self._put(4, now_ns())
        return True

    def beat_coordinator(self) -> None:
        self._put(6, now_ns())

    @property
    def coordinator_beat_ns(self) -> int:
        return self._get(6)

    # per arm
    def state(self, arm: int) -> int:
        return self._get(self._arm(arm, 0))

    def set_state(self, arm: int, state: int) -> None:
        self._put(self._arm(arm, 0), state)

    def beat(self, arm: int) -> None:
        self._put(self._arm(arm, 1), now_ns())

    def heartbeat_ns(self, arm: int) -> int:
        return self._get(self._arm(arm, 1))

    def record_command(self, arm: int, frame: int, t_ns: int) -> None:
        self._put(self._arm(arm, 2), frame)
        self._put(self._arm(arm, 3), t_ns)
        self._put(self._arm(arm, 4), self._get(self._arm(arm, 4)) + 1)

    def commands(self, arm: int) -> int:
        return self._get(self._arm(arm, 4))

    def frame(self, arm: int) -> int:
        return self._get(self._arm(arm, 2))

    def publish_joints(self, arm: int, q: np.ndarray) -> None:
        base = 8 * self._arm(arm, 6)
        for i, value in enumerate(np.asarray(q, dtype=np.float64)[:7]):
            _DSLOT.pack_into(self._mm, base + 8 * i, float(value))
        self._put(self._arm(arm, 5), now_ns())

    def joints(self, arm: int) -> tuple[int, np.ndarray]:
        base = 8 * self._arm(arm, 6)
        q = np.array([_DSLOT.unpack_from(self._mm, base + 8 * i)[0] for i in range(7)])
        return self._get(self._arm(arm, 5)), q


# ------------------------------------------------------------------ geometry ---


def _rotvec(R: np.ndarray) -> np.ndarray:
    return kin._rotvec(np.asarray(R)[None])[0]


def _rotation_from_rotvec(w: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def pose_error(T_actual: np.ndarray, T_target: np.ndarray) -> tuple[float, float]:
    """(metres, degrees) between two poses."""
    dp = float(np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3]))
    dr = float(np.degrees(np.linalg.norm(_rotvec(T_target[:3, :3] @ T_actual[:3, :3].T))))
    return dp, dr


def incremental_command(
    T_current: np.ndarray, T_target: np.ndarray, max_step_m: float, max_step_deg: float
) -> np.ndarray:
    """The target, but no further from the current pose than one step -- the single-arm replay's
    ``_compute_incremental_command_pose``."""
    T = np.array(T_target, dtype=np.float64)
    dp = T_target[:3, 3] - T_current[:3, 3]
    n = float(np.linalg.norm(dp))
    if n > max_step_m:
        T[:3, 3] = T_current[:3, 3] + dp * (max_step_m / n)
    w = _rotvec(T_target[:3, :3] @ T_current[:3, :3].T)
    angle = float(np.linalg.norm(w))
    limit = math.radians(max_step_deg)
    if angle > limit:
        T[:3, :3] = _rotation_from_rotvec(w * (limit / angle)) @ T_current[:3, :3]
    return T


#: The online gate when nothing says better: the C3 floor. ``run`` callers pass ``derive_gate``'s figure.
DEFAULT_D_SAFE_M = 0.02
_MODEL: col.CapsuleModel | None = None


def capsule_model() -> col.CapsuleModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = col.load()
    return _MODEL


def derive_gate(placement_sigma_mm: float | None, *, unknown_sigma_mm: float = 20.0) -> dict[str, Any]:
    """The separation gate on measured joints: the placement's sigmas only. The retarget planned the arms
    ``d_safe`` apart, which adds both arms' execution tails; the gate reads where the arms *are*, so their
    tracking error is already in what it measures, and only the base-to-base uncertainty is left."""
    from tools.fr3 import dual_fr3_retarget as rg

    sigma = unknown_sigma_mm if placement_sigma_mm is None else float(placement_sigma_mm)
    gate = max(rg.K_PLACEMENT * sigma, rg.D_SAFE_FLOOR_MM)
    return {
        "d_gate_m": gate / 1e3,
        "placement_sigma_mm": sigma,
        "k_placement": rg.K_PLACEMENT,
        "floor_mm": rg.D_SAFE_FLOOR_MM,
    }


def arm_separation(
    q_right: np.ndarray, q_left: np.ndarray, T_right_left: np.ndarray
) -> tuple[float, tuple[str, str]]:
    """Smallest surface gap between the two arms, metres (negative: overlapping), and the capsules it is
    between -- C3's collision model (``dual_fr3_collision``), so the gate checks what the retarget planned
    against."""
    c = col.clearance(
        capsule_model(), np.atleast_2d(q_right), np.atleast_2d(q_left), np.asarray(T_right_left)
    )
    return float(c["arm_arm_m"][0]), tuple(c["arm_arm_pair"][0])


# ------------------------------------------------------------------ fake arm ---


#: ReplayRuntimeConfig.position_tolerance_m / orientation_tolerance_deg: frame 0 counts as reached within these.
#: the fake controller clock: a booted-a-while-ago offset, no drift, a LAN's latency, libfranka's 1 ms
DEFAULT_FAKE_CLOCK: dict[str, Any] = {
    "offset_s": 1000.0,
    "drift_ppm": 0.0,
    "latency_ms": [0.2, 0.1],
    "resolution_s": 0.001,
    "seed": 0,
    "delay_frames": 0,
}

START_POSITION_TOLERANCE_M = 0.004
START_ORIENTATION_TOLERANCE_DEG = 3.0


class ArmFaultError(RuntimeError):
    """An arm stopped itself: its gate tripped or its driver failed."""


@dataclass
class FakeFr3Driver:
    """A nominal-kinematics FR3 that follows commands with lag, and fails on request.

    Each ``command`` moves the joints a fraction ``gain`` of the way: toward the commanded joints when the plan
    has them (a joint-position controller), else a damped-least-squares step toward the commanded TCP pose (the
    single-arm replay's Cartesian streaming, the null space left to the IK); either way it trails a moving
    target as the real arm does. ``fault`` = ``{"kind", "at_frame"}``:
    ``raise`` (the driver throws, as libfranka on a reflex), ``exit`` (the process dies), ``hang`` (the call
    never returns), ``freeze`` (the arm stops following; its own tracking gate has to notice).

    It keeps a controller clock as libfranka's ``RobotState.time`` (D2): ``clock`` = ``{"offset_s",
    "drift_ppm", "latency_ms": [min, mean extra], "resolution_s", "seed", "delay_frames"}``; each ``observe``
    samples it one network latency (the minimum plus an exponential extra) before the state arrives, and
    ``stamp`` returns ``(robot time, arrival ns)`` of that state. ``delay_frames`` applies each command that
    many frames late: an arm that moves behind the shared schedule, the skew D2 has to find.
    """

    tool: list[list[float]]
    gain: float = 0.6
    fault: dict[str, Any] | None = None
    clock: dict[str, Any] | None = None
    q: np.ndarray = field(default_factory=lambda: kin.Q_READY.copy())
    frame: int = -1

    def __post_init__(self) -> None:
        self._T_tool = np.asarray(self.tool, dtype=np.float64)
        self._T_tool_inv = np.linalg.inv(self._T_tool)
        clock = {**DEFAULT_FAKE_CLOCK, **(self.clock or {})}
        self._clock = clock
        self._boot_ns = now_ns()
        self._rng = np.random.default_rng(clock["seed"])
        self._pending: list[tuple[np.ndarray, np.ndarray | None]] = []
        self._stamp: tuple[float, int] | None = None

    def move_to_start(self, T_tcp: np.ndarray, joints: list[float] | None = None) -> dict[str, Any]:
        """To the planned start joints when given (the single-arm replay's ``joint_start``), else by IK;
        reached when the TCP is then within the single-arm replay's position/orientation tolerance."""
        if joints is not None:
            self.q = np.clip(np.asarray(joints, dtype=np.float64), kin.Q_LOWER, kin.Q_UPPER)
        else:
            q, ok = kin.ik_retry(kin.NOMINAL, (np.asarray(T_tcp) @ self._T_tool_inv)[None], kin.Q_READY)
            if ok[0]:
                self.q = q[0]
        dp, dr = pose_error(self.observe()[0], np.asarray(T_tcp))
        reached = dp <= START_POSITION_TOLERANCE_M and dr <= START_ORIENTATION_TOLERANCE_DEG
        return {
            "reached": reached,
            "position_error_mm": dp * 1e3,
            "orientation_error_deg": dr,
            "joints": self.q.tolist(),
        }

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        c = self._clock
        lat_min, lat_extra = (x * 1e6 for x in c["latency_ms"])
        latency_ns = lat_min + (self._rng.exponential(lat_extra) if lat_extra > 0 else 0.0)
        arrived = now_ns()
        robot = c["offset_s"] + (arrived - latency_ns - self._boot_ns) / 1e9 * (1.0 + c["drift_ppm"] * 1e-6)
        self._stamp = (math.floor(robot / c["resolution_s"]) * c["resolution_s"], arrived)
        return kin.NOMINAL.fk(self.q)[0] @ self._T_tool, self.q.copy()

    def stamp(self) -> tuple[float, int] | None:
        """``(robot clock s, host arrival ns)`` of the last ``observe`` -- the real driver's is
        ``(state.time.toSec(), time.monotonic_ns() on its return)``."""
        return self._stamp

    def command(self, T_tcp: np.ndarray, frame: int, q_ref: np.ndarray | None = None) -> None:
        self.frame = frame
        fault = self.fault or {}
        if fault and frame >= int(fault.get("at_frame", 0)):
            kind = fault.get("kind")
            if kind == "raise":
                raise RuntimeError(f"injected fault at frame {frame}: libfranka reflex")
            if kind == "exit":
                os._exit(3)
            if kind == "hang":
                time.sleep(3600.0)
            if kind == "freeze":
                return
        delay = int(self._clock["delay_frames"])
        if delay:
            self._pending.append((np.asarray(T_tcp), None if q_ref is None else np.asarray(q_ref)))
            if len(self._pending) <= delay:
                return
            T_tcp, q_ref = self._pending.pop(0)
        if q_ref is not None:
            self.q = np.clip(self.q + self.gain * (np.asarray(q_ref) - self.q), kin.Q_LOWER, kin.Q_UPPER)
            return
        T_flange = np.asarray(T_tcp) @ self._T_tool_inv
        T, J = kin.NOMINAL.kinematics(self.q)
        e = np.concatenate([T_flange[:3, 3] - T[0, :3, 3], _rotvec(T_flange[:3, :3] @ T[0, :3, :3].T)])
        Jt = J[0].T
        dq = Jt @ np.linalg.solve(J[0] @ Jt + 1e-4 * np.eye(6), e)
        self.q = np.clip(self.q + self.gain * dq, kin.Q_LOWER, kin.Q_UPPER)

    def hold(self) -> None:
        pass


# ------------------------------------------------------------------ real arm ---

#: Moving to frame 0: panda_py's move_to_joint_position at this fraction of its speed (its default is 0.2).
REAL_START_SPEED_FACTOR = 0.1
#: The joint controller's set-point filter time constant, in periods: each frame's command is approached
#: over about a third of a period, so 20-50 Hz commands reach the 1 kHz torque loop as ramps, not steps.
REAL_FILTER_TAU_PERIODS = 1.0 / 3.0
#: After the last command, or a stop: the measured joints are held this long before the controller stops.
REAL_HOLD_S = 0.3
#: A state whose robot time has not moved for this long comes from a dead control loop: panda_py's
#: get_state is a cache, frozen whenever no controller runs (10-12).
REAL_STALE_STATE_S = 0.1


class RealFr3Driver:
    """One FR3 through panda_py: the fake's calls (``move_to_start``, ``observe``, ``stamp``, ``command``,
    ``hold``) on joint position control.

    ``move_to_start`` takes the plan's start joints -- a real arm never starts from an IK of its own -- with
    panda_py's ``move_to_joint_position``, then starts a ``JointPosition`` controller (panda_py's default
    gains, as fr3_pivot_auto's) that holds them. Each ``command`` sets the controller's target to the plan's
    joints for the frame, with the step from the previous frame as velocity feed-forward. Cartesian-only
    plans are refused: two arms kept apart by their elbows stay apart only if the elbows are commanded
    (``ArmPlan.joints``). ``observe`` reads the state the controller's 1 kHz loop last received and raises
    libfranka's error once that loop has died (a reflex), or when the state stops moving. ``hold`` holds the
    measured joints and stops the controller; it never raises, being what runs on the way out.

    ``stamp`` is ``(state.time, monotonic ns when get_state returned)``: the state is panda_py's copy of the
    last 1 kHz packet, so its arrival reads up to a millisecond late -- latency above the minimum, which
    D2's lower envelope leaves out.

    ``payload`` is the arm's fr3_arms.json declaration; when declared, Desk has to hold it
    (``fr3_arms.payload_check``) or the arm does not move. ``panda_py_module`` names what provides ``Panda``
    and ``controllers``: ``panda_py`` on the arm, a stand-in in the tests.
    """

    def __init__(
        self,
        *,
        robot_ip: str,
        tool: list[list[float]],
        period_s: float,
        arm: str = "",
        payload: Mapping[str, Any] | None = None,
        panda_py_module: str = "panda_py",
    ):
        import importlib

        module = importlib.import_module(panda_py_module)
        self._controllers = getattr(module, "controllers", None) or importlib.import_module(
            f"{panda_py_module}.controllers"
        )
        self.arm, self.robot_ip, self.payload = arm, robot_ip, payload
        self._T_tool = np.asarray(tool, dtype=np.float64)
        self._period_s = float(period_s)
        self._panda = module.Panda(robot_ip)
        self._ctrl = None
        self._last_ref: np.ndarray | None = None
        self._stamp: tuple[float, int] | None = None
        self._moved: tuple[float, int] | None = None  # (robot time, host ns) when it last advanced

    def _start_problems(self, q_now: np.ndarray, joints: np.ndarray, state: Any) -> list[str]:
        problems = []
        beyond = np.flatnonzero((q_now < kin._WALL_LOWER) | (q_now > kin._WALL_UPPER))
        if len(beyond):
            problems.append(
                f"joints {', '.join(f'j{j + 1}' for j in beyond)} are past panda_py's walls now; "
                "a controller started there aborts -- move the arm back in teaching mode"
            )
        # inside the walls but past the arm's own limits: the transport pose (j4 -3.050 on 10-26's new arm),
        # where the robot refuses motion; only Desk's guiding brings it in
        past_limits = np.flatnonzero((q_now < kin._FR3_LOWER) | (q_now > kin._FR3_UPPER))
        past_limits = [j for j in past_limits if j not in beyond]
        if past_limits:
            problems.append(
                f"joints {', '.join(f'j{j + 1}' for j in past_limits)} are past the FR3 joint limits now "
                "(the transport pose?); guide the arm in with Desk first"
            )
        outside = np.flatnonzero((joints < kin.Q_LOWER) | (joints > kin.Q_UPPER))
        if len(outside):
            problems.append(f"the start joints put {', '.join(f'j{j + 1}' for j in outside)} past the walls")
        if self.payload:
            ok, line = fr3_arms.payload_check(
                {"arm": self.arm, "payload": self.payload}, float(state.m_ee), list(state.F_x_Cee)
            )
            if not ok:
                problems.append(line)
        return problems

    def move_to_start(self, T_tcp: np.ndarray, joints: list[float] | None = None) -> dict[str, Any]:
        if joints is None:
            raise ArmFaultError("a real arm starts at the plan's joints, and this plan has none")
        target = np.asarray(joints, dtype=np.float64)
        state = self._panda.get_robot().read_once()
        q_now = np.asarray(state.q, dtype=np.float64)
        record: dict[str, Any] = {
            "robot_ip": self.robot_ip,
            "q_before": q_now.tolist(),
            "desk_m_ee_kg": float(state.m_ee),
            "desk_F_x_Cee_m": [float(v) for v in state.F_x_Cee],
        }
        problems = self._start_problems(q_now, target, state)
        if problems:
            raise ArmFaultError("not moving to frame 0: " + "; ".join(problems))
        moved = bool(self._panda.move_to_joint_position(target, speed_factor=REAL_START_SPEED_FACTOR))
        alpha = 1.0 - math.exp(-1e-3 / (REAL_FILTER_TAU_PERIODS * self._period_s))
        self._ctrl = self._controllers.JointPosition(filter_coeff=alpha)
        self._panda.start_controller(self._ctrl)  # holds where it is; the loop now refreshes get_state
        T, q = self.observe()
        self._ctrl.set_control(q, np.zeros(7))
        dp, dr = pose_error(T, np.asarray(T_tcp))
        record.update(
            reached=moved and dp <= START_POSITION_TOLERANCE_M and dr <= START_ORIENTATION_TOLERANCE_DEG,
            move_to_joint_position=moved,
            position_error_mm=dp * 1e3,
            orientation_error_deg=dr,
            joints=q.tolist(),
            filter_coeff=alpha,
        )
        return record

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        self._panda.raise_error()  # libfranka's exception, once the control loop has died on it
        state = self._panda.get_state()
        arrived = now_ns()
        robot_s = float(state.time.to_sec())
        if self._moved is None or robot_s != self._moved[0]:
            self._moved = (robot_s, arrived)
        elif self._ctrl is not None and arrived - self._moved[1] > REAL_STALE_STATE_S * 1e9:
            raise ArmFaultError(
                f"the robot state has not moved for {(arrived - self._moved[1]) / 1e6:.0f} ms: no control loop"
            )
        self._stamp = (robot_s, arrived)
        q = np.asarray(state.q, dtype=np.float64)
        return kin.NOMINAL.fk(q)[0] @ self._T_tool, q

    def stamp(self) -> tuple[float, int] | None:
        return self._stamp

    def command(self, T_tcp: np.ndarray, frame: int, q_ref: np.ndarray | None = None) -> None:
        del T_tcp, frame
        if q_ref is None:
            raise ArmFaultError("a real arm takes the plan's joints (C3's joint path), not TCP poses")
        if self._ctrl is None:
            raise ArmFaultError("command before move_to_start")
        q_ref = np.asarray(q_ref, dtype=np.float64)
        outside = np.flatnonzero((q_ref < kin.Q_LOWER) | (q_ref > kin.Q_UPPER))
        if len(outside):
            raise ArmFaultError(f"commanded {', '.join(f'j{j + 1}' for j in outside)} past the walls")
        dq = np.zeros(7) if self._last_ref is None else (q_ref - self._last_ref) / self._period_s
        self._ctrl.set_control(q_ref, dq)
        self._last_ref = q_ref

    def hold(self) -> None:
        if self._ctrl is None:
            return
        # the way out must not raise: the controller is stopped whatever the hold did
        with contextlib.suppress(Exception):
            self._ctrl.set_control(np.asarray(self._panda.get_state().q, dtype=np.float64), np.zeros(7))
            time.sleep(REAL_HOLD_S)
        with contextlib.suppress(Exception):
            self._panda.stop_controller()
        self._ctrl = None


def make_driver(spec: Mapping[str, Any]) -> FakeFr3Driver | RealFr3Driver:
    kind = spec.get("kind")
    if kind == "real":
        return RealFr3Driver(
            robot_ip=str(spec["robot_ip"]),
            tool=spec["tool"],
            period_s=float(spec["period_s"]),
            arm=str(spec.get("arm") or ""),
            payload=spec.get("payload"),
            panda_py_module=str(spec.get("panda_py_module") or "panda_py"),
        )
    if kind != "fake":
        raise ValueError(f"driver kind {kind!r}: 'fake' or 'real'")
    return FakeFr3Driver(
        tool=spec["tool"], gain=float(spec.get("gain", 0.6)), fault=spec.get("fault"), clock=spec.get("clock")
    )


def joint_path(targets: np.ndarray, tool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(q, ok)``: each frame's IK inside the joint walls (``fr3_kinematics.ik_retry``, every frame at once),
    and which frames converged -- the pre-check the single-arm replay gets from its MuJoCo preview. Frames are
    solved independently, so ``q`` may switch branch between frames; ``q[0]`` is the start the arm is moved
    to, and the arm follows the Cartesian stream from there."""
    T_flange = np.asarray(targets) @ np.linalg.inv(np.asarray(tool))
    return kin.ik_retry(kin.NOMINAL, T_flange, kin.Q_READY)


# ------------------------------------------------------------------ one arm ---


@dataclass(frozen=True)
class Settings:
    """Streaming and gates: the single-arm replay's (ReplayRuntimeConfig defaults; streaming as the Thor
    YAML: absolute poses, no incremental planning) and the dual-arm ones."""

    max_target_position_error_m: float = 0.05
    max_target_orientation_error_deg: float = 15.0
    max_command_position_error_m: float = 0.03
    max_command_orientation_error_deg: float = 8.0
    tracking_error_grace_frames: int = 10
    tracking_error_consecutive_frames: int = 5
    use_incremental_planning: bool = False
    max_translation_step_m: float = 0.01
    max_rotation_step_deg: float = 2.0
    d_safe_m: float = DEFAULT_D_SAFE_M
    #: an arm with no heartbeat for this many periods is hung
    stale_periods: float = 3.0
    #: an arm stops itself when the coordinator's heartbeat is this old
    coordinator_stale_s: float = 0.5


def _sleep_until(t_ns: int) -> None:
    while True:
        remaining = t_ns - now_ns()
        if remaining <= 0:
            return
        time.sleep(remaining / 1e9 if remaining > 2_000_000 else 0.0002)


def arm_main(
    index: int,
    sync_path: str,
    plan_path: str,
    driver_spec: dict[str, Any],
    settings: dict[str, Any],
    out_path: str,
) -> None:
    """One arm's process: move to frame 0, wait for t0, stream, stop on the word."""
    sync = ArmSync(sync_path)
    g = Settings(**settings)
    plan = np.load(plan_path)
    targets = plan["targets"]
    joints = plan["joints"] if "joints" in plan.files else None
    period_ns = sync.period_ns
    log: dict[str, Any] = {
        "arm_index": index,
        "command_ns": [],
        "frames": [],
        # D2: per command, the state read after it -- its robot clock, its arrival, its joints
        "state_clock": [],
        "state_q": [],
        "outcome": "",
        "reason": "",
    }
    driver = None
    try:
        driver = make_driver({**driver_spec, "period_s": period_ns / 1e9})
        start = driver.move_to_start(targets[0], driver_spec.get("start_joints"))
        log["start"] = start
        if not start["reached"]:
            raise ArmFaultError("frame 0 is not reachable")
        _, q = driver.observe()
        sync.publish_joints(index, q)
        sync.beat(index)
        sync.set_state(index, AT_START)
        while not sync.t0_ns:
            if sync.stopped():
                log["outcome"], log["reason"] = "stopped_before_start", "the stop word was set"
                return
            if now_ns() - sync.coordinator_beat_ns > g.coordinator_stale_s * 1e9:
                log["outcome"], log["reason"] = "stopped_before_start", "coordinator heartbeat lost"
                return
            sync.beat(index)
            time.sleep(0.001)
        t0 = sync.t0_ns
        while now_ns() < t0 - 2_000_000:  # keep beating through the start lead, or it reads as a hang
            sync.beat(index)
            time.sleep(0.001)
        _sleep_until(t0)
        sync.beat(index)
        sync.set_state(index, RUNNING)
        streak = 0
        for k in range(len(targets)):
            if sync.stopped():  # immediately before the command: the one-period guarantee rests on this
                log["outcome"] = "stopped"
                log["reason"] = f"stop word at frame {k} (by {sync.stop_by})"
                break
            T_now, _ = driver.observe()
            T_cmd = (
                incremental_command(T_now, targets[k], g.max_translation_step_m, g.max_rotation_step_deg)
                if g.use_incremental_planning
                else targets[k]
            )
            driver.command(T_cmd, k, None if joints is None else joints[k])
            t_cmd = now_ns()
            sync.record_command(index, k, t_cmd)
            log["command_ns"].append(t_cmd)
            _sleep_until(t0 + (k + 1) * period_ns)
            T_act, q = driver.observe()
            sync.publish_joints(index, q)
            sync.beat(index)
            stamp = driver.stamp()
            if stamp is not None:
                log["state_clock"].append([stamp[0], stamp[1]])
                log["state_q"].append(q.tolist())
            e_target = pose_error(T_act, targets[k])
            e_command = pose_error(T_act, T_cmd)
            log["frames"].append([k, *e_target, *e_command])
            exceeded = (
                e_target[0] > g.max_target_position_error_m
                or e_target[1] > g.max_target_orientation_error_deg
                or e_command[0] > g.max_command_position_error_m
                or e_command[1] > g.max_command_orientation_error_deg
            )
            if k >= g.tracking_error_grace_frames:
                streak = streak + 1 if exceeded else 0
                if streak >= g.tracking_error_consecutive_frames:
                    raise ArmFaultError(
                        f"tracking error over the limits for {streak} frames at frame {k}: "
                        f"target {e_target[0] * 1e3:.1f} mm/{e_target[1]:.1f} deg, "
                        f"command {e_command[0] * 1e3:.1f} mm/{e_command[1]:.1f} deg"
                    )
            if now_ns() - sync.coordinator_beat_ns > g.coordinator_stale_s * 1e9:
                sync.request_stop(index + 1)
                log["outcome"], log["reason"] = "stopped", "coordinator heartbeat lost"
                break
        else:
            log["outcome"] = "completed"
    except Exception as exc:  # any failure of this arm stops both, first thing
        sync.request_stop(index + 1)
        log["outcome"], log["reason"] = "fault", f"{type(exc).__name__}: {exc}"
    finally:
        # done streaming before the hold: a real arm's hold settles for REAL_HOLD_S, which would read as a hang
        sync.set_state(index, FAULTED if log["outcome"] == "fault" else DONE)
        if driver is not None:
            driver.hold()
        Path(out_path).write_text(json.dumps(log), encoding="utf-8")
        sync.close()


# ------------------------------------------------------------------ coordinator ---


@dataclass
class ArmPlan:
    """One arm's part of a dual replay: its TCP targets in its own base, and how to drive it."""

    name: str
    targets: np.ndarray  # (n, 4, 4) T_base_tcp
    driver: dict[str, Any]
    #: (n, 7) the joints to command with them, when the plan is a joint path (C3). Two arms kept apart by
    #: their elbows -- which is how the retarget clears them -- stay apart only if the elbows are commanded:
    #: Cartesian streaming leaves the null space to the arm's IK, and on 10-26's fake run that walked the
    #: arms from the planned 75 mm to 54 mm.
    joints: np.ndarray | None = None


def _stdin_go() -> bool:
    print("DUAL_REPLAY_AT_TRAJECTORY_START (type go or abort)", flush=True)
    return sys.stdin.readline().strip().lower() == "go"


def run(
    plans: list[ArmPlan],
    *,
    fps: float,
    T_right_left: np.ndarray,
    run_dir: Path,
    settings: Settings | None = None,
    go: Callable[[], bool] = _stdin_go,
    start_lead_s: float = 0.25,
    at_start_timeout_s: float = 60.0,
    timelines: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the arms together and return the report (also written to ``run_dir/dual_replay_report.json``).

    ``plans`` are right first, then left; ``T_right_left`` is the left base in the right base (B3's edge,
    inverted), used by the separation gate only. ``timelines`` (``t_raw``, ``t_ref``: per command, the
    recorded and the reference time) are saved with the executed one, each arm's command times from t0, in
    ``run_dir/timelines.npz``.
    """
    gates = settings or Settings()
    if [p.name for p in plans] != list(ARMS):
        raise ValueError(f"plans must be {ARMS}, in that order")
    if len({len(p.targets) for p in plans}) != 1:
        raise ValueError("the arms' plans have different lengths; a dual replay streams one frame clock")
    run_dir.mkdir(parents=True, exist_ok=True)
    period_ns = int(round(1e9 / fps))
    sync = ArmSync(run_dir / "arm_sync.bin", create=True, n_arms=len(plans), period_ns=period_ns)
    sync.beat_coordinator()
    ctx = mp.get_context("spawn")
    procs, outs = [], []
    for i, plan in enumerate(plans):
        plan_path = run_dir / f"plan_{plan.name}.npz"
        arrays = {"targets": np.asarray(plan.targets, dtype=np.float64)}
        if plan.joints is not None:
            arrays["joints"] = np.asarray(plan.joints, dtype=np.float64)
        np.savez(plan_path, **arrays)
        out = run_dir / f"arm_{plan.name}.json"
        outs.append(out)
        proc = ctx.Process(
            target=arm_main,
            args=(i, str(sync.path), str(plan_path), dict(plan.driver), asdict(gates), str(out)),
            name=f"fr3_{plan.name}",
        )
        proc.start()
        procs.append(proc)

    events: list[dict[str, Any]] = []
    # Until t0 the coordinator may sit in go() for minutes; a thread beats for it so the arms can still
    # tell a dead coordinator from a slow operator. From t0 on the watch loop beats, so a hung watch (no
    # separation gate) also reads as lost.
    beating = threading.Event()
    beating.set()

    def beat() -> None:
        while beating.is_set():
            sync.beat_coordinator()
            time.sleep(0.01)

    beater = threading.Thread(target=beat, daemon=True)
    beater.start()

    def stop(by: int, reason: str) -> None:
        if sync.request_stop(by):
            events.append({"t_ns": sync.stop_ns, "by": by, "reason": reason})

    # The operator's abort (the GUI sends SIGTERM once the arms run): one stop word for both, as for a fault.
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(
                signum, lambda n, _f: stop(STOP_BY_COORDINATOR, f"operator abort ({signal.Signals(n).name})")
            )

    # 1. both at frame 0
    deadline = time.monotonic() + at_start_timeout_s
    while not sync.stopped():
        sync.beat_coordinator()
        if all(sync.state(i) == AT_START for i in range(len(plans))):
            break
        dead = [plans[i].name for i, p in enumerate(procs) if not p.is_alive() and sync.state(i) < AT_START]
        if dead:
            stop(STOP_BY_COORDINATOR, f"{', '.join(dead)} exited before reaching frame 0")
        elif time.monotonic() > deadline:
            stop(STOP_BY_COORDINATOR, f"not at frame 0 within {at_start_timeout_s:.0f} s")
        time.sleep(0.002)
    start_gap = None
    if not sync.stopped():
        start_gap, _ = arm_separation(sync.joints(0)[1], sync.joints(1)[1], T_right_left)
        if start_gap < gates.d_safe_m:
            stop(
                STOP_BY_GATE,
                f"arms {start_gap * 1e3:.0f} mm apart at frame 0 (d_safe {gates.d_safe_m * 1e3:.0f} mm)",
            )
    # 2. one go, one t0
    if not sync.stopped():
        if go():
            sync.beat_coordinator()
            sync.set_t0(now_ns() + int(start_lead_s * 1e9))
        else:
            stop(STOP_BY_COORDINATOR, "operator did not say go")
    beating.clear()
    beater.join()
    # 3. watch
    min_gap, min_pair, last_seen = math.inf, (-1, -1), (0, 0)
    stale_ns = gates.stale_periods * period_ns
    while any(p.is_alive() for p in procs):
        sync.beat_coordinator()
        t = now_ns()
        for i, p in enumerate(procs):
            if not p.is_alive() and sync.state(i) not in (DONE, FAULTED):
                stop(
                    STOP_BY_COORDINATOR, f"{plans[i].name} process exited (rc {p.exitcode}) without stopping"
                )
            elif sync.state(i) == RUNNING and t - sync.heartbeat_ns(i) > stale_ns:
                stop(
                    STOP_BY_COORDINATOR,
                    f"{plans[i].name} hung: no heartbeat for {(t - sync.heartbeat_ns(i)) / 1e6:.0f} ms",
                )
        (t_r, q_r), (t_l, q_l) = sync.joints(0), sync.joints(1)
        if (t_r, t_l) != last_seen and sync.t0_ns:
            last_seen = (t_r, t_l)
            gap, pair = arm_separation(q_r, q_l, T_right_left)
            if gap < min_gap:
                min_gap, min_pair = gap, pair
            if gap < gates.d_safe_m:
                stop(
                    STOP_BY_GATE,
                    f"arms {gap * 1e3:.0f} mm apart (capsules {pair}, d_safe {gates.d_safe_m * 1e3:.0f} mm)",
                )
        if sync.stopped() and t - sync.stop_ns > 1e9:
            for p in procs:  # a hung arm never returns; its robot holds its last command
                if p.is_alive():
                    p.terminate()
        time.sleep(0.001)
    for p in procs:
        p.join()
    for signum, handler in previous.items():
        signal.signal(signum, handler)

    arms = {}
    for plan, out in zip(plans, outs, strict=True):
        log = (
            json.loads(out.read_text(encoding="utf-8"))
            if out.is_file()
            else {"outcome": "no_log", "command_ns": [], "frames": []}
        )
        arms[plan.name] = log
    report = _report(sync, plans, arms, events, period_ns, start_gap, min_gap, min_pair, gates)
    report["clock"] = _clock_report(plans, arms, period_ns, T_right_left, run_dir)
    if timelines is not None:
        t0 = sync.t0_ns
        executed = {
            f"t_executed_{plan.name}": (
                np.asarray(arms[plan.name].get("command_ns") or [], dtype=np.float64) - t0
            )
            / 1e9
            for plan in plans
        }
        np.savez(run_dir / "timelines.npz", **{k: np.asarray(v) for k, v in timelines.items()}, **executed)
        report["timelines"] = (
            "timelines.npz: t_raw, t_ref (per planned command), t_executed_<arm> (per sent command)"
        )
    (run_dir / "dual_replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sync.close()
    return report


def _clock_report(
    plans: list[ArmPlan], arms: Mapping[str, Any], period_ns: int, T_right_left: np.ndarray, run_dir: Path
) -> dict[str, Any]:
    """D2 (``dual_fr3_clock``) on the arms' state stamps; the stamps themselves go to clock_samples.npz."""
    samples, inputs = {}, {}
    for plan in plans:
        log = arms[plan.name]
        stamps = np.asarray(log.get("state_clock") or np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)
        receive_ns = np.asarray([int(s[1]) for s in log.get("state_clock") or []], dtype=np.int64)
        q = np.asarray(log.get("state_q") or np.zeros((0, 7)), dtype=np.float64).reshape(-1, 7)
        samples.update({f"robot_s_{plan.name}": stamps[:, 0], f"receive_ns_{plan.name}": receive_ns,
                        f"q_{plan.name}": q})  # fmt: skip
        inputs[plan.name] = {
            "command_ns": log.get("command_ns") or [],
            "targets": plan.targets,
            "tool": plan.driver["tool"],
            "robot_s": stamps[:, 0].tolist(),
            "receive_ns": receive_ns,
            "q": q,
        }
    np.savez(run_dir / "clock_samples.npz", **samples)
    return clock.analyse(inputs, period_ns=period_ns, T_right_left=T_right_left)


def _report(sync, plans, arms, events, period_ns, start_gap, min_gap, min_pair, gates) -> dict[str, Any]:
    stop_ns, stop_by = sync.stop_ns, sync.stop_by
    if stop_ns and not events and stop_by > 0:
        events.append(
            {
                "t_ns": stop_ns,
                "by": stop_by,
                "reason": f"{plans[stop_by - 1].name}: {arms[plans[stop_by - 1].name].get('reason', '')}",
            }
        )
    rows = {}
    for i, plan in enumerate(plans):
        log = arms[plan.name]
        cmds = log.get("command_ns", [])
        errors = np.array(log.get("frames") or np.zeros((0, 5)))
        row = {
            "outcome": log.get("outcome"),
            "reason": log.get("reason"),
            "frames_commanded": len(cmds),
            "frames_planned": len(plan.targets),
            "start": log.get("start"),
        }
        if len(errors):
            row["target_error_mm_p95"] = round(float(np.percentile(errors[:, 1], 95)) * 1e3, 2)
        if stop_ns and stop_by != i + 1:
            after = [c for c in cmds if c > stop_ns]
            row["commands_after_stop"] = len(after)
            row["last_command_after_stop_ms"] = round((max(after) - stop_ns) / 1e6, 2) if after else None
        rows[plan.name] = row
    firsts = [arms[p.name]["command_ns"][0] for p in plans if arms[p.name].get("command_ns")]
    stopped_together = all(r.get("commands_after_stop", 0) <= 1 for r in rows.values())
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_ms": period_ns / 1e6,
        "settings": asdict(gates),
        "separation_model": f"dual_fr3_collision capsules (model {capsule_model().fingerprint[:12]})",
        "start_skew_ms": round(abs(firsts[0] - firsts[1]) / 1e6, 3) if len(firsts) == 2 else None,
        "start_gap_mm": None if start_gap is None else round(start_gap * 1e3, 1),
        "min_gap_mm": None if not math.isfinite(min_gap) else round(min_gap * 1e3, 1),
        "min_gap_capsules": list(min_pair),
        "stop": None if not stop_ns else {"by": stop_by, "events": events},
        "stopped_together": stopped_together,
        "arms": rows,
    }


# ------------------------------------------------------------------ inputs ---


def _matrix(pose7: list[float]) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = pose7
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = (x, y, z)
    return T


def hand_targets(track: Mapping[int, Mapping[str, Any]], stride: int = 1) -> np.ndarray:
    """(n, 4, 4) poses of one hand's ``bimanual_qc.read_hand`` episode, every ``stride``-th frame; a frame the
    tracker lost holds the last seen pose (the replay holds too), and leading gaps take the first seen."""
    frames = sorted(track)
    seen = [f for f in frames if track[f]["ok"]]
    if not seen:
        raise ValueError("the hand is never seen in this episode")
    last = track[seen[0]]
    out = []
    for f in frames[::stride]:
        if track[f]["ok"]:
            last = track[f]
        out.append(_matrix([*last["p"], *last["q"]]))
    return np.stack(out)


#: What an unmeasured placement is taken to be good to: a base put down by tape measure.
ASSUMED_PLACEMENT_SIGMA_MM = 20.0


def assumed_placement_graph(
    graph: Mapping[str, Any], T_right_left: np.ndarray, note: str, description: Mapping[str, Any]
) -> dict[str, Any]:
    """``graph`` plus an edge placing the base that is not the dataset's (``description``'s other root) where
    it is assumed to be -- for fake runs and offline checks only; the edge's method says so, and no real replay
    can be fed a graph that was not read from disk."""
    roots = {arm: str(description["arms"][arm]["root"]) for arm in ARMS}
    reference = str(description["dataset_frame"])
    if sorted(roots.values()).count(reference) != 1:
        raise ValueError(f"one arm's base must be the dataset's {reference}, not {roots}")
    placed = next(arm for arm in ARMS if roots[arm] != reference)
    # the edge maps the placed base into the reference: T_right_left when the left is placed, else its inverse
    T_ref_placed = np.asarray(T_right_left) if placed == "left" else np.linalg.inv(T_right_left)
    out = json.loads(json.dumps(graph))
    out.setdefault("nodes", []).append(
        {"world_frame_id": roots[placed], "reason": "robot_base", "parent_world_frame_id": None}
    )
    out.setdefault("edges", []).append(
        {
            "from_world_frame_id": roots[placed],
            "to_world_frame_id": reference,
            "T_to_from": T_ref_placed.tolist(),
            "method": f"assumed_placement: {note}; not measured",
            "sigma_mm": ASSUMED_PLACEMENT_SIGMA_MM,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return out


def fake_plans_from_session(
    session_dir: Path,
    episode: int,
    *,
    spacing_m: float,
    stride: int,
    faults: Mapping[str, dict[str, Any]] | None = None,
    embodiment_path: Path = REPO / "tools" / "thor" / "gmsl2" / "embodiments" / "fr3_dual.json",
) -> tuple[list[ArmPlan], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Plans for a fake run of one episode of a two-hand session (e.g. a ``pseudo_bimanual`` pair): each hand
    to its arm through the embodiment, the left base assumed at ``(0, spacing, 0)`` of the right."""
    from tools.thor.gmsl2 import bimanual_qc, embodiment, world_provenance as wp

    description = embodiment.load(embodiment_path)
    T_right_left = np.eye(4)
    T_right_left[1, 3] = spacing_m
    graph = assumed_placement_graph(
        wp.read_world_graph(REPO),
        T_right_left,
        f"left base {spacing_m} m along +y of the right, parallel",
        description,
    )
    tool = kin.urdf_fixed_transform(description["arms"]["right"]["tcp_frame"])
    plans, edges = [], []
    for arm in ARMS:
        hand = next(h for h in description["arms"][arm]["hands"])
        track = bimanual_qc.read_hand(
            session_dir / "derived" / bimanual_qc.SIDECAR / f"state_action.{hand}.csv"
        )[episode]
        T_root_dataset, used, reason = embodiment.arm_transform(description, graph, wp.FR3_BASE_WORLD_ID, arm)
        if T_root_dataset is None:
            raise ValueError(f"{arm}: {reason}")
        edges += used
        targets = np.asarray(T_root_dataset) @ hand_targets(track, stride)
        spec = {"kind": "fake", "tool": tool.tolist(), "fault": (faults or {}).get(arm)}
        plans.append(ArmPlan(arm, targets, spec))
    n = min(len(p.targets) for p in plans)
    feasibility = {}
    for p in plans:
        p.targets = p.targets[:n]
        q, ok = joint_path(p.targets, tool)
        feasibility[p.name] = {
            "frames": n,
            "reachable": int(ok.sum()),
            "first_unreachable": None if ok.all() else int(np.argmin(ok)),
        }
        if ok[0]:
            p.driver["start_joints"] = q[0].tolist()
    return plans, T_right_left, tool, edges, feasibility


def plans_from_retarget(
    session_dir: Path,
    episode: int,
    mode: str,
    *,
    faults: Mapping[str, dict[str, Any]] | None = None,
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[ArmPlan], np.ndarray, float, dict[str, Any], dict[str, Any]]:
    """Plans that stream C3's reference path for ``episode`` (``dual_fr3_executability``): per arm the TCP
    of the retargeted joints on the command clock, started at their first joints. Refused when C4 says the
    episode is not executable -- the reasons are in the error. ``arms`` (each arm's ``fr3_arms.resolve``)
    makes them plans for the real arms; without it they drive the fake ones."""
    from tools.fr3 import dual_fr3_executability as ex

    reference = ex.plans_from_retarget(session_dir, episode, mode)
    report = reference["report"]
    if not report.get("executable"):
        raise ValueError(
            f"episode {episode} ({mode}) is not executable: " + "; ".join(report.get("reasons") or [])
        )
    plans = []
    for arm in ARMS:
        spec = {"tool": reference["tool"].tolist(), "start_joints": reference["q"][arm][0].tolist()}
        if arms is None:
            spec.update(kind="fake", fault=(faults or {}).get(arm))
        else:
            entry = arms[arm]
            spec.update(kind="real", arm=arm, robot_ip=str(entry["robot_ip"]), payload=entry.get("payload"))
        plans.append(ArmPlan(arm, reference["targets"][arm], spec, joints=reference["q"][arm]))
    timelines = {"t_raw": reference["t_raw"], "t_ref": reference["t_ref"]}
    record = {
        "retarget_report": str(ex.retarget_paths(Path(session_dir), episode, mode)[0]),
        "mode": report["mode"],
        "scene_fingerprint": report["scene_fingerprint"],
        "placement_sigma_mm": report["d_safe"]["placement_sigma_mm"],
    }
    return plans, reference["T_right_left"], reference["fps"], timelines, record


def real_arm_entries(registry: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Both arms as fr3_arms.json names them (B2: IP, payload; nothing falls back to a default arm);
    ``fr3_arms.ArmConfigError`` while one is not in service (B1) or both name one IP."""
    registry = registry or fr3_arms.load()
    entries = {arm: fr3_arms.resolve(registry, arm) for arm in ARMS}
    if entries["right"]["robot_ip"] == entries["left"]["robot_ip"]:
        raise fr3_arms.ArmConfigError(f"both arms are {entries['right']['robot_ip']} in fr3_arms.json")
    return entries


def real_backend_refusal(registry: Mapping[str, Any] | None = None, *, panda_py_module: str = "panda_py") -> str:
    """Why the two real FR3s cannot be driven; "" once they can: both in service in fr3_arms.json, and
    panda_py importable here (on Thor it is in ~/Code/infer/.venv-fr3, which the gateway runs this with)."""
    import importlib

    try:
        real_arm_entries(registry)
    except fr3_arms.ArmConfigError as exc:
        return str(exc)
    try:
        importlib.import_module(panda_py_module)
    except ImportError as exc:
        return f"{panda_py_module} does not import in {sys.executable} ({exc}); run with the FR3 venv"
    return ""


#: What a real plan may ask of a joint: the retarget's own velocity_scale of the FR3 limits, and a little.
REAL_VELOCITY_TOLERANCE = 1.05


def real_plan_problems(plans: list[ArmPlan], fps: float) -> list[str]:
    """Why these plans may not go to the real arms: a plan without joints (they take C3's joint path), or a
    joint faster between two frames than the retarget allows (``dual_fr3_retarget``: ``velocity_scale`` of
    the FR3 limits) -- a plan made for another rate, or stale."""
    from tools.fr3 import dual_fr3_retarget as rg

    limit = rg.FR3_VELOCITY_LIMITS * rg.Settings().velocity_scale * REAL_VELOCITY_TOLERANCE
    problems = []
    for plan in plans:
        if plan.joints is None:
            problems.append(f"{plan.name}: no joint path (the real arms replay C3's retarget: --retarget)")
            continue
        speed = np.abs(np.diff(np.asarray(plan.joints), axis=0)) * fps
        if len(speed):
            ratio = speed.max(axis=0) / limit
            j = int(ratio.argmax())
            if ratio[j] > 1.0:
                frame = int(speed[:, j].argmax())
                problems.append(
                    f"{plan.name}: j{j + 1} moves {speed[frame, j]:.2f} rad/s at frame {frame} at {fps:g} Hz "
                    f"(the retarget allows {limit[j] / REAL_VELOCITY_TOLERANCE:.2f})"
                )
    return problems


def _parse_fault(text: str) -> tuple[str, dict[str, Any]]:
    arm, rest = text.split(":", 1)
    kind, at = rest.split("@", 1)
    if arm not in ARMS or kind not in ("raise", "exit", "hang", "freeze"):
        raise argparse.ArgumentTypeError(f"--fault takes ARM:raise|exit|hang|freeze@FRAME, not {text!r}")
    return arm, {"kind": kind, "at_frame": int(at)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dual-FR3 replay of a two-hand session (D1/D3)")
    parser.add_argument("session", type=Path, help="a session directory with left and right sidecars")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--spacing", type=float, default=0.8, help="assumed left base offset along +y, m")
    parser.add_argument("--stride", type=int, default=1, help="take every n-th frame")
    parser.add_argument(
        "--fps", type=float, default=20.0, help="replay rate (the Thor YAML: 20, 3x slow for 60 fps data)"
    )
    parser.add_argument("--fault", type=_parse_fault, action="append", default=[], help="ARM:KIND@FRAME")
    parser.add_argument(
        "--retarget",
        choices=("absolute", "relative"),
        default=None,
        help="stream C3's reference path of this mode (run dual_fr3_executability first) instead of the raw labels",
    )
    parser.add_argument("--d-safe", type=float, default=None, help="gate distance, m (default: derive_gate)")
    parser.add_argument("--yes", action="store_true", help="say go without asking")
    parser.add_argument(
        "--backend",
        choices=("fake", "real"),
        default="fake",
        help="fake arms (the coordination, gates and timing on this machine) or the two FR3s of fr3_arms.json",
    )
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "fr3" / "dual_fr3_replay")
    args = parser.parse_args(argv)
    arms = None
    if args.backend == "real":
        if not args.retarget:
            print("refused: the real arms replay C3's joint path; give --retarget absolute|relative")
            return 2
        if args.fault:
            print("refused: --fault is for the fake arms")
            return 2
        refusal = real_backend_refusal()
        if refusal:
            print(f"refused: {refusal}")
            return 2
        arms = real_arm_entries()
    timelines, record, feasibility, edges = None, {}, None, []
    if args.retarget:
        try:
            plans, T_right_left, fps, timelines, record = plans_from_retarget(
                args.session, args.episode, args.retarget, faults=dict(args.fault), arms=arms
            )
        except (OSError, ValueError) as exc:
            print(f"refused: {exc}")
            return 2
        if arms is not None:
            problems = real_plan_problems(plans, fps)
            if problems:
                print("refused: " + "; ".join(problems))
                return 2
        sigma = record["placement_sigma_mm"]
    else:
        plans, T_right_left, _, edges, feasibility = fake_plans_from_session(
            args.session, args.episode, spacing_m=args.spacing, stride=args.stride, faults=dict(args.fault)
        )
        fps, sigma = args.fps, ASSUMED_PLACEMENT_SIGMA_MM
        print(json.dumps({"ik_precheck": feasibility}))
        blocked = [arm for arm, row in feasibility.items() if row["first_unreachable"] == 0]
        if blocked:
            print(f"refused: frame 0 is out of reach for the {', '.join(blocked)} arm; nothing was started")
            return 2
    gate = derive_gate(sigma)
    d_gate = args.d_safe if args.d_safe is not None else gate["d_gate_m"]
    run_dir = args.out / datetime.now(UTC).strftime(f"{args.backend}_%Y%m%dT%H%M%SZ")
    report = run(
        plans,
        fps=fps,
        T_right_left=T_right_left,
        run_dir=run_dir,
        settings=Settings(d_safe_m=d_gate),
        go=(lambda: True) if args.yes else _stdin_go,
        timelines=timelines,
    )
    report["gate"] = gate if args.d_safe is None else {"d_gate_m": d_gate, "given": True}
    report["edges"] = edges
    report["ik_precheck"] = feasibility
    report["retarget"] = record or None
    report["backend"] = args.backend
    (run_dir / "dual_replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("start_skew_ms", "start_gap_mm", "min_gap_mm", "stop", "stopped_together", "arms")
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")
    print(f"DUAL_REPLAY_REPORT {run_dir / 'dual_replay_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
