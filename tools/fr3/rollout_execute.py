# ruff: noqa: N803, N806  -- T, R, q: the usual names of transforms, rotations and joint vectors
"""Execution of a relative-TCP policy on one FR3: the Rollout page's last stage. Attended, staged, gated.

A person at the e-stop starts every run, from state A (start pose, gripper open, cubes as recorded):

    stage 1  one chunk: the first n_action_steps of one prediction (16 steps, 0.53 s), then hold
    stage 2  closed loop, at most 5 s, the gripper not commanded
    stage 3  closed loop with the gripper, at most 15 s

It is started from the Rollout page by a person who confirms that the arm will move and a hand is on the
e-stop; the gateway refuses it in a sandbox, without the recorder, before the checkpoint's preview passed its
Replay check, and while anything else holds the arm's FCI.

**One instant per observation.** The cameras reach the policy through the recorder's preview frame bus
(about 5 Hz; a frame is 60-260 ms old when read), while the arm's FK is read at the loop's rate. Training
saw each image with the hand's pose of the same instant, so every input is taken at the image's capture
time (its SOF ``sensor_timestamp_ns``, CLOCK_MONOTONIC on Thor -- the instant the training labels are stamped
with, exposure offset 0.0 as in production): the TCP history and the anchor ``T_obs`` are the FK buffer
interpolated there, and step k of the chunk is due at ``t_img + k*dt``. A chunk that arrives late is
joined at its first step still in the future; the steps whose time has passed are not executed. The
policy's speed is kept (it reads speed from its history, and a slowed arm feeds it a history training never
had) except where a joint would have to turn faster than ``Limits.step_joint_rad`` per step: that step is
given more time (``retime``; the path is unchanged), up to ``step_stretch_max``. The first chunk is the
exception: the arm stood still up to that moment, so its anchor is the moment it is accepted (step 1 is then
33 ms away, not already late).

**The command is smooth.** Between steps the joints follow a monotone cubic (PCHIP: continuous speed, never
past a validated step), at rest at the first chunk's anchor and at the last step a chunk may run to; a newer
chunk takes over over ``BLEND_S``. 11-19: a linear command's speed jumped at every 33 ms step (the policy's
steps are uneven), the joint impedance rang at 25-40 Hz and O_F_ext read +-10 N without contact (stage 3
stopped at 15.2 N with the jaw 200 mm from anything).

**Every step is checked before it is commanded** (``assess``): IK inside the joint walls, the training
workspace, the capsule model's clearance to carts/table/cameras (rollout_relative_tcp.PlanContext), at most
STEP limits per 33 ms, and where the demonstrations went (rollout_distribution: near a training pose, no
farther from the anchor than training's actions went by that step). A chunk is taken only when the n_action_steps it may have to carry alone are all
executable; a failing later step only ends the chunk there. When the cursor reaches a step that is not
executable, or a newer chunk does not come, the arm holds. It never clips or detours.

**Stops** (any of them: hold where it is, the gripper left as it is): see :class:`Limits`.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.fr3 import relative_tcp as rt  # noqa: E402

RECORD_SCHEMA = "fr3_rollout_execute/1"
TIME_CONVENTION = (
    "observation instant = the frame's SOF sensor_timestamp_ns (CLOCK_MONOTONIC), the instant training labels "
    "are stamped with (exposure offset 0.0); TCP history and T_obs = FK buffer interpolated there"
)


@dataclass(frozen=True)
class Stage:
    number: int
    chunks: int  # 0 = closed loop until max_s
    max_s: float
    gripper: bool
    label: str


STAGES = {
    1: Stage(1, 1, 3.0, False, "one chunk (n_action_steps), then hold"),
    2: Stage(2, 0, 5.0, False, "closed loop, at most 5 s, gripper not commanded"),
    3: Stage(3, 0, 15.0, True, "closed loop with the gripper, at most 15 s"),
}


@dataclass(frozen=True)
class Limits:
    """Why to stop. Each value says where it comes from."""

    # the first target against where the arm stands (the old runtime's default)
    first_target_mm: float = 30.0
    first_target_deg: float = 10.0
    # one 33 ms step. execute() replaces these with the checkpoint's own (step_limits: what one step of its
    # training data did, p99.5 x1.5, under STEP_*_CEILING); these defaults only stand when there is none.
    # 11-18, stack_act_wrist_v1: demonstration steps p50 5.9 / p99 22 / max 30 mm (~0.9 m/s, not tracking
    # jumps), 12% of them above 15 mm -- the old "peak ~450 mm/s" was wrong
    step_mm: float = 15.0
    step_deg: float = 5.0
    # near the carts/table/cameras the demonstrations were slow: clearance < 100 mm, every step <= 13.4 mm
    # (11-18, the same data through IK and the capsule model); fast only in free space (> 120 mm)
    near_clearance_m: float = 0.10
    near_step_mm: float = 15.0
    # one 33 ms step in j1-j4: 1.5 rad/s, about 60% of their limit (2.62 rad/s); j5-j7 scale with their own
    # limits (JOINT_SCALE: 2.0 / 1.6 / 2.0 x). 11-18 demonstrations through IK: j5/j7 above 0.05 rad in ~6%
    # of steps, within their own 60% in ~2%; j1/j3 above in ~6% too, but at up to 4 rad/s -- past the
    # hardware's 2.62 -- an IK redundancy matter, not a limit to loosen.
    # 11-19: it is the commanded joint SPEED (retime: a step that needs more is given more time, the path
    # unchanged), no longer a refusal by itself. Stage 2 was refused at j1 0.063 rad/step: the policy asked
    # 0.5 m/s (in its training, step1 36 mm) where q0 (j1 +54, j3 -72, j6 ~ pi) leaves no elbow redundancy
    # (the null space is j5/j7), so j1 had to turn 32 deg for an 11.5 deg azimuth change.
    step_joint_rad: float = 0.05
    # ... but a step needing more than this many times its 33 ms is refused: an IK branch change or a wild
    # prediction, not speed (11-19 runs needed at most 1.53 within what a chunk must carry, 1.98 at step 32)
    step_stretch_max: float = 2.0
    # what a step does beyond moving the TCP (its joint motion minus J+ of its own TCP motion): a re-posture,
    # i.e. an IK branch change, refused whatever the speed. 11-19 runs: at most 0.008 rad per step; the
    # IK pulled to the range middle (11-18) re-postured 0.69 rad in step 1
    reposture_rad: float = 0.05
    # a newer chunk against the command it replaces, at the moment of the switch. Two predictions of the
    # policy 13 steps apart (the takeover interval) differ there by themselves: on stack_act_wrist_v1 030000's
    # training episodes p50 8.1, p95 22.7, p99.5 33.8 mm (rollout_chunk_consistency.py, 11-19; held-out
    # episodes p50 31 mm); online 6.6-20.3 mm. 15 mm refused about one normal takeover in seven. p99.5 x1.5
    # (as the step limits); the joints: 50 mm at ~0.6 m is ~0.08 rad, and 0.062 rad of null-space drift was
    # seen with 4 mm -- an IK branch change is ~1 rad, and the reposture check catches it within a chunk.
    switch_mm: float = 50.0
    switch_joint_rad: float = 0.10
    # the TCP (FK of the measured joints) against FK of the command
    tracking_mm: float = 20.0
    # |O_F_ext - its reading at the start|, force part, averaged over FORCE_WINDOW_S ...
    # 11-19: O_F_ext is FR3's model estimate; in motion it swings +-5-10 N within 20-40 ms with the TCP's
    # acceleration (60-75% of x/z explained by a, v) and stopped five runs at 12-15.5 N raw with nothing
    # touched; averaged over 50 ms the same runs peaked at 4.3-8.8 N. A contact lasts: the average sees it
    force_n: float = 15.0
    # ... and raw, any single tick: an impact is not waited for
    force_peak_n: float = 25.0
    # a measured joint speed no plan here asks for (oscillation, runaway); j5-j7 x JOINT_SCALE
    # 11-19 (a person's decision, 11:40 run): 1.6 sat 7% over the commanded cap (1.5) and the joint impedance's
    # catch-up runs up to 10% (0.15 rad/s) over its command, which stopped stage 2 at j3 1.64; 1.9 is 73% of
    # the hardware's 2.62, 27% over anything commanded
    joint_speed_rad_s: float = 1.9
    # a measured joint within this of panda_py's wall (IK keeps 3 deg; Monitor of fr3_pivot_auto keeps 1)
    wall_deg: float = 1.0
    # inputs: the preview frame bus publishes ~5 Hz (age 60-263 ms measured 11-18); execute() takes 0.15 s on
    # the inference bus (rollout_relative_tcp.frame_age_limit_s); BOX_LIVE staleS 6-90 ms
    frame_age_s: float = 0.40
    box_stale_s: float = 0.30
    predict_s: float = 0.10
    # the loop itself: a stall leaves the arm holding a command that the chunk's clock has passed
    loop_stall_s: float = 0.05
    # the arm must stand still to start
    start_speed_rad_s: float = 0.02


FR3_VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # rad/s, dual_fr3_retarget
JOINT_SCALE = FR3_VELOCITY_LIMITS / FR3_VELOCITY_LIMITS.min()  # the joint limits below are for j1-j4
STEP_MM_CEILING = 40.0  # 1.2 m/s, whatever the data says: well under the FR3's Cartesian limit (1.7 m/s)
STEP_DEG_CEILING = 8.0  # 240 deg/s
LOOP_HZ = 200.0
FORCE_WINDOW_S = 0.05  # Limits.force_n is held against O_F_ext averaged over this long
HOLD_S = 0.5  # after a stop the arm brakes to rest (stop_motion), is held there this long, then let go
FORCE_SILENT_S = 0.1  # O_F_ext_hat_K exactly 0 this long while running: the force estimate is gone, stop
GRIPPER_CLOSE_BELOW_M = 0.050  # replay's grasp: below this the jaw is sent to 0 (squeeze, as in the demos)
GRIPPER_OPEN_ABOVE_M = 0.060  # hysteresis: back open only above this
GRIPPER_LEAD_S = 0.20  # the jaw lags its command; the width of 0.2 s ahead is what is sent
GRIPPER_MIN_INTERVAL_S = 0.30
JOINT_STIFFNESS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])  # fr3_pivot_auto
JOINT_DAMPING = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ geometry ---


def _rotvec(R: np.ndarray) -> np.ndarray:
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(angle)
    if s < 1e-9:
        if angle < 1e-6:
            return 0.5 * v
        # near pi: the axis from the symmetric part
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, int(np.argmax(w))] * angle
    return v * (angle / (2.0 * s))


def _expmap(w: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def interpolate_pose(Ta: np.ndarray, Tb: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = Ta[:3, 3] + s * (Tb[:3, 3] - Ta[:3, 3])
    T[:3, :3] = Ta[:3, :3] @ _expmap(s * _rotvec(Ta[:3, :3].T @ Tb[:3, :3]))
    return T


def pose_gap(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(mm, deg) between two poses."""
    return (
        float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1e3),
        float(np.degrees(np.linalg.norm(_rotvec(Ta[:3, :3].T @ Tb[:3, :3])))),
    )


class PoseHistory:
    """The TCP's FK, as the loop reads it, for the last few seconds; read back at any past instant."""

    def __init__(self, span_s: float = 3.0):
        self.span_s = span_s
        self.t: deque[float] = deque()
        self.T: deque[np.ndarray] = deque()
        self.q: deque[np.ndarray] = deque()
        self.lock = threading.Lock()

    def add(self, t: float, T: np.ndarray, q: np.ndarray | None = None) -> None:
        with self.lock:
            self.t.append(t)
            self.T.append(np.array(T, dtype=np.float64))
            self.q.append(np.full(7, np.nan) if q is None else np.array(q, dtype=np.float64))
            while self.t and self.t[0] < t - self.span_s:
                self.t.popleft()
                self.T.popleft()
                self.q.popleft()

    def q_at(self, t: float) -> np.ndarray:
        """The measured joints at t (linear; the newest when t is past it)."""
        with self.lock:
            ts, qs = list(self.t), list(self.q)
        return _q_at(ts, qs, t)

    def at(self, t: float, *, slack_s: float = 0.02) -> np.ndarray:
        with self.lock:
            ts, Ts = list(self.t), list(self.T)
        return _pose_at(ts, Ts, t, slack_s)

    def poses(self, t_obs: float, offsets_s: list[float]) -> np.ndarray:
        """(len(offsets), 4, 4): the TCP at t_obs + each offset (offsets <= 0, the last one 0)."""
        return np.stack([self.at(t_obs + o) for o in offsets_s])


def _q_at(ts, qs, t: float) -> np.ndarray:
    if not len(ts):
        raise RuntimeError("no FK read yet")
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)) if len(ts) > 1 else 0
    if len(ts) == 1 or t >= ts[-1]:
        return np.array(qs[-1], dtype=np.float64)
    if t <= ts[0]:
        return np.array(qs[0], dtype=np.float64)
    s = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return qs[i - 1] + s * (qs[i] - qs[i - 1])


def _pose_at(ts, Ts, t: float, slack_s: float) -> np.ndarray:
    if not len(ts):
        raise RuntimeError("no FK read yet")
    if t < ts[0] - 1e-9:
        raise RuntimeError(f"the FK buffer starts {ts[0] - t:.3f} s after the instant asked for")
    if t > ts[-1]:
        if t - ts[-1] > slack_s:
            raise RuntimeError(f"no FK within {slack_s * 1e3:.0f} ms of the observation (last {t - ts[-1]:.3f} s before)")
        return np.array(Ts[-1], dtype=np.float64)
    i = int(np.searchsorted(ts, t))
    if i == 0 or ts[i] == t:
        return np.array(Ts[i], dtype=np.float64)
    s = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return interpolate_pose(Ts[i - 1], Ts[i], s)


class SharedPoseHistory:
    """PoseHistory in shared memory: the control loop writes it, the predictor's process reads it.

    The writer never waits (a sequence counter, odd while an entry is written; a reader copies, and copies
    again when the counter moved), so a reader in another process can never hold the loop up."""

    CAPACITY = 2048  # entries: more than span_s at LOOP_HZ

    def __init__(self, span_s: float = 3.0, *, name: str | None = None, capacity: int = CAPACITY):
        from multiprocessing import shared_memory

        self.span_s, self.capacity = float(span_s), int(capacity)
        self.owner = name is None
        size = 16 + 8 * self.capacity * (1 + 16 + 7)
        # a spawned reader shares the owner's resource tracker: its attach registers the same name again (a
        # set) and the owner's unlink unregisters it; unregistering here would drop the owner's entry
        self.shm = shared_memory.SharedMemory(name=name, create=self.owner, size=size if self.owner else 0)
        buf, n = self.shm.buf, self.capacity
        self._head = np.ndarray((2,), dtype=np.int64, buffer=buf, offset=0)  # sequence, entries written
        self._t = np.ndarray((n,), dtype=np.float64, buffer=buf, offset=16)
        self._T = np.ndarray((n, 4, 4), dtype=np.float64, buffer=buf, offset=16 + 8 * n)
        self._q = np.ndarray((n, 7), dtype=np.float64, buffer=buf, offset=16 + 8 * n * 17)
        if self.owner:
            self._head[:] = 0

    def spec(self) -> tuple[str, float, int]:
        return self.shm.name, self.span_s, self.capacity

    @classmethod
    def attach(cls, spec: tuple[str, float, int]) -> SharedPoseHistory:
        name, span_s, capacity = spec
        return cls(span_s, name=name, capacity=capacity)

    def add(self, t: float, T: np.ndarray, q: np.ndarray | None = None) -> None:
        i = int(self._head[1]) % self.capacity
        self._head[0] += 1  # odd: an entry is being written
        self._T[i] = T
        self._q[i] = np.nan if q is None else q
        self._t[i] = t
        self._head[1] += 1
        self._head[0] += 1

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(1000):
            seq = int(self._head[0])
            if seq % 2 == 0:
                n = int(self._head[1])
                t, T, q = self._t.copy(), self._T.copy(), self._q.copy()
                if int(self._head[0]) == seq:
                    break
            time.sleep(0)
        else:
            raise RuntimeError("the FK buffer kept changing while it was read")
        m = min(n, self.capacity)
        idx = np.arange(n - m, n) % self.capacity
        t, T, q = t[idx], T[idx], q[idx]
        if m:
            keep = t >= t[-1] - self.span_s
            t, T, q = t[keep], T[keep], q[keep]
        return t, T, q

    def q_at(self, t: float) -> np.ndarray:
        ts, _, qs = self._arrays()
        return _q_at(ts, qs, t)

    def at(self, t: float, *, slack_s: float = 0.02) -> np.ndarray:
        ts, Ts, _ = self._arrays()
        return _pose_at(ts, Ts, t, slack_s)

    def poses(self, t_obs: float, offsets_s: list[float]) -> np.ndarray:
        ts, Ts, _ = self._arrays()
        return np.stack([_pose_at(ts, Ts, t_obs + o, 0.02) for o in offsets_s])

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.shm.close()
        if self.owner:
            with contextlib.suppress(Exception):
                self.shm.unlink()


# --------------------------------------------------------------------- chunks ---


def smooth_steps(actions: np.ndarray) -> np.ndarray:
    """The predicted poses (relative to the anchor) through [1, 2, 1] / 4 over the steps; the width unchanged.

    11-19: ACT's chunks zigzag -- from about step 12 every other step is 20-27% longer (the step-to-step
    change of velocity flips sign 9 times in 10), a 15 Hz ripple the demonstrations do not have (at the
    training's stride the flips are random, 0.51). Commanded, it shakes the arm at 15 Hz and FR3's O_F_ext
    swings +-10 N with it. The filter removes exactly that period (zero gain at 2 steps), passes the slow
    motion with no lag (symmetric), and moves a step by a fraction of the ripple (~1 mm at 0.3 m/s). The
    anchor (identity) is the step before step 1; the step after the last is extrapolated linearly, so the
    end is not pulled back. Everything after it (IK, every check) sees the filtered steps."""
    a = np.array(actions, dtype=np.float64)
    if len(a) < 2:
        return a
    pose = a[:, : rt.POSE_DIM]
    anchor = rt.encode(np.eye(4)[None])[0]
    before = np.vstack([anchor, pose[:-1]])
    after = np.vstack([pose[1:], 2.0 * pose[-1] - pose[-2]])
    a[:, : rt.POSE_DIM] = (before + 2.0 * pose + after) / 4.0
    return a


def joint_speed_limit(limits: Limits, dt: float) -> np.ndarray:
    """(7,) rad/s: what a command may ask of each joint (``step_joint_rad`` per step, JOINT_SCALE)."""
    return limits.step_joint_rad * JOINT_SCALE / dt


def hermite_tangents(q: np.ndarray, h: np.ndarray, *, still_start: bool, end: int) -> np.ndarray:
    """(n, 7) knot velocities (rad/s) of a monotone cubic (PCHIP, Fritsch-Butland) through ``q`` (n, 7) with
    segment durations ``h`` (n-1,). Between two knots no joint leaves their range (no overshoot past a
    validated step) and a knot's velocity never exceeds its neighbouring secants. Zero at knot 0 when the arm
    stood still there, and at ``end``: the arm comes to rest on the last step it may run to."""
    n = len(q)
    d = np.diff(q, axis=0) / h[:, None]
    m = np.zeros_like(q)
    if n > 2:
        h0, h1 = h[:-1, None], h[1:, None]
        w1, w2 = 2.0 * h1 + h0, h1 + 2.0 * h0
        d0, d1 = d[:-1], d[1:]
        same = d0 * d1 > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / d0 + w2 / d1)
        m[1:-1] = np.where(same, hm, 0.0)
    if n > 1:
        m[0] = 0.0 if still_start else d[0]
        m[-1] = d[-1]
    if 0 < end < n:
        m[end] = 0.0
    return m


_U = np.linspace(0.0, 1.0, 17)


def _segment_peak_speeds(q: np.ndarray, h: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(n-1, 7) the largest |velocity| of each joint inside each Hermite segment (sampled, 17 points)."""
    u = _U[None, :, None]
    p0, p1 = q[:-1, None], q[1:, None]
    hh = h[:, None, None]
    m0, m1 = m[:-1, None] * hh, m[1:, None] * hh
    dp = (6 * u**2 - 6 * u) * p0 + (3 * u**2 - 4 * u + 1) * m0 + (-6 * u**2 + 6 * u) * p1 + (3 * u**2 - 2 * u) * m1
    return np.abs(dp / hh).max(axis=1)


#: FR3 joint acceleration limits (rad/s^2, Franka's datasheet), half of which the stop on the last step may use
FR3_ACCELERATION_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
STOP_ACCELERATION = 0.5 * FR3_ACCELERATION_LIMITS  # also the start of a first chunk, from rest
BRAKE_SHARE = 0.4  # the speed profile plans with this share of it: the cubic between steps adds up to ~2x
STEP_DT = 1.0 / 30.0  # a policy step when nothing else says (fps / frame_stride of every checkpoint so far)


def turn_speed_cap(q: np.ndarray) -> np.ndarray:
    """(n,) the fastest each knot may be passed, in steps/s, for the joints' change of direction there to stay
    within STOP_ACCELERATION: along a fixed path a joint's acceleration is q''·ṡ² + q'·s̈, and q'' (rad/step²,
    the second difference of the steps) is the part a speed cap alone never bounds (inf where not judged).

    11-19 (execute_20261119_160814): the policy's path took j2 through 0, where j1 and j3 are collinear; the IK
    turned j1 from +0.85 to -1.16 rad/s within 70 ms (29 rad/s², twice the FR3's rating) under every speed
    limit. The arm overshot the command to 2.05 rad/s. Commands of normal runs stay within ~0.5x the rating."""
    n = len(q)
    cap = np.full(n, np.inf)
    if n >= 3:
        d2 = np.abs(q[2:] - 2.0 * q[1:-1] + q[:-2])
        cap[1:-1] = np.sqrt((STOP_ACCELERATION / np.maximum(d2, 1e-12)).min(axis=1))
    return cap


def stop_motion(q0: np.ndarray, v0: np.ndarray, accel: np.ndarray = FR3_ACCELERATION_LIMITS):
    """(duration, at(t) -> (q, dq)): every joint brakes from ``v0`` to rest together, along the current direction
    in joint space, the joint that needs it longest at its rated deceleration; then holds where it stopped.

    11-19 (execute_20261119_160814): a stop held the measured joints with a zero speed while j1/j3 turned at
    2 rad/s; the impedance let the arm run 0.15 rad past and threw it back to 2.58 rad/s, and the robot's own
    velocity reflex aborted. A stop is a braking trajectory, not a step in the commanded speed."""
    q0, v0 = np.asarray(q0, dtype=np.float64), np.asarray(v0, dtype=np.float64)
    T = float(np.max(np.abs(v0) / accel))

    def at(t: float) -> tuple[np.ndarray, np.ndarray]:
        if T <= 0.0 or t >= T:
            return q0 + 0.5 * v0 * T, np.zeros(7)
        return q0 + v0 * (t - 0.5 * t * t / T), v0 * (1.0 - t / T)

    return T, at


def retime(q: np.ndarray, dt: float, limits: Limits, *, still_start: bool, end: int) -> np.ndarray:
    """(n,) each step's time from the anchor: k*dt, except where a joint would turn faster than
    ``joint_speed_limit`` -- that segment takes longer. The path (every validated step) is unchanged; only
    when the arm gets there. 11-19: refusing such a chunk stopped stage 2 where the policy asked 0.5 m/s, and
    a linear command's speed jumps at every step rang the joint impedance into a false 15 N contact.

    The arm also brakes to rest on ``end`` and, from rest, starts within BRAKE_SHARE x STOP_ACCELERATION:
    a speed profile along the steps (w_k, steps/s at each step; a forward pass from a still start, a backward
    pass from the end), so braking spreads over as many steps as it needs. 11-19: a policy's step 1 is 1-2 mm
    off the anchor even standing still, done in 33 ms from rest it shook the arm (O_F_ext 4-12 N at the start
    of every run); braking only in the last step put 30-60 rad/s^2 into the one before it. The policy's own
    accelerations in between are kept."""
    n = len(q)
    if n < 2:
        return np.zeros(n)
    # a takeover's blend may add BLEND_SPEED_SHARE of the cap on top of a chunk's own speed (11:40: chunk at
    # the cap + blend = 1.64 rad/s): the chunk keeps that share free, so the command never exceeds the cap
    vmax = joint_speed_limit(limits, dt) * (1.0 - BLEND_SPEED_SHARE)
    dq = np.abs(np.diff(q, axis=0))  # (n-1, 7) per segment
    seg_cap = np.minimum(1.0 / dt, (vmax / np.maximum(dq, 1e-12)).min(axis=1))  # steps/s within each segment
    alpha = (BRAKE_SHARE * STOP_ACCELERATION / np.maximum(dq, 1e-12)).min(axis=1)  # steps/s^2 in each segment
    w = np.empty(n)
    w[0], w[-1] = seg_cap[0], seg_cap[-1]
    w[1:-1] = np.minimum(seg_cap[:-1], seg_cap[1:])
    w = np.minimum(w, turn_speed_cap(q))  # a sharp change of direction is passed slowly (the passes ramp to it)
    last = end if 0 < end < n else n - 1
    if 0 < end < n:
        w[end] = 0.0
        for k in range(end - 1, -1, -1):
            w[k] = min(w[k], math.sqrt(w[k + 1] ** 2 + 2.0 * alpha[k]))
    if still_start:
        w[0] = 0.0
    for k in range(last):  # from a still start, and out of every slow knot: within the acceleration
        w[k + 1] = min(w[k + 1], math.sqrt(w[k] ** 2 + 2.0 * alpha[k]))
    h = np.empty(n - 1)
    for k in range(n - 1):
        mean = (w[k] + w[k + 1]) / 2.0  # never above seg_cap[k]: both ends are capped by it
        if k >= last:  # past the end: never run, the plain cap
            h[k] = max(dt, 1.0 / seg_cap[k])
        elif mean > 0:
            h[k] = max(dt, 1.0 / mean)
        else:  # from rest to rest within one step
            h[k] = 2.0 / math.sqrt(alpha[k])
    for _ in range(12):  # the cubic between steps can exceed a secant: stretch those segments until it does not
        over = (_segment_peak_speeds(q, h, hermite_tangents(q, h, still_start=still_start, end=end)) / vmax).max(axis=1)
        if over.max() <= 1.0 + 1e-6:
            break
        h = np.where(over > 1.0, h * over * 1.001, h)
    return np.concatenate([[0.0], np.cumsum(h)])


@dataclass
class Chunk:
    """One prediction made executable: step k (0 = the anchor) is due at ``t0 + offsets[k]`` -- ``k*dt``
    unless a joint needed longer (``retime``, set when the chunk is accepted)."""

    index: int
    t_obs: float  # the observation instant (the image's)
    t0: float  # the anchor's time: t_obs, or the acceptance for the first chunk (the arm stood still)
    dt: float
    T: np.ndarray  # (H+1, 4, 4) TCP: the anchor, then the targets (arm base)
    q: np.ndarray  # (H+1, 7): the anchor's joints, then each target's IK
    ok: np.ndarray  # (H+1,) executable; ok[0] True; once False, False to the end
    why: list[str]  # (H+1,) why a step is not executable, "" otherwise
    widths: np.ndarray  # (H,) predicted gripper width at steps 1..H
    k0: int = 0  # first step still in the future when it was accepted
    end: int = 0  # the last step it may run to (stage 1: n_action_steps)
    offsets: np.ndarray | None = None  # (H+1,) s from t0 per step; None = k*dt
    still_start: bool = False  # the arm stood still at the anchor (the first chunk)
    _m: np.ndarray | None = field(default=None, repr=False)  # knot velocities for (offsets, end)

    def __post_init__(self) -> None:
        if self.offsets is None:
            self.offsets = self.dt * np.arange(len(self.q), dtype=np.float64)

    @property
    def horizon(self) -> int:
        return len(self.q) - 1

    def prepare(self, limits: Limits, end: int) -> None:
        """Fix the run's last step and the steps' times (retime) for that end."""
        self.end = end
        self.offsets = retime(self.q, self.dt, limits, still_start=self.still_start, end=end)
        self._m = None

    def stretch(self) -> np.ndarray:
        """(H,) each step's duration over dt (1 = the policy's speed)."""
        return np.diff(self.offsets) / self.dt

    def cursor(self, t: float) -> float:
        """Where in the chunk time t is, in steps (fractional); beyond the knots at dt per step."""
        tau = t - self.t0
        off = self.offsets
        if tau <= 0.0:
            return tau / self.dt
        if tau >= off[-1]:
            return self.horizon + (tau - off[-1]) / self.dt
        k = int(np.searchsorted(off, tau, side="right")) - 1
        return k + (tau - off[k]) / (off[k + 1] - off[k])

    def time_of(self, cursor: float) -> float:
        k = int(min(max(math.floor(cursor), 0), self.horizon - 1))
        return self.t0 + self.offsets[k] + (cursor - k) * (self.offsets[k + 1] - self.offsets[k])

    def command(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(q, dq, cursor) at time t on the monotone cubic through the steps, clamped to [0, end]; at rest at
        ``end`` (and at the anchor of a first chunk)."""
        c = min(max(self.cursor(t), 0.0), float(self.end))
        if self.end == 0:
            return self.q[0].copy(), np.zeros(7), c
        if self._m is None:
            self._m = hermite_tangents(self.q, np.diff(self.offsets), still_start=self.still_start, end=self.end)
        k = min(int(math.floor(c)), self.end - 1)
        u = c - k
        h = self.offsets[k + 1] - self.offsets[k]
        p0, p1, m0, m1 = self.q[k], self.q[k + 1], self._m[k] * h, self._m[k + 1] * h
        u2, u3 = u * u, u * u * u
        q = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
        if c >= self.end:
            return q, np.zeros(7), c
        dq = ((6 * u2 - 6 * u) * p0 + (3 * u2 - 4 * u + 1) * m0 + (-6 * u2 + 6 * u) * p1 + (3 * u2 - 2 * u) * m1) / h
        return q, dq, c

    def width_at(self, cursor: float) -> float:
        k = int(min(max(round(cursor), 1), self.horizon))
        return float(self.widths[k - 1])


#: a newer chunk takes over from the running command over this long, not in one tick: the two differ by up
#: to switch_mm / switch_joint_rad at the switch, and in their speed
BLEND_S = 0.15
#: ... longer when the two differ more: the blend's own term (the decaying gap) adds at most this share of a
#: joint's speed limit (11-19 11:40: a 0.15 s blend over a ~14 mrad gap put j1 at 1.64 rad/s, cap 1.5)
BLEND_SPEED_SHARE = 0.1
BLEND_MAX_S = 1.0  # a switch_joint_rad gap within BLEND_SPEED_SHARE: 1.5 x 0.10 / (0.1 x 1.5 rad/s)


def blend_duration(gap: np.ndarray, limits: Limits, dt: float) -> float:
    """How long a takeover blends over a joint gap (rad, per joint): smoothstep's peak dw/dt is 1.5 / T."""
    share = (np.abs(gap) / (BLEND_SPEED_SHARE * joint_speed_limit(limits, dt))).max()
    return float(min(BLEND_MAX_S, max(BLEND_S, 1.5 * share)))


def blend(new: Chunk, t: float, t_from: float, t_to: float, r0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(q, dq) at t: ``new``'s command plus the takeover's offset -- ``r0``, ``v0``: what was being commanded
    at t_from minus ``new``'s command and speed there -- decaying to 0 by t_to on a cubic Hermite. C1 from
    whatever was commanded, a takeover in the middle of another's blend included (11-19: blending from the
    replaced chunk's own command dropped what was left of the blend before it, a step of (1 - w) x its gap)."""
    qn, dqn, _ = new.command(t)
    T = t_to - t_from
    if t >= t_to or T <= 0:
        return qn, dqn
    s = min(max((t - t_from) / T, 0.0), 1.0)
    off = r0 * (2 * s**3 - 3 * s**2 + 1) + v0 * T * (s**3 - 2 * s**2 + s)
    doff = r0 * (6 * s**2 - 6 * s) / T + v0 * (3 * s**2 - 4 * s + 1)
    return qn + off, dqn + doff


def step_limits(step1_m: float | None, step1_deg: float | None) -> dict[str, float]:
    """The per-step limits for a checkpoint: what one step of its training data did (rollout_distribution:
    p99.5 x1.5), never above the ceilings. A step limit catches what the demonstrations never did (an IK
    branch change, a wild prediction); it does not slow the policy below its training, which would feed it
    a history it never saw."""
    out = {}
    if step1_m is not None and math.isfinite(step1_m):
        out["step_mm"] = round(min(STEP_MM_CEILING, step1_m * 1e3), 1)
    if step1_deg is not None and math.isfinite(step1_deg):
        out["step_deg"] = round(min(STEP_DEG_CEILING, step1_deg), 2)
    return out


def reposture(q: np.ndarray) -> np.ndarray:
    """(n,) per step (0 = anchor): the largest joint motion not explained by the step's own flange motion
    (dq - J(q[k-1])^+ dx, radians). A step that only moves the TCP is ~0; a branch change is not."""
    from tools.fr3 import fr3_kinematics as kin

    q = np.asarray(q, dtype=np.float64)
    out = np.zeros(len(q))
    if len(q) < 2:
        return out
    T, J = kin.NOMINAL.kinematics(q)
    for k in range(1, len(q)):
        dx = np.r_[T[k][:3, 3] - T[k - 1][:3, 3], _rotvec(T[k][:3, :3] @ T[k - 1][:3, :3].T)]
        out[k] = float(np.abs((q[k] - q[k - 1]) - np.linalg.pinv(J[k - 1]) @ dx).max())
    return out


def assess(
    T: np.ndarray,
    q: np.ndarray,
    plan_ok: np.ndarray,
    plan_why: list[str],
    limits: Limits,
    gap: np.ndarray | None = None,
    dt: float = STEP_DT,
) -> tuple[np.ndarray, list[str]]:
    """Per step (0 = anchor): may it be commanded? A step fails on the plan check or on a STEP limit
    against the step before it, and every step after a failed one fails with it. ``gap`` (per target, m):
    the arm's clearance to the carts/table/cameras there; closer than ``near_clearance_m`` the step limit
    is ``near_step_mm``. A step after a turn that could only be passed slowed more than ``step_stretch_max``
    (turn_speed_cap) fails too: the chunk then ends at rest on the turn instead of running through it. The
    turns at the anchor and step 1 are not judged (the anchor is measured; a first chunk starts from rest)."""
    n = len(q)
    ok = np.ones(n, dtype=bool)
    why = [""] * n
    rep = reposture(q)
    turn = turn_speed_cap(q)
    for k in range(1, n):
        reason = ""
        if not ok[k - 1]:
            reason = f"after step {k - 1}"
        elif not plan_ok[k - 1]:
            reason = plan_why[k - 1] or "plan check"
        else:
            mm, deg = pose_gap(T[k - 1], T[k])
            over = np.abs(q[k] - q[k - 1]) / (limits.step_joint_rad * JOINT_SCALE)
            near = gap is not None and k - 1 < len(gap) and gap[k - 1] < limits.near_clearance_m
            if near and mm > limits.near_step_mm:
                reason = (f"step {k}: {mm:.1f} mm in one step {gap[k - 1] * 1e3:.0f} mm from the carts/table/cameras "
                          f"(limit {limits.near_step_mm:g} closer than {limits.near_clearance_m * 1e3:.0f} mm)")
            elif mm > limits.step_mm:
                reason = f"step {k}: {mm:.1f} mm in one step (limit {limits.step_mm:g})"
            elif deg > limits.step_deg:
                reason = f"step {k}: {deg:.1f}° in one step (limit {limits.step_deg:g})"
            elif over.max() > limits.step_stretch_max:
                j = int(over.argmax())
                reason = (f"step {k}: j{j + 1} moves {abs(q[k][j] - q[k - 1][j]):.3f} rad in one step "
                          f"(limit {limits.step_joint_rad * JOINT_SCALE[j]:.3f} x{limits.step_stretch_max:g} "
                          "slowed down; an IK branch change?)")
            elif k >= 3 and 1.0 / (dt * turn[k - 1]) > limits.step_stretch_max:
                d2 = np.abs(q[k] - 2.0 * q[k - 1] + q[k - 2]) / (dt * dt)
                j = int((d2 / STOP_ACCELERATION).argmax())
                reason = (f"step {k}: j{j + 1} changes direction at step {k - 1} by {d2[j]:.0f} rad/s² at the policy's "
                          f"speed; within {STOP_ACCELERATION[j]:g} it would need x{1.0 / (dt * turn[k - 1]):.1f} slowed "
                          f"(limit x{limits.step_stretch_max:g}; a singularity, e.g. j2 near 0?)")
            elif rep[k] > limits.reposture_rad:
                reason = (f"step {k}: the arm re-postures {rep[k]:.3f} rad beyond what the TCP's motion needs "
                          f"(limit {limits.reposture_rad:g}; an IK branch change)")
        if reason:
            ok[k] = False
            why[k] = reason
    return ok, why


def accept(
    chunk: Chunk,
    now: float,
    *,
    first: bool,
    n_exec: int,
    q_cmd: np.ndarray,
    T_cmd: np.ndarray,
    limits: Limits,
    fk: Callable[[np.ndarray], np.ndarray],
    end_cap: int | None = None,
) -> str:
    """Take the chunk (sets k0, end and the steps' times; the first one is re-anchored to ``now``), or say
    why not. ``end_cap``: the last step it may run to whatever else is executable (stage 1: n_action_steps)."""
    H = chunk.horizon
    executable = np.flatnonzero(chunk.ok)
    end = min(int(executable.max()) if executable.size else 0, H, H if end_cap is None else end_cap)
    if first:
        chunk.t0 = now
        chunk.still_start = True
    chunk.prepare(limits, end)  # before k0: a slowed step is due later, so fewer have passed
    if first:
        chunk.k0 = 1
        mm, deg = pose_gap(T_cmd, chunk.T[1])
        if mm > limits.first_target_mm or deg > limits.first_target_deg:
            return (
                f"the first target is {mm:.1f} mm / {deg:.1f}° from where the arm stands "
                f"(limit {limits.first_target_mm:g} mm / {limits.first_target_deg:g}°)"
            )
    else:
        chunk.k0 = int(math.floor(chunk.cursor(now))) + 1
    if chunk.k0 > n_exec:
        return f"the prediction came {chunk.k0 - 1} steps late (more than n_action_steps={n_exec}); inputs too old"
    need = min(H, chunk.k0 + n_exec - 1)
    bad = [k for k in range(chunk.k0, need + 1) if not chunk.ok[k]]
    if bad:
        return f"step {bad[0]} of the {n_exec} it must be able to carry is not executable: {chunk.why[bad[0]]}"
    if not first:
        q_new, _, _ = chunk.command(now)
        gap = np.abs(q_new - q_cmd)
        dq = float(gap.max())
        mm, _ = pose_gap(T_cmd, fk(q_new))
        if mm > limits.switch_mm or dq > limits.switch_joint_rad:
            return (
                f"the new chunk is {mm:.1f} mm / {dq:.3f} rad (j{int(gap.argmax()) + 1}; joints "
                f"{np.round(q_new - q_cmd, 3).tolist()}) from the command it would replace "
                f"(limit {limits.switch_mm:g} mm / {limits.switch_joint_rad:g} rad)"
            )
    return ""


# ------------------------------------------------------------------ monitors ---


def tick_problem(
    *,
    q: np.ndarray,
    dq: np.ndarray,
    f_ext: np.ndarray,
    f0: np.ndarray,
    T_meas: np.ndarray,
    T_cmd: np.ndarray,
    loop_dt: float,
    limits: Limits,
    wall_lower: np.ndarray,
    wall_upper: np.ndarray,
    f_avg: np.ndarray | None = None,
) -> tuple[str, str]:
    """(kind, why) or ("", ""). Kinds "speed", "force", "tracking" hold where the arm is, not its command.
    ``f_avg``: O_F_ext averaged over FORCE_WINDOW_S (default: this tick's)."""
    speed = np.abs(dq)
    over = speed / (limits.joint_speed_rad_s * JOINT_SCALE)
    if over.max() > 1.0:
        j = int(over.argmax())
        return "speed", (f"j{j + 1} turns at {speed[j]:.2f} rad/s (limit {limits.joint_speed_rad_s * JOINT_SCALE[j]:.2f}: "
                         "oscillation or runaway)")
    peak = float(np.linalg.norm(np.asarray(f_ext)[:3] - np.asarray(f0)[:3]))
    if peak > limits.force_peak_n:
        return "force", f"external force {peak:.1f} N in one tick (limit {limits.force_peak_n:g})"
    force = peak if f_avg is None else float(np.linalg.norm(np.asarray(f_avg)[:3] - np.asarray(f0)[:3]))
    if force > limits.force_n:
        return "force", (f"external force {force:.1f} N over {FORCE_WINDOW_S * 1e3:.0f} ms (limit {limits.force_n:g})")
    mm, _ = pose_gap(T_meas, T_cmd)
    if mm > limits.tracking_mm:
        return "tracking", f"TCP {mm:.1f} mm off its command (limit {limits.tracking_mm:g})"
    margin = np.degrees(np.minimum(q - wall_lower, wall_upper - q))
    if margin.min() < limits.wall_deg:
        return "wall", f"j{int(margin.argmin()) + 1} within {limits.wall_deg:g}° of panda_py's wall"
    if loop_dt > limits.loop_stall_s:
        return "stall", f"the control loop stalled {loop_dt * 1e3:.0f} ms (limit {limits.loop_stall_s * 1e3:.0f})"
    return "", ""


HOLD_MEASURED = frozenset({"speed", "force", "tracking", "robot"})


class GripperDecider:
    """Binary close/open from the predicted width, with hysteresis and a minimum interval."""

    def __init__(self, width_now: float):
        self.closed = width_now < GRIPPER_CLOSE_BELOW_M
        self.last_t = -math.inf

    def decide(self, width: float, t: float) -> str:
        if t - self.last_t < GRIPPER_MIN_INTERVAL_S:
            return ""
        if not self.closed and width < GRIPPER_CLOSE_BELOW_M:
            self.closed, self.last_t = True, t
            return "close"
        if self.closed and width > GRIPPER_OPEN_ABOVE_M:
            self.closed, self.last_t = False, t
            return "open"
        return ""


# -------------------------------------------------------------- the arm (I/O) ---


class ArmIO(Protocol):
    """The loop's view of an arm: a fake in tests, panda_py on Thor."""

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # q, dq, O_F_ext (6)
    def start(self, q: np.ndarray) -> None: ...  # start joint tracking, holding q
    def command(self, q: np.ndarray, dq: np.ndarray) -> None: ...
    def check(self) -> None: ...  # raise if the robot reported an error
    def stop(self) -> None: ...


class PandaArm:
    """panda_py's JointPosition controller (joint tracking: pivot showed it beats soft impedance here)."""

    CONNECT_TRIES = 3  # 11-18: one libfranka "Connection timeout" with the port open, the retry worked

    def __init__(self, robot_ip: str):
        from panda_py import Panda, controllers

        last: Exception | None = None
        for attempt in range(self.CONNECT_TRIES):
            try:
                self.panda = Panda(robot_ip)
                break
            except Exception as exc:  # noqa: BLE001 -- libfranka raises its own types
                last = exc
                time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(f"could not connect to the arm at {robot_ip} after {self.CONNECT_TRIES} tries: {last}")
        self.ctrl = controllers.JointPosition(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING, filter_coeff=1.0)
        self.running = False

    def read(self):
        s = self.panda.get_state() if self.running else self.panda.get_robot().read_once()
        self.last = s
        return np.asarray(s.q, dtype=float), np.asarray(s.dq, dtype=float), np.asarray(s.O_F_ext_hat_K, dtype=float)

    def condition(self) -> str:
        """The robot's own mode and active errors at the last read, for a stop's message: 11-19, O_F_ext_hat_K
        read exactly 0 for minutes at a time, and once "Move command rejected: not possible in the current
        mode" -- the robot's state, which its status light shows, not anything this side sent."""
        s = getattr(self, "last", None)
        if s is None:
            return ""
        errors = [n for n in dir(s.current_errors) if not n.startswith("_") and getattr(s.current_errors, n) is True]
        mode = getattr(s.robot_mode, "name", str(s.robot_mode))
        return f"robot mode {mode}" + (f", errors: {', '.join(errors)}" if errors else "")

    def start(self, q):
        self.panda.start_controller(self.ctrl)
        self.running = True
        self.ctrl.set_control(np.asarray(q, dtype=float), np.zeros(7))

    def command(self, q, dq):
        self.ctrl.set_control(np.asarray(q, dtype=float), np.asarray(dq, dtype=float))

    def check(self):
        self.panda.raise_error()

    def stop(self):
        if self.running:
            self.running = False
            self.panda.stop_controller()


# ------------------------------------------------------------ the prediction ---


@dataclass
class Observation:
    t_img: float  # monotonic seconds
    images: dict[str, np.ndarray]  # cropped, as the policy takes them
    width: float
    wrench: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    obs: Observation | None
    chunk: Chunk | None
    error: str = ""
    latency_s: float = 0.0
    age_s: float = 0.0
    history: np.ndarray | None = None
    actions: np.ndarray | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)  # monotonic instant each phase began (Predictor.busy)


class Predictor:
    """In a thread: on request, the newest observation -> policy -> targets -> IK and plan check per step.

    ``observe(min_t)`` returns an Observation whose image is not older than min_t (or raises);
    ``predict(sample)`` the (H, 10) actions; ``plan(T_obs, actions, seed)`` per-step
    ``{"T", "q", "ok", "why", ...}`` (rollout_relative_tcp.PlanContext.evaluate); the anchor's joints are
    the measured ones at the observation instant (PoseHistory.q_at).
    """

    def __init__(self, *, contract: rt.PolicyContract, history: PoseHistory, observe, predict, plan, sample,
                 limits: Limits, clock=time.monotonic, on_phase: Callable[[str, float], None] | None = None):
        self.c = contract
        self.on_phase = on_phase
        self.history = history
        self.observe, self.predict, self.plan, self.sample = observe, predict, plan, sample
        self.limits = limits
        self.clock = clock
        self.requests: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="rollout-predictor")
        self.count = 0
        self.offsets_s = [o / float(contract.fps) for o in contract.history_offsets]
        # the phase the thread is in and since when ("" = idle): a run that stops while a prediction is
        # pending records where it was (11-19: a second chunk never came back and nothing said why)
        self.busy: tuple[str, float] = ("", 0.0)
        self._timing: dict[str, float] = {}

    def _phase(self, name: str) -> None:
        t = self.clock()
        self.busy = (name, t)
        if name:
            self._timing[name] = t
        if self.on_phase is not None:
            self.on_phase(name, t)

    def start(self) -> None:
        self.thread.start()

    def request(self, *, min_t: float, seed: np.ndarray | Chunk) -> None:
        """``seed``: the joints to start each target's IK from -- one posture, or the chunk being executed
        (each step of the next chunk starts from the posture the current one has at that instant, so the
        arm's redundancy is not resolved differently at the switch)."""
        self.requests.put((min_t, seed if isinstance(seed, Chunk) else np.array(seed, dtype=float), self.clock()))

    def close(self) -> None:
        self.requests.put(None)

    def _run(self) -> None:
        while True:
            item = self.requests.get()
            if item is None:
                return
            self.results.put(self.serve(*item))

    def serve(self, min_t: float, seed: np.ndarray | Chunk, t_request: float) -> Prediction:
        """One request: its prediction, with when each phase began; an exception is the prediction's error."""
        self._timing = {"requested": t_request}
        try:
            pred = self.once(min_t, seed)
        except Exception as exc:  # noqa: BLE001 -- stops the run, with the reason (a dead thread said nothing)
            pred = Prediction(None, None, error=f"predictor: {type(exc).__name__}: {exc}")
        pred.timing = dict(self._timing)
        self._phase("")
        return pred

    def once(self, min_t: float, seed: np.ndarray | Chunk) -> Prediction:
        L = self.limits
        self._phase("observe")
        try:
            obs = self.observe(min_t)
        except Exception as exc:  # noqa: BLE001 -- stops the run, with the reason
            return Prediction(None, None, error=f"input: {exc}")
        age = self.clock() - obs.t_img
        if not 0.0 <= age <= L.frame_age_s:
            return Prediction(obs, None, error=f"the frame is {age * 1e3:.0f} ms old (limit {L.frame_age_s * 1e3:.0f} ms; "
                              "negative means its clock is not CLOCK_MONOTONIC)", age_s=age)
        stale = obs.meta.get("box_stale_s")
        if stale is not None and stale > L.box_stale_s:
            return Prediction(obs, None, error=f"the BOX reading is {stale * 1e3:.0f} ms old (limit {L.box_stale_s * 1e3:.0f})", age_s=age)
        try:
            hist = self.history.poses(obs.t_img, self.offsets_s)
        except RuntimeError as exc:
            return Prediction(obs, None, error=f"TCP history: {exc}", age_s=age)
        self._phase("predict")
        t = self.clock()
        actions = np.asarray(self.predict(self.sample(hist, obs)), dtype=np.float64)
        latency = self.clock() - t
        if latency > L.predict_s:
            return Prediction(obs, None, error=f"prediction took {latency * 1e3:.0f} ms (limit {L.predict_s * 1e3:.0f})",
                              latency_s=latency, age_s=age, history=hist, actions=actions)
        H = len(actions)
        self._phase("plan")
        if isinstance(seed, Chunk):
            seed = np.stack([seed.command(obs.t_img + k * self.c.step_s)[0] for k in range(1, H + 1)])
        plan = self.plan(hist[-1], smooth_steps(actions), seed)  # the record keeps the policy's own actions
        T = np.concatenate([hist[-1][None], plan["T"]])
        q = np.concatenate([self.history.q_at(obs.t_img)[None], plan["q"]])
        gap = plan.get("env_gap_m")
        ok, why = assess(T, q, np.asarray(plan["ok"], dtype=bool), list(plan["why"]), L,
                         None if gap is None else np.asarray(gap, dtype=float), dt=self.c.step_s)
        self._phase("chunk")
        chunk = Chunk(self.count, obs.t_img, obs.t_img, self.c.step_s, T, q, ok, why, actions[:H, rt.POSE_DIM].copy())
        self.count += 1
        return Prediction(obs, chunk, latency_s=latency, age_s=age, history=hist, actions=actions,
                          plan={k: v for k, v in plan.items() if k not in ("T", "q", "ok", "why")})


PHASES = ("", "observe", "predict", "plan", "chunk")


class _PipeResults:
    """``get_nowait`` as on a queue: a prediction from the process, or its end as a prediction's error."""

    def __init__(self, owner: ProcessPredictor):
        self.owner = owner
        self.ended = False

    def get_nowait(self) -> Prediction:
        conn, process = self.owner.conn, self.owner.process
        with contextlib.suppress(EOFError, OSError):
            if conn.poll():
                return conn.recv()
        if not process.is_alive() and not self.ended:
            self.ended = True
            return Prediction(None, None, error=f"the predictor's process ended (exit code {process.exitcode})")
        raise queue.Empty


class ProcessPredictor:
    """The Predictor in a process of its own, so nothing it computes can hold the control loop up.

    11-19 (execute_20261119_153440): in a thread of the loop's process, one plan (IK retries) took 513 ms
    and the 200 Hz loop missed 133 ms; the controller kept the last command and its feed-forward speed, the
    arm ran 17 mm past it. The process is spawned (CUDA cannot be forked): ``build`` = (module, function,
    kwargs) runs there and returns the Predictor's parts, ``{"contract", "limits", "observe", "predict",
    "plan", "sample", "info"}``; ``info`` comes back from :meth:`start`. It reads the loop's FK through a
    :class:`SharedPoseHistory`, and ends when told to or when this process's end of the pipe closes."""

    def __init__(self, build: tuple[str, str, dict[str, Any]], *, history: SharedPoseHistory,
                 clock=time.monotonic, ready_timeout_s: float = 300.0):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.history, self.clock, self.ready_timeout_s = history, clock, ready_timeout_s
        self.conn, self._child_conn = ctx.Pipe()
        self._busy = ctx.Array("d", 2, lock=False)  # PHASES index, since (monotonic)
        self.process = ctx.Process(target=_predictor_process, name="rollout-predictor", daemon=True,
                                   args=(build, self._child_conn, self._busy, history.spec()))
        self.results = _PipeResults(self)
        self.offsets_s: list[float] = []
        self.info: dict[str, Any] = {}
        self._closed = False

    def start(self) -> dict[str, Any]:
        """Start it and wait until the policy is loaded and warmed up: its ``info``. Raises its error."""
        self.process.start()
        self._child_conn.close()  # the child's end lives only in the child: its recv sees EOF when we are gone
        deadline = time.monotonic() + self.ready_timeout_s
        while not self.conn.poll(0.5):
            if not self.process.is_alive():
                raise RuntimeError(f"the predictor's process ended before it was ready (exit code {self.process.exitcode})")
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"the predictor's process was not ready within {self.ready_timeout_s:g} s")
        try:
            kind, payload = self.conn.recv()
        except (EOFError, OSError):  # it ended: poll() sees the closed pipe as readable
            self.process.join(2.0)
            raise RuntimeError(f"the predictor's process ended before it was ready (exit code {self.process.exitcode})") from None
        if kind != "ready":
            self.close()
            raise RuntimeError(str(payload))
        self.info = payload
        self.offsets_s = list(payload["offsets_s"])
        return payload

    def request(self, *, min_t: float, seed: np.ndarray | Chunk) -> None:
        self.conn.send((min_t, seed if isinstance(seed, Chunk) else np.array(seed, dtype=float), self.clock()))

    @property
    def busy(self) -> tuple[str, float]:
        code = int(self._busy[0])
        return (PHASES[code] if 0 <= code < len(PHASES) else "?", float(self._busy[1]))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self.conn.send(None)
        if self.process.pid is not None:
            self.process.join(5.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2.0)
        with contextlib.suppress(Exception):
            self.conn.close()


def _predictor_process(build, conn, busy, history_spec) -> None:
    """The predictor's process (ProcessPredictor). A person's stop reaches the loop's process, which ends
    the run and then this process; SIGINT/SIGTERM here would only cut a prediction short."""
    import importlib
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        module, name, kwargs = build
        parts = getattr(importlib.import_module(module), name)(**kwargs)
        history = SharedPoseHistory.attach(history_spec)

        def on_phase(phase: str, t: float) -> None:
            busy[1] = t
            busy[0] = PHASES.index(phase) if phase in PHASES else -1

        predictor = Predictor(contract=parts["contract"], history=history, observe=parts["observe"],
                              predict=parts["predict"], plan=parts["plan"], sample=parts["sample"],
                              limits=parts["limits"], on_phase=on_phase)
        conn.send(("ready", {**parts.get("info", {}), "offsets_s": predictor.offsets_s}))
    except Exception as exc:  # noqa: BLE001 -- the loop's process raises it; nothing has moved
        with contextlib.suppress(Exception):
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            item = conn.recv()
        except (EOFError, OSError):
            return
        if item is None:
            return
        try:
            conn.send(predictor.serve(*item))
        except (BrokenPipeError, OSError):
            return


RT_PRIORITY = 80  # SCHED_FIFO for the control loop's thread (Thor: the gateway's rtprio limit is 99)


def realtime_thread(priority: int = RT_PRIORITY) -> str:
    """SCHED_FIFO at ``priority`` for the calling thread: "" when granted, else why not. Threads it starts
    afterwards inherit it (panda_py's 1 kHz control thread, started by ``arm.start``)."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except (AttributeError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def normal_thread() -> None:
    with contextlib.suppress(AttributeError, OSError):
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))


def hold_gc() -> Callable[[], None]:
    """No garbage collection during the run (fr3_pivot_auto: a full collection stalled the loop 0.4 s)."""
    gc.collect()
    gc.freeze()
    enabled = gc.isenabled()
    gc.disable()

    def resume() -> None:
        if enabled:
            gc.enable()
        gc.unfreeze()

    return resume


# ---------------------------------------------------------------- the loop ---


@dataclass
class RunResult:
    stopped: str  # "" = completed
    completed: bool
    chunks_accepted: int
    duration_s: float
    ticks: int
    end_cursor: float


def run_loop(
    *,
    arm: ArmIO,
    predictor: Predictor | ProcessPredictor,
    stage: Stage,
    n_exec: int,
    limits: Limits,
    fk_tcp: Callable[[np.ndarray], np.ndarray],
    wall_lower: np.ndarray,
    wall_upper: np.ndarray,
    history: PoseHistory,
    stop_flag: Callable[[], str],
    log: Callable[..., None],
    gripper: Callable[[str], None] | None = None,
    width_now: float = 0.088,
    clock=time.monotonic,
    sleep=time.sleep,
    loop_hz: float = LOOP_HZ,
    record: Callable[[dict[str, Any]], None] | None = None,
    start_check: Callable[[np.ndarray], str] | None = None,
) -> RunResult:
    """The control loop. Returns why it stopped ("" when the stage ran to its end)."""
    period = 1.0 / loop_hz
    q, dq, f0 = arm.read()
    if np.abs(dq).max() > limits.start_speed_rad_s:
        return RunResult(f"the arm is moving ({np.abs(dq).max():.3f} rad/s); it must stand still to start", False, 0, 0.0, 0, 0.0)
    refusal = start_check(q) if start_check is not None else ""
    if refusal:  # rollout_distribution check A: not where the demonstrations began
        return RunResult(refusal, False, 0, 0.0, 0, 0.0)
    t_start = clock()
    # a short still history before anything is asked: the first observation's history is "standing still"
    q, refusal = _stand_still(arm, max(3, int(0.15 * loop_hz)), lambda t, qq: history.add(t, fk_tcp(qq), qq),
                              clock, sleep, period)
    if refusal:
        return RunResult(refusal, False, 0, clock() - t_start, 0, 0.0)
    q_cmd = q.copy()
    dq_cmd = np.zeros(7)
    T_cmd = fk_tcp(q_cmd)
    arm.start(q_cmd)
    chunk: Chunk | None = None
    blending = False  # a takeover's offset (r0, v0) is decaying (blend)
    r0 = v0 = np.zeros(7)
    blend_from = blend_to = 0.0
    accepted = 0
    pending = False
    decider = GripperDecider(width_now)
    why = ""
    kind = ""
    completed = False
    t_last = clock()
    ticks = 0
    cursor = 0.0
    # the first frame must be taken after the FK buffer covers its whole history (offsets back to -67 ms)
    span = -min(predictor.offsets_s) if predictor.offsets_s else 0.0
    predictor.request(min_t=t_start + span + 0.005, seed=q_cmd)
    pending = True
    replan_at = max(1, n_exec - 6)  # updated from each chunk's measured lateness
    f_window: deque[np.ndarray] = deque(maxlen=max(1, round(FORCE_WINDOW_S * loop_hz)))
    silent = 0  # ticks in a row with O_F_ext_hat_K exactly 0
    try:
        while True:
            now = clock()
            loop_dt, t_last = now - t_last, now
            q, dq, f_ext = arm.read()
            T_meas = fk_tcp(q)
            history.add(now, T_meas, q)
            why = stop_flag()
            if why:
                break
            f_window.append(np.asarray(f_ext, dtype=np.float64))
            silent = 0 if np.any(f_ext) else silent + 1
            if silent >= max(2, round(FORCE_SILENT_S * loop_hz)):
                why, kind = (f"the external force estimate read exactly 0 for {silent} ticks{_condition(arm)}: "
                             "the force stop is blind"), "force"
                break
            if ticks > 0:
                kind, why = tick_problem(q=q, dq=dq, f_ext=f_ext, f0=f0, T_meas=T_meas, T_cmd=T_cmd, loop_dt=loop_dt,
                                   limits=limits, wall_lower=wall_lower, wall_upper=wall_upper,
                                   f_avg=np.mean(f_window, axis=0))
                if why:  # the tick that tripped, against the command it was judged by (phase 1)
                    if record:
                        record({"tick": {"t": now, "q": q, "dq": dq, "f_ext": f_ext, "q_cmd": q_cmd.copy(),
                                         "cursor": cursor, "chunk": -1 if chunk is None else chunk.index, "phase": 1}})
                    break
            if now - t_start > stage.max_s:
                why = f"reached the stage's {stage.max_s:g} s"
                completed = stage.chunks == 0
                if completed:
                    why = ""
                break
            # a prediction came back
            try:
                pred: Prediction | None = predictor.results.get_nowait()
            except queue.Empty:
                pred = None
            if pred is not None:
                pending = False
                if pred.error or pred.chunk is None:
                    why = pred.error or "no chunk"
                    log(event="chunk", accepted=False, why=why, latency_ms=pred.latency_s * 1e3, age_ms=pred.age_s * 1e3)
                    if record:
                        record({"prediction": pred, "accepted": False, "why": why, "t": now})
                    break
                new = pred.chunk
                refusal = accept(new, now, first=chunk is None, n_exec=n_exec, q_cmd=q_cmd, T_cmd=T_cmd,
                                 limits=limits, fk=fk_tcp, end_cap=n_exec if stage.chunks and chunk is None else None)
                log(event="chunk", index=new.index, accepted=not refusal, why=refusal, k0=new.k0, end=new.end,
                    latency_ms=round(pred.latency_s * 1e3, 1), age_ms=round(pred.age_s * 1e3, 1))
                if record:
                    record({"prediction": pred, "accepted": not refusal, "why": refusal, "t": now})
                if refusal:
                    why = refusal
                    break
                if chunk is not None:  # from the running command onto the new chunk (blend, blend_duration)
                    qn0, dqn0, _ = new.command(now)
                    r0, v0 = q_cmd + dq_cmd * loop_dt - qn0, dq_cmd - dqn0  # what this tick would have commanded
                    blending, blend_from = True, now
                    blend_to = now + blend_duration(r0, limits, new.dt)
                chunk = new
                accepted += 1
                replan_at = max(1, n_exec - max(0, new.k0 - 1))
            if chunk is None:
                dq_cmd = np.zeros(7)
                arm.command(q_cmd, dq_cmd)
            else:
                if blending and now < blend_to:
                    q_cmd, dq_cmd = blend(chunk, now, blend_from, blend_to, r0, v0)
                    cursor = chunk.cursor(now)
                else:
                    blending = False
                    q_cmd, dq_cmd, cursor = chunk.command(now)
                T_cmd = fk_tcp(q_cmd)
                arm.command(q_cmd, dq_cmd)
                if cursor >= chunk.end:
                    if stage.chunks and accepted >= stage.chunks:
                        completed = True
                        break
                    nxt = chunk.why[chunk.end + 1] if chunk.end < chunk.horizon else "its last step"
                    why = f"chunk {chunk.index} reached step {chunk.end} ({nxt}) before a newer chunk was taken"
                    break
                if (not pending) and (stage.chunks == 0 or accepted < stage.chunks) and cursor >= replan_at:
                    predictor.request(min_t=chunk.t_obs + 1e-3, seed=chunk)
                    pending = True
                    replan_at = math.inf
                if gripper is not None and stage.gripper:
                    action = decider.decide(chunk.width_at(chunk.cursor(now + GRIPPER_LEAD_S)), now)
                    if action:
                        gripper(action)
                        log(event="gripper", action=action, cursor=round(cursor, 2))
                        if record:
                            record({"event": {"kind": "gripper", "t": now, "action": action, "cursor": round(cursor, 2)}})
            arm.check()
            if record:
                record({"tick": {"t": now, "q": q, "dq": dq, "f_ext": f_ext, "q_cmd": q_cmd.copy(), "cursor": cursor,
                                 "chunk": -1 if chunk is None else chunk.index}})
            ticks += 1
            sleep(max(0.0, period - (clock() - now)))
    except Exception as exc:  # noqa: BLE001 -- a robot error ends the run; the hold below still runs
        why, kind = f"{type(exc).__name__}: {exc}", "robot"
    finally:
        # brake to rest (stop_motion) from the last validated command, or from where the arm is and how it
        # moves after contact, tracking, speed or a robot error; then hold there
        busy, since = predictor.busy
        if record and busy:
            record({"event": {"kind": "predictor", "t": clock(), "phase": busy, "since": since}})
        try:
            why = brake_and_hold(arm, kind, q_cmd, dq_cmd, why, record=record, clock=clock, sleep=sleep, period=period,
                                 cursor=cursor, chunk=-1 if chunk is None else chunk.index)
        finally:
            predictor.close()
    return RunResult(why, completed and not why, accepted, clock() - t_start, ticks, cursor)


def _stand_still(arm: ArmIO, n: int, on_read: Callable[[float, np.ndarray], None], clock, sleep,
                 period: float) -> tuple[np.ndarray, str]:
    """``n`` reads of the arm standing still before it is taken: (the last joints, "" or why it may not start).
    An external force estimate always carries noise; exactly 0 on every read is no estimate, and the force
    stop would be blind (11-19: O_F_ext_hat_K read 0 through two whole runs)."""
    forces = []
    q = None
    for _ in range(n):
        q, _, f = arm.read()
        on_read(clock(), q)
        forces.append(np.asarray(f, dtype=np.float64))
        sleep(period)
    if not np.any(forces):
        return q, (f"the arm reports no external force estimate (O_F_ext_hat_K exactly 0 on all {n} reads standing "
                   f"still{_condition(arm)}): the force stop would be blind; nothing moved")
    return q, ""


def _condition(arm: ArmIO) -> str:
    """"; <robot mode, errors>" when the arm can say (PandaArm.condition), else ""."""
    text = arm.condition() if hasattr(arm, "condition") else ""
    return f"; {text}" if text else ""


def brake_and_hold(arm: ArmIO, kind: str, q_cmd: np.ndarray, dq_cmd: np.ndarray, why: str, *, record, clock, sleep,
                   period: float, cursor: float = 0.0, chunk: int = -1) -> str:
    """After any end: brake to rest (stop_motion) from the last command, or from where the arm is and how it
    moves after contact, tracking, speed or a robot error; hold there HOLD_S; let the controller go. Returns
    ``why``, or why the hold failed when nothing else had."""
    try:
        q, dq, _ = arm.read()
        q0, v0 = (q, dq) if kind in HOLD_MEASURED else (q_cmd, dq_cmd)
        t_brake, brake = stop_motion(q0, v0)
        t_hold = clock()
        while clock() - t_hold < t_brake + HOLD_S:
            q_hold, dq_hold = brake(clock() - t_hold)
            arm.command(q_hold, dq_hold)
            if record:  # what the arm did after the stop (phase 2): an overshoot shows here
                qh, dqh, fh = arm.read()
                record({"tick": {"t": clock(), "q": qh, "dq": dqh, "f_ext": fh, "q_cmd": q_hold.copy(),
                                 "cursor": cursor, "chunk": chunk, "phase": 2}})
            sleep(period)
    except Exception as exc:  # noqa: BLE001 -- the controller is let go below either way
        why = why or f"hold: {type(exc).__name__}: {exc}"
    finally:
        arm.stop()
    return why


# ------------------------------------------------------- a move to a start ---

MOVE_SPEED_RAD_S = 0.3  # j1-j4 peak (the wrist by JOINT_SCALE): about panda_py's move_to_start speed
MOVE_ACCELERATION = 0.25 * FR3_ACCELERATION_LIMITS


def joint_move(q0: np.ndarray, q1: np.ndarray, *, speed: float = MOVE_SPEED_RAD_S,
               accel: np.ndarray = MOVE_ACCELERATION) -> tuple[float, Callable[[float], tuple[np.ndarray, np.ndarray]]]:
    """(duration, at(t) -> (q, dq)): the straight line in joint space on a minimum-jerk (quintic) profile, every
    joint together, at rest at both ends; no joint above ``speed`` (scaled per joint) or ``accel`` (a quintic
    peaks at 1.875 d/T and 5.774 d/T^2)."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = q1 - q0
    T = float(max(1e-3, np.max(1.875 * np.abs(d) / (speed * JOINT_SCALE)), np.max(np.sqrt(5.774 * np.abs(d) / accel))))

    def at(t: float) -> tuple[np.ndarray, np.ndarray]:
        s = min(max(t / T, 0.0), 1.0)
        return q0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5), d * (30 * s**2 - 60 * s**3 + 30 * s**4) / T

    return T, at


def run_joint_move(
    *,
    arm: ArmIO,
    q_goal: np.ndarray,
    limits: Limits,
    fk_tcp: Callable[[np.ndarray], np.ndarray],
    wall_lower: np.ndarray,
    wall_upper: np.ndarray,
    stop_flag: Callable[[], str],
    log: Callable[..., None],
    clock=time.monotonic,
    sleep=time.sleep,
    loop_hz: float = LOOP_HZ,
    record: Callable[[dict[str, Any]], None] | None = None,
    q_planned: np.ndarray | None = None,
) -> RunResult:
    """Drive the arm to ``q_goal`` along ``joint_move`` with the run's stops (tick_problem, a silent force
    estimate, the operator's stop); any stop brakes to rest (brake_and_hold). ``q_planned``: the joints the
    move was planned from; refused when the arm stands elsewhere now (the plan's checks were for that path)."""
    period = 1.0 / loop_hz
    q, dq, f0 = arm.read()
    if np.abs(dq).max() > limits.start_speed_rad_s:
        return RunResult(f"the arm is moving ({np.abs(dq).max():.3f} rad/s); it must stand still to start", False, 0, 0.0, 0, 0.0)
    if q_planned is not None and np.abs(q - q_planned).max() > 0.01:
        return RunResult("the arm moved since the move was planned; plan again", False, 0, 0.0, 0, 0.0)
    t_start = clock()
    q, refusal = _stand_still(arm, max(3, int(0.15 * loop_hz)), lambda t, qq: None, clock, sleep, period)
    if refusal:
        return RunResult(refusal, False, 0, clock() - t_start, 0, 0.0)
    duration, path = joint_move(q, q_goal)
    q_cmd, dq_cmd = q.copy(), np.zeros(7)
    arm.start(q_cmd)
    log(event="move", duration_s=round(duration, 2), max_joint_move_deg=round(math.degrees(float(np.abs(q_goal - q).max())), 1))
    f_window: deque[np.ndarray] = deque(maxlen=max(1, round(FORCE_WINDOW_S * loop_hz)))
    why = kind = ""
    silent = ticks = 0
    t0 = t_last = clock()
    try:
        while True:
            now = clock()
            loop_dt, t_last = now - t_last, now
            q, dq, f_ext = arm.read()
            why = stop_flag()
            if why:
                break
            f_window.append(np.asarray(f_ext, dtype=np.float64))
            silent = 0 if np.any(f_ext) else silent + 1
            if silent >= max(2, round(FORCE_SILENT_S * loop_hz)):
                why, kind = (f"the external force estimate read exactly 0 for {silent} ticks{_condition(arm)}: "
                             "the force stop is blind"), "force"
                break
            if ticks > 0:
                kind, why = tick_problem(q=q, dq=dq, f_ext=f_ext, f0=f0, T_meas=fk_tcp(q), T_cmd=fk_tcp(q_cmd),
                                         loop_dt=loop_dt, limits=limits, wall_lower=wall_lower, wall_upper=wall_upper,
                                         f_avg=np.mean(f_window, axis=0))
                if why:
                    break
            t = now - t0
            if t > duration + HOLD_S:  # there, and settled
                break
            q_cmd, dq_cmd = path(t)
            arm.command(q_cmd, dq_cmd)
            arm.check()
            if record:
                record({"tick": {"t": now, "q": q, "dq": dq, "f_ext": f_ext, "q_cmd": q_cmd.copy(),
                                 "cursor": min(t / duration, 1.0), "chunk": -1}})
            ticks += 1
            sleep(max(0.0, period - (clock() - now)))
    except Exception as exc:  # noqa: BLE001 -- a robot error ends the move; the brake below still runs
        why, kind = f"{type(exc).__name__}: {exc}", "robot"
    progress_s = clock() - t0
    why = brake_and_hold(arm, kind, q_cmd, dq_cmd, why, record=record, clock=clock, sleep=sleep, period=period,
                         cursor=min(progress_s / duration, 1.0))
    return RunResult(why, not why, 0, clock() - t_start, ticks, min(progress_s / duration, 1.0))


# ---------------------------------------------------------------- recording ---


class Recorder:
    """Every prediction's inputs and outputs (replayable through the offline path), every tick, and the
    gripper's commands. Read back by rollout_execute_report.

    A prediction (its images, npz, a json line) is written by a thread of its own: 11-19, written from the
    control loop it stalled the loop ~20 ms at every chunk, the command then jumped 0.65 step in one tick."""

    def __init__(self, record_dir: Path):
        self.dir = Path(record_dir)
        (self.dir / "steps").mkdir(parents=True, exist_ok=True)
        self.ticks: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.n = 0
        self.lock = threading.Lock()
        self._todo: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="rollout-recorder")
        self._writer.start()

    def _write_loop(self) -> None:
        while True:
            item = self._todo.get()
            if item is None:
                return
            try:
                self._write(item)
            except Exception as exc:  # noqa: BLE001 -- a lost record line must not take the run down
                print(f"rollout recorder: {type(exc).__name__}: {exc}", file=sys.stderr)

    def __call__(self, item: dict[str, Any]) -> None:
        if "tick" in item:
            self.ticks.append(item["tick"])
            return
        if "event" in item:
            self.events.append(item["event"])
            return
        self._todo.put(item)

    def _write(self, item: dict[str, Any]) -> None:
        pred: Prediction = item["prediction"]
        row: dict[str, Any] = {"step": self.n, "t": item["t"], "accepted": item["accepted"], "why": item["why"],
                               "error": pred.error, "latency_ms": round(pred.latency_s * 1e3, 2),
                               "age_ms": round(pred.age_s * 1e3, 2)}
        if pred.obs is not None:
            row["t_img"] = pred.obs.t_img
            row["obs_meta"] = pred.obs.meta
            row["width"] = pred.obs.width
        if pred.timing:
            row["timing"] = {k: round(v, 4) for k, v in pred.timing.items()}
        if pred.chunk is not None:
            c = pred.chunk
            row.update(index=c.index, k0=c.k0, end=c.end, t0=c.t0, offsets=c.offsets.round(5).tolist(),
                       q=c.q.round(5).tolist(),  # the anchor's measured joints, then each step's IK
                       ok=c.ok.tolist(), why_steps=c.why,
                       targets_base=c.T[:, :3, 3].round(5).tolist(), widths=c.widths.round(4).tolist(),
                       plan=pred.plan)
        if pred.obs is not None and pred.actions is not None and pred.history is not None:
            np.savez_compressed(
                self.dir / "steps" / f"{self.n:06d}.npz",
                width=np.float64(pred.obs.width),
                wrench=np.zeros(0) if pred.obs.wrench is None else pred.obs.wrench,
                actions=pred.actions,
                history=pred.history,
                **{f"image_{name}": img for name, img in pred.obs.images.items()},
            )
        with (self.dir / "chunks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_jsonable) + "\n")
        self.n += 1

    def close(self) -> None:
        self._todo.put(None)
        self._writer.join(timeout=30)
        if self.events:
            (self.dir / "events.json").write_text(json.dumps(self.events, default=_jsonable), encoding="utf-8")
        if not self.ticks:
            return
        keys = ("t", "q", "dq", "f_ext", "q_cmd", "cursor", "chunk")
        arrays = {k: np.asarray([row[k] for row in self.ticks]) for k in keys}
        arrays["phase"] = np.asarray([row.get("phase", 0) for row in self.ticks])  # 0 run, 1 the stop's tick, 2 hold
        np.savez_compressed(self.dir / "ticks.npz", **arrays)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


# ----------------------------------------------------------- gripper via gateway ---


class GatewayGripper:
    """Close/open through the gateway's arm-gripper button path (recorder stdin -> BOX), off the loop's thread."""

    def __init__(self, gateway: str, arm: str, on_error: Callable[[str], None]):
        self.url = gateway.rstrip("/") + "/api/fr3/arm-gripper"
        self.arm = arm
        self.on_error = on_error
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="rollout-gripper").start()

    def __call__(self, action: str) -> None:
        self.q.put(action)

    def _run(self) -> None:
        while True:
            action = self.q.get()
            try:
                request = urllib.request.Request(
                    self.url, method="POST", data=json.dumps({"arm": self.arm, "action": action}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as answer:
                    reply = json.loads(answer.read())
                if not reply.get("ok"):
                    self.on_error(f"gripper {action}: {reply.get('error') or 'refused'}")
            except Exception as exc:  # noqa: BLE001 -- reported as a stop
                self.on_error(f"gripper {action}: {exc}")


def stage_dict(stage: Stage, limits: Limits) -> dict[str, Any]:
    return {"stage": asdict(stage), "limits": asdict(limits), "loop_hz": LOOP_HZ, "hold_s": HOLD_S,
            "gripper": {"close_below_m": GRIPPER_CLOSE_BELOW_M, "open_above_m": GRIPPER_OPEN_ABOVE_M,
                        "lead_s": GRIPPER_LEAD_S, "min_interval_s": GRIPPER_MIN_INTERVAL_S},
            "joint_stiffness": JOINT_STIFFNESS.tolist(), "joint_damping": JOINT_DAMPING.tolist(),
            "step_filter": "smooth_steps: [1, 2, 1] / 4 over the predicted poses before IK and every check",
            "force_window_s": FORCE_WINDOW_S, "force_silent_s": FORCE_SILENT_S,
            "stop": "brake to rest together at the rated deceleration (stop_motion), then hold",
            "turns": "a knot is passed within STOP_ACCELERATION of change of direction (turn_speed_cap)",
            "time_convention": TIME_CONVENTION}
