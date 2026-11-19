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

import gc
import json
import math
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
    # a newer chunk against the command it replaces, at the moment of the switch
    switch_mm: float = 15.0
    switch_joint_rad: float = 0.05
    # the TCP (FK of the measured joints) against FK of the command
    tracking_mm: float = 20.0
    # |O_F_ext - its reading at the start|, force part
    force_n: float = 15.0
    # a measured joint speed no plan here asks for (oscillation, runaway); j5-j7 x JOINT_SCALE
    joint_speed_rad_s: float = 1.6
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
HOLD_S = 0.5  # the controller holds the last command this long after a stop, then lets go
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
        if not ts:
            raise RuntimeError("no FK read yet")
        i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)) if len(ts) > 1 else 0
        if len(ts) == 1 or t >= ts[-1]:
            return qs[-1].copy()
        if t <= ts[0]:
            return qs[0].copy()
        s = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        return qs[i - 1] + s * (qs[i] - qs[i - 1])

    def at(self, t: float, *, slack_s: float = 0.02) -> np.ndarray:
        with self.lock:
            ts, Ts = list(self.t), list(self.T)
        if not ts:
            raise RuntimeError("no FK read yet")
        if t < ts[0] - 1e-9:
            raise RuntimeError(f"the FK buffer starts {ts[0] - t:.3f} s after the instant asked for")
        if t > ts[-1]:
            if t - ts[-1] > slack_s:
                raise RuntimeError(f"no FK within {slack_s * 1e3:.0f} ms of the observation (last {t - ts[-1]:.3f} s before)")
            return Ts[-1].copy()
        i = int(np.searchsorted(ts, t))
        if i == 0 or ts[i] == t:
            return Ts[i].copy()
        s = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        return interpolate_pose(Ts[i - 1], Ts[i], s)

    def poses(self, t_obs: float, offsets_s: list[float]) -> np.ndarray:
        """(len(offsets), 4, 4): the TCP at t_obs + each offset (offsets <= 0, the last one 0)."""
        return np.stack([self.at(t_obs + o) for o in offsets_s])


# --------------------------------------------------------------------- chunks ---


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
STOP_ACCELERATION = 0.5 * FR3_ACCELERATION_LIMITS


def _segment_end_accelerations(q: np.ndarray, h: np.ndarray, m: np.ndarray, k: int) -> np.ndarray:
    """(7,) the largest |acceleration| of each joint in segment k (linear in u: at one of its ends)."""
    p0, p1, m0, m1 = q[k], q[k + 1], m[k] * h[k], m[k + 1] * h[k]
    a0 = -6 * p0 - 4 * m0 + 6 * p1 - 2 * m1
    a1 = 6 * p0 + 2 * m0 - 6 * p1 + 4 * m1
    return np.maximum(np.abs(a0), np.abs(a1)) / h[k] ** 2


def retime(q: np.ndarray, dt: float, limits: Limits, *, still_start: bool, end: int) -> np.ndarray:
    """(n,) each step's time from the anchor: k*dt, except where a joint would turn faster than
    ``joint_speed_limit`` -- that segment takes longer. The path (every validated step) is unchanged; only
    when the arm gets there. 11-19: refusing such a chunk stopped stage 2 where the policy asked 0.5 m/s, and
    a linear command's speed jumps at every step rang the joint impedance into a false 15 N contact.
    The last step (``end``, where the arm comes to rest) also takes long enough to stop within
    STOP_ACCELERATION."""
    n = len(q)
    if n < 2:
        return np.zeros(n)
    vmax = joint_speed_limit(limits, dt)
    h = dt * np.maximum(1.0, (np.abs(np.diff(q, axis=0)) / (vmax * dt)).max(axis=1))
    for _ in range(12):  # a stretched segment lowers its neighbours' knot speeds: converges in a few passes
        m = hermite_tangents(q, h, still_start=still_start, end=end)
        over = (_segment_peak_speeds(q, h, m) / vmax).max(axis=1)
        stop = 1.0
        if 0 < end < n:
            stop = float((_segment_end_accelerations(q, h, m, end - 1) / STOP_ACCELERATION).max())
        if over.max() <= 1.0 + 1e-6 and stop <= 1.0 + 1e-6:
            break
        h = np.where(over > 1.0, h * over * 1.001, h)
        if stop > 1.0:
            h[end - 1] *= math.sqrt(stop) * 1.001
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


#: a newer chunk takes over from the running one over this long (smoothstep), not in one tick: the two
#: predictions differ by up to switch_mm / switch_joint_rad at the switch, and in their speed
BLEND_S = 0.15


def blend(old: Chunk, new: Chunk, t: float, t_from: float, t_to: float) -> tuple[np.ndarray, np.ndarray]:
    """(q, dq) at t going from ``old``'s command to ``new``'s between t_from and t_to (C1 smoothstep)."""
    qn, dqn, _ = new.command(t)
    if t >= t_to or t_to <= t_from:
        return qn, dqn
    qo, dqo, _ = old.command(t)
    s = min(max((t - t_from) / (t_to - t_from), 0.0), 1.0)
    w, dw = s * s * (3 - 2 * s), 6 * s * (1 - s) / (t_to - t_from)
    return (1 - w) * qo + w * qn, (1 - w) * dqo + w * dqn + dw * (qn - qo)


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
) -> tuple[np.ndarray, list[str]]:
    """Per step (0 = anchor): may it be commanded? A step fails on the plan check or on a STEP limit
    against the step before it, and every step after a failed one fails with it. ``gap`` (per target, m):
    the arm's clearance to the carts/table/cameras there; closer than ``near_clearance_m`` the step limit
    is ``near_step_mm``."""
    n = len(q)
    ok = np.ones(n, dtype=bool)
    why = [""] * n
    rep = reposture(q)
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
        dq = float(np.abs(q_new - q_cmd).max())
        mm, _ = pose_gap(T_cmd, fk(q_new))
        if mm > limits.switch_mm or dq > limits.switch_joint_rad:
            return (
                f"the new chunk is {mm:.1f} mm / {dq:.3f} rad from the command it would replace "
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
) -> tuple[str, str]:
    """(kind, why) or ("", ""). Kinds "speed", "force", "tracking" hold where the arm is, not its command."""
    speed = np.abs(dq)
    over = speed / (limits.joint_speed_rad_s * JOINT_SCALE)
    if over.max() > 1.0:
        j = int(over.argmax())
        return "speed", (f"j{j + 1} turns at {speed[j]:.2f} rad/s (limit {limits.joint_speed_rad_s * JOINT_SCALE[j]:.2f}: "
                         "oscillation or runaway)")
    force = float(np.linalg.norm(np.asarray(f_ext)[:3] - np.asarray(f0)[:3]))
    if force > limits.force_n:
        return "force", f"external force {force:.1f} N (limit {limits.force_n:g})"
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
        return np.asarray(s.q, dtype=float), np.asarray(s.dq, dtype=float), np.asarray(s.O_F_ext_hat_K, dtype=float)

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


class Predictor:
    """In a thread: on request, the newest observation -> policy -> targets -> IK and plan check per step.

    ``observe(min_t)`` returns an Observation whose image is not older than min_t (or raises);
    ``predict(sample)`` the (H, 10) actions; ``plan(T_obs, actions, seed)`` per-step
    ``{"T", "q", "ok", "why", ...}`` (rollout_relative_tcp.PlanContext.evaluate); the anchor's joints are
    the measured ones at the observation instant (PoseHistory.q_at).
    """

    def __init__(self, *, contract: rt.PolicyContract, history: PoseHistory, observe, predict, plan, sample,
                 limits: Limits, clock=time.monotonic):
        self.c = contract
        self.history = history
        self.observe, self.predict, self.plan, self.sample = observe, predict, plan, sample
        self.limits = limits
        self.clock = clock
        self.requests: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="rollout-predictor")
        self.count = 0
        self.offsets_s = [o / float(contract.fps) for o in contract.history_offsets]

    def start(self) -> None:
        self.thread.start()

    def request(self, *, min_t: float, seed: np.ndarray | Callable[[float], np.ndarray]) -> None:
        """``seed``: the joints to start each target's IK from -- one posture, or a function of time (the
        chunk being executed: each step of the next chunk starts from the posture the current one has at
        that instant, so the arm's redundancy is not resolved differently at the switch)."""
        self.requests.put((min_t, seed if callable(seed) else np.array(seed, dtype=float)))

    def close(self) -> None:
        self.requests.put(None)

    def _run(self) -> None:
        while True:
            item = self.requests.get()
            if item is None:
                return
            min_t, seed = item
            self.results.put(self.once(min_t, seed))

    def once(self, min_t: float, seed: np.ndarray) -> Prediction:
        L = self.limits
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
        t = self.clock()
        actions = np.asarray(self.predict(self.sample(hist, obs)), dtype=np.float64)
        latency = self.clock() - t
        if latency > L.predict_s:
            return Prediction(obs, None, error=f"prediction took {latency * 1e3:.0f} ms (limit {L.predict_s * 1e3:.0f})",
                              latency_s=latency, age_s=age, history=hist, actions=actions)
        H = len(actions)
        if callable(seed):
            seed = np.stack([seed(obs.t_img + k * self.c.step_s) for k in range(1, H + 1)])
        plan = self.plan(hist[-1], actions, seed)
        T = np.concatenate([hist[-1][None], plan["T"]])
        q = np.concatenate([self.history.q_at(obs.t_img)[None], plan["q"]])
        gap = plan.get("env_gap_m")
        ok, why = assess(T, q, np.asarray(plan["ok"], dtype=bool), list(plan["why"]), L,
                         None if gap is None else np.asarray(gap, dtype=float))
        chunk = Chunk(self.count, obs.t_img, obs.t_img, self.c.step_s, T, q, ok, why, actions[:H, rt.POSE_DIM].copy())
        self.count += 1
        return Prediction(obs, chunk, latency_s=latency, age_s=age, history=hist, actions=actions,
                          plan={k: v for k, v in plan.items() if k not in ("T", "q", "ok", "why")})


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
    predictor: Predictor,
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
    for _ in range(max(3, int(0.15 * loop_hz))):
        q, dq, _ = arm.read()
        history.add(clock(), fk_tcp(q), q)
        sleep(period)
    q_cmd = q.copy()
    T_cmd = fk_tcp(q_cmd)
    arm.start(q_cmd)
    chunk: Chunk | None = None
    previous: Chunk | None = None  # the chunk being blended out (BLEND_S)
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
            if ticks > 0:
                kind, why = tick_problem(q=q, dq=dq, f_ext=f_ext, f0=f0, T_meas=T_meas, T_cmd=T_cmd, loop_dt=loop_dt,
                                   limits=limits, wall_lower=wall_lower, wall_upper=wall_upper)
                if why:
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
                if chunk is not None:  # blend out over BLEND_S, no longer than the old one's run remains
                    previous, blend_from = chunk, now
                    blend_to = min(now + BLEND_S, chunk.time_of(chunk.end))
                chunk = new
                accepted += 1
                replan_at = max(1, n_exec - max(0, new.k0 - 1))
            if chunk is None:
                arm.command(q_cmd, np.zeros(7))
            else:
                if previous is not None and now < blend_to:
                    q_cmd, dq_cmd = blend(previous, chunk, now, blend_from, blend_to)
                    cursor = chunk.cursor(now)
                else:
                    previous = None
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
                    predictor.request(min_t=chunk.t_obs + 1e-3, seed=lambda t, ch=chunk: ch.command(t)[0])
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
        # hold: the last validated command, or where the arm is after contact, tracking or a robot error
        try:
            q, _, _ = arm.read()
            q_hold = q if kind in HOLD_MEASURED else q_cmd
            t_hold = clock()
            while clock() - t_hold < HOLD_S:
                arm.command(q_hold, np.zeros(7))
                sleep(period)
        except Exception as exc:  # noqa: BLE001 -- the controller is let go below either way
            why = why or f"hold: {type(exc).__name__}: {exc}"
        finally:
            arm.stop()
            predictor.close()
    return RunResult(why, completed and not why, accepted, clock() - t_start, ticks, cursor)


# ---------------------------------------------------------------- recording ---


class Recorder:
    """Every prediction's inputs and outputs (replayable through the offline path), every tick, and the
    gripper's commands. Read back by rollout_execute_report."""

    def __init__(self, record_dir: Path):
        self.dir = Path(record_dir)
        (self.dir / "steps").mkdir(parents=True, exist_ok=True)
        self.ticks: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.n = 0
        self.lock = threading.Lock()

    def __call__(self, item: dict[str, Any]) -> None:
        if "tick" in item:
            self.ticks.append(item["tick"])
            return
        if "event" in item:
            self.events.append(item["event"])
            return
        pred: Prediction = item["prediction"]
        row: dict[str, Any] = {"step": self.n, "t": item["t"], "accepted": item["accepted"], "why": item["why"],
                               "error": pred.error, "latency_ms": round(pred.latency_s * 1e3, 2),
                               "age_ms": round(pred.age_s * 1e3, 2)}
        if pred.obs is not None:
            row["t_img"] = pred.obs.t_img
            row["obs_meta"] = pred.obs.meta
            row["width"] = pred.obs.width
        if pred.chunk is not None:
            c = pred.chunk
            row.update(index=c.index, k0=c.k0, end=c.end, t0=c.t0, offsets=c.offsets.round(5).tolist(),
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
        if self.events:
            (self.dir / "events.json").write_text(json.dumps(self.events, default=_jsonable), encoding="utf-8")
        if not self.ticks:
            return
        keys = ("t", "q", "dq", "f_ext", "q_cmd", "cursor", "chunk")
        np.savez_compressed(self.dir / "ticks.npz", **{k: np.asarray([row[k] for row in self.ticks]) for k in keys})


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
            "time_convention": TIME_CONVENTION}
