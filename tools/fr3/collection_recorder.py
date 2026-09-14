"""Recording an unattended run without touching the loop that is driving the arm.

The rule this module exists to enforce is one sentence long: **the control loop publishes, and
nothing else.** A step in `scene_reset._run_step` appends one tuple to an in-memory queue and
carries on. Every expensive thing -- reading a camera, encoding a frame, opening a file, closing
a shard -- happens on the recorder's own thread. The reason is not tidiness. That loop's timeout,
its `SCENE_RESET_REACH_STALL_S` stall detection and its speed limit are all measured against its
own wall clock, so a blocking read inserted into it does not merely slow recording down; it
silently widens every safety margin the reset has, on a rig that will be running with nobody in
the room.

Three properties follow from taking that rule seriously rather than nominally.

**The queue is bounded and drops are counted.** A recorder that falls behind must lose frames --
the alternative is growing into the swap of a machine that is driving an arm -- but a recorder
that loses them quietly is worse than one that stops. `dropped` is part of the status line and
part of every shard footer, so a night that produced thin data says so in the data.

**State and action come from the same instant, and neither is re-read.** The control loop already
reads the arm every step, immediately before it sends. Asking the recorder to read the arm again
would produce a second, differently-timed measurement of the same quantity and pair the action
with the wrong one. So the tap carries the observation the loop already has; the recorder adds
only what the loop does not have, which is pixels.

**Pixels are already decoupled, so pairing them is a read, not a wait.** Both cameras on this rig
are RealSense, and `read_closest(t)` selects from a locked ring buffer filled by the camera's own
background thread. The recorder therefore asks for the frame nearest each sample's timestamp
rather than for "the newest frame now", which is what makes a recorder running at its own rate
produce frames that belong to the steps they are stored beside.

What is written is a shard directory per few hundred frames: a JSONL of rows, the frames as JPEG,
and a `footer.json` written on close. The footer is the point. A shard with one is complete; the
one without is the shard that was in flight when the process died, and nothing else is in doubt.
A night in a single file would put the whole night in doubt -- which is not hypothetical here,
since P0-0 lost 4192 frames that existed only inside a process that never got to finish.

Conversion to a LeRobot dataset is deliberately *not* here. It is an offline pass over finished
shards, and keeping it offline is what lets this module be judged on one question: did it record
what happened.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable


# The per-step displacement the deployed policy is actually bounded by, in millimetres.
#
# Sourced, not chosen, and *not* the number this rig's teleop path uses. The driver's
# `max_target_delta_pos` (1 mm an axis) clamps the relative branch of `send_action`, which is the
# SpaceMouse path; a rollout sends absolute poses and takes the branch with no clamp at all. What
# bounds a rollout is `command_guard.limit_command_for_safety`, whose step limit defaults to
# 5.0 mm of magnitude measured against `prev_cmd`. Auditing auto-collected data against 1 mm an
# axis would fail data the deployment accepts; auditing against nothing would ship data it does
# not.
DEPLOYMENT_STEP_LIMIT_MM = 5.0
# What the demonstrations themselves contain, from the run that sized the guard
# (eeframe_fr3_spacemouse_20260813_160401): p50 1.59 mm, p95 2.93 mm per step. The guard admits
# 99.90% of those frames, so sitting *at* the guard means every recorded step is an outlier by
# the standard of the data it is meant to be mixed with.
DEMO_STEP_P50_MM = 1.59
DEMO_STEP_P95_MM = 2.93
# Where a recorded leg should walk. Between the demonstrations' median and their p95, and
# comfortably inside the guard -- so an auto-collected step looks like a demonstrated step rather
# than like the fastest thing the guard will pass.
DEFAULT_STEP_MM = 2.0
# What counts as "this command went nowhere", in millimetres.
#
# `dagger_dataset.STILL_STEP_MM`'s number, deliberately, so the filter that judges auto-collected
# motion and the writer that judges human motion divide frames the same way. It is redefined here
# rather than imported because that module pulls in numpy, pyarrow and the dataset stack, and this
# one has to import on a machine that is only driving an arm.
#
# It matters more here than it looks. A recorded leg spends its gripper settle republishing the
# same setpoint -- 0.6 s at 30 Hz is 18 identical commands -- so most of what an auto-collected
# episode contains is a command to stay put. Percentiles taken over all of it read zero and say
# nothing about whether the motion resembles a demonstration.
STILL_STEP_MM = 0.05

# How many samples may wait in the queue before the oldest are dropped.
#
# 2048 is about a minute of 30 Hz control. The recorder is expected to run a few frames behind,
# not a minute, so a queue this deep is not a buffer for normal operation -- it is the margin
# that lets a shard rotation or a slow disk flush pass without losing anything, and anything
# beyond it is a recorder that is not keeping up and should be reported as such.
DEFAULT_QUEUE_CAPACITY = 2048
# Frames per shard. 600 is twenty seconds at 30 Hz: small enough that losing the shard in flight
# costs twenty seconds, large enough that a night is thousands of directories rather than
# hundreds of thousands of files in one.
DEFAULT_SHARD_FRAMES = 600
# A shard also closes on age, so a quiet stretch -- the unrecorded legs between episodes -- does
# not leave the last frames of the previous shard unfooted for minutes.
DEFAULT_SHARD_SECONDS = 60.0


class RecorderError(RuntimeError):
    """A recording setup that must not be started."""


@dataclass(frozen=True)
class ControlSample:
    """One published step: what the expert saw, and what it sent from there."""

    t: float
    phase: str
    episode: int
    observation: dict[str, Any]
    action: dict[str, float]
    # How far this command moved the tool from the previous one, or None at the start of an
    # episode. Computed once, where the action is published, rather than twice.
    stepMm: float | None = None


@dataclass(frozen=True)
class Marker:
    """Something that happened between or around steps, and is not a training frame.

    Markers ride the same queue as samples and are never decimated. The unrecorded legs are the
    reason they exist: a cycle's fetch, place and retrieval are environment operations that do not
    belong in a dataset, but a dataset with unexplained gaps between its episodes is
    indistinguishable from one that dropped frames. A marker is what makes the gap a reading.
    """

    t: float
    kind: str
    episode: int
    fields: dict[str, Any]


class ControlTap:
    """The only object the control loop touches. Every method on it is O(1) and allocation-light.

    Deliberately not a `queue.Queue`: a bounded deque drops the *oldest* sample when it is full,
    which is the right thing to lose, and `Queue` would either block the control loop or raise
    inside it. Neither is acceptable in a loop that is walking a setpoint.
    """

    # Row keys a marker's own fields must not shadow. Learned rather than assumed: a cycle's most
    # natural field name is `kind`, and spreading it into the row silently retyped every
    # `episode_start` marker as something that was not a marker at all.
    RESERVED_MARKER_FIELDS = frozenset({"kind", "t", "marker", "episode"})

    def __init__(self, *, capacity: int = DEFAULT_QUEUE_CAPACITY, audit: "StepAudit | None" = None):
        if capacity < 1:
            raise RecorderError("capacity must be at least 1.")
        self._audit = audit
        self._items: deque[ControlSample | Marker] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._published = 0
        self._dropped = 0
        self._markers = 0
        self._last_publish_t = 0.0
        self._episode = 0

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def audit(self) -> "StepAudit | None":
        return self._audit

    def begin_episode(self, episode: int) -> None:
        """Set the episode every later sample is stamped with, so labels are written not inferred.

        The stamp is applied at publish time on purpose. A dataset whose episode boundaries are
        recovered afterwards from timestamps is a dataset whose boundaries are a hypothesis, and
        the first thing that makes it wrong is the very failure the run is trying to record.
        """

        self._episode = int(episode)
        # The audit's episode boundary is this one by construction. Two objects that had to be
        # told separately would eventually be told once, and the gap between two legs would be
        # audited as though it were a policy step.
        if self._audit is not None:
            self._audit.begin_episode()

    def publish(
        self,
        t: float,
        phase: str,
        observation: dict[str, Any],
        action: dict[str, float],
    ) -> None:
        # Audited here rather than on the recorder's thread. It is three floats and a square
        # root -- nothing that can block -- and putting it here is what lets the run *act* on a
        # violation: an audit running behind the control loop cannot answer "was that episode
        # valid" at the moment the loop has to decide.
        step_mm = self._audit.observe(action) if self._audit is not None else None
        sample = ControlSample(
            t=float(t),
            phase=str(phase),
            episode=self._episode,
            observation=observation,
            action=action,
            stepMm=step_mm,
        )
        with self._lock:
            full = len(self._items) == self._items.maxlen
            self._items.append(sample)
            self._published += 1
            self._last_publish_t = sample.t
            if full:
                self._dropped += 1

    def mark(self, kind: str, /, **fields: Any) -> None:
        """`kind` is positional-only, and a field that would shadow a row key is refused.

        Both come from the same defect. A marker's fields are spread into the row it becomes, so a
        field called `kind` overwrote the row's own `kind` and turned every `episode_start` marker
        into a row that no reader would ever recognise as a marker. Silently. Refusing at the call
        site makes that a failure that cannot be shipped rather than data that cannot be read.
        """

        collisions = sorted(self.RESERVED_MARKER_FIELDS.intersection(fields))
        if collisions:
            raise RecorderError(
                f"marker field(s) {collisions} would overwrite the row's own keys "
                f"{sorted(self.RESERVED_MARKER_FIELDS)}. Rename the field."
            )
        marker = Marker(t=time.perf_counter(), kind=str(kind), episode=self._episode, fields=dict(fields))
        with self._lock:
            full = len(self._items) == self._items.maxlen
            self._items.append(marker)
            self._markers += 1
            self._last_publish_t = marker.t
            if full:
                self._dropped += 1

    def drain(self, limit: int = 256) -> list[ControlSample | Marker]:
        out: list[ControlSample | Marker] = []
        with self._lock:
            while self._items and len(out) < limit:
                out.append(self._items.popleft())
        return out

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = len(self._items)
            return {
                "published": self._published,
                "markers": self._markers,
                "dropped": self._dropped,
                "pending": pending,
                "lastPublishT": self._last_publish_t,
                "capacity": self._items.maxlen,
            }


class StepAudit:
    """Does every sent action sit inside the step the deployment will accept? An assertion, not a note.

    The card this comes from asked for the check to be able to *fail* something rather than to be
    a convention two files remind each other about. So a violation marks the episode invalid and
    is counted, and the run's summary carries the count. The measure is the magnitude of the step
    against the previous sent command, because that is precisely what
    `command_guard.limit_command_for_safety` measures -- an audit that used a per-axis bound would
    be testing a different rule from the one deployment applies.

    It also reports the distribution, not just the violations. Data that never trips the guard but
    sits at its ceiling is data whose every frame is a 99.9th-percentile step by the standard of
    the demonstrations it is about to be mixed with, and only the percentiles say so.
    """

    def __init__(self, *, limit_mm: float = DEPLOYMENT_STEP_LIMIT_MM):
        if limit_mm <= 0.0:
            raise RecorderError("limit_mm must be positive.")
        self._limit_mm = float(limit_mm)
        self._previous: tuple[float, float, float] | None = None
        self._steps_mm: list[float] = []
        self._violations = 0
        self._episode_violations = 0

    def begin_episode(self) -> None:
        """Forget the previous command. The gap across a leg boundary is not a policy step."""

        self._previous = None
        self._episode_violations = 0

    def observe(self, action: dict[str, float]) -> float | None:
        xyz = (float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"]))
        previous, self._previous = self._previous, xyz
        if previous is None:
            return None
        step_mm = 1000.0 * math.dist(previous, xyz)
        self._steps_mm.append(step_mm)
        if step_mm > self._limit_mm:
            self._violations += 1
            self._episode_violations += 1
        return step_mm

    @property
    def limit_mm(self) -> float:
        return self._limit_mm

    @property
    def episode_is_valid(self) -> bool:
        return self._episode_violations == 0

    def summary(self) -> dict[str, Any]:
        steps = sorted(self._steps_mm)
        moving = [step for step in steps if step >= STILL_STEP_MM]

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
            return values[index]

        return {
            "limitMm": self._limit_mm,
            "steps": len(steps),
            "violations": self._violations,
            "maxStepMm": steps[-1] if steps else None,
            # Over the moving steps only. The still ones are real commands and are counted in
            # `stillFraction`, but including them in the percentiles would answer a question
            # nobody asked -- "how much of an episode is a gripper settle" -- with the number
            # that was supposed to answer "does the motion look like a demonstration".
            "movingSteps": len(moving),
            "stillFraction": (len(steps) - len(moving)) / len(steps) if steps else None,
            "p50StepMm": percentile(moving, 0.50),
            "p95StepMm": percentile(moving, 0.95),
            "demoP50StepMm": DEMO_STEP_P50_MM,
            "demoP95StepMm": DEMO_STEP_P95_MM,
        }


class FrameSink:
    """Where pixels land. Split out so the recorder can be tested without a camera or a disk."""

    def write(self, shard_dir: Path, index: int, camera: str, frame: Any) -> str:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - nothing to do by default
        return None


class JpegFrameSink(FrameSink):
    """One JPEG per camera per frame, encoded on the recorder's thread.

    JPEG rather than raw because a night of raw 640x480 pairs is hundreds of gigabytes, and rather
    than video because an interrupted video file is not a partial recording, it is a file that has
    to be repaired. Individual frames make a crash cost exactly the frame being written.
    """

    def __init__(self, *, quality: int = 95):
        self._quality = int(quality)
        self._encode: Any = None

    def _encoder(self) -> Any:
        if self._encode is None:
            import cv2  # imported here so the module loads on a machine without OpenCV

            self._encode = cv2
        return self._encode

    def write(self, shard_dir: Path, index: int, camera: str, frame: Any) -> str:
        cv2 = self._encoder()
        name = f"{index:06d}_{camera}.jpg"
        path = shard_dir / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            raise RecorderError(f"could not encode a frame for camera {camera}.")
        path.write_bytes(buffer.tobytes())
        return f"frames/{name}"


def cameras_from_robot(robot: Any, *, max_age_ms: int = 500) -> dict[str, Callable[[float], tuple[Any, float]]]:
    """Adapters that answer "the frame nearest this instant" for each camera the robot has.

    `read_closest` is preferred and is what both RealSense cameras on this rig provide; the
    fallback exists because a camera that can only say "the newest frame" is still recordable, it
    just cannot be paired as tightly, and that difference belongs in one place rather than being
    discovered per call site.
    """

    adapters: dict[str, Callable[[float], tuple[Any, float]]] = {}
    for name, camera in getattr(robot, "cameras", {}).items():
        read_closest = getattr(camera, "read_closest", None)
        if callable(read_closest):
            def closest(t: float, _read: Any = read_closest) -> tuple[Any, float]:
                return _read(t, max_age_ms=max_age_ms)

            adapters[name] = closest
            continue

        read_latest_with_timestamp = getattr(camera, "read_latest_with_timestamp", None)
        if callable(read_latest_with_timestamp):
            def latest_pair(_t: float, _read: Any = read_latest_with_timestamp) -> tuple[Any, float]:
                return _read(max_age_ms=max_age_ms)

            adapters[name] = latest_pair
            continue

        def latest(_t: float, _camera: Any = camera) -> tuple[Any, float]:
            frame = _camera.read_latest(max_age_ms=max_age_ms)
            return frame, float(getattr(_camera, "latest_timestamp", time.perf_counter()))

        adapters[name] = latest
    return adapters


@dataclass
class _Shard:
    index: int
    path: Path
    handle: Any
    opened_t: float
    rows: int = 0
    frames: int = 0
    first_t: float | None = None
    last_t: float | None = None
    camera_failures: int = 0


class Recorder:
    """The thread that does everything the control loop is forbidden to do.

    Its own rate is deliberately independent of the control loop's. The loop steps as fast as the
    setpoint walk needs; a dataset wants a steady period. So samples are decimated to `fps` here,
    which also means a slow leg and a fast leg produce frames at the same spacing rather than at
    whatever the walk happened to cost.
    """

    def __init__(
        self,
        tap: ControlTap,
        root: Path | str,
        *,
        cameras: dict[str, Callable[[float], tuple[Any, float]]] | None = None,
        sink: FrameSink | None = None,
        fps: float = 30.0,
        shard_frames: int = DEFAULT_SHARD_FRAMES,
        shard_seconds: float = DEFAULT_SHARD_SECONDS,
        poll_s: float = 0.005,
    ):
        if fps <= 0.0:
            raise RecorderError("fps must be positive.")
        if shard_frames < 1:
            raise RecorderError("shard_frames must be at least 1.")
        self._tap = tap
        self._root = Path(root)
        self._cameras = dict(cameras or {})
        self._sink = sink if sink is not None else JpegFrameSink()
        self._period_s = 1.0 / float(fps)
        self._fps = float(fps)
        self._shard_frames = int(shard_frames)
        self._shard_seconds = float(shard_seconds)
        self._poll_s = float(poll_s)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._shard: _Shard | None = None
        self._shard_index = 0
        self._rows_written = 0
        self._frames_written = 0
        self._decimated = 0
        self._camera_failures = 0
        self._last_kept_t: float | None = None
        self._last_kept_phase: str | None = None
        self._last_disk_t = 0.0
        self._last_row_t = 0.0
        self._error = ""
        self._stopped = False

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RecorderError("recorder already started.")
        self._root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="collection-recorder", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Stop after draining. A recorder that exits on the stop signal loses the last shard."""

        self._stop.set()
        self._stopped = True
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._close_shard("stopped")
        self._sink.close()
        return self.status()

    def __enter__(self) -> "Recorder":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- the loop --------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._pump():
                    time.sleep(self._poll_s)
            # Drain whatever the control loop published between the last pump and the signal.
            while self._pump():
                pass
        except Exception as exc:  # noqa: BLE001 - a dead recorder must say so, not vanish
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] collection_recorder=died details={self._error}", flush=True)

    def _pump(self) -> bool:
        items = self._tap.drain()
        if not items:
            return False
        for item in items:
            if isinstance(item, Marker):
                self._write_row(
                    {
                        "kind": "marker",
                        "t": item.t,
                        "marker": item.kind,
                        "episode": item.episode,
                        **item.fields,
                    },
                    frames=0,
                )
                continue
            self._handle_sample(item)
        return True

    def _handle_sample(self, sample: ControlSample) -> None:
        # The step was measured where it was published, against every command the arm received --
        # not against the frames that survived decimation, which would be a subsample of the thing
        # the guard actually bounds.
        step_mm = sample.stepMm
        # The first sample of a leg is never decimated. A leg shorter than one record period would
        # otherwise contribute no frames at all and say nothing about it, which is the difference
        # between a thinly sampled approach and an approach that is missing from the dataset.
        new_phase = sample.phase != self._last_kept_phase
        too_soon = self._last_kept_t is not None and sample.t - self._last_kept_t < self._period_s - 1e-9
        if too_soon and not new_phase:
            self._decimated += 1
            return
        self._last_kept_t = sample.t
        self._last_kept_phase = sample.phase

        row: dict[str, Any] = {
            "kind": "frame",
            "t": sample.t,
            "phase": sample.phase,
            "episode": sample.episode,
            "state": {key: value for key, value in sample.observation.items() if _is_scalar(value)},
            "sent_action": dict(sample.action),
        }
        if step_mm is not None:
            row["stepMm"] = step_mm
            audit = self._tap.audit
            if audit is not None and step_mm > audit.limit_mm:
                row["stepOverLimit"] = True

        shard = self._open_shard_if_needed()
        images: dict[str, Any] = {}
        for name, read in self._cameras.items():
            try:
                frame, timestamp = read(sample.t)
            except Exception as exc:  # noqa: BLE001 - one stale camera must not end the night
                self._camera_failures += 1
                shard.camera_failures += 1
                row.setdefault("cameraErrors", {})[name] = f"{type(exc).__name__}: {exc}"
                continue
            images[name] = frame
            row.setdefault("cameraTimestamps", {})[name] = float(timestamp)
            row.setdefault("cameraSkewMs", {})[name] = 1000.0 * (float(timestamp) - sample.t)

        index = shard.rows
        written = 0
        for name, frame in images.items():
            row.setdefault("images", {})[name] = self._sink.write(shard.path, index, name, frame)
            written += 1
        self._write_row(row, frames=written)

    # -- shards ----------------------------------------------------------------------------

    def _open_shard_if_needed(self) -> _Shard:
        if self._shard is not None:
            return self._shard
        path = self._root / f"shard_{self._shard_index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        handle = (path / "rows.jsonl").open("a", encoding="utf-8")
        self._shard = _Shard(index=self._shard_index, path=path, handle=handle, opened_t=time.perf_counter())
        self._shard_index += 1
        return self._shard

    def _write_row(self, row: dict[str, Any], *, frames: int) -> None:
        shard = self._open_shard_if_needed()
        shard.handle.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
        # Flushed per row rather than per shard. The buffering this gives up is the whole reason a
        # crash would otherwise cost more than the frame it happened on.
        shard.handle.flush()
        shard.rows += 1
        shard.frames += frames
        t = float(row.get("t", time.perf_counter()))
        shard.first_t = t if shard.first_t is None else shard.first_t
        shard.last_t = t
        with self._lock:
            self._rows_written += 1
            self._frames_written += frames
            self._last_row_t = time.perf_counter()
            self._last_disk_t = self._last_row_t
        # Checked per row rather than per drained batch. A batch can be hundreds of samples, and
        # rotating only between batches made the configured shard size a floor rather than a size
        # -- which defeats the point of shards, since the shard in flight is exactly what a crash
        # costs.
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        shard = self._shard
        if shard is None:
            return
        if shard.rows >= self._shard_frames or time.perf_counter() - shard.opened_t >= self._shard_seconds:
            self._close_shard("rotated")

    def _close_shard(self, reason: str) -> None:
        shard, self._shard = self._shard, None
        if shard is None:
            return
        footer = {
            "kind": "footer",
            "shard": shard.index,
            "reason": reason,
            "rows": shard.rows,
            "frames": shard.frames,
            "firstT": shard.first_t,
            "lastT": shard.last_t,
            "cameraFailures": shard.camera_failures,
            "tap": self._tap.status(),
            "closedAt": time.time(),
        }
        try:
            shard.handle.flush()
        finally:
            shard.handle.close()
        # The footer is written last and in its own file: a shard directory that has one is a
        # shard nothing was still writing to.
        (shard.path / "footer.json").write_text(
            json.dumps(footer, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self._lock:
            self._last_disk_t = time.perf_counter()

    # -- what a person watching this needs to see ------------------------------------------

    def status(self) -> dict[str, Any]:
        """Two heartbeats and a disk landing, because the moment they diverge is the finding.

        A single "it is running" light cannot distinguish an arm that stopped moving from a
        recorder that stopped recording, and those two need opposite responses. The disk line is
        separate again for the same reason: rows accumulating in a process are not data yet.
        """

        tap = self._tap.status()
        now = time.perf_counter()
        with self._lock:
            rows, frames = self._rows_written, self._frames_written
            last_row_t, last_disk_t, error = self._last_row_t, self._last_disk_t, self._error
        return {
            "controller": {
                "publishedSteps": tap["published"],
                "markers": tap["markers"],
                "lastPublishAgoS": None if not tap["lastPublishT"] else now - tap["lastPublishT"],
                "pending": tap["pending"],
                "dropped": tap["dropped"],
            },
            "recorder": {
                "alive": self._thread is not None and self._thread.is_alive(),
                # Separate from `alive` because they need opposite responses: a recorder that was
                # asked to stop is a finished run, one that is not alive and was not asked is the
                # single worst thing that can happen to an unattended night.
                "stopped": self._stopped,
                "rows": rows,
                "framesWritten": frames,
                "decimated": self._decimated,
                "cameraFailures": self._camera_failures,
                "lastRowAgoS": None if not last_row_t else now - last_row_t,
                "targetFps": self._fps,
                "error": error,
            },
            "disk": {
                "root": str(self._root),
                "shardsOpened": self._shard_index,
                "openShard": None if self._shard is None else self._shard.path.name,
                "rowsInOpenShard": 0 if self._shard is None else self._shard.rows,
                "lastWriteAgoS": None if not last_disk_t else now - last_disk_t,
            },
        }

    def describe_status(self) -> str:
        """The status as the one line a person should be able to read at 2 a.m."""

        status = self.status()
        controller, recorder, disk = status["controller"], status["recorder"], status["disk"]

        def ago(value: float | None) -> str:
            return "never" if value is None else f"{value:.1f}s ago"

        return (
            f"controller: {controller['publishedSteps']} steps, last {ago(controller['lastPublishAgoS'])}, "
            f"queued {controller['pending']}, dropped {controller['dropped']} | "
            f"recorder: {'alive' if recorder['alive'] else ('stopped' if recorder['stopped'] else 'DEAD')}, "
            f"{recorder['rows']} rows, "
            f"last {ago(recorder['lastRowAgoS'])}, camera failures {recorder['cameraFailures']} | "
            f"disk: {disk['shardsOpened']} shards, last write {ago(disk['lastWriteAgoS'])}"
        )


class StopFile:
    """The brake, as a file. Deliberately the dumbest mechanism that works from anywhere.

    A night runs in a `setsid` process that outlives the terminal that started it, and the thing
    that needs to stop it may be a web page, a shell on another machine, or a person who has just
    walked into the room. A file both of them can see needs no socket, no daemon, no protocol
    version, and survives the page being closed -- which is exactly the failure mode the card
    warned about: the operator's console is not part of the control loop and must not be.

    Poll-based rather than signal-based because this is the *gentle* brake: it is read at cycle
    boundaries, so the run ends holding the peg at carry height instead of somewhere a half-run
    cycle left it. The hard brake is SIGINT, which needs nothing from this class.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def request(self, reason: str = "") -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(reason or "stop requested", encoding="utf-8")

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)

    def requested(self) -> bool:
        # Existence, not content: a file that is being written while it is being read should still
        # stop the run, and no possible content means "carry on".
        return self._path.exists()

    def __call__(self) -> bool:
        return self.requested()


def write_session_header(root: Path | str, header: dict[str, Any]) -> Path:
    """The plan, written before the arm moves, next to the shards it is about to produce."""

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "session.json"
    target.write_text(json.dumps(header, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return target


def read_shards(root: Path | str) -> list[dict[str, Any]]:
    """Every shard under `root` with whether it was closed, for the offline pass and for QC."""

    out: list[dict[str, Any]] = []
    for shard_dir in sorted(Path(root).glob("shard_*")):
        footer_path = shard_dir / "footer.json"
        footer = json.loads(footer_path.read_text(encoding="utf-8")) if footer_path.exists() else None
        out.append({"path": shard_dir, "closed": footer is not None, "footer": footer})
    return out


def iter_rows(root: Path | str, *, closed_only: bool = False) -> Iterable[dict[str, Any]]:
    """Rows in order, so an offline pass never has to know how shards are named."""

    for shard in read_shards(root):
        if closed_only and not shard["closed"]:
            continue
        rows_path = shard["path"] / "rows.jsonl"
        if not rows_path.exists():
            continue
        with rows_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _is_scalar(value: Any) -> bool:
    """What belongs in a row's `state`: the numbers, not the pixels.

    Images arrive in the same observation dict as the pose, and a row that JSON-serialised one
    would write a 640x480 array into the log. Testing for the scalar types keeps arrays out by
    construction rather than by naming the camera keys, which would go stale the day a camera is
    renamed."""

    return isinstance(value, (bool, int, float, str))


def _jsonable(value: Any) -> Any:
    for attribute in ("tolist", "item"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001
                continue
    return str(value)
