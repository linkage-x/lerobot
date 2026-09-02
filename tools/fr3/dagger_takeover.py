"""Handing the arm to the operator mid-rollout, and marking what they did with it.

DAgger's whole value is that the correction happens on *the state the policy walked itself
into*. Re-recording demonstrations cannot reach those states: there is no frame in the dataset
of "the peg has already been knocked over", because a demonstrator never gets there. So the
correction has to be applied online, inside the same loop that is running the policy, and the
segment it produced has to come back labelled.

Four design decisions carry the rest of this file.

**The device decides when it is driving.** There is no engage button and no hold-to-take-over
key. Moving the SpaceMouse takes the arm; letting go of it hands the arm back once the device
has been quiet for ``release_after_s``. A button is a second thing to reach for at the moment
something is going wrong, and it is a thing that can be left in the wrong position -- a rollout
that ended with the takeover still latched hands the next one to a device nobody is holding.
The device's own deadbands (``threshold_x`` and friends in ``SpaceMouseTeleopConfig``, applied
after an idle-bias estimate taken at connect) already decide what counts as motion, so nothing
here adds a second threshold: any non-zero delta that reaches this file is real.

**One step is driven by the newest report the device has, and by nothing else.** Three
different things have to be true at once for the arm to follow the hand that is on the puck now.

*The scale is per report, and the report covers a step of real time.* ``translation_scale``
0.000615 m and ``rotation_scale`` 0.000648 rad are "per tick at 200 Hz" -- what the recorder's
loop applies each time round (7.4 deg/s of yaw at full deflection, as ``fr3_record_config.yaml``
says). So what the puck means is a *velocity*: full deflection is 0.000615 * 200 = 0.123 m/s, and
that is what it meant during every demonstration in the dataset. To mean the same thing here, a
step must move the arm by that velocity times the time the step actually lasted.
``motion_gain_for`` sizes one reading for the *nominal* step; ``step_period_s`` lets the loop
correct that by how long the step really took. Nominal alone is a speed nobody asked for whenever
the loop misses its rate -- a rollout at 60 Hz nominal and 40 Hz actual gives the same hand two
thirds of the recorded speed, and the operator cannot push harder because they are already
against the stop. The correction is bounded by ``MIN_STEP_PERIOD_RATIO`` and
``MAX_STEP_PERIOD_RATIO`` so a one-off stall arrives as a slow step and not as a lunge.

*The queue has to be emptied.* ``pyspacemouse`` opens the device non-blocking and takes one
report per read, while the kernel queues every report the device sends -- about 126 a second
with the puck displaced. A loop reading one per step at 30 Hz falls behind by ~96 reports a
second, so within a second the operator is steering with a hand they had half a second ago, and
when they let go the arm flies on through everything still queued. So every step drains the
queue. Only the *newest* report survives that drain: this is a rate control, the newest
deflection is the current one, and summing what the queue held would ask for as many times the
operator's motion as the step had reports in it -- which is the bug that made the arm run at the
clamp with no proportional control at all.

*A copy is not a report.* Once the queue is empty ``poll`` returns the device's cached state
rather than nothing, which is right for teleoperation -- a puck held off centre is a rate
command and should keep applying between reports -- and wrong the moment the hand comes off. The
device stops reporting; the cached state does not fall to zero unless the puck's own resting
position sits inside the deadband; so the loop re-applies that one reading 30 times a second.
The arm keeps flying at teleoperation speed with nobody touching anything, and the idle timer
that is supposed to hand it back is reset by every one of those copies, so it never runs out.
Measured in the first MuJoCo rehearsal: one takeover ran 1196 steps to the end of the rollout
and handed back 271 mm from where it engaged. The driver dates each report, and a timestamp
that has not moved means the device has said nothing since the last step -- which is what that
step then asks the arm to do. The gripper axis is deliberately exempt: a button held down
produces no new reports either, and it is the teleoperator's own state machine, not a stream of
reports, that keeps the fingers closing while it is held.

**One reference for both action sources.** The policy's action is a delta against ``prev_cmd``
-- the pose the last command asked for, not the pose the arm reached. The expert's delta is
defined against exactly the same thing, so the step guard in ``limit_command_for_safety``
bounds the two sources identically and the handoff in either direction is continuous: an
operator who has not yet moved the SpaceMouse re-issues the command that was already in
flight. Anchoring the expert to the *measured* pose instead would make taking over a step
backwards of the size of the tracking lag, which during fast motion is tens of millimetres --
a jolt at the exact moment the operator reached for the control because something was already
going wrong.

Integrating against the last *sent* command also removes windup for free. A reference that
integrated the expert's raw target would keep running while the clamp shortened what was
actually sent, and the operator -- who is closing the loop with their eyes -- would push
harder against a target that had already left the arm behind.

**Taking over does not by itself move the gripper.** The SpaceMouse reports an absolute
gripper position, so passing it through on the first engaged step would snap the gripper to
wherever the operator's buttons last left it. The failure this exists to correct happens
within a few centimetres of the object, often with something already between the fingers.
So the commanded gripper is held at whatever the policy last asked for until the operator
actually presses a button, and only then does the device take over that axis too.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

from lerobot.utils.rotation import Rotation

EE_POSITION_KEYS = ('ee.x', 'ee.y', 'ee.z')
EE_ROTVEC_KEYS = ('ee.wx', 'ee.wy', 'ee.wz')
GRIPPER_KEY = 'gripper.pos'

_DELTA_POSITION_KEYS = ('target_x', 'target_y', 'target_z')
_DELTA_ROTVEC_KEYS = ('target_wx', 'target_wy', 'target_wz')

# How far the device's gripper reading must move from what it read when the operator engaged
# before the device owns that axis. Below this it is the same button state, not a press: the
# incremental tool mode drifts by `incremental_step` per poll while a button is held, and the
# EMA in the teleoperator means even a released button settles rather than snaps.
_GRIPPER_TOUCH_EPS = 0.02

# Smaller than any increment the device can produce, larger than float noise. Only used to ask
# "did this reading move at all", which is what makes a button press count as taking over.
_GRIPPER_MOVED_EPS = 1e-9

# How long the operator has been driving before the arm goes back to the policy.
DEFAULT_RELEASE_AFTER_S = 1.0
# The puck is a rate control, not a stream of displacements, and the scales are calibrated to
# say so: `translation_scale` is metres *per recorder tick*, and the recorder ticks at
# `control_fps` (200) whether or not a fresh report arrived. Full deflection is therefore
# 0.000615 * 200 = 0.123 m/s, and a loop at any rate reproduces it by scaling one reading by
# `tick_hz * step_period_s`. Identity by default so a caller that has not thought about its own
# rate gets the raw reading rather than a silent speed change.
DEFAULT_MOTION_GAIN = 1.0

# How far the measured step may correct the nominal gain. A loop that has settled at half its
# nominal rate is a loop the operator should still be able to steer at the recorded speed, so the
# upper bound has to be well past 2. It exists only to bound the absurd: a one-second freeze must
# arrive as one slow step, not as a 123 mm lunge at the moment the process comes back. Below the
# lower bound the step was shorter than any pacing this loop does, which means the clock, not the
# hand, is the thing that moved.
MIN_STEP_PERIOD_RATIO = 0.5
MAX_STEP_PERIOD_RATIO = 4.0

# Reads allowed per control step while emptying the queue. The kernel's hidraw ring holds 64
# reports and drops the oldest when it overflows, so nothing older than this can still be waiting;
# one step is therefore always enough to catch up, and the cap only bounds a device that has
# started talking faster than it can be read.
MAX_READS_PER_STEP = 64


def motion_gain_for(*, tick_hz: float, step_period_s: float) -> float:
    """Scale one reading so this loop covers the same ground per second as the recorder.

    One reading, however many reads it took to reach it. The extra reads empty the queue -- the
    device sends ~126 reports a second and the driver hands over one per read -- but they are not
    extra motion: past the end of the queue ``PySpaceMouseDriver.poll`` returns the device's last
    state rather than nothing (measured on the rig: 0 empty returns in 620k reads over 2 s), so
    summing what a step read multiplied the operator's hand by however many reads it had time
    for -- 32, at the old cap.
    """
    return max(0.0, float(tick_hz)) * max(0.0, float(step_period_s))


def backend_dates_reports(teleop: Any) -> bool:
    """Whether this driver stamps each report with when the device sent it.

    The question is structural, not a reading: ``last_report_timestamp`` is None until the first
    report arrives, and at startup that is every driver -- the operator is not on the puck yet. So
    what separates a driver that *can* date its reports from one that cannot is whether it carries
    the attribute at all.

    It matters because the degraded path is silent. Without dates :meth:`_newest_report` cannot
    tell a fresh report from the cached copy the driver returns past the end of the queue, so it
    takes one read per step and calls it new -- which is the pre-fix behaviour in full: the arm
    follows a hand from a second ago, and the release timer is reset by every copy, so the arm is
    never handed back. Nothing raises and nothing in the pose looks wrong.
    """
    return hasattr(teleop, 'last_report_timestamp')


def undated_backend_error(*, driver_module: str | None = None) -> str:
    """Why a real rollout will not start on an undated driver, and what to do about it.

    Written out here rather than at the call site because the cause is almost never the device:
    it is which copy of ``lerobot`` the process imported. The fix is a path, so the message has to
    carry the path it actually loaded.
    """
    loaded = f'\n  loaded from: {driver_module}' if driver_module else ''
    return (
        'the SpaceMouse driver does not date its reports (no last_report_timestamp), so a takeover '
        'cannot tell a new report from the cached copy the driver repeats once the queue is empty. '
        'That is the runaway this fix removed: the arm keeps flying at teleoperation speed after '
        'the hand comes off, and the release timer never expires because every copy resets it. '
        'Refusing rather than running it near a real arm.'
        f'{loaded}\n'
        '  Cause: this process imported a copy of lerobot without the driver patch. Run through '
        'tools/fr3/run_pick_place_infer_workstation.sh, which exports PYTHONPATH=$PWD/src, or set '
        'it yourself so this repo\'s src/ comes first.\n'
        '  The MuJoCo rehearsal (dagger_sim) still runs on an undated driver -- there is no arm '
        'there to run away -- but its handback numbers do not transfer to the rig.'
    )


class DeviceInput(NamedTuple):
    """What the device said during one control step."""

    delta_position: np.ndarray
    delta_rotvec: np.ndarray
    gripper: float | None
    gripper_before: float | None
    gripper_moved: bool
    moved: bool
    # Whether this step's reading was a new report rather than the driver's cached copy of the
    # last one. False means the device has been silent since the previous step.
    fresh: bool = True
    # How many reads it took to reach it. More than a couple means the queue had backed up, which
    # is worth seeing in a log: it is the arm following a hand from several steps ago.
    reads: int = 1

    @property
    def eventful(self) -> bool:
        """Whether the operator touched the device at all. Motion or a button, either counts."""
        return bool(self.moved or self.gripper_moved)


def _finite_vector(action: Mapping[str, Any], keys: tuple[str, str, str]) -> np.ndarray:
    """The three deltas under `keys`, or zeros if any of them is not a finite number.

    Zeroing the whole vector rather than the offending axis: a device that reports NaN on one
    axis is not reporting a two-axis motion, it is not reporting.
    """
    values = np.array([float(action.get(key, 0.0) or 0.0) for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return np.zeros(3, dtype=np.float64)
    return values


def _pose_from_command(command: Mapping[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.array([float(command[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
    rotation = Rotation.from_rotvec(np.array([float(command[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64))
    return position, rotation


def _pose_from_observation(observation: Mapping[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.array([float(observation[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
    rotation = Rotation.from_rotvec(np.array([float(observation[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64))
    return position, rotation


def expert_spans(sources: list[str], *, expert_label: str = 'expert') -> list[tuple[int, int]]:
    """The inclusive `(first, last)` step of each stretch the operator was driving.

    Reduced from the per-step source column rather than recorded separately while it happens.
    The trace file is the thing that survives the rollout, so a span the marker reports and the
    CSV does not is a span that cannot be checked afterwards -- and two records of one event is
    two things that can disagree.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, source in enumerate(sources):
        if source == expert_label:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(sources) - 1))
    return spans


class ExpertTakeover:
    """Turns SpaceMouse motion into the same command shape the policy produces.

    Also decides *when* the operator is driving: see the module docstring. The caller passes
    ``latched``, which is the manual override (a key, for an operator who wants the arm held
    still while they think). Everything else follows from what the device reports.
    """

    def __init__(
        self,
        teleop: Any,
        *,
        release_after_s: float = DEFAULT_RELEASE_AFTER_S,
        motion_gain: float = DEFAULT_MOTION_GAIN,
        step_period_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._teleop = teleop
        # `monotonic`, not `time()`: this is a duration between two events seconds apart in one
        # process, and a wall clock that steps backwards mid-rollout would hand the arm over.
        self._clock = clock
        self._release_after_s = max(0.0, float(release_after_s))
        self._motion_gain = max(0.0, float(motion_gain))
        # The step `motion_gain` was sized for. None means the caller has not told this class what
        # its loop rate is supposed to be, so there is nothing to measure the real step against and
        # the fixed gain stands.
        self._step_period_s = None if step_period_s is None else max(0.0, float(step_period_s))
        # When `command` was last called. The interval between two calls is the time the command
        # issued by the first one was held for, which is the interval the operator's velocity has
        # to be integrated over.
        self._last_command_s: float | None = None
        self._engaged = False
        self._gripper_hold: float = 0.0
        self._gripper_at_engage: float | None = None
        self._gripper_owned = False
        self._last_gripper_seen: float | None = None
        # The report the previous step's motion came from. A reading carrying this same
        # timestamp is a copy, not the operator.
        self._last_report_timestamp: float | None = None
        # Never touched, rather than "touched at time zero": on the first step the arm belongs
        # to the policy, and `inf` seconds of idle is the honest way to say so.
        self._last_input_s = float('-inf')
        self._poll_failed = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def auto_enabled(self) -> bool:
        return self._release_after_s > 0.0

    @property
    def dates_reports(self) -> bool:
        """Whether the device under this takeover can say which of its reads were new.

        Asked of the object rather than re-derived by callers, because the answer decides how to
        read everything else it produces: without dates the handback timer is being reset by
        copies, and a gap measured on that path is not the operator's excursion.
        """
        return backend_dates_reports(self._teleop)

    def _newest_report(self) -> tuple[Mapping[str, Any], bool, int]:
        """Empty the device's queue; keep the last report out of it and whether it was new.

        The driver hands over one queued report per read and the device sends about 126 a second,
        so a step that reads once falls behind by three or four reports every time round. Past the
        end of the queue the same read starts returning the device's cached state, dated with the
        timestamp it already had -- which is how the end of the queue is recognised, and how a
        device that has gone quiet is told apart from one the operator is still pushing.

        Only the newest report is returned. Everything the drain passed over is a deflection the
        operator has already moved on from; the alternative -- adding them up -- is what made a
        puck off centre by any amount ask for more motion than the step guard would pass.
        """
        action = self._teleop.get_action()
        timestamp = getattr(self._teleop, 'last_report_timestamp', None)
        if timestamp is None:
            # An undated backend cannot say which of its reads were copies, so one read is all
            # that can be trusted from it: draining it would be back to summing copies, and
            # refusing to act on a reading that might be new would leave the operator pushing a
            # dead puck.
            return action, True, 1

        fresh = timestamp != self._last_report_timestamp
        reads = 1
        while reads < MAX_READS_PER_STEP:
            next_action = self._teleop.get_action()
            next_timestamp = getattr(self._teleop, 'last_report_timestamp', None)
            reads += 1
            if next_timestamp is None or next_timestamp == timestamp:
                break
            action, timestamp, fresh = next_action, next_timestamp, True
        self._last_report_timestamp = timestamp
        return action, fresh, reads

    def _gain_for_step(self, now: float) -> tuple[float, float | None]:
        """The gain for this step, and how long the previous one lasted.

        The puck is a velocity, so the distance it asks for is that velocity times the time the
        command will be held -- which is not known until the next step arrives, so the previous
        interval stands in for it. On a loop that keeps its rate the two are the same number; on
        one that does not, this is the difference between the recorded hand and a slower one.

        The first call of a run has no previous interval, and the gap across a stopped rollout is
        not a control step at all. Both fall back to the nominal gain rather than inventing a
        measurement.
        """
        previous = self._last_command_s
        self._last_command_s = now
        if self._step_period_s is None or self._step_period_s <= 0.0 or previous is None:
            return self._motion_gain, None
        elapsed_s = now - previous
        if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            return self._motion_gain, None
        ratio = min(max(elapsed_s / self._step_period_s, MIN_STEP_PERIOD_RATIO), MAX_STEP_PERIOD_RATIO)
        return self._motion_gain * ratio, elapsed_s

    def _read_device(self, *, motion_gain: float) -> DeviceInput:
        """One reading, rescaled from the recorder's per-tick calibration to this loop's step.

        The reading is the newest report the device had (see :meth:`_newest_report`), and it moves
        the arm only if it is one the operator produced since the last step. A repeat means the
        device has said nothing, and nothing is what it asks for.
        """
        action, fresh, reads = self._newest_report()
        gripper_before = self._last_gripper_seen

        reported_position = _finite_vector(action, _DELTA_POSITION_KEYS) if fresh else np.zeros(3)
        reported_rotvec = _finite_vector(action, _DELTA_ROTVEC_KEYS) if fresh else np.zeros(3)
        # Measured before the gain: whether the operator moved is a property of the reading, and
        # a gain of zero must not read as a hand that let go.
        moved = bool(np.any(reported_position != 0.0) or np.any(reported_rotvec != 0.0))

        gripper = gripper_before
        gripper_moved = False
        reading = action.get('gripper')
        if reading is not None and np.isfinite(float(reading)):
            reading = float(np.clip(float(reading), 0.0, 1.0))
            gripper_moved = (
                self._last_gripper_seen is not None
                and abs(reading - self._last_gripper_seen) > _GRIPPER_MOVED_EPS
            )
            self._last_gripper_seen = reading
            gripper = reading

        return DeviceInput(
            delta_position=reported_position * motion_gain,
            delta_rotvec=reported_rotvec * motion_gain,
            gripper=gripper,
            gripper_before=gripper_before,
            gripper_moved=gripper_moved,
            moved=moved,
            fresh=fresh,
            reads=reads,
        )

    def _engage(
        self,
        *,
        reason: str,
        device_input: DeviceInput | None,
        previous_sent_command: Mapping[str, float] | None,
        robot_observation: Mapping[str, float],
    ) -> None:
        self._engaged = True
        self._gripper_owned = False
        # What the device read *before* this step. An operator who took the arm by pressing a
        # gripper button has already moved it by one increment by the time we get here; measuring
        # ownership from the post-press value would need a second press to notice the first.
        self._gripper_at_engage = None if device_input is None else device_input.gripper_before
        if previous_sent_command is not None and GRIPPER_KEY in previous_sent_command:
            self._gripper_hold = float(previous_sent_command[GRIPPER_KEY])
        else:
            self._gripper_hold = float(robot_observation.get(GRIPPER_KEY, 0.0))
        print(f'[INFO] dagger_takeover=engaged reason={reason} gripper_hold={self._gripper_hold:.3f}')

    def _release(self, *, reason: str, idle_s: float) -> None:
        self._engaged = False
        self._gripper_owned = False
        self._gripper_at_engage = None
        idle_text = '' if not np.isfinite(idle_s) else f' idle_s={idle_s:.2f}'
        print(f'[INFO] dagger_takeover=released reason={reason}{idle_text}')

    def _held_command(
        self,
        policy_command: dict[str, float],
        reference_position: np.ndarray,
        reference_rotation: Rotation,
    ) -> dict[str, float]:
        """The arm stays where the last command put it. Used when the device cannot be read."""
        held = dict(policy_command)
        rotvec = reference_rotation.as_rotvec()
        held.update(
            {
                'ee.x': float(reference_position[0]),
                'ee.y': float(reference_position[1]),
                'ee.z': float(reference_position[2]),
                'ee.wx': float(rotvec[0]),
                'ee.wy': float(rotvec[1]),
                'ee.wz': float(rotvec[2]),
                GRIPPER_KEY: float(self._gripper_hold),
            }
        )
        return held

    def command(
        self,
        *,
        latched: bool = False,
        policy_command: dict[str, float],
        previous_sent_command: Mapping[str, float] | None,
        robot_observation: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """The command for this step, plus what it was and why.

        Returns the policy's own command untouched whenever the operator is not driving, so a
        run with a SpaceMouse plugged in and never touched is byte-identical to one without it.
        """
        now = self._clock()
        motion_gain, elapsed_s = self._gain_for_step(now)
        device_input: DeviceInput | None = None
        try:
            device_input = self._read_device(motion_gain=motion_gain)
        except Exception as exc:  # noqa: BLE001 - a device read must never end a rollout mid-motion
            if not self._poll_failed:
                self._poll_failed = True
                print(f'[WARN] dagger_takeover_poll_failed: {exc}')
        else:
            self._poll_failed = False
            if device_input.eventful:
                self._last_input_s = now

        idle_s = now - self._last_input_s
        if device_input is None:
            # A dead device does not hand the arm back on its own. If the operator was driving,
            # they still are as far as they know, and the alternative -- the policy resuming
            # under their hand -- is the one surprise this path exists to rule out. They hold
            # position (below) until they stop the rollout.
            auto_engaged = self._engaged
        else:
            auto_engaged = self.auto_enabled and idle_s < self._release_after_s
        wanted = bool(latched) or auto_engaged

        if wanted and not self._engaged:
            self._engage(
                reason='latched' if latched and not auto_engaged else 'motion',
                device_input=device_input,
                previous_sent_command=previous_sent_command,
                robot_observation=robot_observation,
            )
        elif not wanted and self._engaged:
            self._release(reason='idle' if self.auto_enabled else 'unlatched', idle_s=idle_s)

        if not self._engaged:
            return policy_command, {'source': 'policy', 'engaged': False, 'idle_s': idle_s}

        # The arm holds where the last command put it. Used both as the delta reference and as
        # the fallback when the device cannot be read.
        if previous_sent_command is not None:
            reference_position, reference_rotation = _pose_from_command(previous_sent_command)
        else:
            reference_position, reference_rotation = _pose_from_observation(robot_observation)

        if device_input is None:
            held = self._held_command(policy_command, reference_position, reference_rotation)
            return held, {'source': 'expert', 'engaged': True, 'status': 'poll_failed'}

        target_position = reference_position + device_input.delta_position
        # Right-multiplied, matching the recorder's own delta convention
        # (processor_franka_research3: `desired_R = reference_R @ delta_R`). A left-multiplied
        # rotation here would mean the SpaceMouse turned the tool about the base axes during
        # takeover and about the tool axes during recording -- the same device, two behaviours.
        target_rotation = reference_rotation * Rotation.from_rotvec(device_input.delta_rotvec)
        target_rotvec = target_rotation.as_rotvec()

        if device_input.gripper is None:
            gripper_command = float(self._gripper_hold)
        else:
            if self._gripper_at_engage is None:
                self._gripper_at_engage = device_input.gripper
            if (
                not self._gripper_owned
                and abs(device_input.gripper - self._gripper_at_engage) >= _GRIPPER_TOUCH_EPS
            ):
                self._gripper_owned = True
                print(f'[INFO] dagger_takeover=gripper_taken value={device_input.gripper:.3f}')
            gripper_command = device_input.gripper if self._gripper_owned else float(self._gripper_hold)

        expert_command = dict(policy_command)
        expert_command.update(
            {
                'ee.x': float(target_position[0]),
                'ee.y': float(target_position[1]),
                'ee.z': float(target_position[2]),
                'ee.wx': float(target_rotvec[0]),
                'ee.wy': float(target_rotvec[1]),
                'ee.wz': float(target_rotvec[2]),
                GRIPPER_KEY: float(gripper_command),
            }
        )
        return expert_command, {
            'source': 'expert',
            'engaged': True,
            # `stale` is not `holding`: holding is a puck at rest under a hand that is still
            # there, stale is a device that has stopped speaking. On the rig they are the
            # difference between "the operator is thinking" and "the operator has let go".
            'status': ('moving' if device_input.moved else 'holding' if device_input.fresh else 'stale'),
            'moved': device_input.moved,
            'gripper_owned': self._gripper_owned,
            'step_mm': float(np.linalg.norm(device_input.delta_position) * 1000.0),
            'reads': int(device_input.reads),
            # What one reading was multiplied by, and the step it was measured over. Steadily
            # above the nominal gain means the loop is not keeping its rate, and the operator is
            # being given back the speed that costs them -- read it next to `loop_ms`, which
            # measures the same thing from the other end.
            'gain': float(motion_gain),
            'step_ms': 0.0 if elapsed_s is None else float(elapsed_s * 1000.0),
            'idle_s': idle_s,
            'latched': bool(latched),
        }

    def close(self) -> None:
        """Release the device. Never raises: this runs in the same `finally` that disconnects
        the robot, and a teleoperator that will not let go must not stop the arm from being
        put down safely."""
        self._engaged = False
        try:
            self._teleop.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] dagger_takeover_disconnect_failed: {exc}')
