"""Driving a checkpoint on the real arm: what to launch, and how to read what comes back.

The command half is deliberately thin. `tools/fr3/run_pick_place_infer_workstation.sh` already
encodes what this rig is -- its robot IP, its gripper, its cameras, its safety envelope, and a
long comment explaining which of those may not be overridden by environment alone. Rebuilding
that argument list here would create a second definition of the rig that could drift from the
one operators use from a terminal, and the failure mode of a drifted rollout is silent. So this
module sets the launcher's documented environment variables and appends the two flags the
browser path needs, and the launcher stays the single description of the hardware.

The parsing half turns the runtime's own log lines into page state. Those lines are a contract
the runtime prints for humans; the markers matched here are the ones it emits unconditionally.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The runtime writes these lines; this is its own reader for them. Imported rather than
# re-matched here because a live frame is the one runtime line whose format is machine-chosen
# on both ends -- a second regex for it would be a second definition of the wire format.
from tools.fr3.live_frames import parse_live_frame  # noqa: F401  (re-exported for the gateway)
# Imported rather than re-implemented: the page and the command line must refuse the same poses,
# and a second copy of "three finite metres" is a second thing to keep in step.
from tools.fr3.terminal_servo import TerminalServoError, parse_terminal_servo_pose

LAUNCHER = Path("tools/fr3/run_pick_place_infer_workstation.sh")

# Where the runtime publishes the frames it is feeding the policy. /dev/shm rather than the
# repo: these are written several times a second for the length of a rollout and are worthless
# once it ends, so they should never touch a disk or survive a reboot.
PREVIEW_DIR = Path("/dev/shm/lerobot_rollout_preview")
PREVIEW_FPS = 5.0
# One directory per launch, because the runtime numbers its traces from 1 every time it starts
# and writes `rollout_001.csv` with no regard for what is already there. Left to the runtime's
# default (a single flat `outputs/rollout_traces`) every browser session silently overwrites the
# last one -- which it did on 2026-09-01, taking four traces of the graded 08-31 batch with it.
TRACE_ROOT = Path("outputs/rollout_traces")


def trace_session_dir(repo_root: Path, stamp: str) -> Path:
    """Where this launch's per-rollout CSVs go. `stamp` is the launch's, shared with its log."""
    return repo_root / TRACE_ROOT / f"session_{stamp}"
# A frame older than this is not a live view. Rollouts run at the dataset's rate, well above the
# preview rate, so the only way to exceed this is a run that has stopped producing frames.
PREVIEW_STALE_S = 3.0
# Stills taken while the arm stands at a commanded calibration point, plus the sidecar naming
# the request each one belongs to. A subdirectory of the preview dir so one runtime argument
# still places everything this process publishes, and so both die with the same reboot.
PROBE_DIR = PREVIEW_DIR / "probe"
PROBE_SIDECAR_PATH = PROBE_DIR / "probe.json"

# Runtime knobs the browser owns. They are cleared from any inherited environment before the
# page applies its explicit choices, because stale shell values are otherwise indistinguishable
# from operator intent once the launcher starts.
ROLLOUT_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "FR3_TASK_PROMPT",
    "FR3_ACT_TEMPORAL_ENSEMBLE_COEFF",
    "FR3_RTC_MODE",
    "FR3_RTC_EXECUTION_HORIZON",
    "FR3_RTC_MAX_GUIDANCE_WEIGHT",
    "FR3_RTC_PREFIX_ATTENTION_SCHEDULE",
    "FR3_RTC_REPLAN_QUEUE_SIZE",
    "FR3_RTC_INFERENCE_DELAY_STEPS",
    "FR3_COMMAND_EMA_ALPHA",
    # Sampling aggregation (E3) and the terminal servo (E5). Cleared with the rest for the same
    # reason the DAgger keys are: a shell that once exported FR3_TERMINAL_SERVO_POSE would
    # otherwise take the arm off the policy in a rollout the browser never asked it to.
    "FR3_ACTION_SAMPLES",
    "FR3_ACTION_AGGREGATE",
    "FR3_ACTION_SAMPLE_HORIZON",
    "FR3_TERMINAL_SERVO_POSE",
    "FR3_TERMINAL_SERVO_HANDOFF_Z",
    "FR3_TERMINAL_SERVO_SEARCH_RING",
    # DAgger takeover. Cleared from the inherited environment like the rest, and for a sharper
    # reason: a shell that once exported FR3_DAGGER_TAKEOVER=1 would otherwise open a second
    # action source onto a moving arm in a rollout the browser never asked to be steerable.
    "FR3_DAGGER_TAKEOVER",
    "FR3_DAGGER_DATASET_ROOT",
    "FR3_DAGGER_RELEASE_AFTER_S",
)
RTC_MODES = {"auto", "enabled", "disabled"}
ACTION_AGGREGATES = {"medoid", "mean"}

# Where corrections go when the operator turns takeover on and names no directory. One directory
# per checkpoint, not per launch: a DAgger dataset is only worth training on once it has enough
# corrections in it, and the states being corrected are the states *this* policy walks into. A
# per-launch directory -- which is right for traces, whose whole job is to stay separable -- would
# scatter one afternoon's corrections across six datasets too small to train on.
#
# Under `outputs/datasets` and not a directory of its own, because the gateway scans exactly one
# level of that root for dataset roots (`_scan_datasets_root`). Anywhere else and the corrections
# are invisible to the export page -- which is the page that merges them with the demonstrations
# into the view the next checkpoint trains on. A dataset no tool on this machine can select is a
# dataset that was not collected.
DAGGER_ROOT = Path("outputs/datasets")
DAGGER_PREFIX = "dagger_"

# What `FR3_DAGGER_DATASET_ROOT` set to the empty string means between `sanitize_rollout_runtime_options`
# and `build_rollout_command`: the operator answered the question and the answer was "nowhere".
# Absent means they did not answer, which is the case that gets `dagger_dataset_dir`. The
# distinction matters because the two must not collapse into one: a blank field that silently
# meant "discard the corrections" is the failure the trace directory already taught us.
DAGGER_STEER_ONLY = ""


def dagger_dataset_dir(repo_root: Path, checkpoint_id: str) -> Path:
    """The default dataset for corrections made against `checkpoint_id`.

    Same flattening the log file uses, so the two land next to each other under names an operator
    can pair up by eye.
    """
    flattened = checkpoint_id.replace("/", "_") or "unknown_checkpoint"
    return repo_root / DAGGER_ROOT / f"{DAGGER_PREFIX}{flattened}"
RTC_PREFIX_ATTENTION_SCHEDULES = {"EXP", "LINEAR", "ONES", "ZEROS"}


@dataclass(frozen=True)
class RolloutMode:
    """One launcher mode, described by the two things an operator must know before pressing it.

    `movesArm` gates the confirmation the page requires. `interactive` decides whether the run
    is steered afterwards or simply runs to its end.

    `takeover` is narrower than `interactive`: it is whether the *launcher* forwards the DAgger
    flags to this mode. It does so only for `real` and `real_debug`, because the runtime refuses
    `--dagger-takeover` without `--interactive-rollouts` and the launcher keeps the flags out of
    the modes that would trip that. Carried here so the page can grey the switch out, and so the
    request can be refused with a reason -- `real_once` would otherwise accept the setting, drop
    it in the launcher's `case`, and run a rollout the operator believed they could steer.
    """

    id: str
    label: str
    description: str
    movesArm: bool
    interactive: bool
    takeover: bool = False


ROLLOUT_MODES: tuple[RolloutMode, ...] = (
    RolloutMode(
        "env",
        "Show resolved settings",
        "Prints the checkpoint, cameras, tool frame, gripper and safety envelope this rollout "
        "would use, and exits. Touches no hardware.",
        movesArm=False,
        interactive=False,
    ),
    RolloutMode(
        "smoke",
        "Smoke (1 step, no motion)",
        "Proves the checkpoint loads, both cameras open under the names the policy asks for, "
        "and one forward pass produces a decodable action. The arm does not move.",
        movesArm=False,
        interactive=False,
    ),
    RolloutMode(
        "preview",
        "Preview (20 steps, no motion)",
        "Homes the arm, then runs 20 policy steps without sending them. Shows what the policy "
        "would do before it is allowed to do it.",
        movesArm=True,
        interactive=False,
    ),
    RolloutMode(
        "real_once",
        "One bounded rollout",
        "Homes the arm and runs a single rollout to a step limit. The arm moves.",
        movesArm=True,
        interactive=False,
    ),
    RolloutMode(
        "real",
        "Interactive rollouts",
        "Homes the arm, then waits. Each Start runs one rollout; Stop ends the current one and "
        "returns to waiting. The arm moves.",
        movesArm=True,
        interactive=True,
        takeover=True,
    ),
    RolloutMode(
        "dagger_sim",
        "DAgger rehearsal (MuJoCo, no arm)",
        "Replays a recorded episode through the simulated arm and lets the operator take over "
        "mid-episode with the SpaceMouse. Rehearses the handoff -- clamp, gripper hold, "
        "handback -- with no hardware. The checkpoint is used only to find the dataset the "
        "episode comes from; no weights are loaded. Watch it in the live 3D view.",
        movesArm=False,
        interactive=True,
    ),
    RolloutMode(
        "real_debug",
        "Interactive + MuJoCo viewer",
        "Interactive rollouts with a MuJoCo window on the rig showing current EE, raw policy "
        "target, clamped target and the action chunk. Needs a display on the rig machine.",
        movesArm=True,
        interactive=True,
        takeover=True,
    ),
)

MODES_BY_ID = {mode.id: mode for mode in ROLLOUT_MODES}


class RolloutError(RuntimeError):
    """Something the operator can fix, reported as a 4xx rather than a traceback."""


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_int_field(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise RolloutError(f"{field} must be an integer, not a boolean.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise RolloutError(f"{field} must be an integer.")
        parsed = int(text)
    if parsed < minimum:
        raise RolloutError(f"{field} must be >= {minimum}.")
    return parsed


def _parse_float_field(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RolloutError(f"{field} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RolloutError(f"{field} must be a number.") from exc
    if not math.isfinite(parsed):
        raise RolloutError(f"{field} must be finite.")
    if minimum is not None and parsed < minimum:
        raise RolloutError(f"{field} must be >= {minimum:g}.")
    return parsed


def _set_optional_int_env(
    options: dict[str, str],
    raw: dict[str, Any],
    field: str,
    env_name: str,
    *,
    minimum: int,
) -> None:
    if field not in raw or raw[field] in (None, ""):
        return
    options[env_name] = str(_parse_int_field(raw[field], field, minimum=minimum))


def _set_optional_float_env(
    options: dict[str, str],
    raw: dict[str, Any],
    field: str,
    env_name: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if field not in raw or raw[field] in (None, ""):
        return None
    parsed = _parse_float_field(raw[field], field, minimum=minimum)
    options[env_name] = f"{parsed:g}"
    return parsed


def _parse_bool_field(value: Any, field: str) -> bool:
    """A switch, from whatever JSON the page sent for it.

    Strings are accepted because the same options dict round-trips through `last_params.json`
    and may be edited by hand there; anything that is not recognisably a yes or a no is refused
    rather than treated as off, because for a takeover switch "off" is the answer that runs.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise RolloutError(f"{field} must be true or false; got {value!r}.")


def sanitize_rollout_runtime_options(raw: Any) -> dict[str, str]:
    """Validate the Rollout page's optional runtime knobs and return launcher env values."""
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise RolloutError("runtimeOptions must be an object.")

    options: dict[str, str] = {}
    task_prompt = _optional_text(raw.get("taskPrompt"))
    if task_prompt:
        options["FR3_TASK_PROMPT"] = task_prompt

    rtc_mode = _optional_text(raw.get("rtcMode"))
    if rtc_mode:
        rtc_mode = rtc_mode.lower()
        if rtc_mode not in RTC_MODES:
            raise RolloutError(
                f"rtcMode must be one of {', '.join(sorted(RTC_MODES))}; got {raw.get('rtcMode')!r}."
            )
        options["FR3_RTC_MODE"] = rtc_mode

    _set_optional_int_env(
        options, raw, "rtcExecutionHorizon", "FR3_RTC_EXECUTION_HORIZON", minimum=1
    )
    _set_optional_float_env(
        options, raw, "rtcMaxGuidanceWeight", "FR3_RTC_MAX_GUIDANCE_WEIGHT", minimum=0.0
    )

    schedule = _optional_text(raw.get("rtcPrefixAttentionSchedule"))
    if schedule:
        schedule = schedule.upper()
        if schedule not in RTC_PREFIX_ATTENTION_SCHEDULES:
            raise RolloutError(
                "rtcPrefixAttentionSchedule must be one of "
                f"{', '.join(sorted(RTC_PREFIX_ATTENTION_SCHEDULES))}; got "
                f"{raw.get('rtcPrefixAttentionSchedule')!r}."
            )
        options["FR3_RTC_PREFIX_ATTENTION_SCHEDULE"] = schedule

    _set_optional_int_env(
        options, raw, "rtcReplanQueueSize", "FR3_RTC_REPLAN_QUEUE_SIZE", minimum=1
    )
    _set_optional_int_env(
        options, raw, "rtcInferenceDelaySteps", "FR3_RTC_INFERENCE_DELAY_STEPS", minimum=0
    )
    ema = _set_optional_float_env(
        options, raw, "commandEmaAlpha", "FR3_COMMAND_EMA_ALPHA", minimum=0.0
    )
    if ema is not None and ema > 1.0:
        raise RolloutError("commandEmaAlpha must be <= 1.")

    _set_optional_int_env(options, raw, "actionSamples", "FR3_ACTION_SAMPLES", minimum=1)
    aggregate = _optional_text(raw.get("actionAggregate"))
    if aggregate:
        aggregate = aggregate.lower()
        if aggregate not in ACTION_AGGREGATES:
            raise RolloutError(
                f"actionAggregate must be one of {', '.join(sorted(ACTION_AGGREGATES))}; "
                f"got {raw.get('actionAggregate')!r}."
            )
        options["FR3_ACTION_AGGREGATE"] = aggregate
    _set_optional_int_env(
        options, raw, "actionSampleHorizon", "FR3_ACTION_SAMPLE_HORIZON", minimum=0
    )

    # E5. Refused here as well as at startup, because the page is where a mistyped pose is
    # cheapest to catch -- the alternative is a launcher that comes up, homes the arm and then
    # exits on its own argument.
    servo_pose = _optional_text(raw.get("terminalServoPose"))
    if servo_pose:
        try:
            parse_terminal_servo_pose(servo_pose)
        except TerminalServoError as exc:
            raise RolloutError(f"terminalServoPose is not usable: {exc}") from exc
        options["FR3_TERMINAL_SERVO_POSE"] = servo_pose
    _set_optional_float_env(
        options, raw, "terminalServoHandoffZ", "FR3_TERMINAL_SERVO_HANDOFF_Z", minimum=0.0
    )
    # E7 route C. Blank leaves the search off, which is E5 exactly -- the arm descends once at
    # the pose above and reports where it stopped. 0.007 is the ring the covering was worked out
    # for; the runtime refuses anything the growth test could not arm inside.
    _set_optional_float_env(
        options, raw, "terminalServoSearchRing", "FR3_TERMINAL_SERVO_SEARCH_RING", minimum=0.0
    )

    if _parse_bool_field(raw.get("daggerTakeover", False), "daggerTakeover"):
        options["FR3_DAGGER_TAKEOVER"] = "1"
        if _parse_bool_field(raw.get("daggerRecord", True), "daggerRecord"):
            root = _optional_text(raw.get("daggerDatasetRoot"))
            if root:
                if any(character in root for character in "\n\r\0"):
                    raise RolloutError("daggerDatasetRoot must be a single path with no line breaks.")
                options["FR3_DAGGER_DATASET_ROOT"] = root
            # Left absent otherwise, for `build_rollout_command` to fill from the checkpoint. See
            # DAGGER_STEER_ONLY for why the blank field is not allowed to mean "nowhere".
        else:
            options["FR3_DAGGER_DATASET_ROOT"] = DAGGER_STEER_ONLY
        # 0 is meaningful: it turns off the automatic handback, leaving the latch as the only way
        # in and out. Worth allowing from the page because it is how an operator rehearses the
        # handoff without the arm deciding for them.
        _set_optional_float_env(
            options, raw, "daggerReleaseAfterS", "FR3_DAGGER_RELEASE_AFTER_S", minimum=0.0
        )
    return options


@dataclass
class RolloutStatus:
    # idle | starting | waiting | finishing | homing | resetting | rolling | complete | error | stopped
    #
    # `waiting` is the narrow one and the load-bearing one: it means the runtime is sitting in
    # `InteractiveRolloutKeyboard.wait_for_command`, which is the only place it can act on a
    # command. It is set by exactly one marker, `interactive_waiting_for_start`, and by nothing
    # else -- see `parse_rollout_line`. `finishing` is what the states that used to claim
    # `waiting` say instead: the activity is over, the runtime is on its way back to the gate,
    # and anything sent before it arrives is dropped on the floor.
    state: str = "idle"
    mode: str = ""
    checkpointId: str = ""
    checkpointPath: str = ""
    policy: str = ""
    datasetRoot: str = ""
    targetFrameName: str = ""
    robotIp: str = ""
    cameraKeys: list[str] = field(default_factory=list)
    interactive: bool = False
    movesArm: bool = False
    # Whether a SpaceMouse was opened, i.e. whether moving it will take the arm over. Read off
    # the takeover key on the runtime's own control-channel line rather than inferred from the
    # mode, because it is the runtime that decides: it refuses to bind that key when no device
    # was opened. The page uses it to say the device is armed, not to offer a button -- takeover
    # engages itself when the device moves.
    takeoverAvailable: bool = False
    # The takeover pre-flight, read off the runtime's `dagger_takeover=ready` banner. `yes` is the
    # only value a real rollout can print -- an undated driver is refused before the arm is even
    # constructed -- so this is not a warning light, it is the operator's confirmation that the
    # check ran. Empty means takeover was never armed for this launch.
    daggerReportTimestamps: str = ""
    # Seconds of a still device before the policy takes the arm back. 0 means automatic handback
    # is off and only the hold latch moves the arm between the two drivers -- which is why the
    # unknown case is None and not 0.0: the page says something different for each, and "we have
    # not read the banner yet" must not render as "the arm will not hand itself back".
    daggerReleaseAfterS: float | None = None
    # Where corrections are being written, or empty when the operator chose to steer without
    # recording. Shown for the reason `tracePath` is: an afternoon of corrections nobody can find
    # afterwards is an afternoon of corrections nobody trains on.
    daggerDatasetPath: str = ""
    # Summed over the session, not per rollout: the question the operator is actually asking is
    # "have I collected enough corrections to retrain yet", which no single rollout answers.
    daggerEpisodes: int = 0
    # Frames a rollout produced past --dagger-max-buffered-frames and had to drop. Non-zero means
    # a correction was longer than the buffer, i.e. the dataset is missing the end of it.
    daggerDroppedFrames: int = 0
    step: int = 0
    maxSteps: int = 0
    commandStatus: str = ""
    # Two different events, deliberately counted apart. A step-limited command means the policy
    # asked for more motion in one tick than the demonstrations ever contained. A leash-limited one
    # means the command is running away from an arm that is not following it -- a much louder
    # signal, and the only one of the two that suggests stopping.
    clampedSteps: int = 0
    leashedSteps: int = 0
    rolloutIndex: int = 0
    lastRolloutStatus: str = ""
    # Whether the arm is at the pose the demonstrations started from. False from the moment a
    # rollout ends until the operator homes it again -- the launcher's homing step runs once,
    # before the runtime exists, so nothing else puts this back to true on its own.
    armAtStart: bool = False
    pid: int | None = None
    message: str = "Pick a checkpoint and a mode to start."
    startedAt: str = ""
    finishedAt: str = ""
    logPath: str = ""
    # This launch's trace directory. Shown for the same reason the log path is: the operator is
    # the one who has to find these afterwards, and a batch whose location is not on screen is a
    # batch that gets analysed as whatever happened to be in the default directory.
    tracePath: str = ""
    previewDir: str = ""
    lastLines: list[str] = field(default_factory=list)
    # Set once the operator has been asked to record how the last rollout went, so the page can
    # prompt exactly once per rollout rather than on every poll.
    pendingOutcomeFor: int = 0
    # Where the last finished rollout put the gripper, in the dataset's own frame. Carried on
    # the status rather than fetched separately because it arrives on the same log line that
    # raises `pendingOutcomeFor`, and the page draws the point before the operator grades it.
    lastRolloutGeometry: dict[str, Any] = field(default_factory=dict)
    # Whether the operator took the arm during the last finished rollout, and for how many
    # steps. Carried beside the geometry and for the same reason -- it arrives on the same end
    # marker, and the page collects the grade before anything else could fetch it -- but kept a
    # separate field because it qualifies the grade rather than describing where the arm went.
    lastRolloutIntervention: dict[str, Any] = field(default_factory=dict)
    # How the last finished rollout's terminal descent ended, from the runtime's own
    # `terminal_servo=done` line. Per rollout and cleared with the geometry, unlike the two
    # fields below it -- the descent happens once inside each rollout, the configuration is
    # announced once for the session.
    lastRolloutTerminalServo: dict[str, Any] = field(default_factory=dict)
    # One entry per takeover in the last finished rollout, keyed by span index. Per rollout and
    # cleared with the geometry. Keyed rather than a list because the lines arrive one at a time
    # and a log that lost one must leave a hole rather than shift every span after it.
    lastRolloutTakeovers: dict[int, dict[str, Any]] = field(default_factory=dict)
    # The arm this session is running: which draw the policy executes (E3) and where the last
    # centimetres are driven to (E5 / E7-C). Session-scoped because the runtime announces both
    # at startup and neither can change without restarting it. Kept on the status so the grade
    # of every rollout in the session can be filed against the configuration that produced it.
    policyArm: dict[str, Any] = field(default_factory=dict)
    terminalServoConfig: dict[str, Any] = field(default_factory=dict)


def build_rollout_command(
    repo_root: Path,
    *,
    mode: str,
    checkpoint_path: str,
    dataset_root: str,
    target_frame_name: str,
    robot_ip: str = "",
    camera_config: str = "",
    max_steps: int = 0,
    move_to_start: bool = True,
    runtime_options: dict[str, str] | None = None,
    preview_dir: Path = PREVIEW_DIR,
    preview_fps: float = PREVIEW_FPS,
    trace_dir: Path | None = None,
    dagger_dataset_fallback: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The launcher invocation for one rollout, plus the environment that configures it.

    `dataset_root` and `target_frame_name` are passed explicitly rather than left to the
    launcher's defaults. Both default to the rig's *current* configuration, which is the right
    answer for a checkpoint trained today and the wrong one for a checkpoint trained before a
    change -- and the dataset root recorded inside a checkpoint is an absolute path on whatever
    machine trained it, which need not be this one.

    `dagger_dataset_fallback` is where corrections go when takeover is on and the operator named
    no directory. Passed in rather than derived here because it is keyed to the checkpoint, which
    this function knows only as a path on disk.

    `base_env` is whatever the caller needs the process to inherit (PYTHONPATH and friends).
    The rollout settings are applied *on top* of it and are never overwritten by it: an
    `FR3_TARGET_FRAME_NAME` left in the gateway's own environment must not be able to silently
    replace the frame this checkpoint was trained against.
    """
    if mode not in MODES_BY_ID:
        raise RolloutError(f"Unknown rollout mode {mode!r}. Expected one of {', '.join(MODES_BY_ID)}.")
    script = repo_root / LAUNCHER
    if not script.is_file():
        raise RolloutError(f"Rollout launcher missing: {script}")
    if not checkpoint_path:
        raise RolloutError("A rollout needs a checkpoint.")

    env = dict(base_env) if base_env is not None else os.environ.copy()
    for key in ROLLOUT_RUNTIME_ENV_KEYS:
        env.pop(key, None)
    for key, value in (runtime_options or {}).items():
        if key not in ROLLOUT_RUNTIME_ENV_KEYS:
            raise RolloutError(f"Unsupported rollout runtime environment key {key!r}.")
        env[key] = value

    if env.get("FR3_DAGGER_TAKEOVER") == "1":
        if not MODES_BY_ID[mode].takeover:
            # Refused rather than dropped. The launcher forwards the DAgger flags to two modes
            # only, so on any other one this setting is a no-op -- and a no-op here reads to the
            # operator as a rollout they can grab the arm out of, which they cannot.
            raise RolloutError(
                f"{MODES_BY_ID[mode].label} does not take DAgger takeover: the runtime refuses it "
                "without interactive rollouts. Use 'Interactive rollouts' or "
                "'Interactive + MuJoCo viewer'."
            )
        if "FR3_DAGGER_DATASET_ROOT" not in env:
            # Absent, not blank: the inherited copy was cleared above, so the only thing that can
            # have put it here is this launch's own options. Blank is the operator saying
            # "steer only" and is left exactly as it is; absent is a question nobody asked them,
            # and the answer that loses corrections is not the one to default to.
            if dagger_dataset_fallback is None:
                raise RolloutError(
                    "DAgger takeover needs somewhere to write corrections, and no default was "
                    "supplied for this checkpoint."
                )
            env["FR3_DAGGER_DATASET_ROOT"] = str(dagger_dataset_fallback)

    env["FR3_INFER_CHECKPOINT"] = checkpoint_path
    env["FR3_MOVE_TO_START"] = "1" if move_to_start else "0"
    env["PYTHONUNBUFFERED"] = "1"
    if dataset_root:
        env["FR3_INFER_DATASET_ROOT"] = dataset_root
    if target_frame_name:
        env["FR3_TARGET_FRAME_NAME"] = target_frame_name
    if robot_ip:
        env["FR3_ROBOT_IP"] = robot_ip
    if camera_config:
        env["FR3_INFER_CAMERA_CONFIG"] = camera_config
    if max_steps > 0:
        env["FR3_INFER_MAX_STEPS"] = str(int(max_steps))
    else:
        # Cleared rather than left alone: inherited from the caller's environment it would put
        # a step bound on a rollout that asked for none, which looks like the policy stopping.
        env.pop("FR3_INFER_MAX_STEPS", None)

    command = ["bash", str(script), mode]
    if mode != "env":
        # Appended after the mode, so they land in the launcher's `extra_args` and override the
        # flags it set for that mode. The window would need an X display on the rig; the JPEG
        # directory reaches a browser anywhere.
        command += [
            "--no-camera-preview-window",
            "--preview-jpeg-dir",
            str(preview_dir),
            "--preview-jpeg-fps",
            str(preview_fps),
            # Every step, because the page draws the arm from these and a gap in them is a jump
            # in the drawing. They cost one short line per step in a log that already carries
            # per-step telemetry, and the launcher's own modes decide whether to forward them.
            "--live-frame-interval",
            "1",
        ]
        if trace_dir is not None:
            # Passed rather than left to the runtime's default: see TRACE_ROOT. A terminal
            # operator picks this per batch; the browser has no place to type it, so the
            # gateway derives one per launch.
            command += ["--rollout-trace-dir", str(trace_dir)]
    return command, env


# ----------------------------------------------------------------- log parsing ---

_STEP_RE = re.compile(r"\bstep=(\d+)\b")
_STATUS_RE = re.compile(r"\bstatus=([A-Za-z_]+)")
_ROLLOUT_START_RE = re.compile(r"interactive_rollout_start index=(\d+)")
_ROLLOUT_END_RE = re.compile(r"interactive_rollout_end index=(\d+) status=(\w+)")
_KEYBOARD_BACKEND_RE = re.compile(r"keyboard_backend=(\w+)")
_DAGGER_READY_RE = re.compile(r"dagger_takeover=ready\b")
_DAGGER_TIMESTAMPS_RE = re.compile(r"\breport_timestamps=(\w+)")
_DAGGER_RELEASE_RE = re.compile(r"\brelease_after_s=([\d.]+)")
_DAGGER_DATASET_RE = re.compile(r"dagger_dataset=(created|extending)\s+root=(\S+)")
_DAGGER_WRITTEN_RE = re.compile(
    r"dagger_dataset_written\s+rollout=\d+\s+episodes=(\d+)\s+frames=(\d+)"
    r"\s+skipped_spans=(\d+)\s+dropped_frames=(\d+)"
)
_ARM_AT_START_RE = re.compile(r"\barm_at_start=([01])\b")
_HOMING_RE = re.compile(r"interactive_homing=(\w+)")
_GEOMETRY_POINT_RE = re.compile(
    r"\b(grasp_xyz|release_xyz|approach_xyz)=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)
_GEOMETRY_SCALAR_RE = re.compile(r"\b(apex_z|lift_m|descent_m)=(-?[\d.]+)")
_GEOMETRY_DRIVER_RE = re.compile(r"\b(grasp_by|release_by|approach_by)=(policy|expert)")
_INTERVENED_RE = re.compile(r"\bintervened=(\d+)")
_EXPERT_STEPS_RE = re.compile(r"\bexpert_steps=(\d+)")
# `expert_spans=41-58;120-133` -- one inclusive step range per stretch the operator was driving.
# The runtime has printed this since takeover shipped; the page threw it away and kept only the
# total, which is the number that cannot answer "how many separate times did you reach in".
_EXPERT_SPANS_RE = re.compile(r"\bexpert_spans=((?:\d+-\d+)(?:;\d+-\d+)*)")
_GEOMETRY_COUNT_RE = re.compile(r"\b(samples|held_steps|closed)=(\d+)")
# The sampling arm the runtime announced for itself at startup (E3). Read from the runtime's
# announce rather than from the launch request, for the reason the outcome record gives for the
# geometry: a page that files its own numbers can file an arm the process never ran.
_POLICY_ARM_RE = re.compile(
    r"\baction_samples=(\d+)\s+aggregate=(\S+)\s+selection_horizon=(\d+)"
)
# E5 / E7-C. Printed once, only when a terminal servo pose is configured, so absence means the
# rollout ended wherever the policy left it.
_TERMINAL_SERVO_CONFIGURED_RE = re.compile(
    r"terminal_servo=configured xyz=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
    r"\s+handoff_z=(-?[\d.]+)\s+max_speed_ms=([\d.]+)"
    r"\s+search_ring_m=([\d.]+)\s+search_landings=(\d+)"
)
_TERMINAL_SERVO_DONE_RE = re.compile(r"terminal_servo=done\b")
_TERMINAL_SERVO_WORD_RE = re.compile(r"\b(verdict|stopped_on|search)=(\w+)")
_TERMINAL_SERVO_NUMBER_RE = re.compile(
    r"\b(stopped_z|above_target_mm|lateral_mm|held_up_mm|growth_mm|peak_growth_mm|lag_mm"
    r"|descent_mm|descent_s|settle_s|settle_mm|search_offset_mm)=([+-]?[\d.]+)"
)
_TERMINAL_SERVO_INDEX_RE = re.compile(r"\bsearch_index=(\d+)/(\d+)")
# P1-9. One line per takeover, emitted just before the end marker. The pose is a measurement and
# the reference is beside it; the subtraction is left to the reader for the reason the runtime's
# own docstring gives.
_EXPERT_SPAN_RE = re.compile(
    r"expert_span\s+index=(\d+)\s+first=(\d+)\s+last=(\d+)\s+step=(\d+)"
    r"\s+xyz=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)
_EXPERT_SPAN_STATUS_RE = re.compile(r"\bpolicy_status=(\w+)")
_EXPERT_SPAN_BUDGET_RE = re.compile(r"\bsteps_left=(\d+)")
_EXPERT_SPAN_REFERENCE_RE = re.compile(
    r"\breference_xyz=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)
_TERMINAL_SERVO_FIELD_NAMES = {
    "verdict": "verdict",
    "stopped_on": "stoppedOn",
    "search": "searchStoppedOn",
    "stopped_z": "stoppedZ",
    "above_target_mm": "aboveTargetMm",
    "lateral_mm": "lateralErrorMm",
    "held_up_mm": "heldUpMm",
    "growth_mm": "heldUpGrowthMm",
    "peak_growth_mm": "peakGrowthMm",
    "lag_mm": "lagMm",
    "descent_mm": "descentMm",
    "descent_s": "descentSeconds",
    "settle_s": "settleSeconds",
    "settle_mm": "settleMm",
    "search_offset_mm": "searchOffsetMm",
}
# The runtime writes these as log fields; the page reads them as JSON. Renamed at this single
# crossing so neither side has to carry the other's convention.
_GEOMETRY_FIELD_NAMES = {
    "grasp_xyz": "graspXyz",
    "release_xyz": "releaseXyz",
    "approach_xyz": "approachXyz",
    "apex_z": "apexZ",
    "lift_m": "liftM",
    "descent_m": "descentM",
    "samples": "samples",
    "held_steps": "heldSteps",
    "closed": "closed",
    "grasp_by": "graspBy",
    "release_by": "releaseBy",
    "approach_by": "approachBy",
}


def parse_rollout_geometry(text: str) -> dict[str, Any]:
    """The landing points the runtime appends to its rollout end marker.

    Returned as a plain dict rather than a typed record because the runtime prints only the
    fields that exist for that rollout: one that never closed its gripper has an approach point
    and no grasp point, and inventing zeros for the missing half would put a rollout at the
    origin of the plot rather than leaving it off.
    """
    geometry: dict[str, Any] = {}
    for match in _GEOMETRY_POINT_RE.finditer(text):
        try:
            geometry[_GEOMETRY_FIELD_NAMES[match.group(1)]] = [
                float(match.group(index)) for index in (2, 3, 4)
            ]
        except ValueError:
            continue
    for match in _GEOMETRY_SCALAR_RE.finditer(text):
        try:
            geometry[_GEOMETRY_FIELD_NAMES[match.group(1)]] = float(match.group(2))
        except ValueError:
            continue
    for match in _GEOMETRY_COUNT_RE.finditer(text):
        try:
            value = int(match.group(2))
        except ValueError:
            continue
        field_name = _GEOMETRY_FIELD_NAMES[match.group(1)]
        geometry[field_name] = bool(value) if field_name == "closed" else value
    # Who was driving at each of those instants. Carried with the point rather than derived from
    # the rollout-level `intervened`, which cannot say *when*: an operator who took the arm after
    # the grasp did not place it, and one who seated the peg did not leave the policy a success.
    for match in _GEOMETRY_DRIVER_RE.finditer(text):
        geometry[_GEOMETRY_FIELD_NAMES[match.group(1)]] = match.group(2)
    return geometry


def parse_rollout_intervention(text: str) -> dict[str, Any]:
    """Whether the operator drove part of this rollout, for how many steps, and in how many goes.

    Kept apart from the geometry although it arrives on the same line: the landing points say
    where the arm ended up, this says whose hand put it there. A rollout the operator finished
    by hand says nothing about the policy's success rate, and a grade that cannot be told apart
    from an unassisted one is how it ends up in that rate anyway.

    The runtime prints `intervened=1` only when its own trace shows expert spans, so on a marker
    that carries a summary at all, absence means no takeover. On one that does not -- a runtime
    older than the field -- the answer is unknown, and the empty dict says so rather than
    reporting an assisted rollout as clean.

    `spans` is the part the total cannot say. 80 expert steps is one long rescue or four short
    ones, and the difference is the whole grade: the stage a rollout earned is the one it reached
    before the *first* takeover, because from that moment on it is continuing out of a state a
    human put it in. A page that only knows the total has to ask the operator to remember where
    the first one was.
    """
    if "samples=" not in text:
        return {}
    match = _INTERVENED_RE.search(text)
    intervened = bool(match and match.group(1) != "0")
    steps = _EXPERT_STEPS_RE.search(text)
    parsed: dict[str, Any] = {
        "intervened": intervened,
        "expertSteps": int(steps.group(1)) if intervened and steps else 0,
    }
    spans = _EXPERT_SPANS_RE.search(text)
    if intervened and spans:
        # A list of pairs rather than the runtime's own string: the page draws them and the log
        # is read by scripts, and neither should have to re-parse `41-58;120-133`.
        parsed["spans"] = [
            [int(first), int(last)]
            for first, last in (part.split("-", 1) for part in spans.group(1).split(";"))
        ]
    return parsed


def parse_rollout_line(line: str) -> dict[str, Any]:
    """Everything a page can learn from one runtime log line.

    Returns only the keys this line actually carries, so a caller can update fields without
    overwriting ones the line says nothing about -- a `step=` line reports no rollout index,
    and treating its absence as zero would reset the counter thirty times a second.
    """
    parsed: dict[str, Any] = {}
    stripped = line.strip()
    if not stripped:
        return parsed

    if stripped.startswith("[INFO] step=") or stripped.startswith("[PREVIEW] step="):
        step_match = _STEP_RE.search(stripped)
        if step_match:
            parsed["step"] = int(step_match.group(1))
        status_match = _STATUS_RE.search(stripped)
        if status_match:
            parsed["commandStatus"] = status_match.group(1)
        return parsed

    if "interactive_waiting_for_start" in stripped:
        parsed["state"] = "waiting"
        at_start = _ARM_AT_START_RE.search(stripped)
        # A runtime that prints no arm_at_start field is telling us nothing, and the honest
        # reading of nothing is "not known to be at the start pose". Being wrong that way costs
        # one press of an idempotent button; being wrong the other way starts a rollout from a
        # pose the dataset frame was never anchored to.
        parsed["armAtStart"] = bool(at_start) and at_start.group(1) == "1"
        parsed["message"] = (
            "Waiting for Start. The arm is at its start pose."
            if parsed["armAtStart"]
            else "Waiting for Start. The arm is where the last rollout left it."
        )
        return parsed

    homing_match = _HOMING_RE.search(stripped)
    if homing_match:
        phase = homing_match.group(1)
        if phase == "start":
            # Its own state, not "waiting". Waiting means the arm is parked and safe to reach
            # into; during this it is moving, and the page has to stop saying otherwise.
            parsed["state"] = "homing"
            parsed["message"] = "Moving the arm back to its start pose."
        elif phase == "done":
            parsed["armAtStart"] = True
            parsed["message"] = "The arm is back at its start pose."
        else:
            # Reported, not fatal: the runtime hands the session back rather than tearing down a
            # loaded policy, so the page has to as well. `armAtStart` stays false, which is what
            # keeps the warning on screen after this message is overwritten by the next line.
            parsed["armAtStart"] = False
            parsed["message"] = stripped[:400]
        return parsed

    start_match = _ROLLOUT_START_RE.search(stripped)
    if start_match:
        parsed["state"] = "rolling"
        parsed["rolloutIndex"] = int(start_match.group(1))
        parsed["step"] = 0
        # From this instant the arm is no longer at the pose the episodes began from, and it
        # will not be again until somebody homes it. Set here rather than on the end marker so
        # a session that dies mid-rollout still leaves the page telling the truth.
        parsed["armAtStart"] = False
        # Cleared here so the plot never shows the previous rollout's landing point attached to
        # the one now running.
        parsed["lastRolloutGeometry"] = {}
        parsed["lastRolloutIntervention"] = {}
        parsed["lastRolloutTerminalServo"] = {}
        parsed["lastRolloutTakeovers"] = {}
        parsed["message"] = f"Rollout {start_match.group(1)} running."
        return parsed

    end_match = _ROLLOUT_END_RE.search(stripped)
    if end_match:
        # Not "waiting". The rollout is over, but the runtime is not at its gate yet: it still has
        # a trace to write and, with takeover on, a DAgger episode to encode -- seconds to tens of
        # seconds during which every pending request is dropped when the gate is finally reached.
        # That window is exactly when the operator grades and reaches for Reset scene, which is
        # how a reset came to be accepted, written to stdin, and then silently swallowed.
        parsed["state"] = "finishing"
        parsed["rolloutIndex"] = int(end_match.group(1))
        parsed["lastRolloutStatus"] = end_match.group(2)
        # The page prompts for an outcome against this index. Recorded here rather than when
        # the rollout starts, because a rollout that never finished has nothing to grade.
        parsed["pendingOutcomeFor"] = int(end_match.group(1))
        parsed["lastRolloutGeometry"] = parse_rollout_geometry(stripped)
        parsed["lastRolloutIntervention"] = parse_rollout_intervention(stripped)
        parsed["message"] = f"Rollout {end_match.group(1)} ended ({end_match.group(2)})."
        return parsed

    arm = parse_policy_arm(stripped)
    if arm:
        parsed["policyArm"] = arm
        # Named the way the control arm deserves. "the medoid of 1 draw" is arithmetically true
        # and reads as a configuration nobody chose; a single draw is what every sampling result
        # is measured against, and the bar should say so.
        parsed["message"] = (
            "Executing a single draw per inference."
            if arm["actionSamples"] <= 1
            else (
                f"Executing the {arm['actionAggregate']} of {arm['actionSamples']} draws "
                f"over {arm['selectionHorizon']} steps."
            )
        )
        return parsed

    servo_config = parse_terminal_servo_config(stripped)
    if servo_config:
        parsed["terminalServoConfig"] = servo_config
        landings = servo_config["terminalServoSearchLandings"]
        parsed["message"] = (
            "Terminal servo configured: "
            + ("one descent at the fixed pose." if landings <= 1 else f"{landings} landings.")
        )
        return parsed

    span_detail = parse_expert_span(stripped)
    if span_detail:
        # Reported for the caller to merge rather than assigned, the same way the DAgger
        # per-rollout counts are: one line carries one span, and assigning here would leave the
        # status holding only the last one.
        parsed["takeoverDetail"] = span_detail
        parsed["message"] = (
            f"Takeover {span_detail['index'] + 1}: steps "
            f"{span_detail['first']}–{span_detail['last']} at z={span_detail['xyz'][2]:.3f}."
        )
        return parsed

    servo_result = parse_terminal_servo_result(stripped)
    if servo_result:
        parsed["lastRolloutTerminalServo"] = servo_result
        verdict = servo_result.get("verdict", "?")
        above = servo_result.get("aboveTargetMm")
        # Deliberately no state. The descent ends inside a rollout that is still running -- the
        # arm still has to release, retreat and hand back -- and reporting `finishing` here would
        # re-open the grading prompt against a rollout index the runtime has not closed yet.
        parsed["message"] = (
            f"Terminal descent: {verdict}"
            + ("" if above is None else f", {above:+.1f} mm above the seated depth.")
        )
        return parsed

    if "scene_reset=start" in stripped:
        parsed["state"] = "resetting"
        parsed["armAtStart"] = False
        parsed["message"] = "Scene reset is moving the peg."
        return parsed

    if "scene_reset_step=" in stripped:
        parsed["state"] = "resetting"
        parsed["message"] = stripped[:400]
        return parsed

    if "scene_reset=done" in stripped:
        # Printed after the return-to-start move, but still before the loop publishes a preview
        # snapshot and re-enters the gate. Same rule as the rollout end marker: the reset being
        # over is not the runtime being ready for the next one.
        parsed["state"] = "finishing"
        parsed["message"] = "Scene reset finished; the runtime is returning to its command gate."
        return parsed

    if "scene_reset=failed" in stripped:
        parsed["state"] = "finishing"
        parsed["armAtStart"] = False
        parsed["message"] = stripped[:400]
        return parsed

    if "interactive_rollouts=stopped" in stripped:
        parsed["state"] = "complete"
        parsed["message"] = "Interactive rollout session ended."
        return parsed

    if _DAGGER_READY_RE.search(stripped):
        # The device is open. Said here as well as on the control-channel line because this is the
        # earlier of the two and the one carrying the pre-flight, and because the operator reads
        # this banner before they touch anything.
        parsed["takeoverAvailable"] = True
        timestamps = _DAGGER_TIMESTAMPS_RE.search(stripped)
        if timestamps:
            parsed["daggerReportTimestamps"] = timestamps.group(1)
        release = _DAGGER_RELEASE_RE.search(stripped)
        if release:
            try:
                parsed["daggerReleaseAfterS"] = float(release.group(1))
            except ValueError:
                pass
        parsed["message"] = "SpaceMouse armed for takeover: " + stripped[len("[INFO] ") :][:360]
        return parsed

    dataset_match = _DAGGER_DATASET_RE.search(stripped)
    if dataset_match:
        # The runtime's own answer, which can differ from the one this launch asked for: it
        # resolves the path and says whether it created the dataset or is extending one. Extending
        # is the ordinary case after the first session and is worth seeing, because the alternative
        # -- a fresh dataset every time, from a path that moved -- looks identical on the page
        # until training day.
        parsed["daggerDatasetPath"] = dataset_match.group(2)
        parsed["message"] = (
            f"DAgger corrections {dataset_match.group(1)}: {dataset_match.group(2)}"
        )
        return parsed

    written_match = _DAGGER_WRITTEN_RE.search(stripped)
    if written_match:
        episodes, frames, skipped, dropped = (int(written_match.group(i)) for i in (1, 2, 3, 4))
        # Summed by the caller, not assigned: see `daggerEpisodes`. Reported under different names
        # from the fields they add into, so a line that says "this rollout wrote two" can never be
        # mistaken for "the session has two".
        parsed["daggerEpisodesWritten"] = episodes
        parsed["daggerDroppedFramesWritten"] = dropped
        skipped_text = f", {skipped} too short to keep" if skipped else ""
        parsed["message"] = (
            f"Wrote {episodes} correction episode(s), {frames} frames{skipped_text}."
            if episodes
            else f"No corrections to write from that rollout{skipped_text or '.'}"
        )
        return parsed

    backend_match = _KEYBOARD_BACKEND_RE.search(stripped)
    if backend_match:
        # The runtime prints `takeover_key='t'` on this same line, and only when it has a device
        # to hand the arm to.
        parsed["takeoverAvailable"] = "takeover_key=" in stripped
        # Deliberately no state. The control channel being open is not the same as the runtime
        # being ready to act on it: `start` is read by the listener thread the moment this line
        # prints, but the loop clears every pending request when it reaches its wait, so a start
        # sent in that window is swallowed without a trace. `interactive_waiting_for_start` is
        # the marker that means the runtime is actually at the gate, and it is the only marker in
        # this function that may report `waiting` -- every other end-of-activity line reports
        # `finishing`, because the gap between the two is real and what is sent into it is lost.
        parsed["message"] = f"Rollout control channel ready ({backend_match.group(1)}); waiting for the runtime to reach its start gate."
        return parsed

    if stripped.startswith("[ERROR]") or "Traceback (most recent call last)" in stripped:
        parsed["message"] = stripped[:400]
        return parsed

    return parsed


def parse_policy_arm(text: str) -> dict[str, Any]:
    """Which draw the runtime is executing, and how it chose it.

    E3 compares medoid against mean against a single draw, and E8 would add a fourth arm on the
    same mechanism. The three numbers that name the arm were settable from the page and reachable
    by the runtime, and then present in neither the status nor the outcome log -- so two arms of
    the same comparison were distinguishable only by which log file a reader happened to open.
    Parsed from the runtime's announce rather than echoed back from the launch request, the same
    provenance rule the landing points follow.
    """
    match = _POLICY_ARM_RE.search(text)
    if not match:
        return {}
    return {
        "actionSamples": int(match.group(1)),
        "actionAggregate": match.group(2),
        "selectionHorizon": int(match.group(3)),
    }


def parse_terminal_servo_config(text: str) -> dict[str, Any]:
    """The fixed pose the last centimetres are driven to, and whether the search ring is on.

    `searchLandings` rather than the ring radius alone, because the radius is not the arm: a
    7 mm ring with one landing and a 7 mm ring with nine cover different discs, and the runtime
    is the only party that has applied its own refusals to the pair.
    """
    match = _TERMINAL_SERVO_CONFIGURED_RE.search(text)
    if not match:
        return {}
    return {
        "terminalServoXyz": [float(match.group(index)) for index in (1, 2, 3)],
        "terminalServoHandoffZ": float(match.group(4)),
        "terminalServoMaxSpeedMs": float(match.group(5)),
        "terminalServoSearchRingM": float(match.group(6)),
        "terminalServoSearchLandings": int(match.group(7)),
    }


def parse_terminal_servo_result(text: str) -> dict[str, Any]:
    """What the terminal descent did, as fields rather than as a sentence in the log.

    The four signatures on the 2026-09-10 card matched the operator's grade on all nine descents
    they were drawn from, which makes this the rig's one machine-readable statement about how a
    rollout ended -- and it was being written to a log file and thrown away. Recorded beside the
    operator's grade, never instead of it: the agreement between the two is the measurement, and
    a column that has replaced the thing it was supposed to be checked against cannot report it.

    `verdict` comes from the runtime, which owns the thresholds
    (`classify_terminal_servo_descent`). A line from a runtime older than that field parses
    without one rather than being classified here, because re-deriving it at this crossing is
    how the page and the rig come to disagree about what "seated" means.
    """
    if not _TERMINAL_SERVO_DONE_RE.search(text):
        return {}
    result: dict[str, Any] = {}
    for match in _TERMINAL_SERVO_WORD_RE.finditer(text):
        result[_TERMINAL_SERVO_FIELD_NAMES[match.group(1)]] = match.group(2)
    for match in _TERMINAL_SERVO_NUMBER_RE.finditer(text):
        try:
            result[_TERMINAL_SERVO_FIELD_NAMES[match.group(1)]] = float(match.group(2))
        except ValueError:
            continue
    index_match = _TERMINAL_SERVO_INDEX_RE.search(text)
    if index_match:
        # Which landing answered, out of how many were available. `3/8` on a seated verdict is
        # the search earning its place; `0/0` is the control arm.
        result["searchIndex"] = int(index_match.group(1))
        result["searchLandings"] = int(index_match.group(2)) + 1
    return result


def parse_expert_span(text: str) -> dict[str, Any]:
    """One takeover: which steps it covered, where the arm was, and what held there.

    The pose rather than a residual, and the reference as its own field when one exists. Every
    candidate reference on this rig is absent, drifting or floored -- the fixture creeps within a
    session -- so a difference computed at write time is a number that expires inside a log that
    cannot be rewritten. Both halves are kept and the subtraction is a read-time operation, which
    is what lets a corrected reference re-read finished records instead of invalidating them.

    `policyStatus` is the machine's half of "why did the operator reach in", read from the last
    policy step before the span. It is absent when the span starts at step 0, because then there
    is no policy step to read and the honest answer is that nobody asked.
    """
    match = _EXPERT_SPAN_RE.search(text)
    if not match:
        return {}
    detail: dict[str, Any] = {
        "index": int(match.group(1)),
        "first": int(match.group(2)),
        "last": int(match.group(3)),
        # The trace CSV's own step number. Equal to `first` unless a sample was dropped, and
        # carried separately because this record exists precisely for the case where the trace
        # is gone -- one browser session overwrote a graded batch's traces on 2026-09-01.
        "step": int(match.group(4)),
        "xyz": [float(match.group(index)) for index in (5, 6, 7)],
    }
    status = _EXPERT_SPAN_STATUS_RE.search(text)
    if status:
        detail["policyStatus"] = status.group(1)
    budget = _EXPERT_SPAN_BUDGET_RE.search(text)
    if budget:
        detail["stepsLeft"] = int(budget.group(1))
    reference = _EXPERT_SPAN_REFERENCE_RE.search(text)
    if reference:
        detail["referenceXyz"] = [float(reference.group(index)) for index in (1, 2, 3)]
    return detail


def is_noise(line: str) -> bool:
    """Per-step telemetry, which is far too dense to keep in the page's rolling log tail.

    Dropped from `lastLines` only. The full log file keeps every line, and the step counter is
    read off exactly these lines before they are discarded.
    """
    stripped = line.strip()
    return stripped.startswith("[INFO] step=") or stripped.startswith("[PREVIEW] step=")
