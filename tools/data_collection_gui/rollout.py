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
)
RTC_MODES = {"auto", "enabled", "disabled"}
RTC_PREFIX_ATTENTION_SCHEDULES = {"EXP", "LINEAR", "ONES", "ZEROS"}


@dataclass(frozen=True)
class RolloutMode:
    """One launcher mode, described by the two things an operator must know before pressing it.

    `movesArm` gates the confirmation the page requires. `interactive` decides whether the run
    is steered afterwards or simply runs to its end.
    """

    id: str
    label: str
    description: str
    movesArm: bool
    interactive: bool


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
    return options


@dataclass
class RolloutStatus:
    state: str = "idle"  # idle | starting | waiting | homing | resetting | rolling | complete | error | stopped
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
    base_env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The launcher invocation for one rollout, plus the environment that configures it.

    `dataset_root` and `target_frame_name` are passed explicitly rather than left to the
    launcher's defaults. Both default to the rig's *current* configuration, which is the right
    answer for a checkpoint trained today and the wrong one for a checkpoint trained before a
    change -- and the dataset root recorded inside a checkpoint is an absolute path on whatever
    machine trained it, which need not be this one.

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
_ARM_AT_START_RE = re.compile(r"\barm_at_start=([01])\b")
_HOMING_RE = re.compile(r"interactive_homing=(\w+)")
_GEOMETRY_POINT_RE = re.compile(
    r"\b(grasp_xyz|release_xyz|approach_xyz)=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)
_GEOMETRY_SCALAR_RE = re.compile(r"\b(apex_z|lift_m|descent_m)=(-?[\d.]+)")
_GEOMETRY_COUNT_RE = re.compile(r"\b(samples|held_steps|closed)=(\d+)")
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
    return geometry


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
        parsed["message"] = f"Rollout {start_match.group(1)} running."
        return parsed

    end_match = _ROLLOUT_END_RE.search(stripped)
    if end_match:
        parsed["state"] = "waiting"
        parsed["rolloutIndex"] = int(end_match.group(1))
        parsed["lastRolloutStatus"] = end_match.group(2)
        # The page prompts for an outcome against this index. Recorded here rather than when
        # the rollout starts, because a rollout that never finished has nothing to grade.
        parsed["pendingOutcomeFor"] = int(end_match.group(1))
        parsed["lastRolloutGeometry"] = parse_rollout_geometry(stripped)
        parsed["message"] = f"Rollout {end_match.group(1)} ended ({end_match.group(2)})."
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
        parsed["state"] = "waiting"
        parsed["message"] = "Scene reset finished; waiting for the next command."
        return parsed

    if "scene_reset=failed" in stripped:
        parsed["state"] = "waiting"
        parsed["armAtStart"] = False
        parsed["message"] = stripped[:400]
        return parsed

    if "interactive_rollouts=stopped" in stripped:
        parsed["state"] = "complete"
        parsed["message"] = "Interactive rollout session ended."
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
        # the marker that means the runtime is actually at the gate, and it is the only one that
        # may enable Start.
        parsed["message"] = f"Rollout control channel ready ({backend_match.group(1)}); waiting for the runtime to reach its start gate."
        return parsed

    if stripped.startswith("[ERROR]") or "Traceback (most recent call last)" in stripped:
        parsed["message"] = stripped[:400]
        return parsed

    return parsed


def is_noise(line: str) -> bool:
    """Per-step telemetry, which is far too dense to keep in the page's rolling log tail.

    Dropped from `lastLines` only. The full log file keeps every line, and the step counter is
    read off exactly these lines before they are discarded.
    """
    stripped = line.strip()
    return stripped.startswith("[INFO] step=") or stripped.startswith("[PREVIEW] step=")
