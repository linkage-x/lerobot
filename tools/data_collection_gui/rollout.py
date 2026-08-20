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

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAUNCHER = Path("tools/fr3/run_pick_place_infer_workstation.sh")

# Where the runtime publishes the frames it is feeding the policy. /dev/shm rather than the
# repo: these are written several times a second for the length of a rollout and are worthless
# once it ends, so they should never touch a disk or survive a reboot.
PREVIEW_DIR = Path("/dev/shm/lerobot_rollout_preview")
PREVIEW_FPS = 5.0
# A frame older than this is not a live view. Rollouts run at the dataset's rate, well above the
# preview rate, so the only way to exceed this is a run that has stopped producing frames.
PREVIEW_STALE_S = 3.0


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


@dataclass
class RolloutStatus:
    state: str = "idle"  # idle | starting | waiting | rolling | complete | error | stopped
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
    pid: int | None = None
    message: str = "Pick a checkpoint and a mode to start."
    startedAt: str = ""
    finishedAt: str = ""
    logPath: str = ""
    previewDir: str = ""
    lastLines: list[str] = field(default_factory=list)
    # Set once the operator has been asked to record how the last rollout went, so the page can
    # prompt exactly once per rollout rather than on every poll.
    pendingOutcomeFor: int = 0


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
    preview_dir: Path = PREVIEW_DIR,
    preview_fps: float = PREVIEW_FPS,
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
        ]
    return command, env


# ----------------------------------------------------------------- log parsing ---

_STEP_RE = re.compile(r"\bstep=(\d+)\b")
_STATUS_RE = re.compile(r"\bstatus=([A-Za-z_]+)")
_ROLLOUT_START_RE = re.compile(r"interactive_rollout_start index=(\d+)")
_ROLLOUT_END_RE = re.compile(r"interactive_rollout_end index=(\d+) status=(\w+)")
_KEYBOARD_BACKEND_RE = re.compile(r"keyboard_backend=(\w+)")


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
        parsed["message"] = "Waiting for Start. The arm is at its start pose."
        return parsed

    start_match = _ROLLOUT_START_RE.search(stripped)
    if start_match:
        parsed["state"] = "rolling"
        parsed["rolloutIndex"] = int(start_match.group(1))
        parsed["step"] = 0
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
        parsed["message"] = f"Rollout {end_match.group(1)} ended ({end_match.group(2)})."
        return parsed

    if "interactive_rollouts=stopped" in stripped:
        parsed["state"] = "complete"
        parsed["message"] = "Interactive rollout session ended."
        return parsed

    backend_match = _KEYBOARD_BACKEND_RE.search(stripped)
    if backend_match:
        parsed["message"] = f"Rollout control channel ready ({backend_match.group(1)})."
        parsed["state"] = "waiting"
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
