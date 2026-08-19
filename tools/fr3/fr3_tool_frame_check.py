#!/usr/bin/env python

"""Which FR3 tool frame a recorded dataset is expressed in, decided from the model.

`pika_task_tcp` and `pika_gripper_ee` are fixed frames on the same URDF, 410.85 mm apart, sharing an
orientation. Nothing in a LeRobot dataset records which of them its `ee.*` columns mean, so a
dataset recorded under one contract and replayed under the other does not fail -- the arm drives the
fingertips to where the other frame's origin used to be and runs the episode to completion.

The recorder homes to the `home` keyframe before every episode, so frame 0 of each one is that pose
seen through whichever frame the config named. Comparing it against the model's forward kinematics
for both candidates answers the question from the URDF rather than from a convention someone wrote
down. `fr3_sim_record_replay_runtime.py` makes the same decision at replay time by solving IK; this
is the keyframe-anchored version, which needs no solver and is exact.

Shared by the migration tool and the replay preflight on purpose: a safety-critical geometric
convention with two copies is a convention with two answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (
    REPO_ROOT / "src" / "lerobot" / "robots" / "franka_research3" / "assets" / "franka_fr3" / "fr3_pika_gripper_scene.xml"
)

OLD_FRAME = "pika_task_tcp"
NEW_FRAME = "pika_gripper_ee"
TOOL_FRAMES = (OLD_FRAME, NEW_FRAME)
BASE_BODY = "fr3_link0"

# pika_task_tcp -> pika_gripper_ee, in the tool frame, for every arm configuration. Pinned by
# tests/robots/test_fr3_tool_frame_geometry.py; changing it silently reinterprets every episode.
TOOL_FRAME_OFFSET_M = np.array([-0.366842, 0.0, 0.185], dtype=np.float64)

# How far an episode's first frame may sit from the home keyframe before the identification stops
# meaning anything. Measured across the 25 recorded episodes the worst was 6.3 mm, and the two
# frames are 410.85 mm apart, so this still separates them by more than 8x.
MAX_HOME_ERROR_M = 0.05
# ... and the runner-up has to be clearly a runner-up, not a coin flip.
MIN_FRAME_SEPARATION_M = 0.2


class ToolFrameError(RuntimeError):
    """A frame question that has no trustworthy answer, or the wrong one."""


def home_frame_positions() -> dict[str, np.ndarray]:
    """Where each candidate tool frame sits, in the robot base frame, at the `home` keyframe."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise ToolFrameError("mujoco is required to identify the tool frame; run this with .venv-fr3") from exc

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def origin(name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ToolFrameError(f"body '{name}' is missing from {SCENE_XML.name}")
        return np.asarray(data.xpos[body_id], dtype=np.float64)

    base = origin(BASE_BODY)
    return {name: origin(name) - base for name in TOOL_FRAMES}


def identify_frame(
    episode_start_positions: np.ndarray, home: dict[str, np.ndarray] | None = None
) -> tuple[str, dict[str, float]]:
    """The tool frame these episode-start positions are expressed in, and the distance to each."""
    home = home if home is not None else home_frame_positions()
    positions = np.atleast_2d(np.asarray(episode_start_positions, dtype=np.float64))
    errors = {
        name: float(np.median(np.linalg.norm(positions - position, axis=1)))
        for name, position in home.items()
    }
    return min(errors, key=errors.__getitem__), errors


def format_distances(errors: dict[str, float]) -> str:
    return ", ".join(f"{name} {errors[name] * 1e3:.1f} mm" for name in sorted(errors))


def require_frame(
    label: str,
    episode_start_positions: np.ndarray,
    expected: str,
    home: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Raise unless the data reads as `expected`, by a margin no arithmetic slip could fake."""
    identified, errors = identify_frame(episode_start_positions, home)
    report = format_distances(errors)
    runner_up = min((value for name, value in errors.items() if name != identified), default=float("inf"))

    if identified != expected:
        raise ToolFrameError(
            f"{label} is expressed in '{identified}', but '{expected}' was expected "
            f"(episode-start distance: {report}). The two frames are "
            f"{np.linalg.norm(TOOL_FRAME_OFFSET_M) * 1e3:.2f} mm apart and share an orientation, so "
            "using one for the other does not fail -- it runs the whole episode that far off."
        )
    if errors[identified] > MAX_HOME_ERROR_M:
        raise ToolFrameError(
            f"{label} is closest to '{identified}' but sits {errors[identified] * 1e3:.1f} mm from it "
            f"(limit {MAX_HOME_ERROR_M * 1e3:.0f} mm; distances: {report}). Either the episodes do not "
            "start from the home keyframe, or the data is in neither frame"
        )
    if runner_up - errors[identified] < MIN_FRAME_SEPARATION_M:
        raise ToolFrameError(
            f"{label} does not separate the two frames: {report}. The check cannot tell them apart, "
            "so it cannot vouch for the data either"
        )
    return errors


def workspace_clip_report(
    positions: np.ndarray, workspace_min, workspace_max
) -> tuple[int, float, list[str]]:
    """How much of a trajectory `send_action`'s workspace clip would flatten.

    The clip is applied to absolute replay commands too (`_make_pose_from_absolute_action`), so a
    box that no longer contains the recorded trajectory does not stop the replay -- it silently
    reshapes it and then the tracking score is computed against the reshaped version.
    """
    positions = np.asarray(positions, dtype=np.float64)
    low = np.asarray(workspace_min, dtype=np.float64)
    high = np.asarray(workspace_max, dtype=np.float64)
    clipped = np.clip(positions, low, high)
    displacement = np.linalg.norm(clipped - positions, axis=1)
    outside = int(np.count_nonzero(displacement > 0))

    details: list[str] = []
    for axis, name in enumerate("xyz"):
        below = int(np.count_nonzero(positions[:, axis] < low[axis]))
        above = int(np.count_nonzero(positions[:, axis] > high[axis]))
        if below:
            details.append(f"{name} < {low[axis]:.3f} on {below} frames")
        if above:
            details.append(f"{name} > {high[axis]:.3f} on {above} frames")
    return outside, float(displacement.max(initial=0.0)), details
