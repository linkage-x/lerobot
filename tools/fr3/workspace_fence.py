"""Where the arm may be told to go, and which file gets to say so.

``send_action`` clips the commanded tool-frame origin to ``workspace_min/max`` inside the driver
(``franka_research3.py``), *after* this process's step and leash guards, and then reports the
clipped pose back as ``prev_cmd``. A fence set too tight therefore does not raise, does not log,
and never reaches the guard: the command is shortened, the arm stops short of where the policy
asked to go, and every number this process prints still says the rollout is healthy.

Which is why the fence gets one derivation and not two. ``fr3_record_config.yaml`` carries it
together with the table it was measured off and the span of the recorded frames it has to
contain. A rollout holding its own copy is a rollout whose reachable region can drift away from
the one the demonstrations were collected in with nobody editing either file -- and it did: the
copy in ``fr3_act_infer_real_runtime.py`` stood at ``z >= 0.05`` while the recording fence was
``z >= 0``, and the demonstrations go below 50 mm routinely. Measured 2026-09-01 over the 50
episodes of ``fr3_spacemouse-insert``: the gripper closes at a median ``z`` of 50.7 mm (sd 3.5),
each episode bottoms out at a median of 48.2 mm, and **30 of the 50 episodes reach below 50 mm**,
the deepest at 40.7 mm. So the rollout fence stood above the lowest point of six demonstrations
in ten, and nothing reported it. (The ``z = 0.028`` quoted elsewhere on this rig is the
*pick-and-place* dataset, not this one; the record config's floor has to clear both, so 28 mm is
the binding number for the fence even though 40.7 mm is the one for this task.)

Nothing here guesses which fence to use. The fence belongs to a table, two rigs run that runtime,
and the launcher is the layer that knows which rig it started: it names a record config, and only
then is a fence read from one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

Fence = tuple[float, float, float]

WORKSPACE_MIN_KEY = 'workspace_min'
WORKSPACE_MAX_KEY = 'workspace_max'

# What a run gets when nobody named a rig. Kept because the other FR3 rig's launcher does not name
# one and this is the box it has always run with -- not because anything derives it. The recording
# config says of its own dataclass ancestor: "do not silently rely on the dataclass default, which
# was written for pika_task_tcp and is meaningless at the tool point." The same holds for this.
DEFAULT_WORKSPACE_MIN: Fence = (0.1, -0.6, 0.05)
DEFAULT_WORKSPACE_MAX: Fence = (0.9, 0.6, 0.8)


def _as_fence(values: Sequence[float], *, what: str) -> Fence:
    numbers = [float(value) for value in values]
    if len(numbers) != 3:
        raise ValueError(f'{what} must be three numbers (x y z); got {len(numbers)}: {numbers}')
    return (numbers[0], numbers[1], numbers[2])


def _validate(minimum: Fence, maximum: Fence, *, source: str) -> None:
    # The same rule ``FrankaResearch3Config`` enforces, applied here so an inverted axis in a YAML
    # names the file it came from rather than surfacing later as a dataclass error about a value
    # nobody typed.
    for axis, (low, high) in enumerate(zip(minimum, maximum, strict=True)):
        if low >= high:
            raise ValueError(
                f'workspace fence from {source} is empty on axis {"xyz"[axis]}: '
                f'min={low} is not below max={high}'
            )


def read_record_config_fence(config_path: str | Path) -> tuple[Fence, Fence]:
    """The ``robot.workspace_min/max`` pair out of a record config.

    Missing keys raise rather than falling back. A rollout that quietly ran with a different fence
    than the recordings is the exact failure this module exists to remove, and a fallback is how it
    would come back.
    """
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    robot = payload.get('robot')
    if not isinstance(robot, dict):
        raise ValueError(f'{path} has no robot: block to read the workspace fence from')
    for key in (WORKSPACE_MIN_KEY, WORKSPACE_MAX_KEY):
        if key not in robot:
            raise ValueError(f'{path} has no robot.{key}; the rollout fence cannot be derived from it')
    minimum = _as_fence(robot[WORKSPACE_MIN_KEY], what=f'{path}: robot.{WORKSPACE_MIN_KEY}')
    maximum = _as_fence(robot[WORKSPACE_MAX_KEY], what=f'{path}: robot.{WORKSPACE_MAX_KEY}')
    _validate(minimum, maximum, source=str(path))
    return minimum, maximum


def resolve_workspace_fence(
    *,
    record_config_path: str | Path | None = None,
    workspace_min: Sequence[float] | None = None,
    workspace_max: Sequence[float] | None = None,
) -> tuple[Fence, Fence, str]:
    """The fence to run with, and where it came from, for the startup banner.

    Precedence is explicit over derived over default, and the source is returned rather than
    inferred by the caller, because "which box am I in" was invisible for exactly as long as it
    was a literal in the runtime.

    An explicit pair overrides the record config entirely -- both halves or neither. Taking one
    axis from the operator and the rest from a file would produce a box neither of them wrote.
    """
    if (workspace_min is None) != (workspace_max is None):
        raise ValueError('workspace_min and workspace_max must be given together, or not at all')
    if workspace_min is not None and workspace_max is not None:
        minimum = _as_fence(workspace_min, what='--workspace-min')
        maximum = _as_fence(workspace_max, what='--workspace-max')
        _validate(minimum, maximum, source='the command line')
        return minimum, maximum, 'command line'
    if record_config_path is not None:
        minimum, maximum = read_record_config_fence(record_config_path)
        return minimum, maximum, f'{record_config_path} (robot.workspace_min/max)'
    return DEFAULT_WORKSPACE_MIN, DEFAULT_WORKSPACE_MAX, 'built-in default (no record config named)'
