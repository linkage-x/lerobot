"""Turning the steps an operator drove into episodes a policy can be trained on.

Everything upstream of this file makes a takeover *happen*: the device decides it
(``dagger_takeover``), one guard bounds it (``command_guard``), the trace records where it
started and stopped (``RolloutGeometryTrace``). None of that produces a training sample. A
correction that leaves nothing behind but a span in a CSV has taught the policy nothing, and
that is the whole output of DAgger -- so this file is the one that makes the loop close.

**What gets written is what was sent, not what the operator asked for.** The action stored for
a step is the post-clamp, post-filter command that actually reached the arm. That is the
recorder's own contract (``lerobot_record`` saves the sent action for the same reason) and it
matters more here than there: the expert's raw delta is routinely larger than the step guard
allows, so training on the ask would teach the policy to command steps the deployment clamp
will refuse -- a policy fluent in a language its own runtime does not speak.

**Only the expert's own steps are written.** Not the frames leading up to the takeover, which
are the policy's own actions in the moments it was going wrong: regressing on those is
training on the mistake being corrected. The span boundaries here are exactly
``expert_spans``', so the trace and the dataset can never disagree about what was corrected.

**One episode per span, written after the rollout, not at the seam.** ``save_episode`` encodes
video; it takes seconds. The instant a span ends is the instant the operator has let go and the
policy is about to resume driving a real arm, which is the worst moment in the whole rollout to
stall the control loop. So spans are held in memory while the arm is moving and written when it
has stopped. That costs RAM -- see ``DEFAULT_MAX_BUFFERED_FRAMES``, which bounds it -- and the
trade is deliberate: memory is cheap and recoverable, a stalled control loop is neither.

The module imports nothing from ``lerobot.policies`` or the FR3 processor chain, for the reason
in ``command_guard``'s docstring: those need a GPU-scale install and are unimportable on the
workstation, and a writer that can only be tested on the rig is a writer whose reasoning errors
are found by the rig. The two pieces that genuinely need that chain -- encoding an absolute
command back into the dataset's action space, and denormalizing the gripper -- are injected as
callables by the runtime, which already holds the real ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.utils import build_dataset_frame

# The column that says a frame came from a human. Every frame this writer produces carries a 1,
# because only expert steps are written -- it is not a filter this dataset needs, it is what
# tells a training run reading this dataset *alongside* the demonstrations which is which.
IS_INTERVENTION_KEY = 'is_intervention'

# 450 frames is fifteen seconds of correction at 30 Hz, spread over however many spans one
# rollout contains. Two 480p RGB cameras make a frame about 1.8 MB, so the cap is roughly
# 830 MB and an ordinary rollout -- a few one-second nudges -- costs well under a tenth of it.
# Past the cap frames are dropped and counted rather than the process being allowed to grow
# into the swap of a machine that is driving an arm.
DEFAULT_MAX_BUFFERED_FRAMES = 450

OBSERVATION_PREFIX = 'observation'
ACTION_PREFIX = 'action'
DAGGER_INFO_PATH = Path('meta/info.json')
DAGGER_TASK_PATHS = (Path('meta/tasks.parquet'), Path('meta/tasks.jsonl'))
DAGGER_EPISODES_DIR = Path('meta/episodes')
DAGGER_DATA_DIR = Path('data')
DAGGER_RECREATABLE_FILES = {DAGGER_INFO_PATH, *DAGGER_TASK_PATHS}


def dagger_dataset_has_tasks(root: Path) -> bool:
    """Whether `root` has task metadata written by `save_episode`."""
    return any((root / path).exists() for path in DAGGER_TASK_PATHS)


def dagger_dataset_can_load_locally(root: Path) -> bool:
    """Whether `LeRobotDataset(repo_id, root=root)` should load without consulting the Hub."""
    return (
        (root / DAGGER_INFO_PATH).exists()
        and dagger_dataset_has_tasks(root)
        and any((root / DAGGER_EPISODES_DIR).rglob('*.parquet'))
        and any((root / DAGGER_DATA_DIR).rglob('*.parquet'))
    )


def dagger_dataset_is_unfinalized(root: Path) -> bool:
    """Whether `root` holds a session that wrote frames but never closed its writers.

    `save_episode` lands the data parquet and the videos, but the episode metadata sits in a
    ten-episode buffer and the data file keeps its footer until `finalize()`. A run that dies
    before that -- the gateway used to SIGTERM a rollout in the same breath as asking it to quit
    -- leaves exactly this shape: `data/` and `videos/` populated, `meta/episodes/` absent, and a
    parquet pyarrow refuses to open.

    Worth telling apart from a directory that merely holds someone else's files, because the
    corrections in it are not misplaced, they are unreadable: an operator told only to "move it
    aside" will keep it waiting for a recovery that cannot come.
    """
    return (
        (root / DAGGER_INFO_PATH).exists()
        and any((root / DAGGER_DATA_DIR).rglob('*.parquet'))
        and not any((root / DAGGER_EPISODES_DIR).rglob('*.parquet'))
    )


def dagger_dataset_root_is_recreatable(root: Path) -> bool:
    """True when a stale DAgger root contains no correction payload worth preserving.

    `LeRobotDataset.create` refuses an existing directory. A previous takeover session can leave
    behind either an empty directory or metadata files before any span is saved, and either form
    should be replaced by a real dataset on the next run. Anything with payload files is treated
    as operator data and must be inspected rather than overwritten.
    """
    if root.is_symlink() or not root.is_dir():
        return False
    files = {path.relative_to(root) for path in root.rglob('*') if path.is_file()}
    return files.issubset(DAGGER_RECREATABLE_FILES)


def dagger_dataset_features(base_features: dict[str, dict]) -> dict[str, dict]:
    """The recorder's feature schema plus the one column that marks a human's steps.

    Built from the *dataset being imitated*, not from a fresh pipeline: a DAgger episode is only
    useful if it is shaped like the demonstrations, and deriving the schema from anything else
    is how the two drift apart one column at a time.
    """
    features = dict(base_features)
    if IS_INTERVENTION_KEY in features:
        return features
    features[IS_INTERVENTION_KEY] = {
        'dtype': 'float32',
        'shape': (1,),
        'names': [IS_INTERVENTION_KEY],
    }
    return features


class DaggerFrameBuffer:
    """The expert steps of one rollout, kept in span order until the arm has stopped.

    Holds frames, not episodes: a span becomes an episode at flush time. Appending is all that
    happens inside the control loop, and appending is a list append.
    """

    def __init__(self, *, max_frames: int = DEFAULT_MAX_BUFFERED_FRAMES):
        self._max_frames = max(0, int(max_frames))
        self._spans: list[list[dict[str, Any]]] = []
        self._open = False
        self._frames = 0
        self._dropped = 0

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def dropped_frames(self) -> int:
        """Frames the cap refused. Non-zero means this rollout's corrections are incomplete."""
        return self._dropped

    @property
    def span_count(self) -> int:
        return len(self._spans)

    def append(self, frame: dict[str, Any], *, is_expert: bool) -> None:
        """Offer one control step. Only the expert's are kept; the rest close the open span."""
        if not is_expert:
            # The span ends the moment the policy is driving again, which is the same instant
            # `expert_spans` ends it. Closing here rather than counting a gap keeps the two
            # definitions of "a span" from having to agree by coincidence.
            self._open = False
            return
        if self._frames >= self._max_frames:
            self._dropped += 1
            # The span stays open: what follows the cap is still the same correction, and
            # reopening a new span for it would invent a boundary the operator never made.
            return
        if not self._open:
            self._spans.append([])
            self._open = True
        self._spans[-1].append(frame)
        self._frames += 1

    def spans(self) -> Iterator[list[dict[str, Any]]]:
        """Each span's frames, oldest first. Empty spans cannot occur: one is opened by a frame."""
        yield from self._spans

    def clear(self) -> None:
        self._spans = []
        self._open = False
        self._frames = 0
        self._dropped = 0


def build_dagger_frame(
    *,
    dataset_features: dict[str, dict],
    observation_values: dict[str, Any],
    action_values: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """One dataset frame, assembled the way ``record_loop`` assembles its own.

    Same helper, same prefixes, same ``task`` key -- because a frame that is merely *similar* to
    a recorded one is a frame that trains a policy on a schema it will not meet again.

    ``observation_values`` must carry each image already in the *dataset's* own geometry: the
    dataset being imitated is usually a training view, whose images are its crop of the camera
    rather than the raw frame. Nothing here can check that -- the shapes are only validated when
    the buffer is flushed, which is after the operator has driven the whole correction -- so the
    caller passes what the policy was shown (``build_policy_observation``'s output), not what the
    robot reported.
    """
    frame = {
        **build_dataset_frame(dataset_features, observation_values, prefix=OBSERVATION_PREFIX),
        **build_dataset_frame(dataset_features, action_values, prefix=ACTION_PREFIX),
        'task': task,
    }
    if IS_INTERVENTION_KEY in dataset_features:
        frame[IS_INTERVENTION_KEY] = np.array([1.0], dtype=np.float32)
    return frame


def image_source_keys(dataset_features: dict[str, dict]) -> list[str]:
    """The observation keys ``build_dataset_frame`` will look for images under.

    It resolves an image feature by stripping the ``observation.images.`` prefix and indexing
    the raw values with what is left, so the camera names the robot reports have to match the
    dataset's. Listing them here means a rollout whose cameras are named differently from the
    dataset's fails with a missing key at the first written frame, rather than silently writing
    episodes with the wrong camera in the wrong column.
    """
    prefix = f'{OBSERVATION_PREFIX}.images.'
    return [
        key.removeprefix(prefix)
        for key, feature in dataset_features.items()
        if key.startswith(prefix) and feature.get('dtype') in ('image', 'video')
    ]


class DaggerEpisodeWriter:
    """Writes the buffered spans of a rollout into a dataset, one episode per span.

    The dataset is passed in rather than opened here: whether it is created or extended is the
    launcher's decision (the recorder's own create-or-extend path), and a writer that opened its
    own would be a second place that decides where DAgger data lives.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        min_span_frames: int = 2,
        emit: Callable[[str], None] = print,
    ):
        self._dataset = dataset
        # A one-frame span is a bumped SpaceMouse, not a correction. Writing it costs an episode
        # of overhead to teach a single transition that the arm was already making.
        self._min_span_frames = max(1, int(min_span_frames))
        self._emit = emit

    def write(self, buffer: DaggerFrameBuffer, *, rollout_index: int) -> dict[str, Any]:
        """Save every span worth saving. Returns what happened, for the rollout's end marker."""
        written = 0
        skipped = 0
        frames_written = 0
        for span in buffer.spans():
            if len(span) < self._min_span_frames:
                skipped += 1
                continue
            for frame in span:
                self._dataset.add_frame(frame)
            # `parallel_encoding=False` for the reason the recorder gives at its own call site:
            # save_episode has been observed never to return with two cameras and parallel
            # encoding on.
            self._dataset.save_episode(parallel_encoding=False)
            written += 1
            frames_written += len(span)
        summary = {
            'episodes': written,
            'frames': frames_written,
            'skipped_spans': skipped,
            'dropped_frames': buffer.dropped_frames,
        }
        self._emit(
            f'[INFO] dagger_dataset_written rollout={rollout_index} episodes={written} '
            f'frames={frames_written} skipped_spans={skipped} dropped_frames={buffer.dropped_frames}'
        )
        if buffer.dropped_frames:
            self._emit(
                f'[WARN] dagger_dataset_truncated rollout={rollout_index} '
                f'dropped_frames={buffer.dropped_frames}: raise --dagger-max-buffered-frames if '
                'these corrections matter.'
            )
        return summary


def sent_command_to_dataset_action(
    sent_command: dict[str, float],
    *,
    T_B_Ws: np.ndarray,
    dataset_observation_i: dict[str, float],
    encode_delta: Callable[[dict[str, float], dict[str, float]], dict[str, float]] | None,
    denormalize_gripper: Callable[[float], float],
    previous_quaternion_xyzw: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """The command that was sent to the arm, expressed in the dataset's own action space.

    The rollout runs this chain forwards every step -- dataset action -> dataset frame -> base
    frame -> the arm. What is needed to record a human's step is the same chain backwards, and
    it has to be *exactly* backwards: an action written in a frame or a unit the dataset does
    not use is a sample that teaches the policy a systematic offset.

    ``encode_delta`` is the recorder's own ``AbsoluteEEToDeltaEEAction`` for a delta-action
    view, injected because it lives behind an import this module deliberately does not make. For
    an absolute-action view it is None and the quaternion is written directly.

    Returns the action values and the quaternion actually written, so the caller can pass it
    back as ``previous_quaternion_xyzw`` and keep the sign continuous across frames -- the same
    continuity the forward path maintains, and for the same reason: a quaternion that flips sign
    between two frames is a 360-degree rotation to anything that differences them.
    """
    base_position = np.asarray(
        [float(sent_command['ee.x']), float(sent_command['ee.y']), float(sent_command['ee.z'])],
        dtype=np.float64,
    )
    base_rotvec = np.asarray(
        [float(sent_command['ee.wx']), float(sent_command['ee.wy']), float(sent_command['ee.wz'])],
        dtype=np.float64,
    )
    base_pose = _pose_from_position_and_rotvec(base_position, base_rotvec)
    dataset_pose = _invert_pose(np.asarray(T_B_Ws, dtype=np.float64)) @ base_pose
    quaternion_xyzw = _continuous_quaternion(
        _quaternion_from_matrix(dataset_pose[:3, :3]),
        previous_quaternion_xyzw,
    )
    gripper_dataset_units = float(denormalize_gripper(float(sent_command['gripper.pos'])))

    absolute_action = {
        'ee.x': float(dataset_pose[0, 3]),
        'ee.y': float(dataset_pose[1, 3]),
        'ee.z': float(dataset_pose[2, 3]),
        'ee.qx': float(quaternion_xyzw[0]),
        'ee.qy': float(quaternion_xyzw[1]),
        'ee.qz': float(quaternion_xyzw[2]),
        'ee.qw': float(quaternion_xyzw[3]),
        'gripper': gripper_dataset_units,
        'gripper.pos': gripper_dataset_units,
    }
    if encode_delta is None:
        return absolute_action, quaternion_xyzw
    return dict(encode_delta(absolute_action, dataset_observation_i)), quaternion_xyzw


def _pose_from_position_and_rotvec(position_xyz: np.ndarray, rotvec_xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _matrix_from_rotvec(rotvec_xyz)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return pose


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    inverted = np.eye(4, dtype=np.float64)
    rotation = pose[:3, :3]
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ pose[:3, 3]
    return inverted


def _matrix_from_rotvec(rotvec_xyz: np.ndarray) -> np.ndarray:
    """Rodrigues, written out rather than imported.

    ``lerobot.utils.rotation`` has this, and ``command_guard`` imports it. Here the whole
    dependency would be three lines of trigonometry, and keeping it out means this module's
    tests are arithmetic against a matrix rather than against another implementation.
    """
    rotvec = np.asarray(rotvec_xyz, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to xyzw quaternion, via the largest-component branch.

    Branching on the largest diagonal term rather than always using the trace: the trace form
    divides by a quantity that vanishes at a 180-degree rotation, which is exactly the pose an
    arm reaches when it is asked to point the other way.
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _continuous_quaternion(
    quaternion_xyzw: np.ndarray,
    previous_xyzw: np.ndarray | None,
) -> np.ndarray:
    """The same rotation, signed to stay on the hemisphere the previous frame was on."""
    if previous_xyzw is None:
        return quaternion_xyzw
    if float(np.dot(quaternion_xyzw, np.asarray(previous_xyzw, dtype=np.float64))) < 0.0:
        return -quaternion_xyzw
    return quaternion_xyzw

