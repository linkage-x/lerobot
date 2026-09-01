"""What a takeover leaves behind that a policy can be trained on.

``test_fr3_dagger_takeover.py`` checks what the device turns into, and
``test_fr3_dagger_handoff.py`` checks what the guard does to it. This file checks the last
step: that the command which actually reached the arm comes back out in the dataset's own
frame, units and action space, and that the spans an operator drove become episodes with the
boundaries the trace says they had.

The arithmetic here is the *inverse* of a chain the rollout runs forwards every step, so most
of these tests are round trips: push a pose out through the forward conversion, pull it back,
and require it to be the pose that went in. An inverse that is merely plausible is the failure
this file exists to catch -- a systematic offset in a training set does not look like a bug, it
looks like a policy that never quite reaches.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.fr3.dagger_dataset import (
    DEFAULT_MAX_BUFFERED_FRAMES,
    IS_INTERVENTION_KEY,
    DaggerEpisodeWriter,
    DaggerFrameBuffer,
    build_dagger_frame,
    dagger_dataset_features,
    sent_command_to_dataset_action,
)


class FakeDataset:
    """Records what the writer asked of it, in order."""

    def __init__(self):
        self.frames: list[dict] = []
        self.saved_episodes: list[list[dict]] = []
        self.save_kwargs: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self, **kwargs) -> None:
        self.save_kwargs.append(kwargs)
        self.saved_episodes.append(self.frames)
        self.frames = []


def command(x=0.400, y=-0.100, z=0.070, rotvec=(0.0, 0.0, 0.0), gripper=1.0):
    return {
        'ee.x': x,
        'ee.y': y,
        'ee.z': z,
        'ee.wx': rotvec[0],
        'ee.wy': rotvec[1],
        'ee.wz': rotvec[2],
        'gripper.pos': gripper,
    }


def pose(position, rotvec):
    """The same pose construction the runtime uses, written independently for the round trips."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        rotation = np.eye(3)
    else:
        axis = rotvec / angle
        cross = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def identity_gripper(value: float) -> float:
    return value


# --- the buffer: which steps are kept, and where a span begins ---------------------------


def test_only_the_steps_the_operator_drove_are_kept():
    buffer = DaggerFrameBuffer()

    buffer.append({'n': 0}, is_expert=False)
    buffer.append({'n': 1}, is_expert=True)
    buffer.append({'n': 2}, is_expert=True)
    buffer.append({'n': 3}, is_expert=False)

    assert buffer.frame_count == 2
    assert [frame['n'] for span in buffer.spans() for frame in span] == [1, 2]


def test_two_corrections_separated_by_the_policy_are_two_spans():
    buffer = DaggerFrameBuffer()

    for is_expert in (True, True, False, False, True):
        buffer.append({'expert': is_expert}, is_expert=is_expert)

    assert buffer.span_count == 2
    assert [len(span) for span in buffer.spans()] == [2, 1]


def test_a_correction_that_never_pauses_stays_one_span():
    buffer = DaggerFrameBuffer()

    for _ in range(5):
        buffer.append({}, is_expert=True)

    assert buffer.span_count == 1
    assert buffer.frame_count == 5


def test_the_cap_drops_frames_and_says_so_instead_of_growing():
    buffer = DaggerFrameBuffer(max_frames=2)

    for index in range(5):
        buffer.append({'n': index}, is_expert=True)

    assert buffer.frame_count == 2
    assert buffer.dropped_frames == 3
    # The correction did not become two corrections just because the cap interrupted it.
    assert buffer.span_count == 1


def test_frames_dropped_by_the_cap_do_not_open_a_new_span_when_room_returns():
    buffer = DaggerFrameBuffer(max_frames=1)
    buffer.append({'n': 0}, is_expert=True)
    buffer.append({'n': 1}, is_expert=True)

    buffer.clear()
    buffer.append({'n': 2}, is_expert=True)

    assert buffer.span_count == 1
    assert buffer.dropped_frames == 0


def test_the_default_cap_is_seconds_of_correction_not_a_whole_rollout():
    # 450 frames at 30 Hz is 15 s. The point of the assertion is the order of magnitude: a cap
    # of a whole rollout would be the memory blow-up the buffer exists to bound.
    assert 200 <= DEFAULT_MAX_BUFFERED_FRAMES <= 1200


# --- the writer: one episode per span ----------------------------------------------------


def test_each_span_becomes_its_own_episode():
    buffer = DaggerFrameBuffer()
    for is_expert in (True, True, False, True, True, True):
        buffer.append({'expert': is_expert}, is_expert=is_expert)
    dataset = FakeDataset()

    summary = DaggerEpisodeWriter(dataset, emit=lambda _: None).write(buffer, rollout_index=3)

    assert summary['episodes'] == 2
    assert summary['frames'] == 5
    assert [len(episode) for episode in dataset.saved_episodes] == [2, 3]


def test_the_spans_are_not_concatenated_into_one_episode():
    """Two corrections in one episode would put a seam in the middle of a trajectory.

    The frames either side of that seam are seconds and centimetres apart; a policy trained
    across it learns a transition the arm never made.
    """
    buffer = DaggerFrameBuffer()
    buffer.append({'n': 0}, is_expert=True)
    buffer.append({'n': 1}, is_expert=True)
    buffer.append({'n': 2}, is_expert=False)
    buffer.append({'n': 3}, is_expert=True)
    buffer.append({'n': 4}, is_expert=True)
    dataset = FakeDataset()

    DaggerEpisodeWriter(dataset, emit=lambda _: None).write(buffer, rollout_index=0)

    assert len(dataset.saved_episodes) == 2
    assert [frame['n'] for frame in dataset.saved_episodes[0]] == [0, 1]
    assert [frame['n'] for frame in dataset.saved_episodes[1]] == [3, 4]


def test_a_single_frame_span_is_a_bumped_device_not_a_correction():
    buffer = DaggerFrameBuffer()
    buffer.append({}, is_expert=True)
    buffer.append({}, is_expert=False)
    buffer.append({}, is_expert=True)
    buffer.append({}, is_expert=True)
    dataset = FakeDataset()

    summary = DaggerEpisodeWriter(dataset, emit=lambda _: None).write(buffer, rollout_index=1)

    assert summary['episodes'] == 1
    assert summary['skipped_spans'] == 1
    assert len(dataset.saved_episodes) == 1


def test_episodes_are_saved_without_parallel_encoding():
    """The recorder's own call site says save_episode has been seen never to return otherwise."""
    buffer = DaggerFrameBuffer()
    buffer.append({}, is_expert=True)
    buffer.append({}, is_expert=True)
    dataset = FakeDataset()

    DaggerEpisodeWriter(dataset, emit=lambda _: None).write(buffer, rollout_index=0)

    assert dataset.save_kwargs == [{'parallel_encoding': False}]


def test_a_rollout_with_no_takeover_writes_nothing():
    buffer = DaggerFrameBuffer()
    for _ in range(10):
        buffer.append({}, is_expert=False)
    dataset = FakeDataset()

    summary = DaggerEpisodeWriter(dataset, emit=lambda _: None).write(buffer, rollout_index=0)

    assert summary == {'episodes': 0, 'frames': 0, 'skipped_spans': 0, 'dropped_frames': 0}
    assert dataset.saved_episodes == []


def test_dropped_frames_are_reported_on_the_writer_line():
    buffer = DaggerFrameBuffer(max_frames=2)
    for _ in range(5):
        buffer.append({}, is_expert=True)
    dataset = FakeDataset()
    lines: list[str] = []

    summary = DaggerEpisodeWriter(dataset, emit=lines.append).write(buffer, rollout_index=7)

    assert summary['dropped_frames'] == 3
    assert any('dagger_dataset_truncated' in line for line in lines)


# --- the action: the dataset's frame, the dataset's units --------------------------------


def test_with_no_frame_offset_the_sent_command_is_the_dataset_action():
    action, _ = sent_command_to_dataset_action(
        command(x=0.4, y=-0.1, z=0.07),
        T_B_Ws=np.eye(4),
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=identity_gripper,
    )

    assert action['ee.x'] == pytest.approx(0.4)
    assert action['ee.y'] == pytest.approx(-0.1)
    assert action['ee.z'] == pytest.approx(0.07)
    assert action['ee.qw'] == pytest.approx(1.0)


def test_the_command_comes_back_in_the_frame_the_dataset_was_recorded_in():
    """Round trip: a dataset pose pushed out to the base frame must come back unchanged.

    This is the test that matters. The rollout maps dataset -> base with ``T_B_Ws`` every step;
    if the inverse here is not exactly that map, every DAgger sample carries the same offset,
    and a training set with a constant offset trains a policy that consistently misses.
    """
    T_B_Ws = pose([0.15, -0.35, 0.02], [0.0, 0.0, 0.7])
    dataset_pose = pose([0.31, 0.12, 0.44], [0.2, -0.1, 0.3])
    base_pose = T_B_Ws @ dataset_pose
    base_rotvec = _rotvec_from_matrix(base_pose[:3, :3])

    action, _ = sent_command_to_dataset_action(
        command(
            x=base_pose[0, 3],
            y=base_pose[1, 3],
            z=base_pose[2, 3],
            rotvec=tuple(base_rotvec),
        ),
        T_B_Ws=T_B_Ws,
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=identity_gripper,
    )

    assert action['ee.x'] == pytest.approx(dataset_pose[0, 3], abs=1e-9)
    assert action['ee.y'] == pytest.approx(dataset_pose[1, 3], abs=1e-9)
    assert action['ee.z'] == pytest.approx(dataset_pose[2, 3], abs=1e-9)

    rebuilt = _matrix_from_quaternion(
        np.array([action['ee.qx'], action['ee.qy'], action['ee.qz'], action['ee.qw']])
    )
    assert np.allclose(rebuilt, dataset_pose[:3, :3], atol=1e-9)


def test_a_half_turn_survives_the_matrix_to_quaternion_branch():
    """180 degrees is where the trace-only formula divides by zero -- and where an arm points back."""
    T_B_Ws = np.eye(4)
    dataset_pose = pose([0.3, 0.0, 0.4], [np.pi, 0.0, 0.0])
    base_rotvec = _rotvec_from_matrix(dataset_pose[:3, :3])

    action, _ = sent_command_to_dataset_action(
        command(x=0.3, y=0.0, z=0.4, rotvec=tuple(base_rotvec)),
        T_B_Ws=T_B_Ws,
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=identity_gripper,
    )

    rebuilt = _matrix_from_quaternion(
        np.array([action['ee.qx'], action['ee.qy'], action['ee.qz'], action['ee.qw']])
    )
    assert np.allclose(rebuilt, dataset_pose[:3, :3], atol=1e-8)


def test_the_quaternion_keeps_the_sign_the_previous_frame_had():
    T_B_Ws = np.eye(4)
    rotvec = _rotvec_from_matrix(pose([0, 0, 0], [0.0, 0.0, 2.0])[:3, :3])

    first, quaternion = sent_command_to_dataset_action(
        command(rotvec=tuple(rotvec)),
        T_B_Ws=T_B_Ws,
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=identity_gripper,
    )
    flipped, _ = sent_command_to_dataset_action(
        command(rotvec=tuple(rotvec)),
        T_B_Ws=T_B_Ws,
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=identity_gripper,
        previous_quaternion_xyzw=-quaternion,
    )

    assert np.dot(
        [first['ee.qx'], first['ee.qy'], first['ee.qz'], first['ee.qw']],
        [flipped['ee.qx'], flipped['ee.qy'], flipped['ee.qz'], flipped['ee.qw']],
    ) < 0.0


def test_the_gripper_is_written_in_the_units_the_dataset_uses():
    """The command carries a normalized gripper; the dataset may not. The inverse is injected."""
    action, _ = sent_command_to_dataset_action(
        command(gripper=0.5),
        T_B_Ws=np.eye(4),
        dataset_observation_i={},
        encode_delta=None,
        denormalize_gripper=lambda value: value * 80.0,
    )

    assert action['gripper'] == pytest.approx(40.0)
    assert action['gripper.pos'] == pytest.approx(40.0)


def test_a_delta_view_is_encoded_by_the_recorders_own_step():
    """No second delta implementation: the injected encoder is what recording used."""
    seen: dict = {}

    def encode_delta(absolute_action, observation):
        seen['absolute'] = absolute_action
        seen['observation'] = observation
        return {'delta_prev_cmd.ee.x': 0.001, 'gripper.pos': absolute_action['gripper.pos']}

    action, _ = sent_command_to_dataset_action(
        command(x=0.42),
        T_B_Ws=np.eye(4),
        dataset_observation_i={'prev_cmd.ee.x': 0.41},
        encode_delta=encode_delta,
        denormalize_gripper=identity_gripper,
    )

    assert action == {'delta_prev_cmd.ee.x': 0.001, 'gripper.pos': 1.0}
    assert seen['absolute']['ee.x'] == pytest.approx(0.42)
    assert seen['observation'] == {'prev_cmd.ee.x': 0.41}


# --- the schema ---------------------------------------------------------------------------


def test_the_schema_is_the_recorders_plus_one_column():
    base = {'observation.state': {'dtype': 'float32', 'shape': (2,), 'names': ['a', 'b']}}

    features = dagger_dataset_features(base)

    assert 'observation.state' in features
    assert features[IS_INTERVENTION_KEY]['shape'] == (1,)
    assert base == {'observation.state': {'dtype': 'float32', 'shape': (2,), 'names': ['a', 'b']}}


def test_adding_the_column_twice_does_not_change_the_schema():
    once = dagger_dataset_features({'action': {'dtype': 'float32', 'shape': (1,), 'names': ['x']}})

    assert dagger_dataset_features(once) == once


def test_a_frame_carries_observation_action_task_and_the_intervention_flag():
    features = dagger_dataset_features(
        {
            'observation.state': {'dtype': 'float32', 'shape': (2,), 'names': ['ee.x', 'ee.y']},
            'action': {'dtype': 'float32', 'shape': (2,), 'names': ['ee.x', 'ee.y']},
        }
    )

    frame = build_dagger_frame(
        dataset_features=features,
        observation_values={'ee.x': 0.4, 'ee.y': -0.1},
        action_values={'ee.x': 0.41, 'ee.y': -0.11},
        task='pick the cube',
    )

    assert np.allclose(frame['observation.state'], [0.4, -0.1])
    assert np.allclose(frame['action'], [0.41, -0.11])
    assert frame['task'] == 'pick the cube'
    assert frame[IS_INTERVENTION_KEY] == np.array([1.0], dtype=np.float32)


# --- helpers used only by the round trips --------------------------------------------------


def _rotvec_from_matrix(matrix: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-8:
        # Near pi the skew-symmetric part vanishes; recover the axis from the symmetric part.
        axis = np.sqrt(np.clip(np.diag(matrix + np.eye(3)) / 2.0, 0.0, None))
        largest = int(np.argmax(axis))
        axis = axis * np.sign(matrix[(largest + 2) % 3, (largest + 1) % 3] or 1.0)
        return axis / np.linalg.norm(axis) * angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def _matrix_from_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
