#!/usr/bin/env python

"""The check that stands between a dataset and a replay in the wrong tool frame.

`pika_task_tcp` and `pika_gripper_ee` are 410.85 mm apart on the same URDF and share an orientation,
and nothing in a LeRobot dataset records which of them its `ee.*` columns mean. Replaying one under
the other therefore does not fail -- it drives the fingertips to where the other frame's origin used
to be, for the whole episode, and then scores how well it tracked. This module is what turns that
into a refusal, so its own failure modes are the ones worth pinning: it must identify both frames
from the model rather than from a stored convention, and it must refuse to answer at all when the
data does not separate them.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.fr3.fr3_tool_frame_check import (
    MAX_HOME_ERROR_M,
    NEW_FRAME,
    OLD_FRAME,
    TOOL_FRAME_OFFSET_M,
    ToolFrameError,
    home_frame_positions,
    identify_frame,
    require_frame,
    workspace_clip_report,
)

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def home() -> dict[str, np.ndarray]:
    return home_frame_positions()


def test_the_two_home_positions_are_the_rigid_offset_apart(home):
    """If they were not, the check would be comparing against something other than the frames."""
    separation = float(np.linalg.norm(home[NEW_FRAME] - home[OLD_FRAME]))
    assert separation == pytest.approx(float(np.linalg.norm(TOOL_FRAME_OFFSET_M)), abs=1e-6)


def test_each_frames_own_home_pose_identifies_as_that_frame(home):
    for name, position in home.items():
        identified, errors = identify_frame(position, home)
        assert identified == name
        assert errors[name] == pytest.approx(0.0, abs=1e-9)


def test_a_recorded_start_pose_survives_the_jitter_the_rig_actually_has(home):
    """Episodes start within ~6 mm of home in practice; the check has to tolerate that and no more."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=0.002, size=(20, 3))

    assert identify_frame(home[OLD_FRAME] + jitter, home)[0] == OLD_FRAME
    assert identify_frame(home[NEW_FRAME] + jitter, home)[0] == NEW_FRAME


def test_require_frame_rejects_the_other_frame_with_the_distance_in_the_message(home):
    with pytest.raises(ToolFrameError) as excinfo:
        require_frame("episode", home[OLD_FRAME], NEW_FRAME, home)

    message = str(excinfo.value)
    assert OLD_FRAME in message and NEW_FRAME in message
    # The operator has to be able to tell "wrong frame" from "not near home at all".
    assert "410.8" in message or "410.9" in message


def test_require_frame_refuses_data_that_is_near_neither_frame(home):
    """A dataset whose episodes do not start from home cannot be vouched for either way."""
    nowhere = home[OLD_FRAME] + np.array([0.0, 0.0, 0.25])

    with pytest.raises(ToolFrameError, match="start from the home keyframe|neither frame"):
        require_frame("episode", nowhere, OLD_FRAME, home)


def test_require_frame_accepts_the_frame_it_is_given(home):
    errors = require_frame("episode", home[NEW_FRAME], NEW_FRAME, home)

    assert errors[NEW_FRAME] < MAX_HOME_ERROR_M
    assert errors[OLD_FRAME] > 0.4


def test_workspace_clip_report_counts_only_what_the_clip_would_move():
    positions = np.array([[0.3, 0.0, 0.2], [0.9, 0.0, 0.2], [0.3, 0.0, -0.1]])

    outside, worst_m, details = workspace_clip_report(positions, (0.18, -0.45, 0.0), (0.70, 0.45, 0.70))

    assert outside == 2
    assert worst_m == pytest.approx(0.2, abs=1e-9)
    assert any("x > 0.700" in detail for detail in details)
    assert any("z < 0.000" in detail for detail in details)


def test_workspace_clip_report_is_silent_when_the_box_contains_the_trajectory():
    positions = np.array([[0.3, 0.0, 0.2], [0.45, -0.2, 0.35]])

    outside, worst_m, details = workspace_clip_report(positions, (0.18, -0.45, 0.0), (0.70, 0.45, 0.70))

    assert (outside, worst_m, details) == (0, 0.0, [])


def test_the_recording_config_and_the_check_agree_on_the_frame():
    """The preflight compares data against the config; this pins that the config is the new frame.

    Together with test_fr3_recording_workspace_contract.py, a revert of the record config would fail
    there rather than turn this check into one that quietly demands the old frame.
    """
    import yaml

    config = yaml.safe_load(Path("tools/fr3/fr3_record_config.yaml").read_text())
    assert config["robot"]["target_frame_name"] == NEW_FRAME
