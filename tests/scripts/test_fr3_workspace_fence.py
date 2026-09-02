#!/usr/bin/env python

"""Which box the rollout runs in, and whether anything says so when it is the wrong one.

The driver clips every commanded pose to ``workspace_min/max`` and reports the *clipped* pose back
as ``prev_cmd``. A fence set too tight therefore raises nothing, logs nothing, and passes the step
and leash guards: the arm stops short of where the policy asked to go and every number the rollout
prints still reads healthy. That is what happened -- the runtime carried its own literal at
``z >= 0.05`` while the rig records with ``z >= 0`` -- and it is invisible by construction, so what
these tests pin is the derivation rather than any single number.
"""

from pathlib import Path

import pytest
import yaml

from tools.fr3.workspace_fence import (
    DEFAULT_WORKSPACE_MIN,
    resolve_workspace_fence,
)

RECORD_CONFIG = Path("tools/fr3/fr3_record_config.yaml")

# The lowest tool-point z in the recorded pick-and-place frames, spanning z [0.028, 0.397]. The
# same number is derived and explained in tests/robots/test_fr3_recording_workspace_contract.py;
# repeated here because this is the depth a *rollout* has to be allowed to reach, which is a
# different claim from the one that file makes about the recording rig.
#
# It is the pick-and-place task's number, and it stays the bound here because one fence serves
# every task recorded on this rig. The task currently gated on is the shallower one:
# `fr3_spacemouse-insert` bottoms out at 40.7 mm, closes at a median 50.7 mm, and puts 30 of its
# 50 episodes below 50 mm -- which is what the old rollout fence at z >= 0.05 stood above.
DEEPEST_RECORDED_FRAME_M = 0.028
INSERT_DEEPEST_FRAME_M = 0.0407


def write_config(tmp_path: Path, robot: dict | None) -> Path:
    payload: dict = {} if robot is None else {"robot": robot}
    path = tmp_path / "record_config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_the_rollout_runs_the_fence_the_rig_records_with():
    """One derivation, not two. The bug was two copies drifting apart with nobody editing either."""
    robot = yaml.safe_load(RECORD_CONFIG.read_text(encoding="utf-8"))["robot"]

    minimum, maximum, source = resolve_workspace_fence(record_config_path=RECORD_CONFIG)

    assert minimum == tuple(robot["workspace_min"])
    assert maximum == tuple(robot["workspace_max"])
    assert str(RECORD_CONFIG) in source


def test_the_resolved_fence_contains_the_depth_the_demonstrations_reached():
    """The failure this module exists to remove, stated as the thing an operator would notice.

    A floor above the recorded frames does not refuse the descent; it shortens it. The policy is
    then asked to reproduce a grasp it is physically prevented from reaching, and the rollout
    reports success on every guard while bottoming out in mid-air.
    """
    minimum, _maximum, _source = resolve_workspace_fence(record_config_path=RECORD_CONFIG)

    assert minimum[2] <= DEEPEST_RECORDED_FRAME_M, (
        f"the rollout fence floors at z={minimum[2] * 1e3:.0f} mm but the demonstrations reach "
        f"z={DEEPEST_RECORDED_FRAME_M * 1e3:.0f} mm. Every command below the floor is clipped by "
        "the driver and reported back as if it had been sent"
    )
    assert minimum[2] <= INSERT_DEEPEST_FRAME_M, (
        f'the rollout fence floors at z={minimum[2] * 1e3:.0f} mm, above the '
        f'{INSERT_DEEPEST_FRAME_M * 1e3:.1f} mm the insert demonstrations reach'
    )
    assert DEFAULT_WORKSPACE_MIN[2] > DEEPEST_RECORDED_FRAME_M, (
        "the built-in default no longer differs from the recording rig on z, which is the drift "
        "these tests exist to catch -- pick a different assertion rather than deleting this one"
    )


def test_naming_no_record_config_falls_back_to_the_default_and_says_which_it_is():
    """The other FR3 rig's launcher names no config. It gets the default, and the banner says so."""
    minimum, _maximum, source = resolve_workspace_fence()

    assert minimum == DEFAULT_WORKSPACE_MIN
    assert "default" in source


def test_an_explicit_pair_replaces_the_record_config_entirely(tmp_path):
    config = write_config(tmp_path, {"workspace_min": [0.18, -0.45, 0.0], "workspace_max": [0.70, 0.45, 0.70]})

    minimum, maximum, source = resolve_workspace_fence(
        record_config_path=config,
        workspace_min=[0.20, -0.40, 0.01],
        workspace_max=[0.60, 0.40, 0.60],
    )

    assert minimum == (0.20, -0.40, 0.01)
    assert maximum == (0.60, 0.40, 0.60)
    assert source == "command line"


def test_half_an_explicit_pair_is_refused_rather_than_mixed_with_the_file():
    # Taking one corner from the operator and the other from a config produces a box neither of
    # them wrote, and it would be reported as coming from whichever half the source string names.
    with pytest.raises(ValueError, match="together"):
        resolve_workspace_fence(workspace_min=[0.20, -0.40, 0.01])


def test_a_record_config_without_a_fence_raises_rather_than_falling_back(tmp_path):
    """A fallback here is how the two-copies bug comes back: a config that names no fence would
    silently run the default's z >= 0.05 while the operator believes they named the rig."""
    config = write_config(tmp_path, {"target_frame_name": "pika_gripper_ee"})

    with pytest.raises(ValueError, match="workspace_min"):
        resolve_workspace_fence(record_config_path=config)


def test_a_config_with_no_robot_block_names_the_file_it_could_not_read(tmp_path):
    config = write_config(tmp_path, None)

    with pytest.raises(ValueError, match=str(config)):
        resolve_workspace_fence(record_config_path=config)


def test_an_inverted_axis_names_the_axis_and_where_it_came_from(tmp_path):
    # An empty box is not caught by the driver's own check until the config is built, several
    # hundred lines later, as a dataclass error about numbers nobody typed at the command line.
    config = write_config(tmp_path, {"workspace_min": [0.18, -0.45, 0.30], "workspace_max": [0.70, 0.45, 0.10]})

    with pytest.raises(ValueError, match="axis z"):
        resolve_workspace_fence(record_config_path=config)


def test_a_corner_that_is_not_three_numbers_is_refused(tmp_path):
    config = write_config(tmp_path, {"workspace_min": [0.18, -0.45], "workspace_max": [0.70, 0.45, 0.70]})

    with pytest.raises(ValueError, match="three numbers"):
        resolve_workspace_fence(record_config_path=config)
