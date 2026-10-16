"""fr3_tcp_pivot collect: a start pose past panda_py's teaching-mode walls is refused before it aborts."""

from __future__ import annotations

import numpy as np

from tools.fr3 import fr3_tcp_pivot as tp

# panda_py.constants on Thor (panda_python 0.8.1): the Panda's limits, not the FR3's.
PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def test_the_10_13_start_pose_is_refused_and_says_which_joint_and_how_far():
    q = np.array([0.0, -1.80144, 0.0, -2.0, 0.0, 1.6, 0.8])
    beyond, close = tp.teaching_wall_problems(q, PANDA_LOWER, PANDA_UPPER)
    assert len(beyond) == 1 and close == []
    assert beyond[0].startswith("j2 = -103.2° is below panda_py's teaching-mode wall -101.0°")
    assert "FR3 [-102.2°, 102.2°]" in beyond[0]
    assert "at least 5.2° back inside" in beyond[0]


def test_a_joint_near_a_wall_is_a_warning_and_home_is_clean():
    home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    assert tp.teaching_wall_problems(home, PANDA_LOWER, PANDA_UPPER) == ([], [])
    near_j6 = home.copy()
    near_j6[5] = np.radians(213.6)  # the 10-13 pivot sat here
    beyond, close = tp.teaching_wall_problems(near_j6, PANDA_LOWER, PANDA_UPPER)
    assert beyond == [] and close == ["j6 = 213.6° is within 3° of the teaching-mode wall"]
