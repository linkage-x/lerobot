"""pseudo_bimanual (roadmap A5 ①): stand-in two-hand sessions from recorded one-hand episodes."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import fr3_kinematics as kin
from tools.thor.gmsl2 import bimanual_qc as bq, pseudo_bimanual as pb, world_provenance as wp

TOOL = kin.urdf_fixed_transform(pb.DEFAULT_TOOL_FRAME)
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
WORLD = "world_test"
# the camera world 1 m in front of the base, turned 90 deg -- anything rigid will do
T_BASE_WORLD = [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]


def _graph() -> dict:
    return {
        "version": 1,
        "nodes": [
            {"world_frame_id": WORLD, "reason": "frozen_reference"},
            {"world_frame_id": pb.DATASET_WORLD, "reason": "robot_base"},
        ],
        "edges": [
            {
                "from_world_frame_id": WORLD,
                "to_world_frame_id": pb.DATASET_WORLD,
                "T_to_from": T_BASE_WORLD,
                "method": "test",
            }
        ],
    }


def _pose7(T: np.ndarray) -> list[float]:
    return [*T[:3, 3].tolist(), *wp._quat_from_matrix(T[:3, :3].tolist())]


def _tcp_path(n: int, phase: float = 0.0, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    s = np.sin(np.linspace(0.0, 2 * np.pi, n) + phase)
    q = np.tile(kin.Q_READY, (n, 1))
    q[:, 0] += 0.3 * s
    q[:, 3] += 0.2 * s
    T = kin.NOMINAL.fk(q) @ TOOL
    T[:, :3, 3] += offset
    return T


def _write_session(root: Path, name: str, tcp_in_base: np.ndarray) -> Path:
    session = root / name
    world_from_base = np.linalg.inv(np.array(T_BASE_WORLD))
    fields = [
        "schema_version", "timestamp_s", "num_cameras", "solve_ok", "camera_set", "camera_set_changed",
        "frame_global_index", "episode_index", "frame_index",
        "state_x_m", "state_y_m", "state_z_m", "state_qx", "state_qy", "state_qz", "state_qw",
    ]  # fmt: skip
    rows = []
    for k, T in enumerate(tcp_in_base):
        pose = _pose7(world_from_base @ T)
        rows.append(
            dict(zip(fields, [3, k / 60.0, 5, True, "06,07,08,09,13", False, k, 0, k, *pose], strict=True))
        )
    path = session / "derived" / bq.SIDECAR / "state_action.right.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (session / "meta").mkdir()
    (session / "meta" / "info.json").write_text(
        json.dumps({"fps": 60, "world_frame": {"world_frame_id": WORLD, "status": "ok"}})
    )
    return session


def test_mirrored_at_the_flange_the_left_arm_reaches_it_with_the_right_arms_joints_mirrored() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(kin.Q_LOWER, kin.Q_UPPER)
        mirrored = np.array(pb._matrix(pb.mirror_pose7(_pose7(kin.NOMINAL.fk(q)[0] @ TOOL), 0.4, TOOL)))
        mirrored[1, 3] -= 0.8  # into the left base, 0.8 m along +y
        assert np.allclose(mirrored, kin.NOMINAL.fk(q * MIRROR)[0] @ TOOL, atol=1e-12)


def test_mirroring_the_tcp_itself_is_a_proper_rotation_and_keeps_gaps() -> None:
    pose = [0.5, 0.1, 0.3, *wp._quat_from_matrix(kin._rpy_matrix(0.3, -0.2, 0.7).tolist())]
    x, y, z, *q = pb.mirror_pose7(pose, 0.4)
    assert (x, y, z) == (0.5, pytest.approx(0.7), 0.3)
    R = np.array(pb._matrix([0, 0, 0, *q]))[:3, :3]
    S = np.diag([1.0, -1.0, 1.0])
    assert np.allclose(R, S @ np.array(pb._matrix(pose))[:3, :3] @ S) and np.linalg.det(R) == pytest.approx(
        1.0
    )
    gap = [math.nan] * 7
    assert all(math.isnan(v) for v in pb.mirror_pose7(gap, 0.4, TOOL))


def test_a_built_pair_is_a_session_that_qc_reads_and_says_what_it_is(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _write_session(root, "take_a", _tcp_path(360))
    _write_session(root, "take_b", _tcp_path(330, phase=1.0))
    _write_session(root, "far_away", _tcp_path(330, offset=(0.6, 0.6, 0.0)))
    found, skipped = pb.candidates(root, _graph())
    assert sorted(c["session"] for c in found) == ["take_a", "take_b"]
    assert any("far_away" in s and "out of reach" in s for s in skipped)
    pairs = pb.auto_pairs(found, 1, seed=0)
    assert {pairs[0][0]["session"], pairs[0][1]["session"]} == {"take_a", "take_b"}

    session = pb.build(root, pairs, tmp_path / "out", graph=_graph(), name="pseudo_bimanual_test")
    info = json.loads((session / "meta" / "info.json").read_text())
    assert pb.is_pseudo(session) and info["world_frame"]["world_frame_id"] == pb.DATASET_WORLD
    assert info["pseudo_bimanual"]["not_for"] == list(pb.NOT_FOR)
    report = bq.session_qc(session)
    assert report["episodes"]["0"]["frames"] == 330  # the shorter half
    assert report["episodes"]["0"]["bimanual_pass"]

    # the left hand is the other take, in fr3_base, mirrored at the flange
    left_src = pairs[0][1]
    source = _tcp_path(
        360 if left_src["session"] == "take_a" else 330, phase=0.0 if left_src["session"] == "take_a" else 1.0
    )
    left = bq.read_hand(session / "derived" / bq.SIDECAR / "state_action.left.csv")[0]
    expected = pb.mirror_pose7(_pose7(source[7]), pb.DEFAULT_SPACING_M / 2, TOOL)
    assert np.allclose(left[7]["p"], expected[:3], atol=1e-9)


def test_a_stand_in_session_must_say_so_in_its_name(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _write_session(root, "take_a", _tcp_path(320))
    pair = ({"session": "take_a", "episode": 0}, {"session": "take_a", "episode": 0})
    with pytest.raises(ValueError, match="must start with"):
        pb.build(root, [pair], tmp_path / "out", graph=_graph(), name="looks_real")


def test_dh_flange_and_tool_are_the_urdfs() -> None:
    rng = np.random.default_rng(1)
    q = rng.uniform(kin.Q_LOWER, kin.Q_UPPER)
    # the URDF chain to the tool, composed by hand: fixed origins, revolute joints about z
    import xml.etree.ElementTree as ET

    root = ET.parse(str(kin.FR3_URDF)).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], pb.DEFAULT_TOOL_FRAME
    while link in joints:
        chain.append(joints[link])
        link = joints[link].find("parent").get("link")
    T, i = np.eye(4), 0
    for joint in reversed(chain):
        origin = joint.find("origin")
        step = np.eye(4)
        step[:3, :3] = kin._rpy_matrix(*[float(v) for v in origin.get("rpy").split()])
        step[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        T = T @ step
        if joint.get("type") == "revolute":
            c, s = math.cos(q[i]), math.sin(q[i])
            T = T @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            i += 1
    assert i == 7 and np.allclose(kin.NOMINAL.fk(q)[0] @ TOOL, T, atol=1e-12)
    with pytest.raises(ValueError, match="not fixed"):
        kin.urdf_fixed_transform(pb.DEFAULT_TOOL_FRAME, parent_link="fr3_link5")
