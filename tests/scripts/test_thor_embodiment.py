"""Embodiment descriptions: roots, arms, and which replay modes the world graph allows."""

import copy
import json
import math
from pathlib import Path

import pytest

from tools.thor.gmsl2 import embodiment as em, world_provenance as wp

REPO = Path(__file__).resolve().parents[2]


def _rz(deg: float, t=(0.0, 0.0, 0.0)) -> list[list[float]]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0, t[0]], [s, c, 0.0, t[1]], [0.0, 0.0, 1.0, t[2]], [0.0, 0.0, 0.0, 1.0]]


def _edge(a: str, b: str, T, **extra) -> dict:
    return {"from_world_frame_id": a, "to_world_frame_id": b, "T_to_from": T, "method": f"{a}->{b}", **extra}


def _dual(right_config: str = "tools/fr3/replay_cube_pose_in_robot_base.thor.right.yaml") -> dict:
    """The tracked two-arm description: since 10-26 the left arm is in fr3_base, the right one is placed."""
    description = json.loads((REPO / em.EMBODIMENT_DIR / "fr3_dual.json").read_text(encoding="utf-8"))
    description["arms"]["right"]["replay_config"] = right_config
    return description


@pytest.mark.parametrize("name", ["fr3_single", "fr3_dual"])
def test_the_tracked_descriptions_load(name: str) -> None:
    description = em.load(em.EMBODIMENT_DIR / f"{name}.json", repo_root=REPO)
    assert description["dataset_frame"] == wp.FR3_BASE_WORLD_ID


def test_the_single_fr3_points_at_the_replay_config_in_service() -> None:
    description = em.load(em.EMBODIMENT_DIR / "fr3_single.json", repo_root=REPO)
    (arm,) = description["arms"].values()
    assert arm["root"] == wp.FR3_BASE_WORLD_ID
    config = (REPO / arm["replay_config"]).read_text(encoding="utf-8")
    assert f"target_frame_name: {arm['tcp_frame']}" in config


def test_validate_names_what_does_not_hold_together(tmp_path: Path) -> None:
    description = _dual()
    description["arms"]["left"]["hands"] = ["right"]
    description["roots"]["fr3_base_right"]["kind"] = "mobile"
    description["arms"]["left"]["replay_config"] = "no/such.yaml"
    problems = " | ".join(em.validate(description, repo_root=REPO))
    assert "hand right is already replayed by arm right" in problems
    assert "a mobile root cannot hold a static edge" in problems
    assert "replay_config no/such.yaml does not exist" in problems
    bad = copy.deepcopy(description)
    bad["dataset_frame"] = "nowhere"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(em.EmbodimentError, match="dataset_frame 'nowhere'"):
        em.load(path, repo_root=REPO)


def test_bimanual_replay_needs_the_base_to_base_edge_for_relative_and_world_edges_for_absolute() -> None:
    description = _dual()
    graph = {"edges": [_edge("world", "fr3_base", _rz(90, (1.2, 0, 0.7)))]}
    modes = em.replay_modes(description, graph, "fr3_base", ["left", "right"])
    assert modes["arms"] == {"left": "left", "right": "right"}
    assert not modes["relative"]["ok"] and "B3" in modes["relative"]["reason"]
    assert not modes["absolute"]["ok"]

    # the right base 0.8 m along -y of the left: fr3_base's origin is 0.8 m along +y in it
    graph["edges"].append(_edge("fr3_base", "fr3_base_right", _rz(0, (0, 0.8, 0)), sigma_mm=0.5))
    modes = em.replay_modes(description, graph, "fr3_base", ["left", "right"])
    assert modes["relative"]["ok"] and [e["to_world_frame_id"] for e in modes["relative"]["edges"]] == ["fr3_base_right"]
    # the dataset is in the left base, so absolute needs only the base-to-base edge too
    assert modes["absolute"]["ok"]
    T, edges, _ = em.arm_transform(description, graph, "fr3_base", "right")
    assert T[1][3] == pytest.approx(0.8) and len(edges) == 1


def test_an_arm_not_in_service_or_one_arm_for_two_hands_is_refused() -> None:
    modes = em.replay_modes(_dual(right_config=""), {"edges": []}, "fr3_base", ["left", "right"])
    assert "not in service" in modes["absolute"]["reason"]
    single = em.load(em.EMBODIMENT_DIR / "fr3_single.json", repo_root=REPO)
    assert em.replay_modes(single, {"edges": []}, "fr3_base", ["left"])["absolute"]["ok"]
    assert "two hands at once" in em.replay_modes(single, {"edges": []}, "fr3_base", ["left", "right"])["relative"]["reason"]


def test_a_mobile_root_has_no_edge_to_read() -> None:
    description = _dual()
    description["roots"]["fr3_base_right"].update(kind="mobile", registration="session_edge")
    graph = {"edges": [_edge("fr3_base", "fr3_base_right", _rz(0, (0, 0.8, 0)))]}
    T, _, reason = em.arm_transform(description, graph, "fr3_base", "right")
    assert T is None and "per session_edge" in reason


# --- a body-mounted arm on a mobile root (Monte02, roadmap docs/monte02_left_arm_roadmap.html M1.5) --------

MONTE02 = em.EMBODIMENT_DIR / "monte02_left.json"
needs_monte02_urdf = pytest.mark.skipif(
    not (REPO / "third_party/opencv_kalibr/monte02_h_v1_description/urdf").is_dir(),
    reason="opencv_kalibr submodule not checked out",
)


def _monte02() -> dict:
    return json.loads((REPO / MONTE02).read_text(encoding="utf-8"))


@needs_monte02_urdf
def test_monte02_loads_and_is_not_in_service_yet() -> None:
    description = em.load(MONTE02, repo_root=REPO)
    assert description["dataset_frame"] == em.DATASET_FRAME_RECORDED
    (root,) = description["roots"].values()
    assert (root["kind"], root["registration"]) == ("mobile", "session_edge")
    modes = em.replay_modes(description, {"edges": []}, "world_20260928_063531", ["left"])
    assert "not in service" in modes["absolute"]["reason"] and "not in service" in modes["relative"]["reason"]
    assert "no arm replays the right hand" in em.replay_modes(description, {"edges": []}, "w", ["right"])["relative"]["reason"]


@needs_monte02_urdf
def test_monte02_in_service_replays_relative_but_absolute_waits_for_the_docking_registration() -> None:
    description = _monte02()
    description["arms"]["left"]["replay_config"] = "tools/thor/gmsl2/embodiments/monte02_left.json"  # any file
    graph = {"edges": [_edge("world_20260928_063531", "monte02_base", _rz(0))]}  # even a stray edge is not read
    modes = em.replay_modes(description, graph, "world_20260928_063531", ["left"])
    assert modes["relative"] == {"ok": True, "edges": []}
    assert not modes["absolute"]["ok"] and "per session_edge" in modes["absolute"]["reason"]


def test_a_dataset_is_never_in_a_mobile_root() -> None:
    description = _monte02()
    description["dataset_frame"] = "monte02_base"
    assert any("mobile root, a different place after every docking" in p for p in em.validate(description, repo_root=REPO))


@needs_monte02_urdf
def test_every_moving_joint_up_to_the_tcp_is_free_or_session_locked() -> None:
    description = _monte02()
    arm = description["arms"]["left"]
    arm["session_locked_joints"] = [j for j in arm["session_locked_joints"] if j != "joint_t2_hip_pitch"]
    arm["free_joints"] = arm["free_joints"][::-1]
    problems = " | ".join(em.validate(description, repo_root=REPO))
    assert "neither free nor session-locked: joint_t2_hip_pitch" in problems
    assert "free joints not in chain order" in problems

    description = _monte02()
    arm = description["arms"]["left"]
    arm["session_locked_joints"] = [*arm["session_locked_joints"], "joint_l4_elbow_pitch", "joint_l9_nothing"]
    arm["kinematics"]["tcp_link"] = "rt_corenetic_socket_ee"  # the other hand: does not hang below the left arm
    arm["collision"] = "no/such.json"
    problems = " | ".join(em.validate(description, repo_root=REPO))
    assert "session-locked joint joint_l9_nothing is not a moving joint" in problems
    assert "both free and session-locked: joint_l4_elbow_pitch" in problems
    assert "collision no/such.json does not exist" in problems

    description = _monte02()
    description["arms"]["left"]["session_locked_joints"].remove("joint_t1_knee_pitch")
    description["arms"]["left"]["kinematics"]["base_link"] = "link_t0_trunk_base"
    problems = " | ".join(em.validate(description, repo_root=REPO))
    assert "base_link link_t0_trunk_base is not the root's link link_b0_base" in problems

    description = _monte02()
    description["arms"]["left"]["kinematics"]["tcp_link"] = "link_r7_wrist_roll"
    description["arms"]["left"]["tcp_frame"] = "link_r7_wrist_roll"
    assert any("free joints not between link_b0_base and link_r7_wrist_roll: joint_l1_shoulder_pitch" in p
               for p in em.validate(description, repo_root=REPO))
