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
