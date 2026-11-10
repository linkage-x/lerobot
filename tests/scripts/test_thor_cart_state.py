"""cart_state: a cart seen to have moved since its edge was solved takes that edge out of every route.

Stdlib only (no cv2): the verdicts are written by cart_redock_check.py ``cart``; here they are written by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.thor.gmsl2 import cart_state, world_graph_guard as guard, world_provenance as wp
from tools.thor.gmsl2.register_fr3_base import write_graph

WORLD = "world_20260928_063531"


def _edge(created: str = "2026-11-10T02:49:19Z", x: float = 1.2) -> dict:
    return {
        "from_world_frame_id": WORLD,
        "to_world_frame_id": "fr3_base_right",
        "T_to_from": [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
        "created_utc": created,
    }


def _repo(tmp_path: Path, *edges: dict) -> Path:
    world = tmp_path / "tools" / "thor" / "gmsl2" / "world"
    world.mkdir(parents=True)
    (world / "world_graph.json").write_text(json.dumps({"version": 1, "nodes": [], "edges": list(edges)}))
    return tmp_path


def _entry(verdict: str, edge: dict, start: str = "2026-11-10T09:00:00+00:00", **kw) -> dict:
    return {
        "verdict": verdict,
        "episode": "/data/ds/episodes/episode_000000",
        "episode_start_utc": start,
        "edge": guard.edge_key(edge),
        "within_reach_bound_mm": 8.5,
        "limit_mm": 3.0,
        **kw,
    }


def test_no_state_blocks_nothing(tmp_path) -> None:
    repo = _repo(tmp_path, _edge())
    graph = wp.read_world_graph(repo)
    assert graph[cart_state.GRAPH_KEY] == {"bases": {}}
    assert len(wp.live_edges(graph)) == 1


def test_a_moved_cart_takes_its_edge_out_of_routes_and_says_why(tmp_path) -> None:
    edge = _edge()
    repo = _repo(tmp_path, edge)
    assert cart_state.update(repo, "fr3_base_right", _entry("moved", edge))
    graph = wp.read_world_graph(repo)
    assert wp.live_edges(graph) == []
    note = wp.retired_note(graph, "fr3_base_right")
    assert "小车被推动过" in note and "8.5 mm" in note
    assert wp.retired_note(graph, "fr3_base") == ""


def test_only_moved_blocks(tmp_path) -> None:
    edge = _edge()
    for verdict in ("ok", "partial", "unseen", "inconsistent", "other_calibration", "reference_stale"):
        repo = _repo(tmp_path / verdict, edge)
        cart_state.update(repo, "fr3_base_right", _entry(verdict, edge))
        assert len(wp.live_edges(wp.read_world_graph(repo))) == 1, verdict


def test_a_verdict_on_an_older_edge_does_not_hold_back_a_re_solved_one(tmp_path) -> None:
    old, new = _edge(), _edge(created="2026-11-12T08:00:00Z", x=1.25)
    repo = _repo(tmp_path, dict(old, retired={"utc": "2026-11-12", "reason": "cart moved"}), new)
    cart_state.update(repo, "fr3_base_right", _entry("moved", old))
    assert wp.live_edges(wp.read_world_graph(repo)) == [new]


def test_a_later_episode_lifts_it_and_an_earlier_one_cannot_overwrite(tmp_path) -> None:
    edge = _edge()
    repo = _repo(tmp_path, edge)
    cart_state.update(repo, "fr3_base_right", _entry("moved", edge, start="2026-11-10T09:00:00+00:00"))
    # an old episode checked again afterwards says nothing about now
    assert not cart_state.update(repo, "fr3_base_right", _entry("ok", edge, start="2026-11-10T08:00:00+00:00"))
    assert wp.live_edges(wp.read_world_graph(repo)) == []
    # the cart pushed back, a new clip recorded
    assert cart_state.update(repo, "fr3_base_right", _entry("ok", edge, start="2026-11-10T10:00:00+00:00"))
    assert len(wp.live_edges(wp.read_world_graph(repo))) == 1


def test_an_unreadable_state_file_blocks_nothing(tmp_path) -> None:
    repo = _repo(tmp_path, _edge())
    (repo / cart_state.STATE_FILE).parent.mkdir(parents=True)
    (repo / cart_state.STATE_FILE).write_text("{not json")
    assert len(wp.live_edges(wp.read_world_graph(repo))) == 1


def test_writing_the_graph_back_leaves_the_cart_state_out(tmp_path) -> None:
    edge = _edge()
    repo = _repo(tmp_path, edge)
    cart_state.update(repo, "fr3_base_right", _entry("moved", edge))
    graph = wp.read_world_graph(repo)
    path = repo / "tools" / "thor" / "gmsl2" / "world" / "world_graph.json"
    write_graph(path, graph)
    written = json.loads(path.read_text())
    assert cart_state.GRAPH_KEY not in written and guard.GRAPH_KEY not in written
