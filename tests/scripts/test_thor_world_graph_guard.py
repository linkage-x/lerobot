"""world_graph_guard: an old world_graph.json brought back by a deploy cannot put anything in a robot base."""

from __future__ import annotations

import json
from pathlib import Path

from tools.thor.gmsl2 import register_fr3_base as reg, world_graph_guard as guard, world_provenance as wp

_T = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
_RETIRED = {"utc": "2026-11-03T08:13:42Z", "reason": "both FR3 bases moved"}


def _edge(a, b, created, **extra):
    return {
        "from_world_frame_id": a,
        "to_world_frame_id": b,
        "T_to_from": _T,
        "method": "test",
        "created_utc": created,
        **extra,
    }


def _repo(tmp_path: Path, edges: list[dict]) -> Path:
    path = tmp_path / guard.GRAPH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [{"world_frame_id": w} for w in ("world_old", "world_a", "fr3_base")]
    path.write_text(json.dumps({"version": 1, "nodes": nodes, "edges": edges}))
    return tmp_path


TRUSTED = [
    _edge("world_old", "world_a", "2026-09-28T00:00:00Z"),  # a camera-world edge: never the guard's business
    _edge("world_a", "fr3_base", "2026-10-15T03:08:47Z", retired=_RETIRED),
]


def test_an_old_graph_that_revives_a_retired_edge_stops_every_route_into_a_base(tmp_path):
    repo = _repo(tmp_path, TRUSTED)
    assert (
        guard.check(repo)["state"] == guard.UNRECORDED
    )  # nothing recorded: nothing to compare, nothing stopped
    guard.record(repo, reason="test")
    assert guard.check(repo)["state"] == guard.OK

    # a deploy from a tree without the retirement: the same edge, live again
    _repo(tmp_path, [TRUSTED[0], {k: v for k, v in TRUSTED[1].items() if k != "retired"}])
    status = guard.check(repo)
    assert status["state"] == guard.DIVERGED
    assert "又成了现役" in status["problems"][0] and "旧代码的部署" in status["problems"][0]

    graph = wp.read_world_graph(repo)
    assert graph[guard.GRAPH_KEY]["state"] == guard.DIVERGED
    assert wp.world_transform(graph, "world_a", "fr3_base") is None  # the revived edge routes nothing
    assert wp.world_transform(graph, "world_old", "world_a") is not None  # camera worlds stay connected
    note = wp.retired_note(graph, "world_a", "fr3_base")
    assert "不一致" in note and "又成了现役" in note and "world_graph_guard accept" in note
    assert wp.retired_note(graph, "world_old") == ""  # nothing about bases asked

    # the verdict is never written into the file
    reg.write_graph(repo / guard.GRAPH_FILE, graph)
    assert guard.GRAPH_KEY not in json.loads((repo / guard.GRAPH_FILE).read_text())

    # accepting trusts it as it is (here: the live edge, on purpose)
    guard.record(repo, reason="accepted")
    graph = wp.read_world_graph(repo)
    assert wp.world_transform(graph, "world_a", "fr3_base") is not None


def test_a_lost_an_edited_and_an_unrecorded_edge_are_each_named(tmp_path):
    new_edge = _edge("world_a", "fr3_base", "2026-11-20T00:00:00Z")
    repo = _repo(tmp_path, [*TRUSTED, new_edge])
    guard.record(repo, reason="applied on the calibration page")

    # a deploy from a tree that never had the page's edge: it is gone
    _repo(tmp_path, TRUSTED)
    assert any("不见了" in p for p in guard.check(repo)["problems"])
    # the same edge with another matrix is not the same edge
    edited = {**new_edge, "T_to_from": [[1.0, 0.0, 0.0, 0.6], *_T[1:]]}
    _repo(tmp_path, [*TRUSTED, edited])
    problems = guard.check(repo)["problems"]
    assert any("不见了或被改了" in p for p in problems) and any("没有的边" in p for p in problems)
    # retired by hand, not through the gateway or the tool
    _repo(tmp_path, [*TRUSTED, {**new_edge, "retired": _RETIRED}])
    assert any("不是经网关做的" in p for p in guard.check(repo)["problems"])


def test_an_unreadable_record_stops_routing_and_the_cli_reports_it(tmp_path, capsys):
    repo = _repo(tmp_path, TRUSTED)
    (repo / guard.GUARD_FILE).parent.mkdir(parents=True, exist_ok=True)
    (repo / guard.GUARD_FILE).write_text("{not json")
    assert guard.check(repo)["state"] == guard.UNREADABLE and guard.blocks(guard.check(repo))
    assert guard.main(["check", "--repo-root", str(repo)]) == 1
    assert guard.main(["accept", "--repo-root", str(repo)]) == 0
    assert guard.main(["check", "--repo-root", str(repo)]) == 0
    assert "trusted 2 edges" in capsys.readouterr().out
