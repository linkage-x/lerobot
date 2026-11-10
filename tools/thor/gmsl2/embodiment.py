"""What a robot is, as far as labels and replay are concerned: its roots, its arms, and how each root gets into the world.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, G1. Until now "the robot" was one FR3 implied by one
replay YAML and one world-graph node, ``fr3_base``. Two FR3s, and later a wheeled humanoid, differ from
that in exactly the things this file names:

* **roots** -- frames the robot commands in. One FR3 has one; two FR3s have two, joined by an edge that is
  measured (B3, arm-arm ball); a humanoid has one, and its two arms are joined by its own kinematics.
* **how a root is registered into the camera world** -- ``static_edge``: an edge of ``world_graph.json``
  that holds until the base is moved (an FR3 bolted down); ``session_edge``: re-measured every time the
  robot stops (a wheeled base, H2); ``tracked``: a pose per frame (manipulation while driving). Only the
  first exists today; the other two are declared so a description can say what it needs and be refused
  until that exists, rather than silently borrowing a static edge.
* **arms** -- which hands of a recording each arm replays (one FR3: either, one at a time), its root, its TCP and the replay config that
  holds its controller settings (IP, gripper BOX, URDF). The config file stays the one place for those;
  this file only points at it, so the two cannot disagree.

A dataset is exported in one frame (``dataset_frame``: ``fr3_base``, the registered arm's base -- the left
arm's since 10-26), both hands together, so ``T_LR = inv(T_L) T_R`` stays a product of two columns of one
file. Getting the other hand's target into the other arm's base is the replay's job, through the graph (:func:`arm_transform`), and the
edges it used go into the replay report. That is the C0 decision: the bases are split in the robot's
description, not in the dataset.

Deliberately stdlib-only, like ``world_provenance``: the exporter and the gateway run in Thor's system python.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402

SCHEMA = "embodiment/v1"
EMBODIMENT_DIR = Path("tools") / "thor" / "gmsl2" / "embodiments"
#: How a root gets into the camera world (see the module docstring).
REGISTRATIONS = ("static_edge", "session_edge", "tracked")
ROOT_KINDS = ("fixed", "mobile")
HANDS = ("left", "right")


class EmbodimentError(ValueError):
    """A description that does not hold together."""


def validate(description: Mapping[str, Any], *, repo_root: Path | str = _REPO_ROOT) -> list[str]:
    """Everything wrong with ``description``; empty when it can be used."""
    problems: list[str] = []
    if description.get("schema") != SCHEMA:
        problems.append(f"schema is {description.get('schema')!r}, expected {SCHEMA!r}")
    roots = description.get("roots")
    if not isinstance(roots, Mapping) or not roots:
        return [*problems, "no roots"]
    for name, root in roots.items():
        if not isinstance(root, Mapping):
            problems.append(f"root {name}: not an object")
            continue
        if root.get("kind") not in ROOT_KINDS:
            problems.append(f"root {name}: kind {root.get('kind')!r} not one of {ROOT_KINDS}")
        if root.get("registration") not in REGISTRATIONS:
            problems.append(
                f"root {name}: registration {root.get('registration')!r} not one of {REGISTRATIONS}"
            )
        if root.get("kind") == "mobile" and root.get("registration") == "static_edge":
            problems.append(
                f"root {name}: a mobile root cannot hold a static edge; it moves between sessions"
            )
    if str(description.get("dataset_frame") or "") not in roots:
        problems.append(f"dataset_frame {description.get('dataset_frame')!r} is not one of the roots")
    arms = description.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        return [*problems, "no arms"]
    hands_seen: dict[str, str] = {}
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            problems.append(f"arm {name}: not an object")
            continue
        if str(arm.get("root") or "") not in roots:
            problems.append(f"arm {name}: root {arm.get('root')!r} is not one of the roots")
        hands = arm.get("hands")
        if not isinstance(hands, list) or not hands:
            problems.append(f"arm {name}: no hands")
            hands = []
        for hand in hands:
            if hand not in HANDS:
                problems.append(f"arm {name}: hand {hand!r} not one of {HANDS}")
            elif hand in hands_seen:
                problems.append(f"arm {name}: hand {hand} is already replayed by arm {hands_seen[hand]}")
            else:
                hands_seen[hand] = name
        if not str(arm.get("tcp_frame") or ""):
            problems.append(f"arm {name}: no tcp_frame")
        config = str(arm.get("replay_config") or "")
        if config and not (Path(repo_root) / config).is_file():
            problems.append(f"arm {name}: replay_config {config} does not exist")
    return problems


def load(path: Path | str, *, repo_root: Path | str = _REPO_ROOT) -> dict[str, Any]:
    """A description from ``path`` (absolute, or relative to the repo), refused unless :func:`validate` passes."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    description = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(description, repo_root=repo_root)
    if problems:
        raise EmbodimentError(f"{path}: " + "; ".join(problems))
    return description


def arm_for_hand(description: Mapping[str, Any], hand: str) -> str | None:
    """The arm that replays a recording's ``hand``, or None when no arm does."""
    return next((name for name, arm in description["arms"].items() if hand in (arm.get("hands") or [])), None)


def arm_transform(
    description: Mapping[str, Any], graph: Mapping[str, Any], dataset_world: str, arm: str
) -> tuple[list[list[float]] | None, list[dict[str, Any]], str]:
    """``(T_root_dataset, edges_used, reason)`` for one arm: how its targets get from the dataset into its base.

    ``T`` is None with a reason when it cannot be had: the root registers per session or per frame (nothing
    to read from the graph), or the graph does not connect the two. A loop in the graph that does not close
    raises :class:`world_provenance.WorldGraphClosureError` -- not a reason to fall back to anything.
    """
    root_id = str(description["arms"][arm]["root"])
    root = description["roots"][root_id]
    if root_id == dataset_world:
        return wp.world_transform(graph, dataset_world, root_id)[0], [], "the dataset is in this arm's base"
    if root.get("registration") != "static_edge":
        return None, [], f"{root_id} registers per {root.get('registration')}; there is no edge to read"
    found = wp.world_transform(graph, dataset_world, root_id)
    if found is None:
        return (
            None,
            [],
            f"no edge of {wp.WORLD_SUBDIR / wp.WORLD_GRAPH_FILE} connects {dataset_world} to {root_id}",
        )
    return found[0], found[1], ""


def _hop_key(edge: Mapping[str, Any]) -> tuple[str, str, str, str, bool]:
    return (
        str(edge.get("from_world_frame_id") or ""),
        str(edge.get("to_world_frame_id") or ""),
        str(edge.get("method") or ""),
        str(edge.get("created_utc") or ""),
        bool(edge.get("traversed_reversed")),
    )


def export_edges_current(world_frame: Mapping[str, Any] | None, graph: Mapping[str, Any]) -> str:
    """``""`` when the edges a re-expressed dataset was exported through are still the graph's route from
    each source world to its frame; otherwise why not.

    A dataset re-expressed into ``fr3_base`` holds poses placed by the edge of its export. When the base has
    been re-registered since (moved, or a better edge), the same labels now belong elsewhere in the base,
    and an absolute replay of the old numbers would put the arm where the object no longer is.
    """
    world_frame = world_frame or {}
    dataset_world = wp.world_frame_id_of(world_frame)
    for source, edges in sorted((world_frame.get("edges_by_source_world") or {}).items()):
        found = wp.world_transform(graph, source, dataset_world)
        if found is None:
            return f"the graph no longer connects {source} to {dataset_world}, which this dataset was exported through"
        if [_hop_key(e) for e in found[1]] != [_hop_key(e) for e in edges]:
            then = ", ".join(f"{e.get('method', '')[:40]} ({e.get('created_utc', '')})" for e in edges)
            now = ", ".join(f"{e.get('method', '')[:40]} ({e.get('created_utc', '')})" for e in found[1])
            return (
                f"exported from {source} through [{then}], but the graph now goes through [{now}]; "
                "re-export the dataset before an absolute replay"
            )
    return ""


def replay_modes(
    description: Mapping[str, Any],
    graph: Mapping[str, Any],
    dataset_world: str,
    hands: list[str],
    *,
    world_frame: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Which replay modes a dataset in ``dataset_world`` allows for ``hands`` on this robot (roadmap C2).

    ``absolute``: every arm's targets reach its base through recorded edges, and (given the dataset's
    ``world_frame`` block) the edges it was exported through are still the graph's. ``relative``: the hands
    share one start alignment, so what has to be known is the arms' bases relative to each other -- the
    root-to-root edge, or one root. Each mode comes with the edges it rests on or why it is refused.
    """
    arms = {hand: arm_for_hand(description, hand) for hand in hands}
    missing = [hand for hand, arm in arms.items() if arm is None]
    reason = f"no arm replays the {', '.join(missing)} hand" if missing else ""
    if not reason and len(set(arms.values())) < len(arms):
        reason = "one arm cannot replay two hands at once"
    idle = sorted({arm for arm in arms.values() if arm and not description["arms"][arm].get("replay_config")})
    if not reason and idle:
        reason = f"arm {', '.join(idle)} is not in service (no replay_config)"
    if reason:
        return {
            "arms": arms,
            "absolute": {"ok": False, "reason": reason},
            "relative": {"ok": False, "reason": reason},
        }

    absolute_edges: dict[str, list[dict[str, Any]]] = {}
    absolute_reason = export_edges_current(world_frame, graph)
    for arm in arms.values():
        T, edges, reason = arm_transform(description, graph, dataset_world, arm)
        if T is None:
            absolute_reason = absolute_reason or f"{arm}: {reason}"
        absolute_edges[str(arm)] = edges

    roots = sorted({str(description["arms"][arm]["root"]) for arm in arms.values()})
    relative: dict[str, Any] = {"ok": True, "edges": []}
    if len(roots) > 1:
        reference = str(description["arms"][arms[hands[0]]]["root"])
        for other in roots:
            if other == reference:
                continue
            found = wp.world_transform(graph, reference, other)
            if found is None:
                relative = {"ok": False, "reason": f"no edge connects {reference} to {other} (B3)"}
                break
            relative["edges"].extend(found[1])
    absolute = (
        {"ok": False, "reason": absolute_reason} if absolute_reason else {"ok": True, "edges": absolute_edges}
    )
    return {"arms": arms, "absolute": absolute, "relative": relative}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check an embodiment description and what it can replay")
    parser.add_argument("description", type=Path)
    parser.add_argument("--dataset-world", default=wp.FR3_BASE_WORLD_ID)
    parser.add_argument("--hands", nargs="+", default=["right"], choices=HANDS)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        description = load(args.description, repo_root=args.repo_root)
    except EmbodimentError as exc:
        print(f"FAIL: {exc}")
        return 1
    modes = replay_modes(description, wp.read_world_graph(args.repo_root), args.dataset_world, args.hands)
    print(json.dumps(modes, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
