"""Correct the world frame stamped into episodes that were recorded in another one.

``world_provenance`` stamps whatever ``world_reference.json`` says at record
time.  That is honest about the file and wrong about the rig whenever a
re-solve minted a world island that was never committed to the reference: from
2026-09-23 18:55 to 2026-09-28 14:30 the reference still named
``world_20260819_031843`` while the cameras had been re-mounted and production
tracked in ``world_20260923_143048``.  Those episodes then claim to be
comparable in absolute terms with the August data, which is the one mistake the
stamp exists to prevent.

The stamp cannot be recomputed from the episode, so the correction is a
decision a human makes from outside evidence (when the rig moved, which
calibration the episode was solved with).  This tool only records that decision
carefully:

* **Nothing is lost.**  The original block is kept verbatim under
  ``restamp.original`` together with the reason, so the correction can be
  audited or undone.
* **Only the named id is replaced.**  An episode that does not carry
  ``--expect-from`` is skipped and reported, so re-running is a no-op and a
  wrong directory in the list cannot re-label an episode from a third world.
* **The target must be a node of ``world_graph.json``.**  A typo cannot invent
  a world.
* **Dry run by default.**  ``--apply`` writes, atomically, one file at a time.

Symlinked ``meta.json`` files (derived trees such as ``*_no_cam12``) are
resolved and rewritten once.  Deliberately stdlib-only, like the stamp itself.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.thor.gmsl2.world_provenance import STATUS_OK, WORLD_SUBDIR

WORLD_GRAPH_FILE = "world_graph.json"


def load_world_node(graph_path: Path, world_frame_id: str) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for node in graph.get("nodes") or []:
        if node.get("world_frame_id") == world_frame_id:
            return dict(node)
    raise KeyError(f"{world_frame_id} is not a node of {graph_path}; add it there first")


def restamped_block(
    original: Mapping[str, Any], node: Mapping[str, Any], reason: str, now_utc: str
) -> dict[str, Any]:
    """The block an episode carries after the correction.

    There is no reference file for an island that was never committed, so the
    reference hash and camera list are deliberately absent rather than copied
    from the file that was wrong.
    """
    return {
        "world_frame_id": str(node["world_frame_id"]),
        "status": STATUS_OK,
        "created_utc": str(node.get("created_utc") or ""),
        "calibration_id": str(node.get("calibration_id") or ""),
        "parent_world_frame_id": node.get("parent_world_frame_id"),
        "reference_path": str(WORLD_SUBDIR / WORLD_GRAPH_FILE),
        "restamp": {
            "restamped_utc": now_utc,
            "reason": reason,
            "from_world_frame_id": str(original.get("world_frame_id") or ""),
            "original": dict(original),
        },
    }


def _targets(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "episodes").glob("episode_*/meta.json"))
    info = dataset_root / "meta" / "info.json"
    if info.is_file():
        paths.append(info)
    return paths


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".restamp_tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def restamp(
    dataset_roots: list[Path],
    *,
    expect_from: str,
    node: Mapping[str, Any],
    reason: str,
    apply: bool,
    now_utc: str | None = None,
) -> tuple[list[str], list[str]]:
    """Returns ``(changed, skipped)`` as printable lines."""
    now_utc = now_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed: list[str] = []
    skipped: list[str] = []
    seen: set[Path] = set()
    for root in dataset_roots:
        targets = _targets(root)
        if not targets:
            skipped.append(f"{root}: no episodes/episode_*/meta.json")
        for path in targets:
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            payload = json.loads(real.read_text(encoding="utf-8"))
            block = payload.get("world_frame")
            current = str(block.get("world_frame_id") or "") if isinstance(block, Mapping) else ""
            if current != expect_from:
                skipped.append(f"{real}: carries {current or '<unstamped>'}, not {expect_from}")
                continue
            payload["world_frame"] = restamped_block(block, node, reason, now_utc)
            if apply:
                _write_atomic(real, payload)
            changed.append(f"{real}: {expect_from} -> {node['world_frame_id']}")
    return changed, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_roots", nargs="+", type=Path, help="directories holding episodes/")
    parser.add_argument("--expect-from", required=True, help="the id the episodes carry now")
    parser.add_argument("--to", required=True, help="the world they were recorded in")
    parser.add_argument("--reason", required=True, help="the evidence, stored in every episode")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = parser.parse_args(argv)

    node = load_world_node(args.repo_root / WORLD_SUBDIR / WORLD_GRAPH_FILE, args.to)
    changed, skipped = restamp(
        args.dataset_roots, expect_from=args.expect_from, node=node, reason=args.reason, apply=args.apply
    )
    for line in changed:
        print(("restamped " if args.apply else "would restamp ") + line)
    for line in skipped:
        print("skipped " + line)
    print(
        f"{len(changed)} file(s) {'rewritten' if args.apply else 'to rewrite (dry run)'}, {len(skipped)} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
