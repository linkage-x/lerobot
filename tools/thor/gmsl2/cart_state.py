"""Is each FR3 cart still where its world edge was solved?  The latest verdict per base, kept outside the code tree.

A world -> robot-base edge (``world_graph.json``) is a fixed transform: it is right only while the arm stands where
it stood when the edge was solved.  The arms stand on carts, and nothing else notices a pushed cart -- the world
graph guard compares files, not positions.  So:

* ``world/cart_reference.json`` (tracked) keeps, per cart, the tags on it as the cameras saw them right after the
  edge was solved, and which edge that was (``world_graph_guard.edge_key``);
* ``cart_redock_check.py cart <episode>`` measures the same tags in a later recording and records the verdict here,
  in ``outputs/cart_check.json`` -- under ``outputs/``, which no deploy touches.  The gateway runs it on every new
  episode;
* ``world_provenance.read_world_graph`` attaches this file, and ``live_edges`` drops an edge into a base whose cart
  ``moved`` since that edge: nothing is exported, checked or replayed in a base the arm has left.  The GUI shows it
  in red.

It lifts by itself: a later episode that finds the cart back in place replaces the verdict, and a new edge for the
base (re-registration) is not the edge the verdict was measured against.  Only ``moved`` blocks.  The others say
how much the latest recording could tell: ``partial`` (two of the tags: the shift is bounded, the rotation is not),
``unseen`` (fewer), ``inconsistent`` (the tags did not move together: one was bumped, peeled or misread),
``other_calibration`` (the cameras were recalibrated since the reference: take a new one), ``reference_stale``
(the reference belongs to another edge).

What it cannot see: a cart pushed after the latest recording.  The operator's rule stays: do not push the carts;
before replaying after a push, record a short clip with the cart's tags in view.

Stdlib only: read by the exporter and the gateway in Thor's system python.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_FILE = Path("outputs") / "cart_check.json"
REFERENCE_FILE = Path("tools") / "thor" / "gmsl2" / "world" / "cart_reference.json"
SCHEMA = "cart_check_state/v1"
#: The key read_world_graph sets on the graph it returns; never written to the file.
GRAPH_KEY = "_cart"
OK, MOVED, PARTIAL, UNSEEN, INCONSISTENT = "ok", "moved", "partial", "unseen", "inconsistent"
OTHER_CALIBRATION, REFERENCE_STALE = "other_calibration", "reference_stale"
VERDICTS = (OK, MOVED, PARTIAL, UNSEEN, INCONSISTENT, OTHER_CALIBRATION, REFERENCE_STALE)
#: Only a cart seen to have moved stops routing into its base; the rest are reported.
BLOCKING = frozenset({MOVED})


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read(repo_root: Path | str) -> dict[str, Any]:
    """``{"bases": {base: entry}}``; empty when there is no file or it cannot be read (nothing blocks then)."""
    try:
        payload = json.loads((Path(repo_root) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"bases": {}}
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or not isinstance(payload.get("bases"), dict):
        return {"bases": {}}
    return payload


def update(repo_root: Path | str, base: str, entry: Mapping[str, Any]) -> bool:
    """Record ``entry`` as the latest verdict for ``base`` unless a recording made later is already recorded.

    Checking an old episode again must not overwrite what a newer one said; a verdict against another edge is
    always replaced (the newer edge is what matters).  Returns whether it was written.
    """
    path = Path(repo_root) / STATE_FILE
    state = read(repo_root)
    have = state["bases"].get(base)
    if (
        isinstance(have, Mapping)
        and _canonical(have.get("edge")) == _canonical(entry.get("edge"))
        and str(have.get("episode_start_utc") or "") > str(entry.get("episode_start_utc") or "")
    ):
        return False
    state.update(schema=SCHEMA, updated_utc=_now_iso())
    state["bases"][base] = dict(entry, recorded_utc=_now_iso())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return True


def blocked_edges(graph: Mapping[str, Any], edge_key: Any) -> dict[str, Mapping[str, Any]]:
    """``{canonical edge key: entry}`` for each edge of ``graph`` whose cart was last seen to have moved.

    ``edge_key`` is ``world_graph_guard.edge_key`` (passed in: this module stays import-free).
    """
    state = graph.get(GRAPH_KEY)
    if not isinstance(state, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for base, entry in (state.get("bases") or {}).items():
        if not isinstance(entry, Mapping) or entry.get("verdict") not in BLOCKING:
            continue
        key = _canonical(entry.get("edge"))
        for edge in graph.get("edges") or []:
            ends = {str(edge.get("from_world_frame_id") or ""), str(edge.get("to_world_frame_id") or "")}
            if base in ends and _canonical(edge_key(edge)) == key:
                out[key] = dict(entry, base=base)
    return out


def edge_is_blocked(graph: Mapping[str, Any], edge: Mapping[str, Any], edge_key: Any) -> bool:
    return _canonical(edge_key(edge)) in blocked_edges(graph, edge_key)


def note(entry: Mapping[str, Any]) -> str:
    """The message a refused export or replay prints, and the GUI's red line."""
    base = entry.get("base") or "?"
    return (
        f"{base} 的小车被推动过：{entry.get('episode') or '?'}（{entry.get('episode_start_utc') or '?'} 录）里车上 tag "
        f"相对参考移动了 {entry.get('within_reach_bound_mm', '?')} mm（工作空间内最大，限 {entry.get('limit_mm', '?')} mm），"
        f"这条到 {base} 的边已停用。把车推回原位后录一段能看到车上 tag 的短视频，检查通过会自动恢复；"
        "推不回去就重新做同球标定。"
    )
