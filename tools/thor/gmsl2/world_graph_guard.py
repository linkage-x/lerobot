"""Notice when ``world_graph.json`` is not the graph this machine last trusted.

The graph is tracked in git and arrives on Thor with every deploy, which mirrors *the deployer's* tree.
Two things make that dangerous for the edges into a robot base:

* the 10-15 world -> fr3_base edge was retired on 2026-11-03 (both bases moved). A deploy from a tree
  without that commit brings it back as live, and the GUI's export re-expresses every dataset in a base
  the arm has left -- without an error;
* an edge applied on the calibration page lives only in Thor's copy until someone copies it back and
  commits; the next deploy from any tree drops it the same way.

So the gateway keeps a record of the graph it trusts in ``outputs/world_graph_guard.json`` -- under
``outputs/``, which no deploy touches -- and every reader compares against it (``world_provenance.
read_world_graph``). On a difference in the edges, nothing routes into a robot base until a person looks
and confirms on the calibration page; edges between camera worlds are untouched. The record is written
when the gateway first starts with this guard, when the calibration page applies or confirms a graph,
and by ``python -m tools.thor.gmsl2.world_graph_guard accept``.

What it cannot do: a deploy of a tree from before this guard replaces the code that checks too. The
record survives that deploy, so the next deploy of a guarded tree checks the graph it brings against it.

Stdlib only: the exporter and the gateway run in Thor's system python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GUARD_FILE = Path("outputs") / "world_graph_guard.json"
GRAPH_FILE = Path("tools") / "thor" / "gmsl2" / "world" / "world_graph.json"
SCHEMA = "world_graph_guard/v1"
#: check() states. Only "diverged" and "unreadable" stop anything; "unrecorded" is a machine without a
#: record (a workstation, a test), where there is nothing to compare against.
OK, DIVERGED, UNRECORDED, UNREADABLE = "ok", "diverged", "unrecorded", "unreadable"
#: The key read_world_graph sets on the graph it returns; never written to the file.
GRAPH_KEY = "_guard"


class WorldGraphGuardError(RuntimeError):
    """The world graph on disk is not the one this machine trusts."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def edge_key(edge: Mapping[str, Any]) -> dict[str, Any]:
    """What identifies an edge for the guard: its ends, when it was solved, its matrix, and whether it is retired."""
    return {
        "from": str(edge.get("from_world_frame_id") or ""),
        "to": str(edge.get("to_world_frame_id") or ""),
        "created_utc": str(edge.get("created_utc") or ""),
        "T_sha256": hashlib.sha256(_canonical(edge.get("T_to_from")).encode()).hexdigest()[:16],
        "retired": bool(edge.get("retired")),
    }


def fingerprint(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted((edge_key(e) for e in graph.get("edges") or []), key=_canonical)


def _label(key: Mapping[str, Any]) -> str:
    return f"{key['from']} -> {key['to']}（{key['created_utc'] or '无时间'} 解出）"


def compare(recorded: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[str]:
    """What changed from the trusted edges to these, worst first; empty when the edges are the same."""

    def same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return {k: v for k, v in a.items() if k != "retired"} == {
            k: v for k, v in b.items() if k != "retired"
        }

    problems: list[str] = []
    for old in recorded:
        match = next((new for new in current if same(old, new)), None)
        if match is None:
            problems.append(f"受信版本里的边 {_label(old)} 不见了或被改了")
        elif old["retired"] and not match["retired"]:
            problems.insert(
                0,
                f"已停用的边 {_label(old)} 又成了现役：world_graph.json 很可能被一次用旧代码的部署覆盖了",
            )
        elif not old["retired"] and match["retired"]:
            problems.append(f"边 {_label(old)} 被停用了，但不是经网关做的")
    for new in current:
        if not any(same(old, new) for old in recorded):
            problems.append(f"多了一条受信版本里没有的边 {_label(new)}")
    return problems


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def record(repo_root: Path | str, *, reason: str, graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Trust the graph on disk (or ``graph``) from now on; returns the record written."""
    root = Path(repo_root)
    if graph is None:
        graph, error = _read_json(root / GRAPH_FILE)
        if graph is None:
            raise WorldGraphGuardError(f"{GRAPH_FILE} is {error}; nothing to trust")
    entry = {"schema": SCHEMA, "recorded_utc": _now_iso(), "reason": reason, "edges": fingerprint(graph)}
    path = root / GUARD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return entry


def check(repo_root: Path | str, graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``{"state", "problems", "recorded_utc", "reason"}`` of the graph on disk (or ``graph``) against the record."""
    root = Path(repo_root)
    trusted, error = _read_json(root / GUARD_FILE)
    if trusted is None:
        if error == "missing":
            return {"state": UNRECORDED, "problems": [], "recorded_utc": "", "reason": ""}
        return {
            "state": UNREADABLE,
            "problems": [f"{GUARD_FILE} 读不出来（{error}）"],
            "recorded_utc": "",
            "reason": "",
        }
    if graph is None:
        graph, error = _read_json(root / GRAPH_FILE)
        if graph is None:
            graph = {"edges": []}
    problems = compare(list(trusted.get("edges") or []), fingerprint(graph))
    return {
        "state": DIVERGED if problems else OK,
        "problems": problems,
        "recorded_utc": str(trusted.get("recorded_utc") or ""),
        "reason": str(trusted.get("reason") or ""),
    }


def blocks(status: Mapping[str, Any] | None) -> bool:
    """Whether a check result stops routing into a robot base."""
    return bool(status) and status.get("state") in (DIVERGED, UNREADABLE)


def note(status: Mapping[str, Any]) -> str:
    """The message every refused export and replay prints."""
    head = (
        f"world_graph.json 与这台机器上次确认的版本（{status.get('recorded_utc') or '—'}，{status.get('reason') or '—'}）不一致，"
        "确认前不经过任何到机器人基座的边："
    )
    tail = (
        "核对后在标定页「FR3 基座注册边」确认当前版本，或 python -m tools.thor.gmsl2.world_graph_guard accept"
    )
    return "\n".join([head, *(f"  - {p}" for p in status.get("problems") or []), tail])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("check", "accept"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--reason", default="accepted from the command line")
    args = parser.parse_args(argv)
    if args.command == "accept":
        entry = record(args.repo_root, reason=args.reason)
        print(f"trusted {len(entry['edges'])} edges ({entry['recorded_utc']}): {args.repo_root / GUARD_FILE}")
        return 0
    status = check(args.repo_root)
    print(f"{status['state']}" + (f"\n{note(status)}" if blocks(status) else ""))
    return 1 if blocks(status) else 0


if __name__ == "__main__":
    sys.exit(main())
