"""Stamp every recorded episode with the world frame it was measured in.

Roadmap Phase 2.4 froze ``tools/thor/gmsl2/world/world_reference.json`` so that
``world_frame_id`` means something: two recordings are comparable in *absolute*
terms exactly when they carry the same one.  Until now nothing carried it.  The
id appeared in the calibration, registration and export paths and nowhere on the
recording side, so an episode recorded today said which cameras were up, which
sensors streamed and at what timestamps -- but not which world its poses would
eventually be expressed in.

That gap is the one kind of provenance that cannot be repaired after the fact.
A missing sync report can be recomputed from the sidecars; a missing world id
cannot be recovered from the episode at all, because the thing it records is
*which frozen file was on disk at the moment the shutter opened*, and the next
re-solve, re-freeze or deploy overwrites the evidence.  A dataset whose episodes
straddle a re-freeze looks perfectly healthy and is silently two coordinate
systems.

Three decisions worth keeping:

* **The recorder reads the frozen file itself; the gateway does not tell it.**
  The gateway holds a calibration name in memory, and that in-memory name is
  precisely the thing that was caught lying on 2026-08-27: the GUI displayed a
  freshly solved calibration while production kept loading the old yaml, for
  seven days.  The tracker will later solve against whatever
  ``world_reference.json`` says, so the honest stamp is that same file, read
  from disk, at record time.

* **Absence is stamped, not defaulted.**  A missing or unreadable reference
  yields ``status`` != ``"ok"`` and an empty ``world_frame_id``, never a
  plausible-looking id.  Recording is not blocked -- an operator who is mid
  session should not lose the take because a json is missing -- but the episode
  says so about itself, and :func:`assert_single_world` will not let an unstamped
  episode be mixed with a stamped one downstream.

* **The sha256 is an audit trail, not the contract.**  ``world_frame_id`` is the
  contract.  The hash changes legitimately whenever a *moved* camera is
  re-registered into the same world (that is the mechanism working as designed),
  so a hash difference between two episodes with equal ids is not a fault; it is
  how you find out which of them predates the re-registration.  A hash difference
  with *unequal* ids is the re-freeze this whole mechanism exists to prevent.

Deliberately stdlib-only.  This runs inside the recorder on Thor, in the same
interpreter that must not import numpy or cv2 to write a json field.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Where the frozen reference lives, relative to the repo root.  It is tracked in
#: git and arrives on Thor by ``rsync`` with the rest of the tree -- never by
#: running ``freeze`` there, which would mint a second id for one physical frame.
WORLD_SUBDIR = Path("tools") / "thor" / "gmsl2" / "world"
WORLD_REFERENCE_FILE = "world_reference.json"
WORLD_REGISTRATION_FILE = "world_registration.json"
WORLD_GRAPH_FILE = "world_graph.json"

#: The arm's base frame as a node of ``world_graph.json``.  It is the frame the
#: FR3 commands in, so a dataset stamped with it replays on the arm without any
#: transform being estimated at deploy time; tools/fr3/fr3_act_infer_real_runtime.py
#: keys on this id.
FR3_BASE_WORLD_ID = "fr3_base"
#: ``export_v3 --target-world`` value meaning: FR3_BASE_WORLD_ID when a recorded
#: edge reaches it from the episodes' world, else the world they were recorded in.
TARGET_WORLD_AUTO = "auto"

#: Value of ``world_frame["status"]``.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_UNREADABLE = "unreadable"
STATUS_INCOMPLETE = "incomplete"

_MISSING_NOTE = (
    "no frozen world reference on disk at record time; this episode's poses "
    "cannot be declared comparable with any other episode's in absolute terms. "
    "Restore tools/thor/gmsl2/world/world_reference.json from git -- do NOT run "
    "freeze, which mints a new id for the same physical frame."
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, STATUS_MISSING
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{STATUS_UNREADABLE}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{STATUS_UNREADABLE}: top level is {type(payload).__name__}, expected object"
    return payload, STATUS_OK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _registration_summary(path: Path, reference_id: str) -> dict[str, Any] | None:
    """The last continuity verdict, compacted.

    This is *not* a verdict about the episode being recorded -- the check runs
    when someone re-solves the extrinsics, which may have been weeks ago.  It is
    kept because it is the only on-disk record of whether the rig was known to
    still be in this world the last time anybody looked, and ``generated_utc``
    is what tells a reader how stale that knowledge is.
    """
    payload, status = _read_json(path)
    if payload is None or status != STATUS_OK:
        return None
    registration_id = str(payload.get("world_frame_id") or "")
    return {
        "state": str(payload.get("world_continuity_state") or ""),
        "generated_utc": str(payload.get("generated_utc") or ""),
        "calibration_id": str(payload.get("calibration_id") or ""),
        "world_frame_id": registration_id,
        # False means the last continuity check landed on a *different* world
        # than the frozen reference names -- i.e. someone minted an island and
        # the reference was not updated, or the reference was replaced after the
        # check ran.  Either way the two files disagree and a human must look.
        "matches_reference": bool(reference_id) and registration_id == reference_id,
    }


def read_world_provenance(repo_root: Path | str) -> dict[str, Any]:
    """The ``world_frame`` block to stamp into an episode.

    Always returns a dict.  ``status == "ok"`` iff ``world_frame_id`` is a
    non-empty string read from a frozen reference that parsed; every other case
    keeps ``world_frame_id`` empty and says why in ``note``.
    """
    root = Path(repo_root)
    reference_path = root / WORLD_SUBDIR / WORLD_REFERENCE_FILE
    relative = str(WORLD_SUBDIR / WORLD_REFERENCE_FILE)

    payload, status = _read_json(reference_path)
    if payload is None:
        return {
            "world_frame_id": "",
            "status": STATUS_MISSING if status == STATUS_MISSING else STATUS_UNREADABLE,
            "reference_path": relative,
            "note": _MISSING_NOTE if status == STATUS_MISSING else f"{_MISSING_NOTE} ({status})",
        }

    world_frame_id = str(payload.get("world_frame_id") or "")
    if not world_frame_id:
        return {
            "world_frame_id": "",
            "status": STATUS_INCOMPLETE,
            "reference_path": relative,
            "reference_sha256": _sha256(reference_path),
            "note": (
                f"{relative} parsed but carries no world_frame_id; it is not a frozen "
                "reference. Restore it from git."
            ),
        }

    block: dict[str, Any] = {
        "world_frame_id": world_frame_id,
        "status": STATUS_OK,
        "created_utc": str(payload.get("created_utc") or ""),
        "calibration_id": str(payload.get("calibration_id") or ""),
        "parent_world_frame_id": payload.get("parent_world_frame_id"),
        "revision_count": len(payload.get("revisions") or []),
        "reference_path": relative,
        "reference_sha256": _sha256(reference_path),
        "reference_cameras": sorted((payload.get("cameras") or {}).keys()),
    }
    registration = _registration_summary(
        root / WORLD_SUBDIR / WORLD_REGISTRATION_FILE, world_frame_id
    )
    if registration is not None:
        block["last_registration"] = registration
    return block


def describe(block: Mapping[str, Any] | None) -> str:
    """One operator-facing line for the recorder log."""
    if not block:
        return "World frame: UNSTAMPED (no provenance block)"
    status = str(block.get("status") or "")
    if status != STATUS_OK:
        return f"WARNING: World frame {status.upper()} -- {block.get('note') or 'no frozen reference'}"
    parts = [f"World frame: {block.get('world_frame_id')}"]
    sha = str(block.get("reference_sha256") or "")
    if sha:
        parts.append(f"ref {sha[:12]}")
    registration = block.get("last_registration")
    if isinstance(registration, Mapping) and registration.get("state"):
        stale = "" if registration.get("matches_reference") else ", DISAGREES WITH REFERENCE"
        parts.append(
            f"last continuity {registration['state']} @ {registration.get('generated_utc') or '?'}{stale}"
        )
    return " | ".join(parts)


def world_frame_id_of(block: Mapping[str, Any] | None) -> str:
    """The id an episode carries, or ``""`` when it carries none."""
    if not isinstance(block, Mapping):
        return ""
    if str(block.get("status") or "") != STATUS_OK:
        return ""
    return str(block.get("world_frame_id") or "")


class MixedWorldError(RuntimeError):
    """Raised when episodes that are about to share a dataset disagree on world."""


def assert_single_world(entries: Iterable[tuple[str, Mapping[str, Any] | None]]) -> str:
    """Refuse to combine episodes measured in different worlds.

    ``entries`` is ``(label, world_frame_block)`` pairs -- the label is whatever
    identifies the episode to a human (a directory name, an index).

    Three outcomes, following the rule ``aggregate.validate_derived_provenance``
    already established for sidecar schema versions:

    * every episode unstamped -> returns ``""``.  These predate the stamp; the
      caller should say so once and carry on, because refusing would make every
      historical dataset unexportable and would not make anyone safer.
    * every episode stamped with the same id -> returns that id.
    * anything else -> :class:`MixedWorldError`.  Mixing stamped with unstamped
      is refused for the same reason a mixed v1/v2 sidecar chain is: the
      unstamped ones *might* be the same world, and "might" is not a coordinate
      system.  Absolute poses from two worlds concatenated into one dataset are
      wrong in a way no downstream check can see.

    Episode-relative motion, bimanual relative pose and contact-local
    trajectories survive a common left-multiplication and are unaffected -- what
    this protects is absolute replay and cross-session comparison.
    """
    stamped: dict[str, list[str]] = {}
    unstamped: list[str] = []
    for label, block in entries:
        world_frame_id = world_frame_id_of(block)
        if world_frame_id:
            stamped.setdefault(world_frame_id, []).append(str(label))
        else:
            unstamped.append(str(label))

    if not stamped:
        return ""
    if len(stamped) > 1 or unstamped:
        lines = [
            f"  {world_frame_id}: {', '.join(labels)}" for world_frame_id, labels in sorted(stamped.items())
        ]
        if unstamped:
            lines.append(f"  <unstamped>: {', '.join(unstamped)}")
        detail = "\n".join(lines)
        raise MixedWorldError(
            "episodes do not share one world frame, so their absolute poses cannot be "
            "concatenated:\n"
            f"{detail}\n"
            "Relative motion within an episode is unaffected. To proceed, export the "
            "episodes of one world at a time; an unstamped episode predates world "
            "provenance and cannot be proven to be in any of them."
        )
    return next(iter(stamped))


# ------------------------------------------------- cross-world registration ---
#
# ``world_graph.json`` edges are written by ``metrology.world_frame.WorldGraph``
# (``T_to_from`` takes coordinates in ``from_world_frame_id`` into
# ``to_world_frame_id``).  The lookup is repeated here, stdlib-only, because the
# exporter runs in Thor's system interpreter; the two must agree on direction,
# and ``test_thor_world_provenance`` pins this one against hand-built chains.


def read_world_graph(repo_root: Path | str) -> dict[str, Any]:
    """The tracked world graph, or an empty one when the file is absent."""
    payload, status = _read_json(Path(repo_root) / WORLD_SUBDIR / WORLD_GRAPH_FILE)
    if payload is None:
        if status == STATUS_MISSING:
            return {"version": 1, "nodes": [], "edges": []}
        raise RuntimeError(f"{WORLD_SUBDIR / WORLD_GRAPH_FILE} is {status}; restore it from git")
    return payload


def _matmul4(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(A[r][k] * B[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def _rigid_inverse(T: list[list[float]]) -> list[list[float]]:
    Rt = [[T[c][r] for c in range(3)] for r in range(3)]
    t = [-sum(Rt[r][k] * T[k][3] for k in range(3)) for r in range(3)]
    return [Rt[0] + [t[0]], Rt[1] + [t[1]], Rt[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def _checked_rigid(value: Any, label: str) -> list[list[float]]:
    try:
        T = [[float(v) for v in row] for row in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: not a 4x4 matrix") from exc
    if len(T) != 4 or any(len(row) != 4 for row in T) or not all(math.isfinite(v) for row in T for v in row):
        raise ValueError(f"{label}: not a finite 4x4 matrix")
    RtR = [[sum(T[k][r] * T[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    if any(abs(RtR[r][c] - (1.0 if r == c else 0.0)) > 1e-6 for r in range(3) for c in range(3)) or T[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"{label}: not a rigid transform")
    return T


def world_transform(
    graph: Mapping[str, Any], from_world: str, to_world: str
) -> tuple[list[list[float]], list[dict[str, Any]]] | None:
    """``(T_to_from, edges_used)``, or ``None`` when the worlds are not connected.

    Breadth-first over the edges in both directions, as
    ``metrology.world_frame.WorldGraph.transform``.  ``None`` is the honest
    answer for two islands; a caller must refuse rather than substitute an
    identity.  ``edges_used`` is each traversed edge's provenance (endpoints,
    method, created_utc, and whether it was walked backwards), for the record a
    re-expressed dataset carries about itself.
    """
    identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    if from_world == to_world:
        return identity, []
    edges = list(graph.get("edges") or [])
    frontier: list[tuple[str, list[list[float]], list[dict[str, Any]]]] = [(from_world, identity, [])]
    seen = {from_world}
    while frontier:
        world, accumulated, path = frontier.pop(0)
        for index, edge in enumerate(edges):
            src = str(edge.get("from_world_frame_id") or "")
            dst = str(edge.get("to_world_frame_id") or "")
            T = _checked_rigid(edge.get("T_to_from"), f"world_graph edge {index} ({src} -> {dst})")
            for a, b, step, reversed_ in ((src, dst, T, False), (dst, src, _rigid_inverse(T), True)):
                if a != world or b in seen:
                    continue
                combined = _matmul4(step, accumulated)
                hop = {
                    "from_world_frame_id": src,
                    "to_world_frame_id": dst,
                    "traversed_reversed": reversed_,
                    "method": str(edge.get("method") or ""),
                    "created_utc": str(edge.get("created_utc") or ""),
                }
                if b == to_world:
                    return combined, [*path, hop]
                seen.add(b)
                frontier.append((b, combined, [*path, hop]))
    return None


def _quat_from_matrix(Rm: list[list[float]]) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw) of a rotation matrix (Shepperd)."""
    trace = Rm[0][0] + Rm[1][1] + Rm[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((Rm[2][1] - Rm[1][2]) / s, (Rm[0][2] - Rm[2][0]) / s, (Rm[1][0] - Rm[0][1]) / s, 0.25 * s)
    if Rm[0][0] > Rm[1][1] and Rm[0][0] > Rm[2][2]:
        s = math.sqrt(1.0 + Rm[0][0] - Rm[1][1] - Rm[2][2]) * 2.0
        return (0.25 * s, (Rm[0][1] + Rm[1][0]) / s, (Rm[0][2] + Rm[2][0]) / s, (Rm[2][1] - Rm[1][2]) / s)
    if Rm[1][1] > Rm[2][2]:
        s = math.sqrt(1.0 + Rm[1][1] - Rm[0][0] - Rm[2][2]) * 2.0
        return ((Rm[0][1] + Rm[1][0]) / s, 0.25 * s, (Rm[1][2] + Rm[2][1]) / s, (Rm[0][2] - Rm[2][0]) / s)
    s = math.sqrt(1.0 + Rm[2][2] - Rm[0][0] - Rm[1][1]) * 2.0
    return ((Rm[0][2] + Rm[2][0]) / s, (Rm[1][2] + Rm[2][1]) / s, 0.25 * s, (Rm[1][0] - Rm[0][1]) / s)


def transform_pose7(T: list[list[float]], pose7: list[float]) -> list[float]:
    """Left-multiply an ``(x, y, z, qx, qy, qz, qw)`` pose by the rigid ``T``.

    A non-finite pose (a frame the tracker did not see) stays exactly as it was,
    so a gap remains a gap rather than becoming ``T``'s translation.  The output
    quaternion is ``q(T) * q`` with no sign canonicalisation: ``q(T)`` is one
    constant, so a column that was sign-continuous stays sign-continuous.
    """
    if len(pose7) != 7 or not all(math.isfinite(v) for v in pose7):
        return list(pose7)
    x, y, z, qx, qy, qz, qw = (float(v) for v in pose7)
    t = [T[r][0] * x + T[r][1] * y + T[r][2] * z + T[r][3] for r in range(3)]
    ax, ay, az, aw = _quat_from_matrix([row[:3] for row in T[:3]])
    out = (
        aw * qx + ax * qw + ay * qz - az * qy,
        aw * qy - ax * qz + ay * qw + az * qx,
        aw * qz + ax * qy - ay * qx + az * qw,
        aw * qw - ax * qx - ay * qy - az * qz,
    )
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [*t, *(v / norm for v in out)]


def reexpressed_world_frame(
    source_block: Mapping[str, Any],
    target_world_frame_id: str,
    T_target_source: list[list[float]],
    edges_used: list[dict[str, Any]],
) -> dict[str, Any]:
    """The ``world_frame`` block of a dataset whose poses were moved to another world.

    The source block is kept whole under ``reexpressed_from``: the recording was
    made in that world, and this dataset's numbers are a function of it *and*
    of the edges named here.  Replacing an edge later changes what the same
    recording should export to, which is why the edges are recorded and not just
    the product.
    """
    return {
        "world_frame_id": target_world_frame_id,
        "status": STATUS_OK,
        "reexpressed_from": dict(source_block),
        "T_target_source": T_target_source,
        "edges": edges_used,
    }


_POSE7_SUFFIXES = ("_x_m", "_y_m", "_z_m", "_qx", "_qy", "_qz", "_qw")


def reexpress_pose_csv(
    src: Path | str,
    dst: Path | str,
    transform_by_episode: Mapping[int, list[list[float]]],
) -> int:
    """Copy a tracking ``state_action`` CSV with every pose7 column group moved by its episode's T.

    A group is any ``<prefix>_x_m ... <prefix>_qw`` set (``state``, ``action``, ...).
    Non-finite poses stay gaps. A row whose ``episode_index`` has no transform raises:
    leaving it as recorded would mix two worlds in one file. Returns the row count.
    """
    with Path(src).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    prefixes = [
        name[: -len("_x_m")]
        for name in fieldnames
        if name.endswith("_x_m") and all(name[: -len("_x_m")] + sfx in fieldnames for sfx in _POSE7_SUFFIXES)
    ]
    for row in rows:
        episode = int(float(row.get("episode_index") or 0))
        if episode not in transform_by_episode:
            raise RuntimeError(f"{src}: episode {episode} has no transform")
        T = transform_by_episode[episode]
        for prefix in prefixes:
            keys = [prefix + sfx for sfx in _POSE7_SUFFIXES]
            try:
                pose = [float(row[k]) for k in keys]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in pose):
                row.update({k: repr(v) for k, v in zip(keys, transform_pose7(T, pose), strict=True)})
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dst)
    return len(rows)


def inspect_dataset(dataset_root: Path | str) -> tuple[int, list[str]]:
    """Report the world every episode under ``dataset_root`` was recorded in.

    This is the smoke check to run straight after the gateway restart that puts
    world provenance into service: record one episode, run this, see an id.

    Returns ``(exit_code, lines)``.  Non-zero when any episode is unstamped or
    the episodes disagree -- so it can be used as a gate in a script, not only
    read by a human.
    """
    root = Path(dataset_root)
    lines: list[str] = [f"dataset: {root}"]
    entries: list[tuple[str, Mapping[str, Any] | None]] = []

    info_path = root / "meta" / "info.json"
    info, status = _read_json(info_path)
    if info is not None and status == STATUS_OK:
        block = info.get("world_frame")
        world_frame_id = world_frame_id_of(block if isinstance(block, Mapping) else None)
        lines.append(f"  meta/info.json: {world_frame_id or '<unstamped>'}")

    episode_metas = sorted((root / "episodes").glob("episode_*/meta.json"))
    if not episode_metas:
        lines.append("  no episodes/episode_*/meta.json found")
        return 2, lines
    for meta_path in episode_metas:
        payload, status = _read_json(meta_path)
        block = payload.get("world_frame") if payload else None
        block = block if isinstance(block, Mapping) else None
        label = meta_path.parent.name
        entries.append((label, block))
        world_frame_id = world_frame_id_of(block)
        if world_frame_id:
            sha = str((block or {}).get("reference_sha256") or "")
            suffix = f"  ref {sha[:12]}" if sha else ""
            lines.append(f"  {label}: {world_frame_id}{suffix}")
        else:
            reason = (block or {}).get("status") if block else "no world_frame block"
            lines.append(f"  {label}: UNSTAMPED ({reason})")

    try:
        shared = assert_single_world(entries)
    except MixedWorldError as exc:
        lines.append(f"FAIL: {exc}")
        return 1, lines
    if not shared:
        lines.append("FAIL: no episode carries a world frame id (all predate world provenance)")
        return 1, lines
    lines.append(f"OK: all {len(entries)} episode(s) in {shared}")
    return 0, lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help="dataset directory to check; omit to print this repo's frozen reference",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repo root holding tools/thor/gmsl2/world/",
    )
    args = parser.parse_args(argv)

    if args.dataset_root is None:
        print(describe(read_world_provenance(args.repo_root)))
        return 0
    code, lines = inspect_dataset(args.dataset_root)
    for line in lines:
        print(line)
    return code


__all__ = [
    "WORLD_SUBDIR",
    "WORLD_REFERENCE_FILE",
    "WORLD_REGISTRATION_FILE",
    "STATUS_OK",
    "STATUS_MISSING",
    "STATUS_UNREADABLE",
    "STATUS_INCOMPLETE",
    "MixedWorldError",
    "read_world_provenance",
    "describe",
    "world_frame_id_of",
    "assert_single_world",
    "inspect_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
