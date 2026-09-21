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

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Where the frozen reference lives, relative to the repo root.  It is tracked in
#: git and arrives on Thor by ``rsync`` with the rest of the tree -- never by
#: running ``freeze`` there, which would mint a second id for one physical frame.
WORLD_SUBDIR = Path("tools") / "thor" / "gmsl2" / "world"
WORLD_REFERENCE_FILE = "world_reference.json"
WORLD_REGISTRATION_FILE = "world_registration.json"

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
