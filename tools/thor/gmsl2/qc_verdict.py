"""Which recorded episodes may go into a training export: each session's QC verdict, bound to its data.

One rule, whichever way an export is started (Dataset Export for one dataset, Dataset Export for a
whole task, or ``export_v3.py`` by hand): an episode is exported only when QC ran on the session's data
*as it is now* and did not fail that episode. Everything else stays out:

* a session QC failed as a whole (a check that cannot name episodes failed);
* a session QC never ran on, or whose QC did not finish;
* a session changed since its QC. Deleting an episode in Episode Replay renumbers the ones after it,
  so an old verdict would name other episodes; a re-run trajectory or a BOX re-pairing rewrites what
  QC judged.

"Not shown to be good" and "shown to be bad" have the same effect on the export; only the way out
differs (run QC again, versus re-record or delete the episode). There is no "export anyway" in the GUI;
``export_v3.py --ignore-qc`` exists for a maintainer and is written into the dataset's info.json.

The gateway's QC stores its verdict in ``meta/processing.json`` (``versions[active_version].qc``) with
``data_fingerprint``: a hash of the names, sizes and modification times of what QC and the export read
(``meta/info.json``, ``meta/episodes/``, ``data/``, ``episodes/`` and the trajectory sidecar's CSVs, not
its subdirectories, which replay and IK previews rewrite on demand). Anything that adds, removes,
rewrites or renumbers those files changes the hash. Stdlib only: the gateway and export_v3 both use it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FINGERPRINT_VERSION = "qcfp1"
TRAJ_SIDECAR_NAME = "april_cube_tracking_in_robot_base"
PROCESSING_META = Path("meta") / "processing.json"

PASS = "pass"  # every episode may be exported
PARTIAL = "partial"  # the episodes in ``excluded`` stay out, the rest may be exported
FAIL = "fail"  # QC failed the session as a whole
NOT_RUN = "not_run"  # no QC verdict for the current trajectory version
STALE = "stale"  # the data changed since QC (or QC predates fingerprints)

EXPORTABLE = (PASS, PARTIAL)


def _walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        out.extend(Path(dirpath) / name for name in sorted(filenames))
    return out


def fingerprint_inputs(session_dir: Path) -> list[Path]:
    """The files whose names, sizes and modification times make up the data fingerprint."""
    session_dir = Path(session_dir)
    files: list[Path] = []
    info = session_dir / "meta" / "info.json"
    if info.is_file():
        files.append(info)
    files += _walk_files(session_dir / "meta" / "episodes")
    files += _walk_files(session_dir / "data")
    files += _walk_files(session_dir / "episodes")
    sidecar = session_dir / "derived" / TRAJ_SIDECAR_NAME
    if sidecar.is_dir():
        files += sorted(p for p in sidecar.glob("*.csv") if p.is_file())
    return files


def data_fingerprint(session_dir: Path) -> str:
    """Hash of what QC judged; changes when any of :func:`fingerprint_inputs` changes."""
    session_dir = Path(session_dir)
    digest = hashlib.sha256()
    # Episode directories by name too: an empty one is still an episode to the export.
    episodes = session_dir / "episodes"
    if episodes.is_dir():
        for path in sorted(p for p in episodes.iterdir() if p.is_dir()):
            digest.update(f"dir\t{path.name}\n".encode())
    for path in fingerprint_inputs(session_dir):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(session_dir).as_posix()
        digest.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return f"{FINGERPRINT_VERSION}:{digest.hexdigest()}"


@dataclass(frozen=True)
class Verdict:
    """What the export may take from one session, and why."""

    session: str
    state: str
    reason: str
    # PARTIAL: episode index -> why it stays out ("QC: frame_continuity, quat_norm").
    excluded: dict[int, str] = field(default_factory=dict)

    @property
    def exportable(self) -> bool:
        return self.state in EXPORTABLE

    def allows(self, episode: int) -> bool:
        return self.exportable and int(episode) not in self.excluded


def active_qc(processing_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """The QC result of the active trajectory version in a processing.json payload, if any."""
    if not isinstance(processing_meta, dict):
        return None
    versions = processing_meta.get("versions")
    active = processing_meta.get("active_version")
    if not isinstance(versions, dict) or not isinstance(active, str):
        return None
    version = versions.get(active)
    qc = version.get("qc") if isinstance(version, dict) else None
    return qc if isinstance(qc, dict) else None


def load_processing_meta(session_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((Path(session_dir) / PROCESSING_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def verdict_from_qc(session_dir: Path, qc: dict[str, Any] | None, *, fingerprint: str | None = None) -> Verdict:
    """The verdict for ``session_dir`` given its active QC result (``None``: QC not run)."""
    session_dir = Path(session_dir)
    name = session_dir.name
    if not isinstance(qc, dict) or not qc.get("status"):
        return Verdict(name, NOT_RUN, "QC has not run on this trajectory version")
    recorded = str(qc.get("data_fingerprint") or "")
    if not recorded:
        return Verdict(name, STALE, "this QC result predates data fingerprints; run QC again")
    current = fingerprint if fingerprint is not None else data_fingerprint(session_dir)
    if recorded != current:
        return Verdict(
            name,
            STALE,
            "the data changed after QC (episodes deleted, trajectory or BOX data rewritten); run QC again",
        )
    status = str(qc.get("status") or "").lower()
    summary = str(qc.get("summary") or status)
    if status in ("pass", "warn"):
        return Verdict(name, PASS, f"QC {status}: {summary}")
    if status == "partial":
        excluded: dict[int, str] = {}
        for row in qc.get("excluded_episodes") or []:
            if not isinstance(row, dict):
                continue
            try:
                episode = int(row.get("episode"))
            except (TypeError, ValueError):
                continue
            checks = [str(c) for c in row.get("checks") or []]
            excluded[episode] = "QC: " + (", ".join(checks) if checks else "failed")
        if not excluded:
            return Verdict(name, FAIL, "QC says partial but names no episode; run QC again")
        return Verdict(name, PARTIAL, f"QC partial: {summary}", excluded)
    return Verdict(name, FAIL, f"QC {status or 'unknown'}: {summary}")


def read_verdict(session_dir: Path) -> Verdict:
    """The verdict for ``session_dir`` from its own meta/processing.json."""
    return verdict_from_qc(session_dir, active_qc(load_processing_meta(session_dir)))
