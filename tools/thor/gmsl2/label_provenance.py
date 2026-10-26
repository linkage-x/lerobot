"""Which measurement chain produced an export's pose labels, and refusing to mix two of them.

``world_frame_id`` says which coordinate system a dataset's absolute poses are in.  It does not
say how well they were measured there, and since 2026-10 several things that move the labels by
millimetres change *without* changing it:

* the **extrinsics** the tracker solved against.  10-14 the rig was recalibrated and kept its
  world id (registered by camera centres), so sessions tracked on calib_20260928 and on
  calib_20261014 are both ``world_20260928_063531``;
* the **camera state**: whether the cameras were still where those extrinsics put them
  (``metrology.camera_state``, corner-BA median > 1 px = stale).  10-09 -> 10-13 seven cameras had
  tipped 0.2-0.6 deg and the labels carried ~4 mm with it, same world id, same extrinsics id;
* the **world -> fr3_base edge**, replaced in place in ``world_graph.json`` (10-09 hand-eye,
  10-14 ball check, 10-15 ball check refit).  Every export through any of them says
  ``fr3_base``;
* the **marker -> TCP bundle** (which physical point the label is), the **BOX pairing** (raw SOF
  vs mid-exposure + d_box, v1.27) and the tracker's **sidecar chain** (schema / pose source /
  smoother).

Each export records all of them per session (``meta/info.json`` ``label_provenance``) and refuses
to put two values of any one into one dataset; training refuses to combine two exports whose
``key`` differs.  This is the rule :func:`world_provenance.assert_single_world` applies to the
world id, extended to the rest of the chain: a missing value is a value, so a dataset that
predates the stamp does not combine with one that has it.

A session with no tracking sidecar carries no pose labels and takes no part in the comparison.

Deliberately stdlib-only, like ``world_provenance``: the exporter runs in Thor's system python.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402

SIDECAR = "april_cube_tracking_in_robot_base"
TRACKING_RUN_SUFFIX = "_thor_april_tracking_in_robot_base"

#: The per-session fields that must agree within an export; each is one link of the label chain.
SESSION_FIELDS = (
    "intrinsics",
    "extrinsics",
    "camera_state",
    "marker_to_tcp",
    "tcp_link",
    "sidecar_chain",
    "box_pairing",
)
#: What a dataset's ``key`` holds: the session fields plus the world and edge it was exported in.
KEY_FIELDS = ("world_frame_id", "edge", *SESSION_FIELDS)
#: Keys of ``derived.<sidecar>`` that identify the tracker's chain (as ``aggregate.DERIVED_CHAIN_KEYS``).
SIDECAR_CHAIN_KEYS = ("sidecar_schema_version", "sidecar_pose_source", "sidecar_smoothing")

UNTRACKED = "untracked"

#: Where the per-BOX delay table lives (thor_lerobot_v3.BOX_TIME_DELAYS_PATH), from the repo root.
BOX_TIME_DELAYS = Path("tools/thor/gmsl2/box_time_delays.json")
MEASURED = "measured"
UNVALIDATED = "unvalidated"


class MixedLabelsError(RuntimeError):
    """Raised when sessions or datasets about to share a training set come from different chains."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def box_pairing(episode_meta: Mapping[str, Any], realignment: Mapping[str, Any] | None) -> str:
    """How an episode's BOX columns were paired with the camera frames.

    ``realignment`` is this episode's entry of the session's ``box_alignment`` (box_realign.py),
    which overrides what the recorder did.  Mid-exposure + d_box is one scheme whether the delay
    came from the table or from the episode's own estimate; a BOX with no delay left at the raw
    SOF inside such an episode does not change the scheme the episode was recorded under.
    """
    if isinstance(realignment, Mapping) and realignment.get("boxes"):
        targets = {str((b or {}).get("target") or "") for b in realignment["boxes"].values()}
        return "mid_exposure_plus_d_box" if "mid_exposure_plus_d_box" in targets else "raw_sof"
    alignment = episode_meta.get("box_camera_alignment")
    if not isinstance(alignment, Mapping):
        return "unrecorded"
    mode = str(alignment.get("mode") or "")
    if mode != "sensor_timestamp_sof":
        return mode or "unrecorded"
    targets = {str((b or {}).get("target") or "") for b in (alignment.get("box_targets") or {}).values()}
    if "mid_exposure_plus_d_box" in targets:
        return "mid_exposure_plus_d_box"
    fraction = float(alignment.get("exposure_fraction") or 0.0)
    return "raw_sof" if fraction == 0.0 else f"exposure_fraction_{fraction:g}"


def _laser_measured(bundle: Mapping[str, Any], cube: str) -> bool:
    entry = (bundle.get("cubes") or {}).get(cube) or {}
    method = str(((bundle.get("provenance") or {}).get(cube) or {}).get("method") or "")
    return "laser" in method or str((entry.get("pivot") or {}).get("source") or "") == "laser_tracker"


def hand_validation(
    cube: str,
    bundle: Mapping[str, Any] | None,
    delays: Mapping[str, Any] | None,
    realigned: Mapping[str, Any],
) -> dict[str, Any]:
    """What stands behind one hand's absolute labels, and what does not yet.

    The known unknowns of a hand's chain (roadmap A3), each read from the artefact that would
    close it, so a hand stops being ``unvalidated`` the day the measurement lands and not the day
    someone edits a list:

    * marker -> TCP measured by the laser tracker (``provenance.<cube>.method`` / ``pivot.source``
      of the bundle).  A camera pivot's fit residual is not its accuracy -- seven single-camera
      solutions of the 08-25 left pivot disagree by 7 mm;
    * marker -> TCP not flagged stale (an ``open_items`` entry starting ``STALE``);
    * the hand's BOX has a measured ``d_box`` (the delay table, or an episode's own estimate in
      the session's ``box_alignment``); without one its BOX columns are read at the raw SOF.

    This does not refuse anything: the labels export as they are, and the reasons travel with
    them so nobody reads an unvalidated hand's absolute error as execution or registration error.
    """
    reasons = []
    entry = ((bundle or {}).get("cubes") or {}).get(cube)
    device = str((entry or {}).get("device_id") or "")
    if bundle is None:
        reasons.append("marker->TCP bundle not readable here; nothing says what this hand's TCP was measured with")
    elif not isinstance(entry, Mapping):
        reasons.append(f"marker->TCP bundle has no entry for {cube!r}")
    else:
        if not _laser_measured(bundle, cube):
            source = str(entry.get("translation_source") or "unrecorded")
            reasons.append(f"marker->TCP never checked by the laser tracker (translation: {source})")
        stale = [str(item) for item in entry.get("open_items") or [] if str(item).startswith("STALE")]
        reasons.extend(f"marker->TCP {item}" for item in stale)
    own = sorted(
        str(index)
        for index, episode in realigned.items()
        if str((((episode or {}).get("boxes") or {}).get(device) or {}).get("source") or "") == "episode"
    )
    in_table = bool(device) and device in ((delays or {}).get("boxes") or {})
    if not in_table and not own:
        reasons.append(
            f"d_box of {device or 'its BOX'} not measured: the BOX columns are read at the raw SOF"
        )
    return {
        "status": UNVALIDATED if reasons else MEASURED,
        "reasons": reasons,
        "box_device_id": device,
        "d_box": "table" if in_table else ("episode" if own else "none"),
        "marker_to_tcp_laser": bool(bundle is not None and isinstance(entry, Mapping) and _laser_measured(bundle, cube)),
    }


def tracked_hands(session_dir: Path) -> list[str]:
    """The hands a session has TCP labels for (``state_action.<cube>.csv`` in its sidecar)."""
    sidecar = session_dir / "derived" / SIDECAR
    return sorted(p.name[len("state_action.") : -len(".csv")] for p in sidecar.glob("state_action.*.csv"))


def session_labels(
    session_dir: Path,
    tracking_run_dir: Path,
    episode_metas: Mapping[int, Mapping[str, Any]],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The label chain of one recorded session, from its tracking run and its own meta.

    ``episode_metas`` are the session's episodes that go into the export, by local index.
    ``repo_root`` resolves the marker->TCP bundle path the tracking run recorded and the BOX
    delay table; each hand's validation (:func:`hand_validation`) rides along in ``hands`` and
    is not part of the key -- it says how far to trust the labels, not which labels they are.
    """
    repo_root = _REPO_ROOT if repo_root is None else Path(repo_root)
    info = _read_json(session_dir / "meta" / "info.json")
    realigned = ((info.get("box_alignment") or {}).get("episodes")) or {}
    pairing = sorted(
        {box_pairing(meta, realigned.get(str(index))) for index, meta in episode_metas.items()}
    )
    derived = (info.get("derived") or {}).get(SIDECAR)
    summary = _read_json(tracking_run_dir / "summary.json")
    tracked = isinstance(derived, Mapping) or bool(summary) or (session_dir / "derived" / SIDECAR).is_dir()
    if not tracked:
        return {"tracked": False, "box_pairing": pairing}

    derived = derived if isinstance(derived, Mapping) else {}
    inputs = summary.get("calibration_inputs") or {}
    ee = summary.get("ee_from_cube") or {}
    state = derived.get("camera_state") if isinstance(derived.get("camera_state"), Mapping) else None
    state = state or (summary.get("camera_state") if isinstance(summary.get("camera_state"), Mapping) else None)
    bundle_path = Path(str(ee.get("marker_to_tcp_calibration_path") or ""))
    bundle_path = bundle_path if bundle_path.is_absolute() else repo_root / bundle_path
    bundle = _read_json(bundle_path) if bundle_path.is_file() else {}
    delays = _read_json(repo_root / BOX_TIME_DELAYS)
    exported = {str(index): realigned.get(str(index)) for index in episode_metas if str(index) in realigned}
    hands = {
        cube: hand_validation(cube, bundle or None, delays or None, exported) for cube in tracked_hands(session_dir)
    }
    return {
        "tracked": True,
        "intrinsics": str(inputs.get("intrinsics_run_name") or ""),
        "extrinsics": str(inputs.get("fixed_camera_run_name") or ""),
        # The verdict is what has to agree; the number and the extrinsics it was read against are
        # kept beside it for whoever has to decide what a stale session is worth.
        "camera_state": str((state or {}).get("verdict") or "unrecorded"),
        "camera_state_px": (state or {}).get("worst_median_reproj_px"),
        "camera_state_extrinsics": str((state or {}).get("extrinsics_run_name") or ""),
        "marker_to_tcp": str(ee.get("marker_to_tcp_calibration_path") or ""),
        "tcp_link": str(ee.get("target_child_link") or ee.get("target_parent_link") or ""),
        "sidecar_chain": _canonical({k: derived.get(k) for k in SIDECAR_CHAIN_KEYS}),
        "box_pairing": pairing,
        "hands": hands,
    }


def _field_values(labels: Mapping[str, Any], field: str) -> list[str]:
    value = labels.get(field)
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value if value is not None else "")]


def mixed_session_fields(sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """``{field: {value: [session, ...]}}`` for every field on which the sessions disagree.

    BOX pairing is compared over every session; the pose-chain fields only over tracked ones.
    """
    mixed: dict[str, dict[str, list[str]]] = {}
    for field in SESSION_FIELDS:
        by_value: dict[str, list[str]] = {}
        for name, labels in sessions.items():
            if field != "box_pairing" and not labels.get("tracked"):
                continue
            for value in _field_values(labels, field):
                by_value.setdefault(value, []).append(name)
        if len(by_value) > 1:
            mixed[field] = by_value
    return mixed


def describe_mixed(mixed: Mapping[str, Mapping[str, list[str]]]) -> str:
    lines = []
    for field, by_value in mixed.items():
        lines.append(f"  {field}:")
        for value, names in sorted(by_value.items()):
            lines.append(f"    {value or '<none>'}: {', '.join(names)}")
    return "\n".join(lines)


def assert_single_session_chain(
    sessions: Mapping[str, Mapping[str, Any]], *, allow_mixed: bool = False
) -> list[str]:
    """Refuse an export whose sessions disagree on a link of the label chain.

    Returns the fields that were mixed (empty unless ``allow_mixed``), for the record.
    """
    mixed = mixed_session_fields(sessions)
    if mixed and not allow_mixed:
        raise MixedLabelsError(
            "the sessions' pose labels do not come from one measurement chain:\n"
            f"{describe_mixed(mixed)}\n"
            "Each line is a few millimetres the labels disagree by with the same world id. Export "
            "the sessions of one chain at a time (re-track, re-pair with box_realign.py, or "
            "leave the odd sessions out), or pass --allow-mixed-labels to export them together "
            "anyway -- the dataset then records which fields it mixed and training will refuse to "
            "combine it with a clean one."
        )
    return sorted(mixed)


def _edge_key(world_frame: Mapping[str, Any] | None) -> Any:
    """The transform an export was moved through, and the edges it was built from.

    The product alone would do to tell two edges apart, but naming the edges (endpoints and
    ``created_utc``) is what lets a reader find which version of ``world_graph.json`` it was.
    """
    if not isinstance(world_frame, Mapping) or not isinstance(world_frame.get("reexpressed_from"), Mapping):
        return None
    matrix = world_frame.get("T_target_source") or []
    return {
        "T_target_source": [[round(float(v), 9) for v in row] for row in matrix],
        "edges": [
            [
                str(hop.get("from_world_frame_id") or ""),
                str(hop.get("to_world_frame_id") or ""),
                str(hop.get("created_utc") or ""),
                bool(hop.get("traversed_reversed")),
            ]
            for hop in world_frame.get("edges") or []
        ],
    }


def dataset_key(
    world_frame: Mapping[str, Any] | None, sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """What two exports must share to be one training set: the union of every session's value."""
    key: dict[str, Any] = {
        "world_frame_id": str((world_frame or {}).get("world_frame_id") or "")
        if str((world_frame or {}).get("status") or "") == "ok"
        else "",
        "edge": _edge_key(world_frame),
    }
    tracked = {name: labels for name, labels in sessions.items() if labels.get("tracked")}
    for field in SESSION_FIELDS:
        pool = sessions if field == "box_pairing" else tracked
        key[field] = sorted({v for labels in pool.values() for v in _field_values(labels, field)})
    return key


def fingerprint(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]


def dataset_hands(sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each hand's validation over an export: ``unvalidated`` if it is in any session."""
    hands: dict[str, dict[str, Any]] = {}
    for name, labels in sorted(sessions.items()):
        for cube, check in (labels.get("hands") or {}).items():
            hand = hands.setdefault(cube, {"status": MEASURED, "reasons": [], "sessions": {}})
            hand["sessions"][name] = check.get("status")
            if check.get("status") != MEASURED:
                hand["status"] = UNVALIDATED
                hand["reasons"].extend(r for r in check.get("reasons") or [] if r not in hand["reasons"])
    return hands


def hand_warnings(block: Mapping[str, Any] | None) -> list[str]:
    """One line per unvalidated hand of an export's ``label_provenance``."""
    out = []
    for cube, hand in sorted(((block or {}).get("hands") or {}).items()):
        if hand.get("status") == UNVALIDATED:
            out.append(
                f"{cube} hand labels are unvalidated ({'; '.join(hand.get('reasons') or [])}): read its "
                "absolute error as label-chain error first, not as execution or registration error"
            )
    return out


def provenance_block(
    world_frame: Mapping[str, Any] | None,
    sessions: Mapping[str, Mapping[str, Any]],
    mixed_fields: list[str],
) -> dict[str, Any]:
    """``info.json["label_provenance"]`` of an export."""
    key = dataset_key(world_frame, sessions)
    return {
        "fingerprint": fingerprint(key),
        "key": key,
        "hands": dataset_hands(sessions),
        "mixed_fields": list(mixed_fields),
        "sessions": {name: dict(labels) for name, labels in sessions.items()},
        "note": (
            "two exports whose fingerprints differ were labelled by different measurement chains "
            "(tools/thor/gmsl2/label_provenance.py) and must not be trained on together"
        ),
    }


def key_of_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """An export's key, read back from its info.json.

    An export from before the stamp gets what its ``world_frame`` says and ``None`` for the rest,
    so it cannot be combined with one that has the stamp -- only with another of its kind
    exported through the same edge.
    """
    block = info.get("label_provenance")
    if isinstance(block, Mapping) and isinstance(block.get("key"), Mapping):
        return dict(block["key"])
    world = info.get("world_frame") if isinstance(info.get("world_frame"), Mapping) else None
    key: dict[str, Any] = dict.fromkeys(KEY_FIELDS)
    key["world_frame_id"] = str((world or {}).get("world_frame_id") or "")
    key["edge"] = _edge_key(world)
    tcp = info.get("tcp_frame") if isinstance(info.get("tcp_frame"), Mapping) else {}
    if tcp:
        key["tcp_link"] = [str(tcp.get("link") or "")]
        key["marker_to_tcp"] = sorted(str(v) for v in tcp.get("marker_to_tcp_calibrations") or [])
    return key


def assert_compatible_datasets(entries: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Refuse to train on exports labelled by different chains; returns the shared key.

    ``entries`` are ``(name, info.json)`` pairs.
    """
    entries = list(entries)
    keys = [(name, key_of_info(info)) for name, info in entries]
    if not keys:
        return {}
    differing = [
        field for field in KEY_FIELDS if len({_canonical(key.get(field)) for _, key in keys}) > 1
    ]
    if differing:
        lines = []
        for field in differing:
            lines.append(f"  {field}:")
            for name, key in keys:
                lines.append(f"    {name}: {_canonical(key.get(field))}")
        raise MixedLabelsError(
            "the source datasets' pose labels come from different measurement chains:\n"
            + "\n".join(lines)
            + "\nRe-export them together (one export_v3 run, which checks every session) or train "
            "on one of them."
        )
    mixed = [
        name
        for name, info in entries
        if isinstance(info.get("label_provenance"), Mapping) and info["label_provenance"].get("mixed_fields")
    ]
    if mixed and len(entries) > 1:
        raise MixedLabelsError(
            f"{', '.join(mixed)} was exported with --allow-mixed-labels; it is one training set only "
            "on its own"
        )
    return keys[0][1]


def label_warnings(key: Mapping[str, Any], current_edge: Any = None) -> list[str]:
    """What a compatible training set should still be told about itself."""
    warnings = []
    states = key.get("camera_state")
    if isinstance(states, list) and any(state != "ok" for state in states):
        warnings.append(
            f"camera state {', '.join(states)}: the extrinsics did not fit the cameras when these were "
            "recorded (corner BA > 1 px); absolute labels are off by millimetres"
        )
    if key.get("edge") is not None and current_edge is not None and _canonical(key["edge"]) != _canonical(current_edge):
        exported = ", ".join(hop[2] for hop in key["edge"].get("edges") or []) or "?"
        current = ", ".join(hop[2] for hop in current_edge.get("edges") or []) or "?"
        warnings.append(
            f"exported through an edge that world_graph.json has since replaced (created {exported}; "
            f"now {current}); re-export to label in the current fr3_base"
        )
    return warnings


def current_edge_key(repo_root: Path, world_frame: Mapping[str, Any] | None) -> Any:
    """The edge key an export of this world would get from today's ``world_graph.json``."""
    if not isinstance(world_frame, Mapping) or not isinstance(world_frame.get("reexpressed_from"), Mapping):
        return None
    source = wp.world_frame_id_of(world_frame["reexpressed_from"])
    target = str(world_frame.get("world_frame_id") or "")
    found = wp.world_transform(wp.read_world_graph(repo_root), source, target) if source else None
    if found is None:
        return None
    matrix, edges = found
    return _edge_key({"reexpressed_from": {}, "T_target_source": matrix, "edges": edges})


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Print exports' label chains and check they combine")
    parser.add_argument("datasets", nargs="+", type=Path, help="export roots (with meta/info.json)")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)

    entries = []
    for root in args.datasets:
        info = _read_json(root / "meta" / "info.json")
        key = key_of_info(info)
        entries.append((root.name, info))
        block = info.get("label_provenance") or {}
        print(f"{root.name}: fingerprint {block.get('fingerprint') or '<unstamped>'}")
        for field in KEY_FIELDS:
            print(f"  {field}: {_canonical(key.get(field))}")
        for warning in label_warnings(key, current_edge_key(args.repo_root, info.get("world_frame"))):
            print(f"  WARNING: {warning}")
        for warning in hand_warnings(block):
            print(f"  WARNING: {warning}")
    try:
        assert_compatible_datasets(entries)
    except MixedLabelsError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: {len(entries)} dataset(s) share one label chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
