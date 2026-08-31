#!/usr/bin/env python
"""Re-reduce archived rollout traces and rewrite the geometry the log got wrong.

The per-step CSVs under `outputs/rollout_traces/` are the record; the `geometry` block in
`outputs/rollouts/rollout_log.jsonl` is a view of them, produced once at the end of each rollout.
When the reduction rule is corrected, the traces are still in hand -- that is what
`RolloutGeometryTrace` was written to preserve -- so the view can simply be read again.

This rewrites nothing else. `outcome` and `note` are the operator's, and they are the only
ground truth for whether a rollout succeeded: the trace cannot distinguish a held object from
a gripper closed on air, so nothing here may touch them.

    python tools/fr3/fr3_rederive_rollout_geometry.py            # dry run, prints the diff
    python tools/fr3/fr3_rederive_rollout_geometry.py --apply    # rewrites, after a backup

Records whose reduction did not change are left byte-identical, so the diff is the evidence of
what the fix actually moved.

One hazard this guards against: `rolloutIndex` restarts at 1 for every interactive session, and
the traces are named by that index alone, so a new session silently overwrites the previous
session's `rollout_001.csv` onward. A log holding more than one session therefore cannot be
re-derived from the live trace directory -- the CSVs no longer belong to the older records. When
that is the case this refuses to guess and asks for `--session` plus the archived `--traces` for
it. Read a refusal as "those traces are gone from the live directory", not as a tool problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tools" / "fr3" / "fr3_act_infer_real_runtime.py"
SECTION_START = "# A rollout's gripper command is the event signal"
SECTION_CLASS = "class RolloutGeometryTrace"

# The runtime prints geometry with `%.4f` and the gateway floats it back, so 4 decimals is the
# precision a fresh record carries. Older records came from a runtime that printed 5, which is
# why re-deriving them differs in the 6th digit on rollouts whose reduction never changed.
# `MATERIAL_M` sits above that rounding noise and far below anything the landing map or the
# roadmap reads: a record is rewritten only when the rule landed somewhere else, never to
# restyle a number.
DECIMALS = 4
MATERIAL_M = 1.5e-4

FIELD_NAMES = {
    "grasp_xyz": "graspXyz",
    "release_xyz": "releaseXyz",
    "approach_xyz": "approachXyz",
    "apex_z": "apexZ",
    "lift_m": "liftM",
    "descent_m": "descentM",
    "samples": "samples",
    "held_steps": "heldSteps",
    "closed": "closed",
}


def load_trace_class() -> Any:
    """`RolloutGeometryTrace`, exec'd from the runtime's own text.

    Imported this way rather than with `from tools.fr3 import ...` because the runtime pulls
    `lerobot.policies`, which drags in transformers and cannot be imported in every env the
    rig uses. The reduction itself is pure numpy and needs none of that.
    """
    source = RUNTIME.read_text()
    start = source.index(SECTION_START)
    class_at = source.index(SECTION_CLASS, start)
    tail = re.search(r"\n(?=(?:class |def )\w)", source[class_at:])
    end = class_at + (tail.start() if tail else len(source) - class_at)
    module = types.ModuleType("fr3_geometry")
    module.__dict__.update({"np": np, "csv": csv, "Path": Path, "Any": Any})
    exec(compile(source[start:end], f"{RUNTIME}<geometry>", "exec"), module.__dict__)
    return module.RolloutGeometryTrace


def rederive(trace_cls: Any, path: Path) -> dict[str, Any]:
    trace = trace_cls(int(path.stem.split("_")[1]))
    for step, row in enumerate(csv.DictReader(path.open())):
        trace.sample(
            step_idx=step,
            position_xyz=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            gripper_command=float(row["gripper_cmd"]),
            gripper_raw=float(row["gripper_raw"]),
            command_status=row["status"],
        )
    out: dict[str, Any] = {}
    for key, value in trace.summary().items():
        name = FIELD_NAMES.get(key, key)
        if isinstance(value, (np.ndarray, list)):
            value = [round(float(v), DECIMALS) for v in value]
        elif isinstance(value, (bool, np.bool_)):
            value = bool(value)
        elif isinstance(value, (float, np.floating)):
            value = round(float(value), DECIMALS)
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        out[name] = value
    return out


def material(old: Any, new: Any) -> bool:
    """Whether the difference is the rule landing elsewhere rather than a rounding artefact."""
    if isinstance(new, bool) or isinstance(old, bool):
        return bool(old) != bool(new)
    if isinstance(new, int):
        return old != new
    if isinstance(new, list):
        if not isinstance(old, list) or len(old) != len(new):
            return True
        return any(abs(float(a) - float(b)) > MATERIAL_M for a, b in zip(old, new))
    if isinstance(new, float):
        return old is None or abs(float(old) - float(new)) > MATERIAL_M
    return old != new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=ROOT / "outputs" / "rollouts" / "rollout_log.jsonl")
    parser.add_argument("--traces", type=Path, default=ROOT / "outputs" / "rollout_traces")
    parser.add_argument("--apply", action="store_true", help="rewrite the log; without it, print the diff only")
    parser.add_argument(
        "--session",
        help="substring of the rollout log filename to restrict to, e.g. 20260831_033716. "
        "Required when the log holds more than one indexed session.",
    )
    args = parser.parse_args()

    trace_cls = load_trace_class()
    records = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]

    indexed = [r for r in records if r.get("rolloutIndex") is not None]
    sessions = sorted({str(r.get("logPath") or "") for r in indexed})
    if args.session:
        selected = [s for s in sessions if args.session in s]
        if len(selected) != 1:
            print(f"--session {args.session!r} matches {len(selected)} of {len(sessions)} sessions:")
            for name in sessions:
                print(f"    {name}")
            return 2
        sessions = selected
    elif len(sessions) > 1:
        # Refusing rather than defaulting to the newest: the failure mode is silent, and it
        # writes one session's landing points onto another session's rollouts.
        print(f"{args.log} holds {len(sessions)} indexed sessions and rolloutIndex restarts at 1 in each:")
        for name in sessions:
            count = sum(1 for r in indexed if r.get("logPath") == name)
            print(f"    {count:>3} rollouts  {name}")
        print(
            "\nThe live trace directory only holds the newest session's CSVs -- the others were\n"
            "overwritten by index collision. Re-run with --session <substring> and point --traces\n"
            "at that session's archived traces."
        )
        return 2

    only = sessions[0] if sessions else None
    changed = 0
    for record in records:
        index = record.get("rolloutIndex")
        if index is None:
            continue
        if only is not None and str(record.get("logPath") or "") != only:
            continue
        trace_path = args.traces / f"rollout_{int(index):03d}.csv"
        if not trace_path.exists():
            print(f"rollout {index:>2}  no retained trace, left alone")
            continue
        new = rederive(trace_cls, trace_path)
        old = record.get("geometry") or {}
        diff = {key: (old.get(key), value) for key, value in new.items() if material(old.get(key), value)}
        stale = [key for key in old if key not in new]
        if not diff and not stale:
            print(f"rollout {index:>2}  reduction unchanged, record left byte-identical")
            continue
        changed += 1
        print(f"rollout {index:>2}  outcome={record.get('outcome')!r} note={record.get('note')!r}")
        for key, (was, now) in diff.items():
            print(f"            {key:<12} {was!r:>34}  ->  {now!r}")
        for key in stale:
            print(f"            {key:<12} {old[key]!r:>34}  ->  (removed)")
        record["geometry"] = new

    if not args.apply:
        print(f"\nDRY RUN. {changed} record(s) would change. Nothing written.")
        return 0
    backup = args.log.with_name(f"{args.log.name}.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.log, backup)
    args.log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"\nWrote {changed} record(s). Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
