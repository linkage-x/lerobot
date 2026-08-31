#!/usr/bin/env python
"""Grade the rollouts recorded before the ladder existed, from what the operator wrote at the time.

One-off migration for the batches of 2026-08-31, which were recorded as `success`/`failure` with
the real information in a free-text note. The mapping below is the whole point of the file: it is
written out rather than inferred so that a later reader can disagree with a specific line instead
of with a regex, and an unrecognised note stops the run rather than being graded as something.

`blocker` is deliberately left `unknown` for every migrated record. The notes say what happened,
not why, and attributing a cause the operator never gave would put invented evidence in the field
whose whole purpose is to separate work items. It starts carrying information with the next batch.

    python tools/fr3/fr3_backfill_rollout_stages.py --session 071141
    python tools/fr3/fr3_backfill_rollout_stages.py --session 071141 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_collection_gui import task_ladders  # noqa: E402

# Matched as substrings against the lowercased note, first match wins, so the more specific
# phrasings come first. Every distinct note in the 08-31 batches is covered.
NOTE_STAGES: tuple[tuple[str, str], ...] = (
    # "Inserted in the hole, but not fully" -- the peg reached the target and the constraint is
    # only partly satisfied. The one rollout of 29 that got this far.
    ("inserted in the hole", "target_contact"),
    # Carried it to the hole and could not get it in. `secure` and `transport` are not separable
    # from these notes, and the operator wrote them from above the hole, so: transport.
    ("failed to insert", "transport"),
    ("not inserted", "transport"),
    ("not insert", "transport"),
    # Closed on the peg and knocked it over: contact was made, no load-bearing constraint formed.
    ("push down", "contact"),
    # Closed on air. Whether the gripper brushed the peg on the way is not recoverable from the
    # note, and the operator confirmed these are to be read as "never established contact".
    ("grasped empty", "approach"),
    ("not grasp", "approach"),
    # Held it and let go early. It was secured, so it is past `contact`.
    ("released early", "secure"),
)


def stage_for(note: str) -> str | None:
    lowered = note.lower()
    for needle, stage_id in NOTE_STAGES:
        if needle in lowered:
            return stage_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=ROOT / "outputs" / "rollouts" / "rollout_log.jsonl")
    parser.add_argument("--task", default="insert_peg")
    parser.add_argument("--session", help="substring of the rollout log filename to restrict to")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    ladder = task_ladders.find_ladder(ROOT, args.task)
    records = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]

    selected, unmatched, changed = [], [], 0
    for record in records:
        if args.session and args.session not in str(record.get("logPath") or ""):
            continue
        if record.get("stage") is not None:
            continue
        note = str(record.get("note") or "")
        if not note:
            continue
        selected.append(record)
        stage_id = stage_for(note)
        if stage_id is None:
            unmatched.append(note)
            continue
        graded = task_ladders.normalize_grade(
            {"stageId": stage_id, "outcome": record.get("outcome")}, ladder
        )
        record.update(graded)
        changed += 1
        print(f"rollout {record.get('rolloutIndex'):>3}  stage {graded['stage']} {graded['stageId']:<15} <- {note!r}")

    if unmatched:
        print(f"\n{len(unmatched)} note(s) match no rule; nothing written. Add a rule for each:")
        for note in sorted(set(unmatched)):
            print(f"    {note!r}")
        return 2

    print(f"\n{'=' * 60}\n漏斗 (graded={changed})")
    for row in task_ladders.stage_funnel(selected, ladder):
        bar = "#" * row["reached"]
        flag = "  <- 最大流失" if row["lost"] and row["lost"] == max(
            r["lost"] for r in task_ladders.stage_funnel(selected, ladder)
        ) else ""
        print(f"  {row['stage']} {row['label']:<6} >= {row['reached']:>3}/{row['graded']:<3} "
              f"流失 {row['lost']:>2}  {bar}{flag}")

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
