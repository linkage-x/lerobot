#!/usr/bin/env python3
"""The pictures taken at each insertion grade, with the person's answer: an in/out judge's data.

Since 11-13 the grasp loop saves every camera at the moment it asks "is the peg in the hole?"
(`trial_XXX_grade_*`: the descent stopped, fingers still closed; for a teacher's or policy's
let-go, fingers already open) and after a release (`trial_XXX_released_*`). This walks grasp-loop
row files and writes one CSV row per graded insertion, so a classifier or a VLM can be trained
and checked against the human grade before it is ever shown on the rollout page.

    python tools/fr3/grade_moment_dataset.py outputs/analysis/grasp_loop/grasp_*.jsonl -o grades.csv

Only rows a person graded are labelled ("in"/"out"); an early "out" (given during the search) is
kept with `early=1`, since its picture is of a landing the person gave up on, not a final stop.
Unattended rows (no answer) are written with an empty label, for shadow scoring only.

Which picture a judge may use: only `gradeImages` for a scripted insertion ("insert"/"rescue").
The judge's job there is to decide whether the fingers open, so it has to work from the picture
taken before they do (fingers closed round the peg, the hole half hidden). `releasedImages` exist
only because a person already said "in" and the fingers opened -- their presence alone gives the
label away, so they never go into training or the gate; they serve to re-check a person's label.
A let-go row ("let_go:*") is different: the policy or teacher already opened the fingers, and its
grade picture shows them open -- score it separately from the scripted insertions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLUMNS = (
    "run", "trial", "kind", "label", "early", "auto", "autoTrusted", "aboveTargetMm", "landing",
    "dfzPeakN", "gradeImages", "releasedImages",
)


def _rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """(trial rows, whether the run was attended -- only then is an answer a person's)."""

    out, attended = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("kind") == "trial":
            out.append(row)
        elif row.get("kind") == "run_start":
            attended = bool((row.get("request") or {}).get("attended"))
    return out, attended


def grade_records(path: Path) -> list[dict[str, Any]]:
    """One record per graded moment in a run's row file (scripted insertion, rescue, or let-go)."""

    records = []
    rows, attended = _rows(path)
    for row in rows:
        trial = int(row.get("trial", 0))
        for kind in ("insert", "rescue"):
            insert = row.get(kind)
            if not isinstance(insert, dict) or not insert.get("gradeSnapshots"):
                continue
            released = [p for p in insert.get("snapshots") or [] if "_released_" in Path(p).name]
            records.append({
                "run": path.stem, "trial": trial, "kind": kind,
                "label": insert.get("grade") if insert.get("grade") in ("in", "out") else "",
                "early": int(bool(insert.get("gradeEarly"))),
                "auto": insert.get("autoVerdict"), "autoTrusted": insert.get("autoTrusted"),
                "aboveTargetMm": insert.get("aboveTargetMm"), "landing": insert.get("searchTried"),
                "dfzPeakN": insert.get("dfzPeakN"),
                "gradeImages": ";".join(insert["gradeSnapshots"]), "releasedImages": ";".join(released),
            })
        if row.get("gradeSnapshots"):
            segment = row.get("insertSegment") or {}
            inserted = row.get("inserted")
            records.append({
                "run": path.stem, "trial": trial, "kind": f"let_go:{row.get('afterClose')}",
                "label": {True: "in", False: "out"}.get(inserted, "") if attended else "",
                "early": 0, "auto": segment.get("outcome"), "autoTrusted": "",
                "aboveTargetMm": segment.get("aboveSeatMm"), "landing": "", "dfzPeakN": "",
                "gradeImages": ";".join(row["gradeSnapshots"]), "releasedImages": "",
            })
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="grasp-loop row files (.jsonl)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [r for path in args.runs for r in grade_records(path)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    labelled = [r for r in records if r["label"]]
    print(
        f"grade moments {len(records)}: labelled {len(labelled)} "
        f"(in {sum(r['label'] == 'in' for r in labelled)}, out {sum(r['label'] == 'out' for r in labelled)}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
