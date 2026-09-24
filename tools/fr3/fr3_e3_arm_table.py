#!/usr/bin/env python
"""E3: the three-arm table, read from the rollout log and the traces it points at.

E3 asks one question -- does executing the medoid of N draws land the peg closer than
executing a single draw, and does averaging them instead give any of that back. The
pre-registered criterion (roadmap, E3 row) is:

    medoid vs N=1 : band EXIT error median falls by >= 3 mm
                    and band ENTRY error does not get worse
    falsifiable   : the three arms' exit errors are indistinguishable
                    => the terminal bottleneck is not "which draw got executed",
                       the sampling-aggregation line closes, and E8 form (i) goes with it.

THAT WORDING PREDATES THE TERMINAL SERVO AND NO LONGER READS WITH IT ON. The servo takes the
arm at z = `handoffZ`, whose default 0.12 is exactly the band's ceiling, and the runtime leaves
the rollout loop the instant it fires -- so the servo's descent is never sampled and the band
holds no policy frames at all. Measured on the 09-10 sessions: 18 of 19 rollouts have no band
segment, and the surviving traces end at z = 0.1201 .. 0.1208. Two readings therefore exist and
this prints both:

    HANDOFF ERROR   lateral error where the policy stopped driving. Always defined, and with
                    the servo on it is the whole of the policy's contribution -- E5 measured the
                    servo closing 15.8-61.1 mm of handoff error to 1.7-2.0 mm every time, so
                    nothing downstream of the handoff can tell the arms apart.
    BAND EXIT       the pre-registered wording. Needs the servo off for the policy to drive
                    through the band at all.

Which one E3 is run against is a decision about what E3 is for: the handoff error asks which
draw aims better, the band exit error asks which draw inserts better. They are not the same
question now that something else does the inserting.

Two files are needed and neither one is optional. `outputs/rollouts/rollout_log.jsonl` says
which arm produced each rollout and how the operator graded it; the per-step CSV under
`outputs/rollout_traces/<session>/rollout_<NNN>.csv` is the only place the band errors exist.
The log's `logPath` carries the session stamp, which is the join.

    python tools/fr3/fr3_e3_arm_table.py
    python tools/fr3/fr3_e3_arm_table.py --since 2026-09-20 --json outputs/analysis/e3.json
    python tools/fr3/fr3_e3_arm_table.py --cross-check outputs/analysis/p10/p10_e2.py

THE REDUCTION IS p10_e2's, DELIBERATELY. Band z in [0.08, 0.12), policy-controlled frames
only, segment running from the transport apex to the first expert frame or the release,
gripper edges taken with hysteresis, hole taken from the demonstrations' own release point.
Every band number in the roadmap was produced that way, so an E3 table computed any other way
could not be read beside them. `--cross-check` re-runs the archived script's segmentation over
the same sessions and asserts the two agree -- that is what keeps this copy honest, rather than
this docstring. Run it once before trusting a new table.

WHAT IT REFUSES. An arm is named by its aggregate and its draw count, and everything else about
a rollout is supposed to have been held fixed. `confounds` checks that instead of assuming it: the
window the draws were scored over, where that window started, the handoff z the error is measured
at, and the checkpoint. Any of those taking two values inside the compared window means the table
is not a comparison of draws, and the criteria print no verdict at all rather than a median that
looks readable. The search ring is audited too but never refused over -- it acts strictly after
the handoff, so it moves the success column and can reach none of the distances.

WHAT THIS DOES NOT DO. It does not grade, and it does not touch the log. `outcome` is the
operator's and is reported beside the band numbers, never merged into them: the trace cannot
tell a seated peg from a gripper that closed on air, and the criterion above is a distance, not
a success rate. At n=8 per arm a success rate on a 62% baseline separates nothing at all; the
exit error is a continuous quantity and is what the criterion was written against.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "outputs" / "rollouts" / "rollout_log.jsonl"
DEFAULT_TRACES = ROOT / "outputs" / "rollout_traces"
DEFAULT_VIEW = (
    ROOT / "outputs" / "exports" / "training_views" / "fr3_spacemouse-insert__delta_ee_from_prev_cmd"
)

# --- the shared segmentation contract (p10_e2.py). Changing any of these forks the caliper. ---
LO, HI = 0.08, 0.12
MIN_TRACE_ROWS = 40
GRASP_HOLD = 30          # frames the gripper must stay closed for a falling edge to be the grasp
RELEASE_HOLD = 10        # frames it must stay open for a rising edge to be the release
MIN_AUTONOMOUS = 40      # policy frames after the grasp, below which the rollout is the operator's
MIN_BAND_FRAMES = 8
GRIPPER_CLOSED = 0.5

SESSION_RE = re.compile(r"(\d{8}_\d{6})(?=\.log$|$)")


class E3Error(RuntimeError):
    pass


# ------------------------------------------------------------------ the reduction ---


def band(seg: np.ndarray, hole: np.ndarray) -> dict[str, float] | None:
    """Band statistics for one descent segment, or None if it barely entered the band."""
    mask = (seg[:, 2] >= LO) & (seg[:, 2] < HI)
    if mask.sum() < MIN_BAND_FRAMES:
        return None
    inside = np.where(mask)[0]
    s = seg[inside[0] : inside[-1] + 1]
    step = np.diff(s[:, :2], axis=0)
    path = float(np.hypot(step[:, 0], step[:, 1]).sum() * 1000.0)
    e0 = float(np.hypot(*(s[0, :2] - hole)) * 1000.0)
    e1 = float(np.hypot(*(s[-1, :2] - hole)) * 1000.0)
    still = float(np.mean(np.hypot(step[:, 0], step[:, 1]) * 1000.0 < 0.1))
    return {
        "n": int(len(s)),
        "path": path,
        "e0": e0,
        "e1": e1,
        "conv": e0 - e1,
        "eff": (e0 - e1) / path if path > 0 else 0.0,
        "still": still,
    }


def load_trace(path: Path) -> list[tuple[float, float, float, float, str]]:
    rows: list[tuple[float, float, float, float, str]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                        float(row["gripper_cmd"]),
                        row["source"],
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def reduce_trace(rows: list[tuple[float, float, float, float, str]], hole: np.ndarray):
    """(row, None) for a usable rollout, or (row, why) for one that cannot be read at all.

    The row always carries the segment's own columns -- where the policy stopped driving and how
    far out it was there -- and carries the band columns as well when the descent actually spent
    time in the band. Two quantities rather than one because since 2026-09-10 they are no longer
    the same question: the terminal servo takes the arm at z = `handoffZ` (0.12 by default),
    which is the band's ceiling, and the runtime returns from the rollout loop the instant it
    fires. The servo's descent is never sampled. So on a servo-era rollout the band is empty by
    construction and `endErr` -- the lateral error at the handoff -- is the only thing the policy
    can still be judged on.

    `reach` is the grasp point's distance to the hole -- the difficulty control from P1-10, kept
    even on a drop so a group that lost its hardest rollouts cannot look easy by accident.
    """
    row: dict[str, Any] = {}
    if len(rows) < MIN_TRACE_ROWS:
        return row, "trace too short"
    a = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=float)
    grip = a[:, 3]

    grasp = None
    for i in range(1, len(grip)):
        if (
            grip[i] < GRIPPER_CLOSED <= grip[i - 1]
            and all(x < GRIPPER_CLOSED for x in grip[i : i + GRASP_HOLD])
            and a[i, 2] < HI
        ):
            grasp = i
            break
    if grasp is None:
        return row, "no grasp below z=%.2f" % HI

    row["reach"] = float(np.hypot(*(a[grasp, :2] - hole)) * 1000.0)

    end = len(rows) - 1
    for i in range(grasp + GRASP_HOLD, len(rows)):
        if grip[i] >= GRIPPER_CLOSED and all(x >= GRIPPER_CLOSED for x in grip[i : i + RELEASE_HOLD]):
            end = i
            break

    stop = end
    for i in range(grasp, end + 1):
        if rows[i][4] != "policy":
            stop = i
            break
    if stop - grasp < MIN_AUTONOMOUS:
        return row, "operator took over %d steps after grasp" % (stop - grasp)

    seg = a[grasp + int(np.argmax(a[grasp : stop + 1, 2])) : stop + 1, :3]
    # Where the policy stopped driving the insert, and how far out it was there. On a servo-era
    # rollout this is the handoff; with the servo off it is the release. Either way it is the
    # last thing the policy is answerable for, and unlike the band it always exists.
    row["zApex"] = float(seg[0, 2])
    row["zEnd"] = float(seg[-1, 2])
    row["endErr"] = float(np.hypot(*(seg[-1, :2] - hole)) * 1000.0)
    row["segSteps"] = int(len(seg))
    # True when the policy let go of the descent above the band floor -- the signature of a
    # handoff to the servo, and the reason a band segment is missing.
    row["stoppedAbove"] = bool(rows[stop][4] == "policy" and seg[-1, 2] > LO)
    stats = band(seg, hole)
    if stats is None:
        row["bandWhy"] = "fewer than %d autonomous frames in the band" % MIN_BAND_FRAMES
    else:
        row.update(stats)
    return row, None


# ------------------------------------------------------------------ the hole ---


def hole_from_training_view(view: Path) -> tuple[np.ndarray, str]:
    """Mean of the demonstrations' own release points -- the reference every band table uses."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise E3Error(
            "reading the demonstrations needs pandas and pyarrow (%s); pass --hole X Y instead" % exc
        ) from exc
    files = sorted((view / "data").rglob("*.parquet"))
    if not files:
        raise E3Error("no parquet under %s/data" % view)
    frame = pd.concat([pq.read_table(p).to_pandas() for p in files], ignore_index=True)
    releases = []
    for _, group in frame.groupby("episode_index"):
        state = np.stack(group.sort_values("frame_index")["observation.state"].to_numpy())
        grip = state[:, 7]
        closed = opened = None
        for i in range(1, len(grip)):
            if closed is None and grip[i] < GRIPPER_CLOSED <= grip[i - 1]:
                closed = i
            elif closed is not None and opened is None and grip[i] >= GRIPPER_CLOSED > grip[i - 1]:
                opened = i
        if closed is None or opened is None:
            continue
        releases.append(state[opened, :3])
    if not releases:
        raise E3Error("no demonstration had both a grasp and a release in %s" % view)
    points = np.array(releases)
    return points[:, :2].mean(axis=0), "demo release mean, n=%d" % len(points)


def resolve_hole(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.hole is not None:
        return np.array(args.hole, dtype=float), "given on the command line"
    return hole_from_training_view(args.view)


# ------------------------------------------------------------------ the log ---


def arm_label(arm: dict[str, Any] | None) -> str:
    """The arm's name, or the one bucket that must never be pooled with a named one."""
    if not isinstance(arm, dict) or "actionSamples" not in arm:
        return "unrecorded"
    samples = int(arm["actionSamples"])
    if samples <= 1:
        return "N=1"
    return "%s x%d" % (arm.get("actionAggregate", "?"), samples)


def session_of(entry: dict[str, Any]) -> str | None:
    match = SESSION_RE.search(Path(str(entry.get("logPath") or "")).stem)
    return match.group(1) if match else None


def read_log(path: Path) -> list[dict[str, Any]]:
    """Every record in the log, with one allowance for reading it while it is being written.

    The log is append-only and this is meant to be run between rollouts, so the last line can be
    half-written at the moment it is read. That one is skipped with a warning -- it is the
    rollout that has not finished being recorded, and it will be there next time. A bad line
    anywhere else is corruption and refuses, because silently dropping a record from the middle
    of a comparison is how an arm loses a rollout without anyone noticing.
    """
    if not path.is_file():
        raise E3Error("no rollout log at %s" % path)
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines):
                print(
                    "WARNING: %s line %d is incomplete and was skipped -- a rollout is still "
                    "being written. Re-run once it has been graded." % (path, number),
                    file=sys.stderr,
                )
                continue
            raise E3Error("%s line %d is not JSON: %s" % (path, number, exc)) from exc
    return entries


def collect(args: argparse.Namespace, hole: np.ndarray) -> tuple[list[dict], list[dict]]:
    """One row per log entry, reduced against its trace. Returns (usable, dropped)."""
    usable: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for entry in read_log(args.log):
        recorded = str(entry.get("recordedAt") or "")
        if args.since and recorded < args.since:
            continue
        if args.until and recorded >= args.until:
            continue
        session = session_of(entry)
        index = int(entry.get("rolloutIndex") or 0)
        arm = entry.get("arm") if isinstance(entry.get("arm"), dict) else None
        label = arm_label(arm)
        if args.arms_only and label == "unrecorded":
            continue
        if args.session and session not in args.session:
            continue
        row = {
            "tag": "%s/%03d" % (session[-6:] if session else "??????", index),
            "session": session,
            "rolloutIndex": index,
            "recordedAt": recorded,
            "arm": label,
            # The whole block, not just the two fields the label is built from. `arm_label` has
            # to stay coarse or it would name every rollout uniquely and compare nothing; the
            # fields it drops are how two rollouts of "the same arm" differ, so they are kept
            # here and audited by `confounds` instead of being thrown away at the label.
            "armFields": dict(arm or {}),
            "outcome": str(entry.get("outcome") or ""),
            "stage": entry.get("stage"),
            "checkpointId": str(entry.get("checkpointId") or ""),
            "terminalServo": entry.get("terminalServo")
            if isinstance(entry.get("terminalServo"), dict)
            else {},
            "takeovers": len(entry.get("expertSpans") or []),
            "mismatch": entry.get("takeoverBlockerMismatch"),
        }
        if session is None:
            row["why"] = "logPath carries no session stamp"
            dropped.append(row)
            continue
        trace = args.traces / ("session_%s" % session) / ("rollout_%03d.csv" % index)
        if not trace.is_file():
            shown = trace.relative_to(ROOT) if trace.is_relative_to(ROOT) else trace
            row["why"] = "no trace at %s" % shown
            dropped.append(row)
            continue
        measured, why = reduce_trace(load_trace(trace), hole)
        row.update(measured)
        if why is not None:
            row["why"] = why
            dropped.append(row)
            continue
        usable.append(row)
    return usable, dropped


# ------------------------------------------------------------------ statistics ---


def median(rows: list[dict], key: str) -> float:
    return float(np.median([r[key] for r in rows])) if rows else math.nan


def mwu(a: list[float], b: list[float]) -> float:
    """Mann-Whitney p, or nan when either side is too thin to be asked."""
    if len(a) < 2 or len(b) < 2:
        return math.nan
    from scipy.stats import mannwhitneyu

    try:
        return float(mannwhitneyu(a, b).pvalue)
    except ValueError:
        return math.nan


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Spearman rho and p, or (nan, nan) when one side is constant and the rank is undefined."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan, math.nan
    from scipy.stats import spearmanr

    result = spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def num(value: float, spec: str = "%.1f") -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else spec % value


# ------------------------------------------------------------------ report ---


def order_arms(labels: set[str]) -> list[str]:
    """medoid, mean, N=1, then anything else -- the order the roadmap states the arms in."""
    def rank(label: str) -> tuple[int, str]:
        if label.startswith("medoid"):
            return (0, label)
        if label.startswith("mean"):
            return (1, label)
        if label == "N=1":
            return (2, label)
        if label == "unrecorded":
            return (4, label)
        return (3, label)

    return sorted(labels, key=rank)


def report(usable: list[dict], dropped: list[dict], hole: np.ndarray, source: str, args) -> None:
    print("hole reference : %.4f %.4f   (%s)" % (hole[0], hole[1], source))
    print("log            : %s" % args.log)
    print("traces         : %s" % args.traces)
    print(
        "window         : %s .. %s"
        % (args.since or "(start of log)", args.until or "(end of log)")
    )
    band_rows = [r for r in usable if "e1" in r]
    handed = [r for r in usable if r.get("stoppedAbove")]
    print(
        "rollouts       : %d usable, %d unreadable; %d have a band segment, %d stopped above the band floor"
        % (len(usable), len(dropped), len(band_rows), len(handed))
    )

    by_arm: dict[str, list[dict]] = {}
    for row in usable:
        by_arm.setdefault(row["arm"], []).append(row)
    labels = order_arms(set(by_arm))

    if handed and len(band_rows) < len(usable) / 2:
        print()
        print("  NOTE: the terminal servo takes the arm at z = %.2f, which is this band's ceiling," % HI)
        print("  and the runtime leaves the rollout loop the instant it fires -- the servo's own")
        print("  descent is never traced. On these rollouts the band is empty by construction and")
        print("  the BAND table below cannot be read. The HANDOFF table is the one that applies:")
        print("  where the policy stopped driving is the last thing the policy is answerable for.")

    print("\n=== PER ROLLOUT ===")
    print(
        "%-12s %-10s %-9s %7s %8s %7s %7s %8s %8s %6s %6s %8s %8s"
        % ("tag", "arm", "outcome", "z end", "end mm", "in mm", "out mm", "closed", "path",
           "steps", "eff", "reach", "rig")
    )
    for row in sorted(usable, key=lambda r: (r["recordedAt"], r["rolloutIndex"])):
        print(
            "%-12s %-10s %-9s %7.4f %8.1f %7s %7s %8s %8s %6s %6s %8.1f %8s"
            % (
                row["tag"],
                row["arm"],
                row["outcome"],
                row["zEnd"],
                row["endErr"],
                num(row.get("e0")),
                num(row.get("e1")),
                num(row.get("conv"), "%+.1f"),
                num(row.get("path")),
                num(row.get("n"), "%.0f"),
                num(row.get("eff"), "%.2f"),
                row["reach"],
                row["terminalServo"].get("verdict", "-"),
            )
        )

    if dropped:
        print("\n=== UNREADABLE (kept visible: a table computed on the rollouts the operator")
        print("    left alone is not the same population as one computed on all of them) ===")
        for row in sorted(dropped, key=lambda r: (r["recordedAt"], r["rolloutIndex"])):
            print(
                "  %-12s %-10s %-52s reach=%s"
                % (row["tag"], row["arm"], row["why"], num(row.get("reach"), "%.0f mm"))
            )

    print("\n=== HANDOFF TABLE   lateral error where the policy stopped driving ===")
    print(
        "%-10s %4s %10s %10s %9s %9s %9s"
        % ("arm", "n", "end err mm", "z end", "seg steps", "reach mm", "success")
    )
    for label in labels:
        rows = by_arm[label]
        wins = sum(1 for r in rows if r["outcome"] == "success")
        print(
            "%-10s %4d %10s %10s %9s %9s %4d/%-4d"
            % (
                label,
                len(rows),
                num(median(rows, "endErr")),
                num(median(rows, "zEnd"), "%.4f"),
                num(median(rows, "segSteps"), "%.0f"),
                num(median(rows, "reach")),
                wins,
                len(rows),
            )
        )

    if band_rows:
        band_by_arm: dict[str, list[dict]] = {}
        for row in band_rows:
            band_by_arm.setdefault(row["arm"], []).append(row)
        print("\n=== BAND TABLE   z in [%.2f, %.2f), policy-controlled frames only ===" % (LO, HI))
        print(
            "%-10s %4s %8s %8s %9s %9s %7s %7s %7s %9s"
            % ("arm", "n", "in mm", "out mm", "closed", "path", "steps", "eff", "still%", "reach mm")
        )
        for label in order_arms(set(band_by_arm)):
            rows = band_by_arm[label]
            print(
                "%-10s %4d %8s %8s %9s %9s %7s %7s %6s%% %9s"
                % (
                    label,
                    len(rows),
                    num(median(rows, "e0")),
                    num(median(rows, "e1")),
                    num(median(rows, "conv"), "%+.1f"),
                    num(median(rows, "path")),
                    num(median(rows, "n"), "%.0f"),
                    num(median(rows, "eff"), "%.2f"),
                    num(100 * median(rows, "still"), "%.0f"),
                    num(median(rows, "reach")),
                )
            )
    print("\n    success is reported, not tested: at n=8 on a 62% baseline it separates nothing.")
    print("    the criteria below are distances, which is what they were written against.")

    blocked = confounds(usable)
    criteria(by_arm, labels, blocked)
    interleaving(usable, labels)
    difficulty(usable)
    terminal_servo(by_arm, labels)
    agreement(usable)


def _pair(by_arm: dict[str, list[dict]], labels: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    medoid = next((by_arm[l] for l in labels if l.startswith("medoid")), [])
    mean = next((by_arm[l] for l in labels if l.startswith("mean")), [])
    return medoid, mean, by_arm.get("N=1", [])


def _criterion(name: str, key: str, medoid: list[dict], mean: list[dict], single: list[dict]) -> None:
    rows = lambda group: [r[key] for r in group if key in r]
    m, s1 = rows(medoid), rows(single)
    print("  -- %s" % name)
    if len(m) < 2 or len(s1) < 2:
        print("     medoid n=%d, N=1 n=%d -> NOT READABLE" % (len(m), len(s1)))
        return
    drop = float(np.median(s1)) - float(np.median(m))
    print(
        "     medoid %.1f vs N=1 %.1f  ->  falls by %+.1f mm   criterion >= 3.0 mm  -> %s   p=%s"
        % (np.median(m), np.median(s1), drop, "PASS" if drop >= 3.0 else "FAIL", num(mwu(m, s1), "%.4f"))
    )
    mn = rows(mean)
    if len(mn) >= 2:
        print(
            "     mean %.1f vs medoid %.1f  (%+.1f mm)  p=%s"
            % (np.median(mn), np.median(m), np.median(mn) - np.median(m), num(mwu(mn, m), "%.4f"))
        )
    groups = [g for g in (m, mn, s1) if len(g) >= 2]
    if len(groups) >= 3:
        from scipy.stats import kruskal

        try:
            p = float(kruskal(*groups).pvalue)
        except ValueError:
            p = math.nan
        print("     three arms, Kruskal-Wallis p=%s" % num(p, "%.4f"))
        if not math.isnan(p) and p >= 0.05:
            print("     -> indistinguishable. The terminal bottleneck is NOT which draw got executed.")
            print("        Sampling aggregation closes, and E8 form (i) closes with it. Go to E4/E5.")


def criteria(by_arm: dict[str, list[dict]], labels: list[str], blocked: list[str]) -> None:
    print("\n=== E3 CRITERIA ===")
    if blocked:
        # Printing the verdict anyway with a warning above it would be the weaker choice: the
        # number looks exactly like a readable one, and a reader who scrolled past the audit has
        # no way to tell. The rollouts are not lost -- narrowing the window to a stretch where
        # these agree gives a smaller but readable comparison.
        print("  NO VERDICT: %s differ inside this window." % ", ".join(blocked))
        print("  Each of them can move the distance being compared, so a median taken across them")
        print("  would look like an arm difference without being one. Narrow with --since/--until")
        print("  or --session until they agree, or re-run the rollouts that differ.")
        return
    medoid, mean, single = _pair(by_arm, labels)

    _criterion(
        "HANDOFF ERROR -- where the policy stopped driving (readable with the servo on)",
        "endErr",
        medoid,
        mean,
        single,
    )
    _criterion(
        "BAND EXIT ERROR -- the roadmap's pre-registered wording (needs the servo OFF)",
        "e1",
        medoid,
        mean,
        single,
    )

    entry_m = [r["e0"] for r in medoid if "e0" in r]
    entry_s = [r["e0"] for r in single if "e0" in r]
    if len(entry_m) >= 2 and len(entry_s) >= 2:
        worse = float(np.median(entry_m)) - float(np.median(entry_s))
        p_entry = mwu(entry_m, entry_s)
        if worse <= 0:
            guard = "PASS (no worse)"
        elif math.isnan(p_entry) or p_entry >= 0.05:
            guard = "PASS (worse by %.1f mm but p=%s)" % (worse, num(p_entry, "%.3f"))
        else:
            guard = "FAIL (worse by %.1f mm, p=%s)" % (worse, num(p_entry, "%.3f"))
        print("  -- GUARD: band entry error must not get worse")
        print("     medoid %.1f vs N=1 %.1f -> %s" % (np.median(entry_m), np.median(entry_s), guard))


# The log's `arm` block is policyArm merged with the terminal servo's configuration
# (`gateway.py::_record_rollout_outcome`), so it already carries everything that decided what a
# rollout was. Whether a field belongs in the label or in this audit is not a matter of taste: it
# is whether the field can move the distance being compared.
#
# These can. A window of a different length, or starting at a different step, scores a different
# stretch of the chunk; a different handoff z moves the very point the handoff error is measured
# at; a different checkpoint is a different policy. A comparison spanning two values of any of
# them is not a comparison of draws, whatever the arm labels say.
BLOCKING_FIELDS = (
    ("selectionHorizon", "steps of each draw the selection scored"),
    ("selectionOffsetSteps", "where that window started; absent = derived from latency"),
    ("terminalServoHandoffZ", "the z at which the handoff error is measured"),
)
# These cannot reach a distance measured at the handoff, because the search happens strictly
# after it -- E5's own argument for why opening the ring does not pollute E3. They do move the
# success column, so they are printed next to it and never refused over.
NOTING_FIELDS = (
    ("terminalServoSearchRingM", "search radius; moves the success column, not the distances"),
    ("terminalServoSearchLandings", "landings; 1 means the search was off"),
    ("terminalServoXyz", "the nominal pose the descent drove to"),
)


def _shown(value: Any) -> str:
    """One field of one rollout, as a value a reader can compare by eye. Absent is not zero."""
    if value is None:
        return "(absent)"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "%g" % value
    if isinstance(value, (list, tuple)):
        return ",".join(_shown(item) for item in value)
    return str(value)


def confounds(usable: list[dict]) -> list[str]:
    """What else changed while the arms were being compared. Returns the fields that block a verdict.

    E3's arms are named by aggregate and draw count, and everything else about a rollout is
    supposed to be held fixed by the operator. Nothing checked that. Two rollouts run a day apart
    with a re-tuned handoff or a re-resolved selection window pooled into one bucket and came out
    as an arm difference -- the same shape as the mistake that made the band criterion unreadable,
    found after the rollouts rather than before them. The cost of finding it late is total: the
    arms have to be interleaved, so a confound discovered afterwards cannot be sliced back out.
    """
    rows = [r for r in usable if r["arm"] != "unrecorded"]
    print("\n=== CONFOUND AUDIT ===")
    if len(rows) < 2:
        print("  fewer than two arm-recorded rollouts; nothing to hold constant.")
        return []

    blocking: list[str] = []
    checked: list[tuple[str, str, object]] = [
        (key, why, lambda r, k=key: r["armFields"].get(k)) for key, why in BLOCKING_FIELDS
    ]
    checked.append(("checkpointId", "the policy under test", lambda r: r["checkpointId"] or None))
    for key, why, read in checked:
        counts = Counter(_shown(read(r)) for r in rows)
        spread = "  ".join("%s x%d" % (value, n) for value, n in sorted(counts.items()))
        if len(counts) > 1:
            blocking.append(key)
            print("  *** %-24s VARIES: %s" % (key, spread))
            print("      %s" % why)
        else:
            print("  %-28s %s   (%s)" % (key, spread, why))

    for key, why in NOTING_FIELDS:
        counts = Counter(_shown(r["armFields"].get(key)) for r in rows)
        spread = "  ".join("%s x%d" % (value, n) for value, n in sorted(counts.items()))
        mark = "note" if len(counts) > 1 else "    "
        print("  %s %-23s %s   (%s)" % (mark, key, spread, why))
    if any(len(Counter(_shown(r["armFields"].get(k)) for r in rows)) > 1 for k, _ in NOTING_FIELDS):
        print("  the noted fields change the success column and cannot change the distances above;")
        print("  a success rate read across them is not comparable with one read at a single setting.")
    return blocking


def interleaving(usable: list[dict], labels: list[str]) -> None:
    """The fixture creeps within a session (r=+0.64, p=0.002). Blocked arms are confounded with it."""
    print("\n=== INTERLEAVING AUDIT ===")
    named = [r for r in usable if r["arm"] != "unrecorded"]
    if len(named) < 6:
        print("  too few rollouts to audit.")
        return
    order = sorted(named, key=lambda r: (r["recordedAt"], r["rolloutIndex"]))
    position = {id(row): index for index, row in enumerate(order)}
    groups = [[position[id(r)] for r in order if r["arm"] == l] for l in labels if l != "unrecorded"]
    groups = [g for g in groups if len(g) >= 2]
    print("  run order: %s" % " ".join(r["arm"].split()[0][:3] for r in order))
    if len(groups) >= 2:
        from scipy.stats import kruskal

        try:
            p = float(kruskal(*groups).pvalue)
        except ValueError:
            p = math.nan
        print("  arm vs position in the run, Kruskal-Wallis p=%s" % num(p, "%.4f"))
        if not math.isnan(p) and p < 0.05:
            print("  *** ARMS ARE BLOCKED IN TIME, NOT INTERLEAVED. The fixture creeps within a")
            print("  *** session, so arm and drift cannot be told apart. This table is NOT readable")
            print("  *** as an arm comparison. Re-run interleaved.")
        else:
            print("  -> no detectable block structure; arm and run position are separable.")
    rho, p = spearman([position[id(r)] for r in order], [r["endErr"] for r in order])
    print("  creep probe: Spearman(position, handoff error) rho=%s p=%s n=%d"
          % (num(rho, "%+.2f"), num(p, "%.3f"), len(order)))


def difficulty(usable: list[dict]) -> None:
    """P1-10: landing error scales with how far the peg started from the hole."""
    rows = [r for r in usable if r["arm"] != "unrecorded"]
    if len(rows) < 4:
        return
    rho, p = spearman([r["reach"] for r in rows], [r["endErr"] for r in rows])
    print("\n=== DIFFICULTY CONTROL ===")
    print("  Spearman(grasp reach, handoff error) rho=%s p=%s n=%d"
          % (num(rho, "%+.2f"), num(p, "%.3f"), len(rows)))
    print("  reach medians are in the tables above; an arm that drew easier pins would show a")
    print("  better error without the policy having done anything differently.")


def terminal_servo(by_arm: dict[str, list[dict]], labels: list[str]) -> None:
    """E7-C comes free: every rollout that reached the servo searched, and said how it ended."""
    print("\n=== E7-C, FREE READOUT (search_for_seat) ===")
    any_seen = False
    for label in labels:
        rows = [r for r in by_arm[label] if r["terminalServo"]]
        if not rows:
            continue
        any_seen = True
        verdicts = Counter(r["terminalServo"].get("verdict", "-") for r in rows)
        landings = [
            r["terminalServo"]["searchIndex"]
            for r in rows
            if isinstance(r["terminalServo"].get("searchIndex"), int)
        ]
        print(
            "  %-10s n=%2d  %s%s"
            % (
                label,
                len(rows),
                " ".join("%s=%d" % kv for kv in sorted(verdicts.items())),
                ("   landing index median=%.0f max=%d" % (np.median(landings), max(landings)))
                if landings
                else "",
            )
        )
    if not any_seen:
        print("  no rollout in this window carried a terminalServo block.")
    else:
        print("  landing index 0 means the nominal pose seated it and the ring was never walked.")
        print("  the per-attempt verdicts are in the runtime log, not here: the rollout log keeps")
        print("  the descent's summary columns, not searchAttempts.")


def agreement(usable: list[dict]) -> None:
    """The rig's reading against the operator's grade. Never reconciled -- the gap is the number."""
    rows = [r for r in usable if r["terminalServo"].get("verdict") and r["outcome"] in ("success", "failure")]
    print("\n=== RIG vs OPERATOR ===")
    if not rows:
        print("  nothing comparable in this window.")
        return
    agree = sum(
        1
        for r in rows
        if (r["terminalServo"]["verdict"] == "seated") == (r["outcome"] == "success")
    )
    print("  %d/%d agree (%.0f%%)" % (agree, len(rows), 100.0 * agree / len(rows)))
    for row in rows:
        if (row["terminalServo"]["verdict"] == "seated") != (row["outcome"] == "success"):
            print("    disagreed: %s rig=%s operator=%s"
                  % (row["tag"], row["terminalServo"]["verdict"], row["outcome"]))
    print("  this is the acceptance number for every unattended loop on this rig; a single")
    print("  reconciled column could not report it, which is why both are stored.")


# ------------------------------------------------------------------ cross-check ---


def cross_check(path: Path, sessions: list[str]) -> int:
    """Re-run the archived reduction and assert this copy agrees with it, rollout for rollout."""
    if not path.is_file():
        raise E3Error("no archived reduction at %s" % path)
    source = path.read_text(encoding="utf-8")
    head = source.split("GROUPS = [")[0]
    namespace: dict[str, Any] = {"__name__": "p10_e2_preamble"}
    exec(compile(head, str(path), "exec"), namespace)  # noqa: S102 -- the archived contract
    their_hole = np.asarray(namespace["HOLE"], dtype=float)
    their_segs = namespace["rollout_segs"]
    traces = Path(namespace["TR"])

    sessions = [s if s.startswith("session_") else "session_%s" % s for s in sessions]
    if not sessions:
        sessions = sorted(p.name for p in traces.iterdir() if p.is_dir())
    print("archived reduction : %s" % path)
    print("hole               : %.4f %.4f" % (their_hole[0], their_hole[1]))
    print("sessions           : %d" % len(sessions))

    theirs, _ = their_segs(sessions)
    by_tag = {r["tag"]: r for r in theirs}

    mine: dict[str, dict] = {}
    for session in sessions:
        for trace in sorted((traces / session).glob("rollout_*.csv")):
            row, why = reduce_trace(load_trace(trace), their_hole)
            # The archived reduction keeps only rollouts that produced a band segment, so the
            # comparison is over exactly those: a rollout this one keeps for its handoff error
            # alone is not something p10_e2 ever had an opinion about.
            if why is None and "e1" in row:
                mine["%s/%s" % (session[-6:], trace.stem[-3:])] = row

    only_theirs = sorted(set(by_tag) - set(mine))
    only_mine = sorted(set(mine) - set(by_tag))
    worst = 0.0
    worst_tag = "-"
    for tag in sorted(set(by_tag) & set(mine)):
        for key in ("e0", "e1", "path", "conv", "eff"):
            delta = abs(float(by_tag[tag][key]) - float(mine[tag][key]))
            if delta > worst:
                worst, worst_tag = delta, "%s.%s" % (tag, key)

    print("kept by both       : %d" % len(set(by_tag) & set(mine)))
    print("kept only by p10   : %s" % (", ".join(only_theirs) or "none"))
    print("kept only by this  : %s" % (", ".join(only_mine) or "none"))
    print("largest difference : %.3e  (%s)" % (worst, worst_tag))
    ok = not only_theirs and not only_mine and worst < 1e-9
    print("VERDICT            : %s" % ("IDENTICAL" if ok else "DIVERGED -- do not read the E3 table"))
    return 0 if ok else 1


# ------------------------------------------------------------------ main ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW,
                        help="training view the hole reference is averaged from")
    parser.add_argument("--hole", type=float, nargs=2, metavar=("X", "Y"),
                        help="override the hole reference instead of recomputing it")
    parser.add_argument("--since", help="ISO instant; keep records recorded at or after it")
    parser.add_argument("--until", help="ISO instant; keep records recorded before it")
    parser.add_argument("--session", action="append", help="restrict to this session stamp (repeatable)")
    parser.add_argument("--all-records", dest="arms_only", action="store_false",
                        help="include records with no arm block (pre-2026-09-20); they are pooled "
                             "nowhere and only appear in the drop list")
    parser.add_argument("--json", type=Path, help="write the per-rollout rows here")
    parser.add_argument("--cross-check", type=Path, metavar="P10_E2",
                        help="re-run the archived reduction and assert this copy agrees")
    parser.set_defaults(arms_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cross_check:
            return cross_check(args.cross_check, args.session or [])
        hole, source = resolve_hole(args)
        usable, dropped = collect(args, hole)
        report(usable, dropped, hole, source, args)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
                "hole": [float(hole[0]), float(hole[1])],
                "holeSource": source,
                "band": [LO, HI],
                "usable": usable,
                "dropped": dropped,
            }
            args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\nwrote %s" % args.json)
    except E3Error as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
