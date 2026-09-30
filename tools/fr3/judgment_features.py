"""Deterministic features for the judge, and the rules that settle what the record already decides.

Two record streams exist, and both are read as they are -- nothing here changes how they are
written:

* `outputs/analysis/grasp_loop/grasp_*.jsonl` -- one `trial` row per grasp-loop trial (the rig's
  grasp verdict from the lifted width, the terminal servo's insertion reading, the funnel's
  handoff), plus `halt` events for trials that died before a row was written.
* `outputs/rollouts/rollout_log.jsonl` -- one line per GUI rollout: the operator's grade
  (outcome, stage, blockers), the takeover spans, the runtime's geometry and servo reading.

Every feature is something a machine measured. What a person said -- the operator's grade and
blockers, an insertion graded in/out, a note -- goes to `labels`, which the judge never sends to
a provider; it is what the provider is benchmarked against. A value that was not measured is
None and named in `missing`. Two references are deliberately not treated as ground truth: the
hole's position is the configured nominal one and it creeps, so a lateral error "to the hole" is
to that nominal point and is named so; and a peg position is only used when the scene reset
staged it there (grasp loop `pegSource == "gt"`, rollout log `resetTarget`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

from tools.fr3.judgment import FeatureSnapshot

# Never recorded in either stream today; listed so a provider is told they are absent rather
# than left to assume they were fine.
NOT_RECORDED = ("policy_action_stats", "ft_time_series_summary")

GRASP_FAILED = ("empty", "no_close", "collision")


def _num(value: Any, ndigits: int = 1) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, ndigits) if math.isfinite(number) else None


def _snapshot(
    episode_id: str,
    features: dict[str, Any],
    *,
    source: str,
    labels: dict[str, Any],
    expert_span_id: str | None = None,
    not_recorded: Iterable[str] = NOT_RECORDED,
) -> FeatureSnapshot:
    missing = sorted({key for key, value in features.items() if value is None} | set(not_recorded))
    return FeatureSnapshot(
        episode_id=episode_id,
        expert_span_id=expert_span_id,
        features={k: v for k, v in features.items() if v is not None},
        missing=tuple(missing),
        source=source,
        labels={k: v for k, v in labels.items() if v not in (None, "", [])},
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


# ------------------------------------------------------------------------------- grasp loop ---


def grasp_trial_outcome(row: dict[str, Any]) -> tuple[str, str]:
    """(outcome, who decided it). The insertion's `inserted` is the operator's in/out on an
    attended run and the rig's release on an unattended one; which one it was is returned."""

    verdict = row.get("verdict")
    if verdict == "voided":
        return "voided", "rig"
    if verdict in GRASP_FAILED:
        return "grasp_failed", "rig"
    insert = row.get("insert")
    if not isinstance(insert, dict):
        return ("held" if verdict == "held" else str(verdict)), "rig"
    source = "human" if insert.get("grade") in ("in", "out") else "rig"
    if row.get("inserted") is True:
        return "inserted", source
    if row.get("inserted") is False:
        return "insert_failed", source
    return "insert_ungraded", source


def grasp_trial_snapshot(row: dict[str, Any], *, run: str, attended: bool | None, source: str) -> FeatureSnapshot:
    trial = int(row.get("trial", -1))
    outcome, outcome_source = grasp_trial_outcome(row)
    peg_gt = row.get("pegSource") == "gt"
    aim = row.get("aimErrorMm") if peg_gt and isinstance(row.get("aimErrorMm"), dict) else {}
    features: dict[str, Any] = {
        "record": "grasp_loop_trial",
        "arm": row.get("arm"),  # A: the policy grasps alone; B: it searches, the funnel closes
        "staging": row.get("staging"),
        "attended": attended,
        "outcome": outcome,
        "outcome_source": outcome_source,
        "peg_start_is_staged_gt": peg_gt,
        "hole_pose_is_gt": False,
        "grasp_verdict": row.get("verdict"),
        "policy_status": row.get("policyStatus"),
        "handover_step": row.get("handoverStep"),
        "close_step": row.get("closeStep"),
        "blocked_steps": row.get("blockedSteps"),
        "premature_descend": row.get("prematureDescend"),
        "realigns": row.get("realigns"),
        "max_residual_mm": _num(row.get("maxResidualMm")),
        "gripper_cmd_at_close": _num(row.get("commandedGripper"), 3),
        "gripper_width_at_close": _num(row.get("widthAtClose"), 3),
        "gripper_width_lifted": _num(row.get("widthLifted"), 3),
        "funnel_entry_reason": row.get("funnelEntryReason") or None,
        "funnel_state_at_end": row.get("funnelState"),
        "funnel_takeover_dz_mm": _num(row.get("funnelAlignDzMm")),
        "trial_s": _num(row.get("trialS")),
    }
    if peg_gt:
        # Against where the scene reset put the peg -- the only peg reference that is not a guess.
        features.update(
            {
                "close_xy_to_peg_mm": _num(row.get("lateralMm")),
                "close_z_above_peg_mm": _num(row.get("closeAboveTargetMm")),
                "lowest_near_peg_mm": _num(row.get("lowestNearPegMm")),
                "funnel_xy_error_at_entry_mm": _num(row.get("xyErrorAtEntryMm")),
                "funnel_xy_error_at_close_mm": _num(row.get("xyErrorAtCloseMm")),
                "policy_aim_error_150mm": _num(aim.get("150")),
                "policy_aim_error_100mm": _num(aim.get("100")),
                "policy_aim_error_60mm": _num(aim.get("60")),
            }
        )
    # Written only on the trials they describe (a miss; a re-grip), so absent is not missing.
    if peg_gt and "pegUntouched" in row:
        features["peg_untouched"] = row["pegUntouched"]
    if "regripWidth" in row:
        features["gripper_regrip_width"] = _num(row["regripWidth"], 3)
    insert = row.get("insert")
    features["insert_reached"] = isinstance(insert, dict)
    if isinstance(insert, dict):
        features.update(
            {
                "servo_auto_verdict": insert.get("autoVerdict"),
                "servo_stopped_on": insert.get("stoppedOn"),
                "servo_search_stopped_on": insert.get("searchStoppedOn"),
                "servo_above_target_mm": _num(insert.get("aboveTargetMm")),
                # To the configured hole position, which creeps: a reading, not a residual.
                "servo_lateral_to_nominal_hole_mm": _num(insert.get("lateralErrorMm")),
                "servo_search_landings": insert.get("searchTried"),
                "ft_dfz_peak_n": _num(insert.get("dfzPeakN")),
                "ft_press_capped": insert.get("pressCapped"),
                "tool_tilt_deg": _num(insert.get("toolTiltDeg"), 2),
                "released_in_hole": insert.get("released"),
                "raised_mm": _num(insert.get("raisedMm")),
            }
        )
    labels = {"insert_grade": (insert or {}).get("grade") if isinstance(insert, dict) else None}
    return _snapshot(f"grasp:{run}:t{trial:03d}", features, source=source, labels=labels)


def halt_snapshot(event: dict[str, Any], *, run: str, source: str) -> FeatureSnapshot:
    trial = int(event.get("trial", -1))
    features = {
        "record": "grasp_loop_halt",
        "outcome": "halted",
        "outcome_source": "rig",
        "halt_reason": event.get("reason"),
        "halt_error": str(event.get("error") or "")[:300] or None,
    }
    return _snapshot(f"grasp:{run}:t{trial:03d}:halt", features, source=source, labels={}, not_recorded=())


def load_grasp_loop_run(path: str | Path) -> list[FeatureSnapshot]:
    """Every trial of one run, plus a halt that stopped a trial before its row was written."""

    path = Path(path)
    rows = list(_read_jsonl(path))
    start = next((r for r in rows if r.get("kind") == "run_start"), {})
    attended = (start.get("request") or {}).get("attended") if start else None
    trials = [r for r in rows if r.get("kind") == "trial"]
    seen = {int(r.get("trial", -1)) for r in trials}
    snaps = [grasp_trial_snapshot(r, run=path.stem, attended=attended, source=str(path)) for r in trials]
    snaps += [
        halt_snapshot(e, run=path.stem, source=str(path))
        for e in rows
        if e.get("kind") == "halt" and int(e.get("trial", -1)) not in seen
    ]
    return snaps


# ------------------------------------------------------------------------------ rollout log ---


def rollout_episode_id(entry: dict[str, Any]) -> str:
    return f"rollout:{entry.get('recordedAt')}:{entry.get('rolloutIndex', '-')}"


def rollout_features(entry: dict[str, Any]) -> dict[str, Any]:
    geometry = entry.get("geometry") if isinstance(entry.get("geometry"), dict) else {}
    servo = entry.get("terminalServo") if isinstance(entry.get("terminalServo"), dict) else {}
    arm = entry.get("arm") if isinstance(entry.get("arm"), dict) else {}
    steps = int(entry.get("steps") or 0)
    spans = entry.get("expertSpans") or []
    reset = entry.get("resetTarget")
    grasp = geometry.get("graspXyz")
    features: dict[str, Any] = {
        "record": "rollout_log",
        "mode": entry.get("mode") or None,
        # The rollout log's outcome is the operator's grade; there is no rig verdict beside it.
        "outcome": entry.get("outcome"),
        "outcome_source": "human",
        "steps": steps or None,
        "intervened": entry.get("intervened"),
        "expert_steps": entry.get("expertSteps"),
        "expert_fraction": _num(entry.get("expertSteps", 0) / steps, 3) if steps and "expertSteps" in entry else None,
        "takeover_count": len(spans) if "intervened" in entry else None,
        "gripper_closed": geometry.get("closed"),
        "held_steps": geometry.get("heldSteps"),
        "grasp_by": geometry.get("graspBy"),
        "release_by": geometry.get("releaseBy"),
        "lift_m": _num(geometry.get("liftM"), 4),
        "descent_m": _num(geometry.get("descentM"), 4),
        "apex_z": _num(geometry.get("apexZ"), 4),
        "peg_start_is_staged_gt": isinstance(reset, list) and len(reset) == 3,
        "hole_pose_is_gt": False,
        "action_aggregate": arm.get("actionAggregate"),
        "servo_handoff_z": _num(arm.get("terminalServoHandoffZ"), 3),
    }
    if spans:
        # Only when there was a takeover: on a rollout without one this is not missing, it is moot.
        features["first_takeover_at_frac"] = _num(spans[0][0] / steps, 3) if steps else None
    if features["peg_start_is_staged_gt"] and isinstance(grasp, list) and len(grasp) == 3:
        features["grasp_xy_to_staged_peg_mm"] = _num(math.dist(grasp[:2], reset[:2]) * 1000.0)
        features["grasp_z_above_staged_peg_mm"] = _num((grasp[2] - reset[2]) * 1000.0)
    features["servo_reached"] = bool(servo)
    if servo:
        features.update(
            {
                "servo_auto_verdict": servo.get("verdict"),
                "servo_stopped_on": servo.get("stoppedOn"),
                "servo_search_stopped_on": servo.get("searchStoppedOn"),
                "servo_above_target_mm": _num(servo.get("aboveTargetMm")),
                "servo_lateral_to_nominal_hole_mm": _num(servo.get("lateralErrorMm")),
                "servo_lag_mm": _num(servo.get("lagMm")),
                "servo_settle_mm": _num(servo.get("settleMm")),
                "servo_held_up_growth_mm": _num(servo.get("heldUpGrowthMm")),
                "servo_descent_s": _num(servo.get("descentSeconds"), 2),
                "servo_search_landings": servo.get("searchLandings"),
            }
        )
    return features


def rollout_labels(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": entry.get("outcome"),
        "stage_id": entry.get("stageId"),
        "blocker": entry.get("blocker"),
        "blockers": entry.get("blockers"),
        "note": entry.get("note"),
    }


def rollout_snapshots(entry: dict[str, Any], *, source: str) -> list[FeatureSnapshot]:
    """The episode, then one snapshot per takeover span (for the span-scoped tasks)."""

    episode_id = rollout_episode_id(entry)
    features = rollout_features(entry)
    labels = rollout_labels(entry)
    snaps = [_snapshot(episode_id, features, source=source, labels=labels)]
    steps = int(entry.get("steps") or 0)
    for index, span in enumerate(entry.get("expertSpans") or []):
        first, last = int(span[0]), int(span[1])
        span_features = {
            **features,
            "span_index": index,
            "span_first_step": first,
            "span_steps": last - first + 1,
            "span_start_frac": _num(first / steps, 3) if steps else None,
            "steps_after_span": steps - last - 1 if steps else None,
            # The rollout log keeps no per-span pose or policy status (takeover details were
            # never populated): named so the provider knows, rather than guessed.
            "span_policy_status": None,
            "span_pose": None,
        }
        snaps.append(
            _snapshot(episode_id, span_features, source=source, labels=labels, expert_span_id=f"{episode_id}/span{index}")
        )
    return snaps


def load_rollout_log(path: str | Path) -> list[FeatureSnapshot]:
    path = Path(path)
    snaps: list[FeatureSnapshot] = []
    for entry in _read_jsonl(path):
        snaps += rollout_snapshots(entry, source=str(path))
    return snaps


# ------------------------------------------------------------------------ applicability/rules ---

FAILED_OUTCOMES = ("grasp_failed", "insert_failed", "voided", "halted", "failure", "aborted")


def is_failure(snapshot: FeatureSnapshot) -> bool:
    return snapshot.features.get("outcome") in FAILED_OUTCOMES


def failure_reason_rule(snapshot: FeatureSnapshot) -> str | None:
    """Only where the record names the reason itself. Everything else is the provider's."""

    f = snapshot.features
    if f.get("record") == "grasp_loop_halt":
        return "RUNTIME_ERROR"
    if f.get("outcome") in ("voided", "aborted"):
        return "ABORTED"
    if f.get("grasp_verdict") == "no_close":
        return "POLICY_STALL"  # the policy used its whole step budget without a settled close
    if f.get("grasp_verdict") == "empty" and f.get("peg_untouched") is True:
        return "EMPTY_GRASP"  # lifted width says empty, and the tool never came low near the peg
    return None


def failure_side_rule(snapshot: FeatureSnapshot) -> str | None:
    f = snapshot.features
    if f.get("record") == "grasp_loop_halt":
        return "RUNTIME"
    if f.get("outcome") in ("voided", "aborted"):
        return "ENVIRONMENT"
    if f.get("grasp_verdict") == "no_close":
        return "POLICY"  # nothing scripted had started: the handover never fired
    if f.get("grasp_verdict") == "empty" and f.get("arm") == "A":
        return "POLICY"  # arm A: the policy's own close, graded by a scripted lift
    return None


RULES = {"failure_reason": failure_reason_rule, "policy_vs_runtime_failure": failure_side_rule}
APPLIES = {"failure_reason": is_failure, "policy_vs_runtime_failure": is_failure}
