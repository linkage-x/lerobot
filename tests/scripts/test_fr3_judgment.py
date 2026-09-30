from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import pytest

from tools.fr3 import fr3_judge_rollouts as cli
from tools.fr3.judgment import (
    AMBIGUOUS,
    SUPERVISORY_ACTIONS,
    TASKS,
    UNKNOWN,
    DecisionLog,
    FeatureSnapshot,
    JevProvider,
    JudgeConfig,
    MockProvider,
    RolloutJudge,
    decision_record,
    judge_from_env,
)
from tools.fr3.judgment_features import (
    APPLIES,
    RULES,
    grasp_trial_snapshot,
    load_grasp_loop_run,
    load_rollout_log,
)

FAILED = FeatureSnapshot("ep1", {"record": "grasp_loop_trial", "outcome": "insert_failed"}, source="t")


def _judge(provider=None, **config) -> RolloutJudge:
    provider = provider or MockProvider()
    return RolloutJudge(provider, JudgeConfig(provider=provider.name, **config), rules=RULES, applies=APPLIES)


def _trial(**over) -> dict:
    row = {
        "kind": "trial", "trial": 3, "arm": "B", "staging": "fixture", "verdict": "held", "policyStatus": "grasp_handover",
        "pegSource": "gt", "lateralMm": 0.6, "closeAboveTargetMm": -5.7, "aimErrorMm": {"150": 58.0, "100": 33.0, "60": 12.0},
        "widthAtClose": 0.31, "widthLifted": 0.31, "commandedGripper": 0.0, "funnelEntryReason": "reached_align_z",
        "insert": {
            "autoVerdict": "standing", "stoppedOn": "contact", "searchStoppedOn": "exhausted", "aboveTargetMm": 4.0,
            "lateralErrorMm": 3.9, "searchTried": 7, "dfzPeakN": -30.1, "pressCapped": True, "released": False, "grade": "out",
        },
        "inserted": False,
    }
    row.update(over)
    return row


# ------------------------------------------------------------------------------ the contract ---


def test_a_provider_answer_becomes_a_ranked_schema_decision():
    provider = MockProvider({"failure_reason": {"MISALIGNMENT": 0.7, "FALSE_CONTACT": 0.2, "OTHER": 0.1}})
    d = _judge(provider).evaluate("failure_reason", FAILED)

    assert (d.decision, d.confidence, d.source, d.status) == ("MISALIGNMENT", 0.7, "provider", "ok")
    assert d.alternatives == [["FALSE_CONTACT", 0.2], ["OTHER", 0.1]]
    assert d.schema_version == "v1" and d.provider == "mock" and d.latency_ms is not None
    assert d.route == "human_review"


def test_the_mock_is_deterministic_and_stays_inside_the_options():
    a = _judge().evaluate("failure_reason", FAILED)
    b = _judge().evaluate("failure_reason", FAILED)
    assert a.decision == b.decision and a.confidence == b.confidence
    assert a.decision in TASKS["failure_reason"].labels


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        ({"probabilities": {"LOW_BATTERY": 1.0}}, "invalid"),  # outside the taxonomy
        ({"probabilities": {"MISALIGNMENT": 0.4}}, "invalid"),  # mass missing
        ({"probabilities": {"MISALIGNMENT": math.nan, "OTHER": 1.0}}, "invalid"),
        ({"probabilities": {"MISALIGNMENT": -0.5, "OTHER": 1.5}}, "invalid"),
        ({"decision": "MISALIGNMENT"}, "invalid"),  # not the schema at all
        ("MISALIGNMENT", "invalid"),
    ],
)
def test_an_answer_outside_the_schema_is_a_fallback_not_a_decision(answer, status):
    d = _judge(MockProvider({"failure_reason": lambda _f: answer})).evaluate("failure_reason", FAILED)
    assert (d.decision, d.source, d.status, d.route) == (AMBIGUOUS, "fallback", status, "human_review")
    assert d.confidence is None


def test_a_provider_that_raises_or_hangs_never_reaches_the_caller():
    def boom(_features):
        raise ConnectionError("down")

    raised = _judge(MockProvider({"failure_reason": boom})).evaluate("failure_reason", FAILED)
    assert (raised.decision, raised.status) == (AMBIGUOUS, "error") and "ConnectionError" in raised.error

    started = time.perf_counter()
    hung = _judge(MockProvider(delay_s=2.0), timeout_s=0.05).evaluate("failure_reason", FAILED)
    assert time.perf_counter() - started < 1.0
    assert (hung.decision, hung.status, hung.route) == (AMBIGUOUS, "timeout", "human_review")


def test_a_yes_no_task_falls_back_to_unknown_not_to_no():
    span = FeatureSnapshot("ep", {"record": "rollout_log"}, expert_span_id="ep/span0")
    d = _judge(MockProvider({"intervention_necessary": lambda _f: {"probabilities": {"MAYBE": 1.0}}})).evaluate(
        "intervention_necessary", span
    )
    assert d.decision == UNKNOWN


def test_jev_without_an_endpoint_is_unavailable_and_fails_safe(monkeypatch):
    monkeypatch.delenv("JEV_API_URL", raising=False)
    d = _judge(JevProvider()).evaluate("failure_reason", FAILED)
    assert (d.decision, d.status, d.provider) == (AMBIGUOUS, "unavailable", "jev")


def test_jev_parses_through_the_same_validation(monkeypatch):
    provider = JevProvider(url="http://jev.invalid", model="jev-test")
    sent = {}

    def fake_request(body):
        sent.update(body)
        return {"probabilities": {"FALSE_CONTACT": 0.97, "MISALIGNMENT": 0.03}, "model_version": "2026.09"}

    monkeypatch.setattr(provider, "_request", fake_request)
    d = _judge(provider).evaluate("failure_reason", FAILED)
    assert (d.decision, d.route, d.model) == ("FALSE_CONTACT", "auto_accept", "jev-test")
    assert provider.version == "2026.09"
    assert sent["task"] == "failure_reason" and set(sent["options"]) == set(TASKS["failure_reason"].labels)
    assert sent["features"] == FAILED.features


def test_routing_bands_come_from_config():
    provider = MockProvider({"failure_reason": {"MISALIGNMENT": 0.8, "OTHER": 0.2}})
    assert _judge(provider).evaluate("failure_reason", FAILED).route == "human_review"
    assert _judge(provider, auto_accept_at=0.75).evaluate("failure_reason", FAILED).route == "auto_accept"
    assert _judge(provider, ambiguous_below=0.85).evaluate("failure_reason", FAILED).route == "ambiguous"


def test_config_rejects_unknown_keys_and_crossed_bands(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"auto_accept_at": 0.9, "threshold": 1}))
    with pytest.raises(ValueError, match="threshold"):
        JudgeConfig.load(path)
    with pytest.raises(ValueError):
        JudgeConfig.load(None, auto_accept_at=0.5, ambiguous_below=0.7)
    path.write_text(json.dumps({"auto_accept_at": 0.9}))
    assert JudgeConfig.load(path, provider="jev").auto_accept_at == 0.9


def test_judging_is_off_unless_the_flag_is_set():
    assert judge_from_env({}) is None
    assert judge_from_env({"FR3_JUDGE_ENABLED": "0", "FR3_JUDGE_PROVIDER": "mock"}) is None
    judge = judge_from_env({"FR3_JUDGE_ENABLED": "1", "FR3_JUDGE_PROVIDER": "mock"})
    assert isinstance(judge.provider, MockProvider)


def test_unknown_tasks_are_the_callers_bug_and_supervision_is_not_a_task():
    with pytest.raises(KeyError):
        _judge().evaluate("pick_next_action", FAILED)
    assert "CONTINUE_POLICY" in SUPERVISORY_ACTIONS
    assert not set(SUPERVISORY_ACTIONS) & {label for t in TASKS.values() for label in t.labels}


# ------------------------------------------------------------------------- rules and scope ---


@pytest.mark.parametrize(
    ("row", "reason", "side"),
    [
        (_trial(verdict="no_close", policyStatus="grasp_timeout", insert=None, inserted=None), "POLICY_STALL", "POLICY"),
        (_trial(verdict="empty", pegUntouched=True, insert=None, inserted=None, arm="A"), "EMPTY_GRASP", "POLICY"),
        (_trial(verdict="voided", insert=None, inserted=None), "ABORTED", "ENVIRONMENT"),
    ],
)
def test_what_the_record_already_decides_never_reaches_the_provider(row, reason, side):
    provider = MockProvider()
    snap = grasp_trial_snapshot(row, run="r", attended=False, source="t")
    judge = _judge(provider)
    a, b = judge.evaluate("failure_reason", snap), judge.evaluate("policy_vs_runtime_failure", snap)
    assert (a.decision, a.source, a.route, b.decision) == (reason, "rule", "auto_accept", side)
    assert provider.calls == []


def test_without_rules_every_case_goes_to_the_provider():
    provider = MockProvider()
    snap = grasp_trial_snapshot(_trial(verdict="voided", insert=None, inserted=None), run="r", attended=False, source="t")
    assert _judge(provider, use_rules=False).evaluate("failure_reason", snap).source == "provider"
    assert provider.calls == [("choice", "failure_reason")]


def test_an_empty_grasp_that_came_low_near_the_peg_is_a_judgment():
    snap = grasp_trial_snapshot(_trial(verdict="empty", pegUntouched=False, insert=None, inserted=None), run="r", attended=False, source="t")
    assert _judge().evaluate("failure_reason", snap).source == "provider"


def test_why_a_success_failed_is_not_asked_and_scopes_do_not_cross():
    ok = grasp_trial_snapshot(_trial(inserted=True, insert={**_trial()["insert"], "grade": "in"}), run="r", attended=True, source="t")
    judge = _judge()
    assert judge.evaluate("failure_reason", ok).source == "not_applicable"
    assert judge.evaluate("intervention_necessary", ok).source == "not_applicable"
    span = FeatureSnapshot("ep", {"outcome": "failure"}, expert_span_id="ep/span0")
    assert judge.evaluate("failure_reason", span).source == "not_applicable"
    assert judge.evaluate("usable_as_correction_data", span).source == "provider"


# ---------------------------------------------------------------------------------- features ---


def test_the_persons_grade_is_a_label_and_never_a_feature():
    snap = grasp_trial_snapshot(_trial(), run="r", attended=True, source="t")
    assert snap.labels == {"insert_grade": "out"}
    blob = json.dumps(snap.payload())
    assert "grade" not in blob and '"out"' not in blob
    assert snap.features["outcome"] == "insert_failed" and snap.features["outcome_source"] == "human"


def test_no_staged_peg_means_no_peg_residuals_and_the_hole_is_never_gt():
    snap = grasp_trial_snapshot(_trial(pegSource=None), run="r", attended=False, source="t")
    assert not any(k.startswith(("close_xy_to_peg", "policy_aim_error", "close_z_above_peg")) for k in snap.features)
    assert snap.features["hole_pose_is_gt"] is False
    assert snap.features["servo_lateral_to_nominal_hole_mm"] == 3.9
    assert {"policy_action_stats", "ft_time_series_summary"} <= set(snap.missing)


def test_an_unmeasured_value_is_named_missing_not_defaulted():
    snap = grasp_trial_snapshot(_trial(widthLifted=None, maxResidualMm="nan"), run="r", attended=False, source="t")
    assert "gripper_width_lifted" in snap.missing and "max_residual_mm" in snap.missing
    assert "gripper_width_lifted" not in snap.features


def test_a_run_file_yields_its_trials_and_the_halts_that_left_no_row(tmp_path):
    path = tmp_path / "grasp_20260930_000000.jsonl"
    rows = [
        {"kind": "run_start", "request": {"attended": False}},
        _trial(trial=0),
        {"kind": "halt", "trial": 1, "reason": "motion_fault", "error": "RTC action queue starved"},
        {"kind": "halt", "trial": 0, "reason": "peg_lost"},
        {"kind": "run_end", "halted": "motion_fault"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    snaps = load_grasp_loop_run(path)
    assert [s.episode_id for s in snaps] == ["grasp:grasp_20260930_000000:t000", "grasp:grasp_20260930_000000:t001:halt"]
    halt = _judge().evaluate("failure_reason", snaps[1])
    assert (halt.decision, halt.source) == ("RUNTIME_ERROR", "rule")


def _rollout_entry(**over) -> dict:
    entry = {
        "recordedAt": "2026-09-22T10:23:19+00:00", "rolloutIndex": 19, "outcome": "failure", "mode": "real", "steps": 400,
        "stageId": "transport", "blocker": "policy_action", "blockers": ["policy_action"], "note": "swung wide",
        "intervened": True, "expertSteps": 60, "expertSpans": [[100, 139], [300, 319]],
        "geometry": {"graspXyz": [0.4324, -0.2081, 0.0563], "closed": True, "graspBy": "policy", "liftM": 0.1},
        "resetTarget": [0.4343, -0.2247, 0.058],
    }
    entry.update(over)
    return entry


def test_a_rollout_log_line_is_one_episode_and_one_snapshot_per_takeover(tmp_path):
    path = tmp_path / "rollout_log.jsonl"
    path.write_text(json.dumps(_rollout_entry()) + "\n")
    episode, first, second = load_rollout_log(path)
    assert episode.expert_span_id is None and second.expert_span_id == f"{episode.episode_id}/span1"
    assert first.features["span_steps"] == 40 and second.features["steps_after_span"] == 80
    assert episode.features["grasp_xy_to_staged_peg_mm"] == pytest.approx(16.7, abs=0.1)
    assert "span_policy_status" in first.missing and "servo_auto_verdict" not in episode.features
    assert episode.labels["blocker"] == "policy_action" and "swung" not in json.dumps(episode.payload())


def test_without_a_reset_target_the_grasp_offset_is_not_invented():
    entry = _rollout_entry()
    del entry["resetTarget"]
    (episode, *_spans) = load_rollout_log_entry(entry)
    assert episode.features["peg_start_is_staged_gt"] is False
    assert "grasp_xy_to_staged_peg_mm" not in episode.features


def load_rollout_log_entry(entry):
    from tools.fr3.judgment_features import rollout_snapshots

    return rollout_snapshots(entry, source="t")


# ----------------------------------------------------------------------------- logging + CLI ---


def test_a_decision_record_carries_what_a_benchmark_needs(tmp_path):
    provider = MockProvider()
    d = _judge(provider).evaluate("failure_reason", FAILED)
    record = decision_record(FAILED, d, provider)
    for key in (
        "episode_id", "expert_span_id", "task_name", "features", "features_sha256", "decision", "confidence",
        "alternatives", "latency_ms", "provider", "model", "provider_version", "schema_version", "timestamp", "route",
    ):
        assert key in record
    log = DecisionLog(tmp_path / "sub" / "d.jsonl")
    log.append(record)
    log.append(record)
    assert len((tmp_path / "sub" / "d.jsonl").read_text().splitlines()) == 2
    assert record["features_sha256"] == FeatureSnapshot("other", dict(FAILED.features)).sha256


def test_judge_then_benchmark_end_to_end(tmp_path, capsys):
    run = tmp_path / "grasp_20260930_000000.jsonl"
    run.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"kind": "run_start", "request": {"attended": True}},
                _trial(trial=0),
                _trial(trial=1, lateralMm=9.0),
                _trial(trial=2, verdict="no_close", insert=None, inserted=None),
                _trial(trial=3, inserted=True, insert={**_trial()["insert"], "grade": "in"}),
            )
        )
    )
    (tmp_path / "grasp_20260930_000000_force.jsonl").write_text('{"name": "x"}\n')
    log = tmp_path / "rollout_log.jsonl"
    log.write_text(json.dumps(_rollout_entry()) + "\n")
    out = tmp_path / "j.jsonl"

    assert cli.main(["judge", "--grasp-loop", str(tmp_path / "grasp_*.jsonl"), "--rollout-log", str(log), "--out", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    by = {(r["episode_id"], r["task_name"]) for r in records}
    assert ("grasp:grasp_20260930_000000:t003", "failure_reason") not in by  # inserted
    assert sum(r["task_name"] == "intervention_necessary" for r in records) == 2
    assert cli.main(["judge", "--grasp-loop", str(run), "--out", str(out)]) == 2  # never overwrites

    sheet = out.with_suffix(".csv")
    rows = list(csv.DictReader(sheet.open()))
    assert rows[0]["human_label"] == "" and "MISALIGNMENT" in rows[0]["options"]
    for row in rows:
        if row["task_name"] == "failure_reason":
            row["human_label"] = row["decision"] if row["decided_by"] == "rule" else "MISALIGNMENT"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cli.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report_path = tmp_path / "report.json"
    assert cli.main(["benchmark", "--decisions", str(out), "--labels", str(sheet), "--out", str(report_path)]) == 0
    report = json.loads(report_path.read_text())["failure_reason"]
    assert report["rules"] == {"n": 1, "accuracy": 1.0, "disagreements": []}
    assert report["provider"]["n"] == report["labelled"] - 1
    assert "MISALIGNMENT" in report["provider"]["confusion"]
    assert [s["auto_accept_at"] for s in report["provider"]["threshold_sweep"]][:2] == [0.5, 0.6]


def test_the_cli_refuses_an_unavailable_provider_up_front(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("JEV_API_URL", raising=False)
    assert cli.main(["judge", "--rollout-log", str(tmp_path / "none.jsonl"), "--provider", "jev", "--out", str(tmp_path / "o.jsonl")]) == 2
    assert "JEV_API_URL" in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()


def test_benchmark_refuses_labels_outside_the_options(tmp_path):
    sheet = tmp_path / "s.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cli.CSV_FIELDS)
        writer.writeheader()
        writer.writerow({"episode_id": "e", "task_name": "failure_reason", "human_label": "gremlins"})
    (tmp_path / "d.jsonl").write_text("")
    assert cli.main(["benchmark", "--decisions", str(tmp_path / "d.jsonl"), "--labels", str(sheet)]) == 2
