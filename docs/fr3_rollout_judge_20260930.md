# FR3 rollout judge (Jev integration, phase 1: offline)

A replaceable judge for the fuzzy, finite-choice questions about a *finished* rollout. It is not a
controller and not a new centre of the system: it reads records that already exist and writes a
decision log beside them. Nothing in the runtime imports it.

```
rollout records ──> deterministic features ──> rules ──(undecided)──> provider (Jev / mock)
                                                  │                         │
                                                  └──> StructuredDecision <─┘ ──> decision log (JSONL)
                                                                             └──> labelling sheet (CSV)
                                                          human labels ──> benchmark ──> thresholds
```

## What the records hold today

| Stream | Written by | Per | Machine-measured | Said by a person |
|---|---|---|---|---|
| `outputs/analysis/grasp_loop/grasp_*.jsonl` | `tools/fr3/grasp_loop.py` | trial row, plus `halt` / `run_end` events | grasp `verdict` (lifted width), `policyStatus`, funnel handoff (`funnelEntryReason`, `funnelState`, xy errors), aim errors at 150/100/60 mm, gripper widths, terminal-servo `insert` (autoVerdict, stoppedOn, aboveTargetMm, lateralErrorMm, dfzPeakN, pressCapped) | `insert.grade` in/out (attended runs only; then `inserted` is the grade) |
| `outputs/rollouts/rollout_log.jsonl` | `tools/data_collection_gui/checkpoints.py::append_rollout_outcome` | GUI rollout | `geometry`, `expertSpans`, `expertSteps`, `terminalServo`, `arm`, `resetTarget` | `outcome`, `stage`/`stageId`, `blocker(s)`, `note` |

On 2026-09-30: 483 grasp-loop trials (121 with an in/out grade), 215 rollout-log lines (158 with a
blocker, 67 with takeover spans). `takeovers` (per-span pose and policy status) is defined but was
never populated. There is no policy action/velocity statistic and no F/T time-series summary in
either stream; features name them as missing. The `_force.jsonl` traces and the runtime logs could
feed those later.

## Boundaries

Stays deterministic (Python), never sent to the judge:

- anything measured or computed: pose transforms, distances, the lifted-width grasp check, the
  servo's stop classification, thresholds, safety limits, collision/reflex handling, e-stop;
- anything the record already states. The rules in `judgment_features.py`: a `halt` → `RUNTIME_ERROR`;
  voided/aborted → `ABORTED`; `no_close` (grasp timeout) → `POLICY_STALL`; empty with the peg
  untouched → `EMPTY_GRASP`; arm A empty → `POLICY`. These are logged with `decided_by: rule`.

Goes to the judge: a finite choice, yes/no or score where the measurements do not decide it — why
an insertion that stopped on contact failed (misalignment vs false contact vs tracking), whether an
empty grasp that came low near the peg knocked it, whether a takeover was necessary or is usable as
correction data, whether a record is inconsistent enough for a person to look.

Never: robot actions, anything in the 1 kHz FR3 / 200 Hz gripper / visual-servo loops, safety.
`SUPERVISORY_ACTIONS` (CONTINUE_POLICY, HANDOFF_VISUAL_SERVO, HANDOFF_FORCE_CONTROL, RELOCALIZE,
ASK_HUMAN, ABORT) is declared for phase 2 and is not a task; `evaluate` refuses it.

## Modules

- `tools/fr3/judgment.py`
  - `TaskSpec` / `TASKS`: `failure_reason` (the taxonomy), `policy_vs_runtime_failure`
    (POLICY/RUNTIME/ENVIRONMENT), `intervention_necessary` and `usable_as_correction_data`
    (span-scoped yes/no), `needs_human_review` (yes/no).
  - `JudgmentProvider` with `choice` / `yes_no` / `score`; `MockProvider`, `JevProvider`.
  - `RolloutJudge.evaluate(task_name, snapshot) -> StructuredDecision`. Scope check → applicability
    → rule → provider (on a daemon thread with a timeout) → schema validation → routing.
  - `JudgeConfig` (JSON-loadable), `judge_from_env()`, `decision_record()`, `DecisionLog`.
- `tools/fr3/judgment_features.py`: `load_grasp_loop_run`, `load_rollout_log`, the rules.
- `tools/fr3/fr3_judge_rollouts.py`: the `judge` and `benchmark` CLI.
- `tests/scripts/test_fr3_judgment.py`.

### Fail-safe

A missing endpoint, an exception, a timeout, a label outside the options, probabilities that are
NaN/negative or do not sum to 1 (±5 %) — each becomes the task's fallback (`AMBIGUOUS` for a
choice, `UNKNOWN` for yes/no), `decided_by: fallback`, `route: human_review`, with the status and
error logged. Only an unknown task name raises, because that is the caller's bug.

### Features

Compact and flat. Only machine-measured values; a value that was not measured is absent from
`features` and named in `missing`. People's answers go to `labels` (logged as `human`) and are never
part of the provider input. Peg residuals are only computed against a staged peg (grasp loop
`pegSource == "gt"`, rollout log `resetTarget`); the hole's position is the configured nominal one
and it creeps, so its lateral reading is named `servo_lateral_to_nominal_hole_mm` and
`hole_pose_is_gt` is always false.

### Decision log (one JSONL line per decision)

`timestamp, episode_id, expert_span_id, source_file, task_name, decision, score, confidence,
alternatives, route, decided_by, status, error, latency_ms, provider, model, provider_version,
schema_version, features_sha256, features, missing, human`.

### Config and switches

`JudgeConfig`: `provider` (mock|jev), `timeout_s` (10), `auto_accept_at` (0.95), `ambiguous_below`
(0.6), `use_rules` (true), `top_alternatives` (3). The bands are a starting point; the benchmark
decides them. Environment: `FR3_JUDGE_ENABLED` (off by default — `judge_from_env()` then returns
None, so a future hook does nothing), `FR3_JUDGE_PROVIDER`, `FR3_JUDGE_CONFIG`; for Jev,
`JEV_API_URL`, `JEV_API_KEY`, `JEV_MODEL`.

## Running it

```bash
PYTHONPATH=$PWD/src:$PWD .venv-fr3/bin/python tools/fr3/fr3_judge_rollouts.py judge \
  --grasp-loop 'outputs/analysis/grasp_loop/grasp_2026*.jsonl' \
  --rollout-log outputs/rollouts/rollout_log.jsonl --provider mock
# -> outputs/analysis/judge/judge_<ts>.jsonl and .csv
```

`--no-rules` sends every case to the provider (to benchmark it on the rule-decided cases too);
`--tasks` narrows the task list; `--config` loads a `JudgeConfig` JSON. The mock's answers are a
hash of the features: they exercise the pipeline and carry no information.

## Swapping in the real Jev API

1. Set `JEV_API_URL` (and `JEV_API_KEY`, `JEV_MODEL`).
2. `JevProvider._request` / `_parse` are the only code that knows the wire format. The current
   format is an assumption: POST `{model, task, kind, question, options, features, missing}`,
   expect `{"probabilities": {label: p}}` (or `{score, confidence}`) plus optional
   `model_version`. Adapt those two methods to the real API (or an SDK call inside `_request`,
   imported there so it stays optional); the judge validates whatever `_parse` returns.
3. `test_jev_parses_through_the_same_validation` shows the seam: monkeypatch `_request`.
4. Run `judge --provider jev` on a small `--limit` first and check `status` counts in the log.

## Benchmark with 50–200 labelled rollouts

1. Run `judge --no-rules` once with the provider under test, and once with `--provider mock` as the
   floor. Both logs share `features_sha256`, so the same snapshots are scored.
2. Label in the CSV: fill `human_label` with one of `options` (`AMBIGUOUS` is allowed). Use the
   `operator_hint` column (the existing blocker/grade) and the photos from
   `tools/fr3/grasp_loop_audit.py`. Stratify: on 2026-09-30 the 208 provider-routed failures are
   27 empty grasps and 16 insertion failures from the grasp loop plus 165 GUI failures; take all
   43 grasp-loop ones and a random ~100 GUI ones, so each class has more than a handful. (The
   `EMPTY_GRASP` rule fired on none of the 27: `pegUntouched` is absent on the 24 from runs that
   predate it and false on the other 3.)
3. `benchmark --decisions <log>.jsonl --labels <sheet>.csv` gives per task: accuracy, confusion
   matrix, per-class precision/recall, calibration by confidence bin, and a threshold sweep
   (coverage vs. accuracy of what would be auto-accepted), with rule rows and fallbacks
   reported separately.
4. Pick `auto_accept_at` from the sweep: the lowest threshold whose accepted accuracy clears the
   bar you need, then write it into the config. With 50–200 labels the per-class numbers have
   wide intervals: read a class with fewer than ~10 labels as unknown.
5. Only then let a decision feed curation (e.g. `usable_as_correction_data=NO` excluding a span),
   and still behind the review queue for anything below the threshold.
