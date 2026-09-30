"""A replaceable judge for the fuzzy, finite-choice questions about a finished rollout.

Some questions about a rollout have a measured answer: the gripper width after the lift says
whether the peg came up, the servo's stop height says where the descent ended, and a pose
difference is arithmetic. Those stay in the code that measures them. Other questions have no
measured answer: why an insertion that stopped on contact failed, whether a takeover was needed,
whether a span is worth training on. Today those become if/else chains nobody trusts, or they wait
for a person. This module asks a `JudgmentProvider` instead, and holds it to a fixed contract:

* a task is a closed set of options (or yes/no, or a score in [0, 1]) -- never an action;
* the provider's answer is validated against that set before anyone sees it;
* any failure -- no provider, a timeout, an exception, an answer outside the set -- becomes the
  task's fail-safe decision (`AMBIGUOUS` / `UNKNOWN`) routed to human review, never an exception
  into the caller;
* every decision, including the rule-decided and the failed ones, is one logged record carrying
  the feature snapshot and its hash, so a person's labels can be scored against it later.

Offline only for now. Nothing here imports the runtime, and nothing in the runtime imports this:
there is no path from a judgment to the robot. `SUPERVISORY_ACTIONS` names the options a later,
low-frequency supervisor might choose between; it is not a task and `evaluate` refuses it.

The provider is chosen by configuration. `MockProvider` needs nothing and is what the tests use;
`JevProvider` needs `JEV_API_URL` (and usually `JEV_API_KEY`); see its docstring for the one
method to adapt once the real API is known.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "v1"

# The fail-safe answers. A choice task's is one of its options, so a person reviewing the queue
# sees it among the others; a yes/no task's is a third answer, because "no" is a real verdict.
AMBIGUOUS = "AMBIGUOUS"
UNKNOWN = "UNKNOWN"

FAILURE_TAXONOMY: dict[str, str] = {
    "EMPTY_GRASP": "the gripper closed and lifted with nothing in it; the peg was not moved",
    "PEG_KNOCKED": "the tool or gripper pushed the peg over or out of place",
    "MISALIGNMENT": "the peg reached the hole off-axis or off-centre and could not go in",
    "FALSE_CONTACT": "the descent stopped on something other than the seated peg (rim, table, fixture)",
    "SERVO_TRACKING_ERROR": "the arm did not follow the servo's setpoints closely enough to finish",
    "POLICY_STALL": "the policy stopped making progress (hovered, oscillated, never closed)",
    "RUNTIME_ERROR": "the software or controller failed (fault, starved queue, reset failure)",
    "ABORTED": "not evidence about the policy: stopped by a person, or the scene was invalid",
    "OTHER": "a failure that none of the other options describes",
    AMBIGUOUS: "the evidence does not decide between the options",
}

FAILURE_SIDES: dict[str, str] = {
    "POLICY": "the learned policy's own actions caused the failure",
    "RUNTIME": "the scripted runtime, controller or software caused it (servo, funnel, reset, fault)",
    "ENVIRONMENT": "the scene did: the peg fell before the policy acted, hardware, a person",
    AMBIGUOUS: "the evidence does not decide between the options",
}

# Phase 2, reserved: what a low-frequency supervisor could route between. Declared so the option
# names are settled before anything depends on them; `evaluate` refuses it, and it must never be
# wired into a control loop -- a supervisor's pick would be checked by deterministic code first.
SUPERVISORY_ACTIONS: tuple[str, ...] = (
    "CONTINUE_POLICY",
    "HANDOFF_VISUAL_SERVO",
    "HANDOFF_FORCE_CONTROL",
    "RELOCALIZE",
    "ASK_HUMAN",
    "ABORT",
)

CHOICE, YES_NO, SCORE = "choice", "yes_no", "score"
YES, NO = "YES", "NO"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # CHOICE | YES_NO | SCORE
    question: str
    options: dict[str, str] = field(default_factory=dict)  # label -> definition; empty for SCORE
    # "episode" or "span": whether one answer covers a whole rollout or one takeover in it.
    scope: str = "episode"

    @property
    def fallback(self) -> str:
        return AMBIGUOUS if self.kind == CHOICE else UNKNOWN

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.options)


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            "failure_reason",
            CHOICE,
            "This rollout failed. Which option best explains why?",
            FAILURE_TAXONOMY,
        ),
        TaskSpec(
            "policy_vs_runtime_failure",
            CHOICE,
            "This rollout failed. Whose failure was it?",
            FAILURE_SIDES,
        ),
        TaskSpec(
            "intervention_necessary",
            YES_NO,
            "A person took over from the policy for this span. Would the rollout have failed without it?",
            {YES: "the policy was heading for a failure", NO: "the policy would have recovered or succeeded"},
            scope="span",
        ),
        TaskSpec(
            "usable_as_correction_data",
            YES_NO,
            "Should this takeover span be trained on as a correction of the policy?",
            {YES: "a clean, purposeful correction", NO: "noisy, aimless, or correcting a non-policy fault"},
            scope="span",
        ),
        TaskSpec(
            "needs_human_review",
            YES_NO,
            "Do these measurements disagree with each other, or with the recorded outcome, enough that a person should look?",
            {YES: "something does not add up", NO: "consistent"},
        ),
    )
}


# ------------------------------------------------------------------------------ inputs/outputs ---


@dataclass(frozen=True)
class FeatureSnapshot:
    """What the judge sees about one episode (or one span in it). Built by deterministic code.

    `features` holds only machine-measured values; a value that was not measured is None and its
    name is in `missing`, never a default. `labels` holds what people said about the episode --
    kept apart so it is never sent to a provider, and used only to benchmark one.
    """

    episode_id: str
    features: dict[str, Any]
    missing: tuple[str, ...] = ()
    expert_span_id: str | None = None
    source: str = ""
    labels: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {"features": self.features, "missing": list(self.missing)}

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class StructuredDecision:
    task_name: str
    decision: str | None
    confidence: float | None
    alternatives: list[list[Any]]  # [[label, probability], ...], most probable first
    model: str
    provider: str
    schema_version: str = SCHEMA_VERSION
    score: float | None = None
    # "provider": the provider answered; "rule": deterministic code decided without asking;
    # "fallback": nothing valid came back and this is the task's fail-safe answer;
    # "not_applicable": the task does not apply to this episode (e.g. why a success failed).
    source: str = "provider"
    status: str = "ok"
    error: str | None = None
    latency_ms: float | None = None
    route: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class InvalidAnswer(ValueError):
    """A provider's answer that does not fit the task's schema."""


# ----------------------------------------------------------------------------------- providers ---


class JudgmentProvider:
    """One way of answering. Every method returns a raw answer that the judge validates:

    * `choice` / `yes_no`: `{"probabilities": {label: p, ...}}` over the given labels;
    * `score`: `{"score": s, "confidence": c}` with both in [0, 1].

    A provider may raise; the judge turns that into the fail-safe decision.
    """

    name = "base"
    model = "none"
    version = "0"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def choice(self, *, task: TaskSpec, options: Mapping[str, str], features: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def yes_no(self, *, task: TaskSpec, features: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def score(self, *, task: TaskSpec, features: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class MockProvider(JudgmentProvider):
    """Deterministic stand-in: the same features always get the same answer, and nothing leaves
    the process. Its answers carry no information -- they exist so the pipeline, the logging and
    the benchmark can be run end to end without an API.

    `answers` overrides it per task: a dict of probabilities, or a callable of the features that
    returns a raw answer (which may be deliberately malformed, or raise, to test the fail-safe).
    `delay_s` sleeps before answering, for the timeout path.
    """

    name = "mock"
    model = "mock-hash"
    version = "1"

    def __init__(self, answers: Mapping[str, Any] | None = None, *, delay_s: float = 0.0):
        self.answers = dict(answers or {})
        self.delay_s = delay_s
        self.calls: list[tuple[str, str]] = []

    def _answer(self, method: str, task: TaskSpec, labels: Sequence[str], features: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((method, task.name))
        if self.delay_s:
            time.sleep(self.delay_s)
        override = self.answers.get(task.name)
        if callable(override):
            return override(features)
        if override is not None:
            return {"probabilities": dict(override)} if method != "score" else dict(override)
        digest = hashlib.sha256(json.dumps(features, sort_keys=True, default=str).encode()).digest()
        if method == "score":
            return {"score": digest[0] / 255.0, "confidence": 0.5 + digest[1] / 510.0}
        weights = [1.0 + digest[i % len(digest)] for i in range(len(labels))]
        top = digest[-1] % len(labels)
        weights[top] *= 4.0
        total = sum(weights)
        return {"probabilities": {label: w / total for label, w in zip(labels, weights)}}

    def choice(self, *, task, options, features):
        return self._answer("choice", task, list(options), features)

    def yes_no(self, *, task, features):
        return self._answer("yes_no", task, [YES, NO], features)

    def score(self, *, task, features):
        return self._answer("score", task, [], features)


class JevProvider(JudgmentProvider):
    """Jev behind the same three methods. Configured from the environment:

        JEV_API_URL   the endpoint (required; without it the provider reports unavailable)
        JEV_API_KEY   sent as a bearer token when set
        JEV_MODEL     the model name to request and to log (default "jev")

    The wire format is an ASSUMPTION until the real API is documented: `_request` posts
    `{"model", "task", "kind", "question", "options", "features", "missing"}` as JSON and
    `_parse` expects `{"probabilities": {...}}` (choice / yes_no) or `{"score", "confidence"}`
    back, optionally with a "model_version". Adapting to the real API means changing those two
    methods and nothing else -- the judge validates whatever `_parse` returns. Only the standard
    library is used, so the integration stays an optional dependency.
    """

    name = "jev"

    def __init__(self, url: str | None = None, api_key: str | None = None, model: str | None = None, *, timeout_s: float = 10.0):
        self.url = url if url is not None else os.environ.get("JEV_API_URL", "")
        self.api_key = api_key if api_key is not None else os.environ.get("JEV_API_KEY", "")
        self.model = model or os.environ.get("JEV_MODEL", "jev")
        self.version = "unknown"
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        if not self.url:
            return False, "JEV_API_URL is not set"
        return True, ""

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 -- configured endpoint
            return json.loads(response.read().decode("utf-8"))

    def _parse(self, reply: dict[str, Any]) -> dict[str, Any]:
        if isinstance(reply, dict) and reply.get("model_version"):
            self.version = str(reply["model_version"])[:64]
        return reply

    def _ask(self, task: TaskSpec, kind: str, options: Mapping[str, str], features: Mapping[str, Any]) -> dict[str, Any]:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        body = {
            "model": self.model,
            "task": task.name,
            "kind": kind,
            "question": task.question,
            "options": dict(options),
            "features": features.get("features", features),
            "missing": features.get("missing", []),
        }
        return self._parse(self._request(body))

    def choice(self, *, task, options, features):
        return self._ask(task, CHOICE, options, features)

    def yes_no(self, *, task, features):
        return self._ask(task, YES_NO, task.options, features)

    def score(self, *, task, features):
        return self._ask(task, SCORE, {}, features)


PROVIDERS: dict[str, Callable[..., JudgmentProvider]] = {"mock": MockProvider, "jev": JevProvider}


# -------------------------------------------------------------------------------------- config ---


@dataclass
class JudgeConfig:
    """Everything tunable. The routing bands are a starting point to be measured, not a policy:
    see the benchmark in fr3_judge_rollouts.py before trusting any of them."""

    provider: str = "mock"
    timeout_s: float = 10.0
    # decision confidence >= auto_accept_at -> "auto_accept"; < ambiguous_below -> "ambiguous";
    # between -> "human_review". Fallbacks always go to "human_review".
    auto_accept_at: float = 0.95
    ambiguous_below: float = 0.6
    # Let deterministic rules settle the cases the record already decides. Off for a benchmark
    # that should put every case to the provider.
    use_rules: bool = True
    top_alternatives: int = 3

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "JudgeConfig":
        values: dict[str, Any] = {}
        if path:
            values.update(json.loads(Path(path).read_text(encoding="utf-8")))
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown judge config keys: {', '.join(unknown)}")
        config = cls(**values)
        if not 0.0 <= config.ambiguous_below <= config.auto_accept_at <= 1.0:
            raise ValueError("need 0 <= ambiguous_below <= auto_accept_at <= 1")
        if config.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {config.provider!r}; known: {', '.join(PROVIDERS)}")
        return config


ENV_ENABLED = "FR3_JUDGE_ENABLED"
ENV_PROVIDER = "FR3_JUDGE_PROVIDER"
ENV_CONFIG = "FR3_JUDGE_CONFIG"


def judge_from_env(environ: Mapping[str, str] | None = None) -> "RolloutJudge | None":
    """The judge a caller should use, or None when judging is switched off (the default).

    Any future hook outside this module goes through here and does nothing on None, so with the
    flag unset the system behaves exactly as before.
    """

    env = os.environ if environ is None else environ
    if str(env.get(ENV_ENABLED, "")).strip().lower() not in ("1", "true", "yes", "on"):
        return None
    config = JudgeConfig.load(env.get(ENV_CONFIG) or None, provider=env.get(ENV_PROVIDER) or None)
    return RolloutJudge(PROVIDERS[config.provider](), config)


# --------------------------------------------------------------------------------------- judge ---

# A rule sees the snapshot and returns a label when the record already decides the task, else None.
Rule = Callable[[FeatureSnapshot], str | None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _call_with_timeout(fn: Callable[[], dict[str, Any]], timeout_s: float) -> dict[str, Any]:
    """Run `fn` on a daemon thread; a provider that hangs costs this call, not the caller."""

    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- reported through the box
            box["error"] = exc

    thread = threading.Thread(target=run, name="judgment-provider", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"provider did not answer within {timeout_s:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _finite_unit(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidAnswer(f"{what} is not a number: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise InvalidAnswer(f"{what} is outside [0, 1]: {value!r}")
    return number


def validate_distribution(raw: Any, labels: Sequence[str]) -> list[tuple[str, float]]:
    """A provider's probabilities as (label, p) most probable first, or InvalidAnswer.

    Every label must be one of the task's; the total must be 1 within 5 % (then renormalised),
    so an answer that silently dropped half its mass is refused rather than read as confident.
    """

    if not isinstance(raw, dict) or not isinstance(raw.get("probabilities"), dict):
        raise InvalidAnswer("answer has no 'probabilities' object")
    probs = raw["probabilities"]
    if not probs:
        raise InvalidAnswer("answer's probabilities are empty")
    strange = sorted(str(k) for k in probs if k not in labels)
    if strange:
        raise InvalidAnswer(f"labels outside the task's options: {', '.join(strange)}")
    pairs = [(str(label), _finite_unit(p, f"p({label})")) for label, p in probs.items()]
    total = sum(p for _, p in pairs)
    if abs(total - 1.0) > 0.05:
        raise InvalidAnswer(f"probabilities sum to {total:.3f}, not 1")
    return sorted(((label, p / total) for label, p in pairs), key=lambda item: -item[1])


class RolloutJudge:
    """`evaluate(task_name, snapshot)` -> a `StructuredDecision`, whatever the provider does.

    Provider-agnostic on purpose: which provider answers is configuration, and the decision
    records it, so two providers (or Jev and a person) can be scored on the same snapshots.
    """

    def __init__(
        self,
        provider: JudgmentProvider,
        config: JudgeConfig | None = None,
        *,
        rules: Mapping[str, Rule] | None = None,
        applies: Mapping[str, Callable[[FeatureSnapshot], bool]] | None = None,
    ):
        self.provider = provider
        self.config = config or JudgeConfig(provider=provider.name)
        self.rules = dict(rules or {})
        self.applies = dict(applies or {})

    def route(self, decision: StructuredDecision) -> str | None:
        if decision.source == "not_applicable":
            return None
        if decision.source == "fallback" or decision.confidence is None:
            return "human_review"
        if decision.confidence >= self.config.auto_accept_at:
            return "auto_accept"
        if decision.confidence < self.config.ambiguous_below:
            return "ambiguous"
        return "human_review"

    def _decision(self, task: TaskSpec, **kwargs: Any) -> StructuredDecision:
        decision = StructuredDecision(
            task_name=task.name,
            model=self.provider.model,
            provider=self.provider.name,
            **{"alternatives": [], "confidence": None, "decision": None, **kwargs},
        )
        decision.route = self.route(decision)
        return decision

    def _fallback(self, task: TaskSpec, status: str, error: str, latency_ms: float | None = None) -> StructuredDecision:
        return self._decision(
            task, decision=task.fallback, source="fallback", status=status, error=error[:500], latency_ms=latency_ms
        )

    def evaluate(self, task_name: str, snapshot: FeatureSnapshot) -> StructuredDecision:
        task = TASKS.get(task_name)
        if task is None:
            # Not a TaskSpec to fall back on, so this one is the caller's bug and says so.
            raise KeyError(f"unknown judgment task {task_name!r}; known: {', '.join(TASKS)}")
        # A span task asks about one takeover, an episode task about the whole rollout.
        if (task.scope == "span") != (snapshot.expert_span_id is not None):
            return self._decision(task, source="not_applicable", status="not_applicable")
        try:
            if task_name in self.applies and not self.applies[task_name](snapshot):
                return self._decision(task, source="not_applicable", status="not_applicable")
            rule = self.rules.get(task_name)
            if self.config.use_rules and rule is not None:
                label = rule(snapshot)
                if label is not None:
                    return self._decision(task, decision=label, confidence=1.0, source="rule")
        except Exception as exc:  # noqa: BLE001 -- a broken rule must not stop a batch
            return self._fallback(task, "rule_error", f"{type(exc).__name__}: {exc}")

        ok, why = self.provider.available()
        if not ok:
            return self._fallback(task, "unavailable", why)
        payload = snapshot.payload()
        if task.kind == CHOICE:
            ask = lambda: self.provider.choice(task=task, options=task.options, features=payload)  # noqa: E731
        elif task.kind == YES_NO:
            ask = lambda: self.provider.yes_no(task=task, features=payload)  # noqa: E731
        else:
            ask = lambda: self.provider.score(task=task, features=payload)  # noqa: E731

        started = time.perf_counter()
        try:
            raw = _call_with_timeout(ask, self.config.timeout_s)
        except TimeoutError as exc:
            return self._fallback(task, "timeout", str(exc), (time.perf_counter() - started) * 1000.0)
        except Exception as exc:  # noqa: BLE001 -- any provider failure is a fallback, by contract
            return self._fallback(task, "error", f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000.0)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)

        try:
            if task.kind == SCORE:
                if not isinstance(raw, dict):
                    raise InvalidAnswer("answer is not an object")
                score = _finite_unit(raw.get("score"), "score")
                confidence = _finite_unit(raw.get("confidence", 1.0), "confidence")
                return self._decision(task, score=round(score, 4), confidence=round(confidence, 4), latency_ms=latency_ms)
            ranked = validate_distribution(raw, task.labels)
        except InvalidAnswer as exc:
            return self._fallback(task, "invalid", str(exc), latency_ms)
        (label, p), rest = ranked[0], ranked[1 : 1 + self.config.top_alternatives]
        return self._decision(
            task,
            decision=label,
            confidence=round(p, 4),
            alternatives=[[alt, round(q, 4)] for alt, q in rest],
            latency_ms=latency_ms,
        )


# ------------------------------------------------------------------------------------- logging ---


def decision_record(snapshot: FeatureSnapshot, decision: StructuredDecision, provider: JudgmentProvider) -> dict[str, Any]:
    """The logged line: enough to redo the confusion matrix without re-running anything."""

    return {
        "timestamp": _now_iso(),
        "episode_id": snapshot.episode_id,
        "expert_span_id": snapshot.expert_span_id,
        "source_file": snapshot.source,
        "task_name": decision.task_name,
        "decision": decision.decision,
        "score": decision.score,
        "confidence": decision.confidence,
        "alternatives": decision.alternatives,
        "route": decision.route,
        "decided_by": decision.source,
        "status": decision.status,
        "error": decision.error,
        "latency_ms": decision.latency_ms,
        "provider": decision.provider,
        "model": decision.model,
        "provider_version": getattr(provider, "version", None),
        "schema_version": decision.schema_version,
        "features_sha256": snapshot.sha256,
        "features": snapshot.features,
        "missing": list(snapshot.missing),
        # People's answers about the episode, for the benchmark. Never part of the provider input.
        "human": snapshot.labels,
    }


class DecisionLog:
    """Append-only JSONL, one `decision_record` per line, flushed per line so a killed batch
    keeps everything it decided."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
