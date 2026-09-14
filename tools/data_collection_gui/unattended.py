"""Runs that outlive the page watching them: start, witness, brake.

Two loops on this rig run for hours with nobody in the room -- `terminal_trials` (E6-lite) and
`auto_collect` (E6-data) -- and they are the same shape. Each expands a seeded schedule before the
arm moves, walks it emitting one row per unit of work, names the condition it stopped on, and
parks holding the peg. What differs is what a unit of work is. So they get one page, not two, and
this module is the contract that lets them share it.

**The state lives on disk, and that is the whole design.** `rollout.py` does the opposite -- it
owns a subprocess and accumulates status in gateway memory from the lines it parses -- and that is
right there, because a rollout is a person driving a policy and the page *is* the session. It is
wrong here. An unattended run must survive the page being closed, the laptop sleeping, the gateway
being restarted, and the ssh connection dropping, because those are ordinary events over eight
hours and none of them is a reason to stop a robot mid-experiment. So the process is detached
(`start_new_session`), everything a reader needs is a file in the run's own directory, and
re-attaching after a gateway restart is not a feature that had to be built -- it is what reading a
directory already does.

**Which means the gateway can be wrong about a run, and says so rather than guessing.** A run whose
process is gone and whose rows end in a summary is `complete`. A run whose process is gone and
whose rows do *not* is `crashed`, and those two must never be collapsed: the second one is a night
that stopped at 2 a.m. for a reason nobody has read yet, and it looks exactly like the first on any
status that only counts rows.

**Two brakes, because they cost different things.** Touching `STOP` ends the run at the next
boundary -- the current trial or cycle finishes, the arm parks holding the peg, and the next run
can start from that state. A signal stops it where it stands, which is the right answer when
something is wrong and the wrong answer otherwise, because the peg ends up somewhere the loop's
invariant does not cover. The page should present them as different actions and not as one button
with a modifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable


class UnattendedError(RuntimeError):
    """A run that must not be started, or a request that cannot be honoured."""


# Where runs live. One directory per run, named by the run id, holding everything a reader needs.
UNATTENDED_ROOT = Path("outputs") / "unattended"
# How many rows the status endpoint returns by default. A night is tens of thousands; the page
# needs the recent ones and the counts, not the night.
DEFAULT_TAIL = 200


@dataclass(frozen=True)
class RunKind:
    """One tenant of the page: how to plan it, how to launch it, how to read its rows.

    `plan` returns the expanded schedule and the QC result *without touching the robot*, which is
    what makes "authorise" mean something: the plan a person approves is the plan that runs, and
    it is checked against the fence before anybody is asked to approve it.
    """

    id: str
    label: str
    script: str
    # request dict -> (plan dict, argv tail). Raises on anything that must not run.
    plan: Callable[[Path, dict[str, Any], Path], tuple[dict[str, Any], list[str]]]
    # What one row of work is called, for the page's own labels.
    unit: str = "cycle"


def _require(request: dict[str, Any], key: str) -> Any:
    if key not in request or request[key] in (None, ""):
        raise UnattendedError(f"{key} is required.")
    return request[key]


def _plan_terminal_trials(
    repo_root: Path, request: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    from tools.fr3.terminal_servo import TerminalServoRequest, parse_terminal_servo_pose
    from tools.fr3.terminal_trials import (
        TerminalTrialsRequest,
        build_trial_schedule,
        describe_schedule,
        validate_terminal_trials,
    )
    from tools.fr3.workspace_fence import resolve_workspace_fence

    hole_pose = str(_require(request, "holePose"))
    offsets = str(request.get("offsetsMm") or "0,2,3,4,5,6,8,10")
    servo = TerminalServoRequest(
        xyz=parse_terminal_servo_pose(hole_pose),
        handoffZ=float(request.get("handoffZ") or 0.12),
        requestId="",
    )
    pick = str(request.get("pickPose") or "0.3640,-0.1370,0.0550").strip()
    trials_request = TerminalTrialsRequest(
        servo=servo,
        offsetsMm=tuple(float(part) for part in offsets.split(",") if part.strip()),
        repeats=int(request.get("repeats") or 6),
        controlEvery=int(request.get("controlEvery") or 4),
        seed=int(request.get("seed") or 0),
        searchRingM=float(request.get("searchRingM") or 0.0),
        maxSeconds=float(request.get("maxSeconds") or 0.0),
        pickXyz=parse_terminal_servo_pose(pick) if pick else None,
        requestId=run_dir.name,
    )
    schedule = build_trial_schedule(trials_request)
    workspace_min, workspace_max, fence_source = resolve_workspace_fence(
        record_config_path=str(request.get("recordConfig") or "tools/fr3/fr3_record_config.yaml")
    )
    qc = validate_terminal_trials(
        trials_request, schedule, workspace_min=workspace_min, workspace_max=workspace_max
    )
    plan = {
        "kind": "terminal_trials",
        "request": trials_request.payload(),
        "schedule": [vars(spec) for spec in schedule],
        "qc": qc,
        "fence": {"min": list(workspace_min), "max": list(workspace_max), "source": fence_source},
        "text": describe_schedule(trials_request, schedule),
        "units": len(schedule),
    }
    argv = [
        "tools/fr3/fr3_terminal_trials_runtime.py",
        "--hole-pose", hole_pose,
        "--offsets-mm", offsets,
        "--repeats", str(trials_request.repeats),
        "--control-every", str(trials_request.controlEvery),
        "--seed", str(trials_request.seed),
        "--search-ring", str(trials_request.searchRingM),
        "--handoff-z", str(servo.handoffZ),
        "--max-seconds", str(trials_request.maxSeconds),
        "--pick-pose", pick,
        "--out", str(run_dir / "rows.jsonl"),
        "--stop-file", str(run_dir / "STOP"),
    ]
    return plan, argv


def _plan_auto_collect(
    repo_root: Path, request: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    from tools.fr3.auto_collect import (
        AutoCollectRequest,
        build_collection_schedule,
        describe_schedule,
        validate_auto_collection,
    )
    from tools.fr3.fr3_auto_collect_runtime import load_mask
    from tools.fr3.workspace_fence import resolve_workspace_fence

    mask_path = str(request.get("mask") or "outputs/metrology/scene_reset_mask.json")
    collect_request = AutoCollectRequest(
        maskStrokes=load_mask(mask_path),
        pickXyz=None,
        placeZ=float(request.get("placeZ") or 0.0550),
        carryZ=float(request.get("carryZ") or 0.1500),
        cycles=int(request.get("cycles") or 50),
        seed=int(request.get("seed") or 0),
        recoveryFraction=float(request.get("recoveryFraction") or 0.0),
        maxSeconds=float(request.get("maxSeconds") or 0.0),
        requestId=run_dir.name,
    )
    schedule = build_collection_schedule(collect_request)
    workspace_min, workspace_max, fence_source = resolve_workspace_fence(
        record_config_path=str(request.get("recordConfig") or "tools/fr3/fr3_record_config.yaml")
    )
    qc = validate_auto_collection(
        collect_request, schedule, workspace_min=workspace_min, workspace_max=workspace_max
    )
    plan = {
        "kind": "auto_collect",
        "request": collect_request.payload(),
        "schedule": [vars(spec) for spec in schedule],
        "qc": qc,
        "fence": {"min": list(workspace_min), "max": list(workspace_max), "source": fence_source},
        "text": describe_schedule(collect_request, schedule),
        "units": len(schedule),
    }
    argv = [
        "tools/fr3/fr3_auto_collect_runtime.py",
        "--mask", mask_path,
        "--place-z", str(collect_request.placeZ),
        "--carry-z", str(collect_request.carryZ),
        "--cycles", str(collect_request.cycles),
        "--seed", str(collect_request.seed),
        "--recovery-fraction", str(collect_request.recoveryFraction),
        "--max-seconds", str(collect_request.maxSeconds),
        "--out", str(run_dir),
        "--stop-file", str(run_dir / "STOP"),
    ]
    return plan, argv


RUN_KINDS: dict[str, RunKind] = {
    kind.id: kind
    for kind in (
        RunKind(
            id="terminal_trials",
            label="E6-lite: terminal trials",
            script="tools/fr3/fr3_terminal_trials_runtime.py",
            plan=_plan_terminal_trials,
            unit="trial",
        ),
        RunKind(
            id="auto_collect",
            label="E6-data: auto collection",
            script="tools/fr3/fr3_auto_collect_runtime.py",
            plan=_plan_auto_collect,
            unit="cycle",
        ),
    )
}


def runs_root(repo_root: Path) -> Path:
    return Path(repo_root) / UNATTENDED_ROOT


def process_alive(pid: int | None) -> bool:
    """Whether the detached run is still running. Signal 0 asks without touching it."""

    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive and owned by somebody else. Reporting it dead would let a second run start
        # against the same arm, which is the one mistake this check exists to prevent.
        return True
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_rows(path: Path, *, tail: int) -> tuple[list[dict[str, Any]], int]:
    """The last `tail` rows and the total count.

    Read line by line rather than parsed whole: a run in flight is being appended to, and the last
    line can be half-written. A row that does not parse is skipped rather than raising, because the
    page must keep rendering while the run continues.
    """

    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    total = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(rows) > tail:
                    rows.pop(0)
    except OSError:
        return [], 0
    return rows, total


def read_run(repo_root: Path, run_id: str, *, tail: int = DEFAULT_TAIL) -> dict[str, Any]:
    """Everything about one run, derived from its directory and nothing else."""

    run_dir = runs_root(repo_root) / run_id
    if not run_dir.is_dir():
        raise UnattendedError(f"no such run: {run_id}")
    meta = _read_json(run_dir / "run.json")
    plan = _read_json(run_dir / "plan.json")
    rows, total = _read_rows(run_dir / "rows.jsonl", tail=tail)

    summary = next((row for row in reversed(rows) if row.get("kind") == "summary"), None)
    units = [row for row in rows if row.get("kind") in ("trial", "cycle") and row.get("ok", True)]
    alive = process_alive(meta.get("pid"))
    last_row_at = meta.get("startedAt", 0.0)
    try:
        last_row_at = (run_dir / "rows.jsonl").stat().st_mtime
    except OSError:
        pass

    if alive:
        state = "running"
    elif summary is not None:
        state = "complete" if summary.get("ok") else "halted"
    elif total > 0:
        # The process is gone and no summary was ever written. Never folded into "complete":
        # this is a night that stopped for a reason nobody has read, and on a row count alone it
        # is indistinguishable from one that finished.
        state = "crashed"
    else:
        state = "starting" if meta.get("pid") else "planned"

    return {
        "id": run_id,
        "kind": meta.get("kind", plan.get("kind", "")),
        "dir": str(run_dir),
        "state": state,
        "pid": meta.get("pid"),
        "alive": alive,
        "startedAt": meta.get("startedAt"),
        "argv": meta.get("argv", []),
        "logPath": str(run_dir / "run.log"),
        "stopRequested": (run_dir / "STOP").exists(),
        "stopReason": _stop_reason(run_dir),
        "plan": plan,
        "rows": rows,
        "rowsTotal": total,
        "unitsDone": len(units),
        "unitsPlanned": plan.get("units", 0),
        "summary": summary,
        "lastRowAgeS": max(0.0, time.time() - last_row_at) if total else None,
    }


def _stop_reason(run_dir: Path) -> str:
    try:
        return (run_dir / "STOP").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def list_runs(repo_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """Newest first, with just enough of each to draw a row."""

    root = runs_root(repo_root)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        try:
            run = read_run(repo_root, run_dir.name, tail=1)
        except UnattendedError:
            continue
        out.append(
            {
                key: run[key]
                for key in ("id", "kind", "state", "startedAt", "unitsDone", "unitsPlanned",
                            "stopRequested", "alive", "lastRowAgeS")
            }
        )
        if len(out) >= limit:
            break
    return out


def active_run(repo_root: Path) -> dict[str, Any] | None:
    """The one run that is still moving the arm, if any.

    One at a time is not a policy choice, it is the rig: there is one arm. `start_run` refuses a
    second, and this is what it asks.
    """

    for entry in list_runs(repo_root, limit=200):
        if entry["state"] in ("running", "starting"):
            return entry
    return None


def plan_run(repo_root: Path, kind_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Expand and check a plan without writing anything or moving anything.

    This is what "authorise" means here: the schedule a person reads is the schedule that runs,
    and it has already been walked against the fence when they read it.
    """

    kind = RUN_KINDS.get(str(kind_id))
    if kind is None:
        raise UnattendedError(f"unknown run kind: {kind_id}")
    run_dir = runs_root(repo_root) / _new_run_id(kind.id)
    plan, argv = kind.plan(Path(repo_root), dict(request or {}), run_dir)
    return {"kind": kind.id, "label": kind.label, "unit": kind.unit, "plan": plan, "argv": argv}


def _new_run_id(kind_id: str) -> str:
    return f"{kind_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def start_run(
    repo_root: Path,
    kind_id: str,
    request: dict[str, Any],
    *,
    python: str = "",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write the plan, then detach the process that will execute it.

    The plan is written *before* the process starts, so a run that dies in its first second still
    leaves behind what it was going to do. The process is detached with `start_new_session`, which
    is `setsid`: closing the page, restarting the gateway, or losing the ssh connection that
    started the gateway must not stop a robot mid-experiment.
    """

    repo_root = Path(repo_root)
    kind = RUN_KINDS.get(str(kind_id))
    if kind is None:
        raise UnattendedError(f"unknown run kind: {kind_id}")
    running = active_run(repo_root)
    if running is not None:
        raise UnattendedError(
            f"{running['id']} is still {running['state']} and there is one arm. Stop it first."
        )

    run_dir = runs_root(repo_root) / _new_run_id(kind.id)
    run_dir.mkdir(parents=True, exist_ok=False)
    plan, argv = kind.plan(repo_root, dict(request or {}), run_dir)
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    interpreter = python or _default_python(repo_root)
    command = [interpreter, "-u", *argv]
    log_handle = (run_dir / "run.log").open("ab")
    process_env = dict(os.environ)
    process_env.setdefault("PYTHONPATH", str(repo_root / "src"))
    if env:
        process_env.update(env)
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is built here, not taken from the request
            command,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=process_env,
        )
    finally:
        log_handle.close()

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "id": run_dir.name,
                "kind": kind.id,
                "pid": process.pid,
                "startedAt": time.time(),
                "argv": command,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return read_run(repo_root, run_dir.name)


def _default_python(repo_root: Path) -> str:
    """The rig's own interpreter when there is one, this process's otherwise."""

    import sys

    candidate = Path(repo_root) / ".venv-fr3" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def request_stop(repo_root: Path, run_id: str, *, mode: str = "boundary") -> dict[str, Any]:
    """Brake. `boundary` finishes the unit in flight; `now` stops where it stands.

    `now` is a signal rather than a file because it has to work when the loop is inside a motion
    and not between two of them -- which is exactly when the file would not be read for another
    minute, and exactly when somebody pressing it does not have a minute.
    """

    run_dir = runs_root(repo_root) / run_id
    if not run_dir.is_dir():
        raise UnattendedError(f"no such run: {run_id}")
    meta = _read_json(run_dir / "run.json")
    if mode == "boundary":
        (run_dir / "STOP").write_text(f"requested at {time.strftime('%H:%M:%S')}", encoding="utf-8")
    elif mode == "now":
        pid = meta.get("pid")
        if not process_alive(pid):
            raise UnattendedError("the run is not running.")
        # SIGINT, not SIGTERM: the loops catch KeyboardInterrupt, park the arm holding whatever
        # they hold, and write their summary. SIGTERM would kill them between two motions with
        # the peg in the air and nothing written down.
        os.kill(int(pid), signal.SIGINT)
    else:
        raise UnattendedError(f"unknown stop mode: {mode}")
    return read_run(repo_root, run_id, tail=1)


def release_brake(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Undo a boundary stop that has not been acted on yet."""

    run_dir = runs_root(repo_root) / run_id
    if not run_dir.is_dir():
        raise UnattendedError(f"no such run: {run_id}")
    (run_dir / "STOP").unlink(missing_ok=True)
    return read_run(repo_root, run_id, tail=1)
