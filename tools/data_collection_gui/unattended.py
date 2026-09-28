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
LOG_TAIL_LINES = 30


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
        graspAttempts=int(request.get("graspAttempts") or 1),
        regripInPlace=_flag(request, "regripInPlace"),
        releaseOnlyWhenSeated=_flag(request, "releaseOnlyWhenSeated"),
        operatorGrade=_flag(request, "operatorGrade"),
        updateReference=not _flag(request, "fixedHole"),
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
        "homeFirst": _flag(request, "homeFirst"),
    }
    # Every value goes as --flag=value: an offsets list starting "-12,..." given as its own argv
    # word is read by argparse as another flag, and the run dies before its first row.
    argv = [
        "tools/fr3/fr3_terminal_trials_runtime.py",
        f"--hole-pose={hole_pose}",
        f"--offsets-mm={offsets}",
        f"--repeats={trials_request.repeats}",
        f"--control-every={trials_request.controlEvery}",
        f"--seed={trials_request.seed}",
        f"--search-ring={trials_request.searchRingM}",
        f"--handoff-z={servo.handoffZ}",
        f"--max-seconds={trials_request.maxSeconds}",
        f"--pick-pose={pick}",
        f"--grasp-attempts={trials_request.graspAttempts}",
        f"--out={run_dir / 'rows.jsonl'}",
        f"--stop-file={run_dir / 'STOP'}",
    ]
    if trials_request.regripInPlace:
        argv.append("--regrip-in-place")
    if trials_request.releaseOnlyWhenSeated:
        argv.append("--release-only-when-seated")
    if trials_request.operatorGrade:
        argv += ["--operator-grade", f"--grade-file={run_dir / 'GRADE'}"]
    if not trials_request.updateReference:
        argv.append("--fixed-hole")
    # The run inherits the wrist from wherever the arm was left, and refuses to start leaning;
    # home is level. 09-28: the run after a wedged peg found the tool 16.2 deg off.
    if _flag(request, "homeFirst"):
        argv.append("--home-first")
    return plan, argv


def _flag(request: dict[str, Any], key: str) -> bool:
    """A yes/no field typed into a text box: 1/true/yes/on, anything else is no."""

    return str(request.get(key) or "").strip().lower() in ("1", "true", "yes", "on", "y")


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
        f"--mask={mask_path}",
        f"--place-z={collect_request.placeZ}",
        f"--carry-z={collect_request.carryZ}",
        f"--cycles={collect_request.cycles}",
        f"--seed={collect_request.seed}",
        f"--recovery-fraction={collect_request.recoveryFraction}",
        f"--max-seconds={collect_request.maxSeconds}",
        f"--out={run_dir}",
        f"--stop-file={run_dir / 'STOP'}",
    ]
    return plan, argv


def _plan_grasp_envelope(
    repo_root: Path, request: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    from tools.fr3.grasp_envelope import (
        GRASP_ENVELOPE_SPOT_XY,
        GraspEnvelopeRequest,
        build_envelope_schedule,
        describe_schedule,
        done_indices,
        parse_offsets_mm,
        parse_points_mm,
        parse_xy,
        read_rows,
        validate_grasp_envelope,
    )
    from tools.fr3.terminal_servo import parse_terminal_servo_pose
    from tools.fr3.workspace_fence import resolve_workspace_fence

    spot = str(request.get("spot") or ",".join(str(v) for v in GRASP_ENVELOPE_SPOT_XY))
    # Present-but-empty means "skip this block", so a fine scan can run its extra points alone;
    # only a key that is absent altogether takes the coarse default.
    xy_offsets = str(request.get("xyOffsetsMm", "-12,-8,-4,0,4,8,12") or "")
    dz_offsets = str(request.get("dzOffsetsMm", "-12,-8,-4,0,4,8,14,20,28") or "")
    extra = str(request.get("extraPointsMm") or "")
    pick = str(request.get("pickPose") or "0.3640,-0.1370,0.0550").strip()
    start = str(request.get("start") or "fixture")
    envelope_request = GraspEnvelopeRequest(
        spotXy=parse_xy(spot),
        xyOffsetsMm=parse_offsets_mm(xy_offsets),
        xyDzMm=float(request.get("xyDzMm") or -6.0),
        dzOffsetsMm=parse_offsets_mm(dz_offsets),
        centreRepeats=int(request.get("centreRepeats") if request.get("centreRepeats") not in (None, "") else 5),
        extraPointsMm=parse_points_mm(extra),
        repeats=int(request.get("repeats") or 1),
        seed=int(request.get("seed") or 0),
        start=start,
        pickXyz=parse_terminal_servo_pose(pick),
        maxSeconds=float(request.get("maxSeconds") or 0.0),
        requestId=run_dir.name,
    )
    schedule = build_envelope_schedule(envelope_request)
    workspace_min, workspace_max, fence_source = resolve_workspace_fence(
        record_config_path=str(request.get("recordConfig") or "tools/fr3/fr3_record_config.yaml")
    )
    qc = validate_grasp_envelope(
        envelope_request, schedule, workspace_min=workspace_min, workspace_max=workspace_max
    )
    text = describe_schedule(envelope_request, schedule)
    resume_from = str(request.get("resumeFrom") or "").strip()
    resume_rows = ""
    if resume_from:
        # A run that halted on a lost peg or a fault is finished by a second run with the same
        # plan: same seed, same order, the finished indices skipped. The peg is wherever the
        # first run left it, which is why `start` is still the operator's to set.
        previous = runs_root(repo_root) / resume_from / "rows.jsonl"
        if not previous.exists():
            raise UnattendedError(f"resumeFrom: no rows at {previous}")
        done = done_indices(read_rows(previous))
        schedule = [point for point in schedule if point.index not in done]
        resume_rows = str(previous)
        text += f"\nresuming {resume_from}: {len(done)} done, {len(schedule)} remaining"
    plan = {
        "kind": "grasp_envelope",
        "request": envelope_request.payload(),
        "schedule": [vars(point) for point in schedule],
        "qc": qc,
        "fence": {"min": list(workspace_min), "max": list(workspace_max), "source": fence_source},
        "text": text,
        "units": len(schedule),
    }
    argv = [
        "tools/fr3/fr3_grasp_envelope_runtime.py",
        f"--spot={spot}",
        f"--xy-offsets-mm={xy_offsets}",
        f"--xy-dz-mm={envelope_request.xyDzMm}",
        f"--dz-offsets-mm={dz_offsets}",
        f"--centre-repeats={envelope_request.centreRepeats}",
        f"--repeats={envelope_request.repeats}",
        f"--extra-points-mm={extra}",
        f"--seed={envelope_request.seed}",
        f"--start={start}",
        f"--pick-pose={pick}",
        f"--max-seconds={envelope_request.maxSeconds}",
        "--home-first",
        f"--out={run_dir / 'rows.jsonl'}",
        f"--stop-file={run_dir / 'STOP'}",
        f"--continue-file={run_dir / 'CONTINUE'}",
    ]
    if resume_rows:
        argv += [f"--resume-rows={resume_rows}"]
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
            id="grasp_envelope",
            label="P0: grasp envelope",
            script="tools/fr3/fr3_grasp_envelope_runtime.py",
            plan=_plan_grasp_envelope,
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
    # A run this gateway launched and that has exited is a zombie until somebody waits on it, and
    # signal 0 succeeds on a zombie. Without the reap a run that died on its first line reads as
    # "running" for as long as the gateway lives.
    try:
        reaped, _ = os.waitpid(int(pid), os.WNOHANG)
        if reaped:
            return False
    except ChildProcessError:
        pass  # not our child: launched by an earlier gateway, which is the normal case after a restart
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass
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
    elif total > 0 or meta.get("pid"):
        # The process is gone and no summary was ever written. Never folded into "complete":
        # this is a night that stopped for a reason nobody has read, and on a row count alone it
        # is indistinguishable from one that finished.
        state = "crashed"
    else:
        state = "planned"

    # Why a run died is in its log, and a crashed run whose page does not show it is one somebody
    # has to ssh in to read.
    log_tail: list[str] = []
    if state == "crashed":
        try:
            log_tail = (run_dir / "run.log").read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
        except OSError:
            pass

    # A run waiting on a person says so. Only while it is alive: a needs_operator row followed by
    # nothing is a run that gave up waiting, and asking somebody to put a peg back for a process
    # that is gone would be a button that does nothing.
    needs_operator = ""
    needs_grade = False
    for row in reversed(rows):
        kind = row.get("kind")
        if kind == "needs_operator":
            needs_operator = str(row.get("message") or "the run needs a person") if alive else ""
            # Not "put the peg back" but "is it in the hole?": two different buttons.
            needs_grade = bool(row.get("grade")) and alive
            break
        if kind in ("operator", "trial", "summary", "staged"):
            break

    return {
        "id": run_id,
        "kind": meta.get("kind", plan.get("kind", "")),
        "needsOperator": needs_operator,
        "needsGrade": needs_grade,
        "dir": str(run_dir),
        "state": state,
        "pid": meta.get("pid"),
        "alive": alive,
        "startedAt": meta.get("startedAt"),
        "argv": meta.get("argv", []),
        "logPath": str(run_dir / "run.log"),
        "logTail": log_tail,
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
        #
        # A second press on a run that is still there is SIGKILL. 09-28 16:40: after a reflex a
        # run hung inside the robot driver's teardown, where SIGINT is never seen, and the page
        # had no way left to end it.
        marker = run_dir / "HALT"
        if marker.exists():
            os.kill(int(pid), signal.SIGKILL)
        else:
            marker.write_text(f"SIGINT at {time.strftime('%H:%M:%S')}", encoding="utf-8")
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


def request_grade(repo_root: Path, run_id: str, grade: str) -> dict[str, Any]:
    """Answer a graded run's question: the peg is "in" the hole or "out" of it."""

    run_dir = runs_root(repo_root) / run_id
    if not run_dir.is_dir():
        raise UnattendedError(f"no such run: {run_id}")
    answer = str(grade or "").strip().lower()
    if answer not in ("in", "out"):
        raise UnattendedError("grade must be 'in' or 'out'.")
    run = read_run(repo_root, run_id, tail=20)
    if not run["needsGrade"]:
        raise UnattendedError("the run is not asking for a grade.")
    (run_dir / "GRADE").write_text(answer, encoding="utf-8")
    return read_run(repo_root, run_id, tail=1)


def request_continue(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Answer a run that is waiting for a person: the peg is back, carry on."""

    run_dir = runs_root(repo_root) / run_id
    if not run_dir.is_dir():
        raise UnattendedError(f"no such run: {run_id}")
    run = read_run(repo_root, run_id, tail=20)
    if not run["needsOperator"] or run["needsGrade"]:
        raise UnattendedError("the run is not waiting for anybody.")
    (run_dir / "CONTINUE").write_text(f"continued at {time.strftime('%H:%M:%S')}", encoding="utf-8")
    return read_run(repo_root, run_id, tail=1)
