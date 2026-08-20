"""Training artifacts: what exists, whether it may be rolled out, and how it did.

Three jobs, deliberately kept together because they are the same question asked at three
moments:

  * **Registry** -- which checkpoints exist, on this machine and on every training host.
  * **Contract** -- whether a checkpoint's dataset agrees with the rig it is about to drive.
  * **Record** -- what happened when it did.

The contract half is the reason this module is not just a directory listing. On this rig a
checkpoint trained against `pika_task_tcp` and rolled out against `pika_gripper_ee` does not
fail: both frames exist on the same URDF, 410.85 mm apart, so the arm runs, tracks its
targets, and is wrong by that offset everywhere. Every mismatch this module reports has that
shape -- something that would run rather than raise.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised implicitly by every gateway import
    from tools.data_collection_gui import training as training_backend
except ImportError:  # pragma: no cover - direct-script fallback, as elsewhere in this package
    import training as training_backend

SCAN_SCRIPT = Path("tools/fr3/scan_checkpoints.py")

# `<job>/<step>`, both segments already constrained by what the trainer will accept as a job
# name. Every path this module builds from an id is built by re-joining validated segments
# rather than by trusting the id, so a `..` or an absolute path cannot escape outputs/train.
CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

# The action encoding the real-robot runtime knows how to execute. A view exported under any
# other mode produces a policy whose outputs mean something else -- absolute poses, joint
# targets -- and the runtime would integrate them as deltas.
ROLLOUT_ACTION_MODE = "delta_ee_from_prev_cmd"

TRAIN_OUTPUTS_SUBDIR = Path("outputs") / "train"
ROLLOUT_LOG_PATH = Path("outputs") / "rollouts" / "rollout_log.jsonl"

# Outcomes an operator can record. `aborted` is not a failure: stopping a rollout because a
# person walked into the cell says nothing about the policy, and folding it into the failure
# count would quietly bias every success rate the page shows.
ROLLOUT_OUTCOMES = ("success", "failure", "aborted")


class CheckpointError(RuntimeError):
    """Something the operator can fix, reported as a 4xx rather than a traceback."""


@dataclass
class ContractIssue:
    """One disagreement between a checkpoint and the rig it would drive.

    `level` is `block` when rolling out anyway would move the arm under a contract nobody
    verified, and `warn` when the run would be valid but is worth a second look.
    """

    level: str  # ok | warn | block
    field: str
    message: str


@dataclass
class RigContract:
    """What the rig is today, as the gateway's own config reports it."""

    robotIp: str = ""
    targetFrameName: str = ""
    cameraKeys: list[str] = field(default_factory=list)
    cameraConfigPath: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_checkpoint_id(checkpoint_id: str) -> tuple[str, str]:
    """Split a `<job>/<step>` id into validated segments.

    Returned as segments rather than as a path so callers cannot accidentally use the raw
    string: this id arrives from an HTTP query and is used to build both local paths and
    remote rsync sources.
    """
    checkpoint_id = (checkpoint_id or "").strip()
    if not CHECKPOINT_ID_RE.match(checkpoint_id):
        raise CheckpointError(
            f"Checkpoint id must look like <job>/<step> (got {checkpoint_id!r}). "
            "Ids build filesystem and rsync paths, so anything else is refused."
        )
    job_name, step_label = checkpoint_id.split("/", 1)
    if job_name in (".", "..") or step_label in (".", ".."):
        raise CheckpointError(f"Checkpoint id {checkpoint_id!r} does not name a directory.")
    return job_name, step_label


# ------------------------------------------------------------------ registry ---


def scan_host(
    repo_root: Path, host: training_backend.TrainingHost, timeout_s: float = 90.0
) -> dict[str, Any]:
    """Run scan_checkpoints.py on `host` and return its report.

    Mirrors training.probe_machine: the remote path pipes the script over stdin so a host
    that has never been synced still answers, and the answer is about the venv that would
    actually run the training rather than about this machine.
    """
    script = repo_root / SCAN_SCRIPT
    if not script.is_file():
        return {"ok": False, "error": f"checkpoint scan script missing: {script}", "checkpoints": []}

    if host.kind == "local":
        command = [training_backend._local_python(repo_root), str(script), str(repo_root)]
        stdin_data = None
    else:
        remote = (
            f"cd {shlex.quote(host.repoDir)} 2>/dev/null || cd /; "
            f"if [ -x {shlex.quote(host.repoDir)}/{shlex.quote(host.pythonPath)} ]; then "
            f"  exec {shlex.quote(host.repoDir)}/{shlex.quote(host.pythonPath)} - {shlex.quote(host.repoDir)}; "
            f"else exec python3 - {shlex.quote(host.repoDir)}; fi"
        )
        command = ["ssh", *training_backend.SSH_OPTS, host.sshTarget, remote]
        stdin_data = script.read_text(encoding="utf-8")

    try:
        result = subprocess.run(
            command, input=stdin_data, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"scan timed out after {timeout_s:.0f}s", "checkpoints": []}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "checkpoints": []}

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return {
            "ok": False,
            "error": f"scan exited {result.returncode}",
            "detail": detail[-4:],
            "checkpoints": [],
        }
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"scan output was not JSON: {exc}", "checkpoints": []}
    return {"ok": False, "error": "scan produced no JSON output", "checkpoints": []}


def parse_inference_contract(text: str) -> dict[str, Any]:
    """Pull the rollout-relevant scalars out of a generated inference config.

    The scan script ships the file as text because it is stdlib-only and may be running on a
    machine with no pyyaml; parsing happens here, where there is one.
    """
    if not text.strip():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - a malformed config must not take the page down
        return {}
    if not isinstance(loaded, dict):
        return {}
    runtime = loaded.get("runtime") if isinstance(loaded.get("runtime"), dict) else {}
    hardware = runtime.get("hardware") if isinstance(runtime.get("hardware"), dict) else {}
    safety = runtime.get("safety") if isinstance(runtime.get("safety"), dict) else {}
    training = loaded.get("training") if isinstance(loaded.get("training"), dict) else {}
    return {
        "robotIp": str(hardware.get("robot_ip") or ""),
        "targetFrameName": str(hardware.get("target_frame_name") or ""),
        "gripperBackend": str(hardware.get("gripper_backend") or ""),
        "gripperPort": str(hardware.get("gripper_port") or ""),
        "cameraConfig": str(runtime.get("camera_config") or ""),
        "cameraKeys": [str(key) for key in (training.get("camera_keys") or [])],
        "policy": str(training.get("policy") or ""),
        "safety": {
            "firstFrameMaxPosDeltaMm": safety.get("first_frame_max_pos_delta_mm"),
            "firstFrameMaxRotDeltaDeg": safety.get("first_frame_max_rot_delta_deg"),
            "maxStepPosDeltaMm": safety.get("max_step_pos_delta_mm"),
            "maxLeashPosDeltaMm": safety.get("max_leash_pos_delta_mm"),
            "maxLeashRotDeltaDeg": safety.get("max_leash_rot_delta_deg"),
            "maxStepRotDeltaDeg": safety.get("max_step_rot_delta_deg"),
        },
    }


def check_contract(
    checkpoint: dict[str, Any], *, rig: RigContract, local: bool
) -> list[ContractIssue]:
    """Everything about this checkpoint that would make a rollout wrong rather than failed.

    Ordered by how quietly the mismatch would pass: the tool frame first, because it is the
    one that produces a clean-looking run at a fixed offset, and the missing-view case last,
    because that one does at least raise.
    """
    issues: list[ContractIssue] = []
    contract = checkpoint.get("contract") or {}
    view = checkpoint.get("view") or {}

    trained_frame = str(contract.get("targetFrameName") or "")
    if trained_frame and rig.targetFrameName and trained_frame != rig.targetFrameName:
        issues.append(
            ContractIssue(
                "block",
                "targetFrameName",
                f"Trained against tool frame {trained_frame}, but this rig is configured for "
                f"{rig.targetFrameName}. These frames sit on the same URDF a fixed distance "
                "apart, so a rollout would track its targets and be wrong by that offset "
                "everywhere. Set FR3_TARGET_FRAME_NAME deliberately or retrain.",
            )
        )
    elif not trained_frame:
        issues.append(
            ContractIssue(
                "warn",
                "targetFrameName",
                "No generated inference config was found for this checkpoint, so the tool "
                "frame it was trained against is unknown. The rollout will use the rig's "
                "current frame, which may not be the one the dataset was anchored to.",
            )
        )

    action_mode = str(view.get("actionMode") or "")
    if view.get("exists") and action_mode and action_mode != ROLLOUT_ACTION_MODE:
        issues.append(
            ContractIssue(
                "block",
                "actionMode",
                f"View was exported as {action_mode}; the real-robot runtime integrates "
                f"actions as {ROLLOUT_ACTION_MODE}. Its outputs would be applied as deltas "
                "regardless of what they actually mean.",
            )
        )

    trained_cameras = sorted(str(key) for key in (checkpoint.get("cameras") or []))
    if trained_cameras and rig.cameraKeys and trained_cameras != sorted(rig.cameraKeys):
        issues.append(
            ContractIssue(
                "block",
                "cameras",
                f"Policy expects camera keys {trained_cameras}; the rig's inference camera "
                f"config provides {sorted(rig.cameraKeys)}. A missing key fails at load, and "
                "a swapped pair does not.",
            )
        )

    trained_ip = str(contract.get("robotIp") or "")
    if trained_ip and rig.robotIp and trained_ip != rig.robotIp:
        issues.append(
            ContractIssue(
                "warn",
                "robotIp",
                f"Checkpoint's config names robot {trained_ip}; this rig is {rig.robotIp}. "
                "The rollout drives this rig -- confirm the checkpoint belongs to it.",
            )
        )

    if not local:
        issues.append(
            ContractIssue(
                "block",
                "location",
                "This checkpoint is on a training host. Fetch it to this machine before "
                "rolling it out -- the robot and its cameras are attached here.",
            )
        )
    elif not view.get("exists"):
        issues.append(
            ContractIssue(
                "block",
                "view",
                f"The training view this checkpoint names is not on this machine "
                f"({checkpoint.get('datasetRoot') or 'unknown path'}). The runtime reads the "
                "dataset's episode start poses to place the trajectory in the workspace, so "
                "the rollout cannot start without it.",
            )
        )
    return issues


def verdict_for(issues: list[ContractIssue]) -> str:
    if any(issue.level == "block" for issue in issues):
        return "block"
    if any(issue.level == "warn" for issue in issues):
        return "warn"
    return "ok"


# --------------------------------------------------------------------- fetch ---

# What a rollout needs from the training view, and nothing else. `videos/` is excluded on
# purpose: the runtime never opens it (it reads episode start states out of the parquet), and
# in an exported view those files are symlinks into the source dataset that would not resolve
# on this machine anyway.
VIEW_FETCH_SUBDIRS = ("meta", "data")


def fetch_checkpoint(
    repo_root: Path,
    host: training_backend.TrainingHost,
    checkpoint: dict[str, Any],
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Copy one checkpoint, and the view it needs, from a training host to this machine.

    Only the weights and the view's metadata cross the wire -- roughly 200 MB rather than the
    whole run directory, whose optimizer state is three times the size of the model and is
    useless anywhere except resuming that training.
    """
    if host.kind == "local":
        raise CheckpointError("This checkpoint is already on this machine.")
    job_name, step_label = validate_checkpoint_id(str(checkpoint.get("id") or ""))

    # Fetching `last` stores it under the number it points at. On the training host `last` is a
    # symlink, and its step is readable from the link; a copy is a real directory whose name
    # carries no number and whose training_state (where the trainer records the step) is not
    # fetched. Landing it as `020000` keeps the step recoverable, and means fetching `last`
    # again after more training adds a checkpoint rather than overwriting a different one.
    alias_of = str(checkpoint.get("aliasOf") or "")
    if alias_of:
        _, step_label = validate_checkpoint_id(f"{job_name}/{alias_of}")

    remote_pretrained = str(checkpoint.get("pretrainedPath") or "")
    if not remote_pretrained.startswith("/"):
        raise CheckpointError(
            f"Remote checkpoint path must be absolute (got {remote_pretrained!r})."
        )

    local_step_dir = repo_root / TRAIN_OUTPUTS_SUBDIR / job_name / "checkpoints" / step_label
    local_step_dir.mkdir(parents=True, exist_ok=True)
    transferred: list[str] = []

    # Trailing slash on the source: copy the directory's *contents* into pretrained_model/,
    # so a re-fetch updates in place instead of nesting a second copy inside the first.
    weights = _rsync(
        [f"{host.sshTarget}:{remote_pretrained}/", str(local_step_dir / "pretrained_model") + "/"],
        timeout_s=timeout_s,
    )
    transferred.extend(weights)

    view_root = str((checkpoint.get("view") or {}).get("root") or checkpoint.get("datasetRoot") or "")
    local_view_root = ""
    if view_root.startswith("/"):
        local_view_root = str(repo_root / "outputs" / "exports" / "training_views" / Path(view_root).name)
        for subdir in VIEW_FETCH_SUBDIRS:
            Path(local_view_root, subdir).mkdir(parents=True, exist_ok=True)
            transferred.extend(
                _rsync(
                    [
                        f"{host.sshTarget}:{view_root}/{subdir}/",
                        f"{local_view_root}/{subdir}/",
                    ],
                    timeout_s=timeout_s,
                )
            )

    return {
        "ok": True,
        "checkpointId": f"{job_name}/{step_label}",
        "localPath": str(local_step_dir),
        "localViewRoot": local_view_root,
        "transferredCount": len(transferred),
        "transferred": transferred[-40:],
        "message": (
            f"Fetched {job_name}/{step_label} from {host.sshTarget} "
            f"({len(transferred)} file(s))."
        ),
    }


def _rsync(paths: list[str], *, timeout_s: float) -> list[str]:
    command = [
        "rsync",
        "-az",
        "--itemize-changes",
        "--partial",
        # `last` is a symlink on the training host; copying it as a link would leave a
        # dangling path here, where the step directory it points at may not have been
        # fetched. -L resolves it into the real files.
        "-L",
        "-e",
        " ".join(["ssh", *training_backend.SSH_OPTS]),
        *paths,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise CheckpointError(f"Fetch timed out after {timeout_s:.0f}s.") from None
    except OSError as exc:
        raise CheckpointError(f"Fetch failed to start: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-6:]
        raise CheckpointError(f"rsync exited {result.returncode}: {' | '.join(detail)}")
    return [line for line in result.stdout.splitlines() if line[:1] in ("<", ">", "c", "*")]


def delete_checkpoint(repo_root: Path, checkpoint_id: str) -> dict[str, Any]:
    """Remove one local checkpoint directory, freeing its weights and optimizer state.

    Refuses `last`: it is a symlink the trainer maintains, and deleting it leaves a run whose
    newest checkpoint has no stable name while freeing nothing.
    """
    import shutil

    job_name, step_label = validate_checkpoint_id(checkpoint_id)
    if step_label == "last":
        raise CheckpointError(
            "`last` is a symlink onto a numbered checkpoint, so removing it frees no space. "
            "Delete the step directory it points at instead."
        )
    target = repo_root / TRAIN_OUTPUTS_SUBDIR / job_name / "checkpoints" / step_label
    resolved = target.resolve()
    train_root = (repo_root / TRAIN_OUTPUTS_SUBDIR).resolve()
    try:
        resolved.relative_to(train_root)
    except ValueError:
        raise CheckpointError(f"{checkpoint_id} does not resolve inside {train_root}.") from None
    if not resolved.is_dir():
        raise CheckpointError(f"No checkpoint directory at {target}.")
    freed = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    shutil.rmtree(resolved)
    return {
        "ok": True,
        "checkpointId": checkpoint_id,
        "freedBytes": freed,
        "message": f"Deleted {checkpoint_id} and freed {freed / 1e6:.0f} MB.",
    }


# ---------------------------------------------------------------- rollout log ---


def rollout_log_path(repo_root: Path) -> Path:
    return repo_root / ROLLOUT_LOG_PATH


def append_rollout_outcome(repo_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one rollout result to the log.

    Kept in a single append-only JSONL outside the checkpoint directories on purpose: a
    checkpoint gets deleted to reclaim disk long before its track record stops being the
    reason you would or would not retrain that way, and a record stored beside the weights
    would go with them.
    """
    outcome = str(record.get("outcome") or "").strip()
    if outcome not in ROLLOUT_OUTCOMES:
        raise CheckpointError(f"Outcome must be one of {', '.join(ROLLOUT_OUTCOMES)} (got {outcome!r}).")
    checkpoint_id = str(record.get("checkpointId") or "")
    validate_checkpoint_id(checkpoint_id)

    entry = {
        "recordedAt": _now(),
        "checkpointId": checkpoint_id,
        "outcome": outcome,
        "mode": str(record.get("mode") or ""),
        "steps": int(record.get("steps") or 0),
        "note": str(record.get("note") or "")[:2000],
        "logPath": str(record.get("logPath") or ""),
    }
    path = rollout_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_rollout_outcomes(repo_root: Path, limit: int = 500) -> list[dict[str, Any]]:
    path = rollout_log_path(repo_root)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded = json.loads(line)
                except json.JSONDecodeError:
                    # One corrupt line (a half-written append during a crash) must not hide
                    # every rollout recorded before it.
                    continue
                if isinstance(loaded, dict):
                    entries.append(loaded)
    except OSError:
        return []
    return entries[-limit:]


def outcome_summary(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-checkpoint tallies, keyed by checkpoint id."""
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        checkpoint_id = str(entry.get("checkpointId") or "")
        if not checkpoint_id:
            continue
        bucket = summary.setdefault(
            checkpoint_id, {"success": 0, "failure": 0, "aborted": 0, "total": 0}
        )
        outcome = str(entry.get("outcome") or "")
        if outcome in bucket:
            bucket[outcome] += 1
        bucket["total"] += 1
    return summary
