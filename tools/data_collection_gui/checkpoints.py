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

try:  # pragma: no cover - same fallback as above
    from tools.data_collection_gui import task_ladders
except ImportError:  # pragma: no cover - direct-script fallback, as elsewhere in this package
    import task_ladders

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


def remove_checkpoint_dir(repo_root: Path, checkpoint_id: str) -> int:
    """Delete one local checkpoint directory; return the bytes it freed.

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
    return freed


def delete_checkpoint(repo_root: Path, checkpoint_id: str) -> dict[str, Any]:
    """Remove one local checkpoint directory, freeing its weights and optimizer state."""
    freed = remove_checkpoint_dir(repo_root, checkpoint_id)
    return {
        "ok": True,
        "checkpointId": checkpoint_id,
        "freedBytes": freed,
        "message": f"Deleted {checkpoint_id} and freed {freed / 1e6:.0f} MB.",
    }


def delete_checkpoints(repo_root: Path, checkpoint_ids: list[str]) -> dict[str, Any]:
    """Remove several checkpoints, reporting each one's fate rather than stopping at the first.

    Deliberately not all-or-nothing. The directories are independent, deleting one cannot make
    deleting the next wrong, and there is nothing to roll back to once bytes are gone -- so a
    batch that hits one bad id should still free the other nine and say which one it skipped.
    Aborting instead would leave the operator to work out by hand which half went.

    `ok` is False only when nothing at all was deleted, so a partly-successful batch reports the
    space it freed *and* the ids it could not touch, instead of one hiding the other.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in checkpoint_ids:
        candidate = str(raw).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    if not ordered:
        raise CheckpointError("No checkpoints were selected for deletion.")

    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    freed = 0
    for checkpoint_id in ordered:
        try:
            freed += remove_checkpoint_dir(repo_root, checkpoint_id)
        except (CheckpointError, OSError) as exc:
            failed.append({"checkpointId": checkpoint_id, "error": str(exc)})
            continue
        deleted.append(checkpoint_id)

    if deleted and failed:
        message = (
            f"Deleted {len(deleted)} checkpoint(s) and freed {freed / 1e6:.0f} MB; "
            f"{len(failed)} could not be deleted."
        )
    elif deleted:
        message = f"Deleted {len(deleted)} checkpoint(s) and freed {freed / 1e6:.0f} MB."
    else:
        message = f"Deleted nothing: all {len(failed)} selected checkpoint(s) failed."
    return {
        "ok": bool(deleted),
        "deleted": deleted,
        "failed": failed,
        "freedBytes": freed,
        "message": message,
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
    # Graded first, because on a ladder the outcome is not an independent field: success is
    # reaching the terminal stage. A record carrying both is a record whose two halves can
    # disagree, which is how a partially inserted peg ended up filed as a plain `failure`.
    try:
        ladder = (
            task_ladders.find_ladder(repo_root, str(record["taskLadder"]))
            if record.get("taskLadder")
            else None
        )
        graded = task_ladders.normalize_grade(record, ladder)
    except task_ladders.LadderError as error:
        raise CheckpointError(str(error)) from error

    outcome = str(graded.get("outcome") or record.get("outcome") or "").strip()
    if outcome not in ROLLOUT_OUTCOMES:
        raise CheckpointError(f"Outcome must be one of {', '.join(ROLLOUT_OUTCOMES)} (got {outcome!r}).")
    checkpoint_id = str(record.get("checkpointId") or "")
    validate_checkpoint_id(checkpoint_id)

    geometry = record.get("geometry")
    entry = {
        "recordedAt": _now(),
        "checkpointId": checkpoint_id,
        "outcome": outcome,
        "mode": str(record.get("mode") or ""),
        "steps": int(record.get("steps") or 0),
        "note": str(record.get("note") or "")[:2000],
        "logPath": str(record.get("logPath") or ""),
        "rolloutIndex": int(record.get("rolloutIndex") or 0),
    }
    # Absent on an ungraded rollout rather than defaulted: stage 0 is a real grade ("never
    # reached the object"), so writing it for "nobody said" would invent evidence.
    for key in ("taskLadder", "stage", "stageId", "terminalStage", "blocker", "inDistribution"):
        if key in graded:
            entry[key] = graded[key]
    # Stored with the grade rather than in a file of its own: a landing point without an outcome
    # is a dot with no meaning, and an outcome without a landing point cannot be put on a map.
    # Written only when the runtime actually reported one, so an older log line -- or a rollout
    # that produced no samples -- leaves the field absent instead of claiming the origin.
    if isinstance(geometry, dict) and geometry:
        entry["geometry"] = _sanitize_rollout_geometry(geometry)
    path = rollout_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


_GEOMETRY_POINT_FIELDS = ("graspXyz", "releaseXyz", "approachXyz")
_GEOMETRY_SCALAR_FIELDS = ("apexZ", "liftM", "descentM")


def _sanitize_rollout_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields this log is allowed to grow, and nothing a caller invented."""
    clean: dict[str, Any] = {}
    for key in _GEOMETRY_POINT_FIELDS:
        value = geometry.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                clean[key] = [round(float(component), 5) for component in value]
            except (TypeError, ValueError):
                continue
    for key in _GEOMETRY_SCALAR_FIELDS:
        value = geometry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[key] = round(float(value), 5)
    for key in ("samples", "heldSteps"):
        value = geometry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
    if "closed" in geometry:
        clean["closed"] = bool(geometry.get("closed"))
    return clean


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


# ------------------------------------------------------- demonstration landmarks ---

# The same rule the runtime applies to a rollout, applied to the demonstrations, so the two sets
# of points on the plot mean the same thing. Keyed on the commanded gripper for the same reason:
# in this dataset the observed width reads 0 on nearly half the frames while the command holds a
# clean 1.0.
_DEMO_GRIPPER_CLOSED_BELOW = 0.5

_demo_landmark_cache: dict[tuple[str, float], dict[str, Any]] = {}


def _feature_index(info: dict[str, Any], feature: str, name: str) -> int | None:
    names = (((info.get("features") or {}).get(feature) or {}).get("names")) or []
    if isinstance(names, list) and name in names:
        return names.index(name)
    return None


def demo_landing_points(dataset_root: Path) -> dict[str, Any]:
    """Where the demonstrations grasped and released, as the backdrop for rollout landing points.

    A rollout landing point on its own says nothing: the question every one of them is asked is
    whether it fell inside the region the demonstrations covered, and that region has to be drawn
    from the same dataset the checkpoint was trained on or the comparison is to a different task.

    Reduced here rather than shipped as a fixture because the dataset is what the checkpoint was
    trained on and it changes; a hard-coded ring would keep describing an older one. Memoised on
    the dataset's own metadata timestamp, since the scan costs a full pass over its parquet.
    """
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    try:
        cache_key = (str(root.resolve()), info_path.stat().st_mtime)
    except OSError:
        return {}
    cached = _demo_landmark_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    state_indices = [_feature_index(info, "observation.state", name) for name in ("ee.x", "ee.y", "ee.z")]
    action_names = (((info.get("features") or {}).get("action") or {}).get("names")) or []
    gripper_index = action_names.index("gripper.pos") if "gripper.pos" in action_names else None
    if any(index is None for index in state_indices) or gripper_index is None:
        return {}

    episodes: dict[int, list[tuple[Any, Any]]] = {}
    try:
        for parquet_path in sorted(root.glob("data/**/*.parquet")):
            table = pq.read_table(
                str(parquet_path), columns=["episode_index", "observation.state", "action"]
            ).to_pydict()
            episode_column = np.asarray(table["episode_index"])
            states = np.asarray([np.asarray(row) for row in table["observation.state"]], dtype=np.float64)
            actions = np.asarray([np.asarray(row) for row in table["action"]], dtype=np.float64)
            for episode in sorted({int(value) for value in episode_column}):
                mask = episode_column == episode
                episodes.setdefault(episode, []).append((states[mask], actions[mask]))
    except (OSError, KeyError, ValueError):
        return {}
    if not episodes:
        return {}

    points: list[dict[str, Any]] = []
    releases: list[list[float]] = []
    for episode in sorted(episodes):
        state = np.concatenate([chunk[0] for chunk in episodes[episode]])
        action = np.concatenate([chunk[1] for chunk in episodes[episode]])
        position = state[:, state_indices]
        gripper = action[:, gripper_index]
        # Falling edge, matching the runtime: an episode that starts with the gripper already
        # commanded shut would otherwise report its grasp at the start pose. The demonstrations
        # start open, so this changes none of them -- it keeps the two point sets on this plot
        # derived by one rule rather than by two that happen to agree today.
        open_steps = np.flatnonzero(gripper >= _DEMO_GRIPPER_CLOSED_BELOW)
        if not open_steps.size:
            continue
        first_open = int(open_steps[0])
        closed_steps = np.flatnonzero(gripper[first_open:] < _DEMO_GRIPPER_CLOSED_BELOW) + first_open
        if not closed_steps.size:
            continue
        close_idx = int(closed_steps[0])
        reopened = np.flatnonzero(gripper[close_idx:] >= _DEMO_GRIPPER_CLOSED_BELOW)
        release_idx = int(close_idx + reopened[0]) if reopened.size else len(gripper) - 1
        apex_z = float(position[close_idx : release_idx + 1, 2].max())
        points.append(
            {
                "episode": int(episode),
                "graspXyz": [round(float(value), 5) for value in position[close_idx]],
                "releaseXyz": [round(float(value), 5) for value in position[release_idx]],
                "liftM": round(apex_z - float(position[close_idx, 2]), 5),
                "descentM": round(apex_z - float(position[release_idx, 2]), 5),
            }
        )
        releases.append([float(position[release_idx, 0]), float(position[release_idx, 1])])

    if not points:
        return {}
    # The demonstrations all release into the same hole, so their mean release point *is* the
    # hole -- measured rather than configured, which keeps it correct if the fixture is moved.
    hole = [
        round(float(np.mean([release[0] for release in releases])), 5),
        round(float(np.mean([release[1] for release in releases])), 5),
    ]
    radii = [
        float(np.hypot(point["graspXyz"][0] - hole[0], point["graspXyz"][1] - hole[1]))
        for point in points
    ]
    landmarks = {
        "datasetRoot": str(root),
        "hole": hole,
        "points": points,
        "graspRadiusM": {
            "min": round(min(radii), 5),
            "max": round(max(radii), 5),
            "mean": round(float(np.mean(radii)), 5),
        },
    }
    _demo_landmark_cache.clear()
    _demo_landmark_cache[cache_key] = landmarks
    return landmarks


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
