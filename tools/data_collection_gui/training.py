"""Training-run backend for the data-collection GUI.

Three things the Training page needs and the rest of the gateway does not have:

1. **A machine to train on.** Recording is pinned to the rig the arm is attached to;
   training is not, and a 4090 sitting on another desk is the usual answer. A host is
   either this one or an ``user@host`` reached over ssh.
2. **Code on that machine.** A remote host trains from *its* checkout, so the run has to
   be preceded by an incremental sync -- otherwise "I fixed the exporter" and "the
   training I just launched" refer to different code, and nothing says so.
3. **A W&B key that is not in the repo.** Stored per host, 0600, never returned to the
   browser and never placed on a command line.

Kept out of gateway.py, which is already 11k lines, and structured so the local and
remote paths differ only in how a command is wrapped -- a remote-only bug in the run
lifecycle would otherwise be invisible until someone picked a remote host.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCAL_HOST_ID = "local"
PROBE_SCRIPT = Path("tools/fr3/probe_training_machine.py")
TRAIN_SCRIPT = Path("tools/fr3/fr3_train_il_policy.py")
# Where a host's W&B key lives. Outside the repo tree on purpose: `outputs/` is excluded
# from the sync, but a secret one `git add -A` away from a commit is a bad shape even so.
SECRETS_DIR = Path.home() / ".config" / "lerobot-gui"
SSH_OPTS = ("-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")
# `user@host` or `user@host:port`. Anything else is rejected rather than passed to ssh:
# this string is used to build commands, so it is the one input that must not be able to
# carry shell metacharacters.
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")


class TrainingError(RuntimeError):
    """Something the operator can fix, reported as a 4xx rather than a traceback."""


@dataclass
class TrainingHost:
    id: str
    label: str
    kind: str  # local | remote
    sshTarget: str = ""
    repoDir: str = ""
    pythonPath: str = ".venv-fr3/bin/python"


@dataclass
class TrainingRunStatus:
    state: str = "idle"  # idle | syncing | starting | running | complete | error | stopped
    hostId: str = ""
    hostLabel: str = ""
    viewName: str = ""
    viewRoot: str = ""
    policy: str = ""
    jobName: str = ""
    outputDir: str = ""
    step: int = 0
    totalSteps: int = 0
    loss: float | None = None
    message: str = "Pick a training view and a machine to start."
    pid: int | None = None
    startedAt: str = ""
    finishedAt: str = ""
    logPath: str = ""
    wandbUrl: str = ""
    wandbEnabled: bool = False
    lastLines: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_ssh_target(target: str) -> str:
    target = (target or "").strip()
    if not SSH_TARGET_RE.match(target):
        raise TrainingError(
            f"Training host must look like user@host (got {target!r}). "
            "Names are used to build ssh commands, so anything else is refused."
        )
    return target


def validate_remote_dir(repo_dir: str) -> str:
    repo_dir = (repo_dir or "").strip()
    if not repo_dir.startswith("/"):
        raise TrainingError(f"Remote repo directory must be an absolute path (got {repo_dir!r}).")
    if any(ch in repo_dir for ch in "'\"\\$`\n"):
        raise TrainingError("Remote repo directory must not contain shell metacharacters.")
    return repo_dir.rstrip("/")


# --------------------------------------------------------------------- hosts ---


def host_id_for(ssh_target: str, repo_dir: str) -> str:
    return f"{ssh_target}:{repo_dir}"


def local_host(repo_root: Path) -> TrainingHost:
    return TrainingHost(
        id=LOCAL_HOST_ID,
        label=f"This machine ({os.uname().nodename})",
        kind="local",
        repoDir=str(repo_root),
    )


def hosts_store_path() -> Path:
    return SECRETS_DIR / "training_hosts.json"


def load_remote_hosts() -> list[TrainingHost]:
    path = hosts_store_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    hosts: list[TrainingHost] = []
    for item in raw.get("hosts", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            ssh_target = validate_ssh_target(str(item.get("sshTarget", "")))
            repo_dir = validate_remote_dir(str(item.get("repoDir", "")))
        except TrainingError:
            continue
        hosts.append(
            TrainingHost(
                id=host_id_for(ssh_target, repo_dir),
                label=str(item.get("label") or ssh_target),
                kind="remote",
                sshTarget=ssh_target,
                repoDir=repo_dir,
                pythonPath=str(item.get("pythonPath") or ".venv-fr3/bin/python"),
            )
        )
    return hosts


def save_remote_hosts(hosts: list[TrainingHost]) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "hosts": [asdict(host) for host in hosts]}
    path = hosts_store_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.chmod(0o600)


def add_remote_host(label: str, ssh_target: str, repo_dir: str, python_path: str) -> TrainingHost:
    ssh_target = validate_ssh_target(ssh_target)
    repo_dir = validate_remote_dir(repo_dir)
    host = TrainingHost(
        id=host_id_for(ssh_target, repo_dir),
        label=(label or "").strip() or ssh_target,
        kind="remote",
        sshTarget=ssh_target,
        repoDir=repo_dir,
        pythonPath=(python_path or "").strip() or ".venv-fr3/bin/python",
    )
    hosts = [existing for existing in load_remote_hosts() if existing.id != host.id]
    hosts.append(host)
    save_remote_hosts(hosts)
    return host


def remove_remote_host(host_id: str) -> None:
    save_remote_hosts([host for host in load_remote_hosts() if host.id != host_id])


def all_hosts(repo_root: Path) -> list[TrainingHost]:
    return [local_host(repo_root), *load_remote_hosts()]


def resolve_host(repo_root: Path, host_id: str | None) -> TrainingHost:
    wanted = (host_id or LOCAL_HOST_ID).strip() or LOCAL_HOST_ID
    for host in all_hosts(repo_root):
        if host.id == wanted:
            return host
    raise TrainingError(f"Unknown training host {wanted!r}. Add it on the Training page first.")


# ----------------------------------------------------------- machine probe ---


def probe_machine(repo_root: Path, host: TrainingHost, timeout_s: float = 45.0) -> dict[str, Any]:
    """Run probe_training_machine.py on `host` and return its report.

    The remote path pipes the script over stdin rather than assuming the target already
    has this checkout: the probe's whole job is to answer questions you ask *before*
    syncing, including "is there a repo there at all".
    """
    script = repo_root / PROBE_SCRIPT
    if not script.is_file():
        return {"ok": False, "error": f"probe script missing: {script}"}

    if host.kind == "local":
        command = [_local_python(repo_root), str(script), str(repo_root)]
        stdin_data = None
    else:
        # `python3 - <repo_dir>` reads the program from stdin; the interpreter is the
        # host's configured venv when it exists and plain python3 when it does not, so a
        # machine that has not been set up yet still reports its GPU instead of failing.
        remote = (
            f"cd {shlex.quote(host.repoDir)} 2>/dev/null || cd /; "
            f"if [ -x {shlex.quote(host.repoDir)}/{shlex.quote(host.pythonPath)} ]; then "
            f"  exec {shlex.quote(host.repoDir)}/{shlex.quote(host.pythonPath)} - {shlex.quote(host.repoDir)}; "
            f"else exec python3 - {shlex.quote(host.repoDir)}; fi"
        )
        command = ["ssh", *SSH_OPTS, host.sshTarget, remote]
        stdin_data = script.read_text(encoding="utf-8")

    try:
        result = subprocess.run(
            command,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"probe timed out after {timeout_s:.0f}s"}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return {
            "ok": False,
            "error": f"probe exited {result.returncode}",
            "detail": detail[-4:],
        }
    try:
        # The interpreter's own warnings can precede the JSON; take the last line that parses.
        for line in reversed(result.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        raise json.JSONDecodeError("no JSON object in probe output", result.stdout, 0)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"probe output was not JSON: {exc}"}


def _local_python(repo_root: Path) -> str:
    for candidate in (".venv-fr3/bin/python", ".venv/bin/python"):
        path = repo_root / candidate
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return "python3"


# ------------------------------------------------------------------- wandb ---


def wandb_key_path(host_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", host_id or LOCAL_HOST_ID)
    return SECRETS_DIR / f"wandb_{safe}.key"


def set_wandb_key(host_id: str, key: str) -> None:
    key = (key or "").strip()
    if not key:
        raise TrainingError("W&B API key is empty.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", key):
        raise TrainingError(
            "That does not look like a W&B API key (expected 20-128 characters of "
            "letters, digits, '-' or '_')."
        )
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    path = wandb_key_path(host_id)
    # Created 0600 before anything is written, so the key is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key)


def clear_wandb_key(host_id: str) -> None:
    wandb_key_path(host_id).unlink(missing_ok=True)


def read_wandb_key(host_id: str) -> str:
    path = wandb_key_path(host_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def wandb_status(host_id: str) -> dict[str, Any]:
    """What the browser is allowed to know: that a key exists, and its last 4 characters.

    The suffix is there so an operator can tell *which* key is stored without the page
    ever having held the key itself.
    """
    key = read_wandb_key(host_id)
    return {
        "configured": bool(key),
        "keySuffix": key[-4:] if key else "",
        "hostId": host_id,
    }


# -------------------------------------------------------------------- sync ---


def sync_repo_to_host(repo_root: Path, host: TrainingHost, timeout_s: float = 600.0) -> dict[str, Any]:
    """Incrementally replace the repo on a remote training host.

    Delegates to run/sync_to_target.sh so the exclude list -- which is what stops a sync
    from deleting the target's .venv-fr3 or its recorded outputs/ -- has one definition
    shared with the deploy path.
    """
    if host.kind == "local":
        return {"ok": True, "skipped": True, "message": "Local host trains from this checkout."}

    script = repo_root / "run" / "sync_to_target.sh"
    if not script.is_file():
        raise TrainingError(f"sync script missing: {script}")
    env = {**os.environ, "REMOTE_DIR": host.repoDir}
    try:
        result = subprocess.run(
            ["bash", str(script), host.sshTarget],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise TrainingError(f"Sync to {host.sshTarget} timed out after {timeout_s:.0f}s.") from None
    except OSError as exc:
        raise TrainingError(f"Sync to {host.sshTarget} failed to start: {exc}") from exc

    changed = [
        line
        for line in result.stdout.splitlines()
        if line[:1] in ("<", ">", "c", "*") and " " in line
    ]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-6:]
        return {"ok": False, "message": f"rsync exited {result.returncode}", "detail": detail}
    return {
        "ok": True,
        "changedCount": len(changed),
        "changed": changed[-40:],
        "message": f"Synced {len(changed)} changed file(s) to {host.sshTarget}:{host.repoDir}",
    }


# --------------------------------------------------------------- run command ---


def build_train_argv(
    *,
    host: TrainingHost,
    view_root: str,
    repo_id: str,
    job_name: str,
    policy: str,
    steps: int,
    batch_size: int,
    num_workers: int,
    save_freq: int,
    log_freq: int,
    device: str,
    use_amp: bool,
    policy_config: str,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
) -> list[str]:
    """The training argv, identical in shape for local and remote.

    ``--skip-prepare`` is the whole point: the view was built and QC gated by the export
    step, so this run trains those frames rather than deriving new ones. It also means the
    source recording does not have to exist on this machine -- only the view does, which is
    what makes a remote training host possible at all.

    The cameras, state keys and action contract are deliberately not passed. They are
    properties of frames that already exist, the view's manifest carries them, and a second
    copy on this command line could only ever be a way for the two to disagree.
    """
    python_path = host.pythonPath if host.kind == "remote" else "{python}"
    argv = [
        python_path,
        str(TRAIN_SCRIPT),
        "--view-root", view_root,
        "--skip-prepare",
        "--repo-id", repo_id,
        "--job-name", job_name,
        "--policy", policy,
        "--steps", str(int(steps)),
        "--batch-size", str(int(batch_size)),
        "--num-workers", str(int(num_workers)),
        "--save-freq", str(int(save_freq)),
        "--log-freq", str(int(log_freq)),
        "--device", device,
    ]
    if use_amp:
        argv.append("--use-amp")
    if policy_config.strip():
        argv += ["--policy-config", policy_config.strip()]
    if wandb_enabled:
        argv.append("--wandb")
        if wandb_project.strip():
            argv += ["--wandb-project", wandb_project.strip()]
        if wandb_entity.strip():
            argv += ["--wandb-entity", wandb_entity.strip()]
    return argv


def build_launch_command(
    repo_root: Path, host: TrainingHost, argv: list[str], *, wandb_key: str
) -> tuple[list[str], dict[str, str] | None]:
    """Turn the training argv into something Popen can run, local or over ssh.

    The W&B key never appears in either argv. Locally it goes in the child's environment;
    remotely it is written to a 0600 file on the target by a separate ssh call and read
    back by the shell that starts training, because a key spelled out in an ssh command
    line is visible in the target's process list to every user on that machine.
    """
    if host.kind == "local":
        env = {**os.environ, "PYTHONPATH": "src:.", "PYTHONUNBUFFERED": "1"}
        if wandb_key:
            env["WANDB_API_KEY"] = wandb_key
        local_argv = [_local_python(repo_root) if item == "{python}" else item for item in argv]
        return local_argv, env

    remote_argv = " ".join(shlex.quote(item) for item in argv)
    key_file = f"{host.repoDir}/.wandb_key"
    key_prefix = (
        f"if [ -r {shlex.quote(key_file)} ]; then WANDB_API_KEY=$(cat {shlex.quote(key_file)}); "
        f"export WANDB_API_KEY; fi; "
        if wandb_key
        else ""
    )
    remote = (
        f"cd {shlex.quote(host.repoDir)} && "
        f"{key_prefix}"
        f"export PYTHONPATH=src:. PYTHONUNBUFFERED=1 && "
        f"exec {remote_argv}"
    )
    return ["ssh", *SSH_OPTS, host.sshTarget, remote], None


def push_wandb_key(host: TrainingHost, key: str) -> None:
    """Place the key on a remote target as a 0600 file, over stdin.

    stdin rather than an argument for the same reason as above: arguments are readable
    from the target's process list while the command runs.
    """
    if host.kind == "local" or not key:
        return
    key_file = f"{host.repoDir}/.wandb_key"
    remote = f"umask 077 && cat > {shlex.quote(key_file)}"
    result = subprocess.run(
        ["ssh", *SSH_OPTS, host.sshTarget, remote],
        input=key,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise TrainingError(
            f"Could not place the W&B key on {host.sshTarget}: {result.stderr.strip()[:200]}"
        )


# ------------------------------------------------------------- log parsing ---

# lerobot_train reports progress two ways, and they are not equally good.
#
# Its own log line formats the step through `format_big_number`, which rounds: at step 1300
# it prints `step:1K`. That is fine for a human reading a log and useless as a progress
# number -- the page would sit on "1" for a thousand steps and then jump.
#
# The tqdm bar it writes alongside carries the exact counter, `1300/20000`, and updates
# every step. So the step comes from the bar and the loss from the log line, which is the
# only place it appears at all.
_TQDM_STEP_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_STEP_RE = re.compile(r"\bstep[:=]?\s*(\d[\d,]*\.?\d*)([KMBTQ]?)", re.IGNORECASE)
_LOSS_RE = re.compile(r"\bloss[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
_WANDB_URL_RE = re.compile(r"(https://(?:\w[\w.-]*\.)?wandb\.ai/\S+)")
# A tqdm bar with nothing after it. tqdm writes without a trailing newline, so the next real
# log line arrives glued to the end of one of these -- which is why this anchors on the end of
# the line rather than the start. Matched on tqdm's own `N/M [ ... ]` shape rather than on
# "ends with a bracket", so an ordinary log line that happens to end in one is still shown.
_BARE_PROGRESS_RE = re.compile(r"\d+/\d+\s*\[[^\]]*\]\s*$")
_PROGRESS_PREFIX_RE = re.compile(r"^.*?\d+/\d+\s*\[[^\]]*\]")

_SUFFIX_SCALE = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 10**12, "Q": 10**15}


def is_progress_bar_noise(line: str) -> bool:
    """True for a tqdm bar carrying no message of its own.

    Kept out of the log tail the page shows: at ~13 steps/s these arrive faster than
    anything else and would push every real line out of a 40-line window within seconds.
    Progress is still read off them first -- see `parse_progress_line`.
    """
    return bool(_BARE_PROGRESS_RE.search(line.strip()))


def strip_progress_prefix(line: str) -> str:
    """Drop the tqdm bar a real log line arrived glued to."""
    return _PROGRESS_PREFIX_RE.sub("", line).strip() or line.strip()


def parse_progress_line(line: str) -> dict[str, Any]:
    """Extract whatever this line says about progress. Absent fields stay absent."""
    found: dict[str, Any] = {}

    bar = _TQDM_STEP_RE.search(line)
    if bar:
        found["step"] = int(bar.group(1))
        found["totalSteps"] = int(bar.group(2))
    else:
        step = _STEP_RE.search(line)
        if step:
            try:
                # Rounded by format_big_number, so this is a fallback, not the truth: `1K`
                # is any step from 500 to 1499.
                scale = _SUFFIX_SCALE[step.group(2).upper()]
                found["step"] = int(float(step.group(1).replace(",", "")) * scale)
            except (ValueError, KeyError):
                pass

    loss = _LOSS_RE.search(line)
    if loss:
        try:
            found["loss"] = float(loss.group(1))
        except ValueError:
            pass
    url = _WANDB_URL_RE.search(line)
    if url:
        found["wandbUrl"] = url.group(1).rstrip(").,")
    return found
