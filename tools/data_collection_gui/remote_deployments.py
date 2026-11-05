"""Policies served by an upstream `lerobot-policy-server` (Zenoh), rolled out through the R4 bridge.

A remote deployment has no weights on this machine: the model and its processors live on the
policy server, and the R4 runtime reaches it through `tools/fr3_wam/run_r4_jepa.sh` in the
experiment checkout (`bridgeRepo`), which runs this repo's launcher unchanged with a bridge policy
in place of the loaded one (see `r4_jepa_entry.py` there). Every R4 guard still applies.

Deployments are registered per machine in `outputs/remote_deployments.json`; nothing is
registered by default, so a host without the file lists no remote entries:

    {"deployments": [{
        "id": "remote/fr3-jepa-j1-s0",            # <job>/<step>, shown as a checkpoint
        "label": "VLA-JEPA J1 s0 011844 (Dante 5090)",
        "endpoint": "tcp/10.77.0.64:8791",
        "deployment": "fr3-jepa-j1-s0",
        "policyType": "vla_jepa",
        "chunkSize": 7,
        "datasetRoot": "outputs/exports/training_views/<the view it was trained on>",
        "targetFrameName": "pika_gripper_ee",
        "robotIp": "192.168.1.206",
        "bridgeRepo": "/data/delta/wam-p3/lerobot-main-1029",
        "outRoot": "/data/delta/wam-p3/out"
    }]}

Each entry is listed with the checkpoints and judged by the same `check_contract`, so a view that
is missing or a tool frame that differs from the rig blocks it like any other checkpoint.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

from . import checkpoints as checkpoint_backend

REGISTRY = Path("outputs/remote_deployments.json")
BRIDGE_LAUNCHER = Path("tools/fr3_wam/run_r4_jepa.sh")
JOB = "remote"
# Launcher modes the bridge supports. The grasp loop and the DAgger rehearsal drive the runtime
# in ways the bridge refuses (takeover, scripted segments) or does not need a policy at all.
MODES = ("env", "smoke", "preview", "real_once", "real", "real_debug")
# Smoke runs one step on the in-process path. Over the bridge the first chunk needs ~3 ticks to
# arrive and those ticks hold, so one step would only ever show a hold; this many shows the
# policy's own actions while still sending nothing.
SMOKE_STEPS = 15
REQUIRED = ("id", "endpoint", "deployment", "datasetRoot", "bridgeRepo", "outRoot")


class RemoteDeploymentError(checkpoint_backend.CheckpointError):
    """A registry entry the operator has to fix; reported like any checkpoint error."""


def load(repo_root: Path) -> list[dict[str, Any]]:
    """The registered deployments, each validated; [] without a registry."""
    path = repo_root / REGISTRY
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteDeploymentError(f"{path}: {exc}") from exc
    entries = raw.get("deployments") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise RemoteDeploymentError(f'{path} must hold {{"deployments": [...]}}')
    deployments = []
    for entry in entries:
        missing = [key for key in REQUIRED if not str((entry or {}).get(key) or "").strip()]
        if missing:
            raise RemoteDeploymentError(f"{path}: deployment {entry!r} is missing {missing}")
        job, _ = checkpoint_backend.validate_checkpoint_id(str(entry["id"]))
        if job != JOB:
            raise RemoteDeploymentError(
                f"{path}: remote deployment ids are {JOB}/<name>, got {entry['id']!r}"
            )
        deployments.append(dict(entry))
    return deployments


def find(repo_root: Path, checkpoint_id: str) -> dict[str, Any] | None:
    """The deployment registered under `checkpoint_id`, if any."""
    return next((d for d in load(repo_root) if d["id"] == checkpoint_id), None)


def is_remote_id(checkpoint_id: str) -> bool:
    """Whether an id belongs to the remote namespace (never a directory under outputs/train)."""
    return str(checkpoint_id or "").startswith(f"{JOB}/")


def endpoint_reachable(endpoint: str, timeout_s: float = 0.5) -> tuple[bool, str]:
    """A TCP connect to the Zenoh listener: answers "is the server up", not "is it free"."""
    proto, _, address = str(endpoint).partition("/")
    host, _, port = address.rpartition(":")
    if proto != "tcp" or not host or not port.isdigit():
        return False, f"not a tcp/<host>:<port> endpoint: {endpoint!r}"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True, "listening"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def entry(repo_root: Path, deployment: dict[str, Any]) -> dict[str, Any]:
    """A `Checkpoint`-shaped listing entry, for `check_contract` and the page."""
    from tools.fr3.scan_checkpoints import describe_view

    view_root = deployment["datasetRoot"]
    absolute = view_root if os.path.isabs(view_root) else str(repo_root / view_root)
    view = describe_view(absolute) or {"root": absolute, "exists": False}
    launcher = Path(deployment["bridgeRepo"]) / BRIDGE_LAUNCHER
    reachable, detail = endpoint_reachable(deployment["endpoint"])
    job, step_label = deployment["id"].split("/", 1)
    return {
        "id": deployment["id"],
        "jobName": job,
        "stepLabel": step_label,
        "step": 0,
        "isLast": False,
        "aliasOf": "",
        "path": deployment["endpoint"],
        "pretrainedPath": "",
        "policyType": str(deployment.get("policyType") or "remote"),
        "chunkSize": deployment.get("chunkSize"),
        "nActionSteps": None,
        "cameras": list(view.get("cameras") or []),
        "totalSteps": 0,
        "datasetRepoId": os.path.basename(view_root.rstrip("/")),
        "datasetRoot": view_root,
        "recordedDatasetRoot": view_root,
        "sizeBytes": 0,
        "modifiedAt": 0,
        "view": view,
        "contract": {
            "robotIp": str(deployment.get("robotIp") or ""),
            "targetFrameName": str(deployment.get("targetFrameName") or ""),
            "cameraKeys": list(view.get("cameras") or []),
            "policy": str(deployment.get("policyType") or ""),
        },
        "inferenceConfigPath": "",
        "remote": {
            "label": str(deployment.get("label") or deployment["deployment"]),
            "endpoint": deployment["endpoint"],
            "deployment": deployment["deployment"],
            "bridgeRepo": deployment["bridgeRepo"],
            "bridgeReady": launcher.is_file(),
            "reachable": reachable,
            "detail": detail,
        },
    }


def extra_issues(listed: dict[str, Any]) -> list[checkpoint_backend.ContractIssue]:
    """What blocks a remote entry beyond the checkpoint contract: the bridge and the server."""
    remote = listed.get("remote") or {}
    issues = []
    if not remote.get("bridgeReady"):
        issues.append(
            checkpoint_backend.ContractIssue(
                "block",
                "bridge",
                f"The R4 bridge launcher is not at {remote.get('bridgeRepo')}/{BRIDGE_LAUNCHER}; "
                "sync the experiment checkout to this machine first.",
            )
        )
    if not remote.get("reachable"):
        issues.append(
            checkpoint_backend.ContractIssue(
                "warn",
                "server",
                f"Policy server {remote.get('endpoint')} is not listening ({remote.get('detail')}). "
                "Start it on the GPU host before Start, or the rollout fails at connect.",
            )
        )
    return issues


def rollout_command(
    command: list[str], env: dict[str, str], *, repo_root: Path, deployment: dict[str, Any], mode: str
) -> tuple[list[str], dict[str, str]]:
    """Turn `build_rollout_command`'s launcher invocation into the bridge's.

    The bridge script runs this repo's launcher itself, with the same mode and the same trailing
    arguments, so everything `build_rollout_command` decided (environment, preview and trace
    directories, live frames) carries over; only the script and the bridge settings change.
    """
    if mode not in MODES:
        raise RemoteDeploymentError(
            f"{deployment['id']} is served remotely; mode {mode!r} is not supported over the bridge "
            f"(supported: {', '.join(MODES)})."
        )
    if env.get("FR3_DAGGER_TAKEOVER") == "1":
        raise RemoteDeploymentError("DAgger takeover is not supported over the bridge; turn it off.")
    if command[:3] != ["bash", str(repo_root / "tools/fr3/run_pick_place_infer_workstation.sh"), mode]:
        raise RemoteDeploymentError(f"unexpected launcher invocation {command[:3]!r}")
    launcher = Path(deployment["bridgeRepo"]) / BRIDGE_LAUNCHER
    if not launcher.is_file():
        raise RemoteDeploymentError(f"bridge launcher missing: {launcher}")
    extra = list(command[3:])
    if mode == "smoke":
        # Appended last, so it overrides the launcher's own `--max-steps 1` (argparse keeps the last).
        extra += ["--max-steps", str(SMOKE_STEPS)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(env)
    env.update(
        {
            "FR3_WAM_WORKSTATION": str(repo_root),
            "FR3_WAM_ENDPOINT": str(deployment["endpoint"]),
            "FR3_WAM_OUT": str(Path(deployment["outRoot"]) / f"r4_{mode}_{stamp}"),
            "FR3_INFER_CHECKPOINT": str(deployment["id"]),
            "FR3_INFER_DATASET_ROOT": str(deployment["datasetRoot"]),
        }
    )
    env.pop("FR3_POLICY_SERVER", None)
    return ["bash", str(launcher), mode, *extra], env
