#!/usr/bin/env python3
"""Report the training checkpoints on a machine, as JSON on stdout.

A checkpoint on disk is not enough to roll out: what makes it usable is the contract it
was trained under -- which cameras, which action encoding, which tool frame the dataset
was anchored to. All of that is recoverable from files the trainer already wrote next to
the weights, so this script reads them rather than asking the operator to remember.

Deliberately stdlib-only and self-contained: the remote path pipes this file over stdin
to a training machine that may not have this repo importable, and a machine whose torch
is broken must still be able to say what it holds.

Usage:
    python3 scan_checkpoints.py [repo_root]
"""

from __future__ import annotations

import json
import os
import sys

# Where lerobot_train writes runs, and where fr3_train_il_policy.py builds views. Relative
# to the repo root because that is the only path both machines agree on.
TRAIN_OUTPUTS_SUBDIR = os.path.join("outputs", "train")
VIEWS_SUBDIR = os.path.join("outputs", "exports", "training_views")

# The weights and the config that names the policy. A directory missing either of these is
# a half-written checkpoint (killed mid-save), not one worth offering for a rollout.
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "config.json")


def read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def dir_size_bytes(root: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                # Not st_size on the link itself: `last` is a symlink to a step directory
                # and reporting 0 bytes for the checkpoint operators actually use is worse
                # than counting its target twice.
                total += os.stat(full).st_size
            except OSError:
                continue
    return total


def camera_keys_from_features(features: dict) -> list[str]:
    prefix = "observation.images."
    return sorted(key[len(prefix):] for key in features if key.startswith(prefix))


def resolve_view_root(recorded_root: str, repo_root: str) -> str:
    """Where this machine actually keeps the view a checkpoint names.

    A checkpoint records its dataset as an absolute path on whatever machine trained it. Fetch
    the checkpoint to a rig whose checkout lives somewhere else and that path is simply gone,
    even though the view was fetched alongside it and is sitting in this repo's views
    directory under the same name.

    Falls back by name, and only to this repo's own views directory -- never by searching, and
    never to a name that does not match. A rollout against the wrong dataset is the silent kind
    of wrong: it would resolve a start pose from someone else's episodes and place the whole
    trajectory there.
    """
    if not recorded_root:
        return ""
    if os.path.isdir(recorded_root):
        return recorded_root
    candidate = os.path.join(repo_root, VIEWS_SUBDIR, os.path.basename(recorded_root.rstrip("/")))
    return candidate if os.path.isdir(candidate) else ""


def describe_view(view_root: str) -> dict:
    """The dataset facts a rollout has to match, read from the view the checkpoint names.

    Returns an empty dict when the view is gone -- a real state, because a view can be
    deleted after training and the checkpoint outlives it. The page says so rather than
    inventing defaults, since a guessed action mode is exactly the kind of silent mismatch
    that runs and is wrong.
    """
    if not view_root or not os.path.isdir(view_root):
        return {}
    info = read_json(os.path.join(view_root, "meta", "info.json"))
    manifest = read_json(os.path.join(view_root, "meta", "il_view_manifest.json"))
    if not info:
        return {}
    return {
        "root": view_root,
        "exists": True,
        "relocated": False,
        "fps": int(info.get("fps") or 0),
        "episodes": int(info.get("total_episodes") or 0),
        "frames": int(info.get("total_frames") or 0),
        "cameras": camera_keys_from_features(info.get("features") or {}),
        "actionMode": str(manifest.get("action_mode") or ""),
        "stateKeys": list(manifest.get("state_keys") or []),
        "sourceFps": manifest.get("source_fps") or {},
        "frameStride": manifest.get("frame_stride") or {},
    }


# Cap on the generated inference config we ship back with each checkpoint. The file this
# reads is one fr3_train_il_policy.py wrote and is ~1.5 KB; the limit only exists so a
# hand-edited or truncated file cannot turn a scan into a megabyte response.
INFERENCE_CONFIG_MAX_BYTES = 64 * 1024


def find_inference_config(view_root: str, job_name: str) -> tuple[str, str]:
    """The rollout contract that belongs to *this* job, with the view-wide file as fallback.

    Runs that train a view someone else built write their generated configs under
    `<view>/runs/<job>/`, because two jobs sharing a view would otherwise overwrite each
    other's. Runs that build their own view leave them at the view root. Preferring the
    per-job copy matters: it is the one whose `checkpoint:` line names this job.

    Returned as text rather than parsed -- this script is stdlib-only so it can be piped to
    a machine that has no pyyaml, and the caller has a real YAML parser.
    """
    if not view_root:
        return "", ""
    candidates = [
        os.path.join(view_root, "runs", job_name, "inference_config.generated.yaml"),
        os.path.join(view_root, "inference_config.generated.yaml"),
    ]
    for path in candidates:
        try:
            if os.path.getsize(path) > INFERENCE_CONFIG_MAX_BYTES:
                continue
            with open(path, "r", encoding="utf-8") as handle:
                return path, handle.read()
        except OSError:
            continue
    return "", ""


def describe_checkpoint(repo_root: str, job_name: str, step_dir: str) -> dict | None:
    pretrained = os.path.join(step_dir, "pretrained_model")
    if not all(os.path.isfile(os.path.join(pretrained, name)) for name in REQUIRED_CHECKPOINT_FILES):
        return None

    policy_config = read_json(os.path.join(pretrained, "config.json"))
    train_config = read_json(os.path.join(pretrained, "train_config.json"))
    dataset = train_config.get("dataset") or {}
    recorded_dataset_root = str(dataset.get("root") or "")
    dataset_root = resolve_view_root(recorded_dataset_root, repo_root)

    step_name = os.path.basename(step_dir)
    # `last` carries no number of its own. Its symlink target does, and that is the only source
    # available for a checkpoint fetched from a training host -- a fetch copies the weights, not
    # the optimizer state that training_step.json sits beside. Falling back to that file keeps
    # the answer right for a run whose `last` was materialized as a real directory.
    if step_name.isdigit():
        step = int(step_name)
    else:
        target_name = os.path.basename(os.path.realpath(step_dir))
        if target_name.isdigit():
            step = int(target_name)
        else:
            training_state = read_json(
                os.path.join(step_dir, "training_state", "training_step.json")
            )
            step = int(training_state.get("step") or 0)

    input_features = policy_config.get("input_features") or {}
    try:
        modified_at = os.stat(pretrained).st_mtime
    except OSError:
        modified_at = 0.0

    # `last` is a symlink onto a numbered step, so its bytes are the same bytes. Naming the
    # step it points at lets the page show one disk figure per run instead of double-counting
    # the newest checkpoint, and lets it mark which numbered step `last` currently means.
    alias_of = ""
    if os.path.islink(step_dir):
        alias_of = os.path.basename(os.path.realpath(step_dir))

    wandb = train_config.get("wandb") or {}
    inference_config_path, inference_config_text = find_inference_config(dataset_root, job_name)
    view = describe_view(dataset_root)
    if view and dataset_root != recorded_dataset_root:
        # Surfaced rather than silently corrected: the operator should be able to see that the
        # dataset being used is not the one the path inside the checkpoint names.
        view["relocated"] = True

    return {
        "id": f"{job_name}/{step_name}",
        "jobName": job_name,
        "stepLabel": step_name,
        "step": step,
        "isLast": step_name == "last",
        "path": step_dir,
        "pretrainedPath": pretrained,
        "policyType": str(policy_config.get("type") or ""),
        "chunkSize": policy_config.get("chunk_size"),
        "nActionSteps": policy_config.get("n_action_steps"),
        "cameras": camera_keys_from_features(input_features),
        "totalSteps": int(train_config.get("steps") or 0),
        "seed": train_config.get("seed"),
        "datasetRepoId": str(dataset.get("repo_id") or ""),
        "datasetRoot": dataset_root or recorded_dataset_root,
        "recordedDatasetRoot": recorded_dataset_root,
        "sizeBytes": dir_size_bytes(pretrained),
        "modifiedAt": modified_at,
        "aliasOf": alias_of,
        "view": view,
        "inferenceConfigPath": inference_config_path,
        "inferenceConfigText": inference_config_text,
        "wandbProject": str(wandb.get("project") or "") if wandb.get("enable") else "",
        "wandbRunId": str(wandb.get("run_id") or "") if wandb.get("enable") else "",
    }


def scan(repo_root: str) -> list[dict]:
    train_root = os.path.join(repo_root, TRAIN_OUTPUTS_SUBDIR)
    if not os.path.isdir(train_root):
        return []
    checkpoints: list[dict] = []
    for job_name in sorted(os.listdir(train_root)):
        checkpoint_root = os.path.join(train_root, job_name, "checkpoints")
        if not os.path.isdir(checkpoint_root):
            continue
        for step_name in sorted(os.listdir(checkpoint_root)):
            step_dir = os.path.join(checkpoint_root, step_name)
            if not os.path.isdir(step_dir):
                continue
            try:
                entry = describe_checkpoint(repo_root, job_name, step_dir)
            except OSError:
                continue
            if entry is not None:
                checkpoints.append(entry)
    # Newest first: an operator picking a checkpoint to roll out almost always wants the one
    # that just finished training, and the alternative orderings (job name, step) bury it.
    checkpoints.sort(key=lambda item: (item["modifiedAt"], item["step"]), reverse=True)
    return checkpoints


def main() -> None:
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    report = {
        "ok": True,
        "repoRoot": repo_root,
        "repoRootExists": os.path.isdir(repo_root),
        "checkpoints": scan(repo_root),
        "viewsRoot": os.path.join(repo_root, VIEWS_SUBDIR),
    }
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
