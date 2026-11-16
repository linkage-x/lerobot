#!/usr/bin/env python
"""Train a policy on QC-gated exports, in the relative-TCP contract, with LeRobot's trainer.

    PYTHONPATH=src:. python tools/fr3/train_relative_tcp.py \
        --export outputs/exports/<name> [--export ...] --hand right \
        --camera cam_10 --camera cam_06:140,220,1180,1000 \
        --policy act --steps 20000 --job-name pick_act_v1

    # resume (same run directory, from its last checkpoint):
    PYTHONPATH=src:. python tools/fr3/train_relative_tcp.py --resume outputs/training/runs/pick_act_v1

What it adds to ``lerobot_train`` (which still does the training, the checkpoints, resume and W&B):

* the exports are checked first (relative_tcp_data.check_exports: QC gate, TCP point, label chains);
* the samples are the relative-TCP ones (relative_tcp.py), from cropped training views of the exports;
* each checkpoint's ``pretrained_model/`` gets ``fr3_policy_contract.json`` (what goes in and out) and
  ``fr3_provenance.json`` (which exports, at which fingerprints, which code, which run, which step);
* after the last checkpoint, it is loaded back from disk and must predict what the trained policy in
  memory predicts on the same samples (``reload_check`` in fr3_provenance.json).

A run directory holds everything: ``run.json`` (arguments, contract, exports, code), ``stats.json`` (the
normaliser's statistics of the relative quantities) and ``lerobot/`` (lerobot_train's output_dir).
Progress goes to stdout as ``FR3_PROGRESS {json}`` lines for the GUI.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.fr3 import relative_tcp as rt  # noqa: E402, I001
from tools.fr3 import relative_tcp_data as rd  # noqa: E402

RUN_SCHEMA = "fr3_training_run/1"
RUN_FILE = "run.json"
STATS_FILE = "stats.json"
DEFAULT_RUNS_ROOT = Path("outputs/training/runs")
DEFAULT_VIEWS_ROOT = Path("outputs/training/views")
PROGRESS_PREFIX = "FR3_PROGRESS "
#: Policy types this script configures. ACT only for now: its chunk prediction is deterministic, so
#: the reload check and inference's offline/online comparison are exact. Diffusion samples noise and
#: keeps an observation queue; both need handling before it is offered.
POLICY_TYPES = ("act",)
CODE_FILES = (
    "tools/fr3/relative_tcp.py",
    "tools/fr3/relative_tcp_data.py",
    "tools/fr3/train_relative_tcp.py",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def progress(**fields: Any) -> None:
    print(PROGRESS_PREFIX + json.dumps(fields), flush=True)


def code_provenance(repo_root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Which code trains: the deployed release (DEPLOYED.json) or the git checkout, and the files' hashes."""
    out: dict[str, Any] = {"files": {}}
    for rel in CODE_FILES:
        path = repo_root / rel
        if path.is_file():
            out["files"][rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    deployed = repo_root / "DEPLOYED.json"
    if deployed.is_file():
        with contextlib.suppress(ValueError):
            out["release"] = json.loads(deployed.read_text(encoding="utf-8"))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, stdin=subprocess.DEVNULL
        ).stdout.strip()
        if commit:
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            ).stdout.strip()
            out["git"] = {"commit": commit, "dirty": bool(dirty)}
    except OSError:
        pass
    return out


def parse_camera(spec: str) -> tuple[str, list[int] | None]:
    """``cam_06`` or ``cam_06:x0,y0,x1,y1``."""
    name, _, crop = spec.partition(":")
    if not name.startswith("cam_"):
        raise argparse.ArgumentTypeError(f"camera {spec!r}: expected cam_NN[:x0,y0,x1,y1]")
    if not crop:
        return name, None
    values = [int(v) for v in crop.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"camera {spec!r}: a crop is x0,y0,x1,y1")
    return name, values


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="append", type=Path, default=[], help="an export (repeatable)")
    ap.add_argument("--hand", choices=("right", "left"), help="whose TCP the policy follows")
    ap.add_argument("--box", default=None, help="BOX column prefix when an export has several (see --check)")
    ap.add_argument("--camera", action="append", type=parse_camera, default=[],
                    help="cam_NN[:x0,y0,x1,y1] (crop in source pixels); repeatable; cam_10 is required")
    ap.add_argument("--image-long-side", type=int, default=320, help="long side of each camera after the crop")
    ap.add_argument("--wrench", action="store_true",
                    help="give the policy the BOX's external wrench (the arm BOX's K must then be in force)")
    ap.add_argument("--frame-stride", type=int, default=2, help="frames between action steps (60 fps / 2 = 30 Hz)")
    ap.add_argument("--horizon", type=int, default=32, help="action steps per chunk")
    ap.add_argument("--n-action-steps", type=int, default=16, help="steps executed per policy call")
    ap.add_argument("--history", type=int, default=2, help="past relative poses in the state")
    ap.add_argument("--policy", choices=POLICY_TYPES, default="act")
    ap.add_argument("--steps", type=int, default=None,
                    help="total optimiser steps (default 20000); with --resume, raise it to train a finished run further")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=2000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--image-aug", action="store_true", help="LeRobot's colour/affine image augmentation")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--policy-arg", action="append", default=[],
                    help="extra lerobot_train argument for the policy, e.g. --policy-arg=--policy.dim_model=256")
    ap.add_argument("--job-name", help="run name (directory under --runs-root)")
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    ap.add_argument("--resume", type=Path, help="a run directory: continue it from its last checkpoint")
    ap.add_argument("--check", action="store_true", help="check the exports, print what would train, write nothing")
    return ap


# ------------------------------------------------------------------- set-up ---


def make_contract(args: argparse.Namespace, check: rd.ExportCheck) -> rt.PolicyContract:
    if not args.camera:
        raise rd.TrainingDataError("pick the cameras (--camera cam_10 [--camera cam_NN:crop])")
    names = [name for name, _ in args.camera]
    if "cam_10" not in names:
        raise rd.TrainingDataError("cam_10 (the camera on the gripper) is the policy's first input; add it")
    if len(set(names)) != len(names):
        raise rd.TrainingDataError("a camera is listed twice")
    cameras = []
    for name, crop in args.camera:
        if name not in check.common_cameras:
            raise rd.TrainingDataError(f"{name} is not in every export ({', '.join(check.common_cameras)})")
        cameras.append(rd.camera_input(name, check.common_cameras[name], crop, args.image_long_side))
    if args.box is None:
        if len(check.boxes) != 1:
            raise rd.TrainingDataError(f"several BOXes ({check.boxes}); pick one with --box")
        box = check.boxes[0]
    else:
        box = args.box
        if box not in check.boxes:
            raise rd.TrainingDataError(f"--box {box!r} is not in every export ({check.boxes})")
    wrench = []
    if args.wrench:
        missing = [s.name for s in check.exports if not s.has_wrench.get(box)]
        if missing:
            raise rd.TrainingDataError(f"no external wrench for BOX {box!r} in {', '.join(missing)}")
        wrench = [f"{box}{rd.WRENCH_PREFIX}{axis}" for axis in rt.WRENCH_AXES]
    contract = rt.PolicyContract(
        hand=args.hand,
        fps=check.exports[0].fps,
        frame_stride=args.frame_stride,
        horizon=args.horizon,
        history=args.history,
        cameras=cameras,
        width_name=f"{box}{rd.WIDTH_SUFFIX}",
        wrench_names=wrench,
    )
    contract.validate()
    if args.n_action_steps > args.horizon:
        raise rd.TrainingDataError("--n-action-steps cannot exceed --horizon")
    return contract


def lerobot_argv(args: argparse.Namespace, run_dir: Path, contract: rt.PolicyContract, views: list[Path]) -> list[str]:
    argv = [
        "lerobot_train",
        f"--dataset.repo_id=local/{run_dir.name}",
        f"--dataset.root={views[0]}",
        f"--policy.type={args.policy}",
        "--policy.push_to_hub=false",
        f"--output_dir={run_dir / 'lerobot'}",
        f"--job_name={run_dir.name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]
    if args.policy == "act":
        argv += [f"--policy.chunk_size={contract.horizon}", f"--policy.n_action_steps={args.n_action_steps}"]
    if args.device:
        argv.append(f"--policy.device={args.device}")
    if args.image_aug:
        argv.append("--dataset.image_transforms.enable=true")
    argv += list(args.policy_arg)
    return argv


# ----------------------------------------------------------- lerobot hooks ---


class _Hooks:
    """What train_relative_tcp patches into lerobot_train: the dataset, the checkpoint extras, progress."""

    def __init__(self, run_dir: Path, run: dict[str, Any], contract: rt.PolicyContract, views: list[Path], stats):
        self.run_dir = run_dir
        self.run = run
        self.contract = contract
        self.views = views
        self.stats = stats
        self.dataset: rd.RelativeTcpDataset | None = None

    def make_dataset(self, cfg):
        from lerobot.datasets.transforms import ImageTransforms

        transforms = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        self.dataset = rd.RelativeTcpDataset(self.views, self.contract, self.stats, image_transforms=transforms)
        progress(event="dataset", frames=self.dataset.num_frames, episodes=self.dataset.num_episodes)
        return self.dataset

    def provenance(self, step: int, total: int) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "run": self.run_dir.name,
            "run_dir": str(self.run_dir.resolve()),
            "step": step,
            "total_steps": total,
            "saved": _now(),
            "exports": self.run["exports"],
            "workspace": self.run.get("workspace"),
            "views": self.run["views"],
            "policy": self.run["policy"],
            "code": self.run["code"],
            "host": self.run["host"],
            "started": self.run["started"],
            "resumes": self.run.get("resumes", []),
        }

    def save_checkpoint(self, original):
        def wrapped(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None, preprocessor=None, postprocessor=None):
            original(checkpoint_dir, step, cfg, policy, optimizer, scheduler, preprocessor, postprocessor)
            model_dir = Path(checkpoint_dir) / "pretrained_model"
            self.contract.write(model_dir)
            prov = self.provenance(step, cfg.steps)
            if step == cfg.steps and self.dataset is not None:
                prov["reload_check"] = reload_check(model_dir, policy, preprocessor, postprocessor, self.dataset)
                progress(event="reload_check", **prov["reload_check"])
            (model_dir / rt.PROVENANCE_FILE).write_text(json.dumps(prov, indent=2), encoding="utf-8")
            progress(event="checkpoint", step=step, path=str(model_dir))

        return wrapped


class _ProgressHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.msg
        if type(msg).__name__ == "MetricsTracker":
            data = msg.to_dict()
            progress(event="train", **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in data.items()})


def reload_check(model_dir: Path, policy, preprocessor, postprocessor, dataset, n: int = 4) -> dict[str, Any]:
    """Load the checkpoint back from disk and compare its predictions with the policy in memory."""
    import torch

    from tools.fr3 import relative_tcp_policy as rtp

    try:
        loaded = rtp.load_checkpoint(model_dir, device=str(next(policy.parameters()).device))
        idx = [int(i * (len(dataset) - 1) / max(1, n - 1)) for i in range(n)]
        diffs = []
        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            for i in idx:
                sample = dataset[i]
                ours = rtp.predict_chunk(policy, preprocessor, postprocessor, sample)
                theirs = loaded.predict(sample)
                diffs.append(float(abs(ours - theirs).max()))
        if was_training:
            policy.train()
        ok = max(diffs) < 1e-4
        return {"ok": ok, "samples": idx, "max_abs_diff": max(diffs), "checked": _now()}
    except Exception as exc:  # noqa: BLE001 -- reported, not raised: the checkpoint is already on disk
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "checked": _now()}


# --------------------------------------------------------------------- main ---


def _host() -> dict[str, Any]:
    info: dict[str, Any] = {"hostname": socket.gethostname(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def prepare(args: argparse.Namespace) -> tuple[Path, dict[str, Any], rt.PolicyContract, list[Path], dict, list[str]]:
    """New run: check, build views, stats, write run.json. Resume: read them back."""
    if args.resume:
        run_dir = args.resume.resolve()
        run = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
        contract = rt.PolicyContract.from_dict(run["contract"])
        views = [Path(v["root"]) for v in run["views"]]
        stats = json.loads((run_dir / STATS_FILE).read_text(encoding="utf-8"))
        last = run_dir / "lerobot" / "checkpoints" / "last" / "pretrained_model"
        if not (last / "train_config.json").is_file():
            raise rd.TrainingDataError(f"{run_dir} has no checkpoint to resume from yet")
        for view in run["views"]:
            if rd._export_fingerprint(Path(view["source"])) != view["source_fingerprint"]:
                raise rd.TrainingDataError(
                    f"{view['source']} changed since this run started; resuming would train on other data. "
                    "Start a new run instead."
                )
        run.setdefault("resumes", []).append({"at": _now(), "from": str(last.parent.resolve()), "code": code_provenance()})
        (run_dir / RUN_FILE).write_text(json.dumps(run, indent=2), encoding="utf-8")
        argv = ["lerobot_train", f"--config_path={last / 'train_config.json'}", "--resume=true"]
        if args.steps is not None:
            argv.append(f"--steps={args.steps}")
            run["policy"]["steps"] = args.steps
            (run_dir / RUN_FILE).write_text(json.dumps(run, indent=2), encoding="utf-8")
        return run_dir, run, contract, views, stats, argv

    args.steps = 20000 if args.steps is None else args.steps
    if not args.export or not args.hand:
        raise rd.TrainingDataError("a new run needs --export and --hand (or --resume <run dir>)")
    check = rd.check_exports(args.export, args.hand)
    for warning in check.warnings:
        print(f"WARNING: {warning}", flush=True)
    check.raise_if_bad()
    contract = make_contract(args, check)
    job = args.job_name or f"{args.policy}_{args.hand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = (args.runs_root / job).resolve()
    if args.check:
        print(json.dumps({"exports": check.to_dict(), "contract": contract.to_dict(), "run_dir": str(run_dir)}, indent=2))
        return run_dir, {}, contract, [], {}, []
    if run_dir.exists():
        raise rd.TrainingDataError(f"{run_dir} exists; pick another --job-name, or --resume it")
    views = [rd.build_view(Path(s.root), contract.cameras, args.views_root) for s in check.exports]
    stats, usable = rd.compute_stats(views, contract)
    run_dir.mkdir(parents=True)
    (run_dir / STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")
    run = {
        "schema": RUN_SCHEMA,
        "job": job,
        "started": _now(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "export"},
        "policy": {"type": args.policy, "steps": args.steps, "batch_size": args.batch_size,
                   "n_action_steps": args.n_action_steps, "extra": list(args.policy_arg)},
        "contract": contract.to_dict(),
        "exports": [
            {**vars(s), "fingerprint": rd._export_fingerprint(Path(s.root))} for s in check.exports
        ],
        "views": [
            {"root": str(v.resolve()), **json.loads((v / rd.VIEW_FILE).read_text(encoding="utf-8"))} for v in views
        ],
        "usable": usable,
        "workspace": rd.training_workspace(views, contract, check.exports[0].world_frame_id),
        "warnings": check.warnings,
        "code": code_provenance(),
        "host": _host(),
    }
    (run_dir / RUN_FILE).write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run_dir, run, contract, views, stats, lerobot_argv(args, run_dir, contract, views)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir, run, contract, views, stats, train_argv = prepare(args)
    except (rd.TrainingDataError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        progress(event="refused", error=str(exc))
        return 2
    if args.check:
        return 0
    progress(event="start", run_dir=str(run_dir), resume=bool(args.resume))

    from lerobot.scripts import lerobot_train

    hooks = _Hooks(run_dir, run, contract, views, stats)
    lerobot_train.make_dataset = hooks.make_dataset
    lerobot_train.save_checkpoint = hooks.save_checkpoint(lerobot_train.save_checkpoint)
    original_init_logging = lerobot_train.init_logging

    def init_logging(*a, **kw):  # lerobot_train resets the root handlers; add ours after it
        original_init_logging(*a, **kw)
        logging.getLogger().addHandler(_ProgressHandler())

    lerobot_train.init_logging = init_logging
    sys.argv = train_argv
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    lerobot_train.main()
    progress(event="done", run_dir=str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
