#!/usr/bin/env python
"""Serve rollout policies' forward passes to robot hosts running `--policy-server`.

A standing process on the GPU machine. It loads whichever checkpoint a robot host names in its
hello -- by its path from `outputs/` on, so the checkpoint has to be in this checkout's outputs
too -- keeps it for the next rollout, and drops it after `--idle-unload-s` without a client so the
card is free for training:

    PYTHONPATH=src:. .venv-fr3/bin/python tools/fr3/fr3_policy_server.py [--port 8790] [--idle-unload-s 900]

`run/policy_server.sh` starts and stops it, and `bash run/deploy.sh nuc` runs that for you.

`--checkpoint` (with any of the runtime's policy flags) loads that policy at startup, so the
first rollout does not wait the minute pi05 takes to load; `--warm-last` does the same for the
last policy this checkout's server loaded, if the card has room for it. Either loads in the
background, and a rollout that arrives meanwhile waits for it. Nothing in this process touches
hardware.

While a training run is going on this machine the server holds nothing: it refuses to load (a
rollout's hello says why), skips the warm-up, and drops an idle policy it already holds. So a
deploy, which restarts and warms this server, is safe at any time. `--allow-training-overlap`
turns that off.

The server records the repo files it has imported, with their hashes, in `CODE_MANIFEST_PATH`;
`run/policy_server.sh ensure` reads it to restart the server only when one of them has changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType

from tools.fr3 import fr3_act_infer_real_runtime as rt
from tools.fr3 import fr3_align_checkpoint_use_amp as use_amp_alignment
from tools.fr3.policy_remote import (
    DEFAULT_POLICY_SERVER_PORT,
    LOAD_SPEC_OVERRIDES,
    LoadedPolicy,
    PolicyBusy,
    PolicyHost,
    policy_fingerprint,
    policy_load_spec,
    resolve_outputs_path,
    serve_connection,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
LAST_SPEC_PATH = Path('run/logs/policy_server_last_spec.json')
CODE_MANIFEST_PATH = Path('run/logs/policy_server_code.json')
# pi05 holds about 10 GB resident; less free than this means something else has the card, and a
# warm-up nobody asked for would take it away. Not how a training run is recognised: on a 32 GB
# card one leaves ~17 GB free (10-21, Dante), and this check let a deploy warm 10 GB beside it.
DEFAULT_WARM_MIN_FREE_GB = 12.0
# A training run on this machine, by the interpreter's argv: the trainer
# (`python -m lerobot.scripts.lerobot_train`, or its script / console entry point) and the fr3
# wrapper that launches it.
TRAINING_MODULES = frozenset({'lerobot.scripts.lerobot_train'})
TRAINING_SCRIPTS = frozenset({'lerobot_train.py', 'lerobot-train', 'fr3_train_il_policy.py'})


def _split_args(argv: list[str] | None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0], add_help=False)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=DEFAULT_POLICY_SERVER_PORT)
    parser.add_argument(
        '--idle-unload-s',
        type=float,
        default=900.0,
        help='Drop the loaded policy after this long without a client (0 keeps it loaded).',
    )
    parser.add_argument(
        '--warm-last',
        action='store_true',
        help=f'At startup, load the last policy this server loaded ({LAST_SPEC_PATH}) if the card has room.',
    )
    parser.add_argument('--warm-min-free-gb', type=float, default=DEFAULT_WARM_MIN_FREE_GB)
    parser.add_argument(
        '--allow-training-overlap',
        action='store_true',
        help='Load and keep policies while a training run is going on this machine (by default the card is left to it).',
    )
    parser.add_argument('-h', '--help', action='store_true')
    server_args, rest = parser.parse_known_args(argv)
    if server_args.help:
        print(__doc__)
        raise SystemExit(0)
    return server_args, rt.parse_args(rest)


class _PolicyStack:
    """Owns the loaded objects so that `release` can actually let go of them."""

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any) -> None:
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor

    def predict(self, observation: dict[str, Any], **kwargs: Any):
        return rt.predict_action_chunk_for_rollout(
            observation, policy=self.policy, preprocessor=self.preprocessor, postprocessor=self.postprocessor, **kwargs
        )

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def release(self) -> None:
        self.policy = self.preprocessor = self.postprocessor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CodeManifest:
    """The repo's .py files this process has imported, hashed when first seen.

    Hashed once each, so a file edited after its import still reads as changed. Modules that only
    the load imports (the policy's own modelling code) are added after it.
    """

    def __init__(self, repo_root: Path, path: Path) -> None:
        self.repo_root, self.path = repo_root, repo_root / path
        self.files: dict[str, str] = {}

    def update(self) -> None:
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if not file or not file.endswith('.py'):
                continue
            try:
                rel = Path(file).resolve().relative_to(self.repo_root)
            except ValueError:
                continue
            if rel.parts[0].startswith('.venv') or str(rel) in self.files:
                continue
            try:
                self.files[str(rel)] = hashlib.sha256((self.repo_root / rel).read_bytes()).hexdigest()
            except OSError:
                continue
        _write_json(self.path, {'pid': os.getpid(), 'files': dict(sorted(self.files.items()))})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1) + '\n')
    tmp.replace(path)


# The guidance delay RTC uses for the warm-up's guided pass; any value inside the chunk does.
WARM_FORWARD_DELAY_STEPS = 8


def warm_forward(stack: _PolicyStack, policy_cfg: Any, device: torch.device, *, task: str | None, log=print) -> None:
    """Run the first forwards on blank inputs, so the first rollout's do not pay for CUDA's warm-up.

    The first predict after a load took 0.43-0.66 s against a p50 of 0.22 (10-16), and the trial's
    guidance delay is the worst latency it has seen, so it ran at 16-21 steps instead of 8. Two
    passes: one unguided, as a trial's first chunk is, and one guided by a prefix, as every later
    one is -- the two take different paths through RTC.
    """
    observation: dict[str, np.ndarray] = {}
    for key, feature in policy_cfg.input_features.items():
        shape = tuple(int(part) for part in feature.shape)
        if feature.type == FeatureType.VISUAL:
            observation[key] = np.zeros((*shape[1:], shape[0]), dtype=np.uint8)
        else:
            observation[key] = np.zeros(shape, dtype=np.float32)
    horizon = int(policy_cfg.rtc_config.execution_horizon)
    common = {'device': device, 'use_amp': bool(policy_cfg.use_amp), 'execution_horizon': horizon, 'task': task}
    started = time.perf_counter()
    original, _ = stack.predict(dict(observation), inference_delay=0, prev_chunk_left_over=None, **common)
    unguided_s = time.perf_counter() - started
    started = time.perf_counter()
    stack.predict(
        dict(observation),
        inference_delay=WARM_FORWARD_DELAY_STEPS,
        prev_chunk_left_over=original[WARM_FORWARD_DELAY_STEPS:],
        **common,
    )
    guided_s = time.perf_counter() - started
    stack.reset()
    log(f'[INFO] policy_server warm_forward unguided_s={unguided_s:.3f} guided_s={guided_s:.3f}')


def training_processes(proc_root: Path = Path('/proc')) -> list[str]:
    """The training runs going on this machine, as 'pid N (what)'; empty when there are none.

    Read off each interpreter's argv, like `run/policy_server.sh` finds the server, so a shell or
    an ssh command that merely names the trainer does not count.
    """
    runs: dict[str, int] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [a.decode('utf-8', 'ignore') for a in (entry / 'cmdline').read_bytes().split(b'\0') if a]
        except OSError:
            continue
        if not args or not os.path.basename(args[0]).startswith('python'):
            continue
        what = next((a for a in args[1:] if a in TRAINING_MODULES or os.path.basename(a) in TRAINING_SCRIPTS), None)
        # One per kind: the trainer's DataLoader workers carry its argv too; the oldest is the trainer.
        if what is not None:
            what = os.path.basename(what)
            runs[what] = min(runs.get(what, int(entry.name)), int(entry.name))
    return sorted(f'pid {pid} ({what})' for what, pid in runs.items())


class TrainingOnCard(RuntimeError):
    """A load refused because a training run has this machine's card."""


def make_loader(repo_root: Path, device: torch.device, *, log=print, on_loaded=None, training=None):
    """`training` lists the training runs going on now (see `training_processes`); None never refuses."""

    def load(spec: dict[str, Any]) -> LoadedPolicy:
        runs = training() if training is not None else []
        if runs:
            raise TrainingOnCard(
                f'this policy server is on a machine that is training ({", ".join(runs)}), so it loads no '
                'policy until that finishes. Roll out after the training, or stop it first.'
            )
        overrides = {key: spec[key] for key in LOAD_SPEC_OVERRIDES if key in spec}
        checkpoint = resolve_outputs_path(repo_root, str(spec['checkpoint']))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f'{spec["checkpoint"]} is not on this policy server (looked in {repo_root}). '
                'Run the policy server on the machine that trained it, or copy the checkpoint there.'
            )
        pretrained_dir = rt.resolve_pretrained_model_dir(checkpoint)
        # The NUC gateway aligns its own copy of the checkpoint before every rollout; this copy has
        # to get the same, or the hello's fingerprints differ on use_amp (R4, 10-27). Warned, not
        # refused, as the gateway does: the fingerprint check still stops a real mismatch.
        try:
            alignment = use_amp_alignment.align_checkpoint_use_amp(pretrained_dir)
        except use_amp_alignment.AlignmentError as exc:
            log(f'[WARN] policy_server could not align use_amp: {exc}')
        else:
            if alignment.changed:
                log(f'[WARN] policy_server aligned use_amp before loading -- {alignment.summary}')
        train_cfg = rt.load_train_config(pretrained_dir)
        explicit_root = None
        if spec.get('dataset_root'):
            explicit_root = resolve_outputs_path(repo_root, str(spec['dataset_root']))
            if not explicit_root.exists():
                # The view's metadata is what the policy reads here (features, stats); the
                # checkpoint's own record of it finds the local copy, as a terminal run would.
                log(f'[WARN] policy_server dataset_root={spec["dataset_root"]} not here; using the checkpoint\'s own')
                explicit_root = None
        dataset_root = rt.resolve_dataset_root(pretrained_dir, train_cfg, explicit_root)
        ds_meta = rt.load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
        policy, preprocessor, postprocessor = rt.load_policy_stack(pretrained_dir, ds_meta=ds_meta, device=device, **overrides)
        fingerprint = policy_fingerprint(policy.config, pretrained_dir)
        stack = _PolicyStack(policy, preprocessor, postprocessor)
        if not fingerprint['rtc_enabled']:
            stack.release()
            raise ValueError('The policy server only serves the RTC chunk path, and RTC is not enabled for this checkpoint.')
        try:
            warm_forward(stack, policy.config, device, task=rt.resolve_rollout_task_prompt(ds_meta, None), log=log)
        except Exception as exc:  # only latency is at stake; the first rollout's forward will say what is wrong
            stack.reset()
            log(f'[WARN] policy_server warm_forward_failed {type(exc).__name__}: {exc}')
        if on_loaded is not None:
            on_loaded(spec)
        return LoadedPolicy(
            spec=dict(spec),
            fingerprint=fingerprint,
            device=device,
            use_amp=bool(policy.config.use_amp),
            predict=stack.predict,
            reset=stack.reset,
            release=stack.release,
        )

    return load


def _flush_print(line: str) -> None:
    print(line)
    sys.stdout.flush()


def last_spec_worth_warming(
    device: torch.device, min_free_gb: float, *, training: list[str] | None = None, log=print
) -> dict[str, Any] | None:
    if training:
        log(f'[INFO] policy_server warm_last skipped: training on this machine ({", ".join(training)})')
        return None
    try:
        spec = json.loads((_REPO_ROOT / LAST_SPEC_PATH).read_text())
    except FileNotFoundError:
        log('[INFO] policy_server warm_last skipped: no policy loaded here yet')
        return None
    except (OSError, ValueError) as exc:
        log(f'[WARN] policy_server warm_last skipped: {LAST_SPEC_PATH} unreadable ({exc})')
        return None
    if device.type == 'cuda':
        free_gb = torch.cuda.mem_get_info(device)[0] / 2**30
        if free_gb < min_free_gb:
            log(f'[INFO] policy_server warm_last skipped: {free_gb:.1f} GB free on the card < {min_free_gb:.1f} (training?)')
            return None
    return spec


def yield_card_to_training(host: PolicyHost, training: list[str], *, log=print) -> bool:
    """Drop the held policy when a training run has started here, unless a rollout is using it.

    A rollout in progress keeps its policy (unloading it would end the trial mid-motion); the next
    check after it lets go drops it.
    """
    if not training:
        return False
    try:
        unloaded = host.unload()
    except PolicyBusy:
        return False
    if unloaded:
        log(f'[INFO] policy_server yielded the card to training ({", ".join(training)})')
    return unloaded


def _warm(host: PolicyHost, spec: dict[str, Any]) -> None:
    try:
        loaded = host.warm(spec)
    except Exception as exc:  # the first rollout will load it, or say why it cannot
        _flush_print(f'[WARN] policy_server warm_failed {type(exc).__name__}: {exc}')
        return
    if loaded is None:
        _flush_print('[INFO] policy_server warm skipped: a rollout took the policy first')
    else:
        _flush_print(f'[INFO] policy_server warmed fingerprint={loaded.fingerprint}')


def main(argv: list[str] | None = None) -> int:
    server_args, args = _split_args(argv)
    if getattr(args, 'policy_server', None):
        raise SystemExit('--policy-server names the server a robot host connects to; drop it here.')
    device = rt.resolve_policy_device(args.device, torch.cuda.is_available())
    idle_unload_s = server_args.idle_unload_s if server_args.idle_unload_s > 0 else None
    manifest = CodeManifest(_REPO_ROOT, CODE_MANIFEST_PATH)

    def on_loaded(spec: dict[str, Any]) -> None:
        manifest.update()
        try:
            _write_json(_REPO_ROOT / LAST_SPEC_PATH, spec)
        except OSError as exc:
            _flush_print(f'[WARN] policy_server last_spec_not_saved {type(exc).__name__}: {exc}')

    training = None if server_args.allow_training_overlap else training_processes
    host = PolicyHost(
        make_loader(_REPO_ROOT, device, log=_flush_print, on_loaded=on_loaded, training=training),
        idle_unload_s=idle_unload_s,
        log=_flush_print,
    )

    # Bound before the preload, so a second server on the same port fails in a second rather than
    # after a minute of loading weights it then has nowhere to serve.
    listener = socket.create_server((server_args.host, server_args.port), reuse_port=False)
    _flush_print(
        f'[INFO] policy_server listening={server_args.host}:{server_args.port} device={device} '
        f'repo={_REPO_ROOT} idle_unload_s={idle_unload_s}'
    )
    manifest.update()
    warm_spec = None
    raw_argv = argv if argv is not None else sys.argv[1:]
    if any(item == '--checkpoint' or item.startswith('--checkpoint=') for item in raw_argv):
        pretrained_dir = rt.resolve_pretrained_model_dir(args.checkpoint)
        spec = policy_load_spec(
            pretrained_dir,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            n_action_steps_override=args.policy_n_action_steps,
            act_temporal_ensemble_coeff=args.act_temporal_ensemble_coeff,
            rtc_mode=args.rtc_mode,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        )
        warm_spec = spec
    elif server_args.warm_last:
        warm_spec = last_spec_worth_warming(
            device, server_args.warm_min_free_gb, training=training() if training else None, log=_flush_print
        )
    if warm_spec is not None:
        threading.Thread(target=_warm, args=(host, warm_spec), name='warm', daemon=True).start()

    stop = threading.Event()

    def idle_watch() -> None:
        while not stop.wait(30.0):
            try:
                yield_card_to_training(host, training() if training else [], log=_flush_print)
                host.unload_if_idle()
            except Exception as exc:  # never let the watcher die quietly
                _flush_print(f'[WARN] policy_server idle_unload_failed {type(exc).__name__}: {exc}')

    threading.Thread(target=idle_watch, name='idle-unload', daemon=True).start()

    def serve(conn: socket.socket, peer: str) -> None:
        with conn:
            serve_connection(conn, host=host, peer=peer, log=_flush_print)

    try:
        while True:
            conn, (addr, port) = listener.accept()
            threading.Thread(target=serve, args=(conn, f'{addr}:{port}'), name=f'client-{addr}:{port}', daemon=True).start()
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        listener.close()


if __name__ == '__main__':
    raise SystemExit(main())
