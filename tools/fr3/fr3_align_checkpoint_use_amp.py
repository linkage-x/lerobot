#!/usr/bin/env python3
"""Align a checkpoint's recorded ``use_amp`` with what its rollouts are compared against.

``use_amp`` is the one field a training run records but does not honour. ``lerobot_train.py``
never reads it: it wraps the step in ``accelerator.autocast()`` and builds its Accelerator
without ``mixed_precision=``, so two runs with opposite settings train identically. The rollout
runtime *does* read it -- ``fr3_act_infer_real_runtime.py`` passes ``bool(policy.config.use_amp)``
into ``predict_action_chunk_for_rollout`` and ``predict_action``, both of which wrap inference in
``torch.autocast`` only when it is true.

So the field carries no information about how a checkpoint was trained while quietly deciding how
it is evaluated. Every bf16 checkpoint this project has rolled out recorded ``use_amp: true``,
which means every success rate on record was measured under autocast. A checkpoint trained after
the GUI's default was turned off records ``false``, and rolling it out would compare a different
inference path rather than a different policy -- the one thing an A/B between two checkpoints is
supposed to hold still.

Only checkpoints whose recorded dtype belongs to the lineage with a settled convention are
touched. An older fp32 checkpoint is left exactly as it was evaluated: this fixes a drift, it
does not retroactively re-run history under new rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

#: What a rollout is compared against. Not a preference -- it is the value recorded by every
#: pi0.5 checkpoint whose success rate is in the rollout log.
ROLLOUT_USE_AMP = True

#: The dtypes whose checkpoints all agree on ROLLOUT_USE_AMP. Anything else predates the
#: convention and is left alone unless the caller insists.
CONVENTION_DTYPES: tuple[str, ...] = ("bfloat16",)

POLICY_CONFIG = "config.json"
TRAIN_CONFIG = "train_config.json"
#: Written beside the config so the change is answerable later. The whole point of aligning is
#: that rollout numbers stay comparable, and a silently rewritten config would leave nobody able
#: to say which inference path a recorded success rate was measured on.
ALIGNMENT_RECORD = "use_amp_alignment.json"


class AlignmentError(RuntimeError):
    """A refusal the operator can fix."""


@dataclass
class AlignmentResult:
    checkpoint: str
    aligned_to: bool
    previous: bool | None = None
    changed: bool = False
    skipped: str = ""
    files: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"{self.checkpoint}: left as use_amp={_spell(self.previous)} ({self.skipped})"
        if not self.changed:
            return f"{self.checkpoint}: already use_amp={_spell(self.aligned_to)}"
        return (
            f"{self.checkpoint}: use_amp {_spell(self.previous)} -> {_spell(self.aligned_to)} "
            f"({', '.join(self.files)})"
        )


def _spell(value: bool | None) -> str:
    return "unset" if value is None else ("true" if value else "false")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except OSError as exc:
        raise AlignmentError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AlignmentError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise AlignmentError(f"{path} is not a JSON object")
    return loaded


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace `path` atomically, so an interrupted write cannot leave an unloadable config.

    The file being rewritten is the one the policy loader reads; a truncated config.json turns a
    recoverable metadata mismatch into a checkpoint that will not load at all.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def pretrained_model_dir(checkpoint: Path) -> Path:
    """The directory holding config.json, given either it or the step directory above it."""
    checkpoint = Path(checkpoint)
    if (checkpoint / POLICY_CONFIG).is_file():
        return checkpoint
    nested = checkpoint / "pretrained_model"
    if (nested / POLICY_CONFIG).is_file():
        return nested
    raise AlignmentError(
        f"no {POLICY_CONFIG} under {checkpoint}; expected a checkpoint step directory or the "
        "pretrained_model directory inside one"
    )


def align_checkpoint_use_amp(
    checkpoint: Path,
    *,
    expect: bool = ROLLOUT_USE_AMP,
    only_dtypes: Sequence[str] | None = CONVENTION_DTYPES,
    dry_run: bool = False,
) -> AlignmentResult:
    """Set the checkpoint's recorded ``use_amp`` to `expect`, and say what that changed.

    `only_dtypes` restricts the edit to checkpoints recording one of those dtypes; pass None to
    align regardless. `dry_run` reports what would change without writing.

    ``train_config.json`` is updated alongside ``config.json`` even though only the latter is what
    the policy loader reads. A checkpoint that disagrees with itself about how it was evaluated is
    a worse artifact than one that is simply wrong in a known way.
    """
    pretrained = pretrained_model_dir(checkpoint)
    name = pretrained.parent.name if pretrained.name == "pretrained_model" else pretrained.name
    policy_config = read_json(pretrained / POLICY_CONFIG)

    raw = policy_config.get("use_amp")
    previous = None if raw is None else bool(raw)

    dtype = str(policy_config.get("dtype") or "")
    if only_dtypes is not None and dtype not in only_dtypes:
        return AlignmentResult(
            checkpoint=name,
            aligned_to=expect,
            previous=previous,
            skipped=f"dtype {dtype or 'unset'} is outside the aligned lineage "
            f"({', '.join(only_dtypes)})",
        )

    if previous == expect:
        return AlignmentResult(checkpoint=name, aligned_to=expect, previous=previous)

    result = AlignmentResult(
        checkpoint=name, aligned_to=expect, previous=previous, changed=True, files=[POLICY_CONFIG]
    )
    train_config_path = pretrained / TRAIN_CONFIG
    train_config = read_json(train_config_path) if train_config_path.is_file() else None
    if isinstance((train_config or {}).get("policy"), dict):
        result.files.append(TRAIN_CONFIG)
    if dry_run:
        return result

    policy_config["use_amp"] = expect
    write_json(pretrained / POLICY_CONFIG, policy_config)
    if TRAIN_CONFIG in result.files:
        train_config["policy"]["use_amp"] = expect  # type: ignore[index]
        write_json(train_config_path, train_config)  # type: ignore[arg-type]
    _append_alignment_record(pretrained, previous=previous, aligned_to=expect)
    return result


def _append_alignment_record(pretrained: Path, *, previous: bool | None, aligned_to: bool) -> None:
    """Append to the audit trail, best-effort.

    Best-effort on purpose: the alignment itself has already landed, and failing here would
    report a rollout as blocked over a note about a change that did happen.
    """
    path = pretrained / ALIGNMENT_RECORD
    entries: list[Any] = []
    if path.is_file():
        try:
            existing = read_json(path).get("entries")
        except AlignmentError:
            existing = None
        if isinstance(existing, list):
            entries = existing
    entries.append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from": previous,
            "to": aligned_to,
            "by": Path(__file__).name,
            "why": "use_amp is inert during training but selects the rollout autocast path",
        }
    )
    try:
        write_json(path, {"entries": entries})
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint step directories")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-amp", dest="expect", action="store_true", default=ROLLOUT_USE_AMP,
        help="align to use_amp=true (the default, and what every rolled-out bf16 checkpoint has)",
    )
    group.add_argument("--no-use-amp", dest="expect", action="store_false")
    parser.add_argument(
        "--force", action="store_true",
        help=f"align regardless of dtype, not only {', '.join(CONVENTION_DTYPES)}",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report without writing; exit 1 if any checkpoint would change",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    changed = False
    for checkpoint in args.checkpoints:
        try:
            result = align_checkpoint_use_amp(
                checkpoint,
                expect=args.expect,
                only_dtypes=None if args.force else CONVENTION_DTYPES,
                dry_run=args.check,
            )
        except AlignmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        changed = changed or result.changed
        if not args.quiet:
            print(result.summary)
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
