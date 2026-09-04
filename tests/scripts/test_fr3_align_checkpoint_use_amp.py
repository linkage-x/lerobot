from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fr3.fr3_align_checkpoint_use_amp import (
    ALIGNMENT_RECORD,
    AlignmentError,
    align_checkpoint_use_amp,
    main,
    pretrained_model_dir,
)


def _checkpoint(
    root: Path,
    *,
    use_amp: bool | None = False,
    dtype: str | None = "bfloat16",
    train_config: bool = True,
) -> Path:
    """A checkpoint step directory shaped like the ones lerobot_train writes."""
    pretrained = root / "checkpoints" / "030000" / "pretrained_model"
    pretrained.mkdir(parents=True)
    config: dict = {"type": "pi05", "chunk_size": 50}
    if dtype is not None:
        config["dtype"] = dtype
    if use_amp is not None:
        config["use_amp"] = use_amp
    (pretrained / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if train_config:
        (pretrained / "train_config.json").write_text(
            json.dumps({"steps": 30000, "policy": {"type": "pi05", "use_amp": use_amp}}),
            encoding="utf-8",
        )
    return pretrained.parent


def _config(step_dir: Path) -> dict:
    return json.loads((step_dir / "pretrained_model" / "config.json").read_text(encoding="utf-8"))


def _train_config(step_dir: Path) -> dict:
    return json.loads(
        (step_dir / "pretrained_model" / "train_config.json").read_text(encoding="utf-8")
    )


def test_alignment_turns_on_the_autocast_the_rollout_reads(tmp_path):
    """The whole reason this exists: `lerobot_train` never reads `use_amp`, so a checkpoint can
    record `false` for a run that was numerically identical to one recording `true` -- and then
    `fr3_act_infer_real_runtime` gates `torch.autocast` on that field and evaluates the two on
    different inference paths."""
    step_dir = _checkpoint(tmp_path, use_amp=False)

    result = align_checkpoint_use_amp(step_dir)

    assert result.changed is True
    assert result.previous is False
    assert result.aligned_to is True
    assert _config(step_dir)["use_amp"] is True


def test_alignment_also_fixes_train_config_so_the_checkpoint_agrees_with_itself(tmp_path):
    """Only config.json is what the policy loader reads, but a checkpoint whose two files
    disagree about how it was evaluated is a worse artifact than one that is simply wrong."""
    step_dir = _checkpoint(tmp_path, use_amp=False)

    align_checkpoint_use_amp(step_dir)

    assert _train_config(step_dir)["policy"]["use_amp"] is True


def test_an_already_aligned_checkpoint_is_left_untouched(tmp_path):
    step_dir = _checkpoint(tmp_path, use_amp=True)

    result = align_checkpoint_use_amp(step_dir)

    assert result.changed is False
    assert not (step_dir / "pretrained_model" / ALIGNMENT_RECORD).exists()


def test_a_checkpoint_outside_the_aligned_lineage_keeps_the_value_it_was_evaluated_under(tmp_path):
    """An fp32 ACT checkpoint from before any of this was evaluated with whatever it recorded.
    Flipping it now would not fix a drift, it would rewrite the conditions its own numbers were
    measured under."""
    step_dir = _checkpoint(tmp_path, use_amp=False, dtype=None)

    result = align_checkpoint_use_amp(step_dir)

    assert result.changed is False
    assert "outside the aligned lineage" in result.skipped
    assert _config(step_dir)["use_amp"] is False
    assert align_checkpoint_use_amp(step_dir, only_dtypes=None).changed is True


def test_a_config_without_the_field_at_all_is_given_one(tmp_path):
    """Absent reads as False through `PreTrainedConfig.use_amp`, so it is the same mismatch."""
    step_dir = _checkpoint(tmp_path, use_amp=None)

    result = align_checkpoint_use_amp(step_dir)

    assert result.previous is None
    assert result.changed is True
    assert _config(step_dir)["use_amp"] is True


def test_the_change_leaves_an_audit_trail(tmp_path):
    """A rollout's success rate is only comparable if it is answerable which inference path it was
    measured on, and a silently rewritten config cannot answer that."""
    step_dir = _checkpoint(tmp_path, use_amp=False)

    align_checkpoint_use_amp(step_dir)
    align_checkpoint_use_amp(step_dir, expect=False)

    record = json.loads(
        (step_dir / "pretrained_model" / ALIGNMENT_RECORD).read_text(encoding="utf-8")
    )
    assert [(entry["from"], entry["to"]) for entry in record["entries"]] == [
        (False, True),
        (True, False),
    ]


def test_a_dry_run_reports_without_writing(tmp_path):
    step_dir = _checkpoint(tmp_path, use_amp=False)

    result = align_checkpoint_use_amp(step_dir, dry_run=True)

    assert result.changed is True
    assert _config(step_dir)["use_amp"] is False


def test_either_the_step_directory_or_the_model_directory_names_a_checkpoint(tmp_path):
    step_dir = _checkpoint(tmp_path, use_amp=False)

    assert pretrained_model_dir(step_dir) == step_dir / "pretrained_model"
    assert pretrained_model_dir(step_dir / "pretrained_model") == step_dir / "pretrained_model"


def test_a_path_that_is_not_a_checkpoint_is_refused_by_name(tmp_path):
    with pytest.raises(AlignmentError, match="config.json"):
        align_checkpoint_use_amp(tmp_path)


def test_a_checkpoint_without_a_train_config_still_aligns_the_one_file_that_matters(tmp_path):
    """A fetched checkpoint is an rsync of pretrained_model; nothing guarantees every file is
    there, and the rollout only needs config.json."""
    step_dir = _checkpoint(tmp_path, use_amp=False, train_config=False)

    result = align_checkpoint_use_amp(step_dir)

    assert result.files == ["config.json"]
    assert _config(step_dir)["use_amp"] is True


def test_check_mode_exits_nonzero_on_a_misaligned_checkpoint(tmp_path, capsys):
    misaligned = _checkpoint(tmp_path / "a", use_amp=False)
    aligned = _checkpoint(tmp_path / "b", use_amp=True)

    assert main([str(aligned), "--check"]) == 0
    assert main([str(misaligned), "--check"]) == 1
    assert _config(misaligned)["use_amp"] is False
    assert "use_amp false -> true" in capsys.readouterr().out


def test_the_cli_can_align_the_other_way(tmp_path):
    step_dir = _checkpoint(tmp_path, use_amp=True)

    assert main([str(step_dir), "--no-use-amp", "--quiet"]) == 0

    assert _config(step_dir)["use_amp"] is False
