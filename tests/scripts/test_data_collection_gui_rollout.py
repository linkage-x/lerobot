#!/usr/bin/env python
"""Checkpoint registry and real-robot rollout control.

The theme of these tests is that the dangerous rollout failure on this rig is not a crash. A
checkpoint trained against one tool frame and rolled out against the other runs, tracks its
targets, and is wrong by 410.85 mm everywhere; a view exported under a different action mode
produces numbers the runtime will integrate as deltas regardless of what they mean. So most of
what follows checks that those cases are *refused*, not that the happy path works.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.data_collection_gui import checkpoints as checkpoint_backend  # noqa: E402
from tools.data_collection_gui import rollout as rollout_backend  # noqa: E402
from tools.data_collection_gui import task_ladders  # noqa: E402
from tools.data_collection_gui import training as training_backend  # noqa: E402

SCAN_SCRIPT = REPO_ROOT / "tools" / "fr3" / "scan_checkpoints.py"


# --------------------------------------------------------------------- fixtures ---


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_view(repo_root: Path, name: str, *, action_mode: str = "delta_ee_from_prev_cmd") -> Path:
    view_root = repo_root / "outputs" / "exports" / "training_views" / name
    _write_json(
        view_root / "meta" / "info.json",
        {
            "fps": 30,
            "total_episodes": 20,
            "total_frames": 10305,
            "features": {
                "observation.images.ee": {},
                "observation.images.side": {},
                "observation.state": {},
            },
        },
    )
    _write_json(
        view_root / "meta" / "il_view_manifest.json",
        {
            "action_mode": action_mode,
            "state_keys": ["observation.state"],
            "repo_id": f"local/{name}",
            "source_fps": {},
            "frame_stride": {},
        },
    )
    return view_root


INFERENCE_CONFIG_TEMPLATE = """version: 1
training:
  policy: act
  camera_keys:
  - ee
  - side
runtime:
  camera_config: tools/fr3/fr3_il_infer_realsense_camera_config.yaml
  hardware:
    robot_ip: 192.168.1.206
    gripper_backend: pika
    target_frame_name: {frame}
  safety:
    first_frame_max_pos_delta_mm: 20.0
    first_frame_max_rot_delta_deg: 8.0
    max_step_pos_delta_mm: 3.0
    max_step_rot_delta_deg: 2.0
"""


def _make_checkpoint(
    repo_root: Path,
    job: str,
    step: str,
    view_root: Path,
    *,
    frame: str = "pika_gripper_ee",
    with_inference_config: bool = True,
) -> Path:
    step_dir = repo_root / "outputs" / "train" / job / "checkpoints" / step
    pretrained = step_dir / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    (pretrained / "model.safetensors").write_bytes(b"\x00" * 1024)
    _write_json(
        pretrained / "config.json",
        {
            "type": "act",
            "chunk_size": 30,
            "n_action_steps": 30,
            "input_features": {
                "observation.images.ee": {},
                "observation.images.side": {},
                "observation.state": {},
            },
        },
    )
    _write_json(
        pretrained / "train_config.json",
        {
            "steps": 20000,
            "seed": 1000,
            "job_name": job,
            "dataset": {"repo_id": f"local/{view_root.name}", "root": str(view_root)},
            "wandb": {"enable": False, "project": "lerobot", "run_id": None},
        },
    )
    if with_inference_config:
        run_dir = view_root / "runs" / job
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "inference_config.generated.yaml").write_text(
            INFERENCE_CONFIG_TEMPLATE.format(frame=frame), encoding="utf-8"
        )
    return step_dir


def _scan(repo_root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), str(repo_root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


RIG = checkpoint_backend.RigContract(
    robotIp="192.168.1.206",
    targetFrameName="pika_gripper_ee",
    cameraKeys=["ee", "side"],
    cameraConfigPath="tools/fr3/fr3_il_infer_realsense_camera_config.yaml",
)


def _entry(repo_root: Path, checkpoint_id: str) -> dict:
    """One scanned checkpoint with its contract parsed, as the gateway assembles it."""
    report = _scan(repo_root)
    raw = next(item for item in report["checkpoints"] if item["id"] == checkpoint_id)
    entry = dict(raw)
    entry["contract"] = checkpoint_backend.parse_inference_contract(
        entry.pop("inferenceConfigText", "")
    )
    return entry


# ------------------------------------------------------------------ scan script ---


def test_scan_reports_the_contract_a_rollout_has_to_match(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)

    entry = _entry(tmp_path, "job_a/020000")

    assert entry["policyType"] == "act"
    assert entry["step"] == 20000
    assert entry["cameras"] == ["ee", "side"]
    assert entry["view"]["actionMode"] == "delta_ee_from_prev_cmd"
    assert entry["view"]["fps"] == 30
    assert entry["view"]["episodes"] == 20
    assert entry["contract"]["targetFrameName"] == "pika_gripper_ee"
    assert entry["contract"]["robotIp"] == "192.168.1.206"


def test_scan_skips_a_checkpoint_that_was_killed_mid_save(tmp_path: Path):
    """A directory with no weights is not a checkpoint, and offering it wastes a rig visit."""
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    half_written = tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "025000"
    (half_written / "pretrained_model").mkdir(parents=True)
    _write_json(half_written / "pretrained_model" / "config.json", {"type": "act"})

    ids = [item["id"] for item in _scan(tmp_path)["checkpoints"]]

    assert "job_a/020000" in ids
    assert "job_a/025000" not in ids


def test_scan_reports_lora_adapter_checkpoints(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    step_dir = _make_checkpoint(tmp_path, "job_pi05", "020000", view)
    pretrained = step_dir / "pretrained_model"
    (pretrained / "model.safetensors").unlink()
    (pretrained / "adapter_model.safetensors").write_bytes(b"adapter" * 1024)
    _write_json(
        pretrained / "adapter_config.json",
        {"base_model_name_or_path": "/home/tele/Models/pi05_base"},
    )
    config = json.loads((pretrained / "config.json").read_text(encoding="utf-8"))
    config["type"] = "pi05"
    _write_json(pretrained / "config.json", config)

    entry = _entry(tmp_path, "job_pi05/020000")

    assert entry["policyType"] == "pi05"
    assert entry["pretrainedPath"].endswith("checkpoints/020000/pretrained_model")
    assert entry["sizeBytes"] >= len(b"adapter" * 1024)


def test_scan_skips_a_half_written_lora_adapter_checkpoint(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    step_dir = _make_checkpoint(tmp_path, "job_pi05", "020000", view)
    pretrained = step_dir / "pretrained_model"
    (pretrained / "model.safetensors").unlink()
    (pretrained / "adapter_model.safetensors").write_bytes(b"adapter")

    ids = [item["id"] for item in _scan(tmp_path)["checkpoints"]]

    assert "job_pi05/020000" not in ids


def test_scan_names_the_step_that_last_points_at(tmp_path: Path):
    """`last` is a symlink, so its bytes are already counted under the numbered step."""
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    checkpoints_dir = tmp_path / "outputs" / "train" / "job_a" / "checkpoints"
    (checkpoints_dir / "last").symlink_to("020000")

    report = _scan(tmp_path)
    last = next(item for item in report["checkpoints"] if item["stepLabel"] == "last")

    assert last["aliasOf"] == "020000"
    # The step number comes from the target, not from the unparseable name.
    assert last["step"] == 20000


def test_scan_prefers_the_per_job_inference_config(tmp_path: Path):
    """Two jobs can share a view; each writes its own config under runs/<job>/."""
    view = _make_view(tmp_path, "v1")
    (view / "inference_config.generated.yaml").write_text(
        INFERENCE_CONFIG_TEMPLATE.format(frame="pika_task_tcp"), encoding="utf-8"
    )
    _make_checkpoint(tmp_path, "job_a", "020000", view, frame="pika_gripper_ee")

    entry = _entry(tmp_path, "job_a/020000")

    assert entry["contract"]["targetFrameName"] == "pika_gripper_ee"
    assert entry["inferenceConfigPath"].endswith("runs/job_a/inference_config.generated.yaml")


def test_scan_falls_back_to_the_view_wide_config(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    (view / "inference_config.generated.yaml").write_text(
        INFERENCE_CONFIG_TEMPLATE.format(frame="pika_gripper_ee"), encoding="utf-8"
    )
    _make_checkpoint(tmp_path, "job_a", "020000", view, with_inference_config=False)

    entry = _entry(tmp_path, "job_a/020000")

    assert entry["contract"]["targetFrameName"] == "pika_gripper_ee"


def test_scan_of_a_machine_with_no_runs_is_empty_not_an_error(tmp_path: Path):
    report = _scan(tmp_path)

    assert report["ok"] is True
    assert report["checkpoints"] == []


# ------------------------------------------------------------- contract checking ---


def test_a_tool_frame_mismatch_blocks_the_rollout(tmp_path: Path):
    """The failure this whole module exists for: it would run, and be wrong by a fixed offset."""
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view, frame="pika_task_tcp")
    entry = _entry(tmp_path, "job_a/020000")

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)

    frame_issues = [issue for issue in issues if issue.field == "targetFrameName"]
    assert [issue.level for issue in frame_issues] == ["block"]
    assert "pika_task_tcp" in frame_issues[0].message
    assert checkpoint_backend.verdict_for(issues) == "block"


def test_a_matching_checkpoint_is_clean(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)

    assert issues == []
    assert checkpoint_backend.verdict_for(issues) == "ok"


def test_an_unknown_tool_frame_warns_rather_than_passing_silently(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view, with_inference_config=False)
    entry = _entry(tmp_path, "job_a/020000")

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)

    assert [issue.level for issue in issues] == ["warn"]
    assert checkpoint_backend.verdict_for(issues) == "warn"


def test_a_non_delta_action_mode_blocks_the_rollout(tmp_path: Path):
    """The runtime integrates actions as deltas whatever they actually encode."""
    view = _make_view(tmp_path, "v1", action_mode="absolute_ee")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)

    assert any(issue.field == "actionMode" and issue.level == "block" for issue in issues)


def test_a_camera_key_mismatch_blocks_the_rollout(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")
    one_camera_rig = checkpoint_backend.RigContract(
        robotIp="192.168.1.206", targetFrameName="pika_gripper_ee", cameraKeys=["ee"]
    )

    issues = checkpoint_backend.check_contract(entry, rig=one_camera_rig, local=True)

    assert any(issue.field == "cameras" and issue.level == "block" for issue in issues)


def test_a_remote_checkpoint_cannot_be_rolled_out_where_it_sits(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=False)

    assert any(issue.field == "location" and issue.level == "block" for issue in issues)


def test_a_missing_view_blocks_the_rollout(tmp_path: Path):
    """The runtime reads episode start poses out of the dataset to place the trajectory."""
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")
    import shutil

    shutil.rmtree(view)
    entry["view"] = {}

    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)

    assert any(issue.field == "view" and issue.level == "block" for issue in issues)


def test_a_robot_ip_mismatch_warns_but_does_not_block(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    entry = _entry(tmp_path, "job_a/020000")
    other_rig = checkpoint_backend.RigContract(
        robotIp="192.168.11.102", targetFrameName="pika_gripper_ee", cameraKeys=["ee", "side"]
    )

    issues = checkpoint_backend.check_contract(entry, rig=other_rig, local=True)

    assert [issue.level for issue in issues if issue.field == "robotIp"] == ["warn"]


# --------------------------------------------------------------------- ids & fs ---


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "/abs/path", "job", "a/b/c", "job/..", "job/../..", "", "job/ step"],
)
def test_checkpoint_ids_that_could_escape_outputs_train_are_refused(hostile: str):
    with pytest.raises(checkpoint_backend.CheckpointError):
        checkpoint_backend.validate_checkpoint_id(hostile)


def test_deleting_last_is_refused_because_it_frees_nothing(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "last").symlink_to("020000")

    with pytest.raises(checkpoint_backend.CheckpointError, match="symlink"):
        checkpoint_backend.delete_checkpoint(tmp_path, "job_a/last")

    assert (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "020000").is_dir()


def test_deleting_a_checkpoint_reports_what_it_freed(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)

    result = checkpoint_backend.delete_checkpoint(tmp_path, "job_a/020000")

    assert result["freedBytes"] > 1000
    assert not (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "020000").exists()


def test_deleting_a_checkpoint_that_is_not_there_is_an_operator_error(tmp_path: Path):
    with pytest.raises(checkpoint_backend.CheckpointError, match="No checkpoint directory"):
        checkpoint_backend.delete_checkpoint(tmp_path, "job_a/020000")


def test_a_batch_delete_frees_every_checkpoint_it_names(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    for step in ("005000", "010000", "015000"):
        _make_checkpoint(tmp_path, "job_a", step, view)

    result = checkpoint_backend.delete_checkpoints(
        tmp_path, ["job_a/005000", "job_a/010000", "job_a/015000"]
    )

    assert result["ok"] is True
    assert result["deleted"] == ["job_a/005000", "job_a/010000", "job_a/015000"]
    assert result["failed"] == []
    assert result["freedBytes"] > 3000
    assert list((tmp_path / "outputs" / "train" / "job_a" / "checkpoints").iterdir()) == []


def test_a_batch_delete_keeps_going_past_the_one_it_cannot_delete(tmp_path: Path):
    """The bytes of a good checkpoint should not be held hostage by a bad id next to it.

    Nothing about deleting one directory makes deleting the next one wrong, and there is no
    undo once they are gone -- so the batch reports per id instead of aborting and leaving the
    operator to work out by hand which half went.
    """
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "005000", view)
    _make_checkpoint(tmp_path, "job_a", "015000", view)
    (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "last").symlink_to("015000")

    result = checkpoint_backend.delete_checkpoints(
        tmp_path, ["job_a/005000", "job_a/last", "job_a/999999", "../escape/x"]
    )

    assert result["ok"] is True
    assert result["deleted"] == ["job_a/005000"]
    assert [item["checkpointId"] for item in result["failed"]] == [
        "job_a/last",
        "job_a/999999",
        "../escape/x",
    ]
    assert not (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "005000").exists()
    assert (tmp_path / "outputs" / "train" / "job_a" / "checkpoints" / "015000").is_dir()


def test_a_batch_that_deletes_nothing_is_not_reported_as_success(tmp_path: Path):
    result = checkpoint_backend.delete_checkpoints(tmp_path, ["job_a/005000"])

    assert result["ok"] is False
    assert result["deleted"] == []
    assert result["freedBytes"] == 0


def test_a_batch_delete_ignores_a_repeated_id_rather_than_failing_on_the_second(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "005000", view)

    result = checkpoint_backend.delete_checkpoints(
        tmp_path, ["job_a/005000", "job_a/005000", " "]
    )

    assert result["deleted"] == ["job_a/005000"]
    assert result["failed"] == []


def test_an_empty_batch_is_refused_rather_than_reported_as_a_no_op(tmp_path: Path):
    with pytest.raises(checkpoint_backend.CheckpointError, match="No checkpoints were selected"):
        checkpoint_backend.delete_checkpoints(tmp_path, [])


def test_fetch_refuses_a_local_host(tmp_path: Path):
    host = training_backend.local_host(tmp_path)

    with pytest.raises(checkpoint_backend.CheckpointError, match="already on this machine"):
        checkpoint_backend.fetch_checkpoint(tmp_path, host, {"id": "job_a/020000"})


def test_fetch_refuses_a_relative_remote_path(tmp_path: Path):
    host = training_backend.TrainingHost(
        id="h", label="h", kind="remote", sshTarget="u@h", repoDir="/srv/repo"
    )

    with pytest.raises(checkpoint_backend.CheckpointError, match="absolute"):
        checkpoint_backend.fetch_checkpoint(
            tmp_path, host, {"id": "job_a/020000", "pretrainedPath": "outputs/train/x"}
        )


# --------------------------------------------------------------- launch command ---


def _command(tmp_path: Path, **overrides):
    (tmp_path / "tools" / "fr3").mkdir(parents=True, exist_ok=True)
    (tmp_path / rollout_backend.LAUNCHER).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    kwargs = {
        "mode": "real",
        "checkpoint_path": "outputs/train/job_a/checkpoints/020000",
        "dataset_root": "/repo/outputs/exports/training_views/v1",
        "target_frame_name": "pika_gripper_ee",
        "robot_ip": "192.168.1.206",
        "camera_config": "tools/fr3/fr3_il_infer_realsense_camera_config.yaml",
    }
    kwargs.update(overrides)
    return rollout_backend.build_rollout_command(tmp_path, **kwargs)


def test_the_launcher_is_told_the_checkpoints_own_tool_frame(tmp_path: Path):
    """Not the rig default: the frame the dataset was anchored to is what the deltas mean."""
    _, env = _command(tmp_path, target_frame_name="pika_task_tcp")

    assert env["FR3_TARGET_FRAME_NAME"] == "pika_task_tcp"


def test_the_launcher_is_told_the_local_view_path(tmp_path: Path):
    """A checkpoint's train_config.json records an absolute path on whatever machine trained it."""
    _, env = _command(tmp_path)

    assert env["FR3_INFER_DATASET_ROOT"] == "/repo/outputs/exports/training_views/v1"


def test_the_browser_path_swaps_the_x_window_for_jpeg_frames(tmp_path: Path):
    command, _ = _command(tmp_path)

    assert command[:3] == ["bash", str(tmp_path / rollout_backend.LAUNCHER), "real"]
    # After the mode, so they land in the launcher's extra_args and override its own flags.
    assert "--no-camera-preview-window" in command
    assert command.index("--no-camera-preview-window") > command.index("real")
    assert "--preview-jpeg-dir" in command


def test_the_settings_dump_mode_gets_no_runtime_flags(tmp_path: Path):
    """`env` prints and exits; appending runtime flags to it would only be confusing."""
    command, _ = _command(tmp_path, mode="env")

    assert command == ["bash", str(tmp_path / rollout_backend.LAUNCHER), "env"]


def test_a_step_limit_only_appears_when_asked_for(tmp_path: Path):
    _, without = _command(tmp_path)
    _, with_limit = _command(tmp_path, mode="real_once", max_steps=300)

    assert "FR3_INFER_MAX_STEPS" not in without
    assert with_limit["FR3_INFER_MAX_STEPS"] == "300"


def test_browser_rollout_runtime_options_are_passed_to_the_launcher(tmp_path: Path):
    runtime_options = rollout_backend.sanitize_rollout_runtime_options(
        {
            "taskPrompt": "Pick up the peg and insert it fully into the hole.",
            "rtcMode": "auto",
            "rtcExecutionHorizon": 10,
            "rtcMaxGuidanceWeight": 10,
            "rtcPrefixAttentionSchedule": "EXP",
            "rtcReplanQueueSize": 30,
            "rtcInferenceDelaySteps": 1,
            "commandEmaAlpha": 0.2,
        }
    )

    _, env = _command(tmp_path, runtime_options=runtime_options)

    assert env["FR3_TASK_PROMPT"] == "Pick up the peg and insert it fully into the hole."
    assert env["FR3_RTC_MODE"] == "auto"
    assert env["FR3_RTC_EXECUTION_HORIZON"] == "10"
    assert env["FR3_RTC_MAX_GUIDANCE_WEIGHT"] == "10"
    assert env["FR3_RTC_PREFIX_ATTENTION_SCHEDULE"] == "EXP"
    assert env["FR3_RTC_REPLAN_QUEUE_SIZE"] == "30"
    assert env["FR3_RTC_INFERENCE_DELAY_STEPS"] == "1"
    assert env["FR3_COMMAND_EMA_ALPHA"] == "0.2"


def test_browser_rollout_runtime_options_clear_stale_shell_values(tmp_path: Path):
    _, env = _command(
        tmp_path,
        base_env={
            "FR3_TASK_PROMPT": "stale prompt",
            "FR3_RTC_MODE": "enabled",
            "FR3_RTC_EXECUTION_HORIZON": "3",
            "FR3_COMMAND_EMA_ALPHA": "0.9",
            "FR3_ACT_TEMPORAL_ENSEMBLE_COEFF": "0.01",
        },
    )

    for key in rollout_backend.ROLLOUT_RUNTIME_ENV_KEYS:
        assert key not in env


def test_rollout_runtime_options_reject_invalid_values():
    with pytest.raises(rollout_backend.RolloutError, match="rtcMode"):
        rollout_backend.sanitize_rollout_runtime_options({"rtcMode": "always"})
    with pytest.raises(rollout_backend.RolloutError, match="rtcExecutionHorizon"):
        rollout_backend.sanitize_rollout_runtime_options({"rtcExecutionHorizon": 0})
    with pytest.raises(rollout_backend.RolloutError, match="commandEmaAlpha"):
        rollout_backend.sanitize_rollout_runtime_options({"commandEmaAlpha": 1.5})


def test_an_unknown_mode_is_refused(tmp_path: Path):
    with pytest.raises(rollout_backend.RolloutError, match="Unknown rollout mode"):
        _command(tmp_path, mode="real_yolo")


def test_a_rollout_without_a_checkpoint_is_refused(tmp_path: Path):
    with pytest.raises(rollout_backend.RolloutError, match="needs a checkpoint"):
        _command(tmp_path, checkpoint_path="")


def test_every_mode_that_moves_the_arm_is_marked_as_such():
    """The page gates confirmation on this flag, so a wrong answer here moves an arm."""
    by_id = rollout_backend.MODES_BY_ID

    assert by_id["env"].movesArm is False
    assert by_id["smoke"].movesArm is False
    # preview homes the arm before running, which is motion even though no policy command lands.
    assert by_id["preview"].movesArm is True
    assert by_id["real"].movesArm is True
    assert by_id["real_once"].movesArm is True
    assert by_id["real_debug"].movesArm is True
    assert [mode.id for mode in rollout_backend.ROLLOUT_MODES if mode.interactive] == [
        "real",
        "real_debug",
    ]


# ------------------------------------------------------------------ log parsing ---


def test_a_step_line_reports_progress_without_resetting_the_rollout_index():
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] step=41 status=pass raw_ee=(0.4021, -0.0113, 0.2887) "
        "safe_ee=(0.4019, -0.0113, 0.2886) model_gripper_raw=0.02"
    )

    assert parsed == {"step": 41, "commandStatus": "pass"}


def test_preview_mode_step_lines_are_parsed_too():
    parsed = rollout_backend.parse_rollout_line("[PREVIEW] step=3 status=hold_first_frame raw_ee=(0,0,0)")

    assert parsed["step"] == 3
    assert parsed["commandStatus"] == "hold_first_frame"


def test_the_rollout_lifecycle_markers_drive_the_page_state():
    waiting = rollout_backend.parse_rollout_line(
        "[INFO] interactive_waiting_for_start press 's' to start, 'q' to quit."
    )
    started = rollout_backend.parse_rollout_line("[INFO] interactive_rollout_start index=3")
    ended = rollout_backend.parse_rollout_line("[INFO] interactive_rollout_end index=3 status=stopped")
    session_over = rollout_backend.parse_rollout_line("[INFO] interactive_rollouts=stopped")

    assert waiting["state"] == "waiting"
    assert started["state"] == "rolling" and started["rolloutIndex"] == 3 and started["step"] == 0
    assert ended["state"] == "waiting" and ended["lastRolloutStatus"] == "stopped"
    # This is what makes the page ask for an outcome exactly once per finished rollout.
    assert ended["pendingOutcomeFor"] == 3
    assert session_over["state"] == "complete"


def test_the_waiting_banner_says_whether_the_arm_is_still_at_its_start_pose():
    """The launcher homes once, before the runtime exists. Nothing else puts this back."""
    fresh = rollout_backend.parse_rollout_line(
        "[INFO] interactive_waiting_for_start arm_at_start=1 press 's' to start."
    )
    displaced = rollout_backend.parse_rollout_line(
        "[INFO] interactive_waiting_for_start arm_at_start=0 press 's' to start."
    )
    started = rollout_backend.parse_rollout_line("[INFO] interactive_rollout_start index=1")

    assert fresh["armAtStart"] is True
    assert displaced["armAtStart"] is False
    # Set on the *start* marker, so a session that dies mid-rollout still leaves the page
    # saying the arm is somewhere the next rollout should not begin from.
    assert started["armAtStart"] is False


def test_a_runtime_that_says_nothing_about_the_start_pose_is_not_taken_as_a_yes():
    """Unknown reads as "not at the start pose": being wrong costs one idempotent button press."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_waiting_for_start press 's' to start, 'q' to quit."
    )

    assert parsed["state"] == "waiting"
    assert parsed["armAtStart"] is False


def test_homing_is_its_own_state_because_the_arm_is_moving_during_it():
    homing = rollout_backend.parse_rollout_line(
        "[INFO] interactive_homing=start gripper_pos=0.512 (gripper is left as it is)"
    )
    done = rollout_backend.parse_rollout_line("[INFO] interactive_homing=done")
    failed = rollout_backend.parse_rollout_line(
        "[WARN] interactive_homing=failed details=FR3 did not reach the configured start pose"
    )

    # Not "waiting": waiting is the page's word for an arm that is parked and safe to reach into.
    assert homing["state"] == "homing"
    assert done["armAtStart"] is True
    # A failed home leaves the flag down, which is what keeps the warning on the page after this
    # message is overwritten by the waiting line that follows it.
    assert failed["armAtStart"] is False
    assert "did not reach" in failed["message"]
    # And it is not a state change: the runtime hands the session back rather than tearing down
    # a loaded policy, so the page must not report the run as finished either.
    assert "state" not in failed


def test_an_open_control_channel_is_not_yet_a_runtime_that_will_act_on_it():
    """The channel opens before the loop reaches its wait, and the wait clears what is pending.

    So a `start` sent on this line is read, cleared, and lost -- the click looks accepted and
    nothing happens. Reported as a message, never as a state, so it cannot enable Start.
    """
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollouts=enabled keyboard_backend=pipe start_key='s' stop_key='x'"
    )

    assert "state" not in parsed
    assert "pipe" in parsed["message"]


def test_per_step_telemetry_is_kept_out_of_the_page_log_tail():
    assert rollout_backend.is_noise("[INFO] step=41 status=pass raw_ee=(0,0,0)")
    assert rollout_backend.is_noise("[PREVIEW] step=1 status=pass")
    assert not rollout_backend.is_noise("[INFO] interactive_rollout_start index=1")
    assert not rollout_backend.is_noise("[WARN] safety clamp engaged")


# ----------------------------------------------------------------- outcome log ---


def test_an_outcome_is_appended_with_the_checkpoint_it_belongs_to(tmp_path: Path):
    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {"checkpointId": "job_a/020000", "outcome": "success", "mode": "real", "steps": 214,
         "note": "picked and placed"},
    )

    assert entry["outcome"] == "success"
    stored = checkpoint_backend.load_rollout_outcomes(tmp_path)
    assert len(stored) == 1
    assert stored[0]["checkpointId"] == "job_a/020000"
    assert stored[0]["steps"] == 214


@pytest.mark.parametrize("bad", ["", "SUCCESS!", "partial", "ok"])
def test_an_outcome_outside_the_vocabulary_is_refused(tmp_path: Path, bad: str):
    with pytest.raises(checkpoint_backend.CheckpointError, match="Outcome must be"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "outcome": bad}
        )


def test_an_outcome_for_a_bogus_checkpoint_id_is_refused(tmp_path: Path):
    with pytest.raises(checkpoint_backend.CheckpointError):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "../../etc/passwd", "outcome": "success"}
        )


def test_aborted_rollouts_do_not_count_against_the_success_rate(tmp_path: Path):
    """Stopping because someone walked into the cell says nothing about the policy."""
    for outcome in ("success", "success", "failure", "aborted"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "outcome": outcome}
        )

    summary = checkpoint_backend.outcome_summary(checkpoint_backend.load_rollout_outcomes(tmp_path))

    assert summary["job_a/020000"] == {"success": 2, "failure": 1, "aborted": 1, "total": 4}


def test_outcomes_are_tallied_per_checkpoint(tmp_path: Path):
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "outcome": "success"}
    )
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/010000", "outcome": "failure"}
    )

    summary = checkpoint_backend.outcome_summary(checkpoint_backend.load_rollout_outcomes(tmp_path))

    assert summary["job_a/020000"]["success"] == 1
    assert summary["job_a/010000"]["failure"] == 1


def test_one_corrupt_line_does_not_hide_every_rollout_recorded_before_it(tmp_path: Path):
    """A half-written append during a crash must not erase the run's history."""
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "outcome": "success"}
    )
    path = checkpoint_backend.rollout_log_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"checkpointId": "job_a/0200\n')
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "outcome": "failure"}
    )

    stored = checkpoint_backend.load_rollout_outcomes(tmp_path)

    assert [item["outcome"] for item in stored] == ["success", "failure"]


def test_the_outcome_log_survives_the_checkpoint_it_describes(tmp_path: Path):
    """Recorded outside outputs/train on purpose -- deleting weights must not erase the record."""
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "outcome": "success"}
    )

    checkpoint_backend.delete_checkpoint(tmp_path, "job_a/020000")

    assert len(checkpoint_backend.load_rollout_outcomes(tmp_path)) == 1


def test_no_log_yet_is_an_empty_history_not_a_failure(tmp_path: Path):
    assert checkpoint_backend.load_rollout_outcomes(tmp_path) == []
    assert checkpoint_backend.outcome_summary([]) == {}


# ------------------------------------------------------------ gateway integration ---

import shutil  # noqa: E402
import time  # noqa: E402

from tools.data_collection_gui import gateway  # noqa: E402

CAMERA_CONFIG_YAML = """robot:
  cameras:
    ee:
      type: intelrealsense
      serial_number_or_name: "315122271876"
    side:
      type: intelrealsense
      serial_number_or_name: "243122071795"
"""

# Stands in for run_pick_place_infer_workstation.sh. It reproduces the two behaviours the
# gateway depends on -- the runtime's lifecycle markers on stdout, and one control word per
# line on stdin -- so the control path is exercised for real rather than mocked.
FAKE_LAUNCHER = """#!/usr/bin/env bash
echo "[INFO] interactive_rollouts=enabled keyboard_backend=pipe start_key='s'"
echo "[INFO] interactive_waiting_for_start arm_at_start=1 press 's' to start."
while IFS= read -r line; do
  case "$line" in
    start) echo "[INFO] interactive_rollout_start index=1"; echo "[INFO] step=1 status=pass" ;;
    stop)  echo "[INFO] interactive_rollout_end index=1 status=stopped"
           echo "[INFO] interactive_waiting_for_start arm_at_start=0 press 's' to start." ;;
    home)  echo "[INFO] interactive_homing=start gripper_pos=0.500 (gripper is left as it is)"
           echo "[INFO] interactive_homing=done"
           echo "[INFO] interactive_waiting_for_start arm_at_start=1 press 's' to start." ;;
    quit)  echo "[INFO] interactive_rollouts=stopped"; exit 0 ;;
  esac
done
"""


def _rollout_state(tmp_path: Path) -> gateway.GatewayState:
    repo = tmp_path / "repo"
    (repo / "tools" / "fr3").mkdir(parents=True)
    shutil.copy(SCAN_SCRIPT, repo / "tools" / "fr3" / "scan_checkpoints.py")
    (repo / "tools" / "fr3" / "fr3_il_infer_realsense_camera_config.yaml").write_text(
        CAMERA_CONFIG_YAML, encoding="utf-8"
    )
    launcher = repo / rollout_backend.LAUNCHER
    launcher.write_text(FAKE_LAUNCHER, encoding="utf-8")
    launcher.chmod(0o755)

    view = _make_view(repo, "v1")
    _make_checkpoint(repo, "job_a", "020000", view)

    state = gateway.GatewayState(
        repo_root=repo,
        config_path=repo / "config.yaml",
        config={
            "robot": {"robot_ip": "192.168.1.206", "target_frame_name": "pika_gripper_ee"},
            "dataset": {"repo_id": "local/test", "root": str(repo / "outputs" / "datasets" / "d")},
        },
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
        datasets_root=repo / "outputs" / "datasets",
    )
    state.profile = "workstation"
    return state


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_the_rig_contract_is_read_from_the_gateways_own_config(tmp_path: Path):
    state = _rollout_state(tmp_path)

    rig = gateway._rig_contract(state)

    assert rig.robotIp == "192.168.1.206"
    assert rig.targetFrameName == "pika_gripper_ee"
    assert rig.cameraKeys == ["ee", "side"]


def test_a_listed_checkpoint_carries_its_verdict(tmp_path: Path):
    state = _rollout_state(tmp_path)

    listing = gateway._checkpoint_entries(state, training_backend.LOCAL_HOST_ID)

    assert listing["ok"] is True
    entry = next(item for item in listing["checkpoints"] if item["id"] == "job_a/020000")
    assert entry["verdict"] == "ok"
    assert entry["issues"] == []
    assert entry["hostId"] == training_backend.LOCAL_HOST_ID


def test_a_mismatched_checkpoint_is_listed_with_a_blocking_verdict(tmp_path: Path):
    """Listed, not hidden: the operator has to be able to see why it cannot be used."""
    state = _rollout_state(tmp_path)
    view = _make_view(state.repo_root, "v2")
    _make_checkpoint(state.repo_root, "job_b", "010000", view, frame="pika_task_tcp")

    listing = gateway._checkpoint_entries(state, training_backend.LOCAL_HOST_ID)

    entry = next(item for item in listing["checkpoints"] if item["id"] == "job_b/010000")
    assert entry["verdict"] == "block"
    assert any(issue["field"] == "targetFrameName" for issue in entry["issues"])


def test_a_motion_mode_is_refused_without_an_explicit_confirmation(tmp_path: Path):
    state = _rollout_state(tmp_path)

    with pytest.raises(ValueError, match="Confirm motion"):
        gateway._start_rollout(state, {"mode": "real", "checkpointId": "job_a/020000"})

    assert state.rollout_process is None


def test_a_non_motion_mode_needs_no_confirmation(tmp_path: Path):
    state = _rollout_state(tmp_path)

    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    assert state.rollout_process is not None
    gateway._stop_rollout(state)


def test_a_rollout_remembers_its_settings_for_the_next_one(tmp_path: Path):
    # Tuning means rolling out the same knobs against checkpoint after checkpoint. Retyping eight
    # RTC values each time is how two runs meant to be comparable end up subtly different.
    state = _rollout_state(tmp_path)

    gateway._start_rollout(
        state,
        {
            "mode": "smoke",
            "checkpointId": "job_a/020000",
            "maxSteps": 500,
            "moveToStart": False,
            "runtimeOptions": {
                "rtcMode": "auto",
                "rtcExecutionHorizon": 16,
                "rtcReplanQueueSize": 25,
                "rtcInferenceDelaySteps": None,
            },
        },
    )
    gateway._stop_rollout(state)

    params = gateway._rollout_last_params(state)
    assert params["mode"] == "smoke"
    assert params["maxSteps"] == 500
    assert params["moveToStart"] is False
    assert params["runtimeOptions"]["rtcExecutionHorizon"] == 16
    # None is a real recorded value -- "let the runtime estimate the delay" -- not a missing key.
    assert params["runtimeOptions"]["rtcInferenceDelaySteps"] is None


def test_the_safety_gates_are_never_carried_to_the_next_rollout(tmp_path: Path):
    # confirmMotion is what stands between a click and an arm that moves; overrideContract is
    # what stands between a click and a checkpoint the rig reported as mismatched. A remembered
    # "yes" is a gate that answers itself.
    state = _rollout_state(tmp_path)

    gateway._start_rollout(
        state,
        {
            "mode": "real",
            "checkpointId": "job_a/020000",
            "confirmMotion": True,
            "overrideContract": True,
        },
    )
    gateway._stop_rollout(state)

    params = gateway._rollout_last_params(state)
    assert "confirmMotion" not in params
    assert "overrideContract" not in params
    # Nor the checkpoint: the next rollout is almost always a different one.
    assert "checkpointId" not in params


def test_a_hand_written_params_file_cannot_smuggle_a_gate_back_in(tmp_path: Path):
    state = _rollout_state(tmp_path)
    path = gateway._rollout_last_params_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "smoke",
                "confirmMotion": True,
                "overrideContract": True,
                "checkpointId": "job_a/020000",
                "runtimeOptions": "not-an-object",
            }
        ),
        encoding="utf-8",
    )

    params = gateway._rollout_last_params(state)

    assert params == {"mode": "smoke"}


def test_no_previous_rollout_means_no_carried_settings(tmp_path: Path):
    state = _rollout_state(tmp_path)

    assert gateway._rollout_last_params(state) == {}


def test_a_blocked_contract_is_refused_unless_overridden(tmp_path: Path):
    state = _rollout_state(tmp_path)
    view = _make_view(state.repo_root, "v2")
    _make_checkpoint(state.repo_root, "job_b", "010000", view, frame="pika_task_tcp")

    with pytest.raises(ValueError, match="does not match the rig"):
        gateway._start_rollout(
            state, {"mode": "smoke", "checkpointId": "job_b/010000"}
        )


def test_an_overridden_contract_uses_the_checkpoints_own_tool_frame(tmp_path: Path):
    """Overriding says "I know"; it does not mean "use the rig's frame anyway"."""
    state = _rollout_state(tmp_path)
    view = _make_view(state.repo_root, "v2")
    _make_checkpoint(state.repo_root, "job_b", "010000", view, frame="pika_task_tcp")

    gateway._start_rollout(
        state,
        {"mode": "smoke", "checkpointId": "job_b/010000", "overrideContract": True},
    )

    assert state.rollout.targetFrameName == "pika_task_tcp"
    gateway._stop_rollout(state)


def test_a_checkpoint_that_is_not_on_this_machine_cannot_be_rolled_out(tmp_path: Path):
    state = _rollout_state(tmp_path)

    with pytest.raises(ValueError, match="Fetch it from its training host"):
        gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_zz/020000"})


def test_a_rollout_will_not_start_while_training_holds_the_gpu(tmp_path: Path, monkeypatch):
    """A policy starved of inference time still sends commands, just late."""
    state = _rollout_state(tmp_path)
    monkeypatch.setattr(gateway, "_training_is_running", lambda _state: True)

    with pytest.raises(ValueError, match="training run is using the GPU"):
        gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})


def test_rollouts_are_refused_off_the_workstation_profile(tmp_path: Path):
    state = _rollout_state(tmp_path)
    state.profile = "thor"

    with pytest.raises(ValueError, match="workstation profile"):
        gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})


def test_the_rollout_owns_its_stdin_and_not_its_stdout(tmp_path: Path):
    """The asymmetry is the design.

    stdout through a pipe is what killed training runs on every gateway restart (SIGPIPE on the
    next write). stdin through a pipe is the only way to stop a moving arm from a browser -- and
    its closing on gateway death is a feature, not a leak.
    """
    state = _rollout_state(tmp_path)

    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    process = state.rollout_process
    assert process is not None
    assert process.stdin is not None
    assert process.stdout is None
    assert Path(state.rollout.logPath).is_file()
    gateway._stop_rollout(state)


def test_start_and_stop_reach_the_running_rollout(tmp_path: Path):
    state = _rollout_state(tmp_path)
    gateway._start_rollout(
        state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True}
    )

    assert _wait_for(lambda: state.rollout.state == "waiting"), state.rollout.message
    gateway._send_rollout_control(state, "start")
    assert _wait_for(lambda: state.rollout.state == "rolling"), state.rollout.message
    assert state.rollout.rolloutIndex == 1

    gateway._send_rollout_control(state, "stop")
    assert _wait_for(lambda: state.rollout.pendingOutcomeFor == 1), state.rollout.message
    assert state.rollout.lastRolloutStatus == "stopped"
    gateway._stop_rollout(state)


def test_control_is_refused_for_a_mode_that_runs_to_completion(tmp_path: Path):
    state = _rollout_state(tmp_path)
    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    with pytest.raises(ValueError, match="not an interactive mode"):
        gateway._send_rollout_control(state, "start")

    gateway._stop_rollout(state)


def test_move_to_start_reaches_the_arm_between_rollouts(tmp_path: Path):
    """The whole point of the button: home the arm without dropping a loaded policy."""
    state = _rollout_state(tmp_path)
    gateway._start_rollout(
        state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True}
    )
    assert _wait_for(lambda: state.rollout.state == "waiting"), state.rollout.message

    gateway._send_rollout_control(state, "start")
    assert _wait_for(lambda: state.rollout.state == "rolling"), state.rollout.message
    gateway._send_rollout_control(state, "stop")
    # Both halves matter: the stop has landed (waiting), and the page now says the arm is not
    # where the episodes began. armAtStart alone would pass while the rollout was still running.
    assert _wait_for(
        lambda: state.rollout.state == "waiting" and state.rollout.armAtStart is False
    ), state.rollout.message

    gateway._send_rollout_control(state, "home")
    assert _wait_for(lambda: state.rollout.armAtStart is True), state.rollout.message
    assert state.rollout.state == "waiting"
    # Still the same process, still holding the policy -- which is why this exists at all.
    assert state.rollout_process is not None and state.rollout_process.poll() is None
    gateway._stop_rollout(state)


def test_move_to_start_is_refused_while_a_rollout_is_running(tmp_path: Path):
    """Rejected with a reason rather than queued: an arm that homes itself the moment a rollout
    ends, seconds after the operator stopped watching, is what interactive mode exists to
    prevent. The runtime drops it too; this one is here so the click does not look accepted."""
    state = _rollout_state(tmp_path)
    gateway._start_rollout(
        state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True}
    )
    assert _wait_for(lambda: state.rollout.state == "waiting"), state.rollout.message
    gateway._send_rollout_control(state, "start")
    assert _wait_for(lambda: state.rollout.state == "rolling"), state.rollout.message

    with pytest.raises(ValueError, match="only runs between rollouts"):
        gateway._send_rollout_control(state, "home")

    assert state.rollout.state == "rolling"
    gateway._stop_rollout(state)


@pytest.mark.parametrize("bad", ["", "s", "h", "START", "rm -rf /", "quit\nstart"])
def test_only_the_control_vocabulary_is_accepted(tmp_path: Path, bad: str):
    state = _rollout_state(tmp_path)
    gateway._start_rollout(
        state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True}
    )

    with pytest.raises(ValueError, match="Rollout control must be one of"):
        gateway._send_rollout_control(state, bad)

    gateway._stop_rollout(state)


def test_control_is_refused_when_nothing_is_running(tmp_path: Path):
    state = _rollout_state(tmp_path)

    with pytest.raises(ValueError, match="No rollout is running"):
        gateway._send_rollout_control(state, "start")


def test_a_second_rollout_cannot_start_over_a_running_one(tmp_path: Path):
    state = _rollout_state(tmp_path)
    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    with pytest.raises(ValueError, match="already running"):
        gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    gateway._stop_rollout(state)


def test_a_late_log_line_does_not_undo_a_stop(tmp_path: Path):
    """The runtime keeps writing for a moment after the operator stops it."""
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="stopped", interactive=True)

    gateway._apply_rollout_output(state, "[INFO] interactive_rollout_start index=2")

    assert state.rollout.state == "stopped"
    # Non-state fields still update: the index is a fact about what happened.
    assert state.rollout.rolloutIndex == 2


def test_step_limited_and_leashed_steps_are_counted_apart(tmp_path: Path):
    """One counter cannot carry both.

    A step-limited command says the policy asked for too much motion; a leashed one says the arm
    stopped following. Summing them into a single "clamped" number is how a rollout whose arm was
    tracking fine came back reading 299 of 299 clamped, with nothing to say which had happened.
    """
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="rolling")

    for status in ("pass", "step_limited", "step_limited", "leash_limited", "pass"):
        gateway._apply_rollout_output(state, f"[INFO] step=1 status={status} raw_ee=(0,0,0)")

    assert state.rollout.clampedSteps == 2
    assert state.rollout.leashedSteps == 1
    # And none of that telemetry reached the page's log tail.
    assert state.rollout.lastLines == []


def test_a_stale_preview_frame_is_not_served_as_live(tmp_path: Path, monkeypatch):
    """A JPEG left in /dev/shm after a run ends looks identical to a live one."""
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    frame = preview_dir / "ee.jpg"
    frame.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)

    assert gateway._rollout_preview_frame("ee") == b"\xff\xd8\xff"

    stale = time.time() - (rollout_backend.PREVIEW_STALE_S + 1)
    os.utime(frame, (stale, stale))
    assert gateway._rollout_preview_frame("ee") is None


def test_recording_an_outcome_clears_the_prompt(tmp_path: Path):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(
        checkpointId="job_a/020000", mode="real", step=214, pendingOutcomeFor=1
    )

    result = gateway._record_rollout_outcome(state, {"outcome": "success", "note": "clean pick"})

    assert result["entry"]["outcome"] == "success"
    assert result["entry"]["steps"] == 214
    assert state.rollout.pendingOutcomeFor == 0


def test_the_gateways_own_environment_cannot_override_the_checkpoints_contract(tmp_path: Path):
    """A stray FR3_* in the gateway's environment must not decide what the arm does.

    The rollout settings are derived from the checkpoint; anything inherited is context. If
    inheritance won, a variable exported once in the shell that launched the gateway would
    quietly redirect every rollout afterwards, including onto the wrong tool frame.
    """
    _, env = _command(
        tmp_path,
        target_frame_name="pika_gripper_ee",
        base_env={
            "FR3_TARGET_FRAME_NAME": "pika_task_tcp",
            "FR3_INFER_CHECKPOINT": "outputs/train/someone_elses/checkpoints/last",
            "FR3_ROBOT_IP": "192.168.11.102",
            "PYTHONPATH": "/repo/src",
        },
    )

    assert env["FR3_TARGET_FRAME_NAME"] == "pika_gripper_ee"
    assert env["FR3_INFER_CHECKPOINT"] == "outputs/train/job_a/checkpoints/020000"
    assert env["FR3_ROBOT_IP"] == "192.168.1.206"
    # Everything the caller needs the process to inherit still comes through.
    assert env["PYTHONPATH"] == "/repo/src"


def test_an_inherited_step_limit_does_not_bound_an_unbounded_rollout(tmp_path: Path):
    """A truncated rollout looks exactly like a policy that stopped on its own."""
    _, env = _command(tmp_path, mode="real", base_env={"FR3_INFER_MAX_STEPS": "50"})

    assert "FR3_INFER_MAX_STEPS" not in env


# ------------------------------------------------------------- relocated views ---


def test_a_fetched_checkpoint_finds_the_view_that_came_with_it(tmp_path: Path):
    """A checkpoint records its dataset as an absolute path on the machine that trained it.

    Fetch it to a rig whose checkout is somewhere else and that path is gone, even though the
    view was fetched alongside and sits in this repo's views directory under the same name.
    Without this fallback every remotely-trained checkpoint would arrive unusable.
    """
    trained_on = tmp_path / "training_box"
    view = _make_view(trained_on, "v1")
    _make_checkpoint(trained_on, "job_a", "020000", view)

    rig = tmp_path / "rig"
    # What a fetch produces: weights under this repo, view under this repo's views directory.
    shutil.copytree(
        trained_on / "outputs" / "train", rig / "outputs" / "train"
    )
    shutil.copytree(view, rig / "outputs" / "exports" / "training_views" / "v1")
    # The training machine is a different machine: its paths do not exist here.
    shutil.rmtree(trained_on)

    entry = _entry(rig, "job_a/020000")

    assert entry["view"]["exists"] is True
    assert entry["view"]["relocated"] is True
    assert entry["view"]["actionMode"] == "delta_ee_from_prev_cmd"
    assert entry["datasetRoot"] == str(rig / "outputs" / "exports" / "training_views" / "v1")
    # The path the checkpoint itself carries is kept, so the move stays visible.
    assert entry["recordedDatasetRoot"] == str(view)
    assert checkpoint_backend.check_contract(entry, rig=RIG, local=True) == []


def test_a_relocated_view_is_only_matched_by_name(tmp_path: Path):
    """Never by search. A rollout against someone else's episodes resolves a start pose from
    them and places the whole trajectory there -- wrong, and it would run."""
    trained_on = tmp_path / "training_box"
    view = _make_view(trained_on, "v1")
    _make_checkpoint(trained_on, "job_a", "020000", view)

    rig = tmp_path / "rig"
    shutil.copytree(trained_on / "outputs" / "train", rig / "outputs" / "train")
    # A different view is present locally. It must not be adopted.
    _make_view(rig, "some_other_view")
    shutil.rmtree(trained_on)

    entry = _entry(rig, "job_a/020000")

    assert entry["view"] == {}
    issues = checkpoint_backend.check_contract(entry, rig=RIG, local=True)
    assert any(issue.field == "view" and issue.level == "block" for issue in issues)


def test_a_view_at_its_recorded_path_is_not_reported_as_relocated(tmp_path: Path):
    view = _make_view(tmp_path, "v1")
    _make_checkpoint(tmp_path, "job_a", "020000", view)

    entry = _entry(tmp_path, "job_a/020000")

    assert entry["view"]["relocated"] is False
    assert entry["datasetRoot"] == entry["recordedDatasetRoot"]


def test_fetching_last_stores_it_under_the_step_it_points_at(tmp_path: Path, monkeypatch):
    """`last` is a symlink remotely and a real directory once copied.

    The copy's name carries no step, and the file where the trainer records one lives in
    training_state, which a fetch deliberately does not bring over. Landing it as its number
    keeps the step readable -- and stops a later fetch of a newer `last` from overwriting it.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(
        checkpoint_backend, "_rsync", lambda paths, *, timeout_s: calls.append(paths) or []
    )
    host = training_backend.TrainingHost(
        id="h", label="box", kind="remote", sshTarget="u@h", repoDir="/srv/repo"
    )

    result = checkpoint_backend.fetch_checkpoint(
        tmp_path,
        host,
        {
            "id": "job_a/last",
            "aliasOf": "020000",
            "pretrainedPath": "/srv/repo/outputs/train/job_a/checkpoints/last/pretrained_model",
            "view": {"root": "/srv/repo/outputs/exports/training_views/v1"},
        },
    )

    assert result["checkpointId"] == "job_a/020000"
    assert result["localPath"].endswith("outputs/train/job_a/checkpoints/020000")
    # The remote source is still `last`: that is where the bytes are.
    assert calls[0][0].endswith("/checkpoints/last/pretrained_model/")


def test_a_fetch_brings_the_views_metadata_but_not_its_videos(tmp_path: Path, monkeypatch):
    """The runtime reads episode start states out of the parquet and never opens the videos.

    In an exported view those files are symlinks into the source dataset, which would not
    resolve here anyway.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(
        checkpoint_backend, "_rsync", lambda paths, *, timeout_s: calls.append(paths) or []
    )
    host = training_backend.TrainingHost(
        id="h", label="box", kind="remote", sshTarget="u@h", repoDir="/srv/repo"
    )

    checkpoint_backend.fetch_checkpoint(
        tmp_path,
        host,
        {
            "id": "job_a/020000",
            "pretrainedPath": "/srv/repo/outputs/train/job_a/checkpoints/020000/pretrained_model",
            "view": {"root": "/srv/repo/outputs/exports/training_views/v1"},
        },
    )

    sources = [pair[0] for pair in calls]
    assert any(src.endswith("/v1/meta/") for src in sources)
    assert any(src.endswith("/v1/data/") for src in sources)
    assert not any("videos" in src for src in sources)


def test_real_debug_is_refused_when_there_is_no_display(tmp_path: Path, monkeypatch):
    """The viewer is the only thing separating real_debug from real.

    Starting it headless would home the arm and run rollouts while the operator waits for a
    window that never opens -- and the MuJoCo viewer is what they were watching for.
    """
    state = _rollout_state(tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)

    with pytest.raises(ValueError, match="no\\s+X display"):
        gateway._start_rollout(
            state,
            {"mode": "real_debug", "checkpointId": "job_a/020000", "confirmMotion": True},
        )

    assert state.rollout_process is None


def test_real_debug_starts_when_a_display_is_present(tmp_path: Path, monkeypatch):
    state = _rollout_state(tmp_path)
    monkeypatch.setenv("DISPLAY", ":0")

    gateway._start_rollout(
        state, {"mode": "real_debug", "checkpointId": "job_a/020000", "confirmMotion": True}
    )

    assert state.rollout.mode == "real_debug"
    gateway._stop_rollout(state)


def test_the_other_modes_do_not_need_a_display(tmp_path: Path, monkeypatch):
    """Everything else renders offscreen or opens no window at all."""
    state = _rollout_state(tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)

    gateway._start_rollout(
        state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True}
    )

    assert state.rollout.mode == "real"
    gateway._stop_rollout(state)


# ------------------------------------------------------------- landing points ---
#
# The map these feed exists because a success rate cannot distinguish six failures spread across
# the workspace from six at the same spot, and those have opposite causes. Everything below is
# about the geometry surviving the trip from the runtime's log line to the graded record without
# acquiring a coordinate the arm never visited.


def test_rollout_end_marker_carries_the_landing_points():
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=3 status=stopped samples=430 closed=1 "
        "grasp_xyz=0.3162,-0.2214,0.0461 release_xyz=0.3599,-0.1330,0.0510 "
        "apex_z=0.1201 lift_m=0.0740 descent_m=0.0691 held_steps=223"
    )

    assert parsed["rolloutIndex"] == 3
    assert parsed["pendingOutcomeFor"] == 3
    geometry = parsed["lastRolloutGeometry"]
    assert geometry["graspXyz"] == [0.3162, -0.2214, 0.0461]
    assert geometry["releaseXyz"] == [0.3599, -0.1330, 0.0510]
    assert geometry["closed"] is True
    assert geometry["descentM"] == pytest.approx(0.0691)
    assert geometry["heldSteps"] == 223


def test_a_rollout_that_never_closed_reports_where_it_reached():
    """The approach point is the whole result for a failure that never gripped.

    Reported as `approachXyz` rather than as a grasp point at the same coordinates, because the
    two mean different things: one is where the gripper closed, the other is where it stopped
    without closing, and a map that drew them identically would hide the more common failure.
    """
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=4 status=stopped samples=380 closed=0 "
        "approach_xyz=0.4410,-0.2600,0.0523"
    )

    geometry = parsed["lastRolloutGeometry"]
    assert geometry["closed"] is False
    assert geometry["approachXyz"] == [0.4410, -0.2600, 0.0523]
    assert "graspXyz" not in geometry
    assert "descentM" not in geometry


def test_an_end_marker_without_geometry_still_parses():
    """A runtime older than this feature, or one whose rollout produced no samples."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=1 status=stopped"
    )

    assert parsed["pendingOutcomeFor"] == 1
    assert parsed["lastRolloutGeometry"] == {}


def test_starting_a_rollout_clears_the_previous_landing_point():
    """Otherwise the running rollout inherits the last one's dot until it finishes."""
    parsed = rollout_backend.parse_rollout_line("[INFO] interactive_rollout_start index=5")

    assert parsed["lastRolloutGeometry"] == {}


def test_recorded_outcomes_keep_the_landing_point(tmp_path: Path):
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "failure",
            "rolloutIndex": 7,
            "geometry": {
                "graspXyz": [0.31, -0.22, 0.046],
                "closed": True,
                "liftM": 0.074,
                "descentM": 0.069,
                "heldSteps": 223,
            },
        },
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert entry["rolloutIndex"] == 7
    assert entry["geometry"]["graspXyz"] == [0.31, -0.22, 0.046]
    assert entry["geometry"]["descentM"] == pytest.approx(0.069)


def test_a_recorded_outcome_cannot_smuggle_in_extra_geometry(tmp_path: Path):
    """The log grows only the fields the runtime measures."""
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "success",
            "geometry": {"graspXyz": [0.31, -0.22, 0.046], "operatorGuess": [1.0, 2.0, 3.0]},
        },
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert "operatorGuess" not in entry["geometry"]


def test_an_outcome_without_geometry_records_no_empty_point(tmp_path: Path):
    """A missing landing point has to stay missing: an absent field keeps the rollout off the
    map, while a zeroed one would put it at the base of the robot."""
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/030000", "outcome": "aborted"}
    )

    assert "geometry" not in checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]


def _write_demo_dataset(root: Path, episodes: dict[int, tuple[list[float], list[float]]]) -> Path:
    """A dataset shaped like the real one: absolute EE state, commanded gripper in the action."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _write_json(
        root / "meta" / "info.json",
        {
            "features": {
                "observation.state": {
                    "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
                },
                "action": {"names": ["dx", "dy", "dz", "drz", "gripper.pos"]},
            }
        },
    )
    episode_index: list[int] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for episode, (heights, gripper) in episodes.items():
        for height, grip in zip(heights, gripper, strict=True):
            episode_index.append(episode)
            # x and y move with the phase so grasp and release land in different places, the
            # way they do on the rig: pick out in the workspace, release into the fixed hole.
            grasping = grip < 0.5
            states.append(
                [0.36 if grasping and height < 0.07 and heights.index(height) > 3 else 0.31,
                 -0.13 if grasping and height < 0.07 and heights.index(height) > 3 else -0.22,
                 height, 0.0, 0.0, 0.0, 1.0, grip]
            )
            actions.append([0.0, 0.0, 0.0, 0.0, grip])
    table = pa.table(
        {
            "episode_index": pa.array(episode_index, pa.int64()),
            "observation.state": pa.array(states, pa.list_(pa.float32())),
            "action": pa.array(actions, pa.list_(pa.float32())),
        }
    )
    path = root / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return root


def test_demo_landing_points_reduce_each_episode_to_its_grasp_and_release(tmp_path: Path):
    """The backdrop of the map: one grasp point per demonstration, by the rule the runtime uses.

    Computed from the dataset rather than kept as a fixture because it describes the checkpoint
    under test -- a hard-coded region would keep describing whichever dataset it was written for.
    """
    heights = [0.20, 0.12, 0.05, 0.05, 0.12, 0.13, 0.06, 0.06, 0.14]
    gripper = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    root = _write_demo_dataset(tmp_path / "view", {0: (heights, gripper), 1: (heights, gripper)})

    landmarks = checkpoint_backend.demo_landing_points(root)

    assert len(landmarks["points"]) == 2
    point = landmarks["points"][0]
    assert point["graspXyz"] == [0.31, -0.22, 0.05]
    assert point["releaseXyz"][2] == pytest.approx(0.06)
    assert point["liftM"] == pytest.approx(0.08)
    # The number the roadmap once reported as ~1 mm by subtracting two table-height events. The
    # insertion travel is the drop from the apex, and it is the whole reason this is measured
    # rather than assumed.
    assert point["descentM"] == pytest.approx(0.07)
    assert landmarks["hole"] == pytest.approx(landmarks["points"][0]["releaseXyz"][:2])


def test_a_demonstration_that_starts_with_the_gripper_shut_grasps_on_the_falling_edge(
    tmp_path: Path,
):
    """The same edge rule the runtime uses, so the two sets of points on the map mean one thing.

    No demonstration in the current dataset starts shut, but a rollout did, and a backdrop
    derived by a rule that only coincidentally agrees with the runtime's is a backdrop that can
    stop agreeing without anybody noticing.
    """
    heights = [0.40, 0.20, 0.12, 0.05, 0.05, 0.12, 0.13, 0.06, 0.06, 0.14]
    gripper = [0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    root = _write_demo_dataset(tmp_path / "view", {0: (heights, gripper)})

    point = checkpoint_backend.demo_landing_points(root)["points"][0]

    # Not [.., 0.40], which is where "first step under the threshold" puts it: the start pose.
    assert point["graspXyz"] == [0.31, -0.22, 0.05]


def test_demo_landing_points_are_memoised_per_dataset(tmp_path: Path):
    """The scan is a full pass over the dataset's parquet; the page asks for it on every mount."""
    heights = [0.20, 0.05, 0.12, 0.06, 0.14]
    root = _write_demo_dataset(tmp_path / "view", {0: (heights, [1.0, 0.0, 0.0, 0.0, 1.0])})

    first = checkpoint_backend.demo_landing_points(root)
    second = checkpoint_backend.demo_landing_points(root)

    assert first is second


def test_demo_landing_points_on_a_dataset_that_is_not_there(tmp_path: Path):
    """A missing or unreadable dataset leaves the map without a backdrop rather than 500ing the
    page: the rollout points are still worth showing."""
    assert checkpoint_backend.demo_landing_points(tmp_path / "absent") == {}


# --------------------------------------------------------------- grading ladders ---
#
# The 20-rollout batch of 2026-08-31 came back 20/20 `failure` while breaking in three different
# places, so the tests below are mostly about the log refusing to store a grade whose meaning a
# later reader could not recover: a stage with no ladder, a stage this task does not have, or an
# outcome that disagrees with its own stage.


def _write_ladder(repo_root: Path, body: str, task: str = "demo_task") -> Path:
    path = repo_root / task_ladders.LADDER_DIR / f"{task}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


_DEMO_LADDER = """
task: demo_task
label: 测试任务
stages:
  - {id: approach, instance: 到了附近}
  - {id: contact, instance: 碰到了}
  - {id: secure, instance: 拿住了}
terminal: secure
blockers:
  - {id: object_pose_offset}
  - {id: unknown}
"""


def test_the_shipped_peg_ladder_loads(tmp_path: Path):
    """The file the rig actually grades against, checked here so a typo in it fails a test run
    rather than a rollout the operator has already performed."""
    ladder = task_ladders.find_ladder(REPO_ROOT, "insert_peg")

    assert ladder.terminal.id == "release_stable"
    # Ordinals are the shared scale, so they must survive the trip through the file unchanged.
    assert [stage.ordinal for stage in ladder.stages] == [1, 2, 3, 4, 5, 6, 7]
    assert "object_pose_offset" in ladder.blockers


def test_reaching_the_terminal_stage_is_what_makes_it_a_success(tmp_path: Path):
    _write_ladder(tmp_path, _DEMO_LADDER)

    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stageId": "secure"},
    )

    assert entry["outcome"] == "success"
    assert entry["stage"] == 3
    assert entry["stageId"] == "secure"


def test_falling_short_grades_as_failure_and_keeps_how_far_it_got(tmp_path: Path):
    """The point of the ladder: `failure` stops being the whole story."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/020000",
            "taskLadder": "demo_task",
            "stage": 2,
            "blocker": "object_pose_offset",
            "inDistribution": True,
        },
    )

    assert entry["outcome"] == "failure"
    assert entry["stage"] == 2
    assert entry["blocker"] == "object_pose_offset"
    assert entry["inDistribution"] is True


def test_an_outcome_that_contradicts_its_own_stage_is_refused(tmp_path: Path):
    """A record carrying both is a record whose halves can disagree. That is the defect this
    replaces: the batch filed a partially inserted peg under `failure` with the fact surviving
    only in prose."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    with pytest.raises(checkpoint_backend.CheckpointError, match="contradicts stage"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path,
            {
                "checkpointId": "job_a/020000",
                "taskLadder": "demo_task",
                "stage": 2,
                "outcome": "success",
            },
        )


def test_aborted_overrides_the_derived_outcome_and_still_records_the_stage(tmp_path: Path):
    """Someone walking into the cell says nothing about the policy, so the round must stay out
    of the rate -- but how far it had got before the stop is still worth keeping."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/020000",
            "taskLadder": "demo_task",
            "stage": 2,
            "outcome": "aborted",
            "blocker": "operator_stop",
        },
    )

    assert entry["outcome"] == "aborted"
    assert entry["stage"] == 2


def test_a_stage_without_a_ladder_is_refused(tmp_path: Path):
    """A bare ordinal is a number no later reader could turn back into a claim."""
    with pytest.raises(checkpoint_backend.CheckpointError, match="without a task ladder"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "outcome": "failure", "stage": 2}
        )


def test_a_stage_this_task_does_not_have_is_refused(tmp_path: Path):
    """`demo_task` stops at `secure`; grading it as an insertion would be inventing a link."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    with pytest.raises(checkpoint_backend.CheckpointError, match="no stage 6"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stage": 6}
        )


def test_a_blocker_outside_the_vocabulary_is_refused(tmp_path: Path):
    """Free-text reasons are what the batch already had, and they cannot be tallied."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    with pytest.raises(checkpoint_backend.CheckpointError, match="not one of"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path,
            {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stage": 2, "blocker": "vibes"},
        )


def test_a_shortfall_with_no_reason_given_records_that_it_was_unknown(tmp_path: Path):
    """Storing nothing would lose the fact that the question was asked at all."""
    _write_ladder(tmp_path, _DEMO_LADDER)

    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stage": 1}
    )

    assert entry["blocker"] == "unknown"


def test_a_ladder_using_a_stage_outside_the_shared_vocabulary_is_refused(tmp_path: Path):
    """The ordinals are only comparable across tasks while every task draws from one list."""
    _write_ladder(tmp_path, _DEMO_LADDER.replace("id: contact", "id: wiggle_it"))

    with pytest.raises(checkpoint_backend.CheckpointError, match="unknown stage"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stage": 1}
        )


def test_a_terminal_below_the_last_stage_is_refused(tmp_path: Path):
    """A stage past success is a stage the grade could never reward, so it is a mistake in the
    ladder rather than a preference."""
    _write_ladder(tmp_path, _DEMO_LADDER.replace("terminal: secure", "terminal: contact"))

    with pytest.raises(checkpoint_backend.CheckpointError, match="below its last stage"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path, {"checkpointId": "job_a/020000", "taskLadder": "demo_task", "stage": 1}
        )


def test_a_task_id_cannot_walk_out_of_the_ladder_directory(tmp_path: Path):
    with pytest.raises(checkpoint_backend.CheckpointError, match="bare name"):
        checkpoint_backend.append_rollout_outcome(
            tmp_path,
            {"checkpointId": "job_a/020000", "taskLadder": "../../etc/passwd", "stage": 1},
        )


def test_an_ungraded_rollout_is_stored_without_inventing_a_stage(tmp_path: Path):
    """Stage 0 is a real grade ("never reached the object"), so it must not stand in for
    "nobody said". Every record written before the ladder existed is in this shape."""
    entry = checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/020000", "outcome": "failure"}
    )

    assert "stage" not in entry
    assert "taskLadder" not in entry


def test_the_funnel_reports_where_rollouts_are_lost(tmp_path: Path):
    """The reason the grade is ordinal: the largest drop between neighbours names the next thing
    to work on. Modelled on the 08-31 batch, where two bottlenecks were the same size and the
    binary rate showed neither."""
    ladder = task_ladders.load_ladder(_write_ladder(tmp_path, _DEMO_LADDER))
    entries = (
        [{"taskLadder": "demo_task", "stage": 1, "outcome": "failure"}] * 3
        + [{"taskLadder": "demo_task", "stage": 2, "outcome": "failure"}] * 8
        + [{"taskLadder": "demo_task", "stage": 3, "outcome": "success"}] * 9
        # Excluded: stopped for reasons that say nothing about the policy.
        + [{"taskLadder": "demo_task", "stage": 1, "outcome": "aborted"}]
        # Excluded: graded the old way, so it has no stage to place.
        + [{"outcome": "failure"}]
    )

    funnel = task_ladders.stage_funnel(entries, ladder)

    assert [(row["stage"], row["reached"], row["lost"]) for row in funnel] == [
        (1, 20, 0),
        (2, 17, 3),
        (3, 9, 8),
    ]
    assert all(row["graded"] == 20 for row in funnel)


def test_the_grading_menu_carries_this_task_s_own_words(tmp_path: Path):
    """The operator grades by matching what they watched against a sentence, not by translating
    the vocabulary's abstract criterion. `describe` is what the page renders, so the task's own
    phrasing has to survive into it -- and so do the blocker labels, or the menu offers ids."""
    ladder = task_ladders.find_ladder(REPO_ROOT, "insert_peg")

    menu = ladder.describe()

    contact = next(stage for stage in menu["stages"] if stage["ordinal"] == 2)
    assert "推倒" in contact["instance"]
    assert contact["criterion"] == "建立接触，但未形成可承载的约束"
    assert menu["terminal"] == 7
    offset = next(item for item in menu["blockers"] if item["id"] == "object_pose_offset")
    assert offset["label"] == "抓取位姿偏移"


def test_a_blocker_the_ladder_did_not_declare_still_gets_a_name(tmp_path: Path):
    """`operator_stop` and `unknown` are merged into every ladder whether or not it lists them,
    so the menu must not fall back to showing a bare id for one nobody wrote a label for."""
    _write_ladder(tmp_path, _DEMO_LADDER.replace("  - {id: unknown}\n", ""))

    menu = task_ladders.find_ladder(tmp_path, "demo_task").describe()

    names = {item["id"]: item["label"] for item in menu["blockers"]}
    assert names["operator_stop"] == "operator_stop"
    assert set(names) >= {"object_pose_offset", "operator_stop", "unknown"}


def test_one_unreadable_ladder_does_not_take_the_others_down(tmp_path: Path):
    """The page fetches every ladder at once. If a half-edited file could raise, editing one
    task's ladder would stop the operator grading the task they are actually running."""
    _write_ladder(tmp_path, _DEMO_LADDER)
    (tmp_path / task_ladders.LADDER_DIR / "broken.yaml").write_text(
        "task: broken\nstages:\n  - {id: no_such_stage}\n", encoding="utf-8"
    )

    assert [ladder.task for ladder in task_ladders.list_ladders(tmp_path)] == ["demo_task"]
