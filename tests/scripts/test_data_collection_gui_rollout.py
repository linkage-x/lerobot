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
from tools.data_collection_gui import table_plane  # noqa: E402
from tools.data_collection_gui import task_ladders  # noqa: E402
from tools.data_collection_gui import training as training_backend  # noqa: E402
from tools.fr3.scene_reset import SceneResetError  # noqa: E402

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
    # The rehearsal is interactive -- it is steered, and it is where the takeover is practised --
    # but the arm it moves is simulated, so it must not ask for motion confirmation.
    assert by_id["dagger_sim"].movesArm is False
    assert by_id["dagger_sim"].interactive is True
    assert [mode.id for mode in rollout_backend.ROLLOUT_MODES if mode.interactive] == [
        "real",
        "dagger_sim",
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


def test_the_checkpoint_is_carried_to_the_next_rollout(tmp_path: Path):
    # Tuning a policy is the same checkpoint rolled out over and over, and finding its row again
    # in a list of forty was a click paid on every run.
    state = _rollout_state(tmp_path)

    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})
    gateway._stop_rollout(state)

    assert gateway._rollout_last_params(state)["checkpointId"] == "job_a/020000"


def test_the_takeover_switch_is_carried_to_the_next_rollout(tmp_path: Path):
    # The one setting that was deliberately forgotten before: a remembered "yes" opens a second
    # action source onto a loop that is driving a real arm. It is remembered now because an
    # afternoon of collecting corrections is otherwise the same switch re-ticked before every
    # single rollout, and what stands in for the protection that dropped is a sentence in the
    # page's carry-over notice -- plus the motion gate below, which is never carried.
    state = _rollout_state(tmp_path)

    gateway._start_rollout(
        state,
        {
            "mode": "real",
            "checkpointId": "job_a/020000",
            "confirmMotion": True,
            "runtimeOptions": {"daggerTakeover": True, "daggerRecord": True},
        },
    )
    gateway._stop_rollout(state)

    options = gateway._rollout_last_params(state)["runtimeOptions"]
    assert options["daggerTakeover"] is True
    assert options["daggerRecord"] is True


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
    # The checkpoint they were answered for does come back, and that is the point: a remembered
    # checkpoint must not arrive with the answers that were given for it last time.
    assert params["checkpointId"] == "job_a/020000"


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

    assert params == {"mode": "smoke", "checkpointId": "job_a/020000"}


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


SLOW_QUIT_LAUNCHER = """#!/usr/bin/env bash
echo "[INFO] interactive_rollouts=enabled keyboard_backend=pipe start_key='s'"
echo "[INFO] interactive_waiting_for_start arm_at_start=1 press 's' to start."
while IFS= read -r line; do
  case "$line" in
    quit)  sleep 1; echo "[INFO] interactive_rollouts=stopped"; exit 0 ;;
  esac
done
"""

DEAF_LAUNCHER = """#!/usr/bin/env bash
echo "[INFO] interactive_rollouts=enabled keyboard_backend=pipe start_key='s'"
echo "[INFO] interactive_waiting_for_start arm_at_start=1 press 's' to start."
while IFS= read -r line; do
  :
done
"""


def _relaunch_with(state: gateway.GatewayState, script: str) -> None:
    """Swap the fake launcher for one that shuts down differently. Call before starting."""
    launcher = state.repo_root / rollout_backend.LAUNCHER
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)


def test_the_drawn_mask_survives_a_page_reload(tmp_path: Path):
    """The region is measured against a camera still of the table, not typed from memory.

    Redrawn every load, it is redrawn slightly differently every load, and a rollout series'
    place points move without anyone deciding to move them.
    """
    state = _rollout_state(tmp_path)
    strokes = [{"x": 0.44, "y": -0.12, "radiusM": 0.035}, {"x": 0.46, "y": -0.10, "radiusM": 0.04}]

    saved = gateway._save_scene_reset_mask(state, {"strokes": strokes})

    assert saved["ok"] is True
    assert saved["updatedAt"]
    assert gateway._load_scene_reset_mask(state)["strokes"] == strokes


def test_a_cleared_mask_is_stored_as_cleared(tmp_path: Path):
    """"No region" is an answer an operator gave, not a save that failed.

    Treated as "nothing to store", the next load would hand back the region they had just
    deleted -- and the Clear button would be the one control on the page that does not last.
    """
    state = _rollout_state(tmp_path)
    gateway._save_scene_reset_mask(state, {"strokes": [{"x": 0.44, "y": -0.12, "radiusM": 0.035}]})

    gateway._save_scene_reset_mask(state, {"strokes": []})

    assert gateway._load_scene_reset_mask(state)["strokes"] == []


def test_a_mask_that_cannot_be_reset_with_is_refused_while_it_is_still_on_screen(tmp_path: Path):
    """Validated on the way in, with the same parser a reset request gets.

    A radius the sampler cannot use is worth refusing while the operator is looking at the
    canvas, rather than a fortnight later out of a file with the drawing long gone.
    """
    state = _rollout_state(tmp_path)

    with pytest.raises(SceneResetError):
        gateway._save_scene_reset_mask(state, {"strokes": [{"x": 0.44, "y": -0.12, "radiusM": 9.0}]})

    assert gateway._scene_reset_mask_path(state).exists() is False


def test_an_unreadable_mask_file_reads_as_no_region(tmp_path: Path):
    """A file this process cannot parse is a region the operator has to draw again, not a page
    that refuses to open."""
    state = _rollout_state(tmp_path)
    path = gateway._scene_reset_mask_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert gateway._load_scene_reset_mask(state) == {"strokes": [], "updatedAt": ""}


def test_a_mask_is_never_read_half_written(tmp_path: Path):
    """It is rewritten on every stroke, and the page that reads it is a different process."""
    state = _rollout_state(tmp_path)

    gateway._save_scene_reset_mask(state, {"strokes": [{"x": 0.44, "y": -0.12, "radiusM": 0.035}]})

    path = gateway._scene_reset_mask_path(state)
    leftovers = [item.name for item in path.parent.iterdir() if item.name.startswith(".")]
    assert leftovers == [], leftovers
    assert json.loads(path.read_text(encoding="utf-8"))["strokes"]


def test_a_stop_lets_the_runtime_close_its_own_dataset(tmp_path: Path):
    """The stop used to write `quit` and signal the process group in the same breath.

    The runtime's shutdown is what calls `dataset.finalize()` -- the episode metadata flush and
    the data parquet's footer -- and a SIGTERM that lands first kills it mid-write. One stop on
    2026-09-02 left a DAgger root of 432 recorded correction frames that nothing can open.

    The exit code is the assertion: a shell killed by SIGTERM reports -15, so this fails the
    moment the grace goes away again.
    """
    state = _rollout_state(tmp_path)
    _relaunch_with(state, SLOW_QUIT_LAUNCHER)
    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})
    process = state.rollout_process
    assert process is not None

    gateway._stop_rollout(state)

    assert _wait_for(lambda: process.poll() is not None), "the rollout never exited"
    assert process.returncode == 0


def test_a_rollout_that_never_answers_is_still_stopped(tmp_path: Path, monkeypatch):
    """The grace is bounded, because the thing being waited on is holding an arm.

    A runtime wedged in a driver is not going to read its stdin however long it is given, and
    the operator pressed stop.
    """
    state = _rollout_state(tmp_path)
    _relaunch_with(state, DEAF_LAUNCHER)
    monkeypatch.setattr(gateway, "ROLLOUT_QUIT_GRACE_S", 0.3)
    monkeypatch.setattr(gateway, "ROLLOUT_TERM_GRACE_S", 0.3)
    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})
    process = state.rollout_process
    assert process is not None

    gateway._stop_rollout(state)

    assert _wait_for(lambda: process.poll() is not None), "the rollout was never signalled"
    assert process.returncode != 0


def test_the_next_rollout_is_refused_while_the_last_one_is_closing(tmp_path: Path):
    """Refused with the reason, not with the generic "already running".

    The two states look the same to `_rollout_is_running` and mean opposite things to an
    operator: one is "you forgot to stop it", the other is "it is stopping, and cutting it short
    is what loses the corrections".
    """
    state = _rollout_state(tmp_path)
    _relaunch_with(state, SLOW_QUIT_LAUNCHER)
    gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})
    process = state.rollout_process
    assert process is not None
    gateway._stop_rollout(state)

    with pytest.raises(ValueError, match="still shutting down"):
        gateway._start_rollout(state, {"mode": "smoke", "checkpointId": "job_a/020000"})

    assert _wait_for(lambda: process.poll() is not None)


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


class _RecordingHandler:
    """Just enough of BaseHTTPRequestHandler for the JPEG writers."""

    def __init__(self) -> None:
        self.status = None
        self.headers: list[tuple[str, str]] = []
        self.body = bytearray()

    def send_response(self, status) -> None:
        self.status = status

    def send_header(self, key, value) -> None:
        self.headers.append((key, value))

    def end_headers(self) -> None:
        pass

    @property
    def wfile(self):
        handler = self

        class _Sink:
            def write(self, payload):
                handler.body.extend(payload)

        return _Sink()


class _RunningProcess:
    pid = 4321

    def poll(self):
        return None


def test_the_camera_still_serves_the_frame_the_live_view_refuses(tmp_path: Path, monkeypatch):
    """The reset map's reference layer is older than the live window by construction.

    Between rollouts the runtime publishes one frame and then blocks on the operator, so the
    file's age says how long they have been thinking, not whether the cameras are alive. The
    live endpoint must keep refusing it; the reset background must not, or the map is drawn on
    an empty grid for every rollout after the first.
    """
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    frame = preview_dir / "side.jpg"
    frame.write_bytes(b"\xff\xd8side")
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)
    parked = time.time() - (rollout_backend.PREVIEW_STALE_S + 30)
    os.utime(frame, (parked, parked))

    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", cameraKeys=["side", "ee"])
    state.rollout_process = _RunningProcess()

    live = _RecordingHandler()
    gateway._serve_rollout_camera_snapshot(live, state=state, camera_key="side")
    assert live.status == gateway.HTTPStatus.SERVICE_UNAVAILABLE

    still = _RecordingHandler()
    gateway._serve_rollout_camera_still(still, state=state, camera_key="side")
    assert still.status == gateway.HTTPStatus.OK
    assert bytes(still.body) == b"\xff\xd8side"


def test_the_camera_still_never_opens_a_camera_the_rollout_holds(tmp_path: Path, monkeypatch):
    """A RealSense is exclusive, so a second pipeline is a timeout, not a picture."""
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)

    def fail_if_preview_spawns(*_args, **_kwargs):
        raise AssertionError("the rollout owns the camera; nothing else may open it")

    monkeypatch.setattr(gateway, "_realsense_device_preview_frame", fail_if_preview_spawns)

    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", cameraKeys=["side"])
    state.rollout_process = _RunningProcess()
    state.devices = [{"id": "side", "kind": "camera", "config": {"type": "intelrealsense"}}]

    handler = _RecordingHandler()
    gateway._serve_rollout_camera_still(handler, state=state, camera_key="side")

    # No frame published yet, and the answer is to say so rather than to fight for the device.
    assert handler.status == gateway.HTTPStatus.SERVICE_UNAVAILABLE


def test_the_camera_still_opens_the_camera_when_no_rollout_is_running(tmp_path: Path, monkeypatch):
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)
    served: list[str] = []

    def record_device_preview(_state, *, device_id, device):  # noqa: ARG001
        served.append(device_id)
        return b"\xff\xd8device", gateway.HTTPStatus.OK, ""

    monkeypatch.setattr(gateway, "_realsense_device_preview_frame", record_device_preview)

    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="idle", cameraKeys=["side"])
    state.rollout_process = None
    state.devices = [{"id": "side", "kind": "camera", "config": {"type": "intelrealsense"}}]

    handler = _RecordingHandler()
    gateway._serve_rollout_camera_still(handler, state=state, camera_key="side")

    assert served == ["side"]
    assert bytes(handler.body) == b"\xff\xd8device"


def test_recording_an_outcome_clears_the_prompt(tmp_path: Path):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(
        checkpointId="job_a/020000", mode="real", step=214, pendingOutcomeFor=1
    )

    result = gateway._record_rollout_outcome(state, {"outcome": "success", "note": "clean pick"})

    assert result["entry"]["outcome"] == "success"
    assert result["entry"]["steps"] == 214
    assert state.rollout.pendingOutcomeFor == 0


def test_an_assisted_rollout_is_graded_as_assisted_whatever_the_page_says(tmp_path: Path):
    """The count comes off the runtime's trace, like the landing point.

    A page that could send its own would be able to file the rollout it drove by hand as one the
    policy did alone, which is the single way this log can lie about a checkpoint.
    """
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(
        checkpointId="job_a/020000",
        mode="real",
        step=782,
        pendingOutcomeFor=1,
        lastRolloutIntervention={"intervened": True, "expertSteps": 476},
    )

    result = gateway._record_rollout_outcome(
        state, {"outcome": "success", "intervened": False, "expertSteps": 0}
    )

    assert result["entry"]["intervened"] is True
    assert result["entry"]["expertSteps"] == 476


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


def test_the_landing_points_say_who_was_driving_when_they_happened():
    """The map draws whoever produced the point, and the grade needs the same fact about the
    terminal event. Rollout-level `intervened` cannot say *when*."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=3 status=stopped samples=430 closed=1 "
        "grasp_xyz=0.3162,-0.2214,0.0461 release_xyz=0.3599,-0.1330,0.0510 "
        "held_steps=223 grasp_by=policy release_by=expert intervened=1 expert_steps=88"
    )

    geometry = parsed["lastRolloutGeometry"]
    assert geometry["graspBy"] == "policy"
    assert geometry["releaseBy"] == "expert"


def test_a_landing_point_from_a_runtime_that_names_no_driver_stays_unattributed():
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=3 status=stopped samples=430 closed=1 "
        "grasp_xyz=0.3162,-0.2214,0.0461"
    )

    assert "graspBy" not in parsed["lastRolloutGeometry"]


def test_a_recorded_landing_point_keeps_who_produced_it(tmp_path: Path):
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "failure",
            "geometry": {
                "graspXyz": [0.31, -0.22, 0.046],
                "graspBy": "policy",
                "releaseBy": "expert",
            },
        },
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert entry["geometry"]["graspBy"] == "policy"
    assert entry["geometry"]["releaseBy"] == "expert"


def test_a_landing_point_cannot_be_attributed_to_something_that_is_not_a_driver(tmp_path: Path):
    """Two words, because there are two things that can drive the arm. Anything else is a caller
    inventing a third."""
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "failure",
            "geometry": {"graspXyz": [0.31, -0.22, 0.046], "graspBy": "the intern"},
        },
    )

    assert "graspBy" not in checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]["geometry"]


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


# ------------------------------------------------------------ who was driving ---
#
# A rollout an operator finished by hand measures the operator. These keep that fact attached to
# the grade, because the outcome log is what two checkpoints are compared on.


def test_the_end_marker_says_the_operator_drove_and_for_how_long():
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=1 status=stopped samples=782 closed=1 "
        "grasp_xyz=0.3881,-0.2503,0.0585 held_steps=239 "
        "intervened=1 expert_steps=476 expert_spans=135-610"
    )

    assert parsed["lastRolloutIntervention"] == {"intervened": True, "expertSteps": 476}


def test_a_rollout_nobody_touched_says_so_rather_than_saying_nothing():
    """False here is a measurement -- the runtime reported a summary and it had no expert spans
    -- and it is what lets the log tell an unassisted rollout from an unexamined one."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=2 status=completed samples=430 closed=1 "
        "grasp_xyz=0.3162,-0.2214,0.0461 held_steps=223"
    )

    assert parsed["lastRolloutIntervention"] == {"intervened": False, "expertSteps": 0}


def test_an_end_marker_without_a_summary_does_not_claim_the_policy_drove():
    """A runtime older than the field counted nobody, which is not the same as counting zero."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] interactive_rollout_end index=1 status=stopped"
    )

    assert parsed["lastRolloutIntervention"] == {}


def test_starting_a_rollout_clears_the_previous_takeover():
    parsed = rollout_backend.parse_rollout_line("[INFO] interactive_rollout_start index=5")

    assert parsed["lastRolloutIntervention"] == {}


def test_a_recorded_outcome_keeps_who_was_driving(tmp_path: Path):
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "success",
            "intervention": {"intervened": True, "expertSteps": 476},
        },
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert entry["intervened"] is True
    assert entry["expertSteps"] == 476


def test_an_unassisted_rollout_is_recorded_as_measured_not_as_unknown(tmp_path: Path):
    checkpoint_backend.append_rollout_outcome(
        tmp_path,
        {
            "checkpointId": "job_a/030000",
            "outcome": "success",
            "intervention": {"intervened": False, "expertSteps": 0},
        },
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert entry["intervened"] is False
    assert entry["expertSteps"] == 0


def test_an_outcome_nobody_counted_records_no_intervention_field(tmp_path: Path):
    """Absent, not false: `false` would put a rollout that may have been driven by hand into the
    same bucket as the ones the policy did alone."""
    checkpoint_backend.append_rollout_outcome(
        tmp_path, {"checkpointId": "job_a/030000", "outcome": "success", "intervention": {}}
    )

    entry = checkpoint_backend.load_rollout_outcomes(tmp_path)[-1]
    assert "intervened" not in entry
    assert "expertSteps" not in entry


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
    # The pick pose a scene reset needs is the release point *with its height*: the peg was let
    # go there, so that is the height a gripper has to return to in order to take it back. The
    # map's `hole` is the same measurement with the z dropped, and derived from it rather than
    # separately, so the two can never disagree about where the demonstrations put the peg.
    assert landmarks["placeXyz"] == pytest.approx(landmarks["points"][0]["releaseXyz"])
    assert landmarks["hole"] == landmarks["placeXyz"][:2]


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


class _StdinProcess:
    """A running rollout whose stdin can be read back, so control lines can be asserted."""

    pid = 5150

    def __init__(self) -> None:
        self.written = bytearray()

        class _Stdin:
            def __init__(self, sink):
                self._sink = sink

            def write(self, payload):
                self._sink.extend(payload)

            def flush(self):
                pass

        self.stdin = _Stdin(self.written)

    def poll(self):
        return None

    def lines(self) -> list[str]:
        return bytes(self.written).decode().splitlines()


def _probe_ready(tmp_path: Path, monkeypatch, *, request_id: str, xyz: list[float], camera: str = "side"):
    """Stand in for the runtime: a still and the sidecar it writes beside it at the point."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(exist_ok=True)
    (probe_dir / f"{camera}.jpg").write_bytes(b"\xff\xd8probe")
    (probe_dir / "probe.json").write_text(
        json.dumps({"requestId": request_id, "xyz": xyz, "cameras": [camera], "at": time.time()})
    )
    monkeypatch.setattr(rollout_backend, "PROBE_DIR", probe_dir)
    monkeypatch.setattr(rollout_backend, "PROBE_SIDECAR_PATH", probe_dir / "probe.json")
    return probe_dir


def test_a_table_probe_sends_the_arm_to_the_point_and_remembers_which_one(tmp_path: Path):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", interactive=True, cameraKeys=["side"])
    process = _StdinProcess()
    state.rollout_process = process

    result = gateway._request_table_probe(state, {"camera": "side", "x": 0.45, "y": 0.05, "z": 0.035})

    assert result["ok"] is True
    line = process.lines()[0]
    assert line.startswith("probe_pose ")
    assert json.loads(line.removeprefix("probe_pose "))["xyz"] == [0.45, 0.05, 0.035]
    # The pending request id is the other end of the match with the still the runtime writes.
    assert state.table_probe["requestId"] == json.loads(line.removeprefix("probe_pose "))["requestId"]
    # The arm is moving, so the page must stop saying the cell is safe to reach into.
    assert state.rollout.state == "resetting"
    assert state.rollout.armAtStart is False


def test_a_table_probe_is_refused_while_a_rollout_is_running(tmp_path: Path):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="rolling", interactive=True, cameraKeys=["side"])
    state.rollout_process = _StdinProcess()

    with pytest.raises(ValueError, match="waiting"):
        gateway._request_table_probe(state, {"camera": "side", "x": 0.45, "y": 0.0, "z": 0.035})


def test_a_table_point_is_labelled_with_where_the_arm_got_to_not_where_it_was_sent(
    tmp_path: Path, monkeypatch
):
    """The browser knows what it asked for; only the runtime knows what happened.

    A probe that was refused, that timed out, or that stopped short would otherwise be recorded
    as if the tool had arrived, and a calibration is exactly the artifact in which that kind of
    error is invisible afterwards.
    """
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", interactive=True, cameraKeys=["side"])
    state.rollout_process = _StdinProcess()
    _probe_ready(tmp_path, monkeypatch, request_id="probe-1", xyz=[0.461, 0.052, 0.035])
    state.table_probe = {"cameraKey": "side", "requestId": "probe-1", "xyz": [0.45, 0.05, 0.035]}

    result = gateway._record_table_point(
        state, {"camera": "side", "u": 210.0, "v": 305.0, "imageWidth": 640, "imageHeight": 480}
    )

    assert result["points"] == [{"u": 210.0, "v": 305.0, "x": 0.461, "y": 0.052}]
    assert result["planeZ"] == pytest.approx(0.035)
    assert result["calibrated"] is False  # one point of the four a plane needs


def test_a_table_point_is_refused_while_the_arm_is_still_on_its_way(tmp_path: Path, monkeypatch):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="resetting", interactive=True, cameraKeys=["side"])
    _probe_ready(tmp_path, monkeypatch, request_id="probe-1", xyz=[0.45, 0.05, 0.035])
    # The still on disk belongs to the previous probe, not the one just asked for.
    state.table_probe = {"cameraKey": "side", "requestId": "probe-2", "xyz": [0.50, 0.05, 0.035]}

    with pytest.raises(ValueError, match="not finished"):
        gateway._record_table_point(
            state, {"camera": "side", "u": 210.0, "v": 305.0, "imageWidth": 640, "imageHeight": 480}
        )

    handler = _RecordingHandler()
    gateway._serve_table_probe_frame(handler, state=state, camera_key="side")
    assert handler.status == gateway.HTTPStatus.SERVICE_UNAVAILABLE


def test_clicking_the_same_probed_point_twice_corrects_it_rather_than_duplicating_it(
    tmp_path: Path, monkeypatch
):
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", interactive=True, cameraKeys=["side"])
    _probe_ready(tmp_path, monkeypatch, request_id="probe-1", xyz=[0.45, 0.05, 0.035])
    state.table_probe = {"cameraKey": "side", "requestId": "probe-1", "xyz": [0.45, 0.05, 0.035]}
    click = {"camera": "side", "imageWidth": 640, "imageHeight": 480}

    gateway._record_table_point(state, {**click, "u": 200.0, "v": 300.0})
    result = gateway._record_table_point(state, {**click, "u": 214.0, "v": 297.0})

    # Two points at one coordinate add no constraint and drag the fit toward the worse click.
    assert result["points"] == [{"u": 214.0, "v": 297.0, "x": 0.45, "y": 0.05}]


def _calibrate_side_camera(state, tmp_path: Path) -> None:
    """A stored calibration for the 'side' camera, as four probes would have left one."""
    calibration = table_plane.TablePlaneCalibration(
        cameraKey="side", planeZ=0.035, imageWidth=640, imageHeight=480
    )
    for (u, v), (x, y) in zip(
        [(180.0, 300.0), (460.0, 300.0), (420.0, 190.0), (220.0, 190.0)],
        [(0.37, 0.08), (0.37, -0.08), (0.53, -0.08), (0.53, 0.08)],
        strict=True,
    ):
        calibration.add_point(table_plane.TablePlanePoint(u=u, v=v, x=x, y=y), plane_z=0.035)
    table_plane.save_calibration(gateway._table_plane_path(state, "side"), calibration)


def test_the_table_view_refuses_to_draw_anything_before_the_camera_is_aligned(
    tmp_path: Path, monkeypatch
):
    """No backdrop is the correct drawing of "we do not know where this picture is".

    The bug this whole path replaces was a still stretched to fill the plot box: it looked like
    a reference and lined up with nothing, so a target region painted against it was aimed at
    the wrong part of the table.
    """
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    (preview_dir / "side.jpg").write_bytes(b"\xff\xd8side")
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)
    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", cameraKeys=["side"])
    state.rollout_process = _RunningProcess()

    handler = _RecordingHandler()
    gateway._serve_table_view(
        handler,
        state=state,
        query={
            "camera": ["side"],
            "minX": ["0.30"],
            "maxX": ["0.60"],
            "minY": ["-0.15"],
            "maxY": ["0.15"],
            "width": ["300"],
            "height": ["300"],
        },
    )

    assert handler.status == gateway.HTTPStatus.CONFLICT


def test_the_table_view_serves_the_window_the_caller_asked_for(tmp_path: Path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), 200, dtype=np.uint8))
    assert ok
    (preview_dir / "side.jpg").write_bytes(buffer.tobytes())
    monkeypatch.setattr(rollout_backend, "PREVIEW_DIR", preview_dir)

    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", cameraKeys=["side"])
    state.rollout_process = _RunningProcess()
    _calibrate_side_camera(state, tmp_path)

    handler = _RecordingHandler()
    gateway._serve_table_view(
        handler,
        state=state,
        query={
            "camera": ["side"],
            "minX": ["0.30"],
            "maxX": ["0.60"],
            "minY": ["-0.15"],
            "maxY": ["0.15"],
            "width": ["300"],
            "height": ["220"],
        },
    )

    assert handler.status == gateway.HTTPStatus.OK
    assert ("Content-Type", "image/jpeg") in handler.headers
    served = cv2.imdecode(np.frombuffer(bytes(handler.body), dtype=np.uint8), cv2.IMREAD_COLOR)
    # Exactly the pixels asked for: the plot places this image over its own axes and any other
    # size would be stretched back into place, which is the failure the window exists to avoid.
    assert served.shape[:2] == (220, 300)


def test_clearing_an_alignment_leaves_the_maps_with_no_backdrop(tmp_path: Path):
    state = _rollout_state(tmp_path)
    _calibrate_side_camera(state, tmp_path)
    assert gateway._load_table_plane(state, "side").calibrated

    result = gateway._clear_table_points(state, {"camera": "side"})

    assert result["calibrated"] is False
    assert result["points"] == []
    assert gateway._load_table_plane(state, "side").calibrated is False


class _FakeArm:
    """Enough of the FR3 for the probe's step loop: it goes where it is told, immediately."""

    def __init__(self) -> None:
        self.xyz = (0.30, 0.0, 0.20)
        self.gripper = 1.0

    def get_observation(self, *, include_cameras=False):  # noqa: ARG002
        return {
            "ee.x": self.xyz[0],
            "ee.y": self.xyz[1],
            "ee.z": self.xyz[2],
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "gripper.pos": self.gripper,
        }

    def send_action(self, action):
        self.xyz = (action["ee.x"], action["ee.y"], action["ee.z"])
        self.gripper = action["gripper.pos"]
        return dict(action)


def test_the_probe_line_the_gateway_writes_is_the_one_the_runtime_acts_on(tmp_path: Path, monkeypatch):
    """The gateway, the control channel and the arm, joined at the seams they meet on.

    Three processes have to agree on this one line: the gateway composes it, the runtime's pipe
    channel resolves the word and the payload, and the probe executes it. Each half is easy to
    change without the others noticing, and the failure is silent -- a calibration built from
    points the arm never visited looks exactly like one built from points it did.
    """
    import io as _io
    import sys as _sys

    from tools.fr3.interactive_control import InteractiveRolloutKeyboard
    from tools.fr3.scene_reset import execute_pose_probe, pose_probe_request_from_payload

    state = _rollout_state(tmp_path)
    state.rollout = rollout_backend.RolloutStatus(state="waiting", interactive=True, cameraKeys=["side"])
    process = _StdinProcess()
    state.rollout_process = process
    gateway._request_table_probe(state, {"camera": "side", "x": 0.45, "y": 0.05, "z": 0.035})
    line = process.lines()[0]

    keyboard = InteractiveRolloutKeyboard(start_key="s", stop_key="x", home_key="h", quit_key="q")
    monkeypatch.setattr(_sys, "stdin", _io.StringIO(line + "\n"))
    keyboard._listen_pipe_loop()
    assert keyboard.probe_pose_requested.is_set()

    request = pose_probe_request_from_payload(keyboard.pop_probe_pose_payload())
    arm = _FakeArm()
    reached: list[tuple[float, float, float]] = []
    result = execute_pose_probe(arm, request, on_arrival=lambda: reached.append(arm.xyz))
    assert result["ok"] is True

    # What the runtime would write beside the still, from where the arm actually ended up.
    _probe_ready(tmp_path, monkeypatch, request_id=request.requestId, xyz=list(reached[0]))
    recorded = gateway._record_table_point(
        state, {"camera": "side", "u": 300.0, "v": 240.0, "imageWidth": 640, "imageHeight": 480}
    )

    assert recorded["points"] == [{"u": 300.0, "v": 240.0, "x": 0.45, "y": 0.05}]
    assert recorded["planeZ"] == pytest.approx(0.035)


def test_each_launch_gets_its_own_trace_directory(tmp_path: Path):
    """The runtime numbers traces from 1 on every start and overwrites what is there.

    A browser operator has nowhere to type a directory, so the gateway derives one per launch.
    Sharing one would mean each session silently destroying the last -- which is what happened on
    2026-09-01, taking four traces of the graded 08-31 batch with them.
    """
    trace_dir = tmp_path / "outputs" / "rollout_traces" / "session_20260901_163400"
    command, _ = _command(tmp_path, trace_dir=trace_dir)

    assert "--rollout-trace-dir" in command
    assert command[command.index("--rollout-trace-dir") + 1] == str(trace_dir)
    assert command.index("--rollout-trace-dir") > command.index("real")


def test_two_launches_a_second_apart_do_not_share_a_trace_directory(tmp_path: Path):
    assert rollout_backend.trace_session_dir(tmp_path, "20260901_163400") != (
        rollout_backend.trace_session_dir(tmp_path, "20260901_163401")
    )


def test_a_settings_dump_gets_no_trace_directory(tmp_path: Path):
    # `env` prints and exits without a runtime, so a directory would only be named and never used.
    command, _ = _command(tmp_path, mode="env", trace_dir=tmp_path / "traces")

    assert "--rollout-trace-dir" not in command


# --------------------------------------------------------- DAgger takeover from the browser ---
#
# The takeover already worked from a terminal. What these cover is the two things that change
# when a browser launches it: the operator cannot type a dataset path per batch, and they cannot
# press `t` -- the gateway holds the runtime's stdin as a pipe, so the keyboard backend is not
# reading a keyboard at all.


def test_takeover_is_refused_on_a_mode_the_launcher_would_silently_drop_it_from(tmp_path: Path):
    """`real_once` accepts the environment variable and never passes the flag on.

    The runtime would refuse `--dagger-takeover` without `--interactive-rollouts`, but the flag
    never reaches it: the launcher's `case` puts `dagger_args` in two branches only. So the
    failure is silent -- a rollout the operator believes they can grab the arm out of, which runs
    to its step limit with the SpaceMouse doing nothing.
    """
    with pytest.raises(rollout_backend.RolloutError, match="interactive"):
        _command(
            tmp_path,
            mode="real_once",
            runtime_options=rollout_backend.sanitize_rollout_runtime_options(
                {"daggerTakeover": True}
            ),
            dagger_dataset_fallback=tmp_path / "outputs" / "dagger" / "job_a",
        )


def test_takeover_gets_a_destination_the_operator_never_typed(tmp_path: Path):
    """A blank field must not mean "throw the corrections away".

    This is the trace-directory lesson applied one layer up: the browser has nowhere to type a
    path per session, and the answer that loses data is not the one to default to.
    """
    fallback = tmp_path / "outputs" / "dagger" / "job_a_020000"
    _, env = _command(
        tmp_path,
        runtime_options=rollout_backend.sanitize_rollout_runtime_options({"daggerTakeover": True}),
        dagger_dataset_fallback=fallback,
    )

    assert env["FR3_DAGGER_TAKEOVER"] == "1"
    assert env["FR3_DAGGER_DATASET_ROOT"] == str(fallback)


def test_a_named_corrections_dataset_beats_the_derived_one(tmp_path: Path):
    _, env = _command(
        tmp_path,
        runtime_options=rollout_backend.sanitize_rollout_runtime_options(
            {"daggerTakeover": True, "daggerDatasetRoot": "outputs/dagger/insert_s6"}
        ),
        dagger_dataset_fallback=tmp_path / "outputs" / "dagger" / "derived",
    )

    assert env["FR3_DAGGER_DATASET_ROOT"] == "outputs/dagger/insert_s6"


def test_steering_without_recording_is_something_the_operator_has_to_say(tmp_path: Path):
    """Not the same as leaving the field blank, and the launcher can tell them apart.

    An empty value fails the launcher's `-n` test exactly as an absent one does, so the takeover
    runs and writes nothing -- which is the right behaviour for feeling out the handoff on a real
    arm, and the wrong one to arrive at by forgetting to fill something in.
    """
    _, env = _command(
        tmp_path,
        runtime_options=rollout_backend.sanitize_rollout_runtime_options(
            {"daggerTakeover": True, "daggerRecord": False}
        ),
        dagger_dataset_fallback=tmp_path / "outputs" / "dagger" / "derived",
    )

    assert env["FR3_DAGGER_TAKEOVER"] == "1"
    assert env["FR3_DAGGER_DATASET_ROOT"] == ""


def test_a_rollout_that_asked_for_no_takeover_opens_no_device(tmp_path: Path):
    _, env = _command(tmp_path, dagger_dataset_fallback=tmp_path / "outputs" / "dagger" / "d")

    assert "FR3_DAGGER_TAKEOVER" not in env
    assert "FR3_DAGGER_DATASET_ROOT" not in env


def test_a_takeover_left_in_the_gateways_own_environment_cannot_arm_a_rollout(tmp_path: Path):
    """The one that would be worst to inherit: a second action source onto a moving arm.

    A shell that once exported this to run a takeover by hand would otherwise hand every browser
    rollout a live SpaceMouse, with nothing on the page saying so.
    """
    _, env = _command(
        tmp_path,
        base_env={"FR3_DAGGER_TAKEOVER": "1", "FR3_DAGGER_DATASET_ROOT": "/somewhere/stale"},
        dagger_dataset_fallback=tmp_path / "outputs" / "dagger" / "d",
    )

    assert "FR3_DAGGER_TAKEOVER" not in env
    assert "FR3_DAGGER_DATASET_ROOT" not in env


def test_corrections_against_one_checkpoint_accumulate_in_one_dataset(tmp_path: Path):
    """Per checkpoint, not per launch -- the opposite of the trace directory, deliberately.

    A trace is only useful separated by batch. A DAgger dataset is only useful once it is big
    enough to train on, and the states it corrects are the states this policy walks into.
    """
    first = rollout_backend.dagger_dataset_dir(tmp_path, "L4_full48_holdout22_40/030000")
    second = rollout_backend.dagger_dataset_dir(tmp_path, "L4_full48_holdout22_40/030000")
    other = rollout_backend.dagger_dataset_dir(tmp_path, "L4_full48_holdout22_40/040000")

    assert first == second
    assert first != other
    # Flattened the same way the log file's name is, so the two sit side by side under names an
    # operator can pair up by eye.
    assert "/" not in first.name


def test_turning_automatic_handback_off_survives_being_zero(tmp_path: Path):
    """0 is an instruction, not a missing field: it leaves Hold as the only way in and out."""
    options = rollout_backend.sanitize_rollout_runtime_options(
        {"daggerTakeover": True, "daggerReleaseAfterS": 0}
    )

    assert options["FR3_DAGGER_RELEASE_AFTER_S"] == "0"


def test_a_takeover_switch_that_is_neither_yes_nor_no_is_refused():
    # Refused rather than read as off, because off is the answer that runs.
    with pytest.raises(rollout_backend.RolloutError, match="daggerTakeover"):
        rollout_backend.sanitize_rollout_runtime_options({"daggerTakeover": "maybe"})


def test_a_corrections_path_cannot_carry_a_second_line():
    with pytest.raises(rollout_backend.RolloutError, match="line breaks"):
        rollout_backend.sanitize_rollout_runtime_options(
            {"daggerTakeover": True, "daggerDatasetRoot": "outputs/dagger/a\nb"}
        )


def test_the_page_learns_the_device_is_armed_from_the_preflight_banner():
    """The banner is what the operator reads before touching anything, so the page reads it too.

    `report_timestamps` is the check that the driver dates its reports -- without dates the arm
    keeps flying after the hand comes off and the handback timer never expires. A real rollout
    cannot start without it, which is exactly why the field is worth showing: "it says yes" is a
    check an operator can make, and an absent field is not.
    """
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] dagger_takeover=ready device_id=0 translation_scale=0.000615 "
        "rotation_scale=0.000648 release_after_s=1.00 motion_gain=6.67 nominal_step_ms=33.3 "
        "full_deflection_mm_per_step=4.1 report_timestamps=yes"
    )

    assert parsed["takeoverAvailable"] is True
    assert parsed["daggerReportTimestamps"] == "yes"
    assert parsed["daggerReleaseAfterS"] == pytest.approx(1.0)


def test_the_page_shows_where_the_runtime_actually_put_the_corrections():
    """The runtime's own answer, which is the one that can differ from what was asked for."""
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] dagger_dataset=extending root=outputs/dagger/job_a repo_id=job_a episodes=12"
    )

    assert parsed["daggerDatasetPath"] == "outputs/dagger/job_a"


def test_episodes_written_are_reported_under_a_name_that_is_not_the_running_total():
    """One rollout's count must not be assignable to the session's, or the total would reset.

    The gateway sums these; the field they add into is `daggerEpisodes`. Named apart so the
    generic setattr loop that applies every other parsed key cannot touch the total.
    """
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] dagger_dataset_written rollout=3 episodes=2 frames=71 skipped_spans=1 "
        "dropped_frames=0"
    )

    assert parsed["daggerEpisodesWritten"] == 2
    assert parsed["daggerDroppedFramesWritten"] == 0
    assert "daggerEpisodes" not in parsed


def test_a_truncated_correction_is_counted_so_the_page_can_say_so():
    parsed = rollout_backend.parse_rollout_line(
        "[INFO] dagger_dataset_written rollout=4 episodes=1 frames=450 skipped_spans=0 "
        "dropped_frames=118"
    )

    assert parsed["daggerDroppedFramesWritten"] == 118


def test_only_the_two_interactive_real_modes_offer_takeover():
    """The page greys the switch out from this, so it has to match the launcher's `case`."""
    offering = {mode.id for mode in rollout_backend.ROLLOUT_MODES if mode.takeover}

    assert offering == {"real", "real_debug"}


def test_an_unread_handback_time_is_not_the_same_as_one_that_is_off():
    """0 says the arm will never hand itself back; the page has to be able to say that only when
    the runtime did. A float default of 0.0 would have every rollout claim it for the first few
    lines, before the banner arrives."""
    assert rollout_backend.RolloutStatus().daggerReleaseAfterS is None


def test_hold_is_refused_on_a_rollout_that_opened_no_device(tmp_path: Path):
    """The runtime would answer this in the log and nowhere else.

    `interactive_pipe_takeover_ignored` is one line among thousands, and the operator pressing
    Hold is reaching for a brake. They have to be told on screen that this rollout has none,
    rather than infer it from an arm that kept going.
    """
    state = _rollout_state(tmp_path)
    gateway._start_rollout(state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True})
    try:
        state.rollout.state = "rolling"
        state.rollout.takeoverAvailable = False

        with pytest.raises(ValueError, match="no device"):
            gateway._send_rollout_control(state, "takeover")
    finally:
        gateway._stop_rollout(state)


def test_hold_reaches_the_runtime_once_a_device_is_armed(tmp_path: Path):
    state = _rollout_state(tmp_path)
    gateway._start_rollout(state, {"mode": "real", "checkpointId": "job_a/020000", "confirmMotion": True})
    try:
        state.rollout.state = "rolling"
        state.rollout.takeoverAvailable = True

        result = gateway._send_rollout_control(state, "takeover")

        assert result["ok"] is True
        # The page must not claim a direction: the latch is a toggle the runtime owns, and a page
        # that guessed would sooner or later disagree with the arm about who is driving.
        assert "toggled" in state.rollout.message
    finally:
        gateway._stop_rollout(state)


def test_corrections_land_where_the_export_page_can_find_them(tmp_path: Path):
    """One level under the datasets root, which is exactly as deep as the gateway scans.

    The export page is what merges corrections with the demonstrations into the view the next
    checkpoint trains on. A DAgger dataset the picker cannot list is one nobody can train on
    without dropping to the CLI, which makes the whole browser path stop one step short of the
    thing it exists to produce.
    """
    root = rollout_backend.dagger_dataset_dir(tmp_path, "L4_full48_holdout22_40/030000")

    assert root.parent == tmp_path / "outputs" / "datasets"
    assert root.name.startswith(rollout_backend.DAGGER_PREFIX)
