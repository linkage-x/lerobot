"""Contract tests for the gateway-driven FR3 recorder and its timestamp-sync audit.

The properties under test are the ones that silently rot: sim and hardware recordings must
keep producing the same dataset schema, the audit must not mislabel which clock produced a
timestamp, and the gateway must parse the recorder's SYNC protocol into a verdict the operator
can act on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tools.data_collection_gui import gateway
from tools.fr3 import fr3_sync_audit
from tools.fr3.fr3_gui_record_runtime import build_sim_robot_config


# ------------------------------------------------------------------ sim/real schema parity ---


def _record_config(tmp_path: Path, camera_size: tuple[int, int] = (320, 240)):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.franka_research3 import FrankaResearch3Config
    from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
    from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig

    width, height = camera_size
    urdf = (
        Path(__file__).resolve().parents[2]
        / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"
    )
    return RecordConfig(
        robot=FrankaResearch3Config(
            gripper_backend="mock",
            urdf_path=str(urdf),
            target_frame_name="pika_task_tcp",
            max_target_delta_pos=(0.001, 0.001, 0.001),
            max_target_delta_rot=(0.01, 0.01, 0.01),
            cameras={
                "ee": OpenCVCameraConfig(index_or_path=0, width=width, height=height, fps=30),
                "side": OpenCVCameraConfig(index_or_path=1, width=width, height=height, fps=30),
            },
        ),
        dataset=DatasetRecordConfig(
            repo_id="test/fr3_gui",
            single_task="test",
            root=str(tmp_path / "ds"),
            fps=10,
            episode_time_s=1,
            num_episodes=1,
            push_to_hub=False,
        ),
        teleop=SpaceMouseTeleopConfig(),
        control_fps=30,
        play_sounds=False,
    )


def test_sim_robot_config_mirrors_hardware_camera_keys_and_envelope(tmp_path):
    cfg = _record_config(tmp_path)
    sim_cfg = build_sim_robot_config(cfg)

    # Camera keys are what end up in the dataset feature names; they must not be renamed.
    assert sim_cfg.camera_names == ("ee", "side")
    assert sim_cfg.camera_name_mapping == {"ee": "ee_cam", "side": "external_cam"}
    assert sim_cfg.camera_width == 320
    assert sim_cfg.camera_height == 240
    # The safety envelope the ee2ee processors clamp against must be identical.
    assert tuple(sim_cfg.workspace_min) == tuple(cfg.robot.workspace_min)
    assert tuple(sim_cfg.workspace_max) == tuple(cfg.robot.workspace_max)
    assert sim_cfg.max_target_delta_pos == cfg.robot.max_target_delta_pos
    assert sim_cfg.max_target_delta_rot == cfg.robot.max_target_delta_rot
    assert sim_cfg.target_frame_name == cfg.robot.target_frame_name


def test_sim_camera_map_override_is_respected(tmp_path):
    cfg = _record_config(tmp_path)
    sim_cfg = build_sim_robot_config(cfg, camera_map_override={"ee": "external_cam"})
    assert sim_cfg.camera_name_mapping["ee"] == "external_cam"
    # Unlisted cameras keep the name-based default rather than silently dropping out.
    assert sim_cfg.camera_name_mapping["side"] == "external_cam"


def test_sim_backend_rejects_mixed_camera_resolutions(tmp_path):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = _record_config(tmp_path)
    cfg.robot.cameras["side"] = OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
    with pytest.raises(ValueError, match="share one resolution"):
        build_sim_robot_config(cfg)


def test_sim_and_hardware_robots_declare_the_same_observation_schema(tmp_path):
    """A sim dataset must be interchangeable with a hardware one, feature for feature."""
    from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Mujoco

    cfg = _record_config(tmp_path)
    hardware = FrankaResearch3(cfg.robot)
    simulated = FrankaResearch3Mujoco(build_sim_robot_config(cfg))

    assert list(simulated.observation_features) == list(hardware.observation_features)
    assert simulated.action_features == hardware.action_features
    # Capture-timestamp columns line up positionally; only the gripper backend name differs,
    # which is exactly the signal the audit uses to tell the two clocks apart.
    hardware_names = hardware.capture_timestamp_feature_names
    sim_names = simulated.capture_timestamp_feature_names
    assert len(sim_names) == len(hardware_names)
    assert sim_names[0] == hardware_names[0] == "fr3.arm.capture_timestamp_s"
    assert sim_names[1] == "sim_gripper.capture_timestamp_s"
    assert sim_names[2:] == hardware_names[2:]


# ------------------------------------------------------------------------- the sync audit ---


def _capture_timestamps(frames: int, *, skew_s: float, interval_s: float) -> np.ndarray:
    base = np.arange(frames, dtype=np.float64) * interval_s
    return np.stack([base, base + skew_s * 0.5, base + skew_s], axis=1)


DEVICE_NAMES = [
    "fr3.arm.capture_timestamp_s",
    "sim_gripper.capture_timestamp_s",
    "camera.ee.capture_timestamp_s",
]


def test_episode_summary_passes_for_tight_alignment():
    frames = 20
    summary = fr3_sync_audit.summarize_episode_capture_timestamps(
        capture_timestamps=_capture_timestamps(frames, skew_s=0.002, interval_s=0.1),
        frame_timestamps=np.arange(frames, dtype=np.float64) * 0.1,
        device_names=DEVICE_NAMES,
        clock_semantics="sim_extraction_wallclock",
    )
    assert summary["status"] == "pass"
    assert summary["skew_over_tolerance_frames"] == 0
    assert summary["max_skew_ms"] == pytest.approx(2.0, abs=1e-6)
    assert summary["measured_frame_interval_ms"] == pytest.approx(100.0, abs=1e-6)


def test_episode_summary_flags_intra_frame_skew():
    frames = 20
    summary = fr3_sync_audit.summarize_episode_capture_timestamps(
        capture_timestamps=_capture_timestamps(frames, skew_s=0.05, interval_s=0.1),
        frame_timestamps=np.arange(frames, dtype=np.float64) * 0.1,
        device_names=DEVICE_NAMES,
        clock_semantics="hardware_mixed",
        tolerance_ms=20.0,
    )
    assert summary["status"] == "fail"
    assert summary["skew_over_tolerance_frames"] == frames


def test_episode_summary_flags_a_control_loop_that_cannot_hold_cadence():
    """The dataset labels frames 1/fps apart; a slower real cadence must not pass silently."""
    frames = 20
    summary = fr3_sync_audit.summarize_episode_capture_timestamps(
        # Captured every 130 ms while the dataset grid claims 100 ms.
        capture_timestamps=_capture_timestamps(frames, skew_s=0.001, interval_s=0.13),
        frame_timestamps=np.arange(frames, dtype=np.float64) * 0.1,
        device_names=DEVICE_NAMES,
        clock_semantics="sim_extraction_wallclock",
    )
    assert summary["status"] == "fail"
    assert summary["skew_over_tolerance_frames"] == 0  # modalities agree; cadence does not
    assert summary["global_lag_over_tolerance_frames"] > 0
    assert summary["measured_frame_interval_ms"] == pytest.approx(130.0, abs=1e-6)
    assert summary["nominal_frame_interval_ms"] == pytest.approx(100.0, abs=1e-6)


def test_episode_summary_reports_bias_relative_to_the_arm_read():
    frames = 10
    summary = fr3_sync_audit.summarize_episode_capture_timestamps(
        capture_timestamps=_capture_timestamps(frames, skew_s=0.004, interval_s=0.1),
        frame_timestamps=np.arange(frames, dtype=np.float64) * 0.1,
        device_names=DEVICE_NAMES,
        clock_semantics="hardware_mixed",
    )
    bias = summary["cross_modality_bias_ms"]
    assert bias["fr3.arm.capture_timestamp_s"] == pytest.approx(0.0, abs=1e-6)
    assert bias["camera.ee.capture_timestamp_s"] == pytest.approx(4.0, abs=1e-6)


def test_episode_summary_rejects_mismatched_device_name_count():
    with pytest.raises(ValueError, match="device names for width"):
        fr3_sync_audit.summarize_episode_capture_timestamps(
            capture_timestamps=_capture_timestamps(4, skew_s=0.001, interval_s=0.1),
            frame_timestamps=np.arange(4, dtype=np.float64) * 0.1,
            device_names=DEVICE_NAMES[:2],
            clock_semantics="hardware_mixed",
        )


def _write_sync_dataset(dataset_root: Path, *, device_names: list[str], robot_type: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 10,
        "robot_type": robot_type,
        "features": {
            "observation.device_capture_timestamp": {
                "dtype": "float64",
                "shape": [len(device_names)],
                "names": device_names,
            }
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    frames = 10
    captures = _capture_timestamps(frames, skew_s=0.002, interval_s=0.1)
    table = pa.table(
        {
            "episode_index": [0] * frames,
            "frame_index": list(range(frames)),
            "timestamp": [index / 10.0 for index in range(frames)],
            "observation.device_capture_timestamp": [row.tolist() for row in captures],
        }
    )
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "file-000.parquet")


def test_file_report_labels_sim_clock_semantics(tmp_path):
    dataset_root = tmp_path / "sim_ds"
    _write_sync_dataset(dataset_root, device_names=DEVICE_NAMES, robot_type="franka_research3_mujoco")
    report, destination = fr3_sync_audit.write_fr3_sync_report(dataset_root)
    assert destination == dataset_root / "meta" / "fr3_sync_report.json"
    assert destination.is_file()
    assert report["clock_semantics"] == "sim_extraction_wallclock"
    assert "not comparable to hardware sensor timestamps" in report["interpretation"]
    assert report["status"] == "pass"


def test_file_report_labels_hardware_clock_semantics(tmp_path):
    dataset_root = tmp_path / "real_ds"
    _write_sync_dataset(
        dataset_root,
        device_names=[
            "fr3.arm.capture_timestamp_s",
            "pika_gripper.capture_timestamp_s",
            "camera.ee.capture_timestamp_s",
        ],
        robot_type="franka_research3",
    )
    report, _ = fr3_sync_audit.write_fr3_sync_report(dataset_root)
    assert report["clock_semantics"] == "hardware_mixed"
    # Camera columns must never be described as exposure midpoints -- nor as the driver
    # handover, which is how they were documented until the stamp was traced to the camera's
    # own post-processing step.
    assert "neither exposure midpoint nor driver handover" in report["interpretation"]
    assert report["device_groups"]["camera"] == ["camera.ee.capture_timestamp_s"]
    assert report["device_groups"]["gripper"] == ["pika_gripper.capture_timestamp_s"]


# ---------------------------------------------------------------- gateway protocol wiring ---


def _workstation_state(tmp_path: Path) -> gateway.GatewayState:
    return gateway.GatewayState(
        repo_root=tmp_path,
        config_path=tmp_path / "fr3.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(),
        profile="workstation",
    )


def test_workstation_profile_selects_the_fr3_recorder(tmp_path):
    state = _workstation_state(tmp_path)
    script, flag = gateway._recorder_script(state)
    assert script == tmp_path / gateway.WORKSTATION_RECORDER_SCRIPT
    # draccus uses the underscore form; the hyphenated Thor convention would not parse.
    assert flag == "--config_path"


def test_thor_profile_keeps_the_handheld_recorder(tmp_path):
    state = _workstation_state(tmp_path)
    state.profile = "thor"
    script, _ = gateway._recorder_script(state)
    assert script == tmp_path / gateway.DEFAULT_RECORDER_SCRIPT


def _finished_mujoco_validation(tmp_path: Path, monkeypatch, *, max_pos_mm: float) -> gateway.GatewayState:
    # The trajectory contract reads the dataset off disk; this test is about what the verdict
    # does to `safety`, so hold the contract at "passed" and let the metrics decide.
    monkeypatch.setattr(
        gateway,
        "_trajectory_contract_for_episode",
        lambda state, dataset_root: {"status": "passed", "checks": [], "failures": []},
    )
    state = _workstation_state(tmp_path)
    state.replay.safety = "locked"
    state.replay.mujocoValidation = {
        "datasetRoot": str(tmp_path / "ds"),
        "episode": 0,
        "fps": 30,
        "hasStructuredResult": True,
        "completedFrames": 10,
        "totalFrames": 10,
        "maxPositionErrorMm": max_pos_mm,
        "maxRotationErrorDeg": 1.0,
        "maxPositionThresholdMm": 20.0,
        "maxRotationThresholdDeg": 15.0,
    }
    gateway._finish_mujoco_validation(state, 0)
    return state


def test_a_failed_mujoco_score_is_not_reported_as_a_hardware_fault(tmp_path, monkeypatch):
    """A trajectory scoring 25 mm is a verdict about the data, not about the rig.

    Writing safety="fault" here mislabelled it and withheld the robot-free controls that are
    exactly what an operator reaches for after a failed score.
    """
    state = _finished_mujoco_validation(tmp_path, monkeypatch, max_pos_mm=25.0)

    assert state.replay.mujocoValidation["status"] == "failed"
    assert state.replay.safety == "locked"


def test_a_passing_mujoco_score_does_not_authorize_the_hardware_path(tmp_path, monkeypatch):
    """Only the real preflight can say the arm is ready; a sim score cannot grant that."""
    state = _finished_mujoco_validation(tmp_path, monkeypatch, max_pos_mm=1.0)

    assert state.replay.mujocoValidation["status"] == "passed"
    assert state.replay.safety == "locked"


def test_real_replay_is_refused_up_front_on_a_profile_that_cannot_do_it(tmp_path):
    """Fail with the reason, not 60 lines later with a missing-file message."""
    state = _workstation_state(tmp_path)
    state.replay.realReplaySupported = False

    with pytest.raises(RuntimeError, match="not wired for the workstation profile"):
        gateway._start_real_replay(state, cube_mode="left", robot_ip="192.168.1.206")


def test_sync_lines_become_a_verdict_the_operator_can_act_on(tmp_path):
    state = _workstation_state(tmp_path)
    gateway._apply_recorder_output(
        state,
        "SYNC episode=0 status=fail clock=hardware_mixed frames=300 skew_p95_ms=31.20 "
        "skew_max_ms=42.10 grid_lag_p95_ms=8.10 interval_ms=33.4/33.3nominal "
        "bad_skew_frames=12 bad_lag_frames=0",
    )
    gateway._apply_recorder_output(state, "SYNC WARN: 12/300 frame(s) exceed the 20.0 ms budget")
    gateway._apply_recorder_output(state, "SYNC report=/tmp/ds/meta/fr3_sync_report.json")

    assert state.recording.syncStatus == "fail"
    assert "skew_max_ms=42.10" in state.recording.syncSummary
    assert state.recording.syncWarnings == ["12/300 frame(s) exceed the 20.0 ms budget"]
    assert state.recording.syncReportPath == "/tmp/ds/meta/fr3_sync_report.json"
    # SYNC lines are diagnostics, not recorder failures: the session must stay alive.
    assert state.recording.state != "error"


def test_a_passing_episode_clears_the_previous_episode_warnings(tmp_path):
    state = _workstation_state(tmp_path)
    gateway._apply_recorder_output(state, "SYNC WARN: 3/100 frame(s) exceed the 20.0 ms budget")
    gateway._apply_recorder_output(
        state, "SYNC episode=1 status=pass clock=hardware_mixed frames=100 skew_p95_ms=1.00"
    )
    assert state.recording.syncStatus == "pass"
    assert state.recording.syncWarnings == []


def test_unavailable_audit_is_distinguished_from_a_failing_one(tmp_path):
    state = _workstation_state(tmp_path)
    gateway._apply_recorder_output(state, "SYNC audit unavailable: parquet not readable")
    assert state.recording.syncStatus == "unavailable"


def test_sync_lines_do_not_pollute_the_recorder_output_ring(tmp_path):
    """The record log is for recorder progress; sync detail has its own panel."""
    state = _workstation_state(tmp_path)
    gateway._apply_recorder_output(state, "Episode 0 ready")
    gateway._apply_recorder_output(state, "SYNC episode=0 status=pass clock=hardware_mixed frames=10")
    assert state.recording.recentOutput == ["Episode 0 ready"]


def test_connect_rejects_an_unknown_backend(tmp_path):
    state = _workstation_state(tmp_path)
    with pytest.raises(ValueError, match="Recording backend must be one of"):
        gateway._connect_recorder(state, backend="hologram")


def test_thor_profile_cannot_request_a_sim_backend(tmp_path):
    state = _workstation_state(tmp_path)
    state.profile = "thor"
    with pytest.raises(ValueError, match="workstation profile"):
        gateway._connect_recorder(state, backend="sim")


# ------------------------------------------------------ delta replay reconstruction ---


def _delta_episode(reference: str, frames: int = 12):
    """Synthesise one episode's action/observation blocks for a delta contract."""
    from lerobot.robots.franka_research3.processor_franka_research3 import delta_ee_action_keys
    from lerobot.utils.rotation import Rotation

    action_names = list(delta_ee_action_keys(reference))
    observation_names = [
        "ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw",
        "prev_cmd.ee.x", "prev_cmd.ee.y", "prev_cmd.ee.z",
        "prev_cmd.ee.qx", "prev_cmd.ee.qy", "prev_cmd.ee.qz", "prev_cmd.ee.qw",
        "gripper.pos",
    ]

    delta_positions = np.tile(np.array([0.001, 0.0, -0.0005]), (frames, 1))
    delta_rotvecs = np.tile(np.array([0.0, 0.002, 0.0]), (frames, 1))
    seed_position = np.array([0.50, 0.10, 0.35])
    seed_rotation = np.eye(3)

    commanded_positions = np.zeros((frames, 3))
    commanded_quaternions = np.zeros((frames, 4))
    position, rotation = seed_position.copy(), seed_rotation.copy()
    for index in range(frames):
        position = position + delta_positions[index]
        rotation = rotation @ Rotation.from_rotvec(delta_rotvecs[index]).as_matrix()
        commanded_positions[index] = position
        commanded_quaternions[index] = Rotation.from_matrix(rotation).as_quat()

    # prev_cmd(k) is the command issued on frame k-1; frame 0's is the seed.
    prev_cmd_positions = np.vstack([seed_position, commanded_positions[:-1]])
    prev_cmd_quaternions = np.vstack(
        [Rotation.from_matrix(seed_rotation).as_quat(), commanded_quaternions[:-1]]
    )
    if reference == "current":
        # Measured pose is the reference, so it must be what each delta was taken against.
        measured_positions = commanded_positions - delta_positions
        measured_quaternions = np.array(
            [
                Rotation.from_matrix(
                    Rotation.from_quat(commanded_quaternions[i]).as_matrix()
                    @ Rotation.from_rotvec(-delta_rotvecs[i]).as_matrix()
                ).as_quat()
                for i in range(frames)
            ]
        )
    else:
        # Arm lags its command by a constant residual; irrelevant to prev_cmd reconstruction.
        measured_positions = commanded_positions - 0.002
        measured_quaternions = commanded_quaternions.copy()

    actions = np.hstack([delta_positions, delta_rotvecs, np.full((frames, 1), 0.5)])
    observations = np.hstack(
        [
            measured_positions,
            measured_quaternions,
            prev_cmd_positions,
            prev_cmd_quaternions,
            np.full((frames, 1), 0.5),
        ]
    )
    return action_names, actions, observation_names, observations, commanded_positions


@pytest.mark.parametrize("reference", ["prev_cmd", "current"])
def test_replay_rebuilds_the_recorded_command_trajectory_from_deltas(reference):
    from tools.fr3.fr3_gui_replay_runtime import reconstruct_absolute_pose_stream

    action_names, actions, observation_names, observations, expected = _delta_episode(reference)
    positions, _quaternions, source = reconstruct_absolute_pose_stream(
        action_names=action_names,
        actions=actions,
        observation_names=observation_names,
        observations=observations,
    )
    assert reference in source
    # Exact: the replay must reproduce the command stream that was recorded, not an approximation.
    assert np.abs(positions - expected).max() < 1e-12


def test_replay_still_reads_an_absolute_ee_dataset_unchanged():
    from tools.fr3.fr3_gui_replay_runtime import reconstruct_absolute_pose_stream

    action_names = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
    actions = np.array([[0.5, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0, 0.5]])
    positions, quaternions, source = reconstruct_absolute_pose_stream(
        action_names=action_names,
        actions=actions,
        observation_names=action_names,
        observations=actions,
    )
    assert source == "absolute_ee action column"
    assert np.allclose(positions[0], [0.5, 0.1, 0.35])
    assert np.allclose(quaternions[0], [0.0, 0.0, 0.0, 1.0])


def test_replay_rejects_an_unknown_action_contract():
    from tools.fr3.fr3_gui_replay_runtime import reconstruct_absolute_pose_stream

    with pytest.raises(ValueError, match="neither absolute EE"):
        reconstruct_absolute_pose_stream(
            action_names=["joint_1.pos"],
            actions=np.zeros((2, 1)),
            observation_names=["joint_1.pos"],
            observations=np.zeros((2, 1)),
        )


def _write_dataset_meta(root: Path, *, with_tasks: bool) -> Path:
    """A dataset root as the recorder finds it on disk: info.json, tasks optional."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    if with_tasks:
        (root / "meta" / "tasks.parquet").write_bytes(b"")
    return root


def test_resume_gate_accepts_a_dataset_that_has_task_metadata(tmp_path):
    from tools.fr3.fr3_gui_record_runtime import _assert_resumable_or_absent

    root = _write_dataset_meta(tmp_path / "ds", with_tasks=True)
    assert _assert_resumable_or_absent(str(root)) is True


def test_resume_gate_treats_a_missing_dataset_as_fresh(tmp_path):
    from tools.fr3.fr3_gui_record_runtime import _assert_resumable_or_absent

    assert _assert_resumable_or_absent(str(tmp_path / "not_created_yet")) is False


def test_resume_gate_refuses_an_info_only_dataset_instead_of_falling_back_to_the_hub(tmp_path):
    """The shell a create-then-discard-everything session leaves behind.

    Treating it as resumable makes LeRobotDatasetMetadata miss its tasks file and pull from the
    Hub, which blocks in TCP connect with no timeout when the Hub is unreachable -- the recorder
    hangs before its first output line and before it reads its own stdin, so the GUI shows only
    the gateway's spawn message and Exit cannot reach it.
    """
    from tools.fr3.fr3_gui_record_runtime import _assert_resumable_or_absent

    root = _write_dataset_meta(tmp_path / "ds", with_tasks=False)
    with pytest.raises(RuntimeError, match="no task metadata"):
        _assert_resumable_or_absent(str(root))


def _session_config(root: str, *, resume: bool = False):
    from types import SimpleNamespace

    return SimpleNamespace(dataset=SimpleNamespace(root=root), resume=resume)


def test_each_session_gets_its_own_stamped_dataset_root():
    """The config names a series of recordings; a session is one dataset inside it."""
    from tools.fr3.fr3_gui_record_runtime import _resolve_workspace_path, _session_dataset_root

    root = _session_dataset_root(_session_config("/lerobot/outputs/datasets/fr3_spacemouse"), "real")

    name = Path(root).name
    assert re.fullmatch(r"fr3_spacemouse_\d{8}_\d{6}", name)
    # The container-style config path still lands in this checkout, stamp and all.
    assert Path(root).parent == Path(_resolve_workspace_path("/lerobot/outputs/datasets"))

    # And the stamp is what makes two sessions two datasets rather than one pile.
    assert _session_dataset_root(_session_config("/lerobot/outputs/datasets/fr3_spacemouse"), "real")


def test_the_stamp_survives_the_sim_suffix_where_the_counter_can_still_strip_it():
    """`<name>_sim_<stamp>`, not `<name>_<stamp>_sim` -- the gateway strips a *trailing* stamp."""
    from tools.fr3.fr3_gui_record_runtime import _session_dataset_root

    root = _session_dataset_root(_session_config("/lerobot/outputs/datasets/fr3_spacemouse"), "sim")

    name = Path(root).name
    assert re.fullmatch(r"fr3_spacemouse_sim_\d{8}_\d{6}", name)
    assert gateway._dataset_name_prefixes(name) >= {"fr3_spacemouse_sim"}


def test_resume_keeps_the_configured_root_because_it_names_one_dataset():
    from tools.fr3.fr3_gui_record_runtime import _resolve_workspace_path, _session_dataset_root

    config = _session_config("/lerobot/outputs/datasets/fr3_spacemouse", resume=True)

    assert _session_dataset_root(config, "real") == _resolve_workspace_path(
        "/lerobot/outputs/datasets/fr3_spacemouse"
    )


def test_an_already_stamped_root_is_extended_rather_than_stamped_twice():
    """Pointing the recorder at one session must not nest a second stamp inside the first."""
    from tools.fr3.fr3_gui_record_runtime import _resolve_workspace_path, _session_dataset_root

    stamped = "/lerobot/outputs/datasets/fr3_spacemouse_20260731_101500"

    assert _session_dataset_root(_session_config(stamped), "real") == _resolve_workspace_path(stamped)


def test_workstation_recorder_runs_offline_so_a_hub_fallback_cannot_hang_it():
    """The FR3 recorder writes only local datasets; Thor's honours dataset.push_to_hub."""
    source = Path(gateway.__file__).read_text()
    marker = 'env["HF_HUB_OFFLINE"] = "1"'
    assert marker in source
    workstation_branch = source.split("if is_workstation:")[-1].split("recorder_log_path =")[0]
    assert marker in workstation_branch


def test_measured_interval_is_the_average_cadence_not_the_median_gap():
    """Jitter is asymmetric: a late frame is followed by an early one.

    Taking the median of per-frame gaps therefore reads above the true average and condemns a
    cadence the episode actually held. Measured on the hardware rig: median gap 35.4 ms against
    a mean of 33.34 ms, for a 30 fps episode whose total duration was correct to 0.03%.
    """
    frames = 300
    nominal_s = 1.0 / 30.0
    # Gaps alternate long/short around the nominal cadence, so total elapsed time is exact
    # while the median gap sits clearly above nominal.
    gaps = np.where(np.arange(frames - 1) % 3 == 2, nominal_s * 0.86, nominal_s * 1.07)
    centres = np.concatenate([[0.0], np.cumsum(gaps)])
    capture = np.repeat(centres[:, None], len(DEVICE_NAMES), axis=1)

    summary = fr3_sync_audit.summarize_episode_capture_timestamps(
        capture_timestamps=capture,
        frame_timestamps=np.arange(frames, dtype=np.float64) * nominal_s,
        device_names=DEVICE_NAMES,
        clock_semantics="hardware_mixed",
    )

    elapsed_mean_ms = (centres[-1] - centres[0]) / (frames - 1) * 1e3
    median_gap_ms = float(np.median(gaps)) * 1e3
    # The fixture is only meaningful if the two statistics actually disagree.
    assert median_gap_ms > elapsed_mean_ms + 1.0
    assert summary["measured_frame_interval_ms"] == pytest.approx(elapsed_mean_ms, abs=1e-6)
    assert summary["measured_frame_interval_ms"] < summary["nominal_frame_interval_ms"] * 1.05


def test_live_and_finalized_paths_agree_about_one_episode(tmp_path):
    """The two audit paths measured the same episode and reported different numbers.

    A v3 dataset holds its parquet open until finalize(), so a just-saved episode must be
    audited from the in-memory buffer while the persisted report is built from the files. Those
    are two code paths by necessity -- but they are not allowed to be two *measurements*. Before
    they shared an implementation, one reported the p95 of |grid lag| and the other the p95 of
    the signed value, describing a single real episode as both 13.49 and -3.75.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    device_names = [
        "fr3.arm.capture_timestamp_s",
        "pika_gripper.capture_timestamp_s",
        "camera.ee.capture_timestamp_s",
        "camera.side.capture_timestamp_s",
    ]
    frames = 60
    rng = np.random.default_rng(7)
    grid = np.arange(frames, dtype=np.float64) / 10.0
    arm = grid + rng.normal(0, 0.002, frames)
    # Cameras genuinely ahead of the arm, the condition that split the two readings apart.
    captures = np.stack(
        [arm, arm + 0.00004, arm - 0.025 + rng.normal(0, 0.002, frames), arm - 0.024], axis=1
    )

    dataset_root = tmp_path / "ds"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "robot_type": "franka_research3",
                "features": {
                    "observation.device_capture_timestamp": {
                        "dtype": "float64",
                        "shape": [len(device_names)],
                        "names": device_names,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * frames,
                "frame_index": list(range(frames)),
                "timestamp": grid.tolist(),
                "observation.device_capture_timestamp": [row.tolist() for row in captures],
            }
        ),
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
    )

    live = fr3_sync_audit.summarize_episode_capture_timestamps(
        capture_timestamps=captures,
        frame_timestamps=grid,
        device_names=device_names,
        clock_semantics="hardware_mixed",
    )
    report = fr3_sync_audit.build_fr3_sync_report(dataset_root=dataset_root)

    finalized_lag_p95_ms = float(report["summary"]["abs_global_lag_s"]["p95"]) * 1e3
    assert live["grid_lag_p95_ms"] == pytest.approx(finalized_lag_p95_ms, abs=1e-9)
    finalized_skew_p95_ms = float(report["summary"]["max_skew_s"]["p95"]) * 1e3
    assert live["p95_skew_ms"] == pytest.approx(finalized_skew_p95_ms, abs=1e-9)
    # And the shared line each path prints quotes the same figure.
    assert f"grid_lag_p95_ms={finalized_lag_p95_ms:.2f}" in fr3_sync_audit.format_sync_summary_line(report)
    assert f"grid_lag_p95_ms={live['grid_lag_p95_ms']:.2f}" in fr3_sync_audit.format_episode_sync_line(
        live, episode=0
    )


def test_grid_lag_is_anchored_to_the_arm_not_the_device_median(tmp_path):
    """Honest camera latency must not be charged to the control loop's cadence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    device_names = [
        "fr3.arm.capture_timestamp_s",
        "pika_gripper.capture_timestamp_s",
        "camera.ee.capture_timestamp_s",
        "camera.side.capture_timestamp_s",
    ]
    frames = 40
    grid = np.arange(frames, dtype=np.float64) / 10.0
    # Arm exactly on the grid; both cameras a real 25 ms ahead.
    captures = np.stack([grid, grid + 0.00004, grid - 0.025, grid - 0.0248], axis=1)

    dataset_root = tmp_path / "ds"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "robot_type": "franka_research3",
                "features": {
                    "observation.device_capture_timestamp": {
                        "dtype": "float64",
                        "shape": [len(device_names)],
                        "names": device_names,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * frames,
                "frame_index": list(range(frames)),
                "timestamp": grid.tolist(),
                "observation.device_capture_timestamp": [row.tolist() for row in captures],
            }
        ),
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
    )

    report = fr3_sync_audit.build_fr3_sync_report(dataset_root=dataset_root)

    # The loop held the grid exactly, so grid lag is ~0 despite the cameras' offset...
    assert abs(float(report["summary"]["abs_global_lag_s"]["p95"])) < 1e-6
    assert report["summary"]["global_lag_over_tolerance_frames"] == 0
    # ...while the camera offset is still reported, as a per-device bias.
    bias = report["cross_modality_bias_ms"]
    assert bias["camera.ee.capture_timestamp_s"] == pytest.approx(-25.0, abs=0.1)
