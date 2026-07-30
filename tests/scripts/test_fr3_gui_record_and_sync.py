"""Contract tests for the gateway-driven FR3 recorder and its timestamp-sync audit.

The properties under test are the ones that silently rot: sim and hardware recordings must
keep producing the same dataset schema, the audit must not mislabel which clock produced a
timestamp, and the gateway must parse the recorder's SYNC protocol into a verdict the operator
can act on.
"""

from __future__ import annotations

import json
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
    # Camera columns must never be described as exposure midpoints.
    assert "not exposure midpoint" in report["interpretation"]
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
