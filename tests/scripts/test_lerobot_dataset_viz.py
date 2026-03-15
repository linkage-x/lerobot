#!/usr/bin/env python

from datetime import datetime, timedelta, timezone
import multiprocessing as mp
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.scripts.lerobot_dataset_viz import (
    EE_RULER_AXIS_COLORS,
    build_episode_process_visualize_kwargs,
    build_episode_switch_visualize_kwargs,
    build_ee_axis_ruler_strips,
    create_episode_process,
    extract_ee_pose,
    get_ee_pose_state_indices,
    get_next_episode_index,
    has_ee_pose,
    is_tcp_port_available,
    make_system_time_anchor,
    run_episode_switch_loop,
    should_enable_episode_switch,
    to_system_timestamp,
    wait_for_tcp_port_available,
)
from lerobot.utils.rotation import Rotation


def test_has_ee_pose_requires_all_pose_fields():
    assert has_ee_pose(
        {
            "ee.x": torch.tensor([0.1]),
            "ee.y": torch.tensor([0.2]),
            "ee.z": torch.tensor([0.3]),
            "ee.qx": torch.tensor([0.0]),
            "ee.qy": torch.tensor([0.0]),
            "ee.qz": torch.tensor([0.0]),
            "ee.qw": torch.tensor([1.0]),
        }
    )
    assert not has_ee_pose(
        {
            "ee.x": torch.tensor([0.1]),
            "ee.y": torch.tensor([0.2]),
            "ee.z": torch.tensor([0.3]),
            "ee.qx": torch.tensor([0.0]),
            "ee.qy": torch.tensor([0.0]),
        }
    )


def test_extract_ee_pose_returns_position_and_rotation_matrix():
    quaternion = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
    batch = {
        "ee.x": torch.tensor([0.4], dtype=torch.float32),
        "ee.y": torch.tensor([-0.1], dtype=torch.float32),
        "ee.z": torch.tensor([0.25], dtype=torch.float32),
        "ee.qx": torch.tensor([quaternion[0]], dtype=torch.float32),
        "ee.qy": torch.tensor([quaternion[1]], dtype=torch.float32),
        "ee.qz": torch.tensor([quaternion[2]], dtype=torch.float32),
        "ee.qw": torch.tensor([quaternion[3]], dtype=torch.float32),
    }

    position, rotation = extract_ee_pose(batch, 0)

    assert np.allclose(position, np.array([0.4, -0.1, 0.25], dtype=np.float32))
    expected_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rotation, expected_rotation, atol=1e-6)


def test_get_ee_pose_state_indices_from_packed_observation_state():
    indices = get_ee_pose_state_indices(
        ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
    )

    assert indices == {
        "ee.x": 0,
        "ee.y": 1,
        "ee.z": 2,
        "ee.qx": 3,
        "ee.qy": 4,
        "ee.qz": 5,
        "ee.qw": 6,
    }


def test_extract_ee_pose_supports_packed_observation_state():
    ee_pose_state_indices = get_ee_pose_state_indices(
        ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
    )
    quaternion = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
    batch = {
        "observation.state": torch.tensor(
            [[0.4, -0.1, 0.25, quaternion[0], quaternion[1], quaternion[2], quaternion[3], 0.5]],
            dtype=torch.float32,
        )
    }

    assert has_ee_pose(batch, ee_pose_state_indices=ee_pose_state_indices)
    position, rotation = extract_ee_pose(batch, 0, ee_pose_state_indices=ee_pose_state_indices)

    assert np.allclose(position, np.array([0.4, -0.1, 0.25], dtype=np.float32))
    expected_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rotation, expected_rotation, atol=1e-6)


def test_build_ee_axis_ruler_strips_for_x_axis_creates_main_line_and_ticks():
    strips = build_ee_axis_ruler_strips("x", 0.1)

    assert len(strips) == 12
    assert np.allclose(
        strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(strips[1], np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32))
    assert np.allclose(strips[6], np.array([[0.05, 0.0, 0.0], [0.05, 0.01, 0.0]], dtype=np.float32))
    assert np.allclose(strips[2], np.array([[0.01, 0.0, 0.0], [0.01, 0.005, 0.0]], dtype=np.float32))


def test_build_ee_axis_ruler_strips_for_y_and_z_axes_use_origin_and_perpendicular_ticks():
    y_strips = build_ee_axis_ruler_strips("y", 0.1)
    z_strips = build_ee_axis_ruler_strips("z", 0.1)

    assert np.allclose(
        y_strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(y_strips[2], np.array([[0.0, 0.01, 0.0], [0.0, 0.01, 0.005]], dtype=np.float32))

    assert np.allclose(
        z_strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(z_strips[2], np.array([[0.0, 0.0, 0.01], [0.005, 0.0, 0.01]], dtype=np.float32))


def test_ee_ruler_axis_colors_follow_rgb_convention():
    assert EE_RULER_AXIS_COLORS == {
        "x": [255, 0, 0, 255],
        "y": [0, 255, 0, 255],
        "z": [0, 0, 255, 255],
    }


def test_system_time_anchor_and_timestamp_conversion():
    now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

    anchor = make_system_time_anchor(1.5, now=now)

    assert anchor == now - timedelta(seconds=1.5)
    assert to_system_timestamp(anchor, 1.5) == now
    assert to_system_timestamp(anchor, 2.0) == now + timedelta(seconds=0.5)


def test_should_enable_episode_switch_only_for_distant_streaming():
    assert should_enable_episode_switch("distant", False)
    assert not should_enable_episode_switch("local", False)
    assert not should_enable_episode_switch("distant", True)


def test_build_episode_switch_visualize_kwargs_reuses_cleaned_cli_kwargs():
    cli_kwargs = {
        "episode_index": 0,
        "batch_size": 32,
        "mode": "distant",
        "grpc_port": 9876,
        "ws_port": None,
    }

    visualize_kwargs = build_episode_switch_visualize_kwargs(cli_kwargs)

    assert "episode_index" not in visualize_kwargs
    assert visualize_kwargs["batch_size"] == 32
    assert visualize_kwargs["mode"] == "distant"
    assert visualize_kwargs["grpc_port"] == 9876


def test_build_episode_process_visualize_kwargs_adds_unique_recording_id():
    cli_kwargs = {
        "batch_size": 32,
        "mode": "distant",
        "grpc_port": 9876,
        "web_port": 9090,
    }

    visualize_kwargs = build_episode_process_visualize_kwargs(
        cli_kwargs,
        rerun_recording_id="recording-123",
    )

    assert visualize_kwargs["batch_size"] == 32
    assert visualize_kwargs["grpc_port"] == 9876
    assert visualize_kwargs["rerun_recording_id"] == "recording-123"


def test_get_next_episode_index_stops_at_last_episode():
    assert get_next_episode_index(0, 3) == 1
    assert get_next_episode_index(1, 3) == 2
    assert get_next_episode_index(2, 3) is None


def test_create_episode_process_is_not_daemonic():
    ctx = mp.get_context("spawn")

    process = create_episode_process(
        ctx,
        repo_id="local/test",
        root=None,
        episode_index=0,
        tolerance_s=1e-4,
        visualize_kwargs={"mode": "distant", "batch_size": 1, "num_workers": 2},
    )

    assert process.daemon is False


def test_run_episode_switch_loop_restarts_process_for_next_episode():
    launched = []
    terminated = []
    commands = iter(["n", "n", "n", "q"])

    def launch_episode(episode_index: int):
        process = f"proc-{episode_index}"
        launched.append((episode_index, process))
        return process

    def terminate_episode(process):
        terminated.append(process)

    def read_command():
        return next(commands)

    run_episode_switch_loop(
        start_episode_index=0,
        total_episodes=3,
        launch_episode=launch_episode,
        terminate_episode=terminate_episode,
        read_command=read_command,
    )

    assert launched == [
        (0, "proc-0"),
        (1, "proc-1"),
        (2, "proc-2"),
    ]
    assert terminated == [
        "proc-0",
        "proc-1",
        "proc-2",
    ]


def test_wait_for_tcp_port_available_returns_once_port_is_free(monkeypatch):
    availability = iter([False, False, True])
    sleeps = []

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_dataset_viz.is_tcp_port_available",
        lambda host, port: next(availability),
    )
    monkeypatch.setattr("lerobot.scripts.lerobot_dataset_viz.time.sleep", lambda seconds: sleeps.append(seconds))

    wait_for_tcp_port_available("0.0.0.0", 9876, timeout_s=1.0, poll_interval_s=0.1, label="gRPC 9876")

    assert sleeps == [0.1, 0.1]


def test_wait_for_tcp_port_available_raises_on_timeout(monkeypatch):
    perf_counter_values = iter([0.0, 0.1, 0.2, 0.3])
    monkeypatch.setattr("lerobot.scripts.lerobot_dataset_viz.is_tcp_port_available", lambda host, port: False)
    monkeypatch.setattr("lerobot.scripts.lerobot_dataset_viz.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("lerobot.scripts.lerobot_dataset_viz.time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="Port gRPC 9876 is still in use"):
        wait_for_tcp_port_available("0.0.0.0", 9876, timeout_s=0.25, poll_interval_s=0.1, label="gRPC 9876")


def test_is_tcp_port_available_returns_false_when_bind_fails(monkeypatch):
    fake_socket = MagicMock()
    fake_socket.__enter__.return_value = fake_socket
    fake_socket.bind.side_effect = OSError("in use")
    fake_socket_factory = MagicMock(return_value=fake_socket)

    monkeypatch.setattr("lerobot.scripts.lerobot_dataset_viz.socket.socket", fake_socket_factory)

    assert is_tcp_port_available("0.0.0.0", 9876) is False
