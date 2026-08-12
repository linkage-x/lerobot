#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.fr3 import fr3_train_il_policy


def test_selected_state_names_supports_feature_name_selector():
    features = {
        "observation.state.right": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"],
        },
        "observation.state_raw": {
            "dtype": "float32",
            "shape": [2],
            "names": [
                "handheld_gripper.pika_right.width_mm",
                "handheld_gripper.pika_left.width_mm",
            ],
        },
    }

    names = fr3_train_il_policy.selected_state_names(
        features,
        [
            "observation.state.right",
            "observation.state_raw:handheld_gripper.pika_left.width_mm",
        ],
    )

    assert names == [
        "observation.state.right.ee.x",
        "observation.state.right.ee.y",
        "observation.state.right.ee.z",
        "observation.state.right.ee.qx",
        "observation.state.right.ee.qy",
        "observation.state.right.ee.qz",
        "observation.state.right.ee.qw",
        "observation.state_raw.handheld_gripper.pika_left.width_mm",
    ]


def test_select_state_matrix_concatenates_full_features_and_selected_dims():
    features = {
        "observation.state.right": {
            "dtype": "float32",
            "shape": [7],
            "names": ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"],
        },
        "observation.state_raw": {
            "dtype": "float32",
            "shape": [2],
            "names": [
                "handheld_gripper.pika_right.width_mm",
                "handheld_gripper.pika_left.width_mm",
            ],
        },
    }
    df = pd.DataFrame(
        {
            "observation.state.right": [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [1.1, 1.2, 1.3, 0.1, 0.2, 0.3, 0.9],
            ],
            "observation.state_raw": [
                [60.0, 61.0],
                [70.0, 71.0],
            ],
        }
    )

    state = fr3_train_il_policy.select_state_matrix(
        df,
        features,
        [
            "observation.state.right",
            "observation.state_raw:handheld_gripper.pika_left.width_mm",
        ],
    )

    assert state.dtype == np.float32
    assert state.shape == (2, 8)
    assert np.allclose(state[:, :7], np.asarray(df["observation.state.right"].tolist(), dtype=np.float32))
    assert np.allclose(state[:, 7], [61.0, 71.0])


# --------------------------------------------------------- episode exclusion in the view ---


def _write_v3_source_dataset(
    root: Path, *, episodes: int = 3, frames: int = 4, camera: str = "observation.images.ee"
) -> None:
    """A minimal LeRobot v3 recording: one parquet, one mp4 per camera, episode metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / camera / "chunk-000").mkdir(parents=True)
    (root / "videos" / camera / "chunk-000" / "file-000.mp4").write_bytes(b"\0" * 16)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_research3",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": episodes,
        "total_frames": episodes * frames,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["ee.x", "gripper.pos"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["ee.x", "gripper.pos"]},
            camera: {"dtype": "video", "shape": [64, 64, 3], "names": ["height", "width", "channel"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["Pick and place"], name="task"))
    tasks.to_parquet(root / "meta" / "tasks.parquet")

    rows = {key: [] for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]}
    rows["observation.state"] = []
    rows["action"] = []
    for episode in range(episodes):
        for frame in range(frames):
            rows["timestamp"].append(frame / 30.0)
            rows["frame_index"].append(frame)
            rows["episode_index"].append(episode)
            rows["index"].append(episode * frames + frame)
            rows["task_index"].append(0)
            # Marks every row with the episode it came from, so a filtered view is checkable.
            rows["observation.state"].append([float(episode * 10 + frame), 0.5])
            rows["action"].append([float(episode * 10 + frame), 0.5])
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")

    episode_rows = {
        "episode_index": list(range(episodes)),
        "tasks": [["Pick and place"]] * episodes,
        "length": [frames] * episodes,
        "data/chunk_index": [0] * episodes,
        "data/file_index": [0] * episodes,
        "dataset_from_index": [episode * frames for episode in range(episodes)],
        "dataset_to_index": [(episode + 1) * frames for episode in range(episodes)],
        f"videos/{camera}/chunk_index": [0] * episodes,
        f"videos/{camera}/file_index": [0] * episodes,
        f"videos/{camera}/from_timestamp": [episode * frames / 30.0 for episode in range(episodes)],
        f"videos/{camera}/to_timestamp": [(episode + 1) * frames / 30.0 for episode in range(episodes)],
    }
    pq.write_table(
        pa.table(episode_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )


def _write_annotations(root: Path, *, excluded: list[int]) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "gui_annotations.json").write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": {
                    str(episode): {"episode": episode, "includeInTraining": False, "quality": "bad"}
                    for episode in excluded
                },
            }
        ),
        encoding="utf-8",
    )


def _build_view(src_root: Path, dst_root: Path, **kwargs):
    import pyarrow.parquet as pq

    camera = "observation.images.ee"
    fr3_train_il_policy.prepare_dataset_view(
        src_root=src_root,
        dst_root=dst_root,
        repo_id="local/test_view",
        camera_keys=[camera],
        state_keys=["observation.state"],
        action_key="action",
        action_npy=None,
        action_append_selectors=[],
        action_append_names=[],
        action_append_shift=1,
        image_resize_shape=None,
        copy_videos=False,
        overwrite=True,
        **kwargs,
    )
    frames = pq.read_table(dst_root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    episodes = pq.read_table(dst_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    info = json.loads((dst_root / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest = json.loads((dst_root / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    return frames, episodes, info, manifest


def test_view_drops_episodes_marked_not_for_training(tmp_path):
    """The annotation is the exclusion; nothing else in the pipeline reads that flag."""
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)
    _write_annotations(src_root, excluded=[1])

    frames, episodes, info, manifest = _build_view(src_root, tmp_path / "view")

    # Source episode 1 carried states 10..13; none of them may survive.
    assert [row[0] for row in frames["observation.state"]] == [0, 1, 2, 3, 20, 21, 22, 23]
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8
    assert info["splits"] == {"train": "0:2"}
    assert manifest["excluded_episodes"][str(src_root.resolve())] == [1]


def test_view_renumbers_episodes_and_rows_after_an_exclusion(tmp_path):
    """A gap in episode_index or index would be a dataset that claims frames it does not have."""
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    frames, episodes, _, manifest = _build_view(
        src_root, tmp_path / "view", exclude_episodes={1}, respect_annotations=False
    )

    assert list(frames["episode_index"]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(frames["index"]) == list(range(8))
    assert list(episodes["episode_index"]) == [0, 1]
    assert list(episodes["dataset_from_index"]) == [0, 4]
    assert list(episodes["dataset_to_index"]) == [4, 8]
    # Videos are symlinked whole, so the surviving episode keeps its own range inside the mp4.
    assert list(episodes["videos/observation.images.ee/from_timestamp"]) == pytest.approx([0.0, 8 / 30.0])
    # Which recording each view episode came from, since the numbering no longer says so.
    assert manifest["episode_source_index"] == [
        {"episode_index": 0, "source_dataset_root": str(src_root.resolve()), "source_episode_index": 0},
        {"episode_index": 1, "source_dataset_root": str(src_root.resolve()), "source_episode_index": 2},
    ]


def test_view_can_be_built_against_the_operators_review(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)
    _write_annotations(src_root, excluded=[0, 2])

    _, _, info, _ = _build_view(src_root, tmp_path / "kept", respect_annotations=False)
    assert info["total_episodes"] == 3

    _, _, info, _ = _build_view(src_root, tmp_path / "reviewed")
    assert info["total_episodes"] == 1


def test_view_refuses_to_exclude_an_episode_that_is_not_there(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    with pytest.raises(ValueError, match="no episode"):
        _build_view(src_root, tmp_path / "view", exclude_episodes={7})


def test_view_refuses_to_build_from_nothing(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)
    _write_annotations(src_root, excluded=[0, 1, 2])

    with pytest.raises(ValueError, match="nothing to build"):
        _build_view(src_root, tmp_path / "view")
